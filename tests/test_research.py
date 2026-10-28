"""Research invariants: split isolation, full scores, and checkpoint round trips."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'prog'))
from data_process import process
from research import build_model, prepare_output_dir, run, response_metrics


def fixture():
    rng = np.random.RandomState(7)
    # Deliberately unsorted source rows and drug insertion order.
    cells = ['cell-c', 'cell-a', 'cell-d', 'cell-b']
    drugs = ['30', '2', '10', '1', '20', '4']
    drug_features = {
        drug: [rng.rand(3, 5).astype('float32'), [[1], [0, 2], [1]], [1, 2, 1]]
        for drug in drugs}
    mutation = pd.DataFrame(rng.rand(4, 34673), index=cells)
    expression = pd.DataFrame(rng.rand(4, 7), index=cells)
    methylation = pd.DataFrame(rng.rand(4, 9), index=cells)
    pairs = [(cell, drug, 1 if (i + j) % 2 else -1)
             for i, cell in enumerate(cells) for j, drug in enumerate(drugs)]
    pairs = pairs[:-2]  # Include two unobserved pairs.
    return drug_features, mutation, expression, methylation, pairs, 4, 6


def prepared(values):
    return process(*values, validation_fraction=0.2, test_fraction=0.2, split_seed=12)


def inputs_for(data):
    drug = next(iter(data.drug_loader))
    cell = next(iter(data.cell_loader))
    return (drug.x, drug.edge_index, drug.batch, *cell, data.train_edges)


def config_for(data):
    return {'hidden_channels': 6, 'output_channels': 8, 'atom_shape': 5,
            'nb_celllines': 4, 'nb_drugs': 6, 'expression_dim': 7,
            'methylation_dim': 9, 'mutation_dim': 34673}


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Avoid excessive thread overhead for small synthetic graphs.
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_split_isolation_and_identifier_order(self):
        values = fixture()
        data = prepared(values)
        again = prepared(values)
        self.assertEqual(data.cell_line_ids, sorted(values[1].index))
        self.assertEqual(data.drug_ids, sorted(values[0]))
        membership = sum(data.masks[name].int() for name in ('train', 'validation', 'test'))
        torch.testing.assert_close(membership, data.masks['observed'].int())
        for name in data.masks:
            torch.testing.assert_close(data.masks[name], again.masks[name])
        self.assertEqual(int(data.masks['observed'].sum()), len(values[4]))
        for cell, drug, response in values[4]:
            index = data.cell_line_ids.index(cell) * 6 + data.drug_ids.index(drug)
            self.assertEqual(int(data.labels[index]), int(response == 1))
        forward_edges = data.train_edges[data.train_edges[:, 0] < 4]
        self.assertEqual(len(forward_edges), int(data.masks['train'].sum()))
        for cell, drug, response in forward_edges:
            index = cell * 6 + drug - 4
            self.assertTrue(data.masks['train'][index])
            self.assertFalse(data.masks['validation'][index] or data.masks['test'][index])
            self.assertEqual(int(data.labels[index]), int(response == 1))

    def test_complete_embedding_and_model_round_trip(self):
        data = prepared(fixture())
        inputs = inputs_for(data)
        torch.manual_seed(4)
        model = build_model(config_for(data))
        model.train()
        output = model(*inputs, return_details=True)
        loss = output[-1]['logits'].square().mean()
        loss.backward()
        self.assertIsNotNone(model.feat.graph_conv[0].lin.weight.grad)
        model.eval()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.no_grad():
            *_, probabilities, details = model(*inputs, return_details=True)
            self.assertEqual(len(model(*inputs)), 5)
        self.assertEqual(details['cell_embeddings'].shape, (4, 16))
        self.assertFalse(model.feat.graph_bn[0].training)
        self.assertIn('feat.graph_conv.0.lin.weight', before)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        products = details['cell_embeddings'][:, None, :] * details['drug_embeddings'][None, :, :]
        torch.testing.assert_close(products.sum(-1), details['logits'])
        torch.testing.assert_close(details['logits'].sigmoid().reshape(-1), probabilities)
        restored = build_model(config_for(data))
        restored.load_state_dict(before, strict=True)
        restored.eval()
        with torch.no_grad():
            restored_details = restored(*inputs, return_details=True)[-1]
        for name in details:
            torch.testing.assert_close(details[name], restored_details[name], rtol=0, atol=0)
        # Distinct graphs must produce distinct graph embeddings.
        self.assertFalse(torch.allclose(output[0], output[1]))

    def test_export_uses_validation_checkpoint_and_reconstructs_predictions(self):
        values = fixture()
        validation_aucs = [0.7, 0.9, 0.8]  # Best epoch is not the last epoch.
        calls = []

        def measured(*args):
            result = response_metrics(*args)
            # Three epoch evaluations must precede final train/validation/test.
            if len(calls) < 3:
                result['auc'] = validation_aucs[len(calls)]
            calls.append(args[3].clone())
            return result

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / 'run'
            argv = ['--epoch', '3', '--seed', '9', '--split-seed', '12',
                    '--hidden_channels', '6', '--output_channels', '8',
                    '--validation-fraction', '0.2', '--test-fraction', '0.2',
                    '--data-dir', temporary, '--output-dir', str(output_dir)]
            with patch('research.load_dataset', return_value=(values, [])), \
                    patch('research.response_metrics', side_effect=measured), \
                    contextlib.redirect_stdout(io.StringIO()):
                run('gdsc', argv)
            checkpoint = torch.load(output_dir / 'best.pt', weights_only=True)
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertEqual(len(calls), 6)
            data = prepared(values)
            for mask in calls[:3]:
                torch.testing.assert_close(mask, data.masks['validation'])
            torch.testing.assert_close(calls[-1], data.masks['test'])
            restored = build_model(checkpoint['model_config'])
            restored.load_state_dict(checkpoint['model_state_dict'])
            restored.eval()
            with torch.no_grad():
                *_, probabilities, details = restored(*inputs_for(data), return_details=True)
            with np.load(output_dir / 'research_objects.npz', allow_pickle=False) as saved:
                self.assertEqual(int(saved['selected_epoch']), 2)
                np.testing.assert_allclose(saved['logits'], details['logits'].numpy(), rtol=0, atol=0)
                np.testing.assert_allclose(saved['probabilities'].ravel(), probabilities.numpy(), rtol=0, atol=0)
                self.assertTrue(np.isnan(saved['labels'][~saved['observed_mask']]).all())
                self.assertEqual(int((saved['split'] == 0).sum()), 2)
                self.assertEqual(saved['cell_line_ids'].tolist(), data.cell_line_ids)
                self.assertEqual(saved['drug_ids'].tolist(), data.drug_ids)
                np.testing.assert_allclose(saved['cell_embeddings'] @ saved['drug_embeddings'].T,
                                           saved['logits'], rtol=1e-5, atol=1e-5)
            predictions = pd.read_csv(output_dir / 'predictions.csv', dtype={'drug_id': str})
            self.assertEqual(len(predictions), 24)
            self.assertEqual(predictions['label'].isna().sum(), 2)
            self.assertEqual(predictions.iloc[0]['drug_id'], data.drug_ids[0])
            np.testing.assert_allclose(predictions['probability'], probabilities.numpy(), rtol=1e-6)
            history = pd.read_csv(output_dir / 'history.csv')
            self.assertFalse(any(name.startswith('test') for name in history))
            report = json.loads((output_dir / 'metrics.json').read_text())
            self.assertEqual(report['selected_epoch'], 2)
            self.assertEqual(report['test']['threshold'], checkpoint['decision_threshold'])
            self.assertTrue((output_dir / 'source' / 'model.py').is_file())
            # A repeated run must not silently overwrite its predecessor.
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                run('gdsc', argv)
            notes = output_dir / 'notes.txt'
            notes.write_text('Keep these research notes.\n')
            stale_source = output_dir / 'source' / 'obsolete.py'
            stale_source.write_text('# Source from an older run.\n')
            overwrite_argv = argv.copy() + ['--overwrite']
            overwrite_argv[overwrite_argv.index('--epoch') + 1] = '1'
            with patch('research.load_dataset', return_value=(values, [])), \
                    contextlib.redirect_stdout(io.StringIO()):
                run('gdsc', overwrite_argv)
            self.assertEqual(len(pd.read_csv(output_dir / 'history.csv')), 1)
            self.assertEqual(torch.load(output_dir / 'best.pt', weights_only=True)['epoch'], 1)
            self.assertEqual(json.loads((output_dir / 'metrics.json').read_text())['selected_epoch'], 1)
            self.assertFalse(stale_source.exists())
            self.assertEqual(notes.read_text(), 'Keep these research notes.\n')
            self.assertTrue((output_dir / 'research_objects.npz').is_file())

    def test_overwrite_rejects_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            notes = output_dir / 'notes.txt'
            notes.write_text('Unrelated project\n')
            with self.assertRaises(ValueError):
                prepare_output_dir(output_dir, overwrite=True)
            self.assertEqual(notes.read_text(), 'Unrelated project\n')


if __name__ == '__main__':
    unittest.main()
