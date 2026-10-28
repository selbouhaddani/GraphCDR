"""Shared training and export of GraphCDR research objects."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, f1_score,
                             precision_recall_curve, roc_auc_score)
import torch
import torch.nn.functional as F

from data_load import dataload
from data_process import process
from model import Encoder, GraphCDR, NodeRepresentation, Summary


SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
SPLIT_CODES = {'unobserved': 0, 'train': 1, 'validation': 2, 'test': 3}


def build_model(config):
    """Rebuild the architecture from checkpoint['model_config']."""
    output = config['output_channels']
    hidden = config['hidden_channels']
    return GraphCDR(
        hidden_channels=hidden, encoder=Encoder(output, hidden),
        summary=Summary(output, hidden), index=config['nb_celllines'],
        feat=NodeRepresentation(config['atom_shape'], config['expression_dim'],
                                config['methylation_dim'], output))


def parser_for(dataset):
    parser = argparse.ArgumentParser(description='Train GraphCDR and save research objects.')
    parser.add_argument('--alph', type=float, default=0.30)
    parser.add_argument('--beta', type=float, default=0.30)
    parser.add_argument('--epoch', type=int, default=350 if dataset == 'gdsc' else 200)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--output_channels', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0, help='Model/training random seed.')
    parser.add_argument('--split-seed', type=int, default=666,
                        help='Independent seed for response splits; keep fixed across models.')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path,
                        help='New run directory. Default: runs/<dataset>/<UTC timestamp>-seed<seed>.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace generated artifacts in an existing GraphCDR run directory.')
    return parser


def load_dataset(dataset, data_dir):
    cell_files = [data_dir / 'Celline' / name for name in [
        'genomic_mutation_34673_demap_features.csv',
        'genomic_expression_561celllines_697genes_demap_features.csv',
        'genomic_methylation_561celllines_808genes_demap_features.csv']]
    if dataset == 'gdsc':
        drug_dir = data_dir / 'Drug'
        files = [drug_dir / '1.Drug_listMon Jun 24 09_00_55 2019.csv',
                 drug_dir / 'drug_threshold.txt', drug_dir / 'drug_graph_feat',
                 data_dir / 'Celline' / 'Cell_lines_annotations.txt',
                 cell_files[0], data_dir / 'Celline' / 'GDSC_IC50.csv',
                 cell_files[1], cell_files[2]]
        values = dataload(*map(str, files))
        inputs = [path for path in files if path.is_file()]
        inputs.extend(sorted(files[2].glob('*.hkl')))
        return values, inputs

    # These dependencies are only needed when generating CCLE drug features.
    from rdkit import Chem
    import deepchem as dc

    mutation, expression, methylation = [
        pd.read_csv(path, index_col=0) for path in cell_files]
    mutation = mutation.loc[expression.index]
    drug_file = data_dir / 'CCLE' / 'CCLE_smiles.csv'
    response_file = data_dir / 'CCLE' / 'CCLE_response.csv'
    drugs = pd.read_csv(drug_file, dtype={'pubchem': str})
    featurizer = dc.feat.ConvMolFeaturizer()
    drug_features = {}
    for pubchem, smiles in zip(drugs['pubchem'], drugs['isosmiles']):
        features = featurizer.featurize([Chem.MolFromSmiles(smiles)])[0]
        drug_features[pubchem] = [features.get_atom_features(), features.get_adjacency_list(), 1]
    responses = pd.read_csv(response_file, dtype={'pubchem': str})
    # Preserve the original duplicate resolution: sensitivity wins a conflict.
    unique = {}
    for cell, drug, score in zip(responses['DepMap_ID'], responses['pubchem'], responses['Z_SCORE']):
        label = 1 if score > 0.8 else -1
        unique[cell, drug] = max(label, unique.get((cell, drug), -1))
    pairs = [(cell, drug, label) for (cell, drug), label in sorted(unique.items(), reverse=True)]
    n_cells = len({pair[0] for pair in pairs})
    n_drugs = len({pair[1] for pair in pairs})
    print('All %d pairs across %d cell lines and %d drugs.' % (len(pairs), n_cells, n_drugs))
    return ((drug_features, mutation, expression, methylation, pairs, n_cells, n_drugs),
            cell_files + [drug_file, response_file])


def validation_threshold(labels, probabilities):
    """Choose an F1 threshold on validation data, never on test labels."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    denom = precision[:-1] + recall[:-1]
    scores = np.divide(2 * precision[:-1] * recall[:-1], denom,
                       out=np.zeros_like(denom), where=denom != 0)
    return float(thresholds[np.argmax(scores)])


def response_metrics(labels, logits, probabilities, mask, threshold):
    y = labels[mask].cpu().numpy()
    p = probabilities[mask].cpu().numpy()
    if len(np.unique(y)) != 2:
        raise ValueError('Each response split must contain both labels to evaluate AUC.')
    precision, recall, _ = precision_recall_curve(y, p)
    return {
        'count': int(mask.sum()),
        'loss': float(F.binary_cross_entropy_with_logits(logits[mask], labels[mask])),
        'auc': float(roc_auc_score(y, p)), 'aupr': float(auc(recall, precision)),
        'f1': float(f1_score(y, p >= threshold, zero_division=0)),
        'accuracy': float(accuracy_score(y, p >= threshold)),
        'threshold': threshold,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare_output_dir(path, overwrite=False):
    """Replace only known generated outputs, preserving unrelated run notes."""
    try:
        path.mkdir(parents=True, exist_ok=False)
        return
    except FileExistsError:
        if not overwrite:
            raise FileExistsError(
                '%s already exists; use --overwrite to replace its previous GraphCDR outputs.' % path
            ) from None
    if not path.is_dir():
        raise ValueError('Output path is not a directory: %s' % path)
    if not any(path.iterdir()):
        return
    try:
        settings = json.loads((path / 'settings.json').read_text())
    except (OSError, ValueError):
        settings = None
    if not isinstance(settings, dict) or settings.get('model_variant') != 'graphcdr_baseline' \
            or settings.get('schema_version') != 1:
        raise ValueError('--overwrite requires an existing GraphCDR run (valid settings.json): %s' % path)
    artifacts = [path / name for name in (
        'best.pt', 'research_objects.npz', 'predictions.csv', 'history.csv', 'metrics.json', 'settings.json')]
    source = path / 'source'
    # Check all types before clearing anything. Never follow a source symlink.
    if any(item.is_dir() and not item.is_symlink() for item in artifacts) \
            or source.is_symlink() or (source.exists() and not source.is_dir()):
        raise ValueError('Unexpected artifact paths; refusing to overwrite: %s' % path)
    for item in artifacts:
        item.unlink(missing_ok=True)
    if source.exists():
        shutil.rmtree(source)


def fingerprints(paths, base):
    result = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        result[str(path.relative_to(base))] = digest.hexdigest()
    return result


def save_objects(output_dir, data, details, probabilities, selected_epoch, threshold):
    """All matrices use exactly the same cell-row/drug-column ordering."""
    shape = (len(data.cell_line_ids), len(data.drug_ids))
    masks = {name: mask.numpy().reshape(shape) for name, mask in data.masks.items()}
    labels = data.labels.numpy().reshape(shape).copy()
    labels[~masks['observed']] = np.nan  # Missing responses are not negatives.
    split = np.zeros(shape, dtype=np.uint8)
    for name in ('train', 'validation', 'test'):
        split[masks[name]] = SPLIT_CODES[name]
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in details.items()}
    p = probabilities.cpu().numpy().reshape(shape)
    np.testing.assert_allclose(arrays['cell_embeddings'] @ arrays['drug_embeddings'].T,
                               arrays['logits'], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(torch.sigmoid(details['logits']).cpu().numpy(), p,
                               rtol=1e-6, atol=1e-7)
    np.savez_compressed(
        output_dir / 'research_objects.npz', schema_version=np.array(1),
        cell_line_ids=np.asarray(data.cell_line_ids), drug_ids=np.asarray(data.drug_ids),
        **arrays, probabilities=p, labels=labels, split=split,
        **{name + '_mask': mask for name, mask in masks.items()},
        train_edges=data.train_edges, selected_epoch=np.array(selected_epoch),
        decision_threshold=np.array(threshold))
    names = np.array(['unobserved', 'train', 'validation', 'test'])
    pd.DataFrame({
        'cell_index': np.repeat(np.arange(shape[0]), shape[1]),
        'drug_index': np.tile(np.arange(shape[1]), shape[0]),
        'cell_line_id': np.repeat(data.cell_line_ids, shape[1]),
        'drug_id': np.tile(data.drug_ids, shape[0]),
        'label': labels.ravel(), 'split': names[split.ravel()],
        'logit': arrays['logits'].ravel(), 'probability': p.ravel(),
        'predicted_label': (p.ravel() >= threshold).astype(int),
    }).to_csv(output_dir / 'predictions.csv', index=False)


def run(dataset, argv=None):
    parser = parser_for(dataset)
    args = parser.parse_args(argv)
    if args.epoch < 1 or min(args.hidden_channels, args.output_channels) < 1:
        parser.error('Epoch and channel counts must be positive.')
    if args.alph < 0 or args.beta < 0 or args.alph + args.beta >= 1:
        parser.error('alph and beta must be nonnegative and sum to less than 1.')
    if not (0 < args.validation_fraction < 1 and 0 < args.test_fraction < 1
            and args.validation_fraction + args.test_fraction < 1):
        parser.error('Validation and test fractions must be positive and sum to less than 1.')
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.data_dir = args.data_dir.expanduser().resolve()
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        args.output_dir = ROOT / 'runs' / dataset / (stamp + '-seed' + str(args.seed))
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output_dir(args.output_dir, overwrite=args.overwrite)

    values, input_files = load_dataset(dataset, args.data_dir)
    data = process(*values, validation_fraction=args.validation_fraction,
                   test_fraction=args.test_fraction, split_seed=args.split_seed)
    model_config = {
        'hidden_channels': args.hidden_channels, 'output_channels': args.output_channels,
        'atom_shape': int(data.atom_shape), 'nb_celllines': len(data.cell_line_ids),
        'nb_drugs': len(data.drug_ids), 'expression_dim': values[2].shape[1],
        'methylation_dim': values[3].shape[1], 'mutation_dim': values[1].shape[1],
    }
    model = build_model(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    # Both loaders contain a single full graph/batch in the saved node order.
    drug = next(iter(data.drug_loader))
    cell = next(iter(data.cell_loader))
    inputs = (drug.x, drug.edge_index, drug.batch, *cell, data.train_edges)
    versions = {}
    for package in ['torch', 'torch-geometric', 'torch-scatter', 'torch-sparse',
                    'numpy', 'pandas', 'scipy', 'scikit-learn', 'hickle', 'deepchem', 'rdkit']:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    settings = {
        'schema_version': 1, 'model_variant': 'graphcdr_baseline', 'dataset': dataset,
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'model_config': model_config, 'optimizer': {'name': 'Adam', 'lr': 0.001, 'weight_decay': 0},
        'selection': 'highest validation AUC; first epoch wins ties; test evaluated after selection',
        'split_policy': 'stratified observed pairs; training edges only in response graph',
        'split_codes': SPLIT_CODES, 'seed': args.seed, 'split_seed': args.split_seed,
        'python': sys.version, 'platform': platform.platform(), 'device': 'cpu', 'packages': versions,
        'torch_num_threads': torch.get_num_threads(),
        'response_label_definition': (
            '1 if ln(IC50) < the supplied drug-specific threshold, else 0'
            if dataset == 'gdsc' else '1 if Z_SCORE > 0.8, else 0'),
        'input_sha256': fingerprints(input_files, args.data_dir),
        'feature_names': data.feature_names,
        'embedding_coordinates': {'response_graph': [0, args.hidden_channels],
                                  'feature_projection': [args.hidden_channels, args.hidden_channels + 10]},
        'implementation_notes': [
            'Molecular layers registered with ModuleList for complete checkpoints and evaluation mode.',
            'Graph normalization is recomputed for positive and negative response graphs.',
            'Classification threshold maximizes validation F1 and is fixed for test evaluation.',
        ],
    }
    # Preserve the exact implementation alongside settings and input hashes.
    source_dir = args.output_dir / 'source'
    source_files = sorted(SOURCE_DIR.rglob('*.py'))
    for source in source_files:
        destination = source_dir / source.relative_to(SOURCE_DIR)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    settings['source_sha256'] = fingerprints(sorted(source_dir.rglob('*.py')), source_dir)
    write_json(args.output_dir / 'settings.json', settings)
    print('Artifacts:', args.output_dir)
    print('Split sizes:', {name: int(data.masks[name].sum()) for name in ('train', 'validation', 'test')})

    best_auc = -float('inf')
    history = []
    for epoch in range(1, args.epoch + 1):
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary_pos, summary_neg, _, details = model(*inputs, return_details=True)
        response_loss = F.binary_cross_entropy_with_logits(
            details['logits'].reshape(-1)[data.masks['train']], data.labels[data.masks['train']])
        loss = ((1 - args.alph - args.beta) * response_loss
                + args.alph * model.loss(pos_z, neg_z, summary_pos)
                + args.beta * model.loss(neg_z, pos_z, summary_neg))
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss; no valid checkpoint can be selected.')
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            *_, probabilities, details = model(*inputs, return_details=True)
            mask = data.masks['validation']
            threshold = validation_threshold(data.labels[mask].numpy(), probabilities[mask].numpy())
            validation = response_metrics(data.labels, details['logits'].reshape(-1), probabilities,
                                          mask, threshold)
        row = {'epoch': epoch, 'train_loss': float(loss.detach()),
               **{'validation_' + key: value for key, value in validation.items()}}
        history.append(row)
        if validation['auc'] > best_auc:
            best_auc = validation['auc']
            torch.save({
                'schema_version': 1, 'model_variant': 'graphcdr_baseline', 'epoch': epoch,
                'model_config': model_config, 'settings': settings,
                'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'torch_rng_state': torch.get_rng_state(), 'validation_metrics': validation,
                'decision_threshold': threshold, 'cell_line_ids': data.cell_line_ids,
                'drug_ids': data.drug_ids, 'train_edges': torch.from_numpy(data.train_edges),
                'labels': data.labels, 'masks': data.masks,
            }, args.output_dir / 'best.pt')
        pd.DataFrame(history).to_csv(args.output_dir / 'history.csv', index=False)
        print('epoch %d | train loss %.4f | validation AUC %.4f | validation AUPR %.4f'
              % (epoch, row['train_loss'], validation['auc'], validation['aupr']))

    checkpoint = torch.load(args.output_dir / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    with torch.no_grad():
        *_, probabilities, details = model(*inputs, return_details=True)
        metrics = {name: response_metrics(data.labels, details['logits'].reshape(-1), probabilities,
                                          data.masks[name], checkpoint['decision_threshold'])
                   for name in ('train', 'validation', 'test')}
        save_objects(args.output_dir, data, details, probabilities,
                     checkpoint['epoch'], checkpoint['decision_threshold'])
    write_json(args.output_dir / 'metrics.json', {
        'selected_epoch': checkpoint['epoch'], 'selection_metric': 'validation_auc',
        'elapsed_seconds': time.time() - started, **metrics})
    print('Selected epoch:', checkpoint['epoch'])
    print('Held-out test:', metrics['test'])
    print('Saved research objects and checkpoint:', args.output_dir)
    return args.output_dir
