"""Ordered model inputs and reproducible, disjoint response splits."""
from collections import namedtuple

import numpy as np
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

from graphset import GraphDataset, collate


PreparedData = namedtuple('PreparedData', [
    'drug_loader', 'cell_loader', 'train_edges', 'labels', 'masks',
    'cell_line_ids', 'drug_ids', 'atom_shape', 'feature_names',
])


def CalculateGraphFeat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i, nodes in enumerate(adj_list):
        for node in nodes:
            adj_mat[i, int(node)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    return [feat_mat, np.vstack(np.where(adj_mat == 1))]


def process(drug_feature, mutation_feature, gexpr_feature, methylation_feature,
            data_new, nb_celllines, nb_drugs, validation_fraction=0.1,
            test_fraction=0.1, split_seed=666):
    """Split observed pairs; only training responses enter message passing.

    Fractions are relative to all observed pairs. Held-out partitions are
    stratified by response label. This holds out pairs, not entire nodes.
    """
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError('Validation and test fractions must be positive and sum to less than 1.')

    cell_ids = sorted({item[0] for item in data_new})
    drug_ids = sorted({item[1] for item in data_new})
    if (len(cell_ids), len(drug_ids)) != (nb_celllines, nb_drugs):
        raise ValueError('Node counts do not match the response identifiers.')
    cell_index = {name: i for i, name in enumerate(cell_ids)}
    drug_index = {name: i for i, name in enumerate(drug_ids)}
    pairs = np.asarray([(cell_index[c], drug_index[d], label)
                        for c, d, label, *_ in data_new], dtype=np.int64)
    observed_indices = pairs[:, 0] * nb_drugs + pairs[:, 1]
    if len(np.unique(observed_indices)) != len(pairs):
        raise ValueError('Response pairs must be deduplicated before splitting.')
    if not np.all(np.isin(pairs[:, 2], [-1, 1])):
        raise ValueError('Expected response labels encoded as -1 or 1.')

    remaining, test_rows = train_test_split(
        np.arange(len(pairs)), test_size=test_fraction, random_state=split_seed,
        stratify=pairs[:, 2])
    train_rows, validation_rows = train_test_split(
        remaining, test_size=validation_fraction / (1 - test_fraction),
        random_state=split_seed, stratify=pairs[remaining, 2])
    masks = {}
    for name, rows in [('train', train_rows), ('validation', validation_rows),
                       ('test', test_rows), ('observed', np.arange(len(pairs)))]:
        mask = torch.zeros(nb_celllines * nb_drugs, dtype=torch.bool)
        mask[observed_indices[rows]] = True
        masks[name] = mask
    labels = torch.zeros(nb_celllines * nb_drugs, dtype=torch.float32)
    labels[observed_indices] = torch.from_numpy((pairs[:, 2] == 1).astype('float32'))

    # Graph nodes are ordered as all cell lines, followed by all drugs.
    train_edges = pairs[train_rows].copy()
    train_edges[:, 1] += nb_celllines
    train_edges = np.vstack((train_edges, train_edges[:, [1, 0, 2]]))

    drug_data = [CalculateGraphFeat(*drug_feature[name][:2]) for name in drug_ids]
    mutation_feature = mutation_feature.loc[cell_ids]
    gexpr_feature = gexpr_feature.loc[cell_ids]
    methylation_feature = methylation_feature.loc[cell_ids]
    mutation = torch.from_numpy(mutation_feature.to_numpy(dtype='float32'))[:, None, None, :]
    gexpr = torch.from_numpy(gexpr_feature.to_numpy(dtype='float32'))
    methylation = torch.from_numpy(methylation_feature.to_numpy(dtype='float32'))
    drug_loader = Data.DataLoader(
        GraphDataset(graphs_dict=drug_data), collate_fn=collate,
        batch_size=nb_drugs, shuffle=False)
    cell_loader = Data.DataLoader(
        Data.TensorDataset(mutation, gexpr, methylation), batch_size=nb_celllines, shuffle=False)
    feature_names = {
        'mutation': list(map(str, mutation_feature.columns)),
        'expression': list(map(str, gexpr_feature.columns)),
        'methylation': list(map(str, methylation_feature.columns)),
    }
    return PreparedData(drug_loader, cell_loader, train_edges, labels, masks,
                        list(map(str, cell_ids)), list(map(str, drug_ids)),
                        drug_data[0][0].shape[-1], feature_names)
