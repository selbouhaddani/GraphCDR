import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


class GraphDataset(Dataset):
    """A list of molecular graphs, batched by the PyG collate function below."""

    def __init__(self, graphs_dict):
        super().__init__()
        self.graphs = [
            Data(x=torch.as_tensor(features, dtype=torch.float32),
                 edge_index=torch.as_tensor(edge_index, dtype=torch.long))
            for features, edge_index in graphs_dict
        ]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def collate(data_list):
    return Batch.from_data_list(data_list)
