import torch
from torch_geometric.data import Data


def mean_normalization(data: Data) -> Data:
    # Center the mesh
    data["pos"] -= data["pos"].mean(axis=0)

    # Scale to unit cube
    max_dim = torch.max(data["pos"].max(axis=0).values - data["pos"].min(axis=0).values)
    data["pos"] /= max_dim

    return data
