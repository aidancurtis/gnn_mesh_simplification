import os

import torch
from torch_geometric.transforms import NormalizeScale

from gnn_mesh_simplification.datasets import TOSCA


def test_tosca():
    data_path = os.path.abspath("data/tosca")
    dataset = TOSCA(data_path, pre_transform=NormalizeScale, force_reload=True)
    data = dataset[0]

    assert len(dataset) == 80
    assert dataset.num_classes == 9
    assert dataset.__repr__() == "TOSCA(80)"

    assert len(data) == 4
    assert torch.norm(data["pos"][0] - torch.tensor([0.0240, 0.3150, 0.0594])) < 1e-3
