import os
from torch_geometric.transforms import NormalizeScale

from gnn_mesh_simplification.datasets import Watertight
from gnn_mesh_simplification.utils.transforms import mean_normalization


def test_tosca():
    data_path = os.path.abspath("data/watertight")
    dataset = Watertight(data_path, pre_transform=NormalizeScale, force_reload=True)

    assert len(dataset) == 400
    assert dataset.num_classes == 20
    assert dataset.__repr__() == "Watertight(400)"

    assert len(dataset[0]) == 4
