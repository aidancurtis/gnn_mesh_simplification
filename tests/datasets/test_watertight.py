import os
from gnn_mesh_simplification.datasets import Watertight
from torch_geometric.data import Data


def test_tosca():
    data_path = os.path.abspath("data/watertight")
    dataset = Watertight(data_path)

    assert len(dataset) == 400
    assert dataset.num_classes == 20
    assert dataset.__repr__() == "Watertight(400)"

    assert len(dataset[0]) == 4
