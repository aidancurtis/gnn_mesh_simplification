import os
from gnn_mesh_simplification.datasets import TOSCA


def test_tosca():
    data_path = os.path.abspath("data/tosca")
    print(data_path)
    dataset = TOSCA(data_path)

    assert len(dataset) == 80
    assert dataset.num_classes == 9
    assert dataset.__repr__() == "TOSCA(80)"

    assert len(dataset[0]) == 4
