import torch

from gnn_mesh_simplification.models import EdgePredictor
from gnn_mesh_simplification.models.layers import DevConv


def test_edge_predictor():
    x = torch.randn(50, 3)
    edges = torch.randint(0, 50, size=(2, 150)).unique(dim=0)
    edge_predictor = EdgePredictor(k=5, in_channels=3, out_channels=64)
    assert edge_predictor.k == 5
    assert isinstance(edge_predictor.devconv, DevConv)

    indices, values = edge_predictor(x, edges)
    assert indices.shape[0] == 2
    assert values.shape[0] == indices.shape[1]

    s = torch.sparse_coo_tensor(indices=indices, values=values, size=(50, 50))
    s.to_dense()
