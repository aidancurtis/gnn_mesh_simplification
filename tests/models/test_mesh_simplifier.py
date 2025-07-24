import torch
import torch.nn as nn
from torch_geometric.data import Data

from gnn_mesh_simplification.models import MeshSimplifier


def test_mesh_simplifier():
    mesh_simplifier = MeshSimplifier(
        point_sampler_in_channels=3,
        point_sampler_out_channels=64,
        point_sampler_num_layers=3,
        edge_predictor_in_channels=3,
        edge_predictor_hidden_channels=16,
        edge_predictor_k=3,
        face_classifier_in_channels=1,
        face_classifier_hidden_channels=64,
        face_classifier_num_layers=3,
        face_classifier_k=3,
        ratio=0.05,
        device=torch.device("cpu"),
    )

    x = torch.randn((10, 3))
    edges = torch.randint(10, size=(2, 20))
    condition = edges[0] != edges[1]
    edges = edges[:, condition].unique(dim=-1)
    data = Data(pos=x, edge_index=edges)

    output = mesh_simplifier(data)
