import torch
import torch.nn as nn

from gnn_mesh_simplification.models import MeshSimplifier


def test_mesh_simplifier():
    mesh_simplifier = MeshSimplifier(
        point_sampler_in_channels=3,
        point_sampler_out_channels=64,
        point_sampler_num_layers=3,
        edge_predictor_in_channels=3,
        edge_predictor_hidden_channels=16,
        edge_predictor_k=16,
        face_classifier_in_channels=1,
        face_classifier_hidden_channels=64,
        face_classifier_num_layers=3,
        face_classifier_k=16,
        ratio=0.05,
        device=torch.device("cpu"),
    )
