import os

import torch
import trimesh

from gnn_mesh_simplification.losses import EdgeCrossingsLoss


def test_edge_crossings_loss():
    model = os.path.abspath("data/test_models/two_triangles.obj")
    mesh = trimesh.load_mesh(model)
    faces = torch.tensor(mesh.faces)
    vertices = torch.tensor(mesh.vertices)
    probs = torch.tensor([1, 1], dtype=torch.float32)

    edge_crossings_loss = EdgeCrossingsLoss(2)
    loss = edge_crossings_loss(vertices, faces, probs)

    assert loss == 1
