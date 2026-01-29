import torch
from torch_cluster import knn

from gnn_mesh_simplification.losses import TriangleCollisionLoss


def test_triangle_collision_loss():
    triangle_collision_loss = TriangleCollisionLoss(10)
    vertices = torch.randn(10, 3)
    faces = torch.tensor(
        [[0, 1, 2], [1, 2, 3], [2, 3, 4], [5, 8, 9], [0, 4, 6], [7, 9, 3]]
    )
    probabilities = torch.softmax(
        torch.randint(10, size=(6,), dtype=torch.float32), dim=0
    )

    loss = triangle_collision_loss(vertices, faces, probabilities)

    assert loss > 0


def test_compute_faces_penetrated():
    # test one
    triangle_collision_loss = TriangleCollisionLoss(4)
    vertices = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32
    )
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    pos = vertices[faces]

    barycenters = triangle_collision_loss.compute_barycenters(vertices, faces)
    edge_index = knn(barycenters, barycenters, k=4)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    faces_penetrated = triangle_collision_loss.compute_faces_penetrated(pos, edge_index)

    assert faces_penetrated.shape[0] == 4
    assert torch.linalg.norm(faces_penetrated - torch.tensor([0, 0, 0, 0])) <= 1e-5

    # test two
    triangle_collision_loss = TriangleCollisionLoss(4)
    vertices = torch.tensor(
        [
            [0, 0, -0.5],
            [0, 0, 0.5],
            [1, 0, 0],
            [0, 0.5, 0],
            [0, -0.5, 0],
            [-1, 0, 0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 2, 1], [3, 4, 5]])
    pos = vertices[faces]
    probabilities = torch.tensor([0.5, 0.5])

    barycenters = triangle_collision_loss.compute_barycenters(vertices, faces)
    edge_index = knn(barycenters, barycenters, k=4)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    faces_penetrated = triangle_collision_loss.compute_faces_penetrated(pos, edge_index)

    loss = triangle_collision_loss(vertices, faces, probabilities)

    assert faces_penetrated.shape[0] == 2
    assert torch.linalg.norm(faces_penetrated - torch.tensor([1, 1])) <= 1e-5

    assert loss == 0.5
