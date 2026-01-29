import torch

from gnn_mesh_simplification.losses import ChamferDistanceLoss


def test_chamfer_distance_loss():
    chamfer_loss = ChamferDistanceLoss()

    input_vertex_set = torch.randn(10, 3)
    simplified_vertex_set = input_vertex_set[torch.tensor([2, 5, 7, 8, 9])]
    probabilities = torch.softmax(
        torch.randint(10, size=(5,), dtype=torch.float32), dim=0
    )
    loss = chamfer_loss(input_vertex_set, simplified_vertex_set, probabilities)

    assert isinstance(loss, torch.Tensor)
    assert loss > 0
