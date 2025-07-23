import torch
import torch.nn as nn

from gnn_mesh_simplification.models import FaceClassifier
from gnn_mesh_simplification.models.layers import TriConv


def test_face_classifier():
    face_classifier = FaceClassifier(
        k=3, in_channels=1, hidden_channels=128, num_layers=3
    )
    assert face_classifier.num_layers == 3
    assert face_classifier.k == 3
    assert isinstance(face_classifier.output_layer, nn.Linear)


def test_compute_rel_pos_encoding():
    pos = torch.rand((10, 3, 3))
    edges = torch.tensor(
        [
            [0, 1, 2, 1, 3, 5, 7, 5, 0, 5, 2, 6],
            [1, 0, 1, 2, 5, 3, 5, 7, 5, 0, 6, 2],
        ],
        dtype=torch.long,
    )

    layer = TriConv(1, 10)
    out = layer.compute_rel_pos_encoding(pos, edges)

    assert out.shape == (12, 3, 9)
