import pytest
import torch
import torch.nn as nn

from gnn_mesh_simplification.models import PointSampler


def test_point_sampler():
    x = torch.randn((100, 3))
    edges = torch.randint(100, (2, 200))

    sampler = PointSampler(in_channels=3, out_channels=32, num_layers=3)

    assert len(sampler.devconvs) == 3
    assert isinstance(sampler.sigmoid, nn.Sigmoid)

    sample_indices, probabilities = sampler.forward_and_sample(x, edges, num_samples=10)
    assert sample_indices.shape[0] == 10
    assert probabilities.shape == (100,)

    with pytest.raises(Exception) as e:
        sampler.forward_and_sample(x, edges, num_samples=250)
    assert (
        e.value.args[0]
        == "Num samples (250) exceeds the maximum number of points (100)"
    )
