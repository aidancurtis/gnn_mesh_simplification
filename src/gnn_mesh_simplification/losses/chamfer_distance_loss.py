import torch
import torch.nn as nn


class ChamferDistanceLoss(nn.Module):
    def __init__(self):
        super(ChamferDistanceLoss, self).__init__()

    def forward(self, input_vertex_set, sampled_points, probs):
        diff = torch.cdist(input_vertex_set, sampled_points) ** 2

        l1 = torch.dot(probs, diff.min(dim=0).values)
        l2 = torch.dot(probs[diff.min(dim=1).indices], diff.min(dim=1).values)

        return l1 + l2

    # def compute_distances(self, p1, p2):
    #     return
