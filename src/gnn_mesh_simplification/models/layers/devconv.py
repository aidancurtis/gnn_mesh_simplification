import torch.nn as nn
from torch_scatter import scatter_max


class DevConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DevConv, self).__init__()
        self.W_theta = nn.Linear(in_channels, out_channels, bias=False)
        self.W_phi = nn.Linear(out_channels, out_channels, bias=False)

        self.activation = nn.ReLU()

    def forward(self, x, edges):
        # x: [num_verts, channels]
        # edges: [2, num_edges]
        row, col = edges
        x_i = x[row]  # [num_edges, in_channels]
        x_j = x[col]

        diff = x_i - x_j  # [num_edges, channels]
        weighted_diff = self.W_theta(diff)  # [num_edges, out_channels]
        max_diff, _ = scatter_max(
            weighted_diff, col, dim=0, dim_size=x.shape[0]
        )  # [num_verts, out_channels]
        out = self.W_phi(max_diff)  # [num_verts, out_channels]
        out = self.activation(out)

        return out
