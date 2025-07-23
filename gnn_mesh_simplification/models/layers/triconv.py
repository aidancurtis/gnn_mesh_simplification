import torch
import torch.nn as nn
from torch_scatter import scatter_min, scatter_max, scatter_add


class TriConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TriConv, self).__init__()

        mlp_in_channels = in_channels + 9

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, pos, edges):
        row, col = edges

        rel_pos_encoding = self.compute_rel_pos_encoding(pos, edges)
        x_diff = x[row] - x[col]

        mlp_in = torch.cat([rel_pos_encoding, x_diff], dim=-1)
        mlp_out = self.mlp(mlp_in)
        out = scatter_add(mlp_out, col, dim=0, dim_size=x.shape[0])

        return out

    def compute_rel_pos_encoding(self, pos, edges):
        # pos: [num_faces, 3, 3]
        row, col = edges  # edges: [2, num_edges]

        edge_vecs = pos[row] - pos[col]  # [num_edges, 3, 3]
        t_max, _ = scatter_max(
            edge_vecs, col, dim=0, dim_size=pos.shape[0]  # [num_faces, 3, 3]
        )
        t_min, _ = scatter_min(
            edge_vecs, col, dim=0, dim_size=pos.shape[0]  # [num_faces, 3, 3]
        )

        t_max_diff = t_max[row] - t_max[col]  # [num_edges, 3, 3]
        t_min_diff = t_min[row] - t_min[col]  # [num_edges, 3, 3]

        barycenter = pos.mean(dim=1, keepdim=True)  # [num_faces, 3, 1]
        bary_diff = barycenter[row] - barycenter[col]  # [num_edges, 3, 1]
        bary_diff = bary_diff.expand_as(t_max_diff)  # [num_edges, 3, 3]

        rel_pos_encoding = torch.cat(
            [t_min_diff, t_max_diff, bary_diff], dim=-1  # [num_edges, 3, 9]
        )
        return rel_pos_encoding
