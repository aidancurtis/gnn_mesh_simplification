import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_softmax

from .layers import DevConv


class EdgePredictor(nn.Module):
    def __init__(self, k, in_channels, hidden_channels):
        super(EdgePredictor, self).__init__()
        self.k = k
        self.devconv = DevConv(in_channels, hidden_channels)

        self.W_q = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.W_k = nn.Linear(hidden_channels, hidden_channels, bias=False)

    def forward(self, x, edges):
        knn_edges = knn_graph(x=x, k=self.k, flow="target_to_source")

        extended_edges = torch.cat([edges, knn_edges], dim=1)
        extended_edges = torch.unique(extended_edges, dim=1)

        features = self.devconv(x, extended_edges)

        attention_scores = self.compute_attention_scores(features, edges)
        indices, vals = self.compute_estimated_adjacency_matrix(
            features, edges, attention_scores
        )

        return indices, vals

    def compute_attention_scores(self, features, edges):
        row, col = edges

        q = self.W_q(features)  # [num_verts, hidden_channels]
        k = self.W_k(features)  # [num_verts, hidden_channels]

        attention = (q[row] * k[col]).sum(dim=-1)
        return scatter_softmax(attention, row, dim=0)

    def compute_estimated_adjacency_matrix(self, features, edges, attn_scores):
        row, col = edges
        num_verts = features.shape[0]

        S = torch.sparse_coo_tensor(
            indices=edges, values=attn_scores, size=(num_verts, num_verts)
        )

        A = torch.sparse_coo_tensor(
            indices=edges,
            values=torch.ones(edges.shape[1]),
            size=(num_verts, num_verts),
        )

        A_s = S @ A @ S.t()

        A_s = A_s.to_sparse_coo()
        edges = A_s.indices()
        row, col = edges
        vals = A_s.values()
        indices = torch.stack([row, col], dim=0)

        return indices, vals
