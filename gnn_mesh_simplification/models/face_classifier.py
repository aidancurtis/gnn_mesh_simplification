import torch
import torch.nn as nn

from .layers import TriConv


class FaceClassifier(nn.Module):
    def __init__(self, k, in_channels, hidden_channels, num_layers):
        super(FaceClassifier, self).__init__()
        self.k = k
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.triconvs = nn.ModuleList()
        self.triconvs.append(TriConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.triconvs.append(TriConv(hidden_channels, hidden_channels))

        self.output_layer = nn.Linear(hidden_channels, 1)

    def forward(self, pos, probs):
        # pos: [num_candidate_faces, 3, 3]
        # probs: [num_candidate_faces, 1]
        if pos.dim() == 3:
            pos = pos.mean(dim=1)

        edges = self.build_knn_graph(x=pos, k=self.k)

        for triconv in self.triconvs:
            probs = triconv(probs, pos, edges)
            probs = torch.relu(probs)

        out = self.output_layer(probs)
        logits = out.squeeze(-1)

        probs = torch.softmax(logits, dim=0)

        return probs

    def build_knn_graph(self, x, k):
        batch_size = 1
        edge_index = []

        batch = x
        n = x.shape[0]

        distances = torch.cdist(batch, batch)
        distances.fill_diagonal_(float("inf"))
        closest_k = torch.topk(distances, k, dim=1, largest=False).indices.flatten()

        nums = torch.arange(0, n, dtype=torch.int32).unsqueeze(1).repeat(1, k).flatten()
        edges = torch.stack((closest_k, nums))
        extended_edges = torch.cat([edges, edges.flip(0)], dim=1).unique(dim=1)

        return extended_edges
