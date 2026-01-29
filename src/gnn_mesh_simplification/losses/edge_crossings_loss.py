import torch
import torch.nn as nn
from torch_cluster import knn


class EdgeCrossingsLoss(nn.Module):
    def __init__(self, k):
        super(EdgeCrossingsLoss, self).__init__()
        self.k = k

    def forward(self, vertices, faces, probabilities):
        num_faces = faces.shape[0]
        pos = vertices[faces]

        barycenters = self.compute_barycenters(vertices, faces)
        edge_index = knn(barycenters, barycenters, k=self.k)
        edge_index = self.remove_redundant_edges(edge_index)

        edges_crossings = self.calculate_edge_crossings(pos, edge_index)
        weighted_edge_crossings = torch.dot(probabilities, edges_crossings)
        return weighted_edge_crossings / num_faces

    def remove_redundant_edges(self, edges):
        m1 = edges[0] != edges[1]
        edges = edges.t()[m1].t()
        return edges

    def compute_barycenters(self, vertices, faces):
        return vertices[faces].mean(dim=1)

    def calculate_edge_crossings(self, pos, knn_edge_index):
        num_faces = pos.shape[0]
        edges = self.compute_face_edges(pos)
        starting_pos = pos[:, torch.tensor([0, 0, 1])]

        edge_crossings = torch.zeros(size=(num_faces,), dtype=torch.float32)
        for idx in range(knn_edge_index.shape[1]):
            tri_idx = int(knn_edge_index[0, idx].item())
            neighbor_idx = int(knn_edge_index[1, idx].item())

            for i in range(3):
                for j in range(3):
                    tri_line = torch.stack(
                        [starting_pos[tri_idx, i], edges[tri_idx, i]]
                    )
                    int_line = torch.stack(
                        [starting_pos[neighbor_idx, j], edges[neighbor_idx, j]]
                    )

                    if self.check_edge_collision(tri_line, int_line):
                        edge_crossings[tri_idx] += 1

        return edge_crossings

    def compute_face_edges(self, pos):
        return pos[:, torch.tensor([1, 2, 2])] - pos[:, torch.tensor([0, 0, 1])]

    def check_edge_collision(self, tri_line, int_line):
        eps = 1e-5
        if torch.linalg.matrix_norm(tri_line - int_line) < eps:
            return False

        a1, b1 = tri_line
        a2, b2 = int_line

        distance = torch.abs(
            torch.dot(a2 - a1, torch.linalg.cross(b1, b2))
        ) / torch.linalg.vector_norm(torch.linalg.cross(b1, b2))

        return distance < 1e-5
