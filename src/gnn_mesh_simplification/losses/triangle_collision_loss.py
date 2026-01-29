import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch_cluster import knn


class TriangleCollisionLoss(nn.Module):
    def __init__(self, k):
        super(TriangleCollisionLoss, self).__init__()
        self.k = k

    def forward(self, vertices, faces, probabilities):
        num_faces = faces.shape[0]
        pos = vertices[faces]

        barycenters = self.compute_barycenters(vertices, faces)
        edge_index = knn(barycenters, barycenters, k=self.k)
        faces_penetrated = self.compute_faces_penetrated(pos, edge_index)

        weighted_faces_penetrated = torch.dot(probabilities, faces_penetrated)
        return weighted_faces_penetrated / num_faces

    def compute_faces_penetrated(self, pos, knn_edge_index):
        num_faces = pos.shape[0]

        edges = self.compute_face_edges(pos)
        normals = normalize(torch.linalg.cross(edges[:, 0], edges[:, 1]))

        faces_penetrated = torch.zeros(size=(num_faces,), dtype=torch.float32)
        for idx in range(knn_edge_index.shape[1]):
            tri_idx = int(knn_edge_index[0, idx].item())
            neighbor_idx = int(knn_edge_index[1, idx].item())

            if tri_idx == neighbor_idx:
                continue

            for k in range(3):
                n = normals[tri_idx]
                p0 = pos[tri_idx, 0]
                l0 = pos[neighbor_idx, k]
                l1 = pos[neighbor_idx, (k + 1) % 3]

                denom = torch.dot(n, l1 - l0)

                if abs(denom) < 1e-3:
                    continue

                t = torch.dot(n, p0 - l0) / denom

                if 0 < t < 1:
                    faces_penetrated[neighbor_idx] += 1
                    break

        return faces_penetrated

    def compute_face_edges(self, pos):
        return pos[:, torch.tensor([1, 2, 2])] - pos[:, torch.tensor([0, 0, 1])]

    def compute_barycenters(self, vertices, faces):
        return vertices[faces].mean(dim=1)
