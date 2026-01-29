# import torch
# import torch.nn as nn
# from torch_cluster import knn
#
#
# class OverlappingTriangleLoss(nn.Module):
#     def __init__(self, k, num_samples, device="cpu"):
#         super(OverlappingTriangleLoss, self).__init__
#         self.k = k
#         self.num_samples = num_samples
#         self.device = device
#
#     def forward(self, vertices, faces, probabilities):
#         num_faces = faces.shape[0]
#         pos = vertices[faces]
#
#         barycenters = self.compute_barycenters(vertices, faces)
#         edge_index = knn(barycenters, barycenters, k=self.k)
#         edge_index = self.remove_redundant_edges(edge_index)
#
#         overlapping_triangles = self.calculate_overlapping_triangles(
#             vertices, faces, edge_index
#         )
#
#     def compute_barycenters(self, vertices, faces):
#         return vertices[faces].mean(dim=1)
#
#     def remove_redundant_edges(self, edges):
#         m1 = edges[0] != edges[1]
#         edges = edges.t()[m1].t()
#         return edges
#
#     def calculate_overlapping_triangles(self, vertices, faces, edge_index):
#         num_faces = faces.shape[0]
#         sample_points = self.generate_sampled_points(vertices, faces)
#
#         overlapping_triangles = torch.zeros(size=(num_faces,), dtype=torch.float32)
#         for i in range(num_faces):
#             for j in range(self.num_samples):
#                 p = sample_points[i, j]
#                 if
#
#     def generate_sampled_points(self, vertices, faces):
#         num_faces = faces.shape[0]
#         pos = vertices[faces]
#
#         s1 = torch.sqrt(torch.rand(num_faces, self.num_samples, 1, device=self.device))
#         r2 = torch.randn(num_faces, self.num_samples, 1, device=self.device)
#
#         a = 1.0 - s1
#         b = s1 * (1 - r2)
#         c = s1 * r2
#
#         samples = a * pos[:, None, 0] + b * pos[:, None, 1] + c * pos[:, None, 2]
#         return samples.reshape(-1, 3)
