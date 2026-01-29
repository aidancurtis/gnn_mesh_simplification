import os

from gnn_mesh_simplification.datasets import TOSCA

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "../data/tosca")

dataset = TOSCA(data_dir)
data = dataset[0]
print(data)
x = data["pos"]
edges = data["edge_index"]
faces = data["face"]
faces = faces.t()
# face_c = FaceClassifier(20, 20, )
# g = face_c(x, edges, faces)
# mesh = trimesh.Trimesh(vertices=x, faces=faces)
# # Number of vertices
# n_vertices = len(mesh.vertices)

# # Default color for all vertices (white)
# colors = np.tile([255, 255, 255, 255], (n_vertices, 1))  # RGBA

# sampler = PointSampler(in_channels=3, out_channels=32, num_layers=3)
# sample_indices, probs = sampler.forward_and_sample(x, edges, num_samples=100)

# print(sample_indices)

# for i in sample_indices:
#     # Set the target vertex to red
#     colors[i] = [255, 0, 0, 255]

# # Assign the colors to the mesh
# mesh.visual.vertex_colors = colors # type: ignore

# # Show the mesh
# mesh.show()

# import pytest
# import torch
# import torch.nn as nn

# from gnn_mesh_simplification.models import EdgePredictor


# def test_edge_predictor():
#     x = torch.randn(50, 3)
#     x = torch.tensor(
#         [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
#         dtype=torch.float,
#     )
#     # edges = torch.randint(0, 50, size=(2, 150)).unique(dim=0)
#     edges = torch.tensor(
#         [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]], dtype=torch.long
#     )
#     edge_predictor = EdgePredictor(k=2, in_channels=3, out_channels=64)
#     indices, values = edge_predictor.forward(x, edges)
#     # print(f"Indices: {indices}")
#     # print(f"Values: {values}")
#     print()
#     s = torch.sparse_coo_tensor(indices=indices, values=values, size=(4, 4))

#     s_dense = s.to_dense()
#     print(s_dense)


# test_edge_predictor()
