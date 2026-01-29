import torch
import torch.nn as nn
from torch_cluster import knn


class SurfaceDistanceLoss(nn.Module):
    """
    A class for the probabilistic surface distance loss
    """

    def __init__(self, k, num_points, device=torch.device("cpu")):
        super(SurfaceDistanceLoss, self).__init__()
        self.k = k
        self.num_sampled = num_points
        self.device = device

    def forward(
        self,
        input_verts,
        input_faces,
        simplified_verts,
        simplified_faces,
        simplified_probabilities,
    ):
        # Calulate forward loss
        forward = self.forward_loss(
            input_verts,
            input_faces,
            simplified_verts,
            simplified_faces,
            simplified_probabilities,
        )

        # Calculate reverse loss
        reverse = self.reverse_loss(
            input_verts,
            input_faces,
            simplified_verts,
            simplified_faces,
            simplified_probabilities,
        )

        return forward + reverse

    def forward_loss(
        self,
        input_verts,
        input_faces,
        simplified_verts,
        simplified_faces,
        simplified_probabilities,
    ):
        input_barycenters = self.compute_barycenters(input_verts, input_faces)
        simplified_barycenters = self.compute_barycenters(
            simplified_verts, simplified_faces
        )

        distances = self.compute_squared_distance(
            input_barycenters, simplified_barycenters
        )

        loss = torch.dot(distances.min(dim=0).values, simplified_probabilities)
        return loss

    def reverse_loss(
        self,
        input_verts,
        input_faces,
        simplified_verts,
        simplified_faces,
        simplified_probabilities,
    ):
        # Get sampled points from input vertex set and sampled vertices
        input_sampled_points = self.generate_sampled_points(
            input_verts, input_faces, self.num_sampled
        )
        simplified_sampled_points = self.generate_sampled_points(
            simplified_verts, simplified_faces, self.num_sampled
        )

        # Calculate sum of k minimum distances from each sampled vertices to sampled points from vertex set
        distances = self.compute_k_min_distances(
            simplified_sampled_points, input_sampled_points, self.k
        )

        # Scale and normalize distances
        max_distance = distances.max()
        scaled_distances = (distances / max_distance) * 0.1

        # Weigh with probabilities
        weighted_distances = scaled_distances * simplified_probabilities

        return weighted_distances

    def generate_sampled_points(self, vertices, faces, num_samples):
        num_faces = faces.shape[0]
        pos = vertices[faces]

        s1 = torch.sqrt(torch.rand(num_faces, num_samples, 1, device=self.device))
        r2 = torch.randn(num_faces, num_samples, 1, device=self.device)

        a = 1.0 - s1
        b = s1 * (1 - r2)
        c = s1 * r2

        samples = a * pos[:, None, 0] + b * pos[:, None, 1] + c * pos[:, None, 2]
        return samples.reshape(-1, 3)

    def compute_k_min_distances(self, points, vertices, k):
        # Get KNN edges
        edge_index = knn(vertices, points, k=k)

        # Calculate distances between sampled points and original mesh
        diff = points[edge_index[0]] - vertices[edge_index[1]]
        distances = torch.linalg.norm(diff, dim=-1)

        # Reshape distances to size [num_sampled_faces, ]
        distances = distances.reshape(self.num_sampled, -1)

        # Return sum for each sampled face
        return distances.sum(-1)

    def compute_barycenters(self, vertices, faces):
        return vertices[faces].mean(dim=1)

    def compute_squared_distance(self, p1, p2):
        return torch.cdist(p1, p2) ** 2
