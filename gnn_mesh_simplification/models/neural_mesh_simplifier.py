import torch
import torch.nn as nn
import torch_geometric.utils

from .point_sampler import PointSampler
from .edge_predictor import EdgePredictor
from .face_classifier import FaceClassifier


class MeshSimplifier(nn.Module):
    def __init__(
        self,
        point_sampler_in_channels,
        point_sampler_out_channels,
        point_sampler_num_layers,
        edge_predictor_in_channels,
        edge_predictor_hidden_channels,
        edge_predictor_k,
        face_classifier_in_channels,
        face_classifier_hidden_channels,
        face_classifier_num_layers,
        face_classifier_k,
        ratio,
        device=torch.device("cpu"),
    ):
        super(MeshSimplifier, self).__init__()
        self.device = device
        self.ratio = ratio
        self.k = face_classifier_k

        self.point_sampler = PointSampler(
            in_channels=point_sampler_in_channels,
            out_channels=point_sampler_out_channels,
            num_layers=point_sampler_num_layers,
        ).to(device)
        self.edge_predictor = EdgePredictor(
            in_channels=edge_predictor_in_channels,
            hidden_channels=edge_predictor_hidden_channels,
            k=edge_predictor_k,
        ).to(device)
        self.face_classifier = FaceClassifier(
            in_channels=face_classifier_in_channels,
            hidden_channels=face_classifier_hidden_channels,
            num_layers=face_classifier_num_layers,
            k=face_classifier_k,
        ).to(device)

    def forward(self, data):
        pos = data.pos
        edges = data.edge_index
        num_nodes = pos.shape[0]

        sampled_indices, sampled_probs = self.sample_points(pos=pos, edges=edges)
        sampled_indices = sampled_indices.to(self.device)
        sampled_pos = pos[sampled_indices].to(self.device)
        sampled_edge_index, _ = torch_geometric.utils.subgraph(
            subset=sampled_indices,
            edge_index=edges,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )

        sampled_edge_index = sampled_edge_index.to(self.device)
        adj_edge_index, adj_edge_prob = self.edge_predictor(
            sampled_pos, sampled_edge_index
        )
        candidate_faces, candidate_faces_probs = self.compute_candidate_faces(
            adj_edge_index, adj_edge_prob
        )

        candidate_faces_probs = candidate_faces_probs.to(self.device)
        face_pos = sampled_pos[candidate_faces].to(self.device)
        face_probs = self.face_classifier(face_pos, candidate_faces_probs)

        if face_probs.shape[0] > 0:
            threshold = torch.quantile(face_probs, 1 - self.ratio)
            simplified_faces = candidate_faces[face_probs > threshold]
        else:
            simplified_faces = torch.empty((0, 3), dtype=torch.long, device=self.device)

        return {
            "sampled_indices": sampled_indices,
            "sampled_probs": sampled_probs,
            "sampled_pos": sampled_pos,
            "adj_edge_index": adj_edge_index,
            "adj_edge_prob": adj_edge_prob,
            "candidate_faces": candidate_faces,
            "candidate_probs": candidate_faces_probs,
            "face_probs": face_probs,
            "simplified_faces": simplified_faces,
        }

    def sample_points(self, pos, edges):
        num_nodes = pos.shape[0]
        target_nodes = min(max(int((1 - self.ratio) * num_nodes), 1), num_nodes)

        sampled_probs = self.point_sampler(pos, edges)
        sampled_indices = self.point_sampler.sample(sampled_probs, target_nodes)
        return sampled_indices, sampled_probs

    def compute_candidate_faces(self, adj_edge_index, adj_edge_prob):
        num_nodes = adj_edge_index.max().item() + 1

        adj_matrix = torch.zeros((num_nodes, num_nodes), device=self.device)
        adj_matrix[adj_edge_index[0], adj_edge_index[1]] = adj_edge_prob

        k = min(self.k, num_nodes - 1)
        _, knn_indices = torch.topk(adj_matrix, k=k, dim=1)

        triangles = []
        triangle_probs = []
        for i in range(num_nodes):
            neighbors = knn_indices[i]
            for j in range(k):
                for k in range(j + 1, k):
                    n1 = neighbors[j]
                    n2 = neighbors[k]
                if adj_matrix[n1, n2] > 0:
                    triangle = torch.tensor([i, n1, n2], device=self.device)
                    triangles.append(triangle)

                    prob = (
                        adj_matrix[i, n1] * adj_matrix[i, n2] * adj_matrix[n1, n2]
                    ) / 3
                    triangle_probs.append(prob)

        if triangles:
            triangles = torch.stack(triangles)
            triangle_probs = torch.tensor(triangle_probs, device=self.device)
        else:
            triangles = torch.empty((0, 3), dtype=torch.long, device=self.device)
            triangle_probs = torch.empty(0, dtype=torch.float16, device=self.device)

        return triangles, triangle_probs
