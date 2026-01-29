import torch
import torch.nn as nn

from .layers.devconv import DevConv


class PointSampler(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(PointSampler, self).__init__()
        self.devconvs = nn.ModuleList()
        self.devconvs.append(DevConv(in_channels, out_channels))
        for _ in range(num_layers - 1):
            self.devconvs.append(DevConv(out_channels, out_channels))

        self.output_layer = nn.Linear(out_channels, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edges):
        for devconv in self.devconvs:
            x = devconv(x, edges)

        scores = self.output_layer(x).squeeze(-1)
        probs = self.sigmoid(scores)

        return probs

    def sample(self, probabilities, num_samples):
        if num_samples > probabilities.shape[0]:
            raise ValueError(
                f"Num samples ({num_samples}) exceeds the maximum number of points ({probabilities.shape[0]})"
            )
        indices = torch.multinomial(probabilities, num_samples)

        return indices

    def forward_and_sample(self, x, edges, num_samples):
        probabilities = self.forward(x, edges)
        indices = self.sample(probabilities, num_samples)
        return indices, probabilities
