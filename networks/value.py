import torch
import torch.nn as nn

from networks.mlp import build_mlp


class ValueNetwork(nn.Module):

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 256,
        hidden_depth: int = 2,
    ):
        super().__init__()

        self.net = build_mlp(
            input_dim=observation_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        return self.net(observations)