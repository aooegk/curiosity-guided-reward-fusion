import torch
import torch.nn as nn

from networks.mlp import build_mlp


class CriticNetwork(nn.Module):


    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_depth: int = 2,
    ):
        super().__init__()

        input_dim = observation_dim + action_dim

        self.q1 = build_mlp(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        )

        self.q2 = build_mlp(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):

        inputs = torch.cat([observations, actions], dim=-1)

        q1 = self.q1(inputs)
        q2 = self.q2(inputs)

        return q1, q2

    def q_min(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:

        q1, q2 = self.forward(observations, actions)
        return torch.min(q1, q2)