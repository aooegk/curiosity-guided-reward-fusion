import torch
import torch.nn as nn
from torch.distributions import Normal

from networks.mlp import build_mlp


class GaussianPolicy(nn.Module):


    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            max_action: float = 1.0,
            hidden_dim: int = 256,
            hidden_depth: int = 2,
            log_std_min: float = -5.0,
            log_std_max: float = 2.0,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.backbone = build_mlp(
            input_dim=observation_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, observations: torch.Tensor):
        features = self.backbone(observations)

        mean = torch.tanh(self.mean_layer(features)) * self.max_action

        log_std = self.log_std.unsqueeze(0).expand_as(mean)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_distribution(self, observations: torch.Tensor) -> Normal:


        mean, log_std = self.forward(observations)

        std = torch.exp(log_std)

        return Normal(mean, std)

    def log_prob(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:


        dist = self.get_distribution(observations)

        log_probs = dist.log_prob(actions)


        log_probs = log_probs.sum(dim=-1, keepdim=True)

        return log_probs

    @torch.no_grad()
    def act(
        self,
        observation,
        device: torch.device,
        deterministic: bool = True,
    ):


        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        dist = self.get_distribution(observation)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()


        action = torch.clamp(action, -self.max_action, self.max_action)


        return action.cpu().numpy()[0]