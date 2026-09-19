import torch
import torch.nn as nn


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    hidden_depth: int = 2,
    activation: nn.Module = nn.ReLU,
):
    layers = []

    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(activation())

    for _ in range(hidden_depth - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation())

    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)