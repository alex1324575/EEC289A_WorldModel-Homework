"""Locked baseline residual MLP."""

from __future__ import annotations

import torch
from torch import nn


class BaselineResidualMLP(nn.Module):
    def __init__(self, obs_dim: int = 4, act_dim: int = 1, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim + act_dim
        for _ in range(int(num_layers)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, obs_dim))
        self.net = nn.Sequential(*layers)

    def initial_hidden(self, batch_size: int, device: torch.device):
        return None

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        return self.net(torch.cat([obs_norm, act_norm], dim=-1)), hidden
