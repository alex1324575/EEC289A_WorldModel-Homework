"""One-step residual MLP baseline for Pendulum dynamics."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class BaselineMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, param_dim: int, num_tasks: int, cfg: dict[str, Any]):
        super().__init__()
        hidden = int(cfg.get("hidden_dim", 128))
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.param_dim = param_dim
        self.num_tasks = num_tasks
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )

    def initial_hidden(self, batch_size: int, device: torch.device) -> None:
        return None

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        task_params: torch.Tensor,
        task_id: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del task_params, task_id
        x = torch.cat([state, action], dim=-1)
        return state + self.net(x), hidden
