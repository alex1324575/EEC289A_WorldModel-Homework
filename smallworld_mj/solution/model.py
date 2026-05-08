"""Reference parameter-conditioned residual GRU dynamics model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ParamResidualGRU(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, param_dim: int, num_tasks: int, cfg: dict[str, Any]):
        super().__init__()
        hidden = int(cfg.get("hidden_dim", 256))
        task_emb = int(cfg.get("task_embedding_dim", 32))
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.param_dim = int(param_dim)
        self.num_tasks = int(num_tasks)
        self.task_emb = nn.Embedding(self.num_tasks, task_emb)
        self.input = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim + self.param_dim + task_emb, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, self.state_dim))

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        task_params: torch.Tensor,
        task_id: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            hidden = self.initial_hidden(state.shape[0], state.device)
        task_e = self.task_emb(task_id.long())
        x = torch.cat([state, action, task_params, task_e], dim=-1)
        feat = self.input(x)
        hidden = self.gru(feat, hidden)
        delta = self.head(hidden)
        return state + delta, hidden
