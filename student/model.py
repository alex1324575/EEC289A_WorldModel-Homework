"""Student world model.

Students may replace this residual MLP with a GRU or another dynamics model,
but the public interface must stay the same.
"""

from __future__ import annotations

import torch
from torch import nn


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_gru: bool = True,
        delta_limit: float = 3.0,
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        self.hidden_dim = hidden_dim

        # Deep encoder: project [obs, act] → hidden features with LayerNorm + SiLU.
        in_dim = obs_dim + act_dim
        enc_layers: list[nn.Module] = []
        for _ in range(int(num_layers)):
            enc_layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*enc_layers)

        # Single GRUCell keeps hidden state as [B, H] (2D), which is compatible
        # with the locked CompiledWorldModel wrapper in eval_compiled.py.
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None

        # Two-layer decoder to improve expressivity after the GRU.
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        feat = self.encoder(torch.cat([obs_norm, act_norm], dim=-1))
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden
        raw_delta = self.decoder(feat)
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, hidden
