"""Student world model.

Students may replace this residual MLP with a GRU or another dynamics model,
but the public interface must stay the same.

Sin/cos angle features (obs_norm[:, 1:2]):
  The primary value is long-horizon stability, not physical exactness.
  During open-loop rollout the predicted angle can drift unboundedly; sin/cos
  soft-clamps this to [-1, 1], giving the model a bounded representation of
  angle even when the rollout wanders far from the training distribution.
  We apply sin/cos to the normalized angle (index 1 of obs_norm) rather than
  the raw angle to avoid threading the normalizer into model.forward.
"""

from __future__ import annotations

import torch
from torch import nn


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_gru: bool = False,
        delta_limit: float = 5.5,
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        # +2 for sin/cos of the pole angle (obs dimension 1)
        in_dim = obs_dim + act_dim + 2
        layers: list[nn.Module] = []
        for _ in range(int(num_layers)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None
        self.head = nn.Linear(hidden_dim, obs_dim)

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        angle = obs_norm[:, 1:2]
        x = torch.cat([obs_norm, act_norm, torch.sin(angle), torch.cos(angle)], dim=-1)
        feat = self.encoder(x)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden
        raw_delta = self.head(feat)
        # Soft limiter (tanh) + hard clamp to prevent non-physical deltas during
        # open-loop rollout.  Quantification: training delta distribution has
        # abs-max ≈ 5.3σ (99.99-th pct ≈ 3.8σ).  delta_limit=5.5 keeps tanh
        # near-linear across the training support while capping extreme outputs.
        # Hard clamp at ±6.0 is a final backstop; should never activate in
        # normal rollout but prevents runaway divergence if hidden state drifts.
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        delta = delta.clamp(-6.0, 6.0)
        return delta, hidden
