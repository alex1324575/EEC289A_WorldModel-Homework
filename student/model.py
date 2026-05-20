"""Student world model.

Students may replace this residual MLP with a GRU or another dynamics model,
but the public interface must stay the same.

Architecture (Exp4): acceleration prediction + semi-implicit Euler integration.
  Instead of predicting a 4D state-delta directly, the model predicts 2D
  accelerations [a_cart, a_pole] and applies the analytically exact kinematic
  update:

    x_dot_new = x_dot + a_cart * dt          (velocity update)
    omega_new = omega + a_pole * dt
    x_new     = x    + x_dot_new * dt        (position uses NEW velocity — semi-implicit)
    theta_new = theta + omega_new * dt

  Semi-implicit Euler is more stable than explicit Euler for oscillatory systems
  because position feedback uses the already-corrected velocity.  The position
  deltas (delta_x, delta_theta) are therefore determined analytically from the
  current state — the model only has to learn the non-trivial part (accelerations).

  Physics buffers (obs_mean, obs_std, delta_mean, delta_std, accel_scale) are
  registered so they follow model.to(device) automatically.  They are populated
  lazily from the normalizer at the start of the first compute_loss call.

Sin/cos angle features (obs_norm[:, 1:2]) — retained from Exp1:
  Primary value: bounding the angle representation to [-1, 1] when rollout
  drifts beyond the training distribution.

Output limiter — retained from Exp2:
  delta_limit=5.5 tanh applied to the normalised delta output as a safety net.
  Training delta abs-max ≈ 5.3σ; limiter keeps tanh near-linear within the
  training support while saturating extreme OOD outputs.
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
        dt: float = 0.04,  # InvertedPendulum-v5: frame_skip=2, timestep=0.02 → dt=0.04
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        self.dt = float(dt)

        # +2 for sin/cos of the pole angle (obs dimension 1)
        in_dim = obs_dim + act_dim + 2
        layers: list[nn.Module] = []
        for _ in range(int(num_layers)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None

        # Head predicts 2D accelerations [a_cart, a_pole] — not 4D delta.
        # Position deltas are derived analytically via semi-implicit Euler.
        self.head = nn.Linear(hidden_dim, 2)

        # Normalizer statistics needed for physics integration inside forward.
        # Filled lazily by losses.compute_loss; follow model.to(device) via buffer.
        self.register_buffer("obs_std",    torch.ones(obs_dim))
        self.register_buffer("obs_mean",   torch.zeros(obs_dim))
        self.register_buffer("delta_std",  torch.ones(obs_dim))
        self.register_buffer("delta_mean", torch.zeros(obs_dim))
        # accel_scale[i] = 5σ of training acceleration for dim i.
        # = 5 * delta_std[2 or 3] / dt.  Filled by lazy init in losses.py.
        self.register_buffer("accel_scale", torch.ones(2))

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        # ── encoder (same as Exp1–3) ──────────────────────────────────────────
        angle = obs_norm[:, 1:2]
        x_enc = torch.cat([obs_norm, act_norm, torch.sin(angle), torch.cos(angle)], dim=-1)
        feat = self.encoder(x_enc)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden

        # ── acceleration head ─────────────────────────────────────────────────
        raw = self.head(feat)  # [B, 2]
        # Soft-clamp into ±accel_scale (≈ ±5σ of training accelerations).
        # accel_scale is in real-space [m/s² or rad/s²] units.
        a = torch.tanh(raw) * self.accel_scale  # [B, 2]
        a_cart = a[:, 0]
        a_pole = a[:, 1]

        # ── semi-implicit Euler integration ───────────────────────────────────
        # Denormalise: obs_norm = (obs - obs_mean) / obs_std  →  obs = obs_norm * obs_std + obs_mean
        obs_real = obs_norm * self.obs_std + self.obs_mean  # [B, 4]
        x_pos = obs_real[:, 0]
        theta = obs_real[:, 1]
        x_dot = obs_real[:, 2]
        omega = obs_real[:, 3]

        dt = self.dt
        # 1. update velocities
        x_dot_new = x_dot + a_cart * dt
        omega_new = omega + a_pole * dt
        # 2. update positions using the NEW velocities (semi-implicit)
        x_new = x_pos + x_dot_new * dt
        theta_new = theta + omega_new * dt

        delta_real = torch.stack(
            [x_new - x_pos, theta_new - theta, x_dot_new - x_dot, omega_new - omega],
            dim=-1,
        )  # [B, 4], real-space units

        # ── normalise delta for predict_next compatibility ────────────────────
        # predict_next will denormalise: delta = delta_norm * delta_std + delta_mean
        # → we must return (delta_real - delta_mean) / delta_std
        delta_norm = (delta_real - self.delta_mean) / self.delta_std  # [B, 4]

        # Final soft limiter (Exp2): training abs-max ≈ 5.3σ, limit=5.5 keeps
        # tanh near-linear within training support.
        delta_norm = self.delta_limit * torch.tanh(delta_norm / self.delta_limit)

        return delta_norm, hidden
