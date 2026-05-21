"""Student world model.

Students may replace this residual MLP with a GRU or another dynamics model,
but the public interface must stay the same.

Sin/cos angle features (obs_norm[:, 1:2]) — Exp1:
  Bounds angle representation to [-1,1] during rollout drift.

Physics correction for pole_angle (dim 1) — Exp6:
  Diagnostic 5 revealed prediction ratios: ang_vel=0.88, cart_vel=0.95,
  cart_pos=0.80, pole_angle=0.45. The model learns ang_vel well but the
  angle delta signal is small in normalised space and gets dominated.
  Fix: after the 4D head output, override dim 1 (pole_angle delta) with
  the analytically exact semi-implicit Euler integration using the model's
  own ang_vel prediction (which is reliable at ratio=0.88):
    Δω_real  = delta_norm[:,3] * delta_std[3] + delta_mean[3]
    ω_real   = obs_norm[:,3]   * obs_std[3]   + obs_mean[3]
    ω_new    = ω_real + Δω_real
    Δθ_phys  = ω_new * dt               (semi-implicit: uses updated ω)
    delta_norm[:,1] = (Δθ_phys - delta_mean[1]) / delta_std[1]
  Applied AFTER tanh + clamp so the physics value is not distorted.
  Buffers obs_{mean,std}, delta_{mean,std} filled lazily by losses.py.
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
        dt: float = 0.04,  # InvertedPendulum-v5: frame_skip=2 * timestep=0.02
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        self.dt = float(dt)
        # Normalizer stats for physics correction — filled lazily by losses.compute_loss.
        # register_buffer so they follow model.to(device) and are saved in checkpoints.
        self.register_buffer("obs_mean",   torch.zeros(obs_dim))
        self.register_buffer("obs_std",    torch.ones(obs_dim))
        self.register_buffer("delta_mean", torch.zeros(obs_dim))
        self.register_buffer("delta_std",  torch.ones(obs_dim))
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

        # Physics correction for pole_angle (dim 1) — applied AFTER tanh + clamp
        # so the physics value is not distorted by the soft limiter.
        # Uses the model's own ang_vel delta (dim 3, ratio=0.88) which is reliable.
        delta_omega_real = delta[:, 3] * self.delta_std[3] + self.delta_mean[3]  # real Δω
        omega_real = obs_norm[:, 3] * self.obs_std[3] + self.obs_mean[3]         # real ω
        omega_new = omega_real + delta_omega_real                                 # updated ω
        delta_theta_real = omega_new * self.dt                                    # Δθ = ω_new*dt
        delta_theta_norm = (delta_theta_real - self.delta_mean[1]) / self.delta_std[1]
        # Replace dim 1 (pole_angle) with physics value; torch.cat preserves autograd.
        delta = torch.cat([
            delta[:, 0:1],                      # cart_pos   — model's
            delta_theta_norm.unsqueeze(-1),      # pole_angle — PHYSICS
            delta[:, 2:3],                       # cart_vel   — model's
            delta[:, 3:4],                       # pole_ang_vel — model's
        ], dim=-1)

        return delta, hidden
