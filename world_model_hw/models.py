"""Small Dreamer-style model components.

This file is intentionally compact and explicit. It is the main place students
should read when connecting the course concepts to code:

- encoder/decoder learn observations
- RSSM learns latent dynamics
- reward and continuation heads make the latent state useful for control
- actor and critic are trained from imagined latent trajectories
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int = 2, activation=nn.ELU) -> nn.Sequential:
    modules: list[nn.Module] = []
    last_dim = input_dim
    for _ in range(layers):
        modules.append(nn.Linear(last_dim, hidden_dim))
        modules.append(activation())
        last_dim = hidden_dim
    modules.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*modules)


def gaussian_stats(raw: torch.Tensor, min_std: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    mean, raw_std = torch.chunk(raw, 2, dim=-1)
    std = F.softplus(raw_std) + min_std
    return mean, std


def sample_normal(mean: torch.Tensor, std: torch.Tensor, sample: bool = True) -> torch.Tensor:
    if not sample:
        return mean
    return mean + std * torch.randn_like(std)


def gaussian_kl(
    mean_q: torch.Tensor,
    std_q: torch.Tensor,
    mean_p: torch.Tensor,
    std_p: torch.Tensor,
) -> torch.Tensor:
    """KL[N(q)||N(p)] summed over the stochastic dimension."""
    var_q = std_q.square()
    var_p = std_p.square()
    kl = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p).square()) / (2.0 * var_p) - 0.5
    return kl.sum(dim=-1)


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def detach(self) -> "RSSMState":
        return RSSMState(
            deter=self.deter.detach(),
            stoch=self.stoch.detach(),
            mean=self.mean.detach(),
            std=self.std.detach(),
        )


class RSSM(nn.Module):
    """A minimal recurrent state-space model with Gaussian stochastic state."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        embed_dim: int,
        deter_dim: int,
        stoch_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.embed_dim = int(embed_dim)
        self.deter_dim = int(deter_dim)
        self.stoch_dim = int(stoch_dim)

        self.encoder = mlp(obs_dim, hidden_dim, embed_dim, layers=2)
        self.gru = nn.GRUCell(stoch_dim + action_dim, deter_dim)
        self.prior = mlp(deter_dim, hidden_dim, 2 * stoch_dim, layers=1)
        self.posterior = mlp(deter_dim + embed_dim, hidden_dim, 2 * stoch_dim, layers=1)

    @property
    def feat_dim(self) -> int:
        return self.deter_dim + self.stoch_dim

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
        mean = torch.zeros(batch_size, self.stoch_dim, device=device)
        std = torch.ones(batch_size, self.stoch_dim, device=device)
        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    def get_feat(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, state.stoch], dim=-1)

    def img_step(self, prev_state: RSSMState, action: torch.Tensor, *, sample: bool = True) -> RSSMState:
        x = torch.cat([prev_state.stoch, action], dim=-1)
        deter = self.gru(x, prev_state.deter)
        mean, std = gaussian_stats(self.prior(deter))
        stoch = sample_normal(mean, std, sample=sample)
        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    def obs_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        embed: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[RSSMState, RSSMState]:
        prior = self.img_step(prev_state, action, sample=sample)
        post_mean, post_std = gaussian_stats(self.posterior(torch.cat([prior.deter, embed], dim=-1)))
        post_stoch = sample_normal(post_mean, post_std, sample=sample)
        posterior = RSSMState(deter=prior.deter, stoch=post_stoch, mean=post_mean, std=post_std)
        return prior, posterior

    def observe(self, obs: torch.Tensor, actions: torch.Tensor, *, sample: bool = True) -> dict[str, torch.Tensor]:
        """Infer posterior states for a batch of sequences.

        Args:
            obs: `[B, L + 1, obs_dim]`
            actions: `[B, L, action_dim]`
        """
        batch_size, seq_plus_one, _ = obs.shape
        seq_len = seq_plus_one - 1
        embeds = self.encoder(obs[:, 1:].reshape(batch_size * seq_len, self.obs_dim))
        embeds = embeds.reshape(batch_size, seq_len, self.embed_dim)

        state = self.initial(batch_size, obs.device)
        priors: list[RSSMState] = []
        posts: list[RSSMState] = []
        prior_feats: list[torch.Tensor] = []
        post_feats: list[torch.Tensor] = []

        for t in range(seq_len):
            prior, state = self.obs_step(state, actions[:, t], embeds[:, t], sample=sample)
            priors.append(prior)
            posts.append(state)
            prior_feats.append(self.get_feat(prior))
            post_feats.append(self.get_feat(state))

        def stack_state(states: list[RSSMState], attr: str) -> torch.Tensor:
            return torch.stack([getattr(s, attr) for s in states], dim=1)

        return {
            "prior_mean": stack_state(priors, "mean"),
            "prior_std": stack_state(priors, "std"),
            "prior_stoch": stack_state(priors, "stoch"),
            "prior_deter": stack_state(priors, "deter"),
            "post_mean": stack_state(posts, "mean"),
            "post_std": stack_state(posts, "std"),
            "post_stoch": stack_state(posts, "stoch"),
            "post_deter": stack_state(posts, "deter"),
            "prior_feat": torch.stack(prior_feats, dim=1),
            "post_feat": torch.stack(post_feats, dim=1),
            "last_state": posts[-1],
        }


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: dict[str, Any]):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.rssm = RSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            embed_dim=int(cfg["embed_dim"]),
            deter_dim=int(cfg["deter_dim"]),
            stoch_dim=int(cfg["stoch_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
        )
        feat_dim = self.rssm.feat_dim
        hidden_dim = int(cfg["hidden_dim"])
        self.decoder = mlp(feat_dim, hidden_dim, obs_dim, layers=2)
        self.reward_head = mlp(feat_dim, hidden_dim, 1, layers=2)
        self.continue_head = mlp(feat_dim, hidden_dim, 1, layers=2)

    def loss(self, batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        continues = 1.0 - dones

        rssm_out = self.rssm.observe(obs, actions, sample=True)
        post_feat = rssm_out["post_feat"]
        pred_obs = self.decoder(post_feat)
        pred_reward = self.reward_head(post_feat).squeeze(-1)
        pred_continue_logits = self.continue_head(post_feat).squeeze(-1)

        obs_loss = F.mse_loss(pred_obs, obs[:, 1:])
        reward_loss = F.mse_loss(pred_reward, rewards)
        continue_loss = F.binary_cross_entropy_with_logits(pred_continue_logits, continues)

        dyn_kl = gaussian_kl(
            rssm_out["post_mean"].detach(),
            rssm_out["post_std"].detach(),
            rssm_out["prior_mean"],
            rssm_out["prior_std"],
        )
        rep_kl = gaussian_kl(
            rssm_out["post_mean"],
            rssm_out["post_std"],
            rssm_out["prior_mean"].detach(),
            rssm_out["prior_std"].detach(),
        )
        free_nats = float(cfg["free_nats"])
        dyn_loss = torch.clamp(dyn_kl, min=free_nats).mean()
        rep_loss = torch.clamp(rep_kl, min=free_nats).mean()

        total = (
            float(cfg["obs_loss_scale"]) * obs_loss
            + float(cfg["reward_loss_scale"]) * reward_loss
            + float(cfg["continue_loss_scale"]) * continue_loss
            + float(cfg["kl_scale"]) * dyn_loss
            + float(cfg["rep_kl_scale"]) * rep_loss
        )
        metrics = {
            "wm/loss": float(total.detach().cpu()),
            "wm/obs_loss": float(obs_loss.detach().cpu()),
            "wm/reward_loss": float(reward_loss.detach().cpu()),
            "wm/continue_loss": float(continue_loss.detach().cpu()),
            "wm/dyn_kl": float(dyn_kl.mean().detach().cpu()),
            "wm/rep_kl": float(rep_kl.mean().detach().cpu()),
        }
        return total, metrics, rssm_out

    def predict_prior(self, obs: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict next observations and rewards from prior features."""
        rssm_out = self.rssm.observe(obs, actions, sample=False)
        prior_feat = rssm_out["prior_feat"]
        return {
            "pred_obs": self.decoder(prior_feat),
            "pred_reward": self.reward_head(prior_feat).squeeze(-1),
            "rssm_out": rssm_out,
        }


class Actor(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, hidden_dim: int, min_std: float = 0.1):
        super().__init__()
        self.net = mlp(feat_dim, hidden_dim, action_dim, layers=2)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.min_std = float(min_std)

    def _dist(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(feat)
        std = F.softplus(self.log_std).expand_as(mean) + self.min_std
        return mean, std

    def sample(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self._dist(feat)
        normal = torch.distributions.Normal(mean, std)
        raw = normal.rsample()
        action = torch.tanh(raw)
        log_prob = normal.log_prob(raw) - torch.log(1.0 - action.square() + 1e-6)
        entropy = normal.entropy().sum(dim=-1)
        return action, log_prob.sum(dim=-1), entropy

    def mode(self, feat: torch.Tensor) -> torch.Tensor:
        mean, _ = self._dist(feat)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(feat_dim, hidden_dim, 1, layers=2)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)

