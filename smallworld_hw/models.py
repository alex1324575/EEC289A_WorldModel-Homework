"""Reward-free RSSM model used by the SmallWorld-Lite benchmark."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from world_model_hw.models import RSSM, RSSMState, gaussian_kl, mlp


class SmallWorldRSSM(nn.Module):
    """RSSM dynamics model with only a state decoder.

    Unlike the Pendulum MiniDreamer agent, this model has no reward head, no
    continuation head, and no actor-critic. It is trained purely to predict
    future fully observable states from state/action sequences.
    """

    def __init__(self, state_dim: int, action_dim: int, cfg: dict[str, Any]):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.rssm = RSSM(
            obs_dim=self.state_dim,
            action_dim=self.action_dim,
            embed_dim=int(cfg["embed_dim"]),
            deter_dim=int(cfg["deter_dim"]),
            stoch_dim=int(cfg["stoch_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
        )
        self.decoder = mlp(self.rssm.feat_dim, int(cfg["hidden_dim"]), self.state_dim, layers=2)

    def loss(self, batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        states = batch["states"]
        actions = batch["actions"]
        rssm_out = self.rssm.observe(states, actions, sample=True)
        post_feat = rssm_out["post_feat"]
        prior_feat = rssm_out["prior_feat"]
        target = states[:, 1:]

        pred_post = self.decoder(post_feat)
        pred_prior = self.decoder(prior_feat)
        state_loss = F.mse_loss(pred_post, target)
        prior_state_loss = F.mse_loss(pred_prior, target)

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
            float(cfg["state_loss_scale"]) * state_loss
            + float(cfg["prior_state_loss_scale"]) * prior_state_loss
            + float(cfg["kl_scale"]) * dyn_loss
            + float(cfg["rep_kl_scale"]) * rep_loss
        )
        metrics = {
            "sw/loss": float(total.detach().cpu()),
            "sw/state_loss": float(state_loss.detach().cpu()),
            "sw/prior_state_loss": float(prior_state_loss.detach().cpu()),
            "sw/dyn_kl": float(dyn_kl.mean().detach().cpu()),
            "sw/rep_kl": float(rep_kl.mean().detach().cpu()),
        }
        return total, metrics

    @torch.no_grad()
    def one_step_prior(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pred = self.rssm.observe(states, actions, sample=False)
        return self.decoder(pred["prior_feat"])

    @torch.no_grad()
    def observe_prefix(self, states: torch.Tensor, actions: torch.Tensor) -> RSSMState:
        out = self.rssm.observe(states, actions, sample=False)
        return out["last_state"]

    @torch.no_grad()
    def open_loop(
        self,
        prefix_states: torch.Tensor,
        warmup_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict future states after a ground-truth warm-up prefix.

        Args:
            prefix_states: `[B, W + 1, state_dim]`
            warmup_actions: `[B, W, action_dim]`
            future_actions: `[B, H, action_dim]`
        """
        batch_size = int(prefix_states.shape[0])
        if prefix_states.shape[1] <= 1:
            state = self.rssm.initial(batch_size, prefix_states.device)
        else:
            state = self.observe_prefix(prefix_states, warmup_actions)
        preds = []
        current = state
        for t in range(future_actions.shape[1]):
            current = self.rssm.img_step(current, future_actions[:, t], sample=False)
            feat = self.rssm.get_feat(current)
            preds.append(self.decoder(feat))
        return torch.stack(preds, dim=1)


def tensor_batch(batch: dict[str, Any], device: torch.device | str) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in batch.items()}
