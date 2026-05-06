"""MiniDreamer training logic."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .models import Actor, Critic, RSSMState, WorldModel


def lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discounts: torch.Tensor,
    bootstrap: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Compute Dreamer-style lambda returns.

    Args:
        rewards: `[H, B]`
        values: `[H, B]`
        discounts: `[H, B]`
        bootstrap: `[B]`
    """
    next_return = bootstrap
    returns = []
    horizon = rewards.shape[0]
    for t in reversed(range(horizon)):
        next_value = bootstrap if t == horizon - 1 else values[t + 1]
        next_return = rewards[t] + discounts[t] * ((1.0 - lambda_) * next_value + lambda_ * next_return)
        returns.append(next_return)
    returns.reverse()
    return torch.stack(returns, dim=0)


@contextmanager
def frozen(module: nn.Module):
    old_flags = [param.requires_grad for param in module.parameters()]
    for param in module.parameters():
        param.requires_grad_(False)
    try:
        yield
    finally:
        for param, flag in zip(module.parameters(), old_flags):
            param.requires_grad_(flag)


class MiniDreamerAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: dict[str, Any], device: str):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.config = config
        self.device = torch.device(device)

        wm_cfg = config["world_model"]
        ac_cfg = config["actor_critic"]
        self.world_model = WorldModel(obs_dim, action_dim, wm_cfg).to(self.device)
        feat_dim = self.world_model.rssm.feat_dim
        self.actor = Actor(feat_dim, action_dim, int(ac_cfg["hidden_dim"]), min_std=float(ac_cfg["min_std"])).to(self.device)
        self.critic = Critic(feat_dim, int(ac_cfg["hidden_dim"])).to(self.device)
        self.target_critic = Critic(feat_dim, int(ac_cfg["hidden_dim"])).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for param in self.target_critic.parameters():
            param.requires_grad_(False)

        self.wm_opt = torch.optim.Adam(self.world_model.parameters(), lr=float(wm_cfg["learning_rate"]))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(ac_cfg["actor_learning_rate"]))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(ac_cfg["critic_learning_rate"]))
        self.update_count = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "wm_opt": self.wm_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.world_model.load_state_dict(state["world_model"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state.get("target_critic", state["critic"]))
        if "wm_opt" in state:
            self.wm_opt.load_state_dict(state["wm_opt"])
            self.actor_opt.load_state_dict(state["actor_opt"])
            self.critic_opt.load_state_dict(state["critic_opt"])
        self.update_count = int(state.get("update_count", 0))

    def _tensor_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in batch.items()
        }

    def train_world_model(self, batch: dict[str, Any]) -> dict[str, float]:
        tensors = self._tensor_batch(batch)
        loss, metrics, _ = self.world_model.loss(tensors, self.config["world_model"])
        self.wm_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), float(self.config["world_model"]["grad_clip"]))
        self.wm_opt.step()
        return metrics

    def _start_state_from_batch(self, tensors: dict[str, torch.Tensor]) -> RSSMState:
        with torch.no_grad():
            rssm_out = self.world_model.rssm.observe(tensors["obs"], tensors["actions"], sample=False)
            return rssm_out["last_state"].detach()

    def train_actor_critic(self, batch: dict[str, Any]) -> dict[str, float]:
        tensors = self._tensor_batch(batch)
        ac_cfg = self.config["actor_critic"]
        start_state = self._start_state_from_batch(tensors)

        with frozen(self.world_model):
            feats, rewards, discounts, entropies = self._imagine(start_state)
            flat_feats = feats.reshape(-1, feats.shape[-1])
            target_values = self.target_critic(flat_feats).reshape(feats.shape[0], feats.shape[1])
            bootstrap = target_values[-1]
            returns = lambda_return(
                rewards=rewards,
                values=target_values,
                discounts=discounts,
                bootstrap=bootstrap,
                lambda_=float(ac_cfg["lambda"]),
            )
            actor_loss = -returns.mean() - float(ac_cfg["entropy_scale"]) * entropies.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(ac_cfg["grad_clip"]))
        self.actor_opt.step()

        critic_values = self.critic(feats.detach().reshape(-1, feats.shape[-1])).reshape(feats.shape[0], feats.shape[1])
        critic_loss = F.mse_loss(critic_values, returns.detach())
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(ac_cfg["grad_clip"]))
        self.critic_opt.step()

        self.update_count += 1
        if self.update_count % int(ac_cfg["target_update_interval"]) == 0:
            self.soft_update_target()

        return {
            "ac/actor_loss": float(actor_loss.detach().cpu()),
            "ac/critic_loss": float(critic_loss.detach().cpu()),
            "ac/imagine_reward": float(rewards.mean().detach().cpu()),
            "ac/imagine_discount": float(discounts.mean().detach().cpu()),
            "ac/entropy": float(entropies.mean().detach().cpu()),
        }

    def _imagine(self, start_state: RSSMState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ac_cfg = self.config["actor_critic"]
        horizon = int(ac_cfg["imag_horizon"])
        discount = float(ac_cfg["discount"])
        state = start_state
        feats = []
        rewards = []
        discounts = []
        entropies = []
        for _ in range(horizon):
            feat = self.world_model.rssm.get_feat(state)
            action, _, entropy = self.actor.sample(feat)
            state = self.world_model.rssm.img_step(state, action, sample=True)
            next_feat = self.world_model.rssm.get_feat(state)
            reward = self.world_model.reward_head(next_feat).squeeze(-1)
            continue_prob = torch.sigmoid(self.world_model.continue_head(next_feat).squeeze(-1))
            feats.append(next_feat)
            rewards.append(reward)
            discounts.append(discount * continue_prob)
            entropies.append(entropy)
        return (
            torch.stack(feats, dim=0),
            torch.stack(rewards, dim=0),
            torch.stack(discounts, dim=0),
            torch.stack(entropies, dim=0),
        )

    def soft_update_target(self) -> None:
        tau = float(self.config["actor_critic"]["target_update_tau"])
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, prev_state: RSSMState | None = None, prev_action: torch.Tensor | None = None, deterministic: bool = True):
        """One-step policy for environment interaction.

        The compact homework agent uses the learned latent dynamics online. For
        the first action in an episode, the state is all zeros; after that, the
        state is updated with the previous action and current observation.
        """
        obs = obs.to(self.device).float().reshape(1, self.obs_dim)
        if prev_state is None:
            prev_state = self.world_model.rssm.initial(1, self.device)
        if prev_action is None:
            prev_action = torch.zeros(1, self.action_dim, device=self.device)
        embed = self.world_model.rssm.encoder(obs)
        _, state = self.world_model.rssm.obs_step(prev_state, prev_action, embed, sample=False)
        feat = self.world_model.rssm.get_feat(state)
        action = self.actor.mode(feat) if deterministic else self.actor.sample(feat)[0]
        return action.squeeze(0), state

