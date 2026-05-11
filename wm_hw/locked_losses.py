"""Locked losses for staff-owned baseline training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def locked_one_step_delta_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    states = batch["states"]
    actions = batch["actions"]
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    pred_delta_norm, _ = model(normalizer.normalize_obs(obs), normalizer.normalize_act(act), None)
    loss = F.mse_loss(pred_delta_norm, normalizer.normalize_delta(target_delta))
    return loss, {
        "loss/total": float(loss.detach().cpu()),
        "loss/one_step": float(loss.detach().cpu()),
        "loss/rollout": 0.0,
    }
