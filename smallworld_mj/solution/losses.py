"""Reference one-step, rollout, and mixed losses."""

from __future__ import annotations

import torch

from .physics_metrics import physical_metrics
from .rollout import open_loop_rollout, teacher_forced_rollout


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(1)
    err = (pred - target).pow(2) * mask
    return err.sum() / mask.sum().clamp_min(1.0) / pred.shape[1]


def compute_loss(model, batch: dict[str, torch.Tensor], cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    loss_cfg = cfg["loss"]
    state_mask = batch["state_mask"]
    one_step_pred = teacher_forced_rollout(model, batch)
    one_step_target = batch["states"][:, 1 : 1 + one_step_pred.shape[1]]
    one_step = masked_mse(one_step_pred, one_step_target, state_mask)

    rollout_h = int(loss_cfg.get("rollout_train_horizon", 15))
    warmup = int(loss_cfg.get("warmup", 10))
    rollout_pred = open_loop_rollout(model, batch, warmup=warmup, horizon=rollout_h)
    rollout_target = batch["states"][:, warmup + 1 : warmup + 1 + rollout_pred.shape[1]]
    rollout = masked_mse(rollout_pred, rollout_target, state_mask)

    phys_tensors = physical_metrics(rollout_pred, rollout_target, batch, as_loss=True)
    phys_loss = sum(phys_tensors.values())

    total = (
        float(loss_cfg.get("one_step_weight", 1.0)) * one_step
        + float(loss_cfg.get("rollout_weight", 0.3)) * rollout
        + float(loss_cfg.get("physics_weight", 0.0)) * phys_loss
    )
    metrics = {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one_step.detach().cpu()),
        "loss/rollout": float(rollout.detach().cpu()),
        "loss/physics": float(phys_loss.detach().cpu()),
    }
    metrics.update({f"phys/{k}": float(v.detach().cpu()) for k, v in phys_tensors.items()})
    return total, metrics
