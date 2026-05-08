"""Reference physical diagnostics for SmallWorld-MJ states."""

from __future__ import annotations

import torch


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def energy_drift(pred: torch.Tensor, target: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
    del target, task_id
    vel = pred[..., 3:6] if pred.shape[-1] >= 6 else pred[..., -1:]
    energy = 0.5 * (vel**2).sum(dim=-1)
    return (energy - energy[:, :1]).abs().mean()


def radius_violation(pred: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
    circle = task_id == 2
    if not torch.any(circle):
        return pred.new_tensor(0.0)
    s = pred[circle]
    radius = torch.sqrt(s[..., 0] ** 2 + s[..., 1] ** 2)
    return (radius - 1.0).abs().mean()


def box_violation(pred: torch.Tensor, task_id: torch.Tensor, task_params: torch.Tensor) -> torch.Tensor:
    bounce = task_id == 3
    if not torch.any(bounce):
        return pred.new_tensor(0.0)
    s = pred[bounce]
    params = task_params[bounce]
    half = params[:, 1].view(-1, 1)
    violation = torch.relu(s[..., :2].abs() - half[..., None])
    return violation.mean()


def no_slip_violation(pred: torch.Tensor, task_id: torch.Tensor, task_params: torch.Tensor) -> torch.Tensor:
    rolling = task_id == 4
    if not torch.any(rolling):
        return pred.new_tensor(0.0)
    s = pred[rolling]
    params = task_params[rolling]
    radius = params[:, 0].view(-1, 1)
    return (s[..., 1] - radius * s[..., 4]).abs().mean()


def phase_drift(pred: torch.Tensor, target: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
    periodic = (task_id == 0) | (task_id == 2) | (task_id == 8)
    if not torch.any(periodic):
        return pred.new_tensor(0.0)
    p = pred[periodic][..., :2]
    t = target[periodic][..., :2]
    return (p - t).pow(2).sum(dim=-1).sqrt().mean()


def physical_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    as_loss: bool = False,
) -> dict[str, torch.Tensor | float]:
    task_id = batch["task_id"].long()
    params = batch["task_params"]
    metrics = {
        "energy_drift": energy_drift(pred, target, task_id),
        "radius_violation": radius_violation(pred, task_id),
        "box_violation": box_violation(pred, task_id, params),
        "no_slip_violation": no_slip_violation(pred, task_id, params),
        "phase_drift": phase_drift(pred, target, task_id),
    }
    if as_loss:
        return metrics
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}
