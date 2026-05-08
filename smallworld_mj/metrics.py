"""Official SmallWorld-MJ evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from smallworld_mj.data import Normalizer
from smallworld_mj.solution.physics_metrics import physical_metrics
from smallworld_mj.solution.rollout import open_loop_rollout, teacher_forced_rollout


def normalized_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, normalizer: Normalizer) -> float:
    pred_n = (pred - normalizer.mean) / (normalizer.std + 1e-6)
    target_n = (target - normalizer.mean) / (normalizer.std + 1e-6)
    while mask.ndim < pred_n.ndim:
        mask = np.expand_dims(mask, 1)
    err = ((pred_n - target_n) ** 2) * mask
    return float(np.sqrt(err.sum() / max(float(mask.sum()) * pred_n.shape[1], 1.0)))


def horizon_curve(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    pred_n = (pred - normalizer.mean) / (normalizer.std + 1e-6)
    target_n = (target - normalizer.mean) / (normalizer.std + 1e-6)
    err = (pred_n - target_n) ** 2
    mask_h = mask[:, None, :]
    return np.sqrt((err * mask_h).sum(axis=(0, 2)) / np.maximum(mask_h.sum(axis=(0, 2)), 1.0)).astype(np.float32)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    normalizer: Normalizer,
    *,
    warmup: int,
    horizon: int,
    device: str | torch.device,
    batch_size: int = 64,
) -> dict[str, Any]:
    device = torch.device(device)
    model.eval()
    one_preds = []
    rollout_preds = []
    rollout_targets = []
    phys_values: list[dict[str, float]] = []
    for start in range(0, len(data["task_id"]), batch_size):
        end = min(start + batch_size, len(data["task_id"]))
        batch = {
            "states": torch.as_tensor(data["states"][start:end], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(data["actions"][start:end], dtype=torch.float32, device=device),
            "task_params": torch.as_tensor(data["task_params"][start:end], dtype=torch.float32, device=device),
            "state_mask": torch.as_tensor(data["state_mask"][start:end], dtype=torch.float32, device=device),
            "action_mask": torch.as_tensor(data["action_mask"][start:end], dtype=torch.float32, device=device),
            "param_mask": torch.as_tensor(data["param_mask"][start:end], dtype=torch.float32, device=device),
            "task_id": torch.as_tensor(data["task_id"][start:end], dtype=torch.long, device=device),
        }
        one = teacher_forced_rollout(model, batch)
        pred = open_loop_rollout(model, batch, warmup=warmup, horizon=horizon)
        target = batch["states"][:, warmup + 1 : warmup + 1 + pred.shape[1]]
        one_preds.append(one.cpu().numpy())
        rollout_preds.append(pred.cpu().numpy())
        rollout_targets.append(target.cpu().numpy())
        phys_values.append(physical_metrics(pred, target, batch))
    one_arr = np.concatenate(one_preds, axis=0)
    pred_arr = np.concatenate(rollout_preds, axis=0)
    target_arr = np.concatenate(rollout_targets, axis=0)
    one_target = data["states"][:, 1 : 1 + one_arr.shape[1]]
    masks = data["state_mask"].astype(np.float32)
    curve = horizon_curve(pred_arr, target_arr, masks, normalizer)
    phys = {key: float(np.mean([p[key] for p in phys_values])) for key in phys_values[0]}
    return {
        "one_step_nrmse": normalized_rmse(one_arr, one_target, masks, normalizer),
        "open_loop_15_nrmse": normalized_rmse(pred_arr[:, :15], target_arr[:, :15], masks, normalizer),
        "open_loop_90_nrmse": normalized_rmse(pred_arr, target_arr, masks, normalizer),
        "horizon_error_auc": float(curve.mean()),
        "horizon_rmse": curve.tolist(),
        "physical_metrics": phys,
        "rollout_pred": pred_arr,
        "rollout_target": target_arr,
    }
