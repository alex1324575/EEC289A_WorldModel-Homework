"""Evaluation metrics for SmallWorld-Lite dynamics prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .models import SmallWorldRSSM
from .tasks import SmallWorldTask


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


@torch.no_grad()
def predict_episode_rollouts(
    model: SmallWorldRSSM,
    data: dict[str, np.ndarray],
    *,
    warmup_steps: int,
    horizon: int,
    device: str | torch.device,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["action"], dtype=np.float32)
    episodes, steps_plus_one, state_dim = states.shape
    steps = steps_plus_one - 1
    if warmup_steps + horizon > steps:
        horizon = max(1, steps - warmup_steps)
    if horizon <= 0:
        raise ValueError(f"Need more than warmup_steps={warmup_steps} steps for evaluation.")

    one_step_preds = []
    open_loop_preds = []
    open_loop_targets = []
    model.eval()
    for start in range(0, episodes, batch_size):
        end = min(start + batch_size, episodes)
        states_t = torch.as_tensor(states[start:end], dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions[start:end], dtype=torch.float32, device=device)
        one_step = model.one_step_prior(states_t, actions_t).detach().cpu().numpy()
        one_step_preds.append(one_step)

        prefix = states_t[:, : warmup_steps + 1]
        warm_actions = actions_t[:, :warmup_steps]
        future_actions = actions_t[:, warmup_steps : warmup_steps + horizon]
        pred = model.open_loop(prefix, warm_actions, future_actions).detach().cpu().numpy()
        target = states[start:end, warmup_steps + 1 : warmup_steps + 1 + horizon]
        open_loop_preds.append(pred)
        open_loop_targets.append(target)

    one_step_arr = np.concatenate(one_step_preds, axis=0)
    open_loop_arr = np.concatenate(open_loop_preds, axis=0)
    open_loop_target = np.concatenate(open_loop_targets, axis=0)
    return {
        "one_step_pred": one_step_arr.astype(np.float32),
        "one_step_target": states[:, 1:].astype(np.float32),
        "open_loop_pred": open_loop_arr.astype(np.float32),
        "open_loop_target": open_loop_target.astype(np.float32),
        "effective_horizon": np.asarray(open_loop_arr.shape[1], dtype=np.int32),
        "state_dim": np.asarray(state_dim, dtype=np.int32),
    }


def horizon_rmse_curve(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    errors = (np.asarray(pred, dtype=np.float32) - np.asarray(target, dtype=np.float32)) ** 2
    return np.sqrt(np.mean(errors, axis=(0, 2))).astype(np.float32)


def _rmse_first_h(pred: np.ndarray, target: np.ndarray, horizon: int) -> float:
    h = min(int(horizon), pred.shape[1])
    return rmse(pred[:, :h], target[:, :h])


def physical_metrics(task: SmallWorldTask, pred: np.ndarray, target: np.ndarray, params: np.ndarray) -> dict[str, float]:
    energy_errors = []
    constraints = []
    for ep in range(pred.shape[0]):
        for t in range(pred.shape[1]):
            if task.energy_fn is not None:
                energy_errors.append(abs(task.energy(pred[ep, t], params[ep]) - task.energy(target[ep, t], params[ep])))
            constraints.append(abs(task.constraint(pred[ep, t], params[ep])))
    return {
        "energy_drift": float(np.mean(energy_errors)) if energy_errors else float("nan"),
        "constraint_violation": float(np.mean(constraints)) if constraints else 0.0,
    }


def compute_smallworld_metrics(
    task: SmallWorldTask,
    rollout: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
) -> dict[str, Any]:
    pred = rollout["open_loop_pred"]
    target = rollout["open_loop_target"]
    curve = horizon_rmse_curve(pred, target)
    metrics = {
        "one_step_state_rmse": rmse(rollout["one_step_pred"], rollout["one_step_target"]),
        "open_loop_15_rmse": _rmse_first_h(pred, target, 15),
        "open_loop_90_rmse": _rmse_first_h(pred, target, 90),
        "horizon_error_auc": float(np.mean(curve)),
        "effective_horizon": int(rollout["effective_horizon"]),
        "horizon_rmse": curve.tolist(),
    }
    metrics.update(physical_metrics(task, pred, target, np.asarray(data["task_params"], dtype=np.float32)))
    return metrics


def lower_better_score(value: float, good: float, bad: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip((bad - value) / (bad - good), 0.0, 1.0))


def composite_score(metrics: dict[str, float], metric_cfg: dict[str, Any]) -> tuple[dict[str, float], float]:
    normalized = {}
    weighted = 0.0
    total = 0.0
    for name, cfg in metric_cfg.items():
        value = float(metrics.get(name, float("nan")))
        score = lower_better_score(value, float(cfg["good"]), float(cfg["bad"]))
        normalized[name] = score
        weight = float(cfg["weight"])
        weighted += weight * score
        total += weight
    return normalized, float(weighted / total) if total else float("nan")
