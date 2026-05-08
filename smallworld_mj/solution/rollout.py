"""Reference rollout implementations."""

from __future__ import annotations

import torch


def teacher_forced_rollout(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    states = batch["states"]
    actions = batch["actions"]
    params = batch["task_params"]
    task_id = batch["task_id"]
    state_mask = batch["state_mask"][:, None, :]
    hidden = model.initial_hidden(states.shape[0], states.device) if hasattr(model, "initial_hidden") else None
    preds = []
    for t in range(actions.shape[1]):
        pred, hidden = model.step(states[:, t], actions[:, t], params, task_id, hidden)
        preds.append(pred * state_mask[:, 0])
    return torch.stack(preds, dim=1)


def open_loop_rollout(model, batch: dict[str, torch.Tensor], warmup: int = 10, horizon: int = 90) -> torch.Tensor:
    states = batch["states"]
    actions = batch["actions"]
    params = batch["task_params"]
    task_id = batch["task_id"]
    state_mask = batch["state_mask"]
    max_horizon = min(int(horizon), actions.shape[1] - int(warmup))
    if max_horizon <= 0:
        raise ValueError("warmup + horizon must fit inside the action sequence.")
    hidden = model.initial_hidden(states.shape[0], states.device) if hasattr(model, "initial_hidden") else None
    for t in range(int(warmup)):
        _, hidden = model.step(states[:, t], actions[:, t], params, task_id, hidden)
    cur = states[:, int(warmup)]
    preds = []
    for k in range(max_horizon):
        pred, hidden = model.step(cur, actions[:, int(warmup) + k], params, task_id, hidden)
        pred = pred * state_mask
        preds.append(pred)
        cur = pred
    return torch.stack(preds, dim=1)
