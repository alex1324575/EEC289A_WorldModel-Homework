"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from smallworld_mj.models.baseline_mlp import BaselineMLP
from smallworld_mj.solution.model import ParamResidualGRU
from smallworld_mj.student.models.student_model import StudentWorldModel


def build_model(model_name: str, dims: tuple[int, int, int], num_tasks: int, cfg: dict[str, Any]) -> torch.nn.Module:
    state_dim, action_dim, param_dim = dims
    model_cfg = cfg.get("model", cfg)
    if model_name == "baseline_mlp":
        return BaselineMLP(state_dim, action_dim, param_dim, num_tasks, model_cfg)
    if model_name == "solution":
        return ParamResidualGRU(state_dim, action_dim, param_dim, num_tasks, model_cfg)
    if model_name == "student":
        return StudentWorldModel(state_dim, action_dim, param_dim, num_tasks, model_cfg)
    raise KeyError(f"Unknown model '{model_name}'.")


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    model_name: str,
    dims: tuple[int, int, int],
    num_tasks: int,
    normalizer: dict[str, Any],
    step: int,
    metrics: dict[str, float],
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": config,
        "model_name": model_name,
        "dims": dims,
        "num_tasks": num_tasks,
        "normalizer": normalizer,
        "step": int(step),
        "metrics": metrics,
    }
    path = checkpoint_dir / "checkpoint.pt"
    torch.save(payload, path)
    return path


def load_checkpoint(checkpoint_dir: str | Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    path = Path(checkpoint_dir) / "checkpoint.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(payload["model_name"], tuple(payload["dims"]), int(payload["num_tasks"]), payload["config"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload
