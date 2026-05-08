"""Checkpoint helpers for SmallWorld-Lite models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from world_model_hw.config import save_json

from .models import SmallWorldRSSM
from .tasks import get_task


CHECKPOINT_NAME = "checkpoint.pt"
MANIFEST_NAME = "manifest.json"


def save_smallworld_checkpoint(
    checkpoint_dir: Path,
    *,
    model: SmallWorldRSSM,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    task_name: str,
    metrics: dict[str, Any],
    update: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    task = get_task(task_name)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "task_name": task_name,
        "state_dim": task.state_dim,
        "action_dim": task.action_dim,
        "metrics": metrics,
        "update": int(update),
    }
    torch.save(payload, checkpoint_dir / CHECKPOINT_NAME)
    save_json(
        checkpoint_dir / MANIFEST_NAME,
        {
            "checkpoint_file": CHECKPOINT_NAME,
            "task_name": task_name,
            "update": int(update),
            "state_dim": task.state_dim,
            "action_dim": task.action_dim,
            "metrics": metrics,
        },
    )


def load_smallworld_checkpoint(checkpoint_dir: Path, device: str) -> tuple[SmallWorldRSSM, dict[str, Any], dict[str, Any]]:
    path = checkpoint_dir / CHECKPOINT_NAME
    if not path.exists():
        raise FileNotFoundError(f"SmallWorld checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    task = get_task(str(payload["task_name"]))
    model = SmallWorldRSSM(task.state_dim, task.action_dim, payload["config"]["world_model"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload["config"], payload
