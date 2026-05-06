"""Checkpoint helpers for MiniDreamer homework scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .agent import MiniDreamerAgent
from .config import save_json


CHECKPOINT_NAME = "checkpoint.pt"
MANIFEST_NAME = "manifest.json"


def save_checkpoint(
    checkpoint_dir: Path,
    *,
    agent: MiniDreamerAgent,
    config: dict[str, Any],
    obs_dim: int,
    action_dim: int,
    metrics: dict[str, Any],
    step: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent": agent.state_dict(),
        "config": config,
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "metrics": metrics,
        "step": int(step),
    }
    torch.save(payload, checkpoint_dir / CHECKPOINT_NAME)
    save_json(
        checkpoint_dir / MANIFEST_NAME,
        {
            "checkpoint_file": CHECKPOINT_NAME,
            "step": int(step),
            "metrics": metrics,
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
        },
    )


def load_checkpoint(checkpoint_dir: Path, device: str) -> tuple[MiniDreamerAgent, dict[str, Any], dict[str, Any]]:
    checkpoint_path = checkpoint_dir / CHECKPOINT_NAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    config = payload["config"]
    agent = MiniDreamerAgent(
        obs_dim=int(payload["obs_dim"]),
        action_dim=int(payload["action_dim"]),
        config=config,
        device=device,
    )
    agent.load_state_dict(payload["agent"])
    agent.world_model.eval()
    agent.actor.eval()
    agent.critic.eval()
    return agent, config, payload

