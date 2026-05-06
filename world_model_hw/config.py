"""Config, JSON, and reproducibility helpers for the MiniDreamer homework."""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "course_config.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
        return to_jsonable(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def deep_update(base: dict[str, Any], updates: dict[str, Any] | None) -> dict[str, Any]:
    """Return a deep-updated copy without mutating the original mapping."""
    result = copy.deepcopy(base)
    if not updates:
        return result
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_stage_overrides(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    """Apply the optional per-stage replay/model/actor overrides."""
    resolved = copy.deepcopy(config)
    stage_cfg = resolved["training_stages"][stage_name]
    resolved["active_stage"] = stage_name
    if "replay_overrides" in stage_cfg:
        resolved["replay"] = deep_update(resolved["replay"], stage_cfg["replay_overrides"])
    if "model_overrides" in stage_cfg:
        resolved["world_model"] = deep_update(resolved["world_model"], stage_cfg["model_overrides"])
    if "actor_critic_overrides" in stage_cfg:
        resolved["actor_critic"] = deep_update(resolved["actor_critic"], stage_cfg["actor_critic_overrides"])
    return resolved


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def choose_device(device_arg: str | None = None) -> str:
    if device_arg:
        return device_arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def set_runtime_env() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def summarize_config(config: dict[str, Any]) -> dict[str, Any]:
    stage_name = config.get("active_stage", "baseline")
    return {
        "homework_name": config["homework_name"],
        "algorithm": config["algorithm"],
        "env": config["env"],
        "stage": stage_name,
        "stage_config": config["training_stages"][stage_name],
        "replay": config["replay"],
        "world_model": config["world_model"],
        "actor_critic": config["actor_critic"],
    }

