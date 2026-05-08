"""Evaluation entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from smallworld_mj.checkpoint import load_checkpoint
from smallworld_mj.config import save_json
from smallworld_mj.data import Normalizer, load_split
from smallworld_mj.metrics import evaluate_model


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    dataset_dir: str | Path,
    split: str,
    output_dir: str | Path,
    *,
    warmup: int,
    horizon: int,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint_dir, device=device)
    normalizer = Normalizer.from_dict(payload["normalizer"])
    data = load_split(dataset_dir, split)
    result = evaluate_model(model, data, normalizer, warmup=warmup, horizon=horizon, device=device)
    metrics = {k: v for k, v in result.items() if k not in {"rollout_pred", "rollout_target"}}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / f"metrics_{split}.json", metrics)
    save_json(output_dir / "physics_metrics.json", metrics["physical_metrics"])
    print(save_json(output_dir / "eval_summary.json", {"split": split, "checkpoint": str(checkpoint_dir), "metrics": metrics}))
    return metrics
