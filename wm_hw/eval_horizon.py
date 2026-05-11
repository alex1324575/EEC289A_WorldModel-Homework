"""Official horizon-to-failure evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

import numpy as np
import torch

from student.rollout import open_loop_rollout
from student.metrics import compute_failure_horizon
from .checkpoint import load_checkpoint
from .config import save_json
from .dataset import load_split
from .horizon import resolve_eval_horizon, resolve_milestones
from .model_utils import predict_next
from .normalizer import Normalizer


@torch.no_grad()
def _one_step_rmse(model, states: torch.Tensor, actions: torch.Tensor, normalizer: Normalizer) -> float:
    hidden = model.initial_hidden(states.shape[0], states.device)
    preds = []
    for t in range(actions.shape[1]):
        pred, hidden = predict_next(model, states[:, t], actions[:, t], hidden, normalizer)
        preds.append(pred)
    pred_t = torch.stack(preds, dim=1)
    return float(torch.sqrt(torch.mean((pred_t - states[:, 1:]) ** 2)).detach().cpu())


@torch.no_grad()
def evaluate_model_on_split(
    model,
    data: dict[str, np.ndarray],
    normalizer: Normalizer,
    cfg: dict[str, Any],
    *,
    device: torch.device,
    max_windows: int | None = None,
    warmup_steps: int | None = None,
    horizon: str | int | None = None,
    milestones: str | list[int] | None = None,
) -> dict[str, Any]:
    model.eval()
    states_np = data["states"][:max_windows]
    actions_np = data["actions"][:max_windows]
    states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
    warmup, horizon = resolve_eval_horizon(
        states_shape=tuple(states.shape),
        actions_shape=tuple(actions.shape),
        cfg=cfg,
        warmup_override=warmup_steps,
        horizon_override=horizon,
    )
    milestones = milestones if milestones is not None else cfg.get("eval", {}).get("milestones")
    milestone_values = resolve_milestones(milestones, horizon)
    preds = open_loop_rollout(model, states, actions, normalizer, warmup_steps=warmup, horizon=horizon)
    targets = states[:, warmup + 1 : warmup + 1 + horizon]
    survival, metrics = compute_failure_horizon(preds, targets, cfg["failure"], milestones=milestone_values)
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2, dim=(0, 2))).detach().cpu().numpy()
    metrics.update(
        {
            "warmup_steps": warmup,
            "max_horizon": horizon,
            "one_step_rmse": _one_step_rmse(model, states, actions, normalizer),
            "open_loop_rmse@horizon": float(torch.sqrt(torch.mean((preds - targets) ** 2)).detach().cpu()),
            "horizon_rmse": rmse.tolist(),
        }
    )
    model.train()
    metrics["survival_steps"] = survival.detach().cpu().numpy().astype(np.int32)
    return metrics


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    dataset_dir: str | Path,
    split: str,
    output_dir: str | Path,
    *,
    warmup_steps: int | None = None,
    horizon: str | int | None = "auto",
    milestones: str | list[int] | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint_dir, device=device)
    data = load_split(dataset_dir, split)
    normalizer = Normalizer.from_dict(payload["normalizer"])
    metrics = evaluate_model_on_split(
        model,
        data,
        normalizer,
        payload["config"],
        device=device,
        warmup_steps=warmup_steps,
        horizon=horizon,
        milestones=milestones,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    survival = np.asarray(metrics.pop("survival_steps"), dtype=np.int32)
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "scoreboard_summary.json", _scoreboard_summary(metrics, payload))
    np.save(output_dir / "per_window_survival.npy", survival)
    return metrics


def _scoreboard_summary(metrics: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    max_horizon = int(metrics["max_horizon"])
    return {
        "model_name": payload["model_name"],
        "checkpoint_step": int(payload["step"]),
        "max_horizon": max_horizon,
        "H80": int(metrics["H80"]),
        "H50": int(metrics["H50"]),
        "H80_fraction": float(metrics["H80"]) / max_horizon,
        "survival_auc": float(metrics["survival_auc"]),
        "mean_survival_steps": float(metrics["mean_survival_steps"]),
        "median_survival_steps": float(metrics["median_survival_steps"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--horizon", default="auto", help="Use an integer horizon or 'auto' for the dataset maximum.")
    parser.add_argument("--milestones", default=None, help="Comma-separated success-rate horizons, e.g. 5,10,50,100,500.")
    args = parser.parse_args()
    metrics = evaluate_checkpoint(
        args.checkpoint_dir,
        args.dataset_dir,
        args.split,
        args.output_dir,
        warmup_steps=args.warmup,
        horizon=args.horizon,
        milestones=args.milestones,
    )
    summary_keys = (
        "max_horizon",
        "H80",
        "H50",
        "survival_auc",
        "mean_survival_steps",
        "median_survival_steps",
        "one_step_rmse",
        "open_loop_rmse@horizon",
    )
    print(
        json.dumps(
            {
                "metrics_summary": {key: metrics[key] for key in summary_keys if key in metrics},
                "metrics_json": str(Path(args.output_dir) / "metrics.json"),
                "scoreboard_summary_json": str(Path(args.output_dir) / "scoreboard_summary.json"),
                "per_window_survival": str(Path(args.output_dir) / "per_window_survival.npy"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
