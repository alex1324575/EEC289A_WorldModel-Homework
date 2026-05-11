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
from .config import load_config, save_json
from .dataset import load_split
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
) -> dict[str, Any]:
    model.eval()
    states_np = data["states"][:max_windows]
    actions_np = data["actions"][:max_windows]
    states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
    warmup = int(cfg["eval"]["warmup_steps"])
    horizon = int(cfg["eval"]["horizon"])
    preds = open_loop_rollout(model, states, actions, normalizer, warmup_steps=warmup, horizon=horizon)
    targets = states[:, warmup + 1 : warmup + 1 + horizon]
    survival, metrics = compute_failure_horizon(preds, targets, cfg["failure"])
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2, dim=(0, 2))).detach().cpu().numpy()
    metrics.update(
        {
            "one_step_rmse": _one_step_rmse(model, states, actions, normalizer),
            "open_loop_rmse@100": float(torch.sqrt(torch.mean((preds - targets) ** 2)).detach().cpu()),
            "horizon_rmse": rmse.tolist(),
        }
    )
    model.train()
    metrics["survival_steps"] = survival.detach().cpu().numpy().astype(np.int32)
    return metrics


def evaluate_checkpoint(checkpoint_dir: str | Path, dataset_dir: str | Path, split: str, output_dir: str | Path) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint_dir, device=device)
    data = load_split(dataset_dir, split)
    normalizer = Normalizer.from_dict(payload["normalizer"])
    metrics = evaluate_model_on_split(model, data, normalizer, payload["config"], device=device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    survival = np.asarray(metrics.pop("survival_steps"), dtype=np.int32)
    save_json(output_dir / "metrics.json", metrics)
    np.save(output_dir / "per_window_survival.npy", survival)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint_dir, args.dataset_dir, args.split, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
