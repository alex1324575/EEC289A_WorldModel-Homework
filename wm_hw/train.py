"""Locked training script for baseline/student/solution world models."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

import numpy as np
import torch

from student.losses import compute_loss
from .checkpoint import build_model, save_checkpoint
from .config import load_config, save_json, set_seed
from .dataset import load_split
from .eval_horizon import evaluate_model_on_split
from .normalizer import Normalizer


def _device(cfg: dict[str, Any]) -> torch.device:
    if cfg.get("device", "auto") == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch(data: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "states": torch.as_tensor(data["states"][indices], dtype=torch.float32, device=device),
        "actions": torch.as_tensor(data["actions"][indices], dtype=torch.float32, device=device),
    }


def train(config_path: str | Path, model_name: str, dataset_dir: str | Path, output_dir: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 0)))
    torch.set_num_threads(int(cfg.get("torch_num_threads", 1)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data = load_split(dataset_dir, "train")
    val_data = load_split(dataset_dir, "val")
    normalizer = Normalizer.from_train(train_data["states"], train_data["actions"])
    normalizer.save(output_dir / "normalizer.json")
    device = _device(cfg)
    model = build_model(model_name, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    updates = int(cfg["training"]["smoke_updates"] if smoke else cfg["training"]["updates"])
    batch_size = int(cfg["training"]["batch_size"])
    eval_every = int(cfg["training"]["smoke_eval_every"] if smoke else cfg["training"]["eval_every"])
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 17)
    best_h80 = -1.0
    best_metrics: dict[str, float] = {}
    print(f"[train] model={model_name} device={device} updates={updates} smoke={smoke}")
    for update in range(1, updates + 1):
        indices = rng.integers(0, len(train_data["states"]), size=batch_size)
        loss, metrics = compute_loss(model, _batch(train_data, indices, device), normalizer, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip_norm"]))
        opt.step()
        if update == 1 or update % eval_every == 0 or update == updates:
            eval_metrics = evaluate_model_on_split(
                model,
                val_data,
                normalizer,
                cfg,
                device=device,
                max_windows=min(128, len(val_data["states"])),
            )
            eval_metrics.pop("survival_steps", None)
            line = " ".join([f"{k}={v:.4f}" for k, v in {**metrics, **eval_metrics}.items() if isinstance(v, (float, int))])
            print(f"[train] update={update} {line}")
            if float(eval_metrics["H80"]) >= best_h80:
                best_h80 = float(eval_metrics["H80"])
                best_metrics = {**metrics, **eval_metrics}
                save_checkpoint(
                    output_dir / "best_checkpoint",
                    model=model,
                    model_name=model_name,
                    config=cfg,
                    normalizer=normalizer.to_dict(),
                    step=update,
                    metrics=best_metrics,
                )
    summary = {"model": model_name, "updates": updates, "best_H80": best_h80, "metrics": best_metrics}
    save_json(output_dir / "train_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["baseline", "student", "solution"], required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.model, args.dataset_dir, args.output_dir, smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
