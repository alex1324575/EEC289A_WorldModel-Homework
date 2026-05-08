#!/usr/bin/env python3
"""Train a reward-free RSSM on one SmallWorld-Lite task."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model_hw.config import choose_device, deep_update, load_json, save_json, set_global_seeds, set_runtime_env

from smallworld_hw.checkpointing import save_smallworld_checkpoint
from smallworld_hw.dataset import generate_task_dataset, load_split, sample_batch, split_path
from smallworld_hw.metrics import compute_smallworld_metrics, predict_episode_rollouts
from smallworld_hw.models import SmallWorldRSSM, tensor_batch
from smallworld_hw.tasks import get_task


DEFAULT_CONFIG = Path("configs/smallworld_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", type=str, default="simple_pendulum")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_config(config: dict[str, Any], *, local_smoke: bool) -> dict[str, Any]:
    resolved = dict(config)
    resolved["active_stage"] = "local_smoke" if local_smoke else "baseline"
    if local_smoke:
        smoke = config["smoke"]
        resolved["training"] = deep_update(
            resolved["training"],
            {
                "updates": smoke["updates"],
                "batch_size": smoke["batch_size"],
                "batch_length": smoke["batch_length"],
                "eval_every_updates": smoke["eval_every_updates"],
            },
        )
    return resolved


@torch.no_grad()
def evaluate_model(model: SmallWorldRSSM, config: dict[str, Any], task_name: str, data: dict[str, np.ndarray], device: str) -> dict[str, Any]:
    task = get_task(task_name)
    eval_cfg = config["evaluation"]
    rollout = predict_episode_rollouts(
        model,
        data,
        warmup_steps=int(eval_cfg["warmup_steps"]),
        horizon=int(eval_cfg["open_loop_horizon"]),
        device=device,
        batch_size=int(eval_cfg["batch_size"]),
    )
    return compute_smallworld_metrics(task, rollout, data)


def ensure_dataset(config: dict[str, Any], task_name: str, dataset_dir: Path, *, local_smoke: bool) -> None:
    train_path = split_path(dataset_dir, task_name, "train")
    if train_path.exists():
        return
    print(f"[smallworld_train] dataset missing; generating {task_name} into {dataset_dir}", flush=True)
    generate_task_dataset(dataset_dir, task_name, config, local_smoke=local_smoke)


def train(config: dict[str, Any], *, task_name: str, dataset_dir: Path, output_dir: Path, device: str) -> dict[str, Any]:
    seed = int(config["seed"])
    set_global_seeds(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_smallworld_config.json", config)

    task = get_task(task_name)
    train_data = load_split(dataset_dir, task_name, "train")
    val_data = load_split(dataset_dir, task_name, "val")
    rng = np.random.default_rng(seed + 123)
    model = SmallWorldRSSM(task.state_dim, task.action_dim, config["world_model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["world_model"]["learning_rate"]))
    train_cfg = config["training"]
    updates = int(train_cfg["updates"])
    batch_size = int(train_cfg["batch_size"])
    batch_length = int(train_cfg["batch_length"])
    eval_every = int(train_cfg["eval_every_updates"])

    print(f"[smallworld_train] task={task_name} device={device} updates={updates}", flush=True)
    best_val = float("inf")
    progress = []
    start = time.monotonic()
    for update in range(1, updates + 1):
        batch_np = sample_batch(train_data, batch_size=batch_size, batch_length=batch_length, rng=rng)
        batch = tensor_batch(batch_np, device)
        model.train()
        loss, metrics = model.loss(batch, config["world_model"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["world_model"]["grad_clip"]))
        optimizer.step()

        if update == updates or update % eval_every == 0:
            model.eval()
            val_metrics = evaluate_model(model, config, task_name, val_data, device)
            metrics.update({f"val/{key}": value for key, value in val_metrics.items() if isinstance(value, (int, float))})
            score_key = float(val_metrics["open_loop_15_rmse"])
            if score_key < best_val:
                best_val = score_key
                save_smallworld_checkpoint(
                    output_dir / "best_checkpoint",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    task_name=task_name,
                    metrics=val_metrics,
                    update=update,
                )
        record = {
            "update": update,
            "elapsed_seconds": float(time.monotonic() - start),
            "metrics": metrics,
        }
        progress.append(record)
        save_json(output_dir / "progress_live.json", progress)
        if update == updates or update % max(1, eval_every // 2) == 0:
            short = " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float)))
            print(f"[smallworld_train] update={update} {short}", flush=True)

    final_metrics = evaluate_model(model, config, task_name, val_data, device)
    save_smallworld_checkpoint(
        output_dir / "latest_checkpoint",
        model=model,
        optimizer=optimizer,
        config=config,
        task_name=task_name,
        metrics=final_metrics,
        update=updates,
    )
    summary = {
        "task": task_name,
        "updates": updates,
        "best_val_open_loop_15_rmse": best_val,
        "final_metrics": final_metrics,
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    set_runtime_env()
    raw_config = load_json(args.config)
    config = resolve_config(raw_config, local_smoke=args.local_smoke)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    task = get_task(args.task)
    dataset_dir = args.dataset_dir or Path(config["dataset"]["default_output_dir"])
    device = choose_device(args.device)
    if args.dry_run:
        model = SmallWorldRSSM(task.state_dim, task.action_dim, config["world_model"]).to(device)
        print(
            {
                "task": task.name,
                "state_dim": task.state_dim,
                "action_dim": task.action_dim,
                "device": device,
                "parameters": sum(param.numel() for param in model.parameters()),
                "dataset_dir": str(dataset_dir),
                "training": config["training"],
            }
        )
        return
    ensure_dataset(config, task.name, dataset_dir, local_smoke=args.local_smoke)
    summary = train(config, task_name=task.name, dataset_dir=dataset_dir, output_dir=args.output_dir, device=device)
    print(summary)


if __name__ == "__main__":
    main()
