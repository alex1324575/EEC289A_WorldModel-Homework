#!/usr/bin/env python3
"""Evaluate a SmallWorld-Lite checkpoint on test and OOD splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_model_hw.config import choose_device, load_json, save_json, set_runtime_env

from smallworld_hw.checkpointing import load_smallworld_checkpoint
from smallworld_hw.dataset import load_split
from smallworld_hw.metrics import composite_score, compute_smallworld_metrics, predict_episode_rollouts
from smallworld_hw.tasks import get_task


DEFAULT_CONFIG = Path("configs/smallworld_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def evaluate_split(model, config, task_name: str, dataset_dir: Path, split: str, device: str):
    task = get_task(task_name)
    data = load_split(dataset_dir, task_name, split)
    eval_cfg = config["evaluation"]
    rollout = predict_episode_rollouts(
        model,
        data,
        warmup_steps=int(eval_cfg["warmup_steps"]),
        horizon=int(eval_cfg["open_loop_horizon"]),
        device=device,
        batch_size=int(eval_cfg["batch_size"]),
    )
    return compute_smallworld_metrics(task, rollout, data), rollout


def main() -> None:
    args = parse_args()
    set_runtime_env()
    device = choose_device(args.device)
    model, ckpt_config, payload = load_smallworld_checkpoint(args.checkpoint_dir, device)
    public_config = load_json(args.config)
    config = ckpt_config
    task_name = str(payload["task_name"])

    test_metrics, _test_rollout = evaluate_split(model, config, task_name, args.dataset_dir, "test", device)
    ood_metrics, _ood_rollout = evaluate_split(model, config, task_name, args.dataset_dir, "ood", device)
    score_metrics = dict(test_metrics)
    score_metrics["ood_open_loop_90_rmse"] = ood_metrics["open_loop_90_rmse"]
    normalized, composite = composite_score(score_metrics, public_config["evaluation"]["metrics"])
    result = {
        "benchmark_name": public_config["benchmark_name"],
        "task_name": task_name,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_update": int(payload["update"]),
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "score_metrics": score_metrics,
        "normalized_scores": normalized,
        "smallworld_composite_score": composite,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "smallworld_eval.json"
    save_json(output_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
