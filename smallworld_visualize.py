#!/usr/bin/env python3
"""Visualize SmallWorld-Lite open-loop predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from world_model_hw.config import choose_device, save_json, set_runtime_env

from smallworld_hw.checkpointing import load_smallworld_checkpoint
from smallworld_hw.dataset import load_split
from smallworld_hw.metrics import compute_smallworld_metrics, predict_episode_rollouts
from smallworld_hw.tasks import get_task
from smallworld_hw.visualization import save_horizon_error_plot, save_rollout_plot, save_rollout_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_runtime_env()
    device = choose_device(args.device)
    model, config, payload = load_smallworld_checkpoint(args.checkpoint_dir, device)
    task_name = str(payload["task_name"])
    task = get_task(task_name)
    data = load_split(args.dataset_dir, task_name, args.split)
    eval_cfg = config["evaluation"]
    rollout = predict_episode_rollouts(
        model,
        data,
        warmup_steps=int(eval_cfg["warmup_steps"]),
        horizon=int(eval_cfg["open_loop_horizon"]),
        device=device,
        batch_size=int(eval_cfg["batch_size"]),
    )
    metrics = compute_smallworld_metrics(task, rollout, data)
    ep = min(max(0, int(args.episode_index)), rollout["open_loop_pred"].shape[0] - 1)
    pred = rollout["open_loop_pred"][ep]
    target = rollout["open_loop_target"][ep]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollout_plot = save_rollout_plot(args.output_dir / "smallworld_rollout.png", target=target, prediction=pred, task_name=task_name)
    horizon_plot = save_horizon_error_plot(
        args.output_dir / "smallworld_horizon_error.png",
        horizon_rmse=metrics["horizon_rmse"],
        task_name=task_name,
    )
    video_path = None
    if not args.no_video:
        video_path = save_rollout_video(args.output_dir / "smallworld_rollout.mp4", task=task, target=target, prediction=pred)
    summary = {
        "task_name": task_name,
        "split": args.split,
        "episode_index": ep,
        "rollout_plot": str(rollout_plot),
        "horizon_error_plot": str(horizon_plot),
        "video_path": str(video_path) if video_path else None,
        "metrics": metrics,
    }
    save_json(args.output_dir / "visualization_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
