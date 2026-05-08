from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from smallworld_mj.checkpoint import load_checkpoint
from smallworld_mj.data import Normalizer, load_split
from smallworld_mj.metrics import evaluate_model
from smallworld_mj.tasks import get_task
from smallworld_mj.visualization import save_horizon_error_plot, save_rollout_plot, save_rollout_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--task", default="bouncing_ball")
    parser.add_argument("--split", default="test")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(args.checkpoint_dir, device=device)
    data = load_split(args.dataset_dir, args.split)
    task_id = get_task(args.task).task_id
    mask = data["task_id"] == task_id
    if not np.any(mask):
        raise SystemExit(f"No task '{args.task}' episodes in split '{args.split}'.")
    task_data = {k: v[mask] if hasattr(v, "__len__") and len(v) == len(mask) else v for k, v in data.items()}
    result = evaluate_model(model, task_data, Normalizer.from_dict(payload["normalizer"]), warmup=args.warmup, horizon=args.horizon, device=device, batch_size=16)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred = result["rollout_pred"][0]
    target = result["rollout_target"][0]
    print(
        {
            "rollout_plot": str(save_rollout_plot(out / "rollout_plot.png", pred, target, args.task)),
            "horizon_error": str(save_horizon_error_plot(out / "horizon_error.png", result["horizon_rmse"], args.task)),
            "video": str(save_rollout_video(out / "smallworld_mj_rollout.mp4", pred, target, args.task)),
        }
    )


if __name__ == "__main__":
    main()
