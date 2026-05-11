"""Plot survival and rollout diagnostics."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_survival_curve(metrics: dict, output_dir: Path, survival_steps: np.ndarray | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if survival_steps is not None:
        max_horizon = int(metrics.get("max_horizon", int(np.max(survival_steps))))
        hs = np.arange(1, max_horizon + 1)
        rates = np.asarray([(survival_steps >= h).mean() for h in hs], dtype=np.float32)
    else:
        pairs = []
        for key, value in metrics.items():
            if key.startswith("success_rate@"):
                pairs.append((int(key.split("@", 1)[1]), float(value)))
        pairs = sorted(pairs)
        hs = np.asarray([h for h, _ in pairs], dtype=np.int32)
        rates = np.asarray([rate for _, rate in pairs], dtype=np.float32)
    path = output_dir / "survival_curve.png"
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hs, rates, marker="o", linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Survival rate")
    ax.set_title("Scoreboard survival curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_horizon_rmse(metrics: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rollout_comparison.png"
    curve = np.asarray(metrics.get("horizon_rmse", []), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(np.arange(1, len(curve) + 1), curve, linewidth=2)
    ax.set_xlabel("Open-loop prediction step")
    ax.set_ylabel("RMSE")
    ax.set_title("Open-loop RMSE by horizon")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with (Path(args.eval_dir) / "metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    survival_path = Path(args.eval_dir) / "per_window_survival.npy"
    survival_steps = np.load(survival_path) if survival_path.exists() else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        {
            "survival_curve": str(plot_survival_curve(metrics, output_dir, survival_steps)),
            "rollout_comparison": str(plot_horizon_rmse(metrics, output_dir)),
        }
    )


if __name__ == "__main__":
    main()
