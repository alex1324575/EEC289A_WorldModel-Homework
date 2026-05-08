"""Plots and videos for SmallWorld-MJ predictions."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from smallworld_mj.tasks import get_task


def save_horizon_error_plot(path: str | Path, horizon_rmse: list[float] | np.ndarray, title: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(np.arange(1, len(horizon_rmse) + 1), horizon_rmse, linewidth=2)
    ax.set_xlabel("Open-loop horizon")
    ax.set_ylabel("Normalized RMSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_rollout_plot(path: str | Path, pred: np.ndarray, target: np.ndarray, task_name: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dims = min(pred.shape[-1], 6)
    fig, axes = plt.subplots(dims, 1, figsize=(8, max(2, 1.4 * dims)), sharex=True)
    if dims == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(target[:, i], label="target", linewidth=2)
        ax.plot(pred[:, i], label="prediction", linestyle="--")
        ax.set_ylabel(f"s[{i}]")
        ax.grid(True, alpha=0.25)
    axes[0].set_title(f"SmallWorld-MJ open-loop rollout: {task_name}")
    axes[-1].set_xlabel("Prediction step")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _frame(task_name: str, pred_state: np.ndarray, target_state: np.ndarray, step: int) -> np.ndarray:
    spec = get_task(task_name)
    pred = spec.position(pred_state)
    target = spec.position(target_state)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter([target[0]], [target[1]], s=100, c="#166534", label="MuJoCo target")
    ax.scatter([pred[0]], [pred[1]], s=100, c="#b91c1c", marker="x", label="model ghost")
    ax.plot([target[0], pred[0]], [target[1], pred[1]], color="0.6", alpha=0.6)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{task_name} step {step}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def save_rollout_video(path: str | Path, pred: np.ndarray, target: np.ndarray, task_name: str, fps: int = 8) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, pred.shape[0] // 90)
    frames = [_frame(task_name, pred[i], target[i], i) for i in range(0, pred.shape[0], stride)]
    imageio.mimsave(path, frames, fps=fps)
    return path
