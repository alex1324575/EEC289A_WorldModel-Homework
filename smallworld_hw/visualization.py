"""Plots and videos for SmallWorld-Lite predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .tasks import SmallWorldTask


def save_rollout_plot(output_path: Path, *, target: np.ndarray, prediction: np.ndarray, task_name: str) -> Path:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    dims = min(target.shape[-1], 4)
    horizon = min(target.shape[0], prediction.shape[0])
    x = np.arange(horizon)
    fig, axes = plt.subplots(dims, 1, figsize=(8, 2.1 * dims), sharex=True)
    if dims == 1:
        axes = [axes]
    for dim, ax in enumerate(axes):
        ax.plot(x, target[:horizon, dim], label=f"target state[{dim}]", linewidth=2)
        ax.plot(x, prediction[:horizon, dim], "--", label=f"pred state[{dim}]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[0].set_title(f"SmallWorld open-loop rollout: {task_name}")
    axes[-1].set_xlabel("open-loop prediction step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_horizon_error_plot(output_path: Path, *, horizon_rmse: np.ndarray, task_name: str) -> Path:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    curve = np.asarray(horizon_rmse, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(curve) + 1), curve, linewidth=2)
    ax.set_title(f"Horizon error curve: {task_name}")
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel("state RMSE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _frame_from_points(task: SmallWorldTask, target_state: np.ndarray, pred_state: np.ndarray, step: int) -> np.ndarray:
    import matplotlib.pyplot as plt

    target_xy = task.position(target_state)
    pred_xy = task.position(pred_state)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
    all_xy = np.stack([target_xy, pred_xy])
    center = all_xy.mean(axis=0)
    span = max(1.0, float(np.max(np.abs(all_xy - center))) * 2.5)
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.scatter([target_xy[0]], [target_xy[1]], s=90, label="true", color="#1769aa")
    ax.scatter([pred_xy[0]], [pred_xy[1]], s=90, label="model", color="#d1495b", alpha=0.75)
    ax.plot([target_xy[0], pred_xy[0]], [target_xy[1], pred_xy[1]], color="#555555", alpha=0.35)
    ax.set_title(f"{task.name} step {step}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def save_rollout_video(
    output_path: Path,
    *,
    task: SmallWorldTask,
    target: np.ndarray,
    prediction: np.ndarray,
    fps: int = 12,
) -> Path | None:
    if len(target) == 0 or len(prediction) == 0:
        return None
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    horizon = min(len(target), len(prediction))
    stride = max(1, horizon // 90)
    frames = [_frame_from_points(task, target[t], prediction[t], t) for t in range(0, horizon, stride)]
    try:
        imageio.mimsave(output_path, frames, fps=fps, macro_block_size=1)
        return output_path
    except Exception:
        fallback = output_path.with_suffix(".gif")
        imageio.mimsave(fallback, frames, fps=fps)
        return fallback
