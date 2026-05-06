"""Small plotting and video helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_prediction_plot(
    output_path: Path,
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    title: str = "World model open-loop prediction",
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    horizon = min(len(target), len(prediction))
    dims = min(target.shape[-1], 3)

    fig, axes = plt.subplots(dims, 1, figsize=(8, 2.2 * dims), sharex=True)
    if dims == 1:
        axes = [axes]
    x = np.arange(horizon)
    for dim, ax in enumerate(axes):
        ax.plot(x, target[:horizon, dim], label=f"target obs[{dim}]", linewidth=2)
        ax.plot(x, prediction[:horizon, dim], label=f"pred obs[{dim}]", linestyle="--")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[0].set_title(title)
    axes[-1].set_xlabel("prediction step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_video(frames: list[np.ndarray], output_path: Path, fps: int = 30) -> Path | None:
    if not frames:
        return None
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(output_path, frames, fps=fps)
        return output_path
    except Exception:
        fallback = output_path.with_suffix(".gif")
        imageio.mimsave(fallback, frames, fps=fps)
        return fallback

