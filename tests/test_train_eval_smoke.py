from __future__ import annotations

from pathlib import Path

import numpy as np

from wm_hw.dataset import generate_dataset
from wm_hw.eval_horizon import evaluate_checkpoint
from wm_hw.plotting import plot_horizon_rmse, plot_survival_curve
from wm_hw.train import train


def test_train_and_eval_smoke(tmp_path: Path):
    data_dir = tmp_path / "data"
    generate_dataset("configs/colab.yaml", data_dir, smoke=True)
    out_dir = tmp_path / "student"
    train("configs/student.yaml", "student", data_dir, out_dir, smoke=True)
    metrics = evaluate_checkpoint(out_dir / "best_checkpoint", data_dir, "test", tmp_path / "eval")
    assert "H80" in metrics
    assert metrics["max_horizon"] == 100
    assert "open_loop_rmse@horizon" in metrics
    assert (tmp_path / "eval" / "metrics.json").exists()
    assert (tmp_path / "eval" / "per_window_survival.npy").exists()
    assert (tmp_path / "eval" / "scoreboard_summary.json").exists()
    override = evaluate_checkpoint(out_dir / "best_checkpoint", data_dir, "test", tmp_path / "eval20", horizon=20)
    assert override["max_horizon"] == 20
    assert "success_rate@20" in override
    plot_dir = tmp_path / "plots"
    survival = np.load(tmp_path / "eval" / "per_window_survival.npy")
    assert plot_survival_curve(metrics, plot_dir, survival).exists()
    assert plot_horizon_rmse(metrics, plot_dir).exists()
