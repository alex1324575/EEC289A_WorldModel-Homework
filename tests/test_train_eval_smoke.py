from __future__ import annotations

from pathlib import Path

from wm_hw.dataset import generate_dataset
from wm_hw.eval_horizon import evaluate_checkpoint
from wm_hw.train import train


def test_train_and_eval_smoke(tmp_path: Path):
    data_dir = tmp_path / "data"
    generate_dataset("configs/colab.yaml", data_dir, smoke=True)
    out_dir = tmp_path / "student"
    train("configs/student.yaml", "student", data_dir, out_dir, smoke=True)
    metrics = evaluate_checkpoint(out_dir / "best_checkpoint", data_dir, "test", tmp_path / "eval")
    assert "H80" in metrics
    assert (tmp_path / "eval" / "metrics.json").exists()
    assert (tmp_path / "eval" / "per_window_survival.npy").exists()
