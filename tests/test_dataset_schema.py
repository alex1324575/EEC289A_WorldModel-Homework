from __future__ import annotations

from wm_hw.config import load_config
from wm_hw.dataset import generate_split


def test_dataset_split_schema_smoke():
    cfg = load_config("configs/colab.yaml")
    data = generate_split("train", cfg, smoke=True)
    assert data["states"].shape == (cfg["smoke"]["train_windows"], 106, 4)
    assert data["actions"].shape == (cfg["smoke"]["train_windows"], 105, 1)
    assert abs(data["states"][:, :, 1]).max() < 0.20
