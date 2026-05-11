from __future__ import annotations

from copy import deepcopy

from wm_hw.config import load_config
from wm_hw.dataset import generate_split
from wm_hw.horizon import dataset_window_spec


def test_dataset_split_schema_smoke():
    cfg = load_config("configs/colab.yaml")
    data = generate_split("train", cfg, smoke=True)
    spec = dataset_window_spec(cfg["dataset"])
    assert data["states"].shape == (cfg["smoke"]["train_windows"], spec["window_states"], 4)
    assert data["actions"].shape == (cfg["smoke"]["train_windows"], spec["window_actions"], 1)
    assert abs(data["states"][:, :, 1]).max() < 0.20


def test_dataset_schema_uses_configured_scoreboard_horizon():
    cfg = deepcopy(load_config("configs/colab.yaml"))
    cfg["dataset"]["max_horizon"] = 37
    cfg["smoke"]["train_windows"] = 4
    data = generate_split("train", cfg, smoke=True)
    spec = dataset_window_spec(cfg["dataset"])
    assert spec["window_states"] == 43
    assert spec["window_actions"] == 42
    assert data["states"].shape == (4, 43, 4)
    assert data["actions"].shape == (4, 42, 1)
