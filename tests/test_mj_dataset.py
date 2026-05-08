from __future__ import annotations

import numpy as np

from smallworld_mj.data.dataset import generate_split, smooth_random_actions


def test_smooth_random_actions_shape_and_bounds():
    actions = smooth_random_actions(np.random.default_rng(0), 12, 3)
    assert actions.shape == (12, 3)
    assert np.max(np.abs(actions)) <= 1.0


def test_generated_split_schema_smoke():
    data = generate_split(["simple_pendulum", "projectile"], "train", episodes_per_task=2, episode_steps=12, seed=0)
    assert data["states"].shape[0] == 4
    assert data["states"].shape[1] == 13
    assert data["actions"].shape[1] == 12
    assert data["task_params"].shape[0] == 4
    assert data["state_mask"].shape[-1] == data["states"].shape[-1]
    assert set(data["task_id"].tolist()) == {0, 1}
