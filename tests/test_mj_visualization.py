from __future__ import annotations

import numpy as np

from smallworld_mj.envs import SmallWorldMJEnv
from smallworld_mj.tasks import get_task
from smallworld_mj.visualization import save_horizon_error_plot, save_rollout_plot, save_rollout_video


def test_visualization_smoke(tmp_path):
    pred = np.zeros((8, 6), dtype=np.float32)
    target = np.ones((8, 6), dtype=np.float32) * 0.1
    assert save_rollout_plot(tmp_path / "rollout.png", pred, target, "projectile").exists()
    assert save_horizon_error_plot(tmp_path / "horizon.png", np.linspace(0, 1, 8), "projectile").exists()
    assert save_rollout_video(tmp_path / "rollout.mp4", pred, target, "projectile", fps=4).exists()


def test_mujoco_headless_render_smoke():
    env = SmallWorldMJEnv(get_task("bouncing_ball"))
    env.reset(np.random.default_rng(0))
    frame = env.render_rgb(width=160, height=120)
    assert frame.shape == (120, 160, 3)
