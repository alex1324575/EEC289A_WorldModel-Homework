import numpy as np

from smallworld_hw.tasks import get_task
from smallworld_hw.visualization import save_horizon_error_plot, save_rollout_plot, save_rollout_video


def test_smallworld_visualizations_write_files(tmp_path):
    task = get_task("bouncing_ball")
    target = np.zeros((4, task.state_dim), dtype=np.float32)
    pred = target.copy()
    pred[:, 0] = np.linspace(0.0, 0.3, 4)
    rollout_path = save_rollout_plot(tmp_path / "rollout.png", target=target, prediction=pred, task_name=task.name)
    curve_path = save_horizon_error_plot(tmp_path / "curve.png", horizon_rmse=np.arange(4), task_name=task.name)
    video_path = save_rollout_video(tmp_path / "rollout.mp4", task=task, target=target, prediction=pred, fps=4)
    assert rollout_path.exists()
    assert curve_path.exists()
    assert video_path is not None
    assert video_path.exists()
