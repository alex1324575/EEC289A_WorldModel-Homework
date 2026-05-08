import numpy as np

from smallworld_hw.metrics import compute_smallworld_metrics, horizon_rmse_curve, rmse
from smallworld_hw.tasks import get_task


def test_smallworld_rmse_and_horizon_curve_zero_for_perfect_prediction():
    target = np.zeros((2, 5, 3), dtype=np.float32)
    pred = target.copy()
    assert rmse(pred, target) == 0.0
    np.testing.assert_allclose(horizon_rmse_curve(pred, target), np.zeros(5))


def test_smallworld_metrics_have_expected_keys():
    task = get_task("simple_pendulum")
    target = np.zeros((2, 20, task.state_dim), dtype=np.float32)
    target[..., 1] = 1.0
    rollout = {
        "one_step_pred": np.zeros((2, 21, task.state_dim), dtype=np.float32),
        "one_step_target": np.zeros((2, 21, task.state_dim), dtype=np.float32),
        "open_loop_pred": target.copy(),
        "open_loop_target": target.copy(),
        "effective_horizon": np.asarray(20, dtype=np.int32),
    }
    data = {"task_params": np.tile(np.array([[9.81, 1.0, 0.0]], dtype=np.float32), (2, 1))}
    metrics = compute_smallworld_metrics(task, rollout, data)
    assert metrics["one_step_state_rmse"] == 0.0
    assert metrics["open_loop_15_rmse"] == 0.0
    assert metrics["open_loop_90_rmse"] == 0.0
    assert metrics["horizon_error_auc"] == 0.0
    assert "energy_drift" in metrics
