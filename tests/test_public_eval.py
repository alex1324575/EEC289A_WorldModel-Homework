import numpy as np

from public_eval import compute_metrics, compute_scores, normalize_rollout


def test_public_eval_metric_shapes():
    bundle = {
        "episode_id": np.array([0, 0, 1, 1], dtype=np.int32),
        "step": np.array([0, 1, 0, 1], dtype=np.int32),
        "obs": np.zeros((4, 3), dtype=np.float32),
        "action": np.zeros((4, 1), dtype=np.float32),
        "reward": np.ones((4,), dtype=np.float32),
        "pred_obs_1step": np.zeros((4, 3), dtype=np.float32),
        "pred_reward_1step": np.ones((4,), dtype=np.float32),
        "open_loop_obs_pred": np.zeros((4, 2, 3), dtype=np.float32),
        "open_loop_obs_target": np.zeros((4, 2, 3), dtype=np.float32),
        "eval_return": np.array([-100.0, -100.0, -200.0, -200.0], dtype=np.float32),
        "action_delta": np.zeros((4,), dtype=np.float32),
    }
    normalized = normalize_rollout(bundle)
    metrics = compute_metrics(normalized)
    assert metrics["mean_return"] == -150.0
    assert metrics["one_step_obs_rmse"] == 0.0
    assert metrics["reward_mae"] == 0.0

    scores, composite = compute_scores(
        metrics,
        {
            "mean_return": {"direction": "higher_better", "weight": 1.0, "good": 0.0, "bad": -300.0},
            "one_step_obs_rmse": {"direction": "lower_better", "weight": 1.0, "good": 0.0, "bad": 1.0},
            "open_loop_obs_rmse": {"direction": "lower_better", "weight": 1.0, "good": 0.0, "bad": 1.0},
            "reward_mae": {"direction": "lower_better", "weight": 1.0, "good": 0.0, "bad": 1.0},
            "action_delta": {"direction": "lower_better", "weight": 1.0, "good": 0.0, "bad": 1.0},
        },
    )
    assert 0.0 <= composite <= 1.0
    assert scores["reward_mae"] == 1.0

