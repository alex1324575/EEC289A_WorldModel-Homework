from __future__ import annotations

import torch

from smallworld_mj.solution.losses import compute_loss
from smallworld_mj.solution.model import ParamResidualGRU
from smallworld_mj.solution.rollout import open_loop_rollout


def fake_batch(batch=3, steps=24, state_dim=6, action_dim=3, param_dim=2):
    return {
        "states": torch.randn(batch, steps + 1, state_dim),
        "actions": torch.randn(batch, steps, action_dim).clamp(-1, 1),
        "task_params": torch.randn(batch, param_dim),
        "task_id": torch.zeros(batch, dtype=torch.long),
        "state_mask": torch.ones(batch, state_dim),
        "action_mask": torch.ones(batch, action_dim),
        "param_mask": torch.ones(batch, param_dim),
    }


def test_open_loop_shape_and_grad_flow():
    model = ParamResidualGRU(6, 3, 2, 10, {"hidden_dim": 32, "task_embedding_dim": 8})
    batch = fake_batch()
    pred = open_loop_rollout(model, batch, warmup=5, horizon=10)
    assert pred.shape == (3, 10, 6)
    loss = pred.sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_mixed_loss_runs():
    model = ParamResidualGRU(6, 3, 2, 10, {"hidden_dim": 32, "task_embedding_dim": 8})
    batch = fake_batch()
    cfg = {"loss": {"warmup": 5, "one_step_weight": 1.0, "rollout_weight": 0.3, "rollout_train_horizon": 10, "physics_weight": 0.01}}
    loss, metrics = compute_loss(model, batch, cfg)
    assert loss.ndim == 0
    assert "loss/rollout" in metrics
