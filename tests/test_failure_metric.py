from __future__ import annotations

import torch

from student.metrics import compute_failure_horizon


TH = {
    "angle_error_rad": 0.075,
    "cart_pos_error_m": 0.10,
    "cart_vel_error_mps": 0.75,
    "pole_vel_error_radps": 1.00,
    "consecutive_fail_steps": 2,
}


def test_perfect_prediction_survives_full_horizon():
    pred = torch.zeros(3, 100, 4)
    target = torch.zeros_like(pred)
    survival, metrics = compute_failure_horizon(pred, target, TH)
    assert survival.tolist() == [100, 100, 100]
    assert metrics["H80"] == 100


def test_two_consecutive_angle_violations_fail():
    pred = torch.zeros(1, 100, 4)
    target = torch.zeros_like(pred)
    pred[0, 9:11, 1] = 0.2
    survival, metrics = compute_failure_horizon(pred, target, TH)
    assert survival.item() == 9
    assert metrics["H50"] == 9


def test_single_step_violation_does_not_fail():
    pred = torch.zeros(1, 100, 4)
    target = torch.zeros_like(pred)
    pred[0, 9, 1] = 0.2
    survival, _ = compute_failure_horizon(pred, target, TH)
    assert survival.item() == 100
