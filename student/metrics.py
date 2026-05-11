"""Student failure-horizon metrics."""

from __future__ import annotations

import numpy as np
import torch


def _threshold_value(thresholds, key: str) -> float:
    if isinstance(thresholds, dict):
        return float(thresholds[key])
    return float(getattr(thresholds, key))


def compute_failure_horizon(preds, targets, thresholds):
    """Compute per-window survival and aggregate H80/H50 metrics."""
    pred_t = torch.as_tensor(preds, dtype=torch.float32)
    target_t = torch.as_tensor(targets, dtype=torch.float32)
    err = (pred_t - target_t).abs()
    violated = (
        (err[..., 1] > _threshold_value(thresholds, "angle_error_rad"))
        | (err[..., 0] > _threshold_value(thresholds, "cart_pos_error_m"))
        | (err[..., 2] > _threshold_value(thresholds, "cart_vel_error_mps"))
        | (err[..., 3] > _threshold_value(thresholds, "pole_vel_error_radps"))
    )
    consecutive = int(_threshold_value(thresholds, "consecutive_fail_steps"))
    batch, horizon = violated.shape
    survival = torch.full((batch,), horizon, dtype=torch.long)
    for b in range(batch):
        run = 0
        for h in range(horizon):
            run = run + 1 if bool(violated[b, h]) else 0
            if run >= consecutive:
                survival[b] = max(0, h + 1 - consecutive)
                break
    surv_np = survival.cpu().numpy()

    def h_percent(percent: float) -> int:
        rates = [(surv_np >= h).mean() for h in range(1, horizon + 1)]
        ok = [i + 1 for i, rate in enumerate(rates) if rate >= percent]
        return int(max(ok) if ok else 0)

    metrics = {
        "H80": h_percent(0.80),
        "H50": h_percent(0.50),
        "mean_survival_steps": float(np.mean(surv_np)),
        "median_survival_steps": float(np.median(surv_np)),
    }
    for h in [5, 10, 25, 50, 100]:
        metrics[f"success_rate@{h}"] = float(np.mean(surv_np >= min(h, horizon)))
    return survival, metrics
