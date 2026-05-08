from __future__ import annotations

import numpy as np
import torch

from smallworld_mj.checkpoint import load_checkpoint, save_checkpoint
from smallworld_mj.data.normalization import Normalizer
from smallworld_mj.metrics import horizon_curve, normalized_rmse
from smallworld_mj.solution.model import ParamResidualGRU


def test_normalized_metric_math():
    pred = np.zeros((2, 4, 3), dtype=np.float32)
    target = np.ones((2, 4, 3), dtype=np.float32)
    mask = np.ones((2, 3), dtype=np.float32)
    normalizer = Normalizer(mean=np.zeros(3, dtype=np.float32), std=np.ones(3, dtype=np.float32))
    np.testing.assert_allclose(normalized_rmse(pred, target, mask, normalizer), 1.0, rtol=1e-5)
    curve = horizon_curve(pred, target, mask, normalizer)
    assert curve.shape == (4,)
    np.testing.assert_allclose(curve, np.ones(4), rtol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    model = ParamResidualGRU(6, 3, 2, 10, {"hidden_dim": 32, "task_embedding_dim": 8})
    save_checkpoint(
        tmp_path / "ckpt",
        model=model,
        config={"model": {"hidden_dim": 32, "task_embedding_dim": 8}},
        model_name="solution",
        dims=(6, 3, 2),
        num_tasks=10,
        normalizer={"mean": [0.0] * 6, "std": [1.0] * 6},
        step=1,
        metrics={},
    )
    loaded, payload = load_checkpoint(tmp_path / "ckpt")
    assert isinstance(loaded, torch.nn.Module)
    assert payload["step"] == 1
