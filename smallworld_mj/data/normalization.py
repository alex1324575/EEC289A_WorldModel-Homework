"""State normalization for padded multi-task datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_train(cls, train: dict[str, np.ndarray]) -> "Normalizer":
        states = np.asarray(train["states"], dtype=np.float32)
        masks = np.asarray(train["state_mask"], dtype=np.float32)
        flat = states.reshape(-1, states.shape[-1])
        counts = np.maximum(masks.sum(axis=0) * states.shape[1], 1.0)
        mean = (flat.sum(axis=0) / counts).astype(np.float32)
        var = (((flat - mean) ** 2).reshape(states.shape) * masks[:, None, :]).sum(axis=(0, 1)) / counts
        std = np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)
        return cls(mean=mean, std=std)

    def normalize_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / (std + 1e-6)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> "Normalizer":
        return cls(mean=np.asarray(payload["mean"], dtype=np.float32), std=np.asarray(payload["std"], dtype=np.float32))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
