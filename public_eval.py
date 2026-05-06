#!/usr/bin/env python3
"""Score a public MiniDreamer rollout bundle.

The benchmark intentionally mixes behavior and model-quality metrics. A good
submission should swing up the pendulum while also learning a useful predictive
world model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from world_model_hw.config import DEFAULT_CONFIG_PATH, load_json, save_json


REQUIRED_FIELDS = [
    "episode_id",
    "step",
    "obs",
    "action",
    "reward",
    "pred_obs_1step",
    "pred_reward_1step",
    "open_loop_obs_pred",
    "open_loop_obs_target",
    "eval_return",
    "action_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-npz", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def normalize_rollout(bundle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = [field for field in REQUIRED_FIELDS if field not in bundle]
    if missing:
        raise KeyError(f"Missing required rollout fields: {missing}")
    normalized = {key: np.asarray(bundle[key]) for key in REQUIRED_FIELDS}
    first_dim = normalized["step"].shape[0]
    for key, value in normalized.items():
        if value.shape[0] != first_dim:
            raise ValueError(f"Field '{key}' has inconsistent first dimension {value.shape[0]} != {first_dim}")
    return normalized


def masked_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mask = np.isfinite(pred) & np.isfinite(target)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - target[mask]) ** 2)))


def compute_metrics(bundle: dict[str, np.ndarray]) -> dict[str, float]:
    episode_id = bundle["episode_id"]
    returns = []
    for eid in np.unique(episode_id):
        mask = episode_id == eid
        returns.append(float(np.asarray(bundle["eval_return"][mask])[0]))
    one_step_target = bundle["open_loop_obs_target"][:, 0, :]
    return {
        "mean_return": float(np.mean(returns)),
        "one_step_obs_rmse": masked_rmse(bundle["pred_obs_1step"], one_step_target),
        "open_loop_obs_rmse": masked_rmse(bundle["open_loop_obs_pred"], bundle["open_loop_obs_target"]),
        "reward_mae": float(np.mean(np.abs(bundle["pred_reward_1step"] - bundle["reward"]))),
        "action_delta": float(np.mean(bundle["action_delta"])),
    }


def lower_better_score(value: float, good: float, bad: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(np.clip((bad - value) / (bad - good), 0.0, 1.0))


def higher_better_score(value: float, good: float, bad: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(np.clip((value - bad) / (good - bad), 0.0, 1.0))


def compute_scores(metrics: dict[str, float], metric_cfg: dict[str, Any]) -> tuple[dict[str, float], float]:
    normalized = {}
    weighted = 0.0
    total_weight = 0.0
    for name, cfg in metric_cfg.items():
        value = metrics[name]
        if cfg["direction"] == "lower_better":
            score = lower_better_score(value, float(cfg["good"]), float(cfg["bad"]))
        elif cfg["direction"] == "higher_better":
            score = higher_better_score(value, float(cfg["good"]), float(cfg["bad"]))
        else:
            raise ValueError(f"Unsupported metric direction for {name}: {cfg['direction']}")
        normalized[name] = score
        if np.isfinite(score):
            weight = float(cfg["weight"])
            weighted += score * weight
            total_weight += weight
    composite = weighted / total_weight if total_weight else float("nan")
    return normalized, float(composite)


def per_episode_summary(bundle: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for eid in np.unique(bundle["episode_id"]):
        mask = bundle["episode_id"] == eid
        rows.append(
            {
                "episode_id": int(eid),
                "num_steps": int(np.sum(mask)),
                "eval_return": float(np.asarray(bundle["eval_return"][mask])[0]),
                "mean_action_delta": float(np.mean(bundle["action_delta"][mask])),
                "reward_mae": float(np.mean(np.abs(bundle["pred_reward_1step"][mask] - bundle["reward"][mask]))),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    raw = dict(np.load(args.rollout_npz))
    bundle = normalize_rollout(raw)
    metrics = compute_metrics(bundle)
    normalized_scores, composite = compute_scores(metrics, config["public_eval"]["metrics"])
    result = {
        "homework_name": config["homework_name"],
        "algorithm": config["algorithm"],
        "num_steps": int(bundle["step"].shape[0]),
        "num_episodes": int(np.unique(bundle["episode_id"]).shape[0]),
        "metrics": metrics,
        "normalized_scores": normalized_scores,
        "course_composite_score": composite,
        "per_episode_summary": per_episode_summary(bundle),
    }
    save_json(args.output_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

