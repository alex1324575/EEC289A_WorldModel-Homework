#!/usr/bin/env python3
"""Inspect the configured Gymnasium environment and action normalization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from world_model_hw.config import DEFAULT_CONFIG_PATH, apply_stage_overrides, load_json, save_json
from world_model_hw.envs import denormalize_action, get_env_info, make_env, reset_env, step_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--stage", choices=["baseline", "local_smoke"], default="baseline")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_stage_overrides(load_json(args.config), args.stage)
    env = make_env(config, seed=int(config["seed"]))
    info = get_env_info(env)

    obs_a = reset_env(env, seed=123)
    obs_b = reset_env(env, seed=123)
    zero_action = np.zeros(info.action_dim, dtype=np.float32)
    env_action = denormalize_action(zero_action, info)
    next_obs, reward, done, _ = step_env(env, zero_action, info)
    summary = {
        "env_name": config["env"]["name"],
        "obs_dim": info.obs_dim,
        "action_dim": info.action_dim,
        "action_low": info.action_low.tolist(),
        "action_high": info.action_high.tolist(),
        "normalized_zero_action_maps_to": env_action.tolist(),
        "reset_reproducible": bool(np.allclose(obs_a, obs_b)),
        "sample_obs": obs_a.tolist(),
        "one_step_next_obs": next_obs.tolist(),
        "one_step_reward": float(reward),
        "one_step_done": bool(done),
        "stage": args.stage,
    }
    env.close()
    if args.output_json:
        save_json(args.output_json, summary)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

