"""Gymnasium helpers and normalized action conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EnvInfo:
    obs_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray


def make_env(config: dict[str, Any], *, seed: int | None = None, render_mode: str | None = None):
    import gymnasium as gym

    kwargs: dict[str, Any] = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make(config["env"]["name"], **kwargs)
    if config["env"].get("max_episode_steps") is not None:
        env = gym.wrappers.TimeLimit(env.env, max_episode_steps=int(config["env"]["max_episode_steps"]))
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def get_env_info(env) -> EnvInfo:
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    if obs_shape is None or len(obs_shape) != 1:
        raise ValueError(f"Expected a flat vector observation space, got {env.observation_space}")
    if action_shape is None or len(action_shape) != 1:
        raise ValueError(f"Expected a flat vector action space, got {env.action_space}")
    return EnvInfo(
        obs_dim=int(obs_shape[0]),
        action_dim=int(action_shape[0]),
        action_low=np.asarray(env.action_space.low, dtype=np.float32),
        action_high=np.asarray(env.action_space.high, dtype=np.float32),
    )


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def denormalize_action(action: np.ndarray, env_info: EnvInfo) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    action = np.clip(action, -1.0, 1.0)
    midpoint = 0.5 * (env_info.action_high + env_info.action_low)
    scale = 0.5 * (env_info.action_high - env_info.action_low)
    return midpoint + scale * action


def normalize_action(action: np.ndarray, env_info: EnvInfo) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    midpoint = 0.5 * (env_info.action_high + env_info.action_low)
    scale = 0.5 * (env_info.action_high - env_info.action_low)
    return np.clip((action - midpoint) / np.maximum(scale, 1e-6), -1.0, 1.0)


def reset_env(env, seed: int | None = None) -> np.ndarray:
    obs, _ = env.reset(seed=seed)
    return normalize_obs(obs)


def step_env(env, normalized_action: np.ndarray, env_info: EnvInfo):
    env_action = denormalize_action(normalized_action, env_info)
    obs, reward, terminated, truncated, info = env.step(env_action)
    done = bool(terminated or truncated)
    return normalize_obs(obs), float(reward), done, info

