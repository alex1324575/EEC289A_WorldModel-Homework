"""Episode replay buffer for sequence model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Episode:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    @property
    def steps(self) -> int:
        return int(self.actions.shape[0])


class EpisodeReplay:
    """Stores complete episodes and samples fixed-length contiguous segments."""

    def __init__(self, capacity_steps: int, obs_dim: int, action_dim: int, seed: int = 0):
        self.capacity_steps = int(capacity_steps)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)
        self.episodes: list[Episode] = []
        self.total_steps = 0
        self._reset_current()

    def _reset_current(self) -> None:
        self._cur_obs: list[np.ndarray] = []
        self._cur_actions: list[np.ndarray] = []
        self._cur_rewards: list[float] = []
        self._cur_dones: list[bool] = []

    def start_episode(self, obs: np.ndarray) -> None:
        self._reset_current()
        self._cur_obs.append(np.asarray(obs, dtype=np.float32).reshape(self.obs_dim))

    def add(self, action: np.ndarray, reward: float, done: bool, next_obs: np.ndarray) -> None:
        if not self._cur_obs:
            raise RuntimeError("Call start_episode(obs) before adding transitions.")
        self._cur_actions.append(np.asarray(action, dtype=np.float32).reshape(self.action_dim))
        self._cur_rewards.append(float(reward))
        self._cur_dones.append(bool(done))
        self._cur_obs.append(np.asarray(next_obs, dtype=np.float32).reshape(self.obs_dim))
        if done:
            self.finish_episode()

    def finish_episode(self) -> None:
        if not self._cur_actions:
            self._reset_current()
            return
        episode = Episode(
            obs=np.asarray(self._cur_obs, dtype=np.float32),
            actions=np.asarray(self._cur_actions, dtype=np.float32),
            rewards=np.asarray(self._cur_rewards, dtype=np.float32),
            dones=np.asarray(self._cur_dones, dtype=np.bool_),
        )
        self.episodes.append(episode)
        self.total_steps += episode.steps
        self._trim_to_capacity()
        self._reset_current()

    def _trim_to_capacity(self) -> None:
        while self.total_steps > self.capacity_steps and self.episodes:
            removed = self.episodes.pop(0)
            self.total_steps -= removed.steps

    def __len__(self) -> int:
        return self.total_steps

    def can_sample(self, batch_length: int) -> bool:
        return any(ep.steps >= batch_length for ep in self.episodes)

    def sample(self, batch_size: int, batch_length: int) -> dict[str, np.ndarray]:
        eligible = [ep for ep in self.episodes if ep.steps >= batch_length]
        if not eligible:
            raise RuntimeError(f"No episode is long enough for batch_length={batch_length}.")

        obs_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        for _ in range(int(batch_size)):
            ep = eligible[int(self.rng.integers(0, len(eligible)))]
            start = int(self.rng.integers(0, ep.steps - batch_length + 1))
            end = start + batch_length
            obs_batch.append(ep.obs[start : end + 1])
            action_batch.append(ep.actions[start:end])
            reward_batch.append(ep.rewards[start:end])
            done_batch.append(ep.dones[start:end])

        return {
            "obs": np.asarray(obs_batch, dtype=np.float32),
            "actions": np.asarray(action_batch, dtype=np.float32),
            "rewards": np.asarray(reward_batch, dtype=np.float32),
            "dones": np.asarray(done_batch, dtype=np.float32),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity_steps": self.capacity_steps,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "episodes": [
                {
                    "obs": ep.obs,
                    "actions": ep.actions,
                    "rewards": ep.rewards,
                    "dones": ep.dones,
                }
                for ep in self.episodes
            ],
            "total_steps": self.total_steps,
        }

