"""MuJoCo trajectory generation for SmallWorld-MJ."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from smallworld_mj.envs import SmallWorldMJEnv
from smallworld_mj.tasks import get_task, max_dims, taskpack


SPLITS = ("train", "val", "test", "ood")


def smooth_random_actions(rng: np.random.Generator, steps: int, action_dim: int) -> np.ndarray:
    actions = np.zeros((steps, action_dim), dtype=np.float32)
    cur = np.zeros(action_dim, dtype=np.float32)
    for t in range(steps):
        eps = rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
        cur = np.clip(0.8 * cur + 0.2 * eps, -1.0, 1.0)
        actions[t] = cur
    return actions


def rollout_episode(env: SmallWorldMJEnv, rng: np.random.Generator, steps: int, *, ood: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = env.reset(rng, ood=ood)
    actions = smooth_random_actions(rng, steps, env.spec.action_dim)
    states = [state]
    for t in range(steps):
        result = env.step(actions[t])
        states.append(result.state)
    return np.asarray(states, dtype=np.float32), actions, env.params.astype(np.float32)


def _pad_episode(
    states: np.ndarray,
    actions: np.ndarray,
    params: np.ndarray,
    *,
    max_state_dim: int,
    max_action_dim: int,
    max_param_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ps = np.zeros((states.shape[0], max_state_dim), dtype=np.float32)
    pa = np.zeros((actions.shape[0], max_action_dim), dtype=np.float32)
    pp = np.zeros((max_param_dim,), dtype=np.float32)
    sm = np.zeros((max_state_dim,), dtype=np.float32)
    am = np.zeros((max_action_dim,), dtype=np.float32)
    pm = np.zeros((max_param_dim,), dtype=np.float32)
    ps[:, : states.shape[-1]] = states
    pa[:, : actions.shape[-1]] = actions
    pp[: params.shape[-1]] = params
    sm[: states.shape[-1]] = 1.0
    am[: actions.shape[-1]] = 1.0
    pm[: params.shape[-1]] = 1.0
    return ps, pa, pp, sm, am, pm


def generate_split(
    tasks: list[str],
    split: str,
    *,
    episodes_per_task: int,
    episode_steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    max_state_dim, max_action_dim, max_param_dim = max_dims(tasks)
    states_out = []
    actions_out = []
    params_out = []
    task_ids = []
    state_masks = []
    action_masks = []
    param_masks = []
    episode_ids = []
    task_names = []
    for task_name in tasks:
        spec = get_task(task_name)
        env = SmallWorldMJEnv(spec)
        for ep in range(episodes_per_task):
            states, actions, params = rollout_episode(env, rng, episode_steps, ood=(split == "ood"))
            ps, pa, pp, sm, am, pm = _pad_episode(
                states,
                actions,
                params,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
                max_param_dim=max_param_dim,
            )
            states_out.append(ps)
            actions_out.append(pa)
            params_out.append(pp)
            task_ids.append(spec.task_id)
            state_masks.append(sm)
            action_masks.append(am)
            param_masks.append(pm)
            episode_ids.append(len(episode_ids))
            task_names.append(task_name)
    return {
        "states": np.asarray(states_out, dtype=np.float32),
        "actions": np.asarray(actions_out, dtype=np.float32),
        "task_params": np.asarray(params_out, dtype=np.float32),
        "task_id": np.asarray(task_ids, dtype=np.int64),
        "state_mask": np.asarray(state_masks, dtype=np.float32),
        "action_mask": np.asarray(action_masks, dtype=np.float32),
        "param_mask": np.asarray(param_masks, dtype=np.float32),
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
        "task_name": np.asarray(task_names),
    }


def split_path(dataset_dir: str | Path, split: str) -> Path:
    return Path(dataset_dir) / f"{split}.npz"


def save_split(dataset_dir: str | Path, split: str, data: dict[str, np.ndarray]) -> Path:
    path = split_path(dataset_dir, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path


def load_split(dataset_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    path = split_path(dataset_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def generate_dataset(config: dict[str, Any], *, taskpack_name: str, profile: str, output_dir: str | Path) -> dict[str, Any]:
    tasks = taskpack(taskpack_name)
    profile_cfg = config["profiles"][profile]
    steps = int(config["dataset"]["episode_steps"])
    output_dir = Path(output_dir)
    written = {}
    summaries = {}
    for i, split in enumerate(SPLITS):
        split_data = generate_split(
            tasks,
            split,
            episodes_per_task=int(profile_cfg["split_episodes"][split]),
            episode_steps=steps,
            seed=int(config["seed"]) + 1009 * i,
        )
        path = save_split(output_dir, split, split_data)
        written[split] = str(path)
        summaries[split] = {k: list(v.shape) for k, v in split_data.items() if hasattr(v, "shape")}
    metadata = {
        "benchmark": "SmallWorld-MJ",
        "taskpack": taskpack_name,
        "tasks": tasks,
        "profile": profile,
        "episode_steps": steps,
        "splits": summaries,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return {"output_dir": str(output_dir), "written": written, "metadata": metadata}
