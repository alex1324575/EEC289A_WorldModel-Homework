"""Dataset generation and loading utilities for SmallWorld-Lite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .tasks import SmallWorldTask, get_task, list_tasks


SPLITS = ("train", "val", "test", "ood")


def task_names_from_arg(task_arg: str) -> list[str]:
    if task_arg == "all":
        return list_tasks()
    return [get_task(task_arg).name]


def split_path(dataset_dir: Path, task_name: str, split: str) -> Path:
    return dataset_dir / task_name / f"{split}.npz"


def make_action_sequence(rng: np.random.Generator, steps: int, action_dim: int) -> np.ndarray:
    """Create smooth random actions so long-horizon prediction is diagnosable."""
    actions = np.zeros((steps, action_dim), dtype=np.float32)
    current = rng.uniform(-0.35, 0.35, size=(action_dim,))
    for t in range(steps):
        current = 0.92 * current + 0.08 * rng.uniform(-1.0, 1.0, size=(action_dim,))
        actions[t] = np.clip(current, -1.0, 1.0)
    return actions


def rollout_episode(
    task: SmallWorldTask,
    *,
    steps: int,
    rng: np.random.Generator,
    ood: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, params = task.reset(rng, ood=ood)
    actions = make_action_sequence(rng, steps, task.action_dim)
    states = [state]
    for t in range(steps):
        state = task.step(state, actions[t], params)
        states.append(state)
    return np.asarray(states, dtype=np.float32), actions.astype(np.float32), params.astype(np.float32)


def generate_split(
    task: SmallWorldTask,
    *,
    split: str,
    episodes: int,
    steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if split not in SPLITS:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {SPLITS}.")
    rng = np.random.default_rng(seed)
    states = []
    actions = []
    params = []
    for _ in range(int(episodes)):
        ep_states, ep_actions, ep_params = rollout_episode(task, steps=int(steps), rng=rng, ood=(split == "ood"))
        states.append(ep_states)
        actions.append(ep_actions)
        params.append(ep_params)
    full_states = np.asarray(states, dtype=np.float32)
    action_arr = np.asarray(actions, dtype=np.float32)
    episode_ids = np.repeat(np.arange(int(episodes), dtype=np.int32)[:, None], int(steps), axis=1)
    step_ids = np.repeat(np.arange(int(steps), dtype=np.int32)[None, :], int(episodes), axis=0)
    return {
        "states": full_states,
        "state": full_states[:, :-1],
        "action": action_arr,
        "next_state": full_states[:, 1:],
        "task_params": np.asarray(params, dtype=np.float32),
        "episode_id": episode_ids,
        "step": step_ids,
        "task_name": np.asarray(task.name),
        "split": np.asarray(split),
    }


def save_split(dataset_dir: Path, task_name: str, split: str, data: dict[str, np.ndarray]) -> Path:
    path = split_path(dataset_dir, task_name, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path


def load_split(dataset_dir: Path, task_name: str, split: str) -> dict[str, np.ndarray]:
    path = split_path(dataset_dir, task_name, split)
    if not path.exists():
        raise FileNotFoundError(f"SmallWorld split not found: {path}")
    return dict(np.load(path, allow_pickle=False))


def generate_task_dataset(
    dataset_dir: Path,
    task_name: str,
    config: dict[str, Any],
    *,
    local_smoke: bool = False,
) -> dict[str, str]:
    task = get_task(task_name)
    data_cfg = config["dataset"]
    split_counts = dict(data_cfg["split_episodes"])
    steps = int(data_cfg["episode_steps"])
    if local_smoke:
        smoke_cfg = config["smoke"]
        split_counts = dict(smoke_cfg["split_episodes"])
        steps = int(smoke_cfg["episode_steps"])
    written = {}
    base_seed = int(config["seed"]) + 1000 * list_tasks().index(task.name)
    for split_idx, split in enumerate(SPLITS):
        data = generate_split(
            task,
            split=split,
            episodes=int(split_counts[split]),
            steps=steps,
            seed=base_seed + split_idx * 97,
        )
        written[split] = str(save_split(dataset_dir, task.name, split, data))
    return written


def sample_batch(
    data: dict[str, np.ndarray],
    *,
    batch_size: int,
    batch_length: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["action"], dtype=np.float32)
    episodes, steps_plus_one, _state_dim = states.shape
    steps = steps_plus_one - 1
    if steps < batch_length:
        raise ValueError(f"Dataset has only {steps} steps, shorter than batch_length={batch_length}.")
    ep_idx = rng.integers(0, episodes, size=int(batch_size))
    starts = rng.integers(0, steps - int(batch_length) + 1, size=int(batch_size))
    state_batch = []
    action_batch = []
    for ep, start in zip(ep_idx, starts):
        end = int(start) + int(batch_length)
        state_batch.append(states[int(ep), int(start) : end + 1])
        action_batch.append(actions[int(ep), int(start) : end])
    return {
        "states": np.asarray(state_batch, dtype=np.float32),
        "actions": np.asarray(action_batch, dtype=np.float32),
    }


def dataset_summary(dataset_dir: Path, task_name: str) -> dict[str, Any]:
    rows = {}
    for split in SPLITS:
        data = load_split(dataset_dir, task_name, split)
        rows[split] = {
            "states": list(data["states"].shape),
            "action": list(data["action"].shape),
            "task_params": list(data["task_params"].shape),
        }
    return rows
