#!/usr/bin/env python3
"""Restore a MiniDreamer checkpoint and evaluate its policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from world_model_hw.checkpointing import load_checkpoint
from world_model_hw.config import choose_device, save_json, set_runtime_env
from world_model_hw.envs import get_env_info, make_env, reset_env, step_env
from world_model_hw.visualization import save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_runtime_env()
    device = choose_device(args.device)
    agent, config, payload = load_checkpoint(args.checkpoint_dir, device)
    render_mode = "rgb_array" if args.render else None
    env = make_env(config, seed=int(config["seed"]) + 5000, render_mode=render_mode)
    info = get_env_info(env)

    returns = []
    lengths = []
    frames = []
    for ep in range(args.num_episodes):
        obs = reset_env(env, seed=int(config["seed"]) + 5000 + ep)
        state = None
        prev_action = None
        total = 0.0
        for t in range(int(config["env"]["max_episode_steps"])):
            if args.render and ep == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            action_tensor, state = agent.act(obs_tensor, state, prev_action, deterministic=True)
            action = action_tensor.detach().cpu().numpy()
            next_obs, reward, done, _ = step_env(env, action, info)
            prev_action = action_tensor.reshape(1, -1).detach()
            obs = next_obs
            total += reward
            if done:
                break
        returns.append(total)
        lengths.append(t + 1)

    video_path = None
    if args.render:
        video_path = save_video(frames, args.output_dir / "demo_policy.mp4", fps=30)
    result = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_step": int(payload.get("step", -1)),
        "num_episodes": int(args.num_episodes),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "returns": [float(x) for x in returns],
        "video_path": str(video_path) if video_path else None,
    }
    save_json(args.output_dir / "eval_metrics.json", result)
    print(result)
    env.close()


if __name__ == "__main__":
    main()
