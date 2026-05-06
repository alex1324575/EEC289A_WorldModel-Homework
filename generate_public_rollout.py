#!/usr/bin/env python3
"""Generate the standardized public evaluation rollout bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from benchmark_specs import public_eval_seeds
from world_model_hw.checkpointing import load_checkpoint
from world_model_hw.config import choose_device, save_json, set_runtime_env
from world_model_hw.envs import get_env_info, make_env, reset_env, step_env
from world_model_hw.models import RSSMState
from world_model_hw.visualization import save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--render-first-episode", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def prior_predictions(agent, obs: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, torch.Tensor]]:
    obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=agent.device)
    act_t = torch.as_tensor(actions[None], dtype=torch.float32, device=agent.device)
    pred = agent.world_model.predict_prior(obs_t, act_t)
    return (
        pred["pred_obs"].squeeze(0).detach().cpu().numpy(),
        pred["pred_reward"].squeeze(0).detach().cpu().numpy(),
        pred["rssm_out"],
    )


@torch.no_grad()
def open_loop_predictions(agent, actions: np.ndarray, rssm_out: dict[str, torch.Tensor], horizon: int) -> np.ndarray:
    num_steps = actions.shape[0]
    obs_dim = agent.obs_dim
    preds = np.full((num_steps, horizon, obs_dim), np.nan, dtype=np.float32)
    act_t = torch.as_tensor(actions, dtype=torch.float32, device=agent.device)
    for t in range(num_steps):
        if t == 0:
            state = agent.world_model.rssm.initial(1, agent.device)
        else:
            idx = t - 1
            state = RSSMState(
                deter=rssm_out["post_deter"][:, idx],
                stoch=rssm_out["post_stoch"][:, idx],
                mean=rssm_out["post_mean"][:, idx],
                std=rssm_out["post_std"][:, idx],
            )
        for h in range(horizon):
            if t + h >= num_steps:
                break
            action = act_t[t + h : t + h + 1]
            state = agent.world_model.rssm.img_step(state, action, sample=False)
            feat = agent.world_model.rssm.get_feat(state)
            pred_obs = agent.world_model.decoder(feat).squeeze(0).detach().cpu().numpy()
            preds[t, h] = pred_obs.astype(np.float32)
    return preds


def run_episode(agent, config, env, env_info, seed: int, *, render: bool) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    obs = reset_env(env, seed=seed)
    state = None
    prev_action = None
    obs_list = [obs]
    actions = []
    rewards = []
    frames = []
    total = 0.0
    for _ in range(int(config["env"]["max_episode_steps"])):
        if render:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
        action_tensor, state = agent.act(obs_tensor, state, prev_action, deterministic=True)
        action = action_tensor.detach().cpu().numpy()
        next_obs, reward, done, _ = step_env(env, action, env_info)
        prev_action = action_tensor.reshape(1, -1).detach()
        actions.append(action.astype(np.float32))
        rewards.append(float(reward))
        obs_list.append(next_obs)
        obs = next_obs
        total += reward
        if done:
            break

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    action_arr = np.asarray(actions, dtype=np.float32)
    reward_arr = np.asarray(rewards, dtype=np.float32)
    pred_obs, pred_reward, rssm_out = prior_predictions(agent, obs_arr, action_arr)
    horizon = int(config["public_eval"]["open_loop_horizon"])
    open_loop_pred = open_loop_predictions(agent, action_arr, rssm_out, horizon)
    open_loop_target = np.full_like(open_loop_pred, np.nan)
    for t in range(action_arr.shape[0]):
        for h in range(horizon):
            if t + h + 1 < obs_arr.shape[0]:
                open_loop_target[t, h] = obs_arr[t + h + 1]

    if len(action_arr) == 0:
        action_delta = np.zeros((0,), dtype=np.float32)
    else:
        prev = np.concatenate([action_arr[:1], action_arr[:-1]], axis=0)
        action_delta = np.linalg.norm(action_arr - prev, axis=-1).astype(np.float32)

    return (
        {
            "obs": obs_arr[:-1],
            "action": action_arr,
            "reward": reward_arr,
            "pred_obs_1step": pred_obs.astype(np.float32),
            "pred_reward_1step": pred_reward.astype(np.float32),
            "open_loop_obs_pred": open_loop_pred.astype(np.float32),
            "open_loop_obs_target": open_loop_target.astype(np.float32),
            "eval_return": np.full((action_arr.shape[0],), total, dtype=np.float32),
            "action_delta": action_delta,
        },
        frames,
    )


def main() -> None:
    args = parse_args()
    set_runtime_env()
    device = choose_device(args.device)
    agent, config, _ = load_checkpoint(args.checkpoint_dir, device)
    num_episodes = args.num_episodes or int(config["public_eval"]["num_episodes"])
    render_mode = "rgb_array" if args.render_first_episode else None
    env = make_env(config, seed=int(config["seed"]) + 7000, render_mode=render_mode)
    env_info = get_env_info(env)

    bundles: dict[str, list[np.ndarray]] = {
        "episode_id": [],
        "step": [],
        "obs": [],
        "action": [],
        "reward": [],
        "pred_obs_1step": [],
        "pred_reward_1step": [],
        "open_loop_obs_pred": [],
        "open_loop_obs_target": [],
        "eval_return": [],
        "action_delta": [],
    }
    frames = []
    for ep_idx, seed in enumerate(public_eval_seeds(num_episodes)):
        episode, ep_frames = run_episode(agent, config, env, env_info, seed, render=args.render_first_episode and ep_idx == 0)
        steps = episode["action"].shape[0]
        bundles["episode_id"].append(np.full((steps,), ep_idx, dtype=np.int32))
        bundles["step"].append(np.arange(steps, dtype=np.int32))
        for key, value in episode.items():
            bundles[key].append(value)
        if ep_frames:
            frames.extend(ep_frames)

    output = {key: np.concatenate(value, axis=0) for key, value in bundles.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = args.output_dir / "rollout_public_eval.npz"
    np.savez_compressed(rollout_path, **output)
    video_path = save_video(frames, args.output_dir / "public_eval_episode0.mp4") if frames else None
    summary = {
        "rollout_npz": str(rollout_path),
        "num_episodes": int(num_episodes),
        "num_steps": int(output["step"].shape[0]),
        "video_path": str(video_path) if video_path else None,
        "fields": {key: list(value.shape) for key, value in output.items()},
    }
    save_json(args.output_dir / "rollout_summary.json", summary)
    print(summary)
    env.close()


if __name__ == "__main__":
    main()
