#!/usr/bin/env python3
"""Create a quick open-loop world-model prediction plot from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from world_model_hw.checkpointing import load_checkpoint
from world_model_hw.config import choose_device, save_json, set_runtime_env
from world_model_hw.envs import get_env_info, make_env, reset_env, step_env
from world_model_hw.models import RSSMState
from world_model_hw.visualization import save_prediction_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=30)
    return parser.parse_args()


@torch.no_grad()
def collect_episode(agent, config, env, env_info, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    obs = reset_env(env, seed=int(config["seed"]) + 9000)
    state = None
    prev_action = None
    obs_list = [obs]
    actions = []
    for _ in range(horizon):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
        action_tensor, state = agent.act(obs_tensor, state, prev_action, deterministic=True)
        action = action_tensor.detach().cpu().numpy()
        obs, _, done, _ = step_env(env, action, env_info)
        prev_action = action_tensor.reshape(1, -1).detach()
        actions.append(action)
        obs_list.append(obs)
        if done:
            break
    return np.asarray(obs_list, dtype=np.float32), np.asarray(actions, dtype=np.float32)


@torch.no_grad()
def predict_open_loop(agent, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    if len(actions) == 0:
        return np.zeros((0, agent.obs_dim), dtype=np.float32)
    obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=agent.device)
    act_t = torch.as_tensor(actions[None], dtype=torch.float32, device=agent.device)
    rssm_out = agent.world_model.rssm.observe(obs_t, act_t, sample=False)
    state = RSSMState(
        deter=rssm_out["post_deter"][:, 0],
        stoch=rssm_out["post_stoch"][:, 0],
        mean=rssm_out["post_mean"][:, 0],
        std=rssm_out["post_std"][:, 0],
    )
    preds = []
    for action in act_t.squeeze(0):
        state = agent.world_model.rssm.img_step(state, action.reshape(1, -1), sample=False)
        feat = agent.world_model.rssm.get_feat(state)
        preds.append(agent.world_model.decoder(feat).squeeze(0).detach().cpu().numpy())
    return np.asarray(preds, dtype=np.float32)


def main() -> None:
    args = parse_args()
    set_runtime_env()
    output_dir = args.output_dir or args.checkpoint_dir.parent / "quick_world_model_check"
    device = choose_device(args.device)
    agent, config, _ = load_checkpoint(args.checkpoint_dir, device)
    env = make_env(config, seed=int(config["seed"]) + 9000)
    env_info = get_env_info(env)
    obs, actions = collect_episode(agent, config, env, env_info, args.horizon)
    pred = predict_open_loop(agent, obs, actions)
    target = obs[1 : 1 + len(pred)]
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "world_model_rollout.png"
    save_prediction_plot(plot_path, target=target, prediction=pred)
    rmse = float(np.sqrt(np.mean((pred - target) ** 2))) if len(pred) else float("nan")
    result = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "plot_path": str(plot_path),
        "horizon": int(len(pred)),
        "open_loop_obs_rmse": rmse,
    }
    save_json(output_dir / "quick_world_model_check.json", result)
    print(result)
    env.close()


if __name__ == "__main__":
    main()
