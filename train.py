#!/usr/bin/env python3
"""Train the MiniDreamer homework baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model_hw.agent import MiniDreamerAgent
from world_model_hw.checkpointing import save_checkpoint
from world_model_hw.config import (
    DEFAULT_CONFIG_PATH,
    apply_stage_overrides,
    choose_device,
    load_json,
    save_json,
    set_global_seeds,
    set_runtime_env,
    summarize_config,
)
from world_model_hw.envs import get_env_info, make_env, reset_env, step_env
from world_model_hw.replay import EpisodeReplay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--stage", choices=["baseline", "local_smoke"], default="baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/run_baseline"))
    parser.add_argument("--device", type=str, default=None, help="Override torch device, for example cpu or cuda.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env-steps", type=int, default=None, help="Override the selected stage step budget.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--local-smoke", action="store_true", help="Shortcut for --stage local_smoke.")
    return parser.parse_args()


def random_action(action_dim: int) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)


def evaluate(agent: MiniDreamerAgent, config: dict[str, Any], env_info, *, episodes: int, seed: int) -> dict[str, float]:
    env = make_env(config, seed=seed)
    returns = []
    lengths = []
    for ep in range(int(episodes)):
        obs = reset_env(env, seed=seed + ep)
        state = None
        prev_action = None
        total = 0.0
        for t in range(int(config["env"]["max_episode_steps"])):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            action_tensor, state = agent.act(obs_tensor, state, prev_action, deterministic=True)
            action = action_tensor.detach().cpu().numpy()
            next_obs, reward, done, _info = step_env(env, action, env_info)
            prev_action = action_tensor.reshape(1, -1).detach()
            obs = next_obs
            total += reward
            if done:
                break
        returns.append(total)
        lengths.append(t + 1)
    env.close()
    return {
        "eval/mean_return": float(np.mean(returns)),
        "eval/std_return": float(np.std(returns)),
        "eval/mean_length": float(np.mean(lengths)),
    }


def collect_random(env, env_info, replay: EpisodeReplay, steps: int, seed: int) -> int:
    obs = reset_env(env, seed=seed)
    replay.start_episode(obs)
    collected = 0
    while collected < steps:
        action = random_action(env_info.action_dim)
        next_obs, reward, done, _ = step_env(env, action, env_info)
        replay.add(action, reward, done, next_obs)
        collected += 1
        obs = next_obs
        if done:
            obs = reset_env(env)
            replay.start_episode(obs)
    return collected


def collect_policy(
    env,
    env_info,
    replay: EpisodeReplay,
    agent: MiniDreamerAgent,
    *,
    steps: int,
    exploration_std: float,
) -> int:
    obs = reset_env(env)
    replay.start_episode(obs)
    state = None
    prev_action = None
    collected = 0
    while collected < steps:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
        action_tensor, state = agent.act(obs_tensor, state, prev_action, deterministic=False)
        action = action_tensor.detach().cpu().numpy()
        if exploration_std > 0:
            action = np.clip(action + np.random.normal(0.0, exploration_std, size=action.shape), -1.0, 1.0).astype(np.float32)
        next_obs, reward, done, _ = step_env(env, action, env_info)
        replay.add(action, reward, done, next_obs)
        collected += 1
        obs = next_obs
        prev_action = torch.as_tensor(action, dtype=torch.float32, device=agent.device).reshape(1, -1)
        if done:
            obs = reset_env(env)
            replay.start_episode(obs)
            state = None
            prev_action = None
    return collected


def train(config: dict[str, Any], output_dir: Path, device: str) -> dict[str, Any]:
    seed = int(config["seed"])
    set_global_seeds(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)

    env = make_env(config, seed=seed)
    env_info = get_env_info(env)
    replay = EpisodeReplay(
        capacity_steps=int(config["replay"]["capacity_steps"]),
        obs_dim=env_info.obs_dim,
        action_dim=env_info.action_dim,
        seed=seed,
    )
    agent = MiniDreamerAgent(env_info.obs_dim, env_info.action_dim, config, device)

    stage_cfg = config["training_stages"][config["active_stage"]]
    prefill_steps = int(config["replay"]["prefill_steps"])
    env_steps = int(stage_cfg["env_steps"])
    collect_steps = int(stage_cfg["collect_steps_per_iter"])
    updates_per_iter = int(stage_cfg["train_updates_per_iter"])
    batch_size = int(config["replay"]["batch_size"])
    batch_length = int(config["replay"]["batch_length"])

    print(f"[train] device={device} stage={config['active_stage']} env_steps={env_steps}", flush=True)
    print(f"[train] prefill random steps={prefill_steps}", flush=True)
    collected = collect_random(env, env_info, replay, prefill_steps, seed)
    total_env_steps = collected
    best_return = -float("inf")
    progress: list[dict[str, Any]] = []
    start_time = time.monotonic()

    while total_env_steps < env_steps:
        total_env_steps += collect_policy(
            env,
            env_info,
            replay,
            agent,
            steps=min(collect_steps, env_steps - total_env_steps),
            exploration_std=float(stage_cfg["exploration_std"]),
        )

        metrics: dict[str, float] = {}
        if replay.can_sample(batch_length):
            for _ in range(updates_per_iter):
                batch = replay.sample(batch_size=batch_size, batch_length=batch_length)
                metrics.update(agent.train_world_model(batch))
                metrics.update(agent.train_actor_critic(batch))

        should_eval = (
            total_env_steps == env_steps
            or total_env_steps % int(stage_cfg["eval_every_env_steps"]) < collect_steps
        )
        if should_eval:
            eval_metrics = evaluate(
                agent,
                config,
                env_info,
                episodes=int(stage_cfg["eval_episodes"]),
                seed=seed + 10_000 + total_env_steps,
            )
            metrics.update(eval_metrics)
            if eval_metrics["eval/mean_return"] > best_return:
                best_return = eval_metrics["eval/mean_return"]
                save_checkpoint(
                    output_dir / "best_checkpoint",
                    agent=agent,
                    config=config,
                    obs_dim=env_info.obs_dim,
                    action_dim=env_info.action_dim,
                    metrics=metrics,
                    step=total_env_steps,
                )

        record = {
            "env_steps": int(total_env_steps),
            "replay_steps": int(len(replay)),
            "elapsed_seconds": float(time.monotonic() - start_time),
            "metrics": metrics,
        }
        progress.append(record)
        save_json(output_dir / "progress_live.json", progress)
        if metrics:
            short = " ".join(f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, (int, float)))
            print(f"[train] steps={total_env_steps} {short}", flush=True)
        else:
            print(f"[train] steps={total_env_steps}", flush=True)

    final_metrics = evaluate(agent, config, env_info, episodes=int(stage_cfg["eval_episodes"]), seed=seed + 999_000)
    save_checkpoint(
        output_dir / "latest_checkpoint",
        agent=agent,
        config=config,
        obs_dim=env_info.obs_dim,
        action_dim=env_info.action_dim,
        metrics=final_metrics,
        step=total_env_steps,
    )
    if not (output_dir / "best_checkpoint" / "checkpoint.pt").exists():
        save_checkpoint(
            output_dir / "best_checkpoint",
            agent=agent,
            config=config,
            obs_dim=env_info.obs_dim,
            action_dim=env_info.action_dim,
            metrics=final_metrics,
            step=total_env_steps,
        )
    summary = {
        "stage": config["active_stage"],
        "env_steps": int(total_env_steps),
        "best_return": float(best_return),
        "final_metrics": final_metrics,
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    env.close()
    return summary


def main() -> None:
    args = parse_args()
    set_runtime_env()
    stage = "local_smoke" if args.local_smoke else args.stage
    config = load_json(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    config = apply_stage_overrides(config, stage)
    if args.env_steps is not None:
        config["training_stages"][stage]["env_steps"] = int(args.env_steps)
    device = choose_device(args.device)

    if args.print_config or args.dry_run:
        save_json(args.output_dir / "dry_run_resolved_config.json", config)
        print(summarize_config(config))
    if args.dry_run:
        env = make_env(config, seed=int(config["seed"]))
        info = get_env_info(env)
        agent = MiniDreamerAgent(info.obs_dim, info.action_dim, config, device)
        print(f"[dry-run] obs_dim={info.obs_dim} action_dim={info.action_dim} device={device}")
        print(f"[dry-run] parameters={sum(p.numel() for p in agent.world_model.parameters()) + sum(p.numel() for p in agent.actor.parameters()) + sum(p.numel() for p in agent.critic.parameters())}")
        env.close()
        return

    summary = train(config, args.output_dir, device)
    print(summary)


if __name__ == "__main__":
    main()
