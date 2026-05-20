"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from wm_hw.model_utils import predict_next


def one_step_delta_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    *,
    noise_sigma: float = 0.0,
) -> torch.Tensor:
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    if noise_sigma > 0.0 and model.training:
        obs_std = torch.as_tensor(normalizer.obs_std, dtype=obs.dtype, device=obs.device)
        obs = obs + torch.randn_like(obs) * (noise_sigma * obs_std)
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.mse_loss(pred_norm, target_norm)


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    *,
    noise_sigma: float = 0.0,
    chunk_size: int = 0,
) -> torch.Tensor:
    # Train local open-loop stability at random positions, not only at the
    # beginning of each stored window.
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]

    # Warmup: feed ground-truth states (with optional noise) to prime hidden state.
    # Noise on warmup inputs simulates the small deviations from ground truth that
    # accumulate during open-loop rollout, teaching robustness to drift.
    # Rollout predictions are NOT noised — they are already closed-loop.
    hidden = model.initial_hidden(sub_states.shape[0], sub_states.device)
    add_noise = noise_sigma > 0.0 and model.training
    if add_noise:
        obs_std = torch.as_tensor(normalizer.obs_std, dtype=sub_states.dtype, device=sub_states.device)
    for t in range(int(warmup_steps)):
        obs_t = sub_states[:, t]
        if add_noise:
            obs_t = obs_t + torch.randn_like(obs_t) * (noise_sigma * obs_std)
        _, hidden = predict_next(model, obs_t, sub_actions[:, t], hidden, normalizer)

    cur = sub_states[:, int(warmup_steps)]
    preds = []
    for h in range(int(horizon)):
        # Truncated BPTT: detach hidden every chunk_size steps to bound the
        # gradient path through the GRU. cur is NOT detached — state gradients
        # still flow across chunk boundaries, preserving the output loss signal.
        # chunk_size=0 disables truncation (default, backward-compatible).
        if chunk_size > 0 and h > 0 and h % chunk_size == 0 and hidden is not None:
            hidden = hidden.detach()
        cur, hidden = predict_next(model, cur, sub_actions[:, int(warmup_steps) + h], hidden, normalizer)
        preds.append(cur)

    preds_t = torch.stack(preds, dim=1)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds_t)
    target_norm = normalizer.normalize_obs(targets)
    return F.mse_loss(pred_norm, target_norm)


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    # Lazy-init physics buffers for the acceleration-prediction model (Exp4).
    # Must run before any model.forward call. Uses register_buffer so values
    # follow model.to(device) automatically after this first call.
    # accel_scale[i] ≈ 5σ of training acceleration: delta_std[2,3] = a*dt → scale = 5*delta_std/dt
    if hasattr(model, "accel_scale") and not getattr(model, "_buffers_initialized", False):
        dev = model.accel_scale.device
        model.obs_std.copy_(torch.as_tensor(normalizer.obs_std,    dtype=torch.float32, device=dev))
        model.obs_mean.copy_(torch.as_tensor(normalizer.obs_mean,   dtype=torch.float32, device=dev))
        model.delta_std.copy_(torch.as_tensor(normalizer.delta_std,  dtype=torch.float32, device=dev))
        model.delta_mean.copy_(torch.as_tensor(normalizer.delta_mean, dtype=torch.float32, device=dev))
        ax = float(5.0 * normalizer.delta_std[2] / model.dt)
        ap = float(5.0 * normalizer.delta_std[3] / model.dt)
        model.accel_scale.copy_(torch.tensor([ax, ap], dtype=torch.float32, device=dev))
        model._buffers_initialized = True
        print(f"[accel-model lazy-init] dt={model.dt}, accel_scale=[{ax:.4f}, {ap:.4f}]")

    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    sigma = float(loss_cfg.get("input_noise_sigma", 0.0))
    one = one_step_delta_loss(model, states, actions, normalizer, noise_sigma=sigma)
    horizon = int(loss_cfg.get("rollout_train_horizon", 5))
    warmup = int(cfg["eval"].get("warmup_steps", 5))
    # Clip to what the batch supports; smoke/short datasets have fewer steps than
    # the configured horizon, and the full scoreboard dataset always satisfies it.
    horizon = min(horizon, states.shape[1] - warmup - 1)
    chunk_size = int(loss_cfg.get("bptt_chunk_size", 0))
    roll = rollout_loss(model, states, actions, normalizer, warmup_steps=warmup, horizon=horizon, noise_sigma=sigma, chunk_size=chunk_size)
    total = float(loss_cfg.get("one_step_weight", 1.0)) * one + float(loss_cfg.get("rollout_weight", 0.3)) * roll
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }
