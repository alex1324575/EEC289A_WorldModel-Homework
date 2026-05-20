"""Student open-loop rollout implementation."""

from __future__ import annotations

import torch

from wm_hw.model_utils import predict_next


def open_loop_rollout(model, states: torch.Tensor, actions: torch.Tensor, normalizer, warmup_steps: int, horizon: int):
    """Roll out `horizon` steps after a ground-truth warmup.

    Future ground-truth states after `warmup_steps` must not be read.
    """
    batch_size = states.shape[0]
    hidden = model.initial_hidden(batch_size, states.device)
    for t in range(int(warmup_steps)):
        _, hidden = predict_next(model, states[:, t], actions[:, t], hidden, normalizer)
    cur = states[:, int(warmup_steps)]
    preds = []
    for h in range(int(horizon)):
        cur, hidden = predict_next(model, cur, actions[:, int(warmup_steps) + h], hidden, normalizer)
        preds.append(cur)
    return torch.stack(preds, dim=1)


def truncated_bptt_rollout(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    chunk_size: int,
) -> torch.Tensor:
    """Open-loop rollout with truncated BPTT.

    Identical to open_loop_rollout except that every `chunk_size` rollout steps
    the GRU hidden state is detached, limiting gradient flow through the recurrent
    connection to at most `chunk_size` steps. `cur` (the state prediction) is NOT
    detached — state gradients still propagate across chunk boundaries, keeping the
    full training signal for the output trajectory.

    No ground-truth states are read after `warmup_steps`. The no-leak contract
    from open_loop_rollout is preserved.

    Returns: [B, horizon, D] predicted states.
    """
    hidden = model.initial_hidden(states.shape[0], states.device)
    for t in range(int(warmup_steps)):
        _, hidden = predict_next(model, states[:, t], actions[:, t], hidden, normalizer)

    cur = states[:, int(warmup_steps)]
    preds = []
    for h in range(int(horizon)):
        # Detach hidden at chunk boundaries (not at h=0 — hidden was just set by warmup).
        if h > 0 and h % int(chunk_size) == 0 and hidden is not None:
            hidden = hidden.detach()
        cur, hidden = predict_next(model, cur, actions[:, int(warmup_steps) + h], hidden, normalizer)
        preds.append(cur)
    return torch.stack(preds, dim=1)
