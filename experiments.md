# Experiment Log — EEC289A WorldModel Homework

---

## Experiment 1: P1+P2+P3 baseline (2026-05-19)

### Hypothesis
The failed previous attempt (test VPT80@0.25=18, OOD=4) suffered from three
compounding problems: training horizon 15 vs eval horizon 1000, no physics
inductive bias, and insufficient updates/capacity. This run fixes all three:

- **P1 (Config)**: Switch to `public_scoreboard` dataset (max_horizon=1000),
  increase `rollout_train_horizon` 15→150, `train_sequence_length` 64→256,
  `updates` 2000→12000. Train the model to be stable across horizons it will
  actually face at eval time.
- **P2 (Inductive Bias)**: Add `sin(obs_norm[:,1])` and `cos(obs_norm[:,1])`
  to encoder input. Primary value: during open-loop rollout the raw angle is
  unbounded, sin/cos soft-clamps it to [-1,1] giving the model a bounded
  angle representation even when drift accumulates. Physical exactness is
  secondary (we use the normalized angle, not the raw one).
- **P3 (Architecture)**: Enable GRU (`use_gru: true`), increase
  `hidden_dim` 128→192. GRU absorbs accumulated rollout error via its
  hidden state. No dropout yet — add only if test vs OOD gap is large.
- **P4/P5 (Hyperparams)**: LR 1e-3→3e-4 (stability for long-horizon
  gradients), `grad_clip_norm` 10→5, `rollout_weight` 1.0→2.0 (prioritize
  long-horizon stability).

Expected result: VPT80@0.25 (test) ≥ 80, OOD within 2× of test.

### Changed Files

**`configs/student.yaml`** — 9 values changed:

| Param | Before | After |
|---|---|---|
| `model.hidden_dim` | 128 | 192 |
| `model.use_gru` | false | true |
| `training.updates` | 2000 | 12000 |
| `training.train_sequence_length` | 64 | 256 |
| `training.learning_rate` | 1.0e-3 | 3.0e-4 |
| `training.grad_clip_norm` | 10.0 | 5.0 |
| `training.eval_every` | 200 | 500 |
| `loss.rollout_weight` | 1.0 | 2.0 |
| `loss.rollout_train_horizon` | 15 | 150 |

**`student/model.py`** — sin/cos angle augmentation in encoder input:
- `in_dim` changed from `obs_dim + act_dim` (5) to `obs_dim + act_dim + 2` (7)
- `forward`: computes `angle = obs_norm[:, 1:2]`, concatenates
  `[obs_norm, act_norm, sin(angle), cos(angle)]` before encoder

**`student/losses.py`** — horizon adaptive clipping:
- Added `horizon = min(horizon, states.shape[1] - warmup - 1)` before
  `rollout_loss` call. Ensures smoke runs (100-step dev dataset) don't crash
  when `rollout_train_horizon=150 > available steps`. Full scoreboard dataset
  always satisfies the configured horizon.

### Full Hyperparameter Table

| Param | Value |
|---|---|
| `hidden_dim` | 192 |
| `num_layers` | 2 |
| `use_gru` | true |
| `learning_rate` | 3.0e-4 |
| `rollout_train_horizon` | 150 |
| `train_sequence_length` | 256 |
| `updates` | 12000 |
| `rollout_weight` | 2.0 |
| `one_step_weight` | 1.0 |
| `grad_clip_norm` | 5.0 |
| `batch_size` | 128 |
| `dataset` | public_scoreboard (max_horizon=1000, train=1024 windows) |

### Results

**Pending Colab run.**

Metrics to fill in after training on `public_scoreboard` dataset:

| Split | VPT80@0.25 | VPT50@0.25 | nMSE@10 | nMSE@100 | nMSE@1000 | nMSE_AUC |
|---|---|---|---|---|---|---|
| test | — | — | — | — | — | — |
| ood | — | — | — | — | — | — |

Decision: keep / revert / iterate

---

## Experiment 2: Output clamp tightening (2026-05-20)


### Diagnosis
Profiling the training delta distribution revealed:

| Statistic | Value (normalized units σ) |
|---|---|
| abs-max | ±5.3 σ |
| 99.99th percentile | ±3.8 σ |
| 99th percentile | ~±2.0 σ |

The Exp1 model used `delta_limit=3.0` with the soft limiter
`delta = 3.0 * tanh(raw / 3.0)`. For `|raw| < 3`, `tanh(x/3) ≈ x/3`, so
the limiter was effectively `delta ≈ raw` — a no-op across almost the entire
training support. This allowed the model to output physically implausible
deltas during rollout (especially when GRU hidden state accumulated error),
causing the observed pole_angle spike at step 20–30.

### Fix

**`student/model.py`** only — two changes:

1. `delta_limit` default: `3.0` → `5.5`
   - Sets the tanh knee at 5.5σ, which is just above the training abs-max
     (5.3σ). The limiter is now near-linear within the training support but
     provides genuine soft saturation beyond it.
2. Hard clamp after tanh: `delta = delta.clamp(-6.0, 6.0)`
   - Final backstop at ±6.0σ. Should never activate during normal rollout;
     catches runaway divergence if hidden state drifts far out-of-distribution.

### Hypothesis
Capping outputs to a range the model was actually trained on should prevent
the step-20-30 collapse. Quantitatively: VPT80@0.25 should push well past 30.
The limiter change is architecture-only; no retraining hyperparams changed.
This commit can be stacked on top of Exp1 for the next Colab run.

### Changed Files

**`student/model.py`**:
- `delta_limit: float = 3.0` → `delta_limit: float = 5.5`
- Added after tanh: `delta = delta.clamp(-6.0, 6.0)`
- Added comment block quantifying the training distribution and explaining
  the two-stage limiting strategy

No changes to `configs/student.yaml`, `student/losses.py`, `student/rollout.py`.

### Results

**Pending Colab run** (stacked with Exp1 changes).

| Split | VPT80@0.25 | VPT50@0.25 | nMSE@10 | nMSE@100 | nMSE@1000 | nMSE_AUC |
|---|---|---|---|---|---|---|
| test | — | — | — | — | — | — |
| ood | — | — | — | — | — | — |

Decision: keep / revert / iterate

---

## Experiment 3: Input noise injection (2026-05-20)

### Diagnosis (from step-level eval of Exp1+2 checkpoint at update 11000)

| Step | nMSE |
|---|---|
| 10 | 0.0029 (very accurate) |
| 50 | 0.21 (collapse onset) |
| 300 | 0.45 |
| 500 | 1.27 |

Classic **rollout horizon mismatch + narrow training distribution**: the model
learned to be highly precise within its training distribution, but once
open-loop rollout drifts the state slightly off-manifold, the model has no
robustness signal and diverges catastrophically. The horizon mismatch fix
(Exp1) addresses *how long* rollout loss reaches; this experiment addresses
*how wide* the training distribution is.

### Hypothesis

Injecting small Gaussian noise `N(0, σ · obs_std)` on input observations
during training simulates the off-manifold states that accumulate during
open-loop rollout. The model is forced to learn a prediction function that
is smooth around the true trajectory, not just accurate on it.
This is a well-understood technique (trajectory smoothing / DAgger-style
data augmentation). With `σ=0.03`, the noise is ~3% of each dimension's
training standard deviation — small enough not to corrupt one-step accuracy,
large enough to extend the effective training distribution into the rollout
drift regime seen at step 50.

Expected result: nMSE@50 drops below 0.10; VPT80@0.25 increases substantially
when stacked with Exp1+2 changes.

### Implementation

**`student/losses.py`**:
- `one_step_delta_loss`: noise applied to all flattened `obs` before
  `normalize_obs`. Active only when `model.training=True`.
- `rollout_loss`: replaced `open_loop_rollout` call with an explicit warmup
  loop using `predict_next` directly (imported from `wm_hw.model_utils`).
  Noise applied to each `obs_t` during the warmup loop only; rollout
  predictions (`cur`) receive no noise — they are already in closed-loop
  drift, adding noise would double-count.
- `noise_sigma` is keyword-only with default `0.0` (backward compatible;
  old configs without `input_noise_sigma` get `sigma=0.0`).
- Per-dim scaling: `noise ~ randn_like(obs) * sigma * obs_std_tensor` where
  `obs_std_tensor = torch.as_tensor(normalizer.obs_std, dtype=..., device=...)`.

**`configs/student.yaml`**:
- Added `loss.input_noise_sigma: 0.03`
- All other hyperparams unchanged from Exp1+2.

### Key design decisions
- Noise on warmup (ground-truth-fed steps) but NOT on rollout (model-fed
  steps): warmup noise trains the model to handle slight mis-calibration at
  the start of open-loop; rollout predictions are what we want to be robust,
  not an additional perturbation source.
- `σ=0.03` is conservative. If VPT still collapses before step 100, try 0.05.
  If one-step RMSE degrades noticeably, try 0.01.

### Results

**Pending Colab run** (stacked with Exp1+2).

| Split | VPT80@0.25 | VPT50@0.25 | nMSE@10 | nMSE@100 | nMSE@1000 | nMSE_AUC |
|---|---|---|---|---|---|---|
| test | — | — | — | — | — | — |
| ood | — | — | — | — | — | — |

Decision: keep / revert / iterate

---

## Experiment 4: Acceleration prediction + Semi-implicit Euler (2026-05-20)

### Diagnosis / Motivation

Exp1–3 all plateau at VPT80@0.25 ≈ 23–24. All three experiments targeted
*how the model is trained* (longer horizon, clamped output, noise injection).
The remaining bottleneck is *what the model predicts*: a raw 4D state-delta.
This forces the model to learn the trivial kinematic coupling
`delta_x ≈ x_dot * dt` and `delta_theta ≈ omega * dt` purely from data,
introducing unnecessary approximation error that compounds over rollout.

### Hypothesis

Replace the 4D delta head with a 2D acceleration head `[a_cart, a_pole]` and
derive the full 4D delta analytically via **semi-implicit Euler integration**:

```
x_dot_new = x_dot + a_cart * dt      # velocities updated first
omega_new = omega + a_pole * dt
x_new     = x    + x_dot_new * dt    # positions use NEW velocity (semi-implicit)
theta_new = theta + omega_new * dt
```

The model only learns the physically non-trivial part (how forces translate to
accelerations). Position deltas are computed exactly. Semi-implicit Euler is
more numerically stable than explicit Euler for oscillatory systems because
updated velocities feed back into the position update immediately.

`dt = 0.04 s` (InvertedPendulum-v5: `frame_skip=2`, MuJoCo timestep `0.02 s`).
Confirmed from Gymnasium source; `env.dt` returns 0.04.

Expected: VPT80@0.25 ≥ 80 on test; nMSE@100 < 0.05 (vs ~0.45 in Exp1–3).

### Implementation

**`student/model.py`** — redesigned `forward`, same encoder/GRU:

- `self.head = nn.Linear(hidden_dim, 2)` — 2D acceleration output
- Five new registered buffers (`obs_std`, `obs_mean`, `delta_std`, `delta_mean`,
  `accel_scale`) initialised to identity; filled lazily by `losses.compute_loss`
  so they follow `model.to(device)` automatically
- `forward`:
  1. Encoder + GRU unchanged (sin/cos features retained from Exp1)
  2. Head → `raw [B,2]` → `a = tanh(raw) * accel_scale` in real acceleration units
  3. Denormalise obs: `obs_real = obs_norm * obs_std + obs_mean` (+ not −)
  4. Semi-implicit Euler → `delta_real [B,4]` in real units
  5. Renormalise: `delta_norm = (delta_real − delta_mean) / delta_std`
  6. Soft limiter: `delta_limit * tanh(delta_norm / delta_limit)` (Exp2)
  7. Return `(delta_norm, hidden)` — interface unchanged for `predict_next`
- `delta_limit=5.5` and `dt=0.04` are hardcoded defaults (not in YAML, since
  `build_model` in locked `checkpoint.py` only passes `hidden_dim/num_layers/use_gru`)

**`student/losses.py`** — lazy buffer init added to `compute_loss`:

- Runs once (guarded by `model._buffers_initialized`) before first `model.forward`
- Copies `normalizer.{obs,delta}_{mean,std}` into the corresponding model buffers
- Sets `accel_scale = [5σ_accel_cart, 5σ_accel_pole]` where
  `σ_accel = delta_std[2 or 3] / dt` (since `delta_ẋ = a_cart * dt`)
- Prints scale values at init for observability in Colab logs

**`configs/student.yaml`** — unchanged (all Exp1–3 settings retained:
`hidden_dim=192, use_gru=true, updates=12000, horizon=150, noise=0.03`).

### Key correctness checks

| Check | Result |
|---|---|
| Denorm formula | `obs = obs_norm * obs_std + obs_mean` (+, not −) ✓ |
| Semi-implicit order | velocities first, then positions with new velocities ✓ |
| `predict_next` compat | model returns `delta_norm` s.t. `denorm(delta_norm) = delta_real` ✓ |
| Buffers in checkpoint | `register_buffer` → saved in state_dict, restored on load ✓ |
| Eval safety | lazy init only triggers if `not _buffers_initialized`; buffers are correct at eval after checkpoint load ✓ |
| Backward compat | `hasattr(model, "accel_scale")` guard — old models without this buffer skip the init block ✓ |

### Risk

Risk: accel_scale = 5σ of training accel might allow large initial outputs
during training. Watch the first few hundred update steps for loss explosions.

### Results

**ABANDONED** — VPT80@0.25 = 19/20 (test/ood), worse than Exp3 baseline (24/24).
Physics-constrained 2D output made the training objective harder to optimise,
and the model couldn't match even the unconstrained 4D-delta head.
Model.py reverted to Exp3 state for Exp5.

| Split | VPT80@0.25 | VPT50@0.25 | nMSE@10 | nMSE@100 | nMSE@1000 | nMSE_AUC |
|---|---|---|---|---|---|---|
| test | 19 | — | — | — | — | — |
| ood | 20 | — | — | — | — | — |

Decision: revert model.py; stack Exp5 on Exp3 base

---

## Experiment 5: Truncated BPTT + train horizon 300 (2026-05-20)

### Context / Motivation

Exp1–3 plateau at VPT80@0.25 ≈ 23–24. The error curve shows nMSE growing
continuously after step 50 — the model is optimised for 150-step rollouts but
evaluated at 1000. Exp4 (physics model) hurt performance, so we revert to the
Exp3 base (sin/cos+GRU, 4D delta head).

Strategy: train with horizon=300 to expose the model to longer drift, and use
**Truncated BPTT** (chunk_size=100) to make the longer rollout computationally
feasible. Truncated BPTT detaches the GRU hidden state every 100 steps,
capping the recurrent gradient path length and preventing vanishing gradients
over 300-step rollouts. The state prediction (`cur`) is NOT detached, so the
loss signal still propagates through the full output trajectory.

### Hypothesis

A 300-step training horizon exposes the model to the regime where error
actually accumulates (step 50–300). Truncated BPTT makes this feasible by
limiting recurrent gradient depth to 100 steps. Expected: VPT80@0.25 ≥ 50,
nMSE@100 < 0.20.

### Changed Files

**`student/rollout.py`** — added `truncated_bptt_rollout`:
- Self-contained standalone function with same no-leak contract as
  `open_loop_rollout`; detaches `hidden` at every `chunk_size` boundary
  (h > 0, h % chunk_size == 0); `cur` is never detached

**`student/losses.py`**:
- Added `chunk_size: int = 0` keyword arg to `rollout_loss`
- BPTT detach inline in the closed-loop rollout loop (before each
  `predict_next` call, when `chunk_size > 0 and h > 0 and h % chunk_size == 0`)
- `chunk_size=0` → no detach → backward-compatible with all previous configs
- `compute_loss` reads `bptt_chunk_size` from config and passes to `rollout_loss`
- Warmup noise (Exp3) is preserved in both BPTT and non-BPTT paths

**`configs/student.yaml`**:

| Param | Before | After |
|---|---|---|
| `loss.rollout_train_horizon` | 150 | 300 |
| `loss.bptt_chunk_size` | — | 100 |
| `training.train_sequence_length` | 256 | 384 (≥ warmup+horizon+1 = 311) |
| `training.updates` | 12000 | 15000 |

**`student/model.py`** — reverted from Exp4 back to Exp3 state:
- `git checkout d02ffa9 -- student/model.py`
- sin/cos features, GRU, 4D delta head, delta_limit=5.5, hard clamp ±6.0

### Gradient flow analysis

With chunk_size=100 and horizon=300:
- GRU hidden detaches at h=100 and h=200
- Each chunk spans 100 steps of GRU gradient
- `cur` (state prediction) gradients still flow back to h=0 through the
  MLP output path (PyTorch autograd follows the `cur` → `predict_next` →
  `cur_new` chain even after `hidden.detach()`)
- This may trigger OOM if PyTorch retains the full 300-step activation graph
  for `cur`. If that happens, detach `cur` too (but at cost of weaker signal).

### Results

**Pending Colab run**.

| Split | VPT80@0.25 | VPT50@0.25 | nMSE@10 | nMSE@100 | nMSE@1000 | nMSE_AUC |
|---|---|---|---|---|---|---|
| test | — | — | — | — | — | — |
| ood | — | — | — | — | — | — |

Decision: keep / revert / iterate

---

## Experiment 6: Physics-based pole_angle correction

**Date**: 2026-05-20
**Branch/commit**: Exp6 (on top of Exp5 / 49302ea)
**Hypothesis**: Diagnostic 5 revealed that the model learns ang_vel (dim 3) well (prediction ratio 0.88) but pole_angle (dim 1) poorly (ratio 0.45). The signal for Δθ in normalized space is tiny and gets dominated by the other loss terms. Instead of making the model learn the relationship Δθ ≈ ω_new·dt, we can enforce it analytically and let the network focus on the harder dynamics.

### Diagnostic 5 findings (motivation)

Per-dimension prediction ratios (σ_pred / σ_target in normalized space):

| Dimension | Ratio |
|---|---|
| cart_pos (dim 0) | 0.80 |
| pole_angle (dim 1) | **0.45** |
| cart_vel (dim 2) | 0.95 |
| pole_ang_vel (dim 3) | 0.88 |

pole_angle is the worst-predicted dimension despite being the most safety-critical for rollout stability. It has a kinematic constraint linking it directly to ang_vel via semi-implicit Euler:

```
ω_new  = ω + Δω                    (Δω from the model — ratio 0.88, reliable)
Δθ     = ω_new * dt                (semi-implicit: position uses updated velocity)
dt     = 0.04 s  (InvertedPendulum-v5: frame_skip=2 × timestep=0.02)
```

### Changes

**`student/model.py`**:

- Added `dt=0.04` constructor parameter; `self.dt = float(dt)`
- Added 4 `register_buffer` calls: `obs_mean`, `obs_std`, `delta_mean`, `delta_std` (all zeros/ones at init; filled lazily by `losses.compute_loss`)
- Added physics correction block AFTER tanh + hard clamp so the physics value is not distorted by the soft limiter:

```python
delta_omega_real = delta[:, 3] * self.delta_std[3] + self.delta_mean[3]  # real Δω
omega_real = obs_norm[:, 3] * self.obs_std[3] + self.obs_mean[3]         # real ω
omega_new = omega_real + delta_omega_real                                 # updated ω
delta_theta_real = omega_new * self.dt                                    # Δθ = ω_new*dt
delta_theta_norm = (delta_theta_real - self.delta_mean[1]) / self.delta_std[1]
delta = torch.cat([delta[:, 0:1],                 # cart_pos   — model's
                   delta_theta_norm.unsqueeze(-1), # pole_angle — PHYSICS
                   delta[:, 2:3],                  # cart_vel   — model's
                   delta[:, 3:4]], dim=-1)          # ang_vel    — model's
```

**`student/losses.py`**:

- Lazy-init block fills 4 physics buffers (`obs_mean`, `obs_std`, `delta_mean`, `delta_std`) from normalizer on first `compute_loss` call, guarded by `_buffers_initialized`
- Guard condition: `hasattr(model, "obs_std") and not getattr(model, "_buffers_initialized", False)`
- Print diagnostic: `[lazy-init] dt={model.dt}, obs_std={normalizer.obs_std}`

**`configs/student.yaml`** (reverted from Exp5 to Exp3 baselines; BPTT not needed for Exp6 scope):

| Param | Before (Exp5) | After |
|---|---|---|
| `training.updates` | 15000 | 12000 |
| `training.train_sequence_length` | 384 | 256 |
| `loss.rollout_train_horizon` | 300 | 150 |
| `loss.bptt_chunk_size` | 100 | 0 |

All other params (hidden_dim=192, num_layers=2, use_gru=true, rollout_weight=2.0, input_noise_sigma=0.03) unchanged from Exp3.

### Key design decisions

1. **Apply physics AFTER tanh+clamp** so the physics value isn't squeezed by the limiter. The learned dims (cart_pos, cart_vel, ang_vel) still pass through tanh+clamp.
2. **Gradient signal preserved**: the physics override uses `delta[:, 3]` (ang_vel) which remains in the autograd graph. Gradients flow through the physics correction back to the ang_vel head — the model is implicitly taught to predict ang_vel well (since that directly determines Δθ).
3. **Lazy buffer init**: buffers start as zeros/ones and are filled from the real normalizer stats on the first training step. `register_buffer` ensures they follow `model.to(device)` and are saved in checkpoints.
4. **dt=0.04 is exact**: InvertedPendulum-v5 uses `frame_skip=2` and `timestep=0.02`, so each transition spans 0.04 s.

### Expected improvement

- pole_angle prediction ratio should jump from 0.45 toward ~1.0 (enforced, not learned)
- VPT80@0.25 target: ≥30 (vs Exp3 baseline of ~23–24)
- The physics override removes the hardest dimension from the learning problem

### Results

**Pending Colab run**.

| Split | VPT80@0.25 | VPT50@0.25 | nMSE@10 | nMSE@100 | nMSE@1000 | nMSE_AUC |
|---|---|---|---|---|---|---|
| test | — | — | — | — | — | — |
| ood | — | — | — | — | — | — |

Decision: keep / revert / iterate

---
