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
