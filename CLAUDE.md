# EEC289A Assignment 2: Inverted Pendulum World Model

## Task
Train a dynamics world model for MuJoCo InvertedPendulum-v5.
- State: 4D (cart pos, pole angle, cart vel, pole angular vel)
- Action: 1D (cart force)
- Primary metric: VPT80@0.25 (higher = better)
- Secondary metric: nMSE@1000, tie-break: nMSE_AUC

The model predicts the next state given (state, action). Eval rolls out
open-loop for up to 1000 steps after a 10-step warmup with ground-truth states.

## Hard constraints (NEVER violate)

### Files you MAY modify
- student/model.py
- student/rollout.py
- student/losses.py
- student/metrics.py
- configs/student.yaml

### Files you MUST NOT modify
- Anything under wm_hw/
- configs/dev.yaml, configs/official_eval.yaml, configs/public_scoreboard.yaml
- Anything under tests/
- Anything under notebooks/

### Other constraints
- `student/rollout.py` must NOT read ground-truth states after the warmup
  window. The test `tests/test_rollout_no_leak.py` enforces this.
- After ANY change to the five modifiable files, run `pytest -q -m "not slow"`
  and confirm all 13 tests pass before proceeding.

## Environment
- Platform: Windows + Git Bash + conda (base)
- I will run this on Colab with GPU for full training. Locally we only do
  smoke tests to confirm pipeline correctness.
- Use `--smoke` flag for local validation, never expect full training to
  finish on this machine.

## Standard workflow
1. Make a change to one of the five allowed files.
2. Run `pytest -q -m "not slow"`. If it fails, fix or revert.
3. Smoke train to confirm the pipeline still works:

4. Log each experiment in `experiments.md` at the repo root with:
   - Date, change summary, hypothesis
   - Resulting smoke metrics (VPT80@0.25, nMSE@10, one_step_rmse)
   - Decision: keep / revert / iterate

## Key technical context
- The system is approximately Markov (4D state captures it), but a GRU can
  still help absorb rollout drift.
- Dominant nonlinearity is gravity: torque on the pole ∝ sin(angle). Feeding
  sin/cos of the angle as auxiliary features is a strong inductive bias.
- The starter trains rollout loss at only 15 steps, but eval rolls out 1000.
  This horizon mismatch is the single largest source of poor VPT scores.
- nMSE is computed in normalized space (divided by train obs_std), so
  meaningful per-step accuracy is in normalized units.

## Previous attempt (failed)
A previous attempt with hidden_dim=256, GRU, 3 ResidualBlocks, and rollout
horizon curriculum 10→50 produced test VPT80@0.25=18, ood VPT80@0.25=4.
This means severe overfitting to the training distribution and rollout
drift kicks in around step ~20. Avoid simply repeating that configuration.
Promising unexplored directions:
- Adding sin/cos of angle as input features
- Training-time rollout horizon ≥ 100 (requires longer sequences from
  configs/public_scoreboard.yaml dataset)
- Predicting acceleration + semi-implicit Euler integration
- Stronger regularization (dropout, weight decay) against overfitting