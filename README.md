# Dreamer World Model Colab Homework

This repository is a teaching-first homework package for learning a small
Dreamer-style world model agent in Google Colab.

The goal is not to reproduce the full official DreamerV3 codebase. Instead,
students run and modify a compact PyTorch `MiniDreamer` baseline that exposes
the complete training loop:

1. collect environment experience
2. train a recurrent state-space world model
3. imagine latent rollouts
4. train actor and critic from imagined trajectories
5. restore checkpoints
6. evaluate policy behavior and model prediction quality
7. submit a standardized public benchmark bundle

## Assignment Scope

- Environment: `Pendulum-v1`
- Observation: state vector from Gymnasium
- Action: continuous action normalized to `[-1, 1]` inside the agent
- Algorithm: MiniDreamer with Gaussian RSSM
- Framework: PyTorch + Gymnasium
- Default baseline budget: `100k` environment steps
- Leaderboard budget limit: `250k` environment steps

## Why This Version Exists

Official DreamerV3 is powerful and should be studied conceptually, but its JAX
and Embodied stack is too large for a first Colab homework. This course version
keeps the important ideas visible as ordinary Python files that students can
read, run, and modify.

Reference materials:

- DreamerV3 code: https://github.com/danijar/dreamerv3
- DreamerV3 paper: https://arxiv.org/abs/2301.04104
- DreamerV3 project page: https://danijar.com/project/dreamerv3/

## File Map

```text
configs/course_config.json          Course knobs and public benchmark settings
configs/colab_requirements.txt      Colab dependencies

world_model_hw/
  agent.py                          MiniDreamer update logic
  checkpointing.py                  Save/load helpers
  config.py                         JSON config and reproducibility helpers
  envs.py                           Gymnasium environment/action helpers
  models.py                         RSSM, world model, actor, critic
  replay.py                         Episode replay buffer
  visualization.py                  Plot/video helpers

train.py                            Collect data and train MiniDreamer
inspect_env.py                      Print environment and config summary
evaluate_policy.py                  Restore checkpoint and evaluate policy
generate_public_rollout.py          Generate standardized benchmark rollout
public_eval.py                      Score benchmark rollout
quick_world_model_check.py          Make a prediction sanity plot
benchmark_specs.py                  Deterministic seeds and benchmark constants
notebooks/dreamer_public_colab_template.ipynb
tests/                              Small unit tests
```

## Recommended Colab Workflow

Use the notebook at `notebooks/dreamer_public_colab_template.ipynb`.

Important: Colab storage under `/content` is temporary. Students should fork or
copy this repository into their own GitHub repository, change `COURSE_REPO_URL`
in the notebook, and push their changes regularly.

## Local Quick Start

Install dependencies:

```bash
python -m pip install -r configs/colab_requirements.txt
```

Inspect the environment:

```bash
python inspect_env.py
```

Dry-run config and model construction:

```bash
python train.py --dry-run
```

Run a tiny smoke test:

```bash
python train.py --local-smoke --output-dir artifacts/smoke
python quick_world_model_check.py --checkpoint-dir artifacts/smoke/best_checkpoint
```

Train the baseline:

```bash
python train.py \
  --config configs/course_config.json \
  --stage baseline \
  --output-dir artifacts/run_baseline
```

Evaluate and render:

```bash
python evaluate_policy.py \
  --checkpoint-dir artifacts/run_baseline/best_checkpoint \
  --render \
  --output-dir artifacts/demo_bundle
```

Generate and score the public rollout:

```bash
python generate_public_rollout.py \
  --checkpoint-dir artifacts/run_baseline/best_checkpoint \
  --output-dir artifacts/public_eval_bundle \
  --num-episodes 8

python public_eval.py \
  --rollout-npz artifacts/public_eval_bundle/rollout_public_eval.npz \
  --output-json artifacts/public_eval_bundle/public_eval.json
```

## Student Modification Boundary

Students should mostly modify:

- `world_model_hw/models.py`
- `world_model_hw/agent.py`
- `configs/course_config.json`

Students should usually not modify:

- `public_eval.py`
- rollout `.npz` field names
- checkpoint loading and saving logic

This keeps submissions comparable across the public and hidden evaluations.

## Required Submission

- `best_checkpoint/`
- `configs/colab_runtime_config.json`
- `public_eval.json`
- `demo_policy.mp4` or `demo_policy.gif`
- `world_model_rollout.png`
- `short_report.pdf`

The short report should explain the RSSM/world-model structure, baseline
curves, at least one attempted improvement, public metric changes, and one
failed idea.
