# Homework: From Pendulum Dynamics to SmallWorld-MJ World Models

This repository is a course-scale implementation of the SmallWorlds benchmark
idea using frozen MuJoCo XML tasks. Students do **not** train a policy, do
**not** use reward, and do **not** write MuJoCo environments. The assignment is
about world models only: learning dynamics from state/action trajectories and
diagnosing long-horizon prediction failure.

## Core Idea

Students start with a one-step `simple_pendulum` dynamics predictor:

```text
s_hat[t+1] = s[t] + MLP(s[t], a[t])
```

That baseline can look reasonable under teacher forcing, but it drifts badly
when rolled out open-loop. Students then extend it into a parameter-conditioned
world model for 10 isolated MuJoCo physical worlds:

```text
f_theta(s_t, a_t, task_params, task_id) -> s_hat[t+1]
```

Official evaluation uses 10 ground-truth conditioning steps followed by 90
autoregressive imagined prediction steps.

## Paper Alignment

| SmallWorlds paper idea | Course implementation |
| --- | --- |
| MuJoCo 3D physical worlds | Frozen MuJoCo XML task suite |
| No artificial reward | No reward, no policy, no actor/critic |
| Fully observable state-space | Position/orientation/velocity/task params |
| Randomly collected episodes | Smooth random forces/torques |
| 10 conditioning + 90 imagined steps | Official eval warmup=10, horizon=90 |
| Diverse physical categories | Gravity, circular motion, contact, rolling, spin |
| Long-horizon error growth | `horizon_error.png` + AUC |
| Architecture comparison | MLP baseline vs GRU student, optional Transformer/RSSM |
| Physical failure diagnosis | Energy/radius/no-slip/contact/phase metrics |

This is not a full reproduction of Diffusion Forcing, MoSim, Atari, Go, Memory
Maze, or the geometry keypoint task. It is a faithful course adaptation of the
paper's isolated, reward-free world-model evaluation protocol.

## Required Tasks

The official SmallWorld-MJ taskpack contains 10 MuJoCo tasks:

```text
simple_pendulum
projectile
circular_motion
bouncing_ball
rolling
free_fall
inclined_plane
elastic_collision
rotation
spin
```

The MuJoCo XML files and dataset generator are frozen benchmark code. Student
work is limited to world-model code and diagnostics.

## Repository Layout

```text
smallworld_mj/
  assets/xml/              Frozen MuJoCo task suite
  envs/                    MuJoCo reset/step/state/render wrappers
  data/                    Dataset generation and normalization
  models/                  One-step MLP baseline
  student/                 Student-editable scaffold
  solution/                Staff reference implementation
  train.py                 Dynamics-only training loop
  eval.py                  Official evaluation helper
  metrics.py               NRMSE, horizon AUC, physical diagnostics
  visualization.py         Rollout plots and videos

configs/
  smallworld_mj.yaml       Benchmark/taskpack/dataset profiles
  pendulum_baseline.yaml   One-step baseline config
  student.yaml             Student model/loss/training config
  solution.yaml            Staff reference config

notebooks/
  smallworld_mj_colab_template.ipynb
```

Students should normally edit only:

```text
smallworld_mj/student/models/student_model.py
smallworld_mj/student/rollout.py
smallworld_mj/student/losses.py
smallworld_mj/student/physics_metrics.py
configs/student.yaml
```

## Quick Start

Install dependencies:

```bash
python -m pip install -r configs/colab_requirements.txt
```

Inspect MuJoCo tasks:

```bash
python inspect_mj_tasks.py --config configs/smallworld_mj.yaml
```

Generate a smoke dataset:

```bash
python generate_mj_dataset.py \
  --config configs/smallworld_mj.yaml \
  --taskpack smallworld_all \
  --profile smoke \
  --output-dir artifacts/smoke/mj_data
```

Run the one-step Pendulum baseline:

```bash
python run_experiment.py \
  --config configs/pendulum_baseline.yaml \
  --task simple_pendulum \
  --model baseline_mlp \
  --dataset-dir artifacts/smoke/mj_data \
  --local-smoke \
  --output-dir artifacts/smoke/pendulum_baseline
```

Run the staff solution smoke:

```bash
python run_experiment.py \
  --config configs/solution.yaml \
  --taskpack smallworld_all \
  --model solution \
  --dataset-dir artifacts/smoke/mj_data \
  --local-smoke \
  --output-dir artifacts/smoke/solution
```

Evaluate and visualize:

```bash
python eval.py \
  --checkpoint-dir artifacts/smoke/solution/best_checkpoint \
  --dataset-dir artifacts/smoke/mj_data \
  --split test \
  --warmup 10 \
  --horizon 90 \
  --output-dir artifacts/smoke/eval_test

python visualize_rollout.py \
  --checkpoint-dir artifacts/smoke/solution/best_checkpoint \
  --dataset-dir artifacts/smoke/mj_data \
  --task bouncing_ball \
  --split test \
  --output-dir artifacts/smoke/viz
```

## Metrics

Official metrics are lower-better:

- `one_step_nrmse`
- `open_loop_15_nrmse`
- `open_loop_90_nrmse`
- `horizon_error_auc`
- OOD versions of the same metrics
- physical diagnostics: `energy_drift`, `radius_violation`,
  `box_violation`, `no_slip_violation`, `phase_drift`

All RMSE values are normalized by train-split state statistics so tasks with
different state scales are comparable.

## Student Deliverables

Required code:

```text
smallworld_mj/student/models/student_model.py
smallworld_mj/student/rollout.py
smallworld_mj/student/losses.py
smallworld_mj/student/physics_metrics.py
configs/student.yaml
```

Required artifacts:

```text
artifacts/pendulum_baseline/metrics_test.json
artifacts/pendulum_baseline/rollout_plot.png
artifacts/pendulum_baseline/horizon_error.png
artifacts/student_all/best_checkpoint/
artifacts/student_all/eval_test/metrics_test.json
artifacts/student_all/eval_ood/metrics_ood.json
artifacts/student_all/viz/rollout_plot.png
artifacts/student_all/viz/horizon_error.png
artifacts/student_all/viz/smallworld_mj_rollout.mp4
short_report.pdf
```

The short report should focus on experimental diagnosis, not paper summary:
why one-step prediction drifts, how rollout loss changes 90-step error, whether
task-parameter conditioning helps OOD, which task is hardest, and which
physical constraints your model violates.

## Validation

```bash
pytest -q
python inspect_mj_tasks.py --config configs/smallworld_mj.yaml
python generate_mj_dataset.py --config configs/smallworld_mj.yaml --taskpack smallworld_all --profile smoke --output-dir artifacts/smoke/mj_data
python run_experiment.py --config configs/solution.yaml --taskpack smallworld_all --model solution --dataset-dir artifacts/smoke/mj_data --local-smoke --output-dir artifacts/smoke/solution
python eval.py --checkpoint-dir artifacts/smoke/solution/best_checkpoint --dataset-dir artifacts/smoke/mj_data --split test --warmup 10 --horizon 90 --output-dir artifacts/smoke/eval_test
python visualize_rollout.py --checkpoint-dir artifacts/smoke/solution/best_checkpoint --dataset-dir artifacts/smoke/mj_data --task bouncing_ball --split test --output-dir artifacts/smoke/viz
```
