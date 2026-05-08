# 作业：From Pendulum Dynamics to SmallWorld-MJ World Models

本作业完全聚焦 **world model dynamics learning**。你不会训练 policy，不会使用 reward，也不会实现 actor/critic。教师已经提供 frozen MuJoCo XML 任务和数据生成器；你的任务是让 world model 从 one-step prediction 扩展到多任务、参数条件化、长时域 open-loop prediction。

## 1. 作业目标

你将从一个只会在 `simple_pendulum` 上做 one-step prediction 的 baseline 开始：

```text
s_hat[t+1] = s[t] + MLP(s[t], a[t])
```

然后把它扩展成 SmallWorld-MJ world model：

```text
f_theta(s_t, a_t, task_params, task_id) -> s_hat[t+1]
```

官方评测使用：

```text
前 10 步：ground-truth conditioning
后 90 步：model 自己 autoregressive open-loop rollout
```

这个任务对应论文 *SmallWorlds: Assessing Dynamics Understanding of World Models in Isolated Environments* 的核心思想：把 world model 从 RL pipeline 中拆出来，在 reward-free、fully observable、isolated physical worlds 中单独评测动力学理解。

## 2. 论文思想如何进入作业

| 论文 | 作业 |
| --- | --- |
| MuJoCo 3D physical worlds | MuJoCo XML task suite |
| No artificial reward | 不使用 reward，不训练 policy |
| Fully observable state-space | position/orientation/velocity/task params |
| Randomly collected episodes | smooth random forces/torques |
| 10 conditioning + 90 imagined steps | official eval warmup=10, horizon=90 |
| Diverse physical categories | gravity, circle, contact, rolling, spin |
| Long-horizon error growth | horizon_error.png + AUC |
| Architecture comparison | baseline MLP vs student GRU/Transformer/RSSM optional |
| Physical failure diagnosis | energy/radius/no-slip/contact/phase metrics |

注意：本作业不是完整复现论文中的 Diffusion Forcing、MoSim、Atari、Go、Memory Maze 或 geometry keypoint task。它是 course-scale SmallWorld-MJ implementation。

## 3. 十个必做任务

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

这些任务覆盖 gravity、周期运动、几何约束、碰撞、旋转和平移耦合、刚体自旋等不同动力学类别。

## 4. 学生需要实现什么

你主要修改：

```text
smallworld_mj/student/models/student_model.py
smallworld_mj/student/rollout.py
smallworld_mj/student/losses.py
smallworld_mj/student/physics_metrics.py
configs/student.yaml
```

你不需要修改：

```text
smallworld_mj/assets/xml/
smallworld_mj/envs/
smallworld_mj/data/
official eval logic
MuJoCo contact/rendering setup
```

必做内容：

1. 实现 open-loop rollout：warmup 后不能偷看 future ground-truth states。
2. 实现 parameter-conditioned residual dynamics model。
3. 实现 one-step + rollout mixed loss。
4. 实现并报告 physical metrics。
5. 在 10 个 SmallWorld-MJ tasks 上报告 test 和 OOD 指标。

## 5. 运行流程

安装依赖：

```bash
python -m pip install -r configs/colab_requirements.txt
```

检查 MuJoCo 任务：

```bash
python inspect_mj_tasks.py --config configs/smallworld_mj.yaml
```

生成 smoke 数据：

```bash
python generate_mj_dataset.py \
  --config configs/smallworld_mj.yaml \
  --taskpack smallworld_all \
  --profile smoke \
  --output-dir artifacts/smoke/mj_data
```

运行 Pendulum one-step baseline：

```bash
python run_experiment.py \
  --config configs/pendulum_baseline.yaml \
  --task simple_pendulum \
  --model baseline_mlp \
  --dataset-dir artifacts/smoke/mj_data \
  --local-smoke \
  --output-dir artifacts/smoke/pendulum_baseline
```

运行 solution smoke：

```bash
python run_experiment.py \
  --config configs/solution.yaml \
  --taskpack smallworld_all \
  --model solution \
  --dataset-dir artifacts/smoke/mj_data \
  --local-smoke \
  --output-dir artifacts/smoke/solution
```

评测：

```bash
python eval.py \
  --checkpoint-dir artifacts/smoke/solution/best_checkpoint \
  --dataset-dir artifacts/smoke/mj_data \
  --split test \
  --warmup 10 \
  --horizon 90 \
  --output-dir artifacts/smoke/eval_test
```

可视化：

```bash
python visualize_rollout.py \
  --checkpoint-dir artifacts/smoke/solution/best_checkpoint \
  --dataset-dir artifacts/smoke/mj_data \
  --task bouncing_ball \
  --split test \
  --output-dir artifacts/smoke/viz
```

## 6. 指标

主要指标：

- `one_step_nrmse`：teacher-forced one-step normalized RMSE。
- `open_loop_15_nrmse`：15-step open-loop normalized RMSE。
- `open_loop_90_nrmse`：90-step open-loop normalized RMSE。
- `horizon_error_auc`：horizon error curve 平均面积。
- OOD split 上的同类指标。

物理诊断：

- `energy_drift`
- `radius_violation`
- `box_violation`
- `no_slip_violation`
- `phase_drift`

这些指标回答的问题不是“agent 拿了多少 reward”，而是“world model 的想象未来是否稳定、是否泛化、是否违反物理规律”。

## 7. 提交要求

代码：

```text
smallworld_mj/student/models/student_model.py
smallworld_mj/student/rollout.py
smallworld_mj/student/losses.py
smallworld_mj/student/physics_metrics.py
configs/student.yaml
```

Artifacts：

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

报告最多 4 页，回答：

1. 为什么 one-step Pendulum baseline 会在 long-horizon rollout 漂移？
2. 你的 model/loss 改了什么？
3. multi-step rollout loss 对 90-step error 有什么影响？
4. task parameter conditioning 对 OOD 有什么影响？
5. 哪个 SmallWorld-MJ task 最难？为什么？
6. 你的模型违反了什么物理约束？
7. 哪些指标提升，哪些指标退化？

## 8. 评分建议

总分 100：

- 15 分：open-loop rollout 正确实现。
- 20 分：parameter-conditioned residual/GRU world model。
- 20 分：one-step + rollout mixed loss。
- 20 分：10 个 SmallWorld-MJ tasks 的 test/OOD 实验。
- 10 分：physical metrics / penalty。
- 10 分：horizon curve、visualization、failure diagnosis。
- 5 分：reproducibility，包括 config、seed、checkpoint、JSON、plots。

加分最多 10：

- Transformer/RSSM/ODE-style model。
- single multi-task model 与 per-task model 对比。
- 更深入的物理 penalty 或 geometry keypoint extension。
