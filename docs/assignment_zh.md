# 作业：Dreamer World Model 训练与评测

## 1. 作业背景

在传统 model-free reinforcement learning 中，策略通常直接通过大量真实环境交互来学习。Dreamer 代表的是另一条路线：先学习一个可以预测未来的 world model，再让策略在这个 learned latent world 里“想象”未来，从而训练 actor 和 critic。

这份作业的目标不是复现完整 DreamerV3，也不是刷榜到研究级 SOTA。目标是让你完整走通并理解一条 model-based RL 训练链路：

1. 从环境收集真实交互数据。
2. 用 replay buffer 训练 RSSM world model。
3. 让 world model 预测 observation、reward 和 continuation。
4. 在 latent state 中进行 imagination rollout。
5. 用 imagined rollout 训练 actor 和 critic。
6. 用统一 public evaluation 同时评估 policy 表现和 world model 预测质量。

你最终需要回答一个核心问题：

**你的 world model 学到了什么？它如何帮助 policy 学得更好？**

本作业现在有第二个同等重要的问题：

**如果没有 reward、没有 policy return，只给 state/action 轨迹，你的 world model 是否真的学会了动力学规律？**

这个问题来自论文 *SmallWorlds: Assessing Dynamics Understanding of World Models in Isolated Environments*。论文的核心不是“多了 10 个任务”这么简单，而是提出一种更干净的 world model 评测协议：把 world model 从 RL pipeline 里拿出来，在 isolated、controllable、fully observable、reward-free 的环境中考察 long-horizon prediction。

## 2. 作业环境与代码入口

本作业使用课程自带的 PyTorch `MiniDreamer`，运行环境是 Google Colab。

- 控制环境：`Pendulum-v1`
- SmallWorld-Lite benchmark：10 个轻量物理动力学任务
- Observation：Gymnasium state vector
- Action：连续动作，agent 内部归一化到 `[-1, 1]`
- 算法：RSSM world model + latent imagination actor-critic；reward-free RSSM dynamics model
- 框架：PyTorch + Gymnasium
- Notebook：`notebooks/dreamer_public_colab_template.ipynb`

你应该优先阅读这些文件：

- `world_model_hw/models.py`
- `world_model_hw/agent.py`
- `train.py`
- `public_eval.py`
- `configs/course_config.json`
- `smallworld_hw/tasks.py`
- `smallworld_hw/models.py`
- `smallworld_train.py`
- `smallworld_eval.py`
- `configs/smallworld_config.json`

## 3. Baseline 要求

`local_smoke` 不是正式 baseline。它只用来检查 Colab、依赖安装、训练循环、checkpoint、视频和 public eval 是否能跑通。

正式训练预算如下：

- Debug smoke：`2k env steps`
- Required baseline：`100k env steps`
- Final / leaderboard maximum：`250k env steps`
- SmallWorld smoke：每个 split 少量随机 trajectory，用于检查 benchmark 链路
- SmallWorld required task：至少完成 `simple_pendulum` 或一个自选任务的完整训练、评测和可视化
- SmallWorld optional full benchmark：生成并分析 10 个任务的数据与指标

你必须至少完成：

1. 跑通 `local_smoke`，确认工程链路正常。
2. 跑通 `baseline`，预算为 `100k env steps`。
3. 基于 baseline 做至少一个有理由的改动。
4. 在不超过 `250k env steps` 的总预算内提交最终模型。
5. 跑通 SmallWorld-Lite smoke，并至少报告一个 SmallWorld 任务的 `10-step warmup + 90-step open-loop prediction` 结果。

推荐命令：

```bash
python train.py --local-smoke --output-dir artifacts/smoke
python quick_world_model_check.py --checkpoint-dir artifacts/smoke/best_checkpoint
```

正式 baseline：

```bash
python train.py \
  --config configs/course_config.json \
  --stage baseline \
  --output-dir artifacts/run_baseline
```

生成 public evaluation：

```bash
python generate_public_rollout.py \
  --checkpoint-dir artifacts/run_baseline/best_checkpoint \
  --output-dir artifacts/public_eval_bundle \
  --num-episodes 8 \
  --render-first-episode

python public_eval.py \
  --rollout-npz artifacts/public_eval_bundle/rollout_public_eval.npz \
  --output-json artifacts/public_eval_bundle/public_eval.json
```

SmallWorld smoke：

```bash
python smallworld_generate_dataset.py \
  --local-smoke \
  --output-dir artifacts/smallworld_smoke/data

python smallworld_train.py \
  --local-smoke \
  --task simple_pendulum \
  --dataset-dir artifacts/smallworld_smoke/data \
  --output-dir artifacts/smallworld_smoke/run

python smallworld_eval.py \
  --checkpoint-dir artifacts/smallworld_smoke/run/best_checkpoint \
  --dataset-dir artifacts/smallworld_smoke/data \
  --output-dir artifacts/smallworld_smoke/eval
```

## 4. 任务 A-B：跑通环境与复现 baseline

### Task A：跑通环境和 smoke test

你需要证明你理解 Colab runtime 和代码入口。

必须完成：

- 成功运行 notebook setup cell。
- 成功运行 `inspect_env.py`。
- 成功运行 `local_smoke`。
- 生成 `best_checkpoint/`。
- 生成 `world_model_rollout.png`。
- 生成一次 `public_eval.json`。

这一步的目的不是训练好模型，而是确认整条管线没有断。

### Task B：复现 baseline

你需要使用默认配置跑 `100k env steps` baseline。

必须记录：

- 使用的 GPU 类型或 CPU runtime。
- wall-clock 训练时间。
- best checkpoint 是在哪个 step 保存的。
- `eval/mean_return` 曲线或 progress 记录。
- public evaluation 的五项指标。

baseline 的 public metrics 包括：

- `mean_return`：policy 在真实环境中的平均回报，越高越好。
- `one_step_obs_rmse`：world model 一步 observation 预测误差，越低越好。
- `open_loop_obs_rmse`：open-loop 多步预测误差，越低越好。
- `reward_mae`：reward 预测误差，越低越好。
- `action_delta`：动作变化幅度，越低通常越平滑。

## 5. 可量化指标和视频指标

本作业不是只看一个 reward。你需要同时报告 policy、world model、动作质量和可视化结果。

### 5.1 Public evaluation 量化指标

`public_eval.py` 会输出下面五个主指标：

```text
mean_return           越高越好，权重 45%
one_step_obs_rmse     越低越好，权重 20%
open_loop_obs_rmse    越低越好，权重 20%
reward_mae            越低越好，权重 10%
action_delta          越低越好，权重 5%
```

含义如下：

- `mean_return` 衡量最终 policy 是否真的能控制 Pendulum。
- `one_step_obs_rmse` 衡量 world model 是否能预测下一步状态。
- `open_loop_obs_rmse` 衡量 world model 在不继续看真实 observation 的情况下，连续想象多步是否还能跟真实轨迹接近。
- `reward_mae` 衡量 reward head 是否学到了任务目标。
- `action_delta` 衡量动作是否平滑，避免策略只靠高频抖动拿分。

最终综合分 `course_composite_score` 由这五项归一化后加权得到。注意：这不是训练 reward，而是课程 public benchmark。

### 5.2 训练过程指标

训练日志中你还应该关注：

- `wm/loss`：world model 总损失。
- `wm/obs_loss`：observation reconstruction / prediction 误差。
- `wm/reward_loss`：reward prediction 误差。
- `wm/dyn_kl` 和 `wm/rep_kl`：RSSM prior/posterior 的 KL 项。
- `ac/actor_loss`：actor 在 imagined rollout 上的优化目标。
- `ac/critic_loss`：critic 对 lambda-return 的拟合误差。
- `ac/imagine_reward`：world model 想象轨迹中的平均 reward。
- `eval/mean_return`：真实环境 rollout 的平均回报。

报告中不需要逐行解释所有 log，但至少要展示 baseline 和改进方法的 `eval/mean_return`、public metrics 和一个 world-model prediction 图。

### 5.3 可以录制和观察的视频

本作业有两类可视化：

1. **Policy behavior video**

   由 `evaluate_policy.py --render` 生成：

   ```bash
   python evaluate_policy.py \
     --checkpoint-dir artifacts/run_baseline/best_checkpoint \
     --render \
     --output-dir artifacts/demo_bundle
   ```

   输出通常是：

   ```text
   artifacts/demo_bundle/demo_policy.mp4
   ```

   这个视频录制的是：**训练好的 actor policy 在真实 Gymnasium `Pendulum-v1` 环境中逐步执行动作时，环境 renderer 画出来的 pendulum 运动画面**。

   更具体地说，每一帧都是一次真实 environment step 后的可视化：

   ```text
   当前 observation -> actor 选择 action/torque -> 环境物理动力学 step -> render 当前 pendulum 角度和速度
   ```

   它不是 world model “脑内想象”的视频，也不是训练过程录像。它是 checkpoint 恢复后，policy 在真实环境里跑一个 episode 的行为录像。

   这个视频回答的问题是：policy 在真实 Gymnasium 物理动力学环境里到底怎么动？Pendulum 是否能被稳定摆起或控制？动作是否抖动？是否只是卡在某个角度不动？

2. **World model rollout plot**

   由 `quick_world_model_check.py` 生成：

   ```bash
   python quick_world_model_check.py \
     --checkpoint-dir artifacts/run_baseline/best_checkpoint \
     --output-dir artifacts/demo_bundle
   ```

   输出通常是：

   ```text
   artifacts/demo_bundle/world_model_rollout.png
   ```

   这个图不是视频，而是 world model 的“想象轨迹”对比图。它固定一段真实 action sequence，然后让 world model 在不继续读取真实 observation 的情况下 open-loop 往前预测，最后把预测 observation 和真实 observation 画在一起。

   这个图回答的问题是：world model open-loop 预测出来的 observation trajectory 是否跟真实 trajectory 有相似趋势？预测误差是不是随着 horizon 变长快速发散？

你的视频和图应该放进报告中。视频本身不是单独打分项，但它能帮助解释为什么某些量化指标好或坏。

### 5.4 可选视频分析指标

如果你想做更深入分析，可以从 rollout 或视频中额外报告：

- episode return 随时间变化。
- pendulum angle 是否接近 upright。
- angular velocity 是否过大。
- action 是否接近饱和。
- action 是否高频抖动。
- world model 预测误差是否随着 horizon 增长快速发散。

这些不是最低提交要求，但适合作为加分分析。

## 6. MiniDreamer 和 Dreamer / DreamerV3 的区别

本作业实现的是课程版 `MiniDreamer`，它借鉴 Dreamer 的核心思想，但刻意做了大量简化，方便你在 Colab 中读懂和修改。

共同点：

- 都学习 latent world model。
- 都使用 recurrent state-space model 思想。
- 都训练 prior / posterior latent dynamics。
- 都用 reward prediction 支持 policy learning。
- 都在 latent imagination rollout 中训练 actor 和 critic。
- 都使用 lambda-return 类型的 imagined return。

主要区别：

```text
Full Dreamer / DreamerV3              本作业 MiniDreamer
---------------------------------------------------------------
大规模通用 agent                       教学用小型 agent
JAX/Embodied 复杂工程栈                 PyTorch 单仓库脚本
图像、Atari、DMC、多任务等              Pendulum-v1 state observation
离散 categorical latent 常见            Gaussian stochastic latent
复杂归一化、symlog、two-hot value       简化 MSE reward/value 训练
更完整的 replay / logging / scaling     最小可读 replay 和训练 loop
研究级性能目标                          课程理解和可修改性优先
```

所以，本作业不是 DreamerV3 的完整复现。它是一个教学切片：保留“learn world model -> imagine -> train actor critic”的主干，把工程复杂度降到学生能完整读完和改动的程度。

你在报告中应该明确说明：你的实验结论只针对这个 MiniDreamer homework baseline，不要声称复现了完整 DreamerV3。

## 7. 是否使用物理引擎？MuJoCo 在哪里？

本作业使用了轻量物理动力学环境，但当前第一版没有引入 MuJoCo locomotion 级别的复杂物理引擎。

`Pendulum-v1` 是 Gymnasium classic-control 环境。Pendulum 的动力学由 Gymnasium 环境代码提供，不需要单独安装完整 MuJoCo、MuJoCo license 或 Isaac Sim。它是一个轻量物理控制环境：状态包含角度的 `cos(theta)`、`sin(theta)` 和角速度，动作是施加到 pendulum 上的 torque。

这意味着：

- 环境是真实 step-by-step dynamics，不是静态 supervised dataset。
- agent 需要通过动作影响未来状态。
- world model 学的是这个动力系统的 latent dynamics。
- policy video 展示的是物理系统中的实际控制行为。

但它和 MuJoCo locomotion/robotics 环境也有区别：

- 没有复杂接触动力学。
- 没有多刚体机器人模型。
- 没有高维图像 observation。
- 训练速度更快，适合教学和 Colab。

如果课程之后要升级，可以把同一个 MiniDreamer 框架迁移到 MuJoCo/Gymnasium 的连续控制任务，例如 `HalfCheetah-v4`、`Walker2d-v4` 或 DeepMind Control Suite。但第一版作业选择 `Pendulum-v1`，是为了保证所有学生能先完整理解 world model 训练流程，而不是被 MuJoCo 安装、机器人接触动力学和大规模训练成本卡住。

## 8. SmallWorld-Lite Benchmark：论文思想融入

论文提出的 SmallWorld Benchmark 有三个关键点：

- **Isolated dynamics**：每个任务只突出一种或少数几种物理规律，例如重力、碰撞、周期运动或旋转。
- **Reward-free evaluation**：训练和评测 world model 时不使用人工 reward，避免把“学会 reward shortcut”误认为“理解 dynamics”。
- **Long-horizon open-loop prediction**：用前几步真实状态 warm-up，然后不再喂真实 observation，让模型自己往后预测，观察误差如何随 horizon 增长。

本作业实现课程版 `SmallWorld-Lite`。它不是论文 MuJoCo benchmark 的完整复现，而是用轻量解析动力学实现同样的评测思想，保证 Colab 可运行、代码可读、指标可解释。

### 8.1 十个 SmallWorld-Lite 任务

本作业包含以下 10 个任务：

- `free_fall`：自由落体和地面弹跳，考察重力与反弹。
- `projectile`：抛体运动，考察水平匀速和竖直加速度叠加。
- `circular_motion`：圆周运动，考察半径约束和切向速度。
- `inclined_plane`：斜面运动，考察重力沿斜面的分解和摩擦。
- `simple_pendulum`：单摆，考察周期运动、相位和能量趋势。
- `rolling`：刚体滚动，考察平动和转动耦合。
- `rotation`：刚体旋转，考察角速度和角度状态更新。
- `spin`：陀螺式旋转失稳，考察旋转衰减和倾斜增长。
- `elastic_collision`：一维弹性碰撞，考察碰撞瞬间速度交换和能量趋势。
- `bouncing_ball`：二维盒中弹跳球，考察边界反射和长期误差累积。

这些任务的目的不是替代 `Pendulum-v1` 控制任务，而是补充它。`Pendulum-v1` 回答“world model 能不能帮助 actor 学 policy”，SmallWorld 回答“world model 本身是否学到了 dynamics”。

### 8.2 SmallWorld 运行方式

生成数据：

```bash
python smallworld_generate_dataset.py \
  --config configs/smallworld_config.json \
  --task all \
  --output-dir artifacts/smallworld_data
```

训练一个 reward-free dynamics model：

```bash
python smallworld_train.py \
  --config configs/smallworld_config.json \
  --task simple_pendulum \
  --dataset-dir artifacts/smallworld_data \
  --output-dir artifacts/smallworld_runs/simple_pendulum
```

评测：

```bash
python smallworld_eval.py \
  --checkpoint-dir artifacts/smallworld_runs/simple_pendulum/best_checkpoint \
  --dataset-dir artifacts/smallworld_data \
  --output-dir artifacts/smallworld_eval/simple_pendulum
```

可视化：

```bash
python smallworld_visualize.py \
  --checkpoint-dir artifacts/smallworld_runs/simple_pendulum/best_checkpoint \
  --dataset-dir artifacts/smallworld_data \
  --output-dir artifacts/smallworld_viz/simple_pendulum
```

### 8.3 SmallWorld 指标

SmallWorld 输出的主要指标包括：

- `one_step_state_rmse`：一步 state prediction error，越低越好。
- `open_loop_15_rmse`：15 步 open-loop state prediction error，越低越好。
- `open_loop_90_rmse`：论文式长时域 90 步 open-loop error，越低越好。
- `horizon_error_auc`：horizon error curve 的平均误差，衡量误差随时间增长的总体趋势。
- `energy_drift`：预测轨迹和真实轨迹的能量差异，适用于 pendulum、collision、bouncing、rolling 等任务。
- `constraint_violation`：预测是否违反几何/物理约束，例如圆周半径约束或 rolling no-slip 约束。
- `ood_open_loop_90_rmse`：在 OOD 初始条件或物理参数下的长时域误差。

SmallWorld 的可视化包括：

- `smallworld_rollout.png`：真实 state trajectory 和模型 open-loop prediction 的逐维对比。
- `smallworld_horizon_error.png`：误差随 prediction horizon 增长的曲线。
- `smallworld_rollout.mp4`：真实轨迹和模型预测 ghost trajectory 的叠加动画。

注意：`smallworld_rollout.mp4` 和 `demo_policy.mp4` 录制的不是同一件事。`demo_policy.mp4` 是 actor 在真实 `Pendulum-v1` 环境中执行 policy；`smallworld_rollout.mp4` 是固定真实 action sequence 后，比较真实物理轨迹和 world model 预测轨迹。

### 8.4 SmallWorld 中学生可以改什么

推荐改动：

- 修改 `smallworld_hw/models.py` 中 RSSM 容量、decoder、prior/posterior 结构。
- 修改 `configs/smallworld_config.json` 中 latent size、batch length、训练步数、loss weight。
- 修改 `smallworld_hw/tasks.py` 中某个任务的参数范围，用于研究 OOD generalization。
- 增加 deterministic latent ablation，比较 stochastic latent 是否必要。
- 增加 Transformer sequence model 作为 optional baseline，和 RSSM 比较 horizon error。

通常不要修改：

- `smallworld_eval.py` 的 metric 定义。
- dataset `.npz` schema。
- checkpoint restore 逻辑。

## 9. 任务 C-F：解释、改进、消融和提交

### Task C：解释 MiniDreamer

你需要在报告中解释以下概念，不要求数学推导很长，但必须说清楚代码对应关系：

- RSSM 中 deterministic state 和 stochastic state 分别是什么。
- posterior 和 prior 的区别是什么。
- 为什么需要 KL loss。
- decoder、reward head、continue head 分别预测什么。
- actor/critic 为什么可以在 imagined latent trajectory 上训练。
- 为什么 public eval 同时看 policy return 和 prediction error。

### Task D：做至少一个改进

你必须至少选择一个方向进行修改，并说明动机。

推荐方向：

- World model 容量：修改 `embed_dim`、`deter_dim`、`stoch_dim`、`hidden_dim`。
- Sequence learning：修改 `batch_length`、`prefill_steps`、replay 采样设置。
- KL / representation learning：修改 `free_nats`、`kl_scale`、`rep_kl_scale`。
- Imagination：修改 `imag_horizon`、`discount`、`lambda`。
- Actor-critic：修改 actor/critic learning rate、entropy scale、target update。
- Prediction heads：在 `models.py` 中修改 decoder/reward head 结构。
- Exploration：在 `course_config.json` 或 `agent.py` 中修改 exploration 行为。

你不需要保证所有指标都提升，但你必须解释 tradeoff。例如，open-loop prediction 变好但 return 没变好，也是有价值的发现。

### Task E：做一次 ablation

你需要比较至少两个实验：

- Baseline run
- Your modified run

报告中至少包含一个表格：

```text
Run              Steps   Mean Return   1-step RMSE   Open-loop RMSE   Reward MAE   Action Delta
baseline         100k    ...           ...           ...              ...          ...
your_method      <=250k  ...           ...           ...              ...          ...
```

### Task F：提交最终结果

最终提交只允许一个模型作为 final submission。

需要提交：

- `best_checkpoint/`
- `configs/colab_runtime_config.json`
- `artifacts/public_eval_bundle/public_eval.json`
- `artifacts/demo_bundle/demo_policy.mp4` 或 `.gif`
- `artifacts/demo_bundle/world_model_rollout.png`
- `artifacts/smallworld_eval/.../smallworld_eval.json`
- `artifacts/smallworld_viz/.../smallworld_rollout.png`
- `artifacts/smallworld_viz/.../smallworld_horizon_error.png`
- `artifacts/smallworld_viz/.../smallworld_rollout.mp4` 或 `.gif`，如果生成成功
- 修改后的代码
- `short_report.pdf`

## 10. 允许修改和禁止修改

推荐修改：

- `configs/course_config.json`
- `world_model_hw/models.py`
- `world_model_hw/agent.py`
- `configs/smallworld_config.json`
- `smallworld_hw/models.py`
- `smallworld_hw/tasks.py`

谨慎修改：

- `train.py`
- `world_model_hw/replay.py`
- `world_model_hw/envs.py`

通常不要修改：

- `public_eval.py`
- `generate_public_rollout.py`
- checkpoint 文件格式
- rollout `.npz` 字段名
- benchmark seeds

如果你修改 public evaluation 或 rollout schema，你的结果将不能和其他同学公平比较。

## 11. 评分标准

总分 100 分。

- 15 分：Colab 和工程链路跑通，包括 Pendulum smoke、SmallWorld smoke、checkpoint、demo、public eval。
- 15 分：成功复现 `100k` Pendulum baseline，并记录完整训练和评测结果。
- 15 分：正确解释 world model、RSSM、KL、imagination actor-critic 的概念和代码对应关系。
- 15 分：正确解释 SmallWorld benchmark 的 reward-free / isolated / long-horizon 评测思想。
- 15 分：完成至少一个有动机的改进，并进行 baseline vs modified 对比。
- 15 分：实验分析质量，包括失败尝试、metric tradeoff、为什么改动有效或无效。
- 10 分：提交规范，包括文件齐全、代码可运行、报告清晰。

加分项最多 10 分：

- 清晰的 ablation，多于一个改动方向但控制变量合理。
- 额外可视化 latent rollout 或 prediction error。
- 对 DreamerV3 论文思想和本作业 MiniDreamer 简化版的差异有准确讨论。
- 在多个 SmallWorld 任务上比较同一个模型改动的效果，并解释不同物理规律上的失败模式。

## 12. 报告要求

报告建议 3-4 页，必须回答：

1. 这个作业中的 world model 输入、latent state、输出分别是什么？
2. RSSM prior/posterior 是如何训练的？
3. Actor 和 critic 是如何通过 imagination 训练的？
4. Baseline 的 public evaluation 结果是什么？
5. 你修改了什么？为什么这样改？
6. 哪些指标提升了？哪些指标变差了？
7. 你失败过什么方案？你从失败中学到了什么？
8. SmallWorld 为什么不使用 reward？它和 Pendulum policy return 评测互补在哪里？
9. 至少一个 SmallWorld 任务中，90-step open-loop prediction 是如何失败或成功的？

## 13. 如何判断训练是否成功

如果只是 `local_smoke`：

- 看到 `best_checkpoint/`，说明训练和 checkpoint 成功。
- 看到 `public_eval.json`，说明评测管线成功。
- return 不好是正常的，因为只有 `2k steps`。

如果是正式 baseline：

- 你应该看到多个 eval 点，而不是只有一个。
- best checkpoint 不一定是最后一个 checkpoint。
- `eval/mean_return` 应该比随机策略更稳定。
- world model prediction plot 应该至少在短 horizon 上有趋势一致性。

如果出现以下情况，需要 debug：

- loss 是 `nan`。
- checkpoint 没有生成。
- public rollout 字段缺失。
- 所有 episode return 都极差且 action 几乎不变。
- open-loop prediction 完全发散，且 one-step prediction 也很差。

如果是 SmallWorld：

- 看到 `smallworld_eval.json`，说明 benchmark scoring 成功。
- 看到 `smallworld_rollout.png` 和 `smallworld_horizon_error.png`，说明可视化成功。
- smoke run 的误差不一定低，因为训练 update 很少。
- 正式 run 中，`one_step_state_rmse` 应明显低于长时域误差。
- 如果 `open_loop_90_rmse` 很差但 `one_step_state_rmse` 还可以，说明模型存在典型 error accumulation。

## 14. 高层意义

这份作业的核心价值是让你理解：智能体不一定只能通过真实试错学习。World model 的意义在于把环境动力学压缩进一个可预测的 latent space，让 agent 能在模型内部做低成本 planning / imagination。

Dreamer 的关键思想可以概括为：

```text
真实环境数据 -> 学习 world model -> 在 latent world 中想象未来 -> 训练 actor/critic -> 回到真实环境收集更好数据
```

这也是很多 embodied AI、robotics、autonomous driving 和 generalist agent 系统中反复出现的思想：先学会预测世界，再学会在预测的世界中做决策。

SmallWorld 补充了另一个关键思想：

```text
去掉 reward 和 policy -> 固定 state/action 数据 -> 长时域 open-loop 预测 -> 诊断 world model 是否理解 dynamics
```

这两条线合在一起，构成了本作业的核心：既理解 Dreamer 如何用 world model 训练控制策略，也理解如何独立评测 world model 本身的动力学理解能力。
