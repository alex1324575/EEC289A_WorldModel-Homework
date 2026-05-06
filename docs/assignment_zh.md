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

## 2. 作业环境与代码入口

本作业使用课程自带的 PyTorch `MiniDreamer`，运行环境是 Google Colab。

- 环境：`Pendulum-v1`
- Observation：Gymnasium state vector
- Action：连续动作，agent 内部归一化到 `[-1, 1]`
- 算法：RSSM world model + latent imagination actor-critic
- 框架：PyTorch + Gymnasium
- Notebook：`notebooks/dreamer_public_colab_template.ipynb`

你应该优先阅读这些文件：

- `world_model_hw/models.py`
- `world_model_hw/agent.py`
- `train.py`
- `public_eval.py`
- `configs/course_config.json`

## 3. Baseline 要求

`local_smoke` 不是正式 baseline。它只用来检查 Colab、依赖安装、训练循环、checkpoint、视频和 public eval 是否能跑通。

正式训练预算如下：

- Debug smoke：`2k env steps`
- Required baseline：`100k env steps`
- Final / leaderboard maximum：`250k env steps`

你必须至少完成：

1. 跑通 `local_smoke`，确认工程链路正常。
2. 跑通 `baseline`，预算为 `100k env steps`。
3. 基于 baseline 做至少一个有理由的改动。
4. 在不超过 `250k env steps` 的总预算内提交最终模型。

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

   这个视频回答的问题是：policy 在真实 Gymnasium 物理动力学环境里到底怎么动？Pendulum 是否能被稳定摆起或控制？

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

   这个图回答的问题是：world model open-loop 预测出来的 observation trajectory 是否跟真实 trajectory 有相似趋势？

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

## 8. 任务 C-F：解释、改进、消融和提交

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
- 修改后的代码
- `short_report.pdf`

## 9. 允许修改和禁止修改

推荐修改：

- `configs/course_config.json`
- `world_model_hw/models.py`
- `world_model_hw/agent.py`

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

## 10. 评分标准

总分 100 分。

- 15 分：Colab 和工程链路跑通，包括 smoke test、checkpoint、demo、public eval。
- 20 分：成功复现 `100k` baseline，并记录完整训练和评测结果。
- 20 分：正确解释 world model、RSSM、KL、imagination actor-critic 的概念和代码对应关系。
- 20 分：完成至少一个有动机的改进，并进行 baseline vs modified 对比。
- 15 分：实验分析质量，包括失败尝试、metric tradeoff、为什么改动有效或无效。
- 10 分：提交规范，包括文件齐全、代码可运行、报告清晰。

加分项最多 10 分：

- 清晰的 ablation，多于一个改动方向但控制变量合理。
- 额外可视化 latent rollout 或 prediction error。
- 对 DreamerV3 论文思想和本作业 MiniDreamer 简化版的差异有准确讨论。

## 11. 报告要求

报告建议 3-4 页，必须回答：

1. 这个作业中的 world model 输入、latent state、输出分别是什么？
2. RSSM prior/posterior 是如何训练的？
3. Actor 和 critic 是如何通过 imagination 训练的？
4. Baseline 的 public evaluation 结果是什么？
5. 你修改了什么？为什么这样改？
6. 哪些指标提升了？哪些指标变差了？
7. 你失败过什么方案？你从失败中学到了什么？

## 12. 如何判断训练是否成功

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

## 13. 高层意义

这份作业的核心价值是让你理解：智能体不一定只能通过真实试错学习。World model 的意义在于把环境动力学压缩进一个可预测的 latent space，让 agent 能在模型内部做低成本 planning / imagination。

Dreamer 的关键思想可以概括为：

```text
真实环境数据 -> 学习 world model -> 在 latent world 中想象未来 -> 训练 actor/critic -> 回到真实环境收集更好数据
```

这也是很多 embodied AI、robotics、autonomous driving 和 generalist agent 系统中反复出现的思想：先学会预测世界，再学会在预测的世界中做决策。
