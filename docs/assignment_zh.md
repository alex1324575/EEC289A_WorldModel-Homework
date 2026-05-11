# 作业：InvertedPendulum Survival Scoreboard World Model

本作业只训练 **world model**，不训练 policy，不使用 reward，不实现 actor/critic，也不维护自定义 MuJoCo XML。环境固定为 Gymnasium 官方 `InvertedPendulum-v5`。

核心问题：

```text
s_hat[t+1] = f_theta(s_hat[t], a[t])
```

给定 MuJoCo 真实 state/action trajectory，模型先看一小段真实状态作为 warm-up，然后只使用自己的预测状态递归 rollout，直到达到配置的 scoreboard horizon 或发生 failure。主指标是 **H80**：

```text
至少 80% 测试 windows 仍未 fail 的最大预测步数
```

## 1. 为什么是 survival scoreboard？

World model 有时用于辅助 policy，但如果只看 policy return，很难判断 world model 是否真的学到了物理规律。这里 MuJoCo 是 ground-truth physics simulator，学生训练的模型是 learned simulator。我们不问它拿多少 reward，而问：

```text
在同一串 action 下，world model 能跟 MuJoCo ground truth 保持一致多少步？
```

因此本作业不是固定预测 100 步，而是用 survival curve 看模型“撑到哪里才崩”。普通 Colab 默认轻量；老师或助教可以用 scoreboard 配置跑 500/1000 步。

## 2. 环境和数据

环境：

```text
gymnasium.make("InvertedPendulum-v5")
```

状态：

```text
s = [x, theta, x_dot, theta_dot]
```

动作：

```text
a = [u], u in [-3, 3]
```

数据窗口由 config 推导：

```text
warmup_steps = 5
max_horizon = configurable
states:  [N, warmup_steps + max_horizon + 1, 4]
actions: [N, warmup_steps + max_horizon, 1]
```

索引规则：

```text
warmup: t = 0..4
current = s_5

h = 1:
  use a_5
  predict s_hat_6
  compare to s_6

h = max_horizon:
  use a_{warmup + h - 1}
  predict s_hat_{warmup + h}
  compare to s_{warmup + h}
```

数据由 locked generator 生成。学生不会训练或提交 controller。action sequence 来自固定 stabilizing feedback 加平滑噪声：

```text
a_t = clip(-K s_t + eps_t, -3, 3)
eps_t = 0.9 eps_{t-1} + sigma xi_t
```

只保留真实 MuJoCo trajectory 在整个 window 内满足：

```text
abs(theta_true) < 0.20
all state values finite
```

## 3. 学生只能修改什么

允许修改：

```text
student/model.py
student/rollout.py
student/losses.py
student/metrics.py
configs/student.yaml
```

不要修改：

```text
wm_hw/env.py
wm_hw/dataset.py
wm_hw/eval_horizon.py
official thresholds
hidden seeds
baseline model
```

## 4. 模型接口

学生模型必须保持接口：

```python
class StudentWorldModel(nn.Module):
    def initial_hidden(self, batch_size: int, device: torch.device):
        return None

    def forward(self, obs_norm, act_norm, hidden=None):
        """
        obs_norm: [B, 4]
        act_norm: [B, 1]
        returns:
            delta_obs_norm: [B, 4]
            next_hidden
        """
```

模型输出 normalized delta observation。locked helper 会负责：

```text
normalize obs/action
model predicts normalized delta
denormalize delta
next_obs = obs + delta
```

## 5. Rollout 实现

你需要实现或改进：

```python
open_loop_rollout(model, states, actions, normalizer, warmup_steps, horizon)
```

核心要求：

```text
warmup 阶段可以使用 ground-truth states
warmup 后不能再偷看 future ground-truth states
```

也就是说，预测 `s_hat_6` 后，下一步必须用 `s_hat_6` 作为输入，而不是用真实 `s_6`。

## 6. Loss

默认训练 loss：

```text
L = L_1step + lambda_rollout L_rollout
```

默认：

```text
one_step_weight = 1.0
rollout_weight = 1.0
rollout_train_horizon = 15
```

训练时只要求 short local rollout stability；scoreboard 可以测试更长 horizon。这个差距正是作业想让你观察的 compounding error。建议报告中比较 `rollout_train_horizon = 1 / 5 / 15 / 30` 对 H80 的影响。

## 7. Failure Horizon Metric

默认 failure thresholds：

```text
angle_error_rad = 0.075
cart_pos_error_m = 0.10
cart_vel_error_mps = 0.75
pole_vel_error_radps = 1.00
consecutive_fail_steps = 2
```

单步 violation：

```text
abs(pred_theta - true_theta) > 0.075
or abs(pred_x - true_x) > 0.10
or abs(pred_xdot - true_xdot) > 0.75
or abs(pred_thetadot - true_thetadot) > 1.00
```

连续 2 步 violation 才算 fail。每个 window 的 survival horizon：

```text
如果第 h 步开始连续 violation:
  survival = h - 1

如果 max_horizon 内没有 fail:
  survival = max_horizon
```

主指标：

```text
H80
```

辅助指标：

```text
H50
survival_auc
mean_survival_steps
median_survival_steps
success_rate@configured_milestones
one_step_rmse
open_loop_rmse@horizon
```

## 8. Colab 运行流程

安装和测试：

```bash
python -m pip install -r requirements.txt
pytest -q
```

生成 public dataset：

```bash
python -m wm_hw.dataset --config configs/colab.yaml --output-dir data/public
```

训练：

```bash
python -m wm_hw.train --config configs/baseline.yaml --model baseline --dataset-dir data/public --output-dir artifacts/baseline
python -m wm_hw.train --config configs/student.yaml --model student --dataset-dir data/public --output-dir artifacts/student
```

评估：

```bash
python -m wm_hw.eval_horizon --checkpoint-dir artifacts/student/best_checkpoint --dataset-dir data/public --split test --horizon auto --output-dir artifacts/student/eval_test
python -m wm_hw.eval_horizon --checkpoint-dir artifacts/student/best_checkpoint --dataset-dir data/public --split ood --horizon auto --output-dir artifacts/student/eval_ood
```

画图：

```bash
python -m wm_hw.plotting --eval-dir artifacts/student/eval_test --output-dir artifacts/student/plots
```

## 9. Scoreboard / Leaderboard

轻量配置：

```text
configs/colab.yaml      max_horizon = 100
```

长 horizon 配置：

```text
configs/scoreboard.yaml max_horizon = 1000
```

老师或助教可以运行：

```bash
python -m wm_hw.dataset --config configs/scoreboard.yaml --output-dir data/scoreboard
python -m wm_hw.eval_horizon --checkpoint-dir artifacts/student/best_checkpoint --dataset-dir data/scoreboard --split test --horizon auto --output-dir artifacts/student/scoreboard_test
```

如果要改成 500 或 2000 步，只改 `configs/scoreboard.yaml` 的 `max_horizon`，不需要改代码。

## 10. 提交内容

```text
student/model.py
student/rollout.py
student/losses.py
student/metrics.py
configs/student.yaml
artifacts/student/eval_test/metrics.json
artifacts/student/eval_ood/metrics.json
artifacts/student/plots/survival_curve.png
artifacts/student/plots/rollout_comparison.png
checkpoint or checkpoint link
Git commit hash
short_report.pdf
```

报告最多 4 页，回答：

1. baseline one-step MLP 为什么会在 long-horizon rollout 漂移？
2. 你的模型结构改了什么？
3. multi-step rollout loss 对 H80 和 survival curve 有什么影响？
4. OOD split 为什么更难？
5. 你的模型最早在哪些 state 维度上 fail？
6. test 和 OOD 的 H80/H50/survival_auc 有什么差异？

## 11. 评分

```text
open-loop rollout correctness: 20
world model implementation: 20
multi-step rollout loss: 20
failure horizon metric: 20
hidden H80 improvement over baseline: 10
reproducibility/artifacts: 10
```

官方评分既会看学生提交的 checkpoint，也可能用隐藏数据重新训练或抽查学生代码。
