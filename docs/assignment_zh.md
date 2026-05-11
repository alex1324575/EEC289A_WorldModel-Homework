# 作业：InvertedPendulum VPT Scoreboard World Model

本作业只训练一个 **world model**。不训练 policy，不使用 reward，不实现 actor/critic，也不要求学生写 MuJoCo XML。Gymnasium `InvertedPendulum-v5` 背后的 MuJoCo 是 ground-truth physics simulator。

核心问题：

```text
s_hat[t+1] = f_theta(s_hat[t], a[t])
```

模型先看 10 步真实状态 warm-up，然后只使用自己的预测状态递归 rollout，最多评估 990 步。主指标是 **VPT80@0.25**：

```text
至少 80% windows 的 normalized state MSE 仍低于 0.25 时，模型最多能预测到第几步。
```

## 1. 为什么不用 policy return？

World model 经常被用于辅助 policy，但 policy return 会把很多问题混在一起：policy 可能避开模型不准的区域，reward 也可能掩盖物理预测误差。本作业把 world model 单独拿出来问：

```text
给定同一串 action，learned world model 能和 MuJoCo ground truth 保持一致多久？
```

这就是论文中 isolated world-model evaluation 思想在课程作业里的简化实现。

## 2. 数据格式

状态：

```text
s = [x, theta, x_dot, theta_dot]
```

动作：

```text
a = [u], u in [-3, 3]
```

窗口长度由 config 推导：

```text
states:  [N, warmup_steps + max_horizon + 1, 4]
actions: [N, warmup_steps + max_horizon, 1]
```

正式 scoreboard 默认：

```text
warmup_steps = 10
max_horizon = 990
states       = [N, 1001, 4]
actions      = [N, 1000, 1]
```

索引规则：

```text
warmup 使用真实 s_0 ... s_9
current = s_10

h = 1:
  使用 a_10
  预测 s_hat_11
  对比 s_11

h = 990:
  使用 a_999
  预测 s_hat_1000
  对比 s_1000
```

数据由 locked generator 生成。action sequence 来自固定 stabilizing feedback 加平滑噪声：

```text
a_t = clip(-K s_t + eps_t, -3, 3)
eps_t = 0.9 eps_{t-1} + sigma xi_t
```

学生不会训练或提交 controller。

## 3. 学生修改范围

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
wm_hw/official_rollout.py
wm_hw/official_metrics.py
wm_hw/eval_horizon.py
hidden seeds
baseline model
```

官方评分使用 locked official rollout 和 locked official metrics，不依赖学生的 `student/metrics.py`。

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

模型输出 normalized delta observation。locked helper 会执行：

```text
normalize obs/action
model predicts normalized delta
denormalize delta
next_obs = obs + delta
```

## 5. Rollout

学生实现或改进：

```python
open_loop_rollout(model, states, actions, normalizer, warmup_steps, horizon)
```

核心要求：

```text
warmup 阶段可以使用 ground-truth states
warmup 后不能再使用 future ground-truth states
```

也就是说，预测 `s_hat_11` 后，下一步必须用 `s_hat_11` 作为输入，而不是用真实 `s_11`。

## 6. Loss 和训练成本

默认训练 loss：

```text
L = L_1step + lambda_rollout L_rollout
```

推荐默认：

```text
one_step_weight = 1.0
rollout_weight = 1.0
rollout_train_horizon = 15
```

正式数据窗口可以有 1000 个 action。为了避免 Colab 成本失控，训练不会默认在整段 990-step window 上 backprop。`training.train_sequence_length` 会随机截取短 subwindow；`rollout_train_horizon` 控制短 BPTT rollout loss。最终评估仍然可以跑 990 步。

## 7. Official Metrics

官方指标是 SmallWorlds-style normalized MSE 和 valid prediction time。

normalized MSE：

```text
nMSE_h = mean(((s_hat_h - s_h) / obs_std_train)^2)
```

报告：

```text
nMSE@1
nMSE@5
nMSE@10
nMSE@90
nMSE@100
nMSE@200
nMSE@500
nMSE@990
nMSE_AUC
VPT80@0.10
VPT80@0.25   # primary metric
VPT80@0.50
VPT50@0.25
```

`VPT80@0.25` 的意思是：每个 window 一旦 nMSE 超过 0.25 就视为该 window 不再 valid；看至少 80% windows 仍 valid 的最大 horizon。

## 8. Colab 流程

安装和快速测试：

```bash
python -m pip install -r requirements.txt
pytest -q -m "not slow"
```

轻量 dev dataset：

```bash
python -m wm_hw.dataset --config configs/dev.yaml --output-dir data/dev --smoke
```

训练 baseline 和 student：

```bash
python -m wm_hw.train --config configs/baseline.yaml --model baseline --dataset-dir data/dev --output-dir artifacts/baseline --smoke
python -m wm_hw.train --config configs/student.yaml --model student --dataset-dir data/dev --output-dir artifacts/student --smoke
```

官方评估：

```bash
python -m wm_hw.eval_horizon \
  --checkpoint-dir artifacts/student/best_checkpoint \
  --dataset-dir data/dev \
  --split test \
  --horizon auto \
  --eval-config configs/official_eval.yaml \
  --output-dir artifacts/student/eval_test
```

画图：

```bash
python -m wm_hw.plotting --eval-dir artifacts/student/eval_test --output-dir artifacts/student/plots
```

## 9. Final Scoreboard

正式 public scoreboard：

```bash
python -m wm_hw.dataset --config configs/public_scoreboard.yaml --output-dir data/public_scoreboard
python -m wm_hw.eval_horizon \
  --checkpoint-dir artifacts/student/best_checkpoint \
  --dataset-dir data/public_scoreboard \
  --split test \
  --horizon auto \
  --eval-config configs/official_eval.yaml \
  --output-dir artifacts/student/public_scoreboard_test
```

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

1. 为什么 one-step prediction 很好也可能 long-horizon drift？
2. 你的模型结构改了什么？
3. rollout loss 对 nMSE_AUC 和 VPT80@0.25 有什么影响？
4. OOD split 为什么更难？
5. 你的模型最早在哪些 state 维度上偏离？
6. test 和 OOD 的 VPT80@0.25 / nMSE@990 有什么差异？

## 11. 评分建议

```text
open-loop rollout correctness: 20
world model implementation: 20
multi-step rollout loss: 20
nMSE/VPT analysis: 20
hidden VPT80@0.25 improvement over baseline: 10
reproducibility/artifacts: 10
```
