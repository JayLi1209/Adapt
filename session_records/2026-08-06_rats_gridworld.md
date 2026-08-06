# Session Record — 2026-08-06 RATS vs ADA-MCTS Gridworld（修复后重跑 + 待办调查）

> 承接 `2026-08-05_rats_gridworld.md`。当日进度以 `## Progress Log` 追加。
> 用户要求：中文交流（术语/log 可用英文）。

## Progress Log

- **2026-08-06**: 创建本文档。完成非平稳 schedule bug 的完整实验重跑
  （cliff 30 trials、bridge 100 trials，6 方法 × 6 个 p 值，γ=0.99，m-sim=1000），
  全部数字修正为"环境真实变化"下的结果。两个待办调查启动：
  (1) K_FORGET/forget 策略调优（cliff 上 adaptive < static）；
  (2) ada_mcts bridge 配置问题（stationary 0.68 vs 0.91）。
  commit：74a9a7f（08-05 记录修正）。

## 1. 做了什么（08-06）

1. **发现并修复非平稳 schedule bug**（commit ca0cb99，详见 08-05 记录 §5.6）：
   `p_schedule=[(0,0.7),(0,p_new)]` 因 `sorted()` 反序，`active_p(t)` 恒返回 0.7，
   环境从未变化。修复后 `change_step<=0` 时省略 ORIG_P 段 + `active_p_fn` 拒绝
   重复时间戳。
2. **完整实验重跑**（本次记录的核心）：cliffwalking + bridge，6 方法 × 6 个
   p 值（0.4/0.5/0.6/0.8/0.9/1.0），环境在 ts 0 真实变为 p_new。
3. 提交 stationary 修复（ffb6a75，08-05 遗留）并验证 adaptive==static。

## 2. 实验 setting

| 项 | cliffwalking | bridge |
|---|---|---|
| 网格 | 4x12（CLIFFWALKING_4x12, grids.py） | 5x8（BRIDGE_5x8） |
| K 方向 | K=3（intended + 2 垂直滑移） | K=2（intended vs opposite 掉头） |
| slip_dist(p) | [p, (1-p)/2, (1-p)/2] | [p, 1-p] |
| 原环境 p | 0.7（预训练目标，BNN Dirichlet head） | 0.7 |
| 变化 | ts 0 起 p → {0.4,0.5,0.6,0.8,0.9,1.0}，首步即在 p_new | 同左 |
| γ | 0.99（用户指定） | 0.99 |
| max_depth (RATS/DP) | 6 | 6 |
| max_steps（截断） | 100 | 10 |
| trials | 30 | 100 |
| heuristic | "potential" = γ^dist(s,goal) | 同左 |
| K_FORGET | 5 | 5 |
| ADA-MCTS | M_SIMULATIONS=1000, CP=√2, EPS_E=0.02, H_ROLLOUT=6 | 同左 |
| reward | G=+1, H(悬崖/洞)=-1, 其余 0；goal rate 统计中 H 记 0（paper 约定） | 同左 |

方法集合（run_gridworld_experiments.py）：
- `dp_nsmdp`（oracle，DP-NSMDP：用真实时变 p(t) 计划）
- `dp_snapshot`（oracle，DP 快照：用真实当前 p 计划）
- `oracle_rats`（oracle，RATS 用真实当前 p 快照）
- `bnn_rats_static`（学习，RATS + 预训练 BNN 快照，无自适应）
- `bnn_rats_adaptive`（学习，RATS + surprise/forget/online-counts 自适应）
- `ada_mcts`（baseline，ADA-MCTS：被通知变化，DPAS 双相采样）

预训练模型（已 push）：`data/cliffwalking/bnn_dirichlet_cliffwalking_k3_p0p7.pth`
（MAE=0.004）、`data/bridge/bnn_dirichlet_bridge_k2_p0p7.pth`（MAE=0.009）。

## 3. 实验结果（修复后，环境真实变化）

### 3.1 cliffwalking — goal rate by p（holes=0）
```
method              p=0.4   p=0.5   p=0.6   p=0.8   p=0.9   p=1.0
dp_nsmdp             1.000   0.867   1.000   1.000   1.000   1.000
dp_snapshot          1.000   0.867   1.000   1.000   1.000   1.000
oracle_rats          0.633   0.700   0.867   1.000   1.000   1.000
bnn_rats_static      0.200   0.533   0.733   1.000   1.000   1.000
bnn_rats_adaptive    0.067   0.400   0.567   0.933   1.000   1.000
ada_mcts             0.467   0.633   0.800   0.933   0.967   1.000
```
γ=0.99 折现 return（holes=-1）：dp 0.56–0.89 / oracle_rats 0.38–0.89 /
bnn_static 0.10–0.87 / bnn_adaptive 0.04–0.87 / ada_mcts 0.25–0.61

### 3.2 bridge — goal rate by p
```
method              p=0.4   p=0.5   p=0.6   p=0.8   p=0.9   p=1.0
dp_nsmdp             0.730   0.550   0.770   0.980   1.000   1.000
dp_snapshot          0.730   0.550   0.770   0.980   1.000   1.000
oracle_rats          0.730   0.550   0.770   0.980   1.000   1.000
bnn_rats_static      0.360   0.540   0.770   0.980   1.000   1.000
bnn_rats_adaptive    0.370   0.530   0.760   0.980   1.000   1.000
ada_mcts             0.050   0.090   0.190   0.310   0.370   0.460
```
γ=0.99 return：oracle 0.52–0.98 / BNN 0.34–0.98 / ada_mcts −0.80 至 −0.07

### 3.3 stationary 验证（p=0.7，修复后干净数字）
- cliff：dp 0.784/1.000、oracle_rats 0.590/0.900、bnn_static 0.699/1.000、
  bnn_adaptive 0.699/1.000（=static，修复生效）、ada_mcts 0.770/1.000
- bridge：dp/oracle_rats/bnn_static/bnn_adaptive 均 0.875/0.910；
  **ada_mcts 0.410/0.680**（旧数字 −0.562/0.200 受 stationary 误触发 bug 污染）
- 结论：预训练模型 stationary 表现好（cliff 1.000、bridge 0.910 goal rate）

## 4. 实验分析

1. **oracle DP 几乎满格**（cliff p=0.5 时 0.867：slip [0.5,0.25,0.25] 下最优策略
   含 ~13% 风险，真实效应）。
2. **oracle_rats 在 cliff p≤0.6 低于 dp**（p=0.4：0.633 vs 1.000）：RATS
   worst-case 半径 c=d·L_p·tau 随深度线性增大（d≤6），真实模型下的最优期望
   策略被过度保守化。与 paper 不同：paper 的不确定半径来自学习模型 posterior，
   我们 oracle 用 L_p=1.0 满半径。bridge 上 RATS==DP（worst-case 与期望一致）。
3. **BNN 快照在 p≤0.6 崩**（cliff p=0.4 仅 0.20）：模型在 0.7 训练、后验紧致、
   "自信地错"→ 沿悬崖激进走位 → 掉洞。
4. **adaptive < static（cliff p≤0.6，如 p=0.4：0.067 vs 0.200）**：K_FORGET=5
   每 5 步 forget + 在线 counts 被 20000 预训练 counts 稀释，自适应反而伤害
   模型。bridge 上 adaptive≈static 因 episode 仅 3–4 步，循环来不及作用。
5. **ada_mcts**：cliff 低 p 反而超过 BNN-RATS（p=0.4：0.467 vs 0.200/0.067，
   在线学习适应快），但 return 恒低于 oracle（p=1.0 goal rate 1.000 但 return
   0.61 vs 0.89，路径绕远）；bridge 上全面最差（stationary 0.68 就弱）。
   bridge 的掉头 slip 可能让 MCTS rollout/heuristic 失效。
6. bridge p=0.5 低谷（0.55）是真实结构效应：掉头 slip 下 p=0.5 时前进/后退
   各半，期望位移 0；p=0.4 时"反向意图"反而可利用掉头（slip 0.6 推向目标）。

## 5. 待办调查（进行中）

1. **K_FORGET/forget 策略调优**：cliff p≤0.6 上 adaptive 变差，扫描
   K_FORGET ∈ {1,3,10,20,∞}（及 surprise 阈值/retain 衰减），找 adaptive ≥ static
   的配置。只跑 BNN 方法，cliff 30 trials，很快。
2. **ada_mcts bridge 配置**：stationary 0.68 vs 0.91。排查 M_SIMULATIONS、
   EPS_E、rollout heuristic、DPAS 双相采样与桥掉头 slip 的交互。

## 6. 复现命令

```bash
python run_gridworld_experiments.py --grid cliffwalking --trials 30 --m-simulations 1000
python run_gridworld_experiments.py --grid bridge --trials 100 --m-simulations 1000
# 只跑指定方法（快）：
python run_gridworld_experiments.py --grid cliffwalking --trials 30 --methods bnn_rats_adaptive bnn_rats_static
```

## 7. 关键文件

- `planning/rats.py` — RATS + DP 基线
- `planning/ada_mcts.py` — ADA-MCTS baseline
- `run_gridworld_experiments.py` — 实验 runner（schedule 修复于 L54-74/L376-382）
- `bnn/dirichlet_workflow.py`（surprise_dirichlet/forget_dirichlet）、`bnn/dirichlet_model.py`、
  `drift.py`（DriftFilterV2）、`config.py`（ETA/GAMMA_UNCERTAINTY）
- 结果 log：`/tmp/grid_cliff_fixed.log`、`/tmp/grid_bridge_fixed.log`（08-06 重跑）
- 08-05 记录：`session_records/2026-08-05_rats_gridworld.md`
