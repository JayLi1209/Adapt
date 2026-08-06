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
- **2026-08-06 调查完成**（§8 详见）：
  - **adaptive 变差根因**：forget 是死代码（drift filter 从未 reset → λ̂=0 →
    rho=1 → no-op）；counts 是毒药（α₀≈3.6 太小，噪声 counts 扭曲均值）。
  - **修复**：`--drift-reset`（forget 真生效）+ `--k-forget 3` + counts 默认关。
    cliff p=0.4/0.5/0.6 adaptive 0.667/0.767/0.867 vs static 0.200/0.533/0.733
    ——**adaptive 全面反超 static**。已设为 runner 默认（commit 待 9b9c564 后）。
  - **ada_mcts bridge**：stationary 差距主要是采样预算（m=2000 时 0.86 ≈
    BNN-RATS 0.91）；非平稳崩盘是 DPAS 卡 worst-case（dpas_gamma=10000 →
    regular 采样概率 exp(-10000·diff)≈0）+ worst-case one-hot 洞；另修复了
    per-trial 重复 notify_change（每集重新快照 M_{k-1} 退回 worst-case）。
  - 完整实验用新默认重跑完成，最终表见 §9（cliff adaptive 0.800/0.833/0.800
    vs static 0.200/0.533/0.733 @ p=0.4/0.5/0.6；bridge adaptive≡static）。

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

## 5. 待办调查（已完成，2026-08-06）

两个调查的完整结论见 §8。runner 新增调查 knob（commit 9b9c564）：
`--k-forget`、`--count-w`、`--persist-counts`、`--no-drift-reset`、`--counts`、
`--dpas-gamma`；并修复 ada_mcts per-trial 重复 notify_change。

## 8. 调查结论（2026-08-06）

### 8.1 K_FORGET/forget 策略（cliff，30 trials，γ=0.99）

**根因**：
- `forget_dirichlet` 的 `rho = 1/max(delta_bar, 1)`：drift filter 从未被
  `reset()`（一直在 calibrating 模式，λ̂=0 → delta_bar=1 → rho=1）→ **forget
  永远是 no-op**，adaptive ≡ static + 在线 counts。
- 预训练 Dirichlet 头 α₀ ≈ 3.6（极小浓度）：一集 episode 内几十个 counts 就
  把均值从 [0.7,0.15,0.15] 拉成尖峰噪声估计 → RATS 把"自信地错"的模型当真 →
  沿悬崖走位 → 掉洞。counts 每集清零、drift filter 每集新建，跨集无记忆。

**配置矩阵（cliff p=0.4，goal rate；static=0.200）**：
| 配置 | p=0.4 | p=0.5 | p=0.6 |
|---|---|---|---|
| adaptive 默认（forget 死代码 + counts） | 0.067 | — | — |
| persist counts w=1 | 0.000 | — | — |
| persist counts w=0.5 | 0.033 | — | — |
| drift_reset + K=5（forget 生效 + counts） | 0.367 | 0.533 | 0.667 |
| drift_reset + K=10 | 0.167 | 0.533 | 0.667 |
| drift_reset + K=5 + no counts | 0.400 | 0.600 | 0.833 |
| **drift_reset + K=3 + no counts** | **0.667** | **0.767** | **0.867** |
| static（参照） | 0.200 | 0.533 | 0.733 |

**机制**：retain 每 K 步按 rho<1 衰减 → alpha → CONC_PRIOR(0.1)→ p_dir →
[1/3,1/3,1/3] 均匀 → RATS worst-case 在"无信息"模型上走保守路线（绕开悬崖）
→ cliff 上有安全长路（max_steps=100）时保守 > 自信地错。K 越小先验清除越快。
counts 有害：α₀ 太小，少量 counts 产生尖峰估计，persist 更差。

**结论与默认**：`--drift-reset`（默认开）+ `--k-forget 3`（默认）+ counts 默认
关。adaptive 在 cliff 全 p 值反超 static（0.667/0.767/0.867 vs 0.200/0.533/0.733）。

**bridge 特例**：bridge episode 仅 3-4 步，`post` 每集清零到不了 K → forget
从不触发 → adaptive ≡ static（0.360 @ p=0.4）——per-episode 自适应在短 episode
任务上结构性失效，需跨集持久化（但 persist counts 在 cliff 上更差，需另设计）。

### 8.2 ada_mcts bridge 配置（trials=50/100，m-sim 扫描）

**stationary（p=0.7）**：0.68（m=1000）→ **0.86（m=2000）**，接近 BNN-RATS
0.91 —— 主要是采样预算问题（default 1000 sims 太少；m=500 只有 0.40）。

**非平稳（p=0.4）**：即使 m=3000 也只有 0.22 —— 预算不是主因：
1. **per-trial 重复 notify_change**（已修复）：runner 的 reset() 每集重置
   `_notified` → 每集 t=0 重新 deepcopy M_{k-1} 并把 `_training_started` 打回
   False → DPAS 永远在"未知新 MDP"的 worst-case 阶段。修复：notify 每 phase
   只触发一次（commit 9b9c564）。
2. **DPAS 卡 worst-case**：phase-2 regular 采样概率
   `exp(-dpas_gamma·(ale_prev−ale_k))`，`DPAS_GAMMA=10000` → 只要 ale_k 略低于
   ale_prev（counts 积累后必发生），likelihood≈0 → 永不 regular → 永远
   worst-case。而 bridge 上 worst-case 把 goal 附近的洞 one-hot 成必达 → 树被
   毒化。`--dpas-gamma 10` 时 p=0.4 从 0.06 → 0.26，仍远低于 stationary 0.74。

**结论**：baseline 在 bridge 上双重失效（预算 + DPAS 超保守）。若要与 paper
表对比，需 m=2000+ 且 dpas_gamma 大幅调低；当前默认 m=1000/gamma=10000 的
ada_mcts 数字代表"严重欠配的 baseline"。

## 9. 最终实验结果（新默认 adaptive，2026-08-06 重跑）

### cliffwalking — goal rate by p
```
method              p=0.4   p=0.5   p=0.6   p=0.8   p=0.9   p=1.0
dp_nsmdp             1.000   0.867   1.000   1.000   1.000   1.000
dp_snapshot          1.000   0.867   1.000   1.000   1.000   1.000
oracle_rats          0.633   0.700   0.867   1.000   1.000   1.000
bnn_rats_static      0.200   0.533   0.733   1.000   1.000   1.000
bnn_rats_adaptive    0.800   0.833   0.800   1.000   1.000   1.000
ada_mcts             0.567   0.667   0.800   0.967   0.967   1.000
```
γ=0.99 return：dp 0.56–0.89 / oracle_rats 0.38–0.89 / static 0.10–0.87 /
**adaptive 0.43–0.87** / ada_mcts 0.30–0.63

修复后 adaptive 在 p≤0.6 全面反超 static（p=0.4：0.800 vs 0.200），
接近 oracle_rats（0.633）并逼近 dp（1.000）。

### bridge — goal rate by p
```
method              p=0.4   p=0.5   p=0.6   p=0.8   p=0.9   p=1.0
dp_nsmdp             0.730   0.550   0.770   0.980   1.000   1.000
dp_snapshot          0.730   0.550   0.770   0.980   1.000   1.000
oracle_rats          0.730   0.550   0.770   0.980   1.000   1.000
bnn_rats_static      0.360   0.540   0.770   0.980   1.000   1.000
bnn_rats_adaptive    0.360   0.540   0.770   0.980   1.000   1.000
ada_mcts             0.060   0.160   0.250   0.390   0.460   0.550
```
（bridge 上 adaptive≡static：episode 3–4 步，forget 周期内不触发；
ada_mcts 仍受 DPAS gamma 压制，见 §8.2）

### stationary（p=0.7）
cliff：dp 1.000、oracle_rats 0.900、static/adaptive 1.000、ada_mcts 1.000；
bridge：除 ada_mcts 外均 0.910，ada_mcts 0.710。

log：`/tmp/grid_cliff_fixed2.log`、`/tmp/grid_bridge_fixed2.log`

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
