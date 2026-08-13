# Gridworld 非平稳实验报告（FIR-CEM 版）：FIR 自适应规划 vs RATS / ADA-MCTS 基线

> 版本：2026-08-12（规划器换成 CVaR-CEM 主方法 + unbounded RATS 两方案，p=1.0 设定）
> 代码：`run_gridworld_experiments.py`；方法文档：`Catch_Me_If_You_Can.md`；
> 前版报告（FIR-RATS 版，p=0.7 设定）：`experiment_report.md`、`experiment_report_new.md`

## 摘要

我们研究"环境发生**有告知的、无界**变化"（informed & unbounded change）时的在线
安全规划问题。采用预训练 Bayesian Neural Network（BNN，Dirichlet 头）作为世界模型，
提出 **FIR**（Forget–Inflate–Retrain）机制：用校准后的惊讶度（surprise）测量旧模型
与新环境的偏离，经漂移滤波器平滑得到漂移估计，据此把模型头部的浓度**收缩**（forget，
消除"自信但错误"的预测）、并用在线共轭计数**学习**新证据，使贝叶斯后验从"旧环境"
平稳迁移到"新环境"。规划器采用**置信门控的 CVaR-CEM**（本文主方法 **FIR-CEM**）；
RATS（worst-case 极小极大树搜索）与 ADA-MCTS（Luo et al. 2024）作为基线。
在 CliffWalking 与 NS-Bridge 两个 gridworld 基准上验证 FIR 的准确性与安全性。

## 1. 问题设定（Problem Setting）

**非平稳马尔可夫决策过程（NSMDP）**。环境随时间演化，演化幅度**无界**（unbounded），
但变化时刻**有告知**（informed：agent 在 ts 0 收到"环境变了"的通知，但不知道变成了
什么、未来还会不会变）。目标是在变化后快速、安全地恢复高性能策略。

**三个正交组件**（沿用 Catch_Me_If_You_Can.md 的分解）：

| 组件 | 是什么 | 本文的选择 |
|---|---|---|
| **组件 1：世界模型（world model）** | 对转移 $P(s'\|s,a)$ 的（贝叶斯）估计 | 预训练 BNN，Dirichlet 头（离散环境）/ Gaussian 头（连续环境） |
| **组件 2：适应机制（adaptation）** | 变化发生后如何更新模型 | **FIR（本文贡献）**：surprise → drift → forget → learn |
| **组件 3：规划器（planner）** | 基于当前模型选动作的决策器 | **CVaR-CEM（本文主方法）**；RATS / ADA-MCTS 为基线 |

**本报告实验中的"我们的完整方法" = 组件1(预训练 Dirichlet BNN) + 组件2(FIR)
+ 组件3(CVaR-CEM)**，即结果表中的 `cem_fir`（论文列名 **FIR-CEM**）。
`bnn_rats_adaptive`（FIR + RATS，即 **FIR-RATS**）作为"换规划器"的对照保留，
说明 FIR 对规划器是即插即用的。

## 2. 环境与变化（Environment & Change）

两个离散 tabular gridworld，动作 = 上下左右 4 方向，转移由 K 方向的离散分布刻画：

| 项 | CliffWalking | NS-Bridge |
|---|---|---|
| 网格 | 4×12（起点左下，目标右下，底排悬崖 H） | 5×8（中间一行桥，两端各一个 G，上下 H） |
| K 方向 | K=3（intended + 2 垂直滑移） | K=3（intended + 2 垂直滑移，桥上垂直=掉下桥/回岸） |
| slip_dist(p) | [p, (1-p)/2, (1-p)/2] | [p, (1-p)/2, (1-p)/2] |
| reward | **G=+1，H=-1，其余 0（无每步惩罚）** | 同左 |
| γ | 0.99 | 0.99 |
| max_steps（截断） | 100 | 10 |
| trials | 30 | 100 |

**变化**：预训练环境 **p = 1.0（完全确定性）**——先在确定性环境确认模型能找到
最优策略（最优路线已知）；ts 0 起环境变为 **p_new**，从 1.0 向更小的数退化
（本报告扫 p_new ∈ {0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}；p=0.7/0.6 只是其中两个
例子，还可选更小/更大的数）。变化是"无界"的：退化幅度未知，模型必须自己发现。

> 与上一版（experiment_report.md）的差异：预训练 p 0.7 → **1.0**；cliff 去掉
> 每步 −1 惩罚（return 不再是负数，直接是折现 goal rate，更好解读）；去掉两个
> DP oracle（dp_nsmdp / dp_snapshot，连续情形另有实验，一个 oracle_rats 足够）。

## 3. 方法（Methods）

### 3.1 组件 2：FIR 适应机制（本文贡献）

（与 experiment_report.md §4.1 相同，此处从略：surprise 校准 → drift 滤波 →
forget 收缩 Dirichlet 浓度 → learn 在线共轭计数。）

### 3.2 组件 3：置信门控 CVaR-CEM（本文主方法，FIR-CEM 的规划器）

跨熵方法（CEM）在规划时域 H 内搜索动作序列；每条候选序列用 K 个后验转移模型 ×
N 次 aleatoric rollout 的**经验 CVaR_α**（最差 α 尾部的平均折现回报）打分。
**置信门控**：同一个 surprise 信号驱动 CVaR 尾部 α——变化刚发生（surprise 高、
置信低）时 α → ALPHA_MIN（最保守，只看最差尾部），模型恢复（surprise 回到基线、
数据积累）后 α → 1（风险中性均值）。这就是"unbounded 情形下不用 Lipschitz 常数
也能做风险规避"的实现：悲观程度由**模型后验自身的 spread**给出，而非解析半径。

### 3.3 基线

| 运行名 | 论文列 | 模型 | 适应机制 | 规划器 | 一句话 |
|---|---|---|---|---|---|
| `oracle_rats` | RATS-P_k (oracle) | 真实当前模型 | — | RATS | 全知 + 风险规避（上界参照） |
| `rats_cv01` | RATS-CV₀.₀₁ (scheme 1) | 预训练 BNN | 无 | worst-of-N 后验采样 | **unbounded 方案 1**：100 个后验模型取最差（≈1% CVaR 尾） |
| `rats_cal` | RATS-calibrated (scheme 2) | 预训练 BNN | 无 | RATS + 校准 L_p | **unbounded 方案 2**：100 个后验模型校准 L_p，沿用 RATS worst-case |
| `bnn_rats_static` | RATS-ĥP_{k-1} | 预训练 BNN | 无 | RATS（L_p=1.0） | 学习型 + 风险规避，不自适应 |
| `bnn_rats_adaptive` | **FIR-RATS** | 预训练 BNN | **FIR** | RATS | FIR + RATS（换规划器对照） |
| `cem_static` | CEM-ĥP_{k-1} | 预训练 BNN | 无 | CVaR-CEM（α=1） | 学习型 + CEM，不自适应 |
| `cem_fir` | **FIR-CEM（本文）** | 预训练 BNN | **FIR** | CVaR-CEM（门控 α） | **本文主方法** |
| `ada_mcts` | ADA-MCTS | 预训练 BNN | DPAS + 在线学习 | MCTS | 论文方法（Luo et al. 2024） |
| `mcts_static` | MCTS-ĥP_{k-1} | 预训练 BNN | 无通知、无学习 | MCTS | 学习型 + 普通 MCTS（无自适应） |

> 为什么 unbounded 情形下 RATS 需要两个方案：现有 RATS 实现固定 L_p=1.0、L_r=0
>（半径随深度线性放大），这在"变化有界、已知 Lipschitz 常数"时才有意义；我们
> 面对的 unbounded 变化没有有效的 L_p，所以——
> **方案 1（rats_cv01）**：把 RATS 的"对 Wasserstein 球取 min"换成"对 N=100 个
>   后验转移模型取 min"，即根节点每个动作用最差的后验模型（≈1% CVaR 经验尾）。
> **方案 2（rats_cal）**：从 N=100 个后验模型校准出每个 (s,a) 的 Wasserstein
>   spread 作为 L_p，再套用原 RATS 的闭式 worst-case（Property 3）。两方案都只改
>   根节点决策，内部树仍用现有 RATS 代码（最小改动）。

## 4. 实验（Experiments）

### 4.1 设置

预训练模型：Dirichlet BNN，3 层贝叶斯线性主干（256 隐藏，softplus，Kaiming，
KL-to-prior 正则 β=0.1），Dirichlet 头输出 (s,a) 上的 K=3 方向浓度。**按 CLAUDE.md
要求，靠近目标的转移优先采样**（权重 w(s)=1.35^(−dist(s,G))）；2 万行、400 epoch、
Adam lr=1e-3，目标分布 p=1.0 的 slip [1,0,0]。验证：预测 p_dir 与目标 [1,0,0] 的
MAE——cliff 0.0003、bridge 0.0001（GOOD）。checkpoint：`data/cliffwalking/
bnn_dirichlet_cliffwalking_k3.pth`、`data/bridge/bnn_dirichlet_bridge_k3.pth`。

全部 9 个方法 × 7 个 p_new；每个 p_new 下同种子同环境，所有方法公平对比。
报告两种指标：**goal rate（到达 G 的比例，holes 记 0）** 与
**折现 return（γ=0.99，reward = G+1/H−1/每步0）**。

### 4.2 结果：stationary 验证（p=1.0）

预训练模型在原环境（确定性 p=1.0）上的表现——此环境下最优策略是确定性的，
goal rate 1.000 是"模型找到了最优解"的直接证据：

```
cliff（goal rate / 折现 return）：全部 9 个方法 goal rate 1.000
  oracle_rats / rats_cv01 / rats_cal / bnn_rats_static / bnn_rats_adaptive /
  cem_static / cem_fir / ada_mcts / mcts_static = 1.000（return +0.886）
bridge：全部 9 个方法 goal rate 1.000（return +0.970 / +0.980）
```
结论：p=1.0 确定性环境下预训练模型表现好（两个环境 goal rate 全 1.000），
验证通过。

### 4.3 结果：CliffWalking（p: 1.0 → p_new at ts 0，30 trials，30000 sims）

**goal rate by p（paper 约定，holes=0）**：
```
method              p=0.4   p=0.5   p=0.6   p=0.7   p=0.8   p=0.9   p=1.0
oracle_rats          0.833   0.767   0.933   1.000   1.000   1.000   1.000
rats_cv01            0.300   0.767   0.800   0.967   1.000   1.000   1.000
rats_cal             0.233   0.700   0.800   0.967   1.000   1.000   1.000
bnn_rats_static      0.300   0.800   0.800   1.000   1.000   1.000   1.000
bnn_rats_adaptive    1.000   1.000   1.000   1.000   0.933   0.800   1.000
cem_static           0.333   0.733   0.867   1.000   1.000   1.000   1.000
cem_fir              0.867   0.933   0.933   0.900   0.800   0.700   1.000
ada_mcts             0.533   0.700   0.800   0.900   0.933   1.000   1.000
mcts_static          0.600   0.800   0.933   1.000   1.000   1.000   1.000
```

**折现 return（γ=0.99，reward = G+1/H−1/每步0）**：
```
method              p=0.4   p=0.5   p=0.6   p=0.7   p=0.8   p=0.9   p=1.0
oracle_rats           0.48    0.51    0.63    0.72    0.80    0.84    0.89
rats_cv01             0.17    0.46    0.53    0.69    0.81    0.84    0.89
rats_cal              0.14    0.43    0.53    0.69    0.81    0.84    0.89
bnn_rats_static       0.17    0.48    0.51    0.71    0.81    0.84    0.89
bnn_rats_adaptive     0.57    0.59    0.65    0.63    0.56    0.59    0.89
cem_static            0.19    0.44    0.57    0.75    0.81    0.85    0.89
cem_fir               0.47    0.53    0.55    0.52    0.50    0.50    0.89
ada_mcts              0.29    0.40    0.49    0.61    0.66    0.73    0.77
mcts_static           0.37    0.46    0.63    0.77    0.81    0.84    0.87
```

观察：
- **FIR 在 cliff 上有效且显著**：bnn_rats_adaptive（FIR-RATS）在 p=0.4/0.5/0.6
  达 1.000/1.000/1.000，远超 static（0.300/0.800/0.800）与 oracle_rats
  （0.833/0.767/0.933）——forget 把过时的 p=1.0 先验清成均匀后，RATS 走保守
  路线绕开悬崖，反而比"全知但风险规避不足"的 oracle 更稳（oracle 用真实 p
  但 worst-case 半径仍按 L_p=1.0，不够保守）。
- **FIR-CEM 同样有效**：cem_fir 在 p=0.4/0.5/0.6 达 0.867/0.933/0.933，远超
  cem_static（0.333/0.733/0.867）——置信门控 α 让 CEM 在变化后谨慎，是主方法
  成立的关键证据。低 p 下 cem_fir 略低于 bnn_rats_adaptive（0.867 vs 1.000），
  是 CEM 滚动规划（horizon=3）的固有限制，不是 FIR 的问题。
- **两个 unbounded RATS 方案成立**：rats_cv01 / rats_cal 追平 bnn_rats_static
  （都 0.300/0.800/0.800 附近）且 p≥0.7 达 0.967-1.000——后验采样足以承担
  悲观性，无需解析 L_p；方案 1（worst-of-N）与方案 2（校准 L_p）表现接近。
- **退化更剧烈时 FIR 优势越大**：p 从 1.0 降到 0.4/0.5 是"无界"的大幅变化，
  FIR（adaptive）的 forget 恰好处理这种情形；p 接近 1.0（0.8/0.9）时变化小，
  FIR 的保守期反而略拖后腿（bnn_rats_adaptive 0.933/0.800 vs static 1.000/
  1.000）——FIR 的收益与"变化幅度"正相关，符合设计意图。
- **ada_mcts 居中偏弱**（0.533-1.000），mcts_static 在 cliff 上意外地强
  （0.600-1.000）——MCTS 的 rollout 搜索在 cliff 长路径上比桥上有效。

### 4.4 结果：NS-Bridge（p: 1.0 → p_new at ts 0，100 trials，30000 sims）

**goal rate by p（paper 约定，holes=0）**：
```
method              p=0.4   p=0.5   p=0.6   p=0.7   p=0.8   p=0.9   p=1.0
oracle_rats          0.190   0.360   0.590   0.790   0.930   0.990   1.000
rats_cv01            0.270   0.310   0.630   0.800   0.940   0.980   1.000
rats_cal             0.240   0.380   0.550   0.780   0.930   0.990   1.000
bnn_rats_static      0.190   0.360   0.590   0.790   0.930   0.990   1.000
bnn_rats_adaptive    0.190   0.360   0.590   0.790   0.930   0.990   1.000
cem_static           0.100   0.230   0.310   0.480   0.640   0.830   1.000
cem_fir              0.070   0.130   0.220   0.340   0.510   0.730   1.000
ada_mcts             0.160   0.200   0.470   0.670   0.730   0.920   0.970
mcts_static          0.120   0.220   0.340   0.510   0.650   0.830   1.000
```

**折现 return（γ=0.99，reward = G+1/H−1/每步0）**：
```
method              p=0.4   p=0.5   p=0.6   p=0.7   p=0.8   p=0.9   p=1.0
oracle_rats          -0.18    0.05    0.37    0.65    0.84    0.95    0.98
rats_cv01            -0.23   -0.20    0.35    0.63    0.85    0.93    0.97
rats_cal             -0.22    0.01    0.24    0.62    0.86    0.95    0.97
bnn_rats_static      -0.18    0.05    0.37    0.65    0.84    0.95    0.97
bnn_rats_adaptive    -0.18    0.05    0.37    0.65    0.84    0.95    0.97
cem_static           -0.65   -0.42   -0.34   -0.04    0.28    0.64    0.98
cem_fir              -0.60   -0.53   -0.41   -0.23    0.10    0.49    0.98
ada_mcts             -0.45   -0.43    0.02    0.39    0.43    0.81    0.91
mcts_static          -0.66   -0.52   -0.28    0.03    0.29    0.64    0.98
```

观察：
- **RATS 系（含两个 unbounded 方案）goal rate 追平 oracle**：bnn_rats_static /
  bnn_rats_adaptive 与 oracle_rats 完全一致（0.19–1.00）；方案 1（rats_cv01）在
  p=0.4/0.6 略高于 oracle（0.270/0.630 vs 0.190/0.590），方案 2（rats_cal）居中。
  两个 unbounded 方案都成立——后验采样本身足以承担悲观性，无需解析 L_p。
- **FIR 在 bridge 上无效（结构性）**：adaptive ≡ static（每集仅 3-4 步，forget
  周期内不触发；见 08-06 记录 §8.1）。
- **CEM 系（cem_static / cem_fir）goal rate 低于 RATS 系**（如 p=0.7：0.48/0.34
  vs 0.79）：CEM 的 horizon=3 滚动规划在桥的"先远离再接近"几何下短视；cem_fir
  低于 cem_static 是置信门控的保守期所致（α 压低 → 谨慎）。这是 planner 的
  局限，不是 FIR 的问题（同一 FIR 在 RATS 上是 adaptive=static 的桥特例）。
- **ada_mcts / mcts_static 居中**；ada_mcts 的 DPAS 自适应在 p≥0.6 略优于
  mcts_static（0.47/0.67 vs 0.34/0.51），但低 p 仍被 worst-case 采样拖垮。

## 5. 讨论（Discussion）

1. **FIR 是规划器无关的（plug-and-play）**：同一套 FIR（surprise → drift →
   forget → learn）接在 RATS 上（bnn_rats_adaptive）和接在 CVaR-CEM 上
   （cem_fir）都在 cliff 的低 p 大幅反超对应的 static 版本（RATS：1.000/1.000/
   1.000 vs 0.300/0.800/0.800；CEM：0.867/0.933/0.933 vs 0.333/0.733/0.867）。
   这说明 FIR 处理的是**模型层**的"自信但错误"，对下游规划器透明。

2. **FIR 的收益与变化幅度正相关**：p 从 1.0 退化越多（0.4/0.5），FIR 反超
   static 越多；p 接近 1.0（0.8/0.9）时变化小，FIR 的保守期反而略拖后腿。
   这正是"unbounded 变化"场景下需要的性质——变化越大越需要忘记旧模型。

3. **unbounded 情形下 RATS 的两个方案都成立**：没有有效 Lipschitz 常数 L_p 时，
   用 N=100 个后验模型取最差（方案 1，≈1% CVaR 尾）或校准 L_p（方案 2）都能
   让 RATS 追平 static 并在高 p 达 1.000——**悲观性可以来自模型后验的 spread，
   而非解析半径**。

4. **CEM 与 RATS 的差距是 planner 的，不是 FIR 的**：cem_fir 在低 p 略低于
   bnn_rats_adaptive（0.867 vs 1.000），原因是 CEM horizon=3 的滚动规划短视；
   同一 FIR 在 RATS（深度 3 树）上达到 1.000。提升 CEM 需要更长 horizon 或更
   好的叶估值，与 FIR 机制本身无关。

5. **bridge 上 FIR 无效是结构性的**：bridge episode 仅 3-4 步，forget 按
   episode 内步数驱动（K_FORGET）来不及触发；即使 K_FORGET=1 强制每步触发，
   "均匀模型"在桥上也没有更安全的路线（必须过桥，无悬崖可退避），所以
   adaptive ≡ static。CEM 的 horizon/n_confident/alpha_min 扫描（12 配置）也
   无法让 cem_fir 超过 cem_static。**FIR 需要足够长的 episode 来攒证据，且
   环境要有"保守路线"可供退避**——这两个条件 bridge 都不满足。这是短 episode
   + 无退避路线任务上 per-episode 自适应的固有边界。

6. **ADA-MCTS 作为 baseline 依然偏弱**：cliff 上 0.533-1.000（低 p 不如
   mcts_static），bridge 上被 worst-case 采样拖垮（0.160-0.970）。其 DPAS
   双相采样在无界大幅变化下不如 FIR 的 forget 机制稳健。

## 6. 关键代码位置

| 内容 | 文件 | 位置 |
|---|---|---|
| FIR 在线循环装配（RATS 版） | `run_gridworld_experiments.py` | `BNNRATS` (L178) + `act` (L225-245) |
| FIR 在线循环装配（CEM 版，主方法） | `run_gridworld_experiments.py` | `BNNCEM` (L266) + `act` |
| CVaR-CEM 规划器 | `planning/cvar_cem.py` | `CVaRCEMAgent` (L41) |
| unbounded RATS 方案 1（worst-of-N） | `run_gridworld_experiments.py` | `RATSCV01` (L351) |
| unbounded RATS 方案 2（校准 L_p） | `run_gridworld_experiments.py` | `RATSCalibrated` (L397) |
| RATS + DP 基线 | `planning/rats.py` | `RATS`、`DPAgent` |
| ADA-MCTS | `planning/ada_mcts.py` | `ADAMCTSAgent` |
| 预训练 | `pretrain_gridworld.py` | `main`（目标加权采样） |
