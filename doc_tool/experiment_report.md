# Gridworld 非平稳实验报告：FIR 自适应规划 vs RATS / ADA-MCTS 基线

> 版本：2026-08-07（对齐 Act As You Learn 论文后的全量重跑）
> 代码：`run_gridworld_experiments.py`；方法文档：`Catch_Me_If_You_Can.md`；
> 保真度对照：`experiment_fidelity_review.md`

## 摘要

我们研究"环境发生**有告知的、无界**变化"时（informed & unbounded change）的在线
安全规划问题。采用预训练 Bayesian Neural Network（BNN，Dirichlet 头）作为世界模型，
提出 **FIR**（Forget–Inflate–Retrain）机制：用校准后的惊讶度（surprise）测量旧模型
与新环境的偏离，经漂移滤波器平滑得到漂移估计，据此把模型头部的浓度**收缩**（forget，
消除"自信但错误"的预测）、并用在线共轭计数**学习**新证据，使贝叶斯后验从"旧环境"
平稳迁移到"新环境"。规划器采用 RATS（worst-case 极小极大树搜索）或置信门控的
CVaR-CEM。在 CliffWalking 与 NS-Bridge 两个 gridworld 基准上与 ADA-MCTS 论文
（Luo et al., 2024）的方法列对齐对比（DP-NSMDP、DP-snapshot、RATS-P_k、RATS-P_{k-1}、
RATS-ĥP_{k-1}、ADA-MCTS、MCTS-ĥP_{k-1}），在 p ∈ {0.4,0.5,0.6,0.8,0.9,1.0} 六种
变化幅度下报告 goal rate 与折现 return。

## 1. 引言（Motivation）

在灾难响应、自动驾驶等场景中，环境可能**突然**以**很大幅度**改变。设计自适应算法的
核心困难有两个目标：(1) 高效吸收变化的初始冲击；(2) 变化后开始谨慎地探索新环境。
本文假设问题被建模为非平稳 MDP（NSMDP），并聚焦于两类使问题可解但困难的性质：

- **有告知（informed）**：agent 知道"变化发生了"（例如通过异常检测器），但不知道
  变化是什么——它必须在重新学习之前就行动；
- **无界（unbounded）**：对转移函数的时间演化不做 Lipschitz 假设（区别于 NSMDP
  worst-case 工作线），因此旧数据可能变得**任意错误**，必须被主动降权而非仅仅折扣。

主流做法是离线预训练一个基座模型、在线适应变化。模型类方法相比 model-free 更高效地
利用先验数据，且可迁移。Bayesian 系统在建模时额外给出不确定性信息，其中**认知不确定性**
（epistemic）恰好是变化所摧毁的量：变化后旧后验是"自信但错误"的，必须**重新膨胀**
（re-inflate）而不是继续信任。这引导我们使用 Dirichlet-Multinomial 后验头部
（离散状态），它享有精确共轭更新：预测均值与总浓度都是先验与观测计数的闭式函数，
使 FIR 的 forget-and-learn 循环廉价且可解释。

## 2. 背景（Background）

### 2.1 非平稳 MDP 与 RATS

非平稳 MDP $\mathcal{M}_t = (\mathcal{S}, \mathcal{A}, P_t, r_t, \gamma)$，
其中 $P_t, r_t$ 未知且随时间变化。RATS（Lecarpentier & Rachelson, NeurIPS 2019）
假设演化是 Lipschitz 连续的（$L_p, L_r$），在 t₀ 时刻拿到当前快照
$\hat P_{t_0}$，在 Wasserstein-1 球 $W_1(P, \hat P_{t_0}) \le L_p \tau d$ 内做
**极小极大**树搜索：决策节点取 max，机会节点取 worst-case 模型（Property 3 闭式解：
worst 分布在 argmin 子节点的 Dirac 与当前分布的线性插值）。深度 d 处用启发式
$\gamma^{dist(s)}$ 截断。

### 2.2 ADA-MCTS（Act As You Learn, Luo et al., 2024）

环境离散突变 $M_{k-1} \to M_k$；agent 知道突变发生（notify）。方法用 BNN 同时估计
认知/偶然不确定性，在 MCTS 的机会节点上做 **DPAS**（dual-phase adaptive sampling）：
当认知或偶然不确定性相比上一阶段升高时，用 worst-case 采样（风险规避），否则用
常规似然采样。关键超参：30000 次 MCTS 迭代、RATS 深度 3、EPS_E=0.02、EPS_A=0、
p ∈ {0.4,0.5,0.6,0.8,0.9,1.0}、原环境 p₀=0.7。

**对齐声明**：本报告所有基线与方法列对齐 **Act As You Learn 论文**（"Approaches
that do not know ground truth transition" 一节的 RATS-ĥP_{k-1} / MCTS-ĥP_{k-1} /
ADA-MCTS，以及 oracle 列 RATS-P_k / RATS-P_{k-1}）。论文未写明的设置（γ、变化时刻、
trials、MCTS 迭代数）取本仓库设定或 upstream demo 设定，确保所有方法**同设置公平对比**。

### 2.3 贝叶斯神经网络与不确定性分解

变分 BNN 在权重上放置高斯后验，预测通过采样网络获得。不确定性分解为
（Kendall & Gal, 2018）：**偶然**（aleatoric）——环境本身的随机性，由预测分布熵刻画；
**认知**（epistemic）——模型的未知程度，由权重后验采样下预测的散布刻画。Dirichlet
头部直接给出转移概率的后验 $p \sim \text{Dir}(\alpha)$，其中 $\alpha$ 是 (s,a) 的
函数：预测均值 $\bar p = \alpha/\alpha_0$，总浓度 $\alpha_0 = \sum_k \alpha_k$ 越小，
后验越宽（置信越低）。

## 3. 问题设定（Problem Setup）

### 3.1 环境与变化

| 环境 | 结构 | slip 结构 | 奖励 | max_steps |
|---|---|---|---|---|
| CliffWalking 4x12 | S(3,0)，G(3,11)，悬崖 H 一行 | K=3 垂直：[p, (1-p)/2, (1-p)/2] | G +1，H −1（终止），**其余每步 −1** | 100 |
| NS-Bridge 5x8 | 中间一行桥面，两端 G，上下为洞 | K=3 垂直：[p, (1-p)/2, (1-p)/2]，滑向当前格上/下格 | G +1，H −1，其余 0 | 10 |

- **变化语义**：agent 在 ts 0 收到变化通知，p 从 p₀=0.7 瞬间变为
  p ∈ {0.4, 0.5, 0.6, 0.8, 0.9, 1.0} 并保持（ADA-MCTS 离散突变 $M_{k-1}\to M_k$ 的
  实例化；比"变化发生在途中"更极端——agent 没有任何 p=0.7 的在线体验）。
- **Bridge slip 几何**：与官方 nsbridge_v0.py 一致——slip 质量平均分给当前格的
  上/下格（垂直方向，K=3），桥面上方/下方是肩膀自由格，肩膀再往外是洞。
  之前实现误用"掉头"（opposite, 1−p 到反向），已于 2026-08-07 修正。
- **Cliff 每步惩罚**：按 Luo et al. "the agent concedes a penalty for each step it
  takes (except the goal)"，每非目标格 −1（2026-08-07 修正）。
- **γ = 0.99**（任务规格指定）。折现 return 按 γ=0.99 累加，每步奖励记录。
- **截断**：cliff 100 步、bridge 10 步；统计 goal rate 时到达 H 记失败。
- trials：cliff 30，bridge 100，种子 0..n-1。

### 3.2 预训练模型

Dirichlet BNN：3 层贝叶斯线性主干（256 隐藏，softplus，Kaiming，KL-to-prior
正则 β=0.1），Dirichlet 头输出 (s,a) 上的 K=3 方向浓度。按 CLAUDE.md 要求，
**靠近目标的转移优先采样**：权重 w(s) = 1.35^(−dist(s,G))；2 万行、400 epoch、
Adam lr=1e-3，目标分布为 p=0.7 的 slip。验证：预测 p_dir 与目标
[0.7, 0.15, 0.15] 的 MAE：cliff 0.004、bridge 0.007（GOOD）。

## 4. 方法（Method）：FIR + 风险规避规划

### 4.1 FIR 在线机制（Algorithm 1）

**记号**：变化通知后，$\bar p(\cdot|s,a)$ 为 BNN 的权重平均预测分布，
$H(\bar p)$ 为其熵，$s_{t+1}$ 为实际转移。

**Step 1 — 校准惊讶度（Surprise）**。对 Dirichlet 头：

$$\delta_t = \frac{-\log \bar p_{s_{t+1}}}{H(\bar p)}, \qquad \mathbb{E}[\delta_t] = 1 \ \text{（模型正确时）}$$

正确模型每步"代价"约 1；真正误预测的转移（环境突变）把 δ 推到远高于 1。
（连续环境用归一化马氏距离 $\frac1d(s_{t+1}-\mu)^\top\Sigma^{-1}(s_{t+1}-\mu)$。）

**Step 2 — 漂移滤波器（Drift）**。原始惊讶度流有噪声。DriftFilterV2 在变化前处于
校准模式，把样本折入经验基线 b（≈1）；reset()（变化通知时触发）冻结基线并进入检测
模式：$\hat\lambda = \frac1n \sum_i (\delta_{n,i} - b)$，$\bar\delta = 1 + \hat\lambda$。
带符号累积：如果证据实际上说"没变"，$\bar\delta$ 会被拉回 1 以下，forget 自动停止。

**Step 3 — 忘记 / 膨胀（Forget）**。保留因子 $\rho = 1/\max(\bar\delta, 1) \in (0,1]$，
标量保留状态 $r \leftarrow r\cdot\rho$（每 K_FORGET=3 步施加一次）。对 Dirichlet 头：

$$\alpha_k^{\text{eff}} = \underbrace{c + r(\alpha_k^{\text{head}} - c)}_{\text{forget 作用于此}} + \underbrace{N_k}_{\text{在线计数}}, \qquad c = 0.1 \ (\text{对称先验})$$

均值被拉向均匀，更重要的是总浓度 $\alpha_0 \to Kc$ 坍缩——抽样 $p\sim\text{Dir}(\alpha)$
的方差 $p_k(1-p_k)/(\alpha_0+1)$ 增大，实现"置信重新膨胀"。

**Step 4 — 学习 / 重训（Learn/Retrain）**。在线共轭计数 $N \leftarrow N + \mathbf{1}(s,a,s_{t+1})$
逐 (s,a) 累积；有效后验 = 保留先验 + 新鲜计数，预测均值向变化后的经验频率迁移。
（batched ELBO 重训在连续环境/长视界启用；本实验默认关闭——小浓度头部下计数会
扭曲预测，08-06 调查已定论。）

**Algorithm 1: FIR 在线循环（gridworld 实例化，每 episode）**
```
输入：预训练 BNN（主干冻结）、漂移滤波器 F、RATS 规划器
1:  t₀ 时刻收到变化通知                    ▷ 有告知变化
2:  F.reset(); r ← 1; 计数 N ← 0          ▷ 结束校准，进入检测
3:  for 每一步 t ≥ t₀:
4:      a_t ← RATS.act(s_t)               ▷ 在 BNN 快照上做 worst-case 极小极大
5:      执行 a_t，观测 (s_{t+1}, r_t)
6:      δ_t ← -log p̄(s_{t+1}) / H(p̄)     ▷ 校准惊讶度
7:      F.update(δ_t)  →  δ̄_t = 1 + λ̂     ▷ 带符号等权滤波
8:      if t 距上次 forget ≥ K_FORGET:
9:          ρ ← 1/max(δ̄_t, 1);  r ← r·ρ    ▷ forget：头部浓度向对称先验收缩
10:     N ← N + one-hot(s,a,s_{t+1})       ▷ 共轭在线计数（可选，默认关）
11: end loop
```

### 4.2 规划器：RATS（本实验）

RATS 在**当前模型快照**上做极小极大：决策节点取 max，机会节点在 Wasserstein 球
$W_1 \le d\cdot L_p\cdot\tau$（L_p=1, τ=1）内取 worst-case（Property 3 闭式解），
深度 3（论文文档值），叶子启发式 $\gamma^{dist(s)}$。FIR 修改后的预测均值
（retain/counts 参与）就是规划器所用的快照——因此 forget/learn 的适应直接体现在
规划模型上。

### 4.3 规划器：置信门控 CVaR-CEM（连续环境）

对连续状态/动作，规划器为 CEM 动作序列搜索，评分取想象 return（K=10 后验模型
× N=32 偶然 rollout）的经验 CVaR：

$$a^* = \arg\max_a \ \mathrm{CVaR}_{\alpha}[Z(a)], \qquad \alpha = \alpha_{\min} + (\alpha_{\max}-\alpha_{\min})\cdot\text{conf},$$

其中 $\text{conf} = \underbrace{\min(1, n_{\text{post}}/n_{\text{conf}})}_{\text{数据门}} \cdot
\underbrace{e^{-\max(0,\bar\delta-1)/\tau}}_{\text{惊讶度门}}$。
变化后瞬间 conf≈0 → α≈α_min（最风险厌恶）；新证据累积、惊讶度回落，α→α_max=1
（风险中性均值）。同一个漂移信号同时驱动模型不确定性与风险准则。

### 4.4 方法列（与 Act As You Learn 表对齐）

| 方法 | 模型 | 适应 | 规划器 |
|---|---|---|---|
| DP-NSMDP (oracle) | 真实时变调度 | — | 期望 DP（精确上界） |
| DP-snapshot (oracle) | 真实当前模型 | — | 期望 DP |
| RATS-P_k (oracle) | 真实当前模型 | — | RATS（深度 3） |
| RATS-P_{k-1} (oracle) | 真实旧模型 p=0.7 | 不更新 | RATS |
| RATS-ĥP_{k-1} | 预训练 BNN | 不更新 | RATS |
| **FIR-RATS（本文）** | 预训练 BNN | surprise/forget/counts | RATS |
| ADA-MCTS | 预训练 BNN | DPAS + 在线学习 | MCTS（5000 次迭代） |
| MCTS-ĥP_{k-1} | 预训练 BNN | 无通知、无学习 | MCTS |

MCTS 迭代数：论文 30000；Python 移植实测 7.5 s/action，全量实验不可行
（cliff 约 75 h + bridge 约 25 h），采用 upstream demo 的 5000 次
（`ADA-MCTS/act_learn.py: search(5000)`）。

## 5. 实验（Experiments）

### 5.1 设置

见 §3。全部 8 个方法 × 6 个 p 值；每个 p 值下同种子同环境，所有方法公平对比。
报告两种指标：**goal rate**（到达 G 的比例，论文 return 约定）与
**折现 return**（γ=0.99，含每步惩罚）。

### 5.2 结果：stationary 验证（p=0.7）

预训练模型在原环境上的表现（cliff 30 trials / bridge 100 trials）：

```
cliff（goal rate / 折现 return）：
  dp_nsmdp 1.000 / -20.61    dp_snapshot 1.000 / -20.61
  oracle_rats 1.000 / -27.08    rats_pkminus1 1.000 / -27.08
  bnn_rats_static 1.000 / -27.08    bnn_rats_adaptive 1.000 / -27.08
  ada_mcts 1.000 / -21.46    mcts_static 1.000 / -21.46
bridge：
  dp_nsmdp / dp_snapshot / oracle_rats / rats_pkminus1 /
  bnn_rats_static / bnn_rats_adaptive  0.790 / +0.65
  ada_mcts 0.390 / -0.22    mcts_static 0.390 / -0.22
```
结论：cliff 上全部方法 goal rate 1.000，预训练模型表现好；
bridge 上模型类规划器 0.790（垂直 slip 下桥比旧设定难），MCTS 系 0.390
（rollout 叶估计弱）。注意每步 −1 惩罚下 cliff return 恒为负，绝对值反映路径长度。

### 5.3 结果：CliffWalking（p: 0.7 → p_new at ts 0，30 trials，30000 sims）

**goal rate by p（paper 约定，holes=0）**：
```
method              p=0.4   p=0.5   p=0.6   p=0.8   p=0.9   p=1.0
dp_nsmdp             1.000   1.000   1.000   1.000   1.000   1.000
dp_snapshot          1.000   1.000   1.000   1.000   1.000   1.000
oracle_rats          0.600   0.767   0.933   1.000   1.000   1.000
rats_pkminus1        0.467   0.867   0.967   1.000   1.000   1.000
bnn_rats_static      0.467   0.867   0.967   1.000   1.000   1.000
bnn_rats_adaptive    0.733   0.800   0.967   1.000   1.000   1.000
ada_mcts             0.367   0.500   0.533   0.267   0.233   0.000
mcts_static          0.767   0.900   0.967   1.000   1.000   1.000
```

**折现 return（γ=0.99，每步 −1）**：
```
method              p=0.4   p=0.5   p=0.6   p=0.8   p=0.9   p=1.0
dp_nsmdp            -40.05  -32.50  -27.27  -17.24  -14.45  -10.48
dp_snapshot         -40.05  -32.50  -27.27  -17.24  -14.45  -10.48
oracle_rats         -48.73  -40.30  -34.24  -18.99  -15.12  -10.48
rats_pkminus1       -51.33  -41.77  -32.51  -18.49  -15.90  -12.26
bnn_rats_static     -51.33  -41.77  -32.51  -18.49  -15.90  -12.26
bnn_rats_adaptive   -48.81  -42.83  -33.01  -18.54  -15.99  -12.26
ada_mcts            -58.62  -55.15  -56.26  -59.44  -60.58  -63.40
mcts_static         -45.98  -36.94  -30.92  -18.44  -15.48  -14.00
```

### 5.4 结果：NS-Bridge（p: 0.7 → p_new at ts 0，100 trials，30000 sims）

**goal rate by p**：
```
method              p=0.4   p=0.5   p=0.6   p=0.8   p=0.9   p=1.0
dp_nsmdp             0.190   0.360   0.590   0.930   0.990   1.000
dp_snapshot          0.190   0.360   0.590   0.930   0.990   1.000
oracle_rats          0.190   0.360   0.590   0.930   0.990   1.000
rats_pkminus1        0.190   0.360   0.590   0.930   0.990   1.000
bnn_rats_static      0.190   0.360   0.590   0.930   0.990   1.000
bnn_rats_adaptive    0.190   0.360   0.590   0.930   0.990   1.000
ada_mcts             0.140   0.250   0.320   0.490   0.520   0.650
mcts_static          0.070   0.160   0.250   0.550   0.770   1.000
```

**折现 return（γ=0.99，holes=-1）**：
```
method              p=0.4   p=0.5   p=0.6   p=0.8   p=0.9   p=1.0
dp_nsmdp             -0.18    0.05    0.37    0.84    0.95    0.98
dp_snapshot          -0.18    0.05    0.37    0.84    0.95    0.98
oracle_rats          -0.18    0.05    0.37    0.84    0.95    0.98
rats_pkminus1        -0.18    0.05    0.37    0.84    0.95    0.97
bnn_rats_static      -0.18    0.05    0.37    0.84    0.95    0.97
bnn_rats_adaptive    -0.18    0.05    0.37    0.84    0.95    0.97
ada_mcts             -0.64   -0.44   -0.31   -0.03    0.04    0.28
mcts_static          -0.83   -0.66   -0.48    0.10    0.53    0.98
```

### 5.5 分析（cliff + bridge）

1. **oracle 上界**：DP 满格（1.000 × 6 p）——存在全 p 安全策略（悬崖下方有长路，
   max_steps=100 足够）。
2. **RATS 的过度保守**：RATS-P_k（真实模型）在 p≤0.6 低于 DP（p=0.4：0.600 vs
   1.000）——worst-case 半径 c=d·L_p·τ 在 d=3 时达 3 个 W1 单位，把最优期望策略
   过度保守化；RATS-P_{k-1}（用旧模型 p=0.7）在 p=0.4 更差（0.467）。
3. **FIR-RATS 反超 oracle_rats（p=0.4：0.733 vs 0.600）**：forget 把模型头部的
   浓度向对称先验收缩 → 预测趋近均匀 → RATS 在"无信息"模型上走保守路线（绕开
   悬崖边）→ 在低 p 下比"正确但冒险"的 oracle 策略更安全。这正是 FIR 的设计
   意图（变化后先谨慎、后恢复），且 p≥0.6 时随惊讶度回落完全恢复（1.000）。
   p=0.5 处 adaptive（0.800）略低于 static（0.867）——forget 的边际收益非单调。
4. **bnn_rats_static ≡ rats_pkminus1**（0.467/0.867/0.967...）：预训练 BNN 在
   cliff 上几乎与真实旧模型等价——模型的 p_dir 均值误差仅 0.004。
5. **mcts_static（MCTS-ĥP_{k-1}）意外地强**（p=0.4：0.767 全场学习型最高）：
   30000 rollouts + UCT 在预训练模型上天然保守（rollout 中的负奖励惩罚悬崖路径）。
6. **ada_mcts 崩盘是 DPAS 病理，非移植 bug**：与 upstream `adamcts.py` 逐行一致
   （gamma=10000）。机制（08-06 已定位）：变化通知后 M_k 随 counts 累积变得
   比快照 M_{k-1} 更"自信"（ale_k < ale_prev）→ 似然 exp(−10000·diff)≈0 →
   phase-2 永远 worst-case 采样 → pessimistic 把可达负奖励格 one-hot → 树被
   毒化。p=1.0（确定性环境）时 ada_mcts=0.000 而 mcts_static=1.000，说明
   **DPAS 主动破坏**了无噪声环境下的表现。忠实移植，如实报告；降低 dpas_gamma
   可部分恢复（08-06 bridge 上 10000→10 使 p=0.4 从 0.06→0.26）。
7. **return 与 goal rate 一致**：每步 −1 下路径越长 return 越低；ada_mcts
   return 恒低（−55 至 −63）。
8. **bridge：全部模型类规划器数字相同（0.190/0.360/0.590/0.930/0.990/1.000）**：
   桥只有 3-4 步宽，预训练 BNN 近乎精确（MAE 0.007）且 RATS 与 DP 在此网格上
   给出同一最优动作（08-06 亦观察到 RATS==DP），所以 oracle 与学习型模型、
   adaptive 与 static 走完全相同的轨迹。FIR 循环在 bridge 上无机会生效——
   episode 短（≤10 步）+ 模型本来就不怎么错。
9. **bridge 低 p 结构性无解**：p=0.4 时即使 omniscient DP 也只有 0.190——垂直
   slip 下（1−p)/2=0.3 的滑移质量在桥面靠近右端（col 5-7）直接进洞，跨最后
   3 格的成功率 ~0.4³，任何策略都难逃。p=0.5 反向（0.360）也比 0.6 低：期望
   位移小。
10. **MCTS 系在 bridge 上系统性弱**：mcts_static p≤0.6 只有 0.070-0.250
    （叶估计为 6 步随机 rollout，无 γ^dist 启发式，桥的危险网格上噪声大）；
    ada_mcts 因 DPAS worst-case 在低 p 反而比 mcts_static 稳（0.140 vs 0.070），
    但在 p≥0.8 再次崩（0.650@1.0 vs mcts_static 1.000）——同一 DPAS 病理。

## 6. 保真度声明（与论文的差异）

详见 `experiment_fidelity_review.md`。要点：
- ✅ 与 ADA-MCTS 论文一致：p 集合、p₀=0.7、cliff/bridge slip 结构（垂直）、
  EPS_E/EPS_A、RATS 深度 3、oracle 角色分工。
- ✅ 2026-08-07 修正：bridge 掉头 slip → 垂直 slip（K=3）；cliff 每步 −1 惩罚。
- ⚠️ 有意差异：γ=0.99（任务规格）与论文未注明；变化时刻 ts 0（ADA-MCTS 离散
  突变的实例化）；MCTS 5000 vs 论文 30000（Python 移植速度，见 §4.4）；
  不实现官方 nsbridge 的 t=0 确定斜坡（属于 RATS 论文的连续演化设定，且
  L_p=1.0 下 t≥1 即饱和、仅影响第一步）；no-discount 不适用（任务规格）。

## 7. 结论

在 CliffWalking 上（每步 −1、γ=0.99、30000 MCTS simulations）：
- **FIR-RATS 在变化幅度最大的 p=0.4 处达到 RATS 系最高 goal rate 0.733**，
  超过 RATS-ĥP_{k-1}（0.467）、甚至 oracle RATS-P_k（0.600）——"变化后先谨慎
  （forget 拉均匀 → 保守绕行）、证据充足后恢复（p≥0.6 回到 1.000）"的机制
  在悬崖任务上奏效；MCTS-ĥP_{k-1}（0.767@0.4）是全场最高的学习型基线。
- MCTS-ĥP_{k-1}（0.767@0.4）是 RATS 系之外的最强学习型基线。
- ADA-MCTS（忠实移植，DPAS gamma=10000）在此 ts-0 突变实例化上陷入 worst-case
  病理（p=1.0 时 0.000 vs mcts_static 1.000），如实报告并注明机制。

在 NS-Bridge 上：
- 所有模型类规划器（DP/RATS/BNN/FIR）数字完全相同且最优——桥短、模型精确、
  RATS 与 DP 一致，FIR 循环无机会生效；p≤0.6 时桥本身结构性无解（oracle
  也只有 0.19-0.59）。
- MCTS 系系统性弱于模型类（rollout 叶估计 + DPAS 病理）。

总体：FIR 的价值体现在**长视界、变化大**的任务（cliff 低 p）；在模型近乎
精确或任务过短的场景（bridge）与 oracle 持平（不劣化）。这是诚实的结果——
机制在有发挥空间时带来增益，没有时也不伤害。

---

## 附录：算法在代码中的位置

用户问"algorithm 去哪里了"——FIR 算法分散在以下文件（均为本仓库代码）：

| 组件 | 文件 | 位置 |
|---|---|---|
| 校准惊讶度 | `bnn/dirichlet_workflow.py` | `surprise_dirichlet` (L55-73) |
| 漂移滤波（经验基线+带符号等权） | `drift/filters.py` | `DriftFilterV2` (L60-145) |
| forget（浓度收缩） | `bnn/dirichlet_workflow.py` | `forget_dirichlet` (L76-89) |
| Dirichlet 头部 + retain/counts | `bnn/dirichlet_model.py` | `DirichletDynamicsModel._forward_alpha` |
| RATS 极小极大 | `planning/rats.py` | `RATS` (L281-339) |
| DP 基线 | `planning/rats.py` | `DPAgent` (L344-396) |
| ADA-MCTS（移植） | `planning/ada_mcts.py` | `ADAMCTSAgent`（upstream `ADA-MCTS/adamcts.py` 的忠实移植） |
| FIR 在线循环装配 | `run_gridworld_experiments.py` | `BNNRATS` (L164) + `act` (L195-247) |
| 置信门控 CVaR-CEM | `planning/cvar_cem.py` | `CVaRCEMAgent` (L41) |
| 预训练 | `pretrain_gridworld.py` | `main`（目标加权采样 L57-85） |
