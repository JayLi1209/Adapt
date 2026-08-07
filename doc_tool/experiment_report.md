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

预训练模型在原环境上的表现（cliff / bridge，待全量重跑填充）：

```
[待全量实验填充]
```

### 5.3 结果：CliffWalking（p: 0.7 → p_new at ts 0）

goal rate / 折现 return 表：

```
[待全量实验填充]
```

### 5.4 结果：NS-Bridge（p: 0.7 → p_new at ts 0）

```
[待全量实验填充]
```

### 5.5 分析

[待结果填充后撰写]

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

[待结果填充后撰写]

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
| FIR 在线循环装配 | `run_gridworld_experiments.py` | `BNNRATS.act` (L~195-247) |
| 置信门控 CVaR-CEM | `planning/cvar_cem.py` | `CVaRCEMAgent` |
| 预训练 | `pretrain_gridworld.py` | `main`（目标加权采样 L57-85） |
