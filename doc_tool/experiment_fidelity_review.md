# 实验保真度审查 — gridworld RATS vs ADA-MCTS 对照论文文档

> 2026-08-07 依据 `doc_tool/` 下四份文档（NSMDP.md、Act as you Learn.md、
> Catch_Me_If_You_Can.md、Math_behind_the_BNN_approach）对本仓库 gridworld
> 实验（`run_gridworld_experiments.py`）的设定逐项核对。
> 结论速览：**大体忠实，但 bridge 环境、cliff 奖励结构、MCTS 预算三处偏差使
> 数字不能与论文表格直接数值比较**。
>
> **2026-08-07 修正状态（对齐 Act As You Learn 论文，用户指示："算法对齐的
> 不应该是 RATS 论文，而是 act as you learn 论文，找不到的设置，就用这个
> repo 里的设置。确保公平比较，实验用的设置一致即可"）**：
>
> - ✅ §2.1 bridge slip 方向：**已修正**。BRIDGE_5x8 改为 K=3 垂直 slip
>   [p,(1-p)/2,(1-p)/2]（官方 nsbridge_v0.py 几何：slip 质量去当前格的上/下格），
>   与 ADA-MCTS 论文一致；不再采用"掉头"。
> - ✅ §2.5 cliff 每步惩罚：**已修正**。GridSpec.step_penalty=-1.0，非目标格
>   每步 −1（"except the goal"），G +1 / H −1 保留。
> - ✅ §2.7 RATS 深度：**已修正**。RATS 默认深度 3（论文文档值）；DP 深度 100
>   （精确）。新增 --rats-depth / --dp-depth。
> - ✅ §2.6 MCTS 迭代数：**部分对齐**。30000 在 Python 移植上实测 7.5s/action
>   （cliff 全量 ≈75h），不可行；采用 upstream demo 的 5000（ADA-MCTS/act_learn.py
>   `search(5000)`），并在报告中声明。
> - ✅ 方法列：**已对齐**。新增 rats_pkminus1（RATS-P_{k-1}，固定 p=0.7 oracle）
>   与 mcts_static（MCTS-ĥP_{k-1}，无通知无在线学习）。
> - ℹ️ §2.3 变化语义：ts-0 突变保留（ADA-MCTS 离散突变 $M_{k-1}\to M_k$ 的
>   实例化，非 RATS 论文连续演化——对齐目标已明确为 ADA-MCTS）。
> - ℹ️ t=0 确定斜坡（官方 nsbridge 的 T[s,a,0]=确定）：**有意不实现**——属于
>   RATS 论文连续演化设定；且 L_p=1.0 下斜坡在 t≥1 即饱和（λ=1/(1−p)），只
>   影响每 episode 第一步。已在实验报告中声明。
> - ✅ 对齐决策：γ=0.99 保留（任务规格）；oracle 角色分工、p 集合、EPS、
>   slip 结构不变。

## 1. 逐项对照表

| 设定 | RATS 论文 (NSMDP.md) | ADA-MCTS 论文 (Act as you Learn) | 我们的实现 | 判定 |
|---|---|---|---|---|
| p 集合 | —（连续变化，ε∈[0,1]） | {0.4,0.5,0.6,0.8,0.9,1.0} | 同左 | ✅ |
| 原环境 p₀ | 0（确定性） | 0.7 | 0.7 | ✅ (ADA-MCTS) |
| cliff slip | — | [p,(1-p)/2,(1-p)/2] 垂直 | K=3 垂直滑移 | ✅ |
| bridge slip | **垂直**（Left/Right slip 到 Up/Down 位置=洞） | **掉头**（1-p 到 opposite） | 掉头（opposite） | ⚠️ 见 §2.1 |
| bridge 几何 | 洞紧邻桥面两侧 | 同左+"extra hole" | row1/3 左 5 格为自由格 | ⚠️ 见 §2.2 |
| 变化语义 | 连续（t=0 确定→slip 增到 0.45） | 离散 M_{k-1}→M_k | ts 0 瞬间 0.7→p_new | ⚠️ 见 §2.3 |
| γ | 0.9 | 未注明（表头"discounted"） | 0.99（用户指定） | ⚠️ 见 §2.4 |
| 每步惩罚 | — | cliff 有 per-step penalty | 无（仅 G+1/H-1） | ❌ 见 §2.5 |
| MCTS 迭代 | — | 30000 | 1000 | ❌ 见 §2.6 |
| RATS 深度 | 6（bridge 叶子=终止态） | 3（计算限制） | 6（cliff 叶子非终止） | ⚠️ 见 §2.7 |
| EPS_E / EPS_A | — | 0.02 / 0 | 0.02 / 0 | ✅ |
| FrozenLake reward | G+1, H−1 | G 正、H cost | G+1, H−1 | ✅ |
| 变化时刻 | 无（连续） | 未注明 | ts 0（首步即新环境） | ⚠️ 合理实例化 |
| oracle 基线 | DP-snapshot / DP-NSMDP | RATS-P_k / MCTS-P_k | dp_snapshot/dp_nsmdp/oracle_rats | ✅ |
| 学习型模型 | —（快照=真值） | BNN + latent param 转移 | 预训练 Dirichlet BNN + 在线 counts/forget | ℹ️ 方法差异，非错误 |

## 2. 偏差详情与后果

### 2.1 bridge slip 方向（与 RATS 原论文矛盾）
RATS 论文正文（NSMDP.md §6）明确：bridge 上 Left/Right 动作的 slip 是
**"reach the positions usually stemming from Up and Down"**——即垂直滑移，
从桥面掉进上下紧邻的洞；且 t=0 时完全确定，随时间 slip 概率**连续**增至
0.45，由 ε∈[0,1] 控制左右哪侧更滑。
ADA-MCTS 论文 §4 对同一 bridge 的描述却是**掉头**（"goes in the opposite
direction with probability 1−p"），两者互相矛盾。
我们采用掉头（opposite），依据是 ADA-MCTS 论文的文字描述。
**后果**：我们的 bridge 结果只能与 ADA-MCTS 论文对比，不能与 RATS 论文
Figure 2 的 return-vs-ε 曲线对比；RATS 在"掉头 slip"bridge 上的 worst-case
行为与原论文（垂直 slip 掉洞）完全不同。

### 2.2 bridge 几何
我们的 BRIDGE_5x8：
```
HHHHHHHH
FFFFFHHH     ← row1 左侧 5 格是自由格
GFFFSFFG
FFFFFHHH     ← row3 左侧 5 格是自由格
HHHHHHHH
```
agent 可以从桥面走到 row1/3 的自由格（"离桥"不致命）。原论文 bridge 中
Up/Down 直接是洞（灰格），无安全肩。ADA-MCTS 论文还"add an extra hole"。
**后果**：我们的 bridge 比原论文"安全"（存在离桥绕行路线），改变了 RATS
风险规避行为的几何约束；且 §3 中 dp 等方法在 p=0.4 可达 0.73 的"反向意图
利用掉头"策略在两种几何下都成立，但具体数值不可比。

### 2.3 变化语义
- RATS 论文：环境**连续**演化（LC-NSMDP 假设），规划器拿当前快照但不知未来。
- ADA-MCTS 论文：M_{k-1}→M_k 离散突变，变化时刻未注明。
- 我们：ts 0 瞬间从 p₀=0.7 跳到 p_new 并保持（agent 首步即在 p_new 下）。
**后果**：ts-0 突变是 ADA-MCTS 离散设定的一种合理实例化（且比"变化在途
中"更极端——agent 没有任何 p=0.7 的在线体验），但**不是** RATS 论文的
"连续演化+快照"设定。RATS 类方法（oracle_rats/dp_nsmdp）在突变下的
worst-case 半径 c=d·L_p·tau 与论文设定无对应关系（论文里 t 从 t₀ 连续推进，
我们这里模型在 t₀ 后不再演化）。

### 2.4 γ
RATS 论文 bridge 用 γ=0.9；ADA-MCTS 论文未注明（表头仅写 "discounted
rewards"）。我们按用户指示（CLAUDE.md）用 0.99。
**后果**：γ 影响 RATS worst-case 树的贴现结构与 DP 值，任何与论文的数值
对比都要先对齐 γ。

### 2.5 cliff 每步惩罚缺失（❌）
ADA-MCTS 论文 §4 明确：cliff 环境 **"the agent concedes a penalty for each
step it takes (except the goal)"**。我们的 cliff 只有到达 G 得 +1、掉进 H
得 −1，中间格 0。
**后果**：我们表里"折现 return"的数值与论文 Table 1（cliff 0.778-0.883 等）
**不可比**——论文的最优策略要权衡"最短路径 vs 每步代价"，我们的最优策略
在 max_steps=100 内无所谓步数。goal rate（论文的 return 约定，holes=0）
不受影响，仍可比。

### 2.6 MCTS 迭代数（❌）
论文全部 30000 次迭代；我们 1000（运行时间考虑）。已验证（2026-08-06
§8.2）：bridge stationary 在 m=2000 时 0.68→0.86，接近 BNN-RATS 0.91。
**后果**：我们的 ada_mcts 数字是"严重欠配"的 baseline，与论文 Table 1
不可比；对比结论（ada_mcts 弱）反映的是欠配配置而非方法本身。

### 2.7 RATS 深度语义
- RATS 论文自己的 bridge 实验：d_max=6，桥小、**叶子即终止态**，无
  heuristic 误差（Property 5 的意义）。
- ADA-MCTS 论文对比：d=3（"higher depths prohibitively expensive"）。
- 我们：d=6；cliff 4x12 上 S→G 约 12 步 > 6，**叶子非终止**，heuristic
  (γ^dist) 介入；bridge 上叶子基本终止（桥 3-4 步内可达）。
**后果**：我们的 oracle_rats 在 cliff 上是"带 heuristic 的 RATS"，
与论文"叶子终止"或"d=3"两种设定都不完全相同；heuristic 误差有界性
(Property 5) 保证误差 ≤ γ^(d_max−d)·δ，γ=0.99 下误差衰减慢（γ^6≈0.94），
heuristic 影响显著——这与我们观察到的 oracle_rats < dp 在 p≤0.6 一致。

## 3. 结论与建议

**可保留的对比（数值有效）**：
- 我们的方法之间的横向对比（dp/oracle_rats/static/adaptive/ada_mcts 在
  同一设定下）——设定一致性保证内部比较成立。
- goal rate（holes=0 约定）与论文表格的**趋势**比较。
- ADA-MCTS 论文的 p 集合、slip 结构、EPS 参数——忠实。

**若要数值复现论文表格，需做**：
1. **RATS 论文 bridge**：重建原版 bridge（洞紧邻桥面、垂直 slip、t=0
   确定、slip 连续增至 0.45、ε∈[0,1] 左右权重）、γ=0.9、L_p=1、d_max=6、
   报告 return-vs-ε 曲线 + CVaR 5%。
2. **ADA-MCTS 论文表**：cliff 加每步惩罚、MCTS 30000 次、RATS d=3、
   对齐 γ 与变化时刻；用论文的 HiP-MDP 转移设定做 \hat p_{k-1} 基线
   （我们用的是预训练 BNN+counts，是不同方法）。
3. 或在论文/文档中明确声明"我们的 gridworld 是变体设定"，列出上述差异。

**已确认无问题的**：p 集合、p₀=0.7、cliff slip 结构、EPS_E/EPS_A、
FrozenLake reward、oracle 基线的角色分工、γ=0.99（用户指定）、
ts-0 突变（ADA-MCTS 离散设定的实例化）。
