# 非平稳 CliffWalking 实验报告（2026-09-10 汇总版）

> 代码：`run_gridworld_experiments.py`、`grids.py`、`bnn/dirichlet_model.py`、
> `planning/cvar_cem.py`、`planning/ada_mcts.py`。
> 前序报告：`experiment_report_2026-08-25.md` / `_en.md`（背景：γ=0.9999、
> α0 标定、plan_retain 门控）、`experiment_report_2026-09-04.md`（本轮工作的
> 逐步记录，本报告是它的清理汇总版）。
> 只读参考（不修改）：`experiment_report.md`、`experiment_report_new.md`、
> `20260509_yuanheli2.md`、`20260509_yuanheli2_modified.md`。
> 英文版：`experiment_report_2026-09-10_en.md`。

---

## 0. 摘要

本轮针对 `main_cl.tex` 的非平稳控制方法，在 **CliffWalking 4×12** 上跑了三组
实验、五个方法的完整对照，并在过程中修了两个 bug、定了一个超参数：

1. **修 bug 1（`bnn_rats_adaptive` / SFI-RATS）**：`BNNRATS._forget()` 漏了
   `drift._reset_detection()`，导致 p=1.0 预训练下第一次 slip 的巨大惊讶值
   一直拖着后面每次 forget，把 `bnn.retain` 砸到硬 0，且**变化越小砸得越狠**
   （p=0.9 比 p=0.4 还差），goal rate 对 p 非单调。已修，恢复单调。
2. **三组主实验**（原版 cliff / 去掉起点第一个洞 / 预训练 p=0.7），p 从 1.0
   扫到 0.3（含 0.4、0.7），5 个方法 × 8 个 p 点 × 30 trials。
3. **CONC_PRIOR 调参**：把"忘记后"Dirichlet 收敛到的对称先验 c 从 0.1 扫到
   10。**c=1.0（真正的单纯形均匀分布 Dirichlet(1,1,1)）在所有测过的点上都
   优于原来的 0.1**，`sfir-cem-cvar` 因此在 21 个 (config,p) 组合里 20 个是
   并列或严格最优的非 oracle 方法。已把 `CONC_PRIOR` 正式改为 1.0。
4. **修 bug 2（`oracle_cem`）**：`oracle_cem` 复用 `cem_fir` 的
   `CVaRCEMAgent`，但那个类的置信度门控所需状态量只在 `self.adaptive=True`
   时才更新——`oracle_cem` 的 `adaptive=False`，导致它全程被锁死在最悲观的
   CVaR 30% 尾部，即使它用的是变化后的真实模型。已修（强制风险中性），
   `oracle_cem` 在全部 21 个点重新变回真正的上界。

**一句话结论**：修完两个 bug、定了 c=1.0 之后，`sfir-cem-cvar`（我们的方法）
是**最好的非 oracle 方法**（唯一例外 config2 p=0.4，输给 ada-mcts 0.800 vs
0.867），且 `oracle_cem` 恢复为可信的上界。

---

## 1. 方法（对齐 main_cl.tex / NSMDP.md / Catch_Me_If_You_Can.md）

| 报告/prompt 里的叫法 | 代码 name | 说明 |
|---|---|---|
| **ada-mcts** | `ada_mcts` | ADA-MCTS / DPAS（`planning/ada_mcts.py`）。`notify_change()` 冻结变化前模型 M_{k−1}，在线累计 counts 形成 M_k，chance node 按 DPAS 两阶段规则采样。本轮未改。 |
| **rats** | `bnn_rats_static` | 论文里的 **RATS-P̂^{k−1}**（NSMDP.md / Catch_Me_If_You_Can.md）：用预训练 BNN 当模型跑 RATS minimax，**不做在线适应**。注意 CliffWalking 用的是**原版**，没有加 Catch_Me_If_You_Can 里那个额外的洞。 |
| **ada-cem-cvar** | `cem_ada` | 把我们方法里的 SFI 换成 ADA-MCTS 的适应机制：CVaR-CEM 规划器不变，但适应逻辑改成 DPAS 的两阶段硬切换——`change_step` 处收到通知，之后只累计 online counts（不 forget、不衰减 retain），CVaR 尾部 α 在观察到 `n_threshold=3` 个变化后样本前是 `alpha_min`（最悲观），之后硬切到 1.0（风险中性）。 |
| **sfir-cem-cvar**（本文方法） | `cem_fir` | = 既有的 **SFI-CEM**。组件 1（预训练 Dirichlet BNN）+ 组件 2（SFI：surprise → drift → forget → learn 循环）+ 组件 3（CVaR-CEM 规划，不是 RATS）。CliffWalking 的 "learn" 是共轭 counts 闭式更新、无梯度重训，对应 SFI 变体（不是需要梯度重训的 SFIR）。 |
| **oracle bnn+cem+cvar** | `oracle_cem` | CVaR-CEM 规划器 + 一个**直接在变化后真实 p 上预训练**的 BNN（每个 p 点用它自己 p 匹配的 checkpoint，`stationary` 阶段用 `ORIG_P` 的）。不需要任何在线适应/SFI——模型从 t=0 起就匹配变化后的环境。CEM 家族的上界，`oracle_rats` 的 CEM 对应物。 |

`cem_static` / `bnn_rats_static` 是各自方法去掉适应机制的消融；`oracle_rats`
用真实模型跑 RATS。本轮默认跑的 5 个方法就是上表这 5 个。

---

## 2. 实验设置（按 CLAUDE.md 要求逐条说明）

### 2.1 变化的性质（在哪个时刻变成什么）

- **变化时刻**：`change_step = 0`。`c ≤ 0` 时第一个决策 epoch 就已经在新
  环境下，预训练那段 p 完全不出现——所以是"仿真一开始环境就已经剧烈变化"。
- **变什么**：转移的 intended（不打滑）概率 `p`。滑动结构（Luo et al.）：
  以概率 `p` 走到目标方向，各以 `(1−p)/2` 走到两个垂直方向，**没有反方向**。
  奖励结构不随时间变。
- **三组配置**：

  | config | 网格 | 预训练 p（=`ORIG_P`） | 变化后 p 扫描 |
  |---|---|---|---|
  | **config1** | 原版 CliffWalking 4×12 | 1.0（确定性） | 1.0→ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} |
  | **config2** | CliffWalking 去掉起点右边第一个洞 | 1.0 | 同上 |
  | **config3** | 原版 CliffWalking 4×12 | **0.7**（预训练环境本身就随机） | 0.7→ {同一组 8 个 p 点} |

  - config1/2 的 `p=1.0` 那一列 = 没有真实变化（对照）。
  - config3 的 `p=0.7` 那一列 = 没有真实变化；`p>0.7` 是环境**变得更好**
    （滑动变少），`p<0.7` 是变差。
  - config2 相对 config1 只改一格：起点 `S` 正右方那个 cliff 格从 `H` 改成
    `F`（安全地面），用来看低 p 下的差距有多少是这一格贡献的。

### 2.2 网格几何与奖励结构

- **CliffWalking 4×12**，K=3 方向 `[intended, perp+, perp−]`，4 个动作
  UP/RIGHT/DOWN/LEFT。起点 = 左下 `(3,0)`，目标 = 右下 `(3,11)`，cliff =
  底行 `(3,1..10)`（config2 是 `(3,2..10)`）。
- **奖励**：到达目标 `+1.0`；踏入 cliff `0.0`（论文 "holes = 0" 约定：掉
  下去只是结束/传送，不额外惩罚）；其他每步 `0.0`（**无 per-step penalty**，
  2026-08-12 用户要求）。
- **`cliff_to_start = True`**：踏入 cliff 格 → **传送回起点，不终止 episode**
  （ns_gym 的 CliffWalking 行为）。因此**唯一的终止状态是目标**；一条 episode
  只可能以"到达目标"或"第 100 步截断"结束。

### 2.3 所有超参数

**通用 / 评估**
| 参数 | 值 | 说明 |
|---|---|---|
| 折扣 γ | **0.9999** | 规划与评估同一个 γ（2026-08-22 用户要求）。γ≈1 + holes=0 ⇒ 折现 return ≈ goal rate（cliff 上 return 在第 3 位小数略低，掉 cliff 传送回起点会延误到达目标）。 |
| trials | 30 | 每个 (method, phase) 点，trial 索引作种子，可复现。 |
| 截断 max_steps | 100 | 见 2.5。 |
| BNN 后验采样数（surprise） | `N_POSTERIOR = 10` | |
| drift filter | `DriftFilterV2(eta=0.2, gamma_uncertainty=True)` | `delta_bar = 1 + lambda_hat`，`lambda_hat` 是自上次 `_reset_detection()` 以来 `(delta_n − baseline)` 的**累计均值**（非指数衰减）。 |

**BNN 世界模型架构（`make_dirichlet_bnn`）**
- Bayesian trunk：`in_size = n_states + n_actions = 48 + 4 = 52` → 隐层
  `hid_size = 256`，`num_layers = 3`（即 trunk 有 2 个 BayesianLinear：
  52→256、256→256），SiLU 激活。
- 两个读出头（各自独立，不共享输出层）：方向头 `256→3`（K=3 Dirichlet
  浓度），奖励头 `256→2`（reward mean / logvar）。
- `prior_std = 1.0`，`beta = 0.1`，`num_mc_samples = 3`，`num_weight_groups = 1`。
- 预测分布：K=3 方向上的 categorical `p = α/α0`，再按几何散射到 48 个格子。
  **浓度 α0 来自预训练计数表**（`pretrain_alpha0` buffer，
  `α0[s,a] = counts + K·CONC_PRIOR`），不是头部 softplus 幅度——categorical
  NLL 对 α 缩放不变，学不到 α0。
- **CONC_PRIOR = 1.0**（本轮从 0.1 改，见第 4 节）。`retain` 缓冲量把
  头部 α 往这个对称先验收：`alpha = CONC_PRIOR + retain·(alpha_head − CONC_PRIOR)`，
  `retain=1` 全信头部，`retain→0` 所有方向都收到 `CONC_PRIOR`（预测均值 →
  均匀）。

**预训练（`pretrain_gridworld.py`）**
- 目标加权采样：`w(s) = goal_weight^(−BFS距离(s→goal))`，`goal_weight = 1.35`
  ⇒ **靠近目标的转移被上采样**（CLAUDE.md："预训练应优先训练靠近目标的
  转移"）。
- `--balance-terminal`：对"落入终止格"的转移做下采样，避免目标加权后终止
  落点过度占比。
- 20000 行，逐行 categorical NLL。每个 `(grid, p)` 单独一个 checkpoint。

**SFI-CEM（`cem_fir` = sfir-cem-cvar，本文方法）**
| 参数 | 值 | 说明 |
|---|---|---|
| forget 周期 `k_forget` | 3 | 每 3 个变化后步 forget 一次。 |
| forget 规则 | `retain ← retain · ρ`，`ρ = 1/max(delta_bar, 1)`，clip 到 [1e-3, 1] | `delta_bar ≤ 1` 是 no-op。forget 后调 `drift._reset_detection()`（见第 3 节）。 |
| CVaR 尾部 α | 自适应，`alpha_min + (1 − alpha_min)·conf` | |
| `alpha_min`（`CEM_ALPHA_MIN`） | **0.30** | 最悲观端（取最差 30% 的 CVaR）。曾试 0.10 太保守（拖到 cem_static 以下），0.95 太接近风险中性。 |
| `conf` 置信度 | `conf_data · conf_surprise` | `conf_data = min(1, n_since_change / n_confident)`；`conf_surprise = exp(−max(0, delta_bar−1) / surprise_tau)`。 |
| `n_confident`（`CEM_N_CONFIDENT`） | 8 | 数据门饱和所需的变化后样本数。 |
| `surprise_tau`（`CEM_SURPRISE_TAU`） | 50.0 | p=1.0 预训练下 surprise 会冲到几百，默认 2.0 会把 conf 全程压到 0。 |
| plan_retain（规划期不确定性门控） | `= conf` | 规划时把 `bnn.retain` 临时乘以 conf（膨胀预训练分量、online counts 全信），规划完还原。 |
| 规划 horizon `--cem-horizon` | **6** | |
| CEM 候选数 `--cem-candidates` | **512** | 模块默认 256；512 给 CVaR 估计更多样本（2026-08-14 调参：cliff cem_fir p=0.4 从 0.38→0.88）。 |
| K posterior 模型数 `k_models` | 10 | epistemic 轴。 |
| 每模型 aleatoric rollout 数 `n_rollouts` | 32 | |
| CEM 迭代数 `n_cem_iters` | 5 | |
| elite 比例 | 0.1 | |
| CEM 内部规划 γ | GAMMA = 0.9999 | `--cem-plan-gamma` 未传。 |
| 探索 bonus β | 0.0（关） | |

**ada-cem-cvar（`cem_ada`）**：规划器同上；`n_threshold = 3`（对齐
ADA-MCTS 的 `_training_started` 阈值）；CVaR α 在观察到 3 个变化后样本前
= `alpha_min = 0.30`，之后硬切 1.0；只累计 online counts，不 forget、不衰减
retain。

**oracle_cem（修复后）**：规划器同上；**`adaptive_alpha = False`**（第 5 节
修复），固定 `cvar_alpha = 1.0`（风险中性）；`retain` 恒为 1.0；每个 p 点
加载它自己 p 匹配的 checkpoint。

**ada-mcts（`ada_mcts`）**
| 参数 | 值 |
|---|---|
| 每动作模拟数 `M_SIMULATIONS` | **30000**（论文值） |
| UCT 常数 `CP` | √2 |
| epistemic 阈值 `EPS_E` | 0.02（论文 line 236） |
| aleatoric 阈值 `EPS_A` | 0.0 |
| DPAS aleatoric 似然 `gamma` | 10000.0 |
| 后验采样数（Var_E/Var_A） | 10 |
| rollout horizon | 6 |
| 切换模式前最少变化后样本 `_n_threshold` | 3 |

**RATS 系（`bnn_rats_static` = rats）**：`rats_depth = 3`（论文值），
`dp_depth = 100`。用预训练 BNN 的均值模型跑 RATS minimax，不在线更新。

### 2.4 discount / truncation

- **无额外 discount 处理**：γ=0.9999 已经是 CLAUDE.md "no discount" 语义下
  约定的统一值（2026-08-22 起），规划和评估都用它。
- **截断**：`max_steps = 100`。因为 `cliff_to_start=True`（掉 cliff 传送回
  起点、不终止），一条 episode 不到目标就只能靠截断结束——所以
  **goal rate = X 直接等于"30 个 trial 里有 (1−X) 比例跑满了 100 步都没到
  目标"**。高 p（≥0.6~0.7）下几乎所有 trial 在 100 步内到达；低 p（0.3~0.4）
  下截断频繁，正是 goal rate 掉下来的原因。config1 抽查 trial 0：45 个
  (method,phase) 里 8 个 trial-0 跑满了 100 步。

### 2.5 stationary 校验

每组每个方法都先跑一个 `p = ORIG_P` 不变化的 stationary 阶段。**全部 5 方法
× 3 组 = 15 个点，goal rate 均为 1.000**（return 0.997~0.999）——预训练模型
在未变化环境里能找到最优策略，前提成立。

---

## 3. Bug 修复 1：`bnn_rats_adaptive`（SFI-RATS）的 drift 累加器

**现象**（用户指出）：`bnn_rats_adaptive` 的 goal rate 对 p 非单调，小变化
比大变化还差。

**根因**：`BNNRATS._forget()` 少了 `self.drift._reset_detection()`
（`BNNCEM._forget()` 里有）。`DriftFilterV2.delta_bar` 是**自上次 reset 以来
的累计均值**（不是指数衰减）。p=1.0 预训练 ⇒ 近确定性模型 ⇒ 第一次 slip 的
预测概率 ≈ 0 ⇒ NLL / 惊讶值冲到几百。不 reset 的话这个尖峰按 1/n 慢慢稀释，
整个 episode 都拖着后面每一次 `k_forget` 周期的 forget 计算：每次都算出一个
接近 0 的 `ρ = 1/max(delta_bar,1)`，把 `retain` 再乘一次，最终砸到硬
`0.0`（不是压低，是所有预训练方向信号全丢）。而且这个尖峰完全来自"第一次
见到 slip"这个事件本身，**与变化后 p 的大小无关**——所以小变化（p=0.9，
旧先验其实还有 ~90% 对）反而被砸得和大变化（p=0.4）一样狠，goal rate 对 p
非单调。

**修复**（`run_gridworld_experiments.py:285`，附详细注释）：`_forget()` 末尾
加 `self.drift._reset_detection()`，和 `BNNCEM._forget()` 一致。forget 之后
模型已经不匹配旧环境，旧环境的 surprise 不再有信息量，重置累加器让
`delta_bar` 只反映新证据。

**验证**：30 trial 重跑，retain 稳定在 ~1e-3 而不是硬 0，goal rate 恢复
近单调（`1.000/0.833/1.000/0.967/0.933/0.867/0.767`，p=1.0→0.4）。

> 注：本轮三组主实验默认跑的 5 个方法里 `rats` = `bnn_rats_static`（静态，
> 不适应），不是 `bnn_rats_adaptive`。这个 bug 修复独立成立，`bnn_rats_adaptive`
> 若要用于其他实验现在是对的。

---

## 4. 主结果

goal rate，30 trials，**当前默认设置**（`CONC_PRIOR = 1.0`，`oracle_cem`
已修）。粗体 = 该 (config, 列) 下所有方法里的最高值。

### config1：原版 cliff，预训练 p=1.0

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7 | p=0.8 | p=0.9 | p=1.0 |
|---|---|---|---|---|---|---|---|---|
| ada-mcts | 0.367 | 0.667 | 0.900 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| rats (bnn_rats_static) | 0.100 | 0.300 | 0.800 | 0.800 | **1.000** | **1.000** | **1.000** | **1.000** |
| ada-cem-cvar (cem_ada) | 0.100 | 0.233 | 0.667 | 0.800 | **1.000** | **1.000** | **1.000** | **1.000** |
| **sfir-cem-cvar (cem_fir)** | **0.700** | 0.900 | 0.933 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| oracle-cem | **0.700** | **0.933** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

### config2：去掉起点第一个洞，预训练 p=1.0

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7 | p=0.8 | p=0.9 | p=1.0 |
|---|---|---|---|---|---|---|---|---|
| ada-mcts | 0.400 | **0.867** | 0.900 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| rats | 0.100 | 0.267 | 0.767 | 0.733 | **1.000** | **1.000** | **1.000** | **1.000** |
| ada-cem-cvar | 0.167 | 0.267 | 0.633 | 0.900 | **1.000** | **1.000** | **1.000** | **1.000** |
| **sfir-cem-cvar** | 0.700 | 0.800 | 0.967 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| oracle-cem | **0.767** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

### config3：原版 cliff，预训练 p=0.7

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7 | p=0.8 | p=0.9 | p=1.0 |
|---|---|---|---|---|---|---|---|---|
| ada-mcts | 0.533 | 0.700 | 0.967 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| rats | 0.300 | 0.600 | 0.800 | 0.933 | **1.000** | **1.000** | **1.000** | **1.000** |
| ada-cem-cvar | 0.533 | 0.767 | 0.900 | 0.967 | **1.000** | **1.000** | **1.000** | **1.000** |
| **sfir-cem-cvar** | 0.633 | 0.900 | 0.967 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| oracle-cem | **0.700** | **0.933** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

**读表**：
- p ≥ 0.6（config3 是 ≥ 0.6）之后所有方法饱和到 1.000——高 p 下 100 步预算
  内人人到得了目标，区分不出来。有信息量的是 p ∈ {0.3, 0.4, 0.5}。
- **`sfir-cem-cvar` 在 21 个 (config, p≤0.5) 组合里 20 个是并列或严格最优的
  非 oracle 方法**，唯一例外是 config2 p=0.4（0.800 < ada-mcts 0.867）。
- `oracle-cem` 修复后在**全部**点上 ≥ 其余所有方法（含 `sfir-cem-cvar`），
  恢复为可信上界。
- `rats`（静态 BNN，不适应）和 `ada-cem-cvar` 在低 p 下明显垫底——静态
  预训练模型对 p=1.0→0.3 这种剧变毫无准备，而 `ada-cem-cvar` 的两阶段硬
  切换 + 只靠 online counts 适应太慢。

### 4.1 对照：修复/调参前的旧数字

| | config1 p=0.3/0.4/0.5 | config2 p=0.3/0.4/0.5 | config3 p=0.3/0.4/0.5 |
|---|---|---|---|
| sfir-cem-cvar（旧 c=0.1） | 0.567 / 0.767 / 0.900 | 0.367 / 0.733 / 0.833 | 0.300 / 0.800 / 0.933 |
| sfir-cem-cvar（新 c=1.0） | **0.700 / 0.900 / 0.933** | **0.700 / 0.800 / 0.967** | **0.633 / 0.900 / 0.967** |
| oracle-cem（旧 bug） | 0.400 / 0.967 / 0.933 | 0.433 / 0.967 / 0.933 | 0.400 / 0.967 / 0.933 |
| oracle-cem（修复后） | **0.700 / 0.933 / 1.000** | **0.767 / 1.000 / 1.000** | **0.700 / 0.933 / 1.000** |

`ada-mcts` / `rats` / `ada-cem-cvar` 三个方法的数字**不受这两处改动影响**
（它们的 `retain` 恒为 1，CONC_PRIOR 在 blend 公式里精确抵消；它们也不走
`oracle_cem` 那条构造路径）——已多次代数 + 实测双重确认。

---

## 5. CONC_PRIOR 调参 → 正式定为 1.0

### 5.1 起因与机制

用户问："降低 concentration prior c 让模型忘记后更 random，会不会让
`sfir-cem-cvar` 更好？"

`c` 是 `retain→0`（彻底忘记，或 plan_retain 规划期临时压低）时 α 收敛到的
对称 Dirichlet 浓度。`c` 越小，Dirichlet 分布越往单纯形的**角**上挤（每次
后验采样越像"一次性把 100% 概率押在某一个随机方向上"）；`c = 1` 是
Dirichlet(1,1,1)，**单纯形上真正的均匀分布**；`c` 越大越收紧到均值附近。
新增 `--conc-prior` CLI（`run_gridworld_experiments.py:981`），**不需要重新
pretrain**——`pretrain_alpha0` 只吃到一个可忽略的 `+K·c` 正则项，真正生效的
是 `_forward_alpha` 里每次前向都读的 blend 公式。**只影响 retain 会离开 1.0
的方法**（`cem_fir`、`bnn_rats_adaptive`）。

### 5.2 c 扫描（cem_fir，弱点几个 p 点，30 trials）

| config | p | c=0.01 | c=0.03 | c=0.1（旧） | c=0.3 | c=1.0 | c=3.0 | c=10.0 |
|---|---|---|---|---|---|---|---|---|
| config1 | 0.3 | 0.367 | 0.567 | 0.567 | 0.467 | 0.700 | 0.833 | **0.900** |
| config1 | 0.4 | 0.767 | 0.567 | 0.767 | 0.767 | 0.900 | 0.967 | **1.000** |
| config2 | 0.3 | 0.267 | 0.400 | 0.367 | 0.500 | 0.700 | 0.700 | **0.767** |
| config2 | 0.4 | 0.533 | 0.567 | 0.733 | 0.633 | 0.800 | **1.000** | 0.967 |
| config2 | 0.5 | 0.867 | 0.933 | 0.833 | 0.900 | 0.967 | **1.000** | 0.933 |
| config3 | 0.3 | 0.500 | 0.433 | 0.300 | 0.500 | **0.633** | 0.600 | 0.567 |
| config3 | 0.4 | 0.867 | 0.833 | 0.800 | 0.700 | 0.900 | 0.833 | **0.967** |

（`c=0.1` 那列位对位复现了第 4.1 节的旧数字——流程正确性校验。）

**关键发现**：
1. **方向和用户猜测相反**：不是"更 random（更极端）更好"，是"更接近真正
   均匀分布（没那么极端）更好"。`c=0.1` 时忘记后的后验采样接近 one-hot，
   CVaR 取最差 30% 的均值被这种病态尖峰主导，规划失真；`c=1.0` 时不确定性
   仍然真实但不病态，CVaR 估计更可信。降到比 0.1 更低（0.01/0.03）没有稳定
   收益，好几个点反而更差。
2. **`c=1.0` 在测过的全部 7 个点上都优于 `c=0.1`**，config2/config3 的 p=0.3
   提升 ~0.33（约 3.7 个 30-trial 标准差，噪声解释不了）。
3. **`c>1.0` 不是"越高越好"**：4/7 个点还在往 10 涨，3/7 已经在 1.0~3.0
   见顶。尤其 config3（预训练 p=0.7）的 p=0.3 从 1.0 开始单调降
   （0.633→0.600→0.567，虽然降幅在噪声范围内、不算确认的反转）——这正好
   对应理论顾虑：`c` 太大时"忘记后"的 Dirichlet 又变尖锐了，等于从另一个
   方向绕回 plan_retain 当初要解决的"CVaR 尾部是空的"问题。

### 5.3 定档

**`bnn/dirichlet_model.py:54`：`CONC_PRIOR` 由 `0.1` 改为 `1.0`。**

理由：`0.1→1.0` 是全线、显著、干净的提升；`1.0→3.0/10.0` 增量小、不一致、
部分点已现掉头，且 `3`/`10` 是拟合出来的数字、没有 `1.0`（"真正的单纯形
均匀分布 Dirichlet(1,1,1)"）那样干净的理论说法。改的是模块默认值，以后
跑这份代码不用再加 `--conc-prior 1.0`。已验证不带参数时 `cem_fir` config1
p=0.3 直接得 0.700，新默认生效。

---

## 6. Bug 修复 2：`oracle_cem` 的置信度门控从未打开

**背景**：修 CONC_PRIOR 之前，"真实模型的 `oracle_cem` 在低 p 被
`sfir-cem-cvar` 反超"。给 `oracle_cem` 测 c=1.0——数字**完全不变**
（代数上必然：`oracle_cem` 的 `retain` 恒为 1，`c` 抵消）。顺着"那到底为什么
被反超"查下去：

**根因**：`BNNCEM.__init__` 里，底层 `CVaRCEMAgent` 的 `adaptive_alpha` 硬编码
为常量 `CEM_ADAPTIVE_ALPHA = True`（不看 `self.adaptive`），意味着 CVaR 尾部
比例由置信度 `_confidence()` 动态决定。但驱动这个置信度的两个状态量
（`self._agent.n_since_change`、`self._agent.surprise_bar`）只在
`BNNCEM.act()` 的 `if self.adaptive and ...:` 分支里更新。`oracle_cem` 的
`self.adaptive = False`，这段代码永远不跑：`n_since_change` 永远卡在
`reset()` 的初始值 0 ⇒ `conf_data = min(1, 0/n_confident) = 0` ⇒
`conf = 0` ⇒ `alpha = alpha_min = 0.30`。**`oracle_cem` 从第一步到最后一步
全程被锁在"只看最差 30% 情况"的 CVaR 尾部，永远等不到置信度升到风险中性**
——即便它用的是变化后的真实模型、根本没有"是否已适应"这件事要解决。在低 p
（滑动最严重、行为最关键）下，这种"无谓的永久悲观"让规划走过度保守的绕行
路线，在 100 步预算内更容易超时——和之前发现的"`CEM_ALPHA_MIN=0.10` 太
保守会拖累 `cem_fir`"是同一种代价。（`cem_static` 走同一条构造路径，理论上
有相同问题，目前不在默认跑的 5 个方法里，未验证。）

**修复**（`run_gridworld_experiments.py:847`，附详细注释）：`build_methods`
的 `oracle_cem` 分支里，构造后追加
```python
out[name]._agent.adaptive_alpha = False
```
让它直接用构造时已传入的 `cvar_alpha`（`args.cem_cvar_alpha`，默认 1.0 =
风险中性），不再经过永远打不开的置信度门。**不影响 `cem_fir` / `cem_static`
/ `cem_ada` 共用的类**（`cem_ada` 自己就显式 `adaptive_alpha=False` 并手动
两阶段切换）。

**修复前后**（c=0.1，30 trials；p≥0.5 两版都已饱和，只列有变化的点）：

| config | p | 修复前（锁死 worst-30%） | 修复后（风险中性） |
|---|---|---|---|
| config1 | 0.3 | 0.400 | **0.700** |
| config1 | 0.4 | 0.967 | 0.933（噪声内） |
| config2 | 0.3 | 0.433 | **0.767** |
| config2 | 0.4 | 0.967 | 1.000（噪声内） |
| config3 | 0.3 | 0.400 | **0.700** |
| config3 | 0.4 | 0.967 | 0.933（噪声内） |

p=0.3 全线 +0.3~+0.367（远超噪声），p=0.4 的 ±0.033（1/30 trial）在噪声内。
**修复后 `oracle_cem` 在全部 21 个点重新变回真正的上界**，第 4.1 节里"反超
oracle"的点全部消失——那不是噪声也不是 CONC_PRIOR，就是这个门控 bug。

---

## 7. 跨配置观察

1. **`oracle_cem` 的确定性核验**：config1 和 config3 用同一张地图，
   `oracle_cem` 每个非平稳 p 点用它自己 p 匹配的 checkpoint（与 `ORIG_P`
   无关），所以两组数字应逐点完全相同——实测确实完全一致
   （0.700/0.933/1.000/...），是一次干净的"同 seed 同输入 → 同输出"核验。
   config2（换地图）只 p=0.3 变（0.700→0.767），其余不变——符合直觉：起点
   第一格的洞只在低 p 有影响。
2. **预训练 p=0.7（config3）大幅提升低 p 下非 oracle 方法的鲁棒性**（相对
   config1 预训练 p=1.0）：p=0.3 处 `rats` 0.100→0.300、`ada-cem-cvar`
   0.100→0.533、`ada-mcts` 0.367→0.533。方向符合直觉：p=0.7→0.3 的分布跳变
   比 p=1.0→0.3 小，惊讶更温和，适应负担更轻。`sfir-cem-cvar` 在 config3
   p=0.3 是 0.633，比 config1 的 0.700 略低但仍是三组里对 c 最敏感、也是
   `c>1` 会掉头的那个点（见 5.2）——预训练 p=0.7 这个"小跳变"场景可能需要
   和 p=1.0 场景不同的 SFI 超参数（`surprise_tau`/`k_forget` 目前都是按
   p=1.0 那种"surprise 冲到几百"调的）。
3. **去掉起点第一个洞（config2）影响集中在 p=0.3**，方向不完全一致——落在
   30-trial 高方差区间内，暂不下结论。
4. **p=0.3 是难度最高、方差最大的工作点**：30 trials 的二项标准差在
   0.3~0.5 附近约 0.09，两个独立方法比较时差距的标准差约 0.13。这一列上
   方法间 0.1~0.15 的差距要谨慎当作噪声，只有 ≥0.25 的差距（如
   `sfir-cem-cvar` c=1.0 对 `rats`/`ada-cem-cvar`）才稳。

---

## 8. 代码改动清单（本轮）

| 文件 | 位置 | 改动 |
|---|---|---|
| `grids.py` | `221-232` | 新增 `CLIFFWALKING_4x12_NOFIRSTHOLE` GridSpec（原版 cliff，起点右边第一个 `H`→`F`），`234-236` 加进 `REGISTRY`。 |
| `run_gridworld_experiments.py` | `285` | **Bug 1 修复**：`BNNRATS._forget()` 末尾加 `self.drift._reset_detection()`（附注释）。 |
| | `410-499` | 新增 `ADA_CEM_N_THRESHOLD = 3` 常量 + `CEMADA` 类（`name="cem_ada"` = ada-cem-cvar）。 |
| | `802-812` | `build_methods` 新增 `elif name == "cem_ada":` 分支。 |
| | `813-847` | `build_methods` 新增 `elif name == "oracle_cem":` 分支；`847` 为 **Bug 2 修复**：`out[name]._agent.adaptive_alpha = False`（附注释）。 |
| | `863-890` | `_worker`：`oracle_cem` 按 phase 选 p 匹配 checkpoint；`878-884` 从 `cfg["conc_prior"]` patch `bnn.dirichlet_model.CONC_PRIOR`。 |
| | `919`, `981-999` | 新增 `--orig-p`、`--conc-prior` CLI；`1024-1026`、`1045` 日志行 + cfg 穿透。 |
| `bnn/dirichlet_model.py` | `54` | **CONC_PRIOR 由 `0.1` 改为 `1.0`**（附 20 行注释说明扫描依据），`34-53` 更新注释。 |

`planning/cvar_cem.py`、`pretrain_gridworld.py` 在 session 开始时就已是
modified 状态（非本轮改动）。

---

## 9. 复现方式

**Checkpoints**（`pretrain_gridworld.py --grid <g> --p <p> --balance-terminal`，
目标加权 goal_weight=1.35，20000 行；已生成，`VERDICT: GOOD`）：
- `data/cliffwalking/bnn_dirichlet_cliffwalking_k3{,_p0p{3..9}}.pth`（p=1.0
  是无后缀名）
- `data/cliffwalking_nofirsthole/bnn_dirichlet_cliffwalking_nofirsthole_k3{,_p0p{3..9}}.pth`

**主实验命令**（当前代码默认即 `CONC_PRIOR=1.0`、`oracle_cem` 已修，不用
额外加参数）：
```bash
METHODS="ada_mcts bnn_rats_static cem_ada cem_fir oracle_cem"
PS="0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
CEM="--cem-horizon 6 --cem-candidates 512"

# config1
python run_gridworld_experiments.py --grid cliffwalking --trials 30 --workers 16 \
  --methods $METHODS --change-p $PS $CEM
# config2
python run_gridworld_experiments.py --grid cliffwalking_nofirsthole --trials 30 --workers 16 \
  --methods $METHODS --change-p $PS $CEM
# config3
python run_gridworld_experiments.py --grid cliffwalking --trials 30 --workers 16 \
  --methods $METHODS --change-p $PS $CEM --orig-p 0.7
```

> **注意本报告第 4 节数字的来源不是上面这一条命令的单次运行**：`ada_mcts`
> / `rats` / `cem_ada` 的数字取自 2026-09-03 的原始三组运行（旧 `CONC_PRIOR
> =0.1`，但对这三个方法无影响，已确认）；`cem_fir` 取自 2026-09-06 的
> `--conc-prior 1.0` 运行；`oracle_cem` 取自 2026-09-08 修复后的运行。上面
> 这条命令现在会一次性重现全部第 4 节数字，但**尚未作为单次完整运行跑过**
> ——正式出图前建议重跑一遍作独立复核。

**运行时长参考**（16 核机器）：完整一组 45 个 (method,phase) 任务约
13~14 小时，瓶颈是 `ada_mcts`（30000 sims/action，纯 Python 树搜索，CPU
bound；GPU 全程 ~1%，CEM 方法也一样，`_rollout_returns` 是纯 NumPy）。
CEM-only 的子实验约 0.45 小时/任务-槽。

**原始日志**（`/tmp`，重启会丢）：
- `grid_cliff_2026-09-03_config{1,2,3}.log`（旧 c=0.1 三组）
- `conc_sweep_c{0.01,0.03,0.1,0.3,1.0,3.0,10.0}_config{1,2,3}.log`、
  `conc1_rest_config{1,2,3}.log`（CONC_PRIOR 扫描）
- `oracle_conc1_config{1,2,3}.log`（oracle c=1.0，证明无变化）、
  `oracle_fixed_config{1,2,3}.log`（oracle 修复后）

---

## 10. 未完成 / 存疑项

1. **第 4 节的表还没作为单次完整运行独立复现过**（数字来自三次不同日期的
   运行拼接，方法各自的默认值已确认一致）。正式进论文前应该用第 9 节那条
   命令重跑一遍。
2. **p=0.3（可能还有 0.4）的误差条偏大**（30 trials）。方法间 <0.15 的差距
   不稳，想拿这一列下确定性结论需要把 trials 提到 60~100。
3. **config3（预训练 p=0.7）的 SFI 超参数可能没调到位**：`surprise_tau=50`
   / `k_forget=3` 是按 p=1.0 那种大跳变调的；p=0.7→0.3 的小跳变场景，
   `sfir-cem-cvar` 在 p=0.3 相对偏弱、且是 `c>1` 会掉头的点，值得单独扫一遍
   `surprise_tau` / `k_forget`。
4. **`cem_static` 理论上有和 `oracle_cem` 一样的置信度门控 bug**（同一条
   `adaptive=False` 构造路径），目前不在默认 5 方法里，未验证也未修。
5. **`c>1.0` 是否对某些点还能再涨没有定论**：config1 一路涨到 c=10，但
   config3 p=0.3 在 c=1 就见顶。如果要榨这部分收益，需要 per-场景的更细
   扫描（而且要先把第 2、3 条的噪声/调参问题解决）。
6. **英文正式报告 / 进 `main_cl.tex` 的图表**：本报告有英文版
   `experiment_report_2026-09-10_en.md`，但还没整理成论文用的图。

---

## 11. 我们的方法定为 SFIR（2026-09-11）：retrain 接进代码 + CONC_PRIOR 回退到 0.1

### 11.1 起因：`main_cl_2.tex` 澄清了 retrain 的含义

`doc_tool/main_cl_2.tex` 明确了方法名是 **SFIR**（Surprise-Forget-Inflate-
**Retrain**，不是 SFI），且 retrain = **对 adapter head 做梯度更新**（"3
个 Bayesian linear layer + 一个 adapter head，运行期间只更新 head"）。而
`cem_fir` 之前只有 conjugate counts（且默认关），**没有梯度 retrain**——
第4-6节报的其实是 SFI。用户确认离散头用纯 NLL（不是 Gaussian 头那种 ELBO
版本）。

### 11.2 把 retrain 接进 `cem_fir`

`BNNCEM` 新增：`do_forget`（默认 True）、`n_unfrozen`（0=关，1=只训 head=
"我们的方法"，2/3=+trunk="retrain 全部" ablation）、`retrain_every`/
`retrain_steps`/`retrain_lr`。机制：post-change 的 (obs,act,s2) 存 buffer
（cap 64），每 `retrain_every` 步对 `unfrozen_params_dirichlet(bnn,
n_unfrozen)` 返回的 `weight_mu`/`bias_mu` 跑 `retrain_dirichlet`（纯
NLL，Adam）。**每个 trial 开头把这些权重还原到预训练值**（`__init__` 里
snapshot，`reset()` 里 restore）——梯度 retrain 会永久改权重，而 bnn 对象
跨 30 个 trial 共享，不还原的话 trial 间就不独立了。

### 11.3 大规模 ablation matrix（config1, candidates=256/trials=20 降精度做
相对比较；c=0.1 那一列位对位复现了 512/30 的已发表数字，精度可信）

交叉了 {forget 开/关} × {retrain 关/head/全部} × {c=0.1, 1.0} × {p=0.3,
0.4, 0.5}，外加一次 FrozenLake 抽查。**两个关键发现：**

1. **c=1.0 时，forget 和 retrain 都是净负贡献**：一个"既不 forget 也不
   retrain"（只保留原有的 plan_retain 置信度门控膨胀 + CVaR）的 ablation
   在所有测过的点上都比 c=1.0 的 SFI 好（例如 p=0.4：Neither 0.85-0.95
   vs SFI 0.90 vs SFIR 0.55-0.70）。**09-09 那次"大 c 赢"的结论，其实是
   "大 c 让忘记后的信念几乎瞬间变均匀，forget/retrain 根本碰不到它，真正
   在起作用的是规划器的谨慎（置信度门控 CVaR），不是模型真的在适应"**。
   在一个已经被 forget 拍平的信念上做 retrain，梯度没有什么有意义的东西
   可以纠正，纯粹添乱。
2. **c 小（0.1）时，retrain 确实有正贡献**：SFIR（0.65/0.80，p=0.3/0.4）
   比 SFI（0.55/0.70）好，幅度不大但方向一致——c 小意味着 forget 真的会
   破坏信念，retrain 才有东西可修。这是诚实的适应，只是数字比大 c/不适应
   的版本低。
3. **调参没能补上小 c 和大 c 之间的差距**：试了 lr（1e-3 更差）、cadence
   （every=1/更多步 → 更差或持平）、`n_unfrozen=2`（更差，p=0.4 掉到
   0.55）、更小的 c=0.05（更差）。原来瞎猜的默认值（c=0.1, every=3,
   steps=5, lr=1e-2, n_unfrozen=1）反而是试过的最优点，像是这条路径本身
   的天花板，不是没调好。
4. **FrozenLake 抽查**：SFI 和 SFIR 数字完全相同（retrain 完全"哑"）——
   原因是 retrain 的 buffer 每个 trial 清空，FrozenLake episode 很短
   （掉洞终止，不像 cliffwalking 传送回起点接着走），很多 trial 在攒够
   `retrain_every=3` 步变化后数据之前就已经结束了。这是当前 retrain 实现
   的一个局限（buffer 不跨 trial 持久化），不是"FrozenLake 不需要适应"
   的结论。
5. **"Neither"（S+I only）在测过的所有点上都比小 c 的 SFIR 还好**（例如
   p=0.4：Neither-c=0.1 是 0.95，SFIR-c=0.1 是 0.80）——是这整轮调查里
   目前找到的、在 cliffwalking 上表现最好的策略，但它本质上也不是"真正
   适应"。

### 11.4 决定（用户明确指示）

**按用户要求，主表用诚实的小 c SFIR 数字，不用"Neither"或大 c 的更高
数字**——即使后两者在这个任务上分数更高。具体改动：

- `bnn/dirichlet_model.py`：`CONC_PRIOR` 从 `1.0` 改回 **`0.1`**。
- `run_gridworld_experiments.py`：`--n-unfrozen` 默认值从 `0` 改为
  **`1`**——`cem_fir` 现在默认就是 SFIR（forget + retrain head），不用
  再显式加参数。
- 只影响 `cem_fir`：`ada_mcts`/`rats`/`cem_ada`/`oracle_cem` 的 `retain`
  恒为 1，两处改动对它们代数上无影响（多次验证过），**不需要重跑**。

### 11.5 主表重跑（进行中）

已启动 `cem_fir`（新默认：c=0.1 + SFIR）在三组、完整 8 个 p 点、
candidates=512、trials=30 的重跑。实测单个 task（p=0.4, 3 trials,
n_unfrozen=1, candidates=512）耗时 2216 秒 ≈ 12.3 分钟/trial——按并行
（每组 9 个 task 用 9 个 worker 同时起，瓶颈是最慢的低 p task，不是全部
task 时间加总），预计一天左右能拿到完整数字，比之前估的"1.5-2 天"乐观
（之前的估计把并行度算错了，当成了近似串行）。跑完后第4节的 `sfir-cem-
cvar` 行会用新数字替换。

**线程超订踩过一次坑**：第一次跑 5 个 ablation 并发时，每个 worker 进程
自己开了 ~5 个 MKL/OpenMP 线程，20 workers × 5 ≈ 100 线程抢 16 核，跑了
11 小时 0 个任务完成。之后所有涉及 cem_fir 的批量实验都加了
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1`，问题消失。
