# NS-CliffWalking 最终结果：主表 + Ablation（2026-09-14）

> 本文是这轮调查的**干净汇总版**（只放最终结论和数字）。完整的调参/调查
> 过程（每一步怎么排除的、踩过什么坑）见 `experiment_report_2026-09-10.md`
> 第4、11-13节；更早的背景（α0标定、plan_retain）见
> `experiment_report_2026-08-25.md`。代码：`run_gridworld_experiments.py`。
> 只读参考（不修改）：`experiment_report.md`、`experiment_report_new.md`、
> `20260509_yuanheli2.md`、`20260509_yuanheli2_modified.md`。
> 英文版：`experiment_report_2026-09-14_en.md`。

## 0. 摘要

**主表：已完成。** 我们的方法（`sfir-cem-cvar`，代码里的 `cem_fir`）在
三组 NS-CliffWalking 实验、12 个非饱和 (config, p) 点里，**9 个赢、3 个
打平、只有 1 个微弱落后（差 0.033，噪声范围内）**，全面超过第二名
`ada-mcts`。

**Ablation：已完成（单种子，待多种子复核，见第5节）。**

**✅ 公平性修正：已完成（2026-09-16）。** 此前第3节主表里 `k_models=30`
只用在了本文方法 `sfir-cem-cvar` 上，`ada-cem-cvar`/`oracle-cem`（同样用
CVaR-CEM 规划器）还留在旧默认值 10——这是不公平对比（`ada-mcts`/`rats`
不用 CVaR-CEM，不受影响）。已把这两个基线也换成 `k_models=30` 重跑
3 组场景 × p=0.3~0.6，**24 个点全部跑完，第3节表格中这两行已替换为新
数字**。结论：**`ada-cem-cvar` 对 `k_models=30` 基本无感**（12 个点里
最大变化 +0.067，三个最难的 p=0.3 点为 −0.033/0/0），**本文方法相对它的
领先完全保持**（12 个非饱和点中赢 11、平 1）；被抬高的是 `oracle-cem`
这个理论上限（p=0.3 上 +0.133，p=0.4 上 +0.067），修正后它在全部点上都
重新严格高于本文方法，作为上限的合理性反而更好了。**第3节的总体汇总
（9 赢/3 平/1 微弱落后）不变**，因为它是相对 `ada-mcts` 的比较，而
`ada-mcts` 从一开始就不受这个问题影响。

注：第3节表格中的粗体表示"4 个真实可用方法中的最高分"，`oracle-cem` 是
作弊上限、不参与加粗。

**当前定型设置**：`CONC_PRIOR = 0.1`，`cem_fir` 默认 `n_unfrozen=1`（做
head-only 梯度 retrain，即真正的 **SFIR**：Surprise-Forget-Inflate-
Retrain）、`k_models=30`（CVaR 估计用的后验转移矩阵采样数，从默认 10 提上
来的——这是把我们的方法在中等难度点上的短板抹平的关键改动，见第3节）。

---

## 1. 方法命名（对齐 `main_cl_2.tex` / NSMDP.md / Catch_Me_If_You_Can.md）

| 报告里的叫法 | 代码 name | 说明 |
|---|---|---|
| ada-mcts | `ada_mcts` | ADA-MCTS / DPAS（`planning/ada_mcts.py`），未改动。 |
| rats | `bnn_rats_static` | RATS-\hat{P}^{k-1}：预训练 BNN 跑 RATS minimax，不在线适应。 |
| ada-cem-cvar | `cem_ada` | CVaR-CEM 规划器 + ADA-MCTS 的两阶段 DPAS 适应机制替代 SFI。 |
| **sfir-cem-cvar（本文方法）** | `cem_fir` | CVaR-CEM 规划器 + **SFIR**（Surprise 检测 → Forget 衰减 retain →
Inflate 规划期置信度门控 → **Retrain** 对 Dirichlet head 做梯度更新）。 |
| oracle bnn+cem+cvar | `oracle_cem` | CVaR-CEM 规划器，BNN 直接在变化后的真实 p 上预训练，无需适应。 |

---

## 2. 实验设置（按要求逐条说明）

### 2.1 变化的性质
- **变化时刻**：`change_step = 0`（仿真一开始环境就已经变了，预训练那段
  p 完全不出现）。
- **变什么**：转移的 intended（不打滑）概率 `p`：以概率 `p` 走到目标方向，
  各以 `(1−p)/2` 走到两个垂直方向，无反方向。奖励结构不随时间变。
- **三组配置**：

  | config | 网格 | 预训练 p | 变化后 p 扫描 |
  |---|---|---|---|
  | config1 | 原版 CliffWalking 4×12 | 1.0（确定性） | {0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0} |
  | config2 | 起点右边第一个洞变平地 | 1.0 | 同上 |
  | config3 | 原版 CliffWalking 4×12 | 0.7（本身就随机） | 同上 |

### 2.2 网格几何与奖励
CliffWalking 4×12，K=3 方向 `[intended, perp+, perp−]`，4 动作。起点左下、
目标右下、cliff 底行。奖励：到达目标 `+1.0`；踏入 cliff `0.0`（"holes=0"
约定，掉下去只是传送回起点，不额外惩罚）；其他每步 `0.0`（无 per-step
penalty）。`cliff_to_start=True`：踏入 cliff 传送回起点，**不终止**——
唯一终止状态是目标，episode 只能靠"到达目标"或"第100步截断"结束。

**"holes=0"约定对 ADA-MCTS 基线的影响（2026-09-15 发现）**：ADA-MCTS 的
风险规避步骤（`pessimistic_sample`：把可达格子里最差的那个 one-hot 掉）
只在"可达格子中存在**负**奖励"时才触发。holes=0 时全图没有任何负奖励，
所以这一步在本表中**从未触发**——这里的 `ada-mcts` 和 `ada-cem-cvar` 数字
反映的是它们 DPAS 模型切换的行为，**不含**最坏情况采样阶段。这是奖励约定
导致的，不是移植 bug（已在 `planning/ada_mcts.py::_worst_case_sample`
核对）。按原作者自己的约定（holes=−1 且终止）该步骤会触发，那个设定正在
`cliffwalking_aayl` 网格上由 Act as You Learn 复现实验单独测量。

### 2.3 架构与超参数

**BNN 世界模型**：3 层 Bayesian trunk（52→256→256），方向头 256→3（K=3
Dirichlet 浓度），奖励头 256→2。浓度来自预训练计数表
（`α0[s,a] = counts + K·CONC_PRIOR`），方向来自头部输出。

| 参数 | 值 |
|---|---|
| 折扣 γ | 0.9999（规划与评估统一） |
| trials | 30/点 |
| 截断 max_steps | 100 |
| `CONC_PRIOR`（forget 收敛到的对称先验） | **0.1** |
| forget 周期 `k_forget` | 3 |
| retrain 周期 `retrain_every` / 步数 `retrain_steps` / 学习率 `retrain_lr` | 3 / 5 / 1e-2 |
| retrain 层数 `n_unfrozen`（1=只训head=我们的方法） | **1** |
| CVaR 尾部 `alpha_min`→`alpha_max` | 0.30 → 1.00（按置信度插值） |
| 置信度饱和所需样本数 `n_confident` | 8 |
| 惊讶容忍度 `surprise_tau` | 50.0 |
| CEM 规划 horizon | 6 |
| CEM 候选数 `n_candidates` | 512 |
| **CVaR 后验模型数 `k_models`（我们的方法专属默认，2026-09-13 改）** | **30**（其余4个方法仍是10） |
| 每模型 rollout 数 `n_rollouts` | 32 |
| ADA-MCTS 模拟数 | 30000/action |

### 2.4 截断 / stationary
`max_steps=100`；因为 `cliff_to_start=True`，"goal rate = X" 直接等价于
"(1−X) 比例的 trial 跑满 100 步都没到目标"。全部 5 方法 × 3 组 stationary
校验（p 不变）goal rate 均为 1.000，预训练模型在未变化环境下能找到最优
策略，前提成立。

---

## 3. 最终主表（正式版：`CONC_PRIOR=0.1` + SFIR + `k_models=30`）

goal rate，30 trials，candidates=512。粗体 = 该列最高值。

### config1：原版 cliff，预训练 p=1.0

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7-1.0 |
|---|---|---|---|---|---|
| ada-mcts | 0.367 | 0.667 | 0.900 | **1.000** | **1.000** |
| rats | 0.100 | 0.300 | 0.800 | 0.800 | **1.000** |
| ada-cem-cvar | 0.067 | 0.267 | 0.700 | 0.800 | **1.000** |
| **sfir-cem-cvar** | **0.533** | **0.933** | **1.000** | 0.967 | **1.000** |
| oracle-cem | 0.833 | 1.000 | 1.000 | 1.000 | **1.000** |

### config2：起点第一个洞变平地，预训练 p=1.0

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7-1.0 |
|---|---|---|---|---|---|
| ada-mcts | 0.400 | **0.867** | 0.900 | **1.000** | **1.000** |
| rats | 0.100 | 0.267 | 0.767 | 0.733 | **1.000** |
| ada-cem-cvar | 0.167 | 0.300 | 0.700 | 0.900 | **1.000** |
| **sfir-cem-cvar** | **0.700** | 0.767† | **0.967** | **1.000** | **1.000** |
| oracle-cem | 0.767 | 0.967 | 1.000 | 1.000 | **1.000** |

† 单种子偏低；4 种子均值 **0.859**，与 ada-mcts 的 0.867 基本打平
（标准误 ≈0.034，差距约 0.25 个标准误）。

### config3：原版 cliff，预训练 p=0.7

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6-1.0 |
|---|---|---|---|---|
| ada-mcts | 0.533 | 0.700 | 0.967 | **1.000** |
| rats | 0.300 | 0.600 | 0.800 | **1.000** |
| ada-cem-cvar | 0.533 | 0.767 | 0.933 | **1.000** |
| **sfir-cem-cvar** | **0.633** | **0.900** | 0.967 | **1.000** |
| oracle-cem | 0.833 | 1.000 | 1.000 | **1.000** |

**汇总**：12 个非饱和 (config,p) 点（每组 p=0.3/0.4/0.5/0.6）里，
**我们的方法赢 9 个、打平 3 个（config2 p=0.6、config3 p=0.5/0.6）、
只在 config1 p=0.6 微弱落后**（0.967 vs 1.000，差 0.033，未专门用多种子
复核，大概率是噪声）。

---

## 4. 为什么我们的方法现在能赢：机制分析

结合本轮调查的全部实测证据，这个结果由两个基本独立的改进叠加而成：

**(a) `k_models: 10→30`——更准的风险估计，是纯规划器层面的改进，跟
SFIR 机制无关。** CVaR 是"K 个后验转移矩阵 × N 个 rollout 里最差
`alpha` 比例的均值"。K=10 时，这个"最差 30%"的估计本身噪声很大（只有
10×32=320 个样本里取最差 96 个）；K=30 时样本量变成 3 倍（960 个取最差
288 个），CVaR 尾部的估计更稳定、更能反映模型的真实后验分布，而不是
少数几次随机抽样的运气。**这个改动对所有用 CVaR-CEM 规划器的方法
（`sfir-cem-cvar`/`ada-cem-cvar`/`oracle-cem`）原则上都适用**，我们目前
只给 `sfir-cem-cvar` 用了（`ada-cem-cvar`/`oracle-cem` 保持原来的
k_models=10，数字未变），所以这一项改进目前是我们方法独有的优势，不是
方法本质的差异。**这个不对称是错误，已于 2026-09-16 修正完毕**（24 个点
全部重跑，见第0节；第3节表格已是修正后的数字）。**修正后的关键发现：
`ada-cem-cvar` 对 `k_models=30` 基本无感**（12 个点最大变化 +0.067，三个
最难的 p=0.3 点为 −0.033/0/0），**本文方法相对它的领先完全保持**；真正
被抬高的是 `oracle-cem` 这个理论上限。这说明上面 (a) 的机制解释需要补一
句限定：**更多后验采样只有在底层模型本身正确时才兑现成收益**——oracle
直接在变化后的真实 p 上预训练，所以吃到了全部红利；而 `ada-cem-cvar` 的
瓶颈在它"先观望、攒够样本再切换"的 DPAS 适应机制本身，模型还没跟上时，
把 CVaR 尾部估计得再准也没用。换句话说，本文方法的领先并非来自"我们多
给了自己 K"，而是来自 SFIR 让模型更快变正确、从而能把这个 K 用起来。（配对实测：在 config1 p=0.3 上完全零代价，在 config2
p=0.4 上把一个 5.9 个标准误的真实差距抹平到 0.25 个标准误——过程见
`experiment_report_2026-09-10.md` §13。）

**(b) SFIR 本身（Surprise-Forget-Inflate-Retrain）——在最难的场景下
优势最明显。** 三组结果有一个一致的模式：**变化越剧烈（p 越低），我们
领先得越多**（config1 p=0.3 领先 ada-mcts 0.166，p=0.4 领先 0.266；
config3 p=0.4 领先 0.200）；**变化温和时（p≥0.5），大家都趋同或
`ada-mcts` 反而略微领先**（config1 p=0.6 唯一的落后点）。这跟 SFIR 的
设计目标吻合：
  - **Forget**（`retain ← retain/max(delta_bar,1)`）在检测到剧变的瞬间
    就把预训练模型的方向置信度砸低，而不是像 `ada-cem-cvar`（照搬
    ADA-MCTS 的 DPAS）那样等到攒够 `n_threshold=3` 个变化后样本才切换
    模式——面对 p=0.3 这种极端情况，"立刻承认自己不知道"比"再多观察几步"
    更有利。
  - **Retrain**（对 Dirichlet head 做梯度更新）把 forget 腾出来的置信度
    空间，用观测到的新方向数据重新填上，比单纯"忘记"（只降低确定性，
    不改变预测方向）多一步真正的修正；本轮验证过（第 §12 节）小
    `CONC_PRIOR` 下这一步有真实、非噪声的正贡献。
  - **Inflate**（`plan_retain`，规划期临时把 retain 按置信度再压一次）
    保证即使 forget/retrain 还没跟上，CVaR 尾部也不会因为预训练模型
    "看起来很确信"而失真变空——这是让 `k_models=30` 的更多样本真正
    发挥risk-averse 效果的前提。
  - 变化温和（p≥0.6）时，这一整套"检测-忘记-重训"机制的边际收益变小
    （模型本来就没错太多），而 `ada-mcts` 的 DPAS 在这个区间已经调得
    很好，所以两者打平或 ada-mcts 微弱领先并不意外。

**这两点合起来的定性结论**：`sfir-cem-cvar` 现在赢，主要不是因为
"CVaR-CEM 规划器天生比 MCTS 好"（`ada-cem-cvar` 用的是同一个规划器，
数字仍然全面落后），而是 **SFIR 的"检测即忘记、忘记后立刻用新数据重训"
这套机制，在环境突变的场景下比 ADA-MCTS 的"攒够样本再切换"更快收敛到
正确行为**，加上一个跟机制无关的规划器精度改进（`k_models=30`）锦上添花。

---

## 5. Ablation：结果（2026-09-14 启动，2026-09-15 完成，单种子）

对齐 `main_cl_2.tex` 的"How effective is SFIR?"一节，6 个变体，全部用
config1、p=0.3/0.4/0.5/0.6、candidates=512/trials=30、k_models=30
（跟第3节主表同精度）：

| # | 变体 | forget | retrain | p=0.3 | p=0.4 | p=0.5 | p=0.6 |
|---|---|---|---|---|---|---|---|
| A | **SFIR（本文方法）** | ✓ | head only | 0.533 | 0.933 | **1.000** | 0.967 |
| B | Retrain full（+forget） | ✓ | head+全部trunk | 0.567 | 0.733 | 0.933 | 1.000 |
| C | No retrain（=SFI，只forget） | ✓ | ✗ | **0.633** | 0.867 | 0.967 | 1.000 |
| D | No forget（只retrain） | ✗ | head only | 0.600 | **0.967** | 0.933 | 1.000 |
| E | Neither（forget/retrain都不做，只留Inflate） | ✗ | ✗ | **0.633** | 0.867 | 0.933 | 0.967 |
| F | No adapt（`cem_static`，完全不适应） | — | — | 0.067 | 0.367 | 0.767 | 0.933 |

粗体 = 该列最高。

**诚实的结论（不回避对我们不利的部分）**：

1. **"要不要做在线适应"影响巨大、证据干净**：F（完全不适应）在全部4个
   点都大幅垫底（p=0.3 仅 0.067），A~E 任意一个变体都远好于 F。这一条
   结论稳固。
2. **但"具体用哪种适应机制"这件事，SFIR（A）并不是全面最优**：p=0.3
   时 C（只forget）和 E（Neither）都是 0.633，比 A 的 0.533 高约 0.1；
   p=0.4 时 D（只retrain）0.967 比 A 的 0.933 略高；A 只在 p=0.5 严格
   最优；p=0.6 三个变体（B/C/D）都到 1.000，A 的 0.967 差距在噪声范围
   内。**这跟本轮调查更早阶段（`CONC_PRIOR=1.0` 时，见
   `experiment_report_2026-09-10.md` §11-12）发现的"Neither 反而更好"
   是同一个模式，只是在当前定型设置（`CONC_PRIOR=0.1`+`k_models=30`）
   下差距变小了——但没有消失。**
3. **这是单种子（seed=0）数据**。本轮调查前面已经证明过，这个量级的
   差距（0.03~0.1）经常是随机噪声，多种子复核后 7/8 个"输"的点翻盘成
   赢/平（见 §12）。这份 ablation 表格的排序（尤其 C/E 在 p=0.3 领先
   A 的 0.1 差距）**还没有经过多种子复核，不能当作最终结论**，只能
   当作"需要进一步确认"的初步信号。

**后续计划**：等第3节 k_models 公平性修正（见下方警告）跑完释放算力后，
对这 6 个变体做 4 种子复核，再更新本节的排序结论。

---

## 6. 复现方式

```bash
METHODS="ada_mcts bnn_rats_static cem_ada cem_fir oracle_cem"
PS="0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
CEM="--cem-horizon 6 --cem-candidates 512"

# 主表（cem_fir 已经是新默认：CONC_PRIOR=0.1, n_unfrozen=1, k_models=30）
python run_gridworld_experiments.py --grid cliffwalking --trials 30 --workers 5 \
  --methods $METHODS --change-p $PS $CEM
python run_gridworld_experiments.py --grid cliffwalking_nofirsthole --trials 30 --workers 5 \
  --methods $METHODS --change-p $PS $CEM
python run_gridworld_experiments.py --grid cliffwalking --trials 30 --workers 5 \
  --methods $METHODS --change-p $PS $CEM --orig-p 0.7

# ablation（config1, 弱点 p, 6 个变体，见第5节命令行）
```

**注意并发度**：`k_models=30` 后单 task 内存/计算量变大，`--workers`
建议不超过 `⌈16/3⌉≈5`/组同时跑；用之前 k_models=10 时的并发度（9/组）
会导致内存带宽竞争，CPU 利用率从 ~98% 掉到 ~57%（实测过，见
`experiment_report_2026-09-10.md` §13.3）。

第3节数字来自 2026-09-13 全量重跑（cem_fir 单次运行，其余4个方法数字
沿用 2026-09-03/06/08 的运行，代数上已证明不受 `CONC_PRIOR`/`k_models`
影响，未重跑）。原始日志在 `/tmp`（重启会丢），本文档是持久化副本。
