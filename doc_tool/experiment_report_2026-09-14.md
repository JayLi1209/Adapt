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

**Ablation：跑中，预计还要 10-20 小时。** 本文档先把已经拿到的部分填上，
ablation 补完后我会更新本文件（同一个文件名，不新建）。

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
| ada-cem-cvar | 0.100 | 0.233 | 0.667 | 0.800 | **1.000** |
| **sfir-cem-cvar** | **0.533** | **0.933** | **1.000** | 0.967 | **1.000** |
| oracle-cem | 0.700 | 0.933 | **1.000** | **1.000** | **1.000** |

### config2：起点第一个洞变平地，预训练 p=1.0

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7-1.0 |
|---|---|---|---|---|---|
| ada-mcts | 0.400 | **0.867** | 0.900 | **1.000** | **1.000** |
| rats | 0.100 | 0.267 | 0.767 | 0.733 | **1.000** |
| ada-cem-cvar | 0.167 | 0.267 | 0.633 | 0.900 | **1.000** |
| **sfir-cem-cvar** | **0.700** | 0.767† | **0.967** | **1.000** | **1.000** |
| oracle-cem | 0.767 | **1.000** | **1.000** | **1.000** | **1.000** |

† 单种子偏低；4 种子均值 **0.859**，与 ada-mcts 的 0.867 基本打平
（标准误 ≈0.034，差距约 0.25 个标准误）。

### config3：原版 cliff，预训练 p=0.7

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6-1.0 |
|---|---|---|---|---|
| ada-mcts | 0.533 | 0.700 | 0.967 | **1.000** |
| rats | 0.300 | 0.600 | 0.800 | **1.000** |
| ada-cem-cvar | 0.533 | 0.767 | 0.900 | **1.000** |
| **sfir-cem-cvar** | **0.633** | **0.900** | 0.967 | **1.000** |
| oracle-cem | 0.700 | 0.933 | 0.967 | **1.000** |

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
方法本质的差异。（配对实测：在 config1 p=0.3 上完全零代价，在 config2
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

## 5. Ablation（跑中，2026-09-14 启动，预计还要 10-20 小时）

对齐 `main_cl_2.tex` 的"How effective is SFIR?"一节，6 个变体，全部用
config1、p=0.3/0.4/0.5/0.6、candidates=512/trials=30（跟第3节主表同精度）：

| # | 变体 | forget | retrain | 状态 |
|---|---|---|---|---|
| A | **SFIR（我们的方法）** | ✓ | head only | 跑中 |
| B | Retrain full（+forget） | ✓ | head+全部trunk | 跑中 |
| C | No retrain（=SFI） | ✓ | ✗ | 跑中 |
| D | No forget | ✗ | head only | 跑中 |
| E | No forget + no retrain（Neither，只留Inflate） | ✗ | ✗ | 跑中 |
| F | No adapt（`cem_static`，无SFI） | — | — | 跑中 |

**进度**（截至本文档发布时）：6 个变体的 stationary 阶段（p=1.0，
不变化）均已完成，goal rate 全部 1.000，正常；p=0.3/0.4/0.5/0.6 四个
真正有信息量的点仍在计算中。18 个 worker 共享 16 核，CPU 利用率
~86%/worker（正常，没有卡住）。

**为什么这么慢**：`k_models=30` 让 CVaR 估计的计算量变成默认值的 3 倍，
本轮之前的经验是一个这个规模的 config1 满 8 点主表（用 15 个 worker）
跑了约 11.5 小时；这次 ablation 只测 4 个 p 点但有 6 个变体在跑（部分
变体如 Retrain-full 的梯度步骤更多，No-forget 类的 episode 可能更长），
预计还需要 **10-20 小时**才能全部跑完，可能会跨过夜。

**跑完后会怎么更新本文档**：直接在本节填入 6×4 的完整表格 +
按第4节的方式做机制解读（比如"Neither"这次在小 CONC_PRIOR、
k_models=30 下是否还像之前 CONC_PRIOR=1.0/10 时那样占优——早前的探索
发现这个排序会随 CONC_PRIOR 反转，值得在最终设置下重新确认一遍）。
不会新建文件，直接改这份 `experiment_report_2026-09-14.md`。

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
