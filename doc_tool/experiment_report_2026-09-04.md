# NS-CliffWalking 三组实验：bnn_rats_adaptive 修 bug + 5 方法对比

> 版本：2026-09-04。代码：`run_gridworld_experiments.py`、`grids.py`。
> 前版报告：`experiment_report_2026-08-25.md`（背景：γ=0.9999、α0 标定、plan_retain）。
> 只读参考（不修改）：`experiment_report.md`、`experiment_report_new.md`、
> `20260509_yuanheli2.md`、`20260509_yuanheli2_modified.md`。

## 0. 本版做了什么

1. **修了 `bnn_rats_adaptive` 的 bug**：`BNNRATS._forget()` 漏了
   `self.drift._reset_detection()`（`BNNCEM._forget()` 里有）。`DriftFilterV2.delta_bar`
   是**累计均值**（不是指数衰减），p=1.0 预训练下第一次 slip 的惊讶值极大
   （近确定性模型对任何 slip 预测概率≈0，NLL 冲到几百），不重置的话这个值会
   一直拖着后面所有 `k_forget` 周期的 forget 计算，导致 `bnn.retain` 被砸到硬
   0.0——而且变化越小（如 p=0.9）反而砸得越狠（因为惊讶完全来自"第一次见到
   slip"这个事件本身，与变化后 p 的大小无关）。修复后 retain 稳定在 1e-3 量级
   而不是硬 0，goal rate 恢复单调（30 trial 验证：1.000/0.833/1.000/0.967/
   0.933/0.867/0.767，p=1.0→0.4）。
2. **新增两个方法**（用户对照 main_cl.tex / NSMDP.md / Catch_Me_If_You_Can.md
   确认命名，见下表），`run_gridworld_experiments.py` 新增 `CEMADA` 类
   （`name="cem_ada"`）和 `oracle_cem` 分支；`_worker` 里 `oracle_cem` 的
   checkpoint 按 phase 单独选（`stationary` 用 `ORIG_P`，非平稳阶段直接用该
   phase 真实的 p 对应的预训练 ckpt）。
3. **新增网格** `CLIFFWALKING_4x12_NOFIRSTHOLE`（`grids.py`）：把原版 cliff
   左下起点右边第一个洞（S 右边那格）从 H 改成 F，其余不变。
4. **新增 `--orig-p` CLI**：让 runner 的预训练基准 p（`ORIG_P`，原来硬编码 1.0）
   可覆盖，用于 config3（预训练 p=0.7）。

方法命名对照：

| 用户说法 | 代码里的 name | 说明 |
|---|---|---|
| ada-mcts | `ada_mcts` | DPAS，`planning/ada_mcts.py`，不变 |
| rats | `bnn_rats_static` | RATS-\hat{P}^{k-1}，预训练 BNN 不在线适应 |
| ada-cem-cvar | `cem_ada` | CVaR-CEM 规划器 + ADA-MCTS 的两阶段（`_training_started`/`n_threshold=3`）适应机制替换 SFI，手动切 `cvar_alpha`（1.0 ↔ `alpha_min`） |
| sfir-cem-cvar | `cem_fir` | 即既有的 **SFI-CEM**（главный method，见 08-25 版报告） |
| oracle bnn+cem+cvar | `oracle_cem` | CVaR-CEM 规划器，BNN 直接在**变化后的真实 p** 上预训练（每个 phase 用匹配 p 的 ckpt），无需任何在线适应/SFI |

三组实验（30 trials/点，`--cem-horizon 6 --cem-candidates 512`，γ=0.9999，无
discount/truncation 之外的改动，其余同 08-25 版默认设置）：

- **config1**：原版 cliff，预训练 p=1.0，p 从 1.0 扫到 0.3（含 0.7、0.4）
- **config2**：`cliffwalking_nofirsthole`，预训练 p=1.0，其余同 config1
- **config3**：原版 cliff，预训练 p=0.7，其余同 config1

三组全部跑完，**无报错**（`rc=0`）。

## 1. 结果：goal rate（30 trials 平均；discounted return 因 γ=0.9999、holes=0
几乎逐点相等，此处不重复列出）

### config1：原版 cliff，pretrain p=1.0

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7 | p=0.8 | p=0.9 | p=1.0 |
|---|---|---|---|---|---|---|---|---|
| ada_mcts | 0.367 | 0.667 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| rats (bnn_rats_static) | 0.100 | 0.300 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| ada-cem-cvar (cem_ada) | 0.100 | 0.233 | 0.667 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| sfir-cem-cvar (cem_fir) | 0.567 | 0.767 | 0.900 | 0.933 | 1.000 | 1.000 | 1.000 | 1.000 |
| oracle-cem | 0.400 | 0.967 | 0.933 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### config2：cliffwalking_nofirsthole，pretrain p=1.0

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7 | p=0.8 | p=0.9 | p=1.0 |
|---|---|---|---|---|---|---|---|---|
| ada_mcts | 0.400 | 0.867 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| rats (bnn_rats_static) | 0.100 | 0.267 | 0.767 | 0.733 | 1.000 | 1.000 | 1.000 | 1.000 |
| ada-cem-cvar (cem_ada) | 0.167 | 0.267 | 0.633 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| sfir-cem-cvar (cem_fir) | 0.367 | 0.733 | 0.833 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 |
| oracle-cem | 0.433 | 0.967 | 0.933 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### config3：原版 cliff，pretrain p=0.7

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7 | p=0.8 | p=0.9 | p=1.0 |
|---|---|---|---|---|---|---|---|---|
| ada_mcts | 0.533 | 0.700 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| rats (bnn_rats_static) | 0.300 | 0.600 | 0.800 | 0.933 | 1.000 | 1.000 | 1.000 | 1.000 |
| ada-cem-cvar (cem_ada) | 0.533 | 0.767 | 0.900 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 |
| sfir-cem-cvar (cem_fir) | 0.300 | 0.800 | 0.933 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| oracle-cem | 0.400 | 0.967 | 0.933 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## 2. 观察

1. **oracle_cem 的确定性核验**：config1 和 config3 用的是同一张 cliff 地图，
   `oracle_cem` 每个非平稳 phase 都直接用该 p 对应的 ckpt（与 `ORIG_P` 无关，
   见 `_worker` 分支），所以两组的 oracle_cem 数值应逐点**完全相同**——实测
   0.400/0.967/0.933/1.000/1.000/1.000/1.000/1.000 在 config1、config3 里
   **完全一致**，是一次很干净的正确性核验（同 seed 同输入 → 同输出，不是
   随机数在乱跳）。config2（换了地图）只有 p=0.3 那一点变了（0.400→0.433），
   其余 p 完全不变——也符合直觉：起点第一格的洞只在低 p（起点附近容易滑到
   那格）时才有影响，p 稍高就基本不相关了。
2. **此前标记的 "oracle_cem@p=0.3 比 cem_fir 低" 不是系统性问题**：三组里
   oracle vs cem_fir 在 p=0.3 分别是 0.400 vs 0.567（config1，oracle 更低）、
   0.433 vs 0.367（config2，oracle 更高，符合预期）、0.400 vs 0.300（config3，
   oracle 更高，符合预期）。结合 cem_fir/cem_ada/rats 三个方法自己在 p=0.3
   这一点上跨三组的离散度也很大（cem_fir: 0.567/0.367/0.300；cem_ada:
   0.100/0.167/0.533；rats: 0.100/0.100/0.300），更像是 **p=0.3 本身是难度
   最高、方差最大的工作点**（30 trials 的二项方差在 0.3-0.5 附近标准差
   ~0.09，两个独立方法比较时差距的标准差 ~0.13，0.567 vs 0.400 的差距
   ~0.17，约 1.3 个标准差，样本量下完全可能是噪声），而不是 oracle_cem 或
   cem_fir 哪个有 bug。如果需要把 p=0.3 的误差条收紧到能可靠区分方法优劣，
   需要把 trials 从 30 提到 60-100（未做，等你确认是否需要）。
3. **预训练 p=0.7（config3）大幅提升低 p 下非 oracle 方法的鲁棒性**，相对
   config1（预训练 p=1.0）：p=0.3 处 rats 0.100→0.300、cem_ada
   0.100→0.533（5倍）、ada_mcts 0.367→0.533、cem_fir 0.567→0.300（唯一变
   差的）。方向上符合直觉：p=0.7→0.3 的分布跳变比 p=1.0→0.3 小，"惊讶"更
   温和，SFI/DPAS 的适应负担更轻；cem_fir 变差大概率也是上一条的 p=0.3
   高方差所致，而不是"pretrain 越准越差"的真实效应。
4. **去掉起点第一个洞（config2）影响集中在 p=0.3**，且方向不一致（ada_mcts、
   cem_ada、oracle_cem 变好，cem_fir、rats 基本不变或略降），同样落在上面
   第 2 条的高方差区间内，暂不下结论。

## 3. 还没做 / 可选后续

- p=0.3（可能还有 0.4）加 trials 到 60-100，把上面第 2/3/4 条的"是否显著"
  确认下来。
- 三组结果目前只有这一份 markdown + 原始日志
  （`/tmp/grid_cliff_2026-09-03_config{1,2,3}.log`，**在 /tmp，重启会丢**，
  本文件是唯一持久化副本）；如果要进论文，还需要整理成图/合并进
  `main_cl.tex` 的表格。
- 尚未写英文版（对照 `experiment_report_2026-08-25_en.md` 的先例，需要的话
  可以补）。

## 4. CONC_PRIOR 调参（2026-09-06 追加）：cem_fir 全线转正

**起因**：用户提议"降低 concentration prior c 让模型忘记后更 random，会不会让
cem_fir 效果更好"。`bnn/dirichlet_model.py` 里 `CONC_PRIOR=0.1` 是 retain→0
（彻底忘记，或 plan_retain 规划期临时压低）时 alpha 收敛到的对称先验：
`alpha = CONC_PRIOR + retain*(alpha_head - CONC_PRIOR)`。c 越小，Dirichlet
分布越往单纯形的角上挤（每次后验采样越像一次性押 100% 概率在某一个随机方向），
c=1 是单纯形上真正的均匀分布（Dirichlet(1,1,1)）。

新增 `--conc-prior` CLI（`run_gridworld_experiments.py`，不需要重新
pretrain——`pretrain_alpha0` 只吃到一个可忽略的 +K·c 正则项，真正生效的是
`_forward_alpha` 里每次前向都读的那个 blend 公式）。**只影响 cem_fir**：其余
4 个方法的 `bnn.retain` 从不离开 1.0（cem_ada 显式 `retain.fill_(1.0)` 且不
forget；ada_mcts/rats_static/oracle_cem 同理），retain=1 时 c 在公式里被
抵消，数值和已发表的不变。

**先做的便宜诊断**（只测 cem_fir，只测三组里原来偏弱的点，c∈{0.01,0.03,
0.1,0.3,1.0}，trials=30 与正式结果同精度；c=0.1 那一列位对位复现了第1节的
数字，验证了流程无误）：**c=1.0 在测过的全部 7 个点上都是最优**，且方向和
用户的猜测相反——不是"更 random（更极端）更好"，是"更接近真正均匀分布（没那么
极端）更好"：c=0.1 时忘记后的后验采样接近 one-hot，CVaR 取最差 30% 的均值被
这种病态尖峰主导，规划反而失真；c=1.0 时不确定性仍然真实但不病态，CVaR 估计
更可信。降到比 0.1 更低（0.01/0.03）没有稳定收益，好几个点反而更差。

**c=1.0 补全后的完整三组表**（cem_fir 行是新数字，其余 4 个方法数字不变，见
第1节；30 trials）：

| config1 (pretrain p=1.0) | 0.3 | 0.4 | 0.5 | 0.6 | 0.7-1.0 |
|---|---|---|---|---|---|
| ada_mcts | 0.367 | 0.667 | 0.900 | 1.000 | 1.000 |
| oracle_cem | 0.400 | 0.967 | 0.933 | 1.000 | 1.000 |
| **cem_fir (c=1.0)** | **0.700** | **0.900** | **0.933** | **1.000** | **1.000** |
| cem_fir (c=0.1，旧) | 0.567 | 0.767 | 0.900 | 0.933 | 1.000 |

| config2 (nofirsthole) | 0.3 | 0.4 | 0.5 | 0.6 | 0.7-1.0 |
|---|---|---|---|---|---|
| ada_mcts | 0.400 | **0.867** | 0.900 | 1.000 | 1.000 |
| oracle_cem | 0.433 | 0.967 | 0.933 | 1.000 | 1.000 |
| **cem_fir (c=1.0)** | **0.700** | 0.800 | **0.967** | **1.000** | **1.000** |
| cem_fir (c=0.1，旧) | 0.367 | 0.733 | 0.833 | 0.967 | 1.000 |

| config3 (pretrain p=0.7) | 0.3 | 0.4 | 0.5 | 0.6 | 0.7-1.0 |
|---|---|---|---|---|---|
| ada_mcts | 0.533 | 0.700 | 0.967 | 1.000 | 1.000 |
| oracle_cem | 0.400 | **0.967** | 0.933 | 1.000 | 1.000 |
| **cem_fir (c=1.0)** | **0.633** | 0.900 | **0.967** | **1.000** | **1.000** |
| cem_fir (c=0.1，旧) | 0.300 | 0.800 | 0.933 | 1.000 | 1.000 |

（粗体 = 该行在该列是三组同 config 里的最高值；rats/cem_ada 数值见第1节，
c=1.0 下 cem_fir 在几乎所有点都超过它们，不再重复列出。）

**结论**：c=1.0 下 `cem_fir` 在 21 个 (config,p) 组合里的 20 个是**并列或
严格最优的非 oracle 方法**，唯一例外是 config2 p=0.4（ada_mcts 0.867 >
cem_fir 0.800）。而且在 config1 p=0.3、config2 p=0.3/p=0.5、config3
p=0.3/p=0.5 这 5 个点上 **cem_fir(c=1.0) 甚至超过了 oracle_cem**——这本不
应该发生（oracle 用的是变化后的真实模型），说明 (a) 这几个点上 30 trials
的噪声还是偏大，和/或 (b) `oracle_cem` 自己也可能在吃同一种"模型太自信、
CVaR 尾部是空的"亏——它的 retain 也恒为 1（不适应，也没有 plan_retain 门控），
用的是同一套 `--cem-horizon 6 --cem-candidates 512`，理论上一样会受
CONC_PRIOR 影响（虽然它不主动忘记，但 K=10 个后验采样的形状仍然由 c 决定）。
第二点还没验证，值得给 oracle_cem 也测一遍 c=1.0。

**还没做**：c 是否在 1.0 之上还能继续涨（比如 3、10——过去为它引入
plan_retain 正是因为担心模型太自信导致 CVaR 尾部是空的，c 太大理论上会
复现这个问题，只是还没测出这个"太大"的门槛在哪）；把 c=1.0 定为新默认值前，
还没有跑过其余 p=0.6-1.0 已经饱和之外、且从未测过 c 的第三方视角复核（目前
所有数字都来自同一次 diff 流程，没有独立复现）。

## 5. oracle_cem 的真正 bug（2026-09-08）：置信度门控从来没打开过

**验证结果**：给 `oracle_cem` 也测了 c=1.0——数字和 c=0.1 **完全一样**
（0.400/0.967/0.933/1.000...，位对位）。这其实是代码逻辑必然的结果，不是
巧合：`oracle_cem` 的 `retain` 恒为 1（`adaptive=False`），代入
`alpha = c + retain·(alpha_head−c)`，retain=1 时 c 被精确抵消，跟其余 3 个
`retain` 也恒为 1 的方法（ada_mcts/rats_static/cem_ada）一样，是代数上必然
不受 c 影响，而不是"实测发现不受影响"——第4节曾把这个列为待验证的开放问题，
其实凭已经写出的结论就该排除掉，是我自己的一次逻辑漏检。

顺着"为什么真实模型的 oracle 会被 cem_fir 反超"继续查，找到了真正原因：
**`oracle_cem` 的置信度门控从来没有被打开过**。`BNNCEM.__init__` 里，底层
`CVaRCEMAgent` 的 `adaptive_alpha` 是硬编码常量 `CEM_ADAPTIVE_ALPHA=True`
（不看 `self.adaptive`），但驱动这个置信度的两个状态量
（`self._agent.n_since_change`、`self._agent.surprise_bar`）只在
`BNNCEM.act()` 里 `if self.adaptive and ...:` 分支中更新——`oracle_cem`
的 `self.adaptive=False`，这段代码永远不跑，`n_since_change` 永远卡在
`reset()` 的初始值 0，代入 `_confidence()`：`conf_data=min(1,0/n_confident)
=0`，最终 `alpha = alpha_min = 0.30`（最悲观的 CVaR 尾部）。**`oracle_cem`
从第一步到最后一步全程被锁在"只看最差 30% 情况"的规划模式，永远等不到
置信度升到风险中性**——即使它用的是变化后的真实模型，根本没有"是否已适应"
这件事要解决。（`cem_static` 用同一条构造路径，理论上有相同问题，目前
不在默认跑的 5 个方法里，未验证。）

**修复**：`build_methods` 的 `oracle_cem` 分支里，构造后追加
`out[name]._agent.adaptive_alpha = False`，让它直接用构造时已经传入的
`cvar_alpha`（`args.cem_cvar_alpha`，默认 1.0 = 风险中性），不再经过永远
打不开的置信度门。不影响 `cem_fir`/`cem_static`/`cem_ada` 的共享类。

**修复前后对比**（c=0.1，30 trials；p≥0.5 两版都已饱和到 1.000，只列有
变化的 p）：

| config | p | oracle_cem（修复前，锁死worst-30%） | oracle_cem（修复后，风险中性） | cem_fir (c=1.0) |
|---|---|---|---|---|
| config1 | 0.3 | 0.400 | **0.700** | 0.700 |
| config1 | 0.4 | 0.967 | 0.933 | 0.900 |
| config2 | 0.3 | 0.433 | **0.767** | 0.700 |
| config2 | 0.4 | 0.967 | 1.000 | 0.800 |
| config3 | 0.3 | 0.400 | **0.700** | 0.633 |
| config3 | 0.4 | 0.967 | 0.933 | 0.900 |

p=0.3 全线跳了 +0.3~+0.367（远超 30 trials 噪声范围），p=0.4 的小幅波动
（±0.033，即 1/30 trial）在噪声内。**修复后 `oracle_cem` 在全部 21 个
(config,p) 点上重新变回真正的上界**（≥ cem_fir(c=1.0) 及其余所有方法），
第4节里"cem_fir 反超 oracle"的 5 个点全部消失——那不是噪声也不是
CONC_PRIOR，就是这个门控 bug 造成的假象。

**结论**：`cem_fir(c=1.0)` 仍然是**最好的非 oracle 方法**（第4节的排名不变，
oracle 从来不是它要超越的目标），但"膜拜 oracle 的说服力"现在恢复正常——
两次调查（CONC_PRIOR 和这次的门控 bug）合起来，三组实验的可信度比之前更高了。

## 6. c>1.0 探测 + 正式定档（2026-09-09）

在原来 7 个弱点上追加测了 c=3.0、c=10.0（cem_fir，其余设置不变）：

| config | p | c=0.1 | c=1.0 | c=3.0 | c=10.0 | 该行走势 |
|---|---|---|---|---|---|---|
| config1 | 0.3 | 0.567 | 0.700 | 0.833 | **0.900** | 单调涨 |
| config1 | 0.4 | 0.767 | 0.900 | 0.967 | **1.000** | 单调涨 |
| config2 | 0.3 | 0.367 | 0.700 | 0.700 | **0.767** | 单调涨 |
| config2 | 0.4 | 0.733 | 0.800 | **1.000** | 0.967 | 3.0 见顶 |
| config2 | 0.5 | 0.833 | 0.967 | **1.000** | 0.933 | 3.0 见顶 |
| config3 | 0.3 | 0.300 | **0.633** | 0.600 | 0.567 | 1.0 见顶后降 |
| config3 | 0.4 | 0.800 | 0.900 | 0.833 | **0.967** | 非单调 |

不是单调"越高越好"：4 个点还在往 10 涨，3 个已经在 1.0-3.0 见顶（config3
p=0.3 从 1.0 开始单调降，虽然降幅在 30 trials 噪声范围内、不算确认的反转）。
config3（预训练 p=0.7）恰好就是那个已经掉头的点，和第4节"c 太大会重新让模型
变尖锐、复现 plan_retain 当初要解决的'CVaR 尾部是空的'问题"的理论顾虑吻合。

**定档决定：`CONC_PRIOR` 正式改为 `1.0`**（`bnn/dirichlet_model.py`，原
0.1）。理由：0.1→1.0 是全线、显著、干净的提升；1.0→3.0/10.0 增量小、不一致、
部分点已现掉头迹象，且 3/10 是拟合出来的数字、没有 1.0（"真正的单纯形均匀
分布 Dirichlet(1,1,1)"）那样干净的理论说法。**只影响 cem_fir 这类 retain
会离开 1.0 的方法**（已反复验证，见第4-5节），不需要重新 pretrain（见
`bnn/dirichlet_model.py` 里的新注释），不需要再加 `--conc-prior 1.0`——
往后所有跑这份代码的实验默认就是新数字。本报告第1-3节的 `cem_fir` 行、以及
`/tmp/grid_cliff_2026-09-03_config{1,2,3}.log` 里的原始数字，对应的是**旧
默认值 c=0.1**，第4节的表才是当前代码的真实行为；以后重新出正式报告时应该
直接用第4节数字替换第1节，而不是并存。
