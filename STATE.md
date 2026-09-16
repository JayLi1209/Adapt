# STATE.md — NS-CliffWalking / SFIR-CEM-CVaR 工作状态

> 持续维护的工作记忆文档，不是最终报告。每次有实质性进展/决定/发现，
> 应该回来更新这个文件（同一个文件，不新建）。最终对外报告见
> `doc_tool/experiment_report_2026-09-14.md`（详细版）和
> `doc_tool/experiment_summary_2026-09-14.md`（自包含、给外部演讲用）。
> 最后更新：2026-09-15 20:55 EDT。

**本文件即为 /clear 或 /compact 前的交接文档**（用户要求：如果 STATE.md
能当交接文档用，就不用另写）。新 session 接手时，建议顺序：
1. 先看这个文件的"正在跑什么"和"下一步"，知道现在卡在哪一步。
2. 用"关键文件与命令"里的资源检查命令看后台实验是不是还活着、跑到哪了。
3. 需要更完整历史时才去读 `doc_tool/experiment_report_2026-09-10.md`
   （调查全过程）或 git log（`git log --oneline` 看本轮提交序列，最新
   一条是 `887193f`，工作分支 `steven202_20260713_v2`，工作目录
   `/home/guo/Adapt`）。
4. 本文件写作时所有代码改动都已提交（`git status` 干净，无未提交的
   `.py` 改动）；未提交的只有实验产出的 checkpoint/data 目录（不用管）。
   本文件本身和两份对外文档已经 commit 到 `887193f`。
5. **已知遗漏**：这次的 ablation 结果和公平性警告只写进了中文版
   `experiment_report_2026-09-14.md`，**英文版 `_en.md` 还没同步更新**
   ——如果需要英文版，记得先同步这两部分内容。

---

## 一句话主题

让本文方法 `sfir-cem-cvar`（代码里叫 `cem_fir`）在 NS-CliffWalking
（环境突变的悬崖行走网格世界）上，在**公平对比**的前提下成为表现最好
的方法，只允许用"调参数"这种最小改动，不允许改定义/作弊。

---

## 已定的决定（用户明确拍板，不要再反复横跳）

1. **方法名以论文为准：SFIR**（Surprise-Forget-Inflate-**Retrain**），
   不是之前发表结果里的 SFI（无 retrain）。Retrain 是对 Dirichlet
   适配头做**真正的梯度更新**，不是别的什么代理机制。
2. **离散头（Dirichlet）用纯 NLL 损失**做 retrain，不用 ELBO。
3. **`CONC_PRIOR`（forget 收敛到的对称先验）定为 0.1**（小、诚实、真
   会去适应），不用更大的 1.0——即使 1.0 在某些点分数更高。原因见下面
   "已被推翻的结论"。
4. **保留诚实 SFIR（有 forget 也有 retrain）作为"本文方法"**，即使
   ablation 显示放弃 retrain 甚至放弃 forget（"Neither"）在某些点分数
   更高。用户明确说："继续用诚实 SFIR（不改，尊重之前的选择）"——不能
   为了刷分改方法定义。
5. **任何为了让本文方法变好而调的参数，必须同步给所有用同一底层组件
   的方法**（重要教训，见下方"规则"）。
6. 实验结果/精度不能因为"加速"而改变；加速只能动并发度、线程数这类
   跟数值无关的旋钮。
7. 4 个 doc_tool 文件只读，不能改：`experiment_report.md`、
   `experiment_report_new.md`、`20260509_yuanheli2.md`、
   `20260509_yuanheli2_modified.md`。

---

## 已经站得住的结论（带数字，当前认为可信）

- **主表（3 组场景 × p=0.3~0.6，12 个非饱和点）**：本文方法
  `sfir-cem-cvar` 相对第二名 `ada-mcts`：**9 赢 / 3 平 / 1 处微弱落后**
  （config1 p=0.6，0.967 vs 1.000，差 0.033，噪声范围内）。
  数字见 `experiment_report_2026-09-14.md` §3（**⚠ 但见下面"未解决问题"
  —— 这批数字里 ada-cem-cvar/oracle-cem 两行的对比目前不公平，正在修正**）。
- **变化越剧烈，本文方法领先越多**：p=0.3/0.4 时领先 ada-mcts 约
  0.17~0.27；p≥0.6 时基本打平甚至微弱落后。跟 SFIR"检测到异常立刻主动
  降低确信度"的设计目标吻合。
- **k_models（CVaR 的后验转移矩阵采样数）从 10 提到 30，对 CVaR 尾部
  估计有实质性、无副作用的改善**：config1 p=0.3 完全零代价（0.650→
  0.650，此为早期探索用另一路径复核的稳定性检查数字），config2 p=0.4
  把一个 5.9 个标准误的真实差距压到 0.25 个标准误（4 种子均值从明显
  落后变成基本打平）。这是纯规划器精度参数，跟 SFIR 机制无关。
- **k_models 提升有效，n_rollouts（每个后验模型下的 rollout 数）提升
  无效甚至有害**：n_rollouts=64 曾把 config1 p=0.3 从 0.650 砸到
  0.300。机制解释：k_models 增加独立的"世界猜测"数量（epistemic，
  CVaR 尾部该反映的不确定性），n_rollouts 只是给同样的猜测做更多次
  重采样（aleatoric），不增加新假设，反而可能让某一个恰好是异常值的
  猜测在尾部占比更大。
- **"要不要做在线适应"这件事影响巨大、证据干净**：ablation 里完全不
  适应（`cem_static`，变体F）在 config1 全部4个点都远远垫底
  （p=0.3 仅 0.067），任何一种适应机制都远好于完全不适应。
- **多种子复现是最有效、零算法改动的"发现噪声"手段**：`--seed` CLI
  加上之后，之前 8 个"中等难度点上输给 ada-mcts"里有 7 个在多种子平均
  后翻盘成赢/平。

---

## 已经被推翻的结论

- ~~`CONC_PRIOR=1.0`（大先验）能让本文方法几乎处处最好~~——**推翻**。
  根因：c 大到一定程度后，forget 把置信度冲刷到均匀分布太快，
  forget/retrain 基本不起作用，真正在起作用的是 CVaR 规划器本身自带的
  置信度门控。证据：一个"既不 forget 也不 retrain"（Neither）的版本
  在 c=1.0 下处处赢了本文方法。**结论**：c=1.0 下的"赢"是"没有真正在
  适应"的假象，不能用。已改回 c=0.1。
- ~~`--retrain-min-conf`（用置信度门控 retrain 触发时机）能帮助本文
  方法~~——**推翻，而且方向是反的**。它在 config2 p=0.4（目标点）从不
  触发，在 config1 p=0.3 反而帮倒忙：剧变时 `delta_bar` 飙升导致置信度
  骤降，门控恰好在最需要 retrain 的时候把它拦住了。
- ~~`delta_bar` 幅度可以用来区分"剧烈变化"和"温和变化"，从而做自适应
  retrain 频率~~——**推翻**。专门测过 config1 p=0.3 和 config2 p=0.4
  的 delta_bar 分布，形状类似（大多数~1，偶尔上千的尖峰），幅度本身
  分不出"严重"和"中等"，这条路线提前放弃，没有浪费更多算力。
- ~~"Neither"在某一次单种子测试里 0.950，看起来完胜~~——**部分推翻**。
  再测 2 个种子后 3 种子均值回落到 0.850，和 ada_mcts 的 0.867 基本
  打平，不是碾压。提醒：n=1 的结果不能直接当结论。
- **本轮又出现的同类现象（尚未被推翻，但也未被多种子确认）**：
  最新 ablation（单种子，2026-09-15）显示，即使在 c=0.1、k_models=30
  的"最终"设置下，"只 forget 不 retrain"（=SFI）和"Neither"在 p=0.3
  仍比完整 SFIR 高约 0.1（0.633 vs 0.533）；"只 retrain 不 forget"在
  p=0.4 也略高（0.967 vs 0.933）。**没有推翻决定 4（继续用诚实
  SFIR），但也没有被多种子复核，目前只是初步信号**，见"下一步"。

---

## 用户的要求/问题（本轮，按时间顺序，供追溯原始意图）

1. 有没有办法让 sfir-cem-cvar 效果最好？比如降低 c（concentration
   prior）？
2. 加速实验，但结果/精度必须完全不变。
3. 确认 `main_cl_2.tex` 更新后 retrain 的定义歧义是否解决。
4. 确认"我们的方法 SFIR+CEM+CVaR 应该有 retrain"。
5. 按论文把离散结果从 SFI 改成真正的 SFIR（真 retrain）；离散头用纯
   NLL。
6. 深挖时明确限定范围：只做 cliffwalking，不要牵扯 frozenlake。
7. 如果探索没有效果，就用小 c 换诚实适应，可以继续调参数提性能。
8. （AskUserQuestion）在"小 c+SFIR"和其它选项间选：选了小 c+SFIR
   （用户承认这个选项当时推荐度低，但坚持要用）。
9. （AskUserQuestion）"Neither"处处赢小c的SFIR，主表怎么定：选了继续
   小c+SFIR。
10. 跑 config1 p=0.4 多种子确认异常是否是 bug；同时尝试更大环境跳变
    （1.0→0.3、0.9→0.3、0.9→0.5 等）看能不能让本文方法赢更多。
11. （Stop-hook 常驻目标）让本文方法成为 performance 最好的方法，最小
    改动（比如调参）。
12. （AskUserQuestion）"放弃 retrain 甚至 forget 能让 config2 p=0.4
    从真实输变成打平，怎么定"：选了继续诚实 SFIR，不改定义。
13. GPU 利用率低，能不能提高（不能牺牲精度/质量）？
14. 把 ablation 也做一下；做完新建文档把主表+ablation放进去。
15. 报告剩余实验时间；报告"我们的方法现在超过其他方法"的数据/实验
    设置/原因。
16. 解释清楚为什么 k_models 改 30 有用、n_rollouts 改了没用；列出其它
    全部参数（含但不限于 CONC_PRIOR）。
17. 写一份精简报告：只要实验配置、数据集、方法区别，分析简洁。
18. 指出精简报告里"主表用30、其他方法用10"是不公平对比——**这是一个
    真实的方法学错误，已承认并在修正**（见下方"未解决问题"）。
19. 后续问题一律用中文回答；赶快测多种子/公平对比看本文方法是不是真
    的最好；以后确保这种"参数不一致"的低级错误不再犯。
20. 新建本文件 `STATE.md`，持续记录状态（本次请求）。

---

## 实验设置

**环境**：CliffWalking 4×12，K=3 方向（intended / 垂直+ / 垂直−）滑动
模型，转移概率 `p`（intended 方向概率）。奖励：到达目标 +1，其余
（含踩悬崖）0；踩悬崖传送回起点、不终止 episode；唯一终止条件是到达
目标或 100 步截断。

**3 组场景**：

| config | 网格改动 | 预训练 p | 测试 p 扫描 |
|---|---|---|---|
| config1 | 无（原版） | 1.0 | 0.3~1.0（非饱和段 0.3-0.6） |
| config2 | 起点右边第一格悬崖→平地 | 1.0 | 同上 |
| config3 | 无（原版） | 0.7（本身随机） | 同上 |

**5 个方法**：`ada_mcts`（MCTS+DPAS 两阶段适应）、`bnn_rats_static`
（RATS minimax，不适应）、`cem_ada`（CVaR-CEM + DPAS）、`cem_fir`（
CVaR-CEM + **SFIR**，本文方法）、`oracle_cem`（CVaR-CEM，直接在真实
突变后 p 上预训练，理论上限，不是真实可用方法）。

**关键超参数**（当前定型值）：
| 参数 | 值 |
|---|---|
| γ | 0.9999（规划/评估统一，无额外 discount/truncation） |
| trials/点 | 30 |
| max_steps | 100 |
| CEM horizon / candidates | 6 / 512 |
| `k_models`（CVaR 后验模型数） | `cem_fir`=30；其余原本=10（**正在修正统一为30，见下**） |
| `n_rollouts` | 32 |
| `CONC_PRIOR` | 0.1 |
| `k_forget` | 3 |
| retrain：层数/周期/步数/学习率 | head only(`n_unfrozen=1`) / 3 / 5 / 1e-2 |
| CVaR alpha_min→alpha_max | 0.30→1.00（按置信度插值） |
| ADA-MCTS 模拟数 | 30000/action |

BNN 世界模型：3 层 Bayesian trunk（52→256→256），方向头 256→3
（Dirichlet 浓度），奖励头 256→2；浓度=预训练计数+`K·CONC_PRIOR`。

---

## 实验结果

### 主表（3节，`experiment_report_2026-09-14.md` §3 有完整5方法×全p数字）

config1（预训练p=1.0）sfir-cem-cvar vs ada-mcts：
0.533/0.933/1.000/0.967 vs 0.367/0.667/0.900/1.000（p=0.3-0.6）

config2（去起点第一个洞）：
0.700/0.767(4种子均值0.859)/0.967/1.000 vs 0.400/0.867/0.900/1.000

config3（预训练p=0.7）：
0.633/0.900/0.967/1.000 vs 0.533/0.700/0.967/1.000

汇总：12点里 9赢/3平/1微弱落后。

### Ablation（config1，单种子，2026-09-15完成）

| 变体 | forget | retrain | p=0.3 | p=0.4 | p=0.5 | p=0.6 |
|---|---|---|---|---|---|---|
| A SFIR（本文方法） | ✓ | head only | 0.533 | 0.933 | **1.000** | 0.967 |
| B retrain全部层 | ✓ | 全trunk | 0.567 | 0.733 | 0.933 | 1.000 |
| C 只forget（=SFI） | ✓ | ✗ | **0.633** | 0.867 | 0.967 | 1.000 |
| D 只retrain | ✗ | head only | 0.600 | **0.967** | 0.933 | 1.000 |
| E Neither | ✗ | ✗ | **0.633** | 0.867 | 0.933 | 0.967 |
| F 完全不适应 | — | — | 0.067 | 0.367 | 0.767 | 0.933 |

---

## 结论/分析

1. 赢的原因分两条、基本独立：(a) `k_models`10→30 是纯规划器精度改进
   （更多独立后验采样→CVaR 尾部估计更稳），跟 SFIR 机制无关；(b) SFIR
   本身在剧变场景优势明显（"检测即忘记、忘记后立刻用新数据重训"比
   ADA-MCTS 的"攒够样本再切换"更快收敛），优势随变化程度增大而增大。
2. **但**：ablation 显示"具体挑哪种适应机制"这件事，SFIR 不是全面
   最优——p=0.3/0.4 有别的变体单种子领先约0.03~0.1。目前定性为
   "初步信号，未经多种子确认"，不推翻决定4（继续用诚实SFIR）。
3. **k_models 不公平问题**：这个"赢"的第一条原因（k_models=30）目前
   只应用在本文方法身上，`ada-cem-cvar`/`oracle-cem` 同样用 CVaR-CEM
   规划器却还留在10——是真实的方法学错误，正在修正（见下）。

---

## 正在跑什么（截至 2026-09-15 20:55，两条线并行）

### 线1：公平性修正（06:45 启动，已跑约 14 小时）

把 `ada-cem-cvar`、`oracle-cem` 也换成
`k_models=30`，重跑 3 组场景 × p=0.3/0.4/0.5/0.6。
- 命令：3 个后台进程（PID 1793674/1793675/1793676），`--workers 5` 各一
  个，共 15 workers/16核。
- 进度（20:49）：3 组的 stationary 都完成（goal rate 1.000）；config3 出
  了第一个非平稳点（p=0.4 cem_ada：return +0.762 / goal rate 0.767）；
  **config1/config2 的日志从 09:25 起 11 小时没有新行**。已核实：15 个
  worker 瞬时 CPU 仍在 70-94%，进程都活着，判断是低 p（0.3/0.4）任务本
  身慢（失败 episode 会跑满 100 步），不是卡死；但仍需继续盯。
- 日志：`$SCRATCH/fair_km30_config{1,2,3}.log`
  （`$SCRATCH=/tmp/claude-1001/-home-guo-Adapt/61e1a280-f090-4695-8c22-6aa922fa1b9d/scratchpad`，
  重启会丢，只是运行时日志，结果要搬进 doc_tool 才算持久化）。
- **修正后的 ETA**：实测单任务约 7-14 小时，30 个任务 /15 worker ≈ 2 轮，
  预计 2026-09-16 凌晨到上午跑完（原估计的 17:00-18:00 已证明过于乐观）。

**中间结果（24 个非平稳点已出 15 个，截至 2026-09-16 02:30）**

`ada-cem-cvar` 在 p=0.3/0.4/0.5 的全部 9 个点已到齐（只差 p=0.6 那一档）：

| config | p | 方法 | 旧 k=10 | 新 k=30 | 变化 | 本文方法该点 |
|---|---|---|---|---|---|---|
| 1 | 0.3 | **ada-cem-cvar** | 0.100 | 0.067 | **−0.033** | **0.533** |
| 1 | 0.4 | ada-cem-cvar | 0.233 | 0.267 | +0.033 | 0.933 |
| 1 | 0.5 | ada-cem-cvar | 0.667 | 0.700 | +0.033 | 1.000 |
| 2 | 0.3 | **ada-cem-cvar** | 0.167 | 0.167 | **0** | **0.700** |
| 2 | 0.4 | ada-cem-cvar | 0.267 | 0.300 | +0.033 | 0.767 |
| 2 | 0.5 | ada-cem-cvar | 0.633 | 0.700 | +0.067 | 0.967 |
| 3 | 0.3 | **ada-cem-cvar** | 0.533 | 0.533 | **0** | **0.633** |
| 3 | 0.4 | ada-cem-cvar | 0.767 | 0.767 | 0 | 0.900 |
| 3 | 0.5 | ada-cem-cvar | 0.900 | 0.933 | +0.033 | 0.967 |
| 1 | 0.3 | oracle | 0.700 | 0.833 | **+0.133** | 0.533 |
| 1 | 0.4 | oracle | 0.933 | 1.000 | +0.067 | 0.933 |
| 2 | 0.3 | oracle | 0.767 | 0.767 | 0 | 0.700 |
| 2 | 0.4 | oracle | 1.000 | 0.967 | −0.033 | 0.767 |
| 3 | 0.3 | oracle | 0.700 | 0.833 | **+0.133** | 0.633 |
| 3 | 0.4 | oracle | 0.933 | 1.000 | +0.067 | 0.900 |

**结论（对 `ada-cem-cvar` 这条已经可以下了）**：**公平对比后，本文方法
相对 `ada-cem-cvar` 的领先完全站得住**。9 个点里最大变化仅 +0.067，三个
p=0.3 点（领先最大处）分别是 −0.033 / 0 / 0，领先幅度 0.533 vs 0.067、
0.700 vs 0.167、0.633 vs 0.533 全部保持。真正被抬高的是 `oracle-cem`
这个理论上限（p=0.3 上 +0.133），而它本来就不参与排名。
机制解读：k_models 提升的是 CVaR 尾部估计的稳定性，只有当模型本身是对的
（oracle 直接在变化后真实 p 上预训练）时才兑现成收益；`ada-cem-cvar` 的
瓶颈在它的两阶段 DPAS 适应机制本身，多给后验采样帮不上忙。
**仍未出的 9 个点**：`ada-cem-cvar` 的 p=0.6 ×3、`oracle` 的 p=0.5/0.6
×6。p=0.6 一档旧值已接近饱和（cem_ada 0.800/0.900/1.000，本文方法
0.967/1.000/1.000），最多在 config1 p=0.6 与我们的 0.967 打平或略高，
不改变整体排名。
注：`oracle_cem` 在 config1 与 config3 逐位相同是设计使然（它总是加载变化
后真实 p 的 checkpoint，与预训练 p 无关），p=0.3 两边都得 0.833、p=0.4 两
边都得 1.000，可作确定性自检。

### 线2：Act as You Learn（ADA-MCTS）复现（20:50 启动）

- PID 2265972，`--workers 7`（stationary + 6 个 p 点，每个 phase 一个
  worker）。完整命令见"关键文件与命令"。
- 论文参数：30000 模拟/动作、N_threshold=50、eps_E=0.02、eps_A=0、
  γ=0.9999、预训练 p=0.7、p∈{0.4,0.5,0.6,0.8,0.9,1.0}、30 trials。
- 日志：`$SCRATCH/aayl_repro_ada_mcts.log`；逐行实时结果在
  `gridworld_cliffwalking_aayl_results.log`（runner 每行 flush，stdout 反
  而会被缓冲，查进度要看这个文件）。
- ETA：30000 模拟下每个动作约 7 秒，episode 上限 100 步 → 单 trial 最坏约
  12 分钟，30 trials ≈ 6 小时/任务；7 任务并行且与线1 抢 CPU，预计
  **8-12 小时，2026-09-16 上午**出全部结果。
- **并行代价**：两条线合计 22 进程抢 16 核，线1 会因此慢约 30%。用户明确
  要求并行（线2 被 collaborator 标为"目前最重要"），已接受这个代价。

---

## 还要跑什么 / 下一步

1. **（进行中）公平性修正**跑完 → 用新数字替换
   `experiment_report_2026-09-14.md` §3 和
   `experiment_summary_2026-09-14.md` §4 里 ada-cem-cvar/oracle-cem
   两行，重新核实本文方法相对它们的领先幅度是否还成立、缩小了多少。
   collaborator说，“把retain factor提高（把lambda hat提高，或是clip rho）能解决p = 0.9很差的问题”。这个也可以试一下，特别是如果k_models=30后，我们的方法很差的话。
2. 来自collaborator的请求：“你可以复现（reproduce）act as you learn 这篇paper的结果嘛？https://arxiv.org/abs/2401.01841
   setting是所有的hole = -1, discount factor = 0.9999 pretrained prob = 0.7. 我复现（reproduce）不出来paper 的结果...你如果有时间可以试试！这个是目前最重要的...“只跑cliff_walking环境，他们提出的ada_mcts方法就行了。他们的环境应该是（相比原版cliff_walking）中间加了一个hole（你确认一下）。
   **状态：2026-09-15 20:50 已启动，见"正在跑什么"线2。**
   **"中间加了一个hole"已核实：不成立**——论文 Fig.2(b) 的 cliff walking
   就是标准 4×12 gym 地图（起点左下、目标右下、底行第1-10列是悬崖），没有
   额外的洞；图注里"We add an extra hole"说的是 Fig.2(c) 的 NS-Bridge。
   详见下面"Act as You Learn 复现：设置与已核实事实"。
3. `k_models=30` 是否也
   该顺手用到 stationary/p=0.7-1.0 那些饱和点上重新验证——大概率无
   影响（已饱和），优先级低。（用户：可能只验证0.7即可？因为如果0.7结果是1，那么0.8/0.9/1.0也都是1，没必要跑。）
4. **Ablation 多种子复核**（p=0.3的C/E领先A约0.1、p=0.4的D领先A的
   点，优先复核）：等公平性修正释放算力后，对6个变体各跑3-4个额外
   种子，取均值判断是不是噪声。
5. 两份对外文档（`experiment_report_2026-09-14.md`、
   `experiment_summary_2026-09-14.md`）目前都已标注"⚠ 待修正"警告，
   修正/复核跑完后要去掉警告、换成最终数字。


---

## Act as You Learn 复现：设置与已核实事实（2026-09-15 新增）

**已核实的事实（有证据，可直接引用）**
1. **cliff walking 没有额外的洞**：论文 PDF 第 9 页 Fig.2(b) 就是标准
   4×12 gym CliffWalking。"We add an extra hole" 是图注里讲 (c) NS-Bridge
   的，那个洞在中间行第 2 列（`G F H F S F F G`）。
   *顺带发现*：我们的 `BRIDGE_HOLE_5x8`（`grids.py`）把额外的洞放在
   (1,4)（起点正上方的肩部），**与论文图 (c) 的 (2,2) 不一致**。本轮用户
   只要 cliff，没有改；将来要用 bridge 结果时必须先修这一条。
2. **官方代码里根本没有 cliff 环境**：`ADA-MCTS/`（用户 fork）以及
   upstream `scope-lab-vu/ADA-MCTS` 的全部分支/PR ref 都只有 frozenlake
   和 nsbridge，没有任何 cliff 文件——论文说"环境随代码提供"，但实际没有。
   所以 cliff 的精确复现只能靠推断作者的约定。
3. **作者的奖励/终止约定**（据其代码血缘：`nsfrozenlake_v0.py`、ns_gym 的
   `nscliff_v0.py`，以及 `adamcts.py` 的 `is_terminal: reward==1 or -1`）：
   **G=+1，H=−1 且立即终止，其余 0**，`discount_factor = 0.9999`
   （`adamcts.py:34`）。与 collaborator 给的设置完全一致。
4. **主表（holes=0）下 ADA-MCTS 的悲观采样从未触发**：
   `_worst_case_sample` 只有"可达格子里存在负奖励"时才一次性 one-hot 到最
   差格子，holes=0 时所有奖励 ≥0 → 直接退化成普通采样。也就是说我们主表里
   的 ADA-MCTS 基线**没有启用它的风险规避阶段**。换成 holes=−1 才会生效。
   这是奖励约定造成的，不是移植 bug，但汇报主表时应当说明。

**我们的移植 vs 论文，已知的 4 处差异**（复现时前两处已按论文对齐）
| 项 | 论文 | 我们 `planning/ada_mcts.py` |
|---|---|---|
| N_threshold | 50 | 原硬编码 3 → 已加 `--ada-n-threshold`，复现用 50 |
| 每个 run 的独立性 | 每个 seed 独立重跑 | 原来 M_k 状态按 phase 共享 → 已加 `--ada-iid-trials`，复现用逐 trial 重置 |
| M_k 更新 | 每 5 步（N_interval）重训 BNN | 每步共轭计数更新（未改） |
| rollout | 随机 rollout 直到终止 | 6 步 rollout + γ^dist 启发式（未改） |

**对论文 Table 1 内部一致性的怀疑（推断，未验证，供 collaborator 参考）**
- 在"γ=0.9999 + ±1 终止奖励"下，回报几乎等于"到达率 − 坠崖率"，取值高度
  双峰；但论文 ADA-MCTS 的误差条只有 ±0.02（cliff 全部 6 个点），这要求回
  报集中在 0.78 附近，**只有当回报随路径长度明显衰减时才可能**（即 γ 远小
  于 0.9999，或存在每步惩罚）。
- 但同表里 RATS 在 p=1.0 恰好是 `0.000 ± 0.00`，**又排除了每步惩罚**（否
  则不到达目标会是负数）。
- 两者合起来意味着：论文 cliff 那一列很可能不是用 γ=0.9999 算的。如果
  collaborator 是按 γ=0.9999 复现，数值对不上是可以预期的。这条要等我们的
  复现结果出来再下结论。

---

## 我提过、还没完整回答的问题

- **"公平对比之后，本文方法是不是真的最好？"**——这是用户最新、最
  重要的问题，**目前还没有答案**，必须等上面"正在跑什么"里的公平性
  修正实验出结果才能回答，不能提前猜。
- ablation 里 SFIR 不是全面最优这件事，"是真实差距还是噪声"——需要
  多种子复核才能回答，目前只有单种子数据。
- **"能不能复现 Act as You Learn 的 cliff walking 结果？"**——线2 已启动，
  预计 2026-09-16 上午出结果，现在还不能回答。注意一个早期信号：冒烟测试
  （仅 300 模拟）里 stationary 阶段 goal rate = 0.000，原因是 γ=0.9999 使
  叶子启发式 γ^dist≈1.0 处处相同、没有指向目标的梯度，只能靠树搜到 13 步
  外的目标——模拟次数不够时会完全找不到目标。正式跑用的是论文的 30000，
  但如果正式结果也接近 0，要优先怀疑这条路径而不是直接下"复现失败"的结论。
- collaborator 的另一条建议"把 retain factor 提高（提高 lambda hat 或 clip
  rho）能解决 p=0.9 很差的问题"——还没试，等线1 结果出来后决定。

---

## 关键文件与命令

**代码**：
- `run_gridworld_experiments.py`——唯一实现整套 SFIR/ablation/多种子
  的文件。关键点：
  - `BNNCEM.__init__`/`act`/`_retrain`（约300-480行）：SFIR机制本体。
  - `build_methods` 的 `cem_fir` 分支（约841-879行）：
    `k_models=(args.cem_k_models if not None else 30)`——**只有这一
    行**让本文方法默认吃到30，其余方法要显式传 `--cem-k-models 30`
    才会用30（这正是本轮公平性bug的根源）。
  - `cem_ada`/`oracle_cem`/`cem_static` 分支：`k_models=args.cem_k_models`
    直接透传，默认是argparse的`None`（→CVaRCEMAgent内部默认10）。
  - `--seed` CLI：多种子复现的入口。
  - `_worker()`：每个(method,phase[,seed])任务的子进程隔离，加新CLI
    参数时必须同步进 `cfg` dict，否则子进程崩溃。
- `bnn/dirichlet_model.py`：`CONC_PRIOR` 常量，当前=0.1。

**当前后台任务**（公平性修正）：
```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
METHODS="cem_ada oracle_cem"; PS="0.3 0.4 0.5 0.6"
CEM="--cem-horizon 6 --cem-candidates 512 --cem-k-models 30"
python run_gridworld_experiments.py --grid cliffwalking --trials 30 --workers 5 --methods $METHODS --change-p $PS $CEM
python run_gridworld_experiments.py --grid cliffwalking_nofirsthole --trials 30 --workers 5 --methods $METHODS --change-p $PS $CEM
python run_gridworld_experiments.py --grid cliffwalking --trials 30 --workers 5 --methods $METHODS --change-p $PS $CEM --orig-p 0.7
```

**Act as You Learn 复现（线2）**：
```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# 新 grid 的 p=0.7 预训练（只需做一次，已完成，VERDICT GOOD）
python pretrain_gridworld.py --grid cliffwalking_aayl --p 0.7 --epochs 400
# 正式复现
python run_gridworld_experiments.py --grid cliffwalking_aayl --orig-p 0.7 \
  --methods ada_mcts --change-p 0.4 0.5 0.6 0.8 0.9 1.0 \
  --trials 30 --workers 7 --ada-n-threshold 50 --ada-iid-trials
```
- 新 grid `cliffwalking_aayl`（`grids.py`）：地图与 `cliffwalking` 相同，
  但 `cliff_to_start=False` + `hole_reward=-1.0`（悬崖=终止洞，付 −1）。
- 新 checkpoint：`data/cliffwalking_aayl/bnn_dirichlet_cliffwalking_aayl_k3_p0p7.pth`。
- 新 CLI：`--ada-n-threshold`（默认 3，不变；论文值 50）、`--ada-iid-trials`
  （默认关，不变）。
- `planning/base.py`：不传 `hole_reward` 时改为从 `bnn.grid` 读（旧 grid 全
  是 0.0，行为逐位不变，已验证），且洞的终止价值 `cell_value[H]` 也改成
  `hole_reward`（否则树内把坠崖记 0、rollout 里记 −1，自相矛盾）。

**检查资源的标准命令（每次跑实验前必做）**：
```bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | sort -t, -k2 -rn | head -8
echo "stray:"; pgrep -f sweep_dirichlet_layers.py | wc -l
uptime
pgrep -af "run_gridworld_experiments.py" | grep -v grep
```

**文档**：
- `doc_tool/experiment_report_2026-09-14.md` + `_en.md`——当前"最终
  状态"详细版（主表+机制分析+ablation），含⚠公平性警告。
- `doc_tool/experiment_summary_2026-09-14.md`——自包含精简版，给外部
  演讲用，不引用其它内部文档，同样含⚠公平性警告。
- `doc_tool/experiment_report_2026-09-10.md` + `_en.md`——完整调查
  历史（13节，调参探索全过程），仅作历史参考，不用于引用当前结论。
- `doc_tool/experiment_report_2026-08-25.md`——更早的背景（α0标定、
  plan_retain）。
- 只读、禁止修改：`doc_tool/experiment_report.md`、
  `doc_tool/experiment_report_new.md`、`doc_tool/20260509_yuanheli2.md`、
  `doc_tool/20260509_yuanheli2_modified.md`。
- `/home/guo/.claude/projects/-home-guo-Adapt/memory/
  gridworld-new-setting-2026-08-12.md` + `MEMORY.md`——跨session记忆，
  记录了本文件里大部分决定的历史来源。

---

## 规则（踩过的坑 / 已有经验）

1. **公平对比铁律（本轮刚踩的坑，最新也最重要）**：任何为了让本文方法
   变好而调的参数，只要其他方法用的是同一底层组件（同一规划器/同一
   网络结构），必须同步给所有这些方法用同样的参数，除非论文本身明确
   设计了不同的对比方式。**定稿任何结果表之前，要把所有方法的每一个
   超参数都过一遍，检查有没有偷偷不一致的地方，不能只检查"我刚改的
   那一个"。** 本轮具体教训：`k_models`只给了`cem_fir`，没给同样用
   CVaR-CEM的`cem_ada`/`oracle_cem`，被用户当场抓到。
2. **线程超订阅**：跑`cem_fir`前必须
   `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`，
   否则 MKL/OpenMP 内部多线程和进程池并发互相打架。
3. **并发度不能跨"数组规模"直接套用旧结论**：`k_models=10`时代验证过
   "进程数超过核数也没关系（CPU共享是免费的）"，但`k_models=30`后单
   任务内存占用变3倍，同样的超订阅（9 workers/16核）会导致内存带宽/
   缓存竞争，CPU利用率从~98%掉到~57%，3.5小时0任务完成。**规律**：
   每次任务的内存footprint变化后，都要重新验证并发度上限，不能想当然
   照搬旧的。目前经验值：`k_models=30`时`--workers`不超过
   `⌈16/3⌉≈5`/组同时跑。
4. **大量kill进程时用更可靠的匹配方式**：`pkill -f "..."`有时匹配不
   稳定/返回非0，改用
   `ps -eo pid,cmd | grep -E "run_gridworld|multiprocessing.spawn" | grep -v grep | awk '{print $1}' | xargs -r kill -9`
   更可靠。
5. **"系统内存不足"提示不一定真——但要验证**：大批量kill后台任务
   偶尔触发一次性的假性内存告警提示，`free -g`确认过实际内存充足后
   排除，不要盲目信一次性提示就改变计划，但也不能不查。
6. **`_worker`是完全独立的子进程**：任何`build_methods`用到的新CLI
   参数，必须同步加进`_worker`读取的`cfg` dict，否则跑起来直接崩溃，
   这个坑在`--seed`、`--retrain-min-conf`等好几次新增参数时都踩过。
7. **单种子结果不可信，尤其差距在0.03~0.15量级时**：本轮多次出现单
   种子"惊艳结果"（如"Neither"0.950）在多种子平均后回归到正常水平
   （0.850）。任何"某个点忽然大幅领先/落后"的结论，先怀疑噪声，用
   `--seed`跑3-4个种子复核再下结论。
8. **CONC_PRIOR大不代表"真的在适应"**：大的浓度先验会让forget很快把
   置信度冲刷到接近均匀，看起来"结果好"但其实规划器自带的置信度门控
   在做全部工作，forget/retrain机制形同虚设——判断"是否真的在利用
   SFIR机制"不能只看最终分数，要看机制本身是否真的被触发/生效。
9. **k_models(K) vs n_rollouts(N) 不是同一回事**：K是独立后验假设数
   （epistemic），CVaR的尾部理应反映这个；N是给定假设下的重采样数
   （aleatoric），增大N不增加新假设，反而可能放大某个恰好异常的假设
   在尾部的占比。调CVaR相关参数时优先想清楚"这个改动是在加假设还是
   在加同一假设的重复采样"。
10. **诊断日志可能挂错方法名，但只是打印，不代表真的传参**：
    `cem_fir SFIR ablation: ...`这行诊断打印是按全局CLI参数打的，即使
    实际跑的是`cem_static`（不接收这些参数，`adaptive=False`绕过整条
    forget/retrain代码路径）也会打印出来，纯粹是日志噪声，已确认不是
    真实bug，不影响结果，不必修。
11. **每次跑实验前按CLAUDE.md检查资源**：
    `nvidia-smi --query-gpu=... ; pgrep -f sweep_dirichlet_layers.py | wc -l`，
    选空闲的资源再跑。
