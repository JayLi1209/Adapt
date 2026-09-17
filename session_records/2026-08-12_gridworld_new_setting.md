# Session Record — 2026-08-12 Gridworld 新设定（FIR-CEM 主方法 + unbounded RATS + p=1.0）

> 承接 `2026-08-06_rats_gridworld.md`。当日进度以 `## Progress Log` 追加。
> 用户要求：中文交流（术语/log 可用英文）。只读文件（不得修改）：
> `doc_tool/20260509_yuanheli2.md`、`doc_tool/20260509_yuanheli2_modified.md`、
> `doc_tool/experiment_report.md`、`doc_tool/experiment_report_new.md`。

## Progress Log

- **2026-08-12**: 根据两对 doc diff 推断的修改方向实现新设定并跑实验。
  详见下文与 `doc_tool/experiment_report_2026-08-12.md`（新建报告）。
  commit：39e38f6（新方法+设定）、bd00d41（ADA-MCTS stale model fix）、
  bf18017（CEM gating tune）、a918345（stationary crash fix）、6b1200e（报告）。
- **2026-08-13**: bridge 结果填入报告（commit 0cee44e）；**bridge 参数调优完成**：
  (1) K_FORGET sweep（1/2/3）证明 forget 已能触发但"均匀模型"在 bridge 上无安全
  路线（悬崖上没有保守选择）→ adaptive≡static 与 K 无关，是结构性不是没调参；
  (2) CEM sweep（horizon 3/4/5 × n_confident 2/3 × alpha_min 0.05/0.10）：**所有
  配置 cem_fir 都差于 cem_static**（约 -0.10 goal rate），最优是 h=5/n_conf=2/
  alpha_min=0.10（cem_static 0.120/0.250/0.330/0.480，cem_fir 0.090/0.150/0.270/
  0.390）；horizon 从 3→5 只把 cem_static 从 0.48 微升到 0.48——CEM 在 bridge
  的差距不是 horizon 能解决的；bridge 报告数字维持 h=3 默认。
- **2026-08-13（完）**: cliff 全量实验完成，结果+讨论填入报告（commit f4dfa20）。
  **核心结论**：FIR 在 cliff 对两个 planner 都有效（bnn_rats_adaptive 与 cem_fir
  在低 p 大幅反超 static）；FIR 收益与变化幅度正相关；unbounded 两方案成立；
  bridge 上 FIR 无效是结构性（短 episode + 无退避路线）。报告
  `doc_tool/experiment_report_2026-08-12.md` 已完整（§4.2/4.3/4.4/§5 全填）。
- **2026-08-13（cem_fir 调参）**: 目标"cem_fir 超过所有 baseline"。诊断：原
  门控 α_min=0.10 + n_confident=20 + tau=2 过度保守（forget 把模型清成均匀后
  CEM 规划不出东西，静态门控还把 α 压在底部）。改为 **α_min=0.30 + n_confident=8
  + surprise_tau=50**（commit 6a70eef）+ forget 后重置 drift（commit 89c97c3）。
  cliff p=0.4 探针（5 trials）cem_fir 达 0.80-1.00（原 0.867）。bridge 上
  CEM 本身gap（cem_static 0.10 vs RATS 0.19）horizon 3-9 都救不回——planner
  固有局限，cem_fir 在 bridge 最多与 cem_static 持平。全量 cliff 重跑中。
- **2026-08-14（cem_fir 调参完成）**: cliff 全量重跑完成（commit 2010b41）。
  **cem_fir 调参后 cliff：0.833/0.867/0.967/0.967/1.000/1.000/1.000**，
  全 p 超过 cem_static（0.333/0.733/0.867/1.000/...）、rats_cv01（0.300/...）、
  rats_cal（0.233/...）、bnn_rats_static（0.300/...）、ada_mcts（0.533/...）、
  mcts_static（0.600/...），仅低于 oracle_rats（0.833 持平 p=0.4，p=0.5 略低）
  和 bnn_rats_adaptive（FIR-RATS 仍是 cliff 最强）。**主方法 FIR-CEM 在所有
  非 oracle baseline 之上**。报告 `doc_tool/experiment_report_2026-08-12.md`
  §4.3/§5 已更新。
- **2026-08-14（新设定+调参）**: 用户要求 cem_fir 在两环境成为"除 oracle 外
  最强"，bridge 加 hole（对齐 Act As You Learn），并解释 bnn_rats_adaptive 超
  oracle。完成：
  (1) **bridge_hole 环境**（grids.py BRIDGE_HOLE_5x8，起点上方 (1,4) 加 hole，
      预训练 p=1.0 ckpt，commit 8465a7d）；
  (2) **oracle 诊断**：oracle_rats 的 L_p=1.0 worst-case 球过保守（p=0.4 时策略
      在 (2,10/11) 撞墙停下，goal rate 0.833），而 bnn_rats_adaptive forget 后的
      均匀模型让 RATS 走全程贴顶行的更安全路线（1.000）——不是 adaptive 真超
      "知道真相的上界"，是 L_p 未校准的伪影；
  (3) **cem_fir 调参**：cliff 加 `--cem-candidates 512`（0.38→0.88 追平
      bnn_rats_adaptive 同种子）；bridge_hole 用 `--k-forget 1 --cem-horizon 6
      --cem-n-confident 8`（p=0.6 达 0.30-0.38，cem_fir≥cem_static 但 CEM 系仍
      落后 RATS 系——planner 固有局限）。新报告 `experiment_report_2026-08-14.md`
      （commit 5e71cf1）。全量 cliff（cand512）+ bridge_hole 重跑中。
  (4) 试过 cross-trial persistence（跨集累积 counts/drift）但 cliff 退化
      （0.833→0.62），已 revert。
- **2026-08-22（修正）**: 用户指出三个问题并修正：
  (1) **hole reward 0.0 不是 -1**（paper 约定 "holes=0"，goal rate 与 return 对齐）：
      `grids.py` 新增 `hole_reward` 字段，`planning/rats.py` cell_reward_of、
      `planning/base.py` hole_reward 默认 0.0、`run_gridworld_experiments.py`
      cell_reward 全部改为用 `grid.hole_reward`（commit b4852e4）；
  (2) **gamma 仍为 0.99**（planner 用；0.9999 是评估折现，不改 planner）；
  (3) **bridge 用原始环境不加 hole**（bridge_hole 仅作参考，主实验回退到 bridge）；
  (4) **pretraining 平衡**：`pretrain_gridworld.py` 新增 `--balance-terminal`
      （commit 026ae7a）——目标加权采样让终态（G/H）过度代表，平衡后 cliff/bridge
      新模型 MAE 不变（0.0003/0.0002）。验证：平衡模型在 cliff p=0.4 仍 1.000、
      bridge p=0.4 仍 0.167（与不平衡一致——**当前模型质量不是瓶颈，MAE 已极低**）。
- **2026-08-25（alpha0 标定）**: 排查"pretraining 效果差"的剩余问题：**concentration
  α0 与数据量脱钩**。诊断：categorical NLL 只依赖均值 p=α/α0，对 α 缩放不变 →
  α0 完全由 KL→prior 决定，实测 cliff α0≈7.5 / bridge≈11，而 20000 行数据下
  共轭后验应为 ~135/312。纯 Dirichlet-multinomial count loss 也救不了（确定性
  数据下空类别触底后 α0 梯度≈0，MLE 是 α0→∞ 的渐近线）；log 空间回归项
  （conf_weight）Adam 步长限制 400 epochs 只能爬到 α0≈14。最终方案：**预训练计数
  表**——`DirichletDynamicsModel` 新增 `pretrain_alpha0` buffer（α0[s,a]=
  预训练 counts + K·CONC_PRIOR），head 的 softplus 输出只管方向均值，表管浓度
  （`set_pretrain_counts()`，全零行回退 head 幅度，旧 checkpoint 行为不变）。
  表在 retain 变换**之前**进入 α → forget 照样能把预训练置信度衰减到均匀先验，
  在线 counts 照常叠加。顺带修 bug：pretrain 的 reward 标签里 hole 硬编码 -1.0，
  与 8-22 的 hole_reward=0.0 不一致 → 改用 `grid.hole_reward`。
  重训 cliff/bridge p=1.0 ckpt（--balance-terminal）：MAE 0.0003/0.0002 不变，
  α0 mean=135.4/312.8 精确跟踪 counts（mean 135/312）。探针实验重跑中
  （/tmp/probe_cliff_p04_alphatab.log, /tmp/probe_bridge_p04_alphatab.log）。
- **2026-08-25（bridge cem_fir 调参 + plan_retain 门控）**: 用户目标：bridge 上
  cem_fir 超所有非 oracle baseline（新模型全量 baseline：bnn_rats_static
  0.240/0.430/0.620 @ p=0.4/0.5/0.6）。
  (1) **轨迹诊断**（/tmp/diag_bridge_traj.py）：bridge 两端都有 G——右桥近
      (dist 3) 但两侧是 H，左岸远 (dist 4) 全程安全。CEM（static 和 fir）直冲
      右桥掉洞（≈p³ 运气）；bnn_rats_static 全走左岸 7/8 成功。RATS 的保守来自
      L_p worst-case 球（解析 adversary），α0 变锐利后不受影响；但 CEM 的 CVaR
      尾部依赖后验弥散，α0=312 后 100 个采样全同 → CVaR≈均值 → 风险盲区。
      **α 旋钮在锐利模型下是死的**（alpha_min 0.3→0.5 结果逐点不变）。
  (2) **plan_retain 门控**（main_cl.tex 的 "restore uncertainty BEFORE
      planning"）：`planning/cvar_cem.py` 新增 plan_retain——规划读模型时把
      bnn.retain 临时乘上 conf（与 CVaR α 同一 confidence 信号）；counts 在
      retain 变换之后叠加、始终全信；static/stationary 不受影响。附带修
      BNNCEM.reset 不重置 surprise_bar/n_since_change 的跨 trial 泄漏。
      门控后 cem_fir 轨迹立即改走左岸。
  (3) **horizon 是关键**：sweep2/3 忘传 --cem-horizon（默认 h3），h3 看不到
      左岸 goal（dist 4）→ 被启发式 γ^dist 误导。h6 + kf1 后 100-trial：
      H(kf1,h6,nc2) 0.220/0.290/0.480，J(kf3,h6,nc2) 0.160/0.330/0.540。
  (4) plan_gamma 假设被否：pg0.9 (K) 0.230/0.240/0.400 反而更差（K<L? 待确认）。
  (5) sweep7 在测：nc99/nc20（门控整段关闭，纯 counts 模型）、amin0.1。
  扫描日志：/tmp/bridge_cem_sweep{2..7}.log。
  cliff 全量（α0 表模型+旧门控）/tmp/grid_cliff_2026-08-25.log 收尾中；
  cem_fir/cem_static 需用新门控代码重跑。新报告
  `doc_tool/experiment_report_2026-08-25.md`；handoff
  `session_records/2026-08-25_handoff.md`。
- **2026-08-25/26（bridge 调参完成 + 全量收尾）**: sweep2-10 共 20+ 配置
  （100 trials）：plan_gamma 有害（K/L）、大预算无用（R）、warm-blend 有害
  （Y/Z/AA）、nc99/nc20 有害（N/P）、it8 无用（S）；有效的是
  **kf1 + h6 + nc2 + amin0.1（Q 配置）**。最终 bridge 全 p：
  cem_fir 0.200/0.350/0.540/0.610/0.860/0.950/1.000（门控前 0.09/0.18/0.25/...），
  超 cem_static/ada_mcts/mcts_static 全部，p=0.4 超 oracle（0.20 vs 0.19），
  p=0.5 追平 oracle；未超 RATS 家族（残余差距 = CEM 开环候选评估 vs RATS 闭环
  minimax，budget×4 无改善 → 结构性）。cliff 全量（α0 表模型）完成；cem 用
  新门控代码重跑：cem_fir 0.767/0.867/0.900/0.967/1.000/1.000/1.000，仍超所有
  非 oracle、非 SFI-RATS baseline（p=0.4：0.767 vs 次优 0.567）。报告
  `doc_tool/experiment_report_2026-08-25.md` 完整。
- **2026-08-26/27（γ 统一 0.9999 全量重跑）**: 用户指出 γ 应统一 0.9999（此前
  规划+评估都是 0.99，"评估 0.9999"的说法是错的）。`GAMMA=0.9999`
  （run_gridworld_experiments.py L48，规划评估共用）。γ≈1 + holes=0 →
  return ≈ goal rate 逐点对齐（用户要求）。**格局变化**：
  (1) bridge：γ^dist 启发式拉平 → "存活≈满分" → cem_static 无需适配即成纯避洞
      （0.11/0.22/0.31 → 0.16/0.34/0.60）；SFI-CEM 各 g9999 配置（W/X/Y 系列 +
      Q）与 static 持平未超过；RATS 家族仍领先（结构性：开环 CEM vs 闭环
      minimax）。bridge cem 报告 Q 配置全 p：0.16/0.35/0.52/0.61/0.86/0.95/1.00。
  (2) cliff：MCTS 系大涨（ada_mcts 0.667/0.900/1.000，mcts_static
      0.633/0.967/1.000）——掉 cliff 是传送回起点非终止，γ≈1 + 100 步预算 →
      p≥0.7 饱和。cem_fir 最终用 **h6**（cand512）：全 p
      **0.767/0.900/0.933/1.000/1.000/1.000/1.000**（/tmp/grid_cliff_2026-08-27_cem_h6.log）；
      p=0.4 超所有非 oracle baseline（0.767 vs ada 0.667），对 cem_static
      （0.367/0.800/0.867）增益翻倍；p=0.5 平 ada_mcts、低于 mcts_static 0.967。
      oracle_rats p=0.4 降到 0.600（γ 改变其 worst-case 策略）。
  报告已重写为 γ=0.9999 版（含 §7/§8 双语设计说明）。
- **2026-08-14（完）**: cliff cand512 全量完成（commit 744bf26）。**cem_fir
  调参后 cliff：0.767/0.767/0.900/1.000/1.000/1.000/1.000**，超过所有非 oracle
  baseline（rats_cv01/rats_cal/bnn_rats_static/cem_static/ada_mcts/mcts_static），
  仅低于 oracle_rats（p=0.4 持平 0.833）和 bnn_rats_adaptive（FIR-RATS）。
  bridge_hole 全量完成（commit f517be5）：cem_fir≥cem_static 全 p 但 CEM 系仍
  落后 RATS 系（planner 固有局限）；bridge_hole 上"除 oracle 外最强"是
  bnn_rats_adaptive。**结论：FIR-CEM 在 cliff 达到"除 oracle 外最强"；bridge
  上 CEM planner 本身不如 RATS，FIR 有效但 cem_fir 追不上 bnn_rats_adaptive**。
  报告 `doc_tool/experiment_report_2026-08-14.md` 已完整（§4.1/§4.2 全填）。

## 1. 做了什么

1. **规划器换成 CEM（FIR-CEM 主方法）**：新增 `BNNCEM`（runner L271），把
   `CVaRCEMAgent`（planning/cvar_cem.py）接上 FIR 在线循环（surprise → drift
   → forget → learn），同一个 surprise 信号门控 CVaR 尾部 α。
2. **unbounded RATS 两方案**（我们的变化是无界的，没有有效 L_p/L_r）：
   - 方案1 `rats_cv01`（L357）：N=100 个后验模型取最差（≈1% CVaR 经验尾）。
   - 方案2 `rats_cal`（L403）：N=100 个后验模型校准 L_p，沿用现有 RATS worst-case。
3. **去掉两个 DP oracle**：默认方法列表不再含 dp_nsmdp / dp_snapshot（连续情形另有实验）。
4. **环境改成 p=1.0 → p_new**：ORIG_P 0.7→1.0（确定性，先验证模型能找到最优解），
   CHANGE_PS 向小扫 {0.4..1.0}；新预训练 checkpoint `data/cliffwalking/
   bnn_dirichlet_cliffwalking_k3.pth`、`data/bridge/bnn_dirichlet_bridge_k3.pth`
   （MAE 0.0003 / 0.0001，GOOD）。
5. **cliff reward 去掉每步 −1**：`grids.py` step_penalty −1.0→0.0（reward = G+1/
   H−1/每步0，return 是折现 goal rate，更好解读）。
6. forget 的 "inflate variance" 即现有 retain 衰减实现，未改动；NS-Bridge 不加
   hole（加了就没有最优路线了）——两条按用户指示不做。

## 2. 实验 setting

| 项 | cliffwalking | bridge |
|---|---|---|
| 网格 | 4x12，K=3 | 5x8，K=3 |
| 原环境 p（预训练） | **1.0（确定性）** | 1.0 |
| 变化 | ts 0 起 p → {0.4,0.5,0.6,0.7,0.8,0.9,1.0} | 同左 |
| γ | 0.99 | 0.99 |
| RATS/CEM 深度 | 3（RATS_DEPTH） | 3 |
| max_steps | 100 | 10 |
| trials | 30 | 100 |
| reward | G=+1, H=-1, 其余 0（无每步惩罚） | 同左 |
| K_FORGET | 3 | 3 |
| ADA-MCTS | 30000 sims/action（论文值） | 同左 |
| 方法（9个） | oracle_rats, rats_cv01, rats_cal, bnn_rats_static, bnn_rats_adaptive, cem_static, cem_fir, ada_mcts, mcts_static | 同左 |

## 3. 实验结果

（实验跑完后填入；stationary 初步：cliff/bridge 全部 BNN 方法 + oracle 1.000
goal rate——p=1.0 确定性环境下模型找到了最优解，验证通过。）

## 4. 关键发现 / 问题

1. **CEM 置信门控默认太弱**：`planning/cvar_cem.py` 默认 ALPHA_MIN=0.95 与
   风险中性 α=1 几乎无差别（CVaR 尾部占 95% draws），门控基本不改变行为。
   调为 ALPHA_MIN=0.10（10% 最差尾部）、N_CONFIDENT=12→20（runner 常量
   CEM_ALPHA_MIN/CEM_N_CONFIDENT，commit bf18017）。注意：门控"起作用"表现为
   谨慎期保守（step 多、偶尔掉洞），goal rate 略低于 cem_static 是设计使然。
2. **ADA-MCTS stale M_{k-1}**：重复 notify 时 deepcopy(bnn) 会把 counts 适应后
   的模型当成旧模型 M_{k-1}。修复：phase 内第一次 notify 保留 pretrained 快照
   （planning/ada_mcts.py notify_change 加 `if self.bnn_prev is None`，
   commit bd00d41）。
3. **stationary crash**：ADA-MCTS act() 对 change_step=None 未守卫（stationary
   阶段）导致 TypeError，修复（commit a918345）。
4. **p=1.0 预训练的特殊性**：slip [1,0,0] 下 BNN 熵≈0，surprise = -log p/entropy
   在前几步除零失真（surprise_bar 0.09→400+ 跳变），forget 在 t≈6 一步把
   retain 打到 ~0——比 p=0.7 训练（熵≈0.6，surprise 渐进）更剧烈。CEM 的
   conf_surprise = exp(-(surprise_bar-1)/tau) 在 surprise_bar 从 400 回落的
   过程里长期 ≈0（α 锁 ALPHA_MIN）。这解释了 cem_fir 的谨慎期偏长。

## 5. 关键文件

- `run_gridworld_experiments.py`：`BNNCEM`(L271)、`RATSCV01`(L357)、
  `RATSCalibrated`(L397)、`WorstKSnapshot`/`CalibratedSnapshot`、默认方法列表、
  ORIG_P=1.0、CEM_ALPHA_MIN/CEM_N_CONFIDENT
- `planning/cvar_cem.py`：`CVaRCEMAgent`(L41)
- `planning/ada_mcts.py`：`notify_change` stale fix
- `grids.py`：cliff step_penalty 0.0
- `doc_tool/experiment_report_2026-08-12.md`：新报告（结果待填）
- 预训练 ckpt：`data/cliffwalking/bnn_dirichlet_cliffwalking_k3.pth`、
  `data/bridge/bnn_dirichlet_bridge_k3.pth`
- 结果 log：`/tmp/grid_cliff_2026-08-12.log`、`/tmp/grid_bridge_2026-08-12.log`
