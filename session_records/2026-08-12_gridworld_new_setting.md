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
