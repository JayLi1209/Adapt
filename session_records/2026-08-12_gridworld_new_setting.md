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
- **2026-08-13**: bridge 结果填入报告（commit 0cee44e）；**bridge 参数调优**：
  K_FORGET sweep（1/2/3）证明 forget 在短 episode 已能触发但"均匀模型"在 bridge
  上无安全路线可走（悬崖上没有保守选择）→ adaptive≡static 与 K 无关；
  CEM horizon/n_confident/alpha_min 扫描中（--cem-horizon 等新 CLI 参数，
  commit 4304561）；cliff 全量实验收尾中（mcts_static 是最后的方法）。

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
