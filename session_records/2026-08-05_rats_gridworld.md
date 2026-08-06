# Session Record — RATS vs ADA-MCTS Gridworld 实验

> 此文件同时充当 session context 与进度记录：新 session 先读本文件恢复上下文，
> 每次有阶段进展时在此追加 `## Progress Log` 条目。
> 用户要求：中文交流（术语/log 可用英文）。

## Progress Log

- **2026-08-05 (session 1)**: 完成 RATS 实现（planning/rats.py）、ADA-MCTS 泛化、
  预训练 cliff/bridge (p=0.7) 并 push、两个完整实验跑完（结果见 §4）。
  发现 stationary 阶段 adaptive 误触发 bug（修复未提交，见 §5.1）。本文档建立。
- **2026-08-05 (session 2)**: 提交 stationary 修复（commit ffb6a75）并验证
  adaptive==static（cliff 1.000/1.000、bridge 0.910/0.910）。
  **发现更严重的 bug（§5.6）**：p_schedule 在 change_step=0 时两个 ts=0 条目被
  `sorted()` 反序，`active_p(t)` 恒返回 0.7 —— **非平稳阶段环境从未真正变化**，
  §4 全部非平稳数字无效。已修复（change_step<=0 时省略 ORIG_P 段 + active_p_fn
  拒绝重复时间戳），修复后 dp_nsmdp==dp_snapshot==oracle_rats 于 bridge p=0.4
  （0.730/0.700，环境真实在 0.4）。完整实验已用修复后代码重跑中，结果待填 §4。

## 1. 任务背景（用户原始请求）

1. 根据 `NSMDP.md`（Lecarpentier & Rachelson, NeurIPS 2019）实现 **RATS**（Risk-Averse Tree-Search）
2. 在 **cliffwalking** 和 **ns-bridge** 上跑实验：
   - 先在 stationary 环境下确认 pretrained model 表现好
   - 预训练模型已 push 上 GitHub
   - 引入新环境（非平稳变化），log 每步 reward/return，**γ=0.99**
   - 与刚 push 的 **ADA-MCTS baseline**（Luo et al. 2024, "Act As You Learn"）对比

## 2. 环境与实验设定

| 项 | cliffwalking | bridge |
|---|---|---|
| 网格 | 4x12（CLIFFWALKING_4x12, grids.py） | 5x8（BRIDGE_5x8） |
| K 方向 | K=3（intended + 2 垂直滑移） | K=2（intended vs opposite 掉头） |
| 原环境 p | 0.7（预训练目标） | 0.7 |
| 变化 | ts 0 起 p → {0.4,0.5,0.6,0.8,0.9,1.0} | 同左 |
| slip_dist(p) | [p, (1-p)/2, (1-p)/2] | [p, 1-p] |
| γ | 0.99（用户指定） | 0.99 |
| max_depth (RATS/DP) | 6 | 6 |
| max_steps | 100 | 10 |
| trials | 30 | 100 |
| heuristic | "potential" = γ^dist(s,goal) | 同左 |
| K_FORGET | 5 | 5 |
| ADA-MCTS | M_SIMULATIONS=1000（默认 2000 太慢）, CP=√2, EPS_E=0.02, H_ROLLOUT=6 | 同左 |
| reward | G=+1, H(悬崖)=-1, 其余 0；goal rate 统计中 H 记 0（paper 约定） | 同左 |

方法集合（run_gridworld_experiments.py 中定义）：
- `dp_nsmdp`（oracle，DP-NSMDP：用真实时变 p(t) 计划，上界）
- `dp_snapshot`（oracle，DP 快照：用真实当前 p 计划）
- `oracle_rats`（oracle，RATS 用真实当前 p 快照）
- `bnn_rats_static`（学习，RATS + 预训练 BNN 快照，无自适应）
- `bnn_rats_adaptive`（学习，RATS + surprise/forget/online-counts 自适应）
- `ada_mcts`（baseline，被通知变化，DPAS 双相采样）

## 3. 已完成的工作

- **`planning/rats.py`**（新文件，核心交付物）：RATS 完整实现
  - `worstcase_distribution_direct_method`（Property 3 闭式解；W1(δ_k, w0)=w0@d[:,k] 免 LP）
  - 树是 DAG，按 (state, depth) memoize，**精确等价**官方全树但 O(|S||A|d)
  - `GridSnapshot`（oracle 静态）/ `DynamicGridSnapshot`（oracle 时变，DP-NSMDP 用）/ `BNNSnapshot`（BNN 确定性均值，counts/retain 参与前向 → 自适应可见）
  - `RATS`（minimax worst-case，_V/_Q memoized）、`DPAgent`（期望树，静态/动态两模式，键 (s,t,d)）
  - 参考：官方代码在 `/tmp/rats-experiments/`
- **`planning/ada_mcts.py`**（修改）：从 FrozenLake 硬编码泛化到任意 grid（nrow/ncol 从 desc 推断、`grid.direction_of`、`grid.dir_actions`+`grid.deltas`）
- **`run_gridworld_experiments.py`**（新文件）：实验 runner（stationary 验证 + 非平稳 + 汇总表 + per-step rewards）
- **`pretrain_gridworld.py`**（已存在）：预训练（goal-weighted 采样；cliff MAE=0.004、bridge MAE=0.009，VERDICT GOOD）
- 预训练模型已 push：`data/cliffwalking/bnn_dirichlet_cliffwalking_k3_p0p7.pth`、`data/bridge/bnn_dirichlet_bridge_k2_p0p7.pth`
- 已 commit `aa92d94`（"Add RATS planner + DP baselines, generalize ADA-MCTS to grids, pretrain models"）并 push origin（JayLi1209/Adapt, 分支 steven202_20260713_v2）

## 4. 最终实验结果（已完成，log 在 /tmp/grid_bridge_run.log、/tmp/grid_cliff_run.log）

### cliffwalking — goal rate by p（holes=0）
```
method              p=0.4   p=0.5   p=0.6   p=0.8   p=0.9   p=1.0
dp_nsmdp             1.000   1.000   1.000   1.000   1.000   1.000
dp_snapshot          1.000   1.000   1.000   1.000   1.000   1.000
oracle_rats          0.900   0.900   0.900   1.000   1.000   1.000
bnn_rats_static      1.000   1.000   1.000   1.000   1.000   1.000
bnn_rats_adaptive    0.833   0.833   0.833   0.933   1.000   1.000
ada_mcts             0.800   0.900   0.933   0.933   1.000   1.000
```
γ=0.99 折现 return：dp_nsmdp 0.72–0.89 / dp_snapshot 0.78–0.89 / oracle_rats 0.59–0.89 / bnn_static 0.70–0.87 / bnn_adaptive 0.61–0.87 / **ada_mcts 0.43–0.58**

### bridge — goal rate by p
```
method              p=0.4   p=0.5   p=0.6   p=0.8   p=0.9   p=1.0
dp_nsmdp             0.160   0.740   0.910   0.980   1.000   1.000
dp_snapshot          0.910   0.910   0.910   0.980   1.000   1.000
oracle_rats          0.910   0.910   0.910   0.980   1.000   1.000
bnn_rats_static      0.910   0.910   0.910   0.980   1.000   1.000
bnn_rats_adaptive    0.910   0.910   0.910   0.980   1.000   1.000
ada_mcts             0.280   0.170   0.270   0.310   0.470   0.410
```
γ=0.99 return：dp_nsmdp 0.15–0.98 / 其余 oracle+BNN 0.88–0.98 / **ada_mcts −0.61 至 −0.06**

### stationary 验证（p=0.7）
- bridge：全部方法（除 ada_mcts）0.875 return / 0.910 goal rate；**ada_mcts 仅 −0.562 / 0.200**
- cliff：dp 1.000 / oracle_rats 0.900 / bnn_static 1.000 / **bnn_adaptive 0.833（bug 污染，见 §5）** / ada_mcts 0.800

## 5. 已发现的问题 / 待调查

1. **⚠ 未提交的修复（必须提交）**：stationary 阶段 adaptive 循环误触发 bug。
   旧代码 change_step=0 → t≥1 就 forget，retain 衰减伤害模型（cliff stationary adaptive 0.833 < static 1.000）。
   已修复但**未提交**：`run_gridworld_experiments.py` 中 `build_methods(..., change_step=None)`（约 L257-287）+ `BNNRATS.act` 加 `self.change_step is not None` 守卫（约 L172）。修复后 stationary 阶段 adaptive 应完全等于 static。
2. **adaptive 在非平稳下仍比 static 差**（cliff p≤0.6：0.833 vs 1.000）——forget every K_FORGET=5 似乎伤害模型而非帮助。需调查：forget 频率、surprise 阈值、或 retain 衰减是否过猛。
3. **bridge p=0.4 时 dp_nsmdp (0.16) 远低于 dp_snapshot (0.91)**：omniscient 上界反而最差，违反直觉。0.16 = 0.4² 说明 dp_nsmdp 的"最优"是走 2 步直达且失败就死；其余方法计划用 0.7 却仍 0.91。可能 bridge 几何（掉头回岸）使 slip 不致命，dp_nsmdp 对 0.4 的悲观期望导致保守路径选择。**需调查 bridge 布局与 move 语义**。
4. **ADA-MCTS baseline 全面落后**（cliff 0.43-0.58 vs RATS 0.70-0.87；bridge 甚至负 return），与 paper 表不符（paper cliff ADA-MCTS 0.778-0.883）。可能原因：M_{k-1} 快照/epsilon 阈值配置、rollout heuristic、或 1000 sims 太少。可作为对比结论，也可继续调。
5. 本次实验用旧代码启动（非平稳阶段不受 bug 影响，数字有效）；stationary 的 adaptive 数字需用修复后代码重跑确认（预期 = static）。
6. **⚠⚠ 重大 bug（session 2 发现，已修复并重跑中）**：`main()` 非平稳阶段构造
   `p_schedule = [(0, ORIG_P), (0, p_new)]`（change_step=0），`active_p_fn` 内
   `sorted()` 对元组排序 → `[(0, p_new), (0, ORIG_P)]`，`active_p(t)` 取"最后一个
   ts<=t 的 p"恒为 **0.7**。即：**环境自始至终运行在 p=0.7，变化从未生效**。
   而 dp_nsmdp 的 oracle `dist_by_time={0: slip(p_new)}`（dict 键冲突，后者胜）
   却相信环境是 p_new —— 所以 dp_nsmdp"按 p_new 规划、在 0.7 环境跑"：
   bridge p=0.4 时 0.16（dp_snapshot 反而用真实 0.7 规划 → 0.91）。§4 非平稳表
   全部作废。修复：change_step<=0 时 `p_schedule=[(0,p_new)]`（首步即在 p_new），
   active_p_fn 增加重复时间戳校验（抛错）。修复后验证：bridge p=0.4 三个 oracle
   方法完全一致 0.730/0.700。

## 6. 待办事项（新 session 从这里继续）

1. **提交未提交的修复**：`git add planning/ada_mcts.py run_gridworld_experiments.py && git commit && git push`
2. 向用户汇报完整结果表（§4），含与 ADA-MCTS 的对比分析
3. （可选）用修复后代码重跑 stationary 验证 adaptive（`--methods bnn_rats_adaptive bnn_rats_static`，很快）
4. （可选）调查 §5.2/5.3/5.4
5. （可选）正式表格整理：按 paper 的 Cliff/NS-Bridge 表格式出对比

## 7. 复现命令

```bash
# 预训练（若模型丢失）
python pretrain_gridworld.py --grid cliffwalking --p 0.7 --epochs 400
python pretrain_gridworld.py --grid bridge --p 0.7 --epochs 400

# 实验
python run_gridworld_experiments.py --grid cliffwalking --trials 30 --m-simulations 1000
python run_gridworld_experiments.py --grid bridge --trials 100 --m-simulations 1000
```

## 8. 关键文件

- `planning/rats.py` — RATS + DP 基线（新，核心）
- `planning/ada_mcts.py` — ADA-MCTS baseline（已泛化；M_SIMULATIONS/CP/EPS_E 等在文件头约 L25-31）
- `run_gridworld_experiments.py` — 实验 runner（含未提交修复）
- `pretrain_gridworld.py` — 预训练脚本
- `grids.py` — GridSpec 注册表（slip_dist/dir_actions/move/deltas）
- `bnn/`（dirichlet_model.py, dirichlet_workflow.py 的 surprise_dirichlet/forget_dirichlet/epistemic_dirichlet）、`drift.py`（DriftFilterV2）、`config.py`（device/ETA/GAMMA_UNCERTAINTY）
- `data/cliffwalking/`、`data/bridge/` — 预训练 ckpt（已 push）
- `/tmp/rats-experiments/` — 官方 RATS 参考实现
- 结果 log：`/tmp/grid_bridge_run.log`、`/tmp/grid_cliff_run.log`；仓库内 `gridworld_bridge_results.log`、`gridworld_cliffwalking_results.log`（同内容）

## 9. 其他约定（CLAUDE.md 摘要）

- 运行实验前先查 GPU：`nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | sort -t, -k2 -rn | head -8`，选最空闲的卡
- 预训练始终优先训练靠近 goal 的 transition
- 用户解释实验设定要求：变化性质（何时变什么）/ 架构 / 超参（尤其 γ、α）/ 截断步数与 trial 数 / FrozenLake reward 结构
- 默认设定：FrozenLake 无折扣无截断 [1,0,0]→[0.7,0.15,0.15] K_FORGET=1 100 trials；pendulum mass 1→4 200ep γ=0.99 gaussian head 100 trials（本次 gridworld 按用户指示用 γ=0.99）
- 每次 prompt 后如有新文件，告知文件名与修改文件的大致行号
