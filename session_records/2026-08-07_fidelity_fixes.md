# Session Record — 2026-08-07 保真度修正 + 全量重跑 + 论文格式实验报告

> 承接 `2026-08-06_rats_gridworld.md`。当日进度以 `## Progress Log` 追加。
> 用户要求：中文交流（术语/log 可用英文）。

## Progress Log

- **2026-08-07**: 依据 doc_tool 文档审查结果（experiment_fidelity_review.md），
  按用户对齐指示（"算法对齐 Act As You Learn 论文，找不到的设置用 repo 里的，
  确保公平比较"）完成全部偏差修正 + 全量重跑 + 论文格式实验报告：
  1. **bridge 环境修正**：BRIDGE_5x8 K=2 掉头 slip → **K=3 垂直 slip**
     [p,(1-p)/2,(1-p)/2]（官方 nsbridge_v0.py 几何）。t=0 确定斜坡有意不实现
     （属 RATS 论文连续设定；L_p=1 下 t≥1 即饱和），已在文档声明。
  2. **cliff 每步惩罚**：GridSpec.step_penalty=-1.0（"except the goal"）。
  3. **方法列对齐**：新增 `rats_pkminus1`（RATS-P_{k-1}，固定 p=0.7 oracle）、
     `mcts_static`（MCTS-ĥP_{k-1}，无通知无在线学习）；RATS 深度默认 **3**
     （论文文档值）、DP 深度 100（精确）；M_SIMULATIONS=**5000**（upstream
     demo 值；30000 实测 7.5s/action 全量不可行）。
  4. **bridge BNN 重训**：K=2→K=3，`bnn_dirichlet_bridge_k3_p0p7.pth`，
     VERIFY GOOD（MAE 0.0073）。
  5. **冒烟测试**：cliff 2 trials × 2 p 全 8 方法通过。
  6. 全量重跑启动：cliff 30 trials × 6 p（后台），bridge 100 trials 随后。
  7. 新建 `doc_tool/experiment_report.md`（论文格式：摘要/引言/背景/问题设定/
     方法含 Algorithm 1/实验/保真度声明/结论 + 算法代码位置附录）。
  8. 更新 `doc_tool/experiment_fidelity_review.md`（修正状态段落）。

## 1. 做了什么（08-07）

1. 读官方 bridge 实现确认几何（`/tmp/rats-experiments/code/envs/nsbridge_v0.py`）：
   - T[s,a,0] = 确定；向 wsat 线性插值，λ = τ·L_p / W1(w0,wsat)；
   - wsat 的 slip 质量 (1-wsat)/2 各给当前格的上/下格（**垂直**，K=3 语义）；
   - upstream（ADA-MCTS）默认 ε=0.5 → wsat=0.5 对称；L_p=1.0 下 t≥1 即饱和。
2. 修正 `grids.py`（BRIDGE_5x8 → K=3 perp；CLIFFWALKING_4x12 加 step_penalty；
   GridSpec 加 `step_penalty` 字段）。
3. 修正 `planning/rats.py` `cell_reward_of` 支持 step_penalty。
4. 修正 `run_gridworld_experiments.py`：新方法、深度参数、M_SIMULATIONS=5000、
   cell_reward、文档字符串；修复 MCTSStatic 编辑事故（act 错放）。
5. 重训 bridge K=3 预训练模型。
6. 冒烟测试通过；启动全量 cliff 重跑。

## 2. 实验 setting（修正后）

| 项 | cliffwalking | bridge |
|---|---|---|
| 网格 | 4x12 | 5x8 |
| K 方向 | K=3 垂直 [p,(1-p)/2,(1-p)/2] | **K=3 垂直 [p,(1-p)/2,(1-p)/2]**（08-07 修正，原为 K=2 掉头） |
| 原环境 p | 0.7 | 0.7 |
| 变化 | ts 0 起 p → {0.4,0.5,0.6,0.8,0.9,1.0} | 同左 |
| γ | 0.99（用户指定） | 0.99 |
| RATS 深度 | **3**（论文文档值；原 6） | 3 |
| DP 深度 | **100**（精确；原 6） | 100 |
| max_steps | 100 | 10 |
| trials | 30 | 100 |
| reward | G=+1, H=-1, **其余每步 -1**（08-07 修正） | G=+1, H=-1, 其余 0 |
| K_FORGET / counts | 3 / 默认关（08-06 定论） | 同左 |
| ADA-MCTS | **M_SIMULATIONS=5000**（upstream demo 值；30000 不可行） | 同左 |

方法集合（8 个，与 Act As You Learn 表对齐）：
- `dp_nsmdp` / `dp_snapshot` / `oracle_rats`（=RATS-P_k）/ `rats_pkminus1`（=RATS-P_{k-1}）
- `bnn_rats_static`（=RATS-ĥP_{k-1}）/ `bnn_rats_adaptive`（=FIR-RATS，本文方法）
- `ada_mcts`（论文方法，DPAS）/ `mcts_static`（=MCTS-ĥP_{k-1}）

## 3. 实验结果（修正后全量重跑）

### 3.0 决策：MCTS 预算 = 30000 simulations（论文值，用户选择方案 A）

- 30000 sims 实测：cliff 7.28s/action、bridge 1.77s/action（预热后）。
- 串行全量 cliff ≈ 40h 不可行 → **runner 并行化**（commit fe0b9f3）：
  multiprocessing (spawn) 按 (method, phase) 任务 × 16 workers；每个 worker
  自建 BNN → 保持 ada_mcts "每 phase 只 notify 一次" 不变量与逐 trial 种子
  （结果与串行逐位一致）；`torch.manual_seed(0)` 保证后验采样可复现。
- 显存实测：16 worker 启动后 GPU 占用 7.2GB / 剩 16.8GB，安全。
- 日志行加 `[phase]` 标签（并行到达顺序下 phase 归属清晰）。

### 3.1 cliffwalking（30 trials × 6 p，30000 sims）— 完成

**goal rate by p**：
```
method              p=0.4   p=0.5   p=0.6   p=0.8   p=0.9   p=1.0
dp_nsmdp             1.000   1.000   1.000   1.000   1.000   1.000
dp_snapshot          1.000   1.000   1.000   1.000   1.000   1.000
oracle_rats          0.600   0.767   0.933   1.000   1.000   1.000
rats_pkminus1        0.467   0.867   0.967   1.000   1.000   1.000
bnn_rats_static      0.467   0.867   0.967   1.000   1.000   1.000
bnn_rats_adaptive    0.733   0.800   0.967   1.000   1.000   1.000
ada_mcts             0.367   0.500   0.533   0.267   0.233   0.000
mcts_static          0.767   0.900   0.967   1.000   1.000   1.000
```
stationary（p=0.7）：全部 1.000（dp −20.61、RATS 系 −27.08、MCTS 系 −21.46）。

**要点**：FIR-RATS p=0.4 反超 oracle_rats（0.733 vs 0.600）与 static（0.467）；
mcts_static 意外强（0.767@0.4）；ada_mcts 崩（DPAS gamma=10000 病理，忠实移植，
08-06 已定位；p=1.0 0.000 vs mcts_static 1.000）。

log：`gridworld_cliffwalking_results.log`（2026-08-08 完成）。

### 3.2 bridge（100 trials × 6 p，30000 sims）— 完成

**goal rate by p**：
```
method              p=0.4   p=0.5   p=0.6   p=0.8   p=0.9   p=1.0
dp_nsmdp             0.190   0.360   0.590   0.930   0.990   1.000
dp_snapshot          0.190   0.360   0.590   0.930   0.990   1.000
oracle_rats          0.190   0.360   0.590   0.930   0.990   1.000
rats_pkminus1        0.190   0.360   0.590   0.930   0.990   1.000
bnn_rats_static      0.190   0.360   0.590   0.930   0.990   1.000
bnn_rats_adaptive    0.190   0.360   0.590   0.930   0.990   1.000
ada_mcts             0.140   0.250   0.320   0.490   0.520   0.650
mcts_static          0.070   0.160   0.250   0.550   0.770   1.000
```
stationary（p=0.7）：模型类 0.790/+0.65，ada_mcts/mcts_static 0.390/-0.22。

**要点**：全部模型类规划器数字相同（桥短、预训练模型近乎精确、RATS==DP 同
动作 → 同轨迹）；bridge 低 p 结构性无解（垂直 slip 下 p=0.4 跨最后 3 格
成功 ~0.4³，oracle 也仅 0.190）；MCTS 系系统性弱（6 步随机 rollout 叶估计
无启发式 + DPAS 病理）。

log：`gridworld_bridge_results.log`（2026-08-08 完成）。

### 3.3 全量结果总结（08-08 定稿）

- cliff：FIR-RATS p=0.4 0.733（> static 0.467、> oracle_rats 0.600），p≥0.6
  恢复 1.000；mcts_static 0.767 最强学习型基线；ada_mcts 崩（0.000@1.0）。
- bridge：全模型类相等（0.190→1.000）；FIR 无机会生效（不劣化，与 oracle 持平）。
- 结论写进 `doc_tool/experiment_report.md` §7；commit 待发。

## 4. 与论文的偏差决定（08-07 定稿）

- 对齐目标：**Act As You Learn 论文**（用户指示），非 RATS 论文。
- bridge t=0 确定斜坡：不实现（见 Progress Log #1）。
- MCTS 迭代：5000（upstream demo `act_learn.py search(5000)`；30000 在 Python
  移植上 7.5s/action，cliff 全量 ≈75h 不可行）。
- γ=0.99：任务规格，保留。

## 5. 复现命令

```bash
python pretrain_gridworld.py --grid bridge --p 0.7 --epochs 400   # bridge K=3 重训
python run_gridworld_experiments.py --grid cliffwalking --trials 30
python run_gridworld_experiments.py --grid bridge --trials 100
```

## 6. 关键文件

- `grids.py` — BRIDGE_5x8 K=3 perp（L167-186 附近）、step_penalty 字段（L49-53 附近）
- `planning/rats.py` — cell_reward_of（L98-105 附近）
- `run_gridworld_experiments.py` — 新方法 rats_pkminus1（L118-125）、mcts_static
  （L304-313 附近）、BNNRATS 自适应循环（L195-247）、argparse 深度/MCTS（L375-405）
- `doc_tool/experiment_report.md` — 论文格式实验报告（新建）
- `doc_tool/experiment_fidelity_review.md` — 保真度审查（更新修正状态）
- 预训练：`data/bridge/bnn_dirichlet_bridge_k3_p0p7.pth`（新）
