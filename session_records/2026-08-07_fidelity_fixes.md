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

[全量重跑进行中，完成后填充]

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
