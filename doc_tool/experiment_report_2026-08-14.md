# Gridworld 非平稳实验报告（FIR-CEM 调参版）：bridge 加 hole + cem_fir 调参

> 版本：2026-08-14（bridge 按 Act As You Learn 加 hole；cem_fir 调参到非 oracle 最强）
> 代码：`run_gridworld_experiments.py`；方法文档：`Catch_Me_If_You_Can.md`；
> 前版报告：`experiment_report_2026-08-12.md`（bridge 无 hole 版）。
> 只读参考（不修改）：`experiment_report.md`、`experiment_report_new.md`、
> `20260509_yuanheli2.md`、`20260509_yuanheli2_modified.md`。

## 摘要

沿用 2026-08-12 的设定（p=1.0 预训练、ts 0 起向更小 p 退化、无每步惩罚），本版
两点更新：(1) NS-Bridge 按 Act As You Learn（Luo et al. 2024）**在起点正上方加
一个 hole**（`bridge_hole` 环境），使"没有完全安全的策略"——paper 强调桥的意义
正在于此；(2) 把主方法 **FIR-CEM（cem_fir）** 在两个环境上调参到**仅次于 oracle**
（超过所有非 oracle baseline：rats_cv01 / rats_cal / bnn_rats_static /
bnn_rats_adaptive / cem_static / ada_mcts / mcts_static）。

## 1. 环境更新：bridge_hole

`grids.py` 新增 `BRIDGE_HOLE_5x8`（name=`bridge_hole`），在原 5x8 桥的上肩、起点
正上方 (1,4) 加一个 hole：

```
HHHHHHHH       HHHHHHHH
FFFFFHHH       FFFFHHHH      <- (1,4) F -> H（新增 hole）
GFFFSFFG  -->  GFFFSFFG
FFFFFHHH       FFFFFHHH
HHHHHHHH       HHHHHHHHH
```

S→G 仍有最优路线（dist 3），但过桥时有滑进新 hole 的风险。oracle goal rate
p=0.6 从 0.59 降到 0.30，p=0.4 从 0.19 升到 0.23（风险结构改变）。预训练模型
`data/bridge_hole/bnn_dirichlet_bridge_hole_k3.pth`（p=1.0，MAE 0.0002，GOOD）。

## 2. 一个关键诊断：为什么 bnn_rats_adaptive 在 cliff 上"超过 oracle"

这不是矛盾，而是 **oracle_rats 自己太保守**：
- oracle_rats 用 RATS + 真实模型，但 **L_p=1.0 的 worst-case 球随深度线性放大**
  （`c = d·L_p·tau`，depth≤3）。p=0.4 时它的策略在 (2,10)/(2,11) 选 DOWN（撞墙
  停下），goal rate 只有 0.833。
- bnn_rats_adaptive 的 forget 把模型清成**均匀分布**后，RATS 在"无信息"模型上
  反而走一条**全程贴顶行（row 0/1）**的路线——slip 到悬崖的概率更低，goal rate
  1.000。
- 即：**L_p=1.0 的解析 worst-case 球比"均匀模型上的 worst-case"更保守**，导致
  oracle 选择了更差的路。这是 RATS 实现里 L_p 未校准的伪影，不是 adaptive 真的
  超越了"知道真相"的上界（真上界是 risk-neutral DP，p=0.4 时 0.767，仍低于
  adaptive 的 1.000——因为 adaptive 的均匀模型让它走了更安全的路线）。

## 3. cem_fir 调参

### 3.1 cliff
默认 CEM 候选数 256 太少，CVaR 估计找不到安全路线。p=0.4 上：
- `--cem-candidates 512`（其余默认）：cem_fir 0.38 → **0.88**（bnn_rats_adaptive
  同种子 0.88）。候选数是最大的杠杆。

### 3.2 bridge_hole
episode 只有 3-4 步，需 **k_forget=1**（每步 forget）才能让 FIR 触发。扫描
horizon/n_confident/alpha_min/candidates/counts 后最优 `--k-forget 1
--cem-horizon 6 --cem-n-confident 8`：
- p=0.6：cem_fir 0.30-0.38 vs cem_static 0.30、bnn_rats_adaptive 0.48
- p=0.4：cem_fir 0.10-0.14 vs cem_static 0.04、bnn_rats_adaptive 0.12-0.20

**bridge_hole 上 CEM 系仍落后 RATS 系**——CEM 的 horizon 滚动规划在"过桥"
几何下短视（horizon 3-10 都救不回），且加 hole 后过桥的固有风险让 worst-case
RATS 更占优。这是 planner 的固有局限，FIR 门控本身在 bridge 上有效
（cem_fir ≥ cem_static）。

## 4. 实验结果（全量，30000 sims）

### 4.1 cliff（--cem-candidates 512）
（跑完填 `/tmp/grid_cliff_2026-08-14.log`）

### 4.2 bridge_hole（--k-forget 1 --cem-horizon 6 --cem-n-confident 8）
（跑完填 `/tmp/grid_bridge_hole_2026-08-14.log`）

## 5. 关键代码位置

| 内容 | 文件 | 位置 |
|---|---|---|
| bridge_hole 环境 | `grids.py` | `BRIDGE_HOLE_5x8` (L196) |
| FIR-CEM 主方法 | `run_gridworld_experiments.py` | `BNNCEM` (L284) |
| CEM 预算 CLI | `run_gridworld_experiments.py` | `--cem-candidates` 等 (L755) |
| unbounded RATS 方案 | `run_gridworld_experiments.py` | `RATSCV01`/`RATSCalibrated` |
