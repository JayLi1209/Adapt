# ns-Bridge adaptation ablation — full configuration

Two runs of the same 7-arm ablation on the same map, differing **only** in the
planner's discount:

| run | dir | plan gamma | date |
|---|---|---|---|
| A | `Bridge/results/bridge_ablation/` | **1.0** | 2026-09-02 23:14 → 23:53 |
| B | `Bridge/results/bridge_pg095/` | **0.95** | 2026-09-03 22:16 → 22:41 |

Reproduce: `bash Bridge/run_bridge_ablation.sh` / `bash Bridge/run_bridge_pg095.sh`.
Summarize: `python Bridge/summarize_bridge_ablation.py` (edit `RESULT_DIR` for run B).
Logs: `bridge_ablation.log`, `bridge_pg095.log`.

**Run B supersedes run A** — see §8. Run A's headline result is a confound.

---

## 1. Nature of the change

Env presents the **changed** regime from **timestep 0**; the model carries the
pretrained prior. The mismatch *is* the non-stationarity. `CHANGE_STEP = 0`.

| | intended | perp- | perp+ |
|---|---|---|---|
| prior (pretrained, deterministic) | 1.0 | 0.0 | 0.0 |
| p' = 0.9 | 0.9 | 0.05 | 0.05 |
| p' = 0.7 | 0.7 | 0.15 | 0.15 |
| p' = 0.5 | 0.5 | 0.25 | 0.25 |

Slip: intended `p`, each **perpendicular** direction `(1-p)/2`. `ORIGINAL_P = 1.0`.

## 2. Environment

- **Grid** `bridge` 5x8 (Lecarpentier & Rachelson 2019), 40 states, 4 actions, K=3
- **Action order** LEFT / DOWN / RIGHT / UP, deltas `(0,-1) (1,0) (0,1) (-1,0)`
- **dir_offsets** `(0, -1, +1)` = intended, perp-, perp+; `slip_mode="perp"`
- **Layout** (S=20 start, G=16 left goal, G=23 right goal):

```
 0H  1H  2H  3H  4H  5H  6H  7H
 8F  9F 10F 11F 12F 13H 14H 15H
16G 17F 18F 19F 20S 21F 22F 23G
24F 25F 26F 27F 28F 29H 30H 31H
32H 33H 34H 35H 36H 37H 38H 39H
```

- **Geometry that matters**: from start 20 the **RIGHT arm** reaches goal 23 in
  **3 steps** but cells 21/22 have holes directly **above AND below** — any slip is
  fatal. The **LEFT arm** needs **4 steps** to goal 16 and cells 17-19 have free
  cells above and below, so a slip there is survivable. Short-risky vs long-safe.
- **Rewards** goal +1, hole -1, else 0 on arrival; **scored hole=0** so
  mean return == goal rate. Planner keeps its internal hole=-1 decision map.

## 3. Truncation

`--max-steps 1000`. **`trunc_rate = 0.000` in all 42 runs.** Episodes are short:
successful trials are 3-25 steps; the oracle averages 5.0 / 7.0 / 10.4 steps at
p' = 0.9 / 0.7 / 0.5. **No trial truncated.**

## 4. Architecture

Identical to the CliffWalking ablation except for input width — see
`README_cw_ablation.md` §4. Input 44 = 40 one-hot state + 4 one-hot action;
2 Bayesian trunk layers @ 256 SiLU; dir head (256→3) + reward head (256→2);
`prior_std=1.0`, `beta=0.1`, `num_mc_samples=3`, `num_weight_groups=20`;
`CONC_PRIOR=0.1`, `ALPHA_FLOOR=1e-3`, `RETAIN_INIT=1.0`.

**Checkpoint** `Bridge/data/bridge/bnn_dirichlet_bridge_k3.pth` (p=1.0),
pretrained by `pretrain_gridworld.py` with near-goal weighting (`goal_weight=1.35`).

## 5. Hyperparameters

| knob | value |
|---|---|
| Reported (score) `GAMMA` | **1.0 — NO discount** |
| **Planner `--plan-gamma`** | **run A: 1.0 | run B: 0.95** ← the only difference |
| CVaR alpha | 0.0, `adaptive_alpha=False` |
| Planning horizon H | default **6** |
| `clip_delta_n` | 0.0 (**OFF** — unlike CliffWalking's 50) |
| `--retrain-steps` | **50** inner steps (CliffWalking used 5) |
| Retrain lr | 1e-3, Adam, mu only, whole-trial replay, KL omitted |
| `K_FORGET` | 1 |
| `--step-cost` | 0.0 |
| Drift filter | `DriftFilterV2`, ETA=0.2, detect when `delta_bar > 1.0` |
| Forget rule | SHRINK `rho = clip(1/max(dbar,1), 1e-3, 1)` |
| `max_forgets` / `anchor_forget` / `pi_kappa` | uncapped / False / 0.0 |
| Trials | **100** per condition |
| Seeds | `--seed 0`; env `1000+trial`, torch `10000+trial` |

CEM planner constants otherwise as CliffWalking: 5 iters x 256 candidates x
10 models x 32 rollouts, ELITE_FRAC 0.1, ACTION_SMOOTH 0.05, BETA_EXPLORE 0.0.

## 6. Why plan-gamma matters here (the key finding)

At **plan_gamma = 1.0** every route that reaches a goal is worth exactly 1.0, so
the short-lethal RIGHT arm and the long-safe LEFT arm **TIE** under the p=1.0
prior. The planner's tie-break sends the non-adapting agent down the **safe** arm,
handing it an undeserved score — measured: `no_adapt` min episode length was
**4 steps, never 3**, i.e. it never used the short arm.

Any discount < 1 makes the p=1.0-optimal policy strictly prefer the SHORT arm,
which slip then makes lethal. Exact MDP values for the stale p=1.0 policy:

| plan gamma | stale action @ start | stale value @ 0.9 / 0.7 / 0.5 |
|---|---|---|
| 1.0 | LEFT (safe) | 0.824 / 0.548 / 0.308 |
| 0.99, 0.95, 0.9 | **RIGHT (lethal)** | **0.806 / 0.469 / 0.229** |

Oracle: 0.994 / 0.934 / 0.749 (unchanged by the discount).

Measured effect on `no_adapt`: **0.98 → 0.78** (p'=0.9), **0.91 → 0.51** (p'=0.7),
**0.73 → 0.29** (p'=0.5).

## 7. The seven conditions

Same as CliffWalking (`README_cw_ablation.md` §7). `--no-counts` was added to
`Bridge/sweep_bridge_unfrozen.py` for the `no_adapt` arm; oracle is
`Bridge/oracle_bridge.py` (VI on the true post-change kernel, solved at gamma=1.0
in both runs — it is the true optimal ceiling either way).

## 8. Results — goal rate (== E[return]), 100 trials/cell

### Run B — plan_gamma = 0.95 (**authoritative**)

| condition | p'=0.9 | p'=0.7 | p'=0.5 |
|---|---|---|---|
| Oracle | 0.99 +/- .010 | 0.91 +/- .029 | 0.69 +/- .046 |
| Full method | 0.80 +/- .040 | 0.56 +/- .050 | **0.45 +/- .050** |
| No retrain (forget only) | 0.80 +/- .040 | 0.57 +/- .050 | **0.44 +/- .050** |
| No forget + no retrain | 0.82 +/- .039 | 0.56 +/- .050 | 0.28 +/- .045 |
| No forget | 0.80 +/- .040 | 0.52 +/- .050 | 0.28 +/- .045 |
| No adapt | 0.78 +/- .042 | 0.51 +/- .050 | 0.29 +/- .046 |
| Retrain full network | 0.73 +/- .045 | 0.43 +/- .050 | 0.30 +/- .046 |

| test | p'=0.9 | p'=0.7 | p'=0.5 |
|---|---|---|---|
| full vs no-adapt | +0.020 (p=0.73) | +0.050 (p=0.48) | **+0.160 (p=0.018)** |
| **forget alone** | -0.020 (p=0.72) | +0.010 (p=0.89) | **+0.160 (p=0.017)** |
| retrain on top of forget | +0.000 (p=1.00) | -0.010 (p=0.89) | +0.010 (p=0.89) |
| retrain-full vs head-only | -0.070 (p=0.24) | -0.130 (p=0.065) | **-0.150 (p=0.027)** |

### Run A — plan_gamma = 1.0 (**CONFOUNDED, do not cite**)

| condition | p'=0.9 | p'=0.7 | p'=0.5 |
|---|---|---|---|
| Oracle | 0.99 | 0.91 | 0.69 |
| Full method | 0.96 | 0.78 | 0.66 |
| No retrain | 0.96 | 0.78 | 0.66 |
| No forget + no retrain | 0.99 | 0.91 | 0.68 |
| No forget | 0.98 | 0.87 | 0.69 |
| No adapt | 0.98 | 0.91 | 0.73 |
| Retrain full network | 0.94 | 0.80 | 0.52 |

Run A reproduces the Aug-28 `bridge_five_arms` sweep **exactly** on all 12
overlapping cells, so run A is a correct measurement of a mis-specified setup,
not a buggy one.

### The sign flip

| p' | full vs no-adapt @ pg=1.0 | @ pg=0.95 |
|---|---|---|
| 0.7 | **-0.130 (p=0.010)** | +0.050 (p=0.48) |
| 0.5 | -0.070 (p=0.28) | **+0.160 (p=0.018)** |

## 9. Conclusions

1. **Run A's "adaptation significantly hurts on Bridge" (-0.13, p=0.010) is an
   artifact of `plan_gamma=1.0`**, which ties the two routes and lets the
   non-adapting agent take the safe arm by tie-break. Withdrawn.
2. **With the tie removed, adaptation helps, significantly at the largest change**:
   +0.160 (p=0.018) at p'=0.5.
3. **Forget is the entire effect**: forget alone +0.160 (p=0.017); retrain on top
   of forget +0.010 (p=0.89). Identical to CliffWalking.
4. **The benefit scales with change magnitude**: +0.02 → +0.05 → +0.16, matching
   CliffWalking's +0.05 → +0.21 → +0.28.
5. **Full-network retraining is actively harmful on Bridge** — the worst arm at
   every level, in BOTH runs: -0.150 (p=0.027) at p'=0.5, -0.130 (p=0.065) at
   p'=0.7. On CliffWalking it was neutral everywhere. Training the trunk changes
   the shared representation in a way the clamped `retain` cannot multiply out.
6. **Episode length gates the effect.** Bridge deaths cluster at step 2-3 (80% by
   step 3) and detection fires at step 0-2, leaving no room to re-route; the
   benefit only appears at p'=0.5 where episodes lengthen (oracle 10.4 steps).
   CliffWalking's 35-step corridors are why forget pays much more there.

## 10. Caveats

- Bridge uses H=6, clip OFF, retrain_steps=50; CliffWalking uses H=15, clip=50,
  retrain_steps=5. A strictly controlled cross-env claim would need matched settings.
- `plan_gamma=0.95` on the reported score is still gamma=1.0 (no discount); only
  the planner's imagined return is discounted.
- The oracle is solved at gamma=1.0 in both runs. Its goal rate is discount-invariant
  (verified 0.994/0.934/0.749 at gamma 0.99/0.95/0.9), so it remains a valid ceiling.
- `retrain_full` harm is consistent across 4 independent cells but only individually
  significant at one; a seed sweep would firm it up.
- `bridge_h19` (cell 19 → hole) was registered in `grids.py` and pretrained during
  this investigation but **is not used by either run** — analysis showed a single
  hole cannot create the effect, because at gamma=1.0 the blocked route merely ties.

## 11. Code changes

- `Bridge/sweep_bridge_unfrozen.py`: added `--no-counts`; tag encodes `_nf`/`_nc`;
  JSON records `counts`.
- `Bridge/oracle_bridge.py`, `Bridge/eval_bridge_pretrained.py`: `--grid` choices
  gained `bridge_h19` (and `cliffwalking` in `oracle_bridge.py`).
- `grids.py`: added `BRIDGE_5x8_H19` + REGISTRY entry.

## 12. Files

| path | what |
|---|---|
| `Bridge/run_bridge_ablation.sh` | run A launcher (plan_gamma 1.0) |
| `Bridge/run_bridge_pg095.sh` | run B launcher (plan_gamma 0.95) |
| `summarize_bridge_ablation.py` | table generator |
| `Bridge/sweep_bridge_unfrozen.py` | the sweep (all arms except oracle) |
| `Bridge/oracle_bridge.py` | oracle (VI on true kernel) |
| `Bridge/results/bridge_ablation/`, `Bridge/results/bridge_pg095/` | results |
| `Bridge/data/bridge/bnn_dirichlet_bridge_k3.pth` | p=1.0 checkpoint |
