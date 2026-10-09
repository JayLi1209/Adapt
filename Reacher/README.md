# Reacher — Gaussian-head BNN world model vs. a true-simulator oracle

Self-contained replication package. Everything needed to pretrain the model and
regenerate the tables below is in this folder; it has no dependency on the
parent repo.

The question this package answers: **how much return does a learned Bayesian
world model give up against an oracle that plans with the true simulator, on the
environment it was pretrained on?**

Both planners are the *same* CEM — same horizon, candidate count, iterations,
elite fraction, MPC warm start, discount. The only substitution is that every
imagined step calls MuJoCo instead of the BNN. Any gap is therefore attributable
to world-model error alone, not to a better or worse search.

---

## 1. Provenance

| | |
|---|---|
| git base commit | `53136ce` (working tree dirty) |
| pretraining driver | `pretrain_reacher.py` |
| audit / report driver | `test_reacher_pretrain_quality.py` |
| oracle planner | `planning/oracle_cem.py` |
| checkpoint | `data/reacher/bnn_dynamics.pth`, md5 `848b05f6c34e197f87fc50d6c597008f` |
| normalizer | `data/reacher/env_stats.pickle`, md5 `ac030121621027220880de854cae4d03` |
| external deps | `torch`, `numpy`, `gymnasium`, `mujoco`, `mbrl` |

### Layout

```
Reacher/
  README.md                          this file
  pretrain_reacher.py                collection + training driver
  test_reacher_pretrain_quality.py   the 4-axis audit (incl. the oracle table)
  config.py                          device selection
  bnn/      layers.py  gaussian_model.py  gaussian_workflow.py
  env/      reacher.py               non-stationary Reacher + manifold projection
  planning/ continuous_cem.py        BNN-model CEM (+CVaR)
            oracle_cem.py            true-MuJoCo CEM (the oracle)
  data/reacher/                      bnn_dynamics.pth, env_stats.pickle,
                                     arch.json, audit.json
  results/                           the four run logs behind §3-§5
```

Reproduce:

```bash
cd Reacher
python3 pretrain_reacher.py                    # ~6 min on one GPU
python3 test_reacher_pretrain_quality.py       # ~50 min (100 trials x 7 arms)
python3 test_reacher_pretrain_quality.py --quick   # smoke test, ~4 min
```

---

## 2. Experiment setting

1. **Nature of the change.** None — this is the *stationary* pretraining
   benchmark. The model is trained and evaluated on stock Reacher-v5
   (gear ×1.0, link mass ×1.0, damping ×1.0).
   `env/reacher.py` ships `gear_schedule` / `mass_schedule` /
   `damping_schedule` for the follow-on non-stationary work; gear ×0.25 is the
   exact analogue of the pendulum's mass 1→4, since pendulum dynamics enter as
   `3u/(m l²)`, so mass ×4 *is* torque authority ÷4.
2. **Architecture.** hid=256, num_layers=4
   (3 hidden), SiLU trunk, mean-field-VI Bayesian linear
   layers, single Gaussian (mean, logvar) head over 11 target dims
   (`[Δobs(10), reward]`); 281,154 params. Observation is 10-dim, action 2-dim.
3. **Hyperparameters.** discount **γ = 1.0 (no discount)**; CVaR α = 0.8 with
   K = 3 posterior draws; β-NLL = 0.5 (Seitzer et al. 2022);
   KL budget 7.8; output scaler on; Adam lr 1e-3 cosine-annealed to 1%,
   grad-norm clip 5.0; 100,000 transitions, 1200 epochs,
   batch 2048. Planner: H=15, J=500, 8 CEM iters, elite frac 0.1.
4. **Truncation.** Collection runs with the TimeLimit **disabled**
   (`max_episode_steps=1e9`) — no transition is ever cut short. Evaluation uses
   Reacher-v5's native **50-step** episode: 50 steps *is* the task definition
   (the target re-randomizes each reset), not an imposed truncation, and it
   keeps returns comparable to published SAC/TD3 numbers. 100 trials, error
   bars are ±1 SE.
5. **Reward structure.** `r_t = −‖fingertip − target‖ − ‖a‖²`, evaluated
   **after** the transition, i.e. `r_t = f(s_{t+1}, a_t)`. Range ≈ [−2.4, 0].

---

## 3. Headline: return vs. the oracle

100 trials × 50 steps, identical CEM budget (H=15, J=500, 8 iters, γ=1.0).

| Planner | Return | Final dist | Last-10 dist |
|---|---|---|---|
| **Oracle CEM** (true MuJoCo dynamics) | **-3.449 ± 0.125** | 0.0011 | 0.0014 |
| **BNN CEM+CVaR, learned reward** (default) | **-4.484 ± 0.191** | 0.0170 | 0.0181 |
| BNN + analytic reward *(ablation)* | **-3.521 ± 0.127** | 0.0014 | 0.0018 |
| BNN + aleatoric noise *(ablation)* | **-4.491 ± 0.186** | 0.0162 | 0.0173 |
| BNN, no manifold projection *(ablation)* | **-4.517 ± 0.180** | 0.0156 | 0.0172 |
| zero action (arm holds still) | **-11.658 ± 0.445** | 0.2331 | 0.2331 |
| random actions | **-42.400 ± 0.388** | 0.1777 | 0.1721 |

**Deficit vs oracle = 1.036 return (30.0% of |oracle|).**

The BNN recovers **97.3%** of the oracle-over-random range —
but random actions are a *flattering* floor here, because random torques mostly
burn control cost (−‖a‖² averages ≈ −0.67/step), so most of that range is won
just by not thrashing the actuators. Against the honest floor of zero action —
the arm holds still and pays only the distance term — the BNN recovers
**87.4%**.

---

## 4. The gap is the reward head, not the dynamics

Swapping *only* the reward source recovers -3.521,
statistically level with the oracle's -3.449. The reward-channel
decomposition explains why — all three numbers come from the **same** network,
scored against the env's true reward on held-out data:

| reward source | RMSE |
|---|---|
| (a) learned head (the 11th output channel) | **0.03681** |
| (b) analytic, on the model's own predicted next state | **0.00148** |
| (c) analytic, on the current state (1-step offset) | 0.01973 |

Reacher's reward depends on the fingertip–target distance *after* the
transition, so the learned head must perform a one-step prediction **and** a norm
inside a single scalar output. Its 0.037 error exceeds the ≈0.03 of distance the
planner must resolve in the final approach, so it cannot rank a near-target
candidate against an on-target one. Route (b) is **25× sharper from the same
weights** — the dynamics are near-oracle; only the reward channel is not.

---

## 5. Model quality

One-step, fresh held-out split (collection seed 777, *not* the seed the
checkpoint early-stopped on):

| | NRMSE | R² | z_std | 1σ cov | 2σ cov |
|---|---|---|---|---|---|
| all states | 0.0637 | 0.9934 | 0.65 | 88.6% | 98.3% |
| goal region (d < 0.03) | 0.0475 | 0.9969 | 0.42 | 96.0% | 99.8% |
| off-goal | 0.0656 | 0.9929 | 0.69 | 86.8% | 97.9% |

The goal region is **1.38× more accurate** than off-goal, so the
CLAUDE.md goal-prioritised pretraining did buy accuracy where the planner lives.

**Known limitation — calibration.** z_std = 0.65 (passes the 0.5–2.0 band, but the
model is *under-confident*: σ runs ≈1.5× large, 1σ coverage 88.6% against a nominal
68%). The online surprise score is a Mahalanobis z, so drift thresholds must be
recalibrated on this checkpoint rather than inherited from the pendulum's.

---

## 6. Design notes worth keeping

* **IK-teleport collection.** Random actions almost never put the fingertip on
  the target (1.6% of steps inside d<0.03). Because the target *is* part of
  `qpos`, the collector samples the target from the env's own distribution
  first, then solves the analytic two-link IK for joint angles placing the
  fingertip a controlled distance away. This keeps the target marginal exactly
  on-distribution (mean radius 0.1327 vs the theoretical 0.1333) while raising
  goal coverage to **20.1%**, a 12.6× improvement.
* **Manifold projection** (`env/reacher.py: project_reacher_obs`). Dims 8:10 are
  an exact function of dims 0:6 through forward kinematics, verified against
  MuJoCo to 1.4e-6 median. Re-imposing it each imagined step cuts the
  reward-carrying dims' H=15 rollout error from 0.0269 to 0.0227 (−16%), though
  it does not move end-to-end return measurably at this budget.
* **joint1's limit is *soft*.** MuJoCo lets it overshoot ±3 to ≈3.43, and the
  (cos, sin) encoding only spans (−π, π] anyway — so a clamp at the nominal
  limit corrupts genuine states while a clamp wide enough to be safe can never
  bind. The projection therefore does **not** clamp theta1.
* **400 epochs was badly undertrained** (NRMSE 0.137 vs 0.064 at 1200; reward
  RMSE 0.064 → 0.037; z_std 0.42 → 0.55). Default is now 1200.
* **β-NLL = 0.5 beats plain NLL** (β=0) on *both* accuracy (NRMSE 0.137 vs
  0.191 at matched 400 epochs) and calibration — so β-NLL is not the source of
  the residual under-confidence. See `results/pretrain_reacher_betanll0.log`.
* **Aleatoric injection is harmless here**, unlike on unstable plants: Reacher
  is heavily damped, so per-step noise does not compound (−4.491 vs −4.484).

---

## 7. Run logs

| file | what |
|---|---|
| `results/pretrain_reacher_1200ep.log` | the shipped checkpoint |
| `results/pretrain_reacher_400ep.log` | undertrained baseline |
| `results/pretrain_reacher_betanll0.log` | β-NLL = 0 ablation |
| `results/audit_reacher.log` | the full 100-trial audit behind §3–§5 |
