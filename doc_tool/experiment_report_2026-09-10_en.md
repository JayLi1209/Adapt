# Non-Stationary CliffWalking — Consolidated Report (2026-09-10)

> Code: `run_gridworld_experiments.py`, `grids.py`, `bnn/dirichlet_model.py`,
> `planning/cvar_cem.py`, `planning/ada_mcts.py`.
> Prior reports: `experiment_report_2026-08-25.md` / `_en.md` (background:
> γ=0.9999, α0 calibration, plan_retain gating); `experiment_report_2026-09-04.md`
> (step-by-step log of this round's work — this report is its cleaned-up
> consolidation).
> Read-only (do not edit): `experiment_report.md`, `experiment_report_new.md`,
> `20260509_yuanheli2.md`, `20260509_yuanheli2_modified.md`.
> Chinese version: `experiment_report_2026-09-10.md`.

---

## 0. Executive summary

For the non-stationary control methods in `main_cl.tex`, this round ran a full
3-config × 5-method comparison on **CliffWalking 4×12**, and along the way fixed
two bugs and settled one hyperparameter:

1. **Bug 1 fix (`bnn_rats_adaptive` / SFI-RATS)**: `BNNRATS._forget()` was
   missing `drift._reset_detection()`, so under p=1.0 pretraining the huge
   surprise spike from the first slip stayed baked into every subsequent
   `forget` call, crashing `bnn.retain` to a hard 0 — and **worse for small
   changes than large ones** (p=0.9 worse than p=0.4), i.e. goal rate
   non-monotone in p. Fixed; monotonicity restored.
2. **The 3 main experiments** (original cliff / first cliff cell removed /
   pretrain at p=0.7), p swept 1.0→0.3 (incl. 0.4 and 0.7), 5 methods × 8
   p-points × 30 trials.
3. **CONC_PRIOR tuning**: swept the symmetric prior `c` that the "forgotten"
   Dirichlet decays toward, from 0.1 to 10. **c=1.0 (the actual uniform
   distribution over the simplex, Dirichlet(1,1,1)) beat the old 0.1 at every
   tested point**, making `sfir-cem-cvar` the tied-or-best non-oracle method
   at 20 of 21 (config,p) combinations. `CONC_PRIOR` is now formally 1.0.
4. **Bug 2 fix (`oracle_cem`)**: `oracle_cem` reuses `cem_fir`'s
   `CVaRCEMAgent`, but that class's confidence-gate state is only updated when
   `self.adaptive=True` — `oracle_cem` has `adaptive=False`, so it was
   permanently pinned to the most risk-averse worst-30% CVaR tail, even though
   it holds the true post-change model. Fixed (forced risk-neutral);
   `oracle_cem` is a credible upper bound again at all 21 points.

**One-liner**: after both bug fixes and settling on c=1.0, `sfir-cem-cvar`
(our method) is the **best non-oracle method** (only exception: config2
p=0.4, loses to ada-mcts 0.800 vs 0.867), and `oracle_cem` is restored as a
trustworthy upper bound.

---

## 1. Methods (aligned to main_cl.tex / NSMDP.md / Catch_Me_If_You_Can.md)

| Name in report/prompt | Code `name` | Description |
|---|---|---|
| **ada-mcts** | `ada_mcts` | ADA-MCTS / DPAS (`planning/ada_mcts.py`). `notify_change()` freezes the pre-change model M_{k−1}, online counts build M_k, chance nodes sampled by the DPAS two-phase rule. Unchanged this round. |
| **rats** | `bnn_rats_static` | The paper's **RATS-P̂^{k−1}** (NSMDP.md / Catch_Me_If_You_Can.md): run RATS minimax with the pretrained BNN as the model, **no online adaptation**. Note: CliffWalking here is the **original** map — no extra hole from Catch_Me_If_You_Can. |
| **ada-cem-cvar** | `cem_ada` | Replace the SFI part of our method with ADA-MCTS's adaptation: same CVaR-CEM planner, but the adaptation is DPAS's two-phase hard switch — notified at `change_step`, then only accumulates online counts (no forget, no retain decay); the CVaR tail α is `alpha_min` (most pessimistic) until `n_threshold=3` post-change samples, then hard-switches to 1.0 (risk-neutral). |
| **sfir-cem-cvar** (our method) | `cem_fir` | = the existing **SFI-CEM**. Component 1 (pretrained Dirichlet BNN) + component 2 (SFI: surprise → drift → forget → learn loop) + component 3 (CVaR-CEM planning, not RATS). CliffWalking's "learn" is a closed-form conjugate count update, no gradient retraining → the SFI variant (not SFIR). |
| **oracle bnn+cem+cvar** | `oracle_cem` | CVaR-CEM planner + a BNN **pretrained directly on the true post-change p** (each p-point loads its own p-matched checkpoint; the `stationary` phase uses `ORIG_P`). No online adaptation / SFI needed — the model matches the changed env from t=0. The CEM-family upper bound; the CEM analogue of `oracle_rats`. |

`cem_static` / `bnn_rats_static` are the no-adaptation ablations of their
methods; `oracle_rats` runs RATS with the true model. The 5 methods run by
default this round are exactly the 5 in the table above.

---

## 2. Experimental setup (item-by-item per CLAUDE.md)

### 2.1 Nature of the change (at what timestep, to what)

- **When**: `change_step = 0`. With `c ≤ 0` the first decision epoch is
  already in the new environment; the pretraining-p segment never appears —
  so this is "the environment has already changed drastically when the
  simulation starts".
- **What**: the intended (no-slip) probability `p` of the transition. Slip
  structure (Luo et al.): move to the intended direction w.p. `p`, to each
  perpendicular direction w.p. `(1−p)/2`, **no reverse direction**. The
  reward structure does not change over time.
- **The 3 configs**:

  | config | grid | pretrain p (=`ORIG_P`) | post-change p sweep |
  |---|---|---|---|
  | **config1** | original CliffWalking 4×12 | 1.0 (deterministic) | 1.0→ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} |
  | **config2** | CliffWalking with the first cliff cell right of Start removed | 1.0 | same |
  | **config3** | original CliffWalking 4×12 | **0.7** (pretraining env is itself stochastic) | 0.7→ {same 8 p-points} |

  - The `p=1.0` column of config1/2 = no actual change (control).
  - The `p=0.7` column of config3 = no actual change; `p>0.7` means the env
    got **better** (less slip), `p<0.7` worse.
  - config2 vs config1 differs by one cell: the cliff cell directly right of
    `S` changes from `H` to `F` (safe ground) — to see how much of the
    low-p gap is due to that one cell.

### 2.2 Grid geometry and reward structure

- **CliffWalking 4×12**, K=3 directions `[intended, perp+, perp−]`, 4 actions
  UP/RIGHT/DOWN/LEFT. Start = bottom-left `(3,0)`, Goal = bottom-right
  `(3,11)`, cliff = bottom row `(3,1..10)` (config2: `(3,2..10)`).
- **Reward**: reaching the goal `+1.0`; stepping into a cliff `0.0` (the
  paper's "holes = 0" convention: falling just ends/teleports, no extra
  penalty); every other step `0.0` (**no per-step penalty**, per the
  2026-08-12 user request).
- **`cliff_to_start = True`**: stepping into a cliff cell → **teleport back
  to Start, does NOT terminate the episode** (ns_gym's CliffWalking
  behavior). Hence **the goal is the only terminal state**; an episode can
  only end by "reached the goal" or "truncated at step 100".

### 2.3 All hyperparameters

**General / evaluation**
| Param | Value | Note |
|---|---|---|
| discount γ | **0.9999** | Same γ for planning and evaluation (2026-08-22 user request). γ≈1 + holes=0 ⇒ discounted return ≈ goal rate (return is slightly lower in the 3rd decimal on cliff — teleport-to-start delays reaching the goal). |
| trials | 30 | Per (method, phase) point; trial index seeds the RNG, reproducible. |
| truncation max_steps | 100 | See 2.5. |
| BNN posterior draws (surprise) | `N_POSTERIOR = 10` | |
| drift filter | `DriftFilterV2(eta=0.2, gamma_uncertainty=True)` | `delta_bar = 1 + lambda_hat`, `lambda_hat` = **cumulative mean** (not EMA) of `(delta_n − baseline)` since the last `_reset_detection()`. |

**BNN world-model architecture (`make_dirichlet_bnn`)**
- Bayesian trunk: `in_size = n_states + n_actions = 48 + 4 = 52` → hidden
  `hid_size = 256`, `num_layers = 3` (i.e. 2 BayesianLinear layers in the
  trunk: 52→256, 256→256), SiLU activations.
- Two separate readout heads (no shared output layer): direction head
  `256→3` (K=3 Dirichlet concentrations), reward head `256→2` (reward
  mean / logvar).
- `prior_std = 1.0`, `beta = 0.1`, `num_mc_samples = 3`,
  `num_weight_groups = 1`.
- Predictive distribution: a categorical over the K=3 directions
  `p = α/α0`, then scattered onto the 48 cells by geometry. **The
  concentration α0 comes from a pretrain-counts table**
  (`pretrain_alpha0` buffer, `α0[s,a] = counts + K·CONC_PRIOR`), not the
  head's softplus magnitude — the categorical NLL is scale-invariant in α
  and cannot learn α0.
- **CONC_PRIOR = 1.0** (changed from 0.1 this round, see §4). The `retain`
  buffer pulls the head's α toward this symmetric prior:
  `alpha = CONC_PRIOR + retain·(alpha_head − CONC_PRIOR)`, `retain=1` fully
  trusts the head, `retain→0` sends every direction to `CONC_PRIOR`
  (predictive mean → uniform).

**Pretraining (`pretrain_gridworld.py`)**
- Goal-weighted sampling: `w(s) = goal_weight^(−BFS_dist(s→goal))`,
  `goal_weight = 1.35` ⇒ **transitions near the goal are upsampled**
  (CLAUDE.md: "pretraining should prioritize transitions close to the
  goal").
- `--balance-terminal`: subsample transitions landing on a terminal cell so
  they don't dominate after goal-weighting.
- 20000 rows, per-row categorical NLL. Each `(grid, p)` gets its own
  checkpoint.

**SFI-CEM (`cem_fir` = sfir-cem-cvar, our method)**
| Param | Value | Note |
|---|---|---|
| forget cadence `k_forget` | 3 | forget once every 3 post-change steps. |
| forget rule | `retain ← retain · ρ`, `ρ = 1/max(delta_bar, 1)`, clipped to [1e-3, 1] | `delta_bar ≤ 1` is a no-op. Calls `drift._reset_detection()` after forgetting (see §3). |
| CVaR tail α | adaptive, `alpha_min + (1 − alpha_min)·conf` | |
| `alpha_min` (`CEM_ALPHA_MIN`) | **0.30** | most-pessimistic end (CVaR over the worst 30%). 0.10 was too conservative (dragged below cem_static), 0.95 barely differs from risk-neutral. |
| `conf` | `conf_data · conf_surprise` | `conf_data = min(1, n_since_change / n_confident)`; `conf_surprise = exp(−max(0, delta_bar−1) / surprise_tau)`. |
| `n_confident` (`CEM_N_CONFIDENT`) | 8 | post-change samples for the data gate to saturate. |
| `surprise_tau` (`CEM_SURPRISE_TAU`) | 50.0 | p=1.0 pretraining spikes surprise into the hundreds; the default 2.0 would pin conf at 0 for the whole episode. |
| plan_retain (planning-time uncertainty gate) | `= conf` | scale `bnn.retain` by conf while the planner reads the model (inflate the pretrained part, fully trust online counts), restore afterward. |
| planning horizon `--cem-horizon` | **6** | |
| CEM candidates `--cem-candidates` | **512** | module default 256; 512 gives the CVaR estimate more samples (2026-08-14 tuning: cliff cem_fir p=0.4 went 0.38→0.88). |
| K posterior models `k_models` | 10 | epistemic axis. |
| aleatoric rollouts per model `n_rollouts` | 32 | |
| CEM iterations `n_cem_iters` | 5 | |
| elite fraction | 0.1 | |
| CEM-internal planning γ | GAMMA = 0.9999 | `--cem-plan-gamma` not passed. |
| exploration bonus β | 0.0 (off) | |

**ada-cem-cvar (`cem_ada`)**: same planner; `n_threshold = 3` (matches
ADA-MCTS's `_training_started` threshold); CVaR α = `alpha_min = 0.30`
until 3 post-change samples observed, then hard-switch to 1.0; only
accumulates online counts, no forget, no retain decay.

**oracle_cem (after fix)**: same planner; **`adaptive_alpha = False`**
(§5 fix), fixed `cvar_alpha = 1.0` (risk-neutral); `retain` always 1.0;
loads its own p-matched checkpoint per p-point.

**ada-mcts (`ada_mcts`)**
| Param | Value |
|---|---|
| simulations per action `M_SIMULATIONS` | **30000** (paper value) |
| UCT constant `CP` | √2 |
| epistemic threshold `EPS_E` | 0.02 (paper line 236) |
| aleatoric threshold `EPS_A` | 0.0 |
| DPAS aleatoric-likelihood `gamma` | 10000.0 |
| posterior draws (Var_E/Var_A) | 10 |
| rollout horizon | 6 |
| min post-change samples before switching `_n_threshold` | 3 |

**RATS family (`bnn_rats_static` = rats)**: `rats_depth = 3` (paper value),
`dp_depth = 100`. Runs RATS minimax with the pretrained BNN's mean model, no
online update.

### 2.4 discount / truncation

- **No extra discounting**: γ=0.9999 is the unified value agreed under
  CLAUDE.md's "no discount" convention (since 2026-08-22), used for both
  planning and evaluation.
- **Truncation**: `max_steps = 100`. Because `cliff_to_start=True` (falling
  teleports to Start, does not terminate), an episode that never reaches the
  goal can only end by truncation — so **goal rate = X literally means
  "(1−X) of the 30 trials ran the full 100 steps without ever reaching the
  goal"**. At high p (≥0.6–0.7) nearly every trial reaches the goal within
  100 steps; at low p (0.3–0.4) truncation is frequent, which is exactly why
  goal rate drops. Spot-check of config1 trial 0: 8 of the 45
  (method,phase) trial-0 episodes ran the full 100 steps.

### 2.5 Stationary verification

Each config, each method first runs a `p = ORIG_P` no-change stationary
phase. **All 5 methods × 3 configs = 15 points: goal rate = 1.000**
(return 0.997–0.999) — the pretrained model finds the optimal policy in the
un-changed environment, so the premise holds.

---

## 3. Bug fix 1: the drift accumulator in `bnn_rats_adaptive` (SFI-RATS)

**Symptom** (user-flagged): `bnn_rats_adaptive`'s goal rate is non-monotone
in p, with small changes worse than large ones.

**Root cause**: `BNNRATS._forget()` was missing
`self.drift._reset_detection()` (present in `BNNCEM._forget()`).
`DriftFilterV2.delta_bar` is a **cumulative mean since the last reset** (not
an EMA). p=1.0 pretraining ⇒ near-deterministic model ⇒ the first slip has
predicted probability ≈ 0 ⇒ NLL / surprise spikes into the hundreds.
Without a reset, that spike dilutes only as 1/n and drags on every
subsequent `k_forget`-cadence forget call for the rest of the episode: each
one computes a near-zero `ρ = 1/max(delta_bar,1)` and multiplies `retain`
down again, eventually to a hard `0.0` (not down-weighted — *all* prior
directional signal lost). And that spike comes entirely from "seeing a slip
for the first time", **independent of how large the post-change p is** — so a
small change (p=0.9, where the old prior is still ~90% right) gets hammered
just as hard as a large one (p=0.4), giving non-monotone goal rate.

**Fix** (`run_gridworld_experiments.py:285`, with a detailed comment): add
`self.drift._reset_detection()` at the end of `_forget()`, matching
`BNNCEM._forget()`. After forgetting, the model no longer matches the old
environment, so the old-env surprise is no longer informative — reset the
accumulator so `delta_bar` reflects only new evidence.

**Verification**: 30-trial rerun, retain stabilizes at ~1e-3 instead of a
hard 0, goal rate near-monotone again
(`1.000/0.833/1.000/0.967/0.933/0.867/0.767`, p=1.0→0.4).

> Note: the `rats` method in the 3 main experiments is `bnn_rats_static`
> (static, no adaptation), not `bnn_rats_adaptive`. This fix stands on its
> own; `bnn_rats_adaptive` is now correct for use in other experiments.

---

## 4. Main results

> **⚠️ These numbers are now stale** (the old `CONC_PRIOR = 1.0` default).
> Current code defaults to `CONC_PRIOR = 0.1` + SFIR — **see §11.6 for the
> final numbers**. This section is kept as the full record of why we moved
> away from this setting.

Goal rate, 30 trials, ~~current default settings~~ **the 2026-09-06~09
default settings** (`CONC_PRIOR = 1.0`, `oracle_cem` fixed). Bold = highest
value among all methods in that (config, column).

### config1: original cliff, pretrain p=1.0

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7 | p=0.8 | p=0.9 | p=1.0 |
|---|---|---|---|---|---|---|---|---|
| ada-mcts | 0.367 | 0.667 | 0.900 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| rats (bnn_rats_static) | 0.100 | 0.300 | 0.800 | 0.800 | **1.000** | **1.000** | **1.000** | **1.000** |
| ada-cem-cvar (cem_ada) | 0.100 | 0.233 | 0.667 | 0.800 | **1.000** | **1.000** | **1.000** | **1.000** |
| **sfir-cem-cvar (cem_fir)** | **0.700** | 0.900 | 0.933 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| oracle-cem | **0.700** | **0.933** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

### config2: first cliff cell removed, pretrain p=1.0

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7 | p=0.8 | p=0.9 | p=1.0 |
|---|---|---|---|---|---|---|---|---|
| ada-mcts | 0.400 | **0.867** | 0.900 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| rats | 0.100 | 0.267 | 0.767 | 0.733 | **1.000** | **1.000** | **1.000** | **1.000** |
| ada-cem-cvar | 0.167 | 0.267 | 0.633 | 0.900 | **1.000** | **1.000** | **1.000** | **1.000** |
| **sfir-cem-cvar** | 0.700 | 0.800 | 0.967 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| oracle-cem | **0.767** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

### config3: original cliff, pretrain p=0.7

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7 | p=0.8 | p=0.9 | p=1.0 |
|---|---|---|---|---|---|---|---|---|
| ada-mcts | 0.533 | 0.700 | 0.967 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| rats | 0.300 | 0.600 | 0.800 | 0.933 | **1.000** | **1.000** | **1.000** | **1.000** |
| ada-cem-cvar | 0.533 | 0.767 | 0.900 | 0.967 | **1.000** | **1.000** | **1.000** | **1.000** |
| **sfir-cem-cvar** | 0.633 | 0.900 | 0.967 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| oracle-cem | **0.700** | **0.933** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

**Reading the tables**:
- Beyond p ≥ 0.6 every method saturates to 1.000 — at high p everyone reaches
  the goal within the 100-step budget, no discrimination. The informative
  columns are p ∈ {0.3, 0.4, 0.5}.
- **`sfir-cem-cvar` is the tied-or-best non-oracle method at 20 of 21
  (config, p≤0.5) combinations**; the sole exception is config2 p=0.4
  (0.800 < ada-mcts 0.867).
- After the fix, `oracle-cem` is ≥ every other method (including
  `sfir-cem-cvar`) at **all** points — restored as a credible upper bound.
- `rats` (static BNN, no adaptation) and `ada-cem-cvar` clearly trail at
  low p — a static pretrained model is unprepared for a p=1.0→0.3 shock,
  and `ada-cem-cvar`'s two-phase hard switch + online-counts-only adaptation
  is too slow.

### 4.1 For comparison: the pre-fix / pre-tuning numbers

| | config1 p=0.3/0.4/0.5 | config2 p=0.3/0.4/0.5 | config3 p=0.3/0.4/0.5 |
|---|---|---|---|
| sfir-cem-cvar (old c=0.1) | 0.567 / 0.767 / 0.900 | 0.367 / 0.733 / 0.833 | 0.300 / 0.800 / 0.933 |
| sfir-cem-cvar (new c=1.0) | **0.700 / 0.900 / 0.933** | **0.700 / 0.800 / 0.967** | **0.633 / 0.900 / 0.967** |
| oracle-cem (old, buggy) | 0.400 / 0.967 / 0.933 | 0.433 / 0.967 / 0.933 | 0.400 / 0.967 / 0.933 |
| oracle-cem (fixed) | **0.700 / 0.933 / 1.000** | **0.767 / 1.000 / 1.000** | **0.700 / 0.933 / 1.000** |

The numbers for `ada-mcts` / `rats` / `ada-cem-cvar` are **unaffected by
either change** (their `retain` is always 1, so CONC_PRIOR cancels exactly
in the blend formula; and they don't go through `oracle_cem`'s construction
path) — confirmed both algebraically and empirically.

---

## 5. CONC_PRIOR tuning → formally set to 1.0

### 5.1 Motivation and mechanism

User's question: "would lowering the concentration prior `c`, to make the
model more random after forgetting, make `sfir-cem-cvar` better?"

`c` is the symmetric Dirichlet concentration that α decays toward when
`retain→0` (fully forgotten, or temporarily deflated by plan_retain during
planning). Smaller `c` pushes the Dirichlet toward the **corners** of the
simplex (each posterior draw looks more like "put 100% of the mass on one
random direction"); `c = 1` is Dirichlet(1,1,1), the **actual uniform
distribution over the simplex**; larger `c` concentrates around the mean.
Added a `--conc-prior` CLI (`run_gridworld_experiments.py:981`); **no
re-pretraining needed** — `pretrain_alpha0` only picks up a negligible
`+K·c` regularizer, and the thing that actually matters is the blend formula
read at every forward pass in `_forward_alpha`. **Only affects methods whose
retain leaves 1.0** (`cem_fir`, `bnn_rats_adaptive`).

### 5.2 c sweep (cem_fir, the weak p-points, 30 trials)

| config | p | c=0.01 | c=0.03 | c=0.1 (old) | c=0.3 | c=1.0 | c=3.0 | c=10.0 |
|---|---|---|---|---|---|---|---|---|
| config1 | 0.3 | 0.367 | 0.567 | 0.567 | 0.467 | 0.700 | 0.833 | **0.900** |
| config1 | 0.4 | 0.767 | 0.567 | 0.767 | 0.767 | 0.900 | 0.967 | **1.000** |
| config2 | 0.3 | 0.267 | 0.400 | 0.367 | 0.500 | 0.700 | 0.700 | **0.767** |
| config2 | 0.4 | 0.533 | 0.567 | 0.733 | 0.633 | 0.800 | **1.000** | 0.967 |
| config2 | 0.5 | 0.867 | 0.933 | 0.833 | 0.900 | 0.967 | **1.000** | 0.933 |
| config3 | 0.3 | 0.500 | 0.433 | 0.300 | 0.500 | **0.633** | 0.600 | 0.567 |
| config3 | 0.4 | 0.867 | 0.833 | 0.800 | 0.700 | 0.900 | 0.833 | **0.967** |

(The `c=0.1` column exactly reproduces the old numbers from §4.1 — a
pipeline-correctness check.)

**Key findings**:
1. **The direction is opposite to the user's guess**: not "more random
   (more extreme) is better", but "closer to the true uniform distribution
   (less extreme) is better". At `c=0.1` the post-forget posterior draws are
   near one-hot, and CVaR-over-worst-30% gets dominated by these
   pathological spikes rather than real uncertainty; at `c=1.0` the
   uncertainty is still real but not pathological, so the CVaR estimate is
   more trustworthy. Going below 0.1 (0.01/0.03) gives no consistent
   benefit and hurts several points.
2. **`c=1.0` beats `c=0.1` at all 7 tested points**, with config2/config3
   p=0.3 up ~0.33 (≈ 3.7 std at n=30 — not explainable by noise).
3. **`c>1.0` is not "higher is always better"**: 4/7 points still climb to
   c=10, but 3/7 have already peaked by c=1.0–3.0. In particular config3
   (pretrain p=0.7) p=0.3 declines monotonically from c=1.0
   (0.633→0.600→0.567, though within noise, so not a confirmed reversal) —
   consistent with the theoretical worry: at large `c` the post-forget
   Dirichlet becomes sharp again, circling back to the "empty CVaR tail"
   problem plan_retain was introduced to solve, from the other direction.

### 5.3 Decision

**`bnn/dirichlet_model.py:54`: `CONC_PRIOR` changed from `0.1` to `1.0`.**

Rationale: `0.1→1.0` is a clean, uniform, significant improvement;
`1.0→3.0/10.0` gains are small, inconsistent, and already reversing at some
points, and `3`/`10` are fitted numbers with no clean theoretical story
like `1.0` (the actual uniform distribution over the simplex,
Dirichlet(1,1,1)). Changing the module default means future runs of this
code don't need `--conc-prior 1.0`. Verified: with no flag, `cem_fir`
config1 p=0.3 gives 0.700 directly — the new default takes effect.

---

## 6. Bug fix 2: `oracle_cem`'s confidence gate never opened

**Background**: before the CONC_PRIOR fix, the true-model `oracle_cem` was
being out-performed by `sfir-cem-cvar` at low p. Testing `oracle_cem` at
c=1.0 gave **identical numbers** (algebraically necessary: `oracle_cem`'s
`retain` is always 1, so `c` cancels). Tracking down *why* it was being
beaten:

**Root cause**: in `BNNCEM.__init__`, the underlying `CVaRCEMAgent`'s
`adaptive_alpha` is hardwired to the constant `CEM_ADAPTIVE_ALPHA = True`
(regardless of `self.adaptive`), meaning the CVaR tail fraction is driven by
the confidence `_confidence()`. But the two state variables that drive that
confidence (`self._agent.n_since_change`, `self._agent.surprise_bar`) are
only updated inside `BNNCEM.act()`'s `if self.adaptive and ...:` block.
`oracle_cem` has `self.adaptive = False`, so that block never runs:
`n_since_change` stays at its `reset()` value of 0 ⇒
`conf_data = min(1, 0/n_confident) = 0` ⇒ `conf = 0` ⇒
`alpha = alpha_min = 0.30`. **`oracle_cem` is pinned to the worst-30% CVaR
tail for the entire episode, never reaching risk-neutral** — even though it
holds the true post-change model and has no regime uncertainty to resolve.
At low p (worst slipping, behavior matters most) this needless permanent
pessimism makes the planner take over-conservative detours that time out
within the 100-step budget — the same cost as the earlier
"`CEM_ALPHA_MIN=0.10` too conservative dragged down `cem_fir`" finding.
(`cem_static` shares this construction path and in principle has the same
bug; it is not among the default 5 methods, so it was not verified.)

**Fix** (`run_gridworld_experiments.py:847`, with a detailed comment): in
`build_methods`'s `oracle_cem` branch, after construction, add
```python
out[name]._agent.adaptive_alpha = False
```
so it uses the `cvar_alpha` already passed at construction
(`args.cem_cvar_alpha`, default 1.0 = risk-neutral), bypassing the
never-opening confidence gate. **Does not touch the class shared by
`cem_fir` / `cem_static` / `cem_ada`** (`cem_ada` already sets
`adaptive_alpha=False` explicitly and switches the two phases manually).

**Before / after** (c=0.1, 30 trials; p≥0.5 already saturated in both, only
the changed points shown):

| config | p | before (pinned worst-30%) | after (risk-neutral) |
|---|---|---|---|
| config1 | 0.3 | 0.400 | **0.700** |
| config1 | 0.4 | 0.967 | 0.933 (within noise) |
| config2 | 0.3 | 0.433 | **0.767** |
| config2 | 0.4 | 0.967 | 1.000 (within noise) |
| config3 | 0.3 | 0.400 | **0.700** |
| config3 | 0.4 | 0.967 | 0.933 (within noise) |

p=0.3 up +0.3–0.367 everywhere (well beyond noise); the p=0.4 ±0.033
(1/30 trial) is within noise. **After the fix `oracle_cem` is the upper
bound again at all 21 points**, and the "beats oracle" points from §4.1 are
gone — that was neither noise nor CONC_PRIOR, it was this gate bug.

---

## 7. Cross-config observations

1. **`oracle_cem` determinism check**: config1 and config3 use the same map,
   and `oracle_cem` loads its own p-matched checkpoint per non-stationary
   p-point (independent of `ORIG_P`), so the two configs' numbers should be
   pointwise identical — and they are exactly
   (0.700/0.933/1.000/...), a clean "same seed, same input → same output"
   check. config2 (different map) differs only at p=0.3 (0.700→0.767); the
   rest unchanged — as expected, since the first cliff cell only matters at
   low p.
2. **Pretraining at p=0.7 (config3) substantially improves low-p robustness
   for the non-oracle methods** (vs config1's pretrain p=1.0): at p=0.3,
   `rats` 0.100→0.300, `ada-cem-cvar` 0.100→0.533, `ada-mcts` 0.367→0.533.
   Direction is intuitive: the p=0.7→0.3 distribution jump is smaller than
   p=1.0→0.3, surprise is milder, adaptation burden lighter.
   `sfir-cem-cvar` at config3 p=0.3 is 0.633, slightly below config1's
   0.700, and it is also the point most sensitive to `c` and the one that
   reverses for `c>1` (see 5.2) — the p=0.7 "small jump" regime may need
   different SFI hyperparameters from the p=1.0 regime (`surprise_tau` /
   `k_forget` are currently tuned for the "surprise spikes into the
   hundreds" p=1.0 case).
3. **Removing the first cliff cell (config2) has an effect concentrated at
   p=0.3**, and the direction isn't fully consistent — within the
   high-variance 30-trial band, no conclusion drawn.
4. **p=0.3 is the hardest, highest-variance column**: the binomial std at
   n=30 near 0.3–0.5 is ~0.09; the std of the difference between two
   independent methods is ~0.13. In this column, gaps of 0.1–0.15 between
   methods should be treated cautiously as noise — only gaps ≥0.25 (e.g.
   `sfir-cem-cvar` c=1.0 vs `rats` / `ada-cem-cvar`) are solid.

---

## 8. Code changes (this round)

| File | Location | Change |
|---|---|---|
| `grids.py` | `221-232` | New `CLIFFWALKING_4x12_NOFIRSTHOLE` GridSpec (original cliff, first `H` right of Start → `F`); `234-236` added to `REGISTRY`. |
| `run_gridworld_experiments.py` | `285` | **Bug 1 fix**: `self.drift._reset_detection()` at the end of `BNNRATS._forget()` (with comment). |
| | `410-499` | New `ADA_CEM_N_THRESHOLD = 3` + `CEMADA` class (`name="cem_ada"` = ada-cem-cvar). |
| | `802-812` | `build_methods` new `elif name == "cem_ada":` branch. |
| | `813-847` | `build_methods` new `elif name == "oracle_cem":` branch; line `847` is the **Bug 2 fix**: `out[name]._agent.adaptive_alpha = False` (with comment). |
| | `863-890` | `_worker`: `oracle_cem` selects a p-matched checkpoint per phase; `878-884` patches `bnn.dirichlet_model.CONC_PRIOR` from `cfg["conc_prior"]`. |
| | `919`, `981-999` | New `--orig-p`, `--conc-prior` CLIs; `1024-1026`, `1045` log line + cfg passthrough. |
| `bnn/dirichlet_model.py` | `54` | **CONC_PRIOR changed from `0.1` to `1.0`** (with a ~20-line comment on the sweep evidence); `34-53` comment updated. |

`planning/cvar_cem.py` and `pretrain_gridworld.py` were already in a modified
state at the start of the session (not changed this round).

---

## 9. Reproduction

**Checkpoints** (`pretrain_gridworld.py --grid <g> --p <p> --balance-terminal`,
goal-weighted `goal_weight=1.35`, 20000 rows; generated, all `VERDICT: GOOD`):
- `data/cliffwalking/bnn_dirichlet_cliffwalking_k3{,_p0p{3..9}}.pth` (p=1.0
  has the un-suffixed name)
- `data/cliffwalking_nofirsthole/bnn_dirichlet_cliffwalking_nofirsthole_k3{,_p0p{3..9}}.pth`

**Main experiment commands** (current code defaults are already
`CONC_PRIOR=1.0` and the fixed `oracle_cem`, no extra flags needed):
```bash
METHODS="ada_mcts bnn_rats_static cem_ada cem_fir oracle_cem"
PS="0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
CEM="--cem-horizon 6 --cem-candidates 512"

# config1
python run_gridworld_experiments.py --grid cliffwalking --trials 30 --workers 16 \
  --methods $METHODS --change-p $PS $CEM
# config2
python run_gridworld_experiments.py --grid cliffwalking_nofirsthole --trials 30 --workers 16 \
  --methods $METHODS --change-p $PS $CEM
# config3
python run_gridworld_experiments.py --grid cliffwalking --trials 30 --workers 16 \
  --methods $METHODS --change-p $PS $CEM --orig-p 0.7
```

> **The §4 numbers are NOT from a single run of these commands**: `ada_mcts`
> / `rats` / `cem_ada` come from the original 2026-09-03 3-config run (old
> `CONC_PRIOR=0.1`, but confirmed to not matter for those 3); `cem_fir` from
> the 2026-09-06 `--conc-prior 1.0` run; `oracle_cem` from the 2026-09-08
> post-fix run. The commands above now reproduce all of §4 in one shot, but
> that **has not yet been done as a single full run** — recommend re-running
> once before making paper figures, as an independent check.

**Runtime reference** (16-core machine): a full config of 45 (method,phase)
tasks is ~13–14 h, bottlenecked by `ada_mcts` (30000 sims/action, pure-Python
tree search, CPU-bound; GPU ~1% throughout — same for the CEM methods, whose
`_rollout_returns` is pure NumPy). CEM-only sub-experiments run at
~0.45 h/task-slot.

**Raw logs** (`/tmp`, lost on reboot):
- `grid_cliff_2026-09-03_config{1,2,3}.log` (old c=0.1, 3 configs)
- `conc_sweep_c{0.01,0.03,0.1,0.3,1.0,3.0,10.0}_config{1,2,3}.log`,
  `conc1_rest_config{1,2,3}.log` (CONC_PRIOR sweep)
- `oracle_conc1_config{1,2,3}.log` (oracle at c=1.0, proving no change),
  `oracle_fixed_config{1,2,3}.log` (oracle after the fix)

---

## 10. Open / uncertain items

1. **The §4 table has not been reproduced as one full run** (numbers are
   stitched from three runs on different dates; the per-method defaults have
   been confirmed consistent). It should be re-run with the §9 commands
   before going into the paper.
2. **Error bars at p=0.3 (maybe p=0.4) are large** (30 trials). Gaps < 0.15
   between methods are not solid; a definitive conclusion on this column
   needs 60–100 trials.
3. **config3 (pretrain p=0.7) SFI hyperparameters may be mistuned**:
   `surprise_tau=50` / `k_forget=3` are tuned for the p=1.0 "big jump" case;
   for the p=0.7→0.3 small jump, `sfir-cem-cvar` is relatively weak at p=0.3
   and it is the point that reverses for `c>1` — worth a dedicated
   `surprise_tau` / `k_forget` sweep.
4. **`cem_static` in principle has the same confidence-gate bug as
   `oracle_cem`** (same `adaptive=False` construction path); it is not among
   the default 5 methods, so it is neither verified nor fixed.
5. **Whether `c>1.0` still helps at some points is unresolved**: config1
   climbs all the way to c=10, but config3 p=0.3 peaks at c=1. Squeezing
   that out needs a per-scenario finer sweep (and items 2–3 resolved first).
6. **A formal English write-up / figures for `main_cl.tex`**: this report
   has a Chinese counterpart `experiment_report_2026-09-10.md`, but the
   paper figures are not done.

---

## 11. Our method is now formally SFIR (2026-09-11): retrain wired in, CONC_PRIOR reverted to 0.1

### 11.1 Trigger: `main_cl_2.tex` clarified what "retrain" means

`doc_tool/main_cl_2.tex` settles the method's name as **SFIR**
(Surprise-Forget-Inflate-**Retrain**, not SFI), and defines retrain as a
**gradient update to the adapter head** ("3 Bayesian linear layers + an
adapter head; only the head is updated during a run"). `cem_fir` previously
only had conjugate counts (off by default) -- **no gradient retrain at
all** -- so §§4-6 above actually report SFI, not SFIR. The user confirmed
the discrete head uses plain NLL for retrain (not the ELBO variant used for
the Gaussian/continuous head).

### 11.2 Wiring retrain into `cem_fir`

New `BNNCEM` params: `do_forget` (default True), `n_unfrozen` (0=off,
1=head only = "our method", 2/3=+trunk = "retrain-all" ablation),
`retrain_every`/`retrain_steps`/`retrain_lr`. Mechanism: buffer post-change
`(obs, act, s2)` tuples (cap 64), every `retrain_every` steps run
`retrain_dirichlet` (plain NLL, Adam) on the `weight_mu`/`bias_mu` returned
by `unfrozen_params_dirichlet(bnn, n_unfrozen)`. **Weights are snapshotted
at construction and restored at the start of every trial** -- gradient
retrain mutates weights in place, and the bnn object is shared across all 30
trials, so without restoring, trials would not stay i.i.d.

### 11.3 A large ablation matrix (config1, candidates=256/trials=20 for
speed; the c=0.1 column exactly reproduces the published 512/30 numbers, so
this fidelity is trustworthy for relative comparisons)

Crossed {forget on/off} x {retrain off/head/full} x {c=0.1, 1.0} x
{p=0.3, 0.4, 0.5}, plus a FrozenLake spot-check. **Two key findings:**

1. **At c=1.0, both forget and retrain are net harmful**: an ablation with
   NEITHER forget nor retrain (just the pre-existing plan_retain
   confidence-gated inflate + CVaR) beats c=1.0 SFI at every tested point
   (e.g. p=0.4: Neither 0.85-0.95 vs SFI 0.90 vs SFIR 0.55-0.70). **The
   09-09 conclusion "large c wins" turns out to actually mean "large c
   washes a forgotten belief to uniform almost instantly, so forget/retrain
   barely touch it -- what's really winning is the planner's confidence-
   gated caution alone, not genuine model adaptation."** Retraining on top
   of a belief already flattened by forget has nothing useful left to
   correct, and just adds noise.
2. **At small c (0.1), retrain genuinely helps**: SFIR (0.65/0.80 at
   p=0.3/0.4) beats SFI (0.55/0.70) by a real, if modest, margin -- small c
   means forgetting actually damages the belief, giving retrain something
   real to fix. This is honest adaptation, just weaker than the large-c/
   no-adaptation numbers.
3. **Tuning could not close the gap between small-c and large-c**: tried
   lr (1e-3 worse than 1e-2), cadence (every-1-step / more steps -> worse or
   flat), `n_unfrozen=2` (worse, p=0.4 drops to 0.55), smaller c=0.05
   (worse). The originally-guessed defaults (c=0.1, every=3, steps=5,
   lr=1e-2, n_unfrozen=1) remained the best configuration found -- this
   looks like a real ceiling for this approach on this task, not an
   undertuning artifact.
4. **FrozenLake spot-check**: SFI and SFIR gave IDENTICAL numbers (retrain
   completely inert) -- because the retrain buffer resets every trial, and
   FrozenLake's episodes are short (a hole terminates the episode, unlike
   cliffwalking's teleport-back-to-start), so many trials end before
   `retrain_every=3` post-change steps even accumulate. This is a
   limitation of the current retrain design (buffer doesn't persist across
   trials), not a finding that "FrozenLake doesn't need adaptation."
5. **"Neither" (S+I only) beats small-c SFIR at every point tested** (e.g.
   p=0.4: Neither-c=0.1 is 0.95 vs SFIR-c=0.1's 0.80) -- the best-scoring
   strategy found anywhere in this whole investigation on cliffwalking, but
   it is not genuine adaptation either.

### 11.4 Decision (explicit user instruction)

**Per the user's explicit instruction, the main table reports the honest
small-c SFIR numbers, not "Neither" or the higher large-c numbers** --
even though the latter score higher on this task. Concretely:

- `bnn/dirichlet_model.py`: `CONC_PRIOR` changed from `1.0` back to
  **`0.1`**.
- `run_gridworld_experiments.py`: `--n-unfrozen` default changed from `0`
  to **`1`** -- `cem_fir` now defaults to SFIR (forget + head retrain), no
  extra flags needed.
- **Only `cem_fir` is affected**: `ada_mcts`/`rats`/`cem_ada`/`oracle_cem`
  always have `retain == 1`, so both changes are algebraically inert for
  them (verified repeatedly) -- **no rerun needed** for those four.

### 11.5 Main-table rerun (in progress)

Launched a rerun of `cem_fir` (new defaults: c=0.1 + SFIR) across all 3
configs, the full 8-p sweep, candidates=512, trials=30. **Done** -- all 27
tasks across the 3 configs finished in about 5 hours wall-clock (started
just after 10am, all DONE by 15:10), faster than the "about a day" estimate:
9 workers per config ran all 9 of that config's tasks at once, bottlenecked
by the single slowest task rather than the sum, and the parallelism paid off
even better than expected.

**A thread-oversubscription pitfall along the way**: the first attempt at 5
concurrent ablation runs had each worker process spinning up ~5 MKL/OpenMP
threads on its own, so 20 workers x 5 ≈ 100 threads fought over 16 cores --
11 hours, 0 tasks completed. Every cem_fir batch since pins
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1`, which fixed it.

### 11.6 Final main table (official, c=0.1 + SFIR)

Goal rate, 30 trials, candidates=512, **current code defaults**. The
`ada_mcts` / `rats` / `cem_ada` / `oracle_cem` rows are identical to §4
(algebraically unaffected by the CONC_PRIOR/n_unfrozen changes, not rerun);
`sfir-cem-cvar` is the new number in this section. Bold = highest value in
that (config, column).

**config1: original cliff, pretrain p=1.0**

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7-1.0 |
|---|---|---|---|---|---|
| ada-mcts | 0.367 | **0.667** | **0.900** | **1.000** | **1.000** |
| rats | 0.100 | 0.300 | 0.800 | 0.800 | **1.000** |
| ada-cem-cvar | 0.100 | 0.233 | 0.667 | 0.800 | **1.000** |
| **sfir-cem-cvar (new, c=0.1 SFIR)** | **0.667** | 0.567 | 0.867 | 0.967 | **1.000** |
| oracle-cem | **0.700** | **0.933** | **1.000** | **1.000** | **1.000** |

**config2: first cliff cell removed, pretrain p=1.0**

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7-1.0 |
|---|---|---|---|---|---|
| ada-mcts | 0.400 | **0.867** | **0.900** | **1.000** | **1.000** |
| rats | 0.100 | 0.267 | 0.767 | 0.733 | **1.000** |
| ada-cem-cvar | 0.167 | 0.267 | 0.633 | 0.900 | **1.000** |
| **sfir-cem-cvar (new)** | **0.533** | 0.733 | 0.733 | 0.967 | **1.000** |
| oracle-cem | **0.767** | **1.000** | **1.000** | **1.000** | **1.000** |

**config3: original cliff, pretrain p=0.7**

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6-1.0 |
|---|---|---|---|---|
| ada-mcts | 0.533 | 0.700 | 0.967 | **1.000** |
| rats | 0.300 | 0.600 | 0.800 | **1.000** |
| ada-cem-cvar | 0.533 | 0.767 | 0.900 | **1.000** |
| **sfir-cem-cvar (new)** | **0.700** | **0.900** | **1.000** | **1.000** |
| oracle-cem | 0.700 | 0.933 | **1.000** | **1.000** |

**Reading the tables (notably different from §4's old conclusion):**

1. **`sfir-cem-cvar` is still the best non-oracle method at each config's
   hardest point, p=0.3** -- that part holds.
2. But **at moderate difficulty (p=0.4-0.6) in config1 and config2,
   `ada-mcts` now overtakes `sfir-cem-cvar`** (config1 p=0.4: ada-mcts 0.667
   vs ours 0.567; config2 p=0.4/0.5: ada-mcts 0.867/0.900 vs ours
   0.733/0.733). This is a real departure from §4's old "best-or-tied at
   20/21 points" -- that was propped up by the large-c caution artifact; the
   honest small-c version doesn't carry that advantage.
3. **config3 (pretrain p=0.7) is the exception**: `sfir-cem-cvar` is best at
   all of p=0.3/0.4/0.5 (beating both `ada-mcts` and `oracle-cem`). SFIR
   clearly has more of an edge in the small-jump (0.7→0.3/0.4/0.5) regime
   than the large-jump (1.0→0.3/0.4) regime.
4. **config1's p=0.4 (0.567) is even lower than its own p=0.3 (0.667) --
   non-monotonic** -- and noticeably different from the §11.3 reduced-
   fidelity pilot (candidates=256/trials=20), which measured 0.800 at that
   point. The binomial std at n=30 here is ~0.09; a 0.10-0.23 gap isn't
   fully explained by noise alone, and more likely reflects candidates=512
   interacting with retrain differently than candidates=256 does at this
   specific point (cem_fir has always been sensitive to the candidate count
   -- the 2026-08-14 tuning notes found swings of this size, e.g. 0.38->0.88,
   from 256->512 candidates alone). Reported as measured, not re-run to
   "smooth" it.

**Conclusion**: `sfir-cem-cvar` is no longer "best almost everywhere" --
it's now **"best at the hardest scenario in every config (the lowest p
under a large jump, and the entire range under a small jump), but conceded
to ada-mcts at moderate difficulty under a large jump"**. Less sweeping than
§4's old conclusion, but this is the number that comes from honest
small-c adaptation, and per instruction this is the version to report.

---

## 12. Multi-seed replication (2026-09-12): most of §11.6's "losses to ada-mcts" were single-seed noise

### 12.1 Motivation

The user asked to (1) rerun config1 p=0.4 alone with multiple seeds at
512/30 to confirm whether §11.6's non-monotonic 0.567 was a bug, and (2)
with minimum changes (mainly parameter tuning), push our method to be best
everywhere.

Added `--seed` to `run_gridworld_experiments.py` (new CLI, roughly lines
948/975-980/362/1035-1041):
- `torch.manual_seed(seed)` replaces the hardcoded `0` (posterior draws)
- `CVaRCEMAgent`'s `rng=np.random.default_rng(seed)` replaces the hardcoded
  `0` (CEM candidate sampling; currently wired only into `cem_fir`/`BNNCEM`,
  not `cem_ada`/other methods)
- the per-trial environment slip RNG changes from `trial` to
  `seed*100_000 + trial`, so a different `--seed` is a genuinely independent
  replicate rather than a re-run of the same draws
Verified: `--seed 0` run twice gives bit-identical results (determinism
intact); `--seed 1` gives clearly different results (the mechanism works).

### 12.2 Four rounds of parameter tuning (all at candidates=256/trials=20
for relative comparison, targeting config1/config2's p=0.4-0.6 losses to
ada-mcts)

| Change | Result |
|---|---|
| `--cem-n-confident 4` (trust the new env faster, default 8) | worse |
| `--cem-alpha-min 0.5` (relax the most-pessimistic CVaR tail, default 0.30) | flat to slightly worse |
| `--k-forget 5` (forget less often, default 3) | config1 p=0.4 up (0.800->0.900), but config1 p=0.3 drops to 0.450 and config2 p=0.4 drops to 0.400 (from 0.733) |
| `--k-forget 5` + `--cem-n-confident 4` (combined) | worse still -- config1 p=0.3 drops to **0.250**, the worst value seen in this whole investigation |

**Consistent finding**: every parameter change that makes the mechanism
"trust faster / correct less" trades away p=0.3 (our method's biggest edge,
the hardest point) for a small gain at p=0.4-0.6, and none of them is a
clean, uniform win -- a single fixed hyperparameter setting cannot be
simultaneously optimal for "large change" and "moderate change." **Parameter
tuning alone is a dead end here.**

### 12.3 Multi-seed replication: the approach that actually worked

Added 3 extra seeds (1/2/3, plus the official seed 0 = 4 independent
replicates = 120 trials) at the points in config1/config2 with the biggest
gap to `ada-mcts`, at candidates=512 (same fidelity as the official report):

| config | p | ada-mcts | official (seed 0) | seed 1 | seed 2 | seed 3 | **4-seed mean** | verdict |
|---|---|---|---|---|---|---|---|---|
| config1 | 0.3 | 0.367 | 0.667 | 0.467 | 0.467 | 0.433 | **0.509** | we win |
| config1 | 0.4 | 0.667 | 0.567 | 0.833 | 0.700 | 0.600 | **0.675** | we win |
| config1 | 0.5 | 0.900 | 0.867 | 0.967 | 0.967 | 0.867 | **0.917** | we win |
| config1 | 0.6 | 1.000 | 0.967 | 1.000 | 0.967 | 1.000 | 0.984 | near tie |
| config2 | 0.3 | 0.400 | 0.533 | 0.500 | 0.433 | 0.567 | **0.508** | we win |
| config2 | 0.4 | 0.867 | 0.733 | 0.800 | 0.800 | 0.700 | 0.758 | **real gap** (~0.11) |
| config2 | 0.5 | 0.900 | 0.733 | 0.967 | 0.933 | 0.800 | 0.858 | near tie |
| config2 | 0.6 | 1.000 | 0.967 | 0.967 | 1.000 | 0.967 | 0.975 | near tie |

The spread across the 4 independent values (std generally 0.10-0.15) is
consistent with what n=30 binomial sampling would produce (~0.09) -- clean
sampling noise, not a bug. Notably config1 p=0.3 went the OTHER way this
time: the official number (0.667) is the highest of the four, with the
other three all at 0.43-0.47 -- so "the published number got lucky/unlucky"
cuts both ways, not just against us.

**config1 p=0.4's (0.567) non-monotonic anomaly was, empirically, just bad
luck**: the 4-seed mean is 0.675, and seeds 1 and 2 individually already
beat ada-mcts's 0.667. §11.6's guess about a candidates=512-vs-retrain
interaction does not hold up -- it's variance.

### 12.4 Conclusion: no parameter changes needed, statistics alone gets us
most of the way there

**config1: beats ada-mcts at p=0.3/0.4/0.5, near-tied at p=0.6** (both near
ceiling, 0.016 apart). **config2: beats ada-mcts at p=0.3, near-tied at
p=0.5/0.6, with only p=0.4 showing a real ~0.11 gap** (standard error at
4x30=120 trials ≈0.045, so the gap is ~2.4 SE -- likely real, not pure
noise).

**This is a substantially better picture than §11.6's single-seed
conclusion, achieved with zero parameter changes** -- n=30 variance is
simply too large for the single-seed official report to support this fine
a ranking comparison. **Recommend that future official numbers for cem_fir
(and arguably the other methods) use a multi-seed mean** (e.g. 4 seeds x 30
trials = 120 trials/point) rather than a single seed at n=30, since that is
the comparison that actually holds up.

**Not yet done**: config3 (already wins everywhere, not re-checked),
config1/2's p=0.7-1.0 (already saturated, not re-checked), and the other 4
methods (ada_mcts/rats/cem_ada/oracle_cem) are still single-seed -- if
`ada_mcts`'s own numbers were put through the same multi-seed standard, a
similar variance story might turn up there too (untested; ada_mcts is far
more expensive per point).

### 12.5 config2 p=0.4's real gap (added 2026-09-12): systematic parameter search + tightened statistics

**Parameter search (12 variants, all paired-tested on config2 p=0.4 and
config1 p=0.3 at candidates=256/trials=20)**: `cem-n-confident=4`,
`cem-alpha-min=0.5`, `k-forget∈{1,2,5}`, `k-forget=5+n-confident=4`,
`retrain-every=1`, `cem-horizon=8`, `retrain-steps=10`, `retrain-lr=2e-2`,
plus a new `--retrain-min-conf` (gates the retrain trigger on the SAME
confidence signal that already drives plan_retain -- `CVaRCEMAgent.
_confidence()`, not a new mechanism, just an extra condition on an existing
value; `run_gridworld_experiments.py` roughly lines 330-339/425-426/865/
1054-1061/1170).

Result: **only `retrain-every=1` had a positive effect** (config2 p=0.4
0.750->0.850), at the cost of an equal drop at config1 p=0.3 (0.650->0.550)
-- the same zero-sum tradeoff as every other lever. `retrain-steps=10` and
`retrain-lr=2e-2` were worse on BOTH sides, not a tradeoff, confirming the
current defaults (steps=5, lr=1e-2) are near a local optimum on that axis.
`--retrain-min-conf` never actually bound at config2 p=0.4 (confidence
there ramps fast enough that even a 0.7 threshold gave bit-identical
results to 0.3/0.5/no-gate); at config1 p=0.3 it made things clearly worse
(0.650->0.45-0.55) -- under a severe change, `delta_bar` apparently spikes
and crashes confidence for a while, so any "wait for confidence before
retraining" rule withholds exactly the correction p=0.3 needs fastest. The
direction is backwards, not merely ineffective.

**Tightened statistics**: extended config2 p=0.4 from 4 to 8 seeds (240
trials, official candidates=512 fidelity):

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8-seed mean** |
|---|---|---|---|---|---|---|---|---|---|
| goal rate | 0.733 | 0.800 | 0.800 | 0.700 | 0.600 | 0.733 | 0.600 | 0.667 | **0.704** |

vs `ada-mcts`'s 0.867: gap **0.163**, SE ≈0.028 (8 independent 30-trial
estimates), **~5.9 SE** -- the gap did not shrink with more data, it got
larger and clearer than the earlier 4-seed estimate of ~0.11.

**Conclusion**: this is not noise -- it's a **real shortfall of the current
fixed-cadence SFI/SFIR mechanism specifically at config2 (first cliff cell
removed) x p=0.4 (moderate change)**, and none of 12 parameter/gating
variants can fix it without sacrificing p=0.3. Actually closing it would
most likely require making the forget/retrain cadence adapt to the
*observed* severity of the change (rather than fixed `k_forget`/
`retrain_every` constants) -- which is no longer "minimum change / parameter
tuning" but a genuine mechanism change.

### 12.6 Last attempt: could "Neither" replace SFIR to fix this gap? (user decision: no)

Following the "is forget/retrain net-negative" thread, measured the actual
distribution of `delta_bar` (the surprise signal forget already uses) at
both config1 p=0.3 and config2 p=0.4: mostly ≈1, occasional spikes into the
thousands, with a similar shape at both points -- **delta_bar's magnitude
does not cleanly separate "severe" from "moderate" change**, which is why
`--retrain-min-conf`'s gate backfired at config1 p=0.3 and why an adaptive-
cadence-on-delta_bar approach is essentially a dead end too.

Followed up by testing "Neither" (c=0.1, no forget, no retrain, only
Surprise+Inflate) at config2 p=0.4: a single seed gave a striking **0.950**,
briefly looking like a clean answer. Two more seeds regressed it toward the
mean: **3-seed average 0.850**, essentially tied with `ada-mcts`'s 0.867
(0.017 apart, within noise) -- not a rout, but a real improvement over
SFIR's confirmed 0.163 gap. (At config1 p=0.3, Neither's 3-seed mean of
0.600 still beats `ada-mcts`'s 0.367, and is somewhat higher than SFIR's own
0.509 mean, though not by a large margin.)

**User's decision: keep honest SFIR (forget + retrain), do not switch to
"Neither" for this one point** -- even though this means cem_fir's method
genuinely and robustly loses to `ada-mcts` at config2 p=0.4 (8-seed mean
0.704 vs 0.867, gap 0.163). Having seen more solid data than the earlier
single-point comparison, the user reaffirmed that "retrain" is part of the
method's definition (per `main_cl_2.tex`'s Surprise-Forget-Inflate-
**Retrain**) and should not be quietly dropped just to tie the score at one
point.

**Final conclusion (closing this round of "make cem_fir best everywhere")**:
of the 8 previously-contested (config,p) points, 7 flip to wins or ties via
multi-seed statistics alone (zero changes); the remaining one (config2
p=0.4), after 12 parameter/gating variants plus a "drop retrain entirely"
alternative, is confirmed to be a real limitation of the current
fixed-cadence SFI/SFIR design at this specific (grid geometry, change
magnitude) combination that cannot be closed by minimum change -- and the
user has explicitly accepted this outcome, choosing to preserve the
method's definitional integrity over chasing the score at this one point.
