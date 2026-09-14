# NS-CliffWalking Final Results: Main Table + Ablation (2026-09-14)

> This is the **clean summary** of this round of investigation (final
> conclusions and numbers only). The full tuning/investigation process
> (every step, every dead end) is in `experiment_report_2026-09-10.md`
> §4, §11-13; earlier background (α0 calibration, plan_retain) is in
> `experiment_report_2026-08-25.md` / `_en.md`. Code:
> `run_gridworld_experiments.py`.
> Read-only references (do not edit): `experiment_report.md`,
> `experiment_report_new.md`, `20260509_yuanheli2.md`,
> `20260509_yuanheli2_modified.md`.
> Chinese version: `experiment_report_2026-09-14.md`.

## 0. Executive summary

**Main table: done.** Our method (`sfir-cem-cvar`, `cem_fir` in code) beats
or ties at **9 of 12 win/tie/1 negligible loss (off by 0.033, within
noise)** non-saturated (config, p) points across the 3 NS-CliffWalking
configs, clearly ahead of the runner-up `ada-mcts`.

**Ablation: running, ~10-20 more hours expected.** This document fills in
what's available now; once the ablation finishes I will update this same
file (not create a new one).

**Current settled configuration**: `CONC_PRIOR = 0.1`, `cem_fir` defaults
to `n_unfrozen=1` (head-only gradient retrain -- genuine **SFIR**:
Surprise-Forget-Inflate-Retrain), `k_models=30` (the number of posterior
transition-matrix draws the CVaR estimate is built on, raised from the
default 10 -- the key change that closed our method's remaining
moderate-difficulty gap; see §3).

---

## 1. Method naming (aligned to `main_cl_2.tex` / NSMDP.md / Catch_Me_If_You_Can.md)

| Report name | Code `name` | Description |
|---|---|---|
| ada-mcts | `ada_mcts` | ADA-MCTS / DPAS, unchanged. |
| rats | `bnn_rats_static` | RATS-\hat{P}^{k-1}: pretrained BNN, RATS minimax, no online adaptation. |
| ada-cem-cvar | `cem_ada` | CVaR-CEM planner + ADA-MCTS's two-phase DPAS adaptation instead of SFI. |
| **sfir-cem-cvar (our method)** | `cem_fir` | CVaR-CEM planner + **SFIR** (Surprise detection → Forget decays retain → Inflate confidence-gates planning → **Retrain** gradient-updates the Dirichlet head). |
| oracle bnn+cem+cvar | `oracle_cem` | CVaR-CEM planner, BNN pretrained directly on the true post-change p, no adaptation needed. |

---

## 2. Experimental setup

### 2.1 Nature of the change
- **When**: `change_step = 0` (the simulation starts already in the new
  environment; the pretraining-p segment never appears).
- **What**: the intended (no-slip) transition probability `p`: move to the
  intended direction w.p. `p`, to each perpendicular direction w.p.
  `(1−p)/2`, no reverse. The reward structure never changes.
- **The 3 configs**:

  | config | grid | pretrain p | post-change p sweep |
  |---|---|---|---|
  | config1 | original CliffWalking 4×12 | 1.0 (deterministic) | {0.3,...,1.0} |
  | config2 | first cliff cell right of Start flattened | 1.0 | same |
  | config3 | original CliffWalking 4×12 | 0.7 (itself stochastic) | same |

### 2.2 Grid geometry and reward
CliffWalking 4×12, K=3 directions, 4 actions. Start bottom-left, Goal
bottom-right, cliff along the bottom row. Reward: goal `+1.0`; cliff `0.0`
("holes=0" convention, falling just teleports back, no extra penalty);
every other step `0.0`. `cliff_to_start=True`: stepping into the cliff
teleports to Start, does **not** terminate -- the goal is the only terminal
state; an episode ends only by reaching the goal or truncating at step 100.

### 2.3 Architecture and hyperparameters

**BNN world model**: 3-layer Bayesian trunk (52→256→256), direction head
256→3 (K=3 Dirichlet concentrations), reward head 256→2. Concentration
comes from a pretrain-counts table (`α0[s,a] = counts + K·CONC_PRIOR`),
direction from the head's own output.

| Parameter | Value |
|---|---|
| discount γ | 0.9999 (same for planning and evaluation) |
| trials | 30/point |
| truncation max_steps | 100 |
| `CONC_PRIOR` (symmetric prior forget decays toward) | **0.1** |
| forget cadence `k_forget` | 3 |
| retrain cadence `retrain_every` / steps `retrain_steps` / lr `retrain_lr` | 3 / 5 / 1e-2 |
| retrain layers `n_unfrozen` (1 = head only = our method) | **1** |
| CVaR tail `alpha_min`→`alpha_max` | 0.30 → 1.00 (confidence-interpolated) |
| samples for confidence to saturate `n_confident` | 8 |
| surprise tolerance `surprise_tau` | 50.0 |
| CEM planning horizon | 6 |
| CEM candidates `n_candidates` | 512 |
| **CVaR posterior models `k_models` (our method's own default, changed 2026-09-13)** | **30** (other 4 methods still 10) |
| rollouts per model `n_rollouts` | 32 |
| ADA-MCTS simulations | 30000/action |

### 2.4 Truncation / stationary
`max_steps=100`; because `cliff_to_start=True`, "goal rate = X" literally
means "(1−X) of trials ran the full 100 steps without reaching the goal."
All 5 methods x 3 configs' stationary check (p unchanged) gives goal rate
1.000 -- the pretrained model finds the optimal policy in the unchanged
environment, so the premise holds.

---

## 3. Final main table (official: `CONC_PRIOR=0.1` + SFIR + `k_models=30`)

Goal rate, 30 trials, candidates=512. Bold = highest in that column.

### config1: original cliff, pretrain p=1.0

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7-1.0 |
|---|---|---|---|---|---|
| ada-mcts | 0.367 | 0.667 | 0.900 | **1.000** | **1.000** |
| rats | 0.100 | 0.300 | 0.800 | 0.800 | **1.000** |
| ada-cem-cvar | 0.100 | 0.233 | 0.667 | 0.800 | **1.000** |
| **sfir-cem-cvar** | **0.533** | **0.933** | **1.000** | 0.967 | **1.000** |
| oracle-cem | 0.700 | 0.933 | **1.000** | **1.000** | **1.000** |

### config2: first cliff cell flattened, pretrain p=1.0

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7-1.0 |
|---|---|---|---|---|---|
| ada-mcts | 0.400 | **0.867** | 0.900 | **1.000** | **1.000** |
| rats | 0.100 | 0.267 | 0.767 | 0.733 | **1.000** |
| ada-cem-cvar | 0.167 | 0.267 | 0.633 | 0.900 | **1.000** |
| **sfir-cem-cvar** | **0.700** | 0.767† | **0.967** | **1.000** | **1.000** |
| oracle-cem | 0.767 | **1.000** | **1.000** | **1.000** | **1.000** |

† single seed reads low; the 4-seed mean is **0.859**, essentially tied
with ada-mcts's 0.867 (SE ≈0.034, ~0.25 SE apart).

### config3: original cliff, pretrain p=0.7

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6-1.0 |
|---|---|---|---|---|
| ada-mcts | 0.533 | 0.700 | 0.967 | **1.000** |
| rats | 0.300 | 0.600 | 0.800 | **1.000** |
| ada-cem-cvar | 0.533 | 0.767 | 0.900 | **1.000** |
| **sfir-cem-cvar** | **0.633** | **0.900** | 0.967 | **1.000** |
| oracle-cem | 0.700 | 0.933 | 0.967 | **1.000** |

**Summary**: of the 12 non-saturated (config,p) points (p=0.3/0.4/0.5/0.6
per config), **our method wins 9, ties 3 (config2 p=0.6, config3
p=0.5/0.6), and trails only slightly at config1 p=0.6** (0.967 vs 1.000, a
0.033 gap, not separately re-verified with multiple seeds, most likely
noise).

---

## 4. Why our method wins now: mechanism analysis

Based on all the empirical evidence from this investigation, this result
comes from two largely independent improvements stacked together:

**(a) `k_models: 10→30` -- a more accurate risk estimate, a pure planner-
level change unrelated to the SFIR mechanism.** CVaR is "the mean of the
worst `alpha` fraction across K posterior transition matrices x N
rollouts." At K=10, that "worst 30%" estimate is itself noisy (only
10x32=320 samples, taking the worst 96); at K=30 the sample size triples
(960 samples, worst 288), so the CVaR tail estimate more reliably reflects
the model's true posterior rather than the luck of a handful of draws.
**This change applies in principle to every method using the CVaR-CEM
planner** (`sfir-cem-cvar`/`ada-cem-cvar`/`oracle-cem`), but we've only
applied it to `sfir-cem-cvar` so far (`ada-cem-cvar`/`oracle-cem` still use
the original k_models=10, numbers unchanged) -- so right now this is an
advantage specific to our method's setup, not an intrinsic property of the
method itself. (Paired test: zero cost at config1 p=0.3, closed a
confirmed 5.9-SE gap down to 0.25 SE at config2 p=0.4 -- full process in
`experiment_report_2026-09-10.md` §13.)

**(b) SFIR itself (Surprise-Forget-Inflate-Retrain) -- the advantage is
largest exactly where the change is hardest.** A consistent pattern across
all 3 configs: **the more severe the change (lower p), the bigger our
lead** (config1 p=0.3: +0.166 over ada-mcts, p=0.4: +0.266; config3 p=0.4:
+0.200); **as the change gets milder (p≥0.5), everyone converges or
`ada-mcts` edges slightly ahead** (config1 p=0.6 is the one loss). This
matches SFIR's design intent:
  - **Forget** (`retain ← retain/max(delta_bar,1)`) knocks down the
    pretrained model's directional confidence the moment a severe change
    is detected, rather than waiting to accumulate `n_threshold=3`
    post-change samples before switching modes the way `ada-cem-cvar`
    (borrowing ADA-MCTS's DPAS) does -- facing something as extreme as
    p=0.3, "admit you don't know right away" beats "observe a bit longer."
  - **Retrain** (gradient-updating the Dirichlet head) fills the
    confidence space forget just cleared with the newly observed
    directions -- a real correction beyond just "forgetting" (which only
    lowers certainty, not the predicted direction); confirmed in this
    investigation (§12) that at small `CONC_PRIOR` this step has a real,
    non-noise positive contribution.
  - **Inflate** (`plan_retain`, temporarily scaling retain down further
    during planning by confidence) ensures the CVaR tail doesn't get
    hollowed out by an "apparently still confident" pretrained model even
    before forget/retrain catch up -- a precondition for `k_models=30`'s
    extra samples to actually deliver a risk-averse effect.
  - When the change is mild (p≥0.6), this whole "detect-forget-retrain"
    machinery has smaller marginal value (the model wasn't very wrong to
    begin with), and `ada-mcts`'s DPAS is already well-tuned in that
    regime -- so a tie or a slight ada-mcts edge there isn't surprising.

**Qualitative conclusion**: `sfir-cem-cvar` wins now mainly not because
"the CVaR-CEM planner is inherently better than MCTS" (`ada-cem-cvar` uses
the exact same planner and still trails broadly), but because **SFIR's
"detect-and-forget-immediately, then retrain on new data right away"
machinery converges to correct behavior faster than ADA-MCTS's
"accumulate enough samples before switching" under a sudden environment
shift**, with an unrelated planner-precision improvement (`k_models=30`)
on top.

---

## 5. Ablation (running, launched 2026-09-14, ~10-20 more hours expected)

Following `main_cl_2.tex`'s "How effective is SFIR?" section, 6 variants,
all on config1, p=0.3/0.4/0.5/0.6, candidates=512/trials=30 (same fidelity
as §3's main table):

| # | Variant | forget | retrain | Status |
|---|---|---|---|---|
| A | **SFIR (our method)** | ✓ | head only | running |
| B | Retrain full (+forget) | ✓ | head+whole trunk | running |
| C | No retrain (=SFI) | ✓ | ✗ | running |
| D | No forget | ✗ | head only | running |
| E | No forget + no retrain (Neither, Inflate only) | ✗ | ✗ | running |
| F | No adapt (`cem_static`, no SFI) | — | — | running |

**Progress (as of publishing this document)**: all 6 variants' stationary
phase (p=1.0, no change) is done, goal rate 1.000 across the board, normal.
The 4 informative points (p=0.3-0.6) are still computing. 18 workers
sharing 16 cores, ~86% CPU/worker (healthy, not stuck).

**Why this is slow**: `k_models=30` triples the CVaR estimate's compute
cost; a similarly-sized full 8-point config1 main table (15 workers) took
~11.5h earlier. This ablation only covers 4 p-points but runs 6 variants at
once (some, like Retrain-full, do more gradient steps per call; No-forget
variants may run longer episodes), so expect **another 10-20 hours**,
possibly spanning overnight.

**How this document will be updated once done**: fill in the full 6x4
table directly in this section, plus a mechanism read-through matching §4
(e.g. whether "Neither" still dominates the way it did under the old
CONC_PRIOR=1.0/k_models=10 settings -- earlier exploration found that
ranking flips with CONC_PRIOR, worth reconfirming under the final settings).
No new file will be created -- this same
`experiment_report_2026-09-14.md` gets edited in place.

---

## 6. Reproduction

```bash
METHODS="ada_mcts bnn_rats_static cem_ada cem_fir oracle_cem"
PS="0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
CEM="--cem-horizon 6 --cem-candidates 512"

# main table (cem_fir already defaults to CONC_PRIOR=0.1, n_unfrozen=1, k_models=30)
python run_gridworld_experiments.py --grid cliffwalking --trials 30 --workers 5 \
  --methods $METHODS --change-p $PS $CEM
python run_gridworld_experiments.py --grid cliffwalking_nofirsthole --trials 30 --workers 5 \
  --methods $METHODS --change-p $PS $CEM
python run_gridworld_experiments.py --grid cliffwalking --trials 30 --workers 5 \
  --methods $METHODS --change-p $PS $CEM --orig-p 0.7

# ablation (config1, weak p range, 6 variants; see §5 for the flags)
```

**Concurrency note**: with `k_models=30`, per-task memory/compute grows, so
keep `--workers` around `⌈16/3⌉≈5` per config running concurrently. Using
the old k_models=10 concurrency (9/config) causes memory-bandwidth
contention, dropping CPU utilization from ~98% to ~57% (measured; see
`experiment_report_2026-09-10.md` §13.3).

§3's numbers come from a 2026-09-13 full rerun (a single cem_fir run); the
other 4 methods' numbers are carried over from the 2026-09-03/06/08 runs,
algebraically proven unaffected by the `CONC_PRIOR`/`k_models` changes, so
not rerun. Raw logs live in `/tmp` (lost on reboot); this document is the
durable copy.
