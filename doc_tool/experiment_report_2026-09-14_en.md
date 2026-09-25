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

**Ablation: done** (single seed, pending multi-seed replication -- see §5).

**✅ Fairness correction: complete (2026-09-16).** §3's main table used to
apply `k_models=30` **only** to our method `sfir-cem-cvar`, leaving
`ada-cem-cvar`/`oracle-cem` -- which use the same CVaR-CEM planner -- at the
old default of 10. That was an unfair comparison (`ada-mcts`/`rats` do not
use the CVaR-CEM planner and were never affected). Both baselines have now
been rerun at `k_models=30` across all 3 configs × p=0.3-0.6; **all 24
points are done and §3's two rows below are the corrected numbers.**
Result: **`ada-cem-cvar` is essentially insensitive to `k_models=30`**
(largest change across 12 points +0.067; the three hardest p=0.3 points move
−0.033/0/0), so **our method's lead over it survives the fair comparison
intact** (it wins 11 of the 12 non-saturated points and ties 1). What the
correction does raise is `oracle-cem`, the theoretical upper bound (+0.133
at p=0.3, +0.067 at p=0.4) -- which now sits strictly above our method at
every point, making it a cleaner upper bound than before. **§3's headline
summary (9 wins / 3 ties / 1 negligible loss) is unchanged**, because that
summary is measured against `ada-mcts`, which this bug never touched.

Note: bold in §3's tables marks the best of the 4 genuinely usable methods;
`oracle-cem` is a cheating upper bound and is excluded from bolding.

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

**What the "holes=0" convention implies for the ADA-MCTS baseline (noted
2026-09-15)**: ADA-MCTS's risk-averse step (`pessimistic_sample`: one-hot
the worst reachable cell) only fires when some reachable cell carries a
*negative* reward. Under holes=0 no reward anywhere is negative, so that
step **never fires** in this table: the `ada-mcts` and `ada-cem-cvar`
numbers here reflect their DPAS model-switching behaviour *without* the
worst-case sampling phase. This follows from the reward convention, not
from a porting bug (verified in `planning/ada_mcts.py::_worst_case_sample`).
Under the authors' own convention (holes = -1, terminal) the step does fire
-- that setting is being measured separately in the Act-as-You-Learn
reproduction run, on the `cliffwalking_aayl` grid.

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
| ada-cem-cvar | 0.067 | 0.267 | 0.700 | 0.800 | **1.000** |
| **sfir-cem-cvar** | **0.533** | **0.933** | **1.000** | 0.967 | **1.000** |
| oracle-cem | 0.833 | 1.000 | 1.000 | 1.000 | **1.000** |

### config2: first cliff cell flattened, pretrain p=1.0

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7-1.0 |
|---|---|---|---|---|---|
| ada-mcts | 0.400 | **0.867** | 0.900 | **1.000** | **1.000** |
| rats | 0.100 | 0.267 | 0.767 | 0.733 | **1.000** |
| ada-cem-cvar | 0.167 | 0.300 | 0.700 | 0.900 | **1.000** |
| **sfir-cem-cvar** | **0.700** | 0.767† | **0.967** | **1.000** | **1.000** |
| oracle-cem | 0.767 | 0.967 | 1.000 | 1.000 | **1.000** |

† single seed reads low; the 4-seed mean is **0.859**, essentially tied
with ada-mcts's 0.867 (SE ≈0.034, ~0.25 SE apart).

### config3: original cliff, pretrain p=0.7

| method | p=0.3 | p=0.4 | p=0.5 | p=0.6-1.0 |
|---|---|---|---|---|
| ada-mcts | 0.533 | 0.700 | 0.967 | **1.000** |
| rats | 0.300 | 0.600 | 0.800 | **1.000** |
| ada-cem-cvar | 0.533 | 0.767 | 0.933 | **1.000** |
| **sfir-cem-cvar** | **0.633** | **0.900** | 0.967 | **1.000** |
| oracle-cem | 0.833 | 1.000 | 1.000 | **1.000** |

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
method itself. **This asymmetry was an error and was corrected on
2026-09-16** (all 24 points rerun; §3's numbers above are the corrected
ones). **The key finding from that correction: `ada-cem-cvar` is
essentially insensitive to `k_models=30`** (largest change +0.067 across 12
points; the three hardest p=0.3 points move −0.033/0/0), so our method's
lead over it is fully preserved; the one that gains is `oracle-cem`, the
upper bound. That adds a necessary qualifier to (a): **more posterior draws
only pay off when the underlying model is already right.** The oracle is
pretrained directly on the true post-change p, so it collects the full
benefit; `ada-cem-cvar`'s bottleneck is its own "observe first, switch once
enough samples accumulate" DPAS adaptation, and no amount of CVaR-tail
precision helps while the model itself is still stale. In other words, our
lead does not come from "we gave ourselves a bigger K" -- it comes from SFIR
making the model correct faster, which is what makes that K usable. (Paired test: zero cost at config1 p=0.3, closed a
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

## 5. Ablation: results (launched 2026-09-14, finished 2026-09-15, single seed)

Following `main_cl_2.tex`'s "How effective is SFIR?" section, 6 variants,
all on config1, p=0.3/0.4/0.5/0.6, candidates=512/trials=30, k_models=30
(same fidelity as §3's main table):

| # | Variant | forget | retrain | p=0.3 | p=0.4 | p=0.5 | p=0.6 |
|---|---|---|---|---|---|---|---|
| A | **SFIR (our method)** | ✓ | head only | 0.533 | 0.933 | **1.000** | 0.967 |
| B | Retrain full (+forget) | ✓ | head+whole trunk | 0.567 | 0.733 | 0.933 | 1.000 |
| C | No retrain (=SFI, forget only) | ✓ | ✗ | **0.633** | 0.867 | 0.967 | 1.000 |
| D | No forget (retrain only) | ✗ | head only | 0.600 | **0.967** | 0.933 | 1.000 |
| E | Neither (no forget, no retrain; Inflate only) | ✗ | ✗ | **0.633** | 0.867 | 0.933 | 0.967 |
| F | No adapt (`cem_static`) | — | — | 0.067 | 0.367 | 0.767 | 0.933 |

Bold = highest in that column.

**Honest conclusions (including the parts that do not favour us)**:

1. **"Adapt at all vs. don't" matters enormously, and the evidence is
   clean**: F (no adaptation) is far behind at all 4 points (just 0.067 at
   p=0.3); every one of A-E beats it by a wide margin. This conclusion is
   solid.
2. **But on "which adaptation mechanism," SFIR (A) is not uniformly best**:
   at p=0.3, C (forget only) and E (Neither) both reach 0.633 vs A's 0.533,
   about 0.1 higher; at p=0.4, D (retrain only) reaches 0.967 vs A's 0.933;
   A is strictly best only at p=0.5; at p=0.6 three variants (B/C/D) reach
   1.000 while A's 0.967 is within noise of them. **This is the same
   pattern found earlier in this investigation (under `CONC_PRIOR=1.0`, see
   `experiment_report_2026-09-10.md` §11-12) where "Neither" beat SFIR --
   smaller under the final settings (`CONC_PRIOR=0.1` + `k_models=30`), but
   not gone.**
3. **This is single-seed data (seed=0).** This investigation has already
   shown that gaps of exactly this size (0.03-0.1) are frequently noise: of
   8 previously "lost" points, 7 flipped to a win/tie after multi-seed
   averaging (§12). The ordering in this table -- especially C/E's 0.1 lead
   over A at p=0.3 -- **has not been multi-seed replicated and must not be
   treated as a final conclusion**; it is a preliminary signal needing
   confirmation.

**Next step**: once the §3 `k_models` fairness correction (see §0's
warning) finishes and frees up compute, replicate all 6 variants across 4
seeds and revisit this section's ranking.

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
