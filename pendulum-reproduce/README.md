# pendulum-reproduce

Standalone reproduction of every Pendulum result in the paper:

| What | Command | Output |
|---|---|---|
| Main result (SURF, 4 shifts) | `python run_main.py` | `results/main_table.md` |
| Ablations (7 columns, 2 shifts) | `python ablation/run_ablation.py` | `results/ablation_table.md` |
| SURF with different planners (CEM-CVaR / MPPI / iLQR / MCTS) | `python planners/run_planners.py` | `results/planner_table.md` |

No flags are needed. Every hyperparameter is fixed in the code, matching the paper runs. Each command:

1. runs all missing trials in parallel on the available GPUs;
2. prints the table;
3. checks every seed against the stored paper runs in `reference/paper_returns.json`.

On the machine that produced the paper, every reproduced seed is **bit-identical** to the stored one.

## Setup

```bash
pip install -r requirements.txt     # torch, numpy, scipy, gymnasium, mbrl
python run_main.py --trials 2       # ~4 min smoke test: seeds 1000-1001, all 4 shifts
                                    # (ablation ~5 min, planners ~15 min with --trials 2)
```

Tested with Python 3.12.3, torch 2.13.0 (CUDA 13.0), numpy 2.5.1, scipy 1.18.0, gymnasium 1.3.0 and mbrl 0.1.5, on one RTX 4090.

### Optional flags (all three scripts)

| Flag | Effect |
|---|---|
| `--trials N` | First N seeds of 1000–1099 only (a quick check). Any N reproduces those seeds exactly. |
| `--jobs N` | Number of concurrent worker processes. The default comes from free GPU memory (≈1.5 GB per job, at most 12 per GPU) and the CPU count. |
| `--table-only` | Print the table from whatever results exist; run nothing. |

Runs are **resumable**. Each job writes `results/<setting>/<arm>__<planner>/t<first>-<end>.json` atomically, and finished jobs are skipped. The three scripts share results: SURF+CEM is computed once and used by all three tables. Per-job logs are in `results/logs/`.

Approximate wall time on one RTX 4090 with 12 workers:

| Script | Wall time |
|---|---|
| Main | ~1.5 h |
| Ablation | ~5 h (the SURF column is shared with main) |
| Planners | ~11 h (iLQR ≈ 5 h, MCTS ≈ 3.5 h, MPPI ≈ 2 h) |

## Expected results

These are the stored paper runs: mean ± SEM over seeds 1000–1099; higher is better.

**Main (`run_main.py`)**

| Setting | SURF |
|---|---|
| m: 1 → 4 | −684.17 ± 26.72 |
| m: 1 → 0.4 | −89.34 ± 5.11 |
| g: 10 → 15 | −465.03 ± 18.57 |
| g: 10 → 5 | −142.62 ± 9.29 |

The paper table prints −684.22 and −89.30 for the first two rows. The stored per-seed data behind them gives −684.17 and −89.34; this folder reproduces the stored data.

**Ablations (`ablation/run_ablation.py`)**

| Setting | SURF | Retrain full | No retrain | No forget + no retrain | No adapt | No forget | Oracle |
|---|---|---|---|---|---|---|---|
| m: 1 → 4 | −684.2 ± 26.7 | −830.1 ± 21.4 | −1065.9 ± 15.7 | −1007.8 ± 17.8 | −1005.3 ± 17.5 | −979.4 ± 16.4 | −508.8 ± 36.0 |
| g: 10 → 15 | −465.0 ± 18.6 | −870.6 ± 23.1 | −1298.3 ± 24.9 | −1084.3 ± 25.3 | −1084.5 ± 25.3 | −1065.5 ± 29.5 | −214.6 ± 11.3 |

"No forget" is SURF's own 1-layer head with forgetting switched off (`head_1l_noforget`), so depth is held fixed. An older version of the table used the 3-layer variant, which gave −966 / −1084.

**Planners (`planners/run_planners.py`)**

| Setting | CEM-CVaR | MPPI | iLQR | MCTS |
|---|---|---|---|---|
| m: 1 → 4 | −684.2 ± 26.7 | −783.5 ± 31.1 | −893.0 ± 23.9 | −793.0 ± 39.8 |
| m: 1 → 0.4 | −89.3 ± 5.1 | −100.0 ± 4.7 | −122.2 ± 7.8 | −127.6 ± 7.3 |
| g: 10 → 15 | −465.0 ± 18.6 | −829.4 ± 29.2 | −1207.0 ± 12.5 | −1098.6 ± 28.9 |
| g: 10 → 5 | −142.6 ± 9.3 | −248.1 ± 18.4 | −185.3 ± 13.1 | −277.9 ± 18.2 |

The script also prints each planner's paired difference from CEM-CVaR on the same seeds. CEM-CVaR is best in every row, with p < 0.003.

## Experiment setting

**Environment.** Gymnasium `Pendulum-v1` dynamics with |u| ≤ 2 and dt = 0.05. The reward is the env's own: r = −(θ² + 0.1 θ̇² + 0.001 u²), so it is ≤ 0 and higher is better. The pole starts from the env's random initial state, seeded by the trial seed.

**The change.** The model is pretrained at mass 1, gravity 10. The shifted dynamics are live from **time step 0** of every trial:

- m 1 → 4 and m 1 → 0.4 (gravity stays 10);
- g 10 → 15 and g 10 → 5 (mass stays 1).

**Episodes and trials.**

- Each trial runs exactly **200 real steps**; nothing terminates it early.
- Every trial is cut off at 200 steps, and that is the only truncation.
- The reported return is the **undiscounted** sum of the 200 rewards.
- There are 100 independent trials per cell, seeds 1000–1099, each starting from the pretrained model.

**Dynamics model** (`data/pendulum/bnn_dynamics.pth`).

- A Bayesian MLP with a Gaussian output head: input 4 → 256 → 256 → output 8.
- The input is [cos θ, sin θ, θ̇, u], normalized with the stored env statistics.
- The two 256-unit hidden layers use SiLU activations.
- The 8 outputs are a mean and a log-variance for each of Δcos, Δsin, Δθ̇ and reward.
- Weights are mean-field Gaussian with prior std 1.0. Aleatoric noise is not injected in imagined rollouts.

**SURF** (`head_1l`).

- The body is frozen and always evaluated at u = 0.
- A **1-layer Bayesian adapter** h(s, u) = W·[cos θ, sin θ, θ̇, u] + b, with W ∈ ℝ^{3×4}, is added to the predicted Δs. It has prior std 0.1 and initial σ 1e-3, and is warm-started so that h = w₀·u reproduces the body's own action gain.
- After every real step:
  - per-dimension surprise δ = ν²/S feeds a per-dimension EWMA drift filter (η = 0.2), giving δ̄;
  - every k_forget = 5 steps, retention forgetting inflates the head's posterior: τ ← τ₀ + ρ(τ − τ₀), with ρ = clip(1/max(δ̄, 1), 1e-3, 1); when it fires, the KL prior is re-anchored to the inflated belief;
  - the head is refit with 5 ELBO steps (Adam, lr 3e-3, 3 MC samples, batch 256 from the trial's replay buffer, β = 1, KL scaled by the buffer size).

**Ablation arms** (`src/experiment.py`, `run_trial`).

| Column | Arm | What it does |
|---|---|---|
| SURF | `head_1l` | As above. |
| Retrain full | `retrain_full` | Whole-network ELBO refit every step (Adam lr 1e-3, 5 steps, 3 MC samples, local reparameterization); no forgetting. |
| No retrain | `no_retrain` | Surprise-driven per-dimension forgetting of the whole network's head rows (retention mode, q_max 2.0) every 5 steps; never refits. |
| No forget + no retrain | `no_forget_no_retrain` | The frozen model, run through the identical surprise/filter bookkeeping. |
| No adapt | `no_adapt` | The frozen pretrained model. |
| No forget | `head_1l_noforget` | SURF without the forgetting step. |
| Oracle | `oracle` | The true shifted dynamics inside the same planner. |

**Planner (all columns except the planner table).**

- CEM over open-loop action sequences: horizon H = 40, 8 iterations × 500 candidates, top 10% elites, σ_init = 1.0, σ_min = 0.05.
- MPC warm start: the previous plan, shifted one step.
- Each candidate is scored by **CVaR_α** over K = 10 posterior draws of the **discounted (γ = 0.99)** imagined return, with α = 1.0 (the CVaR of the whole draw set, i.e. its mean).
- Weights are resampled per imagined step. The planner uses the analytic Pendulum reward, and imagined states are projected back to the unit circle with |θ̇| ≤ 8.

**Planner comparison** (`src/planning/continuous_planners.py`).

All four planners optimise the *same* CVaR objective under the *same* budget: 8 × 500 × 10 × 40 = **1.6M imagined transitions per real step**. Only the search procedure differs.

| Planner | Search procedure | Budget used |
|---|---|---|
| CEM-CVaR | As above. | 1.6M |
| MPPI | Same 8 × 500 samples. The mean is refit to all samples weighted by exp(S / (λ·std S)), λ = 0.57, which matches CEM's 10% elite selection pressure. Fixed σ = 1.0. | Exactly 1.6M |
| MCTS | Open-loop UCT with progressive widening (\|children\| ≤ N^0.5), c_UCT = 1.0 on min-max-normalised Q. 16 waves × 250 simulations; default policy = warm-start plan + N(0, 0.3²). Commits the most-visited root child. | Exactly 1.6M |
| iLQR | Box-constrained iLQR on the K stacked scenarios (weights fixed per act), with an exact CVaR active-set objective. Autograd derivatives, 32 restarts batched per iteration, 6-step line search, Levenberg–Marquardt μ schedule. Converged restarts are re-seeded and iterations continue until the budget is spent. | ~1.54M (96%) |

iLQR's budget counts each linearization point as one transition; the extra autograd cost is not charged. iLQR composes the SURF head into its differentiable model as body(s, 0) + h(s, u), exactly as the sampling planners see it.

## How exact reproduction works

Each trial re-seeds torch (seed + 10000) and the env (seed). The planner's numpy RNG is different:

- **Original runs.** The RNG was created once per process as `default_rng(0)` and carried across all the trials, and all the arms, that process ran. So a stored trial's outcome depends on how many real steps preceded it in its original process.
- **Recorded layouts.** `common.py: layout()` records each stored run's layout: shard size and arm position.
- **CEM and MPPI** draw a fixed 160,000 normals per real step. `src/worker.py` therefore recreates the exact RNG state of any trial by drawing and discarding that many normals up front, and work can be split freely.
- **MCTS and iLQR** draw a data-dependent amount, so their jobs replay each original 10-trial shard from its first trial.

Exactness assumes the same numpy random stream and the same GPU kernels. On a different numpy version, GPU model or CUDA build, per-seed returns can drift by float noise; the means should agree within the SEMs above.

## Layout

```
run_main.py                  main result
ablation/run_ablation.py     ablation table
planners/run_planners.py     planner table
common.py                    stored-run layouts, job planner, GPU scheduler, tables
reference/paper_returns.json per-seed returns of the stored paper runs
data/pendulum/               pretrained dynamics checkpoint + input normalizer stats
src/experiment.py            the experiment driver (run_trial: every arm)
src/worker.py                runs one job (a block of trials) in its own process
src/planning/                continuous_cem.py (CEM-CVaR), continuous_planners.py (MPPI/MCTS/iLQR)
src/bnn/                     Bayesian dynamics model, SURF adapter head, surprise/forget/ELBO
src/drift/                   per-dimension drift filter
src/env/                     non-stationary Pendulum wrapper
```

`src/experiment.py` can also be run directly with the original flags (`--arms`, `--planner`, `--mass`, `--gravity`, ...). Run that way, its trials do not use the fast-forward, so only shard-start trials match the stored seeds.
