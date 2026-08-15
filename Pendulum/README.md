# Ant — frozen body + adapter head on non-stationary Pendulum

Self-contained replication package. Everything needed to regenerate the two
result tables in §7 is in this folder; it has no dependency on the parent
repo.

Frozen-body adaptation on non-stationary Pendulum. The BNN body never changes;
a small head owns the correction. Two head families are compared against the
existing surprise→forget→inflate scheme, on two single-parameter changes.

Everything below is what the code actually does as of this file's writing.
Constants are quoted with `file:line` so they can be checked rather than trusted.

---

## 1. Provenance

| | |
|---|---|
| git base commit | `712e488` (working tree dirty; see §8 for the changed files) |
| driver script | `test_pendulum_default_head.py` |
| adapter implementations | `bnn/gain_adapter.py` |
| checkpoint | `data/pendulum/bnn_dynamics.pth`, md5 `6f664434e602356094490d66b806f551` |
| normalizer | `data/pendulum/env_stats.pickle`, md5 `864d5b211c738060091e58dce940d2a4` |
| external deps | `torch`, `numpy`, `gymnasium`, `mbrl` |

### Layout

```
Ant/
  README.md                      this file
  test_pendulum_default_head.py  the driver (all 6 arms)
  config.py                      device selection
  defaults.py                    the 5 constants inherited from the parent repo
  bnn/     layers.py  gaussian_model.py  gaussian_workflow.py  gain_adapter.py
  drift/   filters.py
  env/     pendulum.py
  planning/continuous_cem.py
  data/pendulum/                 bnn_dynamics.pth, env_stats.pickle
  results/nnhead_mass/           scenario A, seed 1004  (the §7 table)
  results/nnhead_grav/           scenario B, seed 1004  (the §7 table)
  results/pendulum_grav20/       scenario B, 10 trials with SEMs
```

The package `__init__.py` files are trimmed relative to the parent repo: the
Dirichlet model and the FrozenLake envs are not imported, so `grids` and
`ns_gym` are not required. `defaults.py` replaces `run_pendulum_default.py`
(which pulls in that whole stack) and holds the same five constants; line
numbers in §5 are unchanged from the original file.

`ARMS` must read exactly:
```python
ARMS = ["no_adapt", "forget_elbo", "head_lin", "head", "head_noforget", "oracle"]
```
(`test_pendulum_default_head.py:63`). An arm named `head_anchoronly` is **not**
part of this configuration; if it is present the tree is ahead of this spec.

---

## 2. Reproduce

```bash
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Scenario A — mass 1 -> 4 @ ts0
python3 -u test_pendulum_default_head.py --steps 500 --trial-start 4 --trial-end 5 \
    --mass 4.0 --gravity 10.0 --tag s1004 --out-dir results/nnhead_mass

# Scenario B — gravity 10 -> 20 @ ts0
python3 -u test_pendulum_default_head.py --steps 500 --trial-start 4 --trial-end 5 \
    --mass 1.0 --gravity 20.0 --tag s1004 --out-dir results/nnhead_grav
```

Run from inside `Ant/`. `--trial-start 4` selects seed `SEED_BASE + 4 = 1004`.
Multi-trial runs shard by trial index and merge with `--merge`.

**Run all six arms in one process, in `ARMS` order, to reproduce §7 exactly.**
`ContinuousCEMAgent` draws its candidates from numpy's *global* RNG, which
`torch.manual_seed(seed + 10000)` does not reset, so the arms share one stream.
Running a single arm in isolation consumes a different stream and shifts the
return by ~0.05%: verified, `head_lin`/A gives -1184.4 alone versus -1183.6 in
the full sequence, and `head`/B gives -3413.4 alone versus -3414.0. Same
conclusions, not the same digits. Seeding numpy per trial would fix this but
would invalidate the recorded table, so it is deliberately left alone.

---

## 3. Pretrained environment

`data/pendulum/`, produced by `pretrain_pendulum.py`. Real environment rollouts,
not analytic sampling.

| | |
|---|---|
| dynamics | gymnasium `Pendulum-v1`, **mass 1.0, gravity 10.0**, length 1.0, dt 0.05, max_speed 8.0, **\|u\| ≤ 2** |
| collection | 20,000 env steps; torque `1.2·sin(0.05·i) + N(0, 0.9²)` clipped to ±2 |
| coverage | teleport every 25 steps — 50% to near-upright (θ~N(0,0.35²), θ̇~N(0,0.8²)), 50% uniform (θ~U(−π,π), θ̇~U(−4,4)) |
| goal weighting | `w = ((cosθ+1)/2)² · exp(−½(θ̇/2)²) + 0.05` (CLAUDE.md requirement) |
| training | 300 epochs, batch 2048, Adam lr 1e-3, β 1.0 |
| architecture | `layer_sizes = [4, 256, 256, 8]` → `load_arch` returns `(hid=256, num_layers=3)` |
| head | diagonal Gaussian, 4 channels: Δcos, Δsin, Δθ̇, reward (`learned_rewards=True`, `target_is_delta=True`) |
| reward channel | trained on reward rescaled affinely to [0,1] |

**Measured property of this checkpoint.** Its own action gain
`w₀ = dμ/du = [0.00116, 0.00345, 0.12797]` (central difference, `measure_gain`,
3000 uniform states, seed 3). The analytic mass-1 value is **0.150**, so the
checkpoint under-learned its torque response by ~15%, and the gain is mildly
state-dependent (0.098 restricted to bottom states) when physically it is not.
Both are artifacts of training at |u| ≤ 2 where the control term is small next
to gravity.

---

## 4. Deployment scenarios

Identical to pretraining except one parameter, changed at ts 0 and never announced
(the agent gets `notify_change()` only).

|  | passive `3g/(2l)·sinθ` | control `3u/(m l²)` | true `w* = 3·dt/(m l²)` | torque authority |
|---|---|---|---|---|
| pretrained (m=1, g=10) | 15.0 | 3.00 | 0.1500 | 6 / 15 = 0.40 |
| **A: mass 1→4** | 15.0 (unchanged) | **0.75** | **0.0375** | 1.5 / 15 = 0.10 |
| **B: gravity 10→20** | **30.0** | 3.00 (unchanged) | 0.1500 (unchanged) | 6 / 30 = 0.20 |

The two changes are complementary: A moves only the control term, B moves only
the passive term. This is the whole point of the pairing — an adapter spanning
only the action channel covers A exactly and B not at all.

---

## 5. Run configuration

| parameter | value | source |
|---|---|---|
| episode length | **500** steps | `--steps 500` (default `TRIAL_LEN` = `rpd.TRIAL_LEN` = 200) |
| truncation | **none** | `PendulumWrapper` wraps raw `PendulumEnv`; run stops at `--steps` |
| discount γ | 0.99 | `rpd.GAMMA`, `:53` |
| seed | 1004 | `SEED_BASE=1000`, `:54`, + `--trial-start 4` |
| trials | 1 per cell | see §7 caveat |
| actuator \|u\| | 2.0 | `MAX_T`, `:51` |
| planner | CEM + CVaR | `ContinuousCEMAgent` |
| horizon H | 40 | `:52` |
| CEM iterations | 8 | `:52` |
| candidates J | 500 | `:52` |
| elite fraction | 0.1 | `:52` |
| posterior draws K | 10 | `:52` |
| CVaR α | 1.0 | `:52` |
| planner reward | **analytic** `pendulum_reward` | matches `run_pendulum_default.py` (`planner_reward=analytic`) |
| imagined transitions / decision | J·K·H·iters = **1,600,000** | derived |
| `aleatoric_in_rollout` | **False** | deterministic env, p(o′\|o,a)=1 |
| surprise | per-dimension `δ_n = ν²/S` | `N_SURP=3`, `ACTIVE=[0,1,2]`, `:55` |
| surprise draws | 8 | `surprise_composed` default |
| drift filter | `PerDimDriftFilter`, equal-weight cumulative, baseline 1.0 | `drift/filters.py` |
| filter η / γ_unc / κ | 0.2 / True / 0.0 | `config.ETA`, `config.GAMMA_UNCERTAINTY`, `drift.filters.KAPPA` |
| `k_forget` | 5 | `rpd.K_FORGET`, `:54` |
| forget mode | retention, `q_max` 2.0 | `Q_MAX`, `:56` |
| `q_scale` / deadband / `forget_mean` | 1.0 / 0.0 / False | `bnn/gaussian_workflow.py` |
| body ELBO lr / steps / MC / β | 1e-3 / 5 / 3 / 1.0 | `RETRAIN_LR, RETRAIN_STEPS, N_MC, BETA`, `:56` |
| body ELBO batch | **full replay buffer** | `torch.cat(buf_in)` |
| adapter hidden / lr / prior_std | 64 / **3e-3** / 0.1 | `ADAPT_HID, ADAPT_LR, ADAPT_PRIOR`, `:64` |
| adapter ELBO steps / MC / batch | 5 / 3 / **256** | `ADAPT_STEPS, ADAPT_MC, ADAPT_BATCH`, `:65` |
| adapter init σ | 1e-3 (skip + L2) | `NonlinearAdapterHead.__init__` |
| adapter parameter count | 1,060 | 4→64→3 plus a 4→3 skip |
| balance criterion | \|θ\| ≤ 0.2 rad and \|θ̇\| ≤ 1.0 held 20 steps | `UP_ANGLE, UP_SPEED, UP_HOLD`, `:57` |

---

## 6. Arms

| arm | body | head | update rule |
|---|---|---|---|
| `no_adapt` | frozen | — | nothing. Floor. |
| `forget_elbo` | **trained** | — | per-dim surprise → retention inflation of body head rows every 5 steps + re-anchor; whole-network ELBO **every step** on the full buffer |
| `head_lin` | frozen | conjugate linear, 3 params | `Λ ← λΛ + u²/σ_n²`, `b ← λb + u·r/σ_n²`, `w = b/Λ`, `λ = clip(1/max(δ̄,1), 1e-3, 1)`, closed form **every step** |
| `head` | frozen | nonlinear Bayesian, 1060 params | ELBO **every step** (5 steps × 3 MC, batch 256) + retention inflation of the **adapter** every 5 steps + re-anchor |
| `head_noforget` | frozen | same as `head` | ELBO only — **no inflation and no re-anchoring** |
| `oracle` | — | — | true dynamics substituted into the rollout via `alt_dynamics_fn`, same warm-starting agent. Ceiling. |

**Composed model** (both head families): `μ(s,u) = μ_BNN(s,0) + h(s,u)`.
The body is always evaluated at **u = 0** — in-support for any pretraining torque
range, and it isolates the passive dynamics the body already gets right.

**Nonlinear head:** `h(x) = Skip(x) + L2(silu(L1(x)))`, `x = [cosθ, sinθ, θ̇, u]`
raw, all three layers `BayesianLinear`. Warm-started so `h(s,u) = w₀·u` at t=0
(`Skip.weight_mu` zero except the u column = `w₀`; `L2.weight_mu` zero, σ=1e-3),
hence the composed model reproduces the body before any data arrives. Verified:
`h(s, u=2) = [0.002, 0.006, 0.256]` vs `w₀·2 = [0.002, 0.007, 0.256]`, KL ≈ 7.5e−5.

---

## 7. Results obtained (seed 1004, 500 steps, n=1 per cell)

### Scenario A — mass 1 → 4

| arm | ret(500) | ret@200 | swing-up | balance t | 1-step | δ_n med | w[θ̇] |
|---|---|---|---|---|---|---|---|
| `no_adapt` | −2555.2 | −1303.4 | no | — | 0.1803 | — | — |
| `forget_elbo` | −1704.4 | −1200.2 | yes | 205 | 0.0737 | 0.116 | — |
| `head_lin` | **−1183.6** | −1183.4 | yes | **202** | **0.0222** | 0.039 | **0.03741** |
| `head` | −1916.6 | −1390.2 | yes | 260 | 0.0653 | 0.205 | 0.10632 |
| `head_noforget` | −2139.6 | −1294.4 | yes | 389 | 0.1055 | 1.212 | 0.12109 |
| `oracle` | −1083.9 | −1083.7 | yes | 184 | 0.0975 | — | — |

### Scenario B — gravity 10 → 20

| arm | ret(500) | ret@200 | swing-up | min\|θ\| | 1-step | δ_n med | w[θ̇] |
|---|---|---|---|---|---|---|---|
| `no_adapt` | −3540.0 | −1433.4 | no | 1.382 | 0.6347 | — | — |
| `forget_elbo` | −4121.4 | −1654.8 | no | 1.530 | 0.2044 | 0.042 | — |
| `head_lin` | −3844.8 | −1551.0 | no | 1.711 | 0.7955 | **5.585** | **−1.29042** |
| `head` | **−3414.0** | −1412.4 | no | 1.111 | **0.1235** | 0.113 | 0.08565 |
| `head_noforget` | −3569.6 | −1440.8 | no | 1.152 | 0.5372 | 19.267 | 0.12114 |
| `oracle` | −3361.9 | −1371.8 | no | 1.133 | 0.5978 | — | — |

Headline: the linear head is exact on A (w → 0.03741 vs target 0.0375) and
collapses on B (w driven to −1.29, δ_n median 5.6, 77% of steps surprised). The
nonlinear head is the best non-oracle arm on B but loses to the linear one on A.
`forget_elbo` is **worse than doing nothing** on B despite having a better model.

### Supporting run with error bars — Scenario B only

`results/pendulum_grav20/`, 10 trials (seeds 1000–1009), 500 steps, 4 arms:

| arm | ret(500) ± SEM | swing-up | 1-step |
|---|---|---|---|
| `no_adapt` | −3495.4 ± 13.9 | 0/10 | 0.6510 |
| `forget_elbo` | −4407.9 ± 44.4 | 0/10 | 0.1901 |
| `head_lin` | −3410.9 ± 378.9 | 1/10 | 1.0615 |
| `oracle` | −2974.1 ± 330.5 | 1/10 | 0.5511 |

---

## 8. Known confounds — read before drawing conclusions

1. **n = 1 per cell in §7, and seed variance is enormous.** The same linear head
   scores −126.1 with balance at t=57 on seed 1000 but −1183.6 with balance at
   t=202 on seed 1004. Single-trial orderings on this task are not reliable.
2. **`head_noforget` is not a clean ablation.** It removes inflation *and*
   re-anchoring. Without re-anchoring the KL prior stays `N(warm_start, 1e-3²)`
   for the whole run and pins the adapter to its initialisation — its final
   `w[θ̇]` (0.121) is essentially `w₀` (0.128). Any claim that "forgetting helps"
   from this arm is unsupported; the clean ablation keeps re-anchoring and drops
   only inflation.
3. **Learning rate is not matched**: body ELBO 1e-3, adapter ELBO 3e-3.
4. **Batch is not matched**: body uses the full buffer, adapter a 256 minibatch.
   Identical for the first 256 steps, then divergent.
5. **Forget cadence is not matched**: `head_lin` applies λ **every step** inside
   its recursion; `head` and `forget_elbo` apply ρ **every 5 steps**. The linear
   head therefore forgets 5× more often.
6. **Scenario B is compressed against a floor.** Torque authority is 0.20 there,
   and the *oracle* only reaches 1/10 swing-ups. Trust the one-step-error column;
   the returns have little room to separate.
7. **Older result directories predate the `head` rename.** In
   `results/pendulum_default_head_500/` (seed **1000**, not 1004) the arm called
   `head` is the **linear** head. Post-rename, `head` = nonlinear, `head_lin` =
   linear.
8. `results/nnhead_mass_abl/`, `results/nnhead_grav_abl/` are from an arm that no
   longer exists in the code. Ignore them.

---

## 9. Results files

`results/nnhead_mass/shard_s1004.json`, `results/nnhead_grav/shard_s1004.json`,
`results/pendulum_grav20/shard_g*.json`. Each per-trial record carries
`ret`, `ret200`, `pred_err`, `min_abs_theta`, `balanced`, `n_forget`, `w_final`,
and the full traces `per_step_reward`, `theta_trace`, `w_trace`, `dn_trace`,
`dbar_trace`, `fire_trace`, so every column in §7 can be recomputed without
re-running anything.

## 10. What is NOT here

Removed with the rest of the pendulum exploration and not needed for §7:
the torque sweep, terminal-Q, hybrid-rollout, long-retention, H=40 band probe,
fresh-online and ranking-vs-accuracy experiments, and the checkpoints
`data/pendulum_uncapped/`, `data/pendulum_mt*/`, `data/pendulum/{qnet_m1,
qnet_m4,dyn_m4}.pth` (those checkpoints still exist in the parent repo; they are
GPU-hours to regenerate, so they were kept rather than deleted).
