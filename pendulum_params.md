# Pendulum Experiment Parameters & Tuning Notes

## Environment
- **Problem**: Pendulum-v1 swing-up with non-stationary mass
- **State**: [cos(theta), sin(theta), theta_dot] (3-dim continuous)
- **Action**: torque ∈ [-2.0, 2.0] (1-dim continuous)
- **Non-stationarity**: mass 1.0 → 3.0 at t=80
- **Trial length**: 100-150 steps

## Our Method: BNN + CEM + Surprise/Forget

### BNN Architecture
- **Type**: Gaussian-head Bayesian Neural Network (mean-field VI)
- **Architecture**: 3 hidden layers × 256 units, SiLU activation
- **Input**: [cos, sin, theta_dot, torque] (4-dim)
- **Output**: [delta_cos, delta_sin, delta_theta_dot, reward] (4-dim)
- **Training**: 8000 random transitions, 200 epochs, batch=1024, lr=1e-3
- **KL weight (beta)**: 100.0 (higher to preserve epistemic uncertainty)
- **Prior std**: 1.0
- **Resulting sigma**: ~0.15 (weight posterior std)

### CEM Planner
| Parameter | Value | Notes |
|-----------|-------|-------|
| H_PLAN | 10 | Planning horizon |
| N_CEM_ITERS | 3 | CEM refinement iterations |
| N_CANDIDATES | 128 | Action sequences per iteration |
| ELITE_FRAC | 0.1 | Keep top 10% |
| K_MODELS | 1 | BNN posterior draws (1 for speed) |
| GAMMA | 0.99 | Discount factor |
| CVAR_ALPHA | 1.0 | Risk-neutral (mean return) |

### Surprise/Forget
| Parameter | Value | Notes |
|-----------|-------|-------|
| Drift filter | V2 (signed, empirical baseline) | Learned pre-change baseline |
| Surprise signal | nu2 (raw squared error) | Not calibrated delta_n (nu2 < S) |
| INFLATE_MODE | additive | sigma_w^2 += q_hat * prior_var |
| K_FORGET | 3 | Re-inflate every 3 steps post-change |
| ETA | 0.2 | EWMA rate for shadow V1 filter |
| DELTA_CLIP | 1e3 | Clip per-dim normalized delta |

### Key Observations
1. **delta_n < 1 issue**: The calibrated surprise delta_n = nu^2/S is always < 1
   because the BNN's aleatoric variance (exp(logvar) ≈ 0.5-2.0) is large.
   Model was trained on random exploration data → high noise → high output variance.
2. **Fix**: Use raw nu2 as surprise signal instead of calibrated delta_n.
   V2 filter learns pre-change baseline and detects deviations.
3. **Forget needs additive mode**: retention mode (1/max(delta_bar,1)) doesn't
   trigger when delta_bar < 1. Additive mode uses q_hat directly.
4. **Sigma preservation**: beta=100 during training keeps weight uncertainty
   sigma ≈ 0.15 (vs 0.049 with beta=0.1).

### Future Improvements
- Collect training data with a simple controller (not random) for better model accuracy
- Use longer CEM horizon (15-20) for pendulum swing-up
- Add online finetuning during post-change phase
- Batch CEM candidate evaluation for 10x speedup

## Baselines

### Oracle CEM (gold standard)
- Same CEM params as our method
- Uses TRUE analytical pendulum dynamics
- Upper bound on what CEM can achieve

### ADA-MCTS (from bugfix/version repo)
- MCTS with 5 discretized torque actions
- 500 iterations per planning step
- 10-step random rollout
- Oracle dynamics for rollouts
- Issue: ~50s per episode, highly variable returns

## Results
See separate results file when experiments complete.

---

*Tuned on 2026-07-03 by steven202*
