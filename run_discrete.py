"""
run_risk_averse.py -- BNN env-modeling + surprise/forget/learn loop with the CEM +
CVaR-greedy MPC planner of Ayan's note (MPC_Style_Decision_Making_under_Non_
Stationarity), on a scheduled non-stationary 4x4 FrozenLake.

Refactor of notebooks/risk_averse_ayan.py into the repo/ package:
  bnn/      Dirichlet world model + surprise/forget/learn workflow
  planning/ BNNModelPlanner base + CVaRCEMAgent
  env/      scheduled non-stationary FrozenLake
  drift/    the two drift filters
  utils     to_one_hot_action, make_fl_potential
  plot      side-by-side trial-metrics figure

Run (from inside repo/, with mbrl + ns_gym importable):
    conda activate nsgym
    python run_risk_averse.py
"""
import pathlib

import numpy as np
import torch

from config import device, SAVE_DIR, ETA, KAPPA, GAMMA_UNCERTAINTY
from env import build_scheduled_env
from drift import DriftFilterV1, DriftFilterV2
from utils import to_one_hot_action
from plot import plot_trial_metrics
from bnn import (
    make_dirichlet_bnn, surprise_dirichlet, epistemic_dirichlet, forget_dirichlet,
    mean_alpha0, direction_of, CONC_PRIOR, DIRICHLET_CKPT,
)
from planning.cvar_cem import (
    CVaRCEMAgent, H_PLAN, N_CEM_ITERS, N_CANDIDATES, ELITE_FRAC, CVAR_ALPHA,
    K_MODELS, N_ROLLOUTS, BETA_EXPLORE, PLAN_GAMMA, WARM_START, ADAPTIVE_ALPHA,
    ALPHA_MIN, ALPHA_MAX, N_CONFIDENT, SURPRISE_TAU,
)

# ── diagnostic config ─────────────────────────────────────────────────────────
SCHEDULE = [(0, 0.7)]      # deterministic, then slippery at step 1
CHANGE_STEPS = [0]
N_TRIALS = 100
K_FORGET = 1                         # forget EVERY post-change step
TRIAL_LEN = 100
LEARN = True                         # accumulate post-change Dirichlet counts

_HERE = pathlib.Path(__file__).parent
LOG = str(_HERE / "risk_averse_ayan.log")
PLOT = str(_HERE / "risk_averse_ayan_metrics.png")

ACTS = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
DIRS = {0: "intend", 1: "perp-", 2: "perp+", -1: "none"}


def move(s, a, nrow=4, ncol=4):
    r, c = divmod(s, ncol)
    if a == 0:   c = max(c - 1, 0)
    elif a == 1: r = min(r + 1, nrow - 1)
    elif a == 2: c = min(c + 1, ncol - 1)
    elif a == 3: r = max(r - 1, 0)
    return r * ncol + c


@torch.no_grad()
def pred_next_row(dyn, bnn, obs, a_idx, n=16, n_actions=4):
    """The mean transition row T[s,a,:] the planner uses (the Dirichlet mean)."""
    eye_a = torch.eye(n_actions, device=device)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    state = dyn.reset(obs_t)
    nxt, _, _, _ = dyn.sample(eye_a[a_idx].unsqueeze(0), state, deterministic=True)
    p = nxt.clamp_min(0.0).cpu().numpy().ravel() + 1e-6
    return p / p.sum()


def main():
    out = open(LOG, "w")
    def log(*a):
        print(*a, file=out); out.flush()

    rng = np.random.default_rng(seed=0)
    eval_env = build_scheduled_env(SCHEDULE)
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]

    bnn, dyn = make_dirichlet_bnn(obs_dim, act_dim)
    # Prefer the pretrained Dirichlet checkpoint; fall back to a fresh head on the
    # Gaussian trunk (correctness-only) if it's absent.
    ckpt = SAVE_DIR / DIRICHLET_CKPT
    if ckpt.exists():
        bnn.load(SAVE_DIR)
        ckpt_msg = f"loaded pretrained Dirichlet model from {DIRICHLET_CKPT}"
    else:
        loaded = bnn.load_pretrained_trunk(SAVE_DIR)
        ckpt_msg = (f"no {DIRICHLET_CKPT} found -> loaded {len(loaded)} trunk tensors "
                    f"from bnn_dynamics.pth, Dirichlet head FRESH (correctness only)")
    bnn.num_weight_groups = 20
    bnn.use_counts = LEARN          # enable the conjugate Dirichlet-Multinomial update

    # CEM + CVaR-greedy planner.  Plans on the GROUND-TRUTH map reward (goal +1,
    # hole -1, frozen 0, true terminals); transitions still come from the BNN.
    agent = CVaRCEMAgent(dyn, bnn, eval_env.env.desc, device,
                         n_actions=act_dim, rng=rng)

    log("=" * 110)
    log(f"CVaR-CEM (Ayan) PLANNER + DIRICHLET-HEAD model + ONLINE COUNTS (K=3) | "
        f"schedule={SCHEDULE} | change@{CHANGE_STEPS} | trials={N_TRIALS} | "
        f"CONC_PRIOR={CONC_PRIOR} | learn={LEARN}")
    log(f"  {ckpt_msg}")
    log(f"  planner: H={H_PLAN} I={N_CEM_ITERS} J={N_CANDIDATES} elite_frac={ELITE_FRAC} "
        f"CVaR_alpha={CVAR_ALPHA} K_models={K_MODELS} N_rollouts={N_ROLLOUTS} "
        f"beta={BETA_EXPLORE} gamma={PLAN_GAMMA} warm_start={WARM_START}")
    log(f"  adaptive_alpha={ADAPTIVE_ALPHA} alpha_min={ALPHA_MIN} alpha_max={ALPHA_MAX} "
        f"n_confident={N_CONFIDENT} surprise_tau={SURPRISE_TAU} | "
        f"learn=EVERY STEP (conjugate counts)")
    log("=" * 110)

    # Per-trial metrics for the side-by-side summary plot.
    steps_hist, returns_hist, goals_hist = [], [], []

    for trial in range(N_TRIALS):
        obs, _ = eval_env.reset()
        agent.reset()
        bnn.retain.fill_(1.0)           # fresh belief each trial (prior, then relearn)
        bnn.reset_counts()              # post-change counts are per-trial
        drift_v2 = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)  # REAL
        drift_v1 = DriftFilterV1(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)  # shadow
        epi_hist, a0_hist = [], []
        post = 0
        terminated = truncated = False
        reward = 0.0
        total_return = 0.0
        steps = 0
        log(f"\n########## TRIAL {trial+1}/{N_TRIALS} ##########")

        while not (terminated or truncated) and steps < TRIAL_LEN:
            if steps in CHANGE_STEPS:
                drift_v2.reset(); drift_v1.reset()
                agent.notify_change()

            s = int(np.argmax(obs))
            # For the adaptive-alpha feature: feed the agent its current belief.
            agent.surprise_bar = drift_v2.delta_bar
            agent.n_since_change = post
            action = agent.act(obs)
            a = int(np.argmax(action))
            one_hot_action = to_one_hot_action(action)

            pred_row = pred_next_row(dyn, bnn, obs, a, n=obs_dim, n_actions=act_dim)
            intend = move(s, a)

            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            # Cancel the hole penalty in the reported return: hole arrival gives
            # reward -1, which we drop (clamp at 0) so the plotted return is the
            # positive reward earned (goal +1) and never goes negative.
            total_return += max(0.0, float(reward))
            s2 = int(np.argmax(next_obs))
            d = direction_of(s, a, s2)
            p_on_reached = float(pred_row[s2])

            vs = surprise_dirichlet(dyn, bnn, obs, one_hot_action, next_obs, reward)
            eps = epistemic_dirichlet(dyn, bnn, obs, one_hot_action)
            delta_n = vs["delta_n"]
            excess = delta_n - 1.0

            drift_v2.update(delta_n)        # REAL filter (drives forget)
            drift_v1.update(delta_n)        # shadow
            epi_hist.append(eps["epistemic"]); a0_hist.append(vs["alpha0"])

            # ── LEARN: add this realized direction to (s,a)'s counts every step
            #    (conjugate Dirichlet-Multinomial: alpha = retained_prior + counts).
            #    Surprise above was measured BEFORE this update (true predictive).
            post_change = steps >= CHANGE_STEPS[0]
            if LEARN:
                bnn.add_count(s, a, d)
            n_sa = int(bnn.counts[s, a].sum().item())

            pd = vs["p_dir"]
            # Verbose per-step log (toggle on for debugging):
            log(f" t={steps:2d} | {s:2d},{ACTS[a]:5s}->{s2:2d}(obs) | intend={intend:2d}"
                f" dir={DIRS[d]:6s} | p(model s'={s2:2d})={p_on_reached:5.3f}"
                f" || delta_n={delta_n:7.3f} nll={vs['nll']:6.3f} H={vs['entropy']:5.3f}"
                f" alpha0={vs['alpha0']:7.3f}"
                f" p_dir=[{pd[0]:.2f},{pd[1]:.2f},{pd[2]:.2f}]"
                f" || n(s,a)={n_sa:2d}"
                f" || alpha={agent.last_alpha:4.2f} cvar={agent.last_cvar:+6.3f} bonus={agent.last_bonus:6.4f}"
                f" || excess={excess:+7.3f}"
                f" || V2 lam={drift_v2.lambda_hat:+7.3f}(n={drift_v2._n:2d}) "
                f"dbar={drift_v2.delta_bar:6.3f}"
                f" || V1 lam={drift_v1.lambda_hat:7.3f} dbar={drift_v1.delta_bar:6.3f}")

            # Re-inflation: only post-change, every K steps -- v2 estimate drives it.
            if post_change:
                post += 1
                if post % K_FORGET == 0:
                    a0_before = mean_alpha0(dyn, bnn)
                    rho, ret_b, ret_a = forget_dirichlet(bnn, drift_v2)
                    a0_after = mean_alpha0(dyn, bnn)
                    log(f"      >>> FORGET(v2,dirichlet) @ t={steps}: "
                        f"V2 dbar={drift_v2.delta_bar:.3f} "
                        f"drift_est={drift_v2.drift_estimate(KAPPA):+.3f} -> rho={rho:.4f}"
                        f" | retain {ret_b:.4f}->{ret_a:.4f}"
                        f" | mean alpha0 {a0_before:.3f}->{a0_after:.3f}")

            obs = next_obs
            steps += 1

        outcome = ("goal" if reward > 0 else "hole") if terminated else "truncated"
        steps_hist.append(steps)
        returns_hist.append(total_return)
        goals_hist.append(1 if outcome == "goal" else 0)
        epi_a, a0_a = np.array(epi_hist), np.array(a0_hist)
        log(f"  => TRIAL {trial+1} end: {outcome} in {steps} steps")
        log(f"     V2 final: lam={drift_v2.lambda_hat:+.3f} dbar={drift_v2.delta_bar:.3f} "
            f"(n={drift_v2._n})  ||  V1 final: lam={drift_v1.lambda_hat:.3f} "
            f"dbar={drift_v1.delta_bar:.3f}  ||  retain={float(bnn.retain.item()):.4f}")
        log(f"     epistemic(Var_w[p]): mean={epi_a.mean():.5f} max={epi_a.max():.5f} "
            f"last={epi_a[-1]:.5f} | alpha0: mean={a0_a.mean():.3f} last={a0_a[-1]:.3f}")
        # LEARN check: for the most-visited (s,a), do the counts (empirical slip
        # frequencies) match the model's directional predictive p_dir?
        cnt = bnn.counts.sum(-1)                         # (S, A) total per (s,a)
        flat = cnt.flatten()
        for idx in torch.argsort(flat, descending=True)[:3]:
            n_tot = int(flat[idx].item())
            if n_tot == 0:
                break
            s_i = int(idx // act_dim); a_i = int(idx % act_dim)
            c = bnn.counts[s_i, a_i].cpu().numpy()
            emp = c / c.sum()
            obs_i = np.zeros(obs_dim, dtype=np.float32); obs_i[s_i] = 1.0
            act_i = np.zeros(act_dim, dtype=np.float32); act_i[a_i] = 1.0
            mdl = epistemic_dirichlet(dyn, bnn, obs_i, act_i)["p_dir"]
            log(f"     learn ({s_i},{ACTS[a_i]}) n={n_tot:2d} | "
                f"empirical=[{emp[0]:.2f},{emp[1]:.2f},{emp[2]:.2f}] | "
                f"model p_dir=[{mdl[0]:.2f},{mdl[1]:.2f},{mdl[2]:.2f}]")

    # Side-by-side summary: averaged step count / return / goal rate over trials.
    plot_trial_metrics(steps_hist, returns_hist, goals_hist, PLOT)
    log(f"\nWrote trial-metrics plot to {PLOT}")
    log(f"  avg steps={np.mean(steps_hist):.3f} | avg return={np.mean(returns_hist):.3f} "
        f"| goal rate={np.mean(goals_hist):.3f}")

    log("\nDONE.")
    out.close()


if __name__ == "__main__":
    main()
