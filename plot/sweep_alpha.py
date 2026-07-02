"""
sweep_alpha.py -- CVaR tail-fraction (alpha) sweep for the CVaR-CEM risk-averse
FrozenLake demo, on the single-change schedule [(0, 0.7)] (deterministic map that
becomes 0.7-slippery at step 0).

For each alpha in ALPHAS it runs N_TRIALS trials with the discount factor
(planner gamma) pinned to GAMMA, logging the full verbose per-timestep trace to
results/alpha_<alpha>.log (one file per alpha).

Run (from inside the repo, with mbrl + ns_gym importable):
    conda activate nsgym
    python sweep_alpha.py
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
    CVaRCEMAgent, H_PLAN, N_CEM_ITERS, N_CANDIDATES, ELITE_FRAC,
    K_MODELS, N_ROLLOUTS, BETA_EXPLORE, WARM_START, ADAPTIVE_ALPHA,
    ALPHA_MIN, ALPHA_MAX, N_CONFIDENT, SURPRISE_TAU,
)
# Reuse the per-step helpers so the sweep matches the single-run diagnostic exactly.
from run_discrete import move, pred_next_row, ACTS, DIRS

# ── sweep config ───────────────────────────────────────────────────────────────
SCHEDULE = [(0, 0.7)]      # deterministic, then 0.7-slippery at step 0
CHANGE_STEPS = [0]
N_TRIALS = 100
GAMMA = 0.99               # discount factor: planning, decision-making AND return
ALPHAS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # CVaR tail fractions to sweep

K_FORGET = 1                         # forget EVERY post-change step
TRIAL_LEN = 100
LEARN = True                         # accumulate post-change Dirichlet counts

_HERE = pathlib.Path(__file__).parent
RESULTS_DIR = _HERE / "results"

# True post-change directional slip distribution [intend, perp-, perp+] for the
# 0.7-slippery regime; the plotter compares the model's p_dir against this.
TRUE_P_DIR = np.array([0.7, 0.15, 0.15], dtype=np.float64)

_ALL_SA_INPUT = None


def _all_sa_input(obs_dim, act_dim):
    """Cached (obs_dim*act_dim, obs_dim+act_dim) batch of every (s,a) one-hot pair,
    ordered state-major (s=0:a0..a3, s=1:a0..a3, ...)."""
    global _ALL_SA_INPUT
    if _ALL_SA_INPUT is None:
        rows = []
        for s in range(obs_dim):
            for a in range(act_dim):
                v = np.zeros(obs_dim + act_dim, dtype=np.float32)
                v[s] = 1.0
                v[obs_dim + a] = 1.0
                rows.append(v)
        _ALL_SA_INPUT = torch.as_tensor(np.stack(rows), device=device)
    return _ALL_SA_INPUT


@torch.no_grad()
def all_state_pdir(bnn, obs_dim, act_dim):
    """Deterministic posterior-mean directional predictive p_dir for all 16 states.

    Returns (obs_dim, 3): for each state the [intend, perp-, perp+] distribution the
    model currently predicts, averaged over the 4 actions (the slip profile is an
    action-relative quantity, so this is a per-state summary of the learned belief).
    """
    x = _all_sa_input(obs_dim, act_dim)
    saved = bnn.num_weight_groups
    bnn.num_weight_groups = 1
    try:
        alpha, _, _, _ = bnn._forward_alpha(x, sample=False)   # mean weights
    finally:
        bnn.num_weight_groups = saved
    p_dir = (alpha / alpha.sum(-1, keepdim=True)).cpu().numpy()   # (S*A, 3)
    return p_dir.reshape(obs_dim, act_dim, 3).mean(axis=1)        # (S, 3)


def run_alpha(cvar_alpha, log_path, plot_path):
    """Run the full N_TRIALS loop for one CVaR alpha, logging verbose per-step."""
    out = open(log_path, "w")
    def log(*a):
        print(*a, file=out); out.flush()

    rng = np.random.default_rng(seed=0)
    eval_env = build_scheduled_env(SCHEDULE)
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]

    bnn, dyn = make_dirichlet_bnn(obs_dim, act_dim)
    ckpt = SAVE_DIR / DIRICHLET_CKPT
    if ckpt.exists():
        bnn.load(SAVE_DIR)
        ckpt_msg = f"loaded pretrained Dirichlet model from {DIRICHLET_CKPT}"
    else:
        loaded = bnn.load_pretrained_trunk(SAVE_DIR)
        ckpt_msg = (f"no {DIRICHLET_CKPT} found -> loaded {len(loaded)} trunk tensors "
                    f"from bnn_dynamics.pth, Dirichlet head FRESH (correctness only)")
    bnn.num_weight_groups = 20
    bnn.use_counts = LEARN

    # Fixed-alpha risk-averse planner (adaptive_alpha off), discount pinned to GAMMA.
    agent = CVaRCEMAgent(dyn, bnn, eval_env.env.desc, device,
                         n_actions=act_dim, rng=rng,
                         cvar_alpha=cvar_alpha, adaptive_alpha=False, gamma=GAMMA)

    log("=" * 110)
    log(f"CVaR-CEM (Ayan) PLANNER + DIRICHLET-HEAD model + ONLINE COUNTS | "
        f"schedule={SCHEDULE} | change@{CHANGE_STEPS} | trials={N_TRIALS} | "
        f"CONC_PRIOR={CONC_PRIOR} | learn={LEARN}")
    log(f"  {ckpt_msg}")
    log(f"  SWEEP alpha={cvar_alpha} | gamma(discount)={GAMMA}")
    log(f"  planner: H={H_PLAN} I={N_CEM_ITERS} J={N_CANDIDATES} elite_frac={ELITE_FRAC} "
        f"CVaR_alpha={cvar_alpha} K_models={K_MODELS} N_rollouts={N_ROLLOUTS} "
        f"beta={BETA_EXPLORE} gamma={GAMMA} warm_start={WARM_START}")
    log(f"  adaptive_alpha=False alpha_min={ALPHA_MIN} alpha_max={ALPHA_MAX} "
        f"n_confident={N_CONFIDENT} surprise_tau={SURPRISE_TAU} | "
        f"learn=EVERY STEP (conjugate counts)")
    log("=" * 110)

    steps_hist, returns_hist, goals_hist = [], [], []

    for trial in range(N_TRIALS):
        obs, _ = eval_env.reset()
        agent.reset()
        bnn.retain.fill_(1.0)
        bnn.reset_counts()
        drift_v2 = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)  # REAL
        drift_v1 = DriftFilterV1(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)  # shadow
        epi_hist, a0_hist = [], []
        post = 0
        terminated = truncated = False
        reward = 0.0
        total_return = 0.0
        disc = 1.0                       # running gamma^t for the discounted return
        steps = 0
        log(f"\n########## TRIAL {trial+1}/{N_TRIALS} ##########")

        while not (terminated or truncated) and steps < TRIAL_LEN:
            if steps in CHANGE_STEPS:
                drift_v2.reset(); drift_v1.reset()
                agent.notify_change()

            # Log the model's directional predictive for ALL 16 states (belief
            # entering this timestep), so the plotter can compute per-state BNN
            # error against the true slip [0.7,0.15,0.15]. Format is machine-parsed.
            psd = all_state_pdir(bnn, obs_dim, act_dim)   # (16, 3)
            log("PDIR trial={} step={} | ".format(trial + 1, steps)
                + " ".join("{:02d}:{:.4f},{:.4f},{:.4f}".format(si, p[0], p[1], p[2])
                           for si, p in enumerate(psd)))

            s = int(np.argmax(obs))
            agent.surprise_bar = drift_v2.delta_bar
            agent.n_since_change = post
            action = agent.act(obs)
            a = int(np.argmax(action))
            one_hot_action = to_one_hot_action(action)

            pred_row = pred_next_row(dyn, bnn, obs, a, n=obs_dim, n_actions=act_dim)
            intend = move(s, a)

            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            # Discounted realized return with the SAME gamma the planner uses; the
            # hole penalty stays clamped at 0 so the return is the discounted goal
            # reward (gamma^t_goal if the goal is reached, else 0).
            total_return += disc * max(0.0, float(reward))
            disc *= GAMMA
            s2 = int(np.argmax(next_obs))
            d = direction_of(s, a, s2)
            p_on_reached = float(pred_row[s2])

            vs = surprise_dirichlet(dyn, bnn, obs, one_hot_action, next_obs, reward)
            eps = epistemic_dirichlet(dyn, bnn, obs, one_hot_action)
            delta_n = vs["delta_n"]
            excess = delta_n - 1.0

            drift_v2.update(delta_n)
            drift_v1.update(delta_n)
            epi_hist.append(eps["epistemic"]); a0_hist.append(vs["alpha0"])

            post_change = steps >= CHANGE_STEPS[0]
            if LEARN:
                bnn.add_count(s, a, d)
            n_sa = int(bnn.counts[s, a].sum().item())

            pd = vs["p_dir"]
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
        cnt = bnn.counts.sum(-1)
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

    plot_trial_metrics(steps_hist, returns_hist, goals_hist, str(plot_path))
    log(f"\nWrote trial-metrics plot to {plot_path}")
    log(f"  avg steps={np.mean(steps_hist):.3f} | avg return={np.mean(returns_hist):.3f} "
        f"| goal rate={np.mean(goals_hist):.3f}")
    log("\nDONE.")
    out.close()
    return float(np.mean(steps_hist)), float(np.mean(returns_hist)), float(np.mean(goals_hist))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    summary = []
    for cvar_alpha in ALPHAS:
        tag = f"{cvar_alpha:g}".replace(".", "p")
        log_path = RESULTS_DIR / f"alpha_{tag}.log"
        plot_path = RESULTS_DIR / f"alpha_{tag}_metrics.png"
        print(f"[sweep] alpha={cvar_alpha} gamma={GAMMA} -> {log_path}")
        avg_steps, avg_ret, goal_rate = run_alpha(cvar_alpha, log_path, plot_path)
        summary.append((cvar_alpha, avg_steps, avg_ret, goal_rate))
        print(f"        avg_steps={avg_steps:.3f} avg_return={avg_ret:.3f} "
              f"goal_rate={goal_rate:.3f}")

    print("\n=== SWEEP SUMMARY (schedule={}, gamma={}, trials={}) ===".format(
        SCHEDULE, GAMMA, N_TRIALS))
    for cvar_alpha, avg_steps, avg_ret, goal_rate in summary:
        print(f"  alpha={cvar_alpha:<5} | avg_steps={avg_steps:7.3f} | "
              f"avg_return={avg_ret:6.3f} | goal_rate={goal_rate:.3f}")


if __name__ == "__main__":
    main()
