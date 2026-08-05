"""
sweep_alpha.py -- CVaR tail-fraction (alpha) sweep for the CVaR-CEM risk-averse
FrozenLake demo, on the single-change schedule [(0, 0.7)] (deterministic map that
becomes 0.7-slippery at step 0).

For each (alpha, seed) pair it runs --trials trials with the discount factor
(planner gamma) pinned to GAMMA, logging the full verbose per-timestep trace to
results/alpha_<alpha>[_matched][_seed<k>].log (one file per run).

Two variants:
  mismatched (default) : the pretrained deterministic-dynamics model meets the
                         0.7-slippery env, so the planner is wrong at t=0 and
                         must adapt online (the setting graph2 was made for).
  --matched            : before every trial the Dirichlet counts are primed with
                         N_PRIME_MATCHED pseudo-observations of the TRUE slip
                         distribution [0.7, 0.15, 0.15] per (s,a), so the planner
                         effectively plans under the correct slippery dynamics
                         from step 0.  This is the "policy trained under the
                         slippery dynamics" control: if alpha still moves the
                         curves here, the effect is not model mismatch.

Runs whose log already ends in DONE are skipped (resume-friendly); pass --force
to redo them.  Seed 0 keeps the legacy file names (alpha_0p5.log) so existing
logs remain valid seed-0 runs.

Run (from inside the repo, with mbrl + ns_gym importable):
    conda activate nsgym
    python plot/sweep_alpha.py                            # fill the alpha grid, seed 0
    python plot/sweep_alpha.py --alphas 0.5 --seeds 1 2   # extra seeds for one alpha
    python plot/sweep_alpha.py --matched                  # matched-dynamics baseline
"""
import argparse
import pathlib
import sys

# The script lives in plot/ but imports the repo-root modules; put the root on
# sys.path so `python plot/sweep_alpha.py` works directly.
_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
# Default grid at 0.05 resolution; completed runs are skipped, so re-running the
# script only fills whatever is missing.
ALPHAS = [round(0.05 * k, 2) for k in range(1, 21)]   # 0.05, 0.10, ..., 1.00

K_FORGET = 1                         # forget EVERY post-change step
TRIAL_LEN = 100
LEARN = True                         # accumulate post-change Dirichlet counts

# Shared with plot_sweep_summary.py / plot_bnn_error.py: logs live at repo root.
RESULTS_DIR = _ROOT / "results"

# True post-change directional slip distribution [intend, perp-, perp+] for the
# 0.7-slippery regime; the plotter compares the model's p_dir against this.
TRUE_P_DIR = np.array([0.7, 0.15, 0.15], dtype=np.float64)

# --matched: pseudo-observations of TRUE_P_DIR per (s,a) added to the Dirichlet
# counts before every trial.  Dominates the head prior (alpha0 >> CONC_PRIOR), so
# the predictive p_dir pins to the true slip profile and the K posterior draws
# concentrate on the true dynamics -- the planner is effectively trained on them.
N_PRIME_MATCHED = 100

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


def _prime_matched_counts(bnn):
    """Fill every (s,a)'s Dirichlet counts with N_PRIME_MATCHED pseudo-observations
    of the true post-change slip distribution -- the matched-dynamics baseline."""
    prime = torch.as_tensor(TRUE_P_DIR * N_PRIME_MATCHED,
                            dtype=bnn.counts.dtype, device=bnn.counts.device)
    bnn.counts.copy_(prime.expand_as(bnn.counts))


def run_alpha(cvar_alpha, log_path, plot_path, seed=0, matched=False,
              n_trials=N_TRIALS):
    """Run the full n_trials loop for one (CVaR alpha, seed, variant), logging
    verbose per-step.  Returns (avg_steps, avg_return, frac_goal, frac_hole,
    frac_truncated)."""
    out = open(log_path, "w")
    def log(*a):
        print(*a, file=out); out.flush()

    variant = "matched" if matched else "mismatched"
    rng = np.random.default_rng(seed=seed)
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
    bnn.use_counts = LEARN or matched   # matched priming lives in the counts

    # Fixed-alpha risk-averse planner (adaptive_alpha off), discount pinned to GAMMA.
    agent = CVaRCEMAgent(dyn, bnn, eval_env.env.desc, device,
                         n_actions=act_dim, rng=rng,
                         cvar_alpha=cvar_alpha, adaptive_alpha=False, gamma=GAMMA)

    log("=" * 110)
    log(f"CVaR-CEM (Ayan) PLANNER + DIRICHLET-HEAD model + ONLINE COUNTS | "
        f"schedule={SCHEDULE} | change@{CHANGE_STEPS} | trials={n_trials} | "
        f"CONC_PRIOR={CONC_PRIOR} | learn={LEARN}")
    log(f"  {ckpt_msg}")
    log(f"  SWEEP alpha={cvar_alpha} | gamma(discount)={GAMMA} | seed={seed} | "
        f"variant={variant}")
    if matched:
        log(f"  MATCHED baseline: Dirichlet counts primed with {N_PRIME_MATCHED} "
            f"pseudo-obs of true p_dir={TRUE_P_DIR.tolist()} per (s,a) before every trial")
    log(f"  planner: H={H_PLAN} I={N_CEM_ITERS} J={N_CANDIDATES} elite_frac={ELITE_FRAC} "
        f"CVaR_alpha={cvar_alpha} K_models={K_MODELS} N_rollouts={N_ROLLOUTS} "
        f"beta={BETA_EXPLORE} gamma={GAMMA} warm_start={WARM_START}")
    log(f"  adaptive_alpha=False alpha_min={ALPHA_MIN} alpha_max={ALPHA_MAX} "
        f"n_confident={N_CONFIDENT} surprise_tau={SURPRISE_TAU} | "
        f"learn=EVERY STEP (conjugate counts)")
    log("=" * 110)

    steps_hist, returns_hist, goals_hist, outcomes_hist = [], [], [], []

    for trial in range(n_trials):
        # Deterministic per-(seed, trial) env stream so seeds are reproducible
        # and independent.  Older ns-gym wrappers may not accept a reset seed.
        try:
            obs, _ = eval_env.reset(seed=int(1_000_000 * seed + trial))
        except TypeError:
            obs, _ = eval_env.reset()
        agent.reset()
        bnn.retain.fill_(1.0)
        bnn.reset_counts()
        if matched:
            _prime_matched_counts(bnn)
        drift_v2 = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)  # REAL
        drift_v1 = DriftFilterV1(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)  # shadow
        epi_hist, a0_hist = [], []
        post = 0
        terminated = truncated = False
        reward = 0.0
        total_return = 0.0
        disc = 1.0                       # running gamma^t for the discounted return
        steps = 0
        log(f"\n########## TRIAL {trial+1}/{n_trials} ##########")

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
        outcomes_hist.append(outcome)
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
    n_done = len(outcomes_hist)
    frac = {o: outcomes_hist.count(o) / n_done for o in ("goal", "hole", "truncated")}
    log(f"\nWrote trial-metrics plot to {plot_path}")
    log(f"  avg steps={np.mean(steps_hist):.3f} | avg return={np.mean(returns_hist):.3f} "
        f"| goal rate={np.mean(goals_hist):.3f}")
    log(f"  outcomes: goal={frac['goal']:.3f} hole={frac['hole']:.3f} "
        f"truncated={frac['truncated']:.3f} (n={n_done})")
    log("\nDONE.")
    out.close()
    return (float(np.mean(steps_hist)), float(np.mean(returns_hist)),
            frac["goal"], frac["hole"], frac["truncated"])


def run_name(alpha, seed=0, matched=False):
    """Log/plot basename for one run.  Seed 0 mismatched keeps the legacy
    alpha_<tag> names so pre-existing logs stay valid."""
    name = "alpha_" + f"{alpha:g}".replace(".", "p")
    if matched:
        name += "_matched"
    if seed:
        name += f"_seed{seed}"
    return name


def log_is_complete(path):
    """A run is complete iff its log ends with the DONE sentinel."""
    try:
        with open(path) as f:
            return any(line.startswith("DONE") for line in f)
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS,
                    help="CVaR tail fractions to sweep (default: 0.05..1.0 grid)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="independent seeds per alpha (default: 0)")
    ap.add_argument("--trials", type=int, default=N_TRIALS,
                    help=f"trials per (alpha, seed) run (default: {N_TRIALS})")
    ap.add_argument("--matched", action="store_true",
                    help="matched-dynamics baseline: prime the model with the "
                         "true slippery dynamics before every trial")
    ap.add_argument("--force", action="store_true",
                    help="redo runs whose log is already complete")
    ap.add_argument("--out-dir", type=pathlib.Path, default=RESULTS_DIR,
                    help="directory for logs/plots (default: repo results/)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed in args.seeds:
        for cvar_alpha in args.alphas:
            name = run_name(cvar_alpha, seed=seed, matched=args.matched)
            log_path = args.out_dir / f"{name}.log"
            plot_path = args.out_dir / f"{name}_metrics.png"
            if not args.force and log_is_complete(log_path):
                print(f"[sweep] skip alpha={cvar_alpha} seed={seed} "
                      f"(complete: {log_path.name})", flush=True)
                continue
            print(f"[sweep] alpha={cvar_alpha} seed={seed} "
                  f"variant={'matched' if args.matched else 'mismatched'} "
                  f"gamma={GAMMA} -> {log_path}", flush=True)
            stats = run_alpha(cvar_alpha, log_path, plot_path, seed=seed,
                              matched=args.matched, n_trials=args.trials)
            avg_steps, avg_ret, f_goal, f_hole, f_trunc = stats
            summary.append((cvar_alpha, seed) + stats)
            print(f"        avg_steps={avg_steps:.3f} avg_return={avg_ret:.3f} "
                  f"goal={f_goal:.3f} hole={f_hole:.3f} trunc={f_trunc:.3f}",
                  flush=True)

    print("\n=== SWEEP SUMMARY (schedule={}, gamma={}, trials={}, variant={}) ===".format(
        SCHEDULE, GAMMA, args.trials, "matched" if args.matched else "mismatched"))
    for cvar_alpha, seed, avg_steps, avg_ret, f_goal, f_hole, f_trunc in summary:
        print(f"  alpha={cvar_alpha:<5} seed={seed} | avg_steps={avg_steps:7.3f} | "
              f"avg_return={avg_ret:6.3f} | goal={f_goal:.3f} hole={f_hole:.3f} "
              f"trunc={f_trunc:.3f}")


if __name__ == "__main__":
    main()
