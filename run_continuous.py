"""
run_gaussian.py -- the NON-Dirichlet (purely Gaussian-head) surprise/forget loop,
integrated into the repo/ package and driven by the same CVaR-CEM MPC planner.

This is the repo counterpart of notebooks/diagnose_v2.py + bnn_fl_cem_surprise_drift.py:
a Gaussian (mean, logvar) BNN world model whose one-step innovation is scored as a
Mahalanobis chi-square (E[delta_n] = 1 when calibrated), filtered into a squared-drift
estimate, and turned into RECTIFIER re-inflation of the variational weight covariance
(additive process noise or a retention pull toward the prior).

Contrast with run_risk_averse.py (Dirichlet):
  * model      : bnn.gaussian_model.BayesianDynamicsModel (Gaussian head, delta+normalize)
  * surprise   : bnn.gaussian_workflow.surprise_gaussian (nu^2/S, not -log p / H)
  * forget     : bnn.gaussian_workflow.forget_gaussian (weight-covariance re-inflation,
                 not a Dirichlet `retain` scalar)
  * drift      : the RECTIFIED-excess EWMA (DriftFilterV1) drives forget here; the
                 signed equal-weight filter (DriftFilterV2) runs as a shadow.
The planner (planning.cvar_cem.CVaRCEMAgent) is shared unchanged -- it reads the
next-state categorical off dyn.sample() by clamping/normalizing, which works for
both the Dirichlet p(s') output and the Gaussian mean-delta output.

Run (from inside repo/, with mbrl + ns_gym importable):
    conda activate nsgym
    python run_gaussian.py
"""
import copy
import pathlib

import numpy as np
import torch

from config import device, SAVE_DIR, ETA, KAPPA, GAMMA_UNCERTAINTY
from env import build_scheduled_env
from drift import DriftFilterV1, DriftFilterV2
from utils import to_one_hot_action
from plot import plot_trial_metrics
from bnn import (
    make_gaussian_bnn, surprise_gaussian, forget_gaussian, mean_sigma,
    GAUSSIAN_CKPT, DELTA_CLIP, INFLATE_MODE,
)
from planning.cvar_cem import (
    CVaRCEMAgent, H_PLAN, N_CEM_ITERS, N_CANDIDATES, ELITE_FRAC, CVAR_ALPHA,
    K_MODELS, N_ROLLOUTS, BETA_EXPLORE, PLAN_GAMMA, WARM_START, ADAPTIVE_ALPHA,
    ALPHA_MIN, ALPHA_MAX, N_CONFIDENT, SURPRISE_TAU,
)

# ── diagnostic config ─────────────────────────────────────────────────────────
SCHEDULE = [(0, 1.0), (3, 0.7)]      # deterministic, then slippery at step 3
CHANGE_STEPS = [3]
N_TRIALS = 20
K_FORGET = 1                         # re-inflate every K post-change steps
TRIAL_LEN = 60
NUM_WEIGHT_GROUPS = 20               # planning-rollout M x k split

_HERE = pathlib.Path(__file__).parent
LOG = str(_HERE / "run_gaussian.log")
PLOT = str(_HERE / "run_gaussian_metrics.png")

ACTS = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}


def move(s, a, nrow=4, ncol=4):
    r, c = divmod(s, ncol)
    if a == 0:   c = max(c - 1, 0)
    elif a == 1: r = min(r + 1, nrow - 1)
    elif a == 2: c = min(c + 1, ncol - 1)
    elif a == 3: r = max(r - 1, 0)
    return r * ncol + c


@torch.no_grad()
def pred_next_row(dyn, bnn, obs, a_idx, n=16, n_actions=4):
    """The mean transition row T[s,a,:] the planner uses.  For the Gaussian model
    sample() returns obs + predicted delta; clamp/normalize recovers p(s')."""
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

    bnn, dyn = make_gaussian_bnn(obs_dim, act_dim)
    # OneDTransitionRewardModel.load reads the Gaussian checkpoint (bnn_dynamics.pth)
    # AND the input normalizer stats (env_stats.pickle), both under SAVE_DIR.
    dyn.load(SAVE_DIR)
    bnn.num_weight_groups = NUM_WEIGHT_GROUPS
    # Snapshot the pretrained variational params so each trial starts from a clean,
    # un-inflated model (forget only ever loosens within a trial).
    init_state = copy.deepcopy(bnn.state_dict())

    # Shared CVaR-CEM planner.  Plans on the ground-truth map reward (goal +1,
    # hole -1, frozen 0); transitions come from the Gaussian BNN.
    agent = CVaRCEMAgent(dyn, bnn, eval_env.env.desc, device,
                         n_actions=act_dim, rng=rng)

    log("=" * 110)
    log(f"CVaR-CEM PLANNER + GAUSSIAN-HEAD model + surprise/forget (rectifier "
        f"re-inflation) | schedule={SCHEDULE} | change@{CHANGE_STEPS} | "
        f"trials={N_TRIALS} | INFLATE_MODE={INFLATE_MODE} | K_forget={K_FORGET}")
    log(f"  loaded Gaussian model + normalizer from {GAUSSIAN_CKPT}")
    log(f"  planner: H={H_PLAN} I={N_CEM_ITERS} J={N_CANDIDATES} elite_frac={ELITE_FRAC} "
        f"CVaR_alpha={CVAR_ALPHA} K_models={K_MODELS} N_rollouts={N_ROLLOUTS} "
        f"beta={BETA_EXPLORE} gamma={PLAN_GAMMA} warm_start={WARM_START}")
    log(f"  adaptive_alpha={ADAPTIVE_ALPHA} alpha_min={ALPHA_MIN} alpha_max={ALPHA_MAX} "
        f"n_confident={N_CONFIDENT} surprise_tau={SURPRISE_TAU}")
    log("=" * 110)

    steps_hist, returns_hist, goals_hist = [], [], []

    for trial in range(N_TRIALS):
        obs, _ = eval_env.reset()
        agent.reset()
        bnn.load_state_dict(init_state)   # fresh deterministic model / trial
        # RECTIFIED-excess EWMA drives forget; signed filter is a logged shadow.
        drift_v1 = DriftFilterV1(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)  # REAL
        drift_v2 = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)  # shadow
        epi_hist, ale_hist = [], []
        post = 0
        terminated = truncated = False
        reward = 0.0
        total_return = 0.0
        steps = 0
        log(f"\n########## TRIAL {trial+1}/{N_TRIALS} ##########")

        while not (terminated or truncated) and steps < TRIAL_LEN:
            if steps in CHANGE_STEPS:
                drift_v1.reset(); drift_v2.reset()
                agent.notify_change()

            s = int(np.argmax(obs))
            # For the adaptive-alpha feature: feed the agent its current belief.
            agent.surprise_bar = drift_v1.delta_bar
            agent.n_since_change = post
            action = agent.act(obs)
            a = int(np.argmax(action))
            one_hot_action = to_one_hot_action(action)

            pred_row = pred_next_row(dyn, bnn, obs, a, n=obs_dim, n_actions=act_dim)
            intend = move(s, a)

            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            total_return += max(0.0, float(reward))
            s2 = int(np.argmax(next_obs))
            p_on_reached = float(pred_row[s2])

            # Surprise of the executed transition, measured against the CURRENT
            # model (so the calibration fixed point is reachable), then filtered.
            vs = surprise_gaussian(dyn, bnn, obs, one_hot_action, next_obs, reward)
            delta_n = min(vs["delta_n"], DELTA_CLIP)
            excess = delta_n - 1.0

            drift_v1.update(delta_n)        # REAL filter (drives forget)
            drift_v2.update(delta_n)        # shadow
            epi_hist.append(vs["epistemic"]); ale_hist.append(vs["aleatoric"])

            # Re-inflation: only post-change, every K steps -- v1 estimate drives it.
            post_change = steps >= CHANGE_STEPS[0]
            if post_change:
                post += 1
                if post % K_FORGET == 0:
                    sig_before = mean_sigma(bnn)
                    applied = forget_gaussian(bnn, drift_v1)   # v1 drives forget
                    sig_after = mean_sigma(bnn)
                    ratio = sig_after / sig_before if sig_before > 0 else float("nan")
                    log(f"      >>> FORGET(v1,gaussian) @ t={steps}: "
                        f"V1 dbar={drift_v1.delta_bar:.3f} "
                        f"drift_est={drift_v1.drift_estimate(KAPPA):+.3f}"
                        f" -> applied({INFLATE_MODE})={applied:.4f} | sigma "
                        f"{sig_before:.4f}->{sig_after:.4f} ({100*(ratio-1):+.1f}%)")

            obs = next_obs
            steps += 1

        outcome = ("goal" if reward > 0 else "hole") if terminated else "truncated"
        steps_hist.append(steps)
        returns_hist.append(total_return)
        goals_hist.append(1 if outcome == "goal" else 0)
        epi_a, ale_a = np.array(epi_hist), np.array(ale_hist)
        log(f"  => TRIAL {trial+1} end: {outcome} in {steps} steps")
        log(f"     V1 final: lam={drift_v1.lambda_hat:+.3f} dbar={drift_v1.delta_bar:.3f} "
            f" ||  V2 final: lam={drift_v2.lambda_hat:+.3f} dbar={drift_v2.delta_bar:.3f} "
            f"(n={drift_v2._n})")
        log(f"     epistemic(Var_w[mu]): mean={epi_a.mean():.5f} max={epi_a.max():.5f} "
            f"last={epi_a[-1]:.5f} | aleatoric: mean={ale_a.mean():.5f} last={ale_a[-1]:.5f}")

    plot_trial_metrics(steps_hist, returns_hist, goals_hist, PLOT)
    log(f"\nWrote trial-metrics plot to {PLOT}")
    log(f"  avg steps={np.mean(steps_hist):.3f} | avg return={np.mean(returns_hist):.3f} "
        f"| goal rate={np.mean(goals_hist):.3f}")

    log("\nDONE.")
    out.close()


if __name__ == "__main__":
    main()
