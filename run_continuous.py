"""
run_continuous.py — Gaussian BNN + batched CEM + surprise/forget on Pendulum.

Non-stationarity: pendulum mass 1.0 → 3.0 at t=80.
The BNN detects the change via surprise, the drift filter tracks it,
and forget re-inflates weight covariance to restore adaptability.

Run:  python run_continuous.py
"""

import copy
import pathlib

import numpy as np
import torch

from config import device, ETA, GAMMA_UNCERTAINTY
from env import build_pendulum_env
from drift import DriftFilterV2
from bnn import make_gaussian_bnn, surprise_gaussian, forget_gaussian, mean_sigma
from planning.continuous_cem import ContinuousCEMAgent, H_PLAN, N_CEM_ITERS, N_CANDIDATES, K_MODELS, GAMMA

_HERE = pathlib.Path(__file__).parent
LOG = str(_HERE / "run_pendulum_adapt.log")
PENDULUM_SAVE_DIR = _HERE / "data" / "pendulum"

MASS_SCHEDULE = [(0, 1.0), (80, 3.0)]
CHANGE_STEPS = [80]
N_TRIALS = 20
TRIAL_LEN = 150
K_FORGET = 5


def main():
    out = open(LOG, "w")
    def log(*a):
        print(*a, file=out); out.flush()
        print(*a)

    eval_env = build_pendulum_env(MASS_SCHEDULE)
    obs_dim, act_dim = 3, 1

    bnn, dyn = make_gaussian_bnn(obs_dim, act_dim)
    dyn.load(str(PENDULUM_SAVE_DIR))
    log(f"Loaded pretrained BNN from {PENDULUM_SAVE_DIR}")
    bnn.num_weight_groups = 1
    init_state = copy.deepcopy(bnn.state_dict())

    agent = ContinuousCEMAgent(dyn, bnn, obs_dim, act_dim, device=device,
                               horizon=H_PLAN, n_cem_iters=N_CEM_ITERS,
                               n_candidates=N_CANDIDATES,
                               k_models=K_MODELS, gamma=GAMMA)

    log("=" * 80)
    log(f"Pendulum CEM+BNN+forget | mass_schedule={MASS_SCHEDULE} | "
        f"change@{CHANGE_STEPS} | trials={N_TRIALS} | K_forget={K_FORGET}")
    log(f"  H={H_PLAN} I={N_CEM_ITERS} J={N_CANDIDATES} K={K_MODELS} gamma={GAMMA}")
    log("=" * 80)

    returns_hist = []

    for trial in range(N_TRIALS):
        obs, _ = eval_env.reset()
        agent.reset()
        bnn.load_state_dict(init_state)
        drift = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)

        total_return = 0.0
        post = 0

        for step in range(TRIAL_LEN):
            if step in CHANGE_STEPS:
                drift.reset()
                agent.notify_change()

            action = agent.act(obs)
            next_obs, reward, term, trunc, _ = eval_env.step(action)
            total_return += float(reward)

            # Surprise + forget
            act_arr = np.asarray(action, dtype=np.float32).ravel()
            vs = surprise_gaussian(dyn, bnn, obs, act_arr, next_obs, reward)
            raw_s = min(vs["nu2"], 50.0)    # clip to prevent overflow
            drift.update(max(raw_s, 1e-6))

            post_change = step >= CHANGE_STEPS[0]
            if post_change:
                post += 1
                if post % K_FORGET == 0:
                    sig_before = mean_sigma(bnn)
                    applied = forget_gaussian(bnn, drift, inflate_mode="additive",
                                              q_max=0.1)
                    sig_after = mean_sigma(bnn)
                    if applied > 0:
                        log(f"  FORGET t={step}: applied={applied:.4f} "
                            f"sigma {sig_before:.4f}->{sig_after:.4f}")

            if step % 30 == 0:
                a_val = float(act_arr[0])
                log(f"  t={step:3d} | a={a_val:+.3f} r={reward:+.3f} | "
                    f"nu2={raw_s:.3f} | V2 lam={drift.lambda_hat:+.3f} | "
                    f"sigma={mean_sigma(bnn):.4f}")

            obs = next_obs
            if term or trunc:
                break

        returns_hist.append(total_return)
        log(f"  => TRIAL {trial+1}: return={total_return:.1f} | "
            f"V2 baseline={drift.baseline:.3f} lam={drift.lambda_hat:+.3f}")

    log(f"\n{'='*60}")
    log(f"RESULTS: avg={np.mean(returns_hist):.1f} std={np.std(returns_hist):.1f} "
        f"min={np.min(returns_hist):.1f} max={np.max(returns_hist):.1f}")
    log("DONE.")
    out.close()


if __name__ == "__main__":
    main()
