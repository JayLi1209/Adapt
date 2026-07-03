"""
run_continuous.py -- Gaussian BNN + surprise/forget + continuous CEM on Pendulum.

Non-stationarity: pendulum mass changes mid-episode. The Gaussian BNN world model
detects this via Mahalanobis-chi-square surprise, filters the excess into a drift
estimate, and re-inflates the variational weight covariance (forget) to restore
epistemic uncertainty.  A continuous-action CEM planner uses the BNN for rollout
trajectory simulation.

Run:  python run_continuous.py
"""

import copy
import pathlib

import numpy as np
import torch
from torch import optim

from config import device, ETA, GAMMA_UNCERTAINTY
from env import build_pendulum_env
from drift import DriftFilterV1, DriftFilterV2
from plot import plot_trial_metrics
from bnn import (
    make_gaussian_bnn, surprise_gaussian, forget_gaussian, mean_sigma,
    DELTA_CLIP,
)
from planning.continuous_cem import (
    ContinuousCEMAgent, H_PLAN, N_CEM_ITERS, N_CANDIDATES, ELITE_FRAC,
    CVAR_ALPHA, K_MODELS, GAMMA,
)

# ── Pendulum config ─────────────────────────────────────────────────────────────
MASS_SCHEDULE = [(0, 1.0), (80, 3.0)]   # mass 1.0, then 3.0 at step 80
CHANGE_STEPS = [80]
N_TRIALS = 20
TRIAL_LEN = 150
K_FORGET = 3                            # re-inflate every K post-change steps
ONLINE_FINETUNE = True                  # online gradient updates on recent data
FINETUNE_LR = 1e-4
FINETUNE_EVERY = 5                      # finetune every N steps
BUFFER_SIZE = 200                       # sliding window of recent transitions

_HERE = pathlib.Path(__file__).parent
LOG = str(_HERE / "run_pendulum_adapt.log")
PLOT = str(_HERE / "run_pendulum_adapt_metrics.png")
PENDULUM_SAVE_DIR = _HERE / "data" / "pendulum"


def main():
    out = open(LOG, "w")
    def log(*a):
        print(*a, file=out); out.flush()

    rng = np.random.default_rng(seed=0)
    eval_env = build_pendulum_env(MASS_SCHEDULE)
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]

    bnn, dyn = make_gaussian_bnn(obs_dim, act_dim)
    # Load pretrained Pendulum BNN.
    ckpt_path = PENDULUM_SAVE_DIR / "bnn_dynamics.pth"
    if ckpt_path.exists():
        dyn.load(str(PENDULUM_SAVE_DIR))
        log(f"Loaded pretrained Pendulum BNN from {ckpt_path}")
    else:
        log(f"WARNING: no pretrained BNN at {ckpt_path} -- using random init")

    bnn.num_weight_groups = 1  # per-sample weight draws for planning
    init_state = copy.deepcopy(bnn.state_dict())

    agent = ContinuousCEMAgent(
        dyn, bnn, obs_dim, act_dim, device=device,
        horizon=H_PLAN, n_cem_iters=N_CEM_ITERS, n_candidates=N_CANDIDATES,
        elite_frac=ELITE_FRAC, k_models=K_MODELS,
        cvar_alpha=CVAR_ALPHA, gamma=GAMMA, rng=rng,
    )

    log("=" * 80)
    log(f"Pendulum surprise/forget/learn | mass_schedule={MASS_SCHEDULE} | "
        f"change@{CHANGE_STEPS} | trials={N_TRIALS} | "
        f"inflate_mode=additive+V2 | K_forget={K_FORGET} | "
        f"online_finetune={ONLINE_FINETUNE}")
    log(f"  planner: H={H_PLAN} I={N_CEM_ITERS} J={N_CANDIDATES} "
        f"elite_frac={ELITE_FRAC} K_models={K_MODELS} "
        f"gamma={GAMMA} CVaR_alpha={CVAR_ALPHA}")
    log("=" * 80)

    steps_hist, returns_hist, goals_hist = [], [], []

    for trial in range(N_TRIALS):
        obs, _ = eval_env.reset()
        agent.reset()
        bnn.load_state_dict(init_state)

        drift_v1 = DriftFilterV1(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)  # shadow
        drift_v2 = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)  # drives forget
        buffer = []  # sliding window of (obs, act, next_obs, reward)
        optimizer = optim.Adam(bnn.parameters(), lr=FINETUNE_LR) if ONLINE_FINETUNE else None

        post = 0
        terminated = truncated = False
        total_return = 0.0
        steps = 0
        episode_rewards = []
        log(f"\n###### TRIAL {trial+1}/{N_TRIALS} ######")

        while not (terminated or truncated) and steps < TRIAL_LEN:
            if steps in CHANGE_STEPS:
                drift_v1.reset(); drift_v2.reset()
                agent.notify_change()

            agent.surprise_bar = drift_v2.delta_bar
            agent.n_since_change = post

            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = eval_env.step(action)

            total_return += float(reward)
            episode_rewards.append(float(reward))
            a_val = float(np.asarray(action).ravel()[0])

            # Surprise: use per-dim squared error (nu2) as the raw signal.
            # The drift filter V2 learns a pre-change baseline & detects deviations.
            act_onehot = np.asarray(action, dtype=np.float32).ravel()
            vs = surprise_gaussian(dyn, bnn, obs, act_onehot, next_obs, reward)
            raw_surprise = max(vs["nu2"], 1e-6)
            delta_n = min(vs["delta_n"], DELTA_CLIP)
            drift_v1.update(raw_surprise)           # shadow
            drift_v2.update(raw_surprise)           # drives forget

            # Store transition in sliding buffer for online finetuning.
            buffer.append((obs.copy(), act_onehot.copy(), next_obs.copy(), float(reward)))
            if len(buffer) > BUFFER_SIZE:
                buffer.pop(0)

            # ── Forget (re-inflate weight covariance) ──────────────────────────
            post_change = steps >= CHANGE_STEPS[0]
            if post_change:
                post += 1
                if post % K_FORGET == 0:
                    sig_before = mean_sigma(bnn)
                    applied = forget_gaussian(bnn, drift_v2, inflate_mode="additive")
                    sig_after = mean_sigma(bnn)
                    ratio = sig_after / sig_before if sig_before > 0 else float("nan")
                    log(f"    FORGET @ t={steps}: applied={applied:.4f} "
                        f"sigma {sig_before:.4f}->{sig_after:.4f} "
                        f"({100*(ratio-1):+.1f}%)")

            # ── Online finetuning ──────────────────────────────────────────────
            if ONLINE_FINETUNE and optimizer is not None and steps % FINETUNE_EVERY == 0 and len(buffer) >= 32:
                idx = rng.choice(len(buffer), size=min(32, len(buffer)), replace=False)
                for i in idx:
                    o, a, no, r = buffer[i]
                    model_in = np.concatenate([o, a]).astype(np.float32)
                    target = np.concatenate([no - o, [r]]).astype(np.float32)
                    model_in_t = torch.tensor(model_in, device=device).unsqueeze(0)
                    target_t = torch.tensor(target, device=device).unsqueeze(0)
                    bnn.num_train_points = max(len(buffer), 1)
                    loss, _ = bnn.loss(model_in_t, target_t)
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if steps % 20 == 0:
                log(f"  t={steps:3d} | a={a_val:+.3f} r={reward:+.3f} | "
                    f"delta_n={delta_n:.3f} | V2 dbar={drift_v2.delta_bar:.3f} baseline={drift_v2.baseline:.3f} | "
                    f"sigma={mean_sigma(bnn):.5f}")

            obs = next_obs
            steps += 1

        avg_rew = np.mean(episode_rewards) if episode_rewards else 0.0
        outcome = "terminated" if terminated else ("truncated" if truncated else "timeout")
        steps_hist.append(steps)
        returns_hist.append(total_return)
        goals_hist.append(1.0 if total_return > -800 else 0.0)  # heuristic "success" threshold
        log(f"  => TRIAL {trial+1} end: {outcome} in {steps} steps | "
            f"total_return={total_return:.3f} avg_rew={avg_rew:.3f} | "
            f"V2 final: dbar={drift_v2.delta_bar:.3f} baseline={drift_v2.baseline:.3f} | "
            f"sigma={mean_sigma(bnn):.5f}")

    plot_trial_metrics(steps_hist, returns_hist, goals_hist, PLOT)
    log(f"\nWrote trial-metrics plot to {PLOT}")
    log(f"  avg steps={np.mean(steps_hist):.1f} | avg return={np.mean(returns_hist):.1f} "
        f"| goal-like rate={np.mean(goals_hist):.2f}")

    log("\nDONE.")
    out.close()


if __name__ == "__main__":
    main()
