"""Run ADA-MCTS baseline on the same non-stationary FrozenLake schedule as our
CVaR-CEM method, for a fair head-to-head comparison.

    conda activate nsgym  (or:  source .venv/bin/activate)
    python run_ada_mcts_baseline.py
"""

import pathlib
import time

import numpy as np
import torch

from config import device, SAVE_DIR
from env import build_scheduled_env
from bnn import make_dirichlet_bnn, direction_of
from planning.ada_mcts import ADAMCTSAgent
from plot import plot_trial_metrics

# ── config ────────────────────────────────────────────────────────────────────
SCHEDULE = [(0, 1.0), (1, 0.7)]
CHANGE_STEPS = [1]
N_TRIALS = 100
TRIAL_LEN = 100
M_SIMULATIONS = 3000

_HERE = pathlib.Path(__file__).parent
LOG = str(_HERE / "ada_mcts_baseline.log")
PLOT = str(_HERE / "ada_mcts_baseline_metrics.png")


def main():
    out = open(LOG, "w")

    def log(*a):
        print(*a, file=out)
        out.flush()

    rng = np.random.default_rng(seed=0)
    eval_env = build_scheduled_env(SCHEDULE)
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]

    bnn, dyn = make_dirichlet_bnn(obs_dim, act_dim)
    bnn.load(SAVE_DIR)
    bnn.num_weight_groups = 20
    bnn.use_counts = True

    agent = ADAMCTSAgent(dyn, bnn, eval_env.env.desc, device,
                         n_actions=act_dim, rng=rng,
                         m_simulations=M_SIMULATIONS)

    log("=" * 80)
    log(f"ADA-MCTS (PURE MCTS-{M_SIMULATIONS}) BASELINE")
    log(f"  schedule={SCHEDULE} | change@{CHANGE_STEPS} | trials={N_TRIALS}")
    log(f"  m_simulations={M_SIMULATIONS} | cp=sqrt(2) | h_rollout=6")
    log(f"  online counts ENABLED (per-step learn)")
    log("=" * 80)

    steps_hist, returns_hist, goals_hist = [], [], []
    t0 = time.time()

    for trial in range(N_TRIALS):
        obs, _ = eval_env.reset()
        agent.reset()
        bnn.retain.fill_(1.0)
        bnn.reset_counts()

        terminated = truncated = False
        total_return = 0.0
        steps = 0

        while not (terminated or truncated) and steps < TRIAL_LEN:
            if steps in CHANGE_STEPS:
                agent.notify_change()

            action = agent.act(obs)
            a = int(np.argmax(action))
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            total_return += max(0.0, float(reward))

            s = int(np.argmax(obs))
            s2 = int(np.argmax(next_obs))
            if steps >= CHANGE_STEPS[0]:
                agent.learn(s, a, s2)

            obs = next_obs
            steps += 1

        outcome = ("goal" if reward > 0 else "hole") if terminated else "truncated"
        steps_hist.append(steps)
        returns_hist.append(total_return)
        goals_hist.append(1 if outcome == "goal" else 0)

        elapsed = time.time() - t0
        g_rate = np.mean(goals_hist)
        log(f"  TRIAL {trial+1:3d}/{N_TRIALS}: {outcome:>9s} in {steps:3d} steps  |  "
            f"goal rate {g_rate:.3f}  |  elapsed {elapsed:.0f}s")

    plot_trial_metrics(steps_hist, returns_hist, goals_hist, PLOT)
    log(f"\nWrote plot to {PLOT}")
    log(f"  avg steps={np.mean(steps_hist):.3f} | avg return={np.mean(returns_hist):.3f} "
        f"| goal rate={np.mean(goals_hist):.3f}")
    log(f"  total time: {time.time() - t0:.0f}s")
    log("\nDONE.")
    out.close()


if __name__ == "__main__":
    main()
