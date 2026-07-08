"""Comprehensive Pendulum experiment: compare 3 methods.

1. Oracle CEM — CEM with true dynamics (upper bound)
2. BNN-CEM (static) — BNN-trained CEM without adaptation
3. BNN-CEM (adaptive) — BNN-CEM with surprise/forget adaptation

Records parameters and results for comparison.
"""

import copy
import math
import pathlib
import time

import numpy as np
import torch
from torch import optim

from config import device, ETA, GAMMA_UNCERTAINTY
from env import build_pendulum_env
from drift import DriftFilterV1, DriftFilterV2
from bnn import (
    make_gaussian_bnn, surprise_gaussian, forget_gaussian, mean_sigma,
    DELTA_CLIP,
)
from planning.continuous_cem import ContinuousCEMAgent

_HERE = pathlib.Path(__file__).parent
PENDULUM_SAVE_DIR = _HERE / "data" / "pendulum"
RESULTS_FILE = str(_HERE / "pendulum_results.md")

# ── Config ──────────────────────────────────────────────────────────────────────
MASS_SCHEDULE = [(0, 1.0), (80, 3.0)]
CHANGE_STEPS = [80]
N_TRIALS = 10
TRIAL_LEN = 100
K_FORGET = 3
ONLINE_FINETUNE = False  # disable for clean comparison
FINETUNE_LR = 1e-4
FINETUNE_EVERY = 5
BUFFER_SIZE = 200

# CEM params
H_PLAN, N_CEM_ITERS, N_CANDIDATES = 10, 3, 128
ELITE_FRAC, K_MODELS, GAMMA = 0.1, 1, 0.99

# ── Oracle dynamics ─────────────────────────────────────────────────────────────
class PendulumSim:
    def __init__(self):
        self.g = 10.0; self.max_speed = 8.0; self.dt = 0.05
        self.m = 1.0; self.l = 1.0

    def step(self, state, torque):
        th, thdot = state
        torque = float(np.clip(torque, -2.0, 2.0))
        newthdot = thdot + (3*self.g/(2*self.l)*np.sin(th)
                             + 3.0/(self.m*self.l**2)*torque)*self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot*self.dt
        newth = ((newth+np.pi)%(2*np.pi))-np.pi
        cost = float(self._ang(th)**2 + 0.1*thdot**2 + 0.001*torque**2)
        return np.array([newth, newthdot]), -cost

    @staticmethod
    def _ang(x):
        return ((x+np.pi)%(2*np.pi))-np.pi

    def observe(self, s):
        th, thdot = s
        return np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        s = rng.uniform(low=-np.pi, high=np.pi, size=2)
        s[1] *= 0.5
        return s

    def set_mass(self, m):
        self.m = m


def oracle_cem_act(sim, state, rng):
    H = H_PLAN; J = N_CANDIDATES
    n_elite = max(1, int(ELITE_FRAC*J))
    mu = np.zeros(H, dtype=np.float32)
    sigma = np.full(H, 1.0, dtype=np.float32)
    for _ in range(N_CEM_ITERS):
        cand = np.clip(mu+sigma*rng.normal(size=(J,H)).astype(np.float32), -2, 2)
        scores = np.array([_rollout_oracle(sim, state.copy(), cand[j]) for j in range(J)])
        elite = cand[np.argpartition(-scores, n_elite-1)[:n_elite]]
        mu = elite.mean(axis=0); sigma = np.maximum(elite.std(axis=0), 0.05)
    return mu[0]


def _rollout_oracle(sim, s, seq):
    tot, disc = 0.0, 1.0
    for t in range(len(seq)):
        ns, r = sim.step(s.copy(), seq[t])
        tot += disc*r; s = ns; disc *= GAMMA
    return tot


def _rollout_bnn(agent, obs0, action_seq):
    """Rollout using BNN dynamics."""
    H = len(action_seq)
    obs_t = torch.as_tensor(obs0, dtype=torch.float32, device=device).unsqueeze(0)
    state = agent.dyn.reset(obs_t)
    total = 0.0; disc = 1.0
    for t in range(H):
        act_t = torch.tensor([action_seq[t]], dtype=torch.float32, device=device)
        next_obs, rew, _, _ = agent.dyn.sample(act_t, state, deterministic=True)
        total += disc * float(rew.item())
        obs_t = next_obs; disc *= GAMMA
        state = agent.dyn.reset(obs_t)
    return total


def run_oracle_cem():
    """Run oracle CEM baseline."""
    sim = PendulumSim(); rng = np.random.default_rng(0)
    returns = []
    for trial in range(N_TRIALS):
        state = sim.reset(seed=trial); sim.set_mass(1.0)
        total = 0.0
        for step in range(TRIAL_LEN):
            if step in CHANGE_STEPS: sim.set_mass(3.0)
            torque = oracle_cem_act(sim, state.copy(), rng)
            state, rew = sim.step(state, torque)
            total += rew
        returns.append(total)
    return returns


def run_bnn_cem_static():
    """Run BNN-CEM without adaptation (static model)."""
    eval_env = build_pendulum_env(MASS_SCHEDULE)
    bnn, dyn = make_gaussian_bnn(3, 1)
    dyn.load(str(PENDULUM_SAVE_DIR))
    bnn.num_weight_groups = 1
    init_state = copy.deepcopy(bnn.state_dict())

    agent = ContinuousCEMAgent(dyn, bnn, 3, 1, device=device,
                               horizon=H_PLAN, n_cem_iters=N_CEM_ITERS,
                               n_candidates=N_CANDIDATES, elite_frac=ELITE_FRAC,
                               k_models=K_MODELS, gamma=GAMMA)
    returns = []
    for trial in range(N_TRIALS):
        obs, _ = eval_env.reset()
        agent.reset(); bnn.load_state_dict(init_state)
        total = 0.0
        for step in range(TRIAL_LEN):
            if step in CHANGE_STEPS: agent.notify_change()
            action = agent.act(obs)
            obs, rew, term, trunc, _ = eval_env.step(action)
            total += float(rew)
            if term or trunc: break
        returns.append(total)
    return returns


def run_bnn_cem_adaptive():
    """Run BNN-CEM with surprise/forget adaptation."""
    eval_env = build_pendulum_env(MASS_SCHEDULE)
    bnn, dyn = make_gaussian_bnn(3, 1)
    dyn.load(str(PENDULUM_SAVE_DIR))
    bnn.num_weight_groups = 1
    init_state = copy.deepcopy(bnn.state_dict())

    agent = ContinuousCEMAgent(dyn, bnn, 3, 1, device=device,
                               horizon=H_PLAN, n_cem_iters=N_CEM_ITERS,
                               n_candidates=N_CANDIDATES, elite_frac=ELITE_FRAC,
                               k_models=K_MODELS, gamma=GAMMA)
    returns = []
    for trial in range(N_TRIALS):
        obs, _ = eval_env.reset()
        agent.reset(); bnn.load_state_dict(init_state)
        drift_v2 = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)
        total = 0.0; post = 0
        for step in range(TRIAL_LEN):
            if step in CHANGE_STEPS:
                drift_v2.reset(); agent.notify_change()
            agent.surprise_bar = drift_v2.delta_bar
            agent.n_since_change = post

            action = agent.act(obs)
            next_obs, rew, term, trunc, _ = eval_env.step(action)
            total += float(rew)

            # Surprise & forget
            act_arr = np.asarray(action, dtype=np.float32).ravel()
            vs = surprise_gaussian(dyn, bnn, obs, act_arr, next_obs, rew)
            raw_s = max(vs["nu2"], 1e-6)
            drift_v2.update(raw_s)

            post_change = step >= CHANGE_STEPS[0]
            if post_change:
                post += 1
                if post % K_FORGET == 0:
                    forget_gaussian(bnn, drift_v2, inflate_mode="additive")

            if ONLINE_FINETUNE and step % FINETUNE_EVERY == 0:
                pass  # disabled for now

            obs = next_obs
            if term or trunc: break
        returns.append(total)
    return returns


def main():
    print("=" * 60)
    print("Pendulum Experiment: 3-way comparison")
    print(f"  trials={N_TRIALS}, trial_len={TRIAL_LEN}")
    print(f"  mass_schedule={MASS_SCHEDULE}")
    print(f"  CEM: H={H_PLAN}, I={N_CEM_ITERS}, J={N_CANDIDATES}")
    print("=" * 60)

    results = {}

    print("\n--- 1/3: Oracle CEM (true dynamics) ---")
    t0 = time.time()
    oracle_returns = run_oracle_cem()
    dt = time.time() - t0
    results["Oracle CEM"] = oracle_returns
    print(f"  avg={np.mean(oracle_returns):.1f} std={np.std(oracle_returns):.1f} "
          f"min={np.min(oracle_returns):.1f} max={np.max(oracle_returns):.1f} "
          f"time={dt:.1f}s")

    print("\n--- 2/3: BNN-CEM (static, no adaptation) ---")
    t0 = time.time()
    static_returns = run_bnn_cem_static()
    dt = time.time() - t0
    results["BNN-CEM (static)"] = static_returns
    print(f"  avg={np.mean(static_returns):.1f} std={np.std(static_returns):.1f} "
          f"min={np.min(static_returns):.1f} max={np.max(static_returns):.1f} "
          f"time={dt:.1f}s")

    print("\n--- 3/3: BNN-CEM (adaptive, surprise+forget) ---")
    t0 = time.time()
    adapt_returns = run_bnn_cem_adaptive()
    dt = time.time() - t0
    results["BNN-CEM (adaptive)"] = adapt_returns
    print(f"  avg={np.mean(adapt_returns):.1f} std={np.std(adapt_returns):.1f} "
          f"min={np.min(adapt_returns):.1f} max={np.max(adapt_returns):.1f} "
          f"time={dt:.1f}s")

    # Write results
    with open(RESULTS_FILE, "w") as f:
        f.write("# Pendulum Experiment Results\n\n")
        f.write(f"**Config**: trials={N_TRIALS}, trial_len={TRIAL_LEN}, ")
        f.write(f"mass_schedule={MASS_SCHEDULE}\n\n")
        f.write(f"**CEM params**: H={H_PLAN}, I={N_CEM_ITERS}, "
                f"J={N_CANDIDATES}, elite_frac={ELITE_FRAC}, "
                f"K={K_MODELS}, gamma={GAMMA}\n\n")
        f.write("| Method | Avg Return | Std | Min | Max |\n")
        f.write("|--------|-----------|-----|-----|-----|\n")
        for name, rets in results.items():
            f.write(f"| {name} | {np.mean(rets):.1f} | {np.std(rets):.1f} "
                    f"| {np.min(rets):.1f} | {np.max(rets):.1f} |\n")

    print(f"\nResults saved to {RESULTS_FILE}")
    print("DONE.")


if __name__ == "__main__":
    main()
