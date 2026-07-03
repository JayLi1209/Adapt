"""CEM with oracle (true) dynamics as the gold-standard baseline.

Compares against: (1) Our BNN + CEM + surprise/forget method,
(2) MCTS with oracle dynamics.
"""

import math
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).parent
LOG = str(_HERE / "run_pendulum_cem.log")

# ── Config ──────────────────────────────────────────────────────────────────────
MASS_SCHEDULE = [(0, 1.0), (80, 3.0)]
CHANGE_STEPS = [80]
N_TRIALS = 20
TRIAL_LEN = 150

# CEM params (match our planner for fair comparison)
H_PLAN = 10
N_CEM_ITERS = 3
N_CANDIDATES = 128
ELITE_FRAC = 0.1
GAMMA = 0.99
SIGMA_INIT = 1.0
SIGMA_MIN = 0.05


class PendulumSim:
    def __init__(self):
        self.g = 10.0
        self.max_speed = 8.0
        self.dt = 0.05
        self.m = 1.0
        self.l = 1.0

    def step(self, state, torque):
        th, thdot = state
        torque = float(np.clip(torque, -2.0, 2.0))
        newthdot = thdot + (3 * self.g / (2 * self.l) * np.sin(th)
                             + 3.0 / (self.m * self.l ** 2) * torque) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        newth = ((newth + np.pi) % (2 * np.pi)) - np.pi
        cost = float(self._angle_normalize(th) ** 2 + 0.1 * (thdot ** 2) + 0.001 * (torque ** 2))
        return np.array([newth, newthdot]), -cost

    @staticmethod
    def _angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def observe(self, state):
        th, thdot = state
        return np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        state = rng.uniform(low=-np.pi, high=np.pi, size=2)
        state[1] *= 0.5  # limit initial velocity
        return state

    def set_mass(self, m):
        self.m = m


def rollout_return(sim, state, action_seq):
    """Discounted return of an action sequence under oracle dynamics."""
    total = 0.0
    disc = 1.0
    for t in range(len(action_seq)):
        torque = action_seq[t]
        state, rew = sim.step(state, torque)
        total += disc * rew
        disc *= GAMMA
    return total


def cvar_value(returns, alpha=1.0):
    m = len(returns)
    k = max(1, int(math.ceil(alpha * m)))
    return float(np.sort(returns)[:k].mean())


def cem_act(sim, state, rng):
    """Continuous-action CEM with oracle dynamics."""
    H = H_PLAN
    mu = np.zeros(H, dtype=np.float32)
    sigma = np.full(H, SIGMA_INIT, dtype=np.float32)
    J = N_CANDIDATES
    n_elite = max(1, int(ELITE_FRAC * J))

    for _ in range(N_CEM_ITERS):
        candidates = mu + sigma * rng.normal(size=(J, H)).astype(np.float32)
        candidates = np.clip(candidates, -2.0, 2.0)
        scores = np.array([rollout_return(sim, state.copy(), candidates[j]) for j in range(J)])
        elite_idx = np.argpartition(-scores, n_elite - 1)[:n_elite]
        elite_seq = candidates[elite_idx]
        mu = elite_seq.mean(axis=0)
        sigma = np.maximum(elite_seq.std(axis=0), SIGMA_MIN)

    return mu[0]


def main():
    out = open(LOG, "w")
    def log(*a):
        print(*a, file=out); out.flush()
        print(*a)

    sim = PendulumSim()
    rng = np.random.default_rng(0)

    log("=" * 80)
    log(f"CEM (oracle dynamics) baseline | mass_schedule={MASS_SCHEDULE} | trials={N_TRIALS}")
    log(f"  H={H_PLAN}, I={N_CEM_ITERS}, J={N_CANDIDATES}, elite_frac={ELITE_FRAC}")
    log("=" * 80)

    returns_hist = []

    for trial in range(N_TRIALS):
        state = sim.reset(seed=trial)
        sim.set_mass(1.0)
        total_return = 0.0

        for step in range(TRIAL_LEN):
            if step in CHANGE_STEPS:
                sim.set_mass(3.0)
                log(f"  >>> ENV CHANGE (mass -> 3.0) @ t={step}")

            torque = cem_act(sim, state.copy(), rng)
            state, rew = sim.step(state, torque)
            total_return += rew

        returns_hist.append(total_return)
        log(f"TRIAL {trial+1}: return={total_return:.1f}")

    log(f"\navg return={np.mean(returns_hist):.1f} +/- {np.std(returns_hist):.1f}")
    log(f"min={np.min(returns_hist):.1f} max={np.max(returns_hist):.1f}")
    log("DONE.")
    out.close()


if __name__ == "__main__":
    main()
