"""Oracle CEM: CEM planner with TRUE pendulum dynamics (no BNN).

Gold-standard baseline — upper bound on what CEM can achieve.
"""

import math, pathlib, time
import numpy as np

_HERE = pathlib.Path(__file__).parent
LOG = str(_HERE / "run_pendulum_cem.log")

MASS_SCHEDULE = [(0, 1.0), (80, 3.0)]
CHANGE_STEPS = [80]
N_TRIALS = 20
TRIAL_LEN = 150

H, I, J, elite_frac = 12, 5, 256, 0.1
GAMMA = 0.99


class PendulumSim:
    def __init__(self):
        self.g, self.max_speed, self.dt = 10.0, 8.0, 0.05
        self.m, self.l = 1.0, 1.0

    def step(self, s, u):
        th, thdot = s
        u = float(np.clip(u, -2, 2))
        nd = thdot + (1.5*self.g/self.l*np.sin(th) + 3.0/(self.m*self.l**2)*u)*self.dt
        nd = np.clip(nd, -self.max_speed, self.max_speed)
        nth = th + nd*self.dt
        nth = ((nth+np.pi)%(2*np.pi))-np.pi
        c = float(((th+np.pi)%(2*np.pi)-np.pi)**2 + 0.1*thdot**2 + 0.001*u**2)
        return np.array([nth, nd]), -c

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        s = rng.uniform(-np.pi, np.pi, 2); s[1] *= 0.5
        return s

    def set_mass(self, m): self.m = m


def cem_act(sim, s, rng):
    n_elite = max(1, int(elite_frac*J))
    mu = np.zeros(H, dtype=np.float32)
    sigma = np.full(H, 1.0, dtype=np.float32)
    for _ in range(I):
        cand = np.clip(mu+sigma*rng.normal(size=(J,H)).astype(np.float32), -2, 2)
        scores = np.zeros(J)
        for j in range(J):
            ss, tot, disc = s.copy(), 0.0, 1.0
            for t in range(H):
                ss, r = sim.step(ss, cand[j,t])
                tot += disc*r; disc *= GAMMA
            scores[j] = tot
        elite = cand[np.argpartition(-scores, n_elite-1)[:n_elite]]
        mu = elite.mean(axis=0); sigma = np.maximum(elite.std(axis=0), 0.05)
    return mu[0]


def main():
    with open(LOG, "w") as out:
        def log(*a):
            print(*a, file=out); out.flush()
            print(*a)
        sim = PendulumSim(); rng = np.random.default_rng(0)
        log(f"Oracle CEM | H={H} I={I} J={J} | {N_TRIALS} trials")
        returns = []
        for trial in range(N_TRIALS):
            s = sim.reset(seed=trial); sim.set_mass(1.0); total = 0.0
            for step in range(TRIAL_LEN):
                if step in CHANGE_STEPS: sim.set_mass(3.0)
                s, rew = sim.step(s, cem_act(sim, s.copy(), rng))
                total += rew
            returns.append(total)
            log(f"TRIAL {trial+1}: {total:.1f}")
        log(f"\navg={np.mean(returns):.1f} std={np.std(returns):.1f}")
        log("DONE.")


if __name__ == "__main__":
    main()
