"""Collect pendulum transitions using oracle CEM for high-quality BNN training data.
Saves to data/pendulum_oracle/ for training an improved dynamics model.
"""
import pathlib, time
import numpy as np
from oracle_cem_baseline import PendulumSim, cem_act as _cem_act, H, I, J, GAMMA

_HERE = pathlib.Path(__file__).parent
OUT_DIR = _HERE / "data" / "pendulum_oracle"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_EPISODES = 10
EPISODE_LEN = 100

import numpy as np

def _to_obs(th, thdot):
    return np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)

def collect():
    sim = PendulumSim()
    rng = np.random.default_rng(0)
    all_data = []

    for ep in range(N_EPISODES):
        s = sim.reset(seed=ep)
        sim.set_mass(1.0)
        for step in range(EPISODE_LEN):
            if step == 80:
                sim.set_mass(3.0)
            obs = _to_obs(s[0], s[1])
            torque = _cem_act(sim, s.copy(), rng)
            ns, rew = sim.step(s, torque)
            next_obs = _to_obs(ns[0], ns[1])
            all_data.append((obs, np.array([torque], dtype=np.float32),
                             next_obs, np.float32(rew)))
            s = ns

        print(f"  episode {ep+1}/{N_EPISODES} done ({len(all_data)} transitions)")

    np.savez(OUT_DIR / "transitions.npz",
             obs=np.stack([d[0] for d in all_data]),
             act=np.stack([d[1] for d in all_data]),
             next_obs=np.stack([d[2] for d in all_data]),
             reward=np.array([d[3] for d in all_data]))
    print(f"Saved {len(all_data)} transitions to {OUT_DIR}")
    return all_data

if __name__ == "__main__":
    t0 = time.time()
    collect()
    print(f"Done in {time.time()-t0:.1f}s")
