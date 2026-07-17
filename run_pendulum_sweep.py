"""Pendulum parameter sweep: find feasible ranges for mass and gravity.

Tests our method (CEM + surprise/forget) vs Oracle across parameter ranges.
Change at t=0. Uses pretrained BNN from data/pendulum/bnn_dynamics_yuanhe.pth.

Run:  python run_pendulum_sweep.py
"""

import copy, pathlib, time
import numpy as np
import torch

from config import device, ETA, GAMMA_UNCERTAINTY
from env.pendulum import build_pendulum_env
from drift import DualDriftFilter
from bnn import make_gaussian_bnn, surprise_gaussian, forget_gaussian
from planning.continuous_cem import ContinuousCEMAgent, H_PLAN, N_CEM_ITERS, N_CANDIDATES, K_MODELS, GAMMA
from oracle_cem_baseline import PendulumSim, cem_act

_HERE = pathlib.Path(__file__).parent
LOG = str(_HERE / "pendulum_sweep.log")
BNN_PATH = _HERE / "data" / "pendulum" / "bnn_dynamics_yuanhe.pth"
SAVE_DIR = _HERE / "data" / "pendulum"

N_TRIALS = 10
TRIAL_LEN = 100
K_FORGET = 5

# Sweep ranges
MASS_VALUES = [1.0, 2.0, 2.3, 2.5, 3.0, 3.5, 4.0]
GRAVITY_VALUES = [10.0, 12.0, 15.0, 18.0, 20.0]


def run_oracle(mass=None, gravity=None, n_trials=N_TRIALS):
    sim = PendulumSim(); rng = np.random.default_rng(0)
    if mass is not None: sim.set_mass(mass)
    if gravity is not None: sim.set_gravity(gravity)
    returns = []
    for trial in range(n_trials):
        s = sim.reset(seed=trial); total = 0.0
        for _ in range(TRIAL_LEN):
            s, rew = sim.step(s, cem_act(sim, s.copy(), rng)); total += rew
        returns.append(total)
    return np.mean(returns), np.std(returns)


def run_ours(mass=None, gravity=None, bnn=None, dyn=None, n_trials=N_TRIALS):
    kw = {}
    if mass is not None: kw["mass_schedule"] = [(0, mass)]
    if gravity is not None: kw["gravity_schedule"] = [(0, gravity)]
    env = build_pendulum_env(**kw)
    bnn.num_weight_groups = 1
    init_state = copy.deepcopy(bnn.state_dict())
    agent = ContinuousCEMAgent(dyn, bnn, 3, 1, device=device,
                               horizon=H_PLAN, n_cem_iters=N_CEM_ITERS,
                               n_candidates=N_CANDIDATES,
                               k_models=K_MODELS, gamma=GAMMA)
    returns = []
    for trial in range(n_trials):
        obs, _ = env.reset(); agent.reset(); bnn.load_state_dict(init_state)
        drift = DualDriftFilter(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)
        total, post, warmup = 0.0, 0, 20
        for step in range(TRIAL_LEN):
            if step == 0: agent.notify_change()
            if step == warmup: drift.reset()  # freeze calibrated baselines
            action = agent.act(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            total += float(reward)
            act_arr = np.asarray(action, dtype=np.float32).ravel()
            vs = surprise_gaussian(dyn, bnn, obs, act_arr, next_obs, reward)
            drift.update(vs["nu2"], vs["delta_n"])
            post += 1
            if post % K_FORGET == 0:
                f_info = forget_gaussian(bnn, drift, inflate_mode="additive", q_max=0.1)
            obs = next_obs
            if term or trunc: break
        returns.append(total)
    env.close()
    return np.mean(returns), np.std(returns)


def main():
    out = open(LOG, "w")
    def log(*a):
        print(*a, file=out); out.flush()
        print(*a)

    bnn, dyn = make_gaussian_bnn(3, 1)
    dyn.load(str(SAVE_DIR))
    ckpt = torch.load(str(BNN_PATH), map_location=device)
    bnn.load_state_dict(ckpt, strict=False)

    log("=" * 70)
    log(f"Pendulum parameter sweep (change at t=0, {N_TRIALS} trials x {TRIAL_LEN} steps)")
    log(f"  CEM: H={H_PLAN} I={N_CEM_ITERS} J={N_CANDIDATES} K={K_MODELS}")
    log("=" * 70)

    # ── Mass sweep ────────────────────────────────────────────────────────
    log("\n--- Mass sweep (gravity=10.0) ---")
    log(f"{'Mass':>6s}  {'Oracle':>10s}  {'Ours':>10s}  {'Ratio':>6s}")
    log("-" * 40)
    for m in MASS_VALUES:
        t0 = time.time()
        avg_o, std_o = run_oracle(mass=m)
        avg_u, std_u = run_ours(mass=m, bnn=bnn, dyn=dyn)
        ratio = avg_u / avg_o if avg_o != 0 else float('inf')
        log(f"{m:6.1f}  {avg_o:8.1f}±{std_o:5.0f}  {avg_u:8.1f}±{std_u:5.0f}  {ratio:5.2f}x  ({time.time()-t0:.0f}s)")

    # ── Gravity sweep ─────────────────────────────────────────────────────
    log("\n--- Gravity sweep (mass=1.0) ---")
    log(f"{'Grav':>6s}  {'Oracle':>10s}  {'Ours':>10s}  {'Ratio':>6s}")
    log("-" * 40)
    for g in GRAVITY_VALUES:
        t0 = time.time()
        avg_o, std_o = run_oracle(gravity=g)
        avg_u, std_u = run_ours(gravity=g, bnn=bnn, dyn=dyn)
        ratio = avg_u / avg_o if avg_o != 0 else float('inf')
        log(f"{g:6.1f}  {avg_o:8.1f}±{std_o:5.0f}  {avg_u:8.1f}±{std_u:5.0f}  {ratio:5.2f}x  ({time.time()-t0:.0f}s)")

    log("\nDONE.")
    out.close()


if __name__ == "__main__":
    main()
