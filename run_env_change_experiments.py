"""Test BNN adaptation to environment changes at t=0.

Two types of non-stationarity:
  1. Mass: 1.0 → 4.0  at t=0
  2. Gravity: 10 → 20 at t=0

Compares Oracle CEM, our method (CEM+surprise/forget), and ADA-MCTS.
Uses pretrained standard Gaussian BNN from data/pendulum/bnn_dynamics_yuanhe.pth.

Run:  python run_env_change_experiments.py
"""

import copy, math, pathlib, sys, time
import numpy as np
import torch
from torch import optim

from config import device, ETA, GAMMA_UNCERTAINTY
from env.pendulum import build_pendulum_env
from drift import DriftFilterV2
from bnn import make_gaussian_bnn, surprise_gaussian, forget_gaussian, mean_sigma
from planning.continuous_cem import ContinuousCEMAgent, H_PLAN, N_CEM_ITERS, N_CANDIDATES, K_MODELS, GAMMA

_HERE = pathlib.Path(__file__).parent
LOG = str(_HERE / "env_change_experiments.log")
BNN_PATH = _HERE / "data" / "pendulum" / "bnn_dynamics_yuanhe.pth"
NORMALIZER_DIR = _HERE / "data" / "pendulum"

# ── Experiments ────────────────────────────────────────────────────────────────
EXPERIMENTS = {
    "mass_1_to_4":     {"mass_schedule": [(0, 4.0)]},
    "gravity_10_to_20": {"gravity_schedule": [(0, 20.0)]},
}

N_TRIALS = 10
TRIAL_LEN = 100
K_FORGET = 5

# ── Oracle CEM ─────────────────────────────────────────────────────────────────
from oracle_cem_baseline import PendulumSim, cem_act


def run_oracle(exp_name, config, rng):
    """Oracle CEM with true pendulum dynamics."""
    sim = PendulumSim()
    # Apply t=0 changes
    if "mass_schedule" in config:
        sim.set_mass(config["mass_schedule"][0][1])
    if "gravity_schedule" in config:
        sim.set_gravity(config["gravity_schedule"][0][1])

    returns = []
    for trial in range(N_TRIALS):
        s = sim.reset(seed=trial)
        total = 0.0
        for step in range(TRIAL_LEN):
            torque = cem_act(sim, s.copy(), rng)
            s, rew = sim.step(s, torque)
            total += rew
        returns.append(total)
    return np.mean(returns), np.std(returns)


# ── Our method: CEM + surprise/forget (standard BNN) ────────────────────────────
@torch.no_grad()
def run_ours(exp_name, config, bnn, dyn):
    env = build_pendulum_env(**config)
    bnn.num_weight_groups = 1
    init_state = copy.deepcopy(bnn.state_dict())

    agent = ContinuousCEMAgent(dyn, bnn, 3, 1, device=device,
                               horizon=H_PLAN, n_cem_iters=N_CEM_ITERS,
                               n_candidates=N_CANDIDATES,
                               k_models=K_MODELS, gamma=GAMMA)

    returns = []
    for trial in range(N_TRIALS):
        obs, _ = env.reset()
        agent.reset()
        bnn.load_state_dict(init_state)
        drift = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)
        total, post = 0.0, 0

        for step in range(TRIAL_LEN):
            if step == 0:
                drift.reset()
                agent.notify_change()

            action = agent.act(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            total += float(reward)

            act_arr = np.asarray(action, dtype=np.float32).ravel()
            vs = surprise_gaussian(dyn, bnn, obs, act_arr, next_obs, reward)
            raw_s = min(vs["nu2"], 50.0)
            drift.update(max(raw_s, 1e-6))

            post += 1
            if post % K_FORGET == 0:
                forget_gaussian(bnn, drift, inflate_mode="additive", q_max=0.1)

            obs = next_obs
            if term or trunc:
                break
        returns.append(total)
    return np.mean(returns), np.std(returns)


# ── ADA-MCTS (adapted for standard BNN, no latent) ──────────────────────────────
N_ACTIONS = 7
TORQUES = np.linspace(-2.0, 2.0, N_ACTIONS, dtype=np.float32)
MCTS_SIMS = 200
ROLLOUT_H = 15
CP = 50.0
EPS_E = 0.01


class _Node:
    __slots__ = ("state", "action", "parent", "children", "visits", "value")
    def __init__(self, state, action=None, parent=None):
        self.state = state; self.action = action; self.parent = parent
        self.children = []; self.visits = 0; self.value = 0.0


@torch.no_grad()
def _batched_predict(dyn, obs_arr, actions):
    K = len(actions)
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
    obs_t = obs_t.unsqueeze(0).expand(K, -1)
    act_t = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(-1)
    state = dyn.reset(obs_t)
    next_obs, rew, _, _ = dyn.sample(act_t, state, deterministic=True)
    return next_obs.cpu().numpy(), rew.squeeze(-1).cpu().numpy()


@torch.no_grad()
def _batched_rollout(dyn, obs_arr):
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
    state = dyn.reset(obs_t)
    total, disc = 0.0, 1.0
    for _ in range(ROLLOUT_H):
        act = np.random.randint(0, N_ACTIONS)
        act_t = torch.tensor([[TORQUES[act]]], dtype=torch.float32, device=device)
        ns, rew, _, _ = dyn.sample(act_t, state, deterministic=True)
        total += disc * float(rew.item())
        obs_t = ns; state = dyn.reset(obs_t); disc *= GAMMA
    return total


def _uct(child, parent_visits):
    if child.visits == 0:
        return float("inf")
    return child.value/child.visits + CP*math.sqrt(math.log(max(1,parent_visits))/child.visits)


def _mcts_act(dyn, bnn_k, bnn_prev, obs):
    root = _Node(obs.copy())
    saved = bnn_k.num_weight_groups
    bnn_k.num_weight_groups = 1
    if bnn_prev is not None:
        bnn_prev.num_weight_groups = 1
    try:
        next_obs_all, rews_all = _batched_predict(dyn, obs, TORQUES)
        for a in range(N_ACTIONS):
            child = _Node(next_obs_all[a], action=a, parent=root)
            child.visits = 1; child.value = float(rews_all[a])
            root.children.append(child)

        for _ in range(MCTS_SIMS):
            node = root
            while node.children:
                node = max(node.children, key=lambda c: _uct(c, node.visits))

            ns_all, rs_all = _batched_predict(dyn, node.state, TORQUES)
            for a in range(N_ACTIONS):
                node.children.append(_Node(ns_all[a], action=a, parent=node))

            delta = _batched_rollout(dyn, node.children[0].state)
            c = node.children[0]
            while c is not None:
                c.visits += 1; c.value += delta
                c = c.parent
                if c is not None: delta *= GAMMA
    finally:
        bnn_k.num_weight_groups = saved
        if bnn_prev is not None:
            bnn_prev.num_weight_groups = saved

    visits = np.zeros(N_ACTIONS)
    for child in root.children:
        visits[child.action] = child.visits
    return int(np.argmax(visits)), TORQUES[int(np.argmax(visits))]


def run_ada_mcts(exp_name, config, bnn, dyn):
    env = build_pendulum_env(**config)
    bnn.num_weight_groups = 1
    init_state = copy.deepcopy(bnn.state_dict())

    returns = []
    for trial in range(N_TRIALS):
        obs, _ = env.reset()
        bnn.load_state_dict(init_state)
        bnn_prev = None  # no frozen snapshot for standard BNN (no change point to freeze at)
        total = 0.0

        for step in range(TRIAL_LEN):
            bnn_prev_snapshot = copy.deepcopy(bnn) if step == 0 and bnn_prev is None else bnn_prev
            # At t=0: create snapshot, then use regular MCTS (no DPAS without latent)
            a_idx, torque = _mcts_act(dyn, bnn, bnn_prev_snapshot, obs)
            act_arr = np.array([torque], dtype=np.float32)
            next_obs, reward, term, trunc, _ = env.step(act_arr)
            total += float(reward)
            obs = next_obs
            if term or trunc:
                break
        returns.append(total)
    return np.mean(returns), np.std(returns)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    out = open(LOG, "w")
    def log(*a):
        print(*a, file=out); out.flush()
        print(*a)

    # Load pretrained standard BNN
    bnn, dyn = make_gaussian_bnn(3, 1)
    dyn.load(str(NORMALIZER_DIR))
    ckpt = torch.load(str(BNN_PATH), map_location=device)
    bnn.load_state_dict(ckpt, strict=False)
    log(f"Loaded pretrained BNN from {BNN_PATH}")

    rng = np.random.default_rng(0)

    log("=" * 70)
    log(f"Environment change experiments (change at t=0)")
    log(f"  trials={N_TRIALS}  steps={TRIAL_LEN}")
    log(f"  CEM: H={H_PLAN} I={N_CEM_ITERS} J={N_CANDIDATES} K={K_MODELS}")
    log(f"  MCTS: sims={MCTS_SIMS} actions={N_ACTIONS} rollout={ROLLOUT_H}")
    log("=" * 70)

    for exp_name, config in EXPERIMENTS.items():
        label = f"mass={config.get('mass_schedule',[(0,1)])[0][1]}" if "mass_schedule" in config else \
                f"g={config.get('gravity_schedule',[(0,10)])[0][1]}"
        log(f"\n--- {exp_name} ({label}) ---")

        # Oracle
        t0 = time.time()
        avg_o, std_o = run_oracle(exp_name, config, rng)
        log(f"  Oracle CEM:        {avg_o:8.1f} ± {std_o:.1f}  ({time.time()-t0:.0f}s)")

        # Our method
        t0 = time.time()
        avg_u, std_u = run_ours(exp_name, config, bnn, dyn)
        log(f"  Our method (CEM):   {avg_u:8.1f} ± {std_u:.1f}  ({time.time()-t0:.0f}s)")

        # ADA-MCTS
        t0 = time.time()
        avg_a, std_a = run_ada_mcts(exp_name, config, bnn, dyn)
        log(f"  ADA-MCTS:           {avg_a:8.1f} ± {std_a:.1f}  ({time.time()-t0:.0f}s)")

    log("\nDONE.")
    out.close()


if __name__ == "__main__":
    main()
