"""Fair comparison: two pretrained BNNs vs Oracle vs ADA-MCTS.
Tests mass 1→4 at t=0. All methods use standard Gaussian BNN architecture.

Run:  python compare_models.py
"""

import copy, math, pathlib, sys, time
import numpy as np
import torch

from config import device, ETA, GAMMA_UNCERTAINTY
from env.pendulum import build_pendulum_env
from drift import DriftFilterV2
from bnn import make_gaussian_bnn, surprise_gaussian, forget_gaussian, mean_sigma
from planning.continuous_cem import ContinuousCEMAgent, H_PLAN, N_CEM_ITERS, N_CANDIDATES, K_MODELS, GAMMA
from oracle_cem_baseline import PendulumSim, cem_act

_HERE = pathlib.Path(__file__).parent
LOG = str(_HERE / "compare_models.log")
SAVE_DIR = _HERE / "data" / "pendulum"

MASS_SCHEDULE = [(0, 4.0)]
N_TRIALS = 20
TRIAL_LEN = 100
K_FORGET = 5

# ── ADA-MCTS (standard BNN, no latent) ────────────────────────────────────────
N_ACTIONS = 7
TORQUES = np.linspace(-2.0, 2.0, N_ACTIONS, dtype=np.float32)
MCTS_SIMS = 200
ROLLOUT_H = 8
CP = math.sqrt(2.0)


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
    if child.visits == 0: return float("inf")
    return child.value/child.visits + CP*math.sqrt(math.log(max(1,parent_visits))/child.visits)


def _mcts_act(dyn, obs):
    root = _Node(obs.copy())
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
        pass
    visits = np.zeros(N_ACTIONS)
    for child in root.children:
        visits[child.action] = child.visits
    return int(np.argmax(visits)), TORQUES[int(np.argmax(visits))]


def main():
    out = open(LOG, "w")
    def log(*a):
        print(*a, file=out); out.flush()
        print(*a)

    log("=" * 70)
    log(f"Model comparison: mass 1→4 at t=0, {N_TRIALS} trials x {TRIAL_LEN} steps")
    log("=" * 70)

    # ── Oracle ──────────────────────────────────────────────────────────
    sim = PendulumSim(); sim.set_mass(4.0); rng = np.random.default_rng(0)
    oracle_r = []
    t0 = time.time()
    for trial in range(N_TRIALS):
        s = sim.reset(seed=trial); total = 0.0
        for _ in range(TRIAL_LEN):
            s, rew = sim.step(s, cem_act(sim, s.copy(), rng)); total += rew
        oracle_r.append(total)
    log(f"Oracle CEM:          {np.mean(oracle_r):8.1f} ± {np.std(oracle_r):.1f}  ({time.time()-t0:.0f}s)")

    # ── Our method with each BNN ────────────────────────────────────────
    for model_name in ['bnn_dynamics.pth', 'bnn_dynamics_yuanhe.pth']:
        env = build_pendulum_env(mass_schedule=MASS_SCHEDULE)
        bnn, dyn = make_gaussian_bnn(3, 1)
        dyn.load(str(SAVE_DIR))
        ckpt = torch.load(str(SAVE_DIR / model_name), map_location=device)
        bnn.load_state_dict(ckpt, strict=False)
        bnn.num_weight_groups = 1
        init_state = copy.deepcopy(bnn.state_dict())

        agent = ContinuousCEMAgent(dyn, bnn, 3, 1, device=device,
                                   horizon=H_PLAN, n_cem_iters=N_CEM_ITERS,
                                   n_candidates=N_CANDIDATES,
                                   k_models=K_MODELS, gamma=GAMMA)
        returns = []
        t0 = time.time()
        for trial in range(N_TRIALS):
            obs, _ = env.reset(); agent.reset(); bnn.load_state_dict(init_state)
            drift = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)
            total, post = 0.0, 0
            for step in range(TRIAL_LEN):
                if step == 0: drift.reset(); agent.notify_change()
                action = agent.act(obs)
                next_obs, reward, term, trunc, _ = env.step(action)
                total += float(reward)
                act_arr = np.asarray(action, dtype=np.float32).ravel()
                vs = surprise_gaussian(dyn, bnn, obs, act_arr, next_obs, reward)
                drift.update(max(min(vs["nu2"], 50.0), 1e-6))
                post += 1
                if post % K_FORGET == 0:
                    forget_gaussian(bnn, drift, inflate_mode="additive", q_max=0.1)
                obs = next_obs
                if term or trunc: break
            returns.append(total)
        label = model_name.replace('.pth','').replace('bnn_dynamics_','')
        log(f"Our method ({label:>7s}): {np.mean(returns):8.1f} ± {np.std(returns):.1f}  ({time.time()-t0:.0f}s)")

    # ── ADA-MCTS with best BNN (yuanhe) ─────────────────────────────────
    env = build_pendulum_env(mass_schedule=MASS_SCHEDULE)
    bnn, dyn = make_gaussian_bnn(3, 1)
    dyn.load(str(SAVE_DIR))
    ckpt = torch.load(str(SAVE_DIR / "bnn_dynamics_yuanhe.pth"), map_location=device)
    bnn.load_state_dict(ckpt, strict=False)
    bnn.num_weight_groups = 1

    returns = []
    t0 = time.time()
    for trial in range(N_TRIALS):
        obs, _ = env.reset(); total = 0.0
        for step in range(TRIAL_LEN):
            a_idx, torque = _mcts_act(dyn, obs)
            act_arr = np.array([torque], dtype=np.float32)
            next_obs, reward, term, trunc, _ = env.step(act_arr)
            total += float(reward)
            obs = next_obs
            if term or trunc: break
        returns.append(total)
    log(f"ADA-MCTS (yuanhe):   {np.mean(returns):8.1f} ± {np.std(returns):.1f}  ({time.time()-t0:.0f}s)")

    log("\nDONE.")
    out.close()


if __name__ == "__main__":
    main()
