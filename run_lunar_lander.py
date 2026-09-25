"""Our method on Lunar Lander: MCTS + BNN dynamics + surprise/forget.

Non-stationarity: wind changes at t=0.
Compares against reference PA-MCTS performance from the paper.

Run:  python run_lunar_lander.py
"""

import copy, math, pathlib, sys, time
import numpy as np
import torch

from config import device, ETA, GAMMA_UNCERTAINTY
from env.lunar_lander import build_lunar_lander_env
from drift import DualDriftFilter
from bnn import make_gaussian_bnn, surprise_gaussian, forget_gaussian, mean_sigma

_HERE = pathlib.Path(__file__).parent
LOG = str(_HERE / "lunar_lander_experiments.log")
SAVE_DIR = _HERE / "data" / "lunar_lander"

# ── Config ─────────────────────────────────────────────────────────────────────
WIND_VALUES = [0.0, 10.0, 15.0, 20.0]  # from PA-MCTS paper
N_TRIALS = 10
TRIAL_LEN = 1000  # Lunar Lander episodes can be long
K_FORGET = 5

# ── MCTS params ────────────────────────────────────────────────────────────────
N_ACTIONS = 4
TORQUES = np.array([0, 1, 2, 3], dtype=np.int64)  # discrete Lunar Lander actions
MCTS_SIMS = 200
ROLLOUT_H = 10
CP = 50.0  # PA-MCTS paper value (sqrt(2) ≈ 1.4 is too small)
GAMMA = 0.99


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
        act_t = torch.tensor([[float(act)]], dtype=torch.float32, device=device)
        ns, rew, _, _ = dyn.sample(act_t, state, deterministic=True)
        total += disc * float(rew.item())
        obs_t = ns; state = dyn.reset(obs_t); disc *= GAMMA
    return total


def _uct(child, parent_visits):
    if child.visits == 0: return float("inf")
    return child.value/child.visits + CP*math.sqrt(math.log(max(1,parent_visits))/child.visits)


def mcts_act(dyn, obs):
    root = _Node(obs.copy())
    actions = list(range(N_ACTIONS))
    try:
        next_obs_all, rews_all = _batched_predict(dyn, obs, actions)
        for a in range(N_ACTIONS):
            child = _Node(next_obs_all[a], action=a, parent=root)
            child.visits = 1; child.value = float(rews_all[a])
            root.children.append(child)

        for _ in range(MCTS_SIMS):
            node = root
            while node.children:
                node = max(node.children, key=lambda c: _uct(c, node.visits))

            ns_all, rs_all = _batched_predict(dyn, node.state, actions)
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
    return int(np.argmax(visits))


def main():
    out = open(LOG, "w")
    def log(*a):
        msg = " ".join(str(x) for x in a)
        out.write(msg + "\n"); out.flush()
        print(msg, flush=True)

    bnn, dyn = make_gaussian_bnn(8, 1)
    dyn.load(str(SAVE_DIR))
    ckpt_path = SAVE_DIR / "bnn_dynamics.pth"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        bnn.load_state_dict(ckpt, strict=False)
        log(f"Loaded pretrained BNN from {SAVE_DIR}")
    else:
        log("WARNING: no pretrained BNN, using random weights")

    bnn.num_weight_groups = 1
    init_state = copy.deepcopy(bnn.state_dict())

    log("=" * 70)
    log(f"Lunar Lander: MCTS+BNN+forget | sims={MCTS_SIMS} rollout={ROLLOUT_H}")
    log(f"  wind={WIND_VALUES} | trials={N_TRIALS}x{TRIAL_LEN}")
    log("=" * 70)

    for wind in WIND_VALUES:
        wind_schedule = [(0, wind)]
        env = build_lunar_lander_env(wind_schedule=wind_schedule)

        returns = []
        t0 = time.time()
        for trial in range(N_TRIALS):
            obs, _ = env.reset()
            bnn.load_state_dict(init_state)
            drift = DualDriftFilter(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)
            total, post, forget_count, warmup = 0.0, 0, 0, 20

            for step in range(TRIAL_LEN):
                action = mcts_act(dyn, obs)
                next_obs, reward, term, trunc, _ = env.step(action)
                total += float(reward)

                act_arr = np.array([float(action)], dtype=np.float32)
                vs = surprise_gaussian(dyn, bnn, obs, act_arr, next_obs, reward)
                drift.update(vs["nu2"], vs["delta_n"])

                post += 1
                if post == warmup:
                    drift.reset()
                if post >= warmup and post % K_FORGET == 0:
                    f_info = forget_gaussian(bnn, drift, inflate_mode="additive", q_max=0.1)
                    if f_info["triggered"]:
                        forget_count += 1

                obs = next_obs
                if term or trunc:
                    break

            returns.append(total)
            log(f"  trial {trial+1}/{N_TRIALS}: return={total:.1f} steps={post} forget={forget_count}")

        log(f"wind={wind:5.0f}: {np.mean(returns):8.1f} ± {np.std(returns):.1f}  ({time.time()-t0:.0f}s)")
        env.close()

    log("\nDONE.")
    out.close()


if __name__ == "__main__":
    main()
