"""ADA-MCTS baseline for Pendulum using the same PyTorch Gaussian BNN as our method.

GPU-batched: queries all N actions at a node in a single BNN forward pass.
Uses the same learned dynamics model for fair comparison with our CEM method.
"""

import copy, math, time, pathlib
import numpy as np
import torch

from config import device
from env.pendulum import build_pendulum_env
from bnn.gaussian_model import make_gaussian_bnn
from bnn.gaussian_workflow import surprise_gaussian

_HERE = pathlib.Path(__file__).parent
LOG = str(_HERE / "ada_mcts_pendulum.log")
BNN_DIR = _HERE / "data" / "pendulum_oracle"

MASS_SCHEDULE = [(0, 1.0), (80, 3.0)]
CHANGE_STEPS = [80]
N_TRIALS = 20
TRIAL_LEN = 100

N_ACTIONS = 5                     # discretized torque bins
TORQUES = np.linspace(-2.0, 2.0, N_ACTIONS, dtype=np.float32)
MCTS_SIMS = 80                    # MCTS iterations per step
ROLLOUT_H = 4                     # rollout horizon
CP = math.sqrt(2.0)               # UCT exploration
GAMMA = 0.99


class _Node:
    __slots__ = ("state", "action", "parent", "children", "visits", "value")
    def __init__(self, state, action=None, parent=None):
        self.state = state          # np array (obs_dim,)
        self.action = action        # int action index
        self.parent = parent
        self.children = []          # list of _Node
        self.visits = 0
        self.value = 0.0


@torch.no_grad()
def batched_query(dyn, bnn, obs_arr, actions):
    """Query BNN for ALL actions at once: (N_ACTIONS, ...) -> returns (N_ACTIONS,) of next_obs + reward.

    obs_arr: (obs_dim,) flat array.
    Returns: next_obs (N_ACTIONS, obs_dim), rewards (N_ACTIONS,), states (list of N_ACTIONS state dicts).
    """
    K = len(actions)
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
    obs_t = obs_t.unsqueeze(0).expand(K, -1)                         # (K, obs_dim)
    act_t = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(-1)  # (K, 1)
    state = dyn.reset(obs_t)
    next_obs, rew, _, _ = dyn.sample(act_t, state, deterministic=True)
    return next_obs.cpu().numpy(), rew.squeeze(-1).cpu().numpy()


@torch.no_grad()
def batched_rollout(dyn, bnn, obs_arr, horizon=ROLLOUT_H):
    """Random rollout using BNN. Returns total discounted return."""
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
    state = dyn.reset(obs_t)
    total = 0.0; disc = 1.0
    for _ in range(horizon):
        act = np.random.randint(0, N_ACTIONS)
        act_t = torch.tensor([[TORQUES[act]]], dtype=torch.float32, device=device)
        next_obs, rew, _, _ = dyn.sample(act_t, state, deterministic=True)
        total += disc * float(rew.item())
        obs_t = next_obs; state = dyn.reset(obs_t); disc *= GAMMA
    return total


def uct(child, parent_visits):
    if child.visits == 0: return float("inf")
    return child.value/child.visits + CP * math.sqrt(math.log(max(1, parent_visits))/child.visits)


def mcts_act(dyn, bnn, obs):
    """One step of MCTS planning with BNN dynamics. GPU-batched action queries."""
    root = _Node(obs.copy())
    # Pre-expand: query all actions in one GPU batch
    saved = bnn.num_weight_groups
    bnn.num_weight_groups = 1
    try:
        next_obs_all, rews_all = batched_query(dyn, bnn, obs, TORQUES)
        for a in range(N_ACTIONS):
            child = _Node(next_obs_all[a], action=a, parent=root)
            child.visits = 1
            child.value = float(rews_all[a])
            root.children.append(child)

        for _ in range(MCTS_SIMS):
            # Selection
            node = root
            while node.children:
                node = max(node.children, key=lambda c: uct(c, node.visits))
            # Expansion
            next_obs_all2, rews_all2 = batched_query(dyn, bnn, node.state, TORQUES)
            for a in range(N_ACTIONS):
                child = _Node(next_obs_all2[a], action=a, parent=node)
                node.children.append(child)
            # Rollout from first new child
            delta = batched_rollout(dyn, bnn, node.children[0].state)
            # Backprop
            c = node.children[0]
            while c is not None:
                c.visits += 1; c.value += delta
                c = c.parent
                if c is not None: delta *= GAMMA
    finally:
        bnn.num_weight_groups = saved

    visits = np.zeros(N_ACTIONS)
    for child in root.children:
        visits[child.action] = child.visits
    return int(np.argmax(visits)), TORQUES[int(np.argmax(visits))]


def main():
    out = open(LOG, "w")
    def log(*a):
        print(*a, file=out); out.flush()
        print(*a)

    env = build_pendulum_env(MASS_SCHEDULE)
    if (BNN_DIR / "bnn_dynamics.pth").exists():
        bnn, dyn = make_gaussian_bnn(3, 1)
        dyn.load(str(BNN_DIR))
        log(f"Loaded BNN from {BNN_DIR}")
    else:
        log("ERROR: need BNN trained on oracle data first")
        return

    bnn.num_weight_groups = 1
    init_state = copy.deepcopy(bnn.state_dict())

    log("=" * 60)
    log(f"ADA-MCTS on Pendulum | actions={N_ACTIONS} | sims={MCTS_SIMS}")
    log(f"  rollout_h={ROLLOUT_H} | gamma={GAMMA} | {N_TRIALS} trials")
    log("=" * 60)

    returns = []
    for trial in range(N_TRIALS):
        obs, _ = env.reset()
        bnn.load_state_dict(init_state)
        total = 0.0
        t0 = time.time()
        for step in range(TRIAL_LEN):
            if step in CHANGE_STEPS:
                pass  # static model for baseline
            a_idx, torque = mcts_act(dyn, bnn, obs)
            obs, rew, term, trunc, _ = env.step(np.array([torque], dtype=np.float32))
            total += float(rew)
            if term or trunc:
                break
        returns.append(total)
        log(f"TRIAL {trial+1}: {total:.1f}  ({time.time()-t0:.1f}s)")

    log(f"\navg={np.mean(returns):.1f} std={np.std(returns):.1f}")
    out.close()


if __name__ == "__main__":
    main()
