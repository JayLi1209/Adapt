"""BNNModelPlanner -- the BNN env-modeling plumbing shared by model-based planners.

This is the minimal slice of the original mcts_drift_copy_v2.MCTSAgent that the
CVaR-CEM planner actually inherits: it sets up the map-derived reward / terminal /
leaf-value arrays and reads K posterior transition matrices off the BNN.  The UCT
tree search (simulate / UCB / pessimism / backup) that CVaRCEMAgent overrides and
ignores has been dropped, per "keep only what is used".
"""
import numpy as np
import torch

import mbrl.planning as planning

from utils import make_fl_potential


class BNNModelPlanner(planning.Agent):
    """Reads a discrete FrozenLake MDP off a BNN dynamics model.

    State = cell index (0..n-1); actions = 0..n_actions-1.  `_model_matrices`
    returns K posterior transition rows T[s,a,:] (and reward-head predictions) read
    off the BNN -- deterministic=True gives the sharp mean model, deterministic=False
    gives K coherent Thompson draws (the epistemic axis the planner spreads over).

    __init__ also builds, from the map `desc`:
      terminal[s]    : absorbing (goal/hole) mask
      cell_value[s]  : value AT an absorbing cell (goal 1, hole hole_reward)
      cell_reward[s] : reward on arrival (goal +1, hole hole_reward, frozen 0)
      heuristic[s]   : gamma^dist(s, goal) leaf value V(s_H), terminals overridden
    """

    def __init__(self, dynamics_model, bnn, desc, device, n_actions=4,
                 gamma=0.97, hole_reward=None, rng=None, **kwargs):
        self.dyn = dynamics_model
        self.bnn = bnn
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.rng = rng if rng is not None else np.random.default_rng(0)
        # hole_reward=None -> the grid's own convention (bnn.grid.hole_reward:
        # 0.0 everywhere except cliffwalking_aayl's -1), so every planner sees
        # the reward the env actually pays -- planning/rats.py reads
        # grid.hole_reward the same way.
        if hole_reward is None:
            hole_reward = float(getattr(getattr(bnn, "grid", None),
                                        "hole_reward", 0.0))

        flat = [c.decode() for c in desc.flatten()]
        self.n = len(flat)
        # Absorbing cell values: goal = 1, hole = hole_reward (both terminal).
        self.terminal = np.zeros(self.n, dtype=bool)
        self.cell_value = np.zeros(self.n, dtype=np.float32)
        # Reward-on-arrival map: hole = hole_reward (default 0.0 -- a hole ends
        # the episode with no further reward), goal = +1, frozen = 0.
        self.cell_reward = np.zeros(self.n, dtype=np.float64)
        for i, ch in enumerate(flat):
            if ch in "GH":
                self.terminal[i] = True
            if ch == "G":
                self.cell_value[i] = 1.0
                self.cell_reward[i] = 1.0
            if ch == "H":
                self.cell_value[i] = hole_reward
                self.cell_reward[i] = hole_reward
        self._eye_s = np.eye(self.n, dtype=np.float32)
        self._eye_a = torch.eye(n_actions, dtype=torch.float32, device=device)

        # Leaf heuristic V_h(s) = gamma^dist(s, goal), dist = BFS over the
        # traversable grid (holes are walls -> dist = inf -> V_h = 0).
        phi = make_fl_potential(desc).cpu().numpy()      # -dist (holes very neg)
        dist = -phi
        self.heuristic = (gamma ** dist).astype(np.float64)
        self.heuristic[self.terminal] = self.cell_value[self.terminal]

    @torch.no_grad()
    def _model_matrices(self, k, deterministic):
        """k (transition, reward) models read off the BNN.

        Returns Ts (k, n, n_actions, n) and Rs (k, n, n_actions): for each model j,
        Ts[j, s, a, :] is the categorical next-state row and Rs[j, s, a] is the
        reward-head prediction r(s, a).

        deterministic=True -> the mean model (sharp; the default substrate).
        deterministic=False -> one coherent posterior weight sample per model
        (num_weight_groups forced to 1 so all cells share a hypothesis), repeated
        k times -> k Thompson models.
        """
        saved = self.bnn.num_weight_groups
        self.bnn.num_weight_groups = 1
        Ts = np.empty((k, self.n, self.n_actions, self.n), dtype=np.float64)
        Rs = np.empty((k, self.n, self.n_actions), dtype=np.float64)
        try:
            for j in range(k):
                for a in range(self.n_actions):
                    model_state = self.dyn.reset(self._eye_s)
                    act = self._eye_a[a].unsqueeze(0).expand(self.n, -1)
                    next_obs, rew, _, _ = self.dyn.sample(
                        act, model_state, deterministic=deterministic
                    )
                    probs = next_obs.clamp_min(0.0).cpu().numpy() + 1e-6
                    probs /= probs.sum(axis=-1, keepdims=True)
                    Ts[j, :, a, :] = probs
                    Rs[j, :, a] = rew.squeeze(-1).cpu().numpy()
        finally:
            self.bnn.num_weight_groups = saved
        return Ts, Rs

    def reset(self):
        pass

    def notify_change(self):
        pass
