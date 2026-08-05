"""Risk-Averse Tree-Search (RATS) + DP baselines for non-stationary grids.

Faithful port of the official implementation (SuReLI/rats-experiments) of
Lecarpentier & Rachelson, "Non-Stationary Markov Decision Processes: a
Worst-Case Approach using Model-Based Reinforcement Learning" (NeurIPS 2019),
adapted to this repo's gridworld + Dirichlet-BNN infrastructure.

The agent plans a minimax tree: decision nodes (max) pick the action, chance
nodes (min) evaluate the WORST-CASE model inside the Lipschitz ball of radius
L_p * tau * depth around the current snapshot model (p_{t0}, R_{t0}):

    R_hat(s,a) = R_{t0}(s,a) - L_r * tau * depth
    p_hat      = argmin over W1(p, p_{t0}) <= L_p * tau * depth of E_p[V]

with the closed-form direct method (Property 3): p_hat = (1-lambda) p0 +
lambda * onehot(argmin V) with lambda = min(1, c / W1(onehot, p0)).

The tree is a DAG (the same (state, depth) pair recurs at every depth level),
so values are memoized per (s, d) -- EXACTLY equivalent to the official full
tree (the paper's Bellman equations for V*_{t0,t}(s)) but O(|S||A|d) instead of
exponential.  Depth 6 on the cliff (48 cells) is milliseconds per call.

Three snapshot models are provided:
  GridSnapshot        : ORACLE snapshot -- true grid dynamics at intended-prob p
                        (the paper's "agent knows MDP_{t0}" hypothesis).
  DynamicGridSnapshot : ORACLE time-varying model -- the scheduled p(t) at each
                        epoch (used by the DP-NSMDP omniscient baseline).
  BNNSnapshot         : LEARNED snapshot -- the Dirichlet BNN predictive mean
                        (counts/retain participate, so surprise/forget adapt it).

Agents:
  RATS    -- minimax with the worst-case min operator (Algorithm 1).
  DPAgent -- expectation tree (no min); is_model_dynamic=False is DP-snapshot,
             is_model_dynamic=True is DP-NSMDP (uses DynamicGridSnapshot).
"""

import collections
import itertools

import numpy as np
import torch
from scipy.optimize import linprog

from config import device

# ── 1-Wasserstein helpers (ported from the official code) ───────────────────────

def wass_dual(u, v, d):
    """1-Wasserstein distance between discrete u, v given distances matrix d.

    Dual LP: max f.(u-v) s.t. |f_i - f_j| <= d_ij.  Two-atom closed form (the
    common case: bridge K=2 support); scipy LP otherwise.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    n = d.shape[0]
    if n == 2:
        return abs(u[0] - v[0]) * d[0, 1]
    comb = np.array(list(itertools.combinations(range(n), 2)))
    obj = u - v
    Au = np.zeros(shape=(n * (n - 1), n))
    bu = np.zeros(shape=(n * (n - 1)))
    for i in range(len(comb)):
        Au[2 * i][comb[i][0]] = +1.0
        Au[2 * i][comb[i][1]] = -1.0
        Au[2 * i + 1][comb[i][0]] = -1.0
        Au[2 * i + 1][comb[i][1]] = +1.0
        bu[2 * i] = d[comb[i][0]][comb[i][1]]
        bu[2 * i + 1] = d[comb[i][0]][comb[i][1]]
    res = linprog(obj, A_ub=Au, b_ub=bu)
    return -res.fun


def worstcase_distribution_direct_method(v, w0, c, d):
    """min_w w.v  s.t.  W1(w, w0) <= c  (closed-form, Property 3 direct method).

    The worst target is a Dirac at the argmin child.  W1(dirac_k, w0) has the
    closed form sum_i w0_i d(i,k) (optimal transport pushes every unit of mass
    to k at cost d(i,k)), so no LP is needed.
    """
    v = np.asarray(v, dtype=float)
    w0 = np.asarray(w0, dtype=float)
    n = len(v)
    if c <= 1e-12 or np.allclose(v, v[0] * np.ones(n)):
        return w0.copy()
    k = int(np.argmin(v))
    w_worst = np.zeros(n)
    w_worst[k] = 1.0
    w1 = float(w0 @ d[:, k])           # W1(dirac_k, w0) = sum_i w0_i d(i,k)
    if w1 <= c:
        return w_worst
    lbd = c / w1
    return (1.0 - lbd) * w0 + lbd * w_worst


# ── grid geometry helpers ───────────────────────────────────────────────────────

def cell_reward_of(grid):
    """Reward-on-arrival per cell (goal +1, hole -1, frozen/start 0)."""
    return np.array([{"G": 1.0, "H": -1.0}.get(c, 0.0) for c in grid.flat_desc])


def bfs_dist_to_goal(grid):
    """BFS distance from each traversable cell to the nearest goal (holes walls)."""
    flat = grid.flat_desc
    goals = [i for i, c in enumerate(flat) if c == "G"]
    dist = {g: 0 for g in goals}
    q = collections.deque(goals)
    while q:
        s = q.popleft()
        for a in range(grid.n_actions):
            for s_prev in range(grid.n_states):
                if flat[s_prev] in "HG":
                    continue
                if grid.move(s_prev, a) == s and s_prev not in dist:
                    dist[s_prev] = dist[s] + 1
                    q.append(s_prev)
    far = grid.n_states * 2
    return np.array([dist.get(i, far) for i in range(grid.n_states)], dtype=np.float64)


def _scatter(grid, slip):
    """slip (K,) -> (n_states, n_actions, n_states) transition rows."""
    cells = grid.build_dir_cells().numpy()          # (nS, nA, K)
    P = np.zeros((grid.n_states, grid.n_actions, grid.n_states), dtype=np.float64)
    for s in range(grid.n_states):
        for a in range(grid.n_actions):
            for k in range(grid.k_dir):
                P[s, a, cells[s, a, k]] += slip[k]
    return P


def _dist_matrix(grid, states):
    """Manhattan distance submatrix over a list of cells."""
    n = len(states)
    D = np.zeros((n, n))
    for i in range(n):
        ri, ci = divmod(int(states[i]), grid.ncol)
        for j in range(i + 1, n):
            rj, cj = divmod(int(states[j]), grid.ncol)
            di = abs(ri - rj) + abs(ci - cj)
            D[i, j] = di
            D[j, i] = di
    return D


# ── snapshot models ─────────────────────────────────────────────────────────────

class GridSnapshot:
    """ORACLE snapshot: the true grid dynamics frozen at one intended-prob p.

    The paper's "the agent knows the current model MDP_{t0} but not its future
    evolution" hypothesis.  p(s'|s,a) = slip_dist(p) scattered onto cells.
    """

    def __init__(self, grid, p, L_p=1.0, L_r=0.0, tau=1.0):
        self.grid = grid
        self.p = float(p)
        self.L_p = L_p
        self.L_r = L_r
        self.tau = tau
        self.n_states = grid.n_states
        self.n_actions = grid.n_actions
        self.P = _scatter(grid, np.array(grid.slip_dist(self.p), dtype=np.float64))
        self.cell_reward = cell_reward_of(grid)
        self.terminal = np.array([c in "GH" for c in grid.flat_desc])

    def p_cells(self, s, a, t=None):
        return self.P[int(s), int(a)]

    def expected_reward(self, s, a, t=None):
        return float(self.P[int(s), int(a)] @ self.cell_reward)

    def is_terminal(self, s):
        return bool(self.terminal[int(s)])

    def reward(self, s):
        return float(self.cell_reward[int(s)])


class DynamicGridSnapshot:
    """ORACLE time-varying model: the true scheduled p(t) (DP-NSMDP baseline).

    dist_by_time maps timestep -> the K-vector active from that step on (same
    semantics as TabularGridEnv).  p(s'|s,a,t) uses the distribution active at
    decision epoch t.
    """

    def __init__(self, grid, dist_by_time, L_p=1.0, L_r=0.0, tau=1.0):
        self.grid = grid
        self.dist_by_time = {int(t): np.asarray(d, dtype=np.float64)
                             for t, d in dist_by_time.items()}
        self.times = sorted(self.dist_by_time)
        self.L_p = L_p
        self.L_r = L_r
        self.tau = tau
        self.n_states = grid.n_states
        self.n_actions = grid.n_actions
        self.P_by_t = {t: _scatter(grid, self.dist_by_time[t]) for t in self.times}
        self.cell_reward = cell_reward_of(grid)
        self.terminal = np.array([c in "GH" for c in grid.flat_desc])

    def _active(self, t):
        t = int(t)
        active = self.P_by_t[self.times[0]]
        for tt in self.times:
            if tt <= t:
                active = self.P_by_t[tt]
            else:
                break
        return active

    def p_cells(self, s, a, t=None):
        return self._active(t)[int(s), int(a)]

    def expected_reward(self, s, a, t=None):
        return float(self._active(t)[int(s), int(a)] @ self.cell_reward)

    def is_terminal(self, s):
        return bool(self.terminal[int(s)])

    def reward(self, s):
        return float(self.cell_reward[int(s)])


class BNNSnapshot:
    """LEARNED snapshot: the Dirichlet BNN predictive mean as the model.

    The deterministic (sample=False) readout goes through _forward_alpha, so the
    online counts and the retain/forget factor participate: surprise/forget/learn
    adaptation shows up in the model RATS plans with.  Cell rewards come from the
    ground-truth map (the repo convention -- planners never use the reward head).
    """

    def __init__(self, dyn, bnn, grid, L_p=1.0, L_r=0.0, tau=1.0):
        self.dyn = dyn
        self.bnn = bnn
        self.grid = grid
        self.L_p = L_p
        self.L_r = L_r
        self.tau = tau
        self.n_states = grid.n_states
        self.n_actions = grid.n_actions
        self.cell_reward = cell_reward_of(grid)
        self.terminal = np.array([c in "GH" for c in grid.flat_desc])
        self._eye_a = torch.eye(self.n_actions, dtype=torch.float32, device=device)
        self._cache = {}

    @torch.no_grad()
    def p_cells(self, s, a, t=None):
        s, a = int(s), int(a)
        key = (s, a)
        if key in self._cache:
            return self._cache[key]
        obs = np.zeros(self.n_states, dtype=np.float32)
        obs[s] = 1.0
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        act_t = self._eye_a[a].unsqueeze(0)
        state = self.dyn.reset(obs_t)
        next_obs, _, _, _ = self.dyn.sample(act_t, state, deterministic=True)
        p = next_obs[0, :self.n_states].clamp_min(0.0).cpu().numpy()
        p /= p.sum()
        self._cache[key] = p
        return p

    def expected_reward(self, s, a, t=None):
        return float(self.p_cells(s, a) @ self.cell_reward)

    def is_terminal(self, s):
        return bool(self.terminal[int(s)])

    def reward(self, s):
        return float(self.cell_reward[int(s)])

    def clear_cache(self):
        self._cache = {}


# ── RATS agent (worst-case minimax, memoized over (state, depth)) ───────────────

class RATS:
    """Risk-Averse Tree-Search: minimax where Nature picks the worst-case model
    inside the LC ball around the snapshot (Algorithm 1 of the paper)."""

    def __init__(self, model, gamma=0.99, max_depth=6, heuristic="potential",
                 grid=None):
        self.model = model
        self.gamma = gamma
        self.max_depth = max_depth
        grid = grid if grid is not None else model.grid
        dist = bfs_dist_to_goal(grid)
        if heuristic == "zero":
            self._heur = lambda s, d: 0.0
        elif heuristic == "paper":
            # Property 6: H(s,t) = V^pi_{MDP_t0}(s) - (t-t0) L_R / (1-gamma),
            # with V^pi the greedy-toward-goal snapshot value gamma^dist(s).
            L_R = model.L_p + model.L_r
            self._heur = lambda s, d: gamma ** dist[s] - d * L_R / (1.0 - gamma)
        elif heuristic == "potential":
            self._heur = lambda s, d: gamma ** dist[s]
        else:
            raise ValueError(f"unknown heuristic {heuristic!r}")
        self._memo = {}

    def _V(self, s, d):
        """Max-node (decision) value at state s, depth d."""
        key = (int(s), d)
        if key in self._memo:
            return self._memo[key]
        model = self.model
        if model.is_terminal(s):
            v = model.reward(s)
        elif d == self.max_depth:
            v = self._heur(s, d)
        else:
            v = max(self._Q(s, a, d) for a in range(model.n_actions))
        self._memo[key] = v
        return v

    def _Q(self, s, a, d):
        """Min-node (chance) value for (s,a) at depth d: worst-case model."""
        model = self.model
        p_cells = model.p_cells(s, a)
        children = [int(i) for i in range(model.n_states) if p_cells[i] > 1e-12]
        v_child = np.array([self._V(s2, d + 1) for s2 in children])
        w0 = np.array([p_cells[s2] for s2 in children], dtype=np.float64)
        w0 /= w0.sum()
        c = d * model.L_p * model.tau
        d_mat = _dist_matrix(model.grid, children)
        w = worstcase_distribution_direct_method(v_child, w0, c, d_mat)
        v = self.gamma * float(w @ v_child) + model.expected_reward(
            s, a) - model.L_r * model.tau * d
        return v

    def act(self, s, t0=0):
        """Plan from cell s at decision epoch t0; return the best action."""
        self._memo = {}
        return max(range(self.model.n_actions),
                   key=lambda a: self._Q(int(s), a, 0))


# ── DP agents (expectation, no min) ─────────────────────────────────────────────

class DPAgent:
    """Expectation tree-search on a model (the paper's DP-snapshot / DP-NSMDP).

    is_model_dynamic=False -> the snapshot model at t0 (DP-snapshot).
    is_model_dynamic=True  -> the time-varying model indexed by node.time, i.e.
                              the omniscient DP-NSMDP upper bound (requires a
                              DynamicGridSnapshot).
    """

    def __init__(self, model, gamma=0.99, max_depth=6, is_model_dynamic=False,
                 heuristic="potential", grid=None):
        self.model = model
        self.gamma = gamma
        self.max_depth = max_depth
        self.is_model_dynamic = is_model_dynamic
        grid = grid if grid is not None else model.grid
        dist = bfs_dist_to_goal(grid)
        if heuristic == "zero":
            self._heur = lambda s, d: 0.0
        elif heuristic == "potential":
            self._heur = lambda s, d: gamma ** dist[s]
        else:
            raise ValueError(f"unknown heuristic {heuristic!r}")
        self._memo = {}

    def _V(self, s, t, d):
        key = (int(s), t, d)
        if key in self._memo:
            return self._memo[key]
        model = self.model
        if model.is_terminal(s):
            v = model.reward(s)
        elif d == self.max_depth:
            v = self._heur(s, d)
        else:
            v = max(self._Q(s, a, t, d) for a in range(model.n_actions))
        self._memo[key] = v
        return v

    def _Q(self, s, a, t, d):
        model = self.model
        t_next = t + model.tau if self.is_model_dynamic else t
        p_cells = model.p_cells(s, a, t)
        children = [int(i) for i in range(model.n_states) if p_cells[i] > 1e-12]
        v = sum(float(p_cells[s2]) * self._V(s2, t_next, d + 1)
                for s2 in children)
        return self.gamma * v + model.expected_reward(s, a, t)

    def act(self, s, t0=0):
        """Plan from cell s at decision epoch t0; return the best action."""
        self._memo = {}
        return max(range(self.model.n_actions),
                   key=lambda a: self._Q(int(s), a, t0, 0))
