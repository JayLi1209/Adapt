"""ADA-MCTS: Adaptive Monte Carlo Tree Search for non-stationary MDPs.

Implements Algorithm 2 from "Act As You Learn" (Luo et al., 2024).
Uses MCTS with dual-phase adaptive sampling (DPAS) at chance nodes,
switching between worst-case (risk-averse) and regular sampling based on
epistemic and aleatoric uncertainty comparisons between M_k and M_{k-1}.

Reuses the Dirichlet BNN for transition estimation and uncertainty
quantification — the same model our CVaR-CEM planner uses.
"""

import copy
import math

import numpy as np
import torch

import mbrl.models as models

from planning.base import BNNModelPlanner
from bnn.dirichlet_workflow import epistemic_dirichlet
from bnn.dirichlet_model import ALPHA_FLOOR, SURPRISE_EPS

# ── ADA-MCTS hyperparameters ──────────────────────────────────────────────────
M_SIMULATIONS = 3000      # MCTS simulations (rollouts) per action (paper: 30000)
CP = math.sqrt(2.0)       # UCT exploration constant
EPS_E = 0.02              # epistemic uncertainty threshold (paper line 236)
EPS_A = 0.0               # aleatoric uncertainty threshold
DPAS_GAMMA = 10000.0      # upstream adamcts.py `gamma` in the aleatoric likelihood
N_POSTERIOR = 10          # posterior draws for Var_E / Var_A
H_ROLLOUT = 6             # rollout horizon (match our planner)


class _Node:
    """Tree node for MCTS.  Decision node → child chance nodes (one per action);
    chance node → one child decision node (whose state is determined by DPAS)."""

    __slots__ = ("state", "action", "parent", "children", "visits", "value",
                 "node_type")

    def __init__(self, state, action=None, parent=None, node_type="decision"):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.node_type = node_type  # "decision" | "chance"


class ADAMCTSAgent(BNNModelPlanner):
    """ADA-MCTS agent for FrozenLake non-stationary MDP.

    On `notify_change()` snapshots the current BNN as M_{k-1} (frozen) and
    starts exploring the new MDP M_k via risk-averse MCTS, gradually switching
    to regular (reward-maximising) sampling as epistemic uncertainty drops.
    """

    def __init__(self, dynamics_model, bnn, desc, device, n_actions=4,
                 gamma=0.95, rng=None,
                 m_simulations=M_SIMULATIONS,
                 cp=CP,
                 eps_e=EPS_E,
                 eps_a=EPS_A,
                 dpas_gamma=DPAS_GAMMA,
                 h_rollout=H_ROLLOUT,
                 **kwargs):
        super().__init__(dynamics_model, bnn, desc, device,
                         n_actions=n_actions, gamma=gamma, rng=rng, **kwargs)
        # Grid geometry (generalises the FrozenLake 4x4 assumptions).
        self.nrow, self.ncol = int(desc.shape[0]), int(desc.shape[1])
        self.m_simulations = m_simulations
        self.cp = cp
        self.eps_e = eps_e
        self.eps_a = eps_a
        self.dpas_gamma = dpas_gamma
        self.h_rollout = h_rollout

        # M_{k-1} frozen snapshot (created at change notification)
        self.bnn_prev = None
        self.dyn_prev = None

        # Per-act prediction cache: (s, a) -> dict with epistemic, aleatoric, p_cells
        self._cache = {}

        # Average aleatoric uncertainty trackers for both models
        self._au_k = []   # M_k aleatoric history (entropy values)
        self._au_prev = []  # M_{k-1} aleatoric history

        # Post-change tracking
        self._post_change_steps = 0
        self._training_started = True   # starts True (pre-change, trust model)
        self._n_threshold = 3  # min post-change samples before switching mode
        self._last_dpas_mode = "reg"  # track DPAS decisions for logging
        self._wc_count = 0            # worst-case samples drawn this act()
        self._reg_count = 0           # regular samples drawn this act()

    # ── snapshot management ──────────────────────────────────────────────────
    def notify_change(self):
        """Freeze current BNN as M_{k-1}, reset M_k exploration state."""
        if self.bnn_prev is None:
            # first notification in this phase: snapshot the OLD model (M_{k-1}
            # = the pretrained model before any post-change evidence arrives)
            self.bnn_prev = copy.deepcopy(self.bnn)
        self.dyn_prev = models.OneDTransitionRewardModel(
            self.bnn_prev,
            target_is_delta=False,
            normalize=False,
            learned_rewards=True,
        )
        self._cache = {}
        self._au_k = []
        self._au_prev = []
        self._post_change_steps = 0
        self._training_started = False

    def reset(self):
        self._cache = {}

    def learn(self, s, a, s2):
        """Update online counts after observing (s,a) → s2."""
        d = self.bnn.grid.direction_of(s, a, s2)
        self.bnn.add_count(s, a, d)
        self._post_change_steps += 1
        if (not self._training_started and
                self._post_change_steps >= self._n_threshold):
            self._training_started = True

    # ── uncertainty query (cached per (s,a) within one act()) ────────────────
    def _query(self, s, a):
        """Return cached or freshly-computed BNN predictions for (s,a)."""
        key = (int(s), int(a))
        if key in self._cache:
            return self._cache[key]

        obs = np.zeros(self.n, dtype=np.float32)
        obs[s] = 1.0
        act = np.zeros(self.n_actions, dtype=np.float32)
        act[a] = 1.0

        # M_k (current model — may have online counts / forget updates)
        res_k = epistemic_dirichlet(self.dyn, self.bnn, obs, act,
                                     n_draws=N_POSTERIOR)
        epistemic_k = res_k["epistemic"]
        aleatoric_k = res_k["entropy"]
        p_cells_k = res_k["p_cells"]

        # Track aleatoric history for DPAS
        self._au_k.append(aleatoric_k)

        # M_{k-1} (frozen snapshot, if available)
        if self.bnn_prev is not None:
            res_prev = epistemic_dirichlet(self.dyn_prev, self.bnn_prev,
                                            obs, act, n_draws=N_POSTERIOR)
            epistemic_prev = res_prev["epistemic"]
            aleatoric_prev = res_prev["entropy"]
            p_cells_prev = res_prev["p_cells"]
            self._au_prev.append(aleatoric_prev)
        else:
            epistemic_prev = epistemic_k
            aleatoric_prev = aleatoric_k
            p_cells_prev = p_cells_k

        entry = {
            "epistemic_k": epistemic_k,
            "epistemic_prev": epistemic_prev,
            "aleatoric_k": aleatoric_k,
            "aleatoric_prev": aleatoric_prev,
            "p_cells_k": p_cells_k,
            "p_cells_prev": p_cells_prev,
            "alpha0_k": res_k["alpha0"],
            "alpha0_prev": res_prev["alpha0"] if self.bnn_prev is not None else res_k["alpha0"],
        }
        self._cache[key] = entry
        return entry

    def _var_a_bar(self, history):
        """Average aleatoric uncertainty (entropy) over queried pairs."""
        if not history:
            return 0.0
        return float(np.mean(history[-min(100, len(history)):]))

    # ── DPAS (dual-phase adaptive sampling) ──────────────────────────────────
    def _dpas(self, s, a, child_values):
        """Dual-Phase Adaptive Sampling -- the ADA-MCTS decision rule.

        Ported verbatim from the submodule's ADA-MCTS/adamcts.py
        `Node.expand()` (chance-node branch).  Upstream names model 1 / model 2;
        here model 1 = M_k (the CURRENT bnn, which the forget/counts loop is
        adapting) and model 2 = M_{k-1} (the snapshot frozen at notify_change).

            if (epi_1 + threshold < epi_2) or (not training_started):
                s' ~ pessimistic(P_1)                       # worst-case, M_k
            elif ale_1 < ale_2:
                lik = exp(-gamma * (ale_2 - ale_1))
                if (ale_2 - ale_1) < 1 and U(0,1) < lik:
                    s' ~ P_2                                # regular,    M_{k-1}
                else:
                    s' ~ pessimistic(P_2)                   # worst-case, M_{k-1}
            else:
                s' ~ P_2                                    # regular,    M_{k-1}

        `pessimistic(.)` is Node.pessimistic_sample with danger=False (the
        setting act_learn.py runs): one-hot the worst-reward reachable cell if
        any reachable cell has negative reward, else pass the distribution
        through unchanged.  See _worst_case_sample.
        """
        q = self._query(s, a)
        epi_k, epi_prev = q["epistemic_k"], q["epistemic_prev"]
        ale_k, ale_prev = q["aleatoric_k"], q["aleatoric_prev"]
        p_k, p_prev = q["p_cells_k"], q["p_cells_prev"]

        # Phase 1: uninformed about the new MDP -> act worst-case under M_k.
        if (epi_k + self.eps_e < epi_prev) or (not self._training_started):
            self._last_dpas_mode = "wc_k"
            self._wc_count += 1
            return self._worst_case_sample(s, a, child_values, p_k)

        # Phase 2: gated by the aleatoric comparison against M_{k-1}.
        if ale_k + self.eps_a < ale_prev:
            diff = ale_prev - ale_k
            likelihood = math.exp(-self.dpas_gamma * diff)
            if diff < 1.0 and float(self.rng.random()) < likelihood:
                self._last_dpas_mode = "reg_prev"
                self._reg_count += 1
                return self._categorical_sample(p_prev)
            self._last_dpas_mode = "wc_prev"
            self._wc_count += 1
            return self._worst_case_sample(s, a, child_values, p_prev)

        self._last_dpas_mode = "reg_prev"
        self._reg_count += 1
        return self._categorical_sample(p_prev)

    def _categorical_sample(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-9, None)
        probs /= probs.sum()
        return int(self.rng.choice(len(probs), p=probs))

    def _worst_case_sample(self, s, a, child_values, p_cells):
        """Paper's pessimistic_sample: if any reachable state has negative reward,
        one-hot the worst one; otherwise return the original distribution.

        Matches adamcts.py Node.pessimistic_sample() logic:
        1. Compute direct reward for each reachable state
        2. If all >= 0 → return p_cells (no pessimism, safe)
        3. Else → one-hot at argmin reward

        Reachable cells come from the grid's K directions (grid-agnostic; the
        cliff is read as a hole -- no teleport-to-start, the env's terminal_cliff
        already ended the episode).
        """
        grid = self.bnn.grid
        r, c = divmod(int(s), self.ncol)
        reachable = set()
        for d in grid.dir_actions(int(a)):
            dr, dc = grid.deltas[int(d)]
            nr = min(max(r + dr, 0), self.nrow - 1)
            nc = min(max(c + dc, 0), self.ncol - 1)
            reachable.add(nr * self.ncol + nc)

        # Immediate reward for each reachable state
        rewards = {s2: float(self.cell_reward[s2]) for s2 in reachable}
        if all(r >= 0 for r in rewards.values()):
            # No negative outcome possible → no need for pessimism
            return self._categorical_sample(p_cells)
        # One-hot the worst (most negative) reachable state
        worst = min(rewards, key=rewards.get)
        return worst

    # ── rollout ──────────────────────────────────────────────────────────────
    def _rollout(self, s0):
        """Uniform random rollout from s0 using M_k model, for H_ROLLOUT steps."""
        total = 0.0
        disc = 1.0
        s = s0
        for _ in range(self.h_rollout):
            if self.terminal[s]:
                break
            a = self.rng.integers(0, self.n_actions)
            q = self._query(s, a)
            s2 = self._categorical_sample(q["p_cells_k"])
            total += disc * float(self.cell_reward[s2])
            s = s2
            disc *= self.gamma
        if not self.terminal[s]:
            total += disc * float(self.heuristic[s])
        return total

    # ── backpropagate ────────────────────────────────────────────────────────
    def _backprop(self, leaf, delta):
        """Climb from leaf to root, adding discounted delta."""
        node = leaf
        while node is not None:
            node.visits += 1
            node.value += delta
            node = node.parent
            # Discount when climbing through a decision node
            # (in their code, discount is applied at decision nodes)
            if node is not None and node.node_type == "decision":
                delta *= self.gamma

    # ── tree policy ──────────────────────────────────────────────────────────
    def _uct_select(self, node):
        """Pick child with highest UCT score."""
        best_score = -np.inf
        best_child = None
        log_n = math.log(max(1, node.visits))
        for child in node.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = child.value / child.visits
                explore = self.cp * math.sqrt(log_n / child.visits)
                score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    # ── traverse ─────────────────────────────────────────────────────────────
    def _traverse(self, root):
        """Walk down the tree using UCT at decision nodes, DPAS at chance nodes.
        Returns the leaf node reached."""
        node = root
        depth = 0
        while True:
            if self.terminal[node.state]:
                return node
            if node.node_type == "decision":
                # If not fully expanded, expand all actions
                if not node.children:
                    # Expand: create one chance child per action
                    for a in range(self.n_actions):
                        child = _Node(node.state, action=a, parent=node,
                                       node_type="chance")
                        node.children.append(child)
                    # Return a random newly-expanded child
                    return self.rng.choice(node.children)
                else:
                    node = self._uct_select(node)
            else:  # chance node
                # DPAS determines the next state
                s = node.state
                a = node.action

                # Collect known child values
                child_values = {}
                for child in node.children:
                    child_values[child.state] = (
                        child.value / child.visits if child.visits > 0
                        else self.heuristic[child.state]
                    )
                s2 = self._dpas(s, a, child_values)

                # Find or create child decision node at s2
                existing = None
                for child in node.children:
                    if child.state == s2:
                        existing = child
                        break
                if existing is not None:
                    node = existing
                else:
                    child = _Node(s2, parent=node, node_type="decision")
                    node.children.append(child)
                    node = child
            depth += 1
            if depth > 100:  # safety
                return node

    # ── act (main entry point) ───────────────────────────────────────────────
    @torch.no_grad()
    def act(self, obs, **kwargs):
        s0 = int(np.argmax(obs))
        if self.terminal[s0]:
            return self._eye_a[0].cpu().numpy()

        self._cache = {}
        self._wc_count = 0
        self._reg_count = 0
        root = _Node(s0, node_type="decision")

        for _ in range(self.m_simulations):
            leaf = self._traverse(root)
            if self.terminal[leaf.state]:
                delta = float(self.cell_value[leaf.state])
            else:
                delta = self._rollout(leaf.state)
            self._backprop(leaf, delta)

        # Action selection: pick most-visited child of root
        visits = np.zeros(self.n_actions, dtype=np.float64)
        for child in root.children:
            if child.action is not None:
                visits[child.action] = child.visits
        # Fallback: if no visits (shouldn't happen), use heuristic
        if visits.sum() == 0:
            visits[self.rng.integers(0, self.n_actions)] = 1.0
        a = int(np.argmax(visits))

        # Diagnostics for logging
        self.last_visits = visits
        self.last_cache_size = len(self._cache)
        self.last_training = self._training_started
        self.last_post_steps = self._post_change_steps
        a0_samples = [self._cache[k]["alpha0_k"] for k in self._cache]
        self.last_alpha0 = float(np.mean(a0_samples)) if a0_samples else 0.0

        return self._eye_a[a].cpu().numpy()
