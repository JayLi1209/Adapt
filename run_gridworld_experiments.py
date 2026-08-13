"""Gridworld RATS experiments: stationary verification + non-stationary change.

Mirrors the ADA-MCTS paper's (Luo et al. 2024) Cliff-Walking / NS-Bridge tables,
plus our adaptive variant.  The model is pretrained on the ORIGINAL env (p=0.7);
"introducing the new environment" changes p to {0.4, 0.5, 0.6, 0.8, 0.9, 1.0}.

Methods (column alignment with the ADA-MCTS paper, Luo et al. 2024, Table 1/2):
  DP-NSMDP (oracle)     omniscient -- plans with the true schedule
  DP-snapshot (oracle)  re-plans each step with the true CURRENT model
  RATS-P_k (oracle)     RATS with the true CURRENT model
  RATS-P_{k-1} (oracle) RATS with the true OLD model (p=0.7, never updated)
  RATS-P_hat_{k-1}      RATS with the pretrained BNN, no adaptation
  FIR-RATS (ours)       RATS + surprise/forget/online-counts on the same BNN
  ADA-MCTS              the paper's method (DPAS, notified of the change)
  MCTS-P_hat_{k-1}      plain MCTS with the pretrained BNN, no notification

gamma = 0.99 (per user request; CLAUDE.md's no-discount default is overridden).
Per-step rewards are logged; both the raw discounted return (per-step penalty
incl.) and the goal rate (holes 0, the paper's convention) are reported.
RATS depth defaults to 3 (the paper's documented value); DP to 100 (exact).

Run:
    python run_gridworld_experiments.py --grid cliffwalking --trials 30
    python run_gridworld_experiments.py --grid bridge --trials 100
"""

import argparse
import multiprocessing
import pathlib

import numpy as np
import torch

from config import device, ETA, GAMMA_UNCERTAINTY
from grids import get_grid
from bnn import make_dirichlet_bnn, surprise_dirichlet
from bnn.dirichlet_model import ckpt_name
from drift import DriftFilterV2
from planning.rats import (
    GridSnapshot, DynamicGridSnapshot, BNNSnapshot, RATS, DPAgent,
    _dist_matrix, worstcase_distribution_direct_method,
)
from planning.ada_mcts import ADAMCTSAgent, DPAS_GAMMA
from planning.cvar_cem import CVaRCEMAgent

_HERE = pathlib.Path(__file__).parent

GAMMA = 0.99            # discount (user request)
ORIG_P = 1.0            # "original" env the model is pretrained on: fully
                        # deterministic, so the optimal policy is known and the
                        # pretrained model can be verified to find it
                        # (2026-08-12 setting; was 0.7)
CHANGE_PS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # degrade p from 1.0 downward
K_FORGET = 3            # forget every K post-change steps (was 5; 3 beats
                        # static at all degraded p on cliff, see 08-06 log)
M_SIMULATIONS = 30000   # ADA-MCTS baseline simulations (rollouts) per action,
                        # per the paper.  7.3s/action (cliff) serial; the
                        # parallel runner (--workers, 16-core machine) makes the
                        # full sweep feasible: ~6-8h for cliff, ~1.5h for bridge.
N_POSTERIOR = 10        # BNN posterior draws for surprise
RATS_DEPTH = 3          # RATS depth (paper's documented value)
DP_DEPTH = 100          # DP depth (>= horizon -> exact)
# FIR-CEM (main method) planner settings
CEM_CVAR_ALPHA = 1.0    # CVaR tail fraction when not adaptive (1.0 = risk-neutral)
CEM_ADAPTIVE_ALPHA = True  # confidence-gated alpha (FIR surprise gates the tail)
CEM_ALPHA_MIN = 0.10    # most risk-averse CVaR tail (the 10% worst returns); the
                        # planner default 0.95 is too mild to change behavior
CEM_N_CONFIDENT = 20    # post-change samples for the data gate to saturate; with
                        # ~40-60 steps/episode this leaves the cautious phase long
                        # enough to matter (planner default 12 saturates too fast)
# unbounded-RATS baselines (scheme 1 worst-of-K / scheme 2 calibrated L_p)
N_MODEL_DRAWS = 100     # posterior model draws
RATS_CVAR_ALPHA = 0.01  # scheme 1: ~1% empirical tail over model draws


def cell_reward(grid, s):
    c = grid.flat_desc[int(s)]
    if c == "G":
        return 1.0
    if c == "H":
        return -1.0
    return grid.step_penalty


def active_p_fn(p_schedule):
    """Return active_p(t) for a (timestep, p) schedule (last p with ts <= t).

    Timestamps must be strictly increasing; a duplicate ts silently changes
    which p wins the "last entry" lookup, so reject it loudly.
    """
    schedule = sorted((int(t), float(p)) for t, p in p_schedule)
    ts = [t for t, _ in schedule]
    if len(ts) != len(set(ts)):
        raise ValueError(f"duplicate timestamps in p_schedule: {schedule}")
    times = [t for t, _ in schedule]

    def active_p(t):
        p = schedule[0][1]
        for tt, pp in schedule:
            if tt <= t:
                p = pp
            else:
                break
        return p

    return active_p


# ── methods (each is a class: act(s, t, p) + per-trial reset()) ────────────────

class OracleRATS:
    """RATS with the true model.  fixed_p=None -> the true CURRENT model each
    step (the paper's RATS-P_k); fixed_p -> a FROZEN true model at that p
    (RATS-P_{k-1}: the pre-change model, never updated)."""

    name = "oracle_rats"

    def __init__(self, grid, gamma=GAMMA, max_depth=RATS_DEPTH, fixed_p=None):
        self.grid = grid
        self.gamma = gamma
        self.max_depth = max_depth
        self.fixed_p = fixed_p
        self._agent = RATS(None, gamma=gamma, max_depth=max_depth, grid=grid)

    def reset(self):
        pass

    def observe(self, s, a, s2):
        pass

    def act(self, s, t, p):
        p_use = self.fixed_p if self.fixed_p is not None else p
        self._agent.model = GridSnapshot(self.grid, p_use)
        return self._agent.act(s, t0=t)


class RatsPkMinus1(OracleRATS):
    """RATS-P_{k-1}: RATS with the true OLD model (p=0.7) -- an oracle baseline
    that ignores the change, as in the ADA-MCTS paper's table."""

    name = "rats_pkminus1"

    def __init__(self, grid, gamma=GAMMA, max_depth=RATS_DEPTH):
        super().__init__(grid, gamma=gamma, max_depth=max_depth, fixed_p=ORIG_P)


class DPSnapshot:
    name = "dp_snapshot"

    def __init__(self, grid, gamma=GAMMA, max_depth=DP_DEPTH):
        self.grid = grid
        self._agent = DPAgent(None, gamma=gamma, max_depth=max_depth, grid=grid)

    def reset(self):
        pass

    def observe(self, s, a, s2):
        pass

    def act(self, s, t, p):
        self._agent.model = GridSnapshot(self.grid, p)
        return self._agent.act(s, t0=t)


class DPNSMDP:
    name = "dp_nsmdp"

    def __init__(self, grid, dist_by_time, gamma=GAMMA, max_depth=DP_DEPTH):
        self._agent = DPAgent(DynamicGridSnapshot(grid, dist_by_time),
                              gamma=gamma, max_depth=max_depth,
                              is_model_dynamic=True)

    def reset(self):
        pass

    def observe(self, s, a, s2):
        pass

    def act(self, s, t, p):
        return self._agent.act(s, t0=t)


class BNNRATS:
    """RATS with the Dirichlet-BNN snapshot; adaptive=True enables the
    surprise/forget/online-counts loop (our method).

    knobs (for the adaptive-loop investigation, 2026-08-06):
      use_counts     : online counts participate in the predictive mean
      persist_counts : counts survive across episodes (else reset per episode)
      count_w        : weight per online count observation
      drift_reset    : switch the drift filter to detection mode at the change
                       (without it, lambda_hat stays 0 and forget is a no-op)
    """

    name = None

    def __init__(self, bnn, dyn, grid, gamma=GAMMA, max_depth=RATS_DEPTH,
                 adaptive=False, change_step=0, k_forget=K_FORGET,
                 use_counts=True, persist_counts=False, count_w=1.0,
                 drift_reset=False):
        self.bnn = bnn
        self.dyn = dyn
        self.grid = grid
        self.adaptive = adaptive
        self.change_step = change_step
        self.k_forget = k_forget
        self.use_counts = use_counts
        self.persist_counts = persist_counts
        self.count_w = count_w
        self.drift_reset = drift_reset
        self.gamma = gamma
        self.max_depth = max_depth
        self.name = "bnn_rats_adaptive" if adaptive else "bnn_rats_static"

    def reset(self):
        self.bnn.use_counts = self.adaptive and self.use_counts
        self.bnn.retain.fill_(1.0)
        if not (self.adaptive and self.persist_counts):
            self.bnn.reset_counts()
        self.snap = BNNSnapshot(self.dyn, self.bnn, self.grid)
        self._agent = RATS(self.snap, gamma=self.gamma,
                           max_depth=self.max_depth)
        self.drift = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)
        self._drift_done = False
        self.post = 0
        self._last = None        # (obs, act, next_obs, s, a, s2) of t-1

    def observe(self, s, a, s2):
        """Called by the runner right after the transition (s,a)->s2."""
        if not self.adaptive:
            return
        n = self.grid.n_states
        obs = np.zeros(n, dtype=np.float32)
        obs[s] = 1.0
        act_v = np.zeros(self.grid.n_actions, dtype=np.float32)
        act_v[int(a)] = 1.0
        nxt = np.zeros(n, dtype=np.float32)
        nxt[s2] = 1.0
        self._last = (obs, act_v, nxt, int(s), int(a), int(s2))

    def act(self, s, t, p):
        if (self.adaptive and self.change_step is not None
                and self._last is not None and t > 0):
            obs, act_v, nxt, ps, pa, ps2 = self._last
            vs = surprise_dirichlet(self.dyn, self.bnn, obs, act_v, nxt, 0.0,
                                    n_draws=N_POSTERIOR)
            if self.drift_reset and not self._drift_done:
                # change already happened at ts 0: stop "calibrating" so
                # lambda_hat reflects the post-change surprise and forget fires
                self.drift.reset()
                self._drift_done = True
            self.drift.update(vs["delta_n"])
            if t >= self.change_step + 1:
                self.post += 1
                if self.post % self.k_forget == 0:
                    self._forget()
            d = self.grid.direction_of(ps, pa, ps2)
            if d >= 0:
                self.bnn.add_count(ps, pa, d, w=self.count_w)
            self.snap.clear_cache()
        return self._agent.act(s, t0=t)

    def _forget(self):
        from bnn.dirichlet_workflow import forget_dirichlet
        forget_dirichlet(self.bnn, self.drift)
        self.snap.clear_cache()


# ── FIR-CEM (main method): CVaR-CEM planner + the FIR adaptation loop ──────────

class BNNCEM:
    """CVaR-CEM planner with the Dirichlet-BNN; adaptive=True runs the same FIR
    loop as BNNRATS (surprise -> drift -> forget -> learn), and gates the CVaR
    tail alpha by the smoothed surprise (the paper's confidence-gated planner).

    This is the report's "FIR-CEM (本文)" method: component 1 (pretrained
    Dirichlet BNN) + component 2 (FIR) + component 3 (CVaR-CEM, not RATS).
    """

    name = None

    def __init__(self, bnn, dyn, grid, gamma=GAMMA, change_step=0,
                 k_forget=K_FORGET, use_counts=True, persist_counts=False,
                 count_w=1.0, drift_reset=False, adaptive=False,
                 horizon=RATS_DEPTH, alpha_min=CEM_ALPHA_MIN,
                 n_confident=CEM_N_CONFIDENT, cvar_alpha=CEM_CVAR_ALPHA,
                 surprise_tau=None):
        self.bnn = bnn
        self.dyn = dyn
        self.grid = grid
        self.adaptive = adaptive
        self.change_step = change_step
        self.k_forget = k_forget
        self.use_counts = use_counts
        self.persist_counts = persist_counts
        self.count_w = count_w
        self.drift_reset = drift_reset
        self.name = "cem_fir" if adaptive else "cem_static"
        kw = {}
        if surprise_tau is not None:
            kw["surprise_tau"] = surprise_tau
        self._agent = CVaRCEMAgent(
            dyn, bnn, grid.desc_bytes(), device, n_actions=grid.n_actions,
            gamma=gamma, horizon=horizon, cvar_alpha=cvar_alpha,
            adaptive_alpha=CEM_ADAPTIVE_ALPHA, alpha_min=alpha_min,
            n_confident=n_confident, rng=np.random.default_rng(0), **kw)

    def reset(self):
        self.bnn.use_counts = self.adaptive and self.use_counts
        self.bnn.retain.fill_(1.0)
        if not (self.adaptive and self.persist_counts):
            self.bnn.reset_counts()
        self._agent.reset()
        self._agent.pi = self._agent._uniform.copy()
        self.drift = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)
        self._drift_done = False
        self.post = 0
        self._last = None

    def observe(self, s, a, s2):
        if not self.adaptive:
            return
        n = self.grid.n_states
        obs = np.zeros(n, dtype=np.float32)
        obs[s] = 1.0
        act_v = np.zeros(self.grid.n_actions, dtype=np.float32)
        act_v[int(a)] = 1.0
        nxt = np.zeros(n, dtype=np.float32)
        nxt[s2] = 1.0
        self._last = (obs, act_v, nxt, int(s), int(a), int(s2))

    def act(self, s, t, p):
        if (self.adaptive and self.change_step is not None
                and self._last is not None and t > 0):
            obs, act_v, nxt, ps, pa, ps2 = self._last
            vs = surprise_dirichlet(self.dyn, self.bnn, obs, act_v, nxt, 0.0,
                                    n_draws=N_POSTERIOR)
            if self.drift_reset and not self._drift_done:
                self.drift.reset()
                self._drift_done = True
            self.drift.update(vs["delta_n"])
            # confidence-gated alpha: same surprise signal gates the CVaR tail
            self._agent.surprise_bar = self.drift.delta_bar
            self._agent.n_since_change = self.post
            if t >= self.change_step + 1:
                self.post += 1
                if self.post % self.k_forget == 0:
                    self._forget()
            d = self.grid.direction_of(ps, pa, ps2)
            if d >= 0:
                self.bnn.add_count(ps, pa, d, w=self.count_w)
        obs = np.zeros(self.grid.n_states, dtype=np.float32)
        obs[int(s)] = 1.0
        return int(np.argmax(self._agent.act(obs)))

    def _forget(self):
        from bnn.dirichlet_workflow import forget_dirichlet
        forget_dirichlet(self.bnn, self.drift)
        # FIR-CEM specific: after forgetting, the model no longer matches the OLD
        # env by construction, so the stale-env surprise is no longer informative.
        # Reset the drift accumulator so conf_surprise recovers on the NEW-env
        # evidence instead of staying pinned to 0 for the whole episode.
        self.drift._reset_detection()

# ── unbounded-RATS baselines (we face an UNBOUNDED change: L_p / L_r unknown) ─

class RATSCV01(BNNRATS):
    """Scheme 1: sample N posterior models, take the WORST at every chance node
    (~1%-CVaR tail over the model posterior, matching RATS's worst-case over a
    Wasserstein ball but without needing the Lipschitz constant L_p).

    In the unbounded case there is no valid L_p, so the pessimism comes from
    the model's own posterior spread instead of an analytically-sized ball.
    """

    name = "rats_cv01"

    def __init__(self, bnn, dyn, grid, gamma=GAMMA, max_depth=RATS_DEPTH,
                 n_draws=N_MODEL_DRAWS, alpha=RATS_CVAR_ALPHA, **kw):
        super().__init__(bnn, dyn, grid, gamma=gamma, max_depth=max_depth,
                         adaptive=False, **kw)
        self.n_draws = n_draws
        self.alpha = alpha

    def reset(self):
        super().reset()
        self.snap = WorstKSnapshot(self.dyn, self.bnn, self.grid,
                                   n_draws=self.n_draws)
        self._agent = RATS(self.snap, gamma=self.gamma,
                           max_depth=self.max_depth)

    def act(self, s, t, p):
        """Worst-of-K at the root: for each action, compute the child values v
        on the (worst-case, mean-model) RATS tree, then pick the action whose
        worst posterior draw -- ranked by w @ v -- is HIGHEST (the 1%-CVaR
        analogue of RATS's worst-case ball, in the unbounded case)."""
        self._agent._memo = {}
        best_a, best_val = 0, -np.inf
        for a in range(self.grid.n_actions):
            mean = self.snap.p_cells(s, a)                     # mean row (K,)
            p_cells = mean
            children = [int(i) for i in range(self.grid.n_states)
                        if p_cells[i] > 1e-12]
            v_child = np.array([self._agent._V(s2, 1) for s2 in children])
            rows = self.snap._draw_rows(s, a)[:, children]     # (K, |children|)
            worst = float(np.min(rows @ v_child))              # worst draw
            val = self.gamma * worst + self.snap.expected_reward(s, a)
            if val > best_val:
                best_val, best_a = val, a
        return best_a


class RATSCalibrated(BNNRATS):
    """Scheme 2: sample N posterior models, CALIBRATE L_p from their Wasserstein
    spread (the change is unbounded, so we estimate the plausible-evolution
    radius empirically), then run the unmodified RATS worst-case over the
    L_p-sized ball.  Minimizes changes to the existing RATS code."""

    name = "rats_cal"

    def __init__(self, bnn, dyn, grid, gamma=GAMMA, max_depth=RATS_DEPTH,
                 n_draws=N_MODEL_DRAWS, **kw):
        super().__init__(bnn, dyn, grid, gamma=gamma, max_depth=max_depth,
                         adaptive=False, **kw)
        self.n_draws = n_draws

    def reset(self):
        super().reset()
        self.snap = CalibratedSnapshot(self.dyn, self.bnn, self.grid,
                                       n_draws=self.n_draws)
        self._agent = RATS(self.snap, gamma=self.gamma,
                           max_depth=self.max_depth)

    def act(self, s, t, p):
        """Calibrated worst-case at the root: per action, size the chance-node
        ball with the per-(s,a) calibrated L_p (spread of the posterior draws),
        then apply the closed-form Property-3 worst-case distribution against
        the tree's child values.  Interior nodes keep the model's fixed L_p."""
        self._agent._memo = {}
        best_a, best_val = 0, -np.inf
        for a in range(self.grid.n_actions):
            mean = self.snap.p_cells(s, a)
            children = [int(i) for i in range(self.grid.n_states)
                        if mean[i] > 1e-12]
            v_child = np.array([self._agent._V(s2, 1) for s2 in children])
            w0 = np.array([mean[s2] for s2 in children], dtype=np.float64)
            w0 /= w0.sum()
            c = self.snap.calibrated_Lp(s, a)          # calibrated ball radius
            d_mat = _dist_matrix(self.grid, children)
            w = worstcase_distribution_direct_method(v_child, w0, c, d_mat)
            val = self.gamma * float(w @ v_child) + \
                self.snap.expected_reward(s, a)
            if val > best_val:
                best_val, best_a = val, a
        return best_a


class WorstKSnapshot(BNNSnapshot):
    """Scheme 1 model: per-(s,a) chance node, sample n_draws posterior
    transition rows and return the one with the LOWEST expected value of the
    reachable children -- the empirical worst over the model posterior.

    RATS already memoizes _V over (state, depth), so p_cells is consulted AFTER
    the child values v_child exist: _V_worst uses a nested RATS that reads the
    mean row from this snapshot, then ranks draws by w @ v_child.  p_cells is
    the plain deterministic mean (what _V computes against); the "worst-of-K"
    only matters at the root action choice."""

    def __init__(self, dyn, bnn, grid, n_draws=N_MODEL_DRAWS, L_p=1.0,
                 L_r=0.0, tau=1.0):
        super().__init__(dyn, bnn, grid, L_p=L_p, L_r=L_r, tau=tau)
        self.n_draws = n_draws

    @torch.no_grad()
    def _draw_rows(self, s, a):
        obs = np.zeros(self.n_states, dtype=np.float32)
        obs[s] = 1.0
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        act_t = self._eye_a[a].unsqueeze(0)
        state = self.dyn.reset(obs_t)
        rows = []
        for _ in range(self.n_draws):
            next_obs, _, _, _ = self.dyn.sample(act_t, state, deterministic=False)
            p = next_obs[0, :self.n_states].clamp_min(0.0).cpu().numpy()
            p /= p.sum()
            rows.append(p)
        return np.stack(rows)                       # (K, n_states)


def _w1_between(p, q):
    """Wasserstein-1 surrogate between two distributions over the same cell
    support: identical cells match exactly, the surplus mass moves at unit cost
    (grid Manhattan >= 1 for distinct cells) -> W1 ~ half the L1 distance."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0, None)
    q = np.clip(np.asarray(q, dtype=np.float64), 0, None)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    return float(np.abs(p - q).sum() / 2.0)


class CalibratedSnapshot(BNNSnapshot):
    """Scheme 2 model: mean transition row (as BNNSnapshot) + a per-(s,a)
    calibrated L_p = Wasserstein spread of the posterior draws around the mean,
    used by RATS to size the worst-case ball per chance node."""

    def __init__(self, dyn, bnn, grid, n_draws=N_MODEL_DRAWS, L_p=1.0,
                 L_r=0.0, tau=1.0):
        super().__init__(dyn, bnn, grid, L_p=L_p, L_r=L_r, tau=tau)
        self.n_draws = n_draws

    def calibrated_Lp(self, s, a):
        rows = WorstKSnapshot._draw_rows(self, int(s), int(a))
        mean = rows.mean(0)
        return float(np.mean([_w1_between(r, mean) for r in rows]))


class ADAMCTS:
    name = "ada_mcts"

    def __init__(self, bnn, dyn, grid, gamma=GAMMA, m_simulations=M_SIMULATIONS,
                 change_step=0, rng=None, dpas_gamma=DPAS_GAMMA, counts=True):
        self.bnn = bnn
        self.grid = grid
        self.change_step = change_step
        self.counts = counts
        self.rng = rng or np.random.default_rng(0)
        self._agent = ADAMCTSAgent(dyn, bnn, grid.desc_bytes(), device,
                                   n_actions=grid.n_actions, gamma=gamma,
                                   rng=self.rng,
                                   m_simulations=m_simulations,
                                   dpas_gamma=dpas_gamma)
        self._notified = False

    def reset(self):
        self.bnn.use_counts = self.counts
        self.bnn.retain.fill_(1.0)
        self.bnn.reset_counts()
        self._agent.reset()
        # NOTE: do NOT reset _notified here.  The env changes once per phase
        # (ts 0), so notify_change must fire once per phase -- per-trial
        # re-notify re-snapshots M_{k-1} every episode and keeps DPAS stuck in
        # worst-case mode (on bridge this one-hots holes near the goal and
        # collapses the goal rate to ~0.1-0.5).
        self._last = None

    def observe(self, s, a, s2):
        if self._notified:
            self._agent.learn(int(s), int(a), int(s2))

    def act(self, s, t, p):
        if (self.change_step is not None and t == self.change_step
                and not self._notified):
            self._agent.notify_change()
            self._notified = True
        obs = np.zeros(self.grid.n_states, dtype=np.float32)
        obs[int(s)] = 1.0
        return int(np.argmax(self._agent.act(obs)))


class MCTSStatic(ADAMCTS):
    """MCTS-P_hat_{k-1}: plain MCTS on the pretrained BNN, no change
    notification, no online learning -- the paper's non-adaptive learned-model
    baseline (ADA-MCTS with the DPAS adaptation disabled)."""

    name = "mcts_static"

    def __init__(self, bnn, dyn, grid, gamma=GAMMA,
                 m_simulations=M_SIMULATIONS, rng=None, dpas_gamma=DPAS_GAMMA):
        super().__init__(bnn, dyn, grid, gamma=gamma,
                         m_simulations=m_simulations, change_step=None,
                         rng=rng, dpas_gamma=dpas_gamma, counts=False)


def run_episode(grid, method, p_schedule, seed, max_steps):
    """One episode; returns (discounted_return, goal, per-step rewards)."""
    active_p = active_p_fn(p_schedule)
    rng = np.random.default_rng(seed)
    s = grid.start_state
    t = 0
    G = 0.0
    disc = 1.0
    rewards = []
    while t < max_steps:
        p = active_p(t)
        slip = np.array(grid.slip_dist(p))
        a = method.act(s, t, p)
        k = int(rng.choice(grid.k_dir, p=slip))
        s2 = grid.move(s, grid.dir_actions(int(a))[k])
        r = cell_reward(grid, s2)
        G += disc * r
        rewards.append(r)
        disc *= GAMMA
        method.observe(s, int(a), s2)
        s = s2
        t += 1
        if grid.flat_desc[s] in "GH":
            break
    goal = 1 if grid.flat_desc[s] == "G" else 0
    return G, goal, rewards


def build_methods(args, grid, bnn, dyn, dist_by_time, names, change_step=None):
    """Build method wrappers.

    change_step=None (the STATIONARY phase) disables the adaptation loops: the
    surprise/forget/counts loop and the ADA-MCTS change notification never fire,
    so "adaptive" behaves exactly like "static" -- the stationary check measures
    the pretrained model, not the adaptation.
    """
    out = {}
    for name in names:
        if name == "dp_nsmdp":
            out[name] = DPNSMDP(grid, dist_by_time, gamma=GAMMA,
                                max_depth=args.dp_depth)
        elif name == "dp_snapshot":
            out[name] = DPSnapshot(grid, gamma=GAMMA, max_depth=args.dp_depth)
        elif name == "oracle_rats":
            out[name] = OracleRATS(grid, gamma=GAMMA, max_depth=args.rats_depth)
        elif name == "rats_pkminus1":
            out[name] = RatsPkMinus1(grid, gamma=GAMMA, max_depth=args.rats_depth)
        elif name == "bnn_rats_static":
            out[name] = BNNRATS(bnn, dyn, grid, gamma=GAMMA,
                                max_depth=args.rats_depth, adaptive=False)
        elif name == "bnn_rats_adaptive":
            out[name] = BNNRATS(bnn, dyn, grid, gamma=GAMMA,
                                max_depth=args.rats_depth, adaptive=True,
                                change_step=change_step,
                                k_forget=args.k_forget,
                                use_counts=args.use_counts,
                                persist_counts=args.persist_counts,
                                count_w=args.count_w,
                                drift_reset=args.drift_reset)
        elif name == "ada_mcts":
            out[name] = ADAMCTS(bnn, dyn, grid, gamma=GAMMA,
                                m_simulations=args.m_simulations,
                                change_step=change_step,
                                dpas_gamma=args.dpas_gamma)
        elif name == "cem_fir":
            out[name] = BNNCEM(bnn, dyn, grid, gamma=GAMMA,
                               change_step=change_step,
                               k_forget=args.k_forget,
                               use_counts=args.use_counts,
                               persist_counts=args.persist_counts,
                               count_w=args.count_w,
                               drift_reset=args.drift_reset,
                               adaptive=True,
                               horizon=args.cem_horizon,
                               alpha_min=args.cem_alpha_min,
                               n_confident=args.cem_n_confident,
                               cvar_alpha=args.cem_cvar_alpha,
                               surprise_tau=args.cem_surprise_tau)
        elif name == "cem_static":
            out[name] = BNNCEM(bnn, dyn, grid, gamma=GAMMA,
                               change_step=change_step,
                               k_forget=args.k_forget,
                               use_counts=args.use_counts,
                               persist_counts=args.persist_counts,
                               count_w=args.count_w,
                               drift_reset=args.drift_reset,
                               adaptive=False,
                               horizon=args.cem_horizon,
                               alpha_min=args.cem_alpha_min,
                               n_confident=args.cem_n_confident,
                               cvar_alpha=args.cem_cvar_alpha,
                               surprise_tau=args.cem_surprise_tau)
        elif name == "rats_cv01":
            out[name] = RATSCV01(bnn, dyn, grid, gamma=GAMMA,
                                 max_depth=args.rats_depth)
        elif name == "rats_cal":
            out[name] = RATSCalibrated(bnn, dyn, grid, gamma=GAMMA,
                                       max_depth=args.rats_depth)
        elif name == "mcts_static":
            out[name] = MCTSStatic(bnn, dyn, grid, gamma=GAMMA,
                                   m_simulations=args.m_simulations,
                                   dpas_gamma=args.dpas_gamma)
        else:
            raise ValueError(f"unknown method {name}")
    return out


def _worker(task):
    """Run ONE (phase, method) task in a worker process and return its results.

    Each worker builds its own grid/BNN/method instance, so no state is shared
    across tasks -- in particular the ADA-MCTS "notify once per phase" invariant
    (M_{k-1} is snapshotted once per phase, NOT per trial) is preserved exactly
    as in the serial runner.  Trials keep their index seeds, so every number is
    bit-identical to a serial run.

    task = (grid_name, method_name, phase_label, p_schedule, dist_by_time,
            change_step, trials, max_steps, cfg_dict)
    """
    (grid_name, method_name, phase_label, p_schedule, dist_by_time,
     change_step, trials, max_steps, cfg) = task
    torch.manual_seed(0)   # posterior draws reproducible across workers/runs
    grid = get_grid(grid_name)
    bnn, dyn = make_dirichlet_bnn(grid.n_states, grid.n_actions, grid=grid)
    ckpt = _HERE / "data" / grid_name / ckpt_name(grid, cfg["orig_p"])
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found -- run pretrain_gridworld.py")
    bnn.load(ckpt.parent, filename=ckpt.name)
    bnn.num_weight_groups = 1
    bnn.num_train_points = 20000
    args = argparse.Namespace(**cfg)
    method = build_methods(args, grid, bnn, dyn, dist_by_time, [method_name],
                           change_step=change_step)[method_name]
    Gs, goals, trial0 = [], [], None
    for trial in range(trials):
        method.reset()
        G, goal, rewards = run_episode(grid, method, p_schedule, trial, max_steps)
        Gs.append(G)
        goals.append(goal)
        if trial == 0:
            trial0 = rewards
    return dict(phase=phase_label, method=method_name, Gs=Gs, goals=goals,
                trial0=trial0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="cliffwalking")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--workers", type=int, default=12,
                    help="parallel processes (tasks = methods x phases; "
                         "16-core machine -> 12 is a good default)")
    ap.add_argument("--change-p", type=float, nargs="*", default=None)
    ap.add_argument("--change-step", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=None,
                    help="DEPRECATED: use --rats-depth / --dp-depth")
    ap.add_argument("--rats-depth", type=int, default=RATS_DEPTH)
    ap.add_argument("--dp-depth", type=int, default=DP_DEPTH)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--m-simulations", type=int, default=M_SIMULATIONS)
    ap.add_argument("--dpas-gamma", type=float, default=DPAS_GAMMA)
    ap.add_argument("--k-forget", type=int, default=K_FORGET)
    ap.add_argument("--count-w", type=float, default=1.0)
    ap.add_argument("--persist-counts", action="store_true")
    # Fixed 2026-08-06: drift_reset on by default (forget actually fires) and
    # online counts off by default (they distort the tiny-concentration head).
    ap.add_argument("--no-drift-reset", dest="drift_reset",
                    action="store_false", default=True)
    ap.add_argument("--counts", dest="use_counts",
                    action="store_true", default=False)
    ap.add_argument("--methods", nargs="*", default=None)
    # CEM planner tuning (2026-08-13, bridge short-episode tuning)
    ap.add_argument("--cem-horizon", type=int, default=RATS_DEPTH,
                    help="CEM planning horizon (bridge's full trip is ~2-4 steps)")
    ap.add_argument("--cem-alpha-min", type=float, default=CEM_ALPHA_MIN)
    ap.add_argument("--cem-n-confident", type=int, default=CEM_N_CONFIDENT)
    ap.add_argument("--cem-cvar-alpha", type=float, default=CEM_CVAR_ALPHA,
                    help="fixed CVaR tail when adaptive_alpha is off")
    ap.add_argument("--cem-surprise-tau", type=float, default=None,
                    help="surprise sensitivity for the confidence gate (planner "
                         "default 2.0; p=1.0 pretraining makes surprise spike to "
                         "hundreds, so a larger tau relaxes the gate faster)")
    args = ap.parse_args()
    if args.max_depth is not None:
        args.rats_depth = args.dp_depth = args.max_depth

    grid = get_grid(args.grid)
    max_steps = args.max_steps or (10 if grid.name == "bridge" else 100)
    change_ps = args.change_p if args.change_p else CHANGE_PS
    methods = args.methods or ["oracle_rats", "rats_cv01", "rats_cal",
                               "bnn_rats_static", "bnn_rats_adaptive",
                               "cem_static", "cem_fir",
                               "ada_mcts", "mcts_static"]
    log_path = _HERE / f"gridworld_{grid.name}_results.log"
    out = open(log_path, "w")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        out.write(msg + "\n")
        out.flush()
        print(msg, flush=True)

    log("=" * 90)
    log(f"Gridworld RATS experiments | grid={grid.name} ({grid.nrow}x{grid.ncol}, "
        f"K={grid.k_dir})")
    log(f"  original p={ORIG_P} | change at ts {args.change_step} -> "
        f"p in {change_ps}")
    log(f"  gamma={GAMMA} | rats_depth={args.rats_depth} dp_depth={args.dp_depth} "
        f"| max_steps={max_steps} | trials={args.trials} | workers={args.workers}")
    log(f"  methods: {methods}")
    log(f"  ADA-MCTS: {args.m_simulations} simulations/action (paper value)")
    log("=" * 90)

    # ── build (phase, method) tasks ─────────────────────────────────────────
    cfg = dict(orig_p=ORIG_P, rats_depth=args.rats_depth, dp_depth=args.dp_depth,
               k_forget=args.k_forget, use_counts=args.use_counts,
               persist_counts=args.persist_counts, count_w=args.count_w,
               drift_reset=args.drift_reset,
               m_simulations=args.m_simulations, dpas_gamma=args.dpas_gamma,
               cem_horizon=args.cem_horizon, cem_alpha_min=args.cem_alpha_min,
               cem_n_confident=args.cem_n_confident,
               cem_cvar_alpha=args.cem_cvar_alpha,
               cem_surprise_tau=args.cem_surprise_tau)
    tasks = []
    # 1. stationary verification (change_step=None disables adaptation)
    tasks.append((args.grid, "stationary", [(0, ORIG_P)],
                  {0: grid.slip_dist(ORIG_P)}, None))
    for p_new in change_ps:
        # A change at ts c means p_new is active from decision epoch c onward.
        # With c <= 0 the first action already runs under p_new, so the ORIG_P
        # segment is dropped entirely (two entries with the same ts would make
        # active_p_fn return the last-sorted one -- silently keeping the env at
        # ORIG_P forever, as happened in the first experiment round).
        if args.change_step <= 0:
            p_schedule = [(0, p_new)]
            dist_by_time = {0: grid.slip_dist(p_new)}
        else:
            p_schedule = [(0, ORIG_P), (args.change_step, p_new)]
            dist_by_time = {0: grid.slip_dist(ORIG_P),
                            args.change_step: grid.slip_dist(p_new)}
        tasks.append((args.grid, f"p={p_new:g}", p_schedule, dist_by_time,
                      args.change_step))
    full_tasks = [
        (args.grid, name, phase, p_sched, dist, cs, args.trials, max_steps, cfg)
        for (_, phase, p_sched, dist, cs) in tasks for name in methods
    ]

    # ── run in parallel; results come back in completion order ─────────────
    # spawn (NOT fork): torch's CUDA state cannot be re-initialized in forked
    # children once the parent has touched the driver (torch.cuda.is_available()
    # in config.py already does).  Workers re-import and build everything.
    ctx = multiprocessing.get_context("spawn")
    summary = {}
    seen_phases = set()
    with ctx.Pool(args.workers) as pool:
        for res in pool.imap_unordered(_worker, full_tasks):
            phase, name = res["phase"], res["method"]
            if phase not in seen_phases:
                seen_phases.add(phase)
                log("\n" + "=" * 90)
                if phase == "stationary":
                    log(f"1. STATIONARY verification (p={ORIG_P}) -- pretrained "
                        f"model must perform well")
                else:
                    log(f"2. NON-STATIONARY: p {ORIG_P} -> {phase} at "
                        f"ts {args.change_step}")
                log("=" * 90)
            Gs, goals = res["Gs"], res["goals"]
            if phase != "stationary":
                p_val = float(phase.split("=")[1])
                summary.setdefault(name, {})[p_val] = (np.mean(Gs), np.mean(goals))
            log(f"  [{phase:<8s}] {name:<18s}: return {np.mean(Gs):+.3f} | "
                f"goal rate {np.mean(goals):.3f}")
            if res["trial0"] is not None:
                log(f"      per-step rewards (trial 0): "
                    f"{[round(r, 1) for r in res['trial0']]}")

    # ── summary table ──────────────────────────────────────────────────────
    log("\n" + "=" * 90)
    log("SUMMARY: goal rate by p (paper's return convention, holes = 0)")
    log("-" * 90)
    header = f"{'method':<18s}"
    for p in change_ps:
        header += f"  p={p:<4.1f}"
    log(header)
    for name in methods:
        if name not in summary:
            continue
        line = f"{name:<18s}"
        for p in change_ps:
            if p in summary[name]:
                line += f"  {summary[name][p][1]:6.3f}"
            else:
                line += "       "
        log(line)
    log("\nSUMMARY: discounted return (gamma=0.99, holes = -1)")
    log("-" * 90)
    header = f"{'method':<18s}"
    for p in change_ps:
        header += f"  p={p:<4.1f}"
    log(header)
    for name in methods:
        if name not in summary:
            continue
        line = f"{name:<18s}"
        for p in change_ps:
            if p in summary[name]:
                line += f"  {summary[name][p][0]:6.2f}"
            else:
                line += "       "
        log(line)

    log("\nDONE.")
    out.close()


if __name__ == "__main__":
    main()
