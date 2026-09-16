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

GAMMA = 0.9999          # discount (user request 2026-08-22: 0.9999 for BOTH
                        # planning and evaluation -- planners receive this gamma,
                        # run_episode discounts returns with it)
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
CEM_ALPHA_MIN = 0.30    # most risk-averse CVaR tail (30% worst returns); 0.10 was
                        # too conservative (over-cautious phases dragged cem_fir
                        # below cem_static), 0.95 barely differs from risk-neutral
CEM_N_CONFIDENT = 8     # post-change samples for the data gate to saturate
CEM_SURPRISE_TAU = 50.0  # surprise sensitivity; p=1.0 pretraining spikes the
                         # surprise into the hundreds, so the default tau=2.0
                         # would pin conf at 0 for the whole episode
# unbounded-RATS baselines (scheme 1 worst-of-K / scheme 2 calibrated L_p)
N_MODEL_DRAWS = 100     # posterior model draws
RATS_CVAR_ALPHA = 0.01  # scheme 1: ~1% empirical tail over model draws


def cell_reward(grid, s):
    c = grid.flat_desc[int(s)]
    if c == "G":
        return 1.0
    if c == "H":
        return grid.hole_reward      # 0.0 per the paper's "holes = 0" convention
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
        # BUG FIX (2026-09-03, matches BNNCEM._forget): without this, delta_bar
        # is a cumulative mean of ALL post-detection excess -- one huge spike
        # from the FIRST slip under p=1.0 pretraining (near-deterministic model,
        # so any slip has near-zero predicted probability -> NLL/entropy blows
        # up into the hundreds) stays baked into delta_bar for the rest of the
        # episode, since it decays only as 1/n.  Every subsequent k_forget tick
        # then computes another near-zero rho and multiplies retain down AGAIN,
        # driving it to a hard 0.0 (all prior directional signal lost, not just
        # down-weighted) regardless of how large the actual change was.  This
        # made bnn_rats_adaptive WORSE at small changes (p=0.9, where the old
        # prior was still ~90% correct) than at large ones (p=0.4) --
        # non-monotonic in p.  Resetting the drift accumulator after forgetting
        # lets delta_bar reflect only NEW evidence, so retain stops crashing
        # from stale surprise it has already acted on.
        self.drift._reset_detection()


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
                 surprise_tau=None, n_candidates=None, k_models=None,
                 n_rollouts=None, n_cem_iters=None, warm_blend=0.0,
                 do_forget=True, n_unfrozen=0, retrain_every=3,
                 retrain_steps=5, retrain_lr=1e-2, retrain_buf_cap=64,
                 seed=0, retrain_min_conf=0.0):
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
        # SFIR ablation knobs (2026-09-10).  do_forget=False removes the "F"
        # (retain never decays); n_unfrozen>0 adds gradient "R" -- retrain the
        # top n_unfrozen layers of the Dirichlet DIRECTION path
        # (1=head only = "our method", 3=head+whole trunk) on the post-change
        # buffer via retrain_dirichlet, every retrain_every post-change steps.
        self.do_forget = do_forget
        self.n_unfrozen = int(n_unfrozen)
        self.retrain_every = int(retrain_every)
        self.retrain_steps = int(retrain_steps)
        self.retrain_lr = float(retrain_lr)
        # Gate retrain on the SAME confidence signal that already drives
        # plan_retain/the CVaR tail (self._agent._confidence(), computed from
        # n_since_change + surprise_bar -- no new mechanism, just reusing an
        # existing computed value as a second condition).  0.0 = always fire
        # on cadence (old behaviour).  Motivation: retraining the head on a
        # handful of samples while confidence is still near 0 (right after a
        # SEVERE change, e.g. p=0.3) risks overfitting noise before there is
        # enough data to trust; delaying retrain until confidence has partly
        # recovered lets forget do the early work and retrain only sharpen
        # an already-reasonable belief.
        self.retrain_min_conf = float(retrain_min_conf)
        self.retrain_buf_cap = int(retrain_buf_cap)
        # Gradient retrain mutates weight_mu/bias_mu IN PLACE, and the bnn object
        # is shared across all trials in a _worker -- retain/counts are reset in
        # reset() but weights are not.  Snapshot the direction path's mean
        # weights (superset: n_unfrozen=3) so reset() can restore a pristine
        # model each trial.
        self._retrain_pristine = []
        if self.n_unfrozen > 0:
            from bnn.dirichlet_workflow import unfrozen_params_dirichlet
            self._retrain_pristine = [
                (p, p.detach().clone())
                for p in unfrozen_params_dirichlet(bnn, 3)]
        self._retrain_buf = []
        self._opt = None
        self.name = "cem_fir" if adaptive else "cem_static"
        kw = {}
        if surprise_tau is not None:
            kw["surprise_tau"] = surprise_tau
        if n_candidates is not None:
            kw["n_candidates"] = n_candidates
        if k_models is not None:
            kw["k_models"] = k_models
        if n_rollouts is not None:
            kw["n_rollouts"] = n_rollouts
        if n_cem_iters is not None:
            kw["n_cem_iters"] = n_cem_iters
        if warm_blend:
            kw["warm_blend"] = warm_blend
        self._agent = CVaRCEMAgent(
            dyn, bnn, grid.desc_bytes(), device, n_actions=grid.n_actions,
            gamma=gamma, horizon=horizon, cvar_alpha=cvar_alpha,
            adaptive_alpha=CEM_ADAPTIVE_ALPHA, alpha_min=alpha_min,
            n_confident=n_confident, rng=np.random.default_rng(seed), **kw)

    def reset(self):
        self.bnn.use_counts = self.adaptive and self.use_counts
        self.bnn.retain.fill_(1.0)
        if not (self.adaptive and self.persist_counts):
            self.bnn.reset_counts()
        self._agent.reset()
        self._agent.pi = self._agent._uniform.copy()
        # Per-trial gate state: trials must be i.i.d. -- previously
        # surprise_bar / n_since_change persisted across trials, so trial 2+
        # started with the data gate already saturated.
        self._agent.surprise_bar = 1.0
        self._agent.n_since_change = 0
        self._agent.plan_retain = None
        self.drift = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)
        self._drift_done = False
        self.post = 0
        self._last = None
        # undo any gradient retrain from the previous trial (weights are shared
        # across trials; restore the pristine direction path so trials stay i.i.d.)
        for p, c in self._retrain_pristine:
            p.data.copy_(c)
        self._retrain_buf = []
        self._opt = None

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
                if self.do_forget and self.post % self.k_forget == 0:
                    self._forget()
                # SFIR "R": buffer this post-change transition, gradient-retrain
                # the top n_unfrozen direction-path layers every retrain_every
                # steps (no-op when n_unfrozen == 0).
                if self.n_unfrozen > 0:
                    self._retrain_buf.append((obs, act_v, ps2))
                    if len(self._retrain_buf) > self.retrain_buf_cap:
                        self._retrain_buf = self._retrain_buf[-self.retrain_buf_cap:]
                    if (self.post % self.retrain_every == 0
                            and self._agent._confidence() >= self.retrain_min_conf):
                        self._retrain()
            d = self.grid.direction_of(ps, pa, ps2)
            if d >= 0:
                self.bnn.add_count(ps, pa, d, w=self.count_w)
        # Planning-time uncertainty gate (SFI "inflate before planning"): scale
        # the planner's view of the pretrained model by the same confidence that
        # gates the CVaR tail.  At the announced change (no post-change data)
        # conf = 0 -> the planner scores candidates on an inflated (near-uniform)
        # model, so the CVaR tail is real even though the pretrained alpha0 is
        # large; as counts accumulate conf -> 1 and the true model is trusted.
        # Static / stationary phases keep plan_retain = None (full trust).
        if self.adaptive and self.change_step is not None:
            self._agent.plan_retain = self._agent._confidence()
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

    def _retrain(self):
        """SFIR "R": gradient-retrain the top n_unfrozen layers of the direction
        path (head only for n_unfrozen=1) on the post-change buffer, via the bare
        categorical NLL (no KL -- matches retrain_dirichlet's design).  retain /
        counts participate in the forward exactly as at plan time."""
        if not self._retrain_buf:
            return
        import torch
        from bnn.dirichlet_workflow import (retrain_dirichlet,
                                            unfrozen_params_dirichlet)
        if self._opt is None:
            self._opt = torch.optim.Adam(
                unfrozen_params_dirichlet(self.bnn, self.n_unfrozen),
                lr=self.retrain_lr)
        obs = torch.as_tensor(np.stack([b[0] for b in self._retrain_buf]),
                              dtype=torch.float32, device=device)
        act = torch.as_tensor(np.stack([b[1] for b in self._retrain_buf]),
                              dtype=torch.float32, device=device)
        model_in = self.dyn._get_model_input(obs, act)
        s2_idx = torch.as_tensor([[int(b[2])] for b in self._retrain_buf],
                                 dtype=torch.long, device=device)
        with torch.enable_grad():
            retrain_dirichlet(self.bnn, self._opt, model_in, s2_idx,
                              n_steps=self.retrain_steps)

ADA_CEM_N_THRESHOLD = 3   # post-change samples before switching phases -- matches
                          # ADA-MCTS's DPAS _n_threshold (planning/ada_mcts.py)


class CEMADA:
    """CVaR-CEM planner + ADA-MCTS's adaptation instead of SFI (2026-09-03 user
    request: "ada-cem-cvar是把sfir-cem-cvar中的sfir换成ada-mcts的方法").

    Same planner as cem_fir/cem_static (CVaRCEMAgent), but the adaptation loop
    is ADA-MCTS's DPAS two-phase switch, not surprise/forget/plan_retain:
      - notified once at change_step (like ADAMCTS.notify_change), not via a
        surprise detector.
      - ONLY online counts accumulate post-notification (no retain decay, no
        forget) -- bnn.add_count every step, matching ADAMCTS.learn().
      - the CVaR tail is a HARD two-phase switch: alpha_min (worst-case) until
        n_threshold post-change samples observed, then alpha=1.0 (risk-neutral)
        -- the direct analogue of DPAS's `_training_started` boolean gate,
        applied to the CEM tail instead of MCTS's chance-node sampling rule.
    """

    name = "cem_ada"

    def __init__(self, bnn, dyn, grid, gamma=GAMMA, change_step=0,
                 n_threshold=ADA_CEM_N_THRESHOLD, alpha_min=CEM_ALPHA_MIN,
                 horizon=RATS_DEPTH, n_candidates=None, k_models=None,
                 n_rollouts=None, n_cem_iters=None):
        self.bnn = bnn
        self.dyn = dyn
        self.grid = grid
        self.change_step = change_step
        self.n_threshold = n_threshold
        self.alpha_min = alpha_min
        kw = {}
        if n_candidates is not None:
            kw["n_candidates"] = n_candidates
        if k_models is not None:
            kw["k_models"] = k_models
        if n_rollouts is not None:
            kw["n_rollouts"] = n_rollouts
        if n_cem_iters is not None:
            kw["n_cem_iters"] = n_cem_iters
        self._agent = CVaRCEMAgent(
            dyn, bnn, grid.desc_bytes(), device, n_actions=grid.n_actions,
            gamma=gamma, horizon=horizon, cvar_alpha=1.0,
            adaptive_alpha=False,   # alpha driven manually below (two-phase)
            alpha_min=alpha_min, rng=np.random.default_rng(0), **kw)
        self._notified = False
        self._post = 0

    def reset(self):
        self.bnn.use_counts = True
        self.bnn.retain.fill_(1.0)   # ADA never forgets -- no retain decay
        self.bnn.reset_counts()
        self._agent.reset()
        self._agent.pi = self._agent._uniform.copy()
        self._notified = False
        self._post = 0

    def observe(self, s, a, s2):
        if not self._notified:
            return
        d = self.grid.direction_of(s, a, s2)
        if d >= 0:
            self.bnn.add_count(int(s), int(a), d)
        self._post += 1

    def act(self, s, t, p):
        if (self.change_step is not None and t == self.change_step
                and not self._notified):
            self._notified = True
            self._post = 0
        self._agent.cvar_alpha = (
            1.0 if (not self._notified) or self._post >= self.n_threshold
            else self.alpha_min)
        obs = np.zeros(self.grid.n_states, dtype=np.float32)
        obs[int(s)] = 1.0
        return int(np.argmax(self._agent.act(obs)))


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
                 change_step=0, rng=None, dpas_gamma=DPAS_GAMMA, counts=True,
                 n_threshold=3, iid_trials=False, rollout_to_terminal=False,
                 max_rollout_steps=100):
        self.bnn = bnn
        self.grid = grid
        self.change_step = change_step
        self.counts = counts
        self.iid_trials = iid_trials
        self.rng = rng or np.random.default_rng(0)
        self._agent = ADAMCTSAgent(dyn, bnn, grid.desc_bytes(), device,
                                   n_actions=grid.n_actions, gamma=gamma,
                                   rng=self.rng,
                                   m_simulations=m_simulations,
                                   dpas_gamma=dpas_gamma,
                                   n_threshold=n_threshold,
                                   rollout_to_terminal=rollout_to_terminal,
                                   max_rollout_steps=max_rollout_steps)
        self._notified = False

    def reset(self):
        self.bnn.use_counts = self.counts
        self.bnn.retain.fill_(1.0)
        self.bnn.reset_counts()
        self._agent.reset()
        # NOTE: do NOT reset _notified here by default.  The env changes once
        # per phase (ts 0), so notify_change must fire once per phase -- per-
        # trial re-notify keeps DPAS in its worst-case warm-up every episode
        # (on bridge this one-hots holes near the goal and collapses the goal
        # rate to ~0.1-0.5).  iid_trials=True opts into exactly that: each trial
        # is an independent run with its own N_threshold warm-up, the paper's
        # per-seed protocol (Act As You Learn reproduction, 2026-09-15).
        # notify_change() keeps an existing bnn_prev, so M_{k-1} stays the
        # pretrained snapshot either way.
        if self.iid_trials:
            self._notified = False
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
                                dpas_gamma=args.dpas_gamma,
                                n_threshold=getattr(args, "ada_n_threshold", 3),
                                iid_trials=getattr(args, "ada_iid_trials", False),
                                rollout_to_terminal=getattr(
                                    args, "ada_rollout_to_terminal", False),
                                max_rollout_steps=getattr(
                                    args, "max_steps", None) or 100)
        elif name == "cem_fir":
            out[name] = BNNCEM(bnn, dyn, grid,
                               gamma=args.cem_plan_gamma or GAMMA,
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
                               surprise_tau=args.cem_surprise_tau,
                               n_candidates=args.cem_candidates,
                               # k_models default raised 10->30 for cem_fir ONLY
                               # (2026-09-13): more posterior transition-matrix
                               # draws per CVaR estimate.  Closes config2 p=0.4's
                               # confirmed ~0.163 gap to ada_mcts down to ~0.008
                               # (4-seed mean 0.859 vs 0.867, well within 1 SE),
                               # with config1 p=0.3 unchanged (0.650->0.650) and
                               # no regression found anywhere else swept -- a
                               # clean win, not a tradeoff.  --cem-k-models still
                               # overrides for cem_static/cem_ada/oracle_cem
                               # (unaffected, keep the original default there) and
                               # for cem_fir itself if explicitly passed.
                               k_models=(args.cem_k_models if args.cem_k_models
                                        is not None else 30),
                               n_rollouts=args.cem_n_rollouts,
                               n_cem_iters=args.cem_iters,
                               warm_blend=args.cem_warm_blend,
                               do_forget=args.do_forget,
                               n_unfrozen=args.n_unfrozen,
                               retrain_every=args.retrain_every,
                               retrain_steps=args.retrain_steps,
                               retrain_lr=args.retrain_lr,
                               retrain_min_conf=args.retrain_min_conf,
                               seed=getattr(args, "seed", 0))
        elif name == "cem_static":
            out[name] = BNNCEM(bnn, dyn, grid,
                               gamma=args.cem_plan_gamma or GAMMA,
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
                               surprise_tau=args.cem_surprise_tau,
                               n_candidates=args.cem_candidates,
                               k_models=args.cem_k_models,
                               n_rollouts=args.cem_n_rollouts,
                               n_cem_iters=args.cem_iters,
                               warm_blend=args.cem_warm_blend)
        elif name == "cem_ada":
            out[name] = CEMADA(bnn, dyn, grid,
                               gamma=args.cem_plan_gamma or GAMMA,
                               change_step=change_step,
                               n_threshold=args.cem_ada_threshold,
                               alpha_min=args.cem_alpha_min,
                               horizon=args.cem_horizon,
                               n_candidates=args.cem_candidates,
                               k_models=args.cem_k_models,
                               n_rollouts=args.cem_n_rollouts,
                               n_cem_iters=args.cem_iters)
        elif name == "oracle_cem":
            # Same planner/params as cem_static; _worker loads a checkpoint
            # pretrained DIRECTLY on this phase's true p (see _worker), so no
            # adaptation loop is needed -- the model already matches the
            # post-change env from t=0 (oracle upper bound for the CEM family,
            # the CEM analogue of oracle_rats).
            out[name] = BNNCEM(bnn, dyn, grid,
                               gamma=args.cem_plan_gamma or GAMMA,
                               change_step=change_step,
                               adaptive=False,
                               horizon=args.cem_horizon,
                               alpha_min=args.cem_alpha_min,
                               n_confident=args.cem_n_confident,
                               cvar_alpha=args.cem_cvar_alpha,
                               n_candidates=args.cem_candidates,
                               k_models=args.cem_k_models,
                               n_rollouts=args.cem_n_rollouts,
                               n_cem_iters=args.cem_iters)
            out[name].name = "oracle_cem"
            # BUG FIX (2026-09-08): CVaRCEMAgent.adaptive_alpha is hardwired True
            # (CEM_ADAPTIVE_ALPHA) inside BNNCEM.__init__ regardless of `adaptive`,
            # but the two state vars that DRIVE its confidence gate
            # (self._agent.n_since_change, self._agent.surprise_bar) are only ever
            # updated inside BNNCEM.act()'s `if self.adaptive and ...:` block --
            # which never runs for oracle_cem (adaptive=False).  So n_since_change
            # stays at its reset() value of 0 forever -> conf_data=0 forever ->
            # alpha = alpha_min (the MOST risk-averse worst-case CVaR tail) for
            # the entire episode, even though the oracle already has the TRUE
            # post-change model and has no regime uncertainty left to resolve.
            # This silently made oracle_cem plan under needless permanent worst-
            # case pessimism instead of the risk-neutral cvar_alpha it was given
            # (args.cem_cvar_alpha, default 1.0) -- plausible cause of the
            # oracle sometimes losing to cem_fir at low p (09-06 report).  Force
            # the fixed tail here; cem_fir/cem_static are untouched.
            out[name]._agent.adaptive_alpha = False
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
    seed = int(cfg.get("seed") or 0)
    torch.manual_seed(seed)   # posterior draws reproducible across workers/runs
    if cfg.get("conc_prior") is not None:
        # Must patch the ATTRIBUTE on the actual module object: dirichlet_model.py
        # resolves CONC_PRIOR as a plain module-global at every _forward_alpha
        # call, so this needs to happen before any such call in this process
        # (fresh spawn per task -> once here is enough for the whole task).
        import bnn.dirichlet_model as _dm
        _dm.CONC_PRIOR = float(cfg["conc_prior"])
    grid = get_grid(grid_name)
    bnn, dyn = make_dirichlet_bnn(grid.n_states, grid.n_actions, grid=grid)
    if method_name == "oracle_cem":
        # Oracle: pretrained DIRECTLY on this phase's true p, not on orig_p --
        # "stationary" (p=orig_p, no change) still matches orig_p exactly.
        p_ckpt = cfg["orig_p"] if phase_label == "stationary" \
            else float(phase_label.split("=")[1])
    else:
        p_ckpt = cfg["orig_p"]
    ckpt = _HERE / "data" / grid_name / ckpt_name(grid, p_ckpt)
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
        # --seed shifts the per-trial env RNG into a disjoint block (trials
        # stay 0..trials-1 within a block) so a second --seed is an
        # independent replicate, not a re-run of the same env draws.
        G, goal, rewards = run_episode(grid, method, p_schedule,
                                       seed * 100_000 + trial, max_steps)
        Gs.append(G)
        goals.append(goal)
        if trial == 0:
            trial0 = rewards
    return dict(phase=phase_label, method=method_name, Gs=Gs, goals=goals,
                trial0=trial0)


def main():
    global ORIG_P
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="cliffwalking")
    ap.add_argument("--orig-p", type=float, default=None,
                    help="override ORIG_P (the pretraining p / stationary-phase "
                         "p / non-oracle checkpoint to load); default: module "
                         "ORIG_P=1.0.  Needs a matching checkpoint from "
                         "pretrain_gridworld.py --p <orig-p> --balance-terminal.")
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
    ap.add_argument("--ada-n-threshold", type=int, default=3,
                    help="ada_mcts: post-change samples before DPAS leaves its "
                         "forced worst-case phase (paper N_threshold = 50)")
    ap.add_argument("--ada-iid-trials", action="store_true",
                    help="ada_mcts: restart the post-change state every trial "
                         "(paper's independent-run protocol) instead of once "
                         "per phase")
    ap.add_argument("--ada-rollout-to-terminal", action="store_true",
                    help="ada_mcts: roll out until a terminal cell with NO "
                         "gamma^dist leaf bootstrap (upstream adamcts.py's "
                         "behaviour).  Required whenever holes pay a negative "
                         "reward: with the bootstrap, 'wander forever' is "
                         "worth ~0.9987 vs 1.0 for reaching the goal, so any "
                         "risk of a -1 hole makes hovering optimal (measured: "
                         "goal rate 0.000 on cliffwalking_aayl)")
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
    # SFIR ablation knobs (2026-09-10, default flipped 2026-09-11) -- cem_fir
    # only.  Default = "our method" = SFIR (main_cl_2.tex: Surprise-Forget-
    # Inflate-RETRAIN): forget on, gradient retrain of the adapter HEAD on.
    # Pass --n-unfrozen 0 to get the older SFI ablation (no retrain).
    ap.add_argument("--no-forget", dest="do_forget", action="store_false",
                    default=True,
                    help="cem_fir: remove the 'F' -- retain never decays")
    ap.add_argument("--n-unfrozen", type=int, default=1,
                    help="cem_fir gradient retrain 'R': # of top DIRECTION-path "
                         "layers to retrain on the post-change buffer.  0=off "
                         "(SFI, the pre-09-11 default), 1=Dirichlet head only "
                         "('our method' = SFIR, default), 2=head+top trunk, "
                         "3=head+whole trunk (retrain-all ablation).")
    ap.add_argument("--retrain-every", type=int, default=3,
                    help="cem_fir: gradient-retrain cadence in post-change steps")
    ap.add_argument("--retrain-steps", type=int, default=5,
                    help="cem_fir: Adam steps per retrain call")
    ap.add_argument("--retrain-lr", type=float, default=1e-2,
                    help="cem_fir: Adam lr for the retrain (bare NLL, no KL)")
    ap.add_argument("--retrain-min-conf", type=float, default=0.0,
                    help="cem_fir: withhold retrain until the SAME confidence "
                         "signal driving plan_retain/the CVaR tail reaches "
                         "this level (0.0 = always fire on cadence, old "
                         "behaviour).  Lets forget do the early work under a "
                         "severe change instead of retraining on a few "
                         "still-noisy post-change samples.")
    ap.add_argument("--seed", type=int, default=0,
                    help="cem_fir only: seeds torch (posterior draws) and the "
                         "CVaRCEMAgent's own rng (candidate sampling); also "
                         "offsets the per-trial env RNG into a disjoint block "
                         "(seed*100000 + trial) so a different --seed is an "
                         "independent replicate, not a re-run of the same "
                         "draws. Other methods are unaffected (still seed 0).")
    # CEM planner tuning (2026-08-13, bridge short-episode tuning)
    ap.add_argument("--cem-horizon", type=int, default=RATS_DEPTH,
                    help="CEM planning horizon (bridge's full trip is ~2-4 steps)")
    ap.add_argument("--cem-alpha-min", type=float, default=CEM_ALPHA_MIN)
    ap.add_argument("--cem-n-confident", type=int, default=CEM_N_CONFIDENT)
    ap.add_argument("--cem-cvar-alpha", type=float, default=CEM_CVAR_ALPHA,
                    help="fixed CVaR tail when adaptive_alpha is off")
    ap.add_argument("--cem-surprise-tau", type=float, default=CEM_SURPRISE_TAU,
                    help="surprise sensitivity for the confidence gate (planner "
                         "default 2.0; p=1.0 pretraining makes surprise spike to "
                         "hundreds, so a larger tau relaxes the gate faster)")
    # CEM search-budget knobs (2026-08-14 tuning: cand512 lifts cliff cem_fir
    # p=0.4 from 0.38 -> 0.88 by giving the CVaR estimate more samples)
    ap.add_argument("--cem-candidates", type=int, default=None)
    ap.add_argument("--cem-k-models", type=int, default=None)
    ap.add_argument("--cem-n-rollouts", type=int, default=None)
    ap.add_argument("--cem-iters", type=int, default=None)
    ap.add_argument("--cem-plan-gamma", type=float, default=None,
                    help="CEM-internal planning discount (default: GAMMA=0.99).  "
                         "At gamma=0.99 with horizon 6 the heuristic bootstrap "
                         "gamma^dist(~0.93-0.99) nearly matches actually reaching "
                         "the goal (gamma^4=0.96), which makes CEM dither next "
                         "to the goal; a smaller plan gamma sharpens the "
                         "goal-vs-hover contrast.  Evaluation gamma unchanged.")
    ap.add_argument("--cem-ada-threshold", type=int, default=ADA_CEM_N_THRESHOLD,
                    help="cem_ada: post-change samples before switching from "
                         "worst-case (alpha_min) to risk-neutral (alpha=1.0) "
                         "-- the DPAS _training_started threshold, ported from "
                         "ADA-MCTS (default 3, matches planning/ada_mcts.py)")
    ap.add_argument("--cem-warm-blend", type=float, default=0.0,
                    help="fraction of the previous step's refit plan (shifted "
                         "forward) blended into the CEM init; 0 = re-seed from "
                         "the model policy every step.  Commitment against "
                         "per-step re-planning noise (dithering).")
    ap.add_argument("--conc-prior", type=float, default=None,
                    help="override bnn.dirichlet_model.CONC_PRIOR (default 0.1 "
                         "as of 2026-09-11, was briefly 1.0 -- see the module "
                         "comment), "
                         "the symmetric Dirichlet concentration that retain=0 "
                         "(fully forgotten / fully plan_retain-deflated) decays "
                         "toward.  Lower -> more extreme/high-variance posterior "
                         "draws when uncertain (spikier CVaR tail); higher -> "
                         "draws stay closer to uniform (milder tail).  Does NOT "
                         "need repretraining: pretrain_alpha0 uses it only as a "
                         "negligible +K*conc_prior regularizer on top of real "
                         "counts (~100s), so old checkpoints stay valid -- only "
                         "the retain-blend formula (evaluated fresh each forward "
                         "pass) sees the new value.")
    args = ap.parse_args()
    if args.max_depth is not None:
        args.rats_depth = args.dp_depth = args.max_depth
    if args.orig_p is not None:
        ORIG_P = args.orig_p

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
    log(f"  rewards: goal +1 | hole {grid.hole_reward:+g} "
        f"({'teleport-to-start' if grid.cliff_to_start else 'terminal'}) | "
        f"step {grid.step_penalty:+g}")
    log(f"  ADA-MCTS: {args.m_simulations} simulations/action (paper value) | "
        f"n_threshold={args.ada_n_threshold} iid_trials={args.ada_iid_trials} "
        f"rollout_to_terminal={args.ada_rollout_to_terminal}")
    if args.conc_prior is not None:
        log(f"  CONC_PRIOR override: {args.conc_prior} (module default 0.1)")
    if not args.do_forget or args.n_unfrozen > 0:
        log(f"  cem_fir SFIR ablation: do_forget={args.do_forget} "
            f"n_unfrozen={args.n_unfrozen} retrain_every={args.retrain_every} "
            f"retrain_steps={args.retrain_steps} retrain_lr={args.retrain_lr}")
    log("=" * 90)

    # ── build (phase, method) tasks ─────────────────────────────────────────
    cfg = dict(orig_p=ORIG_P, rats_depth=args.rats_depth, dp_depth=args.dp_depth,
               k_forget=args.k_forget, use_counts=args.use_counts,
               persist_counts=args.persist_counts, count_w=args.count_w,
               drift_reset=args.drift_reset,
               m_simulations=args.m_simulations, dpas_gamma=args.dpas_gamma,
               ada_n_threshold=args.ada_n_threshold,
               ada_iid_trials=args.ada_iid_trials,
               ada_rollout_to_terminal=args.ada_rollout_to_terminal,
               max_steps=max_steps,
               cem_horizon=args.cem_horizon, cem_alpha_min=args.cem_alpha_min,
               cem_n_confident=args.cem_n_confident,
               cem_cvar_alpha=args.cem_cvar_alpha,
               cem_surprise_tau=args.cem_surprise_tau,
               cem_candidates=args.cem_candidates,
               cem_k_models=args.cem_k_models,
               cem_n_rollouts=args.cem_n_rollouts,
               cem_iters=args.cem_iters,
               cem_plan_gamma=args.cem_plan_gamma,
               cem_warm_blend=args.cem_warm_blend,
               cem_ada_threshold=args.cem_ada_threshold,
               conc_prior=args.conc_prior,
               do_forget=args.do_forget, n_unfrozen=args.n_unfrozen,
               retrain_every=args.retrain_every,
               retrain_steps=args.retrain_steps, retrain_lr=args.retrain_lr,
               retrain_min_conf=args.retrain_min_conf, seed=args.seed)
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
            se = np.std(Gs, ddof=1) / np.sqrt(len(Gs)) if len(Gs) > 1 else 0.0
            log(f"  [{phase:<8s}] {name:<18s}: return {np.mean(Gs):+.3f} "
                f"(se {se:.3f}) | goal rate {np.mean(goals):.3f}")
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
    log(f"\nSUMMARY: discounted return (gamma={GAMMA}, holes = 0)")
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
