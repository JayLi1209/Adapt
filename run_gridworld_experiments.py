"""Gridworld RATS experiments: stationary verification + non-stationary change.

Mirrors the ADA-MCTS paper's (Luo et al. 2024) Cliff-Walking / NS-Bridge tables,
plus our adaptive variant.  The model is pretrained on the ORIGINAL env (p=0.7);
"introducing the new environment" changes p to {0.4, 0.5, 0.6, 0.8, 0.9, 1.0}.

Methods:
  DP-NSMDP (oracle)      omniscient -- plans with the true schedule
  DP-snapshot (oracle)   re-plans each step with the true CURRENT model
  Oracle RATS (P_k)      RATS with the true CURRENT model
  BNN RATS static        RATS with the pretrained BNN (P-hat_{k-1}), no adaptation
  BNN RATS adaptive      RATS + surprise/forget/online-counts on the same BNN
  ADA-MCTS               the baseline (notified of the change)

gamma = 0.99 (per user request; CLAUDE.md's no-discount default is overridden).
Per-step rewards are logged; both the raw discounted return (holes -1) and the
goal rate (holes 0, the paper's convention) are reported.

Run:
    python run_gridworld_experiments.py --grid cliffwalking --trials 30
    python run_gridworld_experiments.py --grid bridge --trials 100
"""

import argparse
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
)
from planning.ada_mcts import ADAMCTSAgent, DPAS_GAMMA

_HERE = pathlib.Path(__file__).parent

GAMMA = 0.99            # discount (user request)
ORIG_P = 0.7            # "original" env the model is pretrained on
CHANGE_PS = [0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
K_FORGET = 5            # forget every K post-change steps
M_SIMULATIONS = 2000    # ADA-MCTS baseline iterations per action
N_POSTERIOR = 10        # BNN posterior draws for surprise


def cell_reward(grid, s):
    return {"G": 1.0, "H": -1.0}.get(grid.flat_desc[int(s)], 0.0)


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
    name = "oracle_rats"

    def __init__(self, grid, gamma=GAMMA, max_depth=6):
        self.grid = grid
        self.gamma = gamma
        self.max_depth = max_depth
        self._agent = RATS(None, gamma=gamma, max_depth=max_depth, grid=grid)

    def reset(self):
        pass

    def observe(self, s, a, s2):
        pass

    def act(self, s, t, p):
        self._agent.model = GridSnapshot(self.grid, p)
        return self._agent.act(s, t0=t)


class DPSnapshot:
    name = "dp_snapshot"

    def __init__(self, grid, gamma=GAMMA, max_depth=6):
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

    def __init__(self, grid, dist_by_time, gamma=GAMMA, max_depth=6):
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

    def __init__(self, bnn, dyn, grid, gamma=GAMMA, max_depth=6,
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


class ADAMCTS:
    name = "ada_mcts"

    def __init__(self, bnn, dyn, grid, gamma=GAMMA, m_simulations=M_SIMULATIONS,
                 change_step=0, rng=None, dpas_gamma=DPAS_GAMMA):
        self.bnn = bnn
        self.grid = grid
        self.change_step = change_step
        self.rng = rng or np.random.default_rng(0)
        self._agent = ADAMCTSAgent(dyn, bnn, grid.desc_bytes(), device,
                                   n_actions=grid.n_actions, gamma=gamma,
                                   rng=self.rng,
                                   m_simulations=m_simulations,
                                   dpas_gamma=dpas_gamma)
        self._notified = False

    def reset(self):
        self.bnn.use_counts = True
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
        if t == self.change_step and not self._notified:
            self._agent.notify_change()
            self._notified = True
        obs = np.zeros(self.grid.n_states, dtype=np.float32)
        obs[int(s)] = 1.0
        return int(np.argmax(self._agent.act(obs)))


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
                                max_depth=args.max_depth)
        elif name == "dp_snapshot":
            out[name] = DPSnapshot(grid, gamma=GAMMA, max_depth=args.max_depth)
        elif name == "oracle_rats":
            out[name] = OracleRATS(grid, gamma=GAMMA, max_depth=args.max_depth)
        elif name == "bnn_rats_static":
            out[name] = BNNRATS(bnn, dyn, grid, gamma=GAMMA,
                                max_depth=args.max_depth, adaptive=False)
        elif name == "bnn_rats_adaptive":
            out[name] = BNNRATS(bnn, dyn, grid, gamma=GAMMA,
                                max_depth=args.max_depth, adaptive=True,
                                change_step=change_step,
                                k_forget=args.k_forget,
                                use_counts=not args.no_counts,
                                persist_counts=args.persist_counts,
                                count_w=args.count_w,
                                drift_reset=args.drift_reset)
        elif name == "ada_mcts":
            out[name] = ADAMCTS(bnn, dyn, grid, gamma=GAMMA,
                                m_simulations=args.m_simulations,
                                change_step=change_step,
                                dpas_gamma=args.dpas_gamma)
        else:
            raise ValueError(f"unknown method {name}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="cliffwalking")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--change-p", type=float, nargs="*", default=None)
    ap.add_argument("--change-step", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--m-simulations", type=int, default=M_SIMULATIONS)
    ap.add_argument("--dpas-gamma", type=float, default=DPAS_GAMMA)
    ap.add_argument("--k-forget", type=int, default=K_FORGET)
    ap.add_argument("--count-w", type=float, default=1.0)
    ap.add_argument("--persist-counts", action="store_true")
    ap.add_argument("--no-counts", action="store_true")
    ap.add_argument("--drift-reset", action="store_true")
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()

    grid = get_grid(args.grid)
    max_steps = args.max_steps or (10 if grid.name == "bridge" else 100)
    change_ps = args.change_p if args.change_p else CHANGE_PS
    methods = args.methods or ["dp_nsmdp", "dp_snapshot", "oracle_rats",
                               "bnn_rats_static", "bnn_rats_adaptive",
                               "ada_mcts"]
    out_dir = _HERE / "data" / grid.name
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
    log(f"  gamma={GAMMA} | max_depth={args.max_depth} | max_steps={max_steps} "
        f"| trials={args.trials}")
    log(f"  methods: {methods}")
    log("=" * 90)

    # ── load the pretrained BNN ────────────────────────────────────────────
    bnn, dyn = make_dirichlet_bnn(grid.n_states, grid.n_actions, grid=grid)
    ckpt = out_dir / ckpt_name(grid, ORIG_P)
    if ckpt.exists():
        bnn.load(out_dir, filename=ckpt_name(grid, ORIG_P))
        log(f"loaded pretrained model {ckpt}")
    else:
        log(f"WARNING: {ckpt} not found -- run pretrain_gridworld.py first!")
    bnn.num_weight_groups = 1
    bnn.num_train_points = 20000

    # ── 1. STATIONARY verification (p = ORIG_P throughout) ────────────────
    log("\n" + "=" * 90)
    log(f"1. STATIONARY verification (p={ORIG_P}) -- pretrained model must "
        f"perform well")
    log("=" * 90)
    p_schedule = [(0, ORIG_P)]
    dist_by_time = {0: grid.slip_dist(ORIG_P)}
    methods_map = build_methods(args, grid, bnn, dyn, dist_by_time, methods,
                                change_step=None)
    for name, method in methods_map.items():
        Gs, goals = [], []
        for trial in range(args.trials):
            method.reset()
            G, goal, _ = run_episode(grid, method, p_schedule, trial, max_steps)
            Gs.append(G)
            goals.append(goal)
        log(f"  {name:<18s}: return {np.mean(Gs):+.3f} | goal rate "
            f"{np.mean(goals):.3f}")

    # ── 2. NON-STATIONARY: introduce the change ────────────────────────────
    summary = {}
    for p_new in change_ps:
        log("\n" + "=" * 90)
        log(f"2. NON-STATIONARY: p {ORIG_P} -> {p_new} at ts {args.change_step}")
        log("=" * 90)
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
        methods_map = build_methods(args, grid, bnn, dyn, dist_by_time, methods,
                                    change_step=args.change_step)
        for name, method in methods_map.items():
            Gs, goals = [], []
            trial0_rewards = None
            for trial in range(args.trials):
                method.reset()
                G, goal, rewards = run_episode(grid, method, p_schedule, trial,
                                               max_steps)
                Gs.append(G)
                goals.append(goal)
                if trial == 0:
                    trial0_rewards = rewards
            summary.setdefault(name, {})[p_new] = (np.mean(Gs), np.mean(goals))
            log(f"  {name:<18s}: return {np.mean(Gs):+.3f} | goal rate "
                f"{np.mean(goals):.3f}")
            log(f"      per-step rewards (trial 0): "
                f"{[round(r, 1) for r in trial0_rewards]}")

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
