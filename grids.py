"""Grid-world geometry specs shared by the env builders and the Dirichlet model.

The Dirichlet world model predicts a categorical over K *directions* (not over
raw cells) and then scatters that onto the cell simplex.  Which K directions,
and where each one lands, is the only thing that differs between the grid
worlds we run -- so it is captured here once and injected into both sides.

  FrozenLake 4x4   : K=3, dirs [a, a-1, a+1]           (intended, perp-, perp+)
                     actions LEFT/DOWN/RIGHT/UP = 0/1/2/3
  CliffWalking 4x12: K=3, dirs [a, a+1, a-1]           (intended, perp+, perp-)
                     actions UP/RIGHT/DOWN/LEFT = 0/1/2/3
                     Same slip structure as FrozenLake (Luo et al.): intended p,
                     each perpendicular (1-p)/2, NO opposite direction.  ns_gym's
                     NSCliffWalkingWrapper carries a 4th (opposite) outcome slot,
                     which build_env pads with probability 0.

`cliff_to_start` reproduces CliffWalking's teleport-on-cliff: when the wrapper
is built with terminal_cliff=False, stepping into the cliff returns the agent to
the start cell instead of terminating.  With terminal_cliff=True (our default)
the cliff behaves like a FrozenLake hole and the landing cell is the cliff cell.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class GridSpec:
    name: str
    nrow: int
    ncol: int
    n_actions: int
    k_dir: int
    # (drow, dcol) per action index
    deltas: Tuple[Tuple[int, int], ...]
    # offsets added to the action index (mod n_actions) to form the K directions
    dir_offsets: Tuple[int, ...]
    # How the (1-p) slip mass is distributed among the non-intended directions:
    #   "perp"     -> split equally over the PERPENDICULAR dirs; opposite gets 0
    #                 (FrozenLake & CliffWalking, per Luo et al.: (1-p)/2 each)
    #   "opposite" -> all of (1-p) goes to the OPPOSITE direction
    #                 (Bridge, per Lecarpentier & Rachelson 2019)
    slip_mode: str = "perp"
    # char map, row strings (S start, G goal, H hole/cliff, F free)
    desc: Tuple[str, ...] = ()
    # CliffWalking-style teleport: cliff cells send the agent back to start
    cliff_to_start: bool = False

    @property
    def n_states(self) -> int:
        return self.nrow * self.ncol

    @property
    def start_state(self) -> int:
        for i, ch in enumerate(self.flat_desc):
            if ch == "S":
                return i
        return 0

    @property
    def flat_desc(self) -> List[str]:
        return [c for row in self.desc for c in row]

    def desc_bytes(self) -> np.ndarray:
        """(nrow, ncol) array of bytes -- the `desc` format BNNModelPlanner wants."""
        return np.asarray([list(r) for r in self.desc], dtype="c")

    # ── geometry ─────────────────────────────────────────────────────────────
    def move(self, s: int, a: int) -> int:
        """Deterministic (no-slip) next cell for action a, clipped at walls."""
        r, c = divmod(int(s), self.ncol)
        dr, dc = self.deltas[int(a)]
        r = min(max(r + dr, 0), self.nrow - 1)
        c = min(max(c + dc, 0), self.ncol - 1)
        s2 = r * self.ncol + c
        if self.cliff_to_start and self.flat_desc and self.flat_desc[s2] == "H":
            return self.start_state
        return s2

    def dir_actions(self, a: int) -> List[int]:
        """The K action-directions that action a can actually realize."""
        return [(int(a) + off) % self.n_actions for off in self.dir_offsets]

    # ── which of the K directions are intended / perpendicular / opposite ──────
    def _dir_kinds(self):
        na = self.n_actions
        intended, opposite, perp = [], [], []
        for i, off in enumerate(self.dir_offsets):
            m = off % na
            if m == 0:
                intended.append(i)
            elif m == na // 2:
                opposite.append(i)
            else:
                perp.append(i)
        return intended, perp, opposite

    def slip_dist(self, p: float) -> List[float]:
        """The K-vector transition distribution for intended-prob p, following
        this grid's slip_mode.

          perp     : [p, (1-p)/#perp on each perp dir, 0 on opposite]
          opposite : [p, 0 on perps, (1-p) on the opposite dir]
        """
        p = float(np.clip(p, 0.0, 1.0))
        intended, perp, opposite = self._dir_kinds()
        dist = [0.0] * self.k_dir
        for i in intended:
            dist[i] = p / max(1, len(intended))
        if self.slip_mode == "perp":
            share = (1.0 - p) / max(1, len(perp))
            for i in perp:
                dist[i] = share
        elif self.slip_mode == "opposite":
            share = (1.0 - p) / max(1, len(opposite))
            for i in opposite:
                dist[i] = share
        else:
            raise ValueError(f"unknown slip_mode {self.slip_mode!r}")
        return dist

    def build_dir_cells(self) -> torch.Tensor:
        """(n_states, n_actions, K) long tensor: cell each direction lands in."""
        out = np.zeros((self.n_states, self.n_actions, self.k_dir), dtype=np.int64)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                out[s, a] = [self.move(s, d) for d in self.dir_actions(a)]
        return torch.from_numpy(out)

    def direction_of(self, s: int, a: int, s2: int) -> int:
        """Realized direction index in [0,K) or -1 if no direction explains it."""
        for d_idx, d in enumerate(self.dir_actions(a)):
            if self.move(s, d) == int(s2):
                return d_idx
        return -1


# ── the registry ──────────────────────────────────────────────────────────────
FROZENLAKE_4x4 = GridSpec(
    name="frozenlake",
    nrow=4, ncol=4, n_actions=4, k_dir=3,
    # LEFT, DOWN, RIGHT, UP
    deltas=((0, -1), (1, 0), (0, 1), (-1, 0)),
    dir_offsets=(0, -1, 1),
    desc=("SFFF", "FHFH", "FFFH", "HFFG"),
)

CLIFFWALKING_4x12 = GridSpec(
    name="cliffwalking",
    nrow=4, ncol=12, n_actions=4, k_dir=3,
    # UP, RIGHT, DOWN, LEFT  (gymnasium CliffWalking order)
    deltas=((-1, 0), (0, 1), (1, 0), (0, -1)),
    dir_offsets=(0, 1, -1),          # intended, perp+, perp-  (same as FrozenLake)
    desc=("FFFFFFFFFFFF",
          "FFFFFFFFFFFF",
          "FFFFFFFFFFFF",
          "SHHHHHHHHHHG"),
    # ns_gym sets next_state = start_state for every cliff landing, regardless of
    # terminal_cliff (terminal_cliff only controls the `terminated` flag).
    cliff_to_start=True,
)

# Bridge (Lecarpentier & Rachelson 2019): intended prob p, each PERPENDICULAR
# direction (1-p)/2 -- the same slip structure as FrozenLake/CliffWalking.
# 5x8 map, goals on both ends of the middle row, holes above/below the bridge.
#
# Perpendicular (not opposite) slip is what makes this env the bridge it is meant
# to be.  The original ADA-MCTS nsbridge_v0.py puts the slip mass on the cells
# ABOVE and BELOW (its `wsat` fills s_up / s_dw), so a slip on the narrow right
# arm drops the agent into a hole.  With opposite-direction slip a horizontal
# action can only ever move horizontally, so the agent is confined to the middle
# row, can never fall in, and goal rate is pinned at 1.0 by construction.
#
# Geometry that results: from the start (20) the RIGHT arm reaches goal 23 in 3
# steps but cells 21/22 have holes directly above AND below -- any slip is fatal.
# The LEFT arm needs 4 steps to goal 16 but cells 17-19 have free cells above and
# below, so a slip there is survivable.  Short-and-risky vs long-and-safe.
BRIDGE_5x8 = GridSpec(
    name="bridge",
    nrow=5, ncol=8, n_actions=4, k_dir=3,
    # LEFT, DOWN, RIGHT, UP  (same action order as FrozenLake)
    deltas=((0, -1), (1, 0), (0, 1), (-1, 0)),
    dir_offsets=(0, -1, 1),             # intended, perp-, perp+ (as FrozenLake)
    slip_mode="perp",
    desc=("HHHHHHHH",
          "FFFFFHHH",
          "GFFFSFFG",
          "FFFFFHHH",
          "HHHHHHHH"),
)

# Bridge variant with cell 17 (row 2, col 1) turned into a HOLE.  17 sits on the
# LEFT (long, otherwise-safe) arm, one step short of the left goal 16, so this
# variant puts a hazard on BOTH routes: the right arm is lethal on a slip, and
# the left arm now has a hole to squeeze past.  Reported alongside the default
# map so the effect of that one cell is visible.
BRIDGE_5x8_H17 = GridSpec(
    name="bridge_h17",
    nrow=5, ncol=8, n_actions=4, k_dir=3,
    deltas=((0, -1), (1, 0), (0, 1), (-1, 0)),
    dir_offsets=(0, -1, 1),
    slip_mode="perp",
    desc=("HHHHHHHH",
          "FFFFFHHH",
          "GHFFSFFG",      # col 1 of the middle row is H, not F
          "FFFFFHHH",
          "HHHHHHHH"),
)

# Bridge variant with cell 18 (row 2, col 2) turned into a HOLE, cell 17 left FREE.
# 18 sits in the MIDDLE of the left arm (two steps from the left goal 16), so
# unlike bridge_h17 the hazard does not sit adjacent to the goal: the agent must
# route AROUND it via row 1 or row 3, both of which are free on the left half.
BRIDGE_5x8_H18 = GridSpec(
    name="bridge_h18",
    nrow=5, ncol=8, n_actions=4, k_dir=3,
    deltas=((0, -1), (1, 0), (0, 1), (-1, 0)),
    dir_offsets=(0, -1, 1),
    slip_mode="perp",
    desc=("HHHHHHHH",
          "FFFFFHHH",
          "GFHFSFFG",      # col 2 of the middle row is H; col 1 stays F
          "FFFFFHHH",
          "HHHHHHHH"),
)

# Bridge variant with cell 19 (row 2, col 3) turned into a HOLE.  19 sits on the
# LEFT (long, otherwise-safe) arm, DIRECTLY ADJACENT to the start 20, so it does
# not merely add a hazard to the left route -- it BLOCKS it at row 2.
#
# This is the one single-cell placement that makes the stale p=1.0 policy and the
# post-change optimal policy choose DIFFERENT ROUTES:
#   * Under p=1.0 the agent can dodge 19 deterministically: DOWN to 28, LEFT
#     along row 3, UP to goal 16.  Row 3 sits directly above row 4, which is all
#     holes, so that detour has ZERO clearance -- safe only if slip is impossible.
#   * Under any slip the row-3 detour is lethal, and the optimal policy switches
#     to RIGHT along the short arm instead.
# So a non-adapting agent commits to a route that slip has made deadly, which is
# exactly the CliffWalking failure mode the default bridge map lacks.
BRIDGE_5x8_H19 = GridSpec(
    name="bridge_h19",
    nrow=5, ncol=8, n_actions=4, k_dir=3,
    deltas=((0, -1), (1, 0), (0, 1), (-1, 0)),
    dir_offsets=(0, -1, 1),
    slip_mode="perp",
    desc=("HHHHHHHH",
          "FFFFFHHH",
          "GFFHSFFG",      # col 3 of the middle row is H
          "FFFFFHHH",
          "HHHHHHHH"),
)

REGISTRY = {g.name: g for g in (FROZENLAKE_4x4, CLIFFWALKING_4x12, BRIDGE_5x8,
                                BRIDGE_5x8_H17, BRIDGE_5x8_H18, BRIDGE_5x8_H19)}


def get_grid(name: str) -> GridSpec:
    if name not in REGISTRY:
        raise ValueError(f"unknown grid {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]
