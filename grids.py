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

# Bridge (Lecarpentier & Rachelson 2019): intended prob p, OPPOSITE prob 1-p.
# 5x8 map, goals on both ends of the middle row, holes above/below the bridge.
BRIDGE_5x8 = GridSpec(
    name="bridge",
    nrow=5, ncol=8, n_actions=4, k_dir=2,
    # LEFT, DOWN, RIGHT, UP  (same action order as FrozenLake)
    deltas=((0, -1), (1, 0), (0, 1), (-1, 0)),
    dir_offsets=(0, 2),                 # intended, opposite
    slip_mode="opposite",
    desc=("HHHHHHHH",
          "FFFFFHHH",
          "GFFFSFFG",
          "FFFFFHHH",
          "HHHHHHHH"),
)

REGISTRY = {g.name: g for g in (FROZENLAKE_4x4, CLIFFWALKING_4x12, BRIDGE_5x8)}


def get_grid(name: str) -> GridSpec:
    if name not in REGISTRY:
        raise ValueError(f"unknown grid {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]
