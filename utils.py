"""Small shared helpers: one-hot action encoding and the FrozenLake potential.

Both extracted verbatim from bnn_fl_cem.py (the only two symbols that module
contributed to the risk-averse path).
"""
from collections import deque

import numpy as np
import torch


def to_one_hot_action(action: np.ndarray) -> np.ndarray:
    """argmax -> one-hot, matching what the env executes and the BNN expects."""
    one_hot = np.zeros_like(action, dtype=np.float32)
    one_hot[int(np.argmax(action))] = 1.0
    return one_hot


def make_fl_potential(desc: np.ndarray):
    """Per-cell potential Phi = -(grid distance to goal); holes = -BIG.

    BFS over the traversable grid (holes are walls).  The planner uses
    gamma^dist(s, goal) as the leaf value V(s_H); this returns -dist.
    """
    flat = [c.decode() for c in desc.flatten()]
    nrow, ncol = desc.shape
    n = nrow * ncol
    goals = [i for i, ch in enumerate(flat) if ch == "G"]
    holes = {i for i, ch in enumerate(flat) if ch == "H"}
    # Multi-source BFS: distance to the NEAREST goal (handles the 2-goal bridge;
    # identical to single-goal BFS when there is only one G).
    dist = {g: 0 for g in goals}
    q = deque(goals)
    while q:
        c = q.popleft()
        r, col = divmod(c, ncol)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, col + dc
            if 0 <= nr < nrow and 0 <= nc < ncol:
                nb = nr * ncol + nc
                if nb in holes or nb in dist:
                    continue
                dist[nb] = dist[c] + 1
                q.append(nb)
    big = (max(dist.values()) if dist else 1) + 1
    phi = torch.tensor(
        [-float(dist.get(i, big)) for i in range(n)], dtype=torch.float32
    )
    return phi
