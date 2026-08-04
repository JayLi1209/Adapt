"""Pretrain a Dirichlet world model on the ORIGINAL dynamics of a grid.

Following Luo et al. ("Act As You Learn"), the ORIGINAL environment has
intended-prob p=0.7; non-stationarity then moves p among {0.4,...,1}.  So the
model's prior belief -- what we pretrain here -- is the p-slip transition
(default p=0.7), NOT the deterministic one.  We fit it by SAMPLING each row's
realized direction from grid.slip_dist(p): with enough samples per (s,a) the
Dirichlet head's predictive mean converges to slip_dist(p).

(The shipped FrozenLake checkpoint bnn_dirichlet_k3.pth is a SEPARATE,
deterministic model and is left untouched; pass --p 1.0 to reproduce it.)

Per CLAUDE.md, transitions CLOSE TO THE GOAL are prioritised: each (s,a) row is
sampled with weight proportional to  goal_weight ** (-BFS distance from s to the
goal), so states near the goal are seen far more often.  Terminal states are
excluded (nothing to predict from them).

    conda activate nsgym
    python pretrain_gridworld.py --grid cliffwalking --p 0.7 --epochs 400
"""
import argparse
import collections
import pathlib
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

from config import device, SAVE_DIR
from grids import get_grid
from bnn import make_dirichlet_bnn
from bnn.dirichlet_model import ckpt_name


def bfs_dist_to_goal(grid):
    """BFS distance from every cell to the nearest goal over traversable cells."""
    flat = grid.flat_desc
    goals = [i for i, c in enumerate(flat) if c == "G"]
    dist = {g: 0 for g in goals}
    q = collections.deque(goals)
    while q:
        s = q.popleft()
        for a in range(grid.n_actions):
            # walk BACKWARDS: any cell that can reach s in one deterministic move
            for s_prev in range(grid.n_states):
                if flat[s_prev] in "HG":
                    continue
                if grid.move(s_prev, a) == s and s_prev not in dist:
                    dist[s_prev] = dist[s] + 1
                    q.append(s_prev)
    far = grid.n_states * 2
    return np.array([dist.get(i, far) for i in range(grid.n_states)], dtype=np.float64)


def build_dataset(grid, goal_weight, n_rows, p):
    """Sample (s,a) rows weighted toward the goal; realized direction ~ slip_dist(p).

    p=1.0 reduces to the deterministic dataset (always the intended direction).
    """
    flat = grid.flat_desc
    usable = [s for s in range(grid.n_states) if flat[s] not in "HG"]
    d = bfs_dist_to_goal(grid)
    w = np.array([goal_weight ** (-d[s]) for s in usable], dtype=np.float64)
    w /= w.sum()
    slip = np.array(grid.slip_dist(p), dtype=np.float64)   # (K,)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(usable), size=n_rows, p=w)
    acts = rng.integers(0, grid.n_actions, size=n_rows)
    dirs = rng.choice(grid.k_dir, size=n_rows, p=slip)     # realized slip direction

    X = np.zeros((n_rows, grid.n_states + grid.n_actions), dtype=np.float32)
    Y = np.zeros((n_rows, grid.n_states + 1), dtype=np.float32)
    for i, (u, a, k) in enumerate(zip(idx, acts, dirs)):
        s = usable[u]
        d_action = grid.dir_actions(a)[k]    # which of the K directions realized
        s2 = grid.move(s, d_action)
        X[i, s] = 1.0
        X[i, grid.n_states + a] = 1.0
        Y[i, s2] = 1.0
        Y[i, grid.n_states] = 1.0 if flat[s2] == "G" else (
            -1.0 if flat[s2] == "H" else 0.0)
    return torch.from_numpy(X), torch.from_numpy(Y), d, usable, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="cliffwalking")
    ap.add_argument("--p", type=float, default=0.7,
                    help="intended-prob of the ORIGINAL env to fit (paper: 0.7). "
                         "1.0 gives the deterministic model.")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--goal-weight", type=float, default=1.35,
                    help="w(s) = goal_weight^(-dist(s,goal)); >1 upweights goal-near")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    grid = get_grid(args.grid)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else SAVE_DIR.parent / grid.name
    out_dir.mkdir(parents=True, exist_ok=True)

    X, Y, dist, usable, w = build_dataset(grid, args.goal_weight, args.rows, args.p)
    X, Y = X.to(device), Y.to(device)
    target = grid.slip_dist(args.p)
    print(f"[{grid.name}] grid {grid.nrow}x{grid.ncol} n_states={grid.n_states} "
          f"K={grid.k_dir} | ORIGINAL p={args.p} -> target dist {[round(x,3) for x in target]} "
          f"| {len(usable)} usable cells | {args.rows} rows")
    order = np.argsort(dist[usable])
    print("  goal-weighting (nearest 5 / farthest 5 usable cells):")
    for tag, sel in (("near", order[:5]), ("far ", order[-5:])):
        print(f"    {tag}: " + ", ".join(
            f"s={usable[i]}(d={dist[usable[i]]:.0f},w={w[i]:.4f})" for i in sel))

    bnn, dyn = make_dirichlet_bnn(grid.n_states, grid.n_actions, grid=grid)
    bnn.num_train_points = args.rows
    bnn.learn_reward = True
    opt = torch.optim.Adam(bnn.parameters(), lr=args.lr)

    n = X.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            loss, meta = bnn.loss(X[b], Y[b])
            loss.backward()
            opt.step()
            tot += float(loss.item())
        if ep % 50 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:4d}: loss={tot / max(1, n // args.batch):.4f} "
                  f"nll={meta['nll']:.4f} kl={meta['kl']:.1f}")

    fn = ckpt_name(grid, args.p)
    bnn.save(out_dir, filename=fn)
    print(f"saved {out_dir / fn}")

    # ── verify: predictive p_dir should match the ORIGINAL target slip_dist(p) ──
    from bnn import epistemic_dirichlet
    bnn.use_counts = False; bnn.retain.fill_(1.0)
    pdirs = []
    for s in usable:
        o = np.zeros(grid.n_states, dtype=np.float32); o[s] = 1.0
        for a in range(grid.n_actions):
            ac = np.zeros(grid.n_actions, dtype=np.float32); ac[a] = 1.0
            pdirs.append(epistemic_dirichlet(dyn, bnn, o, ac, n_draws=30)["p_dir"])
    pdirs = np.array(pdirs)                              # (n_usable*A, K)
    tgt = np.array(target)
    mae = float(np.abs(pdirs - tgt[None, :]).mean())
    mean_pdir = pdirs.mean(0)
    print(f"VERIFY p_dir: mean={[round(x,3) for x in mean_pdir]} target={[round(x,3) for x in tgt]} "
          f"| mean|abs err|={mae:.4f} max|abs err|={float(np.abs(pdirs-tgt[None,:]).max()):.4f}")
    print("VERDICT:", "GOOD" if mae < 0.05 else "SUSPECT")


if __name__ == "__main__":
    main()
