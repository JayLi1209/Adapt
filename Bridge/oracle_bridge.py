"""TRUE oracle: the optimal policy for the ACTUAL post-change dynamics.

This is the ceiling the adaptive agent is chasing.  Unlike
run_frozenlake_optimal_policy.py -- which solves the DETERMINISTIC (pre-change)
MDP and then suffers in the slippery env, i.e. the "stale policy" floor -- this
solves value iteration on the TRUE p'=0.7 transition kernel.  It knows the new
slip exactly and never has to learn it, so no adaptive method can beat it except
by luck.

Both references matter:
  stale  (--model-p 1.0)  : optimal for the OLD dynamics, run in the new env
                            == what you get if you never adapt at all
  oracle (--model-p 0.7)  : optimal for the NEW dynamics
                            == what perfect instantaneous adaptation would give

Scoring matches the rest of the suite: gamma=1, holes scored 0 (so mean return
== goal rate), no truncation, env seeds seed+1000+trial.

    python oracle_bridge.py --grid bridge --env-p 0.7 --trials 100
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from grids import get_grid
from env.gridworlds import build_env


def build_true_T(grid, p):
    """Exact (S,A,S) kernel for intended-prob p under this grid's slip mode."""
    S, A = grid.n_states, grid.n_actions
    T = np.zeros((S, A, S))
    dist = grid.slip_dist(p)                       # (K,)
    flat = grid.flat_desc
    for s in range(S):
        if flat[s] in "GH":                        # absorbing
            T[s, :, s] = 1.0
            continue
        for a in range(A):
            for k, d in enumerate(grid.dir_actions(a)):
                T[s, a, grid.move(s, d)] += dist[k]
    return T


def value_iteration(grid, T, gamma=1.0, hole_reward=-1.0, iters=5000, tol=1e-12):
    """Optimal policy for kernel T.  Reward on ARRIVAL: goal +1, hole `hole_reward`.

    hole=-1 is the DECISION-side map (a hole is actively bad, as in the planner);
    returns are still scored hole=0 elsewhere.
    """
    S, A = grid.n_states, grid.n_actions
    flat = grid.flat_desc
    r = np.zeros(S)
    term = np.zeros(S, bool)
    for i, ch in enumerate(flat):
        if ch == "G":
            r[i] = 1.0; term[i] = True
        elif ch == "H":
            r[i] = hole_reward; term[i] = True
    cont = (~term).astype(float)
    exp_r = (T * r[None, None, :]).sum(-1)                 # (S,A)
    V = np.zeros(S)
    for _ in range(iters):
        Q = exp_r + gamma * (T * (V * cont)[None, None, :]).sum(-1)
        Vn = Q.max(1); Vn[term] = 0.0
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn; break
        V = Vn
    return Q.argmax(1), V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="bridge",
                    choices=("bridge", "bridge_h17", "bridge_h18", "bridge_h19", "frozenlake",
                             "cliffwalking"))
    ap.add_argument("--env-p", type=float, required=True,
                    help="TRUE intended-prob of the env being evaluated in")
    ap.add_argument("--model-p", type=float, default=None,
                    help="dynamics the POLICY is solved for; default = env-p "
                         "(the true oracle).  Pass 1.0 for the stale-policy floor.")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--out-dir", default="results/oracle")
    args = ap.parse_args()

    model_p = args.env_p if args.model_p is None else args.model_p
    grid = get_grid(args.grid)
    env, _ = build_env(args.grid, [(0, args.env_p)], max_episode_steps=args.max_steps)

    T_policy = build_true_T(grid, model_p)
    pi, V = value_iteration(grid, T_policy)

    kind = "ORACLE (knows new dynamics)" if abs(model_p - args.env_p) < 1e-9 else "STALE policy"
    tag = f"oracle_{args.grid}_env{args.env_p:g}_pol{model_p:g}".replace(".", "p")
    print(f"[{tag}] {kind} | grid={grid.name} | policy solved on p={model_p:g} "
          f"={[round(x,3) for x in grid.slip_dist(model_p)]} | env p={args.env_p:g} "
          f"={[round(x,3) for x in grid.slip_dist(args.env_p)]} | V*(start s={grid.start_state})={V[grid.start_state]:.4f}",
          flush=True)

    records = []
    for trial in range(args.trials):
        obs, _ = env.reset(seed=args.seed + 1000 + trial)
        s = int(np.argmax(obs))
        terminated = truncated = False
        steps, total, reward = 0, 0.0, 0.0
        while not (terminated or truncated) and steps < args.max_steps:
            a = int(pi[s])
            act = np.zeros(grid.n_actions, dtype=np.float32); act[a] = 1.0
            obs, reward, terminated, truncated, _ = env.step(act)
            s = int(np.argmax(obs))
            total += max(0.0, float(reward))          # hole = 0 scoring
            steps += 1
        outcome = ("goal" if reward > 0 else "hole") if terminated else "truncated"
        records.append(dict(ret=total, steps=steps, outcome=outcome))

    rets = np.array([r["ret"] for r in records])
    goals = np.array([r["outcome"] == "goal" for r in records], float)
    sem = float(rets.std(ddof=1) / np.sqrt(len(rets))) if len(rets) > 1 else 0.0
    out = dict(kind=kind, grid=grid.name, env_p=args.env_p, policy_p=model_p,
               gamma=1.0, trials=args.trials, seed=args.seed, hole_scoring=0,
               v_star_start=float(V[grid.start_state]),
               mean_return=float(rets.mean()), sem_return=sem,
               goal_rate=float(goals.mean()),
               hole_rate=float(np.mean([r["outcome"] == "hole" for r in records])),
               trunc_rate=float(np.mean([r["outcome"] == "truncated" for r in records])),
               mean_steps=float(np.mean([r["steps"] for r in records])),
               records=records)
    od = pathlib.Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
    (od / f"{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[{tag}] saved: goal {goals.mean():.3f} +/- {sem:.3f} | hole {out['hole_rate']:.3f} "
          f"| steps {out['mean_steps']:.1f}", flush=True)


if __name__ == "__main__":
    main()
