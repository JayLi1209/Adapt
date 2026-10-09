"""ORACLE baseline: the policy that is optimal for the TRUE post-change dynamics.

This is the upper reference point for the adaptation ablations in
sweep_unfrozen_layers.py.  Where run_frozenlake_optimal_policy.py solves value
iteration on the DETERMINISTIC ([1,0,0]) model and then suffers in the slippery
env -- i.e. the "stale policy, never adapted" lower reference -- this script
solves VI on the ACTUAL post-change transition kernel [0.7, 0.15, 0.15].  It is
what a learner would converge to if adaptation were instantaneous and perfect,
so no online method run in this env should beat it (up to sampling noise).

The oracle knows p exactly; it does no learning, carries no model uncertainty,
and never plans with an ensemble.  It is a fixed, exactly-optimal stochastic-MDP
policy executed greedily.

TRANSITION MODEL (must match env/gridworlds + grids.FROZENLAKE_4x4):
  For intended action a the K=3 directions are [a, a-1, a+1] (dir_offsets
  (0,-1,+1), slip_mode "perp"), landing in cells _move(s, .), with probability
  [p, (1-p)/2, (1-p)/2].  Cells are clipped at the grid border, so a slip into a
  wall self-transitions -- exactly as the env does.

DEFAULT FrozenLake setting (CLAUDE.md):
  * change [1,0,0] -> [0.7,0.15,0.15] at ts=0 (SCHEDULE=[(0,0.7)])
  * NO truncation (--max-steps raises both the TimeLimit and the per-trial cap)
  * NO discount when scoring (gamma=1); returns scored hole=0 (== goal rate)
  * 100 trials + SEM, env seeds seed+1000 .. (matches the sweep/planner runs)

DECISION-side reward map is the planner's (hole=-1, goal=+1) so the oracle is
risk-aware about holes; SCORING is hole=0 (== goal rate) as everywhere else.
With gamma=1 and hole=-1 the VI fixed point is well defined (holes are
absorbing with negative value, so the agent strictly prefers reaching the goal).

    python run_frozenlake_oracle.py --trials 100 --max-steps 1000 \
        --out-dir results/fl_ablation
"""
import argparse
import json
import pathlib

import numpy as np

from env import build_scheduled_env
from env.frozenlake import MODIFIED_REWARDS
from grids import FROZENLAKE_4x4
from bnn.dirichlet_model import N_STATES, N_ACTIONS, _move

SCHEDULE = [(0, 0.7)]
CHANGE_P = 0.7            # the TRUE intended-direction probability after the change
GAMMA = 1.0               # NO discount (experiment convention)
HOLES = {5, 7, 11, 12}
GOAL = 15
ACTS = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}


def terminal(s):
    return s in HOLES or s == GOAL


def reward_of(s):
    if s in HOLES:
        return MODIFIED_REWARDS["H"]
    if s == GOAL:
        return MODIFIED_REWARDS["G"]
    return MODIFIED_REWARDS["F"]


def build_kernel(p=CHANGE_P):
    """P[s, a, s'] for the TRUE slippery dynamics, from the shared GridSpec.

    Uses FROZENLAKE_4x4.slip_dist(p) == [p, (1-p)/2, (1-p)/2] and the grid's
    dir_offsets, so this kernel is the same one the env samples from.
    """
    dist = FROZENLAKE_4x4.slip_dist(p)                 # over the K=3 directions
    offsets = FROZENLAKE_4x4.dir_offsets               # (0, -1, +1)
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        if terminal(s):
            P[s, :, s] = 1.0                           # absorbing
            continue
        for a in range(N_ACTIONS):
            for k, off in enumerate(offsets):
                a_eff = (a + off) % N_ACTIONS
                P[s, a, _move(s, a_eff)] += dist[k]
    return P


def value_iteration(P, tol=1e-14, max_iters=100000):
    """Exact VI on the TRUE stochastic MDP (gamma=1, hole=-1, goal=+1).

    Reward is a function of the LANDING state, so Q(s,a) = sum_s' P[s,a,s'] *
    (r(s') + gamma * V(s') * [s' not terminal]).
    """
    r = np.array([reward_of(s) for s in range(N_STATES)])
    nonterm = np.array([0.0 if terminal(s) else 1.0 for s in range(N_STATES)])
    V = np.zeros(N_STATES)
    for _ in range(max_iters):
        Q = P @ (r + GAMMA * V * nonterm)              # (S, A)
        V_new = Q.max(axis=1)
        for s in range(N_STATES):
            if terminal(s):
                V_new[s] = 0.0
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < tol:
            break
    Q = P @ (r + GAMMA * V * nonterm)
    return V, Q


def optimal_actions(Q):
    """Per non-terminal state, the set of Q-optimal actions (ties kept)."""
    out = {}
    for s in range(N_STATES):
        if terminal(s):
            continue
        q = Q[s]
        out[s] = [a for a in range(N_ACTIONS) if q[a] >= q.max() - 1e-9]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tie", choices=["first", "random"], default="first",
                    help="tie-break among equally-optimal actions")
    ap.add_argument("--change-p", type=float, default=CHANGE_P,
                    help="TRUE post-change intended-direction prob p' (change at ts=0)")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--out-dir", default="results/fl_ablation")
    ap.add_argument("--log-steps", action="store_true",
                    help="also write a PER-STEP JSONL trace (<tag>_steps.jsonl)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    change_p = args.change_p
    schedule = [(0, change_p)]
    tag = f"oracle_{args.tie}" + ("" if change_p == CHANGE_P
                                  else "_p" + f"{change_p:g}".replace(".", "p"))

    P = build_kernel(change_p)
    V, Q = value_iteration(P)
    opt = optimal_actions(Q)
    print(f"[{tag}] ORACLE policy: VI on the TRUE p={change_p} kernel "
          f"(gamma={GAMMA}, decision map hole=-1/goal=+1):")
    for s in sorted(opt):
        acts = "/".join(ACTS[a] for a in opt[s])
        print(f"  s={s:2d} V*={V[s]:+.4f} -> {acts}")

    step_path = out_dir / f"{tag}_steps.jsonl" if args.log_steps else None
    step_log = open(step_path, "w") if step_path else None

    env = build_scheduled_env(schedule, max_episode_steps=args.max_steps)
    rng = np.random.default_rng(args.seed)
    seed_base = args.seed + 1000
    records = []
    try:
        for trial in range(args.trials):
            obs, _ = env.reset(seed=seed_base + trial)
            terminated = truncated = False
            steps, total_return, reward = 0, 0.0, 0.0
            while not (terminated or truncated) and steps < args.max_steps:
                s = int(np.argmax(obs))
                cand = opt[s]
                a = cand[0] if args.tie == "first" else int(rng.choice(cand))
                action = np.zeros(N_ACTIONS, dtype=np.float32)
                action[a] = 1.0
                obs, reward, terminated, truncated, _ = env.step(action)
                s2 = int(np.argmax(obs))
                total_return += max(0.0, float(reward))       # hole=0 scoring
                if step_log is not None:
                    step_log.write(json.dumps(dict(
                        trial=trial, t=steps, s=s, a=a, s2=s2,
                        slipped=bool(s2 != _move(s, a)),
                        reward=float(reward), terminated=bool(terminated),
                        V_s=float(V[s]), Q_sa=float(Q[s, a]),
                    )) + "\n")
                steps += 1
            outcome = ("goal" if reward > 0 else "hole") if terminated else "truncated"
            records.append(dict(ret=total_return, steps=steps, outcome=outcome))
            print(f"[{tag}]   trial {trial+1}/{args.trials}: {outcome} "
                  f"steps={steps} ret={total_return:.0f}", flush=True)
    finally:
        if step_log:
            step_log.close()

    rets = np.array([r["ret"] for r in records])
    goals = np.array([r["outcome"] == "goal" for r in records], dtype=float)
    steps_a = np.array([r["steps"] for r in records], dtype=float)

    def sem(x):
        return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0

    out = dict(
        policy=f"ORACLE: value iteration on the TRUE p={change_p} kernel",
        tie_break=args.tie, gamma=GAMMA, trials=args.trials, seed=args.seed,
        schedule=schedule, change_p=change_p, trial_len=args.max_steps,
        hole_scoring=0,
        mean_return=float(rets.mean()), sem_return=sem(rets),
        goal_rate=float(goals.mean()), sem_goal_rate=sem(goals),
        hole_rate=float(np.mean([r["outcome"] == "hole" for r in records])),
        trunc_rate=float(np.mean([r["outcome"] == "truncated" for r in records])),
        mean_steps=float(steps_a.mean()), sem_steps=sem(steps_a),
        V_star_start=float(V[0]),
        policy_table={str(s): opt[s] for s in sorted(opt)},
        records=records,
    )
    (out_dir / f"{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[{tag}] {args.trials} trials in p={change_p} env: "
          f"goal rate {out['goal_rate']:.3f} +/- {out['sem_goal_rate']:.3f} (SEM) | "
          f"hole {out['hole_rate']:.3f} trunc {out['trunc_rate']:.3f} | "
          f"steps {out['mean_steps']:.1f} +/- {out['sem_steps']:.1f} | "
          f"V*(start)={V[0]:.4f}")


if __name__ == "__main__":
    main()
