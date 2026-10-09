"""Baseline: how well does the PRETRAINED bridge model do, with no adaptation?

Runs the same CVaR+CEM planner as sweep_bridge_unfrozen.py, but with the whole
online-adaptation stack switched OFF (no surprise, no drift, no forget, no
conjugate counts, no gradient retrain).  The model is the shipped checkpoint and
it never changes during an episode.  This is the reference number the adaptive
runs are compared against.

Two flavours of "no adaptation", selected by --frozen-policy:

  default (replanning) : the model is frozen, but the PLANNER still runs every
      step -- fresh Thompson draws, fresh CEM sampling, fresh info bonus.  The
      weights never move, yet the policy is re-derived stochastically each step,
      so this is strictly stronger than "the decisions it pretrained on".
  --frozen-policy      : the policy is solved ONCE from the pretrained model
      (value iteration on the model's own mean transitions, the planner's own
      `_pretrained_policy`) and then executed as a FIXED table pi[s].  No
      replanning, no sampling, no adaptation -- purely the decisions the model
      committed to at p=1.0.  This is the true "never adapt" floor.

Two settings worth measuring:
  --env-p 1.0   the env the model was PRETRAINED on (matched).  This is the
                "pretrained env goal rate/return": how good the policy is when
                its world model is exactly right.
  --env-p 0.7   the post-change env, with the p=1.0 model and no adaptation.
                The no-adaptation floor for the 1 -> 0.7 experiment.

Returns score holes as 0 (the settled convention), so mean return == goal rate.

    python eval_bridge_pretrained.py --env-p 1.0 --trials 30
"""
import argparse
import json
import pathlib
import warnings

import os
import sys

# This script lives in Bridge/; the shared modules (config, grids, bnn, env,
# planning, utils, drift) live at the repo root, so put the root on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Bridge checkpoints live under Bridge/data/<grid>/, not the repo-root data/.
BRIDGE_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

from config import device, SAVE_DIR
from env.gridworlds import build_env
from grids import get_grid
from utils import to_one_hot_action
from bnn import make_dirichlet_bnn
from bnn.dirichlet_model import ckpt_name
from planning.cvar_cem import CVaRCEMAgent

GAMMA = 1.0          # NO discount (experiment convention)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-p", type=float, required=True,
                    help="intended-prob of the env to evaluate in")
    ap.add_argument("--model-p", type=float, default=1.0,
                    help="which pretrained checkpoint to load")
    ap.add_argument("--cvar-alpha", type=float, default=0.0)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--plan-gamma", type=float, default=1.0,
                    help="discount INSIDE the planner; the reported score is "
                         "always undiscounted (see sweep_bridge_unfrozen.py).")
    ap.add_argument("--step-cost", type=float, default=0.0,
                    help="per-step living cost charged inside the planner only.")
    ap.add_argument("--frozen-policy", action="store_true",
                    help="solve the policy ONCE from the pretrained model and "
                         "execute it as a fixed table (no per-step replanning). "
                         "This is the pure 'decisions it pretrained on' baseline.")
    ap.add_argument("--trace", action="store_true", help="write per-step JSONL")
    ap.add_argument("--out-dir", default="results/bridge_pretrained")
    ap.add_argument("--grid", default="bridge", choices=("bridge", "bridge_h17", "bridge_h18", "bridge_h19"),
                    help="bridge = 17,18 both FREE; bridge_h17 = cell 17 HOLE; "
                         "bridge_h18 = cell 18 HOLE with 17 FREE.")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"pretrained_m{args.model_p:g}_env{args.env_p:g}_a{args.cvar_alpha:g}"
           + (f"_pg{args.plan_gamma:g}" if args.plan_gamma != 1.0 else "")
           + (f"_sc{args.step_cost:g}" if args.step_cost > 0 else "")
           + ("_fp" if args.frozen_policy else "")).replace(".", "p")

    def log(*a):
        print(f"[{tag}]", *a, flush=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    env, grid = build_env(args.grid, [(0, args.env_p)], max_episode_steps=args.max_steps)
    bnn, dyn = make_dirichlet_bnn(grid.n_states, grid.n_actions, grid=grid)
    fn = ckpt_name(grid, args.model_p)
    bnn.load(pathlib.Path(BRIDGE_DATA) / grid.name, filename=fn)

    # Adaptation fully OFF: fixed model, no conjugate counts, retain pinned at 1.
    bnn.use_counts = False
    bnn.retain.fill_(1.0)
    bnn.num_weight_groups = 20

    rng = np.random.default_rng(seed=0)
    kw = {} if args.horizon is None else {"horizon": args.horizon}
    agent = CVaRCEMAgent(dyn, bnn, grid.desc_bytes(), device,
                         n_actions=grid.n_actions, rng=rng,
                         cvar_alpha=args.cvar_alpha, adaptive_alpha=False,
                         gamma=args.plan_gamma, step_cost=args.step_cost, **kw)

    ACT_NAMES = ("LEFT", "DOWN", "RIGHT", "UP")

    # --frozen-policy: derive pi ONCE from the pretrained model and freeze it.
    # _pretrained_policy() is value iteration on the model's own deterministic
    # (mean-weight) transitions -- the same routine the CEM planner uses to seed
    # its proposal -- so this is exactly "what the p=1.0 model would decide".
    fixed_pi = None
    if args.frozen_policy:
        fixed_pi, _ = agent._pretrained_policy()
        log("frozen policy pi[s] (solved once on the pretrained model): "
            + " ".join(f"{s_}:{ACT_NAMES[int(fixed_pi[s_])]}"
                       for s_ in range(grid.n_states)
                       if grid.flat_desc[s_] not in "GH"))

    log(f"grid={grid.name} {grid.nrow}x{grid.ncol} K={grid.k_dir} | NO adaptation "
        f"| ckpt={fn} (model p={args.model_p:g}) | env p={args.env_p:g} "
        f"={[round(x,3) for x in grid.slip_dist(args.env_p)]} "
        f"| alpha={args.cvar_alpha:g} score_gamma={GAMMA} "
        f"plan_gamma={args.plan_gamma:g} step_cost={args.step_cost:g} "
        f"trial_len={args.max_steps} hole=0-scored")

    trace_fh = open(out_dir / f"{tag}_trace.jsonl", "w") if args.trace else None
    records = []
    for trial in range(args.trials):
        torch.manual_seed(args.seed + 10000 + trial)
        obs, _ = env.reset(seed=args.seed + 1000 + trial)
        agent.reset()
        bnn.retain.fill_(1.0)
        terminated = truncated = False
        steps, total_return, reward = 0, 0.0, 0.0
        while not (terminated or truncated) and steps < args.max_steps:
            s_prev = int(np.argmax(obs))
            if fixed_pi is not None:
                action = np.zeros(grid.n_actions, dtype=np.float32)
                action[int(fixed_pi[s_prev])] = 1.0      # fixed pretrained decision
            else:
                action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += max(0.0, float(reward))     # hole=0 scoring
            if trace_fh is not None:
                a = int(np.argmax(action))
                trace_fh.write(json.dumps(dict(
                    trial=trial, step=steps, s=s_prev, a=a, act=ACT_NAMES[a],
                    s2=int(np.argmax(obs)), reward=float(reward),
                    terminated=bool(terminated))) + "\n")
            steps += 1
        outcome = ("goal" if reward > 0 else "hole") if terminated else "truncated"
        records.append(dict(ret=total_return, steps=steps, outcome=outcome))
        log(f"  trial {trial+1}/{args.trials}: {outcome:9s} steps={steps:4d} ret={total_return:.0f}")

    if trace_fh is not None:
        trace_fh.close()

    rets = np.array([r["ret"] for r in records], dtype=float)
    goals = np.array([r["outcome"] == "goal" for r in records], dtype=float)
    sem = float(rets.std(ddof=1) / np.sqrt(len(rets))) if len(rets) > 1 else 0.0
    out = dict(
        grid=grid.name, model_p=args.model_p, env_p=args.env_p,
        env_dist=grid.slip_dist(args.env_p), cvar_alpha=args.cvar_alpha,
        gamma=GAMMA, trials=args.trials, seed=args.seed, trial_len=args.max_steps,
        hole_scoring=0, horizon=args.horizon,
        plan_gamma=args.plan_gamma, step_cost=args.step_cost,
        frozen_policy=args.frozen_policy,
        adaptation=("OFF (fixed pretrained policy, no replanning)"
                    if args.frozen_policy else "OFF (frozen model, planner still replans)"),
        mean_return=float(rets.mean()), sem_return=sem,
        goal_rate=float(goals.mean()),
        hole_rate=float(np.mean([r["outcome"] == "hole" for r in records])),
        trunc_rate=float(np.mean([r["outcome"] == "truncated" for r in records])),
        mean_steps=float(np.mean([r["steps"] for r in records])),
        records=records,
    )
    (out_dir / f"{tag}.json").write_text(json.dumps(out, indent=2))
    log(f"saved: return {rets.mean():.3f} +/- {sem:.3f} | goal {goals.mean():.3f} "
        f"| hole {out['hole_rate']:.3f} | trunc {out['trunc_rate']:.3f} "
        f"| steps {out['mean_steps']:.1f}")


if __name__ == "__main__":
    main()
