"""Assemble the FrozenLake adaptation-ablation table.

Reads the JSONs written by sweep_unfrozen_layers.py (+ run_frozenlake_oracle.py)
into results/fl_ablation and reports, per condition, the three headline metrics:

  goal rate       fraction of trials ending on the goal (+/- SEM)
  steps           mean episode length (+/- SEM), reported BOTH over all trials
                  and over successful (goal) trials only -- a method that dies
                  in a hole early has a short episode, so the all-trials mean
                  conflates "fast" with "fails fast"
  expected return hole=0 / goal=+1, no discount => numerically == goal rate

All conditions share the 0.83 reference setup: CVaR+CEM planner at fixed
cvar_alpha=0.0, gamma=1.0, [1,0,0]->[0.7,0.15,0.15] at ts=0, 100 trials, and
--max-steps 1000 so nothing truncates.

    python summarize_fl_ablation.py
"""
import json
import pathlib

import numpy as np

RESULT_DIR = pathlib.Path("results/fl_ablation")
REFERENCE = pathlib.Path("results/unfrozen_a0/U1.json")   # the 0.83 run

# (label, file, what the condition actually switches off)
ROWS = [
    ("Full method (0.83 ref)", REFERENCE,
     "forget + head retrain + counts"),
    ("No forget", RESULT_DIR / "U1_nf.json",
     "forget OFF; head retrain + counts ON"),
    ("No retrain", RESULT_DIR / "U0.json",
     "retrain OFF; forget + counts ON"),
    ("No forget + no retrain", RESULT_DIR / "U0_nf.json",
     "forget + retrain OFF; counts ON"),
    ("Retrain full network", RESULT_DIR / "U3.json",
     "head + BOTH trunk layers trained"),
    ("No adapt", RESULT_DIR / "U0_nf_nc.json",
     "forget + retrain + counts ALL OFF"),
    ("Oracle", RESULT_DIR / "oracle_first.json",
     "VI on the TRUE p=0.7 kernel"),
]


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main():
    print(f"{'condition':<24} {'goal rate':>16} {'steps (all)':>16} "
          f"{'steps (goal)':>16} {'E[return]':>16} {'trunc':>7}")
    print("-" * 100)
    for label, path, _note in ROWS:
        if not path.exists():
            print(f"{label:<24} {'-- missing --':>16}   ({path})")
            continue
        d = json.load(open(path))
        recs = d["records"]
        rets = [r["ret"] for r in recs]
        steps_all = [r["steps"] for r in recs]
        steps_goal = [r["steps"] for r in recs if r["outcome"] == "goal"]
        gr = float(np.mean([r["outcome"] == "goal" for r in recs]))
        gr_sem = sem([float(r["outcome"] == "goal") for r in recs])
        sg = (f"{np.mean(steps_goal):.1f}+/-{sem(steps_goal):.1f}"
              if steps_goal else "n/a")
        print(f"{label:<24} {gr:.3f}+/-{gr_sem:.3f}".ljust(24 + 17)
              + f"{np.mean(steps_all):.1f}+/-{sem(steps_all):.1f}".rjust(16)
              + f"{sg}".rjust(17)
              + f"{np.mean(rets):.3f}+/-{sem(rets):.3f}".rjust(17)
              + f"{d.get('trunc_rate', 0.0):.2f}".rjust(8))

    print("\nnotes:")
    for label, path, note in ROWS:
        print(f"  {label:<24} {note}")
    print("\nreward: hole=0, goal=+1, no discount => E[return] == goal rate.")
    print("steps (goal) excludes hole trials, which end early by failing.")


if __name__ == "__main__":
    main()
