"""Assemble the ns-Bridge adaptation-ablation table.

Reads the JSONs written by sweep_bridge_unfrozen.py (+ oracle_bridge.py) into
Bridge/results/bridge_ablation and reports, per change level and condition:

  goal rate       fraction of trials ending on the goal (+/- SEM)
  steps           mean episode length, BOTH over all trials and over successful
                  (goal) trials only -- a method that falls in a hole early has
                  a short episode, so the all-trials mean conflates "fast" with
                  "fails fast"
  expected return hole=0 / goal=+1, no discount => numerically == goal rate

All conditions share: ns-Bridge 5x8, deterministic prior p=1.0, CVaR+CEM at
fixed cvar_alpha=0.0, gamma=1.0, plan_gamma=1.0, retrain_steps=50, K_FORGET=1, 100 trials,
--max-steps 1000 so nothing truncates.

    python summarize_bridge_ablation.py
"""
import glob
import json
import pathlib

import numpy as np

RESULT_DIR = pathlib.Path("Bridge/results/bridge_ablation")
LEVELS = ["0.9", "0.7", "0.5"]

# (label, arm dir prefix, what the condition actually switches off)
ROWS = [
    ("Full method",            "both",                 "forget + head retrain + counts"),
    ("No forget",              "no_forget",            "forget OFF; head retrain + counts ON"),
    ("No retrain",             "no_retrain",           "retrain OFF; forget + counts ON"),
    ("No forget + no retrain", "no_forget_no_retrain", "forget + retrain OFF; counts ON"),
    ("Retrain full network",   "retrain_full",         "head + BOTH trunk layers trained"),
    ("No adapt",               "no_adapt",             "forget + retrain + counts ALL OFF"),
    ("Oracle",                 "oracle",               "VI on the TRUE post-change kernel"),
]


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def load(arm, p):
    hits = glob.glob(str(RESULT_DIR / f"{arm}_p{p}" / "*.json"))
    return json.load(open(hits[0])) if hits else None


def main():
    for p in LEVELS:
        print(f"\n=== ns-Bridge 5x8 | p: 1.0 -> {p} at ts=0 | 100 trials "
              f"| alpha=0, gamma=1, plan_gamma=1, rs=50, no truncation ===")
        print(f"{'condition':<24} {'goal rate':>16} {'steps (all)':>16} "
              f"{'steps (goal)':>16} {'E[return]':>16} {'trunc':>7}")
        print("-" * 100)
        for label, arm, _note in ROWS:
            d = load(arm, p)
            if d is None:
                print(f"{label:<24} {'-- missing --':>16}")
                continue
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

    print("\n=== goal rate summary (rows = condition, cols = p') ===")
    print(f"{'condition':<24}" + "".join(f"{p:>18}" for p in LEVELS))
    print("-" * (24 + 18 * len(LEVELS)))
    for label, arm, _note in ROWS:
        row = f"{label:<24}"
        for p in LEVELS:
            d = load(arm, p)
            if d is None:
                row += f"{'--':>18}"
                continue
            recs = d["records"]
            gr = float(np.mean([r["outcome"] == "goal" for r in recs]))
            s = sem([float(r["outcome"] == "goal") for r in recs])
            row += f"{gr:.2f}+/-{s:.3f}".rjust(18)
        print(row)

    print("\nnotes:")
    for label, _arm, note in ROWS:
        print(f"  {label:<24} {note}")
    print("\nreward: hole=0, goal=+1, no discount => E[return] == goal rate.")
    print("steps (goal) excludes hole trials, which end early by failing.")


if __name__ == "__main__":
    main()
