"""Merge the shards of results/win5_n99 and report, per planner x p':
goal rate +/- SEM, dbar at FIRST detection (mean +/- SEM over trials that
detected), median final retain, #trials hitting the 1000-step cap, mean length.

    python summarize_win5_n99.py
"""
import glob
import json

import numpy as np

ROOT = "results/win5_n99"
PS = ["0.9", "0.7", "0.5", "0.3", "0.1"]
REF = dict(zip(PS, [0.818, 0.576, 0.677, 0.677, 0.707]))
ROWS = [("cem", "CEM (γp=0.95)"), ("mppi", "MPPI (γp=0.95)"),
        ("ilqr", "iLQR/iLQG (γp=0.95)"), ("mcts", "SAFIR-MCTS Alg.2"),
        ("cem_g1", "CEM (γp=1.0)")]


def merged(row, p):
    recs = []
    for f in sorted(glob.glob(f"{ROOT}/{row}/win5_p{p}/shard*/*.json")):
        recs += json.load(open(f))["records"]
    return recs


def sem(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


tables = {k: {} for k in ("goal", "dbar", "retain", "cap")}
for row, lab in ROWS:
    for p in PS:
        r = merged(row, p)
        if not r:
            continue
        g = [x["outcome"] == "goal" for x in r]
        d = [x["first_detect_dbar"] for x in r if x["first_detect_dbar"] is not None]
        tables["goal"][row, p] = (f"{np.mean(g):.3f}±{sem(g):.3f}", len(r), np.mean(g))
        tables["dbar"][row, p] = (f"{np.mean(d):.1f}±{sem(d):.1f} ({len(d)})"
                                  if d else "never", len(r), None)
        tables["retain"][row, p] = (f"{np.median([x['final_retain'] for x in r]):.1e}",
                                    len(r), None)
        tables["cap"][row, p] = (f"{sum(x['outcome'] == 'truncated' for x in r)}"
                                 f" / {np.mean([x['steps'] for x in r]):.0f}", len(r), None)

for key, title in [("goal", "goal rate ± SEM  [n trials]"),
                   ("dbar", "dbar at first detection ± SEM (n detected)"),
                   ("retain", "median final retain"),
                   ("cap", "trials hitting 1000-step cap / mean episode length")]:
    print(f"\n== window 5: {title} ==")
    print(f"{'planner':22}" + "".join(f"{'p=' + p:>20}" for p in PS)
          + ("     mean" if key == "goal" else ""))
    if key == "goal":
        print(f"{'REFERENCE':22}" + "".join(f"{REF[p]:>20.3f}" for p in PS)
              + f"{np.mean(list(REF.values())):>9.3f}")
    for row, lab in ROWS:
        cells = [tables[key].get((row, p)) for p in PS]
        txt = "".join(f"{(c[0] + (f' [{c[1]}]' if key == 'goal' else '')) if c else '--':>20}"
                      for c in cells)
        if key == "goal" and all(cells):
            txt += f"{np.mean([c[2] for c in cells]):>9.3f}"
        print(f"{lab:22}" + txt)
