#!/usr/bin/env python3
"""Turn the --save-traces JSONs into cumulative-return curves for plotting.

Emits one JSON of {cell: {arm: {mean: [...], sem: [...], final: float}}},
where mean/sem are over seeds at each timestep of the CUMULATIVE return
sum(rew[0..t]) -- the quantity the pendulum figures plot.

Shards of the same cell are pooled by seed before any statistic is taken.
"""
import json, glob, os, re, collections
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "traces")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "trace_curves.json")

CELLS = {
    "wind_ks0.5": "Wind k_s = 0.5",
    "wind_ks15":  "Wind k_s = 15",
    "rot150":     "Rotation 150 deg",
    "rot30":      "Rotation 30 deg",
}


def main():
    # cell -> arm -> seed -> cumulative curve
    acc = collections.defaultdict(lambda: collections.defaultdict(dict))
    for p in sorted(glob.glob(os.path.join(RES, "*.json"))):
        base = os.path.basename(p)
        m = re.match(r"(.+?)_s\d+\.json$", base)
        if not m:
            continue
        cell = m.group(1)
        with open(p) as f:
            d = json.load(f)
        for r in d["rows"]:
            for t in r["trials"]:
                if "rew_t" not in t:
                    continue
                acc[cell][r["arm"]][t["seed"]] = np.cumsum(
                    np.asarray(t["rew_t"], dtype=float))

    out = {}
    for cell, arms in acc.items():
        out[cell] = {"label": CELLS.get(cell, cell), "arms": {}}
        for arm, per_seed in arms.items():
            seeds = sorted(per_seed)
            M = np.stack([per_seed[s] for s in seeds])      # (n_seeds, T)
            mean = M.mean(0)
            sem = M.std(0, ddof=1) / np.sqrt(len(seeds)) if len(seeds) > 1 \
                  else np.zeros_like(mean)
            out[cell]["arms"][arm] = {
                "n": len(seeds),
                "mean": [round(float(x), 4) for x in mean],
                "sem": [round(float(x), 4) for x in sem],
                "final": round(float(mean[-1]), 3),
            }
        print(f"{cell:14s} {len(arms)} arms, "
              f"n={min(len(v) for v in arms.values())}..{max(len(v) for v in arms.values())} seeds, "
              f"T={len(next(iter(next(iter(arms.values())).values())))}")
    with open(OUT, "w") as f:
        json.dump(out, f)
    print("wrote", OUT, f"({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
