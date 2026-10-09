"""Merge results/planners/<cell>/<planner>/shard_*.json (run_planners.sh) and print
SURF x planner returns next to the stored SURF+CEM column of the paper table.

SURF+CEM is read from the table's own runs (head_1l arm), seeds 1000-1099, so
every planner row is PAIRED with it by seed: diff = planner - CEM per seed, with
mean, SEM and a 95% t-interval.  Budget columns report the imagined transitions
per real step each planner actually spent (cap = CEM's 1.6M).
"""
import json, math, pathlib
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).parent / "results"
CELLS = [("mass4", "m: 1 -> 4", "head1l_mass"), ("mass0p4", "m: 1 -> 0.4", "easy_mass0p4"),
         ("grav15", "g: 10 -> 15", "head1l_grav15"), ("grav5", "g: 10 -> 5", "easy_grav5")]
PLANNERS = ["mppi", "mcts", "ilqr"]


def load_shards(d):
    rows = []
    for f in sorted(d.glob("shard_*.json")):
        rows += json.load(open(f)).get("head_1l", [])
    return {r["seed"]: r for r in rows}


def fmt(x):
    return f"{np.mean(x):9.2f} +/- {np.std(x, ddof=1) / math.sqrt(len(x)):6.2f}" if len(x) > 1 else "-"


def main():
    out = {}
    print(f"{'cell':<13}{'planner':<7}{'n':>4}  {'return mean +/- SEM':>22}  "
          f"{'paired diff vs CEM [95% CI]':>34}{'p':>9}  {'budget used':>12}{'s/act':>7}{'balanced':>9}")
    for cell, label, cem_dir in CELLS:
        cem = {r["seed"]: r for r in json.load(open(ROOT / cem_dir / "merged.json"))["head_1l"]}
        c = np.array([r["ret"] for r in cem.values()])
        print(f"{label:<13}{'cem':<7}{len(c):>4}  {fmt(c):>22}  {'(stored table run)':>34}{'':>9}  "
              f"{'1600000':>12}{'':>7}{np.mean([r['balanced'] is not None for r in cem.values()]):>9.2f}")
        out[cell] = {"cem": {"ret": c.tolist()}}
        for p in PLANNERS:
            rows = load_shards(ROOT / "planners" / cell / p)
            if not rows:
                continue
            seeds = sorted(rows)
            x = np.array([rows[s]["ret"] for s in seeds])
            paired = [s for s in seeds if s in cem]
            d = np.array([rows[s]["ret"] - cem[s]["ret"] for s in paired])
            if len(d) > 1:
                se = d.std(ddof=1) / math.sqrt(len(d))
                h = stats.t.ppf(0.975, len(d) - 1) * se
                pv = stats.ttest_rel([rows[s]["ret"] for s in paired],
                                     [cem[s]["ret"] for s in paired]).pvalue
                dtxt, ptxt = f"{d.mean():+8.2f} [{d.mean()-h:+8.2f},{d.mean()+h:+8.2f}]", f"{pv:9.1e}"
            else:
                dtxt, ptxt = "-", ""
            used = np.mean([rows[s]["sim_steps_mean"] for s in seeds])
            print(f"{'':<13}{p:<7}{len(x):>4}  {fmt(x):>22}  {dtxt:>34}{ptxt}  {used:>12.0f}"
                  f"{np.mean([rows[s]['act_sec_mean'] for s in seeds]):>7.2f}"
                  f"{np.mean([rows[s]['balanced'] is not None for s in seeds]):>9.2f}")
            out[cell][p] = {"seeds": seeds, "ret": x.tolist(), "diff_vs_cem": d.tolist(),
                            "sim_steps_mean": used}
            json.dump({str(s): rows[s] for s in seeds},
                      open(ROOT / "planners" / cell / p / "merged.json", "w"), indent=1)
    json.dump(out, open(ROOT / "planners" / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
