"""Error bounds for the paper table's Reacher rows, from per-trial returns.

The stored runs are sharded 5 x 20 seeds; each row's `trials` list holds the
per-seed records, so the same bounds computed for Pendulum are available here:
95% t CI, bootstrap CI, sd/IQR, and seed-paired contrasts against SFIR.
"""
import json, glob, pathlib
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).parent / "results"
HERE = ROOT / "traces100"
ABL = ROOT / "abl100"
# table row -> (shard glob, SFIR arm name in that run)
ROWS = [("k_s = 0.5", "wind_ks0.5_s*.json", "nl1_forget_qv"),
        ("k_s = 5",   "wind_ks5_s*.json",   "nl1_forget_qv"),
        ("tau = 90",  "rot90_s*.json",      "nl1_forget_qv"),
        ("tau = 150", "rot150_s*.json",     "nl1_forget_qv")]
# table column -> arm key
COLS = [("SFIR", None), ("Retrain full", "nl1_forget_qv_body"),
        ("No retrain", "nl1_forget_noelbo"),
        ("No forget +no retrain", "nl1_noforget_noelbo"),
        ("No adapt", "no_adapt"),
        ("No forget", "nl1_noforget"), ("Oracle", "oracle")]
NBOOT = 10000


def gather(pattern):
    """Merge shards into {arm: {seed: ret}}, across both result dirs.

    traces100/ holds SFIR + the original context arms; abl100/ holds the three
    paper-table ablation columns, run with identical flags and seed grid.
    """
    out = {}
    for f in sorted(glob.glob(str(HERE / pattern))
                    + glob.glob(str(ABL / pattern))):
        r = json.load(open(f))
        for row in r["rows"]:
            d = out.setdefault(row["arm"], {})
            for t in row["trials"]:
                d[t["seed"]] = t["ret"]
    return out


def boot(x, rng, nboot=NBOOT):
    idx = rng.integers(0, len(x), size=(nboot, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def main():
    all_out = {}
    for name, pat, sfir in ROWS:
        arms = gather(pat)
        if not arms:
            print(f"[skip] {name}: no shards at {pat}")
            continue
        rng = np.random.default_rng(0)
        # common seed grid across the arms this row reports
        keys = [a for _, a in COLS if a] + [sfir]
        present = [a for a in keys if a in arms]
        common = sorted(set.intersection(*[set(arms[a]) for a in present]))
        base = np.array([arms[sfir][s] for s in common]) if sfir in arms else None
        n = len(common)
        tcrit = stats.t.ppf(0.975, n - 1)
        print(f"\n=== Reacher  {name}   (n={n} seeds, shards {pat}) ===")
        row = {}
        for label, key in COLS:
            a = sfir if key is None else key
            if a not in arms:
                print(f"  {label:<12} —")
                continue
            x = np.array([arms[a][s] for s in common])
            sem = x.std(ddof=1) / np.sqrt(n)
            lo, hi = boot(x, rng)
            e = {"n": n, "mean": float(x.mean()), "sem": float(sem),
                 "sd": float(x.std(ddof=1)),
                 "ci95": [float(x.mean() - tcrit * sem), float(x.mean() + tcrit * sem)],
                 "boot95": [float(lo), float(hi)],
                 "iqr": [float(np.percentile(x, 25)), float(np.median(x)),
                         float(np.percentile(x, 75))]}
            if key is not None and base is not None:
                diff = base - x
                dsem = diff.std(ddof=1) / np.sqrt(n)
                t_st, p = stats.ttest_rel(base, x)
                e["vs"] = {"diff": float(diff.mean()),
                           "ci95": [float(diff.mean() - tcrit * dsem),
                                    float(diff.mean() + tcrit * dsem)],
                           "p": float(p), "wins": int((diff > 0).sum())}
            row[label] = e
            v = f"{e['mean']:.1f} ± {sem:.1f}"
            print(f"  {label:<12}{v:>16}   95% CI [{e['ci95'][0]:7.1f},{e['ci95'][1]:7.1f}]"
                  f"   boot [{lo:7.1f},{hi:7.1f}]   sd {e['sd']:5.1f}"
                  + (f"   vs SFIR {e['vs']['diff']:+7.1f} p={e['vs']['p']:.1e}"
                     f" wins {e['vs']['wins']}/{n}" if "vs" in e else ""))
        all_out[name] = row
    (pathlib.Path(__file__).parent / "table_bounds.json").write_text(json.dumps(all_out))


if __name__ == "__main__":
    main()
