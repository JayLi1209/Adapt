"""Merge the ablation shards and emit the paper table's Pendulum row.

Combines the three new ablation arms (retrain_full / no_retrain /
no_forget_no_retrain) with the arms already stored under head1l_* and nf1l_*,
then prints every column of the paper table with mean +/- 1 s.e.m. and a 95% CI,
plus seed-paired contrasts against SFIR (head_1l).
"""
import json, glob, pathlib
import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).parent
# paper column -> arm key
COLS = [("SFIR", "head_1l"), ("Retrain full", "retrain_full"),
        ("No retrain", "no_retrain"),
        ("No forget +no retrain", "no_forget_no_retrain"),
        ("No adapt", "no_adapt"), ("No forget", "head_1l_noforget"),
        ("Oracle", "oracle")]
SCEN = [("m = 1 -> 4", "results/head1l_mass", "results/nf1l_mass", "results/abl_mass"),
        ("g = 10 -> 15", "results/head1l_grav15", "results/nf1l_grav15", "results/abl_grav")]
NBOOT = 10000


def gather(base, nf, abl):
    rows = json.load(open(HERE / base / "merged.json"))
    for d in (nf, abl):
        for f in sorted(glob.glob(str(HERE / d / "shard_*.json"))):
            for k, v in json.load(open(f)).items():
                rows.setdefault(k, []).extend(v)
    return {k: {t["seed"]: t for t in v} for k, v in rows.items()}


def boot(x, rng):
    idx = rng.integers(0, len(x), size=(NBOOT, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def main():
    out = {}
    for label, base, nf, abl in SCEN:
        arms = gather(base, nf, abl)
        rng = np.random.default_rng(0)
        keys = [k for _, k in COLS if k in arms]
        common = sorted(set.intersection(*[set(arms[k]) for k in keys]))
        n = len(common)
        tcrit = stats.t.ppf(0.975, n - 1)
        ref = np.array([arms["head_1l"][s]["ret"] for s in common])
        print(f"\n=== Pendulum {label}   (n={n} seeds shared across arms) ===")
        row = {}
        for name, k in COLS:
            if k not in arms:
                print(f"  {name:<24} —   (not run)")
                continue
            got = len(arms[k])
            x = np.array([arms[k][s]["ret"] for s in common])
            bal = sum(arms[k][s]["balanced"] is not None for s in common)
            sem = x.std(ddof=1) / np.sqrt(n)
            lo, hi = boot(x, rng)
            e = {"n": n, "stored": got, "mean": float(x.mean()), "sem": float(sem),
                 "sd": float(x.std(ddof=1)), "bal": bal,
                 "ci95": [float(x.mean() - tcrit * sem), float(x.mean() + tcrit * sem)],
                 "boot95": [float(lo), float(hi)]}
            line = (f"  {name:<24}{e['mean']:>9.1f} ± {sem:>5.1f}"
                    f"   95% CI [{e['ci95'][0]:>8.1f},{e['ci95'][1]:>8.1f}]"
                    f"   bal {100*bal//n:>3}%   n={got}")
            if k != "head_1l":
                d = ref - x
                dsem = d.std(ddof=1) / np.sqrt(n)
                _t, p = stats.ttest_rel(ref, x)
                e["vs"] = {"diff": float(d.mean()), "p": float(p),
                           "wins": int((d > 0).sum()),
                           "ci95": [float(d.mean() - tcrit * dsem),
                                    float(d.mean() + tcrit * dsem)]}
                line += (f"   ΔSFIR {d.mean():+7.1f}"
                         f" [{e['vs']['ci95'][0]:+6.1f},{e['vs']['ci95'][1]:+6.1f}]"
                         f" p={p:.1e} w={e['vs']['wins']}/{n}")
            print(line)
            row[name] = e
        out[label] = row
    (HERE / "table_row.json").write_text(json.dumps(out))

    print("\n\n--- paper table format (mean ± 1 s.e.m.) ---\n")
    w = 23
    print(f"{'Setting':<14}" + "".join(f"{c:>{w}}" for c, _ in COLS))
    for label in out:
        cells = []
        for c, _ in COLS:
            e = out[label].get(c)
            cells.append(f"{e['mean']:.1f} ± {e['sem']:.1f}" if e else "—")
        print(f"{label:<14}" + "".join(f"{v:>{w}}" for v in cells))


if __name__ == "__main__":
    main()
