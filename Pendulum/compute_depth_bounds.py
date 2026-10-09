"""Full error bounds for the Pendulum depth x forget figure.

The report shipped means with a +/-1 s.e.m. band and nothing else. This adds,
for every arm in both scenarios:

  * 95% CI on the mean return (t, df = n-1) and the per-step CI band
  * bootstrap percentile CI on the mean (10k resamples, seed-matched)
  * spread of the trial distribution: sd, IQR, min/max
  * Wilson 95% interval on the balance rate (a proportion, so s.e.m. is wrong)
  * seed-paired contrasts vs head_1l: mean diff, 95% CI, paired-t p, win rate
    with its own Wilson interval, and Cohen's d_z

Writes bounds.json (consumed by the report) and prints a text table.
"""
import json, glob, pathlib
import numpy as np
from scipy import stats

ARMS = ["head_lin", "head_1l", "head", "forget_elbo", "head_noforget",
        "head_1l_noforget", "no_adapt", "oracle"]
SCEN = [("mass", "results/head1l_mass", "results/nf1l_mass"),
        ("grav", "results/head1l_grav15", "results/nf1l_grav15")]
HERE = pathlib.Path(__file__).parent
NBOOT = 10000


def load(base, new):
    rows = json.load(open(HERE / base / "merged.json"))
    for f in sorted(glob.glob(str(HERE / new / "shard_*.json"))):
        for k, v in json.load(open(f)).items():
            rows.setdefault(k, []).extend(v)
    # sort every arm by seed so trials line up across arms for pairing
    return {k: sorted(v, key=lambda t: t["seed"]) for k, v in rows.items()}


def wilson(k, n, z=1.959963985):
    """Wilson score interval -- correct for rates at 0%, 100%, and small n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def boot_ci(x, rng, nboot=NBOOT):
    idx = rng.integers(0, len(x), size=(nboot, len(x)))
    means = x[idx].mean(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def main():
    out = {}
    for scen, base, new in SCEN:
        rows = load(base, new)
        rng = np.random.default_rng(0)
        seeds = {a: [t["seed"] for t in rows[a]] for a in ARMS if a in rows}
        ref = seeds["head_1l"]
        for a, s in seeds.items():
            assert s == ref, f"{scen}/{a}: seed grid differs -- pairing invalid"

        # per-trial cumulative-return curves, shape (n_trials, T)
        cum = {a: np.array([np.cumsum(t["per_step_reward"]) for t in rows[a]])
               for a in ARMS if a in rows}
        n = len(ref)
        tcrit = stats.t.ppf(0.975, n - 1)
        base_ret = cum["head_1l"][:, -1]

        arms = {}
        for a in ARMS:
            if a not in cum:
                print(f"  [skip] {scen}: {a}")
                continue
            C = cum[a]
            fin = C[:, -1]
            sd = fin.std(ddof=1)
            sem = sd / np.sqrt(n)
            lo, hi = boot_ci(fin, rng)
            bal = sum(bool(t["balanced"]) for t in rows[a])
            wlo, whi = wilson(bal, n)

            # per-step band: mean +/- t*s.e.m. at every timestep
            mu_t = C.mean(0)
            sem_t = C.std(0, ddof=1) / np.sqrt(n)

            d = {"n": n,
                 "mean": float(fin.mean()), "sem": float(sem), "sd": float(sd),
                 "ci95": [float(fin.mean() - tcrit * sem), float(fin.mean() + tcrit * sem)],
                 "boot95": [float(lo), float(hi)],
                 "q25": float(np.percentile(fin, 25)), "med": float(np.median(fin)),
                 "q75": float(np.percentile(fin, 75)),
                 "min": float(fin.min()), "max": float(fin.max()),
                 "bal": bal, "bal_ci": [float(wlo), float(whi)],
                 "cum": [round(float(v), 1) for v in mu_t],
                 "sem_t": [round(float(v), 2) for v in sem_t],
                 "ci_t": [round(float(tcrit * v), 2) for v in sem_t]}

            if a != "head_1l":
                diff = base_ret - fin          # >0 means head_1l is better
                dsd = diff.std(ddof=1)
                dsem = dsd / np.sqrt(n)
                t_st, p = stats.ttest_rel(base_ret, fin)
                wins = int((diff > 0).sum())
                wl, wh = wilson(wins, n)
                d["vs_1l"] = {
                    "diff": float(diff.mean()),
                    "ci95": [float(diff.mean() - tcrit * dsem),
                             float(diff.mean() + tcrit * dsem)],
                    "p": float(p), "dz": float(diff.mean() / dsd),
                    "wins": wins, "win_ci": [float(wl), float(wh)]}
            arms[a] = d
        out[scen] = arms

    (HERE / "bounds.json").write_text(json.dumps(out))

    for scen in out:
        print(f"\n=== {scen} (n={out[scen]['head_1l']['n']}, 200 steps) ===")
        print(f"{'arm':<18}{'mean':>9}{'95% CI':>21}{'bootstrap 95%':>21}"
              f"{'sd':>8}{'IQR':>18}{'bal%':>6}{'bal 95% CI':>14}")
        for a in ARMS:
            if a not in out[scen]:
                continue
            d = out[scen][a]
            ci = f"[{d['ci95'][0]:.1f}, {d['ci95'][1]:.1f}]"
            bt = f"[{d['boot95'][0]:.1f}, {d['boot95'][1]:.1f}]"
            iq = f"[{d['q25']:.0f}, {d['q75']:.0f}]"
            bc = f"[{d['bal_ci'][0]:.0f}, {d['bal_ci'][1]:.0f}]"
            print(f"{a:<18}{d['mean']:>9.1f}{ci:>21}{bt:>21}{d['sd']:>8.1f}"
                  f"{iq:>18}{100*d['bal']/d['n']:>6.0f}{bc:>14}")
        print("\n  paired vs head_1l (positive = head_1l better)")
        for a in ARMS:
            if a not in out[scen] or a == "head_1l":
                continue
            v = out[scen][a]["vs_1l"]
            print(f"  {a:<18}{v['diff']:>8.1f}  CI[{v['ci95'][0]:>7.1f},{v['ci95'][1]:>7.1f}]"
                  f"  p={v['p']:.2e}  d_z={v['dz']:>5.2f}"
                  f"  wins {v['wins']}/{out[scen][a]['n']} "
                  f"CI[{v['win_ci'][0]:.0f},{v['win_ci'][1]:.0f}]")


if __name__ == "__main__":
    main()
