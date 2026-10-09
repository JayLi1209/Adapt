"""CLIP 2 (--clip-delta-n 2) from the DETERMINISTIC p=1.0 prior, 99 trials/cell.

Re-runs the earlier n=30 table at n=99 and adds the p'=0.7 level, with proper
interval estimates.

ERROR BOUNDS.  A goal rate is a binomial proportion, so three quantities are
reported rather than one:
  * SEM          sqrt(p(1-p)/n)  -- the plain standard error
  * Wilson 95%   the interval that stays inside [0,1] and is accurate near the
                 ends, where the normal approximation fails
  * p-value      two-proportion z-test vs the shipped arm at the same level
Differences also carry a 95% CI on the DIFFERENCE of two proportions.

THE DEGENERACY CHECK.  Clipping delta_n low enough drags mean(delta_n) below the
calibrated baseline of 1, lambda_hat goes permanently negative and forgetting
stops firing -- the arm becomes no-forget by another route.  `fires/trial` is
therefore printed beside every goal rate: an arm at ~0 firings is measuring the
no-forget result, not a better forgetting rule.  At p'=0.9 from this prior
no-forget scores 0.88 (n=100), so a degenerate arm looks like a big win here;
the p=0.7-prior control (results/saved/clip2_prior_p0p7) shows the same knob
scoring 0.23 where no-forget is worth only 0.30.

Settings: n_unfrozen=1 (head-only), Dirichlet head, CVaR+CEM at cvar_alpha=0.0,
gamma=1.0 (NO discount; discount only inside the planner), K_FORGET=1, change at
ts=0 ([1,0,0] -> [p',(1-p')/2,(1-p')/2]), --max-steps 1000 (NO truncation),
holes scored 0 so mean return == goal rate, seed 0.

    python FrozenLake/summarize_clip2_n99.py
"""
import json
import pathlib
from math import erfc, sqrt

import numpy as np

N99 = pathlib.Path("results/fl_clip2_n99")
# Shipped (no-clip) baselines from the same prior.
SHIPPED = {
    "0.9": pathlib.Path("results/fl_changep/U1_p0p9.json"),
    "0.7": N99 / "U1_p0p7.json",              # run alongside; falls back below
    "0.5": pathlib.Path("results/fl_changep/U1_p0p5.json"),
    "0.3": pathlib.Path("results/fl_changep/U1_p0p3.json"),
    "0.1": pathlib.Path("results/fl_changep/U1_p0p1.json"),
}
SHIPPED_FALLBACK = {"0.7": pathlib.Path("results/unfrozen_a0/U1.json")}
LEVELS = ["0.9", "0.7", "0.5", "0.3", "0.1"]
NOFORGET_09 = 0.88          # n=100, results/fl_forget_mag/U1_nf_p0p9.json


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def two_prop(k1, n1, k2, n2, z=1.96):
    """(delta, lo, hi, p) for p1 - p2, Wald CI on the difference."""
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    se_d = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    p = (k1 + k2) / (n1 + n2)
    se_p = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    pv = 1.0 if se_p == 0 else erfc(abs(d / se_p) / sqrt(2))
    return d, d - z * se_d, d + z * se_d, pv


def load(path):
    path = pathlib.Path(path)
    if not path.exists():
        return None
    d = json.load(open(path))
    r = d["records"]
    n = len(r)
    # Legacy JSONs (e.g. results/unfrozen_a0, written before these fields
    # existed) lack n_forget_fired / final_retain; report nan rather than
    # dropping an otherwise valid baseline.
    def agg(key, fn):
        v = [x[key] for x in r if key in x]
        return float(fn(v)) if v else float("nan")
    return dict(
        n=n, k=int(sum(x["outcome"] == "goal" for x in r)),
        fires=agg("n_forget_fired", np.mean),
        steps=agg("steps", np.mean),
        retain=agg("final_retain", np.median),
        trunc=float(d.get("trunc_rate", 0.0)),
    )


def main():
    print("=" * 122)
    print("CLIP 2 (delta_n <- min(delta_n, 2)) -- pretrained p = 1.0 (DETERMINISTIC), 99 trials/cell")
    print("=" * 122)
    print("  Model believes deterministic, so SMALLER p' = BIGGER change.")
    print("  Change [1,0,0] -> [p',(1-p')/2,(1-p')/2] at ts=0; no truncation; no discount;")
    print("  CVaR+CEM alpha=0.0; K_FORGET=1; holes scored 0 => mean return == goal rate.\n")
    print(f"  {'p prime':<8} {'arm':<9} {'n':>4} {'goal +/- SEM':>15} {'Wilson 95% CI':>18} "
          f"{'delta [95% CI]':>24} {'p':>8} {'fires':>8} {'steps':>7}  {'note':<22}")
    print("  " + "-" * 118)
    rows = []
    for lv in LEVELS:
        sp = SHIPPED.get(lv)
        b = load(sp)
        if b is None and lv in SHIPPED_FALLBACK:
            b = load(SHIPPED_FALLBACK[lv])
        c = load(N99 / f"U1_p{lv.replace('.','p')}_clip2.json")
        for lab, s in (("shipped", b), ("clip 2", c)):
            if s is None:
                print(f"  {lv:<8} {lab:<9} {'-- running / missing --':>20}")
                continue
            g = s["k"] / s["n"]
            lo, hi = wilson(s["k"], s["n"])
            if lab == "clip 2" and b is not None:
                d, dlo, dhi, pv = two_prop(s["k"], s["n"], b["k"], b["n"])
                dstr = f"{d:+.3f} [{dlo:+.3f},{dhi:+.3f}]"
                pstr = f"{pv:.4f}"
            else:
                dstr, pstr = "--", "--"
            note = ("" if s["fires"] != s["fires"] else
                    "DEGENERATE == no-forget" if s["fires"] < 1.0
                    else ("near-deadband" if s["fires"] < 10 else ""))
            print(f"  {lv:<8} {lab:<9} {s['n']:>4} "
                  f"{g:>7.3f} +/-{sqrt(g*(1-g)/s['n']):.3f} "
                  f"[{lo:.3f}, {hi:.3f}]".rjust(19)
                  + f"{dstr:>25} {pstr:>8} {s['fires']:>8.1f} {s['steps']:>7.1f}  {note:<22}")
            if lab == "clip 2":
                rows.append((lv, b, s))
        print()

    print("  " + "=" * 118)
    print("  SUMMARY: clip 2 minus shipped, with 95% CI on the difference")
    print("  " + "=" * 118)
    print(f"\n  {'p prime':<9} {'shipped':>9} {'clip 2':>9} {'delta':>9} "
          f"{'95% CI on delta':>22} {'p':>9} {'fires':>8}  {'reading':<26}")
    print("  " + "-" * 108)
    for lv, b, c in rows:
        if b is None:
            continue
        d, dlo, dhi, pv = two_prop(c["k"], c["n"], b["k"], b["n"])
        read = ("no-forget in disguise" if c["fires"] < 1.0
                else ("CI excludes 0" if dlo > 0 or dhi < 0 else "CI includes 0"))
        print(f"  {lv:<9} {b['k']/b['n']:>9.3f} {c['k']/c['n']:>9.3f} {d:>+9.3f} "
              f"[{dlo:+.3f}, {dhi:+.3f}]".rjust(23)
              + f"{pv:>9.4f} {c['fires']:>8.1f}  {read:<26}")
    print(f"""
  Wilson intervals are used for single rates (correct near 0 and 1); the delta
  carries a Wald CI on the difference of two proportions.  At n=99 the SEM of a
  rate near 0.5 is ~0.050, so differences below ~0.14 are not resolvable.
  For scale, no-forget at p'=0.9 from this prior scores {NOFORGET_09:.2f} (n=100): any arm
  with ~0 fires/trial is measuring THAT, not a better forgetting rule.
""")


if __name__ == "__main__":
    main()
