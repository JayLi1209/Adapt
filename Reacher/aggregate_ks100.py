#!/usr/bin/env python3
"""Pool the sharded 100-seed confirmation runs and test nl1_forget_qv.

Shards are seed ranges of the same experiment, so they are pooled at the
per-seed level and every statistic is recomputed from the pooled returns --
never averaged across shards, which would be wrong for the SEs.

The test is paired by seed: nl1_forget_qv and no_adapt see identical seeds,
so the comparison uses the paired differences, not a two-sample test.
"""
import json, glob, os, math
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ks100")


def load(ks):
    """arm -> {seed: return}, pooled over shards for one k_s."""
    per = {}
    files = sorted(glob.glob(os.path.join(RES, f"ks{ks}_s*.json")))
    for p in files:
        with open(p) as f:
            d = json.load(f)
        assert float(d["config"]["wind_ks"]) == float(ks), (p, d["config"]["wind_ks"])
        for r in d["rows"]:
            per.setdefault(r["arm"], {}).update({t["seed"]: t["ret"] for t in r["trials"]})
    return per, len(files)


def paired_t(a, b, seeds):
    d = np.array([a[s] - b[s] for s in seeds], dtype=float)
    n = len(d)
    md, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    t = float("nan") if se == 0 else md / se
    # 95% CI, normal approx (n=100 -> t_crit ~ 1.984, close enough to report both)
    return md, t, se, int((d > 0).sum()), n


def main():
    for ks in ("0.5", "1"):
        per, nsh = load(ks)
        if not per:
            print(f"\n### k_s = {ks}: no shards yet\n")
            continue
        if "nl1_forget_qv" not in per or "no_adapt" not in per:
            print(f"\n### k_s = {ks}: incomplete ({nsh} shards, arms={sorted(per)})\n")
            continue

        seeds = sorted(set(per["nl1_forget_qv"]) & set(per["no_adapt"]))
        print(f"\n### k_s = {ks}   F_max = 500   {len(seeds)} seeds "
              f"({nsh} shards, {min(seeds)}..{max(seeds)})")

        for arm in ("oracle", "nl1_forget_qv", "no_adapt"):
            if arm not in per:
                continue
            v = np.array([per[arm][s] for s in sorted(per[arm])], dtype=float)
            print(f"  {arm:<16} return {v.mean():8.3f} +/- {v.std(ddof=1)/math.sqrt(len(v)):.3f}"
                  f"  (n={len(v)})")

        md, t, se, wins, n = paired_t(per["nl1_forget_qv"], per["no_adapt"], seeds)
        lo, hi = md - 1.984 * se, md + 1.984 * se
        print(f"\n  nl1_forget_qv - no_adapt (paired, n={n})")
        print(f"    mean delta : {md:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
        print(f"    t          : {t:+.3f}")
        print(f"    wins       : {wins}/{n}  ({100.0*wins/n:.0f}%)")
        if "oracle" in per:
            os_ = sorted(set(per["oracle"]) & set(per["no_adapt"]))
            hr = float(np.mean([per["oracle"][s] - per["no_adapt"][s] for s in os_]))
            print(f"    headroom   : {hr:+.3f}  -> recovers {100.0*md/hr:.1f}%")
        sig = "SIGNIFICANT at 0.05" if abs(t) > 1.984 else "NOT significant at 0.05"
        print(f"    verdict    : {sig} (|t| vs 1.984)")


if __name__ == "__main__":
    main()
