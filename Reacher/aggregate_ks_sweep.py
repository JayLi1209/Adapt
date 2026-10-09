#!/usr/bin/env python3
"""Aggregate the F_max=500 k_s sweep into one cross-k_s table.

For each k_s cell, reports every arm's return +/- SE, its delta against
no_adapt, a paired t-statistic over the shared seeds, and a per-seed win
count -- the same columns as the F=5 / F=500 tables it extends.

Pairing is by seed: every arm in a cell runs seeds 1000..1002, so the
no_adapt comparison is a paired test, not a two-sample one.
"""
import json, glob, os, math
import numpy as np

KS_ORDER = [0.5, 1.0, 5.0, 15.0]
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ks_sweep")


def load():
    cells = {}
    for p in sorted(glob.glob(os.path.join(RES, "ks*.json"))):
        with open(p) as f:
            d = json.load(f)
        ks = float(d["config"]["wind_ks"])
        # arm -> {seed: return}
        per = {}
        agg = {}
        for r in d["rows"]:
            per[r["arm"]] = {t["seed"]: t["ret"] for t in r["trials"]}
            agg[r["arm"]] = (r["ret"], r["ret_se"], r["pred_err"], r["final_dist"])
        cells[ks] = (per, agg)
    return cells


def paired(a, b):
    """Paired t over seeds present in both. Returns (mean diff, t, wins, n)."""
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return float("nan"), float("nan"), 0, 0
    d = np.array([a[s] - b[s] for s in seeds], dtype=float)
    n = len(d)
    md = float(d.mean())
    wins = int((d > 0).sum())
    if n < 2:
        return md, float("nan"), wins, n
    sd = d.std(ddof=1)
    t = float("nan") if sd == 0 else md / (sd / math.sqrt(n))
    return md, t, wins, n


def main():
    cells = load()
    if not cells:
        print("no results yet in", RES)
        return

    for ks in KS_ORDER:
        if ks not in cells:
            print(f"\n### k_s = {ks:g}  -- NOT PRESENT\n")
            continue
        per, agg = cells[ks]
        if "no_adapt" not in per or "oracle" not in per:
            print(f"\n### k_s = {ks:g}  -- incomplete (missing reference arm)\n")
            continue
        base = per["no_adapt"]
        headroom = agg["oracle"][0] - agg["no_adapt"][0]
        n_seeds = len(base)
        print(f"\n### k_s = {ks:g}   F_max = 500   {n_seeds} seeds"
              f"   headroom (oracle - no_adapt) = {headroom:+.3f}")
        print(f"  {'arm':<20}{'return':>10}{'±SE':>8}{'vs no_adapt':>13}"
              f"{'t':>8}{'wins':>7}{'pred_err':>10}")
        rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
        for arm, (ret, se, pe, fd) in rows:
            if arm == "no_adapt":
                print(f"  {arm:<20}{ret:>10.3f}{se:>8.3f}{'--':>13}"
                      f"{'':>8}{'':>7}{pe:>10.4f}")
                continue
            md, t, wins, n = paired(per[arm], base)
            ts = "  n/a" if math.isnan(t) else f"{t:+.2f}"
            print(f"  {arm:<20}{ret:>10.3f}{se:>8.3f}{md:>+13.3f}"
                  f"{ts:>8}{f'{wins}/{n}':>7}{pe:>10.4f}")

    # cross-k_s view: does any arm's advantage move with k_s?
    arms = sorted({a for _, (p, _) in cells.items() for a in p})
    print("\n\n### vs no_adapt (return delta), by k_s")
    hdr = "".join(f"{('k=' + f'{k:g}'):>12}" for k in KS_ORDER if k in cells)
    print(f"  {'arm':<20}{hdr}")
    for arm in arms:
        if arm == "no_adapt":
            continue
        cs = []
        for k in KS_ORDER:
            if k not in cells:
                continue
            per, _ = cells[k]
            if arm not in per or "no_adapt" not in per:
                cs.append(f"{'--':>12}")
            else:
                md, _, _, _ = paired(per[arm], per["no_adapt"])
                cs.append(f"{md:>+12.3f}")
        print(f"  {arm:<20}{''.join(cs)}")

    print("\n### headroom (oracle - no_adapt), by k_s")
    for k in KS_ORDER:
        if k in cells:
            _, agg = cells[k]
            if "oracle" in agg and "no_adapt" in agg:
                print(f"  k_s={k:<6g}{agg['oracle'][0] - agg['no_adapt'][0]:>10.3f}"
                      f"   (no_adapt {agg['no_adapt'][0]:+.3f},"
                      f" oracle {agg['oracle'][0]:+.3f})")


if __name__ == "__main__":
    main()
