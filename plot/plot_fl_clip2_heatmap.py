"""Heatmap: reward over (post-change p', retain floor) with the surprise clip at 2.

Model pretrained DETERMINISTIC (p=1); change at ts=0 to [p', (1-p')/2, (1-p')/2].
Every cell additionally clips the per-step surprise at delta_n <- min(delta_n, 2),
which is what rescues the p'=0.9 column: capping delta_n keeps the smoothed dbar
below the delta_bar > 1 trigger when the change is small, so forgetting does not
fire on a model that is already nearly right.

Vertical axis is the retain floor rho_min, the lower clip on the per-firing
retention factor.  rho_min = 1e-3 is shipped (effectively unbounded forgetting,
since retain compounds every step); rho_min -> 1 approaches no forgetting.

Cells come from results/fl_clip2_grid (this sweep) plus the shipped-floor row
already in results/fl_clip2_n99 where a matching n=100 run is absent.  Cells with
no run are left blank rather than interpolated.

    python plot/plot_fl_clip2_heatmap.py
"""
import copy
import glob
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path("plot/fl_clip2_floor_heatmap.png")
P_VALS = [0.1, 0.3, 0.5, 0.7, 0.9]
FLOORS = [0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999]


def collect():
    """(p', rho_floor) -> (reward, sem, n, src) for clip=2 runs."""
    cells = {}
    for f in glob.glob("results/**/*.json", recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or "goal_rate" not in d:
            continue
        try:
            clip = float(d.get("clip_delta_n", d.get("surprise_clip", 0)) or 0)
        except Exception:
            continue
        if clip != 2.0:
            continue
        if d.get("n_unfrozen") != 1 or d.get("cvar_alpha") != 0.0:
            continue
        if d.get("forget") is not True or d.get("counts") is not True:
            continue
        if d.get("forget_strength", 1.0) != 1.0:
            continue
        if d.get("max_forgets") is not None or d.get("retrain_variance"):
            continue
        if "p07" in str(d.get("ckpt", "")):
            continue
        sch = d.get("schedule")
        if not sch:
            continue
        p, fl = sch[0][1], d.get("rho_floor", 0.001)
        if p not in P_VALS or fl not in FLOORS:
            continue
        gr = float(d["goal_rate"])
        n = len(d["records"])
        sem = float(np.sqrt(gr * (1 - gr) / (n - 1))) if n > 1 else 0.0
        prev = cells.get((p, fl))
        # prefer the larger-n run when duplicates exist
        if prev is None or n > prev[2]:
            cells[(p, fl)] = (gr, sem, n, os.path.relpath(f))
    return cells


def main():
    cells = collect()
    M = np.full((len(FLOORS), len(P_VALS)), np.nan)
    S = np.full_like(M, np.nan)
    N = np.zeros_like(M)
    for i, fl in enumerate(FLOORS):
        for j, p in enumerate(P_VALS):
            if (p, fl) in cells:
                M[i, j], S[i, j], N[i, j], _ = cells[(p, fl)]

    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    cmap = copy.copy(plt.get_cmap("viridis"))
    cmap.set_bad("#e8e8e8")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0.0, vmax=1.0,
                   aspect="auto", origin="lower")

    ax.set_xticks(range(len(P_VALS)))
    ax.set_xticklabels([f"{p:g}" for p in P_VALS])
    ax.set_yticks(range(len(FLOORS)))
    ax.set_yticklabels([("1e-3\n(shipped)" if f == 0.001 else f"{f:g}") for f in FLOORS],
                       fontsize=9)
    ax.set_xlabel(r"post-change slip probability  $p'$")
    ax.set_ylabel(r"retain floor  $\kappa_{\min}$")

    for i in range(len(FLOORS)):
        for j in range(len(P_VALS)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=10, color="#999999")
            else:
                col = "white" if M[i, j] < 0.55 else "black"
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        fontsize=9, color=col)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("reward (hole=0, goal=+1)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(f"wrote {OUT}")

    done = len(cells)
    print(f"\n{done}/{len(FLOORS) * len(P_VALS)} cells present")
    for (p, fl), (gr, sem, n, src) in sorted(cells.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"  p'={p:<4g} floor={fl:<6g} reward={gr:.3f} +/- {sem:.3f}  n={n:<4g} {src}")


if __name__ == "__main__":
    main()
