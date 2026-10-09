"""FrozenLake paper figures: retain-factor curves and reward heatmaps -> graph/.

  graph/frozenlake_retain.{pdf,png}         retain r_t vs step, one line per p'
                                            (left: first 30 steps, linear;
                                             right: full horizon, log)
  graph/frozenlake_floor_heatmap.{pdf,png}  reward over (p', rho_min), no clip
  graph/frozenlake_clip2_heatmap.{pdf,png}  reward over (p', rho_min), delta_n<=2

Setting (every cell and curve): 4x4 FrozenLake, Dirichlet head pretrained on the
deterministic map (p=1); at t=0 the slip becomes [p', (1-p')/2, (1-p')/2].
Reward: goal=+1, hole=0, else 0, so reward == goal rate.  CVaR+CEM alpha=0,
gamma=1 (no discount), K_FORGET=1, n_unfrozen=1, counts on, forget_strength=1,
max-steps 1000 (0-4% of trials per cell reach it; trunc_rate is stored per
cell in the heatmap json), 100 trials.

Data lives in graph/data/ so the figures re-render without results/:
  frozenlake_retain.json        mean / s.e.m. / n_alive of r_t per step, per p',
                                from results/fl_changep/U1_p0pX_steps.jsonl
                                (shipped floor rho_min=1e-3, no surprise clip)
  frozenlake_floor_heatmap.json reward / s.e.m. / n / source run per cell, for
                                the no-clip and clip=2 grids, from results/**.json

Usage:
  python plot/plot_fl_graphs.py            # re-extract from results/, then plot
  python plot/plot_fl_graphs.py --no-extract   # plot from graph/data/ only
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_report_graphs import (AXIS, DATA, INK, INK2, INK3, ROOT,  # noqa: E402
                                plt, save, style)

RETAIN_JSON = os.path.join(DATA, "frozenlake_retain.json")
HEAT_JSON = os.path.join(DATA, "frozenlake_floor_heatmap.json")

P_VALS = [0.1, 0.3, 0.5, 0.7, 0.9]
FLOORS_NOCLIP = [0.001, 0.95, 0.99, 0.999]
FLOORS_CLIP2 = [0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999]
MIN_TRIALS = 10          # stop a retain curve once fewer trials are still running
T_MAX = 400

# ordinal blue ramp (steps 250..650): low p' (big change) light, p'=0.9 dark
P_COL = dict(zip(P_VALS, ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]))
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])
SEQ.set_bad("#f3f3f1")


# ─────────────────────────── extraction ───────────────────────────
def extract_retain():
    arms = {}
    for p in P_VALS:
        path = os.path.join(ROOT, "results/fl_changep",
                            f"U1_p{str(p).replace('.', 'p')}_steps.jsonl")
        if not os.path.exists(path):
            print("  missing", path)
            continue
        per = collections.defaultdict(dict)
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    per[r["trial"]][r["t"]] = r["retain"]
        t_, mu, se, na = [], [], [], []
        for t in range(T_MAX):
            v = np.array([d[t] for d in per.values() if t in d], dtype=float)
            if len(v) < MIN_TRIALS:
                break
            t_.append(t)
            mu.append(float(v.mean()))
            se.append(float(v.std(ddof=1) / np.sqrt(len(v))))
            na.append(int(len(v)))
        arms[str(p)] = {"t": t_, "mean": mu, "sem": se, "n_alive": na,
                        "n_trials": len(per)}
    return {"min_trials": MIN_TRIALS, "rho_floor": 0.001, "clip": None,
            "arms": arms}


def _admissible(d):
    if not isinstance(d, dict) or "goal_rate" not in d:
        return False
    return (d.get("n_unfrozen") == 1 and d.get("cvar_alpha") == 0.0
            and d.get("forget") is True and d.get("counts") is True
            and d.get("forget_strength", 1.0) == 1.0
            and d.get("max_forgets") is None and not d.get("retrain_variance")
            and "p07" not in str(d.get("ckpt", ""))       # p=0.7-pretrained grid
            and bool(d.get("schedule")))


def extract_heatmaps():
    """Same admission rules as plot_fl_floor_heatmap / plot_fl_clip2_heatmap."""
    grids = {"noclip": {}, "clip2": {}}
    for f in sorted(glob.glob(os.path.join(ROOT, "results/**/*.json"), recursive=True)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not _admissible(d):
            continue
        try:
            clip = float(d.get("clip_delta_n", d.get("surprise_clip", 0)) or 0)
        except Exception:
            continue
        p, fl = d["schedule"][0][1], d.get("rho_floor", 0.001)
        if p not in P_VALS:
            continue
        if clip == 0.0 and fl in FLOORS_NOCLIP and d.get("trials") == 100 \
                and d.get("trial_len") == 1000:
            key = "noclip"
        elif clip == 2.0 and fl in FLOORS_CLIP2:
            key = "clip2"
        else:
            continue
        gr = float(d["goal_rate"])
        n = len(d["records"])
        cell = {"p": p, "floor": fl, "reward": gr,
                "sem": float(np.sqrt(gr * (1 - gr) / (n - 1))) if n > 1 else 0.0,
                "n": n, "trunc_rate": d.get("trunc_rate"),
                "gamma": d.get("gamma"), "k_forget": d.get("k_forget"),
                "first_detect_dbar_mean": d.get("first_detect_dbar_mean"),
                "first_detect_dbar_sem": d.get("first_detect_dbar_sem"),
                "src": os.path.relpath(f, ROOT)}
        k = f"{p}|{fl}"
        prev = grids[key].get(k)
        if prev is None or n > prev["n"]:          # prefer the larger-n duplicate
            grids[key][k] = cell
    return {"p_vals": P_VALS,
            "noclip": {"floors": FLOORS_NOCLIP, "cells": list(grids["noclip"].values())},
            "clip2": {"floors": FLOORS_CLIP2, "cells": list(grids["clip2"].values())}}


# ─────────────────────────── figures ───────────────────────────
def retain_figure(R):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6),
                             gridspec_kw={"wspace": 0.42})
    for ax, logy, tmax in ((axes[0], False, 30), (axes[1], True, T_MAX)):
        style(ax)
        for p in P_VALS:
            a = R["arms"].get(str(p))
            if a is None:
                continue
            t = np.array(a["t"])
            keep = t < tmax
            t, mu, se = t[keep], np.array(a["mean"])[keep], np.array(a["sem"])[keep]
            col = P_COL[p]
            lo = mu - se
            if logy:
                lo = np.maximum(lo, mu * 1e-2)           # keep band on the log axis
            ax.fill_between(t, lo, mu + se, color=col, alpha=0.16, lw=0)
            ax.plot(t, mu, color=col, lw=2.0, label=f"p′ = {p:g}")
            if logy:
                ax.plot(t[-1], mu[-1], "o", ms=4, color=col, mec="white", mew=1)
        ax.axhline(1.0, color=AXIS, lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("step t   (change applied at t = 0)")
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-48, 3.0)      # mean r_t bottoms out near 1e-46
            ax.set_xlim(0, T_MAX)
            ax.set_title("Full episode (log scale)", loc="left", color=INK)
            ax.text(0.99, 0.97, f"curves end when < {R['min_trials']} trials remain;\n"
                    "points where mean r_t = 0 exactly are not drawn",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color=INK3)
        else:
            ax.set_ylim(-0.02, 1.05)
            ax.set_xlim(0, tmax - 1)
            ax.set_ylabel(r"retain  $r_t$", color=INK3)
            ax.set_title("First 30 steps", loc="left", color=INK)
            # every curve ends near 0, so direct end labels would collide
            ax.legend(loc="upper right", handlelength=1.4)
    fig.suptitle("FrozenLake · Dirichlet retain factor after the change   "
                 "(pretrained p = 1, κ_min = 1e-3, no clip, 100 trials, mean ± s.e.m.)",
                 x=0.06, ha="left", fontsize=9.5, color=INK2, y=1.02)
    save(fig, "frozenlake_retain")


def heatmap_figure(H, key, title, name, height):
    floors = H[key]["floors"]
    M = np.full((len(floors), len(P_VALS)), np.nan)
    S = np.full_like(M, np.nan)
    for c in H[key]["cells"]:
        M[floors.index(c["floor"]), P_VALS.index(c["p"])] = c["reward"]
        S[floors.index(c["floor"]), P_VALS.index(c["p"])] = c["sem"]

    fig, ax = plt.subplots(figsize=(6.4, height))
    im = ax.imshow(np.ma.masked_invalid(M), cmap=SEQ, vmin=0, vmax=1,
                   aspect="auto", origin="lower")
    # 2px-style surface gaps between cells
    ax.set_xticks(np.arange(-.5, len(P_VALS)), minor=True)
    ax.set_yticks(np.arange(-.5, len(floors)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_xticks(range(len(P_VALS)))
    ax.set_xticklabels([f"{p:g}" for p in P_VALS])
    ax.set_yticks(range(len(floors)))
    ax.set_yticklabels(["1e-3 (shipped)" if f == 0.001 else f"{f:g}" for f in floors])
    ax.set_xlabel("post-change slip probability  p′")
    ax.set_ylabel(r"retain floor  $\kappa_{\min}$")
    ax.set_title(title, loc="left", color=INK)

    for i in range(len(floors)):
        for j in range(len(P_VALS)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "not run", ha="center", va="center",
                        fontsize=7.5, color=INK3, style="italic")
            else:
                col = "white" if M[i, j] > 0.5 else INK
                ax.text(j, i, f"{M[i, j]:.2f}\n±{S[i, j]:.2f}", ha="center",
                        va="center", fontsize=7.5, color=col, linespacing=1.2)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    cb.set_label("reward = goal rate  (goal +1, hole 0)", color=INK3)
    save(fig, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-extract", action="store_true",
                    help="plot from graph/data/ without reading results/")
    args = ap.parse_args()

    if not args.no_extract:
        R, H = extract_retain(), extract_heatmaps()
        json.dump(R, open(RETAIN_JSON, "w"))
        json.dump(H, open(HEAT_JSON, "w"), indent=1)
        print("wrote", os.path.relpath(RETAIN_JSON, ROOT), "and",
              os.path.relpath(HEAT_JSON, ROOT))
    R, H = json.load(open(RETAIN_JSON)), json.load(open(HEAT_JSON))

    retain_figure(R)
    heatmap_figure(H, "noclip", "Reward by retain floor · no surprise clip",
                   "frozenlake_floor_heatmap", 3.6)
    heatmap_figure(H, "clip2", "Reward by retain floor · surprise clip δₙ ≤ 2",
                   "frozenlake_clip2_heatmap", 6.4)

    for p in P_VALS:
        a = R["arms"].get(str(p))
        if a:
            m = a["mean"]
            print(f"  retain p'={p:g}: r_1={m[1]:.3f} r_10={m[10]:.3g} "
                  f"r_50={m[50]:.3g}  to t={a['t'][-1]} ({a['n_alive'][-1]} alive)")
    for key in ("noclip", "clip2"):
        cells = H[key]["cells"]
        print(f"\n{key}: {len(cells)}/{len(H[key]['floors']) * len(P_VALS)} cells")
        for c in sorted(cells, key=lambda c: (c["floor"], c["p"])):
            print(f"  p'={c['p']:<4g} floor={c['floor']:<6g} reward={c['reward']:.3f}"
                  f" ± {c['sem']:.3f}  n={c['n']:<4} trunc={c['trunc_rate']}"
                  f"  dbar1={c['first_detect_dbar_mean']}  {c['src']}")


if __name__ == "__main__":
    main()
