"""Re-render every chart from the two published reports as paper figures.

  Adapter Depth on Pendulum   -> graph/pendulum_*.{pdf,png}   (5 figures)
  Reacher Adaptation Curves   -> graph/reacher_*.{pdf,png}    (4 figures)

Data lives in graph/data/:
  pendulum_adapter_depth.json      the report's embedded dataset, verbatim
                                   (cum return / s.e.m. / w[theta_dot] traces,
                                   101 points = every 2nd step of 200)
  reacher_adaptation_curves.json   recovered from the report's static SVG paths
                                   (the raw results/traces100/ is not on this
                                   machine); endpoints match the report's tables
                                   to within 0.02.

Usage:  python plot/plot_report_graphs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH = os.path.join(ROOT, "graph")
DATA = os.path.join(GRAPH, "data")

INK, INK2, INK3 = "#14161a", "#4a4f57", "#767c85"
GRID, AXIS = "#eeefec", "#c9ccd1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK3, "axes.linewidth": 0.8,
    "xtick.color": INK3, "ytick.color": INK3,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "legend.fontsize": 8, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def end_labels(ax, items, lo, hi, x, min_gap_frac=0.045):
    """Direct labels at the right edge, nudged apart so they never overlap.
    items: (y, text, color, bold); placed in data coords at x (past the axis)."""
    gap = (hi - lo) * min_gap_frac
    items = sorted(items, key=lambda t: -t[0])
    prev = None
    for y, txt, col, bold in items:
        yy = min(max(y, lo), hi)
        if prev is not None and prev - yy < gap:
            yy = prev - gap
        prev = yy
        ax.text(x, yy, txt, color=col, fontsize=7.5, va="center",
                fontweight="bold" if bold else "normal", clip_on=False,
                family="DejaVu Sans Mono")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(GRAPH, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ─────────────────────────── pendulum ───────────────────────────
P_COL = {"head_1l": "#2a78d6", "head": "#4a3aa7", "head_lin": "#1baf7a",
         "oracle": "#a8adb4", "no_adapt": "#c9ccd1", "forget_elbo": "#c9ccd1",
         "head_noforget": "#c9ccd1"}
P_TGT = "#eb6834"
P_CTX = {"no_adapt", "forget_elbo", "head_noforget"}
P_SHORT = {"head_noforget": "head_nf"}
P_ORDER = ["head_lin", "head_1l", "head", "forget_elbo", "head_noforget",
           "no_adapt", "oracle"]
P_LEGEND = [("head_lin", "head_lin (closed form)"), ("head_1l", "head_1l (1 layer)"),
            ("head", "head (3 layers)"), ("oracle", "oracle"), ("no_adapt", "context arms")]


def p_lines(ax, arms, key, skey, names, t):
    for a in names:
        y = np.array(arms[a][key])
        em = a not in P_CTX
        if em and skey in arms[a]:
            s = np.array(arms[a][skey])
            ax.fill_between(t, y - s, y + s, color=P_COL[a], alpha=0.15, lw=0)
        ax.plot(t, y, color=P_COL[a], lw=1.8 if em else 1.1,
                alpha=1 if em else 0.85, solid_capstyle="round")


def p_legend(ax, extra=()):
    hs = [plt.Line2D([], [], color=P_COL[a], lw=2) for a, _ in P_LEGEND]
    ls = [l for _, l in P_LEGEND]
    for h, l in extra:
        hs.append(h); ls.append(l)
    ax.legend(hs, ls, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
              handlelength=1.4, columnspacing=1.2, borderaxespad=0.4)


def pendulum_return(D, scen, ymin, title, name):
    A = D[scen]["arms"]
    n = len(A["head_1l"]["cum"])
    t = np.arange(n) * 2
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    style(ax)
    p_lines(ax, A, "cum", "sem", P_ORDER, t)
    ax.set_xlim(0, 200); ax.set_ylim(ymin, 0)
    ax.set_yticks(np.arange(ymin, 1, 200)); ax.set_xticks([0, 100, 200])
    ax.set_xlabel("timestep"); ax.set_ylabel("cumulative return")
    end_labels(ax, [(A[a]["cum"][-1], f"{P_SHORT.get(a, a)} {A[a]['cum'][-1]:.0f}",
                     P_COL[a] if a not in P_CTX else INK3, a not in P_CTX)
                    for a in P_ORDER], ymin, 0, 205)
    p_legend(ax)
    ax.set_title(title, pad=34, loc="left", color=INK)
    save(fig, name)


def pendulum_gain(D, scen, lo, hi, ticks, title, name):
    A = D[scen]["arms"]
    tgt = D[scen]["target"]
    arms = ["head_lin", "head_1l", "head", "head_noforget"]
    n = len(A["head_1l"]["w"])
    t = np.arange(n) * 2
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    style(ax)
    ax.axhline(tgt, color=P_TGT, lw=1.6, ls=(0, (5, 4)))
    p_lines(ax, A, "w", "wsem", arms, t)
    ax.set_xlim(0, 200); ax.set_ylim(lo, hi)
    ax.set_yticks(ticks); ax.set_xticks([0, 100, 200])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("timestep"); ax.set_ylabel(r"learned action gain  $w[\dot\theta]$")
    items = [(A[a]["w"][-1], f"{P_SHORT.get(a, a)} {A[a]['w'][-1]:.3f}",
              P_COL[a] if a not in P_CTX else INK3, a not in P_CTX) for a in arms]
    items.append((tgt, f"target {tgt}", P_TGT, False))
    end_labels(ax, items, lo, hi, 205)
    hs = [plt.Line2D([], [], color=P_COL[a], lw=2) for a in arms[:3]] + \
         [plt.Line2D([], [], color=P_COL["head_noforget"], lw=2),
          plt.Line2D([], [], color=P_TGT, lw=1.6, ls=(0, (5, 4)))]
    ax.legend(hs, ["head_lin (closed form)", "head_1l (1 layer)", "head (3 layers)",
                   "head_noforget", "target"], loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=3, handlelength=1.6, columnspacing=1.2,
              borderaxespad=0.4)
    ax.set_title(title, pad=34, loc="left", color=INK)
    save(fig, name)


def pendulum_bars(D, name):
    rows = P_ORDER
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    h = 0.36
    y = np.arange(len(rows))[::-1]
    for i, a in enumerate(rows):
        for sc, off, alpha in (("mass", h / 2 + 0.02, 0.45), ("grav", -h / 2 - 0.02, 1.0)):
            v = D[sc]["arms"][a]["stats"]["balpct"]
            ax.barh(y[i] + off, max(v, 0.6 if v > 0 else 0), height=h,
                    color=P_COL[a], alpha=alpha, lw=0)
            ax.text(v + 1.2, y[i] + off, f"{v}%", va="center", fontsize=7.5,
                    color=INK3, family="DejaVu Sans Mono")
    ax.set_yticks(y)
    ax.set_yticklabels([P_SHORT.get(a, a) for a in rows], family="DejaVu Sans Mono",
                       color=INK2)
    ax.set_xlim(0, 104); ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.9); ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xlabel(r"share of 100 trials balanced  ($|\theta| \leq 0.2$, $|\dot\theta| \leq 1.0$, held 20 steps)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color="#767c85", alpha=0.45, lw=0),
               plt.Rectangle((0, 0), 1, 1, color="#767c85", lw=0)],
              ["mass 1 → 4 (upper, light)", "gravity 10 → 15 (lower, solid)"],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.4)
    ax.set_title("Swing-up rate", pad=22, loc="left", color=INK)
    save(fig, name)


# ─────────────────────────── reacher ───────────────────────────
R_COL = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
         "ref": "#8A8D93", "ref2": "#4A4E55"}


def reacher(cell, name):
    arms, order = cell["arms"], cell["order"]
    ticks = cell["yticks"]
    lo, hi = min(ticks), 0.0
    lo = min(lo, min(min(arms[a]["lo"]) for a in arms))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    style(ax)
    for a in order:
        c = R_COL[arms[a]["color"]]
        x = np.array(arms[a]["x"])
        ax.fill_between(x, arms[a]["lo"], arms[a]["hi"], color=c, alpha=0.14, lw=0)
        ax.plot(x, arms[a]["mean"], color=c, lw=1.8, solid_capstyle="round")
        ax.plot(x[-1], arms[a]["mean"][-1], "o", ms=5.5, color=c, mec="white", mew=1.2)
    ax.set_xlim(0, 199); ax.set_ylim(lo * 1.03, hi)
    ax.set_yticks(ticks); ax.set_xticks([0, 100, 199])
    ax.set_xlabel("timestep"); ax.set_ylabel("cumulative return")
    tab = {r["arm"]: r for r in cell["table"]}
    items = []
    for a in order:
        r = tab[a]
        d = "" if r["delta"] in ("—", "") else f"  ({float(r['delta']):+.1f})"
        items.append((r["final"], f"{a} {r['final']:.1f}{d}", R_COL[arms[a]["color"]],
                      a not in ("no_adapt", "oracle")))
    end_labels(ax, items, lo * 1.03, hi, 203)
    hs = [plt.Line2D([], [], color=R_COL[arms[a]["color"]], lw=2) for a in order]
    ax.legend(hs, order, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
              handlelength=1.4, columnspacing=1.2, borderaxespad=0.4)
    ax.set_title(cell["title"].replace("ks", r"$k_s$"), pad=34, loc="left", color=INK)
    save(fig, name)


def main():
    D = json.load(open(os.path.join(DATA, "pendulum_adapter_depth.json")))
    pendulum_return(D, "mass", -1200, "Cumulative return · mass 1 → 4",
                    "pendulum_return_mass")
    pendulum_return(D, "grav", -1400, "Cumulative return · gravity 10 → 15",
                    "pendulum_return_gravity")
    pendulum_bars(D, "pendulum_balance_rate")
    pendulum_gain(D, "mass", 0, 0.16, [0, .04, .08, .12, .16],
                  r"Learned action gain · mass (target 0.0375)", "pendulum_gain_mass")
    pendulum_gain(D, "grav", 0.06, 0.18, [.06, .09, .12, .15, .18],
                  r"Learned action gain · gravity (unchanged at 0.15)",
                  "pendulum_gain_gravity")

    R = json.load(open(os.path.join(DATA, "reacher_adaptation_curves.json")))
    names = ["reacher_wind_ks0p5", "reacher_wind_ks5",
             "reacher_rotation_90", "reacher_rotation_150"]
    for cell, nm in zip(R, names):
        reacher(cell, nm)


if __name__ == "__main__":
    main()
