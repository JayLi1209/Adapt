"""Paper figures for the Pendulum adapter-depth and Reacher adaptation-curve reports.

Every figure uses the paper-table naming and one fixed colour per arm, so a line
means the same thing in every PNG.  Sources are the same runs that back the paper
table (see Pendulum/merge_ablations.py and Reacher/compute_table_bounds.py):

    paper name              Pendulum arm            Reacher arm
    SURF                   head_1l                 nl1_forget_qv
    Retrain full            retrain_full            nl1_forget_qv_body
    No retrain              no_retrain              nl1_forget_noelbo
    No forget + no retrain  no_forget_no_retrain    nl1_noforget_noelbo
    No adapt                no_adapt                no_adapt
    No forget               head_1l_noforget        nl1_noforget
    Oracle (reference)      oracle                  oracle

Bands are +/- 1 s.e.m. over 100 seeds (1000-1099); gamma = 1 when summing
(cumulative raw reward), 200 steps, no truncation.

    python3 graph/make_graphs.py        # writes graph/*.png
"""
import glob
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "graph"

ARMS = ["SURF", "Retrain full", "No retrain", "No forget + no retrain",
        "No adapt", "No forget", "Oracle"]
# Categorical slots 1-5 for the adaptation arms (validated: worst adjacent CVD
# dE 9.1, normal-vision dE 19.6); neutral greys for the two reference lines.
COLOR = {"SURF": "#2a78d6", "Retrain full": "#eb6834", "No retrain": "#1baf7a",
         "No forget + no retrain": "#eda100", "No forget": "#e87ba4",
         "No adapt": "#4a4f57", "Oracle": "#a8adb4"}
# Depth-comparison heads (pendulum_gain_depth_*.png only; not paper-table arms).
DEPTH_COLOR = {"3l": "#4a3aa7", "lin": "#008300", "nf3": "#767c85"}
PEND_ARM = {"SURF": "head_1l", "Retrain full": "retrain_full",
            "No retrain": "no_retrain", "No forget + no retrain": "no_forget_no_retrain",
            "No adapt": "no_adapt", "No forget": "head_1l_noforget", "Oracle": "oracle"}
REAC_ARM = {"SURF": "nl1_forget_qv", "Retrain full": "nl1_forget_qv_body",
            "No retrain": "nl1_forget_noelbo", "No forget + no retrain": "nl1_noforget_noelbo",
            "No adapt": "no_adapt", "No forget": "nl1_noforget", "Oracle": "oracle"}

INK, INK2, MUTE, GRID = "#14161a", "#4a4f57", "#767c85", "#eeefec"
LEGEND_FS, BOTTOM_FS = 12, 11.5   # legend text; x-axis labels under the plots
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9ccd1",
    "axes.labelcolor": INK2, "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.9, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 220,
})


# ---------------------------------------------------------------- loading
# scen -> (run holding merged.json, extra shard dirs).  mass/grav are the paper-table
# shifts; mass0p4/grav5 are the easy-direction shifts (SURF/no adapt/oracle from
# easy_*, the other four arms from Pendulum/run_easy_ablations.sh).
PEND_SRC = {"mass": ("head1l_mass", ("nf1l_mass", "abl_mass")),
            "grav": ("head1l_grav15", ("nf1l_grav15", "abl_grav")),
            "mass0p4": ("easy_mass0p4", ("abl_mass0p4",)),
            "grav5": ("easy_grav5", ("abl_grav5",))}


def pendulum(scen):
    """scen in PEND_SRC -> {paper name: list of 100 trial dicts, seed-sorted}."""
    main, extra = PEND_SRC[scen]
    pool = json.loads((ROOT / "Pendulum/results" / main / "merged.json").read_text())
    for d in extra:
        for f in sorted(glob.glob(str(ROOT / "Pendulum/results" / d / "shard_*.json"))):
            for arm, rows in json.loads(pathlib.Path(f).read_text()).items():
                pool.setdefault(arm, []).extend(rows)
    out = {}
    for name, arm in PEND_ARM.items():
        rows = sorted(pool[arm], key=lambda r: r["seed"])
        assert [r["seed"] for r in rows] == list(range(1000, 1100)), (scen, arm)
        out[name] = rows
    return out


def reacher(prefix):
    """prefix like "wind_ks0.5" -> {paper name: (100, 200) per-step reward array}."""
    pool = {}
    for d in ("traces100", "abl100"):
        for f in sorted(glob.glob(str(ROOT / "Reacher/results" / d / f"{prefix}_s*.json"))):
            for row in json.loads(pathlib.Path(f).read_text())["rows"]:
                pool.setdefault(row["arm"], []).extend(row["trials"])
    out = {}
    for name, arm in REAC_ARM.items():
        rows = sorted(pool[arm], key=lambda t: t["seed"])
        assert [t["seed"] for t in rows] == list(range(1000, 1100)), (prefix, arm)
        out[name] = np.array([t["rew_t"] for t in rows], dtype=float)
    return out


def sem(x, axis=0):
    return x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis])


# ---------------------------------------------------------------- drawing
def end_labels(ax, items, fmt):
    """Right-edge direct labels, nudged apart so none overlap. items: (name, y)."""
    lo, hi = ax.get_ylim()
    gap = 0.052 * (hi - lo)
    items = sorted(items, key=lambda t: -t[1])
    pos = []
    for _, y in items:
        p = y if not pos else min(y, pos[-1] - gap)
        pos.append(p)
    # if the stack ran off the bottom, push it back up
    shift = max(0.0, (lo + 0.5 * gap) - pos[-1])
    pos = [p + shift for p in pos]
    x1 = ax.get_xlim()[1]
    xa, xb, xt = x1 * 1.008, x1 * 1.03, x1 * 1.045
    for (name, y), p in zip(items, pos):
        # elbow leader: short flat stub, then a bend that ends before the text
        ax.plot([x1, xa, xb], [y, y, p], color=COLOR[name], lw=1.0, clip_on=False,
                solid_capstyle="butt")
        ax.text(xt, p, f"{name}  {fmt(y)}", va="center", ha="left", fontsize=8.6,
                color=INK, clip_on=False)


def curve_fig(series, title, subtitle, fname, ylabel="Cumulative return"):
    """series: {paper name: (n_seeds, T) per-step reward}."""
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    T = next(iter(series.values())).shape[1]
    t = np.arange(1, T + 1)
    ends = []
    # draw references first so SURF sits on top
    for name in ["Oracle", "No adapt", "No forget + no retrain", "No retrain",
                 "Retrain full", "No forget", "SURF"]:
        cum = np.cumsum(series[name], axis=1)
        m, s = cum.mean(0), sem(cum)
        ls = (0, (4, 3)) if name == "Oracle" else "-"
        lw = 2.6 if name == "SURF" else 1.8
        ax.fill_between(t, m - s, m + s, color=COLOR[name], alpha=0.14, lw=0)
        ax.plot(t, m, color=COLOR[name], lw=lw, ls=ls, solid_capstyle="round",
                zorder=5 if name == "SURF" else 3)
        ends.append((name, m[-1]))
    ax.set_xlim(0, T)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12.5, color=INK, fontweight="bold", pad=18)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.8, color=MUTE)
    end_labels(ax, ends, lambda v: f"{v:.1f}")
    legend(fig, x=0.5)
    fig.subplots_adjust(left=0.1, right=0.72, top=0.86, bottom=0.29)
    fig.savefig(OUT / fname)
    plt.close(fig)


def legend(fig, names=ARMS, x=0.45):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[n], lw=2.6 if n == "SURF" else 1.8,
                ls=(0, (4, 3)) if n == "Oracle" else "-") for n in names]
    fig.legend(h, names, loc="lower center", ncol=4, frameon=False, fontsize=LEGEND_FS,
               bbox_to_anchor=(x, 0.0), handlelength=2.2, columnspacing=1.4)


# ---------------------------------------------------------------- figures
PEND_SUB = ("Pendulum · frozen body + adapter head · n = 100 · band ±1 s.e.m. · "
            "CEM+CVaR, H=40, α=1.0 · 200 steps")
REAC_SUB = ("Reacher · frozen BNN + adapter head · n = 100 · band ±1 s.e.m. · "
            "CEM+CVaR, H=15, α=0.8 · 200 steps")


def grid_fig(panels, nrow, ncol, size, fname):
    """Merged cumulative-return figure: one short tag per panel, no titles, no
    end labels, one shared legend. panels: [(tag, {paper name: (n, T) rewards})]."""
    fig, axes = plt.subplots(nrow, ncol, figsize=size, squeeze=False)
    for ax, (tag, series) in zip(axes.flat, panels):
        T = next(iter(series.values())).shape[1]
        t = np.arange(1, T + 1)
        for name in ["Oracle", "No adapt", "No forget + no retrain", "No retrain",
                     "Retrain full", "No forget", "SURF"]:
            cum = np.cumsum(series[name], axis=1)
            m, s = cum.mean(0), sem(cum)
            ax.fill_between(t, m - s, m + s, color=COLOR[name], alpha=0.14, lw=0)
            ax.plot(t, m, color=COLOR[name], lw=2.4 if name == "SURF" else 1.6,
                    ls=(0, (4, 3)) if name == "Oracle" else "-",
                    zorder=5 if name == "SURF" else 3)
        ax.set_xlim(0, T)
        ax.set_title(tag, loc="left", fontsize=11, color=INK, pad=6)
    for ax in axes[-1]:
        ax.set_xlabel("Timestep", fontsize=BOTTOM_FS)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative return")
    # legend height is fixed in inches, so the bottom margin scales with figure height
    bottom = 0.215 * 6.6 / size[1]
    legend(fig, x=0.535)  # centred under the panels (axes span 0.09-0.98)
    fig.subplots_adjust(left=0.09, right=0.98, top=1 - 0.33 / size[1], bottom=bottom,
                        hspace=0.32, wspace=0.2)
    fig.savefig(OUT / fname)
    plt.close(fig)


def pendulum_figs():
    P = {s: pendulum(s) for s in ("mass", "grav")}
    titles = {"mass": "Pendulum · mass 1 → 4", "grav": "Pendulum · gravity 10 → 15"}
    tag = {"mass": "mass", "grav": "gravity"}
    for s in P:
        series = {n: np.array([r["per_step_reward"] for r in P[s][n]]) for n in ARMS}
        curve_fig(series, titles[s], PEND_SUB, f"pendulum_return_{tag[s]}.png")
    # all four shifts, 2x2 like the Reacher figure: mass on top, gravity below
    P.update({s: pendulum(s) for s in ("mass0p4", "grav5")})
    grid_fig([(t, {n: np.array([r["per_step_reward"] for r in P[k][n]]) for n in ARMS})
              for k, t in (("mass", "m = 1 → 4"), ("mass0p4", "m = 1 → 0.4"),
                           ("grav", "g = 10 → 15"), ("grav5", "g = 10 → 5"))],
             2, 2, (9.2, 6.6), "pendulum_return_all.png")
    P = {s: P[s] for s in ("mass", "grav")}         # balance-rate figure: table shifts

    # balance rate: share of trials holding |θ|<=0.2, |θ̇|<=1.0 for 20 steps;
    # the second copy drops the title/subtitle for use under an external caption
    from matplotlib.patches import Patch
    for header, fname in ((True, "pendulum_balance_rate.png"),
                          (False, "pendulum_balance_rate_noheader.png")):
        fig, ax = plt.subplots(figsize=(7.6, 4.0 if header else 3.6))
        w = 0.11
        for i, name in enumerate(ARMS):
            vals = [100 * np.mean([r["balanced"] is not None for r in P[s][name]])
                    for s in ("mass", "grav")]
            xs = np.arange(2) + (i - (len(ARMS) - 1) / 2) * (w + 0.012)
            ax.bar(xs, vals, width=w, color=COLOR[name],
                   hatch="//" if name == "Oracle" else None, edgecolor="white", lw=0)
            for x, v in zip(xs, vals):
                ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7.8,
                        color=INK2)
        ax.set_xticks([0, 1], ["Mass 1 → 4", "Gravity 10 → 15"])
        ax.tick_params(axis="x", length=0, labelsize=BOTTOM_FS + 0.5, labelcolor=INK)
        ax.set_ylim(0, 108)
        ax.set_ylabel("Balanced trials (%)")
        ax.grid(axis="x", visible=False)
        if header:
            ax.set_title("Pendulum · swing-up and balance rate", loc="left", fontsize=12.5,
                         color=INK, fontweight="bold", pad=18)
            ax.text(0, 1.02, "Share of 100 trials reaching |θ| ≤ 0.2 and |θ̇| ≤ 1.0 and "
                    "holding it 20 steps", transform=ax.transAxes, fontsize=8.8, color=MUTE)
        h = [Patch(color=COLOR[n], hatch="//" if n == "Oracle" else None) for n in ARMS]
        fig.legend(h, ARMS, loc="lower center", ncol=4, frameon=False, fontsize=LEGEND_FS)
        fig.subplots_adjust(left=0.1, right=0.97, top=0.86 if header else 0.97,
                            bottom=0.30 if header else 0.33)
        fig.savefig(OUT / fname)
        plt.close(fig)

    # learned action gain w[θ̇] -- only the arms that own an adapter head
    # (value, label, label x, vertical alignment) -- label placed off the SURF trace
    target = {"mass": (0.0375, "target 0.0375 (0.15 / 4)", 100, "top"),
              "grav": (0.15, "true gain unchanged at 0.15", 4, "bottom")}
    head_arms = ["No forget", "SURF"]
    for s in P:
        fig, ax = plt.subplots(figsize=(7.6, 4.0))
        ends = []
        for name in head_arms:
            wt = np.array([r["w_trace"] for r in P[s][name]])
            m, e = wt.mean(0), sem(wt)
            t = np.arange(1, wt.shape[1] + 1)
            ax.fill_between(t, m - e, m + e, color=COLOR[name], alpha=0.16, lw=0)
            ax.plot(t, m, color=COLOR[name], lw=2.6 if name == "SURF" else 1.8)
            ends.append((name, m[-1]))
        y0, lab, lx, va = target[s]
        ax.axhline(y0, color=MUTE, lw=1.1, ls=(0, (4, 3)))
        ax.text(lx, y0 + (-0.002 if va == "top" else 0.001), lab, va=va,
                fontsize=8.6, color=MUTE)
        ax.set_xlim(0, 200)
        ax.set_ylim(min(0.02, y0 - 0.015), 0.17)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Learned action gain w[θ̇]")
        ax.set_title(titles[s] + " · learned action gain", loc="left", fontsize=12.5,
                     color=INK, fontweight="bold", pad=18)
        ax.text(0, 1.02, "Mean ±1 s.e.m. over 100 seeds · only arms with an adapter "
                "head have this gain", transform=ax.transAxes, fontsize=8.8, color=MUTE)
        end_labels(ax, ends, lambda v: f"{v:.4f}")
        legend(fig, head_arms[::-1])
        fig.subplots_adjust(left=0.11, right=0.74, top=0.86, bottom=0.2)
        fig.savefig(OUT / f"pendulum_gain_{tag[s]}.png")
        plt.close(fig)

    # the report's depth version of the same gain plot: SURF against the 3-layer
    # head, the closed-form affine head and the 3-layer no-forget head
    for s in P:
        main = {"mass": "head1l_mass", "grav": "head1l_grav15"}[s]
        pool = json.loads((ROOT / "Pendulum/results" / main / "merged.json").read_text())
        depth = [("No forget (3-layer head)", "head_noforget", DEPTH_COLOR["nf3"], 1.4, (0, (4, 3))),
                 ("Closed-form head", "head_lin", DEPTH_COLOR["lin"], 1.8, "-"),
                 ("3-layer head", "head", DEPTH_COLOR["3l"], 1.8, "-"),
                 ("SURF", "head_1l", COLOR["SURF"], 2.6, "-")]
        if s == "grav":
            # the closed-form head owns only w·u, so a passive-term change is outside
            # its span and its gain thrashes (seed-mean moves ~±0.1 per step); drawn
            # raw it buries the other lines, so it is left out and the subtitle says so
            depth = [d for d in depth if d[1] != "head_lin"]
        fig, ax = plt.subplots(figsize=(7.6, 4.0))
        ends, handles = [], []
        for name, arm, c, lw, ls in depth:
            rows = sorted(pool[arm], key=lambda r: r["seed"])
            assert [r["seed"] for r in rows] == list(range(1000, 1100)), (s, arm)
            wt = np.array([r["w_trace"] for r in rows])
            m, e = wt.mean(0), sem(wt)
            t = np.arange(1, wt.shape[1] + 1)
            ax.fill_between(t, m - e, m + e, color=c, alpha=0.16, lw=0)
            h, = ax.plot(t, m, color=c, lw=lw, ls=ls)
            handles.append((name, h))
            ends.append((name, m[-1]))
            COLOR.setdefault(name, c)  # end_labels looks leader colours up here
        y0, lab, lx, va = target[s]
        # SURF hugs the mass target from above and the closed-form head from just
        # below, so that label drops into the clear band under the green trace
        dy = {"mass": -0.0045, "grav": 0.001}[s]
        va = "top" if dy < 0 else "bottom"
        ax.axhline(y0, color=MUTE, lw=1.1, ls=(0, (1, 2)))
        ax.text(130 if s == "mass" else lx, y0 + dy, lab, va=va, fontsize=8.6, color=MUTE)
        ax.set_xlim(0, 200)
        ax.set_ylim(min(0.02, y0 - 0.015), 0.17)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Learned action gain w[θ̇]")
        ax.set_title(titles[s] + " · action gain by head depth", loc="left",
                     fontsize=12.5, color=INK, fontweight="bold", pad=18)
        sub = ("Mean ±1 s.e.m. over 100 seeds · SURF = 1-layer head (30 params); "
               "3-layer head = 1060 params")
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.8, color=MUTE)
        if s == "grav":
            ax.text(0.98, 0.04, "Closed-form head omitted: it owns only the action\n"
                    "channel, so under a passive-term change its gain thrashes",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8.4,
                    color=MUTE, style="italic")
        end_labels(ax, ends, lambda v: f"{v:.4f}")
        order = [n for n in ["SURF", "3-layer head", "Closed-form head",
                             "No forget (3-layer head)"] if n in dict(handles)]
        hd = dict(handles)
        fig.legend([hd[n] for n in order], order, loc="lower center", ncol=4,
                   frameon=False, fontsize=8.8, handlelength=2.2)
        fig.subplots_adjust(left=0.11, right=0.7, top=0.86,
                            bottom=0.2)
        fig.savefig(OUT / f"pendulum_gain_depth_{tag[s]}.png")
        plt.close(fig)


def reacher_figs():
    cells = [("wind_ks0.5", "Reacher · wind, k$_s$ = 0.5", "reacher_return_wind_ks0.5.png"),
             ("wind_ks5", "Reacher · wind, k$_s$ = 5", "reacher_return_wind_ks5.png"),
             ("rot90", "Reacher · action rotation 90°", "reacher_return_rot90.png"),
             ("rot150", "Reacher · action rotation 150°", "reacher_return_rot150.png")]
    data = {}
    for prefix, title, fname in cells:
        data[prefix] = reacher(prefix)
        curve_fig(data[prefix], title, REAC_SUB, fname)

    # all four cells in one 2x2 figure: no titles, no end labels, one shared legend
    tags = {"wind_ks0.5": "$k_s$ = 0.5", "wind_ks5": "$k_s$ = 5",
            "rot90": "θ = 90°", "rot150": "θ = 150°"}
    grid_fig([(tags[p], data[p]) for p, _, _ in cells], 2, 2, (9.2, 6.6),
             "reacher_return_all.png")
    # the two harder cells side by side, same size as the 1x2 Pendulum figure
    grid_fig([(tags[p], data[p]) for p in ("wind_ks5", "rot150")], 1, 2, (9.2, 3.9),
             "reacher_return_ks5_rot150.png")

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pendulum_figs()
    reacher_figs()
    for p in sorted(OUT.glob("*.png")):
        print(p.relative_to(ROOT))
