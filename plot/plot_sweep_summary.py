"""
plot_sweep_summary.py -- graph2: expected episode length, expected discounted
return, and episode-outcome fractions vs the CVaR tail fraction alpha.

Reads every sweep log written by sweep_alpha.py (results/alpha_*.log), pools the
per-trial outcomes across seeds for each (variant, alpha), and draws a
three-panel figure sharing the alpha axis:

  1. expected episode length (timesteps), mean +/- 95% CI over trials
  2. expected discounted return,          mean +/- 95% CI over trials
  3. fraction of episodes ending in goal / hole / truncation

(The old dual-axis steps/return chart is split into panels 1 and 2: two y-scales
on one plot invite reading fake crossings, and the error bars need their own
scale to be legible.)

If matched-baseline logs (alpha_*_matched*.log, from `sweep_alpha.py --matched`)
are present, the matched sweep is overlaid dashed, so "does alpha matter when
the policy is NOT mismatched?" is read directly off the same panels.

Per-trial values are re-derived from each log's
    => TRIAL <k> end: <outcome> in <N> steps
lines, so logs written before the outcome-fraction summary line existed plot
identically.  The discounted return of a goal trial that ends in N steps is
exactly GAMMA**(N-1) (goal reward collected on the final, 0-indexed step); hole
and truncated trials contribute 0.

Output: results/graph2/sweep_alpha_summary.png (+ a stdout table).

Usage:
    python plot/plot_sweep_summary.py                  # all results/alpha_*.log
    python plot/plot_sweep_summary.py results/alpha_0p5.log [more.log ...]
"""
import re
import sys
import pathlib
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

_HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = _HERE.parent / "results"
OUT_DIR = RESULTS_DIR / "graph2"

# Discount factor for the return -- MUST match GAMMA in sweep_alpha.py (used for
# planning, decision-making and the return).
GAMMA = 0.99

_ALPHA_RE = re.compile(r"SWEEP alpha=([-\d.]+)")
_SEED_RE = re.compile(r"\bseed=(\d+)")
_VARIANT_RE = re.compile(r"\bvariant=(\w+)")
_TRIAL_END_RE = re.compile(r"=> TRIAL \d+ end: (\w+) in (\d+) steps")

OUTCOMES = ("goal", "hole", "truncated")
VARIANTS = ("mismatched", "matched")

# ── palette (light surface) ────────────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # titles
INK_2 = "#52514e"        # axis labels, legend text
INK_MUTED = "#898781"    # tick labels
GRID = "#e1e0d9"         # hairline gridlines
AXIS = "#c3c2b7"         # baseline spines
VARIANT_COLOR = {"mismatched": "#2a78d6", "matched": "#1baf7a"}
VARIANT_LS = {"mismatched": "-", "matched": "--"}
VARIANT_LABEL = {"mismatched": "mismatched (pretrained on deterministic)",
                 "matched": "matched (true slippery dynamics)"}
# Outcome states use the reserved status palette, not series hues.
OUTCOME_COLOR = {"goal": "#0ca30c", "hole": "#d03b3b", "truncated": "#898781"}


def parse_log(path):
    """One log -> {alpha, seed, variant, steps[], outcomes[]} or None if the run
    is incomplete (no DONE sentinel).

    seed/variant come from the SWEEP header line when present (new logs) and
    fall back to the filename / legacy defaults (seed 0, mismatched) otherwise.
    """
    alpha, seed, variant = None, 0, "mismatched"
    done = False
    steps, outcomes = [], []
    with open(path) as f:
        for line in f:
            m = _ALPHA_RE.search(line)
            if m:
                alpha = float(m.group(1))
                ms = _SEED_RE.search(line)
                if ms:
                    seed = int(ms.group(1))
                mv = _VARIANT_RE.search(line)
                if mv:
                    variant = mv.group(1)
            m = _TRIAL_END_RE.search(line)
            if m:
                outcomes.append(m.group(1))
                steps.append(int(m.group(2)))
            if line.startswith("DONE"):
                done = True
    if alpha is None or not steps or not done:
        return None
    name = pathlib.Path(path).name
    if "_matched" in name:
        variant = "matched"
    ms = re.search(r"_seed(\d+)", name)
    if ms:
        seed = int(ms.group(1))
    return {"alpha": alpha, "seed": seed, "variant": variant,
            "steps": np.array(steps, dtype=np.float64),
            "outcomes": np.array(outcomes)}


def mean_ci95(x):
    """(mean, half-width of the 95% CI) over trials, normal approximation."""
    x = np.asarray(x, dtype=np.float64)
    m = float(x.mean())
    if len(x) < 2:
        return m, 0.0
    return m, float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def aggregate(records):
    """Pool trials across seeds -> {variant: {alpha: stats dict}}.

    stats: n, n_seeds, (mean, ci) for steps / return / each outcome fraction.
    """
    pooled = defaultdict(lambda: defaultdict(lambda: {"steps": [], "outcomes": [],
                                                      "seeds": set()}))
    for r in records:
        g = pooled[r["variant"]][r["alpha"]]
        g["steps"].append(r["steps"])
        g["outcomes"].append(r["outcomes"])
        g["seeds"].add(r["seed"])

    out = {}
    for variant, by_alpha in pooled.items():
        out[variant] = {}
        for alpha, g in by_alpha.items():
            steps = np.concatenate(g["steps"])
            outcomes = np.concatenate(g["outcomes"])
            is_goal = outcomes == "goal"
            # Discounted return, recomputed identically for old and new logs.
            returns = np.where(is_goal, GAMMA ** (steps - 1), 0.0)
            stats = {"n": len(steps), "n_seeds": len(g["seeds"]),
                     "steps": mean_ci95(steps), "return": mean_ci95(returns)}
            for o in OUTCOMES:
                stats[o] = mean_ci95((outcomes == o).astype(np.float64))
            out[variant][alpha] = stats
    return out


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel(r"CVaR tail fraction $\alpha$", color=INK_2, fontsize=10)


def _errbar(ax, x, series, color, ls, label, mfc=None):
    """One mean +/- 95% CI line: 2px line, ringed >=8px markers, capped bars."""
    mean = np.array([v[0] for v in series])
    ci = np.array([v[1] for v in series])
    ax.errorbar(x, mean, yerr=ci, color=color, ls=ls, lw=1.8,
                marker="o", ms=6, mfc=mfc or color, mec=SURFACE, mew=1.0,
                capsize=2.5, elinewidth=1.1, label=label, zorder=3)


def plot_summary(agg, out_path):
    variants = [v for v in VARIANTS if v in agg]
    n_variants = len(variants)

    fig, (ax_steps, ax_ret, ax_frac) = plt.subplots(
        1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    for ax in (ax_steps, ax_ret, ax_frac):
        _style_axis(ax)

    for variant in variants:
        by_alpha = agg[variant]
        alphas = np.array(sorted(by_alpha))
        stats = [by_alpha[a] for a in alphas]
        color, ls = VARIANT_COLOR[variant], VARIANT_LS[variant]
        label = VARIANT_LABEL[variant] if n_variants > 1 else None
        _errbar(ax_steps, alphas, [s["steps"] for s in stats], color, ls, label)
        _errbar(ax_ret, alphas, [s["return"] for s in stats], color, ls, label)
        # Outcome panel: color = outcome (status), linestyle/fill = variant.
        mfc = SURFACE if variant == "matched" else None
        for o in OUTCOMES:
            _errbar(ax_frac, alphas, [s[o] for s in stats], OUTCOME_COLOR[o],
                    ls, o if variant == variants[0] else None, mfc=mfc)

    n_trials = sorted({s["n"] for v in agg.values() for s in v.values()})
    n_note = "/".join(str(n) for n in n_trials)
    ax_steps.set_title("Expected episode length", color=INK, fontsize=11)
    ax_steps.set_ylabel("timesteps (mean ± 95% CI)", color=INK_2, fontsize=10)
    ax_steps.set_ylim(bottom=0)
    ax_ret.set_title("Expected discounted return", color=INK, fontsize=11)
    ax_ret.set_ylabel(f"return, $\\gamma$={GAMMA} (mean ± 95% CI)",
                      color=INK_2, fontsize=10)
    ax_ret.set_ylim(bottom=0)
    ax_frac.set_title("Episode outcome fractions", color=INK, fontsize=11)
    ax_frac.set_ylabel("fraction of trials (mean ± 95% CI)", color=INK_2,
                       fontsize=10)
    ax_frac.set_ylim(-0.03, 1.03)

    if n_variants > 1:
        for ax in (ax_steps, ax_ret):
            leg = ax.legend(loc="best", fontsize=8, frameon=False)
            for t in leg.get_texts():
                t.set_color(INK_2)
    # Outcome legend; add linestyle keys for the variants when both are present.
    handles, labels = ax_frac.get_legend_handles_labels()
    if n_variants > 1:
        for variant in variants:
            handles.append(Line2D([], [], color=INK_MUTED, ls=VARIANT_LS[variant],
                                  lw=1.8))
            labels.append(variant)
    leg = ax_frac.legend(handles, labels, loc="best", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle(f"CVaR-CEM on FrozenLake, 0.7-slippery at t=0 — "
                 f"{n_note} trials per $\\alpha$ (pooled over seeds)",
                 color=INK, fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def print_table(agg):
    for variant in (v for v in VARIANTS if v in agg):
        print(f"  variant={variant}")
        for alpha in sorted(agg[variant]):
            s = agg[variant][alpha]
            print(f"    alpha={alpha:<5} n={s['n']:<4} seeds={s['n_seeds']} | "
                  f"steps={s['steps'][0]:7.3f}±{s['steps'][1]:5.3f} | "
                  f"return={s['return'][0]:.3f}±{s['return'][1]:.3f} | "
                  f"goal={s['goal'][0]:.3f}±{s['goal'][1]:.3f} "
                  f"hole={s['hole'][0]:.3f}±{s['hole'][1]:.3f} "
                  f"trunc={s['truncated'][0]:.3f}±{s['truncated'][1]:.3f}")


def main():
    args = sys.argv[1:]
    logs = [pathlib.Path(a) for a in args] if args else sorted(
        RESULTS_DIR.glob("alpha_*.log"))
    if not logs:
        print("No logs found. Run sweep_alpha.py first (or pass log paths).")
        return

    records = []
    for log in logs:
        parsed = parse_log(log)
        if parsed is None:
            print(f"[skip] {log.name}: no complete summary "
                  "(sweep may not have finished)")
            continue
        records.append(parsed)

    if not records:
        print("No complete sweep logs to plot.")
        return

    agg = aggregate(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "sweep_alpha_summary.png"
    plot_summary(agg, out)
    print(f"[ok] wrote {out}")
    print_table(agg)


if __name__ == "__main__":
    main()
