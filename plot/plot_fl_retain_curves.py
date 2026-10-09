"""Retain factor r_t over time, one line per post-change slip probability p'.

r_t is the scalar retention state of the Dirichlet head: r_0 = 1 and
r_{t+1} <- r_t * rho every K_FORGET steps whenever forgetting fires, with

    alpha_k <- c + r_t (alpha_k - c) + N_k .

So r_t -> 0 means the pretrained head's concentration has been scaled out and
alpha collapses onto the prior's scale (K*c); r_t = 1 means the belief is
untouched.  Because the update is MULTIPLICATIVE and fires every step, r_t is a
running product and decays geometrically once the drift filter is triggered --
the "ratchet" that makes a small, genuine-but-mild change over-forgettable.

Source: results/fl_changep/U1_p0pX_steps.jsonl, the shipped-floor (rho_min=1e-3),
UNCAPPED (no surprise clip) runs from the deterministic p=1 checkpoint, 100
trials each, CVaR+CEM alpha=0, gamma=1, K_FORGET=1, no truncation.

Trials have different lengths, so at each t the curve averages over the trials
still running at that step; the shaded band is +/- 1 SEM across those trials, and
the curve is drawn only while at least MIN_TRIALS remain (beyond that the mean is
dominated by a handful of long episodes and is not comparable across p').

    python plot/plot_fl_retain_curves.py
"""
import collections
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = pathlib.Path("results/fl_changep")
OUT_LIN = pathlib.Path("plot/fl_retain_curves.png")
OUT_LOG = pathlib.Path("plot/fl_retain_curves_log.png")
ARMS = [("0.1", "U1_p0p1"), ("0.3", "U1_p0p3"), ("0.5", "U1_p0p5"),
        ("0.7", "U1_p0p7"), ("0.9", "U1_p0p9")]
MIN_TRIALS = 10          # stop a curve once fewer than this many trials remain
T_MAX = 400


def load(tag):
    """trial -> list of retain values indexed by step."""
    per = collections.defaultdict(dict)
    path = DIR / f"{tag}_steps.jsonl"
    if not path.exists():
        return None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            per[r["trial"]][r["t"]] = r["retain"]
    return per


def curve(per, t_max=T_MAX):
    """Mean +/- SEM of r_t at each t, over the trials still alive at t."""
    ts, mu, se, n_alive = [], [], [], []
    for t in range(t_max):
        vals = [d[t] for d in per.values() if t in d]
        if len(vals) < MIN_TRIALS:
            break
        v = np.asarray(vals, dtype=float)
        ts.append(t)
        mu.append(v.mean())
        se.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        n_alive.append(len(v))
    return np.array(ts), np.array(mu), np.array(se), np.array(n_alive)


def draw(logy):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(x) for x in np.linspace(0.85, 0.05, len(ARMS))]

    for (label, tag), col in zip(ARMS, colors):
        per = load(tag)
        if per is None:
            print(f"  (missing trace for p'={label})")
            continue
        ts, mu, se, n = curve(per)
        if not len(ts):
            continue
        ax.plot(ts, mu, color=col, lw=1.9, label=f"$p'$ = {label}")
        ax.fill_between(ts, mu - se, mu + se, color=col, alpha=0.18, lw=0)
        print(f"  p'={label}: r_1={mu[0]:.4f}  r_10={mu[10] if len(mu)>10 else float('nan'):.4f}  "
              f"r_50={mu[50] if len(mu)>50 else float('nan'):.4g}  "
              f"curve to t={ts[-1]} ({n[-1]} trials alive)")

    if logy:
        ax.set_yscale("log")
        ax.set_ylim(1e-12, 2.0)
        ax.set_ylabel(r"retain  $r_t$   (log scale)")
    else:
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(r"retain  $r_t$")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("step  $t$  (change applied at $t=0$)")
    ax.set_title("Dirichlet retain factor over time, by post-change slip probability\n"
                 "pretrained $p=1$, shipped floor $\\rho_{\\min}=10^{-3}$, no surprise clip, "
                 "100 trials/arm", fontsize=9.5)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    out = OUT_LOG if logy else OUT_LIN
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")
    plt.close(fig)


def draw_zoom(t_zoom=30):
    """The decay all happens in the first ~25 steps; show that window alone."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(x) for x in np.linspace(0.85, 0.05, len(ARMS))]

    for ax, logy in zip(axes, (False, True)):
        for (label, tag), col in zip(ARMS, colors):
            per = load(tag)
            if per is None:
                continue
            ts, mu, se, _ = curve(per, t_max=t_zoom)
            if not len(ts):
                continue
            ax.plot(ts, mu, color=col, lw=2.0, marker="o", ms=3,
                    label=f"$p'$ = {label}")
            ax.fill_between(ts, mu - se, mu + se, color=col, alpha=0.18, lw=0)
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("step  $t$")
        ax.grid(alpha=0.25, lw=0.6)
        if logy:
            ax.set_yscale("log")
            ax.set_ylabel(r"retain  $r_t$   (log)")
            ax.set_ylim(1e-12, 2.0)
        else:
            ax.set_ylabel(r"retain  $r_t$")
            ax.set_ylim(-0.02, 1.02)
            ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = pathlib.Path("plot/fl_retain_curves_zoom.png")
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")
    plt.close(fig)


def main():
    print("linear:")
    draw(logy=False)
    print("log:")
    draw(logy=True)
    print("zoom:")
    draw_zoom()
    print(f"\ncurves truncated once fewer than {MIN_TRIALS} trials remain alive,")
    print("so late-time values are not dominated by a few long episodes.")


if __name__ == "__main__":
    main()
