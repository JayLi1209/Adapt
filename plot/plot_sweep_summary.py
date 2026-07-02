"""
plot_sweep_summary.py -- expected timestep & expected return vs alpha.

Reads the sweep logs written by sweep_alpha.py (one per CVaR alpha) and plots a
dual-axis figure:
    x axis     : CVaR tail fraction alpha
    left y      : expected (trial-averaged) episode length in timesteps
    right y     : expected (trial-averaged) return

The values come from each log's final summary line
    avg steps=<..> | avg return=<..> | goal rate=<..>
and the alpha from its header line
    SWEEP alpha=<..> | gamma(discount)=<..>

Output: results/graph2/sweep_alpha_summary.png

Usage:
    python plot_sweep_summary.py                 # all results/alpha_*.log
    python plot_sweep_summary.py results/alpha_0p5.log [more.log ...]
"""
import re
import sys
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).parent
RESULTS_DIR = _HERE / "results"
OUT_DIR = RESULTS_DIR / "graph2"

# Discount factor for the return -- MUST match GAMMA in sweep_alpha.py (used for
# planning, decision-making and the return).
GAMMA = 0.99

_ALPHA_RE = re.compile(r"SWEEP alpha=([-\d.]+)")
_TRIAL_END_RE = re.compile(r"=> TRIAL \d+ end: (\w+) in (\d+) steps")


def parse_log(path):
    """(alpha, avg_steps, avg_return, goal_rate) or None if the log is incomplete.

    avg_steps  : mean episode length over trials.
    avg_return : mean DISCOUNTED realized return. Trajectories are unaffected by
                 how the return is accumulated, so the discounted return of a trial
                 that reaches the goal in N steps is exactly GAMMA**(N-1) (the goal
                 reward is collected on the final, 0-indexed step N-1); non-goal
                 trials (hole clamped to 0, or truncated) contribute 0. This matches
                 what sweep_alpha.py now logs, and is recomputable from any log.
    goal_rate  : fraction of trials that reached the goal.
    """
    alpha = None
    done = False
    steps, returns, goals = [], [], []
    with open(path) as f:
        for line in f:
            m = _ALPHA_RE.search(line)
            if m:
                alpha = float(m.group(1))
            m = _TRIAL_END_RE.search(line)
            if m:
                outcome, n = m.group(1), int(m.group(2))
                steps.append(n)
                is_goal = (outcome == "goal")
                goals.append(1 if is_goal else 0)
                returns.append(GAMMA ** (n - 1) if is_goal else 0.0)
            if line.startswith("DONE"):
                done = True
    if alpha is None or not steps or not done:
        return None
    return (alpha, float(np.mean(steps)), float(np.mean(returns)),
            float(np.mean(goals)))


def main():
    args = sys.argv[1:]
    logs = [pathlib.Path(a) for a in args] if args else sorted(
        RESULTS_DIR.glob("alpha_*.log"))
    if not logs:
        print("No logs found. Run sweep_alpha.py first (or pass log paths).")
        return

    rows = []
    for log in logs:
        parsed = parse_log(log)
        if parsed is None:
            print(f"[skip] {log.name}: no complete summary line "
                  "(sweep may not have finished)")
            continue
        rows.append(parsed)

    if not rows:
        print("No complete sweep logs to plot.")
        return

    rows.sort(key=lambda r: r[0])
    alphas = np.array([r[0] for r in rows])
    steps = np.array([r[1] for r in rows])
    returns = np.array([r[2] for r in rows])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    c_steps, c_return = "tab:blue", "tab:red"
    fig, ax_l = plt.subplots(figsize=(9, 6))
    ax_r = ax_l.twinx()

    l1, = ax_l.plot(alphas, steps, marker="o", color=c_steps, lw=2,
                    label="expected timestep")
    l2, = ax_r.plot(alphas, returns, marker="s", color=c_return, lw=2,
                    label="expected return")

    ax_l.set_xlabel(r"CVaR tail fraction $\alpha$")
    ax_l.set_ylabel("expected timestep (avg over trials)", color=c_steps)
    ax_r.set_ylabel("expected return (avg over trials)", color=c_return)
    ax_l.tick_params(axis="y", labelcolor=c_steps)
    ax_r.tick_params(axis="y", labelcolor=c_return)
    ax_l.set_xticks(alphas)
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(handles=[l1, l2], loc="best")
    ax_l.set_title("Expected timestep and return vs alpha")

    fig.tight_layout()
    out = OUT_DIR / "sweep_alpha_summary.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[ok] wrote {out}")
    for a, s, r, g in rows:
        print(f"    alpha={a:<5} avg_steps={s:7.3f} avg_return={r:6.3f} goal_rate={g}")


if __name__ == "__main__":
    main()
