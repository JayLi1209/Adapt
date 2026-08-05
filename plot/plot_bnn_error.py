"""
plot_bnn_error.py -- per-state BNN error vs timestep, averaged over trials.

Reads the verbose sweep logs written by sweep_alpha.py, which now emit a
machine-parseable per-timestep line for every one of the 16 states:

    PDIR trial=<t> step=<k> | 00:i,pm,pp 01:i,pm,pp ... 15:i,pm,pp

where (i, pm, pp) is the model's directional predictive p_dir = [intend, perp-,
perp+] for that state. The BNN error is the L-infinity distance to the true
post-change slip distribution p = [0.7, 0.15, 0.15]:

    err(state, step, trial) = || p_hat - p ||_inf

For each state we average this error across the 100 trials at each timestep and
plot error (y) vs timestep (x). One figure per alpha log, written to
results/graph1/.

Usage:
    python plot_bnn_error.py                 # all results/alpha_*.log
    python plot_bnn_error.py results/alpha_0p5.log [more.log ...]
"""
import re
import sys
import pathlib
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = _HERE.parent / "results"      # sweep logs live at the repo root
OUT_DIR = RESULTS_DIR / "graph1"

TRUE_P_DIR = np.array([0.7, 0.15, 0.15], dtype=np.float64)
N_STATES = 16

# Standard gym 4x4 FrozenLake map. Hole ('H') and goal ('G') cells are terminal, so
# the BNN never learns a slip distribution there -- exclude them from the plots.
FL_MAP = "SFFF" "FHFH" "FFFH" "HFFG"
EXCLUDE_STATES = {i for i, c in enumerate(FL_MAP) if c in "HG"}   # {5,7,11,12,15}

_LINE_RE = re.compile(r"^PDIR trial=(\d+) step=(\d+) \| (.*)$")
_STATE_RE = re.compile(r"(\d{2}):([-\d.]+),([-\d.]+),([-\d.]+)")


def parse_log(path):
    """Return errs[state] -> dict(step -> list of per-trial L-inf errors)."""
    errs = [defaultdict(list) for _ in range(N_STATES)]
    with open(path) as f:
        for line in f:
            m = _LINE_RE.match(line.strip())
            if not m:
                continue
            step = int(m.group(2))
            for sm in _STATE_RE.finditer(m.group(3)):
                si = int(sm.group(1))
                p_hat = np.array([float(sm.group(2)), float(sm.group(3)),
                                  float(sm.group(4))], dtype=np.float64)
                err = float(np.abs(p_hat - TRUE_P_DIR).max())
                errs[si][step].append(err)
    return errs


def mean_curve(step_map):
    """(steps, mean_err, n_trials) sorted by step for one state."""
    steps = sorted(step_map)
    mean = np.array([np.mean(step_map[k]) for k in steps])
    n = np.array([len(step_map[k]) for k in steps])
    return np.array(steps), mean, n


def plot_log(path):
    path = pathlib.Path(path)
    errs = parse_log(path)
    if not any(errs[s] for s in range(N_STATES)):
        print(f"[skip] no PDIR lines in {path} (re-run sweep_alpha.py to log them)")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 4x4 grid of per-state error curves (mirrors the FrozenLake layout).
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
    all_steps = set()
    for si in range(N_STATES):
        ax = axes[si // 4][si % 4]
        if si in EXCLUDE_STATES:
            ax.set_title(f"s{si} ({FL_MAP[si]})", fontsize=9)
            ax.axis("off")
            continue
        if not errs[si]:
            ax.set_title(f"s{si} (unvisited)", fontsize=9)
            ax.grid(True, alpha=0.3)
            continue
        steps, mean, n = mean_curve(errs[si])
        all_steps.update(steps.tolist())
        ax.plot(steps, mean, marker=".", ms=4, lw=1.3)
        ax.set_title(f"s{si}  (max n={int(n.max())})", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(0.05, TRUE_P_DIR.max()))
    for si in range(N_STATES):
        ax = axes[si // 4][si % 4]
        if si // 4 == 3:
            ax.set_xlabel("timestep")
        if si % 4 == 0:
            ax.set_ylabel(r"$\|\hat p - p\|_\infty$")
    fig.suptitle(f"Per-state BNN error vs timestep (avg over trials)  --  {path.name}\n"
                 r"error = $\|\hat p_{dir} - [0.7,0.15,0.15]\|_\infty$", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    grid_out = OUT_DIR / f"{path.stem}_bnn_error_grid.png"
    fig.savefig(grid_out, dpi=120)
    plt.close(fig)

    # Overlay: all states + the across-state mean on one axis.
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    per_step_all = defaultdict(list)
    for si in range(N_STATES):
        if si in EXCLUDE_STATES or not errs[si]:
            continue
        steps, mean, _ = mean_curve(errs[si])
        ax2.plot(steps, mean, lw=0.8, alpha=0.35)
        for k, v in errs[si].items():
            per_step_all[k].extend(v)
    if per_step_all:
        ks = sorted(per_step_all)
        overall = np.array([np.mean(per_step_all[k]) for k in ks])
        ax2.plot(ks, overall, color="black", lw=2.5, label="mean over states")
    ax2.set_xlabel("timestep")
    ax2.set_ylabel(r"BNN error  $\|\hat p - p\|_\infty$")
    ax2.set_title(f"BNN error vs timestep (avg over trials)  --  {path.name}")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    overlay_out = OUT_DIR / f"{path.stem}_bnn_error_overlay.png"
    fig2.savefig(overlay_out, dpi=120)
    plt.close(fig2)

    print(f"[ok] {path.name}: wrote {grid_out.name}, {overlay_out.name}")


def main():
    args = sys.argv[1:]
    logs = [pathlib.Path(a) for a in args] if args else sorted(
        RESULTS_DIR.glob("alpha_*.log"))
    if not logs:
        print("No logs found. Run sweep_alpha.py first (or pass log paths).")
        return
    for log in logs:
        plot_log(log)


if __name__ == "__main__":
    main()
