"""Eight-arm cumulative-return plot for the depth x forget 2x2 on Pendulum.

Completes the ablation grid that depth_two_scenarios.html left half-empty:

                 forget ON        forget OFF
  3 layers       head             head_noforget
  1 layer        head_1l          head_1l_noforget   <- added here

Curves are per-trial cumsum of per_step_reward, averaged over trials, with a
+/-1 s.e.m. band -- the same aggregation the original report used.
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["no_adapt", "forget_elbo", "head_lin", "head", "head_1l",
        "head_noforget", "head_1l_noforget", "oracle"]
LABEL = {"no_adapt": "no_adapt", "forget_elbo": "forget_elbo", "head_lin": "head_lin",
         "head": "head — 3 layers", "head_1l": "head_1l — 1 layer",
         "head_noforget": "head_nf — 3L, no forget",
         "head_1l_noforget": "head_1l_nf — 1L, no forget", "oracle": "oracle"}
# 2x2 cells share a hue; the no-forget partner is the dashed/lighter twin.
STYLE = {"no_adapt":         dict(c="#8a8a8a", ls=":",  lw=1.4),
         "forget_elbo":      dict(c="#b07d3a", ls=":",  lw=1.4),
         "head_lin":         dict(c="#2ca86b", ls="-",  lw=1.9),
         "head":             dict(c="#8b6ff0", ls="-",  lw=2.1),
         "head_noforget":    dict(c="#8b6ff0", ls="--", lw=1.7),
         "head_1l":          dict(c="#3b8ef0", ls="-",  lw=2.1),
         "head_1l_noforget": dict(c="#3b8ef0", ls="--", lw=1.7),
         "oracle":           dict(c="#4a4a4a", ls="-.", lw=1.6)}

SCEN = [("Mass 1 → 4", "results/head1l_mass", "results/nf1l_mass"),
        ("Gravity 10 → 15", "results/head1l_grav15", "results/nf1l_grav15")]


def load(base_dir, new_dir):
    """Merge the 7 stored arms with the newly-run 8th."""
    rows = json.load(open(pathlib.Path(base_dir) / "merged.json"))
    p = pathlib.Path(new_dir) / "merged.json"
    if p.exists():
        rows.update(json.load(open(p)))
    return rows


def curve(trials):
    cum = np.array([np.cumsum(t["per_step_reward"]) for t in trials])
    n = len(cum)
    return cum.mean(0), cum.std(0, ddof=1) / np.sqrt(n), n


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    summary = {}
    for ax, (title, base, new) in zip(axes, SCEN):
        rows = load(base, new)
        summary[title] = {}
        for arm in ARMS:
            if arm not in rows or not rows[arm]:
                print(f"  [skip] {title}: {arm} has no data")
                continue
            mu, sem, n = curve(rows[arm])
            x = np.arange(1, len(mu) + 1)
            st = STYLE[arm]
            ax.plot(x, mu, label=f"{LABEL[arm]}  {mu[-1]:.0f}", **st)
            ax.fill_between(x, mu - sem, mu + sem, color=st["c"], alpha=0.13, lw=0)
            summary[title][arm] = (mu[-1], sem[-1], n)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("timestep"); ax.set_ylabel("cumulative return")
        ax.grid(alpha=0.18, lw=0.6)
        ax.legend(fontsize=8.5, loc="lower left", framealpha=0.92)
    fig.suptitle("Pendulum depth × forget ablation — cumulative return "
                 "(band = ±1 s.e.m., 100 trials)", fontsize=14)
    fig.tight_layout()
    out = "depth_forget_2x2.png"
    fig.savefig(out, dpi=155, bbox_inches="tight")
    print(f"\nwrote {out}")

    for title, d in summary.items():
        print(f"\n=== {title} ===")
        print(f"{'arm':<20}{'n':>5}{'final return':>15}{'SEM':>8}")
        for arm in ARMS:
            if arm in d:
                m, s, n = d[arm]
                print(f"{arm:<20}{n:>5}{m:>15.1f}{s:>8.1f}")
        # the 2x2 itself
        print("  depth x forget grid (final return):")
        for lbl, on, off in [("3 layers", "head", "head_noforget"),
                             ("1 layer", "head_1l", "head_1l_noforget")]:
            if on in d and off in d:
                delta = d[on][0] - d[off][0]
                print(f"    {lbl:<10} forget_on={d[on][0]:8.1f}  "
                      f"forget_off={d[off][0]:8.1f}  gain_from_forget={delta:+8.1f}")


if __name__ == "__main__":
    main()
