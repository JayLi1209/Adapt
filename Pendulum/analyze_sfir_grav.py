"""Analysis for the paper-faithful SFIR (forget_elbo) gravity 10->15 run."""
import glob, json, math
import numpy as np
from scipy import stats

OUT = "results/sfir_true_grav15"


def load(pat):
    rows = {}
    for f in sorted(glob.glob(pat)):
        for arm, rs in json.load(open(f)).items():
            rows.setdefault(arm, []).extend(rs)
    return rows


def summ(r):
    r = np.asarray(r, dtype=float)
    n = len(r)
    return n, r.mean(), r.std(ddof=1) / math.sqrt(n), r.std(ddof=1)


def main():
    new = load(f"{OUT}/shard_g*.json")["forget_elbo"]
    new = sorted(new, key=lambda x: x["seed"])
    r = np.array([x["ret"] for x in new])
    n, m, sem, sd = summ(r)
    print(f"SFIR (forget_elbo), gravity 10->15, {new[0]['steps']} steps")
    print(f"  n={n}  return = {m:.1f} +/- {sem:.1f} (SEM)   sd={sd:.1f}")
    ci = stats.t.interval(0.95, n - 1, loc=m, scale=sem)
    print(f"  95% CI [{ci[0]:.1f}, {ci[1]:.1f}]")
    print(f"  pred_err  = {np.mean([x['pred_err'] for x in new]):.4f}")
    print(f"  n_forget  = {np.mean([x['n_forget'] for x in new]):.2f}")
    print(f"  balanced  = {np.mean([x['balanced'] is not None for x in new]):.2f}")
    print(f"  min|theta|= {np.mean([x['min_abs_theta'] for x in new]):.3f}")

    # consistency check against the stored run with identical config
    try:
        old = load("results/head1l_grav15/merged.json")["forget_elbo"]
        old = {x["seed"]: x["ret"] for x in old}
        k = [x["seed"] for x in new if x["seed"] in old]
        if k:
            A = np.array([x["ret"] for x in new if x["seed"] in old])
            B = np.array([old[s] for s in k])
            d = A - B
            p = stats.ttest_rel(A, B).pvalue
            print(f"\nvs stored forget_elbo run (same seeds, n={len(k)}):")
            print(f"  stored {B.mean():.1f}   new {A.mean():.1f}   "
                  f"diff {d.mean():+.1f}  p={p:.3f}")
    except Exception as e:
        print("stored-run comparison unavailable:", e)

    # per-step traces
    L = min(len(x["per_step_reward"]) for x in new)
    R = np.array([x["per_step_reward"][:L] for x in new])
    D = np.array([x["dbar_trace"][:L] for x in new]) if new[0]["dbar_trace"] else None
    print(f"\nper-step log: {len(new)} trials x {L} steps")
    print("  step   reward(mean)     dbar[thdot]")
    for t in [0, 1, 4, 9, 24, 49, 99, 199]:
        if t < L:
            db = f"{D[:, t].mean():12.3f}" if D is not None else "          --"
            print(f"  {t:4d}   {R[:, t].mean():12.3f}   {db}")

    np.save(f"{OUT}/per_step_reward.npy", R)
    if D is not None:
        np.save(f"{OUT}/dbar_trace.npy", D)
    json.dump(
        dict(arm="forget_elbo", setting="gravity 10->15", steps=new[0]["steps"],
             n=n, mean=float(m), sem=float(sem), sd=float(sd),
             ci95=[float(ci[0]), float(ci[1])],
             pred_err=float(np.mean([x["pred_err"] for x in new])),
             n_forget=float(np.mean([x["n_forget"] for x in new])),
             trials=[{k: x[k] for k in ("seed", "ret", "pred_err", "n_forget",
                                        "balanced", "min_abs_theta")} for x in new]),
        open(f"{OUT}/summary.json", "w"), indent=1)
    print(f"\nwrote {OUT}/summary.json, per_step_reward.npy, dbar_trace.npy")


if __name__ == "__main__":
    main()
