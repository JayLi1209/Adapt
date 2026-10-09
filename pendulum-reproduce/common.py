"""Shared job planner / scheduler / table printer for the three run scripts.

A result is identified by (cell, arm, planner).  Each is computed as a set of
jobs, each job one worker process running a contiguous block of trials, written
to results/<cell>/<arm>__<planner>/t<first>-<end>.json.  Finished jobs are
skipped, so every run script is resumable and the three scripts share results
(e.g. SURF+CEM on m: 1 -> 4 is computed once and used by all three tables).

ORIGINAL LAYOUT.  The stored per-seed returns in reference/ came from processes
that each ran `shard_size` consecutive trials of several arms in a fixed order
(see src/worker.py for why that matters).  layout() records, for every result,
that shard size and the arm's position in the shard, which fixes how far the
planner RNG had advanced when each trial started.
"""
import argparse, json, math, os, pathlib, subprocess, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REFERENCE = json.load(open(ROOT / "reference" / "paper_returns.json"))
SEED_BASE, N_TRIALS, TRIAL_LEN = 1000, 100, 200

# cell -> (table label, deployment mass, deployment gravity); pretrained at m=1, g=10
CELLS = {"mass4":   ("m: 1 -> 4",   4.0, 10.0),
         "mass0p4": ("m: 1 -> 0.4", 0.4, 10.0),
         "grav15":  ("g: 10 -> 15", 1.0, 15.0),
         "grav5":   ("g: 10 -> 5",  1.0, 5.0)}

# arm order inside the original processes that produced the stored runs
HEAD1L_SHARD = ["no_adapt", "forget_elbo", "head_lin", "head", "head_1l",
                "head_noforget", "oracle"]                      # 4 x 25 trials
ABL_SHARD = ["retrain_full", "no_retrain", "no_forget_no_retrain"]  # 5 x 20 trials
FIXED_DRAW = ("cem", "mppi")              # planner RNG use independent of the data
COST = {"cem": 1.0, "mppi": 1.0, "mcts": 2.0, "ilqr": 3.3}   # relative s/trial
CHUNK = 5                                 # trials per job when fast-forwardable


def layout(cell, arm, planner):
    """(shard_size, position of this arm in its shard) of the stored run."""
    if planner != "cem":
        return 10, 0                      # planners/: one arm, 10 trials per process
    if cell in ("mass4", "grav15"):
        if arm in HEAD1L_SHARD:
            return 25, HEAD1L_SHARD.index(arm)
        if arm == "head_1l_noforget":
            return 25, 0
        if arm in ABL_SHARD:
            return 20, ABL_SHARD.index(arm)
    elif arm in ("head_1l", "no_adapt", "oracle"):
        return 20, 0                      # easy shifts: one arm per process, 5 x 20
    raise KeyError(f"no stored layout for {cell}/{arm}/{planner}")


def plan_jobs(cell, arm, planner, n_trials):
    S, pos = layout(cell, arm, planner)
    label, mass, grav = CELLS[cell]
    jobs = []
    for s0 in range(0, n_trials, S):                     # one original shard at a time
        s1 = min(s0 + S, n_trials)
        step = CHUNK if planner in FIXED_DRAW else S     # MCTS/iLQR: replay whole shard
        for a in range(s0, s1, step):
            b = min(a + step, s1)
            skip = (pos * S + (a - s0)) * TRIAL_LEN
            out = RESULTS / cell / f"{arm}__{planner}" / f"t{a:03d}-{b:03d}.json"
            jobs.append(dict(cell=cell, mass=mass, gravity=grav, arm=arm, planner=planner,
                             trials=list(range(a, b)), rng_skip_acts=skip, out=str(out)))
    return jobs


# ── GPU-aware scheduler ───────────────────────────────────────────────────────
def gpu_slots(mb_per_job=1500, max_per_gpu=12):
    """{gpu index: concurrent jobs}, from free memory (most-free GPUs first)."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {None: 1}                                   # CPU fallback
    free = sorted(((int(i), int(m)) for i, m in
                   (l.split(",") for l in q.strip().splitlines())), key=lambda x: -x[1])
    slots = {i: min(max_per_gpu, m // mb_per_job) for i, m in free}
    return {i: n for i, n in slots.items() if n > 0} or {free[0][0]: 1}


def run_jobs(jobs, max_jobs=None):
    todo = [j for j in jobs if not pathlib.Path(j["out"]).exists()]
    if not todo:
        return
    slots = gpu_slots()
    cap = max_jobs or max(1, min(sum(slots.values()), (os.cpu_count() or 2) - 2))
    print(f"{len(todo)} jobs to run ({len(jobs) - len(todo)} already done); "
          f"GPU slots {slots}, running {cap} at a time", flush=True)
    todo.sort(key=lambda j: -COST[j["planner"]] * len(j["trials"]))   # longest first
    logs = RESULTS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running, done, failed, t0 = [], 0, [], time.time()
    while todo or running:
        while todo and len(running) < cap:
            load = {g: sum(1 for _, gg, _ in running if gg == g) for g in slots}
            g = min(slots, key=lambda k: load[k] / slots[k])
            j = todo.pop(0)
            out = pathlib.Path(j["out"]); out.parent.mkdir(parents=True, exist_ok=True)
            name = f"{j['cell']}_{j['arm']}_{j['planner']}_{out.stem}"
            spec = logs / f"{name}.job.json"
            json.dump(j, open(spec, "w"))
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            if g is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            p = subprocess.Popen([sys.executable, str(ROOT / "src" / "worker.py"), str(spec)],
                                 stdout=open(logs / f"{name}.log", "w"),
                                 stderr=subprocess.STDOUT, env=env)
            running.append((p, g, name))
        time.sleep(5)
        for item in list(running):
            p, g, name = item
            if p.poll() is not None:
                running.remove(item)
                done += 1
                if p.returncode:
                    failed.append(name)
                print(f"  [{time.time() - t0:7.0f}s] {done}/{done + len(todo) + len(running)} "
                      f"{name} {'FAILED (see results/logs)' if p.returncode else 'ok'}",
                      flush=True)
    if failed:
        sys.exit(f"{len(failed)} job(s) failed: {failed}")


# ── results ───────────────────────────────────────────────────────────────────
def load(cell, arm, planner):
    rows = {}
    for f in sorted((RESULTS / cell / f"{arm}__{planner}").glob("t*.json")):
        for r in json.load(open(f))["rows"]:
            rows[r["seed"]] = r
    return rows


def stored(cell, arm, planner):
    return {int(s): v for s, v in REFERENCE[cell].get(f"{arm}__{planner}", {}).items()}


def mean_sem(x):
    x = np.asarray(x, dtype=float)
    return (x.mean(), x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else (x.mean(), float("nan"))


def cell_text(rows):
    if not rows:
        return "-"
    m, s = mean_sem([r["ret"] for r in rows.values()])
    return f"{m:.2f} ± {s:.2f}" + ("" if len(rows) == N_TRIALS else f" (n={len(rows)})")


def print_table(title, columns, cells, fname, paired_with=None):
    """columns: [(header, arm, planner)].  Writes results/<fname> as markdown and
    prints it, followed by a per-seed check against the stored paper runs."""
    lines = [f"### {title}", "",
             "| Setting | " + " | ".join(h for h, _, _ in columns) + " |",
             "|---|" + "---|" * len(columns)]
    checks = []
    for cell in cells:
        cols = []
        for h, arm, pl in columns:
            rows = load(cell, arm, pl)
            cols.append(cell_text(rows))
            ref = stored(cell, arm, pl)
            common = [s for s in rows if s in ref]
            if common:
                d = np.abs([rows[s]["ret"] - ref[s] for s in common])
                rm = np.mean([ref[s] for s in common])
                checks.append(f"| {CELLS[cell][0]} | {h} | {len(common)} | "
                              f"{np.mean([rows[s]['ret'] for s in common]):.2f} | {rm:.2f} | "
                              f"{d.max():.3g} | {int((d < 1e-6).sum())}/{len(common)} |")
        lines.append(f"| {CELLS[cell][0]} | " + " | ".join(cols) + " |")
    if paired_with:
        base_h, base_arm, base_pl = paired_with
        lines += ["", f"Paired difference vs {base_h} (same seeds; mean ± SEM, p from a paired t-test):", "",
                  "| Setting | " + " | ".join(h for h, a, p in columns if (a, p) != (base_arm, base_pl)) + " |",
                  "|---|" + "---|" * (len(columns) - 1)]
        from scipy import stats
        for cell in cells:
            base = load(cell, base_arm, base_pl)
            cols = []
            for h, arm, pl in columns:
                if (arm, pl) == (base_arm, base_pl):
                    continue
                rows = load(cell, arm, pl)
                sd = [s for s in rows if s in base]
                if len(sd) < 2:
                    cols.append("-"); continue
                d = [rows[s]["ret"] - base[s]["ret"] for s in sd]
                m, se = mean_sem(d)
                p = stats.ttest_rel([rows[s]["ret"] for s in sd], [base[s]["ret"] for s in sd]).pvalue
                cols.append(f"{m:+.1f} ± {se:.1f} (p={p:.1e})")
            lines.append(f"| {CELLS[cell][0]} | " + " | ".join(cols) + " |")
    lines += ["", "Check against the stored paper runs (same seeds):", "",
              "| Setting | Column | n | reproduced mean | stored mean | max abs per-seed diff | bit-identical seeds |",
              "|---|---|---|---|---|---|---|"] + checks
    text = "\n".join(lines) + "\n"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / fname).write_text(text)
    print("\n" + text)
    print(f"(written to {RESULTS / fname})")


def cli(description):
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=N_TRIALS,
                    help="first N seeds of 1000-1099 (default all 100); a small N is a quick check")
    ap.add_argument("--jobs", type=int, default=None,
                    help="concurrent worker processes (default: from free GPU memory and CPU count)")
    ap.add_argument("--table-only", action="store_true",
                    help="print the table from whatever results exist, run nothing")
    return ap.parse_args()


def execute(description, specs, title, columns, cells, fname, paired_with=None):
    """specs: [(cell, arm, planner)] to compute; then print the table."""
    A = cli(description)
    if not A.table_only:
        jobs = [j for c, a, p in specs for j in plan_jobs(c, a, p, A.trials)]
        run_jobs(jobs, A.jobs)
    print_table(title, columns, cells, fname, paired_with)
