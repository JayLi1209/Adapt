"""Run ONE job: a contiguous block of trials of one (setting, arm, planner).

    python src/worker.py <job.json>

Job fields (written by common.py): cell, mass, gravity, arm, planner, trials
(0-based trial indices; seed = 1000 + index), rng_skip_acts, out.

REPRODUCIBILITY.  Every trial re-seeds torch (seed + 10000) and the env (seed),
but the PLANNER's numpy Generator is created once per process, as
default_rng(0), and carried across every trial (and arm) that process runs.
The stored results were produced by processes that ran several arms x 20-25
trials back-to-back, so trial i of an arm saw the planner RNG advanced by every
real step taken before it in its shard.  CEM and MPPI draw exactly
n_iters * J * H * A standard normals per real step whatever the data, so that
state is recreated EXACTLY by drawing and discarding rng_skip_acts * that many
normals up front -- which is what lets common.py split the work freely and
still regenerate the stored per-seed returns.  MCTS and iLQR draw a
data-dependent amount, so their jobs always replay an original shard from its
first trial (rng_skip_acts = 0).
"""
import json, os, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import torch

import experiment as X

FIXED_DRAW_PLANNERS = ("cem", "mppi")
TRACE_KEYS = ("dbar_trace", "dn_trace", "fire_trace", "theta_trace", "w_trace",
              "perr_trace")


def fast_forward(rng, n_acts, per_act):
    n = int(n_acts) * int(per_act)
    while n:
        k = min(n, 1 << 24)
        rng.standard_normal(k)
        n -= k


def main():
    job = json.load(open(sys.argv[1]))
    out = pathlib.Path(job["out"])
    bnn, dyn, init_state, agent, w0 = X.setup(job["planner"], job["mass"], job["gravity"])
    if job["rng_skip_acts"]:
        if job["planner"] not in FIXED_DRAW_PLANNERS:
            raise ValueError(f"{job['planner']} cannot be fast-forwarded")
        t = time.time()
        fast_forward(agent.rng, job["rng_skip_acts"],
                     X.CEM_ITERS * X.CANDIDATES * X.H * 1)
        print(f"planner RNG fast-forwarded {job['rng_skip_acts']} real steps "
              f"[{time.time() - t:.0f}s]", flush=True)
    print(f"{job['cell']} {job['arm']} {job['planner']}  mass {job['mass']} gravity "
          f"{job['gravity']}  trials {job['trials'][0]}..{job['trials'][-1]}  "
          f"device {X.device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})",
          flush=True)
    rows, t0 = [], time.time()
    for i in job["trials"]:
        r = X.run_trial(job["arm"], X.SEED_BASE + i, bnn, dyn, init_state, agent, w0)
        for k in TRACE_KEYS:
            r.pop(k, None)
        rows.append(r)
        print(f"  trial {i:3d} seed {X.SEED_BASE + i} ret={r['ret']:9.2f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    tmp = out.with_suffix(".tmp")
    json.dump(dict(job=job, rows=rows), open(tmp, "w"))
    tmp.replace(out)                       # atomic: a killed job leaves no result
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
