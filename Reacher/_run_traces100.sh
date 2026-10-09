#!/bin/bash
# Cumulative-return traces: 4 cells x 5 arms x 100 seeds (1000..1099).
#
# Cells:  wind k_s=0.5, wind k_s=5, rotation 90 deg, rotation 150 deg
# Arms:   nl1_forget_qv, nl1_noforget, no_adapt, forget_elbo, oracle
#
# --save-traces keeps the per-step rew/dists/perr the rollout loop already
# accumulates, so cumulative-return curves can be plotted.  Returns are
# bit-identical with the flag on or off (verified) -- this reproduces the
# existing experiment, it does not redefine it.
#
# --fmax 0 is REQUIRED on the rotation cells: the driver defaults --fmax to 500
# and would otherwise run wind AND rotation simultaneously.
#
# SHARDING: run_trial() does load_state_dict(init_state) + manual_seed(seed)
# per trial, so a trial is a pure function of its seed and splitting the seed
# range across processes is exactly equivalent to one long run.  Work is
# CPU-bound (GPU util 0%).  5 shards x 20 seeds per cell, cells run
# SEQUENTIALLY so each completed cell is usable before the next starts.
#
# Resumable: every shard writes a .done marker; re-running skips finished work.
set -u
cd /home/yli113/Adapt/Reacher
mkdir -p results/traces100

ARMS="nl1_forget_qv,nl1_noforget,no_adapt,forget_elbo,oracle"
NSHARD=5
PER=20

run () {  # tag  seedbase  env-flags...
  local tag=$1 base=$2; shift 2
  [ -f "results/traces100/${tag}.done" ] && { echo "SKIP ${tag}"; return 0; }
  python3 -u test_reacher_default_head.py "$@" --save-traces \
    --trials $PER --trial-len 200 --seed-base "$base" --arms "$ARMS" \
    --out "results/traces100/${tag}.json" > "results/traces100/${tag}.log" 2>&1
  if [ $? -eq 0 ]; then touch "results/traces100/${tag}.done"; echo "DONE ${tag}"
  else echo "FAIL ${tag}"; fi
  return 0
}

cell () {  # cellname  env-flags...
  local name=$1; shift
  echo "=== START ${name} :: $* :: ${NSHARD}x${PER} seeds $(date '+%H:%M') ==="
  for i in $(seq 0 $((NSHARD-1))); do
    run "${name}_s$((1000+i*PER))" $((1000+i*PER)) "$@" &
  done
  wait
  echo "=== END ${name} $(date '+%H:%M') ==="
}

cell wind_ks0.5 --fmax 500 --wind-ks 0.5
cell wind_ks5   --fmax 500 --wind-ks 5
cell rot90      --fmax 0   --rot 90
cell rot150     --fmax 0   --rot 150

echo "ALL_TRACES100_DONE"
