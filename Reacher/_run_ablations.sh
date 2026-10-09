#!/bin/bash
# Paper-table ablation columns for the four Reacher cells.
#
# Cells:  wind k_s=0.5, wind k_s=5, rotation 90 deg, rotation 150 deg
#         (identical flags to _run_traces100.sh, so these merge with the
#          existing SFIR / no_adapt / no_forget / oracle numbers)
#
# Column mapping -- unlike Pendulum, two of the three already existed:
#   Retrain full           nl1_forget_qv_body   SFIR head + full-network retrain
#   No retrain             nl1_forget_noelbo    forget fires, head never fit
#   No forget + no retrain nl1_noforget_noelbo  NEW: neither knob (validity cell)
#
# --fmax 0 is REQUIRED on the rotation cells (the driver defaults it to 500 and
# would otherwise run wind AND rotation at once).
#
# Work is CPU-bound (GPU util ~0%), so shards are capped on CPU cores.
# Resumable: every shard writes a .done marker; re-running skips finished work.
set -u
cd /home/yli113/Adapt/Reacher
mkdir -p results/abl100

ARMS="nl1_forget_qv_body,nl1_forget_noelbo,nl1_noforget_noelbo"
NSHARD=5
PER=20

run () {  # tag  seedbase  env-flags...
  local tag=$1 base=$2; shift 2
  [ -f "results/abl100/${tag}.done" ] && { echo "SKIP ${tag}"; return 0; }
  python3 -u test_reacher_default_head.py "$@" --save-traces \
    --trials $PER --trial-len 200 --seed-base "$base" --arms "$ARMS" \
    --out "results/abl100/${tag}.json" > "results/abl100/${tag}.log" 2>&1
  if [ $? -eq 0 ]; then touch "results/abl100/${tag}.done"; echo "DONE ${tag}"
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

echo "ALL_ABL100_DONE"
