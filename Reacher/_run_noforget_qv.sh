#!/bin/bash
# Scope-MATCHED "No forget" column for the four Reacher cells.
#
# The table currently fills "No forget" with nl1_noforget (all-dims), while SFIR
# is nl1_forget_qv (qvel-restricted surprise + ELBO active set).  That confounds
# the forgetting knob with the restriction, the same way Pendulum's "No forget"
# confounded it with depth.  nl1_noforget_qv is the arm that differs from SFIR in
# forgetting ALONE -- it exists in the driver but had never been run.
#
# Waits for the ablation driver so the two jobs never oversubscribe the cores.
# Same flags/seed grid as _run_traces100.sh, so results merge with everything.
set -u
cd /home/yli113/Adapt/Reacher
mkdir -p results/nfqv100

while pgrep -f _run_ablations.sh >/dev/null; do sleep 120; done

ARMS="nl1_noforget_qv"
NSHARD=5
PER=20

run () {  # tag  seedbase  env-flags...
  local tag=$1 base=$2; shift 2
  [ -f "results/nfqv100/${tag}.done" ] && { echo "SKIP ${tag}"; return 0; }
  python3 -u test_reacher_default_head.py "$@" --save-traces \
    --trials $PER --trial-len 200 --seed-base "$base" --arms "$ARMS" \
    --out "results/nfqv100/${tag}.json" > "results/nfqv100/${tag}.log" 2>&1
  if [ $? -eq 0 ]; then touch "results/nfqv100/${tag}.done"; echo "DONE ${tag}"
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

echo "ALL_NFQV100_DONE"
