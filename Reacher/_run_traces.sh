#!/bin/bash
# Trace re-run: regenerate the headline cells with --save-traces so cumulative
# return curves (mean +/- s.e.m. over seeds) can be plotted per timestep.
#
# Returns are bit-identical to the earlier runs -- --save-traces only KEEPS the
# per-step rew/dists/perr the loop already accumulated.  Verified: sum(rew_t)
# == ret, and a with/without pair matched exactly.
#
# Cells chosen as the two faults' most informative operating points:
#   wind  k_s=0.5  -- where nl1_forget_qv wins (+8.32, t=+6.60 at n=100)
#   wind  k_s=15   -- where it loses (-3.34), the old calibration point
#   rot   150 deg  -- where act_rot wins biggest (+62.5, 79.5% of headroom)
#   rot   30 deg   -- below the crossover, where adaptation does not pay
# --fmax 0 REQUIRED on the rotation lines (driver default is 500).
#
# 40 seeds on wind (enough for tight s.e.m. bands, ~2.5x cheaper than 100),
# 20 on rotation.  Sharded by seed range; trials are pure functions of seed.
set -u
cd /home/yli113/Adapt/Reacher
mkdir -p results/traces

WIND_ARMS="oracle,no_adapt,nl1_forget_qv,lin1_noforget"
ROT_ARMS="oracle,no_adapt,act_rot,nl1_forget_qv"

run () {  # tag arms nseeds seedbase env-flags...
  local tag=$1 arms=$2 n=$3 base=$4; shift 4
  [ -f "results/traces/${tag}.done" ] && { echo "SKIP ${tag}"; return 0; }
  python3 -u test_reacher_default_head.py "$@" --save-traces \
    --trials "$n" --trial-len 200 --seed-base "$base" --arms "$arms" \
    --out "results/traces/${tag}.json" > "results/traces/${tag}.log" 2>&1
  if [ $? -eq 0 ]; then touch "results/traces/${tag}.done"; echo "DONE ${tag}"
  else echo "FAIL ${tag}"; fi
  return 0
}

echo "=== wind cells (k_s 0.5 and 15), 4 shards x 10 seeds each ==="
for i in 0 1 2 3; do
  run "wind_ks0.5_s$((1000+i*10))" "$WIND_ARMS" 10 $((1000+i*10)) --fmax 500 --wind-ks 0.5 &
done
for i in 0 1 2 3; do
  run "wind_ks15_s$((1000+i*10))"  "$WIND_ARMS" 10 $((1000+i*10)) --fmax 500 --wind-ks 15 &
done
wait
echo "=== rotation cells (150 and 30 deg), 2 shards x 10 seeds each ==="
for i in 0 1; do
  run "rot150_s$((1000+i*10))" "$ROT_ARMS" 10 $((1000+i*10)) --fmax 0 --rot 150 &
  run "rot30_s$((1000+i*10))"  "$ROT_ARMS" 10 $((1000+i*10)) --fmax 0 --rot 30 &
done
wait
echo "ALL_TRACES_DONE"
