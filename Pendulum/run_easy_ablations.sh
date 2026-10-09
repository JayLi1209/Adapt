#!/bin/bash
# The four ablation arms still missing on the easy-direction shifts, so the
# ablation figure can show all four Pendulum settings:
#   retrain_full / no_retrain / no_forget_no_retrain / head_1l_noforget
#   x  m 1->0.4 (results/abl_mass0p4)  and  g 10->5 (results/abl_grav5).
# SURF / no_adapt / oracle for these settings already exist in easy_mass0p4/,
# easy_grav5/.  Same config as every table run (200 steps, gamma 0.99, CEM+CVaR
# H=40 K=10 alpha=1.0, k_forget 5), seeds 1000-1099, ONE ARM PER PROCESS in
# 5 x 20-trial shards -- the easy_* layout, so the planner RNG of trial i is
# default_rng(0) advanced by (i mod 20) trials.
cd /home/yli113/Adapt/Pendulum
HEAVY=${HEAVY:-3} LIGHT=${LIGHT:-6}
mkdir -p logs/easy_abl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run() {  # run <outdir> <tag> <arm> <start> <envflags>
  if [ -f "$1/shard_$2.json" ]; then echo "skip $2"; return; fi
  python3 test_pendulum_default_head.py --trials 100 --arms "$3" \
    --trial-start "$4" --trial-end "$(( $4 + 20 ))" $5 \
    --out-dir "$1" --tag "$2" > "logs/easy_abl/$2.log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') done $2 (exit $rc)"
}
echo "$(date '+%F %T') start: retrain_full <= $HEAVY concurrent, other arms <= $LIGHT"
launch() {  # launch <cap> <arms...>: queue every shard of these arms, at most <cap> at once
  local cap=$1; shift
  for arm in "$@"; do
    for s in 0 20 40 60 80; do
      for sc in mass grav; do
        if [ $sc = mass ]; then OUT=results/abl_mass0p4; FL="--mass 0.4 --gravity 10.0"
        else OUT=results/abl_grav5; FL="--mass 1.0 --gravity 5.0"; fi
        while [ "$(jobs -rp | wc -l)" -ge "$cap" ]; do wait -n; done
        run "$OUT" "easy_${sc}_${arm}_s$s" "$arm" "$s" "$FL" &
      done
    done
  done
  wait
}
# retrain_full refits the whole network on its growing buffer every step and
# peaks at ~4-5 GB per process (12 at once OOMed a 24 GB card); the others ~1 GB.
launch "$HEAVY" retrain_full &
launch "$LIGHT" head_1l_noforget no_retrain no_forget_no_retrain &
wait
echo "$(date '+%F %T') ALL_SHARDS_COMPLETE"
