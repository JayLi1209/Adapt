#!/bin/bash
# Paper-table ablation columns for both Pendulum settings.
#   retrain_full / no_retrain / no_forget_no_retrain, 100 trials each,
#   seeds 1000-1099, sharded 5 x 20 and capped at 3 concurrent processes so
#   the single GPU (~18 GB free, ~4.6 GB/proc) is not oversubscribed.
cd /home/yli113/Adapt/Pendulum
ARMS=retrain_full,no_retrain,no_forget_no_retrain
MAXJOBS=3
run() {  # run <outdir> <tag> <start> <end> <envflag>
  python3 test_pendulum_default_head.py --trials 100 \
    --trial-start "$3" --trial-end "$4" --arms "$ARMS" \
    $5 --out-dir "$1" --tag "$2" > "logs/$2.log" 2>&1
  echo "done $2"
}
for s in 0 20 40 60 80; do
  e=$((s+20))
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
  run results/abl_mass "mass_s$s" "$s" "$e" "--mass 4.0" &
done
for s in 0 20 40 60 80; do
  e=$((s+20))
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
  run results/abl_grav "grav_s$s" "$s" "$e" "--mass 1.0 --gravity 15.0" &
done
wait
echo "ALL_SHARDS_COMPLETE"
