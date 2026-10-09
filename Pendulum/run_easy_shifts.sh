#!/bin/bash
# "Easy-direction" Pendulum shifts, 100 trials each: m 1->0.4 and g 10->5.
#   SAFIR (head_1l) plus the no_adapt floor and oracle ceiling, seeds
#   1000-1099, sharded 5 x 20 per arm, capped at 5 concurrent processes
#   (~4.6 GB/proc on the single ~24 GB GPU).  Same config as the table runs.
cd /home/yli113/Adapt/Pendulum
MAXJOBS=5
run() {  # run <outdir> <tag> <arm> <start> <end> <envflags>
  python3 test_pendulum_default_head.py --trials 100 \
    --trial-start "$4" --trial-end "$5" --arms "$3" \
    $6 --out-dir "$1" --tag "$2" > "logs/$2.log" 2>&1
  echo "done $2"
}
for arm in head_1l no_adapt oracle; do
  for sc in mass grav; do
    if [ $sc = mass ]; then OUT=results/easy_mass0p4; FL="--mass 0.4 --gravity 10.0"
    else OUT=results/easy_grav5; FL="--mass 1.0 --gravity 5.0"; fi
    for s in 0 20 40 60 80; do
      while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
      run "$OUT" "easy_${sc}_${arm}_s$s" "$arm" "$s" "$((s+20))" "$FL" &
    done
  done
done
wait
echo "ALL_SHARDS_COMPLETE"
