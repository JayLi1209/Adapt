#!/bin/bash
# SURF (head_1l) x planner, the four Pendulum cells of the paper table.
#   planners  mppi / mcts / ilqr  (planning/continuous_planners.py); SURF+CEM is
#             NOT rerun -- the stored head_1l arm is reproduced bit-for-bit by
#             the refactored CEM (seed 1000, m=0.4: -1.8041675126777177).
#   budget    every planner capped at CEM's I*J*K*H = 8*500*10*40 = 1.6M
#             imagined transitions per real step (iLQR spends ~96%).
#   setting   200 steps, gamma 0.99, K=10, CVaR alpha 1.0, H=40, |u|<=2,
#             seeds 1000-1099, k_forget 5 -- identical to the table runs.
#   layout    results/planners/<cell>/<planner>/shard_s<start>.json, 10 trials
#             per shard, MAXJOBS concurrent (~0.5-0.9 GB GPU each), longest
#             planner first so the tail of the queue is short jobs.
#   merge     python3 aggregate_planners.py
cd /home/yli113/Adapt/Pendulum
MAXJOBS=${MAXJOBS:-12}
STEP=10
mkdir -p logs/planners
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run() {  # run <planner> <cell> <envflags> <start>
  local out="results/planners/$2/$1" tag="s$4"
  mkdir -p "$out"
  if [ -f "$out/shard_$tag.json" ]; then echo "skip $1 $2 $tag"; return; fi
  python3 test_pendulum_default_head.py --trials 100 --arms head_1l \
    --planner "$1" --trial-start "$4" --trial-end "$(( $4 + STEP ))" $3 \
    --out-dir "$out" --tag "$tag" > "logs/planners/$1_$2_$tag.log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') done $1 $2 $tag (exit $rc)"
}
declare -A FLAGS=([mass4]="--mass 4.0 --gravity 10.0" [mass0p4]="--mass 0.4 --gravity 10.0"
                  [grav15]="--mass 1.0 --gravity 15.0" [grav5]="--mass 1.0 --gravity 5.0")
echo "$(date '+%F %T') start, MAXJOBS=$MAXJOBS"
for planner in ilqr mcts mppi; do
  for s in $(seq 0 $STEP 99); do
    for cell in mass4 grav15 mass0p4 grav5; do
      while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
      run "$planner" "$cell" "${FLAGS[$cell]}" "$s" &
    done
  done
done
wait
echo "$(date '+%F %T') ALL_SHARDS_COMPLETE"
