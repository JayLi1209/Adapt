#!/bin/bash
# ns-Bridge 5x8 adaptation ablation at PLAN-GAMMA 0.95: p 1.0 -> {0.9, 0.7, 0.5} at ts=0.
# Six arms + full-method reference x three change levels x 100 trials.
#
# WHY plan_gamma=0.95 (the whole point of this run):
#   At plan_gamma=1.0 every route that reaches a goal is worth exactly 1.0, so the
#   short-but-lethal RIGHT arm (cells 21,22 -- holes directly above AND below) and
#   the long-but-safe LEFT arm TIE under the p=1.0 prior, and the agent takes the
#   safe one by tie-break.  That is why no_adapt scored 0.91 vs a 0.91 oracle in
#   results/bridge_ablation and adaptation looked harmful.
#   Any discount < 1 makes the p=1.0-optimal policy strictly prefer the SHORT right
#   arm, which slip then makes lethal -- restoring the intended short-risky vs
#   long-safe tradeoff.  Exact MDP values for the stale p=1.0 policy at pg=0.95:
#     p'=0.9 -> 0.806 | p'=0.7 -> 0.469 | p'=0.5 -> 0.229
#   vs oracle 0.994 / 0.934 / 0.749.  Everything else matches run_bridge_ablation.sh.
#   score gamma = 1.0, PLAN GAMMA = 0.95, no step cost, alpha = 0, K_FORGET = 1,
#   retrain lr = 1e-3 (50 inner steps), clip OFF, H = default(6), uncapped
#   forgets, --max-steps 1000 (nothing truncates), seed 0, holes scored 0
#   (so mean return == goal rate).
#
# The seven conditions (matching the FrozenLake / CliffWalking ablation table):
#   both                 --n-unfrozen 1               (forget + counts + retrain)
#   no_forget            --no-forget --n-unfrozen 1   (retrain + counts, NO forget)
#   no_retrain           --n-unfrozen 0               (forget + counts, NO retrain)
#   no_forget_no_retrain --no-forget --n-unfrozen 0   (counts only)
#   retrain_full         --n-unfrozen 3               (head + BOTH trunk layers)
#   no_adapt             --no-forget --n-unfrozen 0 --no-counts  (nothing adapts)
#   oracle               value iteration on the TRUE post-change kernel
set -u
cd "$(dirname "$0")"
PY=/home/yli113/.conda/envs/nsgym/bin/python
OUT=results/bridge_pg095; mkdir -p $OUT
TRIALS=${TRIALS:-100}
GRID=${GRID:-bridge}
NGPU=8

JOBS=()
for P in 0.9 0.7 0.5; do
  for ARM in both no_forget no_retrain no_forget_no_retrain retrain_full no_adapt oracle; do
    JOBS+=("$ARM|$P")
  done
done

echo "$(date '+%F %T') bridge ablation pg0.95 ($GRID): ${#JOBS[@]} jobs, $TRIALS trials each"

run_job(){
  local SPEC=$1 GPU=$2
  IFS='|' read -r ARM P <<< "$SPEC"
  local NAME="${ARM}_p${P}"
  local C="--grid $GRID --trials $TRIALS --seed 0 --max-steps 1000 --cvar-alpha 0.0 \
           --plan-gamma 0.95 --step-cost 0.0 --retrain-steps 50"
  case $ARM in
    both)
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY -u sweep_bridge_unfrozen.py \
        $C --change-p $P --n-unfrozen 1 --out-dir $OUT/$NAME > $OUT/${NAME}.out 2>&1 ;;
    no_forget)
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY -u sweep_bridge_unfrozen.py \
        $C --change-p $P --n-unfrozen 1 --no-forget --out-dir $OUT/$NAME > $OUT/${NAME}.out 2>&1 ;;
    no_retrain)
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY -u sweep_bridge_unfrozen.py \
        $C --change-p $P --n-unfrozen 0 --out-dir $OUT/$NAME > $OUT/${NAME}.out 2>&1 ;;
    no_forget_no_retrain)
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY -u sweep_bridge_unfrozen.py \
        $C --change-p $P --n-unfrozen 0 --no-forget --out-dir $OUT/$NAME > $OUT/${NAME}.out 2>&1 ;;
    retrain_full)
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY -u sweep_bridge_unfrozen.py \
        $C --change-p $P --n-unfrozen 3 --out-dir $OUT/$NAME > $OUT/${NAME}.out 2>&1 ;;
    no_adapt)
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY -u sweep_bridge_unfrozen.py \
        $C --change-p $P --n-unfrozen 0 --no-forget --no-counts \
        --out-dir $OUT/$NAME > $OUT/${NAME}.out 2>&1 ;;
    oracle)
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY -u oracle_bridge.py \
        --grid $GRID --trials $TRIALS --seed 0 --max-steps 1000 \
        --env-p $P --model-p $P --out-dir $OUT/$NAME > $OUT/${NAME}.out 2>&1 ;;
  esac
  echo "$(date '+%F %T') done $NAME: $(tail -1 $OUT/${NAME}.out)"
}

i=0
for SPEC in "${JOBS[@]}"; do
  GPU=$(( i % NGPU ))
  run_job "$SPEC" "$GPU" &
  i=$(( i + 1 ))
  while [ "$(jobs -rp | wc -l)" -ge "$NGPU" ]; do sleep 5; done
done
wait
echo "$(date '+%F %T') ALL DONE"
