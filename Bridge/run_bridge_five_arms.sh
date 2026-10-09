#!/bin/bash
# bridge (cells 17 AND 18 both FREE -- the original map), p: 1.0 -> {0.7, 0.9, 0.5} at ts=0.
# Five arms x three change levels x 100 trials, FrozenLake-0.83 settings throughout:
#   score gamma = 1.0, plan gamma = 1.0, no step cost, alpha = 0, K_FORGET = 1,
#   retrain lr = 1e-3 (50 inner steps), uncapped forgets, no truncation, seed 0,
#   holes scored 0 (so mean return == goal rate).
#
# The five arms:
#   no_adapt        : frozen BNN, planner STILL replans each step
#   no_adapt_frozen : policy solved ONCE from the p=1.0 model, executed fixed
#                     (the pure "decisions it pretrained on" floor)
#   oracle   : value iteration on the TRUE post-change kernel (the ceiling)
#   retrain  : --no-forget --n-unfrozen 1   (retrain ONLY)
#   forget   : --n-unfrozen 0               (forget + counts, NO retrain)  == U0
#   both     : --n-unfrozen 1               (forget + counts + retrain)    == U1
set -u
cd "$(dirname "$0")"
PY=/home/yli113/.conda/envs/nsgym/bin/python
OUT=results/bridge_five_arms; mkdir -p $OUT
TRIALS=${TRIALS:-100}
GRID=bridge
NGPU=8

JOBS=()
for P in 0.7 0.9 0.5; do
  JOBS+=("no_adapt|$P")
  JOBS+=("no_adapt_frozen|$P")
  JOBS+=("oracle|$P")
  JOBS+=("retrain|$P")
  JOBS+=("forget|$P")
  JOBS+=("both|$P")
done

echo "$(date '+%F %T') bridge five-arm sweep: ${#JOBS[@]} jobs, $TRIALS trials each"

run_job(){
  local SPEC=$1 GPU=$2
  IFS='|' read -r ARM P <<< "$SPEC"
  local NAME="${ARM}_p${P}"
  local COMMON="--grid $GRID --trials $TRIALS --seed 0 --max-steps 1000 --cvar-alpha 0.0"
  case $ARM in
    no_adapt)
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY -u eval_bridge_pretrained.py \
        $COMMON --env-p $P --model-p 1.0 --plan-gamma 1.0 --step-cost 0.0 \
        --out-dir $OUT/$NAME > $OUT/${NAME}.out 2>&1 ;;
    no_adapt_frozen)
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY -u eval_bridge_pretrained.py \
        $COMMON --env-p $P --model-p 1.0 --plan-gamma 1.0 --step-cost 0.0 \
        --frozen-policy --out-dir $OUT/$NAME > $OUT/${NAME}.out 2>&1 ;;
    oracle)
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY -u oracle_bridge.py \
        --grid $GRID --trials $TRIALS --seed 0 --max-steps 1000 \
        --env-p $P --model-p $P --out-dir $OUT/$NAME > $OUT/${NAME}.out 2>&1 ;;
    retrain)
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY -u sweep_bridge_unfrozen.py \
        $COMMON --n-unfrozen 1 --no-forget --change-p $P --original-p 1.0 \
        --plan-gamma 1.0 --step-cost 0.0 --retrain-steps 50 \
        --out-dir $OUT/$NAME > $OUT/${NAME}.out 2>&1 ;;
    forget)
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY -u sweep_bridge_unfrozen.py \
        $COMMON --n-unfrozen 0 --change-p $P --original-p 1.0 \
        --plan-gamma 1.0 --step-cost 0.0 --retrain-steps 50 \
        --out-dir $OUT/$NAME > $OUT/${NAME}.out 2>&1 ;;
    both)
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY -u sweep_bridge_unfrozen.py \
        $COMMON --n-unfrozen 1 --change-p $P --original-p 1.0 \
        --plan-gamma 1.0 --step-cost 0.0 --retrain-steps 50 \
        --out-dir $OUT/$NAME > $OUT/${NAME}.out 2>&1 ;;
  esac
  echo "$(date '+%F %T') done $NAME: $(grep -h 'saved:' $OUT/${NAME}.out | tail -1)"
}

i=0
for SPEC in "${JOBS[@]}"; do
  run_job "$SPEC" $(( i % NGPU )) &
  i=$(( i + 1 ))
  if (( i % NGPU == 0 )); then wait; fi
done
wait
echo "$(date '+%F %T') ALL DONE"
