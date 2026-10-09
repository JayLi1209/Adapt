#!/bin/bash
# OCCUPANCY RE-RUN: same cells as the table (minus cem_g1), with per-trial state
# sequences now saved in each record ("states") for plot/plot_fl_occupancy.py.
# Window-5 (--drift-window 5) at 99 trials for every planner x p' cell of the
# 10-trial comparison.  Each cell = 11 shards x 9 trials (trials 0..98); env and
# torch seeds depend only on the trial index, so shards == one 99-trial run
# (up to the planner's own sampling RNG stream, which restarts per shard).
#
# Rows (flags identical to the 10-trial runs they scale up):
#   cem      CVaR-CEM, plan gamma 0.95
#   mppi     MPPI,     plan gamma 0.95
#   ilqr     iLQR,     plan gamma 0.95
#   mcts     SAFIR-MCTS, Algorithm 2, 3000 sims, plan gamma 0.99
#   cem_g1   CVaR-CEM, plan gamma 1.0 (the original retain-floor sweep)
# Conventions: [1,0,0] -> [p,(1-p)/2,(1-p)/2] at ts=0, Dirichlet head, U1,
# K_FORGET=1, CVaR alpha 0, NO env discount, NO truncation (--max-steps 1000),
# holes scored 0, seed 0.  CPU only.
set -u
cd /home/yli113/Adapt
PY=/home/yli113/.conda/envs/nsgym/bin/python
OUT=results/win5_n99_occ; mkdir -p $OUT
PAR=${PAR:-70}
NSHARD=11; PER=9

declare -A ROW=(
  [cem]="--planner cem --plan-gamma 0.95"
  [mppi]="--planner mppi --plan-gamma 0.95"
  [ilqr]="--planner ilqr --plan-gamma 0.95"
  [mcts]="--planner mcts --mcts-variant alg2 --mcts-sims 3000 --plan-gamma 0.99"
)

run_one(){ # row p shard flags...
  local R=$1 P=$2 K=$3; shift 3
  local D=$OUT/$R/win5_p$P
  mkdir -p $D
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 $PY -u sweep_unfrozen_layers.py \
      --n-unfrozen 1 --trials $PER --trial-start $(( K * PER )) --seed 0 \
      --max-steps 1000 --cvar-alpha 0.0 --drift-window 5 --change-p $P "$@" \
      --out-dir $D/shard$K > $D/shard$K.out 2>&1
  echo "$(date '+%F %T') done $R p=$P shard$K: $(grep -h 'goal rate' $D/shard$K.out | tail -1 | grep -oE 'goal rate [0-9.]+')"
}
export -f run_one; export PY OUT PER

echo "$(date '+%F %T') win5 n99: ${#ROW[@]} rows x 5 p x $NSHARD shards x $PER trials"
for P in 0.9 0.7 0.5 0.3 0.1; do          # slowest cells first
  for R in cem mcts mppi ilqr; do
    for K in $(seq 0 $(( NSHARD - 1 ))); do echo "$R $P $K ${ROW[$R]}"; done
  done
done | xargs -P $PAR -L 1 bash -c 'run_one "$@"' _
echo "$(date '+%F %T') ALL DONE"
