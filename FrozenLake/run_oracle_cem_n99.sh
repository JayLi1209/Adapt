#!/bin/bash
# ORACLE CVaR-CEM (planning/oracle_cem.py): SURF's planner -- same I=5, J=256,
# K=10, N=32, H=6 => 2,457,600 simulated transitions per real step -- planning
# on the TRUE post-change kernel instead of the BNN.  99 trials per p' (trials
# 0..98, the table's n), same seeds / CVaR alpha / planning discount as the SURF
# cells it sits next to:
#   FrozenLake : sweep_unfrozen_layers.py   (results/fl_clip2_n99 settings)
#   NS-Bridge  : Bridge/sweep_bridge_unfrozen.py (bridge_five_arms 'both' settings)
# Adaptation is OFF (--n-unfrozen 0 --no-forget): the planner never reads the BNN.
# Conventions: [1,0,0] -> [p',(1-p')/2,(1-p')/2] at ts=0, NO env discount,
# NO truncation (--max-steps 1000), holes scored 0 (return == goal rate), CPU only.
set -u
cd /home/yli113/Adapt
PY=/home/yli113/.conda/envs/nsgym/bin/python
OUT=results/oracle_cem_n99; mkdir -p $OUT
BOUT=Bridge/results/oracle_cem_n99; mkdir -p $BOUT
NSHARD=11; PER=9
COMMON="--oracle --n-unfrozen 0 --no-forget --seed 0 --max-steps 1000 --cvar-alpha 0.0"

for P in 0.9 0.7 0.5 0.3 0.1; do
  for K in $(seq 0 $(( NSHARD - 1 ))); do
    mkdir -p $OUT/fl_p$P
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 $PY -u FrozenLake/sweep_unfrozen_layers.py $COMMON \
      --trials $PER --trial-start $(( K * PER )) --change-p $P \
      --out-dir $OUT/fl_p$P/shard$K > $OUT/fl_p$P/shard$K.out 2>&1 &
  done
  ( cd Bridge && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 $PY -u sweep_bridge_unfrozen.py $COMMON \
      --grid bridge --trials 99 --change-p $P --original-p 1.0 --plan-gamma 1.0 \
      --step-cost 0.0 --out-dir results/oracle_cem_n99/p$P > results/oracle_cem_n99/p$P.out 2>&1 ) &
done
wait
echo "$(date '+%F %T') ALL DONE"
