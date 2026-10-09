#!/bin/bash
# nl1_forget_qv MULTIPLICATIVE-on-qvel vs the additive counterpart.
# Defaults throughout: K_RETRAIN=1, uncapped forgets, K_MODELS=3, alpha=0.8,
# 50 ELBO steps, 5 seeds (1000-1004), 200 steps, gamma=1.0.
set -u
cd /home/yli113/Adapt/Reacher
python3 -u test_reacher_default_head.py --fmax 500 --trials 5 --trial-len 200 \
  --seed-base 1000 --arms nl1_forget_qv_mul \
  --out results/mulqv_wind_f500.json > results/mulqv_wind_f500.log 2>&1
echo "WIND_DONE"
python3 -u test_reacher_default_head.py --rot 60 --trials 5 --trial-len 200 \
  --seed-base 1000 --arms nl1_forget_qv_mul,nl1_forget_qv \
  --out results/mulqv_rot60.json > results/mulqv_rot60.log 2>&1
echo "ROT60_DONE"
echo "ALL_MULQV_DONE"
