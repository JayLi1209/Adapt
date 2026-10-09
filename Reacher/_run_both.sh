#!/bin/bash
# (B) FIXED actuator saturation tau = 0.4*tanh(a/0.25), no wind
CUDA_VISIBLE_DEVICES=0 python3 -u test_reacher_default_head.py \
  --fmax 0 --sat 0.4,0.25 --trials 50 --trial-len 200 \
  --arms oracle,no_adapt,lin1_discount,lin1_noforget,forget_elbo \
  --out results/sat_g04_a025_50trials.json > results/sat_g04_a025_50trials.log 2>&1
echo "=== saturation run done ==="
# (A) original non-linear wind, F_max=500
CUDA_VISIBLE_DEVICES=0 python3 -u test_reacher_default_head.py \
  --fmax 500 --trials 50 --trial-len 200 \
  --arms oracle,no_adapt,lin1_discount,lin1_noforget,forget_elbo \
  --out results/wind_f500_50trials.json > results/wind_f500_50trials.log 2>&1
echo "=== wind run done ==="
