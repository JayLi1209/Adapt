#!/bin/bash
# Small-batch CVaR alpha sweep. K=5 draws so a 0.3 tail is 2 samples rather
# than 1 (at K=3, alpha<=0.33 collapses to the single worst draw).
for A in 1.0 0.8 0.5 0.3; do
  CUDA_VISIBLE_DEVICES=0 python3 -u test_reacher_default_head.py \
    --fmax 500 --trials 2 --trial-len 200 \
    --alpha $A --k-models 5 \
    --arms no_adapt,lin1_noforget,lin1_discount,forget_elbo \
    --out results/alpha_${A}_f500.json > results/alpha_${A}_f500.log 2>&1
  echo "=== alpha=$A done ==="
done
