#!/bin/bash
for TH in 30 60; do
  CUDA_VISIBLE_DEVICES=0 python3 -u test_reacher_default_head.py \
    --fmax 0 --rot $TH --trials 20 --trial-len 200 \
    --arms oracle,no_adapt,lin1_discount,lin1_noforget,forget_elbo \
    --out results/rot${TH}_20trials.json > results/rot${TH}_20trials.log 2>&1
  echo "=== rot $TH done ==="
done
