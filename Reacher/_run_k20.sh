#!/bin/bash
cd "$(dirname "$0")"; export CUDA_VISIBLE_DEVICES=0
# head-only retrain (the default: BNN body frozen, only the 520-param head trains)
python3 -u test_reacher_default_head.py \
  --fmax 0 --rot 60 --trials 5 --trial-len 200 --k-models 20 \
  --arms nl1_forget_qv \
  --out results/k20_head_rot60.json > results/k20_head_rot60.log 2>&1
echo "=== head-only done ==="
# full-network retrain (body + head)
python3 -u test_reacher_default_head.py \
  --fmax 0 --rot 60 --trials 5 --trial-len 200 --k-models 20 \
  --arms nl1_forget_qv_body \
  --out results/k20_body_rot60.json > results/k20_body_rot60.log 2>&1
echo "=== full-network done ==="
