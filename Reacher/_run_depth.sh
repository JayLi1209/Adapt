#!/bin/bash
# wait for the in-flight theta=90 sweep to finish
while pgrep -f "test_reacher_default_head.py --fmax 0 --rot 90 --trials 50" >/dev/null; do sleep 60; done
for TH in 30 60 90; do
  CUDA_VISIBLE_DEVICES=0 python3 -u test_reacher_default_head.py \
    --fmax 0 --rot $TH --trials 50 --trial-len 200 \
    --arms oracle,no_adapt,nl1_noforget,nl2_noforget \
    --out results/depth_rot${TH}_50trials.json > results/depth_rot${TH}_50trials.log 2>&1
  echo "=== depth theta=$TH done ==="
done
echo "DEPTH SWEEP DONE"
