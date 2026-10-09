#!/bin/bash
# theta sweep at 50 seeds. Arms: the two references (oracle, no_adapt), the
# winning config (lin1_steps50), its combination with L2, the no-forgetting
# baseline it must beat, and forget_elbo as the incumbent scheme.
for TH in 30 60 90; do
  CUDA_VISIBLE_DEVICES=0 python3 -u test_reacher_default_head.py \
    --fmax 0 --rot $TH --trials 50 --trial-len 200 \
    --arms oracle,no_adapt,lin1_noforget,lin1_steps50,lin1_l2_steps50,forget_elbo \
    --out results/sweep_rot${TH}_50trials.json > results/sweep_rot${TH}_50trials.log 2>&1
  echo "=== theta=$TH done ==="
done
echo "ALL DONE"
