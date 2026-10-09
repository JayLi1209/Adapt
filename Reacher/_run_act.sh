#!/bin/bash
# act_rot: closed-form action-interface identification vs output-side SFIR.
# --fmax 0 is REQUIRED: the driver defaults to 500, which would add wind on top
# of the rotation and corrupt the qvel residual identification regresses on.
set -u
cd /home/yli113/Adapt/Reacher
for TH in 30 60 90; do
  python3 -u test_reacher_default_head.py --fmax 0 --rot $TH --trials 5 \
    --trial-len 200 --seed-base 1000 \
    --arms act_rot,nl1_forget_qv,no_adapt,oracle \
    --out results/act_rot${TH}.json > results/act_rot${TH}.log 2>&1
  echo "ROT${TH}_DONE"
done
echo "ALL_ACT_DONE"
