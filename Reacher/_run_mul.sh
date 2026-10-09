#!/bin/bash
# nl1_forget_qv with MULTIPLICATIVE composition, wind + rotation.
set -u
cd /home/yli113/Adapt/Reacher
python3 -u test_reacher_default_head.py --fmax 500 --trials 5 --trial-len 200 \
  --seed-base 1000 --arms nl1_forget_qv_mul \
  --out results/mul_wind_f500.json > results/mul_wind_f500.log 2>&1
for TH in 30 60 90; do
  python3 -u test_reacher_default_head.py --rot $TH --trials 5 --trial-len 200 \
    --seed-base 1000 --arms nl1_forget_qv_mul \
    --out results/mul_rot${TH}.json > results/mul_rot${TH}.log 2>&1
done
echo "ALL_MUL_DONE"
