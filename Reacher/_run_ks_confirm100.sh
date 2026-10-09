#!/bin/bash
# 100-seed confirmation of the nl1_forget_qv advantage at low wind frequency.
#
# The 3-seed sweep (results/ks_sweep) showed nl1_forget_qv beating no_adapt by
# +13.9 at k_s=0.5 and +12.3 at k_s=1, 3/3 wins each, but t=+1.79/+1.68 at n=3
# is not significant.  This runs 100 seeds (1000..1099) on the three arms that
# the claim depends on: nl1_forget_qv, no_adapt (the comparison) and oracle
# (the headroom denominator).
#
# SHARDING.  run_trial() does bnn.load_state_dict(init_state) and
# torch.manual_seed(seed+10000) at entry, so a trial is a pure function of its
# seed: splitting seeds across processes is exactly equivalent to one long run.
# Work is CPU-bound (MuJoCo + CEM; GPU util measured at 0%), 16 cores, and each
# process takes ~3 cores, so 5 shards x 20 seeds keeps the box busy without
# oversubscribing.  Cells run sequentially (k_s=0.5 then k_s=1) so the first
# result is complete and usable early rather than both arriving half-done.
#
# --fmax 500 explicit on every line (driver default is 500; being explicit is
# what keeps a stray rotation/wind mix-up from recurring).
set -u
cd /home/yli113/Adapt/Reacher
mkdir -p results/ks100

ARMS="nl1_forget_qv,no_adapt,oracle"
NSHARD=5
PERSHARD=20

run_shard () {
  local ks=$1 lo=$2
  local tag="ks${ks}_s${lo}"
  [ -f "results/ks100/${tag}.done" ] && { echo "SKIP ${tag}"; return 0; }
  python3 -u test_reacher_default_head.py \
    --fmax 500 --wind-ks "$ks" \
    --trials $PERSHARD --trial-len 200 --seed-base "$lo" \
    --arms "$ARMS" \
    --out "results/ks100/${tag}.json" > "results/ks100/${tag}.log" 2>&1
  if [ $? -eq 0 ]; then touch "results/ks100/${tag}.done"; echo "DONE ${tag}"
  else echo "FAIL ${tag}"; fi
  return 0
}

for KS in 0.5 1; do
  echo "=== START k_s=${KS} : ${NSHARD} shards x ${PERSHARD} seeds ==="
  for i in $(seq 0 $((NSHARD - 1))); do
    run_shard "$KS" $((1000 + i * PERSHARD)) &
  done
  wait
  echo "=== DONE k_s=${KS} ==="
done
echo "ALL_KS100_DONE"
