#!/bin/bash
# k_s sweep at F_max = 500, 3 seeds (1000..1002), all 12 arms.
#
# Sweeps the wind spatial frequency k_s in {0.5, 1, 5, 15} holding F_max=500
# fixed.  k_s=15 reproduces results/wind_f500_3seeds.json + the lin1/fe_split
# companions, so it doubles as a regression check on the rest of the sweep.
#
# --fmax 500 is EXPLICIT on every line even though it is the driver default,
# so a future reader never has to guess whether wind was on.
# --wind-ks is the swept axis; the oracle reads env.wind_ks back, so the
# headroom at each k_s is computed against a correctly-informed oracle.
#
# One process per k_s, run concurrently; 12 arms serial inside each.
# Resumable: each cell writes a .done marker; re-running skips finished cells.
set -u
cd /home/yli113/Adapt/Reacher
mkdir -p results/ks_sweep

# The 12 arms of the F=500 table, in the screenshot's order.
ARMS="oracle,no_adapt,lin1_noforget,nl1_noforget,forget_elbo,nl1_forget_qv,nl1_forget_qv_mul,lin1_discount,head_noforget,retrain,head,head_meanroll"

run_ks () {
  local KS=$1
  local tag="ks${KS}"
  if [ -f "results/ks_sweep/${tag}.done" ]; then
    echo "SKIP ${tag} (already done)"; return 0
  fi
  echo "START ${tag}"
  python3 -u test_reacher_default_head.py \
    --fmax 500 --wind-ks "$KS" \
    --trials 3 --trial-len 200 --seed-base 1000 \
    --arms "$ARMS" \
    --out "results/ks_sweep/${tag}.json" > "results/ks_sweep/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "results/ks_sweep/${tag}.done"; echo "DONE ${tag}"
  else echo "FAIL ${tag} rc=${rc}"; fi
  return 0
}

for KS in 0.5 1 5 15; do
  run_ks "$KS" &
done
wait
echo "ALL_KS_SWEEP_DONE"
