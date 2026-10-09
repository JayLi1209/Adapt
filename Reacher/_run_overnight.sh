#!/bin/bash
# Overnight 2x2: {wind F=500, rotation 60} x {additive, multiplicative}
#
# EXPLICIT FLAGS ON EVERY LINE.  --fmax defaults to 500 in the driver, so the
# rotation cells MUST pass --fmax 0 or they run wind and rotation at once (this
# exact omission invalidated an earlier act_rot batch).
#
# Resumable: each cell writes a .done marker; re-running skips finished cells.
set -u
cd /home/yli113/Adapt/Reacher
mkdir -p results/overnight

ADD="nl1_forget_qv,nl1_forget,nl1_noforget"
MUL="nl1_forget_qv_mul,nl1_forget_mul,nl1_noforget_mul"
REF="oracle,no_adapt"

run () {  # $1=tag  $2=arms  $3...=env flags
  local tag=$1 arms=$2; shift 2
  if [ -f "results/overnight/${tag}.done" ]; then
    echo "SKIP ${tag} (already done)"; return 0
  fi
  echo "START ${tag} :: ${arms} :: $*"
  python3 -u test_reacher_default_head.py "$@" --trials 10 --trial-len 200 \
    --seed-base 1000 --arms "$arms" \
    --out "results/overnight/${tag}.json" > "results/overnight/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "results/overnight/${tag}.done"; echo "DONE ${tag}"
  else echo "FAIL ${tag} rc=${rc}"; fi
  return 0            # never abort the queue on one bad cell
}

# References first: cheapest, and every comparison depends on them.
run wind_ref  "$REF" --fmax 500
run rot60_ref "$REF" --fmax 0 --rot 60
# Additive
run wind_add  "$ADD" --fmax 500
run rot60_add "$ADD" --fmax 0 --rot 60
# Multiplicative
run wind_mul  "$MUL" --fmax 500
run rot60_mul "$MUL" --fmax 0 --rot 60

echo "ALL_OVERNIGHT_DONE"
