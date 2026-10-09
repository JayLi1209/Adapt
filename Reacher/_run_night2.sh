#!/bin/bash
# Part A: nl1_noforget (ELBO only) + nl1_forget_noelbo (forget only) at 30/60/90.
#         Decomposes SFIR into its two halves against the existing
#         nl1_forget_qv / act_rot / no_adapt / oracle results at those angles.
# Part B: all 6 arms at 120 and 150 deg, plus references.
#
# --fmax 0 REQUIRED on every line: the driver defaults to 500 and would add wind.
set -u
cd /home/yli113/Adapt/Reacher
mkdir -p results/night2

run () {
  local tag=$1 arms=$2; shift 2
  [ -f "results/night2/${tag}.done" ] && { echo "SKIP ${tag}"; return 0; }
  echo "START ${tag} :: ${arms}"
  python3 -u test_reacher_default_head.py "$@" --trials 5 --trial-len 200 \
    --seed-base 1000 --arms "$arms" \
    --out "results/night2/${tag}.json" > "results/night2/${tag}.log" 2>&1
  [ $? -eq 0 ] && { touch "results/night2/${tag}.done"; echo "DONE ${tag}"; } \
                || echo "FAIL ${tag}"
  return 0
}

HALVES="nl1_noforget,nl1_forget_noelbo"
ALL6="nl1_noforget,nl1_forget,nl1_forget_noelbo,nl1_forget_qv,act_rot,no_adapt"

# Part A: the two halves at angles already covered
for TH in 30 60 90; do run "halves_rot${TH}" "$HALVES" --fmax 0 --rot $TH; done

# Part B: 120 and 150 -- references first so comparisons survive a later failure
for TH in 120 150; do
  run "ref_rot${TH}"  "oracle,no_adapt" --fmax 0 --rot $TH
  run "all6_rot${TH}" "$ALL6"           --fmax 0 --rot $TH
done
echo "ALL_NIGHT2_DONE"
