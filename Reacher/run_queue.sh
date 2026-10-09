#!/bin/bash
# Resumable experiment queue.
#
# Each (experiment, theta) writes a .done marker on success.  Re-running this
# script SKIPS anything already marked done, so it is safe to invoke repeatedly
# -- on boot, from cron, or by hand after a crash.  A machine shutdown therefore
# costs at most the single arm that was in flight, not the whole sweep.
cd "$(dirname "$0")" || exit 1
export CUDA_VISIBLE_DEVICES=0
mkdir -p results/.done

# Single-instance lock: a boot-triggered copy and a hand-run copy must not both
# grab the GPU.  flock releases automatically if this process is killed.
exec 9> results/.queue.lock
if ! flock -n 9; then
  echo "another run_queue.sh holds the lock; exiting"; exit 0
fi

run () {          # run <tag> <extra-args...>
  local tag="$1"; shift
  if [ -f "results/.done/$tag" ]; then
    echo "[skip] $tag already complete"; return 0
  fi
  echo "[run ] $tag  $(date -Is)"
  if python3 -u test_reacher_default_head.py "$@" \
        --out "results/${tag}.json" > "results/${tag}.log" 2>&1; then
    # only mark done if the driver actually wrote its summary
    if grep -q "=== SUMMARY" "results/${tag}.log"; then
      touch "results/.done/$tag"; echo "[done] $tag"
    else
      echo "[FAIL] $tag -- no summary written"; return 1
    fi
  else
    echo "[FAIL] $tag exit=$?"; return 1
  fi
}

# 1) finish the rotation sweep (theta=90 was mid-flight)
run sweep_rot90_50trials --fmax 0 --rot 90 --trials 50 --trial-len 200 \
    --arms oracle,no_adapt,lin1_noforget,lin1_steps50,lin1_l2_steps50,forget_elbo

# 2) depth sweep: 1-layer vs 2-layer NON-LINEAR branch.
#    oracle / no_adapt are NOT re-run: identical-setting 50-seed results already
#    exist in results/sweep_rot{30,60,90}_50trials.json and are reused as the
#    references (same seeds 1000-1049, same planner config).
for TH in 30 60 90; do
  run depth_rot${TH}_50trials --fmax 0 --rot $TH --trials 50 --trial-len 200 \
      --arms nl1_noforget,nl2_noforget
done
# 3) SFIR on the one-layer non-linear head.  Depth-1 is fully compatible with
#    surprise->forget->inflate: forget() acts per-dim on skip and l_nl.
#    nl1_forget  = legacy precision-only inflation
#    nl1_drift   = prior-drift (decays the mean too), the coherent version
#    nl2_forget  = same mechanism on the two-layer head, for the depth contrast
for TH in 30 60 90; do
  run nlforget_rot${TH}_50trials --fmax 0 --rot $TH --trials 50 --trial-len 200 \
      --arms nl1_forget,nl1_drift,nl2_forget
done
echo "QUEUE COMPLETE $(date -Is)"
