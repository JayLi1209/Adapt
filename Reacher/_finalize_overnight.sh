#!/bin/bash
# Waits for _run_ablations.sh to finish, then produces every downstream artefact
# so the finished table is on disk by morning rather than waiting on a session.
#
# Safe to run alongside the driver: it only reads results until the driver is
# done, and every step it runs is idempotent.
set -u
cd /home/yli113/Adapt/Reacher
LOG=results/finalize.log
: > "$LOG"

echo "[$(date '+%F %T')] waiting for ALL_ABL100_DONE" >> "$LOG"
while ! grep -q "ALL_ABL100_DONE" results/ablation_driver.log 2>/dev/null; do
  if ! pgrep -f _run_ablations.sh >/dev/null; then
    echo "[$(date '+%F %T')] driver exited without ALL_ABL100_DONE -- finalizing anyway" >> "$LOG"
    break
  fi
  sleep 120
done

# The scope-matched no-forget arm is queued behind the driver; wait for it too,
# otherwise the table would be generated before its column exists.
echo "[$(date '+%F %T')] ablations done; waiting for nl1_noforget_qv" >> "$LOG"
while pgrep -f _run_noforget_qv.sh >/dev/null; do sleep 120; done

echo "[$(date '+%F %T')] abl shards: $(ls results/abl100/*.done 2>/dev/null | wc -l)/20" >> "$LOG"
echo "[$(date '+%F %T')] nfqv shards: $(ls results/nfqv100/*.done 2>/dev/null | wc -l)/20" >> "$LOG"
grep -c "^FAIL" results/ablation_driver.log >/dev/null 2>&1 && \
  grep "^FAIL" results/ablation_driver.log >> "$LOG"
grep "^FAIL" results/nfqv_driver.log >> "$LOG" 2>/dev/null

echo "=== bounds ===" >> "$LOG"
python3 compute_table_bounds.py >> "$LOG" 2>&1

echo "=== final table ===" >> "$LOG"
python3 make_final_table.py >> "$LOG" 2>&1

echo "[$(date '+%F %T')] FINALIZE_COMPLETE" >> "$LOG"
