"""Ablation table: SURF vs its knock-outs, m: 1 -> 4 and g: 10 -> 15.

    python ablation/run_ablation.py
    python ablation/run_ablation.py --trials 2   # quick check

Columns (arm names in src/experiment.py):
  SURF                    head_1l               1-layer Bayesian adapter head on the frozen
                                                body; surprise-driven forgetting + ELBO refit
  Retrain full            retrain_full          whole-network ELBO refit every step, no forgetting
  No retrain              no_retrain            surprise -> forget/inflate every k_forget, no refit
  No forget + no retrain  no_forget_no_retrain  neither knob (frozen model, same bookkeeping)
  No adapt                no_adapt              frozen pretrained model
  No forget               head_1l_noforget      SURF without the forgetting step (depth-matched)
  Oracle                  oracle                true shifted dynamics inside the same planner
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import execute

COLUMNS = [("SURF", "head_1l", "cem"), ("Retrain full", "retrain_full", "cem"),
           ("No retrain", "no_retrain", "cem"),
           ("No forget + no retrain", "no_forget_no_retrain", "cem"),
           ("No adapt", "no_adapt", "cem"), ("No forget", "head_1l_noforget", "cem"),
           ("Oracle", "oracle", "cem")]
CELLS = ["mass4", "grav15"]

if __name__ == "__main__":
    execute(__doc__, [(c, a, p) for c in CELLS for _, a, p in COLUMNS],
            "Pendulum ablations (mean ± SEM over seeds 1000-1099)",
            COLUMNS, CELLS, "ablation_table.md", paired_with=COLUMNS[0])
