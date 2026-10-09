"""SURF paired with different planners, all at CEM's compute budget.

    python planners/run_planners.py
    python planners/run_planners.py --trials 2   # quick check

Every planner optimises the same objective (CVaR over K=10 posterior draws of
the discounted 40-step imagined return) under the same budget of
8 x 500 x 10 x 40 = 1.6M imagined transitions per real step; only the search
procedure differs.  Implementations: src/planning/continuous_cem.py (CEM-CVaR)
and src/planning/continuous_planners.py (MPPI, MCTS, iLQR).
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import CELLS, execute

COLUMNS = [("CEM-CVaR", "head_1l", "cem"), ("MPPI", "head_1l", "mppi"),
           ("iLQR", "head_1l", "ilqr"), ("MCTS", "head_1l", "mcts")]

if __name__ == "__main__":
    execute(__doc__, [(c, a, p) for c in CELLS for _, a, p in COLUMNS],
            "Pendulum: SURF x planner (mean ± SEM over seeds 1000-1099, equal budget)",
            COLUMNS, list(CELLS), "planner_table.md", paired_with=COLUMNS[0])
