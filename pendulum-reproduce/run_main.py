"""Main result: SURF (CEM-CVaR planner) on the four Pendulum shifts.

    python run_main.py              # all 4 settings x 100 seeds, then the table
    python run_main.py --trials 2   # quick check: seeds 1000-1001 only
"""
from common import CELLS, execute

COLUMNS = [("SURF", "head_1l", "cem")]

if __name__ == "__main__":
    execute(__doc__, [(c, a, p) for c in CELLS for _, a, p in COLUMNS],
            "Pendulum: SURF returns (mean ± SEM over seeds 1000-1099)",
            COLUMNS, list(CELLS), "main_table.md")
