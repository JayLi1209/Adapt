"""Table 2 + the ORACLE column (run_oracle_cem_n99.sh).

Oracle = SURF's CVaR-CEM (I=5, J=256, K=10, N=32, H=6 -> 2,457,600 simulated
transitions per real step) planning on the TRUE post-change kernel
(planning/oracle_cem.py).  Goal rate over trials 0..98, SEM = sqrt(p(1-p)/n),
the convention of the other columns.

The other three columns are the published Table 2 values; each was traced to
its run (goal rate over the first 99 trials / seeds):
  SURF      FL : results/fl_clip2_n99/U1_p*_clip2.json            (clip 2)
            NS : Bridge/results/bridge_five_arms|bridge_extreme_p/both_p*/U1_*_rs50.json
  ADA-MCTS  FL : ADA-MCTS repo logs/fl_adamcts{,_p90,_p50,_p30,_p10}_100ep.log
            NS : ADA-MCTS repo logs/bridgeopen_sweep/bo_ada_p*.log, logs/bridge_open_adamcts_1to07_100ep.log
  RATS      FL : ADA-MCTS repo logs/fl_rats{,_p90,_p50,_p30,_p10}_100ep.log
            NS : ADA-MCTS repo results/bridge_rats_bridge_open_p100_to_*_d3.json

    python FrozenLake/summarize_oracle_cem_n99.py
"""
import glob
import json
from math import sqrt

LEVELS = ["0.9", "0.7", "0.5", "0.3", "0.1"]
TABLE = {  # (SURF, ADA-MCTS, RATS) as published
    "Frozen Lake": {"0.9": (0.818, 0.859, 0.889), "0.7": (0.576, 0.535, 0.596),
                    "0.5": (0.677, 0.212, 0.333), "0.3": (0.677, 0.051, 0.061),
                    "0.1": (0.707, 0.111, 0.010)},
    "NS-Bridge": {"0.9": (0.960, 0.939, 0.929), "0.7": (0.778, 0.788, 0.667),
                  "0.5": (0.657, 0.354, 0.333), "0.3": (0.354, 0.141, 0.182),
                  "0.1": (0.283, 0.020, 0.020)},
}
N_TAB = 99


def oracle_records(env, p):
    if env == "Frozen Lake":
        files = sorted(glob.glob(f"results/oracle_cem_n99/fl_p{p}/shard*/oracle_*.json"))
    else:
        files = sorted(glob.glob(f"Bridge/results/oracle_cem_n99/p{p}/oracle_*.json"))
    recs = []
    for f in files:
        recs += json.load(open(f))["records"]
    return recs


def fmt(p, n):
    return "{:.3f} +- {:.3f}".format(p, sqrt(p * (1 - p) / n))


print("{:<12} {:>4}  {:>15} {:>15} {:>15}   {:>22}  {:>9} {:>6}".format(
    "env", "p'", "SURF", "ADA-MCTS", "RATS", "ORACLE (true physics)", "steps", "n"))
for env, cells in TABLE.items():
    for p in LEVELS:
        surf, ada, rats = cells[p]
        recs = oracle_records(env, p)
        if recs:
            g = sum(r["outcome"] == "goal" for r in recs)
            n = len(recs)
            orc = fmt(g / n, n)
            steps = sum(r["steps"] for r in recs) / n
            extra = "{:>9.1f} {:>6}".format(steps, n)
        else:
            orc, extra = "(not yet)", ""
        print("{:<12} {:>4}  {:>15} {:>15} {:>15}   {:>22}  {}".format(
            env, p, fmt(surf, N_TAB), fmt(ada, N_TAB), fmt(rats, N_TAB), orc, extra))
    print()
