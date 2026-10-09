# ns-Bridge

Non-stationary Bridge (Lecarpentier & Rachelson 2019), 5x8, run with the same
adaptation stack as the FrozenLake "0.83" experiment.

## Layout

    Bridge/
      sweep_bridge_unfrozen.py       adaptive runs (forget + counts + gradient retrain)
      eval_bridge_pretrained.py      matched NO-adaptation baseline
      run_bridge_experiment.sh       the 48-job sweep driver
      summarize_bridge_experiment.py pairs each adaptive cell with its baseline
      data/<grid>/*.pth              pretrained checkpoints
      results/                       all runs

Shared modules (`config`, `grids`, `bnn`, `env`, `planning`, `drift`, `utils`)
still live at the repo root; the scripts put the root on `sys.path` themselves,
so they can be run from anywhere.

## The map

    row 0    H H H H H H H H
    row 1    F F F F F H H H
    row 2    G x F F S F F G      <- start s=20, goals s=16 (left) / s=23 (right)
    row 3    F F F F F H H H
    row 4    H H H H H H H H

`x` = cell 17, the one map difference between the two registered grids:

  * `bridge`      cell 17 = F -- the long LEFT arm is safe, slips there recover
  * `bridge_h17`  cell 17 = H -- a hazard on the left arm too, so BOTH routes risk

The right arm is 3 steps but cells 21/22 have holes directly above AND below, so
any perpendicular slip is fatal.  The left arm is 4 steps and (on `bridge`)
survivable.  Short-and-risky vs long-and-safe is the whole point of the env.

Slip is PERPENDICULAR (K=3, `slip_dist(p) = [p, (1-p)/2, (1-p)/2]`), matching the
original ADA-MCTS `nsbridge_v0.py`.  An earlier opposite-slip (K=2) version made
the bridge unfallable -- goal rate was pinned at 1.0 by construction; those old
checkpoints are kept under `data/bridge/k2_opposite_archive/`.

## Pretraining

    python ../FrozenLake/pretrain_gridworld.py --grid bridge     --p 1.0
    python ../FrozenLake/pretrain_gridworld.py --grid bridge_h17 --p 1.0

Note `pretrain_gridworld.py` writes to the repo-root `data/<grid>/`; move the
resulting checkpoint under `Bridge/data/<grid>/`, which is where these scripts
read from.  Both shipped checkpoints verified GOOD (predictive p_dir vs target
MAE 1e-4).

## Scoring conventions (FrozenLake-0.83 convention)

  * reported return scores holes as 0, so mean return == goal rate
  * score gamma = 1.0 (NO discount in the reported number)
  * `--plan-gamma` discounts INSIDE the planner only.  This matters: the leaf
    heuristic is V_h(s) = plan_gamma^dist(s,goal), so at plan_gamma=1.0 it is
    flat 1.0 everywhere and the planner has no gradient toward the goal beyond
    its H=6 horizon.
  * `--step-cost` is likewise charged inside the planner only.
  * no truncation (`--max-steps 1000`), K_FORGET=1, retrain lr=1e-3 x 5 steps.

## Running the sweep

    ./run_bridge_experiment.sh                  # 48 jobs, 50 trials each
    python summarize_bridge_experiment.py --in-dir results/bridge_experiment

## Result so far

p=1.0 -> p'=0.7 at ts=0.  No configuration made adaptation SIGNIFICANTLY beat
no-adaptation (best +0.080, p=0.42).  Two caveats on that number:

  * U0 and U1 are byte-identical in 13/16 cells -- `retain` collapses to the 1e-3
    clamp floor (forget fires ~4.8x/trial), and since
    `alpha = CONC_PRIOR + retain*(alpha_head - CONC_PRIOR)`, a collapsed retain
    multiplies the retrained head out of the prediction.  So the sweep did not
    really test retraining; it tested forget twice.  `--clip-delta-n 50` and
    `--max-forgets 2` are the untested fixes.
  * plan_gamma=0.95 makes adaptation lead on `bridge` but drops absolute
    performance (0.90 -> 0.52): the discount pushes the agent onto the risky arm,
    so everyone does worse and adaptation merely does less badly.
