# Risk-averse CVaR-CEM planning on non-stationary FrozenLake

Dirichlet-head Bayesian world model with an online surprise → forget → learn loop,
driven by the CEM + CVaR-greedy MPC planner of Ayan's note, on a scheduled
non-stationary 4×4 FrozenLake.

## Run

```bash
conda activate nsgym          # provides ns_gym + mbrl + torch
cd repo
python run_risk_averse.py
```

Outputs (written next to the script):
- `risk_averse_ayan.log` — per-trial summary log
- `risk_averse_ayan_metrics.png` — side-by-side averaged step count / return / goal rate

## Layout

```
run_risk_averse.py      entry point (the trial loop)
config.py               device, SAVE_DIR, drift-filter constants (ETA/KAPPA/...)
utils.py                to_one_hot_action, make_fl_potential
plot.py                 plot_trial_metrics (the 3-panel figure)
bnn/                    Bayesian world model
  layers.py             BayesianLinear (mean-field VI linear layer)
  dirichlet_model.py    DirichletDynamicsModel, make_dirichlet_bnn, geometry, constants
  dirichlet_workflow.py surprise / epistemic / forget / mean_alpha0
planning/               model-based planners
  base.py               BNNModelPlanner (BNN→MDP plumbing + _model_matrices)
  cvar_cem.py           CVaRCEMAgent (the CEM + CVaR planner)
env/
  frozenlake.py         FLOneHotWrapper, build_scheduled_env, slip_to_dist
drift/
  filters.py            DriftFilterV1 (EWMA), DriftFilterV2 (equal-weight signed)
data/frozenlake/
  bnn_dirichlet_k3.pth  pretrained Dirichlet checkpoint (loaded at startup)
  bnn_dynamics.pth      Gaussian trunk (only read if the Dirichlet ckpt is absent)
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install mbrl==0.1.5 --no-deps
pip install -r requirements.txt
python run_risk_averse.py
```
