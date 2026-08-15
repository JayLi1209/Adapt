"""Shared runtime configuration for the risk-averse CVaR-CEM FrozenLake demo.

These constants were previously scattered across bnn_fl_cem_surprise_drift.py; they
are collected here so every sub-package reads them from one place.
"""
import pathlib

import torch

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── Checkpoints ───────────────────────────────────────────────────────────────
# Self-contained: the pretrained Dirichlet head (and the Gaussian trunk used as a
# fallback) live under repo/data/frozenlake.
SAVE_DIR = pathlib.Path(__file__).parent / "data" / "frozenlake"

# ── Drift-filter calibration (shared by both DriftFilter variants) ────────────
ETA = 0.2                   # EWMA rate for the v1 filter; window ~ (2-ETA)/ETA ~ 9
GAMMA_UNCERTAINTY = True    # track the second moment so a lambda_sd is available
KAPPA = 0.0                 # risk-aversion weight on lambda_sd in drift_estimate
