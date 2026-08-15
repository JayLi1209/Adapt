"""Gaussian-head Bayesian dynamics model + frozen-body adapter heads.

Trimmed relative to the parent repo: the Dirichlet variant is not imported, so
this package has no FrozenLake / ns_gym dependency.
"""
from bnn.layers import BayesianLinear
from bnn.gaussian_model import (
    BayesianDynamicsModel, make_gaussian_bnn, load_arch, GAUSSIAN_CKPT,
)
from bnn.gaussian_workflow import (
    surprise_gaussian, epistemic_gaussian, forget_gaussian, mean_sigma,
    forget_gaussian_perdim, head_row_sigma, anchor_head_rows_to_current,
    retrain_gaussian_full_elbo, INFLATE_MODE,
)
from bnn.gain_adapter import (
    ScalarGainAdapter, LinearActionHead, NonlinearAdapterHead, measure_gain,
    equivalence_error, surprise_composed, elbo_step_adapter,
)

__all__ = [
    "BayesianLinear", "BayesianDynamicsModel", "make_gaussian_bnn", "load_arch",
    "GAUSSIAN_CKPT", "surprise_gaussian", "epistemic_gaussian", "forget_gaussian",
    "mean_sigma", "forget_gaussian_perdim", "head_row_sigma",
    "anchor_head_rows_to_current", "retrain_gaussian_full_elbo", "INFLATE_MODE",
    "ScalarGainAdapter", "LinearActionHead", "NonlinearAdapterHead",
    "measure_gain", "equivalence_error", "surprise_composed", "elbo_step_adapter",
]
