"""Gaussian-head Bayesian dynamics model for the Reacher world model.

Trimmed relative to the parent repo: the Dirichlet variant is not imported, so
this package has no FrozenLake / ns_gym / grids dependency.  gaussian_workflow
is carried even though the pretraining + audit path does not call it, because it
is what the follow-on surprise -> forget -> inflate loop on non-stationary
Reacher will need, and it costs nothing here.
"""
from bnn.layers import BayesianLinear
from bnn.gaussian_model import (
    BayesianDynamicsModel, make_gaussian_bnn, load_arch, GAUSSIAN_CKPT,
)
from bnn.gaussian_workflow import (
    surprise_gaussian, epistemic_gaussian, forget_gaussian, mean_sigma,
    forget_gaussian_perdim, anchor_head_rows_to_current, INFLATE_MODE,
)
from bnn.gain_adapter import (
    NonlinearAdapterHead, LinearActionHead, measure_gain, elbo_step_adapter,
    surprise_composed,
)

__all__ = [
    "BayesianLinear", "BayesianDynamicsModel", "make_gaussian_bnn", "load_arch",
    "GAUSSIAN_CKPT", "surprise_gaussian", "epistemic_gaussian", "forget_gaussian",
    "mean_sigma", "forget_gaussian_perdim", "anchor_head_rows_to_current",
    "INFLATE_MODE", "NonlinearAdapterHead", "LinearActionHead", "measure_gain",
    "elbo_step_adapter", "surprise_composed",
]
