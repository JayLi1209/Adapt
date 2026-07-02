"""Bayesian world models for FrozenLake: Dirichlet-head and Gaussian-head."""
from bnn.layers import BayesianLinear
from bnn.dirichlet_model import (
    DirichletDynamicsModel, make_dirichlet_bnn, direction_of,
    CONC_PRIOR, DIRICHLET_CKPT,
)
from bnn.dirichlet_workflow import (
    epistemic_dirichlet, surprise_dirichlet, forget_dirichlet, mean_alpha0,
)
from bnn.gaussian_model import (
    BayesianDynamicsModel, make_gaussian_bnn, GAUSSIAN_CKPT,
)
from bnn.gaussian_workflow import (
    surprise_gaussian, epistemic_gaussian, forget_gaussian, mean_sigma,
    inflate_by_process_noise, inflate_toward_prior,
    N_SURPRISE_DRAWS, DELTA_CLIP, INFLATE_MODE,
)

__all__ = [
    "BayesianLinear",
    # Dirichlet variant
    "DirichletDynamicsModel", "make_dirichlet_bnn",
    "direction_of", "CONC_PRIOR", "DIRICHLET_CKPT",
    "epistemic_dirichlet", "surprise_dirichlet", "forget_dirichlet", "mean_alpha0",
    # Gaussian variant
    "BayesianDynamicsModel", "make_gaussian_bnn", "GAUSSIAN_CKPT",
    "surprise_gaussian", "epistemic_gaussian", "forget_gaussian", "mean_sigma",
    "inflate_by_process_noise", "inflate_toward_prior",
    "N_SURPRISE_DRAWS", "DELTA_CLIP", "INFLATE_MODE",
]
