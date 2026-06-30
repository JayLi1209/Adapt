"""Dirichlet-head Bayesian world model for FrozenLake."""
from bnn.layers import BayesianLinear
from bnn.dirichlet_model import (
    DirichletDynamicsModel, make_dirichlet_bnn, direction_of,
    CONC_PRIOR, DIRICHLET_CKPT,
)
from bnn.dirichlet_workflow import (
    epistemic_dirichlet, surprise_dirichlet, forget_dirichlet, mean_alpha0,
)

__all__ = [
    "BayesianLinear", "DirichletDynamicsModel", "make_dirichlet_bnn",
    "direction_of", "CONC_PRIOR", "DIRICHLET_CKPT",
    "epistemic_dirichlet", "surprise_dirichlet", "forget_dirichlet", "mean_alpha0",
]
