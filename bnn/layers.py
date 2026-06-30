"""Bayesian (mean-field VI) linear layer -- the only BNN building block the
Dirichlet world model needs.  Extracted verbatim from bnn_nsant_cem.py.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class BayesianLinear(nn.Module):
    """Linear layer with mean-field VI posterior q(w) = N(mu, softplus(rho)^2)."""

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        self.weight_mu = Parameter(torch.empty(out_features, in_features))
        self.weight_rho = Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = Parameter(torch.zeros(out_features))
        self.bias_rho = Parameter(torch.full((out_features,), -3.0))

        nn.init.kaiming_uniform_(self.weight_mu, nonlinearity="relu")

        # Prior MEAN for the KL term.  Default 0 (the usual N(0, prior_std) prior).
        # `anchor_prior_to_current` snapshots the current weights here so on-line
        # fine-tuning is regularized toward THOSE weights (a trust region) instead
        # of toward 0 -- which keeps predictions at unvisited inputs (no data
        # gradient) pinned to the good pretrained model rather than decaying.
        self.register_buffer("prior_weight_mu", torch.zeros(out_features, in_features))
        self.register_buffer("prior_bias_mu", torch.zeros(out_features))

    @property
    def weight_sigma(self):
        return F.softplus(self.weight_rho)

    @property
    def bias_sigma(self):
        return F.softplus(self.bias_rho)

    def forward(
        self, x: torch.Tensor, sample: bool = True, num_weight_groups: int = 1
    ) -> torch.Tensor:
        # Deterministic, or the classic single-draw-shared-across-the-batch path.
        if not sample:
            return F.linear(x, self.weight_mu, self.bias_mu)
        if num_weight_groups <= 1:
            weight = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_mu)
            return F.linear(x, weight, bias)

        # M x k split: draw M independent weight sets; row r uses set (r % M),
        # so groups of k = B/M rows share a draw. `x.view(B//M, M, F)[t, g]` is
        # row t*M + g, whose group id is g -> the M-axis indexes the weight set.
        M = num_weight_groups
        B, in_f = x.shape
        out_f = self.out_features
        assert B % M == 0, (
            f"batch size {B} must be divisible by num_weight_groups {M}; "
            "for clean per-sequence grouping, also have M divide num_particles"
        )
        noise_w = torch.randn(M, out_f, in_f, device=x.device, dtype=x.dtype)
        noise_b = torch.randn(M, out_f, device=x.device, dtype=x.dtype)
        weight = self.weight_mu.unsqueeze(0) + self.weight_sigma.unsqueeze(0) * noise_w
        bias = self.bias_mu.unsqueeze(0) + self.bias_sigma.unsqueeze(0) * noise_b
        xr = x.view(B // M, M, in_f).permute(1, 0, 2)  # (M, B/M, in)
        y = torch.bmm(xr, weight.transpose(1, 2)) + bias.unsqueeze(1)  # (M, B/M, out)
        return y.permute(1, 0, 2).reshape(B, out_f)

    def kl_divergence(self) -> torch.Tensor:
        log_prior = np.log(self.prior_std)

        def _kl(mu, sigma, prior_mu):
            return 0.5 * (
                (sigma / self.prior_std) ** 2
                + ((mu - prior_mu) / self.prior_std) ** 2
                - 1.0
                + 2.0 * (log_prior - torch.log(sigma))
            ).sum()

        return (
            _kl(self.weight_mu, self.weight_sigma, self.prior_weight_mu)
            + _kl(self.bias_mu, self.bias_sigma, self.prior_bias_mu)
        )

    def anchor_prior_to_current(self):
        """Set the KL prior mean to the current weights (trust-region anchor)."""
        self.prior_weight_mu.data.copy_(self.weight_mu.data)
        self.prior_bias_mu.data.copy_(self.bias_mu.data)
