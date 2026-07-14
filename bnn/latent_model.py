"""Latent-factor BNN for ADA-MCTS. Uses K latent basis factors whose outputs
are linearly combined by learned latent weights w_b ~ P_W.

Follows the HiPMDP linear_latent_weights design (Killian et al. 2017) and
ADA-MCTS Algorithm 1 (Luo et al. 2024):
  - BNN predicts K separate basis dynamics functions (the "factors")
  - w_b (K-dim) combines them: output = sum_k w_b[k] * basis_k(s,a)
  - During stationary pretraining w_b is learned once
  - During non-stationary adaptation only output layer + w_b are fine-tuned
"""

import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bnn.gaussian_model import BayesianDynamicsModel
from config import device

NUM_LATENT_FACTORS = 5    # K basis dynamics predictors
LATENT_CKPT = "latent_bnn.pth"


class LatentBNN(BayesianDynamicsModel):
    """Gaussian BNN with K latent basis factors combined by learned weights w_b.

    Architecture:
      Input:  (obs, act)                          → dim = obs_dim + act_dim
      Hidden: standard BayesianLinear layers
      Output: K × (obs_dim + 1) raw predictions    → K factors
      Final:  output = Σ w_b[k] * factor_k(s,a)   → dim = obs_dim + 1
    """

    def __init__(self, obs_dim, act_dim, num_factors=NUM_LATENT_FACTORS, **kwargs):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_factors = num_factors
        per_factor_out = obs_dim + 1                       # delta-obs + reward
        internal_out = num_factors * per_factor_out        # K factors concatenated
        super().__init__(in_size=obs_dim + act_dim, out_size=internal_out,
                         device=device, **kwargs)
        self.in_size = obs_dim + act_dim   # normalizer uses this (no latent in input)
        # w_b: latent combination weights drawn from prior P_W ~ N(0, I)
        self.latent = nn.Parameter(torch.randn(num_factors, device=device))

    def _run_network(self, x, sample=True, num_weight_groups=1):
        """Run BNN to get K factor predictions, then combine with latent w_b.
        x: (B, obs_dim + act_dim) — raw (obs, act), NO latent concatenation.

        BayesianLinear handles num_weight_groups internally without changing
        output shape, so the parent always returns (B, K*P).
        """
        B = x.shape[0]
        K = self.num_factors
        P = self.obs_dim + 1    # per-factor output dim (delta-obs + reward)

        # Parent forward pass → raw factor predictions: (B, K*P)
        mean_raw, logvar_raw = super()._run_network(
            x, sample=sample, num_weight_groups=num_weight_groups)

        # Reshape to separate factors: (B, K, P)
        mean = mean_raw.view(B, K, P)
        logvar = logvar_raw.view(B, K, P)

        w = self.latent  # (K,)

        # ── Combine K factors via weighted sum ──────────────────────────
        # μ_combined = Σ w_k * μ_k
        mean_c = torch.einsum('bkp,k->bp', mean, w)           # (B, P)

        # σ²_combined = Σ w_k² * σ²_k  (independent factors)
        var = torch.exp(logvar)                                # (B, K, P)
        var_c = torch.einsum('bkp,k->bp', var, w ** 2)         # (B, P)
        logvar_c = torch.log(var_c + 1e-8)

        return mean_c, logvar_c

    def freeze_trunk(self):
        """Freeze all hidden layers; only output layer + latent remain trainable."""
        for layer in self.bayes_layers[:-1]:
            for p in layer.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        for layer in self.bayes_layers:
            for p in layer.parameters():
                p.requires_grad = True

    def head_parameters(self):
        """Parameters to train during adaptation: output layer + latent w_b."""
        params = list(self.bayes_layers[-1].parameters())
        params.append(self.latent)
        params.extend([self.max_logvar, self.min_logvar])
        return params

    @torch.no_grad()
    def epistemic_variance(self, x, n_draws=10):
        """Epistemic uncertainty: variance of per-draw means for input x.
        x: (B, obs_dim + act_dim) — raw (obs, act), factors+combination in _run_network.
        """
        means = []
        saved = self.num_weight_groups
        self.num_weight_groups = 1
        try:
            for _ in range(n_draws):
                m, _ = self._run_network(x, sample=True)
                means.append(m)
        finally:
            self.num_weight_groups = saved
        means = torch.stack(means)
        return float(means.var(0, unbiased=False).mean().item())

    def save_latent(self, save_dir, filename=LATENT_CKPT):
        torch.save({"model": self.state_dict(), "latent": self.latent.data.clone()},
                   pathlib.Path(save_dir) / filename)

    def load_latent(self, load_dir, filename=LATENT_CKPT):
        ckpt = torch.load(pathlib.Path(load_dir) / filename, map_location=self.device)
        self.load_state_dict(ckpt["model"], strict=False)
        if "latent" in ckpt:
            self.latent.data.copy_(ckpt["latent"])


def make_latent_bnn(obs_dim, act_dim, num_factors=NUM_LATENT_FACTORS):
    """Build a LatentBNN wrapped in OneDTransitionRewardModel for the planner."""
    import mbrl.models as models
    bnn = LatentBNN(obs_dim, act_dim, num_factors=num_factors)
    dyn = models.OneDTransitionRewardModel(
        bnn, target_is_delta=True, normalize=True, learned_rewards=True
    )
    return bnn, dyn
