"""Latent-factor BNN for ADA-MCTS. Concatenates a 5-dim learnable latent vector
to the (obs, action) input so the model can represent hidden dynamics parameters.

During stationary pretraining the latent is learned once. During non-stationary
adaptation only the output layer + latent are fine-tuned; the trunk is frozen.
"""

import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bnn.gaussian_model import BayesianDynamicsModel
from config import device

LATENT_DIM = 5
LATENT_CKPT = "latent_bnn.pth"


class LatentBNN(BayesianDynamicsModel):
    """Gaussian BNN with an extra 5-dim latent input for hidden dynamics."""

    def __init__(self, obs_dim, act_dim, latent_dim=LATENT_DIM, **kwargs):
        internal_in = obs_dim + act_dim + latent_dim
        super().__init__(in_size=internal_in, out_size=obs_dim + 1, device=device, **kwargs)
        self.in_size = obs_dim + act_dim   # external interface: (obs, act) only
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.latent = nn.Parameter(torch.randn(latent_dim, device=device) * 0.1)

    def _run_network(self, x, sample=True, num_weight_groups=1):
        """Concatenate latent to input, then run the Bayesian network."""
        B = x.shape[0]
        lat = self.latent.unsqueeze(0).expand(B, -1)  # (B, latent_dim)
        x_aug = torch.cat([x, lat], dim=-1)            # (B, in_size + latent_dim)
        return super()._run_network(x_aug, sample=sample,
                                    num_weight_groups=num_weight_groups)

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
        """Parameters to train during adaptation: output layer + latent."""
        params = list(self.bayes_layers[-1].parameters())
        params.append(self.latent)
        # Also include logvar bounds
        params.extend([self.max_logvar, self.min_logvar])
        return params

    @torch.no_grad()
    def epistemic_variance(self, x, n_draws=10):
        """Epistemic uncertainty: variance of per-draw means for input x.
        x: (B, in_size) — (obs, act) WITHOUT latent (added by _run_network).
        """
        means = []
        saved = self.num_weight_groups
        self.num_weight_groups = 1
        try:
            for _ in range(n_draws):
                m, _ = self._run_network(x, sample=True)  # latent added inside
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


def make_latent_bnn(obs_dim, act_dim, latent_dim=LATENT_DIM):
    """Build a LatentBNN wrapped in OneDTransitionRewardModel for the planner."""
    import mbrl.models as models
    bnn = LatentBNN(obs_dim, act_dim, latent_dim=latent_dim)
    # target_is_delta=True: model predicts delta-obs + reward
    # normalize=True: normalizer expected in save_dir
    dyn = models.OneDTransitionRewardModel(
        bnn, target_is_delta=True, normalize=True, learned_rewards=True
    )
    return bnn, dyn
