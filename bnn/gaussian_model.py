"""Gaussian-head BNN world model for ns-gym FrozenLake.

The NON-Dirichlet variant used in notebooks/diagnose_v2.py: a shared Bayesian
trunk with a single Gaussian readout that emits (mean, logvar) over the raw output
space (delta-obs + reward).  The predictive next-state categorical the planner
reads is recovered by clamping the mean-delta prediction to the non-negative
simplex (see planning.base._model_matrices / run_gaussian.pred_next_row) rather
than by a dedicated categorical head.

Contrast with bnn.dirichlet_model.DirichletDynamicsModel:

  * output layer is `out_size * 2` (a mean and a log-variance per output dim),
    not K=3 Dirichlet concentrations -> the head is a plain heteroscedastic
    Gaussian and surprise is a Mahalanobis chi-square (see gaussian_workflow), not
    a categorical -log p / H.
  * target_is_delta=True, normalize=True (the wrapper predicts obs-deltas from
    normalized inputs), whereas the Dirichlet model predicts p(s') directly from
    raw one-hot inputs.
  * re-inflation acts on the variational weight covariance (softplus(rho)) via
    additive process noise or a retention pull toward the prior, NOT on a
    `retain` scalar multiplying a Dirichlet head.

Extracted from notebooks/bnn_nsant_cem.py (BayesianDynamicsModel) and
notebooks/bnn_fl_cem_surprise_drift.py (make_bnn); the BayesianLinear layer is
shared with the Dirichlet model via bnn.layers.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import mbrl.models as models

from bnn.layers import BayesianLinear
from config import device

# Shared Gaussian checkpoint (mean/logvar trunk+head, saved by BayesianDynamicsModel).
GAUSSIAN_CKPT = "bnn_dynamics.pth"


class BayesianDynamicsModel(models.Model):
    """BNN dynamics model with a Gaussian (mean, logvar) head, mean-field VI.

    bayes_layers = [trunk hidden..., output], where the output layer emits
    out_size*2 units split into (mean, logvar).  The `forget`/inflate utilities in
    gaussian_workflow iterate bayes_layers and touch weight_rho/bias_rho, exactly
    as they do for the Dirichlet trunk.
    """

    def __init__(self, in_size, out_size, device, hid_size=256, num_layers=3,
                 prior_std=1.0, beta=0.1, num_mc_samples=3, num_weight_groups=1,
                 n_adapters=0):
        print("Entered Gaussian BNN Model")
        super().__init__(device)
        self.in_size = in_size
        self.out_size = out_size
        self.beta = beta
        self.num_mc_samples = num_mc_samples
        # Planning-rollout M x k split (see BayesianLinear): per timestep draw
        # `num_weight_groups` (M) independent weight sets, each shared by k rows.
        self.num_weight_groups = num_weight_groups
        # Pre-noise per-row network means from the most recent stochastic sample_1d
        # call -- lets a caller read epistemic uncertainty off the planning pass.
        self.last_pred_mean = None
        # Size of the FULL training set, used to scale the KL term in the ELBO.
        self.num_train_points = None

        layer_sizes = [in_size] + [hid_size] * (num_layers - 1) + [out_size * 2]
        self.bayes_layers = nn.ModuleList(
            [
                BayesianLinear(layer_sizes[i], layer_sizes[i + 1], prior_std=prior_std)
                for i in range(len(layer_sizes) - 1)
            ]
        )

        # Online-adaptation stack: n_adapters PURE linear (no activation) layers
        # inserted between the frozen trunk and the output layer, identity-
        # initialized so at reset the predictions are exactly the pretrained
        # model's.  Kept OUT of bayes_layers: checkpoints load unchanged and
        # forget/inflate never touch them -- they are gradient-retrained online
        # instead (see gaussian_workflow.retrain_gaussian).
        self.adapt_layers = nn.ModuleList([
            BayesianLinear(hid_size, hid_size, prior_std=prior_std)
            for _ in range(n_adapters)
        ])
        self.reset_adapters()

        # Smooth log-variance clamp bounds (PETS-style), learned.
        self.max_logvar = Parameter(0.5 * torch.ones(out_size))
        self.min_logvar = Parameter(-10.0 * torch.ones(out_size))

        self.to(device)

    def reset_adapters(self):
        """Re-init every adapter to near-deterministic identity (weight_mu = I,
        bias 0, sigma ~ 1e-4): the stack is a no-op until it is retrained."""
        with torch.no_grad():
            for layer in self.adapt_layers:
                layer.weight_mu.copy_(torch.eye(
                    layer.out_features, layer.in_features,
                    device=layer.weight_mu.device))
                layer.bias_mu.zero_()
                layer.weight_rho.fill_(-9.0)
                layer.bias_rho.fill_(-9.0)

    def _run_network(self, x, sample=True, num_weight_groups=1):
        hidden_layers = self.bayes_layers[:-1]
        output_layer = self.bayes_layers[-1]

        h = x
        for layer in hidden_layers:
            h = F.silu(layer(h, sample=sample, num_weight_groups=num_weight_groups))
        for layer in self.adapt_layers:
            h = layer(h, sample=sample, num_weight_groups=num_weight_groups)
        raw = output_layer(h, sample=sample, num_weight_groups=num_weight_groups)

        mean, raw_logvar = raw.chunk(chunks=2, dim=-1)
        # Smoothly clamp log-variance (as in PETS) for stable training.
        bounded = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        final_logvar = self.min_logvar + F.softplus(bounded - self.min_logvar)
        return mean, final_logvar

    def forward(self, x, sample=True, num_weight_groups=1):
        return self._run_network(x, sample=sample, num_weight_groups=num_weight_groups)

    def _total_kl(self):
        return sum(layer.kl_divergence() for layer in self.bayes_layers)

    def anchor_prior_to_current(self, include_sigma: bool = False, layers=None):
        """Re-anchor the KL prior.  `layers=None` anchors the whole trunk+head;
        pass a subset to re-anchor only the layers an online ELBO will train."""
        for layer in (self.bayes_layers if layers is None else layers):
            layer.anchor_prior_to_current(include_sigma=include_sigma)

    def loss(self, model_in, target=None):
        B = model_in.numel() // self.in_size
        nll_acc = torch.zeros(1, device=self.device)
        for _ in range(self.num_mc_samples):
            mean, logvar = self._run_network(model_in, sample=True)
            nll = 0.5 * (logvar + (target - mean) ** 2 / torch.exp(logvar))
            nll_acc = nll_acc + nll.mean()
        avg_nll = nll_acc / self.num_mc_samples
        kl = self._total_kl()
        kl_denom = self.num_train_points if self.num_train_points else B
        loss = avg_nll + self.beta * kl / kl_denom
        return loss, {"nll": avg_nll.item(), "kl": kl.item()}

    def eval_score(self, model_in, target=None):
        with torch.no_grad():
            mean, _ = self._run_network(model_in, sample=False)
            score = F.mse_loss(mean, target, reduction="none")
        return score, {}

    def set_elite(self, elite_indices):
        pass

    def sample_1d(self, model_in, model_state, rng=None, deterministic=False):
        with torch.no_grad():
            mean, logvar = self._run_network(
                model_in, sample=not deterministic,
                num_weight_groups=self.num_weight_groups,
            )
            self.last_pred_mean = mean.clone()
            if deterministic:
                return mean, {}
            std = torch.exp(0.5 * logvar)
            preds = mean + std * torch.randn_like(mean)
        return preds, {}

    def reset_1d(self, obs, rng=None):
        return {}

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, save_dir, filename=GAUSSIAN_CKPT):
        torch.save(self.state_dict(), pathlib.Path(save_dir) / filename)

    def load(self, load_dir, filename=GAUSSIAN_CKPT):
        # strict=False so checkpoints saved before the prior-mean buffers existed
        # still load; those buffers then keep their zero init (the N(0) prior).
        self.load_state_dict(
            torch.load(pathlib.Path(load_dir) / filename, map_location=self.device),
            strict=False,
        )


def load_arch(model_dir):
    """Read arch.json from a checkpoint dir; fall back to the shipped 256x2 trunk."""
    import json as _json, pathlib as _pl
    f = _pl.Path(model_dir) / "arch.json"
    if f.exists():
        d = _json.loads(f.read_text())
        return int(d.get("hid_size", 256)), int(d.get("num_layers", 3))
    return 256, 3


def make_gaussian_bnn(obs_dim, act_dim, n_train_layers=1,
                      hid_size=256, num_layers=3):
    """Gaussian-head model wrapped for the planner.

    target_is_delta=True -> sample() returns obs + predicted delta (a 16-cell
                            vector clamped/normalized into p(s') downstream).
    normalize=True        -> model_in is normalized by the loaded env_stats, matching
                            how the checkpoint was pretrained.
    hid_size / num_layers -> trunk shape.  layer_sizes = [in] + [hid]*(num_layers-1)
                            + [out*2], so num_layers=3 is the shipped 2-hidden-layer
                            256-wide net and num_layers=5, hid_size=512 is a
                            4-hidden-layer 512-wide net.  A checkpoint only loads
                            into the SAME shape, so pretraining and evaluation must
                            agree (see arch.json written by pretrain_pendulum.py).
    n_train_layers        -> size of the online-adaptation stack COUNTING the
                            output layer: n_train_layers-1 identity-init linear
                            layers are inserted below it and gradient-retrained
                            online.  The default 1 inserts nothing -- the shipped
                            architecture, unchanged (checkpoints load at any
                            value; the adapters are always fresh).
    """
    bnn = BayesianDynamicsModel(
        in_size=obs_dim + act_dim,
        out_size=obs_dim + 1,
        device=device,
        hid_size=hid_size,
        num_layers=num_layers,
        prior_std=1.0,
        beta=0.1,
        num_mc_samples=3,
        num_weight_groups=1,
        n_adapters=n_train_layers - 1,
    )
    dynamics_model = models.OneDTransitionRewardModel(
        bnn, target_is_delta=True, normalize=True, learned_rewards=True
    )
    return bnn, dynamics_model
