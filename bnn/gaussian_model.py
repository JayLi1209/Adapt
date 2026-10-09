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
        # beta-NLL (Seitzer et al., ICLR 2022).  0.0 = plain Gaussian NLL (the
        # shipped behaviour).  See `loss` for what it fixes and why.
        self.beta_nll = 0.0
        # Planning-rollout M x k split (see BayesianLinear): per timestep draw
        # `num_weight_groups` (M) independent weight sets, each shared by k rows.
        self.num_weight_groups = num_weight_groups
        # Pre-noise per-row network means from the most recent stochastic sample_1d
        # call -- lets a caller read epistemic uncertainty off the planning pass.
        self.last_pred_mean = None
        # Whether a stochastic sample_1d ADDS the predicted observation noise on
        # top of the weight-sampled mean.  True (default) is the shipped
        # behaviour: preds = mu_w + sigma_aleatoric * eps.
        #
        # Set False for a DETERMINISTIC env, where the aleatoric channel is not
        # real process noise but absorbed model error, and injecting it every
        # step of a self-fed rollout simulates randomness the env does not have.
        # On an open-loop-UNSTABLE plant that is fatal: InvertedPendulum's error
        # doubles every ~4 steps, so per-step noise of a fifth of the true state
        # change compounds until every imagined trajectory falls over, regardless
        # of the actions -- the planner then cannot rank candidates at all.
        # With this False the K posterior draws still differ (each uses its own
        # weight sample), so CVaR still averages a genuine tail -- one made of
        # EPISTEMIC disagreement rather than fictitious observation noise.
        self.aleatoric_in_rollout = True
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

        # OUTPUT standardizer (see set_output_scaler).  Identity by default, so
        # models that do not set it behave exactly as before.
        self.register_buffer("out_mu", torch.zeros(out_size))
        self.register_buffer("out_std", torch.ones(out_size))

        self.to(device)

    def set_output_scaler(self, mu, std, eps=1e-8):
        """Make the network predict STANDARDIZED targets internally.

        mbrl normalizes model INPUTS but not targets, which is fine when every
        output dim has a similar scale (Pendulum: delta-obs ~0.1, reward ~1) and
        fatal when it does not.  On LunarLander the nine target dims span
        std 0.0037 (delta x) to 1.0 (reward): a Kaiming-init net starts three
        orders of magnitude above delta x, the shared trunk spends its capacity
        on the loud dims, and the log-variance bounds (a single clamp applied to
        all dims) cannot be right for both ends at once.

        With a scaler set, `_run_network` trains in standardized space and then
        maps back:  mean = mean_std * std + mu,  logvar = logvar_std + 2 log std.
        So EVERY caller (mbrl's wrapper, surprise_gaussian, forget, retrain)
        still sees raw target units -- only the parameters live in the well-
        conditioned space.  The buffers ride along in state_dict, so a checkpoint
        reloads with its scaler.
        """
        with torch.no_grad():
            self.out_mu.copy_(torch.as_tensor(mu, dtype=torch.float32,
                                              device=self.out_mu.device))
            self.out_std.copy_(torch.as_tensor(std, dtype=torch.float32,
                                               device=self.out_std.device
                                               ).clamp_min(eps))

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
        # De-standardize into raw target units (identity unless a scaler was set).
        return (mean * self.out_std + self.out_mu,
                final_logvar + 2.0 * torch.log(self.out_std))

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
        """Negative ELBO: (beta-)Gaussian NLL + beta * KL / N.

        With `beta_nll` > 0 this is the beta-NLL of Seitzer et al. (ICLR 2022),
        which fixes a specific pathology of plain heteroscedastic NLL:

            dNLL/dmu = -(y - mu) / sigma^2

        so every sample's pull on the MEAN is weighted by 1/sigma^2.  Wherever the
        net predicts a large sigma it also stops learning mu there -- the fit is
        "honestly calibrated" and simultaneously bad.  On this env that region is
        ground contact: measured on held-out data, contact rows supply only 6.7%
        of ang_vel's mu-gradient while carrying 75.3% of its error mass (11x
        starvation), and ang_vel is duly the worst-predicted dim.

        beta-NLL multiplies the per-element NLL by a DETACHED sigma^(2*beta),
        cancelling that weighting: beta=0 is plain NLL, beta=1 makes the
        mu-gradient exactly MSE's, beta=0.5 is the paper's robust default.  sigma
        still gets its own gradient, so calibration is retained.

        The weights are normalised to mean 1 PER DIM, which does two things:
        it keeps the loss on the same scale as plain NLL (so `beta`, set from
        --kl-budget against the un-weighted NLL, stays calibrated), and it makes
        the result invariant to whether sigma is measured in raw or standardized
        units -- the per-dim target scale from set_output_scaler cancels out, so
        this only redistributes weight ACROSS SAMPLES within a dim and never
        re-introduces the cross-dim imbalance the output scaler exists to remove.
        """
        B = model_in.numel() // self.in_size
        nll_acc = torch.zeros(1, device=self.device)
        for _ in range(self.num_mc_samples):
            mean, logvar = self._run_network(model_in, sample=True)
            nll = 0.5 * (logvar + (target - mean) ** 2 / torch.exp(logvar))
            if self.beta_nll > 0.0:
                w = torch.exp(logvar).detach() ** self.beta_nll
                w = w / w.mean(dim=0, keepdim=True).clamp_min(1e-12)
                nll = nll * w
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
            if deterministic or not self.aleatoric_in_rollout:
                # `mean` here is already the WEIGHT-SAMPLED mean whenever
                # deterministic is False, so returning it keeps the epistemic
                # spread across posterior draws and drops only the observation
                # noise.  Under deterministic=True it is the mean-weight
                # prediction, exactly as before.
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
