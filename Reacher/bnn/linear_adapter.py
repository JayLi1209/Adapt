r"""A genuine ONE-LAYER Bayesian adapter head, gradient-trained by ELBO.

WHY THIS EXISTS, AND WHAT IT GIVES UP
-------------------------------------
`bnn.gain_adapter.LinearActionHead` is CONJUGATE: mu = mu_BNN(s,0) + w*u with w
from a closed-form Gaussian recursion, so "forgetting" is exactly a discount on
sufficient statistics --

    Lambda <- lambda*Lambda + u^2/sigma_n^2,   b <- lambda*b + u*r/sigma_n^2

-- and one knob provably moves the posterior mean (w = b/Lambda) and its variance
(1/Lambda) together.  That guarantee is the whole appeal, and it is bought by
restricting the head to the action channel.

The moment the head is an expressive nonlinear net fit by backprop through an
ELBO, the guarantee is GONE.  There is no closed form, no sufficient statistic
to discount, and nothing forces mean and variance to move coherently.
`NonlinearAdapterHead.forget` inherits the conjugate *formula* anyway: it
inflates the variational PRECISION (weight_rho) toward the prior and never
touches weight_mu.  Under a conjugate recursion that is equivalent to
down-weighting old evidence, because the mean is a deterministic function of the
statistics.  Under backprop it is not: the mean is a free parameter carrying
everything the head has learned, so inflating only the precision widens the
belief while leaving the point estimate exactly where it was.  Measured on
Reacher wind, that is destructive -- skip/l2 sigma 0.005 -> 0.08 (16x) with the
head's spread across draws (3.49) exceeding the correction it is making (1.36).

So under gradient training "forgetting" has to be re-read as one of two honest
things, and this module implements both explicitly rather than pretending the
conjugate identity still holds:

  FORGET_MODE = "prior_drift"   a regularizer.  Pull the MEAN toward the prior
      mean as well as the precision, both by the same rho:
          mu  <- rho * mu                       (prior mean is 0 for the
                                                 nonlinear part; the warm-start
                                                 action block is the anchor)
          tau <- tau0 + rho (tau - tau0)
      This is the coherent generalisation: it is what the conjugate update
      REDUCES TO when the likelihood is switched off, i.e. genuine decay of
      accumulated evidence toward the prior, applied to both moments.

  FORGET_MODE = "discount"      a learned discount on the LOSS instead of on the
      parameters.  Leave the posterior alone and exponentially down-weight old
      replay samples by rho^age when forming the ELBO's expected NLL, so old
      evidence ages out of the FIT rather than being erased from the belief.
      This is the other faithful reading of "discount the statistics", and it
      is the one that survives non-conjugacy without touching the parameters.

STRUCTURE (one layer, and one only)
-----------------------------------
    mu(s,u) = mu_BNN(s, 0) + h_phi(s, u)
    h_phi(x) = W x + b,        x = [obs, u]          <- ONE BayesianLinear

No hidden layer, no second branch.  Contrast NonlinearAdapterHead, whose
skip + l2(silu(l1)) is three layers and ~1.4k parameters on Reacher -- fit
online against a few hundred visited states.  This head has obs_dim*(obs_dim +
act_dim) + obs_dim = 130 parameters on Reacher, ~11x fewer, and being affine in
(s,u) it cannot manufacture confident structure in rollout states it never
visited.  It is still strictly more expressive than the conjugate action head,
which spans only w*u: this one spans an arbitrary affine function of the STATE
too, which is what a position-dependent force field needs.

Warm start is the same property the conjugate head had: the action block of W is
set to the body's measured gain W0 and the state block is zeroed, so at step 0
h(s,u) = W0 @ u and the composed model reproduces the frozen body exactly.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LinearAdapterHead", "elbo_step_linear_adapter"]


class LinearAdapterHead(nn.Module):
    """One Bayesian affine layer over [obs, act], added to the frozen body at u=0."""

    def __init__(self, w0, obs_dim, act_dim=1, prior_std=0.1, init_sigma=1e-3):
        super().__init__()
        from bnn.layers import BayesianLinear
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.lin = BayesianLinear(self.obs_dim + self.act_dim, self.obs_dim,
                                  prior_std=prior_std)
        rho0 = math.log(math.expm1(init_sigma))
        with torch.no_grad():
            self.lin.weight_mu.zero_()
            w0_t = torch.as_tensor(np.asarray(w0), dtype=torch.float32)
            if self.act_dim == 1:
                self.lin.weight_mu[:, -1] = w0_t.reshape(-1)
            else:
                self.lin.weight_mu[:, -self.act_dim:] = w0_t.reshape(
                    self.obs_dim, self.act_dim)
            self.lin.bias_mu.zero_()
            self.lin.weight_rho.fill_(rho0)
            self.lin.bias_rho.fill_(rho0)
        # The warm start IS the anchor: prior_drift decays toward this, not
        # toward zero, so forgetting returns the head to the frozen body rather
        # than to a headless model with no action response at all.
        self.register_buffer("anchor_w", self.lin.weight_mu.detach().clone())
        self.register_buffer("anchor_b", self.lin.bias_mu.detach().clone())
        self.anchor(include_sigma=True)
        self.use_body_reward = True
        self.rollout_mean_only = False
        self._orig_sample = None
        self._dyn = None

    @property
    def layers(self):
        return [self.lin]

    def forward(self, x, sample=True):
        return self.lin(x, sample=sample)

    def kl(self):
        return self.lin.kl_divergence()

    def anchor(self, include_sigma=True):
        self.lin.anchor_prior_to_current(include_sigma=include_sigma)

    # ── composition (identical contract to NonlinearAdapterHead) ─────────────
    def predict(self, bnn, dyn, obs, act, sample=False, n_groups=1):
        mean, logvar = bnn._run_network(
            dyn._get_model_input(obs, torch.zeros_like(act)),
            sample=sample, num_weight_groups=n_groups)
        h = self.forward(torch.cat([obs, act], dim=-1), sample=sample)
        mean = mean.clone()
        mean[:, :self.obs_dim] = mean[:, :self.obs_dim] + h
        return mean, logvar

    def attach(self, dyn):
        if self._orig_sample is not None:
            return self
        orig, head = dyn.sample, self

        def composed(act, model_state, deterministic=False, rng=None):
            nxt, _rew0, term, ns = orig(torch.zeros_like(act), model_state,
                                        deterministic=deterministic, rng=rng)
            # Reward read at the TRUE action: the head corrects dynamics only, and
            # reading the body's reward channel at u=0 would discard Reacher's
            # -||u||^2 control cost.  See gain_adapter.attach for the full note.
            if head.use_body_reward:
                _n1, rew, _t1, _s1 = orig(act, model_state,
                                          deterministic=deterministic, rng=rng)
            else:
                rew = _rew0
            obs = model_state["obs"]
            with torch.no_grad():
                h = head.forward(torch.cat([obs, act], dim=-1),
                                 sample=(not deterministic
                                         and not head.rollout_mean_only))
            nxt = nxt.clone()
            nxt[:, :head.obs_dim] = nxt[:, :head.obs_dim] + h
            ns["obs"] = nxt
            return nxt, rew, term, ns

        dyn.sample = composed
        self._orig_sample, self._dyn = orig, dyn
        return self

    def detach(self):
        if self._orig_sample is not None:
            self._dyn.sample = self._orig_sample
            self._orig_sample = None
        return self

    # ── forgetting, re-read for a NON-CONJUGATE head ─────────────────────────
    @torch.no_grad()
    def forget(self, delta_bar, mode="prior_drift", rho_floor=1e-3,
               decay_step=None):
        """rho = 1/max(delta_bar, 1), per output dim.

        mode="prior_drift": decay BOTH moments toward the warm-start anchor,
            mu  <- anchor + rho (mu - anchor)
            tau <- tau0   + rho (tau - tau0)
        so the head genuinely un-learns accumulated evidence rather than merely
        becoming unsure of a point estimate it still fully trusts.

        mode="precision_only": the legacy conjugate formula (inflate tau, leave
            mu). Kept so the incoherence can be measured rather than asserted.
        """
        db = np.asarray(delta_bar, dtype=np.float64)[:self.obs_dim]
        # rho_floor BOUNDS the decay.  The unbounded rule reaches rho ~ 1.5e-3,
        # which wipes ~99.85% of the head's accumulated correction in ONE call --
        # measured against 5 Adam steps that can only move it a little, so
        # inflation always wins the tug-of-war and |W - anchor| stays pinned near
        # its init.  A floor of e.g. 0.9 caps the per-call erasure at 10%.
        rho = np.clip(1.0 / np.maximum(db, 1.0), rho_floor, 1.0)
        if decay_step is not None:
            # SOFT ADDITIVE decay: move a FIXED fraction toward the anchor per
            # call, independent of how large delta_bar is, so a huge surprise can
            # no longer cause an unbounded reset.
            rho = np.where(rho < 1.0, 1.0 - float(decay_step), 1.0)
        if (rho >= 1.0).all():
            return rho, False
        layer = self.lin
        tau0 = 1.0 / (layer.prior_std ** 2)
        for r in range(layer.weight_rho.shape[0]):
            rr = float(rho[r])
            if rr >= 1.0:
                continue
            if mode == "prior_drift":
                layer.weight_mu.data[r] = (
                    self.anchor_w[r] + rr * (layer.weight_mu.data[r] - self.anchor_w[r]))
                layer.bias_mu.data[r] = (
                    self.anchor_b[r] + rr * (layer.bias_mu.data[r] - self.anchor_b[r]))
            for rho_p in (layer.weight_rho, layer.bias_rho):
                sigma = F.softplus(rho_p.data[r])
                tau_new = tau0 + rr * (1.0 / (sigma ** 2) - tau0)
                rho_p.data[r] = torch.log(
                    torch.expm1((1.0 / tau_new).sqrt()).clamp_min(1e-6))
        return rho, True


def elbo_step_linear_adapter(head, opt, bnn, dyn, obs_b, act_b, tgt_b, active_dims,
                             n_steps=5, n_mc=3, beta=1.0, kl_denom=None,
                             grad_clip=10.0, sample_w=None, l2_anchor=0.0,
                             l2_scale=None):
    r"""ELBO update on head-corrected residuals, with optional per-sample weights.

        r    = delta_s_obs - mu_BNN(s, 0)
        NLL  = 1/D sum_d 0.5[ log sigma_n,d^2 + (r_d - h_d)^2 / sigma_n,d^2 ]
        loss = E_q[NLL] + beta * KL(q||p) / N

    `sample_w` (B,) implements FORGET_MODE="discount": exponentially aged weights
    rho^age applied to the expected NLL, so old evidence leaves the FIT without
    the parameters being touched.  None = uniform, the ordinary ELBO.
    """
    active = sorted(int(d) for d in active_dims)
    N = float(kl_denom) if kl_denom else float(obs_b.shape[0])
    with torch.no_grad():
        m0, lv0 = bnn._run_network(
            dyn._get_model_input(obs_b, torch.zeros_like(act_b)), sample=False)
        r = tgt_b[:, :head.obs_dim] - m0[:, :head.obs_dim]
        s2 = torch.exp(lv0[:, :head.obs_dim]).clamp_min(1e-8)
    x = torch.cat([obs_b, act_b], dim=-1)
    if sample_w is not None:
        w = sample_w / sample_w.sum().clamp_min(1e-12) * sample_w.shape[0]
    nll = kl = None
    for _ in range(n_steps):
        opt.zero_grad()
        acc = 0.0
        for _ in range(n_mc):
            h = head(x, sample=True)
            per_d = []
            for d in active:
                e = 0.5 * (torch.log(s2[:, d]) + (r[:, d] - h[:, d]) ** 2 / s2[:, d])
                per_d.append((e * w).mean() if sample_w is not None else e.mean())
            acc = acc + sum(per_d) / float(len(active))
        nll = acc / n_mc
        kl = head.kl()
        loss = nll + beta * kl / N
        if l2_anchor > 0.0:
            # FORGETTING AS A LOSS TERM, not a parameter reset.  Instead of
            # snapping mu back toward the anchor between gradient steps, add
            #     l2 * ||mu - anchor||^2
            # so the pull toward the anchor and the pull toward the data are
            # resolved by the SAME optimiser, in one objective, and can reach an
            # equilibrium rather than fighting.  l2_scale (per-dim, from the
            # drift filter) lets a dim that is genuinely surprising be held more
            # loosely than one that is not.
            dw = head.lin.weight_mu - head.anchor_w
            dbs = head.lin.bias_mu - head.anchor_b
            if l2_scale is not None:
                sc = torch.as_tensor(l2_scale, dtype=dw.dtype, device=dw.device
                                     ).view(-1, 1)
                pen = (sc * dw.pow(2)).sum() + (sc.view(-1) * dbs.pow(2)).sum()
            else:
                pen = dw.pow(2).sum() + dbs.pow(2).sum()
            loss = loss + l2_anchor * pen
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
        opt.step()
    return float(nll.item()), float(kl.item())
