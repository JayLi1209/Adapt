r"""One-parameter adaptation: a scalar gain on the ACTION pathway of a frozen model.

MOTIVATION.  On Pendulum the mass enters the dynamics in exactly one place:

    dw/dt = 3g/(2l) * sin(theta)  +  3/(m l^2) * u
            \_______ passive _______/  \___ control ___/

The passive term carries NO mass.  So a network pretrained at mass m0 already
predicts the passive dynamics of mass m1 correctly, and the ONLY thing it gets
wrong is the control term -- by exactly the factor m0/m1.  Feeding that frozen
network `g*u` instead of `u`, with g = m0/m1, makes its control term
3/(m0 l^2) * (g u) = 3/(m1 l^2) * u -- the mass-m1 truth.  One scalar recovers
the whole change, and it is exact rather than approximate.

This is the extreme opposite of the ~200k-parameter ELBO retrain: nothing in the
network moves, including the Gaussian head, so the predictive variance and every
surprise statistic keep their calibrated pretrained meaning and there is no
inflate/retrain tug-of-war to ratchet.

WHERE THE GAIN IS APPLIED.  On the RAW action, before the input normalizer.  The
normalizer is affine, normalize(x) = (x - mu)/sigma, so scaling after it would
give (u - mu)/sigma * g, which is not the input the network would see for a
smaller torque.  `attach` wraps OneDTransitionRewardModel._get_model_input, the
single choke point every read of the model goes through -- planner rollouts,
surprise, probes -- so no planner code changes.

The gain multiplies the action ONLY on the way into the model.  The environment
still receives the true action, and the analytic reward is still charged the true
0.001*u^2, so the gain is a pure model reparameterisation, not a policy change.

FITTING (closed form, one scalar).  For a frozen net f and transition (o, u, o'):

    b     = f(o, 0)          the passive prediction -- independent of g
    d     = f(o, u) - b      the model's action-attributable delta at g = 1
    y     = (o' - o) - b     the observed residual after removing the passive part

If f is locally linear in u then f(o, g*u) - b = g*d, and minimising
sum_i ||y_i - g d_i||^2 gives ordinary least squares in one unknown:

    g* = sum_i <y_i, d_i> / sum_i <d_i, d_i>

Both sums are scalars, so this is an O(1)-per-step recursive estimator with no
gradients and no optimiser.  `decay` < 1 applies exponential forgetting to both
accumulators, which is how the estimator tracks a LATER change: old evidence ages
out on a fixed timescale instead of being drowned by new evidence.

b and d never change (the net is frozen), so the accumulators are append-only and
the estimate never has to be recomputed from the buffer.

`refine` drops the local-linearity assumption and minimises the true objective
sum_i ||f(o_i, g u_i) - (o'_i - o_i)||^2 by direct 1-D search.  Comparing it with
the closed form measures how nonlinear the network actually is in the action.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ScalarGainAdapter", "LinearActionHead", "NonlinearAdapterHead",
           "measure_gain", "equivalence_error", "surprise_composed",
           "elbo_step_adapter"]


class ScalarGainAdapter:
    """A single multiplier on the action input of a frozen dynamics model.

    Args:
        obs_dim:   number of state dims scored by the fit (the reward channel of
                   the head is excluded -- it is not a dynamics residual).
        g0:        initial gain.  1.0 = the pretrained model, unchanged.
        decay:     exponential forgetting on the LS accumulators.  1.0 keeps all
                   evidence; <1 gives an effective window of 1/(1-decay) steps.
        g_bounds:  clamp on the estimate.  Guards the first few steps, where
                   sum <d,d> is tiny and the ratio is numerically wild.
        min_evidence: until sum <d,d> exceeds this, `g` stays at g0.  Without it
                   the first transition -- which may have had near-zero torque,
                   hence near-zero d -- sets an arbitrary gain.
    """

    def __init__(self, obs_dim: int, g0: float = 1.0, decay: float = 1.0,
                 g_bounds: tuple[float, float] = (0.02, 5.0),
                 min_evidence: float = 1e-6):
        self.obs_dim = int(obs_dim)
        self.g0 = float(g0)
        self.g = float(g0)
        self.decay = float(decay)
        self.g_bounds = g_bounds
        self.min_evidence = float(min_evidence)
        self.s_yd = 0.0          # sum <y_i, d_i>
        self.s_dd = 0.0          # sum <d_i, d_i>
        self.n = 0
        self._orig_get_model_input = None
        self._dyn = None

    # ── plumbing ─────────────────────────────────────────────────────────────
    def attach(self, dyn):
        """Route every model read through `g * action`.  Idempotent."""
        if self._orig_get_model_input is not None:
            return self
        orig = dyn._get_model_input

        def gated(obs, action):
            # Scale the RAW action, before the affine input normalizer.
            return orig(obs, self.g * action)

        dyn._get_model_input = gated
        self._orig_get_model_input = orig
        self._dyn = dyn
        return self

    def detach(self):
        if self._orig_get_model_input is not None:
            self._dyn._get_model_input = self._orig_get_model_input
            self._orig_get_model_input = None
        return self

    def _raw_input(self, obs, action):
        """Model input at gain 1, bypassing the wrapper."""
        f = self._orig_get_model_input or self._dyn._get_model_input
        return f(obs, action)

    # ── the least-squares fit ────────────────────────────────────────────────
    @torch.no_grad()
    def _bd(self, bnn, obs, act):
        """(b, d) for a batch: passive prediction and action-attributable delta."""
        zero = torch.zeros_like(act)
        b, _ = bnn._run_network(self._raw_input(obs, zero), sample=False)
        f, _ = bnn._run_network(self._raw_input(obs, act), sample=False)
        b = b[:, :self.obs_dim]
        return b, f[:, :self.obs_dim] - b

    @torch.no_grad()
    def update(self, bnn, obs, act, next_obs):
        """Fold one transition (or a batch of them) into the estimate.

        obs/act/next_obs are (B, ·) tensors on the model's device.
        Returns the updated gain.
        """
        b, d = self._bd(bnn, obs, act)
        y = (next_obs - obs)[:, :self.obs_dim] - b
        if self.decay < 1.0:
            self.s_yd *= self.decay
            self.s_dd *= self.decay
        self.s_yd += float((y * d).sum())
        self.s_dd += float((d * d).sum())
        self.n += int(obs.shape[0])
        if self.s_dd > self.min_evidence:
            self.g = float(np.clip(self.s_yd / self.s_dd, *self.g_bounds))
        return self.g

    @torch.no_grad()
    def refine(self, bnn, obs, act, next_obs, lo=None, hi=None, iters=60):
        """Exact 1-D minimisation of the TRUE objective, dropping local linearity.

        Golden-section on sum_i ||f(o_i, g u_i) - (o'_i - o_i)||^2.  Cheap: one
        batched forward per evaluation, ~60 evaluations for a single scalar.
        Does not touch the recursive accumulators -- it returns a value, so the
        caller decides whether to adopt it.
        """
        lo = self.g_bounds[0] if lo is None else lo
        hi = self.g_bounds[1] if hi is None else hi
        tgt = (next_obs - obs)[:, :self.obs_dim]

        def sse(g):
            m, _ = bnn._run_network(self._raw_input(obs, g * act), sample=False)
            return float((m[:, :self.obs_dim] - tgt).pow(2).sum())

        phi = (5 ** 0.5 - 1) / 2
        c, d_ = hi - phi * (hi - lo), lo + phi * (hi - lo)
        fc, fd = sse(c), sse(d_)
        for _ in range(iters):
            if fc < fd:
                hi, d_, fd = d_, c, fc
                c = hi - phi * (hi - lo); fc = sse(c)
            else:
                lo, c, fc = c, d_, fd
                d_ = lo + phi * (hi - lo); fd = sse(d_)
            if hi - lo < 1e-6:
                break
        return 0.5 * (lo + hi)

    def reset(self):
        self.s_yd = self.s_dd = 0.0
        self.n = 0
        self.g = self.g0
        return self

    def __repr__(self):
        return (f"ScalarGainAdapter(g={self.g:.5f}, n={self.n}, "
                f"evidence={self.s_dd:.3e}, decay={self.decay})")


# ═══════════════════════════════════════════════════════════════════════════
class LinearActionHead:
    r"""Conjugate linear head owning the ENTIRE action channel of a frozen BNN.

        mu(s,u) = mu_BNN(s, 0) + w * u        (phi(u) = u, one w per output dim)

    WHY ANCHOR AT u=0 rather than scale the action into the network
    (ScalarGainAdapter above).  Two reasons, one exact and one empirical:

      exact       The true control term is 3u/(m l^2) -- linear in u with no
                  state coupling.  A head that owns the whole action channel
                  therefore represents it EXACTLY, not approximately; the
                  gain-scaling variant is only exact if the network happens to
                  be linear in u, which it is only to ~2%.
      in-support  mu_BNN(s,0) is evaluated at an action the network has seen
                  under any pretraining torque range.  When the checkpoint was
                  pretrained at |u|<=2 and deployed at |u|<=20, evaluating
                  mu_BNN(s,15) extrapolates 7x outside the training support;
                  u=0 never does.  (Check `support_gap` before assuming this
                  applies -- data/pendulum_uncapped was pretrained at |u|<=20,
                  so for THAT checkpoint the argument is the exactness one only.)

    WARM START.  w0 is measured off the frozen network itself, as the central
    difference d mu / du.  Pre-change the composed model then reproduces the BNN,
    so the innovation -- and therefore delta_n, delta_bar and the forget gate --
    stay exactly where they were.  The head does nothing until something changes.
    That is the stability half of the calibration story, and `equivalence_error`
    measures how well it actually holds.

    THE RECURSION is the Gaussian-sufficient-statistic image of discounting
    Dirichlet counts.  Per output dim d, with observation noise sigma_n^2 taken
    from the frozen model's own aleatoric channel:

        Lambda_d <- lambda_d * Lambda_d + u^2 / sigma_n^2
        b_d      <- lambda_d * b_d      + u  * r_d / sigma_n^2
        w_d       = b_d / Lambda_d

    r_d = (delta s)_d - mu_BNN,d(s,0) is the residual the head must explain.
    The discount is the SAME function retention-mode forgetting applies to the
    network's posterior width, driven by the same smoothed surprise:

        lambda_d = clip(1 / max(delta_bar_d, 1), lam_min, 1)

    so one statistic simultaneously decides how fast the network forgets its
    weights and how fast the head forgets its evidence. Lambda is initialised at
    tau0 (the prior precision on w0) and floored at `lam_floor` so that a long
    run of near-zero torque -- which carries no information about w -- cannot
    drive the precision to zero and make w = b/Lambda blow up.
    """

    def __init__(self, w0, tau0=1.0, lam_min=1e-3, lam_floor=1e-8,
                 sigma_n2_floor=1e-8):
        self.w0 = np.asarray(w0, dtype=np.float64).copy()
        self.n_dims = len(self.w0)
        self.tau0 = float(tau0)
        self.lam_min, self.lam_floor = float(lam_min), float(lam_floor)
        self.sigma_n2_floor = float(sigma_n2_floor)
        self.reset()

    def reset(self):
        self.w = self.w0.copy()
        self.Lam = np.full(self.n_dims, self.tau0, dtype=np.float64)
        self.b = self.Lam * self.w0
        self.n = 0
        return self

    # ── the discount, shared with retention-mode forgetting ──────────────────
    def lam_of(self, delta_bar):
        db = np.asarray(delta_bar, dtype=np.float64)[:self.n_dims]
        return np.clip(1.0 / np.maximum(db, 1.0), self.lam_min, 1.0)

    def update(self, u, r, sigma_n2, delta_bar):
        """One transition.  u scalar, r and sigma_n2 per-dim, delta_bar per-dim."""
        lam = self.lam_of(delta_bar)
        s2 = np.maximum(np.asarray(sigma_n2, dtype=np.float64)[:self.n_dims],
                        self.sigma_n2_floor)
        u = float(u)
        self.Lam = np.maximum(lam * self.Lam + (u * u) / s2, self.lam_floor)
        self.b = lam * self.b + (u * np.asarray(r, dtype=np.float64)[:self.n_dims]) / s2
        self.w = self.b / self.Lam
        self.n += 1
        return self.w, lam

    @property
    def w_std(self):
        """Posterior sd of w -- shrinks as torque-bearing evidence accumulates."""
        return 1.0 / np.sqrt(np.maximum(self.Lam, self.lam_floor))

    # ── composition ──────────────────────────────────────────────────────────
    def predict(self, bnn, dyn, obs, act, sample=False, n_groups=1, sample_w=False):
        """Composed (mean, logvar).  The network is always evaluated at u=0."""
        zero = torch.zeros_like(act)
        mean, logvar = bnn._run_network(dyn._get_model_input(obs, zero),
                                        sample=sample, num_weight_groups=n_groups)
        w = self._w_tensor(obs.device, act.shape[0], sample_w)
        mean = mean.clone()
        mean[:, :self.n_dims] = mean[:, :self.n_dims] + act * w
        return mean, logvar

    def _w_tensor(self, device, batch, sample_w):
        w = torch.as_tensor(self.w, dtype=torch.float32, device=device).view(1, -1)
        if not sample_w:
            return w
        sd = torch.as_tensor(self.w_std, dtype=torch.float32, device=device).view(1, -1)
        return w + sd * torch.randn(batch, self.n_dims, device=device)

    def attach(self, dyn, sample_w=True):
        """Route the planner's rollout through the composed model.  Idempotent.

        Wraps OneDTransitionRewardModel.sample: the network is called with a ZERO
        action and the head's contribution is added to the predicted delta.  With
        sample_w the head also contributes epistemic spread across the K
        posterior draws, which shrinks as Lambda grows -- so CVaR keeps a
        genuine tail over the action channel too, instead of treating w as known.
        """
        if getattr(self, "_orig_sample", None) is not None:
            return self
        orig = dyn.sample
        head = self

        def composed(act, model_state, deterministic=False, rng=None):
            nxt, rew, term, ns = orig(torch.zeros_like(act), model_state,
                                      deterministic=deterministic, rng=rng)
            w = head._w_tensor(act.device, act.shape[0],
                               sample_w and not deterministic)
            nxt = nxt.clone()
            nxt[:, :head.n_dims] = nxt[:, :head.n_dims] + act * w
            ns["obs"] = nxt
            return nxt, rew, term, ns

        dyn.sample = composed
        self._orig_sample = orig
        self._dyn = dyn
        return self

    def detach(self):
        if getattr(self, "_orig_sample", None) is not None:
            self._dyn.sample = self._orig_sample
            self._orig_sample = None
        return self

    def __repr__(self):
        return (f"LinearActionHead(w={np.array2string(self.w, precision=5)}, "
                f"w_std={np.array2string(self.w_std, precision=5)}, n={self.n})")


@torch.no_grad()
def measure_gain(bnn, dyn, obs, n_dims=3, u=1.0):
    """w0: central difference d mu / du of the frozen network, per output dim."""
    d = []
    for s in (-u, u):
        a = torch.full((obs.shape[0], 1), s, device=obs.device)
        m, _ = bnn._run_network(dyn._get_model_input(obs, a), sample=False)
        d.append(m[:, :n_dims])
    return ((d[1] - d[0]) / (2 * u)).mean(0).cpu().numpy().astype(np.float64)


@torch.no_grad()
def equivalence_error(head, bnn, dyn, obs, acts):
    """||mu_BNN(s,u) - [mu_BNN(s,0) + w0*u]|| -- how exactly the warm-started
    head reproduces the network it replaces.  If this is ~0 the composed model
    is the BNN pre-change and the surprise statistics are undisturbed."""
    direct, _ = bnn._run_network(dyn._get_model_input(obs, acts), sample=False)
    comp, _ = head.predict(bnn, dyn, obs, acts, sample=False)
    e = (comp[:, :head.n_dims] - direct[:, :head.n_dims])
    return float(e.pow(2).sum(-1).sqrt().mean()), float(e.abs().mean(0).max())


@torch.no_grad()
def surprise_composed(head, bnn, dyn, obs, act, next_obs, n_draws=8, eps=1e-12,
                      sample_w=True):
    """delta_n for the COMPOSED model: same chi-square calibration as
    surprise_gaussian, but the mean includes the head and the epistemic term
    includes the head's own posterior spread.  Scored on state dims only."""
    o = obs.unsqueeze(0) if obs.ndim == 1 else obs
    a = act.view(1, -1)
    tgt = (next_obs - obs).view(1, -1)[:, :head.n_dims]
    saved, bnn.num_weight_groups = bnn.num_weight_groups, 1
    try:
        ms, lvs = [], []
        for _ in range(n_draws):
            m, lv = head.predict(bnn, dyn, o, a, sample=True, sample_w=sample_w)
            ms.append(m[:, :head.n_dims]); lvs.append(lv[:, :head.n_dims])
    finally:
        bnn.num_weight_groups = saved
    ms = torch.stack(ms)
    mu = ms.mean(0)
    S = ms.var(0, unbiased=False) + torch.exp(torch.stack(lvs)).mean(0) + eps
    nu = tgt - mu
    return (nu.pow(2) / S).squeeze(0).cpu().numpy().astype(np.float64), \
           nu.squeeze(0).cpu().numpy().astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
class NonlinearAdapterHead(nn.Module):
    r"""Gradient-trained, expressive adapter over a FROZEN body.

        mu(s,u) = mu_BNN(s, 0) + h_phi(s, u)
        h_phi(x) = Skip(x) + L2(silu(L1(x))),      x = [cos, sin, thdot, u]

    This replaces LinearActionHead's closed-form conjugate fit.  The trade it
    makes: the linear head is EXACT for anything of the form w*u (a mass change)
    and structurally blind to everything else (a gravity change lives in
    3g/(2l)*sin(theta), which no multiple of u can express -- measured: w drifting
    0.09..0.18 with delta_n median 8.1).  This head takes the full state as input,
    so a passive-dynamics change is inside its span.  It pays for that with
    gradient training, a KL, and no closed form.

    THREE STRUCTURAL CHOICES

    (i) Body evaluated at u=0, exactly as before.  mu_BNN(s,0) stays in-support
        whatever torque range the body was pretrained on (measured: the |u|<=2
        checkpoint is off by 6.97 when extrapolated to |u|=20, versus 0.026 at
        u=0), and it isolates the passive dynamics the body already gets right.

    (ii) WARM START.  `Skip.weight_mu` is zeroed except the u column, which is set
        to the body's own measured gain w0, and `L2.weight_mu` is zeroed with a
        tiny sigma.  So at initialisation h(s,u) = w0*u and the composed model
        reproduces the body -- the same stability property the conjugate head
        had, now expressed as an initialisation instead of a prior mean.  The
        nonlinear branch starts at zero and only earns its output.

    (iii) ALL THREE LAYERS ARE BAYESIAN, so the ELBO is a genuine variational
        posterior over the adapter and the head contributes epistemic spread to
        the planner's K rollout draws (which shrinks as it learns), rather than
        being a point estimate bolted onto a Bayesian body.

    FORGETTING still applies, and to the ADAPTER rather than the body: retention
    inflation tau <- tau0 + rho(tau - tau0) with the same rho = 1/max(delta_bar,1)
    used everywhere else.  Skip and L2 have one row per output dim so they take
    the per-dim rho; L1 is shared across dims so it takes the strongest
    (min) rho.  `anchor()` then re-anchors the KL prior to the inflated belief so
    the next ELBO step contracts from it rather than from N(0, prior_std).
    """

    def __init__(self, w0, obs_dim=3, hid=64, prior_std=0.1, init_sigma=1e-3):
        super().__init__()
        from bnn.layers import BayesianLinear
        self.obs_dim = int(obs_dim)
        in_dim = obs_dim + 1
        self.skip = BayesianLinear(in_dim, obs_dim, prior_std=prior_std)
        self.l1 = BayesianLinear(in_dim, hid, prior_std=prior_std)
        self.l2 = BayesianLinear(hid, obs_dim, prior_std=prior_std)
        rho0 = math.log(math.expm1(init_sigma))
        with torch.no_grad():
            self.skip.weight_mu.zero_()
            self.skip.weight_mu[:, -1] = torch.as_tensor(w0, dtype=torch.float32)
            self.skip.bias_mu.zero_(); self.skip.weight_rho.fill_(rho0)
            self.skip.bias_rho.fill_(rho0)
            self.l2.weight_mu.zero_(); self.l2.bias_mu.zero_()
            self.l2.weight_rho.fill_(rho0); self.l2.bias_rho.fill_(rho0)
        self.anchor(include_sigma=True)
        self._orig_sample = None
        self._dyn = None

    @property
    def layers(self):
        return [self.skip, self.l1, self.l2]

    def forward(self, x, sample=True):
        return self.skip(x, sample=sample) + self.l2(
            F.silu(self.l1(x, sample=sample)), sample=sample)

    def kl(self):
        return sum(l.kl_divergence() for l in self.layers)

    def anchor(self, include_sigma=True):
        for l in self.layers:
            l.anchor_prior_to_current(include_sigma=include_sigma)

    # ── composition (mirrors LinearActionHead.predict / attach) ──────────────
    def predict(self, bnn, dyn, obs, act, sample=False, n_groups=1):
        """Composed (mean, logvar).  Body always evaluated at u=0."""
        mean, logvar = bnn._run_network(dyn._get_model_input(obs, torch.zeros_like(act)),
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
            nxt, rew, term, ns = orig(torch.zeros_like(act), model_state,
                                      deterministic=deterministic, rng=rng)
            obs = model_state["obs"]
            with torch.no_grad():
                h = head.forward(torch.cat([obs, act], dim=-1),
                                 sample=not deterministic)
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

    # ── surprise-driven retention on the ADAPTER's own posterior ─────────────
    @torch.no_grad()
    def forget(self, delta_bar):
        """tau <- tau0 + rho (tau - tau0),  rho = 1/max(delta_bar, 1).  Bounded by
        the prior, so it cannot ratchet.  Returns rho actually applied per dim."""
        db = np.asarray(delta_bar, dtype=np.float64)[:self.obs_dim]
        rho = np.clip(1.0 / np.maximum(db, 1.0), 1e-3, 1.0)
        if (rho >= 1.0).all():
            return rho, False
        for layer, per_dim in ((self.skip, True), (self.l2, True), (self.l1, False)):
            tau0 = 1.0 / (layer.prior_std ** 2)
            for mu_p, rho_p in ((layer.weight_mu, layer.weight_rho),
                                (layer.bias_mu, layer.bias_rho)):
                for r in range(rho_p.shape[0]):
                    rr = float(rho[r]) if per_dim else float(rho.min())
                    if rr >= 1.0:
                        continue
                    sigma = F.softplus(rho_p.data[r])
                    tau_new = tau0 + rr * (1.0 / (sigma ** 2) - tau0)
                    rho_p.data[r] = torch.log(
                        torch.expm1((1.0 / tau_new).sqrt()).clamp_min(1e-6))
        return rho, True


def elbo_step_adapter(head, opt, bnn, dyn, obs_b, act_b, tgt_b, active_dims,
                      n_steps=5, n_mc=3, beta=1.0, kl_denom=None, grad_clip=10.0):
    r"""One ELBO update of the adapter on head-corrected residuals.

        r      = delta_s_obs - mu_BNN(s, 0)                     (target)
        NLL    = 1/D sum_d 0.5[ log sigma_n,d^2 + (r_d - h_d)^2 / sigma_n,d^2 ]
        loss   = E_q[NLL] + beta * KL(q(phi) || p(phi)) / N

    sigma_n^2 is the FROZEN body's aleatoric channel at (s, 0) -- the same noise
    model the conjugate head used, so switching linear->nonlinear changes only
    the mean function and the fitting procedure, nothing else.  The body is never
    in the graph: only adapter parameters receive gradient.
    """
    active = sorted(int(d) for d in active_dims)
    N = float(kl_denom) if kl_denom else float(obs_b.shape[0])
    with torch.no_grad():
        m0, lv0 = bnn._run_network(
            dyn._get_model_input(obs_b, torch.zeros_like(act_b)), sample=False)
        r = tgt_b[:, :head.obs_dim] - m0[:, :head.obs_dim]
        s2 = torch.exp(lv0[:, :head.obs_dim]).clamp_min(1e-8)
    x = torch.cat([obs_b, act_b], dim=-1)
    nll = kl = None
    for _ in range(n_steps):
        opt.zero_grad()
        acc = 0.0
        for _ in range(n_mc):
            h = head(x, sample=True)
            per_d = [(0.5 * (torch.log(s2[:, d]) + (r[:, d] - h[:, d]) ** 2 / s2[:, d])
                      ).mean() for d in active]
            acc = acc + sum(per_d) / float(len(active))
        nll = acc / n_mc
        kl = head.kl()
        (nll + beta * kl / N).backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
        opt.step()
    return float(nll.item()), float(kl.item())
