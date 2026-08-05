"""Gaussian-native drift workflow: surprise / epistemic / re-inflate (forget).

The NON-Dirichlet counterpart to bnn.dirichlet_workflow.  These operate on a
BayesianDynamicsModel (bnn) + its OneDTransitionRewardModel wrapper (dyn).

Surprise here is the Mahalanobis chi-square of the one-step innovation in the
model's raw output space (delta-obs + reward), calibrated so E[delta_n] = 1:

    nu = target - mu_bar,   S = epistemic + aleatoric,   delta_n = mean(nu^2 / S).

Re-inflation ("forget") adds the filtered squared-drift estimate back onto the
variational weight covariance -- either as ADDITIVE process noise
(sigma_w^2 <- sigma_w^2 + q_hat * prior_var, unbounded) or as a RETENTION pull
toward the prior (tau <- tau0 + rho (tau - tau0), bounded).  This is the
"rectifier re-inflation" of the Gaussian pipeline: the drift filter's rectified
excess drives how much epistemic uncertainty is put back.

Extracted from notebooks/bnn_fl_cem_surprise_drift.py (surprise_score, forget,
inflate_*) and notebooks/diagnose_v2.py (verbose_surprise, mean_sigma).
"""
import numpy as np
import torch
import torch.nn.functional as F

from config import device, KAPPA

# ── Surprise (Mahalanobis chi-square) ─────────────────────────────────────────
N_SURPRISE_DRAWS = 20      # weight draws (M) for the predictive covariance S
SURPRISE_EPS = 1e-6        # floor on S_i to avoid divide-by-zero
# The pretrained model is near-deterministic (tiny S), so one off-prediction yields
# an astronomically large standardized residual.  Clip the per-dim-normalized delta
# before it reaches the filter so a single step cannot dominate the whole window.
DELTA_CLIP = 1e3

# ── Forgetting (process-noise re-inflation) ───────────────────────────────────
# "retention": tau <- tau0 + rho (tau - tau0), rho = 1/max(delta_bar,1)  (bounded
#              pull toward the prior -- the self-limiting default).
# "additive" : sigma_w^2 <- sigma_w^2 + q_hat * prior_var  (unbounded process noise;
#              capped per application by Q_MAX to avoid overshoot).
INFLATE_MODE = "retention"
Q_SCALE = 1.0              # multiplies lambda_hat before it becomes process noise
Q_MAX = 2.0               # (additive) cap on q_hat per application, anti-overshoot
LAMBDA_DEADBAND = 0.0      # only inflate once the drift estimate exceeds this
FORGET_MEAN = False        # also shrink the weight MEANS toward the prior


@torch.no_grad()
def surprise_gaussian(dyn, bnn, obs, action, next_obs, reward,
                      n_draws=N_SURPRISE_DRAWS, eps=SURPRISE_EPS,
                      use_reward_dim=True):
    """Per-dim-standardized squared one-step innovation delta_n = delta / d.

    Innovation is taken in the model's raw output space (delta-obs + reward),
    where (mean, logvar) live and the chi-square calibration holds.  S combines
    epistemic (variance of per-draw means) and aleatoric (mean exp(logvar)).
    E[delta_n] = 1 when the model is calibrated, so delta_n - 1 is the per-dim
    excess (an estimate of lambda / d).  Returns the internals for logging.
    """
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(action, dtype=torch.float32, device=device)
    nxt_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    model_in = dyn._get_model_input(obs_t.unsqueeze(0), act_t.unsqueeze(0))
    target = torch.cat(
        [nxt_t - obs_t, torch.tensor([reward], dtype=torch.float32, device=device)]
    ).unsqueeze(0)

    means, logvars = [], []
    saved = bnn.num_weight_groups
    bnn.num_weight_groups = 1
    try:
        for _ in range(n_draws):
            m, lv = bnn._run_network(model_in, sample=True)
            means.append(m)
            logvars.append(lv)
    finally:
        bnn.num_weight_groups = saved
    means = torch.stack(means)                            # (M, 1, out)
    mu_bar = means.mean(0)                                # (1, out)
    epistemic = means.var(0, unbiased=False)             # (1, out)
    aleatoric = torch.exp(torch.stack(logvars)).mean(0)  # (1, out)

    # Which output channels the drift statistic is scored over.  The model's last
    # channel is the LEARNED REWARD.  When the planner uses an analytic
    # ground-truth reward, that channel's error is irrelevant to control yet still
    # perturbs the surprise -> filter -> forget path, so it can be excluded.
    if use_reward_dim:
        tgt_s, mu_s, epi_s, ale_s = target, mu_bar, epistemic, aleatoric
    else:
        tgt_s, mu_s = target[:, :-1], mu_bar[:, :-1]      # state dims only
        epi_s, ale_s = epistemic[:, :-1], aleatoric[:, :-1]

    S = epi_s + ale_s + eps
    nu = tgt_s - mu_s
    delta_n = float((nu ** 2 / S).mean().item())
    return dict(delta_n=delta_n,
                epistemic=float(epi_s.mean().item()),
                aleatoric=float(ale_s.mean().item()),
                S=float(S.mean().item()),
                nu2=float((nu ** 2).mean().item()))


@torch.no_grad()
def epistemic_gaussian(dyn, bnn, obs, action, n_draws=N_SURPRISE_DRAWS):
    """Epistemic / aleatoric readout for one (s,a): spread of the per-draw means
    (Var_w[mu], epistemic) and the mean output-Gaussian variance (aleatoric)."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(action, dtype=torch.float32, device=device)
    model_in = dyn._get_model_input(obs_t.unsqueeze(0), act_t.unsqueeze(0))
    means, logvars = [], []
    saved = bnn.num_weight_groups
    bnn.num_weight_groups = 1
    try:
        for _ in range(n_draws):
            m, lv = bnn._run_network(model_in, sample=True)
            means.append(m)
            logvars.append(lv)
    finally:
        bnn.num_weight_groups = saved
    means = torch.stack(means)
    epistemic = float(means.var(0, unbiased=False).mean().item())
    aleatoric = float(torch.exp(torch.stack(logvars)).mean(0).mean().item())
    return dict(epistemic=epistemic, aleatoric=aleatoric)


def inflate_by_process_noise(bnn, q_hat, forget_mean=FORGET_MEAN):
    """Additive process noise on the variational covariance: P <- P + Q_hat.

    Adds q_hat * prior_std^2 to every weight/bias variance, i.e.
        sigma_w^2 <- sigma_w^2 + q_hat * prior_std^2.
    Unbounded, so a large measured drift inflates the epistemic uncertainty past
    the prior -- the correct behavior when the drift says the parameters moved a lot.
    """
    if q_hat <= 0.0:
        return
    with torch.no_grad():
        for layer in bnn.bayes_layers:
            prior_var = layer.prior_std ** 2
            q = q_hat * prior_var
            for mu_p, rho_p in (
                (layer.weight_mu, layer.weight_rho),
                (layer.bias_mu, layer.bias_rho),
            ):
                sigma = F.softplus(rho_p.data)
                sigma_new = (sigma ** 2 + q).sqrt()
                rho_p.data.copy_(torch.log(torch.expm1(sigma_new).clamp_min(1e-6)))
                if forget_mean:
                    mu_p.data.mul_(1.0 / (1.0 + q_hat))


def inflate_toward_prior(bnn, rho, forget_mean=FORGET_MEAN):
    """Multiplicative retention pull: tau <- tau0 + rho (tau - tau0).

    Re-inflates the posterior variance toward (but not past) the prior by the
    retention factor rho = 1/max(delta_bar, 1).  Bounded / self-limiting.
    """
    if rho >= 1.0:
        return
    with torch.no_grad():
        for layer in bnn.bayes_layers:
            tau0 = 1.0 / (layer.prior_std ** 2)
            for mu_p, rho_p in (
                (layer.weight_mu, layer.weight_rho),
                (layer.bias_mu, layer.bias_rho),
            ):
                sigma = F.softplus(rho_p.data)
                tau = 1.0 / (sigma ** 2)
                tau_new = tau0 + rho * (tau - tau0)
                sigma_new = (1.0 / tau_new).sqrt()
                rho_p.data.copy_(torch.log(torch.expm1(sigma_new).clamp_min(1e-6)))
                if forget_mean:
                    mu_p.data.mul_(rho)


def forget_gaussian(bnn, drift_filter, inflate_mode=INFLATE_MODE,
                    q_scale=Q_SCALE, q_max=Q_MAX, deadband=LAMBDA_DEADBAND,
                    kappa=KAPPA, forget_mean=FORGET_MEAN):
    """Gaussian-native re-inflation.  Applies the configured mode and returns the
    q_hat (additive) or rho (retention) actually used (0.0 if below the deadband).

    forget_mean also SHRINKS the weight means toward the prior (mu *= 1/(1+q_hat)
    additive, mu *= rho retention).  This is distinct from gradient retraining:
    retraining ADDS new evidence (n), forget_mean REMOVES stale evidence (m).
    With forget_mean=False the pretrained mean keeps its full grip no matter how
    surprising the data is, and only the error bars widen.
    """
    q_hat = q_scale * drift_filter.drift_estimate(kappa)
    if q_hat <= deadband:
        return 0.0
    if inflate_mode == "additive":
        q_hat = min(q_hat, q_max)     # cap the per-application jump (anti-overshoot)
        inflate_by_process_noise(bnn, q_hat, forget_mean=forget_mean)
        return q_hat
    elif inflate_mode == "retention":
        rho = float(np.clip(1.0 / max(drift_filter.delta_bar, 1.0), 1e-3, 1.0))
        inflate_toward_prior(bnn, rho, forget_mean=forget_mean)
        return rho
    raise ValueError(f"unknown inflate_mode {inflate_mode!r}")


def adapter_params_gaussian(bnn, update_final_layer):
    """Mean weights of the online-adaptation stack: every inserted adapter plus,
    when update_final_layer, the Gaussian output layer.  Mu only -- the
    variational widths (rho) stay untouched, so forget / surprise keep their
    meaning."""
    params = []
    for layer in bnn.adapt_layers:
        params += [layer.weight_mu, layer.bias_mu]
    if update_final_layer:
        head = bnn.bayes_layers[-1]
        params += [head.weight_mu, head.bias_mu]
    return params


def trunk_params_gaussian(bnn, n_unfreeze):
    """SCOPE B -- true fine-tuning: mean weights of the top `n_unfreeze` PRETRAINED
    layers (output head first, then downward into the trunk).  Unlike the adapter
    stack (scope A), this thaws the ACTUAL pretrained weights rather than adding
    fresh identity layers.  Mu only -- rho (variational width) stays untouched so
    forget / surprise keep their meaning.  n_unfreeze=1 == head only (identical to
    adapter_params_gaussian(bnn, update_final_layer=True) with no adapters)."""
    params = []
    for layer in list(bnn.bayes_layers)[-n_unfreeze:]:
        params += [layer.weight_mu, layer.bias_mu]
    return params


def retrain_gaussian(bnn, opt, model_in, target, n_steps=5, use_reward_dim=True):
    """Gradient-retrain the adaptation stack on the post-change buffer.

    model_in : (B, in) normalized rows (as fed to _run_network);
    target   : (B, out) raw-space targets (delta-obs + reward).
    Loss is the heteroscedastic Gaussian NLL under the deterministic
    (mean-weight) forward.  Returns the last NLL.

    use_reward_dim=False drops the LEARNED REWARD channel from the loss, so the
    fit concentrates on the state dynamics -- the only thing the planner consumes
    when it scores rollouts with an analytic ground-truth reward.
    """
    nll = None
    for _ in range(n_steps):
        opt.zero_grad()
        mean, logvar = bnn._run_network(model_in, sample=False)
        if use_reward_dim:
            m_s, lv_s, tgt_s = mean, logvar, target
        else:
            m_s, lv_s, tgt_s = mean[:, :-1], logvar[:, :-1], target[:, :-1]
        nll = (0.5 * (lv_s + (tgt_s - m_s) ** 2 / torch.exp(lv_s))).mean()
        nll.backward()
        opt.step()
    return float(nll.item())


def retrain_layers_gaussian(bnn, n_unfreeze=1, use_adapters=True):
    """The LAYER OBJECTS the online retrain touches: every inserted adapter, plus
    the top `n_unfreeze` pretrained layers (output head first, then downward).

    Returned as layers rather than parameters so the caller can also compute the
    KL over exactly the trained subset and re-anchor exactly that subset."""
    layers = list(bnn.adapt_layers) if use_adapters else []
    if n_unfreeze > 0:
        layers += list(bnn.bayes_layers)[-n_unfreeze:]
    return layers


def variational_params_gaussian(layers, train_rho=True):
    """Parameters of `layers` to hand the optimizer.

    train_rho=False reproduces the mu-only scope of adapter_params_gaussian /
    trunk_params_gaussian.  train_rho=True ALSO optimizes the variational widths
    -- which only does anything under retrain_gaussian_elbo's SAMPLED forward;
    under retrain_gaussian's `sample=False` forward rho is not in the graph and
    would silently receive zero gradient."""
    params = []
    for layer in layers:
        params += [layer.weight_mu, layer.bias_mu]
        if train_rho:
            params += [layer.weight_rho, layer.bias_rho]
    return params


def retrain_gaussian_elbo(bnn, opt, model_in, target, kl_layers, n_steps=5,
                          n_mc=3, beta=1.0, kl_denom=None, grad_clip=10.0,
                          use_reward_dim=True, params=None):
    """Retrain mu AND rho on the post-change buffer by minimizing the negative ELBO.

    Two differences from `retrain_gaussian`, both required for the widths to move:

      1. the forward is SAMPLED (reparameterized: w = mu + softplus(rho) * eps),
         so weight_rho is in the graph and receives gradient.  retrain_gaussian
         uses sample=False, under which rho gets exactly zero gradient no matter
         what is in the optimizer.
      2. the loss carries a KL term.  Bare NLL under a sampled forward drives
         sigma -> 0 (a deterministic net); the KL is what makes the width a
         posterior rather than a free parameter.

    `kl_layers` is the subset whose KL enters the loss -- normally the same layers
    being optimized, so frozen layers do not contribute a constant.  `kl_denom`
    scales the KL (the standard mean-field VI `NLL + KL/N`); pass the buffer size
    for a per-observation Bayesian update.  Larger kl_denom => the data dominates
    => stronger contraction of sigma.

    IMPORTANT: what the KL contracts TOWARD is whatever the prior buffers hold.
    Call `bnn.anchor_prior_to_current(include_sigma=True, layers=kl_layers)` right
    after forgetting to make this a predict/update pair; otherwise the prior is
    N(pretrained_mu, prior_std=1) and the KL will push sigma UP toward 1.0.

    Returns (nll, kl, mean_sigma_of_kl_layers) from the last step.
    """
    B = model_in.shape[0]
    denom = float(kl_denom) if kl_denom else float(B)
    if params is None:
        params = [p for g in opt.param_groups for p in g["params"]]
    nll = kl = None
    for _ in range(n_steps):
        opt.zero_grad()
        nll_acc = 0.0
        for _ in range(n_mc):
            mean, logvar = bnn._run_network(model_in, sample=True)
            if use_reward_dim:
                m_s, lv_s, tgt_s = mean, logvar, target
            else:
                m_s, lv_s, tgt_s = mean[:, :-1], logvar[:, :-1], target[:, :-1]
            nll_acc = nll_acc + (0.5 * (lv_s + (tgt_s - m_s) ** 2
                                        / torch.exp(lv_s))).mean()
        nll = nll_acc / n_mc
        kl = sum(layer.kl_divergence() for layer in kl_layers)
        loss = nll + beta * kl / denom
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
    with torch.no_grad():
        sig = torch.cat([F.softplus(l.weight_rho.data).flatten() for l in kl_layers])
        sig_mean = float(sig.mean())
    return float(nll.item()), float(kl.item()), sig_mean


def mean_sigma(bnn):
    """Average posterior std (softplus(rho)) over all variational weights/biases --
    a scalar summary of how much epistemic uncertainty the model currently holds."""
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for layer in bnn.bayes_layers:
            for rho_p in (layer.weight_rho, layer.bias_rho):
                s = F.softplus(rho_p.data)
                tot += float(s.sum()); cnt += s.numel()
    return tot / cnt
