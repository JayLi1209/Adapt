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
                      n_draws=N_SURPRISE_DRAWS, eps=SURPRISE_EPS):
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
    S = epistemic + aleatoric + eps
    nu = target - mu_bar
    delta_n = float((nu ** 2 / S).mean().item())
    return dict(delta_n=delta_n,
                epistemic=float(epistemic.mean().item()),
                aleatoric=float(aleatoric.mean().item()),
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
                    kappa=KAPPA):
    """Gaussian-native re-inflation.  Applies the configured mode and returns a dict
    with keys: triggered (bool), q_hat/rho (float), sigma_before, sigma_after,
    delta_sigma, and optional dual-channel info."""
    # Handle DualDriftFilter
    if hasattr(drift_filter, 'mean_channel'):
        q_hat, ch_info = drift_filter.drift_estimate(kappa)
    else:
        q_hat = drift_filter.drift_estimate(kappa)
        ch_info = {}
    q_hat = q_scale * q_hat

    sigma_before = mean_sigma(bnn)

    if q_hat <= deadband:
        return {"triggered": False, "q_hat": q_hat,
                "sigma_before": sigma_before, "sigma_after": sigma_before,
                "delta_sigma": 0.0, **ch_info}

    if inflate_mode == "additive":
        q_hat = min(q_hat, q_max)
        inflate_by_process_noise(bnn, q_hat)
        sigma_after = mean_sigma(bnn)
        return {"triggered": True, "q_hat": q_hat,
                "sigma_before": sigma_before, "sigma_after": sigma_after,
                "delta_sigma": sigma_after - sigma_before, **ch_info}
    elif inflate_mode == "retention":
        rho = float(np.clip(1.0 / max(drift_filter.delta_bar, 1.0), 1e-3, 1.0))
        inflate_toward_prior(bnn, rho)
        sigma_after = mean_sigma(bnn)
        return {"triggered": True, "rho": rho,
                "sigma_before": sigma_before, "sigma_after": sigma_after,
                "delta_sigma": sigma_after - sigma_before, **ch_info}
    raise ValueError(f"unknown inflate_mode {inflate_mode!r}")


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
