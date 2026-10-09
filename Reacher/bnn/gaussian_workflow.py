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
    # PER-DIM vectors, before the reduction.  The scalars below are exactly their
    # means, so every existing caller is unaffected; the vectors let a caller run
    # one drift filter / inflation / retrain gate PER OUTPUT DIM instead of
    # averaging the dims into a single statistic first (see
    # inflate_head_rows_by_process_noise and drift.PerDimDriftFilter).  Each
    # delta_n_vec[d] is separately calibrated to E[.] = 1, so the filters need no
    # per-dim rescaling.
    delta_vec = (nu ** 2 / S)[0]
    nu2_vec = (nu ** 2)[0]
    delta_n = float(delta_vec.mean().item())
    return dict(delta_n=delta_n,
                epistemic=float(epi_s.mean().item()),
                aleatoric=float(ale_s.mean().item()),
                S=float(S.mean().item()),
                nu2=float(nu2_vec.mean().item()),
                delta_n_vec=delta_vec.cpu().numpy(),
                nu2_vec=nu2_vec.cpu().numpy(),
                epistemic_vec=epi_s[0].cpu().numpy(),
                aleatoric_vec=ale_s[0].cpu().numpy())


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


# ── Per-row (per-output-dim) re-inflation ─────────────────────────────────────
# The head is a DIAGONAL Gaussian: bayes_layers[-1].weight_mu is (2*out_size, hid),
# where row d feeds mean_d and row out_size+d feeds logvar_d.  With the trunk frozen
# those two rows are the ONLY parameters that touch output dim d, so a per-dim drift
# estimate can be applied to exactly its own rows and to nothing else.  The trunk is
# shared across dims and has no per-dim rows, so it is never inflated here.

def _head_rows_for(out_size, d):
    """The two head rows owned by output dim d: (mean row, logvar row)."""
    return d, out_size + d


def inflate_head_rows_by_process_noise(bnn, q_vec, out_size, forget_mean=FORGET_MEAN):
    """Additive process noise on ONLY the head rows of each output dim.

    q_vec[d] <= 0 leaves dim d completely untouched -- including the learned-reward
    channel, which carries no q when it is excluded from the surprise.  The sigma
    update is identical to inflate_by_process_noise:
        sigma_w^2 <- sigma_w^2 + q_d * prior_var.
    Returns the number of dims actually inflated.
    """
    q_vec = np.asarray(q_vec, dtype=np.float64)
    head = bnn.bayes_layers[-1]
    prior_var = head.prior_std ** 2
    n_applied = 0
    with torch.no_grad():
        for d, q_hat in enumerate(q_vec):
            if q_hat <= 0.0:
                continue
            q = float(q_hat) * prior_var
            for r in _head_rows_for(out_size, d):
                for mu_p, rho_p in ((head.weight_mu, head.weight_rho),
                                    (head.bias_mu, head.bias_rho)):
                    sigma = F.softplus(rho_p.data[r])
                    sigma_new = (sigma ** 2 + q).sqrt()
                    rho_p.data[r] = torch.log(
                        torch.expm1(sigma_new).clamp_min(1e-6))
                    if forget_mean:
                        mu_p.data[r].mul_(1.0 / (1.0 + float(q_hat)))
            n_applied += 1
    return n_applied


def inflate_head_rows_toward_prior(bnn, rho_vec, out_size, forget_mean=FORGET_MEAN):
    """Per-row RETENTION pull, the bounded counterpart of the additive path:
        tau <- tau0 + rho_d (tau - tau0)   on dim d's two head rows only.

    rho_d >= 1 is a no-op for that dim.  Returns the number of dims pulled.
    """
    rho_vec = np.asarray(rho_vec, dtype=np.float64)
    head = bnn.bayes_layers[-1]
    tau0 = 1.0 / (head.prior_std ** 2)
    n_applied = 0
    with torch.no_grad():
        for d, rho in enumerate(rho_vec):
            if rho >= 1.0:
                continue
            for r in _head_rows_for(out_size, d):
                for mu_p, rho_p in ((head.weight_mu, head.weight_rho),
                                    (head.bias_mu, head.bias_rho)):
                    sigma = F.softplus(rho_p.data[r])
                    tau_new = tau0 + float(rho) * (1.0 / (sigma ** 2) - tau0)
                    sigma_new = (1.0 / tau_new).sqrt()
                    rho_p.data[r] = torch.log(
                        torch.expm1(sigma_new).clamp_min(1e-6))
                    if forget_mean:
                        mu_p.data[r].mul_(float(rho))
            n_applied += 1
    return n_applied


def forget_gaussian_perdim(bnn, perdim_filter, out_size, inflate_mode=INFLATE_MODE,
                           q_scale=Q_SCALE, q_max=Q_MAX, deadband=LAMBDA_DEADBAND,
                           kappa=KAPPA, forget_mean=FORGET_MEAN):
    """Per-row re-inflation: every output dim forgets on its OWN drift evidence.

    Mirrors forget_gaussian's deadband / cap semantics, applied per dim: dim d
    inflates only if its own drift estimate clears the deadband.  Both modes are
    supported, since the two envs default differently (Pendulum "additive",
    LunarLander "retention"):

        additive  : sigma_d^2 <- sigma_d^2 + q_d * prior_var,  q_d capped at q_max
        retention : tau_d     <- tau0 + rho_d (tau_d - tau0),  rho_d = 1/max(dbar_d,1)

    Returns (applied, fired):
      applied  (out_size,) the q_d (additive) or rho_d (retention) actually used,
                           0 / 1 respectively for dims that did not fire;
      fired    (out_size,) bool -- the per-dim retrain gate.  Dims the filter does
                           not score (e.g. the reward channel) are always False.
    """
    est = np.asarray(perdim_filter.drift_estimate(kappa), dtype=np.float64)
    n = min(len(est), out_size)
    q = np.zeros(out_size, dtype=np.float64)
    q[:n] = q_scale * est[:n]
    # deadband may be a scalar or PER-DIM.  Per-dim is what an unbiased (calibrated)
    # baseline needs: with the baseline at the model's own clean-data level the
    # excess is zero-mean under no change, so without a per-dim noise threshold the
    # filter fires on half the clean-data noise.
    db = np.zeros(out_size, dtype=np.float64)
    db[:] = np.broadcast_to(np.asarray(deadband, dtype=np.float64), (out_size,))
    fired = np.zeros(out_size, dtype=bool)
    fired[:n] = q[:n] > db[:n]

    if inflate_mode == "additive":
        applied = np.where(fired, np.minimum(q, q_max), 0.0)
        inflate_head_rows_by_process_noise(bnn, applied, out_size,
                                           forget_mean=forget_mean)
    elif inflate_mode == "retention":
        dbar = np.ones(out_size, dtype=np.float64)
        dbar[:n] = np.asarray(perdim_filter.delta_bar, dtype=np.float64)[:n]
        applied = np.where(fired, np.clip(1.0 / np.maximum(dbar, 1.0), 1e-3, 1.0), 1.0)
        # rho == 1 is already a no-op, so a dim that fired but whose dbar <= 1 is
        # simply not pulled -- same self-limiting behaviour as the scalar path.
        fired &= applied < 1.0
        inflate_head_rows_toward_prior(bnn, applied, out_size, forget_mean=forget_mean)
    else:
        raise ValueError(f"unknown inflate_mode {inflate_mode!r}")
    return applied, fired


def forget_gaussian_head_only(bnn, drift_filter, out_size, inflate_mode=INFLATE_MODE,
                              q_scale=Q_SCALE, q_max=Q_MAX, deadband=LAMBDA_DEADBAND,
                              kappa=KAPPA, forget_mean=FORGET_MEAN):
    """CONTROL for forget_gaussian_perdim: the ordinary SCALAR drift estimate, but
    applied to the head rows only (trunk untouched).  Isolates "head-only scope"
    from "per-dim signal" -- without it, per-dim mode changes both at once and any
    effect is unattributable.  Returns the q_hat (additive) or rho (retention)
    actually used, or 0.0 if the estimate was below the deadband.
    """
    q_hat = q_scale * drift_filter.drift_estimate(kappa)
    if q_hat <= deadband:
        return 0.0
    if inflate_mode == "additive":
        q_hat = min(q_hat, q_max)
        inflate_head_rows_by_process_noise(bnn, np.full(out_size, q_hat), out_size,
                                           forget_mean=forget_mean)
        return q_hat
    elif inflate_mode == "retention":
        rho = float(np.clip(1.0 / max(drift_filter.delta_bar, 1.0), 1e-3, 1.0))
        inflate_head_rows_toward_prior(bnn, np.full(out_size, rho), out_size,
                                       forget_mean=forget_mean)
        return rho
    raise ValueError(f"unknown inflate_mode {inflate_mode!r}")


@torch.no_grad()
def head_row_sigma(bnn, out_size):
    """Mean posterior width sigma over each output dim's own head rows, (out_size,).

    mean_sigma averages the WHOLE net, which barely moves once only the head is
    inflated; this is the readout that actually tracks per-row forgetting.
    """
    head = bnn.bayes_layers[-1]
    w_sig, b_sig = F.softplus(head.weight_rho.data), F.softplus(head.bias_rho.data)
    out = np.zeros(out_size, dtype=np.float64)
    for d in range(out_size):
        rows = _head_rows_for(out_size, d)
        tot = sum(float(w_sig[r].sum() + b_sig[r]) for r in rows)
        cnt = sum(w_sig[r].numel() + 1 for r in rows)
        out[d] = tot / cnt
    return out


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


def retrain_gaussian_perdim(bnn, opt, model_in, target, active_dims, out_size,
                            n_dims_total, n_steps=5):
    """Gated per-dim retrain of the diagonal head's rows.

    Only the dims in `active_dims` are updated; the rest are left EXACTLY as they
    were.  Two details make that true rather than approximately true:

      1. The loss is normalised by `n_dims_total`, not by len(active_dims), so
         gating a dim off does not silently scale up every surviving dim's
         gradient.  With all dims active this is bit-identical to the
         `.mean()`-over-dims normalisation retrain_gaussian uses.
      2. Inactive rows are snapshotted and RESTORED after opt.step().  A zero
         gradient is not enough to freeze them: Adam keeps stepping from stale
         momentum (exp_avg decays by beta1 but stays non-zero), so a dim that
         fired earlier would keep drifting on every later step.

    Returns (total_nll, per_dim_nll) where per_dim_nll[d] is NaN for inactive dims.
    """
    active = sorted(int(d) for d in active_dims)
    per_dim = np.full(out_size, np.nan)
    if not active:
        return None, per_dim

    head = bnn.bayes_layers[-1]
    keep = torch.ones(head.weight_mu.shape[0], dtype=torch.bool,
                      device=head.weight_mu.device)          # True == restore (frozen)
    for d in active:
        for r in _head_rows_for(out_size, d):
            keep[r] = False

    total = None
    for _ in range(n_steps):
        opt.zero_grad()
        mean, logvar = bnn._run_network(model_in, sample=False)
        losses = [(0.5 * (logvar[:, d] + (target[:, d] - mean[:, d]) ** 2
                          / torch.exp(logvar[:, d]))).mean() for d in active]
        total = sum(losses) / float(n_dims_total)
        total.backward()
        w_snap = head.weight_mu.data.clone()
        b_snap = head.bias_mu.data.clone()
        opt.step()
        head.weight_mu.data[keep] = w_snap[keep]
        head.bias_mu.data[keep] = b_snap[keep]
    for d, l in zip(active, losses):
        per_dim[d] = float(l.item())
    return float(total.item()), per_dim


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


# ── Per-dimension ELBO retrain (mean AND predictive variance) ─────────────────
# retrain_gaussian_perdim above is mu-only: its forward is sample=False, so
# weight_rho is not in the graph and receives exactly zero gradient no matter
# what sits in the optimizer.  The pair below closes that gap per dimension:
# a SAMPLED (reparameterized) forward puts rho in the graph, and a KL restricted
# to dim d's own head rows makes the width a posterior rather than a free
# parameter -- so forget's inflation can actually be CONTRACTED by evidence
# instead of ratcheting up forever.


def _kl_head_rows(head, rows):
    """KL( q || p ) summed over just `rows` of the head layer (weights + biases).

    Same Gaussian KL as BayesianLinear.kl_divergence, but row-restricted, so a
    dim that did not fire contributes no gradient and no constant."""
    idx = torch.as_tensor(list(rows), device=head.weight_mu.device, dtype=torch.long)
    w_mu, w_sig = head.weight_mu[idx], F.softplus(head.weight_rho[idx])
    b_mu, b_sig = head.bias_mu[idx], F.softplus(head.bias_rho[idx])
    pw_mu, pw_sig = head.prior_weight_mu[idx], head.prior_weight_sigma[idx]
    pb_mu, pb_sig = head.prior_bias_mu[idx], head.prior_bias_sigma[idx]

    def _kl(mu, sig, pmu, psig):
        return 0.5 * ((sig / psig) ** 2 + ((mu - pmu) / psig) ** 2 - 1.0
                      + 2.0 * (torch.log(psig) - torch.log(sig))).sum()

    return _kl(w_mu, w_sig, pw_mu, pw_sig) + _kl(b_mu, b_sig, pb_mu, pb_sig)


@torch.no_grad()
def anchor_head_rows_to_current(bnn, dims, out_size, include_sigma=True):
    """ROW-WISE counterpart of bnn.anchor_prior_to_current.

    The layer-level version copies whole tensors, so calling it after a per-dim
    forget would also re-anchor dims that never fired and destroy their priors.
    This snapshots only the rows owned by `dims`, which is what makes per-dim
    forget -> retrain a genuine predict/update pair: the KL below then contracts
    FROM the just-inflated belief instead of from N(pretrained, prior_std=1)."""
    head = bnn.bayes_layers[-1]
    for d in dims:
        for r in _head_rows_for(out_size, int(d)):
            head.prior_weight_mu.data[r] = head.weight_mu.data[r]
            head.prior_bias_mu.data[r] = head.bias_mu.data[r]
            if include_sigma:
                head.prior_weight_sigma.data[r] = F.softplus(head.weight_rho.data[r])
                head.prior_bias_sigma.data[r] = F.softplus(head.bias_rho.data[r])


def retrain_gaussian_perdim_elbo(bnn, opt, model_in, target, active_dims, out_size,
                                 n_dims_total, n_steps=5, n_mc=3, beta=1.0,
                                 kl_denom=None, grad_clip=10.0,
                                 local_reparam=True):
    """Per-dim ELBO retrain: updates BOTH the mean and the predictive variance of
    each active output dim, and touches nothing else.

    Differences from retrain_gaussian_perdim (mu-only), all required for rho to
    move at all:
      1. the forward is SAMPLED, so weight_rho is in the graph;
      2. the loss carries a KL over exactly the active dims' head rows -- bare
         NLL under a sampled forward drives sigma -> 0 (a deterministic net);
      3. inactive rows are snapshotted and restored for mu AND rho after
         opt.step(), because a zero gradient does not freeze a parameter under
         Adam (stale momentum keeps stepping it).

    The NLL is normalised by `n_dims_total`, not len(active_dims), so gating a
    dim off does not silently rescale every surviving dim's gradient.

    Returns (nll, kl, per_dim_sigma) from the last step; per_dim_sigma[d] is the
    mean posterior width over dim d's head rows (NaN for inactive dims).
    """
    active = sorted(int(d) for d in active_dims)
    per_dim_sigma = np.full(out_size, np.nan)
    if not active:
        return None, None, per_dim_sigma

    head = bnn.bayes_layers[-1]
    rows = [r for d in active for r in _head_rows_for(out_size, d)]
    keep = torch.ones(head.weight_mu.shape[0], dtype=torch.bool,
                      device=head.weight_mu.device)          # True == restore (frozen)
    keep[torch.as_tensor(rows, device=keep.device, dtype=torch.long)] = False

    denom = float(kl_denom) if kl_denom else float(model_in.shape[0])
    params = [p for g in opt.param_groups for p in g["params"]]
    nll = kl = None
    for _ in range(n_steps):
        opt.zero_grad()
        nll_acc = 0.0
        for _ in range(n_mc):
            # local_reparam: one INDEPENDENT weight draw per buffer row
            # (num_weight_groups = B).  With the default single draw shared
            # across the batch, the sampling error is common-mode -- every row
            # is perturbed by the same weights -- so averaging over the buffer
            # cannot cancel it, and after forget inflates sigma the shared draw
            # swamps the signal (measured: theta_dot mean wobbles by 13.1 vs a
            # true per-step delta of 0.05, rows 0.60-correlated).  Per-row draws
            # decorrelate it, so the batch mean is a real Monte-Carlo average.
            mean, logvar = bnn._run_network(
                model_in, sample=True,
                num_weight_groups=(model_in.shape[0] if local_reparam else 1))
            per_d = [(0.5 * (logvar[:, d] + (target[:, d] - mean[:, d]) ** 2
                             / torch.exp(logvar[:, d]))).mean() for d in active]
            nll_acc = nll_acc + sum(per_d) / float(n_dims_total)
        nll = nll_acc / n_mc
        kl = _kl_head_rows(head, rows)
        loss = nll + beta * kl / denom
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        w_mu_s, b_mu_s = head.weight_mu.data.clone(), head.bias_mu.data.clone()
        w_rho_s, b_rho_s = head.weight_rho.data.clone(), head.bias_rho.data.clone()
        opt.step()
        head.weight_mu.data[keep] = w_mu_s[keep]
        head.bias_mu.data[keep] = b_mu_s[keep]
        head.weight_rho.data[keep] = w_rho_s[keep]
        head.bias_rho.data[keep] = b_rho_s[keep]

    with torch.no_grad():
        sig = head_row_sigma(bnn, out_size)
    for d in active:
        per_dim_sigma[d] = sig[d]
    return float(nll.item()), float(kl.item()), per_dim_sigma


def retrain_gaussian_full_elbo(bnn, opt, model_in, target, active_dims, n_dims_total,
                               n_steps=5, n_mc=3, beta=1.0, kl_denom=None,
                               grad_clip=10.0, local_reparam=True):
    """WHOLE-NETWORK ELBO retrain -- trunk + head, mean and variance.

    The per-dim variant can only ever adapt a linear readout on features frozen
    at the pre-change dynamics: the head is diagonal (row d owns output dim d) but
    the trunk is SHARED across dims, so there is no per-dim decomposition of it.
    This trains every BayesianLinear instead, which is the only way the learned
    FEATURES can change -- at the cost of losing the per-dim gating (a trunk
    weight contributes to all outputs at once).

    The NLL is still restricted to `active_dims` and normalised by
    `n_dims_total`, so the learned-reward channel stays out of the objective
    exactly as before; its head rows then move only under the KL.

    IMPORTANT: the KL here covers every layer, and an unanchored prior is
    N(0, prior_std=1) -- fitting against that would drag the whole pretrained
    trunk toward zero.  Call bnn.anchor_prior_to_current(include_sigma=True)
    once before the first update so the KL is a trust region around the
    pretrained weights rather than a pull toward the origin.

    Returns (nll, kl) from the last step.
    """
    active = sorted(int(d) for d in active_dims)
    if not active:
        return None, None
    denom = float(kl_denom) if kl_denom else float(model_in.shape[0])
    params = [p for g in opt.param_groups for p in g["params"]]
    nll = kl = None
    for _ in range(n_steps):
        opt.zero_grad()
        nll_acc = 0.0
        for _ in range(n_mc):
            mean, logvar = bnn._run_network(
                model_in, sample=True,
                num_weight_groups=(model_in.shape[0] if local_reparam else 1))
            per_d = [(0.5 * (logvar[:, d] + (target[:, d] - mean[:, d]) ** 2
                             / torch.exp(logvar[:, d]))).mean() for d in active]
            nll_acc = nll_acc + sum(per_d) / float(n_dims_total)
        nll = nll_acc / n_mc
        kl = bnn._total_kl()
        loss = nll + beta * kl / denom
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
    return float(nll.item()), float(kl.item())
