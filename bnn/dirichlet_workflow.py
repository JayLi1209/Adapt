"""Dirichlet-native drift workflow: epistemic / surprise / re-inflate (forget).

These operate on a DirichletDynamicsModel (bnn) + its OneDTransitionRewardModel
wrapper (dyn).  Extracted verbatim from bnn_fl_dirichlet.py; only the imports
changed (model constants now come from bnn.dirichlet_model, KAPPA from config).
"""
import numpy as np
import torch

from config import device
from bnn.dirichlet_model import N_STATES, N_ACTIONS, SURPRISE_EPS, direction_of


@torch.no_grad()
def _alpha_draws(dyn, bnn, obs, action, n_draws):
    """n_draws posterior weight samples of (p_cells, p_dir, alpha) for one (s,a)."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(action, dtype=torch.float32, device=device)
    model_in = dyn._get_model_input(obs_t.unsqueeze(0), act_t.unsqueeze(0))
    p_cells, p_dirs, alphas = [], [], []
    saved = bnn.num_weight_groups
    bnn.num_weight_groups = 1
    try:
        for _ in range(n_draws):
            alpha, _, _, cells = bnn._forward_alpha(model_in, sample=True)
            p_dir = alpha / alpha.sum(dim=-1, keepdim=True)
            p_cells.append(bnn._cells_from_dir(p_dir, cells)[0])
            p_dirs.append(p_dir[0])
            alphas.append(alpha[0])
    finally:
        bnn.num_weight_groups = saved
    return (torch.stack(p_cells), torch.stack(p_dirs), torch.stack(alphas))


@torch.no_grad()
def epistemic_dirichlet(dyn, bnn, obs, action, n_draws=20):
    """Dirichlet-native epistemic readout for one (s,a).

    epistemic : spread of the predictive categorical p across posterior weight
                draws (Var_w[p], averaged over cells).
    alpha0    : posterior-mean total concentration (model's own confidence).
    entropy   : entropy of the mean predictive categorical (aleatoric spread).
    """
    p_cells, p_dirs, alphas = _alpha_draws(dyn, bnn, obs, action, n_draws)
    p_bar = p_cells.mean(0)
    epistemic = float(p_cells.var(0, unbiased=False).mean().item())
    alpha0 = float(alphas.sum(-1).mean().item())
    nz = p_bar.clamp_min(SURPRISE_EPS)
    entropy = float((-(nz * torch.log(nz)).sum()).item())
    return dict(epistemic=epistemic, alpha0=alpha0, entropy=entropy,
                p_dir=p_dirs.mean(0).cpu().numpy(), p_cells=p_bar.cpu().numpy())


@torch.no_grad()
def reference_entropy_table(dyn, bnn, n_states=N_STATES, n_actions=N_ACTIONS,
                            n_draws=20, smooth=0.0):
    """H(p0) for every (s,a) under the CURRENT belief -- the fixed reference.

    Snapshot this ONCE (at the change notification, before any adaptation) and
    hand it to surprise_dirichlet as `ref_entropy`.  See that function for why a
    frozen denominator is the point.

    smooth (eps) : mix the reference predictive toward UNIFORM over the k_dir
        directions before taking its entropy,

            p0_eps = (1 - eps) * p0 + eps * (1/k_dir)

        WHY THIS MATTERS.  With a change at ts=0 the snapshot p0 is the
        PRETRAINED belief, which here is (near-)deterministic: H(p0) -> 0, so an
        unsmoothed fixed denominator is SMALLER than the live one on a typical
        step and inflates the bulk of delta_n instead of taming it (measured:
        median delta_n rose 0.79 -> 7.13 at p'=0.5).  eps floors H_0 at the
        entropy of the eps-mixture, independent of how sharp p0 is:

            eps 0.10 -> H_0 ~ 0.29     eps 0.20 -> H_0 ~ 0.49
            eps 0.15 -> H_0 ~ 0.39     eps 0.30 -> H_0 ~ 0.64

        For scale, the TRUE post-change entropies are 0.394 (p'=0.9), 1.040
        (p'=0.5), 1.096 (p'=0.3), and log k_dir = 1.099 is the maximum.  eps is
        therefore the assumed "how non-deterministic could the world be" prior
        that the surprise is measured against.

        The mixture is applied on the DIRECTION simplex (before the scatter onto
        cells), because that is where the model's k_dir-way choice lives; walls
        merge several directions onto one cell, so mixing post-scatter would
        spread mass onto unreachable cells.

        0.0 (default) = no smoothing, the raw reference entropy.

    Returns a (n_states, n_actions) float array.
    """
    eps = float(smooth)
    K = int(bnn.k_dir)
    H = np.zeros((n_states, n_actions), dtype=np.float64)
    for s in range(n_states):
        obs = np.zeros(n_states, dtype=np.float32); obs[s] = 1.0
        for a in range(n_actions):
            act = np.zeros(n_actions, dtype=np.float32); act[a] = 1.0
            p_cells, p_dirs, _ = _alpha_draws(dyn, bnn, obs, act, n_draws)
            if eps > 0.0:
                # Smooth on the direction simplex, then scatter to cells the
                # same way the model does, so H_0 lives in the same space as
                # the live entropy it replaces.
                p_dir = p_dirs.mean(0)
                p_dir = (1.0 - eps) * p_dir + eps / K
                cells = bnn.dir_cells[s, a].unsqueeze(0)          # (1,K)
                p_bar = bnn._cells_from_dir(p_dir.unsqueeze(0), cells)[0]
            else:
                p_bar = p_cells.mean(0)
            p_bar = p_bar.clamp_min(SURPRISE_EPS)
            H[s, a] = float((-(p_bar * torch.log(p_bar)).sum()).item())
    return H


@torch.no_grad()
def surprise_dirichlet(dyn, bnn, obs, action, next_obs, reward, n_draws=20,
                       ref_entropy=None):
    """Calibrated categorical surprise (the Dirichlet replacement for delta_n).

        delta_n = -log p(s2) / H(p)        (weight-averaged predictive p)

    Because E_{s2~p}[-log p(s2)] = H(p), this has E[delta_n] = 1 under a correct
    model -- an *expected* slip costs ~1 and does NOT register as drift.

    THE PROBLEM WITH THE LIVE DENOMINATOR.  H(p) is the entropy of the model's
    CURRENT predictive, which the conjugate counts sharpen every step.  As the
    head approaches determinism the numerator grows only logarithmically
    (-log p_slip) while H -> 0 fast, so delta_n ~ 1/H DIVERGES: measured median
    delta_n at the first slip is 95 at p'=0.9 vs 26-28 at p'=0.3/0.1, i.e. the
    MILDEST change produces the LARGEST surprise, because its prior survived
    longest and was sharpest when the slip landed.  Worse, the spread scales the
    same way (sd 27.9 at q=0.999 vs 0.33 at q=0.5), so one rare slip dominates
    the equal-weight drift mean.

    ref_entropy : (n_states, n_actions) array of H(p0) captured ONCE from a
        reference predictive p0 -- in practice the belief frozen at the change
        point.  When given,

            delta_n = -log p_t(s2) / H_0(s,a)

        E[delta_n] = H(p_t)/H(p0), which is EXACTLY 1 while p_t == p0 (the null
        "nothing has changed"), and falls below 1 as the live head sharpens --
        correctly reporting "less surprising than the reference".  The spike is
        then bounded by -log p_t(s2)/H_0: LOGARITHMIC in p_slip rather than 1/H.
        None (default) reproduces the shipped live-entropy behaviour exactly.
    """
    s = int(np.argmax(obs)); a = int(np.argmax(action)); s2 = int(np.argmax(next_obs))
    p_cells, p_dirs, alphas = _alpha_draws(dyn, bnn, obs, action, n_draws)
    p_bar = p_cells.mean(0).clamp_min(SURPRISE_EPS)        # (16,)
    nll = float(-torch.log(p_bar[s2]).item())
    entropy = float((-(p_bar * torch.log(p_bar)).sum()).item())
    denom = entropy if ref_entropy is None else float(ref_entropy[s, a])
    delta_n = nll / (denom + SURPRISE_EPS)
    d = bnn.grid.direction_of(s, a, s2)
    return dict(delta_n=delta_n, nll=nll, entropy=entropy,
                ref_entropy=(None if ref_entropy is None else denom),
                alpha0=float(alphas.sum(-1).mean().item()),
                p_dir=p_dirs.mean(0).cpu().numpy(), direction=d,
                p_reached=float(p_bar[s2].item()))


def forget_dirichlet(bnn, drift_filter, strength=1.0, rho_floor=1e-3,
                     trigger=1.0, symmetric=False, lambda_cap=None,
                     compound=True):
    """Dirichlet re-inflation.  Two modes, selected by `bnn.anchor_forget`:

    ANCHOR (Algorithm 3 line 26, alpha_0 a NAMED anchor):

        alpha <- alpha_0 + rho * (alpha - alpha_0),   alpha_0 = bnn.conc_prior

    An affine pull applied to the persistent per-(s,a) alpha.  alpha_0 is a fixed
    point: repeated forgetting converges TO alpha_0 and stays there, so a
    non-symmetric anchor such as [1,5,1] actually shapes the forgotten belief.

    SHRINK (the shipped Algorithm 2 path, default):

        retain <- retain * rho,  with  alpha = alpha_0 + retain*(alpha_head-alpha_0)

    Algebraically an affine pull too, but `retain` COMPOUNDS toward 0, so the
    head's contribution is scaled out and alpha collapses onto alpha_0's *scale*
    (~0.1 per direction) -- far below the counts, which are added at full weight.
    That is why an anchor set only at init has no lasting effect here.

    rho == 1 (delta_bar <= 1, "nothing changed") is a no-op in both modes.
    Returns (rho, retain_before, retain_after); in anchor mode the retain values
    are reported unchanged (retain is not the mechanism there).

    MAGNITUDE KNOBS (both default to the shipped behaviour exactly):

    strength : interpolate the raw rho toward 1 (== no forgetting):

        rho_eff = 1 - strength * (1 - rho_raw)

        strength = 1.0 -> shipped rho (full-strength forgetting)
        strength = 0.5 -> half the shrink per firing
        strength = 0.0 -> forgetting disabled (rho_eff == 1, a no-op)

        This scales HOW HARD each firing pulls, leaving WHEN it fires
        (delta_bar > 1) untouched -- so it separates "forget too eagerly" from
        "forget too hard", which the detection threshold alone conflates.

    rho_floor : lower clip on rho_eff.  The shipped 1e-3 lets a single firing
        cut `retain` by 1000x; because SHRINK compounds (retain *= rho every
        K_FORGET steps) a run of firings drives retain -> 0 and discards the
        pretrained head entirely.  Raising the floor bounds the per-firing
        damage without changing the trigger.

    trigger : delta_bar threshold that must be EXCEEDED before forgetting fires
        at all.  The shipped 1.0 means "fire whenever the filter is above its
        calibrated baseline", which at a large change is essentially every step
        (~260 firings/trial at p'=0.9).  Raising it to e.g. 10 demands a much
        larger filtered surprise before any belief is discarded.

        Below the trigger this is a hard no-op (rho == 1).  ABOVE it the pull
        keeps its shipped magnitude 1/delta_bar -- deliberately NOT rescaled to
        trigger/delta_bar -- so this knob changes only WHEN forgetting fires and
        leaves HOW HARD to `strength`/`rho_floor`.  That keeps the trigger axis
        orthogonal to the rho_floor sweep, which found the magnitude knob inert.

    lambda_cap : ceiling L on the drift estimate that drives rho, applied as

        lambda_eff = min(lambda_hat, L)   ->   rho = 1 / (1 + lambda_eff)

        so a single firing can never pull harder than 1/(1+L).  This is the SAME
        arithmetic bound as rho_floor = 1/(1+L), but stated in DRIFT units rather
        than retain units, which is the natural scale for the thing being capped:
        lambda_hat at p'=0.9 spikes to ~18 on first detection (see
        first_detect_dbar in the p'=0.9 JSONs), and each such spike alone costs a
        ~19x cut in retain.  Capping at L=1 bounds every firing to rho >= 0.5
        while leaving the many small-lambda firings exactly as shipped -- the
        distinction from rho_floor is only bookkeeping for a single firing, but
        it makes the swept quantity commensurate with the logged lambda_hat.

        None (default) = uncapped, the shipped behaviour.

    compound : whether `retain` ACCUMULATES the pulls (shipped) or reflects only
        the CURRENT drift.

            compound=True  (shipped):  retain <- retain * rho
            compound=False (this):     retain <- rho

        The shipped rule makes retain the PRODUCT of every rho since the change,
        so it is monotonically non-increasing and can only ratchet toward 0: even
        a mild rho = 0.95 applied 200x leaves 1e-5.  Measured at p'=0.9 from the
        p=0.7 prior, median rho per firing is 0.61 while median retain is 5.5e-40
        -- the per-firing pull is moderate, the ACCUMULATION is total.

        With compound=False, retain is a function of the drift filter's CURRENT
        estimate alone.  It is bounded below by the current rho, and -- crucially
        -- it can RECOVER when the filter calms down, instead of being an
        absorbing state at 0.  On the same trace that puts median retain at 0.61
        instead of 5.5e-40, i.e. the head keeps ~60% of its pretrained belief
        while the conjugate counts supply the rest.

        This is the other half of the diagnosis: fixing the SIGNAL (see
        surprise_dirichlet's ref_entropy) leaves the SCHEDULE untouched, and a
        well-behaved lambda ~ 0.4 still annihilates retain in ~12 compounding
        steps.  compound=False attacks the schedule.

    symmetric : drive rho by the MAGNITUDE of the drift, ignoring its sign:

        rho = 1 / (1 + |lambda_hat|)        in (0, 1]

        The shipped (asymmetric) path computes rho = 1/max(delta_bar, 1) and is
        gated on delta_bar > trigger.  Since delta_bar = 1 + lambda_hat, BOTH the
        gate and the max() clamp discard every step where lambda_hat < 0 -- the
        filter saying "the new dynamics are LESS surprising than the calibrated
        baseline".  That is not a no-change signal: a change toward determinism
        (e.g. p 0.7 -> 0.9) drives lambda_hat negative while still invalidating
        the pretrained head, and the shipped path refuses to forget on it.
        Measured on results/fl_from07 (p'=0.9 from a p=0.7 prior), lambda_hat is
        negative on 14.1% of all steps and in 100/100 trials, concentrated at
        t=0-9 where 73-90% of trials sit below zero (median ~ -0.6).

        With symmetric=True a mismatch of size |lambda_hat| shrinks retain by the
        same factor whichever way it points, and the trigger gate is applied to
        |lambda_hat| rather than to the signed delta_bar.
    """
    if symmetric:
        lam = abs(float(drift_filter.delta_bar) - 1.0)      # |lambda_hat|
        # Cap AFTER the gate check below would be wrong (the gate must see the
        # true magnitude), so cap only the value that drives rho.
        lam_rho = lam if lambda_cap is None else min(lam, float(lambda_cap))
        # Gate on the MAGNITUDE, so a negative-lambda change is not discarded.
        # trigger is expressed in delta_bar units (shipped default 1.0), so the
        # equivalent magnitude threshold is trigger - 1.
        if lam <= float(trigger) - 1.0:
            before = float(bnn.retain.item())
            return 1.0, before, before
        rho_raw = float(np.clip(1.0 / (1.0 + lam_rho), 0.0, 1.0))
        rho = 1.0 - float(strength) * (1.0 - rho_raw)
        rho = float(np.clip(rho, rho_floor, 1.0))
        before = float(bnn.retain.item())
        if getattr(bnn, "anchor_forget", False):
            if rho < 1.0:
                bnn.anchor_pull(rho)
            return rho, before, before
        after = (before * rho if compound else rho) if rho < 1.0 else before
        bnn.retain.fill_(after)
        return rho, before, after
    if drift_filter.delta_bar <= float(trigger):
        before = float(bnn.retain.item())
        return 1.0, before, before
    # Gate above saw the TRUE delta_bar; here cap only the value driving rho.
    _db = float(drift_filter.delta_bar)
    if lambda_cap is not None:
        _db = min(_db, 1.0 + float(lambda_cap))
    rho_raw = float(np.clip(1.0 / max(_db, 1.0), 0.0, 1.0))
    rho = 1.0 - float(strength) * (1.0 - rho_raw)
    rho = float(np.clip(rho, rho_floor, 1.0))
    before = float(bnn.retain.item())
    if getattr(bnn, "anchor_forget", False):
        if rho < 1.0:
            bnn.anchor_pull(rho)
        return rho, before, before
    after = (before * rho if compound else rho) if rho < 1.0 else before
    bnn.retain.fill_(after)
    return rho, before, after


def unfrozen_params_dirichlet(bnn, n_unfrozen):
    """Mean weights of the top `n_unfrozen` layers of the DIRECTION path, counted
    from the Dirichlet head downward into the PRETRAINED trunk.

        n_unfrozen = 0 -> nothing trained (fully frozen; original pipeline)
        n_unfrozen = 1 -> Dirichlet head only            (head-only adaptation)
        n_unfrozen = 2 -> head + top trunk layer         (unfreeze 1 pretrained)
        n_unfrozen = 3 -> head + both trunk layers        (unfreeze 2 pretrained,
                                                            i.e. the whole path)

    This is the knob for the "how many pretrained layers stay frozen" experiment:
    with `num_layers=3` the trunk has 2 Bayesian layers, so the direction path is
    [trunk0, trunk1, dir_head] and n_unfrozen in {1,2,3} sweeps head-only ->
    head+1-trunk -> head+2-trunk.  Only mu (posterior means) are returned; the
    variational widths (rho) stay fixed so surprise / forget keep their meaning.
    """
    # Direction path ordered top(output) -> down(input): head, then trunk.
    path = [bnn.bayes_layers[bnn.n_trunk]]                    # Dirichlet head
    path += list(reversed(list(bnn.bayes_layers[:bnn.n_trunk])))  # trunk, top first
    n = max(0, int(n_unfrozen))
    params = []
    for layer in path[:n]:
        params += [layer.weight_mu, layer.bias_mu]
    return params


def retrain_dirichlet(bnn, opt, model_in, s2_idx, n_steps=5):
    """Gradient-retrain the adaptation stack on the post-change buffer.

    model_in : (B, obs+act) raw one-hot rows;  s2_idx : (B, 1) realized cells.
    Loss is the categorical NLL of the realized cell under the deterministic
    (mean-weight) forward -- counts/retain participate exactly as at plan time.
    Returns the last NLL.
    """
    nll = None
    for _ in range(n_steps):
        opt.zero_grad()
        mean, _ = bnn._run_network(model_in, sample=False)
        p = mean[:, :bnn.n_states].clamp_min(SURPRISE_EPS)
        # The loss function is intentionally missing the KL term for adaptation!
        nll = -torch.log(p.gather(1, s2_idx)).mean()
        nll.backward()
        opt.step()
    return float(nll.item())


@torch.no_grad()
def mean_alpha0(dyn, bnn, n_draws=4):
    """Average posterior-mean concentration alpha0 over all (s,a) -- a scalar
    confidence summary to log alongside retain."""
    tot, cnt = 0.0, 0
    for s in range(bnn.n_states):
        obs = np.zeros(bnn.n_states, dtype=np.float32); obs[s] = 1.0
        for a in range(bnn.n_actions):
            act = np.zeros(bnn.n_actions, dtype=np.float32); act[a] = 1.0
            _, _, alphas = _alpha_draws(dyn, bnn, obs, act, n_draws)
            tot += float(alphas.sum(-1).mean().item()); cnt += 1
    return tot / cnt
