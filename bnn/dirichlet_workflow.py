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
def surprise_dirichlet(dyn, bnn, obs, action, next_obs, reward, n_draws=20):
    """Calibrated categorical surprise (the Dirichlet replacement for delta_n).

        delta_n = -log p(s2) / H(p)        (weight-averaged predictive p)

    Because E_{s2~p}[-log p(s2)] = H(p), this has E[delta_n] = 1 under a correct
    model -- an *expected* slip costs ~1 and does NOT register as drift.
    """
    s = int(np.argmax(obs)); a = int(np.argmax(action)); s2 = int(np.argmax(next_obs))
    p_cells, p_dirs, alphas = _alpha_draws(dyn, bnn, obs, action, n_draws)
    p_bar = p_cells.mean(0).clamp_min(SURPRISE_EPS)        # (16,)
    nll = float(-torch.log(p_bar[s2]).item())
    entropy = float((-(p_bar * torch.log(p_bar)).sum()).item())
    delta_n = nll / (entropy + SURPRISE_EPS)
    # std of -log p(s2') under s2' ~ p (for the standardized score
    # (nll - H) / nll_std, 2026-09-24); informational, does not change delta_n.
    nll_std = float(torch.sqrt((p_bar * (-torch.log(p_bar) - entropy) ** 2)
                               .sum()).item())
    d = bnn.grid.direction_of(s, a, s2)
    return dict(delta_n=delta_n, nll=nll, entropy=entropy, nll_std=nll_std,
                alpha0=float(alphas.sum(-1).mean().item()),
                p_dir=p_dirs.mean(0).cpu().numpy(), direction=d,
                p_reached=float(p_bar[s2].item()))


def forget_dirichlet(bnn, drift_filter, rho_floor=1e-3, retain_floor=0.0):
    """Dirichlet-native re-inflation: retain the head's alpha toward the SYMMETRIC
    prior, which pulls the predictive mean toward uniform (so surprise drops).

        retain <- max(retain * rho, retain_floor),
        rho = clip(1 / max(delta_bar, 1), rho_floor, 1]

    rho == 1 (delta_bar <= 1, "nothing changed") is a no-op.  Returns
    (rho, retain_before, retain_after).

    `rho_floor` / `retain_floor` make forgetting GENTLER (2026-09-16,
    collaborator's "raise the retain factor / clip rho" suggestion).  Defaults
    reproduce the original behaviour exactly.

      rho_floor    : lower bound on a single tick's shrink factor.  The original
                     1e-3 lets one huge delta_bar crush retain by 1000x in a
                     single application; raising it (e.g. 0.5) caps how much any
                     one forget tick can forget, so retain decays gradually
                     instead of collapsing.
      retain_floor : hard lower bound on retain itself, i.e. a guaranteed
                     residual trust in the pretrained belief no matter how much
                     surprise accumulates.  Targets the known failure mode where
                     mild changes (p=0.9, the old prior still ~90% right) end up
                     WORSE than severe ones because retain is driven to ~0
                     regardless of how large the real change was.
    """
    rho = float(np.clip(1.0 / max(drift_filter.delta_bar, 1.0),
                        float(rho_floor), 1.0))
    before = float(bnn.retain.item())
    after = before * rho if rho < 1.0 else before
    after = max(after, float(retain_floor))
    bnn.retain.fill_(after)
    return rho, before, after


ML_RETAIN_GRID = np.concatenate([[0.0], np.logspace(-3, 0, 31)])


@torch.no_grad()
def ml_retain_dirichlet(bnn, dyn, buf, retain_grid=ML_RETAIN_GRID):
    """Evidence-calibrated retain (2026-09-23, "forget-mode ml").

    The rho = 1/delta_bar rule measures how SURPRISING the data is, not how
    much the env CHANGED: a model pretrained at p=1.0 has ~zero entropy, so ONE
    slip gives delta_bar ~ 1e3 whether the new p is 0.9 or 0.3, and retain is
    crushed to the 1e-3 floor either way.  At p=0.9 that throws away a belief
    that is still 90% right.

    Instead pick the retain that best EXPLAINS the post-change transitions:
        retain* = argmax_r  sum_{(s,a,s2) in buf} log pbar_r(s2 | s,a),
        pbar_r  = mean of Dir(CONC_PRIOR + r * (alpha_head - CONC_PRIOR))
    i.e. the same retain-blend the model already uses, without the online
    counts (they are the separate, local evidence channel).  Mild changes
    keep most of the prior (r* large), severe ones still go to ~0.

    buf : list of (obs_onehot, act_onehot, s2_idx).  Returns retain*.
    """
    from bnn.dirichlet_model import CONC_PRIOR, ALPHA_FLOOR
    if not buf:
        return float(bnn.retain.item())
    obs = torch.as_tensor(np.stack([b[0] for b in buf]), dtype=torch.float32,
                          device=device)
    act = torch.as_tensor(np.stack([b[1] for b in buf]), dtype=torch.float32,
                          device=device)
    s2 = torch.as_tensor([int(b[2]) for b in buf], dtype=torch.long,
                         device=device)
    model_in = dyn._get_model_input(obs, act)
    saved_r, saved_c = float(bnn.retain.item()), bnn.use_counts
    bnn.retain.fill_(1.0)
    bnn.use_counts = False
    try:
        alpha_head, _, _, cells = bnn._forward_alpha(model_in, sample=False)
    finally:
        bnn.retain.fill_(saved_r)
        bnn.use_counts = saved_c
    best_r, best_ll = saved_r, -np.inf
    for r in retain_grid:
        alpha = (CONC_PRIOR + float(r) * (alpha_head - CONC_PRIOR)).clamp_min(
            ALPHA_FLOOR)
        p_dir = alpha / alpha.sum(-1, keepdim=True)
        p_cells = bnn._cells_from_dir(p_dir, cells).clamp_min(SURPRISE_EPS)
        ll = float(torch.log(p_cells.gather(1, s2.unsqueeze(1))).sum().item())
        if ll > best_ll + 1e-9:
            best_r, best_ll = float(r), ll
    return best_r


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
