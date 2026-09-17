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
    d = bnn.grid.direction_of(s, a, s2)
    return dict(delta_n=delta_n, nll=nll, entropy=entropy,
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
