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
    d = direction_of(s, a, s2)
    return dict(delta_n=delta_n, nll=nll, entropy=entropy,
                alpha0=float(alphas.sum(-1).mean().item()),
                p_dir=p_dirs.mean(0).cpu().numpy(), direction=d,
                p_reached=float(p_bar[s2].item()))


def forget_dirichlet(bnn, drift_filter):
    """Dirichlet-native re-inflation: retain the head's alpha toward the SYMMETRIC
    prior, which pulls the predictive mean toward uniform (so surprise drops).

        retain <- retain * rho,    rho = 1 / max(delta_bar, 1)  in (0, 1].

    rho == 1 (delta_bar <= 1, "nothing changed") is a no-op.  Returns
    (rho, retain_before, retain_after).
    """
    rho = float(np.clip(1.0 / max(drift_filter.delta_bar, 1.0), 1e-3, 1.0))
    before = float(bnn.retain.item())
    after = before * rho if rho < 1.0 else before
    bnn.retain.fill_(after)
    return rho, before, after


@torch.no_grad()
def mean_alpha0(dyn, bnn, n_draws=4):
    """Average posterior-mean concentration alpha0 over all (s,a) -- a scalar
    confidence summary to log alongside retain."""
    tot, cnt = 0.0, 0
    for s in range(N_STATES):
        obs = np.zeros(N_STATES, dtype=np.float32); obs[s] = 1.0
        for a in range(N_ACTIONS):
            act = np.zeros(N_ACTIONS, dtype=np.float32); act[a] = 1.0
            _, _, alphas = _alpha_draws(dyn, bnn, obs, act, n_draws)
            tot += float(alphas.sum(-1).mean().item()); cnt += 1
    return tot / cnt
