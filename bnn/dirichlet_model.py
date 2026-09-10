"""Dirichlet-head BNN world model for ns-gym FrozenLake.

A Dirichlet head on a shared Bayesian
trunk: the head emits concentrations alpha; the predictive categorical is
p = alpha/alpha0 (fed to the planner directly), and alpha0 = sum(alpha) is the
epistemic confidence the forgetting loop turns down.  See the docstring of
`DirichletDynamicsModel` and the workflow functions in dirichlet_workflow.py for
the full surprise / forget / learn loop.

Refactor note: BayesianLinear now lives in bnn.layers; runtime constants
(SAVE_DIR, device) in config.  Otherwise unchanged from bnn_fl_dirichlet.py.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mbrl.models as models

from bnn.layers import BayesianLinear
from config import SAVE_DIR, device
from grids import GridSpec, FROZENLAKE_4x4

# ── geometry / head config ─────────────────────────────────────────────────────
NROW = 4
NCOL = 4
N_STATES = NROW * NCOL
N_ACTIONS = 4
K_DIR = 3                    # Dirichlet categories: [intended, perp(a-1), perp(a+1)]
ALPHA_FLOOR = 1e-3          # floor on each concentration (keeps alpha > 0)
RETAIN_INIT = 1.0          # initial retention (1 = trust the head fully)
# Symmetric-prior concentration per direction.  `forget` retains the head's alpha
# toward THIS symmetric prior, so as drift accumulates the predictive mean is
# pulled toward uniform (max entropy) -- which is what makes a slip stop being
# surprising.  A *multiplicative* scale would leave the mean p=alpha/alpha0 fixed
# (scale cancels) and could never lower delta_n; a symmetric affine retention can.
# 1.0 = Dirichlet(1,1,1), the actual uniform distribution OVER the simplex, not
# just uniform in expectation.  Was 0.1 (pulls toward the simplex CORNERS: a
# "fully forgotten" posterior draw is then near one-hot on a random direction),
# which made cem_fir's CVaR-over-worst-30% get dominated by these pathological
# near-deterministic draws instead of real uncertainty.  Swept c in
# {0.01,0.03,0.1,0.3,1.0,3.0,10.0} on cliffwalking (2026-09-06/09): 1.0 was a
# clean, uniform win over 0.1 at every tested (config,p) point; 3.0/10.0 helped
# a few points further but not all (one point -- pretrain-p=0.7, p=0.3 --
# already trends back down past 1.0, consistent with the model getting sharp
# again at large c, the same "empty CVaR tail" failure plan_retain exists to
# avoid) and needs materially more validation to trust, so 1.0 (the principled,
# argument-free value) is the chosen default.  Only affects methods whose
# retain actually leaves 1.0 (cem_fir, bnn_rats_adaptive) -- see
# doc_tool/experiment_report_2026-09-04.md SS4-5 for the full sweep + the
# oracle_cem confidence-gate bug this investigation also turned up.
CONC_PRIOR = 1.0
SURPRISE_EPS = 1e-6
LOGVAR_MIN, LOGVAR_MAX = -10.0, 0.5

# Dedicated checkpoint for the Dirichlet model -- kept separate from the shared
# Gaussian `bnn_dynamics.pth` (which holds other models) so saving never clobbers
# it.  The Gaussian file is only ever *read* (trunk source) by load_pretrained_trunk.
DIRICHLET_CKPT = f"bnn_dirichlet_k{K_DIR}.pth"
GAUSSIAN_CKPT = "bnn_dynamics.pth"


def ckpt_name(grid, p=None) -> str:
    """Per-grid, per-original-p checkpoint filename.

    p is None or 1.0 -> the DETERMINISTIC model; FrozenLake at that setting keeps
    the shipped name bnn_dirichlet_k3.pth (never overwritten by a p-sweep).
    Otherwise the original intended-prob p is encoded, e.g.
        bnn_dirichlet_cliffwalking_k4_p0p7.pth
    """
    deterministic = (p is None) or (abs(float(p) - 1.0) < 1e-9)
    if grid.name == "frozenlake" and deterministic:
        return DIRICHLET_CKPT
    ptag = "" if deterministic else f"_p{float(p):g}".replace(".", "p")
    return f"bnn_dirichlet_{grid.name}_k{grid.k_dir}{ptag}.pth"


def _move(s, a, nrow=NROW, ncol=NCOL):
    """Deterministic (no-slip) next cell for action a."""
    r, c = divmod(int(s), ncol)
    if a == 0:   c = max(c - 1, 0)          # LEFT
    elif a == 1: r = min(r + 1, nrow - 1)   # DOWN
    elif a == 2: c = min(c + 1, ncol - 1)   # RIGHT
    elif a == 3: r = max(r - 1, 0)          # UP
    return r * ncol + c


def _dir_actions(a):
    """The 3 directions for action a: intended, then the two perpendiculars."""
    return [a, (a - 1) % N_ACTIONS, (a + 1) % N_ACTIONS]


def build_dir_cells(nrow=NROW, ncol=NCOL, n_actions=N_ACTIONS):
    """(n_states, n_actions, 3) long tensor of the cell each direction lands in."""
    n = nrow * ncol
    out = np.zeros((n, n_actions, K_DIR), dtype=np.int64)
    for s in range(n):
        for a in range(n_actions):
            out[s, a] = [_move(s, d, nrow, ncol) for d in _dir_actions(a)]
    return torch.from_numpy(out)


def direction_of(s, a, s2, nrow=NROW, ncol=NCOL):
    """Realized direction index 0/1/2 (intended/perp-/perp+) or -1 if no match."""
    for d_idx, d in enumerate(_dir_actions(a)):
        if _move(s, d, nrow, ncol) == int(s2):
            return d_idx
    return -1


class DirichletDynamicsModel(models.Model):
    """BNN with a Bayesian trunk and a single K=3 Dirichlet readout head.

    bayes_layers = [trunk hidden..., dir head, reward head]; `forget`/inflate
    utilities that iterate bayes_layers and touch weight_rho/bias_rho keep working.
    """

    def __init__(self, in_size, out_size, device, hid_size=256, num_layers=3,
                 prior_std=1.0, beta=0.1, num_mc_samples=3, num_weight_groups=1,
                 grid: GridSpec = FROZENLAKE_4x4):
        super().__init__(device)
        # Grid geometry (FrozenLake 4x4 by default -- the shipped behaviour).
        self.grid = grid
        self.n_states = grid.n_states
        self.n_actions = grid.n_actions
        self.k_dir = grid.k_dir
        self.in_size = in_size           # obs_dim + act_dim (raw one-hot concat)
        self.out_size = out_size         # obs_dim + 1 (16 cells + reward)
        self.beta = beta
        self.num_mc_samples = num_mc_samples
        self.num_weight_groups = num_weight_groups
        self.num_train_points = None
        self.last_pred_mean = None

        # Trunk: in -> hid -> ... -> hid (same shapes as the Gaussian model's
        # hidden layers, so a pretrained trunk can be partially loaded).
        hidden_sizes = [in_size] + [hid_size] * (num_layers - 1)
        layers = [
            BayesianLinear(hidden_sizes[i], hidden_sizes[i + 1], prior_std=prior_std)
            for i in range(len(hidden_sizes) - 1)
        ]
        self.n_trunk = len(layers)
        # Separate r and s' prediction so that they don't share the same output layer.
        layers.append(BayesianLinear(hid_size, self.k_dir, prior_std=prior_std))  # dir head
        layers.append(BayesianLinear(hid_size, 2, prior_std=prior_std))       # reward head
        self.bayes_layers = nn.ModuleList(layers)

        # Retention factor in (0, 1]; forget lowers it to pull alpha toward the
        # symmetric prior CONC_PRIOR (mean -> uniform, entropy up, surprise down).
        self.register_buffer("retain", torch.tensor(float(RETAIN_INIT)))
        # Direction -> cell geometry (fixed grid).
        self.register_buffer("dir_cells", grid.build_dir_cells())

        # Pretraining-concentration table: alpha0 per (s,a) = pretraining
        # directional counts + K*CONC_PRIOR (the exact conjugate posterior
        # concentration).  The categorical NLL only sees the mean p=alpha/alpha0
        # and cannot identify alpha0 (KL pins it near the prior scale ~7 even
        # with 20k samples), so the head supplies the DIRECTION (p mean) while
        # this table supplies the SCALE.  Zeros = unset -> fall back to the
        # head's softplus magnitude (old checkpoints keep their behaviour).
        # Because the table feeds alpha_head BEFORE the retention transform,
        # `forget` attenuates pretraining confidence exactly like the head.
        self.register_buffer("pretrain_alpha0",
                             torch.zeros(self.n_states, self.n_actions))

        # Conjugate Dirichlet-Multinomial online counts: when `use_counts` is on,
        # per-(s,a) directional counts are ADDED to the head's alpha, so the
        # predictive mean migrates toward the empirical post-change frequencies and
        # the concentration (alpha0) tightens as evidence arrives -- the "learn"
        # half of the loop (forget is the "loosen" half).
        self.use_counts = False
        self.register_buffer("counts",
                             torch.zeros(self.n_states, self.n_actions, self.k_dir))

        # Whether `loss` trains the reward head.  Default True for generality.
        self.learn_reward = True

        self.to(device)

    def add_count(self, s, a, d, w=1.0):
        """Accumulate one realized directional outcome (d in 0/1/2) for (s,a)."""
        if d >= 0:
            self.counts[int(s), int(a), int(d)] += w

    def set_pretrain_counts(self, counts):
        """Fill the pretraining-concentration table from directional counts.

        counts: (n_states, n_actions, K) array/tensor of pretraining
        directional outcomes; alpha0[s,a] = counts.sum(-1) + K*CONC_PRIOR
        (the conjugate Dirichlet posterior concentration).  Rows with zero
        counts are left unset (0 -> head magnitude fallback).
        """
        if not torch.is_tensor(counts):
            counts = torch.from_numpy(np.asarray(counts, dtype=np.float32))
        a0 = counts.sum(-1).float() + self.k_dir * CONC_PRIOR
        a0 = torch.where(counts.sum(-1) > 0, a0, torch.zeros_like(a0))
        self.pretrain_alpha0.copy_(a0.to(self.pretrain_alpha0.device))

    def reset_counts(self):
        self.counts.zero_()

    # ── core forward: features -> (alpha, reward mu/logvar, geometry) ──────────
    def _forward_alpha(self, x, sample=True, num_weight_groups=1):
        """Return per-row (alpha (B,K), r_mean (B,1), r_logvar (B,1), cells (B,K))."""
        s = x[:, :self.n_states].argmax(dim=-1)                          # (B,)
        a = x[:, self.n_states:self.n_states + self.n_actions].argmax(dim=-1)
        cells = self.dir_cells[s, a]                            # (B,K) long

        h = x
        for layer in self.bayes_layers[:self.n_trunk]:
            h = F.silu(layer(h, sample=sample, num_weight_groups=num_weight_groups))
        dir_head = self.bayes_layers[self.n_trunk]
        reward_head = self.bayes_layers[self.n_trunk + 1]
        # Separate readouts off the shared trunk features.
        raw_alpha = dir_head(h, sample=sample, num_weight_groups=num_weight_groups)  # (B,K)
        rew = reward_head(h, sample=sample, num_weight_groups=num_weight_groups)     # (B,2)
        r_mean = rew[:, 0:1]
        r_logvar = rew[:, 1:2].clamp(LOGVAR_MIN, LOGVAR_MAX)
        # Symmetric-prior retention: alpha = prior + retain * (alpha_head - prior).
        # retain == 1 -> the head's alpha; retain -> 0 -> all dims -> CONC_PRIOR
        # (equal), i.e. the predictive mean collapses to uniform.
        alpha_head = F.softplus(raw_alpha) + ALPHA_FLOOR
        pa0 = self.pretrain_alpha0[s, a]                            # (B,)
        if bool((pa0 > 0).any()):
            # Head gives the direction (mean), the pretraining-counts table
            # gives the concentration (conjugate posterior alpha0).  Rows
            # whose (s,a) was never pretrained keep the head's magnitude.
            p_mean = alpha_head / alpha_head.sum(-1, keepdim=True)
            alpha_head = torch.where(
                (pa0 > 0).unsqueeze(-1),
                p_mean * pa0.clamp_min(self.k_dir * CONC_PRIOR).unsqueeze(-1),
                alpha_head)
        alpha = (CONC_PRIOR + self.retain * (alpha_head - CONC_PRIOR)).clamp_min(ALPHA_FLOOR)
        if self.use_counts:
            # alpha_effective = (retained prior) + observed counts  (conjugate update)
            alpha = alpha + self.counts[s, a]
        return alpha, r_mean, r_logvar, cells

    def _cells_from_dir(self, p_dir, cells):
        """Scatter the K directional probs onto the 16-cell simplex (merges walls)."""
        B = p_dir.shape[0]
        p_cells = torch.zeros(B, self.n_states, device=p_dir.device, dtype=p_dir.dtype)
        p_cells.scatter_add_(1, cells, p_dir)
        return p_cells

    def _run_network(self, x, sample=True, num_weight_groups=1):
        """(mean, logvar) compatibility face.  mean = [p_cells(16), r_mean(1)]."""
        alpha, r_mean, r_logvar, cells = self._forward_alpha(
            x, sample=sample, num_weight_groups=num_weight_groups)
        p_dir = alpha / alpha.sum(dim=-1, keepdim=True)
        p_cells = self._cells_from_dir(p_dir, cells)
        mean = torch.cat([p_cells, r_mean], dim=-1)
        # Categorical (Bernoulli-per-cell) variance as the aleatoric stand-in.
        var_cells = (p_cells * (1.0 - p_cells)).clamp_min(SURPRISE_EPS)
        logvar = torch.cat(
            [torch.log(var_cells), r_logvar], dim=-1
        ).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mean, logvar

    def forward(self, x, sample=True, num_weight_groups=1):
        return self._run_network(x, sample=sample, num_weight_groups=num_weight_groups)

    # ── planner sampling interface ────────────────────────────────────────────
    def sample_1d(self, model_in, model_state, rng=None, deterministic=False):
        with torch.no_grad():
            if deterministic:
                mean, _ = self._run_network(
                    model_in, sample=False, num_weight_groups=self.num_weight_groups)
                self.last_pred_mean = mean.clone()
                return mean, {}
            # One coherent posterior weight draw -> alpha; then a Dirichlet draw
            # of p (so alpha0 / forget actually disperses the planner's samples).
            alpha, r_mean, r_logvar, cells = self._forward_alpha(
                model_in, sample=True, num_weight_groups=self.num_weight_groups)
            p_dir = torch.distributions.Dirichlet(alpha).sample()   # (B,K)
            p_cells = self._cells_from_dir(p_dir, cells)
            r = r_mean + torch.exp(0.5 * r_logvar) * torch.randn_like(r_mean)
            preds = torch.cat([p_cells, r], dim=-1)
            self.last_pred_mean = self._cells_from_dir(
                alpha / alpha.sum(-1, keepdim=True), cells)
        return preds, {}

    def reset_1d(self, obs, rng=None):
        return {}

    # ── training (categorical NLL on the realized cell; reward optional) ───────
    def loss(self, model_in, target=None):
        # When `learn_reward` is False the reward head is NOT trained -- reward is
        # supplied by a lookup table at plan time, so the model only needs the
        # transition (Dirichlet) head.
        B = model_in.shape[0]
        s2 = target[:, :self.n_states].argmax(dim=-1, keepdim=True)  # realized cell
        r_tgt = target[:, self.n_states:self.n_states + 1]
        nll_acc = torch.zeros(1, device=self.device)
        for _ in range(self.num_mc_samples):
            mean, logvar = self._run_network(model_in, sample=True)
            p_cells = mean[:, :self.n_states].clamp_min(SURPRISE_EPS)
            cat_nll = -torch.log(p_cells.gather(1, s2)).mean()
            nll_acc = nll_acc + cat_nll
            if self.learn_reward:
                r_mean = mean[:, self.n_states:self.n_states + 1]
                r_logvar = logvar[:, self.n_states:self.n_states + 1]
                rew_nll = 0.5 * (r_logvar + (r_tgt - r_mean) ** 2 / torch.exp(r_logvar))
                nll_acc = nll_acc + rew_nll.mean()
        avg_nll = nll_acc / self.num_mc_samples
        kl = self._total_kl()
        kl_denom = self.num_train_points if self.num_train_points else B
        loss = avg_nll + self.beta * kl / kl_denom
        return loss, {"nll": avg_nll.item(), "kl": kl.item()}

    def loss_dm(self, model_in, dir_counts, r_tgt=None):
        """Dirichlet-multinomial NLL on aggregated per-(s,a) direction counts.

        The plain categorical `loss` depends only on the predictive mean
        p = alpha/alpha0, which is invariant to rescaling alpha -- so alpha0
        (the epistemic confidence the forget/learn loop modulates) receives
        NO learning signal from data and ends up set by the KL-to-prior term
        alone (~7 even with 20k samples).  The DM likelihood of aggregated
        counts identifies alpha0: many consistent outcomes push the
        concentration up with the sample count, exactly like the conjugate
        posterior.  Rows are weighted by their counts implicitly (ll sums
        over the n observations), so goal-weighted sampling also yields
        goal-near states being more CONFIDENT, not just more accurate.
        """
        dir_counts = dir_counts.to(model_in.device)
        n = dir_counts.sum(-1)                                   # (B,)
        nll_acc = torch.zeros(1, device=self.device)
        for _ in range(self.num_mc_samples):
            alpha, r_mean, r_logvar, _ = self._forward_alpha(model_in, sample=True)
            a0 = alpha.sum(-1)                                   # (B,)
            ll = (torch.lgamma(a0) - torch.lgamma(a0 + n)
                  + (torch.lgamma(alpha + dir_counts)
                     - torch.lgamma(alpha)).sum(-1))             # (B,)
            nll_acc = nll_acc + (-ll).mean()
            if self.learn_reward and r_tgt is not None:
                # Count-weighted Gaussian NLL on the per-(s,a) mean reward
                # (each row summarizes n reward observations).
                w = (n > 0).float()
                rm = r_mean.squeeze(-1)
                rl = r_logvar.squeeze(-1)
                rew_nll = 0.5 * (rl + (r_tgt - rm) ** 2 / torch.exp(rl)) * n * w
                nll_acc = nll_acc + rew_nll.mean()
        avg_nll = nll_acc / self.num_mc_samples
        kl = self._total_kl()
        kl_denom = self.num_train_points if self.num_train_points else model_in.shape[0]
        loss = avg_nll + self.beta * kl / kl_denom
        return loss, {"nll": avg_nll.item(), "kl": kl.item()}

    def eval_score(self, model_in, target=None):
        with torch.no_grad():
            mean, _ = self._run_network(model_in, sample=False)
            s2 = target[:, :self.n_states].argmax(dim=-1, keepdim=True)
            p = mean[:, :self.n_states].clamp_min(SURPRISE_EPS).gather(1, s2)
            score = (-torch.log(p)).expand(-1, 1)
        return score, {}

    def set_elite(self, elite_indices):
        pass

    def _total_kl(self):
        return sum(layer.kl_divergence() for layer in self.bayes_layers)

    def anchor_prior_to_current(self):
        for layer in self.bayes_layers:
            layer.anchor_prior_to_current()

    # ── persistence (partial trunk load from a Gaussian checkpoint) ───────────
    def save(self, save_dir, filename=DIRICHLET_CKPT):
        torch.save(self.state_dict(), pathlib.Path(save_dir) / filename)

    def load(self, load_dir, filename=DIRICHLET_CKPT):
        """Shape-tolerant load: copy every tensor whose name+shape matches, skip
        the rest (a checkpoint from a different head architecture loads what it
        can instead of crashing -- but warns, since a skipped head means retrain)."""
        ckpt = torch.load(pathlib.Path(load_dir) / filename, map_location=self.device)
        own = self.state_dict()
        match = {k: v for k, v in ckpt.items()
                 if k in own and own[k].shape == v.shape}
        skipped = [k for k in own if k not in match]
        own.update(match)
        self.load_state_dict(own, strict=False)
        # KL-prior buffers (prior_*_mu / prior_*_sigma) are absent from older
        # checkpoints BY DESIGN -- they correctly fall back to their init (zeros /
        # prior_std), so their absence is not a head mismatch and must not trigger
        # the "retrain the head" warning.
        head_skipped = [k for k in skipped
                        if k.startswith(f"bayes_layers.{self.n_trunk}")
                        and ".prior_" not in k]
        if head_skipped:
            print(f"WARNING: {filename} mismatched the head ({len(head_skipped)} "
                  f"tensors skipped) -- heads are FRESH; retrain pretrain_dirichlet.py.")
        return sorted(match.keys())

    def load_pretrained_trunk(self, load_dir, filename=GAUSSIAN_CKPT):
        """Load only the hidden-trunk weights whose shapes match a Gaussian
        checkpoint (the Dirichlet head + retain stay freshly initialized)."""
        ckpt = torch.load(pathlib.Path(load_dir) / filename, map_location=self.device)
        own = self.state_dict()
        loaded = {
            k: v for k, v in ckpt.items()
            if k in own and own[k].shape == v.shape
        }
        own.update(loaded)
        self.load_state_dict(own, strict=False)
        return sorted(loaded.keys())


def make_dirichlet_bnn(obs_dim, act_dim, grid: GridSpec = FROZENLAKE_4x4):
    """Dirichlet-head model wrapped so the planner reads p(s') directly.

    target_is_delta=False  -> sample() returns the categorical p(s') (no delta).
    normalize=False        -> model_in is the raw (obs,act) one-hot concat, so the
                              head can recover (s,a) by argmax for the geometry.

    Online adaptation is done by gradient-retraining the top layers of the
    direction path in place (see dirichlet_workflow.unfrozen_params_dirichlet);
    there is no inserted adapter stack.
    """
    bnn = DirichletDynamicsModel(
        in_size=obs_dim + act_dim,
        out_size=obs_dim + 1,
        device=device,
        hid_size=256,
        num_layers=3,
        prior_std=1.0,
        beta=0.1,
        num_mc_samples=3,
        num_weight_groups=1,
        grid=grid,
    )
    dynamics_model = models.OneDTransitionRewardModel(
        bnn, target_is_delta=False, normalize=False, learned_rewards=True
    )
    return bnn, dynamics_model
