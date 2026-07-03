"""Continuous-action CEM planner with CVaR scoring and BNN rollouts.

Uses the Gaussian-head BNN as a learned dynamics model to simulate trajectories.
Unlike planning/cvar_cem.py (grid-based, discrete FrozenLake), this planner
operates on continuous state/action spaces suitable for Pendulum.
"""

import numpy as np
import torch

from config import device

# ── Planner hyperparameters ────────────────────────────────────────────────────
H_PLAN = 10              # planning horizon
N_CEM_ITERS = 3          # CEM refinement iterations
N_CANDIDATES = 128       # action sequences per iteration
ELITE_FRAC = 0.1         # keep top fraction
CVAR_ALPHA = 1.0         # CVaR tail (1.0 = risk-neutral mean)
K_MODELS = 1             # posterior BNN draws per candidate (epistemic)
GAMMA = 0.99             # discount
ACTION_STD_INIT = 1.0    # initial action-sampling std


class ContinuousCEMAgent:
    """CEM over continuous action sequences, scored by BNN-rollout CVaR."""

    def __init__(self, dyn, bnn, obs_dim, act_dim, device=device,
                 horizon=H_PLAN, n_cem_iters=N_CEM_ITERS,
                 n_candidates=N_CANDIDATES, elite_frac=ELITE_FRAC,
                 k_models=K_MODELS, cvar_alpha=CVAR_ALPHA, gamma=GAMMA,
                 action_std_init=ACTION_STD_INIT,
                 rng=None, **kwargs):
        self.dyn = dyn
        self.bnn = bnn
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.horizon = horizon
        self.n_cem_iters = n_cem_iters
        self.n_candidates = n_candidates
        self.n_elite = max(1, int(elite_frac * n_candidates))
        self.k_models = k_models
        self.cvar_alpha = cvar_alpha
        self.gamma = gamma
        self.rng = rng if rng is not None else np.random.default_rng(0)

        # CEM distribution: per-timestep Gaussian (mu, sigma) over actions.
        self.action_low = -2.0
        self.action_high = 2.0
        self._mu = np.zeros((horizon, act_dim), dtype=np.float32)
        self._sigma = np.full((horizon, act_dim), action_std_init, dtype=np.float32)
        self._sigma_min = 0.05

        self.surprise_bar = 1.0
        self.n_since_change = 0

    def reset(self):
        self._mu = np.zeros((self.horizon, self.act_dim), dtype=np.float32)
        self._sigma = np.full((self.horizon, self.act_dim),
                              ACTION_STD_INIT, dtype=np.float32)
        self.surprise_bar = 1.0
        self.n_since_change = 0

    def notify_change(self):
        self.reset()

    @torch.no_grad()
    def _rollout_one(self, obs0, action_seq):
        """Rollout one action sequence through K BNN posterior draws.

        Returns (K,) numpy array of discounted returns.
        """
        H = len(action_seq)
        K = self.k_models
        returns = np.zeros(K, dtype=np.float32)

        for k in range(K):
            obs_t = torch.as_tensor(obs0, dtype=torch.float32, device=self.device)
            obs_t = obs_t.unsqueeze(0)                                       # (1, obs_dim)
            state = self.dyn.reset(obs_t)
            total = 0.0
            disc = 1.0

            for t in range(H):
                act_t = torch.as_tensor(action_seq[t], dtype=torch.float32, device=self.device)
                act_t = act_t.unsqueeze(0)                                   # (1, act_dim)

                next_obs, rew, _, _ = self.dyn.sample(act_t, state, deterministic=True)
                total += disc * float(rew.item())
                obs_t = next_obs
                state = self.dyn.reset(obs_t)
                disc *= self.gamma

            returns[k] = total

        return returns

    @torch.no_grad()
    def act(self, obs):
        """Return the first action of the optimised CEM mean sequence."""
        import time as _time
        _t0 = _time.time()
        obs_arr = np.asarray(obs, dtype=np.float32).ravel()
        H, A = self.horizon, self.act_dim
        J = self.n_candidates

        saved_groups = self.bnn.num_weight_groups
        self.bnn.num_weight_groups = 1
        try:
            for it in range(self.n_cem_iters):
                # Sample J action sequences from the current Gaussian.
                noise = self.rng.normal(size=(J, H, A)).astype(np.float32)
                candidates = self._mu + self._sigma * noise
                candidates = np.clip(candidates, self.action_low, self.action_high)

                # Evaluate each candidate.
                scores = np.zeros(J, dtype=np.float32)
                for j in range(J):
                    returns = self._rollout_one(obs_arr, candidates[j])
                    scores[j] = self._cvar_value(returns)

                # Select elites.
                elite_idx = np.argpartition(-scores, self.n_elite - 1)[:self.n_elite]
                elite_seq = candidates[elite_idx]  # (n_elite, H, A)

                # Refit: MLE of Gaussian.
                self._mu = elite_seq.mean(axis=0)
                self._sigma = np.maximum(elite_seq.std(axis=0), self._sigma_min)
        finally:
            self.bnn.num_weight_groups = saved_groups

        self._last_plan_time = _time.time() - _t0
        return self._mu[0].copy()

    def _cvar_value(self, returns):
        """Empirical CVaR_alpha: mean of the worst ceil(alpha*m) returns."""
        m = len(returns)
        k = max(1, int(np.ceil(self.cvar_alpha * m)))
        worst = np.sort(returns)[:k]
        return float(worst.mean())
