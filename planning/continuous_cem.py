"""Continuous-action CEM planner with batched GPU rollouts.

All J candidates are rolled out in parallel through the BNN at each timestep,
giving ~J x higher GPU utilization compared to sequential (batch=1) rollouts.
"""

import numpy as np
import torch

from config import device

# ── Planner hyperparameters ────────────────────────────────────────────────────
H_PLAN = 12              # planning horizon
N_CEM_ITERS = 5          # CEM refinement iterations
N_CANDIDATES = 256       # action sequences per iteration (must be even)
ELITE_FRAC = 0.1         # keep top fraction
CVAR_ALPHA = 1.0         # CVaR tail (1.0 = risk-neutral mean)
K_MODELS = 1             # posterior BNN draws (1 = fast, >1 = epistemic diversity)
GAMMA = 0.99             # discount


class ContinuousCEMAgent:
    """CEM over continuous action sequences with batched BNN rollouts."""

    def __init__(self, dyn, bnn, obs_dim, act_dim, device=device,
                 horizon=H_PLAN, n_cem_iters=N_CEM_ITERS,
                 n_candidates=N_CANDIDATES, elite_frac=ELITE_FRAC,
                 k_models=K_MODELS, cvar_alpha=CVAR_ALPHA, gamma=GAMMA,
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

        self.action_low = -2.0
        self.action_high = 2.0
        self._mu = np.zeros((horizon, act_dim), dtype=np.float32)
        self._sigma = np.full((horizon, act_dim), 1.0, dtype=np.float32)
        self._sigma_min = 0.05

        self.surprise_bar = 1.0
        self.n_since_change = 0

    def reset(self):
        self._mu = np.zeros((self.horizon, self.act_dim), dtype=np.float32)
        self._sigma = np.full((self.horizon, self.act_dim), 1.0, dtype=np.float32)

    def notify_change(self):
        self.reset()

    @torch.no_grad()
    def act(self, obs):
        obs_arr = np.asarray(obs, dtype=np.float32).ravel()
        H, A, J = self.horizon, self.act_dim, self.n_candidates
        K = self.k_models

        # Total batch = J * K (J candidates × K posterior draws each)
        total_batch = J * K
        saved_groups = self.bnn.num_weight_groups
        self.bnn.num_weight_groups = K if K > 1 else 1
        try:
            for _ in range(self.n_cem_iters):
                # Sample J action sequences: (J, H, A)
                noise = self.rng.normal(size=(J, H, A)).astype(np.float32)
                candidates = self._mu + self._sigma * noise
                candidates = np.clip(candidates, self.action_low, self.action_high)

                # Repeat each candidate K times for posterior diversity: (J*K, H, A)
                if K > 1:
                    act_seqs = np.repeat(candidates, K, axis=0)  # (J*K, H, A)
                else:
                    act_seqs = candidates

                # ── Batched rollout: all J*K trajectories in parallel ──────
                obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
                obs_t = obs_t.unsqueeze(0).expand(total_batch, -1)       # (J*K, obs_dim)
                state = self.dyn.reset(obs_t)
                returns = torch.zeros(total_batch, device=self.device)
                disc = 1.0

                for t in range(H):
                    act_t = torch.as_tensor(act_seqs[:, t, :], dtype=torch.float32,
                                            device=self.device)          # (J*K, act_dim)
                    next_obs, rew, _, _ = self.dyn.sample(
                        act_t, state, deterministic=True)

                    returns += disc * rew.squeeze(-1)
                    obs_t = next_obs
                    state = self.dyn.reset(obs_t)
                    disc *= self.gamma

                returns_np = returns.cpu().numpy()                       # (J*K,)

                # ── Score each candidate as CVaR over its K returns ──────
                if K > 1:
                    returns_grouped = returns_np.reshape(J, K)           # (J, K)
                    scores = np.array([self._cvar_value(returns_grouped[j])
                                       for j in range(J)])
                else:
                    scores = returns_np

                # ── Select elites and refit Gaussian ────────────────────
                elite_idx = np.argpartition(-scores, self.n_elite - 1)[:self.n_elite]
                elite_seq = candidates[elite_idx]                        # (n_elite, H, A)
                self._mu = elite_seq.mean(axis=0)
                self._sigma = np.maximum(elite_seq.std(axis=0), self._sigma_min)
        finally:
            self.bnn.num_weight_groups = saved_groups

        return self._mu[0].copy()

    def _cvar_value(self, returns):
        m = len(returns)
        k = max(1, int(np.ceil(self.cvar_alpha * m)))
        return float(np.sort(returns)[:k].mean())
