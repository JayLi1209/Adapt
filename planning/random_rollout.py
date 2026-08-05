"""Random-rollout + argmax MPC planner for discrete FrozenLake.

The risk-neutral, refinement-free counterpart to CVaRCEMAgent, for the ablation
"what if we drop CEM and CVaR but keep the surprise/drift/forget model loop?".

Per act():
  1. read K posterior transition matrices off the BNN (the same epistemic axis
     CVaRCEMAgent spreads over -- forget/inflate widens these);
  2. draw J UNIFORM-RANDOM action sequences of length H (random rollout -- no
     CEM categorical, no elite refit, no policy seeding);
  3. roll each out through K models x N aleatoric samples, summing reward-on-
     arrival (+ a terminal bootstrap for still-running rollouts);
  4. score each candidate by its MEAN return over the K*N rollouts (risk-neutral
     -- NOT CVaR) and execute the FIRST action of the argmax candidate.

Holes still carry reward -1 internally (base default) so the argmax avoids them;
only the *reported* return scores holes as 0.  Inherits the BNN plumbing (reward /
terminal / heuristic arrays, _model_matrices) from BNNModelPlanner.
"""
import numpy as np
import torch

from planning.base import BNNModelPlanner

# ── Random-shooting hyperparameters ───────────────────────────────────────────
H_PLAN = 6              # planning horizon (matches CVaRCEMAgent)
N_CANDIDATES = 1280     # J random action sequences per step (== CEM's 256 x 5
                        # iters, so total rollouts are compute-comparable)
K_MODELS = 10           # K posterior transition matrices (epistemic axis)
N_ROLLOUTS = 32         # N aleatoric rollouts per model per candidate


class RandomRolloutAgent(BNNModelPlanner):
    def __init__(self, dynamics_model, bnn, desc, device, n_actions=4,
                 horizon=H_PLAN, n_candidates=N_CANDIDATES, k_models=K_MODELS,
                 n_rollouts=N_ROLLOUTS, gamma=1.0, rng=None, **kwargs):
        super().__init__(dynamics_model, bnn, desc, device, n_actions=n_actions,
                         gamma=gamma, rng=rng, **kwargs)
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.k_models = k_models
        self.n_rollouts = n_rollouts
        # Reward / terminal / leaf arrays as plain numpy (as in CVaRCEMAgent).
        self.reward_vec = self.cell_reward.astype(np.float64)     # goal +1, hole -1
        self.is_terminal = self.terminal.copy()
        self.terminal_value = self.heuristic.astype(np.float64)
        # Attributes the env loop sets each step; unused here (kept for API parity).
        self.surprise_bar = 1.0
        self.n_since_change = 0
        self.last_alpha = 1.0
        self.last_cvar = 0.0
        self.last_bonus = 0.0

    def reset(self):
        pass

    def notify_change(self):
        self.n_since_change = 0
        self.surprise_bar = 1.0

    def _rollout_returns(self, s0, A_seq, Ts):
        """Vectorised imagined returns for every candidate x model x rollout.

        A_seq : (J, H) int actions.  Ts : (K, S, A, S).  Returns (J, K*N):
            Z = sum_t gamma^t reward_vec[s_{t+1}] + gamma^H V(s_H),
        terminating (bootstrap 0) on the first absorbing cell.  Identical rollout
        to CVaRCEMAgent._rollout_returns -- only the SCORING (mean vs CVaR) differs.
        """
        J, H = A_seq.shape
        K, N = self.k_models, self.n_rollouts
        M = J * K * N
        cand_idx = np.repeat(np.arange(J), K * N)
        model_idx = np.tile(np.repeat(np.arange(K), N), J)
        state = np.full(M, s0, dtype=np.int64)
        done = np.zeros(M, dtype=bool)
        returns = np.zeros(M, dtype=np.float64)
        disc = 1.0
        for t in range(H):
            a = A_seq[cand_idx, t]
            probs = Ts[model_idx, state, a]
            cdf = np.cumsum(probs, axis=1)
            u = self.rng.random((M, 1)) * cdf[:, -1:]
            s2 = (cdf >= u).argmax(axis=1)
            s2 = np.where(done, state, s2)
            returns += disc * np.where(done, 0.0, self.reward_vec[s2])
            done = done | self.is_terminal[s2]
            state = s2
            disc *= self.gamma
        returns += disc * np.where(done, 0.0, self.terminal_value[state])
        return returns.reshape(J, K * N)

    @torch.no_grad()
    def act(self, obs, **kwargs):
        s0 = int(np.argmax(obs))
        if self.is_terminal[s0]:
            return self._eye_a[0].cpu().numpy()

        Ts, _ = self._model_matrices(self.k_models, deterministic=False)
        J, H, A = self.n_candidates, self.horizon, self.n_actions
        # Uniform-random action sequences (the "random rollout").
        A_seq = self.rng.integers(0, A, size=(J, H))
        Z = self._rollout_returns(s0, A_seq, Ts)     # (J, K*N)
        scores = Z.mean(axis=1)                      # risk-NEUTRAL mean return
        a0 = int(A_seq[int(np.argmax(scores)), 0])   # argmax candidate -> first act

        # Diagnostics: mean score per first action (for logging parity).
        qmean = np.full(A, -np.inf)
        for ai in range(A):
            mask = A_seq[:, 0] == ai
            if mask.any():
                qmean[ai] = scores[mask].mean()
        self.last_cvar = float(qmean[a0]) if np.isfinite(qmean[a0]) else 0.0
        return self._eye_a[a0].cpu().numpy()
