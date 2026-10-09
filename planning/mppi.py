"""MPPI planner for discrete FrozenLake (drop-in for CVaRCEMAgent).

Model Predictive Path Integral control (Williams et al., 2017) with the
information-theoretic update specialised to CATEGORICAL controls:

    sample   a^j_{0:H-1} ~ pi            (j = 1..J, pi = per-timestep categorical)
    score    S_j = CVaR_alpha[ Z(a^j) ]  (same K-model x N-rollout evaluator as CEM)
    weight   w_j = exp((S_j - max S) / lambda) / sum_i exp(...)
    update   pi[t] <- (1-smooth) * sum_j w_j onehot(a^j_t) + smooth * uniform

The only difference from CVaR-CEM is the update: CEM refits pi to the top 10%
(hard elite cut), MPPI refits it to ALL samples, soft-weighted by exp(S/lambda).
Everything else -- K Thompson models, horizon, discount, leaf value, CVaR score,
policy-seeded init -- is inherited unchanged, so the planner is the only change.
"""
import numpy as np
import torch

from planning.cvar_cem import CVaRCEMAgent

MPPI_LAMBDA = 0.1        # temperature; scores (CVaR returns) live in [-1, 1]
MPPI_ITERS = 5           # refinement iterations (matched to CEM's 5 x 256)


class MPPIAgent(CVaRCEMAgent):
    def __init__(self, *args, mppi_lambda=MPPI_LAMBDA, mppi_iters=MPPI_ITERS,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mppi_lambda = float(mppi_lambda)
        self.mppi_iters = int(mppi_iters)

    @torch.no_grad()
    def act(self, obs, **kwargs):
        s0 = int(np.argmax(obs))
        if self.is_terminal[s0]:
            return self._eye_a[0].cpu().numpy()
        Ts, _ = self._model_matrices(self.k_models, deterministic=False)
        self.pi = self._policy_init_pi(s0)
        conf = self._confidence() if self.adaptive_alpha else 1.0
        alpha = self._adaptive_alpha(conf)
        self.last_alpha = alpha

        J, H, A = self.n_candidates, self.horizon, self.n_actions
        eye = np.eye(A)
        for _ in range(self.mppi_iters):
            cdf = np.cumsum(self.pi, axis=1)
            u = self.rng.random((J, H))
            A_seq = (u[:, :, None] < cdf[None, :, :]).argmax(axis=2)   # (J, H)
            S = self._cvar(self._rollout_returns(s0, A_seq, Ts), alpha)  # (J,)
            w = np.exp((S - S.max()) / self.mppi_lambda)
            w /= w.sum()
            pi = np.einsum("j,jta->ta", w, eye[A_seq])                    # (H, A)
            self.pi = ((1.0 - self.action_smooth) * pi
                       + self.action_smooth / A)
        a0 = int(np.argmax(self.pi[0]))
        self.last_cvar = float(S[A_seq[:, 0] == a0].mean()) if (A_seq[:, 0] == a0).any() else 0.0
        return self._eye_a[a0].cpu().numpy()
