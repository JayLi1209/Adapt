"""iLQR planner for discrete FrozenLake, in belief space (drop-in for CVaRCEMAgent).

iLQR needs a smooth state and control, so the discrete MDP is lifted:

  control  u_t in R^A      action LOGITS;  pi_t = softmax(u_t)
  state    x_t = [B_t, G_t]
           B_t (K, S)      alive probability mass over cells under EACH of the
                           K posterior transition models (Thompson draws)
           G_t (K,)        return accumulated so far under each model
  dynamics (per model k, bilinear in (B, pi); aleatoric noise integrated EXACTLY)
           Bn_k   = (B_k * nt) @ sum_a pi_a T_k[:, a, :]
           B'_k   = Bn_k * nt                    (mass reaching goal/hole leaves)
           G'_k   = G_k + gamma^t * Bn_k . R     (R: goal +1, hole -1 on arrival)
  terminal J_k    = G_k + gamma^H * B_k . V      (V: CEM's gamma^dist leaf value)
  objective        maximise softmin_tau(J_1..J_K)   (cvar_alpha < 1)
                   maximise mean(J)                  (cvar_alpha >= 1)
  cost             C = -objective + (eps/2) sum_t ||u_t||^2

softmin_tau is the smooth stand-in for CVaR_alpha at alpha -> 0 (the worst
model), since the empirical CVaR is not differentiable.  Because the transition
noise is carried exactly in B, the dynamics are deterministic: iLQG's
process-noise terms vanish and iLQG reduces to exactly this iLQR.

Algorithm: standard iLQR (Li & Todorov 2004; Tassa et al. 2012) -- analytic
first-order dynamics Jacobians, quadratic terminal cost, backward Riccati pass
with Levenberg-Marquardt regularisation mu, backtracking forward line search.
Initial controls = log of CEM's policy-seeded categorical, so the planner
starts where CEM starts.  Horizon, discount, K models, leaf value: inherited.
"""
import numpy as np
import torch

from planning.cvar_cem import CVaRCEMAgent

ILQR_TAU = 0.05          # soft-min temperature (returns live in [-1, 1])
ILQR_EPS = 1e-3          # control (logit) regulariser; also fixes softmax gauge
ILQR_ITERS = 30
ILQR_MU0, ILQR_MU_MAX = 1e-3, 1e8
LINE_SEARCH = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)


def _softmax(u):
    z = np.exp(u - u.max())
    return z / z.sum()


class ILQRAgent(CVaRCEMAgent):
    def __init__(self, *args, ilqr_tau=ILQR_TAU, ilqr_eps=ILQR_EPS,
                 ilqr_iters=ILQR_ITERS, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = float(ilqr_tau)
        self.eps = float(ilqr_eps)
        self.iters = int(ilqr_iters)
        self.nt = (~self.is_terminal).astype(np.float64)          # (S,)
        self.Vleaf = self.terminal_value * self.nt                # (S,)
        self.last_ilqr_iters = 0

    # ── model ────────────────────────────────────────────────────────────────
    def _step(self, x, u, t, Ts):
        K, S = Ts.shape[0], self.n
        B, G = x[:K * S].reshape(K, S), x[K * S:]
        pi = _softmax(u)
        b = B * self.nt                                            # (K, S)
        M = np.einsum("a,ksat->kst", pi, Ts)                       # (K, S, S)
        Bn = np.einsum("ks,kst->kt", b, M)                         # (K, S)
        g = self.gamma ** t
        G2 = G + g * Bn @ self.reward_vec
        return np.concatenate([(Bn * self.nt).ravel(), G2])

    def _jac(self, x, u, t, Ts):
        """Analytic f_x (D, D) and f_u (D, A)."""
        K, S, A = Ts.shape[0], self.n, self.n_actions
        D = K * S + K
        B = x[:K * S].reshape(K, S)
        pi = _softmax(u)
        b = B * self.nt
        M = np.einsum("a,ksat->kst", pi, Ts)
        g = self.gamma ** t
        R = self.reward_vec
        fx = np.zeros((D, D))
        C = np.einsum("ks,ksat->kat", b, Ts)                       # dBn_k/dpi_a
        dpi_du = np.diag(pi) - np.outer(pi, pi)                    # (A, A)
        fpi = np.zeros((D, A))
        for k in range(K):
            blk = slice(k * S, (k + 1) * S)
            # B'_k[i] = nt[i] * sum_j nt[j] B_k[j] M_k[j, i]
            fx[blk, blk] = self.nt[:, None] * M[k].T * self.nt[None, :]
            # G'_k = G_k + g * sum_j nt[j] B_k[j] (M_k R)[j]
            fx[K * S + k, blk] = g * self.nt * (M[k] @ R)
            fx[K * S + k, K * S + k] = 1.0
            fpi[blk, :] = (C[k] * self.nt[None, :]).T              # (S, A)
            fpi[K * S + k, :] = g * C[k] @ R
        return fx, fpi @ dpi_du

    def _J_of_x(self, K):
        """Linear map x -> J (K,), as a (K, D) matrix."""
        S = self.n
        Amat = np.zeros((K, K * S + K))
        for k in range(K):
            Amat[k, k * S:(k + 1) * S] = self.gamma ** self.horizon * self.Vleaf
            Amat[k, K * S + k] = 1.0
        return Amat

    def _objective(self, J):
        if self.last_alpha >= 1.0:
            return float(J.mean())
        z = -J / self.tau
        m = z.max()
        return float(-self.tau * (m + np.log(np.mean(np.exp(z - m)))))

    def _terminal_derivs(self, xH, Amat):
        """phi = -objective:  phi_x, phi_xx."""
        J = Amat @ xH
        K = len(J)
        if self.last_alpha >= 1.0:
            return -Amat.T @ np.full(K, 1.0 / K), np.zeros((Amat.shape[1],) * 2)
        p = _softmax(-J / self.tau)
        hess_obj = -(np.diag(p) - np.outer(p, p)) / self.tau      # d2 obj / dJ2
        return -Amat.T @ p, -Amat.T @ hess_obj @ Amat

    def _rollout(self, x0, U, Ts, Amat):
        xs = [x0]
        for t in range(self.horizon):
            xs.append(self._step(xs[-1], U[t], t, Ts))
        cost = (-self._objective(Amat @ xs[-1])
                + 0.5 * self.eps * float((U ** 2).sum()))
        return np.array(xs), cost

    # ── iLQR ─────────────────────────────────────────────────────────────────
    def _ilqr(self, x0, U, Ts):
        H, A = self.horizon, self.n_actions
        Amat = self._J_of_x(Ts.shape[0])
        X, cost = self._rollout(x0, U, Ts, Amat)
        mu = ILQR_MU0
        it = 0
        for it in range(self.iters):
            fxs, fus = zip(*[self._jac(X[t], U[t], t, Ts) for t in range(H)])
            # backward pass
            while True:
                Vx, Vxx = self._terminal_derivs(X[-1], Amat)
                ks, Ks, ok = [None] * H, [None] * H, True
                for t in reversed(range(H)):
                    fx, fu = fxs[t], fus[t]
                    Qx = fx.T @ Vx
                    Qu = self.eps * U[t] + fu.T @ Vx
                    Qxx = fx.T @ Vxx @ fx
                    Quu = self.eps * np.eye(A) + fu.T @ Vxx @ fu
                    Qux = fu.T @ Vxx @ fx
                    Quu_r = Quu + mu * np.eye(A)
                    try:
                        L = np.linalg.cholesky(Quu_r)
                    except np.linalg.LinAlgError:
                        ok = False
                        break
                    inv = np.linalg.inv(L.T) @ np.linalg.inv(L)
                    k, Kt = -inv @ Qu, -inv @ Qux
                    ks[t], Ks[t] = k, Kt
                    Vx = Qx + Kt.T @ Quu @ k + Kt.T @ Qu + Qux.T @ k
                    Vxx = Qxx + Kt.T @ Quu @ Kt + Kt.T @ Qux + Qux.T @ Kt
                    Vxx = 0.5 * (Vxx + Vxx.T)
                if ok:
                    break
                mu *= 10.0
                if mu > ILQR_MU_MAX:
                    return U, it
            # forward line search
            accepted = False
            for a_ls in LINE_SEARCH:
                xn, Un = x0, np.empty_like(U)
                for t in range(H):
                    Un[t] = U[t] + a_ls * ks[t] + Ks[t] @ (xn - X[t])
                    xn = self._step(xn, Un[t], t, Ts)
                Xn, cn = self._rollout(x0, Un, Ts, Amat)
                if cn < cost - 1e-10:
                    accepted = True
                    break
            if accepted:
                done = (cost - cn) < 1e-7 * max(1.0, abs(cost))
                X, U, cost = Xn, Un, cn
                mu = max(1e-6, mu / 10.0)
                if done:
                    break
            else:
                mu *= 10.0
                if mu > ILQR_MU_MAX:
                    break
        return U, it + 1

    @torch.no_grad()
    def act(self, obs, **kwargs):
        s0 = int(np.argmax(obs))
        if self.is_terminal[s0]:
            return self._eye_a[0].cpu().numpy()
        Ts, _ = self._model_matrices(self.k_models, deterministic=False)
        conf = self._confidence() if self.adaptive_alpha else 1.0
        self.last_alpha = self._adaptive_alpha(conf)
        K, S = Ts.shape[0], self.n
        x0 = np.zeros(K * S + K)
        x0[[k * S + s0 for k in range(K)]] = 1.0
        U0 = np.log(self._policy_init_pi(s0))                      # CEM's seed
        U, self.last_ilqr_iters = self._ilqr(x0, U0, Ts)
        a0 = int(np.argmax(U[0]))
        return self._eye_a[a0].cpu().numpy()
