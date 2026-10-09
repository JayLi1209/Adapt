r"""Closed-form identification of an ACTION-INTERFACE fault.

WHY NOT AN OUTPUT-SIDE HEAD.  The transmission fault is a fixed orthogonal map
on the INPUT of the dynamics: the plant runs f(s, R a) while the frozen model
believes f(s, a).  An output-side adapter (SFIR's head) must therefore represent
the INDUCED residual

    r_t = s_{t+1} - f(s_t, a_t)  ~=  B(s_t) (R - I) a_t,
    B(s) = df/da  ~  dt * M(q)^{-1}

which is STATE-DEPENDENT through the mass matrix.  A frozen-body linear head can
only fit the workspace-average of that correction, and because M(q)^{-1} swings
with configuration, the average has the WRONG SIGN over part of the state space.
That is a bias-inducing edit to an otherwise honest model -- the precise shape of
"adaptation is worse than not adapting", which is what every output-side arm on
this env has measured.

Putting the correction where the fault actually lives removes the state
dependence entirely: A is four constants, not a function of s.

IDENTIFICATION.  Linearising the frozen model in the action around the planned
a_t and stacking transitions,

    r_t ~= B_t (A - I) a_t = (a_t^T kron B_t) vec(A - I)

so vec(A - I) is the solution of an ordinary weighted least-squares problem in
FOUR unknowns.  B_t = d mu_body / da at (s_t, a_t) comes from autodiff through
the frozen BNN mean.  Restricting rows to the qvel dims is not a heuristic here:
torque enters qpos only at O(dt^2), so those are the rows where B_t is
numerically meaningful and the rest contribute mostly noise.

Weighting is by the body's own aleatoric precision on those rows, so the fit
inherits the model's calibrated notion of which dims it trusts.

PROJECTION.  The fault is known to be orthogonal, so the raw LS solution is
projected onto SO(2) by orthogonal Procrustes:

    A = U S V^T   ->   R_hat = U diag(1, det(U V^T)) V^T

The det guard is not cosmetic.  Without it the SVD can return a REFLECTION
(det = -1), which is not a rotation and would emit a mirrored command that the
plant never applies -- the estimate would sit in the wrong component of O(2) and
never recover.

USE.  Plan with the UNTOUCHED frozen model to get a*, then emit R_hat^T a*.  The
plant applies R, so it receives R R_hat^T a* -> a* when R_hat -> R.  Because
R_hat^T is orthogonal, ||R_hat^T a*|| == ||a*||, so the control cost -||a||^2 is
unchanged and the learned/analytic reward identity survives untouched.  The
planner needs no modification at all.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["ActionInterfaceAdapter"]


class ActionInterfaceAdapter:
    """Recursive weighted LS for a 2x2 action map, projected to SO(2)."""

    def __init__(self, act_dim, obs_dim, rows, ridge=1e-4, decay=1.0,
                 min_samples=10):
        self.act_dim = int(act_dim)
        self.obs_dim = int(obs_dim)
        self.rows = list(rows)
        self.ridge = float(ridge)
        self.decay = float(decay)
        self.min_samples = int(min_samples)
        p = self.act_dim * self.act_dim
        # Normal equations for vec(A - I): G x = c
        self.G = np.zeros((p, p), dtype=np.float64)
        self.c = np.zeros(p, dtype=np.float64)
        self.n = 0
        self.R_hat = np.eye(self.act_dim, dtype=np.float64)
        self._raw = np.eye(self.act_dim, dtype=np.float64)

    # ── B_t = d mu_body / da, by autodiff through the frozen mean ────────────
    @staticmethod
    def jacobian(bnn, dyn, obs, act, rows):
        """(len(rows), act_dim) Jacobian of the frozen BNN mean wrt the action."""
        a = act.detach().clone().requires_grad_(True)
        mean, _ = bnn._run_network(dyn._get_model_input(obs, a), sample=False)
        cols = []
        for d in rows:
            g, = torch.autograd.grad(mean[0, d], a, retain_graph=True)
            cols.append(g.reshape(-1))
        return torch.stack(cols, dim=0).detach()

    # ── iterated refit: re-linearise about the ESTIMATED applied torque ──────
    #
    # Expanding f about the PLANNED a leaves a first-order error proportional to
    # (A - I) a, which grows with the size of the fault -- measured on this env
    # the naive one-shot fit over-estimates by 2.2 deg at 30 deg, 6.2 at 60 and
    # 10.6 at 90.  The frozen model is very nearly linear in a (<=1.8% deviation
    # out to |a| = 1), so the culprit is the EXPANSION POINT, not curvature.
    # Re-linearising about A_hat a and refitting converges in ~2 sweeps and cuts
    # the 90 deg error from 10.57 to 0.24 deg.
    def refit(self, bnn, dyn, buf, n_iter=3):
        """Batch re-solve over a buffer of (obs, emitted_act, delta) tensors.

        CLOSED-LOOP CORRECTNESS.  The regressor must be the action the PLANT
        received, not the one the planner proposed.  Once compensation is live
        the two differ by R_hat^T, and that map CHANGES as the estimate updates,
        so a buffer keyed on planned actions mixes transitions generated under
        several different effective maps and the fit drifts -- measured here:
        62.3 deg at t=25 decaying to 51.0 deg at t=50 against a true 60.  Keyed
        on emitted actions, every sample satisfies the same relation
        delta = f(s, R a_emitted), and R is identifiable from all of them
        regardless of what compensation was active when each was collected.
        """
        if len(buf) < self.min_samples:
            return self.R_hat
        p = self.act_dim * self.act_dim
        R = self.R_hat.copy()
        for _ in range(int(n_iter)):
            G = np.zeros((p, p)); c = np.zeros(p)
            for o_t, a_t, d_t in buf:
                a = a_t.detach().cpu().numpy().reshape(-1)
                a_lin = torch.as_tensor((R @ a).astype(np.float32),
                                        device=a_t.device).unsqueeze(0)
                B = self.jacobian(bnn, dyn, o_t, a_lin, self.rows).cpu().numpy()
                with torch.no_grad():
                    mean, lv = bnn._run_network(
                        dyn._get_model_input(o_t, a_lin), sample=False)
                # Residual referred back to the linearisation point: the model's
                # own prediction at A a already contains B(A a), so adding it
                # back makes the target consistent with M vec(A).
                r = (d_t.detach().cpu().numpy()[self.rows]
                     - mean[0, self.rows].cpu().numpy() + B @ (R @ a))
                w = np.exp(-lv[0, self.rows].detach().cpu().numpy())
                M = np.kron(a.reshape(1, -1), B)
                G += M.T @ (w[:, None] * M); c += M.T @ (w * r)
            try:
                x = np.linalg.solve(G + self.ridge * np.eye(p), c)
            except np.linalg.LinAlgError:
                return self.R_hat
            if not np.all(np.isfinite(x)):
                return self.R_hat
            A = x.reshape(self.act_dim, self.act_dim, order='F')
            self._raw = A.copy()
            U, _S, Vt = np.linalg.svd(A)
            d = np.sign(np.linalg.det(U @ Vt)) or 1.0
            R = U @ np.diag([1.0] * (self.act_dim - 1) + [d]) @ Vt
        self.R_hat = R
        self.n = len(buf)
        return self.R_hat

    def update(self, bnn, dyn, obs_t, act_t, delta_t, logvar_t=None):
        """Accumulate one transition.

        obs_t, act_t: (1, ·) tensors -- act_t is the PLANNED action, the one the
        frozen model was queried with, not what was emitted to the plant.
        delta_t: (obs_dim,) observed s_{t+1} - s_t.
        """
        B = self.jacobian(bnn, dyn, obs_t, act_t, self.rows).cpu().numpy()
        with torch.no_grad():
            mean, lv = bnn._run_network(dyn._get_model_input(obs_t, act_t),
                                        sample=False)
        r = (delta_t.detach().cpu().numpy()[self.rows]
             - mean[0, self.rows].cpu().numpy())
        a = act_t.detach().cpu().numpy().reshape(-1)
        if logvar_t is None:
            logvar_t = lv
        w = np.exp(-logvar_t[0, self.rows].detach().cpu().numpy())  # precision
        # M = a^T kron B  ->  (len(rows), act_dim^2)
        M = np.kron(a.reshape(1, -1), B)
        if self.decay < 1.0:
            self.G *= self.decay; self.c *= self.decay; self.n *= self.decay
        self.G += M.T @ (w[:, None] * M)
        self.c += M.T @ (w * r)
        self.n += 1
        return self

    def solve(self):
        """Weighted LS then Procrustes projection onto SO(2)."""
        if self.n < self.min_samples:
            return self.R_hat
        p = self.G.shape[0]
        try:
            x = np.linalg.solve(self.G + self.ridge * np.eye(p), self.c)
        except np.linalg.LinAlgError:
            return self.R_hat
        if not np.all(np.isfinite(x)):
            return self.R_hat
        A = np.eye(self.act_dim) + x.reshape(self.act_dim, self.act_dim, order='F')
        self._raw = A.copy()
        U, _S, Vt = np.linalg.svd(A)
        d = np.sign(np.linalg.det(U @ Vt))
        if d == 0.0:
            d = 1.0
        # Reflection guard: force det = +1 so the estimate stays in SO(2).
        D = np.diag([1.0] * (self.act_dim - 1) + [d])
        self.R_hat = U @ D @ Vt
        return self.R_hat

    def emit(self, a_star):
        """Map a planned action to the command that cancels the fault."""
        return (self.R_hat.T @ np.asarray(a_star, np.float64).reshape(-1)
                ).astype(np.float32)

    @property
    def angle_deg(self):
        return float(np.degrees(np.arctan2(self.R_hat[1, 0], self.R_hat[0, 0])))
