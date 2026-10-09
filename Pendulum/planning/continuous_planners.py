"""MPPI / MCTS / iLQR for continuous control (drop-ins for ContinuousCEMAgent).

All three keep CEM's OBJECTIVE and swap only the OPTIMIZER, so a planner
comparison isolates the search procedure:

    objective  CVaR_alpha over K posterior draws of the discounted H-step
               imagined return of an open-loop action sequence
               (ContinuousCEMAgent._rollout_returns + _scores: per-step weight
               resampling, aleatoric noise, obs projection, reward_fn / learned
               reward head, reward clip -- all inherited unchanged).
    shared     horizon, gamma, K, cvar_alpha, action bounds, MPC warm start
               (previous plan shifted one step).

MPPI   (Williams et al., 2017).  Same I x J sampling budget as CEM; the update
       refits the mean to ALL samples soft-weighted by exp(S / lambda_eff)
       instead of the top-10% elites.  lambda_eff = lambda * std(S): under a
       Gaussian score spread the exponentially tilted mean sits 1/lambda std
       above the pool mean, so lambda = 0.57 reproduces the selection pressure
       of CEM's 10% elite cut (whose mean is 1.755 std up) and the ONLY
       difference left is hard-vs-soft weighting.  Sampling sigma is FIXED (no
       covariance adaptation), as in standard MPPI; no control-cost term, so the
       objective is exactly CEM's.

MCTS   Open-loop UCT with progressive widening (Couetoux et al., 2011; open-loop
       search, Perez et al., 2015).  A node is an action PREFIX; its state is
       re-simulated from the root every time, which is the right semantics for
       a stochastic model with K draws.  Each simulation selects/expands one
       prefix, completes it with the default policy (the warm-start plan + small
       Gaussian noise), and is scored by the shared CVaR evaluator; the score is
       backed up as a running mean.  Simulations run in WAVES of B with virtual
       visits (counts only) so each wave is one batched GPU rollout:
       n_waves * B = CEM's I * J sequences per decision.  Commit the most
       visited root child; warm-start from the best sequence through it.

iLQR   Box-constrained iLQR on the STACKED state of K fixed scenarios.  Per
       decision, draw K weight samples per imagined step and K aleatoric noise
       vectors per step (exactly the randomness one CEM rollout uses), which
       makes each scenario k a deterministic, differentiable model f_k,t.
       Optimise one shared control sequence against
           max  CVaR_alpha( J_1 .. J_K ),  J_k = sum_t gamma^t r_k,t
       via the active-set form of CVaR: with the tail set T (the worst
       ceil(alpha K) scenarios of the current nominal) held fixed, the
       objective is a weighted sum (1/|T| on T), i.e. an ordinary iLQR cost; T
       is re-identified after every accepted iteration and the line search
       accepts on the TRUE CVaR.  Derivatives (dynamics Jacobian, reward
       gradient and Hessian) come from autograd through the network; the
       per-scenario 4x4 cost Hessian is projected to PSD.  Torque bounds are
       handled by the scalar box-QP (clamp the feedforward, drop the feedback
       gain when clamped; Tassa et al., 2014).  Multi-start (warm start, zero,
       +/-1, +/-2, random) mitigates the swing-up local minimum; all restarts
       and all line-search step sizes are batched into one GPU rollout.

Pendulum copy of ../continous_planners.py.  Differences from that file:
  * ContinuousCEMAgent here exposes _rollout_returns / _scores / sim_budget
    (the CEM loop refactored, numerics unchanged), which MPPI and MCTS need.
  * every planner records self.sim_steps (imagined transitions in the last
    act()); all are held to CEM's sim_budget() = I * J * K * H.
  * iLQR iterates until the budget is spent (re-seeding converged restarts)
    instead of a fixed 10 iterations, and composes the attached adapter head
    (body(s, 0) + h(s, u)) into its differentiable model -- see the class.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from config import device
from planning.continuous_cem import ContinuousCEMAgent

# ── MPPI ──────────────────────────────────────────────────────────────────────
MPPI_LAMBDA = 0.57       # temperature in units of the score std (see docstring)
MPPI_SIGMA = 1.0         # fixed sampling std (CEM's sigma_init)


class ContinuousMPPIAgent(ContinuousCEMAgent):
    """MPPI over continuous action sequences, scored by CEM's CVaR evaluator."""

    def __init__(self, *args, mppi_lambda=MPPI_LAMBDA, mppi_sigma=MPPI_SIGMA,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mppi_lambda = float(mppi_lambda)
        self.mppi_sigma = float(mppi_sigma)
        self.last_ess = float("nan")

    @torch.no_grad()
    def act(self, obs):
        obs_arr = np.asarray(obs, dtype=np.float32).ravel()
        H, A, J, K = self.horizon, self.act_dim, self.n_candidates, self.k_models
        saved_groups = self.bnn.num_weight_groups
        self.bnn.num_weight_groups = K if K > 1 else 1
        self.sim_steps = 0
        self._mu = np.roll(self._mu, -1, axis=0)        # MPC warm start
        self._mu[-1] = 0.0
        try:
            for _it in range(self.n_cem_iters):
                noise = self.rng.normal(size=(J, H, A)).astype(np.float32)
                cand = np.clip(self._mu + self.mppi_sigma * noise,
                               self.action_low, self.action_high)
                S = self._scores(self._rollout_returns(obs_arr, cand))
                ok = np.isfinite(S)
                if not ok.any():
                    continue
                s = np.where(ok, S, -np.inf)
                scale = self.mppi_lambda * max(float(S[ok].std()), 1e-8)
                w = np.exp((s - s[ok].max()) / scale)
                w /= w.sum()
                self._mu = np.einsum("j,jha->ha", w, cand).astype(np.float32)
                self.last_plan_return = float(np.sum(w[ok] * S[ok]))
                self.last_plan_return_best = float(S[ok].max())
                self.last_ess = float(1.0 / np.sum(w ** 2))
        finally:
            self.bnn.num_weight_groups = saved_groups
        return self._mu[0].copy()


# ── MCTS ──────────────────────────────────────────────────────────────────────
MCTS_WAVES = 16          # sequential batches of simulations per decision
MCTS_C_UCT = 1.0         # UCB constant on min-max normalised Q in [0, 1]
MCTS_PW_C, MCTS_PW_ALPHA = 1.0, 0.5   # progressive widening: |children| <= c N^alpha
MCTS_TAIL_SIGMA = 0.3    # default policy: warm-start plan + N(0, sigma^2)


class _Node:
    __slots__ = ("depth", "actions", "children", "n", "nv", "w")

    def __init__(self, depth):
        self.depth = depth
        self.actions = []        # action (A,) leading to each child
        self.children = []
        self.n = 0               # real visits (scored simulations)
        self.nv = 0              # real + in-flight virtual visits
        self.w = 0.0             # sum of scores of real visits


class ContinuousMCTSAgent(ContinuousCEMAgent):
    """Open-loop progressive-widening UCT, scored by CEM's CVaR evaluator.

    Budget: n_waves * sims_per_wave sequences per decision; the runner sets
    sims_per_wave = I * J / n_waves so the imagined-transition count equals CEM's.
    """

    def __init__(self, *args, mcts_waves=MCTS_WAVES, mcts_c_uct=MCTS_C_UCT,
                 mcts_pw_c=MCTS_PW_C, mcts_pw_alpha=MCTS_PW_ALPHA,
                 mcts_tail_sigma=MCTS_TAIL_SIGMA, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_waves = int(mcts_waves)
        self.sims_per_wave = max(1, (self.n_cem_iters * self.n_candidates) // self.n_waves)
        self.c_uct = float(mcts_c_uct)
        self.pw_c, self.pw_alpha = float(mcts_pw_c), float(mcts_pw_alpha)
        self.tail_sigma = float(mcts_tail_sigma)
        self.last_tree = {}

    def _select(self, root, nominal, q_lo, q_hi):
        """One selection/expansion pass; returns (path, prefix actions)."""
        node, path, prefix = root, [root], []
        span = q_hi - q_lo
        while node.depth < self.horizon:
            limit = max(1, math.ceil(self.pw_c * (node.nv + 1) ** self.pw_alpha))
            if len(node.children) < limit:
                a = (nominal[node.depth].copy() if not node.children else
                     self.rng.uniform(self.action_low, self.action_high,
                                      self.act_dim).astype(np.float32))
                child = _Node(node.depth + 1)
                node.actions.append(a)
                node.children.append(child)
                path.append(child)
                prefix.append(a)
                break
            # parent's mean as first-play value for children with no real visit
            fpu = node.w / node.n if node.n else 0.5 * (q_lo + q_hi)
            log_n = math.log(max(node.nv, 1))
            best, best_i = -math.inf, 0
            for i, c in enumerate(node.children):
                q = c.w / c.n if c.n else fpu
                qn = (q - q_lo) / span if span > 0 else 0.5
                u = qn + self.c_uct * math.sqrt(log_n / max(c.nv, 1))
                if u > best:
                    best, best_i = u, i
            prefix.append(node.actions[best_i])
            node = node.children[best_i]
            path.append(node)
        return path, prefix

    @torch.no_grad()
    def act(self, obs):
        obs_arr = np.asarray(obs, dtype=np.float32).ravel()
        H, A, K = self.horizon, self.act_dim, self.k_models
        B = self.sims_per_wave
        self.sim_steps = 0
        saved_groups = self.bnn.num_weight_groups
        self.bnn.num_weight_groups = K if K > 1 else 1
        nominal = np.roll(self._mu, -1, axis=0)          # MPC warm start
        nominal[-1] = 0.0

        root = _Node(0)
        q_lo, q_hi = math.inf, -math.inf
        best_by_child = {}                                # root child idx -> (score, seq)
        max_depth = 0
        try:
            for _w in range(self.n_waves):
                lo, hi = (q_lo, q_hi) if q_hi > q_lo else (0.0, 0.0)
                paths, seqs = [], np.empty((B, H, A), dtype=np.float32)
                for b in range(B):
                    path, prefix = self._select(root, nominal, lo, hi)
                    for nd in path:
                        nd.nv += 1                        # virtual visit
                    d = len(prefix)
                    max_depth = max(max_depth, d)
                    tail = nominal[d:] + self.tail_sigma * self.rng.normal(
                        size=(H - d, A)).astype(np.float32)
                    seqs[b, :d] = np.asarray(prefix, dtype=np.float32).reshape(d, A)
                    seqs[b, d:] = tail
                    paths.append(path)
                seqs = np.clip(seqs, self.action_low, self.action_high)
                S = self._scores(self._rollout_returns(obs_arr, seqs))
                for b, path in enumerate(paths):
                    s = float(S[b]) if np.isfinite(S[b]) else (q_lo if np.isfinite(q_lo) else -1e6)
                    for nd in path:
                        nd.n += 1
                        nd.w += s
                    q_lo, q_hi = min(q_lo, s), max(q_hi, s)
                    ci = root.children.index(path[1])
                    if ci not in best_by_child or s > best_by_child[ci][0]:
                        best_by_child[ci] = (s, seqs[b].copy())
        finally:
            self.bnn.num_weight_groups = saved_groups

        ci = int(np.argmax([c.n for c in root.children]))   # robust child
        child = root.children[ci]
        self._mu = best_by_child[ci][1]
        self.last_plan_return = child.w / max(child.n, 1)
        self.last_plan_return_best = float(max(v[0] for v in best_by_child.values()))
        self.last_tree = dict(root_children=len(root.children), max_depth=max_depth,
                              commit_visits=child.n)
        return root.actions[ci].copy()


# ── iLQR ──────────────────────────────────────────────────────────────────────
ILQR_RESTARTS = 32      # restarts batched per iteration (R sweep: 8..64 reach the same CVaR; 32 is fastest within budget)
ILQR_MU0, ILQR_MU_MIN, ILQR_MU_MAX = 1e-2, 1e-6, 1e8
ILQR_LINE_SEARCH = (1.0, 0.5, 0.25, 0.1, 0.03, 0.01)
ILQR_TOL = 1e-6          # relative CVaR improvement below which a restart is done
ILQR_REINIT_SIGMA = 0.5  # re-seeded restart: best plan so far + N(0, sigma^2)


class ContinuousILQRAgent(ContinuousCEMAgent):
    """Multi-start box-iLQR on K stacked fixed scenarios with an exact CVaR
    active-set objective (see module docstring).

    BUDGET.  Iterates until the next iteration would exceed sim_budget() (= CEM's
    I*J*K*H imagined transitions).  Accounting, per restart row: an open-loop
    rollout costs K*H; a linearisation costs K*H (one model query per nominal
    (t, k) point, Jacobian/Hessian by autograd at that point); a line search
    costs L*K*H.  A restart that converges (or whose regulariser blows up) is
    RE-SEEDED -- alternately best-so-far + noise and uniform random -- so the
    budget is spent searching rather than re-confirming a converged optimum.
    The committed plan is the best CVaR seen over all restarts.

    ADAPTER HEAD.  When `adapter_head` is set (a NonlinearAdapterHead attached to
    dyn), the fixed-scenario model is the SAME composition the head patches into
    dyn.sample:  s' = s + body(s, u=0) + h(s, u).  Without this, iLQR would plan
    with the raw body at the real u, i.e. ignore the adaptation entirely.  Head
    weights are drawn once per imagined step and shared across the K scenarios,
    exactly as the patched dyn.sample draws them in a CEM rollout.
    """

    def __init__(self, *args, ilqr_restarts=ILQR_RESTARTS, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_restarts = int(ilqr_restarts)
        self.adapter_head = None
        self.last_ilqr = {}

    # ── fixed-scenario network ───────────────────────────────────────────────
    def _layers(self):
        b = self.bnn
        return list(b.bayes_layers[:-1]), list(b.adapt_layers), b.bayes_layers[-1]

    def _draw_scenarios(self):
        """Per imagined step t and scenario k: one weight sample of every layer
        and one aleatoric noise vector -- the randomness of one CEM rollout.
        The SAMPLED weights are materialised once here (parameters are fixed
        for the whole act()), so the many model calls below only slice them."""
        H, K = self.horizon, self.k_models
        self._W = []                                      # per layer: (W, b) samples
        hid, ada, out = self._layers()
        for L in hid + ada + [out]:
            o, i = L.weight_mu.shape
            ew = torch.randn(H, K, o, i, device=self.device)
            eb = torch.randn(H, K, o, device=self.device)
            self._W.append((L.weight_mu + L.weight_sigma * ew,
                            L.bias_mu + L.bias_sigma * eb))
        n_out = out.weight_mu.shape[0] // 2
        self._xi = torch.randn(H, K, n_out, device=self.device)
        self._head_W = []
        if self.adapter_head is not None:
            for L in self.adapter_head.layers:            # one draw per t, shared over k
                o, i = L.weight_mu.shape
                ew = torch.randn(H, o, i, device=self.device)
                eb = torch.randn(H, o, device=self.device)
                self._head_W.append(
                    ((L.weight_mu + L.weight_sigma * ew).repeat_interleave(K, 0).view(H, K, o, i),
                     (L.bias_mu + L.bias_sigma * eb).repeat_interleave(K, 0).view(H, K, o)))

    @staticmethod
    def _tslice(x, ts):
        """Rows of the (H, K, ...) sample tensor x for imagined steps ts, flattened
        to (len(ts) * K, ...).  ts is either a single step (t,) or range-like."""
        if len(ts) == 1:
            return x[ts[0]]
        if ts[-1] - ts[0] + 1 == len(ts):
            return x[ts[0]:ts[-1] + 1].flatten(0, 1)
        return x[list(ts)].flatten(0, 1)

    @staticmethod
    def _glinear(x, W, b):
        """Grouped linear: W (M, out, in), b (M, out); row i uses group i % M."""
        M, o, i = W.shape
        xr = x.view(-1, M, i).permute(1, 0, 2)
        y = torch.bmm(xr, W.transpose(1, 2)) + b.unsqueeze(1)
        return y.permute(1, 0, 2).reshape(-1, o)

    def _net(self, model_in, ts):
        """bnn._run_network with the stored per-(t, k) weight samples.
        ts: tuple of imagined steps covered by the batch (rows ordered
        [..., t, k] innermost, so group index = t_local * K + k)."""
        net = self.bnn
        hid, ada, out = self._layers()
        h = model_in
        for li in range(len(hid) + len(ada) + 1):
            W, bb = self._W[li]
            h = self._glinear(h, self._tslice(W, ts), self._tslice(bb, ts))
            if li < len(hid):
                h = F.silu(h)
        mean, raw_logvar = h.chunk(2, dim=-1)
        bounded = net.max_logvar - F.softplus(net.max_logvar - raw_logvar)
        logvar = net.min_logvar + F.softplus(bounded - net.min_logvar)
        mean = mean * net.out_std + net.out_mu
        logvar = logvar + 2.0 * torch.log(net.out_std)
        if net.aleatoric_in_rollout:
            xi = self._tslice(self._xi, ts)              # (T*K, n_out)
            mean = mean + torch.exp(0.5 * logvar) * xi.repeat(mean.shape[0] // xi.shape[0], 1)
        return mean

    def _head(self, x, ts):
        """adapter_head.forward with the stored per-t weight samples (same row
        ordering as _net; the draw for step t is repeated over its K rows)."""
        head, K = self.adapter_head, self.k_models
        sample = K > 1                                    # CEM: sample=not deterministic

        def lin(li, L, z):
            if not sample:
                return F.linear(z, L.weight_mu, L.bias_mu)
            W, bb = self._head_W[li]
            return self._glinear(z, self._tslice(W, ts), self._tslice(bb, ts))

        h = lin(0, head.skip, x)
        if head.hid > 0:
            h = h + lin(2, head.l2, F.silu(lin(1, head.l1, x)))
        return h

    def _step(self, s, u, ts):
        """(s_t, u_t) -> (s_{t+1}, r_t) under the fixed scenarios."""
        if self.adapter_head is None:
            pred = self._net(self.dyn._get_model_input(s, u), ts)
            s2 = s + pred[:, :self.obs_dim]
        else:                                             # body at u=0 + head(s, u)
            pred = self._net(self.dyn._get_model_input(s, torch.zeros_like(u)), ts)
            s2 = s + pred[:, :self.obs_dim]
            s2 = s2 + self._head(torch.cat([s, u], dim=-1), ts)
        if self.obs_project is not None:
            s2 = self.obs_project(s2)
        r = self.reward_fn(s, u) if self.reward_fn is not None else pred[:, self.obs_dim]
        if self.reward_clip is not None:
            r = r.clamp(self.reward_clip[0], self.reward_clip[1])
        return s2, r

    def _rollout_open(self, s0, U):
        """Open-loop rollout of (N, H, A) controls, every row through all K
        scenarios.  Returns states (N, H+1, K, D) and rewards (N, H, K)."""
        N, H, K, D = U.shape[0], self.horizon, self.k_models, self.obs_dim
        s = s0.view(1, 1, D).expand(N, K, D).reshape(N * K, D)
        Ut = torch.as_tensor(U, dtype=torch.float32, device=self.device)
        X, R = [s.reshape(N, K, D)], []
        for t in range(H):
            u = Ut[:, t].repeat_interleave(K, dim=0)
            s, r = self._step(s, u, (t,))
            X.append(s.reshape(N, K, D))
            R.append(r.view(N, K))
        self.sim_steps += N * K * H
        return torch.stack(X, 1), torch.stack(R, 1)

    def _returns(self, Rw):
        disc = self.gamma ** torch.arange(self.horizon, device=self.device,
                                          dtype=torch.float32)
        return (Rw * disc.view(1, -1, 1)).sum(1)          # (N, K)

    def _derivs(self, X, U):
        """Jacobians and reward Hessians at every (n, t, k) nominal point.
        X (N, H+1, K, D), U (N, H, A) -> numpy float64:
          fs (N,H,K,D,D), fu (N,H,K,D,A), g (N,H,K,D+A), Hm (N,H,K,D+A,D+A)."""
        N, H, K, D, A = X.shape[0], self.horizon, self.k_models, self.obs_dim, self.act_dim
        s = X[:, :H].reshape(-1, D)                                   # rows (n, t, k)
        u = torch.as_tensor(U, dtype=torch.float32, device=self.device)
        u = u.unsqueeze(2).expand(N, H, K, A).reshape(-1, A)
        inp = torch.cat([s, u], 1).detach().requires_grad_(True)
        with torch.enable_grad():
            s2, r = self._step(inp[:, :D], inp[:, D:], tuple(range(H)))
            jac = [torch.autograd.grad(s2[:, j].sum(), inp, retain_graph=True)[0]
                   for j in range(D)]
            g = torch.autograd.grad(r.sum(), inp, create_graph=True)[0]
            hess = [torch.autograd.grad(g[:, j].sum(), inp, retain_graph=True,
                                        allow_unused=True)[0] for j in range(D + A)]
        self.sim_steps += N * H * K
        jac = torch.stack(jac, 1)                                     # (rows, D, D+A)
        hess = torch.stack([h if h is not None else torch.zeros_like(inp)
                            for h in hess], 1)                        # (rows, D+A, D+A)
        sh = (N, H, K)
        jac = jac.detach().double().cpu().numpy().reshape(*sh, D, D + A)
        return (jac[..., :D], jac[..., D:],
                g.detach().double().cpu().numpy().reshape(*sh, D + A),
                hess.detach().double().cpu().numpy().reshape(*sh, D + A, D + A))

    def _tail_weights(self, J):
        """CVaR active set: 1/m on the m = ceil(alpha K) worst scenarios."""
        N, K = J.shape
        m = max(1, int(np.ceil(self.cvar_alpha * K)))
        w = np.zeros((N, K))
        idx = np.argsort(J, axis=1)[:, :m]
        np.put_along_axis(w, idx, 1.0 / m, axis=1)
        return w

    def _cvar_rows(self, J):
        m = max(1, int(np.ceil(self.cvar_alpha * J.shape[1])))
        return np.sort(J, axis=1)[:, :m].mean(1)

    def _backward(self, fs, fu, g, Hm, w, U, mu):
        """Box-iLQR backward pass for all N restarts at once (A == 1).
        Cost per scenario c_k,t = -w_k gamma^t r_k,t; Hessian projected to PSD.
        Returns k (N,H), Kfb (N,H,K*D) and ok (N,) per-restart success."""
        N, H, K, D = fs.shape[0], self.horizon, self.k_models, self.obs_dim
        KD = K * D
        lo, hi = self.action_low, self.action_high
        k_ff = np.zeros((N, H))
        K_fb = np.zeros((N, H, KD))
        ok = np.ones(N, dtype=bool)
        Vx = np.zeros((N, KD))
        Vxx = np.zeros((N, KD, KD))
        # PSD-project every (n, t, k) 4x4 cost Hessian (-w gamma^t d2r) at once
        Hc = -Hm                                                      # (N, H, K, 4, 4)
        Hc = 0.5 * (Hc + np.swapaxes(Hc, -1, -2))
        ev, evec = np.linalg.eigh(Hc)
        Hc = (evec * np.maximum(ev, 0.0)[..., None, :]) @ np.swapaxes(evec, -1, -2)
        coef_all = w[:, None, :] * (self.gamma ** np.arange(H))[None, :, None]  # (N, H, K)
        Hc = Hc * coef_all[..., None, None]
        blk = [(slice(kk * D, (kk + 1) * D)) for kk in range(K)]
        for t in reversed(range(H)):
            coef = coef_all[:, t, :, None]                            # (N, K, 1)
            lx = (-coef * g[:, t, :, :D]).reshape(N, KD)
            lu = (-coef[..., 0] * g[:, t, :, D]).sum(1)               # (N,)
            Fx = np.zeros((N, KD, KD))
            lxx = np.zeros((N, KD, KD))
            for kk in range(K):
                Fx[:, blk[kk], blk[kk]] = fs[:, t, kk]
                lxx[:, blk[kk], blk[kk]] = Hc[:, t, kk, :D, :D]
            Fu = fu[:, t, :, :, 0].reshape(N, KD)                     # (N, KD)
            lux = Hc[:, t, :, D, :D].reshape(N, KD)
            luu = Hc[:, t, :, D, D].sum(1)
            FxT = np.swapaxes(Fx, 1, 2)
            Qx = lx + (FxT @ Vx[..., None])[..., 0]
            Qu = lu + (Fu * Vx).sum(1)
            VF = Vxx @ Fx
            Qxx = lxx + FxT @ VF
            Qux = lux + (Fu[:, None, :] @ VF)[:, 0]
            Quu = luu + ((Fu[:, None, :] @ Vxx)[:, 0] * Fu).sum(1)
            Quu_r = Quu + mu
            bad = Quu_r <= 1e-12
            ok &= ~bad
            Quu_r = np.where(bad, 1.0, Quu_r)
            kt = -Qu / Quu_r
            Kt = -Qux / Quu_r[:, None]
            u_new = U[:, t, 0] + kt
            clamped = (u_new < lo) | (u_new > hi)
            kt = np.where(clamped, np.clip(u_new, lo, hi) - U[:, t, 0], kt)
            Kt = np.where(clamped[:, None], 0.0, Kt)
            Vx = Qx + Kt * (Quu * kt)[:, None] + Kt * Qu[:, None] + Qux * kt[:, None]
            KQ = Kt[:, :, None] * Qux[:, None, :]
            Vxx = (Qxx + Quu[:, None, None] * Kt[:, :, None] * Kt[:, None, :]
                   + KQ + np.swapaxes(KQ, 1, 2))
            Vxx = 0.5 * (Vxx + np.swapaxes(Vxx, 1, 2))
            k_ff[:, t], K_fb[:, t] = kt, Kt
        return k_ff, K_fb, ok

    def _forward_ls(self, s0, X, U, k_ff, K_fb):
        """Closed-loop forward pass for every restart x line-search step at once.
        Returns new controls (N, L, H, A), states (N, L, H+1, K, D) and scenario
        returns (N, L, K)."""
        N, H, K, D = U.shape[0], self.horizon, self.k_models, self.obs_dim
        L = len(ILQR_LINE_SEARCH)
        al = torch.tensor(ILQR_LINE_SEARCH, device=self.device).view(1, L)
        Xn = torch.as_tensor(X, dtype=torch.float32, device=self.device)       # (N,H+1,K,D)
        Ut = torch.as_tensor(U[..., 0], dtype=torch.float32, device=self.device)
        kf = torch.as_tensor(k_ff, dtype=torch.float32, device=self.device)
        Kf = torch.as_tensor(K_fb, dtype=torch.float32, device=self.device)
        s = s0.view(1, 1, 1, D).expand(N, L, K, D).reshape(-1, D)
        Un, Rs, Xs = [], [], [s.reshape(N, L, K, D)]
        for t in range(H):
            dx = (s.reshape(N, L, K * D) - Xn[:, t].reshape(N, 1, K * D))
            u = Ut[:, t:t + 1] + al * kf[:, t:t + 1] + torch.einsum("nlx,nx->nl", dx, Kf[:, t])
            u = u.clamp(self.action_low, self.action_high)                # (N, L)
            s, r = self._step(s, u.reshape(N * L, 1).repeat_interleave(K, dim=0), (t,))
            Un.append(u)
            Rs.append(r.view(N, L, K))
            Xs.append(s.reshape(N, L, K, D))
        self.sim_steps += N * L * K * H
        disc = self.gamma ** torch.arange(H, device=self.device, dtype=torch.float32)
        J = (torch.stack(Rs, 2) * disc.view(1, 1, -1, 1)).sum(2)            # (N, L, K)
        return (torch.stack(Un, 2).unsqueeze(-1).cpu().numpy(), torch.stack(Xs, 2),
                J.cpu().double().numpy())

    def _init_controls(self):
        H, A, R = self.horizon, self.act_dim, self.n_restarts
        warm = np.roll(self._mu, -1, axis=0)
        warm[-1] = 0.0
        base = [warm, np.zeros((H, A)), np.full((H, A), self.action_low),
                np.full((H, A), self.action_high), np.full((H, A), 0.5 * self.action_low),
                np.full((H, A), 0.5 * self.action_high)]
        while len(base) < R:
            base.append(self.rng.uniform(self.action_low, self.action_high, (H, A)))
        return np.stack(base[:R]).astype(np.float64)

    @torch.no_grad()
    def act(self, obs):
        if self.alt_dynamics_fn is not None:
            raise NotImplementedError("iLQR has no hybrid/oracle rollout path")
        self.sim_steps = 0
        budget = self.sim_budget()
        H, K = self.horizon, self.k_models
        self._draw_scenarios()
        s0 = torch.as_tensor(np.asarray(obs, dtype=np.float32).ravel(), device=self.device)
        U = self._init_controls()                                       # (N, H, A)
        N = U.shape[0]
        roll = N * K * H
        iter_cost = roll * (1 + len(ILQR_LINE_SEARCH))                  # derivs + line search
        X, Rw = self._rollout_open(s0, U)
        J = self._returns(Rw).double().cpu().numpy()                     # (N, K)
        obj = self._cvar_rows(J)
        mu = np.full(N, ILQR_MU0)
        n_acc = np.zeros(N, dtype=int)
        best_i = int(np.argmax(obj))
        best_obj, best_U = float(obj[best_i]), U[best_i].copy()
        it = n_reseed = 0
        while self.sim_steps + iter_cost <= budget:
            it += 1
            w = self._tail_weights(J)
            fs, fu, g, Hm = self._derivs(X, U)
            k_ff, K_fb, ok = self._backward(fs, fu, g, Hm, w, U, mu)
            Un, Xn, Jn = self._forward_ls(s0, X.cpu().numpy(), U, k_ff, K_fb)
            objn = np.stack([self._cvar_rows(Jn[:, l]) for l in range(Jn.shape[1])], 1)
            objn = np.where(np.isfinite(objn), objn, -np.inf)
            l_best = objn.argmax(1)
            cand = objn[np.arange(N), l_best]
            acc = ok & (cand > obj)
            rel = (cand - obj) / np.maximum(1.0, np.abs(obj))
            if acc.any():
                rows = np.arange(N)
                U = np.where(acc[:, None, None], Un[rows, l_best], U)
                acc_t = torch.as_tensor(acc, device=self.device).view(N, 1, 1, 1)
                ti = torch.as_tensor(rows, device=self.device)
                X = torch.where(acc_t, Xn[ti, torch.as_tensor(l_best, device=self.device)], X)
                J = np.where(acc[:, None], Jn[rows, l_best], J)
                obj = np.where(acc, cand, obj)
                n_acc += acc
            if obj.max() > best_obj:
                best_i = int(np.argmax(obj))
                best_obj, best_U = float(obj[best_i]), U[best_i].copy()
            mu = np.where(acc, np.maximum(ILQR_MU_MIN, mu / 10.0), mu * 10.0)
            # converged (tiny accepted gain) or stuck (regulariser blown up):
            # re-seed the row so the remaining budget keeps searching
            done = (acc & (rel < ILQR_TOL)) | (mu > ILQR_MU_MAX)
            nd = int(done.sum())
            if nd and self.sim_steps + nd * K * H + iter_cost <= budget:
                idx = np.where(done)[0]
                for j, r in enumerate(idx):
                    if (n_reseed + j) % 2 == 0:
                        U[r] = np.clip(best_U + ILQR_REINIT_SIGMA * self.rng.normal(
                            size=best_U.shape), self.action_low, self.action_high)
                    else:
                        U[r] = self.rng.uniform(self.action_low, self.action_high,
                                                best_U.shape)
                n_reseed += nd
                Xd, Rd = self._rollout_open(s0, U[idx])
                ti = torch.as_tensor(idx, device=self.device)
                X[ti] = Xd
                J[idx] = self._returns(Rd).double().cpu().numpy()
                obj[idx] = self._cvar_rows(J[idx])
                mu[idx] = ILQR_MU0
                n_acc[idx] = 0
                if obj.max() > best_obj:
                    best_i = int(np.argmax(obj))
                    best_obj, best_U = float(obj[best_i]), U[best_i].copy()
        self._mu = best_U.astype(np.float32)
        self.last_plan_return = best_obj
        self.last_plan_return_best = best_obj
        self.last_ilqr = dict(iters=it, reseeds=n_reseed, sim_steps=self.sim_steps)
        return self._mu[0].copy()
