"""CVaR-CEM-greedy MPC planner (Ayan's note, Sec. 3) for discrete FrozenLake.

    a*_{0:H-1} = argmax_{a_{0:H-1}}  CVaR_alpha[ Z(a_{0:H-1}) ]  +  beta * U(s0, a0)
    Z = sum_t gamma^t r_t  +  gamma^H V(s_H)          (imagined return)

A categorical CEM loop maintains a per-timestep distribution pi[t] over the 4
actions, samples J sequences, scores each by the empirical CVaR over K posterior
transition models x N aleatoric rollouts, keeps the top elites, and refits pi.
Inherits the BNN env-modeling plumbing from BNNModelPlanner; the planning logic
is unchanged from risk_averse_ayan.CVaRCEMAgent.
"""
import numpy as np
import torch

from planning.base import BNNModelPlanner

# ── CEM + CVaR planner hyperparameters (Ayan's note, Sec. 3) ──────────────────
H_PLAN = 6             # planning horizon
N_CEM_ITERS = 5        # CEM refinement iterations I
N_CANDIDATES = 256     # action sequences J per iteration
ELITE_FRAC = 0.1       # rho: elite fraction
CVAR_ALPHA = 1.0       # CVaR tail fraction alpha when ADAPTIVE_ALPHA=False (1.0 =
                       # full dist = risk-neutral; lower toward 0.1 = risk-averse)

# ── adaptive risk-aversion (alpha) ────────────────────────────────────────────
# confidence = conf_data * conf_surprise drives alpha from ALPHA_MIN (worst-tail,
# risk-averse, uninformed) to ALPHA_MAX (risk-neutral, confident).
ADAPTIVE_ALPHA = False
ALPHA_MIN = 0.95       # most risk-averse end (alpha->0 => CVaR = single worst return)
ALPHA_MAX = 1.0        # least risk-averse alpha (confident -> risk-neutral mean)
N_CONFIDENT = 12       # post-change samples for the data gate to saturate
SURPRISE_TAU = 2.0     # surprise sensitivity: bigger -> tolerate more surprise
K_MODELS = 10          # K posterior transition matrices (epistemic axis)
N_ROLLOUTS = 32        # N aleatoric rollouts per model per candidate
BETA_EXPLORE = 0.0     # information-bonus weight beta (OFF by default; confidence-gated)
PLAN_GAMMA = 0.95      # discount gamma for the imagined return
ACTION_SMOOTH = 0.05   # Laplace smoothing when refitting the categorical pi[t]
WARM_START = True      # shift the previous plan forward as the CEM init


class CVaRCEMAgent(BNNModelPlanner):
    def __init__(self, dynamics_model, bnn, desc, device, n_actions=4,
                 n_cem_iters=N_CEM_ITERS,
                 n_candidates=N_CANDIDATES,
                 k_models=K_MODELS,
                 n_rollouts=N_ROLLOUTS,
                 horizon=H_PLAN,
                 elite_frac=ELITE_FRAC,
                 cvar_alpha=CVAR_ALPHA,
                 adaptive_alpha=ADAPTIVE_ALPHA, alpha_min=ALPHA_MIN,
                 alpha_max=ALPHA_MAX, n_confident=N_CONFIDENT,
                 surprise_tau=SURPRISE_TAU,
                 beta=BETA_EXPLORE, gamma=PLAN_GAMMA, action_smooth=ACTION_SMOOTH,
                 warm_start=WARM_START, rng=None, **kwargs):
        # BNNModelPlanner sets up the map-derived reward/terminal/heuristic arrays.
        super().__init__(dynamics_model, bnn, desc, device, n_actions=n_actions,
                         gamma=gamma, rng=rng, **kwargs)
        self.horizon = horizon
        self.n_cem_iters = n_cem_iters
        self.n_candidates = n_candidates
        self.n_elite = max(1, int(round(elite_frac * n_candidates)))
        self.cvar_alpha = cvar_alpha
        # ── adaptive risk-aversion state ─────────────────────────────────────
        self.adaptive_alpha = adaptive_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.n_confident = n_confident
        self.surprise_tau = surprise_tau
        # Belief about the (possibly changed) env, set by the env loop each step:
        #   surprise_bar : smoothed calibrated surprise delta_bar (~1 == model fits)
        #   n_since_change : post-change transitions observed so far
        self.surprise_bar = 1.0
        self.n_since_change = 0
        self.last_alpha = cvar_alpha
        self.k_models = k_models
        self.n_rollouts = n_rollouts
        self.beta = beta
        self.action_smooth = action_smooth
        self.warm_start = warm_start
        # Reward-on-arrival vector and terminal/leaf arrays as plain numpy.
        self.reward_vec = self.cell_reward.astype(np.float64)        # goal +1, hole -1
        self.is_terminal = self.terminal.copy()                     # bool (S,)
        self.terminal_value = self.heuristic.astype(np.float64)     # gamma^dist(s,goal)
        self._uniform = np.full((self.horizon, self.n_actions),
                                1.0 / self.n_actions, dtype=np.float64)
        self.pi = self._uniform.copy()      # per-timestep categorical (H, A)
        # diagnostics from the last act() (for logging)
        self.last_cvar = 0.0
        self.last_bonus = 0.0
        self.last_qrisk = np.zeros(self.n_actions)

    # The CEM planner does not use a persistent tree; reset/notify only reset the
    # warm-started plan so a regime change doesn't bias the next plan.
    def reset(self):
        self.pi = self._uniform.copy()

    def notify_change(self):
        self.pi = self._uniform.copy()
        # The new regime is unknown: drop to no-data so alpha falls to ALPHA_MIN
        # (maximally risk-averse) until post-change evidence rebuilds confidence.
        self.n_since_change = 0
        self.surprise_bar = 1.0

    def _confidence(self):
        """How much we trust the current model of the (possibly changed) env, in
        [0,1].  Two gates that must BOTH hold:
          conf_data     : ramps 0->1 as post-change samples accumulate.
          conf_surprise : 1 when smoothed surprise is calibrated, ->0 on a spike.
        """
        conf_data = min(1.0, self.n_since_change / max(1, self.n_confident))
        excess = max(0.0, self.surprise_bar - 1.0)
        conf_surprise = float(np.exp(-excess / self.surprise_tau))
        return conf_data * conf_surprise

    def _adaptive_alpha(self, conf):
        """CVaR tail alpha from confidence: ALPHA_MIN (risk-averse, worst-tail) when
        uninformed -> ALPHA_MAX (risk-neutral) once confident."""
        if not self.adaptive_alpha:
            return self.cvar_alpha
        return self.alpha_min + (self.alpha_max - self.alpha_min) * conf

    def _rollout_returns(self, s0, A_seq, Ts):
        """Vectorised imagined returns Z for every candidate x model x rollout.

        A_seq : (J, H) int action sequences.
        Ts    : (K, S, A, S) posterior transition matrices.
        Returns Z of shape (J, K*N): for each candidate j, K*N sampled returns
            Z = sum_t gamma^t reward_vec[s_{t+1}] + gamma^H V(s_H),
        terminating (and bootstrapping with 0) on the first absorbing cell.
        """
        J, H = A_seq.shape
        K, N = self.k_models, self.n_rollouts
        M = J * K * N
        cand_idx = np.repeat(np.arange(J), K * N)            # (M,)
        model_idx = np.tile(np.repeat(np.arange(K), N), J)   # (M,)
        state = np.full(M, s0, dtype=np.int64)
        done = np.zeros(M, dtype=bool)
        returns = np.zeros(M, dtype=np.float64)
        disc = 1.0
        for t in range(H):
            a = A_seq[cand_idx, t]                           # (M,)
            probs = Ts[model_idx, state, a]                  # (M, S)
            cdf = np.cumsum(probs, axis=1)
            u = self.rng.random((M, 1)) * cdf[:, -1:]
            s2 = (cdf >= u).argmax(axis=1)                   # sample next cell
            s2 = np.where(done, state, s2)
            returns += disc * np.where(done, 0.0, self.reward_vec[s2])
            done = done | self.is_terminal[s2]
            state = s2
            disc *= self.gamma
        # terminal bootstrap V(s_H) only for trajectories still running
        returns += disc * np.where(done, 0.0, self.terminal_value[state])
        return returns.reshape(J, K * N)

    @staticmethod
    def _cvar(Z, alpha):
        """Empirical CVaR_alpha per candidate: mean of the worst ceil(alpha*m)
        returns in each row of Z."""
        m = Z.shape[1]
        k = max(1, int(np.ceil(alpha * m)))
        worst = np.sort(Z, axis=1)[:, :k]        # k lowest returns per candidate
        return worst.mean(axis=1)

    def _info_bonus(self, s0, Ts):
        """U(s0, a): epistemic spread of the (s0, a) transition across the K
        posterior T's -- summed variance of the next-state row over models."""
        rows = Ts[:, s0, :, :]                    # (K, A, S)
        return rows.var(axis=0).sum(axis=1)       # (A,)

    def _pretrained_policy(self):
        """Greedy policy of the model: value iteration on the model's own
        deterministic-mean transitions and reward-on-arrival.  Returns
        (policy (S,), T_mean (S,A,S)).  Recomputed each act() so it tracks the
        online count / forgetting updates."""
        Tm, _ = self._model_matrices(1, deterministic=True)
        T = Tm[0]                                          # (S,A,S)
        r = self.reward_vec                                # (S,) reward on arrival
        term = self.is_terminal
        cont = (~term).astype(np.float64)                  # bootstrap 0 at absorbers
        exp_r = (T * r[None, None, :]).sum(-1)             # (S,A) E[reward on arrival]
        V = np.zeros(self.n, dtype=np.float64)
        Q = exp_r
        for _ in range(500):
            Q = exp_r + self.gamma * (T * (V * cont)[None, None, :]).sum(-1)
            Vn = Q.max(1); Vn[term] = 0.0
            if np.max(np.abs(Vn - V)) < 1e-10:
                V = Vn; break
            V = Vn
        return Q.argmax(1), T

    def _policy_init_pi(self, s0):
        """Per-timestep categorical seeded with the pretrained policy UNROLLED from
        s0 under the mean dynamics (peaked on the policy action, lightly smoothed so
        CEM still explores around it)."""
        policy, T = self._pretrained_policy()
        pi = np.full((self.horizon, self.n_actions), self.action_smooth, dtype=np.float64)
        s = s0
        for t in range(self.horizon):
            a = int(policy[s])
            pi[t, a] += 1.0
            s = int(np.argmax(T[s, a]))                    # expected next state
        pi /= pi.sum(1, keepdims=True)
        return pi

    @torch.no_grad()
    def act(self, obs, **kwargs):
        s0 = int(np.argmax(obs))
        if self.is_terminal[s0]:
            return self._eye_a[0].cpu().numpy()

        # K posterior transition matrices (Thompson draws -> epistemic axis).
        Ts, _ = self._model_matrices(self.k_models, deterministic=False)

        # CEM init: seed the per-timestep categorical with the model's OWN greedy
        # policy, unrolled from s0, so the rollouts evaluate continuations that
        # FOLLOW THE MODEL POLICY rather than random walks.
        self.pi = self._policy_init_pi(s0)

        # ── Confidence-gated risk control ────────────────────────────────────
        # One confidence signal drives BOTH knobs: alpha (risk tail) and whether
        # the exploration bonus is on.
        conf = self._confidence() if self.adaptive_alpha else 1.0
        alpha = self._adaptive_alpha(conf)
        eff_beta = self.beta * conf
        bonus = eff_beta * self._info_bonus(s0, Ts)          # (A,)
        self.last_alpha = alpha

        J, H, A = self.n_candidates, self.horizon, self.n_actions
        last_scores = None
        last_seqs = None
        for _ in range(self.n_cem_iters):
            # Sample J action sequences from the per-timestep categorical.
            cdf = np.cumsum(self.pi, axis=1)                 # (H, A)
            u = self.rng.random((J, H))
            A_seq = (u[:, :, None] < cdf[None, :, :]).argmax(axis=2)  # (J, H)
            # Score each candidate: empirical CVaR + info bonus on a0.
            Z = self._rollout_returns(s0, A_seq, Ts)         # (J, K*N)
            cvar = self._cvar(Z, alpha)                      # (J,)
            scores = cvar + bonus[A_seq[:, 0]]               # (J,)
            # Select elites and refit pi[t] to their action frequencies.
            elites = np.argpartition(-scores, self.n_elite - 1)[:self.n_elite]
            elite_seq = A_seq[elites]                        # (n_elite, H)
            counts = np.stack([np.bincount(elite_seq[:, t], minlength=A)
                               for t in range(H)]).astype(np.float64)
            self.pi = (counts + self.action_smooth)
            self.pi /= self.pi.sum(axis=1, keepdims=True)
            last_scores, last_seqs = scores, A_seq

        # Diagnostics: per-first-action mean CVaR over the final population.
        a0 = int(np.argmax(self.pi[0]))
        qrisk = np.full(A, -np.inf)
        for ai in range(A):
            mask = last_seqs[:, 0] == ai
            if mask.any():
                qrisk[ai] = last_scores[mask].mean()
        self.last_qrisk = qrisk
        self.last_cvar = float(qrisk[a0]) if np.isfinite(qrisk[a0]) else 0.0
        self.last_bonus = float(bonus[a0])
        return self._eye_a[a0].cpu().numpy()
