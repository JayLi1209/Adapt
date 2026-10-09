"""Continuous-action CEM planner with batched GPU rollouts.

All J candidates are rolled out in parallel through the BNN at each timestep,
giving ~J x higher GPU utilization compared to sequential (batch=1) rollouts.
"""

import numpy as np
import torch

from config import device

# ── Planner hyperparameters ────────────────────────────────────────────────────
# Swing-up needs ~2s of lookahead (H=40 @ dt=0.05); H=12 CANNOT swing up even with
# perfect dynamics (see collect_oracle_demos.py) -- so these defaults are the
# accurate config sweep_pendulum_alpha.py already uses, and every consumer
# (run_continuous, run_comparison, eval_pendulum*) now starts here.
H_PLAN = 40              # planning horizon (>= ~40 required for swing-up)
N_CEM_ITERS = 8          # CEM refinement iterations
N_CANDIDATES = 500       # action sequences per iteration (must be even)
ELITE_FRAC = 0.1         # keep top fraction
CVAR_ALPHA = 1.0         # CVaR tail (1.0 = risk-neutral mean)
K_MODELS = 1             # posterior BNN draws (1 = fast, >1 = epistemic diversity)
GAMMA = 0.99             # discount


PENDULUM_MAX_SPEED = 8.0     # gymnasium PendulumEnv.max_speed


def project_unit_circle(next_obs, max_speed=PENDULUM_MAX_SPEED):
    """Pendulum obs projection onto the VALID state manifold.

    (1) renormalise (cos,sin) [dims 0,1] to the unit circle;
    (2) clamp theta_dot [dim 2] to +/- max_speed.

    (2) matters as much as (1): the real env does
    `newthdot = np.clip(newthdot, -max_speed, max_speed)` every step, so
    |theta_dot| <= 8 ALWAYS holds in reality and the model was only ever trained
    on that range.  The learned model has no such clip, so across a 40-step
    self-feeding rollout imagined theta_dot can run to arbitrary magnitude, far
    off-distribution -- which both wrecks the state predictions and makes the
    analytic cost's 0.1*theta_dot^2 term explode (planned returns of -1e35 and
    float overflow to nan).  Clamping mirrors the env and keeps imagined states
    inside the training range.

    Pass max_speed=None to restore the old unclamped behaviour.
    """
    cs = next_obs[:, :2]
    norm = cs.norm(dim=1, keepdim=True).clamp_min(1e-6)
    thdot = next_obs[:, 2:]
    if max_speed is not None:
        thdot = thdot.clamp(-max_speed, max_speed)
    return torch.cat([cs / norm, thdot], dim=1)


def make_box_projection(low, high):
    """Obs projection that CLAMPS every dim into [low, high] (the box analogue of
    project_unit_circle, for envs whose observation is a raw state vector).

    Same rationale: the real env's state is bounded (MuJoCo joint limits, and in
    practice a bounded velocity envelope), but a learned model self-fed for H
    steps has no such bound, so imagined states can run far off-distribution and
    both wreck the predictions and corrupt the return.  Clamping mirrors the env
    and keeps imagined states inside the training range.
    """
    lo = torch.as_tensor(low, dtype=torch.float32, device=device)
    hi = torch.as_tensor(high, dtype=torch.float32, device=device)

    def _project(next_obs):
        return torch.clamp(next_obs, lo, hi)

    return _project


# ── InvertedPendulum-v5 (MuJoCo cart-pole) ────────────────────────────────────
# obs = [x, theta, x_dot, theta_dot]; reward = int(not terminated) = +1 per
# surviving step; terminated iff |theta| > 0.2 rad (or the state goes non-finite).
IP_ANGLE_LIMIT = 0.2
IP_THETA_DIM = 1


def inverted_pendulum_terminated(next_obs, angle_limit=IP_ANGLE_LIMIT):
    """GROUND-TRUTH InvertedPendulum-v5 termination, as a per-row bool tensor.

    Mirrors InvertedPendulumEnv.step exactly:
        terminated = not isfinite(obs).all() or |obs[1]| > 0.2
    """
    finite = torch.isfinite(next_obs).all(dim=1)
    upright = next_obs[:, IP_THETA_DIM].abs() <= angle_limit
    return ~(finite & upright)


def make_inverted_pendulum_terminal(weight=0.5, theta_ref=IP_ANGLE_LIMIT,
                                    theta_dot_ref=1.0, x_ref=1.0, x_dot_ref=1.0):
    """Terminal potential Phi(s_H) for the InvertedPendulum survival objective.

    WHY THIS IS NEEDED.  The env pays +1 per surviving step, so a candidate's
    score is decided entirely by WHEN it dies.  Near the upright equilibrium the
    pole physically cannot fall inside the planning horizon -- 20 steps is 0.8 s,
    about 3.2 instability time constants, so an uncontrolled deviation from
    theta ~ 1e-3 only reaches ~0.025, well short of the 0.2 limit.  Virtually
    every candidate therefore scores the identical maximum, the elite set is an
    arbitrary tie-break, the refitted mean collapses toward zero action, and the
    planner does nothing until the state has drifted far enough out that the
    horizon finally becomes informative -- by which point recovery is marginal.
    Lengthening the horizon instead is not available: past ~25 steps the model's
    imagined trajectory has diverged and it mislabels who survives.

    WHY IT DOES NOT CHANGE THE PROBLEM.  This is POTENTIAL-BASED shaping, whose
    per-step form F(s, s') = gamma * Phi(s') - Phi(s) provably leaves the optimal
    policy unchanged (Ng et al., 1999).  Summed over an H-step rollout it
    telescopes to gamma^H Phi(s_H) - Phi(s_0), and s_0 is shared by every
    candidate, so adding gamma^H Phi(s_H) IS that shaping, exactly.  Read the
    other way it is the horizon's missing terminal value: "how much room is left
    at the end", which is what the truncated sum threw away.

    The magnitude is deliberately capped below the value of one extra step of
    life (Phi in [-weight, 0] with weight 0.5, versus ~0.83 for surviving the
    final step), so shaping can only ever break ties AMONG survivors -- it can
    never make a candidate that dies outrank one that lives.

    The evaluated return is untouched by any of this: the agent is still scored
    on the env's own +1-per-step reward.
    """
    ref = torch.tensor([x_ref, theta_ref, x_dot_ref, theta_dot_ref],
                       dtype=torch.float32, device=device)

    def _terminal(final_obs, alive):
        cost = ((final_obs / ref) ** 2).sum(dim=1).clamp(max=1.0)
        # A dead trajectory has no margin left, so it takes the full penalty
        # rather than being scored on a meaningless post-fall state.
        cost = alive * cost + (1.0 - alive) * 1.0
        return -weight * cost

    return _terminal


def inverted_pendulum_reward(obs, act):
    """GROUND-TRUTH InvertedPendulum-v5 reward: +1 for every step.

    Constant, hence carrying no gradient on its own -- the planner's entire
    signal comes from multiplying this by the alive mask that `term_fn` drives
    (see ContinuousCEMAgent.act).  Kept explicit rather than folded into the mask
    so the reward and the termination stay separately auditable, and so the
    learned reward head is never consulted for a quantity we know exactly.
    """
    return torch.ones(obs.shape[0], dtype=torch.float32, device=obs.device)


# Operating envelope of a competent controller on this env, measured from a
# discrete-time LQR on the linearised simulator: |x| < 0.03, |theta| < 0.01,
# |x_dot| < 0.04, |theta_dot| < 0.07.  The running cost is normalised by scales a
# few times larger than these, so it varies over the region where control
# actually happens instead of being numerically flat there.
IP_STATE_REF = (0.30, 0.02, 0.30, 0.20)     # x, theta, x_dot, theta_dot


def make_inverted_pendulum_reward(w_state=0.3, w_act=0.02, refs=IP_STATE_REF,
                                  act_scale=3.0):
    """Regularised MPC running reward:  1 - w_state * c(s) - w_act * (a/a_max)^2.

    WHY THE BARE +1 REWARD IS NOT PLANNABLE.  With survival as the only term, a
    candidate is scored purely by WHEN it dies, and near the equilibrium nothing
    dies inside the horizon -- so several hundred candidates tie at the maximum.
    CEM then averages the first actions of an arbitrary elite subset.  Those
    first actions disagree wildly (measured: +/-0.7 where LQR wants ~0.001)
    because in an open-loop plan almost any a_0 can still be corrected by later
    actions in the same sequence: a_0 is genuinely UNDER-DETERMINED, and the
    committed action is therefore noise.  The pole then random-walks out until
    the horizon finally sees a fall, by which point recovery is marginal.

    The two extra terms are the textbook MPC regularisers that remove that
    degeneracy, and each fixes a distinct half of it:

      w_act   penalises control effort, which makes the minimiser of a_0 UNIQUE
              (of all the recoverable first actions, prefer the smallest);
      w_state penalises deviation at EVERY step rather than only at the horizon,
              so a candidate that wanders and returns loses to one that stays put.

    Both are bounded well below the +1 survival term (c is clamped to 1), so
    ranking still puts survival first and these only order the survivors.

    The EVALUATED return is never touched: the agent is still scored on the env's
    own +1-per-step reward.  This is the planner's internal objective only.
    """
    ref = torch.tensor(refs, dtype=torch.float32, device=device)

    def _reward(obs, act):
        cost = ((obs / ref) ** 2).sum(dim=1).clamp(max=1.0)
        effort = (act[:, 0] / act_scale) ** 2
        return 1.0 - w_state * cost - w_act * effort

    return _reward


def pendulum_reward(obs, act):
    """GROUND-TRUTH Pendulum-v1 reward from (obs, action) -- the analytic cost.

    Mirrors gymnasium PendulumEnv.step exactly:
        costs = angle_normalize(theta)^2 + 0.1*theta_dot^2 + 0.001*u^2
        reward = -costs
    computed from the CURRENT state and action (as the env does), with u clipped
    to the actuator limit.  obs = [cos(theta), sin(theta), theta_dot]; atan2
    already returns theta in [-pi, pi], so it IS angle_normalize(theta).

    Using this in the planner instead of the BNN's learned reward head bounds
    every imagined step's reward at <= 0 (the true MDP's bound), which the
    unbounded learned head does not respect off-distribution.
    """
    theta = torch.atan2(obs[:, 1], obs[:, 0])
    theta_dot = obs[:, 2]
    u = act[:, 0].clamp(-2.0, 2.0)
    costs = theta ** 2 + 0.1 * theta_dot ** 2 + 0.001 * u ** 2
    return -costs


class ContinuousCEMAgent:
    """CEM over continuous action sequences with batched BNN rollouts."""

    def __init__(self, dyn, bnn, obs_dim, act_dim, device=device,
                 horizon=H_PLAN, n_cem_iters=N_CEM_ITERS,
                 n_candidates=N_CANDIDATES, elite_frac=ELITE_FRAC,
                 k_models=K_MODELS, cvar_alpha=CVAR_ALPHA, gamma=GAMMA,
                 obs_project=None, reward_fn=None, reward_clip=None,
                 term_fn=None, terminal_fn=None, action_low=-2.0, action_high=2.0,
                 sigma_init=1.0, rng=None, **kwargs):
        self.dyn = dyn
        self.bnn = bnn
        # Optional GROUND-TRUTH termination predicate term_fn(next_obs) -> bool
        # tensor.  When supplied, the rollout carries an ALIVE MASK: once a
        # candidate's imagined trajectory terminates, every later step of that
        # trajectory contributes zero reward, exactly as the real MDP does.
        #
        # This is mandatory for InvertedPendulum, whose reward is the constant +1
        # -- without the mask every action sequence scores identically and CEM is
        # a no-op.  It is left OFF by default because masking is only correct for
        # non-negative rewards: on Pendulum (reward <= 0 always) zeroing the tail
        # would make dying the highest-scoring outcome.
        self.term_fn = term_fn
        # Optional terminal value terminal_fn(final_obs, alive) -> (B,) added to
        # the return with the horizon's discount, i.e. gamma^H Phi(s_H).  This is
        # potential-based shaping (policy-preserving) / the truncated horizon's
        # missing terminal value.  None = the plain H-step sum, unchanged.
        self.terminal_fn = terminal_fn
        # Optional analytic reward r(obs_t, act_t) used INSTEAD of the model's
        # learned reward head during imagined rollouts.  None = use the learned
        # head (previous behaviour, unbounded off-distribution).
        self.reward_fn = reward_fn
        # Optional (lo, hi) clamp on every imagined step's reward.  The analytic
        # Pendulum reward is <= 0 but UNBOUNDED BELOW, because imagined theta_dot
        # is not clipped the way the env clips it (max_speed=8), so 0.1*theta_dot^2
        # can explode and overflow the summed return.  Clamping to the true
        # per-step range keeps the H-step return finite and comparable.
        self.reward_clip = reward_clip
        # Optional callable applied to each imagined next_obs in the rollout, to
        # keep it on the state manifold (e.g. re-normalise Pendulum's (cos,sin)
        # to the unit circle) so multi-step model error compounds less.
        self.obs_project = obs_project
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

        # Actuator limits.  These MUST match the env: candidates are clipped to
        # them, so bounds wider than the env's make the planner commit to forces
        # the env will silently clip (a model/plan mismatch), and narrower bounds
        # hide usable control authority.  Pendulum-v1 is [-2, 2] (the default);
        # InvertedPendulum-v5 is [-3, 3].
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self._mu = np.zeros((horizon, act_dim), dtype=np.float32)
        self._sigma = np.full((horizon, act_dim), sigma_init, dtype=np.float32)
        self._sigma_min = 0.05
        self._sigma_init = float(sigma_init)  # sigma each replan restarts from (see act())

        self.surprise_bar = 1.0
        self.n_since_change = 0
        # Diagnostics: the committed plan's predicted (discounted, H-step) return.
        # last_plan_return = mean CVaR score of the elite set; _best = top candidate.
        self.last_plan_return = float("nan")
        self.last_plan_return_best = float("nan")
        # When True, act() records per-CEM-iteration diagnostics into self.iter_diag
        # (pool diversity, score distribution, posterior-draw spread, elites).  Off
        # by default -- zero overhead on the normal path.
        self.record_diag = False
        self.iter_diag = []

    def reset(self):
        self._mu = np.zeros((self.horizon, self.act_dim), dtype=np.float32)
        self._sigma = np.full((self.horizon, self.act_dim), self._sigma_init,
                              dtype=np.float32)

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

        # ── MPC warm start ────────────────────────────────────────────────────
        # Shift the previous plan forward one step, and RESET sigma so this
        # replan explores.  self._sigma is overwritten with the elite std at the
        # end of every CEM iteration below; without re-initialising it here it
        # carries over from the last act() at ~_sigma_min (0.05), so from step 2
        # onward every candidate is mu + 0.05*noise around a stale, unshifted mu.
        # The planner then effectively solves once at step 1 and coasts: it can
        # never re-plan, and with all candidates near-identical their returns are
        # near-identical, which also makes CVaR-alpha a no-op after step 1.
        self._mu = np.roll(self._mu, -1, axis=0)
        self._mu[-1] = 0.0
        self._sigma = np.full((H, A), self._sigma_init, dtype=np.float32)

        if self.record_diag:
            self.iter_diag = []

        try:
            for _it in range(self.n_cem_iters):
                # Sample J action sequences: (J, H, A)
                noise = self.rng.normal(size=(J, H, A)).astype(np.float32)
                candidates = self._mu + self._sigma * noise
                candidates = np.clip(candidates, self.action_low, self.action_high)

                # ── Batched rollout + CVaR score of every candidate ──────
                returns_grouped = self._rollout_returns(obs_arr, candidates)  # (J, K)
                scores = self._scores(returns_grouped)                         # (J,)

                # ── Select elites and refit Gaussian ────────────────────
                elite_idx = np.argpartition(-scores, self.n_elite - 1)[:self.n_elite]
                elite_seq = candidates[elite_idx]                        # (n_elite, H, A)
                self._mu = elite_seq.mean(axis=0)
                self._sigma = np.maximum(elite_seq.std(axis=0), self._sigma_min)
                # Predicted return of the current plan (last iter's values persist).
                self.last_plan_return = float(scores[elite_idx].mean())
                self.last_plan_return_best = float(scores.max())

                if self.record_diag:
                    finite = scores[np.isfinite(scores)]
                    top5 = np.sort(finite)[-5:][::-1] if len(finite) else np.array([])
                    # epistemic spread = how differently the K posterior draws of the
                    # SAME candidate score, averaged over candidates (diverse pool?).
                    epi_within = (float(np.nanmean(returns_grouped.std(axis=1)))
                                  if K > 1 else 0.0)
                    self.iter_diag.append(dict(
                        it=_it,
                        # candidate-pool diversity (action space)
                        pool_a0_std=float(candidates[:, 0, 0].std()),
                        pool_seq_std=float(candidates.std(axis=0).mean()),
                        # score distribution across the J candidates
                        score_min=float(finite.min()) if len(finite) else float("nan"),
                        score_mean=float(finite.mean()) if len(finite) else float("nan"),
                        score_max=float(finite.max()) if len(finite) else float("nan"),
                        score_std=float(finite.std()) if len(finite) else float("nan"),
                        n_nonfinite=int(np.sum(~np.isfinite(scores))),
                        # posterior-draw diversity within a candidate (epistemic)
                        epi_within=epi_within,
                        elite_score_mean=float(self.last_plan_return),
                        top5_scores=[round(float(x), 3) for x in top5],
                        mu_a0=float(self._mu[0, 0]),
                        sigma_a0=float(self._sigma[0, 0]),
                    ))
        finally:
            self.bnn.num_weight_groups = saved_groups

        return self._mu[0].copy()

    @torch.no_grad()
    def _rollout_returns(self, obs_arr, candidates):
        """Imagined discounted H-step return of every candidate under K posterior
        draws: (J, H, A) action sequences -> (J, K) returns.

        This is the planner-independent EVALUATOR (MPPI / MCTS in
        planning/continuous_planners.py score their sequences through it too, so
        a planner comparison changes only the optimizer).  Caller must have set
        bnn.num_weight_groups = K.  Row j*K + k uses weight group k, so at every
        imagined step all candidates share the same K fresh weight draws.
        """
        J, H = candidates.shape[0], candidates.shape[1]
        K = self.k_models
        total_batch = J * K
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
        # alive[i] == 1.0 while candidate i's imagined trajectory has not
        # yet terminated.  Stays all-ones when term_fn is None, so the
        # masked update below is bit-identical to the unmasked rollout.
        alive = torch.ones(total_batch, device=self.device)
        disc = 1.0

        # With K>1 posterior draws we MUST sample weights (deterministic
        # =False), else BayesianLinear returns the mean weights for every
        # row (see bnn/layers.py: `if not sample: use weight_mu`), the K
        # draws collapse to identical returns, and CVaR-alpha has NO effect.
        # K==1 keeps the deterministic mean-model rollout (risk-neutral).
        rollout_det = (K <= 1)
        for t in range(H):
            act_t = torch.as_tensor(act_seqs[:, t, :], dtype=torch.float32,
                                    device=self.device)          # (J*K, act_dim)
            cur_obs = obs_t          # state at imagined time t (pre-transition)
            next_obs, rew, _, _ = self.dyn.sample(
                act_t, state, deterministic=rollout_det)
            if self.obs_project is not None:
                next_obs = self.obs_project(next_obs)

            # Reward from the ANALYTIC cost r(s_t, a_t) when supplied (the
            # env computes its reward from the pre-transition state too);
            # otherwise fall back to the model's learned reward channel.
            if self.reward_fn is not None:
                step_rew = self.reward_fn(cur_obs, act_t)
            else:
                step_rew = rew.squeeze(-1)
            if self.reward_clip is not None:
                step_rew = step_rew.clamp(self.reward_clip[0],
                                          self.reward_clip[1])
            if self.term_fn is not None:
                # The env pays reward for step t iff s_{t+1} is NOT the
                # terminating state (InvertedPendulumEnv: reward =
                # int(not terminated)), so the mask is updated with the
                # freshly imagined next_obs BEFORE it scales this step's
                # reward.  Once dead a trajectory stays dead.
                alive = alive * (~self.term_fn(next_obs)).float()
                step_rew = step_rew * alive
            returns += disc * step_rew
            obs_t = next_obs
            state = self.dyn.reset(obs_t)
            disc *= self.gamma

        if self.terminal_fn is not None:
            # disc == gamma^H here (it was multiplied once per step).
            returns = returns + disc * self.terminal_fn(obs_t, alive)

        return returns.cpu().numpy().reshape(J, K)

    def _scores(self, returns_grouped):
        """Score each candidate as CVaR_alpha over its K returns: (J, K) -> (J,)."""
        if returns_grouped.shape[1] > 1:
            return np.array([self._cvar_value(r) for r in returns_grouped])
        return returns_grouped[:, 0]

    def _cvar_value(self, returns):
        m = len(returns)
        k = max(1, int(np.ceil(self.cvar_alpha * m)))
        return float(np.sort(returns)[:k].mean())
