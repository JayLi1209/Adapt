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


def pendulum_reward(obs, act, max_torque=2.0):
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
    u = act[:, 0].clamp(-max_torque, max_torque)
    costs = theta ** 2 + 0.1 * theta_dot ** 2 + 0.001 * u ** 2
    return -costs


class ContinuousCEMAgent:
    """CEM over continuous action sequences with batched BNN rollouts."""

    def __init__(self, dyn, bnn, obs_dim, act_dim, device=device,
                 horizon=H_PLAN, n_cem_iters=N_CEM_ITERS,
                 n_candidates=N_CANDIDATES, elite_frac=ELITE_FRAC,
                 k_models=K_MODELS, cvar_alpha=CVAR_ALPHA, gamma=GAMMA,
                 obs_project=None, reward_fn=None, reward_clip=None,
                 terminal_value_fn=None, alt_dynamics_fn=None,
                 alt_dynamics_after=None, action_low=-2.0, action_high=2.0,
                 rng=None, **kwargs):
        self.dyn = dyn
        self.bnn = bnn
        # Optional learned TERMINAL VALUE V(s_H) added as gamma^H * V(s_H) to every
        # candidate's score.  With it the planner no longer has to see the whole
        # task inside the rollout, so H can be cut -- which is the point: every
        # imagined step compounds the BNN's one-step error, so a 40-step rollout
        # accumulates ~4x more model error than a 10-step one.  The tail beyond H
        # is then supplied by a function of the SINGLE state s_H rather than by
        # 30 more error-compounding model steps.  None = previous behaviour.
        # Signature: terminal_value_fn(obs) -> (B,) tensor, obs is (B, obs_dim).
        self.terminal_value_fn = terminal_value_fn
        # Optional HYBRID rollout: use self.dyn (the BNN) for imagined steps
        # t < alt_dynamics_after, then switch to alt_dynamics_fn(obs, act) for the
        # rest.  Lets a rollout mix a stale/uncertain model near the root with a
        # different transition model in the tail, to attribute plan quality to one
        # segment or the other.  Requires reward_fn (the alternate model supplies
        # states only, not rewards).  None = single-model rollout, as before.
        self.alt_dynamics_fn = alt_dynamics_fn
        self.alt_dynamics_after = alt_dynamics_after
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

        # Actuator bounds the CEM samples within.  Defaults are Pendulum-v1's
        # shipped +/-2; widen them together with the env's max_torque to plan
        # over an uncapped actuator.
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self._mu = np.zeros((horizon, act_dim), dtype=np.float32)
        # Exploration widths SCALE WITH THE ACTUATOR RANGE, so a wider actuator is
        # actually explored: at the shipped +/-2 these evaluate to the historical
        # 1.0 / 0.05, but with +/-20 a fixed 1.0 would sample only ~+/-2 around mu
        # and the extra authority would never be tried.
        _rng_a = self.action_high - self.action_low
        self._sigma_init = _rng_a / 4.0            # 1.0 at +/-2
        self._sigma_min = _rng_a / 80.0            # 0.05 at +/-2
        self._sigma = np.full((horizon, act_dim), self._sigma_init, dtype=np.float32)

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
                    if (self.alt_dynamics_fn is not None
                            and t >= self.alt_dynamics_after):
                        next_obs = self.alt_dynamics_fn(cur_obs, act_t)
                        rew = None           # reward_fn supplies the reward below
                    else:
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
                    returns += disc * step_rew
                    obs_t = next_obs
                    state = self.dyn.reset(obs_t)
                    disc *= self.gamma

                # Terminal value: `disc` is exactly gamma^H here (multiplied once
                # per completed step), and obs_t is s_H, so this appends the
                # discounted tail value the truncated rollout would otherwise drop.
                if self.terminal_value_fn is not None:
                    returns += disc * self.terminal_value_fn(obs_t)

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

    def _cvar_value(self, returns):
        m = len(returns)
        k = max(1, int(np.ceil(self.cvar_alpha * m)))
        return float(np.sort(returns)[:k].mean())
