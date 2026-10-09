"""Oracle CEM: the same CEM planner, but rolling out the TRUE MuJoCo dynamics.

This is the gold-standard upper bound the learned world model is measured
against.  It is deliberately the SAME search algorithm as
planning.continuous_cem.ContinuousCEMAgent -- identical horizon, candidate
count, iteration count, elite fraction, MPC warm start, sigma schedule and
action clipping -- with exactly one substitution: every imagined step calls
MuJoCo instead of the BNN, and every imagined reward is the env's own reward
rather than a learned head.

That is what makes the comparison mean something.  Any gap between the two is
attributable to WORLD-MODEL ERROR alone, not to a better or worse search: run
the same CEM twice, once on a learned model and once on the simulator, and the
difference is the price of not knowing the dynamics.

Notes
-----
* CVaR is absent here on purpose.  CVaR-alpha averages the lower tail across K
  posterior draws of a *Bayesian* model; the oracle has no posterior and the
  plant is deterministic, so all K draws would be identical and any alpha would
  be a no-op.  The oracle is therefore risk-neutral by construction -- which is
  correct: there is nothing to be risk-averse about when you know the dynamics.
* Rollouts go through `mujoco.rollout`, which evaluates all J candidates in a
  C++ thread pool.  Verified against gymnasium's env.step to ~1e-7 (float32
  observation roundoff) over 15-step open-loop sequences.
* One env step is `frame_skip` MuJoCo steps holding the same ctrl, so a
  horizon-H plan is nstep = H * frame_skip and the state is read off at every
  frame_skip-th row.
"""

import numpy as np
import mujoco
from mujoco import rollout

# control_spec bits: the wind enters the oracle as a per-substep xfrc_applied
# "control" alongside ctrl, so the C++ thread pool still does the physics.
_SPEC_CTRL = int(mujoco.mjtState.mjSTATE_CTRL)
_SPEC_CTRL_XFRC = _SPEC_CTRL | int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)


class MuJoCoOracleCEM:
    """CEM over continuous action sequences with true-simulator rollouts.

    Parameters
    ----------
    model : mjModel                 the plant (already carrying whatever
                                    non-stationary physics is currently in force)
    frame_skip : int                MuJoCo steps per env step
    reward_fn : callable            (states, acts) -> (J, H) per-step rewards.
                                    `states` is (J, H, nstate) in
                                    mjSTATE_FULLPHYSICS layout [time, qpos, qvel]
                                    and holds the POST-transition state of each
                                    step, so a reward defined on (s_{t+1}, a_t)
                                    -- as Reacher's is -- is exact here.
    """

    def __init__(self, model, frame_skip, reward_fn, act_dim,
                 horizon=15, n_cem_iters=5, n_candidates=300, elite_frac=0.1,
                 gamma=1.0, action_low=-1.0, action_high=1.0, rng=None,
                 nthread=8, wind_fn=None, wind_bodies=None, ctrl_fn=None):
        self.model = model
        self.frame_skip = int(frame_skip)
        self.reward_fn = reward_fn
        self.act_dim = act_dim
        self.horizon = horizon
        self.n_cem_iters = n_cem_iters
        self.n_candidates = n_candidates
        self.n_elite = max(1, int(elite_frac * n_candidates))
        self.gamma = gamma
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        # One MjData per worker thread; rollout infers nthread = len(data).
        self.data = [mujoco.MjData(model) for _ in range(max(1, nthread))]
        self.nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        # wind_fn(qpos) -> (fx, fy), both (B,), evaluated on the CURRENT state of
        # every candidate.  None = no wind, and then the rollout takes the fast
        # single-call path exactly as before.
        # ctrl_fn(a, h) -> tau maps a candidate's COMMANDED action at imagined
        # step h to the torque the plant actually applies.  Without it the oracle
        # would plan through an unsaturated actuator and stop being an oracle.
        self.ctrl_fn = ctrl_fn
        self.wind_fn = wind_fn
        self.wind_bodies = list(wind_bodies or [])
        self.nbody = model.nbody

        # Same exploration-width schedule as ContinuousCEMAgent: widths scale
        # with the actuator range, so at +/-1 these are 0.5 and 0.025.
        _rng_a = self.action_high - self.action_low
        self._sigma_init = _rng_a / 4.0
        self._sigma_min = _rng_a / 80.0
        self._mu = np.zeros((horizon, act_dim), dtype=np.float32)
        self._sigma = np.full((horizon, act_dim), self._sigma_init, dtype=np.float32)
        self.last_plan_return = float("nan")
        self.last_plan_return_best = float("nan")

    def reset(self):
        self._mu = np.zeros((self.horizon, self.act_dim), dtype=np.float32)
        self._sigma = np.full((self.horizon, self.act_dim), self._sigma_init,
                              dtype=np.float32)

    def notify_change(self):
        self.reset()

    def _rollout(self, state0, act_seqs):
        """(J, H, A) action sequences from a single start state -> (J, H, nstate).

        Controls are held for frame_skip MuJoCo steps each, matching
        MujocoEnv.do_simulation, and only the state at each env-step boundary is
        returned.
        """
        J, H, A = act_seqs.shape
        fs = self.frame_skip
        tau = act_seqs
        if self.ctrl_fn is not None:
            tau = np.stack([self.ctrl_fn(act_seqs[:, h, :], h) for h in range(H)],
                           axis=1)
        ctrl = np.repeat(tau, fs, axis=1)                   # (J, H*fs, A)
        if self.wind_fn is None:
            st, _ = rollout.rollout(self.model, self.data,
                                    np.tile(state0, (J, 1)), ctrl, nstep=H * fs)
            return st.reshape(J, H, fs, self.nstate)[:, :, -1, :]
        return self._rollout_windy(state0, ctrl, J, H, fs)

    def _rollout_windy(self, state0, ctrl, J, H, fs):
        """Closed-loop wind: the force depends on where each candidate's fingertip
        IS, which an open-loop rollout cannot know in advance.

        `mujoco.rollout` cannot recompute it either, so we advance ONE MuJoCo
        substep at a time and feed the field in as an xfrc_applied "control"
        (control_spec = CTRL | XFRC_APPLIED).  The physics still runs in the C++
        thread pool and the force computation is vectorised over all J candidates
        in numpy, so this costs H*fs rollout calls instead of one -- not J*H*fs
        Python-level steps.

        This mirrors ReacherWrapper._step_windy exactly: same per-substep refresh,
        same field, so the oracle is planning in the environment it is scored in.
        """
        nq = self.model.nq
        state = np.tile(state0, (J, 1))
        out = np.empty((J, H, self.nstate))
        ctrl_dim = ctrl.shape[2]
        u = np.zeros((J, 1, ctrl_dim + 6 * self.nbody))
        for h in range(H):
            for k in range(fs):
                qpos = state[:, 1:1 + nq]
                fx, fy = self.wind_fn(qpos)
                u[:, 0, :ctrl_dim] = ctrl[:, h * fs + k, :]
                u[:, 0, ctrl_dim:] = 0.0
                for bid in self.wind_bodies:
                    base = ctrl_dim + 6 * bid
                    u[:, 0, base + 0] = fx
                    u[:, 0, base + 1] = fy
                state, _ = rollout.rollout(self.model, self.data, state, u,
                                           control_spec=_SPEC_CTRL_XFRC, nstep=1)
                state = state[:, -1, :]
            out[:, h, :] = state
        return out

    def act(self, state0):
        """Plan from an mjSTATE_FULLPHYSICS vector; return the first action.

        Mirrors ContinuousCEMAgent.act step for step, including the MPC warm
        start (shift mu forward one step, re-widen sigma so the replan actually
        explores rather than coasting on a stale plan).
        """
        H, A, J = self.horizon, self.act_dim, self.n_candidates
        state0 = np.asarray(state0, dtype=np.float64).ravel()

        self._mu = np.roll(self._mu, -1, axis=0)
        self._mu[-1] = 0.0
        self._sigma = np.full((H, A), self._sigma_init, dtype=np.float32)

        disc = self.gamma ** np.arange(H)
        for _it in range(self.n_cem_iters):
            noise = self.rng.normal(size=(J, H, A)).astype(np.float32)
            candidates = np.clip(self._mu + self._sigma * noise,
                                 self.action_low, self.action_high)
            states = self._rollout(state0, candidates)
            rew = self.reward_fn(states, candidates)              # (J, H)
            scores = (rew * disc).sum(axis=1)

            elite_idx = np.argpartition(-scores, self.n_elite - 1)[:self.n_elite]
            elite_seq = candidates[elite_idx]
            self._mu = elite_seq.mean(axis=0)
            self._sigma = np.maximum(elite_seq.std(axis=0), self._sigma_min)
            self.last_plan_return = float(scores[elite_idx].mean())
            self.last_plan_return_best = float(scores.max())

        return self._mu[0].copy()


def get_state(env):
    """Read the current mjSTATE_FULLPHYSICS vector out of a gymnasium MuJoCo env."""
    u = env.unwrapped
    n = mujoco.mj_stateSize(u.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    s = np.zeros(n)
    mujoco.mj_getState(u.model, u.data, s, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return s
