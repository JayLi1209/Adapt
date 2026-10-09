"""Non-stationary Reacher environment with gear / mass / damping schedules.

Wraps gymnasium's Reacher-v5 and adds scheduled changes to the actuator gears,
the link masses, or the joint damping, mirroring env/pendulum.py's
PendulumWrapper and env/ant.py's AntWrapper so the same online
surprise -> forget -> learn loop applies unchanged.

Observations (10 dims, Reacher-v5)
----------------------------------
    0:2   cos(theta0), cos(theta1)      joint angles, angle-wrapped encoding
    2:4   sin(theta0), sin(theta1)
    4:6   target x, y                   qpos[2:4]; CONSTANT within an episode
    6:8   theta_dot0, theta_dot1        qvel[:2]
    8:10  fingertip - target  (x, y)    the reward-carrying dims

Note Reacher-v5 dropped v4's always-zero third component of the fingertip-target
vector, so this is 10 dims rather than 11.

Reward (Reacher-v5):  -||fingertip - target||  -  sum(a^2)
                      = -||obs[8:10]||  -  sum(a^2)

evaluated AFTER the transition (the env calls _get_rew post-do_simulation), so
r_t is a function of (s_{t+1}, a_t).  The learned reward head sees (s_t, a_t)
and absorbs the one-step lookahead; `reacher_reward` below is the analytic
version and carries the same pre-transition offset the Ant/Pendulum analytic
costs do -- see its docstring.

Redundancy and the state manifold
---------------------------------
The observation is a redundant encoding of a 6-dim state
(theta0, theta1, theta_dot0, theta_dot1, target_x, target_y):

  * (cos, sin) pairs must lie on the unit circle,
  * dims 8:10 are a function of dims 0:6 through the two-link forward
    kinematics.  Measured against MuJoCo over 20k random-action steps this
    identity holds to 1.4e-6 median (float32 roundoff); it degrades only while
    joint1's limit constraint is actively engaged (|theta1| > 2.9, 12% of steps),
    where MuJoCo's constraint correction puts qpos and the reported fingertip
    body position slightly out of sync -- p99 6.8e-4, max 7.3e-3, i.e. still
    ~150x below the ~0.1 scale of the distance itself.

A learned model self-fed for H steps drifts off both constraints, and the
second is the expensive one: dims 8:10 are exactly what the reward is computed
from, so their compounding error is the planner's error.  `project_reacher_obs`
re-imposes all three every imagined step -- the direct analogue of
planning.continuous_cem.project_unit_circle for the pendulum, but stronger,
because here the projection RECOVERS the reward-carrying dims rather than merely
renormalising them.
"""

import numpy as np
import gymnasium as gym

# ── Observation layout ────────────────────────────────────────────────────────
OBS_DIM = 10
ACT_DIM = 2
COS_IDX = slice(0, 2)      # cos(theta0), cos(theta1)
SIN_IDX = slice(2, 4)      # sin(theta0), sin(theta1)
TARGET_IDX = slice(4, 6)   # target (x, y) -- constant within an episode
QVEL_IDX = slice(6, 8)     # joint velocities
DIST_IDX = slice(8, 10)    # fingertip - target  (the reward-carrying dims)

# ── Geometry / limits (read off the shipped reacher.xml) ──────────────────────
L0, L1 = 0.1, 0.11         # link lengths; max reach 0.21
TARGET_RADIUS = 0.2        # env samples the goal uniformly in the disk r < 0.2
QVEL_CLIP = 50.0           # |theta_dot| ceiling for imagined rollouts (see below).
                           # Measured max |qvel| under random actions is ~30, so
                           # this bounds runaway rollouts without ever biting
                           # on-distribution.
# joint1 carries a hinge limit of +/-3 rad, but MuJoCo enforces it SOFTLY: driven
# hard the joint overshoots to |theta1| ~ 3.43.  Since the (cos, sin) encoding can
# only represent angles in (-pi, pi] anyway, and pi < 3.43, a clamp at the nominal
# limit would corrupt genuine states while a clamp wide enough to be safe could
# never bind.  The projection below therefore does NOT clamp theta1.
EP_LEN = 50                # Reacher-v5's TimeLimit: the task IS 50 steps

# ── Non-linear wind field ─────────────────────────────────────────────────────
# A STATIC-per-episode, SPATIALLY-varying force field over the workspace:
#
#   F_x(x,y) = F_max * tanh( sin(2 k_s x + C_x1) + sin(pi k_s y + C_x2) )
#   F_y(x,y) = F_max * tanh( sin(2 k_s y + C_y1) + sin(pi k_s x + C_y2) )
#
# The four phase offsets (C_x1, C_x2, C_y1, C_y2) are drawn once at reset and
# held fixed for the whole episode, so within an episode the wind is a fixed
# function of POSITION -- not a time-varying disturbance.  That distinction is
# the point: a fixed field is *learnable* by a state-conditioned adapter, which
# is exactly what the head is asked to do.
#
# WIND_KS: Reacher's workspace is a disk of radius 0.21, which is tiny next to
# the unit spatial scale the formula implies.  At k_s = 1 the sine arguments
# sweep only ~0.4 rad across the whole reachable area and the "field" degenerates
# to a near-uniform push -- which any constant-offset adapter would capture, and
# which would not test spatial structure at all.  WIND_KS = 15 makes the
# 2*k_s*x term sweep ~2*pi over the workspace diameter (2*15*0.42 = 12.6 rad),
# i.e. about one full spatial period, so the fingertip genuinely traverses
# crests and troughs on its way to the target.
WIND_KS = 15.0
WIND_FMAX = 5.0            # CLAUDE.md reacher default: intensity 0 -> 5 at ts 0

# ── Non-stationary actuator saturation ────────────────────────────────────────
# tau_i = g_t * tanh(a_i / alpha_t)
#
# Both the GAIN g_t and the KNEE alpha_t drift over the episode.  This is a
# fundamentally different perturbation from the wind: the wind is an additive,
# state-dependent force independent of the action, whereas this reshapes the
# ACTION->TORQUE map itself and is independent of state.
#
#   * alpha_t is the knee.  For |a| << alpha the map is near-linear with slope
#     g/alpha; for |a| >> alpha it saturates at g.  Shrinking alpha therefore
#     costs the planner its high-torque range while leaving small commands
#     almost untouched -- a genuinely NON-LINEAR change, not a rescaling.
#   * g_t is the ceiling.  Scaling g alone IS a pure gain change, which the
#     conjugate action head spans exactly; it is included so the two halves can
#     be separated.
#
# Drift is linear in t between the schedule's endpoints, so unlike the wind
# (frozen within an episode) this perturbation is genuinely TIME-VARYING: the
# adapter is chasing a moving target rather than fitting a fixed one.
SAT_G0, SAT_A0 = 1.0, 1e9   # identity-ish default: tanh(a/1e9)*1e9 ~= a

def action_rotate(a, theta_deg):
    """Transmission-coupling fault: tau = R(theta) @ a.

    A cable- or tendon-driven arm has a motor->joint coupling matrix; a
    re-tensioning or re-routing changes it.  R(theta) is ORTHOGONAL, so ||tau||
    is exactly ||a||: the planner keeps its entire torque budget and cannot lose
    by "getting there slower".  The whole performance gap is DIRECTION error,
    which is what makes theta a single clean difficulty knob -- expected progress
    toward the target falls roughly as cos(theta), from no-change at 0 deg to a
    full channel swap-and-flip at 90 deg.
    """
    th = np.deg2rad(float(theta_deg))
    c, s_ = np.cos(th), np.sin(th)
    R = np.array([[c, -s_], [s_, c]], dtype=np.float64)
    return (R @ np.asarray(a, dtype=np.float64).reshape(2)).astype(np.float64)


def actuator_saturate(a, g, alpha):
    """tau = g * tanh(a / alpha), elementwise.  alpha -> inf recovers tau = a*g/alpha
    only in the limit, so the shipped default uses a large alpha WITH g = alpha so
    the map is the identity when no schedule is active (see ReacherWrapper)."""
    a = np.asarray(a, dtype=np.float64)
    return g * np.tanh(a / max(float(alpha), 1e-12))


def wind_force(x, y, f_max=WIND_FMAX, k_s=WIND_KS, offs=(0.0, 0.0, 0.0, 0.0)):
    """The non-linear wind field at workspace point(s) (x, y).

    Vectorised over numpy arrays and torch tensors alike, so the same function
    serves the env (scalar, per substep), the oracle planner (batched over CEM
    candidates) and any analysis code -- there is exactly one definition of the
    field in the package.
    """
    cx1, cx2, cy1, cy2 = offs
    try:
        import torch
        if isinstance(x, torch.Tensor):
            fx = f_max * torch.tanh(torch.sin(2.0 * k_s * x + cx1)
                                    + torch.sin(np.pi * k_s * y + cx2))
            fy = f_max * torch.tanh(torch.sin(2.0 * k_s * y + cy1)
                                    + torch.sin(np.pi * k_s * x + cy2))
            return fx, fy
    except ImportError:
        pass
    fx = f_max * np.tanh(np.sin(2.0 * k_s * x + cx1) + np.sin(np.pi * k_s * y + cx2))
    fy = f_max * np.tanh(np.sin(2.0 * k_s * y + cy1) + np.sin(np.pi * k_s * x + cy2))
    return fx, fy


def fingertip_xy(theta0, theta1):
    """Two-link planar forward kinematics -> fingertip (x, y).

    Exact against MuJoCo's body('fingertip').xpos[:2] for the shipped model.
    Works on numpy arrays or torch tensors (uses the argument's own cos/sin).
    """
    try:
        import torch
        if isinstance(theta0, torch.Tensor):
            return (L0 * torch.cos(theta0) + L1 * torch.cos(theta0 + theta1),
                    L0 * torch.sin(theta0) + L1 * torch.sin(theta0 + theta1))
    except ImportError:
        pass
    return (L0 * np.cos(theta0) + L1 * np.cos(theta0 + theta1),
            L0 * np.sin(theta0) + L1 * np.sin(theta0 + theta1))


def project_reacher_obs(next_obs, vel_clip=QVEL_CLIP):
    """Project an imagined observation back onto the valid state manifold.

    Three constraints, in order:
      1. theta_i := atan2(sin_i, cos_i), then (cos, sin) re-encoded -- i.e. the
         pair is renormalised onto the unit circle, exactly as
         planning.continuous_cem.project_unit_circle does for the pendulum.
         theta itself is left unwrapped-but-equivalent: only cos/sin are used
         downstream, so the atan2 branch cut is immaterial.
      2. theta_dot clamped to +/- vel_clip, mirroring how the pendulum's
         projection clamps theta_dot to the env's max_speed: without it a
         self-fed rollout can run the velocity dims off to arbitrary magnitude,
         far outside the training range.
      3. dims 8:10 RECOMPUTED as fingertip(theta) - target rather than trusted
         from the model.  This is an identity in the real env, and it is the
         highest-value part of the projection: the reward is -||dims 8:10||, so
         any drift there is a direct planning error.

    Torch in, torch out -- called once per imagined step by the CEM planner.
    """
    import torch
    cos = next_obs[:, COS_IDX]
    sin = next_obs[:, SIN_IDX]
    theta = torch.atan2(sin, cos)
    qvel = next_obs[:, QVEL_IDX].clamp(-vel_clip, vel_clip)
    target = next_obs[:, TARGET_IDX]
    fx, fy = fingertip_xy(theta[:, 0], theta[:, 1])
    dist = torch.stack([fx - target[:, 0], fy - target[:, 1]], dim=1)
    return torch.cat([torch.cos(theta), torch.sin(theta), target, qvel, dist], dim=1)


def reacher_reward(obs, act, dist_weight=1.0, ctrl_weight=1.0):
    """Analytic Reacher-v5 reward from (obs, act), EXCLUDING nothing --

        r = -dist_weight * ||fingertip - target||  -  ctrl_weight * sum(a^2)

    but with one approximation: the env evaluates the distance term AFTER the
    transition, r_t = f(s_{t+1}, a_t), while the planner calls reward_fn with the
    PRE-transition state, so this returns -||d_t|| where the env charges
    -||d_{t+1}||.  Over an H-step rollout that is a one-step shift of the same
    sum (it drops the terminal step's distance and double-counts the initial
    one), which costs the ranking of candidates almost nothing at H >> 1 but is
    not exactly the env's return.  The same convention is used by
    env.ant.ant_reward and planning.continuous_cem.pendulum_reward.

    The LEARNED reward head has no such offset: it is trained on
    (s_t, a_t) -> r_t and absorbs the lookahead into the regression, which is why
    it is the CLAUDE.md default.  Works on torch tensors or numpy arrays.
    """
    try:
        import torch
        if isinstance(obs, torch.Tensor):
            d = obs[..., DIST_IDX].norm(dim=-1)
            return -dist_weight * d - ctrl_weight * (act ** 2).sum(dim=-1)
    except ImportError:
        pass
    obs = np.asarray(obs)
    d = np.linalg.norm(obs[..., DIST_IDX], axis=-1)
    return -dist_weight * d - ctrl_weight * np.sum(np.asarray(act) ** 2, axis=-1)


class ReacherWrapper(gym.Wrapper):
    """Reacher-v5 with scheduled actuator / mass / damping changes.

    gear_schedule / mass_schedule / damping_schedule: lists of (timestep, value)
    pairs applied at the listed step; a t==0 entry sets the value at reset.
    Every value is a MULTIPLIER on the shipped baseline, so (0, 1.0) is the
    stock robot -- the same convention as AntWrapper.

      * gear_schedule scales actuator_gear (baseline 200 on both hinges).  This
        is the direct analogue of the pendulum's mass change: the pendulum's
        dynamics enter as 3u/(m l^2), so mass x4 IS torque authority /4, and
        here gear x0.25 is the same non-stationarity in the same units.
      * mass_schedule scales the two link masses (arm inertia).
      * damping_schedule scales dof_damping (baseline 1.0 on both hinges).

    Baselines are captured before any schedule fires, so multipliers always
    compose against the shipped robot rather than against the last change.
    info["change_occurred"] is True on any step where a change fires.
    """

    def __init__(self, gear_schedule=None, mass_schedule=None,
                 damping_schedule=None, wind_schedule=None, wind_ks=WIND_KS,
                 wind_bodies=("fingertip",), sat_gain_schedule=None,
                 sat_knee_schedule=None, rot_schedule=None, **env_kwargs):
        env = gym.make("Reacher-v5", **env_kwargs)
        super().__init__(env)
        u = self.unwrapped
        self._base_gear = u.model.actuator_gear.copy()
        self._base_mass = u.model.body_mass.copy()
        self._base_damping = u.model.dof_damping.copy()

        self.gear_schedule = sorted(gear_schedule or [], key=lambda x: x[0])
        self.mass_schedule = sorted(mass_schedule or [], key=lambda x: x[0])
        self.damping_schedule = sorted(damping_schedule or [], key=lambda x: x[0])
        self.wind_schedule = sorted(wind_schedule or [], key=lambda x: x[0])
        self.wind_ks = float(wind_ks)
        # Which bodies the field pushes on.  Default: the fingertip alone, so the
        # disturbance is a pure end-effector force and the field the arm feels is
        # exactly F(fingertip_x, fingertip_y) -- the cleanest reading of "wind at
        # the spatial point (x, y)" and the one the adapter has to model.
        import mujoco as _mj
        self._wind_bids = [_mj.mj_name2id(u.model, _mj.mjtObj.mjOBJ_BODY, b)
                           for b in wind_bodies]
        self.wind_fmax = 0.0
        self.wind_offsets = (0.0, 0.0, 0.0, 0.0)
        # Actuator saturation: (t, value) knots, linearly interpolated in t, so
        # g_t and alpha_t DRIFT rather than jumping.  Empty = actuator untouched.
        self.sat_gain_schedule = sorted(sat_gain_schedule or [], key=lambda x: x[0])
        self.sat_knee_schedule = sorted(sat_knee_schedule or [], key=lambda x: x[0])
        self.sat_g = None
        self.sat_alpha = None
        # Action-space rotation: (t, theta_deg) knots, piecewise-constant via the
        # same interpolator (equal endpoints => fixed rotation from ts 0).
        self.rot_schedule = sorted(rot_schedule or [], key=lambda x: x[0])
        self.rot_theta = 0.0
        self._step_count = 0
        self._change_steps = sorted({t for t, _ in self.gear_schedule}
                                    | {t for t, _ in self.mass_schedule}
                                    | {t for t, _ in self.damping_schedule}
                                    | {t for t, _ in self.wind_schedule}
                                    | {t for t, _ in self.sat_gain_schedule}
                                    | {t for t, _ in self.sat_knee_schedule}
                                    | {t for t, _ in self.rot_schedule})

    # ── physics setters ───────────────────────────────────────────────────────
    def _set_gear(self, scale):
        self.unwrapped.model.actuator_gear[:] = self._base_gear * float(scale)

    def _set_mass(self, scale):
        # Bodies 1 and 2 are the two links; body 3/4 are the massless fingertip
        # site and the target marker, which must keep their shipped values.
        m = self.unwrapped.model
        m.body_mass[1:3] = self._base_mass[1:3] * float(scale)

    def _set_damping(self, scale):
        # dofs 0,1 are the hinges; dofs 2,3 are the target's slide joints, which
        # are unactuated and must stay at their (zero) baseline.
        m = self.unwrapped.model
        m.dof_damping[:2] = self._base_damping[:2] * float(scale)

    def reset(self, **kwargs):
        self._step_count = 0
        obs, info = self.env.reset(**kwargs)
        self._set_gear(next((v for t, v in self.gear_schedule if t == 0), 1.0))
        self._set_mass(next((v for t, v in self.mass_schedule if t == 0), 1.0))
        self._set_damping(next((v for t, v in self.damping_schedule if t == 0), 1.0))
        # Static per-episode phase offsets, drawn from the ENV's own rng so a
        # given reset(seed=k) reproduces the same field every time.
        self.wind_offsets = tuple(
            self.unwrapped.np_random.uniform(0.0, 2.0 * np.pi, size=4).tolist())
        self.wind_fmax = float(next((v for t, v in self.wind_schedule if t == 0), 0.0))
        self._apply_wind()
        return np.asarray(obs, dtype=np.float32), info

    @staticmethod
    def _interp_schedule(knots, t, default):
        """Piecewise-linear value of a (timestep, value) schedule at step t."""
        if not knots:
            return default
        if t <= knots[0][0]:
            return float(knots[0][1])
        if t >= knots[-1][0]:
            return float(knots[-1][1])
        for (t0, v0), (t1, v1) in zip(knots, knots[1:]):
            if t0 <= t <= t1:
                w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return float(v0 + w * (v1 - v0))
        return float(knots[-1][1])

    def _sat_params(self, t):
        """(g_t, alpha_t) at step t, or None if no saturation schedule is active."""
        if not self.sat_gain_schedule and not self.sat_knee_schedule:
            return None
        g = self._interp_schedule(self.sat_gain_schedule, t, 1.0)
        a = self._interp_schedule(self.sat_knee_schedule, t, 1.0)
        return g, a

    # ── wind ──────────────────────────────────────────────────────────────────
    def _apply_wind(self):
        """Write the field's force at each pushed body's CURRENT position into
        data.xfrc_applied.  Called once per MuJoCo SUBSTEP (see step)."""
        d = self.unwrapped.data
        if self.wind_fmax == 0.0:
            d.xfrc_applied[:] = 0.0
            return
        for bid in self._wind_bids:
            px, py = d.xipos[bid][0], d.xipos[bid][1]
            fx, fy = wind_force(px, py, self.wind_fmax, self.wind_ks,
                                self.wind_offsets)
            d.xfrc_applied[bid, 0] = fx
            d.xfrc_applied[bid, 1] = fy
            d.xfrc_applied[bid, 2] = 0.0

    def step(self, action):
        self._step_count += 1
        change_occurred = False
        for t, v in self.gear_schedule:
            if self._step_count == t:
                self._set_gear(v); change_occurred = True
        for t, v in self.mass_schedule:
            if self._step_count == t:
                self._set_mass(v); change_occurred = True
        for t, v in self.damping_schedule:
            if self._step_count == t:
                self._set_damping(v); change_occurred = True
        for t, v in self.wind_schedule:
            if self._step_count == t:
                self.wind_fmax = float(v); change_occurred = True
        action = np.clip(np.asarray(action, dtype=np.float32).ravel(),
                         self.action_space.low, self.action_space.high)
        # ACTUATOR SATURATION.  tau = g_t * tanh(a / alpha_t) is applied to the
        # COMMANDED action on its way into the simulator.  The reward is still
        # charged on the commanded `a` (Reacher's control cost is -||a||^2), so
        # this perturbs the DYNAMICS only and leaves the reward function fixed --
        # the same discipline the wind follows.
        # ROTATION first: it is a property of the transmission, so it maps the
        # command to a torque vector which any saturation then acts on.
        if self.rot_schedule:
            self.rot_theta = self._interp_schedule(self.rot_schedule,
                                                   self._step_count, 0.0)
            cmd_pre_rot = action
            action = action_rotate(action, self.rot_theta).astype(np.float32)
        sat = self._sat_params(self._step_count)
        if sat is not None:
            self.sat_g, self.sat_alpha = sat
            cmd = action
            action = actuator_saturate(action, self.sat_g, self.sat_alpha
                                       ).astype(np.float32)
            change_occurred = change_occurred or True
        else:
            cmd = action
        if self.rot_schedule:
            # Reward is charged on the COMMAND, as with saturation: the fault
            # perturbs dynamics only, never the reward function.  R is orthogonal
            # so ||cmd||==||tau|| and the control cost is in fact unchanged.
            cmd = cmd_pre_rot
        if self.wind_fmax == 0.0:
            # No wind: defer to the stock env, so the zero-wind path is bit-for-bit
            # the environment the model was pretrained on.
            obs, reward, terminated, truncated, info = self.env.step(action)
            if sat is not None or self.rot_schedule:
                # self.env.step charged -||tau||^2; the agent must be charged for
                # what it COMMANDED, so swap the control-cost term for -||a||^2.
                reward = float(reward) - float((cmd ** 2).sum()) \
                                       + float((action ** 2).sum())
                info["reward_ctrl"] = -float((cmd ** 2).sum())
        else:
            obs, reward, terminated, truncated, info = self._step_windy(action, cmd)
        info["change_occurred"] = change_occurred
        info["wind_fmax"] = self.wind_fmax
        if self.rot_schedule:
            info["rot_theta"] = self.rot_theta
        if sat is not None:
            info["sat_g"], info["sat_alpha"] = self.sat_g, self.sat_alpha
        return (np.asarray(obs, dtype=np.float32), float(reward),
                terminated, truncated, info)

    def _step_windy(self, action, cmd=None):
        """One env step with the wind field refreshed EVERY MuJoCo substep.

        Reacher's frame_skip is 2, and the fingertip can move ~0.13 m within a
        single env step at high joint speed -- against a field wavelength of
        ~0.4 m that is a 30% phase excursion, so holding the force constant
        across the two substeps would visibly misstate the disturbance.  We
        therefore step the simulator one substep at a time, recomputing the
        force from the CURRENT fingertip position each time.

        Everything else (observation, reward, the TimeLimit) is exactly what
        MujocoEnv.step would produce, computed from the same helpers.
        """
        import mujoco
        u = self.unwrapped
        u.do_simulation_ctrl = None                  # (unused; kept explicit)
        u.data.ctrl[:] = action
        for _ in range(u.frame_skip):
            self._apply_wind()
            mujoco.mj_step(u.model, u.data, nstep=1)
        mujoco.mj_rnePostConstraint(u.model, u.data)
        obs = u._get_obs()
        reward, reward_info = u._get_rew(action if cmd is None else cmd)
        info = dict(reward_info)
        # Reproduce the TimeLimit wrapper's truncation, which we bypassed by not
        # calling self.env.step: gym.make wraps Reacher-v5 in TimeLimit, whose
        # counter we must advance ourselves to keep episode lengths identical.
        truncated = False
        tl = self.env
        while tl is not None and not isinstance(tl, gym.wrappers.TimeLimit):
            tl = getattr(tl, "env", None)
        if tl is not None:
            tl._elapsed_steps += 1
            truncated = tl._elapsed_steps >= tl._max_episode_steps
        return obs, reward, False, truncated, info

    @property
    def change_steps(self):
        return self._change_steps

    # ── state teleporting (data collection) ───────────────────────────────────
    def set_state_obs(self, qpos, qvel):
        """Place the arm at (qpos, qvel) and return the resulting observation.

        qpos = [theta0, theta1, target_x, target_y], qvel = [dtheta0, dtheta1, 0, 0].
        Used by pretrain_reacher.py to teleport into the near-goal region
        (fingertip on the target) that random actions almost never visit.
        """
        self.unwrapped.set_state(np.asarray(qpos, dtype=np.float64),
                                 np.asarray(qvel, dtype=np.float64))
        return np.asarray(self.unwrapped._get_obs(), dtype=np.float32)


def build_reacher_env(gear_schedule=None, mass_schedule=None,
                      damping_schedule=None, wind_schedule=None,
                      sat_gain_schedule=None, sat_knee_schedule=None,
                      rot_schedule=None, **kwargs):
    """Build a Reacher env with the given schedules.

    Examples:
        build_reacher_env()                          # default dynamics
        build_reacher_env([(0, 1.0), (25, 0.25)])    # motors x0.25 at t=25
        build_reacher_env([(0, 0.25)])               # weak motors from the start
        build_reacher_env(damping_schedule=[(0, 5.0)])   # stiff, sluggish joints
        build_reacher_env(wind_schedule=[(0, 5.0)])      # CLAUDE.md reacher default
        build_reacher_env(sat_gain_schedule=[(0,1.0),(200,0.4)],
                          sat_knee_schedule=[(0,1.0),(200,0.25)])  # drifting saturation
    """
    return ReacherWrapper(gear_schedule=gear_schedule, mass_schedule=mass_schedule,
                          damping_schedule=damping_schedule,
                          wind_schedule=wind_schedule,
                          sat_gain_schedule=sat_gain_schedule,
                          sat_knee_schedule=sat_knee_schedule,
                          rot_schedule=rot_schedule, **kwargs)
