"""Non-stationary Pendulum environment with mass AND gravity schedules.

Wraps gymnasium's Pendulum-v1 and adds scheduled changes to the pendulum mass
and/or gravity to create non-stationarity. The agent (with a BNN pretrained on
the DEFAULT dynamics) must adapt online when mass/gravity shift.
"""

import numpy as np
import gymnasium as gym
from gymnasium.envs.classic_control.pendulum import PendulumEnv  # noqa: F401


class PendulumWrapper(gym.Wrapper):
    """Pendulum-v1 with scheduled mass/gravity changes for non-stationarity.

    Observations: [cos(theta), sin(theta), theta_dot] (3-dim, float32)
    Actions: [torque] (1-dim, float32, range [-2.0, 2.0])

    mass_schedule / gravity_schedule: lists of (timestep, value) pairs. At each
    listed timestep the pendulum's mass (or gravity) is set to the given value.
    A t==0 entry sets the value at reset. info["change_occurred"] is True on any
    step where a mass or gravity change fires.
    """

    def __init__(self, mass_schedule=None, gravity_schedule=None, max_torque=None):
        env = gym.make("Pendulum-v1")
        super().__init__(env)
        # Actuator limit.  None keeps Pendulum-v1's shipped 2.0.  Raising it
        # UNCAPS the controller: at mass m the control term is 3u/(m l^2) while
        # gravity peaks at 3g/(2l)=15, so u_max=20 gives 0.75*20=15 at m=4 --
        # exactly enough torque to hold the pole statically at ANY angle.  The
        # reward's -0.001 u^2 term still penalises large torque, so the optimum
        # remains finite rather than running off to the bound.
        if max_torque is not None:
            self.unwrapped.max_torque = float(max_torque)
            self.unwrapped.action_space = gym.spaces.Box(
                low=-float(max_torque), high=float(max_torque), shape=(1,),
                dtype=np.float32)
            self.action_space = self.unwrapped.action_space
        self._max_torque = float(self.unwrapped.max_torque)
        self.mass_schedule = sorted(mass_schedule or [], key=lambda x: x[0])
        self.gravity_schedule = sorted(gravity_schedule or [], key=lambda x: x[0])
        self._base_mass = float(self.unwrapped.m)
        self._base_grav = float(self.unwrapped.g)
        self._step_count = 0
        self._change_steps = sorted({t for t, _ in self.mass_schedule}
                                    | {t for t, _ in self.gravity_schedule})

    def reset(self, **kwargs):
        self._step_count = 0
        obs, info = self.unwrapped.reset(**kwargs)
        # t==0 entries set the starting mass/gravity (else keep the base value).
        self.unwrapped.m = next((m for t, m in self.mass_schedule if t == 0),
                                self._base_mass)
        self.unwrapped.g = next((g for t, g in self.gravity_schedule if t == 0),
                                self._base_grav)
        return obs.astype(np.float32), info

    def step(self, action):
        self._step_count += 1
        change_occurred = False
        for t, mass in self.mass_schedule:
            if self._step_count == t:
                self.unwrapped.m = mass
                change_occurred = True
        for t, grav in self.gravity_schedule:
            if self._step_count == t:
                self.unwrapped.g = grav
                change_occurred = True
        action = np.clip(np.asarray(action, dtype=np.float32).ravel(),
                         -self._max_torque, self._max_torque)
        obs, reward, terminated, truncated, info = self.unwrapped.step(action)
        info["change_occurred"] = change_occurred
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    @property
    def change_steps(self):
        return self._change_steps


def build_pendulum_env(mass_schedule=None, gravity_schedule=None, max_torque=None):
    """Build a Pendulum env with the given mass and/or gravity schedules.

    Examples:
        build_pendulum_env([(0, 1.0), (100, 3.0)])                 # mass 1->3 at t=100
        build_pendulum_env([(0, 1.0)], [(0, 10.0), (100, 18.0)])   # gravity 10->18 at t=100
    """
    return PendulumWrapper(
        mass_schedule=([(0, 1.0)] if mass_schedule is None else list(mass_schedule)),
        gravity_schedule=(list(gravity_schedule) if gravity_schedule else None),
        max_torque=max_torque,
    )
