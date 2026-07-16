"""Non-stationary Pendulum environment with mass / gravity schedule.

Wraps gymnasium's Pendulum-v1 and adds scheduled mass or gravity changes
to create non-stationarity.  The agent must adapt when the dynamics change.
"""

import numpy as np
import gymnasium as gym
from gymnasium.envs.classic_control.pendulum import PendulumEnv


class PendulumWrapper(gym.Wrapper):
    """Pendulum-v1 with scheduled mass / gravity changes for non-stationarity testing.

    Observations: [cos(theta), sin(theta), theta_dot] (3-dim, float32)
    Actions: [torque] (1-dim, float32, range [-2.0, 2.0])

    mass_schedule: list of (timestep, mass) pairs
    gravity_schedule: list of (timestep, g) pairs

    At each timestep the corresponding parameter is changed.
    info["change_occurred"] = True on change steps.
    """

    def __init__(self, mass_schedule=None, gravity_schedule=None):
        env = gym.make("Pendulum-v1")
        super().__init__(env)
        self.mass_schedule = sorted(mass_schedule or [], key=lambda x: x[0])
        self.gravity_schedule = sorted(gravity_schedule or [], key=lambda x: x[0])
        self._base_mass = float(self.unwrapped.m)
        self._base_g = float(self.unwrapped.g)
        self._step_count = 0
        all_changes = set()
        for t, _ in self.mass_schedule:
            all_changes.add(t)
        for t, _ in self.gravity_schedule:
            all_changes.add(t)
        self._change_steps = sorted(all_changes)

    def reset(self, **kwargs):
        self._step_count = 0
        obs, info = self.unwrapped.reset(**kwargs)
        # Apply t=0 changes immediately
        for t, mass in self.mass_schedule:
            if t == 0:
                self.unwrapped.m = mass
        for t, g in self.gravity_schedule:
            if t == 0:
                self.unwrapped.g = g
        return obs.astype(np.float32), info

    def step(self, action):
        self._step_count += 1
        change_occurred = False
        for t, mass in self.mass_schedule:
            if self._step_count == t:
                self.unwrapped.m = mass
                change_occurred = True
        for t, g in self.gravity_schedule:
            if self._step_count == t:
                self.unwrapped.g = g
                change_occurred = True
        action = np.clip(np.asarray(action, dtype=np.float32).ravel(), -2.0, 2.0)
        obs, reward, terminated, truncated, info = self.unwrapped.step(action)
        info["change_occurred"] = change_occurred
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    @property
    def change_steps(self):
        return self._change_steps


def build_pendulum_env(mass_schedule=None, gravity_schedule=None):
    """Build a Pendulum environment with the given schedules.

    Example:
        env = build_pendulum_env(mass_schedule=[(0, 4.0)])
        env = build_pendulum_env(gravity_schedule=[(0, 20.0)])
    """
    return PendulumWrapper(
        mass_schedule=(list(mass_schedule) if mass_schedule else []),
        gravity_schedule=(list(gravity_schedule) if gravity_schedule else []),
    )
