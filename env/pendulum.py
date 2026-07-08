"""Non-stationary Pendulum environment with mass schedule.

Wraps gymnasium's Pendulum-v1 and adds scheduled mass changes to create
non-stationarity. The agent must adapt when the pendulum's mass changes.
"""

import numpy as np
import gymnasium as gym
from gymnasium.envs.classic_control.pendulum import PendulumEnv


class PendulumWrapper(gym.Wrapper):
    """Pendulum-v1 with scheduled mass changes for non-stationarity testing.

    Observations: [cos(theta), sin(theta), theta_dot] (3-dim, float32)
    Actions: [torque] (1-dim, float32, range [-2.0, 2.0])

    mass_schedule: list of (timestep, mass) pairs. At each timestep in the
    schedule, the pendulum's mass changes to the specified value. The
    wrapper returns info["change_occurred"] = True on change steps.
    """

    def __init__(self, mass_schedule=None):
        env = gym.make("Pendulum-v1")
        super().__init__(env)
        self.mass_schedule = sorted(mass_schedule or [], key=lambda x: x[0])
        self._base_mass = float(self.unwrapped.m)
        self._step_count = 0
        self._change_steps = [t for t, _ in self.mass_schedule]

    def reset(self, **kwargs):
        self._step_count = 0
        obs, info = self.unwrapped.reset(**kwargs)
        if self.mass_schedule:
            t0_mass = next((m for t, m in self.mass_schedule if t == 0), self._base_mass)
            self.unwrapped.m = t0_mass
        return obs.astype(np.float32), info

    def step(self, action):
        self._step_count += 1
        change_occurred = False
        for t, mass in self.mass_schedule:
            if self._step_count == t:
                self.unwrapped.m = mass
                change_occurred = True
                break
        action = np.clip(np.asarray(action, dtype=np.float32).ravel(), -2.0, 2.0)
        obs, reward, terminated, truncated, info = self.unwrapped.step(action)
        info["change_occurred"] = change_occurred
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    @property
    def change_steps(self):
        return self._change_steps


def build_pendulum_env(mass_schedule=None):
    """Build a Pendulum environment with the given mass schedule.

    Example:
        schedule = [(0, 1.0), (80, 3.0)]  # mass 1.0 then changes to 3.0 at t=80
        env = build_pendulum_env(schedule)
    """
    return PendulumWrapper(mass_schedule=([(0, 1.0)] if mass_schedule is None else list(mass_schedule)))
