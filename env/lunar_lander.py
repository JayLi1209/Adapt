"""Non-stationary Lunar Lander environment with wind schedule.

Wraps gymnasium's LunarLander-v3 and adds scheduled wind changes.
Non-stationarity: wind_power changes → the lander gets pushed sideways.

Reference: PA-MCTS paper uses wind ∈ {0, 10, 15, 20}.
"""

import numpy as np
import gymnasium as gym


class LunarLanderWrapper(gym.Wrapper):
    """LunarLander-v3 with scheduled wind changes.

    Observations: 8-dim [x, y, vx, vy, angle, angular_vel, left_contact, right_contact]
    Actions: 4 discrete {nothing, left engine, main engine, right engine}

    wind_schedule: list of (timestep, wind_power) pairs.
    """

    def __init__(self, wind_schedule=None):
        env = gym.make("LunarLander-v3")
        super().__init__(env)
        self.wind_schedule = sorted(wind_schedule or [], key=lambda x: x[0])
        self._base_wind = float(self.unwrapped.wind_power)
        self._step_count = 0
        self._change_steps = [t for t, _ in self.wind_schedule]

    def reset(self, **kwargs):
        self._step_count = 0
        obs, info = self.unwrapped.reset(**kwargs)
        for t, wind in self.wind_schedule:
            if t == 0:
                self.unwrapped.wind_power = wind
        return obs.astype(np.float32), info

    def step(self, action):
        self._step_count += 1
        change_occurred = False
        for t, wind in self.wind_schedule:
            if self._step_count == t:
                self.unwrapped.wind_power = wind
                change_occurred = True
                break
        obs, reward, terminated, truncated, info = self.unwrapped.step(action)
        info["change_occurred"] = change_occurred
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    @property
    def change_steps(self):
        return self._change_steps


def build_lunar_lander_env(wind_schedule=None):
    """Build a Lunar Lander environment with wind schedule.

    Example: env = build_lunar_lander_env([(0, 0.0), (50, 15.0)])
    """
    return LunarLanderWrapper(wind_schedule=list(wind_schedule) if wind_schedule else [])
