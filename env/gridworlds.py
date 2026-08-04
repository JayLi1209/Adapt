"""Modular non-stationary grid-world builders (FrozenLake / CliffWalking).

One factory, `build_env(name, schedule, ...)`, returns a one-hot-wrapped ns_gym
env for any registered grid in `grids.REGISTRY`.  The existing
`env.frozenlake.build_scheduled_env` is left untouched -- this module is additive,
and `build_env("frozenlake", ...)` reproduces it.

Non-stationarity is expressed the same way for every grid: a schedule of
(timestep, intended_prob) pairs.  `intended_prob` p is the probability of moving
in the INTENDED direction; the remaining 1-p is split equally over the other
K-1 directions.  So for

  FrozenLake (K=3): p -> [p, (1-p)/2, (1-p)/2]
  CliffWalking (K=4): p -> [p, (1-p)/3, (1-p)/3, (1-p)/3]

Reward convention is shared (`MODIFIED_REWARDS`): goal +1, hole/cliff -1,
frozen 0.  Reported returns still score holes as 0 (see the runners), so
return == goal rate.

CliffWalking note: ns_gym always teleports a cliff landing back to the start
cell; `terminal_cliff` only decides whether the episode also ENDS there.  We
default terminal_cliff=True so the cliff behaves like a FrozenLake hole, which
keeps the hole=0 return convention meaningful.
"""
import numpy as np
import gymnasium as gym

import ns_gym.wrappers as ns_wrappers
import ns_gym.schedulers as schedulers
import ns_gym.update_functions as update_functions

from grids import get_grid

# Shared reward-on-arrival map: hole/cliff -1, goal +1, frozen/start 0.
MODIFIED_REWARDS = {"H": -1.0, "G": 1.0, "F": 0.0, "S": 0.0}


class OneHotGridWrapper(gym.Wrapper):
    """Dict/int observations -> one-hot Box; 4-dim action vector -> argmax.

    The grid-agnostic generalisation of env.frozenlake.FLOneHotWrapper.
    """

    def __init__(self, env, n_states, n_actions):
        super().__init__(env)
        self.n_states = n_states
        self.n_actions = n_actions
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(n_states,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(n_actions,), dtype=np.float32)

    def _one_hot(self, state):
        v = np.zeros(self.n_states, dtype=np.float32)
        v[int(state)] = 1.0
        return v

    @staticmethod
    def _state_of(obs):
        # ns_gym returns either a dict with "state" or a bare int.
        return obs["state"] if isinstance(obs, dict) else obs

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self._one_hot(self._state_of(obs)), info

    def step(self, action):
        a = int(np.argmax(action))
        obs, reward, terminated, truncated, info = self.env.step(a)
        return (self._one_hot(self._state_of(obs)), float(reward),
                terminated, truncated, info)


class TabularGridEnv(gym.Env):
    """Self-contained one-hot grid env driven directly by a GridSpec.

    Used for grids ns_gym cannot provide (Bridge: ns_gym.envs.Bridge is broken
    and, even when patched, samples PERPENDICULAR slips rather than the paper's
    intended-vs-opposite dynamics).  Transitions come from grid.slip_dist(p) and
    grid.move, so the realized dynamics are EXACTLY the paper's for that grid.

    Non-stationarity: `dist_by_time` maps timestep -> K-vector; the active slip
    distribution switches at those steps (an internal step counter, reset each
    episode).  Reward-on-arrival = MODIFIED_REWARDS; goal/hole are terminal.
    """

    def __init__(self, grid, dist_by_time, max_episode_steps=None):
        super().__init__()
        self.grid = grid
        self.dist_by_time = {int(t): np.asarray(d, float) for t, d in dist_by_time.items()}
        self.max_episode_steps = max_episode_steps
        self.flat = grid.flat_desc
        self.observation_space = gym.spaces.Box(0.0, 1.0, (grid.n_states,), np.float32)
        self.action_space = gym.spaces.Box(0.0, 1.0, (grid.n_actions,), np.float32)
        self._rng = np.random.default_rng(0)
        self._cur = self.dist_by_time.get(0, np.asarray(grid.slip_dist(1.0)))
        self.s = grid.start_state
        self.t = 0

    def _one_hot(self, s):
        v = np.zeros(self.grid.n_states, dtype=np.float32); v[int(s)] = 1.0
        return v

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.s = self.grid.start_state
        self.t = 0
        self._cur = self.dist_by_time.get(0, self._cur)
        return self._one_hot(self.s), {}

    def step(self, action):
        if self.t in self.dist_by_time:              # scheduled change
            self._cur = self.dist_by_time[self.t]
        a = int(np.argmax(action))
        k = int(self._rng.choice(self.grid.k_dir, p=self._cur))  # realized direction
        d_action = self.grid.dir_actions(a)[k]
        s2 = self.grid.move(self.s, d_action)
        ch = self.flat[s2]
        reward = MODIFIED_REWARDS.get(ch, 0.0)
        terminated = ch in ("G", "H")
        self.s = s2
        self.t += 1
        truncated = (self.max_episode_steps is not None
                     and self.t >= self.max_episode_steps)
        return self._one_hot(s2), float(reward), terminated, truncated, {}


def build_env(name, intended_prob_schedule, max_episode_steps=None,
              terminal_cliff=True):
    """Non-stationary grid world whose slip follows `intended_prob_schedule`.

    name  : key in grids.REGISTRY ("frozenlake" | "cliffwalking")
    schedule : list of (timestep, intended_prob) pairs; the t=0 entry (if any)
               also sets the initial distribution.
    max_episode_steps : override gym's TimeLimit (None -> the registered default;
               pass a large value to effectively disable truncation).
    """
    grid = get_grid(name)
    schedule = sorted(intended_prob_schedule, key=lambda tp: tp[0])
    dist_by_time = {int(t): grid.slip_dist(p) for t, p in schedule}
    initial = dist_by_time.get(0, grid.slip_dist(1.0))

    mk = {} if max_episode_steps is None else {"max_episode_steps": max_episode_steps}
    fire_times = sorted(dist_by_time)

    def _pad4(d):
        # ns_gym CliffWalking has a 4th (opposite) outcome slot; the paper's slip
        # is perpendicular-only, so append probability 0.  Order matches its
        # b_actions = [intended, perp+, perp-, opposite].
        return list(d) + [0.0]

    if name == "frozenlake":
        update_fn = update_functions.DistributionStepWiseUpdate(
            scheduler=schedulers.DiscreteScheduler(set(fire_times)),
            update_values=[dist_by_time[t] for t in fire_times])
        base = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, **mk)
        ns_env = ns_wrappers.NSFrozenLakeWrapper(
            base, {"P": update_fn}, change_notification=True,
            initial_prob_dist=initial, modified_rewards=MODIFIED_REWARDS)
    elif name == "cliffwalking":
        update_fn = update_functions.DistributionStepWiseUpdate(
            scheduler=schedulers.DiscreteScheduler(set(fire_times)),
            update_values=[_pad4(dist_by_time[t]) for t in fire_times])
        base = gym.make("CliffWalking-v1", **mk)
        ns_env = ns_wrappers.NSCliffWalkingWrapper(
            base, {"P": update_fn}, change_notification=True,
            initial_prob_dist=_pad4(initial), modified_rewards=MODIFIED_REWARDS,
            terminal_cliff=terminal_cliff)
    elif name == "bridge":
        # ns_gym's Bridge is broken and non-faithful; use the self-contained
        # tabular env, whose dynamics are exactly grid.slip_dist (intended p,
        # opposite 1-p).  Already one-hot, so no OneHotGridWrapper needed.
        return TabularGridEnv(grid, dist_by_time, max_episode_steps), grid
    else:
        raise ValueError(f"no builder for grid {name!r}")

    return OneHotGridWrapper(ns_env, grid.n_states, grid.n_actions), grid
