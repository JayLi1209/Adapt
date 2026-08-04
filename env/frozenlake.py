"""Scheduled non-stationary 4x4 FrozenLake (ns-gym), one-hot encoded for the BNN.

Combines the pieces the risk-averse path used from bnn_fl.py (FLOneHotWrapper,
MODIFIED_REWARDS) and mcts_drift_copy_v2.py (slip_to_dist, build_scheduled_env).
"""
import numpy as np
import gymnasium as gym

import ns_gym.wrappers as ns_wrappers
import ns_gym.schedulers as schedulers
import ns_gym.update_functions as update_functions

# Reward-on-arrival map: hole -1, goal +1, frozen/start 0.
MODIFIED_REWARDS = {"H": -1.0, "G": 1.0, "F": 0.0, "S": 0.0}

# Default slip schedule: deterministic, then slippery (intended prob 0.7) at step 1.
INTENDED_PROB_SCHEDULE = [(0, 1.0), (1, 0.7)]


class FLOneHotWrapper(gym.Wrapper):
    """Adapts NSFrozenLakeWrapper's dict/int observations to one-hot Boxes.

    Observations: the discrete state index becomes a one-hot float vector so the
    continuous BNN can consume it.  Actions: the agent acts with a 4-dim vector
    whose argmax selects the discrete action.
    """

    def __init__(self, env):
        super().__init__(env)
        self.n_states = env.nS
        self.n_actions = env.nA
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n_states,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n_actions,), dtype=np.float32
        )

    def _one_hot(self, state: int) -> np.ndarray:
        vec = np.zeros(self.n_states, dtype=np.float32)
        vec[int(state)] = 1.0
        return vec

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return self._one_hot(obs_dict["state"]), info

    def step(self, action):
        discrete_action = int(np.argmax(action))
        obs_dict, reward, terminated, truncated, info = self.env.step(discrete_action)
        return self._one_hot(obs_dict["state"]), float(reward), terminated, truncated, info


def slip_to_dist(intended_prob):
    """intended_prob p -> the full categorical [p, (1-p)/2, (1-p)/2].

    p is the probability of moving in the INTENDED direction; the remaining 1-p
    is split equally between the two PERPENDICULAR (slip) directions.
    """
    p = float(np.clip(intended_prob, 0.0, 1.0))
    side = (1.0 - p) / 2.0
    return [p, side, side]


def build_scheduled_env(intended_prob_schedule=INTENDED_PROB_SCHEDULE,
                        max_episode_steps=None):
    """4x4 ns-gym FrozenLake whose slipperiness follows `intended_prob_schedule`.

    `intended_prob_schedule` is a list of (timestep, intended_prob) pairs.
    `max_episode_steps` overrides gym's TimeLimit (default None -> the registered
    100-step limit; pass a large value to effectively disable truncation).
    """
    schedule = sorted(intended_prob_schedule, key=lambda tp: tp[0])
    dist_by_time = {int(t): slip_to_dist(p) for t, p in schedule}
    # Starting distribution = the t=0 entry if given, else deterministic.
    initial_prob_dist = dist_by_time.get(0, [1.0, 0.0, 0.0])

    make_kwargs = {} if max_episode_steps is None else {"max_episode_steps": max_episode_steps}
    base_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False,
                        **make_kwargs)
    # DistributionStepWiseUpdate installs `update_values` one per scheduler fire,
    # in ascending-time order; the per-trial reset replays the schedule identically.
    fire_times = sorted(dist_by_time)
    update_fn = update_functions.DistributionStepWiseUpdate(
        scheduler=schedulers.DiscreteScheduler(set(fire_times)),
        update_values=[dist_by_time[t] for t in fire_times],
    )
    ns_env = ns_wrappers.NSFrozenLakeWrapper(
        base_env,
        {"P": update_fn},
        change_notification=True,
        initial_prob_dist=initial_prob_dist,
        modified_rewards=MODIFIED_REWARDS,
    )
    return FLOneHotWrapper(ns_env)
