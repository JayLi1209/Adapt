"""Scheduled non-stationary 4x4 FrozenLake (ns-gym), one-hot encoded for the BNN.

This module is now a THIN COMPATIBILITY LAYER over the grid-agnostic path in
`env.gridworlds` + `grids`, which FrozenLake, CliffWalking and Bridge all share.
Every name it used to define is preserved with identical behaviour; the bodies
just delegate instead of re-implementing:

  MODIFIED_REWARDS  -- re-exported from env.gridworlds (one definition, not two)
  slip_to_dist(p)   -- FROZENLAKE_4x4.slip_dist(p).  The grid is K=3 with
                       slip_mode="perp" and one intended direction, so
                       slip_dist(p) == [p, (1-p)/2, (1-p)/2], the old formula
  FLOneHotWrapper   -- OneHotGridWrapper with the sizes read off the wrapped
                       ns_gym env, which is what this class did on its own
  build_scheduled_env -- build_env("frozenlake", ...), which constructs the
                       identical NSFrozenLakeWrapper stack

Callers that want CliffWalking or Bridge should use `env.gridworlds.build_env`
directly; it returns (env, grid) rather than just the env.
"""
from env.gridworlds import MODIFIED_REWARDS, OneHotGridWrapper, build_env
from grids import FROZENLAKE_4x4

# Default slip schedule: deterministic, then slippery (intended prob 0.7) at step 1.
INTENDED_PROB_SCHEDULE = [(0, 1.0), (1, 0.7)]

__all__ = ["MODIFIED_REWARDS", "INTENDED_PROB_SCHEDULE", "FLOneHotWrapper",
           "slip_to_dist", "build_scheduled_env"]


class FLOneHotWrapper(OneHotGridWrapper):
    """Adapts NSFrozenLakeWrapper's dict/int observations to one-hot Boxes.

    Observations: the discrete state index becomes a one-hot float vector so the
    continuous BNN can consume it.  Actions: the agent acts with a 4-dim vector
    whose argmax selects the discrete action.

    Kept as a named subclass so the one-argument constructor (sizes taken from
    the wrapped env's nS/nA) still works for existing callers.
    """

    def __init__(self, env):
        super().__init__(env, env.nS, env.nA)


def slip_to_dist(intended_prob):
    """intended_prob p -> the full categorical [p, (1-p)/2, (1-p)/2].

    p is the probability of moving in the INTENDED direction; the remaining 1-p
    is split equally between the two PERPENDICULAR (slip) directions.  Delegates
    to the shared GridSpec so FrozenLake, CliffWalking and Bridge derive their
    slip distributions from one implementation.
    """
    return FROZENLAKE_4x4.slip_dist(intended_prob)


def build_scheduled_env(intended_prob_schedule=INTENDED_PROB_SCHEDULE,
                        max_episode_steps=None):
    """4x4 ns-gym FrozenLake whose slipperiness follows `intended_prob_schedule`.

    `intended_prob_schedule` is a list of (timestep, intended_prob) pairs.
    `max_episode_steps` overrides gym's TimeLimit (default None -> the registered
    100-step limit; pass a large value to effectively disable truncation).

    Returns only the env (not the (env, grid) pair `build_env` returns), matching
    this function's original signature.
    """
    env, _grid = build_env("frozenlake", intended_prob_schedule,
                           max_episode_steps=max_episode_steps)
    return env
