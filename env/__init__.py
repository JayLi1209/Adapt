from env.frozenlake import (
    FLOneHotWrapper, build_scheduled_env, slip_to_dist, MODIFIED_REWARDS,
)
from env.pendulum import PendulumWrapper, build_pendulum_env

__all__ = [
    "FLOneHotWrapper", "build_scheduled_env", "slip_to_dist", "MODIFIED_REWARDS",
    "PendulumWrapper", "build_pendulum_env",
    "LunarLanderWrapper", "build_lunarlander_env", "heuristic_action",
    "NO_DELTA_DIMS", "OBS_LOW", "OBS_HIGH", "ACTION_NAMES",
    "InvertedPendulumWrapper", "build_inverted_pendulum_env",
    "ANGLE_LIMIT", "MAX_EPISODE_STEPS", "OBS_NAMES",
]

_LL_NAMES = ("LunarLanderWrapper", "build_lunarlander_env", "heuristic_action",
             "NO_DELTA_DIMS", "OBS_LOW", "OBS_HIGH", "ACTION_NAMES")
_IP_NAMES = ("InvertedPendulumWrapper", "build_inverted_pendulum_env",
             "ANGLE_LIMIT", "MAX_EPISODE_STEPS", "OBS_NAMES")


def __getattr__(name):
    # LunarLander needs Box2D and InvertedPendulum needs MuJoCo; import both
    # lazily so `from env import ...` still works for FrozenLake/Pendulum users
    # who have installed neither.  OBS_LOW/OBS_HIGH are defined by both modules;
    # `from env import OBS_LOW` resolves to LunarLander's for backward
    # compatibility -- import env.inverted_pendulum directly for this env's.
    if name in _LL_NAMES:
        import env.lunarlander as _ll
        return getattr(_ll, name)
    if name in _IP_NAMES:
        import env.inverted_pendulum as _ip
        return getattr(_ip, name)
    raise AttributeError(name)
