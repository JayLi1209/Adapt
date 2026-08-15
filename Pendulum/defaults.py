"""The five constants the experiment inherits from the parent repo's
run_pendulum_default.py, so this package does not need that module (which pulls
in the whole FrozenLake/Dirichlet stack).

Values are the CLAUDE.md pendulum default setting.  GAMMA is derived from the
planner rather than duplicated, so the two cannot drift apart.
"""
from planning.continuous_cem import GAMMA          # 0.99

DEFAULT_MASS, DEFAULT_GRAV = 1.0, 10.0
TARGET_MASS = 4.0
CHANGE_STEP = 0        # the shifted dynamics are live from the very first step
TRIAL_LEN = 200        # "stop at episode 200"; the runs here override to 500
K_FORGET = 5
