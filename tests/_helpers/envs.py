"""`make_env`: one Cs2Env with every knob at its default, for env-level tests.

#321 moved it here from `cs2rl.train.envs`: nothing in src/ or scripts/ called it, and
training builds its envs through `cs2rl.train.envs.build_env_factory`. It is the one
caller of the env factory's `external` role (`cs2rl.env.factory.build_external_env`).
"""

from cs2rl.env.factory import build_external_env


def make_env(team_spirit=None, map_data=None):
    """A default-knob Cs2Env: `team_spirit` None, `map_data` None (dust2).

    The optional defaults stay HERE rather than in `build_external_env`: that builder
    requires both arguments so a caller that forgets to forward one gets a TypeError
    instead of a dust2 env.
    """
    return build_external_env(team_spirit=team_spirit, map_data=map_data)
