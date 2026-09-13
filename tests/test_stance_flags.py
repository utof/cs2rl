"""W5 (#156): `crouch_enabled` / `jump_enabled` are SIM-LEVEL invariants, not masks.

Before W5 both flags only fed `compute_masks` (cs2_env.h), so they constrained
what a *policy sampling under the mask* could pick and nothing else. Any caller
that hands raw actions to `env.step()` — scripted bots (#152), BC replay, a
buggy trainer, these tests — could still crouch and jump with the flag at 0.
W5 moves the enforcement into `process_movement` (cs2_movement.h) so a disabled
stance is unreachable through every path. The masks stay (a masked policy should
not waste probability mass on a bin the sim will drop).

Two assertion families per flag, and they are NOT redundant — each one is blind
to a different mis-placement of the guard, which is why the spec names the
insertion point down to the line:

  * **state/physics** (`vz` for jump, `obs[13]` + the 0.34x speed factor for
    crouch) — catches a guard placed AFTER the state write, i.e. one that
    silences the histogram while the agent still crouches/jumps.
  * **histogram** (`action_jump_1` / `action_crouch_1` == 0) — catches a guard
    placed after the `count_action` feed. That variant stops the physics but
    leaves the histogram counting ATTEMPTED inputs. Gate readers treat the
    action histograms as ground truth of sim behaviour (rung1a pre-flight 4b
    reads move-bin fractions), so attempted-but-ignored presses would poison
    the reading with no other test noticing. A vel_z-only acceptance passes
    against that weakening — hence these assertions exist.

The histogram values are read through `_build_terminal_info()`, the real
producer of the `action_jump_*` / `action_crouch_*` summary keys
(cs2_env.py) — not off the C counter array directly — so the assertion is
about the emitted metric key, not an internal that could be re-keyed later.

Signal choices (spec §2 W5):
  * jump PRIMARY signal is `vz`, not `is_airborne`: `is_airborne` has three
    writers including fall detection, which any z relief can trip, so it is
    only ever a secondary assert here.
  * crouch signal is `obs[13]` (the is_crouching observation), NOT height:
    crouching touches neither z nor vz in this sim, so a height assertion
    would pass with and without the guard.
"""
import math

import numpy as np
import pytest

N_AGENTS, ACTION_DIM, AIM_DIM = 10, 7, 2
HEAD_MOVE = 0                          # cs2_types.h enum
HEAD_CROUCH = 5
HEAD_JUMP = 6
OBS_IS_CROUCHING = 13                  # cs2_observations.h — obs[13] = a->is_crouching
MOVE_FORWARD = 1                       # facing-local move bin, any non-zero dir works

_SEED = 1


def _forced_step(map_data,
                 *,
                 crouch_enabled=1,
                 jump_enabled=1,
                 crouch_act=0,
                 jump_act=0,
                 move_dir=0):
    """Reset a fresh env, press the given stance inputs on EVERY agent for one
    tick, and return a plain-Python snapshot.

    `env.step()` takes raw actions and does not apply `env.masks`, which is the
    whole point: these tests exercise the path a scripted bot uses, where the
    mask is advisory. The snapshot is materialised into floats/ints and the env
    is closed inside the helper, so no caller can read through `game.agents` or
    the obs view after the C arena is freed.
    """
    from c_env.cs2_env import make_env
    from env_config import EnvConfig
    env = make_env(map_data=map_data,
                   config=EnvConfig(crouch_enabled=crouch_enabled, jump_enabled=jump_enabled),
                   seed=_SEED)
    try:
        env.reset()
        agents = env._c_env.game.agents
        start = [(agents[i].x, agents[i].y) for i in range(N_AGENTS)]
        act = np.zeros((N_AGENTS, ACTION_DIM), np.int32)
        cont = np.zeros((N_AGENTS, AIM_DIM), np.float32)
        act[:, HEAD_MOVE] = move_dir
        act[:, HEAD_CROUCH] = crouch_act
        act[:, HEAD_JUMP] = jump_act
        obs = env.step(act, cont)[0]
        # `disp` is ground speed times dt, but recovered by subtracting two
        # ~1e3-magnitude float32 coordinates, which quantises the ~1.6u result at
        # ~6e-5. That is fine for an exact same-vs-same comparison and too coarse
        # to pin a RATIO against, so ratio assertions use `speed`, which carries
        # the 0.34x crouch factor exactly.
        disp = [
            math.hypot(agents[i].x - start[i][0], agents[i].y - start[i][1])
            for i in range(N_AGENTS)
        ]
        speed = [math.hypot(agents[i].vx, agents[i].vy) for i in range(N_AGENTS)]
        return {
            "vz": [float(agents[i].vz) for i in range(N_AGENTS)],
            "airborne": [int(agents[i].is_airborne) for i in range(N_AGENTS)],
            "is_crouching": [int(agents[i].is_crouching) for i in range(N_AGENTS)],
            "obs13": [float(obs[i, OBS_IS_CROUCHING]) for i in range(N_AGENTS)],
            "alive": [int(agents[i].alive) for i in range(N_AGENTS)],
            "disp": disp,
            "speed": speed,
            "summary": dict(env._build_terminal_info()),
        }
    finally:
        env.close()


# ── jump ────────────────────────────────────────────────────────────────────


def test_jump_enabled_control_actually_jumps(simple_map):
    """Control: without this, a guard that disabled jumping unconditionally
    would pass every assertion in the test below."""
    s = _forced_step(simple_map, jump_enabled=1, jump_act=1)
    assert all(a == 1 for a in s["alive"]), "fixture assumption: all agents spawn alive"
    assert all(vz > 0.0 for vz in s["vz"]), s["vz"]
    assert all(s["airborne"]), s["airborne"]
    assert s["summary"]["action_jump_1"] == N_AGENTS
    assert s["summary"]["action_jump_0"] == 0


def test_jump_ignored_when_jump_disabled(simple_map):
    """jump_enabled=0 + a forced jump on every agent => no jump, in the sim."""
    s = _forced_step(simple_map, jump_enabled=0, jump_act=1)
    # PRIMARY: no jump impulse reached vz.
    assert all(vz == 0.0 for vz in s["vz"]), s["vz"]
    # SECONDARY only (three writers, see module docstring).
    assert not any(s["airborne"]), s["airborne"]
    # Histograms must report EFFECTIVE actions, so the guard has to sit before
    # count_action, not merely before the vz physics. This assertion is the only
    # thing in the suite that can tell those two placements apart.
    assert s["summary"]["action_jump_1"] == 0, s["summary"]["action_jump_1"]
    assert s["summary"]["action_jump_0"] == N_AGENTS


# ── crouch ──────────────────────────────────────────────────────────────────


def test_crouch_enabled_control_actually_crouches(simple_map):
    """Control for the guarded case below, and the source of the 0.34x number:
    `get_move_speed` (cs2_weapons.h) scales by 0.34 while `is_crouching`."""
    s = _forced_step(simple_map, crouch_enabled=1, crouch_act=1, move_dir=MOVE_FORWARD)
    assert all(c == 1.0 for c in s["obs13"]), s["obs13"]
    assert all(s["is_crouching"]), s["is_crouching"]
    assert s["summary"]["action_crouch_1"] == N_AGENTS
    assert s["summary"]["action_crouch_0"] == 0

    walking = _forced_step(simple_map, crouch_enabled=1, crouch_act=0, move_dir=MOVE_FORWARD)
    assert all(v > 0.0 for v in walking["speed"]), walking["speed"]
    for crouched, upright in zip(s["speed"], walking["speed"], strict=True):
        # rel tolerance, not exact: the sim runs float32 and the ratio is
        # reconstructed in float64 here. 1e-6 still separates 0.34x from 1.0x
        # by six orders of magnitude.
        assert crouched == pytest.approx(0.34 * upright, rel=1e-6), (crouched, upright)


def test_crouch_ignored_when_crouch_disabled(simple_map):
    """crouch_enabled=0 + a forced crouch on every agent => the agent stays
    upright AND moves at full speed AND the crouch histogram stays empty."""
    s = _forced_step(simple_map, crouch_enabled=0, crouch_act=1, move_dir=MOVE_FORWARD)
    # PRIMARY: the is_crouching observation the policy actually sees.
    assert all(c == 0.0 for c in s["obs13"]), s["obs13"]
    assert not any(s["is_crouching"]), s["is_crouching"]
    # Stronger discriminator: per-tick displacement must match an agent that
    # never pressed crouch, i.e. the 0.34x multiplier never fired. Same seed and
    # same code path on both sides, so this is an exact comparison.
    upright = _forced_step(simple_map, crouch_enabled=1, crouch_act=0, move_dir=MOVE_FORWARD)
    assert all(d > 0.0 for d in upright["disp"]), upright["disp"]
    assert s["disp"] == upright["disp"], (s["disp"], upright["disp"])
    assert s["speed"] == upright["speed"], (s["speed"], upright["speed"])
    # Same insertion-point argument as the jump case: a guard that only fixed
    # is_crouching (leaving crouch_act=1 to reach count_action) passes every
    # assertion above and fails this one.
    assert s["summary"]["action_crouch_1"] == 0, s["summary"]["action_crouch_1"]
    assert s["summary"]["action_crouch_0"] == N_AGENTS
