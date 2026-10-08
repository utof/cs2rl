"""Arrange a Cs2Env's live state for a test through valid moves, not field by field (#170).

A test that needs a scenario (two agents facing off, a dead carrier, a ready rifle) used to
write `env._c_env.game` field by field and keep the fields consistent by hand: area_idx with
the position, z with the floor, hp with alive, a clip with the magazine. A slip built a
state the sim cannot reach, and the test then asserted on it. Measured at 977f1b8 (#170):
9 tests stepped an area_idx that disagreed with their agent's position, 5 an over-full
rifle clip and 2 a dead agent with hp left.

The helpers own those couplings, and every rule comes from the sim, never from a Python copy:
  * area_idx and the floor z: `binding.ground_at` (cs2_movement.h's `_area_at` and
    `_surface_z`, the rules movement and the ground-snap use);
  * clip limits: `binding.weapon_defs` (cs2_weapons.h's WEAPON_DEFS);
  * the bomb carrier: `Cs2Env.give_bomb` (#164), or `setup_bomb_carrier` in
    cs2rl.eval.scripted_expert when the role bit and the knife go with it.
`assert_state_consistent` checks over a whole state: area vs position, alive vs hp, team,
participating and the parked state, weapon slot and ammo limits (not z or the bomb carrier);
call it after a hand-built setup, before the step under test.

What stays a raw write: a field no invariant here couples (e.g. facing, has_kit, armor, a
live agent's hp above 0, round_over and winner, the StaticData clocks); a planted bomb through
the BombState table, which `binding.step` checks; and a deliberately impossible state whose
rejection a test pins.

PITFALL: a helper that does the sim's job hides the sim's behaviour from the test. A test of
the ground-snap must place an unsettled z (`place_agent(..., z=...)`) and let the step under
test settle it; with the default the helper has already written the answer.
"""

from cs2rl.env.c import binding
from cs2rl.env.nav import N_AGENTS, TEAM_SIZE
from cs2rl.spec.action import ACTION_DIM, AIM_DIM

# cs2_types.h INVALID_AREA_IDX: the area_idx of a parked slot (env_reset) and of "nowhere".
INVALID_AREA_IDX = -1


def zero_actions(n_agents=N_AGENTS):
    """All-zero (discrete int32 (n, ACTION_DIM), continuous float32 (n, AIM_DIM)) buffers.

    Zero discrete actions neither move nor shoot; a zero continuous row turns by 0 and, under
    v1c's absolute pitch, sets pitch 0. Fresh arrays on every call: callers fill them in.
    """
    import numpy as np
    return (np.zeros((n_agents, ACTION_DIM),
                     dtype=np.int32), np.zeros((n_agents, AIM_DIM), dtype=np.float32))


def ground_at(env, x, y):
    """(area_idx, floor z) the sim resolves for (x, y); (INVALID_AREA_IDX, None) off the mesh."""
    return binding.ground_at(env._capsule, float(x), float(y))


def _agent(env, idx):
    a = env._c_env.game.agents[idx]
    if not (a.alive and a.participating):
        raise ValueError(f"agent {idx} is dead or parked: a scenario moves only live agents")
    return a


def place_agent(env, idx, x, y, *, z=None, airborne=False):
    """Move live agent `idx` to (x, y), at rest, with area_idx from the sim's own rule.

    z: None stands it on the floor (what the grounded ground-snap writes). A z given for a
    grounded agent is left for the next step to snap; pass one only when the snap is under
    test. airborne=True needs a z above the floor and leaves the agent at the top of a fall
    (vz 0), which the next step integrates.
    Raises ValueError off the mesh, for a dead or parked agent, and for an airborne z that
    is not above the floor. Returns the agent.
    KNOWN LIMIT: the hull is not checked; a point within the hull radius of a wall
    (AGENT_HULL_RADIUS, cs2_types.h) is placeable though walking cannot reach it.
    """
    a = _agent(env, idx)
    area, floor = ground_at(env, x, y)
    if area == INVALID_AREA_IDX:
        raise ValueError(f"({x}, {y}) is off the mesh: no area holds it")
    if airborne and (z is None or z <= floor):
        raise ValueError(f"airborne agent {idx} needs a z above the floor {floor}, got {z}")
    a.x, a.y = x, y
    a.area_idx = area
    a.z = floor if z is None else z
    a.vx = a.vy = a.vz = 0.0
    a.is_airborne = int(airborne)
    return a


def area_centroid(env, area_idx):
    """The (x, y) the sim's StaticData holds for area_idx's centroid (spawn_team's spot)."""
    sd = env._c_env.sd.contents
    return float(sd.centroid_xy[2 * area_idx]), float(sd.centroid_xy[2 * area_idx + 1])


def place_in_area(env, idx, area_idx, **kw):
    """place_agent at area_idx's centroid, where spawn_team would put it.

    Raises ValueError when the sim resolves that centroid to another area: measured at
    977f1b8, 175 of dust2's 2248 area centroids lie outside their own area by `_area_at`
    (every simple-map and arena centroid is inside its own area).
    """
    x, y = area_centroid(env, area_idx)
    area, _ = ground_at(env, x, y)
    if area != area_idx:
        raise ValueError(f"area {area_idx}'s centroid ({x}, {y}) resolves to area {area}")
    return place_agent(env, idx, x, y, **kw)


def kill_agent(env, idx):
    """Kill agent `idx`: hp 0 and alive 0 together, as a kill in process_combat writes them.

    The sim's own consequences, a carrier's drop and the elimination check, run on the next
    step. Killing a parked or dead agent raises ValueError.
    """
    a = _agent(env, idx)
    a.hp = 0
    a.alive = 0
    return a


def kill_all_but(env, *keep):
    """kill_agent every live agent whose index is not in `keep`; returns the killed indices."""
    killed = [i for i in range(N_AGENTS) if i not in keep and env._c_env.game.agents[i].alive]
    for i in killed:
        kill_agent(env, i)
    return killed


def face(env, idx, other):
    """Turn agent `idx` to face agent `other` in the xy plane (facing = atan2 of the offset).

    Only the yaw: pitch is absolute per step (v1c), set through the continuous buffer.
    """
    import math
    a, b = env._c_env.game.agents[idx], env._c_env.game.agents[other]
    a.facing = math.atan2(b.y - a.y, b.x - a.x)
    return a


def place_duel(env, facing0=0.0):
    """Reset, then stand agent 5 (CT) 40u east of agent 0 (T), at rest, facing back at it.

    Agent 0 keeps its spawn spot and gets `facing0`; facing 0 aims it dead-on at agent 5.
    Returns zero (discrete, continuous) actions with agent 0's SHOOT set (head 1). Was
    `_place_duel`, forked in test_pitch_pin and test_stepstats_export, which
    copied agent 0's area_idx onto agent 5 instead of deriving it.
    """
    import math
    env.reset()
    a0 = env._c_env.game.agents[0]
    a0.facing = facing0
    place_agent(env, 5, a0.x + 40.0, a0.y).facing = math.pi
    act, cont = zero_actions()
    act[0, 1] = 1
    return act, cont


def visible_area_pair(env, *, min_dist, max_dist, first=400, window=200):
    """The first (area_a, area_b) whose centroids are strictly between min_dist and max_dist
    apart with line of sight, scanning each area index a < `first` against the `window` - 1
    areas after it (a+1 .. a+window-1).

    Line of sight is MapData.line_of_sight_2d, the Python mirror of cs2_combat.h's raycast
    that build_vis_matrix runs on positions (not the centroid-baked vis_matrix). Raises
    AssertionError when no pair qualifies.
    """
    md = env.map_data
    for a in range(min(first, md.N)):
        ax, ay = area_centroid(env, a)
        for b in range(a + 1, min(a + window, md.N)):
            bx, by = area_centroid(env, b)
            if not min_dist < ((bx - ax)**2 + (by - ay)**2)**0.5 < max_dist:
                continue
            if md.line_of_sight_2d(ax, ay, bx, by):
                return a, b
    raise AssertionError(f"no area pair {min_dist}..{max_dist} apart with line of sight")


def weapon_defs():
    """cs2_weapons.h's WEAPON_DEFS, one dict per weapon slot (binding.weapon_defs)."""
    return binding.weapon_defs()


def ready_to_fire(env, idx):
    """Clear the three weapon timers of live agent `idx` so it can fire this tick.

    fire_cd, reload_ticks and switch_ticks go to 0 together. A cleared switch timer means
    the drawn weapon is out, so weapon_slot must equal weapon_slot_target, and a gun needs a
    round in its clip; either failing raises ValueError. Returns the agent.
    """
    a = _agent(env, idx)
    if a.weapon_slot != a.weapon_slot_target:
        raise ValueError(f"agent {idx} is switching weapons ({a.weapon_slot} -> "
                         f"{a.weapon_slot_target}); finish the switch first")
    if weapon_defs()[a.weapon_slot]["mag_size"] >= 0 and a.ammo_clip[a.weapon_slot] <= 0:
        raise ValueError(f"agent {idx}'s weapon {a.weapon_slot} has an empty clip")
    a.fire_cd = a.reload_ticks = a.switch_ticks = 0
    return a


def set_clip(env, idx, rounds, *, slot=None):
    """Set the clip of `slot` (default: the drawn weapon) of live agent `idx` to `rounds`.

    Raises ValueError for a count outside 0..mag_size, and for the knife, which has no clip.
    """
    a = _agent(env, idx)
    slot = a.weapon_slot if slot is None else slot
    mag = weapon_defs()[slot]["mag_size"]
    if mag < 0:
        raise ValueError(f"weapon slot {slot} has no clip")
    if not 0 <= rounds <= mag:
        raise ValueError(f"{rounds} rounds do not fit weapon slot {slot}'s {mag}-round clip")
    a.ammo_clip[slot] = rounds
    return a


def state_violations(env):
    """Every way the agents of `env` break an invariant the sim keeps; [] for a sim-reachable state.

    Per agent i, all from the sim's own rules:
      * team is the slot's (agents 0..TEAM_SIZE-1 are T);
      * participating is the active-slot rule (i % TEAM_SIZE < n_active_per_team);
      * a parked slot is dead, hp 0, at INVALID_AREA_IDX (env_reset's parked state);
      * dead means hp 0 (a kill writes both), live means hp > 0;
      * a live agent's area_idx is the one ground_at resolves for (x, y), or (x, y) is that
        area's centroid: spawn_team places agents on their spawn area's centroid, which on
        dust2 can lie in another area's raster cell (area 1645, measured);
      * a live agent holds a weapon slot of WEAPON_DEFS, and each gun's clip and reserve
        are within its mag_size and reserve_mags.
    Not checked: z (a grounded agent's z is snapped on the next step, as after a spawn), and
    the bomb, which binding.step checks itself (bomb_state_error).
    """
    sd = env._c_env.sd.contents
    defs = weapon_defs()
    out = []
    for i in range(N_AGENTS):
        a = env._c_env.game.agents[i]
        if a.team != int(i >= TEAM_SIZE):
            out.append(f"agent {i}: team {a.team} is not its slot's team")
        active = i % TEAM_SIZE < sd.n_active_per_team
        if a.participating != int(active):
            out.append(f"agent {i}: participating {a.participating} breaks the active-slot rule")
        if not a.participating:
            if a.alive or a.hp != 0 or a.area_idx != INVALID_AREA_IDX:
                out.append(f"agent {i}: parked but alive {a.alive}, hp {a.hp}, area {a.area_idx}")
            continue
        if not a.alive:
            if a.hp != 0:
                out.append(f"agent {i}: dead with hp {a.hp}")
            continue
        if a.hp <= 0:
            out.append(f"agent {i}: alive with hp {a.hp}")
        area, _ = ground_at(env, a.x, a.y)
        if a.area_idx < 0 or a.area_idx >= sd.N:
            out.append(f"agent {i}: alive at area_idx {a.area_idx}, which is no area")
        elif area != a.area_idx and (a.x, a.y) != area_centroid(env, a.area_idx):
            out.append(f"agent {i}: area_idx {a.area_idx} but ({a.x}, {a.y}) is in area {area}")
        if not 0 <= a.weapon_slot < len(defs):
            out.append(f"agent {i}: weapon slot {a.weapon_slot} is not in WEAPON_DEFS")
            continue
        for s, d in enumerate(defs):
            if d["mag_size"] >= 0 and not 0 <= a.ammo_clip[s] <= d["mag_size"]:
                out.append(
                    f"agent {i}: clip {a.ammo_clip[s]} outside slot {s}'s 0..{d['mag_size']}")
            if d["reserve_mags"] >= 0 and not 0 <= a.ammo_reserve[s] <= d["reserve_mags"]:
                out.append(f"agent {i}: reserve {a.ammo_reserve[s]} outside slot {s}'s "
                           f"0..{d['reserve_mags']}")
    return out


def assert_state_consistent(env):
    """Fail with every violation state_violations finds; call it before the step under test."""
    problems = state_violations(env)
    assert not problems, "scenario breaks the sim's invariants:\n  " + "\n  ".join(problems)
