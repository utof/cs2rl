"""Scripted walk-to-bombsite-and-plant expert (Batch 6 Task 2, spec D-3/D-5).

Extracted from tests/test_env_feasibility.py + scripts/measure_budget.py so
that (a) the feasibility tests import one canonical implementation and (b) the
BC demo generator (Task 3) can replay the expert through the REAL action
interface and record per-tick (obs, discrete, continuous) triples.

Two drivers live here, deliberately kept separate:

  * drive_agent_through_area_path — the legacy TEST driver. It pokes
    `agent.facing` directly (a ctypes struct write, not an action) and is the
    fastest possible traversal. Only tests use it; it must NEVER be used for
    demo recording because the facing poke would fabricate the aim label.
  * ScriptedBomber — the BC-expert driver. Steers facing exclusively through
    the continuous [dyaw, pitch] head and plants via the discrete USE head,
    i.e. only through env.step(discrete, continuous) — the exact interface
    the policy uses, so recorded actions are legitimate BC labels (spec D-5).

Pitfalls carried over from the Gate 0 measurement (scripts/measure_budget.py):

  * auto_reset=False on the env — a mid-run silent reset corrupts tick counts
    and splices two rounds into one recorded trajectory.
  * The bomb carrier should hold the knife (weapon_slot=2, highest wishspeed
    250 u/s); setup_bomb_carrier pokes it in as round-state initialization.
  * The plant press (discrete head 4 = USE) must be held BOMB_PLANT_TIME
    ticks, and those ticks count against the same ROUND_TIME budget.
  * One-tick-ahead steering: process_movement (cs2_env.h:107) consumes
    `a->facing` BEFORE the dyaw action lands (cs2_env.h:133). A dyaw emitted
    at tick t only steers movement at t+1, so each tick we command the facing
    we want for the NEXT tick's movement and accept the one-tick lag.
  * dyaw is clamped by the env to +/-MAX_TURN_SPEED_RAD (pi/4 rad/tick); we
    pre-clamp so the recorded label equals what the env actually applied.
  * Poking pitch does NOT survive env.step (absolute-pitch overwrite,
    cs2_env.h:170); the expert always emits pitch=0 (level) as the label.
"""
import math
from collections import deque

import numpy as np

from _action_spec import AIM_DIM
from nav import ACTION_DIM, MAX_TURN_SPEED_RAD, N_AGENTS, ROUND_TIME

HEAD_MOVE = 0                          # discrete head order: move=0 shoot=1 reload=2 weapon=3 use=4 crouch=5 jump=6
HEAD_USE = 4

# Facing jitter used to wiggle off walls/corners when position stalls: the
# straight line to a centroid can clip a wall (collision zeroes velocity), so
# after every 3 stuck ticks we widen the probe angle. Mirrors what a trained
# policy learns; good enough without a sub-cell planner.
JITTER_SEQ = (0.0, math.pi / 8, -math.pi / 8, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2)


def wrap_pi(x: float) -> float:
    """Wrap an angle to (-pi, pi]. Needed before clamping dyaw so a 350deg
    'turn left' is not mistaken for a near-full right turn."""
    return (x + math.pi) % (2 * math.pi) - math.pi


def bombsite_areas(map_data) -> set:
    """Area ids (not indices) flagged as bombsite in `map_data`.

    Takes MapData, not an env: pathing must not depend on env.nav_graph,
    which is None for make_simple_map() (see _id_to_idx/centroid notes in
    bfs_area_path)."""
    return {int(aid) for i, aid in enumerate(map_data.area_ids) if map_data.bombsite_by_idx[i]}


def _id_to_idx(map_data) -> dict:
    """area_id -> row index into map_data arrays.

    GOTCHA this mapping exists to absorb: simple maps have area_id == index
    (area_ids = arange(N)), but de_dust2's NavGraph uses sparse real ids.
    MapData.area_ids/centroids are filled for BOTH map types (src/map.py), so
    deriving the mapping here works uniformly and nothing in this module may
    touch env.nav_graph."""
    return {int(aid): i for i, aid in enumerate(map_data.area_ids)}


def area_centroid(map_data, area_id: int, id_to_idx: dict | None = None):
    """(x, y) centroid of an area id, via the uniform MapData arrays."""
    if id_to_idx is None:
        id_to_idx = _id_to_idx(map_data)
    return map_data.centroids[id_to_idx[int(area_id)]]


def bfs_area_path(map_data, start_area: int, goal_areas) -> list[int]:
    """Area-level BFS over map_data.adjacency. Returns area ids
    [start, ..., goal] or [] if unreachable.

    Only picks WHICH areas to traverse; actual movement is done by a driver
    (drive_agent_through_area_path or ScriptedBomber). BFS (not A*) is fine:
    the graph is small and hop count is the budget that matters."""
    goal_areas = set(int(g) for g in goal_areas)
    id_to_idx = _id_to_idx(map_data)
    area_ids = map_data.area_ids
    adjacency = map_data.adjacency

    q = deque([int(start_area)])
    prev = {int(start_area): None}
    found = None
    while q:
        cur = q.popleft()
        if cur in goal_areas:
            found = cur
            break
        for nbr_idx in np.flatnonzero(adjacency[id_to_idx[cur]]):
            nbr_area = int(area_ids[int(nbr_idx)])
            if nbr_area == cur or nbr_area in prev:
                continue
            prev[nbr_area] = cur
            q.append(nbr_area)

    if found is None:
        return []
    path = []
    cur = found
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def drive_agent_through_area_path(env, agent_idx, area_path, max_ticks_per_hop=256) -> bool:
    """TEST-ONLY driver: walk an agent through `area_path` by poking
    `a->facing` each tick and stepping with move-forward.

    Why the poke is legal here (and only here): the default continuous buffer
    is all-zero, so dyaw=0 preserves the poked facing through env.step. This
    is the fastest traversal (no turn-rate limit) and keeps the feasibility
    tests cheap on 30+ hop dust2 paths. It fabricates the aim label, so demo
    recording must use ScriptedBomber instead (spec D-5).

    Movement is facing-local with Source-style accel + friction, so we face
    the next-area centroid and let the agent roll up to wishspeed (~3 ticks)
    and curve toward it. Stalls are unstuck via JITTER_SEQ facing offsets.

    Returns True iff the agent ends inside `area_path[-1]`."""
    map_data = env.map_data
    id_to_idx = _id_to_idx(map_data)
    for target_area in area_path[1:]:
        last_pos = None
        stuck = 0
        reached = False
        for _ in range(max_ticks_per_hop):
            ca = env._c_env.game.agents[agent_idx]
            if int(map_data.area_ids[ca.area_idx]) == target_area:
                reached = True
                break

            pos = (round(float(ca.x), 1), round(float(ca.y), 1))
            if pos == last_pos:
                stuck += 1
            else:
                stuck = 0
                last_pos = pos

            cx, cy = area_centroid(map_data, target_area, id_to_idx)
            base = math.atan2(float(cy) - float(ca.y), float(cx) - float(ca.x))
            ca.facing = base + JITTER_SEQ[min(stuck // 3, len(JITTER_SEQ) - 1)]

            actions = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int64)
            actions[agent_idx, HEAD_MOVE] = 1          # facing-local "move forward"
            env.step(actions)
        if not reached:
            return False
    return True


def setup_bomb_carrier(env, bomber_idx: int):
    """Round-state initialization pokes for the scripted bomber: make
    `bomber_idx` the sole bomb carrier and hand it the knife.

    These ARE ctypes struct pokes, but deliberately so: they configure the
    round (who spawned with the bomb, which weapon is out), they are not
    policy actions and are never recorded as labels. Keep this separate from
    the driving loop so the action stream stays pure (spec D-5).

    Knife rationale: weapon_slot=2 has the highest wishspeed (250 u/s), so
    the bomber reaches max speed in ~3 accel ticks and spends the least
    budget per area hop; switch_ticks=0 skips the draw animation."""
    g = env._c_env.game
    for i in range(N_AGENTS):
        g.agents[i].has_bomb = 0
    bomber = g.agents[bomber_idx]
    bomber.has_bomb = 1
    g.bomb_carrier_id = bomber_idx
    bomber.weapon_slot = 2
    bomber.weapon_slot_target = 2
    bomber.switch_ticks = 0


class ScriptedBomber:
    """Walk the bomb carrier along the BFS spawn->bombsite path and plant,
    using ONLY the real action interface (spec D-5) — the BC demo expert.

    Steering: each tick, discrete = move-forward (or USE while planting) and
    continuous[bomber] = [dyaw, 0.0] with
        dyaw = clamp(wrap_pi(target_facing - facing_now), +/-MAX_TURN_SPEED_RAD).
    The dyaw commanded now lands AFTER this tick's movement (cs2_env.h:107 vs
    :133), i.e. we steer one tick ahead; the walk tolerates the resulting
    slight curve. Rate-limited turning makes this strictly slower than the
    facing-poke test driver — Gate 0 measured 54-93 ticks spawn->plant on the
    simple map, comfortably inside ROUND_TIME=640.

    Task-3 usage (demo recording):

        setup_bomb_carrier(env, idx)
        bomber = ScriptedBomber(env, idx)
        for disc, cont in bomber.run():
            # env has NOT stepped yet: env.observations is obs_t and
            # (disc, cont) is exactly the action about to be executed.
            record(env.observations.copy(), disc.copy(), cont.copy())
        keep = bomber.planted    # discard failed trajectories

    The generator yields BEFORE stepping and steps on resume, so obs/action
    pairing is exact. Copy the arrays: the same buffers are reused each tick.
    After exhaustion check `planted`, `reached_site`, `ticks`, `reach_ticks`.

    Requires auto_reset=False (see module docstring) and a freshly reset env;
    the path is planned from the bomber's position at construction time.
    """

    def __init__(self, env, bomber_idx: int, max_ticks: int = ROUND_TIME):
        self.env = env
        self.bomber_idx = int(bomber_idx)
        self.max_ticks = int(max_ticks)

        self._map_data = env.map_data
        self._id_to_idx = _id_to_idx(self._map_data)
        self._agent = env._c_env.game.agents[self.bomber_idx]

        self.bombsites = bombsite_areas(self._map_data)
        start_area = int(self._map_data.area_ids[self._agent.area_idx])
        self.path = bfs_area_path(self._map_data, start_area, self.bombsites)

        self.ticks = 0                 # env.step calls issued so far
        self.reach_ticks = None        # ticks when the final path area was entered
        self.reached_site = False
        self.planted = False

        # Reused action buffers — callers must .copy() when recording.
        self._disc = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32)
        self._cont = np.zeros((N_AGENTS, AIM_DIM), dtype=np.float32)

        self._hop = 1                  # index into self.path of the current target area
        self._last_pos = None
        self._stuck = 0

    def _current_area(self) -> int:
        return int(self._map_data.area_ids[self._agent.area_idx])

    def _advance_hops(self) -> bool:
        """Skip past every path hop already satisfied; flag site arrival.
        Returns True while there is still a hop to walk."""
        cur = self._current_area()
        while self._hop < len(self.path) and cur == self.path[self._hop]:
            self._hop += 1
            self._last_pos = None      # fresh stuck tracking per hop
            self._stuck = 0
        if self._hop >= len(self.path):
            if not self.reached_site:
                self.reached_site = True
                self.reach_ticks = self.ticks
            return False
        return True

    def _next_action(self):
        """Compute (discrete, continuous) for the CURRENT env state, or None
        when the episode is over (planted, or tick budget exhausted)."""
        if self.planted or self.ticks >= self.max_ticks:
            return None
        self._disc[:] = 0
        self._cont[:] = 0.0

        if self._advance_hops():
            # WALK phase: face the next-area centroid (one tick ahead), move forward.
            pos = (round(float(self._agent.x), 1), round(float(self._agent.y), 1))
            if pos == self._last_pos:
                self._stuck += 1
            else:
                self._stuck = 0
                self._last_pos = pos

            cx, cy = area_centroid(self._map_data, self.path[self._hop], self._id_to_idx)
            base = math.atan2(float(cy) - float(self._agent.y), float(cx) - float(self._agent.x))
            target = base + JITTER_SEQ[min(self._stuck // 3, len(JITTER_SEQ) - 1)]
            dyaw = wrap_pi(target - float(self._agent.facing))
            self._cont[self.bomber_idx, 0] = max(-MAX_TURN_SPEED_RAD, min(MAX_TURN_SPEED_RAD, dyaw))
            self._disc[self.bomber_idx, HEAD_MOVE] = 1
        else:
            # PLANT phase: hold USE (BOMB_PLANT_TIME ticks) until bomb_planted.
            self._disc[self.bomber_idx, HEAD_USE] = 1
        return self._disc, self._cont

    def run(self):
        """Generator: yields the (discrete, continuous) arrays about to be
        executed, then steps the env on resume. See class docstring for the
        recording pattern; buffers are reused, so copy them to keep them."""
        if not self.path:
            return
        while True:
            action = self._next_action()
            if action is None:
                return
            yield action
            self.env.step(*action)
            self.ticks += 1
            self.planted = bool(self.env._c_env.game.bomb_planted)

    def drive(self) -> bool:
        """Run to completion without recording. Returns True iff planted."""
        for _ in self.run():
            pass
        return self.planted
