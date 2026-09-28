"""Fixed-baseline evaluation (Rung 0 R0-I, Task 13).

WHAT
----
A parent-process, serial ``Cs2Env`` driven by scripted / live-policy actors:

  * ``RandomActor``  — uniform-random discrete heads, uniform aim in ±MTS.
  * ``OracleActor``  — scripted perfect aim (nav-graph approach, fires only at a
                       VISIBLE enemy in range). The "can the pipeline kill at
                       all" control and the strongest fixed opponent.
  * ``IdleActor``    — all-zero actions (tests: isolates one side's shots in
                       the env-wide episode_stats counters).
  * ``PolicyActor``  — the live training policy (or a checkpoint), driven with
                       the same LSTM-state / mask contract as the PPO rollout.

``BaselineEvaluator`` plays the policy vs each baseline for N episodes, half
as T and half as CT, and reports elimination-only win rates under ``eval/*``.

WHY a separate module
---------------------
train.py's self-play win rate is a moving target (opponent = past self) and
``environment/winner_ct`` counts timeouts. A fixed opponent on a fixed map with
``episode_outcome`` (kills only) is the one number that is comparable across
runs and across the Rung-1 ladder.

PROVENANCE / PITFALLS
---------------------
The block between the two ``# ── vendored`` markers is copied from
``probe/combat-dead:scripts/probe/combat_rollout.py`` lines 123–558 (see that
file's module docstring for the sim conventions it relies on: Δyaw relative /
pitch absolute, env_step ordering, obs after reset() is all zeros, a killed
enemy reads invisible on its death tick). Edits after vendoring are limited to
``PolicyActor`` taking a live policy (+ ``from_policy`` / ``from_checkpoint``).

* Geometry constants below MIRROR cs2_combat.h; the header is not exported
  through the binding. If HIT_HALF_WIDTH / EYE_* / TORSO_* change there, the
  oracle silently drifts — ``tests/test_eval_baselines.py::test_oracle_never_blind``
  and ``test_oracle_beats_random`` are the tripwires.
* ``vis_from_obs`` MUST be threaded tick-to-tick (``vis_prev``): the oracle
  only fires when the target was visible on the previous tick's obs. Feeding
  None every tick degrades it to a walking, non-firing actor.
* The evaluator reads every loop bound from the env INSTANCE ``round_time``
  (R0-G knob), never ``nav.ROUND_TIME``.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from cs2rl._action_spec import ACTION_HEAD_NAMES, ACTION_HEAD_SIZES
from cs2rl._obs_spec import OBS_BLOCKS, OBS_ENEMY_COUNT, OBS_ENEMY_STRIDE
from cs2rl.c_env.cs2_env import N_AGENTS, TEAM_SIZE

# The eval/* analysis contract, RE-EXPORTED. It used to be DEFINED in this file;
# W4 moved it to the metrics registry so there is one authority for every key.
# The direction is load-bearing, not stylistic: this module imports torch and
# c_env.cs2_env at module scope (just above), so a registry that did
# `from cs2rl.eval_baselines import EVAL_KEYS` would make a tuple of eight strings cost
# a torch import and break the import-lightness invariant every new module is
# held to (tests/test_w1_modules.py). metrics_schema imports nothing from src
# except `_action_spec`, so this edge is acyclic and cheap in the one direction
# that matters.
from cs2rl.metrics_schema import EVAL_KEYS             # noqa: F401  (re-export)

HEAD_SIZES = ACTION_HEAD_SIZES                                             # probe name, kept for the vendored code
OBS_ENEMY_BASE = OBS_BLOCKS["enemy"][0]
ACTION_DIM, AIM_DIM = len(ACTION_HEAD_SIZES), 2
H_MOVE, H_SHOOT, H_RELOAD, H_WEAPON, H_USE, H_CROUCH, H_JUMP = range(7)
if ACTION_HEAD_NAMES[H_SHOOT] != "shoot" or ACTION_HEAD_NAMES[H_CROUCH] != "crouch":
    raise RuntimeError(f"action head order changed: {ACTION_HEAD_NAMES}; "
                       "re-derive H_* indices in eval_baselines.py")
                                                                           # Geometry MIRRORS of cs2_combat.h — silent-drift hazard, this branch edits that
                                                                           # header. Verified equal on main: HIT_HALF_WIDTH :200, EYE_HEIGHT_* / TORSO_OFFSET_* :234-237.
HIT_HALF_WIDTH = 16.0
HIT_HALF_HEIGHT_STAND, HIT_HALF_HEIGHT_CROUCH = 36.0, 27.0                 # v1c ellipsoid (gh #150)
EYE_STAND, EYE_CROUCH = 48.0, 24.0
TORSO_STAND, TORSO_CROUCH = 48.0, 24.0
TICK_DT = 1.0 / 16.0


# ── vendored from probe/combat-dead:scripts/probe/combat_rollout.py:123-558 ──
def wrap_pi(a):
    """Wrap an angle (or array) into [-π, π] — same semantics as C wrap_pi."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ── C-state readout ──────────────────────────────────────────────────────────


class StateReader:
    """Pulls the per-agent fields this probe needs out of the live C struct.

    Deliberately field-by-field via ctypes attribute access rather than a
    numpy structured-dtype overlay: the overlay would have to duplicate
    AgentStateC's exact padding, and a silent misalignment there would
    corrupt every number in the report. ~180 attribute reads per tick is
    negligible next to the step itself at these episode counts.
    """

    FIELDS_F = ("x", "y", "z", "facing", "pitch", "vx", "vy", "vz")
    FIELDS_I = ("hp", "alive", "team", "is_crouching", "fired_this_tick", "fire_cd", "reload_ticks",
                "switch_ticks", "weapon_slot", "area_idx", "armor")

    def __init__(self, env):
        self._agents = env._c_env.game.agents
        self._game = env._c_env.game
        self.f = {k: np.zeros(N_AGENTS, dtype=np.float64) for k in self.FIELDS_F}
        self.i = {k: np.zeros(N_AGENTS, dtype=np.int64) for k in self.FIELDS_I}
        self.ammo_clip = np.zeros(N_AGENTS, dtype=np.int64)
        self.ammo_reserve = np.zeros(N_AGENTS, dtype=np.int64)

    def read(self):
        ag = self._agents
        for n in range(N_AGENTS):
            a = ag[n]
            for k in self.FIELDS_F:
                self.f[k][n] = getattr(a, k)
            for k in self.FIELDS_I:
                self.i[k][n] = getattr(a, k)
            slot = int(a.weapon_slot)
            self.ammo_clip[n] = a.ammo_clip[slot]
            self.ammo_reserve[n] = a.ammo_reserve[slot]
        return self

    def snapshot(self):
        """Deep copy of the mutable arrays (the read buffers are reused)."""
        return {
            **{
                k: v.copy()
                for k, v in self.f.items()
            },
            **{
                k: v.copy()
                for k, v in self.i.items()
            },
            "ammo_clip": self.ammo_clip.copy(),
            "ammo_reserve": self.ammo_reserve.copy(),
            "tick": int(self._game.tick),
            "round_over": int(self._game.round_over),
            "winner": int(self._game.winner),
            "bomb_planted": int(self._game.bomb_planted),
        }


def enemy_range(team_val):
    """Enemy agent indices for an agent on `team_val` (0 = T, 1 = CT)."""
    return range(TEAM_SIZE, N_AGENTS) if team_val == 0 else range(0, TEAM_SIZE)


def vis_from_obs(obs, st, map_diag, match_tol=2.0):
    """Recover the sim's own vis10 (alive-gated) from the observation block.

    obs[i, enemy_slot + 3] is ``en->alive ? vis10[i][ej] : 0``; slots are
    sorted by known distance so the enemy index has to be recovered by
    matching the slot's (dx, dy) against the true relative positions in `st`
    (which must be the state from the SAME tick the obs was written).

    POST-VENDORING EDIT (Task 13, deviation from the probe): since R0-E.1
    (#130, cs2_observations.h "FACING-RELATIVE frame") slot +0/+1 hold
    (rx, ry)/map_diag rotated by -facing (+x = ahead, +y = left), not world
    dx/dy. Rotate back with the same-tick ``st["facing"][i]`` (nothing
    mutates facing between compute_observations and step() returning) before
    matching. Without this every visible slot is "unmatched", vis is all
    False, and the oracle silently never engages (walks into the enemy and
    times out — the failure mode test_oracle_never_blind + test_oracle_beats_random
    catch).

    Returns (vis[10,10] bool, n_unmatched). n_unmatched > 0 means a visible
    slot could not be tied to an agent within `match_tol` world units, which
    would invalidate the visibility numbers — it is reported in the summary
    and should always be 0.
    """
    vis = np.zeros((N_AGENTS, N_AGENTS), dtype=bool)
    unmatched = 0
    x, y, team, facing = st["x"], st["y"], st["team"], st["facing"]
    for i in range(N_AGENTS):
        ens = list(enemy_range(int(team[i])))
        cf, sf = math.cos(facing[i]), math.sin(facing[i])
        for s in range(OBS_ENEMY_COUNT):
            b = OBS_ENEMY_BASE + s * OBS_ENEMY_STRIDE
            if obs[i, b + 3] < 0.5:
                continue
            rx = float(obs[i, b + 0]) * map_diag
            ry = float(obs[i, b + 1]) * map_diag
            # inverse of C: rx = dx*cf + dy*sf, ry = -dx*sf + dy*cf
            dx = rx * cf - ry * sf
            dy = rx * sf + ry * cf
            best, bestd = -1, 1e30
            for j in ens:
                d = abs((x[j] - x[i]) - dx) + abs((y[j] - y[i]) - dy)
                if d < bestd:
                    bestd, best = d, j
            if best >= 0 and bestd <= match_tol:
                vis[i, best] = True
            else:
                unmatched += 1
    return vis, unmatched


def _vis_at_combat(vis_now, vis_prev, died_mask):
    """Visibility as process_combat saw it this tick.

    compute_observations zeroes can_see for enemies that died THIS tick, so
    the killer's own target reads invisible in the post-step obs. Back-fill
    those columns from the previous tick's obs, which is the closest
    available proxy (positions move <16 u/tick, LoS rarely flips).
    """
    if not died_mask.any() or vis_prev is None:
        return vis_now
    v = vis_now.copy()
    for j in np.nonzero(died_mask)[0]:
        v[:, j] = vis_prev[:, j]
    return v


def aim_geometry(st, i, j):
    """Exact combat-ray geometry from shooter i to target j (post-step state).

    Returns (rx, ry, rz, dist3d, true_yaw, true_pitch) where true_yaw /
    true_pitch are the facing/pitch that would put the ray dead-centre on
    the target's torso, using the sim's own eye/torso offsets.
    """
    eye_z = st["z"][i] + (EYE_CROUCH if st["is_crouching"][i] else EYE_STAND)
    torso_z = st["z"][j] + (TORSO_CROUCH if st["is_crouching"][j] else TORSO_STAND)
    rx = st["x"][j] - st["x"][i]
    ry = st["y"][j] - st["y"][i]
    rz = torso_z - eye_z
    dist = math.sqrt(rx * rx + ry * ry + rz * rz)
    true_yaw = math.atan2(ry, rx)
    true_pitch = math.atan2(rz, math.hypot(rx, ry))
    return rx, ry, rz, dist, true_yaw, true_pitch


def perp_and_forward(st, i, j):
    """Replicate cs2_combat.h's 3D gate for shooter i against target j.

    Returns (perp, forward, dist) where `perp` is the v1c ELLIPSOID-normalised
    perpendicular: HIT_HALF_WIDTH * sqrt((p_h/16)² + (p_v/HH)²), HH by the
    target's stance — so the v1b rule "connects iff dist <= laser_range,
    dist > 0, forward > 0 and perp <= HIT_HALF_WIDTH" still reads correctly.
    At pitch 0 with the same z AND the same stance it equals the plain
    perpendicular distance. Mixed stance is not that case: p_v = rz = ±24 even
    at pitch 0 and equal z, which is exactly the shot v1c turns into a hit.
    """
    rx, ry, rz, dist, _, _ = aim_geometry(st, i, j)
    yaw, pitch = st["facing"][i], st["pitch"][i]
    cp, sp = math.cos(pitch), math.sin(pitch)
    dx, dy, dz = cp * math.cos(yaw), cp * math.sin(yaw), sp
    forward = rx * dx + ry * dy + rz * dz
    px, py, pz = rx - forward * dx, ry - forward * dy, rz - forward * dz
    hh = HIT_HALF_HEIGHT_CROUCH if st["is_crouching"][j] else HIT_HALF_HEIGHT_STAND
    perp = HIT_HALF_WIDTH * math.sqrt((px * px + py * py) / HIT_HALF_WIDTH**2 + pz * pz / hh**2)
    return perp, forward, dist


# ── action sources ───────────────────────────────────────────────────────────


class RandomActor:
    """Uniform-random over every discrete head; cont aim uniform in ±MTS.

    ±max_turn_speed matches the policy's own bounded output range (tanh ×
    max_turn_speed), so A and B differ only in *where* they aim, not in how
    far they can turn per tick.
    """

    name = "random"

    def __init__(self, rng, max_turn_speed):
        self.rng = rng
        self.mts = max_turn_speed

    def reset(self):
        pass

    def act(self, obs, st, vis_prev, env):
        act = np.stack([self.rng.integers(0, n, size=N_AGENTS) for n in HEAD_SIZES],
                       axis=1).astype(np.int32)
        cont = self.rng.uniform(-self.mts, self.mts, size=(N_AGENTS, AIM_DIM)).astype(np.float32)
        return act, cont


class PolicyActor:
    """Trained checkpoint, driven exactly like the PPO rollout.

    Same call shape as train.py's rollout: forward_eval with the carried
    LSTM state + done flags, then _hybrid_sample_logits with the C action
    masks. Not select_policy_actions_native, because that helper does NOT
    pass masks — sampling an invalid bin (e.g. shoot while on cooldown) is
    a no-op in C but shifts the discrete distribution away from what
    training actually saw.
    """

    name = "policy"

    def __init__(self, policy, device):
        # Post-vendoring edit (Task 13): takes a LIVE policy module — the
        # training loop hands its own `policy` in; use from_checkpoint for the
        # probe's original load-from-file behaviour. Late import: train.py
        # imports this module, a top-level import would be circular.
        from cs2rl.train import _hybrid_sample_logits, init_policy_state
        self.torch = torch
        self._sample = _hybrid_sample_logits
        self._init_state = init_policy_state
        self.device = device
        self.policy = policy
        self.mts = float(self.policy.max_turn_speed.item())
        self.state = None
        self.done = np.zeros(N_AGENTS, dtype=np.float32)

    def reset(self):
        self.state = self._init_state(self.policy, self.device)
        self.done[:] = 0.0

    def act(self, obs, st, vis_prev, env):
        torch = self.torch
        obs_t = torch.as_tensor(np.ascontiguousarray(obs), device=self.device)
        mask_t = torch.as_tensor(np.ascontiguousarray(env._masks_view), device=self.device) != 0
        self.state["done"] = torch.as_tensor(self.done, device=self.device)
        with torch.no_grad():
            logits, mu, log_std, value = self.policy.forward_eval(obs_t, self.state)
            act_t, cont_t, *_ = self._sample((logits, mu, log_std, value),
                                             max_turn_speed=self.mts,
                                             mask=mask_t)
        return (act_t.cpu().numpy().astype(np.int32), cont_t.cpu().numpy().astype(np.float32))

    def mark_done(self, terms, truncs):
        self.done[:] = np.logical_or(np.asarray(terms), np.asarray(truncs)).astype(np.float32)

    @classmethod
    def from_policy(cls, policy, device):
        """Wrap the live training policy (no load). One actor = one LSTM state:
        build a DISTINCT actor per side when the same module plays both."""
        return cls(policy, device)

    @classmethod
    def from_checkpoint(cls, ckpt, device, **build_kwargs):
        """Probe-style: rebuild from a bare state_dict file. `build_kwargs`
        (aim_log_std_max, pin_pitch) are RUN properties the checkpoint cannot
        tell you — pass the run's values (see load_policy_from_checkpoint)."""
        from cs2rl.train import load_policy_from_checkpoint
        return cls(load_policy_from_checkpoint(ckpt, device, **build_kwargs), device)


# Mirror of cs2_movement.h SV_MAX_STEP_HEIGHT_CS — the up-step the cliff
# guard in _resolve_xy_collision permits into a non-ramp area.
SV_MAX_STEP_HEIGHT = 18.0


class NavHelper:
    """Next-hop lookup for the oracle's approach phase, over the C adjacency.

    WHY this is needed at all: on dust2 the two spawns are ~3300 world units
    apart with the whole map between them. Walking in a straight line toward
    the enemy wedges the agent into the first wall it meets and the teams
    never make contact — a "walk at the enemy" oracle produces a 640-tick
    round with literally zero visible-enemy ticks, which is
    indistinguishable from a broken visibility system. Pathing is what makes
    C a control for COMBAT rather than a test of straight-line walking.

    WHY the C adjacency and not NavGraph.graph: movement is gated by
    ``sd->adjacency`` in _resolve_xy_collision — a step into an area that is
    not raster-adjacent is rejected outright. NavGraph.graph carries the raw
    awpy nav connections, which include links the 2D movement model cannot
    execute. Routing on those wedges the agent permanently against a wall
    (observed: agents freeze ~80 ticks in with vx == vy == 0 forever). The
    same cliff prune the C guard applies (Δz > 18 u into a non-ramp area) is
    replicated here so the route never asks for a step movement will refuse.

    Implementation: one BFS distance field per goal area, LRU-cached (each
    field is an int array over N≈2248 areas), then greedy descent to the
    adjacent area with the smallest distance-to-goal. Indices throughout are
    C ``area_idx`` (dense 0-based), never awpy area_ids.
    """

    def __init__(self, sd, cache_size=256):
        import ctypes
        n = int(sd.N)
        self.n = n
        adj = np.ctypeslib.as_array(ctypes.cast(sd.adjacency, ctypes.POINTER(ctypes.c_int8)),
                                    shape=(n * n, )).reshape(n, n) != 0
        self.centroid_xy = np.ctypeslib.as_array(ctypes.cast(sd.centroid_xy,
                                                             ctypes.POINTER(ctypes.c_float)),
                                                 shape=(n * 2, )).reshape(n, 2).astype(np.float64)
        cz = np.ctypeslib.as_array(ctypes.cast(sd.centroids_z, ctypes.POINTER(ctypes.c_float)),
                                   shape=(n, )).astype(np.float64)
        ramp = np.ctypeslib.as_array(ctypes.cast(sd.is_ramp, ctypes.POINTER(ctypes.c_int8)),
                                     shape=(n, )) != 0
        # Cliff prune, mirroring _resolve_xy_collision's L11 guard.
        dz = cz[None, :] - cz[:, None]
        walkable = adj & ((dz <= SV_MAX_STEP_HEIGHT) | ramp[None, :])
        np.fill_diagonal(walkable, False)
        self.neighbors = [np.nonzero(row)[0] for row in walkable]
        self._fields: dict[int, np.ndarray] = {}
        self._order: list[int] = []
        self._cache_size = cache_size

    def _field(self, goal_idx):
        """BFS hop-count from every area to `goal_idx` (UNREACHED = -1)."""
        f = self._fields.get(goal_idx)
        if f is not None:
            return f
        from collections import deque
        dist = np.full(self.n, -1, dtype=np.int32)
        dist[goal_idx] = 0
        q = deque([goal_idx])
        nb = self.neighbors
        while q:
            u = q.popleft()
            du = dist[u] + 1
            for v in nb[u]:
                if dist[v] < 0:
                    dist[v] = du
                    q.append(v)
        self._fields[goal_idx] = dist
        self._order.append(goal_idx)
        if len(self._order) > self._cache_size:
            self._fields.pop(self._order.pop(0), None)
        return dist

    def next_hop_xy(self, my_idx, goal_idx):
        """World (x, y) of the next area centroid on the path, or None."""
        if my_idx < 0 or goal_idx < 0 or my_idx == goal_idx:
            return None
        f = self._field(goal_idx)
        here = f[my_idx]
        if here < 0:
            return None                # disconnected component
        nbs = self.neighbors[my_idx]
        if len(nbs) == 0:
            return None
        d = f[nbs]
        ok = d >= 0
        if not ok.any():
            return None
        cand = nbs[ok]
        dc = d[ok]
        best = cand[int(np.argmin(dc))]
        if f[best] >= here:
            return None                # local minimum / already adjacent
        return self.centroid_xy[best]


class OracleActor:
    """Scripted perfect aim — the "can the pipeline kill at all" control.

    Per agent, per tick:
      * target = nearest VISIBLE enemy (the sim's own vis10, recovered from
        last tick's obs); if none is visible, the nearest enemy by 3D
        distance, approached along the nav graph.
      * Δyaw = wrap_pi(bearing_to_target − current facing), clamped to
        ±max_turn_speed exactly as C will clamp it.
      * pitch = the absolute pitch that puts the ray on the target's torso
        (given the full ±π/2 the C clamp permits — the ±max_turn_speed bound
        on pitch is a property of the policy's tanh, not of the env).
      * fire whenever the target is visible, in range, and the weapon is
        actually ready (fire_cd / reload / switch / ammo); reload when dry.

    One-tick velocity lead: the aim command is applied AFTER movement in
    env_step, so the target has already moved by the time process_combat
    runs. Positions are extrapolated by v·dt (dt = 1/16 s). Without it, a
    target crossing at 220 u/s moves ~14 u per tick, which is essentially
    the entire 16 u hit window.

    Movement is coupled to aim by the sim (process_movement rotates the
    facing-local bin by a->facing), so "walk to the waypoint" necessarily
    means "look at the waypoint". The oracle therefore only navigates while
    it has no visible target; the moment one appears it plants and aims.

    No aim noise, no exploration — if this actor cannot kill, no policy can.
    """

    name = "oracle"

    def __init__(self, rng, max_turn_speed, laser_range, nav):
        self.rng = rng
        self.mts = max_turn_speed
        self.laser_range = laser_range
        self.nav = nav

    def reset(self):
        pass

    def act(self, obs, st, vis_prev, env):
        act = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32)
        cont = np.zeros((N_AGENTS, AIM_DIM), dtype=np.float32)
        x, y, z = st["x"], st["y"], st["z"]
        vx, vy = st["vx"], st["vy"]
        alive, team, crouch = st["alive"], st["team"], st["is_crouching"]
        area = st["area_idx"]
        vis = vis_prev if vis_prev is not None else np.zeros((N_AGENTS, N_AGENTS), dtype=bool)

        for i in range(N_AGENTS):
            if not alive[i]:
                continue
            eye_z = z[i] + (EYE_CROUCH if crouch[i] else EYE_STAND)
            best_j, best_d, best_vis = -1, 1e30, False
            for j in enemy_range(int(team[i])):
                if not alive[j]:
                    continue
                # 1-tick lead: where the target will be when combat runs.
                tx = x[j] + vx[j] * TICK_DT
                ty = y[j] + vy[j] * TICK_DT
                tz = z[j] + (TORSO_CROUCH if crouch[j] else TORSO_STAND)
                d = math.sqrt((tx - x[i])**2 + (ty - y[i])**2 + (tz - eye_z)**2)
                seen = bool(vis[i, j])
                # Prefer any visible enemy over any invisible one; break ties
                # on distance. Shooting an invisible enemy is a guaranteed
                # miss that still burns the fire cooldown and a round.
                better = (seen and not best_vis) or (seen == best_vis and d < best_d)
                if better:
                    best_j, best_d, best_vis = j, d, seen
            if best_j < 0:
                continue

            j = best_j
            engage = best_vis and best_d <= self.laser_range
            if engage:
                aim_x = x[j] + vx[j] * TICK_DT
                aim_y = y[j] + vy[j] * TICK_DT
                aim_z = z[j] + (TORSO_CROUCH if crouch[j] else TORSO_STAND)
            else:
                hop = self.nav.next_hop_xy(int(area[i]), int(area[j]))
                if hop is None:
                    # Same area, no path, or off-mesh: head straight at them.
                    aim_x, aim_y = x[j], y[j]
                else:
                    aim_x, aim_y = float(hop[0]), float(hop[1])
                aim_z = eye_z          # level look while walking
                act[i, H_MOVE] = 1     # facing-local bin 1 = forward

            rx, ry, rz = aim_x - x[i], aim_y - y[i], aim_z - eye_z
            tgt_yaw = math.atan2(ry, rx)
            tgt_pitch = math.atan2(rz, math.hypot(rx, ry))
            dyaw = wrap_pi(tgt_yaw - st["facing"][i])
            cont[i, 0] = float(np.clip(dyaw, -self.mts, self.mts))
            cont[i, 1] = float(np.clip(tgt_pitch, -math.pi / 2, math.pi / 2))

            ready = (st["fire_cd"][i] == 0 and st["reload_ticks"][i] == 0
                     and st["switch_ticks"][i] == 0)
            has_ammo = st["ammo_clip"][i] > 0
            if engage and ready and has_ammo:
                act[i, H_SHOOT] = 1
            elif not has_ammo and st["reload_ticks"][i] == 0 and st["ammo_reserve"][i] > 0:
                act[i, H_RELOAD] = 1
        return act, cont


# ── episode driver ───────────────────────────────────────────────────────────

# ── end vendored block ──


class IdleActor:
    """All-zero actions for every row (stand still, never fire).

    WHY: episode_stats counters (shots_fired, shots_with_enemy_in_los, kills_*)
    are ENV-WIDE, so a test that wants "every shot was the oracle's" needs an
    opponent that never fires. Also the scripted-kill control in the n>1
    win-definition test. Not an eval baseline (a policy beating a statue says
    nothing) — tests only.
    """

    name = "idle"

    def reset(self):
        pass

    def act(self, obs, st, vis_prev, env):
        return (np.zeros((N_AGENTS, ACTION_DIM),
                         dtype=np.int32), np.zeros((N_AGENTS, AIM_DIM), dtype=np.float32))


def episode_outcome(kills_for: int, kills_against: int) -> float:
    """Win = opposing participating agent eliminated and own agent alive.

    Timeouts score 0 (the C `winner_ct` counts them as CT wins — that is why
    train.py's self-play feed subtracts `timed_out`, see
    elimination_only_win_rates). Trade = 0.5 (unreachable at n=1: the round
    ends on the first death). PITFALL: at n_active_per_team>1 the inputs are
    TEAM kill counts, so this is team credit, not the policy agent's own.
    """
    if kills_for >= 1 and kills_against == 0:
        return 1.0
    if kills_for >= 1 and kills_against >= 1:
        return 0.5
    return 0.0


# EVAL_KEYS — the eval/* keys evaluate() below returns, i.e. the analysis
# contract — is NOT defined here any more. It lives in src/cs2rl/metrics_schema.py
# (W4, spec 2026-08-31 §2 W4) and is imported at the top of this file, which
# re-exports it for existing `eval_baselines.EVAL_KEYS` consumers.


class BaselineEvaluator:
    """Parent-process serial Cs2Env, built once, reused across evals (R0-I).

    Per episode the policy fills ITS side's rows and the opponent actor the
    other side's; both return full (N_AGENTS, …) arrays and we splice by team
    slot. Policy LSTM state is carried on its rows only (PolicyActor keeps the
    full-N state; unused rows are simply ignored). Eval never feeds training.

    PITFALLS
    * The env must be built with ``auto_reset=False``: with auto-reset the
      post-step C state on the terminal tick is already the next round's
      spawn and episode_stats would be cleared before we read it.
    * Every loop bound reads ``env.round_time`` (the R0-G instance knob), so
      an arena eval at 160 ticks does not spin to nav.ROUND_TIME.
    * ``episodes`` must be even and ≥2: half the episodes are played as T,
      half as CT; an odd count would silently bias one side.
    """

    def __init__(self, env, episodes=40, seed=0):
        if episodes < 2 or episodes % 2:
            raise ValueError(f"episodes must be even and >= 2 (half per side), got {episodes}")
        if getattr(env, "_auto_reset", False):
            raise ValueError("BaselineEvaluator needs an env built with auto_reset=False "
                             "(terminal-tick episode_stats would be wiped by the auto reset)")
        self.env = env
        self.episodes = episodes
        self.seed = seed
        sd = env._c_env.sd.contents
        self.max_turn_speed = float(sd.max_turn_speed)
        self.laser_range = float(sd.laser_range)
        # vis_from_obs() expects distances in world units; obs stores dist/map_diag
        # (probe driver combat_rollout.py's main() derives it the same way).
        self.map_diag = math.sqrt((1.0 / sd.inv_x_range)**2 + (1.0 / sd.inv_y_range)**2)
        self.nav = NavHelper(sd)
        self.reader = StateReader(env)

    def _episode(self, side_actor, opp_actor, side):
        """One round; returns (win score, kills_for, episode_stats dict).

        `side` 0 ⇒ side_actor owns T rows [0, TEAM_SIZE), 1 ⇒ CT rows.
        kills_for/against are the env's TEAM counters (team credit at n>1).
        """
        env = self.env
        obs, _ = env.reset()
        side_actor.reset()
        opp_actor.reset()
        st = self.reader.read().snapshot()
        vis_prev = None
        rows = slice(0, TEAM_SIZE) if side == 0 else slice(TEAM_SIZE, N_AGENTS)
        term = trunc = None
        for _ in range(int(env.round_time) + 1):
            a1, c1 = side_actor.act(obs, st, vis_prev, env)
            a2, c2 = opp_actor.act(obs, st, vis_prev, env)
            act, cont = a2.copy(), c2.copy()
            act[rows], cont[rows] = a1[rows], c1[rows]
            obs, rew, term, trunc, info = env.step(act, cont)
            st = self.reader.read().snapshot()
            # OracleActor's peek/memory branch reads vis_prev; feeding None every
            # tick silently degrades it to a blind actor (probe :642-666 threads it).
            vis_prev, _unmatched = vis_from_obs(obs, st, self.map_diag)
            for actor in (side_actor, opp_actor):
                if hasattr(actor, "mark_done"):
                    actor.mark_done(term, trunc)
            if term.any() or trunc.any():
                break
        else:
            raise RuntimeError(
                f"episode did not terminate within round_time+1={env.round_time + 1} "
                "ticks — the env's timeout terminal is broken")
        es = env._c_env.episode_stats
        kt, kc = int(es.kills_t), int(es.kills_ct)
        kf, ka = (kt, kc) if side == 0 else (kc, kt)
        stats = {
            "shots_fired": int(es.shots_fired),
            "shots_with_enemy_in_los": int(es.shots_with_enemy_in_los),
            "timed_out": int(es.timed_out),
        }
        return episode_outcome(kf, ka), kf, stats

    def run_pair(self, policy_actor, opp_actor):
        """episodes/2 rounds per side; env-wide shot counters are summed (both actors)."""
        half = self.episodes // 2
        wins = {0: [], 1: []}
        kills = []
        shots = {"shots_fired": 0, "shots_with_enemy_in_los": 0, "timed_out": 0}
        for side in (0, 1):
            for _ in range(half):
                w, k, stats = self._episode(policy_actor, opp_actor, side)
                wins[side].append(w)
                kills.append(k)
                for key in shots:
                    shots[key] += stats[key]
        return {
            "win_as_t": float(np.mean(wins[0])),
            "win_as_ct": float(np.mean(wins[1])),
            "win": float(np.mean(wins[0] + wins[1])),
            "kills_per_episode": float(np.mean(kills)),
            **shots,
        }

    def evaluate(self, policy, device) -> dict:
        """Policy vs random, then vs oracle → the EVAL_KEYS dict (all floats).

        Determinism contract (spec §6 seeding): eval must not perturb the training
        RNG streams (torch fork_rng; numpy draws use dedicated Generators), and
        each baseline gets its own fresh generator so the random-opponent draw
        count cannot shift the oracle's tie-break draws. The policy module is
        shared with PPO: forward runs under no_grad and train() mode is restored.
        """
        dev = torch.device(device)
        fork_devices = [dev.index or 0] if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            pa = PolicyActor.from_policy(policy, device)
            rnd = RandomActor(np.random.default_rng(self.seed), self.max_turn_speed)
            orc = OracleActor(np.random.default_rng(self.seed + 1), self.max_turn_speed,
                              self.laser_range, self.nav)
            r = self.run_pair(pa, rnd)
            o = self.run_pair(pa, orc)
        policy.train()
        out = {
            "eval/win_vs_random": r["win"],
            "eval/win_vs_random_as_t": r["win_as_t"],
            "eval/win_vs_random_as_ct": r["win_as_ct"],
            "eval/kills_per_episode_vs_random": r["kills_per_episode"],
            "eval/win_vs_oracle": o["win"],
            "eval/win_vs_oracle_as_t": o["win_as_t"],
            "eval/win_vs_oracle_as_ct": o["win_as_ct"],
            "eval/kills_per_episode_vs_oracle": o["kills_per_episode"],
        }
        if set(out) != set(EVAL_KEYS):
            raise RuntimeError("EVAL_KEYS drifted from evaluate()")
        return out
