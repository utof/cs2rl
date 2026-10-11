"""Scripted duel bot: a dialled rifle duellist that shoots back (#152 layer L2).

WHAT
----
``DuelBot`` plays any number of agent rows, vectorised over the rows with one numpy
``Generator``. Per tick and row it reads the row's own observation and action mask
(``step_obs``), or the C state for the ground-truth instrument mode (``step_state``),
and returns the row's discrete actions and (Δyaw, pitch). ``DuelBotActor`` adapts it to
the ``act(obs, st, vis_prev, env)`` shape of the cs2rl.eval.baselines actors, for the
scripted-bot harnesses (cs2rl.experiment.oracle_duel).

The dials (``DuelBotParams``) are the CS2 bot-profile dials at the sim's 16 Hz
(``PROFILES``, from CS2's shipped botprofile.db), plus lead and a movement style:

  * ``reaction_ticks`` k: ReactionTime x 16, rounded half up as ReGameDLL rounds its
    queue steps (``int(t / interval + 0.5)``). ``reaction_mode`` says what is late:
      recognition (default)  the bot engages an enemy it saw k ticks ago and still sees,
                             and aims at where it sees it now: ReGameDLL's reaction
                             queue stores who was seen, and Upkeep aims at the enemy's
                             current origin;
      percept                the bot aims at the enemy as it was seen k ticks ago (its
                             position and velocity), so it trails a moving target by k
                             ticks of its motion.
    Measured (lane E, 2026-10-11): the hard stop bot hit 0.0327 of its shots at the
    hard runner as a percept bot, 0.1084 as a recognition bot.
  * the focus: an aim offset (yaw, pitch), each drawn uniform in [-F, F] (ReGameDLL's
    ``RANDOM_FLOAT(-error, error)`` per axis) every ``focus_interval_ticks`` while the
    bot is engaged. F starts at ``focus_initial_deg`` on the tick the bot acquires its
    target and shrinks by the factor ``focus_decay`` per second of engagement
    (botprofile.db: "how much focus shrinks per second (.25 = 25% of size after 1
    sec)"). ``focus_offset_scale`` s makes the aim trail the target between draws:
    aim = bearing + offset - s * (bearing - bearing at the last draw). s = 0 tracks
    perfectly, s near 1 leaves the aim where it was drawn (botprofile.db: "controls
    accuracy when tracking to target (0 == perfect, should always be < 1)"; "smaller
    intervals means better movement tracking").
    INFERRED: the db names the four constants and says what each means; CS2's code is
    not public, so the composition above is this module's, not Valve's.
  * ``attack_delay_ticks``: no shot before this many ticks after acquisition
    (AttackDelay; ReGameDLL's FireWeaponAtEnemy waits GetAttackDelay() after
    acquiring the enemy).
  * ``lead``: aim where the enemy the bot aims at (its percept, now or k ticks old)
    will be when the shot is resolved, one tick of its velocity ahead, as OracleActor
    does (the sim moves both agents, then resolves combat).
  * ``style``, the movement around the shot (0.34 x vmax is where the cone starts,
    cs2_weapons.h ACCURACY_RAMP_START_FRAC):
      plant  stands still and fires whenever ready (OracleActor's habit);
      stop   strafes until engaged, then counter-strafes (presses the move bin most
             opposite its velocity while that brakes harder than releasing), stands,
             and fires only when its post-move speed is under 0.34 x vmax;
      run    strafes all the time and fires whenever ready (run-and-gun);
      adad   strafes during the weapon's cooldown, counter-strafes on the tick it is
             ready and fires if that brings it under 0.34 x vmax, then strafes the
             other way.
    A strafe holds the facing-local right or left bin for ``strafe_ticks`` ticks, and
    reverses early when the next 3 ticks would take it within ``WALL_MARGIN`` of the
    map's extent.

Every style aims from its own predicted post-move position (the sim moves the shooter
before the shot) and fires only on a tick whose turn reaches its aim (ReGameDLL's
FireWeaponAtEnemy pulls the trigger only while its view is within a tolerance of its aim
spot), so it never wastes the opening shot of a turn wider than ``max_turn_speed``.

WHY
---
L2 must show a duel in this sim is winnable and symmetric, that stopping before a
rifle shot wins under the movement-inaccuracy cone, and that lead matters. The same
bot, at the CS2 profiles' dials, is meant as Rung 1c's curriculum opponent, so it
decides from what a trainer has for a row: the row's observation, its action mask, its
own state from the previous ticks, and a seeded Generator. Nothing here reads the C
state in ``step_obs``.

PITFALLS
--------
* The enemy's velocity comes from two consecutive observations (positions decoded
  from the obs, differenced over one tick). The sim integrates position with the
  post-step velocity (cs2_movement.h), so the difference IS the velocity the ground
  truth reads, to float32 rounding. Unseen on either tick: 0.
* Own position is decoded from the self block (obs[3], obs[4] are the position
  normalised to [-1, 1] over the map's extent), in a frame centred on the map. The
  wall check uses that extent: right for the open arena, wrong for a map with inner
  walls (KNOWN LIMIT: open arenas only, as #152 L2 is).
* The obs carries no enemy stance or facing: the aim point is the standing target's
  height on its axis (``aim_dz``), in both modes, so the two modes aim at the same
  point. Pitch only matters where the env does not pin it (KNOWN LIMIT: the bot never
  crouches; its eye is the standing one).
* The movement constants mirror cs2_movement.h's ground path (``_ground_velocity``);
  tests/eval/test_duel_bot.py pins the prediction against env.step.
* The bot's random draws come from its own Generator, never from the env's streams.
  Rows draw in row order, only on the ticks they redraw focus.
* Module scope imports numpy and cs2rl.spec only (like eval.walker), so a trainer's
  light modules may import it. ``DuelBotActor`` imports the env side lazily.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cs2rl.spec.action import ACTION_DIM, ACTION_HEAD_NAMES, ACTION_HEAD_SIZES, AIM_DIM
from cs2rl.spec.obs import OBS_BLOCKS, OBS_ENEMY_COUNT, OBS_ENEMY_STRIDE

TICK_RATE = 16
DT = 1.0 / TICK_RATE

H_MOVE = ACTION_HEAD_NAMES.index("move")
H_SHOOT = ACTION_HEAD_NAMES.index("shoot")
H_RELOAD = ACTION_HEAD_NAMES.index("reload")
# Bin 1 ("legal") of a two-bin head in the flat action mask: the heads' masks are laid out
# back to back in ACTION_HEAD_SIZES order (cs2_env.h compute_masks).
_MASK_OFFSETS = np.concatenate(([0], np.cumsum(ACTION_HEAD_SIZES)[:-1]))
MASK_SHOOT = int(_MASK_OFFSETS[H_SHOOT]) + 1
MASK_RELOAD = int(_MASK_OFFSETS[H_RELOAD]) + 1

# ── MIRRORS of cs2_observations.h's self block and enemy slot (no generated names exist) ──
# The obs-only parity check (tests/experiment/test_oracle_duel.py: the bots on their obs rows
# against the same bots on the C state) is the tripwire: a wrong index decodes a wrong
# position and the two modes stop matching.
SELF_X, SELF_Y = 3, 4                  # position normalised to [-1, 1] over the map's extent
SELF_VX, SELF_VY = 6, 7                # velocity / 250
SELF_SIN_YAW, SELF_COS_YAW = 9, 10
SELF_CLIP = 17                         # clip / mag_size
SELF_ALIVE = 23
OBS_VEL_SCALE = 250.0
OBS_Z_SCALE = 128.0
EN_RX, EN_RY, EN_DZ, EN_SEE, EN_ALIVE, EN_SIN, EN_COS, EN_DIST = range(OBS_ENEMY_STRIDE)
_EN_BASES = OBS_BLOCKS["enemy"][0] + OBS_ENEMY_STRIDE * np.arange(OBS_ENEMY_COUNT)

# ── MIRRORS of cs2_movement.h's ground path (pinned against env.step by the tests) ──
SV_FRICTION = 5.2
SV_ACCELERATE = 5.5
SV_STOPSPEED = 80.0
_LOCAL_X = np.array([0.0, 0.0, 0.70710678, 1.0, 0.70710678, 0.0, -0.70710678, -1.0, -0.70710678])
_LOCAL_Y = np.array([0.0, 1.0, 0.70710678, 0.0, -0.70710678, -1.0, -0.70710678, 0.0, 0.70710678])
STRAFE_RIGHT, STRAFE_LEFT = 3, 7       # facing-local right / left (test_strafe_labels_match_geometry)
                                       # Where the movement-inaccuracy cone starts, as a fraction of the weapon's move speed:
                                       # MIRROR of cs2_weapons.h ACCURACY_RAMP_START_FRAC.
ACCURACY_THRESHOLD_FRAC = 0.34

MAX_REACTION_TICKS = 16                # the delay line's length: 1 s, above Easy's 10
                                       # A turn "lands" when the aim is within max_turn_speed + LANDS_TOL. WHY the tolerance: a CT
                                       # spawns facing float32(pi), so its turn to a 45-degree bearing (the arena's rows 200 u apart
                                       # at 200 u) is max_turn_speed plus rounding, while the T's is exact; without it a perfect CT
                                       # fired a tick late on those spawns and lost 128 of 1200 mirror rounds (lane E, 2026-10-11).
                                       # 1e-5 rad is 0.004 u at the arena's 400 u, far inside a head.
LANDS_TOL = 1e-5
WALL_MARGIN = 40.0                     # u
WALL_LOOKAHEAD_TICKS = 3
STYLES = ("plant", "stop", "run", "adad")
REACTION_MODES = ("recognition", "percept")


def _ticks(seconds: float) -> int:
    """Seconds at 16 Hz, rounded half up (ReGameDLL's ``int(t / interval + 0.5)``)."""
    return int(seconds * TICK_RATE + 0.5)


@dataclass(frozen=True)
class DuelBotParams:
    """One bot's dials (see the module docstring). The default is a perfect planted shooter:
    no delay, no aim error, lead on, OracleActor's habit with the turn-lands fire rule."""
    reaction_ticks: int = 0
    reaction_mode: str = "recognition"
    attack_delay_ticks: int = 0
    focus_initial_deg: float = 0.0
    focus_decay: float = 1.0           # factor per second
    focus_offset_scale: float = 0.0
    focus_interval_ticks: int = 1
    lead: bool = True
    style: str = "plant"
    strafe_ticks: int = 8

    def __post_init__(self):
        if not 0 <= self.reaction_ticks <= MAX_REACTION_TICKS:
            raise ValueError(f"need 0 <= reaction_ticks <= {MAX_REACTION_TICKS}, "
                             f"got {self.reaction_ticks}")
        if self.attack_delay_ticks < 0 or self.focus_interval_ticks < 1 or self.strafe_ticks < 1:
            raise ValueError(f"need attack_delay_ticks >= 0 and focus_interval_ticks, "
                             f"strafe_ticks >= 1: {self}")
        if self.focus_initial_deg < 0 or not 0 < self.focus_decay <= 1:
            raise ValueError(f"need focus_initial_deg >= 0 and 0 < focus_decay <= 1: {self}")
        if not 0 <= self.focus_offset_scale < 1:
            raise ValueError(f"need 0 <= focus_offset_scale < 1, got {self.focus_offset_scale}")
        if self.reaction_mode not in REACTION_MODES:
            raise ValueError(
                f"reaction_mode must be one of {REACTION_MODES}, got {self.reaction_mode!r}")
        if self.style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, got {self.style!r}")


def _profile(reaction_s, attack_delay_s, initial_deg, decay, offset_scale, interval_s):
    """A botprofile.db skill template at 16 Hz (seconds -> ticks, the rest as shipped)."""
    return DuelBotParams(reaction_ticks=_ticks(reaction_s),
                         attack_delay_ticks=_ticks(attack_delay_s),
                         focus_initial_deg=initial_deg,
                         focus_decay=decay,
                         focus_offset_scale=offset_scale,
                         focus_interval_ticks=max(_ticks(interval_s), 1))


# CS2's botprofile.db skill templates, best first: (ReactionTime s, AttackDelay s,
# AimFocusInitial deg, AimFocusDecay, AimFocusOffsetScale, AimfocusInterval s), with
# the Default's AttackDelay 0 where a template sets none. Style, lead and strafe are not
# profile dials: replace() them.
PROFILES = {
    "elite": _profile(0.05, 0.0, 0.5, 0.1, 0.05, 0.05),
    "expert": _profile(0.20, 0.0, 2.0, 0.2, 0.15, 0.20),
    "veryhard": _profile(0.30, 0.0, 5.0, 0.3, 0.17, 0.30),
    "hard": _profile(0.40, 0.0, 10.0, 0.4, 0.20, 0.40),
    "tough": _profile(0.50, 0.70, 10.0, 0.4, 0.25, 0.50),
    "normal": _profile(0.60, 0.80, 12.0, 0.5, 0.35, 0.60),
    "fair": _profile(0.60, 0.90, 17.0, 0.6, 0.5, 0.70),
    "easy": _profile(0.60, 0.70, 20.0, 0.7, 0.6, 0.70),
}


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _ground_velocity(vx, vy, move, facing, wishspeed):
    """The velocity after one ground tick of cs2_movement.h: friction, then acceleration
    toward the facing-local ``move`` bin (0 = no key), capped at ``wishspeed``."""
    speed = np.hypot(vx, vy)
    control = np.maximum(speed, SV_STOPSPEED)
    new = np.maximum(speed - control * SV_FRICTION * DT, 0.0)
    frac = np.where(speed > 0.0, new / np.where(speed > 0.0, speed, 1.0), 0.0)
    vx, vy = vx * frac, vy * frac
    fx, fy = _LOCAL_X[move], _LOCAL_Y[move]
    ca, sa = np.cos(facing), np.sin(facing)
    wx, wy = fy * ca + fx * sa, fy * sa - fx * ca
    add = wishspeed - (vx * wx + vy * wy)
    acc = np.where((move > 0) & (add > 0.0), np.minimum(SV_ACCELERATE * DT * wishspeed, add), 0.0)
    return vx + acc * wx, vy + acc * wy


class DuelBot:
    """The duel bot over ``n_rows`` rows (see the module docstring).

    ``params`` is one ``DuelBotParams`` for every row or a sequence of one per row;
    ``configure(rows, params)`` changes them later (a curriculum's hook). The map is
    given by its extent's half-sizes ``half_x``, ``half_y`` (u): the obs normalises
    positions by them and relative positions by their diagonal. ``aim_dz`` is the aim
    point's height above the eye for a standing pair (hit_model.hit_geometry), and
    ``move_speed`` the rifle's (binding.weapon_defs()). ``centre`` is the extent's centre in
    world coordinates, read only by ``step_state`` (the obs is already centred).

    State per row: the delay line of percepts, the previous enemy position (for its
    velocity), the focus, and the strafe. ``reset(rows)`` clears a row's state for a
    new episode; call it before the row's first step.
    """

    def __init__(self,
                 n_rows: int,
                 rng: np.random.Generator,
                 params,
                 *,
                 max_turn_speed: float,
                 half_x: float,
                 half_y: float,
                 aim_dz: float,
                 move_speed: float,
                 centre=(0.0, 0.0)):
        self.n_rows, self.rng = int(n_rows), rng
        self.centre = (float(centre[0]), float(centre[1]))
        self.mts, self.aim_dz, self.vmax = float(max_turn_speed), float(aim_dz), float(move_speed)
        self.half = (float(half_x), float(half_y))
        self.map_diag = float(np.hypot(half_x, half_y))
        n = self.n_rows
        self._p = {
            k: np.zeros(n)
            for k in ("react", "percept", "delay", "f0", "decay", "scale", "interval", "lead",
                      "style", "strafe")
        }
        self.configure(np.arange(n), params)
        # Delay line: per row, MAX_REACTION_TICKS + 1 percepts of (seen, ex, ey, ez, evx, evy).
        self._buf = np.zeros((n, MAX_REACTION_TICKS + 1, 6))
        self._w = np.zeros(n, dtype=np.int64)
        self._prev = np.zeros((n, 3))                  # previous tick's (seen, ex, ey), obs mode only
        self._engaged = np.zeros(n, dtype=bool)
        self._since = np.zeros(n, dtype=np.int64)      # ticks since acquisition
        self._offset = np.zeros((n, 2))                # focus offset (yaw, pitch), rad
        self._anchor = np.zeros(n)                     # bearing at the last focus draw
        self._redraw = np.zeros(n, dtype=np.int64)     # ticks to the next draw
        self._sign = np.ones(n)                        # strafe direction: +1 right, -1 left
        self._hold = np.zeros(n, dtype=np.int64)
        self.reset()

    def configure(self, rows, params) -> None:
        """Set ``rows``' dials: one ``DuelBotParams`` for all of them, or one per row."""
        idx = np.arange(self.n_rows)[rows]
        plist = [params] * len(idx) if isinstance(params, DuelBotParams) else list(params)
        if len(plist) != len(idx):
            raise ValueError(f"{len(plist)} params for {len(idx)} rows")
        p = self._p
        p["react"][idx] = [q.reaction_ticks for q in plist]
        p["percept"][idx] = [q.reaction_mode == "percept" for q in plist]
        p["delay"][idx] = [q.attack_delay_ticks for q in plist]
        p["f0"][idx] = np.radians([q.focus_initial_deg for q in plist])
        p["decay"][idx] = [q.focus_decay**DT for q in plist]
        p["scale"][idx] = [q.focus_offset_scale for q in plist]
        p["interval"][idx] = [q.focus_interval_ticks for q in plist]
        p["lead"][idx] = [float(q.lead) for q in plist]
        p["style"][idx] = [STYLES.index(q.style) for q in plist]
        p["strafe"][idx] = [q.strafe_ticks for q in plist]

    def reset(self, rows=None) -> None:
        """Start a new episode on ``rows`` (default every row)."""
        rows = slice(None) if rows is None else rows
        self._buf[rows] = 0.0
        self._w[rows] = 0
        self._prev[rows] = 0.0
        self._engaged[rows] = False
        self._sign[rows] = 1.0
        self._hold[rows] = self._p["strafe"][rows]

    # ── percepts ─────────────────────────────────────────────────────────────

    def step_obs(self, obs, mask):
        """One tick from the rows' observations ``obs`` (n, OBS_DIM) and action masks
        ``mask`` (n, ACTION_MASK_DIM): returns int32 (n, ACTION_DIM), float32 (n, AIM_DIM)."""
        o = np.asarray(obs, dtype=np.float64)
        n = self.n_rows
        hx, hy = self.half
        alive = o[:, SELF_ALIVE] > 0.5
        facing = np.arctan2(o[:, SELF_SIN_YAW], o[:, SELF_COS_YAW])
        sx, sy = o[:, SELF_X] * hx, o[:, SELF_Y] * hy
        svx, svy = o[:, SELF_VX] * OBS_VEL_SCALE, o[:, SELF_VY] * OBS_VEL_SCALE
        slots = o[:, _EN_BASES[:, None] + np.arange(OBS_ENEMY_STRIDE)] # (n, 5, 8)
        visible = (slots[:, :, EN_SEE] > 0.5) & (slots[:, :, EN_ALIVE] > 0.5)
        key = np.where(visible, slots[:, :, EN_DIST], np.inf)
        best = slots[np.arange(n), np.argmin(key, axis=1)]
        seen = np.isfinite(key.min(axis=1))
        rx, ry = best[:, EN_RX] * self.map_diag, best[:, EN_RY] * self.map_diag
        cf, sf = np.cos(facing), np.sin(facing)
        ex, ey = sx + rx * cf - ry * sf, sy + rx * sf + ry * cf
        ez = best[:, EN_DZ] * OBS_Z_SCALE
                                                                       # One tick's displacement is the post-step velocity (module PITFALLS).
        both = seen & (self._prev[:, 0] > 0.5)
        evx = np.where(both, (ex - self._prev[:, 1]) / DT, 0.0)
        evy = np.where(both, (ey - self._prev[:, 2]) / DT, 0.0)
        self._prev[:] = np.stack([seen, ex, ey], axis=1)
        m = np.asarray(mask)
        ready = m[:, MASK_SHOOT] != 0
        dry = (o[:, SELF_CLIP] <= 0.0) & (m[:, MASK_RELOAD] != 0)
        return self._decide(alive, seen, ex, ey, ez, evx, evy, sx, sy, svx, svy, facing, ready, dry)

    def step_state(self, st, rows, seen_by):
        """One tick from the ground truth: the ``StateReader`` snapshot ``st``, the env rows
        this bot plays, in its row order, and ``seen_by`` (N_AGENTS, N_AGENTS) bool, the
        visibility of the previous observation (``vis_from_obs``). Same return as step_obs.

        An instrument mode (the obs-parity check): the same decision on the same quantities,
        read off the C state. Weapon readiness is the action mask's rule (cs2_env.h
        compute_masks: the counters as they will be after next tick's decrement)."""
        rows = list(rows)
        hx, hy = self.half
        cx, cy = self.centre
        x, y, z = st["x"] - cx, st["y"] - cy, st["z"]
        team, alive_all = st["team"], st["alive"].astype(bool)
        n = self.n_rows
        out = {k: np.zeros(n) for k in ("ex", "ey", "ez", "evx", "evy")}
        seen = np.zeros(n, dtype=bool)
        for r, i in enumerate(rows):
            cands = [
                j for j in range(len(team)) if team[j] != team[i] and alive_all[j] and seen_by[i, j]
            ]
            if not cands:
                continue
            j = min(cands, key=lambda j: (x[j] - x[i])**2 + (y[j] - y[i])**2)
            seen[r] = True
            out["ex"][r], out["ey"][r], out["ez"][r] = x[j], y[j], z[j] - z[i]
            out["evx"][r], out["evy"][r] = st["vx"][j], st["vy"][j]
        i = np.asarray(rows)
        fire_cd, reload_t, switch_t = st["fire_cd"][i], st["reload_ticks"][i], st["switch_ticks"][i]
        clip = st["ammo_clip"][i]
        ready = (fire_cd <= 1) & (reload_t <= 1) & (switch_t <= 1) & (clip > 0)
        dry = (clip <= 0) & (reload_t == 0) & (switch_t == 0) & (st["ammo_reserve"][i] > 0)
        return self._decide(alive_all[i], seen, out["ex"], out["ey"], out["ez"], out["evx"],
                            out["evy"], x[i], y[i], st["vx"][i], st["vy"][i], st["facing"][i],
                            ready, dry)

    # ── the decision ─────────────────────────────────────────────────────────

    def _decide(self, alive, seen, ex, ey, ez, evx, evy, sx, sy, svx, svy, facing, ready, dry):
        n, p, ar = self.n_rows, self._p, np.arange(self.n_rows)
        L = MAX_REACTION_TICKS + 1
        now = np.stack([seen, ex, ey, ez, evx, evy], axis=1)
        self._buf[ar, self._w] = now
        old = self._buf[ar, (self._w - p["react"].astype(np.int64)) % L]
        self._w = (self._w + 1) % L
        percept = p["percept"] > 0.5
        engaged = alive & (old[:, 0] > 0.5) & (percept | seen)
        old = np.where(percept[:, None], old, now)     # what the aim reads
        acquired = engaged & ~self._engaged
        self._since = np.where(acquired, 0, self._since + 1)
        self._engaged = engaged
        style = p["style"].astype(np.int64)

        # Movement candidates: the strafe bin, the counter-strafe bin, and no key.
        self._hold -= 1
        flip = self._hold <= 0
        self._sign = np.where(flip, -self._sign, self._sign)
        self._hold = np.where(flip, p["strafe"].astype(np.int64), self._hold)
        strafe = np.where(self._sign > 0, STRAFE_RIGHT, STRAFE_LEFT)
        v_strafe = _ground_velocity(svx, svy, strafe, facing, self.vmax)
        far = WALL_LOOKAHEAD_TICKS * DT
        wall = ((np.abs(sx + v_strafe[0] * far) > self.half[0] - WALL_MARGIN)
                | (np.abs(sy + v_strafe[1] * far) > self.half[1] - WALL_MARGIN))
        self._sign = np.where(wall, -self._sign, self._sign)
        self._hold = np.where(wall, p["strafe"].astype(np.int64), self._hold)
        strafe = np.where(self._sign > 0, STRAFE_RIGHT, STRAFE_LEFT)
        v_strafe = _ground_velocity(svx, svy, strafe, facing, self.vmax)
        bins = np.arange(1, 9)
        wx = _LOCAL_Y[bins] * np.cos(facing)[:, None] + _LOCAL_X[bins] * np.sin(facing)[:, None]
        wy = _LOCAL_Y[bins] * np.sin(facing)[:, None] - _LOCAL_X[bins] * np.cos(facing)[:, None]
        counter = bins[np.argmin(wx * svx[:, None] + wy * svy[:, None], axis=1)]
        v_counter = _ground_velocity(svx, svy, counter, facing, self.vmax)
        v_free = _ground_velocity(svx, svy, np.zeros(n, dtype=np.int64), facing, self.vmax)
        use_counter = np.hypot(*v_counter) < np.hypot(*v_free)
        brake = np.where(use_counter, counter, 0)
        v_brake = (np.where(use_counter, v_counter[0],
                            v_free[0]), np.where(use_counter, v_counter[1], v_free[1]))

        plant, stop, run, adad = (style == k for k in range(4))
        braking = (stop & engaged) | (adad & engaged & ready)
        move = np.where(plant, 0, np.where(braking, brake, strafe))
        vx_next = np.where(plant, v_free[0], np.where(braking, v_brake[0], v_strafe[0]))
        vy_next = np.where(plant, v_free[1], np.where(braking, v_brake[1], v_strafe[1]))
        slow = np.hypot(vx_next, vy_next) < ACCURACY_THRESHOLD_FRAC * self.vmax

        # Aim: from the predicted post-move position at the delayed (optionally led) enemy.
        lead = p["lead"] * DT
        tx, ty = old[:, 1] + old[:, 4] * lead, old[:, 2] + old[:, 5] * lead
        px, py = sx + vx_next * DT, sy + vy_next * DT
        bearing = np.arctan2(ty - py, tx - px)
        dist = np.hypot(tx - px, ty - py)
        elev = np.arctan2(old[:, 3] + self.aim_dz, np.maximum(dist, 1e-6))

        # Focus: draw at acquisition and every interval; F decays from acquisition.
        self._redraw = np.where(acquired, 0, self._redraw - 1)
        draw = engaged & (self._redraw <= 0)
        k = int(draw.sum())
        if k:
            spread = p["f0"][draw] * p["decay"][draw]**self._since[draw]
            self._offset[draw] = (2.0 * self.rng.random((k, 2)) - 1.0) * spread[:, None]
            self._anchor[draw] = bearing[draw]
            self._redraw[draw] = p["interval"][draw].astype(np.int64)
        aim = bearing + self._offset[:, 0] - p["scale"] * _wrap(bearing - self._anchor)
        dyaw = _wrap(aim - facing)
        lands = np.abs(dyaw) <= self.mts + LANDS_TOL

        fire = (engaged & ready & lands & (self._since >= p["delay"])
                & ((plant | run) | ((stop | adad) & slow)))
        self._sign = np.where(adad & fire, -self._sign, self._sign)    # ADAD: the next strafe reverses

        act = np.zeros((n, ACTION_DIM), dtype=np.int32)
        cont = np.zeros((n, AIM_DIM), dtype=np.float32)
        act[:, H_MOVE] = np.where(alive, move, 0)
        act[:, H_SHOOT] = fire
        act[:, H_RELOAD] = alive & ~fire & dry
        cont[:, 0] = np.where(engaged, np.clip(dyaw, -self.mts, self.mts), 0.0)
        cont[:, 1] = np.where(engaged, np.clip(elev + self._offset[:, 1], -np.pi / 2, np.pi / 2),
                              0.0)
        return act, cont


class DuelBotActor:
    """A ``DuelBot`` over the given env rows, in the eval.baselines actor shape.

    ``act`` returns (N_AGENTS, ACTION_DIM) int32 and (N_AGENTS, AIM_DIM) float32 with only
    ``rows`` filled, so another actor's rows are untouched. ``obs_only`` (default) decides
    from ``obs[rows]`` and the env's action masks, as a trainer would; False reads the C
    state (``DuelBot.step_state``), the instrument mode the obs-only bot is compared with.

    The core is built on the first ``act``, from the env it is handed there (the
    harnesses build their env after their actors): the turn clamp, the map's extent, the
    hit model's aim height and the rifle's move speed.
    """

    name = "duel-bot"

    def __init__(self, rows, params, rng: np.random.Generator, obs_only: bool = True):
        self.rows = list(rows)
        self.params, self.rng, self.obs_only = params, rng, bool(obs_only)
        self.core = None

    def _build(self, env) -> DuelBot:
        from cs2rl.env.c import binding
        from cs2rl.env.c.hit_model import hit_geometry
        sd = env._c_env.sd.contents
        geo = hit_geometry(env.config.hitboxes)
        half_x, half_y = 1.0 / sd.inv_x_range, 1.0 / sd.inv_y_range
        return DuelBot(len(self.rows),
                       self.rng,
                       self.params,
                       max_turn_speed=sd.max_turn_speed,
                       half_x=half_x,
                       half_y=half_y,
                       aim_dz=geo.target_z(0.0, False) - geo.eye_z(0.0, False),
                       move_speed=binding.weapon_defs()[0]["move_speed"],
                       centre=(sd.x_offset * half_x, sd.y_offset * half_y))

    def reset(self):
        if self.core is not None:
            self.core.reset()

    def act(self, obs, st, vis_prev, env):
        if self.core is None:
            self.core = self._build(env)
        n = len(obs)
        act = np.zeros((n, ACTION_DIM), dtype=np.int32)
        cont = np.zeros((n, AIM_DIM), dtype=np.float32)
        if self.obs_only:
            a, c = self.core.step_obs(obs[self.rows], env._masks_view[self.rows])
        else:
            seen = vis_prev if vis_prev is not None else np.zeros((n, n), dtype=bool)
            a, c = self.core.step_state(st, self.rows, seen)
        act[self.rows], cont[self.rows] = a, c
        return act, cont
