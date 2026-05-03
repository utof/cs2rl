# src/c_env/cs2_env.py — PufferEnv subclass backed by binding.c C API bridge.

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

import gymnasium
import numpy as np
import pufferlib

import nav
from _action_spec import ACTION_DIM, ACTION_HEAD_SIZES, ACTION_MASK_DIM, AIM_DIM
from map import make_cs2_map
from nav import N_AGENTS, OBS_DIM, ROUND_TIME

_DIR = Path(__file__).parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
import binding                         # noqa: E402

# ── Viz dataclasses (used by snapshot_state) ─────────────────────────────────


@dataclass
class VizAgentState:
    agent_id: int
    team: int
    pos: np.ndarray
    facing: float
    hp: int
    alive: bool
    has_bomb: bool
    has_kit: bool


@dataclass
class VizGameState:
    agents: list
    bomb_planted: bool
    bomb_pos: np.ndarray


# ── ctypes struct definitions (read/write overlay for _c_env) ─────────────────


class StaticDataC(ctypes.Structure):
    # fmt: off  -- YAPF aligns standalone comments to trailing-comment column; suppress here
    _fields_ = [
        ("N", ctypes.c_int),
        ("vis_matrix", ctypes.POINTER(ctypes.c_int8)),
        ("raster_grid", ctypes.POINTER(ctypes.c_int32)),
        ("adjacency", ctypes.POINTER(ctypes.c_int8)),
        ("centroid_xy", ctypes.POINTER(ctypes.c_float)),
        # T2 (verticality): per-area terrain elevation and ramp flag.
        # Field order MUST stay in sync with:
        #   - StaticData struct in cs2_types.h  (C canonical source)
        #   - PyArg_ParseTuple format string in binding.c py_init()
        # Mismatch here silently corrupts all pointer fields that follow.
        ("centroids_z", ctypes.POINTER(ctypes.c_float)),               # float32[N] — terrain z per area  # noqa: E501
        ("area_ids", ctypes.POINTER(ctypes.c_int32)),
        ("bombsite_mask", ctypes.POINTER(ctypes.c_int8)),
        ("bombsite_by_idx", ctypes.POINTER(ctypes.c_int8)),
        # is_ramp: MapData bool is converted to int8 in Cs2Env.__init__ before passing
        ("is_ramp", ctypes.POINTER(ctypes.c_int8)),                    # int8[N] — 1=ramp/stairs
        ("bombsite_dist", ctypes.POINTER(ctypes.c_float)),
        ("grid_w", ctypes.c_int),
        ("grid_h", ctypes.c_int),
        ("max_area_id", ctypes.c_int),
        ("grid_x_min", ctypes.c_float),
        ("grid_y_min", ctypes.c_float),
        ("grid_inv_cell", ctypes.c_float),
        ("inv_x_range", ctypes.c_float),
        ("inv_y_range", ctypes.c_float),
        ("x_offset", ctypes.c_float),
        ("y_offset", ctypes.c_float),
        ("bombsite_dist_scale", ctypes.c_float),
        ("laser_damage", ctypes.c_int32),
        ("laser_range", ctypes.c_float),
        ("laser_range_sq", ctypes.c_float),
        ("shoot_cooldown", ctypes.c_int32),
        ("bomb_plant_time", ctypes.c_int32),
        ("bomb_defuse_time", ctypes.c_int32),
        ("bomb_defuse_kit", ctypes.c_int32),
        ("bomb_timer", ctypes.c_int32),
        ("round_time", ctypes.c_int32),
        ("footstep_radius_sq", ctypes.c_float),
        ("gunshot_radius_sq", ctypes.c_float),
        ("enemy_memory_ticks", ctypes.c_int32),
        ("stale_memory_tick", ctypes.c_int32),
        ("pbrs_gamma", ctypes.c_float),
        ("delta_x", ctypes.c_float * 9),
        ("delta_y", ctypes.c_float * 9),
        ("dir_facing", ctypes.c_float * 9),
        ("t_spawns", ctypes.c_int32 * 15),
        ("n_t_spawns", ctypes.c_int),
        ("ct_spawns", ctypes.c_int32 * 5),
        ("n_ct_spawns", ctypes.c_int),
        ("max_turn_speed", ctypes.c_float),
        # legacy symmetric — superseded by per-mechanism fields (Batch 1)
        ("reward_win", ctypes.c_float),
        # Batch 1 (RL overhaul): per-outcome win magnitudes (Task 3).
        # Must stay in same order as StaticData in cs2_types.h.
        ("reward_win_t_detonation", ctypes.c_float),                   # default 5.0
        ("reward_win_t_elimination", ctypes.c_float),                  # default 3.0
        ("reward_win_ct_defuse", ctypes.c_float),                      # default 5.0
        ("reward_win_ct_timeout", ctypes.c_float),                     # default 4.0
        ("reward_win_ct_elimination", ctypes.c_float),                 # default 3.0
        ("reward_kill", ctypes.c_float),
        ("reward_death", ctypes.c_float),
        ("reward_bombsite_entry", ctypes.c_float),
        ("reward_plant_bonus", ctypes.c_float),
        ("reward_plant_base", ctypes.c_float),
        ("reward_plant_progress_scale", ctypes.c_float),
        ("reward_plant_interrupted", ctypes.c_float),
        ("reward_defuse", ctypes.c_float),
        ("reward_shot_penalty", ctypes.c_float),
        ("reward_ct_survival", ctypes.c_float),
        ("reward_inaction", ctypes.c_float),
        ("pbrs_alive_weight", ctypes.c_float),
        ("pbrs_hp_weight", ctypes.c_float),
        ("pbrs_site_weight", ctypes.c_float),
        ("pbrs_bomb_progress_weight", ctypes.c_float),
        ("pbrs_nav_weight_t", ctypes.c_float),
        ("pbrs_nav_weight_ct", ctypes.c_float),
    ]
    # fmt: on


class AgentStateC(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("area_idx", ctypes.c_int32),
        ("facing", ctypes.c_float),
        ("hp", ctypes.c_int32),
        ("fire_cd", ctypes.c_int32),
        ("alive", ctypes.c_int8),
        ("has_bomb", ctypes.c_int8),
        ("has_kit", ctypes.c_int8),
        ("team", ctypes.c_int8),
        ("is_moving", ctypes.c_int8),
        ("fired_this_tick", ctypes.c_int8),
        ("_pad0", ctypes.c_int8 * 2),
        ("enemy_mem_idx", ctypes.c_int32 * 5),
        ("enemy_mem_tick", ctypes.c_int32 * 5),
                                                       # Phase 4b additions
        ("vx", ctypes.c_float),
        ("vy", ctypes.c_float),
        ("is_crouching", ctypes.c_int8),
        ("_pad1", ctypes.c_int8 * 3),
        ("crouch_cd", ctypes.c_int32),
        ("armor", ctypes.c_int32),
        ("has_helmet", ctypes.c_int8),
        ("weapon_slot", ctypes.c_int8),
        ("weapon_slot_target", ctypes.c_int8),
        ("_pad2", ctypes.c_int8 * 1),
        ("ammo_clip", ctypes.c_int32 * 3),
        ("ammo_reserve", ctypes.c_int32 * 3),
        ("reload_ticks", ctypes.c_int32),
        ("switch_ticks", ctypes.c_int32),
                                                       # Phase 6 additions
        ("human_controlled", ctypes.c_uint8),
        ("_pad3", ctypes.c_int8 * 3),
        ("aim_rad", ctypes.c_float),
                                                       # Phase 7 additions (jump)
        ("vz", ctypes.c_float),
        ("is_airborne", ctypes.c_int8),
        ("_pad4", ctypes.c_int8 * 3),
        ("jump_cd", ctypes.c_int32),
                                                       # Batch 3.5: pitch — appended, mirror cs2_types.h AgentState.
        ("pitch", ctypes.c_float),
    ]


class GameStateC(ctypes.Structure):
    _fields_ = [
        ("tick", ctypes.c_int32),
        ("round_ticks_left", ctypes.c_int32),
        ("agents", AgentStateC * 10),
        ("bomb_planted", ctypes.c_int8),
        ("round_over", ctypes.c_int8),
        ("winner", ctypes.c_int32),
        ("bomb_carrier_id", ctypes.c_int32),
        ("bomb_area_idx", ctypes.c_int32),
        ("bomb_x", ctypes.c_float),
        ("bomb_y", ctypes.c_float),
        ("bomb_z", ctypes.c_float),
        ("bomb_ticks_left", ctypes.c_int32),
        ("bomb_being_planted_by", ctypes.c_int32),
        ("bomb_plant_ticks", ctypes.c_int32),
        ("bomb_being_defused_by", ctypes.c_int32),
        ("bomb_defuse_ticks", ctypes.c_int32),
                                                                       # Batch 2: round-fixed designated carrier — mirrors C GameState.  # noqa: E501
                                                                       # See cs2_types.h for why/pitfalls; insertion order matters for alignment.  # noqa: E501
        ("round_designated_carrier_id", ctypes.c_int32),
        ("bombsite_entered", ctypes.c_int8 * 5),
        ("bomb_is_dropped", ctypes.c_int8),
        ("_pad_gs", ctypes.c_int8 * 2),
    ]


class StepStatsC(ctypes.Structure):
    _fields_ = [
        ("bomb_planted", ctypes.c_int32),
        ("bomb_defused", ctypes.c_int32),
        ("kills_t", ctypes.c_int32),
        ("kills_ct", ctypes.c_int32),
        ("blocked_moves_t", ctypes.c_int32),
        ("blocked_moves_ct", ctypes.c_int32),
        ("winner", ctypes.c_int32),
        ("winner_t", ctypes.c_int32),
        ("winner_ct", ctypes.c_int32),
        ("timed_out", ctypes.c_int32),
        ("alive_t_end", ctypes.c_int32),
        ("alive_ct_end", ctypes.c_int32),
        ("round_length", ctypes.c_int32),
        ("action_move", ctypes.c_int32 * 9),
        ("action_shoot", ctypes.c_int32 * 2),
        ("action_use", ctypes.c_int32 * 2),
        ("action_last", ctypes.c_int32 * 2),
                                                       # Batch 3: continuous-aim Δyaw stats (mirror C StepStats fields).  # noqa: E501
                                                       # Replaces the 16-bin action_aim histogram (64B) with a Welford-style  # noqa: E501
                                                       # triple (sum + sq_sum + count = 12B). No explicit _pad_aim_delta —  # noqa: E501
                                                       # the three int32-aligned fields slot in cleanly between action_last  # noqa: E501
                                                       # and action_reload. See cs2_types.h StepStats comment.  # noqa: E501
        ("aim_delta_sum", ctypes.c_float),
        ("aim_delta_sq_sum", ctypes.c_float),
        ("aim_delta_count", ctypes.c_int32),
                                                       # Batch 3.5: pitch Welford triple (mirror cs2_types.h StepStats fields).
        ("aim_delta_pitch_sum", ctypes.c_float),
        ("aim_delta_pitch_sq_sum", ctypes.c_float),
        ("aim_delta_pitch_count", ctypes.c_int32),
        ("action_reload", ctypes.c_int32 * 2),
        ("action_weapon", ctypes.c_int32 * 3),
        ("action_crouch", ctypes.c_int32 * 2),
        ("action_jump", ctypes.c_int32 * 2),
        ("reward_win", ctypes.c_float),
        ("reward_kills", ctypes.c_float),
        ("reward_deaths", ctypes.c_float),
        ("reward_bomb", ctypes.c_float),
        ("reward_pbrs", ctypes.c_float),
        ("reward_shots", ctypes.c_float),
        ("reward_survival", ctypes.c_float),
        ("reward_inaction", ctypes.c_float),
                                                       # Batch 1 (RL overhaul): round-end win classification flags.  # noqa: E501
                                                       # Cleared by round_reset (Task 2). Set by compute_rewards (Task 3).  # noqa: E501
                                                       # Consumed by split_into_channels to route reward_win:  # noqa: E501
                                                       #   detonation/defuse → objective channel; else → combat channel.  # noqa: E501
        ("win_by_detonation", ctypes.c_int8),          # 1 when bomb detonated (T wins)
        ("win_by_defuse", ctypes.c_int8),              # 1 when bomb was defused (CT wins)
        ("_pad_ss_wins", ctypes.c_int8 * 2),           # pad to 4-byte boundary
    ]


class StepStatsView:
    """Dict-like zero-copy view over a ctypes StepStatsC struct.

    Cs2Env.step() returns a reference to this wrapper in info[0]["step_stats"]
    when include_step_stats_in_info=True. The wrapper proxies __getitem__ to
    attribute access on the underlying struct, so downstream consumers (e.g.
    split_into_channels in src/train_helpers_batch1.py) can read fields by
    name with zero per-tick allocation.

    ndim is set to 0 so split_into_channels's length-1 squeeze (which only
    fires for ndim==1 structured arrays) does not trigger on this wrapper.
    """
    __slots__ = ("_ss", )
    ndim = 0                           # class attr so split_into_channels's ndim==1 check is False

    def __init__(self, ss):
        self._ss = ss

    def __getitem__(self, key):
        return getattr(self._ss, key)

    def get(self, key, default=None):
        return getattr(self._ss, key, default)


class Dust2EnvC(ctypes.Structure):
    _fields_ = [
        ("sd", ctypes.POINTER(StaticDataC)),
        ("game", GameStateC),
        ("step_stats", StepStatsC),
        ("episode_stats", StepStatsC),
        ("observations", ctypes.c_float * (N_AGENTS * OBS_DIM)),
        ("rewards", ctypes.c_float * N_AGENTS),
        ("terminals", ctypes.c_int8 * N_AGENTS),
        ("truncations", ctypes.c_int8 * N_AGENTS),
        ("team_spirit", ctypes.c_float),
        ("rng", ctypes.c_uint32),
        ("masks", ctypes.c_int8 * (N_AGENTS * ACTION_MASK_DIM)),
        ("client", ctypes.c_void_p),                                   # Client* (NULL in training)
    ]


# Sanity-check struct sizes match the C layout — catches future drift early.
# Sizes updated for Phase 7 (jump): AgentState +12 bytes (vz, is_airborne,
# pad, jump_cd), StepStats +8 bytes (action_jump[2]). GameState rolls up
# the agent-array delta (10×12=120). Dust2EnvC rolls up game + 2× stats +
# 20 bytes of added mask slots + alignment.
# Batch 1 (RL overhaul): StepStats +4 bytes (win_by_detonation, win_by_defuse,
# _pad_ss_wins[2]). Dust2EnvC +8 bytes (2× StepStats).
# Batch 2 task 1: GameState +4 from round_designated_carrier_id (int32).
# Dust2EnvC grows by +8 (not +4): the extra 4 bytes from `game` push the
# trailing `client` void* pointer past an 8-byte alignment boundary, so the
# C compiler inserts a 4-byte pad before `client`, giving a net +8 for
# Dust2EnvC.
# Batch 2 task 2: OBS_DIM 104 → 105 — Dust2EnvC `observations` array
# (c_float * (10 * 105)) is +40 bytes vs task 1.
# Batch 3: StepStats −52 bytes (action_aim[16]=64B → aim_delta_*=12B).
# Dust2EnvC −264 bytes nominal: 2× StepStats (−104) + masks shrink
# (10×38→10×22 = −160). Verify empirically on first build — alignment
# surprises are routine; values updated below to match observed sizeof.
# Batch 3.5: AgentStateC +4 (float pitch), GameStateC +40 (×10 agents),
# Dust2EnvC +40 (GameState) +80 (observations: 10×(107−105)×4).
assert ctypes.sizeof(AgentStateC) == 156, (
    f"AgentStateC size mismatch: {ctypes.sizeof(AgentStateC)} (expected 156)")
assert ctypes.sizeof(GameStateC) == 1628, (
    f"GameStateC size mismatch: {ctypes.sizeof(GameStateC)} (expected 1628)")
assert ctypes.sizeof(StepStatsC) == 208, (
    f"StepStatsC size mismatch: {ctypes.sizeof(StepStatsC)} (expected 208)")
assert ctypes.sizeof(Dust2EnvC) == 6632, (
    f"Dust2EnvC size mismatch: {ctypes.sizeof(Dust2EnvC)} (expected 6632)")

# ctypes helper to extract raw pointer from PyCapsule
_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

# Module-level cache for map data
_ENV_CACHE: dict = {}

# ── Cs2Env ────────────────────────────────────────────────────────────────────


class Cs2Env(pufferlib.PufferEnv):

    def __init__(
            self,
            seed=0,
            team_spirit=0.0,
            buf=None,
            nav_graph=None,
            auto_reset=True,
            map_data=None,
            reward_win=1.0,
            reward_kill=0.3,
            reward_death=0.1,
            reward_bombsite_entry=0.3,
            reward_plant_bonus=3.0,
            reward_plant_base=0.2,
            reward_plant_progress_scale=0.05,
            reward_plant_interrupted=0.1,
            reward_defuse=0.2,
            reward_shot_penalty=0.005,
            reward_ct_survival=0.001,
            reward_inaction=0.0005,
            pbrs_alive_weight=0.3,
            pbrs_hp_weight=0.002,
            pbrs_site_weight=0.2,
            pbrs_bomb_progress_weight=0.3,
            pbrs_nav_weight_t=0.04,
            pbrs_nav_weight_ct=0.15,
            pbrs_gamma=0.99,
                                                                                # Batch 1 (RL overhaul): per-outcome win magnitudes.  # noqa: E501
                                                                                # These supersede the symmetric reward_win at round end.  # noqa: E501
                                                                                # Defaults chosen to make detonation/defuse > timeout > elimination.  # noqa: E501
            reward_win_t_detonation=5.0,
            reward_win_t_elimination=3.0,
            reward_win_ct_defuse=5.0,
            reward_win_ct_timeout=4.0,
            reward_win_ct_elimination=3.0,
            include_step_stats_in_info: bool = False,                           # Task 6a (utof/cs2rl#7)  # noqa: E501
    ):
        self.single_observation_space = gymnasium.spaces.Box(low=-5.0,
                                                             high=5.0,
                                                             shape=(OBS_DIM, ),
                                                             dtype=np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete(list(ACTION_HEAD_SIZES))
        self.num_agents = N_AGENTS
        super().__init__(buf)

        self.nav_graph = nav_graph
        self.map_data = map_data
        self._auto_reset = bool(auto_reset)
        self._uses_external_buffers = buf is not None

        # team_spirit may be float or multiprocessing.Value
        self._team_spirit_shared = team_spirit if hasattr(team_spirit, "value") else None
        if self._team_spirit_shared is not None:
            init_team_spirit = float(team_spirit.value)
        elif team_spirit is None:
            init_team_spirit = 0.0
        else:
            init_team_spirit = float(team_spirit)

        md = map_data

        def _arr(a, dtype):
            return np.ascontiguousarray(a.astype(dtype, copy=False).flatten())

        vis_matrix = _arr(md.vis_matrix, np.int8)
        raster_grid = _arr(md.grid, np.int32)
        adjacency = _arr(md.adjacency, np.int8)
        centroid_xy = _arr(md.centroids, np.float32)
        # T2 (verticality): centroids_z must be float32 (C side reads as float*).
        # _arr coerces dtype via astype, but if MapData ever passes the wrong dtype
        # we want a loud error here, not silent coercion poisoning the C pointer.
        centroids_z = _arr(md.centroids_z, np.float32)
        if centroids_z.dtype != np.float32:
            raise TypeError(f"centroids_z float32 conversion failed: dtype={centroids_z.dtype}; "
                            "this would dangle the C-side sd->centroids_z pointer")
        area_ids = _arr(md.area_ids, np.int32)
        bombsite_mask = _arr(md.bombsite_mask, np.int8)
        bombsite_by_idx = _arr(md.bombsite_by_idx, np.int8)
        # T2 (verticality): is_ramp is bool in MapData but C reads int8*.
        # Convert explicitly — numpy bool layout is platform-dependent.
        # Pitfall: do NOT pass md.is_ramp directly — bool dtype may not
        # be 1-byte on all platforms; int8 is guaranteed portable.
        # np.ascontiguousarray guards against stride surprises post-astype.
        is_ramp_int8 = np.ascontiguousarray(md.is_ramp.astype(np.int8))
        if is_ramp_int8.dtype != np.int8 or is_ramp_int8.itemsize != 1:
            raise ValueError(f"is_ramp int8 conversion failed: dtype={is_ramp_int8.dtype}, "
                             f"itemsize={is_ramp_int8.itemsize}; "
                             "this would dangle the C-side sd->is_ramp pointer")
        bombsite_dist = _arr(md.bombsite_dist, np.float32)

        inv_x = 2.0 / (md.x_max - md.x_min)
        inv_y = 2.0 / (md.y_max - md.y_min)
        x_off = (md.x_max + md.x_min) / (md.x_max - md.x_min)
        y_off = (md.y_max + md.y_min) / (md.y_max - md.y_min)

        id2idx = {int(aid): i for i, aid in enumerate(md.area_ids)}
        t_spawns = np.array([id2idx[a] for a in md.t_spawn_areas], dtype=np.int32)
        ct_spawns = np.array([id2idx[a] for a in md.ct_spawn_areas], dtype=np.int32)
        delta_x = np.array([float(nav._DELTA_VECTORS[k][0]) for k in range(9)], dtype=np.float32)
        delta_y = np.array([float(nav._DELTA_VECTORS[k][1]) for k in range(9)], dtype=np.float32)
        dir_facing = np.array([float(nav._DIR_FACING[k]) for k in range(9)], dtype=np.float32)

        # Keep refs alive — prevents GC of backing numpy arrays.
        # T2 (verticality): centroids_z and is_ramp_int8 added here so the C pointers
        # sd->centroids_z and sd->is_ramp remain valid for the env's lifetime.
        # is_ramp_int8 is a NEW array (result of .astype); it would be collected
        # immediately if not held here — the C pointer would then dangle.
        self._refs = [
            vis_matrix,
            raster_grid,
            adjacency,
            centroid_xy,
            centroids_z,
            area_ids,
            bombsite_mask,
            bombsite_by_idx,
            is_ramp_int8,
            bombsite_dist,
            t_spawns,
            ct_spawns,
            delta_x,
            delta_y,
            dir_facing,
        ]

        # Call binding.init() — positional order matches C format string.
        # T2 (verticality): centroids_z inserted at pos 4 (after centroid_xy);
        # is_ramp_int8 inserted at pos 8 (after bombsite_by_idx).
        # CRITICAL: positions must stay in sync with StaticDataC._fields_ in this file
        # and StaticData in cs2_types.h — mismatch silently corrupts pointer assignments.
        self._capsule = binding.init(
            vis_matrix,
            raster_grid,
            adjacency,
            centroid_xy,                                               # 0-3
            centroids_z,                                               # 4: T2 terrain z per area
            area_ids,
            bombsite_mask,
            bombsite_by_idx,
            is_ramp_int8,                                              # 8: T2 ramp (int8)
            bombsite_dist,                                             # 9
            int(md.N),
            int(md.grid.shape[1]),
            int(md.grid.shape[0]),                                     # 10-12: N, grid_w, grid_h
            int(md.area_ids.max()),                                    # 13: max_area_id
            float(md.grid_x_min),
            float(md.grid_y_min),                                      # 14-15
            float(1.0 / md.grid_cell_size),                            # 16: grid_inv_cell
            float(inv_x),
            float(inv_y),
            float(x_off),
            float(y_off),                                              # 17-20
            float(md.bombsite_dist_scale),                             # 21
            int(nav.LASER_DAMAGE),                                     # 22
            float(nav.LASER_RANGE),
            float(nav.LASER_RANGE * nav.LASER_RANGE),                  # 23-24
            int(nav.SHOOT_COOLDOWN),
            int(nav.BOMB_PLANT_TIME),                                  # 25-26
            int(nav.BOMB_DEFUSE_TIME),
            int(nav.BOMB_DEFUSE_KIT),                                  # 27-28
            int(nav.BOMB_TIMER),
            int(nav.ROUND_TIME),                                       # 29-30
            float(nav.FOOTSTEP_RADIUS * nav.FOOTSTEP_RADIUS),          # 31
            float(nav.GUNSHOT_RADIUS * nav.GUNSHOT_RADIUS),            # 32
            int(nav.ENEMY_MEMORY_TICKS),
            int(nav.STALE_MEMORY_TICK),                                # 33-34
            float(pbrs_gamma),                                         # 35: pbrs_gamma
            delta_x,
            delta_y,
            dir_facing,
            t_spawns,                                                  # 36-39
            int(len(md.t_spawn_areas)),                                # 40: n_t_spawns
            ct_spawns,                                                 # 41
            int(len(md.ct_spawn_areas)),                               # 42: n_ct_spawns
            float(nav.MAX_TURN_SPEED_RAD),                             # 43
            int(seed) & 0xFFFFFFFF,                                    # 44: seed (uint32)
            float(init_team_spirit),                                   # 45
            float(reward_win),                                         # 46
            float(reward_win_t_detonation),                            # 47: Batch 1 per-mechanism
            float(reward_win_t_elimination),                           # 48
            float(reward_win_ct_defuse),                               # 49
            float(reward_win_ct_timeout),                              # 50
            float(reward_win_ct_elimination),                          # 51
            float(reward_kill),                                        # 52
            float(reward_death),                                       # 53
            float(reward_bombsite_entry),                              # 54
            float(reward_plant_bonus),                                 # 55
            float(reward_plant_base),                                  # 56
            float(reward_plant_progress_scale),                        # 57
            float(reward_plant_interrupted),                           # 58
            float(reward_defuse),                                      # 59
            float(reward_shot_penalty),                                # 60
            float(reward_ct_survival),                                 # 61
            float(reward_inaction),                                    # 62
            float(pbrs_alive_weight),                                  # 63
            float(pbrs_hp_weight),                                     # 64
            float(pbrs_site_weight),                                   # 65
            float(pbrs_bomb_progress_weight),                          # 66
            float(pbrs_nav_weight_t),                                  # 67
            float(pbrs_nav_weight_ct),                                 # 68
        )

        # ctypes overlay of the C-allocated Dust2Env (tests + snapshot only)
        # BindingEnv has env as first field, so capsule ptr == &env
        env_ptr = _PyCapsule_GetPointer(self._capsule, None)
        self._c_env = Dust2EnvC.from_address(env_ptr)

        # Zero-copy NumPy views into C buffers
        obs_ptr, rew_ptr, term_ptr, trunc_ptr = binding.get_buffers(self._capsule)
        masks_ptr = binding.get_masks(self._capsule)
        self._masks_view = np.frombuffer(
            (ctypes.c_int8 * (N_AGENTS * ACTION_MASK_DIM)).from_address(masks_ptr),
            dtype=np.int8,
        ).reshape(N_AGENTS, ACTION_MASK_DIM)
        self._obs_view = np.frombuffer(
            (ctypes.c_float * (N_AGENTS * OBS_DIM)).from_address(obs_ptr),
            dtype=np.float32,
        ).reshape(N_AGENTS, OBS_DIM)
        self._rew_view = np.frombuffer((ctypes.c_float * N_AGENTS).from_address(rew_ptr),
                                       dtype=np.float32)
        self._term_view = np.frombuffer((ctypes.c_bool * N_AGENTS).from_address(term_ptr),
                                        dtype=np.bool_)
        self._trunc_view = np.frombuffer((ctypes.c_bool * N_AGENTS).from_address(trunc_ptr),
                                         dtype=np.bool_)

        if not self._uses_external_buffers:
            self.observations = self._obs_view
            self.rewards = self._rew_view
            self.terminals = self._term_view
            self.truncations = self._trunc_view

        self._actions_shape = (N_AGENTS, ACTION_DIM)
        self._actions_scratch = np.zeros(self._actions_shape, dtype=np.int32)
        # Batch 3: continuous-aim Δyaw scratch buffer (N_AGENTS, AIM_DIM=1) float32.
        # Reused per-step when the caller passes None (default zero) or a
        # non-contiguous / wrong-dtype array. Owning the scratch here means we
        # never allocate on the hot path; copy-into is the worst case.
        self._cont_actions_shape = (N_AGENTS, AIM_DIM)
        self._cont_actions_scratch = np.zeros(self._cont_actions_shape, dtype=np.float32)
        # Batch 3 (T5b): Optional shared-memory view onto a parent-process
        # cont-action buffer. Set by _attach_cont_action_view() when the env
        # is constructed by the Multiprocessing vecenv path; the worker reads
        # whatever the trainer wrote into the shared RawArray BEFORE step.
        # When set, _prepare_continuous_actions(None) returns this view
        # instead of the (always-zero) scratch — so MP workers actually see
        # the policy's Δyaw sample. Serial backend continues to bypass this
        # via the per-env step wrapper and a Python attr stash on the
        # vecenv (see _patch_trainer_with_hybrid_aim in src/train.py).
        self._cont_action_view = None
        # Hold the parent-process RawArray to keep it from being GC'd if the
        # caller passes it transiently (it is also kept alive on the trainer
        # side, but defensive double-anchoring is cheap).
        self._cont_action_shm_ref = None
        self._terminal_rewards = np.empty(N_AGENTS, dtype=np.float32)
        self._terminal_terminals = np.empty(N_AGENTS, dtype=bool)
        self._terminal_truncations = np.empty(N_AGENTS, dtype=bool)
        self._empty_infos = []
        self._include_step_stats_in_info = bool(include_step_stats_in_info)
        # Task 6a: optional per-tick step_stats view in info (see utof/cs2rl#7).
        # Zero cost when flag is off; constructed once at init when flag is on.
        # _nonterminal_infos is a pre-built list[dict] reused every non-terminal
        # step to avoid per-tick allocation — the view object is a stable proxy
        # over the ctypes struct, so the same reference is safe to return each tick.
        if self._include_step_stats_in_info:
            self._step_stats_view = StepStatsView(self._c_env.step_stats)
            self._nonterminal_infos = [{"step_stats": self._step_stats_view}]
        else:
            self._step_stats_view = None
            self._nonterminal_infos = self._empty_infos

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None):
        # seed accepted for API compatibility; C RNG is set at init time
        self._sync_team_spirit()
        binding.reset(self._capsule)
        self._sync_outputs()
        return self.observations, self._empty_infos

    def step(self, actions, continuous_actions=None):
        """Step the env one tick.

        Batch 3: actions is (N_AGENTS, ACTION_DIM=7) int32 — discrete heads.
        continuous_actions is (N_AGENTS, AIM_DIM=1) float32 — Δyaw rad. If
        None (legacy callers, tests, render path), a zero buffer is supplied
        so RL agents do not turn. Wrong shape raises ValueError before the
        C call to prevent OOB reads.

        Pitfall: shape strictness is critical — the C side does no bounds
        check on continuous_actions[i*AIM_DIM+0]; a (10,2) buffer would
        silently read garbage from a misaligned slot.
        """
        self._sync_team_spirit()
        actions_arr = self._prepare_actions(actions)
        cont_arr = self._prepare_continuous_actions(continuous_actions)
        binding.step(self._capsule, actions_arr, cont_arr)
        self._sync_outputs()
        infos = self._empty_infos
        rewards = self.rewards
        terminals = self.terminals
        truncations = self.truncations
        if bool(self._c_env.game.round_over):
            summary = self._build_terminal_info()
            if self._include_step_stats_in_info:
                # Task 6a: merge step_stats into the terminal summary dict so
                # consumers see both round-end stats and per-tick stats in one dict.
                summary["step_stats"] = self._step_stats_view
            infos = [summary]
            if self._auto_reset:
                if not self._uses_external_buffers:
                    np.copyto(self._terminal_rewards, self._rew_view)
                    np.copyto(self._terminal_terminals, self._term_view)
                    np.copyto(self._terminal_truncations, self._trunc_view)
                    rewards = self._terminal_rewards
                    terminals = self._terminal_terminals
                    truncations = self._terminal_truncations
                binding.reset(self._capsule)
                self._sync_observations()
        elif self._include_step_stats_in_info:
            # Task 6a: non-terminal tick — return the pre-built singleton info list
            # (no per-tick allocation; the StepStatsView proxies the live struct).
            infos = self._nonterminal_infos
        return self.observations, rewards, terminals, truncations, infos

    def set_team_spirit(self, value: float):
        self._c_env.team_spirit = float(value)

    def close(self):
        binding.close(self._capsule)

    @property
    def round_time(self):
        return ROUND_TIME

    def snapshot_state(self):
        g = self._c_env.game
        agents = []
        for i in range(N_AGENTS):
            agent = g.agents[i]
            agents.append(
                VizAgentState(
                    agent_id=i,
                    team=int(agent.team),
                    pos=np.array([agent.x, agent.y, agent.z], dtype=np.float32),
                    facing=float(agent.facing),
                    hp=int(agent.hp),
                    alive=bool(agent.alive),
                    has_bomb=bool(agent.has_bomb),
                    has_kit=bool(agent.has_kit),
                ))
        return VizGameState(
            agents=agents,
            bomb_planted=bool(g.bomb_planted),
            bomb_pos=np.array([g.bomb_x, g.bomb_y, g.bomb_z], dtype=np.float32),
        )

    def _sync_team_spirit(self):
        if self._team_spirit_shared is not None:
            self._c_env.team_spirit = float(self._team_spirit_shared.value)

    def _prepare_actions(self, actions):
        if (isinstance(actions, np.ndarray) and actions.dtype == np.int32
                and actions.shape == self._actions_shape and actions.flags.c_contiguous):
            return actions

        actions_arr = np.asarray(actions, dtype=np.int32)
        if actions_arr.shape != self._actions_shape:
            actions_arr = actions_arr.reshape(self._actions_shape)
        if actions_arr.flags.c_contiguous:
            return actions_arr

        np.copyto(self._actions_scratch, actions_arr, casting="no")
        return self._actions_scratch

    def _attach_cont_action_view(self, raw_shm, env_idx):
        """Attach a numpy view onto a parent-process RawArray slice (Batch 3 T5b).

        Called by the env factory when the Multiprocessing vecenv path is
        live. The trainer allocates a single ``multiprocessing.RawArray('f',
        num_envs * N_AGENTS * AIM_DIM)`` BEFORE forking workers; each worker
        receives the same raw buffer (forked file mapping, no pipe transfer)
        and we carve out this env's per-agent slice as a numpy view.

        After attach, ``_prepare_continuous_actions(None)`` returns this view
        instead of the all-zero scratch — so the trainer can write Δyaw into
        the shm slice on the main process and the worker's ``Cs2Env.step``
        will see it on the very next tick.

        Pitfall: ``raw_shm`` MUST have at least
        ``(env_idx + 1) * N_AGENTS * AIM_DIM`` float32 slots. We do NOT
        revalidate the global length here (we don't know num_envs); a too-
        small allocation will manifest as np.frombuffer raising or as
        out-of-range data. The caller (env factory in src/train.py) owns
        the sizing.
        """
        if raw_shm is None:
            return
        per_env = N_AGENTS * AIM_DIM
        # ctypes float = 4 bytes; np.frombuffer with offset/count addresses
        # the slice without copying. np.frombuffer DOES set the resulting
        # array's .base to raw_shm (so the buffer is technically reachable),
        # but env_factory's local reference to raw_shm goes out of scope
        # right after this method returns. Keeping _cont_action_shm_ref as
        # an explicit anchor on self protects against future code that might
        # replace _cont_action_view (e.g. with a reshape or a slice) and
        # accidentally flatten the .base chain — at which point the OS could
        # reclaim the mapping and the next read would be UB.
        flat = np.frombuffer(raw_shm, dtype=np.float32, count=per_env, offset=env_idx * per_env * 4)
        self._cont_action_view = flat.reshape(self._cont_actions_shape)
        self._cont_action_shm_ref = raw_shm

    def _prepare_continuous_actions(self, cont):
        """Coerce caller-supplied Δyaw buffer to (N_AGENTS, AIM_DIM) float32 contiguous.

        Batch 3: shape mismatch ALWAYS raises (silent reshape would conceal a
        bug given AIM_DIM=1 — e.g. a (10,) accidentally passed as (1,10) would
        slip through). dtype mismatch is forgiven via cast. None → cached
        zero scratch (RL training path before the policy is wired uses this),
        OR if ``_attach_cont_action_view`` has installed a shared-memory
        view (Batch 3 T5b — Multiprocessing vecenv path), that view is
        returned instead so workers actually consume the trainer's Δyaw.
        """
        if cont is None:
            if self._cont_action_view is not None:
                # MP vecenv path: trainer wrote into the parent-process shm;
                # the view is the workers' window onto the same physical
                # bytes. Returning the view directly means the C binding
                # reads the live data on this tick.
                return self._cont_action_view
            # Reuse the pre-zeroed scratch — zeroing every step would be wasteful.
            # Tests that mutate this scratch via env.step(...) MUST pass an
            # explicit buffer; the scratch is treated as read-only zeroes here.
            return self._cont_actions_scratch
        if not isinstance(cont, np.ndarray):
            cont = np.asarray(cont, dtype=np.float32)
        if cont.shape != self._cont_actions_shape:
            raise ValueError(f"continuous_actions shape {cont.shape} != "
                             f"{self._cont_actions_shape} (expected (N_AGENTS, AIM_DIM))")
        if cont.dtype == np.float32 and cont.flags.c_contiguous:
            return cont
        # Fall through: cast/copy into scratch. casting="unsafe" allows
        # float64 → float32 (RL policies sometimes emit float32 already, but
        # numpy generic dtypes from `np.zeros((10,1))` default to float64).
        np.copyto(self._cont_actions_scratch, cont.astype(np.float32, copy=False), casting="unsafe")
        return self._cont_actions_scratch

    def _sync_outputs(self):
        if not self._uses_external_buffers:
            return
        self._sync_observations()
        np.copyto(self.rewards, self._rew_view)
        np.copyto(self.terminals, self._term_view)
        np.copyto(self.truncations, self._trunc_view)

    def _sync_observations(self):
        if self._uses_external_buffers:
            np.copyto(self.observations, self._obs_view)

    def _build_terminal_info(self):
        stats = self._c_env.episode_stats
        summary = {
            "bomb_planted": int(stats.bomb_planted),
            "bomb_defused": int(stats.bomb_defused),
            "kills_t": int(stats.kills_t),
            "kills_ct": int(stats.kills_ct),
            "blocked_moves_t": int(stats.blocked_moves_t),
            "blocked_moves_ct": int(stats.blocked_moves_ct),
        }
        for idx in range(9):
            summary[f"action_move_{idx}"] = int(stats.action_move[idx])
        for idx in range(2):
            summary[f"action_shoot_{idx}"] = int(stats.action_shoot[idx])
            summary[f"action_use_{idx}"] = int(stats.action_use[idx])
            summary[f"action_last_{idx}"] = int(stats.action_last[idx])
        # Batch 3: continuous-aim stats — emit Welford triple instead of
        # 16-bin histogram. Consumers that previously summed action_aim_*
        # to derive total turns should now use aim_delta_count; mean/var
        # via standard formulas. mean = sum / count;
        # var = sq_sum / count - mean².
        summary["aim_delta_sum"] = float(stats.aim_delta_sum)
        summary["aim_delta_sq_sum"] = float(stats.aim_delta_sq_sum)
        summary["aim_delta_count"] = int(stats.aim_delta_count)
        # Batch 3.5 (#24): pitch Welford triple (mirrors yaw fields above).
        # Consumers compute mean/var/std the same way: mean = sum / count;
        # var = sq_sum / count - mean²; std = sqrt(max(0, var)).
        # Pitch_log_std non-collapse is the spec's load-bearing acceptance
        # signal — diagnostics need their own surface.
        summary["aim_delta_pitch_sum"] = float(stats.aim_delta_pitch_sum)
        summary["aim_delta_pitch_sq_sum"] = float(stats.aim_delta_pitch_sq_sum)
        summary["aim_delta_pitch_count"] = int(stats.aim_delta_pitch_count)
        for idx in range(2):
            summary[f"action_reload_{idx}"] = int(stats.action_reload[idx])
        for idx in range(3):
            summary[f"action_weapon_{idx}"] = int(stats.action_weapon[idx])
        for idx in range(2):
            summary[f"action_crouch_{idx}"] = int(stats.action_crouch[idx])
        for idx in range(2):
            summary[f"action_jump_{idx}"] = int(stats.action_jump[idx])
        summary.update({
            "winner": int(stats.winner),
            "winner_t": int(stats.winner_t),
            "winner_ct": int(stats.winner_ct),
            "timed_out": int(stats.timed_out),
            "alive_t_end": int(stats.alive_t_end),
            "alive_ct_end": int(stats.alive_ct_end),
            "round_length": int(stats.round_length),
        })
        summary.update({
            "reward_win": float(stats.reward_win),
            "reward_kills": float(stats.reward_kills),
            "reward_deaths": float(stats.reward_deaths),
            "reward_bomb": float(stats.reward_bomb),
            "reward_pbrs": float(stats.reward_pbrs),
            "reward_shots": float(stats.reward_shots),
            "reward_survival": float(stats.reward_survival),
            "reward_inaction": float(stats.reward_inaction),
        })
        return summary


# ── make_env ──────────────────────────────────────────────────────────────────


def make_env(
        seed=0,
        team_spirit=0.0,
        auto_reset=True,
        buf=None,
        map_data=None,
        reward_win=1.0,
        reward_kill=0.3,
        reward_death=0.1,
        reward_bombsite_entry=0.3,
        reward_plant_bonus=3.0,
        reward_plant_base=0.2,
        reward_plant_progress_scale=0.05,
        reward_plant_interrupted=0.1,
        reward_defuse=0.2,
        reward_shot_penalty=0.005,
        reward_ct_survival=0.001,
        reward_inaction=0.0005,
        pbrs_alive_weight=0.3,
        pbrs_hp_weight=0.002,
        pbrs_site_weight=0.2,
        pbrs_bomb_progress_weight=0.3,
        pbrs_nav_weight_t=0.04,
        pbrs_nav_weight_ct=0.15,
        pbrs_gamma=0.99,
                                                                       # Batch 1 (RL overhaul): per-outcome win magnitudes (Task 3).  # noqa: E501
        reward_win_t_detonation=5.0,
        reward_win_t_elimination=3.0,
        reward_win_ct_defuse=5.0,
        reward_win_ct_timeout=4.0,
        reward_win_ct_elimination=3.0,
        include_step_stats_in_info: bool = False,                      # Task 6a (utof/cs2rl#7)
):
    """Load map data and return a ready-to-use Cs2Env."""
    if map_data is None:
        key = (nav.NAV_PATH, nav.CACHE_PATH)
        md = _ENV_CACHE.get(key)
        if md is None:
            md = make_cs2_map(nav.NAV_PATH, nav.CACHE_PATH)
            _ENV_CACHE[key] = md
    else:
        md = map_data
    return Cs2Env(
        seed=seed,
        team_spirit=team_spirit,
        buf=buf,
        nav_graph=md.nav_graph,
        auto_reset=auto_reset,
        map_data=md,
        reward_win=reward_win,
        reward_kill=reward_kill,
        reward_death=reward_death,
        reward_bombsite_entry=reward_bombsite_entry,
        reward_plant_bonus=reward_plant_bonus,
        reward_plant_base=reward_plant_base,
        reward_plant_progress_scale=reward_plant_progress_scale,
        reward_plant_interrupted=reward_plant_interrupted,
        reward_defuse=reward_defuse,
        reward_shot_penalty=reward_shot_penalty,
        reward_ct_survival=reward_ct_survival,
        reward_inaction=reward_inaction,
        pbrs_alive_weight=pbrs_alive_weight,
        pbrs_hp_weight=pbrs_hp_weight,
        pbrs_site_weight=pbrs_site_weight,
        pbrs_bomb_progress_weight=pbrs_bomb_progress_weight,
        pbrs_nav_weight_t=pbrs_nav_weight_t,
        pbrs_nav_weight_ct=pbrs_nav_weight_ct,
        pbrs_gamma=pbrs_gamma,
        reward_win_t_detonation=reward_win_t_detonation,
        reward_win_t_elimination=reward_win_t_elimination,
        reward_win_ct_defuse=reward_win_ct_defuse,
        reward_win_ct_timeout=reward_win_ct_timeout,
        reward_win_ct_elimination=reward_win_ct_elimination,
        include_step_stats_in_info=include_step_stats_in_info,
    )
