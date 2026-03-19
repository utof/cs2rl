import ctypes
from dataclasses import dataclass
import numpy as np
import subprocess
import gymnasium
import pufferlib
from pathlib import Path

_DIR = Path(__file__).parent
_SO  = _DIR / "dust2_env.so"
_SRC = _DIR / "dust2_env.c"
_HDR = _DIR / "dust2_env.h"

def _ensure_built():
    newest_src = max(_SRC.stat().st_mtime, _HDR.stat().st_mtime)
    if not _SO.exists() or _SO.stat().st_mtime < newest_src:
        subprocess.run(["make", "-C", str(_DIR)], check=True)

_ensure_built()
_lib = ctypes.CDLL(str(_SO))

N_AGENTS = 10
OBS_DIM  = 71
ACTION_DIM = 4
_STATIC_DATA_CACHE = {}


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


class StaticDataC(ctypes.Structure):
    _fields_ = [
        ("N",               ctypes.c_int),
        ("vis_matrix",      ctypes.POINTER(ctypes.c_int8)),
        ("raster_grid",     ctypes.POINTER(ctypes.c_int32)),
        ("adjacency",       ctypes.POINTER(ctypes.c_int8)),
        ("centroid_xy",     ctypes.POINTER(ctypes.c_float)),
        ("area_ids",        ctypes.POINTER(ctypes.c_int32)),
        ("bombsite_mask",   ctypes.POINTER(ctypes.c_int8)),
        ("bombsite_by_idx", ctypes.POINTER(ctypes.c_int8)),
        ("bombsite_dist",   ctypes.POINTER(ctypes.c_float)),
        ("grid_w",          ctypes.c_int),
        ("grid_h",          ctypes.c_int),
        ("max_area_id",     ctypes.c_int),
        ("grid_x_min",      ctypes.c_float),
        ("grid_y_min",      ctypes.c_float),
        ("grid_inv_cell",   ctypes.c_float),
        ("inv_x_range",     ctypes.c_float),
        ("inv_y_range",     ctypes.c_float),
        ("x_offset",        ctypes.c_float),
        ("y_offset",        ctypes.c_float),
        ("bombsite_dist_scale", ctypes.c_float),
        ("laser_damage",    ctypes.c_int32),
        ("laser_range",     ctypes.c_float),
        ("laser_range_sq",  ctypes.c_float),
        ("shoot_cooldown",  ctypes.c_int32),
        ("bomb_plant_time", ctypes.c_int32),
        ("bomb_defuse_time", ctypes.c_int32),
        ("bomb_defuse_kit", ctypes.c_int32),
        ("bomb_timer",      ctypes.c_int32),
        ("round_time",      ctypes.c_int32),
        ("footstep_radius_sq", ctypes.c_float),
        ("gunshot_radius_sq", ctypes.c_float),
        ("enemy_memory_ticks", ctypes.c_int32),
        ("stale_memory_tick", ctypes.c_int32),
        ("pbrs_gamma",      ctypes.c_float),
        ("delta_x",         ctypes.c_float * 9),
        ("delta_y",         ctypes.c_float * 9),
        ("dir_facing",      ctypes.c_float * 9),
        ("t_spawns",        ctypes.c_int32 * 15),
        ("n_t_spawns",      ctypes.c_int),
        ("ct_spawns",       ctypes.c_int32 * 5),
        ("n_ct_spawns",     ctypes.c_int),
    ]


class AgentStateC(ctypes.Structure):
    _fields_ = [
        ("x",               ctypes.c_float),
        ("y",               ctypes.c_float),
        ("z",               ctypes.c_float),
        ("area_idx",        ctypes.c_int32),
        ("facing",          ctypes.c_float),
        ("hp",              ctypes.c_int32),
        ("shoot_cd",        ctypes.c_int32),
        ("alive",           ctypes.c_int8),
        ("has_bomb",        ctypes.c_int8),
        ("has_kit",         ctypes.c_int8),
        ("team",            ctypes.c_int8),
        ("is_moving",       ctypes.c_int8),
        ("fired_this_tick", ctypes.c_int8),
        ("enemy_mem_idx",   ctypes.c_int32 * 5),
        ("enemy_mem_tick",  ctypes.c_int32 * 5),
    ]


class GameStateC(ctypes.Structure):
    _fields_ = [
        ("tick",                   ctypes.c_int32),
        ("round_ticks_left",       ctypes.c_int32),
        ("agents",                 AgentStateC * 10),
        ("bomb_planted",           ctypes.c_int8),
        ("round_over",             ctypes.c_int8),
        ("winner",                 ctypes.c_int32),
        ("bomb_carrier_id",        ctypes.c_int32),
        ("bomb_area_idx",          ctypes.c_int32),
        ("bomb_x",                 ctypes.c_float),
        ("bomb_y",                 ctypes.c_float),
        ("bomb_z",                 ctypes.c_float),
        ("bomb_ticks_left",        ctypes.c_int32),
        ("bomb_being_planted_by",  ctypes.c_int32),
        ("bomb_plant_ticks",       ctypes.c_int32),
        ("bomb_being_defused_by",  ctypes.c_int32),
        ("bomb_defuse_ticks",      ctypes.c_int32),
    ]


class StepStatsC(ctypes.Structure):
    _fields_ = [
        ("bomb_planted",    ctypes.c_int32),
        ("bomb_defused",    ctypes.c_int32),
        ("kills_t",         ctypes.c_int32),
        ("kills_ct",        ctypes.c_int32),
        ("blocked_moves_t", ctypes.c_int32),
        ("blocked_moves_ct", ctypes.c_int32),
        ("winner",          ctypes.c_int32),
        ("winner_t",        ctypes.c_int32),
        ("winner_ct",       ctypes.c_int32),
        ("timed_out",       ctypes.c_int32),
        ("alive_t_end",     ctypes.c_int32),
        ("alive_ct_end",    ctypes.c_int32),
        ("round_length",    ctypes.c_int32),
        ("action_move",     ctypes.c_int32 * 9),
        ("action_shoot",    ctypes.c_int32 * 2),
        ("action_use",      ctypes.c_int32 * 2),
        ("action_last",     ctypes.c_int32 * 2),
    ]


class Dust2EnvC(ctypes.Structure):
    _fields_ = [
        ("sd",           ctypes.POINTER(StaticDataC)),
        ("game",         GameStateC),
        ("step_stats",   StepStatsC),
        ("episode_stats", StepStatsC),
        ("observations", ctypes.c_float * (N_AGENTS * OBS_DIM)),
        ("rewards",      ctypes.c_float * N_AGENTS),
        ("terminals",    ctypes.c_int8  * N_AGENTS),
        ("truncations",  ctypes.c_int8  * N_AGENTS),
        ("team_spirit",  ctypes.c_float),
        ("rng",          ctypes.c_uint32),
    ]


# Sanity-check struct sizes match the C layout — catches future drift early
assert ctypes.sizeof(AgentStateC) == 76, \
    f"AgentStateC size mismatch: {ctypes.sizeof(AgentStateC)} (expected 76)"
assert ctypes.sizeof(GameStateC) == 816, \
    f"GameStateC size mismatch: {ctypes.sizeof(GameStateC)} (expected 816)"

_lib.env_init.argtypes  = [ctypes.POINTER(Dust2EnvC), ctypes.POINTER(StaticDataC),
                            ctypes.c_uint32, ctypes.c_float]
_lib.env_init.restype   = None
_lib.env_reset.argtypes = [ctypes.POINTER(Dust2EnvC)]
_lib.env_reset.restype  = None
_lib.env_step.argtypes  = [ctypes.POINTER(Dust2EnvC), ctypes.c_void_p]
_lib.env_step.restype   = None
_lib.env_close.argtypes = [ctypes.POINTER(Dust2EnvC)]
_lib.env_close.restype  = None


def build_static_data(nav_graph, area_adjacency, bombsite_mask,
                      t_spawn_areas, ct_spawn_areas,
                      bombsite_distance_lookup, bombsite_distance_scale):
    """Build StaticDataC from Python nav data. Returns (sd, refs).
    Caller must keep refs alive to prevent GC of backing numpy arrays."""
    import sim
    sd   = StaticDataC()
    refs = []

    def ptr(arr, dtype, ctype):
        a = np.ascontiguousarray(arr.astype(dtype, copy=False).flatten())
        refs.append(a)
        return a.ctypes.data_as(ctypes.POINTER(ctype))

    sd.N              = nav_graph.N
    sd.vis_matrix     = ptr(nav_graph.vis_matrix,        np.int8,    ctypes.c_int8)
    sd.raster_grid    = ptr(nav_graph._pos_grid,         np.int32,   ctypes.c_int32)
    sd.adjacency      = ptr(area_adjacency,              np.int8,    ctypes.c_int8)
    sd.centroid_xy    = ptr(nav_graph._centroid_matrix,  np.float32, ctypes.c_float)

    area_ids_arr      = np.array(nav_graph.area_ids, dtype=np.int32)
    sd.area_ids       = ptr(area_ids_arr,                np.int32,   ctypes.c_int32)

    bm = bombsite_mask.astype(np.int8)
    sd.bombsite_mask  = ptr(bm,                          np.int8,    ctypes.c_int8)
    sd.bombsite_dist  = ptr(bombsite_distance_lookup,    np.float32, ctypes.c_float)

    by_idx = np.array([
        int(bm[aid]) if 0 <= aid < len(bm) else 0
        for aid in nav_graph.area_ids
    ], dtype=np.int8)
    sd.bombsite_by_idx = ptr(by_idx,                     np.int8,    ctypes.c_int8)

    sd.grid_w         = nav_graph._grid_w
    sd.grid_h         = nav_graph._grid_h
    sd.max_area_id    = max(nav_graph.area_ids)
    sd.grid_x_min     = float(nav_graph._grid_x_min)
    sd.grid_y_min     = float(nav_graph._grid_y_min)
    sd.grid_inv_cell  = float(nav_graph._grid_inv_cell)

    sd.inv_x_range    = float(sim._INV_MAP_X_RANGE)
    sd.inv_y_range    = float(sim._INV_MAP_Y_RANGE)
    sd.x_offset       = float(sim._MAP_X_OFFSET)
    sd.y_offset       = float(sim._MAP_Y_OFFSET)
    sd.bombsite_dist_scale = float(bombsite_distance_scale)
    sd.laser_damage   = int(sim.LASER_DAMAGE)
    sd.laser_range    = float(sim.LASER_RANGE)
    sd.laser_range_sq = float(sim.LASER_RANGE * sim.LASER_RANGE)
    sd.shoot_cooldown = int(sim.SHOOT_COOLDOWN)
    sd.bomb_plant_time = int(sim.BOMB_PLANT_TIME)
    sd.bomb_defuse_time = int(sim.BOMB_DEFUSE_TIME)
    sd.bomb_defuse_kit = int(sim.BOMB_DEFUSE_KIT)
    sd.bomb_timer     = int(sim.BOMB_TIMER)
    sd.round_time     = int(sim.ROUND_TIME)
    sd.footstep_radius_sq = float(sim.FOOTSTEP_RADIUS * sim.FOOTSTEP_RADIUS)
    sd.gunshot_radius_sq = float(sim.GUNSHOT_RADIUS * sim.GUNSHOT_RADIUS)
    sd.enemy_memory_ticks = int(sim.ENEMY_MEMORY_TICKS)
    sd.stale_memory_tick = int(sim.STALE_MEMORY_TICK)
    sd.pbrs_gamma     = float(0.99)
    for i in range(9):
        delta = sim._DELTA_VECTORS[i]
        sd.delta_x[i] = float(delta[0])
        sd.delta_y[i] = float(delta[1])
        sd.dir_facing[i] = float(sim._DIR_FACING[i])

    id2idx = nav_graph._id_to_idx
    sd.n_t_spawns = len(t_spawn_areas)
    assert len(t_spawn_areas) <= 15, f"t_spawn_areas overflow: {len(t_spawn_areas)} > 15"
    for i, aid in enumerate(t_spawn_areas):
        sd.t_spawns[i] = id2idx[aid]

    sd.n_ct_spawns = len(ct_spawn_areas)
    assert len(ct_spawn_areas) <= 5, f"ct_spawn_areas overflow: {len(ct_spawn_areas)} > 5"
    for i, aid in enumerate(ct_spawn_areas):
        sd.ct_spawns[i] = id2idx[aid]

    return sd, refs


class Dust2CEnv(pufferlib.PufferEnv):
    def __init__(self, sd, refs, seed=0, team_spirit=0.0, buf=None, nav_graph=None, auto_reset=True):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([9, 2, 2, 2])
        self.num_agents = N_AGENTS
        super().__init__(buf)

        self._sd    = sd
        self._refs  = refs   # keep alive — prevents GC of numpy backing arrays
        self.nav_graph = nav_graph
        self._auto_reset = bool(auto_reset)
        self._uses_external_buffers = buf is not None
        self._c_env = Dust2EnvC()
        self._c_env_p = ctypes.byref(self._c_env)
        self._team_spirit_shared = team_spirit if hasattr(team_spirit, "value") else None
        if self._team_spirit_shared is not None:
            init_team_spirit = float(team_spirit.value)
        elif team_spirit is None:
            init_team_spirit = 0.0
        else:
            init_team_spirit = float(team_spirit)
        _lib.env_init(self._c_env_p, ctypes.byref(self._sd),
                      ctypes.c_uint32(seed), ctypes.c_float(init_team_spirit))

        # Pre-create numpy views of C buffers — avoids recreating each step
        self._obs_view  = np.frombuffer(self._c_env.observations, dtype=np.float32).reshape(N_AGENTS, OBS_DIM)
        self._rew_view  = np.frombuffer(self._c_env.rewards,      dtype=np.float32)
        self._term_view = np.frombuffer(self._c_env.terminals,    dtype=np.bool_)
        self._trunc_view = np.frombuffer(self._c_env.truncations, dtype=np.bool_)
        if not self._uses_external_buffers:
            self.observations = self._obs_view
            self.rewards = self._rew_view
            self.terminals = self._term_view
            self.truncations = self._trunc_view
        self._actions_shape = (N_AGENTS, ACTION_DIM)
        self._actions_scratch = np.zeros(self._actions_shape, dtype=np.int32)
        self._actions_scratch_addr = int(self._actions_scratch.ctypes.data)
        self._terminal_rewards = np.empty(N_AGENTS, dtype=np.float32)
        self._terminal_terminals = np.empty(N_AGENTS, dtype=bool)
        self._terminal_truncations = np.empty(N_AGENTS, dtype=bool)
        self._empty_infos = []

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None):
        # seed is accepted for API compatibility but C RNG is set at env_init time
        self._sync_team_spirit()
        _lib.env_reset(self._c_env_p)
        self._sync_outputs()
        return self.observations, self._empty_infos

    def step(self, actions):
        self._sync_team_spirit()
        _actions_ref, actions_p = self._prepare_actions(actions)
        _lib.env_step(self._c_env_p, actions_p)
        self._sync_outputs()
        infos = self._empty_infos
        rewards = self.rewards
        terminals = self.terminals
        truncations = self.truncations
        if bool(self._c_env.game.round_over):
            summary = self._build_terminal_info()
            infos = [summary]
            if self._auto_reset:
                if not self._uses_external_buffers:
                    np.copyto(self._terminal_rewards, self._rew_view)
                    np.copyto(self._terminal_terminals, self._term_view)
                    np.copyto(self._terminal_truncations, self._trunc_view)
                    rewards = self._terminal_rewards
                    terminals = self._terminal_terminals
                    truncations = self._terminal_truncations
                _lib.env_reset(self._c_env_p)
                self._sync_observations()
        return self.observations, rewards, terminals, truncations, infos

    def set_team_spirit(self, value: float):
        self._c_env.team_spirit = float(value)

    def close(self):
        _lib.env_close(self._c_env_p)

    @property
    def round_time(self):
        return int(self._sd.round_time)

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
                )
            )
        return VizGameState(
            agents=agents,
            bomb_planted=bool(g.bomb_planted),
            bomb_pos=np.array([g.bomb_x, g.bomb_y, g.bomb_z], dtype=np.float32),
        )

    def _sync_team_spirit(self):
        if self._team_spirit_shared is not None:
            self._c_env.team_spirit = float(self._team_spirit_shared.value)

    def _prepare_actions(self, actions):
        if (
            isinstance(actions, np.ndarray)
            and actions.dtype == np.int32
            and actions.shape == self._actions_shape
            and actions.flags.c_contiguous
        ):
            return actions, int(actions.ctypes.data)

        actions_arr = np.asarray(actions, dtype=np.int32)
        if actions_arr.shape != self._actions_shape:
            actions_arr = actions_arr.reshape(self._actions_shape)
        if actions_arr.flags.c_contiguous:
            return actions_arr, int(actions_arr.ctypes.data)

        np.copyto(self._actions_scratch, actions_arr, casting="no")
        return self._actions_scratch, self._actions_scratch_addr

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
        summary.update(
            {
                "winner": int(stats.winner),
                "winner_t": int(stats.winner_t),
                "winner_ct": int(stats.winner_ct),
                "timed_out": int(stats.timed_out),
                "alive_t_end": int(stats.alive_t_end),
                "alive_ct_end": int(stats.alive_ct_end),
                "round_length": int(stats.round_length),
            }
        )
        return summary


def _load_c_static_bundle():
    import sim

    key = (sim.NAV_PATH, sim.CACHE_PATH)
    bundle = _STATIC_DATA_CACHE.get(key)
    if bundle is not None:
        return bundle

    static = sim._load_dust2_static_data(*key)
    nav = static["nav_graph"]
    sd, refs = build_static_data(
        nav,
        static["area_adjacency"],
        static["bombsite_mask"],
        static["t_spawn_areas"],
        static["ct_spawn_areas"],
        static["bombsite_distance_lookup"],
        static["bombsite_distance_scale"],
    )
    bundle = (sd, tuple(refs), nav)
    _STATIC_DATA_CACHE[key] = bundle
    return bundle


def make_env(seed=0, team_spirit=0.0, auto_reset=True, buf=None):
    """Load static data and return a ready-to-use Dust2CEnv."""
    sd, refs, nav = _load_c_static_bundle()
    return Dust2CEnv(
        sd,
        refs,
        seed=seed,
        team_spirit=team_spirit,
        buf=buf,
        nav_graph=nav,
        auto_reset=auto_reset,
    )
