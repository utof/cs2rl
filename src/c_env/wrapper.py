import ctypes
import subprocess
from dataclasses import dataclass
from pathlib import Path

import gymnasium
import numpy as np
import pufferlib

import sim
from map import MapData, make_cs2_map
from sim import ACTION_DIM, N_AGENTS, OBS_DIM

_DIR = Path(__file__).parent
_SO = _DIR / "dust2_env.so"
_SRC = _DIR / "dust2_env.c"
_HDR = _DIR / "dust2_env.h"


def _ensure_built():
    newest_src = max(_SRC.stat().st_mtime, _HDR.stat().st_mtime)
    if not _SO.exists() or _SO.stat().st_mtime < newest_src:
        subprocess.run(["make", "-C", str(_DIR)], check=True)


_ensure_built()
_lib = ctypes.CDLL(str(_SO))

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
        ("N", ctypes.c_int),
        ("vis_matrix", ctypes.POINTER(ctypes.c_int8)),
        ("raster_grid", ctypes.POINTER(ctypes.c_int32)),
        ("adjacency", ctypes.POINTER(ctypes.c_int8)),
        ("centroid_xy", ctypes.POINTER(ctypes.c_float)),
        ("area_ids", ctypes.POINTER(ctypes.c_int32)),
        ("bombsite_mask", ctypes.POINTER(ctypes.c_int8)),
        ("bombsite_by_idx", ctypes.POINTER(ctypes.c_int8)),
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
    ]


class AgentStateC(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("area_idx", ctypes.c_int32),
        ("facing", ctypes.c_float),
        ("hp", ctypes.c_int32),
        ("shoot_cd", ctypes.c_int32),
        ("alive", ctypes.c_int8),
        ("has_bomb", ctypes.c_int8),
        ("has_kit", ctypes.c_int8),
        ("team", ctypes.c_int8),
        ("is_moving", ctypes.c_int8),
        ("fired_this_tick", ctypes.c_int8),
        ("enemy_mem_idx", ctypes.c_int32 * 5),
        ("enemy_mem_tick", ctypes.c_int32 * 5),
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
        ("bombsite_entered", ctypes.c_int8 * 5),
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
    ]


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
    ]


# Sanity-check struct sizes match the C layout — catches future drift early
assert ctypes.sizeof(AgentStateC) == 76, (
    f"AgentStateC size mismatch: {ctypes.sizeof(AgentStateC)} (expected 76)"
)
assert ctypes.sizeof(GameStateC) == 824, (
    f"GameStateC size mismatch: {ctypes.sizeof(GameStateC)} (expected 824)"
)

_lib.env_init.argtypes = [
    ctypes.POINTER(Dust2EnvC),
    ctypes.POINTER(StaticDataC),
    ctypes.c_uint32,
    ctypes.c_float,
]
_lib.env_init.restype = None
_lib.env_reset.argtypes = [ctypes.POINTER(Dust2EnvC)]
_lib.env_reset.restype = None
_lib.env_step.argtypes = [ctypes.POINTER(Dust2EnvC), ctypes.c_void_p]
_lib.env_step.restype = None
_lib.env_close.argtypes = [ctypes.POINTER(Dust2EnvC)]
_lib.env_close.restype = None


def build_static_data(map_data: MapData):
    """Build StaticDataC from a MapData. Returns (sd, refs).
    Caller must keep refs alive to prevent GC of backing numpy arrays."""
    sd = StaticDataC()
    refs = []

    def ptr(arr, dtype, ctype):
        a = np.ascontiguousarray(arr.astype(dtype, copy=False).flatten())
        refs.append(a)
        return a.ctypes.data_as(ctypes.POINTER(ctype))

    sd.N = map_data.N
    sd.vis_matrix = ptr(map_data.vis_matrix, np.int8, ctypes.c_int8)
    sd.raster_grid = ptr(map_data.grid, np.int32, ctypes.c_int32)
    sd.adjacency = ptr(map_data.adjacency, np.int8, ctypes.c_int8)
    sd.centroid_xy = ptr(map_data.centroids, np.float32, ctypes.c_float)
    sd.area_ids = ptr(map_data.area_ids, np.int32, ctypes.c_int32)
    sd.bombsite_mask = ptr(map_data.bombsite_mask, np.int8, ctypes.c_int8)
    sd.bombsite_by_idx = ptr(map_data.bombsite_by_idx, np.int8, ctypes.c_int8)
    sd.bombsite_dist = ptr(map_data.bombsite_dist, np.float32, ctypes.c_float)

    sd.grid_w = map_data.grid.shape[1]
    sd.grid_h = map_data.grid.shape[0]
    sd.max_area_id = int(map_data.area_ids.max())
    sd.grid_x_min = float(map_data.grid_x_min)
    sd.grid_y_min = float(map_data.grid_y_min)
    sd.grid_inv_cell = float(1.0 / map_data.grid_cell_size)

    inv_x = 2.0 / (map_data.x_max - map_data.x_min)
    inv_y = 2.0 / (map_data.y_max - map_data.y_min)
    x_off = (map_data.x_max + map_data.x_min) / (map_data.x_max - map_data.x_min)
    y_off = (map_data.y_max + map_data.y_min) / (map_data.y_max - map_data.y_min)
    sd.inv_x_range = float(inv_x)
    sd.inv_y_range = float(inv_y)
    sd.x_offset = float(x_off)
    sd.y_offset = float(y_off)

    sd.bombsite_dist_scale = float(map_data.bombsite_dist_scale)
    sd.laser_damage = int(sim.LASER_DAMAGE)
    sd.laser_range = float(sim.LASER_RANGE)
    sd.laser_range_sq = float(sim.LASER_RANGE * sim.LASER_RANGE)
    sd.shoot_cooldown = int(sim.SHOOT_COOLDOWN)
    sd.bomb_plant_time = int(sim.BOMB_PLANT_TIME)
    sd.bomb_defuse_time = int(sim.BOMB_DEFUSE_TIME)
    sd.bomb_defuse_kit = int(sim.BOMB_DEFUSE_KIT)
    sd.bomb_timer = int(sim.BOMB_TIMER)
    sd.round_time = int(sim.ROUND_TIME)
    sd.footstep_radius_sq = float(sim.FOOTSTEP_RADIUS * sim.FOOTSTEP_RADIUS)
    sd.gunshot_radius_sq = float(sim.GUNSHOT_RADIUS * sim.GUNSHOT_RADIUS)
    sd.enemy_memory_ticks = int(sim.ENEMY_MEMORY_TICKS)
    sd.stale_memory_tick = int(sim.STALE_MEMORY_TICK)
    sd.pbrs_gamma = 0.99
    for i in range(9):
        delta = sim._DELTA_VECTORS[i]
        sd.delta_x[i] = float(delta[0])
        sd.delta_y[i] = float(delta[1])
        sd.dir_facing[i] = float(sim._DIR_FACING[i])
    sd.max_turn_speed = float(sim.MAX_TURN_SPEED_RAD)

    id2idx = {int(aid): i for i, aid in enumerate(map_data.area_ids)}
    t_spawn_areas = map_data.t_spawn_areas
    ct_spawn_areas = map_data.ct_spawn_areas
    sd.n_t_spawns = len(t_spawn_areas)
    assert len(t_spawn_areas) <= 15, f"t_spawn_areas overflow: {len(t_spawn_areas)} > 15"
    for i, aid in enumerate(t_spawn_areas):
        sd.t_spawns[i] = id2idx[aid]

    sd.n_ct_spawns = len(ct_spawn_areas)
    assert len(ct_spawn_areas) <= 5, f"ct_spawn_areas overflow: {len(ct_spawn_areas)} > 5"
    for i, aid in enumerate(ct_spawn_areas):
        sd.ct_spawns[i] = id2idx[aid]

    return sd, refs


class RunningMeanStd:
    """Online running mean and variance (Welford's algorithm, parallel batch update).

    Tracks statistics per observation feature over all agent×step samples seen
    so far.  Thread-/process-local — each worker env maintains its own stats,
    which independently converge to the same distribution.
    """

    def __init__(self, shape, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        """Update from a batch of samples (shape: [N, *feature_dims])."""
        batch_mean = np.mean(x, axis=0, dtype=np.float64)
        batch_var = np.var(x, axis=0, dtype=np.float64)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.var = M2 / tot
        self.count = tot


class Dust2CEnv(pufferlib.PufferEnv):
    def __init__(
        self,
        sd,
        refs,
        seed=0,
        team_spirit=0.0,
        buf=None,
        nav_graph=None,
        auto_reset=True,
        map_data=None,
        normalize_obs: bool = True,
    ):
        # Observation space bounds depend on normalization setting
        obs_bounds = (-5.0, 5.0) if normalize_obs else (-1.0, 1.0)
        self.single_observation_space = gymnasium.spaces.Box(
            low=obs_bounds[0], high=obs_bounds[1], shape=(OBS_DIM,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.MultiDiscrete([9, 2, 2, 2])
        self.num_agents = N_AGENTS
        super().__init__(buf)

        self._sd = sd
        self._refs = refs  # keep alive — prevents GC of numpy backing arrays
        self.nav_graph = nav_graph
        self.map_data = map_data
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
        _lib.env_init(
            self._c_env_p,
            ctypes.byref(self._sd),
            ctypes.c_uint32(seed),
            ctypes.c_float(init_team_spirit),
        )

        # Pre-create numpy views of C buffers — avoids recreating each step
        self._obs_view = np.frombuffer(self._c_env.observations, dtype=np.float32).reshape(
            N_AGENTS, OBS_DIM
        )
        self._rew_view = np.frombuffer(self._c_env.rewards, dtype=np.float32)
        self._term_view = np.frombuffer(self._c_env.terminals, dtype=np.bool_)
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

        # Observation normalisation — RunningMeanStd per feature, clip to [-5, 5]
        self._normalize_obs = normalize_obs
        # Initialized unconditionally; set to RunningMeanStd if normalize_obs=True
        self._obs_rms = None
        if normalize_obs:
            self._obs_rms = RunningMeanStd(shape=(OBS_DIM,))

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

    def _normalize_obs_inplace(self, obs: np.ndarray) -> None:
        """Update running stats and normalise *obs* in-place, clipping to [-5, 5]."""
        if self._obs_rms is None:
            return
        self._obs_rms.update(obs)
        std = np.sqrt(self._obs_rms.var.astype(np.float32) + 1e-8)
        mean = self._obs_rms.mean.astype(np.float32)
        obs[:] = np.clip((obs - mean) / std, -5.0, 5.0)

    def _sync_outputs(self):
        if not self._uses_external_buffers:
            if self._normalize_obs:
                # self.observations IS self._obs_view (C buffer) — safe to modify in-place
                # because C code (dust2_env.c) only WRITES to observations[], never reads from it.
                # Each step, C populates observations from raw game state, then normalize in-place.
                self._normalize_obs_inplace(self.observations)
            return
        self._sync_observations()
        np.copyto(self.rewards, self._rew_view)
        np.copyto(self.terminals, self._term_view)
        np.copyto(self.truncations, self._trunc_view)

    def _sync_observations(self):
        if self._uses_external_buffers:
            np.copyto(self.observations, self._obs_view)
        if self._normalize_obs:
            self._normalize_obs_inplace(self.observations)

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
    key = (sim.NAV_PATH, sim.CACHE_PATH)
    bundle = _STATIC_DATA_CACHE.get(key)
    if bundle is not None:
        return bundle

    md = make_cs2_map(sim.NAV_PATH, sim.CACHE_PATH)
    sd, refs = build_static_data(md)
    bundle = (sd, tuple(refs), md)
    _STATIC_DATA_CACHE[key] = bundle
    return bundle


def make_env(seed=0, team_spirit=0.0, auto_reset=True, buf=None, map_data=None, normalize_obs=True):
    """Load static data and return a ready-to-use Dust2CEnv."""
    if map_data is None:
        sd, refs, md = _load_c_static_bundle()
    else:
        md = map_data
        sd, refs = build_static_data(md)
    return Dust2CEnv(
        sd,
        refs,
        seed=seed,
        team_spirit=team_spirit,
        buf=buf,
        nav_graph=md.nav_graph,
        auto_reset=auto_reset,
        map_data=md,
        normalize_obs=normalize_obs,
    )
