import ctypes
import numpy as np
import subprocess
import gymnasium
import pufferlib
from pathlib import Path

_DIR = Path(__file__).parent
_SO  = _DIR / "dust2_env.so"
_SRC = _DIR / "dust2_env.c"

def _ensure_built():
    if not _SO.exists() or _SO.stat().st_mtime < _SRC.stat().st_mtime:
        subprocess.run(["make", "-C", str(_DIR)], check=True)

_ensure_built()
_lib = ctypes.CDLL(str(_SO))

N_AGENTS = 10
OBS_DIM  = 71


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


class Dust2EnvC(ctypes.Structure):
    _fields_ = [
        ("sd",           ctypes.POINTER(StaticDataC)),
        ("game",         GameStateC),
        ("observations", ctypes.c_float * (N_AGENTS * OBS_DIM)),
        ("rewards",      ctypes.c_float * N_AGENTS),
        ("terminals",    ctypes.c_int8  * N_AGENTS),
        ("truncations",  ctypes.c_int8  * N_AGENTS),
        ("actions",      ctypes.c_int32 * (N_AGENTS * 4)),
        ("team_spirit",  ctypes.c_float),
        ("delta_x",      ctypes.c_float * 9),
        ("delta_y",      ctypes.c_float * 9),
        ("dir_facing",   ctypes.c_float * 9),
        ("rng",          ctypes.c_uint32),
    ]


_lib.env_init.argtypes  = [ctypes.POINTER(Dust2EnvC), ctypes.POINTER(StaticDataC),
                            ctypes.c_uint32, ctypes.c_float]
_lib.env_init.restype   = None
_lib.env_reset.argtypes = [ctypes.POINTER(Dust2EnvC)]
_lib.env_reset.restype  = None
_lib.env_step.argtypes  = [ctypes.POINTER(Dust2EnvC)]
_lib.env_step.restype   = None
_lib.env_close.argtypes = [ctypes.POINTER(Dust2EnvC)]
_lib.env_close.restype  = None


def build_static_data(nav_graph, area_adjacency, bombsite_mask,
                      t_spawn_areas, ct_spawn_areas):
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

    id2idx = nav_graph._id_to_idx
    sd.n_t_spawns = len(t_spawn_areas)
    for i, aid in enumerate(t_spawn_areas):
        sd.t_spawns[i] = id2idx[aid]

    sd.n_ct_spawns = len(ct_spawn_areas)
    for i, aid in enumerate(ct_spawn_areas):
        sd.ct_spawns[i] = id2idx[aid]

    return sd, refs


class Dust2CEnv(pufferlib.PufferEnv):
    def __init__(self, sd, refs, seed=0, team_spirit=0.0, buf=None):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([9, 2, 2, 2])
        self.num_agents = N_AGENTS
        super().__init__(buf)

        self._sd    = sd
        self._refs  = refs   # keep alive — prevents GC of numpy backing arrays
        self._c_env = Dust2EnvC()
        _lib.env_init(ctypes.byref(self._c_env), ctypes.byref(self._sd),
                      ctypes.c_uint32(seed), ctypes.c_float(team_spirit))

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None):
        _lib.env_reset(ctypes.byref(self._c_env))
        self.observations[:] = np.frombuffer(
            self._c_env.observations, dtype=np.float32).reshape(N_AGENTS, OBS_DIM)
        return self.observations, []

    def step(self, actions):
        flat = np.asarray(actions, dtype=np.int32).flatten()
        ctypes.memmove(self._c_env.actions, flat.ctypes.data, flat.nbytes)
        _lib.env_step(ctypes.byref(self._c_env))
        self.observations[:] = np.frombuffer(
            self._c_env.observations, dtype=np.float32).reshape(N_AGENTS, OBS_DIM)
        self.rewards[:]     = np.frombuffer(self._c_env.rewards,   dtype=np.float32)
        self.terminals[:]   = np.frombuffer(self._c_env.terminals, dtype=np.int8).astype(bool)
        self.truncations[:] = False
        return self.observations, self.rewards, self.terminals, self.truncations, []

    def set_team_spirit(self, value: float):
        self._c_env.team_spirit = float(value)

    def close(self):
        _lib.env_close(ctypes.byref(self._c_env))


def make_env(seed=0, team_spirit=0.0):
    """Load static data and return a ready-to-use Dust2CEnv."""
    import sim
    static    = sim._load_dust2_static_data(sim.NAV_PATH, sim.CACHE_PATH)
    nav       = static["nav_graph"]
    adj       = static["area_adjacency"]
    bm        = static["bombsite_mask"]
    t_spawns  = static["t_spawn_areas"]
    ct_spawns = static["ct_spawn_areas"]
    sd, refs  = build_static_data(nav, adj, bm, t_spawns, ct_spawns)
    return Dust2CEnv(sd, refs, seed=seed, team_spirit=team_spirit)
