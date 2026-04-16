# src/c_env/cs2_env.py — PufferEnv subclass backed by binding.c C API bridge.

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

import gymnasium
import numpy as np
import pufferlib

import nav
from map import make_cs2_map
from nav import ACTION_DIM, ACTION_MASK_DIM, N_AGENTS, OBS_DIM, ROUND_TIME

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
        ("reward_win", ctypes.c_float),
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
        ("action_aim", ctypes.c_int32 * 16),
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
        ("masks", ctypes.c_int8 * (N_AGENTS * ACTION_MASK_DIM)),
        ("client", ctypes.c_void_p),                                   # Client* — NULL during training
    ]


# Sanity-check struct sizes match the C layout — catches future drift early.
# Sizes updated for Phase 7 (jump): AgentState +12 bytes (vz, is_airborne,
# pad, jump_cd), StepStats +8 bytes (action_jump[2]). GameState rolls up
# the agent-array delta (10×12=120). Dust2EnvC rolls up game + 2× stats +
# 20 bytes of added mask slots + alignment.
assert ctypes.sizeof(AgentStateC) == 152, (
    f"AgentStateC size mismatch: {ctypes.sizeof(AgentStateC)} (expected 152)")
assert ctypes.sizeof(GameStateC) == 1584, (
    f"GameStateC size mismatch: {ctypes.sizeof(GameStateC)} (expected 1584)")
assert ctypes.sizeof(StepStatsC) == 244, (
    f"StepStatsC size mismatch: {ctypes.sizeof(StepStatsC)} (expected 244)")
assert ctypes.sizeof(Dust2EnvC) == 6696, (
    f"Dust2EnvC size mismatch: {ctypes.sizeof(Dust2EnvC)} (expected 6696)")

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
    ):
        self.single_observation_space = gymnasium.spaces.Box(low=-5.0,
                                                             high=5.0,
                                                             shape=(OBS_DIM, ),
                                                             dtype=np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([9, 16, 2, 2, 3, 2, 2, 2])
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
        area_ids = _arr(md.area_ids, np.int32)
        bombsite_mask = _arr(md.bombsite_mask, np.int8)
        bombsite_by_idx = _arr(md.bombsite_by_idx, np.int8)
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

        # Keep refs alive — prevents GC of backing numpy arrays
        self._refs = [
            vis_matrix,
            raster_grid,
            adjacency,
            centroid_xy,
            area_ids,
            bombsite_mask,
            bombsite_by_idx,
            bombsite_dist,
            t_spawns,
            ct_spawns,
            delta_x,
            delta_y,
            dir_facing,
        ]

        # Call binding.init() — positional order matches C format string
        self._capsule = binding.init(
            vis_matrix,
            raster_grid,
            adjacency,
            centroid_xy,                                               # 0-3
            area_ids,
            bombsite_mask,
            bombsite_by_idx,
            bombsite_dist,                                             # 4-7
            int(md.N),
            int(md.grid.shape[1]),
            int(md.grid.shape[0]),                                     # 8-10: N, grid_w, grid_h
            int(md.area_ids.max()),                                    # 11: max_area_id
            float(md.grid_x_min),
            float(md.grid_y_min),                                      # 12-13
            float(1.0 / md.grid_cell_size),                            # 14: grid_inv_cell
            float(inv_x),
            float(inv_y),
            float(x_off),
            float(y_off),                                              # 15-18
            float(md.bombsite_dist_scale),                             # 19
            int(nav.LASER_DAMAGE),                                     # 20
            float(nav.LASER_RANGE),
            float(nav.LASER_RANGE * nav.LASER_RANGE),                  # 21-22
            int(nav.SHOOT_COOLDOWN),
            int(nav.BOMB_PLANT_TIME),                                  # 23-24
            int(nav.BOMB_DEFUSE_TIME),
            int(nav.BOMB_DEFUSE_KIT),                                  # 25-26
            int(nav.BOMB_TIMER),
            int(nav.ROUND_TIME),                                       # 27-28
            float(nav.FOOTSTEP_RADIUS * nav.FOOTSTEP_RADIUS),          # 29
            float(nav.GUNSHOT_RADIUS * nav.GUNSHOT_RADIUS),            # 30
            int(nav.ENEMY_MEMORY_TICKS),
            int(nav.STALE_MEMORY_TICK),                                # 31-32
            float(pbrs_gamma),                                         # 33: pbrs_gamma
            delta_x,
            delta_y,
            dir_facing,
            t_spawns,                                                  # 34-37
            int(len(md.t_spawn_areas)),                                # 38: n_t_spawns
            ct_spawns,                                                 # 39
            int(len(md.ct_spawn_areas)),                               # 40: n_ct_spawns
            float(nav.MAX_TURN_SPEED_RAD),                             # 41
            int(seed) & 0xFFFFFFFF,                                    # 42: seed (uint32)
            float(init_team_spirit),                                   # 43
            float(reward_win),                                         # 44
            float(reward_kill),                                        # 45
            float(reward_death),                                       # 46
            float(reward_bombsite_entry),                              # 47
            float(reward_plant_bonus),                                 # 48
            float(reward_plant_base),                                  # 49
            float(reward_plant_progress_scale),                        # 50
            float(reward_plant_interrupted),                           # 51
            float(reward_defuse),                                      # 52
            float(reward_shot_penalty),                                # 53
            float(reward_ct_survival),                                 # 54
            float(reward_inaction),                                    # 55
            float(pbrs_alive_weight),                                  # 56
            float(pbrs_hp_weight),                                     # 57
            float(pbrs_site_weight),                                   # 58
            float(pbrs_bomb_progress_weight),                          # 59
            float(pbrs_nav_weight_t),                                  # 60
            float(pbrs_nav_weight_ct),                                 # 61
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
        self._terminal_rewards = np.empty(N_AGENTS, dtype=np.float32)
        self._terminal_terminals = np.empty(N_AGENTS, dtype=bool)
        self._terminal_truncations = np.empty(N_AGENTS, dtype=bool)
        self._empty_infos = []

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None):
        # seed accepted for API compatibility; C RNG is set at init time
        self._sync_team_spirit()
        binding.reset(self._capsule)
        self._sync_outputs()
        return self.observations, self._empty_infos

    def step(self, actions):
        self._sync_team_spirit()
        actions_arr = self._prepare_actions(actions)
        binding.step(self._capsule, actions_arr)
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
                binding.reset(self._capsule)
                self._sync_observations()
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
        for idx in range(16):
            summary[f"action_aim_{idx}"] = int(stats.action_aim[idx])
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
    )
