/* cs2_demo.c — Standalone CS2RL Raylib demo. Phase 6. */
#include "cs2_env.h"
#include "nav_data.h"
#include "cs2_render.h"
#include "cs2_input.h"
#include <stdio.h>
#include <string.h>

/* Populate StaticData from baked nav_data.h constants. Verticality fields
 * (centroids_z, is_ramp) are baked alongside the rest by scripts/bake_nav.py
 * — re-run that whenever map.py or SIMPLE_ROOMS changes. */
static void load_nav_data(StaticData* sd) {
    memset(sd, 0, sizeof(StaticData));
    sd->N                   = NAV_N;
    sd->vis_matrix          = (int8_t*)NAV_VIS_MATRIX;
    sd->raster_grid         = (int32_t*)NAV_RASTER_GRID;
    sd->adjacency           = (int8_t*)NAV_ADJACENCY;
    sd->centroid_xy         = (float*)NAV_CENTROID_XY;
    sd->centroids_z         = (float*)NAV_CENTROIDS_Z; /* verticality batch */
    sd->area_ids            = (int32_t*)NAV_AREA_IDS;
    sd->bombsite_mask       = (int8_t*)NAV_BOMBSITE_MASK;
    sd->bombsite_by_idx     = (int8_t*)NAV_BOMBSITE_BY_IDX;
    sd->is_ramp             = (int8_t*)NAV_IS_RAMP; /* verticality batch */
    sd->bombsite_dist       = (float*)NAV_BOMBSITE_DIST;
    sd->grid_w              = NAV_GRID_W;
    sd->grid_h              = NAV_GRID_H;
    sd->max_area_id         = NAV_MAX_AREA_ID;
    sd->grid_x_min          = NAV_GRID_X_MIN;
    sd->grid_y_min          = NAV_GRID_Y_MIN;
    sd->grid_inv_cell       = NAV_GRID_INV_CELL;
    sd->inv_x_range         = NAV_INV_X_RANGE;
    sd->inv_y_range         = NAV_INV_Y_RANGE;
    sd->x_offset            = NAV_X_OFFSET;
    sd->y_offset            = NAV_Y_OFFSET;
    sd->bombsite_dist_scale = NAV_BOMBSITE_DIST_SCALE;
    sd->laser_damage        = CFG_LASER_DAMAGE;
    sd->laser_range         = CFG_LASER_RANGE;
    sd->laser_range_sq      = CFG_LASER_RANGE_SQ;
    sd->shoot_cooldown      = CFG_SHOOT_COOLDOWN;
    sd->bomb_plant_time     = CFG_BOMB_PLANT_TIME;
    sd->bomb_defuse_time    = CFG_BOMB_DEFUSE_TIME;
    sd->bomb_defuse_kit     = CFG_BOMB_DEFUSE_KIT;
    sd->bomb_timer          = CFG_BOMB_TIMER;
    sd->round_time          = CFG_ROUND_TIME;
    sd->footstep_radius_sq  = CFG_FOOTSTEP_RADIUS_SQ;
    sd->gunshot_radius_sq   = CFG_GUNSHOT_RADIUS_SQ;
    sd->enemy_memory_ticks  = CFG_ENEMY_MEMORY_TICKS;
    sd->stale_memory_tick   = CFG_STALE_MEMORY_TICK;
    sd->max_turn_speed      = CFG_MAX_TURN_SPEED;
    /* Copy fixed-size arrays */
    for (int i = 0; i < 9; i++) {
        sd->delta_x[i]    = NAV_DELTA_X[i];
        sd->delta_y[i]    = NAV_DELTA_Y[i];
        sd->dir_facing[i] = NAV_DIR_FACING[i];
    }
    for (int i = 0; i < NAV_N_T_SPAWNS; i++)
        sd->t_spawns[i] = NAV_T_SPAWNS[i];
    sd->n_t_spawns = NAV_N_T_SPAWNS;
    for (int i = 0; i < NAV_N_CT_SPAWNS; i++)
        sd->ct_spawns[i] = NAV_CT_SPAWNS[i];
    sd->n_ct_spawns = NAV_N_CT_SPAWNS;
    /* Reward weights — use defaults matching Python defaults */
    sd->reward_win                  = 1.0f;
    sd->reward_kill                 = 0.3f;
    sd->reward_death                = 0.1f;
    sd->reward_bombsite_entry       = 0.3f;
    sd->reward_plant_bonus          = 3.0f;
    sd->reward_plant_base           = 0.2f;
    sd->reward_plant_progress_scale = 0.05f;
    sd->reward_plant_interrupted    = 0.1f;
    sd->reward_defuse               = 0.2f;
    sd->reward_shot_penalty         = 0.005f;
    sd->reward_ct_survival          = 0.001f;
    sd->reward_inaction             = 0.0005f;
    sd->pbrs_alive_weight           = 0.3f;
    sd->pbrs_hp_weight              = 0.002f;
    sd->pbrs_site_weight            = 0.2f;
    sd->pbrs_bomb_progress_weight   = 0.3f;
    sd->pbrs_nav_weight_t           = 0.04f;
    sd->pbrs_nav_weight_ct          = 0.15f;
    sd->pbrs_gamma                  = 0.99f;
}

int main(int argc, char** argv) {
    int human_idx   = 0;
    int fog_enabled = 0;
    /* Argv parsing — order-independent so --spectate --fog and --fog --spectate
     * both work. Unknown args are silently ignored (keeps backward compat with
     * existing scripts that pass --record, --eval, etc. to the trainer demo).
     *
     *   --spectate : detach camera from any agent (free-fly, render all).
     *   --fog      : human-agent fog-of-war — only draw enemies your agent's
     *                line_of_sight_2d says are visible. Forces you to play
     *                with the SAME perception the bot gets in its obs vector.
     *                Useful for debugging "why didn't the bot react to that?"
     *                — if you also can't see them with --fog, the bot's obs
     *                doesn't contain that enemy either. Ignored in spectate
     *                mode (no "viewer" agent to filter from). */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--spectate") == 0)
            human_idx = -1;
        else if (strcmp(argv[i], "--fog") == 0)
            fog_enabled = 1;
    }

    StaticData sd;
    load_nav_data(&sd);

    Dust2Env env;
    memset(&env, 0, sizeof(Dust2Env));
    env.sd  = &sd;
    env.rng = 12345;
    env_reset(&env);

    int32_t actions[N_AGENTS * ACTION_DIM];
    memset(actions, 0, sizeof(actions));
    /* Batch 3 (continuous-aim H-PPO): env_step gained a second action buffer
     * for the Gaussian aim head — (N_AGENTS, AIM_DIM) float32. Demo doesn't
     * need policy-driven aim (the human player has aim_rad set via mouse
     * delta in human_input(); RL agents in this demo path get zeros). */
    float continuous_actions[N_AGENTS * AIM_DIM];
    memset(continuous_actions, 0, sizeof(continuous_actions));

    Client* cl      = make_client(&env, human_idx, (const float*)NAV_AREA_BOUNDS);
    cl->fog_enabled = fog_enabled;

    double next_step = GetTime();
    while (!WindowShouldClose()) {
        double now = GetTime();
        if (now >= next_step) {
            snapshot_prev(cl, &env);
            if (human_idx >= 0)
                human_input(cl, &env, actions);
            env_step(&env, actions, continuous_actions);
            snapshot_curr(cl, &env);
            cl->last_step_time  = now;
            next_step          += 1.0 / 16.0;
        }
        c_render(cl, &env);
        if (env.terminals[0]) {
            env_reset(&env);
            _copy_agents_to_snapshot(&env, cl->curr);
            memcpy(cl->prev, cl->curr, sizeof(cl->curr));
        }
    }

    c_close(&env);
    return 0;
}
