/* cs2_demo.c — Standalone CS2RL Raylib demo. Phase 6. */
#include "cs2_env.h"
#include "nav_data.h"
#include "cs2_render.h"
#include "cs2_input.h"
#include "cs2_demo_events.h"
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

/* Copy env.game into a DemoWorldTick. Pose snapshots (AgentSnapshot) cannot
 * drive audio: they lack fired_this_tick / is_airborne / bomb_ticks_left.
 * Bomb xyz is NOT on DemoWorldTick — plant/beep spatial reads env.game
 * after the step (spec §3.1). */
static void copy_game_to_world(const Dust2Env* env, DemoWorldTick* w) {
    const GameState* g = &env->game;
    int              i;
    w->bomb_planted    = g->bomb_planted;
    w->bomb_ticks_left = g->bomb_ticks_left;
    for (i = 0; i < N_AGENTS; i++) {
        const AgentState* a          = &g->agents[i];
        w->agents[i].x               = a->x;
        w->agents[i].y               = a->y;
        w->agents[i].z               = a->z;
        w->agents[i].alive           = a->alive;
        w->agents[i].team            = a->team;
        w->agents[i].is_airborne     = a->is_airborne;
        w->agents[i].fired_this_tick = a->fired_this_tick;
    }
}

/* Official-example spatial play of remaining detect bits. 3 Hz foot drop
 * must already have cleared extra bits — this function does not rate-limit. */
static void demo_play_events(Client* cl, Dust2Env* env, const DemoWorldTick* curr, DemoEvents ev) {
    int i;
    for (i = 0; i < N_AGENTS; i++) {
        if (ev.shot_mask & (1u << i))
            _demo_play_at(cl,
                          DEMO_VOICE_SHOT,
                          curr->agents[i].x,
                          curr->agents[i].y,
                          curr->agents[i].z,
                          800.0f);
        if (ev.foot_mask & (1u << i))
            _demo_play_at(cl,
                          DEMO_VOICE_FOOT,
                          curr->agents[i].x,
                          curr->agents[i].y,
                          curr->agents[i].z,
                          400.0f);
    }
    /* DemoWorldTick has no bomb xyz; wrapper reads the post-step game. */
    if (ev.plant)
        _demo_play_at(
            cl, DEMO_VOICE_PLANT, env->game.bomb_x, env->game.bomb_y, env->game.bomb_z, 800.0f);
    if (ev.beep)
        _demo_play_at(
            cl, DEMO_VOICE_BEEP, env->game.bomb_x, env->game.bomb_y, env->game.bomb_z, 1200.0f);
}

/* View-kick on the local shot only (human agent, else spectate-0).
 * Applied once per sim tick, not per render frame. Do not write yaw /
 * pitch / aim_rad — human_input copies those into the sim. Render
 * decays punch and adds it to camera look only. */
static void demo_apply_local_punch(Client* cl, const Dust2Env* env, unsigned shot_mask) {
    int      local = (cl->human_agent_idx >= 0) ? cl->human_agent_idx : 0;
    unsigned u;
    float    n;
    if ((shot_mask & (1u << local)) == 0)
        return;
    cl->punch_pitch += 0.045f; /* mid-range of spec ~0.03–0.06 rad */
    /* Deterministic tiny yaw noise from the sim tick; no GetRandomValue. */
    u              = (unsigned)env->game.tick * 1664525u + 1013904223u;
    n              = ((float)((u >> 16) & 0xffff) / 32767.5f) - 1.0f;
    cl->punch_yaw += n * 0.008f;
}

/* Record alive 1→0 edges for the kill feed. Last 4, timestamped now.
 * draw_hud fades each row out over 3 s. */
static void
demo_record_kill_feed(Client* cl, const DemoWorldTick* prev, const DemoWorldTick* curr) {
    int    i;
    double t = GetTime();
    for (i = 0; i < N_AGENTS; i++) {
        if (!(prev->agents[i].alive && !curr->agents[i].alive))
            continue;
        if (cl->kill_feed_n < DEMO_KILL_FEED_N) {
            int k                 = cl->kill_feed_n++;
            cl->kill_feed_idx[k]  = i;
            cl->kill_feed_team[k] = curr->agents[i].team;
            cl->kill_feed_t[k]    = t;
        } else {
            memmove(cl->kill_feed_idx, cl->kill_feed_idx + 1, (DEMO_KILL_FEED_N - 1) * sizeof(int));
            memmove(
                cl->kill_feed_team, cl->kill_feed_team + 1, (DEMO_KILL_FEED_N - 1) * sizeof(int));
            memmove(cl->kill_feed_t, cl->kill_feed_t + 1, (DEMO_KILL_FEED_N - 1) * sizeof(double));
            cl->kill_feed_idx[DEMO_KILL_FEED_N - 1]  = i;
            cl->kill_feed_team[DEMO_KILL_FEED_N - 1] = curr->agents[i].team;
            cl->kill_feed_t[DEMO_KILL_FEED_N - 1]    = t;
        }
    }
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

    /* Both world ticks start as a copy of the post-reset game. A
     * prev!=curr spawn pair would look like a teleport and footstep
     * every agent (spec §3.1). Same for env_reset below. */
    DemoWorldTick prev_world, curr_world;
    copy_game_to_world(&env, &curr_world);
    prev_world = curr_world;

    double next_step = GetTime();
    while (!WindowShouldClose()) {
        double now = GetTime();
        if (now >= next_step) {
            DemoEvents ev;
            int        i;
            snapshot_prev(cl, &env);
            prev_world = curr_world; /* same shift as pose snapshots */
            if (human_idx >= 0)
                human_input(cl, &env, actions);
            env_step(&env, actions, continuous_actions);
            snapshot_curr(cl, &env);
            copy_game_to_world(&env, &curr_world);
            ev = demo_detect_events(&prev_world, &curr_world);
            /* 3 Hz drop BEFORE PlaySound. Helper emits every hypot>1 walk. */
            {
                double tnow = GetTime();
                for (i = 0; i < N_AGENTS; i++) {
                    if ((ev.foot_mask & (1u << i)) == 0)
                        continue;
                    if (tnow - cl->last_footstep_t[i] < (1.0 / 3.0))
                        ev.foot_mask &= ~(1u << i);
                    else
                        cl->last_footstep_t[i] = tnow;
                }
            }
            demo_play_events(cl, &env, &curr_world, ev);
            demo_apply_local_punch(cl, &env, ev.shot_mask);
            demo_record_kill_feed(cl, &prev_world, &curr_world);
            cl->last_step_time  = now;
            next_step          += 1.0 / 16.0;
        }
        c_render(cl, &env);
        if (env.terminals[0]) {
            env_reset(&env);
            _copy_agents_to_snapshot(&env, cl->curr);
            memcpy(cl->prev, cl->curr, sizeof(cl->curr));
            copy_game_to_world(&env, &curr_world);
            prev_world = curr_world;
        }
    }

    c_close(&env);
    return 0;
}
