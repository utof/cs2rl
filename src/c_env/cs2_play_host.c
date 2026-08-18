/* cs2_play_host.c — the one TU that includes cs2_render.h / cs2_input.h. */
#include "cs2_play_host.h"
#include "cs2_env.h"
#include "cs2_render.h"
#include "cs2_input.h"
#include "cs2_demo_events.h"
#include <stdlib.h>
#include <string.h>

struct PlayHost {
    Dust2Env*     env;
    Client*       cl;
    DemoWorldTick prev_world, curr_world;
};

/* Copy env.game into a DemoWorldTick. Pose snapshots (AgentSnapshot) cannot
 * drive audio: they lack fired_this_tick / is_airborne / bomb_ticks_left /
 * reload_ticks. Bomb xyz is NOT on DemoWorldTick — plant/beep spatial
 * reads env.game after the step (spec §3.1). */
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
        w->agents[i].reload_ticks    = a->reload_ticks;
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
        /* Every set bit, including the human. No rate-limit. */
        if ((ev.reload_mask | ev.reload_end_mask) & (1u << i))
            _demo_play_at(cl,
                          DEMO_VOICE_RELOAD,
                          curr->agents[i].x,
                          curr->agents[i].y,
                          curr->agents[i].z,
                          800.0f);
    }
    /* DemoWorldTick has no bomb xyz; wrapper reads the post-step game. */
    if (ev.plant)
        _demo_play_at(
            cl, DEMO_VOICE_PLANT, env->game.bomb_x, env->game.bomb_y, env->game.bomb_z, 800.0f);
    if (ev.beep)
        _demo_play_at(
            cl, DEMO_VOICE_BEEP, env->game.bomb_x, env->game.bomb_y, env->game.bomb_z, 1200.0f);
}

/* View-kick is sim punch (#120). Client fields are unused leftovers. */
static void demo_apply_local_punch(Client* cl, const Dust2Env* env, unsigned shot_mask) {
    (void)cl;
    (void)env;
    (void)shot_mask;
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

PlayHost* play_host_attach(void*        env_void,
                           int          human_idx,
                           int          fog_enabled,
                           const float* area_bounds,
                           int          n_areas,
                           const char*  resource_dir) {
    Dust2Env* env = (Dust2Env*)env_void;
    PlayHost* h;
    if (!env || !env->sd || n_areas != env->sd->N)
        return NULL;
    h = calloc(1, sizeof(*h));
    if (!h)
        return NULL;
    h->env = env;
    h->cl  = make_client(env, human_idx, area_bounds, resource_dir);
    if (!h->cl) {
        free(h);
        return NULL;
    }
    h->cl->fog_enabled = fog_enabled;
    copy_game_to_world(env, &h->curr_world);
    h->prev_world = h->curr_world;
    return h;
}

void play_host_begin_tick(PlayHost* h) {
    snapshot_prev(h->cl, h->env);
    h->prev_world = h->curr_world;
}

void play_host_apply_human(PlayHost* h, int32_t* actions) {
    human_input(h->cl, h->env, actions);
}

void play_host_end_tick(PlayHost* h) {
    DemoEvents ev;
    int        i;
    snapshot_curr(h->cl, h->env);
    copy_game_to_world(h->env, &h->curr_world);
    ev = demo_detect_events(&h->prev_world, &h->curr_world);
    /* 3 Hz drop BEFORE PlaySound. Helper emits every hypot>1 walk. */
    {
        double tnow = GetTime();
        for (i = 0; i < N_AGENTS; i++) {
            if ((ev.foot_mask & (1u << i)) == 0)
                continue;
            if (tnow - h->cl->last_footstep_t[i] < (1.0 / 3.0))
                ev.foot_mask &= ~(1u << i);
            else
                h->cl->last_footstep_t[i] = tnow;
        }
    }
    demo_play_events(h->cl, h->env, &h->curr_world, ev);
    demo_apply_local_punch(h->cl, h->env, ev.shot_mask);
    demo_record_kill_feed(h->cl, &h->prev_world, &h->curr_world);
    h->cl->last_step_time = GetTime();
}

void play_host_render(PlayHost* h) {
    c_render(h->cl, h->env);
}

void play_host_on_reset(PlayHost* h) {
    snapshot_curr(h->cl, h->env);
    memcpy(h->cl->prev, h->cl->curr, sizeof(h->cl->curr));
    copy_game_to_world(h->env, &h->curr_world);
    h->prev_world = h->curr_world;
}

void play_host_detach(PlayHost* h) {
    if (!h)
        return;
    c_close(h->env);
    free(h);
}

int play_host_should_close(PlayHost* h) {
    (void)h;
    return WindowShouldClose();
}

double play_host_time(PlayHost* h) {
    (void)h;
    return GetTime();
}

const char* play_host_app_dir(void) {
    return GetApplicationDirectory();
}
