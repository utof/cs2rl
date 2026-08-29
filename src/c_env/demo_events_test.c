/* demo_events_test.c — headless checks for cs2_demo_events.h / cs2_demo_viz.h
 * / cs2_solids.h (viz includes cs2_terrain.h for ramp / terrain_z).
 *
 * Built/run by `zig build demo_events_test`. No Raylib, no binding.so,
 * no ctypes. The play wrapper's 3 Hz footstep drop is intentionally
 * NOT tested here — a walking tick with hypot>1 must set the foot bit
 * every sim tick. Punch decay is the formula only; no InitWindow.
 * Aim / ramp helpers are Raylib-free math; reload is a snapshot-diff bit.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cs2_demo_events.h"
#include "cs2_demo_viz.h"
#include "cs2_solids.h"

#ifdef RAYLIB_H
#error "cs2_demo_events.h / cs2_demo_viz.h / cs2_solids.h must not include raylib.h"
#endif
#ifdef WALL_DEPTH
#error "cs2_solids.h must not pull cs2_render.h — the draw offset stays render-side"
#endif

static int g_fails;

static void check_u(const char* name, unsigned got, unsigned want) {
    if (got != want) {
        fprintf(stderr, "FAIL %s: got %u want %u\n", name, got, want);
        g_fails++;
    }
}

static void check_i(const char* name, int got, int want) {
    if (got != want) {
        fprintf(stderr, "FAIL %s: got %d want %d\n", name, got, want);
        g_fails++;
    }
}

static void check_ok(const char* name, int cond) {
    if (!cond) {
        fprintf(stderr, "FAIL %s\n", name);
        g_fails++;
    }
}

static void check_f_near(const char* name, float got, float want, float eps) {
    float d = got - want;
    if (d < 0.0f)
        d = -d;
    if (d > eps) {
        fprintf(stderr, "FAIL %s: got %g want %g\n", name, (double)got, (double)want);
        g_fails++;
    }
}

/* Zeroed tick = make_client / env_reset copy of an empty world. */
static DemoWorldTick tick_zero(void) {
    DemoWorldTick w;
    memset(&w, 0, sizeof(w));
    return w;
}

/* Shot: only curr.fired_this_tick sets the bit. fire_cd is not in this POD. */
static void test_shot_pulse(void) {
    DemoWorldTick prev             = tick_zero();
    DemoWorldTick curr             = tick_zero();
    curr.agents[0].fired_this_tick = 1;
    DemoEvents ev                  = demo_detect_events(&prev, &curr);
    check_u("shot pulse bit 0", ev.shot_mask, 1u);
    check_u("shot pulse no foot", ev.foot_mask, 0u);
    check_u("shot pulse no reload", ev.reload_mask, 0u);
    check_i("shot pulse no plant", ev.plant, 0);
    check_i("shot pulse no beep", ev.beep, 0);
}

static void test_no_shot_when_bit0(void) {
    DemoWorldTick prev             = tick_zero();
    DemoWorldTick curr             = tick_zero();
    prev.agents[0].fired_this_tick = 1; /* prev-only pulse must not fire */
    curr.agents[0].fired_this_tick = 0;
    DemoEvents ev                  = demo_detect_events(&prev, &curr);
    check_u("no shot when bit 0", ev.shot_mask, 0u);
}

static void test_shot_other_agent(void) {
    DemoWorldTick prev             = tick_zero();
    DemoWorldTick curr             = tick_zero();
    curr.agents[3].fired_this_tick = 1;
    DemoEvents ev                  = demo_detect_events(&prev, &curr);
    check_u("shot bit 3 only", ev.shot_mask, 1u << 3);
}

/* All N_AGENTS bits (macro, not a hardcoded 10) can be set in one tick. */
static void test_all_agents_shot(void) {
    DemoWorldTick prev = tick_zero();
    DemoWorldTick curr = tick_zero();
    unsigned      want = 0;
    int           i;
    for (i = 0; i < N_AGENTS; i++) {
        curr.agents[i].fired_this_tick  = 1;
        want                           |= 1u << i;
    }
    DemoEvents ev = demo_detect_events(&prev, &curr);
    check_u("all agents shot", ev.shot_mask, want);
}

static void test_footstep_hypot(void) {
    DemoWorldTick prev   = tick_zero();
    DemoWorldTick curr   = tick_zero();
    prev.agents[0].alive = 1;
    curr.agents[0].alive = 1;

    curr.agents[0].x = 1.1f;
    DemoEvents ev    = demo_detect_events(&prev, &curr);
    check_u("foot hypot 1.1", ev.foot_mask, 1u);

    curr.agents[0].x = 0.5f;
    ev               = demo_detect_events(&prev, &curr);
    check_u("foot hypot 0.5", ev.foot_mask, 0u);

    /* Predicate is strictly > 1.0f, not >=. */
    curr.agents[0].x = 1.0f;
    ev               = demo_detect_events(&prev, &curr);
    check_u("foot hypot 1.0", ev.foot_mask, 0u);
}

/* Detect hits every walking sim tick. 3 Hz drop lives in the play wrapper. */
static void test_footstep_two_ticks(void) {
    DemoWorldTick a   = tick_zero();
    DemoWorldTick b   = tick_zero();
    DemoWorldTick c   = tick_zero();
    a.agents[1].alive = 1;
    b.agents[1].alive = 1;
    c.agents[1].alive = 1;
    a.agents[1].x     = 0.0f;
    b.agents[1].x     = 1.1f;
    c.agents[1].x     = 2.2f;
    DemoEvents ev1    = demo_detect_events(&a, &b);
    DemoEvents ev2    = demo_detect_events(&b, &c);
    check_u("walk tick 1 foot bit 1", ev1.foot_mask, 1u << 1);
    check_u("walk tick 2 foot bit 1", ev2.foot_mask, 1u << 1);
}

static void test_airborne_no_foot(void) {
    DemoWorldTick prev         = tick_zero();
    DemoWorldTick curr         = tick_zero();
    prev.agents[0].alive       = 1;
    curr.agents[0].alive       = 1;
    curr.agents[0].is_airborne = 1;
    curr.agents[0].x           = 1.1f;
    DemoEvents ev              = demo_detect_events(&prev, &curr);
    check_u("airborne no foot", ev.foot_mask, 0u);
}

static void test_dead_no_foot(void) {
    DemoWorldTick prev   = tick_zero();
    DemoWorldTick curr   = tick_zero();
    curr.agents[0].alive = 0;
    curr.agents[0].x     = 1.1f;
    DemoEvents ev        = demo_detect_events(&prev, &curr);
    check_u("dead no foot", ev.foot_mask, 0u);
}

/* z-only motion is not a footstep — helper is 2D hypot(dx, dy). */
static void test_z_only_no_foot(void) {
    DemoWorldTick prev   = tick_zero();
    DemoWorldTick curr   = tick_zero();
    prev.agents[0].alive = 1;
    curr.agents[0].alive = 1;
    curr.agents[0].z     = 8.0f;
    DemoEvents ev        = demo_detect_events(&prev, &curr);
    check_u("z-only no foot", ev.foot_mask, 0u);
}

/* Live plant tick is bomb_timer then -- in the same process_bomb, so
 * the first planted snapshot is 639, not 640 or 100. Must emit BOTH. */
static void test_plant_and_beep_0_to_639(void) {
    DemoWorldTick prev   = tick_zero();
    DemoWorldTick curr   = tick_zero();
    prev.bomb_planted    = 0;
    prev.bomb_ticks_left = 0;
    curr.bomb_planted    = 1;
    curr.bomb_ticks_left = 639;
    DemoEvents ev        = demo_detect_events(&prev, &curr);
    check_i("plant 0->639", ev.plant, 1);
    check_i("beep 0->639", ev.beep, 1);
    check_u("plant tick no shot", ev.shot_mask, 0u);
}

/* 100/16 == 99/16 == 6 — same slow bucket, already planted. */
static void test_no_beep_100_to_99(void) {
    DemoWorldTick prev   = tick_zero();
    DemoWorldTick curr   = tick_zero();
    prev.bomb_planted    = 1;
    prev.bomb_ticks_left = 100;
    curr.bomb_planted    = 1;
    curr.bomb_ticks_left = 99;
    DemoEvents ev        = demo_detect_events(&prev, &curr);
    check_i("no plant 100->99", ev.plant, 0);
    check_i("no beep 100->99", ev.beep, 0);
}

/* 96/16 == 6, 95/16 == 5 — bucket edge on the slow (p=16) schedule. */
static void test_beep_96_to_95(void) {
    DemoWorldTick prev   = tick_zero();
    DemoWorldTick curr   = tick_zero();
    prev.bomb_planted    = 1;
    prev.bomb_ticks_left = 96;
    curr.bomb_planted    = 1;
    curr.bomb_ticks_left = 95;
    DemoEvents ev        = demo_detect_events(&prev, &curr);
    check_i("no plant 96->95", ev.plant, 0);
    check_i("beep 96->95", ev.beep, 1);
}

/* p is chosen from *curr* ticks: 79 <= 80 ⇒ p=8; 80/8 != 79/8. */
static void test_beep_80_to_79(void) {
    DemoWorldTick prev   = tick_zero();
    DemoWorldTick curr   = tick_zero();
    prev.bomb_planted    = 1;
    prev.bomb_ticks_left = 80;
    curr.bomb_planted    = 1;
    curr.bomb_ticks_left = 79;
    DemoEvents ev        = demo_detect_events(&prev, &curr);
    check_i("beep 80->79 p=8", ev.beep, 1);
    check_i("no plant 80->79", ev.plant, 0);
}

/* Init/reset copies env.game into both snapshots — must be silent. */
static void test_prev_eq_curr_after_init(void) {
    DemoWorldTick w = tick_zero();
    int           i;
    for (i = 0; i < N_AGENTS; i++) {
        w.agents[i].alive = 1;
        w.agents[i].x     = 10.0f * (float)i;
        w.agents[i].y     = 4.0f;
    }
    w.bomb_planted    = 0;
    w.bomb_ticks_left = 0;
    DemoEvents ev     = demo_detect_events(&w, &w);
    check_u("init no shot", ev.shot_mask, 0u);
    check_u("init no foot", ev.foot_mask, 0u);
    check_u("init no reload", ev.reload_mask, 0u);
    check_i("init no plant", ev.plant, 0);
    check_i("init no beep", ev.beep, 0);
}

/* Unplanted tick countdown must not beep (p-bucket is gated on planted). */
static void test_unplanted_no_beep(void) {
    DemoWorldTick prev   = tick_zero();
    DemoWorldTick curr   = tick_zero();
    prev.bomb_planted    = 0;
    prev.bomb_ticks_left = 96;
    curr.bomb_planted    = 0;
    curr.bomb_ticks_left = 95;
    DemoEvents ev        = demo_detect_events(&prev, &curr);
    check_i("unplanted no beep", ev.beep, 0);
    check_i("unplanted no plant", ev.plant, 0);
}

/* Punch decay is render-only juice. next = prev * exp(-dt/0.08).
 * Same sign, smaller abs. Must not live behind raylib.h. */
static void test_punch_decay_formula(void) {
    const float prev = 0.045f;
    const float dt   = 1.0f / 60.0f;
    float       next = demo_decay_punch(prev, dt);
    float       want = prev * expf(-dt / 0.08f);
    check_f_near("punch decay formula", next, want, 1e-6f);
}

static void test_punch_decay_same_sign_smaller_abs(void) {
    const float dt = 1.0f / 60.0f;
    float       p  = demo_decay_punch(0.045f, dt);
    check_ok("pos punch same sign", p > 0.0f);
    check_ok("pos punch smaller abs", p < 0.045f);

    float n = demo_decay_punch(-0.008f, dt);
    check_ok("neg punch same sign", n < 0.0f);
    check_ok("neg punch smaller abs", (-n) < 0.008f);
}

static void test_punch_decay_zero_and_dt0(void) {
    check_f_near("zero punch stays 0", demo_decay_punch(0.0f, 0.016f), 0.0f, 1e-7f);
    check_f_near("dt=0 punch unchanged", demo_decay_punch(0.045f, 0.0f), 0.045f, 1e-7f);
}

static void test_recoil_decay_punch_by_name(void) {
    const float dt = 1.0f / 16.0f;
    float       n  = recoil_decay_punch(0.045f, dt);
    check_f_near("recoil_decay_punch formula", n, 0.045f * expf(-dt / 0.08f), 1e-6f);
    check_ok("recoil_decay same sign", n > 0.0f);
    check_ok("recoil_decay smaller abs", n < 0.045f);
    check_f_near("recoil_decay dt=0", recoil_decay_punch(0.045f, 0.0f), 0.045f, 1e-7f);
}

/* Combat aim: d = (cos p cos y, cos p sin y, sin p). Stick is Raylib
 * (x, z+108, y) + length*(dx, dz, dy). Punch is the caller's job. */
static void test_aim_yaw0_pitch0(void) {
    float dx, dy, dz;
    demo_aim_dir_sim(0.0f, 0.0f, &dx, &dy, &dz);
    check_f_near("aim y0p0 dx", dx, 1.0f, 1e-5f);
    check_f_near("aim y0p0 dy", dy, 0.0f, 1e-5f);
    check_f_near("aim y0p0 dz", dz, 0.0f, 1e-5f);
    check_f_near("aim y0p0 |d|", sqrtf(dx * dx + dy * dy + dz * dz), 1.0f, 1e-5f);

    float start[3], end[3];
    demo_aim_stick_rl(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 60.0f, start, end);
    check_f_near("stick y0p0 start x", start[0], 0.0f, 1e-5f);
    check_f_near("stick y0p0 start y", start[1], 108.0f, 1e-5f);
    check_f_near("stick y0p0 start z", start[2], 0.0f, 1e-5f);
    check_f_near("stick y0p0 end x", end[0], 60.0f, 1e-5f);
    check_f_near("stick y0p0 end y", end[1], 108.0f, 1e-5f);
    check_f_near("stick y0p0 end z", end[2], 0.0f, 1e-5f);
}

static void test_aim_yaw0_pitch_halfpi(void) {
    const float half_pi = acosf(-1.0f) * 0.5f;
    float       dx, dy, dz;
    demo_aim_dir_sim(0.0f, half_pi, &dx, &dy, &dz);
    check_f_near("aim y0pπ/2 dx", dx, 0.0f, 1e-5f);
    check_f_near("aim y0pπ/2 dy", dy, 0.0f, 1e-5f);
    check_f_near("aim y0pπ/2 dz", dz, 1.0f, 1e-5f);
    check_f_near("aim y0pπ/2 |d|", sqrtf(dx * dx + dy * dy + dz * dz), 1.0f, 1e-5f);

    float start[3], end[3];
    demo_aim_stick_rl(0.0f, 0.0f, 0.0f, 0.0f, half_pi, 60.0f, start, end);
    check_f_near("stick y0pπ/2 start y", start[1], 108.0f, 1e-5f);
    check_f_near("stick y0pπ/2 end x", end[0], 0.0f, 1e-5f);
    check_f_near("stick y0pπ/2 end y", end[1], 168.0f, 1e-5f);
    check_f_near("stick y0pπ/2 end z", end[2], 0.0f, 1e-5f);
}

static void test_aim_yaw_halfpi_pitch0(void) {
    const float half_pi = acosf(-1.0f) * 0.5f;
    float       dx, dy, dz;
    demo_aim_dir_sim(half_pi, 0.0f, &dx, &dy, &dz);
    check_f_near("aim yπ/2p0 dx", dx, 0.0f, 1e-5f);
    check_f_near("aim yπ/2p0 dy", dy, 1.0f, 1e-5f);
    check_f_near("aim yπ/2p0 dz", dz, 0.0f, 1e-5f);
    check_f_near("aim yπ/2p0 |d|", sqrtf(dx * dx + dy * dy + dz * dz), 1.0f, 1e-5f);

    float start[3], end[3];
    demo_aim_stick_rl(0.0f, 0.0f, 0.0f, half_pi, 0.0f, 60.0f, start, end);
    check_f_near("stick yπ/2p0 start y", start[1], 108.0f, 1e-5f);
    check_f_near("stick yπ/2p0 end x", end[0], 0.0f, 1e-5f);
    check_f_near("stick yπ/2p0 end y", end[1], 108.0f, 1e-5f);
    check_f_near("stick yπ/2p0 end z", end[2], 60.0f, 1e-5f);
}

/* T-ramp: low west room, ramp, high east room. West edge of the ramp
 * must pick the z=0 neighbor (longest overlap), east the z=64 room. */
static void test_ramp_t_topology(void) {
    const float bounds[] = {
        400.0f,
        192.0f,
        750.0f,
        416.0f, /* area 0 floor */
        750.0f,
        192.0f,
        820.0f,
        416.0f, /* area 1 ramp */
        820.0f,
        192.0f,
        1100.0f,
        416.0f, /* area 2 floor */
    };
    const float  zs[] = {0.0f, 64.0f, 64.0f};
    DemoRampQuad q;
    demo_ramp_quad(1, 3, bounds, zs, &q);

    check_f_near("ramp c0 x", q.x[0], 750.0f, 1e-4f);
    check_f_near("ramp c0 y", q.y[0], 192.0f, 1e-4f);
    check_f_near("ramp c0 z", q.z[0], 0.0f, 1e-4f);
    check_f_near("ramp c1 x", q.x[1], 820.0f, 1e-4f);
    check_f_near("ramp c1 y", q.y[1], 192.0f, 1e-4f);
    check_f_near("ramp c1 z", q.z[1], 64.0f, 1e-4f);
    check_f_near("ramp c2 x", q.x[2], 820.0f, 1e-4f);
    check_f_near("ramp c2 y", q.y[2], 416.0f, 1e-4f);
    check_f_near("ramp c2 z", q.z[2], 64.0f, 1e-4f);
    check_f_near("ramp c3 x", q.x[3], 750.0f, 1e-4f);
    check_f_near("ramp c3 y", q.y[3], 416.0f, 1e-4f);
    check_f_near("ramp c3 z", q.z[3], 0.0f, 1e-4f);
    /* West z=0, east z=64 → |Δz_x| > |Δz_y| so the slope is along X. */
    check_ok("ramp slope along X", q.z[0] == q.z[3] && q.z[1] == q.z[2]);
}

/* SIMPLE_ROOMS stairs (16) + catwalk west (15) + CT-corridor north (7).
 * South/east have no neighbor → this area's z=128. |Δz_y| > |Δz_x|
 * so the else-Y branch runs. Swapping z_s/z_n would flip 128 and 0. */
static void test_ramp_y_slope_stairs(void) {
    const float bounds[] = {
        1170.0f,
        80.0f,
        1300.0f,
        192.0f, /* 0 stairs */
        820.0f,
        80.0f,
        1170.0f,
        192.0f, /* 1 catwalk west z=128 */
        1170.0f,
        192.0f,
        1600.0f,
        512.0f, /* 2 CT-corridor north z=0 */
    };
    const float  zs[] = {128.0f, 128.0f, 0.0f};
    DemoRampQuad q;
    demo_ramp_quad(0, 3, bounds, zs, &q);

    check_f_near("ystairs c0 x", q.x[0], 1170.0f, 1e-4f);
    check_f_near("ystairs c0 y", q.y[0], 80.0f, 1e-4f);
    check_f_near("ystairs c0 z", q.z[0], 128.0f, 1e-4f);
    check_f_near("ystairs c1 x", q.x[1], 1300.0f, 1e-4f);
    check_f_near("ystairs c1 y", q.y[1], 80.0f, 1e-4f);
    check_f_near("ystairs c1 z", q.z[1], 128.0f, 1e-4f);
    check_f_near("ystairs c2 x", q.x[2], 1300.0f, 1e-4f);
    check_f_near("ystairs c2 y", q.y[2], 192.0f, 1e-4f);
    check_f_near("ystairs c2 z", q.z[2], 0.0f, 1e-4f);
    check_f_near("ystairs c3 x", q.x[3], 1170.0f, 1e-4f);
    check_f_near("ystairs c3 y", q.y[3], 192.0f, 1e-4f);
    check_f_near("ystairs c3 z", q.z[3], 0.0f, 1e-4f);
    check_ok("ystairs slope along Y", q.z[0] == q.z[1] && q.z[2] == q.z[3]);
}

/* Same T-ramp fixture as test_ramp_t_topology. Bilinear on the quad:
 * west x=750 → z≈0, east x=820 → z≈64, mid x=785 → z≈32.
 * Non-ramp area 0 stays 0. NULL bounds fall back to centroids_z (64). */
static void test_terrain_z_t_ramp(void) {
    const float bounds[] = {
        400.0f,
        192.0f,
        750.0f,
        416.0f, /* area 0 floor */
        750.0f,
        192.0f,
        820.0f,
        416.0f, /* area 1 ramp */
        820.0f,
        192.0f,
        1100.0f,
        416.0f, /* area 2 floor */
    };
    const float  zs[]      = {0.0f, 64.0f, 64.0f};
    const int8_t is_ramp[] = {0, 1, 0};

    check_f_near(
        "terrain west", demo_terrain_z(1, 750.0f, 300.0f, 3, bounds, zs, is_ramp), 0.0f, 1e-3f);
    check_f_near(
        "terrain east", demo_terrain_z(1, 820.0f, 300.0f, 3, bounds, zs, is_ramp), 64.0f, 1e-3f);
    check_f_near(
        "terrain mid", demo_terrain_z(1, 785.0f, 300.0f, 3, bounds, zs, is_ramp), 32.0f, 1e-3f);
    check_f_near("terrain flat room",
                 demo_terrain_z(0, 500.0f, 300.0f, 3, bounds, zs, is_ramp),
                 0.0f,
                 1e-4f);
    check_f_near("terrain null bounds",
                 demo_terrain_z(1, 785.0f, 300.0f, 3, NULL, zs, is_ramp),
                 64.0f,
                 1e-4f);
}

/* Same Y-stairs fixture as test_ramp_y_slope_stairs.
 * South y=80 → 128, north y=192 → 0, mid y=136 → 64. */
static void test_terrain_z_y_stairs(void) {
    const float bounds[] = {
        1170.0f,
        80.0f,
        1300.0f,
        192.0f, /* 0 stairs */
        820.0f,
        80.0f,
        1170.0f,
        192.0f, /* 1 catwalk west z=128 */
        1170.0f,
        192.0f,
        1600.0f,
        512.0f, /* 2 CT-corridor north z=0 */
    };
    const float  zs[]      = {128.0f, 128.0f, 0.0f};
    const int8_t is_ramp[] = {1, 0, 0};

    check_f_near(
        "ystairs south", demo_terrain_z(0, 1235.0f, 80.0f, 3, bounds, zs, is_ramp), 128.0f, 1e-3f);
    check_f_near(
        "ystairs north", demo_terrain_z(0, 1235.0f, 192.0f, 3, bounds, zs, is_ramp), 0.0f, 1e-3f);
    check_f_near(
        "ystairs mid", demo_terrain_z(0, 1235.0f, 136.0f, 3, bounds, zs, is_ramp), 64.0f, 1e-3f);
}

static void test_reload_start_agent0(void) {
    DemoWorldTick prev          = tick_zero();
    DemoWorldTick curr          = tick_zero();
    prev.agents[0].reload_ticks = 0;
    curr.agents[0].reload_ticks = 30;
    DemoEvents ev               = demo_detect_events(&prev, &curr);
    check_u("reload start bit 0", ev.reload_mask, 1u);
}

static void test_reload_start_agent3(void) {
    DemoWorldTick prev          = tick_zero();
    DemoWorldTick curr          = tick_zero();
    prev.agents[3].reload_ticks = 0;
    curr.agents[3].reload_ticks = 30;
    DemoEvents ev               = demo_detect_events(&prev, &curr);
    check_u("reload start bit 3", ev.reload_mask, 1u << 3);
}

static void test_reload_prev_eq_curr_silent(void) {
    DemoWorldTick w          = tick_zero();
    w.agents[0].reload_ticks = 30;
    DemoEvents ev            = demo_detect_events(&w, &w);
    check_u("reload prev==curr silent", ev.reload_mask, 0u);
}

static void test_reload_countdown_silent(void) {
    DemoWorldTick prev          = tick_zero();
    DemoWorldTick curr          = tick_zero();
    prev.agents[0].reload_ticks = 10;
    curr.agents[0].reload_ticks = 9;
    DemoEvents ev               = demo_detect_events(&prev, &curr);
    check_u("reload countdown silent", ev.reload_mask, 0u);
}

static void test_reload_end_agent0(void) {
    DemoWorldTick prev          = tick_zero();
    DemoWorldTick curr          = tick_zero();
    prev.agents[0].reload_ticks = 1;
    curr.agents[0].reload_ticks = 0;
    DemoEvents ev               = demo_detect_events(&prev, &curr);
    check_u("reload end bit 0", ev.reload_end_mask, 1u);
    check_u("reload start silent on end", ev.reload_mask, 0u);
}

static void test_reload_countdown_not_end(void) {
    DemoWorldTick prev          = tick_zero();
    DemoWorldTick curr          = tick_zero();
    prev.agents[0].reload_ticks = 10;
    curr.agents[0].reload_ticks = 9;
    DemoEvents ev               = demo_detect_events(&prev, &curr);
    check_u("countdown not end", ev.reload_end_mask, 0u);
}

/* ── cs2_solids.h — baked room-face solids ────────────────────────────────
 *
 * These stack a minimal StaticData (only the fields build_solids_from_rooms
 * reads: N / area_bounds / centroids_z / is_ramp / adjacency / wall_list) and
 * assert on the baked seg list plus the two queries movement and LoS will
 * share. Every fixture frees the list at the end — the bake is the ONLY
 * malloc site, so a leak here is a leak in the demo too.
 */

/* Fixture arrays live in the struct so the StaticData pointers stay valid for
 * the whole test; a StaticData with dangling area_bounds bakes garbage.
 * Sized for the widest fixture below (the 3-room partial-coverage doorway);
 * sd.N is what the bake reads, so a 2-room fixture just leaves the tail zero.
 * Pitfall: adjacency is indexed [i*sd.N + j], NOT [i*SOLIDS_FIX_ROOMS + j] —
 * solids_fix_adj() below does that arithmetic so tests cannot get it wrong. */
#define SOLIDS_FIX_ROOMS 4
typedef struct {
    StaticData sd;
    float      bounds[SOLIDS_FIX_ROOMS * 4]; /* per room: x0,y0,x1,y1 */
    float      zs[SOLIDS_FIX_ROOMS];
    int8_t     ramps[SOLIDS_FIX_ROOMS];
    int8_t     adj[SOLIDS_FIX_ROOMS * SOLIDS_FIX_ROOMS]; /* row-major [i*N+j] */
} SolidsFix;

/* solids_fix_n — wire the raw arrays onto a zeroed StaticData with N rooms.
 *
 * Every room starts self-adjacent and mutually unconnected; callers open the
 * portals they want with solids_fix_adj.
 *
 * Pitfall: memset the StaticData first. build_solids_from_rooms frees
 * wall_list.walls if it is non-NULL, so an uninitialised pointer here is a
 * free() of a stack address.
 */
static void solids_fix_n(SolidsFix* f, int n) {
    int i;
    memset(f, 0, sizeof(*f));
    f->sd.N           = n;
    f->sd.area_bounds = f->bounds;
    f->sd.centroids_z = f->zs;
    f->sd.is_ramp     = f->ramps;
    f->sd.adjacency   = f->adj;
    /* Self-adjacency is always true in map.py. */
    for (i = 0; i < n; i++)
        f->adj[i * n + i] = 1;
}

static void solids_fix(SolidsFix* f) {
    solids_fix_n(f, 2);
}

/* One room quad. z is terrain elevation, ramp is the cliff-guard exemption. */
static void
solids_fix_room(SolidsFix* f, int i, float x0, float y0, float x1, float y1, float z, int ramp) {
    f->bounds[i * 4 + 0] = x0;
    f->bounds[i * 4 + 1] = y0;
    f->bounds[i * 4 + 2] = x1;
    f->bounds[i * 4 + 3] = y1;
    f->zs[i]             = z;
    f->ramps[i]          = (int8_t)(ramp != 0);
}

/* Symmetric adjacency, indexed against the fixture's live sd.N. */
static void solids_fix_adj(SolidsFix* f, int i, int j, int connected) {
    int n             = f->sd.N;
    f->adj[i * n + j] = (int8_t)(connected != 0);
    f->adj[j * n + i] = (int8_t)(connected != 0);
}

/* Two flush 100x100 rooms sharing x=100, connected → that edge is a portal. */
static void solids_fix_two_rooms(SolidsFix* f) {
    solids_fix(f);
    solids_fix_room(f, 0, 0.0f, 0.0f, 100.0f, 100.0f, 0.0f, 0);
    solids_fix_room(f, 1, 100.0f, 0.0f, 200.0f, 100.0f, 0.0f, 0);
    solids_fix_adj(f, 0, 1, 1);
}

/* Room 0 at z=0, room 1 stacked north at z=128, adjacency cliff-pruned.
 * Mirrors catwalk(128) over bombsite(64): a shared edge that is NOT a
 * portal, so the drop must come back as a lip, not as a doorway.
 * z1 is a parameter because a lip whose drop equals SOLID_WALL_HEIGHT is
 * numerically indistinguishable from an exterior wall of the lower room. */
static void solids_fix_cliff_z(SolidsFix* f, float z1) {
    solids_fix(f);
    solids_fix_room(f, 0, 0.0f, 0.0f, 100.0f, 100.0f, 0.0f, 0);
    solids_fix_room(f, 1, 0.0f, 100.0f, 100.0f, 200.0f, z1, 0);
    /* cross terms stay 0 — the cliff prune */
}

/* Segs whose infinite line is x==v (vertical) / y==v (horizontal). */
static int solids_count_vline(const StaticData* sd, float v) {
    int i, n = 0;
    for (i = 0; i < sd->wall_list.count; i++) {
        const Wall* w = &sd->wall_list.walls[i];
        if (fabsf(w->x1 - w->x0) < 1e-3f && fabsf(w->x0 - v) < 1e-3f)
            n++;
    }
    return n;
}

static int solids_count_hline(const StaticData* sd, float v) {
    int i, n = 0;
    for (i = 0; i < sd->wall_list.count; i++) {
        const Wall* w = &sd->wall_list.walls[i];
        if (fabsf(w->y1 - w->y0) < 1e-3f && fabsf(w->y0 - v) < 1e-3f)
            n++;
    }
    return n;
}

/* solids_poison_hit — seed a SolidHit with a value no real hit can produce.
 *
 * What: t = 2.0f, normal zeroed. Call this instead of memset before every
 *       solid_sweep_xy that inspects `hit`.
 * Why:  the sweep writes *hit ONLY when it returns 1, so a hit left at zero
 *       makes assertions about a MISS pass. This bit for real: with the t<0
 *       clamp reverted, `check_f_near("... t clamped", hit.t, 0.0f, ...)` kept
 *       passing on the zeroed struct while every other check in the same test
 *       failed, i.e. the flagship regression test's headline assertion was
 *       the one thing not being tested.
 * Pitfall: the sentinel has to break BOTH shapes of t assertion this file
 *          uses — `== 0` and `< 1`. -1.0f satisfies `t < 1.0f` and would
 *          leave those vacuous, so the out-of-range value is on the high
 *          side: 2.0f, the same seed solid_sweep_xy gives best_t.
 */
static void solids_poison_hit(SolidHit* h) {
    h->t  = 2.0f;
    h->nx = 0.0f;
    h->ny = 0.0f;
}

/* Outline is solid, the shared edge is not. Six exterior faces: room 0 keeps
 * west/south/north, room 1 keeps east/south/north. */
static void test_solids_bake_two_rooms(void) {
    SolidsFix f;
    solids_fix_two_rooms(&f);
    build_solids_from_rooms(&f.sd);

    check_i("solids 2room count", f.sd.wall_list.count, 6);
    check_i("solids 2room portal x=100", solids_count_vline(&f.sd, 100.0f), 0);
    check_i("solids 2room west x=0", solids_count_vline(&f.sd, 0.0f), 1);
    check_i("solids 2room east x=200", solids_count_vline(&f.sd, 200.0f), 1);
    check_i("solids 2room south y=0", solids_count_hline(&f.sd, 0.0f), 2);
    check_i("solids 2room north y=100", solids_count_hline(&f.sd, 100.0f), 2);
    /* True edge, never line-WALL_DEPTH/2: an x=-4 seg would fail the x=0 count. */
    check_i("solids 2room no draw offset", solids_count_vline(&f.sd, -4.0f), 0);

    free_solids(&f.sd);
}

/* Exterior wall height is SOLID_WALL_HEIGHT + this room's terrain z, based at
 * z0=0 — walls must not float on centroids_z. */
static void test_solids_exterior_height(void) {
    SolidsFix f;
    int       i;
    solids_fix_two_rooms(&f);
    build_solids_from_rooms(&f.sd);
    for (i = 0; i < f.sd.wall_list.count; i++) {
        check_f_near("solids exterior z0", f.sd.wall_list.walls[i].z0, 0.0f, 1e-4f);
        check_f_near(
            "solids exterior height", f.sd.wall_list.walls[i].height, SOLID_WALL_HEIGHT, 1e-4f);
    }
    free_solids(&f.sd);
}

/* Walking west out of room 0 hits its west face. Normal is the crossed face's
 * outward normal, so it points -X for a west wall. */
static void test_solids_sweep_west_wall(void) {
    SolidsFix f;
    SolidHit  hit;
    solids_fix_two_rooms(&f);
    build_solids_from_rooms(&f.sd);

    solids_poison_hit(&hit);
    check_i("sweep west hit",
            solid_sweep_xy(&f.sd, 20.0f, 50.0f, -10.0f, 50.0f, 0.0f, AGENT_HULL_RADIUS, &hit),
            1);
    check_ok("sweep west nx<0", hit.nx < 0.0f);
    check_f_near("sweep west ny", hit.ny, 0.0f, 1e-4f);
    check_ok("sweep west t in [0,1)", hit.t >= 0.0f && hit.t < 1.0f);

    free_solids(&f.sd);
}

/* The doorway must stay walkable: crossing x=100 is a miss. */
static void test_solids_sweep_portal(void) {
    SolidsFix f;
    SolidHit  hit;
    solids_fix_two_rooms(&f);
    build_solids_from_rooms(&f.sd);
    check_i("sweep portal miss",
            solid_sweep_xy(&f.sd, 80.0f, 50.0f, 120.0f, 50.0f, 0.0f, AGENT_HULL_RADIUS, &hit),
            0);
    free_solids(&f.sd);
}

/* Already past the plane (t<0) is a miss — v1 has no depenetration. */
static void test_solids_sweep_no_depenetration(void) {
    SolidsFix f;
    SolidHit  hit;
    solids_fix_two_rooms(&f);
    build_solids_from_rooms(&f.sd);
    /* Starts at x=-30, i.e. already outside past the x=0 face, moving further
     * out: the expanded plane is behind the start, so t<0 and we report free. */
    check_i("sweep t<0 miss",
            solid_sweep_xy(&f.sd, -30.0f, 50.0f, -60.0f, 50.0f, 0.0f, AGENT_HULL_RADIUS, &hit),
            0);
    free_solids(&f.sd);
}

/* Regression: starting INSIDE the r-band but still on the approach side of a
 * face is a hit clamped to t=0, not a miss.
 *
 * The old `if (t < 0 || t >= 1) continue;` rejected both cases together and
 * so left a 12u-wide hole along the approach side of every baked face: an
 * agent that got within AGENT_HULL_RADIUS of a wall (by jumping over a lip,
 * say) could then walk straight out through it. Pitfall when editing: keep
 * the "already past the plane" half a MISS, or v1 gains a depenetration
 * behaviour that _resolve_xy_collision is not written for.
 */
static void test_solids_sweep_inside_band(void) {
    SolidsFix f;
    SolidHit  hit;
    solids_fix_two_rooms(&f);
    build_solids_from_rooms(&f.sd);

    /* 5u inside the x=0 west face — closer than r=12 — heading out. */
    solids_poison_hit(&hit);
    check_i("sweep band west hit",
            solid_sweep_xy(&f.sd, 5.0f, 50.0f, -20.0f, 50.0f, 0.0f, AGENT_HULL_RADIUS, &hit),
            1);
    check_f_near("sweep band west t clamped", hit.t, 0.0f, 1e-6f);
    check_ok("sweep band west nx<0", hit.nx < 0.0f);

    /* Same start, heading back INTO the room: the face is behind us in the
     * direction of travel, so it must not block. */
    check_i("sweep band inward miss",
            solid_sweep_xy(&f.sd, 5.0f, 50.0f, 30.0f, 50.0f, 0.0f, AGENT_HULL_RADIUS, NULL),
            0);

    /* Horizontal faces take the same path: 5u inside y=0 heading south. */
    solids_poison_hit(&hit);
    check_i("sweep band south hit",
            solid_sweep_xy(&f.sd, 50.0f, 5.0f, 50.0f, -20.0f, 0.0f, AGENT_HULL_RADIUS, &hit),
            1);
    check_f_near("sweep band south t clamped", hit.t, 0.0f, 1e-6f);
    check_ok("sweep band south ny<0", hit.ny < 0.0f);

    /* Standing exactly on the wall line is the no-depenetration case in both
     * directions — an agent pinned on a seg must never be frozen solid. */
    check_i("sweep on-plane outward miss",
            solid_sweep_xy(&f.sd, 0.0f, 50.0f, -20.0f, 50.0f, 0.0f, AGENT_HULL_RADIUS, NULL),
            0);

    free_solids(&f.sd);
}

/* LoS through the doorway is clear; the shared north wall blocks. */
static void test_solids_ray_through_door(void) {
    SolidsFix f;
    solids_fix_two_rooms(&f);
    build_solids_from_rooms(&f.sd);
    check_i(
        "ray through door", solid_ray_clear(&f.sd, 10.0f, 50.0f, 48.0f, 190.0f, 50.0f, 48.0f), 1);
    free_solids(&f.sd);
}

static void test_solids_ray_north_wall(void) {
    SolidsFix f;
    solids_fix_two_rooms(&f);
    build_solids_from_rooms(&f.sd);
    check_i("ray north wall blocks",
            solid_ray_clear(&f.sd, 50.0f, 50.0f, 48.0f, 50.0f, 150.0f, 48.0f),
            0);
    free_solids(&f.sd);
}

/* Cliff-pruned shared edge → one lip spanning the drop, not a portal and not
 * a full SOLID_WALL_HEIGHT slab (that would hide the overlook).
 *
 * The drop is 64u, NOT SOLID_WALL_HEIGHT: with a 128u drop the lip
 * (z0=0, height=128) is bit-for-bit what an exterior wall of the low room
 * would be, so the test could not tell a lip from a mis-emitted wall. The
 * kind assertion below is the other half of that — a lip mis-tagged
 * EXTERIOR draws 4u pushed into the neighbouring room. */
static void test_solids_cliff_lip(void) {
    SolidsFix   f;
    int         i, lips = 0;
    const Wall* lip = NULL;
    solids_fix_cliff_z(&f, 64.0f);
    build_solids_from_rooms(&f.sd);

    for (i = 0; i < f.sd.wall_list.count; i++) {
        const Wall* w = &f.sd.wall_list.walls[i];
        if (fabsf(w->y1 - w->y0) < 1e-3f && fabsf(w->y0 - 100.0f) < 1e-3f) {
            lips++;
            lip = w;
        }
    }
    check_i("cliff lip emitted once", lips, 1);
    if (lip != NULL) {
        check_i("cliff lip kind", lip->kind, SOLID_KIND_LIP);
        check_f_near("cliff lip z0", lip->z0, 0.0f, 1e-4f);
        check_f_near("cliff lip height", lip->height, 64.0f, 1e-4f);
        check_f_near("cliff lip x0", lip->x0, 0.0f, 1e-4f);
        check_f_near("cliff lip x1", lip->x1, 100.0f, 1e-4f);
        /* Emitted by the HIGH room (1), so the outward normal points south,
         * down onto the low room. A flipped sign would draw the cube inside
         * the catwalk instead of over the drop. */
        check_f_near("cliff lip nx", lip->nx, 0.0f, 1e-4f);
        check_f_near("cliff lip ny", lip->ny, -1.0f, 1e-4f);
    }

    /* Eye level inside the drop is blocked; above the lip is the overlook
     * (an agent standing on the high room has its eye at 64+48=112). */
    check_i("cliff lip blocks eye",
            solid_ray_clear(&f.sd, 50.0f, 50.0f, 48.0f, 50.0f, 150.0f, 48.0f),
            0);
    check_i("cliff lip clear above",
            solid_ray_clear(&f.sd, 50.0f, 50.0f, 112.0f, 50.0f, 150.0f, 112.0f),
            1);

    /* Standing on the high room at z=64 must not walk off: slab overlap is
     * inclusive on [z0, z0+height]. One unit higher there is nothing left. */
    check_i("cliff lip blocks step-off",
            solid_sweep_xy(&f.sd, 50.0f, 150.0f, 50.0f, 50.0f, 64.0f, AGENT_HULL_RADIUS, NULL),
            1);
    check_i("cliff lip above slab is free",
            solid_sweep_xy(&f.sd, 50.0f, 150.0f, 50.0f, 50.0f, 65.0f, AGENT_HULL_RADIUS, NULL),
            0);

    free_solids(&f.sd);
}

/* IMPORTANT: adjacency wins over the drop. map.py exempts ramp endpoints from
 * cliff pruning, so on the real map catwalk(z=128) stays adjacent to
 * CT-ramp(z=64) across a 64u drop. Emitting a lip there put a 64u wall across
 * an edge the nav graph — and therefore nav-distance shaping — treats as
 * walkable, i.e. exactly the sim/nav drift this list exists to remove. */
static void test_solids_connected_drop_is_portal(void) {
    SolidsFix f;
    int       i, lips = 0;
    solids_fix_cliff_z(&f, 64.0f);
    solids_fix_adj(&f, 0, 1, 1); /* nav says this drop is walkable */
    build_solids_from_rooms(&f.sd);

    for (i = 0; i < f.sd.wall_list.count; i++)
        if (f.sd.wall_list.walls[i].kind == SOLID_KIND_LIP)
            lips++;
    check_i("connected drop no lip", lips, 0);
    check_i("connected drop count", f.sd.wall_list.count, 6);
    check_i("connected drop nothing on y=100", solids_count_hline(&f.sd, 100.0f), 0);
    /* Walking up the drop must be free in both directions. */
    check_i("connected drop walk north",
            solid_sweep_xy(&f.sd, 50.0f, 50.0f, 50.0f, 150.0f, 0.0f, AGENT_HULL_RADIUS, NULL),
            0);
    check_i("connected drop walk south",
            solid_sweep_xy(&f.sd, 50.0f, 150.0f, 50.0f, 50.0f, 64.0f, AGENT_HULL_RADIUS, NULL),
            0);
    free_solids(&f.sd);
}

/* A ramp never emits a lip even when its adjacency is pruned — its high edge
 * IS the walk-up. Removing the ramp guard walls off every ramp top. */
static void test_solids_ramp_emits_no_lip(void) {
    SolidsFix f;
    int       i, lips = 0;
    solids_fix_cliff_z(&f, 64.0f);
    f.ramps[1] = 1; /* the high room is the ramp */
    build_solids_from_rooms(&f.sd);

    for (i = 0; i < f.sd.wall_list.count; i++)
        if (f.sd.wall_list.walls[i].kind == SOLID_KIND_LIP)
            lips++;
    check_i("ramp no lip", lips, 0);
    check_i("ramp nothing on y=100", solids_count_hline(&f.sd, 100.0f), 0);
    free_solids(&f.sd);
}

/* Mirror of the above: the ramp is the LOW room, so the NON-ramp room is the
 * one that would emit (only the higher side emits a lip). Both sides name the
 * same line, so a guard that only tests the emitting room passes the test
 * above and still drops a 64u wall over the ramp's top here — which is the
 * one face you must be able to walk through to get off the ramp.
 * Pitfall when editing the bake: keep BOTH orientations, they exercise
 * different halves of `ramp_i || ramp_j`. */
static void test_solids_ramp_neighbour_emits_no_lip(void) {
    SolidsFix f;
    int       i, lips = 0;
    solids_fix_cliff_z(&f, 64.0f);
    f.ramps[0] = 1; /* the LOW room is the ramp; room 1 above would emit */
    build_solids_from_rooms(&f.sd);

    for (i = 0; i < f.sd.wall_list.count; i++)
        if (f.sd.wall_list.walls[i].kind == SOLID_KIND_LIP)
            lips++;
    check_i("ramp neighbour no lip", lips, 0);
    check_i("ramp neighbour lip count", f.sd.wall_list.count, 6);
    check_i("ramp neighbour nothing on y=100", solids_count_hline(&f.sd, 100.0f), 0);
    /* The whole point: stepping down off the ramp top must stay free. */
    check_i("ramp neighbour walk off ramp",
            solid_sweep_xy(&f.sd, 50.0f, 150.0f, 50.0f, 50.0f, 64.0f, AGENT_HULL_RADIUS, NULL),
            0);
    free_solids(&f.sd);
}

/* Flush + unconnected → ONE divider, emitted by the lower-indexed room.
 * SOLID_KIND_DIVIDER is unreachable on SIMPLE_ROOMS (map.py only prunes on
 * |Δz| > MAX_STEP_HEIGHT, which lands in the lip branch), so without this
 * fixture the whole branch — and the i<jj anti-duplicate guard — is dead. */
static void test_solids_divider(void) {
    SolidsFix   f;
    int         i, divs = 0;
    const Wall* div = NULL;
    solids_fix_two_rooms(&f);
    solids_fix_adj(&f, 0, 1, 0); /* prune the shared edge */
    build_solids_from_rooms(&f.sd);

    for (i = 0; i < f.sd.wall_list.count; i++) {
        const Wall* w = &f.sd.wall_list.walls[i];
        if (w->kind == SOLID_KIND_DIVIDER) {
            divs++;
            div = w;
        }
    }
    check_i("divider emitted once", divs, 1);
    check_i("divider total count", f.sd.wall_list.count, 7);
    check_i("divider on the shared line", solids_count_vline(&f.sd, 100.0f), 1);
    if (div != NULL) {
        check_f_near("divider z0", div->z0, 0.0f, 1e-4f);
        check_f_near("divider height", div->height, SOLID_WALL_HEIGHT, 1e-4f);
        check_f_near("divider y0", div->y0, 0.0f, 1e-4f);
        check_f_near("divider y1", div->y1, 100.0f, 1e-4f);
        /* Room 0 owns it (i<jj), so the normal points east, away from room 0. */
        check_f_near("divider nx", div->nx, 1.0f, 1e-4f);
    }
    /* The pruned edge must actually block, or the prune became a doorway. */
    check_i("divider blocks",
            solid_sweep_xy(&f.sd, 80.0f, 50.0f, 120.0f, 50.0f, 0.0f, AGENT_HULL_RADIUS, NULL),
            1);
    free_solids(&f.sd);
}

/* Same fixture, but the emitting room is a ramp: the guard covers the divider
 * branch too, not just the lip. A divider on a ramp face walls off the
 * connector exactly like a lip would. */
static void test_solids_ramp_emits_no_divider(void) {
    SolidsFix f;
    int       i, divs = 0;
    solids_fix_two_rooms(&f);
    solids_fix_adj(&f, 0, 1, 0);
    f.ramps[0] = 1; /* room 0 is the one that would emit (i<jj) */
    build_solids_from_rooms(&f.sd);

    for (i = 0; i < f.sd.wall_list.count; i++)
        if (f.sd.wall_list.walls[i].kind == SOLID_KIND_DIVIDER)
            divs++;
    check_i("ramp no divider", divs, 0);
    check_i("ramp divider count", f.sd.wall_list.count, 6);
    check_i("ramp nothing on x=100", solids_count_vline(&f.sd, 100.0f), 0);
    free_solids(&f.sd);
}

/* Mirror of the above, with the ramp at index 1. The divider is emitted by
 * the LOWER-indexed room (the `i < jj` anti-duplicate tie-break), so a guard
 * that only tests the emitting room makes the outcome depend on nothing but
 * array order: ramp at index 0 was silent, ramp at index 1 got a 128u wall
 * across its face. Array order is assigned by map.py's room list and carries
 * no geometric meaning, so this asymmetry is a bug, not a convention. */
static void test_solids_ramp_neighbour_emits_no_divider(void) {
    SolidsFix f;
    int       i, divs = 0;
    solids_fix_two_rooms(&f);
    solids_fix_adj(&f, 0, 1, 0);
    f.ramps[1] = 1; /* the ramp is the NON-emitting side (room 0 has i<jj) */
    build_solids_from_rooms(&f.sd);

    for (i = 0; i < f.sd.wall_list.count; i++)
        if (f.sd.wall_list.walls[i].kind == SOLID_KIND_DIVIDER)
            divs++;
    check_i("ramp neighbour no divider", divs, 0);
    check_i("ramp neighbour divider count", f.sd.wall_list.count, 6);
    check_i("ramp neighbour nothing on x=100", solids_count_vline(&f.sd, 100.0f), 0);
    /* Crossing onto the ramp must stay free in both directions. */
    check_i("ramp neighbour walk onto ramp",
            solid_sweep_xy(&f.sd, 80.0f, 50.0f, 120.0f, 50.0f, 0.0f, AGENT_HULL_RADIUS, NULL),
            0);
    free_solids(&f.sd);
}

/* Every exterior face is tagged EXTERIOR and carries the unit outward normal
 * of the room that emitted it. draw_walls pushes the 8u cube WALL_DEPTH/2
 * along that normal, so a flipped sign parks the cube INSIDE walkable tile
 * and a wrong kind moves it on the wrong faces. Checks all four edge
 * orientations — the west face alone cannot catch a per-edge sign flip. */
static void test_solids_exterior_normals(void) {
    SolidsFix f;
    int       i;
    solids_fix_two_rooms(&f);
    build_solids_from_rooms(&f.sd);

    for (i = 0; i < f.sd.wall_list.count; i++) {
        const Wall* w    = &f.sd.wall_list.walls[i];
        int         vert = fabsf(w->x1 - w->x0) < 1e-3f;
        check_i("exterior kind", w->kind, SOLID_KIND_EXTERIOR);
        /* Axis-aligned unit normal: exactly one component is ±1. */
        check_f_near("exterior normal unit", w->nx * w->nx + w->ny * w->ny, 1.0f, 1e-4f);
        if (vert) {
            check_f_near("exterior vert ny", w->ny, 0.0f, 1e-4f);
            check_f_near("exterior vert nx", w->nx, (fabsf(w->x0) < 1e-3f) ? -1.0f : 1.0f, 1e-4f);
        } else {
            check_f_near("exterior horiz nx", w->nx, 0.0f, 1e-4f);
            check_f_near("exterior horiz ny", w->ny, (fabsf(w->y0) < 1e-3f) ? -1.0f : 1.0f, 1e-4f);
        }
    }
    free_solids(&f.sd);
}

/* Exterior height is SOLID_WALL_HEIGHT + the EMITTING room's terrain z, so a
 * wall around an elevated room still reaches the same absolute top as its
 * flat neighbours. Both other fixtures sit at z=0, where dropping the `+ zi`
 * term is invisible. */
static void test_solids_exterior_height_elevated(void) {
    SolidsFix f;
    int       i;
    solids_fix_cliff_z(&f, 64.0f); /* room 0 at z=0, room 1 north at z=64 */
    build_solids_from_rooms(&f.sd);

    for (i = 0; i < f.sd.wall_list.count; i++) {
        const Wall* w = &f.sd.wall_list.walls[i];
        /* Owner: the lip sits on y=100; everything else belongs to whichever
         * room its span lies in (room 0 south of y=100, room 1 north). */
        float mid_y = (w->y0 + w->y1) * 0.5f;
        float want;
        if (w->kind == SOLID_KIND_LIP)
            continue;
        want = (mid_y < 100.0f) ? SOLID_WALL_HEIGHT : SOLID_WALL_HEIGHT + 64.0f;
        check_f_near("elevated exterior z0", w->z0, 0.0f, 1e-4f);
        check_f_near("elevated exterior height", w->height, want, 1e-4f);
    }
    free_solids(&f.sd);
}

/* Partial edge coverage: a doorway flanked by two walls on ONE edge.
 *
 * This is the case the map is built around and the only one that reaches the
 * mid-edge gap emit in Pass A (`covs[k].lo > cursor + SOLID_EPS`). Every
 * other fixture has edges that are either fully covered or fully open, so
 * that branch never runs and the gap walls could silently vanish.
 *
 * Room 0 is a tall 100x300 room; room 1 is a 100x100 room touching the
 * middle third of its east edge, connected. Room 0's east edge must come
 * back as: wall [0,100] · doorway [100,200] · wall [200,300].
 *
 * Room 0 sits at z=64 so the flank walls also pin the `SOLID_WALL_HEIGHT +
 * zi` term on the mid-edge emit — every other elevated fixture is fully
 * covered or fully open, so it only exercises the trailing emit.
 */
static void solids_fix_doorway(SolidsFix* f) {
    solids_fix(f);
    solids_fix_room(f, 0, 0.0f, 0.0f, 100.0f, 300.0f, 64.0f, 0);
    solids_fix_room(f, 1, 100.0f, 100.0f, 200.0f, 200.0f, 0.0f, 0);
    solids_fix_adj(f, 0, 1, 1);
}

static void test_solids_partial_edge_doorway(void) {
    SolidsFix   f;
    int         i, flanks = 0;
    const Wall *south = NULL, *north = NULL;
    solids_fix_doorway(&f);
    build_solids_from_rooms(&f.sd);

    for (i = 0; i < f.sd.wall_list.count; i++) {
        const Wall* w = &f.sd.wall_list.walls[i];
        if (fabsf(w->x1 - w->x0) < 1e-3f && fabsf(w->x0 - 100.0f) < 1e-3f) {
            flanks++;
            if (w->y0 < 50.0f)
                south = w;
            else
                north = w;
        }
    }
    check_i("doorway two flanking walls", flanks, 2);
    check_i("doorway total count", f.sd.wall_list.count, 8);
    if (south != NULL) {
        /* The gap BEFORE the covered interval — the mid-edge emit. */
        check_f_near("doorway south flank y0", south->y0, 0.0f, 1e-4f);
        check_f_near("doorway south flank y1", south->y1, 100.0f, 1e-4f);
        check_i("doorway south flank kind", south->kind, SOLID_KIND_EXTERIOR);
        /* Room 0 is elevated: the mid-edge emit must add its zi too. */
        check_f_near("doorway south flank z0", south->z0, 0.0f, 1e-4f);
        check_f_near("doorway south flank height", south->height, SOLID_WALL_HEIGHT + 64.0f, 1e-4f);
        /* Outward normal (+x: room 0 lies west of its east edge). The mid-edge
         * emit is the only path that sets nx/ny on a gap wall, and no other
         * fixture reaches it, so a sign flip there is invisible elsewhere. */
        check_f_near("doorway south flank nx", south->nx, 1.0f, 1e-4f);
        check_f_near("doorway south flank ny", south->ny, 0.0f, 1e-4f);
    }
    if (north != NULL) {
        /* The trailing gap after the last covered interval. */
        check_f_near("doorway north flank y0", north->y0, 200.0f, 1e-4f);
        check_f_near("doorway north flank y1", north->y1, 300.0f, 1e-4f);
        check_f_near("doorway north flank height", north->height, SOLID_WALL_HEIGHT + 64.0f, 1e-4f);
        check_f_near("doorway north flank nx", north->nx, 1.0f, 1e-4f);
        check_f_near("doorway north flank ny", north->ny, 0.0f, 1e-4f);
    }

    /* Walk through the door: free. Walk at the flank: blocked. */
    check_i("doorway walk through",
            solid_sweep_xy(&f.sd, 80.0f, 150.0f, 120.0f, 150.0f, 0.0f, AGENT_HULL_RADIUS, NULL),
            0);
    check_i("doorway walk into flank",
            solid_sweep_xy(&f.sd, 80.0f, 50.0f, 120.0f, 50.0f, 0.0f, AGENT_HULL_RADIUS, NULL),
            1);
    /* LoS agrees with movement: clear through the gap, blocked at the flank. */
    check_i("doorway ray through",
            solid_ray_clear(&f.sd, 50.0f, 150.0f, 48.0f, 150.0f, 150.0f, 48.0f),
            1);
    check_i("doorway ray into flank",
            solid_ray_clear(&f.sd, 50.0f, 50.0f, 48.0f, 150.0f, 50.0f, 48.0f),
            0);
    free_solids(&f.sd);
}

/* The sweep is a capsule, not a point: r expands the face BOTH along its
 * normal (you are stopped r units early) and past its endpoints (you clip
 * the door jamb). Zeroing r in either place lets agents corner-clip through
 * geometry that is drawn solid. */
static void test_solids_sweep_hull_radius(void) {
    SolidsFix f;
    SolidHit  hit;
    solids_fix_doorway(&f);
    build_solids_from_rooms(&f.sd);

    /* Stops short of x=100: the destination never reaches the plane, but the
     * hull does. With r=0 this is a miss. */
    solids_poison_hit(&hit);
    check_i("sweep r stops short",
            solid_sweep_xy(&f.sd, 80.0f, 50.0f, 95.0f, 50.0f, 0.0f, AGENT_HULL_RADIUS, &hit),
            1);
    check_ok("sweep r hit before dest", hit.t < 1.0f);
    check_i("sweep r=0 same move misses",
            solid_sweep_xy(&f.sd, 80.0f, 50.0f, 95.0f, 50.0f, 0.0f, 0.0f, NULL),
            0);

    /* Door jamb: crossing at y=108 is 8u inside the doorway [100,200] but
     * only 8u past the flank wall's end at y=100, so the hull clips it. */
    check_i("sweep r clips jamb",
            solid_sweep_xy(&f.sd, 80.0f, 108.0f, 120.0f, 108.0f, 0.0f, AGENT_HULL_RADIUS, NULL),
            1);
    check_i("sweep r=0 clears jamb",
            solid_sweep_xy(&f.sd, 80.0f, 108.0f, 120.0f, 108.0f, 0.0f, 0.0f, NULL),
            0);
    /* Well clear of both jambs is free at any radius. */
    check_i("sweep mid-door free",
            solid_sweep_xy(&f.sd, 80.0f, 150.0f, 120.0f, 150.0f, 0.0f, AGENT_HULL_RADIUS, NULL),
            0);
    free_solids(&f.sd);
}

/* Two faces on one diagonal move → the NEAREST must win, or slide response
 * projects against a wall the agent has not reached yet.
 *
 * Scope: the start is OUTSIDE both r-bands, which is the only regime where
 * this property holds. From inside a band the band face clamps to t=0 and
 * wins whatever lies ahead — see the CALLER CONTRACT block on solid_sweep_xy.
 * Do not "strengthen" this test by moving the start closer to a wall. */
static void test_solids_sweep_nearest_hit(void) {
    SolidsFix f;
    SolidHit  hit;
    solids_fix_two_rooms(&f);
    build_solids_from_rooms(&f.sd);

    /* From (20,60) to (-40,-60): the west face x=0 is crossed at t≈0.133,
     * the south face y=0 at t≈0.4. */
    solids_poison_hit(&hit);
    check_i("sweep corner hit",
            solid_sweep_xy(&f.sd, 20.0f, 60.0f, -40.0f, -60.0f, 0.0f, AGENT_HULL_RADIUS, &hit),
            1);
    check_f_near("sweep nearest t", hit.t, 8.0f / 60.0f, 1e-4f);
    check_f_near("sweep nearest nx", hit.nx, -1.0f, 1e-4f);
    check_f_near("sweep nearest ny", hit.ny, 0.0f, 1e-4f);
    free_solids(&f.sd);
}

/* A ray lying exactly in a face's plane is blocked — the collinear branch and
 * _solid_span_range, which no other test reaches. An agent standing dead on a
 * wall line must not get free LoS along it. */
static void test_solids_ray_collinear(void) {
    SolidsFix f;
    solids_fix_two_rooms(&f);
    build_solids_from_rooms(&f.sd);

    /* Along the x=0 face, inside its y span [0,100] and its z slab [0,128]. */
    check_i(
        "ray collinear blocked", solid_ray_clear(&f.sd, 0.0f, 10.0f, 48.0f, 0.0f, 90.0f, 48.0f), 0);
    /* Same line, but the segment sits past the end of the face's span. */
    check_i("ray collinear off-span clear",
            solid_ray_clear(&f.sd, 0.0f, 150.0f, 48.0f, 0.0f, 250.0f, 48.0f),
            1);
    /* Same line and span, but above the 128u slab. */
    check_i("ray collinear above slab clear",
            solid_ray_clear(&f.sd, 0.0f, 10.0f, 200.0f, 0.0f, 90.0f, 200.0f),
            1);
    free_solids(&f.sd);
}

/* Rebaking must free the previous list, not leak or double-free it. */
static void test_solids_rebake(void) {
    SolidsFix f;
    int       first;
    solids_fix_two_rooms(&f);
    build_solids_from_rooms(&f.sd);
    first = f.sd.wall_list.count;
    build_solids_from_rooms(&f.sd);
    check_i("rebake same count", f.sd.wall_list.count, first);
    check_ok("rebake list live", f.sd.wall_list.walls != NULL);
    /* The bake allocates an O(N^2) upper bound and then shrinks to fit; a
     * capacity still stuck at 8N(N+1) means the realloc tail went missing. */
    check_i("rebake capacity shrunk", f.sd.wall_list.capacity, first);
    free_solids(&f.sd);
    check_ok("free_solids nulls", f.sd.wall_list.walls == NULL);
    check_i("free_solids count", f.sd.wall_list.count, 0);
    check_i("free_solids capacity", f.sd.wall_list.capacity, 0);
    /* Idempotent: c_close may run after an explicit free. */
    free_solids(&f.sd);
    check_ok("free_solids idempotent", f.sd.wall_list.walls == NULL);
}

/* dust2 publishes no room quads — the bake is a no-op and both queries stay
 * permissive so the raster path keeps owning collision there. */
static void test_solids_null_bounds_noop(void) {
    SolidsFix f;
    solids_fix_two_rooms(&f);
    f.sd.area_bounds = NULL;
    build_solids_from_rooms(&f.sd);
    check_i("null bounds no walls", f.sd.wall_list.count, 0);
    check_ok("null bounds no alloc", f.sd.wall_list.walls == NULL);
    check_i("null bounds sweep free",
            solid_sweep_xy(&f.sd, 20.0f, 50.0f, -10.0f, 50.0f, 0.0f, AGENT_HULL_RADIUS, NULL),
            0);
    check_i("null bounds ray clear",
            solid_ray_clear(&f.sd, 50.0f, 50.0f, 48.0f, 50.0f, 150.0f, 48.0f),
            1);
    free_solids(&f.sd);
}

int main(void) {
    test_shot_pulse();
    test_no_shot_when_bit0();
    test_shot_other_agent();
    test_all_agents_shot();
    test_footstep_hypot();
    test_footstep_two_ticks();
    test_airborne_no_foot();
    test_dead_no_foot();
    test_z_only_no_foot();
    test_plant_and_beep_0_to_639();
    test_no_beep_100_to_99();
    test_beep_96_to_95();
    test_beep_80_to_79();
    test_prev_eq_curr_after_init();
    test_unplanted_no_beep();
    test_punch_decay_formula();
    test_punch_decay_same_sign_smaller_abs();
    test_punch_decay_zero_and_dt0();
    test_recoil_decay_punch_by_name();
    test_aim_yaw0_pitch0();
    test_aim_yaw0_pitch_halfpi();
    test_aim_yaw_halfpi_pitch0();
    test_ramp_t_topology();
    test_ramp_y_slope_stairs();
    test_terrain_z_t_ramp();
    test_terrain_z_y_stairs();
    test_reload_start_agent0();
    test_reload_start_agent3();
    test_reload_prev_eq_curr_silent();
    test_reload_countdown_silent();
    test_reload_end_agent0();
    test_reload_countdown_not_end();
    test_solids_bake_two_rooms();
    test_solids_exterior_height();
    test_solids_sweep_west_wall();
    test_solids_sweep_portal();
    test_solids_sweep_no_depenetration();
    test_solids_sweep_inside_band();
    test_solids_ray_through_door();
    test_solids_ray_north_wall();
    test_solids_cliff_lip();
    test_solids_connected_drop_is_portal();
    test_solids_ramp_emits_no_lip();
    test_solids_ramp_neighbour_emits_no_lip();
    test_solids_divider();
    test_solids_ramp_emits_no_divider();
    test_solids_ramp_neighbour_emits_no_divider();
    test_solids_exterior_normals();
    test_solids_exterior_height_elevated();
    test_solids_partial_edge_doorway();
    test_solids_sweep_hull_radius();
    test_solids_sweep_nearest_hit();
    test_solids_ray_collinear();
    test_solids_rebake();
    test_solids_null_bounds_noop();

    if (g_fails) {
        fprintf(stderr, "demo_events_test: %d check(s) failed\n", g_fails);
        return 1;
    }
    printf("demo_events_test: all checks passed\n");
    return 0;
}
