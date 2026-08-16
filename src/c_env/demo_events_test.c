/* demo_events_test.c — headless checks for cs2_demo_events.h / cs2_demo_viz.h.
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

#ifdef RAYLIB_H
#error "cs2_demo_events.h must not include raylib.h (demo/tests only)"
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
    test_reload_start_agent0();
    test_reload_start_agent3();
    test_reload_prev_eq_curr_silent();
    test_reload_countdown_silent();

    if (g_fails) {
        fprintf(stderr, "demo_events_test: %d check(s) failed\n", g_fails);
        return 1;
    }
    printf("demo_events_test: all checks passed\n");
    return 0;
}
