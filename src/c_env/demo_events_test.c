/* demo_events_test.c — headless checks for cs2_demo_events.h.
 *
 * Built/run by `zig build demo_events_test`. No Raylib, no binding.so,
 * no ctypes. The play wrapper's 3 Hz footstep drop is intentionally
 * NOT tested here — a walking tick with hypot>1 must set the foot bit
 * every sim tick.
 */
#include <stdio.h>
#include <string.h>

#include "cs2_demo_events.h"

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

    if (g_fails) {
        fprintf(stderr, "demo_events_test: %d check(s) failed\n", g_fails);
        return 1;
    }
    printf("demo_events_test: all checks passed\n");
    return 0;
}
