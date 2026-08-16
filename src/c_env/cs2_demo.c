#define _DEFAULT_SOURCE
/* cs2_demo.c — Standalone CS2RL Raylib demo. Phase 6. */
#include "cs2_env.h"
#include "nav_data.h"
#include "cs2_play_host.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <limits.h>

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

static int path_exists(const char* p) {
    return p && p[0] && access(p, F_OK) == 0;
}

static void join_path(char* out, size_t n, const char* a, const char* b) {
    size_t la = strlen(a);
    if (la > 0 && a[la - 1] == '/')
        snprintf(out, n, "%s%s", a, b);
    else
        snprintf(out, n, "%s/%s", a, b);
}

static int dirname_inplace(char* path) {
    size_t n = strlen(path);
    while (n > 1 && path[n - 1] == '/')
        path[--n] = '\0';
    char* slash = strrchr(path, '/');
    if (!slash)
        return 0;
    if (slash == path) {
        path[1] = '\0';
        return 1;
    }
    *slash = '\0';
    return 1;
}

/* realpath if the file exists; else keep absolute PATH, else cwd + "/" + PATH. */
static void absolutize_policy(const char* path, char* out, size_t n) {
    if (realpath(path, out))
        return;
    if (path[0] == '/') {
        snprintf(out, n, "%s", path);
        return;
    }
    char cwd[PATH_MAX];
    if (!getcwd(cwd, sizeof(cwd))) {
        snprintf(out, n, "%s", path);
        return;
    }
    join_path(out, n, cwd, path);
}

/* Walk up from the binary dir for pyproject.toml + src/play.py. */
static int find_repo(char* out, size_t n) {
    const char* app = play_host_app_dir();
    if (!app || !app[0])
        return 0;
    char cur[PATH_MAX];
    snprintf(cur, sizeof(cur), "%s", app);
    size_t len = strlen(cur);
    while (len > 1 && cur[len - 1] == '/')
        cur[--len] = '\0';
    for (;;) {
        char toml[PATH_MAX], play[PATH_MAX];
        join_path(toml, sizeof(toml), cur, "pyproject.toml");
        join_path(play, sizeof(play), cur, "src/play.py");
        if (path_exists(toml) && path_exists(play)) {
            snprintf(out, n, "%s", cur);
            return 1;
        }
        if (cur[0] == '/' && cur[1] == '\0')
            return 0;
        if (!dirname_inplace(cur))
            return 0;
    }
}

static int resolve_python(const char* repo, char* out, size_t n) {
    const char* uve = getenv("UV_PROJECT_ENVIRONMENT");
    if (uve && uve[0]) {
        join_path(out, n, uve, "bin/python");
        if (path_exists(out))
            return 1;
    }
    const char* cs2 = getenv("CS2RL_VENV");
    if (cs2 && cs2[0]) {
        join_path(out, n, cs2, "bin/python");
        if (path_exists(out))
            return 1;
    }
    join_path(out, n, repo, ".venv/bin/python");
    if (path_exists(out))
        return 1;
    char cur[PATH_MAX];
    snprintf(cur, sizeof(cur), "%s", repo);
    while (dirname_inplace(cur)) {
        join_path(out, n, cur, ".venv/bin/python");
        if (path_exists(out))
            return 1;
        if (cur[0] == '/' && cur[1] == '\0')
            break;
    }
    return 0;
}

static void print_borrow_hint(const char* abs_policy, int argc, char** argv) {
    fprintf(stderr, "cs2_demo: $PYTHON src/play.py --policy %s", abs_policy);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--policy") == 0 && i + 1 < argc) {
            i++;
            continue;
        }
        fprintf(stderr, " %s", argv[i]);
    }
    fprintf(stderr,
            "\nset UV_PROJECT_ENVIRONMENT to the parent .venv; do not uv sync in the worktree\n");
}

/* --policy path: never InitWindow / play_host_attach. exec borrowed python. */
static int demo_exec_play(int argc, char** argv, const char* policy_path) {
    char abs_policy[PATH_MAX];
    char repo[PATH_MAX];
    char python[PATH_MAX];

    absolutize_policy(policy_path, abs_policy, sizeof(abs_policy));

    if (!find_repo(repo, sizeof(repo)) || !resolve_python(repo, python, sizeof(python))) {
        print_borrow_hint(abs_policy, argc, argv);
        return 2;
    }

    /* $UV_PROJECT_ENVIRONMENT / $CS2RL_VENV may be relative to launch cwd.
     * realpath the parent, not the file: .venv/bin/python is often a symlink
     * to the base interpreter, and execv of that target drops the venv. */
    char abs_python[PATH_MAX];
    char parent[PATH_MAX];
    char abs_parent[PATH_MAX];
    const char* base = strrchr(python, '/');
    if (!base || !base[1]) {
        print_borrow_hint(abs_policy, argc, argv);
        return 2;
    }
    snprintf(parent, sizeof(parent), "%s", python);
    if (!dirname_inplace(parent) || !realpath(parent, abs_parent)) {
        print_borrow_hint(abs_policy, argc, argv);
        return 2;
    }
    join_path(abs_python, sizeof(abs_python), abs_parent, base + 1);

    char* eargv[argc + 5]; /* python, src/play.py, --policy, abs, rest, NULL */
    int   n = 0;
    eargv[n++] = abs_python;
    eargv[n++] = "src/play.py";
    eargv[n++] = "--policy";
    eargv[n++] = abs_policy;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--policy") == 0 && i + 1 < argc) {
            i++;
            continue;
        }
        eargv[n++] = argv[i];
    }
    eargv[n] = NULL;

    if (chdir(repo) != 0) {
        print_borrow_hint(abs_policy, argc, argv);
        return 2;
    }
    execv(abs_python, eargv);
    print_borrow_hint(abs_policy, argc, argv);
    return 2;
}

int main(int argc, char** argv) {
    int         human_idx   = 0;
    int         fog_enabled = 0;
    const char* policy      = NULL;
    /* Argv parsing — order-independent so --spectate --fog and --fog --spectate
     * both work. Unknown args are silently ignored (keeps backward compat with
     * existing scripts that pass --record, --eval, etc. to the trainer demo).
     *
     *   --policy PATH : exec src/play.py (never open a window here).
     *   --spectate : detach camera from any agent (free-fly, render all).
     *   --fog      : human-agent fog-of-war — only draw enemies your agent's
     *                line_of_sight_2d says are visible. Forces you to play
     *                with the SAME perception the bot gets in its obs vector.
     *                Useful for debugging "why didn't the bot react to that?"
     *                — if you also can't see them with --fog, the bot's obs
     *                doesn't contain that enemy either. Ignored in spectate
     *                mode (no "viewer" agent to filter from). */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--policy") == 0 && i + 1 < argc)
            policy = argv[++i];
        else if (strcmp(argv[i], "--spectate") == 0)
            human_idx = -1;
        else if (strcmp(argv[i], "--fog") == 0)
            fog_enabled = 1;
    }

    if (policy)
        return demo_exec_play(argc, argv, policy);

    StaticData sd;
    load_nav_data(&sd);

    Dust2Env env;
    memset(&env, 0, sizeof(Dust2Env));
    env.sd  = &sd;
    env.rng = 12345;
    env_reset(&env);
    env.recoil_enabled = 1; /* #120: punch on the hit ray + camera */

    PlayHost* h =
        play_host_attach(&env, human_idx, fog_enabled, (const float*)NAV_AREA_BOUNDS, NAV_N, NULL);
    if (!h) {
        fprintf(stderr, "play_host_attach failed\n");
        return 2;
    }

    int32_t actions[N_AGENTS * ACTION_DIM] = {0};
    /* Batch 3 (continuous-aim H-PPO): env_step gained a second action buffer
     * for the Gaussian aim head — (N_AGENTS, AIM_DIM) float32. Demo doesn't
     * need policy-driven aim (the human player has aim_rad set via mouse
     * delta in human_input(); RL agents in this demo path get zeros). */
    float cont[N_AGENTS * AIM_DIM] = {0};

    double next_step = play_host_time(h);
    while (!play_host_should_close(h)) {
        double now = play_host_time(h);
        if (now >= next_step) {
            play_host_begin_tick(h);
            if (human_idx >= 0)
                play_host_apply_human(h, actions);
            env_step(&env, actions, cont);
            play_host_end_tick(h);
            next_step += 1.0 / 16.0;
        }
        play_host_render(h);
        if (env.terminals[0]) {
            env_reset(&env);
            play_host_on_reset(h);
        }
    }

    play_host_detach(h);
    return 0;
}
