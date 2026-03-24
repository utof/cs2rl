/* cs2_env.c — standalone smoke test binary.
 * Not compiled into the Python extension (see CMakeLists.txt).
 * Links against a zero-initialised StaticData — no nav mesh loaded.
 * Purpose: verify the env compiles and runs without Python. */
#include "cs2_env.h"
#include <stdio.h>

int main(void) {
    StaticData sd;
    int8_t     vis_matrix[1]     = {0};
    float      centroid_xy[2]    = {0.0f, 0.0f};
    int32_t    area_ids[1]       = {0};
    int8_t     bombsite_by_idx[1] = {0};
    float      bombsite_dist[1]  = {0.0f};
    memset(&sd, 0, sizeof(sd));
    sd.N                  = 1;
    sd.vis_matrix         = vis_matrix;
    sd.centroid_xy        = centroid_xy;
    sd.area_ids           = area_ids;
    sd.bombsite_by_idx    = bombsite_by_idx;
    sd.bombsite_dist      = bombsite_dist;
    sd.max_area_id        = 0;
    sd.round_time         = 640;
    sd.stale_memory_tick  = -9999;
    sd.n_t_spawns         = 1;
    sd.t_spawns[0]        = 0;
    sd.n_ct_spawns        = 1;
    sd.ct_spawns[0]       = 0;
    Dust2Env env;
    env_init(&env, &sd, 42, 0.0f);
    env_reset(&env);
    int32_t actions[N_AGENTS * ACTION_DIM] = {0};
    env_step(&env, actions);
    env_close(&env);
    printf("cs2_env standalone: OK\n");
    return 0;
}
