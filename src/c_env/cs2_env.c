/* cs2_env.c — standalone smoke test binary.
 * Not compiled into the Python extension (see CMakeLists.txt).
 * Links against a zero-initialised StaticData — no nav mesh loaded.
 * Purpose: verify the env compiles and runs without Python. */
#include "cs2_env.h"
#include <stdio.h>

int main(void) {
    StaticData sd;
    memset(&sd, 0, sizeof(sd));
    sd.round_time         = 640;
    sd.stale_memory_tick  = -9999;
    sd.n_t_spawns         = 0;
    sd.n_ct_spawns        = 0;
    Dust2Env env;
    env_init(&env, &sd, 42, 0.0f);
    env_reset(&env);
    int32_t actions[N_AGENTS * ACTION_DIM] = {0};
    env_step(&env, actions);
    env_close(&env);
    printf("cs2_env standalone: OK\n");
    return 0;
}
