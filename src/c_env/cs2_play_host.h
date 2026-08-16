/* cs2_play_host.h — opaque Raylib attach ABI for cs2_demo + ctypes.
 * Demo target only. Do not include from cs2_env.h or binding.c. */
#pragma once
#include <stdint.h>

typedef struct PlayHost PlayHost;

PlayHost*   play_host_attach(void*        env,
                             int          human_idx,
                             int          fog_enabled,
                             const float* area_bounds,
                             int          n_areas,
                             const char*  resource_dir);
int         play_host_should_close(PlayHost* h);
double      play_host_time(PlayHost* h);
void        play_host_begin_tick(PlayHost* h);
void        play_host_apply_human(PlayHost* h, int32_t* actions);
void        play_host_end_tick(PlayHost* h);
void        play_host_render(PlayHost* h);
void        play_host_on_reset(PlayHost* h);
void        play_host_detach(PlayHost* h); /* NULL is a no-op */
const char* play_host_app_dir(void);
