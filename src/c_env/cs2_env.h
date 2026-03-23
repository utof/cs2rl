#pragma once
#include "cs2_types.h"

/* ── Public interface ────────────────────────────────────────────────────── */
void env_init(Dust2Env* env, StaticData* sd, uint32_t seed, float team_spirit);
void env_reset(Dust2Env* env);
void env_step(Dust2Env* env, const int32_t* actions);
void env_close(Dust2Env* env);
