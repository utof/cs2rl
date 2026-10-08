#pragma once
#include "cs2_types.h"
#include "cs2_bomb.h"
#include "cs2_weapons.h"
#include "cs2_movement.h" /* SV_JUMP_IMPULSE_CS for obs[8] normalisation */

static void
compute_observations(Dust2Env* env, int t_alive, int ct_alive, int8_t vis10[N_AGENTS][N_AGENTS]) {
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;
    float       map_diag;
    {
        float xr = 1.0f / sd->inv_x_range; /* x range */
        float yr = 1.0f / sd->inv_y_range;
        map_diag = sqrtf(xr * xr + yr * yr);
    }

    /* Batch 6 Task 2.5 (spec R9/D4): per-agent nearest bombsite-area centroid,
     * found in ONE pass over the nav areas (outside the agent loop) instead of
     * a per-agent scan. No fixed-size site list: dust2 flags 530 bombsite
     * areas (285 A + 245 B), so any "sites are few" cap assumption breaks.
     * Cost: N flag checks + n_sites×N_AGENTS distance updates per tick.
     * best_site == -1 ⇔ the map has no bombsite (slots then stay 0). */
    int   best_site[N_AGENTS];
    float best_site_d2[N_AGENTS];
    for (int i = 0; i < N_AGENTS; i++) {
        best_site[i]    = -1;
        best_site_d2[i] = 1e30f;
    }
    for (int k = 0; k < sd->N; k++) {
        if (!sd->bombsite_by_idx[k])
            continue;
        float cx = sd->centroid_xy[k * 2];
        float cy = sd->centroid_xy[k * 2 + 1];
        for (int i = 0; i < N_AGENTS; i++) {
            AgentState* ai = &g->agents[i];
            float       dx = cx - ai->x, dy = cy - ai->y;
            float       d2 = dx * dx + dy * dy;
            if (d2 < best_site_d2[i]) {
                best_site_d2[i] = d2;
                best_site[i]    = k;
            }
        }
    }

    for (int i = 0; i < N_AGENTS; i++) {
        float*      obs = &env->observations[i * OBS_DIM];
        AgentState* a   = &g->agents[i];

        memset(obs, 0, OBS_DIM * sizeof(float));

        /* ── Self state (0-27; +25..+27 are the Batch 6 goal-direction slots) ── */
        int              slot = a->weapon_slot;
        const WeaponDef* def  = &WEAPON_DEFS[slot];
        obs[0]                = a->hp / 100.0f;
        obs[1]                = a->armor / 100.0f;
        obs[2]                = (float)a->has_helmet;
        obs[3]                = a->x * sd->inv_x_range - sd->x_offset;
        obs[4]                = a->y * sd->inv_y_range - sd->y_offset;
        /* Self Z: jump apex is ~57 u, so normalise by 128 (WALL_HEIGHT-ish)
         * to keep values within ~[-1,1] for the foreseeable range.
         * Range note (Batch 5 verticality): with terrain z up to 128 (catwalk)
         * plus jump apex ≈57u, obs[5] can reach ≈1.45. Network ingests the
         * unclamped float same as velocity slots; no scaling change needed.
         * Teammate/enemy z-delta slots (were 0.0f placeholders) are now
         * (other->z - self->z)/128 — positive = other is above self. */
        obs[5] = a->z / 128.0f;
        obs[6] = (map_diag > 0.0f) ? a->vx / 250.0f : 0.0f;
        obs[7] = (map_diag > 0.0f) ? a->vy / 250.0f : 0.0f;
        /* Vz: jump impulse = 302 u/s, terminal gravity fall ~580 u/s before
         * damage threshold — normalise by the impulse so obs[8] ≈ 1.0 at
         * jump start and ≈ -1.0 at apex-return landing speed. */
        obs[8]  = a->vz / SV_JUMP_IMPULSE_CS;
        obs[9]  = sinf(a->facing);
        obs[10] = cosf(a->facing);
        /* Batch 3.5 (#24): pitch sin/cos appended after yaw cos. Mirrors the yaw
         * encoding pattern (sin/cos pair) so a linear model can recover pitch
         * directly. Range: pitch ∈ [-π/2, +π/2] → sin ∈ [-1,1], cos ∈ [0,1].
         * Pitfall: this insertion shifts EVERY downstream obs[N] for N≥13 by +2.
         * Audit obs index manifest comments AND test files for hardcoded indices. */
        obs[11] = sinf(a->pitch);
        obs[12] = cosf(a->pitch);
        obs[13] = (float)a->is_crouching;
        obs[14] = (slot == 0) ? 1.0f : 0.0f;
        obs[15] = (slot == 1) ? 1.0f : 0.0f;
        obs[16] = (slot == 2) ? 1.0f : 0.0f;
        obs[17] = (def->mag_size > 0) ? a->ammo_clip[slot] / (float)def->mag_size : 1.0f;
        obs[18] = (def->reserve_mags > 0) ? a->ammo_reserve[slot] / (float)def->reserve_mags : 1.0f;
        obs[19] = (a->reload_ticks > 0) ? 1.0f : 0.0f;
        obs[20] = (a->reload_ticks > 0 && def->reload_ticks > 0)
                      ? (def->reload_ticks - a->reload_ticks) / (float)def->reload_ticks
                      : 0.0f;
        obs[21] =
            (a->fire_cd > 0 && def->cycle_ticks > 0) ? a->fire_cd / (float)def->cycle_ticks : 0.0f;
        obs[22] = (float)(a->team == 0 && bomb_carrier(g) == i);
        obs[23] = (float)a->alive;
        obs[24] = (float)(a->team == 0);

        /* ── Goal direction (Batch 6 Task 2.5, spec R9/D4): slots +25..+27 ──
         * [sin(rel_bearing), cos(rel_bearing), xy_dist/map_diag] to the
         * Euclidean-NEAREST bombsite area centroid, where
         *   rel_bearing = wrap_pi(atan2(site_y - y, site_x - x) - facing).
         * Why: without a goal-direction slot a BC clone must memorize
         * absolute-position → direction over the whole map (spec R9); with it,
         * "turn until sin≈0 with cos>0, then walk" generalizes off the
         * demonstrated routes. sin/cos pair (not a normalized angle) matches
         * every other angle encoding in this file and stays continuous when
         * the site is directly behind. Sign: rel_bearing > 0 ⇔ site is
         * counter-clockwise of facing ⇔ positive Δyaw turns toward it.
         * Written for ALL agents (CTs know the map too), dead or alive, same
         * as the rest of the self block.
         * Pitfalls: straight-line XY bearing may point through a wall (the
         * policy/expert still routes via nav); at the exact centroid
         * atan2f(0,0)=0 makes bearing meaningless — but dist≈0 there, which is
         * the "arrived" signal; distance uses the same map_diag half-diagonal
         * normalizer as the teammate/enemy dx/dy/dist slots. */
        if (best_site[i] >= 0) {
            float bx  = sd->centroid_xy[best_site[i] * 2] - a->x;
            float by  = sd->centroid_xy[best_site[i] * 2 + 1] - a->y;
            float rel = wrap_pi(atan2f(by, bx) - a->facing);

            obs[OBS_SELF_BASE + 25] = sinf(rel);
            obs[OBS_SELF_BASE + 26] = cosf(rel);
            obs[OBS_SELF_BASE + 27] = (map_diag > 0.0f) ? sqrtf(best_site_d2[i]) / map_diag : 0.0f;
        }
        /* No bombsite on the map: slots stay 0 from memset. */

        /* ── Teammates (OBS_TEAMMATE_BASE ..): 4 × 7 ── */
        int tm_start = (a->team == 0) ? 0 : TEAM_SIZE;
        int tm_count = 0;
        for (int j = tm_start; j < tm_start + TEAM_SIZE && tm_count < 4; j++) {
            if (j == i)
                continue;
            AgentState* tm   = &g->agents[j];
            int         base = OBS_TEAMMATE_BASE + tm_count * OBS_TEAMMATE_STRIDE;
            if (tm->alive) {
                float dx = tm->x - a->x, dy = tm->y - a->y;
                obs[base + 0] = (map_diag > 0.0f) ? dx / map_diag : 0.0f;
                obs[base + 1] = (map_diag > 0.0f) ? dy / map_diag : 0.0f;
                /* Relative z-delta: positive = teammate above us.  Same /128
                 * scale as self obs[5].  Range: catwalk(128) - spawn(0) = +1.0;
                 * spawn - catwalk = -1.0.  Was constant-0 placeholder pre-T4. */
                obs[base + 2] = (tm->z - a->z) / 128.0f;
                obs[base + 3] = tm->hp / 100.0f;
                obs[base + 4] = 1.0f;
                float angle   = atan2f(dy, dx);
                obs[base + 5] = sinf(angle);
                obs[base + 6] = cosf(angle);
            }
            /* dead teammate: all zeros (already memset) */
            tm_count++;
        }

        /* ── Enemies (OBS_ENEMY_BASE ..): 5 × 8 ── */
        int en_start = (a->team == 0) ? TEAM_SIZE : 0;
        /* Sort by KNOWN distance — insertion sort over 5 elements.
         * F10 (2026-07-06 adversarial review): the key used to be the TRUE
         * distance for all 5 enemies unconditionally, so slot order (and
         * per-slot flag churn) leaked the rank of enemies the agent could
         * not see — an unseen enemy walking closer would reorder the slots.
         * The key now uses only information the policy legitimately has:
         *   visible enemy          → true squared distance,
         *   invisible w/ memory    → squared distance to LAST-KNOWN centroid
         *                            (the same position the obs slot emits),
         *   invisible, no memory   → 1e30f sentinel (sorted last; insertion
         *   (incl. dead: can_see=0)  sort is stable ⇒ ties keep index order).
         * Dead enemies always have can_see=0 (see the slot loop below), so
         * they rank by stale memory or the sentinel — never by their true
         * corpse position. */
        int   order[TEAM_SIZE];
        float dists[TEAM_SIZE];
        for (int s = 0; s < TEAM_SIZE; s++) {
            order[s]        = en_start + s;
            AgentState* en  = &g->agents[order[s]];
            int         ecs = en->alive ? vis10[i][order[s]] : 0;
            if (ecs) {
                float dx = en->x - a->x, dy = en->y - a->y;
                dists[s] = dx * dx + dy * dy;
            } else if (a->enemy_mem_idx[s] != INVALID_AREA_IDX) {
                float mx = sd->centroid_xy[a->enemy_mem_idx[s] * 2] - a->x;
                float my = sd->centroid_xy[a->enemy_mem_idx[s] * 2 + 1] - a->y;
                dists[s] = mx * mx + my * my;
            } else {
                dists[s] = 1e30f;
            }
        }
        for (int s = 1; s < TEAM_SIZE; s++) {
            int   ko = order[s];
            float kd = dists[s];
            int   t  = s - 1;
            while (t >= 0 && dists[t] > kd) {
                order[t + 1] = order[t];
                dists[t + 1] = dists[t];
                t--;
            }
            order[t + 1] = ko;
            dists[t + 1] = kd;
        }
        for (int slot2 = 0; slot2 < TEAM_SIZE; slot2++) {
            int         ej      = order[slot2];
            AgentState* en      = &g->agents[ej];
            int         base    = OBS_ENEMY_BASE + slot2 * OBS_ENEMY_STRIDE;
            int         mem_s   = ej - en_start;
            int         can_see = en->alive ? vis10[i][ej] : 0;

            obs[base + 4] = (float)en->alive;
            obs[base + 3] = (float)can_see;

            if (can_see) {
                float dx = en->x - a->x, dy = en->y - a->y;
                float dist = sqrtf(dx * dx + dy * dy);
                /* R0-E.1 (#130): FACING-RELATIVE frame, mirroring the bombsite
                 * bearing in the self block above (wrap_pi(atan2 - facing)).
                 * rel-pos is rotated by -facing so +x = "ahead", +y = "left";
                 * bearing sin/cos below are of the same relative angle, so a
                 * Δyaw = rel puts the enemy dead ahead — the yaw head no
                 * longer has to learn sin(θ - f) from an absolute θ and its
                 * own facing. Teammate slots stay absolute — not aim targets.
                 * PITFALL: obs semantics changed (SIM_OBS_VERSION sim-v3);
                 * deploy OBS_VERSION / the ONNX sidecar are still absolute —
                 * do not export post-R0-E.1 policies. */
                float rel = wrap_pi(atan2f(dy, dx) - a->facing);
                float cf = cosf(a->facing), sf = sinf(a->facing);
                float rx = dx * cf + dy * sf, ry = -dx * sf + dy * cf;
                obs[base + 0] = (map_diag > 0.0f) ? rx / map_diag : 0.0f;
                obs[base + 1] = (map_diag > 0.0f) ? ry / map_diag : 0.0f;
                /* Relative z-delta: positive = enemy above us.  Same /128
                 * scale as self obs[5] and teammate slot.  Range mirrors
                 * teammate block.  Was constant-0 placeholder pre-T4.
                 * NB: visibility-gated — when !can_see (invisible / memory-only),
                 * obs[base+2] stays 0 from memset.  Asymmetry vs teammate slot
                 * (which writes z-delta whenever tm->alive).  Policies must
                 * disambiguate "0 = invisible enemy" from "0 = same height" via
                 * obs[base+3] (can_see flag at base+3). */
                obs[base + 2] = (en->z - a->z) / 128.0f;
                obs[base + 5] = sinf(rel);
                obs[base + 6] = cosf(rel);
                obs[base + 7] = (map_diag > 0.0f) ? dist / map_diag : 0.0f;
            } else if (a->enemy_mem_idx[mem_s] != INVALID_AREA_IDX) {
                /* Use last-known position from memory — R0-E.1: same
                 * facing-relative frame as the visible branch, so slot +0/+1
                 * mean "ahead/left" regardless of which branch wrote them. */
                float mx = sd->centroid_xy[a->enemy_mem_idx[mem_s] * 2] - a->x;
                float my = sd->centroid_xy[a->enemy_mem_idx[mem_s] * 2 + 1] - a->y;
                float cf = cosf(a->facing), sf = sinf(a->facing);
                float rx = mx * cf + my * sf, ry = -mx * sf + my * cf;
                obs[base + 0] = (map_diag > 0.0f) ? rx / map_diag : 0.0f;
                obs[base + 1] = (map_diag > 0.0f) ? ry / map_diag : 0.0f;
            }
        }

        /* ── Global / bomb block: obs[OBS_GLOBAL_BASE + 0 .. +13] ──
         * Written base-relative (not bare 93..106) so the whole block shifts
         * automatically if an upstream block (self/teammate/enemy) is resized.
         * Slot map: +0 round-time · +1..4 bomb one-hot · +5,6 bomb xy ·
         * +7 bomb-z placeholder · +8 bomb timer · +9 plant · +10 defuse ·
         * +11 t_alive · +12 ct_alive · +13 designated-carrier bit. */
        int gb      = OBS_GLOBAL_BASE;
        obs[gb + 0] = g->round_ticks_left / (float)sd->round_time;
        /* Bomb status one-hot (+1..+4): carried by self, carried by a
         * teammate (a CT sees neither), dropped, planted (from the plant on,
         * including the resolved phases). */
        int carrier = bomb_carrier(g);
        if (carrier >= 0) {
            if (carrier == i)
                obs[gb + 1] = 1.0f; /* carried by self */
            else if (a->team == 0)
                obs[gb + 2] = 1.0f; /* carried by teammate */
        } else if (g->bomb.phase == BOMB_DROPPED) {
            obs[gb + 3] = 1.0f;
        } else if (bomb_planted(g)) {
            obs[gb + 4] = 1.0f;
        }
        /* bomb position (+5,+6) */
        if (bomb_on_ground(g)) {
            float bx = g->bomb.x - a->x, by = g->bomb.y - a->y;
            obs[gb + 5] = (map_diag > 0.0f) ? bx / map_diag : 0.0f;
            obs[gb + 6] = (map_diag > 0.0f) ? by / map_diag : 0.0f;
        } else if (carrier >= 0 && a->team == 0 && carrier != i) {
            /* Teammate carrying: show their position */
            float bx    = g->agents[carrier].x - a->x;
            float by    = g->agents[carrier].y - a->y;
            obs[gb + 5] = (map_diag > 0.0f) ? bx / map_diag : 0.0f;
            obs[gb + 6] = (map_diag > 0.0f) ? by / map_diag : 0.0f;
        }
        /* Bomb z deferred — out of scope for T4 (spec L4 covers teammate/enemy
         * z-delta only).  Bomb z would require obs version contract for plug-in.
         * See gh #(filed) for the bomb-z-aware obs follow-up. */
        obs[gb + 7] = 0.0f; /* bomb z placeholder (intentionally constant pre-followup) */
        obs[gb + 8] = bomb_planted(g) ? g->bomb.ticks_left / (float)sd->bomb_timer : 0.0f;
        obs[gb + 9] = (g->bomb.phase == BOMB_PLANTING && sd->bomb_plant_time > 0)
                          ? g->bomb.progress / (float)sd->bomb_plant_time
                          : 0.0f;
        /* +10: defuse progress (DEFUSING, and DEFUSED on its completion tick)
         * — extract to avoid GCC statement-expression */
        {
            float defuse_prog = 0.0f;
            int   defuser     = bomb_defuser(g);
            if (defuser >= 0) {
                AgentState* def2  = &g->agents[defuser];
                int         dtime = def2->has_kit ? sd->bomb_defuse_kit : sd->bomb_defuse_time;
                if (dtime > 0)
                    defuse_prog = g->bomb.progress / (float)dtime;
            }
            obs[gb + 10] = defuse_prog;
        }
        /* Rung 0: normalise by the ACTIVE team size so 1v1 reads 1.0, not 0.2. */
        obs[gb + 11] = t_alive / (float)sd->n_active_per_team;
        obs[gb + 12] = ct_alive / (float)sd->n_active_per_team;

        /* Batch 2: round-fixed designated-carrier role bit (T-side semantic).
         * 1.0 only when this agent is the round's designated bomb carrier
         * (set in env_reset, never reassigned). Distinct from obs[22]
         * (transient self-has-bomb) — gives the policy a stable identity
         * signal that survives drop/pickup. CT agents always read 0.0. */
        obs[gb + 13] = (a->team == 0 && i == g->round_designated_carrier_id) ? 1.0f : 0.0f;

        /* Clip all obs to (-5, 5) */
        for (int k = 0; k < OBS_DIM; k++) {
            if (obs[k] > 5.0f)
                obs[k] = 5.0f;
            else if (obs[k] < -5.0f)
                obs[k] = -5.0f;
        }
    }
}
