#!/usr/bin/env python
"""Gate 0 feasibility measurement for the BC warm-start batch (Batch 6, Task 1).

Measures spawn->plant tick budgets for the scripted bomber on the SIMPLE map
(make_simple_map(), 1 bombsite, area 6) against the REAL round timer
(ROUND_TIME = 640 ticks; never lifted). Two driver variants:

  A. direct-poke  — the tests/test_env_feasibility.py driver: poke a->facing
     each tick, action = move-forward. Fastest possible traversal; upper bound
     on feasibility.
  B. action-interface — the real BC-expert interface (spec D-5/F2): facing is
     steered via the continuous [dyaw, pitch] head, dyaw clamped by the env to
     +/-max_turn_speed (pi/4 rad/tick) and applied AFTER movement (cs2_env.h:107
     movement, :133 dyaw). So each tick we command the facing we want for the
     NEXT tick's movement — "one tick ahead". Strictly slower than A; this is
     what demo generation (Task 3) will actually run.

Decision rule (plan Task 1): if a workable fraction of spawn->plant runs
complete within 640 ticks -> GO (proceed to Task 2). If near-zero -> STOP and
escalate (shorten routes / speed driver / curriculum-first).

Why a standalone throwaway script and not a test: this is a one-shot
measurement that gates the batch; tests/test_scripted_expert.py is reserved
for Task 2's extracted-expert regression test.

Pitfalls encoded here (so later tasks don't rediscover them):
  * auto_reset=False — a mid-run silent reset would corrupt tick counts.
  * knife (weapon_slot=2) poked on the bomber — highest wishspeed (250 u/s);
    the Task-2/3 expert should do the same (or switch via the weapon head and
    eat the switch ticks).
  * The plant press (discrete head 4 = USE) must be held BOMB_PLANT_TIME
    ticks and those ticks count against the same 640 budget.
  * Direct facing pokes survive env.step only because the default continuous
    buffer is all-zero (dyaw=0 keeps the poked value). Poking pitch does NOT
    survive (absolute-pitch overwrite, cs2_env.h:170) — irrelevant here.
"""

import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np                     # noqa: E402

from _action_spec import AIM_DIM                                                       # noqa: E402
from c_env.cs2_env import make_env                                                     # noqa: E402
from map import make_simple_map                                                        # noqa: E402
from nav import ACTION_DIM, BOMB_PLANT_TIME, MAX_TURN_SPEED_RAD, ROUND_TIME, TEAM_SIZE # noqa: E402

# Batch 6 Task 2: helpers extracted from tests/test_env_feasibility.py into
# src/scripted_expert.py; they now take MapData directly (no NavGraph/shim).
from scripted_expert import area_centroid, bfs_area_path, bombsite_areas # noqa: E402

N_AGENTS = 10
HEAD_USE = 4                           # discrete head order: move=0 shoot=1 reload=2 weapon=3 use=4 crouch=5 jump=6
JITTER_SEQ = [0.0, math.pi / 8, -math.pi / 8, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2]


def _wrap_pi(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def run_episode(map_data, seed: int, bomber_idx: int, variant: str):
    """One spawn->plant attempt. Returns a result dict.

    variant: "poke" (driver A) or "iface" (driver B). Tick accounting: every
    env.step is one tick; the run FAILS the budget the moment ticks would
    exceed ROUND_TIME without bomb_planted.
    """
    env = make_env(seed=seed, map_data=map_data, auto_reset=False)
    env.reset()
    g = env._c_env.game

    # Carrier assignment + knife, exactly as the feasibility test does.
    for i in range(N_AGENTS):
        g.agents[i].has_bomb = 0
    bomber = g.agents[bomber_idx]
    bomber.has_bomb = 1
    g.bomb_carrier_id = bomber_idx
    bomber.weapon_slot = 2
    bomber.weapon_slot_target = 2
    bomber.switch_ticks = 0

    bombsites = bombsite_areas(env.map_data)
    start_area = int(env.map_data.area_ids[bomber.area_idx])
    path = bfs_area_path(env.map_data, start_area, bombsites)
    if not path:
        env.close()
        return dict(seed=seed,
                    bomber=bomber_idx,
                    start=start_area,
                    hops=0,
                    reach_ticks=None,
                    total_ticks=None,
                    planted=False,
                    note="NO-PATH")

    ticks = 0
    disc = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32)
    cont = np.zeros((N_AGENTS, AIM_DIM), dtype=np.float32)

    def step_move(target_facing: float):
        """One driving tick toward target_facing (variant-dependent steering)."""
        nonlocal ticks
        disc[:] = 0
        disc[bomber_idx, 0] = 1                        # move forward (facing-local)
        cont[:] = 0.0
        if variant == "poke":
            bomber.facing = target_facing              # dyaw stays 0 -> poke survives
        else:
                                                       # One-tick-ahead steering: movement this tick uses facing from the
                                                       # END of last tick; the dyaw we command now lands after movement
                                                       # and is what NEXT tick's movement will use (spec D-5/F2).
            dyaw = _wrap_pi(target_facing - float(bomber.facing))
            cont[bomber_idx, 0] = max(-MAX_TURN_SPEED_RAD, min(MAX_TURN_SPEED_RAD, dyaw))
        env.step(disc, cont)
        ticks += 1

    # --- walk the area path (adapted from _drive_agent_through_area_path,
    #     instrumented with a global ROUND_TIME budget instead of per-hop only)
    reached = True
    for target_area in path[1:]:
        last_pos, stuck, hop_ok = None, 0, False
        while ticks < ROUND_TIME:
            cur_area = int(env.map_data.area_ids[bomber.area_idx])
            if cur_area == target_area:
                hop_ok = True
                break
            pos = (round(float(bomber.x), 1), round(float(bomber.y), 1))
            if pos == last_pos:
                stuck += 1
            else:
                stuck, last_pos = 0, pos
            cx, cy = area_centroid(env.map_data, target_area)
            base = math.atan2(float(cy) - float(bomber.y), float(cx) - float(bomber.x))
            step_move(base + JITTER_SEQ[min(stuck // 3, len(JITTER_SEQ) - 1)])
        if not hop_ok:
            reached = False
            break

    reach_ticks = ticks if reached else None

    # --- plant (USE held for BOMB_PLANT_TIME ticks, still on the budget)
    planted = False
    if reached:
        while ticks < ROUND_TIME and not bool(g.bomb_planted):
            disc[:] = 0
            disc[bomber_idx, HEAD_USE] = 1
            cont[:] = 0.0
            env.step(disc, cont)
            ticks += 1
        planted = bool(g.bomb_planted)

    env.close()
    return dict(seed=seed,
                bomber=bomber_idx,
                start=start_area,
                hops=len(path) - 1,
                reach_ticks=reach_ticks,
                total_ticks=ticks if planted else None,
                planted=planted,
                note="ok" if planted else ">640/stuck")


def report(tag: str, results: list[dict]):
    print(f"\n=== {tag} ===")
    print(f"{'seed':>4} {'agent':>5} {'spawn':>5} {'hops':>4} {'reach':>6} {'total':>6} "
          f"{'planted':>7}  note")
    for r in results:
        print(f"{r['seed']:>4} {r['bomber']:>5} {r['start']:>5} {r['hops']:>4} "
              f"{r['reach_ticks'] if r['reach_ticks'] is not None else '-':>6} "
              f"{r['total_ticks'] if r['total_ticks'] is not None else '-':>6} "
              f"{str(r['planted']):>7}  {r['note']}")
    totals = [r["total_ticks"] for r in results if r["planted"]]
    n = len(results)
    print(f"-- {len(totals)}/{n} planted within {ROUND_TIME} ticks "
          f"({100.0 * len(totals) / max(n, 1):.0f}% in budget)")
    if totals:
        print(f"-- ticks-to-plant  min={min(totals)}  median={statistics.median(totals):.0f}  "
              f"max={max(totals)}   (plant press = {BOMB_PLANT_TIME} ticks of that)")
    return len(totals), n


def main():
    seeds = list(range(10))
    map_data = make_simple_map()

    # Variant A: direct-poke, every T agent (spawn slot) x every seed.
    res_a = [run_episode(map_data, s, b, "poke") for s in seeds for b in range(TEAM_SIZE)]
    ok_a, n_a = report(
        f"Variant A — direct-poke driver ({len(seeds)} seeds x {TEAM_SIZE} T-spawns)", res_a)

    # Variant B: real action interface (rate-limited dyaw, one-tick-ahead facing).
    res_b = [run_episode(map_data, s, b, "iface") for s in seeds for b in range(TEAM_SIZE)]
    ok_b, n_b = report(
        f"Variant B — action-interface driver ({len(seeds)} seeds x {TEAM_SIZE} T-spawns)", res_b)

    print("\n=== Gate 0 verdict ===")
    frac_b = ok_b / max(n_b, 1)
    if frac_b >= 0.8:
        print(f"GO — action-interface variant plants in budget on {ok_b}/{n_b} runs.")
    elif frac_b > 0.05:
        print(
            f"MARGINAL — only {ok_b}/{n_b} in budget on the real interface; review per-spawn table."
        )
    else:
        print(
            f"STOP — near-zero in-budget completions ({ok_b}/{n_b}); batch infeasible as designed (spec §8 R1)."
        )


if __name__ == "__main__":
    main()
