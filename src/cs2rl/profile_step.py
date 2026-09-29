"""Performance profiler for the Dust2 simulation stack.

This script is meant to be rerun whenever `env/nav.py`, `c_env`, or the binding
changes. It benchmarks the same environment at several layers so regressions
are easy to localize:

- cs2_env: full Cs2Env.step() wrapper path
- cs2_env_shared_buf: same, with external/shared buffers (PufferLib-like path)
- cs2_env_manual_reset: wrapper with explicit terms.any() + manual reset
- binding_direct: binding.step() called directly — minimal Python overhead

It also captures `cProfile` summaries for the cs2_env path and
writes machine-readable JSON reports for regression tracking.

Examples:
    uv run python -m cs2rl.profile_step
    uv run python -m cs2rl.profile_step --steps 30000 --action-mode noop
    uv run python -m cs2rl.profile_step --action-mode random --no-cprofile
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cs2rl.c_env.cs2_env import make_env as make_c_env
from cs2rl.env.nav import N_AGENTS, OBS_DIM
from cs2rl.spec.action import ACTION_DIM, AIM_DIM
from cs2rl.spec.paths import LOGS_DIR

REPORT_DIR = LOGS_DIR / "profiles"
NOOP_ACTION_C = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32)
# Batch 3: zero Δyaw buffer for the binding-direct profile stepper.
# binding.step now requires a third arg (float32 (N_AGENTS, AIM_DIM)).
NOOP_CONT_C = np.zeros((N_AGENTS, AIM_DIM), dtype=np.float32)


@dataclass
class BenchResult:
    name: str
    steps: int
    seconds: float
    sps: float
    us_per_step: float


def _build_external_buf() -> dict[str, np.ndarray]:
    return {
        "observations": np.zeros((N_AGENTS, OBS_DIM), dtype=np.float32),
        "rewards": np.zeros(N_AGENTS, dtype=np.float32),
        "terminals": np.zeros(N_AGENTS, dtype=bool),
        "truncations": np.zeros(N_AGENTS, dtype=bool),
        "masks": np.ones(N_AGENTS, dtype=bool),
        "actions": np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32),
    }


def _timed_loop(step_fn, steps: int) -> BenchResult:
    t0 = time.perf_counter()
    for _ in range(steps):
        step_fn()
    elapsed = time.perf_counter() - t0
    sps = steps / elapsed
    return BenchResult(
        name="",
        steps=steps,
        seconds=elapsed,
        sps=sps,
        us_per_step=1e6 / sps,
    )


def _profile_loop(step_fn, steps: int, top_n: int) -> dict[str, list[dict[str, Any]]]:
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(steps):
        step_fn()
    pr.disable()
    return {
        "cumulative": _extract_profile_rows(pr, "cumulative", top_n),
        "self_time": _extract_profile_rows(pr, "tottime", top_n),
    }


def _extract_profile_rows(pr: cProfile.Profile, sort_by: str, top_n: int) -> list[dict[str, Any]]:
    stats = pstats.Stats(pr, stream=io.StringIO())
    items = []
    for func, (cc, nc, tt, ct, _callers) in stats.stats.items():
        filename, line, name = func
        items.append({
            "function": f"{Path(filename).name}:{line}:{name}",
            "primitive_calls": int(cc),
            "total_calls": int(nc),
            "self_seconds": float(tt),
            "cumulative_seconds": float(ct),
        })
    key = "cumulative_seconds" if sort_by == "cumulative" else "self_seconds"
    items.sort(key=lambda row: row[key], reverse=True)
    return items[:top_n]


def _make_cs2_env_stepper(seed: int, action_mode: str, external_buf: bool = False):
    env = make_c_env(seed=seed, buf=_build_external_buf() if external_buf else None)
    env.reset(seed=seed)
    joint_space = env.action_space

    def step():
        if action_mode == "random":
            actions = np.asarray(joint_space.sample(), dtype=np.int32)
        else:
            actions = NOOP_ACTION_C
        env.step(actions)

    return step


def _make_cs2_env_manual_reset_stepper(seed: int):
    env = make_c_env(seed=seed, auto_reset=False)
    env.reset(seed=seed)

    def step():
        _obs, _rewards, terms, _truncs, _infos = env.step(NOOP_ACTION_C)
        if terms.any():
            env.reset(seed=seed)

    return step


def _make_binding_direct_stepper(seed: int):
    from cs2rl.c_env import binding

    env = make_c_env(seed=seed, auto_reset=False)
    env.reset(seed=seed)
    capsule = env._capsule

    def step():
        binding.step(capsule, NOOP_ACTION_C, NOOP_CONT_C)
        if env._c_env.game.round_over:
            binding.reset(capsule)

    return step


def _run_benchmark(name: str, stepper_factory, steps: int, warmup: int) -> BenchResult:
    step_fn = stepper_factory()
    for _ in range(warmup):
        step_fn()
    result = _timed_loop(step_fn, steps)
    result.name = name
    return result


def _collect_low_hanging_fruit(benchmarks: dict[str, BenchResult]) -> list[str]:
    notes = []
    raw = benchmarks["binding_direct"].sps
    wrapper = benchmarks["cs2_env"].sps
    shared = benchmarks["cs2_env_shared_buf"].sps
    manual = benchmarks["cs2_env_manual_reset"].sps

    if wrapper < raw * 0.8:
        notes.append(
            "Cs2Env wrapper overhead is significant relative to binding_direct; inspect "
            "action marshaling, buffer syncing, and terminal/reset handling in cs2_env.py.")
    if shared < wrapper * 0.9:
        notes.append("Shared-buffer mode is materially slower than the internal-buffer path; "
                     "external output copies are a likely low-hanging fruit.")
    if manual < wrapper * 0.9:
        notes.append("Benchmark harness overhead is distorting SPS; avoid terms.any() + manual "
                     "reset when profiling the native auto-reset env.")
    return notes


def _write_reports(report: dict[str, Any], report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    latest_path = report_dir / "latest.json"
    archive_path = report_dir / f"profile-{timestamp}.json"
    latest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    archive_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return latest_path, archive_path


def _print_summary(
    benchmarks: dict[str, BenchResult],
    wrapper_profile: dict[str, list[dict[str, Any]]] | None,
    notes: list[str],
    latest_path: Path,
    archive_path: Path,
):
    print("\n=== BENCHMARKS ===")
    for key in (
            "cs2_env",
            "cs2_env_shared_buf",
            "cs2_env_manual_reset",
            "binding_direct",
    ):
        result = benchmarks[key]
        print(f"{result.name:24} {result.sps:10.0f} SPS  "
              f"{result.us_per_step:8.2f} us/step  "
              f"{result.seconds:7.3f}s total")

    raw = benchmarks["binding_direct"].sps
    wrapper = benchmarks["cs2_env"].sps
    shared = benchmarks["cs2_env_shared_buf"].sps
    print("\n=== RATIOS ===")
    print(f"cs2_env/binding_direct {wrapper / raw:10.3f}")
    print(f"shared_buf/cs2_env     {shared / wrapper:10.3f}")

    if notes:
        print("\n=== LOW-HANGING FRUIT ===")
        for note in notes:
            print(f"- {note}")

    def print_profile_block(title: str, block: dict[str, list[dict[str, Any]]] | None):
        if not block:
            return
        print(f"\n=== {title} TOP CUMULATIVE ===")
        for row in block["cumulative"][:10]:
            print(f"- {row['function']}  cum={row['cumulative_seconds']:.4f}s  "
                  f"self={row['self_seconds']:.4f}s  calls={row['total_calls']}")
        print(f"\n=== {title} TOP SELF TIME ===")
        for row in block["self_time"][:10]:
            print(f"- {row['function']}  self={row['self_seconds']:.4f}s  "
                  f"cum={row['cumulative_seconds']:.4f}s  calls={row['total_calls']}")

    print_profile_block("CS2_ENV", wrapper_profile)

    print("\n=== REPORTS ===")
    print(f"- latest:  {latest_path}")
    print(f"- archive: {archive_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--warmup", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-mode", choices=("noop", "random"), default="noop")
    parser.add_argument("--profile-steps", type=int, default=2_000)
    parser.add_argument("--profile-top", type=int, default=20)
    parser.add_argument("--no-cprofile", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    benchmarks = {
        "cs2_env":
        _run_benchmark(
            "cs2_env",
            lambda: _make_cs2_env_stepper(args.seed, args.action_mode, external_buf=False),
            args.steps,
            args.warmup,
        ),
        "cs2_env_shared_buf":
        _run_benchmark(
            "cs2_env_shared_buf",
            lambda: _make_cs2_env_stepper(args.seed, args.action_mode, external_buf=True),
            args.steps,
            args.warmup,
        ),
        "cs2_env_manual_reset":
        _run_benchmark(
            "cs2_env_manual_reset",
            lambda: _make_cs2_env_manual_reset_stepper(args.seed),
            args.steps,
            args.warmup,
        ),
        "binding_direct":
        _run_benchmark(
            "binding_direct",
            lambda: _make_binding_direct_stepper(args.seed),
            args.steps,
            args.warmup,
        ),
    }

    wrapper_profile = None
    if not args.no_cprofile:
        wrapper_profile = _profile_loop(
            _make_cs2_env_stepper(args.seed, args.action_mode, external_buf=False),
            args.profile_steps,
            args.profile_top,
        )

    notes = _collect_low_hanging_fruit(benchmarks)
    report = {
        "meta": {
            "steps": args.steps,
            "warmup": args.warmup,
            "seed": args.seed,
            "action_mode": args.action_mode,
            "profile_steps": args.profile_steps,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "benchmarks": {
            name: asdict(result)
            for name, result in benchmarks.items()
        },
        "profiles": {
            "cs2_env": wrapper_profile,
        },
        "notes": notes,
    }
    latest_path, archive_path = _write_reports(report, args.report_dir)
    _print_summary(
        benchmarks,
        wrapper_profile,
        notes,
        latest_path,
        archive_path,
    )


if __name__ == "__main__":
    main()
