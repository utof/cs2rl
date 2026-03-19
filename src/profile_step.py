"""Performance profiler for the Dust2 simulation stack.

This script is meant to be rerun whenever `sim.py`, `c_env`, or the wrapper
changes. It benchmarks the same environment at several layers so regressions
are easy to localize:

- Python env step() in `sim.py`
- Native C wrapper step() in `src/c_env/wrapper.py`
- Native C wrapper with external/shared buffers (PufferLib-like path)
- Raw `env_step()` C kernel without Python-side wrapper work
- A deliberately "bad" benchmark path that does `terms.any()` + manual reset

It also captures `cProfile` summaries for the Python and wrapper paths and
writes machine-readable JSON reports for regression tracking.

Examples:
    uv run python src/profile_step.py
    uv run python src/profile_step.py --steps 30000 --action-mode noop
    uv run python src/profile_step.py --action-mode random --no-cprofile
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

from c_env.wrapper import ACTION_DIM, N_AGENTS, OBS_DIM, _lib
from c_env.wrapper import make_env as make_c_env
from paths import LOGS_DIR
from sim import Dust2Env

REPORT_DIR = LOGS_DIR / "profiles"
NOOP_ACTION_C = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32)
NOOP_ACTION_PY = {
    aid: np.zeros(ACTION_DIM, dtype=np.int32)
    for aid in [*(f"t{i}" for i in range(5)), *(f"ct{i}" for i in range(5))]
}


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
        items.append(
            {
                "function": f"{Path(filename).name}:{line}:{name}",
                "primitive_calls": int(cc),
                "total_calls": int(nc),
                "self_seconds": float(tt),
                "cumulative_seconds": float(ct),
            }
        )
    key = "cumulative_seconds" if sort_by == "cumulative" else "self_seconds"
    items.sort(key=lambda row: row[key], reverse=True)
    return items[:top_n]


def _make_python_stepper(seed: int, action_mode: str):
    env = Dust2Env()
    env.reset(seed=seed)
    action_spaces = {aid: env.action_space(aid) for aid in env.possible_agents}

    def step():
        if action_mode == "random":
            actions = {
                aid: np.asarray(action_spaces[aid].sample(), dtype=np.int32) for aid in env.agents
            }
        else:
            actions = NOOP_ACTION_PY

        _obs, _rewards, terms, _truncs, _infos = env.step(actions)
        if all(terms.get(aid, False) for aid in env.possible_agents):
            env.reset(seed=seed)

    return step


def _make_c_wrapper_stepper(seed: int, action_mode: str, external_buf: bool = False):
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


def _make_c_manual_reset_stepper(seed: int):
    env = make_c_env(seed=seed)
    env.reset(seed=seed)

    def step():
        _obs, _rewards, terms, _truncs, _infos = env.step(NOOP_ACTION_C)
        if terms.any():
            env.reset(seed=seed)

    return step


def _make_raw_c_stepper(seed: int):
    env = make_c_env(seed=seed)
    env.reset(seed=seed)
    action_addr = int(NOOP_ACTION_C.ctypes.data)

    def step():
        _lib.env_step(env._c_env_p, action_addr)
        if env._c_env.game.round_over:
            _lib.env_reset(env._c_env_p)

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
    raw = benchmarks["c_raw_kernel"].sps
    wrapper = benchmarks["c_wrapper"].sps
    shared = benchmarks["c_wrapper_shared_buf"].sps
    manual = benchmarks["c_wrapper_manual_reset"].sps
    python = benchmarks["python_env"].sps

    if wrapper < raw * 0.8:
        notes.append(
            "C wrapper overhead is significant relative to the raw kernel; inspect "
            "action marshaling, "
            "buffer syncing, and terminal/reset handling in src/c_env/wrapper.py."
        )
    if shared < wrapper * 0.9:
        notes.append(
            "Shared-buffer mode is materially slower than the internal-buffer path; "
            "external output copies "
            "are a likely low-hanging fruit."
        )
    if manual < wrapper * 0.9:
        notes.append(
            "Benchmark harness overhead is distorting SPS; avoid terms.any() + manual "
            "reset when profiling "
            "the native auto-reset env."
        )
    if python < wrapper * 0.5:
        notes.append(
            "The Python sim is far slower than the native path; start with the top "
            "cumulative cProfile rows "
            "from sim.py for the next optimization pass."
        )
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
    python_profile: dict[str, list[dict[str, Any]]] | None,
    wrapper_profile: dict[str, list[dict[str, Any]]] | None,
    notes: list[str],
    latest_path: Path,
    archive_path: Path,
):
    print("\n=== BENCHMARKS ===")
    for key in (
        "python_env",
        "c_wrapper",
        "c_wrapper_shared_buf",
        "c_wrapper_manual_reset",
        "c_raw_kernel",
    ):
        result = benchmarks[key]
        print(
            f"{result.name:24} {result.sps:10.0f} SPS  "
            f"{result.us_per_step:8.2f} us/step  "
            f"{result.seconds:7.3f}s total"
        )

    raw = benchmarks["c_raw_kernel"].sps
    wrapper = benchmarks["c_wrapper"].sps
    shared = benchmarks["c_wrapper_shared_buf"].sps
    python = benchmarks["python_env"].sps
    print("\n=== RATIOS ===")
    print(f"wrapper/raw_c          {wrapper / raw:10.3f}")
    print(f"shared_buf/wrapper     {shared / wrapper:10.3f}")
    print(f"python/wrapper         {python / wrapper:10.3f}")

    if notes:
        print("\n=== LOW-HANGING FRUIT ===")
        for note in notes:
            print(f"- {note}")

    def print_profile_block(title: str, block: dict[str, list[dict[str, Any]]] | None):
        if not block:
            return
        print(f"\n=== {title} TOP CUMULATIVE ===")
        for row in block["cumulative"][:10]:
            print(
                f"- {row['function']}  cum={row['cumulative_seconds']:.4f}s  "
                f"self={row['self_seconds']:.4f}s  calls={row['total_calls']}"
            )
        print(f"\n=== {title} TOP SELF TIME ===")
        for row in block["self_time"][:10]:
            print(
                f"- {row['function']}  self={row['self_seconds']:.4f}s  "
                f"cum={row['cumulative_seconds']:.4f}s  calls={row['total_calls']}"
            )

    print_profile_block("PYTHON ENV", python_profile)
    print_profile_block("C WRAPPER", wrapper_profile)

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
        "python_env": _run_benchmark(
            "python_env",
            lambda: _make_python_stepper(args.seed, args.action_mode),
            args.steps,
            args.warmup,
        ),
        "c_wrapper": _run_benchmark(
            "c_wrapper",
            lambda: _make_c_wrapper_stepper(args.seed, args.action_mode, external_buf=False),
            args.steps,
            args.warmup,
        ),
        "c_wrapper_shared_buf": _run_benchmark(
            "c_wrapper_shared_buf",
            lambda: _make_c_wrapper_stepper(args.seed, args.action_mode, external_buf=True),
            args.steps,
            args.warmup,
        ),
        "c_wrapper_manual_reset": _run_benchmark(
            "c_wrapper_manual_reset",
            lambda: _make_c_manual_reset_stepper(args.seed),
            args.steps,
            args.warmup,
        ),
        "c_raw_kernel": _run_benchmark(
            "c_raw_kernel",
            lambda: _make_raw_c_stepper(args.seed),
            args.steps,
            args.warmup,
        ),
    }

    python_profile = None
    wrapper_profile = None
    if not args.no_cprofile:
        python_profile = _profile_loop(
            _make_python_stepper(args.seed, args.action_mode),
            args.profile_steps,
            args.profile_top,
        )
        wrapper_profile = _profile_loop(
            _make_c_wrapper_stepper(args.seed, args.action_mode, external_buf=False),
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
        "benchmarks": {name: asdict(result) for name, result in benchmarks.items()},
        "profiles": {
            "python_env": python_profile,
            "c_wrapper": wrapper_profile,
        },
        "notes": notes,
    }
    latest_path, archive_path = _write_reports(report, args.report_dir)
    _print_summary(
        benchmarks,
        python_profile,
        wrapper_profile,
        notes,
        latest_path,
        archive_path,
    )


if __name__ == "__main__":
    main()
