#!/usr/bin/env python
"""cs2_demo P1 — load a .pt, nine policies + you. No Raylib import at module level."""
import argparse
import ctypes
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from play_actions import (
    area_bounds_from_simple_rooms,
    find_repo_root,
    play_fill_actions,
    play_mark_done,
    play_reset_round,
    resolve_policy_path,
)

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--spectate", action="store_true")
    p.add_argument("--fog", action="store_true")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _load_play_lib(repo: Path):
    candidates = []
    envp = os.environ.get("CS2_PLAY_LIB")
    if envp:
        candidates.append(Path(envp))
    candidates += [
        repo / "src/c_env/zig-out/lib/libcs2_play.so",
        repo / "src/c_env/zig-out/bin/libcs2_play.so",
    ]
    lib = None
    for c in candidates:
        if c.is_file():
            lib = ctypes.CDLL(str(c))
            break
    if lib is None:
        print(
            "build with: uv run --with 'ziglang>=0.14,<0.15' zig build cs2_demo (from src/c_env)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    lib.play_host_attach.restype = ctypes.c_void_p
    lib.play_host_attach.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,
    ]
    lib.play_host_should_close.restype = ctypes.c_int
    lib.play_host_should_close.argtypes = [ctypes.c_void_p]
    lib.play_host_time.restype = ctypes.c_double
    lib.play_host_time.argtypes = [ctypes.c_void_p]
    lib.play_host_begin_tick.restype = None
    lib.play_host_begin_tick.argtypes = [ctypes.c_void_p]
    lib.play_host_apply_human.restype = None
    lib.play_host_apply_human.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
    lib.play_host_end_tick.restype = None
    lib.play_host_end_tick.argtypes = [ctypes.c_void_p]
    lib.play_host_render.restype = None
    lib.play_host_render.argtypes = [ctypes.c_void_p]
    lib.play_host_on_reset.restype = None
    lib.play_host_on_reset.argtypes = [ctypes.c_void_p]
    lib.play_host_detach.restype = None
    lib.play_host_detach.argtypes = [ctypes.c_void_p]
    return lib


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        ckpt = resolve_policy_path(args.policy)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2

    from train import (
        load_policy_from_checkpoint,
        select_policy_actions_native,
        init_policy_state,
    )
    policy = load_policy_from_checkpoint(str(ckpt), args.device)
    policy_state = init_policy_state(policy, args.device)

    from c_env.cs2_env import make_env
    from map import make_simple_map
    import numpy as np

    repo = find_repo_root(Path(__file__))
    lib = _load_play_lib(repo)
    env = make_env(
        seed=args.seed, auto_reset=False, recoil=True, map_data=make_simple_map())
    obs, _ = env.reset(seed=args.seed)
    # First select sees zero obs (env_reset does not compute_observations). Same as record.

    act_buf = np.zeros((10, 7), dtype=np.int32)
    cont_buf = np.zeros((10, 2), dtype=np.float32)
    bounds = np.ascontiguousarray(area_bounds_from_simple_rooms().reshape(-1))
    resource_dir = str(repo / "src/c_env/zig-out/bin/resources").encode()
    human_idx = -1 if args.spectate else 0
    mode = "sample" if args.sample else "greedy"
    env_ptr = _PyCapsule_GetPointer(env._capsule, None)
    h = None
    try:
        h = lib.play_host_attach(
            env_ptr,
            ctypes.c_int(human_idx),
            ctypes.c_int(1 if args.fog else 0),
            bounds.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(bounds.size // 4),
            resource_dir,
        )
        if not h:
            print("play_host_attach failed", file=sys.stderr)
            return 2
        next_step = lib.play_host_time(h)
        while not lib.play_host_should_close(h):
            now = lib.play_host_time(h)
            if now >= next_step:
                lib.play_host_begin_tick(h)
                pa, pc = select_policy_actions_native(
                    policy, obs, args.device, policy_state, mode)
                play_fill_actions(act_buf, cont_buf, pa, pc)
                lib.play_host_apply_human(
                    h, act_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
                obs, _, terms, truncs, _ = env.step(act_buf, cont_buf)
                lib.play_host_end_tick(h)
                play_mark_done(policy_state, terms, truncs)
                next_step += 1.0 / 16.0
            lib.play_host_render(h)
            if env.terminals[0]:
                policy_state = play_reset_round(
                    env, policy, args.device,
                    on_reset=lambda: lib.play_host_on_reset(h))
    finally:
        lib.play_host_detach(h)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
