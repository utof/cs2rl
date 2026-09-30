#!/usr/bin/env python
"""cs2_demo P1 — load a .pt, nine policies + you. No Raylib import at module level."""
import argparse
import ctypes
import os
import sys
from pathlib import Path

from cs2rl.env.c import SOURCE_DIR, ZIG_OUT
from cs2rl.viz.play_actions import (
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


def _load_play_lib(zig_out: Path = ZIG_OUT):
    """Load libcs2_play.so from $CS2_PLAY_LIB, then `zig_out`/lib and /bin; exit 2 if none loads.

    `zig_out` defaults to the C package's own zig-out/ (cs2rl.env.c.ZIG_OUT), so the
    lookup follows the package wherever it moves; a test passes a tmp directory. A
    candidate that is absent or fails to load is skipped, so a stale location is not an
    error here: it only ends in the "build with" hint.
    """
    candidates = []
    envp = os.environ.get("CS2_PLAY_LIB")
    if envp:
        candidates.append(Path(envp))
    candidates += [
        zig_out / "lib" / "libcs2_play.so",
        zig_out / "bin" / "libcs2_play.so",
    ]
    lib = None
    for c in candidates:
        if not c.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(c))
            break
        except OSError:
            continue
    if lib is None:
        print(
            f"build with: uvx --from 'ziglang>=0.14,<0.15' python -m ziglang build cs2_demo (from {SOURCE_DIR})",
            file=sys.stderr,
        )
        raise SystemExit(2)
    lib.play_host_attach.restype = ctypes.c_void_p
    lib.play_host_attach.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_char_p,
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

    from cs2rl.policy import (
        init_policy_state,
        load_policy_from_checkpoint,
        select_policy_actions_native,
    )
    policy = load_policy_from_checkpoint(str(ckpt), args.device)
    policy_state = init_policy_state(policy, args.device)

    import numpy as np

    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map

    lib = _load_play_lib()
    md = make_simple_map()
    # `recoil` is the one non-default this viewer wants; everything else is
    # EnvConfig's default.
    #
    # THERE WERE TWO EARLIER STATES, NOT ONE, and this comment used to fuse
    # them: it said "pre-#165 this was `recoil=True` as a bare keyword, which
    # `make_env` translated through `from_legacy_kwargs`". First clause true of
    # pre-#165, second clause not. Measured 2026-09-11:
    #   * PRE-#165, at `21f984a` (the Phase-A gate baseline, tree `139a3a3`):
    #     this line was `recoil=True` as a bare keyword, and `make_env` declared
    #     `recoil` as an EXPLICIT parameter of its own, so the keyword bound
    #     directly and nothing translated it. `from_legacy_kwargs` did not exist
    #     to translate with: `src/env_config.py` was ADDED by `30e36a3`, the
    #     first #165 Phase A commit, and
    #     `git show 30e36a3^:src/c_env/cs2_env.py | grep -c from_legacy_kwargs`
    #     returns 0.
    #   * MID-#165, at `6b3bf29` (this branch's base, after Phase A, B1 and B2):
    #     `make_env` takes `config` plus six named runtime keywords and
    #     `**legacy`, `recoil` is no longer a parameter of its own, and this file
    #     still wrote the bare keyword — so THERE the routing through
    #     `EnvConfig.from_legacy_kwargs` is exactly what happened.
    # PR B3 is what replaced the keyword with the object below: same env, one
    # frame earlier and type-checked.
    env = make_env(config=EnvConfig(recoil=True), seed=args.seed, auto_reset=False, map_data=md)
    obs, _ = env.reset(seed=args.seed)
    # First select sees zero obs (env_reset does not compute_observations). Same as record.

    act_buf = np.zeros((10, 7), dtype=np.int32)
    cont_buf = np.zeros((10, 2), dtype=np.float32)
    # Same room quad MapData already published into sd->area_bounds.
    bounds = np.ascontiguousarray(md.area_bounds.reshape(-1))
    resource_dir = str(ZIG_OUT / "bin" / "resources").encode()
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
                pa, pc = select_policy_actions_native(policy, obs, args.device, policy_state, mode)
                play_fill_actions(act_buf, cont_buf, pa, pc)
                lib.play_host_apply_human(h, act_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
                obs, _, terms, truncs, _ = env.step(act_buf, cont_buf)
                lib.play_host_end_tick(h)
                play_mark_done(policy_state, terms, truncs)
                next_step += 1.0 / 16.0
            lib.play_host_render(h)
            if env.terminals[0]:
                policy_state = play_reset_round(env,
                                                policy,
                                                args.device,
                                                on_reset=lambda: lib.play_host_on_reset(h))
    finally:
        lib.play_host_detach(h)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
