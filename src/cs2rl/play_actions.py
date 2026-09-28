"""Play-vs-bots helpers. No ctypes, no Raylib, safe to import from tests."""
from pathlib import Path

import numpy as np

from cs2rl.map import make_simple_map
from cs2rl.train import init_policy_state


def play_fill_actions(act_buf, cont_buf, pa, pc):
    act_buf[:] = pa
    cont_buf[:] = pc


def play_mark_done(policy_state, terms, truncs):
    policy_state["done"] = policy_state["done"].new_tensor(
        np.logical_or(terms, truncs).astype(np.float32))


def play_reset_round(env, policy, device, on_reset=None):
    env.reset(seed=None)
    if on_reset is not None:
        on_reset()
    return init_policy_state(policy, device)


def area_bounds_from_simple_rooms():
    """Room AABB from MapData — same array make_simple_map publishes to C."""
    return np.ascontiguousarray(make_simple_map().area_bounds)


def find_repo_root(start: Path) -> Path:
    cur = Path(start).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").is_file() and (p / "src" / "cs2rl" / "play.py").is_file():
            return p
    raise FileNotFoundError(f"no repo root (pyproject.toml + src/cs2rl/play.py) above {start}")


def resolve_policy_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p
