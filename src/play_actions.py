"""Play-vs-bots helpers. No ctypes, no Raylib, safe to import from tests."""
from pathlib import Path
import numpy as np
from map import SIMPLE_ROOMS
from train import init_policy_state

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
    n = len(SIMPLE_ROOMS)
    out = np.zeros((n, 4), dtype=np.float32)
    for idx, x0, y0, x1, y1, *_ in SIMPLE_ROOMS:
        out[idx] = (x0, y0, x1, y1)
    return out

def find_repo_root(start: Path) -> Path:
    cur = Path(start).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").is_file() and (p / "src" / "play.py").is_file():
            return p
    raise FileNotFoundError(f"no repo root (pyproject.toml + src/play.py) above {start}")

def resolve_policy_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p
