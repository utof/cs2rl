# tests/test_play_policy.py
from pathlib import Path

import numpy as np
import pytest

from cs2rl.c_env import SOURCE_DIR, ZIG_OUT
from cs2rl.play_actions import (
    area_bounds_from_simple_rooms,
    play_fill_actions,
    play_mark_done,
    play_reset_round,
    resolve_policy_path,
)


def test_fill_then_human_same_buffers_to_step():
    act_buf = np.zeros((10, 7), np.int32)
    cont_buf = np.zeros((10, 2), np.float32)
    pa = np.zeros((10, 7), np.int32)
    pa[:, 0] = 1
    pc = np.stack([np.arange(10, dtype=np.float32), -np.arange(10, dtype=np.float32)], axis=1)
    seen = {}

    def stub_step(actions, cont):
        seen["act_id"] = id(actions)
        seen["cont_id"] = id(cont)

    play_fill_actions(act_buf, cont_buf, pa, pc)
    act_buf[0] = 7
    stub_step(act_buf, cont_buf)
    assert seen["act_id"] == id(act_buf) and seen["cont_id"] == id(cont_buf)
    assert act_buf[0].tolist() == [7] * 7
    assert np.all(act_buf[1:, 0] == 1)
    assert np.array_equal(cont_buf[1:], pc[1:])


def test_spectate_keeps_all_policy_rows():
    act_buf = np.zeros((10, 7), np.int32)
    cont_buf = np.zeros((10, 2), np.float32)
    pa = np.ones((10, 7), np.int32)
    pc = np.zeros((10, 2), np.float32)
    play_fill_actions(act_buf, cont_buf, pa, pc)
    # human_idx < 0: do not write row 0
    assert np.array_equal(act_buf, pa)


def test_missing_checkpoint_raises():
    with pytest.raises(FileNotFoundError):
        resolve_policy_path("/no/such/cs2rl-policy.pt")


def test_area_bounds_match_simple_rooms():
    b = area_bounds_from_simple_rooms()
    assert b.shape == (17, 4) and b.dtype == np.float32
    assert b[0].tolist() == [0.0, 416.0, 256.0, 672.0]


class _TinyPol:
    hidden_size = 4

    def forward_eval(self, x, state):
        state["lstm_h"] = state["lstm_h"] + 1
        return None


def test_mark_done_is_torch_float_tensor():
    import torch

    from cs2rl.train import init_policy_state
    policy = _TinyPol()
    st = init_policy_state(policy, "cpu")
    terms = np.zeros(10, dtype=np.bool_)
    truncs = np.zeros(10, dtype=np.bool_)
    terms[0] = True
    play_mark_done(st, terms, truncs)
    assert hasattr(st["done"], "float")
    assert st["done"].dtype == torch.float32
    assert float(st["done"][0]) == 1.0


def test_reset_round_zeros_hidden():
    import torch

    from cs2rl.train import init_policy_state
    policy = _TinyPol()
    st = init_policy_state(policy, "cpu")
    st["lstm_h"] += 3

    class _Env:

        def reset(self, seed=None):
            return None, None

    st2 = play_reset_round(_Env(), policy, "cpu")
    assert torch.count_nonzero(st2["lstm_h"]) == 0


def test_make_client_takes_resource_dir():
    text = (SOURCE_DIR / "cs2_render.h").read_text()
    assert "make_client" in text and "const char* resource_dir" in text


def test_cs2_demo_policy_missing_exits_nonzero():
    import os
    import subprocess
    demo = ZIG_OUT / "bin" / "cs2_demo"
    if not demo.is_file():
        pytest.skip("cs2_demo not built")
    # DISPLAY unset: dispatcher must fail before InitWindow
    env = {**os.environ, "DISPLAY": ""}
    r = subprocess.run(
        [str(demo), "--policy", "/no/such/cs2rl-policy.pt"],
        cwd=str(demo.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert r.returncode != 0
    # either printed the uv/venv hint, or exec'd python which FileNotFound
    blob = (r.stderr or "") + (r.stdout or "")
    assert "cs2rl.play" in blob or "Checkpoint" in blob or "UV_PROJECT_ENVIRONMENT" in blob


def test_env_scripted_movers_not_statues(make_map):
    from cs2rl.c_env.cs2_env import make_env
    from cs2rl.env.config import EnvConfig
    env = make_env(seed=0, auto_reset=False, map_data=make_map, config=EnvConfig(recoil=True))
    env.reset()
    acts = np.zeros((10, 7), np.int32)
    cont = np.zeros((10, 2), np.float32)
    acts[1:, 0] = 1
    p0 = [(env._c_env.game.agents[i].x, env._c_env.game.agents[i].y) for i in range(10)]
    for _ in range(16):
        env.step(acts, cont)
    moved = []
    for i in range(10):
        dx = env._c_env.game.agents[i].x - p0[i][0]
        dy = env._c_env.game.agents[i].y - p0[i][1]
        moved.append((dx * dx + dy * dy)**0.5)
    assert max(moved[1:]) > 10
    assert moved[0] < 10
    env.close()


def test_play_cli_missing_pt_exits_2():
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "-m", "cs2rl.play", "--policy", "/no/such/cs2rl-policy.pt"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert r.returncode == 2
    assert "Checkpoint" in (r.stderr + r.stdout) or "not found" in (r.stderr + r.stdout).lower()


def test_load_play_lib_unloadable_so_exits_2(tmp_path, monkeypatch, capsys):
    from cs2rl.play import _load_play_lib
    bad = tmp_path / "libcs2_play.so"
    bad.write_bytes(b"not-an-elf")
    monkeypatch.setenv("CS2_PLAY_LIB", str(bad))
    with pytest.raises(SystemExit) as ei:
        _load_play_lib(zig_out=tmp_path)
    assert ei.value.code == 2
    assert "zig build cs2_demo" in capsys.readouterr().err


def test_cs2_demo_relative_venv_is_realpathd(tmp_path):
    import os
    import subprocess
    repo = Path(__file__).resolve().parents[1]
    demo = ZIG_OUT / "bin" / "cs2_demo"
    if not demo.is_file():
        pytest.skip("cs2_demo not built")
    venv = None
    for p in [repo, *repo.parents]:
        cand = p / ".venv"
        if (cand / "bin" / "python").is_file():
            venv = cand
            break
    if venv is None:
        pytest.skip("no ancestor .venv to borrow")
    rel = os.path.relpath(venv, tmp_path)
    assert not rel.startswith("/")
    env = {**os.environ, "DISPLAY": "", "UV_PROJECT_ENVIRONMENT": rel}
    env.pop("CS2RL_VENV", None)
    r = subprocess.run(
        [str(demo), "--policy", "/no/such/cs2rl-policy.pt"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert r.returncode == 2
    blob = (r.stderr or "") + (r.stdout or "")
    assert "Checkpoint" in blob
    assert "set UV_PROJECT_ENVIRONMENT" not in blob
