"""R0-D (#135): --seed drives random/numpy/torch + env RNGs.

WHAT: pins (a) the C-side seed mixing in env_init (adjacent seeds used to
alias: `seed ? seed : 1` mapped 0 and 1 onto the same xorshift32 stream),
(b) the Python-side seed derivation handed to pufferlib.vector.make, and
(c) end-to-end reproducibility of a one-epoch training run in the explicit
`training` selection, (d) seed_everything (python/numpy/torch) in-process.

PITFALL: the subprocess tests need the C extension built in-place. Run both
same-seed proofs and their different-seed control with `-m training` or `-m ''`.
"""

import json
import subprocess
import sys
from multiprocessing import Value

import numpy as np
import pytest

from tests.conftest import REPO_ROOT

EXCLUDE_PREFIX = ("uptime", "SPS", "timing/", "performance/", "run_id")


def _mix(seed):
    """Python mirror of env_init's golden-ratio mix (uint32 wrap, never 0)."""
    return (seed + 0x9E3779B9) & 0xFFFFFFFF or 1


def test_c_rng_mixing_distinct_for_adjacent_seeds(simple_map):
    from cs2rl.env.c.cs2_env import make_env
    e0 = make_env(map_data=simple_map, seed=0)
    e1 = make_env(map_data=simple_map, seed=1)
    try:
        r0, r1 = e0._c_env.rng, e1._c_env.rng
        assert r0 != r1 and r0 != 0 and r1 != 0
        assert r0 == _mix(0) and r1 == _mix(1)
    finally:
        e0.close()
        e1.close()


def test_c_rng_mixing_zero_fallback_only_at_the_one_wrapping_seed(simple_map):
    """The `: 1u` fallback fires only when seed + 0x9E3779B9 wraps to 0, i.e.
    seed == 0x61C88647. Every other seed maps to its (nonzero) mixed value."""
    from cs2rl.env.c.cs2_env import make_env
    env = make_env(map_data=simple_map, seed=0x61C88647)
    try:
        assert env._c_env.rng == 1
    finally:
        env.close()


def test_env_seed_ranges_of_adjacent_seeds_disjoint():
    """Env i of --seed s gets env_seed_base(s) + i (train()'s _per_env_kwargs)
    — no pufferlib composition is involved. This only pins the 100_000 spacing:
    the (s, i) -> seed map stays injective for any num_envs below it."""
    from cs2rl.train.envs import env_seed_base
    seen = {}
    for s in (0, 1):
        base = env_seed_base(s)
        for i in range(256):
            key = base + i
            assert key not in seen, (s, i, seen[key])
            seen[key] = (s, i)
    assert env_seed_base(1) - env_seed_base(0) == 100_000


def test_eval_seed_cannot_collide_with_worker_seeds():
    """Task 13's eval env uses seed 10_000_003. Worker env i sits at
    env_seed_base(seed) + i, so 10_000_003 is reachable ONLY as --seed 100,
    i=3. For --seed<=4 with any num_envs < 100_000 there is no collision.
    (The original ruling assumed pufferlib's (base+w)*E+j composition, under
    which --seed 4 / E=25 / j=3 DID collide — that path is not used.)"""
    from cs2rl.train.envs import env_seed_base
    hits = [(s, i) for s in range(5) for i in range(100_000) if env_seed_base(s) + i == 10_000_003]
    assert not hits, hits
    assert env_seed_base(100) + 3 == 10_000_003        # documents the one reachable collision


def test_env_seed_base_rejects_uint32_overflow():
    """py_init masks the C seed with & 0xFFFFFFFF; base + i must fit uint32.
    The first overflowing --seed (42_950, i=0) is refused, the last fitting
    one (42_949) is not, and negatives are refused too."""
    from cs2rl.train.envs import env_seed_base
    assert env_seed_base(42_949) < 2**32
    with pytest.raises(ValueError):
        env_seed_base(42_950)
    with pytest.raises(ValueError):
        env_seed_base(-1)


def test_seed_everything_is_deterministic():
    """Fast always-on pin for the host-side RNG set (python/numpy/torch):
    seed_everything(3) twice yields identical draws, seed_everything(4) differs.
    Before this test, deleting those calls from train() passed the default
    suite (only the 2-subprocess e2e test noticed;
    test_two_runs_same_seed_identical[3] supplies that proof in the training tier)."""
    import random

    import torch

    from cs2rl.train.resume import seed_everything

    def draw():
        return torch.rand(3).tolist(), np.random.random(), random.random()

    seed_everything(3)
    a = draw()
    seed_everything(3)
    b = draw()
    seed_everything(4)
    c = draw()
    assert a == b
    assert a[0] != c[0] and a[1] != c[1] and a[2] != c[2]


def test_train_routes_seed_via_per_env_kwargs():
    """Source-text pin (same style as test_train_uses_build_train_env_factory):
    train() must call seed_everything and hand each env `_seed` through
    _per_env_kwargs — NOT via pufferlib.vector.make(seed=), which is a silent
    no-op (vector.make never forwards its own `seed` parameter)."""
    import inspect

    from cs2rl.train import loop as train_loop
    src = inspect.getsource(train_loop.train)
    assert "seed_everything(args.seed)" in src, \
        "train() no longer seeds python/numpy/torch via seed_everything"
    assert '"_seed": env_seed_base(args.seed) + i' in src, \
        "train() no longer routes --seed to the envs via _per_env_kwargs"
    # pufferlib.vector.make swallows `seed=` (its own named parameter, never
    # forwarded to the backend) — a reintroduced kwarg there is a silent no-op.
    assert "seed=env_seed_base" not in src


def test_serial_vecenv_envs_get_distinct_rng_streams(simple_map):
    """Build a Serial vecenv the way train() does (build_env_factory +
    pufferlib.vector.make with per-env `_seed` kwargs) and check every env's
    C rng is the mixed env_seed_base(3) + i — distinct and nonzero. Would
    catch the seed falling back to pufferlib's default base (env i -> i)."""
    import pufferlib.vector

    from cs2rl.train.envs import build_env_factory, env_seed_base
    n = 4
    factory = build_env_factory(shared_ts=Value("f", 0.3), map_data=simple_map)
    vecenv = pufferlib.vector.make([factory] * n,
                                   env_args=[[] for _ in range(n)],
                                   env_kwargs=[{
                                       "_seed": env_seed_base(3) + i
                                   } for i in range(n)],
                                   num_envs=n,
                                   backend=pufferlib.vector.Serial)
    try:
        rngs = {e._c_env.rng for e in vecenv.envs}
        assert len(rngs) == n and 0 not in rngs, rngs
        assert rngs == {_mix(env_seed_base(3) + i) for i in range(n)}
    finally:
        vecenv.close()


def _run(tmp, seed, timesteps=10240):
    # 16 envs => batch_size 10240 = ONE epoch at the default timesteps. 16 is the
    # CLI floor (minibatch_size pinned at 8192).
    ck = tmp / f"s{seed}"
    r = subprocess.run([
        sys.executable, "-m", "cs2rl.train", "--train", "--device", "cpu", "--vec-backend",
        "serial", "--num_envs", "16", "--no-self-play", "--no-dead-run-abort", "--seed",
        str(seed), "--timesteps",
        str(timesteps), "--checkpoint-dir",
        str(ck), "--save_every_sec", "100000", "--run-id", "rid"
    ],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True,
                       timeout=1500)
    assert r.returncode == 0, r.stderr[-3000:]
    rows = [json.loads(line) for line in (ck / "metrics.jsonl").read_text().splitlines()]
    return {
        k: v
        for k, v in rows[-1].items()
        if not k.startswith(EXCLUDE_PREFIX) and isinstance(v, (int, float))
    }


def test_in_process_env_determinism(simple_map):
    """Fast always-on check: two envs with the same seed produce identical obs/rewards."""
    from cs2rl.env.c.cs2_env import make_env
    outs = []
    for _ in range(2):
        env = make_env(map_data=simple_map, seed=7)
        rng = np.random.default_rng(1)
        env.reset()
        acc = []
        for _ in range(50):
            act = np.stack([rng.integers(0, n, size=10) for n in (9, 2, 2, 3, 2, 2, 2)],
                           1).astype(np.int32)
            cont = rng.uniform(-0.5, 0.5, (10, 2)).astype(np.float32)
            obs, rew, *_ = env.step(act, cont)
            acc.append((obs.copy(), rew.copy()))
        env.close()
        outs.append(acc)
    for (o1, r1), (o2, r2) in zip(*outs, strict=True):
        assert np.array_equal(o1, o2) and np.array_equal(r1, r2)


@pytest.mark.training
@pytest.mark.parametrize(
    "seed",
    [
        3,                                             # e2e determinism proof in the training tier
        pytest.param(0, marks=pytest.mark.slow),
    ])
@pytest.mark.timeout(1800)
def test_two_runs_same_seed_identical(tmp_path_factory, seed):
    """Two 1-epoch CPU trainings with the same --seed write identical
    metrics.jsonl rows. The checkpoint dirs live under this session's basetemp.
    A session run without --basetemp gets a numbered pytest-<N> basetemp of its
    own (pytest's default layout, which tests/conftest.py only relocates,
    gh#219), so such sessions cannot collide on --checkpoint-dir; sessions
    handed the same explicit --basetemp still can."""
    tmp = tmp_path_factory.mktemp(f"seed{seed}_", numbered=True)
    a = _run(tmp / "a", seed)
    b = _run(tmp / "b", seed)
    assert a.keys() == b.keys()
    diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    assert not diff, diff


@pytest.mark.training
@pytest.mark.slow                      # 2 subprocess trainings
@pytest.mark.timeout(1800)
def test_two_runs_different_seed_differ(tmp_path):
    """Inverse check (preflight ruling): the identical-runs test passes trivially
    if --seed is ignored. Seeds 3 and 4 must differ on at least one signal key.
    PITFALL: metrics.jsonl carries NO losses/* keys (those only reach the
    dashboard), so the seed-sensitive rows are game/* (rollout outcomes, env
    RNG) and policy/* (learned aim log-std, torch RNG)."""
    a = _run(tmp_path / "a", 3)
    b = _run(tmp_path / "b", 4)
    signal_keys = [k for k in a if k.startswith(("game/", "policy/"))]
    assert signal_keys, sorted(a)
    assert any(a[k] != b[k] for k in signal_keys), {k: (a[k], b[k]) for k in signal_keys}
