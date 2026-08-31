"""`build_env_for` must work in a WORKER PROCESS, not just in-process.

WHY THIS FILE EXISTS. Spec §2 W3 assumed fork-safety of the factory's returned
callables was "gated by the existing multiprocessing-backend tests". It is not:
`tests/test_binding.py`'s MP test builds its own inline factory over
`c_env.cs2_env.make_env` and never touches `build_env_factory`, and every other
`build_env_factory` test in the suite is Serial or in-process. So the one path
where the migrated closure crosses a process boundary — and therefore the one
place `build_env_for`'s function-local `from train import make_puffer_env`
executes anywhere other than the main interpreter — had no coverage at all.

WHAT COULD GO WRONG THERE, concretely. The function-local import is what makes
the module cycle work, and it depends on `sys.modules` containing a `train` entry
that is the ALREADY-EXECUTED module — which in a real run is only true because of
`sys.modules.setdefault("train", sys.modules["__main__"])` at the top of train.py's
`__main__` block. In-process tests never exercise that: pytest imports `train`
normally, so the name is present for a reason production does not rely on. Get it
wrong and every env in every worker is built by a SECOND copy of train.py, or the
import raises inside a forked child where the traceback is easy to lose.

TWO TESTS, because fork and cold import are different failures:

  1. `test_train_closure_builds_envs_in_forked_workers` drives the real
     `build_env_factory` closure through pufferlib's Multiprocessing backend. On
     Linux that backend forks, so the child inherits a `sys.modules` that already
     holds `train` — this proves the CLOSURE survives pickling/inheritance and
     that envs really are constructed and stepped on the far side, but it does
     NOT prove the function-local import can stand up on its own.
  2. `test_build_env_for_works_from_a_cold_interpreter` closes exactly that gap
     in a fresh subprocess that has never imported `train`: it imports only
     `env_factory` and calls `build_env_for`, so the `from train import
     make_puffer_env` has to do the whole job from nothing. That is the
     worker-side-import half the fork smoke cannot reach.

Neither is a substitute for the other, and the pair is deliberately cheap: two
envs, three ticks, a simple 5-room map instead of dust2.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

NUM_ENVS = 2
TICKS = 3

# Knobs recorded per construction, in the fixed slot order the shared Array
# below uses. All int-valued, which is what lets the record cross the process
# boundary in a plain `multiprocessing.Array("i", ...)` with no manager process.
_RECORDED_KNOBS = ("n_active_per_team", "pin_pitch", "crouch_enabled", "jump_enabled")
_RECORD_WIDTH = 1 + len(_RECORDED_KNOBS)               # pid, then one slot per knob


@pytest.mark.timeout(180)
def test_train_closure_builds_envs_in_forked_workers():
    """The migrated train closure constructs, resets and steps across a fork.

    Non-default knobs and reward overrides on purpose. A worker whose env fell
    back to `make_puffer_env`'s defaults would still reset and step happily —
    that is the silent-default failure this whole workstream exists to catch — so
    the assertion has to be about the env's CONFIG, not about "it ran".

    WHY THE KNOBS ARE READ OUT OF SHARED MEMORY AND NOT OFF `vecenv.driver_env`.
    Measured: the parent process calls the factory ONCE itself, for the driver
    env pufferlib keeps locally for space introspection. So `driver_env`'s knobs
    are the PARENT's construction, and asserting on them would pass unchanged
    even if every worker built a defaults env — the exact vacuous shape this
    test was added to remove. Instead the factory is wrapped so each construction
    writes its own pid and its env's knobs into a `multiprocessing.Array`, and
    the assertions are made only over records whose pid is NOT this process's.

    `_seed` per env mirrors production's R0-D routing (`env_kwargs[i]["_seed"]`),
    the one kwarg pufferlib forwards to the backend rather than consuming itself.
    """
    import multiprocessing
    import os
    from argparse import Namespace
    from multiprocessing import Array, Value

    import pufferlib.vector

    from map import make_simple_map
    from train import build_env_factory, env_knobs_from_args, reward_overrides_from_args

    args = Namespace(reward_ct_survival=0.0,
                     n_active_per_team=3,
                     pin_pitch=1,
                     crouch_enabled=0,
                     jump_enabled=1,
                     gamma=0.995)
    knobs = env_knobs_from_args(args)
    factory = build_env_factory(shared_ts=Value("f", 0.3),
                                map_data=make_simple_map(),
                                reward_overrides=reward_overrides_from_args(args),
                                reward_symmetrize=True,
                                env_knobs=knobs)

    # Room for the parent's driver env plus one per worker, with slack.
    built = Value("i", 0)
    records = Array("i", _RECORD_WIDTH * (NUM_ENVS + 4))

    def recording_factory(*fargs, **fkwargs):
        env = factory(*fargs, **fkwargs)
        with built.get_lock():
            slot = built.value
            built.value += 1
        if _RECORD_WIDTH * (slot + 1) <= len(records):
            row = [os.getpid()] + [int(getattr(env, k)) for k in _RECORDED_KNOBS]
            records[_RECORD_WIDTH * slot:_RECORD_WIDTH * (slot + 1)] = row
        return env

    vecenv = pufferlib.vector.make([recording_factory] * NUM_ENVS,
                                   env_args=[[] for _ in range(NUM_ENVS)],
                                   env_kwargs=[{
                                       "_seed": 100 + i
                                   } for i in range(NUM_ENVS)],
                                   num_envs=NUM_ENVS,
                                   backend=pufferlib.vector.Multiprocessing,
                                   num_workers=NUM_ENVS,
                                   batch_size=NUM_ENVS,
                                   zero_copy=True)
    try:
        obs, _ = vecenv.reset(seed=0)
        assert obs.shape[0] > 0 and np.isfinite(obs).all(), "forked workers returned no usable obs"

        space = getattr(vecenv, "single_action_space", None) or vecenv.action_space
        for _ in range(TICKS):
            actions = np.stack([space.sample() for _ in range(obs.shape[0])])
            obs, rewards, _terms, _truncs, _infos = vecenv.step(actions)
            assert np.isfinite(rewards).all(), "non-finite reward from a forked worker"

        n = built.value
        all_rows = [list(records[_RECORD_WIDTH * i:_RECORD_WIDTH * (i + 1)]) for i in range(n)]
        worker_rows = [r for r in all_rows if r[0] != os.getpid()]
        assert worker_rows, (
            f"every one of the {n} constructions happened in this process ({os.getpid()}): "
            f"{all_rows}. The vecenv is not running the closure in workers at all, so this test "
            "would prove nothing about the fork boundary.")

        want = [knobs[k] for k in _RECORDED_KNOBS]
        for row in worker_rows:
            assert row[1:] == want, (
                f"a worker (pid {row[0]}) built an env with "
                f"{dict(zip(_RECORDED_KNOBS, row[1:], strict=True))}, not the "
                f"{dict(zip(_RECORDED_KNOBS, want, strict=True))} the closure was given — the "
                "factory fell through to make_puffer_env's defaults on the far side of the fork")
    finally:
        vecenv.close()
        # pufferlib's close() terminates the workers but does not wait on them,
        # so they sit as zombies in this process's table for the rest of the
        # pytest session. `active_children()` joins every finished child, which
        # is what actually reaps them — cheap, and it keeps a test that forks
        # from leaving debris behind for whatever runs next.
        multiprocessing.active_children()


def test_build_env_for_works_from_a_cold_interpreter():
    """A fresh process that has NEVER imported `train` can still build an env.

    The half the fork smoke above cannot prove. On Linux pufferlib's
    Multiprocessing backend forks, so its children inherit a `sys.modules` in
    which `train` is already present and executed; the function-local
    `from train import make_puffer_env` is then a dictionary hit and its real
    behaviour is untested. Here the child imports `env_factory` ALONE and the
    import has to resolve, execute train.py and hand back a working
    `make_puffer_env` — with `sys.path` carrying only `src/`, as a worker would
    have it.

    The `assert "train" not in sys.modules` before the call is what keeps this
    from silently becoming the same test as the one above: without it, any future
    import added to `env_factory`'s module scope would pre-load `train` and this
    would go back to measuring a dictionary hit. (That import would also break
    the W1 import-lightness invariant, which `test_w1_modules.py` guards
    separately — this assertion is the local statement of what THIS test needs.)
    """
    code = f"""
import sys
sys.path.insert(0, {str(SRC)!r})
import env_factory
assert "train" not in sys.modules, (
    "env_factory pulled `train` at module scope; this test can no longer see the "
    "function-local import it exists to exercise")
env = env_factory.build_env_for("smoke")
try:
    assert "train" in sys.modules, "build_env_for did not import train"
    assert sys.modules["train"].make_puffer_env is not None
    obs, _ = env.reset(seed=env_factory.SMOKE_SEED)
    assert obs.shape[0] == 10, obs.shape
    print("COLD-IMPORT-OK", obs.shape)
finally:
    env.close()
"""
    r = subprocess.run([sys.executable, "-c", code],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True,
                       timeout=300)
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    assert "COLD-IMPORT-OK" in r.stdout, (
        f"the child never reached its own assertion — the check is vacuous.\nSTDOUT:\n{r.stdout}")
