"""`build_env_for` must work in a WORKER PROCESS, not just in-process.

WHY THIS FILE EXISTS. Spec §2 W3 assumed fork-safety of the factory's returned
callables was "gated by the existing multiprocessing-backend tests". It is not:
`tests/env/c/test_binding.py`'s MP test builds its own inline factory over
`env.c.cs2_env.make_env` and never touches `build_env_factory`, and every other
`build_env_factory` test in the suite is Serial or in-process. So the one path
where the migrated closure crosses a process boundary — and therefore the only
place `build_env_for`'s function-local import of the env constructor executes
anywhere other than the main interpreter — had no coverage at all.

WHAT COULD GO WRONG THERE, concretely. A worker whose env fell back to the field
defaults still resets and steps happily: the failure is a WORKING env built on
the wrong config, not a crash. The closure has to cross the fork boundary
carrying its EnvConfig, and the function-local import has to resolve on the far
side from whatever `sys.modules` the child happens to have.

TWO TESTS, because fork and cold import are different failures:

  1. `test_train_closure_builds_envs_in_forked_workers` drives the real
     `build_env_factory` closure through pufferlib's Multiprocessing backend. On
     Linux that backend forks, so the child inherits the parent's `sys.modules`
     wholesale — this proves the CLOSURE survives inheritance and that envs
     really are constructed, knobbed and stepped on the far side, but it proves
     nothing about the import, which is a dictionary hit there.
  2. `test_build_env_for_works_from_a_cold_interpreter` closes exactly that gap
     in a fresh subprocess that has imported NOTHING of the training stack: it
     imports `cs2rl.env.factory` alone and calls `build_env_for`, and asserts
     that `cs2rl.train` is still absent afterwards.

     THAT ASSERTION IS INVERTED FROM WHAT IT USED TO BE, and the inversion is
     the point. Before #165 PR B2 the child asserted `train` WAS imported,
     because `build_env_for` reached the env through `train.make_puffer_env`.
     Since B2 it imports `env.c.cs2_env.make_env` directly, so env construction
     has NO L3 dependency at all — a strictly stronger property, and this is
     where it is stated from a cold interpreter rather than inferred.

     ITS LIMIT, stated because a reader will otherwise over-read it: the child
     exercises the `smoke` role only, so it proves THAT path is `train`-free,
     not all six. The other five are covered in-process by
     tests/env/test_env_factory.py's `_construct`, which patches
     `env.c.cs2_env.make_env` and asserts exactly one call through it.

Neither is a substitute for the other, and the pair is deliberately cheap: two
envs, three ticks, a simple 5-room map instead of dust2.
"""
import subprocess
import sys

import numpy as np
import pytest

from tests.conftest import REPO_ROOT

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

    Non-default knobs and a non-default weight on purpose. A worker whose env
    fell back to the FIELD defaults would still reset and step happily — that is
    the silent-default failure this whole workstream exists to catch — so the
    assertion has to be about the env's CONFIG, not about "it ran".

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

    from cs2rl.env.map import make_simple_map
    from cs2rl.train.config import env_config_from_args
    from cs2rl.train.envs import build_env_factory

    args = Namespace(reward_ct_survival=0.0,
                     n_active_per_team=3,
                     pin_pitch=1,
                     crouch_enabled=0,
                     jump_enabled=1,
                     reward_symmetrize=True,
                     gamma=0.995)
    cfg = env_config_from_args(args)
    factory = build_env_factory(shared_ts=Value("f", 0.3), map_data=make_simple_map(), config=cfg)

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

        want = [getattr(cfg, k) for k in _RECORDED_KNOBS]
        for row in worker_rows:
            assert row[1:] == want, (
                f"a worker (pid {row[0]}) built an env with "
                f"{dict(zip(_RECORDED_KNOBS, row[1:], strict=True))}, not the "
                f"{dict(zip(_RECORDED_KNOBS, want, strict=True))} the closure was given — the "
                "factory fell through to the field defaults on the far side of the fork")
    finally:
        vecenv.close()
        # pufferlib's close() terminates the workers but does not wait on them,
        # so they sit as zombies in this process's table for the rest of the
        # pytest session. `active_children()` joins every finished child, which
        # is what actually reaps them — cheap, and it keeps a test that forks
        # from leaving debris behind for whatever runs next.
        multiprocessing.active_children()


def test_build_env_for_works_from_a_cold_interpreter():
    """A fresh process with NOTHING of the training stack imported builds an env.

    The half the fork smoke above cannot prove. On Linux pufferlib's
    Multiprocessing backend forks, so its children inherit the parent's
    `sys.modules` wholesale and the function-local import is a dictionary hit
    whose real behaviour is untested. Here the child imports
    `cs2rl.env.factory` ALONE, and the import inside
    `build_env_for` has to do the whole job from nothing.

    WHAT IT ASSERTS AFTER THE CALL IS THE INVERSE OF WHAT IT USED TO. Since #165
    PR B2 `build_env_for` imports `env.c.cs2_env.make_env` directly, so env
    construction has NO L3 dependency: `train` must still be ABSENT once the env
    is built, and `env.c.cs2_env` must be PRESENT. The old assertion (`train` in
    sys.modules) would now pass only if the dependency came back.

    The two PRE-call assertions are the anti-vacuity controls, one per module.
    Without the `train` one, any future module-scope import in `env.factory`
    would pre-load it and the post-call check would be measuring a dictionary
    miss it never created. Without the `env.c.cs2_env` one, the post-call
    positive control could be satisfied by a module-scope import in
    `env.factory` rather than by `build_env_for` — and that import would break
    the W1 import-lightness invariant besides, which `test_w1_modules.py` guards
    separately.

    SCOPE: the `smoke` role only. See the module docstring.
    """
    code = """
import sys
from cs2rl.env import factory as env_factory
assert "cs2rl.train" not in sys.modules, (
    "env_factory pulled `cs2rl.train` at module scope; this test can no longer see the "
    "function-local import it exists to exercise")
assert "cs2rl.env.c.cs2_env" not in sys.modules, (
    "env_factory pulled the C env at module scope; the post-call assertion below "
    "would then be satisfied by the import rather than by build_env_for, and the "
    "W1 import-lightness invariant is broken besides")
env = env_factory.build_env_for("smoke")
try:
    assert "cs2rl.train" not in sys.modules, (
        "building an env pulled `cs2rl.train`. Since #165 PR B2 env construction has NO L3 "
        "dependency at all — build_env_for imports env.c.cs2_env.make_env directly — and "
        "this assertion is what keeps that true from a cold interpreter")
    assert "cs2rl.env.c.cs2_env" in sys.modules, "build_env_for did not import the C env module"
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
