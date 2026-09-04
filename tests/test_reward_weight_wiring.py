"""Reward-weight wiring — the args→override→env-factory path (spec 2026-08-01 §6.1).

Since #165 the 23 weights are declared once, in `env_config.RewardWeights`, so
the hand-maintained duplication this file was originally built to pin no longer
exists. Gone with it: the three signature-introspection tests that pinned the
defaults against `make_env` (`make_env` no longer declares the weights at all —
it takes an `EnvConfig`), and the three `reward_overrides_from_args` tests,
whose dataclass-reading replacements live in `tests/test_env_knobs.py`.
`tests/test_env_config.py` owns the declaration itself.

What remains here is everything downstream of the declaration: that the env
factory actually injects the overrides it is handed, and that the last boundary
before the C env rejects a bad key or value. Those cover real wiring, not a
duplication. PR B2 of #165 retires this file when the legacy override dict goes
away.

PITFALL: do NOT re-add a `make_env` signature check. `make_env`'s weight names
now live behind `**legacy`, so `inspect.signature` sees none of them and any
such test would pass vacuously.
"""
from argparse import Namespace

import pytest


def test_env_factory_injects_reward_overrides():
    """Spec §6.3: overrides must arrive through the FACTORY path.

    Deliberately does not call make_puffer_env directly — the discard trap
    (env_factory swallows **kwargs) lives in the factory, so a direct
    make_puffer_env test would pass while training ran the baseline.
    Read back through the ctypes overlay: sd is a POINTER, .contents required.
    """
    import multiprocessing as mp

    import train

    # PARTIAL dict on purpose (review fix 5): a full REWARD_WEIGHT_DEFAULTS
    # copy would set reward_kill to its default explicitly, so the "untouched"
    # assertion below would pass even with the omitted-key fallback broken.
    overrides = {
        "reward_ct_survival": 0.0,                                     # A1 arm
        "reward_win_ct_timeout": 3.0,                                  # A1b arm
        "pbrs_nav_weight_t": 0.07,                                     # non-`reward_`-prefixed
    }
    factory = train.build_env_factory(shared_ts=mp.Value("f", 0.3),
                                      map_data=None,
                                      reward_overrides=overrides)
    env = factory(seed=0)
    try:
        sd = env._c_env.sd.contents
        assert sd.reward_ct_survival == pytest.approx(0.0)
        assert sd.reward_win_ct_timeout == pytest.approx(3.0)
        assert sd.pbrs_nav_weight_t == pytest.approx(0.07)
        assert sd.reward_kill == pytest.approx(0.3), "untouched weight must keep its default"
    finally:
        env.close()


def test_env_factory_without_overrides_keeps_defaults():
    """reward_overrides=None must reproduce today's env exactly."""
    import multiprocessing as mp

    import train
    from env_config import RewardWeights

    factory = train.build_env_factory(shared_ts=mp.Value("f", 0.3), map_data=None)
    env = factory(seed=0)
    try:
        sd = env._c_env.sd.contents
        for name, default in RewardWeights().as_dict().items():
            assert getattr(sd, name) == pytest.approx(default), name
    finally:
        env.close()


def test_unknown_reward_override_key_is_rejected():
    """Fail loud, not with a bare make_env TypeError deep in a forked worker."""
    import train
    with pytest.raises(ValueError, match="reward_ct_surival"):
        train.make_puffer_env(reward_overrides={"reward_ct_surival": 0.0})


def test_env_factory_rejects_unexpected_kwargs():
    """Review fix 1: the catch-all **kwargs must be fatal, not silent.

    pufferlib only passes buf/seed/env_kwargs[i], all named parameters, so a
    stray kwarg means someone routed reward keys through _per_env_kwargs —
    the discard trap. Crash instead of training the baseline.
    """
    import multiprocessing as mp

    import train

    factory = train.build_env_factory(shared_ts=mp.Value("f", 0.3), map_data=None)
    with pytest.raises(TypeError, match="reward_ct_survival"):
        factory(seed=0, reward_ct_survival=0.0)


def test_build_train_env_factory_carries_args_overrides():
    """Review fix 2: the train() → factory seam, without launching a run.

    Reads the returned closure's cells: if the wiring ever regresses to
    reward_overrides=None (or to the defaults regardless of args), the
    non-default weight below stops arriving and this fails.
    """
    import multiprocessing as mp

    import train

    args = Namespace(reward_ct_survival=0.0)
    factory = train.build_train_env_factory(args, shared_ts=mp.Value("f", 0.3), map_data=None)
    cells = dict(
        zip(factory.__code__.co_freevars, (c.cell_contents for c in factory.__closure__),
            strict=True))
    assert cells["reward_overrides"] == pytest.approx(train.reward_overrides_from_args(args))
    assert cells["reward_overrides"]["reward_ct_survival"] == 0.0


def test_train_uses_build_train_env_factory():
    """Pin train()'s call site itself — the one line no test can execute.

    PITFALL: this is a source-text assertion, deliberately. Everything else in
    train() needs a real run to reach, and the failure this guards (dropping
    the overrides) is invisible at runtime: the arm just trains the baseline.
    If you legitimately rename the helper, update this string.
    """
    import inspect

    import train

    src = inspect.getsource(train.train)
    assert "build_train_env_factory(" in src, "train() no longer builds envs through the seam"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.3", None])
def test_make_puffer_env_validates_override_values(bad):
    """Review fix 3: validate at the last boundary before the C env too.

    Direct callers (train_test_harness, future sweep scripts) bypass
    reward_overrides_from_args, so its finiteness check alone is not enough.
    """
    import train

    with pytest.raises(ValueError, match="reward_kill"):
        train.make_puffer_env(reward_overrides={"reward_kill": bad})


def test_unknown_reward_override_error_suggests_the_flag():
    """Review fix 4: a typo'd key should name the flag the user meant."""
    import train

    with pytest.raises(ValueError, match=r"--reward-ct-survival"):
        train.make_puffer_env(reward_overrides={"reward_ct_surival": 0.0})
