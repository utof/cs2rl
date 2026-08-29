"""Reward-weight wiring — the §6.1 default-equality pin (spec 2026-08-01).

WHY this test exists: train.REWARD_WEIGHT_DEFAULTS duplicates make_env's
defaults by hand. It has to — train.py imports c_env lazily (inside
functions) so that --dump-config stays free of the map/binding/torch import
cost, and a module-level `inspect.signature(make_env)` would break that
guarantee. This test is the price of the duplication: it fails the moment
either side drifts, in EITHER direction (a weight added to make_env, a
default changed, a key dropped from the tuple).

PITFALL: _NON_WEIGHT_PARAMS lists every make_env parameter that is
deliberately NOT threaded, each for cause (spec §4.1). If you add a genuinely
new reward weight to make_env, add it to REWARD_WEIGHT_DEFAULTS — do NOT
"fix" this test by widening the exclusion set.
"""
import inspect
from argparse import Namespace

import pytest

# Params of make_env that are NOT reward weights, each excluded for cause:
#   seed/team_spirit/auto_reset/buf/map_data — not weights at all;
#   pbrs_gamma — must equal training gamma or PBRS loses policy invariance;
#     make_puffer_env already has a dedicated guarded parameter for it
#     (test_pbrs_gamma_matches_training_gamma);
#   include_step_stats_in_info — the issue #100 decision, out of scope;
#   reward_symmetrize — a bool knob, not a weight (Task 3; listed here from
#   the start so this test does not break when Task 3 lands);
#   recoil — physics switch (#120), not a weight. Default off so existing
#     recipes keep today's hitscan; do not thread it through REWARD_WEIGHT_*.
#   n_active_per_team / pin_pitch / crouch_enabled — Rung 0 sim knobs (spec
#     2026-08-29 §2.1, R0-E.2). They change what the SIM DOES (how many agents
#     spawn, whether pitch/crouch actions are honoured), not how it pays out, so
#     REWARD_WEIGHT_DEFAULTS is the wrong channel: an "unflagged run is
#     byte-identical" equality over reward scalars says nothing about them.
#     They get their own trainer-side path (env_knobs), because the trainer must
#     also mask parked rows out of the loss — something no reward weight needs.
_NON_WEIGHT_PARAMS = {
    "seed",
    "team_spirit",
    "auto_reset",
    "buf",
    "map_data",
    "pbrs_gamma",
    "include_step_stats_in_info",
    "reward_symmetrize",
    "recoil",
    "n_active_per_team",
    "pin_pitch",
    "crouch_enabled",
}


def _make_env_signature():
    from c_env.cs2_env import make_env
    return inspect.signature(make_env)


def test_reward_weight_keys_cover_exactly_the_weight_params():
    params = set(_make_env_signature().parameters) - _NON_WEIGHT_PARAMS
    from train import REWARD_WEIGHT_KEYS
    assert set(REWARD_WEIGHT_KEYS) == params, (
        "REWARD_WEIGHT_KEYS drifted from make_env's signature; "
        f"missing={params - set(REWARD_WEIGHT_KEYS)} "
        f"extra={set(REWARD_WEIGHT_KEYS) - params}")
    # NOTE: the two count assertions below are INFORMATIONAL redundancy over the
    # signature set-equality (one edit → three failures is intended as a loud signal,
    # not a prohibition): when legitimately adding a 24th weight, bump them alongside
    # REWARD_WEIGHT_DEFAULTS (review finding 6).
    assert len(REWARD_WEIGHT_KEYS) == 23, (
        f"spec §4.1 pins 23 threaded weights, got {len(REWARD_WEIGHT_KEYS)}")
    # Six of them do NOT start with reward_ — a prefix scan is wrong by
    # construction (spec §4.2). Pin that so nobody "simplifies" the tuple away.
    assert sum(1 for k in REWARD_WEIGHT_KEYS if not k.startswith("reward_")) == 6


def test_reward_weight_defaults_equal_make_env_defaults():
    sig = _make_env_signature()
    from train import REWARD_WEIGHT_DEFAULTS
    for name, default in REWARD_WEIGHT_DEFAULTS.items():
        # Membership first: indexing sig.parameters directly would blow up as a
        # bare KeyError for a renamed/removed kwarg, hiding WHICH key drifted
        # behind a traceback instead of naming it in the failure message.
        assert name in sig.parameters, (
            f"{name} is in REWARD_WEIGHT_DEFAULTS but not in make_env's "
            "signature — it was renamed or removed; fix the dict, not this test")
        assert sig.parameters[name].default == pytest.approx(default), (
            f"{name}: train default {default} != make_env default "
            f"{sig.parameters[name].default} — an unflagged run would no "
            "longer be byte-identical to the pre-wiring env")


def test_reward_weight_keys_are_tuple_of_defaults_dict():
    """Single derivation point (spec §4.2): the tuple IS the dict's keys."""
    from train import REWARD_WEIGHT_DEFAULTS, REWARD_WEIGHT_KEYS
    assert REWARD_WEIGHT_KEYS == tuple(REWARD_WEIGHT_DEFAULTS)


# ── reward_overrides_from_args: the helper Task 2's env factory closes over ──
# Its three load-bearing properties are pinned directly here rather than only
# through the CLI round-trip, because the factory path calls it with args
# objects the CLI never produces (harness/eval/record namespaces).


def test_reward_overrides_from_args_covers_every_key_and_coerces_to_float():
    """All 23 keys, always, and always genuine floats.

    Task 2 splats the result into make_env, so a missing key would silently
    fall back to the C default and an int would change ctypes coercion.
    """
    from train import REWARD_WEIGHT_KEYS, reward_overrides_from_args

    out = reward_overrides_from_args(Namespace(reward_kill=1))         # int on purpose
    assert set(out) == set(REWARD_WEIGHT_KEYS)
    assert out["reward_kill"] == 1.0
    assert type(out["reward_kill"]) is float, "int leaked through un-coerced"


def test_reward_overrides_from_args_falls_back_to_defaults_on_bare_namespace():
    """The getattr-fallback branch: an args object with NONE of the flags.

    This is the path taken by any caller predating these flags; it must
    reproduce the pre-wiring env exactly rather than raising AttributeError.
    """
    from train import REWARD_WEIGHT_DEFAULTS, reward_overrides_from_args

    assert reward_overrides_from_args(Namespace()) == pytest.approx(REWARD_WEIGHT_DEFAULTS)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_reward_overrides_from_args_rejects_non_finite(bad):
    """`--reward-kill nan` must die at startup, naming the key.

    argparse's type=float happily accepts "nan"/"inf"; without this guard the
    first NaN surfaces hours later as a NaN loss with no provenance.
    """
    from train import reward_overrides_from_args

    with pytest.raises(ValueError, match="reward_kill"):
        reward_overrides_from_args(Namespace(reward_kill=bad))


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

    factory = train.build_env_factory(shared_ts=mp.Value("f", 0.3), map_data=None)
    env = factory(seed=0)
    try:
        sd = env._c_env.sd.contents
        for name, default in train.REWARD_WEIGHT_DEFAULTS.items():
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
