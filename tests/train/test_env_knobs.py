"""R0-G (Rung 0 spec 2026-08-29): env knobs via CLI.

`--round-time-ticks` / `--laser-range` / `--max-turn-speed` flow
CLI → env_config_from_args → make_env → the packed StaticData buffer
binding.init copies (StaticData.round_time / laser_range + laser_range_sq /
max_turn_speed). None ⇒ env/nav.py constant, so every caller that does not pass a
knob keeps today's values (fingerprints unchanged at default).

PITFALL (Task 14): the CLI-name ↔ field-name map is hand-written, split over
train_config._ARGS_KNOB_FIELDS and train_shared._R0G_KNOBS, so a knob added to
EnvConfig without a route through one of them is unreachable from the CLI and
every run silently keeps its default. test_args_knob_coverage_is_exhaustive
below is the assertion that fails when that happens.
"""
import numpy as np
import pytest

from tests.conftest import REPO_ROOT

N_AGENTS, ACTION_DIM, AIM_DIM = 10, 7, 2


def test_round_time_ticks_sets_episode_length(simple_map):
    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.env.config import EnvConfig
    env = make_env(map_data=simple_map, config=EnvConfig(round_time=160), seed=1, auto_reset=False)
    try:
        assert env.round_time == 160
        env.reset()
        a = np.zeros((N_AGENTS, ACTION_DIM), np.int32)
        c = np.zeros((N_AGENTS, AIM_DIM), np.float32)
        t = 0
        while True:
            _, _, term, trunc, _ = env.step(a, c)
            t += 1
            if term.any() or trunc.any():
                break
            assert t < 400
        assert t == 160
    finally:
        env.close()


def test_laser_and_turn_speed_reach_static_data(simple_map):
    from cs2rl.env.c import binding
    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.env.config import EnvConfig
    env = make_env(map_data=simple_map,
                   config=EnvConfig(laser_range=300.0, max_turn_speed=0.5),
                   seed=1)
    try:
        sc = binding.static_data_scalars(env._capsule)
        assert sc["laser_range"] == pytest.approx(300.0)
        # laser_range_sq is not a kwarg: it must be derived from the SAME value
        # (two independent values would make range checks and damage falloff disagree).
        assert sc["laser_range_sq"] == pytest.approx(90000.0)
        assert sc["max_turn_speed"] == pytest.approx(0.5)
        assert sc["round_time"] == env.round_time
    finally:
        env.close()


def test_default_knobs_match_nav_constants(simple_map):
    """None (the default) must resolve to the env/nav.py constants, not 0 / garbage."""
    from cs2rl.env import nav
    from cs2rl.env.c import binding
    from cs2rl.env.c.cs2_env import make_env
    env = make_env(map_data=simple_map, seed=1)
    try:
        sc = binding.static_data_scalars(env._capsule)
        assert sc["round_time"] == nav.ROUND_TIME == env.round_time
        assert sc["laser_range"] == pytest.approx(nav.LASER_RANGE)
        assert sc["laser_range_sq"] == pytest.approx(nav.LASER_RANGE * nav.LASER_RANGE)
        assert sc["max_turn_speed"] == pytest.approx(nav.MAX_TURN_SPEED_RAD)
    finally:
        env.close()


@pytest.mark.parametrize("bad", [
    dict(round_time=0),
    dict(round_time=-5),
    dict(round_time=2.5),
    dict(laser_range=0.0),
    dict(laser_range=-1.0),
    dict(max_turn_speed=0.0),
    dict(max_turn_speed=-0.1),
])
def test_invalid_knobs_raise_value_error(simple_map, bad):
    """Validation happens in Python BEFORE binding.init: a C-side assert would
    abort a forked Puffer worker with no traceback."""
    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.env.config import EnvConfig
    with pytest.raises(ValueError):
        make_env(map_data=simple_map, seed=1, config=EnvConfig(**bad))


def test_config_json_records_the_effective_knobs():
    """config.json records what the envs ran with, never the CLI default.

    A run that masked crouch/jump has to be distinguishable from one that did
    not, months later, from the run dir alone — so config.json must carry the
    EFFECTIVE value.

    ONLY the provenance half lives here. The args → EnvConfig resolution half is
    test_env_config_from_args_reads_every_channel below, which drives all four
    channels (weights, flag knobs, the R0-G trio, pbrs_gamma) off a similar
    namespace; re-asserting the resolution here in dataclass form would be the
    second copy of one rule that #165 exists to remove.
    """
    import types

    from cs2rl.train.config import build_train_config, compute_batch_dims
    args = types.SimpleNamespace(device="cpu",
                                 seed=1,
                                 timesteps=100_000,
                                 checkpoint_dir="/tmp/x",
                                 round_time_ticks=160,
                                 laser_range=300.0,
                                 max_turn_speed=None,
                                 n_active_per_team=1,
                                 pin_pitch=1,
                                 crouch_enabled=0,
                                 jump_enabled=0,
                                 gamma=0.999,
                                 pbrs_gamma=None)
    _, bptt, bs = compute_batch_dims(16)
    cfg = build_train_config(args, batch_size=bs, bptt_horizon=bptt)
    for key in ("round_time_ticks", "laser_range", "max_turn_speed"):
        assert key in cfg
    assert cfg["round_time_ticks"] == 160 and cfg["laser_range"] == 300.0
    assert cfg["max_turn_speed"] is None
    assert cfg["crouch_enabled"] == 0 and cfg["jump_enabled"] == 0


def test_env_config_from_args_reads_every_channel():
    """args → EnvConfig: weights, flag knobs, the R0-G trio and pbrs_gamma.

    One test over all four channels on purpose: they are read by four different
    mechanisms (REWARD_FIELDS loop, _ARGS_KNOB_FIELDS loop, _R0G_KNOBS pairs,
    resolve_gammas) and a per-channel test would let a whole mechanism go
    missing while its neighbours stayed green.
    """
    import types

    from cs2rl.env.config import EnvConfig
    from cs2rl.train.config import env_config_from_args
    args = types.SimpleNamespace(reward_ct_survival=0.0,
                                 pbrs_nav_weight_t=0.07,
                                 n_active_per_team=1,
                                 pin_pitch=1,
                                 crouch_enabled=0,
                                 jump_enabled=0,
                                 reward_symmetrize=True,
                                 round_time_ticks=160,
                                 laser_range=300.0,
                                 max_turn_speed=None,
                                 gamma=0.999,
                                 pbrs_gamma=None)
    cfg = env_config_from_args(args)
    assert isinstance(cfg, EnvConfig)
    assert cfg.rewards.reward_ct_survival == 0.0 and cfg.rewards.pbrs_nav_weight_t == 0.07
    assert cfg.rewards.reward_kill == 0.3, "an unflagged weight must keep its field default"
    assert (cfg.n_active_per_team, cfg.pin_pitch, cfg.crouch_enabled, cfg.jump_enabled) == (1, 1, 0,
                                                                                            0)
    assert cfg.reward_symmetrize is True
    assert cfg.round_time == 160 and cfg.laser_range == 300.0 and cfg.max_turn_speed is None
    assert cfg.pbrs_gamma == 0.999, "R0-J: pbrs_gamma follows gamma when --pbrs-gamma is absent"


def test_env_config_from_args_pbrs_gamma_follows_gamma_unless_overridden():
    """R0-J in both directions: --gamma carries into pbrs_gamma, --pbrs-gamma wins.

    Kept apart from test_env_config_from_args_reads_every_channel because that
    namespace is checked channel-by-channel against the dataclass defaults, so
    its gamma equals the pbrs_gamma field default and its pbrs_gamma assertion
    holds even for a resolver that never consults gamma at all. Here both probe
    values sit off the field default, so the first case goes red for a resolver
    that drops the gamma fallback and the second goes red for one that ignores
    an explicit --pbrs-gamma. Neither can be satisfied by the field default.

    WHY THIS EARNS ITS OWN TEST: PBRS is only policy-invariant (Ng et al.) when
    the shaping discount equals the PPO discount, so a --gamma that failed to
    reach the env would mis-shape an entire run with nothing in the logs to say
    so. The precedence itself lives in resolve_gammas — this pins that
    env_config_from_args keeps delegating to it rather than inventing a rule.
    """
    import types

    from cs2rl.env.config import EnvConfig
    from cs2rl.train.config import env_config_from_args
    field_default = EnvConfig().pbrs_gamma
    carried, explicit = 0.97, 0.95
    # Both probe values must sit off the field default, or a resolver that never
    # reads args at all would satisfy the assertions below.
    assert carried != field_default and explicit != field_default
    cfg = env_config_from_args(types.SimpleNamespace(gamma=carried, pbrs_gamma=None))
    assert cfg.pbrs_gamma == carried, "R0-J: pbrs_gamma follows gamma when --pbrs-gamma is absent"
    cfg = env_config_from_args(types.SimpleNamespace(gamma=carried, pbrs_gamma=explicit))
    assert cfg.pbrs_gamma == explicit, "an explicit --pbrs-gamma must win over --gamma"


def test_env_config_from_args_on_a_bare_namespace_is_the_default_config():
    """Harness / dump-config namespaces predate every flag: all defaults, by OMISSION.

    This is the "unflagged run is today's env" guarantee at the parse layer. It
    is spelled as equality against `EnvConfig()` rather than field-by-field so a
    knob added to the dataclass without a reading rule fails here.
    """
    import types

    from cs2rl.env.config import EnvConfig
    from cs2rl.train.config import env_config_from_args
    assert env_config_from_args(types.SimpleNamespace()) == EnvConfig()


def test_env_config_from_args_coerces_and_rejects_weights():
    """int in, float out; nan/inf out, naming the key.

    argparse's type=float accepts "nan" and "inf" happily, and a NaN weight
    surfaces hours into a run as a NaN loss with no provenance — so it must die
    at parse time, naming the knob. (Moved here from test_reward_weight_wiring.py
    when env_config_from_args became the single funnel every --reward-* flag is
    read through.)
    """
    import types

    import pytest as _pytest

    from cs2rl.train.config import env_config_from_args
    cfg = env_config_from_args(types.SimpleNamespace(reward_kill=1))   # int on purpose
    assert type(cfg.rewards.reward_kill) is float and cfg.rewards.reward_kill == 1.0
    for bad in (float("nan"), float("inf"), float("-inf")):
        with _pytest.raises(ValueError, match="reward_kill"):
            env_config_from_args(types.SimpleNamespace(reward_kill=bad))


def test_env_config_from_args_takes_exactly_one_positional_parameter():
    """§8.5 gate (f) / spec §6 criterion 2's B1 third: one parameter, no modes.

    R3's whole claim is that there is ONE args → EnvConfig rule. A later
    `def env_config_from_args(args, *, strict=False)` would give the resolver a
    second mode, and every existing caller would keep passing — nothing else in
    the suite looks at this signature, so the drift would be invisible.
    """
    import inspect

    from cs2rl.train.config import env_config_from_args
    params = list(inspect.signature(env_config_from_args).parameters.values())
    assert [p.name for p in params] == ["args"]
    assert params[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_args_knob_coverage_is_exhaustive():
    """Every EnvConfig knob has a decided route from args — or this fails.

    _ARGS_KNOB_FIELDS and _R0G_KNOBS are hand-written (they are the CLI-name ↔
    field-name map), so an eleventh knob added to EnvConfig would otherwise be
    silently unreachable from the CLI and every run would keep its default with
    the whole suite green. `recoil` is listed as deliberately unreachable: there
    is no flag and reading one would be new behaviour.
    """
    from cs2rl.env.config import KNOB_FIELDS
    from cs2rl.train.config import _ARGS_KNOB_FIELDS, _R0G_KNOBS
    routed = set(_ARGS_KNOB_FIELDS) | {f for _, f in _R0G_KNOBS} | {"pbrs_gamma", "recoil"}
    assert routed == set(KNOB_FIELDS)


def test_make_env_forwards_knobs(simple_map):
    from cs2rl.env.c import binding
    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.env.config import EnvConfig
    env = make_env(map_data=simple_map,
                   config=EnvConfig(round_time=160, laser_range=300.0, max_turn_speed=0.5))
    try:
        sc = binding.static_data_scalars(env._capsule)
        assert sc["round_time"] == 160
        assert sc["laser_range"] == pytest.approx(300.0)
        assert sc["max_turn_speed"] == pytest.approx(0.5)
    finally:
        env.close()


@pytest.mark.parametrize("flag", [0, 1])
def test_stance_knobs_reach_static_data_through_env_knobs(simple_map, flag):
    """Rung 1a T2b: --crouch-enabled / --jump-enabled reach StaticData through
    the PRODUCTION path (args → env_config_from_args → make_env), not just via a
    direct make_env kwarg (that layer is covered by tests/test_pitch_pin.py).

    PITFALL: env_config_from_args stores these under EnvConfig FIELD names while
    the train() eval/driver agreement loop compares Cs2Env ATTRIBUTE names — the
    field, the attribute Cs2Env.__init__ copies it onto, and the StaticData key
    binding.init packs must stay spelled alike, so the last two are asserted
    here and the first is what the assertions read through. A rename in only one
    of the three surfaces as a TypeError from EnvConfig(**knobs) if the field
    moved, or — if only the attribute moved — as an AttributeError raised by
    assert_eval_env_agreement at the first scheduled eval, hours into a run.
    """
    import types

    from cs2rl.env.c import binding
    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.train.config import env_config_from_args
    args = types.SimpleNamespace(crouch_enabled=flag, jump_enabled=flag)
    env = make_env(config=env_config_from_args(args), map_data=simple_map)
    try:
        assert env.crouch_enabled == flag and env.jump_enabled == flag
        sc = binding.static_data_scalars(env._capsule)
        assert sc["crouch_enabled"] == flag and sc["jump_enabled"] == flag
    finally:
        env.close()


def test_jump_enabled_is_in_the_eval_driver_agreement_loop():
    """The loop lives in train.assert_eval_env_agreement, which
    tests/test_env_factory.py::test_eval_env_agreement_two_directions calls
    directly — so the BEHAVIOUR (that a knob mismatch raises, from which check,
    with which message) is covered there, not here.

    THIS SCAN IS NO LONGER WHAT GUARDS THE KEY LIST, and reading it as that is
    the mistake. Until #165 PR B2's fix wave the behavioural test differed two
    fields, so a key dropped from the tuple stayed invisible to it and this scan
    was the only cover. That test is now parametrized over every EnvConfig field
    AND writes these five names down, so a drop turns it red on that key's own
    case — measured 2026-09-04 by dropping crouch_enabled: the crouch_enabled
    case failed.

    WHAT IT STILL ADDS, cheaply: the behavioural test walks
    `dataclasses.fields(EnvConfig)`, so it says nothing about a name in the tuple
    that is not a field, and this asserts the loop is FINDABLE at all — the
    precondition every source scan in this file shares. What is at stake is
    unchanged — an eval env built from one EnvConfig while the workers ran
    another would silently score the policy on a DIFFERENT sim than it trains on,
    and the mismatch would never surface in metrics. Mirrors
    test_cli_flags_declared_default_none's source-scan rationale."""
    import re
    src = (REPO_ROOT / "src" / "cs2rl" / "train" / "envs.py").read_text()
    m = re.search(r"for _k in \((.*?)\):", src, re.S)
    assert m, "eval/driver agreement loop not found in train/envs.py"
    keys = m.group(1)
    for knob in ("n_active_per_team", "pin_pitch", "crouch_enabled", "jump_enabled", "round_time"):
        assert f'"{knob}"' in keys, f"{knob} missing from the eval/driver agreement loop"


def test_policy_max_turn_speed_assert(simple_map):
    from cs2rl.train.envs import assert_max_turn_speed_agreement
    from tests._helpers.trainer_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=4, map_data=simple_map)
    try:
        assert_max_turn_speed_agreement(trainer.vecenv, trainer.policy)
        trainer.policy.max_turn_speed.fill_(0.123)
        with pytest.raises(RuntimeError):
            assert_max_turn_speed_agreement(trainer.vecenv, trainer.policy)
    finally:
        cleanup()


def test_cli_flags_declared_default_none():
    """The parser is built inline under ``if __name__ == "__main__"`` (not
    importable), so check the source: each flag is declared, defaults to None
    (⇒ env default) and has the matching dest. The modal arity mirror's flag
    names are checked against train.py by
    `test_live_train_option_mirror_matches_train_py` in
    tests/test_modal_request.py."""
    import re
    src = (REPO_ROOT / "src" / "cs2rl" / "train" / "__main__.py").read_text()
    for flag, dest, typ in (("--round-time-ticks", "round_time_ticks", "int"),
                            ("--laser-range", "laser_range", "float"), ("--max-turn-speed",
                                                                        "max_turn_speed", "float")):
        # yapf may put the flag on its own line after `add_argument(`; allow
        # any whitespace between the paren and the flag literal.
        m = re.search(rf'add_argument\(\s*"{flag}",(.*?)\)\n', src, re.S)
        assert m, flag
        body = m.group(1)
        assert f"type={typ}" in body and "default=None" in body and f'dest="{dest}"' in body, flag


def test_stance_flags_declared_default_on():
    """--crouch-enabled / --jump-enabled are 0/1 knobs that must default to 1.

    They are NOT default=None like the R0-G knobs: there is no "env decides"
    value for a mask bit, and a default of 0 would silently mask the action for
    every run that never asked for the Rung 1a diagnostic. Same source-scan
    reason as above (the parser is not importable).

    The default is read from `EnvConfig()` (bound once as `_ENV_DEFAULTS` above
    the parser) rather than written as `1`, so the flag and the env cannot
    drift; R11's argparse probe is what enforces that direction."""
    import re
    src = (REPO_ROOT / "src" / "cs2rl" / "train" / "__main__.py").read_text()
    for flag, dest in (("--crouch-enabled", "crouch_enabled"), ("--jump-enabled", "jump_enabled")):
        m = re.search(rf'add_argument\(\s*"{flag}",(.*?)\)\n', src, re.S)
        assert m, flag
        body = m.group(1)
        assert "type=int" in body and "choices=(0, 1)" in body, flag
        assert f"default=_ENV_DEFAULTS.{dest}" in body and f'dest="{dest}"' in body, flag
