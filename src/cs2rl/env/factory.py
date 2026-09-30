"""The one place a Cs2Env is constructed.

Envs are keyed by the ROLE they are built for. The self-play manager is NOT built
here any more: its single builder, `cs2rl.train.selfplay.build_selfplay_manager`
(one builder because its three sites turned out to differ in exactly one argument),
moved up beside `SelfPlayManager` in #205 part 3 (#92). Left here, it had to import
`cs2rl.train`, an upward edge the layers contract needed an ignore entry for.

WHY THIS MODULE EXISTS (#154, spec 2026-08-31 §2 W3). `make_puffer_env` had
seven call sites that drifted independently: the training closure, two
"legacy defaults" eval sites, the eval driver, the smoke test, the public
`make_env` wrapper and the test harness. Nothing tied them together, so a knob
added to one silently skipped the rest — `--smoke --reward-ct-survival 0.0`
running the DEFAULT weights is the documented instance (`cs2rl.train.envs.build_env_factory`'s
docstring), and #143 is the still-open one. Naming the roles turns
"which sites did I forget" into a list this module owns.

THE ROLES ARE NOT INTERCHANGEABLE, and the differences are the point:

  train        the vecenv factory's per-env construction. The whole payload —
               reward weights, symmetrization and the sim knobs — rides
               build_env_factory's closure as ONE EnvConfig; the per-env seed
               (R0-D) wins over the one pufferlib passes.
  eval         the fixed-baseline evaluator's env. `auto_reset=False` is
               load-bearing — eval.baselines raises without it, because it reads
               the terminal tick's C state after step() returns.
  eval_legacy  `load_policy_from_checkpoint` and `evaluate_checkpoint`. Both
               build `config=EnvConfig()`, so every knob they get is the field
               default `src/cs2rl/env/config.py` DECLARES — that module is the one
               declaration, and tests/env/test_env_config.py pins those fields
               against the trained baseline. They deliberately do NOT get the
               training knobs. #143 tracks that; this module is not the fix, it
               just reduces the future fix to one role's config source.
  smoke        `smoke_test`, fixed seed 42.
  harness      `tests._helpers.trainer_harness._build_trainer_for_test`, whose contract is
               "shaped exactly like production" — hence its own role rather
               than a reuse of `train`, since it adds
               `include_step_stats_in_info=True` and builds its config from
               plain function arguments rather than from a CLI args namespace.
  external     the public `make_env(team_spirit, map_data)` wrapper.

There is deliberately NO `record` role, and the reason is NOT that `--record`
reuses one of the roles above — it does not. `cs2rl.train.record.record_episode` builds its
own env with a bare `make_c_env(...)`, the LOWER-layer constructor, which W3
never banned, and THAT is the env every recorded tick comes from: both its
`log_tick` calls take `env.snapshot_state()`, and the name `policy_env` never
appears in it. The `eval_legacy` env `cs2rl.policy.load_policy_from_checkpoint` builds
on the way in is a SECOND, separate env, and it is only ever read — never reset,
never stepped. Read twice, and it is the second read that surprises people: once
for its obs_dim, which the checkpoint's must match, and once inside
`build_policy`, which pulls the policy's `max_turn_speed` buffer off
`driver_env._c_env.sd.contents`. That second read is UNCONDITIONAL — passing
`obs_dim_override`, as this caller does, skips only the obs_dim read — so do not
read the override as "the policy is built without touching the env" (measured
2026-09-04: with the override supplied, `build_policy` reads exactly
`driver_env._c_env.sd.contents.max_turn_speed` and never
`single_observation_space`; hand it an env with no `_c_env` and it dies with
`AttributeError`). It is closed before that function returns, on both of its
exits — explicitly ahead of the obs_dim-mismatch raise, and in a `finally` on the
normal path — so it is not the env anything is recorded from.

CITED BY QUALNAME, NOT BY LINE, and that is the rule for the paragraph above: the
four `train.py:<lineno>` citations it used to carry pointed four lines past their
subjects on `main` and eighty-one past them here, after #165 PR B2 shrank
`make_puffer_env` (measured 2026-09-04). A qualname survives every edit short of a
rename, and a rename that breaks it is the one case where being wrong is loud.

So the recording path is genuinely uncovered by this module, by
the same deliberate scope decision that leaves the other eleven lower-layer sites
uncovered — the census and the reasoning are in
`tests/integration/test_env_construction_enforcement.py`'s LOWER_LAYER_SITES. Adding a role
here would not change that: `record_episode` would still have to be migrated onto
it, and an enum member no call site can reach is a divergence trap — the next
person adds a knob to it and nothing changes.

WHY THE IMPORT IS FUNCTION-LOCAL. Since #165 PR B2 `build_env_for` imports
`env.c.cs2_env.make_env` DIRECTLY, so the module cycle that import used to break is
gone — env construction has no L3 dependency at all and this module never names
`train` (#205 part 3 moved the last such name out with `build_selfplay_manager`).
The import stays function-local anyway because `env.c.cs2_env` is HEAVY (ctypes plus
the compiled binding) and this module is imported at module scope by `cs2rl.policy`,
`cs2rl.train.envs` and `cs2rl.train.evaluate`, which `cs2rl.train.__main__` imports at
ITS module level: a module-scope import here would put the C env on `--dump-config`'s
path and break the import-lightness invariant `tests/train/test_w1_modules.py` enforces,
which is what makes `--dump-config` cost ~1 s instead of ~30 s (measured on the flat
train.py, before #205 part 3). Only `env.config` is imported at module scope, and it
is the stdlib-only leaf.

NO `cs2rl.train` IMPORT, AT ANY SCOPE. This module used to need the flat train.py's `__main__`
self-alias: `build_selfplay_manager` did a function-local `from cs2rl.train import
SelfPlayManager`, and a script run (`python -m cs2rl.train`) held the flat train.py in
`sys.modules` as `__main__`, so that import would have EXECUTED TRAIN.PY A SECOND TIME
under the name `cs2rl.train`. Both halves are gone: the builder moved to
`cs2rl.train.selfplay`, and a package `__main__` is imported by no module, so there is
nothing to alias (`tests/train/test_w1_modules.py::test_script_run_has_exactly_one_cli_module`
pins that no module body runs twice). `env` sits below `train` in pyproject.toml's
`cs2rl layers` contract, which has no `ignore_imports` entry left to hide an import back.

PITFALL — the `make_env` import is INSIDE `build_env_for`, not cached at module scope,
on purpose: `build_env_for` re-reads `env.c.cs2_env.make_env` every time, so a test
that rebinds that attribute still sees its stand-in used, and
tests/env/test_env_factory.py's `_construct` is built on exactly that.

`cs2rl.train.selfplay.build_selfplay_manager` (moved there from this module) covers
the three pre-migration `SelfPlayManager` sites (`train()` plus two in
`_build_trainer_for_test`); they are two calls now, `cs2rl.train.loop.train()` and
the one in tests/_helpers/trainer_harness.py's `_harness_parts`. Its own pre-migration
capture is `tests/fixtures/selfplay_kwargs_pre_w3.json`, recorded one commit before
the builder was written for the same reason the env capture was — a builder
transcribed from the sites it is meant to check asserts nothing.

WHY NEITHER GUARD FLAGS `make_env` IN THIS FILE — and they are two
DIFFERENT guards, which is the part that is easy to get wrong.

`tests/integration/test_env_construction_enforcement.py`'s enforcement scan bans exactly two
symbols in `src/`, `scripts/` and `tests/_helpers/`: `make_puffer_env` and
`SelfPlayManager`. `make_env` is not one of them. That scan no longer has a subject in
this file: the `SelfPlayManager(...)` it used to find here moved with
`build_selfplay_manager` to `cs2rl.train.selfplay` (#205 part 3), which is the home its
CONSTRUCTION_HOMES table now names for that symbol.
What #165 PR B2 changed is the CODE side, not this scan's: the function-local
`from train import make_puffer_env` and `builder(make_puffer_env, **kwargs)`
became `from c_env.cs2_env import make_env` and `builder(make_env, **kwargs)`.
The scan saw no change — neither spelling is a CALL node, so the pre-B2 file
(`46e7ed6:src/env_factory.py:272`, `:274`) had no `make_puffer_env(...)` call
either. `make_puffer_env` does survive in this docstring's prose, so a grep
here still hits it; the scan reads call nodes, not text.

`make_env` belongs instead to `LOWER_LAYER_SITES`, the per-file DISCLOSURE census
that `test_the_unbanned_lower_layer_census_is_accurate` asserts by exact
equality. This file needs no entry there either: `build_env_for` imports
`env.c.cs2_env.make_env` and hands it to the role builder as a VALUE
(`builder(make_env, **kwargs)`), so no CALL node here names it. If the `_make`
parameter is ever inlined into the builders, this file starts showing up in that
census and the entry has to be added.
"""
# Module scope, unlike this module's one function-local import (`make_env` in
# `build_env_for`):
# `env.config` is the stdlib-only leaf of the config graph — it imports nothing
# heavier than `dataclasses` — so this costs nothing on `--dump-config`'s path.
# tests/train/test_w1_modules.py::test_only_sibling_edge_is_to_the_leaf
# lets a split-out module import only the LEAVES (`cs2rl.policy`, `env.config` and
# `env.factory`, this module) — and `env.config` is one of them.
from cs2rl.env.config import EnvConfig

# The role names, in the order the spec lists them. Callers pass one of these
# strings; anything else is a ValueError naming the whole set, because a typo'd
# role that silently fell through to a default would construct the WRONG env and
# nothing downstream would notice.
ROLES = ("train", "eval", "eval_legacy", "smoke", "harness", "external")

# The fixed-baseline evaluator's env seed. A constant rather than a parameter:
# it is what makes eval episodes comparable across runs, so a caller that could
# vary it would be able to make two runs' eval numbers incomparable by accident.
EVAL_SEED = 10_000_003

# smoke_test's seed. Same reasoning, smaller stakes.
SMOKE_SEED = 42


class _Unset:
    """Sentinel distinguishing "no seed argument" from "seed=None".

    Load-bearing for `eval_legacy`, whose two call sites differ ONLY in whether
    they pass `seed`, and `make_env`'s own default is `seed=0`. Spelling the
    absent case as `seed=None` would forward None where the pre-migration bare
    call forwarded nothing and the env therefore saw 0 — a real behaviour change
    that `static_data_scalars()` cannot see, because seed is not a StaticData
    field.
    """

    def __repr__(self):
        return "<unset>"


UNSET = _Unset()


def _build_train(_make, /, *, shared_ts, buf, seed, _seed, map_data, config):
    """The training vecenv's per-env construction.

    ``_seed`` (R0-D, #135) is the per-env seed train() routes through
    ``env_kwargs``; it WINS over ``seed`` because `pufferlib.vector.make` takes
    `seed` as its own named parameter and never forwards it to the backend, so
    every env would otherwise land on pufferlib's default base regardless of
    --seed. ``seed or 0`` for the fallback is intentional: pufferlib passes
    seed=None for some backends.

    ``config`` is REQUIRED and passed straight through. Before #165 PR B2 this
    builder took three separate payload arguments — ``reward_overrides``,
    ``reward_symmetrize`` and ``env_knobs`` — and the LAST of them carried an
    ``or {}`` None-guard, so "the caller passed nothing" and "the caller passed
    the defaults" were different code paths here. One frozen EnvConfig collapses
    that: build_env_factory resolves the default ABOVE its closure, so a forked
    worker can never be handed a None to guard against — pinned by the BARE
    `build_env_factory(...)` call in
    tests/env/test_env_factory.py::test_build_train_env_factory_carries_args_config,
    which reads the resolved config back out of the closure's cell. Five bare
    calls in tests/ REACH that resolution (AST census, 2026-09-04) and only that
    one can SEE it: a None in the cell would arrive here and be forwarded to
    `make_env`, which resolves None itself, so every env the other four build
    comes out identical either way. That is what made deleting the resolution
    invisible to the whole suite until the assertion was added.
    """
    return _make(config=config,
                 team_spirit=shared_ts,
                 buf=buf,
                 seed=_seed if _seed is not None else (seed or 0),
                 map_data=map_data)


def _build_eval(_make, /, *, map_data, config):
    """The fixed-baseline evaluator's env.

    ``auto_reset=False`` is the reason this role cannot be folded into any
    other: eval.baselines reads the terminal tick's C state and episode_stats
    AFTER step() returns, which auto-reset would already have overwritten, and
    it raises outright without it.

    ``config.replace(reward_symmetrize=False)`` is a RULE, not a default. Eval
    reports RAW rewards so its numbers stay comparable across runs and against a
    ``--reward-symmetrize`` training run; the zero-sum transform is a
    training-time device. The line would read the same if the field default were
    the other way round, which is why `tests/integration/test_no_restated_env_defaults.py`
    ALLOWLISTS it instead of counting it as a restatement. Before #165 this fell
    out of `make_puffer_env`'s own parameter default, i.e. eval got raw rewards
    by ACCIDENT of the caller not passing the flag; stating it here is the point
    of the migration.

    ``config`` is required, with no None default: the call site derives it from
    `env_config_from_args`, which always returns an EnvConfig, and the eval env's
    config is cross-checked against the driver env's immediately after
    construction (`cs2rl.train.envs.assert_eval_env_agreement`). Accepting None here would
    let that check compare a default eval env against a knobbed driver.
    """
    return _make(config=config.replace(reward_symmetrize=False),
                 team_spirit=None,
                 seed=EVAL_SEED,
                 map_data=map_data,
                 auto_reset=False)


def _build_eval_legacy(_make, /, *, seed=UNSET):
    """`load_policy_from_checkpoint` (no seed) and `evaluate_checkpoint` (seed=).

    Two call shapes, one role. See `_Unset` for why the absent case is not
    spelled `seed=None`.

    These deliberately carry NO training knobs. `config=EnvConfig()` is what
    says so: a bare EnvConfig IS the declared field defaults, and
    `src/cs2rl/env/config.py` is the single place those are declared. That is today's
    behaviour and #143, not a bug to fix in passing here. Naming the config
    object moved no value, and that is asserted rather than asserted-by-hand:
    the pre-migration capture's two `eval_legacy` rows record an
    `expected_config` equal to `EnvConfig()` field for field, and
    tests/env/test_env_factory.py compares this builder's output against them.

    ``team_spirit=None`` is spelled explicitly, and the two defaults it sits
    between genuinely differ: the old chain reached the env through
    `make_puffer_env`, whose `team_spirit` parameter defaults to None, while
    `make_env`'s own defaults to a float. `Cs2Env.__init__` maps BOTH to the
    same initial team spirit (it special-cases None), so nothing observable
    turns on it — but the per-role oracle compares the kwargs by VALUE, and
    spelling this one keeps that comparison a measurement instead of an argument
    about equivalence.
    """
    if seed is UNSET:
        return _make(config=EnvConfig(), team_spirit=None)
    return _make(config=EnvConfig(), team_spirit=None, seed=seed)


def _build_smoke(_make, /):
    """`smoke_test`'s env: one fixed seed, nothing else.

    ``team_spirit=None`` is spelled explicitly for the same reason as
    `_build_eval_legacy`'s: the pre-#165-B2 chain inherited it from
    `make_puffer_env`'s parameter default, `make_env`'s own default differs, and
    `Cs2Env` maps both to the same initial team spirit — so stating it keeps the
    oracle comparing values rather than arguing equivalence.
    """
    return _make(config=EnvConfig(), team_spirit=None, seed=SMOKE_SEED)


def _build_harness(_make, /, *, shared_ts, buf, seed, map_data, config):
    """`tests._helpers.trainer_harness._build_trainer_for_test`'s per-env construction.

    ``0 if seed is None else seed`` — an explicit None check, NOT ``seed or 0``:
    pufferlib forwards seed=None for the first reset, and a falsy-remap would
    silently turn a legitimate ``seed=0`` into the same thing by accident. The
    two spellings agree on today's values and would diverge the moment anyone
    changed the fallback, which is why the harness spells it this way.

    ``include_step_stats_in_info=True`` always, so a harness-built trainer has a
    uniform attribute/info surface across selfplay and no-selfplay modes.

    ``config`` is REQUIRED and passed straight through. The four knobs it
    carries used to be four separate parameters here; since #165 PR B2 the
    mapping from `_build_trainer_for_test`'s plain arguments into an EnvConfig
    lives at the CALL SITE, and that is where the coverage question moved with
    it — see the closure comment in `tests/_helpers/trainer_harness.py` for which knob
    each test can and cannot see going missing.

    NOTE the mask view is NOT attached here. `env._attach_mask_view(mask_shm,
    _mask_idx)` is post-construction wiring that needs the caller's shm handle
    and per-env index, and it stays at the call site — this module builds envs,
    it does not own the trainer's shared memory.
    """
    return _make(config=config,
                 team_spirit=shared_ts,
                 buf=buf,
                 seed=0 if seed is None else seed,
                 map_data=map_data,
                 include_step_stats_in_info=True)


def _build_external(_make, /, *, team_spirit, map_data):
    """The public `make_env(team_spirit, map_data)` wrapper's env.

    Both parameters are REQUIRED even though the PUBLIC WRAPPER `cs2rl.train.envs.make_env`
    declares its own two as optional. (Qualified deliberately: since #165 PR B2
    this module names two different `make_env`s — the wrapper, and the lower-layer
    `env.c.cs2_env.make_env` that `build_env_for` now imports — and both default
    those parameters, so an unqualified sentence would say nothing.) The
    defaulting belongs to the wrapper, because that is its published signature,
    and repeating it here would mean a caller that forgot to forward `map_data`
    got a dust2 env instead of a TypeError, which is the silent-default failure
    every other builder in this module is spelled to avoid.
    """
    return _make(config=EnvConfig(), team_spirit=team_spirit, map_data=map_data)


_ROLE_BUILDERS = {
    "train": _build_train,
    "eval": _build_eval,
    "eval_legacy": _build_eval_legacy,
    "smoke": _build_smoke,
    "harness": _build_harness,
    "external": _build_external,
}


def build_env_for(role, **kwargs):
    """Construct the Cs2Env for ``role`` — the one place a role's env is built.

    Each role's builder has an EXPLICIT keyword signature rather than a
    ``**kwargs`` passthrough, so a caller that misspells a knob gets a TypeError
    naming it at the call rather than an env quietly built on defaults. That is
    the same discipline as build_env_factory's own stray-kwargs guard, which
    exists because a reward key routed through the wrong channel once vanished
    and made an experiment arm train the baseline.

    Import callers as ``from cs2rl.env.factory import build_env_for``, never
    ``from cs2rl.env import factory as env_factory``: three functions this factory is
    called from bind a LOCAL named ``env_factory`` (the nested closures in
    `build_env_factory` and `_build_trainer_for_test`, and
    ``env_factory = build_train_env_factory(...)`` in `train()`), and inside those
    an attribute access on the module name would resolve to the local instead.
    """
    try:
        builder = _ROLE_BUILDERS[role]
    except KeyError:
        raise ValueError(f"unknown env role {role!r}; expected one of {ROLES}") from None

    # Function-local and re-read per call. Two reasons: `env.c.cs2_env` is HEAVY —
    # ctypes plus the compiled binding — and this module is imported at module scope
    # by cs2rl.policy and by train modules the CLI imports at its module level, so a
    # module-scope import here would put the C env on `--dump-config`'s path and break
    # the import-lightness invariant tests/train/test_w1_modules.py enforces; and
    # re-reading per call keeps a test that rebinds env.c.cs2_env.make_env able to see
    # its stand-in used. (The cycle this import used to break is gone, and so is the
    # last `train` import in this module: see the NO `cs2rl.train` IMPORT paragraph in
    # the module docstring.)
    from cs2rl.env.c.cs2_env import make_env

    return builder(make_env, **kwargs)
