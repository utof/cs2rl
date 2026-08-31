"""The one place a Cs2Env or a SelfPlayManager is constructed.

Envs are keyed by the ROLE they are built for; the self-play manager has a single
builder (`build_selfplay_manager`) because its three sites turned out to differ in
exactly one argument.

WHY THIS MODULE EXISTS (#154, spec 2026-08-31 §2 W3). `make_puffer_env` had
seven call sites that drifted independently: the training closure, two
"legacy defaults" eval sites, the eval driver, the smoke test, the public
`make_env` wrapper and the test harness. Nothing tied them together, so a knob
added to one silently skipped the rest — `--smoke --reward-ct-survival 0.0`
running the DEFAULT weights is the documented instance (build_env_factory's
docstring in train.py), and #143 is the still-open one. Naming the roles turns
"which sites did I forget" into a list this module owns.

THE ROLES ARE NOT INTERCHANGEABLE, and the differences are the point:

  train        the vecenv factory's per-env construction. Reward overrides,
               reward symmetrization and the Rung-0 env knobs all ride closure
               state from build_env_factory; the per-env seed (R0-D) wins over
               the one pufferlib passes.
  eval         the fixed-baseline evaluator's env. `auto_reset=False` is
               load-bearing — eval_baselines raises without it, because it reads
               the terminal tick's C state after step() returns.
  eval_legacy  `load_policy_from_checkpoint` and `evaluate_checkpoint`. These
               keep today's DOCUMENTED bare-call defaults ("the defaults
               reproduce the pre-Rung-0 env exactly"): they deliberately do NOT
               get the training knobs. #143 tracks that; this module is not the
               fix, it just reduces the future fix to one role's knob source.
  smoke        `smoke_test`, fixed seed 42.
  harness      `train_test_harness._build_trainer_for_test`, whose contract is
               "shaped exactly like production" — hence its own role rather
               than a reuse of `train`, since it adds
               `include_step_stats_in_info=True` and takes its knobs as plain
               arguments rather than from a CLI-derived dict.
  external     the public `make_env(team_spirit, map_data)` wrapper.

There is deliberately NO `record` role: the `--record` path reaches its env
through `load_policy_from_checkpoint`, i.e. `eval_legacy`. An enum member no
call site can reach is a divergence trap — the next person adds a knob to it and
nothing changes.

WHY THE IMPORTS ARE FUNCTION-LOCAL. `make_puffer_env` stays DEFINED in train.py
(moving it drags a large dependency web and breaks ~42 test uses), and train.py
imports this module at its own module level. The function-local
`from train import make_puffer_env` inside `build_env_for` is the standard
module-cycle breaker, and it keeps this module's scope free of torch / nav /
c_env — the import-lightness invariant `tests/test_w1_modules.py` enforces,
which is what makes `train.py --dump-config` cost ~1 s instead of ~30 s.

PITFALL — WHY THAT IMPORT NEEDS train.py's `__main__` SELF-ALIAS. Every real run
executes train.py as a SCRIPT, so the module sits in `sys.modules` as
`__main__`, not `train`. Without the
`sys.modules.setdefault("train", sys.modules["__main__"])` at the top of
train.py's `if __name__ == "__main__":` block, the import below would EXECUTE
TRAIN.PY A SECOND TIME under the name `train`, leaving two live copies of it per
process — two sets of module constants, cross-copy `isinstance` silently False,
and any `train.<attr>` monkeypatch unreachable from script runs. That alias is a
hard prerequisite of this module, pinned by
`test_train_aliases_itself_into_sys_modules_first` and its behavioural twin.

PITFALL — the import is INSIDE the call, not cached at module scope, on purpose:
it re-reads `train.make_puffer_env` every time, so a test that rebinds that
attribute still sees its stand-in used.

`build_selfplay_manager` covers the three `SelfPlayManager` sites (train.py's
`train()` plus two in `_build_trainer_for_test`). Its own pre-migration capture is
`tests/fixtures/selfplay_kwargs_pre_w3.json`, recorded one commit before the
builder was written for the same reason the env capture was — a builder
transcribed from the sites it is meant to check asserts nothing.

WHY THE ENFORCEMENT SCAN CANNOT SEE THE `make_puffer_env` HALF OF THIS MODULE.
`tests/test_env_construction_enforcement.py` flags `make_puffer_env(...)` and
`SelfPlayManager(...)` calls in `src/` and `scripts/`, exempting this file. It
finds the `SelfPlayManager(...)` below, but it finds NO `make_puffer_env(...)`
here: `build_env_for` imports that function and passes it to the role builder as
a VALUE (`builder(_make, **kwargs)`), so no call node in this file names it. That
is why the enforcement test takes its `make_puffer_env` positive control from
`tests/` (where ~40 direct constructions legitimately live) and from a planted
construction in a scratch tree, rather than from here. If the `_make` parameter is
ever inlined into the builders, this file will start showing up in that scan and
the exemption already covers it.
"""

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
    they pass `seed`, and `make_puffer_env`'s own default is `seed=0`. Spelling
    the absent case as `seed=None` would forward None where the pre-migration
    bare call forwarded nothing and the env therefore saw 0 — a real behaviour
    change that `static_data_scalars()` cannot see, because seed is not a
    StaticData field.
    """

    def __repr__(self):
        return "<unset>"


UNSET = _Unset()


def _build_train(_make, /, *, shared_ts, buf, seed, _seed, map_data, reward_overrides,
                 reward_symmetrize, env_knobs):
    """The training vecenv's per-env construction.

    ``_seed`` (R0-D, #135) is the per-env seed train() routes through
    ``env_kwargs``; it WINS over ``seed`` because `pufferlib.vector.make` takes
    `seed` as its own named parameter and never forwards it to the backend, so
    every env would otherwise land on pufferlib's default base regardless of
    --seed. ``seed or 0`` for the fallback is intentional: pufferlib passes
    seed=None for some backends.

    ``env_knobs or {}`` rather than ``**env_knobs``: None means "make_puffer_env's
    own defaults", i.e. the pre-Rung-0 env, and must splat nothing.
    """
    return _make(team_spirit=shared_ts,
                 buf=buf,
                 seed=_seed if _seed is not None else (seed or 0),
                 map_data=map_data,
                 reward_overrides=reward_overrides,
                 reward_symmetrize=reward_symmetrize,
                 **(env_knobs or {}))


def _build_eval(_make, /, *, map_data, reward_overrides, env_knobs):
    """The fixed-baseline evaluator's env.

    ``auto_reset=False`` is the reason this role cannot be folded into any
    other: eval_baselines reads the terminal tick's C state and episode_stats
    AFTER step() returns, which auto-reset would already have overwritten, and
    it raises outright without it.

    ``env_knobs`` is required and splatted directly (no ``or {}``): the call site
    derives it from `env_knobs_from_args`, which always returns a dict, and the
    eval env's knobs are cross-checked against the driver env's immediately
    after construction. Accepting None here would let that check compare an
    unknobbed eval env against a knobbed driver.
    """
    return _make(team_spirit=None,
                 seed=EVAL_SEED,
                 map_data=map_data,
                 auto_reset=False,
                 reward_overrides=reward_overrides,
                 **env_knobs)


def _build_eval_legacy(_make, /, *, seed=UNSET):
    """`load_policy_from_checkpoint` (no seed) and `evaluate_checkpoint` (seed=).

    Two call shapes, one role. See `_Unset` for why the absent case is not
    spelled `seed=None`.

    These deliberately carry NO training knobs — that is today's documented
    behaviour ("the defaults reproduce the pre-Rung-0 env exactly") and #143,
    not a bug to fix in passing here.
    """
    if seed is UNSET:
        return _make()
    return _make(seed=seed)


def _build_smoke(_make, /):
    """`smoke_test`'s env: one fixed seed, nothing else."""
    return _make(seed=SMOKE_SEED)


def _build_harness(_make, /, *, shared_ts, buf, seed, map_data, n_active_per_team, pin_pitch,
                   crouch_enabled, jump_enabled):
    """`train_test_harness._build_trainer_for_test`'s per-env construction.

    ``0 if seed is None else seed`` — an explicit None check, NOT ``seed or 0``:
    pufferlib forwards seed=None for the first reset, and a falsy-remap would
    silently turn a legitimate ``seed=0`` into the same thing by accident. The
    two spellings agree on today's values and would diverge the moment anyone
    changed the fallback, which is why the harness spells it this way.

    ``include_step_stats_in_info=True`` always, so a harness-built trainer has a
    uniform attribute/info surface across selfplay and no-selfplay modes.

    ``crouch_enabled`` / ``jump_enabled`` are forwarded explicitly even though no
    test passes either to `_build_trainer_for_test` today and their defaults
    equal `make_puffer_env`'s. Dropping them here would therefore be invisible to
    the entire suite; the captured-kwargs oracle is the only thing that sees it.

    NOTE the mask view is NOT attached here. `env._attach_mask_view(mask_shm,
    _mask_idx)` is post-construction wiring that needs the caller's shm handle
    and per-env index, and it stays at the call site — this module builds envs,
    it does not own the trainer's shared memory.
    """
    return _make(team_spirit=shared_ts,
                 buf=buf,
                 seed=0 if seed is None else seed,
                 map_data=map_data,
                 include_step_stats_in_info=True,
                 n_active_per_team=n_active_per_team,
                 pin_pitch=pin_pitch,
                 crouch_enabled=crouch_enabled,
                 jump_enabled=jump_enabled)


def _build_external(_make, /, *, team_spirit, map_data):
    """The public `make_env(team_spirit, map_data)` wrapper's env.

    Both parameters are REQUIRED even though `make_env`'s own are optional. The
    defaulting belongs to the wrapper — it is `make_env`'s published signature —
    and repeating it here would mean a caller that forgot to forward `map_data`
    got a dust2 env instead of a TypeError, which is the silent-default failure
    every other builder in this module is spelled to avoid.
    """
    return _make(team_spirit=team_spirit, map_data=map_data)


_ROLE_BUILDERS = {
    "train": _build_train,
    "eval": _build_eval,
    "eval_legacy": _build_eval_legacy,
    "smoke": _build_smoke,
    "harness": _build_harness,
    "external": _build_external,
}


def build_env_for(role, **kwargs):
    """Construct the Cs2Env for ``role``. The only `make_puffer_env` call site.

    Each role's builder has an EXPLICIT keyword signature rather than a
    ``**kwargs`` passthrough, so a caller that misspells a knob gets a TypeError
    naming it at the call rather than an env quietly built on defaults. That is
    the same discipline as build_env_factory's own stray-kwargs guard, which
    exists because a reward key routed through the wrong channel once vanished
    and made an experiment arm train the baseline.

    Import callers as ``from env_factory import build_env_for``, never
    ``import env_factory``: three functions this factory is called from bind a
    LOCAL named ``env_factory`` (the nested closures in `build_env_factory` and
    `_build_trainer_for_test`, and ``env_factory = build_train_env_factory(...)``
    in `train()`), and inside those an attribute access on the module name would
    resolve to the local instead.
    """
    try:
        builder = _ROLE_BUILDERS[role]
    except KeyError:
        raise ValueError(f"unknown env role {role!r}; expected one of {ROLES}") from None

    # Function-local and re-read per call — see the module docstring for the
    # cycle, the import-lightness invariant and the `__main__` alias it needs.
    from train import make_puffer_env

    return builder(make_puffer_env, **kwargs)


def build_selfplay_manager(*, self_play_enabled, aim_log_std_max, pin_pitch, opponent_mode):
    """Construct the run's `SelfPlayManager`. The only `SelfPlayManager(...)` call site.

    ONE BUILDER, NOT A ROLE ENUM, because the three pre-migration sites —
    `train()` and both branches of `_build_trainer_for_test` — differed in
    `p_past` and nothing else, and even that was the same rule twice:
    ``0.3 if <self-play on> else 0.0``. train() wrote it as a conditional
    expression, the harness spelled the two values out as separate constructions
    under ``if not with_selfplay:``. Naming a role per site would have frozen a
    distinction that does not exist and invited the next knob to be added to one
    "role" only — the exact drift this module exists to stop.
    `tests/fixtures/selfplay_kwargs_pre_w3.json` records all three shapes as they
    stood before this function, so the collapse is asserted rather than assumed.

    ``self_play_enabled`` is the flag, not ``p_past`` itself: p_past is derived
    policy, and a caller that could pass it directly could set 0.15 at one site
    and 0.3 at another without anything noticing.

    ``bool(pin_pitch)`` here rather than at the callers, which is where both
    pre-migration sites did it (`bool(args.pin_pitch)` / `bool(pin_pitch)`).
    `SelfPlayManager.__init__` also coerces, so this is belt-and-braces on the
    class's side — but it keeps the KWARG the callers produce identical to the
    captured one, which is what the fixture compares.

    ``aim_log_std_max`` is required rather than defaulted to None: train() reads
    it as ``getattr(args, "aim_log_std_max", None)`` and the harness takes it as a
    parameter, so both callers always have a value to pass, and a default here
    would let a future caller silently un-pin the aim head's log-std cap on past
    policies (R0-E, #131) — a divergence no gate on this branch can see.

    The four pool constants are literals with the call sites' own comments
    attached; they were identical across all three sites and are now stated once.
    """
    # Function-local for the same cycle/lightness/`__main__`-alias reasons as
    # `build_env_for`'s import — see the module docstring.
    from train import SelfPlayManager

    return SelfPlayManager(
        pool_size=15,
        p_past=0.3 if self_play_enabled else 0.0,
        save_every_epochs=25,                          # ~2M steps per save at batch_size=81920
        win_threshold=0.6,
        phase_length=50,                               # switch opponent team every ~4M steps
        aim_log_std_max=aim_log_std_max,
        pin_pitch=bool(pin_pitch),
                                                       # Rung 1a T3: "noop" ⇒ the patched evaluate() drives opponent_team as a
                                                       # statue. train() guards that it cannot combine with self-play, so
                                                       # opponent_team stays INITIAL_OPPONENT_TEAM — the same team
                                                       # build_participating_rows masked out. The harness mirrors that guard
                                                       # (assert_opponent_self_play_compatible) and passes the mode anyway, so
                                                       # the self-play and no-self-play paths cannot drift if the pairing is
                                                       # ever allowed.
        opponent_mode=opponent_mode,
    )
