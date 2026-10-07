"""Training-env wiring: map selection, env seeds, the vec-env factories
and the train/eval env agreement checks, plus the `--smoke` env sanity run.
"""

import dataclasses
import time

import numpy as np

from cs2rl.env.config import EnvConfig
from cs2rl.env.factory import build_harness_env, build_smoke_env, build_train_env
from cs2rl.spec.action import ACTION_HEAD_SIZES
from cs2rl.spec.obs import OBS_DIM
from cs2rl.train.config import env_config_from_args


def auto_vec_workers(num_envs: int, physical_cores: int) -> int:
    """Largest worker count <= min(num_envs, physical_cores) that divides num_envs.

    WHY: pufferlib.vector.make raises APIUsageError unless
    num_envs % num_workers == 0. The old default min(num_envs, cores)
    violated that on any box whose core count doesn't divide num_envs
    (6-core VM, 12-core laptop...), crashing every launch until someone
    hand-picked --vec-num-workers — a recurring failure, fixed 2026-08-13.

    PITFALL: this is only the DEFAULT. An explicit --vec-num-workers is
    passed through unvalidated on purpose — pufferlib's own error is the
    right feedback for a deliberate bad choice.
    """
    cap = max(1, min(num_envs, physical_cores))
    for k in range(cap, 0, -1):
        if num_envs % k == 0:
            return k
    return 1


# First --seed whose base 42_950 * 100_000 = 4_295_000_000 exceeds 2**32 - 1
# (= 4_294_967_295). At seed 42_949 env i fits for i <= 67_295, i.e. any
# realistic num_envs. See env_seed_base.
_MAX_SEED = 42_950


def env_seed_base(seed: int) -> int:
    """R0-D (#135): base seed handed to pufferlib.vector.make from --seed.

    WHAT: --seed * 100_000. Env i (global index, 0..num_envs-1) gets C seed
    base + i via env_kwargs["_seed"] (see build_env_factory), identically
    under Serial and Multiprocessing — pufferlib's own (base + w) * E + j
    composition is NOT used because vector.make drops its `seed` argument.
    env_init then mixes the value (cs2_env.h) so adjacent seeds never alias.
    WHY x100_000: keeps the env-seed ranges of consecutive --seed values
    disjoint for any num_envs < 100_000, so "seed 3" and "seed 4" share no
    env stream — pinned by test_env_seed_ranges_of_adjacent_seeds_disjoint.
    PITFALL: Task 13's eval env is pinned at seed 10_000_003 = base(100) + 3;
    only --seed 100 with >=4 envs collides. For --seed <= 4 no worker env seed
    equals it (test_eval_seed_cannot_collide_with_worker_seeds).
    PITFALL (uint32): py_init masks the C seed with & 0xFFFFFFFF, so base + i
    must stay below 2**32. --seed >= 42_950 (_MAX_SEED) wraps at i=0 and could
    alias another seed's env streams — rejected with ValueError rather than
    silently wrapped (test_env_seed_base_rejects_uint32_overflow).
    """
    seed = int(seed)
    if not 0 <= seed < _MAX_SEED:
        raise ValueError(f"--seed must be in [0, {_MAX_SEED}) so env_seed_base(seed) + i "
                         f"fits uint32 (C seed is masked & 0xFFFFFFFF); got {seed}")
    return seed * 100_000


# ── SECTION: Smoke Test ────────────────────────────────────────────────────


def smoke_test():
    print("[Smoke] Initialising environment...")
    # W3 (#154): the construction seed now lives in env.factory.SMOKE_SEED. The
    # reset() seed below is deliberately NOT routed through the factory — it is
    # this function's own episode seed, not part of the env's construction, and
    # the two happening to be 42 is a coincidence the factory must not encode.
    env = build_smoke_env()
    try:
        obs, _ = env.reset(seed=42)

        assert obs.shape == (10, OBS_DIM), f"Expected obs shape (10, {OBS_DIM}), got {obs.shape}"
        assert np.isfinite(obs).all(), "NaN in initial obs"

        steps = 20_000
        actions = np.zeros((10, len(ACTION_HEAD_SIZES)), dtype=np.int32)
        print(f"[Smoke] Running {steps} steps...")
        t0 = time.perf_counter()
        step_count = 0

        for step_n in range(steps):
            obs, rewards, terms, truncs, infos = env.step(actions)

            assert obs.shape == (10, OBS_DIM), f"Unexpected obs shape at step {step_n}: {obs.shape}"
            assert rewards.shape == (10, ), (
                f"Unexpected reward shape at step {step_n}: {rewards.shape}")
            assert terms.shape == (10, ), f"Unexpected term shape at step {step_n}: {terms.shape}"
            assert truncs.shape == (
                10, ), f"Unexpected trunc shape at step {step_n}: {truncs.shape}"
            assert np.isfinite(obs).all(), f"NaN in obs at step {step_n}"
            assert np.isfinite(rewards).all(), f"NaN in rewards at step {step_n}"

            step_count += 1

        elapsed = time.perf_counter() - t0
        sps = step_count / elapsed

        print(f"[Smoke] Completed {step_count} steps at {sps:.0f} steps/sec")
        print("[Smoke] Throughput gate lives in: uv run pytest tests/env/c/smoke_test.py -q -s")
    finally:
        env.close()


# ── SECTION: PufferLib env factory ─────────────────────────────────────────
# (dead TRAINING_CONFIG dict removed here — zero readers repo-wide, referenced
# a nonexistent sim.py, and its gamma=0.99 contradicted build_train_config;
# finding 21f of docs/2026-07-06-adversarial-review-verification.md)


def build_env_factory(*, shared_ts, map_data, config=None, role="train"):
    """Return the per-env factory callable handed to pufferlib.vector.make.

    WHAT: a closure over the shared team-spirit Value, the preloaded map data
    and ONE frozen EnvConfig; it builds one Cs2Env and attaches the cont-action
    / action-mask shared-memory views.

    WHY the config is CLOSURE state and not per-env kwargs (spec §4.2): the
    returned factory's parameters are all explicitly named, and anything else is
    now a hard error (see below). Reward keys added to the _per_env_kwargs list
    in train() used to be silently dropped, which would have made every
    experiment arm train the default weights. Closure state crosses the fork
    boundary the same way shared_ts and map_data already do (proven). Before
    #165 PR B2 this was THREE parameters — the override dict, the symmetrize
    bool and the knob dict — kept in step by hand for the same reason; one
    frozen object is the same argument made once.

    WHY module-level rather than nested in train(): the §6.3 test has to
    exercise this exact code path, and a closure defined inside train() is
    unreachable without launching a run.

    `config=None` ⇒ `EnvConfig()`, resolved ONCE above the closure so the
    closure captures a config and never a None. Same shape as
    `env.c.cs2_env.make_env`'s own `config=None` default; the one production
    caller (build_train_env_factory) always passes a config.
    test_build_train_env_factory_carries_args_config pins BOTH halves and
    needs two calls to do it: the args-built factory for "the caller passes a
    config", and a BARE `build_env_factory(...)` for the resolution itself, which
    no call that always passes a config can reach. Until that second call was
    added, deleting the resolution below left the pre-fix test green (re-measured
    2026-09-04; the review measured the whole non-slow suite green with it) while
    this paragraph claimed it was pinned.

    PITFALL (review finding 1): within a training run the run's config reaches
    TWO envs, not one — this factory's, and the fixed-baseline eval env behind
    `--eval-interval`, which train() builds as `build_eval_env(...,
    config=env_config_from_args(args))` from the same resolver
    `build_train_env_factory` reads here, with `assert_eval_env_agreement`
    cross-checking the two right after. What the run's config does NOT reach is
    the OTHER entry points: `--smoke` and `--eval` get `EnvConfig()` from their
    role builders (`--eval` reaches `build_legacy_eval_env`; there is no
    --eval-legacy flag), and `--record` builds its env straight off the
    lower-layer `make_env` naming no config at all, which lands on the same
    thing. So `--smoke --reward-ct-survival 0.0` silently runs default weights.
    Symmetrization is the one field even the config-carrying eval env
    deliberately diverges on — `build_eval_env` forces it off so eval reports raw,
    cross-run-comparable rewards. Known limitation, #143's neighbourhood; do not
    fix in this branch.
    ROLE (#92): "train" builds through `build_train_env`; "harness" builds the test
    trainer's envs (`cs2rl.train.compose.build_trainer` with env_role="harness")
    through `build_harness_env`, which takes pufferlib's seed only, so a `_seed` there
    is a TypeError. Both roles attach the same shared-memory views.
    PITFALL: `seed or 0` is intentional — pufferlib passes seed=None for some
    backends. Keep it.
    R0-D (#135) `_seed`: build_trainer routes the train role's per-env seed
    through env_kwargs (`_seed = env_seed_base(--seed) + i`) because
    pufferlib.vector.make takes
    `seed` as ITS OWN named parameter and never forwards it to the backend —
    `make(..., seed=X)` is a silent no-op and every env lands on pufferlib's
    default base (env i -> seed i) regardless of --seed. When `_seed` is given
    it wins over pufferlib's `seed`; the legacy path is unchanged otherwise.
    """
    if role not in ("train", "harness"):
        raise ValueError(f"role={role!r} must be 'train' or 'harness'")
    config = EnvConfig() if config is None else config

    def env_factory(*_args,
                    buf=None,
                    seed=None,
                    _cont_shm=None,
                    _cont_idx=None,
                    _mask_shm=None,
                    _seed=None,
                    **kwargs):
        # STRICT catch-all (review fix 1): pufferlib only ever passes buf,
        # seed and the env_kwargs[i] dict, all of which are named parameters
        # above — so nothing legitimate lands here. Swallowing strays instead
        # would resurrect the discard trap: a reward key routed through
        # _per_env_kwargs would vanish and the arm would train the baseline.
        if kwargs:
            raise TypeError(f"env_factory got unexpected kwargs {sorted(kwargs)}; "
                            "per-env kwargs are discarded — pass via build_env_factory "
                            "closure state")
        # W3 (#154), retyped by #165 PR B2: construction — and ONLY
        # construction — routes through the role factory. The `_seed`/`seed`
        # precedence rule moved with it and now lives in
        # env.factory.build_train_env; the three payload arguments this call used
        # to pass are one EnvConfig, resolved above the closure so a forked
        # worker can never receive None.
        # tests/fixtures/env_config_pre_165b.json recorded this call before it
        # was typed — its three `train` rows ARE the three seed branches — and
        # tests/env/test_env_factory.py drives this closure against each of them
        # (test_train_call_site_forwards_the_captured_kwargs) as well as
        # pinning its spelling against the recorded call source
        # (test_migrated_site_still_reads_what_the_old_site_read).
        if role == "harness":
            if _seed is not None:
                raise TypeError("the harness role takes pufferlib's seed only; _seed is the "
                                "train role's per-env seed")
            env = build_harness_env(shared_ts=shared_ts,
                                    buf=buf,
                                    seed=seed,
                                    map_data=map_data,
                                    config=config)
        else:
            env = build_train_env(shared_ts=shared_ts,
                                  buf=buf,
                                  seed=seed,
                                  _seed=_seed,
                                  map_data=map_data,
                                  config=config)
        # Attach the shared-memory views so the env (whether running in the
        # main process under Serial, or a forked worker under
        # Multiprocessing) can pull cont_actions written by the trainer and
        # publish action masks back to it (F8). _cont_idx may be None when
        # env_factory is called outside the train() codepath (eg. legacy
        # callers); both attaches are no-ops then.
        if _cont_shm is not None and _cont_idx is not None:
            env._attach_cont_action_view(_cont_shm, _cont_idx)
        if _mask_shm is not None and _cont_idx is not None:
            env._attach_mask_view(_mask_shm, _cont_idx)
        return env

    return env_factory


def build_train_env_factory(args, *, shared_ts, map_data, role="train"):
    """A trainer vecenv's env factory: the run's EnvConfig, derived from args.

    WHY this exists as its own function (review fix 2): it is the seam between
    args and the envs. Inlined in train() it was untestable without launching a
    run, so nothing caught a regression that dropped the run's weights — exactly
    the silent-baseline failure this whole change is guarding against.
    test_build_train_env_factory_carries_args_config reads the config back out of
    the closure it returns. Its caller is `cs2rl.train.compose.build_trainer`, for
    the CLI run (role "train") and for test trainers (role "harness").

    Derives ONE config from the SAME resolver build_train_config uses
    (`env_config_from_args`), so config.json provenance and the envs that
    actually ran cannot disagree — about a weight, about symmetrization or about
    a sim knob. Before #165 PR B2 that was three separate derivations here, each
    with its own way to fall out of step; build_trainer also asserts the built
    driver_env agrees with the participation vector it derives from the same
    args.
    """
    return build_env_factory(shared_ts=shared_ts,
                             map_data=map_data,
                             config=env_config_from_args(args),
                             role=role)


# ── SECTION: Dead Run Detector ─────────────────────────────────────────────

MAP_NAMES = ("simple", "dust2", "arena-duel")


def build_map_data(name: str):
    """R0-H: `--map` name → the MapData the envs run on (None ⇒ the cs2 nav map).

    WHAT: "simple" → map.make_simple_map(); "arena-duel" → map.make_arena_duel_
    map(); "dust2" → None, which is exactly what make_env(map_data=None)
    understands (it loads nav via _ENV_CACHE — pin_pitch_for_map resolves None
    the same way, so the two never disagree on which map "None" is).

    WHY a function: the CLI needs the map ABOVE the --dump-config exit (the
    Modal runner fingerprints every launch from that dump and config.json must
    carry the geometry-resolved pin_pitch and the env label), and train()'s
    spawn-count guard needs the same name → the one table lives here.

    PITFALL: ValueError (never assert) on an unknown name; argparse `choices`
    already rejects it on the CLI, this is for programmatic callers. Importing
    `map` costs ~0.8 s (0.76 s measured, Task 12 report); loading dust2 is
    deferred to make_env / pin_pitch_for_map (cached, ~1 s from the nav cache).
    """
    if name not in MAP_NAMES:
        raise ValueError(f"unknown map {name!r}; expected one of {MAP_NAMES}")
    if name == "dust2":
        return None
    if name == "arena-duel":
        from cs2rl.env.map import make_arena_duel_map
        return make_arena_duel_map()
    from cs2rl.env.map import make_simple_map
    return make_simple_map()


def check_spawn_counts(vecenv, map_name: str) -> tuple[int, int]:
    """R0-H startup guard: the C StaticData spawn lists are within capacity and
    match the preset. Returns (n_t_spawns, n_ct_spawns).

    WHAT: reads sd->n_t_spawns / n_ct_spawns off the driver env (same path as
    assert_pin_pitch_agreement). Generic bounds are ASYMMETRIC ON PURPOSE —
    StaticData has t_spawns[15] / ct_spawns[5] (cs2_types.h) — and the arena
    must have exactly 4 + 4 (ARENA_DUEL_V1; fewer rows would silently weaken
    the load-bearing spawn randomisation, ≥ TEAM_SIZE would flip spawn_team to
    the shuffle path and change the RNG draw count).

    PITFALL: RuntimeError, never a bare assert (python -O strips asserts). A
    non-Cs2Env driver is a wiring bug and must also stop the run.
    """
    env = getattr(vecenv, "driver_env", vecenv)
    try:
        sd = env._c_env.sd.contents
        n_t, n_ct = int(sd.n_t_spawns), int(sd.n_ct_spawns)
    except AttributeError as e:
        raise RuntimeError("check_spawn_counts: driver_env is not a Cs2Env") from e
    if not (1 <= n_t <= 15 and 1 <= n_ct <= 5):
        raise RuntimeError(f"spawn counts out of StaticData capacity: n_t_spawns={n_t} (1..15) "
                           f"n_ct_spawns={n_ct} (1..5)")
    if map_name == "arena-duel" and (n_t, n_ct) != (4, 4):
        raise RuntimeError(f"ARENA_DUEL_V1 expects 4 T + 4 CT spawn areas, env has {n_t} + {n_ct}")
    return n_t, n_ct


def resolve_pin_pitch(args, verbose: bool = True, build_vis: bool = True) -> int:
    """R0-E.2 (#131): set/validate args.pin_pitch from args.map_data; returns it.

    WHAT: ``args.pin_pitch is None`` (CLI default) ⇒ pin_pitch_for_map(
    args.map_data). An explicit 0/1 is cross-checked against the same test
    and refused with ValueError (never assert) when it disagrees with the map.

    WHY a separate function: train() is too heavy to exercise in a unit test,
    and this block MUST run before build_train_env_factory — that call reads
    args.pin_pitch through env_config_from_args and bakes the resulting
    EnvConfig into every worker env at vector.make; resolving later would leave
    the envs unpinned while the policy gets aim_dim_mask=[1,0] and
    assert_pin_pitch_agreement aborts the run.

    PITFALL: args.map_data is None for `--map dust2`/`--dust2`; the helper
    LOADS the map (cached). main() calls this ABOVE the --dump-config exit on
    purpose — the Modal fingerprint dump must carry the geometry-resolved value
    (costs ~1 s for dust2 from the nav cache, ~0.8 s for `from cs2rl.env import map`). train()
    calls it again as a cache-safe cross-check for programmatic callers (second
    call is silent, see `verbose`). verbose=False for the train() cross-check
    so the value is printed once per launch. main() passes build_vis=False for
    --dump-config (gh#251): the dump needs geometry only, and a cold dust2 vis
    cache would otherwise fork cpu_count() build workers before the dump exits.
    """
    flat = bool(pin_pitch_for_map(getattr(args, "map_data", None), build_vis=build_vis))
    if getattr(args, "pin_pitch", None) is None:
        args.pin_pitch = int(flat)
    if bool(args.pin_pitch) != flat:
        raise ValueError(f"pin_pitch={args.pin_pitch} but map flat={flat}: pin pitch only on "
                         f"flat maps (pass --pin-pitch {int(flat)} or omit it)")
    if verbose:
        print(f"[Train] pin_pitch={int(args.pin_pitch)} (map flat={flat})")
    return int(args.pin_pitch)


def assert_pin_pitch_agreement(vecenv, policy):
    """R0-E.2 (#131) startup check: env sd->pin_pitch ⇔ policy.aim_dim_mask[1] == 0.

    WHAT: reads StaticData.pin_pitch off the driver env (same path
    build_policy uses for max_turn_speed) and compares it with the policy's
    aim-dim mask. Raises RuntimeError on mismatch.

    WHY: the two sides are set independently (env_config_from_args bakes the
    flag into the EnvConfig every worker is built with at vector.make time;
    build_policy sets the mask from args.pin_pitch) and a mismatch is silent —
    the env would ignore a dim the trainer still scores, or score a dim the env
    still applies.

    PITFALL: unlike _kill_reward_is_active there is NO soft fallback — a
    non-C env here is a wiring bug and must stop the run (RuntimeError, never
    a bare assert: python -O would strip it).
    """
    env = getattr(vecenv, "driver_env", vecenv)
    try:
        c_pin = int(env._c_env.sd.contents.pin_pitch)
    except AttributeError as e:
        raise RuntimeError("assert_pin_pitch_agreement: driver_env is not a Cs2Env") from e
    p_pin = int(float(policy.aim_dim_mask[1]) == 0.0)
    if c_pin != p_pin:
        raise RuntimeError(f"pin_pitch mismatch: env={c_pin} policy={p_pin} "
                           f"(aim_dim_mask={policy.aim_dim_mask.tolist()})")


def assert_max_turn_speed_agreement(vecenv, policy):
    """R0-G startup check: env sd->max_turn_speed == policy.max_turn_speed.

    WHAT: reads StaticData.max_turn_speed off the driver env and compares it
    with the policy's non-trainable max_turn_speed buffer (the tanh scale on
    the aim head). Raises RuntimeError on mismatch (>1e-6 rad/tick).

    WHY: build_policy copies the value from the driver env at construction,
    but a resumed/warm-started checkpoint carries its OWN buffer — a run
    resumed with a different --max-turn-speed would have the policy emit aim
    deltas the env then clamps, silently changing the action semantics.

    PITFALL: RuntimeError, never a bare assert (python -O strips asserts). A
    non-Cs2Env driver is a wiring bug and must also stop the run.
    """
    env = getattr(vecenv, "driver_env", vecenv)
    try:
        c = float(env._c_env.sd.contents.max_turn_speed)
    except AttributeError as e:
        raise RuntimeError("assert_max_turn_speed_agreement: driver_env is not a Cs2Env") from e
    p = float(policy.max_turn_speed)
    if abs(c - p) >= 1e-6:
        raise RuntimeError(f"max_turn_speed mismatch: env={c} policy={p}")


def assert_eval_env_agreement(eval_env, driver_env):
    """R0-I startup check: the fixed-baseline eval env matches the training envs.

    WHAT: two comparisons, in this order.
      (a) the five APPLIED attributes, read off the two live Cs2Envs. These
          are the RESOLVED values: for round_time that is the tick count the
          sentinel becomes, not the sentinel — but the resolution is a
          deterministic function of the config field (Cs2Env.__init__ falls
          back to the env/nav.py constant when it is None), and the other four are
          plain copies of the config fields, so (a) cannot fire anywhere (b)
          is silent. It runs FIRST so that a divergence in one of the five
          still raises with the message the inline loop raised before this
          function existed — bare knob name, no `config.` prefix.
          NEITHER (a) NOR (b) reads C state: (a) reads Python attributes (and
          the round_time property, which returns Cs2Env._round_time) and (b)
          reads the frozen config object. The two agreement checks above go
          through env._c_env.sd.contents; this one never touches ctypes.
      (b) every EnvConfig field except reward_symmetrize, read off the two
          objects' `.config`. This is INTENT, and it covers the 27 scalars
          (a) cannot see — the 23 reward weights (no attribute of the env
          exposes them) plus the four knobs that are fields but not attributes
          of the env. (EnvConfig has 11 fields; (b) compares 10 of them, and
          the four non-attribute knobs are pbrs_gamma, recoil, laser_range and
          max_turn_speed. 23 + 4 = 27; MEASURE this again if a field is ever
          added, and do not write a number here you have not counted off
          dataclasses.fields.)

    WHY reward_symmetrize is skipped and nothing else is: env.factory.build_eval_env
    FORCES it off — `config.replace(reward_symmetrize=False)` — so the eval env
    reports raw rewards while the training envs take the flag from args. Since
    #165 PR B2 that is an explicit rule at the builder rather than, as before, a
    parameter the eval chain simply never passed. Either way the two configs are
    MEANT to differ there and only there, and skipping any other field would
    hide a real divergence.

    DISCLOSURE — NEITHER CHECK CAN FIRE ON ANY INPUT REACHABLE TODAY, and since
    #165 PR B2 that is structural rather than measured. train()'s only call site
    compares the env from build_eval_env(..., config=env_config_from_args(
    args)) against the driver env from build_train_env_factory(args, ...), which
    passes build_env_factory the config from that SAME resolver called on the
    same args. So the two sides are one config expression evaluated twice, and
    the only field either builder then touches is reward_symmetrize — the field
    (b) skips and (a) does not compare. Corroborated by measurement, re-run
    2026-09-04 on this tree by driving those two real constructions over four
    arg sets (no knobs; --reward-symmetrize; crouch_enabled and jump_enabled
    both off; --reward-symmetrize with --round-time-ticks and --laser-range):
    reward_symmetrize was the ONLY EnvConfig field that ever differed, and
    neither check raised. Both checks are therefore guards against a FUTURE
    divergence, not checks with anything to catch now; keep them, and re-derive
    this paragraph the day either builder starts setting a field the other does
    not.

    PITFALL — WHAT THAT SINGLE EXPRESSION IS KEEPING SAFE. (b) compares
    `round_time` as a CONFIG FIELD, and that field has two spellings for one
    applied value: None means "the env/nav.py constant" and Cs2Env resolves it, so a
    config pair holding None on one side and that same constant on the other is
    behaviourally identical and would still abort the run here. Unreachable only
    because both sides come from one `env_config_from_args(args)`, never from
    two independently-written knob sources. If a future caller ever builds the
    eval config separately, normalise round_time before comparing it — do not
    discover this by aborting a run for no behavioural reason.

    WHY THIS IS A MODULE-LEVEL FUNCTION and not the inline loop it replaces:
    the loop sat inside train(), which needs a real run to reach — and not even
    that by default, since it is behind `--eval-interval`, which is 0 unless
    asked for. So the check nobody could run was also the check nobody could
    test. assert_pin_pitch_agreement and assert_max_turn_speed_agreement above
    have the same shape — module-level, called from train(), and called
    DIRECTLY by tests (tests/train/test_pitch_pin.py and
    tests/train/test_env_knobs.py::test_policy_max_turn_speed_assert respectively).

    PITFALL: check (a) runs FIRST, so a test that tries to prove (b) exists by
    differing one of the five names in the tuple below will raise from (a) and
    prove nothing. tests/env/test_env_factory.py::test_eval_env_agreement_two_directions
    handles that by demanding the MESSAGE rather than just a raise: it differs
    EVERY EnvConfig field, one per parametrized case, and requires `on <knob>`
    (which is (a)'s spelling, and which (b)'s `on config.<knob>` does not contain)
    for the five names below, `on config.<field>` for the five fields that are
    outside the tuple and still compared by (b) — rewards, pbrs_gamma, recoil,
    laser_range, max_turn_speed — and NO raise for reward_symmetrize, the sixth
    field outside the tuple and the one (b) skips. Those five are the only cases
    that can come from (b) alone, so they are what makes a widened skip list
    above visible. Before that parametrization the test differed two fields, and
    widening the skip list to nine left it green (re-measured 2026-09-04 against
    the pre-fix test body; the review measured the whole non-slow suite green
    with it).

    PITFALL: RuntimeError, never a bare assert (python -O strips asserts).
    """
    for _k in ("n_active_per_team", "pin_pitch", "crouch_enabled", "jump_enabled", "round_time"):
        if getattr(eval_env, _k) != getattr(driver_env, _k):
            raise RuntimeError(f"[Eval] eval env / driver env disagree on {_k}: "
                               f"{getattr(eval_env, _k)!r} vs {getattr(driver_env, _k)!r}")
    for _f in dataclasses.fields(EnvConfig):
        if _f.name == "reward_symmetrize":
            continue
        if getattr(eval_env.config, _f.name) != getattr(driver_env.config, _f.name):
            raise RuntimeError(f"[Eval] eval env / driver env disagree on config.{_f.name}: "
                               f"{getattr(eval_env.config, _f.name)!r} vs "
                               f"{getattr(driver_env.config, _f.name)!r}")


def pin_pitch_for_map(map_data, *, build_vis: bool = True) -> int:
    """R0-E.2 (#131): 1 iff the map is FLAT (every area centroid shares one z).

    WHAT: pure geometry test on the MapData the envs will actually run on.
    ``map_data=None`` means "the cs2 nav map" (exactly what make_env(map_data=
    None) loads, via the same _ENV_CACHE), so None is resolved by LOADING that
    map — never by treating the sentinel as a map property.

    WHY: a flat map has nothing to aim up/down at, so pitch is pure noise
    (spec §R0-E.2) and gets pinned; a map with elevation must keep the pitch
    dim trainable. Deciding on the sentinel (`map_data is None` ⇒ flat) was
    the Task 9 review's Critical #1: the CLI `--dust2` path passes None, so
    the value MUST come from the loaded map, not from the marker.

    PITFALL: the in-sim dust2 (map.make_cs2_map, "verticality deferred")
    zero-fills centroids_z, so today this returns 1 for dust2 — by spec (plan
    §R0-E.2: pinned on flat maps incl. dust2). There is NO name-based table:
    the `--map` path (build_map_data → resolve_pin_pitch) and the `--dust2`
    path both end here. When real dust2 verticality lands this flips to 0 by
    itself and every dust2 resume is refused by the config guard (pin_pitch is
    not allowlisted) — the intended tripwire.

    ``build_vis=False`` (gh#251, `--dump-config` only): resolve None WITHOUT
    the visibility matrix — make_cs2_map(build_vis=False), NOT stored in
    _ENV_CACHE. The answer reads centroids_z only, so it is identical; what is
    skipped is the cold-cache vis build, which forks cpu_count() workers that a
    killed dump used to orphan (~900 MB each). A warm _ENV_CACHE entry is still
    reused.
    """
    md = map_data
    if md is None:
        # Same cache key make_env uses, so train() never loads the nav twice.
        from cs2rl.env import nav
        from cs2rl.env.c.cs2_env import _ENV_CACHE
        from cs2rl.env.map import make_cs2_map
        key = (nav.NAV_PATH, nav.CACHE_PATH)
        md = _ENV_CACHE.get(key)
        if md is None:
            md = make_cs2_map(nav.NAV_PATH, nav.CACHE_PATH, build_vis=build_vis)
            if build_vis:              # a vis-less MapData must never reach make_env
                _ENV_CACHE[key] = md
    z = np.asarray(md.centroids_z, dtype=np.float32)
    return int(float(z.max() - z.min()) == 0.0)
