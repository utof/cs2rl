"""Minimal PuffeRL trainer harness for Batch-1 trainer-level tests (Task 6b).

Downstream tasks in the Batch-1 rewards overhaul (utof/cs2rl#9 Task 6c,
Tasks 7-11) need to call ``trainer.evaluate()`` / ``trainer.train()`` on a real
PuffeRL instance to exercise rollout-time code paths (reward clamp removal,
Welford normalization, event masks, etc.). Spinning up the full production
``train()`` flow in a test is too heavy — it pulls in multiprocessing vec
workers, wandb init, checkpoint dirs, dead-run detection, etc. This module
builds the smallest PuffeRL trainer that still has the production attribute
surface the tests need.

What's stripped vs. production ``src.train.train()``:
    - vec backend: always ``pufferlib.vector.Serial`` (no worker processes, no
      shared-memory handshake, deterministic teardown).
    - checkpoint_dir: a ``tempfile.mkdtemp()`` scratch dir, wiped by cleanup().
    - wandb / metrics.jsonl / dead-run detection / save loop: omitted.
    - nothing on the trainer itself (gh#168 W1.5): the harness constructs
      ``trainer.Cs2PuffeRL``, the production class, so return-norm, hybrid-aim,
      self-play evaluate() and full checkpointing are ALWAYS on, exactly as in
      ``train()``. Before W1.5 it composed a bare PuffeRL plus two patches and
      26 call statements (several inside shared helpers) applied return-norm
      themselves; ``tests/`` now applies no patch function. (The timing patch no longer exists: 2cffc35 moved timing to
      train()'s call sites.)

Design contract:
    - Public API is ``_build_trainer_for_test(...)``, which returns
      ``(trainer, cleanup)`` with ``type(trainer) is trainer.Cs2PuffeRL``
      (pinned by tests/train/test_trainer_composition.py), plus (gh#168 W1)
      ``_harness_parts(...)``, which returns everything built BEFORE the
      trainer constructor so a test can construct ``Cs2PuffeRL(**parts)``
      itself. Tuple rather than contextmanager because
      (a) the smoke-test shape in the task spec uses try/finally, and
      (b) downstream tests may need to leak the trainer between helper
      functions — easier with an explicit cleanup callable.
    - ``with_selfplay`` defaults to False (Task 6c needs True; Tasks 7-11
      default False to keep the attribute surface small). A test that needs
      its OWN SelfPlayManager (a pre-seeded pool, p_past=1.0) passes it as
      ``self_play_mgr=`` and the harness constructs the trainer around it.

If PufferLib ever renames ``PuffeRL`` or drops a constructor kwarg, this file
is where the breakage will first surface — giving us a single ~line-diff review
surface for API drift instead of chasing it across every Batch-1 test.
"""

from __future__ import annotations

import multiprocessing as mp
import shutil
import tempfile
import types

from cs2rl.env.config import EnvConfig
from cs2rl.env.factory import build_env_for
from cs2rl.train.selfplay import build_selfplay_manager

# The harness's four env-knob defaults are the dataclass's, read once rather
# than copied. Four literals here would be four more places #165 has to keep in
# step with env/config.py, and tests/integration/test_no_restated_env_defaults.py fails on
# exactly that shape — including the `: int = <literal>` spelling, which a
# regex written for `name = value` alone cannot see.
_ENV_DEFAULTS = EnvConfig()


def _harness_parts(
    num_envs: int = 32,
    with_selfplay: bool = False,
    device: str = "cpu",
    seed: int = 0,
    tct_split_heads: bool = False,
    tct_split_trunk: bool = False,
    n_active_per_team: int = _ENV_DEFAULTS.n_active_per_team,
    map_data=None,
    pin_pitch: int = _ENV_DEFAULTS.pin_pitch,
    crouch_enabled: int = _ENV_DEFAULTS.crouch_enabled,
    jump_enabled: int = _ENV_DEFAULTS.jump_enabled,
    aim_log_std_max=None,
    aim_entropy_bonus: bool = True,
    opponent: str = "self",
    self_play_mgr=None,
):
    """Everything ``_build_trainer_for_test`` builds BEFORE the trainer constructor.

    WHAT: the Serial vecenv, the policy, the train config, the action-mask shm view,
    the participation vector and the self-play manager, returned as two dicts:
    ``parts`` (keyed exactly like ``trainer.Cs2PuffeRL.__init__``'s parameters, so
    ``Cs2PuffeRL(**parts)`` constructs the production class) and ``pins`` (the
    ``mask_shm`` RawArray the trainer must hold against GC, and ``tmp_checkpoint_dir``
    for cleanup). Parameters are ``_build_trainer_for_test``'s; see its docstring.

    WHY a separate function (gh#168 W1): production's train() builds the participation
    vector and the (possibly pre-seeded) self-play manager BEFORE constructing the
    trainer, because Cs2PuffeRL.__init__ reads both at patch time. The harness used
    to build them after a bare PuffeRL and hand them to the patchers one by one. This
    is the same hoist, so tests/train/test_trainer_composition.py can construct the
    production class from the harness's parts, and (W1.5) ``_build_trainer_for_test``
    itself returns that class with no further re-ordering.
    Measured neutral: build_participating_rows is pure and SelfPlayManager.__init__
    draws no RNG (construct_snapshot.py: 39 attrs x 4 configs, 0 diffs).

    PITFALL: ``cont_action_view_main`` is None on purpose. The harness is Serial-only,
    and its HybridAimVecEnv receives no Multiprocessing shared-memory view.
    The None is also explicit in ``parts`` for ``Cs2PuffeRL(**parts)``.

    ``self_play_mgr`` (gh#168 W1.5): a caller-built SelfPlayManager is used AS IS and
    ``build_selfplay_manager`` is not called (tests/train/test_selfplay_factory.py spies on
    that call and must see exactly one when nothing is given). The two tests that need
    their own manager (tests/train/test_resume_state.py, tests/train/test_pitch_pin.py) used to
    apply the self-play monkey-patch a second time on top of the harness's; now
    ``Cs2PuffeRL._init_selfplay`` reads the manager once. Its pool is read only inside evaluate(),
    so a caller may seed it AFTER construction.
    """
    # Imports are function-local so importing this module in a test that
    # doesn't actually call the factory (e.g. a smoke import test) is free.
    # Cs2PuffeRL is lazy-imported inside train.train() in production (gh#168 W1
    # dropped train()'s own PuffeRL import) and inside _build_trainer_for_test
    # here; keeping every heavy import function-local also keeps this module
    # importable even if pufferlib's optional torch deps are mid-install in an
    # isolated test runner.
    import pufferlib.vector

    from cs2rl.env.map import make_simple_map
    from cs2rl.policy import build_policy
    from cs2rl.train.config import (
        assert_opponent_self_play_compatible,
        build_participating_rows,
        build_train_config,
        compute_batch_dims,
    )
    from cs2rl.train.selfplay import SelfPlayManager
    from cs2rl.train.trainer import HybridAimVecEnv

    # Rung 1a T3: mirror of train()'s startup guard. `with_selfplay` is the
    # harness's spelling of "self-play bookkeeping on" (it is what sets
    # p_past > 0), so it is the flag to check.
    assert_opponent_self_play_compatible(opponent, with_selfplay)

    # ── Scratch dir for config.json + any checkpoints PuffeRL writes ────────
    # PuffeRL's constructor doesn't actually write to data_dir during __init__,
    # but build_train_config requires it as a string. Using mkdtemp keeps each
    # harness instance isolated; cleanup removes it wholesale.
    tmp_checkpoint_dir = tempfile.mkdtemp(prefix="cs2rl-harness-")
    # Everything from here to the return runs under ONE try (gh#168 W1.5 review):
    # `cleanup` exists only once _build_trainer_for_test returns, so any raise in
    # between (build_train_config → resolve_opponent_mode on a bad `opponent`,
    # build_policy on a cap outside the band, build_participating_rows,
    # build_selfplay_manager, the given-manager asserts above the return) used
    # to leak the scratch dir and, once built, the Serial vecenv. Measured:
    # `opponent="bogus"` left a /tmp/cs2rl-harness-* behind.
    vecenv = None
    try:
        # ── Shared team-spirit value (production pattern) ───────────────────────
        # Production uses mp.Value so multiprocess workers can read a scalar that
        # the main trainer anneals each epoch. Serial backend doesn't actually
        # need shared memory, but the harness role builder requires a shared_ts.
        shared_ts = mp.Value("f", 0.3)

        # Simple 5-room map — the production default for non-dust2 runs. Avoids
        # depending on any pre-generated mapdata file on disk. R0-E: overridable.
        if map_data is None:
            map_data = make_simple_map()

        # ── F8: action-mask shm, same env→trainer pattern as production ─────────
        # Serial backend runs envs in-process, but the RawArray pattern is kept
        # identical to train.train() so harness-built trainers exercise the REAL
        # masked rollout path (sampler + rollout buffer + PPO-loss mask).
        from multiprocessing import RawArray

        import numpy as np

        from cs2rl.spec.action import ACTION_MASK_DIM

        # (`from nav import TEAM_SIZE` used to sit in this import block for the
        # participation-row formula below; Rung 1a T3 moved that formula into
        # train.build_participating_rows, the one copy production also calls.)
        _agents_per_env = 10
        mask_shm = RawArray("b", num_envs * _agents_per_env * ACTION_MASK_DIM)
        mask_view_main = np.frombuffer(mask_shm,
                                       dtype=np.int8).reshape(num_envs * _agents_per_env,
                                                              ACTION_MASK_DIM)

        # One config for every env this trainer builds, so the harness cannot drift
        # from production in the only way that matters: what the env is constructed
        # with. The four values come from this function's own parameters, and the
        # `args` namespace below is built from the SAME four, so build_train_config
        # records exactly what the envs ran with.
        config = EnvConfig(n_active_per_team=n_active_per_team,
                           pin_pitch=pin_pitch,
                           crouch_enabled=crouch_enabled,
                           jump_enabled=jump_enabled)

        def env_factory(*_args, buf=None, seed=None, _mask_idx=None, **_kwargs):
            # W3 (#154): construction routes through the role factory. What USED to
            # be spelled out here — the `0 if seed is None else seed` remap (an
            # explicit None check, not `seed or 0`, so a legitimate seed=0 survives)
            # and the unconditional include_step_stats_in_info=True (uniform
            # attribute/info surface across selfplay and no-selfplay modes; one
            # pre-built singleton dict per env, no per-tick allocation) — now lives
            # in env.factory._build_harness with the same reasoning attached.
            #
            # The harness is production-SHAPED on purpose, but it is not the `train`
            # role: it adds include_step_stats_in_info and takes its knobs as plain
            # arguments rather than from a CLI-derived dict, so it has its own.
            #
            # WHAT COVERS THE FOUR-KNOB MAPPING ABOVE, knob by knob — it is not one
            # test, and #165 PR B2 changed which.
            # tests/fixtures/env_config_pre_165b.json holds the config and runtime
            # kwargs this call produced before the builders were typed, and
            # tests/env/test_env_factory.py drives this closure against both harness rows
            # (test_harness_call_site_forwards_the_captured_kwargs). Those two rows
            # differ from each other in n_active_per_team and jump_enabled, so the
            # fixture sees either of THOSE dropped from the mapping — but both rows
            # hold the FIELD DEFAULT for pin_pitch and for crouch_enabled, so it is
            # blind to either of those two going missing.
            #   pin_pitch     is caught outside this file, by
            #                 tests/train/test_pitch_pin.py::test_env_trainer_pin_agreement_raises:
            #                 it builds a harness trainer with a non-default pin and
            #                 then calls assert_pin_pitch_agreement, which reads
            #                 StaticData.pin_pitch off the DRIVER ENV and compares it
            #                 with the policy mask. A mapping that dropped pin_pitch
            #                 would send the envs the field default while build_policy
            #                 still got the parameter, and that check would raise.
            #   crouch_enabled is caught by NOTHING ELSE — no test in the tree passes
            #                 it to _build_trainer_for_test. Its only cover is
            #                 test_harness_config_carries_the_knobs_no_fixture_row_varies
            #                 in tests/env/test_env_factory.py, which drives this closure
            #                 off-fixture with a non-default crouch. Delete that test
            #                 and this comment becomes false in the same edit.
            env = build_env_for("harness",
                                shared_ts=shared_ts,
                                buf=buf,
                                seed=seed,
                                map_data=map_data,
                                config=config)
            # STAYS AT THE CALL SITE, outside the factory: this needs the harness's
            # own shm handle and the per-env index pufferlib passes in, neither of
            # which is the factory's business.
            if _mask_idx is not None:
                env._attach_mask_view(mask_shm, _mask_idx)
            return env

        # ── Vec env (Serial: no worker processes) ───────────────────────────────
        # Serial makes teardown synchronous and deterministic — critical for
        # pytest where a lingering process would block the whole session.
        # Factory-list form (not single callable) for the same reason as
        # production: per-env kwargs survive only when env_creators is a list
        # (pufferlib vector.py broadcast quirk).
        vecenv = pufferlib.vector.make(
            [env_factory] * num_envs,
            env_args=[[] for _ in range(num_envs)],
            env_kwargs=[{
                "_mask_idx": i
            } for i in range(num_envs)],
            num_envs=num_envs,
            backend=pufferlib.vector.Serial,
        )

        vecenv = HybridAimVecEnv(vecenv)

        # ── Minimal argparse-shaped config object ───────────────────────────────
        # build_train_config reads these attributes. Everything else in the
        # production parser (wandb, vec-backend, etc.) is irrelevant once we've
        # already instantiated the vecenv.
        # Tiny horizon — ONE evaluate() round is all downstream tests need.
        # PuffeRL requires total_timesteps >= batch_size; pad by NUM_ROLLOUT_ROUNDS
        # so a few back-to-back evaluate() calls in a single test stay within the
        # configured timestep budget.
        NUM_ROLLOUT_ROUNDS = 4
        _agents_per_env, bptt_horizon, batch_size = compute_batch_dims(num_envs)
        args = types.SimpleNamespace(
            device=device,
            seed=seed,
            timesteps=batch_size * NUM_ROLLOUT_ROUNDS,
            checkpoint_dir=tmp_checkpoint_dir,
            n_active_per_team=n_active_per_team,
            pin_pitch=pin_pitch,
            crouch_enabled=crouch_enabled,
            jump_enabled=jump_enabled,
            aim_log_std_max=aim_log_std_max,
            aim_entropy_bonus=aim_entropy_bonus,
            opponent=opponent,
        )
        train_config = build_train_config(args, batch_size=batch_size, bptt_horizon=bptt_horizon)

        # Small-env tests: build_train_config pins minibatch_size =
        # max_minibatch_size = 8192, and PuffeRL raises APIUsageError when
        # batch_size < minibatch_size (pufferl.py:121-124). batch_size =
        # num_envs*640, so anything under 16 envs cannot construct a trainer at
        # all. Clamp HERE (harness only) so tests can use num_envs=4/8 without
        # touching the production (fingerprinted) config. Subprocess tests that go
        # through train.py's real CLI must still use --num_envs >= 16.
        # PITFALL: this changes total_minibatches / accumulate_minibatches for
        # sub-16-env harness trainers — do not port it into build_train_config.
        train_config["minibatch_size"] = train_config["max_minibatch_size"] = min(8192, batch_size)

        # R0-E: cap + pin go to the policy exactly as train() passes them, so the
        # harness policy carries aim_log_std_max / aim_dim_mask. build_policy
        # raises ValueError on a cap outside the band; the enclosing try closes the
        # vecenv so a refused harness does not leak the Serial envs.
        policy = build_policy(vecenv,
                              device,
                              tct_split_heads=tct_split_heads,
                              tct_split_trunk=tct_split_trunk,
                              aim_log_std_max=aim_log_std_max,
                              pin_pitch=bool(pin_pitch))
        # Rung 0 §2.2 + Rung 1a T3: the SHARED helper train() calls, not a copy of
        # its formula. The copy was the hazard: a participation change patched into
        # only one of the two left the headline harness test green against a
        # formula production never ran. Built here, BEFORE the constructor and not
        # inside the trainer, so the harness mirrors production's call order:
        # train() builds it before Cs2PuffeRL._init_hybrid_aim reads it.
        participating_rows = build_participating_rows(num_envs,
                                                      n_active_per_team,
                                                      opponent_mode=opponent,
                                                      hero_team=SelfPlayManager.initial_hero_team())

        # ── Self-play manager (the patch is always applied at T5) ──────────────
        # Pre-Batch-3: this was gated on `with_selfplay` so the no-selfplay path
        # could exercise PufferLib's library evaluate(). T4 changed the policy
        # contract to a 4-tuple; PufferLib's library evaluate still expects a
        # 2-tuple, so the no-selfplay path can't run end-to-end without our
        # hybrid-aware evaluate() override. Cs2PuffeRL defines that method
        # directly; it never calls the stock evaluate();
        # `with_selfplay=False` now means "no past-policy mixing" (empty pool
        # never activates) — the replacement evaluate() still runs. The
        # `with_selfplay=True` path additionally pre-seeds the manager. This
        # keeps the test harness honest with production where the hybrid-aim
        # rollout requires the patched evaluate path.
        #
        # W3 (#154): this used to be an `if not with_selfplay: ... else: ...` whose
        # two SelfPlayManager constructions were IDENTICAL apart from
        # `p_past=0.0` / `p_past=0.3` — and production's third copy computed the same
        # two values as `0.3 if self_play_enabled else 0.0`. That rule is now
        # build_selfplay_manager's, taking the FLAG, so all three sites collapse onto
        # one call and the branch disappears with them. The pre-migration shapes of
        # all three are frozen in tests/fixtures/selfplay_kwargs_pre_w3.json and
        # tests/train/test_selfplay_factory.py asserts the builder still produces each —
        # which is the only oracle here, since the §3 gate runs --no-self-play and
        # nothing on this branch reaches the harness at all.
        if self_play_mgr is None:
            self_play_mgr = build_selfplay_manager(
                self_play_enabled=with_selfplay,
                aim_log_std_max=aim_log_std_max,
                pin_pitch=pin_pitch,
                opponent_mode=opponent,
            )
        else:
            # A caller-built manager must agree with the envs and the policy this
            # call built, or the harness would compose a trainer production can
            # never reach (the statue team and the pitch mask are read from the
            # manager inside evaluate()). When a manager is given its p_past IS
            # the self-play flag, so the startup guard (run on `with_selfplay`
            # above, before mkdtemp) is re-run on the manager here, not
            # replaced. `aim_log_std_max` is trusted: the
            # manager only forwards it to past-policy loading, which the two
            # current callers (tests/train/test_resume_state.py, tests/train/test_pitch_pin.py)
            # never reach with a non-default cap.
            assert_opponent_self_play_compatible(opponent, self_play_mgr.p_past > 0)
            assert (self_play_mgr.opponent_mode == opponent
                    and self_play_mgr.pin_pitch == bool(pin_pitch)), (
                        f"self_play_mgr disagrees with the harness knobs: manager "
                        f"opponent_mode={self_play_mgr.opponent_mode!r} pin_pitch="
                        f"{self_play_mgr.pin_pitch!r} vs opponent={opponent!r} "
                        f"pin_pitch={bool(pin_pitch)!r}")
        parts = dict(
            config=train_config,
            vecenv=vecenv,
            policy=policy,
            cont_action_view_main=None,
            mask_view_main=mask_view_main,
            participating_rows=participating_rows,
            self_play_mgr=self_play_mgr,
        )
        pins = dict(mask_shm=mask_shm, tmp_checkpoint_dir=tmp_checkpoint_dir)
    except BaseException:
        if vecenv is not None:
            vecenv.close()
        shutil.rmtree(tmp_checkpoint_dir, ignore_errors=True)
        raise
    return parts, pins


def _build_trainer_for_test(
    num_envs: int = 32,
    with_selfplay: bool = False,
    device: str = "cpu",
    seed: int = 0,
    tct_split_heads: bool = False,
    tct_split_trunk: bool = False,
    n_active_per_team: int = _ENV_DEFAULTS.n_active_per_team,
    map_data=None,
    pin_pitch: int = _ENV_DEFAULTS.pin_pitch,
    crouch_enabled: int = _ENV_DEFAULTS.crouch_enabled,
    jump_enabled: int = _ENV_DEFAULTS.jump_enabled,
    aim_log_std_max=None,
    aim_entropy_bonus: bool = True,
    opponent: str = "self",
    self_play_mgr=None,
):
    """Build a tiny in-process ``Cs2PuffeRL`` trainer for trainer-level tests.

    Parameters
    ----------
    num_envs : int
        Vectorised-env count. 32 is enough to populate the rollout buffer in
        one evaluate() round on the Serial backend in under ~5s; bump only if
        a test specifically needs more parallelism (cost is roughly linear).
    with_selfplay : bool
        The ``self_play_enabled`` flag handed to ``build_selfplay_manager``
        (p_past 0.3 when True, 0.0 when False). Either way the trainer's
        evaluate() is the self-play replacement (Cs2PuffeRL always installs
        it) and the manager's pool starts EMPTY, so ``should_use_past()``
        returns False and the past-policy branch is NOT exercised unless a
        test seeds the pool. Required by Task 6c (reward-clamp removal test).
    device : str
        Torch device. Default "cpu" keeps tests deterministic and CI-friendly.
    seed : int
        Passed into both the env factory and the PPO config. Fixed default so
        back-to-back harness invocations produce identical rollouts.
    tct_split_heads : bool
        Batch 7 (spec 2026-08-13): build the policy with per-team T/CT policy
        heads. Default False keeps every existing harness caller on the legacy
        architecture bit-for-bit. Exists so trainer-level split contracts (the
        obs-bit ⇔ slot-index invariant) can be pinned against a real rollout.
    tct_split_trunk : bool
        Spec 2026-08-15: build the policy with per-team T/CT encoder+LSTM.
        Default False keeps every existing harness caller on the shared trunk.
        Independent of ``tct_split_heads`` — either bit can be on alone.
    n_active_per_team : int
        Rung 0 (spec 2026-08-29 §2.1/§2.2): agents per team the env spawns;
        slots ``n..4`` of each team are parked (noop-masked, zero reward).
        Threaded into BOTH the envs (through the EnvConfig the closure below
        builds) and the trainer (``participating_rows`` →
        ``trainer.participating``), because a test
        that set only one of the two would be testing a configuration
        production can never reach. The default is EnvConfig's own field default
        — full N-vs-N, i.e. an all-ones participation mask — which is what every
        pre-Rung-0 harness caller gets. Named, never spelled: a literal here is
        a second declaration of the value, and it sits on a different line from
        the parameter so tests/integration/test_no_restated_env_defaults.py's line probe
        cannot see it go stale.
    map_data : MapData or None
        R0-E: the map every env is built on. None ⇒ ``make_simple_map()`` (the
        pre-R0-E hardcoded default). Pass the session ``simple_map`` fixture or
        an arena map; the harness never inspects flatness (that check lives in
        train() only), so pin_pitch=1 on a non-flat map is allowed HERE.
    pin_pitch, crouch_enabled : int
        R0-E.2 sim knobs, threaded into BOTH the envs (through the EnvConfig
        the closure below builds) and the policy (``build_policy(pin_pitch=)``
        → aim_dim_mask) and both SelfPlayManager constructions — exactly like production, so
        ``assert_pin_pitch_agreement`` holds on a harness trainer.
    jump_enabled : int
        Rung 1a sim knob (spec 2026-08-30 T2b): 0 masks the jump action.
        Threaded into the envs and into ``args`` (⇒ config ``jump_enabled``),
        but NOT into build_policy — unlike pin_pitch it changes no action
        dimension, only a mask bit, so there is no policy-side mirror to keep
        in agreement. The default is EnvConfig's own field default — jump live,
        i.e. every pre-Rung-1a caller is unaffected. Named rather than spelled,
        for the reason under n_active_per_team above.
    aim_log_std_max : float or None
        R0-E.3 per-run σ cap → ``policy.aim_log_std_max`` and config
        ``aim_log_std_max``. None ⇒ LOG_STD_MAX.
    aim_entropy_bonus : bool
        R0-E.4 → config ``aim_entropy_bonus`` (the CLI spells it "on"/"off";
        build_train_config accepts both).
    opponent : str
        Rung 1a T3 (spec 2026-08-30): ``"self"`` (default, today's behaviour —
        both teams train) or ``"noop"`` (the opponent team is a stationary
        statue: no-op action bin on every head and NON-participating rows).
        Threaded into ``args`` (⇒ config ``opponent`` and the halved
        participating-step budget), into ``build_participating_rows`` and into
        BOTH SelfPlayManager constructions — the same three surfaces train()
        touches, so a harness rollout exercises the production statue path
        rather than a harness-only imitation of it.
        PITFALL: ``opponent="noop"`` with ``with_selfplay=True`` is refused
        here exactly as train() refuses ``--opponent noop`` without
        ``--no-self-play`` — a past-policy opponent is not a statue, and
        maybe_switch_teams would flip the statue's team mid-run.
    self_play_mgr : SelfPlayManager or None
        gh#168 W1.5: a caller-built manager the constructor uses INSTEAD of
        ``build_selfplay_manager``'s (see ``_harness_parts``). None (default)
        builds one from ``with_selfplay`` and the knobs above.

    Returns
    -------
    trainer : trainer.Cs2PuffeRL
        The production trainer class (gh#168 W1.5), fully composed: return-norm
        train(), hybrid-aim buffers, self-play evaluate(), full checkpointing
        save_checkpoint() and ``_timing`` are all present, exactly as train()
        builds it (tests/train/test_trainer_composition.py pins the surface, and
        tests/train/test_train_harness_smoke.py the stock attributes).
    cleanup : Callable[[], None]
        Idempotent teardown: closes the vecenv, wipes the scratch checkpoint
        dir. Tests MUST call this in a ``finally:`` to avoid leaking fd's /
        shared-memory / tmp dirs.

    Pitfalls
    --------
    - Re-using the same mp.Value across two harness instances in the same
      process is fine — ``shared_ts`` is a local to this call.
    - ``args`` is a ``types.SimpleNamespace``, not an argparse Namespace — just
      mirrors the attribute access that ``build_train_config`` does.
    - Serial backend means ``trainer.vecenv`` has a synchronous ``send``/``recv``
      cycle; tests can inject observations by monkey-patching those if needed.
    """
    # Lazy for the reason _harness_parts gives above its own import block (gh#168
    # W1): trainer.py subclasses PuffeRL, so it imports torch, and train, at module
    # scope. At THIS module's scope, `from cs2rl import train_test_harness` would load
    # torch, cs2rl.train and cs2rl.trainer, none of which it loads today (the torch is
    # what tests/train/test_w1_modules.py::test_import_train_test_harness_stays_light catches).
    # It would not be a cycle: with it at module scope, importing this module alone, or
    # train first and then this module, still succeeds.
    from cs2rl.train.trainer import Cs2PuffeRL

    parts, pins = _harness_parts(
        num_envs=num_envs,
        with_selfplay=with_selfplay,
        device=device,
        seed=seed,
        tct_split_heads=tct_split_heads,
        tct_split_trunk=tct_split_trunk,
        n_active_per_team=n_active_per_team,
        map_data=map_data,
        pin_pitch=pin_pitch,
        crouch_enabled=crouch_enabled,
        jump_enabled=jump_enabled,
        aim_log_std_max=aim_log_std_max,
        aim_entropy_bonus=aim_entropy_bonus,
        opponent=opponent,
        self_play_mgr=self_play_mgr,
    )
    vecenv = parts["vecenv"]
    tmp_checkpoint_dir = pins["tmp_checkpoint_dir"]
    # gh#168 W1.5: the production class, from the same parts train() builds. Its
    # __init__ applies return-norm, hybrid-aim (mask_view_main plumbed so harness
    # rollouts run MASKED, F8), the self-play evaluate() and full checkpointing,
    # so a harness test can no longer forget return-norm and run stock
    # PuffeRL.train() (gh#169), which cannot unpack the 4-tuple policy output.
    # The constructor is wrapped for the same reason _harness_parts runs under
    # one try: a raise here (PuffeRL's APIUsageError on a config it refuses, or
    # a patch-time assert) would otherwise leak the Serial envs and the scratch
    # dir, since `cleanup` only exists once we return. What this wrapper does
    # NOT cover is the Utilization thread PuffeRL.__init__ starts: it is
    # non-daemon and only its own stop() ends it, so a raise AFTER
    # super().__init__ would hang the interpreter at exit. Cs2PuffeRL.__init__
    # stops it itself on that path (tests/train/test_trainer_composition.py pins it).
    try:
        trainer = Cs2PuffeRL(**parts)
    except BaseException:
        vecenv.close()
        shutil.rmtree(tmp_checkpoint_dir, ignore_errors=True)
        raise
    # Pin the RawArray on the trainer against GC, after the constructor exactly
    # as train() pins its shm (the pin is a GC anchor, not trainer state).
    trainer._action_mask_shm = pins["mask_shm"]

    def cleanup():
        """Idempotent teardown. Safe to call twice.

        Both .close() calls are wrapped in try/except so a cleanup failure
        never masks a test assertion error. If PufferLib drops trainer.close()
        in a future release, this silently no-ops — the smoke test pins
        ``close`` in the attribute surface so the rename will be caught there.

        Since W1.5 the trainer is a Cs2PuffeRL, so ``trainer.close()`` runs
        the FULL-checkpointing ``save_checkpoint`` (policy + optimizer + RNG +
        self-play pool) before returning. That is a cost only, never a
        leak: ``data_dir`` is ``args.checkpoint_dir`` (train_config.py), i.e.
        this scratch dir, which the rmtree below removes wholesale.
        """
        try:
            trainer.close()
        except Exception:              # noqa: BLE001 — best-effort cleanup
            pass
                                       # vecenv.close() is safe to call multiple times on Serial.
        try:
            vecenv.close()
        except Exception:              # noqa: BLE001
            pass
        shutil.rmtree(tmp_checkpoint_dir, ignore_errors=True)

    return trainer, cleanup
