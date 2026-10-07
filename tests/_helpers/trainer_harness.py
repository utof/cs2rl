"""A small CPU ``Cs2PuffeRL`` for trainer-level tests, built by the production builder.

``_build_trainer_for_test(...)`` returns ``(trainer, cleanup)``. The trainer is built by
``cs2rl.train.compose.build_trainer``, the same function ``cs2rl.train.loop.train`` calls,
so its vector env, policy, participation rows, self-play manager and constructor are
production code. This module only chooses test inputs (#92):

    - an args namespace and a train config built from this function's parameters;
    - ``env_role="harness"``: Serial envs from ``build_harness_env`` (step stats in every
      info, pufferlib's seed), and no continuous-action shared array;
    - a scratch ``checkpoint_dir`` (``tempfile.mkdtemp``) that cleanup() removes;
    - minibatch_size clamped to the batch, so 4 or 8 envs can build a trainer.

What it does NOT do, because ``train()`` does it around the builder: W&B, metrics.jsonl,
the run id, seeding (call ``cs2rl.train.resume.seed_everything`` yourself), weight decay
and the separate aim-σ param group (a harness trainer keeps PuffeRL's one-group Adam with
no decay), the full-state resume, the eval hook, dead-run detection and the epoch loop.

``(trainer, cleanup)`` rather than a context manager: tests use try/finally and some
pass the trainer between helper functions.

``mp`` and ``tempfile`` are module attributes on purpose:
tests/env/test_env_factory.py replaces them on THIS module to capture the env factory.
"""

from __future__ import annotations

import multiprocessing as mp
import shutil
import tempfile
import types

from cs2rl.env.config import EnvConfig
from cs2rl.train.compose import PolicyInit, build_trainer
from cs2rl.train.config import build_train_config, compute_batch_dims

# The harness's four env-knob defaults are the dataclass's, read once rather
# than copied. Four literals here would be four more places #165 has to keep in
# step with env/config.py, and tests/integration/test_no_restated_env_defaults.py fails on
# exactly that shape — including the `: int = <literal>` spelling, which a
# regex written for `name = value` alone cannot see.
_ENV_DEFAULTS = EnvConfig()

# PuffeRL refuses total_timesteps < batch_size; four rounds of headroom let a test call
# evaluate() a few times back to back.
NUM_ROLLOUT_ROUNDS = 4


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
        Recorded as config ``seed``. It does not seed anything: the harness role
        builds env i with pufferlib's seed i whatever this is, and the policy's
        weights come from the global torch RNG, so call
        ``cs2rl.train.resume.seed_everything`` first for a reproducible trainer.
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
        build_trainer threads it into BOTH the envs (``env_config_from_args``)
        and the trainer (``participating_rows`` → ``trainer.participating``),
        exactly as for a CLI run. The default is EnvConfig's own field default
        — full N-vs-N, i.e. an all-ones participation mask — which is what every
        pre-Rung-0 harness caller gets. Named, never spelled: a literal here is
        a second declaration of the value, and it sits on a different line from
        the parameter so tests/integration/test_no_restated_env_defaults.py's line probe
        cannot see it go stale.
    map_data : MapData or None
        R0-E: the map every env is built on. None ⇒ ``make_simple_map()`` (the
        pre-R0-E hardcoded default). Pass the session ``simple_map`` fixture or
        an arena map. Flatness is never checked here (``resolve_pin_pitch`` runs
        in the CLI and train() only), so pin_pitch=1 on a non-flat map is allowed.
    pin_pitch, crouch_enabled : int
        R0-E.2 sim knobs. build_trainer threads pin_pitch into the envs, the
        policy (``build_policy(pin_pitch=)`` → aim_dim_mask) and the self-play
        manager, and runs ``assert_pin_pitch_agreement`` on the result;
        crouch_enabled reaches the envs only.
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
        Set on ``args``, so config ``opponent``, the participating-step budget,
        the participation rows and the self-play manager all come from it, through
        the same code as ``--opponent``.
        PITFALL: ``opponent="noop"`` with ``with_selfplay=True`` is refused by
        build_trainer, as the CLI refuses ``--opponent noop`` without
        ``--no-self-play`` — a past-policy opponent is not a statue, and
        maybe_switch_teams would flip the statue's team mid-run.
    self_play_mgr : SelfPlayManager or None
        A caller-built manager used INSTEAD of ``build_selfplay_manager``'s;
        build_trainer refuses one whose opponent_mode or pin_pitch disagrees
        with ``opponent`` / ``pin_pitch``. Its pool is read only inside
        evaluate(), so a test may seed it after construction. None (default)
        builds one from ``with_selfplay`` and the knobs above.

    Returns
    -------
    trainer : trainer.Cs2PuffeRL
        The production trainer class, built by ``build_trainer``
        (tests/train/test_trainer_composition.py pins the surface, and
        tests/train/test_train_harness_smoke.py the stock attributes).
    cleanup : Callable[[], None]
        Idempotent teardown: closes the trainer and its vecenv, removes the
        scratch checkpoint dir. Tests MUST call this in a ``finally:`` to avoid
        leaking fd's / shared memory / tmp dirs.

    Pitfalls
    --------
    - Re-using the same mp.Value across two harness instances in the same
      process is fine — ``shared_ts`` is a local to this call.
    - ``args`` is a ``types.SimpleNamespace``, not an argparse Namespace. It has
      the attributes build_train_config and build_trainer read; the rest fall
      back to their getattr defaults (field defaults for every other env knob).
      It has no ``map``, so config ``env`` is "cs2-dust2" whatever the map.
    - Serial backend means ``trainer.vecenv`` has a synchronous ``send``/``recv``
      cycle; tests can inject observations by monkey-patching those if needed.
    """
    # Function-local: cs2rl.env.map loads cs2rl.env.nav, which
    # tests/train/test_w1_modules.py::test_import_train_test_harness_stays_light forbids
    # at this module's scope.
    from cs2rl.env.map import make_simple_map

    tmp_checkpoint_dir = tempfile.mkdtemp(prefix="cs2rl-harness-")
    try:
        _, bptt_horizon, batch_size = compute_batch_dims(num_envs)
        # The env knobs reach the envs through env_config_from_args(args), the CLI's
        # resolver, so config.json and the envs read the same four values. Coverage of
        # that mapping, knob by knob: tests/env/test_env_factory.py drives the real
        # harness factory against tests/fixtures/env_config_pre_165b.json (whose two
        # harness rows differ in n_active_per_team and jump_enabled) and, off-fixture,
        # with a non-default crouch_enabled and pin_pitch
        # (test_harness_config_carries_the_knobs_no_fixture_row_varies);
        # tests/train/test_pitch_pin.py builds pin_pitch=1 trainers.
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
            num_envs=num_envs,
            self_play=with_selfplay,
            map_data=make_simple_map() if map_data is None else map_data,
            vec_backend="serial",
        )
        config = build_train_config(args, batch_size=batch_size, bptt_horizon=bptt_horizon)
        # build_train_config pins minibatch_size = max_minibatch_size = 8192, and PuffeRL
        # refuses batch_size < minibatch_size; batch_size = num_envs*640, so under 16 envs
        # no trainer could be built. Clamped HERE only: it changes total_minibatches and
        # accumulate_minibatches, so it must not move into build_train_config. CLI runs
        # still need --num_envs >= 16.
        config["minibatch_size"] = config["max_minibatch_size"] = min(8192, batch_size)
        trainer = build_trainer(args,
                                config,
                                shared_ts=mp.Value("f", 0.3),
                                env_role="harness",
                                policy_init=PolicyInit(tct_split_heads=tct_split_heads,
                                                       tct_split_trunk=tct_split_trunk),
                                self_play_mgr=self_play_mgr)
    except BaseException:
        # build_trainer releases what it acquired; the scratch dir is this module's.
        shutil.rmtree(tmp_checkpoint_dir, ignore_errors=True)
        raise

    def cleanup():
        """Idempotent teardown: close the trainer and its vecenv, remove the scratch dir.

        ``trainer.close()`` writes the full checkpoint (policy, optimizer, RNG, self-play
        pool) into ``config["data_dir"]``, the scratch dir the rmtree removes. Both closes
        are best-effort so a cleanup failure never masks a test's own assertion error;
        a second ``vecenv.close()`` is a no-op on Serial.
        """
        try:
            trainer.close()
        except Exception:              # noqa: BLE001 — best-effort cleanup
            pass
        try:
            trainer.vecenv.close()
        except Exception:              # noqa: BLE001
            pass
        shutil.rmtree(tmp_checkpoint_dir, ignore_errors=True)

    return trainer, cleanup
