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
    - return-norm & timing patches: skipped. Tests can apply them explicitly
      if they need to exercise those patches.

Design contract:
    - Public API is a single ``_build_trainer_for_test(...)`` function that
      returns ``(trainer, cleanup)``. Tuple rather than contextmanager because
      (a) the smoke-test shape in the task spec uses try/finally, and
      (b) downstream tests may need to leak the trainer between helper
      functions — easier with an explicit cleanup callable.
    - ``with_selfplay`` defaults to False (Task 6c needs True; Tasks 7-11
      default False to keep the attribute surface small).

If PufferLib ever renames ``PuffeRL`` or drops a constructor kwarg, this file
is where the breakage will first surface — giving us a single ~line-diff review
surface for API drift instead of chasing it across every Batch-1 test.
"""

from __future__ import annotations

import multiprocessing as mp
import shutil
import tempfile
import types


def _build_trainer_for_test(
    num_envs: int = 32,
    with_selfplay: bool = False,
    device: str = "cpu",
    seed: int = 0,
    tct_split_heads: bool = False,
    tct_split_trunk: bool = False,
):
    """Build a tiny in-process PuffeRL trainer for Batch-1 trainer-level tests.

    Parameters
    ----------
    num_envs : int
        Vectorised-env count. 32 is enough to populate the rollout buffer in
        one evaluate() round on the Serial backend in under ~5s; bump only if
        a test specifically needs more parallelism (cost is roughly linear).
    with_selfplay : bool
        If True, apply ``_patch_trainer_with_selfplay`` so ``trainer.evaluate``
        is the self-play variant. The SelfPlayManager is constructed with an
        EMPTY pool, so ``should_use_past()`` always returns False — the past-
        policy branch is NOT exercised, but the patched evaluate() wrapper is.
        Required by Task 6c (reward-clamp removal test).
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

    Returns
    -------
    trainer : pufferlib.pufferl.PuffeRL
        Fully-constructed trainer with all production attributes the Batch-1
        downstream tests assert against (see tests/test_train_harness_smoke.py
        for the pinned surface).
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
    # Imports are function-local so importing this module in a test that
    # doesn't actually call the factory (e.g. a smoke import test) is free.
    # PuffeRL is lazy-imported inside train.train() in production; do the same
    # here so this module is importable even if pufferlib's optional torch deps
    # are mid-install in an isolated test runner.
    import pufferlib.vector
    from pufferlib.pufferl import PuffeRL

    from map import make_simple_map
    from train import (
        SelfPlayManager,
        _patch_trainer_with_hybrid_aim,
        _patch_trainer_with_selfplay,
        build_policy,
        build_train_config,
        compute_batch_dims,
        make_puffer_env,
    )

    # ── Scratch dir for config.json + any checkpoints PuffeRL writes ────────
    # PuffeRL's constructor doesn't actually write to data_dir during __init__,
    # but build_train_config requires it as a string. Using mkdtemp keeps each
    # harness instance isolated; cleanup removes it wholesale.
    tmp_checkpoint_dir = tempfile.mkdtemp(prefix="cs2rl-harness-")

    # ── Shared team-spirit value (production pattern) ───────────────────────
    # Production uses mp.Value so multiprocess workers can read a scalar that
    # the main trainer anneals each epoch. Serial backend doesn't actually
    # need shared memory, but make_puffer_env expects this interface.
    shared_ts = mp.Value("f", 0.3)

    # Simple 5-room map — the production default for non-dust2 runs. Avoids
    # depending on any pre-generated mapdata file on disk.
    map_data = make_simple_map()

    # ── F8: action-mask shm, same env→trainer pattern as production ─────────
    # Serial backend runs envs in-process, but the RawArray pattern is kept
    # identical to train.train() so harness-built trainers exercise the REAL
    # masked rollout path (sampler + rollout buffer + PPO-loss mask).
    from multiprocessing import RawArray

    import numpy as np

    from _action_spec import ACTION_MASK_DIM

    _agents_per_env = 10
    mask_shm = RawArray("b", num_envs * _agents_per_env * ACTION_MASK_DIM)
    mask_view_main = np.frombuffer(mask_shm, dtype=np.int8).reshape(num_envs * _agents_per_env,
                                                                    ACTION_MASK_DIM)

    def env_factory(*_args, buf=None, seed=None, _mask_idx=None, **_kwargs):
        # Mirrors the closure in train.train() lines ~1367-1368. The seed
        # forwarded by pufferlib.vector can be None for the first reset; use
        # explicit `is None` check so a legitimate seed=0 is preserved rather
        # than silently falsy-remapped.
        #
        # include_step_stats_in_info: always True so the harness-built trainer
        # has a uniform attribute/info surface across selfplay and no-selfplay
        # modes (Task 6c consumes it; Tasks 7-11 ignore it). Cost is a single
        # pre-built singleton dict per env; no per-tick allocation.
        env = make_puffer_env(
            team_spirit=shared_ts,
            buf=buf,
            seed=0 if seed is None else seed,
            map_data=map_data,
            include_step_stats_in_info=True,
        )
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

    # ── Minimal argparse-shaped config object ───────────────────────────────
    # build_train_config reads these four attributes. Everything else in the
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
    )
    train_config = build_train_config(args, batch_size=batch_size, bptt_horizon=bptt_horizon)

    policy = build_policy(vecenv,
                          device,
                          tct_split_heads=tct_split_heads,
                          tct_split_trunk=tct_split_trunk)
    trainer = PuffeRL(train_config, vecenv, policy)

    # Batch 3 (T5): the hybrid-aim patcher is REQUIRED for any test that
    # exercises evaluate() / train() because those code paths now read
    # trainer.cont_actions / trainer.logprobs_{d,c}. Apply unconditionally
    # so the harness shape matches production. _patch_trainer_with_return_norm
    # is intentionally NOT applied here — harness tests that need it apply
    # it explicitly (matches the pre-Batch-3 contract documented at module
    # docstring "return-norm & timing patches: skipped").
    # F8: mask_view_main plumbed so harness rollouts run MASKED, same as
    # production. Pin the RawArray on the trainer against GC (prod pattern).
    trainer._action_mask_shm = mask_shm
    _patch_trainer_with_hybrid_aim(trainer, mask_view_main=mask_view_main)

    # ── Self-play patch (always applied at T5) ──────────────────────────────
    # Pre-Batch-3: this was gated on `with_selfplay` so the no-selfplay path
    # could exercise PufferLib's library evaluate(). T4 changed the policy
    # contract to a 4-tuple; PufferLib's library evaluate still expects a
    # 2-tuple, so the no-selfplay path can't run end-to-end without our
    # hybrid-aware evaluate wrapper. The selfplay patch IS that wrapper;
    # `with_selfplay=False` now means "no past-policy mixing" (empty pool
    # never activates) — the patch wrapper still runs. The
    # `with_selfplay=True` path additionally pre-seeds the manager. This
    # keeps the test harness honest with production where the hybrid-aim
    # rollout requires the patched evaluate path.
    if not with_selfplay:
        self_play_mgr = SelfPlayManager(
            pool_size=15,
            p_past=0.0,
            save_every_epochs=25,
            win_threshold=0.6,
            phase_length=50,
        )
        _patch_trainer_with_selfplay(trainer, self_play_mgr)
    else:
        self_play_mgr = SelfPlayManager(
            pool_size=15,
            p_past=0.3,
            save_every_epochs=25,
            win_threshold=0.6,
            phase_length=50,
        )
        _patch_trainer_with_selfplay(trainer, self_play_mgr)

    def cleanup():
        """Idempotent teardown. Safe to call twice.

        Both .close() calls are wrapped in try/except so a cleanup failure
        never masks a test assertion error. If PufferLib drops trainer.close()
        in a future release, this silently no-ops — the smoke test pins
        ``close`` in the attribute surface so the rename will be caught there.
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
