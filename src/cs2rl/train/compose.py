"""The trainer's composition root: `build_trainer` assembles one `Cs2PuffeRL`.

`build_trainer(args, config, ...)` builds the shared memory, the vector env, the policy,
the participation rows, the self-play manager and the trainer, in the order they depend
on each other. `cs2rl.train.loop.train` calls it for a CLI run and
`tests._helpers.trainer_harness._build_trainer_for_test` calls it for a test trainer, so
both are built by this code. They differ only in what they pass:

  env_role="train"    the CLI. Per-env seeds `env_seed_base(--seed) + i`, and a
                      continuous-action shared array that Multiprocessing workers read.
  env_role="harness"  tests. `build_harness_env` envs (step stats in every info), the
                      seed pufferlib passes, and no continuous-action shared array (the
                      Serial per-env step wrapper carries the aim).

Everything about the RUN rather than the trainer stays in `train()`: W&B, metrics.jsonl,
the run id, seeding, reading and converting a resume checkpoint, the optimizer's weight
decay and aim-σ group, the full-state resume, the eval hook and the epoch loop. A
harness trainer therefore has PuffeRL's plain Adam: no weight decay, one param group.

`cs2rl.train.trainer` and `pufferlib.vector` are imported inside the functions that use
them: trainer.py imports torch at module scope, and `python -m cs2rl.train --dump-config`
imports this module (through `cs2rl.train.loop`) and must stay torch-free.
"""

import os
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from types import TracebackType
from typing import Literal, get_args

import numpy as np

from cs2rl.policy import AGENT_IDS, build_policy, load_state_dict_arch_checked
from cs2rl.spec.action import ACTION_MASK_DIM, AIM_DIM
from cs2rl.train.config import (
    assert_opponent_self_play_compatible,
    build_participating_rows,
    env_config_from_args,
    resolve_opponent_mode,
)
from cs2rl.train.envs import (
    assert_max_turn_speed_agreement,
    assert_pin_pitch_agreement,
    auto_vec_workers,
    build_train_env_factory,
    check_spawn_counts,
    env_seed_base,
)
from cs2rl.train.selfplay import SelfPlayManager, build_selfplay_manager

# pyrefly checks a literal role at each call site; build_trainer re-checks at runtime.
EnvRole = Literal["train", "harness"]
ENV_ROLES = get_args(EnvRole)


@dataclass(frozen=True)
class PolicyInit:
    """The policy architecture, and optionally the weights to load into it.

    ``state_dict`` must already match the architecture: `train()` resolves the split
    bits from the resume checkpoint and converts the dict before calling
    `build_trainer` (`resolve_resume_split` and the `convert_*` helpers in
    `cs2rl.train.resume`). ``source`` names the weights in errors and in the log.
    """
    tct_split_heads: bool = False
    tct_split_trunk: bool = False
    state_dict: dict | None = None
    source: str | None = None


@dataclass(frozen=True)
class _SharedMemory:
    """The trainer's shared arrays and their main-process numpy views.

    The action mask goes env -> trainer (envs write their masks, the trainer reads them
    after recv()). The continuous action goes trainer -> env (HybridAimVecEnv.send
    writes it, Multiprocessing workers read it). Both are allocated BEFORE
    pufferlib.vector.make, because workers fork there and only inherit memory that
    exists by then.
    """
    mask_shm: object
    mask_view: np.ndarray
    cont_shm: object | None = None
    cont_view: np.ndarray | None = None


def _close_on_exit(close: Callable[[], object], _exc_type: type[BaseException] | None,
                   error: BaseException | None, _traceback: TracebackType | None) -> bool:
    """Adapt a close callback to ExitStack without replacing an active failure.

    ExitStack still unwinds every registered owner. Its default policy would
    replace a training failure with a later close failure; keep the first error
    instead and attach subsequent failures as traceback notes. On a successful
    run the first cleanup failure propagates normally, including BaseException.
    """
    try:
        close()
    except BaseException as cleanup_error:
        if error is None:
            raise
        error.add_note(f"Training cleanup also failed: {cleanup_error!r}")
    return False


def _allocate_shared_memory(num_envs: int, *, cont_actions: bool) -> _SharedMemory:
    """RawArrays for ``num_envs`` envs of 10 agents; the continuous one only if asked."""
    from multiprocessing import RawArray

    rows = num_envs * len(AGENT_IDS)
    cont_shm = cont_view = None
    if cont_actions:
        cont_shm = RawArray("f", rows * AIM_DIM)
        cont_view = np.frombuffer(cont_shm, dtype=np.float32).reshape(rows, AIM_DIM)
    mask_shm = RawArray("b", rows * ACTION_MASK_DIM)
    mask_view = np.frombuffer(mask_shm, dtype=np.int8).reshape(rows, ACTION_MASK_DIM)
    return _SharedMemory(mask_shm, mask_view, cont_shm, cont_view)


def _per_env_kwargs(args, shm: _SharedMemory, *, route_seed: bool) -> list[dict]:
    """pufferlib's env_kwargs list: env i's shm index, and its seed under the train role.

    The seed rides here because pufferlib.vector.make takes ``seed`` as its own
    parameter and never forwards it to the backend (see build_env_factory). Everything
    else an env needs is closure state of the factory, never a per-env kwarg.
    """
    out = []
    for i in range(args.num_envs):
        kwargs = {"_cont_shm": shm.cont_shm, "_cont_idx": i, "_mask_shm": shm.mask_shm}
        if route_seed:
            kwargs["_seed"] = env_seed_base(args.seed) + i
        out.append(kwargs)
    return out


def _vec_backend(args):
    """(name, backend class, worker count, extra vector.make kwargs) for --vec-backend."""
    import pufferlib.vector

    name = args.vec_backend.lower()
    if name == "multiprocessing":
        import psutil

        physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        workers = args.vec_num_workers or auto_vec_workers(args.num_envs, physical_cores)
        return name, pufferlib.vector.Multiprocessing, workers, {
            "num_workers": workers,
            "batch_size": args.num_envs,
            "zero_copy": True,
            "overwork": args.vec_overwork,
        }
    if name == "serial":
        return name, pufferlib.vector.Serial, 1, {}
    raise ValueError(f"Unsupported vec backend: {args.vec_backend}")


def _make_vecenv(args, env_factory, per_env_kwargs):
    """pufferlib.vector.make over ``args.num_envs`` copies of one factory.

    PITFALL: the creators are a LIST of the same factory. With a single callable,
    pufferlib's broadcast step replaces the per-env kwargs list, so every env would
    get env 0's seed and shm index.
    """
    import pufferlib.vector

    name, backend, workers, vec_kwargs = _vec_backend(args)
    print(f"[Train] Creating {args.num_envs} vectorised envs "
          f"(backend={name}, workers={workers})...")
    return pufferlib.vector.make(
        [env_factory] * args.num_envs,
        env_args=[[] for _ in range(args.num_envs)],
        env_kwargs=per_env_kwargs,
        num_envs=args.num_envs,
        backend=backend,
        **vec_kwargs,
    )


def _build_policy(vecenv, args, policy_init: PolicyInit):
    """build_policy for this run's knobs, then the given weights if any.

    build_policy reads the driver env (observation size, max_turn_speed), so it runs
    after the vecenv exists; the weights load before the agreement checks in
    build_trainer, because a checkpoint may carry its own max_turn_speed buffer.
    """
    print(f"[Train] Building policy on device={args.device} "
          f"(tct_split_heads={policy_init.tct_split_heads}, "
          f"tct_split_trunk={policy_init.tct_split_trunk})...")
    policy = build_policy(vecenv,
                          args.device,
                          tct_split_heads=policy_init.tct_split_heads,
                          tct_split_trunk=policy_init.tct_split_trunk,
                          aim_log_std_max=getattr(args, "aim_log_std_max", None),
                          pin_pitch=bool(args.pin_pitch))
    if policy_init.state_dict is not None:
        load_state_dict_arch_checked(policy, policy_init.state_dict, source=policy_init.source)
        print(f"[Train] Resumed from checkpoint: {policy_init.source}")
    return policy


def _participating_rows(args, vecenv, opponent_mode: str) -> np.ndarray:
    """The static participation vector, checked against the env that was built.

    Under --opponent self it selects slots 0..n-1 of both teams, the slots the C env
    spawns; under noop or walker the hero team's only. The vector comes from args and the envs
    from the env factory, so the assert stops a disagreement that would otherwise
    train on the wrong rows without an error.
    """
    n_active = env_config_from_args(args).n_active_per_team
    rows = build_participating_rows(args.num_envs,
                                    n_active,
                                    opponent_mode=opponent_mode,
                                    hero_team=SelfPlayManager.initial_hero_team())
    assert vecenv.driver_env.n_active_per_team == n_active, "driver env / args disagree"
    return rows


def _selfplay_manager(args, opponent_mode: str, given):
    """``given`` after checking it agrees with args, else build_selfplay_manager's.

    A given manager's p_past IS the self-play flag, so the opponent guard is re-run on
    it, and its statue team and pitch mask must match the knobs the envs and policy
    were built with: evaluate() reads both from the manager. ``aim_log_std_max`` is
    not compared; the manager only forwards it when it loads a past policy.
    """
    if given is None:
        return build_selfplay_manager(self_play_enabled=bool(getattr(args, "self_play", True)),
                                      aim_log_std_max=getattr(args, "aim_log_std_max", None),
                                      pin_pitch=args.pin_pitch,
                                      opponent_mode=opponent_mode)
    assert_opponent_self_play_compatible(opponent_mode, given.p_past > 0)
    assert given.opponent_mode == opponent_mode and given.pin_pitch == bool(args.pin_pitch), (
        f"self_play_mgr disagrees with args: manager opponent_mode={given.opponent_mode!r} "
        f"pin_pitch={given.pin_pitch!r} vs opponent={opponent_mode!r} "
        f"pin_pitch={bool(args.pin_pitch)!r}")
    return given


def build_trainer(args,
                  config: dict,
                  *,
                  shared_ts,
                  env_role: EnvRole = "train",
                  policy_init: PolicyInit | None = None,
                  self_play_mgr=None):
    """Build the vector env, the policy and the `Cs2PuffeRL` trainer for one run.

    WHAT: returns the trainer. The caller owns it from then on: ``trainer.close()``
    (checkpointing) or ``trainer.close_resources()`` (no checkpoint) releases the
    vector env and PufferLib's Utilization thread. ``args`` is the CLI namespace or
    any object with the attributes read here and by `env_config_from_args`;
    ``config`` is `build_train_config`'s dict for the same args. ``shared_ts`` is the
    team-spirit Value every env reads. ``self_play_mgr`` replaces the built manager
    (tests that need a pre-seeded pool or p_past=1).

    WHY one function: a test trainer is then the CLI's assembly given test inputs,
    differing only as the module docstring lists. A second copy of the assembly would
    drift from this one unnoticed: its tests would pass while the CLI changed (#92).

    ORDER: the shared arrays exist before the vecenv (workers fork inside
    vector.make); the policy is built from the driver env and loaded before the
    pin-pitch and max-turn-speed agreement checks; the participation rows and the
    self-play manager exist before the constructor, which reads both. HybridAimVecEnv
    wraps the backend just before the constructor, as PuffeRL sees only the wrapper.

    CLEANUP: on any failure, everything acquired here is released before the error
    propagates, newest owner first: the trainer's ``close_resources`` once it exists
    (never ``close``, which would checkpoint a half-built run), else the wrapper's or
    the backend's ``close``. A failed wrapper or constructor leaves the previous
    owner registered.
    """
    from cs2rl.train.trainer import Cs2PuffeRL, HybridAimVecEnv

    if env_role not in ENV_ROLES:
        raise ValueError(f"env_role={env_role!r} must be one of {ENV_ROLES}")
    policy_init = policy_init or PolicyInit()
    opponent_mode = resolve_opponent_mode(args)
    # Before any env is built; a given manager is checked again on its own p_past.
    assert_opponent_self_play_compatible(opponent_mode, bool(getattr(args, "self_play", True)))
    shm = _allocate_shared_memory(args.num_envs, cont_actions=env_role == "train")
    env_factory = build_train_env_factory(args,
                                          shared_ts=shared_ts,
                                          map_data=args.map_data,
                                          role=env_role)
    per_env_kwargs = _per_env_kwargs(args, shm, route_seed=env_role == "train")
    with ExitStack() as owned:
        vecenv = _make_vecenv(args, env_factory, per_env_kwargs)
        owned.push(partial(_close_on_exit, vecenv.close))
        # `map` is absent on harness args: generic spawn bounds only.
        check_spawn_counts(vecenv, getattr(args, "map", None) or "")
        policy = _build_policy(vecenv, args, policy_init)
        participating_rows = _participating_rows(args, vecenv, opponent_mode)
        self_play_mgr = _selfplay_manager(args, opponent_mode, self_play_mgr)
        vecenv = HybridAimVecEnv(vecenv, shm.cont_view)
        owned.pop_all()
        owned.push(partial(_close_on_exit, vecenv.close))
        trainer = Cs2PuffeRL(config,
                             vecenv,
                             policy,
                             cont_action_view_main=shm.cont_view,
                             mask_view_main=shm.mask_view,
                             participating_rows=participating_rows,
                             self_play_mgr=self_play_mgr)
        owned.pop_all()
        owned.push(partial(_close_on_exit, trainer.close_resources))
        # Owners of the shared arrays for the trainer's lifetime: an unreferenced
        # RawArray's block goes back to multiprocessing's heap, and a later RawArray can
        # be given memory that Multiprocessing workers still use. The pin is a second
        # owner: the trainer holds a numpy view of each array (the wrapper too, of the
        # continuous one), whose base chain ends at the RawArray, and a Serial env keeps
        # its own reference. It still holds if a view is dropped (the equivalence tool's
        # no_mask_view case drops one).
        if shm.cont_shm is not None:
            trainer._cont_action_shm = shm.cont_shm
        trainer._action_mask_shm = shm.mask_shm
        # R0-E.2: env pitch flag == policy pitch mask; R0-G: env aim clamp == policy
        # tanh scale. Either disagreement stops the run before the first rollout.
        assert_pin_pitch_agreement(vecenv, policy)
        assert_max_turn_speed_agreement(vecenv, policy)
        owned.pop_all()
    return trainer
