#!/usr/bin/env python
"""Fingerprint seeded CPU trainer epochs: the bit-identity oracle for trainer refactors (#352).

WHAT: each CASE seeds python/numpy/torch, builds a small trainer with the production
builder (``cs2rl.train.compose.build_trainer``, 4 Serial envs, the test role), applies
the case's knobs, then runs a few ``evaluate()`` + ``train()`` epochs. It records digests
of every deterministic piece of trainer state: once right after construction and once
per epoch. ``run`` writes them to a JSON file; ``compare`` diffs two such files component
by component and exits 1 on any difference. ``seedctl`` is the positive control: the
same case under seed 0 and seed 1 must differ.

USAGE, for a change that claims zero behaviour change: run at the base commit, run at
HEAD, compare. From each checkout's root:

    env CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/src python scripts/trainer_equivalence.py run --out base.json
    env CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/src python scripts/trainer_equivalence.py run --out head.json
    env CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/src python scripts/trainer_equivalence.py compare base.json head.json

``run --cases default noop_n1`` runs a subset; ``seedctl default`` runs the control. A
base commit that predates this file: feed it on stdin from that checkout's root, so
cs2rl's import guard sees the script and the package in one checkout,
``python - run --out base.json < /path/to/trainer_equivalence.py``. The base must have
``cs2rl.train.compose`` (#92 part 2); older commits built their test trainers in
``tests/_helpers/trainer_harness.py`` only. All 26 cases ran in 91 s on the development VM
(2026-10-07).

WHAT IS FINGERPRINTED (``snapshot``):
  - every ``vars(trainer)`` value, minus EXCLUDED (wall clock, handles, and the objects
    fingerprinted separately below). Tensors by dtype/shape/bytes, floats by repr, dict
    and list order kept. Plus the sorted attribute names.
  - the policy, Adam, alpha-optimizer and LR-scheduler state; the train config (minus
    its scratch ``data_dir``); the driver env's EnvConfig.
  - the ordered ``losses`` dict, the evaluate()/train() ``stats``, train()'s return value
    (minus SPS, uptime and performance/*), and the "[Train]" / "[hybrid_aim" lines
    printed during the epoch.
  - the torch, numpy and python RNG states, and the full-state checkpoint files under
    ``data_dir/<run_id>`` (minus their wall-clock ``run_id`` entry).

CASES reach the update's branches: default; the noop statue and parked rows; past-policy
self-play (with and without the action-mask view); TAG; KL early stop; warm-start entropy
(grace/floor, ramp, hand-off, collapse watch, non-positive h0); the NaN guard (including a
NaN in the middle of an accumulation window, and a NaN entropy, whose alpha loss is NaN so
alpha skips that minibatch's step); empty minibatches; value clipping with prioritised event replay; the
checkpoint/done tail; CPU bf16 autocast (``fp32_*`` is its positive control: same case
without autocast, its digest must differ).

PITFALLS:
  - CPU only. Run with ``CUDA_VISIBLE_DEVICES=`` (empty): CUDA runs are seeded but not
    bit-exact, and the trainer captures CUDA RNG state whenever CUDA is visible (#307).
  - Both runs need the same torch thread count, so compare runs made on one machine with
    one thread environment. A different count changes float reduction order and so the
    digests, starting with the policy's initial weights. torch fixes the count at import
    from OMP_NUM_THREADS and MKL_NUM_THREADS; ``cs2rl.train`` is imported first, so each
    is 1 unless the environment sets it. Each case records the count, so a mismatch is
    an explained DIFFERENT (``construction:torch_threads`` among the components), never a
    false IDENTICAL, and ``compare`` prints a WARN line for it.
  - It only sees what its cases reach. The ``use_rnn=False`` reshape, the warm-start
    non-finite-entropy print and ndarray/list info values are not reached.
  - It builds what ``tests._helpers.trainer_harness._build_trainer_for_test`` builds
    (``env_role="harness"``): no weight decay, no separate aim-σ param group, no W&B,
    metrics file, eval hook or per-env ``_seed``. ``cs2rl.train.loop.train`` adds those
    for a CLI run; compare a short CLI run's outputs for that layer.
  - Not a pytest golden test: an intended numeric change moves the digests by design.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib
import io
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

# Before torch loads: cs2rl.train's __init__ sets the BLAS/OpenMP thread counts to 1 unless
# the environment already sets them, as it does for `python -m cs2rl.train`. torch fixes its
# thread count when it is imported, and a different count changes float reduction order, so
# two runs under different counts never match. Each case records the count (`torch_threads`).
# import_module rather than a bare import: an unused-import noqa trailing comment is
# realigned by yapf into a form ruff's import sorter (I001) rejects.
importlib.import_module("cs2rl.train")

# Attributes whose value is wall clock or a handle, or that are fingerprinted separately.
EXCLUDED = {
    "start_time", "last_log_time", "utilization", "profile", "logger", "vecenv", "amp_context",
    "policy", "uncompiled_policy", "optimizer", "scheduler", "_alpha_optimizer", "_self_play_mgr",
    "config", "_action_mask_shm", "_action_mask_view_main", "_cont_action_view_main",
    "_last_nan_warn_t", "stats", "losses"
}
WALL_CLOCK_LOG_KEYS = ("SPS", "uptime")


def _h(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:16]


def canon(v):
    """A JSON-able canonical form: exact for tensors (bytes) and floats (repr)."""
    import numpy as np
    import torch

    if torch.is_tensor(v):
        t = v.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            t = t.view(torch.int16)
        return ["T", str(v.dtype), list(v.shape), _h(t.numpy().tobytes())]
    if isinstance(v, np.ndarray):
        return ["A", str(v.dtype), list(v.shape), _h(np.ascontiguousarray(v).tobytes())]
    if isinstance(v, (bool, int, float, str, type(None), np.generic)):
        return [type(v).__name__, repr(v)]
    if isinstance(v, dict):
        return ["D", type(v).__name__, [[repr(k), canon(x)] for k, x in v.items()]]
    if isinstance(v, (list, tuple)):
        return ["L", type(v).__name__, [canon(x) for x in v]]
    if isinstance(v, slice):
        return ["S", repr(v)]
    if hasattr(v, "__dict__") and type(v).__name__ == "WelfordStd":
        return ["W", canon(vars(v))]
    return ["O", type(v).__name__]


def digest(obj) -> str:
    return _h(json.dumps(canon(obj), sort_keys=False).encode())


def _optim_canon(opt):
    sd = opt.state_dict()
    return {"state": sd["state"], "param_groups": sd["param_groups"]}


def construction(trainer) -> dict:
    """What construction alone decides, before any case knob or epoch runs."""
    import numpy as np
    import torch

    vec = trainer.vecenv
    return {
        "torch_threads":
        repr(torch.get_num_threads()),
        "attrs":
        digest({
            k: v
            for k, v in sorted(vars(trainer).items()) if k not in EXCLUDED
        }),
        "attr_names":
        digest(sorted(vars(trainer))),
        "config":
        digest({
            k: v
            for k, v in sorted(trainer.config.items()) if k != "data_dir"
        }),
        "env_config":
        digest(dataclasses.asdict(vec.driver_env.config)),
        "vecenv":
        repr((type(vec).__name__, type(vec._backend).__name__, vec._cont_action_view_main
              is None, trainer._cont_action_view_main is None, trainer._action_mask_view_main
              is None)),
        "policy":
        digest(dict(trainer.uncompiled_policy.state_dict())),
        "optimizer":
        digest(_optim_canon(trainer.optimizer)),
        "scheduler":
        digest(trainer.scheduler.state_dict()),
        "rng_torch":
        _h(torch.get_rng_state().numpy().tobytes()),
        "rng_numpy":
        digest(list(np.random.get_state())),
        "rng_python":
        _h(repr(random.getstate()).encode()),
    }


def snapshot(trainer, stats_after_eval) -> dict:
    """Every deterministic piece of trainer state after one epoch."""
    import numpy as np
    import torch

    comp = {}
    for name, val in sorted(vars(trainer).items()):
        if name not in EXCLUDED:
            comp[f"attr:{name}"] = digest(val)
    comp["attr_names"] = digest(sorted(vars(trainer)))
    # Whether the NaN guard has warned (the time itself is wall clock, so EXCLUDED). The
    # constructor declares 0.0; before gh#92 part 3 the attribute appeared at the first
    # warning, so this value equals that era's `in vars(trainer)` and old runs compare.
    comp["has_last_nan_warn_t"] = repr(trainer._last_nan_warn_t > 0.0)
    comp["policy"] = digest(dict(trainer.uncompiled_policy.state_dict()))
    comp["optimizer"] = digest(_optim_canon(trainer.optimizer))
    comp["alpha_optimizer"] = digest(_optim_canon(trainer._alpha_optimizer))
    comp["scheduler"] = digest(trainer.scheduler.state_dict())
    comp["config"] = digest({k: v for k, v in sorted(trainer.config.items()) if k != "data_dir"})
    comp["losses"] = digest(trainer.losses)
    comp["losses_type"] = type(trainer.losses).__name__
    comp["stats_after_eval"] = digest(stats_after_eval)
    comp["stats_after_train"] = digest(dict(trainer.stats))
    comp["rng_torch"] = _h(torch.get_rng_state().numpy().tobytes())
    comp["rng_numpy"] = digest(list(np.random.get_state()))
    comp["rng_python"] = _h(repr(random.getstate()).encode())
    mgr = trainer._self_play_mgr
    comp["selfplay_mgr"] = digest({k: v for k, v in vars(mgr).items() if k != "pool"})
    comp["selfplay_pool_len"] = repr(len(mgr.pool))
    comp["msg"] = repr(getattr(trainer, "msg", None))
    return comp


def readable_losses(trainer):
    """Ordered (key, type, repr): small and exact, for reading a DIFF."""
    return [[k, type(v).__name__, repr(v)] for k, v in trainer.losses.items()]


# ── fault injection at the loss boundary the trainer module resolves ────────────────
def _patch_loss(mode):
    """Replace trainer._hybrid_ppo_loss; returns the restore callable.

    nan_odd: calls 1, 3, 5, ... return a NaN loss. nan_second_of_three: calls 2, 5, 8, ...
    neg_entropy: the entropy term is replaced by -1. nan_entropy_first: call 1 returns a
    NaN entropy, so that minibatch's alpha loss is NaN too.
    """
    from cs2rl.train import trainer as trainer_module
    real = trainer_module._hybrid_ppo_loss
    calls = {"n": 0}

    def patched(*args, **kwargs):
        out = real(*args, **kwargs)
        calls["n"] += 1
        if mode == "nan_odd" and calls["n"] % 2 == 1:
            return (out[0] * float("nan"), *out[1:])
        if mode == "nan_second_of_three" and calls["n"] % 3 == 2:
            return (out[0] * float("nan"), *out[1:])
        if mode == "neg_entropy":
            return (out[0], out[1] * 0.0 - 1.0, *out[2:])
        if mode == "nan_entropy_first" and calls["n"] == 1:
            return (out[0], out[1] * float("nan"), *out[2:])
        return out

    trainer_module._hybrid_ppo_loss = patched
    return lambda: setattr(trainer_module, "_hybrid_ppo_loss", real)


def _seed_pool(trainer):
    """Put a snapshot of the live policy in the self-play pool (a p_past=1 case plays it)."""
    import torch

    snap = Path(tempfile.mkdtemp(prefix="trainer-fingerprint-pool-")) / "sp_000000.pt"
    torch.save(trainer.policy.state_dict(), snap)
    trainer._self_play_mgr._add_to_pool(snap)


def build(seed,
          num_envs=4,
          mgr_p_past=None,
          no_mask_view=False,
          opponent="self",
          n_active_per_team=None,
          with_selfplay=False):
    """Seed, then build with the production builder; returns (trainer, cleanup).

    The inputs are the test harness's: the env knobs at their EnvConfig defaults (read,
    never restated), a scratch checkpoint dir, four rounds of timesteps and the
    minibatch clamped to the batch. ``mgr_p_past`` passes a caller-built manager, as a
    test does to force the past-policy branch.
    """
    import multiprocessing as mp
    import shutil
    import types

    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    from cs2rl.train.compose import build_trainer
    from cs2rl.train.config import build_train_config, compute_batch_dims
    from cs2rl.train.resume import seed_everything
    from cs2rl.train.selfplay import build_selfplay_manager

    defaults = EnvConfig()
    seed_everything(seed)
    mgr = None
    if mgr_p_past is not None:
        # build_selfplay_manager derives p_past from the self-play flag (0.3 or 0.0), and
        # scripts/ may not call SelfPlayManager(...) itself
        # (tests/integration/test_env_construction_enforcement.py), so the case's p_past
        # is set on the built manager.
        mgr = build_selfplay_manager(self_play_enabled=True,
                                     aim_log_std_max=None,
                                     pin_pitch=defaults.pin_pitch,
                                     opponent_mode="self")
        mgr.p_past = mgr_p_past
    checkpoint_dir = tempfile.mkdtemp(prefix="trainer-equivalence-")
    try:
        _, bptt_horizon, batch_size = compute_batch_dims(num_envs)
        args = types.SimpleNamespace(
            device="cpu",
            seed=seed,
            timesteps=batch_size * 4,
            checkpoint_dir=checkpoint_dir,
            n_active_per_team=(defaults.n_active_per_team
                               if n_active_per_team is None else n_active_per_team),
            pin_pitch=defaults.pin_pitch,
            crouch_enabled=defaults.crouch_enabled,
            jump_enabled=defaults.jump_enabled,
            aim_log_std_max=None,
            aim_entropy_bonus=True,
            opponent=opponent,
            num_envs=num_envs,
            self_play=with_selfplay,
            map_data=make_simple_map(),
            vec_backend="serial",
        )
        config = build_train_config(args, batch_size=batch_size, bptt_horizon=bptt_horizon)
        config["minibatch_size"] = config["max_minibatch_size"] = min(8192, batch_size)
        trainer = build_trainer(args,
                                config,
                                shared_ts=mp.Value("f", 0.3),
                                env_role="harness",
                                self_play_mgr=mgr)
    except BaseException:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        raise
    if no_mask_view:
        # Equivalent to constructing with mask_view_main=None: construction only stores
        # the view, and evaluate() reads it.
        trainer._action_mask_view_main = None

    def cleanup():
        """Close the trainer (it writes its checkpoint) and remove the scratch dir."""
        try:
            trainer.close()
        finally:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    return trainer, cleanup


# ── case hooks: hook(trainer, epoch, phase), phase "built" or "after_eval" ──────────
def _cfg(**c):

    def hook(trainer, epoch, phase):
        if phase == "built":
            trainer.config.update(c)

    return hook


def _attrs(**a):

    def hook(trainer, epoch, phase):
        if phase == "built":
            for k, v in a.items():
                setattr(trainer, k, v)

    return hook


def _after_eval(fn):

    def hook(trainer, epoch, phase):
        if phase == "after_eval":
            fn(trainer, epoch)

    return hook


def _chain(*hooks):

    def hook(trainer, epoch, phase):
        for h in hooks:
            h(trainer, epoch, phase)

    return hook


def _cpu_bf16_autocast(trainer, epoch, phase):
    import torch

    if phase == "built":
        trainer.amp_context = torch.amp.autocast("cpu", dtype=torch.bfloat16)


def _keep_participating(step):

    def fn(trainer, epoch):
        trainer.participating.zero_()
        trainer.participating[::step] = True

    return fn


def _force_events(trainer, epoch):
    trainer._event_mask[::3] = True


def _collapse_watch(trainer, epoch):
    if epoch == 1:
        trainer._warmstart_h0 = 1e6


_WS_GRACE = dict(warmstart_entropy=True, warmstart_alpha_ceiling=0.0, warmstart_grace_steps=10**12)
_TAG = dict(tag_diagnostic=True, tag_every=1, target_kl=None)
# name -> build kwargs, epoch count and optional hook/flags; run_case reads each entry
# through _CaseSpec, so a key it does not read is a TypeError, not a no-op.
CASES: dict[str, dict[str, Any]] = {
    "default":
    dict(build={}, epochs=2),
    "noop_n1":
    dict(build=dict(opponent="noop", n_active_per_team=1), epochs=2),
    "selfplay_past":
    dict(build=dict(mgr_p_past=1.0), epochs=2, seed_pool=True),
    "selfplay_past_no_mask_view":
    dict(build=dict(mgr_p_past=1.0, no_mask_view=True), epochs=2, seed_pool=True),
    "tag_every2_nokl":
    dict(build={}, epochs=3, hook=_cfg(**{
        **_TAG, "tag_every": 2
    })),
    "tag_klstop_7mb":
    dict(build={},
         epochs=2,
         hook=_chain(_cfg(**{
             **_TAG, "target_kl": -1.0
         }), _attrs(total_minibatches=7, minibatch_segments=20))),
    "kl_trip_7mb":
    dict(build={},
         epochs=2,
         hook=_chain(_cfg(target_kl=1e-4), _attrs(total_minibatches=7, minibatch_segments=20))),
    "ws_grace_floor":
    dict(build={},
         epochs=2,
         hook=_chain(_cfg(**_WS_GRACE, warmstart_ramp_steps=10_000_000),
                     _attrs(_entropy_floor=1e6))),
    "ws_ramp":
    dict(build={},
         epochs=3,
         hook=_cfg(warmstart_entropy=True,
                   warmstart_alpha_ceiling=0.0,
                   warmstart_grace_steps=0,
                   warmstart_ramp_steps=10**12)),
    "ws_off_handoff":
    dict(build={},
         epochs=3,
         hook=_cfg(warmstart_entropy=True, warmstart_grace_steps=0, warmstart_ramp_steps=1)),
    "ws_collapse_watch":
    dict(build={}, epochs=2, hook=_chain(_cfg(**_WS_GRACE), _after_eval(_collapse_watch))),
    "ws_h0_nonpositive":
    dict(build={},
         epochs=2,
         loss_patch="neg_entropy",
         hook=_cfg(warmstart_entropy=True, warmstart_grace_steps=10**12)),
    "nan_guard_tag":
    dict(build={},
         epochs=2,
         loss_patch="nan_odd",
         hook=_chain(_cfg(**_TAG), _attrs(total_minibatches=4))),
    "empty_all":
    dict(build={}, epochs=2, hook=_after_eval(lambda trainer, e: trainer.participating.zero_())),
    "empty_some":
    dict(build=dict(opponent="noop", n_active_per_team=1),
         epochs=2,
         hook=_chain(_cfg(prio_alpha=0.6, target_kl=None),
                     _attrs(total_minibatches=6, minibatch_segments=2),
                     _after_eval(_keep_participating(7)))),
    "vf_clip_prio_events_accum2_offload":
    dict(build=dict(with_selfplay=True),
         epochs=2,
         hook=_chain(_cfg(vf_clip_coef=0.2, prio_alpha=0.6, cpu_offload=True, target_kl=None),
                     _attrs(accumulate_minibatches=2, total_minibatches=4),
                     _after_eval(_force_events))),
    "throttled_ckpt_to_done":
    dict(build={}, epochs=5, throttled=True, hook=_cfg(checkpoint_interval=2)),
                                                                                                    # The skip's zero_grad is visible only with a gradient pending from an earlier
                                                                                                    # minibatch of the same accumulation window.
    "nan_guard_accum2":
    dict(build={},
         epochs=2,
         loss_patch="nan_odd",
         hook=_chain(_cfg(target_kl=None), _attrs(accumulate_minibatches=2, total_minibatches=4))),
    "nan_mid_window_accum3":
    dict(build={},
         epochs=2,
         loss_patch="nan_second_of_three",
         hook=_chain(_cfg(target_kl=None), _attrs(accumulate_minibatches=3, total_minibatches=6))),
    "nan_entropy_alpha_skip":
    dict(build={},
         epochs=2,
         loss_patch="nan_entropy_first",
         hook=_chain(_cfg(target_kl=None), _attrs(total_minibatches=3))),
                                                                                                    # TAG on a past-policy epoch: tag/selfplay_active reads _selfplay_used_past.
    "tag_selfplay_past":
    dict(build=dict(mgr_p_past=1.0), epochs=2, seed_pool=True, hook=_cfg(**_TAG)),
    "kl_trip_empty_some":
    dict(build=dict(opponent="noop", n_active_per_team=1),
         epochs=2,
         hook=_chain(_cfg(prio_alpha=0.6, target_kl=1e-4),
                     _attrs(total_minibatches=7, minibatch_segments=2),
                     _after_eval(_keep_participating(5)))),
    "noop_tag_ws_ramp":
    dict(build=dict(opponent="noop", n_active_per_team=2),
         epochs=3,
         hook=_cfg(tag_diagnostic=True,
                   tag_every=1,
                   warmstart_entropy=True,
                   warmstart_grace_steps=0,
                   warmstart_ramp_steps=10**12)),
    "events_partial_participation":
    dict(build=dict(opponent="noop", n_active_per_team=1),
         epochs=2,
         hook=_chain(_cfg(prio_alpha=0.6), _after_eval(_force_events))),
    "bf16_selfplay_past_tag":
    dict(build=dict(mgr_p_past=1.0),
         epochs=2,
         seed_pool=True,
         hook=_chain(_cpu_bf16_autocast, _cfg(**_TAG))),
    "fp32_selfplay_past_tag":
    dict(build=dict(mgr_p_past=1.0), epochs=2, seed_pool=True, hook=_cfg(**_TAG)),
}


def _ckpt_canon(trainer):
    import torch

    d = Path(trainer.config["data_dir"]) / trainer.logger.run_id
    if not d.exists():
        return []
    out = []
    for p in sorted(d.iterdir()):
        content = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(content, dict):  # the logger's wall-clock id
            content = {k: v for k, v in content.items() if k != "run_id"}
        out.append([p.name, digest(content)])
    return out


def _filter_logs(logs):
    if logs is None:
        return None
    return {
        k: v
        for k, v in logs.items()
        if k not in WALL_CLOCK_LOG_KEYS and not k.startswith("performance/")
    }


def _warnings(text):
    return [ln for ln in text.splitlines() if ln.startswith(("[Train]", "[hybrid_aim"))]


@dataclasses.dataclass(frozen=True)
class _CaseSpec:
    """One CASES entry, as run_case reads it.

    WHY a dataclass: its generated __init__ raises TypeError on a keyword it does not
    declare, so a misspelt key (``throtled=True``) fails the run instead of being
    ignored while the case still compares IDENTICAL. Every field is read by run_case.
    KNOWN LIMIT: no test pins the field defaults; a changed default applies to every
    CASES entry that omits that key.
    """
    build: dict[str, Any]
    epochs: int
    hook: Any = None
    loss_patch: str | None = None
    seed_pool: bool = False
    throttled: bool = False


def run_case(name, seed=0):
    """Build, run the case's epochs; returns {"construction": ..., "epochs": [...]}."""
    spec = _CaseSpec(**CASES[name])
    restore = _patch_loss(spec.loss_patch) if spec.loss_patch else None
    try:
        trainer, cleanup = build(seed, **spec.build)
        try:
            built = construction(trainer)
            hook = spec.hook or (lambda trainer, e, p: None)
            hook(trainer, -1, "built")
            if spec.seed_pool:
                _seed_pool(trainer)
            epochs = []
            for epoch in range(spec.epochs):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    stats = trainer.evaluate()
                    stats_copy = {k: list(v) for k, v in stats.items()}
                    hook(trainer, epoch, "after_eval")
                    trainer.last_log_time = 1e18 if spec.throttled else 0.0
                    logs = trainer.train()
                comp = snapshot(trainer, stats_copy)
                comp["train_return"] = digest(_filter_logs(logs))
                comp["train_return_keys"] = digest(sorted(logs) if logs else None)
                comp["warnings"] = digest(_warnings(buf.getvalue()))
                comp["checkpoints"] = digest(_ckpt_canon(trainer))
                losses = trainer.losses or {}
                epochs.append({
                    "components":
                    comp,
                    "losses":
                    readable_losses(trainer),
                    "warnings":
                    _warnings(buf.getvalue()),
                    "facts": [losses.get("minibatches_run"),
                              losses.get("empty_minibatches")],
                })
            return {"construction": built, "epochs": epochs}
        finally:
            cleanup()
    finally:
        if restore:
            restore()


def case_digest(result):
    return _h(
        json.dumps([result["construction"]] + [e["components"] for e in result["epochs"]],
                   sort_keys=True).encode())


def main_run(out, names):
    unknown = sorted(set(names) - set(CASES))
    if unknown:
        raise SystemExit(f"unknown case(s) {unknown}; known: {sorted(CASES)}")
    from cs2rl.train import trainer as trainer_module
    print(f"TRAINER_MODULE {trainer_module.__file__}", flush=True)
    result = {}
    for name in names or list(CASES):
        result[name] = run_case(name)
        facts = " | ".join(f"run={e['facts'][0]} empty={e['facts'][1]} warn={len(e['warnings'])}"
                           for e in result[name]["epochs"])
        print(
            f"CASE {name} digest={case_digest(result[name])} "
            f"epochs={len(result[name]['epochs'])} FACTS {facts}",
            flush=True)
    Path(out).write_text(json.dumps(result, indent=1))
    print(f"WROTE {out} cases={len(result)}")


def main_compare(a_path, b_path):
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    print(f"INPUT cases_a={len(a)} cases_b={len(b)}")
    threads = [
        sorted({str(r["construction"].get("torch_threads"))
                for r in d.values()}) for d in (a, b)
    ]
    if threads[0] != threads[1]:
        print(f"WARN torch_threads a={threads[0]} b={threads[1]}: the digests are expected to "
              "differ; rerun both sides with one thread environment")
    bad = 0
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            print(f"CASE {name} MISSING in {'a' if name not in a else 'b'}")
            bad += 1
            continue
        ra, rb = a[name], b[name]
        diffs = [
            f"construction:{k}" for k in sorted(set(ra["construction"]) | set(rb["construction"]))
            if ra["construction"].get(k) != rb["construction"].get(k)
        ]
        ea, eb = ra["epochs"], rb["epochs"]
        if len(ea) != len(eb):
            diffs.append(f"epochs {len(ea)} vs {len(eb)}")
        for i, (x, y) in enumerate(zip(ea, eb, strict=False)):
            diffs += [
                f"epoch{i}:{k}" for k in sorted(set(x["components"]) | set(y["components"]))
                if x["components"].get(k) != y["components"].get(k)
            ]
            diffs += [f"epoch{i}:{k}_readable" for k in ("losses", "warnings") if x[k] != y[k]]
        ncomp = len(ra["construction"]) + sum(len(e["components"]) for e in ea)
        if diffs:
            bad += 1
            print(f"CASE {name} DIFF ({len(diffs)}): {', '.join(diffs[:14])}")
        else:
            print(f"CASE {name} IDENTICAL components_compared={ncomp} digest={case_digest(ra)}")
    print("RESULT", "IDENTICAL" if bad == 0 else f"DIFFERENT cases={bad}")
    return 0 if bad == 0 else 1


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("run with CUDA_VISIBLE_DEVICES= (empty): CPU runs are the bit-exact "
                         "ones, and a visible GPU changes the checkpointed RNG state (#307)")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run cases and write their digests to a JSON file")
    p_run.add_argument("--out", required=True, help="the JSON file to write")
    p_run.add_argument("--cases",
                       nargs="+",
                       default=[],
                       metavar="CASE",
                       help=f"default: all {len(CASES)} cases ({', '.join(CASES)})")
    p_cmp = sub.add_parser("compare", help="diff two run outputs; exit 1 on any difference")
    p_cmp.add_argument("a")
    p_cmp.add_argument("b")
    p_seed = sub.add_parser("seedctl", help="positive control: seed 0 and seed 1 must differ")
    p_seed.add_argument("case", choices=sorted(CASES))
    args = ap.parse_args()
    if args.cmd == "run":
        main_run(args.out, args.cases)
    elif args.cmd == "compare":
        sys.exit(main_compare(args.a, args.b))
    else:
        a, b = case_digest(run_case(args.case, seed=0)), case_digest(run_case(args.case, seed=1))
        print(f"SEEDCTL {args.case} seed0={a} seed1={b} DIFFERS={a != b}")
        sys.exit(0 if a != b else 1)


if __name__ == "__main__":
    main()
