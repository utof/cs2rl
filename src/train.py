#!/usr/bin/env python
"""CS2 RL Sim — training entry point.

Usage:
  python src/train.py --smoke       # sanity check: 20k native-env steps, no crash, print steps/sec
  python src/train.py --train       # full PPO self-play training (PufferLib 3.0)
  python src/train.py --record      # run 1 episode, save rerun recording (random policy)
  python src/train.py --eval        # evaluate a checkpoint across many seeds
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import random
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np

# _action_spec is generated from cs2_types.h. Notes on the names imported below:
#   ACTION_MASK_DIM - F8: trainer-side mask buffer width (= sum of head sizes).
#   AIM_DIM         - T4→T5 carry-forward (M-1): unused in this module, imported so the
#                     T6 ONNX exporter can pull it from `train` (see __all__ below).
# PITFALL (gh#97): no trailing comments anywhere in this import block. yapf snaps
# trailing comments to its spaces_before_comment stops (40/56/72) while ruff's isort
# wants exactly one space, so any comment in here makes the two formatters fight and
# ruff reports I001 forever. Second gh#97 trap: a suppression directive must END its
# line — trailing prose after its code list makes the directive malformed and inert
# (and ruff then parses the prose as rule codes, warning on every invocation).
from _action_spec import (
    ACTION_HEAD_NAMES,
    ACTION_HEAD_SIZES,
    ACTION_MASK_DIM,
    AIM_DIM,
)
from env_config import EnvConfig, RewardWeights
from env_factory import build_env_for, build_selfplay_manager
from paths import CHECKPOINTS_DIR, RECORDINGS_DIR
from resume_state import (
    _install_full_checkpointing,
    _rng_load_state_dict,
    _rng_state_dict,
    check_checkpoint_set,
    check_resume_config,
    check_resume_metrics_bound,
    collect_train_state,
    load_full_resume,
    resolve_resume_run,
    restore_train_state,
    seed_everything,
)
from train_config import (
    OPPONENT_MODES,
    assert_opponent_self_play_compatible,
    build_participating_rows,
    build_train_config,
    compute_batch_dims,
    env_config_from_args,
    resolve_opponent_mode,
    validate_aim_log_std_max,
)
from train_metrics import (
    ScheduledEval,
    _inject_tag_metrics,
    compute_game_metrics,
    compute_head_divergence,
    compute_network_health,
    compute_trunk_divergence,
    log_aim_log_std,
)
from train_shared import (
    _LOG_2PI,
    _MASK_HEAD_SLICES,
    _R0G_KNOBS,
    _WARMSTART_ATTRS,
    AIM_LOG_STD_CAP_MIN_HEADROOM,
    AIM_LOG_STD_INIT_MARGIN,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_GAMMA,
    LOG_STD_INIT,
    LOG_STD_MAX,
    LOG_STD_MIN,
    RESUME_CONFIG_ALLOWLIST,
    TEAM_SIZE,
    _apply_action_masks,
    _atomic_save_state_dict,
    pin_pitch_for_map,
    resolve_aim_log_std_init,
    resolve_gammas,
)
from train_update import (
    _aim_dim_weight,
    _hybrid_ppo_loss,
    _patch_trainer_with_return_norm,
    _scheduled_target_entropy,
    _tag_param_groups,
    masked_explained_variance,
    masked_mean,
    masked_normalize_adv,
    masked_std_unbiased,
    tag_grad_cossim,
)

# Explicit re-export marker: naming a symbol here is what tells ruff that an
# otherwise-unused import is intentional. A trailing per-line F401 suppression cannot
# be used instead, for the formatter reason above. Nothing does `from train import *`,
# so this tuple has no star-import effect on any caller: it is a lint marker AND the
# inventory of what `train` re-exports.
#
# The five `from train_shared/resume_state/train_config/train_metrics/train_update
# import` blocks above are SHIMS (post-rung1a refactor, 2026-08-31): those symbols are
# now DEFINED in the split-out modules and re-exported here so the existing
# `from train import X` call sites in tests/, scripts/ and src/ keep working without
# being rewritten (rewriting them is a separate, mechanical branch — spec §4).
# A SHIM IS NOT A PATCH POINT: `monkeypatch.setattr(train, "<name>", ...)` on any name
# below rebinds only train's own reference. Call sites inside the defining module
# resolve through THAT module's globals and never see the patch, so such a test passes
# while asserting nothing. Patch the defining module (see tests/test_tag_trainer.py).
# PLACEMENT IS LOAD-BEARING, not an isort accident: train.py's own module body READS
# moved names while it executes. SEVEN such reads, measured 2026-09-04, and ALL of
# them now sit inside `if __name__ == "__main__":` — three argparse defaults and
# choices (DEFAULT_CHECKPOINT_INTERVAL, OPPONENT_MODES, DEFAULT_GAMMA) and four
# calls (validate_aim_log_std_max, assert_opponent_self_play_compatible,
# compute_batch_dims, build_train_config). WHAT THE NUMBER COUNTS, so the next
# reader can re-derive it rather than trust it: `ast.Name` loads of a
# shim-imported name reachable from a module-level statement, counting `def`
# default expressions (they DO execute at import) and not function bodies (they
# do not). #165 PR B2 deleted the one read outside the __main__ block —
# make_puffer_env's `n_active_per_team=TEAM_SIZE` default — so this comment no
# longer cites it.
# Every one of those sits BELOW this block, so moving the shims below the
# __main__ block breaks every real launch — and ONLY a real launch, now that no
# read is left outside it. Measured by doing it: `import train` still succeeds,
# while `python src/train.py --help` dies in its own argparse setup with
# `NameError: name 'DEFAULT_CHECKPOINT_INTERVAL' is not defined`. The SUITE DOES
# catch that, so this comment is not standing in for a missing test: 25 tests go
# red under exactly that move (measured 2026-09-04, whole suite, -p no:randomly),
# and every one of the 22 functions behind them launches this file in a child
# interpreter — 14 of the 15 in tests/test_train_cli.py, the rest in
# tests/test_arena_duel.py, tests/test_resume_state.py, tests/test_run_rung1_sh.py,
# tests/test_seed_reproducible.py and tests/test_w1_modules.py. The 15th
# test_train_cli case stays green because it is a source scan — the
# "an import-only test cannot see this" point in miniature. What the comment adds
# is not coverage but a NAME: all 25 report a NameError inside argparse, which
# tells you the symbol and not the rule.
__all__ = (
    "AIM_DIM",
    "AIM_LOG_STD_CAP_MIN_HEADROOM",
    "AIM_LOG_STD_INIT_MARGIN",
    "DEFAULT_CHECKPOINT_INTERVAL",
    "DEFAULT_GAMMA",
    "LOG_STD_INIT",
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "OPPONENT_MODES",
    "RESUME_CONFIG_ALLOWLIST",
    "ScheduledEval",
    "TEAM_SIZE",
    "_LOG_2PI",
    "_MASK_HEAD_SLICES",
    "_R0G_KNOBS",
    "_WARMSTART_ATTRS",
    "_aim_dim_weight",
    "_apply_action_masks",
    "_atomic_save_state_dict",
    "_hybrid_ppo_loss",
    "_inject_tag_metrics",
    "_install_full_checkpointing",
    "_patch_trainer_with_return_norm",
    "_rng_load_state_dict",
    "_rng_state_dict",
    "_scheduled_target_entropy",
    "_tag_param_groups",
    "assert_opponent_self_play_compatible",
    "build_participating_rows",
    "build_train_config",
    "check_checkpoint_set",
    "check_resume_config",
    "check_resume_metrics_bound",
    "collect_train_state",
    "compute_batch_dims",
    "compute_game_metrics",
    "compute_head_divergence",
    "compute_network_health",
    "compute_trunk_divergence",
    "env_config_from_args",
    "load_full_resume",
    "log_aim_log_std",
    "masked_explained_variance",
    "masked_mean",
    "masked_normalize_adv",
    "masked_std_unbiased",
    "pin_pitch_for_map",
    "resolve_aim_log_std_init",
    "resolve_gammas",
    "resolve_opponent_mode",
    "resolve_resume_run",
    "restore_train_state",
    "seed_everything",
    "tag_grad_cossim",
    "validate_aim_log_std_max",
)

# MUST stay a bare integer literal: scripts/exp_lib.py fingerprints the env by
# regex-grepping `OBS_DIM = <int>` out of this file's source text (env_fingerprint),
# so it cannot be an `import`. Mirrors nav.OBS_DIM / _obs_spec.OBS_DIM (generated
# from cs2_types.h); cross-checked by test_obs_dim_constant_consistency (test_train_env.py).
# On an OBS_DIM bump, update cs2_types.h + rerun the generator, then bump this literal.
OBS_DIM = 110


def isolate_aim_log_std_param_group(trainer, weight_decay: float = 0.0):
    """Give every ``aim_log_std*`` parameter its own decay-free optimizer group.

    WHAT: removes the σ parameters from whatever group Adam put them in and
    re-adds them as a new param group that clones group 0's hyper-parameters
    except ``weight_decay`` (0.0 by default). Returns the number of parameters
    moved (0 if the policy has none — e.g. a stub policy in a unit test).

    WHY (spec 2026-08-30 T1, load-bearing for the Rung 1a gate): the run sets
    ``param_groups[0]["weight_decay"] = 1e-4`` and Adam's decay adds ``wd·θ``
    to the gradient. ``aim_log_std`` is always NEGATIVE (σ < 1), so decay pushes
    it UPWARD even at exactly zero true gradient — measured on smoke-v1c/s0,
    ``health/weight_norm_aim_log_std`` 3.2564 → 2.87122 over 30 epochs (raw
    per-dim movement 0.272) while σ was provably gradient-dead the whole run.
    Left in, decay would (a) fake the gate's "σ moved ≥ 0.1 ⇒ the head receives
    gradient" signal and (b) eat the −0.2 init margin within ~11 epochs and
    re-freeze σ against the cap. Weight decay on a log-scale noise parameter is
    meaningless anyway: it is not a capacity knob, it is a distribution shape.

    PITFALLS:
      * ``scheduler.base_lrs`` MUST grow with the group. PufferLib builds
        CosineAnnealingLR from the one-group optimizer (pufferl.py:171), and
        ``LRScheduler.step()`` zips ``param_groups`` against the values derived
        from ``base_lrs`` NON-strictly — a missing entry means the σ group's LR
        silently never anneals. Handled here; ``restore_train_state`` zips the
        same two lists with strict=True, so a mismatch would also fail loudly
        on resume.
      * Call AFTER the weight_decay=1e-4 line and BEFORE load_full_resume.
        Resuming a PRE-branch checkpoint (whose optimizer state has one group)
        into the two-group optimizer raises in torch's own load_state_dict —
        loud, and accepted: every pre-branch run is complete (spec §2 T3).
      * Parameters are matched by NAME (same regex as
        reinit_frozen_aim_log_std) so the split-heads copies aim_log_std_t /
        aim_log_std_ct are covered too, then by identity when pruning the old
        groups — ``add_param_group`` raises if a parameter ends up in two.
    """
    import re as _re

    # uncompiled_policy is the raw module; a torch.compile wrapper would prefix
    # names with "_orig_mod." and break the name match (the parameter OBJECTS
    # are shared, so the identity prune below is unaffected either way).
    policy = getattr(trainer, "uncompiled_policy", None)
    if policy is None:
        policy = trainer.policy
    sigma_params = [
        p for name, p in policy.named_parameters() if _re.search(r"aim_log_std(_t|_ct)?$", name)
    ]
    if not sigma_params:
        return 0
    sigma_ids = {id(p) for p in sigma_params}
    opt = trainer.optimizer
    # Idempotent: a second call would strand an empty group and push base_lrs
    # out of step with param_groups for good.
    for group in opt.param_groups:
        if {id(p) for p in group["params"]} == sigma_ids:
            group["weight_decay"] = weight_decay
            return len(sigma_params)
    for group in opt.param_groups:
        group["params"] = [p for p in group["params"] if id(p) not in sigma_ids]
    # Clone group 0's hypers (lr, betas, eps, and the initial_lr the scheduler
    # stamped on) so the σ group anneals on the same schedule; only the decay
    # differs.
    new_group = {k: v for k, v in opt.param_groups[0].items() if k != "params"}
    new_group["params"] = sigma_params
    new_group["weight_decay"] = weight_decay
    opt.add_param_group(new_group)
    sch = getattr(trainer, "scheduler", None)
    if sch is not None and hasattr(sch, "base_lrs"):
        sch.base_lrs.append(float(new_group.get("initial_lr", new_group["lr"])))
        sch._last_lr = [g["lr"] for g in opt.param_groups]
    return len(sigma_params)


# gh#91: σ to widen a BC-frozen aim head to at PPO resume. BC detaches
# aim_log_std (spec D-6) so bc_warmstart.pt carries σ=0.1 while fitting
# obs-dependent |μ| up to ~0.63 rad — one lr=3e-4 Adam step then moves μ a
# full σ and continuous approx_kl (~1.4) blows past target_kl (0.03),
# throttling every update to ~1 minibatch. σ=0.3 drops that per-step KL ~9×
# while staying inside [σ_min, σ_max]. Applied by reinit_frozen_aim_log_std.
AIM_LOG_STD_RESUME_INIT = math.log(0.3)


def reinit_frozen_aim_log_std(state_dict, *, atol=1e-6, cap=None):
    """gh#91: widen a BC-frozen aim head before PPO resumes from it.

    WHAT: if ``state_dict`` carries an ``aim_log_std`` tensor still sitting
    exactly at LOG_STD_INIT (every element, within ``atol``), overwrite it
    in-place with AIM_LOG_STD_RESUME_INIT (σ 0.1 → 0.3) and return True.
    Any other value — i.e. a checkpoint whose aim head actually trained —
    is left untouched (returns False).

    WHY: BC detaches aim_log_std (spec D-6), so bc_warmstart.pt pairs a
    near-deterministic σ=0.1 with large obs-dependent aim means. Resuming
    PPO from that puts one Adam step a full σ away → continuous approx_kl
    ~1.4 ≫ target_kl 0.03 → the KL early-stop throttles updates to ~1
    minibatch/epoch for ~85 epochs (root-caused 2026-08-01, run
    checkpoints-20260801-022606; companion metrics bug gh#90).

    PITFALLS:
      * Detection is by VALUE, not filename — any un-trained aim_log_std is
        the BC signature (an RL run moves it within its first updates). A
        trained checkpoint landing back on exactly log(0.1) elementwise is
        measure-zero.
      * Mutates ``state_dict`` (pre-``load_state_dict``), matching dtype/
        device of the stored tensor via full_like.
      * Matches any key ENDING in "aim_log_std" so a future wrapper prefix
        (e.g. "policy.aim_log_std") keeps working.
      * Batch 7: the matcher covers aim_log_std, aim_log_std_t and
        aim_log_std_ct. The legacy→split warm conversion must still call this
        FIRST, on the legacy dict — see convert_legacy_state_dict_to_split.
      * R0-E.3 (#131): ``cap`` is the run's --aim-log-std-max. The fill value
        is min(AIM_LOG_STD_RESUME_INIT, cap) — widening to log(0.3) under a
        log(0.05) cap would be clamped away in every forward anyway, but the
        stored parameter would sit outside the band and the σ gradient would
        be dead (clamp has zero gradient outside its range). None = no cap.
    """
    import re as _re

    import torch as _torch

    changed = False
    for key, val in state_dict.items():
        # Batch 7 (spec §3.3): also match the split copies. The bare
        # endswith("aim_log_std") this replaces is FALSE for "aim_log_std_t"
        # and "aim_log_std_ct" — belt-and-braces so a future split-format
        # warmstart is widened by VALUE too. It does NOT relieve the caller of
        # the ordering rule: on a legacy→split resume this helper must run on
        # the LEGACY dict, before convert_legacy_state_dict_to_split.
        if _re.search(r"aim_log_std(_t|_ct)?$", key) and _torch.allclose(
                val, _torch.full_like(val, LOG_STD_INIT), atol=atol):
            fill = AIM_LOG_STD_RESUME_INIT if cap is None else min(AIM_LOG_STD_RESUME_INIT,
                                                                   float(cap))
            state_dict[key] = _torch.full_like(val, fill)
            changed = True
    return changed


def state_dict_is_split(state_dict):
    """True if this checkpoint was written by a T/CT split policy (spec §3.3).

    WHAT: presence of the `aim_log_std_t` parameter is the marker — it exists
    in exactly one architecture and nowhere else in the key space.

    WHY key inference rather than the config flag: config.json is rewritten
    unconditionally on every launch (the try-wrapped `write_text` inside
    `train()`, src/train.py:3522 as of 2026-08-31 — NOT the `--dump-config`
    early-exit write, which is conditional), so a flag-less
    crash-resume would stamp `tct_split_heads: false` over a split run's
    provenance. Deciding from the keys means resume, self-play snapshot
    loading and the eval/record loader all do the right thing with no flag at
    all — the flag governs only fresh construction and the legacy→split
    conversion direction.

    PITFALL: "aim_log_std_ct".endswith("aim_log_std_t") is False, so this does
    not accidentally fire on a CT-only key set; it is nonetheless deliberate
    that the marker is the T copy, since both are always written together.
    """
    return any(k.endswith("aim_log_std_t") for k in state_dict)


def state_dict_is_trunk_split(state_dict):
    """True if encoder/LSTM were written as per-team copies (spec §3.3).

    WHAT: presence of `encoder_t.0.weight` is the trunk-split marker — the T
    encoder first-layer weight exists in exactly that architecture. Both
    `encoder_t`/`encoder_ct` (and both LSTMs) are always written together.

    WHY key inference rather than the config flag: same as
    `state_dict_is_split` — `config.json` is rewritten on every launch, so a
    flag-less crash-resume must recover trunk-ness from the keys. The heads
    helper stays the heads marker; loaders consult both bits independently.

    PITFALL: `"encoder_ct.0.weight".endswith("encoder_t.0.weight")` is False,
    so a CT-only key set does not fire. The `k ==` clause is the bare-key
    form every checkpoint this project writes; `endswith` covers a future
    wrapper prefix (e.g. `policy.encoder_t.0.weight`).
    """
    return any(k == "encoder_t.0.weight" or k.endswith("encoder_t.0.weight") for k in state_dict)


def convert_legacy_state_dict_to_split(state_dict):
    """Warm split: duplicate a legacy checkpoint's heads into both team copies.

    WHAT: returns a NEW dict where `action_heads.*` → `action_heads_t.*` +
    `action_heads_ct.*`, `aim_mu.*` → `aim_mu_t.*` + `aim_mu_ct.*`,
    `aim_log_std` → `aim_log_std_t` + `aim_log_std_ct`. Everything else
    (encoder, lstm, value_head) passes through untouched — the trunk and the
    critic stay shared.

    WHY duplicate rather than re-initialize one side: the BC warmstart heads
    encode "how to act at all". Starting CT from random heads would confound
    the experiment with a relearning phase. Warm split means both teams start
    IDENTICAL and the divergence itself is the treatment.

    PITFALLS:
      * ORDER (spec §3.3, the gh#91 trap): call reinit_frozen_aim_log_std on
        the LEGACY dict BEFORE this function. The un-widened matcher used to
        miss the `_t`/`_ct` keys entirely; running the re-init afterwards
        would silently leave σ=0.1 and throttle every PPO update of the run
        through the KL early-stop, with no error and no log line. The matcher
        is now widened as belt-and-braces, but the ordering is still the
        contract — pinned by
        test_warm_split_duplicates_heads_and_reinits_sigma_in_both_copies.
      * Keys are matched strictly (no wrapper prefix like "policy."). Every
        checkpoint this project writes is bare-keyed; a prefixed dict would
        pass through unconverted and then fail loudly at
        load_state_dict_arch_checked rather than half-loading.
      * Tensors are cloned so the two copies never alias — an in-place
        optimizer step on one would otherwise move the other.
    """
    out = {}
    for key, val in state_dict.items():
        if key.startswith("action_heads."):
            suffix = key[len("action_heads."):]
            out[f"action_heads_t.{suffix}"] = val.clone()
            out[f"action_heads_ct.{suffix}"] = val.clone()
        elif key.startswith("aim_mu."):
            suffix = key[len("aim_mu."):]
            out[f"aim_mu_t.{suffix}"] = val.clone()
            out[f"aim_mu_ct.{suffix}"] = val.clone()
        elif key == "aim_log_std":
            out["aim_log_std_t"] = val.clone()
            out["aim_log_std_ct"] = val.clone()
        else:
            out[key] = val
    return out


def convert_shared_trunk_to_split(state_dict):
    """Warm split: duplicate a shared encoder+LSTM into both team copies.

    WHAT: returns a NEW dict where `encoder.*` → `encoder_t.*` +
    `encoder_ct.*` and `lstm.*` → `lstm_t.*` + `lstm_ct.*`. Everything else
    (`aim_log_std`, already-split `encoder_t.*`/`lstm_t.*`, `value_head`,
    action heads) passes through untouched — this convert is the trunk axis
    only.

    WHY duplicate rather than re-initialize one side: same as the heads
    warm split. The BC/legacy trunk encodes "how to see at all"; starting
    CT from a random encoder+LSTM would confound the experiment with a
    relearning phase. Both teams start IDENTICAL and the divergence is the
    treatment.

    PITFALLS:
      * ORDER (spec §3.3): `reinit_frozen_aim_log_std` on the LEGACY dict
        FIRST, then `convert_legacy_state_dict_to_split` (needs bare
        `aim_log_std`), THEN this function. This helper does not touch σ
        keys, but running it first is still wrong if a later heads convert
        is expected to see `encoder.*`/`lstm.*` or bare `aim_log_std`.
      * Keys are matched strictly (`encoder.` / `lstm.` prefixes, no
        wrapper). Already-split `encoder_t.*` / `lstm_t.*` do not match
        those prefixes and pass through — a second convert is a no-op on
        a trunk-split dict.
      * Tensors are cloned so the two copies never alias — an in-place
        optimizer step on one would otherwise move the other.
    """
    out = {}
    for key, val in state_dict.items():
        if key.startswith("encoder."):
            suf = key[len("encoder."):]
            out[f"encoder_t.{suf}"] = val.clone()
            out[f"encoder_ct.{suf}"] = val.clone()
        elif key.startswith("lstm."):
            suf = key[len("lstm."):]
            out[f"lstm_t.{suf}"] = val.clone()
            out[f"lstm_ct.{suf}"] = val.clone()
        else:
            # aim_log_std, value_head, heads, already-split encoder_t/lstm_t.
            out[key] = val
    return out


def load_state_dict_arch_checked(policy, state_dict, *, source):
    """load_state_dict with a loud architecture-mismatch error (spec §3.3).

    WHAT: compares the checkpoint's two architecture bits (heads via
    `state_dict_is_split`, trunk via `state_dict_is_trunk_split`) against
    the policy's (`policy.tct_split_heads`, `policy.tct_split_trunk`) and
    raises a message naming BOTH axes before loading anything. Never a
    silent partial load.

    WHY it exists even though every construction site infers: the sites that
    RECEIVE a pre-built policy and then load into it (train main's resume,
    load_policy_from_checkpoint, SelfPlayManager.load_past_policy) can be
    handed a mismatched pair by a caller that bypassed inference. A bare
    load_state_dict there raises a wall of missing/unexpected keys that names
    neither architecture — the operator's first hypothesis becomes "corrupt
    checkpoint", which is wrong and expensive.

    PITFALL: this does NOT convert. Legacy→split conversion is a deliberate
    act with a σ-re-init → heads convert → trunk convert ordering
    constraint, so it stays at the one call site that means it (the
    train-main warm split). Policies built before the trunk attr exists
    compare as trunk-off via getattr(..., False).
    """
    ckpt_heads = state_dict_is_split(state_dict)
    ckpt_trunk = state_dict_is_trunk_split(state_dict)
    # Today's policies have no tct_split_trunk attr; treat missing as off.
    policy_heads = bool(getattr(policy, "tct_split_heads", False))
    policy_trunk = bool(getattr(policy, "tct_split_trunk", False))
    if ckpt_heads != policy_heads or ckpt_trunk != policy_trunk:

        def _name_heads(flag):
            return "SPLIT (per-team T/CT policy heads)" if flag else "LEGACY (shared policy heads)"

        def _name_trunk(flag):
            return ("SPLIT (per-team T/CT encoder+LSTM)"
                    if flag else "LEGACY (shared encoder+LSTM)")

        raise ValueError(
            f"policy/checkpoint architecture mismatch loading {source}: the checkpoint is "
            f"heads={_name_heads(ckpt_heads)}, trunk={_name_trunk(ckpt_trunk)} but the policy is "
            f"heads={_name_heads(policy_heads)}, trunk={_name_trunk(policy_trunk)}. Rebuild the "
            f"policy with build_policy(..., tct_split_heads={ckpt_heads}, "
            f"tct_split_trunk={ckpt_trunk}) — loaders are supposed to infer both axes from the "
            f"checkpoint keys (state_dict_is_split / state_dict_is_trunk_split), see spec "
            f"2026-08-15 §3.3.")
    policy.load_state_dict(state_dict)


def resolve_resume_split(resume_path, *, heads_flag, trunk_flag, map_location="cpu"):
    """Decide heads- and trunk-split-ness BEFORE build_policy, from resume.

    Returns ``(heads_split, trunk_split, state_dict_or_None, Path_or_None)``.

    WHY this shape (spec §3.3 ordering constraint): in train() the policy is
    constructed before train_config is built and before the resume
    checkpoint is otherwise read — at construction time neither flag's
    config key nor the checkpoint's keys are in scope. So the checkpoint
    is sniffed once here, both decisions are passed into build_policy, and
    the already-loaded dict is handed back for reuse at the load site
    (no double I/O on a 2.5 MB file, and no chance of the two reads
    disagreeing).

    Decision (each bit independently; omitted flags never narrow):
      no resume            → (bool(heads_flag), bool(trunk_flag), None, None)
      ckpt bit off + flag  → flag WIDENS that axis (warm split at load)
      ckpt bit on + flag   → inference WINS (True regardless of flag)

    Each flag can only WIDEN its axis 0→1; it can never narrow a split
    checkpoint back to a shared policy. That asymmetry is what makes a
    flag-less crash-resume of a split run correct — the routine reality on
    this box, not an edge case.

    PITFALL: loads with map_location="cpu" regardless of the training device.
    load_state_dict copies into the policy's own (possibly CUDA) tensors, so
    this is safe and avoids allocating a second copy on the GPU during
    startup. Do not keep a 3-tuple / singular ``flag=`` shim — a leftover
    ``flag=`` call would TypeError, which is the intended tripwire.
    """
    import torch as _torch

    if not resume_path:
        return bool(heads_flag), bool(trunk_flag), None, None
    resume_path = Path(resume_path)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    state_dict = _torch.load(resume_path, map_location=map_location, weights_only=True)
    # Omitted flags never narrow: flag can only OR the ckpt bit to True.
    heads = bool(heads_flag) or state_dict_is_split(state_dict)
    trunk = bool(trunk_flag) or state_dict_is_trunk_split(state_dict)
    return heads, trunk, state_dict, resume_path


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


def resolve_run_name(name: str) -> str:
    """Return a run name prefixed with DDMMYY-N- where N is the count of existing
    checkpoint dirs that already start with today's date prefix."""
    from datetime import date

    today = date.today()
    date_prefix = today.strftime("%d%m%y")                             # e.g. "200326"
    checkpoints_dir = CHECKPOINTS_DIR
    count = 0
    if checkpoints_dir.exists():
        prefix = date_prefix + "-"
        count = sum(1 for d in checkpoints_dir.iterdir()
                    if d.is_dir() and d.name.startswith(prefix))
    return f"{date_prefix}-{count}-{name}"


AGENT_IDS = tuple([f"t{i}" for i in range(5)] + [f"ct{i}" for i in range(5)])

# ── SECTION: Smoke Test ────────────────────────────────────────────────────


def smoke_test():
    print("[Smoke] Initialising environment...")
    # W3 (#154): the construction seed now lives in env_factory.SMOKE_SEED. The
    # reset() seed below is deliberately NOT routed through the factory — it is
    # this function's own episode seed, not part of the env's construction, and
    # the two happening to be 42 is a coincidence the factory must not encode.
    env = build_env_for("smoke")
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
        print("[Smoke] Throughput gate lives in: uv run pytest tests/smoke_test.py -q -s")
    finally:
        env.close()


# ── SECTION: Shared eval / record helpers ──────────────────────────────────


def load_policy_from_checkpoint(checkpoint_path, device, aim_log_std_max=None, pin_pitch=False):
    """Rebuild a policy from a bare state_dict checkpoint (eval / record / probe).

    R0-E (#131): ``aim_log_std_max`` / ``pin_pitch`` are RUN properties, not
    checkpoint state (non-persistent buffers on the policy), so the caller
    must pass the run's values — a checkpoint cannot tell you whether its
    run pinned pitch. Defaults reproduce the pre-R0-E policy.
    """
    import torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Policy] Loading checkpoint -> {checkpoint_path}")
    # weights_only=True: every checkpoint this project writes is a bare tensor
    # state_dict, and the other loaders (resume sniff, self-play pool) already
    # load with it — a checkpoint that fails here is untrusted or corrupt.
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Infer obs_dim from checkpoint to handle checkpoints trained with different obs sizes.
    # Trunk-split checkpoints have no shared encoder — the T copy is the marker
    # (same key state_dict_is_trunk_split uses). Both copies share obs_dim.
    if "encoder_t.0.weight" in state_dict:
        ckpt_obs_dim = state_dict["encoder_t.0.weight"].shape[1]
    else:
        ckpt_obs_dim = state_dict["encoder.0.weight"].shape[1]
    # W3 (#154): role eval_legacy — the DOCUMENTED bare-call defaults, which are
    # now EnvConfig()'s own field defaults: the eval_legacy builder passes a
    # bare EnvConfig(), and env_config.py declares those fields to be the
    # trained baseline, which is exactly the pre-Rung-0 env (full 5v5, pitch
    # live, crouch and jump enabled); test_defaults_equal_the_139a3a3_values in
    # tests/test_env_config.py pins them. Passing no knobs is the behaviour, not
    # an oversight; #143 tracks whether it should change, and the factory reduces
    # that future fix to one role's knob source.
    policy_env = build_env_for("eval_legacy")

    # Batch 3.5 (#24, Opus I3): defensive obs_dim consistency check.
    # The function rebuilds the policy with the *checkpoint's* obs_dim
    # (obs_dim_override=ckpt_obs_dim). For any cross-version checkpoint
    # (e.g., Batch-3 105-dim loaded against Batch-3.5 107-dim env), the
    # policy will silently mis-interpret obs slots after the insertion
    # point. Fail loud at load time instead.
    # Pitfall: compare DISK shape to LIVE shape (ckpt_obs_dim vs
    # env_obs_dim), not derived-to-derived (policy.obs_dim is set to
    # obs_dim_override and would be self-referentially equal).
    env_obs_dim = policy_env.single_observation_space.shape[0]
    if ckpt_obs_dim != env_obs_dim:
        policy_env.close()
        raise ValueError(
            f"checkpoint obs_dim={ckpt_obs_dim} ≠ env obs_dim={env_obs_dim}; "
            f"checkpoint is from a different obs schema. Retrain or use a matching env.")

    try:
        # Batch 7 / trunk split (spec §3.3): both architecture bits inferred
        # from the checkpoint keys, exactly like obs_dim above — this loader
        # gets no flag and needs none. A trunk-split file has no encoder.0.weight.
        policy = build_policy(policy_env,
                              device,
                              obs_dim_override=ckpt_obs_dim,
                              tct_split_heads=state_dict_is_split(state_dict),
                              tct_split_trunk=state_dict_is_trunk_split(state_dict),
                              aim_log_std_max=aim_log_std_max,
                              pin_pitch=pin_pitch)
    finally:
        policy_env.close()

    load_state_dict_arch_checked(policy, state_dict, source=str(checkpoint_path))
    policy.eval()
    return policy


def init_policy_state(policy, device):
    import torch

    if policy is None:
        return None

    return {
        "done": torch.zeros(len(AGENT_IDS), device=device),
        "lstm_h": torch.zeros(len(AGENT_IDS), policy.hidden_size, device=device),
        "lstm_c": torch.zeros(len(AGENT_IDS), policy.hidden_size, device=device),
    }


def init_obs_buffer():
    return {aid: np.zeros((OBS_DIM, ), dtype=np.float32) for aid in AGENT_IDS}


def update_obs_buffer(obs_buffer, obs, terms=None, truncs=None):
    for aid, ob in obs.items():
        obs_buffer[aid] = ob

    for aid in AGENT_IDS:
        if aid not in obs:
            obs_buffer[aid].fill(0.0)


def select_policy_actions(policy, obs_buffer, active_agents, device, policy_state, policy_mode):
    import torch

    if policy_mode == "random":
        raise ValueError("Random action selection should bypass select_policy_actions")

    obs_arr = np.stack([obs_buffer[aid] for aid in AGENT_IDS])
    obs_t = torch.as_tensor(obs_arr, device=device)

    with torch.no_grad():
        # Batch 3 (T5): policy now emits 4-tuple (logits, mu_aim, log_std, value).
        # This helper is eval/inspection only, and nothing calls it today —
        # record_episode uses select_policy_actions_native. It does not return the
        # continuous (Δyaw) component, so the sampled cont_t is dropped on the
        # floor. The cont_action is still SAMPLED (sample mode) so the policy
        # state advances identically to training; we just don't emit it. If a
        # future eval path needs Δyaw, return (act_dict, cont_dict) — keeping
        # the int-action signature for now (no caller exists to adapt today).
        logits, mu_aim, log_std_aim, _ = policy.forward_eval(obs_t, policy_state)
        if policy_mode == "sample":
            # Fix #1: 6-tuple return; only need action + cont (logp/entropy unused here).
            act_t, _cont_t, *_ = _hybrid_sample_logits(
                (logits, mu_aim, log_std_aim, None),
                max_turn_speed=policy.max_turn_speed.item(),
            )
        else:
            # Greedy: argmax discrete + μ-only continuous (no exploration).
            # Greedy callers care about deterministic playback, so the σ noise
            # would actively hurt — μ_aim is the policy's best guess.
            act_t = torch.stack([head.argmax(dim=-1) for head in logits], dim=-1)

    act_np = act_t.cpu().numpy().astype(np.int64)
    return {aid: act_np[i] for i, aid in enumerate(AGENT_IDS) if aid in active_agents}


def select_policy_actions_native(policy, obs, device, policy_state, policy_mode):
    """Run policy in eval mode; return BOTH discrete and continuous actions.

    Returns
    -------
    (actions, cont) : (np.int32 (N, ACTION_DIM), np.float32 (N, AIM_DIM))
        actions : per-head argmax (greedy) or per-head sample (sample mode).
        cont    : continuous aim head output. Greedy uses μ directly (already
                  bounded by tanh*max_turn_speed); sample draws from
                  Normal(μ, exp(log_std)) clamped to ±max_turn_speed (matches
                  the rollout sampler's behaviour exactly).

    Why both: env.step now requires (actions, continuous_actions). Earlier
    (Batch 3) this helper returned discrete-only and the cont buffer was
    dropped silently — recording/eval paths effectively passed cont=zeros
    every tick, freezing aim at spawn (the "agents look forward in rerun"
    bug). Returning both lets callers feed the env exactly the actions the
    policy produced.
    """
    import torch

    if policy_mode == "random":
        raise ValueError("Random action selection should bypass select_policy_actions_native")

    obs_t = torch.as_tensor(obs, device=device)
    if hasattr(policy, "obs_dim") and obs_t.shape[-1] != policy.obs_dim:
        obs_t = obs_t[..., :policy.obs_dim]
    with torch.no_grad():
        logits, mu_aim, log_std_aim, _ = policy.forward_eval(obs_t, policy_state)
        if policy_mode == "sample":
            # 6-tuple return; we keep action + cont (logp/entropy unused here).
            act_t, cont_t, *_ = _hybrid_sample_logits(
                (logits, mu_aim, log_std_aim, None),
                max_turn_speed=policy.max_turn_speed.item(),
            )
        else:
            # Greedy: per-head argmax for discrete, μ directly for continuous.
            # μ is already tanh-squashed × max_turn_speed in HybridPolicy.forward
            # (the `torch.tanh(self.aim_mu...)` lines in build_policy's nested
            # class), so it's already bounded — no extra clamp needed.
            act_t = torch.stack([head.argmax(dim=-1) for head in logits], dim=-1)
            cont_t = mu_aim

    return (act_t.cpu().numpy().astype(np.int32), cont_t.cpu().numpy().astype(np.float32))


def resolve_policy_mode(checkpoint_path, policy_mode):
    if checkpoint_path and policy_mode != "random":
        return "greedy" if policy_mode == "auto" else policy_mode
    return "random"


def extract_env_info(infos):
    if isinstance(infos, list):
        for info in infos:
            if info:
                return info
        return {}
    for aid in AGENT_IDS:
        info = infos.get(aid)
        if info:
            return info
    return {}


def format_histogram_line(label, counts):
    total = int(np.sum(counts))
    if total <= 0:
        return f"{label}: []"

    parts = []
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        pct = 100.0 * float(count) / total
        parts.append(f"{idx}={count} ({pct:.1f}%)")
    return f"{label}: [{', '.join(parts)}]"


def format_train_status(epoch, ts_val, logs):
    sps = logs.get("SPS", 0.0)
    timeout = logs.get("environment/timed_out", 0.0)
    t_win = logs.get("environment/winner_t", 0.0)
    ct_win = logs.get("environment/winner_ct", 0.0)
    plant = logs.get("environment/bomb_planted", 0.0)
    kills_t = logs.get("environment/kills_t", 0.0)
    kills_ct = logs.get("environment/kills_ct", 0.0)
    round_len = logs.get("environment/round_length", 0.0)
    move_1 = logs.get("environment/action_move_1", 0.0)
    # Batch 3.5: per-axis aim log_std (clamped). Defaults to 0.0 if missing.
    # T7 acceptance gate 2 greps for aim_log_std_pitch= — keep this substring.
    aim_log_std_yaw = logs.get("policy/aim_log_std_yaw", 0.0)
    aim_log_std_pitch = logs.get("policy/aim_log_std_pitch", 0.0)
    return (f"Epoch {epoch} | SPS: {sps:.0f} | Timeout: {timeout:.3f} | "
            f"TWin: {t_win:.3f} | CTWin: {ct_win:.3f} | Plant: {plant:.3f} | "
            f"Kills(T/CT): {kills_t:.2f}/{kills_ct:.2f} | RoundLen: {round_len:.1f} | "
            f"Move1: {move_1:.1f} | TS: {ts_val:.3f} | "
            f"aim_log_std_yaw={aim_log_std_yaw:.4f} aim_log_std_pitch={aim_log_std_pitch:.4f}")


# ── SECTION: Record Episode ────────────────────────────────────────────────


def rewards_array_to_dict(rewards):
    return {aid: float(rewards[i]) for i, aid in enumerate(AGENT_IDS)}


def record_episode(
        checkpoint_path=None,
        device="cpu",
        seed=0,
        policy_mode="auto",
        save_path=str(RECORDINGS_DIR / "latest.rrd"),
        map_data=None,
):
    from c_env.cs2_env import make_env as make_c_env
    from map import make_cs2_map
    from nav import CACHE_PATH, NAV_PATH
    from viz import init_recording, log_navmesh, log_tick, log_trimap

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Record] Initialising rerun recording -> {save_path}")
    init_recording(save_path=str(save_path))
    env = make_c_env(seed=seed, auto_reset=False, map_data=map_data)
    if map_data is None:
        md = make_cs2_map(NAV_PATH, CACHE_PATH)
        log_trimap()
        log_navmesh(md.nav_graph)
    else:
        from viz import log_simple_map

        log_simple_map(env.map_data)

    obs, _ = env.reset(seed=seed)

    policy = None
    policy_mode = resolve_policy_mode(checkpoint_path, policy_mode)
    if policy_mode != "random":
        policy = load_policy_from_checkpoint(checkpoint_path, device)

    policy_state = init_policy_state(policy, device)

    done = False
    step_count = 0
    zero_rewards = {aid: 0.0 for aid in AGENT_IDS}
    log_tick(env.snapshot_state(), step_count, zero_rewards)
    while not done and step_count < env.round_time * 2:
        if policy_mode == "random":
            actions = np.asarray(env.action_space.sample(), dtype=np.int32)
            # Random policy doesn't have a continuous head; pass zeros. This
            # leaves agents with pitch=0 / Δyaw=0 every tick, which is fine
            # for "random eval baseline" but obviously no aim variation.
            cont = np.zeros((actions.shape[0], 2), dtype=np.float32)
        else:
            # Returns (actions, cont) — the policy's actual continuous-aim
            # output. Without this, recordings/eval used cont=zeros and the
            # rerun replay showed agents stuck at spawn facing (the "look
            # forward" bug). AIM_DIM=2 is hardcoded against cs2_types.h;
            # if AIM_DIM ever changes the binding-side shape check will
            # raise before we ever silently miscount.
            actions, cont = select_policy_actions_native(policy, obs, device, policy_state,
                                                         policy_mode)

        obs, rewards, terms, truncs, infos = env.step(actions, cont)
        step_count += 1
        log_tick(env.snapshot_state(), step_count, rewards_array_to_dict(rewards))
        done = bool(np.all(terms))
        if policy_state is not None:
            policy_state["done"] = policy_state["done"].new_tensor(
                np.logical_or(terms, truncs).astype(np.float32))

    print(f"[Record] Episode complete ({step_count} ticks). Saved to {save_path}")
    print(f"[Record] View with: python -m rerun {save_path}")


# ── SECTION: Checkpoint evaluation ─────────────────────────────────────────


def evaluate_checkpoint(checkpoint_path=None,
                        device="cpu",
                        start_seed=0,
                        num_episodes=50,
                        policy_mode="auto"):
    policy = None
    policy_mode = resolve_policy_mode(checkpoint_path, policy_mode)
    if policy_mode != "random":
        policy = load_policy_from_checkpoint(checkpoint_path, device)

    metrics = Counter()
    action_hist = [np.zeros(size, dtype=np.int64) for size in ACTION_HEAD_SIZES]
    joint_hist = Counter()

    for episode_idx in range(num_episodes):
        seed = start_seed + episode_idx
        # W3 (#154): the SECOND eval_legacy site, and the only one that passes a
        # seed. That difference is the whole reason the role's builder takes an
        # UNSET sentinel rather than seed=None — c_env.cs2_env.make_env's own
        # default is 0 (not train.py's own make_env, which takes no seed), so spelling the other site's absent seed as None would have changed
        # the env it builds, invisibly to static_data_scalars().
        env = build_env_for("eval_legacy", seed=seed)
        obs, _ = env.reset(seed=seed)
        policy_state = init_policy_state(policy, device)

        done = False
        step_count = 0
        # R0-G: the env's INSTANCE round_time (a --round-time-ticks run may be
        # far shorter than nav.ROUND_TIME); ×2 is the runaway guard only.
        while not done and step_count < env.round_time * 2:
            if policy_mode == "random":
                actions = np.asarray(env.action_space.sample(), dtype=np.int32)
                # See record_episode comment: random has no continuous head;
                # pass zeros. Eval metrics under random policy reflect "no aim
                # input" which is the prior behaviour anyway.
                cont = np.zeros((actions.shape[0], 2), dtype=np.float32)
            else:
                actions, cont = select_policy_actions_native(policy, obs, device, policy_state,
                                                             policy_mode)

            for action in actions:
                for head_idx, action_value in enumerate(action):
                    action_hist[head_idx][int(action_value)] += 1
                joint_hist[tuple(int(v) for v in action)] += 1

            obs, rewards, terms, truncs, infos = env.step(actions, cont)
            step_count += 1

            step_info = extract_env_info(infos)
            metrics["bomb_planted"] += int(step_info.get("bomb_planted", 0))
            metrics["bomb_defused"] += int(step_info.get("bomb_defused", 0))
            metrics["kills_t"] += int(step_info.get("kills_t", 0))
            metrics["kills_ct"] += int(step_info.get("kills_ct", 0))
            metrics["blocked_moves_t"] += int(step_info.get("blocked_moves_t", 0))
            metrics["blocked_moves_ct"] += int(step_info.get("blocked_moves_ct", 0))

            done = bool(np.all(terms))
            if policy_state is not None:
                policy_state["done"] = policy_state["done"].new_tensor(
                    np.logical_or(terms, truncs).astype(np.float32))

            if done:
                metrics["episodes"] += 1
                metrics["winner_t"] += int(step_info.get("winner_t", 0))
                metrics["winner_ct"] += int(step_info.get("winner_ct", 0))
                metrics["timed_out"] += int(step_info.get("timed_out", 0))
                metrics["alive_t_end"] += int(step_info.get("alive_t_end", 0))
                metrics["alive_ct_end"] += int(step_info.get("alive_ct_end", 0))
                metrics["round_length"] += int(step_info.get("round_length", step_count))

    episodes = max(1, metrics["episodes"])
    total_actions = sum(joint_hist.values())

    print(f"[Eval] checkpoint={checkpoint_path or 'None'} policy={policy_mode} "
          f"episodes={metrics['episodes']} seeds={start_seed}..{start_seed + num_episodes - 1}")
    print(f"[Eval] timeout_rate={metrics['timed_out'] / episodes:.3f} "
          f"t_win_rate={metrics['winner_t'] / episodes:.3f} "
          f"ct_win_rate={metrics['winner_ct'] / episodes:.3f}")
    print(f"[Eval] plant_rate={metrics['bomb_planted'] / episodes:.3f} "
          f"defuse_rate={metrics['bomb_defused'] / episodes:.3f} "
          f"kills_t_per_round={metrics['kills_t'] / episodes:.3f} "
          f"kills_ct_per_round={metrics['kills_ct'] / episodes:.3f}")
    print(f"[Eval] avg_round_length={metrics['round_length'] / episodes:.1f} "
          f"avg_alive_t_end={metrics['alive_t_end'] / episodes:.2f} "
          f"avg_alive_ct_end={metrics['alive_ct_end'] / episodes:.2f}")
    print(f"[Eval] blocked_moves_t_per_round={metrics['blocked_moves_t'] / episodes:.2f} "
          f"blocked_moves_ct_per_round={metrics['blocked_moves_ct'] / episodes:.2f}")

    for head_name, counts in zip(ACTION_HEAD_NAMES, action_hist, strict=True):
        print(f"[Eval] {format_histogram_line(head_name, counts)}")

    top_joint = joint_hist.most_common(5)
    if total_actions > 0 and top_joint:
        parts = []
        for action, count in top_joint:
            parts.append(f"{list(action)}={count} ({100.0 * count / total_actions:.1f}%)")
        print(f"[Eval] top_actions: {', '.join(parts)}")


# ── SECTION: PufferLib env factory ─────────────────────────────────────────
# (dead TRAINING_CONFIG dict removed here — zero readers repo-wide, referenced
# a nonexistent sim.py, and its gamma=0.99 contradicted build_train_config;
# finding 21f of docs/2026-07-06-adversarial-review-verification.md)


def make_env(team_spirit=None, map_data=None):
    """Public env wrapper. W3 (#154): a thin delegate to the `external` role.

    The optional defaults stay HERE, on the published signature, rather than
    moving into `_build_external` — that builder requires both arguments so a
    caller that forgets to forward one gets a TypeError instead of a dust2 env.
    """
    return build_env_for("external", team_spirit=team_spirit, map_data=map_data)


def build_env_factory(*, shared_ts, map_data, config=None):
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
    closure captures a config and never a None. Same shape as `make_env`'s own
    default; the one production caller (build_train_env_factory) always passes a
    config. test_build_train_env_factory_carries_args_config pins BOTH halves and
    needs two calls to do it: the args-built factory for "the caller passes a
    config", and a BARE `build_env_factory(...)` for the resolution itself, which
    no call that always passes a config can reach. Until that second call was
    added, deleting the resolution below left the pre-fix test green (re-measured
    2026-09-04; the review measured the whole non-slow suite green with it) while
    this paragraph claimed it was pinned.

    PITFALL (review finding 1): within a training run the run's config reaches
    TWO envs, not one — this factory's, and the fixed-baseline eval env behind
    `--eval-interval`, which train() builds as `build_env_for("eval", ...,
    config=env_config_from_args(args))` from the same resolver
    `build_train_env_factory` reads here, with `assert_eval_env_agreement`
    cross-checking the two right after. What the run's config does NOT reach is
    the OTHER entry points: `--smoke` and `--eval` get `EnvConfig()` from their
    role builders (`--eval` reaches `_build_eval_legacy`; there is no
    --eval-legacy flag), and `--record` builds its env straight off the
    lower-layer `make_env` naming no config at all, which lands on the same
    thing. So `--smoke --reward-ct-survival 0.0` silently runs default weights.
    Symmetrization is the one field even the config-carrying eval env
    deliberately diverges on — `_build_eval` forces it off so eval reports raw,
    cross-run-comparable rewards. Known limitation, #143's neighbourhood; do not
    fix in this branch.
    PITFALL: `seed or 0` is intentional — pufferlib passes seed=None for some
    backends. Keep it.
    R0-D (#135) `_seed`: train() routes the per-env seed through env_kwargs
    (`_seed = env_seed_base(--seed) + i`) because pufferlib.vector.make takes
    `seed` as ITS OWN named parameter and never forwards it to the backend —
    `make(..., seed=X)` is a silent no-op and every env lands on pufferlib's
    default base (env i -> seed i) regardless of --seed. When `_seed` is given
    it wins over pufferlib's `seed`; the legacy path is unchanged otherwise.
    """
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
        # env_factory._build_train; the three payload arguments this call used
        # to pass are one EnvConfig, resolved above the closure so a forked
        # worker can never receive None.
        # tests/fixtures/env_config_pre_165b.json recorded this call before it
        # was typed — its three `train` rows ARE the three seed branches — and
        # tests/test_env_factory.py drives this closure against each of them
        # (test_train_call_site_forwards_the_captured_kwargs) as well as
        # pinning its spelling against the recorded call source
        # (test_migrated_site_still_reads_what_the_old_site_read).
        env = build_env_for("train",
                            shared_ts=shared_ts,
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


def build_train_env_factory(args, *, shared_ts, map_data):
    """The training path's env factory: the run's EnvConfig, derived from args.

    WHY this exists as its own function (review fix 2): it is the seam between
    the CLI and the envs. Inlined in train() it was untestable without
    launching a run, so nothing caught a regression that dropped the run's
    weights — exactly the silent-baseline failure this whole change is guarding
    against. test_train_uses_build_train_env_factory pins train() to it, and
    test_build_train_env_factory_carries_args_config reads the config back out
    of the closure it returns.

    Derives ONE config from the SAME resolver build_train_config uses
    (`env_config_from_args`), so config.json provenance and the envs that
    actually ran cannot disagree — about a weight, about symmetrization or about
    a sim knob. Before #165 PR B2 that was three separate derivations here, each
    with its own way to fall out of step; train() also asserts the built
    driver_env agrees with the participation vector it derives from the same
    args.
    """
    return build_env_factory(shared_ts=shared_ts,
                             map_data=map_data,
                             config=env_config_from_args(args))


# ── SECTION: Policy ────────────────────────────────────────────────────────


def build_policy(vecenv,
                 device,
                 obs_dim_override=None,
                 tct_split_heads=False,
                 tct_split_trunk=False,
                 aim_log_std_max=None,
                 pin_pitch=False):
    """Build the Dust2 recurrent policy.

    aim_log_std_max / pin_pitch (R0-E.3 / R0-E.2, #131): per-RUN aim-head
    properties. The cap replaces LOG_STD_MAX at every σ clamp site; pin_pitch
    sets ``policy.aim_dim_mask`` to [1, 0] so the pitch dim drops out of
    log_prob_c / entropy_c. Both live as NON-persistent buffers/attrs so old
    checkpoints still load and a checkpoint never carries them — every loader
    (SelfPlayManager.load_past_policy, load_policy_from_checkpoint) must pass
    the run's values explicitly. Raises ValueError if the cap leaves the
    (LOG_STD_MIN, LOG_STD_MAX] band.

    tct_split_heads (Batch 7, spec 2026-08-13): when True the policy-head
    group — the 7 discrete action_heads, the aim_mu projection and the
    aim_log_std parameter — is duplicated per team (`_t` / `_ct` suffixes) and
    each row is routed to its own team's copy by the obs team bit obs[24].
    value_head stays SHARED. Default False builds the legacy head modules
    and executes the legacy head-forward lines verbatim, pinned by
    tests/test_tct_split.py::test_flag_off_builds_exactly_the_legacy_modules.

    tct_split_trunk (spec 2026-08-15): when True the trunk — encoder + LSTM —
    is replaced by per-team copies (`encoder_t`/`lstm_t`, `encoder_ct`/`lstm_ct`).
    Each team LSTM sees only its own encoder's activations; hidden and the
    rollout (h,c) blend on obs[24]. Default False keeps today's
    `self.encoder` / `self.lstm` construction verbatim (legacy RNG pin).

    PITFALL: callers must not decide split-ness from config alone — every
    loader infers it from the checkpoint's keys (state_dict_is_split /
    state_dict_is_trunk_split), because config.json is rewritten on each
    launch and a flag-less crash-resume would otherwise rebuild the wrong
    architecture (spec §3.3).
    """
    import pufferlib.pytorch
    import torch
    import torch.nn as nn

    driver_env = getattr(vecenv, "driver_env", vecenv)
    obs_dim = (obs_dim_override
               if obs_dim_override is not None else driver_env.single_observation_space.shape[0])
    hidden = 256
    _cap = validate_aim_log_std_max(aim_log_std_max)
    # Rung 1a T1: the FRESH σ init follows the cap (see resolve_aim_log_std_init)
    # — LOG_STD_INIT at the 5v5 default cap, cap − 0.2 under a tight one, never
    # AT the cap where clamp would zero the gradient forever.
    _log_std_init = resolve_aim_log_std_init(_cap)

    class Dust2Policy(nn.Module):

        def __init__(self):
            super().__init__()
            self.hidden_size = hidden  # required by PufferLib LSTM logic
            self.obs_dim = obs_dim

            # Trunk-off keeps today's encoder/lstm construction verbatim so
            # the flag-off RNG stream (and LEGACY_PARAM_NAMES) stay pinned.
            # Trunk-on REPLACES those modules — do not keep a shared encoder
            # or lstm beside the copies.
            if not tct_split_trunk:
                self.encoder = nn.Sequential(
                    pufferlib.pytorch.layer_init(nn.Linear(obs_dim, hidden)),
                    nn.ReLU(),
                    pufferlib.pytorch.layer_init(nn.Linear(hidden, hidden)),
                    nn.ReLU(),
                )
                self.lstm = nn.LSTM(hidden, hidden, batch_first=False)
                for name, p in self.lstm.named_parameters():
                    if "bias" in name:
                        nn.init.constant_(p, 0)
                    elif "weight" in name:
                        nn.init.orthogonal_(p, gain=1.0)
            else:
                self.encoder_t = nn.Sequential(
                    pufferlib.pytorch.layer_init(nn.Linear(obs_dim, hidden)),
                    nn.ReLU(),
                    pufferlib.pytorch.layer_init(nn.Linear(hidden, hidden)),
                    nn.ReLU(),
                )
                self.lstm_t = nn.LSTM(hidden, hidden, batch_first=False)
                for name, p in self.lstm_t.named_parameters():
                    if "bias" in name:
                        nn.init.constant_(p, 0)
                    elif "weight" in name:
                        nn.init.orthogonal_(p, gain=1.0)
                # RNG hygiene (same as heads §3.7): CT construction is forked
                # so the subsequent heads draw stays at the same stream point
                # as flag-off. devices=[] forks the CPU generator only.
                with torch.random.fork_rng(devices=[]):
                    self.encoder_ct = nn.Sequential(
                        pufferlib.pytorch.layer_init(nn.Linear(obs_dim, hidden)),
                        nn.ReLU(),
                        pufferlib.pytorch.layer_init(nn.Linear(hidden, hidden)),
                        nn.ReLU(),
                    )
                    self.lstm_ct = nn.LSTM(hidden, hidden, batch_first=False)
                    for name, p in self.lstm_ct.named_parameters():
                        if "bias" in name:
                            nn.init.constant_(p, 0)
                        elif "weight" in name:
                            nn.init.orthogonal_(p, gain=1.0)

            # Batch 7 (spec 2026-08-13 §3.1): plain bool, NOT a buffer — it
            # must never enter state_dict() or every existing checkpoint would
            # gain a key. Loaders read it to detect a policy/checkpoint
            # architecture mismatch (load_state_dict_arch_checked).
            # `tct_split_heads` here is build_policy's parameter, captured by
            # closure exactly like `obs_dim` and `hidden` above — the inner
            # class takes no new constructor argument. Same for tct_split_trunk.
            self.tct_split_heads = bool(tct_split_heads)
            self.tct_split_trunk = bool(tct_split_trunk)

            # Separate heads for MultiDiscrete(ACTION_HEAD_SIZES)
            #
            # Batch 3: continuous Gaussian aim head.
            # mu_aim → (B, AIM_DIM); tanh-squashed and scaled by max_turn_speed
            #   in forward(). State-DEPENDENT (per-step linear projection) so
            #   the policy can react to the current obs (visible enemies, yaw
            #   delta to target, etc.) when picking the mean Δyaw.
            # aim_log_std → (AIM_DIM,) — state-INDEPENDENT learnable parameter
            #   per Fan et al. IJCAI 2019 H-PPO baseline. Clamped in forward()
            #   to [LOG_STD_MIN, LOG_STD_MAX] so neither σ collapse (entropy
            #   loss → −∞) nor explosion (σ floods policy) is reachable.
            #   Rung 1a T1: the init is _log_std_init, not LOG_STD_INIT — it
            #   must start strictly INSIDE that clamp band or the parameter
            #   receives zero gradient for the whole run.
            # Pitfall: keep `std=0.01` on aim_mu init so the pre-tanh mean
            #   starts ~zero — otherwise the policy starts saturated and
            #   learning the Gaussian head is much slower.
            #
            # Batch 7 note on the deliberate duplication of these three
            # expressions across the two branches: the construction ORDER
            # (7 discrete heads → value_head → aim_mu → aim_log_std) is what
            # determines how many draws each layer takes from the global torch
            # RNG. Factoring the head group into a shared helper would move
            # value_head's draw and change every layer's init relative to the
            # legacy baseline at the same seed. Repetition here buys exact
            # RNG-stream parity between the flag-off and flag-on `_t` copies,
            # which is the whole point of spec §3.7.
            if not self.tct_split_heads:
                self.action_heads = nn.ModuleList([
                    pufferlib.pytorch.layer_init(nn.Linear(hidden, n), std=0.01)
                    for n in ACTION_HEAD_SIZES
                ])
                self.value_head = pufferlib.pytorch.layer_init(nn.Linear(hidden, 1), std=1.0)
                self.aim_mu = pufferlib.pytorch.layer_init(nn.Linear(hidden, AIM_DIM), std=0.01)
                self.aim_log_std = nn.Parameter(torch.full((AIM_DIM, ), _log_std_init))
            else:
                self.action_heads_t = nn.ModuleList([
                    pufferlib.pytorch.layer_init(nn.Linear(hidden, n), std=0.01)
                    for n in ACTION_HEAD_SIZES
                ])
                self.value_head = pufferlib.pytorch.layer_init(nn.Linear(hidden, 1), std=1.0)
                self.aim_mu_t = pufferlib.pytorch.layer_init(nn.Linear(hidden, AIM_DIM), std=0.01)
                self.aim_log_std_t = nn.Parameter(torch.full((AIM_DIM, ), _log_std_init))
                # RNG hygiene (spec §3.7): the CT copy's construction is what
                # draws from the default stream, so it is forked — post-hoc
                # weight cloning would NOT restore stream parity. Without this
                # the flag-on run's every subsequent sample shifts relative to
                # the baseline at the same seed and "the split is the only
                # changed variable" is strictly false. devices=[] forks the CPU
                # generator only (construction is on CPU; .to(device) happens
                # after) and skips CUDA device enumeration.
                with torch.random.fork_rng(devices=[]):
                    self.action_heads_ct = nn.ModuleList([
                        pufferlib.pytorch.layer_init(nn.Linear(hidden, n), std=0.01)
                        for n in ACTION_HEAD_SIZES
                    ])
                    self.aim_mu_ct = pufferlib.pytorch.layer_init(nn.Linear(hidden, AIM_DIM),
                                                                  std=0.01)
                self.aim_log_std_ct = nn.Parameter(torch.full((AIM_DIM, ), _log_std_init))

            # max_turn_speed mirrors C sd->max_turn_speed (StaticData, π/4
            # default). Pulled from the vecenv's static-data block so the
            # policy stays bound to the env's actual cap even if it changes
            # at env construction time. Stored as a buffer (no grad, not a
            # learnable param, follows .to(device)). T5 carry-forward (I-1):
            # reuse the `driver_env` helper resolved at the top of build_policy instead of
            # an inline hasattr ladder — the helper already handles the
            # vecenv-vs-driver-env duality (test path passes a bare env;
            # production passes a Multiprocessing/Serial vecenv). One source
            # of truth for the "what is the env?" question.
            self.register_buffer(
                'max_turn_speed',
                torch.tensor(driver_env._c_env.sd.contents.max_turn_speed, dtype=torch.float32),
            )
            # R0-E.3/4 (#131): run properties, NOT checkpoint state
            # (persistent=False so old checkpoints load and new ones don't
            # carry them; SelfPlayManager re-applies them to past policies).
            # aim_log_std_max caps σ in every forward (replaces LOG_STD_MAX at
            # all clamp sites below); aim_dim_mask weights the per-dim
            # Gaussian log-prob/entropy terms ([1,0] when pitch is pinned).
            # PITFALL: sampling still draws BOTH dims (the env ignores dim 1
            # when pinned) — only the density is masked, so the stored
            # cont_action stays byte-identical to what the env consumed.
            self.aim_log_std_max = _cap
            self.register_buffer("aim_dim_mask",
                                 torch.tensor([1.0, 0.0] if pin_pitch else [1.0, 1.0]),
                                 persistent=False)

        @staticmethod
        def _blend(mask, out_t, out_ct):
            """Route a per-row output to its team's head copy (spec §3.2).

            mask is 0/1 with 1.0 == T, broadcastable over out_t's trailing
            dims. Branch-free (GPU-friendly) and autograd-exact: a T row's
            blend weight on the CT copy is literally 0, so it contributes zero
            gradient there — that is the routing correctness proof, pinned by
            test_pure_team_batch_leaves_other_copy_gradient_exactly_zero.

            PITFALL: the cast is load-bearing. A float32 mask multiplied into
            fp16 head outputs would silently promote them under any future
            autocast; casting to the output dtype keeps the arithmetic in the
            head's own precision.
            """
            m = mask.to(out_t.dtype)
            return m * out_t + (1.0 - m) * out_ct

        def get_value(self, x, lstm_state=None, done=None):
            hidden_out, lstm_state = self._forward_core(x, lstm_state, done)
            return self.value_head(hidden_out), lstm_state

        def get_action_and_value(self,
                                 x,
                                 lstm_state=None,
                                 done=None,
                                 action=None,
                                 continuous_action=None):
            """Hybrid sampler combining 7 categorical heads + 1 Gaussian aim head.

            Args:
                x: (B, OBS_DIM) observation batch.
                lstm_state: optional (h, c) tuple for the LSTM rollout.
                done: optional (B,) done-mask used to reset LSTM state.
                action: (B, ACTION_DIM=7) int64 — discrete actions; if None,
                    sample from the categorical heads.
                continuous_action: (B, AIM_DIM=2) float32 — (Δyaw, Δpitch) in
                    radians, already in [-max_turn_speed, +max_turn_speed];
                    if None, sample from the Normal head.

            Returns:
                (action, continuous_action, log_prob, entropy, value, lstm_state)
                log_prob and entropy aggregate across all 8 factors (7
                categorical + 1 Normal) — discrete factors are independent so
                their log-probs sum, and the Gaussian factor adds to the
                total. PPO loss assembly + the matching trainer side
                (rollout buffer for continuous_action, ratio computation)
                lands in task 5 via _patch_trainer_with_hybrid_aim.
            """
            hidden_out, lstm_state = self._forward_core(x, lstm_state, done)
            if self.tct_split_heads:
                # 2D input (B, obs): the team bit is a column. (The 3D
                # timestep trap lives in forward(), not here.)
                mask = x[:, 24:25]
                logits = [
                    self._blend(mask, ht(hidden_out), hct(hidden_out))
                    for ht, hct in zip(self.action_heads_t, self.action_heads_ct, strict=True)
                ]
            else:
                logits = [head(hidden_out) for head in self.action_heads]

            # Discrete sample / log-prob / entropy.
            dists = [torch.distributions.Categorical(logits=h) for h in logits]
            if action is None:
                action = torch.stack([d.sample() for d in dists], dim=-1)
            log_prob_d = sum(d.log_prob(action[..., i]) for i, d in enumerate(dists))
            entropy_d = sum(d.entropy() for d in dists)

            # Continuous (Normal) sample / log-prob / entropy. tanh+scale
            # bounds μ ∈ [-max_turn_speed, +max_turn_speed]; σ is clamped so
            # the Normal can't collapse or explode mid-training.
            if self.tct_split_heads:
                mu_aim = self._blend(mask,
                                     torch.tanh(self.aim_mu_t(hidden_out)) * self.max_turn_speed,
                                     torch.tanh(self.aim_mu_ct(hidden_out)) * self.max_turn_speed)
                # clamp EACH copy, then blend (spec §3.2) — identical result
                # for a 0/1 mask, but it matches the legacy clamp-at-use
                # semantics and keeps §3.6's per-team σ logs interpretable.
                log_std = self._blend(
                    mask,
                    torch.clamp(self.aim_log_std_t, LOG_STD_MIN,
                                self.aim_log_std_max).expand_as(mu_aim),
                    torch.clamp(self.aim_log_std_ct, LOG_STD_MIN,
                                self.aim_log_std_max).expand_as(mu_aim))
            else:
                mu_aim = torch.tanh(self.aim_mu(hidden_out)) * self.max_turn_speed
                log_std = torch.clamp(self.aim_log_std, LOG_STD_MIN, self.aim_log_std_max)
            sigma = torch.exp(log_std).expand_as(mu_aim)
            aim_dist = torch.distributions.Normal(mu_aim, sigma)
            if continuous_action is None:
                # rsample preserves the reparameterised path through μ in case
                # the trainer ever uses pathwise gradients (PPO doesn't, but
                # cheap to keep this future-proof).
                continuous_action = aim_dist.rsample()
                # Re-clamp post-sample (T5 carry-forward I-2): σ exploration
                # can land outside the tanh band. The C env (env_step, cs2_env.h)
                # clamps |Δyaw| ≤ max_turn_speed silently with fminf/fmaxf —
                # NOT an assert. The Python-side clamp keeps the recorded
                # `continuous_action` byte-identical to what the env actually
                # consumed, which matters for PPO's importance ratio: if we
                # stored the unclamped sample and the env clipped it, the
                # ratio re-evaluation in _hybrid_ppo_loss would be wrong by
                # the clipping amount on every saturated step.
                continuous_action = torch.clamp(
                    continuous_action,
                    -self.max_turn_speed,
                    self.max_turn_speed,
                )
            # R0-E.2: per-dim weight applied BEFORE the sum so a pinned dim
            # contributes neither log-prob nor entropy (mirrors
            # _hybrid_sample_logits / _hybrid_ppo_loss).
            log_prob_c = (aim_dist.log_prob(continuous_action) * self.aim_dim_mask).sum(-1)
            # Closed-form Gaussian entropy: 0.5·log(2πe·σ²), summed across
            # AIM_DIM. .entropy() returns per-dim, so .sum(-1) is correct
            # for AIM_DIM=1 today and stays correct if AIM_DIM bumps to ≥2.
            entropy_c = (aim_dist.entropy() * self.aim_dim_mask).sum(-1)

            log_prob = log_prob_d + log_prob_c
            entropy = entropy_d + entropy_c
            value = self.value_head(hidden_out)
            return action, continuous_action, log_prob, entropy, value, lstm_state

        def forward_eval(self, x, state):
            # Batch 3: returns 4-tuple (logits, mu_aim, log_std, value) so
            # downstream samplers (eval loop / record / past-policy mixing)
            # can construct the full hybrid action. Existing 2-tuple
            # consumers break here — task 5 updates them.
            done = state.get("done")
            if done is None:
                done = x.new_zeros(x.shape[0])

            lstm_state = None
            if state.get("lstm_h") is not None and state.get("lstm_c") is not None:
                lstm_state = (state["lstm_h"], state["lstm_c"])

            hidden_out, lstm_state = self._forward_core(x, lstm_state, done)
            if lstm_state is not None:
                state["lstm_h"], state["lstm_c"] = lstm_state

            if self.tct_split_heads:
                mask = x[:, 24:25]                                                                # 2D input (B, obs)
                logits = [
                    self._blend(mask, ht(hidden_out), hct(hidden_out))
                    for ht, hct in zip(self.action_heads_t, self.action_heads_ct, strict=True)
                ]
                mu_aim = self._blend(mask,
                                     torch.tanh(self.aim_mu_t(hidden_out)) * self.max_turn_speed,
                                     torch.tanh(self.aim_mu_ct(hidden_out)) * self.max_turn_speed)
                log_std = self._blend(
                    mask,
                    torch.clamp(self.aim_log_std_t, LOG_STD_MIN,
                                self.aim_log_std_max).expand_as(mu_aim),
                    torch.clamp(self.aim_log_std_ct, LOG_STD_MIN,
                                self.aim_log_std_max).expand_as(mu_aim))
            else:
                logits = [head(hidden_out) for head in self.action_heads]
                                                                                                  # μ is bounded by tanh*max_turn_speed; log_std broadcasts to μ
                                                                                                  # shape so callers can build Normal(mu, exp(log_std)) directly
                                                                                                  # without an extra .expand call.
                mu_aim = torch.tanh(self.aim_mu(hidden_out)) * self.max_turn_speed
                log_std = torch.clamp(self.aim_log_std, LOG_STD_MIN,
                                      self.aim_log_std_max).expand_as(mu_aim)
            value = self.value_head(hidden_out)
            return logits, mu_aim, log_std, value

        def forward(self, x, state):
            # Training-path forward: time-batched BPTT (LSTM-BPTT fix).
            #
            # WHAT: same 4-tuple contract as forward_eval, but a 3D input
            #   (B=segments, T=bptt_horizon, OBS_DIM) is now unrolled through
            #   the LSTM along T — mirroring upstream PufferLib 3.0's
            #   models.LSTMWrapper.forward (encode flat → reshape seq-first →
            #   one nn.LSTM call → heads on the flat output). Pre-fix this
            #   flattened to (B*T, OBS) and ran the LSTM stateless per tick
            #   (seq-len 1, zero state), so the recurrent weights never saw
            #   through-time gradients and the PPO update recomputed
            #   logprobs/values under a DIFFERENT function than the rollout
            #   (forward_eval carries state tick-to-tick) — importance
            #   ratios ≠ 1 before the first gradient step.
            #
            # WHY zero initial state is CORRECT here (not an approximation):
            #   evaluate() zeroes trainer.lstm_h/c at its start, and with
            #   compute_batch_dims' segments == total_agents each agent row
            #   fills exactly ONE bptt_horizon segment per evaluate() call —
            #   so every stored segment really did start from zero state.
            #   PITFALL: if batch dims ever change so a row fills >1 segment
            #   per evaluate(), zero-init becomes wrong for the later
            #   segments and initial states must be stored at rollout time.
            #
            # state keys consumed (all optional; dict is NOT mutated):
            #   lstm_h / lstm_c — initial state override, (B, H) or (1, B, H).
            #     The trainer passes None → zero init (see above).
            #   terminals — (B, T) done flags from the rollout buffer;
            #     replicates forward_eval's (1-done)*state reset mid-segment
            #     (see _lstm_bptt). Omit for the ONNX / single-tick path.
            #
            # ONNX (task 6): a 2D (B, OBS_DIM) input takes T=1 through the
            # same code — one seq-len-1 LSTM call from zero state, identical
            # math to the pre-fix path — so the export stays single-pathway.
            if x.ndim == 3:
                B, TT = x.shape[0], x.shape[1]
            else:
                B, TT = x.shape[0], 1

            lstm_h = state.get("lstm_h") if isinstance(state, dict) else None
            lstm_c = state.get("lstm_c") if isinstance(state, dict) else None
            terminals = state.get("terminals") if isinstance(state, dict) else None
            H = self.hidden_size

            if not self.tct_split_trunk:
                h = self.encoder(x.reshape(B * TT, x.shape[-1]).float())
                # (T, B, H) seq-first
                h = h.reshape(B, TT, H).transpose(0, 1)
                if lstm_h is not None and lstm_c is not None:
                    hc = (lstm_h.reshape(1, B, H), lstm_c.reshape(1, B, H))
                else:
                    hc = (h.new_zeros(1, B, H), h.new_zeros(1, B, H))
                h = self._lstm_bptt(self.lstm, h, hc, terminals)
                # transpose back to (B, T, H) then flatten row-major so flat row
                # b*T + t lines up with mb_actions.reshape(-1, ...) in
                # _hybrid_ppo_loss — segment-major, time-minor. Changing this
                # ordering silently misaligns every logprob/advantage pairing.
                hidden_out = h.transpose(0, 1).reshape(B * TT, H)
            else:
                # Encoder is stateless: both copies see the same flat rows.
                # LSTM is not a head: each team LSTM sees ONLY its encoder's
                # activations. Never feed a mixed batch through one LSTM.
                x_flat = x.reshape(B * TT, x.shape[-1]).float()
                h_t = self.encoder_t(x_flat).reshape(B, TT, H).transpose(0, 1)
                h_ct = self.encoder_ct(x_flat).reshape(B, TT, H).transpose(0, 1)
                # zero-init BOTH team states when the trainer does not pass lstm_h/c
                if lstm_h is not None and lstm_c is not None:
                    hc_t = (lstm_h.reshape(1, B, H), lstm_c.reshape(1, B, H))
                    hc_ct = (lstm_h.reshape(1, B, H), lstm_c.reshape(1, B, H))
                else:
                    hc_t = (h_t.new_zeros(1, B, H), h_t.new_zeros(1, B, H))
                    hc_ct = (h_ct.new_zeros(1, B, H), h_ct.new_zeros(1, B, H))
                y_t = self._lstm_bptt(self.lstm_t, h_t, hc_t, terminals)
                y_ct = self._lstm_bptt(self.lstm_ct, h_ct, hc_ct, terminals)
                # PITFALL (spec §3.2): mask from 3D x with x[..., 24], never
                # x[:, 24] — that silently selects TIMESTEP 24.
                mask = x[..., 24].reshape(B * TT, 1)
                hidden_out = self._blend(mask,
                                         y_t.transpose(0, 1).reshape(B * TT, H),
                                         y_ct.transpose(0, 1).reshape(B * TT, H))

            if self.tct_split_heads:
                # PITFALL (spec §3.2 — the bug class this comment exists to
                # prevent): build the mask from the 3D x with x[..., 24].
                # Writing x[:, 24] on a (B, T, obs) input silently selects
                # TIMESTEP 24 instead of the team column. The reshape to
                # (B*TT, 1) is aligned with hidden_out's
                # h.transpose(0,1).reshape(B*TT, H) — both segment-major,
                # time-minor. Works unchanged for the 2D/ONNX path, where
                # TT == 1 and x[..., 24] is already the team column.
                mask = x[..., 24].reshape(B * TT, 1)
                logits = [
                    self._blend(mask, ht(hidden_out), hct(hidden_out))
                    for ht, hct in zip(self.action_heads_t, self.action_heads_ct, strict=True)
                ]
                mu_aim = self._blend(mask,
                                     torch.tanh(self.aim_mu_t(hidden_out)) * self.max_turn_speed,
                                     torch.tanh(self.aim_mu_ct(hidden_out)) * self.max_turn_speed)
                log_std = self._blend(
                    mask,
                    torch.clamp(self.aim_log_std_t, LOG_STD_MIN,
                                self.aim_log_std_max).expand_as(mu_aim),
                    torch.clamp(self.aim_log_std_ct, LOG_STD_MIN,
                                self.aim_log_std_max).expand_as(mu_aim))
            else:
                logits = [head(hidden_out) for head in self.action_heads]
                mu_aim = torch.tanh(self.aim_mu(hidden_out)) * self.max_turn_speed
                log_std = torch.clamp(self.aim_log_std, LOG_STD_MIN,
                                      self.aim_log_std_max).expand_as(mu_aim)
            value = self.value_head(hidden_out)
            return logits, mu_aim, log_std, value

        def _lstm_bptt(self, lstm, h_seq, hc, terminals):
            """Run one LSTM over a full (T, B, H) segment with done-masking.

            WHAT: one `lstm(...)` call when the segment contains no episode
            boundaries (the common case — native PufferLib BPTT); otherwise
            the sequence is split at every tick where ANY row has a done and
            h/c are zero-masked per-row at those ticks before continuing.
            `lstm` is the module to run — flag-off forward passes
            `self.lstm`; trunk-on passes `self.lstm_t` / `self.lstm_ct`
            separately so each copy sees only its encoder's activations.

            WHY: the rollout (forward_eval → _forward_core) multiplies the
            carried state by (1 - done) BEFORE processing each tick, so a
            new episode starts memory-free. Training must replicate that
            reset or the recomputed logprobs at post-done ticks come from a
            different function than the rollout stored (biased PPO ratios)
            and gradients leak across episode boundaries. Upstream
            LSTMWrapper skips this (it never resets on done, rollout OR
            train, so it is self-consistent); we reset in rollout, hence we
            must also reset here. Passing the module in avoids copy-pasting
            this done-chunk loop per team.

            PITFALLS:
              * terminals[:, t] == 1 means "the obs at tick t is the FIRST
                obs of a new episode" (PufferLib autoreset delivers the done
                flag alongside the reset obs) — mask BEFORE consuming tick t,
                not after. Off-by-one here shifts every episode boundary.
              * The chunked split is exact, not an approximation: an LSTM
                over [t0, t1) then [t1, t2) with state carried equals one
                call over [t0, t2). Splits only cost extra kernel launches;
                a no-done minibatch stays a single cuDNN/oneDNN call.
              * .tolist() forces one device→host sync per minibatch —
                acceptable (the train loop already syncs via .item()s).
              * LSTM must not see the other team's rows via a shared module:
                the caller encodes per team, then calls this twice. Do not
                blend encoder outputs and run one LSTM.
            """
            if terminals is None:
                out, _ = lstm(h_seq, hc)
                return out
            TT, B, _H = h_seq.shape
            term = terminals.reshape(B, TT) > 0.5
            reset_ticks = torch.nonzero(term.any(dim=0)).flatten().tolist()
            if not reset_ticks:
                out, _ = lstm(h_seq, hc)
                return out
            outs = []
            h0, c0 = hc
            t0 = 0
            for t in reset_ticks:
                if t > t0:
                    out, (h0, c0) = lstm(h_seq[t0:t], (h0, c0))
                    outs.append(out)
                keep = (~term[:, t]).float().view(1, B, 1)
                h0 = h0 * keep
                c0 = c0 * keep
                t0 = t
            out, _ = lstm(h_seq[t0:], (h0, c0))
            outs.append(out)
            return torch.cat(outs, dim=0)

        def _forward_core(self, x, lstm_state, done):
            """Single-tick encode + LSTM for rollout / eval.

            WHAT: one seq-len-1 LSTM step. Trunk-off is today's
            `self.encoder` then `self.lstm(h.unsqueeze(0), ...)`. Trunk-on
            runs both team encoders+lstms the same way, blends hidden, and
            blends the returned `(h,c)` with `mask.view(1, B, 1)` so the
            trainer still stores one pair.

            WHY: `forward_eval` / `get_action_and_value` inherit routing
            from here. The trainer LSTM buffers stay one `(h,c)` per agent
            (do not change PufferLib's rollout state).

            PITFALLS:
              * Do not route this through `_lstm_bptt` — that helper is the
                training-path T-unroll. This must stay the per-tick
                `lstm(h.unsqueeze(0))` call.
              * LSTM must not see the other team's encoder activations:
                each copy is fed only its encoder's h. The incoming blended
                state is `(1-done)`-reset once, then fed to BOTH team LSTMs
                (unused output dropped by the 0/1 blend; used path is exact).
              * 2D mask is `x[:, 24:25]`. The 3D timestep-24 trap lives in
                `forward()`, not here.
            """
            if not self.tct_split_trunk:
                h = self.encoder(x.float())
                # lstm expects (seq, batch, features)
                if lstm_state is not None:
                    done = done.float()
                    h, lstm_state = self.lstm(
                        h.unsqueeze(0),
                        (
                            (1.0 - done).view(1, -1, 1) * lstm_state[0],
                            (1.0 - done).view(1, -1, 1) * lstm_state[1],
                        ),
                    )
                    h = h.squeeze(0)
                else:
                    h, lstm_state = self.lstm(h.unsqueeze(0))
                    h = h.squeeze(0)
                return h, lstm_state

            # 2D path: team bit is a column.
            mask = x[:, 24:25]
            h_t = self.encoder_t(x.float())
            h_ct = self.encoder_ct(x.float())
            if lstm_state is not None:
                done = done.float()
                reset_state = (
                    (1.0 - done).view(1, -1, 1) * lstm_state[0],
                    (1.0 - done).view(1, -1, 1) * lstm_state[1],
                )
                y_t, state_t = self.lstm_t(h_t.unsqueeze(0), reset_state)
                y_ct, state_ct = self.lstm_ct(h_ct.unsqueeze(0), reset_state)
            else:
                y_t, state_t = self.lstm_t(h_t.unsqueeze(0))
                y_ct, state_ct = self.lstm_ct(h_ct.unsqueeze(0))
            hidden_out = self._blend(mask, y_t.squeeze(0), y_ct.squeeze(0))
            m_state = mask.view(1, x.shape[0], 1)
            lstm_state = (
                self._blend(m_state, state_t[0], state_ct[0]),
                self._blend(m_state, state_t[1], state_ct[1]),
            )
            return hidden_out, lstm_state

    return Dust2Policy().to(device)


# ── SECTION: R0-I fixed-baseline evaluation hooks ─────────────────────────


def elimination_only_win_rates(logs):
    """R0-I: (win_rate_t, win_rate_ct) with timeouts removed from the CT side.

    WHY: cs2_rewards.h scores a timeout as a CT win (`winner_ct`), so on a
    bombsite-less duel map a CT that never engages "wins" every round and
    SelfPlayManager.win_threshold would pool-save a statue. `winner_t` is
    elimination-only on bombsites=[] maps. All three keys are window means
    over the same episode set, so the subtraction is exact; clamped at 0 for
    the float-noise case. Missing keys (first epoch) → 0.0, never KeyError.
    """
    wt = float(logs.get("environment/winner_t", 0.0))
    wct = max(
        0.0,
        float(logs.get("environment/winner_ct", 0.0)) -
        float(logs.get("environment/timed_out", 0.0)))
    return wt, wct


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
        from map import make_arena_duel_map
        return make_arena_duel_map()
    from map import make_simple_map
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
    (costs ~1 s for dust2 from the nav cache, ~0.8 s for `import map`). train()
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
          back to the nav.py constant when it is None), and the other four are
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

    WHY reward_symmetrize is skipped and nothing else is: env_factory._build_eval
    FORCES it off — `config.replace(reward_symmetrize=False)` — so the eval env
    reports raw rewards while the training envs take the flag from args. Since
    #165 PR B2 that is an explicit rule at the builder rather than, as before, a
    parameter the eval chain simply never passed. Either way the two configs are
    MEANT to differ there and only there, and skipping any other field would
    hide a real divergence.

    DISCLOSURE — NEITHER CHECK CAN FIRE ON ANY INPUT REACHABLE TODAY, and since
    #165 PR B2 that is structural rather than measured. train()'s only call site
    compares the env from build_env_for("eval", ..., config=env_config_from_args(
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
    applied value: None means "the nav.py constant" and Cs2Env resolves it, so a
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
    DIRECTLY by tests (tests/test_pitch_pin.py and
    tests/test_env_knobs.py::test_policy_max_turn_speed_assert respectively).

    PITFALL: check (a) runs FIRST, so a test that tries to prove (b) exists by
    differing one of the five names in the tuple below will raise from (a) and
    prove nothing. tests/test_env_factory.py::test_eval_env_agreement_two_directions
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


def _kill_reward_is_active(vecenv):
    """True if the env pays a nonzero per-kill reward (gh#93).

    WHAT: reads StaticData.reward_kill out of the C env behind a vecenv, the
    same `driver_env._c_env.sd` path build_policy uses for max_turn_speed.

    WHY: DeadRunDetector's zero-kills rule is only meaningful when kills are
    something the reward function actually asks for. With the weight at 0,
    kills_per_episode==0 is the configuration, not a dead run.

    PITFALLS: returns True (alert stays armed) for ANY env it cannot read —
    unknown must not silently disable a safety check. Note this only covers the
    weight-is-zero case; a nonzero weight whose behaviour has simply not emerged
    yet is handled by the non-accumulating alert inside check().
    """
    try:
        driver_env = getattr(vecenv, "driver_env", vecenv)
        return float(driver_env._c_env.sd.contents.reward_kill) != 0.0
    except Exception:
        return True


class DeadRunDetector:
    """Checks training metrics every check_interval steps for degenerate runs.

    Raises RuntimeError on NaN/Inf; accumulates soft warnings and prints a
    DEAD RUN banner when five or more accumulate, returning True. F14
    (2026-07-06 adversarial review): the train() loop now ACTS on that True —
    autopsy checkpoint + SystemExit(3) — unless --no-dead-run-abort is set.
    Callers embedding this class elsewhere must handle the return themselves;
    a discarded return silently reduces it to a log line (the failure mode
    that let the 30M degenerate run burn ~150 post-verdict epochs).
    """

    def __init__(self, check_interval=10_000, kills_expected=True):
        """kills_expected: False suppresses the zero-kills rule entirely (gh#93).

        Pass False when the environment's kill reward is switched off, i.e. when
        kills_per_episode==0 is the configured outcome rather than evidence of a
        dead run. Callers get this from _kill_reward_is_active(vecenv); the
        default stays True so an unknown/unreadable env keeps the alert.
        """
        self.check_interval = check_interval
        self.kills_expected = kills_expected
        self.alerts = []

    def check(self, step, metrics):
        """Return True if the run appears dead (enough alerts accumulated)."""
        if step < self.check_interval:
            return False

        # Critical: NaN / Inf in any float metric
        for v in metrics.values():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                raise RuntimeError(f"NaN/Inf detected at step {step}: {v}")

        # Clear alerts if metrics are healthy now
        if metrics.get("game/kills_per_episode", 0) > 0.5:
            self.alerts = [a for a in self.alerts if "kills" not in a]

        if step > 50_000:
            # R0-J: the outer log dict is prefixed `losses/` (pufferl.py
            # mean_and_log) — the old `entropy/total` / `approx_kl` keys never
            # matched, so the entropy and KL rules were dead since day one.
            # The unprefixed fallbacks keep the harness / older callers working.
            entropy_total = metrics.get("losses/entropy", metrics.get("entropy/total", 5.0))
            if entropy_total < 0.5:
                self.alerts.append(
                    f"CRITICAL: Entropy collapsed to {entropy_total:.2f} at step {step}")
            timeout_rate = metrics.get("game/timeout_rate", 0.0)
            # Non-accumulating (like zero-kills, gh#93): at Rung 1 every
            # no-kill round is a timeout, so an untrained policy sits at ~1.0
            # and would abort itself in five checks. One live alert, cleared
            # when the rate recovers.
            if timeout_rate > 0.95:
                if not any("Timeout" in a for a in self.alerts):
                    self.alerts.append(f"WARNING: Timeout rate {timeout_rate:.0%} at step {step}")
            else:
                self.alerts = [a for a in self.alerts if "Timeout" not in a]

        # gh#93: the zero-kills rule used to append a FRESH alert on every check
        # past 500k while kills stayed 0, so a single persistent condition
        # manufactured the 5 alerts that trip the abort verdict on its own (the
        # task5-rerun was killed at step 1.3M this way, and every 30M run to date
        # has had kills_per_episode==0 throughout — combat has not emerged yet).
        # Two guards now:
        #   - kills_expected=False drops the rule outright (kill reward is off,
        #     so zero kills is the configured outcome, not a symptom);
        #   - otherwise at most ONE zero-kills alert is live at a time, so the
        #     rule can contribute to a verdict but never reach it alone.
        # R0-J: timeout and KL are single-live alerts too (see above / below);
        # entropy is the ONLY rule that still accumulates — sustained collapse
        # genuinely is worse the longer it persists, and it is the one rule
        # that cannot be a structural artefact of an untrained policy.
        if self.kills_expected and step > 500_000:
            kills_per_ep = metrics.get("game/kills_per_episode", 1.0)
            if kills_per_ep == 0 and not any("Zero kills" in a for a in self.alerts):
                self.alerts.append(f"WARNING: Zero kills by step {step}")

        if step > 100_000:
            approx_kl = metrics.get("losses/approx_kl", metrics.get("approx_kl", 0.0))
            # R0-J: single live alert (target_kl already clips each epoch, so a
            # persistently high approx_kl is one condition, not five).
            if approx_kl > 0.05:
                if not any("KL" in a for a in self.alerts):
                    self.alerts.append(f"WARNING: KL divergence {approx_kl:.3f} at step {step}")
            else:
                self.alerts = [a for a in self.alerts if "KL" not in a]

        if len(self.alerts) >= 5:
            print("DEAD RUN DETECTED:")
            for alert in self.alerts:
                print(f"  {alert}")
            return True

        return False


# ── SECTION: Self-Play ─────────────────────────────────────────────────────


def self_play_used_past_metric(trainer) -> float:
    """0.0/1.0 for metrics.jsonl. Persist filter drops non-floats.

    WHAT: expose whether this epoch's evaluate() rollout used a past-policy
      opponent (`trainer._selfplay_used_past`, set in
      `_patch_trainer_with_selfplay`).
    WHY: the persist filter on the outer logs dict drops non-floats, so a
      bool never reaches metrics.jsonl. Callers write the returned float
      onto the outer dict next to self_play/pool_size — never under
      losses/. Missing attr (no-selfplay / unpatched path) is 0.0, not
      an error.
    PITFALL: do not log self_play/opponent_id. `load_past_policy` keeps
      the chosen path as a local; a string would also be dropped by the
      persist filter, and inventing a pool schema is out of scope.
    """
    return float(getattr(trainer, "_selfplay_used_past", False))


class SelfPlayManager:
    """Manages a pool of past checkpoints for self-play training.

    Every save_every_epochs epochs (or when the hero team win_rate > win_threshold),
    the current policy is saved to a pool.  With probability p_past, a random past
    checkpoint is used to supply actions for the *opponent* team during rollout
    collection.  This prevents the co-adaptation collapse that arises when both
    teams train against only the latest version of each other.

    Agent layout (per env, 10 agents total):
        slots 0-4  → T team
        slots 5-9  → CT team
    """

    AGENTS_PER_ENV = 10
    T_SLOTS = slice(0, 5)
    CT_SLOTS = slice(5, 10)
    # The team that plays OPPONENT at construction (CT attacks second). A
    # constant, not a bare literal in __init__, because Rung 1a T3 builds the
    # participation vector for the complement team BEFORE any manager exists —
    # see initial_hero_team().
    INITIAL_OPPONENT_TEAM = "ct"

    @classmethod
    def initial_hero_team(cls) -> str:
        """Team the HERO policy plays before any maybe_switch_teams flip.

        Rung 1a T3: build_participating_rows needs this at trainer-construction
        time, which is upstream of the SelfPlayManager instance. Deriving it
        from INITIAL_OPPONENT_TEAM (rather than hardcoding "t" at the call
        site) is what keeps the participation vector and the statue mask from
        silently disagreeing if the initial sides are ever swapped.
        Under --opponent noop the value is constant for the whole run: the mode
        forbids self-play, and maybe_switch_teams is the only thing that flips
        opponent_team.
        """
        return "t" if cls.INITIAL_OPPONENT_TEAM == "ct" else "ct"

    def __init__(
        self,
        pool_size: int = 15,
        p_past: float = 0.3,
        save_every_epochs: int = 25,
        win_threshold: float = 0.6,
        phase_length: int = 50,
        aim_log_std_max=None,
        pin_pitch: bool = False,
        opponent_mode: str = "self",
    ):
        # R0-E (#131): run properties re-applied to every past policy built by
        # load_past_policy (they are non-persistent on the policy, so the
        # snapshot cannot carry them). A past opponent with an unpinned mask
        # would sample a live pitch dim the env ignores — harmless for the
        # env, but its stored logprob_c would include a factor the live
        # policy's does not, and self-play ratio_c would silently drift.
        self.aim_log_std_max = aim_log_std_max
        self.pin_pitch = bool(pin_pitch)
        # Rung 1a T3: "noop" makes the patched evaluate() overwrite this team's
        # actions with the no-op bin on every head (see _patch_trainer_with_
        # selfplay). Validated here so a typo'd mode cannot reach the rollout
        # as a silently-inactive branch. Callers that pass "noop" MUST also
        # have passed assert_opponent_self_play_compatible.
        if opponent_mode not in OPPONENT_MODES:
            raise ValueError(f"opponent_mode={opponent_mode!r} must be one of {OPPONENT_MODES}")
        self.opponent_mode = opponent_mode
        self.pool: list[Path] = []
        self.pool_size = pool_size
        self.p_past = p_past
        self.save_every_epochs = save_every_epochs
        self.win_threshold = win_threshold
        self.phase_length = phase_length
        self.opponent_team = self.INITIAL_OPPONENT_TEAM                # CT is opponent first; T learns to attack
        self._milestone_count = 0
        self._last_save_epoch = -1

    def state_dict(self) -> dict:
        """R0-C (#134): everything a full-state resume must restore.

        Paths are stringified AND resolve()d: the pool is filled with paths
        relative to --checkpoint-dir as given, while --resume-run resolves the
        run dir to absolute — a resume from another cwd would otherwise fail
        every exists() check in load_state_dict and empty the pool. The knobs
        (pool_size, p_past, ...) are NOT saved — they are rebuilt from args and
        guarded by check_resume_config via config.json.
        """
        return {
            "pool": [str(Path(p).resolve()) for p in self.pool],
            "opponent_team": self.opponent_team,
            "_milestone_count": self._milestone_count,
            "_last_save_epoch": self._last_save_epoch,
        }

    def load_state_dict(self, state: dict):
        """Inverse of state_dict. Pool entries whose file vanished are dropped
        (a later past-policy draw would crash on torch.load). opponent_team is
        restored explicitly: rebuilt-at-default would invert every later
        maybe_switch_teams toggle relative to the pre-crash run."""
        self.pool = [Path(p) for p in state["pool"] if Path(p).exists()]
        dropped = len(state["pool"]) - len(self.pool)
        if dropped:
            print(f"[SelfPlay] WARN: dropped {dropped}/{len(state['pool'])} pool entries whose "
                  "file no longer exists")
        print(f"[SelfPlay] pool restored: {len(self.pool)} entries, "
              f"opponent_team={state['opponent_team']}")
        self.opponent_team = state["opponent_team"]
        self._milestone_count = int(state["_milestone_count"])
        self._last_save_epoch = int(state["_last_save_epoch"])

    def maybe_save(
        self,
        policy,
        checkpoint_dir: Path,
        epoch: int,
        win_rate_t: float,
        win_rate_ct: float,
    ):
        """Save current policy to the pool if conditions are met."""
        import torch

        hero_win = win_rate_t if self.opponent_team == "ct" else win_rate_ct
        # Schedule: every save_every_epochs epochs, OR when hero is dominating
        # (win_threshold) but only if enough epochs have passed since last save.
        since_last = epoch - self._last_save_epoch
        scheduled = epoch % self.save_every_epochs == 0
        dominant = hero_win > self.win_threshold and since_last >= self.save_every_epochs // 2
        if scheduled or dominant:
            path = checkpoint_dir / f"sp_{epoch:06d}.pt"
            torch.save(policy.state_dict(), path)
            self._add_to_pool(path)
            self._milestone_count += 1
            self._last_save_epoch = epoch
            print(f"[SelfPlay] Saved checkpoint → {path.name}  "
                  f"(pool={len(self.pool)}, hero_win={hero_win:.2f})")

    def _add_to_pool(self, path: Path):
        self.pool.append(path)
        if len(self.pool) > self.pool_size:
            # Keep every 5th entry as milestone; evict the most recent non-milestone
            non_milestones = [i for i in range(len(self.pool) - 1) if i % 5 != 0]
            evict = non_milestones[-1] if non_milestones else 0
            evicted = self.pool.pop(evict)
            if evicted.exists():
                evicted.unlink(missing_ok=True)

    def maybe_switch_teams(self, epoch: int):
        if epoch > 0 and epoch % self.phase_length == 0:
            old = self.opponent_team
            self.opponent_team = "ct" if self.opponent_team == "t" else "t"
            print(f"[SelfPlay] Epoch {epoch}: opponent {old} → {self.opponent_team}")

    def should_use_past(self) -> bool:
        return bool(self.pool) and random.random() < self.p_past

    def load_past_policy(self, device, vecenv):
        """Load a random past checkpoint. Returns the policy module or None.

        Batch 7 (spec §3.3): the state_dict is read BEFORE build_policy so
        BOTH architecture bits (heads + trunk) can be inferred from its keys.
        This method receives no config and no flag — during a split run the
        pool fills with split snapshots, and a flag-only design would raise
        here on ~30% of epochs (p_past=0.3), hours into the run. Inference
        also lets a split run mix in pre-split snapshots from an older pool.
        """
        import torch

        if not self.pool:
            return None
        path = random.choice(self.pool)
        if not path.exists():
            self.pool.remove(path)
            return None
        state_dict = torch.load(path, map_location=device, weights_only=True)
        # Both bits inferred from keys — this method receives no config.
        policy = build_policy(vecenv,
                              device,
                              tct_split_heads=state_dict_is_split(state_dict),
                              tct_split_trunk=state_dict_is_trunk_split(state_dict),
                              aim_log_std_max=self.aim_log_std_max,
                              pin_pitch=self.pin_pitch)
        load_state_dict_arch_checked(policy, state_dict, source=str(path))
        policy.eval()
        return policy

    def get_opponent_mask(self, batch_n: int, device) -> "torch.Tensor":
        """Bool mask of shape (batch_n,): True for every opponent-team agent slot."""
        import torch

        n_envs = batch_n // self.AGENTS_PER_ENV
        mask = torch.zeros(batch_n, dtype=torch.bool, device=device)
        slots = self.CT_SLOTS if self.opponent_team == "ct" else self.T_SLOTS
        for e in range(n_envs):
            base = e * self.AGENTS_PER_ENV
            mask[base + slots.start:base + slots.stop] = True
        return mask


def _patch_trainer_with_selfplay(trainer, self_play_mgr: SelfPlayManager):
    """Monkey-patch trainer.evaluate() to inject past-policy actions for the opponent team.

    For each evaluation epoch SelfPlayManager.should_use_past() decides (once) whether
    to activate self-play.  When active, a random past checkpoint is loaded and its
    actions+logprobs replace the current-policy outputs for the opponent-team slots in
    the rollout buffer.  The current policy's LSTM state is updated normally; the past
    policy has its own independent LSTM state tensors.

    Training (trainer.train()) sees the overridden actions as if they came from the
    current policy at collection time.  The importance ratio (π_new / π_old) is
    well-defined because we store the *past* policy's logprobs as π_old.

    Rung 1a T3: the patched evaluate() carries a SECOND, unconditional opponent
    override for ``self_play_mgr.opponent_mode == "noop"`` — the stationary
    statue. It is independent of the past-policy branch above (which is dead at
    p_past = 0, the only configuration noop allows) and pairs with the
    hero-team-only participation vector from build_participating_rows.
    """
    import pufferlib
    import pufferlib.pytorch
    import torch

    # Task 6c (utof/cs2rl#9): Batch 1 reward-architecture helpers.
    # Imported lazily here (not at module scope) to keep train.py import
    # cost flat for callers that never hit the self-play path.
    from train_helpers_batch1 import WelfordStd, process_step_rewards

    # Past-policy LSTM state — same dict structure as trainer.lstm_h
    # key → (agents_per_batch, hidden_size)
    past_lstm_h = {k: torch.zeros_like(v) for k, v in trainer.lstm_h.items()}
    past_lstm_c = {k: torch.zeros_like(v) for k, v in trainer.lstm_h.items()}

    # Task 6c: per-channel online-std estimators + segment-level event mask
    # buffers. Attached to the trainer (not closure-local) so downstream
    # tasks (Task 7 aggregation, Task 9 return-norm reset) can read them.
    # prior_std=1.0 + min_count=1000 gives a conservative warmup: channels
    # with few non-zero samples (rare-event, e.g. win/defuse) default to
    # std=1 until we have >=1000 observations — prevents a spuriously small
    # std from blowing up the normalized reward during early training.
    trainer._batch1_welford_combat = WelfordStd(prior_std=1.0, min_count=1000)
    trainer._batch1_welford_objective = WelfordStd(prior_std=1.0, min_count=1000)
    trainer._batch1_welford_positional = WelfordStd(prior_std=1.0, min_count=1000)
    # Segment-level event mask (populated by Task 7; init here so Task 6c's
    # tests don't fail on missing attr and Task 7 can start by just writing).
    # Dimension split:
    #   _batch1_event_mask           — one bool PER SEGMENT (buffer row),
    #                                   consumed when prio_probs sampling picks
    #                                   which completed segments to replay.
    #   _batch1_current_segment_has_event — one bool PER AGENT ROW (= total_agents),
    #                                   live accumulator during rollout; OR'd
    #                                   into the segment row when a segment closes.
    # They're different shapes because one tracks "which rows in the finished
    # buffer contain an event" and the other tracks "does the currently-rolling
    # segment on this agent row contain an event yet".
    _dev = trainer.config["device"]
    trainer._batch1_event_mask = torch.zeros(trainer.segments, dtype=torch.bool, device=_dev)
    trainer._batch1_current_segment_has_event = torch.zeros(trainer.total_agents,
                                                            dtype=torch.bool,
                                                            device=_dev)
    # Host scratch for the batched per-tick reward assembly (see
    # process_step_rewards). Grown on demand to len(info); starts empty because
    # the per-tick env count isn't known until the first recv().
    trainer._batch1_reward_scratch = np.empty(0, dtype=np.float32)

    def _evaluate_with_selfplay(self):
        profile = self.profile
        epoch = self.epoch
        profile("eval", epoch)
        profile("eval_misc", epoch, nest=True)

        cfg = self.config
        dev = cfg["device"]

        if cfg["use_rnn"]:
            for k in self.lstm_h:
                self.lstm_h[k].zero_()
                self.lstm_c[k].zero_()

        # ── Decide self-play for this epoch ────────────────────────────────
        use_past = self_play_mgr.should_use_past()
        past_policy = None
        if use_past:
            past_policy = self_play_mgr.load_past_policy(dev, self.vecenv)
            use_past = past_policy is not None

        # TAG (spec 2026-08-13 §4.2): expose whether THIS epoch's rollout
        # used a past-policy opponent — on those epochs one team's rows are
        # off-policy and cross-team cos-sim measures on-vs-off-policy
        # asymmetry, not T/CT conflict; the analyzer drops them.
        self._selfplay_used_past = bool(use_past)

        if use_past:
            for k in past_lstm_h:
                past_lstm_h[k].zero_()
                past_lstm_c[k].zero_()
        # ───────────────────────────────────────────────────────────────────

        self.full_rows = 0
        while self.full_rows < self.segments:
            profile("env", epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile("eval_misc", epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)
            # Rung 0 §2.2: global_step counts PARTICIPATING agent-steps, so
            # --timesteps means the same thing at any n_active_per_team.
            # `mask` is the recv() chunk's live-agent mask (all-True for this
            # env, vector.py:188); env_id is the agent-row slice bound just
            # above, so the static row flags line up element-for-element.
            self.global_step += int(
                (np.asarray(mask, dtype=bool) & self._participating_rows_np[env_id]).sum())

            profile("eval_copy", epoch)
            o = torch.as_tensor(o)
            o_device = o.to(dev)
            r = torch.as_tensor(r).to(dev)
            d = torch.as_tensor(d).to(dev)

            # F8: pull the C-computed action masks for exactly this batch of
            # agent rows. env_id indexes agent rows, matching the shm layout
            # (num_envs*N_AGENTS, ACTION_MASK_DIM). `!= 0` both converts to
            # bool AND copies — the shm bytes get overwritten by the next
            # worker step, so we must not keep a view. None ⇒ unmasked
            # (legacy trainer built without the mask shm).
            mask_view = getattr(self, "_action_mask_view_main", None)
            action_mask = None
            if mask_view is not None:
                action_mask = torch.as_tensor(mask_view[env_id]).to(dev) != 0

            profile("eval_forward", epoch)
            with torch.no_grad(), self.amp_context:
                state = dict(reward=r, done=d, env_id=env_id, mask=mask)
                if cfg["use_rnn"]:
                    state["lstm_h"] = self.lstm_h[env_id.start]
                    state["lstm_c"] = self.lstm_c[env_id.start]

                # Batch 3 (T5) + Fix #1: hybrid rollout. Policy returns 4-tuple
                # (logits, mu_aim, log_std, value). _hybrid_sample_logits now
                # returns the per-factor log-prob halves directly (6-tuple),
                # eliminating the previous double-construction of 7 Categorical
                # + 1 Normal at the rollout site (was +437 ms/epoch on the
                # smoke benchmark per perf investigation post-PR #28).
                logits, mu_aim, log_std_aim, value = self.policy.forward_eval(o_device, state)
                action, cont_action, logprob_d, logprob_c, _, _ = _hybrid_sample_logits(
                    (logits, mu_aim, log_std_aim, value),
                    max_turn_speed=self.policy.max_turn_speed.item(),
                    mask=action_mask,
                    aim_dim_mask=getattr(self.policy, "aim_dim_mask", None),
                )
                # Joint log-prob for self.logprobs (back-compat slot read by
                # PufferLib's diagnostics + the KL/clipfrac path). Per-factor
                # halves go to self.logprobs_d / self.logprobs_c for the
                # H-PPO clip in _hybrid_ppo_loss.
                logprob = logprob_d + logprob_c

                # ── Task 6c: per-channel reward norm + symlog (replaces the
                # old hard-clip of r to [-1, 1]). Pipeline:
                #   step_stats (Task 6a info payload)
                #     → split_into_channels (Task 4)
                #     → WelfordStd.update + normalize per channel (Task 5)
                #     → sum channels → symlog (Task 4) → r written to buffer
                #
                # Info shape: PufferLib's Serial/Multiprocessing backend
                # collects info with list-extend semantics (pufferlib/vector.py
                # ~L149-153). Cs2Env returns `[{"step_stats": view}]` per tick
                # so `len(info)` is the number of envs in this batch, while
                # r.shape[0] == len(info) * agents_per_env (10 for Cs2Env).
                # All 10 agents in an env share the same step_stats because
                # step_stats aggregates team-level reward fields; we update
                # Welford ONCE per env (not per agent — that would over-count
                # by 10x) and apply the same symlog'd channel sum to every
                # agent row in that env.
                #
                # Fallback: if info[e] lacks step_stats (flag off OR an older
                # info entry that predates Task 6a), pass raw r through for
                # that env's rows unchanged — the minimal-disruption path if
                # the flag gets toggled or an upstream change sneaks through.
                #
                # Task 7: process_step_rewards also ORs the per-tick
                # bomb_planted flag into _batch1_current_segment_has_event for
                # every agent row in the env. The C side sets ss->bomb_planted
                # only on the transition tick (process_bomb, cs2_bomb.h — guarded by
                # `if g->bomb_plant_ticks >= sd->bomb_plant_time`) and StepStats
                # is cleared every step via clear_stats(ss) at the top of
                # env_step (cs2_env.h), so the field is already a per-tick delta (1 only
                # on the plant tick) — NO edge-trigger needed. All 10 agent rows
                # in an env share the event state; it is flushed into
                # _batch1_event_mask at the segment boundary below.
                #
                # PERF: the per-env loop lives in process_step_rewards() and
                # builds the tick's rewards in a host float32 scratch buffer, so
                # the device sees one H2D copy + one symlog per tick instead of
                # three single-scalar torch.tensor() constructions per env
                # (~213k launch-bound CUDA ops/epoch at 256 envs x 64 ticks,
                # inside this timed eval_forward region). The helper's docstring
                # carries the bit-exactness invariants — read it before touching
                # the arithmetic.
                agents_per_env_local = self.vecenv.driver_env.num_agents
                if self._batch1_reward_scratch.shape[0] < len(info):
                    self._batch1_reward_scratch = np.empty(len(info), dtype=np.float32)
                r = process_step_rewards(
                    info,
                    r,
                    agents_per_env_local,
                    self._batch1_welford_combat,
                    self._batch1_welford_objective,
                    self._batch1_welford_positional,
                    self._batch1_reward_scratch,
                    current_segment_has_event=self._batch1_current_segment_has_event,
                )

                # ── SELF-PLAY: override opponent-team actions ───────────────
                if use_past:
                    batch_n = o_device.shape[0]
                    opp_mask = self_play_mgr.get_opponent_mask(batch_n, dev)
                    opp_idx = torch.where(opp_mask)[0]

                    past_state = {
                        "done": d[opp_mask],
                        "lstm_h": past_lstm_h[env_id.start][opp_mask],
                        "lstm_c": past_lstm_c[env_id.start][opp_mask],
                    }
                    # Batch 3 (T5) + Fix #1: past policy is a HybridPolicy too;
                    # same 4-tuple contract. _hybrid_sample_logits now surfaces
                    # the per-factor log-prob halves directly (6-tuple), so we
                    # no longer reconstruct 7 Categorical + 1 Normal here. The
                    # rollout buffer entries stored at this opponent slot stay
                    # consistent with the current-policy branch (PPO update
                    # treats them indistinguishably).
                    opp_logits, opp_mu, opp_log_std, _opp_value = past_policy.forward_eval(
                        o_device[opp_mask], past_state)
                    (opp_action, opp_cont_action, opp_logprob_d, opp_logprob_c, _,
                     _) = _hybrid_sample_logits(
                         (opp_logits, opp_mu, opp_log_std, None),
                         max_turn_speed=past_policy.max_turn_speed.item(),
                         mask=action_mask[opp_mask] if action_mask is not None else None,
                         aim_dim_mask=getattr(past_policy, "aim_dim_mask", None),
                     )
                    opp_logprob = opp_logprob_d + opp_logprob_c

                    # Write back updated past-policy LSTM states (cast from fp16 if needed)
                    past_lstm_h[env_id.start][opp_mask] = past_state["lstm_h"].to(
                        past_lstm_h[env_id.start].dtype)
                    past_lstm_c[env_id.start][opp_mask] = past_state["lstm_c"].to(
                        past_lstm_c[env_id.start].dtype)

                    # Replace opponent slots in action & logprob buffers.
                    # Cast to destination dtype (amp_context may yield fp16).
                    # Continuous action and per-factor logprobs are also
                    # spliced in so train()'s _hybrid_ppo_loss sees consistent
                    # mb_cont_actions / mb_old_logp_{d,c} for opponent rows.
                    action[opp_idx] = opp_action.to(action.dtype)
                    logprob[opp_idx] = opp_logprob.to(logprob.dtype)
                    cont_action[opp_idx] = opp_cont_action.to(cont_action.dtype)
                    logprob_d[opp_idx] = opp_logprob_d.to(logprob_d.dtype)
                    logprob_c[opp_idx] = opp_logprob_c.to(logprob_c.dtype)
                # ──────────────────────────────────────────────────────────

                # ── STATUE OPPONENT (--opponent noop, Rung 1a T3) ──────────
                # UNCONDITIONAL branch, deliberately NOT nested in the
                # `if use_past:` splice above: that branch never runs at
                # p_past = 0, which is exactly the configuration noop demands
                # (assert_opponent_self_play_compatible). It sits AFTER the
                # splice so the statue would win if both were ever live, and
                # BEFORE both the buffer scatter and vecenv.send below — the
                # same tensors feed the rollout buffer and the env.
                #
                # Bin 0 on every discrete head is the no-op action by
                # construction (cs2_env.h:51-63; move_dir == 0 is genuinely
                # stationary, valid_dir in process_movement) and is never masked out by
                # the C-side action mask, so this cannot sample an illegal
                # action. cont_action = 0 means zero Δyaw/Δpitch: the statue
                # keeps its spawn orientation.
                #
                # The stored logprobs go to 0 for the same reason the past-
                # policy splice rewrites them: they are the π_old the PPO
                # update would divide by. Under noop these rows are
                # non-participating, so every loss masks them out anyway —
                # this keeps the buffer self-consistent rather than carrying
                # log-probs of actions that were never sampled.
                if self_play_mgr.opponent_mode == "noop":
                    opp_idx = torch.where(self_play_mgr.get_opponent_mask(o_device.shape[0],
                                                                          dev))[0]
                    action[opp_idx] = 0
                    cont_action[opp_idx] = 0
                    logprob[opp_idx] = 0
                    logprob_d[opp_idx] = 0
                    logprob_c[opp_idx] = 0
                    # Redundant with the participation scatter below (which
                    # multiplies values by the row flag) and kept anyway: the
                    # statue's critic output must never bootstrap GAE, no
                    # matter which of the two masks a future edit touches.
                    value[opp_idx] = 0
                # ──────────────────────────────────────────────────────────

            profile("eval_copy", epoch)
            with torch.no_grad():
                if cfg["use_rnn"]:
                    self.lstm_h[env_id.start] = state["lstm_h"]
                    self.lstm_c[env_id.start] = state["lstm_c"]

                seq_pos = self.ep_lengths[env_id.start].item()
                batch_rows = slice(
                    self.ep_indices[env_id.start].item(),
                    1 + self.ep_indices[env_id.stop - 1].item(),
                )

                if cfg["cpu_offload"]:
                    self.observations[batch_rows, seq_pos] = o
                else:
                    self.observations[batch_rows, seq_pos] = o_device

                self.actions[batch_rows, seq_pos] = action
                self.logprobs[batch_rows, seq_pos] = logprob
                # Batch 3 (T5): parallel writes for the new buffers added by
                # _patch_trainer_with_hybrid_aim. The PPO update (the replacement
                # train() body, `_train_with_return_norm` in src/train_update.py:
                # `mb_cont_actions = self.cont_actions[idx]` and the two logprob
                # reads beside it) reads these by the same idx; missing this write would
                # silently feed zeros to _hybrid_ppo_loss → ratio_c always
                # equals exp(new_logp_c - 0), which would diverge.
                self.cont_actions[batch_rows, seq_pos] = cont_action
                self.logprobs_d[batch_rows, seq_pos] = logprob_d
                self.logprobs_c[batch_rows, seq_pos] = logprob_c
                # F8: persist the masks the sampler just used so the PPO
                # update (mb_masks in _hybrid_ppo_loss) recomputes logprobs
                # over the identical masked distribution. Skipped when
                # unmasked — the buffer's all-ones default is the no-op mask.
                if action_mask is not None:
                    self.action_masks[batch_rows, seq_pos] = action_mask
                self.rewards[batch_rows, seq_pos] = r
                self.terminals[batch_rows, seq_pos] = d.float()
                # Rung 0 §2.2: scatter the static row flag into buffer layout,
                # and zero the critic output on parked rows (defence in depth —
                # the masked reductions in train() are what make it correct;
                # this just keeps GAE from propagating a bootstrap value
                # through rows whose reward is identically 0).
                # OUTSIDE the `if action_mask is not None:` guard above ON
                # PURPOSE: inside it, `participating` would stay all-zero for a
                # mask-less run and the per-epoch any() assert would fire.
                _part_rows = self._participating_rows[env_id]
                self.participating[batch_rows, seq_pos] = _part_rows
                self.values[batch_rows, seq_pos] = value.flatten() * _part_rows.to(value.dtype)

                self.ep_lengths[env_id] += 1
                if seq_pos + 1 >= cfg["bptt_horizon"]:
                    num_full = env_id.stop - env_id.start
                    # Task 7: flush the live event accumulator → segment mask
                    # BEFORE overwriting ep_indices. Each agent row's current
                    # segment index lives in self.ep_indices[env_id]; once we
                    # reassign ep_indices to (free_idx + arange(num_full)) a
                    # few lines down, the old segment index is lost. Clone
                    # first, write to _batch1_event_mask at those OLD slots,
                    # then reset the live accumulator so the next segment
                    # starts clean. Pitfall: writing AFTER the re-index would
                    # clobber freshly-allocated future segments (off-by-one
                    # bug that would silently mark the wrong rollout rows).
                    old_seg_indices = self.ep_indices[env_id].clone().long()
                    self._batch1_event_mask[old_seg_indices] = (
                        self._batch1_current_segment_has_event[env_id])
                    self._batch1_current_segment_has_event[env_id] = False
                    self.ep_indices[env_id] = (self.free_idx +
                                               torch.arange(num_full, device=dev).int())
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    import numpy as _np

                    lo, hi = self.vecenv.action_space.low, self.vecenv.action_space.high
                    action = _np.clip(action, lo, hi)

            profile("eval_misc", epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)

            profile("env", epoch)
            # Batch 3 (T5/T5b): vecenv.send patched by
            # _patch_trainer_with_hybrid_aim to accept (action, cont_action)
            # tuple. Discrete action is the numpy int32 buffer that the C
            # env still receives positionally. For the Serial backend
            # cont_action is forwarded to env.step's continuous_actions
            # kwarg via the per-env step wrapper. For the Multiprocessing
            # backend cont_action is mirrored into a multiprocessing.RawArray
            # shm view by _hybrid_send before orig_send runs, so workers
            # see the same Δyaw on their next Cs2Env.step via the per-env
            # numpy view installed by _attach_cont_action_view (see the
            # _patch_trainer_with_hybrid_aim docstring for the shm pattern).
            self.vecenv.send((action, cont_action))

        profile("eval_misc", epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=dev, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    trainer.evaluate = types.MethodType(_evaluate_with_selfplay, trainer)
    print("[Train] Self-play evaluate patch enabled.")
    return trainer


# ── SECTION: Batch 3 hybrid PPO helpers ────────────────────────────────────
#
# These three functions are the trainer-side complement of T4's HybridPolicy
# (mu_aim + log_std_aim Gaussian head bolted onto 7 categorical heads).
#
#   _hybrid_sample_logits  — the rollout-time replacement for
#                            pufferlib.pytorch.sample_logits(logits[, action]).
#                            Pure function; takes the 4-tuple emitted by
#                            HybridPolicy.forward / forward_eval and produces
#                            (action, continuous_action, log_prob, entropy).
#                            Joint factorised log-prob = sum of categorical
#                            log-probs + Normal log-prob (independence
#                            assumption per spec L8).
#
#   _hybrid_ppo_loss       — the PPO-update-time replacement, doing the
#                            forward pass + per-factor clipped policy loss
#                            (Fan et al. IJCAI 2019 H-PPO baseline). Returns
#                            split discrete/continuous ratios so the caller
#                            can keep KL/clipfrac diagnostics on the discrete
#                            half (back-compat with the existing log surface).
#
#   _patch_trainer_with_hybrid_aim — extends the rollout buffer with
#                            cont_actions / logprobs_d / logprobs_c parallel
#                            to the existing actions / logprobs, and wraps
#                            vecenv.send (the wrapper calls the original) to
#                            forward the float buffer to the env. train()
#                            applies it AFTER _patch_trainer_with_return_norm
#                            (which REPLACES train() via types.MethodType and
#                            never calls the stock body) and BEFORE
#                            _patch_trainer_with_selfplay (which REPLACES
#                            evaluate() the same way). The dependency is at
#                            CALL time, not patch time: the replacement
#                            train()/evaluate() bodies read the buffers this
#                            patcher allocates, so all three must be applied
#                            before the first evaluate()/train() call.


def _hybrid_sample_logits(policy_out,
                          action=None,
                          continuous_action=None,
                          max_turn_speed=None,
                          mask=None,
                          aim_dim_mask=None):
    """Hybrid sampler for the 4-tuple HybridPolicy output (Batch 3 task 5).

    Replaces the four in-tree usages of
    ``pufferlib.pytorch.sample_logits(logits[, action=...])`` that previously
    assumed a 2-tuple policy contract. Pure function (no monkey-patching) so
    it can be unit-tested without spinning up a trainer.

    Inputs
    ------
    policy_out : 4-tuple
        (logits_list[7], mu_aim, log_std_aim, value) — the canonical T4
        output of HybridPolicy.forward / forward_eval. The value slot is
        ignored here; the caller already has it from the original call.
    action : (B, ACTION_DIM=7) int64 tensor, or None
        If None, sample fresh from the categorical heads. If supplied
        (PPO update pass), evaluate log-prob under the new policy without
        re-sampling — this is the difference between rollout and update.
    continuous_action : (B, AIM_DIM=2) float32 tensor, or None
        If None, sample (Δyaw, Δpitch) from Normal(mu_aim, exp(log_std_aim))
        and clamp to ±max_turn_speed. If supplied, evaluate log-prob without
        re-sampling.
    max_turn_speed : float or None
        Hard clamp on sampled Δyaw. None means no clamp (only sane in the
        update-pass path where continuous_action is provided pre-clamped).
    mask : (B, ACTION_MASK_DIM) bool/int8 tensor, or None (F8)
        C-computed action masks (1 = valid; see cs2_env.h compute_masks).
        When given, invalid bins are excluded from sampling AND from the
        log-prob/entropy — the distribution IS the masked distribution, so
        the stored logprobs stay consistent with _hybrid_ppo_loss as long as
        the update pass receives the SAME mask (mb_masks). None = unmasked
        (legacy eval/record callers that have no mask plumbing).
    aim_dim_mask : (AIM_DIM,) float tensor, or None (R0-E.2, #131)
        Per-dimension weight on the Gaussian log-prob / entropy terms, applied
        BEFORE the sum over AIM_DIM. [1, 0] when pitch is pinned (the env
        ignores cont[:, 1], so its density must not enter the ratio). None ⇒
        all-ones ⇒ today's behaviour bit-for-bit. Sampling is NOT masked —
        the pinned dim is still drawn (and discarded by the env).

    Returns
    -------
    action, continuous_action, log_prob_d, log_prob_c, entropy_d, entropy_c
        action : (B, 7) int64
        continuous_action : (B, AIM_DIM=2) float32, ∈ [-max_turn_speed, max_turn_speed]
        log_prob_d : (B,) — discrete factor log-prob (sum over 7 categoricals).
            The PPO loss assembly in _hybrid_ppo_loss applies the clip to
            this half independently of log_prob_c (Fan et al. 2019 Eq 8).
            Callers that want the rollout-stored joint log_prob simply do
            `log_prob_d + log_prob_c` (joint factorised under independence,
            spec L8) — surfacing the halves directly here saves the rollout
            from reconstructing 7 Categorical + 1 Normal a second time.
        log_prob_c : (B,) — continuous factor log-prob (Normal sum-of-dims).
        entropy_d : (B,) — discrete entropy (sum of 7 categoricals).
        entropy_c : (B,) — Normal entropy 0.5·log(2πe·σ²). NEGATIVE for
            σ < 1/√(2πe) ≈ 0.242 — at σ_init=0.1 it is ≈ −0.886. This is
            mathematically correct; do NOT clip or assert entropy >= 0
            anywhere downstream. Callers that don't need entropy can ignore
            with `*_entropies` unpacking.

    Pre-Fix#1 (Batch 3 T5) this returned the SUMMED log_prob and SUMMED
    entropy as a 4-tuple, forcing the rollout caller to reconstruct
    distributions to recover the per-factor halves for self.logprobs_d /
    self.logprobs_c. That double-construction was measured at +437 ms/epoch
    on the i7-9750H smoke (16.0 ms/step actual vs 9.1 ms/step minimal).
    Surfacing the halves directly drops that cost to ~9.1 ms/step.
    """
    import torch
    import torch.nn.functional as F

    logits_list, mu_aim, log_std_aim, _value = policy_out

    # F8: mask BEFORE log_softmax so sampling, log-prob and entropy all see
    # the same (masked) distribution. Dead agents collapse to deterministic
    # per-head no-ops (entropy 0) instead of burning exploration samples.
    if mask is not None:
        logits_list = _apply_action_masks(logits_list, mask)

    # ── Discrete: 7 independent categorical heads — hand-rolled (Fix #2) ──
    # We avoid `torch.distributions.Categorical` because constructing 7 of them
    # per rollout step (×64 bptt × ~12 epochs/sec) accumulates measurable
    # Python-side overhead. The math is straightforward:
    #   sample(logits) ≡ multinomial(softmax(logits), 1)
    #   log_prob(a)    ≡ log_softmax(logits)[a]
    #   entropy()      ≡ -Σ p · log_softmax  where p = exp(log_softmax)
    # Bench at production batch=2560 measured 9.59 ms → 5.91 ms / step
    # (1.62× speedup, ~235 ms/epoch saved). Numerical equivalence vs
    # torch.distributions: |Δ| ≤ 1.91e-6 (different reduction order in
    # softmax; well within the fp32 tolerance PPO already runs at).
    log_probs_per_head = [F.log_softmax(lg, dim=-1) for lg in logits_list]
    if action is None:
        action = torch.stack(
            [torch.multinomial(lp.exp(), 1).squeeze(-1) for lp in log_probs_per_head],
            dim=-1,
        )
    log_prob_d = sum(
        lp.gather(-1, action[..., i:i + 1]).squeeze(-1) for i, lp in enumerate(log_probs_per_head))
    # Entropy: H = -Σ p log p. log_softmax already gives log p; multiply by
    # exp(log_softmax) = p. Single pass per head, no extra softmax call.
    entropy_d = sum(-(lp.exp() * lp).sum(-1) for lp in log_probs_per_head)

    # ── Continuous: 1D Gaussian aim head — hand-rolled (Fix #2) ──
    # σ comes pre-clamped from forward()/forward_eval() (LOG_STD_MIN/MAX), so
    # we don't re-clamp here — would silently mask a regression in the policy
    # if the clamp were removed upstream.
    # Analytic forms (replace torch.distributions.Normal):
    #   sample(μ, σ)   = μ + σ · randn_like(μ)         (vs rsample; no autograd
    #                                                    graph, PPO doesn't use
    #                                                    pathwise gradients)
    #   log_prob(x)    = -½((x-μ)/σ)² - log σ - ½ log 2π
    #   entropy()      = ½ + ½ log 2π + log σ
    sigma = torch.exp(log_std_aim).expand_as(mu_aim)
    if continuous_action is None:
        continuous_action = mu_aim + sigma * torch.randn_like(mu_aim)
        if max_turn_speed is not None:
            # Same clamp logic as HybridPolicy.get_action_and_value (T4).
            # The C env (env_step, cs2_env.h) clamps silently with fminf/fmaxf;
            # storing the post-clamp value keeps the PPO ratio honest.
            continuous_action = torch.clamp(continuous_action, -max_turn_speed, max_turn_speed)
    diff = (continuous_action - mu_aim) / sigma
    # log_std_aim has shape (AIM_DIM,); expand_as(mu_aim) broadcasts to (B, AIM_DIM)
    # so .sum(-1) sums over AIM_DIM correctly.
    log_std_b = log_std_aim.expand_as(mu_aim)
    # R0-E.2: per-dimension weight (AIM_DIM,), ones ⇒ today's behaviour.
    # Applied BEFORE .sum(-1) so both log-prob and entropy exclude pinned dims.
    w = _aim_dim_weight(aim_dim_mask, mu_aim)
    log_prob_c = ((-0.5 * diff * diff - log_std_b - 0.5 * _LOG_2PI) * w).sum(-1)
    entropy_c = ((0.5 + 0.5 * _LOG_2PI + log_std_b) * w).sum(-1)

    return action, continuous_action, log_prob_d, log_prob_c, entropy_d, entropy_c


def _patch_trainer_with_hybrid_aim(trainer,
                                   cont_action_view_main=None,
                                   mask_view_main=None,
                                   participating_rows=None):
    """Extend trainer with continuous-action rollout storage + vecenv plumbing.

    train() applies this AFTER _patch_trainer_with_return_norm (which
    REPLACES train() via types.MethodType; it does not wrap the stock body)
    and BEFORE the first evaluate()/train() call. The ordering is a
    CALL-time dependency, not a patch-time one: the replacement train()
    body reads self.cont_actions / self.logprobs_{d,c}, which this patcher
    allocates, and nothing at patch time checks they exist. The
    PPO-update-side rewrites live in src/train_update.py (_hybrid_ppo_loss,
    called from _train_with_return_norm); this patcher only handles the
    rollout/storage side.

    Multiprocessing vecenv path (Batch 3 T5b)
    ─────────────────────────────────────────
    PufferLib's Multiprocessing vecenv uses ``multiprocessing.RawArray`` for
    its shm dict and forks workers AFTER allocation (see
    ``.venv/.../pufferlib/vector.py:300-346``). Workers inherit the OS
    shared mapping, so main process and workers see the same physical bytes
    via different numpy views.

    To carry continuous (Δyaw) actions across the fork boundary we mirror
    that pattern — train() allocates its own RawArray BEFORE
    pufferlib.vector.make and threads it through env_kwargs to each env's
    ``Cs2Env._attach_cont_action_view``. The trainer-side numpy view over
    the SAME RawArray is passed in here as ``cont_action_view_main``;
    inside the patched ``_hybrid_send`` we write the policy's Δyaw sample
    into it ``BEFORE`` calling ``orig_send(action)``. Workers' next
    ``Cs2Env.step`` reads the data via their attached view, returning the
    correct value from ``_prepare_continuous_actions(None)``.

    Backwards compat: if ``cont_action_view_main`` is None (legacy callers
    such as the test harness that builds a trainer without the shm path),
    only the in-process Serial wrapper at the bottom of this function is
    used. The Serial path uses a ``trainer.vecenv._cont_action_buf`` Python
    attr stash + a per-env step wrapper, untouched from T5.

    ``participating_rows`` (Rung 0 §2.2): bool array over agent rows saying
    which ones the C env actually spawns. Allocated here rather than in a
    patcher of its own because the buffers it sizes (`trainer.participating`)
    must match `trainer.actions`, which this function already mirrors — and
    because the rollout writes that fill it live in the evaluate() this
    patcher's sibling installs. None ⇒ every row participates.
    """
    import torch
    # Stash on the trainer so _hybrid_send (defined below) can close over
    # it via attribute access. Storing on trainer (not closure-captured
    # local) keeps it visible to instrumentation/inspection.
    trainer._cont_action_view_main = cont_action_view_main
    # F8: main-process numpy view over the mask shm (env→trainer direction;
    # see Cs2Env._attach_mask_view). _evaluate_with_selfplay reads rows for
    # the recv'd env_id slice right after recv() — the workers finished their
    # step by then, so the bytes are the masks for the obs batch in hand.
    # None ⇒ rollout runs unmasked (legacy callers without shm plumbing) and
    # action_masks stays all-ones, which makes the update path a no-op mask.
    trainer._action_mask_view_main = mask_view_main

    # ── Rollout buffer extension (step 5.4) ──
    # self.actions has shape (segments, bptt_horizon, ACTION_DIM=7) int32 —
    # we mirror with AIM_DIM trailing dim, float32. self.logprobs is
    # (segments, bptt_horizon) float32; we add per-factor halves with the
    # same shape so the caller can fetch self.logprobs_d[idx] etc. without
    # any reshaping.
    trainer.cont_actions = torch.zeros(
        (*trainer.actions.shape[:-1], AIM_DIM),
        dtype=torch.float32,
        device=trainer.actions.device,
    )
    trainer.logprobs_d = torch.zeros_like(trainer.logprobs)
    trainer.logprobs_c = torch.zeros_like(trainer.logprobs)
    # F8: per-step action masks, parallel to actions but ACTION_MASK_DIM wide.
    # Initialised to ONES (= everything valid): rows never written (mask shm
    # absent, or rollout rounds that don't fill every segment) degrade to the
    # exact pre-F8 unmasked behaviour instead of masking everything to the
    # no-op. bool keeps the buffer small (segments × 64 × 22 bytes).
    trainer.action_masks = torch.ones(
        (*trainer.actions.shape[:-1], ACTION_MASK_DIM),
        dtype=torch.bool,
        device=trainer.actions.device,
    )

    # ── Rung 0 §2.2: per-row participation ───────────────────────────────
    # participating_rows: numpy bool (total_agents,) — STATIC per run, derived
    # from args.n_active_per_team in train(). None ⇒ all rows participate
    # (harness default, exact identity with pre-Rung-0 behaviour).
    # trainer.participating is the BUFFER-LAYOUT flag [segments, bptt],
    # scattered by evaluate() via ep_indices exactly like action_masks.
    # ZERO-initialised (unlike action_masks, which defaults to all-ones): a
    # skipped write must mask EVERYTHING and trip the per-epoch
    # `participating.any()` assert in train(), never silently train on parked
    # rows. _participating_rows_np is kept beside the torch copy because
    # evaluate()'s global_step accounting works on the numpy `mask` recv()
    # returns; converting per-recv would allocate on every rollout tick.
    n_rows = trainer.total_agents
    if participating_rows is None:
        participating_rows = np.ones(n_rows, dtype=bool)
    participating_rows = np.asarray(participating_rows, dtype=bool).reshape(-1)
    assert participating_rows.shape == (n_rows, ), (participating_rows.shape, n_rows)
    assert participating_rows.any(), "no participating rows — n_active_per_team=0?"
    trainer._participating_rows_np = participating_rows
    trainer._participating_rows = torch.as_tensor(participating_rows, device=trainer.actions.device)
    trainer.participating = torch.zeros(trainer.actions.shape[:-1],
                                        dtype=torch.bool,
                                        device=trainer.actions.device)

    # ── vecenv.send patch (step 5.5) ──
    # Goal: forward both the int discrete buffer and the float cont buffer
    # to env.step(). _evaluate_with_selfplay calls self.vecenv.send(action)
    # with a numpy int array; we change that callsite to send a tuple
    # (action, cont_action) and the wrapper here unpacks. For the
    # Multiprocessing backend cont_action lands on a vecenv-local stash
    # only — see class docstring. For Serial, we forward via positional
    # kwarg into env.step(actions, continuous_actions=...).
    orig_send = trainer.vecenv.send

    def _hybrid_send(action_pair):
        """vecenv.send(...) wrapper accepting (action, cont_action) tuple.

        Backwards-compatible with bare ndarrays so legacy callers (e.g. the
        record path) continue to work — cont_action defaults to None which
        makes Cs2Env.step fall back to its zero scratch buffer.

        T5b dual-path:
        - Serial backend: stash on `vecenv._cont_action_buf`; the per-env
          step wrapper installed below reads it and forwards to
          `Cs2Env.step(continuous_actions=...)`.
        - Multiprocessing backend: also write the same buffer into the
          shared-memory view (`trainer._cont_action_view_main`). Workers
          read via `Cs2Env._cont_action_view` on the very next step.
        Doing BOTH covers the test harness (which uses Serial wrapped in
        the patcher) AND production training (Serial or MP).
        """
        if isinstance(action_pair, tuple):
            action, cont_action = action_pair
        else:
            action, cont_action = action_pair, None
        if cont_action is not None and hasattr(cont_action, 'cpu'):
            cont_action = cont_action.cpu().numpy().astype(np.float32, copy=False)
        # Stash on the vecenv so the Serial backend's send path (below) and
        # any custom step wrapper can pull it. None on a non-Serial path is
        # the documented fallback (zero Δyaw → no turning).
        trainer.vecenv._cont_action_buf = cont_action
        # T5b: mirror the cont buffer into the shared-memory window so MP
        # workers' attached views see the latest sample. We DELIBERATELY
        # write all-zeros when cont_action is None so a stale prior write
        # doesn't bleed into the next tick. The view shape is
        # (num_envs * N_AGENTS, AIM_DIM) — same flatten as the per-env
        # cont_action that comes from _hybrid_sample_logits.
        view = trainer._cont_action_view_main
        if view is not None:
            if cont_action is None:
                view.fill(0.0)
            else:
                # cont_action shape may be (total_agents, AIM_DIM) or
                # already flat. We assert total element count matches
                # view.shape before reshape — this catches a future
                # rollout-side shape change loudly instead of silently
                # broadcasting (project style: strict shape validation,
                # see _prepare_continuous_actions).
                assert cont_action.size == view.size, (
                    f"cont_action.size={cont_action.size} but "
                    f"view.size={view.size} (view.shape={view.shape})")
                view[:] = cont_action.reshape(view.shape)
        return orig_send(action)

    trainer.vecenv.send = _hybrid_send

    # ── Serial backend: extend send() to actually forward cont_action ──
    # PufferLib's Serial.send loops env.step(atns) → we monkey-patch the
    # individual env step to consult vecenv._cont_action_buf and forward
    # the matching slice to Cs2Env.step(actions, continuous_actions=...).
    # On Multiprocessing, trainer.vecenv has no .envs attribute — skip.
    if hasattr(trainer.vecenv, 'envs'):
        envs = trainer.vecenv.envs
        agents_per_env = trainer.vecenv.driver_env.num_agents
        # Pre-compute per-env cont slices once so the wrapper closure is O(1).
        for env_idx, env in enumerate(envs):
            row_start = env_idx * agents_per_env
            row_end = row_start + agents_per_env
            orig_step = env.step

            def _make_step_wrapper(orig, rs, re):

                def _hybrid_env_step(actions):
                    cont_buf = getattr(trainer.vecenv, '_cont_action_buf', None)
                    cont = None
                    if cont_buf is not None:
                        # cont_buf is a flat numpy array shaped
                        # (total_agents, AIM_DIM) — slice this env's chunk.
                        cont = cont_buf[rs:re]
                    return orig(actions, continuous_actions=cont)

                return _hybrid_env_step

            env.step = _make_step_wrapper(orig_step, row_start, row_end)

    print("[Train] Hybrid-aim trainer patch enabled "
          f"(cont_actions buffer={trainer.cont_actions.shape}, "
          f"vecenv_kind={type(trainer.vecenv).__name__}).")
    return trainer


# ── SECTION: PufferLib training ────────────────────────────────────────────


def train(args):
    """Run PPO training via PufferLib 3.0."""
    import pufferlib.vector
    import torch

    # Fix #2 (perf): disable torch.distributions argument validation globally.
    # Most of our hot paths replaced torch.distributions with hand-rolled
    # log_softmax+gather + analytic Normal already, but a few diagnostic /
    # legacy paths (e.g. NaN-guard sanity prints, exploratory test paths) still
    # construct distributions. validate_args=False removes the per-call
    # constraint check overhead for those residual sites at zero risk —
    # validation is purely a sanity check and any production code passes
    # validated inputs by construction. Per the perf research subagent:
    # PyTorch issue #11747 / #30968 confirmed Categorical's structural
    # overhead is the logits.logsumexp allocation in __init__, NOT the
    # validation; this toggle gives the residual ~few-percent gain on
    # whatever still routes through torch.distributions.
    torch.distributions.Distribution.set_default_validate_args(False)

    # Load .env from repo root if present (sets WANDB_* vars picked up by wandb)
    _env_file = Path(__file__).parent.parent / ".env"
    if _env_file.exists():
        for _line in _env_file.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

    device = args.device

    if getattr(args, "name", None):
        resolved_name = resolve_run_name(args.name)
        args.checkpoint_dir = str(CHECKPOINTS_DIR / resolved_name)
        print(f"[Train] Run name resolved to: {resolved_name}")

    # ── R0-E.2 (#131): pin_pitch resolution ────────────────────────────────
    # resolve_pin_pitch loads the REAL map when args.map_data is None (the
    # `--dust2` CLI path) and decides from geometry; see pin_pitch_for_map for
    # why the sentinel itself must never decide. MUST run (a) before
    # build_train_env_factory (env_config_from_args bakes args.pin_pitch into
    # the EnvConfig every worker env is built with) and (b) BEFORE the
    # --resume-run config guard below: the CLI default is pin_pitch=None, which
    # EnvConfig normalises to the un-pinned flag value, while a
    # pinned run's config.json holds 1 — resolving after the guard refused
    # every flag-less resume of a flat-map run (Task 12 ruling). The CLI
    # already resolved it above --dump-config; here it is a cache-safe
    # cross-check for programmatic callers.
    resolve_pin_pitch(args, verbose=False)
    # R0-D: refuse an out-of-range --seed BEFORE W&B init / metrics.jsonl open
    # (the real call is in _per_env_kwargs below).
    env_seed_base(args.seed)
    # Rung 1a T3: same fail-early rationale for --opponent noop + self-play.
    # main() already refused it above the --dump-config exit; repeated here so
    # a programmatic train(args) cannot start a run whose statue team would be
    # swapped out from under the participation vector at epoch 50.
    _opponent_mode = resolve_opponent_mode(args)
    assert_opponent_self_play_compatible(_opponent_mode, bool(getattr(args, "self_play", True)))

    # ── R0-C (#134): --resume-run resolution (before run_label / metrics / config) ──
    resume_run = getattr(args, "resume_run", None)
    _resume_paths = None
    if resume_run:
        if getattr(args, "resume", None):
            raise SystemExit("[Resume] --resume and --resume-run are mutually exclusive")
        _run_dir = Path(resume_run).resolve()
        # --checkpoint-dir has default=None in the CLI precisely so this check
        # can tell "given" from "omitted"; None → CHECKPOINTS_DIR is resolved
        # AFTER this block.
        if args.checkpoint_dir is not None and Path(args.checkpoint_dir).resolve() != _run_dir:
            raise SystemExit(f"[Resume] --checkpoint-dir {args.checkpoint_dir} disagrees with "
                             f"--resume-run {_run_dir}")
        args.checkpoint_dir = str(_run_dir)
        _resume_paths = resolve_resume_run(_run_dir, getattr(args, "run_id", None))
        args.resume = str(_resume_paths["model_path"])                 # weights go through resolve_resume_split
        args.run_id = _resume_paths["run_id"]
                                                                       # Config guard runs HERE, before the vecenv is built, so a knob
                                                                       # mismatch fails in milliseconds instead of after 256 env spawns.
                                                                       # build_train_config is pure in (args, batch dims), so this is the
                                                                       # same dict train_config below is built from.
        _, _g_bptt, _g_bs = compute_batch_dims(args.num_envs)
        check_resume_config(_run_dir,
                            build_train_config(args, batch_size=_g_bs, bptt_horizon=_g_bptt))
    if args.checkpoint_dir is None:                                    # was the argparse default; now resolved here
        args.checkpoint_dir = str(CHECKPOINTS_DIR)

    run_label = Path(args.checkpoint_dir).name

    # ── W&B init ────────────────────────────────────────────────────────────
    wandb_run = None
    if getattr(args, "wandb", False):
        import wandb

        wandb_run = wandb.init(
            project=getattr(args, "wandb_project", "cs2rl"),
            entity=getattr(args, "wandb_entity", None) or None,
            name=run_label,
        )
        print(f"[Train] W&B run: {wandb_run.url}")

    # ── JSONL metrics file ───────────────────────────────────────────────────
    metrics_path = Path(args.checkpoint_dir) / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    _metrics_file = metrics_path.open("a")
    # N2 (2026-07-06 adversarial review): the file is opened in APPEND mode,
    # so back-to-back runs concatenate silently — 15 runs shared one file with
    # no separator and every analysis had to re-segment by agent_steps resets.
    # Stamp every row with a per-process run id (label + launch timestamp;
    # the label alone is NOT unique because re-runs into the same checkpoint
    # dir share it). Old rows lack the key — segment those the legacy way.
    # R0-C: --run-id (or the id read back from trainer_state.pt on --resume-run)
    # overrides the timestamped default so resumed rows share the id.
    run_id = getattr(args, "run_id", None) or f"{run_label}-{time.strftime('%Y%m%d-%H%M%S')}"

    # Shared team spirit value — all envs read it at episode start
    shared_ts = mp.Value("f", 0.3)

    _map_data = args.map_data

    # ── R0-D (#135): deterministic seeding ──────────────────────────────────
    # pufferl.py has its seeding commented out. Seed BEFORE build_policy
    # (weight init), before the vecenv (env seeds via env_seed_base below) and
    # before any random.* consumer (SelfPlayManager draws from module-level
    # random). This runs BEFORE load_full_resume, so on --resume-run the
    # python/numpy/torch states saved in train_state.pt (_rng_state_dict)
    # override this fresh seed — the same RNG set, one path. Env xorshift32
    # state is NOT restored on resume (C side; see load_full_resume's WARN).
    # Eval env seed 10_000_003 (Task 13) cannot collide with worker env seeds
    # env_seed_base(--seed) + i for --seed<=4 (any num_envs) — see env_seed_base.
    # CAVEAT: this makes CPU runs bit-exact; CUDA runs are seeded but NOT
    # bit-exact (no torch.use_deterministic_algorithms / cudnn flags are set).
    seed_everything(args.seed)

    # ── Batch 3 (T5b): cont-action shared memory across the fork boundary ──
    # PufferLib's Multiprocessing backend forks workers AFTER allocating its
    # own shm dict, so any Python attribute set on the main vecenv after
    # fork is invisible to workers. Mirror the pattern with our own
    # RawArray('f', num_envs * N_AGENTS * AIM_DIM) allocated BEFORE
    # pufferlib.vector.make runs. The trainer-side patch
    # (_patch_trainer_with_hybrid_aim) writes the policy's Δyaw sample into
    # `_cont_action_view_main` every send(); each worker's Cs2Env receives a
    # numpy view onto the same physical bytes via _attach_cont_action_view
    # (called inside env_factory below). For the Serial backend the view is
    # also attached, but the per-env step wrapper installed by the patcher
    # takes precedence — see that function for the dual-path docstring.
    from multiprocessing import RawArray

    # 10 (5 T + 5 CT) — N_AGENTS not exported via _action_spec; use AGENT_IDS.
    _agents_per_env = len(AGENT_IDS)
    _per_env_floats = _agents_per_env * AIM_DIM
    _cont_action_shm = RawArray("f", args.num_envs * _per_env_floats)
    _cont_action_view_main = np.frombuffer(_cont_action_shm, dtype=np.float32).reshape(
        args.num_envs * _agents_per_env, AIM_DIM)

    # ── F8: action-mask shared memory, the REVERSE direction (env→trainer) ──
    # Same fork-inheritance pattern as the cont-action RawArray above, but the
    # envs write (Cs2Env copies its C-computed masks into its slice at the end
    # of every step/reset) and the trainer reads right after vecenv.recv().
    # recv() is the synchronisation point: the worker finished its step before
    # the batch is handed over, so the bytes always match the obs in hand.
    _mask_shm = RawArray("b", args.num_envs * _agents_per_env * ACTION_MASK_DIM)
    _mask_view_main = np.frombuffer(_mask_shm,
                                    dtype=np.int8).reshape(args.num_envs * _agents_per_env,
                                                           ACTION_MASK_DIM)

    # Reward-weight overrides (spec 2026-08-01) ride in as closure state, NOT
    # per-env kwargs. NOTE: train_config is built AFTER the vecenv exists
    # (below), which is why this reads args directly rather than the config
    # dict. Keep this call — it is what the seam test pins.
    env_factory = build_train_env_factory(args, shared_ts=shared_ts, map_data=_map_data)

    # Per-env kwargs list — pufferlib.vector.make accepts a list of dicts
    # (one per env). All args propagate verbatim through fork because
    # they're stored on env_kwargs[i] BEFORE Process.start() (see
    # .venv/lib/.../pufferlib/vector.py:333-346).
    # R0-D (#135): the env seed rides here too — pufferlib.vector.make would
    # silently drop a `seed=` kwarg (see build_env_factory's docstring).
    _per_env_kwargs = [{
        "_cont_shm": _cont_action_shm,
        "_cont_idx": i,
        "_mask_shm": _mask_shm,
        "_seed": env_seed_base(args.seed) + i,
    } for i in range(args.num_envs)]

    backend_name = args.vec_backend.lower()
    if backend_name == "multiprocessing":
        import psutil

        backend = pufferlib.vector.Multiprocessing
        physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        num_workers = args.vec_num_workers or auto_vec_workers(args.num_envs, physical_cores)
        vec_kwargs = {
            "num_workers": num_workers,
            "batch_size": args.num_envs,
            "zero_copy": True,
            "overwork": args.vec_overwork,
        }
    elif backend_name == "serial":
        backend = pufferlib.vector.Serial
        num_workers = 1
        vec_kwargs = {}
    else:
        raise ValueError(f"Unsupported vec backend: {args.vec_backend}")

    print(f"[Train] Creating {args.num_envs} vectorised envs "
          f"(backend={backend_name}, workers={num_workers})...")
    # pufferlib.vector.make quirk: if env_creator is a single callable AND
    # env_kwargs is a per-env list, the broadcast logic at vector.py:672-684
    # overwrites the per-env list. Pass env_creators as an explicit list of
    # N copies of the same factory to make per-env kwargs survive. The
    # env_args list is required to match length.
    vecenv = pufferlib.vector.make(
        [env_factory] * args.num_envs,
        env_args=[[] for _ in range(args.num_envs)],
        env_kwargs=_per_env_kwargs,
        num_envs=args.num_envs,
        backend=backend,
        **vec_kwargs,
    )
    # R0-H: spawn lists within StaticData capacity, and 4 + 4 on the arena.
    # `map` is absent on harness/legacy args objects ⇒ generic bounds only.
    check_spawn_counts(vecenv, getattr(args, "map", None) or "")

    # Batch 7 (spec §3.3): sniff the resume checkpoint BEFORE build_policy —
    # the architecture decision has to exist at construction time, and neither
    # train_config (built below) nor the checkpoint read (further below) is
    # available yet. The sniffed dict is reused at the load site.
    # getattr on the flag keeps harness/older args objects working.
    resume_path = getattr(args, "resume", None)
    # Both bits are resolved here so a flag-less crash-resume never narrows
    # either axis, then passed into build_policy (omitted flags never drop a
    # split checkpoint back to the shared vintage).
    tct_split_heads, tct_split_trunk, _resume_state_dict, resume_path = resolve_resume_split(
        resume_path,
        heads_flag=bool(getattr(args, "tct_split_heads", False)),
        trunk_flag=bool(getattr(args, "tct_split_trunk", False)))

    print(f"[Train] Building policy on device={device} "
          f"(tct_split_heads={tct_split_heads}, tct_split_trunk={tct_split_trunk})...")
    policy = build_policy(vecenv,
                          device,
                          tct_split_heads=tct_split_heads,
                          tct_split_trunk=tct_split_trunk,
                          aim_log_std_max=getattr(args, "aim_log_std_max", None),
                          pin_pitch=bool(args.pin_pitch))

    agents_per_env, bptt_horizon, batch_size = compute_batch_dims(args.num_envs)
    # batch_size = 128 * 10 * 64 = 81920 → 81920 / 8192 = 10 minibatches per epoch

    train_config = build_train_config(args, batch_size=batch_size, bptt_horizon=bptt_horizon)

    # Provenance dump — the fingerprint hash is captured at --dump-config time,
    # this write is just for later inspection. Wrapped safely so a serialization
    # hiccup never kills training. sort_keys=True makes the file byte-stable so
    # diffing two runs' config.json shows only real HP changes.
    try:
        (Path(args.checkpoint_dir) / "config.json").write_text(
            json.dumps(train_config, sort_keys=True, indent=2, default=str))
    except Exception as _e:
        print(f"[Train] WARN: failed to write config.json: {_e}")

    # ── Resume from checkpoint ───────────────────────────────────────────────
    # resume_path / _resume_state_dict come from the pre-build_policy sniff
    # above; nothing is re-read from disk here.
    if resume_path:
        state_dict = _resume_state_dict
        # gh#91: BC warm-start checkpoints carry aim_log_std frozen at
        # LOG_STD_INIT — widen to AIM_LOG_STD_RESUME_INIT before loading or
        # the KL early-stop throttles the whole run (see the helper's doc).
        # ORDER is load-bearing (spec 2026-08-15 §3.3): σ re-init on the
        # LEGACY dict, then heads convert (needs bare aim_log_std), then
        # trunk convert. Duplicating heads first would hide the σ key.
        # R0-C: a full-state resume restores the exact pre-crash σ — never widen.
        if not resume_run and reinit_frozen_aim_log_std(state_dict,
                                                        cap=getattr(args, "aim_log_std_max", None)):
            print(f"[Train] BC-frozen aim_log_std detected in {resume_path.name}: "
                  f"re-initialized to log(0.3) ≈ {AIM_LOG_STD_RESUME_INIT:.3f} (gh#91)")
        if tct_split_heads and not state_dict_is_split(state_dict):
            state_dict = convert_legacy_state_dict_to_split(state_dict)
            print("[Train] Warm split: duplicated the legacy policy heads into per-team "
                  "T/CT copies (spec 2026-08-13 §3.3) — both teams start identical.")
        if tct_split_trunk and not state_dict_is_trunk_split(state_dict):
            state_dict = convert_shared_trunk_to_split(state_dict)
            print("[Train] Warm split: duplicated the shared encoder+LSTM into per-team "
                  "T/CT copies (spec 2026-08-15 §3.3) — both teams start identical.")
        load_state_dict_arch_checked(policy, state_dict, source=str(resume_path))
        print(f"[Train] Resumed from checkpoint: {resume_path}")
    # ────────────────────────────────────────────────────────────────────────

    if train_config.get("warmstart_entropy") and not resume_path:
        print("[Train] WARN: --warmstart-entropy without --resume — the grace window "
              "will suppress entropy pressure on a from-scratch policy (legal, but "
              "probably not what you want).")

    # Rung 0 §2.2 + Rung 1a T3: static per-run participation vector, env-row
    # major (10 rows per env: T at 0-4, CT at 5-9). Under --opponent self it
    # selects slots 0..n-1 of BOTH teams — the exact slots the C env spawns
    # (cs2_env.py, n_active_per_team); under --opponent noop, the hero team's
    # slots only. THE SAME helper backs train_test_harness, so a harness test
    # can never be green against a formula production does not run.
    # The assert is the agreement check: the vector is derived from args while
    # the envs were built from build_train_env_factory, and a disagreement
    # would mask the wrong rows silently rather than crash.
    _n_active = env_config_from_args(args).n_active_per_team
    _participating_rows = build_participating_rows(args.num_envs,
                                                   _n_active,
                                                   opponent_mode=_opponent_mode,
                                                   hero_team=SelfPlayManager.initial_hero_team())
    assert vecenv.driver_env.n_active_per_team == _n_active, "driver env / args disagree"

    # ── Self-play setup ──────────────────────────────────────────────────────
    # F11 (2026-07-06 adversarial review): the selfplay replacement evaluate()
    # is the ONLY rollout path that understands the hybrid 4-tuple policy
    # contract — stock PuffeRL.evaluate crashes on the forward_eval tuple
    # unpack at its first call, so --no-self-play was broken in production.
    # The patch is now applied UNCONDITIONALLY (mirroring train_test_harness,
    # which adopted this shape at T5); --no-self-play means "no past-policy
    # mixing": p_past=0.0 with an empty, never-seeded pool ⇒ should_use_past()
    # is always False, and the pool save / team-switch bookkeeping in the
    # main loop is skipped via self_play_enabled below.
    self_play_enabled = bool(getattr(args, "self_play", True))
    # W3 (#154): the pool constants (15 / 25 / 0.6 / 50) and the
    # `0.3 if <on> else 0.0` p_past rule now live in
    # env_factory.build_selfplay_manager, which is also what the two harness
    # sites call — they used to spell the same construction out twice more.
    # `self_play_enabled` is passed as the FLAG, not a p_past value, so no caller
    # can set a different mixing probability at one site than another.
    self_play_mgr = build_selfplay_manager(
        self_play_enabled=self_play_enabled,
        aim_log_std_max=getattr(args, "aim_log_std_max", None),
        pin_pitch=args.pin_pitch,
        opponent_mode=_opponent_mode,
    )
    # R0-C: on --resume-run the pool comes back from train_state.pt — no re-seed.
    if self_play_enabled and resume_path and resume_path.exists() and not resume_run:
        import shutil as _shutil

        seed_path = Path(args.checkpoint_dir) / "sp_seed.pt"
        _shutil.copy2(resume_path, seed_path)
        self_play_mgr._add_to_pool(seed_path)
        print(f"[SelfPlay] Pool pre-seeded with resume checkpoint ({seed_path.name})")

    # gh#168 W1 (ADR 0002): the trainer is a SUBCLASS, not a PuffeRL mutated in
    # place. Everything the four patch functions read at patch time —
    # participating_rows, the shm views, the (possibly pre-seeded) self-play
    # manager — is built ABOVE this line, exactly as before; everything that
    # used to be set on the instance after construction (run_id, weight_decay,
    # the aim-σ param group, the GC pins) stays below it, because no patch
    # reads it at patch time (spec §W1 table; both byte gates pin this order).
    # Function-local import ON PURPOSE: trainer.py subclasses PuffeRL and so
    # imports torch at module scope; `import train` must stay torch-free
    # (tests/test_w1_modules.py).
    from trainer import Cs2PuffeRL
    trainer = Cs2PuffeRL(train_config,
                         vecenv,
                         policy,
                         cont_action_view_main=_cont_action_view_main,
                         mask_view_main=_mask_view_main,
                         participating_rows=_participating_rows,
                         self_play_mgr=self_play_mgr)
    # R0-C: PuffeRL's NoLogger invents a timestamp run_id; pin ours so
    # <data_dir>/<run_id>/ matches the metrics rows and --resume-run can find it.
    trainer.logger.run_id = run_id
    trainer.optimizer.param_groups[0]["weight_decay"] = 1e-4
    # Rung 1a T1: …but NOT on the aim σ. Decay adds wd·θ to the gradient and
    # aim_log_std is always negative, so it would drift σ upward at exactly
    # zero true gradient — faking the gate's learning signal and eating the
    # init's clamp margin. Must run after the line above (it clones group 0's
    # hypers) and before load_full_resume (see the helper's PITFALLS).
    _n_sigma = isolate_aim_log_std_param_group(trainer)
    print(f"[Train] aim_log_std: {_n_sigma} parameter(s) moved to a weight_decay=0 param group "
          f"(fresh init {train_config['aim_log_std_init']:.4f}, "
          f"cap {train_config['aim_log_std_max']:.4f})")
    # Batch 3 (T5): Cs2PuffeRL.__init__ applies the hybrid-aim patch after
    # return_norm, but the dependency is at CALL time, not patch time: the
    # replacement train() body that return_norm installs (types.MethodType,
    # train_update.py; it never calls the stock train()) reads
    # self.cont_actions / self.logprobs_{d,c} on its first call, and nothing
    # at patch time checks they exist. Likewise selfplay's replacement
    # evaluate() writes those buffers every rollout. The only load-bearing
    # order is "all patches applied before the first evaluate()/train() call".
    # Pin the shm + view on the trainer so neither is GC'd mid-run. Without
    # holding _cont_action_shm here, Python could free the RawArray once
    # this function returns (Python doesn't know workers/numpy views are
    # using it via the OS-level mapping).
    trainer._cont_action_shm = _cont_action_shm
    trainer._action_mask_shm = _mask_shm               # F8: same GC-pinning rationale

    # R0-E.2: env flag ⇔ policy mask, or stop before the first rollout.
    assert_pin_pitch_agreement(vecenv, policy)
    # R0-G: env aim clamp ⇔ policy tanh scale (a resumed checkpoint may carry
    # a different buffer than the env it is now paired with).
    assert_max_turn_speed_agreement(vecenv, policy)
    if not self_play_enabled:
        # Mode-aware on purpose: "both teams use the current policy" is FALSE
        # under --opponent noop (the statue team is driven by the evaluate()
        # override, not by the policy), and it printed one line above the noop
        # provenance line that T4's pre-flight reads — two adjacent, mutually
        # contradictory claims about the same run in the same log.
        if _opponent_mode == "noop":
            print("[Train] Self-play mixing disabled (--no-self-play): the hero team "
                  "uses the current policy every epoch; the opponent team is a statue "
                  "(--opponent noop), not the current policy.")
        else:
            print("[Train] Self-play mixing disabled (--no-self-play): "
                  "both teams use the current policy every epoch.")
    if _opponent_mode == "noop":
        # Rung 1a T3: the run log is what T4's pre-flight reads, so state which
        # team is frozen, how many rows actually train, and on what horizon —
        # the three things a short/mis-masked run would get wrong silently.
        print(f"[Train] Opponent mode 'noop': team "
              f"{self_play_mgr.opponent_team.upper()} is a stationary statue; "
              f"{int(_participating_rows.sum()):,} of {_participating_rows.size:,} agent rows "
              f"participate (raw horizon {train_config['total_timesteps']:,} rows = "
              f"{train_config['participating_timesteps']:,} hero steps).")
    _resumed_from_step = None
    if resume_run:
        _info = load_full_resume(trainer, self_play_mgr, _resume_paths)
        _resumed_from_step = _info["resumed_from_step"]
        # Spec §R0-C bound vs the last metrics row (participating units); see
        # check_resume_metrics_bound for why both sides are checkpoint_interval
        # epochs wide.
        # Rung 1a T3 (spec, "Rung 1b note"): this bound assumes BOTH teams
        # participate, so under --opponent noop it is 2× too WIDE — i.e. only
        # ever too permissive, never a false alarm. Harmless for T4 (which does
        # not resume); halve it here before Rung 1b resumes a noop run.
        _B = batch_size * train_config["n_active_per_team"] // TEAM_SIZE
        _last = None
        if metrics_path.exists():
            for _line in metrics_path.read_text().splitlines():
                try:
                    _row = json.loads(_line)
                except json.JSONDecodeError:
                    continue
                if _row.get("run_id") == run_id:
                    _last = _row.get("step", _last)
        if _last is not None:
            check_resume_metrics_bound(_resumed_from_step, _last,
                                       train_config["checkpoint_interval"], _B)
            print(f"[Resume] global_step {_resumed_from_step:,} (last metrics row {_last:,}, "
                  f"gap {_resumed_from_step - _last:+,}) epoch {trainer.epoch}")
    # ────────────────────────────────────────────────────────────────────────

    # ── R0-I (Task 13): fixed-baseline eval env — parent-process, serial, the
    # SAME config as the workers: it comes from env_config_from_args, the one
    # resolver build_train_env_factory also uses, so a --laser-range /
    # --round-time-ticks run evaluates on what it trains on. Seed 10_000_003:
    # worker env seeds are env_seed_base(--seed) + i, so the only collision is
    # --seed 100 with >= 4 envs (env 3) — see env_seed_base. team_spirit=None →
    # raw rewards (eval never feeds training).
    _eval_hook = None
    _eval_interval = int(getattr(args, "eval_interval", 0) or 0)
    if _eval_interval > 0:
        from eval_baselines import BaselineEvaluator
        # W3 (#154), retyped by #165 PR B2: role eval. `team_spirit=None`, the
        # 10_000_003 seed, the load-bearing `auto_reset=False` AND the
        # raw-reward rule all live in env_factory._build_eval; this site passes
        # only what comes from THIS run's args, which is now one EnvConfig from
        # the same resolver the workers' factory reads. Requiring that config
        # (the builder has no default) is what stops a caller handing the eval
        # env a bare config while the driver env has the run's knobs — a
        # disagreement assert_eval_env_agreement right below would then have
        # something to catch.
        _eval_env = build_env_for("eval", map_data=_map_data, config=env_config_from_args(args))
        assert_eval_env_agreement(_eval_env, trainer.vecenv.driver_env)
        _eval_hook = ScheduledEval(BaselineEvaluator(_eval_env, episodes=40, seed=args.seed),
                                   _eval_interval, policy, device)
        print(f"[Eval] fixed-baseline eval every {_eval_interval} epochs "
              f"(40 episodes vs random + oracle, round_time={_eval_env.round_time})")

    save_path = Path(args.checkpoint_dir) / "dust2_policy.pt"
    last_save = time.time()

    # gh#93: arm the zero-kills rule only when the env actually rewards kills.
    _kills_expected = _kill_reward_is_active(trainer.vecenv)
    if not _kills_expected:
        print("[Train] Kill reward is 0 — dead-run zero-kills alert disabled (gh#93).")
    dead_run_detector = DeadRunDetector(kills_expected=_kills_expected)

    print(f"[Train] Starting PufferLib PPO for {args.timesteps:,} env steps...")
    while trainer.epoch < trainer.total_epochs:
        trainer._tag_metrics = None    # TAG: drop any un-injected measurement
        t0 = time.perf_counter()
        trainer.evaluate()
        trainer._timing["collect_ms"] = (time.perf_counter() - t0) * 1000.0

        # Rung 0 §2.2: the participating buffer is zero-initialised, so an
        # all-False buffer means evaluate() never ran its scatter — every
        # masked reduction below would then divide by the clamp floor and
        # train on nothing. Fail loudly instead.
        assert trainer.participating.any(), "participating buffer never written this epoch"
        t0 = time.perf_counter()
        logs = trainer.train()
        trainer._timing["update_ms"] = (time.perf_counter() - t0) * 1000.0
        # Injected HERE, not in the isinstance(logs, dict) block further down:
        # _timed_train used to inject before returning, so _eval_hook.after_train
        # already sees these keys. Folding this into the later block would change
        # what the hook is handed.
        if isinstance(logs, dict):
            logs["timing/collect_ms"] = trainer._timing["collect_ms"]
            logs["timing/update_ms"] = trainer._timing["update_ms"]

        # Team spirit annealing: 0.3→0.7 over 5M participating-agent steps
        ts_val = min(0.7, 0.3 + trainer.global_step / 5_000_000)
        shared_ts.value = ts_val

        # R0-I: OUTSIDE the isinstance(logs, dict) guard on purpose — see
        # ScheduledEval (the 0.25 s log throttle must not skip an eval epoch).
        if _eval_hook is not None:
            _eval_hook.after_train(trainer, logs)

        if isinstance(logs, dict):
            game_metrics = compute_game_metrics(logs)
            logs.update(game_metrics)
            # F14 (2026-07-06 adversarial review): the detector's return was
            # previously discarded — the 30M degenerate run printed its banner
            # and kept burning compute for another ~150 epochs. Now: save an
            # autopsy checkpoint and abort with a NONZERO exit code so shell
            # wrappers / experiment runners see the failure. Opt out with
            # --no-dead-run-abort (e.g. when deliberately probing degenerate
            # regimes). NaN/Inf still raises inside check() as before.
            if (dead_run_detector.check(trainer.global_step, logs)
                    and getattr(args, "dead_run_abort", True)):
                autopsy_path = Path(args.checkpoint_dir) / "dust2_policy_dead.pt"
                autopsy_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(policy.state_dict(), autopsy_path)
                _metrics_file.flush()
                print(f"[Train] DEAD RUN — aborting at step {trainer.global_step:,}. "
                      f"Autopsy checkpoint: {autopsy_path}")
                if wandb_run is not None:
                    wandb_run.finish(exit_code=3)
                trainer.close()
                raise SystemExit(3)

            # Network health monitoring every 5 epochs (too expensive every epoch)
            if trainer.epoch % 5 == 0:
                health_metrics = compute_network_health(policy, device)
                logs.update(health_metrics)

            # ── Self-play bookkeeping ────────────────────────────────────────
            # F11: gated on the FLAG, not the manager (the manager now always
            # exists for the evaluate patch) — no pool saves / team switches
            # under --no-self-play.
            if self_play_enabled:
                self_play_mgr.maybe_switch_teams(trainer.epoch)
                # R0-I: elimination-only — winner_ct counts timeouts, which
                # would pool-save a passive CT as "dominant".
                win_rate_t, win_rate_ct = elimination_only_win_rates(logs)
                self_play_mgr.maybe_save(
                    policy,
                    Path(args.checkpoint_dir),
                    trainer.epoch,
                    win_rate_t,
                    win_rate_ct,
                )
                logs["self_play/pool_size"] = float(len(self_play_mgr.pool))
                # Observe-only (spec 2026-08-15 §3.4): 0.0/1.0 float on the
                # OUTER logs dict, next to pool_size. Not trainer.losses —
                # `_selfplay_used_past` is set during evaluate(), not train().
                # Do not log self_play/opponent_id (string, persist-dropped).
                logs["self_play/used_past"] = self_play_used_past_metric(trainer)
                # opponent_team flag: 1.0 = CT opponent, 0.0 = T opponent.
                logs["self_play/opponent_team"] = float(self_play_mgr.opponent_team == "ct")
                # ────────────────────────────────────────────────────────────

            # Batch 3.5 (#24): per-axis aim log_std metrics. Read CLAMPED values
            # (the values the policy actually used at this iteration), not the raw
            # nn.Parameter. Load-bearing for T7 acceptance gate 2:
            # aim_log_std_pitch > -3.5 at 30M steps, and format_train_status must
            # keep the 'aim_log_std_pitch=' substring greppable.
            # Batch 7: the reader branches on architecture inside the helper —
            # a split policy has no `aim_log_std` attribute at all (spec §3.6).
            log_aim_log_std(policy, logs)

            # Batch 7 (spec §3.4): split/active is the analyzer's labeling
            # signal for the structurally-zero policy_heads TAG cells. It is
            # derived from the POLICY OBJECT, never from config.json — the
            # config is rewritten unconditionally at every launch, so a
            # flag-less crash-resume of a split run (which key inference
            # deliberately supports) would stamp tct_split_heads:false and
            # silently disarm the labeling. A metrics key travels with the rows
            # the analyzer already reads and survives resume seams.
            #
            # PLACEMENT IS PART OF THE CONTRACT: this belongs HERE, in the
            # unconditional outer-loop logging block, NOT inside the TAG hook
            # and NOT behind `tag_diagnostic` / `epoch % tag_every`. Every
            # logged epoch's row must carry it. Gating it on the TAG throttle
            # would leave ~80% of rows unlabeled at the default --tag-every 5,
            # and any future analyzer that inspects a non-measurement row (a
            # dead-window scan, a σ trajectory, a divergence plot) would read
            # the missing key as "legacy run" — the exact misidentification the
            # key exists to prevent. It is also independent of the TAG flag
            # entirely: a split run launched WITHOUT --tag-diagnostic still
            # labels every row.
            logs["split/active"] = float(hasattr(policy, "aim_log_std_t"))
            # Trunk-split twin (spec 2026-08-15 §3.4): same unconditional
            # placement as split/active. 1.0 iff the live policy has
            # encoder_t — derived from the object, never config.json.
            # Analyzer keys the trunk-structural verdict and the "no actor
            # TAG cell is a decision metric" footer on this key. Do not
            # gate on --tag-every / --tag-diagnostic.
            logs["split/trunk_active"] = float(hasattr(policy, "encoder_t"))
            logs.update(compute_head_divergence(policy))
            logs.update(compute_trunk_divergence(policy))

            # TAG injection — MUST stay after dead_run_detector.check above
            # (deliberate NaNs; see _inject_tag_metrics docstring).
            _inject_tag_metrics(trainer, logs)

            # ── Persist metrics ──────────────────────────────────────────────
            log_entry = {
                "run_id": run_id,                                           # N2: string key — segment runs by this, not by agent_steps resets
                "step": trainer.global_step,
                "epoch": trainer.epoch,
                "team_spirit": ts_val,
                **{
                    k: v
                    for k, v in logs.items() if isinstance(v, (int, float))
                },
            }
                                                                            # R0-C: stamp the FIRST row after a --resume-run (analysis seam marker).
            if _resumed_from_step is not None:
                log_entry["resumed_from_step"] = _resumed_from_step
                _resumed_from_step = None
            _metrics_file.write(json.dumps(log_entry) + "\n")
            _metrics_file.flush()
            if wandb_run is not None:
                wandb_run.log(log_entry, step=trainer.global_step)

        if time.time() - last_save > args.save_every_sec:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_save_state_dict(policy.state_dict(), save_path)
            last_save = time.time()
            print(f"Saved checkpoint to {save_path}")

        if isinstance(logs, dict):
            if trainer.epoch % 10 == 0:
                print(format_train_status(trainer.epoch, ts_val, logs))
            print(f"[Timing] collect={trainer._timing['collect_ms']:.0f}ms  "
                  f"update={trainer._timing['update_ms']:.0f}ms  "
                  f"SPS={logs.get('SPS', 0):.0f}")

    trainer.close()
    if _eval_hook is not None:
        _eval_hook.close()

    # Final checkpoint save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_state_dict(policy.state_dict(), save_path)
    print(f"[Train] Final checkpoint saved to {save_path}")

    _metrics_file.close()
    print(f"[Train] Metrics saved to {metrics_path}")
    if wandb_run is not None:
        wandb_run.finish()

    print("[Train] Done.")


# ── SECTION: CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # MUST BE THE FIRST STATEMENT IN THIS BLOCK.
    #
    # WHAT: running `python src/train.py` binds THIS file's module object to the
    # name "__main__", leaving sys.modules["train"] empty. Any runtime
    # `from train import ...` then RE-EXECUTES this whole module body under the
    # name "train", and the process ends up holding two independent copies of it:
    # two sets of module-level constants, two of every class object, and
    # `isinstance` between them silently False. eval_baselines does exactly that
    # import, function-locally inside PolicyActor.__init__ and
    # PolicyActor.from_checkpoint, to dodge a circular top-level import — so a
    # plain `--eval-interval N` script run is enough to trigger it.
    #
    # WHY setdefault and not `=`: under `import train` (the whole test suite,
    # scripts/, the Modal runner) "train" is already a real, fully-initialised
    # module and this block is never reached anyway; setdefault keeps the
    # invariant "the first binding wins" true in every launch mode.
    #
    # PITFALL for whoever tests this: an AST pin proves the statement is WRITTEN,
    # not that it does anything, and the §3 determinism gate cannot see it — that
    # gate runs `--no-self-play --eval-interval 0`, precisely the flag set on
    # which no runtime `from train import` ever fires. The behavioural check is a
    # child interpreter under `-X importtime`: WITHOUT this line its stderr
    # carries an `import time: ... | train` line (the second body execution),
    # WITH it none. Do NOT spell that check as
    # `runpy.run_path(..., run_name="__main__")` plus a post-hoc identity assert:
    # run_path swaps sys.modules["__main__"] only for the duration of the call
    # and restores it on return, so the assert compares the alias against the
    # RESTORED __main__ and reports a false failure (measured: False after the
    # call, True inside it).
    sys.modules.setdefault("train", sys.modules["__main__"])

    # The env's own defaults, read from the dataclass that declares them, so the
    # CLI cannot drift from the env (spec 2026-09-03 R11). Bound once here rather
    # than per-flag: three flags below read it, and a second EnvConfig() would be
    # a second place to look when a default changes.
    _ENV_DEFAULTS = EnvConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dust2",
        action="store_true",
        help="Alias for --map dust2 (kept for the Modal runner / old scripts); --map wins",
    )
    parser.add_argument("--map",
                        choices=MAP_NAMES,
                        default=None,
                        help="R0-H: map preset; takes precedence over --dust2. Default: "
                        "dust2 if --dust2 else simple. Sets config['env']=cs2-<map>.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Load policy weights from .pt file before training (optimizer state not restored)",
    )
    parser.add_argument(
        "--resume-run",
        type=str,
        default=None,
        metavar="RUN_DIR",
        dest="resume_run",
        help="R0-C: full-state resume from <run_dir> (== --checkpoint-dir of the run): "
        "policy, optimizer, step counters, α/scheduler/return-norm/warm-start/"
        "self-play/RNG. --timesteps is the TOTAL budget, not additional.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        dest="run_id",
        help="Metrics-row and <checkpoint_dir>/<run_id>/ id (default <label>-<timestamp>).")
    parser.add_argument("--checkpoint-interval",
                        type=int,
                        default=DEFAULT_CHECKPOINT_INTERVAL,
                        dest="checkpoint_interval",
                        help="Epochs between full-state checkpoints (default 200; Rung 1 uses 10).")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_every_sec", type=int, default=300)
    parser.add_argument(
        "--checkpoint_dir",
        "--checkpoint-dir",
        type=str,
        default=None,
        dest="checkpoint_dir",
        help="default: CHECKPOINTS_DIR; must match --resume-run when both are given",
    )
    parser.add_argument(
        "--dump-config",
        action="store_true",
        help=("Write <checkpoint_dir>/config.json with the train_config dict "
              "and exit (no training)."),
    )
    parser.add_argument("--vec-backend", type=str, default="multiprocessing")
    parser.add_argument("--vec-num-workers", type=int, default=0)
    parser.add_argument("--vec-overwork", action="store_true")
    parser.add_argument("--record-out", type=str, default=str(RECORDINGS_DIR / "latest.rrd"))
    parser.add_argument("--record-policy",
                        type=str,
                        choices=("auto", "random", "sample", "greedy"),
                        default="auto")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-policy",
                        type=str,
                        choices=("auto", "random", "sample", "greedy"),
                        default="auto")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=("Run name; auto-prefixed with DDMMYY-N- where N = count of existing checkpoint dirs "
              "starting with today's date. E.g. --name 1M-ct → '200326-3-1M-ct'."),
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="cs2rl", dest="wandb_project")
    parser.add_argument("--wandb-entity", type=str, default="", dest="wandb_entity")
    parser.add_argument(
        "--no-self-play",
        action="store_false",
        dest="self_play",
        help="Disable self-play (both teams always use current policy)",
    )
    parser.add_argument(
        "--no-dead-run-abort",
        action="store_false",
        dest="dead_run_abort",
        help=("F14: by default a DEAD RUN verdict (5+ accumulated degeneracy alerts) "
              "saves an autopsy checkpoint and exits with code 3. Pass this to only "
              "print the banner and keep training (e.g. when deliberately studying "
              "degenerate regimes)."),
    )
    parser.add_argument("--n-active-per-team",
                        type=int,
                        default=_ENV_DEFAULTS.n_active_per_team,
                        dest="n_active_per_team",
                        help="Rung 0: agents per team that spawn; the rest are parked "
                        "(noop-masked, zero reward, excluded from every trainer statistic). "
                        "--timesteps counts PARTICIPATING agent-steps.")
    parser.add_argument("--pin-pitch",
                        type=int,
                        choices=(0, 1),
                        default=None,
                        dest="pin_pitch",
                        help="R0-E.2: 1 = env ignores the pitch action and the policy drops the "
                        "pitch dim from log_prob_c. Default: auto (1 iff the map is flat); "
                        "an explicit value that disagrees with the map is refused.")
    parser.add_argument("--crouch-enabled",
                        type=int,
                        choices=(0, 1),
                        default=_ENV_DEFAULTS.crouch_enabled,
                        dest="crouch_enabled",
                        help="R0-E.2: 0 masks the crouch action (stance parity for pinned-pitch "
                        "duels; a crouched target is an unobservable guaranteed miss).")
    parser.add_argument("--jump-enabled",
                        type=int,
                        choices=(0, 1),
                        default=_ENV_DEFAULTS.jump_enabled,
                        dest="jump_enabled",
                        help="Rung 1a: 0 masks the jump action (height parity for pinned-pitch "
                        "duels; an airborne target sits outside the 36u vertical semi-axis and "
                        "is an unobservable guaranteed miss). Default 1 = today's env.")
    parser.add_argument("--opponent",
                        choices=OPPONENT_MODES,
                        default="self",
                        dest="opponent",
                        help="Rung 1a T3: 'noop' turns the opponent team into a stationary "
                        "statue (no-op bin on every action head, zero aim delta) and excludes "
                        "its rows from participation, from --timesteps and from every loss. "
                        "Requires --no-self-play. Default 'self' = today's behaviour.")
    # R0-G env knobs. Default None ⇒ the env's nav.py constant (config.json
    # records None, not a copied constant). Not in RESUME_CONFIG_ALLOWLIST:
    # changing any of them on --resume-run is a different experiment.
    # PITFALL: a config.json written before R0-G/R0-I/R0-J lacks these keys;
    # check_resume_config treats missing ≠ None as a mismatch, so such run dirs
    # cannot --resume-run (by design — same as the R0-E keys).
    # R0-I (Task 13): fixed-baseline eval cadence. 0 = off (default: the
    # 40-episode serial eval costs wall time every epoch it runs).
    parser.add_argument("--eval-interval",
                        type=int,
                        default=0,
                        dest="eval_interval",
                        help="R0-I: run the fixed-baseline eval (40 episodes vs random and vs "
                        "oracle on the training map/knobs) every N epochs; eval/* keys land on "
                        "the next logged metrics row. 0 = off.")
    parser.add_argument("--round-time-ticks",
                        type=int,
                        default=None,
                        dest="round_time_ticks",
                        help="R0-G: ticks per round (episode length). Default: nav.ROUND_TIME.")
    parser.add_argument("--laser-range",
                        type=float,
                        default=None,
                        dest="laser_range",
                        help="R0-G: hitscan reach in map units. Default: nav.LASER_RANGE.")
    parser.add_argument(
        "--max-turn-speed",
        type=float,
        default=None,
        dest="max_turn_speed",
        help="R0-G: max yaw/pitch delta per tick (rad). Default: nav.MAX_TURN_SPEED_RAD. "
        "Rung 1 must not set this — it rescales the aim action.")
    # R0-J (Task 14): PPO discount and PBRS discount. Both are config keys and
    # NOT allowlisted for --resume-run. --pbrs-gamma default None ⇒ follows
    # --gamma (resolve_gammas); pass it only to deliberately break invariance.
    parser.add_argument("--gamma",
                        type=float,
                        default=DEFAULT_GAMMA,
                        help="R0-J: PPO discount factor (default 0.999). Also the PBRS "
                        "shaping discount unless --pbrs-gamma is given.")
    parser.add_argument("--pbrs-gamma",
                        type=float,
                        default=None,
                        dest="pbrs_gamma",
                        help="R0-J: PBRS shaping discount. Default: equal to --gamma (the only "
                        "policy-invariant choice). Set explicitly only for experiments that "
                        "deliberately decouple the two.")
    parser.add_argument("--aim-entropy-bonus",
                        choices=("on", "off"),
                        default="on",
                        dest="aim_entropy_bonus",
                        help="R0-E.4: include the Gaussian aim entropy in the entropy objective "
                        "(default on). 'off' stops the +1/dim gradient that pins aim σ at the cap.")
    parser.add_argument("--aim-log-std-max",
                        type=float,
                        default=None,
                        dest="aim_log_std_max",
                        help="R0-E.3: per-run cap on aim log σ, in (LOG_STD_MIN + 0.4, log 0.5] "
                        "≈ (-4.205, -0.693] (default LOG_STD_MAX = log 0.5). Rung 1a T1: the "
                        "lower bound leaves room for the σ init to sit 0.2 BELOW the cap and "
                        "still stay above LOG_STD_MIN — see validate_aim_log_std_max.")
    parser.add_argument("--warmstart-entropy",
                        action="store_true",
                        dest="warmstart_entropy",
                        help="Two-phase entropy override for BC-warm-started runs: grace window "
                        "(alpha~0, floor off) then target ramp re-anchored at measured entropy. "
                        "Pair with --resume; see spec 2026-08-01.")
    parser.add_argument("--warmstart-grace-steps",
                        type=int,
                        default=5_000_000,
                        dest="warmstart_grace_steps")
    parser.add_argument("--warmstart-ramp-steps",
                        type=int,
                        default=10_000_000,
                        dest="warmstart_ramp_steps")
    parser.add_argument("--warmstart-alpha-ceiling",
                        type=float,
                        default=0.0,
                        dest="warmstart_alpha_ceiling")
    # ── Reward weights (spec 2026-08-01 §4.2 → 2026-09-03 §2.3) ──
    # Generated from RewardWeights' fields so flag name, dest and default can
    # never disagree with the config key — the dest-typo class of bug (commit
    # 4d9dfa0) is impossible by construction. Flag == field name with dashes;
    # default == the field default, so omitting a flag reproduces today's env.
    for _rw_name, _rw_default in RewardWeights().as_dict().items():
        parser.add_argument(f"--{_rw_name.replace('_', '-')}",
                            type=float,
                            default=_rw_default,
                            dest=_rw_name,
                            help=f"Env reward weight {_rw_name} (default {_rw_default}).")
    del _rw_name, _rw_default                                                    # module scope is `if __name__` here — don't leak loop vars
    parser.add_argument(
        "--reward-symmetrize",
        action="store_true",
        dest="reward_symmetrize",
        help="Zero-sum the per-tick reward vector in Python after each step: "
        "r_i' = 0.5*(r_i - mean over the opposing team). Removes every private "
        "per-team subsidy from the shared policy's gradient (spec 2026-08-01 §4.3).")

    # ── TAG gradient-conflict diagnostic (spec 2026-08-13) ──
    parser.add_argument("--tag-diagnostic",
                        action="store_true",
                        dest="tag_diagnostic",
                        help="Measure T-vs-CT policy-gradient cosine similarity per parameter "
                        "group during PPO updates (tag/* metrics). Zero behavioral effect on "
                        "training — pinned bitwise by tests/test_tag_trainer.py.")
    parser.add_argument("--tag-every",
                        type=int,
                        default=5,
                        dest="tag_every",
                        help="Measure on epochs where epoch %% tag_every == 0 (default 5; "
                        "values < 1 clamp to 1 at the hook).")

    # ── Batch 7: T/CT policy-heads split (spec 2026-08-13) ──
    parser.add_argument(
        "--tct-split-heads",
        action="store_true",
        dest="tct_split_heads",
        help="Give each team its own copy of the policy heads (action_heads, aim_mu, "
        "aim_log_std), routed by the obs team bit; trunk and value head stay shared. "
        "Only affects FRESH construction and the legacy->split warm conversion — every "
        "loader infers split-ness from the checkpoint's keys, so a crash-resume without "
        "this flag still rebuilds a split policy.")
    parser.add_argument(
        "--tct-split-trunk",
        action="store_true",
        dest="tct_split_trunk",
        help="Give each team its own copy of the actor trunk (encoder, LSTM), routed "
        "by the obs team bit; policy heads stay as --tct-split-heads decides and the "
        "value head stays shared. Only affects FRESH construction and the "
        "legacy->split warm conversion — every loader infers split-ness from the "
        "checkpoint's keys, so a crash-resume without this flag still rebuilds a "
        "split-trunk policy.")
    args = parser.parse_args()
    # Every mode (record/eval too) derives env seeds from --seed; fail here,
    # not deep in a mode.
    env_seed_base(args.seed)
    # Same idea for the aim σ cap: range-check it here (torch-free) so
    # --dump-config / the sweep fingerprint reject a bad --aim-log-std-max
    # instead of build_policy() 30 s into every retry.
    validate_aim_log_std_max(args.aim_log_std_max)
    # Rung 1a T3: --opponent noop is only coherent with self-play bookkeeping
    # off. Checked HERE, above the --dump-config exit, for the same reason as
    # the σ cap: the Modal/run_rung1 fingerprint step must reject the launch
    # before any env is built.
    assert_opponent_self_play_compatible(args.opponent, args.self_play)

    # ── R0-H: map name → MapData → pin_pitch, ABOVE the --dump-config exit ──
    # The Modal runner fingerprints every launch from --dump-config, so the
    # dump must carry the same env label and the same geometry-resolved
    # pin_pitch the run's own config.json will (Task 9 ruling: the value comes
    # from the LOADED map, never a name table). Cost: ~0.8 s (`import map`)
    # for simple/arena, ~1 s for dust2 from the nav cache (pin_pitch_for_map(
    # None) loads it via the same _ENV_CACHE make_env uses, so nothing is
    # loaded twice). PITFALL: `--dump-config --map dust2` (or --dust2)
    # therefore needs nav/de_dust2.nav on the HOST that runs the dump (Modal
    # fingerprints run host-side). It does NOT need the vis cache: the dump
    # loads the map with build_vis=False (gh#251), because building that
    # cache forks a 12-worker pool that a killed dump leaves orphaned.
    if args.map is None:
        args.map = "dust2" if args.dust2 else "simple"
    args.map_data = build_map_data(args.map)
    print(f"[Map] Using {args.map} map")
    # gh#251: the dump must exit before ANY process is forked. For dust2 the
    # only fork on this path is the cold-cache vis-matrix build (cpu_count()
    # workers, ~900 MB each, orphaned to PID 1 when a test killed the dump);
    # pin_pitch reads centroids_z only, so the dump skips the vis build.
    resolve_pin_pitch(args, build_vis=not args.dump_config)

    if args.dump_config:
        # Zero-side-effect mode: write config.json and exit. Runs BEFORE device
        # detection so no torch import is triggered (the map is built above:
        # config.json needs its pin_pitch/env label). This lets
        # scripts/run_experiment.py fingerprint the HPs cheaply (no env, no
        # CUDA probe). Keep this branch lean — anything imported here adds
        # startup cost to every experiment launch.
        if args.device is None:
            args.device = "cpu"        # placeholder; never used for training

        # --checkpoint-dir defaults to None (so --resume-run can tell "given" from
        # "omitted"); resolve here too — this block never reaches train().
        ckpt_dir = Path(args.checkpoint_dir if args.checkpoint_dir is not None else CHECKPOINTS_DIR)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Shared helper with train() so the fingerprint dict can't drift.
        _, bptt_horizon, batch_size = compute_batch_dims(args.num_envs)

        cfg = build_train_config(args, batch_size=batch_size, bptt_horizon=bptt_horizon)
        (ckpt_dir / "config.json").write_text(json.dumps(cfg, sort_keys=True, indent=2,
                                                         default=str))
        print(f"[DumpConfig] Wrote {ckpt_dir / 'config.json'}")
        sys.exit(0)

    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        smoke_test()
    elif args.train:
        train(args)
    elif args.record:
        record_checkpoint = args.checkpoint
        record_save_path = args.record_out
        if getattr(args, "name", None):
            import re

            name_arg = args.name
            # If already fully resolved (starts with DDMMYY-N- pattern), use as-is
            if re.match(r"^\d{6}-\d+-", name_arg):
                resolved = name_arg
            else:
                # Find the most recently modified checkpoint dir ending with -<name>
                suffix = f"-{name_arg}"
                candidates = ([
                    d for d in CHECKPOINTS_DIR.iterdir() if d.is_dir() and d.name.endswith(suffix)
                ] if CHECKPOINTS_DIR.exists() else [])
                if not candidates:
                    raise FileNotFoundError(
                        f"No checkpoint dir in {CHECKPOINTS_DIR} ending with '{suffix}'")
                resolved = max(candidates, key=lambda d: os.path.getmtime(d)).name
            record_checkpoint = str(CHECKPOINTS_DIR / resolved / "dust2_policy.pt")
            record_save_path = str(RECORDINGS_DIR / f"{resolved}.rrd")
            print(f"[Record] Resolved run name: {resolved}")
        record_episode(
            checkpoint_path=record_checkpoint,
            device=args.device,
            seed=args.seed,
            policy_mode=args.record_policy,
            save_path=record_save_path,
            map_data=args.map_data,
        )
    elif args.eval:
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            device=args.device,
            start_seed=args.seed,
            num_episodes=args.eval_episodes,
            policy_mode=args.eval_policy,
        )
    else:
        parser.print_help()
