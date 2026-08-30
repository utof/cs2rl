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
import json
import math
import multiprocessing as mp
import numbers
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
from paths import CHECKPOINTS_DIR, RECORDINGS_DIR

# Explicit re-export marker: naming AIM_DIM here is what tells ruff that the
# otherwise-unused import is intentional. A trailing per-line F401 suppression cannot
# be used instead, for the formatter reason above. Nothing does `from train import *`,
# so narrowing star-imports to this one name has no effect on any caller.
__all__ = ("AIM_DIM", )

# MUST stay a bare integer literal: scripts/exp_lib.py fingerprints the env by
# regex-grepping `OBS_DIM = <int>` out of this file's source text (env_fingerprint),
# so it cannot be an `import`. Mirrors nav.OBS_DIM / _obs_spec.OBS_DIM (generated
# from cs2_types.h); the three are cross-checked by tests/test_train_env.py:873.
# On an OBS_DIM bump, update cs2_types.h + rerun the generator, then bump this literal.
OBS_DIM = 110

# Agents per team. A bare literal ON PURPOSE, for the same class of reason as
# OBS_DIM above: train.py must stay import-light (`--dump-config` guarantees no
# torch/nav import — see _atomic_save_state_dict's docstring), and `nav` pulls
# awpy/polars/shapely (+0.6 s and a polars warning) just to read one 5.
# Cross-checked against nav.TEAM_SIZE and cs2_env.TEAM_SIZE by
# tests/test_train_env.py::test_obs_dim_constant_consistency.
TEAM_SIZE = 5

# Batch 3 (continuous aim H-PPO): state-independent log_std parameter
# for the Gaussian aim head. σ_init = 0.1 rad ≈ 5.7° matches mega-spec
# §9 lock and the H-PPO literature default. σ_min = 0.01 rad ≈ 0.6° —
# floors entropy without flooding the policy with noise; tanh+max_turn_speed
# clamp dominates the per-tick range regardless of σ. σ_max = 0.5 rad ≈ 28.6°
# — symmetric bound prevents explosion that would mask μ.
# Module-level so tests can `import train; train.LOG_STD_MIN` without poking
# at the inner Dust2Policy class. Used in build_policy() forward paths and
# in the max_entropy calc that drives the SAC-α dual loop.
LOG_STD_INIT = math.log(0.1)
LOG_STD_MIN = math.log(0.01)
LOG_STD_MAX = math.log(0.5)

# Rung 1a T1 (spec 2026-08-30): how far BELOW the run's σ cap a fresh
# aim_log_std starts. `torch.clamp` back-propagates zero gradient strictly
# outside [min, max], so a parameter initialised AT the cap is gradient-dead
# from step 0 — that is exactly what killed the Rung 1 treatment arm (init
# log 0.1 under a log 0.05 cap: σ was a constant for 10M steps and the logged
# "log σ = −2.996" was the clamp, not a measurement). 0.2 in log-space ≈ an
# 18 % σ gap: wide enough that Adam needs many steps to walk into the clamp,
# small enough that the run still trains near its intended σ.
AIM_LOG_STD_INIT_MARGIN = 0.2
# The matching floor-side head-room the cap must leave. The init sits one
# margin below the cap, so a cap closer than 2× the margin to LOG_STD_MIN puts
# the init at/below the FLOOR, where the lower clamp kills the gradient just as
# dead as the upper one. Enforcing head-room on the cap (rather than clamping
# the init up with a max(LOG_STD_MIN + m, …) floor) is deliberate: a floor
# merely moves the dead zone from one end of the band to the other, silently.
AIM_LOG_STD_CAP_MIN_HEADROOM = 2 * AIM_LOG_STD_INIT_MARGIN


def validate_aim_log_std_max(aim_log_std_max) -> float:
    """Resolve + range-check the run's aim σ cap (R0-E.3, #131).

    Returns the float cap (LOG_STD_MAX when None). Raises ValueError unless
    LOG_STD_MIN + 0.4 < cap <= LOG_STD_MAX, i.e. σ in (0.0149, 0.5].

    WHY a separate torch-free helper: make_policy() only runs after the env
    and torch are up, so a bad --aim-log-std-max used to surface ~30 s into a
    launch AND slip past `--dump-config` (the Modal/run_rung1 fingerprint
    step). main() now calls this right after parse_args(), above the
    --dump-config exit, so the fingerprint catches it.
    PITFALL (2026-08-30, rung1 sweep): the bound is INCLUSIVE at LOG_STD_MAX =
    log 0.5 = -0.693147..., so a hand-rounded "-0.6931" is > the cap by 5e-5
    and is REJECTED — pass -0.69315 (or omit the flag) for "σ cap 0.5".
    PITFALL (Rung 1a T1): the LOWER bound is no longer LOG_STD_MIN itself but
    LOG_STD_MIN + AIM_LOG_STD_CAP_MIN_HEADROOM — caps that narrow leave no room
    for the strictly-inside-the-band init (see resolve_aim_log_std_init) and
    would hand the run a gradient-dead σ. This NARROWS the accepted CLI range;
    σ caps below ~0.0149 rad (0.85°) have no experimental use (the recoil/
    hitbox scale alone is larger), so nothing legitimate is lost.
    """
    cap = float(LOG_STD_MAX if aim_log_std_max is None else aim_log_std_max)
    lo = LOG_STD_MIN + AIM_LOG_STD_CAP_MIN_HEADROOM
    if not (lo < cap <= LOG_STD_MAX):
        raise ValueError(
            f"aim_log_std_max={cap} must lie in ({lo}, {LOG_STD_MAX}] "
            f"(σ in ({math.exp(lo):.4f}, 0.5]). The lower bound is "
            f"LOG_STD_MIN + {AIM_LOG_STD_CAP_MIN_HEADROOM} rather than LOG_STD_MIN: the aim σ "
            f"is initialised {AIM_LOG_STD_INIT_MARGIN} below the cap so it starts strictly "
            f"inside the clamp band, and a cap this close to the σ floor would "
            f"put that init at or under LOG_STD_MIN={LOG_STD_MIN}, where clamp "
            f"back-propagates zero gradient and σ can never train.")
    return cap


def resolve_aim_log_std_init(cap) -> float:
    """The log σ a FRESH aim head starts at under this run's cap (Rung 1a T1).

    WHAT: min(LOG_STD_INIT, cap − AIM_LOG_STD_INIT_MARGIN). At the 5v5 default
    cap (log 0.5) that is LOG_STD_INIT unchanged — every pre-Rung-1a run keeps
    its σ=0.1 start. Under a tight cap (Rung 1a's log 0.05) it is cap − 0.2,
    i.e. σ ≈ 0.041, strictly inside [LOG_STD_MIN, cap].

    WHY: `torch.clamp(x, lo, hi)` passes gradient only for lo <= x <= hi. An
    init at or above the cap is therefore a permanently frozen σ — the policy
    samples at exactly the cap forever and `policy/aim_log_std_yaw` reports the
    cap, which reads like a converged value rather than a dead parameter. This
    is the Rung 1 defect (spec 2026-08-30 §1(iv)).

    PITFALL: this is the FRESH-construction init only. A resume restores the
    checkpoint's σ verbatim, and the BC-frozen widener has its own rule
    (reinit_frozen_aim_log_std, min(AIM_LOG_STD_RESUME_INIT, cap) — that one
    may land exactly ON the cap, gh#91, untouched by T1). Callers that need the
    value in config.json must go through this helper, never re-derive it, so
    config and policy cannot drift.
    NOTE: whenever cap >= LOG_STD_INIT + AIM_LOG_STD_INIT_MARGIN (the 5v5
    default included) the init IS LOG_STD_INIT, so reinit_frozen_aim_log_std's
    "still exactly at LOG_STD_INIT ⇒ BC-frozen" signature matches a *fresh*
    policy. That was already true before T1 and stays harmless — the widener
    only ever runs on a checkpoint being resumed, never on a fresh build.
    """
    return min(LOG_STD_INIT, float(cap) - AIM_LOG_STD_INIT_MARGIN)


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
# Fix #2: precomputed log(2π) for the analytic Normal log-prob/entropy
# replacing torch.distributions.Normal in _hybrid_sample_logits.
_LOG_2PI = math.log(2.0 * math.pi)


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
    unconditionally on every launch (src/train.py:3668), so a flag-less
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


# F8 (2026-07-06 adversarial review): per-head [start, end) column ranges of
# the flat (ACTION_MASK_DIM,) action-mask row, derived from ACTION_HEAD_SIZES
# exactly like the C side derives moff[] in compute_masks (cs2_env.h). Layout
# at time of writing: move 0-8, shoot 9-10, reload 11-12, weapon 13-15,
# use 16-17, crouch 18-19, jump 20-21.
_MASK_HEAD_SLICES = []
_off = 0
for _sz in ACTION_HEAD_SIZES:
    _MASK_HEAD_SLICES.append((_off, _off + _sz))
    _off += _sz
del _off, _sz


def _apply_action_masks(logits_list, mask):
    """Mask invalid action bins out of the per-head logits (F8).

    mask : (B, ACTION_MASK_DIM) bool/int8 tensor, 1 = valid — the C-computed
    masks from cs2_env.h compute_masks, sliced per head via _MASK_HEAD_SLICES.
    Invalid bins are filled with finfo.min/2, NOT -inf: after log_softmax the
    masked log-prob stays FINITE (≈ dtype-min/2, since any real logit is
    negligible against it), so entropy terms are exactly p·logp = 0·finite = 0
    instead of 0·(-inf) = NaN. exp(min/2 - lse) underflows to exactly 0, so
    multinomial can never draw a masked bin. The C side guarantees ≥1 valid
    bin per head per agent (dead agents get per-head no-ops), so the masked
    softmax is always well-defined — do NOT relax that invariant in C without
    revisiting this function.

    Returns a NEW list; input logits are not mutated (callers may hold them).
    """
    import torch

    masked = []
    for (lo, hi), lg in zip(_MASK_HEAD_SLICES, logits_list, strict=True):
        head_valid = mask[..., lo:hi] != 0
        fill = torch.finfo(lg.dtype).min / 2
        masked.append(lg.masked_fill(~head_valid, fill))
    return masked


# ── Masked reductions over participating rows (Rung 0, spec 2026-08-29 §2.2) ──
# WHY these are free functions and not methods on the trainer: the trainer is a
# monkey-patched PuffeRL instance (pufferl.py is a site-package and is never
# edited), so every reduction the update path needs has to live here where a
# unit test can call it without building a trainer.
# DTYPE CONTRACT used by every caller below: the *bool* [S,T] mask is for
# INDEXING (`sel[mb_part]`); the *float* copy (`mb_part.to(torch.float32)`) is
# the weight `w` these helpers take. `sel[mb_part_f]` is an IndexError and
# `masked_mean(x, bool_mask)` would work only by accident — the helpers
# `.to(x.dtype)` their weight, so pass whichever, but do not swap the two roles.
# SHAPE PITFALL: `w` is multiplied (not indexed) against `x`, so it must be
# broadcast-compatible with `x`. Reducing a FLAT (S*T,) tensor (entropy,
# per-head entropies, ratio_d/ratio_c) with an [S,T] weight silently
# broadcasts to [S, S*T] — pass the flattened weight for flat tensors.


def masked_mean(x, w):
    """Mean of x over rows where w == 1. w broadcasts to x; w.sum() == 0 ⇒ 0.

    Rung 0 §2.2: parked agent rows (noop-masked, entropy exactly 0, zero
    reward) must not enter any trainer statistic, or every all-row mean at
    n_active=1 is diluted 5×. Masked mean = (x·w).sum() / max(w.sum(), 1).
    """
    w = w.to(x.dtype)
    return (x * w).sum() / w.sum().clamp(min=1.0)


def masked_std_unbiased(x, w, mean):
    """Unbiased (n−1) std over the w == 1 rows, matching torch .std() on the subset.

    `mean` is the caller's already-computed masked_mean — passed in rather than
    recomputed so the two-pass (mean, then deviation) reduction is done once.
    PITFALL: the (n−1) clamp means a single participating row yields std 0, not
    NaN; masked_normalize_adv's +1e-8 then makes that row's normalised
    advantage 0 rather than inf.
    """
    w = w.to(x.dtype)
    n = w.sum()
    return (((x - mean)**2 * w).sum() / (n - 1.0).clamp(min=1.0)).sqrt()


def masked_normalize_adv(flat_adv, w):
    """(adv − mean) / (std + 1e-8) over participating rows; parked rows → 0.

    Uses the same unbiased std as the unmasked path so the two agree exactly
    when w is all-ones. Zeroing parked rows makes their pg contribution 0
    regardless of ratio, which is what the masked pg mean then divides out.
    """
    w = w.to(flat_adv.dtype).reshape(-1)
    m = masked_mean(flat_adv, w)
    s = masked_std_unbiased(flat_adv, w, m)
    return (flat_adv - m) / (s + 1e-8) * w


def masked_explained_variance(y_pred, y_true, part):
    """explained_variance over part == True rows (whole-buffer, Rung 0 §2.2).

    Returns nan when the participating y_true has zero variance, mirroring
    PufferLib's `torch.nan if var_y == 0` convention. `part` is the BOOL mask
    here (this one indexes rather than weights — the variance of a weighted
    tensor is not the variance of the subset).
    """
    import torch

    part = part.to(torch.bool)
    yt = y_true[part]
    yp = y_pred[part]
    if yt.numel() < 2:
        return float("nan")
    var_y = yt.var()
    if var_y == 0:
        return float("nan")
    return float(1 - (yt - yp).var() / var_y)


# ── Reward weights: the single wiring source of truth (spec 2026-08-01 §4.2) ──
# Config key == make_env kwarg name == CLI flag (dashes) — NO prefix rewriting.
# Six of the 23 do not start with `reward_` (the pbrs_* group), so any code
# that discovers weights by scanning for a `reward_` prefix is wrong by
# construction. This dict is the ONLY derivation point: build_train_config's
# keys, the CLI flags, the env-factory override dict and the §6.1 pin test all
# read it.
#
# WHY the defaults are duplicated here instead of read from the signature:
# train.py imports c_env lazily (inside functions) so `--dump-config` costs no
# map/binding import; a module-level inspect.signature(make_env) would undo
# that. tests/test_reward_weight_wiring.py pins both sides against each other,
# so the duplication cannot silently drift.
#
# PITFALL: do NOT "clean up" a value here or in make_env. These defaults ARE
# the trained baseline; an unflagged run must stay byte-identical to the
# pre-wiring env.
#
# Deliberately NOT threaded here: pbrs_gamma (threaded as a non-weight knob by
# env_knobs_from_args via resolve_gammas, R0-J), team_spirit (config-threaded
# separately), include_step_stats_in_info (issue #100, out of scope).
REWARD_WEIGHT_DEFAULTS = {
                                                       # ── non-potential (hackable — sweep with care) ──
    "reward_win": 1.0,
    "reward_kill": 0.3,
    "reward_death": 0.1,
    "reward_bombsite_entry": 0.3,
    "reward_plant_bonus": 3.0,
    "reward_plant_base": 0.2,
    "reward_plant_progress_scale": 0.05,
    "reward_plant_interrupted": 0.1,
    "reward_defuse": 0.2,
    "reward_shot_penalty": 0.005,
    "reward_ct_survival": 0.001,                       # the CT stall drip — A1 arm sets this to 0.0
    "reward_inaction": 0.0005,
    "reward_win_t_detonation": 5.0,
    "reward_win_t_elimination": 3.0,
    "reward_win_ct_defuse": 5.0,
    "reward_win_ct_timeout": 4.0,                      # NOTE: exceeds ct_elimination — A1b arm
    "reward_win_ct_elimination": 3.0,
                                                       # ── PBRS-potential (optimum-safe per Ng et al. 1999; tunes
                                                       # equilibrium selection in MARL per Devlin & Kudenko 2011) ──
    "pbrs_alive_weight": 0.3,
    "pbrs_hp_weight": 0.002,
    "pbrs_site_weight": 0.2,
    "pbrs_bomb_progress_weight": 0.3,
    "pbrs_nav_weight_t": 0.04,
    "pbrs_nav_weight_ct": 0.15,
}
REWARD_WEIGHT_KEYS = tuple(REWARD_WEIGHT_DEFAULTS)


def reward_overrides_from_args(args) -> dict:
    """Build the env-side reward-weight override dict from parsed args.

    WHAT: {kwarg_name: float} for all 23 weights, taking the CLI value when
    present and the make_env default otherwise.

    WHY a shared helper: build_train_config (provenance) and train()'s env
    factory (behavior) MUST agree exactly. train_config is not available at
    env-construction time — it is built after pufferlib.vector.make — so both
    call sites derive from this one function instead.

    PITFALL: the getattr fallbacks are load-bearing for harness/dump-config
    args objects that predate these flags; do not tighten them to attribute
    access.

    PITFALL: argparse's type=float accepts "nan"/"inf", so the finiteness
    check belongs here — the one funnel both call sites pass through. A NaN
    weight otherwise surfaces hours into a run as a NaN loss with no clue
    which knob produced it, so raise at startup and name the key.
    """
    overrides = {}
    for k, d in REWARD_WEIGHT_DEFAULTS.items():
        v = float(getattr(args, k, d))
        if not math.isfinite(v):
            raise ValueError(f"reward weight {k}={v} is not finite; "
                             "pass a real number (this would poison the loss silently)")
        overrides[k] = v
    return overrides


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


def _atomic_save_state_dict(state_dict, path):
    """torch.save via sibling .tmp + os.replace so a crash never corrupts ``path``.

    WHY: the periodic save in train() overwrites ONE file (dust2_policy.pt)
    every --save_every_sec. The training box's GPU is known to fall off the
    PCI bus under thermal load (hard crash, 2026-08-13); a plain torch.save
    interrupted mid-write would leave the ONLY recovery checkpoint torn.
    os.replace() is an atomic rename on POSIX, so ``path`` always holds a
    complete checkpoint — old or new, never partial.

    PITFALL: torch is imported lazily — train.py's module level must stay
    torch-free so --dump-config keeps its no-heavy-imports guarantee.
    """
    import torch

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_dict, tmp)
    os.replace(tmp, path)


# ── R0-C (#134): full-state checkpoint / resume ───────────────────────────
# PufferLib 3.0 saves model + optimizer + step counters (pufferl.py
# save_checkpoint) and ships NO loader. Everything train.py layers on top
# (SAC-α, LR scheduler, return normaliser, warm-start machine, self-play pool,
# RNGs) lives here in a third file, train_state.pt, next to PufferLib's two.
# Budget keys are allowlisted ON PURPOSE: the whole point of --resume-run is
# `while trainer.epoch < trainer.total_epochs` (train()) continuing past a
# crash, and run_rung1.sh's retry loop must be able to extend --timesteps.
# check_resume_config prints a WARN line for every allowlisted key that changed.
# PITFALL: n_active_per_team is deliberately NOT allowlisted — it changes the
# unit of global_step (participating agent-steps) and the participating buffer
# layout, so a resumed run under a different value would be nonsense.
# PITFALL (R0-D #135): `seed` is NOT allowlisted either. A resumed run's RNG
# streams come back from train_state.pt (restore_train_state), so a changed
# --seed would be silently ignored for python/numpy/torch yet still re-seed
# the freshly built envs — an inconsistent, unlabelled run. Refuse instead;
# pass the original --seed (config.json has it) when resuming.
RESUME_CONFIG_ALLOWLIST = frozenset(
    {"data_dir", "device", "run_id", "total_timesteps", "participating_timesteps"})
# R0-C: epochs between full-state checkpoint sets. ONE constant for the CLI
# default and build_train_config's getattr fallback (harness / SimpleNamespace
# callers without the flag) — two literals drifted once (final review #7).
DEFAULT_CHECKPOINT_INTERVAL = 200
# Trainer attrs of the warm-start entropy machine + SAC target (all set in
# _patch_trainer_with_return_norm). Plain Python scalars/None — pickled as-is.
_WARMSTART_ATTRS = ("_batch1_warmstart_phase", "_batch1_last_entropy_mean",
                    "_batch1_log_alpha_reset_done", "_batch1_current_target_entropy",
                    "_batch1_warmstart_h_anchor", "_batch1_warmstart_h0",
                    "_batch1_warmstart_warn_epoch")


def _rng_state_dict():
    """Snapshot python/numpy/torch(+cuda) RNG states. Env xorshift32 state is
    NOT included (lives in C; see load_full_resume's WARN)."""
    import torch
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state()
    }
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state_all()
    return st


def seed_everything(seed: int) -> None:
    """R0-D (#135): seed every host-side RNG that _rng_state_dict snapshots.

    WHAT: random, numpy (legacy global), torch CPU and — when available — all
    CUDA devices. This is the MIRROR of _rng_state_dict/_rng_load_state_dict:
    the same RNG set, one fresh-seed path here and one resume path there. Add a
    new RNG to all three or resume will silently diverge from a fresh run.
    WHY a function: train() used to inline these calls, and no default-suite
    test noticed when they were dropped (only the 2-subprocess e2e test did;
    test_two_runs_same_seed_identical[3] now runs in the default suite).
    test_seed_everything_is_deterministic pins it now.
    PITFALL: seeds only — it does NOT set torch.use_deterministic_algorithms
    or cudnn flags, so CUDA runs are seeded but not bit-exact reproducible.
    Env xorshift32 streams are seeded separately via env_seed_base.
    """
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_load_state_dict(st):
    """Inverse of _rng_state_dict. A CUDA state saved on a GPU box is skipped
    silently on a CPU-only resume (device is allowlisted)."""
    import torch
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch_cpu"])
    if "torch_cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["torch_cuda"])


def collect_train_state(trainer, self_play_mgr) -> dict:
    """Everything train.py adds on top of PuffeRL's trainer_state.pt, CPU-side
    so the file is device-agnostic. Requires _patch_trainer_with_return_norm
    (the _log_alpha_tensor / _alpha_optimizer / _ret_* aliases)."""
    return {
                                                                       # Set identity (fix round 1, review #1): load_full_resume refuses a
                                                                       # sidecar whose epoch/global_step disagree with trainer_state.pt — a
                                                                       # crash between the three writes must never pair epoch-N weights with
                                                                       # epoch-(N-1) optimizer/α/scheduler state silently.
        "epoch": int(trainer.epoch),
        "global_step": int(trainer.global_step),
        "log_alpha": trainer._log_alpha_tensor.detach().cpu().clone(),
        "alpha_optimizer": trainer._alpha_optimizer.state_dict(),
                                                                       # CosineAnnealingLR is stepped per epoch, not a fn of global_step;
                                                                       # its T_max is overridden on restore (see restore_train_state).
        "scheduler": trainer.scheduler.state_dict(),
        "ret_mean": trainer._ret_mean.detach().cpu().clone(),
        "ret_var": trainer._ret_var.detach().cpu().clone(),
        "ret_count": trainer._ret_count.detach().cpu().clone(),
        "warmstart": {
            k: getattr(trainer, k)
            for k in _WARMSTART_ATTRS
        },
        "self_play": self_play_mgr.state_dict(),
        "rng": _rng_state_dict(),
    }


def restore_train_state(trainer, self_play_mgr, state: dict):
    """In-place restore of collect_train_state's dict.

    PITFALL: `_ret_*` and `log_alpha` are closure-locals aliased onto the
    trainer (_patch_trainer_with_return_norm) — copy_() into them, never
    rebind, or the closure keeps training on its own stale copy.
    """
    import torch                                                       # local ON PURPOSE: train.py module scope stays torch-free
    with torch.no_grad():
        trainer._log_alpha_tensor.data.copy_(state["log_alpha"].to(
            trainer._log_alpha_tensor.device))
        trainer._ret_mean.copy_(state["ret_mean"].to(trainer._ret_mean.device))
        trainer._ret_var.copy_(state["ret_var"].to(trainer._ret_var.device))
        trainer._ret_count.copy_(state["ret_count"].to(trainer._ret_count.device))
    trainer._alpha_optimizer.load_state_dict(state["alpha_optimizer"])
                                                                       # CosineAnnealingLR.state_dict() carries T_max, so a wholesale load would
                                                                       # re-install the OLD horizon; past it the recursive cosine (torch
                                                                       # lr_scheduler CosineAnnealingLR.get_lr) bounces the LR back UP — a
                                                                       # periodic LR on any allowlisted --timesteps extension. Keep
                                                                       # last_epoch/_step_count, adopt the NEW trainer's horizon (pufferl.py:
                                                                       # total_timesteps // batch_size). When the horizon changed, drop the LR
                                                                       # onto the closed-form cosine at the restored epoch so the extension
                                                                       # continues annealing from there (the recursive form scales the PREVIOUS
                                                                       # lr, and an already-finished run sits at lr=0, which would otherwise stay
                                                                       # 0 forever). Same-budget resumes leave the optimizer lr untouched — the
                                                                       # round trip stays bit-exact.
    sd = dict(state["scheduler"])
    old_t_max, sd["T_max"] = sd["T_max"], trainer.scheduler.T_max
    trainer.scheduler.load_state_dict(sd)
    if old_t_max != trainer.scheduler.T_max:
        sch = trainer.scheduler
        for group, base in zip(trainer.optimizer.param_groups, sch.base_lrs, strict=True):
            group["lr"] = sch.eta_min + (base - sch.eta_min) * (
                1 + math.cos(math.pi * sch.last_epoch / sch.T_max)) / 2
        sch._last_lr = [g["lr"] for g in trainer.optimizer.param_groups]
    for k, v in state["warmstart"].items():
        setattr(trainer, k, v)
    self_play_mgr.load_state_dict(state["self_play"])
    _rng_load_state_dict(state["rng"])


def _install_full_checkpointing(trainer, self_play_mgr):
    """Override PuffeRL.save_checkpoint on this instance (same MethodType
    pattern as _patch_trainer_with_return_norm). Differences from stock:
    no `model_path exists → return` early-out (a resumed run re-saves the
    same epoch after loading, and a crash between the model write and the
    state writes must not freeze the state files), atomic writes for all
    three files, and the train_state.pt sidecar. Still returns the model
    path — PuffeRL.close() copies it to <data_dir>/<run_id>.pt.

    WRITE ORDER is load-bearing: model → train_state → trainer_state. The
    LAST file written (trainer_state.pt) names the model (model_name) and
    carries the epoch the sidecar is checked against, so a crash anywhere
    in the sequence leaves a set that resolve_resume_run/load_full_resume
    either accept whole (all three from the same epoch) or refuse — never
    a newer model with an older optimizer."""

    def _save_checkpoint(self):
        run_id = self.logger.run_id
        path = Path(self.config["data_dir"]) / run_id
        path.mkdir(parents=True, exist_ok=True)
        model_name = f"model_{self.epoch:06d}.pt"
        model_path = path / model_name
        _atomic_save_state_dict(self.uncompiled_policy.state_dict(), model_path)
        _atomic_save_state_dict(collect_train_state(self, self_play_mgr), path / "train_state.pt")
        _atomic_save_state_dict(
            {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "agent_step": self.global_step,
                "update": self.epoch,
                "model_name": model_name,
                "run_id": run_id,
            }, path / "trainer_state.pt")
        return str(model_path)

    trainer.save_checkpoint = types.MethodType(_save_checkpoint, trainer)
    return trainer


def resolve_resume_run(run_dir: Path, run_id: str | None = None) -> dict:
    """Locate the checkpoint SET under <run_dir>/<run_id>/: the model named by
    trainer_state.pt['model_name'] (NOT max(model_*.pt) — a crash after the
    model write but before trainer_state.pt leaves a newer orphan model whose
    optimizer state was never saved), plus trainer_state.pt + train_state.pt.
    run_id=None ⇒ the unique subdir holding trainer_state.pt (error if 0 or
    >1) and the id is read from trainer_state.pt['run_id']. Every failure is
    a SystemExit with a [Resume] message (a stock-PuffeRL run dir has no
    train_state.pt sidecar and must not die with a raw traceback)."""
    import torch
    run_dir = Path(run_dir)
    if run_id is None:
        cands = sorted(p.parent for p in run_dir.glob("*/trainer_state.pt"))
        if len(cands) != 1:
            raise SystemExit(f"[Resume] expected exactly one <run_id>/trainer_state.pt under "
                             f"{run_dir}, found {len(cands)}: pass --run-id")
        run_id = cands[0].name
    d = run_dir / run_id
    ts_path = d / "trainer_state.pt"
    if not ts_path.exists():
        raise SystemExit(f"[Resume] {ts_path} not found")
    ts = torch.load(ts_path, map_location="cpu", weights_only=False)
    model_name = ts.get("model_name")
    if not model_name:
        raise SystemExit(f"[Resume] {ts_path} has no model_name — not a full-state checkpoint")
    model_path = d / model_name
    if not model_path.exists():
        raise SystemExit(f"[Resume] {ts_path} names {model_name} but {model_path} is missing")
    newer = [p.name for p in d.glob("model_*.pt") if p.name > model_name]
    if newer:
        print(f"[Resume] WARN: ignoring {len(newer)} model file(s) newer than {model_name} "
              f"({', '.join(sorted(newer))}) — their optimizer state was never saved")
    st_path = d / "train_state.pt"
    if not st_path.exists():
        raise SystemExit(f"[Resume] {st_path} not found — run predates full-state checkpointing "
                         "(R0-C); use --resume <model.pt> for a weights-only restart")
    return {
        "run_id": ts.get("run_id", run_id),
        "model_path": model_path,
        "trainer_state_path": ts_path,
        "train_state_path": st_path
    }


def check_resume_config(run_dir: Path, new_cfg: dict, allow=RESUME_CONFIG_ALLOWLIST):
    """Hard-error unless every key of the on-disk config.json equals new_cfg
    except `allow`. Compared through the JSON round-trip (sort_keys, default=str)
    so tuples/Paths compare the way they were written. MUST run before the
    unconditional config.json rewrite in train()."""
    cfg_path = Path(run_dir) / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"[Resume] {cfg_path} not found — cannot guard against a config change")
    old = json.loads(cfg_path.read_text())
    new = json.loads(json.dumps(new_cfg, sort_keys=True, default=str))

    def _same(k):
        a, b = old.get(k, "<missing>"), new.get(k, "<missing>")
        # data_dir: train() rewrites args.checkpoint_dir to the ABSOLUTE run
        # dir on --resume-run, so a run launched with a relative/--name path
        # would otherwise WARN on every resume and train users to ignore it.
        if k == "data_dir" and isinstance(a, str) and isinstance(b, str):
            return Path(a).resolve() == Path(b).resolve()
        return a == b

    changed = sorted(k for k in (old.keys() | new.keys()) if not _same(k))
    for k in changed:
        if k in allow:
            print(
                f"[Resume] WARN: allowlisted config key changed: {k}: {old.get(k)!r} -> {new.get(k)!r}"
            )
    diffs = [k for k in changed if k not in allow]
    if diffs:
        raise SystemExit("[Resume] config.json mismatch on non-allowlisted keys: " + ", ".join(
            f"{k}: {old.get(k, '<missing>')!r} -> {new.get(k, '<missing>')!r}" for k in diffs))


def check_checkpoint_set(model_path, ts: dict, st: dict) -> None:
    """Set consistency (review #1): model_<epoch>.pt, trainer_state.pt and
    train_state.pt must all come from ONE epoch. The model is already the one
    trainer_state.pt names (resolve_resume_run); this checks the sidecar's
    own epoch/global_step against both. SystemExit, never a bare assert
    (stripped under -O). Pure in its inputs so the mismatch cases are unit-
    testable without a trainer."""
    model_epoch = int(Path(model_path).stem.split("_")[-1])
    ts_epoch, st_epoch = int(ts["update"]), int(st.get("epoch", -1))
    ts_step, st_step = int(ts["global_step"]), int(st.get("global_step", -1))
    if not (model_epoch == ts_epoch == st_epoch and ts_step == st_step):
        raise SystemExit(f"[Resume] inconsistent checkpoint set under {Path(model_path).parent}: "
                         f"model epoch {model_epoch}, trainer_state epoch {ts_epoch} "
                         f"(global_step {ts_step}), train_state epoch {st_epoch} "
                         f"(global_step {st_step}) — a crash mid-save; resume from an older "
                         "complete set or use --resume <model.pt>")


def load_full_resume(trainer, self_play_mgr, paths: dict) -> dict:
    """Policy weights are loaded by the --resume path (resolve_resume_split);
    this restores optimizer, counters and the sidecar. Returns
    {"resumed_from_step", "epoch"}. Must run AFTER every trainer patch so the
    aliases restore_train_state writes into exist."""
    import torch
    ts = torch.load(paths["trainer_state_path"],
                    map_location=trainer.config["device"],
                    weights_only=False)
    st = torch.load(paths["train_state_path"], map_location="cpu", weights_only=False)
    check_checkpoint_set(paths["model_path"], ts, st)
    trainer.optimizer.load_state_dict(ts["optimizer_state_dict"])
    trainer.global_step = int(ts["global_step"])
    trainer.epoch = int(ts["update"])
    restore_train_state(trainer, self_play_mgr, st)
    print("[Resume] WARN: resume not bit-exact for env sampling (env xorshift32 state is "
          "not checkpointed).")
    return {"resumed_from_step": trainer.global_step, "epoch": trainer.epoch}


def check_resume_metrics_bound(resumed_step: int, last_row_step: int, checkpoint_interval: int,
                               steps_per_epoch: int) -> tuple[int, int]:
    """Spec §R0-C sanity bound of a restored global_step against the last
    metrics.jsonl row of the same run_id. Returns (lo, hi); raises SystemExit
    (never a bare assert — stripped under -O) when outside.

    Both sides are checkpoint_interval epochs wide: checkpoints fire every
    checkpoint_interval epochs unconditionally (pufferl.py train loop), but
    a row is only written when PuffeRL builds `logs`, which it throttles to
    ≥0.25 s since the last log — so with fast epochs the last row can be up
    to checkpoint_interval-1 epochs BEHIND the checkpoint (hence + rather
    than the one-epoch upper side the brief sketched), and the checkpoint
    can be up to checkpoint_interval epochs behind the last row."""
    lo = last_row_step - checkpoint_interval * steps_per_epoch
    hi = last_row_step + checkpoint_interval * steps_per_epoch
    if not lo <= resumed_step <= hi:
        raise SystemExit(f"[Resume] restored global_step {resumed_step} outside [{lo}, {hi}] "
                         f"around last metrics row {last_row_step} "
                         f"(checkpoint_interval={checkpoint_interval}, "
                         f"steps/epoch={steps_per_epoch})")
    return lo, hi


def compute_batch_dims(num_envs: int) -> tuple[int, int, int]:
    """Return (agents_per_env, bptt_horizon, batch_size) used by both training
    and --dump-config. Single source of truth so the fingerprint dict captured
    pre-training cannot drift from what train() actually runs.
    """
    agents_per_env = 10
    bptt_horizon = 64
    batch_size = num_envs * agents_per_env * bptt_horizon
    return agents_per_env, bptt_horizon, batch_size


# ── Rung 1a T3 (spec 2026-08-30 §2): opponent-team control mode ────────────
#   "self" — the opponent team is driven by the current (or, on a self-play
#            epoch, a past) policy and its rows train exactly like the hero's.
#            Every pre-Rung-1a run.
#   "noop" — the opponent team is a STATIONARY STATUE: discrete bin 0 on every
#            action head, zero aim delta, and its rows are excluded from
#            participation (no global_step, no gradient, no loss term) and from
#            the --timesteps budget.
OPPONENT_MODES = ("self", "noop")


def resolve_opponent_mode(args) -> str:
    """Read ``--opponent`` off an args object, with the legacy-args fallback.

    WHAT: returns "self" or "noop"; raises ValueError on anything else. The
    getattr default keeps harness / ``--dump-config`` / older SimpleNamespace
    callers (which predate the flag) on the historical self-play behaviour —
    same contract as env_knobs_from_args' stance knobs.

    WHY validate here instead of trusting argparse's ``choices=``:
    build_train_config is also reached from hand-built namespaces (the test
    harness, sweep scripts), where a typo'd mode would fall through to the
    `self` budget formula while the evaluate() statue override silently never
    fires — a run that looks healthy and trains on the wrong horizon.
    """
    mode = getattr(args, "opponent", "self")
    if mode not in OPPONENT_MODES:
        raise ValueError(f"opponent={mode!r} must be one of {OPPONENT_MODES}")
    return mode


def assert_opponent_self_play_compatible(opponent: str, self_play_enabled: bool) -> None:
    """Startup guard: ``--opponent noop`` requires ``--no-self-play``.

    WHY (spec 2026-08-30 §2 T3, "team constancy"): the statue is whichever team
    SelfPlayManager.opponent_team names. That starts at "ct" but self-play
    bookkeeping flips it every ``phase_length`` (50) epochs via
    maybe_switch_teams — which would hand the hero the OTHER side of a
    spawn-asymmetric map partway through the run while the participation vector
    (built ONCE, statically, for the initial hero team) kept masking the old
    side. Self-play also mixes past-policy actions into the opponent rows,
    which is the opposite of a statue.

    Called from main() ABOVE the --dump-config exit (so the Modal/sweep
    fingerprint step rejects the combination in milliseconds, like
    validate_aim_log_std_max) and again at the top of train() for programmatic
    callers. Raising ValueError matches validate_aim_log_std_max's precedent.
    """
    if opponent == "noop" and self_play_enabled:
        raise ValueError("--opponent noop requires --no-self-play. The statue team is "
                         "SelfPlayManager.opponent_team, which self-play flips every "
                         "phase_length epochs (and mixes past policies into), while the "
                         "participating-rows vector is built once for the initial hero "
                         "team — the two would silently disagree mid-run.")


def build_participating_rows(num_envs: int,
                             n_active: int,
                             opponent_mode: str = "self",
                             hero_team: str = "t") -> np.ndarray:
    """Static per-run participation vector over agent ROWS (Rung 0 §2.2 / T3).

    WHAT: bool array of length ``num_envs * agents_per_env`` in the env-row-major
    layout the vecenv hands back (10 rows per env: T at slots 0-4, CT at 5-9).
    True = the row TRAINS — it counts toward ``global_step``, is scattered into
    ``trainer.participating`` and survives every masked reduction in train().

      - ``opponent_mode="self"``: slots 0..n_active-1 of BOTH teams. Bit-identical
        to the pre-T3 expression ``(i % TEAM_SIZE) < n_active``.
      - ``opponent_mode="noop"``: those slots of the HERO team only. The statue
        team neither learns nor is counted.

    WHY a shared helper: this vector used to be built by two copies of the same
    expression (train() and train_test_harness), and it sits UPSTREAM of
    global_step, the buffer scatter, every masked loss and
    losses/participating_rows. Patching one copy would have left the headline
    harness test green while production still trained on both teams — precisely
    the silent failure this experiment cannot afford.

    PITFALLS
    - Derived from ARGS, while the envs are built separately from the same
      args; train() keeps an explicit driver-env agreement assert beside its
      call site, and _patch_trainer_with_hybrid_aim re-checks the length.
    - ``hero_team`` must stay the complement of SelfPlayManager.opponent_team
      (use SelfPlayManager.initial_hero_team()). Under "noop" that team is
      constant for the whole run because the mode forbids self-play; a
      disagreement would mask the statue's rows IN and the learner's rows OUT
      while every metric still looked plausible.
    """
    if opponent_mode not in OPPONENT_MODES:
        raise ValueError(f"opponent_mode={opponent_mode!r} must be one of {OPPONENT_MODES}")
    if hero_team not in ("t", "ct"):
        raise ValueError(f"hero_team={hero_team!r} must be 't' or 'ct'")
    if not 1 <= n_active <= TEAM_SIZE:
        raise ValueError(f"n_active={n_active} outside 1..{TEAM_SIZE}")
    agents_per_env, _, _ = compute_batch_dims(num_envs)
    # The T/CT halves are what make `hero_team` meaningful; assert rather than
    # assume, so a future roster change fails here instead of silently marking
    # half of some other layout.
    assert agents_per_env == 2 * TEAM_SIZE, (agents_per_env, TEAM_SIZE)
    slot = np.arange(num_envs * agents_per_env) % agents_per_env
    rows = (slot % TEAM_SIZE) < n_active
    if opponent_mode == "noop":
        rows &= (slot < TEAM_SIZE) if hero_team == "t" else (slot >= TEAM_SIZE)
    return rows


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


def build_train_config(args, batch_size: int, bptt_horizon: int) -> dict:
    """Construct the train_config dict identically to the training path.

    Extracted so --dump-config can produce the exact same dict without
    spinning up an env. Any future changes to training HPs must live here,
    not duplicated in train(). Keep this semantically identical to what
    train() used to build inline — scripts/run_experiment.py hashes this
    dict as a fingerprint, so silent drift here invalidates experiment
    provenance.
    """
    # ── Warm-start entropy mode (spec 2026-08-01) ──
    # Two-phase override for BC-warm-started runs: GRACE (alpha clamped to the
    # ceiling, alpha-optimizer paused, hard entropy floor disabled) then RAMP
    # (target re-anchored at measured H, rising to base_frac*max; floor still
    # off, re-arms at ramp end). At the default ceiling of 0.0 the GRACE window
    # turns the entropy bonus fully OFF — pure PPO on reward, not merely a small
    # bonus. Explicit flag, NO auto-detection: config.json is dumped BEFORE the
    # resume block loads the checkpoint, so an auto-set flag would be recorded
    # False — provenance poison (spec finding 3). The getattr defaults keep
    # harness/dump-config args objects (which may predate these flags) working.
    # Read out here rather than inline in the dict below: yapf snaps that dict's
    # comment column past the longest line in the block, so long inline
    # getattr() calls would re-indent every comment in it.
    #
    # CONTRACTS for the trainer wiring (do not re-derive these downstream):
    # 1. Both *_steps are trainer.global_step units — PARTICIPATING agent steps
    #    (Rung 0: 5× fewer raw env steps per unit at n_active=1), the same
    #    counter entropy_target_warmup_steps and target_entropy_schedule use.
    # 2. No CLI validation, deliberately. The pure schedule helper
    #    warmstart_entropy_state treats ramp_steps <= 0 as "jump straight to OFF
    #    at grace end" (tested: test_ramp_steps_zero_goes_straight_to_off), so 0
    #    is a legal no-ramp request, NOT a divide-by-zero; negative values are
    #    documented as equivalent to 0. A negative grace_steps simply means the
    #    grace window ends immediately (the test is step < grace_steps).
    # 3. The ceiling applies to the LINEAR effective alpha —
    #    torch.clamp(alpha, max=ceiling) — never to log_alpha, since the 0.0
    #    default would be log(0) = -inf. The hard entropy floor must be gated
    #    off before/independently of the ceiling clamp, or the floor's
    #    clamp(min=0.5) and this clamp(max=0.0) fight each other.
    ws_entropy = bool(getattr(args, "warmstart_entropy", False))
    ws_grace = int(getattr(args, "warmstart_grace_steps", 5_000_000))
    ws_ramp = int(getattr(args, "warmstart_ramp_steps", 10_000_000))
    ws_alpha_ceil = float(getattr(args, "warmstart_alpha_ceiling", 0.0))

    # ── Reward weights + symmetrization (spec 2026-08-01) ──
    # Read out here (not inline in the dict) for the same reason as the
    # warmstart block above: yapf snaps the returned dict's comment column to
    # its longest line. Grouping/labelling of the weights lives on
    # REWARD_WEIGHT_DEFAULTS, the single source of truth.
    reward_weights = reward_overrides_from_args(args)
    reward_symmetrize = bool(getattr(args, "reward_symmetrize", False))

    # ── TAG diagnostic (spec 2026-08-13 §4.1) ──
    # getattr fallbacks keep harness/dump-config args objects that predate
    # these flags working, same pattern as the warmstart block above.
    # NOTE: these keys change exp_lib.behavior_hash for ALL future runs
    # (hash covers sorted config.json) — recorded decision, spec §4.1.
    tag_diagnostic = bool(getattr(args, "tag_diagnostic", False))
    tag_every = int(getattr(args, "tag_every", 5))

    # ── Batch 7 heads split (spec 2026-08-13 §2) ──
    # getattr fallback, same pattern as the TAG block above. This records the
    # FLAG, not the resolved architecture: a flag-less crash-resume of a split
    # run writes false here on purpose, which is precisely why the analyzer
    # reads the per-epoch split/active metric rather than config.json
    # (spec §3.4). Adding this key also shifts exp_lib.behavior_hash for all
    # future runs — recorded decision, spec §6.
    tct_split_heads = bool(getattr(args, "tct_split_heads", False))
    # Trunk twin (spec 2026-08-15): same FLAG-not-architecture contract as
    # heads. A flag-less crash-resume of a trunk-split run writes false
    # here on purpose; the analyzer reads split/trunk_active instead.
    tct_split_trunk = bool(getattr(args, "tct_split_trunk", False))

    # ── Rung 0 §2.2 + Rung 1a T3: participating-units budget ──
    # --timesteps is the PARTICIPATING agent-step budget (what the policy
    # actually learns from), NOT the raw row count. PufferLib's epoch cap
    # (total_epochs = total_timesteps // batch_size, pufferl.py:168-170, which
    # also sets the cosine-LR T_max) counts RAW buffer rows, so the raw budget
    # handed to it is scaled by (rows per env) / (participating rows per env).
    # Both numbers are recorded: done_training in _train_with_return_norm
    # compares global_step (participating units) against
    # participating_timesteps, and the epoch clause catches the floor-division
    # slack.
    # T3: the pre-T3 formula (args.timesteps * TEAM_SIZE // n_active) hardcoded
    # "2 participating rows per env per active slot", i.e. BOTH teams. Under
    # --opponent noop only the hero team participates, so that formula ends the
    # run at HALF the requested budget with exit 0 — a silent short run
    # (verified against smoke-v1c/s0: ~491,520 hero steps for a 1M request).
    # The generalised form below reduces EXACTLY to the old one under
    # --opponent self (10t/2n and 5t/n are the same rational, so the floor
    # divisions agree for every t and n), keeping the `self` path bit-identical
    # while putting cosine-LR T_max on the real horizon under noop:
    # 1M requested at n_active=1, num_envs=256 ⇒ total_timesteps = 10M ⇒
    # 61 epochs (batch 163,840; 61 × 16,384 = 999,424 hero steps).
    # PITFALL: adding these keys shifts exp_lib.behavior_hash for all future
    # runs (the hash covers sorted config.json) — recorded decision, same as
    # the TAG/tct keys above, and the same for `opponent`, `jump_enabled` and
    # `aim_log_std_init` below (plus the σ weight-decay exclusion of T1, which
    # changes behaviour for every run without touching config.json at all).
    # (the raw budget is bound to a local, not inlined in the dict below, for
    # the same yapf reason as the warmstart block above: a long value
    # expression inside the dict re-indents every trailing comment in it.)
    knobs = env_knobs_from_args(args)
    n_active = knobs["n_active_per_team"]
    assert 1 <= n_active <= TEAM_SIZE, n_active
    opponent = resolve_opponent_mode(args)
    _part_per_env = n_active * (1 if opponent == "noop" else 2)
    raw_timesteps = args.timesteps * (TEAM_SIZE * 2) // _part_per_env
    # R0-E.3/4 (#131): aim-head knobs. CLI gives "on"/"off" for the entropy
    # bonus (argparse choices); the test harness passes a bool — accept both so
    # neither caller has to know the other's spelling. None cap ⇒ LOG_STD_MAX,
    # so config.json always records the EFFECTIVE cap (provenance), never null.
    # None of these are in RESUME_CONFIG_ALLOWLIST on purpose: a resume with a
    # changed aim knob is a different experiment and must be refused.
    _aeb = getattr(args, "aim_entropy_bonus", "on")
    aim_entropy_bonus = _aeb if isinstance(_aeb, bool) else (_aeb == "on")
    _cap = getattr(args, "aim_log_std_max", None)
    aim_log_std_max = float(LOG_STD_MAX if _cap is None else _cap)
    # Rung 1a T1: the σ a fresh aim head actually starts at — DERIVED from the
    # cap, so it is provenance, not a knob (there is no --aim-log-std-init).
    # Recorded because the gate's "σ moved ≥ 0.1" reading is |raw − init| and
    # the reader must not have to re-derive the formula. Goes through the same
    # helper build_policy uses so config.json and the policy cannot drift.
    # PITFALL: this key AND the σ weight-decay exclusion
    # (isolate_aim_log_std_param_group) shift exp_lib.behavior_hash for all
    # future runs — recorded decision, same convention as the TAG/tct keys
    # above. The weight-decay change is a real behaviour change for EVERY run,
    # not just capped ones; the init only moves when cap < LOG_STD_INIT + 0.2.
    aim_log_std_init = resolve_aim_log_std_init(aim_log_std_max)

    # R0-H: env LABEL from the resolved map name. The CLI always sets args.map
    # (above the --dump-config exit); the harness / older SimpleNamespace
    # callers have no `map` attr and keep the historical "cs2-dust2". Not
    # allowlisted for --resume-run: a different map is a different experiment.
    map_name = getattr(args, "map", None) or "dust2"
    # R0-J: --gamma / --pbrs-gamma. Same helper as env_knobs_from_args so the
    # env's PBRS discount and the PPO discount cannot resolve differently.
    gamma, pbrs_gamma = resolve_gammas(args)
    cfg = {
                                                                       # Core PPO
        "env": f"cs2-{map_name}",
        "device": args.device,
        "seed": args.seed,
        "total_timesteps": raw_timesteps,
        "participating_timesteps": args.timesteps,
        "n_active_per_team": n_active,
        "pin_pitch": knobs["pin_pitch"],
        "crouch_enabled": knobs["crouch_enabled"],
                                                                       # Rung 1a T2b: provenance for --jump-enabled. Like the other Rung 0
                                                                       # knobs it is NOT allowlisted for --resume-run (a run that masks
                                                                       # jump is a different experiment) and adding it shifts
                                                                       # exp_lib.behavior_hash for all future runs — recorded decision.
        "jump_enabled": knobs["jump_enabled"],
                                                                       # Rung 1a T3: "self" (both teams learn) or "noop" (statue opponent —
                                                                       # hero-team-only participation AND budget, see raw_timesteps above).
                                                                       # NOT a make_puffer_env knob: the statue is enforced trainer-side, in
                                                                       # the patched evaluate(), so the env is identical either way. Not
                                                                       # allowlisted for --resume-run — a different opponent is a different
                                                                       # experiment.
        "opponent": opponent,
                                                                       # R0-G: recorded as given (None ⇒ env default), read from args
                                                                       # directly so None survives — env_knobs_from_args drops None keys.
        **{
            a: getattr(args, a, None)
            for a, _ in _R0G_KNOBS
        },
        "aim_entropy_bonus": aim_entropy_bonus,
        "aim_log_std_max": aim_log_std_max,
                                                                       # Rung 1a T1: DERIVED from the cap (no CLI flag of its own); the
                                                                       # gate reads σ movement as |raw − init|, so it is provenance.
        "aim_log_std_init": aim_log_std_init,
                                                                       # R0-I: fixed-baseline eval cadence (0 = off). Provenance only —
                                                                       # not allowlisted for --resume-run (Task 7 rule: new CLI flags are
                                                                       # config keys, not allowlist entries).
        "eval_interval": int(getattr(args, "eval_interval", 0) or 0),
        "batch_size": batch_size,
        "bptt_horizon": bptt_horizon,
        "minibatch_size": 8192,
        "max_minibatch_size": 8192,
        "update_epochs": 3,
        "learning_rate": 3e-4,
        "gamma": gamma,
        "pbrs_gamma": pbrs_gamma,                                      # R0-J: provenance; not allowlisted for --resume-run
        "gae_lambda": 0.95,
        "clip_coef": 0.15,
        "vf_coef": 0.5,
        "vf_clip_coef": None,
        "ent_coef": 0.1,                                               # fallback; adaptive alpha overrides
        "max_grad_norm": 0.5,
        "target_kl": 0.03,
        "use_rnn": True,
        "weight_decay": 1e-4,
                                                                       # Extras required by PuffeRL constructor
        "compile": False,
        "compile_mode": "default",
        "compile_fullgraph": False,
        "cpu_offload": False,
        "torch_deterministic": False,
        "optimizer": "adam",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "anneal_lr": True,
        "checkpoint_interval":
        int(getattr(args, "checkpoint_interval",
                    DEFAULT_CHECKPOINT_INTERVAL)),                     # R0-C: --checkpoint-interval
        "data_dir": args.checkpoint_dir,
        "precision": "float32",
        "prio_alpha": 0.0,
        "prio_beta0": 1.0,
        "vtrace_rho_clip": 1.0,
        "vtrace_c_clip": 1.0,
                                                                       # ── Entropy-target schedule (finding 4 residual, 2026-07-06 review) ──
                                                                       # Linear ramp warmup_frac→base_frac (× max_entropy ≈ 8.21 nats) over
                                                                       # warmup_steps, held constant after; consumed by the SAC-style α
                                                                       # controller via _scheduled_target_entropy. Previous hardcoded values
                                                                       # (0.7→0.5) kept the target so high the controller steered the policy
                                                                       # toward near-uniform indefinitely (the 30M degenerate run). 0.35·max
                                                                       # ≈ 2.87 nats still allows broad exploration but permits commitment.
                                                                       # PITFALL: keep base_frac ABOVE 0.3 — the hard entropy floor in
                                                                       # _patch_trainer_with_return_norm clamps α ≥ 0.5 when H < 0.3·max;
                                                                       # a base target below the floor would make the two mechanisms fight.
        "entropy_target_warmup_frac": 0.5,
        "entropy_target_base_frac": 0.35,
        "entropy_target_warmup_steps": 10_000_000,
                                                                       # ── Warm-start entropy mode: see the comment above ──
        "warmstart_entropy": ws_entropy,
        "warmstart_grace_steps": ws_grace,
        "warmstart_ramp_steps": ws_ramp,
        "warmstart_alpha_ceiling": ws_alpha_ceil,
        "reward_symmetrize": reward_symmetrize,
        "tag_diagnostic": tag_diagnostic,
        "tag_every": tag_every,
        "tct_split_heads": tct_split_heads,
        "tct_split_trunk": tct_split_trunk,
    }

    # ── Reward wiring: 23 make_env weights, verbatim key names ──
    # (grouped + annotated on REWARD_WEIGHT_DEFAULTS, the single source of
    # truth). Merged via an explicit collision check rather than a trailing
    # `**reward_weights` splat: a splat in last position would SILENTLY
    # overwrite an existing config key if someone ever adds a make_env weight
    # named like one of the keys above, and the resulting config would look
    # perfectly well-formed. This assert is the only thing that actually
    # catches that — the pin test compares against make_env's signature and
    # would not notice a collision on this side. Key order does not matter:
    # every config.json / fingerprint dump uses sort_keys=True.
    assert not (cfg.keys() & reward_weights.keys()), (
        "reward weight name collides with an existing config key: "
        f"{sorted(cfg.keys() & reward_weights.keys())}")
    cfg.update(reward_weights)
    return cfg


def _scheduled_target_entropy(config, global_step: int, max_entropy: float) -> float:
    """Config-driven entropy target for the SAC-style α controller.

    Single source for both the patch-time seed and the per-train()-call
    recompute in _patch_trainer_with_return_norm — keeping them identical
    means a checkpoint-resumed trainer seeds at its true scheduled value
    instead of a hardcoded warmup constant. `config` is anything with
    .get() (PuffeRL config or a plain dict); missing keys fall back to the
    build_train_config defaults so harness/older-checkpoint configs keep
    working.
    """
    from train_helpers_batch1 import target_entropy_schedule
    return target_entropy_schedule(
        global_step,
        max_entropy,
        warmup_end=config.get("entropy_target_warmup_steps", 10_000_000),
        warmup_high_frac=config.get("entropy_target_warmup_frac", 0.5),
        base_frac=config.get("entropy_target_base_frac", 0.35),
    )


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
    env = make_puffer_env(seed=42)
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


def make_puffer_env(team_spirit=None,
                    record_fn=None,
                    buf=None,
                    seed=0,
                    episode_stats=True,
                    map_data=None,
                    include_step_stats_in_info=False,
                    pbrs_gamma=None,
                    reward_overrides=None,
                    reward_symmetrize=False,
                    n_active_per_team=TEAM_SIZE,
                    pin_pitch=0,
                    crouch_enabled=1,
                    jump_enabled=1,
                    round_time=None,
                    laser_range=None,
                    max_turn_speed=None,
                    auto_reset=True):
    """Create the native C PufferEnv used by smoke/train/eval.

    ``auto_reset`` (R0-I, Task 13): forwarded to make_env. The training
    vecenv keeps the default True; the fixed-baseline evaluator passes False
    so the terminal tick's C state / episode_stats are still readable after
    step() returns (with auto-reset they would already be the next spawn).

    ``include_step_stats_in_info`` (Task 6a, utof/cs2rl#7): when True the env
    emits ``info = [{"step_stats": StepStatsView}]`` on every tick so trainer
    patches (Task 6c onward) can read per-channel raw reward fields. Defaults
    to False so production code paths that don't consume step_stats (e.g. eval
    scripts, viz) stay zero-cost.

    ``pbrs_gamma`` (finding 2 / N3, 2026-07-06 review): PBRS shaping discount.
    None (default) uses the env-side default, which is pinned to the training
    gamma (0.999) and drift-guarded by test_pbrs_gamma_matches_training_gamma.
    Pass explicitly only for experiments that also change the training gamma —
    the two MUST move together or PBRS loses policy-invariance.

    ``reward_overrides`` (spec 2026-08-01 §4.2): optional {kwarg: value} dict
    over train.REWARD_WEIGHT_KEYS, forwarded verbatim to make_env. None
    (default) means every weight keeps its make_env default, so all existing
    callers (eval, record, smoke, tests) are unaffected.

    ``reward_symmetrize`` (spec 2026-08-01 §4.3): when True the env applies the
    zero-sum post-step transform r_i' = 0.5*(r_i - mean of the opposing five).
    It is a dedicated parameter, NOT a reward_overrides key — it is a bool knob
    rather than a weight, and the override validator above rejects it by name.
    Defaults False so eval/record/smoke keep raw, comparable reward numbers;
    only the training factory turns it on.

    ``n_active_per_team`` / ``pin_pitch`` / ``crouch_enabled`` (Rung 0, spec
    2026-08-29 §2.1) and ``jump_enabled`` (Rung 1a, spec 2026-08-30 T2b):
    non-weight env knobs forwarded verbatim to make_env. They are NOT
    reward_overrides keys for the same reason reward_symmetrize is not.
    The defaults reproduce the pre-Rung-0 env exactly (full 5v5, pitch live,
    crouch and jump enabled), so every non-training caller (eval, record,
    smoke, viz) is unaffected. Training callers get them from
    env_knobs_from_args(args) — do NOT re-derive them from args anywhere else,
    or config.json provenance and the envs that actually ran can disagree.

    ``round_time`` / ``laser_range`` / ``max_turn_speed`` (Rung 0 R0-G):
    sim knobs forwarded verbatim to make_env. None (default) ⇒ make_env falls
    back to the nav.py constant, so this function's default output is
    byte-identical to before (sim fingerprints at defaults unchanged). Set
    only via env_knobs_from_args, which omits None-valued knobs. PITFALL:
    max_turn_speed rescales the aim action (policy.max_turn_speed is read
    from the driver env at build time); assert_max_turn_speed_agreement
    guards the pairing at train() startup.
    """
    from c_env.cs2_env import make_env as make_c_env

    if record_fn is not None:
        raise ValueError("record_fn is only supported by the Python recording env")
    kwargs = {}
    if pbrs_gamma is not None:
        kwargs["pbrs_gamma"] = pbrs_gamma
    if reward_overrides:
        # Validate here, not at make_env: an unknown key would otherwise
        # surface as a TypeError inside a forked vecenv worker, where the
        # traceback is far from the mistake. This is the LAST boundary before
        # the C env, so it also re-checks value sanity — direct callers
        # (train_test_harness, future sweep scripts) can hand us a dict that
        # never passed through reward_overrides_from_args.
        unknown = set(reward_overrides) - set(REWARD_WEIGHT_KEYS)
        if unknown:
            import difflib
            hints = []
            for key in sorted(unknown):
                near = difflib.get_close_matches(key, REWARD_WEIGHT_KEYS, n=1)
                if near:
                    hints.append(f"{key!r} — did you mean --{near[0].replace('_', '-')}?")
            raise ValueError(f"unknown reward override keys: {sorted(unknown)}. " +
                             (" ".join(hints) + " " if hints else "") +
                             f"Valid keys: {sorted(REWARD_WEIGHT_KEYS)}. Non-weight env knobs "
                             "(pbrs_gamma, reward_symmetrize) are NOT overrides — they have "
                             "dedicated make_puffer_env parameters.")
        for key, val in reward_overrides.items():
            # numbers.Real, not a bare float() call: float("0.3") succeeds, so
            # a string weight from a YAML sweep file would sail through here
            # and only misbehave at the ctypes boundary. bool is a Real in
            # Python, hence the explicit exclusion.
            if isinstance(val, bool) or not isinstance(val, numbers.Real):
                raise ValueError(f"reward override {key}={val!r} is not a real number "
                                 f"(got {type(val).__name__})")
            fval = float(val)
            if not math.isfinite(fval):
                raise ValueError(f"reward weight {key}={fval} is not finite; "
                                 "pass a real number (this would poison the loss silently)")
            kwargs[key] = fval
    return make_c_env(
        seed=seed,
        team_spirit=team_spirit,
        buf=buf,
        map_data=map_data,
        include_step_stats_in_info=include_step_stats_in_info,
        reward_symmetrize=reward_symmetrize,
        n_active_per_team=n_active_per_team,
        pin_pitch=pin_pitch,
        crouch_enabled=crouch_enabled,
        jump_enabled=jump_enabled,
        round_time=round_time,
        laser_range=laser_range,
        max_turn_speed=max_turn_speed,
        auto_reset=auto_reset,
        **kwargs,
    )


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
    policy_env = make_puffer_env()

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
        # This helper is eval/inspection only — used by record_episode and the
        # Python-side scripted rollout. Its callers don't currently consume the
        # continuous (Δyaw) component, so the sampled cont_t is dropped on the
        # floor. The cont_action is still SAMPLED (sample mode) so the policy
        # state advances identically to training; we just don't emit it. If a
        # future eval path needs Δyaw, return (act_dict, cont_dict) — keeping
        # the int-action signature for now to avoid touching every caller.
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
            # (~line 679), so it's already bounded — no extra clamp needed.
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
        env = make_puffer_env(seed=seed)
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
    return make_puffer_env(team_spirit=team_spirit, map_data=map_data)


DEFAULT_GAMMA = 0.999                  # R0-J: the historical PPO discount; --gamma default


def resolve_gammas(args) -> tuple[float, float]:
    """Return ``(gamma, pbrs_gamma)`` from the args object.

    WHAT: ``gamma`` is ``args.gamma`` (default DEFAULT_GAMMA for harness /
    dump-config args objects that predate the flag); ``pbrs_gamma`` is
    ``args.pbrs_gamma`` when given, else ``gamma``.

    WHY one helper: build_train_config (provenance + the PPO discount) and
    env_knobs_from_args (the env's PBRS discount) must agree on the SAME
    resolution rule — PBRS is only policy-invariant (Ng et al.) when
    γ_pbrs == γ, and before R0-J the two lived as unrelated literals (train.py
    0.999 vs cs2_env.py 0.999) held together by a single drift test.
    ``--pbrs-gamma`` exists ONLY for experiments that deliberately break the
    pairing; a run that omits it always gets γ_pbrs = γ.

    PITFALL: both values are config.json keys and NOT in
    RESUME_CONFIG_ALLOWLIST — changing either on --resume-run is refused.
    """
    gamma = getattr(args, "gamma", None)
    gamma = DEFAULT_GAMMA if gamma is None else float(gamma)
    if not (0.0 < gamma < 1.0):
        raise ValueError(f"--gamma must be in (0, 1), got {gamma}")
    pbrs_gamma = getattr(args, "pbrs_gamma", None)
    pbrs_gamma = gamma if pbrs_gamma is None else float(pbrs_gamma)
    if not (0.0 < pbrs_gamma <= 1.0):
        raise ValueError(f"--pbrs-gamma must be in (0, 1], got {pbrs_gamma}")
    if pbrs_gamma != gamma:
        print(f"WARNING: --pbrs-gamma {pbrs_gamma} != --gamma {gamma}: PBRS shaping is no "
              "longer policy-invariant (Ng et al.); only do this on purpose.")
    return gamma, pbrs_gamma


# (args attr, make_puffer_env kwarg) — single source for env_knobs_from_args
# AND build_train_config, so a knob added to one cannot be missed by the other
# (config.json would then silently under-record the experiment).
_R0G_KNOBS = (("round_time_ticks", "round_time"), ("laser_range", "laser_range"),
              ("max_turn_speed", "max_turn_speed"))


def env_knobs_from_args(args) -> dict:
    """Non-weight env knobs (Rung 0) as make_puffer_env kwargs.

    WHY one helper: build_train_config (provenance), build_train_env_factory
    (the envs) and train()'s participating-row vector must all read the same
    values; a second copy of these getattr defaults would be the next
    silent-baseline bug (the class of bug build_train_env_factory exists to
    prevent for reward weights). getattr defaults keep harness/dump-config
    args objects — which predate these flags — working.

    PITFALL: this returns make_puffer_env KWARG names, not config keys. It is
    splatted straight into make_puffer_env(**env_knobs); renaming a key here
    without renaming the parameter there raises TypeError inside a forked
    vecenv worker, far from the mistake.

    R0-G knobs (round_time_ticks → round_time, laser_range, max_turn_speed):
    None-valued ones are OMITTED from the dict rather than forwarded as None,
    so the env's own nav.py default applies and config.json records None
    instead of a duplicated constant that would silently drift from nav.py.

    R0-J: ``pbrs_gamma`` is ALWAYS present (resolved via resolve_gammas, so it
    equals the training gamma unless --pbrs-gamma was given). Unlike the R0-G
    knobs it is never omitted: the env default (cs2_env.py 0.999) would
    silently disagree with a non-default --gamma.
    """
    knobs = {
        "n_active_per_team": int(getattr(args, "n_active_per_team", TEAM_SIZE)),
                                                                                 # R0-E.2: `or 0` — args.pin_pitch is None on the CLI until train()
                                                                                 # resolves it from map flatness (see the pin_pitch block in train()).
        "pin_pitch": int(getattr(args, "pin_pitch", 0) or 0),
        "crouch_enabled": int(getattr(args, "crouch_enabled", 1)),
                                                                                 # Rung 1a T2b: same shape as crouch_enabled — always present (1 =
                                                                                 # today's env), never omitted, so a legacy args object cannot
                                                                                 # silently leave the env on a different jump setting than config.json.
        "jump_enabled": int(getattr(args, "jump_enabled", 1)),
        "pbrs_gamma": resolve_gammas(args)[1],
    }
    for arg_name, env_name in _R0G_KNOBS:
        v = getattr(args, arg_name, None)
        if v is not None:
            knobs[env_name] = v
    return knobs


def build_env_factory(*,
                      shared_ts,
                      map_data,
                      reward_overrides=None,
                      reward_symmetrize=False,
                      env_knobs=None):
    """Return the per-env factory callable handed to pufferlib.vector.make.

    WHAT: a closure over the shared team-spirit Value, the preloaded map data
    and the reward-weight overrides; it builds one Cs2Env and attaches the
    cont-action / action-mask shared-memory views.

    WHY the overrides are CLOSURE state and not per-env kwargs (spec §4.2):
    the returned factory's parameters are all explicitly named, and anything
    else is now a hard error (see below). Reward keys added to the
    _per_env_kwargs list in train() used to be silently dropped, which would
    have made every experiment arm train the default weights. Closure state
    crosses the fork boundary the same way shared_ts and map_data already do
    (proven).

    WHY module-level rather than nested in train(): the §6.3 test has to
    exercise this exact code path, and a closure defined inside train() is
    unreachable without launching a run.

    reward_symmetrize (spec §4.3) rides along as closure state for the same
    reason, but through its OWN parameter rather than the overrides dict: it is
    a bool knob, not a weight, and make_puffer_env's validator rejects it as an
    override key by name.

    env_knobs (Rung 0 §2.1) is the same story once more: a dict of non-weight
    make_puffer_env kwargs (n_active_per_team / pin_pitch / crouch_enabled)
    from env_knobs_from_args, closure state so it survives the fork. None ⇒
    make_puffer_env's defaults, i.e. the pre-Rung-0 env.

    PITFALL (review finding 1): reward_overrides and reward_symmetrize reach
    ONLY the training env factory — the --smoke/--record/--eval paths call
    make_puffer_env without them, so `--smoke --reward-ct-survival 0.0`
    silently runs default weights. For symmetrization that is deliberate:
    eval/record must report raw, cross-run-comparable rewards. Known
    limitation, stated here and in the final report; do not fix in this branch.
    PITFALL: `seed or 0` is intentional — pufferlib passes seed=None for some
    backends. Keep it.
    R0-D (#135) `_seed`: train() routes the per-env seed through env_kwargs
    (`_seed = env_seed_base(--seed) + i`) because pufferlib.vector.make takes
    `seed` as ITS OWN named parameter and never forwards it to the backend —
    `make(..., seed=X)` is a silent no-op and every env lands on pufferlib's
    default base (env i -> seed i) regardless of --seed. When `_seed` is given
    it wins over pufferlib's `seed`; the legacy path is unchanged otherwise.
    """

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
        env = make_puffer_env(team_spirit=shared_ts,
                              buf=buf,
                              seed=_seed if _seed is not None else (seed or 0),
                              map_data=map_data,
                              reward_overrides=reward_overrides,
                              reward_symmetrize=reward_symmetrize,
                              **(env_knobs or {}))
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
    """The training path's env factory: reward overrides derived from args.

    WHY this exists as its own function (review fix 2): it is the seam between
    the CLI and the envs. Inlined in train() it was untestable without
    launching a run, so nothing caught a regression that dropped the overrides
    — exactly the silent-baseline failure this whole change is guarding
    against. test_train_uses_build_train_env_factory pins train() to it.

    Derives the overrides from the same helper build_train_config uses, so
    config.json provenance and the envs' actual weights cannot disagree. Same
    for reward_symmetrize: read off args with the identical getattr default
    build_train_config uses, so the logged "reward_symmetrize" key always
    describes the envs that actually ran. Same again for the Rung 0 env knobs
    via env_knobs_from_args — build_train_config records exactly what this
    hands the envs, and train() asserts the built driver_env agrees.
    """
    return build_env_factory(shared_ts=shared_ts,
                             map_data=map_data,
                             reward_overrides=reward_overrides_from_args(args),
                             reward_symmetrize=bool(getattr(args, "reward_symmetrize", False)),
                             env_knobs=env_knobs_from_args(args))


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
            # at make_puffer_env time. Stored as a buffer (no grad, not a
            # learnable param, follows .to(device)). T5 carry-forward (I-1):
            # reuse the `driver_env` helper resolved at line ~526 instead of
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
                # can land outside the tanh band. The C env (cs2_env.h:129)
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


# ── SECTION: Network Health Monitoring ────────────────────────────────────


def compute_network_health(model, device):
    """Compute network health metrics for logging.

    Returns a dict with:
      - health/weight_norm_<name>: L2 norm of each named parameter
      - health/lstm_h_norm: norm of LSTM hidden state (TODO: requires trainer access)

    Alarm thresholds (informational, not enforced here):
      - dead neurons > 20% (not tracked — would require forward hooks)
      - effective rank < 30 (not tracked — expensive)
      - lstm_h_norm > 50
    """

    metrics = {}

    # Weight norms per named parameter
    for name, param in model.named_parameters():
        safe_name = name.replace(".", "_")
        metrics[f"health/weight_norm_{safe_name}"] = param.norm().item()

    # TODO: LSTM hidden state norm requires access to trainer's stored LSTM state,
    # which is not easily accessible from outside PufferLib's training loop.
    # Would need trainer.policy or similar. Skipping for now.

    return metrics


def log_aim_log_std(policy, logs):
    """Emit policy/aim_log_std_* into `logs` under BOTH architectures (spec §3.6).

    WHAT — the exact meaning of every emitted key, per architecture:

      legacy policy (one `aim_log_std`):
        policy/aim_log_std_{yaw,pitch}         the CLAMPED parameter
        policy/aim_log_std_{yaw,pitch}_raw     the UNCLAMPED parameter

      split policy (`aim_log_std_t` + `aim_log_std_ct`):
        policy/aim_log_std_{yaw,pitch}_{t,ct}      that team's CLAMPED copy
        policy/aim_log_std_{yaw,pitch}_{t,ct}_raw  that team's UNCLAMPED copy
        policy/aim_log_std_{yaw,pitch}         MEAN of the two CLAMPED copies
        policy/aim_log_std_{yaw,pitch}_raw     MAX of the two UNCLAMPED copies

    Under pin_pitch (aim_dim_mask[1] == 0) every `pitch` key above is omitted.

    WHY the raw twin (Rung 1a T1, spec 2026-08-30 §3): the clamped key is
    censored at the cap, so a σ that the optimizer has pushed past the cap —
    the state in which clamp zeroes its gradient and σ is dead — is
    indistinguishable from a σ sitting happily AT the cap. The gate reads
    `policy/aim_log_std_yaw_raw` for both of its σ questions: "did σ move ≥ 0.1
    from its init" (⇒ the continuous head receives gradient at all) and "is raw
    ≤ cap" (⇒ the movement measurement is still meaningful). health/
    weight_norm_aim_log_std already exposes a raw NORM, but only every 5 epochs
    and unsigned/aggregated — per-row, per-dim and signed is what the gate
    needs.

    WHY the legacy-named `_raw` key is a MAX under the split while its clamped
    twin stays a MEAN: the two keys answer different questions and must be
    aggregated differently. The clamped key reports the σ the policy actually
    used, and the mean of the two copies is the honest summary of that. The raw
    key exists solely to answer "has any σ overshot the cap and gone gradient-
    dead", and a mean HIDES exactly that: with cap = −2.9957, a T copy at
    cap + 0.5 = −2.4957 (dead) averaged with a healthy CT copy reads −3.2479,
    i.e. below the cap, so the pre-flight `raw ≤ cap` check passes on a frozen
    σ. Max is the aggregation that answers the overshoot question truthfully.

    PITFALL — what the split `_raw` key does NOT promise: for the *movement*
    question the max is only conservative in one direction. Two copies that
    both moved up, or any copy that moved up, show through; a single copy that
    moved only DOWN while the other sat at its init is invisible in the max
    (max == init ⇒ "no movement"). Read the per-team `_t_raw`/`_ct_raw` keys
    whenever per-copy movement is the question. This does not affect the Rung
    1a gate, which runs the legacy architecture, where the key is the exact
    unclamped parameter.

    WHY the legacy keys survive as a mean rather than being replaced: the T7
    acceptance gate greps the status line for `aim_log_std_pitch=` (see
    format_train_status), and every dashboard/analysis consumer reads those
    two names. Adding the per-team keys alongside is what exposes the actually
    interesting Batch 7 signal — whether the teams learn different aim noise.

    WHY this is a function and not three inline lines in the outer loop: the
    pre-Batch-7 code read policy.aim_log_std unconditionally, which raises
    AttributeError on a split policy before the run writes a single metrics
    row. Extracting it makes that path directly testable (spec §5 test 10,
    which re-review N3 flagged as untested).

    PITFALL: CLAMP EACH COPY, THEN AVERAGE — never average then clamp. The
    forward path clamps per copy (spec §3.2), so a mean-then-clamp here would
    report a σ the policy never used whenever one copy sits outside the band.

    Architecture is detected from the policy object (hasattr aim_log_std_t),
    never from config — config.json is rewritten every launch and lies after a
    flag-less resume (spec §3.4).
    """
    # train.py imports torch lazily inside functions (module import stays cheap
    # for the CLI/help paths) — keep that convention here.
    import torch

    # R0-E.3: clamp to the RUN's cap (policy.aim_log_std_max), not the module
    # constant — otherwise the log would report a σ the forward never used.
    cap = float(getattr(policy, "aim_log_std_max", LOG_STD_MAX))
    # R0-E.2 (#131): when pitch is pinned (aim_dim_mask[1] == 0) the pitch σ
    # is a dead parameter — never sampled, never in log_prob_c, never
    # updated. SKIP its keys (not NaN: NaN survives json.dumps only as the
    # non-standard `NaN` token and would read as a live-but-broken signal on
    # a dashboard). Every consumer already tolerates absence:
    # format_train_status uses logs.get(..., 0.0); the T7 gate is 5v5-only.
    mask = getattr(policy, "aim_dim_mask", None)
    pitch_live = mask is None or float(mask[1]) != 0.0
    with torch.no_grad():
        if hasattr(policy, "aim_log_std_t"):
            raw_t = policy.aim_log_std_t.detach().cpu().numpy()
            raw_ct = policy.aim_log_std_ct.detach().cpu().numpy()
            ls_t = torch.clamp(policy.aim_log_std_t, LOG_STD_MIN, cap).cpu().numpy()
            ls_ct = torch.clamp(policy.aim_log_std_ct, LOG_STD_MIN, cap).cpu().numpy()
            logs["policy/aim_log_std_yaw_t"] = float(ls_t[0])
            logs["policy/aim_log_std_yaw_ct"] = float(ls_ct[0])
            logs["policy/aim_log_std_yaw_t_raw"] = float(raw_t[0])
            logs["policy/aim_log_std_yaw_ct_raw"] = float(raw_ct[0])
            if pitch_live:
                logs["policy/aim_log_std_pitch_t"] = float(ls_t[1])
                logs["policy/aim_log_std_pitch_ct"] = float(ls_ct[1])
                logs["policy/aim_log_std_pitch_t_raw"] = float(raw_t[1])
                logs["policy/aim_log_std_pitch_ct_raw"] = float(raw_ct[1])
            clamped = 0.5 * (ls_t + ls_ct)
            # MAX, deliberately NOT the mean that the clamped key uses: the raw
            # key's job is "did any copy overshoot the cap and go gradient-
            # dead", and averaging a dead copy with a healthy one reads as
            # healthy. See the docstring for the worked counterexample and for
            # the one thing max under-reports (a copy that moved only down).
            raw = np.maximum(raw_t, raw_ct)
        else:
            raw = policy.aim_log_std.detach().cpu().numpy()
            clamped = torch.clamp(policy.aim_log_std, LOG_STD_MIN, cap).cpu().numpy()
    logs["policy/aim_log_std_yaw"] = float(clamped[0])
    logs["policy/aim_log_std_yaw_raw"] = float(raw[0])
    if pitch_live:
        logs["policy/aim_log_std_pitch"] = float(clamped[1])
        logs["policy/aim_log_std_pitch_raw"] = float(raw[1])


def compute_head_divergence(policy):
    """split/head_l2_rel/<module> — how far the two team head copies have moved apart.

    Metric (spec §4 Q3):  ‖W_t − W_ct‖ / (0.5‖W_t‖ + 0.5‖W_ct‖)

    WHY relative and not raw L2: the optimizer runs weight_decay=1e-4
    (see the Adam construction in train()), so even a copy that receives zero
    gradient keeps moving. Raw L2 therefore has no achievable null. Normalising
    by the mean norm of the two copies makes "how different are the teams'
    heads" scale-free; the honest null is still a decay-aware control (zero-
    advantage steps), which is what tests/test_tct_split.py exercises, and the
    run readout reports the TRAJECTORY, not a binary.

    Returns {} for a legacy policy — the metric is undefined with one copy,
    and emitting a fake 0.0 would read as "the teams agree" to anyone
    plotting it.

    PITFALL: modules are grouped, not per-tensor — all 7 discrete heads
    contribute to one `action_heads` number. Per-head keys would be 9 series
    per epoch of mostly-identical curves; if a per-head breakdown is ever
    needed, add it as a separate function rather than widening this one.
    """
    import torch

    if not hasattr(policy, "aim_log_std_t"):
        return {}

    def _flat(obj):
        if isinstance(obj, torch.nn.Parameter):
            return obj.detach().reshape(-1)
        return torch.cat([p.detach().reshape(-1) for p in obj.parameters()])

    out = {}
    with torch.no_grad():
        for name, mod_t, mod_ct in (
            ("action_heads", policy.action_heads_t, policy.action_heads_ct),
            ("aim_mu", policy.aim_mu_t, policy.aim_mu_ct),
            ("aim_log_std", policy.aim_log_std_t, policy.aim_log_std_ct),
        ):
            w_t, w_ct = _flat(mod_t), _flat(mod_ct)
            denom = 0.5 * float(w_t.norm()) + 0.5 * float(w_ct.norm())
            if denom > 0.0:
                # NaN in either copy propagates through the ratio — visible,
                # never masked as "teams identical".
                val = float((w_t - w_ct).norm()) / denom
            else:
                # denom == 0.0 → both copies all-zero → genuinely identical.
                # denom NaN fails both comparisons → emit NaN, not a fake 0.0.
                val = 0.0 if denom == 0.0 else float("nan")
            out[f"split/head_l2_rel/{name}"] = val
    return out


def compute_trunk_divergence(policy):
    """split/trunk_l2_rel/<module> — how far the two team trunk copies have moved apart.

    WHAT: relative L2 between the T and CT copies of encoder and lstm.
    Metric is the same formula as compute_head_divergence (spec §4 Q3):
        ‖W_t − W_ct‖ / (0.5‖W_t‖ + 0.5‖W_ct‖)
    Keys: split/trunk_l2_rel/encoder, split/trunk_l2_rel/lstm.

    WHY relative and not raw L2: the optimizer runs weight_decay=1e-4, so
    even a copy that receives zero gradient keeps moving. Raw L2 has no
    achievable null. The ratio is scale-free; the honest null is still a
    decay-aware control. Gate is hasattr(policy, "encoder_t") — the live
    architecture, never config.json — matching split/trunk_active.

    Returns {} when there is no encoder_t. The metric is undefined with
    one copy, and emitting a fake 0.0 would read as "the teams agree".

    PITFALL: modules are grouped, not per-tensor — every Linear in the
    Sequential encoder and every LSTM weight (ih/hh/bias) contribute to
    one number. A per-layer series is a separate function if ever needed.
    T=1 + zero LSTM state leaves weight_hh unmoved; that does not make
    the encoder ratio 0 after a team-asymmetric step.
    """
    import torch

    if not hasattr(policy, "encoder_t"):
        return {}

    def _flat(obj):
        if isinstance(obj, torch.nn.Parameter):
            return obj.detach().reshape(-1)
        return torch.cat([p.detach().reshape(-1) for p in obj.parameters()])

    out = {}
    with torch.no_grad():
        for name, mod_t, mod_ct in (
            ("encoder", policy.encoder_t, policy.encoder_ct),
            ("lstm", policy.lstm_t, policy.lstm_ct),
        ):
            w_t, w_ct = _flat(mod_t), _flat(mod_ct)
            denom = 0.5 * float(w_t.norm()) + 0.5 * float(w_ct.norm())
            if denom > 0.0:
                val = float((w_t - w_ct).norm()) / denom
            else:
                val = 0.0 if denom == 0.0 else float("nan")
            out[f"split/trunk_l2_rel/{name}"] = val
    return out


# ── SECTION: Timing patch ─────────────────────────────────────────────────


def _patch_trainer_with_timing(trainer):
    """Monkey-patch trainer.evaluate() and trainer.train() to record wall-clock timing.

    After each call, trainer._timing holds:
        collect_ms  — ms spent in evaluate() (env stepping + rollout collection)
        update_ms   — ms spent in train() (forward + backward + optimizer step)

    Both values are also written into the logs dict returned by train() as
    timing/collect_ms and timing/update_ms for W&B / metrics.jsonl logging.
    """
    trainer._timing = {"collect_ms": 0.0, "update_ms": 0.0}
    _orig_evaluate = trainer.evaluate
    _orig_train = trainer.train

    def _timed_evaluate(*args, **kwargs):
        t0 = time.perf_counter()
        result = _orig_evaluate(*args, **kwargs)
        trainer._timing["collect_ms"] = (time.perf_counter() - t0) * 1000.0
        return result

    def _timed_train(*args, **kwargs):
        t0 = time.perf_counter()
        result = _orig_train(*args, **kwargs)
        trainer._timing["update_ms"] = (time.perf_counter() - t0) * 1000.0
        if isinstance(result, dict):
            result["timing/collect_ms"] = trainer._timing["collect_ms"]
            result["timing/update_ms"] = trainer._timing["update_ms"]
        return result

    trainer.evaluate = _timed_evaluate
    trainer.train = _timed_train


# ── SECTION: Value target normalization ───────────────────────────────────


def _patch_trainer_with_return_norm(trainer):
    """Monkey-patch a PuffeRL instance to normalize value regression targets.

    MAPPO's strongest recommendation: normalize the returns (advantages + values)
    that the value function regresses against.  This stabilizes value learning
    especially with high gamma (0.999) where return variance is large.

    Implementation: maintain a running mean/std of returns on the training
    device; before the value loss, normalize mb_returns to zero-mean unit-std.
    The value head learns to predict normalized returns; no denormalization is
    needed (unlike PopArt) because we don't use the raw value for anything
    outside the loss.
    """
    # gh #85: BPTT zero-init exactness (Dust2Policy.forward/_lstm_bptt) is only
    # EXACT when each agent row fills exactly one buffer segment per evaluate(),
    # i.e. segments == total_agents. Upstream PuffeRL only enforces
    # total_agents <= segments (pufferl.py:83-86); our equality holds by
    # construction in compute_batch_dims but nothing asserted it — one
    # batch_size/bptt_horizon config edit away from silently-biased importance
    # ratios. Fail loudly at patch time instead.
    assert trainer.segments == trainer.total_agents, (
        f"segments ({trainer.segments}) != total_agents ({trainer.total_agents}): "
        "BPTT zero-init exactness broken — revisit batch_size/bptt_horizon "
        "(compute_batch_dims) or the _lstm_bptt initial-state design. See gh #85.")

    import time
    import types
    from collections import defaultdict

    import torch
    from pufferlib.pufferl import compute_puff_advantage

    # Warm-start entropy mode (spec 2026-08-01). Function-local like every
    # other import here — train.py has no module-level train_helpers_batch1
    # import. _train_with_return_norm is nested inside this function, so it
    # picks these up as closure freevars. WS_RAMP is not needed here.
    from train_helpers_batch1 import WS_GRACE, WS_OFF, warmstart_entropy_state

    # Running stats for return normalization (Welford-style, torch tensors)
    device = trainer.config["device"]
    _ret_mean = torch.zeros(1, device=device)
    _ret_var = torch.ones(1, device=device)
    _ret_count = torch.zeros(1, device=device)

    # ── Batch 1 Task 9a: force-reset return-norm stats + expose on trainer ──
    # WHAT: zero _ret_mean/_ret_count and set _ret_var=1 in-place at patch
    #   apply time, then attach the tensors to the trainer instance.
    # WHY: Task 6c symlog-compresses rewards before they enter the rollout
    #   buffer, so mb_returns = advantages + values lives in symlog space.
    #   The return-norm running stats must therefore start fresh — carrying
    #   stale raw-scale stats from a pre-Batch-1 checkpoint would contaminate
    #   the symlog-space computation throughout warmup.
    # PITFALL: use in-place .zero_()/.fill_() rather than reassigning the
    #   names. The trainer attribute below is meant to be the same tensor
    #   reference the closure mutates, so `_update_return_stats` writes are
    #   visible via trainer._ret_var (and conversely tests reading the attr
    #   see the live value, not a stale snapshot).
    _ret_mean.zero_()
    _ret_var.fill_(1.0)
    _ret_count.zero_()
    trainer._ret_mean = _ret_mean
    trainer._ret_var = _ret_var
    trainer._ret_count = _ret_count
    # ──────────────────────────────────────────────────────────────────────

    # ── ADAPTIVE ENTROPY (Lagrangian / SAC-style alpha) ────────────────────
    # Batch 3: max_entropy = sum of discrete max entropies + closed-form
    # Gaussian entropy at σ = exp(LOG_STD_MAX). Used for entropy-coefficient
    # annealing schedules (target_entropy ramp in Task 9A) and the SAC-α
    # dual loop's bookkeeping. Discrete heads contribute log(N_i) each;
    # the Gaussian contributes 0.5·log(2πe·σ²) per AIM_DIM — using σ_max
    # is the conservative ceiling, since the policy's actual σ is clamped
    # ≤ exp(LOG_STD_MAX) in every forward call.
    # R0-E (#131): the ceiling follows the run — σ cap (policy.aim_log_std_max),
    # number of live aim dims (aim_dim_mask.sum(): 1 when pitch is pinned) and
    # the entropy-bonus switch (0 continuous entropy when the Gaussian is
    # excluded from the objective, so the target/floor track the discrete
    # heads only). getattr defaults keep pre-R0-E policies/configs working.
    max_entropy_discrete = sum(np.log(n) for n in ACTION_HEAD_SIZES)
    _cap = float(getattr(trainer.policy, "aim_log_std_max", LOG_STD_MAX))
    _n_aim = float(getattr(trainer.policy, "aim_dim_mask", torch.ones(AIM_DIM)).sum())
    _bonus = bool(trainer.config.get("aim_entropy_bonus", True))
    max_entropy_continuous = (_n_aim * 0.5 * np.log(2 * np.pi * np.e * np.exp(_cap)**2)) \
        if _bonus else 0.0
    max_entropy = max_entropy_discrete + max_entropy_continuous
    # Task 9A: target_entropy is no longer a static scalar — it's recomputed
    # each train() call from a linear ramp 0.7→0.5*max_entropy across
    # [0, 10_000_000] global steps (see target_entropy_schedule). The live
    # value lives in the closure-local _t9_target_entropy in
    # _train_with_return_norm and on trainer._batch1_current_target_entropy.
    entropy_floor = 0.3 * max_entropy  # collapse threshold
    import math

    log_alpha = torch.tensor([math.log(0.1)], requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-4)
    # R0-C (#134): the train_state.pt sidecar needs both; they are closure-
    # locals, so alias them here. Restore MUST be in place
    # (`log_alpha.data.copy_`) — rebinding the attribute would leave the
    # closure (and alpha_optimizer's param list) on the old tensor.
    trainer._log_alpha_tensor = log_alpha
    trainer._alpha_optimizer = alpha_optimizer

    # Task 9A/9B: trainer-level state for target_entropy schedule + log_alpha
    # reset. Attached to the trainer (not closure-local) so:
    #   - tests can inspect/pin _batch1_max_entropy and current_target_entropy
    #   - the wandb log layer can read _batch1_current_target_entropy without
    #     reaching into the closure of _train_with_return_norm.
    # _batch1_log_alpha_reset_done is the idempotency flag for Task 9B —
    # the first train() call after this patch is applied resets log_alpha to
    # log(ent_coef); every later train() call leaves log_alpha alone so the
    # SAC dual-gradient loop can do its job.
    trainer._batch1_max_entropy = float(max_entropy)
    trainer._batch1_log_alpha_reset_done = False
    # Seed from the schedule at the CURRENT global_step (not a hardcoded
    # warmup constant) so checkpoint-resumed trainers start consistent;
    # the per-train()-call recompute below overwrites it every call anyway.
    trainer._batch1_current_target_entropy = _scheduled_target_entropy(
        trainer.config, trainer.global_step, float(max_entropy))
    # Pre-init effective_alpha + grad_norm metrics (utof/cs2rl#16). The
    # post-loop reads in _train_with_return_norm refresh these, but if the
    # target_kl early-break trips on mb=0 OR no accumulation boundary fires,
    # the local names are never bound — leaving the trainer attrs
    # AttributeError on first read. Seeding them with sane defaults here
    # turns those edge cases into "stale-from-previous-call" instead of a
    # crash, and the post-loop refresh overwrites whenever the loop runs
    # all the way through.
    trainer._batch1_effective_alpha = float(trainer.config["ent_coef"])
    trainer._batch1_grad_norm = 0.0

    # ── Warm-start entropy mode state (spec 2026-08-01) ────────────────────
    # h_anchor: mean policy entropy captured at grace end (None until then);
    # last_entropy_mean: previous update's post-divisor losses["entropy"] —
    # the ONLY valid anchor source (there is no entropy EMA in this codebase,
    # and trainer.losses is refreshed only inside the throttled log-flush
    # block, so it can be several updates stale — spec finding 4).
    # h0: first update's mean entropy, denominator of warmstart_h_over_h0
    # (grace collapse watch — with the floor disabled AND alpha~0 the run
    # has no anti-collapse guard, spec finding 6).
    trainer._batch1_warmstart_h_anchor = None
    trainer._batch1_warmstart_h0 = None
    trainer._batch1_warmstart_phase = WS_OFF
    trainer._batch1_last_entropy_mean = None
    trainer._batch1_warmstart_warn_epoch = -10**9

    # ──────────────────────────────────────────────────────────────────────

    def _update_return_stats(returns_flat):
        nonlocal _ret_mean, _ret_var, _ret_count
        with torch.no_grad():
            n = returns_flat.numel()
            if n == 0:
                return
            batch_mean = returns_flat.mean()
            batch_var = returns_flat.var(unbiased=False)
            batch_count = torch.tensor(float(n), device=device)

            delta = batch_mean - _ret_mean
            tot = _ret_count + batch_count
            new_mean = _ret_mean + delta * batch_count / tot
            m_a = _ret_var * _ret_count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta.pow(2) * _ret_count * batch_count / tot
            new_var = m2 / tot

            _ret_mean.copy_(new_mean)
            _ret_var.copy_(new_var)
            _ret_count.copy_(tot)

    def _normalize_returns(mb_returns, mb_part=None):
        """Return normalized copy of mb_returns; update running stats first.

        mb_part (Rung 0 §2.2): BOOL [S, T] participation mask. The running
        mean/var are updated from the PARTICIPATING rows ONLY — parked rows
        carry reward 0 and value 0, so feeding them in would drag the return
        scale toward 0 by a factor of n_active/TEAM_SIZE and shrink every
        normalized value target. None ⇒ all rows (pre-Rung-0 behaviour,
        bit-identical).
        PITFALL: the whole tensor is still normalized and returned — masking
        applies to the STATISTICS, not the output. The parked rows' value loss
        is dropped later, by the masked_mean over the same mask.
        """
        sel = mb_returns.detach()
        if mb_part is not None:
            sel = sel[mb_part]
        _update_return_stats(sel.flatten())
        std = (_ret_var + 1e-8).sqrt()
        return (mb_returns - _ret_mean) / std

    # Test hook (tests/test_parked_rows_masked.py): the closure is otherwise
    # unreachable, and the participating-rows-only stats update is exactly the
    # part worth pinning.
    trainer._normalize_returns = _normalize_returns

    def _train_with_return_norm(self):
        profile = self.profile
        epoch = self.epoch
        profile("train", epoch)
        losses = defaultdict(float)
        # Rung 0 §2.2 diagnostics. Local ints (not losses[...] entries) because
        # they are ABSOLUTE counts: anything accumulated into `losses` inside
        # the loop gets divided by _mb_run afterwards (gh#90). They are written
        # onto `losses` after that divisor.
        _floor_fires = 0               # minibatches where the entropy-floor clamp bound
        _empty_mb = 0                  # minibatches with zero participating rows (skipped)
        config = self.config
        device = config["device"]

        b0 = config["prio_beta0"]
        a = config["prio_alpha"]
        clip_coef = config["clip_coef"]
        vf_clip = config["vf_clip_coef"]
        anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
        self.ratio[:] = 1

        # Task 9A: recompute target_entropy from the linear ramp once per
        # train() call. WHY here (not inside the minibatch loop): the schedule
        # is keyed on global_step which is fixed for the duration of a single
        # train() call, so recomputing per-minibatch would burn cycles for no
        # signal. We mirror the value onto trainer._batch1_current_target_entropy
        # so the wandb log layer can read it without touching this closure.
        # Fracs/warmup_steps come from config via _scheduled_target_entropy
        # (finding 4 residual — previously hardcoded 0.7→0.5).
        # PITFALL: do NOT capture max_entropy from the outer closure here —
        # use trainer._batch1_max_entropy. Closure capture would silently break
        # if the patch were re-applied on the same trainer instance.
        _t9_target_entropy = _scheduled_target_entropy(config, self.global_step,
                                                       trainer._batch1_max_entropy)
        trainer._batch1_current_target_entropy = float(_t9_target_entropy)

        # Task 9B: one-shot log_alpha reset on the first train() call after
        # this patch. WHY: the entropy schedule + log_alpha are coupled — the
        # outer training loop can leave log_alpha at a stale value from a
        # previous run / re-init, and we need a deterministic starting point
        # of log(ent_coef) so the SAC dual-gradient loop converges from a
        # known floor. The flag is on the trainer (not the closure) so a
        # checkpoint-restored trainer that gets re-patched still resets once.
        if not trainer._batch1_log_alpha_reset_done:
            with torch.no_grad():
                log_alpha.fill_(math.log(config["ent_coef"]))
            trainer._batch1_log_alpha_reset_done = True

        # ── Warm-start entropy mode: resolve phase once per train() call ───
        # (global_step only advances in evaluate(), so it is constant here —
        # same reasoning as the Task 9A recompute above; all transitions land
        # on update boundaries.) Ordering vs Task 9B: 9B runs FIRST and sets
        # log_alpha to log(ent_coef) — exactly the operating point warm-start
        # wants (continuity comes from target==h_anchor at release, never
        # from moving log_alpha: Adam(lr=1e-4) travels ~1e-4/minibatch, so a
        # parked log_alpha is stranded — spec finding 1).
        # PITFALL: keep grace+ramp >= entropy_target_warmup_steps. OFF falls
        # through to the Task 9A schedule (see the override condition below),
        # and 9A ramps DOWNWARD — warmup_high_frac*max (0.5) at step 0 to
        # base_frac*max (0.35) at entropy_target_warmup_steps — so during
        # warmup it reads strictly ABOVE the base_frac*max the warm-start ramp
        # lands on. With defaults (grace 5M + ramp 10M = 15M >= 10M warmup) 9A
        # has already flattened at base_frac*max and the handoff is exactly
        # continuous. But e.g. grace=2M+ramp=3M puts ramp_end at 5M, where 9A
        # still reads 0.425*max: the target jumps UPWARD 0.35*max -> 0.425*max,
        # i.e. 2.87 -> 3.49 nats (+0.62, at max_entropy=8.21), at the exact
        # boundary the spec promises is clean.
        _ws_enabled = bool(config.get("warmstart_entropy", False))
        _ws_floor_active = True
        if _ws_enabled:
            _ws_grace = int(config.get("warmstart_grace_steps", 5_000_000))
            if (trainer._batch1_warmstart_h_anchor is None and self.global_step >= _ws_grace
                    and trainer._batch1_last_entropy_mean is not None):
                # one-shot anchor capture (idempotent: guarded on None).
                # Finite-check (Task 1 review): a NaN/inf entropy mean latched
                # here would poison target and alpha_loss for the whole ramp —
                # skip the capture (stay GRACE) and shout instead.
                if math.isfinite(trainer._batch1_last_entropy_mean):
                    trainer._batch1_warmstart_h_anchor = float(trainer._batch1_last_entropy_mean)
                else:
                    print(f"[Train] WARN warm-start: non-finite entropy mean "
                          f"{trainer._batch1_last_entropy_mean} at grace end — "
                          f"anchor capture skipped, staying in GRACE.")
            _ws = warmstart_entropy_state(
                self.global_step,
                grace_steps=_ws_grace,
                ramp_steps=int(config.get("warmstart_ramp_steps", 10_000_000)),
                h_anchor=trainer._batch1_warmstart_h_anchor,
                base_target=(config.get("entropy_target_base_frac", 0.35) *
                             trainer._batch1_max_entropy))
            trainer._batch1_warmstart_phase = _ws.phase
            _ws_floor_active = _ws.floor_active
            if _ws.phase != WS_OFF and _ws.target is not None:
                # Override the Task 9A schedule during the ramp AND mirror it,
                # or the wandb target trace plots the unmodified base schedule
                # (spec finding 9). Effectively RAMP-only: GRACE carries
                # target=None (no target is consumed while alpha is ceilinged).
                # PITFALL: the WS_OFF guard is load-bearing — the helper returns
                # target=base_target (NOT None) once OFF, so testing target
                # alone would pin the target at base_frac*max for the rest of
                # the run and silently flatten the tail of the 9A warmup ramp
                # whenever grace+ramp < entropy_target_warmup_steps. Falling
                # through here is what makes OFF byte-for-byte pre-feature
                # behavior at ANY config, which is what the spec promises.
                _t9_target_entropy = _ws.target
                trainer._batch1_current_target_entropy = float(_ws.target)
        else:
            # Config can be toggled off in-process (tests do this; production
            # builds the config once). Re-seed the phase so a stale GRACE can
            # never keep the alpha optimizer frozen after the mode is disabled.
            trainer._batch1_warmstart_phase = WS_OFF

        # Task 8: raw event-segment fraction (mask mean) — computed once per
        # train() call because _batch1_event_mask doesn't change inside the
        # minibatch loop. Persisted onto losses["event_oversample_fraction"]
        # AFTER the gh#90 divisor loop (per-call scalar, like ret_mean).
        _t8_event_mask = getattr(self, "_batch1_event_mask", None)
        # Masked over participating segments (participating[:, 0] is the
        # per-segment flag) so parked rows don't dilute the fraction at n<5.
        self._batch1_event_oversample_fraction = (float(
            masked_mean(_t8_event_mask.float(), self.participating[:, 0].float()))
                                                  if _t8_event_mask is not None else 0.0)

        # ── gh#90: KL early-stop bookkeeping ───────────────────────────────
        # WHAT: the target_kl early-stop is (a) gated to update-epoch
        #   boundaries and (b) decoupled from the losses/* divisor.
        # WHY (root-caused 2026-08-01, run checkpoints-20260801-022606):
        #   the old inline `break` sat before the logging block while every
        #   losses/* metric divided by the PLANNED self.total_minibatches —
        #   a truncated update silently scaled all logged losses by k/N
        #   ("importance=0.0167" was really ratio=1.0 with k=1). And because
        #   this flattened loop collapses all update_epochs into one range,
        #   one KL spike aborted passes over data never visited — harsher
        #   than standard PPO, which finishes the current epoch first.
        # HOW: losses accumulate RAW sums inside the loop and are divided by
        #   the EXECUTED count (_mb_run) after it; a KL trip sets _kl_stop
        #   and the loop exits at the next epoch boundary, so epoch 0 always
        #   completes (⇒ _mb_run >= 1, and the effective_alpha / advantages
        #   post-loop reads can no longer see an mb=0 abort).
        # PITFALL: total_minibatches need not divide update_epochs evenly
        #   (harness: 7 mbs / 3 epochs) — the boundary stride uses floor
        #   division with a >=1 clamp, never a modulo of zero.
        target_kl = config.get("target_kl", None)
        _mbs_per_epoch = max(1,
                             self.total_minibatches // max(1, int(config.get("update_epochs", 1))))
        _kl_stop = False
        _mb_run = 0

        for mb in range(self.total_minibatches):
            if _kl_stop and mb % _mbs_per_epoch == 0:
                break                  # epoch boundary: honor the KL trip
            profile("train_misc", epoch, nest=True)
            self.amp_context.__enter__()

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = compute_puff_advantage(
                self.values,
                self.rewards,
                self.terminals,
                self.ratio,
                advantages,
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )

            profile("train_copy", epoch)
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)

            # ── Batch 1 Task 8: event-biased prio_probs oversampling ──────
            # WHAT: when the segment-level event mask is populated and at
            #   least one segment contains a bomb-plant event, multiply the
            #   prio_probs of those segments by OVERSAMPLE_FACTOR before
            #   renormalising. The downstream torch.multinomial call then
            #   draws biased samples without any further changes — and the
            #   importance-sampling correction (mb_prio, consumed inside
            #   _hybrid_ppo_loss) uses the BOOSTED prio_probs[idx], so the
            #   gradient stays unbiased.
            # WHY: bomb-plant events are sparse in early training (the exact
            #   fraction is itself a Task 9 metric, reported via
            #   _batch1_event_oversample_fraction). Uniform prio sampling
            #   under-replays them; oversampling accelerates value-function
            #   fit on the rare-but-decisive transitions. Plan §Task 8
            #   target: event-mask hit-rate among sampled segments >= 25%.
            # PITFALLS:
            #   * mask absent / all-False → skip the boost so pre-Batch-1
            #     training paths and the warm-up pass before any plant
            #     happens still work (no division by zero, no NaN).
            #   * Boost the prob, not the weight — boosting `prio_weights`
            #     and re-running the (w+1e-6)/(sum+1e-6) renorm would alter
            #     the abs-advantage prior shape; multiplying prio_probs and
            #     dividing by sum keeps the prior intact on non-event rows.
            #   * Cloning before the in-place mul protects callers that
            #     might still hold a reference to the original prio_probs.
            #   * The exposed metric is the RAW event fraction (mask mean),
            #     NOT the post-boost sampled fraction. Written onto
            #     losses["event_oversample_fraction"] after the divisor
            #     loop — do not accumulate it inside this minibatch loop.
            OVERSAMPLE_FACTOR = 4.0
            if _t8_event_mask is not None and _t8_event_mask.any():
                boosted = prio_probs.clone()
                boosted[_t8_event_mask] *= OVERSAMPLE_FACTOR
                prio_probs = boosted / boosted.sum()
            # ──────────────────────────────────────────────────────────────

            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments * prio_probs[idx, None])**-anneal_beta
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            # (mb_rewards pull removed with the dead per-minibatch
            # compute_puff_advantage recompute — see finding-1 note below)
            mb_terminals = self.terminals[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]
            # Batch 3 (T5): pull continuous actions + per-factor old logprobs
            # from the parallel buffers added by _patch_trainer_with_hybrid_aim.
            # mb_logprobs (the SUM) stays the canonical "logp from rollout" for
            # KL/clipfrac diagnostics below; the per-factor halves drive the
            # per-factor PPO clip in _hybrid_ppo_loss.
            mb_cont_actions = self.cont_actions[idx]
            mb_old_logp_d = self.logprobs_d[idx]
            mb_old_logp_c = self.logprobs_c[idx]
            # F8: rollout-stored action masks (all-ones = unmasked fallback).
            # getattr for trainers built before _patch_trainer_with_hybrid_aim
            # ran (shouldn't happen in prod; keeps direct-call tests working).
            _masks_buf = getattr(self, "action_masks", None)
            mb_masks = _masks_buf[idx] if _masks_buf is not None else None

            # ── Rung 0 §2.2: participating mask for this minibatch ───────────
            # Two dtypes on purpose (see the masked_* helpers' contract):
            # mb_part is BOOL for indexing, mb_part_f/flat_part are the FLOAT
            # weights the reductions take. flat_part is for the flat (S*T,)
            # tensors (entropy, ratio_d/ratio_c, per-head entropies);
            # mb_part_f for the [S, T]-shaped ones (v_loss, value writeback).
            # Shapes: mb_part / mb_part_f are [S, T]; flat_part is (S*T,).
            mb_part = self.participating[idx]
            mb_part_f = mb_part.to(torch.float32)
            n_part = mb_part_f.sum()
            # Empty-minibatch tripwire: only reachable with non-uniform segment
            # sampling (prio_alpha != 0 or a marked event segment) that happens
            # to draw an all-parked minibatch — see spec §2.2. Skipping is the
            # right call (every reduction below would be 0/0), but it must be
            # VISIBLE, so it is counted into losses/empty_minibatches rather
            # than silently swallowed. The `continue` sits before _mb_run += 1,
            # so the gh#90 divisor keeps counting only executed minibatches.
            if n_part.item() == 0:
                _empty_mb += 1
                # No zero_grad here: accumulate_minibatches is always 1 today,
                # so no partial gradient can be pending. Revisit if that changes.
                continue
            flat_part = mb_part_f.reshape(-1)

            # ── VALUE TARGET NORMALISATION ─────────────────────────────────
            # Normalize returns before value regression.  The value head learns
            # to predict normalized targets; advantages are unaffected.
            mb_returns_norm = _normalize_returns(mb_returns, mb_part)
            # Also normalize the stored baseline values so clipping stays valid
            mb_values_norm = (mb_values - _ret_mean) / (_ret_var + 1e-8).sqrt()
            # ──────────────────────────────────────────────────────────────

            profile("train_forward", epoch)
            if not config["use_rnn"]:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            # LSTM-BPTT fix: lstm_h/lstm_c None → zero initial state, which
            # is exact (each stored segment began at evaluate()'s zeroed
            # state — see Dust2Policy.forward doc). terminals drives the
            # mid-segment done-reset inside _lstm_bptt so the training
            # forward replicates the rollout's (1-done)*state masking.
            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
                terminals=mb_terminals,
            )

            # Batch 3 (T5): hybrid PPO update — per-factor clipped loss
            # (Fan et al. IJCAI 2019). The helper does the policy forward
            # pass (returning mu_aim/log_std + value) and assembles the
            # clipped policy loss with INDEPENDENT discrete and continuous
            # ratios. F16 (2026-07-06 adversarial review): it now also
            # returns the logits it computed, killing the redundant no-grad
            # diagnostic forward that used to run here per minibatch
            # (halves update-forward cost). Post-F8 the returned logits are
            # MASKED, so the per-head entropy diagnostics below report the
            # true sampled distribution.
            (pg_loss, entropy, newvalue, newlogprob, ratio_d, ratio_c, logits) = _hybrid_ppo_loss(
                self.policy,
                mb_obs,
                mb_actions,
                mb_cont_actions,
                mb_old_logp_d,
                mb_old_logp_c,
                mb_advantages,
                clip_coef,
                state,
                mb_prio=mb_prio,
                mb_masks=mb_masks,
                mb_part=mb_part_f,
                aim_dim_mask=getattr(self.policy, "aim_dim_mask", None),
                aim_entropy_bonus=bool(config.get("aim_entropy_bonus", True)),
            )
            # NOTE: pre-Batch-3 the inline `actions = ...` from sample_logits
            # was used by downstream diagnostics; T5 dropped that consumer
            # (mb_actions is the canonical stored discrete action). No
            # rebinding here — the variable is unused after this point.
            # (F16: the former no-grad diagnostic re-forward that lived here
            # is gone — `logits` now comes straight from _hybrid_ppo_loss.)

            profile("train_misc", epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            # Batch 3: keep the joint ratio for KL/clipfrac diagnostics so the
            # existing log surface (approx_kl, clipfrac, importance) is
            # backwards-compatible. ratio_d is what gets stored in self.ratio
            # because compute_puff_advantage was tuned for the discrete-head
            # importance ratio in pre-Batch-3 runs; substituting ratio_d here
            # preserves vtrace's behaviour.
            ratio = logratio.exp()
            # Batch 3 (T5): _hybrid_ppo_loss returns flat (B*T,) ratios.
            # self.ratio is (segments, bptt_horizon); reshape ratio_d to
            # match so the indexed-write writes the right shape. ratio
            # (joint) is already reshaped by mb_logprobs.shape on the
            # previous line.
            self.ratio[idx] = ratio_d.detach().reshape(mb_logprobs.shape)

            with torch.no_grad():
                # Rung 0 §2.2: every diagnostic below is a mean over rows, so
                # every one of them is masked. _pm is mb_part_f reshaped to the
                # joint ratio's [S, T] layout; ratio_d/ratio_c are flat, hence
                # flat_part. An unmasked KL here would be diluted 5× at
                # n_active=1 and the target_kl early-stop would never fire.
                _pm = mb_part_f.reshape(logratio.shape)
                old_approx_kl = masked_mean(-logratio, _pm)
                approx_kl = masked_mean((ratio - 1) - logratio, _pm)
                clipfrac = masked_mean(((ratio - 1.0).abs() > config["clip_coef"]).float(), _pm)
                # Observe-only (spec 2026-08-15 §3.4): same formula as the
                # joint `clipfrac` above, split by the per-factor ratios
                # `_hybrid_ppo_loss` already returns. `.item()` into the
                # logging dict only — NEVER add these to the `loss` tensor
                # (they are diagnostics, not a training signal). Last-
                # minibatch-only is forbidden; they accumulate like
                # `clipfrac` and ride the existing `_mb_run` divisor.
                clipfrac_d = masked_mean(((ratio_d - 1.0).abs() > config["clip_coef"]).float(),
                                         flat_part)
                clipfrac_c = masked_mean(((ratio_c - 1.0).abs() > config["clip_coef"]).float(),
                                         flat_part)

            # Early stopping (gh#90): a KL trip finishes the CURRENT epoch
            # (this minibatch included — matches standard PPO's post-epoch
            # check) and stops at the next epoch boundary via the loop-top
            # gate, instead of the old immediate mid-pass break.
            if target_kl is not None and approx_kl.item() > target_kl:
                _kl_stop = True

            # Batch 3 (T5): pg_loss already computed by _hybrid_ppo_loss above
            # via per-factor clipping (the pre-Batch-3 single-ratio block
            # would over-clip — spec L8 decision). Advantage normalization +
            # the mb_prio importance weight now live INSIDE _hybrid_ppo_loss
            # (finding 1, 2026-07-06 adversarial review); the orphaned
            # normalization stub and the discarded per-minibatch
            # compute_puff_advantage recompute that used to sit here were
            # dead compute and have been removed.

            newvalue = newvalue.view(mb_returns_norm.shape)
            v_loss_unclipped = (newvalue - mb_returns_norm)**2
            if vf_clip is not None:
                v_clipped = mb_values_norm + torch.clamp(newvalue - mb_values_norm, -vf_clip,
                                                         vf_clip)
                v_loss_clipped = (v_clipped - mb_returns_norm)**2
                v_loss = 0.5 * masked_mean(torch.max(v_loss_unclipped, v_loss_clipped), mb_part_f)
            else:
                v_loss = 0.5 * masked_mean(v_loss_unclipped, mb_part_f)

            # Rung 0 §2.2: the entropy the SAC-α dual loop and the collapse
            # floor react to must be the participating rows' entropy. Parked
            # rows are noop-masked (exactly one valid bin per head ⇒ discrete
            # entropy 0), so an unmasked mean at n_active=1 reads ~1/5 of the
            # truth and would peg alpha at the floor forever.
            current_entropy = masked_mean(entropy, flat_part)
            entropy_unmasked = entropy.mean()          # diagnostic only (losses/entropy_unmasked)

            # ── ADAPTIVE ALPHA (SAC-style Lagrangian entropy tuning) ───────
            alpha = log_alpha.exp()
            # Task 9A: use the scheduled target_entropy (recomputed at top of
            # this train() call) instead of the static fallback. _t9_target_entropy
            # is a Python float; .detach() on a tensor minus a float is fine —
            # autograd treats the float as a constant.
            # alpha_loss is computed UNCONDITIONALLY (the logging block below
            # accumulates it every minibatch — spec finding 7); during the
            # warm-start GRACE phase only the optimizer step is skipped, so
            # log_alpha stays at its operating point (see phase-resolution
            # comment above for why that matters).
            alpha_loss = (log_alpha * (current_entropy - _t9_target_entropy).detach()).mean()
            if trainer._batch1_warmstart_phase != WS_GRACE:
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

            effective_alpha = alpha.detach()
            if trainer._batch1_warmstart_phase == WS_GRACE:
                # grace: entropy pressure ceilinged (default 0.0 — pure
                # PPO+reward; the knob exists for a nonzero-alpha rerun if
                # the collapse watch fires)
                effective_alpha = torch.clamp(effective_alpha,
                                              max=float(config.get("warmstart_alpha_ceiling", 0.0)))
            # Entropy floor: prevent collapse. Gated off for the ENTIRE
            # warm-start window (grace+ramp): the BC policy lives below the
            # floor by design, and re-arming mid-ramp would jump effective
            # alpha ~1e-3 -> 0.5 in one minibatch (spec finding 2). It re-arms
            # at ramp_end — a plotted boundary.
            if _ws_floor_active and current_entropy.item() < entropy_floor:
                effective_alpha = torch.clamp(effective_alpha, min=0.5)
                _floor_fires += 1

            entropy_loss = -effective_alpha * current_entropy
            # ──────────────────────────────────────────────────────────────

            loss = pg_loss + config["vf_coef"] * v_loss + entropy_loss
            self.amp_context.__enter__()

            # Denormalize before writing back so advantage computation stays in raw scale
            std = (_ret_var + 1e-8).sqrt()
            # Rung 0 §2.2: keep parked rows at exactly 0 in the value buffer —
            # the rollout wrote 0 there and the next epoch's GAE reads it.
            self.values[idx] = (newvalue.detach().float() * std + _ret_mean) * mb_part_f

            # ── PER-HEAD ENTROPY ──────────────────────────────────────────
            with torch.no_grad():
                _dists = [torch.distributions.Categorical(logits=lgt) for lgt in logits]
                # Batch 3: head names sourced from _action_spec.ACTION_HEAD_NAMES
                # (auto-gen from cs2_types.h). Pre-Batch-3 hardcoded "aim" here;
                # now removed since aim is a continuous head emitted on a separate
                # path. zip(strict=True) catches any future drift between
                # _action_spec and the policy logits list.
                _head_names = list(ACTION_HEAD_NAMES)
                for _hi, (_hn, _hd) in enumerate(zip(_head_names, _dists, strict=True)):
                    # flat_part: _hd.entropy() is flat (S*T,), like `entropy`.
                    losses[f"entropy/{_hn}"] += masked_mean(_hd.entropy(), flat_part).item()
            losses["entropy/total"] += current_entropy.item()
            # ──────────────────────────────────────────────────────────────

            # Logging
            profile("train_misc", epoch)
            losses["policy_loss"] += pg_loss.item()
            losses["value_loss"] += v_loss.item()
            losses["entropy"] += current_entropy.item()
            # Rung 0 §2.2: the same mean WITHOUT the mask. Diagnostic only —
            # its ratio to losses/entropy is the live check that the mask is
            # actually doing something (≈ n_active/TEAM_SIZE at steady state).
            losses["entropy_unmasked"] += entropy_unmasked.item()
            losses["alpha"] += alpha.detach().item()
            losses["alpha_loss"] += alpha_loss.item()
            losses["old_approx_kl"] += old_approx_kl.item()
            losses["approx_kl"] += approx_kl.item()
            losses["clipfrac"] += clipfrac.item()
            losses["clipfrac_d"] += clipfrac_d.item()
            losses["clipfrac_c"] += clipfrac_c.item()
            losses["importance"] += masked_mean(ratio, _pm).item()
            # gh#90: count EXECUTED minibatches — the divisor for every
            # accumulated losses/* above and the per-head entropy block.
            # Incremented here (with the stats) so a future early-`continue`
            # placed above the logging block can't desync count from sums.
            _mb_run += 1

            # Learn on accumulated minibatches
            profile("learn", epoch)
            # ── Batch 3 (T5) NaN guard ─────────────────────────────────────
            # The continuous Gaussian aim head can emit non-finite μ / log_std
            # during pathological early training (e.g. an obs that drives the
            # tanh into hard saturation while σ explores LOG_STD_MAX — the
            # log-prob of a far-tail sample under near-zero σ blows up).
            # Skip optimizer.step() with a throttled stdout warning and
            # zero out grads so the next minibatch starts from a clean slate.
            # Do NOT raise — one bad minibatch shouldn't kill a run. `continue`
            # is correct here: the enclosing `for mb in range(...)` is the
            # PPO update loop. There is no nested loop between this check and
            # that for-statement (verified before landing T5).
            if not torch.isfinite(loss).all():
                _now = time.time()
                _last = getattr(self, '_last_nan_warn_t', 0.0)
                if _now - _last > 60.0:
                    print(f"[hybrid_aim NaN guard] non-finite loss "
                          f"({float(loss.detach())}); skipping optimizer step")
                    self._last_nan_warn_t = _now
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # ── TAG diagnostic hook (spec 2026-08-13 §4.2/§4.3) ────────────
            # mb0 = pre-update on-policy regime. mbL = last EXECUTED
            # minibatch: total_minibatches-1 normally, or the final mb of
            # the epoch the KL gate tripped on (the loop-top gate exits at
            # the next epoch boundary) — conditioning mbL on the gate NOT
            # tripping would select against the late-update regime it
            # exists to observe. Placed AFTER the NaN guard (never measure
            # a batch the update skips) and BEFORE loss.backward() (.grad
            # still untouched). Results stash on the TRAINER — see
            # _inject_tag_metrics for the two routing constraints.
            if config.get("tag_diagnostic", False) \
                    and epoch % max(1, int(config.get("tag_every", 5))) == 0:
                _tag_mb0 = (mb == 0)
                _tag_mbL = (mb == self.total_minibatches - 1
                            or (_kl_stop and (mb + 1) % _mbs_per_epoch == 0))
                if _tag_mb0 or _tag_mbL:
                    _tag = tag_grad_cossim(
                        self.policy,
                        mb_obs=mb_obs,
                        mb_actions=mb_actions,
                        mb_cont_actions=mb_cont_actions,
                        mb_old_logp_d=mb_old_logp_d,
                        mb_old_logp_c=mb_old_logp_c,
                        mb_advantages=mb_advantages,
                        clip_coef=clip_coef,
                        state=state,
                        mb_prio=mb_prio,
                        mb_masks=mb_masks,
                        mb_returns_norm=mb_returns_norm,
                        idx=idx,
                        mb_label="mb0" if _tag_mb0 else "mbL",
                        mb_part=mb_part_f,
                        aim_dim_mask=getattr(self.policy, "aim_dim_mask", None),
                        aim_entropy_bonus=bool(config.get("aim_entropy_bonus", True)),
                    )
                    if getattr(trainer, "_tag_metrics", None) is None:
                        trainer._tag_metrics = {}
                    trainer._tag_metrics.update(_tag)
                    if not _tag_mb0:
                        trainer._tag_metrics["tag/mbL_index"] = float(mb)
                    trainer._tag_metrics["tag/selfplay_active"] = float(
                        getattr(self, "_selfplay_used_past", False))
            # ──────────────────────────────────────────────────────────────
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                # Task 9C: capture pre-clip grad norm. clip_grad_norm_ returns
                # the total norm computed BEFORE clipping (PyTorch contract,
                # see torch.nn.utils.clip_grad_norm_ docs). Storing it on the
                # trainer makes it available to the log layer; the .item()
                # call forces a host sync which is fine here because the
                # caller already syncs via .item() on losses below.
                _t9_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                               config["max_grad_norm"])
                trainer._batch1_grad_norm = float(_t9_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # gh#90: normalize the accumulated losses/* sums by the EXECUTED
        # minibatch count. Must run BEFORE the scalar (non-accumulated) keys
        # below (explained_variance, ret_mean, ...) are inserted — dividing
        # those would corrupt them. minibatches_run itself is added after
        # the division for the same reason. max(_mb_run, 1) is pure belt-and-
        # braces: the epoch-boundary gate guarantees epoch 0 completes.
        for _lk in list(losses):
            losses[_lk] /= max(_mb_run, 1)
        losses["minibatches_run"] = _mb_run

        # Rung 0 §2.2 counters/absolutes — inserted AFTER the divisor (gh#90
        # trap: anything written before it is silently scaled by 1/_mb_run).
        losses["entropy_floor_fires"] = float(_floor_fires)
        losses["empty_minibatches"] = float(_empty_mb)
        losses["participating_rows"] = float(self.participating.sum().item())

        # Warm-start metrics are ABSOLUTE values — inserted after the gh#90
        # divisor loop above, alongside minibatches_run, or they'd be divided
        # by the executed-minibatch count (the exact bug class gh#90 fixed).
        if config.get("warmstart_entropy", False):
            losses["warmstart_phase"] = trainer._batch1_warmstart_phase
            # h0 is captured on the mode's first update, when entropy is
            # healthy (BC policy ~1.8 nats) — the >1e-9 guard exists because
            # total entropy (discrete + Gaussian differential) CAN go
            # non-positive in the collapse regime, and a non-positive
            # denominator would flip the watch's sign. If capture is ever
            # skipped, say so once instead of silently disabling the watch.
            if trainer._batch1_warmstart_h0 is None:
                if losses["entropy"] > 1e-9:
                    trainer._batch1_warmstart_h0 = float(losses["entropy"])
                else:
                    print(f"[Train] WARN warm-start: first-update entropy "
                          f"{losses['entropy']:.3f} <= 0 — h_over_h0 collapse "
                          f"watch cannot arm (will retry next update).")
            if trainer._batch1_warmstart_h0:
                losses["warmstart_h_over_h0"] = losses["entropy"] / trainer._batch1_warmstart_h0
                # collapse watch (spec finding 6): grace disables BOTH
                # anti-collapse guards (floor clamp + alpha), so shout —
                # throttled to every 20 epochs — if H halves.
                if (trainer._batch1_warmstart_phase == WS_GRACE
                        and losses["warmstart_h_over_h0"] < 0.5
                        and self.epoch - trainer._batch1_warmstart_warn_epoch >= 20):
                    trainer._batch1_warmstart_warn_epoch = self.epoch
                    print(f"[Train] WARN warm-start grace: entropy at "
                          f"{losses['warmstart_h_over_h0']:.2f} of its start value "
                          f"({losses['entropy']:.3f} nats) with alpha ceilinged and the "
                          f"entropy floor disabled — collapse watch (spec finding 6).")
        # Anchor source: maintained EVERY update, unconditionally (mode may be
        # enabled on a later resume of this process in tests; cost is one float).
        trainer._batch1_last_entropy_mean = float(losses["entropy"])

        # Reprioritize experience
        profile("train_misc", epoch)
        if config["anneal_lr"]:
            self.scheduler.step()

        # Rung 0 §2.2: whole-buffer EV over PARTICIPATING rows only. Parked
        # rows have value 0 and advantage 0, i.e. a perfectly-predicted
        # constant — leaving them in would inflate EV toward 1 by exactly the
        # parked fraction and make the critic look healthy at n_active=1
        # regardless of what it learned.
        losses["explained_variance"] = masked_explained_variance(
            self.values.flatten(),
            advantages.flatten() + self.values.flatten(), self.participating.flatten())
        losses["ret_mean"] = _ret_mean.item()
        losses["ret_std"] = (_ret_var + 1e-8).sqrt().item()
        # Observe-only persist (spec 2026-08-15 §3.4). Task 8 already
        # computes `_batch1_event_oversample_fraction` once per train()
        # call; older comments that say the wandb/log layer already
        # reports it were stale. MUST sit after the gh#90 divisor loop —
        # this is a per-call scalar like ret_mean, not a minibatch sum.
        # Expected ~0.0 while #100 keeps include_step_stats_in_info=False
        # (no event mask); that zero is the production signal. The
        # KL-break harness turns the flag on, so its test must not
        # assert == 0.0.
        losses["event_oversample_fraction"] = float(
            getattr(self, "_batch1_event_oversample_fraction", 0.0))
        losses["log_alpha"] = log_alpha.item()

        # Task 9C: expose per-train()-call metrics on the trainer for the
        # wandb log layer. Captured here (not inside the minibatch loop)
        # because the log layer reports one value per train() call, not
        # per-minibatch — and effective_alpha / log_alpha are last-write-
        # wins after the inner loop anyway.
        # PITFALL: effective_alpha is bound inside the minibatch loop;
        # Python keeps the last bound value visible at this scope so
        # reading it here works in the happy path. Since gh#90 the
        # target_kl early-stop can only exit at an epoch boundary (epoch 0
        # always completes), so the old "break on mb=0 leaves
        # effective_alpha unbound" NameError is structurally impossible —
        # the try/except below stays as defense-in-depth only. The NaN
        # guard's `continue` can still skip the optimizer-step site, so
        # _batch1_grad_norm keeps its pre-seeded default in that edge.
        trainer._batch1_log_alpha = float(log_alpha.item())
        # effective_alpha may be unbound this call if target_kl early-broke
        # on mb=0 — leave the pre-initialised trainer attr (set in the patch
        # block above) intact in that case rather than crashing.
        try:
            trainer._batch1_effective_alpha = float(effective_alpha.detach().item())
        except (NameError, UnboundLocalError):
            pass
        # Rung 0 §2.2: log the alpha the loss ACTUALLY used (post ceiling /
        # post floor clamp), not just the raw log_alpha.exp() above — with the
        # floor now counted by entropy_floor_fires, the pair says whether the
        # collapse guard is holding the run up. Absolute value, so it sits here
        # after the gh#90 divisor, and it reads the trainer attr rather than
        # the loop-local so the KL-early-break edge cannot NameError.
        losses["effective_alpha"] = float(trainer._batch1_effective_alpha)
        # Welford std exposure: guard with getattr+fallback because
        # _patch_trainer_with_selfplay (Task 6c, where these get attached)
        # may not have been applied — preserves the no-selfplay code path.
        _w_combat = getattr(trainer, "_batch1_welford_combat", None)
        trainer._batch1_std_combat = (float(_w_combat.std()) if _w_combat is not None else 1.0)
        _w_obj = getattr(trainer, "_batch1_welford_objective", None)
        trainer._batch1_std_objective = (float(_w_obj.std()) if _w_obj is not None else 1.0)
        _w_pos = getattr(trainer, "_batch1_welford_positional", None)
        trainer._batch1_std_positional = (float(_w_pos.std()) if _w_pos is not None else 1.0)

        profile.end()
        logs = None
        self.epoch += 1
        # Rung 0 §2.2: global_step is in PARTICIPATING units, so it must be
        # compared against participating_timesteps (= the user's --timesteps),
        # not against total_timesteps (the TEAM_SIZE/n_active-scaled raw-row
        # budget PufferLib's epoch cap needs). .get() keeps trainers built from
        # a pre-Rung-0 config.json working. The epoch clause is load-bearing,
        # not belt-and-braces: total_epochs floor-divides, so at the defaults
        # (--timesteps 10M, --num_envs 256 ⇒ batch 163840, n_active=1 ⇒ raw
        # budget 50M) the 305 epochs PufferLib allows collect only
        # 305 × 163840 / 5 = 9,994,240 participating steps — the first clause
        # would never fire and the last epoch would never checkpoint.
        done_training = (self.global_step >= config.get("participating_timesteps",
                                                        config["total_timesteps"])
                         or self.epoch >= self.total_epochs)
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            # R0-A: episode count for the window, written INTO self.stats so
            # mean_and_log (pufferl.py mean_and_log) emits environment/episodes
            # through the logger too. DECLARED DEVIATION from spec §3 R0-A
            # ("inject after mean_and_log returns"): the logger call is inside
            # mean_and_log, so the post-hoc form would reach metrics.jsonl
            # only. `.get` — a defaultdict(list) `[]` read would insert an
            # empty list and np.mean([]) → NaN. Unit: terminal infos (rounds),
            # one per env per round regardless of n_active_per_team.
            self.stats["episodes"] = [float(len(self.stats.get("kills_t", ())))]
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config["checkpoint_interval"] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f"Checkpoint saved at update {self.epoch}"

        return logs

    # Bind the patched method to the specific trainer instance
    trainer.train = types.MethodType(_train_with_return_norm, trainer)
    print("[Train] Value target normalization enabled (running mean/std of returns).")
    return trainer


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


class ScheduledEval:
    """Runs the fixed-baseline eval every `interval` epochs and delivers its
    keys on the next LOGGED metrics row.

    WHY a buffer: PuffeRL's train() throttles mean_and_log to once per 0.25 s
    (logs is None on the other epochs). A naive `if isinstance(logs, dict)
    and epoch % interval == 0` silently skips the eval whenever the eval epoch
    happens to be throttled — on a fast CPU box that is most epochs. So the
    eval decision is made on `trainer.epoch` alone, and the result waits in
    `pending` until a dict row comes through. `eval/epoch` stamps the epoch
    the numbers were measured at (may lag the row's epoch by a few).

    PITFALL: call after_train() on EVERY epoch, outside any `isinstance(logs,
    dict)` guard, or the buffer never drains.
    """

    def __init__(self, evaluator, interval, policy, device):
        if int(interval) <= 0:
            raise ValueError(f"ScheduledEval interval must be >= 1, got {interval}")
        self.evaluator = evaluator
        self.interval = int(interval)
        self.policy = policy
        self.device = device
        self.pending = {}

    def after_train(self, trainer, logs):
        if trainer.epoch % self.interval == 0:
            t0 = time.time()
            self.pending.update(self.evaluator.evaluate(self.policy, self.device))
            self.pending["eval/epoch"] = int(trainer.epoch)
            self.pending["eval/wall_s"] = time.time() - t0
        if isinstance(logs, dict) and self.pending:
            logs.update(self.pending)
            self.pending = {}

    def close(self):
        self.evaluator.env.close()


# ── SECTION: Game Metrics Dashboard ───────────────────────────────────────


def compute_game_metrics(logs):
    """Extract and normalize game metrics from the training logs dict.

    The C env exposes per-episode stats as ``environment/<key>`` entries in
    the logs dict returned by PufferLib's ``mean_and_log()``.  Values are
    already averaged over the collection window, so most just need re-keying
    and minor arithmetic.

    Always-present ``game/*`` keys (already in today's terminal info) are
    re-keyed with ``_get(..., default=0.0)``. ``game/plant_tick`` and
    ``game/win_by_*`` are presence-gated: emit them only when the source
    key already exists in ``logs``. A synthetic 0.0 would make old log
    dicts look new-format.

    Returns a flat dict with ``game/*`` and ``actions/*`` keys ready to be
    merged back into logs for W&B or stdout.
    """
    if not isinstance(logs, dict):
        return {}

    def _get(key, default=0.0):
        return logs.get(f"environment/{key}", logs.get(key, default))

    winner_t = _get("winner_t", 0.0)
    winner_ct = _get("winner_ct", 0.0)
    timed_out = _get("timed_out", 0.0)
    kills_t = _get("kills_t", 0.0)
    kills_ct = _get("kills_ct", 0.0)
    bomb_planted = _get("bomb_planted", 0.0)
    round_length = _get("round_length", 0.0)

    # win rates: already normalised per-episode by PufferLib's mean_and_log
    game_metrics = {
        "game/win_rate_t": winner_t,
        "game/win_rate_ct": winner_ct,
        "game/timeout_rate": timed_out,
        "game/kills_per_episode": kills_t + kills_ct,
        "game/bomb_plant_rate": bomb_planted,
        "game/avg_episode_length": round_length,
    }

    # Always-present splits/rewards: these keys already land in logs today
    # via _build_terminal_info. game/reward/win is NOT emitted (#128, R0-A):
    # C reward_win is the cross-team sum and nets ~0 by the zero-sum
    # identity; the one-sided game/reward/win_t|win_ct below carry the signal.
    game_metrics["game/defuse_rate"] = _get("bomb_defused", 0.0)
    game_metrics["game/kills_t"] = kills_t
    game_metrics["game/kills_ct"] = kills_ct
    for src, dst in (
        ("reward_kills", "game/reward/kills"),
        ("reward_deaths", "game/reward/deaths"),
        ("reward_bomb", "game/reward/bomb"),
        ("reward_pbrs", "game/reward/pbrs"),
        ("reward_shots", "game/reward/shots"),
        ("reward_survival", "game/reward/survival"),
        ("reward_inaction", "game/reward/inaction"),
    ):
        game_metrics[dst] = _get(src, 0.0)

    # R0-A: combat counters are window MEANS per episode (mean_and_log).
    for k in ("shots_fired", "shots_with_enemy_in_los", "shots_facing_enemy", "shots_on_target",
              "shots_hit", "shots_stance_blocked", "damage_dealt", "mutual_vis_pair_ticks",
              "agent_ticks_with_visible_enemy"):
        game_metrics[f"game/{k}"] = _get(k, 0.0)
    game_metrics["game/reward/win_t"] = _get("reward_win_t", 0.0)
    game_metrics["game/reward/win_ct"] = _get("reward_win_ct", 0.0)
    # Ratio of window means = conditional mean over episodes where a pair
    # coexisted. A max(·,1) guard would silently return the unconditional
    # mean — keep the explicit zero-valid branch.
    _med_sum = _get("min_enemy_distance_sum", 0.0)
    _med_valid = _get("min_enemy_distance_valid", 0.0)
    game_metrics["game/min_enemy_distance"] = (_med_sum / _med_valid) if _med_valid > 0 else 0.0
    game_metrics["game/min_enemy_distance_valid_frac"] = _med_valid

    # Presence-gate plant_tick / win_by_*: a synthetic 0.0 would make old
    # log dicts look new-format. Do not _get(..., default=0.0) these three.
    def _maybe(src, dst):
        if f"environment/{src}" in logs or src in logs:
            game_metrics[dst] = _get(src)

    _maybe("plant_tick", "game/plant_tick")
    _maybe("win_by_detonation", "game/win_by_detonation")
    _maybe("win_by_defuse", "game/win_by_defuse")

    # actions/use_at_site_frac — logged directly by the C env if available
    use_at_site = _get("use_at_site_frac", None)
    if use_at_site is not None:
        game_metrics["actions/use_at_site_frac"] = use_at_site

    return game_metrics


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


def pin_pitch_for_map(map_data) -> int:
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
    """
    md = map_data
    if md is None:
        # Same cache key make_env uses, so train() never loads the nav twice.
        import nav
        from c_env.cs2_env import _ENV_CACHE
        from map import make_cs2_map
        key = (nav.NAV_PATH, nav.CACHE_PATH)
        md = _ENV_CACHE.get(key)
        if md is None:
            md = make_cs2_map(nav.NAV_PATH, nav.CACHE_PATH)
            _ENV_CACHE[key] = md
    z = np.asarray(md.centroids_z, dtype=np.float32)
    return int(float(z.max() - z.min()) == 0.0)


def resolve_pin_pitch(args, verbose: bool = True) -> int:
    """R0-E.2 (#131): set/validate args.pin_pitch from args.map_data; returns it.

    WHAT: ``args.pin_pitch is None`` (CLI default) ⇒ pin_pitch_for_map(
    args.map_data). An explicit 0/1 is cross-checked against the same test
    and refused with ValueError (never assert) when it disagrees with the map.

    WHY a separate function: train() is too heavy to exercise in a unit test,
    and this block MUST run before build_train_env_factory — env_knobs_from_
    args(args) bakes args.pin_pitch into every worker env at vector.make;
    resolving later would leave the envs unpinned while the policy gets
    aim_dim_mask=[1,0] and assert_pin_pitch_agreement aborts the run.

    PITFALL: args.map_data is None for `--map dust2`/`--dust2`; the helper
    LOADS the map (cached). main() calls this ABOVE the --dump-config exit on
    purpose — the Modal fingerprint dump must carry the geometry-resolved value
    (costs ~1 s for dust2 from the nav cache, ~0.8 s for `import map`). train()
    calls it again as a cache-safe cross-check for programmatic callers (second
    call is silent, see `verbose`). verbose=False for the train() cross-check
    so the value is printed once per launch.
    """
    flat = bool(pin_pitch_for_map(getattr(args, "map_data", None)))
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

    WHY: the two sides are set independently (env_knobs_from_args bakes the
    flag into every worker at vector.make time; build_policy sets the mask
    from args.pin_pitch) and a mismatch is silent — the env would ignore a
    dim the trainer still scores, or score a dim the env still applies.

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
                # only on the transition tick (cs2_bomb.h:60 — guarded by
                # `if g->bomb_plant_ticks >= sd->bomb_plant_time`) and StepStats
                # is cleared every step via clear_stats(ss) at the top of
                # cs2_env.h:75, so the field is already a per-tick delta (1 only
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
                # stationary, cs2_movement.h:214) and is never masked out by
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
                # _patch_trainer_with_hybrid_aim. The PPO update at line ~1085
                # reads these by the same idx; missing this write would
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
#                            to the existing actions / logprobs, and patches
#                            vecenv.send to forward the float buffer to the
#                            env. Applied AFTER _patch_trainer_with_return_norm
#                            (which wraps train()) and BEFORE
#                            _patch_trainer_with_selfplay (which wraps
#                            evaluate()). Order matters: train() reads the
#                            cont buffer that this patcher allocates.


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
            # The C env (cs2_env.h:129) clamps silently with fminf/fmaxf;
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


def _aim_dim_weight(aim_dim_mask, mu_aim):
    """(AIM_DIM,) weight for the per-dim Gaussian terms (R0-E.2, #131).

    WHAT: ``aim_dim_mask`` moved to mu_aim's device/dtype, or all-ones when
    None. Shared by _hybrid_sample_logits and _hybrid_ppo_loss so the rollout
    and the update can never disagree on which dims are live — that
    disagreement would be an importance-ratio bug no single-site test sees.
    PITFALL: returns ones (not None) on the None path so callers can multiply
    unconditionally; the multiply by ones is exact in fp32.
    """
    import torch

    if aim_dim_mask is None:
        return torch.ones(mu_aim.shape[-1], device=mu_aim.device, dtype=mu_aim.dtype)
    return aim_dim_mask.to(device=mu_aim.device, dtype=mu_aim.dtype)


def _hybrid_ppo_loss(policy,
                     mb_obs,
                     mb_actions,
                     mb_cont_actions,
                     mb_old_logp_d,
                     mb_old_logp_c,
                     mb_advantages,
                     clip_coef,
                     state,
                     mb_prio=None,
                     mb_masks=None,
                     return_pg_rows=False,
                     mb_part=None,
                     aim_dim_mask=None,
                     aim_entropy_bonus=True):
    """Per-factor PPO clipped loss (H-PPO, Fan et al. IJCAI 2019).

    aim_dim_mask (R0-E.2): same per-dim weight the rollout sampler used
    (policy.aim_dim_mask) — MUST match, or ratio_c ≠ 1 for an unchanged
    policy. None ⇒ all-ones (pre-R0-E behaviour).
    aim_entropy_bonus (R0-E.4, #131): False ⇒ the returned ``entropy`` is the
    DISCRETE entropy only, so the entropy objective (and its α dual loop)
    stops pushing aim σ to the cap. The Gaussian log-prob still enters
    ratio_c either way — only the bonus is switched off.

    THE CORE OF T5. Re-runs the policy on mb_obs with the stored
    mb_actions / mb_cont_actions, computes new log-probs split into
    discrete and continuous halves, and applies the PPO clip
    INDEPENDENTLY to each half before summing. This is the spec L8
    decision: discrete head saturating doesn't have to drag the
    continuous head's gradient through clipping (and vice versa).

    Returns
    -------
    pg_loss : scalar — sum of clipped discrete + clipped continuous losses.
    entropy : (B,) — joint factorised entropy (same shape as old logprobs).
    new_value : (B, 1) — fresh value estimate for the value-loss path.
    new_logp_total : (B,) — sum of new discrete + new continuous log-probs;
        used for KL/clipfrac diagnostics in the caller.
    ratio_d, ratio_c : (B,) — exposed so the caller can attribute clipfrac
        per factor and substitute ratio_d into the existing self.ratio
        slot for vtrace advantages (spec carry-forward: keep diagnostics
        backwards-compatible by using the discrete ratio there).
    logits_list : list of 7 (B, head_size) tensors — the per-head logits
        this loss was computed from (F16: post-mask when mb_masks is given,
        so per-head entropy diagnostics reflect the TRUE sampled
        distribution). Returned so the caller doesn't need a second full
        forward pass for diagnostics — that redundant no-grad forward used
        to double the update-forward cost. Autograd-attached; consume under
        torch.no_grad() and don't hold past backward() if memory matters.

    LSTM-BPTT fix: `state` must carry `terminals` (the minibatch's
    (segments, bptt_horizon) done flags) so the policy forward runs
    done-masked BPTT over the time dimension — without it the recomputed
    logprobs at post-done ticks silently diverge from the rollout-stored
    ones (state["lstm_h"]/["lstm_c"]=None means zero init, which is exact;
    see Dust2Policy.forward). Test-path callers passing flat 2D tensors may
    omit terminals: T=1 has no through-time state to reset.

    mb_masks (F8): the rollout-stored action masks for this minibatch,
    (segments, bptt_horizon, ACTION_MASK_DIM) bool (or flat (B, MASK_DIM) on
    the test path). MUST be the same masks the rollout sampler used —
    masking here but not there (or vice versa) silently skews the PPO
    ratios for any agent-step where a mask bit was 0. None = unmasked
    (pre-F8 callers / BC paths).

    return_pg_rows (TAG diagnostic, spec 2026-08-13 §4.3): when True the
    return tuple gains an 8th element — the per-row pg loss vector
    max(pg_d_un, pg_d_cl) + max(pg_c_un, pg_c_cl), flat (B*T,), graph-
    attached, advantage-normalized + prio-weighted exactly like pg_loss
    (whose value is the mean of the two factor vectors separately; the sum
    vector's .mean() equals it up to fp reduction order). TAG forms subset
    losses as weighted means over this vector so every subset gradient is a
    true restriction of the real gradient from ONE forward pass. False (the
    default, all production update paths) returns the existing 7-tuple
    bitwise-identically — pinned by
    test_return_pg_rows_default_is_bitwise_identical_7_tuple.

    mb_part (Rung 0 §2.2): FLOAT participation weights, broadcastable to
    mb_advantages (the trainer passes [S, T]). When given, BOTH reductions in
    this function switch to their masked forms — advantage normalisation over
    participating rows only, and a masked mean over the per-row pg terms.
    Doing only one of the two would be wrong in a way no test at
    n_active=TEAM_SIZE can see: unmasked normalisation shifts the parked rows'
    advantage off 0, and that offset survives into pg through their (arbitrary)
    ratio. None ⇒ the pre-Rung-0 lines run verbatim; that path is what
    test_return_pg_rows_default_is_bitwise_identical_7_tuple pins, so
    production at n_active=5 takes the MASKED branch with an all-ones weight
    and is identical to the old numbers only to fp tolerance, not bitwise.
    """
    import torch
    import torch.nn.functional as F

    logits_list, mu_aim, log_std_aim, new_value = policy(mb_obs, state)

    # ── Shape harmonisation ──
    # PufferLib's PPO update path passes mb_obs with shape (segments,
    # bptt_horizon, OBS_DIM); HybridPolicy.forward flattens to (B*T, ...)
    # before the heads, so logits/mu_aim/log_std/new_value come back at the
    # FLAT batch dim while mb_actions / mb_cont_actions / mb_advantages /
    # mb_old_logp_{d,c} retain their original (segments, bptt_horizon, …)
    # shape. Flatten the latter to match logits' batch dim. If mb_actions
    # is already 2D (test path passing flat tensors directly), .view keeps
    # it 2D — a no-op.
    flat_actions = mb_actions.reshape(-1, mb_actions.shape[-1])
    flat_cont = mb_cont_actions.reshape(-1, mb_cont_actions.shape[-1])
    flat_old_d = mb_old_logp_d.reshape(-1)
    flat_old_c = mb_old_logp_c.reshape(-1)
    flat_adv = mb_advantages.reshape(-1)

    # ── Advantage normalization + prio-IS weight (finding 1, 2026-07-06
    # adversarial review) ──
    # Stock PufferLib normalizes per-minibatch and applies the prioritized-
    # replay importance weight BEFORE the pg term:
    #     adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)
    # The T5 refactor moved the pg term into this function but fed it RAW
    # advantages, orphaning the normalization at the call site. Consequence
    # (verified at 98e3f32): with sparse rewards the pg gradient scale was
    # ~0, so the entropy objective faced no counter-pressure and the 30M
    # run drifted to an exactly-uniform discrete policy. Normalizing HERE
    # (not at the call site) makes the contract self-contained and lets the
    # test assert scale-invariance of pg_loss directly.
    # PITFALLS:
    #   * Normalize the RAW adv first, then multiply by mb_prio — reversing
    #     the order changes the statistics (matches stock).
    #   * mb_prio arrives as (segments, 1) from the trainer (broadcast over
    #     bptt_horizon) or (B,) from tests; expand_as handles both. None ⇒
    #     uniform replay (weight 1), e.g. BC/eval callers.
    #   * A constant-adv minibatch has std 0 ⇒ normalized adv is exactly 0
    #     (0/1e-8); pg_loss 0, no NaN.
    #   * mb_part (Rung 0 §2.2): stats over participating rows only, and
    #     parked rows are zeroed so they contribute nothing to pg regardless
    #     of their ratio — which is what the masked pg mean below then
    #     divides out. The mb_prio multiply stays AFTER, unchanged.
    if mb_part is None:
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
    else:
        flat_adv = masked_normalize_adv(flat_adv, mb_part.reshape(-1))
    if mb_prio is not None:
        flat_adv = mb_prio.expand_as(mb_advantages).reshape(-1) * flat_adv

    # ── Re-evaluate discrete and continuous halves under the new policy ──
    # Fix #3 (perf): replaces 7× torch.distributions.Categorical(logits=lg) +
    # 1× torch.distributions.Normal(mu, sigma) construction per minibatch
    # with the same hand-rolled forms used in _hybrid_sample_logits (Fix #2).
    # Microbench measured 8.6 ms/MB savings; PPO update calls this 10×4 = 40
    # times per epoch → ~345 ms/epoch saved on heavy-update epochs. Same
    # numerical contract as before: |Δ| ≤ ~2e-6 vs torch.distributions
    # reference (different softmax reduction order; well within fp32 noise).
    # F8: apply the rollout's action masks to the fresh logits so new_logp /
    # entropy are computed over the SAME masked distribution the sampler drew
    # from — otherwise ratios drift wherever a mask bit was 0.
    if mb_masks is not None:
        flat_masks = mb_masks.reshape(-1, mb_masks.shape[-1])
        logits_list = _apply_action_masks(logits_list, flat_masks)
    log_probs_per_head = [F.log_softmax(lg, dim=-1) for lg in logits_list]
    new_logp_d = sum(
        lp.gather(-1, flat_actions[..., i:i + 1]).squeeze(-1)
        for i, lp in enumerate(log_probs_per_head))
    entropy_d = sum(-(lp.exp() * lp).sum(-1) for lp in log_probs_per_head)

    sigma = torch.exp(log_std_aim).expand_as(mu_aim)
    log_std_b = log_std_aim.expand_as(mu_aim)
    diff = (flat_cont - mu_aim) / sigma
    w_aim = _aim_dim_weight(aim_dim_mask, mu_aim)
    new_logp_c = ((-0.5 * diff * diff - log_std_b - 0.5 * _LOG_2PI) * w_aim).sum(-1)
    entropy_c = ((0.5 + 0.5 * _LOG_2PI + log_std_b) * w_aim).sum(-1)
    entropy = entropy_d + entropy_c if aim_entropy_bonus else entropy_d

    # ── Per-factor PPO ratios + clipped loss ──
    # max(unclipped, clipped) is taken element-wise per factor; the per-
    # element scalars are then meaned. Summing the two means matches the
    # H-PPO Eq. 8 in Fan et al. — equal weighting of the two heads. If a
    # future variant wants weighted heads (e.g. up-weight continuous early
    # in training), introduce per-factor coefficients HERE, not by scaling
    # ratios.
    ratio_d = torch.exp(new_logp_d - flat_old_d)
    ratio_c = torch.exp(new_logp_c - flat_old_c)
    pg_d_un = -flat_adv * ratio_d
    pg_d_cl = -flat_adv * torch.clamp(ratio_d, 1 - clip_coef, 1 + clip_coef)
    pg_c_un = -flat_adv * ratio_c
    pg_c_cl = -flat_adv * torch.clamp(ratio_c, 1 - clip_coef, 1 + clip_coef)
    pg_d_rows = torch.max(pg_d_un, pg_d_cl)
    pg_c_rows = torch.max(pg_c_un, pg_c_cl)
    if mb_part is None:
        pg_loss = pg_d_rows.mean() + pg_c_rows.mean()
    else:
        # Rung 0 §2.2: parked rows are already exactly 0 in these vectors
        # (their normalised advantage is 0), so the mask only fixes the
        # DENOMINATOR — without it the gradient is scaled by n_active/5.
        _w = mb_part.reshape(-1).to(pg_d_rows.dtype)
        pg_loss = masked_mean(pg_d_rows, _w) + masked_mean(pg_c_rows, _w)

    if return_pg_rows:
        return (pg_loss, entropy, new_value, new_logp_d + new_logp_c, ratio_d, ratio_c, logits_list,
                pg_d_rows + pg_c_rows)
    return pg_loss, entropy, new_value, new_logp_d + new_logp_c, ratio_d, ratio_c, logits_list


def _tag_param_groups(policy):
    """Partition policy params into the TAG groups (spec 2026-08-13 §4.2).

    trunk        — encoder.* + lstm.* (620,544 of 626,971 trainable params,
                   99.0%, LSTM alone 526,336; this is why no 'total' group
                   exists — it would replicate trunk while reading as
                   independent signal). Trunk-split (spec 2026-08-15 §3.4):
                   encoder_t./encoder_ct./lstm_t./lstm_ct. map into this
                   SAME group, doubling it. The union is what makes the
                   T-vs-CT trunk cross cos-sim exactly 0.0 — each team's
                   gradient is zero on the other's copy — which the
                   analyzer labels structural via split/trunk_active.
    policy_heads — action_heads.* + aim_mu.* + the aim_log_std parameter
                   (6,170 params). Batch 7: under --tct-split-heads BOTH team
                   copies (action_heads_t/_ct, aim_mu_t/_ct, aim_log_std_t/_ct)
                   map to this ONE group, doubling it to 12,340. The union is
                   deliberate — it is what makes the T-vs-CT cross cos-sim
                   exactly 0.0 (each team's gradient is zero on the other's
                   copy), which the analyzer labels as a structural artifact
                   rather than a conflict (spec §3.4). Splitting the group per
                   team instead would produce a within-copy number that
                   answers a different question than the trunk cells.
    value_head   — value_head.* (257 params; used ONLY for the vf control —
                   pg metrics skip it, the pg graph never touches it).

    Uses named_parameters() filtered to requires_grad — NOT state_dict(),
    which would sweep in non-trainable buffers (e.g. max_turn_speed).
    PITFALL: an unmapped parameter RAISES. Silent fallthrough would let a
    renamed module drop out of every group and the metric would quietly
    measure a subset of the network.
    """
    groups = {"trunk": [], "policy_heads": [], "value_head": []}
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(
            ("encoder.", "encoder_t.", "encoder_ct.", "lstm.", "lstm_t.", "lstm_ct.")):
            groups["trunk"].append(p)
        elif name.startswith(("action_heads.", "action_heads_t.", "action_heads_ct.",
                              "aim_mu.", "aim_mu_t.", "aim_mu_ct.")) \
                or name in ("aim_log_std", "aim_log_std_t", "aim_log_std_ct"):
            groups["policy_heads"].append(p)
        elif name.startswith("value_head."):
            groups["value_head"].append(p)
        else:
            raise AssertionError(
                f"TAG: unmapped policy parameter {name!r} — update _tag_param_groups")
    return groups


def tag_grad_cossim(policy,
                    *,
                    mb_obs,
                    mb_actions,
                    mb_cont_actions,
                    mb_old_logp_d,
                    mb_old_logp_c,
                    mb_advantages,
                    clip_coef,
                    state,
                    mb_prio,
                    mb_masks,
                    mb_returns_norm,
                    idx,
                    mb_label,
                    mb_part=None,
                    aim_dim_mask=None,
                    aim_entropy_bonus=True):
    """T-vs-CT gradient cosine-similarity measurement (spec 2026-08-13 §4.2).

    aim_dim_mask / aim_entropy_bonus (R0-E): forwarded verbatim to the inner
    _hybrid_ppo_loss so its ratio_c matches the real update's — otherwise the
    TAG subset gradients would include a pinned pitch factor and stop being
    restrictions of the actual gradient.

    WHAT: ONE extra forward via _hybrid_ppo_loss(return_pg_rows=True), then
    six subset losses as weighted means over the per-row pg vector — T, CT,
    and the env-parity halves T_a/T_b, CT_a/CT_b — each differentiated with
    torch.autograd.grad against the SAME graph (retain_graph=True because
    successive grad calls need it; the real update's graph is a separate,
    untouched object). Advantage normalization is shared over the full
    minibatch (it lives inside _hybrid_ppo_loss, before the per-row max),
    so every subset gradient is a true restriction of the real gradient.
    vf-only T/CT losses on the same forward's new_value feed the
    known-anticorrelated tag/cossim_vf control over value_head params.

    Cost per measured minibatch: 1 forward + 8 backwards (spec §4.3 budget:
    ≤5% of epoch time at --tag-every 5; raise tag_every if exceeded, don't
    optimize).

    WHY cross_half exists: the criterion statistic is cos(g_Ta, g_CTa) —
    size-matched to the within-team null (all arms at n/2 rows). The
    full-size cos(g_T, g_CT) is reported as the lower-noise descriptive
    number but has a LARGER expected same-distribution cosine than any n/2
    statistic, which would bias within − cross toward "no conflict"
    (plan-review finding 2).

    WHY the entropy term is absent: the pg vector contains no entropy —
    deliberate (spec §4.2): entropy is team-agnostic (pushes cos-sim toward
    +1 mechanically) and its effective_alpha is warmstart-phase-dependent.

    ROW IDENTITY: segment index ≡ global agent index (env-major, 10/env, T
    at slots 0-4 — the trainer asserts segments == total_agents, gh#85), so
    team T rows are (idx % 10) < 5 and env parity is (idx // 10) % 2, where
    idx is the minibatch's multinomial segment gather.

    PITFALLS:
    * Never touches .grad, self.ratio, KL bookkeeping, or the Welford
      return-norm state — mb_returns_norm arrives already normalized.
      Training with the flag on is bitwise-identical (pinned by
      tests/test_tag_trainer.py).
    * Zero-norm subsets (subset advantage exactly 0 after shared
      normalization) yield a DELIBERATE NaN cos-sim (0/0) and gnorm 0 —
      analysis drops them; do not "fix" with an epsilon. These NaNs are
      also why the outer-loop injection must stay after
      dead_run_detector.check (see _inject_tag_metrics).
    * pg metrics cover trunk + policy_heads only — the pg graph never
      touches value_head, so those keys would be dead NaN/0 noise.
    * The loss path evaluates stored actions and samples nothing, so there
      is no RNG interaction.
    * mb_part (Rung 0 §2.2) is forwarded verbatim to _hybrid_ppo_loss so the
      shared advantage normalisation is the SAME one the real update used —
      that shared normalisation is what makes each subset gradient a true
      restriction of the real gradient. Parked rows land in whichever team
      subset their slot belongs to, but their pg_rows entries are exactly 0,
      so they only inflate the subset means' denominators (w.sum() here
      counts rows, not participation) — a uniform rescale that cosine
      similarity is invariant to. The reported gnorms ARE scaled by it.
      The vf control is weaker: mb_returns_norm on a parked row is
      -_ret_mean/std, not 0, so parked rows add a common-mode residual to
      BOTH team value gradients and bias tag/cossim_vf upward at
      n_active < TEAM_SIZE. Read that control with n_active in mind.
    """
    import torch

    groups = _tag_param_groups(policy)
    pg_group_names = ("trunk", "policy_heads")
    pg_params = [p for g in pg_group_names for p in groups[g]]
    sizes = [len(groups[g]) for g in pg_group_names]
    bounds = [sum(sizes[:i]) for i in range(len(sizes) + 1)]

    team_t = (idx % 10) < 5
    env_even = ((idx // 10) % 2) == 0                  # env-parity split (exchangeable)
    subsets = {
        "T": team_t,
        "CT": ~team_t,
        "T_a": team_t & env_even,
        "T_b": team_t & ~env_even,
        "CT_a": (~team_t) & env_even,
        "CT_b": (~team_t) & ~env_even,
    }

    def _row_weights(mask):
        # segment mask → flat per-row weights, matching pg_rows' layout
        # ((segments, bptt).reshape(-1) segment-major; flat test path is 1:1)
        w = mask.to(mb_advantages.dtype)
        if mb_advantages.dim() > 1:
            w = w.unsqueeze(1)
        return w.expand_as(mb_advantages).reshape(-1)

    def _flat(grads):
        return torch.cat([g.reshape(-1) for g in grads])

    def _cos(a, b):
        # 0-norm ⇒ 0/0 ⇒ NaN, deliberately (see docstring)
        return float((a @ b) / (a.norm() * b.norm()))

    (_, _, newvalue, *_rest, pg_rows) = _hybrid_ppo_loss(policy,
                                                         mb_obs,
                                                         mb_actions,
                                                         mb_cont_actions,
                                                         mb_old_logp_d,
                                                         mb_old_logp_c,
                                                         mb_advantages,
                                                         clip_coef,
                                                         state,
                                                         mb_prio=mb_prio,
                                                         mb_masks=mb_masks,
                                                         return_pg_rows=True,
                                                         mb_part=mb_part,
                                                         aim_dim_mask=aim_dim_mask,
                                                         aim_entropy_bonus=aim_entropy_bonus)

    pg_grads = {}                                                      # subset -> {group: flat grad}
    vf_grads = {}                                                      # 'T'/'CT' -> flat value_head grad
    for name, mask in subsets.items():
        w = _row_weights(mask)
        loss_s = (pg_rows * w).sum() / w.sum()
        gs = torch.autograd.grad(loss_s,
                                 pg_params,
                                 retain_graph=True,
                                 allow_unused=True,
                                 materialize_grads=True)
        pg_grads[name] = {
            g: _flat(gs[bounds[i]:bounds[i + 1]]).detach()
            for i, g in enumerate(pg_group_names)
        }
        if name in ("T", "CT"):
                                                                       # vf-only control: unclipped value loss restricted to the subset
            newv = newvalue.view(mb_returns_norm.shape)
            w_full = w.reshape(mb_returns_norm.shape)
            vf_s = 0.5 * (((newv - mb_returns_norm)**2) * w_full).sum() / w_full.sum()
            vgs = torch.autograd.grad(vf_s,
                                      groups["value_head"],
                                      retain_graph=True,
                                      allow_unused=True,
                                      materialize_grads=True)
            vf_grads[name] = _flat(vgs).detach()

    out = {}
    for g in pg_group_names:
        out[f"tag/cossim_cross/{g}/{mb_label}"] = _cos(pg_grads["T"][g], pg_grads["CT"][g])
        out[f"tag/cossim_cross_half/{g}/{mb_label}"] = _cos(pg_grads["T_a"][g], pg_grads["CT_a"][g])
        out[f"tag/cossim_within_t/{g}/{mb_label}"] = _cos(pg_grads["T_a"][g], pg_grads["T_b"][g])
        out[f"tag/cossim_within_ct/{g}/{mb_label}"] = _cos(pg_grads["CT_a"][g], pg_grads["CT_b"][g])
        out[f"tag/gnorm_t/{g}/{mb_label}"] = float(pg_grads["T"][g].norm())
        out[f"tag/gnorm_ct/{g}/{mb_label}"] = float(pg_grads["CT"][g].norm())
    out[f"tag/cossim_vf/{mb_label}"] = _cos(vf_grads["T"], vf_grads["CT"])
    return out


def _inject_tag_metrics(trainer, logs):
    """Move pending TAG metrics into this epoch's logs dict (spec §4.2).

    CALL-ORDER CONSTRAINT: must run AFTER dead_run_detector.check(...) in
    the outer loop — tag/* carries deliberate NaNs (zero-norm subsets,
    documented in tag_grad_cossim) and check() raises RuntimeError on any
    NaN in the metrics dict; injecting earlier aborts the run with exit
    code 3 on the first degenerate subset. Also never route these through
    the `losses` dict: its keys are divided by _mb_run (gh#90), prefixed
    losses/, and lag environment/* by one epoch.

    logs=None (throttled epoch) is a no-op: the top-of-loop reset then
    DROPS the measurement — injecting it next epoch would mislabel its
    step/epoch (spec §4.2 drop semantics).
    """
    pending = getattr(trainer, "_tag_metrics", None)
    if pending and isinstance(logs, dict):
        logs.update(pending)


def _patch_trainer_with_hybrid_aim(trainer,
                                   cont_action_view_main=None,
                                   mask_view_main=None,
                                   participating_rows=None):
    """Extend trainer with continuous-action rollout storage + vecenv plumbing.

    Apply AFTER _patch_trainer_with_return_norm (so train() is wrapped) and
    BEFORE the rollout begins. The PPO-update-side rewrites (callsite at
    src/train.py:~1050) are inlined directly inside _train_with_return_norm
    via the helpers above; this patcher only handles the rollout/storage
    side.

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
    from pufferlib.pufferl import PuffeRL

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
    # build_train_env_factory (env_knobs_from_args bakes args.pin_pitch into
    # every worker env) and (b) BEFORE the --resume-run config guard below:
    # the CLI default is pin_pitch=None ⇒ env_knobs_from_args yields 0, while a
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

    trainer = PuffeRL(train_config, vecenv, policy)
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
    _patch_trainer_with_return_norm(trainer)
    # Batch 3 (T5): hybrid-aim patch ALWAYS runs after return_norm because the
    # train() wrapper installed by return_norm reads self.cont_actions /
    # self.logprobs_{d,c} which this patcher allocates. Order also matters
    # vs. selfplay: selfplay only wraps evaluate(), not train(), so the
    # rollout-side cont_action plumbing must be in place before evaluate()
    # is first called.
    # Pin the shm + view on the trainer so neither is GC'd mid-run. Without
    # holding _cont_action_shm here, Python could free the RawArray once
    # this function returns (Python doesn't know workers/numpy views are
    # using it via the OS-level mapping).
    trainer._cont_action_shm = _cont_action_shm
    trainer._action_mask_shm = _mask_shm               # F8: same GC-pinning rationale

    # Rung 0 §2.2 + Rung 1a T3: static per-run participation vector, env-row
    # major (10 rows per env: T at 0-4, CT at 5-9). Under --opponent self it
    # selects slots 0..n-1 of BOTH teams — the exact slots the C env spawns
    # (cs2_env.py, n_active_per_team); under --opponent noop, the hero team's
    # slots only. THE SAME helper backs train_test_harness, so a harness test
    # can never be green against a formula production does not run.
    # The assert is the agreement check: the vector is derived from args while
    # the envs were built from build_train_env_factory, and a disagreement
    # would mask the wrong rows silently rather than crash.
    _n_active = env_knobs_from_args(args)["n_active_per_team"]
    _participating_rows = build_participating_rows(args.num_envs,
                                                   _n_active,
                                                   opponent_mode=_opponent_mode,
                                                   hero_team=SelfPlayManager.initial_hero_team())
    assert trainer.vecenv.driver_env.n_active_per_team == _n_active, "driver env / args disagree"
    _patch_trainer_with_hybrid_aim(trainer,
                                   cont_action_view_main=_cont_action_view_main,
                                   mask_view_main=_mask_view_main,
                                   participating_rows=_participating_rows)

    # ── Self-play setup ──────────────────────────────────────────────────────
    # F11 (2026-07-06 adversarial review): the selfplay evaluate() wrapper is
    # the ONLY rollout path that understands the hybrid 4-tuple policy
    # contract — stock PuffeRL.evaluate crashes on the forward_eval tuple
    # unpack at its first call, so --no-self-play was broken in production.
    # The patch is now applied UNCONDITIONALLY (mirroring train_test_harness,
    # which adopted this shape at T5); --no-self-play means "no past-policy
    # mixing": p_past=0.0 with an empty, never-seeded pool ⇒ should_use_past()
    # is always False, and the pool save / team-switch bookkeeping in the
    # main loop is skipped via self_play_enabled below.
    self_play_enabled = bool(getattr(args, "self_play", True))
    self_play_mgr = SelfPlayManager(
        pool_size=15,
        p_past=0.3 if self_play_enabled else 0.0,
        save_every_epochs=25,                                          # ~2M steps per save at batch_size=81920
        win_threshold=0.6,
        phase_length=50,                                               # switch opponent team every ~4M steps
        aim_log_std_max=getattr(args, "aim_log_std_max", None),
        pin_pitch=bool(args.pin_pitch),
                                                                       # Rung 1a T3: "noop" ⇒ the patched evaluate() drives opponent_team as a
                                                                       # statue. Guarded above: it cannot combine with self-play, so
                                                                       # opponent_team stays INITIAL_OPPONENT_TEAM — the same team
                                                                       # build_participating_rows masked out.
        opponent_mode=_opponent_mode,
    )
                                                                       # R0-C: on --resume-run the pool comes back from train_state.pt — no re-seed.
    if self_play_enabled and resume_path and resume_path.exists() and not resume_run:
        import shutil as _shutil

        seed_path = Path(args.checkpoint_dir) / "sp_seed.pt"
        _shutil.copy2(resume_path, seed_path)
        self_play_mgr._add_to_pool(seed_path)
        print(f"[SelfPlay] Pool pre-seeded with resume checkpoint ({seed_path.name})")
    _patch_trainer_with_selfplay(trainer, self_play_mgr)
    # R0-E.2: env flag ⇔ policy mask, or stop before the first rollout.
    assert_pin_pitch_agreement(vecenv, policy)
    # R0-G: env aim clamp ⇔ policy tanh scale (a resumed checkpoint may carry
    # a different buffer than the env it is now paired with).
    assert_max_turn_speed_agreement(vecenv, policy)
    if not self_play_enabled:
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
    # timing is the outermost wrapper so it sees all evaluate() calls regardless of selfplay
    _patch_trainer_with_timing(trainer)
    # ────────────────────────────────────────────────────────────────────────

    # ── R0-C (#134): full-state checkpointing + restore ─────────────────────
    # Installed after EVERY patch so the sidecar sees the final aliases.
    _install_full_checkpointing(trainer, self_play_mgr)
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
    # SAME knobs as the workers (env_knobs_from_args + reward_overrides_from_args
    # so a --laser-range / --round-time-ticks run evaluates on what it trains
    # on). Seed 10_000_003: worker env seeds are env_seed_base(--seed) + i, so
    # the only collision is --seed 100 with >= 4 envs (env 3) — see
    # env_seed_base. team_spirit=None → raw rewards (eval never feeds training).
    _eval_hook = None
    _eval_interval = int(getattr(args, "eval_interval", 0) or 0)
    if _eval_interval > 0:
        from eval_baselines import BaselineEvaluator
        _eval_env = make_puffer_env(team_spirit=None,
                                    seed=10_000_003,
                                    map_data=_map_data,
                                    auto_reset=False,
                                    reward_overrides=reward_overrides_from_args(args),
                                    **env_knobs_from_args(args))
        _d = trainer.vecenv.driver_env
        for _k in ("n_active_per_team", "pin_pitch", "crouch_enabled", "jump_enabled",
                   "round_time"):
            if getattr(_eval_env, _k) != getattr(_d, _k):
                raise RuntimeError(f"[Eval] eval env / driver env disagree on {_k}: "
                                   f"{getattr(_eval_env, _k)!r} vs {getattr(_d, _k)!r}")
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
        trainer.evaluate()

        # Rung 0 §2.2: the participating buffer is zero-initialised, so an
        # all-False buffer means evaluate() never ran its scatter — every
        # masked reduction below would then divide by the clamp floor and
        # train on nothing. Fail loudly instead.
        assert trainer.participating.any(), "participating buffer never written this epoch"
        logs = trainer.train()

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
                        default=5,
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
                        default=1,
                        dest="crouch_enabled",
                        help="R0-E.2: 0 masks the crouch action (stance parity for pinned-pitch "
                        "duels; a crouched target is an unobservable guaranteed miss).")
    parser.add_argument("--jump-enabled",
                        type=int,
                        choices=(0, 1),
                        default=1,
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
    # ── Reward weights (spec 2026-08-01 §4.2) ──
    # Generated from REWARD_WEIGHT_DEFAULTS so flag name, dest and default can
    # never disagree with the config key — the dest-typo class of bug (commit
    # 4d9dfa0) is impossible by construction here. Flag == kwarg name with
    # dashes; default == the make_env default, so omitting a flag reproduces
    # today's env exactly.
    for _rw_name, _rw_default in REWARD_WEIGHT_DEFAULTS.items():
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
    # instead of make_policy() 30 s into every retry.
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
    # therefore needs nav/de_dust2.nav + the vis cache on the HOST that runs
    # the dump (Modal fingerprints run host-side).
    if args.map is None:
        args.map = "dust2" if args.dust2 else "simple"
    args.map_data = build_map_data(args.map)
    print(f"[Map] Using {args.map} map")
    resolve_pin_pitch(args)

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
