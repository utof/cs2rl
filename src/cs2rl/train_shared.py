"""Leaf module: constants and pure helpers shared by train.py and its siblings.

WHAT: the symbols both ``train.py`` and the modules carved out of it
(``resume_state.py``, ``train_config.py``, ...) need. Moved here VERBATIM from
train.py by the 2026-08-31 post-rung1a refactor: no renames, no signature
changes, no behaviour change. ``train.py`` re-exports every name below (see its
``__all__``), so the existing ``from cs2rl.train import X`` call sites — tests,
scripts and src/ siblings alike — keep working unchanged.

WHY a leaf: this module imports NOTHING from train.py or from any other module
split out of it. That is what keeps the new dependency graph acyclic; every
other new module may import this one, never the reverse. An import of ``train``
here is a cycle: pyproject.toml's `cs2rl acyclic siblings` contract rejects it.

IMPORT-LIGHTNESS INVARIANT (measured, load-bearing): module scope here must stay
free of torch, nav and env.c. train.py imports this module at ITS module level,
and ``train.py --dump-config`` guarantees no torch/nav import
(tests/test_train_cli.py::test_dump_config_writes_json — "zero side-effects"),
so a heavy import added here silently costs every --dump-config call ~30 s and
breaks the Modal/run_rung1 fingerprint step. Every torch/nav/env.c import below
is function-local ON PURPOSE. tests/test_w1_modules.py enforces this in a fresh
interpreter.
"""
import math
import os
from pathlib import Path

import numpy as np

from cs2rl.spec.action import ACTION_HEAD_SIZES

# Agents per team. A bare literal ON PURPOSE: this leaf and train.py must both
# stay import-light (`--dump-config` guarantees no torch/nav import — see
# _atomic_save_state_dict's docstring below), and `nav` pulls
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
# Module-level so tests can `from cs2rl import train; train.LOG_STD_MIN` without poking
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


# Fix #2: precomputed log(2π) for the analytic Normal log-prob/entropy
# replacing torch.distributions.Normal in _hybrid_sample_logits.
_LOG_2PI = math.log(2.0 * math.pi)

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


def _atomic_save_state_dict(state_dict, path):
    """torch.save via sibling .tmp + os.replace so a crash never corrupts ``path``.

    WHY: the periodic save in train() overwrites ONE file (dust2_policy.pt)
    every --save_every_sec. The training box's GPU is known to fall off the
    PCI bus under thermal load (hard crash, 2026-08-13); a plain torch.save
    interrupted mid-write would leave the ONLY recovery checkpoint torn.
    os.replace() is an atomic rename on POSIX, so ``path`` always holds a
    complete checkpoint — old or new, never partial.

    PITFALL: torch is imported lazily — this module's level (and train.py's,
    which imports it) must stay torch-free so --dump-config keeps its
    no-heavy-imports guarantee.
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
# Cs2PuffeRL._init_return_norm, src/cs2rl/trainer.py). Plain Python scalars/None — pickled as-is.
_WARMSTART_ATTRS = ("_batch1_warmstart_phase", "_batch1_last_entropy_mean",
                    "_batch1_log_alpha_reset_done", "_batch1_current_target_entropy",
                    "_batch1_warmstart_h_anchor", "_batch1_warmstart_h0",
                    "_batch1_warmstart_warn_epoch")

DEFAULT_GAMMA = 0.999                  # R0-J: the historical PPO discount; --gamma default


def resolve_gammas(args) -> tuple[float, float]:
    """Return ``(gamma, pbrs_gamma)`` from the args object.

    WHAT: ``gamma`` is ``args.gamma`` (default DEFAULT_GAMMA for harness /
    dump-config args objects that predate the flag); ``pbrs_gamma`` is
    ``args.pbrs_gamma`` when given, else ``gamma``.

    WHY one helper: its two callers, build_train_config (provenance + the PPO
    discount) and env_config_from_args (the env's PBRS discount, stored on the
    EnvConfig every env is built from), must agree on the SAME resolution rule
    — PBRS is only policy-invariant (Ng et al.) when
    γ_pbrs == γ, and before R0-J the two lived as unrelated literals (train.py
    0.999 vs cs2_env.py 0.999, now one field default in env/config.py) held
    together by a single drift test.
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


# (args attr, EnvConfig field) — single source for env_config_from_args's R0-G
# pairs AND build_train_config's provenance keys (which record the value under
# the ARGS attr name), so a knob added to one cannot be missed by the other
# (config.json would then silently under-record the experiment). Paired with
# train_config._ARGS_KNOB_FIELDS; see its comment for the coverage rule.
_R0G_KNOBS = (("round_time_ticks", "round_time"), ("laser_range", "laser_range"),
              ("max_turn_speed", "max_turn_speed"))


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
