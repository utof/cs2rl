"""Full-state checkpoint / resume surface (R0-C #134), split out of the flat train.py.

WHAT: RNG snapshot/restore + seeding, the ``train_state.pt`` sidecar
(collect_train_state / restore_train_state), and the ``--resume-run`` resolution
and guard chain. Moved here VERBATIM by the 2026-08-31 post-rung1a refactor: no
renames, no signature changes, no behaviour change. The flat ``train.py``
re-exported every name below (see its ``__all__``) until #205 part 3 removed the
re-exports: ``cs2rl.train`` exports nothing now, so import from this module.

WHY its own module: this is the one surface whose silent breakage a training run
cannot detect from its own metrics — a resumed run that quietly diverges looks
exactly like a healthy run. It earns a file and a test file of its own rather
than sitting 800 lines into a 7,000-line entry point.

PICKLE COMPAT: collect_train_state stores only ints, tensors and state_dicts —
never a class defined in the flat train.py — so moving these definitions cannot
invalidate an existing ``train_state.pt``.

IMPORT-LIGHTNESS INVARIANT: module scope stays torch/nav/env.c-free, for the
reason spelled out in tests/train/test_w1_modules.py's docstring (WHY property 3 is
load-bearing). Every torch import below is function-local ON PURPOSE.
"""

import json
import math
import os
import random
from pathlib import Path

import numpy as np

from cs2rl.policy import LOG_STD_INIT, state_dict_is_split, state_dict_is_trunk_split
from cs2rl.spec.paths import CHECKPOINTS_DIR

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


def _atomic_save_state_dict(state_dict, path):
    """torch.save via sibling .tmp + os.replace so a crash never corrupts ``path``.

    WHY: the periodic save in train() overwrites ONE file (dust2_policy.pt)
    every --save_every_sec. The training box's GPU is known to fall off the
    PCI bus under thermal load (hard crash, 2026-08-13); a plain torch.save
    interrupted mid-write would leave the ONLY recovery checkpoint torn.
    os.replace() is an atomic rename on POSIX, so ``path`` always holds a
    complete checkpoint — old or new, never partial.

    PITFALL: torch is imported lazily — this module's level (and
    cs2rl.train.__main__'s, which reaches it through cs2rl.train.loop) must stay
    torch-free so --dump-config keeps its no-heavy-imports guarantee.
    """
    import torch

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_dict, tmp)
    os.replace(tmp, path)


# ── R0-C (#134): full-state checkpoint / resume ───────────────────────────
# PufferLib 3.0 saves model + optimizer + step counters (pufferl.py
# save_checkpoint) and ships NO loader. Everything cs2rl.train.trainer.Cs2PuffeRL layers on top
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
# Trainer attrs of the warm-start entropy machine + SAC target (all set in
# cs2rl.train.trainer.Cs2PuffeRL._init_return_norm). Plain Python scalars/None — pickled as-is.
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
    """Everything Cs2PuffeRL adds on top of PuffeRL's trainer_state.pt, CPU-side
    so the file is device-agnostic. Requires a Cs2PuffeRL (the _log_alpha_tensor /
    _alpha_optimizer / _ret_* attributes its _init_return_norm sets, gh#168 W2a)."""
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

    PITFALL: copy_() into `_ret_*` and `_log_alpha_tensor`, never rebind.
    Until gh#168 W2a all four were closure locals aliased onto the trainer,
    so a rebind of any of them left the closure training on its own stale
    copy. Since W2a the two cases differ: `_log_alpha_tensor` MUST stay in
    place, because `_alpha_optimizer` holds the original tensor as its
    parameter and a rebound one would never be stepped; `_ret_*` could be
    rebound harmlessly now (Cs2PuffeRL.train, _update_return_stats and
    collect_train_state all read the attribute), and copy_() is kept there
    for the bit-exact round trip and so that all four follow one rule.
    """
    # Function-local ON PURPOSE: this module's scope stays torch-free.
    # (The comment sits on its own line, not after the import: a trailing
    # comment there makes yapf's column aligner and ruff's isort fight
    # forever over I001 — gh#97.)
    import torch
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
