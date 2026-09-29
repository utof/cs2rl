"""Full-state checkpoint / resume surface (R0-C #134), split out of train.py.

WHAT: RNG snapshot/restore + seeding, the ``train_state.pt`` sidecar
(collect_train_state / restore_train_state), and the ``--resume-run`` resolution
and guard chain. Moved here VERBATIM by the 2026-08-31 post-rung1a refactor: no
renames, no signature changes, no behaviour change. ``train.py`` re-exports
every name below (see its ``__all__``), so existing ``from cs2rl.train import X`` call
sites keep working unchanged.

WHY its own module: this is the one surface whose silent breakage a training run
cannot detect from its own metrics — a resumed run that quietly diverges looks
exactly like a healthy run. It earns a file and a test file of its own rather
than sitting 800 lines into a 7,000-line entry point.

PICKLE COMPAT: collect_train_state stores only ints, tensors and state_dicts —
never a class defined in train.py — so moving these definitions cannot
invalidate an existing ``train_state.pt``.

IMPORT-LIGHTNESS INVARIANT: module scope stays torch/nav/env.c-free, for the
reason spelled out in train_shared.py's header. Every torch import below is
function-local ON PURPOSE.
"""
import json
import math
import random
from pathlib import Path

import numpy as np

from cs2rl.train_shared import _WARMSTART_ATTRS, RESUME_CONFIG_ALLOWLIST


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
    # forever over I001 — gh#97, the same trap train.py's import block
    # documents.)
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
