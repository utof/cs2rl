"""The training run driver: `train(args)` and the helpers only it reads.

`train()` prepares the run, builds the trainer with `cs2rl.train.compose.build_trainer`,
then runs the epoch loop (checkpoints, dead-run detection, the fixed-baseline eval
hook). torch and `cs2rl.eval.baselines` are imported inside the functions that use
them: `python -m cs2rl.train --dump-config` imports this module and must not load torch.
"""

import json
import multiprocessing as mp
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from cs2rl.env.factory import build_eval_env
from cs2rl.policy import state_dict_is_split, state_dict_is_trunk_split
from cs2rl.spec.paths import CHECKPOINTS_DIR
from cs2rl.train.compose import PolicyInit, _close_on_exit, build_trainer
from cs2rl.train.config import (
    TEAM_SIZE,
    assert_opponent_self_play_compatible,
    build_train_config,
    compute_batch_dims,
    env_config_from_args,
    resolve_opponent_mode,
)
from cs2rl.train.envs import assert_eval_env_agreement, env_seed_base, resolve_pin_pitch
from cs2rl.train.metrics import (
    ScheduledEval,
    _inject_tag_metrics,
    compute_game_metrics,
    compute_head_divergence,
    compute_network_health,
    compute_trunk_divergence,
    elimination_only_win_rates,
    format_train_status,
    log_aim_log_std,
)
from cs2rl.train.resume import (
    AIM_LOG_STD_RESUME_INIT,
    _atomic_save_state_dict,
    check_resume_config,
    check_resume_metrics_bound,
    convert_legacy_state_dict_to_split,
    convert_shared_trunk_to_split,
    load_full_resume,
    reinit_frozen_aim_log_std,
    resolve_resume_run,
    resolve_resume_split,
    resolve_run_name,
    seed_everything,
)
from cs2rl.train.selfplay import self_play_used_past_metric


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


# ── SECTION: PufferLib training ────────────────────────────────────────────


@dataclass(frozen=True)
class _RunPlan:
    """What `_prepare_run` resolved before any output is opened or env is built."""
    config: dict                       # build_train_config's dict, built once
    opponent_mode: str
    resume_paths: dict | None          # resolve_resume_run's paths on --resume-run


@dataclass(frozen=True)
class _RunOutputs:
    """The run's sinks; train()'s outermost ExitStack closes them."""
    run_id: str
    metrics_path: Path
    metrics_file: Any
    wandb_run: Any


@dataclass
class _EpochLoop:
    """The state the epoch loop reads and updates."""
    args: Any
    trainer: Any
    outputs: _RunOutputs
    shared_ts: Any
    eval_hook: Any
    dead_run_detector: DeadRunDetector
    self_play_enabled: bool
    resumed_from_step: int | None      # stamped on the first new row, then None
    save_path: Path
    last_save: float


def _load_dotenv():
    """Load <repo>/.env into os.environ, never overriding a set variable (WANDB_* for --wandb).

    PITFALL: this file is <repo>/src/cs2rl/train/loop.py, so the repo root is
    parents[3]. parents[2] is src/: a wrong index silently reads <repo>/src/.env
    and a local --wandb run loses every WANDB_* setting.
    """
    env_file = Path(__file__).parents[3] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def _resolve_resume_run(args) -> dict | None:
    """R0-C (#134): --resume-run's checkpoint paths; None without the flag.

    Sets ``args.checkpoint_dir`` to the run dir, ``args.resume`` to its newest model
    (the weights then go through resolve_resume_split like --resume's) and
    ``args.run_id`` to the id saved with it, so resumed metrics rows share the id.
    """
    resume_run = getattr(args, "resume_run", None)
    if not resume_run:
        return None
    if getattr(args, "resume", None):
        raise SystemExit("[Resume] --resume and --resume-run are mutually exclusive")
    run_dir = Path(resume_run).resolve()
    # --checkpoint-dir has default=None in the CLI precisely so this check can tell
    # "given" from "omitted"; _prepare_run resolves None to CHECKPOINTS_DIR afterwards.
    if args.checkpoint_dir is not None and Path(args.checkpoint_dir).resolve() != run_dir:
        raise SystemExit(f"[Resume] --checkpoint-dir {args.checkpoint_dir} disagrees with "
                         f"--resume-run {run_dir}")
    args.checkpoint_dir = str(run_dir)
    paths = resolve_resume_run(run_dir, getattr(args, "run_id", None))
    args.resume = str(paths["model_path"])
    args.run_id = paths["run_id"]
    return paths


def _prepare_run(args) -> _RunPlan:
    """Resolve and validate everything that needs no env, before any output is opened.

    Every refusal here (a bad --seed, --opponent noop with self-play, a --resume-run
    whose config.json disagrees) happens before W&B or metrics.jsonl exist and in
    milliseconds rather than after the env spawns.

    ORDER: pin_pitch is resolved (R0-E.2, #131) before the config is built, because
    env_config_from_args bakes args.pin_pitch into config.json and into every env, and
    so before the --resume-run config guard: the CLI default pin_pitch=None normalises
    to the un-pinned flag while a pinned run's config.json holds 1, and resolving after
    the guard refused every flag-less resume of a flat-map run (Task 12 ruling). The
    CLI already resolved it above --dump-config; here it is a cache-safe cross-check
    for programmatic callers, and it loads the real map when args.map_data is None
    (the dust2 path). The config is built once, after --resume-run has set the run
    dir, and the guard and the run share it.
    """
    if getattr(args, "name", None):
        resolved_name = resolve_run_name(args.name)
        args.checkpoint_dir = str(CHECKPOINTS_DIR / resolved_name)
        print(f"[Train] Run name resolved to: {resolved_name}")
    resolve_pin_pitch(args, verbose=False)
    # R0-D: refuses an out-of-range --seed.
    env_seed_base(args.seed)
    # Rung 1a T3: the CLI refused this above --dump-config; repeated so a programmatic
    # train(args) cannot start a run whose statue team self-play would swap out.
    opponent_mode = resolve_opponent_mode(args)
    assert_opponent_self_play_compatible(opponent_mode, bool(getattr(args, "self_play", True)))
    resume_paths = _resolve_resume_run(args)
    if args.checkpoint_dir is None:    # the CLI default
        args.checkpoint_dir = str(CHECKPOINTS_DIR)
    _, bptt_horizon, batch_size = compute_batch_dims(args.num_envs)
    config = build_train_config(args, batch_size=batch_size, bptt_horizon=bptt_horizon)
    if resume_paths is not None:
        check_resume_config(Path(args.checkpoint_dir), config)
    return _RunPlan(config=config, opponent_mode=opponent_mode, resume_paths=resume_paths)


def _open_run_outputs(args, cleanup: ExitStack) -> _RunOutputs:
    """W&B (with --wandb) and metrics.jsonl, each registered on ``cleanup`` once acquired."""
    run_label = Path(args.checkpoint_dir).name
    wandb_run = None
    if getattr(args, "wandb", False):
        import wandb

        wandb_run = wandb.init(
            project=getattr(args, "wandb_project", "cs2rl"),
            entity=getattr(args, "wandb_entity", None) or None,
            name=run_label,
        )

        def finish_wandb(exc_type, error, traceback):
            """Finalize once, using the failure the caller will actually receive."""
            exit_code = 0 if error is None else 1
            if isinstance(error, SystemExit):
                exit_code = error.code if isinstance(error.code, int) else int(
                    error.code is not None)
            return _close_on_exit(partial(wandb_run.finish, exit_code=exit_code), exc_type, error,
                                  traceback)

        cleanup.push(finish_wandb)
        print(f"[Train] W&B run: {wandb_run.url}")

    metrics_path = Path(args.checkpoint_dir) / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_path.open("a")

    def close_metrics():
        """Close the buffered file before declaring metrics persisted."""
        metrics_file.close()
        print(f"[Train] Metrics saved to {metrics_path}")

    cleanup.push(partial(_close_on_exit, close_metrics))
    # N2: the file is opened in APPEND mode, so runs into one checkpoint dir share it.
    # Every row carries a per-process run id (label + launch timestamp; the label alone
    # is shared by re-runs into the same dir). R0-C: --run-id, or the id --resume-run
    # read back from the checkpoint, overrides the timestamped default so resumed rows
    # share the id. Old rows lack the key.
    run_id = getattr(args, "run_id", None) or f"{run_label}-{time.strftime('%Y%m%d-%H%M%S')}"
    return _RunOutputs(run_id=run_id,
                       metrics_path=metrics_path,
                       metrics_file=metrics_file,
                       wandb_run=wandb_run)


def _resume_policy_init(args, *, full_state: bool) -> PolicyInit:
    """The policy's architecture, and the --resume / --resume-run weights converted to it.

    WHY here, before build_trainer (Batch 7, spec §3.3): the split bits must be known
    when build_policy runs, so the checkpoint is read once, its keys decide the
    architecture (omitted flags never narrow a split checkpoint), and the same dict
    is loaded after construction. A missing checkpoint fails here, before any env.

    ORDER is load-bearing (spec 2026-08-15 §3.3): σ re-init on the LEGACY dict, then
    the heads convert (it needs the bare aim_log_std), then the trunk convert.
    gh#91: a BC warm-start checkpoint carries aim_log_std frozen at LOG_STD_INIT, and
    without the re-init the KL early stop throttles the run. A full-state resume
    (``full_state``) restores the exact pre-crash σ and is never widened.
    """
    heads, trunk, state_dict, resume_path = resolve_resume_split(
        getattr(args, "resume", None),
        heads_flag=bool(getattr(args, "tct_split_heads", False)),
        trunk_flag=bool(getattr(args, "tct_split_trunk", False)))
    if resume_path is None:
        return PolicyInit(tct_split_heads=heads, tct_split_trunk=trunk)
    if not full_state and reinit_frozen_aim_log_std(state_dict,
                                                    cap=getattr(args, "aim_log_std_max", None)):
        print(f"[Train] BC-frozen aim_log_std detected in {resume_path.name}: "
              f"re-initialized to log(0.3) ≈ {AIM_LOG_STD_RESUME_INIT:.3f} (gh#91)")
    if heads and not state_dict_is_split(state_dict):
        state_dict = convert_legacy_state_dict_to_split(state_dict)
        print("[Train] Warm split: duplicated the legacy policy heads into per-team "
              "T/CT copies (spec 2026-08-13 §3.3) — both teams start identical.")
    if trunk and not state_dict_is_trunk_split(state_dict):
        state_dict = convert_shared_trunk_to_split(state_dict)
        print("[Train] Warm split: duplicated the shared encoder+LSTM into per-team "
              "T/CT copies (spec 2026-08-15 §3.3) — both teams start identical.")
    return PolicyInit(tct_split_heads=heads,
                      tct_split_trunk=trunk,
                      state_dict=state_dict,
                      source=str(resume_path))


def _write_config_json(args, config: dict):
    """Provenance dump: <checkpoint_dir>/config.json, sorted so two runs diff by HP only.

    The fingerprint hash is taken at --dump-config time; this copy is for later
    inspection, so a serialization failure is a warning, never a dead run.
    """
    try:
        (Path(args.checkpoint_dir) / "config.json").write_text(
            json.dumps(config, sort_keys=True, indent=2, default=str))
    except Exception as e:
        print(f"[Train] WARN: failed to write config.json: {e}")


def _preseed_selfplay_pool(args, trainer, resume_path: Path):
    """Copy the warm-start checkpoint into the run dir and put it in the self-play pool.

    The pool is read only inside evaluate(), so seeding after construction is the same
    as before it. Not on --resume-run: there the pool comes back from train_state.pt.
    """
    import shutil

    seed_path = Path(args.checkpoint_dir) / "sp_seed.pt"
    shutil.copy2(resume_path, seed_path)
    trainer._self_play_mgr._add_to_pool(seed_path)
    print(f"[SelfPlay] Pool pre-seeded with resume checkpoint ({seed_path.name})")


def _configure_run_trainer(trainer, run_id: str, config: dict):
    """What a CLI run sets on the built trainer: the run id, weight decay, the σ group.

    ORDER: before load_full_resume, whose optimizer state has the σ group; the σ group
    clones group 0's hyper-parameters, so it is split off after the decay is set.
    """
    # R0-C: PuffeRL's NoLogger invents a timestamp run_id; pin ours so
    # <data_dir>/<run_id>/ matches the metrics rows and --resume-run can find it.
    trainer.logger.run_id = run_id
    trainer.optimizer.param_groups[0]["weight_decay"] = config["weight_decay"]
    # Rung 1a T1: …but NOT on the aim σ. Decay adds wd·θ to the gradient and
    # aim_log_std is always negative, so it would drift σ upward at exactly
    # zero true gradient — faking the gate's learning signal and eating the
    # init's clamp margin (see the helper's PITFALLS).
    n_sigma = isolate_aim_log_std_param_group(trainer)
    print(f"[Train] aim_log_std: {n_sigma} parameter(s) moved to a weight_decay=0 param group "
          f"(fresh init {config['aim_log_std_init']:.4f}, "
          f"cap {config['aim_log_std_max']:.4f})")


def _print_opponent_setup(trainer, plan: _RunPlan, self_play_enabled: bool):
    """The run log's statement of who plays whom (T4's pre-flight reads the noop line)."""
    if not self_play_enabled:
        # Mode-aware: "both teams use the current policy" is FALSE under --opponent
        # noop, and the run log must not print two contradictory claims.
        if plan.opponent_mode == "noop":
            print("[Train] Self-play mixing disabled (--no-self-play): the hero team "
                  "uses the current policy every epoch; the opponent team is a statue "
                  "(--opponent noop), not the current policy.")
        else:
            print("[Train] Self-play mixing disabled (--no-self-play): "
                  "both teams use the current policy every epoch.")
    if plan.opponent_mode == "noop":
        # Rung 1a T3: which team is frozen, how many rows train, and on what horizon —
        # the three things a short or mis-masked run would get wrong silently.
        rows = trainer._participating_rows_np
        print(f"[Train] Opponent mode 'noop': team "
              f"{trainer._self_play_mgr.opponent_team.upper()} is a stationary statue; "
              f"{int(rows.sum()):,} of {rows.size:,} agent rows "
              f"participate (raw horizon {plan.config['total_timesteps']:,} rows = "
              f"{plan.config['participating_timesteps']:,} hero steps).")


def _last_metrics_step(metrics_path: Path, run_id: str) -> int | None:
    """The ``step`` of this run id's last metrics.jsonl row, skipping unparsable lines."""
    last = None
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_id") == run_id:
                last = row.get("step", last)
    return last


def _resume_full_state(trainer, plan: _RunPlan, outputs: _RunOutputs) -> int:
    """R0-C: restore the trainer from --resume-run and bound it by the last metrics row.

    Returns the resumed global step. The bound (check_resume_metrics_bound) is
    checkpoint_interval epochs wide on both sides, in participating units. Rung 1a T3:
    it assumes BOTH teams participate, so under --opponent noop it is 2× too wide —
    only ever too permissive, never a false alarm; halve it before a noop run resumes.
    """
    info = load_full_resume(trainer, trainer._self_play_mgr, plan.resume_paths)
    resumed = info["resumed_from_step"]
    config = plan.config
    bound = config["batch_size"] * config["n_active_per_team"] // TEAM_SIZE
    last = _last_metrics_step(outputs.metrics_path, outputs.run_id)
    if last is not None:
        check_resume_metrics_bound(resumed, last, config["checkpoint_interval"], bound)
        print(f"[Resume] global_step {resumed:,} (last metrics row {last:,}, "
              f"gap {resumed - last:+,}) epoch {trainer.epoch}")
    return resumed


def _build_eval_hook(args, trainer, cleanup: ExitStack):
    """R0-I (Task 13): the fixed-baseline eval hook for --eval-interval N > 0, else None.

    The eval env is a parent-process Serial env with the run's EnvConfig, from
    env_config_from_args, the resolver the training envs' factory also reads, so a
    --laser-range / --round-time-ticks run evaluates on what it trains on;
    build_eval_env owns its seed 10_000_003, auto_reset=False and raw rewards.
    ``cleanup`` owns the env from construction, so a failed agreement check or
    evaluator still closes it, and then the hook, which closes the env.
    """
    interval = int(getattr(args, "eval_interval", 0) or 0)
    if interval <= 0:
        return None
    from cs2rl.eval.baselines import BaselineEvaluator

    eval_env = build_eval_env(map_data=args.map_data, config=env_config_from_args(args))
    cleanup.push(partial(_close_on_exit, eval_env.close))
    assert_eval_env_agreement(eval_env, trainer.vecenv.driver_env)
    hook = ScheduledEval(BaselineEvaluator(eval_env, episodes=40, seed=args.seed), interval,
                         trainer.uncompiled_policy, args.device)
    cleanup.pop_all()
    cleanup.push(partial(_close_on_exit, hook.close))
    print(f"[Eval] fixed-baseline eval every {interval} epochs "
          f"(40 episodes vs random + oracle, round_time={eval_env.round_time})")
    return hook


def _dead_run_detector(trainer) -> DeadRunDetector:
    """gh#93: the zero-kills rule is armed only when the env rewards kills."""
    kills_expected = _kill_reward_is_active(trainer.vecenv)
    if not kills_expected:
        print("[Train] Kill reward is 0 — dead-run zero-kills alert disabled (gh#93).")
    return DeadRunDetector(kills_expected=kills_expected)


def _save_policy(trainer, path: Path):
    """dust2_policy.pt: the policy weights alone, written atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_state_dict(trainer.uncompiled_policy.state_dict(), path)


def _abort_dead_run(run: _EpochLoop):
    """F14: save an autopsy checkpoint and exit 3, so wrappers see the failure.

    The detector's verdict used to be discarded: the 30M degenerate run printed its
    banner and kept training for ~150 epochs. Opt out with --no-dead-run-abort.
    """
    import torch

    trainer = run.trainer
    autopsy_path = Path(run.args.checkpoint_dir) / "dust2_policy_dead.pt"
    autopsy_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.uncompiled_policy.state_dict(), autopsy_path)
    run.outputs.metrics_file.flush()
    print(f"[Train] DEAD RUN — aborting at step {trainer.global_step:,}. "
          f"Autopsy checkpoint: {autopsy_path}")
    raise SystemExit(3)


def _log_selfplay(run: _EpochLoop, logs: dict):
    """Self-play bookkeeping: team switch, pool save, and the self_play/* keys.

    F11: called only with self-play on (the FLAG; the manager always exists, because
    evaluate() needs it), so --no-self-play makes no pool saves or team switches.
    """
    trainer = run.trainer
    mgr = trainer._self_play_mgr
    mgr.maybe_switch_teams(trainer.epoch)
    # R0-I: elimination-only — winner_ct counts timeouts, which would pool-save a
    # passive CT as "dominant".
    win_rate_t, win_rate_ct = elimination_only_win_rates(logs)
    mgr.maybe_save(trainer.uncompiled_policy, Path(run.args.checkpoint_dir), trainer.epoch,
                   win_rate_t, win_rate_ct)
    logs["self_play/pool_size"] = float(len(mgr.pool))
    # Observe-only (spec 2026-08-15 §3.4): a 0.0/1.0 float on the OUTER logs dict, set
    # by evaluate(), not train(). self_play/opponent_id is a string the persist filter
    # would drop.
    logs["self_play/used_past"] = self_play_used_past_metric(trainer)
    logs["self_play/opponent_team"] = float(mgr.opponent_team == "ct") # 1.0 = CT opponent


def _persist_row(run: _EpochLoop, logs: dict, ts_val: float):
    """Append one metrics.jsonl row (numeric logs only) and send it to W&B."""
    trainer = run.trainer
    log_entry = {
        "run_id": run.outputs.run_id,                                  # N2: segment runs by this, not by agent_steps resets
        "step": trainer.global_step,
        "epoch": trainer.epoch,
        "team_spirit": ts_val,
        **{
            k: v
            for k, v in logs.items() if isinstance(v, (int, float))
        },
    }
    if run.resumed_from_step is not None:                              # R0-C: the analysis seam marker
        log_entry["resumed_from_step"] = run.resumed_from_step
        run.resumed_from_step = None
    run.outputs.metrics_file.write(json.dumps(log_entry) + "\n")
    run.outputs.metrics_file.flush()
    if run.outputs.wandb_run is not None:
        run.outputs.wandb_run.log(log_entry, step=trainer.global_step)


def _log_epoch(run: _EpochLoop, logs: dict, ts_val: float):
    """Enrich one epoch's logs dict, check it for a dead run, and persist it.

    ORDER: the dead-run check reads the game metrics, so they are added first; the
    TAG injection comes after the check, because TAG cells carry deliberate NaNs
    (see _inject_tag_metrics).
    """
    trainer, policy = run.trainer, run.trainer.uncompiled_policy
    logs.update(compute_game_metrics(logs))
    # NaN/Inf raises inside check(); a True verdict aborts unless --no-dead-run-abort.
    if (run.dead_run_detector.check(trainer.global_step, logs)
            and getattr(run.args, "dead_run_abort", True)):
        _abort_dead_run(run)
    # Network health is too expensive for every epoch.
    if trainer.epoch % 5 == 0:
        logs.update(compute_network_health(policy, run.args.device))
    if run.self_play_enabled:
        _log_selfplay(run, logs)
    # Batch 3.5 (#24): per-axis aim log_std, the CLAMPED values the policy used (T7
    # gate 2 reads aim_log_std_pitch; format_train_status keeps it greppable). The
    # helper branches on architecture: a split policy has no `aim_log_std`.
    log_aim_log_std(policy, logs)
    # Batch 7 (spec §3.4) and its trunk twin (spec 2026-08-15 §3.4): the analyzer's
    # split labels, derived from the POLICY OBJECT, never from config.json (which a
    # flag-less crash-resume of a split run rewrites as false). Written on EVERY row,
    # not behind --tag-diagnostic / --tag-every: an unlabeled row reads as a legacy run.
    logs["split/active"] = float(hasattr(policy, "aim_log_std_t"))
    logs["split/trunk_active"] = float(hasattr(policy, "encoder_t"))
    logs.update(compute_head_divergence(policy))
    logs.update(compute_trunk_divergence(policy))
    _inject_tag_metrics(trainer, logs)
    _persist_row(run, logs, ts_val)


def _run_epochs(run: _EpochLoop):
    """evaluate() and train() until the epoch budget is spent; log, evaluate and save."""
    trainer = run.trainer
    while trainer.epoch < trainer.total_epochs:
        # TAG: drop any un-injected measurement.
        trainer._tag_metrics = None
        t0 = time.perf_counter()
        trainer.evaluate()
        trainer._timing["collect_ms"] = (time.perf_counter() - t0) * 1000.0
        # Rung 0 §2.2: the participating buffer is zero-initialised, so all-False means
        # evaluate() never ran its scatter and every masked reduction would divide by
        # the clamp floor and train on nothing.
        assert trainer.participating.any(), "participating buffer never written this epoch"
        t0 = time.perf_counter()
        logs = trainer.train()
        trainer._timing["update_ms"] = (time.perf_counter() - t0) * 1000.0
        # Before the eval hook, which is handed this dict with the timing keys in it.
        if isinstance(logs, dict):
            logs["timing/collect_ms"] = trainer._timing["collect_ms"]
            logs["timing/update_ms"] = trainer._timing["update_ms"]
        # Team spirit annealing: 0.3→0.7 over 5M participating-agent steps.
        ts_val = min(0.7, 0.3 + trainer.global_step / 5_000_000)
        run.shared_ts.value = ts_val
        # R0-I: OUTSIDE the isinstance(logs, dict) guard on purpose — see ScheduledEval
        # (the 0.25 s log throttle must not skip an eval epoch).
        if run.eval_hook is not None:
            run.eval_hook.after_train(trainer, logs)
        if isinstance(logs, dict):
            _log_epoch(run, logs, ts_val)
        if time.time() - run.last_save > run.args.save_every_sec:
            _save_policy(trainer, run.save_path)
            run.last_save = time.time()
            print(f"Saved checkpoint to {run.save_path}")
        if isinstance(logs, dict):
            if trainer.epoch % 10 == 0:
                print(format_train_status(trainer.epoch, ts_val, logs))
            print(f"[Timing] collect={trainer._timing['collect_ms']:.0f}ms  "
                  f"update={trainer._timing['update_ms']:.0f}ms  "
                  f"SPS={logs.get('SPS', 0):.0f}")


def train(args):
    """Run PPO training via PufferLib 3.0 (`python -m cs2rl.train --train`).

    WHAT: resolves the run (`_prepare_run`), opens its outputs, builds the trainer
    with `cs2rl.train.compose.build_trainer`, sets what only a CLI run has (run id,
    weight decay, aim-σ group, full-state resume, eval hook, dead-run detector), runs
    the epochs and saves dust2_policy.pt.

    OWNERSHIP: run_cleanup owns W&B and metrics.jsonl for the whole run.
    train_cleanup owns the trainer: `close_resources` until every setup and resume
    check has passed, then `close`, which also writes PufferLib's checkpoint, so a
    refused or partial resume never overwrites the checkpoint it resumed from.
    eval_cleanup owns the eval env, then the eval hook. Training closes before
    evaluation, then the final policy save runs, then the outputs close.
    """
    import torch

    # The few paths that still build torch.distributions objects skip the per-call
    # argument validation; their inputs are valid by construction.
    torch.distributions.Distribution.set_default_validate_args(False)
    _load_dotenv()
    plan = _prepare_run(args)
    full_state = plan.resume_paths is not None
    with ExitStack() as run_cleanup:
        outputs = _open_run_outputs(args, run_cleanup)
        # Team spirit; every env reads it at episode start.
        shared_ts = mp.Value("f", 0.3)
        # R0-D (#135): pufferl.py has its seeding commented out. Seed before the policy
        # and the vecenv are built (weight init; env seeds come from env_seed_base) and
        # before any random.* consumer. On --resume-run, load_full_resume then restores
        # the saved python/numpy/torch states over this seed; the env xorshift32 state
        # is not restored (see its WARN). CPU runs are bit-exact; CUDA runs are seeded
        # but not bit-exact (no deterministic-algorithm flags are set).
        seed_everything(args.seed)
        policy_init = _resume_policy_init(args, full_state=full_state)
        if plan.config.get("warmstart_entropy") and policy_init.state_dict is None:
            print("[Train] WARN: --warmstart-entropy without --resume — the grace window "
                  "will suppress entropy pressure on a from-scratch policy (legal, but "
                  "probably not what you want).")
        _write_config_json(args, plan.config)
        with ExitStack() as eval_cleanup, ExitStack() as train_cleanup:
            trainer = build_trainer(args, plan.config, shared_ts=shared_ts, policy_init=policy_init)
            train_cleanup.push(partial(_close_on_exit, trainer.close_resources))
            self_play_enabled = bool(getattr(args, "self_play", True))
            if self_play_enabled and policy_init.source is not None and not full_state:
                _preseed_selfplay_pool(args, trainer, Path(policy_init.source))
            _configure_run_trainer(trainer, outputs.run_id, plan.config)
            _print_opponent_setup(trainer, plan, self_play_enabled)
            resumed_from_step = _resume_full_state(trainer, plan, outputs) if full_state else None
            eval_hook = _build_eval_hook(args, trainer, eval_cleanup)
            run = _EpochLoop(args=args,
                             trainer=trainer,
                             outputs=outputs,
                             shared_ts=shared_ts,
                             eval_hook=eval_hook,
                             dead_run_detector=_dead_run_detector(trainer),
                             self_play_enabled=self_play_enabled,
                             resumed_from_step=resumed_from_step,
                             save_path=Path(args.checkpoint_dir) / "dust2_policy.pt",
                             last_save=time.time())
            print(f"[Train] Starting PufferLib PPO for {args.timesteps:,} env steps...")
            train_cleanup.pop_all()
            train_cleanup.push(partial(_close_on_exit, trainer.close))
            _run_epochs(run)
        _save_policy(trainer, run.save_path)
        print(f"[Train] Final checkpoint saved to {run.save_path}")
    print("[Train] Done.")
