"""The training run driver: `train(args)` and the helpers only it reads.

`train()` builds the vec env, the policy and the `Cs2PuffeRL` trainer, then runs the
epoch loop (checkpoints, dead-run detection, the fixed-baseline eval hook).
`cs2rl.train.trainer` and `cs2rl.eval.baselines` are imported inside `train()`: both
load torch at module scope, and `python -m cs2rl.train --dump-config` must not.
"""

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from cs2rl.env.factory import build_env_for
from cs2rl.policy import (
    AGENT_IDS,
    build_policy,
    load_state_dict_arch_checked,
    state_dict_is_split,
    state_dict_is_trunk_split,
)
from cs2rl.spec.action import ACTION_MASK_DIM, AIM_DIM
from cs2rl.spec.paths import CHECKPOINTS_DIR
from cs2rl.train.config import (
    TEAM_SIZE,
    assert_opponent_self_play_compatible,
    build_participating_rows,
    build_train_config,
    compute_batch_dims,
    env_config_from_args,
    resolve_opponent_mode,
)
from cs2rl.train.envs import (
    assert_eval_env_agreement,
    assert_max_turn_speed_agreement,
    assert_pin_pitch_agreement,
    auto_vec_workers,
    build_train_env_factory,
    check_spawn_counts,
    env_seed_base,
    resolve_pin_pitch,
)
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
from cs2rl.train.selfplay import SelfPlayManager, build_selfplay_manager, self_play_used_past_metric


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
    # This file is <repo>/src/cs2rl/train.py, so the repo root is parents[2].
    _env_file = Path(__file__).parents[2] / ".env"
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
    # pufferlib.vector.make runs. HybridAimVecEnv.send writes the policy's
    # Δyaw sample into `_cont_action_view_main` every send(); each worker's Cs2Env receives a
    # numpy view onto the same physical bytes via _attach_cont_action_view
    # (called inside env_factory below). For the Serial backend the view is
    # also attached, but the per-env step wrapper installed by HybridAimVecEnv
    # takes precedence — see HybridAimVecEnv for the dual-path contract.
    from multiprocessing import RawArray

    # 10 (5 T + 5 CT) — N_AGENTS not exported via spec.action; use AGENT_IDS.
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
    # env.factory.build_selfplay_manager, which is also what the two harness
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
    # imports torch at module scope; `from cs2rl import train` must stay torch-free
    # (tests/test_w1_modules.py).
    from cs2rl.train.trainer import Cs2PuffeRL, HybridAimVecEnv
    vecenv = HybridAimVecEnv(vecenv, _cont_action_view_main)
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
    # The constructor allocates hybrid rollout buffers before the first
    # evaluate()/train() call; the vecenv wrapper was installed above.
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
        from cs2rl.eval.baselines import BaselineEvaluator
        # W3 (#154), retyped by #165 PR B2: role eval. `team_spirit=None`, the
        # 10_000_003 seed, the load-bearing `auto_reset=False` AND the
        # raw-reward rule all live in env.factory._build_eval; this site passes
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
