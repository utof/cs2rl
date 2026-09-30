"""CS2 RL Sim: the training entry point, `python -m cs2rl.train`.

Usage:
  python -m cs2rl.train --smoke     # sanity check: 20k native-env steps, no crash, print steps/sec
  python -m cs2rl.train --train     # full PPO self-play training (PufferLib 3.0)
  python -m cs2rl.train --record    # run 1 episode, save rerun recording (random policy)
  python -m cs2rl.train --eval      # evaluate a checkpoint across many seeds

This module is the command line only: it parses the flags, validates them torch-free
(so `--dump-config` stays cheap), then dispatches one mode: --dump-config, --smoke,
--train (cs2rl.train.loop.train), --record (cs2rl.train.record) or --eval
(cs2rl.train.evaluate). A package `__main__` runs under the name `__main__` and is
imported by no module, so the old train.py self-alias
(`sys.modules.setdefault("cs2rl.train", ...)`) has nothing left to protect.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from cs2rl.env.config import EnvConfig, RewardWeights
from cs2rl.policy import validate_aim_log_std_max
from cs2rl.spec.paths import CHECKPOINTS_DIR, RECORDINGS_DIR
from cs2rl.train.config import (
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_GAMMA,
    OPPONENT_MODES,
    assert_opponent_self_play_compatible,
    build_train_config,
    compute_batch_dims,
)
from cs2rl.train.envs import MAP_NAMES, build_map_data, env_seed_base, resolve_pin_pitch, smoke_test
from cs2rl.train.evaluate import evaluate_checkpoint
from cs2rl.train.loop import train
from cs2rl.train.record import record_episode

# ── SECTION: CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":

    # The env's own defaults, read from the dataclass that declares them, so the
    # CLI cannot drift from the env (spec 2026-09-03 R11). Bound once here rather
    # than per-flag: three flags below read it, and a second EnvConfig() would be
    # a second place to look when a default changes.
    _ENV_DEFAULTS = EnvConfig()

    # prog is spelled out: as a package __main__ argparse would print `usage: __main__.py`.
    parser = argparse.ArgumentParser(prog="python -m cs2rl.train")
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
    # R0-G env knobs. Default None ⇒ the env/nav.py constant (config.json
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
                        "training — pinned bitwise by tests/train/test_tag_trainer.py.")
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
    # from the LOADED map, never a name table). Cost: ~0.8 s (`from cs2rl.env import map`)
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
        # config.json needs its pin_pitch/env label). This lets the Modal
        # runner (scripts/modal_runner/preflight.py hashes the dumped
        # config.json) fingerprint the HPs cheaply (no env, no CUDA probe).
        # Keep this branch lean — anything imported here adds startup cost
        # to every launch.
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
