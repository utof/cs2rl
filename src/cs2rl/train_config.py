"""Config + CLI-resolution seam (#144), split out of train.py.

WHAT: ``build_train_config`` (the config.json / provenance dict), the
args-to-env-knob and args-to-mode resolvers, the batch-dimension formula, the
opponent-mode vocabulary and the static participating-rows vector. Moved here
VERBATIM by the 2026-08-31 post-rung1a refactor: no renames, no signature
changes, no behaviour change. ``train.py`` re-exports every name below (see its
``__all__``), so existing ``from cs2rl.train import X`` call sites keep working
unchanged.

WHY its own module: the Modal runner hashes the config.json that --dump-config
writes from build_train_config's dict (its config_hash), so this surface is
PROVENANCE, not plumbing. Silent drift here changes every future run's recorded
hash, which is much easier to review in one small file than inside the entry
point.

PITFALL: the argparse parser itself stays in train.py (it is built inline under
``if __name__ == "__main__"`` and is not importable). Several tests source-scan
train.py for ``add_argument`` literals and source-scan THIS file for
``OPPONENT_MODES`` / ``compute_batch_dims`` — the two halves of the CLI contract
now live in two files and both are pinned.

IMPORT-LIGHTNESS INVARIANT: module scope stays torch/nav/env.c-free, for the
reason spelled out in train_shared.py's header — ``--dump-config`` reaches
build_train_config and must still cost no torch/nav import.
"""
import math

import numpy as np

from cs2rl.env.config import REWARD_FIELDS, UNSET, EnvConfig, RewardWeights
from cs2rl.train_shared import (
    _R0G_KNOBS,
    AIM_LOG_STD_CAP_MIN_HEADROOM,
    AIM_LOG_STD_INIT_MARGIN,
    DEFAULT_CHECKPOINT_INTERVAL,
    LOG_STD_MAX,
    LOG_STD_MIN,
    TEAM_SIZE,
    resolve_aim_log_std_init,
    resolve_gammas,
)


def validate_aim_log_std_max(aim_log_std_max) -> float:
    """Resolve + range-check the run's aim σ cap (R0-E.3, #131).

    Returns the float cap (LOG_STD_MAX when None). Raises ValueError unless
    LOG_STD_MIN + 0.4 < cap <= LOG_STD_MAX, i.e. σ in (0.0149, 0.5].

    WHY a separate torch-free helper: build_policy() only runs after the env
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
    same contract as env_config_from_args' flag knobs, which read every one of
    them as ``getattr(args, name, UNSET)`` and omit the absent ones so the
    EnvConfig field default applies.

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
      call site, and Cs2PuffeRL._init_hybrid_aim re-checks the length.
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


def build_train_config(args, batch_size: int, bptt_horizon: int) -> dict:
    """Construct the train_config dict identically to the training path.

    Extracted so --dump-config can produce the exact same dict without
    spinning up an env. Any future changes to training HPs must live here,
    not duplicated in train(). Keep this semantically identical to what
    train() used to build inline — the Modal runner hashes the dumped dict
    (config_hash), so silent drift here changes run provenance.
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

    # ── Env config (spec 2026-09-03 §2.3) ──
    # ONE resolver for the weights, the flag knobs, the R0-G trio and pbrs_gamma;
    # the env factory reaches it through the wrappers below (PR B2 hands it the
    # EnvConfig itself), so provenance and the envs cannot disagree. Bound to
    # `env_cfg`, never `env_config`: that name reads as the MODULE, cs2rl.env.config,
    # this function's own import comes from.
    env_cfg = env_config_from_args(args)

    # ── TAG diagnostic (spec 2026-08-13 §4.1) ──
    # getattr fallbacks keep harness/dump-config args objects that predate
    # these flags working, same pattern as the warmstart block above.
    # NOTE: these keys change the Modal runner's config_hash for ALL future runs
    # (hash covers sorted config.json) — recorded decision, spec §4.1.
    tag_diagnostic = bool(getattr(args, "tag_diagnostic", False))
    tag_every = int(getattr(args, "tag_every", 5))

    # ── Batch 7 heads split (spec 2026-08-13 §2) ──
    # getattr fallback, same pattern as the TAG block above. This records the
    # FLAG, not the resolved architecture: a flag-less crash-resume of a split
    # run writes false here on purpose, which is precisely why the analyzer
    # reads the per-epoch split/active metric rather than config.json
    # (spec §3.4). Adding this key also changes the Modal runner's config_hash for all
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
    # Both numbers are recorded: done_training in Cs2PuffeRL.train
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
    # PITFALL: adding these keys changes the Modal runner's config_hash for all future
    # runs (the hash covers sorted config.json) — recorded decision, same as
    # the TAG/tct keys above, and the same for `opponent`, `jump_enabled` and
    # `aim_log_std_init` below (plus the σ weight-decay exclusion of T1, which
    # changes behaviour for every run without touching config.json at all).
    # (the raw budget is bound to a local, not inlined in the dict below, for
    # the same yapf reason as the warmstart block above: a long value
    # expression inside the dict re-indents every trailing comment in it.)
    n_active = env_cfg.n_active_per_team
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
    # (isolate_aim_log_std_param_group) change the Modal runner's config_hash for all
    # future runs — recorded decision, same convention as the TAG/tct keys
    # above. The weight-decay change is a real behaviour change for EVERY run,
    # not just capped ones; the init only moves when cap < LOG_STD_INIT + 0.2.
    aim_log_std_init = resolve_aim_log_std_init(aim_log_std_max)

    # R0-H: env LABEL from the resolved map name. The CLI always sets args.map
    # (above the --dump-config exit); the harness / older SimpleNamespace
    # callers have no `map` attr and keep the historical "cs2-dust2". Not
    # allowlisted for --resume-run: a different map is a different experiment.
    map_name = getattr(args, "map", None) or "dust2"
    # R0-J: --gamma / --pbrs-gamma. Same helper env_config_from_args ran for
    # env_cfg.pbrs_gamma, so the two discounts cannot resolve differently.
    gamma = resolve_gammas(args)[0]
    cfg = {
                                                                       # Core PPO
        "env": f"cs2-{map_name}",
        "device": args.device,
        "seed": args.seed,
        "total_timesteps": raw_timesteps,
        "participating_timesteps": args.timesteps,
                                                                       # Rung 1a T3: "self" (both teams learn) or "noop" (statue opponent —
                                                                       # hero-team-only participation AND budget, see raw_timesteps above).
                                                                       # NOT an EnvConfig knob: the statue is enforced trainer-side, in
                                                                       # the patched evaluate(), so the env is identical either way. Not
                                                                       # allowlisted for --resume-run — a different opponent is a different
                                                                       # experiment.
        "opponent": opponent,
                                                                       # R0-G: recorded as given (None ⇒ env default), read from args
                                                                       # directly under their CLI names — env_cfg renames and coerces.
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
                                                                       # Cs2PuffeRL.train (self._entropy_floor) clamps α ≥ 0.5 when H < 0.3·max;
                                                                       # a base target below the floor would make the two mechanisms fight.
        "entropy_target_warmup_frac": 0.5,
        "entropy_target_base_frac": 0.35,
        "entropy_target_warmup_steps": 10_000_000,
                                                                       # ── Warm-start entropy mode: see the comment above ──
        "warmstart_entropy": ws_entropy,
        "warmstart_grace_steps": ws_grace,
        "warmstart_ramp_steps": ws_ramp,
        "warmstart_alpha_ceiling": ws_alpha_ceil,
        "tag_diagnostic": tag_diagnostic,
        "tag_every": tag_every,
        "tct_split_heads": tct_split_heads,
        "tct_split_trunk": tct_split_trunk,
    }

    # ── Env provenance: 23 weights + 6 knobs, verbatim key names ──
    # EnvConfig.to_config_dict() is the single declaration of what config.json
    # records about the env (spec 2026-09-03 §2.1): the 23 weights plus
    # pbrs_gamma, reward_symmetrize, n_active_per_team, pin_pitch,
    # crouch_enabled and jump_enabled. The R0-G trio is NOT in it — those are
    # recorded above under their CLI names so a None survives as None.
    # None of these keys is in RESUME_CONFIG_ALLOWLIST: a run with different
    # weights, a different roster or a different PBRS discount is a different
    # experiment. Adding or removing one changes the Modal runner's config_hash for every
    # future run.
    # Merged via an explicit collision check rather than a trailing splat: a
    # splat in last position would SILENTLY overwrite an existing config key if
    # a weight were ever named like one of the keys above, and the result would
    # look perfectly well-formed. Key order does not matter — every config.json
    # dump uses sort_keys=True.
    env_keys = env_cfg.to_config_dict()
    assert not (cfg.keys() & env_keys.keys()), (
        "env config key collides with an existing config key: "
        f"{sorted(cfg.keys() & env_keys.keys())}")
    cfg.update(env_keys)
    return cfg


# Knobs whose CLI dest IS the field name. Hand-written on purpose, and paired
# with train_shared._R0G_KNOBS (which maps DIFFERENT names, e.g.
# --round-time-ticks → round_time): together they are the CLI-name ↔ field-name
# map, and tests/test_env_knobs.py::test_args_knob_coverage_is_exhaustive asserts
# the two cover every EnvConfig knob except pbrs_gamma (resolved through
# resolve_gammas) and recoil (no flag). Kept separate from _R0G_KNOBS so the
# R0-G "None is the stored value" rule is never applied to a flag knob.
_ARGS_KNOB_FIELDS = ("n_active_per_team", "pin_pitch", "crouch_enabled", "jump_enabled",
                     "reward_symmetrize")


def env_config_from_args(args) -> EnvConfig:
    """The single args → EnvConfig resolver (spec 2026-09-03 §2.3, Phase B R3).

    WHAT: reads the 23 reward flags, the five flag knobs, the R0-G trio and
    pbrs_gamma off `args` and returns one frozen EnvConfig — one resolver, one
    object. Until #165 Phase B the weights and the non-weight knobs were read by
    two separate helpers into two separate dicts that had to be kept in step by
    hand, and a knob added to one and missed by the other was silent; there is
    no second dict left for this one to drift from.

    WHY NO DEFAULT IS RESTATED HERE: every absent flag is reached by OMISSION —
    `getattr(args, name, UNSET)` and then simply not passing it — so the field
    default in env/config.py is the only declaration of the value. Spelling a
    fallback as `getattr(args, "<knob>", <the field default>)` instead would put
    a second copy of six defaults in this file, which is the duplication #165
    exists to remove and which tests/test_no_restated_env_defaults.py fails on.
    That probe reads PROSE as well as code, so this paragraph names no value
    either.

    PITFALL: the getattr fallbacks are load-bearing for harness / --dump-config
    args objects that predate these flags; do not tighten them to attribute
    access.

    R0-G (round_time / laser_range / max_turn_speed): read as `None` and STORED
    as None — None means "the env/nav.py constant", resolved inside Cs2Env, and
    config.json records None rather than a copied constant that would drift.

    R0-J: pbrs_gamma is ALWAYS resolved through resolve_gammas, never omitted —
    the field default (0.999) would silently disagree with a non-default --gamma.

    `recoil` is deliberately never read: there is no CLI flag, and inventing one
    here would be new behaviour (tests/test_recoil.py pins that).
    """
    weights = {}
    for name in REWARD_FIELDS:
        v = getattr(args, name, UNSET)
        if v is not UNSET:
            weights[name] = v
    knobs = {}
    # The historical `or 0` on pin_pitch is subsumed, not dropped: on the CLI
    # that flag stays None until train() resolves it from map flatness, and
    # EnvConfig.__post_init__ runs every flag knob through int(bool(...)), which
    # maps that None to the same value `or 0` produced.
    for name in _ARGS_KNOB_FIELDS:
        v = getattr(args, name, UNSET)
        if v is not UNSET:
            knobs[name] = v
    knobs["pbrs_gamma"] = resolve_gammas(args)[1]
    for arg_name, field_name in _R0G_KNOBS:
        knobs[field_name] = getattr(args, arg_name, None)
    return EnvConfig(rewards=RewardWeights(**weights), **knobs)
