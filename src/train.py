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
import random
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np

from _action_spec import (
    ACTION_HEAD_NAMES,
    ACTION_HEAD_SIZES,
    ACTION_MASK_DIM,                   # F8: trainer-side mask buffer width (= sum of head sizes)
    AIM_DIM,                           # noqa: F401  T4→T5 carry-forward (M-1): T6 ONNX exporter consumes this
)                                      # from cs2_types.h
from paths import CHECKPOINTS_DIR, RECORDINGS_DIR

# MUST stay a bare integer literal: scripts/exp_lib.py fingerprints the env by
# regex-grepping `OBS_DIM = <int>` out of this file's source text (env_fingerprint),
# so it cannot be an `import`. Mirrors nav.OBS_DIM / _obs_spec.OBS_DIM (generated
# from cs2_types.h); the three are cross-checked by tests/test_train_env.py:873.
# On an OBS_DIM bump, update cs2_types.h + rerun the generator, then bump this literal.
OBS_DIM = 110

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


def reinit_frozen_aim_log_std(state_dict, *, atol=1e-6):
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
    """
    import torch as _torch

    changed = False
    for key, val in state_dict.items():
        if key.endswith("aim_log_std") and _torch.allclose(
                val, _torch.full_like(val, LOG_STD_INIT), atol=atol):
            state_dict[key] = _torch.full_like(val, AIM_LOG_STD_RESUME_INIT)
            changed = True
    return changed


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


def compute_batch_dims(num_envs: int) -> tuple[int, int, int]:
    """Return (agents_per_env, bptt_horizon, batch_size) used by both training
    and --dump-config. Single source of truth so the fingerprint dict captured
    pre-training cannot drift from what train() actually runs.
    """
    agents_per_env = 10
    bptt_horizon = 64
    batch_size = num_envs * agents_per_env * bptt_horizon
    return agents_per_env, bptt_horizon, batch_size


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
    # 1. Both *_steps are trainer.global_step units — agent steps, the same
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

    return {
                                                       # Core PPO
        "env": "cs2-dust2",
        "device": args.device,
        "seed": args.seed,
        "total_timesteps": args.timesteps,
        "batch_size": batch_size,
        "bptt_horizon": bptt_horizon,
        "minibatch_size": 8192,
        "max_minibatch_size": 8192,
        "update_epochs": 3,
        "learning_rate": 3e-4,
        "gamma": 0.999,
        "gae_lambda": 0.95,
        "clip_coef": 0.15,
        "vf_coef": 0.5,
        "vf_clip_coef": None,
        "ent_coef": 0.1,                               # fallback; adaptive alpha overrides
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
        "checkpoint_interval": 200,
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
    }


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
                    pbrs_gamma=None):
    """Create the native C PufferEnv used by smoke/train/eval.

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
    """
    from c_env.cs2_env import make_env as make_c_env

    if record_fn is not None:
        raise ValueError("record_fn is only supported by the Python recording env")
    kwargs = {}
    if pbrs_gamma is not None:
        kwargs["pbrs_gamma"] = pbrs_gamma
    return make_c_env(
        seed=seed,
        team_spirit=team_spirit,
        buf=buf,
        map_data=map_data,
        include_step_stats_in_info=include_step_stats_in_info,
        **kwargs,
    )


def load_policy_from_checkpoint(checkpoint_path, device):
    import torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Policy] Loading checkpoint -> {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Infer obs_dim from checkpoint to handle checkpoints trained with different obs sizes
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
        policy = build_policy(policy_env, device, obs_dim_override=ckpt_obs_dim)
    finally:
        policy_env.close()

    policy.load_state_dict(state_dict)
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
    from nav import ROUND_TIME

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
        while not done and step_count < ROUND_TIME * 2:
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


# ── SECTION: Policy ────────────────────────────────────────────────────────


def build_policy(vecenv, device, obs_dim_override=None):
    import pufferlib.pytorch
    import torch
    import torch.nn as nn

    driver_env = getattr(vecenv, "driver_env", vecenv)
    obs_dim = (obs_dim_override
               if obs_dim_override is not None else driver_env.single_observation_space.shape[0])
    hidden = 256

    class Dust2Policy(nn.Module):

        def __init__(self):
            super().__init__()
            self.hidden_size = hidden  # required by PufferLib LSTM logic
            self.obs_dim = obs_dim

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

            # Separate heads for MultiDiscrete(ACTION_HEAD_SIZES)
            self.action_heads = nn.ModuleList([
                pufferlib.pytorch.layer_init(nn.Linear(hidden, n), std=0.01)
                for n in ACTION_HEAD_SIZES
            ])
            self.value_head = pufferlib.pytorch.layer_init(nn.Linear(hidden, 1), std=1.0)

            # Batch 3: continuous Gaussian aim head.
            # mu_aim → (B, AIM_DIM); tanh-squashed and scaled by max_turn_speed
            #   in forward(). State-DEPENDENT (per-step linear projection) so
            #   the policy can react to the current obs (visible enemies, yaw
            #   delta to target, etc.) when picking the mean Δyaw.
            # aim_log_std → (AIM_DIM,) — state-INDEPENDENT learnable parameter
            #   per Fan et al. IJCAI 2019 H-PPO baseline. Clamped in forward()
            #   to [LOG_STD_MIN, LOG_STD_MAX] so neither σ collapse (entropy
            #   loss → −∞) nor explosion (σ floods policy) is reachable.
            # Pitfall: keep `std=0.01` on aim_mu init so the pre-tanh mean
            #   starts ~zero — otherwise the policy starts saturated and
            #   learning the Gaussian head is much slower.
            self.aim_mu = pufferlib.pytorch.layer_init(nn.Linear(hidden, AIM_DIM), std=0.01)
            self.aim_log_std = nn.Parameter(torch.full((AIM_DIM, ), LOG_STD_INIT))

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
            mu_aim = torch.tanh(self.aim_mu(hidden_out)) * self.max_turn_speed
            log_std = torch.clamp(self.aim_log_std, LOG_STD_MIN, LOG_STD_MAX)
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
            log_prob_c = aim_dist.log_prob(continuous_action).sum(-1)
            # Closed-form Gaussian entropy: 0.5·log(2πe·σ²), summed across
            # AIM_DIM. .entropy() returns per-dim, so .sum(-1) is correct
            # for AIM_DIM=1 today and stays correct if AIM_DIM bumps to ≥2.
            entropy_c = aim_dist.entropy().sum(-1)

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

            logits = [head(hidden_out) for head in self.action_heads]
            value = self.value_head(hidden_out)
            # μ is bounded by tanh*max_turn_speed; log_std broadcasts to μ
            # shape so callers can build Normal(mu, exp(log_std)) directly
            # without an extra .expand call.
            mu_aim = torch.tanh(self.aim_mu(hidden_out)) * self.max_turn_speed
            log_std = torch.clamp(self.aim_log_std, LOG_STD_MIN, LOG_STD_MAX).expand_as(mu_aim)
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

            h = self.encoder(x.reshape(B * TT, x.shape[-1]).float())
            h = h.reshape(B, TT, self.hidden_size).transpose(0, 1)     # (T, B, H) seq-first

            lstm_h = state.get("lstm_h") if isinstance(state, dict) else None
            lstm_c = state.get("lstm_c") if isinstance(state, dict) else None
            if lstm_h is not None and lstm_c is not None:
                hc = (lstm_h.reshape(1, B,
                                     self.hidden_size), lstm_c.reshape(1, B, self.hidden_size))
            else:
                hc = (h.new_zeros(1, B, self.hidden_size), h.new_zeros(1, B, self.hidden_size))

            terminals = state.get("terminals") if isinstance(state, dict) else None
            h = self._lstm_bptt(h, hc, terminals)
            # transpose back to (B, T, H) then flatten row-major so flat row
            # b*T + t lines up with mb_actions.reshape(-1, ...) in
            # _hybrid_ppo_loss — segment-major, time-minor. Changing this
            # ordering silently misaligns every logprob/advantage pairing.
            hidden_out = h.transpose(0, 1).reshape(B * TT, self.hidden_size)

            logits = [head(hidden_out) for head in self.action_heads]
            value = self.value_head(hidden_out)
            mu_aim = torch.tanh(self.aim_mu(hidden_out)) * self.max_turn_speed
            log_std = torch.clamp(self.aim_log_std, LOG_STD_MIN, LOG_STD_MAX).expand_as(mu_aim)
            return logits, mu_aim, log_std, value

        def _lstm_bptt(self, h_seq, hc, terminals):
            """Run the LSTM over a full (T, B, H) segment with done-masking.

            WHAT: one nn.LSTM call when the segment contains no episode
            boundaries (the common case — native PufferLib BPTT); otherwise
            the sequence is split at every tick where ANY row has a done and
            h/c are zero-masked per-row at those ticks before continuing.

            WHY: the rollout (forward_eval → _forward_core) multiplies the
            carried state by (1 - done) BEFORE processing each tick, so a
            new episode starts memory-free. Training must replicate that
            reset or the recomputed logprobs at post-done ticks come from a
            different function than the rollout stored (biased PPO ratios)
            and gradients leak across episode boundaries. Upstream
            LSTMWrapper skips this (it never resets on done, rollout OR
            train, so it is self-consistent); we reset in rollout, hence we
            must also reset here.

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
            """
            if terminals is None:
                out, _ = self.lstm(h_seq, hc)
                return out
            TT, B, _H = h_seq.shape
            term = terminals.reshape(B, TT) > 0.5
            reset_ticks = torch.nonzero(term.any(dim=0)).flatten().tolist()
            if not reset_ticks:
                out, _ = self.lstm(h_seq, hc)
                return out
            outs = []
            h0, c0 = hc
            t0 = 0
            for t in reset_ticks:
                if t > t0:
                    out, (h0, c0) = self.lstm(h_seq[t0:t], (h0, c0))
                    outs.append(out)
                keep = (~term[:, t]).float().view(1, B, 1)
                h0 = h0 * keep
                c0 = c0 * keep
                t0 = t
            out, _ = self.lstm(h_seq[t0:], (h0, c0))
            outs.append(out)
            return torch.cat(outs, dim=0)

        def _forward_core(self, x, lstm_state, done):
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
    max_entropy_discrete = sum(np.log(n) for n in ACTION_HEAD_SIZES)
    max_entropy_continuous = AIM_DIM * 0.5 * np.log(2 * np.pi * np.e * np.exp(LOG_STD_MAX)**2)
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

    def _normalize_returns(mb_returns):
        """Return normalized copy of mb_returns; update running stats first."""
        _update_return_stats(mb_returns.detach().flatten())
        std = (_ret_var + 1e-8).sqrt()
        return (mb_returns - _ret_mean) / std

    def _train_with_return_norm(self):
        profile = self.profile
        epoch = self.epoch
        profile("train", epoch)
        losses = defaultdict(float)
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
        # minibatch loop. Reported to the log layer as event_oversample_fraction.
        _t8_event_mask = getattr(self, "_batch1_event_mask", None)
        self._batch1_event_oversample_fraction = (float(_t8_event_mask.float().mean())
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
            #     NOT the post-boost sampled fraction — that's what the
            #     wandb/log layer reports as `event_oversample_fraction`.
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

            # ── VALUE TARGET NORMALISATION ─────────────────────────────────
            # Normalize returns before value regression.  The value head learns
            # to predict normalized targets; advantages are unaffected.
            mb_returns_norm = _normalize_returns(mb_returns)
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
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config["clip_coef"]).float().mean()

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
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * v_loss_unclipped.mean()

            current_entropy = entropy.mean()

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

            entropy_loss = -effective_alpha * current_entropy
            # ──────────────────────────────────────────────────────────────

            loss = pg_loss + config["vf_coef"] * v_loss + entropy_loss
            self.amp_context.__enter__()

            # Denormalize before writing back so advantage computation stays in raw scale
            std = (_ret_var + 1e-8).sqrt()
            self.values[idx] = newvalue.detach().float() * std + _ret_mean

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
                    losses[f"entropy/{_hn}"] += _hd.entropy().mean().item()
            losses["entropy/total"] += current_entropy.item()
            # ──────────────────────────────────────────────────────────────

            # Logging
            profile("train_misc", epoch)
            losses["policy_loss"] += pg_loss.item()
            losses["value_loss"] += v_loss.item()
            losses["entropy"] += current_entropy.item()
            losses["alpha"] += alpha.detach().item()
            losses["alpha_loss"] += alpha_loss.item()
            losses["old_approx_kl"] += old_approx_kl.item()
            losses["approx_kl"] += approx_kl.item()
            losses["clipfrac"] += clipfrac.item()
            losses["importance"] += ratio.mean().item()
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

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
        losses["explained_variance"] = explained_var.item()
        losses["ret_mean"] = _ret_mean.item()
        losses["ret_std"] = (_ret_var + 1e-8).sqrt().item()
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
        done_training = self.global_step >= config["total_timesteps"]
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
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


# ── SECTION: Game Metrics Dashboard ───────────────────────────────────────


def compute_game_metrics(logs):
    """Extract and normalize game metrics from the training logs dict.

    The C env exposes per-episode stats as ``environment/<key>`` entries in
    the logs dict returned by PufferLib's ``mean_and_log()``.  Values are
    already averaged over the collection window, so most just need re-keying
    and minor arithmetic.

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

    # actions/use_at_site_frac — logged directly by the C env if available
    use_at_site = _get("use_at_site_frac", None)
    if use_at_site is not None:
        game_metrics["actions/use_at_site_frac"] = use_at_site

    return game_metrics


# ── SECTION: Dead Run Detector ─────────────────────────────────────────────


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

    def __init__(self, check_interval=10_000):
        self.check_interval = check_interval
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
            entropy_total = metrics.get("entropy/total", metrics.get("entropy", 5.0))
            if entropy_total < 0.5:
                self.alerts.append(
                    f"CRITICAL: Entropy collapsed to {entropy_total:.2f} at step {step}")
            timeout_rate = metrics.get("game/timeout_rate", 0.0)
            if timeout_rate > 0.95:
                self.alerts.append(f"WARNING: Timeout rate {timeout_rate:.0%} at step {step}")

        if step > 500_000:
            kills_per_ep = metrics.get("game/kills_per_episode", 1.0)
            if kills_per_ep == 0:
                self.alerts.append(f"WARNING: Zero kills by step {step}")

        if step > 100_000:
            approx_kl = metrics.get("approx_kl", 0.0)
            if approx_kl > 0.05:
                self.alerts.append(f"WARNING: KL divergence {approx_kl:.3f} at step {step}")

        if len(self.alerts) >= 5:
            print("DEAD RUN DETECTED:")
            for alert in self.alerts:
                print(f"  {alert}")
            return True

        return False


# ── SECTION: Self-Play ─────────────────────────────────────────────────────


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

    def __init__(
        self,
        pool_size: int = 15,
        p_past: float = 0.3,
        save_every_epochs: int = 25,
        win_threshold: float = 0.6,
        phase_length: int = 50,
    ):
        self.pool: list[Path] = []
        self.pool_size = pool_size
        self.p_past = p_past
        self.save_every_epochs = save_every_epochs
        self.win_threshold = win_threshold
        self.phase_length = phase_length
        self.opponent_team = "ct"      # CT is opponent first; T learns to attack
        self._milestone_count = 0
        self._last_save_epoch = -1

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
        """Load a random past checkpoint. Returns the policy module or None."""
        import torch

        if not self.pool:
            return None
        path = random.choice(self.pool)
        if not path.exists():
            self.pool.remove(path)
            return None
        policy = build_policy(vecenv, device)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        policy.load_state_dict(state_dict)
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
    """
    import pufferlib
    import pufferlib.pytorch
    import torch

    # Task 6c (utof/cs2rl#9): Batch 1 reward-architecture helpers.
    # Imported lazily here (not at module scope) to keep train.py import
    # cost flat for callers that never hit the self-play path.
    from train_helpers_batch1 import WelfordStd, split_into_channels, symlog

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
            self.global_step += int(mask.sum())

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
                agents_per_env_local = self.vecenv.driver_env.num_agents
                r_new = torch.empty_like(r)
                for e in range(len(info)):
                    row_start = e * agents_per_env_local
                    row_end = row_start + agents_per_env_local
                    ss = info[e].get("step_stats", None) if isinstance(info[e], dict) else None
                    if ss is None:
                        # No per-tick step_stats: leave this env's rows as-is.
                        r_new[row_start:row_end] = r[row_start:row_end]
                        continue
                    channels = split_into_channels(ss)
                    # Welford.update takes scalar floats (one observation per env/tick).
                    self._batch1_welford_combat.update(channels["combat"])
                    self._batch1_welford_objective.update(channels["objective"])
                    self._batch1_welford_positional.update(channels["positional"])
                    # Task 7: OR the per-tick bomb_planted flag into the live
                    # event accumulator for every agent row in this env. The
                    # C side sets ss->bomb_planted only on the transition tick
                    # (cs2_bomb.h:60 — guarded by `if g->bomb_plant_ticks >=
                    # sd->bomb_plant_time`) and StepStats is cleared every step
                    # via clear_stats(ss) at the top of cs2_env.h:75. So
                    # step_stats['bomb_planted'] is already a per-tick delta
                    # (1 only on the plant tick) — NO edge-trigger needed.
                    # All 10 agent rows in an env share the same event state:
                    # if the bomb plants this tick, every row's current segment
                    # now contains an event. Flushed to _batch1_event_mask at
                    # the segment boundary below (see ~30 lines down).
                    # bool(int(...)) is deliberate: stubs or numpy scalars may
                    # not truthy-coerce cleanly; int() normalises to a Python
                    # int first so bool() is guaranteed. Do not strip the cast.
                    if bool(int(ss.get("bomb_planted", 0))):
                        self._batch1_current_segment_has_event[row_start:row_end] = True
                    # Normalize per-channel (divide by running std), sum, compress.
                    # Build the scalar sum on CPU (cheap — 3 floats) then broadcast
                    # to the 10-agent slice; avoids per-agent torch.tensor churn.
                    combat_t = torch.tensor(channels["combat"], device=_dev, dtype=r.dtype)
                    objective_t = torch.tensor(channels["objective"], device=_dev, dtype=r.dtype)
                    positional_t = torch.tensor(channels["positional"], device=_dev, dtype=r.dtype)
                    r_sum = (self._batch1_welford_combat.normalize(combat_t) +
                             self._batch1_welford_objective.normalize(objective_t) +
                             self._batch1_welford_positional.normalize(positional_t))
                    r_new[row_start:row_end] = symlog(r_sum)
                # If info was shorter than the batch (e.g. some envs didn't
                # emit info this tick), copy through any remaining raw rows.
                used = len(info) * agents_per_env_local
                if used < r.shape[0]:
                    r_new[used:] = r[used:]
                r = r_new

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
                self.values[batch_rows, seq_pos] = value.flatten()

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
                          mask=None):
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
    log_prob_c = (-0.5 * diff * diff - log_std_b - 0.5 * _LOG_2PI).sum(-1)
    entropy_c = (0.5 + 0.5 * _LOG_2PI + log_std_b).sum(-1)

    return action, continuous_action, log_prob_d, log_prob_c, entropy_d, entropy_c


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
                     mb_masks=None):
    """Per-factor PPO clipped loss (H-PPO, Fan et al. IJCAI 2019).

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
    flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
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
    new_logp_c = (-0.5 * diff * diff - log_std_b - 0.5 * _LOG_2PI).sum(-1)
    entropy_c = (0.5 + 0.5 * _LOG_2PI + log_std_b).sum(-1)
    entropy = entropy_d + entropy_c

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
    pg_loss = torch.max(pg_d_un, pg_d_cl).mean() + torch.max(pg_c_un, pg_c_cl).mean()

    return pg_loss, entropy, new_value, new_logp_d + new_logp_c, ratio_d, ratio_c, logits_list


def _patch_trainer_with_hybrid_aim(trainer, cont_action_view_main=None, mask_view_main=None):
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
    run_id = f"{run_label}-{time.strftime('%Y%m%d-%H%M%S')}"

    # Shared team spirit value — all envs read it at episode start
    shared_ts = mp.Value("f", 0.3)

    _map_data = args.map_data

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

    def env_factory(*_args,
                    buf=None,
                    seed=None,
                    _cont_shm=None,
                    _cont_idx=None,
                    _mask_shm=None,
                    **kwargs):
        env = make_puffer_env(team_spirit=shared_ts, buf=buf, seed=seed or 0, map_data=_map_data)
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

    # Per-env kwargs list — pufferlib.vector.make accepts a list of dicts
    # (one per env). All args propagate verbatim through fork because
    # they're stored on env_kwargs[i] BEFORE Process.start() (see
    # .venv/lib/.../pufferlib/vector.py:333-346).
    _per_env_kwargs = [{
        "_cont_shm": _cont_action_shm,
        "_cont_idx": i,
        "_mask_shm": _mask_shm,
    } for i in range(args.num_envs)]

    backend_name = args.vec_backend.lower()
    if backend_name == "multiprocessing":
        import psutil

        backend = pufferlib.vector.Multiprocessing
        physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        num_workers = args.vec_num_workers or min(args.num_envs, physical_cores)
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

    print(f"[Train] Building policy on device={device}...")
    policy = build_policy(vecenv, device)

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
    resume_path = getattr(args, "resume", None)
    if resume_path:
        import torch as _torch

        resume_path = Path(resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        state_dict = _torch.load(resume_path, map_location=device, weights_only=True)
        # gh#91: BC warm-start checkpoints carry aim_log_std frozen at
        # LOG_STD_INIT — widen to AIM_LOG_STD_RESUME_INIT before loading or
        # the KL early-stop throttles the whole run (see the helper's doc).
        if reinit_frozen_aim_log_std(state_dict):
            print(f"[Train] BC-frozen aim_log_std detected in {resume_path.name}: "
                  f"re-initialized to log(0.3) ≈ {AIM_LOG_STD_RESUME_INIT:.3f} (gh#91)")
        policy.load_state_dict(state_dict)
        print(f"[Train] Resumed from checkpoint: {resume_path}")
    # ────────────────────────────────────────────────────────────────────────

    if train_config.get("warmstart_entropy") and not resume_path:
        print("[Train] WARN: --warmstart-entropy without --resume — the grace window "
              "will suppress entropy pressure on a from-scratch policy (legal, but "
              "probably not what you want).")

    trainer = PuffeRL(train_config, vecenv, policy)
    trainer.optimizer.param_groups[0]["weight_decay"] = 1e-4
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
    trainer._action_mask_shm = _mask_shm                                         # F8: same GC-pinning rationale
    _patch_trainer_with_hybrid_aim(trainer,
                                   cont_action_view_main=_cont_action_view_main,
                                   mask_view_main=_mask_view_main)

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
        save_every_epochs=25,                          # ~2M steps per save at batch_size=81920
        win_threshold=0.6,
        phase_length=50,                               # switch opponent team every ~4M steps
    )
    if self_play_enabled and resume_path and resume_path.exists():
        import shutil as _shutil

        seed_path = Path(args.checkpoint_dir) / "sp_seed.pt"
        _shutil.copy2(resume_path, seed_path)
        self_play_mgr._add_to_pool(seed_path)
        print(f"[SelfPlay] Pool pre-seeded with resume checkpoint ({seed_path.name})")
    _patch_trainer_with_selfplay(trainer, self_play_mgr)
    if not self_play_enabled:
        print("[Train] Self-play mixing disabled (--no-self-play): "
              "both teams use the current policy every epoch.")
    # timing is the outermost wrapper so it sees all evaluate() calls regardless of selfplay
    _patch_trainer_with_timing(trainer)
    # ────────────────────────────────────────────────────────────────────────

    save_path = Path(args.checkpoint_dir) / "dust2_policy.pt"
    last_save = time.time()

    dead_run_detector = DeadRunDetector()

    print(f"[Train] Starting PufferLib PPO for {args.timesteps:,} env steps...")
    while trainer.epoch < trainer.total_epochs:
        trainer.evaluate()
        logs = trainer.train()

        # Team spirit annealing: 0.3→0.7 over 5M steps
        ts_val = min(0.7, 0.3 + trainer.global_step / 5_000_000)
        shared_ts.value = ts_val

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
                win_rate_t = logs.get("environment/winner_t", 0.0)
                win_rate_ct = logs.get("environment/winner_ct", 0.0)
                self_play_mgr.maybe_save(
                    policy,
                    Path(args.checkpoint_dir),
                    trainer.epoch,
                    win_rate_t,
                    win_rate_ct,
                )
                logs["self_play/pool_size"] = float(len(self_play_mgr.pool))
                # opponent_team flag: 1.0 = CT opponent, 0.0 = T opponent.
                logs["self_play/opponent_team"] = float(self_play_mgr.opponent_team == "ct")
                # ────────────────────────────────────────────────────────────

            # Batch 3.5 (#24): per-axis aim log_std metrics. Read CLAMPED values
            # (the values the policy actually used at this iteration), not the raw
            # nn.Parameter. LOG_STD_MIN/MAX are module-globals at lines 49-50.
            # Load-bearing for T7 acceptance gate 2: aim_log_std_pitch > -3.5
            # at 30M steps. Format string in format_train_status must keep the
            # 'aim_log_std_pitch=' substring greppable.
            with torch.no_grad():
                clamped_log_std = torch.clamp(policy.aim_log_std, LOG_STD_MIN,
                                              LOG_STD_MAX).cpu().numpy()
            logs["policy/aim_log_std_yaw"] = float(clamped_log_std[0])
            logs["policy/aim_log_std_pitch"] = float(clamped_log_std[1])

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
            _metrics_file.write(json.dumps(log_entry) + "\n")
            _metrics_file.flush()
            if wandb_run is not None:
                wandb_run.log(log_entry, step=trainer.global_step)

        if time.time() - last_save > args.save_every_sec:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), save_path)
            last_save = time.time()
            print(f"Saved checkpoint to {save_path}")

        if isinstance(logs, dict):
            if trainer.epoch % 10 == 0:
                print(format_train_status(trainer.epoch, ts_val, logs))
            print(f"[Timing] collect={trainer._timing['collect_ms']:.0f}ms  "
                  f"update={trainer._timing['update_ms']:.0f}ms  "
                  f"SPS={logs.get('SPS', 0):.0f}")

    trainer.close()

    # Final checkpoint save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), save_path)
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
        help="Use the full dust2 map instead of the default simple 5-room map",
    )
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
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_every_sec", type=int, default=300)
    parser.add_argument(
        "--checkpoint_dir",
        "--checkpoint-dir",
        type=str,
        default=str(CHECKPOINTS_DIR),
        dest="checkpoint_dir",
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
    args = parser.parse_args()

    if args.dump_config:
        # Zero-side-effect mode: write config.json and exit. Runs BEFORE device
        # detection and map loading so no torch/map imports are triggered. This
        # lets scripts/run_experiment.py fingerprint the HPs cheaply (no env,
        # no CUDA probe). Keep this branch lean — anything imported here adds
        # startup cost to every experiment launch.
        if args.device is None:
            args.device = "cpu"        # placeholder; never used for training

        ckpt_dir = Path(args.checkpoint_dir)
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

    if args.dust2:
        args.map_data = None
        print("[Map] Using dust2 map")
    else:
        from map import make_simple_map

        args.map_data = make_simple_map()
        print("[Map] Using simple 5-room map")

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
