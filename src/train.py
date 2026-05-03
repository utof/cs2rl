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
    AIM_DIM,                           # noqa: F401  T4→T5 carry-forward (M-1): T6 ONNX exporter consumes this
)                                      # from cs2_types.h
from paths import CHECKPOINTS_DIR, RECORDINGS_DIR

OBS_DIM = 107                          # Batch 3.5 (#24): mirrors nav.OBS_DIM; tests cross-check the two via tests/test_train_env.py:873.

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
# Fix #2: precomputed log(2π) for the analytic Normal log-prob/entropy
# replacing torch.distributions.Normal in _hybrid_sample_logits.
_LOG_2PI = math.log(2.0 * math.pi)


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
    }


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
                    include_step_stats_in_info=False):
    """Create the native C PufferEnv used by smoke/train/eval.

    ``include_step_stats_in_info`` (Task 6a, utof/cs2rl#7): when True the env
    emits ``info = [{"step_stats": StepStatsView}]`` on every tick so trainer
    patches (Task 6c onward) can read per-channel raw reward fields. Defaults
    to False so production code paths that don't consume step_stats (e.g. eval
    scripts, viz) stay zero-cost.
    """
    from c_env.cs2_env import make_env as make_c_env

    if record_fn is not None:
        raise ValueError("record_fn is only supported by the Python recording env")
    return make_c_env(
        seed=seed,
        team_spirit=team_spirit,
        buf=buf,
        map_data=map_data,
        include_step_stats_in_info=include_step_stats_in_info,
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
    import torch

    if policy_mode == "random":
        raise ValueError("Random action selection should bypass select_policy_actions_native")

    obs_t = torch.as_tensor(obs, device=device)
    if hasattr(policy, "obs_dim") and obs_t.shape[-1] != policy.obs_dim:
        obs_t = obs_t[..., :policy.obs_dim]
    with torch.no_grad():
        # Batch 3 (T5): same change as select_policy_actions above. This native
        # helper feeds evaluate_checkpoint and the smoke path; both consume only
        # discrete int32 actions today. cont_t is dropped on the floor; if Δyaw
        # is wanted in eval recordings, plumb it through here in T6 alongside
        # the ONNX export wiring. See sibling helper for greedy-vs-sample notes.
        logits, mu_aim, log_std_aim, _ = policy.forward_eval(obs_t, policy_state)
        if policy_mode == "sample":
            # Fix #1: 6-tuple return; only need action + cont (logp/entropy unused here).
            act_t, _cont_t, *_ = _hybrid_sample_logits(
                (logits, mu_aim, log_std_aim, None),
                max_turn_speed=policy.max_turn_speed.item(),
            )
        else:
            act_t = torch.stack([head.argmax(dim=-1) for head in logits], dim=-1)

    return act_t.cpu().numpy().astype(np.int32)


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
    return (f"Epoch {epoch} | SPS: {sps:.0f} | Timeout: {timeout:.3f} | "
            f"TWin: {t_win:.3f} | CTWin: {ct_win:.3f} | Plant: {plant:.3f} | "
            f"Kills(T/CT): {kills_t:.2f}/{kills_ct:.2f} | RoundLen: {round_len:.1f} | "
            f"Move1: {move_1:.1f} | TS: {ts_val:.3f}")


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
        else:
            actions = select_policy_actions_native(policy, obs, device, policy_state, policy_mode)

        obs, rewards, terms, truncs, infos = env.step(actions)
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
            else:
                actions = select_policy_actions_native(policy, obs, device, policy_state,
                                                       policy_mode)

            for action in actions:
                for head_idx, action_value in enumerate(action):
                    action_hist[head_idx][int(action_value)] += 1
                joint_hist[tuple(int(v) for v in action)] += 1

            obs, rewards, terms, truncs, infos = env.step(actions)
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


# ── SECTION: Training config ───────────────────────────────────────────────

TRAINING_CONFIG = dict(
    gamma=0.99,                        # used by PBRS shaping in sim.py and test_reward.py
)

# ── SECTION: PufferLib env factory ─────────────────────────────────────────


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
                continuous_action: (B, AIM_DIM=1) float32 — Δyaw in radians
                    already in [-max_turn_speed, +max_turn_speed]; if None,
                    sample from the Normal head.

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
            # Batch 3: same 4-tuple contract as forward_eval. forward() is
            # the path PufferLib's vectorised rollout uses (no LSTM state
            # carry) — it stays in lockstep with forward_eval to keep the
            # ONNX export single-pathway in task 6.
            if x.ndim == 3:
                x_flat = x.reshape(-1, x.shape[-1])
            else:
                x_flat = x

            hidden_out, _ = self._forward_core(x_flat, None, None)
            logits = [head(hidden_out) for head in self.action_heads]
            value = self.value_head(hidden_out)
            mu_aim = torch.tanh(self.aim_mu(hidden_out)) * self.max_turn_speed
            log_std = torch.clamp(self.aim_log_std, LOG_STD_MIN, LOG_STD_MAX).expand_as(mu_aim)
            return logits, mu_aim, log_std, value

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
    import time
    import types
    from collections import defaultdict

    import torch
    from pufferlib.pufferl import compute_puff_advantage

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
    trainer._batch1_current_target_entropy = 0.7 * float(max_entropy)
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
        # PITFALL: do NOT capture max_entropy from the outer closure here —
        # use trainer._batch1_max_entropy. Closure capture would silently break
        # if the patch were re-applied on the same trainer instance.
        from train_helpers_batch1 import target_entropy_schedule
        _t9_target_entropy = target_entropy_schedule(self.global_step,
                                                     trainer._batch1_max_entropy,
                                                     warmup_end=10_000_000)
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

        # Task 8: raw event-segment fraction (mask mean) — computed once per
        # train() call because _batch1_event_mask doesn't change inside the
        # minibatch loop. Reported to the log layer as event_oversample_fraction.
        _t8_event_mask = getattr(self, "_batch1_event_mask", None)
        self._batch1_event_oversample_fraction = (float(_t8_event_mask.float().mean())
                                                  if _t8_event_mask is not None else 0.0)

        for mb in range(self.total_minibatches):
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
            #   importance-sampling correction below uses the BOOSTED
            #   prio_probs[idx], so the gradient stays unbiased.
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
            mb_rewards = self.rewards[idx]
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

            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
            )

            # Batch 3 (T5): hybrid PPO update — per-factor clipped loss
            # (Fan et al. IJCAI 2019). The helper does the policy forward
            # pass (returning mu_aim/log_std + value) and assembles the
            # clipped policy loss with INDEPENDENT discrete and continuous
            # ratios. We still also need the raw logits for the per-head
            # entropy diagnostics below, so re-pull them via a NO-grad path
            # — _hybrid_ppo_loss already consumed them on the gradient path.
            pg_loss, entropy, newvalue, newlogprob, ratio_d, ratio_c = _hybrid_ppo_loss(
                self.policy,
                mb_obs,
                mb_actions,
                mb_cont_actions,
                mb_old_logp_d,
                mb_old_logp_c,
                mb_advantages,
                clip_coef,
                state,
            )
            with torch.no_grad():
                # Logits-only path for the per-head entropy diagnostic block
                # below. Cheaper than re-running _hybrid_ppo_loss; we already
                # have the loss + entropy from the gradient pass.
                logits, _mu_diag, _log_std_diag, _ = self.policy(mb_obs, state)
            # NOTE: pre-Batch-3 the inline `actions = ...` from sample_logits
            # was used by downstream diagnostics; T5 dropped that consumer
            # (mb_actions is the canonical stored discrete action). No
            # rebinding here — the variable is unused after this point.

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

            # Early stopping: stop update if KL divergence exceeds target
            target_kl = config.get("target_kl", None)
            if target_kl is not None and approx_kl.item() > target_kl:
                break

            adv = advantages[idx]
            adv = compute_puff_advantage(
                mb_values,
                mb_rewards,
                mb_terminals,
                ratio,
                adv,
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )
            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            # Batch 3 (T5): pg_loss already computed by _hybrid_ppo_loss above
            # via per-factor clipping. The pre-Batch-3 single-ratio block lived
            # here; it would over-clip (ratios from a Gaussian factor can vary
            # very differently from categorical factors), so the spec L8
            # decision is to clip per factor and sum. Keeping a no-op stub to
            # make the diff easier to read and to flag where the change lives
            # for future archaeologists.
            _ = adv                    # adv computed above for vtrace; pg_loss already set

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
            alpha_loss = (log_alpha * (current_entropy - _t9_target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            # Entropy floor: prevent collapse
            effective_alpha = alpha.detach()
            if current_entropy.item() < entropy_floor:
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
                    losses[f"entropy/{_hn}"] += _hd.entropy().mean().item() / self.total_minibatches
            losses["entropy/total"] += current_entropy.item() / self.total_minibatches
            # ──────────────────────────────────────────────────────────────

            # Logging
            profile("train_misc", epoch)
            losses["policy_loss"] += pg_loss.item() / self.total_minibatches
            losses["value_loss"] += v_loss.item() / self.total_minibatches
            losses["entropy"] += current_entropy.item() / self.total_minibatches
            losses["alpha"] += alpha.detach().item() / self.total_minibatches
            losses["alpha_loss"] += alpha_loss.item() / self.total_minibatches
            losses["old_approx_kl"] += old_approx_kl.item() / self.total_minibatches
            losses["approx_kl"] += approx_kl.item() / self.total_minibatches
            losses["clipfrac"] += clipfrac.item() / self.total_minibatches
            losses["importance"] += ratio.mean().item() / self.total_minibatches

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
        # reading it here works in the happy path. The real risk is the
        # `target_kl` early-break path inside the loop: if minibatch 0
        # exceeds the KL threshold and breaks before the alpha block
        # binds effective_alpha, this read would NameError on the very
        # first train() call. Same applies to _batch1_grad_norm captured
        # at the optimizer-step site if accumulation never fires.
        # Tracked: utof/cs2rl issue (early-break unbound state).
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

    Raises RuntimeError on NaN/Inf; accumulates soft warnings and prints
    a DEAD RUN banner when five or more accumulate.
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


def _hybrid_sample_logits(policy_out, action=None, continuous_action=None, max_turn_speed=None):
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
    continuous_action : (B, AIM_DIM=1) float32 tensor, or None
        If None, sample Δyaw from Normal(mu_aim, exp(log_std_aim)) and
        clamp to ±max_turn_speed. If supplied, evaluate log-prob without
        re-sampling.
    max_turn_speed : float or None
        Hard clamp on sampled Δyaw. None means no clamp (only sane in the
        update-pass path where continuous_action is provided pre-clamped).

    Returns
    -------
    action, continuous_action, log_prob_d, log_prob_c, entropy_d, entropy_c
        action : (B, 7) int64
        continuous_action : (B, 1) float32, ∈ [-max_turn_speed, max_turn_speed]
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


def _hybrid_ppo_loss(policy, mb_obs, mb_actions, mb_cont_actions, mb_old_logp_d, mb_old_logp_c,
                     mb_advantages, clip_coef, state):
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

    # ── Re-evaluate discrete and continuous halves under the new policy ──
    # Fix #3 (perf): replaces 7× torch.distributions.Categorical(logits=lg) +
    # 1× torch.distributions.Normal(mu, sigma) construction per minibatch
    # with the same hand-rolled forms used in _hybrid_sample_logits (Fix #2).
    # Microbench measured 8.6 ms/MB savings; PPO update calls this 10×4 = 40
    # times per epoch → ~345 ms/epoch saved on heavy-update epochs. Same
    # numerical contract as before: |Δ| ≤ ~2e-6 vs torch.distributions
    # reference (different softmax reduction order; well within fp32 noise).
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

    return pg_loss, entropy, new_value, new_logp_d + new_logp_c, ratio_d, ratio_c


def _patch_trainer_with_hybrid_aim(trainer, cont_action_view_main=None):
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

    def env_factory(*_args, buf=None, seed=None, _cont_shm=None, _cont_idx=None, **kwargs):
        env = make_puffer_env(team_spirit=shared_ts, buf=buf, seed=seed or 0, map_data=_map_data)
        # Attach the shared-memory view so the env (whether running in the
        # main process under Serial, or a forked worker under
        # Multiprocessing) can pull cont_actions written by the trainer.
        # _cont_idx may be None when env_factory is called outside the
        # train() codepath (eg. legacy callers); attach is a no-op then.
        if _cont_shm is not None and _cont_idx is not None:
            env._attach_cont_action_view(_cont_shm, _cont_idx)
        return env

    # Per-env kwargs list — pufferlib.vector.make accepts a list of dicts
    # (one per env). Both args propagate verbatim through fork because
    # they're stored on env_kwargs[i] BEFORE Process.start() (see
    # .venv/lib/.../pufferlib/vector.py:333-346).
    _per_env_kwargs = [{
        "_cont_shm": _cont_action_shm,
        "_cont_idx": i,
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
        policy.load_state_dict(state_dict)
        print(f"[Train] Resumed from checkpoint: {resume_path}")
    # ────────────────────────────────────────────────────────────────────────

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
    _patch_trainer_with_hybrid_aim(trainer, cont_action_view_main=_cont_action_view_main)

    # ── Self-play setup ──────────────────────────────────────────────────────
    self_play_mgr = None
    if getattr(args, "self_play", True):
        self_play_mgr = SelfPlayManager(
            pool_size=15,
            p_past=0.3,
            save_every_epochs=25,                      # ~2M steps per save at batch_size=81920
            win_threshold=0.6,
            phase_length=50,                           # switch opponent team every ~4M steps
        )
        if resume_path and resume_path.exists():
            import shutil as _shutil

            seed_path = Path(args.checkpoint_dir) / "sp_seed.pt"
            _shutil.copy2(resume_path, seed_path)
            self_play_mgr._add_to_pool(seed_path)
            print(f"[SelfPlay] Pool pre-seeded with resume checkpoint ({seed_path.name})")
        _patch_trainer_with_selfplay(trainer, self_play_mgr)
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
            dead_run_detector.check(trainer.global_step, logs)

            # Network health monitoring every 5 epochs (too expensive every epoch)
            if trainer.epoch % 5 == 0:
                health_metrics = compute_network_health(policy, device)
                logs.update(health_metrics)

            # ── Self-play bookkeeping ────────────────────────────────────────
            if self_play_mgr is not None:
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

            # ── Persist metrics ──────────────────────────────────────────────
            log_entry = {
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
