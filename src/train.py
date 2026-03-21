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
import multiprocessing as mp
import random
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np

from paths import CHECKPOINTS_DIR, RECORDINGS_DIR

OBS_DIM = 72


def resolve_run_name(name: str) -> str:
    """Return a run name prefixed with DDMMYY-N- where N is the count of existing
    checkpoint dirs that already start with today's date prefix."""
    from datetime import date

    today = date.today()
    date_prefix = today.strftime("%d%m%y")  # e.g. "200326"
    checkpoints_dir = CHECKPOINTS_DIR
    count = 0
    if checkpoints_dir.exists():
        prefix = date_prefix + "-"
        count = sum(
            1 for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)
        )
    return f"{date_prefix}-{count}-{name}"


AGENT_IDS = tuple([f"t{i}" for i in range(5)] + [f"ct{i}" for i in range(5)])
ACTION_HEAD_NAMES = ("move", "shoot", "use", "last")
ACTION_HEAD_SIZES = (9, 2, 2, 2)


# ── SECTION: Smoke Test ────────────────────────────────────────────────────


def smoke_test():
    print("[Smoke] Initialising environment...")
    env = make_puffer_env(seed=42)
    try:
        obs, _ = env.reset(seed=42)

        assert obs.shape == (10, OBS_DIM), f"Expected obs shape (10, {OBS_DIM}), got {obs.shape}"
        assert np.isfinite(obs).all(), "NaN in initial obs"

        steps = 20_000
        actions = np.zeros((10, 4), dtype=np.int32)
        print(f"[Smoke] Running {steps} steps...")
        t0 = time.perf_counter()
        step_count = 0

        for step_n in range(steps):
            obs, rewards, terms, truncs, infos = env.step(actions)

            assert obs.shape == (10, OBS_DIM), f"Unexpected obs shape at step {step_n}: {obs.shape}"
            assert rewards.shape == (10,), (
                f"Unexpected reward shape at step {step_n}: {rewards.shape}"
            )
            assert terms.shape == (10,), f"Unexpected term shape at step {step_n}: {terms.shape}"
            assert truncs.shape == (10,), f"Unexpected trunc shape at step {step_n}: {truncs.shape}"
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


def make_puffer_env(
    team_spirit=None, record_fn=None, buf=None, seed=0, episode_stats=True, map_data=None
):
    """Create the native C PufferEnv used by smoke/train/eval."""
    from c_env.wrapper import make_env as make_c_env

    if record_fn is not None:
        raise ValueError("record_fn is only supported by the Python recording env")
    return make_c_env(seed=seed, team_spirit=team_spirit, buf=buf, map_data=map_data)


def load_policy_from_checkpoint(checkpoint_path, device):
    import torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Policy] Loading checkpoint -> {checkpoint_path}")
    policy_env = make_puffer_env()
    try:
        policy = build_policy(policy_env, device)
    finally:
        policy_env.close()

    state_dict = torch.load(checkpoint_path, map_location=device)
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
    return {aid: np.zeros((OBS_DIM,), dtype=np.float32) for aid in AGENT_IDS}


def update_obs_buffer(obs_buffer, obs, terms=None, truncs=None):
    for aid, ob in obs.items():
        obs_buffer[aid] = ob

    for aid in AGENT_IDS:
        if aid not in obs:
            obs_buffer[aid].fill(0.0)


def select_policy_actions(policy, obs_buffer, active_agents, device, policy_state, policy_mode):
    import pufferlib.pytorch
    import torch

    if policy_mode == "random":
        raise ValueError("Random action selection should bypass select_policy_actions")

    obs_arr = np.stack([obs_buffer[aid] for aid in AGENT_IDS])
    obs_t = torch.as_tensor(obs_arr, device=device)

    with torch.no_grad():
        logits, _ = policy.forward_eval(obs_t, policy_state)
        if policy_mode == "sample":
            act_t, _, _ = pufferlib.pytorch.sample_logits(logits)
        else:
            act_t = torch.stack([head.argmax(dim=-1) for head in logits], dim=-1)

    act_np = act_t.cpu().numpy().astype(np.int64)
    return {aid: act_np[i] for i, aid in enumerate(AGENT_IDS) if aid in active_agents}


def select_policy_actions_native(policy, obs, device, policy_state, policy_mode):
    import pufferlib.pytorch
    import torch

    if policy_mode == "random":
        raise ValueError("Random action selection should bypass select_policy_actions_native")

    obs_t = torch.as_tensor(obs, device=device)
    with torch.no_grad():
        logits, _ = policy.forward_eval(obs_t, policy_state)
        if policy_mode == "sample":
            act_t, _, _ = pufferlib.pytorch.sample_logits(logits)
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
    return (
        f"Epoch {epoch} | SPS: {sps:.0f} | Timeout: {timeout:.3f} | "
        f"TWin: {t_win:.3f} | CTWin: {ct_win:.3f} | Plant: {plant:.3f} | "
        f"Kills(T/CT): {kills_t:.2f}/{kills_ct:.2f} | RoundLen: {round_len:.1f} | "
        f"Move1: {move_1:.1f} | TS: {ts_val:.3f}"
    )


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
    from c_env.wrapper import make_env as make_c_env
    from sim import CACHE_PATH, NAV_PATH, _load_dust2_static_data
    from viz import init_recording, log_navmesh, log_tick, log_trimap

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Record] Initialising rerun recording -> {save_path}")
    init_recording(save_path=str(save_path))
    env = make_c_env(seed=seed, auto_reset=False, map_data=map_data)
    if map_data is None:
        static = _load_dust2_static_data(NAV_PATH, CACHE_PATH)
        log_trimap()
        log_navmesh(static["nav_graph"])
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
                np.logical_or(terms, truncs).astype(np.float32)
            )

    print(f"[Record] Episode complete ({step_count} ticks). Saved to {save_path}")
    print(f"[Record] View with: python -m rerun {save_path}")


# ── SECTION: Checkpoint evaluation ─────────────────────────────────────────


def evaluate_checkpoint(
    checkpoint_path=None, device="cpu", start_seed=0, num_episodes=50, policy_mode="auto"
):
    from sim import ROUND_TIME

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
                actions = select_policy_actions_native(
                    policy, obs, device, policy_state, policy_mode
                )

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
                    np.logical_or(terms, truncs).astype(np.float32)
                )

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

    print(
        f"[Eval] checkpoint={checkpoint_path or 'None'} policy={policy_mode} "
        f"episodes={metrics['episodes']} seeds={start_seed}..{start_seed + num_episodes - 1}"
    )
    print(
        f"[Eval] timeout_rate={metrics['timed_out'] / episodes:.3f} "
        f"t_win_rate={metrics['winner_t'] / episodes:.3f} "
        f"ct_win_rate={metrics['winner_ct'] / episodes:.3f}"
    )
    print(
        f"[Eval] plant_rate={metrics['bomb_planted'] / episodes:.3f} "
        f"defuse_rate={metrics['bomb_defused'] / episodes:.3f} "
        f"kills_t_per_round={metrics['kills_t'] / episodes:.3f} "
        f"kills_ct_per_round={metrics['kills_ct'] / episodes:.3f}"
    )
    print(
        f"[Eval] avg_round_length={metrics['round_length'] / episodes:.1f} "
        f"avg_alive_t_end={metrics['alive_t_end'] / episodes:.2f} "
        f"avg_alive_ct_end={metrics['alive_ct_end'] / episodes:.2f}"
    )
    print(
        f"[Eval] blocked_moves_t_per_round={metrics['blocked_moves_t'] / episodes:.2f} "
        f"blocked_moves_ct_per_round={metrics['blocked_moves_ct'] / episodes:.2f}"
    )

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
    gamma=0.99,  # used by PBRS shaping in sim.py and test_reward.py
)


# ── SECTION: TeamSpirit callback (legacy shim — kept for tests) ────────────


class TeamSpiritCallback:
    """Linearly anneals sim._TEAM_SPIRIT 0→1 over anneal_steps env steps.

    _on_step() is the SB3-shim used by tests.
    """

    def __init__(self, anneal_steps: int = 5_000_000):
        self.anneal_steps = anneal_steps
        self.num_timesteps: int = 0  # set by tests

    def _on_step(self) -> bool:
        import sim as _sim

        _sim._TEAM_SPIRIT = min(1.0, self.num_timesteps / self.anneal_steps)
        return True


# ── SECTION: PufferLib env factory ─────────────────────────────────────────


def make_env(team_spirit=None, map_data=None):
    return make_puffer_env(team_spirit=team_spirit, map_data=map_data)


# ── SECTION: Policy ────────────────────────────────────────────────────────


def build_policy(vecenv, device):
    import pufferlib.pytorch
    import torch
    import torch.nn as nn

    driver_env = getattr(vecenv, "driver_env", vecenv)
    obs_dim = driver_env.single_observation_space.shape[0]  # 71
    hidden = 256

    class Dust2Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = hidden  # required by PufferLib LSTM logic

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

            # Separate heads for MultiDiscrete([9,2,2,2])
            self.action_heads = nn.ModuleList(
                [pufferlib.pytorch.layer_init(nn.Linear(hidden, n), std=0.01) for n in [9, 2, 2, 2]]
            )
            self.value_head = pufferlib.pytorch.layer_init(nn.Linear(hidden, 1), std=1.0)

        def get_value(self, x, lstm_state=None, done=None):
            hidden_out, lstm_state = self._forward_core(x, lstm_state, done)
            return self.value_head(hidden_out), lstm_state

        def get_action_and_value(self, x, lstm_state=None, done=None, action=None):
            hidden_out, lstm_state = self._forward_core(x, lstm_state, done)
            logits = [head(hidden_out) for head in self.action_heads]

            # MultiCategorical distribution
            dists = [torch.distributions.Categorical(logits=head_logits) for head_logits in logits]
            if action is None:
                action = torch.stack([d.sample() for d in dists], dim=-1)

            log_prob = sum(d.log_prob(action[..., i]) for i, d in enumerate(dists))
            entropy = sum(d.entropy() for d in dists)
            value = self.value_head(hidden_out)
            return action, log_prob, entropy, value, lstm_state

        def forward_eval(self, x, state):
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
            return logits, value

        def forward(self, x, state):
            if x.ndim == 3:
                x_flat = x.reshape(-1, x.shape[-1])
            else:
                x_flat = x

            hidden_out, _ = self._forward_core(x_flat, None, None)
            logits = [head(hidden_out) for head in self.action_heads]
            value = self.value_head(hidden_out)

            return logits, value

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

    import pufferlib.pytorch
    import torch
    from pufferlib.pufferl import compute_puff_advantage

    # Running stats for return normalization (Welford-style, torch tensors)
    device = trainer.config["device"]
    _ret_mean = torch.zeros(1, device=device)
    _ret_var = torch.ones(1, device=device)
    _ret_count = torch.zeros(1, device=device)

    # ── ADAPTIVE ENTROPY (Lagrangian / SAC-style alpha) ────────────────────
    max_entropy = np.log(9) + 3 * np.log(2)  # ≈ 4.276 for MultiDiscrete([9,2,2,2])
    target_entropy = 0.5 * max_entropy  # ≈ 2.14
    entropy_floor = 0.3 * max_entropy  # collapse threshold
    import math

    log_alpha = torch.tensor([math.log(0.1)], requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-4)
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
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments * prio_probs[idx, None]) ** -anneal_beta
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_rewards = self.rewards[idx]
            mb_terminals = self.terminals[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

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

            logits, newvalue = self.policy(mb_obs, state)
            actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(
                logits, action=mb_actions
            )

            profile("train_misc", epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

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

            # Losses — use normalized returns as regression target
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns_norm.shape)
            v_loss_unclipped = (newvalue - mb_returns_norm) ** 2
            if vf_clip is not None:
                v_clipped = mb_values_norm + torch.clamp(
                    newvalue - mb_values_norm, -vf_clip, vf_clip
                )
                v_loss_clipped = (v_clipped - mb_returns_norm) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * v_loss_unclipped.mean()

            current_entropy = entropy.mean()

            # ── ADAPTIVE ALPHA (SAC-style Lagrangian entropy tuning) ───────
            alpha = log_alpha.exp()
            alpha_loss = (log_alpha * (current_entropy - target_entropy).detach()).mean()
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
                _head_names = ["move", "shoot", "use", "last"]
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
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config["max_grad_norm"])
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
                    f"CRITICAL: Entropy collapsed to {entropy_total:.2f} at step {step}"
                )
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
        self.opponent_team = "ct"  # CT is opponent first; T learns to attack
        self._milestone_count = 0

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
        if epoch % self.save_every_epochs == 0 or hero_win > self.win_threshold:
            path = checkpoint_dir / f"sp_{epoch:06d}.pt"
            torch.save(policy.state_dict(), path)
            self._add_to_pool(path)
            self._milestone_count += 1
            print(
                f"[SelfPlay] Saved checkpoint → {path.name}  "
                f"(pool={len(self.pool)}, hero_win={hero_win:.2f})"
            )

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
            mask[base + slots.start : base + slots.stop] = True
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

    # Past-policy LSTM state — same dict structure as trainer.lstm_h
    # key → (agents_per_batch, hidden_size)
    past_lstm_h = {k: torch.zeros_like(v) for k, v in trainer.lstm_h.items()}
    past_lstm_c = {k: torch.zeros_like(v) for k, v in trainer.lstm_h.items()}

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

                logits, value = self.policy.forward_eval(o_device, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                r = torch.clamp(r, -1, 1)

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
                    opp_logits, _ = past_policy.forward_eval(o_device[opp_mask], past_state)
                    opp_action, opp_logprob, _ = pufferlib.pytorch.sample_logits(opp_logits)

                    # Write back updated past-policy LSTM states (cast from fp16 if needed)
                    past_lstm_h[env_id.start][opp_mask] = past_state["lstm_h"].to(
                        past_lstm_h[env_id.start].dtype
                    )
                    past_lstm_c[env_id.start][opp_mask] = past_state["lstm_c"].to(
                        past_lstm_c[env_id.start].dtype
                    )

                    # Replace opponent slots in action & logprob buffers
                    # Cast to destination dtype (amp_context may yield fp16)
                    action[opp_idx] = opp_action.to(action.dtype)
                    logprob[opp_idx] = opp_logprob.to(logprob.dtype)
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
                self.rewards[batch_rows, seq_pos] = r
                self.terminals[batch_rows, seq_pos] = d.float()
                self.values[batch_rows, seq_pos] = value.flatten()

                self.ep_lengths[env_id] += 1
                if seq_pos + 1 >= cfg["bptt_horizon"]:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = (
                        self.free_idx + torch.arange(num_full, device=dev).int()
                    )
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
            self.vecenv.send(action)

        profile("eval_misc", epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=dev, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    trainer.evaluate = types.MethodType(_evaluate_with_selfplay, trainer)
    print("[Train] Self-play evaluate patch enabled.")
    return trainer


# ── SECTION: PufferLib training ────────────────────────────────────────────


def train(args):
    """Run PPO training via PufferLib 3.0."""
    import pufferlib.vector
    import torch
    from pufferlib.pufferl import PuffeRL

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

    def env_factory(*_args, buf=None, seed=None, **kwargs):
        return make_puffer_env(team_spirit=shared_ts, buf=buf, seed=seed or 0, map_data=_map_data)

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

    print(
        f"[Train] Creating {args.num_envs} vectorised envs "
        f"(backend={backend_name}, workers={num_workers})..."
    )
    vecenv = pufferlib.vector.make(
        env_factory,
        num_envs=args.num_envs,
        backend=backend,
        **vec_kwargs,
    )

    print(f"[Train] Building policy on device={device}...")
    policy = build_policy(vecenv, device)

    agents_per_env = 10
    bptt_horizon = 64
    batch_size = args.num_envs * agents_per_env * bptt_horizon
    # batch_size = 128 * 10 * 64 = 81920 → 81920 / 8192 = 10 minibatches per epoch

    train_config = {
        # Core PPO
        "env": "cs2-dust2",
        "device": device,
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
        "ent_coef": 0.1,  # fallback; adaptive alpha overrides this in the patched train method
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

    trainer = PuffeRL(train_config, vecenv, policy)
    trainer.optimizer.param_groups[0]["weight_decay"] = 1e-4
    _patch_trainer_with_return_norm(trainer)

    # ── Self-play setup ──────────────────────────────────────────────────────
    self_play_mgr = None
    if getattr(args, "self_play", True):
        self_play_mgr = SelfPlayManager(
            pool_size=15,
            p_past=0.3,
            save_every_epochs=25,  # ~2M steps per save at batch_size=81920
            win_threshold=0.6,
            phase_length=50,  # switch opponent team every ~4M steps
        )
        _patch_trainer_with_selfplay(trainer, self_play_mgr)
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
                logs["self_play/opponent_team"] = float(
                    self_play_mgr.opponent_team == "ct"
                )  # 1.0 = CT opponent, 0.0 = T opponent
            # ────────────────────────────────────────────────────────────────

            # ── Persist metrics ──────────────────────────────────────────────
            log_entry = {
                "step": trainer.global_step,
                "epoch": trainer.epoch,
                "team_spirit": ts_val,
                **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
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

        if trainer.epoch % 10 == 0 and isinstance(logs, dict):
            print(format_train_status(trainer.epoch, ts_val, logs))

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
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_every_sec", type=int, default=300)
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINTS_DIR))
    parser.add_argument("--vec-backend", type=str, default="multiprocessing")
    parser.add_argument("--vec-num-workers", type=int, default=0)
    parser.add_argument("--vec-overwork", action="store_true")
    parser.add_argument("--record-out", type=str, default=str(RECORDINGS_DIR / "latest.rrd"))
    parser.add_argument(
        "--record-policy", type=str, choices=("auto", "random", "sample", "greedy"), default="auto"
    )
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument(
        "--eval-policy", type=str, choices=("auto", "random", "sample", "greedy"), default="auto"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Run name; auto-prefixed with DDMMYY-N- where N = count of existing checkpoint dirs "
            "starting with today's date. E.g. --name 1M-ct → '200326-3-1M-ct'."
        ),
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
                candidates = (
                    [d for d in CHECKPOINTS_DIR.iterdir() if d.is_dir() and d.name.endswith(suffix)]
                    if CHECKPOINTS_DIR.exists()
                    else []
                )
                if not candidates:
                    raise FileNotFoundError(
                        f"No checkpoint dir in {CHECKPOINTS_DIR} ending with '{suffix}'"
                    )
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
