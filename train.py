#!/usr/bin/env python
"""CS2 RL Sim — training entry point.

Usage:
  python train.py --smoke       # sanity check: 20k native-env steps, no crash, print steps/sec
  python train.py --train       # full PPO self-play training (PufferLib 3.0)
  python train.py --record      # run 1 episode, save rerun recording (random policy)
  python train.py --eval        # evaluate a checkpoint across many seeds
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import numpy as np

OBS_DIM = 71

AGENT_IDS = tuple([f"t{i}" for i in range(5)] + [f"ct{i}" for i in range(5)])
ACTION_HEAD_NAMES = ("move", "shoot", "use", "last")
ACTION_HEAD_SIZES = (9, 2, 2, 2)


# ── SECTION: Smoke Test ────────────────────────────────────────────────────


def smoke_test():
    print("[Smoke] Initialising environment...")
    env = make_puffer_env(seed=42)
    try:
        obs, _ = env.reset(seed=42)

        assert obs.shape == (10, 71), f"Expected obs shape (10, 71), got {obs.shape}"
        assert np.isfinite(obs).all(), "NaN in initial obs"

        steps = 20_000
        actions = np.zeros((10, 4), dtype=np.int32)
        print(f"[Smoke] Running {steps} steps...")
        t0 = time.perf_counter()
        step_count = 0

        for step_n in range(steps):
            obs, rewards, terms, truncs, infos = env.step(actions)

            assert obs.shape == (10, 71), f"Unexpected obs shape at step {step_n}: {obs.shape}"
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


def make_puffer_env(team_spirit=None, record_fn=None, buf=None, seed=0, episode_stats=True):
    """Create the native C PufferEnv used by smoke/train/eval."""
    from c_env.wrapper import make_env as make_c_env

    if record_fn is not None:
        raise ValueError("record_fn is only supported by the Python recording env")
    return make_c_env(seed=seed, team_spirit=team_spirit, buf=buf)


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
    save_path="recordings/latest.rrd",
):
    import os

    from c_env.wrapper import make_env as make_c_env
    from sim import CACHE_PATH, NAV_PATH, _load_dust2_static_data
    from viz import init_recording, log_navmesh, log_tick, log_trimap

    os.makedirs("recordings", exist_ok=True)

    print(f"[Record] Initialising rerun recording -> {save_path}")
    init_recording(save_path=save_path)
    static = _load_dust2_static_data(NAV_PATH, CACHE_PATH)
    env = make_c_env(seed=seed, auto_reset=False)
    log_trimap()
    log_navmesh(static["nav_graph"])

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


_team_spirit_cb = TeamSpiritCallback(anneal_steps=5_000_000)


# ── SECTION: PufferLib env factory ─────────────────────────────────────────


def make_env(team_spirit=None):
    return make_puffer_env(team_spirit=team_spirit)


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


# ── SECTION: PufferLib training ────────────────────────────────────────────


def train(args):
    """Run PPO training via PufferLib 3.0."""
    import pufferlib.vector
    import torch
    from pufferlib.pufferl import PuffeRL

    device = args.device

    # Shared team spirit value — all envs read it at episode start
    shared_ts = mp.Value("f", 0.0)

    def env_factory(*args, buf=None, seed=None, **kwargs):
        return make_puffer_env(team_spirit=shared_ts, buf=buf, seed=seed or 0)

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

    train_config = {
        # Core PPO
        "env": "cs2-dust2",
        "device": device,
        "seed": args.seed,
        "total_timesteps": args.timesteps,
        "batch_size": 20480,  # must be >= num_envs * agents_per_env * bptt_horizon = 64*10*32
        "bptt_horizon": 32,
        "minibatch_size": 4096,
        "max_minibatch_size": 4096,
        "update_epochs": 4,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.1,
        "vf_coef": 0.5,
        "vf_clip_coef": 0.1,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "use_rnn": True,
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

    save_path = Path(args.checkpoint_dir) / "dust2_policy.pt"
    last_save = time.time()

    print(f"[Train] Starting PufferLib PPO for {args.timesteps:,} env steps...")
    while trainer.epoch < trainer.total_epochs:
        trainer.evaluate()
        logs = trainer.train()

        # Team spirit annealing: 0→1 over 5M steps
        ts_val = min(1.0, trainer.global_step / 5_000_000)
        shared_ts.value = ts_val

        if time.time() - last_save > args.save_every_sec:
            save_path.parent.mkdir(exist_ok=True)
            torch.save(policy.state_dict(), save_path)
            last_save = time.time()
            print(f"Saved checkpoint to {save_path}")

        if trainer.epoch % 10 == 0 and isinstance(logs, dict):
            print(format_train_status(trainer.epoch, ts_val, logs))

    trainer.close()
    print("[Train] Done.")


# ── SECTION: CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_every_sec", type=int, default=300)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--vec-backend", type=str, default="multiprocessing")
    parser.add_argument("--vec-num-workers", type=int, default=0)
    parser.add_argument("--vec-overwork", action="store_true")
    parser.add_argument("--record-out", type=str, default="recordings/latest.rrd")
    parser.add_argument(
        "--record-policy", type=str, choices=("auto", "random", "sample", "greedy"), default="auto"
    )
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument(
        "--eval-policy", type=str, choices=("auto", "random", "sample", "greedy"), default="auto"
    )
    args = parser.parse_args()

    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        smoke_test()
    elif args.train:
        train(args)
    elif args.record:
        record_episode(
            checkpoint_path=args.checkpoint,
            device=args.device,
            seed=args.seed,
            policy_mode=args.record_policy,
            save_path=args.record_out,
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
