#!/usr/bin/env python
"""CS2 RL Sim — training entry point.

Usage:
  python train.py --smoke       # sanity check: 1000 steps, no crash, print steps/sec
  python train.py --train       # full PPO self-play training (PufferLib 3.0)
  python train.py --record      # run 1 episode, save rerun recording (random policy)
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
import gymnasium as gym
from sim import Dust2Env


# ── SECTION: Smoke Test ────────────────────────────────────────────────────

def smoke_test():
    print("[Smoke] Initialising environment...")
    env = Dust2Env()
    obs, _ = env.reset(seed=42)

    assert len(obs) == 10, f"Expected 10 agents, got {len(obs)}"
    for aid, ob in obs.items():
        assert ob.shape == (71,), f"{aid}: shape {ob.shape} != (71,)"
        assert np.isfinite(ob).all(), f"{aid}: NaN in initial obs"

    print("[Smoke] Running 1000 steps...")
    t0 = time.time()
    step_count = 0

    # Cache action spaces to avoid repeated re-seeding overhead
    action_spaces = {aid: env.action_space(aid) for aid in env.possible_agents}

    for step_n in range(1000):
        actions = {aid: action_spaces[aid].sample() for aid in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

        for aid, ob in obs.items():
            assert np.isfinite(ob).all(), f"NaN at step {step_n} agent {aid}"

        step_count += 1

        if all(terms.get(aid, False) for aid in env.possible_agents):
            obs, _ = env.reset()
            action_spaces = {aid: env.action_space(aid) for aid in env.possible_agents}

    elapsed = time.time() - t0
    sps = step_count / elapsed

    print(f"SMOKE TEST PASSED — {sps:.0f} steps/sec")
    assert sps >= 300, f"Smoke test FAILED: {sps:.0f} steps/sec is below the 300 minimum — profile step() with cProfile."
    if sps < 5000:
        print("INFO: Acceptable speed. Target is >5000 for training.")


# ── SECTION: Record Episode ────────────────────────────────────────────────

def record_episode(checkpoint_path=None):
    import os
    from viz import init_recording, log_navmesh, log_tick, log_trimap
    from sim import ROUND_TIME

    os.makedirs("recordings", exist_ok=True)
    save_path = "recordings/latest.rrd"

    print(f"[Record] Initialising rerun recording -> {save_path}")
    init_recording(save_path=save_path)

    def record_fn(state, tick, rewards):
        log_tick(state, tick, rewards)

    env = Dust2Env(record_fn=record_fn)
    log_trimap()
    log_navmesh(env.nav_graph)

    obs, _ = env.reset(seed=0)

    if checkpoint_path:
        print("[Record] WARNING: SB3 checkpoint loading removed; using random policy.")

    done = False
    step_count = 0
    while not done and step_count < ROUND_TIME * 2:
        actions = {aid: env.action_space(aid).sample() for aid in env.agents}

        obs, rewards, terms, truncs, infos = env.step(actions)
        step_count += 1
        done = all(terms.get(aid, False) for aid in env.possible_agents)

    print(f"[Record] Episode complete ({step_count} ticks). Saved to {save_path}")
    print(f"[Record] View with: python -m rerun {save_path}")


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
    """Create a PettingZooPufferEnv wrapping Dust2Env."""
    import pufferlib
    from pufferlib.emulation import PettingZooPufferEnv
    raw_env = Dust2Env(team_spirit=team_spirit)
    return PettingZooPufferEnv(env=raw_env)


# ── SECTION: Policy ────────────────────────────────────────────────────────

def build_policy(vecenv, device):
    import torch
    import torch.nn as nn
    import pufferlib.pytorch

    obs_dim = vecenv.driver_env.single_observation_space.shape[0]  # 71
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
                if 'bias' in name:
                    nn.init.constant_(p, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(p, gain=1.0)

            # Separate heads for MultiDiscrete([9,2,2,2])
            self.action_heads = nn.ModuleList([
                pufferlib.pytorch.layer_init(nn.Linear(hidden, n), std=0.01)
                for n in [9, 2, 2, 2]
            ])
            self.value_head = pufferlib.pytorch.layer_init(nn.Linear(hidden, 1), std=1.0)

        def get_value(self, x, lstm_state=None, done=None):
            hidden_out, lstm_state = self._forward_core(x, lstm_state, done)
            return self.value_head(hidden_out), lstm_state

        def get_action_and_value(self, x, lstm_state=None, done=None, action=None):
            hidden_out, lstm_state = self._forward_core(x, lstm_state, done)
            logits = [head(hidden_out) for head in self.action_heads]

            # MultiCategorical distribution
            dists = [torch.distributions.Categorical(logits=l) for l in logits]
            if action is None:
                action = torch.stack([d.sample() for d in dists], dim=-1)

            log_prob = sum(d.log_prob(action[..., i]) for i, d in enumerate(dists))
            entropy  = sum(d.entropy() for d in dists)
            value    = self.value_head(hidden_out)
            return action, log_prob, entropy, value, lstm_state

        def _forward_core(self, x, lstm_state, done):
            h = self.encoder(x.float())
            # lstm expects (seq, batch, features)
            if lstm_state is not None:
                h, lstm_state = self.lstm(
                    h.unsqueeze(0),
                    (
                        (1.0 - done).view(1, -1, 1) * lstm_state[0],
                        (1.0 - done).view(1, -1, 1) * lstm_state[1],
                    )
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
    import torch
    import pufferlib.vector
    from pufferlib.pufferl import PuffeRL

    device = args.device

    # Shared team spirit value — all envs read it at episode start
    shared_ts = mp.Value('f', 0.0)

    def env_factory(*args, buf=None, seed=None, **kwargs):
        from pufferlib.emulation import PettingZooPufferEnv
        raw_env = Dust2Env(team_spirit=shared_ts)
        return PettingZooPufferEnv(env=raw_env, buf=buf, seed=seed or 0)

    print(f"[Train] Creating {args.num_envs} vectorised envs...")
    vecenv = pufferlib.vector.make(
        env_factory,
        num_envs=args.num_envs,
        backend=pufferlib.vector.Serial,
    )

    print(f"[Train] Building policy on device={device}...")
    policy = build_policy(vecenv, device)

    train_config = {
        # Core PPO
        'env': 'cs2-dust2',
        'device': device,
        'seed': args.seed,
        'total_timesteps': args.timesteps,
        'batch_size': 8192,
        'bptt_horizon': 32,
        'minibatch_size': 2048,
        'max_minibatch_size': 2048,
        'update_epochs': 4,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_coef': 0.1,
        'vf_coef': 0.5,
        'vf_clip_coef': 0.1,
        'ent_coef': 0.01,
        'max_grad_norm': 0.5,
        'use_rnn': True,
        # Extras required by PuffeRL constructor
        'compile': False,
        'compile_mode': 'default',
        'compile_fullgraph': False,
        'cpu_offload': False,
        'torch_deterministic': False,
        'optimizer': 'adam',
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_eps': 1e-8,
        'anneal_lr': True,
        'checkpoint_interval': 200,
        'data_dir': args.checkpoint_dir,
        'precision': 'float32',
        'prio_alpha': 0.0,
        'prio_beta0': 1.0,
        'vtrace_rho_clip': 1.0,
        'vtrace_c_clip': 1.0,
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

        if trainer.epoch % 10 == 0:
            sps = logs.get('SPS', 0) if isinstance(logs, dict) else 0
            ret = logs.get('return', 0) if isinstance(logs, dict) else 0
            print(f"Epoch {trainer.epoch} | SPS: {sps:.0f} | Return: {ret:.3f} | TS: {ts_val:.3f}")

    trainer.close()
    print("[Train] Done.")


# ── SECTION: CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",          action="store_true")
    parser.add_argument("--train",          action="store_true")
    parser.add_argument("--record",         action="store_true")
    parser.add_argument("--eval",           action="store_true")
    parser.add_argument("--checkpoint",     type=str, default=None)
    parser.add_argument("--timesteps",      type=int, default=10_000_000)
    parser.add_argument("--num_envs",       type=int, default=64)
    parser.add_argument("--seed",           type=int, default=1)
    parser.add_argument("--device",         type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--save_every_sec", type=int, default=300)
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoints')
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
    elif args.train:
        train(args)
    elif args.record:
        record_episode(checkpoint_path=args.checkpoint)
    elif args.eval:
        print("Eval not implemented yet")
    else:
        parser.print_help()
