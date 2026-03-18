#!/usr/bin/env python
"""CS2 RL Sim — training entry point.

Usage:
  python train.py --smoke       # sanity check: 1000 steps, no crash, print steps/sec
  python train.py --train       # full PPO self-play training
  python train.py --record      # run 1 episode, save rerun recording
  python train.py --eval --checkpoint checkpoints/latest.zip
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import time
import numpy as np
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
    if sps < 500:
        print("WARNING: Very slow (<500 steps/sec). Profile step() with cProfile.")
    elif sps < 5000:
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
        from stable_baselines3 import PPO
        model = PPO.load(checkpoint_path)
        print(f"[Record] Loaded checkpoint: {checkpoint_path}")
    else:
        model = None

    done = False
    step_count = 0
    while not done and step_count < ROUND_TIME * 2:
        if model:
            obs_arr = np.stack([obs[aid] for aid in env.possible_agents])
            acts, _ = model.predict(obs_arr, deterministic=True)
            actions = {aid: acts[i] for i, aid in enumerate(env.possible_agents)}
        else:
            actions = {aid: env.action_space(aid).sample() for aid in env.agents}

        obs, rewards, terms, truncs, infos = env.step(actions)
        step_count += 1
        done = all(terms.get(aid, False) for aid in env.possible_agents)

    print(f"[Record] Episode complete ({step_count} ticks). Saved to {save_path}")
    print(f"[Record] View with: python -m rerun {save_path}")


# ── SECTION: Training ─────────────────────────────────────────────────────

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

TRAINING_CONFIG = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=512,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
    verbose=1,
    device="cuda",
)
CHECKPOINT_EVERY  = 100_000
OPPONENT_UPDATE   = 50_000


class OpponentPoolCallback(BaseCallback):
    """Saves policy to opponent pool every N timesteps."""

    def __init__(self, update_freq: int, pool_dir: str = "checkpoints/pool"):
        super().__init__()
        self.update_freq = update_freq
        self.pool_dir = pool_dir
        self._last_update = 0
        os.makedirs(pool_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self.update_freq:
            path = os.path.join(self.pool_dir, f"step_{self.num_timesteps}.zip")
            self.model.save(path)
            self._last_update = self.num_timesteps
            print(f"[OpponentPool] Saved {path}")
        return True


def train(args):
    from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

    os.makedirs("checkpoints", exist_ok=True)

    # Pre-build vis cache once before spawning parallel envs to avoid race conditions
    print("[Train] Pre-building vis cache (one-time)...")
    _warmup = Dust2Env()
    del _warmup

    n_envs = 8

    print(f"[Train] Creating {n_envs} parallel envs...")

    def make_vec():
        env = Dust2Env()
        env = pettingzoo_env_to_vec_env_v1(env)
        return env

    vec_env = concat_vec_envs_v1(
        make_vec, n_envs, num_cpus=1, base_class="stable_baselines3"
    )

    model = PPO("MlpPolicy", vec_env, **TRAINING_CONFIG)

    callbacks = [
        CheckpointCallback(
            save_freq=CHECKPOINT_EVERY,
            save_path="checkpoints/",
            name_prefix="cs2rl",
        ),
        OpponentPoolCallback(update_freq=OPPONENT_UPDATE),
    ]

    print(f"[Train] Starting PPO for {args.timesteps:,} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save("checkpoints/final.zip")
    print("[Train] Done. Saved checkpoints/final.zip")


# ── SECTION: CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",      action="store_true")
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--record",     action="store_true")
    parser.add_argument("--eval",       action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--timesteps",  type=int, default=10_000_000)
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
