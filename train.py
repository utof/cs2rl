#!/usr/bin/env python
"""CS2 RL Sim — training entry point.

Usage:
  python train.py --smoke       # sanity check: 1000 steps, no crash, print steps/sec
  python train.py --train       # full APPO self-play training (Sample Factory)
  python train.py --record      # run 1 episode, save rerun recording (random policy)
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import time
import numpy as np
import gymnasium as gym
from sim import Dust2Env
from sample_factory.algo.utils.context import global_env_registry
from sample_factory.envs.pettingzoo_envs import PettingZooParallelEnv

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

    model = None
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

CHECKPOINT_EVERY = 100_000


# ── SECTION: TeamSpirit callback (SB3-shim + SF daemon) ───────────────────

class TeamSpiritCallback:
    """Linearly anneals sim._TEAM_SPIRIT 0→1 over anneal_steps env steps.

    _on_step() is the SB3-shim used by tests.
    The SF training loop uses _make_daemon_thread() instead.
    """

    def __init__(self, anneal_steps: int = 5_000_000):
        self.anneal_steps = anneal_steps
        self.num_timesteps: int = 0  # set by tests

    def _on_step(self) -> bool:
        import sim as _sim
        _sim._TEAM_SPIRIT = min(1.0, self.num_timesteps / self.anneal_steps)
        return True

    def _make_daemon_thread(self, runner):
        """Returns a stop Event for a started daemon thread that polls runner."""
        import threading, sim as _sim

        stop = threading.Event()
        # NOTE: This daemon updates sim._TEAM_SPIRIT only in the main process.
        # SF worker processes are spawned separately and maintain their own copy of
        # this module global, so they see a static _TEAM_SPIRIT = 0.0 throughout
        # training. True inter-process annealing requires a multiprocessing.Value
        # or SF reward-shaping hooks — tracked as a future improvement.

        def _loop():
            # total_env_steps_since_resume is the attr confirmed in Runner source
            steps_attr = next(
                (a for a in ("total_env_steps_since_resume", "env_steps", "total_env_steps")
                 if hasattr(runner, a)),
                None,
            )
            while not stop.is_set():
                raw = getattr(runner, steps_attr, 0) if steps_attr else 0
                # env_steps is a dict[PolicyID, int]; total_env_steps_since_resume is int
                steps = sum(raw.values()) if isinstance(raw, dict) else (raw or 0)
                _sim._TEAM_SPIRIT = min(1.0, steps / self.anneal_steps)
                stop.wait(timeout=1.0)

        threading.Thread(target=_loop, daemon=True, name="TeamSpiritAnneal").start()
        return stop

    @staticmethod
    def stop_daemon(stop_event):
        stop_event.set()


# OpponentPoolCallback removed — SF handles checkpoints via --save_every_steps.

_team_spirit_cb = TeamSpiritCallback(anneal_steps=5_000_000)


# ── SECTION: Sample Factory env registration ──────────────────────────────


class _MultiDiscreteTupleWrapper:
    """Converts MultiDiscrete action space to Tuple[Discrete] for SF compatibility.

    SF's action distribution code supports Discrete, Tuple, and Box but not
    MultiDiscrete.  This wrapper converts on both sides transparently.

    Wraps a PettingZoo ParallelEnv (not a gym.Env), so we don't inherit from
    gym.Wrapper — just delegate everything.
    """

    def __init__(self, env):
        self._orig_env = env

    @property
    def unwrapped(self):
        return self._orig_env

    def action_space(self, agent):
        md = self._orig_env.action_space(agent)
        return gym.spaces.Tuple([gym.spaces.Discrete(int(n)) for n in md.nvec])

    def observation_space(self, agent):
        return self._orig_env.observation_space(agent)

    def step(self, actions):
        # Convert tuple actions back to numpy arrays for the underlying env
        converted = {}
        for aid, act in actions.items():
            if isinstance(act, (tuple, list)):
                converted[aid] = np.array([int(a) for a in act], dtype=np.int64)
            else:
                converted[aid] = act
        return self._orig_env.step(converted)

    def reset(self, **kwargs):
        return self._orig_env.reset(**kwargs)

    def render(self):
        return self._orig_env.render()

    def close(self):
        return self._orig_env.close()

    @property
    def possible_agents(self):
        return self._orig_env.possible_agents

    @property
    def agents(self):
        return self._orig_env.agents

    @property
    def max_num_agents(self):
        return self._orig_env.max_num_agents

    @property
    def metadata(self):
        return self._orig_env.metadata

    @property
    def render_mode(self):
        return getattr(self._orig_env, "render_mode", None)


def _make_cs2_env(full_env_name: str, cfg=None, env_config=None, render_mode=None) -> PettingZooParallelEnv:
    """Factory function registered with Sample Factory."""
    return PettingZooParallelEnv(_MultiDiscreteTupleWrapper(Dust2Env()))


global_env_registry()["cs2-dust2"] = _make_cs2_env


# ── SECTION: SF training ───────────────────────────────────────────────────

def train(args):
    """Run APPO training via Sample Factory."""
    from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
    from sample_factory.train import make_runner

    os.makedirs(args.train_dir, exist_ok=True)

    print("[Train] Pre-building vis cache (one-time)...")
    _warmup = Dust2Env()
    del _warmup

    argv = [
        "--env", "cs2-dust2",
        "--algo", "APPO",
        "--experiment", "cs2rl",
        "--train_dir", args.train_dir,
        "--num_workers", str(args.num_workers),
        "--num_envs_per_worker", str(args.num_envs_per_worker),
        "--batch_size", "512",
        "--num_batches_per_epoch", "1",
        "--num_epochs", "4",
        "--rollout", "64",
        "--gamma", str(TRAINING_CONFIG["gamma"]),
        "--gae_lambda", "0.95",
        "--exploration_loss_coeff", "0.01",
        "--max_grad_norm", "0.5",
        "--train_for_env_steps", str(args.timesteps),
        "--save_every_sec", "3600",
    ]

    parser, _ = parse_sf_args(argv=argv)
    cfg = parse_full_cfg(parser, argv=argv)
    cfg, runner = make_runner(cfg)
    runner.init()

    stop_event = _team_spirit_cb._make_daemon_thread(runner)
    try:
        print(f"[Train] Starting SF APPO for {args.timesteps:,} env steps...")
        runner.run()
    finally:
        TeamSpiritCallback.stop_daemon(stop_event)

    print("[Train] Done.")


# ── SECTION: CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",               action="store_true")
    parser.add_argument("--train",               action="store_true")
    parser.add_argument("--record",              action="store_true")
    parser.add_argument("--eval",                action="store_true")
    parser.add_argument("--checkpoint",          type=str, default=None)
    parser.add_argument("--timesteps",           type=int, default=10_000_000)
    parser.add_argument("--num_workers",         type=int, default=8)
    parser.add_argument("--num_envs_per_worker", type=int, default=8)
    parser.add_argument("--train_dir",           type=str, default="checkpoints")
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
