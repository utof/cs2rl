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
import multiprocessing
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

    def _make_daemon_thread(self, runner, shared_ts):
        """Returns a stop Event for a started daemon thread that polls runner.

        shared_ts: multiprocessing.Value('f', 0.0) captured by the env factory
        closure so all SF worker processes see annealing updates in real time.
        """
        import threading

        stop = threading.Event()

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
                shared_ts.value = min(1.0, steps / self.anneal_steps)
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


class _CS2EnvFactory:
    """Picklable SF env factory.

    Defined at module level so multiprocessing.spawn can pickle it.
    When team_spirit is a multiprocessing.Value, all SF worker processes share
    the same underlying memory and see daemon-thread annealing updates in real time.
    """

    def __init__(self, team_spirit=None):
        self.team_spirit = team_spirit

    def __call__(self, full_env_name: str, cfg=None, env_config=None, render_mode=None):
        cfg = cfg or {}  # SF's create_env does `"episode_counter" in cfg` before calling us
        return PettingZooParallelEnv(_MultiDiscreteTupleWrapper(Dust2Env(team_spirit=self.team_spirit)))


global_env_registry()["cs2-dust2"] = _CS2EnvFactory()  # module-level (no shared value)


def _patch_sample_factory_scalar_outputs():
    """Normalize size-1 policy outputs for non-batched sampling rollout buffers.

    Sample Factory stores some scalar outputs as shape-(1,) arrays before writing
    them into scalar trajectory buffer slots. Newer NumPy rejects that assignment
    with "setting an array element with a sequence", so we squeeze these values
    to true scalars/0-D tensors first.
    """
    import torch
    from sample_factory.algo.utils.tensor_dict import TensorDict

    if getattr(TensorDict, "_cs2rl_scalar_patch", False):
        return

    def _patched_set_data_func(self, x, index, new_data):
        if isinstance(new_data, (dict, TensorDict)):
            for new_data_key, new_data_value in new_data.items():
                self._set_data_func(x.get(new_data_key), index, new_data_value)
            return

        if torch.is_tensor(x):
            if isinstance(new_data, torch.Tensor):
                t = new_data
            elif isinstance(new_data, np.ndarray):
                t = torch.from_numpy(new_data)
            else:
                raise ValueError(f"Type {type(new_data)} not supported in set_data_func")

            if x[index].ndim == 0 and t.numel() == 1:
                t = t.reshape(())
            x[index].copy_(t)
            return

        if isinstance(x, np.ndarray):
            if isinstance(new_data, torch.Tensor):
                n = new_data.cpu().numpy()
            elif isinstance(new_data, np.ndarray):
                n = new_data
            else:
                raise ValueError(f"Type {type(new_data)} not supported in set_data_func")

            if np.asarray(x[index]).ndim == 0 and np.asarray(n).size == 1:
                n = np.asarray(n).reshape(())
            x[index] = n
            return

    TensorDict._set_data_func = _patched_set_data_func
    TensorDict._cs2rl_scalar_patch = True


_patch_sample_factory_scalar_outputs()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch

        return "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_worker_num_splits(num_envs_per_worker: int) -> int:
    return 2 if num_envs_per_worker > 1 and num_envs_per_worker % 2 == 0 else 1


# ── SECTION: SF training ───────────────────────────────────────────────────

def train(args):
    """Run APPO training via Sample Factory."""
    from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
    from sample_factory.train import make_runner

    os.makedirs(args.train_dir, exist_ok=True)
    device = _resolve_device(args.device)
    worker_num_splits = _resolve_worker_num_splits(args.num_envs_per_worker)

    print("[Train] Pre-building vis cache (one-time)...")
    _warmup = Dust2Env()
    del _warmup
    print(f"[Train] Using Sample Factory device: {device}")
    print(f"[Train] Using worker_num_splits={worker_num_splits}")

    argv = [
        "--env", "cs2-dust2",
        "--algo", "APPO",
        "--experiment", args.experiment,
        "--train_dir", args.train_dir,
        "--num_workers", str(args.num_workers),
        "--num_envs_per_worker", str(args.num_envs_per_worker),
        "--worker_num_splits", str(worker_num_splits),
        "--batch_size", str(args.batch_size),
        "--num_batches_per_epoch", str(args.num_batches_per_epoch),
        "--num_epochs", str(args.num_epochs),
        "--rollout", str(args.rollout),
        "--gamma", str(TRAINING_CONFIG["gamma"]),
        "--gae_lambda", "0.95",
        "--exploration_loss_coeff", "0.01",
        "--max_grad_norm", "0.5",
        "--train_for_env_steps", str(args.timesteps),
        "--save_every_sec", str(args.save_every_sec),
        "--device", device,
    ]

    parser, _ = parse_sf_args(argv=argv)
    cfg = parse_full_cfg(parser, argv=argv)
    cfg, runner = make_runner(cfg)
    runner.init()

    shared_ts = multiprocessing.get_context("spawn").Value("f", 0.0)
    global_env_registry()["cs2-dust2"] = _CS2EnvFactory(shared_ts)

    stop_event = _team_spirit_cb._make_daemon_thread(runner, shared_ts)
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
    parser.add_argument("--batch_size",          type=int, default=512)
    parser.add_argument("--num_batches_per_epoch", type=int, default=1)
    parser.add_argument("--num_epochs",          type=int, default=4)
    parser.add_argument("--rollout",             type=int, default=64)
    parser.add_argument("--save_every_sec",      type=int, default=3600)
    parser.add_argument("--train_dir",           type=str, default="checkpoints")
    parser.add_argument("--experiment",          type=str, default="cs2rl")
    parser.add_argument("--device",              type=str, choices=("auto", "cpu", "gpu"), default="cpu")
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
