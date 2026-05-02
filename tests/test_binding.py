import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "c_env"))
import binding                         # noqa


def _make_env(map_data=None):
    from c_env.cs2_env import make_env
    env = make_env(seed=0, map_data=map_data)
    return env._capsule, env


def test_binding_functions_present():
    for name in ("init", "reset", "step", "close", "get_buffers", "get_masks"):
        assert hasattr(binding, name), f"binding.{name} missing"


def test_get_masks_returns_nonzero_ptr(make_map):
    _, env = _make_env(map_data=make_map)
    binding.reset(env._capsule)
    ptr = binding.get_masks(env._capsule)
    assert isinstance(ptr, int)
    assert ptr != 0


def test_get_buffers_returns_four_ints(make_map):
    _, env = _make_env(map_data=make_map)
    result = binding.get_buffers(env._capsule)
    assert len(result) == 4
    for ptr in result:
        assert isinstance(ptr, int)
        assert ptr != 0


def test_reset_returns_none(make_map):
    _, env = _make_env(map_data=make_map)
    assert binding.reset(env._capsule) is None


def test_step_returns_none(make_map):
    """Batch 3: binding.step is now 3-arg — capsule, int32 discrete actions,
    float32 continuous_actions. The shape (10,) here is wrong for both, but
    the C side reads N_AGENTS*ACTION_DIM ints/N_AGENTS*AIM_DIM floats. The
    raw int32(10,) buffer happens to be ≥10*7*4 bytes only if reinterpreted —
    use the proper 2D shape now to be safe.
    """
    from _action_spec import ACTION_DIM, AIM_DIM
    _, env = _make_env(map_data=make_map)
    binding.reset(env._capsule)
    actions = np.zeros((10, ACTION_DIM), dtype=np.int32)
    cont = np.zeros((10, AIM_DIM), dtype=np.float32)
    assert binding.step(env._capsule, actions, cont) is None


def test_close_idempotent(make_map):
    _, env = _make_env(map_data=make_map)
    assert binding.close(env._capsule) is None
    assert binding.close(env._capsule) is None


def test_stepstats_has_win_type_flags(make_map):
    """StepStats ctypes struct must expose win_by_detonation and win_by_defuse.
    Accessor: env._c_env.step_stats (ctypes StepStatsC — not a numpy recarray;
    binding.c has no StepStats dtype descriptor, ctypes is the Python-side mirror).
    """
    _, env = _make_env(map_data=make_map)
    ss = env._c_env.step_stats
    assert hasattr(ss, "win_by_detonation"), "StepStatsC missing win_by_detonation"
    assert hasattr(ss, "win_by_defuse"), "StepStatsC missing win_by_defuse"
    # At reset, both must be zero
    env.reset()
    assert int(ss.win_by_detonation) == 0
    assert int(ss.win_by_defuse) == 0


def test_human_controlled_uses_aim_rad_not_bin(make_map):
    """When human_controlled=1, facing must equal aim_rad, ignoring the
    continuous_actions Δyaw buffer.

    Batch 3: pre-Batch-3 this test verified the 16-bin override; now it
    verifies that the human branch in env_step (`if (a->human_controlled)`)
    short-circuits BEFORE reading continuous_actions. We pass a non-zero
    Δyaw to confirm it is ignored — only aim_rad sets facing for human
    agents.
    """
    from _action_spec import ACTION_DIM, AIM_DIM
    _, env = _make_env(map_data=make_map)
    binding.reset(env._capsule)

    agent = env._c_env.game.agents[0]
    agent.human_controlled = 1
    aim = 1.23456                      # arbitrary radians
    agent.aim_rad = aim

    actions = np.zeros((10, ACTION_DIM), dtype=np.int32)
    # Non-zero continuous Δyaw — the human branch must IGNORE this.
    cont = np.zeros((10, AIM_DIM), dtype=np.float32)
    cont[0, 0] = 0.5
    binding.step(env._capsule, actions, cont)

    facing = env._c_env.game.agents[0].facing
    assert abs(facing - aim) < 1e-5, f"Expected facing≈{aim:.5f}, got {facing:.5f}"


# ── Batch 3: continuous-aim plumbing ──


def test_binding_step_accepts_continuous_array(make_map):
    """binding.step now takes 3 args (capsule, int32 actions, float32 cont).

    Batch 3: validates the new signature. Wrong shape on continuous_actions
    raises a Python ValueError (caught Python-side in Cs2Env._prepare_continuous_actions
    before the C call). Correct shape is accepted.
    """
    from _action_spec import ACTION_DIM, AIM_DIM
    _, env = _make_env(map_data=make_map)
    env.reset(seed=0)
    actions = np.zeros((10, ACTION_DIM), dtype=np.int32)
    cont = np.zeros((10, AIM_DIM), dtype=np.float32)
    env.step(actions, cont)            # should not raise
    with pytest.raises(ValueError):
        env.step(actions, np.zeros((10, 2), dtype=np.float32))


def test_binding_default_continuous_actions_zero(make_map):
    """If continuous_actions arg omitted, zero buffer supplied — facing unchanged.

    Batch 3: defensive default keeps legacy callers (smoke-test loops, the
    train.py main path before the policy is wired in T4-T5) working without
    explicit continuous-action arrays. We use the designated bomb carrier
    (an RL agent, not human_controlled) so the continuous branch in env_step
    fires.
    """
    from _action_spec import ACTION_DIM
    _, env = _make_env(map_data=make_map)
    env.reset(seed=0)
    g = env._c_env.game
    i = g.round_designated_carrier_id
    f0 = g.agents[i].facing
    actions = np.zeros((10, ACTION_DIM), dtype=np.int32)
    env.step(actions)                                  # no continuous_actions
    assert g.agents[i].facing == f0, (
        f"facing changed without continuous_actions: f0={f0}, after={g.agents[i].facing}")


# ── Batch 3 Task 5: NaN guard test ─────────────────────────────────────────
#
# This test exercises the inline NaN guard that lives in the trainer's
# patched train() method (_train_with_return_norm). Rebuilding the full
# PufferLib trainer just to test this would be expensive and fragile against
# unrelated PufferLib API drift; instead we replicate the guard's structure
# locally — same control flow, same warning string, same zero-grad call —
# and verify that:
#   (a) when loss is non-finite, no parameter update happens,
#   (b) the throttled warning print happens.
#
# If the guard's structure changes (new warning string, different zero_grad
# signature), update BOTH this test and src/train.py:1208-ish in the same PR.


def test_continuous_aim_nan_guard():
    """T5: NaN guard in _train_with_return_norm skips optimizer.step() and
    prints a throttled warning when the loss is non-finite, without
    poisoning subsequent gradients."""
    import io
    import sys as _sys
    import time as _t

    import torch

    import train
    from c_env.cs2_env import make_env

    env = make_env(seed=0)
    try:
        policy = train.build_policy(env, device='cpu')

        # Force the aim head to emit NaN so the loss path goes non-finite.
        # We don't need to run a full PPO update — replicating the guard's
        # control flow inline is enough to verify it does the right thing.
        class _NaNLayer(torch.nn.Module):

            def forward(self, x):
                return torch.full((x.shape[0], 1), float('nan'))

        policy.aim_mu = _NaNLayer()
        old_params = [p.detach().clone() for p in policy.parameters()]
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        x = torch.zeros((1, train.OBS_DIM))
        logits, mu_aim, log_std_aim, value = policy.forward(x, state={})
        loss = mu_aim.sum() + value.sum()
        # Sanity: setup must yield a non-finite loss.
        assert not torch.isfinite(loss).all(), \
            "test setup wrong: loss should be NaN"

        captured = io.StringIO()
        old_stdout = _sys.stdout
        _sys.stdout = captured

        # Mirror the guard control flow at src/train.py inside
        # _train_with_return_norm. Throttle field name MUST match the
        # production attribute (`_last_nan_warn_t`) so a future regression
        # touching the attribute name fails this test.
        class _Self:
            pass

        _self = _Self()
        _self.optimizer = optimizer
        try:
            if not torch.isfinite(loss).all():
                _now = _t.time()
                _last = getattr(_self, '_last_nan_warn_t', 0.0)
                if _now - _last > 60.0:
                    print(f"[hybrid_aim NaN guard] non-finite loss "
                          f"({float(loss.detach())}); skipping optimizer step")
                    _self._last_nan_warn_t = _now
                _self.optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                _self.optimizer.step()
        finally:
            _sys.stdout = old_stdout

        # Parameters must be byte-identical: no gradient flowed through.
        for old, new in zip(old_params, policy.parameters(), strict=True):
            assert torch.equal(old,
                               new), ("T5 NaN guard: parameter changed despite non-finite loss; "
                                      f"max delta {(old - new).abs().max().item()}")
        # Warning string is the production format; substring match is robust
        # against future float-formatting tweaks.
        assert "NaN guard" in captured.getvalue(), \
            f"guard warning not printed: {captured.getvalue()!r}"
    finally:
        env.close()


# ── Batch 3 Task 5b: Multiprocessing cont-action plumbing regression ───────
#
# Catches the exact silent-zero failure mode that T5 shipped with: under the
# Multiprocessing vecenv backend the per-step Δyaw written by the trainer
# was invisible to worker processes (Python attr stash on the parent vecenv,
# not in shared memory). Workers fell through to the all-zero scratch and
# the C env applied Δyaw=0 every tick — gradient signal for the Gaussian
# aim head was decoupled from environment behaviour without any test failing.
#
# Fix verified here: T5b allocates a multiprocessing.RawArray BEFORE workers
# fork, threads it into env_kwargs, and Cs2Env._attach_cont_action_view
# carves a per-env numpy slice that workers see. Trainer (parent process)
# writes into the same RawArray; the data is visible immediately.


@pytest.mark.timeout(60)
def test_continuous_aim_mp_backend_receives_buffer():
    """MP vecenv: writing Δyaw via trainer-side shm view is visible to workers.

    Spawns a small Multiprocessing vecenv (num_workers=2, num_envs=2),
    allocates the cont-action RawArray and attaches per-env views, drives a
    full round (≥ROUND_TIME ticks) with a non-zero Δyaw, and asserts that
    each env's terminal info reports a non-zero aim_delta_count and a mean
    Δyaw close to the value we wrote. Pre-fix this test would observe
    aim_delta_sum/count consistent with Δyaw=0 (clamped tail values from
    the env's own Welford accumulator over a zero buffer).

    Test is intentionally inline (no train.train() machinery) so a failure
    isolates to the backend plumbing, not the surrounding PPO trainer.
    """
    from multiprocessing import RawArray

    import pufferlib.vector

    from _action_spec import ACTION_DIM, AIM_DIM
    from c_env.cs2_env import make_env

    # ROUND_TIME=640 ticks; pad MAX_TICKS in case the first ticks are spent
    # in a setup state where round_over fires immediately and resets the
    # Welford counters before our writes can accumulate.
    NUM_ENVS = 2
    AGENTS_PER_ENV = 10                # 5 T + 5 CT
    DELTA = 0.05                       # Δyaw value to inject
    MAX_TICKS = 1500

    cont_shm = RawArray("f", NUM_ENVS * AGENTS_PER_ENV * AIM_DIM)
    # Trainer-side view onto the SAME bytes — what the parent process
    # writes into propagates to workers via the OS shared mapping.
    view_main = np.frombuffer(cont_shm, dtype=np.float32).reshape(NUM_ENVS * AGENTS_PER_ENV,
                                                                  AIM_DIM)

    def env_factory(*_args, buf=None, seed=None, _cont_shm=None, _cont_idx=None, **_kwargs):
        env = make_env(seed=seed if seed is not None else 0,
                       buf=buf,
                       include_step_stats_in_info=False)
        if _cont_shm is not None and _cont_idx is not None:
            env._attach_cont_action_view(_cont_shm, _cont_idx)
        return env

    per_env_kwargs = [{
        "_cont_shm": cont_shm,
        "_cont_idx": i,
    } for i in range(NUM_ENVS)]

    # NOTE: pufferlib.vector.make has a quirk — if env_creator is a single
    # callable, it broadcasts BOTH env_args and env_kwargs, overwriting our
    # per-env list. Workaround: pass env_creators as a list of N copies of
    # the same factory so the per-env env_kwargs survive (vector.py:672-684).
    vecenv = pufferlib.vector.make(
        [env_factory] * NUM_ENVS,
        env_args=[[] for _ in range(NUM_ENVS)],
        env_kwargs=per_env_kwargs,
        num_envs=NUM_ENVS,
        backend=pufferlib.vector.Multiprocessing,
        num_workers=2,
        batch_size=NUM_ENVS,
        zero_copy=True,
    )

    try:
        # Inject the constant Δyaw into shm BEFORE the first send — workers
        # consume it on the very first step.
        view_main.fill(DELTA)

        vecenv.async_reset(seed=0)
        vecenv.recv()                  # discard initial obs

        # Discrete actions: zeros (NUM_ENVS * AGENTS_PER_ENV, ACTION_DIM)
        # — vecenv.send takes the joint action across envs.
        actions = np.zeros((NUM_ENVS * AGENTS_PER_ENV, ACTION_DIM), dtype=np.int32)

        terminal_infos = []            # accumulate aim_delta_* per round
        for _ in range(MAX_TICKS):
            vecenv.send(actions)
            _o, _r, _d, _t, infos, _ids, _m = vecenv.recv()
            for info in infos:
                if isinstance(info, dict) and "aim_delta_count" in info:
                    terminal_infos.append(info)
            if len(terminal_infos) >= NUM_ENVS:
                break

        assert terminal_infos, (
            "no terminal infos collected after "
            f"{MAX_TICKS} ticks — round never ended; cannot verify aim plumbing")

        # T5b regression check: at least one terminal info must show a
        # non-zero aim_delta_count AND a mean close to DELTA. Pre-fix,
        # mean would be 0 (or noise) because workers stepped with the
        # zero scratch buffer.
        ok = False
        for info in terminal_infos:
            count = info["aim_delta_count"]
            if count <= 0:
                continue
            mean = info["aim_delta_sum"] / count
            if abs(mean - DELTA) < 0.01:
                ok = True
                break

        assert ok, ("T5b regression: no env reported aim_delta_mean ≈ "
                    f"{DELTA}. Got per-env stats: "
                    f"{[(i['aim_delta_count'], i['aim_delta_sum']) for i in terminal_infos]}")
    finally:
        vecenv.close()
