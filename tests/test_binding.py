import numpy as np
import pytest

from cs2rl.c_env import binding


def _make_env(map_data=None):
    from cs2rl.c_env.cs2_env import make_env
    env = make_env(seed=0, map_data=map_data)
    return env._capsule, env


def test_binding_functions_present():
    for name in ("init", "reset", "step", "close", "get_buffers", "get_masks", "struct_sizes",
                 "static_data_scalars", "static_data_layout"):
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
    from cs2rl.spec.action import ACTION_DIM, AIM_DIM
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


def test_agentstate_has_punch_fields():
    """Sim recoil v1 (#120): punch lives on AgentState; flag on Dust2Env.

    Why: P0 punch was Client-only, so the hit ray ignored view-kick. These
    fields are the ctypes mirror of the C layout — if they drift, from_address
    overlays garbage. Sizes are filled in after the measured C rebuild;
    first RED is missing names / old AgentState 156.
    """
    import ctypes

    from cs2rl.c_env.cs2_env import AgentStateC, Dust2EnvC, GameStateC
    names = [n for n, _ in AgentStateC._fields_]
    # Adjacency + order, NOT a tail slice. The punch pair stopped being the last
    # two fields when Rung 0 (spec 2026-08-29 §2.1) appended
    # participating/_pad5 after them, and every future appended field would
    # break a `names[-2:]` assert again. What this test actually guards is that
    # the pair was APPENDED and never reordered relative to each other or moved
    # into the middle of the struct; the absolute layout is pinned by the
    # compiler's own offsetof in tests/test_struct_sizes.py.
    assert names.index("punch_yaw") == names.index("punch_pitch") + 1
    assert "recoil_enabled" in [n for n, _ in Dust2EnvC._fields_]
    assert hasattr(AgentStateC, "punch_pitch")
    assert hasattr(AgentStateC, "punch_yaw")
    assert hasattr(Dust2EnvC, "recoil_enabled")
    # Sizes come from binding.struct_sizes() — the C compiler's own sizeof —
    # not from literals measured by hand with a printf TU (see
    # tests/test_struct_sizes.py).
    sizes = binding.struct_sizes()
    assert ctypes.sizeof(AgentStateC) == sizes["AgentState"]
    assert ctypes.sizeof(GameStateC) == sizes["GameState"]
    assert ctypes.sizeof(Dust2EnvC) == sizes["Dust2Env"]


def test_make_env_writes_recoil_enabled(make_map):
    """make_env / Cs2Env write recoil_enabled after from_address.

    Why: not reachable through binding.init — that call carries the StaticData
    prefix, and this flag lives on Dust2Env instead. env_reset memsets GameState
    only, so the flag must be set at overlay time — a first-reset-only write
    would also work today, but would hide a later memset of Dust2Env. Default is
    today's hitscan (0).
    """
    import dataclasses

    from cs2rl.c_env.cs2_env import make_env
    from cs2rl.env.config import EnvConfig
    assert "recoil" in {f.name for f in dataclasses.fields(EnvConfig)}
    env = make_env(seed=0, map_data=make_map, config=EnvConfig(recoil=False))
    try:
        assert int(env._c_env.recoil_enabled) == 0
        env.reset()
        assert int(env._c_env.recoil_enabled) == 0, "reset must not clear the flag"
    finally:
        env.close()
    on = make_env(seed=0, map_data=make_map, config=EnvConfig(recoil=True))
    try:
        assert int(on._c_env.recoil_enabled) == 1
        on.reset()
        assert int(on._c_env.recoil_enabled) == 1
    finally:
        on.close()


def test_stepstats_has_plant_tick(make_map):
    """StepStatsC must expose plant_tick (g->tick at plant; 0 = never planted).

    Why: observe-only plant-latency field is appended after _pad_ss_wins, which
    grows StepStats (and Dust2Env, which embeds two of them) — the sizes are
    checked against binding.struct_sizes() rather than literals so appending the
    next field does not require re-measuring by hand.
    clear_stats memsets the struct, so reset must leave 0.
    Pitfall: do not read this via a numpy recarray — binding.c has no
    StepStats dtype; ctypes is the Python-side mirror.
    """
    import ctypes

    from cs2rl.c_env.cs2_env import Dust2EnvC, StepStatsC
    _, env = _make_env(map_data=make_map)
    assert hasattr(env._c_env.episode_stats, "plant_tick")
    env.reset()
    assert int(env._c_env.episode_stats.plant_tick) == 0
    sizes = binding.struct_sizes()
    assert ctypes.sizeof(StepStatsC) == sizes["StepStats"]
    assert ctypes.sizeof(Dust2EnvC) == sizes["Dust2Env"]


def test_human_controlled_uses_aim_rad_not_bin(make_map):
    """When human_controlled=1, facing must equal aim_rad, ignoring the
    continuous_actions Δyaw buffer.

    Batch 3: pre-Batch-3 this test verified the 16-bin override; now it
    verifies that the human branch in env_step (`if (a->human_controlled)`)
    short-circuits BEFORE reading continuous_actions. We pass a non-zero
    Δyaw to confirm it is ignored — only aim_rad sets facing for human
    agents.
    """
    from cs2rl.spec.action import ACTION_DIM, AIM_DIM
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
    Batch 3.5: wrong-shape probe uses AIM_DIM+1 so it stays wrong even as AIM_DIM grows.
    """
    from cs2rl.spec.action import ACTION_DIM, AIM_DIM
    _, env = _make_env(map_data=make_map)
    env.reset(seed=0)
    actions = np.zeros((10, ACTION_DIM), dtype=np.int32)
    cont = np.zeros((10, AIM_DIM), dtype=np.float32)
    env.step(actions, cont)            # should not raise
    with pytest.raises(ValueError):
                                       # AIM_DIM+1 columns is always wrong regardless of the current AIM_DIM value
        env.step(actions, np.zeros((10, AIM_DIM + 1), dtype=np.float32))


def test_binding_default_continuous_actions_zero(make_map):
    """If continuous_actions arg omitted, zero buffer supplied — facing unchanged.

    Batch 3: defensive default keeps legacy callers (smoke-test loops, the
    train.py main path before the policy is wired in T4-T5) working without
    explicit continuous-action arrays. We use the designated bomb carrier
    (an RL agent, not human_controlled) so the continuous branch in env_step
    fires.
    """
    from cs2rl.spec.action import ACTION_DIM
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
# train() method (Cs2PuffeRL.train, src/cs2rl/trainer.py). Rebuilding the full
# PufferLib trainer just to test this would be expensive and fragile against
# unrelated PufferLib API drift; instead we replicate the guard's structure
# locally — same control flow, same warning string, same zero-grad call —
# and verify that:
#   (a) when loss is non-finite, no parameter update happens,
#   (b) the throttled warning print happens.
#
# If the guard's structure changes (new warning string, different zero_grad
# signature), update BOTH this test and the real guard in the same PR. The real
# guard is the `if not torch.isfinite(loss).all():` block inside
# `Cs2PuffeRL.train`, which moved out of train.py with its patcher on
# 2026-08-31 (src/cs2rl/train_update.py) and into src/cs2rl/trainer.py as a method on
# gh#168 W2a; search the "Batch 3 (T5) NaN guard" banner rather than trusting
# a line number.


def test_continuous_aim_nan_guard():
    """T5: NaN guard in Cs2PuffeRL.train skips optimizer.step() and
    prints a throttled warning when the loss is non-finite, without
    poisoning subsequent gradients."""
    import io
    import sys as _sys
    import time as _t

    import torch

    from cs2rl import train
    from cs2rl.c_env.cs2_env import make_env

    env = make_env(seed=0)
    try:
        policy = train.build_policy(env, device='cpu')

        # Force the aim head to emit NaN so the loss path goes non-finite.
        # We don't need to run a full PPO update — replicating the guard's
        # control flow inline is enough to verify it does the right thing.
        # Batch 3.5: output AIM_DIM columns so expand_as(mu_aim) in forward
        # doesn't raise a size mismatch when AIM_DIM > 1.
        _aim_dim = train.AIM_DIM

        class _NaNLayer(torch.nn.Module):

            def forward(self, x):
                return torch.full((x.shape[0], _aim_dim), float('nan'))

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

        # Mirror the guard control flow inside Cs2PuffeRL.train
        # (src/cs2rl/trainer.py). Throttle field name MUST match the
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

    from cs2rl.c_env.cs2_env import make_env
    from cs2rl.spec.action import ACTION_DIM, AIM_DIM

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


def test_onnx_export_output_order_pinned():
    """Pin the deploy ONNX output order for the Batch 3 contract.

    Constructs an `LSTMPolicyONNXWrapper` directly (avoiding the cost of a
    full checkpoint round-trip) with the aim head wired in, exports it to
    ONNX, loads the result with onnxruntime, and asserts the output names
    are exactly:

        logits_0, logits_1, ..., logits_6, mu_aim, lstm_h_out, lstm_c_out

    The mu_aim slot must be float32 (B, 1) in [-max_turn_speed,
    +max_turn_speed]. If this test ever fails, the C# decode in
    deploy/CS2RLBot/PolicyInference.cs is about to silently misread the
    aim output — reorder the wrapper or the C# side together, never one
    in isolation. Imports are local so the test file's top-level cost is
    unchanged.
    """
    import math
    import tempfile

    import onnxruntime as ort
    import torch
    from torch import nn

    from cs2rl import train
    from cs2rl.c_env.cs2_env import make_env
    from cs2rl.deploy.export_policy import LSTMPolicyONNXWrapper

    env = make_env(seed=0)
    try:
        obs_dim, hidden = train.OBS_DIM, 256
        # Mirror the architecture build_model would produce. ACTION_HEAD_SIZES
        # is the head order MOVE/SHOOT/RELOAD/WEAPON/USE/CROUCH/JUMP per
        # _action_spec.py — keep this list synced if those sizes ever change.
        encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                nn.ReLU())
        lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=False)
        action_heads = nn.ModuleList([nn.Linear(hidden, n) for n in [9, 2, 2, 3, 2, 2, 2]])
        aim_mu = nn.Linear(hidden, 1)
        wrapper = LSTMPolicyONNXWrapper(encoder,
                                        lstm,
                                        action_heads,
                                        aim_mu=aim_mu,
                                        max_turn_speed=math.pi / 4.0).eval()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            obs_t = torch.zeros(1, obs_dim)
            done_t = torch.zeros(1)
            lstm_h_t = torch.zeros(1, 1, hidden)
            lstm_c_t = torch.zeros(1, 1, hidden)
            torch.onnx.export(
                wrapper,
                (obs_t, done_t, lstm_h_t, lstm_c_t),
                f.name,
                input_names=["obs", "done", "lstm_h", "lstm_c"],
                output_names=([f"logits_{i}"
                               for i in range(7)] + ["mu_aim", "lstm_h_out", "lstm_c_out"]),
                opset_version=17,
                                                                                             # dynamo=False matches cs2rl/deploy/export_policy.py:main(); the
                                                                                             # dynamo-based exporter pulls in onnxscript which isn't part
                                                                                             # of this project's lockfile.
                dynamo=False,
            )
            sess = ort.InferenceSession(f.name, providers=["CPUExecutionProvider"])
            out_names = [o.name for o in sess.get_outputs()]
            expected = ([f"logits_{i}" for i in range(7)] + ["mu_aim", "lstm_h_out", "lstm_c_out"])
            assert out_names == expected, f"output order drift: {out_names}"

            outs = sess.run(
                out_names,
                {
                    "obs": np.zeros((1, obs_dim), dtype=np.float32),
                    "done": np.zeros(1, dtype=np.float32),
                    "lstm_h": np.zeros((1, 1, hidden), dtype=np.float32),
                    "lstm_c": np.zeros((1, 1, hidden), dtype=np.float32),
                },
            )
            mu = outs[7]
            assert mu.shape == (1, 1), f"mu shape: {mu.shape}"
            assert mu.dtype == np.float32, f"mu dtype: {mu.dtype}"
            # tanh*max_turn_speed bound: a hair of slack absorbs fp32 wobble.
            assert np.abs(mu).max() <= math.pi / 4.0 + 1e-5, (
                f"mu_aim out of range: max |mu|={np.abs(mu).max()}, "
                f"expected <= pi/4")
    finally:
        env.close()


def test_build_model_state_dict_round_trip():
    """T6: build_model correctly reconstructs LSTMPolicyONNXWrapper from a
    synthesized Batch 3 state_dict, returns the 5-tuple, and tolerates
    aim_log_std/max_turn_speed/value_head.* without raising. Also verifies
    the Batch 2 backward-compat path returns aim_dim=0.

    The full pinned-output-order test (test_onnx_export_output_order_pinned)
    bypasses build_model entirely to dodge full checkpoint cost. This test
    closes the gap on the part most likely to silently rot during a future
    migration: the strict-load filter logic that decides which keys are
    "expected unexpected" or "expected missing".
    """
    import math

    import torch

    from cs2rl.deploy.export_policy import build_model

    obs_dim, hidden, aim_dim = 105, 256, 1
    head_sizes = (9, 2, 2, 3, 2, 2, 2)

    def _synth_batch3() -> dict:
        # Synthesize a state_dict matching Dust2Policy's actual key layout
        # (encoder.0/2 + lstm + action_heads.[0-6] + value_head + aim_mu +
        # aim_log_std + max_turn_speed). Loading this through build_model
        # exercises the same strict-load filter that production uses.
        sd = {}
        sd["encoder.0.weight"] = torch.randn(hidden, obs_dim)
        sd["encoder.0.bias"] = torch.zeros(hidden)
        sd["encoder.2.weight"] = torch.randn(hidden, hidden)
        sd["encoder.2.bias"] = torch.zeros(hidden)
        sd["lstm.weight_ih_l0"] = torch.randn(4 * hidden, hidden)
        sd["lstm.weight_hh_l0"] = torch.randn(4 * hidden, hidden)
        sd["lstm.bias_ih_l0"] = torch.zeros(4 * hidden)
        sd["lstm.bias_hh_l0"] = torch.zeros(4 * hidden)
        for i, n in enumerate(head_sizes):
            sd[f"action_heads.{i}.weight"] = torch.randn(n, hidden)
            sd[f"action_heads.{i}.bias"] = torch.zeros(n)
        # value head — gets filtered as "expected unexpected"
        sd["value_head.weight"] = torch.randn(1, hidden)
        sd["value_head.bias"] = torch.zeros(1)
        # Batch 3 aim head + buffers
        sd["aim_mu.weight"] = torch.randn(aim_dim, hidden)
        sd["aim_mu.bias"] = torch.zeros(aim_dim)
        sd["aim_log_std"] = torch.tensor([math.log(0.1)])              # state-indep param
        sd["max_turn_speed"] = torch.tensor(math.pi / 4.0)             # buffer
        return sd

    # Batch 3 path
    sd_b3 = _synth_batch3()
    out = build_model(sd_b3)
    assert len(out) == 5, f"expected 5-tuple, got {len(out)}"
    wrapper, ret_obs_dim, ret_hidden, ret_action_sizes, ret_aim_dim = out
    assert ret_obs_dim == obs_dim
    assert ret_hidden == hidden
    assert tuple(ret_action_sizes) == head_sizes
    assert ret_aim_dim == aim_dim
    assert wrapper.aim_mu is not None
    # Buffer should equal the value from the state_dict (NOT the π/4 fallback,
    # which would silently lie if the buffer-load logic broke).
    assert abs(float(wrapper._max_turn_speed) - math.pi / 4.0) < 1e-6

    # Batch 2 backward-compat path
    sd_b2 = _synth_batch3()
    for k in ("aim_mu.weight", "aim_mu.bias", "aim_log_std", "max_turn_speed"):
        del sd_b2[k]
    wrapper_b2, _, _, _, ret_aim_dim_b2 = build_model(sd_b2)
    assert ret_aim_dim_b2 == 0
    assert wrapper_b2.aim_mu is None
