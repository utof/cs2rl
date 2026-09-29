"""tests/test_train_bc_smoke.py — Batch 6 Task 4 end-to-end smoke.

Covers the parts test_bc_loss.py deliberately does not: the real demo files,
the CLI, the checkpoint contract, and the eval harness.

The load-bearing tests here:

  * `test_eval_harness_replays_expert_exactly` drives `rollout_episode` with
    the demo's OWN recorded actions and asserts the rollout reproduces the
    demonstration tick-for-tick, including the obs. That pins two things a
    plant-rate number alone could never distinguish —
      (1) the eval harness itself is correct (env setup, carrier pokes, priming
          step, per-tick stepping); a 0% gate result therefore means the POLICY
          failed, not the harness;
      (2) the eval-time obs (masked by train_bc.mask_idle_agent_blocks) is
          byte-identical to what bc_demos recorded (spec R8). The two masks
          live in different files by design — this test is what keeps them equal.
  * `test_gate_rollout_carries_lstm_state` pins that the GATE measures the
    function PPO inherits. The gate used to run a stateless per-tick forward,
    which reported a 1.000 plant rate for a checkpoint whose carried-state rate
    was 0.200. See train_bc's module docstring.

Demo generation is real (not fixtures) and module-scoped: ~5 episodes of ~70
ticks, a few seconds.
"""
from pathlib import Path

import numpy as np
import pytest

from cs2rl import bc_demos, train_bc

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    """One real seed × 5 carrier slots — every distinct spawn area on the
    simple map (spawns are deterministic centroids, so this is the whole
    start-state distribution)."""
    out = tmp_path_factory.mktemp("bc_demos")
    stats = bc_demos.generate_demos(1, out)
    assert stats["kept"] == 5, f"expected 5/5 kept episodes, got {stats}"
    return out


def test_load_demos_reads_the_real_set(demo_dir):
    demos = train_bc.load_demos(demo_dir)
    assert demos.n_files == 5
    assert demos.n_episodes == 5       # 1 seed ⇒ nothing to dedupe
    assert demos.n_duplicates == 0
    assert demos.spawn_areas() == [0, 1, 2, 3, 4]
    assert len(demos) == sum(e["tick_count"] for e in demos.episodes)
    assert demos.lengths.tolist() == [e["tick_count"] for e in demos.episodes]
    assert demos.obs.dtype == np.float32
    assert demos.discrete.dtype == np.int64
    assert demos.continuous.dtype == np.float32
                                       # One done per episode, on its last tick — which is exactly where
                                       # as_sequences puts each episode's boundary.
    assert demos.dones.sum() == 5


def test_as_sequences_round_trips_the_real_episodes(demo_dir):
    """Every recorded episode must come back out of the padded batch intact,
    with `valid` covering exactly its real ticks. Episode lengths differ by
    ~40 ticks here, so this genuinely exercises the padding."""
    demos = train_bc.load_demos(demo_dir)
    obs, disc, cont, valid = demos.as_sequences()
    assert obs.shape[0] == 5 and obs.shape[1] == int(demos.lengths.max())
    off = 0
    for i, n in enumerate(demos.lengths.tolist()):
        np.testing.assert_array_equal(obs[i, :n], demos.obs[off:off + n])
        np.testing.assert_array_equal(disc[i, :n], demos.discrete[off:off + n])
        np.testing.assert_array_equal(cont[i, :n], demos.continuous[off:off + n])
        assert valid[i, :n].all() and not valid[i, n:].any()
        off += n


def test_cli_trains_and_saves_a_bare_state_dict(demo_dir, tmp_path):
    """2 epochs through main() — the deliverable is the checkpoint CONTRACT:
    a bare state_dict that loads into the RL policy architecture unchanged
    (spec D-8; `--resume` does a plain torch.load + load_state_dict with no
    shape guard, so anything else fails only much later)."""
    out = tmp_path / "bc_warmstart_smoke.pt"
    rc = train_bc.main(
        ["--demos", str(demo_dir), "--out",
         str(out), "--epochs", "2", "--skip-eval"])
    assert rc == 0
    assert out.exists()

    state_dict = torch.load(out, map_location="cpu", weights_only=True)
    assert isinstance(state_dict, dict)
    assert "encoder.0.weight" in state_dict, "not a bare state_dict (wrapped?)"
    assert state_dict["encoder.0.weight"].shape[1] == train_bc.OBS_DIM
    # Arch-identical to RL: build_policy's own module must accept it strictly.
    policy = train_bc.build_bc_policy(device="cpu")
    policy.load_state_dict(state_dict)


def test_eval_harness_replays_expert_exactly(demo_dir, monkeypatch):
    """Feed rollout_episode the demo's recorded actions instead of a policy's:
    the episode must plant on exactly the recorded tick and every obs the
    harness builds must equal the recorded obs (see module docstring)."""
    demo = np.load(sorted(Path(demo_dir).glob("*.npz"))[0])
    seed, carrier = int(demo["seed"]), int(demo["carrier_idx"])
    disc_labels, cont_labels = demo["discrete_actions"], demo["continuous_actions"]

    seen_obs = []

    def replay(policy, obs_row, state=None, action_mask=None, device="cpu"):
        i = len(seen_obs)
        seen_obs.append(obs_row.copy())
        return disc_labels[i], cont_labels[i]

    monkeypatch.setattr(train_bc, "greedy_carrier_action", replay)
    result = train_bc.rollout_episode(policy=None, seed=seed, carrier_idx=carrier)

    assert result["planted"], "expert action replay failed to plant — eval harness is broken"
    assert result["ticks"] == int(demo["tick_count"])
    np.testing.assert_allclose(np.stack(seen_obs), demo["obs"], rtol=0, atol=0)
    np.testing.assert_array_equal(result["first_obs"], demo["obs"][0])


def test_gate_rollout_carries_lstm_state(demo_dir):
    """The gate's rollout must thread ONE LSTM state through the episode
    (`forward_eval` with a persistent dict), and the stateless variant must
    not. Recorded by spying on the state dict `greedy_carrier_action` receives.

    This is the harness-level half of the anti-regression whose numerical half
    lives in test_bc_loss.test_bc_forward_contract_is_ppo_sequence_unroll.
    """
    demo = np.load(sorted(Path(demo_dir).glob("*.npz"))[0])
    seed, carrier = int(demo["seed"]), int(demo["carrier_idx"])
    policy = train_bc.build_bc_policy(device="cpu", seed=0)

    seen_states = []
    real = train_bc.greedy_carrier_action

    def spy(policy, obs_row, state=None, action_mask=None, device="cpu"):
        seen_states.append(None if state is None else (state.get("lstm_h") is not None))
        return real(policy, obs_row, state=state, action_mask=action_mask, device=device)

    import unittest.mock as mock
    with mock.patch.object(train_bc, "greedy_carrier_action", spy):
        train_bc.rollout_episode(policy, seed=seed, carrier_idx=carrier, max_ticks=6)
        carried = list(seen_states)
        seen_states.clear()
        train_bc.rollout_episode(policy,
                                 seed=seed,
                                 carrier_idx=carrier,
                                 max_ticks=6,
                                 carry_state=False)
        stateless = list(seen_states)

    # carry_state=True: an empty dict on tick 0, populated with lstm_h after.
    assert carried[0] is False, "gate rollout started with a pre-filled LSTM state"
    assert all(carried[1:]), "gate rollout did not carry lstm_h between ticks"
    # carry_state=False: no state dict at all — a zero-state T=1 forward.
    assert all(s is None for s in stateless)


def test_greedy_action_respects_the_c_action_masks(demo_dir):
    """A masked-out bin must be unreachable by argmax (review finding 7).

    Built by handing `greedy_carrier_action` a mask that forbids the bin the
    unmasked greedy action would have chosen: the masked call must return a
    different bin for that head, and it must be one the mask allows.
    """
    from cs2rl.spec.action import ACTION_HEAD_SIZES
    policy = train_bc.build_bc_policy(device="cpu", seed=0)
    obs_row = np.load(sorted(Path(demo_dir).glob("*.npz"))[0])["obs"][0].astype(np.float32)

    unmasked, _ = train_bc.greedy_carrier_action(policy, obs_row)
    mask = np.ones(train_bc.ACTION_MASK_DIM, dtype=np.int8)
    off = 0
    for head, size in enumerate(ACTION_HEAD_SIZES):
        mask[off + int(unmasked[head])] = 0            # forbid the greedy pick
        off += size

    masked, _ = train_bc.greedy_carrier_action(policy, obs_row, action_mask=mask)
    off = 0
    for head, size in enumerate(ACTION_HEAD_SIZES):
        assert masked[head] != unmasked[head], f"head {head} ignored its mask"
        assert mask[off + int(masked[head])] == 1, f"head {head} picked a masked-out bin"
        off += size


def test_rollout_episode_respects_the_tick_budget():
    """A policy that always emits action 0 (stand still) must burn exactly the
    tick budget and report planted=False — the gate must be able to FAIL."""

    class DoNothing:
        pass

    def stand_still(policy, obs_row, state=None, action_mask=None, device="cpu"):
        return (np.zeros(train_bc.ACTION_DIM,
                         dtype=np.int64), np.zeros(train_bc.AIM_DIM, dtype=np.float32))

    import unittest.mock as mock
    with mock.patch.object(train_bc, "greedy_carrier_action", stand_still):
        result = train_bc.rollout_episode(DoNothing(), seed=0, carrier_idx=0, max_ticks=12)
    assert result["planted"] is False
    assert result["ticks"] == 12


def test_start_state_probe_collapses_seeds_to_five_states():
    """Spawns are deterministic centroids and seeds only permute which agent
    gets which area (plan Task 1 RESULT), so 3 seeds × 5 carriers is 5 distinct
    start states — the fact that makes the gate's dedupe honest rather than a
    shortcut. If this ever fails, the gate denominator must go back up."""
    keys = {train_bc.probe_start_state(s, c) for s in range(3) for c in range(5)}
    assert len(keys) == 5, f"expected 5 distinct start states across 3 seeds, got {len(keys)}"


def test_eval_plant_rate_aggregates_over_distinct_states(monkeypatch):
    """eval_plant_rate's arithmetic with the (expensive) rollout stubbed: the
    denominator is DISTINCT start states, and the duplicates it dropped are
    reported rather than silently folded into the average (review finding 6)."""
    calls = []

    def fake_rollout(policy, seed, carrier_idx, device="cpu", carry_state=True):
        calls.append((seed, carrier_idx))
        planted = carrier_idx < 2      # 2 of every 5 slots plant
        return {"seed": seed, "carrier_idx": carrier_idx, "planted": planted, "ticks": 70}

    # Deterministic spawns: the start state depends only on the carrier slot.
    monkeypatch.setattr(train_bc, "probe_start_state", lambda s, c: bytes([c]))
    monkeypatch.setattr(train_bc, "rollout_episode", fake_rollout)
    summary = train_bc.eval_plant_rate(policy=None, seeds=range(3), verbose=False, label="x")
    assert len(calls) == 5, "ran duplicate start states"
    assert summary["episodes"] == 5
    assert summary["grid_size"] == 15
    assert summary["duplicates"] == 10
    assert summary["planted"] == 2
    assert summary["plant_rate"] == pytest.approx(2 / 5)
    assert summary["ticks_median"] == 70

    calls.clear()
    no_dedupe = train_bc.eval_plant_rate(policy=None,
                                         seeds=range(3),
                                         verbose=False,
                                         label="x",
                                         dedupe_start_states=False)
    assert len(calls) == 15 and no_dedupe["episodes"] == 15 and no_dedupe["duplicates"] == 0
