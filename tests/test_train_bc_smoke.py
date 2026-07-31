"""tests/test_train_bc_smoke.py — Batch 6 Task 4 end-to-end smoke.

Covers the parts test_bc_loss.py deliberately does not: the real demo files,
the CLI, the checkpoint contract, and the eval harness.

The load-bearing test here is `test_eval_harness_replays_expert_exactly`: it
drives `rollout_episode` with the demo's OWN recorded actions and asserts the
rollout reproduces the demonstration tick-for-tick, including the obs. That
pins two things a plant-rate number alone could never distinguish —

  (1) the eval harness itself is correct (env setup, carrier pokes, priming
      step, per-tick stepping); a 0% gate result therefore means the POLICY
      failed, not the harness;
  (2) the eval-time obs (masked by train_bc.mask_idle_agent_blocks) is
      byte-identical to what gen_bc_demos recorded (spec R8). The two masks
      live in different files by design — this test is what keeps them equal.

Demo generation is real (not fixtures) and module-scoped: ~5 episodes of ~70
ticks, a few seconds.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import gen_bc_demos                    # noqa: E402

import train_bc                        # noqa: E402

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    """One real seed × 5 carrier slots — every distinct spawn area on the
    simple map (spawns are deterministic centroids, so this is the whole
    start-state distribution)."""
    out = tmp_path_factory.mktemp("bc_demos")
    stats = gen_bc_demos.generate_demos(1, out)
    assert stats["kept"] == 5, f"expected 5/5 kept episodes, got {stats}"
    return out


def test_load_demos_reads_the_real_set(demo_dir):
    demos = train_bc.load_demos(demo_dir)
    assert demos.n_files == 5
    assert len(demos.episodes) == 5    # 1 seed ⇒ nothing to dedupe
    assert demos.n_duplicates == 0
    assert demos.spawn_areas() == [0, 1, 2, 3, 4]
    assert len(demos) == sum(e["tick_count"] for e in demos.episodes)
    assert demos.obs.dtype == np.float32
    assert demos.discrete.dtype == np.int64
    assert demos.continuous.dtype == np.float32
                                       # dones only mark episode ends; per-tick BC never crosses them (D-7).
    assert demos.dones.sum() == 5


def test_cli_trains_and_saves_a_bare_state_dict(demo_dir, tmp_path):
    """2 epochs through main() — the deliverable is the checkpoint CONTRACT:
    a bare state_dict that loads into the RL policy architecture unchanged
    (spec D-8; `--resume` does a plain torch.load + load_state_dict with no
    shape guard, so anything else fails only much later)."""
    out = tmp_path / "bc_warmstart_smoke.pt"
    rc = train_bc.main([
        "--demos",
        str(demo_dir), "--out",
        str(out), "--epochs", "2", "--batch-size", "64", "--skip-eval"
    ])
    assert rc == 0
    assert out.exists()

    state_dict = torch.load(out, map_location="cpu")
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

    def replay(policy, obs_row, device="cpu"):
        i = len(seen_obs)
        seen_obs.append(obs_row.copy())
        return disc_labels[i], cont_labels[i]

    monkeypatch.setattr(train_bc, "greedy_carrier_action", replay)
    result = train_bc.rollout_episode(policy=None, seed=seed, carrier_idx=carrier)

    assert result["planted"], "expert action replay failed to plant — eval harness is broken"
    assert result["ticks"] == int(demo["tick_count"])
    np.testing.assert_allclose(np.stack(seen_obs), demo["obs"], rtol=0, atol=0)


def test_rollout_episode_respects_the_tick_budget():
    """A policy that always emits action 0 (stand still) must burn exactly the
    tick budget and report planted=False — the gate must be able to FAIL."""

    class DoNothing:
        pass

    def stand_still(policy, obs_row, device="cpu"):
        return (np.zeros(train_bc.ACTION_DIM,
                         dtype=np.int64), np.zeros(train_bc.AIM_DIM, dtype=np.float32))

    import unittest.mock as mock
    with mock.patch.object(train_bc, "greedy_carrier_action", stand_still):
        result = train_bc.rollout_episode(DoNothing(), seed=0, carrier_idx=0, max_ticks=12)
    assert result["planted"] is False
    assert result["ticks"] == 12


def test_eval_plant_rate_aggregates(monkeypatch):
    """eval_plant_rate's arithmetic, with the (expensive) rollout stubbed."""
    calls = []

    def fake_rollout(policy, seed, carrier_idx, device="cpu"):
        calls.append((seed, carrier_idx))
        planted = carrier_idx < 2      # 2 of every 5 slots plant
        return {"seed": seed, "carrier_idx": carrier_idx, "planted": planted, "ticks": 70}

    monkeypatch.setattr(train_bc, "rollout_episode", fake_rollout)
    summary = train_bc.eval_plant_rate(policy=None, seeds=range(3), verbose=False, label="x")
    assert len(calls) == 15
    assert summary["episodes"] == 15
    assert summary["planted"] == 6
    assert summary["plant_rate"] == pytest.approx(6 / 15)
    assert summary["ticks_median"] == 70
