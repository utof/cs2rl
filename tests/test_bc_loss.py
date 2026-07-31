"""tests/test_bc_loss.py — Batch 6 Task 4 (spec D-6/D-7).

Pins the BC objective itself, independently of any demo data:

  * `bc_loss` equals the hand-rolled maximum-likelihood math (per-head
    categorical CE + Gaussian NLL − λ·entropy) computed straight from the
    policy's own 4-tuple output. If someone "simplifies" the loss to, say,
    an MSE on μ or drops a head, this fails.
  * It decreases on a tiny fit — the loss is not just correct but trainable.
  * `aim_log_std` is FROZEN (spec D-6): σ must survive BC untouched so PPO
    resumes with the exploration noise it expects. The failure mode this
    guards is silent and expensive — a trainable σ collapses toward 0
    because that is the cheapest way to cut a Gaussian NLL.
  * `load_demos` rejects demos whose self-identifying schema disagrees with
    the live constants (spec R7 — `--resume` has no shape guard, so this
    assert is the only thing standing between a stale demo set and a
    confusing shape error deep inside a training run).

The policy is built once per module (it needs a real C env for obs_dim and
max_turn_speed) and deep-copied per test that mutates weights.
"""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import train_bc                                        # noqa: E402
from _action_spec import ACTION_DIM, AIM_DIM           # noqa: E402
from _obs_spec import OBS_DIM                          # noqa: E402

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def policy():
    """One BC-arch policy (identical to the RL policy — build_policy) for the
    whole module. Building it spins up and closes a C env, so it is expensive
    enough to share; tests that take gradient steps deep-copy it first."""
    return train_bc.build_bc_policy(device="cpu", seed=0)


def _fake_batch(n=8, seed=0):
    """Random obs + VALID action labels (per-head ranges, |Δyaw| in band).

    Not demo data on purpose: the loss must be correct for arbitrary
    (obs, action) pairs, and this keeps the test independent of outputs/demos.
    """
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((n, OBS_DIM), dtype=np.float32)
    from _action_spec import ACTION_HEAD_SIZES
    disc = np.stack([rng.integers(0, s, size=n) for s in ACTION_HEAD_SIZES], axis=-1)
    cont = rng.uniform(-0.5, 0.5, size=(n, AIM_DIM)).astype(np.float32)
    return (torch.as_tensor(obs), torch.as_tensor(disc, dtype=torch.int64),
            torch.as_tensor(cont, dtype=torch.float32))


def test_bc_loss_matches_hand_rolled_math(policy):
    """L = mean(−(log_prob_d + log_prob_c)) − λ·mean(H_d + H_c), computed here
    from first principles against the policy's raw logits/μ/σ."""
    obs_t, disc_t, cont_t = _fake_batch()
    lam = 1e-3
    loss, stats = train_bc.bc_loss(policy, obs_t, disc_t, cont_t, entropy_coef=lam)

    with torch.no_grad():
        logits, mu_aim, log_std, _value = policy(obs_t, {})
        # Discrete: per-head log_softmax gathered at the label, summed over heads.
        log_prob_d = torch.zeros(obs_t.shape[0])
        entropy_d = torch.zeros(obs_t.shape[0])
        for head, lg in enumerate(logits):
            lp = torch.log_softmax(lg, dim=-1)
            log_prob_d += lp.gather(-1, disc_t[:, head:head + 1]).squeeze(-1)
            entropy_d += -(lp.exp() * lp).sum(-1)
        # Continuous: plain diagonal-Gaussian NLL. No tanh change-of-variables
        # — the head squashes the MEAN only (spec D-6), so the density is the
        # ordinary Normal one.
        sigma = torch.exp(log_std).expand_as(mu_aim)
        diff = (cont_t - mu_aim) / sigma
        log_prob_c = (-0.5 * diff * diff - torch.log(sigma) - 0.5 * math.log(2 * math.pi)).sum(-1)
        entropy_c = (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(sigma)).sum(-1)

        expected = (-(log_prob_d + log_prob_c).mean() - lam * (entropy_d + entropy_c).mean())

    assert loss.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-6)
    assert stats["nll_d"] == pytest.approx(-log_prob_d.mean().item(), rel=1e-5)
    assert stats["nll_c"] == pytest.approx(-log_prob_c.mean().item(), rel=1e-5)
    # σ_init = 0.1 < 1/√(2πe) ⇒ the Gaussian entropy is NEGATIVE. Documented in
    # _hybrid_sample_logits; asserted here so nobody "fixes" it with a clamp.
    assert stats["entropy_c"] < 0


def test_bc_loss_ignores_entropy_bonus_when_coef_zero(policy):
    """λ=0 must reduce the loss to the pure NLL — pins the sign/placement of
    the entropy term (a `+λH` typo would only show up as slower collapse)."""
    obs_t, disc_t, cont_t = _fake_batch(seed=1)
    loss, stats = train_bc.bc_loss(policy, obs_t, disc_t, cont_t, entropy_coef=0.0)
    assert loss.item() == pytest.approx(stats["nll"], rel=1e-6)
    loss_pos, stats_pos = train_bc.bc_loss(policy, obs_t, disc_t, cont_t, entropy_coef=1e-2)
    # Positive λ REWARDS entropy, so with the same batch the loss must drop by
    # exactly λ·H (H_d + H_c > 0 for a freshly-initialised policy).
    assert loss_pos.item() < loss.item()
    assert loss_pos.item() == pytest.approx(stats["nll"] - 1e-2 *
                                            (stats_pos["entropy_d"] + stats_pos["entropy_c"]),
                                            rel=1e-5)


def test_bc_loss_decreases_on_tiny_fit(policy):
    """Overfit 4 fixed (obs, action) pairs: the loss must fall substantially.
    Uses a deep copy so the module-scoped policy stays at init for the other
    tests."""
    p = copy.deepcopy(policy)
    obs_t, disc_t, cont_t = _fake_batch(n=4, seed=2)
    opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    first, last = None, None
    for step in range(60):
        loss, _ = train_bc.bc_loss(p, obs_t, disc_t, cont_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first - 1.0, f"BC loss barely moved: {first:.3f} → {last:.3f}"


def test_aim_log_std_is_frozen_during_bc(policy):
    """Spec D-6: σ must not train. Checked two ways — no gradient reaches the
    parameter, and its value is bit-identical after real optimizer steps."""
    p = copy.deepcopy(policy)
    obs_t, disc_t, cont_t = _fake_batch(n=8, seed=3)
    before = p.aim_log_std.detach().clone()

    loss, _ = train_bc.bc_loss(p, obs_t, disc_t, cont_t)
    loss.backward()
    assert p.aim_log_std.grad is None or torch.all(p.aim_log_std.grad == 0), \
        "aim_log_std received gradient — the .detach() in bc_loss is gone"

    opt = torch.optim.Adam(p.parameters(), lr=1e-2)
    for _ in range(20):
        loss, _ = train_bc.bc_loss(p, obs_t, disc_t, cont_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert torch.equal(p.aim_log_std.detach(), before), "aim_log_std moved during BC"
    # ...while the rest of the policy DID train (guards against the test
    # passing because nothing at all was updated).
    assert not torch.equal(p.aim_mu.weight.detach(), policy.aim_mu.weight.detach())


def test_train_bc_history_decreases_and_returns_stateless_policy(policy):
    """train_bc's own loop (shuffled per-tick minibatches, D-7) on a synthetic
    DemoSet: history must show the loss falling from first to last epoch."""
    rng = np.random.default_rng(4)
    n = 32
    from _action_spec import ACTION_HEAD_SIZES
    demos = train_bc.DemoSet(
        obs=rng.standard_normal((n, OBS_DIM), dtype=np.float32),
        discrete=np.stack([rng.integers(0, s, size=n) for s in ACTION_HEAD_SIZES], axis=-1),
        continuous=rng.uniform(-0.5, 0.5, size=(n, AIM_DIM)).astype(np.float32),
        dones=np.zeros(n, dtype=bool),
        episodes=[{
            "spawn_area": 0
        }],
        n_files=1,
    )
    p, history = train_bc.train_bc(demos,
                                   policy=copy.deepcopy(policy),
                                   epochs=25,
                                   batch_size=16,
                                   verbose=False)
    assert len(history) == 25
    assert history[-1]["loss"] < history[0]["loss"] - 0.5
    assert not p.training, "train_bc must leave the policy in eval mode"


def test_load_demos_rejects_schema_mismatch(tmp_path):
    """Spec R7/§7: a demo whose declared OBS_DIM differs from the live one is
    a hard error, not a warning — silently training on it would produce a
    checkpoint that explodes inside `--resume`."""
    T = 3
    good = dict(
        obs=np.zeros((T, OBS_DIM), dtype=np.float32),
        discrete_actions=np.zeros((T, ACTION_DIM), dtype=np.int64),
        continuous_actions=np.zeros((T, AIM_DIM), dtype=np.float32),
        dones=np.array([False, False, True]),
        OBS_DIM=OBS_DIM,
        ACTION_DIM=ACTION_DIM,
        AIM_DIM=AIM_DIM,
        seed=0,
        carrier_idx=0,
        spawn_area=0,
        bombsite_area=6,
        tick_count=T,
        git_sha="0" * 40,
        map=train_bc.EXPECTED_MAP,
    )
    np.savez(tmp_path / "ok.npz", **good)
    loaded = train_bc.load_demos(tmp_path)
    assert len(loaded) == T

    bad_dim = dict(good, OBS_DIM=OBS_DIM + 1)
    np.savez(tmp_path / "bad_dim.npz", **bad_dim)
    with pytest.raises(ValueError, match="demo schema"):
        train_bc.load_demos(tmp_path)
    (tmp_path / "bad_dim.npz").unlink()

    bad_map = dict(good, map="de_dust2")
    np.savez(tmp_path / "bad_map.npz", **bad_map)
    with pytest.raises(ValueError, match="map"):
        train_bc.load_demos(tmp_path)
    (tmp_path / "bad_map.npz").unlink()

    # Δyaw labels above the env's clamp mean the recorded action is not the
    # executed one — the labels would be lies.
    bad_cont = dict(good)
    bad_cont["continuous_actions"] = np.full((T, AIM_DIM), 10.0, dtype=np.float32)
    np.savez(tmp_path / "bad_cont.npz", **bad_cont)
    with pytest.raises(ValueError, match="MAX_TURN_SPEED_RAD"):
        train_bc.load_demos(tmp_path)


def test_load_demos_dedupes_byte_identical_episodes(tmp_path):
    """The demo set is 5 unique trajectories × 10 seeds (deterministic spawns,
    plan Task 1 RESULT). Dedupe must drop exact copies and keep everything
    that differs by even one tick."""
    T = 3
    base = dict(
        obs=np.ones((T, OBS_DIM), dtype=np.float32),
        discrete_actions=np.zeros((T, ACTION_DIM), dtype=np.int64),
        continuous_actions=np.zeros((T, AIM_DIM), dtype=np.float32),
        dones=np.array([False, False, True]),
        OBS_DIM=OBS_DIM,
        ACTION_DIM=ACTION_DIM,
        AIM_DIM=AIM_DIM,
        seed=0,
        carrier_idx=0,
        spawn_area=0,
        bombsite_area=6,
        tick_count=T,
        git_sha="0" * 40,
        map=train_bc.EXPECTED_MAP,
    )
    np.savez(tmp_path / "a.npz", **base)
    np.savez(tmp_path / "b.npz", **dict(base, seed=1))                 # identical arrays
    different = dict(base, seed=2)
    different["obs"] = np.full((T, OBS_DIM), 2.0, dtype=np.float32)
    np.savez(tmp_path / "c.npz", **different)

    deduped = train_bc.load_demos(tmp_path, dedupe=True)
    assert len(deduped.episodes) == 2 and deduped.n_duplicates == 1
    assert len(deduped) == 2 * T

    kept_all = train_bc.load_demos(tmp_path, dedupe=False)
    assert len(kept_all.episodes) == 3 and kept_all.n_duplicates == 0
    assert len(kept_all) == 3 * T
