"""tests/test_bc_loss.py — Batch 6 Task 4 (spec D-6).

Pins the BC objective itself, independently of any demo data:

  * `bc_loss` equals the hand-rolled maximum-likelihood math (per-head
    categorical CE + Gaussian NLL − λ·entropy) computed straight from the
    policy's own 4-tuple output. If someone "simplifies" the loss to, say,
    an MSE on μ or drops a head, this fails.
  * It runs over SEQUENCES through PPO's own forward path, and right-padding
    ticks are excluded from both means.
  * **BC's forward contract equals PPO's** — `test_bc_forward_contract_*`.
    This is the anti-regression for the bug that motivated the rewrite: BC
    used to train and gate a STATELESS function (`policy(x, {})` per tick from
    zero LSTM state) while PPO's rollout carries lstm_h/lstm_c tick-to-tick and
    its update unrolls whole segments. The shipped stateless checkpoint scored
    a 1.000 stateless plant rate and a 0.200 carried-state one. If anyone
    reverts BC to per-tick training, these two tests fail.
  * It decreases on a tiny fit — the loss is not just correct but trainable.
  * `aim_log_std` is FROZEN (spec D-6): σ must survive BC untouched so PPO
    resumes with the exploration noise it expects. The failure mode this
    guards is silent and expensive — a trainable σ collapses toward 0
    because that is the cheapest way to cut a Gaussian NLL.
  * `load_demos` rejects demos whose self-identifying schema disagrees with
    the live constants (spec R7 — `--resume` has no shape guard, so this
    assert is the only thing standing between a stale demo set and a
    confusing shape error deep inside a training run), INCLUDING a real
    git-sha provenance check rather than a length check.
  * `build_bc_policy` is reproducible under `--seed`.

The policy is built once per module (it needs a real C env for obs_dim and
max_turn_speed) and deep-copied per test that mutates weights.
"""
import copy
import math

import numpy as np
import pytest

from cs2rl import train_bc
from cs2rl.spec.action import ACTION_DIM, AIM_DIM
from cs2rl.spec.obs import OBS_DIM

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def policy():
    """One BC-arch policy (identical to the RL policy — build_policy) for the
    whole module. Building it spins up and closes a C env, so it is expensive
    enough to share; tests that take gradient steps deep-copy it first."""
    return train_bc.build_bc_policy(device="cpu", seed=0)


def _fake_batch(b=2, t=4, seed=0, pad=0):
    """Random obs sequences + VALID action labels (per-head ranges, |Δyaw| in
    band), shaped like a BC minibatch: (B, T, ...) plus a (B, T) validity mask.

    Not demo data on purpose: the loss must be correct for arbitrary
    (obs, action) pairs, and this keeps the test independent of outputs/demos.
    `pad` marks that many trailing ticks of the LAST row invalid, mimicking a
    short episode right-padded up to T_max.
    """
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((b, t, OBS_DIM), dtype=np.float32)
    from cs2rl.spec.action import ACTION_HEAD_SIZES
    disc = np.stack([rng.integers(0, s, size=(b, t)) for s in ACTION_HEAD_SIZES], axis=-1)
    cont = rng.uniform(-0.5, 0.5, size=(b, t, AIM_DIM)).astype(np.float32)
    valid = np.ones((b, t), dtype=bool)
    if pad:
        valid[-1, t - pad:] = False
    return (torch.as_tensor(obs), torch.as_tensor(disc, dtype=torch.int64),
            torch.as_tensor(cont, dtype=torch.float32), torch.as_tensor(valid))


def _hand_rolled(policy, obs_t, disc_t, cont_t, valid_t):
    """Reference log-probs/entropies from first principles, on the FLATTENED
    (B*T) rows the policy emits. Returns (log_prob_d, log_prob_c, H_d, H_c, w)
    where w is the flattened validity weight."""
    flat_disc = disc_t.reshape(-1, disc_t.shape[-1])
    flat_cont = cont_t.reshape(-1, cont_t.shape[-1])
    with torch.no_grad():
        logits, mu_aim, log_std, _value = policy(obs_t, {})
        n = flat_disc.shape[0]
        # Discrete: per-head log_softmax gathered at the label, summed over heads.
        log_prob_d = torch.zeros(n)
        entropy_d = torch.zeros(n)
        for head, lg in enumerate(logits):
            lp = torch.log_softmax(lg, dim=-1)
            log_prob_d += lp.gather(-1, flat_disc[:, head:head + 1]).squeeze(-1)
            entropy_d += -(lp.exp() * lp).sum(-1)
        # Continuous: plain diagonal-Gaussian NLL. No tanh change-of-variables
        # — the head squashes the MEAN only (spec D-6), so the density is the
        # ordinary Normal one.
        sigma = torch.exp(log_std).expand_as(mu_aim)
        diff = (flat_cont - mu_aim) / sigma
        log_prob_c = (-0.5 * diff * diff - torch.log(sigma) - 0.5 * math.log(2 * math.pi)).sum(-1)
        entropy_c = (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(sigma)).sum(-1)
    w = (valid_t.reshape(-1).float() if valid_t is not None else torch.ones(n))
    return log_prob_d, log_prob_c, entropy_d, entropy_c, w


def test_bc_loss_matches_hand_rolled_math(policy):
    """L = mean_valid(−(log_prob_d + log_prob_c)) − λ·mean_valid(H_d + H_c),
    computed here from first principles against the policy's raw logits/μ/σ."""
    obs_t, disc_t, cont_t, valid_t = _fake_batch()
    lam = 1e-3
    loss, stats = train_bc.bc_loss(policy, obs_t, disc_t, cont_t, valid=valid_t, entropy_coef=lam)

    lp_d, lp_c, h_d, h_c, w = _hand_rolled(policy, obs_t, disc_t, cont_t, valid_t)
    mean = lambda x: (x * w).sum() / w.sum()           # noqa: E731
    expected = -(mean(lp_d) + mean(lp_c)) - lam * (mean(h_d) + mean(h_c))

    assert loss.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-6)
    assert stats["nll_d"] == pytest.approx(-mean(lp_d).item(), rel=1e-5)
    assert stats["nll_c"] == pytest.approx(-mean(lp_c).item(), rel=1e-5)
    # σ_init = 0.1 < 1/√(2πe) ⇒ the Gaussian entropy is NEGATIVE. Documented in
    # _hybrid_sample_logits; asserted here so nobody "fixes" it with a clamp.
    assert stats["entropy_c"] < 0


def test_bc_loss_excludes_padding_ticks(policy):
    """Right-padded ticks must not enter either mean.

    Checked by construction, not by trusting the mask: the padded batch's loss
    must equal the hand-rolled mean over the VALID rows only, and must differ
    from the same batch scored with valid=None. A mask that silently did
    nothing would pass the first assert and fail the second.
    """
    obs_t, disc_t, cont_t, valid_t = _fake_batch(b=3, t=6, seed=7, pad=4)
    masked, _ = train_bc.bc_loss(policy, obs_t, disc_t, cont_t, valid=valid_t, entropy_coef=0.0)
    unmasked, _ = train_bc.bc_loss(policy, obs_t, disc_t, cont_t, valid=None, entropy_coef=0.0)

    lp_d, lp_c, _h_d, _h_c, w = _hand_rolled(policy, obs_t, disc_t, cont_t, valid_t)
    expected = -((lp_d + lp_c) * w).sum() / w.sum()
    assert masked.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-6)
    assert masked.item() != pytest.approx(unmasked.item(), rel=1e-4)


def test_bc_forward_contract_is_ppo_sequence_unroll(policy):
    """THE anti-regression (see module docstring).

    The forward BC trains through — `policy(x, {})` on a (1, T, OBS_DIM)
    sequence, i.e. `Dust2Policy.forward` → `_lstm_bptt` — must produce exactly
    what PPO's ROLLOUT produces: `forward_eval` walked tick-by-tick with the
    LSTM state carried in the state dict. Same weights, same obs, same numbers.

    The second half of the test is what gives it teeth: the STATELESS sweep
    (`policy(x_t, {})` per tick, zero state every time — what BC used to train
    and gate on) must NOT match. If a refactor makes the sequence path
    collapse to per-tick stateless behaviour, that assert fires.
    """
    t = 12
    rng = np.random.default_rng(11)
    obs = torch.as_tensor(rng.standard_normal((1, t, OBS_DIM), dtype=np.float32))

    with torch.no_grad():
        seq_logits, seq_mu, _ls, seq_value = policy(obs, {})

        state = {}
        carried_logits, carried_mu, carried_value = [], [], []
        stateless_mu = []
        for i in range(t):
            x = obs[:, i, :]
            lg, mu, _s, v = policy.forward_eval(x, state)
            carried_logits.append([h.clone() for h in lg])
            carried_mu.append(mu.clone())
            carried_value.append(v.clone())
            lg0, mu0, _s0, _v0 = policy(x, {})
            stateless_mu.append(mu0.clone())

    # B=1, so the flat (B*T, ...) sequence output is just tick 0..T-1 in order.
    for i in range(t):
        for head in range(len(seq_logits)):
            torch.testing.assert_close(seq_logits[head][i],
                                       carried_logits[i][head][0],
                                       rtol=1e-5,
                                       atol=1e-6)
        torch.testing.assert_close(seq_mu[i], carried_mu[i][0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(seq_value[i], carried_value[i][0], rtol=1e-5, atol=1e-6)

    # ...and the stateless per-tick function is a DIFFERENT function. Tick 0
    # necessarily agrees (both start from zero state); everything after must
    # diverge, or the LSTM is not carrying state at all.
    torch.testing.assert_close(seq_mu[0], stateless_mu[0][0], rtol=1e-5, atol=1e-6)
    diverged = sum(
        int(not torch.allclose(seq_mu[i], stateless_mu[i][0], rtol=1e-4, atol=1e-5))
        for i in range(1, t))
    assert diverged == t - 1, (
        "the stateless per-tick forward matches the sequence unroll — the LSTM is not "
        "carrying state, so BC and PPO would again optimise different functions")


def test_train_bc_feeds_sequences_not_shuffled_ticks(policy):
    """`train_bc` must hand the policy 3D (B, T, OBS_DIM) batches.

    A forward pre-hook records the rank of every input tensor the policy sees
    during a short fit. Per-tick BC (the reverted-bug shape) would show up here
    as rank 2, and the LSTM would never receive a through-time gradient.
    """
    p = copy.deepcopy(policy)
    ranks = []
    handle = p.register_forward_pre_hook(lambda _m, inputs: ranks.append(inputs[0].ndim))
    try:
        train_bc.train_bc(_seq_demoset(), policy=p, epochs=3, verbose=False)
    finally:
        handle.remove()
    assert ranks, "policy.forward was never called"
    assert set(ranks) == {3}, f"BC fed the policy rank-{sorted(set(ranks))} inputs, expected 3D"


def _seq_demoset(seed=4, lengths=(5, 3)):
    """Synthetic multi-episode DemoSet with UNEQUAL episode lengths, so every
    consumer has to deal with padding."""
    from cs2rl.spec.action import ACTION_HEAD_SIZES
    rng = np.random.default_rng(seed)
    n = int(sum(lengths))
    return train_bc.DemoSet(
        obs=rng.standard_normal((n, OBS_DIM), dtype=np.float32),
        discrete=np.stack([rng.integers(0, s, size=n) for s in ACTION_HEAD_SIZES], axis=-1),
        continuous=rng.uniform(-0.5, 0.5, size=(n, AIM_DIM)).astype(np.float32),
        dones=np.zeros(n, dtype=bool),
        lengths=np.array(lengths, dtype=np.int64),
        episodes=[{
            "spawn_area": i
        } for i in range(len(lengths))],
        n_files=len(lengths),
    )


def test_demoset_as_sequences_right_pads(policy):
    """Padding goes at the END of each row (the LSTM runs over it, so it must
    never precede a real tick) and `valid` marks exactly the real ticks."""
    demos = _seq_demoset(lengths=(5, 3))
    obs, disc, cont, valid = demos.as_sequences()
    assert obs.shape == (2, 5, OBS_DIM)
    assert disc.shape == (2, 5, ACTION_DIM) and cont.shape == (2, 5, AIM_DIM)
    assert valid[0].tolist() == [True] * 5
    assert valid[1].tolist() == [True, True, True, False, False]
    np.testing.assert_array_equal(obs[0], demos.obs[:5])
    np.testing.assert_array_equal(obs[1, :3], demos.obs[5:])
    assert not obs[1, 3:].any(), "padding must be zeros"


def test_demoset_rejects_inconsistent_lengths():
    """sum(lengths) != n_ticks would shear every episode boundary and train the
    LSTM on spliced trajectories — it must be impossible to construct."""
    with pytest.raises(ValueError, match="episode boundaries"):
        train_bc.DemoSet(
            obs=np.zeros((10, OBS_DIM), dtype=np.float32),
            discrete=np.zeros((10, ACTION_DIM), dtype=np.int64),
            continuous=np.zeros((10, AIM_DIM), dtype=np.float32),
            dones=np.zeros(10, dtype=bool),
            lengths=np.array([4, 4], dtype=np.int64),
        )


def test_bc_loss_ignores_entropy_bonus_when_coef_zero(policy):
    """λ=0 must reduce the loss to the pure NLL — pins the sign/placement of
    the entropy term (a `+λH` typo would only show up as slower collapse)."""
    obs_t, disc_t, cont_t, valid_t = _fake_batch(seed=1)
    loss, stats = train_bc.bc_loss(policy, obs_t, disc_t, cont_t, valid=valid_t, entropy_coef=0.0)
    assert loss.item() == pytest.approx(stats["nll"], rel=1e-6)
    loss_pos, stats_pos = train_bc.bc_loss(policy,
                                           obs_t,
                                           disc_t,
                                           cont_t,
                                           valid=valid_t,
                                           entropy_coef=1e-2)
    # Positive λ REWARDS entropy, so with the same batch the loss must drop by
    # exactly λ·H (H_d + H_c > 0 for a freshly-initialised policy).
    assert loss_pos.item() < loss.item()
    assert loss_pos.item() == pytest.approx(stats["nll"] - 1e-2 *
                                            (stats_pos["entropy_d"] + stats_pos["entropy_c"]),
                                            rel=1e-5)


def test_bc_loss_decreases_on_tiny_fit(policy):
    """Overfit 2 short fixed sequences: the loss must fall substantially.
    Uses a deep copy so the module-scoped policy stays at init for the other
    tests."""
    p = copy.deepcopy(policy)
    obs_t, disc_t, cont_t, valid_t = _fake_batch(b=2, t=3, seed=2)
    opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    first, last = None, None
    for step in range(60):
        loss, _ = train_bc.bc_loss(p, obs_t, disc_t, cont_t, valid=valid_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first - 1.0, f"BC loss barely moved: {first:.3f} → {last:.3f}"


def test_bc_gradient_reaches_the_lstm(policy):
    """The recurrent weights must receive gradient.

    Under the old per-tick training the LSTM was in the graph but only ever saw
    a seq-len-1 unroll from zero state, so `weight_hh_l0` — the ONLY parameter
    that carries information between ticks — got gradient exclusively through
    the zero initial state, i.e. nothing that shapes recurrence. Sequence
    training is what makes this parameter actually learn.
    """
    p = copy.deepcopy(policy)
    obs_t, disc_t, cont_t, valid_t = _fake_batch(b=2, t=8, seed=5)
    loss, _ = train_bc.bc_loss(p, obs_t, disc_t, cont_t, valid=valid_t)
    loss.backward()
    g = p.lstm.weight_hh_l0.grad
    assert g is not None and float(g.abs().max()) > 0, \
        "no through-time gradient reached lstm.weight_hh_l0 — BC is not training sequences"


def test_aim_log_std_is_frozen_during_bc(policy):
    """Spec D-6: σ must not train. Checked two ways — no gradient reaches the
    parameter, and its value is bit-identical after real optimizer steps."""
    p = copy.deepcopy(policy)
    obs_t, disc_t, cont_t, valid_t = _fake_batch(b=2, t=4, seed=3)
    before = p.aim_log_std.detach().clone()

    loss, _ = train_bc.bc_loss(p, obs_t, disc_t, cont_t, valid=valid_t)
    loss.backward()
    assert p.aim_log_std.grad is None or torch.all(p.aim_log_std.grad == 0), \
        "aim_log_std received gradient — the .detach() in bc_loss is gone"

    opt = torch.optim.Adam(p.parameters(), lr=1e-2)
    for _ in range(20):
        loss, _ = train_bc.bc_loss(p, obs_t, disc_t, cont_t, valid=valid_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert torch.equal(p.aim_log_std.detach(), before), "aim_log_std moved during BC"
    # ...while the rest of the policy DID train (guards against the test
    # passing because nothing at all was updated).
    assert not torch.equal(p.aim_mu.weight.detach(), policy.aim_mu.weight.detach())


def test_train_bc_history_decreases_and_returns_eval_mode_policy(policy):
    """train_bc's own loop (sequence minibatches) on a synthetic DemoSet:
    history must show the loss falling from first to last epoch."""
    p, history = train_bc.train_bc(_seq_demoset(seed=4, lengths=(16, 16)),
                                   policy=copy.deepcopy(policy),
                                   epochs=60,
                                   batch_size=2,
                                   verbose=False)
    assert len(history) == 60
    assert history[-1]["loss"] < history[0]["loss"] - 0.5
    assert not p.training, "train_bc must leave the policy in eval mode"


def test_build_bc_policy_is_reproducible_under_seed():
    """`--seed` must actually determine the weights. build_bc_policy seeds
    BEFORE build_policy because layer_init draws from the global RNG at
    construction time; seeding afterwards (the old order) made two identically
    seeded builds differ."""
    a = train_bc.build_bc_policy(device="cpu", seed=123)
    b = train_bc.build_bc_policy(device="cpu", seed=123)
    c = train_bc.build_bc_policy(device="cpu", seed=124)
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), f"{k} differs between two seed=123 builds"
    assert not torch.equal(a.aim_mu.weight, c.aim_mu.weight), \
        "seed 123 and 124 produced identical weights — the seed is not reaching build_policy"


# ── demo provenance ────────────────────────────────────────────────────────


def _demo_dict(**overrides):
    """Minimal spec-§7-shaped demo payload; overrides patch individual keys."""
    T = 3
    d = dict(
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
        git_sha=train_bc._git("rev-parse", "HEAD")[1],
        map=train_bc.EXPECTED_MAP,
    )
    d.update(overrides)
    return d


def test_load_demos_rejects_schema_mismatch(tmp_path):
    """Spec R7/§7: a demo whose declared OBS_DIM differs from the live one is
    a hard error, not a warning — silently training on it would produce a
    checkpoint that explodes inside `--resume`."""
    good = _demo_dict()
    np.savez(tmp_path / "ok.npz", **good)
    loaded = train_bc.load_demos(tmp_path, verbose=False)
    assert len(loaded) == 3
    assert loaded.lengths.tolist() == [3]

    bad_dim = dict(good, OBS_DIM=OBS_DIM + 1)
    np.savez(tmp_path / "bad_dim.npz", **bad_dim)
    with pytest.raises(ValueError, match="demo schema"):
        train_bc.load_demos(tmp_path, verbose=False)
    (tmp_path / "bad_dim.npz").unlink()

    bad_map = dict(good, map="de_dust2")
    np.savez(tmp_path / "bad_map.npz", **bad_map)
    with pytest.raises(ValueError, match="map"):
        train_bc.load_demos(tmp_path, verbose=False)
    (tmp_path / "bad_map.npz").unlink()

    # Δyaw labels above the env's clamp mean the recorded action is not the
    # executed one — the labels would be lies.
    bad_cont = dict(good)
    bad_cont["continuous_actions"] = np.full((3, AIM_DIM), 10.0, dtype=np.float32)
    np.savez(tmp_path / "bad_cont.npz", **bad_cont)
    with pytest.raises(ValueError, match="MAX_TURN_SPEED_RAD"):
        train_bc.load_demos(tmp_path, verbose=False)


def test_check_demo_sha_accepts_head_and_rejects_unknown_shas():
    """The predecessor of this guard only checked `len(sha) == 40`, so a demo
    recorded against an obs layout that no longer exists sailed through
    (review finding 5). HEAD must pass; 40 hex characters git has never heard
    of must not."""
    head = train_bc._git("rev-parse", "HEAD")[1]
    assert len(head) == 40, "test needs a real git checkout"
    assert train_bc.check_demo_sha(head) is None

    unknown = "d" * 40
    with pytest.raises(ValueError, match="STALE DEMO"):
        train_bc.check_demo_sha(unknown)
    # ...and the escape hatch downgrades it to a note rather than silence.
    note = train_bc.check_demo_sha(unknown, allow_stale=True)
    assert note is not None and "STALE" in note

    with pytest.raises(ValueError, match="not self-identifying"):
        train_bc.check_demo_sha("abc123")


def test_check_demo_sha_tolerates_commits_that_cannot_change_a_demo():
    """A demo is only stale if something that DETERMINES it changed. An older
    commit that touched nothing under DEMO_RELEVANT_PATHS must pass with a
    note, not fail — otherwise every unrelated commit invalidates the demo set
    and the escape hatch becomes the normal path.

    Finds such a commit from real history; skips if this checkout has none.
    """
    rc, log = train_bc._git("log", "--format=%H", "-n", "40")
    if rc != 0 or not log:
        pytest.skip("no git history available")
    head = train_bc._git("rev-parse", "HEAD")[1]
    for sha in log.splitlines()[1:]:
        if train_bc._git("diff", "--quiet", sha, "--", *train_bc.DEMO_RELEVANT_PATHS)[0] == 0:
            note = train_bc.check_demo_sha(sha)
            assert note is not None and "still reproducible" in note
            assert sha[:9] in note and head[:9] in note
            return
    pytest.skip("every recent commit touched the demo-relevant surface")


def test_load_demos_dedupes_byte_identical_episodes(tmp_path):
    """The demo set is 5 unique trajectories × 10 seeds (deterministic spawns,
    plan Task 1 RESULT). Dedupe must drop exact copies and keep everything
    that differs by even one tick — and the surviving per-episode `lengths`
    must describe exactly the kept episodes."""
    T = 3
    base = _demo_dict(obs=np.ones((T, OBS_DIM), dtype=np.float32))
    np.savez(tmp_path / "a.npz", **base)
    np.savez(tmp_path / "b.npz", **dict(base, seed=1))                 # identical arrays
    different = dict(base, seed=2)
    different["obs"] = np.full((T, OBS_DIM), 2.0, dtype=np.float32)
    np.savez(tmp_path / "c.npz", **different)

    deduped = train_bc.load_demos(tmp_path, dedupe=True, verbose=False)
    assert deduped.n_episodes == 2 and deduped.n_duplicates == 1
    assert len(deduped) == 2 * T
    assert deduped.lengths.tolist() == [T, T]

    kept_all = train_bc.load_demos(tmp_path, dedupe=False, verbose=False)
    assert kept_all.n_episodes == 3 and kept_all.n_duplicates == 0
    assert len(kept_all) == 3 * T


def test_bc_loss_pinned_policy_excludes_pitch_dim(policy):
    """R0-E.2 (#131): a policy built with pin_pitch=True carries
    aim_dim_mask=[1,0]; bc_loss must forward it to _hybrid_sample_logits so
    nll_c / entropy_c cover the yaw dim ONLY (the env ignores cont[:,1], so
    fitting demo pitch would be gradient on a dead dimension)."""
    p = copy.deepcopy(policy)
    p.aim_dim_mask.copy_(torch.tensor([1.0, 0.0]))
    obs_t, disc_t, cont_t, valid_t = _fake_batch(seed=5)
    lam = 1e-3
    loss, stats = train_bc.bc_loss(p, obs_t, disc_t, cont_t, valid=valid_t, entropy_coef=lam)

    flat_cont = cont_t.reshape(-1, cont_t.shape[-1])
    with torch.no_grad():
        _logits, mu_aim, log_std, _v = p(obs_t, {})
        sigma = torch.exp(log_std).expand_as(mu_aim)
        diff = (flat_cont - mu_aim) / sigma
        lp_c_dim = -0.5 * diff * diff - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
        h_c_dim = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(sigma)
    lp_d, _lp_c_full, h_d, _h_c_full, w = _hand_rolled(p, obs_t, disc_t, cont_t, valid_t)
    mean = lambda x: (x * w).sum() / w.sum()           # noqa: E731
    expected = -(mean(lp_d) + mean(lp_c_dim[:, 0])) - lam * (mean(h_d) + mean(h_c_dim[:, 0]))
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-6)
    assert stats["nll_c"] == pytest.approx(-mean(lp_c_dim[:, 0]).item(), rel=1e-5)
                                                       # and it is genuinely different from the unpinned value (dim 1 dropped)
    assert stats["nll_c"] != pytest.approx(-mean(_lp_c_full).item(), rel=1e-3)
