"""Spec 2026-08-29 §2.2 (i): every trainer reduction, computed with the
participating mask, equals the same reduction over the participating
sub-tensor alone. Plus the harness-level wiring check at n_active_per_team=1.
§2.2 (ii), the 20-update ratio identity, is APPENDED TO THIS FILE in Task 9
(it needs the harness's `aim_entropy_bonus=False` kwarg, which Task 9 adds).

Rung 1a T3 (spec 2026-08-30) extends the same subject: `--opponent noop` makes
the participation vector hero-team-only and halves the participating-step
budget, so the helper that builds that vector, the budget formula it feeds and
the statue rollout it implies are pinned at the bottom of this file.
"""
import numpy as np
import pytest
import torch

from _action_spec import ACTION_HEAD_SIZES
from train import masked_explained_variance, masked_mean, masked_normalize_adv, masked_std_unbiased


@pytest.fixture
def fixed():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, 8, generator=g)
    part = torch.zeros(40, 8, dtype=torch.bool)
    part[::5, :] = True                # rows 0,5,10,... participate (1-in-5, like 1v1)
    return x, part


def test_masked_mean_equals_subset_mean(fixed):
    x, part = fixed
    assert masked_mean(x, part.float()).item() == pytest.approx(x[part].mean().item(), abs=1e-6)


def test_masked_std_equals_subset_std(fixed):
    x, part = fixed
    w = part.float()
    m = masked_mean(x, w)
    assert masked_std_unbiased(x, w, m).item() == pytest.approx(x[part].std().item(), abs=1e-6)


def test_masked_normalize_adv_matches_subset_normalisation(fixed):
    x, part = fixed
    out = masked_normalize_adv(x.reshape(-1), part.float().reshape(-1))
    sub = x[part]
    ref = (sub - sub.mean()) / (sub.std() + 1e-8)
    assert torch.allclose(out.reshape(40, 8)[part], ref, atol=1e-6)
    assert torch.all(out.reshape(40, 8)[~part] == 0.0)


def test_masked_explained_variance_matches_subset(fixed):
    x, part = fixed
    y_true = x
    torch.manual_seed(0)               # reproducible failure output; assertion is analytic
    y_pred = x + 0.1 * torch.randn_like(x)
    ev = masked_explained_variance(y_pred.flatten(), y_true.flatten(), part.flatten())
    yt, yp = y_true[part], y_pred[part]
    ref = 1 - (yt - yp).var() / yt.var()
    assert ev == pytest.approx(ref.item(), abs=1e-6)


def test_masked_mean_all_ones_is_plain_mean(fixed):
    x, _ = fixed
    assert masked_mean(x, torch.ones_like(x)).item() == pytest.approx(x.mean().item(), abs=1e-6)


def test_pg_loss_masked_equals_participating_subtensor():
    """Spec §2.2 (i) for pg_loss: the COMPOSITE (masked adv-norm → per-row PPO
    terms → masked mean) must equal _hybrid_ppo_loss run on the participating
    rows alone with mb_part=None. This is the non-obvious claim; helper-level
    tests do not cover it."""
    from train import _hybrid_ppo_loss
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=4, n_active_per_team=1)
    try:
        torch.manual_seed(0)
        pol = trainer.policy
        S, T = 10, 8                                                   # 10 segments (2 teams×5 rows), horizon 8
        obs = torch.randn(S, T, trainer.observations.shape[-1])
        act = torch.stack([torch.randint(0, n, (S, T)) for n in ACTION_HEAD_SIZES], -1)
        cont = torch.randn(S, T, 2) * 0.1
        old_d, old_c = torch.randn(S, T), torch.randn(S, T)
        adv = torch.randn(S, T)
        part = torch.zeros(S, dtype=torch.bool)
        part[0] = part[5] = True                                       # n_active=1
        part_st = part[:, None].expand(S, T)
        state = dict(action=None, lstm_h=None, lstm_c=None, terminals=torch.zeros(S, T))
        full = _hybrid_ppo_loss(pol,
                                obs,
                                act,
                                cont,
                                old_d,
                                old_c,
                                adv,
                                0.2,
                                state,
                                mb_part=part_st.to(torch.float32))
        sub = _hybrid_ppo_loss(pol,
                               obs[part],
                               act[part],
                               cont[part],
                               old_d[part],
                               old_c[part],
                               adv[part],
                               0.2,
                               dict(action=None,
                                    lstm_h=None,
                                    lstm_c=None,
                                    terminals=torch.zeros(2, T)),
                               mb_part=None)
                                                                       # full[0] / sub[0] = pg_loss
        assert full[0].item() == pytest.approx(sub[0].item(), abs=1e-6)
                                                                       # full[1] / sub[1] = entropy, returned PER ROW (the caller reduces it
                                                                       # via masked_mean) — an .item() on the 80-element tensor would raise.
                                                                       # Reduce both sides the way the caller does.
        full_entropy = masked_mean(full[1], part_st.reshape(-1).float()).item()
        assert full_entropy == pytest.approx(sub[1].mean().item(), abs=1e-6)
    finally:
        cleanup()


def test_return_stats_update_on_participating_rows_only():
    """Spec §2.2 (i) for _ret_mean/_ret_var: one _normalize_returns call from the
    zero-count state must leave the running stats equal to the participating
    sub-tensor's mean / population variance (Welford's first update)."""
    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=4, n_active_per_team=1)
    try:
        _patch_trainer_with_return_norm(trainer)
        torch.manual_seed(1)
        x = torch.randn(10, 8) * 3 + 1
        part = torch.zeros(10, 8, dtype=torch.bool)
        part[0] = part[5] = True
        trainer._normalize_returns(x, part)            # alias installed in Step 6
        sel = x[part]
        assert trainer._ret_mean.item() == pytest.approx(sel.mean().item(), abs=1e-6)
        assert trainer._ret_var.item() == pytest.approx(sel.var(unbiased=False).item(), abs=1e-6)
        assert trainer._ret_count.item() == pytest.approx(sel.numel())
    finally:
        cleanup()


def test_harness_n_active_1_masks_four_fifths_of_rows():
    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=16, n_active_per_team=1)
    try:
        _patch_trainer_with_return_norm(trainer)
        bs = trainer.config["batch_size"]
        assert trainer.config["participating_timesteps"] * 5 == trainer.config["total_timesteps"]
        trainer.evaluate()
        assert trainer.global_step == bs // 5
        n_part = trainer.participating.sum().item()
        assert n_part == trainer.segments * trainer.config["bptt_horizon"] // 5
        # parked rows carry zero critic output in the buffer
        assert torch.all(trainer.values[~trainer.participating] == 0.0)
        # force the throttled block that assigns trainer.losses
        trainer.last_log_time = 0.0
        trainer.train()
        losses = trainer.losses
        assert losses["participating_rows"] == n_part
        assert losses["empty_minibatches"] == 0
        assert 0 < trainer._ret_count.item() <= trainer.config["update_epochs"] * n_part
        assert np.isfinite(losses["entropy"]) and np.isfinite(losses["entropy_unmasked"])
        # parked rows are noop-masked ⇒ 0 discrete entropy, so the unmasked
        # mean is dragged down by the 4/5 of rows the masked mean drops
        assert losses["entropy"] > losses["entropy_unmasked"]
    finally:
        cleanup()


@pytest.mark.slow
def test_twenty_update_ratio_identity_n_active_1():
    """Spec §2.2 (ii): at n_active=1 with the aim entropy bonus OFF, parked rows are
    noop-masked ⇒ losses/entropy ≈ 5 × losses/entropy_unmasked over the MEAN of 20
    updates (±5%; per-update σ≈4.6% from the hypergeometric minibatch draw, ≈1.0% over
    the mean). Preconditions are load-bearing: prioritised sampling or any event
    segment would sample parked segments non-uniformly and break the identity."""
    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=16,
                                               n_active_per_team=1,
                                               aim_entropy_bonus=False)
    try:
        _patch_trainer_with_return_norm(trainer)
        assert trainer.config["prio_alpha"] == 0
        assert not bool(trainer._batch1_event_mask.any())
        ent, ent_u, floor, mbs, alpha = [], [], 0.0, 0.0, {}
        for u in range(1, 21):
            trainer.evaluate()
            trainer.last_log_time = 0.0
            trainer.train()
            # trainer.losses, not train()'s return: mean_and_log() runs BEFORE
            # self.losses is assigned, so the returned logs lag one update.
            logs = trainer.losses
            ent.append(logs["entropy"])
            ent_u.append(logs["entropy_unmasked"])
            floor += logs["entropy_floor_fires"]
            mbs += logs["minibatches_run"]
            assert logs["empty_minibatches"] == 0
            assert logs["participating_rows"] == trainer.participating.sum().item()
            if u in (5, 20):
                alpha[u] = logs["effective_alpha"]
        ratio = np.mean(ent) / np.mean(ent_u)
        assert ratio == pytest.approx(5.0, rel=0.05), ratio
        assert floor / mbs < 0.10, (floor, mbs)
        assert alpha[20] <= 2 * alpha[5] and alpha[5] <= 2 * alpha[20], alpha
    finally:
        cleanup()


# ── Rung 1a T3: --opponent noop (stationary statue) ─────────────────────────


def test_build_participating_rows_noop_marks_the_hero_team_only():
    """The single source of truth for participation (spec 2026-08-30 §2 T3).

    `self` must stay bit-identical to the pre-T3 expression — it is what every
    existing run's config.json and every masked-loss test was computed against
    — while `noop` drops the statue team entirely. Both call sites (train() and
    train_test_harness) call THIS function, so this test plus the two call
    sites is what makes the harness and production agree by construction.
    """
    from train import TEAM_SIZE, build_participating_rows

    n_envs = 3
    self_rows = build_participating_rows(n_envs, 1, "self", "t")
    assert np.array_equal(self_rows,
                          np.array([(i % TEAM_SIZE) < 1 for i in range(n_envs * 10)], dtype=bool))
    assert self_rows.sum() == 2 * n_envs               # both teams, 1 slot each

    noop_rows = build_participating_rows(n_envs, 1, "noop", "t")
    assert noop_rows.sum() == n_envs   # hero team only
    slot = np.arange(n_envs * 10) % 10
    assert not noop_rows[slot >= TEAM_SIZE].any(), "CT (statue) rows must never participate"
    assert np.array_equal(noop_rows, self_rows & (slot < TEAM_SIZE))

    # hero_team is honoured, not assumed: the CT-hero mirror image.
    ct_hero = build_participating_rows(n_envs, 2, "noop", "ct")
    assert np.array_equal(
        ct_hero,
        np.array([TEAM_SIZE <= (i % 10) < TEAM_SIZE + 2 for i in range(n_envs * 10)], dtype=bool))
    # SelfPlayManager starts with CT as the opponent, so the hero is T.
    from train import SelfPlayManager
    assert SelfPlayManager.initial_hero_team() == "t"
    assert SelfPlayManager().opponent_team == "ct"


@pytest.mark.parametrize("bad", [
    dict(opponent_mode="statue"),
    dict(hero_team="T"),
    dict(n_active=0),
    dict(n_active=6),
])
def test_build_participating_rows_refuses_nonsense(bad):
    """Every argument is validated: a silently-wrong participation vector is
    the one failure mode this experiment cannot detect from its metrics."""
    from train import build_participating_rows
    kwargs = dict(num_envs=2, n_active=1, opponent_mode="self", hero_team="t")
    kwargs.update(bad)
    with pytest.raises(ValueError):
        build_participating_rows(**kwargs)


def test_opponent_mode_resolution_and_self_play_guard():
    """Rung 1a T3: `--opponent` resolution + the noop/self-play startup guard.

    The legacy-args fallback matters as much as the guard: harness and
    --dump-config namespaces predate the flag and must keep resolving to
    "self" (a fallback to "noop" would statue every legacy caller's opponent).
    """
    import types

    from train import assert_opponent_self_play_compatible, resolve_opponent_mode

    assert resolve_opponent_mode(types.SimpleNamespace()) == "self"
    assert resolve_opponent_mode(types.SimpleNamespace(opponent="noop")) == "noop"
    with pytest.raises(ValueError):
        resolve_opponent_mode(types.SimpleNamespace(opponent="statue"))

    assert_opponent_self_play_compatible("self", True)                 # today's default: fine
    assert_opponent_self_play_compatible("noop", False)                # T4's launch: fine
    with pytest.raises(ValueError, match="no-self-play"):
        assert_opponent_self_play_compatible("noop", True)


def test_noop_budget_doubles_total_timesteps_and_records_the_mode():
    """Spec §2 T3 budget: --timesteps is a PARTICIPATING-step budget, and under
    noop only half as many rows participate per env, so the raw horizon handed
    to PufferLib must double. The concrete Rung 1a T4 launch is asserted:
    1M requested at n_active=1 / 256 envs ⇒ 10M raw ⇒ 61 epochs. Without the
    fix the run would stop at ~491k hero steps with exit code 0.

    The `self` half is the regression guard: the generalised formula must
    reduce EXACTLY to the old `timesteps * TEAM_SIZE // n_active` for every
    n_active, or every pre-T3 run's horizon silently changes.
    """
    import types

    from train import build_train_config, compute_batch_dims

    def cfg(**over):
        kwargs = dict(device="cpu",
                      seed=1,
                      timesteps=1_000_000,
                      checkpoint_dir="/tmp/x",
                      n_active_per_team=1)
        kwargs.update(over)
        _, bptt, bs = compute_batch_dims(256)
        return build_train_config(types.SimpleNamespace(**kwargs), batch_size=bs,
                                  bptt_horizon=bptt), bs

    noop, batch_size = cfg(opponent="noop")
    assert batch_size == 163_840
    assert noop["opponent"] == "noop"
    assert noop["participating_timesteps"] == 1_000_000
    assert noop["total_timesteps"] == 10_000_000
    assert noop["total_timesteps"] // batch_size == 61

    selfp, _ = cfg(opponent="self")
    assert selfp["opponent"] == "self"
    assert selfp["total_timesteps"] == 5_000_000

    # Legacy args (no `opponent` attribute) resolve to "self" AND to the exact
    # pre-T3 horizon at every n_active.
    for n in range(1, 6):
        legacy, _ = cfg(n_active_per_team=n)
        assert legacy["opponent"] == "self"
        assert legacy["total_timesteps"] == 1_000_000 * 5 // n


def test_noop_opponent_rows_are_statues_excluded_from_global_step():
    """Spec §2 T3 test (i), end-to-end through the PRODUCTION rollout path.

    One evaluate() round at n_active_per_team=1 with --opponent noop must:
      - drive every statue row with the no-op bin on every discrete head and a
        zero aim delta (the env's own stationarity contract, cs2_env.h:51-63);
      - leave those rows non-participating, with zero stored log-probs and a
        zero critic output (they must not bootstrap GAE);
      - count ONLY the hero rows into global_step — 1 row per env instead of
        the 2 the `self` path counts (tests/…::test_harness_n_active_1_masks_
        four_fifths_of_rows pins that comparison at bs // 5), which is the
        whole reason the budget formula had to change;
      - and still train the hero rows (losses/participating_rows == the hero
        row count is exactly T4's pre-flight #1, scaled down).

    Row mapping: the harness's first evaluate() round fills one segment per
    agent row (segments == total_agents, ep_indices starts as arange), so
    buffer row i IS agent row i — asserted below rather than assumed.
    """
    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=16, n_active_per_team=1, opponent="noop")
    try:
        _patch_trainer_with_return_norm(trainer)
        bs = trainer.config["batch_size"]
        assert trainer.config["opponent"] == "noop"
        assert trainer.config["participating_timesteps"] * 10 == trainer.config["total_timesteps"]
        assert trainer.segments == trainer.total_agents                # ⇒ buffer row i == agent row i

        trainer.evaluate()

        # T is the hero (SelfPlayManager.initial_hero_team()); CT slots 5-9 of
        # every env are the statue — ALL of them, parked ones included.
        slot = np.arange(trainer.total_agents) % 10
        statue = torch.as_tensor(slot >= 5)
        hero = torch.as_tensor(slot == 0)              # n_active_per_team=1 ⇒ T slot 0

        assert trainer.global_step == bs // 10, "statue rows must not count toward the budget"
        assert not trainer.participating[statue].any()
        assert trainer.participating[hero].all()
        assert torch.all(trainer.actions[statue] == 0), "statue must play the no-op bin"
        assert torch.all(trainer.cont_actions[statue] == 0), "statue must not turn"
        for buf in (trainer.logprobs, trainer.logprobs_d, trainer.logprobs_c, trainer.values):
            assert torch.all(buf[statue] == 0)
        # ...and the override is SCOPED: the hero still samples a live policy.
        assert trainer.actions[hero].any()
        assert trainer.cont_actions[hero].abs().sum() > 0

        trainer.last_log_time = 0.0    # force the throttled losses assignment
        trainer.train()
        n_part = trainer.segments * trainer.config["bptt_horizon"] // 10
        assert trainer.losses["participating_rows"] == n_part
        assert np.isfinite(trainer.losses["entropy"])
    finally:
        cleanup()


def test_harness_refuses_noop_with_selfplay():
    """The harness mirrors train()'s startup guard, so no test can construct a
    trainer in a configuration production refuses (a past-policy opponent is
    not a statue, and maybe_switch_teams would move the statue's team)."""
    from train_test_harness import _build_trainer_for_test
    with pytest.raises(ValueError, match="no-self-play"):
        _build_trainer_for_test(num_envs=4, opponent="noop", with_selfplay=True)
