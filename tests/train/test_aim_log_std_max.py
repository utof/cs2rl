"""R0-E.3/4 (#131): per-run aim σ cap, entropy-bonus switch, §2.2(ii) parked-row integration.

Rung 1a T1 (spec 2026-08-30) extends the file with the σ *trainability*
contract — init strictly inside the clamp band, no weight decay, raw logging.
The two live together on purpose: every one of those knobs is a way for the aim
σ to end up silently frozen, and the Rung 1 failure was exactly that (init
log 0.1 under a log 0.05 cap ⇒ clamp zeroed the gradient for 10M steps while
`policy/aim_log_std_yaw` reported a plausible-looking −2.996).
"""
import math

import numpy as np
import pytest
import torch


def test_cap_applied_in_forward_and_metrics(simple_map):
    from cs2rl.policy import LOG_STD_MAX
    from cs2rl.train.metrics import log_aim_log_std
    from tests._helpers.trainer_harness import _build_trainer_for_test
    cap = math.log(0.05)
    trainer, cleanup = _build_trainer_for_test(num_envs=4, map_data=simple_map, aim_log_std_max=cap)
    try:
        pol = trainer.policy
        assert pol.aim_log_std_max == pytest.approx(cap)
        with torch.no_grad():
            pol.aim_log_std.fill_(LOG_STD_MAX)
        obs = torch.zeros(4, trainer.vecenv.single_observation_space.shape[0])
        _, _, log_std, _ = pol.forward_eval(obs, {
            "lstm_h": None,
            "lstm_c": None,
            "done": torch.zeros(4, dtype=torch.bool)
        })
        assert float(log_std.max()) <= cap + 1e-6
        # the BPTT training forward and the nn.Module sampler take the same cap
        _, _, log_std_tr, _ = pol(obs, {})
        assert float(log_std_tr.max()) <= cap + 1e-6
        with torch.no_grad():
            _a, c, *_ = pol.get_action_and_value(obs)
        assert c.shape == (4, 2)
        logs = {}
        log_aim_log_std(pol, logs)
        assert logs["policy/aim_log_std_yaw"] == pytest.approx(cap)
        assert logs["policy/aim_log_std_pitch"] == pytest.approx(cap)
    finally:
        cleanup()


def test_cap_outside_band_is_refused(simple_map):
    from cs2rl.policy import LOG_STD_MAX, LOG_STD_MIN
    from tests._helpers.trainer_harness import _build_trainer_for_test
    for bad in (LOG_STD_MIN, LOG_STD_MAX + 0.1, LOG_STD_MIN - 1.0):
        with pytest.raises(ValueError):
            trainer, cleanup = _build_trainer_for_test(num_envs=4,
                                                       map_data=simple_map,
                                                       aim_log_std_max=bad)
            cleanup()


def test_reinit_frozen_respects_cap():
    from cs2rl.policy import LOG_STD_INIT
    from cs2rl.train.resume import AIM_LOG_STD_RESUME_INIT, reinit_frozen_aim_log_std
    sd = {"aim_log_std": torch.full((2, ), LOG_STD_INIT)}
    assert reinit_frozen_aim_log_std(sd, cap=math.log(0.05))
    assert torch.allclose(sd["aim_log_std"], torch.full((2, ), math.log(0.05)))
    sd = {"aim_log_std": torch.full((2, ), LOG_STD_INIT)}
    assert reinit_frozen_aim_log_std(sd)
    assert torch.allclose(sd["aim_log_std"], torch.full((2, ), AIM_LOG_STD_RESUME_INIT))
    # a cap ABOVE the resume init leaves the resume init in charge
    sd = {"aim_log_std": torch.full((2, ), LOG_STD_INIT)}
    assert reinit_frozen_aim_log_std(sd, cap=math.log(0.4))
    assert torch.allclose(sd["aim_log_std"], torch.full((2, ), AIM_LOG_STD_RESUME_INIT))


def test_max_entropy_reflects_cap_pin_and_bonus(simple_map):
    from cs2rl.spec.action import ACTION_HEAD_SIZES
    from tests._helpers.trainer_harness import _build_trainer_for_test
    disc = sum(math.log(n) for n in ACTION_HEAD_SIZES)
    cap = math.log(0.05)
    for pin, bonus, n_dims in ((0, True, 2), (1, True, 1), (0, False, 0)):
        trainer, cleanup = _build_trainer_for_test(num_envs=4,
                                                   map_data=simple_map,
                                                   aim_log_std_max=cap,
                                                   pin_pitch=pin,
                                                   aim_entropy_bonus=bonus)
        try:
            assert trainer.config["aim_entropy_bonus"] is bonus
            assert trainer.config["aim_log_std_max"] == pytest.approx(cap)
            assert trainer.config["pin_pitch"] == pin
            cont = n_dims * 0.5 * math.log(2 * math.pi * math.e * math.exp(cap)**2)
            assert trainer._max_entropy == pytest.approx(disc + cont)
        finally:
            cleanup()


def test_ppo_loss_entropy_bonus_switch():
    """aim_entropy_bonus=False ⇒ the entropy the loss returns is the DISCRETE
    entropy only; True (default) ⇒ discrete + Gaussian. Pure-function check on
    a fake 2D-input policy so it needs no env."""
    from cs2rl.policy import _LOG_2PI
    from cs2rl.spec.action import ACTION_HEAD_SIZES, AIM_DIM
    from cs2rl.train.update import _hybrid_ppo_loss
    B = 5

    class _Pol:

        def __call__(self, obs, state):
            logits = [torch.zeros(B, n) for n in ACTION_HEAD_SIZES]
            return logits, torch.zeros(B, AIM_DIM), torch.full((B, AIM_DIM), math.log(0.1)), \
                torch.zeros(B, 1)

    obs = torch.zeros(B, 3)
    acts = torch.zeros(B, len(ACTION_HEAD_SIZES), dtype=torch.int64)
    cont = torch.zeros(B, AIM_DIM)
    z = torch.zeros(B)
    ent_on = _hybrid_ppo_loss(_Pol(), obs, acts, cont, z, z, torch.randn(B), 0.2, {})[1]
    ent_off = _hybrid_ppo_loss(_Pol(),
                               obs,
                               acts,
                               cont,
                               z,
                               z,
                               torch.randn(B),
                               0.2, {},
                               aim_entropy_bonus=False)[1]
    ent_pin = _hybrid_ppo_loss(_Pol(),
                               obs,
                               acts,
                               cont,
                               z,
                               z,
                               torch.randn(B),
                               0.2, {},
                               aim_dim_mask=torch.tensor([1.0, 0.0]))[1]
    disc = sum(math.log(n) for n in ACTION_HEAD_SIZES)
    gauss = 0.5 + 0.5 * _LOG_2PI + math.log(0.1)
    assert torch.allclose(ent_on, torch.full((B, ), disc + 2 * gauss), atol=1e-5)
    assert torch.allclose(ent_off, torch.full((B, ), disc), atol=1e-5)
    assert torch.allclose(ent_pin, torch.full((B, ), disc + gauss), atol=1e-5)


@pytest.mark.training
def test_parked_rows_do_not_move_objective(simple_map):
    """Spec §2.2(ii): with parked rows, adv-norm/objective ignore them.
    Setup: n_active=1, entropy bonus off; run one train(); then perturb the
    parked rows' stored advantages/logprobs by a huge amount, re-run the
    minibatch reductions through a second identical trainer and compare
    losses/policy_loss, losses/entropy, losses/approx_kl."""
    from tests._helpers.trainer_harness import _build_trainer_for_test

    def _one(perturb):
        torch.manual_seed(0)
        np.random.seed(0)
        trainer, cleanup = _build_trainer_for_test(num_envs=16,
                                                   map_data=simple_map,
                                                   n_active_per_team=1,
                                                   aim_entropy_bonus=False)
        try:
            trainer.evaluate()
            if perturb:
                parked = ~trainer.participating
                trainer.logprobs_d[parked] += 50.0
                trainer.logprobs_c[parked] -= 50.0
                trainer.rewards[parked] += 1e3
            # (do NOT set trainer.config["update_epochs"] here — total_minibatches is
            # fixed in PuffeRL.__init__; the mutation would only disable the KL
            # early-abort boundary check)
            trainer.last_log_time = 0.0
            trainer.train()
            # trainer.losses, not train()'s return: mean_and_log() runs BEFORE
            # self.losses is assigned, so the returned logs carry the PREVIOUS
            # update's losses/* (documented one-epoch lag).
            return {k: trainer.losses[k] for k in ("policy_loss", "entropy", "approx_kl")}
        finally:
            cleanup()

    a, b = _one(False), _one(True)
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-4, abs=1e-6), (k, a[k], b[k])


# ── Rung 1a T1: trainable, drift-free σ ─────────────────────────────────────
# Three independent ways the aim σ can be dead, one test each:
#   (a) init AT/above the cap        → clamp zeroes the gradient from step 0;
#   (b) weight decay on log σ        → σ moves without any learning signal,
#                                       faking the gate and eating (a)'s margin;
#   (c) only the CLAMPED value logged → "σ sits at the cap" and "σ was pushed
#                                       past the cap and is now frozen" read
#                                       identically on the dashboard.
# Plus (d) the validator that keeps (a)'s margin reachable and (e) the
# no-regression case: at the 5v5 default cap nothing about the init changes.


def _policy_with_cap(cap, **kw):
    """Build a bare policy at a given σ cap — no trainer, no vec workers.

    build_policy only reads scalars off the env (obs dim, max_turn_speed) at
    construction time, so closing the env immediately afterwards is safe; this
    is the same pattern as tests/test_tct_split.py's `env` fixture usage and it
    keeps the policy-only tests below at ~env-construction cost.
    """
    from cs2rl import policy as policy_mod
    from cs2rl.env.c.cs2_env import make_env
    env = make_env(seed=0)
    try:
        return policy_mod.build_policy(env, device="cpu", aim_log_std_max=cap, **kw)
    finally:
        env.close()


def test_tight_cap_inits_below_the_cap_with_a_live_gradient():
    """T1(a): under a tight cap the fresh init is cap − AIM_LOG_STD_INIT_MARGIN
    and σ still receives gradient; parked ON the cap it receives none.

    The second half is the control that gives the first half meaning — it
    reproduces the Rung 1 defect exactly (σ one step past the cap) and shows
    the gradient is then identically zero, not merely small.
    """
    from cs2rl.policy import AIM_LOG_STD_INIT_MARGIN, LOG_STD_INIT
    cap = math.log(0.05)
    pol = _policy_with_cap(cap)
    init = cap - AIM_LOG_STD_INIT_MARGIN
    assert init < LOG_STD_INIT, "cap must be tight enough to actually move the init"
    assert torch.allclose(pol.aim_log_std, torch.full_like(pol.aim_log_std, init))

    obs = torch.zeros(4, pol.obs_dim)
    ent = pol.get_action_and_value(obs)[3]
    ent.sum().backward()
    assert pol.aim_log_std.grad is not None
    assert float(pol.aim_log_std.grad.abs().min()) > 0.0

    pol.zero_grad(set_to_none=True)
    with torch.no_grad():              # the Rung 1 state: raw σ one step outside the band
        pol.aim_log_std.fill_(cap + 0.1)
    pol.get_action_and_value(obs)[3].sum().backward()
    assert float(pol.aim_log_std.grad.abs().max()) == 0.0


def test_sigma_param_group_is_not_weight_decayed(simple_map):
    """T1(b): with a zero σ-gradient, N optimizer steps must leave the raw
    parameter EXACTLY where it started.

    Zero grads are set explicitly rather than left as None — None makes Adam
    skip the parameter entirely (including its decay), which would pass this
    test for the wrong reason. Grad-present-but-zero is also the real
    clamp-dead state: the clamp stays in the graph and hands back a 0.

    The `isolate=False` half is the negative control: it is the pre-T1 wiring
    and MUST drift, upward (decay adds wd·θ and log σ is negative), by roughly
    N·lr = 30 × 3e-4 ≈ 9e-3. Without it, a broken helper that moved nothing at
    all would still make the first half green.
    """
    from cs2rl.train.loop import isolate_aim_log_std_param_group
    from tests._helpers.trainer_harness import _build_trainer_for_test

    def _drift(isolate):
        trainer, cleanup = _build_trainer_for_test(num_envs=4,
                                                   map_data=simple_map,
                                                   aim_log_std_max=math.log(0.05))
        try:
            # the two production lines, in production order (src/cs2rl/train.py train())
            trainer.optimizer.param_groups[0]["weight_decay"] = 1e-4
            if isolate:
                assert isolate_aim_log_std_param_group(trainer) == 1
            sigma = trainer.policy.aim_log_std
            before = sigma.detach().clone()
            for _ in range(30):
                for p in trainer.policy.parameters():
                    p.grad = torch.zeros_like(p)
                trainer.optimizer.step()
            return before, sigma.detach().clone()
        finally:
            cleanup()

    before, after = _drift(isolate=True)
    assert torch.equal(before, after), f"σ drifted with zero gradient: {before} → {after}"
    before, after = _drift(isolate=False)
    assert float((after - before).min()) > 1e-3, ("negative control: pre-T1 wiring must drift σ "
                                                  "upward, else this test proves nothing")


def test_sigma_group_anneals_on_the_same_schedule(simple_map):
    """T1(b), scheduler half: adding a param group without extending
    `scheduler.base_lrs` is a SILENT failure — CosineAnnealingLR.get_lr() zips
    base_lrs against param_groups non-strictly, so the σ group would keep its
    launch LR for the whole run while every other group anneals.
    """
    from cs2rl.train.loop import isolate_aim_log_std_param_group
    from tests._helpers.trainer_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=4,
                                               map_data=simple_map,
                                               aim_log_std_max=math.log(0.05))
    try:
        trainer.optimizer.param_groups[0]["weight_decay"] = 1e-4
        isolate_aim_log_std_param_group(trainer)
        opt, sch = trainer.optimizer, trainer.scheduler
        assert len(opt.param_groups) == 2
        assert len(sch.base_lrs) == len(opt.param_groups)
        assert opt.param_groups[0]["weight_decay"] == 1e-4
        assert opt.param_groups[1]["weight_decay"] == 0.0
        # σ moved, not copied: it must be in the new group and gone from group 0
        sigma_id = id(trainer.policy.aim_log_std)
        assert [id(p) for p in opt.param_groups[1]["params"]] == [sigma_id]
        assert sigma_id not in {id(p) for p in opt.param_groups[0]["params"]}

        before = [g["lr"] for g in opt.param_groups]
        for _ in range(3):
            sch.step()
        after = [g["lr"] for g in opt.param_groups]
        assert after[1] < before[1], "σ group's LR never annealed (missing base_lrs entry?)"
        assert after[1] == pytest.approx(after[0])
    finally:
        cleanup()


def test_raw_key_reports_a_sigma_pushed_past_the_cap():
    """T1(c): the `_raw` key carries the true (>cap) parameter while the
    clamped key saturates at the cap — the only way to tell a healthy σ resting
    at the cap from a frozen one that overshot it (gate pre-flight 5).
    """
    from cs2rl.train.metrics import log_aim_log_std
    cap = math.log(0.05)
    pol = _policy_with_cap(cap)
    # one entropy-maximising step, lr large enough to clear the 0.2 margin
    opt = torch.optim.Adam([pol.aim_log_std], lr=0.3)
    obs = torch.zeros(4, pol.obs_dim)
    (-pol.get_action_and_value(obs)[3].sum()).backward()
    opt.step()
    raw = float(pol.aim_log_std[0])
    assert raw > cap, raw

    logs = {}
    log_aim_log_std(pol, logs)
    assert logs["policy/aim_log_std_yaw_raw"] == pytest.approx(raw)
    assert logs["policy/aim_log_std_yaw"] == pytest.approx(cap)
    assert logs["policy/aim_log_std_pitch_raw"] == pytest.approx(float(pol.aim_log_std[1]))
    # and from here σ is frozen — which is precisely what the raw key exposes
    pol.zero_grad(set_to_none=True)
    (-pol.get_action_and_value(obs)[3].sum()).backward()
    assert float(pol.aim_log_std.grad.abs().max()) == 0.0


@pytest.mark.parametrize("split", [False, True], ids=["legacy", "split"])
def test_logged_sigma_clamps_to_the_policy_floor(split):
    """The σ log clamps to the policy's own floor, as forward() does.

    build_policy always passes LOG_STD_MIN as aim_log_std_min, so the test sets another
    floor on the built policy and parks every σ parameter below it: forward's log_std
    and every clamped σ key (per team on a split policy) must read that floor.
    """
    from cs2rl.train.metrics import log_aim_log_std
    pol = _policy_with_cap(None, tct_split_heads=split)
    floor = math.log(0.02)
    pol.aim_log_std_min = floor
    params = [pol.aim_log_std_t, pol.aim_log_std_ct] if split else [pol.aim_log_std]
    with torch.no_grad():
        for p in params:
            p.fill_(floor - 1.0)
        _, _, log_std, _ = pol(torch.zeros(4, pol.obs_dim), {})
    assert torch.allclose(log_std, torch.full_like(log_std, floor))
    logs = {}
    log_aim_log_std(pol, logs)
    clamped = [k for k in logs if not k.endswith("_raw")]
    assert len(clamped) == (6 if split else 2), sorted(logs)
    for key in clamped:
        assert logs[key] == pytest.approx(floor), key


def test_cap_too_close_to_the_sigma_floor_is_refused():
    """T1(d): a cap within AIM_LOG_STD_CAP_MIN_HEADROOM of LOG_STD_MIN would
    put the init at/below the FLOOR — dead at the lower clamp instead of the
    upper one. Rejected loudly rather than silently floored, and the message
    has to say why (the operator picked the number; only the error can tell
    them the band moved).
    """
    from cs2rl.policy import (
        AIM_LOG_STD_CAP_MIN_HEADROOM,
        LOG_STD_MIN,
        resolve_aim_log_std_init,
        validate_aim_log_std_max,
    )
    for bad in (LOG_STD_MIN + 0.1, LOG_STD_MIN + AIM_LOG_STD_CAP_MIN_HEADROOM):
        with pytest.raises(ValueError) as excinfo:
            validate_aim_log_std_max(bad)
        assert "LOG_STD_MIN" in str(excinfo.value)
    ok = LOG_STD_MIN + AIM_LOG_STD_CAP_MIN_HEADROOM + 1e-9
    assert validate_aim_log_std_max(ok) == pytest.approx(ok)
    assert resolve_aim_log_std_init(ok) > LOG_STD_MIN


def test_default_cap_leaves_the_init_at_log_std_init():
    """T1(e): no behaviour change for every run that does not pass a tight cap
    — the 5v5 default (and an omitted flag) still start at σ = 0.1.
    """
    from cs2rl.policy import LOG_STD_INIT, LOG_STD_MAX, resolve_aim_log_std_init
    assert resolve_aim_log_std_init(LOG_STD_MAX) == pytest.approx(LOG_STD_INIT)
    for cap in (LOG_STD_MAX, None):
        pol = _policy_with_cap(cap)
        assert torch.allclose(pol.aim_log_std, torch.full_like(pol.aim_log_std, LOG_STD_INIT))


def test_config_records_the_resolved_sigma_init():
    """T1: config.json carries `aim_log_std_init` — the gate reads σ movement
    as |raw − init|, so the baseline must be provenance, not folklore. Pinned
    against the SAME helper build_policy uses so the two cannot drift.
    """
    import types

    from cs2rl.policy import resolve_aim_log_std_init
    from cs2rl.train.config import build_train_config, compute_batch_dims
    _, bptt, bs = compute_batch_dims(16)
    for cap in (math.log(0.05), None):
        args = types.SimpleNamespace(device="cpu",
                                     seed=0,
                                     timesteps=bs,
                                     checkpoint_dir="/tmp/none",
                                     n_active_per_team=1,
                                     pin_pitch=0,
                                     crouch_enabled=0,
                                     aim_log_std_max=cap,
                                     aim_entropy_bonus=False)
        cfg = build_train_config(args, batch_size=bs, bptt_horizon=bptt)
        assert cfg["aim_log_std_init"] == pytest.approx(
            resolve_aim_log_std_init(cfg["aim_log_std_max"]))
