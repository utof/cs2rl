"""TAG diagnostic — trainer-contract tests (spec 2026-08-13 §5, tests 1/3/4).

Uses train_test_harness._build_trainer_for_test, whose trainer is Cs2PuffeRL
(gh#168 W1.5), whose train() is the return-norm body (a method since gh#168
W2a), the same pattern as tests/train/test_warmstart_entropy_trainer.py. Config keys
(target_kl, tag_*) are injected into trainer.config after construction; they
are read per train() call, not at construction.

PITFALL: target_kl is set to None in TAG trainer tests so the KL early-stop
cannot end the update mid-way — the mbL measurement then lands
deterministically on total_minibatches-1 for the key-name assertions. (In
production a KL trip is handled: the hook fires on the last EXECUTED
minibatch, spec §4.2.) The no-perturbation test sets it on BOTH arms so the
two runs stay comparable.
"""
import math
import re
from types import SimpleNamespace

import pytest
import torch


def _build(tag_on, seed=0):
    from tests._helpers.trainer_harness import _build_trainer_for_test
    torch.manual_seed(seed)            # identical policy init both arms
    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True, seed=seed)
    trainer.config["target_kl"] = None
    if tag_on:
        trainer.config["tag_diagnostic"] = True
        trainer.config["tag_every"] = 1
    return trainer, cleanup


def _run_once(trainer):
    trainer.evaluate()
    trainer.last_log_time = 0.0
    trainer.train()


@pytest.mark.training
def test_flag_off_is_inert(monkeypatch):
    """Spec §5 test 1: flag off ⇒ helper never called, nothing pending in _tag_metrics."""
    # PATCH THE MODULE THE CALL SITE RESOLVES THROUGH, NOT the one that defines
    # the function: tag_grad_cossim is DEFINED in cs2rl.train.update (formerly
    # train_update.py), but its call site is Cs2PuffeRL._record_tag in
    # src/cs2rl/train/trainer.py, which imports the name at module level and
    # resolves it through trainer.py's globals. A patch on the defining module is
    # unreachable and this test would pass while asserting nothing — the positive control below
    # (test_monkeypatch_target_actually_reaches_the_hook) goes red if the patch
    # point drifts again.
    from cs2rl.train import trainer as tag_mod
    calls = []
    real = tag_mod.tag_grad_cossim
    monkeypatch.setattr(tag_mod, "tag_grad_cossim",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    trainer, cleanup = _build(tag_on=False)
    try:
        _run_once(trainer)
        assert calls == [], "tag_grad_cossim ran with the flag off"
        assert trainer._tag_metrics is None
    finally:
        cleanup()


@pytest.mark.training
def test_monkeypatch_target_actually_reaches_the_hook(monkeypatch):
    """Positive pin on the PATCH POINT used by test_flag_off_is_inert above.

    WHAT: force the flag ON with the same monkeypatch and assert the
    interceptor fired. Nothing else — this measures reachability, not TAG.

    WHY it is a separate test: `test_flag_off_is_inert` asserts `calls == []`,
    which is green whether or not the patch can reach the call site at all. A
    patch aimed at the wrong module is therefore indistinguishable from a
    correctly-inert hook, and the test silently stops testing anything. That
    is not hypothetical: the call site in `Cs2PuffeRL._record_tag`
    (src/cs2rl/train/trainer.py) resolves `tag_grad_cossim` through `trainer`'s globals, so
    patching `train` (its pre-2026-08-31 home, a re-exporting shim) or
    `train_update` (where it is defined, and where the call site lived until
    W2a) is unreachable — measured, both directions, at each move. This test
    goes red for that, for a future move of the call site, and for a changed
    call site.

    PITFALL: assert ONLY `calls != []`. With the flag on, `_tag_metrics` is
    populated, so reusing the inert test's second assert here would fail for
    a reason that has nothing to do with the patch point.

    COST: one real trainer build (~12 s). That is the price of the pin — do
    not swap in a stub trainer, which would stop exercising the real call
    site and reintroduce exactly the vacuity this test exists to prevent.
    """
    from cs2rl.train import trainer as tag_mod
    calls = []
    real = tag_mod.tag_grad_cossim
    monkeypatch.setattr(tag_mod, "tag_grad_cossim",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    trainer, cleanup = _build(tag_on=True)
    try:
        _run_once(trainer)
        assert calls != [], (
            "monkeypatch never intercepted tag_grad_cossim with the flag ON — the patch "
            "point is unreachable, so test_flag_off_is_inert is green while asserting nothing")
    finally:
        cleanup()


def test_inject_tag_metrics_lifecycle():
    """Spec §5 test 1 (injection half): _inject_tag_metrics moves pending
    metrics into logs — INCLUDING NaN values, which must survive to
    metrics.jsonl (the DeadRunDetector constraint is positional, handled by
    call order in the outer loop, not by filtering) — and tolerates
    logs=None (throttled epoch: the top-of-loop reset drops the
    measurement; injecting here would mislabel its epoch).
    """
    from cs2rl.train.metrics import _inject_tag_metrics
    trainer = SimpleNamespace(_tag_metrics={
        "tag/cossim_cross/trunk/mb0": 0.4,
        "tag/cossim_within_ct/trunk/mb0": float("nan"),
    })
    logs = {}
    _inject_tag_metrics(trainer, logs)
    assert logs["tag/cossim_cross/trunk/mb0"] == 0.4
    assert math.isnan(logs["tag/cossim_within_ct/trunk/mb0"])

    # The trainer declares _tag_metrics at construction: no getattr default to hide a rename.
    with pytest.raises(AttributeError, match="_tag_metrics"):
        _inject_tag_metrics(SimpleNamespace(), {})
    _inject_tag_metrics(SimpleNamespace(_tag_metrics=None), {})
    _inject_tag_metrics(trainer, None)                 # throttled epoch: no crash


_KEY_RE = re.compile(r"^tag/(cossim_cross|cossim_cross_half|cossim_within_t|cossim_within_ct"
                     r"|gnorm_t|gnorm_ct)/(trunk|policy_heads)/(mb0|mbL)$"
                     r"|^tag/cossim_vf/(mb0|mbL)$"
                     r"|^tag/(selfplay_active|mbL_index)$")


@pytest.mark.training
def test_flag_on_emits_final_key_names():
    """Spec §5 tests 1+4 (positive half): the hook fires at mb0 AND mbL,
    every emitted key matches the FINAL metric-name contract the analyzer
    parses, and the bookkeeping keys are present — guards against a
    silently no-op'd hook passing the bitwise test.

    Cos-sims may legitimately be NaN on a degenerate early-training batch
    (zero-norm subset, spec §4.2) — so assert finite OR the matching gnorm
    is 0, not blanket finiteness (plan-review finding 15).
    """
    trainer, cleanup = _build(tag_on=True)
    try:
        _run_once(trainer)
        m = trainer._tag_metrics
        assert m, "hook did not populate trainer._tag_metrics"
        for k in m:
            assert _KEY_RE.match(k), f"unexpected metric key {k!r}"
        assert m["tag/selfplay_active"] in (0.0, 1.0)
        assert m["tag/mbL_index"] >= 0
        for mb in ("mb0", "mbL"):
            for g in ("trunk", "policy_heads"):
                c = m[f"tag/cossim_cross/{g}/{mb}"]
                gt = m[f"tag/gnorm_t/{g}/{mb}"]
                gct = m[f"tag/gnorm_ct/{g}/{mb}"]
                assert math.isfinite(c) or gt == 0.0 or gct == 0.0, (
                    f"NaN cos-sim at {g}/{mb} without a zero-norm side "
                    f"(gnorm_t={gt}, gnorm_ct={gct})")
    finally:
        cleanup()


@pytest.mark.training
def test_row_mask_matches_obs_team_bit():
    """Spec §5 test 3: (segment % 10) < 5 ⇔ team T, pinned against the
    INDEPENDENT obs-side team bit obs[24] = (team == 0) the C env writes
    (src/cs2rl/env/c/cs2_observations.h:96; spawn slots src/cs2rl/env/c/cs2_round.h:32).
    Fails if PufferLib segment ordering or the env's slot layout changes.
    """
    trainer, cleanup = _build(tag_on=False)
    try:
        trainer.evaluate()
        seg = torch.arange(trainer.segments)
        expected_t = ((seg % 10) < 5).float()
        # Probe timestep 1, not 0: on the FIRST evaluate() the t=0 slot still
        # holds the zero-initialized pre-step obs (verified empirically:
        # eval-0 t0 is all zeros, t1+ and every later eval match exactly).
        # The TAG mask keys on segment index, never on obs content, so the
        # invariant under test is unaffected by which timestep we pin.
        team_bits = trainer.observations[:, 1, 24].float().cpu()
        assert torch.equal(team_bits,
                           expected_t), ("segment%10 team mask disagrees with obs[24] team bit — "
                                         "PufferLib segment ordering or env slot layout changed")
    finally:
        cleanup()


@pytest.mark.training
def test_flag_on_does_not_perturb_training_bitwise():
    """Spec §5 test 4: one evaluate+train with the flag on vs off, identical
    seeds, CPU ⇒ bitwise-equal post-step parameters. CPU keeps this exact —
    a CUDA tolerance compare would hide real bugs. The helper-called
    positive assertion lives in test_flag_on_emits_final_key_names.
    """
    trainer_a, cleanup_a = _build(tag_on=False, seed=0)
    try:
        _run_once(trainer_a)
        params_off = {n: p.detach().clone() for n, p in trainer_a.policy.named_parameters()}
    finally:
        cleanup_a()

    trainer_b, cleanup_b = _build(tag_on=True, seed=0)
    try:
        _run_once(trainer_b)
        for n, p in trainer_b.policy.named_parameters():
            assert torch.equal(params_off[n],
                               p.detach()), (f"parameter {n} diverged with --tag-diagnostic on")
    finally:
        cleanup_b()


@pytest.mark.training
def test_row_mask_matches_obs_team_bit_on_a_split_trainer():
    """Spec §5 test 4b: the team-identity INVARIANT re-pinned on the split
    path.

    Two independent identity sources must agree or the whole batch is
    meaningless: TAG partitions by slot index ((idx % 10) < 5, inside
    `tag_grad_cossim` — which moved out of train.py with its patcher on
    2026-08-31 and now lives in src/cs2rl/train_update.py, :1403 at that commit;
    search the symbol, not the line) while the split routes by the obs bit
    obs[24]. If they ever
    disagree, TAG would silently measure the wrong partition of a correctly
    routed network — and nothing would crash. The legacy-path version of this
    pin is test_row_mask_matches_obs_team_bit above; this one proves the
    invariant survives building the trainer with a split policy.
    """
    from tests._helpers.trainer_harness import _build_trainer_for_test
    torch.manual_seed(0)
    trainer, cleanup = _build_trainer_for_test(num_envs=32,
                                               with_selfplay=True,
                                               seed=0,
                                               tct_split_heads=True)
    trainer.config["target_kl"] = None
    try:
        assert trainer.policy.tct_split_heads is True
        trainer.evaluate()
        seg = torch.arange(trainer.segments)
        expected_t = ((seg % 10) < 5).float()
        # Probe timestep 1, NOT 0: on the FIRST-ever evaluate() the t=0 slot
        # still holds the zero-initialized pre-step obs (same artifact the
        # legacy pin above documents and dodges the same way).
        team_bits = trainer.observations[:, 1, 24].float().cpu()
        assert torch.equal(team_bits, expected_t), (
            "segment%10 team mask disagrees with obs[24] team bit on the split path — "
            "TAG would measure a different partition than the router")
    finally:
        cleanup()


@pytest.mark.training
def test_row_mask_matches_obs_team_bit_on_a_both_flags_trainer():
    """Spec §5 test 4 (trunk half): re-pin obs[24] == ((idx % 10) < 5)
    on a both-flags trainer.

    WHAT: tct_split_heads=True, tct_split_trunk=True; after one evaluate()
    the segment-index team mask still equals the C-env team bit. Probe
    timestep 1, not 0 (same t=0 zero-obs artifact as the heads-only pin).

    WHY: trunk routing also keys on obs[24]. If PufferLib segment order
    or the env slot layout drifted only under a split trunk, TAG would
    measure the wrong partition of a correctly routed network. The
    heads-only pin is test_row_mask_matches_obs_team_bit_on_a_split_trainer.

    PITFALL: do not probe t=0 on the first evaluate() — that slot is
    still the zero-initialized pre-step obs.
    """
    from tests._helpers.trainer_harness import _build_trainer_for_test
    torch.manual_seed(0)
    trainer, cleanup = _build_trainer_for_test(num_envs=32,
                                               with_selfplay=True,
                                               seed=0,
                                               tct_split_heads=True,
                                               tct_split_trunk=True)
    trainer.config["target_kl"] = None
    try:
        assert trainer.policy.tct_split_heads is True
        assert trainer.policy.tct_split_trunk is True
        trainer.evaluate()
        seg = torch.arange(trainer.segments)
        expected_t = ((seg % 10) < 5).float()
        team_bits = trainer.observations[:, 1, 24].float().cpu()
        assert torch.equal(
            team_bits, expected_t), ("segment%10 team mask disagrees with obs[24] team bit on the "
                                     "both-flags path — TAG would measure a different partition "
                                     "than the trunk router")
    finally:
        cleanup()
