"""TAG diagnostic — trainer-contract tests (spec 2026-08-13 §5, tests 1/3/4).

Uses train_test_harness._build_trainer_for_test + _patch_trainer_with_
return_norm, the same pattern as tests/test_warmstart_entropy_trainer.py.
Config keys are injected into trainer.config before patching.

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

import torch


def _build(tag_on, seed=0):
    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test
    torch.manual_seed(seed)            # identical policy init both arms
    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True, seed=seed)
    trainer.config["target_kl"] = None
    if tag_on:
        trainer.config["tag_diagnostic"] = True
        trainer.config["tag_every"] = 1
    _patch_trainer_with_return_norm(trainer)
    return trainer, cleanup


def _run_once(trainer):
    trainer.evaluate()
    trainer.last_log_time = 0.0
    trainer.train()


def test_flag_off_is_inert(monkeypatch):
    """Spec §5 test 1: flag off ⇒ helper never called, no _tag_metrics."""
    import train as train_mod
    calls = []
    real = train_mod.tag_grad_cossim
    monkeypatch.setattr(train_mod, "tag_grad_cossim",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    trainer, cleanup = _build(tag_on=False)
    try:
        _run_once(trainer)
        assert calls == [], "tag_grad_cossim ran with the flag off"
        assert getattr(trainer, "_tag_metrics", None) in (None, {})
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
    from train import _inject_tag_metrics
    trainer = SimpleNamespace(_tag_metrics={
        "tag/cossim_cross/trunk/mb0": 0.4,
        "tag/cossim_within_ct/trunk/mb0": float("nan"),
    })
    logs = {}
    _inject_tag_metrics(trainer, logs)
    assert logs["tag/cossim_cross/trunk/mb0"] == 0.4
    assert math.isnan(logs["tag/cossim_within_ct/trunk/mb0"])

    _inject_tag_metrics(SimpleNamespace(), {})         # no stash: no-op
    _inject_tag_metrics(SimpleNamespace(_tag_metrics=None), {})
    _inject_tag_metrics(trainer, None)                 # throttled epoch: no crash


_KEY_RE = re.compile(r"^tag/(cossim_cross|cossim_cross_half|cossim_within_t|cossim_within_ct"
                     r"|gnorm_t|gnorm_ct)/(trunk|policy_heads)/(mb0|mbL)$"
                     r"|^tag/cossim_vf/(mb0|mbL)$"
                     r"|^tag/(selfplay_active|mbL_index)$")


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


def test_row_mask_matches_obs_team_bit():
    """Spec §5 test 3: (segment % 10) < 5 ⇔ team T, pinned against the
    INDEPENDENT obs-side team bit obs[24] = (team == 0) the C env writes
    (src/c_env/cs2_observations.h:96; spawn slots src/c_env/cs2_round.h:32).
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


def test_row_mask_matches_obs_team_bit_on_a_split_trainer():
    """Spec §5 test 4b: the team-identity INVARIANT re-pinned on the split
    path.

    Two independent identity sources must agree or the whole batch is
    meaningless: TAG partitions by slot index ((idx % 10) < 5, src/train.py
    ~:3237) while the split routes by the obs bit obs[24]. If they ever
    disagree, TAG would silently measure the wrong partition of a correctly
    routed network — and nothing would crash. The legacy-path version of this
    pin is test_row_mask_matches_obs_team_bit above; this one proves the
    invariant survives building the trainer with a split policy.
    """
    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test
    torch.manual_seed(0)
    trainer, cleanup = _build_trainer_for_test(num_envs=32,
                                               with_selfplay=True,
                                               seed=0,
                                               tct_split_heads=True)
    trainer.config["target_kl"] = None
    _patch_trainer_with_return_norm(trainer)
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
