"""Batch 7 T/CT policy-heads split — policy-level contracts.

Historical design: Batch 7, gh#112; current checkpoint contract: docs/formats.md#checkpoints
(tests 1, 2, 3, 4a, 6, 7, 9, 10 land in this file; the trainer-level
invariant re-pin is test 4b in tests/train/test_tag_trainer.py, the TAG param-group
partition is test 5a in tests/train/test_tag_diagnostic.py, the analyzer labeling is
test 5b in tests/experiment/test_analyze_tag.py, and the CLI round-trip is test 8 in
tests/train/test_train_cli.py).

Fixture pattern mirrors tests/train/test_tag_diagnostic.py: one real
make_puffer_env, policies built off it on CPU. The env is module-scoped
because make_puffer_env loads the nav graph / visibility matrix (~seconds)
and every test here only needs its observation space + static data.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from cs2rl import policy as policy_mod
from cs2rl.env.c.cs2_env import make_env
from cs2rl.spec import obs as spec_obs
from cs2rl.spec.action import AIM_DIM
from cs2rl.train import loop as train_loop
from cs2rl.train import metrics as train_metrics
from cs2rl.train import resume as train_resume
from cs2rl.train import selfplay as train_selfplay


@pytest.fixture(scope="module")
def env():
    e = make_env(seed=0)
    try:
        yield e
    finally:
        e.close()


# The legacy parameter-name set, SPELLED OUT (spec §5 test 1). Hardcoded on
# purpose: diffing the flag-off constructor against itself would be
# tautological, so the pin is against this literal list. A construction change
# — renamed module, extra layer, split leaking into the flag-off path — fails
# here loudly instead of silently changing what every checkpoint contains.
LEGACY_PARAM_NAMES = {
    "encoder.0.weight",
    "encoder.0.bias",
    "encoder.2.weight",
    "encoder.2.bias",
    "lstm.weight_ih_l0",
    "lstm.weight_hh_l0",
    "lstm.bias_ih_l0",
    "lstm.bias_hh_l0",
    "action_heads.0.weight",
    "action_heads.0.bias",
    "action_heads.1.weight",
    "action_heads.1.bias",
    "action_heads.2.weight",
    "action_heads.2.bias",
    "action_heads.3.weight",
    "action_heads.3.bias",
    "action_heads.4.weight",
    "action_heads.4.bias",
    "action_heads.5.weight",
    "action_heads.5.bias",
    "action_heads.6.weight",
    "action_heads.6.bias",
    "value_head.weight",
    "value_head.bias",
    "aim_mu.weight",
    "aim_mu.bias",
    "aim_log_std",
}


def _obs(n_t, n_ct, seed=0):
    """(n_t + n_ct, OBS_DIM) batch: first n_t rows are T (obs[24] == 1)."""
    torch.manual_seed(seed)
    x = torch.randn(n_t + n_ct, spec_obs.OBS_DIM) * 0.5
    x[:n_t, 24] = 1.0
    x[n_t:, 24] = 0.0
    return x


def test_flag_off_builds_exactly_the_legacy_modules(env):
    """Spec §5 test 1 (structural half): the default constructor's parameter
    name set is EXACTLY the legacy list, no _t/_ct module exists, and the
    split marker is False. Behavioral coverage of the legacy path comes from
    the whole existing suite, which exercises it heavily.
    """
    p = policy_mod.build_policy(env, device="cpu")
    assert {n for n, _ in p.named_parameters()} == LEGACY_PARAM_NAMES
    assert p.tct_split_heads is False
    assert p.tct_split_trunk is False
    assert not any(
        n.endswith(("_t", "_ct")) or "_t." in n or "_ct." in n for n, _ in p.named_parameters())
    assert not hasattr(p, "aim_log_std_t")


def test_flag_off_forward_equals_direct_legacy_head_application(env):
    """Spec §5 test 1 (behavioral half): with the flag off, forward's outputs
    equal the heads applied directly to the trunk output — proving the blend
    path is not executed at all (a blend with a degenerate all-ones mask would
    also match numerically, so this asserts against the SAME module objects
    and is paired with the structural assertion above).
    """
    p = policy_mod.build_policy(env, device="cpu")
    x = _obs(4, 4)
    logits, mu, log_std, value = p(x, state={})
    # Replicate forward's trunk with the IDENTICAL op sequence (reshape →
    # seq-first → zero-state LSTM → flatten back) so the comparison is
    # bitwise, not merely close — a re-derivation via _forward_core would
    # differ in reduction order and force a tolerance that hides real bugs.
    B, TT = x.shape[0], 1
    h = p.encoder(x.reshape(B * TT, x.shape[-1]).float())
    h = h.reshape(B, TT, p.hidden_size).transpose(0, 1)
    hc = (h.new_zeros(1, B, p.hidden_size), h.new_zeros(1, B, p.hidden_size))
    h, _ = p.lstm(h, hc)
    h = h.transpose(0, 1).reshape(B * TT, p.hidden_size)
    for i, lg in enumerate(logits):
        assert torch.equal(lg, p.action_heads[i](h))
    assert torch.equal(value, p.value_head(h))
    assert torch.equal(mu, torch.tanh(p.aim_mu(h)) * p.max_turn_speed)


def test_split_constructor_shapes_and_names(env):
    """The split policy carries BOTH head copies and no shared copy — the
    shared trunk + value head are untouched (spec §3.1).
    """
    p = policy_mod.build_policy(env, device="cpu", tct_split_heads=True)
    names = {n for n, _ in p.named_parameters()}
    assert p.tct_split_heads is True
    for stem in ("action_heads_t.0.weight", "action_heads_ct.0.weight", "aim_mu_t.weight",
                 "aim_mu_ct.weight", "aim_log_std_t", "aim_log_std_ct"):
        assert stem in names, stem
    assert "aim_log_std" not in names
    assert not any(n.startswith(("action_heads.", "aim_mu.")) for n in names)
    assert {"encoder.0.weight", "lstm.weight_ih_l0", "value_head.weight"} <= names
    assert p.aim_log_std_t.shape == (AIM_DIM, )
    assert "tct_split_heads" not in p.state_dict(), \
        "the split marker must be a plain attribute, never a state_dict entry"


def test_pure_team_batch_leaves_other_copy_gradient_exactly_zero(env):
    """Spec §5 test 2: a batch of pure-T rows must leave EVERY CT-copy
    parameter's gradient exactly zero (and vice versa); a mixed batch makes
    both nonzero. This is the routing correctness proof — the blend weight is
    0 on the other team's copy, so autograd contributes literally nothing.
    """
    p = policy_mod.build_policy(env, device="cpu", tct_split_heads=True)

    def _grads(x):
        p.zero_grad(set_to_none=True)
        logits, mu, _log_std, _v = p(x, state={})
        (sum(lg.sum() for lg in logits) + mu.sum()).backward()
        out = {}
        for name, param in p.named_parameters():
            out[name] = 0.0 if param.grad is None else param.grad.abs().sum().item()
        return out

    g_t = _grads(_obs(8, 0))
    assert all(v == 0.0 for n, v in g_t.items() if "_ct" in n), \
        "pure-T batch leaked gradient into a CT copy"
    assert any(v > 0.0 for n, v in g_t.items() if "_t" in n)

    g_ct = _grads(_obs(0, 8))
    assert all(v == 0.0 for n, v in g_ct.items() if "_t" in n), \
        "pure-CT batch leaked gradient into a T copy"
    assert any(v > 0.0 for n, v in g_ct.items() if "_ct" in n)

    g_mix = _grads(_obs(4, 4))
    assert any(v > 0.0 for n, v in g_mix.items() if "_t" in n)
    assert any(v > 0.0 for n, v in g_mix.items() if "_ct" in n)


def test_obs_bit_selects_the_serving_copy_in_every_forward_path(env):
    """Spec §5 test 4a: flipping obs[24] on a row flips which copy serves it,
    in forward (2D and 3D) and forward_eval. Constructed by zeroing one copy's
    aim_mu bias and setting the other's to a marker value, so the served
    copy is readable straight off mu_aim's sign.

    PITFALL under test (spec §3.2): on a 3D (B, T, obs) input the mask must be
    x[..., 24].reshape(B*TT, 1). Writing x[:, 24] there selects TIMESTEP 24 —
    the exact silent bug this test exists to catch, which is why the 3D case
    uses T > 1 with a per-row (not per-timestep) team assignment.
    """
    p = policy_mod.build_policy(env, device="cpu", tct_split_heads=True)
    with torch.no_grad():
        for m in (p.aim_mu_t, p.aim_mu_ct):
            m.weight.zero_()
        p.aim_mu_t.bias.fill_(5.0)     # tanh(+5) ≈ +1 → mu > 0 means "T copy served"
        p.aim_mu_ct.bias.fill_(-5.0)   # tanh(-5) ≈ -1 → mu < 0 means "CT copy served"

    x2 = _obs(3, 3)
    for path in (lambda z: p(z, state={}), lambda z: p.forward_eval(z, state={})):
        _lg, mu, _ls, _v = path(x2)
        assert (mu[:3] > 0).all(), "T rows must be served by the _t copy"
        assert (mu[3:] < 0).all(), "CT rows must be served by the _ct copy"

    # 3D training path: 4 segments × 6 timesteps; segments 0/1 are T, 2/3 CT.
    x3 = torch.randn(4, 6, spec_obs.OBS_DIM) * 0.5
    x3[:2, :, 24] = 1.0
    x3[2:, :, 24] = 0.0
    _lg, mu3, _ls, _v = p(x3, state={})
    mu3 = mu3.reshape(4, 6, AIM_DIM)
    assert (mu3[:2] > 0).all(), "3D path: T segments must hit the _t copy"
    assert (mu3[2:] < 0).all(), "3D path: CT segments must hit the _ct copy"


def test_log_std_is_clamped_per_copy_then_blended(env):
    """Spec §3.2: clamp each copy, THEN blend. With one copy driven far above
    LOG_STD_MAX the served value must be the CLAMP, not a blend of raw
    parameters — same result for a 0/1 mask either way, but this pins the
    order §3.6's per-team σ logs depend on.
    """
    p = policy_mod.build_policy(env, device="cpu", tct_split_heads=True)
    with torch.no_grad():
        p.aim_log_std_t.fill_(10.0)    # far above LOG_STD_MAX
        p.aim_log_std_ct.fill_(policy_mod.LOG_STD_MIN)
    _lg, _mu, log_std, _v = p(_obs(2, 2), state={})
    assert torch.allclose(log_std[:2], torch.full_like(log_std[:2], policy_mod.LOG_STD_MAX))
    assert torch.allclose(log_std[2:], torch.full_like(log_std[2:], policy_mod.LOG_STD_MIN))


def test_get_action_and_value_routes_by_team(env):
    """Spec §3.1: get_action_and_value is split for consistency even though no
    production path calls it — every in-tree caller is a test
    (test_hybrid_sample_writes_two_buffers in tests/train/test_train_env.py, plus
    tests/train/test_aim_log_std_max.py and tests/train/test_pitch_pin.py); train_bc.py
    uses forward_eval. Same marker trick as the forward test, read off the
    returned value/continuous action.
    """
    p = policy_mod.build_policy(env, device="cpu", tct_split_heads=True)
    with torch.no_grad():
        for m in (p.aim_mu_t, p.aim_mu_ct):
            m.weight.zero_()
        p.aim_mu_t.bias.fill_(5.0)
        p.aim_mu_ct.bias.fill_(-5.0)
        p.aim_log_std_t.fill_(policy_mod.LOG_STD_MIN)  # σ ≈ 0.01: sample ≈ μ
        p.aim_log_std_ct.fill_(policy_mod.LOG_STD_MIN)
    torch.manual_seed(0)
    _a, cont, _lp, _ent, _v, _st = p.get_action_and_value(_obs(3, 3))
    assert (cont[:3] > 0).all()
    assert (cont[3:] < 0).all()


def _legacy_frozen_state_dict(env):
    """A legacy state_dict with BC-frozen σ — the bc_warmstart.pt signature.

    Synthetic rather than reading outputs/checkpoints/bc_warmstart.pt so the
    test runs on any checkout; the real file's σ-widening is asserted at
    launch time via the startup stdout check (plan Task 9.2).
    """
    p = policy_mod.build_policy(env, device="cpu")
    with torch.no_grad():
        p.aim_log_std.fill_(policy_mod.LOG_STD_INIT)
    return {k: v.clone() for k, v in p.state_dict().items()}


def test_state_dict_is_split_discriminates_both_vintages(env):
    """Spec §3.3: split-ness is read off the KEYS, at every load."""
    legacy = policy_mod.build_policy(env, device="cpu").state_dict()
    split = policy_mod.build_policy(env, device="cpu", tct_split_heads=True).state_dict()
    assert policy_mod.state_dict_is_split(legacy) is False
    assert policy_mod.state_dict_is_split(split) is True


def test_state_dict_is_trunk_split_and_convert():
    sd = {
        "encoder.0.weight": torch.ones(2, 2),
        "lstm.weight_ih_l0": torch.ones(3, 3),
        "aim_log_std": torch.zeros(2),
        "value_head.weight": torch.ones(1, 2),
    }
    assert policy_mod.state_dict_is_split(sd) is False
    assert policy_mod.state_dict_is_trunk_split(sd) is False
    out = train_resume.convert_shared_trunk_to_split(sd)
    assert "encoder_t.0.weight" in out and "encoder_ct.0.weight" in out
    assert "encoder.0.weight" not in out
    assert "lstm_t.weight_ih_l0" in out and "lstm_ct.weight_ih_l0" in out
    assert "aim_log_std" in out        # heads untouched
    assert policy_mod.state_dict_is_trunk_split(out) is True


def test_warm_split_duplicates_heads_and_reinits_sigma_in_both_copies(env):
    """Spec §5 test 3: the warm-split path applied in the spec's ORDER
    (re-init σ on the legacy dict FIRST, then duplicate) leaves both copies
    equal to the legacy tensors and σ == AIM_LOG_STD_RESUME_INIT in BOTH
    aim_log_std_t and aim_log_std_ct.

    This is the gh#91 trap: reinit_frozen_aim_log_std matches
    endswith("aim_log_std"), which is FALSE for "aim_log_std_t" — running it
    after duplication would leave σ=0.1 and throttle every update of a 30M
    run with no error message.
    """
    legacy = _legacy_frozen_state_dict(env)
    assert train_resume.reinit_frozen_aim_log_std(legacy) is True
    split_sd = train_resume.convert_legacy_state_dict_to_split(legacy)

    for copy in ("aim_log_std_t", "aim_log_std_ct"):
        assert torch.allclose(split_sd[copy],
                              torch.full_like(split_sd[copy],
                                              train_resume.AIM_LOG_STD_RESUME_INIT)), copy
    assert "aim_log_std" not in split_sd
    for i in range(7):
        for suffix in ("weight", "bias"):
            src = legacy[f"action_heads.{i}.{suffix}"]
            assert torch.equal(split_sd[f"action_heads_t.{i}.{suffix}"], src)
            assert torch.equal(split_sd[f"action_heads_ct.{i}.{suffix}"], src)
    assert torch.equal(split_sd["aim_mu_t.weight"], legacy["aim_mu.weight"])
    assert torch.equal(split_sd["aim_mu_ct.weight"], legacy["aim_mu.weight"])
    assert torch.equal(split_sd["encoder.0.weight"], legacy["encoder.0.weight"])
    assert torch.equal(split_sd["value_head.weight"], legacy["value_head.weight"])

    # and it actually loads
    p = policy_mod.build_policy(env, device="cpu", tct_split_heads=True)
    p.load_state_dict(split_sd)


def test_reinit_matcher_also_catches_split_sigma_keys(env):
    """Spec §3.3 belt-and-braces: the widened matcher catches a future
    split-format warmstart by VALUE too, so a split checkpoint whose σ is
    still frozen at log(0.1) is widened on resume like a legacy one.
    """
    p = policy_mod.build_policy(env, device="cpu", tct_split_heads=True)
    with torch.no_grad():
        p.aim_log_std_t.fill_(policy_mod.LOG_STD_INIT)
        p.aim_log_std_ct.fill_(policy_mod.LOG_STD_INIT)
    sd = {k: v.clone() for k, v in p.state_dict().items()}
    assert train_resume.reinit_frozen_aim_log_std(sd) is True
    for copy in ("aim_log_std_t", "aim_log_std_ct"):
        assert torch.allclose(sd[copy],
                              torch.full_like(sd[copy], train_resume.AIM_LOG_STD_RESUME_INIT))


def test_arch_mismatch_raises_naming_both_architectures(env):
    """Spec §3.3: never a silent partial load. The error must name what the
    checkpoint is AND what the policy is — a bare load_state_dict KeyError
    tells the operator neither.
    """
    legacy_p = policy_mod.build_policy(env, device="cpu")
    split_p = policy_mod.build_policy(env, device="cpu", tct_split_heads=True)
    split_sd = split_p.state_dict()
    legacy_sd = legacy_p.state_dict()

    with pytest.raises(ValueError, match=r"SPLIT.*LEGACY|LEGACY.*SPLIT"):
        policy_mod.load_state_dict_arch_checked(legacy_p, split_sd, source="snap.pt")
    with pytest.raises(ValueError, match=r"SPLIT.*LEGACY|LEGACY.*SPLIT"):
        policy_mod.load_state_dict_arch_checked(split_p, legacy_sd, source="snap.pt")


def test_split_checkpoint_round_trips_bitwise(env):
    """Spec §5 test 3 (third clause): split → split is a plain load."""
    a = policy_mod.build_policy(env, device="cpu", tct_split_heads=True)
    b = policy_mod.build_policy(env, device="cpu", tct_split_heads=True)
    policy_mod.load_state_dict_arch_checked(b, a.state_dict(), source="a")
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.equal(pa, pb), n


def test_resolve_resume_split_infers_and_never_narrows(env, tmp_path):
    """Spec §5 test 9 + §3.3's ordering constraint: the pre-build_policy sniff.

    Four cases, and the one that matters most is row 3 — a SPLIT checkpoint
    resumed WITHOUT the flag still builds a split policy. That is the
    crash-resume path, and on this GPU box crash-resume is a first-class case,
    not an edge (the operator relaunches without re-reading the flag list).
    Each flag can only WIDEN its axis; it can never narrow split→legacy.
    These fixtures have no encoder_t.0.weight, so the trunk bit stays False
    when trunk_flag is omitted/False.
    """
    legacy_pt = tmp_path / "legacy.pt"
    split_pt = tmp_path / "split.pt"
    torch.save(policy_mod.build_policy(env, device="cpu").state_dict(), legacy_pt)
    torch.save(
        policy_mod.build_policy(env, device="cpu", tct_split_heads=True).state_dict(), split_pt)

    heads, trunk, sd, path = train_resume.resolve_resume_split(None,
                                                               heads_flag=False,
                                                               trunk_flag=False)
    assert (heads, trunk, sd, path) == (False, False, None, None)
    heads, trunk, sd, _ = train_resume.resolve_resume_split(None, heads_flag=True, trunk_flag=False)
    assert heads is True and trunk is False and sd is None

    heads, trunk, sd, path = train_resume.resolve_resume_split(str(legacy_pt),
                                                               heads_flag=False,
                                                               trunk_flag=False)
    assert heads is False and trunk is False and sd is not None and path == legacy_pt
    heads, trunk, _sd, _ = train_resume.resolve_resume_split(str(legacy_pt),
                                                             heads_flag=True,
                                                             trunk_flag=False)
    assert heads is True, "flag must widen a legacy checkpoint to a warm split"
    assert trunk is False

    heads, trunk, sd, _ = train_resume.resolve_resume_split(str(split_pt),
                                                            heads_flag=False,
                                                            trunk_flag=False)
    assert heads is True, "split checkpoint must be detected without the flag"
    assert trunk is False, "heads-split fixtures have no encoder_t.0.weight"
    assert "aim_log_std_t" in sd, "the sniffed dict must be returned for reuse"
    heads, trunk, _sd, _ = train_resume.resolve_resume_split(str(split_pt),
                                                             heads_flag=True,
                                                             trunk_flag=False)
    assert heads is True
    assert trunk is False

    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        train_resume.resolve_resume_split(str(tmp_path / "nope.pt"),
                                          heads_flag=False,
                                          trunk_flag=False)


def test_self_play_loads_both_checkpoint_vintages(env, tmp_path):
    """Spec §5 test 6: a legacy snapshot AND a split snapshot each load into
    the past-policy slot without error.

    Why this is load-bearing rather than tidy: self-play activates on ~30% of
    epochs (p_past=0.3) and load_past_policy gets no config — under a
    flag-only design a split run would crash hours in, on a random epoch. Key
    inference makes both vintages work in both directions, so a split run can
    also mix in pre-split snapshots from an earlier pool.
    """
    legacy_pt = tmp_path / "past_legacy.pt"
    split_pt = tmp_path / "past_split.pt"
    torch.save(policy_mod.build_policy(env, device="cpu").state_dict(), legacy_pt)
    torch.save(
        policy_mod.build_policy(env, device="cpu", tct_split_heads=True).state_dict(), split_pt)

    for path, expect_split in ((legacy_pt, False), (split_pt, True)):
        # Constructor is all-default at HEAD, and Cs2PuffeRL._draw_past_policy calls
        # `self._self_play_mgr.load_past_policy(dev, self.vecenv)` — mirrored here.
        mgr = train_selfplay.SelfPlayManager()
        mgr.pool = [path]
        past = mgr.load_past_policy("cpu", env)
        assert past is not None, path
        assert past.tct_split_heads is expect_split, path
        assert not past.training, "past policies must be in eval mode"


def test_load_policy_from_checkpoint_infers_split(env, tmp_path):
    """Spec §3.3: the eval/record loader keeps working on split checkpoints —
    it constructs from the keys, so no flag reaches it and none is needed.
    """
    split_pt = tmp_path / "eval_split.pt"
    torch.save(
        policy_mod.build_policy(env, device="cpu", tct_split_heads=True).state_dict(), split_pt)
    p = policy_mod.load_policy_from_checkpoint(split_pt, "cpu")
    assert p.tct_split_heads is True

    # Legacy vintage through the same loader — this function has no other test
    # coverage in the repo, so pin both directions here.
    legacy_pt = tmp_path / "eval_legacy.pt"
    torch.save(policy_mod.build_policy(env, device="cpu").state_dict(), legacy_pt)
    p = policy_mod.load_policy_from_checkpoint(legacy_pt, "cpu")
    assert p.tct_split_heads is False


def test_log_aim_log_std_legacy_keys_unchanged(env):
    """Spec §5 test 10 (legacy half): on a legacy policy the emitted keys are
    exactly the two clamped ones (no per-team keys), with today's values.

    Rung 1a T1 added a `_raw` twin per clamped key (the UNCLAMPED parameter —
    see log_aim_log_std's docstring), so the exact set is four. The point of
    the assert is unchanged: a legacy policy must not emit `_t`/`_ct` keys.
    Both σ values here are inside the band, so raw == clamped.
    """
    p = policy_mod.build_policy(env, device="cpu")
    with torch.no_grad():
        p.aim_log_std.copy_(torch.tensor([-1.5, -2.0]))
    logs = {}
    train_metrics.log_aim_log_std(p, logs)
    assert set(logs) == {
        "policy/aim_log_std_yaw", "policy/aim_log_std_pitch", "policy/aim_log_std_yaw_raw",
        "policy/aim_log_std_pitch_raw"
    }
    assert logs["policy/aim_log_std_yaw"] == pytest.approx(-1.5)
    assert logs["policy/aim_log_std_pitch"] == pytest.approx(-2.0)
    assert logs["policy/aim_log_std_yaw_raw"] == pytest.approx(-1.5)
    assert logs["policy/aim_log_std_pitch_raw"] == pytest.approx(-2.0)


def test_log_aim_log_std_split_emits_mean_plus_per_team(env):
    """Spec §5 test 10 + §3.6 (the remedy for review-1 BLOCKER 2): under the
    split the legacy keys become the MEAN of the two CLAMPED copies — that
    preserves the T7 acceptance gate and every dashboard consumer — and four
    per-team keys carry the actually interesting signal (do the teams learn
    different aim noise?).

    The clamp is applied per copy BEFORE averaging, matching the forward path:
    aim_log_std_t is set above LOG_STD_MAX here, so a mean-then-clamp
    implementation lands on a different number and fails.

    Rung 1a T1 review (I1/M1): this also pins the FULL emitted key set and
    every `_raw` key. The two aggregations differ on purpose — the clamped
    legacy key is the MEAN of the copies, the raw legacy key is the MAX (see
    log_aim_log_std's docstring) — and yaw_t above the cap is the one setup
    where raw != clamped, so both conventions are exercised here.
    """
    p = policy_mod.build_policy(env, device="cpu", tct_split_heads=True)
    with torch.no_grad():
        p.aim_log_std_t.copy_(torch.tensor([10.0, -2.0]))              # yaw above LOG_STD_MAX
        p.aim_log_std_ct.copy_(torch.tensor([-3.0, -1.0]))
    logs = {}
    train_metrics.log_aim_log_std(p, logs)
    assert set(logs) == {
        "policy/aim_log_std_yaw",
        "policy/aim_log_std_pitch",
        "policy/aim_log_std_yaw_raw",
        "policy/aim_log_std_pitch_raw",
        "policy/aim_log_std_yaw_t",
        "policy/aim_log_std_yaw_ct",
        "policy/aim_log_std_pitch_t",
        "policy/aim_log_std_pitch_ct",
        "policy/aim_log_std_yaw_t_raw",
        "policy/aim_log_std_yaw_ct_raw",
        "policy/aim_log_std_pitch_t_raw",
        "policy/aim_log_std_pitch_ct_raw",
    }
    assert logs["policy/aim_log_std_yaw_t"] == pytest.approx(policy_mod.LOG_STD_MAX)
    assert logs["policy/aim_log_std_yaw_ct"] == pytest.approx(-3.0)
    assert logs["policy/aim_log_std_pitch_t"] == pytest.approx(-2.0)
    assert logs["policy/aim_log_std_pitch_ct"] == pytest.approx(-1.0)
    assert logs["policy/aim_log_std_yaw"] == pytest.approx(0.5 * (policy_mod.LOG_STD_MAX + -3.0))
    assert logs["policy/aim_log_std_pitch"] == pytest.approx(-1.5)
                                                                       # Per-team raws are the untouched parameters, cap or no cap.
    assert logs["policy/aim_log_std_yaw_t_raw"] == pytest.approx(10.0)
    assert logs["policy/aim_log_std_yaw_ct_raw"] == pytest.approx(-3.0)
    assert logs["policy/aim_log_std_pitch_t_raw"] == pytest.approx(-2.0)
    assert logs["policy/aim_log_std_pitch_ct_raw"] == pytest.approx(-1.0)
                                                                       # Legacy-named raws are the MAX over the copies (10.0, not the mean 3.5;
                                                                       # -1.0, not the mean -1.5).
    assert logs["policy/aim_log_std_yaw_raw"] == pytest.approx(10.0)
    assert logs["policy/aim_log_std_pitch_raw"] == pytest.approx(-1.0)


def test_log_aim_log_std_split_raw_key_exposes_a_single_capped_copy(env):
    """Rung 1a T1 review I1: one team's σ above the cap — i.e. clamp-frozen and
    gradient-dead — must show through `policy/aim_log_std_yaw_raw`, the key the
    gate reads to answer "is raw <= cap".

    Regression against the mean aggregation this key used to carry: at
    cap = log 0.05 = -2.9957, a T copy at cap + 0.5 = -2.4957 (dead) averaged
    with a healthy CT copy at -4.0 reads -3.2479 — BELOW the cap, so the
    pre-flight check passes on a frozen sigma. The max reads -2.4957 and the
    check fails, which is the honest answer.
    """
    cap = math.log(0.05)
    p = policy_mod.build_policy(env, device="cpu", tct_split_heads=True, aim_log_std_max=cap)
    with torch.no_grad():
        p.aim_log_std_t.copy_(torch.tensor([cap + 0.5, -4.0]))         # yaw: above the cap
        p.aim_log_std_ct.copy_(torch.tensor([-4.0, -4.0]))             # healthy on both dims
    logs = {}
    train_metrics.log_aim_log_std(p, logs)
                                                                       # The dead copy is visible per-team...
    assert logs["policy/aim_log_std_yaw_t_raw"] == pytest.approx(cap + 0.5)
    assert logs["policy/aim_log_std_yaw_ct_raw"] == pytest.approx(-4.0)
                                                                       # ...and, the point of I1, through the legacy-named key the gate reads.
    assert logs["policy/aim_log_std_yaw_raw"] == pytest.approx(cap + 0.5)
    assert logs["policy/aim_log_std_yaw_raw"] > cap
                                                                       # The mean convention would have landed here and read as healthy.
    assert logs["policy/aim_log_std_yaw_raw"] != pytest.approx(0.5 * (cap + 0.5 + -4.0))
                                                                       # The CLAMPED twin is untouched by this fix: still the per-copy-clamped mean,
                                                                       # and still censored at the cap (which is exactly why the raw key exists).
    assert logs["policy/aim_log_std_yaw_t"] == pytest.approx(cap)
    assert logs["policy/aim_log_std_yaw"] == pytest.approx(0.5 * (cap + -4.0))


def test_status_line_keeps_the_t7_gate_substring(env):
    """Spec §5 test 10 (last clause): T7 acceptance gate 2 greps stdout for
    'aim_log_std_pitch=' — a split run must still print it.
    """
    p = policy_mod.build_policy(env, device="cpu", tct_split_heads=True)
    logs = {}
    train_metrics.log_aim_log_std(p, logs)
    # Verified at HEAD: format_train_status(epoch, ts_val, logs) — the exact
    # signature the outer loop calls. The contract under test is only that the
    # formatted line still contains the 'aim_log_std_pitch=' substring the T7
    # gate greps.
    line = train_metrics.format_train_status(7, 0.5, logs)
    assert "aim_log_std_pitch=" in line


def test_head_divergence_zero_at_warm_split_and_keys_present(env):
    """Spec §5 test 7 (first clause): split/head_l2_rel/* keys exist and are
    ~0 immediately after a warm split, because both copies are identical.
    Legacy policies emit nothing (the metric is undefined without two copies).
    """
    legacy = _legacy_frozen_state_dict(env)
    split_sd = train_resume.convert_legacy_state_dict_to_split(legacy)
    p = policy_mod.build_policy(env, device="cpu", tct_split_heads=True)
    p.load_state_dict(split_sd)
    d = train_metrics.compute_head_divergence(p)
    assert set(d) == {
        "split/head_l2_rel/action_heads", "split/head_l2_rel/aim_mu",
        "split/head_l2_rel/aim_log_std"
    }
    assert all(v == 0.0 for v in d.values()), d
    assert train_metrics.compute_head_divergence(policy_mod.build_policy(env, device="cpu")) == {}


def test_head_divergence_exceeds_the_decay_aware_null(env):
    """Spec §5 test 7 + §4 Q3: divergence under team-asymmetric advantages
    must exceed a same-steps, same-optimizer control with ZERO advantages.

    Why a control at all: the optimizer carries weight_decay=1e-4 (the Adam
    construction in train()) which moves even a zero-gradient copy, and head_l2_rel
    is a RATIO — shrinking both copies inflates it with no learning at all. So
    raw L2 has no achievable null and the honest comparison is
    asymmetric-vs-zero advantage over the identical number of steps.

    The two arms start from the same state, seeded with a small pre-existing
    T/CT gap. A zero-gap start would make the null trivially 0.0 (decay acts
    identically on identical copies) and the inequality vacuous; seeding the
    gap is what makes the control actually control something.
    """

    def _arm(asymmetric):
        torch.manual_seed(0)
        p = policy_mod.build_policy(env, device="cpu", tct_split_heads=True)
        with torch.no_grad():          # identical seeded gap in both arms
            p.aim_mu_ct.bias.add_(0.05)
            p.action_heads_ct[0].bias.add_(0.05)
        opt = torch.optim.Adam(p.parameters(), lr=1e-3, weight_decay=1e-4)
        x = _obs(8, 8, seed=1)
        adv = (torch.cat([torch.ones(8), -torch.ones(8)]) if asymmetric else torch.zeros(16))
        for _ in range(20):
            logits, mu, _ls, _v = p(x, state={})
            logp = sum(F.log_softmax(lg, dim=-1)[:, 0] for lg in logits) + mu.sum(-1)
            opt.zero_grad(set_to_none=True)
            (-(adv * logp).mean()).backward()
            opt.step()
        return train_metrics.compute_head_divergence(p)

    treated = _arm(asymmetric=True)
    null = _arm(asymmetric=False)
    for key in ("split/head_l2_rel/action_heads", "split/head_l2_rel/aim_mu"):
        assert treated[key] > null[key], (
            f"{key}: asymmetric-advantage divergence {treated[key]:.6f} did not exceed "
            f"the zero-advantage weight-decay floor {null[key]:.6f}")
    assert math.isfinite(null["split/head_l2_rel/aim_log_std"])


def test_pure_team_batch_zeros_other_trunk_grad(env):
    """A pure-T batch must leave every CT-trunk parameter's gradient at 0.

    WHAT: encoder_ct/lstm_ct get no autograd contribution from T-only rows;
    every encoder_t/lstm_t parameter that requires grad is in the graph.

    WHY: this is the trunk routing proof (heads already have
    test_pure_team_batch_leaves_other_copy_gradient_exactly_zero). A shared
    LSTM on mixed encoder outputs, or a leftover self.encoder, fails here.

    PITFALL: T=1 + zero LSTM state makes weight_hh a zero-times-h path, so
    the input is stacked to T>1. 2D `_obs(4, 0)` would leave lstm_t.weight_hh
    at exactly 0 and fail a correct implementation.
    """
    p = policy_mod.build_policy(env, "cpu", tct_split_heads=True, tct_split_trunk=True)
    # T>1 so lstm.weight_hh is in the graph (T=1 zero-state zeroes it).
    x = _obs(4, 0).unsqueeze(1).expand(-1, 3, -1).contiguous()
    state = {}
    logits, mu, ls, v = p.forward(x, state)
    (sum(lg.sum() for lg in logits) + mu.sum() + v.sum()).backward()
    assert not hasattr(p, "encoder") and not hasattr(p, "lstm")
    assert hasattr(p, "encoder_t") and hasattr(p, "lstm_t")
    assert hasattr(p, "encoder_ct") and hasattr(p, "lstm_ct")
    for n, par in p.named_parameters():
        if n.startswith(("encoder_ct.", "lstm_ct.")):
            assert par.grad is None or float(par.grad.abs().sum()) == 0.0, n
        if n.startswith(("encoder_t.", "lstm_t.")) and par.requires_grad:
            assert par.grad is not None and float(par.grad.abs().sum()) > 0.0, n


def test_mixed_batch_nonzero_both_trunk_grads(env):
    """A mixed T/CT batch must send grad into BOTH team trunks.

    WHAT: encoder-only AND LSTM-only prefixes are asserted separately for
    each team, on both the 2D training path and the seq-len-1 eval path.

    WHY: a bug that splits the encoder but shares one LSTM (or never calls
    lstm_ct) still produces encoder_ct grads from a mixed batch; the LSTM
    half is what fails that implementation. forward_eval goes through
    _forward_core, so a trunk-aware forward() with a stale _forward_core
    would pass the 3D half and fail here.

    PITFALL: T=1 zero-state zeroes weight_hh; the 3D case uses T>1 so the
    recurrent weights are actually in the graph.
    """
    p = policy_mod.build_policy(env, "cpu", tct_split_heads=True, tct_split_trunk=True)

    def _assert_both_trunks(x):
        p.zero_grad(set_to_none=True)
        logits, mu, _ls, v = p.forward(x, state={})
        (sum(lg.sum() for lg in logits) + mu.sum() + v.sum()).backward()
        for prefix in ("encoder_t.", "encoder_ct.", "lstm_t.", "lstm_ct."):
            matched = [(n, par) for n, par in p.named_parameters()
                       if n.startswith(prefix) and par.requires_grad]
            assert matched, prefix
            assert any(par.grad is not None and float(par.grad.abs().sum()) > 0.0
                       for _n, par in matched), prefix

    _assert_both_trunks(_obs(2, 2))
    _assert_both_trunks(_obs(2, 2).unsqueeze(1).expand(-1, 3, -1).contiguous())

    p.zero_grad(set_to_none=True)
    logits, mu, _ls, v = p.forward_eval(_obs(2, 2), state={})
    (sum(lg.sum() for lg in logits) + mu.sum() + v.sum()).backward()
    for prefix in ("encoder_t.", "encoder_ct.", "lstm_t.", "lstm_ct."):
        matched = [(n, par) for n, par in p.named_parameters()
                   if n.startswith(prefix) and par.requires_grad]
        assert any(par.grad is not None and float(par.grad.abs().sum()) > 0.0
                   for _n, par in matched), f"forward_eval {prefix}"


def test_obs24_flip_switches_trunk(env):
    """Flip obs[24] changes value_head output on a trunk-split policy.

    WHAT: the same row with only column 24 flipped must produce a different
    shared-critic value (and a different hidden), because it is encoded and
    recurred by the other team trunk.

    WHY: heads-only split still has one encoder+LSTM, so flipping the team
    bit would not change value_head(hidden). This is the trunk-routing
    observable that does not depend on the split heads.

    PITFALL (spec §3.2): the 3D case must use x[..., 24] — x[:, 24] on a
    (B, T, obs) input selects timestep 24. T=6 here is short enough that
    that form would IndexError; T is still >1 so a silent wrong-timestep
    mask on a longer horizon is the class of bug the 3D assert pins.
    """
    p = policy_mod.build_policy(env, "cpu", tct_split_heads=True, tct_split_trunk=True)
    assert p.tct_split_trunk is True
    assert "tct_split_trunk" not in p.state_dict()

    x = _obs(1, 0)
    x_flip = x.clone()
    x_flip[:, 24] = 1.0 - x_flip[:, 24]
    with torch.no_grad():
        *_, v = p.forward(x, state={})
        *_, v_flip = p.forward(x_flip, state={})
        *_, v_eval = p.forward_eval(x, state={})
        *_, v_eval_flip = p.forward_eval(x_flip, state={})
    assert not torch.allclose(v, v_flip), "2D forward: flipping obs[24] must switch trunks"
    assert not torch.allclose(v_eval, v_eval_flip), "forward_eval must inherit trunk routing"

    x3 = torch.randn(2, 6, spec_obs.OBS_DIM) * 0.5
    x3[:, :, 24] = 1.0
    x3_flip = x3.clone()
    x3_flip[:, :, 24] = 0.0
    with torch.no_grad():
        *_, v3 = p.forward(x3, state={})
        *_, v3_flip = p.forward(x3_flip, state={})
    assert not torch.allclose(v3, v3_flip), "3D forward: flipping obs[24] must switch trunks"


def test_resolve_resume_split_two_bits(tmp_path, env):
    import torch
    # heads-only ckpt + both flags omitted → heads on, trunk off
    p = policy_mod.build_policy(env, "cpu", tct_split_heads=True, tct_split_trunk=False)
    path = tmp_path / "heads.pt"
    torch.save(p.state_dict(), path)
    h, t, sd, rp = train_resume.resolve_resume_split(path, heads_flag=False, trunk_flag=False)
    assert (h, t) == (True, False)
    # trunk-only + omitted flags → trunk on, heads off
    p2 = policy_mod.build_policy(env, "cpu", tct_split_heads=False, tct_split_trunk=True)
    path2 = tmp_path / "trunk.pt"
    torch.save(p2.state_dict(), path2)
    h, t, _, _ = train_resume.resolve_resume_split(path2, heads_flag=False, trunk_flag=False)
    assert (h, t) == (False, True)
    # heads-only + trunk_flag True → both on (widen)
    h, t, _, _ = train_resume.resolve_resume_split(path, heads_flag=False, trunk_flag=True)
    assert (h, t) == (True, True)


def test_legacy_warm_split_both_axes_sigma_then_heads_then_trunk(env):
    """Spec §3.3 order: σ re-init → heads convert → trunk convert.

    WHAT/WHY: bc_warmstart-shaped dict + both flags. Both head copies equal
    the re-inited σ; both trunk copies equal the legacy encoder/lstm.
    Heads-first-then-σ would leave σ=0.1 (gh#91).

    PITFALL: production order only — trunk-first still works on a legacy
    dict (heads keys pass through) and would hide a σ-order regression.
    """
    legacy = _legacy_frozen_state_dict(env)
    enc = {k: v.clone() for k, v in legacy.items() if k.startswith("encoder.")}
    lstm = {k: v.clone() for k, v in legacy.items() if k.startswith("lstm.")}
    assert train_resume.reinit_frozen_aim_log_std(legacy) is True
    sd = train_resume.convert_legacy_state_dict_to_split(legacy)
    sd = train_resume.convert_shared_trunk_to_split(sd)
    for copy in ("aim_log_std_t", "aim_log_std_ct"):
        assert torch.allclose(sd[copy],
                              torch.full_like(sd[copy], train_resume.AIM_LOG_STD_RESUME_INIT)), copy
    assert "encoder.0.weight" not in sd
    assert "encoder_t.0.weight" in sd
    for suf, src in enc.items():
        stem = suf[len("encoder."):]
        assert torch.equal(sd[f"encoder_t.{stem}"], src)
        assert torch.equal(sd[f"encoder_ct.{stem}"], src)
    for suf, src in lstm.items():
        stem = suf[len("lstm."):]
        assert torch.equal(sd[f"lstm_t.{stem}"], src)
        assert torch.equal(sd[f"lstm_ct.{stem}"], src)


def test_trunk_split_into_heads_only_policy_raises_naming_trunk(env):
    """Trunk-split ckpt into a heads-only (tct_split_trunk=False) policy.

    WHAT/WHY: the message must name trunk. Heads is SPLIT on both sides
    here, so a SPLIT/LEGACY-only regex would miss this case.
    """
    heads_only = policy_mod.build_policy(env,
                                         device="cpu",
                                         tct_split_heads=True,
                                         tct_split_trunk=False)
    trunk_sd = policy_mod.build_policy(env,
                                       device="cpu",
                                       tct_split_heads=True,
                                       tct_split_trunk=True).state_dict()
    with pytest.raises(ValueError, match=r"trunk"):
        policy_mod.load_state_dict_arch_checked(heads_only, trunk_sd, source="trunk.pt")


def test_trunk_split_checkpoint_round_trips_bitwise(env):
    """Trunk-split → trunk-split is a plain load_state_dict."""
    a = policy_mod.build_policy(env, device="cpu", tct_split_heads=True, tct_split_trunk=True)
    b = policy_mod.build_policy(env, device="cpu", tct_split_heads=True, tct_split_trunk=True)
    policy_mod.load_state_dict_arch_checked(b, a.state_dict(), source="a")
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.equal(pa, pb), n


def test_loaders_infer_both_bits_and_warm_split_trunk_only(env, tmp_path):
    """Heads-only + trunk_flag widens trunk only; omitted flags infer both;
    load_policy_from_checkpoint does not KeyError on encoder_t.

    WHAT/WHY: trunk-split files have no encoder.0.weight. Crash-resume
    without flags must rebuild both axes from keys.

    PITFALL: this warm-split is trunk-only (heads already split).
    """
    import inspect

    heads_pt = tmp_path / "heads.pt"
    both_pt = tmp_path / "both.pt"
    trunk_pt = tmp_path / "trunk.pt"
    torch.save(
        policy_mod.build_policy(env, "cpu", tct_split_heads=True,
                                tct_split_trunk=False).state_dict(), heads_pt)
    torch.save(
        policy_mod.build_policy(env, "cpu", tct_split_heads=True,
                                tct_split_trunk=True).state_dict(), both_pt)
    torch.save(
        policy_mod.build_policy(env, "cpu", tct_split_heads=False,
                                tct_split_trunk=True).state_dict(), trunk_pt)

    # Heads-only + trunk_flag=True → both on; warm-split trunk only.
    h, t, sd, _ = train_resume.resolve_resume_split(heads_pt, heads_flag=False, trunk_flag=True)
    assert (h, t) == (True, True)
    assert policy_mod.state_dict_is_split(sd) and not policy_mod.state_dict_is_trunk_split(sd)
    warm = train_resume.convert_shared_trunk_to_split(sd)
    assert policy_mod.state_dict_is_trunk_split(warm)
    assert "aim_log_std_t" in warm and "aim_log_std" not in warm
    assert "encoder.0.weight" not in warm and "encoder_t.0.weight" in warm

    # both-split + flags omitted → both bits; build_policy accepts them.
    h, t, _, _ = train_resume.resolve_resume_split(both_pt, heads_flag=False, trunk_flag=False)
    assert (h, t) == (True, True)
    p = policy_mod.build_policy(env, "cpu", tct_split_heads=h, tct_split_trunk=t)
    assert p.tct_split_heads is True and p.tct_split_trunk is True

    # load_policy_from_checkpoint on a trunk-split file (no encoder.0.weight).
    loaded = policy_mod.load_policy_from_checkpoint(trunk_pt, "cpu")
    assert loaded.tct_split_trunk is True
    assert loaded.tct_split_heads is False
    loaded_both = policy_mod.load_policy_from_checkpoint(both_pt, "cpu")
    assert loaded_both.tct_split_heads is True and loaded_both.tct_split_trunk is True

    # self-play pool has no config — must infer both bits from keys.
    mgr = train_selfplay.SelfPlayManager()
    mgr.pool = [trunk_pt]
    past = mgr.load_past_policy("cpu", env)
    assert past is not None and past.tct_split_trunk is True and past.tct_split_heads is False

    # Train-main must not discard the resolved trunk bit.
    src = inspect.getsource(train_loop.train)
    assert "tct_split_trunk=tct_split_trunk" in src
    assert ", _tct_split_trunk," not in src
    assert "duplicated the shared encoder+LSTM into per-team" in src
