"""Bit-exact equivalence between the old per-scalar reward loop and the batched one.

The rollout in src/cs2rl/train.py used to build three single-scalar CUDA tensors per
env per tick to normalise the reward channels. process_step_rewards() batches
that into one host-to-device copy + one symlog. Training values must not move by
a single ULP, so `_reference_loop` below is a verbatim transcription of the old
inline code (torch scalar tensors, WelfordStd.normalize, per-env symlog) and
every test asserts torch.equal — not allclose — against it.

Both paths are driven with independent-but-identically-seeded WelfordStd
instances so the update/normalize interleaving is exercised for real.
"""
import numpy as np
import pytest
import torch

from cs2rl.train_helpers_batch1 import (
    WelfordStd,
    process_step_rewards,
    split_into_channels,
    symlog,
)

_SS_DTYPE = [
    ("reward_kills", "f4"),
    ("reward_deaths", "f4"),
    ("reward_shots", "f4"),
    ("reward_bomb", "f4"),
    ("reward_pbrs", "f4"),
    ("reward_survival", "f4"),
    ("reward_inaction", "f4"),
    ("reward_win", "f4"),
    ("win_by_detonation", "i4"),
    ("win_by_defuse", "i4"),
    ("bomb_planted", "i4"),
]

AGENTS_PER_ENV = 10


class _StepStatsStub:
    """Mimics Cs2Env.StepStatsView: dict-like, ndim==0, supports .get()."""
    ndim = 0

    def __init__(self, rec):
        self._rec = rec

    def __getitem__(self, key):
        return self._rec[key]

    def get(self, key, default=None):
        try:
            return self._rec[key]
        except (KeyError, ValueError):
            return default


def _make_ss(rng, *, bomb_planted=0, scale=1.0):
    rec = np.zeros(1, dtype=_SS_DTYPE)[0]
    for f, _ in _SS_DTYPE:
        if f.startswith("reward_"):
            rec[f] = np.float32(rng.normal(0.0, scale))
    rec["win_by_detonation"] = int(rng.random() < 0.05)
    rec["win_by_defuse"] = int(rng.random() < 0.05)
    rec["bomb_planted"] = bomb_planted
    return _StepStatsStub(rec)


def _new_welfords():
    return (WelfordStd(prior_std=1.0, min_count=8), WelfordStd(prior_std=1.0, min_count=8),
            WelfordStd(prior_std=1.0, min_count=8))


def _reference_loop(info, r, agents_per_env, wc, wo, wp, event=None):
    """Verbatim transcription of the pre-batching inline loop in src/cs2rl/train.py."""
    dev = r.device
    r_new = torch.empty_like(r)
    for e in range(len(info)):
        row_start = e * agents_per_env
        row_end = row_start + agents_per_env
        ss = info[e].get("step_stats", None) if isinstance(info[e], dict) else None
        if ss is None:
            r_new[row_start:row_end] = r[row_start:row_end]
            continue
        channels = split_into_channels(ss)
        wc.update(channels["combat"])
        wo.update(channels["objective"])
        wp.update(channels["positional"])
        if event is not None and bool(int(ss.get("bomb_planted", 0))):
            event[row_start:row_end] = True
        combat_t = torch.tensor(channels["combat"], device=dev, dtype=r.dtype)
        objective_t = torch.tensor(channels["objective"], device=dev, dtype=r.dtype)
        positional_t = torch.tensor(channels["positional"], device=dev, dtype=r.dtype)
        r_sum = (wc.normalize(combat_t) + wo.normalize(objective_t) + wp.normalize(positional_t))
        r_new[row_start:row_end] = symlog(r_sum)
    used = len(info) * agents_per_env
    if used < r.shape[0]:
        r_new[used:] = r[used:]
    return r_new


def _run_both(ticks, n_rows_fn=None, device="cpu"):
    """Drive a sequence of ticks through both paths; return (ref_out, new_out) lists."""
    ref_w = _new_welfords()
    new_w = _new_welfords()
    scratch = np.empty(max(len(t) for t in ticks) if ticks else 1, dtype=np.float32)
    ref_out, new_out = [], []
    for i, info in enumerate(ticks):
        n_rows = n_rows_fn(i, info) if n_rows_fn else len(info) * AGENTS_PER_ENV
        r = torch.arange(n_rows, dtype=torch.float32, device=device) * 0.125 - 3.0
        ref_out.append(_reference_loop(info, r, AGENTS_PER_ENV, *ref_w))
        new_out.append(process_step_rewards(info, r, AGENTS_PER_ENV, *new_w, scratch))
    return ref_out, new_out


def _assert_bit_exact(ref_out, new_out):
    for i, (a, b) in enumerate(zip(ref_out, new_out, strict=True)):
        assert torch.equal(a, b), (f"tick {i} diverged\n ref={a[::AGENTS_PER_ENV]}\n "
                                   f"new={b[::AGENTS_PER_ENV]}")


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_bit_exact_over_long_sequence(device):
    """64 ticks x 32 envs: crosses min_count, so std() flips from prior to real mid-run."""
    rng = np.random.default_rng(0)
    ticks = [[{"step_stats": _make_ss(rng)} for _ in range(32)] for _ in range(64)]
    _assert_bit_exact(*_run_both(ticks, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_bit_exact_with_wide_magnitudes(device):
    """Mixed scales stress the float32-vs-float64 divergence the batching could cause."""
    rng = np.random.default_rng(7)
    ticks = [[{
        "step_stats": _make_ss(rng, scale=10.0**(e % 5 - 2))
    } for e in range(16)] for _ in range(40)]
    _assert_bit_exact(*_run_both(ticks, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_ss_is_none_fallback_passes_raw_r_through(device):
    """Envs without step_stats keep raw r (NOT symlog'd) while their neighbours normalise."""
    rng = np.random.default_rng(3)
    ticks = []
    for t in range(12):
        info = []
        for e in range(8):
            if (e + t) % 3 == 0:
                info.append({} if e % 2 == 0 else "not-a-dict")
            else:
                info.append({"step_stats": _make_ss(rng)})
        ticks.append(info)
    ref_out, new_out = _run_both(ticks, device=device)
    _assert_bit_exact(ref_out, new_out)
    # Guard against both paths being trivially equal (e.g. all-fallback batches).
    assert any(o.unique().numel() > 4 for o in new_out)


@pytest.mark.parametrize("device", DEVICES)
def test_short_info_tail_preserved(device):
    """len(info)*agents_per_env < r.shape[0]: trailing rows must survive untouched."""
    rng = np.random.default_rng(11)
    ticks = [[{"step_stats": _make_ss(rng)} for _ in range(6)] for _ in range(10)]
    ref_out, new_out = _run_both(ticks,
                                 n_rows_fn=lambda i, info: (len(info) + 2) * AGENTS_PER_ENV,
                                 device=device)
    _assert_bit_exact(ref_out, new_out)
    assert torch.equal(new_out[0][60:],
                       torch.arange(60, 80, dtype=torch.float32, device=device) * 0.125 - 3.0)


@pytest.mark.parametrize("device", DEVICES)
def test_welford_state_identical_after_sequence(device):
    """The interleaved update/normalize order must leave both estimators identical."""
    rng = np.random.default_rng(5)
    ticks = [[{"step_stats": _make_ss(rng)} for _ in range(9)] for _ in range(30)]
    ref_w, new_w = _new_welfords(), _new_welfords()
    scratch = np.empty(9, dtype=np.float32)
    for info in ticks:
        r = torch.zeros(90, dtype=torch.float32, device=device)
        _reference_loop(info, r, AGENTS_PER_ENV, *ref_w)
        process_step_rewards(info, r, AGENTS_PER_ENV, *new_w, scratch)
    for a, b in zip(ref_w, new_w, strict=True):
        assert (a.count, a.mean, a.m2) == (b.count, b.mean, b.m2)
        assert a.std() == b.std()


@pytest.mark.parametrize("device", DEVICES)
def test_bomb_planted_event_rows_match(device):
    """The per-agent-row event accumulator must be written identically."""
    rng = np.random.default_rng(13)
    ticks = [[{
        "step_stats": _make_ss(rng, bomb_planted=int((e + t) % 7 == 0))
    } for e in range(5)] for t in range(9)]
    ref_w, new_w = _new_welfords(), _new_welfords()
    scratch = np.empty(5, dtype=np.float32)
    ref_ev = torch.zeros(50, dtype=torch.bool, device=device)
    new_ev = torch.zeros(50, dtype=torch.bool, device=device)
    for info in ticks:
        r = torch.zeros(50, dtype=torch.float32, device=device)
        _reference_loop(info, r, AGENTS_PER_ENV, *ref_w, event=ref_ev)
        process_step_rewards(info,
                             r,
                             AGENTS_PER_ENV,
                             *new_w,
                             scratch,
                             current_segment_has_event=new_ev)
    assert ref_ev.any() and torch.equal(ref_ev, new_ev)


@pytest.mark.parametrize("device", DEVICES)
def test_repeat_factor_is_not_hardcoded(device):
    """agents_per_env is dynamic — verify a non-10 factor broadcasts correctly."""
    rng = np.random.default_rng(17)
    info = [{"step_stats": _make_ss(rng)} for _ in range(4)]
    wref, wnew = _new_welfords(), _new_welfords()
    r = torch.zeros(4 * 3, dtype=torch.float32, device=device)
    ref = _reference_loop(info, r, 3, *wref)
    new = process_step_rewards(info, r, 3, *wnew, np.empty(4, dtype=np.float32))
    assert torch.equal(ref, new)
    assert new[0] == new[1] == new[2] and new[0] != new[3]
