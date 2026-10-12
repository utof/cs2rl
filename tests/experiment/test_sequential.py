"""#417: the pre-registered sequential stop for a duel cell (cs2rl.experiment.sequential).

WHAT IS PINNED HERE
-------------------
* The boundary: the per-look levels of the published O'Brien-Fleming constant 2.040 for 5
  looks, and the look schedule ceil(k cap / 5).
* The false-decision rate under a hero share of 0.5, by a seeded Monte Carlo through
  ``sign_stop``: two arrangements with a T/CT gap (T share 0.68 / CT 0.32), 10% draws, cap 100
  per arrangement. #417 K-Q1 computed this point's exact rate as 0.02882; peeking at 1.96, a flat
  boundary, counting draws as losses, reporting "undecided at cap" as a decision, or looking at
  the T arrangement before the CT one each push the rate past 0.05.
* Draws are not losses; "undecided at cap" carries the fixed-n Wilson interval and no
  decision; every look sees equal rounds of every arrangement.
* The ladder's precision stop and its anytime-valid interval.
"""
import math

import numpy as np
import pytest
from scipy import special, stats

from cs2rl.experiment import sequential as Q


def test_the_boundary_is_the_published_obrien_fleming_one():
    # 2 (1 - Phi(2.040 sqrt(5 / k))), k = 1..5 (#417 K-Q1; Jennison & Turnbull's 2.040).
    assert Q.LOOK_ALPHAS == pytest.approx(
        (5.0770e-06, 1.25743e-03, 8.44778e-03, 2.256092e-02, 4.135033e-02), rel=1e-4)
    assert Q.look_rounds(1000) == (200, 400, 600, 800, 1000)
    assert Q.look_rounds(150) == (30, 60, 90, 120, 150)
    assert Q.look_rounds(101)[-1] == 101
    with pytest.raises(ValueError):
        Q.look_rounds(4)


def test_sign_p_is_binomtests_two_sided_p_value():
    pairs = [(w, lo) for w in range(0, 120, 7) for lo in range(0, 120, 11)] + [(5, 5), (1000, 940),
                                                                               (1, 0)]
    for w, lo in pairs:
        ref = stats.binomtest(w, w + lo, 0.5).pvalue if w + lo else 1.0
        assert Q.sign_p(w, lo) == pytest.approx(ref, rel=1e-9, abs=1e-12), (w, lo)


def _null_cells(rng, n_cells, cap, draw, gap):
    """Outcome lists for two arrangements whose pooled hero share is 0.5: the hero wins
    0.5 + gap of its decided rounds on T and 0.5 - gap on CT."""
    cells = []
    for _ in range(n_cells):
        arrs = []
        for p in (0.5 + gap, 0.5 - gap):
            probs = [(1 - draw) * p, (1 - draw) * (1 - p), draw]
            arrs.append(rng.choice(["win", "loss", "double"], size=cap, p=probs).tolist())
        cells.append(arrs)
    return cells


def test_the_false_decision_rate_under_a_half_share_is_at_most_alpha():
    # A stop that never decides would pass the upper bound, so the rate must also sit near the
    # exact value (4 standard errors).
    n = 2000
    cells = _null_cells(np.random.default_rng(417), n, cap=100, draw=0.10, gap=0.18)
    decided = sum(Q.sign_stop(arrs, 100).decision in ("hero", "opponent") for arrs in cells)
    rate, se = decided / n, math.sqrt(0.02882 * (1 - 0.02882) / n)
    assert 0.02882 - 4 * se <= rate <= Q.ALPHA, rate


def test_draws_cost_a_round_but_are_not_losses():
    # Per 10 rounds 3 wins, 1 loss, 6 draws: at look 4 (80 rounds) 24 wins against 8 losses,
    # p = 0.0070 <= 0.0226. Counted as losses the draws would make it 24 against 56.
    pattern = [
        "win", "double", "win", "timeout", "double", "win", "double", "loss", "double", "double"
    ]
    s = Q.sign_stop([pattern * 10], 100)
    assert (s.decision, s.looks[-1].look) == ("hero", 4)
    assert (s.looks[-1].wins, s.looks[-1].losses, s.looks[-1].draws) == (24, 8, 48)
    # The exact test, not its z-scale form (0.0047 here): that form exceeds 0.05 elsewhere on the
    # K-Q1 grid while staying under it at the Monte Carlo test's point.
    assert s.looks[-1].p_value == pytest.approx(stats.binomtest(24, 32).pvalue)
    assert s.wilson is None
    with pytest.raises(ValueError, match="not one of"):
        Q.sign_stop([["win"] * 19 + ["draw"]], 100)


def test_undecided_at_cap_is_no_decision_and_carries_the_fixed_n_wilson():
    # 55 wins, 45 losses in a steady mix: no look rejects (p 0.37 at the cap).
    s = Q.sign_stop([["win", "loss"] * 45 + ["win"] * 10], 100)
    assert s.decision == Q.UNDECIDED and len(s.looks) == Q.LOOKS
    ci = stats.binomtest(55, 100).proportion_ci(confidence_level=0.95, method="wilson")
    assert s.wilson == pytest.approx((ci.low, ci.high))
    assert Q.sign_stop([["double"] * 100], 100).wilson == pytest.approx((math.nan, math.nan),
                                                                        nan_ok=True)


def test_every_look_sees_equal_rounds_of_every_arrangement():
    # The hero wins every T round and loses every CT round: pooled over equal rounds it is
    # exactly even, so a correct stop never decides; a look that read T's extra rounds would.
    t, ct = ["win"] * 100, ["loss"] * 100
    s = Q.sign_stop([t, ct], 100)
    assert s.decision == Q.UNDECIDED
    assert [(lk.rounds, lk.wins, lk.losses) for lk in s.looks] == [(r, r, r)
                                                                   for r in Q.look_rounds(100)]
    running = Q.sign_stop([t, ct[:20]], 100)           # mid-cell: CT has played one block
    assert running.decision == Q.RUNNING and [lk.wins for lk in running.looks] == [20]
    with pytest.raises(ValueError, match="past the cap"):
        Q.sign_stop([t + ["win"], ct], 100)


def test_the_precision_stop_stops_once_the_interval_is_narrow_enough():
    s = Q.precision_stop([["win"] * 500, ["win"] * 500], 500)
    assert (s.status, s.rounds, s.wins) == ("precise", 40, 80)
    assert s.interval[1] - s.interval[0] <= Q.WIDTH
    lo, hi = Q.mixture_interval(60, 0)                 # the look before: 30 rounds a side
    assert hi - lo > Q.WIDTH
    even = Q.precision_stop([["win", "loss"] * 250, ["loss", "win"] * 250], 500)
    assert even.status == "at cap" and even.interval[1] - even.interval[0] > Q.WIDTH
    assert Q.precision_stop([["win"] * 500, ["win"] * 30], 500).status == Q.RUNNING


@pytest.mark.parametrize("wins,losses", [(0, 5), (5, 0), (30, 10), (642, 287), (500, 500)])
def test_the_mixture_interval_is_where_the_mixture_ratio_reaches_one_over_alpha(wins, losses):
    lo, hi = Q.mixture_interval(wins, losses)
    p = np.linspace(1e-9, 1 - 1e-9, 400001)
    log_m = special.betaln(1 + wins, 1 + losses) - special.xlogy(wins, p) - special.xlog1py(
        losses, -p)
    inside = p[log_m < math.log(1 / Q.ALPHA)]
    assert (lo, hi) == pytest.approx((inside.min(), inside.max()), abs=1e-5)
    assert Q.mixture_interval(0, 0) == (0.0, 1.0)
