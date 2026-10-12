"""Pre-registered sequential stopping for a duel cell (#417): stop a cell once its answer is in.

WHAT
----
A duel cell (cs2rl.experiment.oracle_duel, duel_eval) asks whether the hero wins more than half
of its decided rounds against one opponent, and used to answer it at a fixed round count. Here
the cell is played in LOOKS blocks up to a cap of rounds per arrangement and tested after each
block; it stops at the first look whose test rejects 0.5. Pure scipy: no env import.

``sign_stop`` answers that sign question. ``precision_stop`` is for ORDERING questions (a ladder
of dial values or profiles), where the cell's estimate matters, not its side of 0.5: it stops
when an anytime-valid interval of the share is at most a pre-registered width.

THE PRE-REGISTRATION (#417 K-Q1, measured on a grid of caps 100, 300, 500 and 1000 rounds per
arrangement, draw rates 0 to 0.25 and hero shares 0.5 + d on T and 0.5 - d on CT for d up to
0.18; lane E's and lane J's duel cells have draw rates up to 0.2433 and T/CT share gaps up to
0.1704)
  * looks: LOOKS = 5. At look k every arrangement has played ``look_rounds(cap)[k - 1]`` =
    ceil(k cap / 5) rounds, so each look sees EQUAL rounds of the T and the CT arrangement.
  * test: the exact two-sided sign test of the hero's wins against its losses at 0.5
    (``sign_p``, scipy.stats.binomtest's p-value). A double kill or a time-out is a draw: it
    costs a round and is not counted.
  * boundary: O'Brien-Fleming for 5 looks at two-sided 0.05, the published constant
    OBF_C = 2.040 (the K = 5 entry of the table J.-p. Liu's 2006 lecture notes reproduce from
    Jennison & Turnbull; 2.0401 when computed, #417). Look k rejects
    when the p-value is at most LOOK_ALPHAS[k - 1] = 2 (1 - Phi(2.040 sqrt(5 / k))): 5.1e-6,
    0.00126, 0.0084, 0.0226, 0.0414.
  * outcome: "hero" or "opponent" (the sign of wins - losses) at the first rejecting look, else
    UNDECIDED ("undecided at cap") with the fixed-n 95% Wilson interval of the full cell.
  * false-decision rate when the hero's share is 0.5: at most 0.04818 at those caps by an exact
    lattice computation; a seeded Monte Carlo of 400000 sequences a point put its 95% upper end
    at most 0.04898. With no draws every cap from 5 to 3000 stays under 0.05 (worst 0.04997, cap
    750, two arrangements; #417 review); the boundary's Gaussian limit is 0.050009, so the bound
    rests on the exact test's lattice.

PITFALLS
--------
* A cell stopped early has a biased share: it stopped because the share looked extreme. Shares
  of sign-stopped cells are not comparable across cells and have no valid fixed-n interval
  (``SignStop.wilson`` is None for them). A ladder uses ``precision_stop``.
* Never stop on "the fixed-n 95% interval excludes 0.5" while looking repeatedly: checked after
  every round from the 10th decided one, that rule decided 0.4560 of 1000-round null cells
  (#407), and 1.96 at each of 5 looks has a Gaussian crossing probability of 0.14169.
* Every look must see equal rounds of every arrangement. run_duel (main a01308a) plays all of
  the T arrangement before the CT one, so a look taken from its row order would see one side.
* The z-scale form of the same boundary, |wins - losses| / sqrt(wins + losses) >= 2.040
  sqrt(5 / k), exceeds 0.05 on the binomial lattice (0.05504 exact at cap 300 with two
  arrangements and no draws). The exact test is the pre-registered rule.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

from scipy import optimize, special, stats

LOOKS = 5
ALPHA = 0.05
OBF_C = 2.040
LOOK_ALPHAS = tuple(2 * float(stats.norm.sf(OBF_C * math.sqrt(LOOKS / k)))
                    for k in range(1, LOOKS + 1))
UNDECIDED = "undecided at cap"
RUNNING = "running"
DRAWS = ("double", "timeout")
# The ladder's pre-registered width (#407's MIXW) and its rounds per arrangement between looks.
WIDTH = 0.10
LADDER_BLOCK = 10


def look_rounds(cap: int, looks: int = LOOKS) -> tuple[int, ...]:
    """The rounds each arrangement has played at each look: ceil(k cap / looks), the last = cap."""
    if cap < looks:
        raise ValueError(f"cap must be at least {looks} rounds per arrangement, got {cap}")
    return tuple(math.ceil(k * cap / looks) for k in range(1, looks + 1))


def _check(arrangements: Sequence[Sequence[str]], cap: int) -> None:
    if not arrangements:
        raise ValueError("no arrangement")
    for outcomes in arrangements:
        if len(outcomes) > cap:
            raise ValueError(f"an arrangement has {len(outcomes)} rounds, past the cap {cap}")


def _counts(arrangements: Sequence[Sequence[str]], rounds: int) -> tuple[int, int, int]:
    """(wins, losses, draws) over the first ``rounds`` rounds of EVERY arrangement."""
    w = lo = d = 0
    for outcomes in arrangements:
        head = list(outcomes[:rounds])
        w, lo = w + head.count("win"), lo + head.count("loss")
        d += sum(head.count(x) for x in DRAWS)
    if w + lo + d != rounds * len(arrangements):
        raise ValueError(f"an outcome is not one of win, loss, {', '.join(DRAWS)}")
    return w, lo, d


def sign_p(wins: int, losses: int) -> float:
    """The exact two-sided sign-test p-value of ``wins`` against ``losses`` at 0.5: what
    scipy.stats.binomtest(wins, wins + losses).pvalue returns, as 2 P(X <= min) of the symmetric
    binomial through scipy.special.bdtr. The two agreed to 1.05e-12 on 7775 pairs, and took
    17.6 us against 9.6 ms a call under a load average of 22 (#417). 1.0 with no decided round."""
    n = wins + losses
    return min(1.0, 2 * float(special.bdtr(min(wins, losses), n, 0.5))) if n else 1.0


@dataclasses.dataclass(frozen=True)
class Look:
    """One look: its number (1-based), the rounds per arrangement, the pooled counts, the test."""
    look: int
    rounds: int
    wins: int
    losses: int
    draws: int
    p_value: float
    alpha: float


@dataclasses.dataclass(frozen=True)
class SignStop:
    """``decision``: "hero", "opponent", UNDECIDED, or RUNNING (the next look is not reached yet).
    ``looks``: every look taken."""
    decision: str
    looks: tuple[Look, ...]

    @property
    def wilson(self) -> tuple[float, float] | None:
        """The full cell's fixed-n 95% Wilson interval for UNDECIDED, else None; (nan, nan) when
        no round was decided. A property, not a field: binomtest's proportion_ci took 7.1 ms a
        call (#417), which a Monte Carlo over thousands of cells should not pay. PITFALL:
        dataclasses.asdict omits it."""
        if self.decision != UNDECIDED:
            return None
        w, lo = self.looks[-1].wins, self.looks[-1].losses
        if w + lo == 0:
            return (math.nan, math.nan)
        ci = stats.binomtest(w, w + lo).proportion_ci(confidence_level=1 - ALPHA, method="wilson")
        return (float(ci.low), float(ci.high))


def sign_stop(arrangements: Sequence[Sequence[str]], cap: int) -> SignStop:
    """The pre-registered sign test (module docstring) over ``arrangements``: one outcome list
    per arrangement (oracle_duel's row["outcome"] values, in the order played), each at most
    ``cap`` long. Takes, in order, every look that all arrangements have reached and stops at the
    first rejection; RUNNING when the next look is not reached yet."""
    _check(arrangements, cap)
    taken: list[Look] = []
    for k, rounds in enumerate(look_rounds(cap), start=1):
        if any(len(o) < rounds for o in arrangements):
            return SignStop(RUNNING, tuple(taken))
        w, lo, d = _counts(arrangements, rounds)
        p = sign_p(w, lo)
        taken.append(Look(k, rounds, w, lo, d, p, LOOK_ALPHAS[k - 1]))
        if p <= LOOK_ALPHAS[k - 1]:
            return SignStop("hero" if w > lo else "opponent", tuple(taken))
    return SignStop(UNDECIDED, tuple(taken))


def _lower_end(wins: int, losses: int, thr: float) -> float:
    """The lower end of ``mixture_interval``: the root of log M(p) = thr below wins / n."""
    if wins == 0:
        return 0.0                     # log M(0) = betaln(1, 1 + losses) <= 0 < thr
    base = float(special.betaln(1 + wins, 1 + losses))

    def excess(p):
        return base - special.xlogy(wins, p) - special.xlog1py(losses, -p) - thr

    # excess(1e-300) >= wins (689.8 + log(wins / n)) - log(n + 1) - thr > 0 for n below 1e100,
    # and excess(wins / n) < 0: one sign change.
    return float(
        optimize.root_scalar(excess, bracket=(1e-300, wins / (wins + losses)),
                             method="brentq").root)


def mixture_interval(wins: int, losses: int, alpha: float = ALPHA) -> tuple[float, float]:
    """The anytime-valid 1 - alpha interval of the win share: the p whose Beta(1, 1) mixture
    likelihood ratio M(p) = B(1 + wins, 1 + losses) / (p^wins (1 - p)^losses) is below 1 / alpha.
    By Ville's inequality the chance it ever excludes the true share, at any number of looks, is
    at most alpha. log M is convex with its minimum, at most 0, at wins / n, so each end is one
    root; the upper end is the lower end of the mirrored counts. (0, 1) before any decided round."""
    thr = math.log(1 / alpha)
    return (_lower_end(wins, losses, thr), 1.0 - _lower_end(losses, wins, thr))


@dataclasses.dataclass(frozen=True)
class PrecisionStop:
    """``status``: "precise" (the interval is at most the width), "at cap", or RUNNING.
    ``rounds``: per arrangement at the last look; ``interval``: ``mixture_interval`` there."""
    status: str
    rounds: int
    wins: int
    losses: int
    draws: int
    interval: tuple[float, float]


def precision_stop(arrangements: Sequence[Sequence[str]],
                   cap: int,
                   width: float = WIDTH,
                   block: int = LADDER_BLOCK) -> PrecisionStop:
    """The ladder's stop: a look every ``block`` rounds per arrangement and at ``cap``, equal
    rounds of every arrangement at each, stopping once ``mixture_interval`` is at most ``width``
    wide. Compare stopped cells by their intervals: an anytime-valid interval keeps its coverage
    at whatever round the stop chose, a fixed-n interval does not."""
    _check(arrangements, cap)
    rounds, w, lo, d, ci = 0, 0, 0, 0, (0.0, 1.0)
    for r in sorted({*range(block, cap, block), cap}):
        if any(len(o) < r for o in arrangements):
            return PrecisionStop(RUNNING, rounds, w, lo, d, ci)
        rounds, (w, lo, d) = r, _counts(arrangements, r)
        ci = mixture_interval(w, lo)
        if ci[1] - ci[0] <= width:
            return PrecisionStop("precise", rounds, w, lo, d, ci)
    return PrecisionStop("at cap", rounds, w, lo, d, ci)
