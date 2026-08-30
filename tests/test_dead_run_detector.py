"""DeadRunDetector alert accounting — gh#93 false-positive regression tests.

Background: the zero-kills rule appended a fresh alert on every check past
500k steps, so one persistent condition reached the 5-alert abort verdict by
itself. Kills are structurally 0 in the current pre-combat phase, which made the
default-on abort a guaranteed false kill for anyone who forgot
--no-dead-run-abort. These tests pin both guards (weight gate + single
non-accumulating alert) and, importantly, pin that the rule still WORKS as a
contributor so the fix cannot decay into "zero kills is never reported".

No env is built here: check() is pure metric bookkeeping.
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train                           # noqa: E402

_STEP = 600_000                        # past the rule's 500k arming threshold


def _healthy_metrics(**over):
    """Metrics that trip no rule; callers override the one field under test."""
    m = {
        "game/kills_per_episode": 1.0,
        "game/timeout_rate": 0.0,
        "losses/entropy": 5.0,
        "losses/approx_kl": 0.0,
    }
    m.update(over)
    return m


def test_zero_kills_alert_does_not_accumulate():
    d = train.DeadRunDetector()
    metrics = _healthy_metrics(**{"game/kills_per_episode": 0})
    for i in range(20):
        assert d.check(
            _STEP + i * 10_000,
            metrics) is False, ("zero kills alone must never reach the dead-run verdict (gh#93)")
    assert len(d.alerts) == 1, f"expected a single live zero-kills alert, got {d.alerts}"


def test_zero_kills_alert_is_still_raised_and_still_counts():
    """The guard bounds the alert; it must not delete it.

    R0-J: timeout and KL are single-live alerts too, so the verdict here is
    reached as 1 zero-kills + 1 KL + 1 timeout + 3 accumulating entropy
    alerts (entropy is the only rule that still accumulates on purpose).
    """
    d = train.DeadRunDetector()
    d.check(_STEP, _healthy_metrics(**{"game/kills_per_episode": 0}))
    assert any("Zero kills" in a for a in d.alerts)
    bad = _healthy_metrics(
        **{
            "game/kills_per_episode": 0,
            "losses/approx_kl": 0.9,
            "game/timeout_rate": 1.0,
            "losses/entropy": 0.1
        })
    verdicts = [d.check(_STEP + (i + 1) * 10_000, bad) for i in range(3)]
    assert verdicts[-1] is True, f"verdict never fired; alerts={d.alerts}"
    assert sum("Zero kills" in a for a in d.alerts) == 1
    assert sum("KL" in a for a in d.alerts) == 1
    assert sum("Timeout" in a for a in d.alerts) == 1
    assert sum("Entropy" in a for a in d.alerts) == 3


def test_zero_kills_rule_disabled_when_kill_reward_is_off():
    d = train.DeadRunDetector(kills_expected=False)
    for i in range(20):
        d.check(_STEP + i * 10_000, _healthy_metrics(**{"game/kills_per_episode": 0}))
    assert d.alerts == [], f"kill reward is off — no zero-kills alert expected, got {d.alerts}"


def test_recovering_kills_clears_the_alert():
    d = train.DeadRunDetector()
    d.check(_STEP, _healthy_metrics(**{"game/kills_per_episode": 0}))
    assert d.alerts
    d.check(_STEP + 10_000, _healthy_metrics(**{"game/kills_per_episode": 2.0}))
    assert d.alerts == [], "kills recovered — the stale alert must be cleared"


def test_kill_reward_probe_defaults_to_armed_on_unreadable_env():
    """An env we cannot introspect must leave the safety check ON, not off."""

    class _Opaque:
        pass

    assert train._kill_reward_is_active(_Opaque()) is True
    assert train._kill_reward_is_active(None) is True


def test_kill_reward_probe_reads_the_static_data_weight():
    """Mirrors the real ctypes shape: vecenv.driver_env._c_env.sd.contents.reward_kill."""

    class _Ptr:

        def __init__(self, w):
            self.contents = type("_SD", (), {"reward_kill": w})()

    class _CEnv:

        def __init__(self, w):
            self.sd = _Ptr(w)

    class _Env:

        def __init__(self, w):
            self._c_env = _CEnv(w)

    class _Vec:

        def __init__(self, w):
            self.driver_env = _Env(w)

    assert train._kill_reward_is_active(_Vec(0.3)) is True
    assert train._kill_reward_is_active(_Vec(0.0)) is False
    # bare env (no vecenv wrapper) is the test-harness shape
    assert train._kill_reward_is_active(_Env(0.0)) is False


# ── R0-J (Task 14): key prefixes + non-accumulating timeout / KL rules ────────
# pufferl.py's outer log dict prefixes every loss stat with `losses/`, so the
# old `entropy/total` / `approx_kl` lookups never matched and those rules were
# dead since day one. Timeout and KL are now single-live alerts (like zero
# kills, gh#93): at Rung 1 every no-kill round is a timeout, so an untrained
# policy sits at timeout_rate≈1.0 and would otherwise abort itself in 5 checks.


def _m(**kw):
    base = {
        "game/kills_per_episode": 1.0,
        "game/timeout_rate": 0.0,
        "losses/entropy": 5.0,
        "losses/approx_kl": 0.0,
    }
    base.update(kw)
    return base


def test_timeout_alert_does_not_accumulate():
    d = train.DeadRunDetector()
    for step in (60_000, 70_000, 80_000, 90_000, 100_000, 110_000):
        assert not d.check(step, _m(**{"game/timeout_rate": 1.0}))
    assert sum("Timeout" in a for a in d.alerts) == 1


def test_timeout_alert_clears_when_rate_recovers():
    d = train.DeadRunDetector()
    d.check(60_000, _m(**{"game/timeout_rate": 1.0}))
    assert any("Timeout" in a for a in d.alerts)
    d.check(70_000, _m(**{"game/timeout_rate": 0.5}))
    assert not any("Timeout" in a for a in d.alerts), d.alerts


def test_kl_rule_reads_losses_prefix_and_does_not_accumulate():
    d = train.DeadRunDetector()
    for step in (110_000, 120_000, 130_000):
        d.check(step, _m(**{"losses/approx_kl": 0.5}))
    assert sum("KL" in a for a in d.alerts) == 1


def test_kl_alert_clears_when_kl_recovers():
    d = train.DeadRunDetector()
    d.check(110_000, _m(**{"losses/approx_kl": 0.5}))
    assert any("KL" in a for a in d.alerts)
    d.check(120_000, _m(**{"losses/approx_kl": 0.01}))
    assert not any("KL" in a for a in d.alerts), d.alerts


def test_entropy_rule_reads_losses_prefix_and_accumulates():
    d = train.DeadRunDetector()
    for step in (60_000, 70_000, 80_000):
        d.check(step, _m(**{"losses/entropy": 0.1}))
    assert sum("Entropy" in a for a in d.alerts) == 3


def test_abort_reachable_with_timeout_kl_and_entropy():
    d = train.DeadRunDetector()
    tripped = False
    for step in (110_000, 120_000, 130_000, 140_000):
        tripped = d.check(
            step, _m(**{
                "game/timeout_rate": 1.0,
                "losses/approx_kl": 0.5,
                "losses/entropy": 0.1
            }))
    assert tripped                     # 1 timeout + 1 KL + ≥3 entropy ≥ 5
