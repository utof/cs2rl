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
        "entropy/total": 5.0,
        "approx_kl": 0.0,
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
    """The guard bounds the alert; it must not delete it."""
    d = train.DeadRunDetector()
    d.check(_STEP, _healthy_metrics(**{"game/kills_per_episode": 0}))
    assert any("Zero kills" in a for a in d.alerts)
    # four genuine alerts from other rules + the zero-kills one = verdict
    bad = _healthy_metrics(**{"game/kills_per_episode": 0, "approx_kl": 0.9})
    verdicts = [d.check(_STEP + (i + 1) * 10_000, bad) for i in range(4)]
    assert verdicts[-1] is True, f"verdict never fired; alerts={d.alerts}"


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
