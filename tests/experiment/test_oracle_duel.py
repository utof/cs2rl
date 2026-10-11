"""#152 L2: two duel bots in the arena (cs2rl.experiment.oracle_duel).

WHAT IS PINNED HERE (seed 0, obs-only bots; the values observed 2026-10-11)
---------------------------------------------------------------------------
* The default mirror (hard, style=stop) passes the L2 verdict at 100 rounds a side: per
  side shots sum to the env totals, no time-out, the T share's and the hero share's
  95% Wilson intervals contain 0.5 (observed hero share 0.5600 [0.4859, 0.6315]).
* Obs-only = ground truth, round for round (outcome, length, each side's shots), on the
  default mirror and on stop against run, 40 rounds a side. KNOWN LIMIT: a ray grazing a
  hitbox can flip on float32 rounding (the obs normalises positions); lane E found two such
  first divergences in 3000 rounds over five cells (600 a cell), the earlier at round 79 of
  its arrangement. If this goes red after a sim change, find the first step whose state
  differs before suspecting the decoding.
* A perfect mirror is all double kills, and the verdict fails it as "no decided round".
  That pins the simultaneous combat (cs2_combat.h alive_pre, cs2_env.h WINNER_DRAW) and
  duel_bot.LANDS_TOL together: without either, one side wins rounds.
* Floors: under the cone and the hitboxes the hard stop bot beats the hard runner (its
  share's interval above 0.5 at 150 rounds a side; observed 0.6220 [0.5650, 0.6758]);
  without the cone it does not (observed 0.4437 [0.3870, 0.5018]). The same holds for the
  hard ADAD bot, which brakes only while engaged with its weapon ready (its own brake term):
  observed 0.5860 [0.5280, 0.6416] with the cone, 0.4725 [0.4141, 0.5317] without. Lead:
  the perfect planted bot's hit/fired on the L1 walker is above 0.6 led (observed 0.7665)
  and below 0.35 unled (observed 0.2176), at 50 rounds a side.
* The verdict's rules, parse_bot's spec grammar, and main's exit code and switches.
"""

from typing import Any

import pytest

from cs2rl.experiment import oracle_duel as D

KEYS = ("outcome", "ticks", "hero_fired", "hero_hit", "hero_fast", "opp_fired", "opp_hit",
        "opp_fast")


def _duel(hero, opp, rounds, obs_only=True, accuracy=None, hitboxes=None):
    return D.run_duel(D.parse_bot(hero), D.parse_bot(opp), rounds, 0, obs_only, accuracy, hitboxes)


def _checks(res, expect):
    ok, checks = D.verdict(res, expect)
    return ok, {name: (good, detail) for name, good, detail in checks}


def test_the_default_mirror_passes_the_l2_verdict():
    res = _duel(D.DEFAULT_BOT, D.DEFAULT_BOT, 100)
    ok, checks = _checks(res, "even")
    assert ok, checks
    for a in res["accounting"]:
        assert all(a[k][0] + a[k][1] == a[k][2] and a[k][2] > 0 for k in ("fired", "hit")), a
    assert D.summary(res["rows"])["timeout"] == 0


@pytest.mark.parametrize("opp", [D.DEFAULT_BOT, "hard,style=run"])
def test_obs_only_bots_equal_ground_truth_bots_round_for_round(opp):
    obs, gt = _duel(D.DEFAULT_BOT, opp, 40), _duel(D.DEFAULT_BOT, opp, 40, obs_only=False)
    diff = [(a["seed"], a["episode"]) for a, b in zip(obs["rows"], gt["rows"], strict=True)
            if [a[k] for k in KEYS] != [b[k] for k in KEYS]]
    assert not diff, f"rounds that differ (seed, episode): {diff}"


def test_a_perfect_mirror_is_all_double_kills_and_fails_the_verdict():
    res = _duel("perfect", "perfect", 40)
    assert {r["outcome"] for r in res["rows"]} == {"double"}
    ok, checks = _checks(res, "even")
    assert not ok
    assert checks["T share 95% CI contains 0.5"] == (False, "no decided round: every round a draw")


@pytest.mark.parametrize("hero", [D.DEFAULT_BOT, "hard,style=adad"])
@pytest.mark.parametrize("cone", [True, False])
def test_stopping_beats_running_only_under_the_cone(hero, cone):
    s = D.summary(_duel(hero, "hard,style=run", 150, accuracy=cone, hitboxes=True)["rows"])
    lo, hi = s["hero_ci"]
    assert (lo > 0.5) == cone, (hero, cone, s["hero_share"], s["hero_ci"])


@pytest.mark.parametrize("lead", [1, 0])
def test_lead_is_what_hits_the_walker_under_hitboxes(lead):
    s = D.summary(
        _duel(f"perfect,style=plant,lead={lead}", "walker", 50, accuracy=True,
              hitboxes=True)["rows"])
    rate = s["hero_hit"] / s["hero_fired"]
    assert s["win"] == s["rounds"] == 100
    assert (rate > 0.6) if lead else (rate < 0.35), (lead, rate)


def _rows(win=0, loss=0, double=0, timeout=0, t_won=None) -> dict[str, Any]:
    """Synthetic rows for the verdict: ``t_won`` of the decided rounds are T wins."""
    rows: list[dict[str, Any]] = []
    for outcome, n in (("win", win), ("loss", loss), ("double", double), ("timeout", timeout)):
        rows += [{
            "outcome": outcome,
            "t_won": False,
            "ct_won": False,
            "kill_tick": None,
            "death_tick": None,
            **{
                f"{w}_{k}": 0
                for w in ("hero", "opp")
                for k in ("fired", "hit", "fast")
            }
        } for _ in range(n)]
    decided = [r for r in rows if r["outcome"] in ("win", "loss")]
    for i, r in enumerate(decided):
        r["t_won"] = i < (len(decided) // 2 if t_won is None else t_won)
        r["ct_won"] = not r["t_won"]
    acct = {k: (1, 1, 2) for k in ("fired", "hit", "fast")}
    return {"rows": rows, "accounting": [{**acct, "unmatched": 0}]}


@pytest.mark.parametrize("res, expect, failing", [
    (_rows(win=50, loss=50), "even", []),
    (_rows(win=50, loss=50, timeout=6), "even", ["timeouts <= 0.05 of rounds"]),
    (_rows(win=50, loss=50, t_won=80), "even", ["T share 95% CI contains 0.5"]),
    (_rows(win=80, loss=20), "even", ["hero share 95% CI: even"]),
    (_rows(win=80, loss=20), "hero", []),
    (_rows(win=20, loss=80), "opponent", []),
    (_rows(win=50, loss=50), "hero", ["hero share 95% CI: hero"]),
    (_rows(win=80, loss=20), "any", []),
])
def test_the_verdict_rules(res, expect, failing):
    ok, checks = _checks(res, expect)
    assert [name for name, (good, _) in checks.items() if not good] == failing
    assert ok == (not failing)


def test_the_verdict_fails_a_side_whose_shots_do_not_sum():
    res = _rows(win=50, loss=50)
    res["accounting"][0]["hit"] = (1, 0, 2)
    assert _checks(res, "even")[1]["per-side shots sum to the env totals"][0] is False
    res = _rows(win=50, loss=50)
    res["accounting"][0]["unmatched"] = 1
    assert _checks(res, "even")[1]["unmatched vis slots == 0"][0] is False


def test_parse_bot():
    from cs2rl.eval.duel_bot import PROFILES, DuelBotParams
    assert D.parse_bot("perfect") == DuelBotParams()
    assert D.parse_bot("hard") == PROFILES["hard"]
    p = D.parse_bot(
        "expert,style=adad,lead=0,reaction_ticks=2,focus_decay=0.5,reaction_mode=percept")
    assert isinstance(p, DuelBotParams)
    assert (p.style, p.lead, p.reaction_ticks, p.focus_decay, p.reaction_mode) == ("adad", False, 2,
                                                                                   0.5, "percept")
    assert p.focus_initial_deg == PROFILES["expert"].focus_initial_deg
    assert D.parse_bot("walker") == D.WALKER
    for bad in ("godlike", "hard,aim=1", "walker,lead=0", "hard,style=crouch",
                "hard,reaction_ticks=x"):
        with pytest.raises(ValueError):
            D.parse_bot(bad)


def test_main_exits_on_the_verdict_and_passes_the_switches(monkeypatch, capsys):
    seen = []

    def fake(hero, opp, rounds, seed, obs_only, accuracy, hitboxes):
        seen.append((rounds, seed, obs_only, accuracy, hitboxes))
        res = _rows(win=50, loss=50) if hero == opp else _rows(double=10)
        for k, r in enumerate(res["rows"]):
            r.update(seed=seed,
                     episode=k,
                     t_xy=(0.0, 0.0),
                     ct_xy=(0.0, 0.0),
                     hero_side="T",
                     ticks=1)
        return {
            **res, "rounds": rounds,
            "seed": seed,
            "obs_only": obs_only,
            "accuracy": True,
            "hitboxes": True
        }

    monkeypatch.setattr(D, "run_duel", fake)
    assert D.main(["--rounds", "7"]) == 0 and "PASS" in capsys.readouterr().out
    assert D.main([
        "--opponent", "hard", "--accuracy", "0", "--hitboxes", "1", "--ground-truth", "--expect",
        "even"
    ]) == 1
    assert "no decided round" in capsys.readouterr().out
    assert seen == [(7, 0, True, None, None), (500, 0, False, False, True)]
    with pytest.raises(SystemExit):
        D.main(["--hero", "walker"])
