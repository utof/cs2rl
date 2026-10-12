"""#417 phase 2: the sequential stop in oracle_duel.run_duel, its verdict, report and JSON, and
in duel_eval.

WHAT IS PINNED HERE (seed 0 unless named; the values observed 2026-10-12)
--------------------------------------------------------------------------
* An arrangement kept open (``oracle_duel._Arrangement``) and played in alternating T and CT
  blocks plays the rows and the accounting of the fixed-n run; rebuilding its env per block
  would replay the first block's spawns.
* A sign-stopped cell's rows are the first rows of the fixed-n cell: the hard stop bot against
  the walker, cap 20, is decided "hero" at look 2 of 5, 8 rounds a side.
* A policy hero's two arrangements, interleaved in blocks, play duel_eval's one-side cells
  (fork_rng and manual_seed(seed + k * CT_SEED_OFFSET)), in the sampled mode where the torch
  stream decides actions: each arrangement keeps its own torch RNG state.
* The sign test reads one outcome list per arrangement, at looks of equal rounds, a T block
  then a CT block: a hero that wins every T round and loses every CT round is undecided at cap.
* The precision stop plays LADDER_BLOCK rounds per arrangement between looks and returns what
  sequential.precision_stop returns on the rounds played.
* The verdict reads the stop: ``--expect hero`` fails an undecided stop even when the Wilson
  interval of the rows lies above 0.5, and passes a "hero" decision even when it does not. A
  cell decided at a look prints no Wilson interval of the hero's share, in oracle_duel's report
  and in duel_eval's table, and an undecided cell prints the full cell's.
* The stop reads back from JSON with its fixed-n "wilson" (dataclasses.asdict would drop it).
* duel_eval passes the stop to run_duel and each cell carries it.
"""
import contextlib
import dataclasses
import json
from typing import Any

import pytest

from cs2rl.experiment import oracle_duel as D
from cs2rl.experiment import sequential as S
from tests.experiment.test_oracle_duel import _rows


def _open(stack, hero, opp, seed):
    return [
        stack.enter_context(contextlib.closing(D._Arrangement(hero, opp, k, seed))) for k in (0, 1)
    ]


def test_an_arrangement_played_in_blocks_plays_the_fixed_n_rows():
    hero, opp = D.parse_bot("hard,style=stop"), D.parse_bot("hard,style=run")
    ref = D.run_duel(hero, opp, 8, 0)
    with contextlib.ExitStack() as stack:
        arrs = _open(stack, hero, opp, 0)
        for block in (3, 3, 2):
            for arr in arrs:
                arr.play(block)
    assert [r for arr in arrs for r in arr.rows] == ref["rows"]
    assert [arr.accounting() for arr in arrs] == ref["accounting"]


def test_a_sign_stopped_cell_is_the_first_rows_of_the_fixed_n_cell():
    hero = D.parse_bot("hard,style=stop")
    res = D.run_duel(hero, D.WALKER, 20, 0, stop="sign")
    st = res["stop"]
    assert (st["rule"], st["cap"], st["decision"], st["wilson"]) == ("sign", 20, "hero", None)
    assert [look["rounds"] for look in st["looks"]] == [4, 8]
    ref = D.run_duel(hero, D.WALKER, 8, 0)
    assert res["rows"] == ref["rows"] and res["accounting"] == ref["accounting"]


def test_interleaved_policy_arrangements_play_duel_evals_one_side_cells():
    import torch

    from cs2rl.experiment import duel_eval as E
    from tests.experiment.test_duel_eval import _random_policy
    hero = E.policy_hero(_random_policy(), "sample", "carried", 64)
    opp = D.parse_bot("expert,style=stop")
    cells = []
    for k in (0, 1):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3 + k * D.CT_SEED_OFFSET)
            cells.append(D.run_duel(hero, opp, 4, 3, sides=(k, ))["rows"])
    with contextlib.ExitStack() as stack:
        arrs = _open(stack, hero, opp, 3)
        for _ in range(2):
            for arr in arrs:
                arr.play(2)
    assert [arr.rows for arr in arrs] == cells


def _planted(log: list, t: str, ct: str):
    """An ``_Arrangement`` stand-in whose hero's every round is ``t`` on T and ``ct`` on CT;
    ``log`` records each block as (side, rounds)."""

    class Planted:

        def __init__(self, hero, opponent, hero_side, *seed_obs_only_accuracy_hitboxes):
            self.side, self.rows, self.flags = hero_side, [], (True, True)

        def play(self, k):
            log.append((self.side, k))
            self.rows += [{
                "hero_side": D.SIDES[self.side],
                "outcome": ct if self.side else t
            } for _ in range(k)]

        def accounting(self):
            return {"unmatched": 0}

        def close(self):
            pass

    return Planted


def _planted_run(monkeypatch, t, ct, cap=20, stop="sign"):
    log: list = []
    monkeypatch.setattr(D, "_Arrangement", _planted(log, t, ct))
    return D.run_duel(None, None, cap, 0, stop=stop), log


def test_the_sign_test_reads_one_outcome_list_per_arrangement(monkeypatch):
    res, log = _planted_run(monkeypatch, "win", "loss")
    st = res["stop"]
    assert st["decision"] == S.UNDECIDED
    assert [(look["rounds"], look["wins"], look["losses"])
            for look in st["looks"]] == [(r, r, r) for r in S.look_rounds(20)]
    assert log == [(0, 4), (1, 4)] * 5


def test_the_precision_stop_looks_every_ladder_block(monkeypatch):
    res, log = _planted_run(monkeypatch, "win", "win", cap=100, stop="precision")
    want = S.precision_stop([["win"] * 40] * 2, 100)
    assert (want.status, want.rounds) == ("precise", 40)
    assert res["stop"] == {
        "rule": "precision",
        "cap": 100,
        **dataclasses.asdict(want), "width": S.WIDTH
    }
    assert log == [(0, S.LADDER_BLOCK), (1, S.LADDER_BLOCK)] * 4


def test_an_unknown_stop_rule_is_refused():
    with pytest.raises(ValueError, match="unknown stop rule"):
        D.run_duel(None, None, 20, 0, stop="peek")


def _report_res(rows: dict[str, Any], stop: dict) -> dict[str, Any]:
    """``rows`` (test_oracle_duel's synthetic rows) as a run_duel result with ``stop``."""
    for k, r in enumerate(rows["rows"]):
        r.update(seed=0, episode=k, t_xy=(0.0, 0.0), ct_xy=(0.0, 0.0), hero_side="T", ticks=1)
    return {
        **rows, "rounds": 50,
        "seed": 0,
        "obs_only": True,
        "accuracy": True,
        "hitboxes": True,
        "stop": stop
    }


def test_the_verdict_and_the_report_read_the_stop(monkeypatch):
    undecided = _planted_run(monkeypatch, "win", "loss")[0]["stop"]
    decided = _planted_run(monkeypatch, "win", "win")[0]["stop"]
    assert (undecided["decision"], decided["decision"]) == (S.UNDECIDED, "hero")
    name = "hero share sign stop: hero"
    above = _report_res(_rows(win=80, loss=20), undecided)             # Wilson above 0.5
    ok, checks = D.verdict(above, "hero")
    assert not ok and [n for n, good, _ in checks if not good] == [name]
    assert D.verdict(above, "even")[0]
                                                                       # Hero share 0.6, T share 0.5: the report's only interval of 0.6 would be the hero's.
    even = _report_res(_rows(win=60, loss=40, t_won=50), decided)
    assert D.verdict(even, "hero")[0]
    lo, hi = D.summary(even["rows"])["hero_ci"]
    report = D.format_report(even, "a", "b", "hero")
    assert "stop (sign, #417): hero at look" in report
    assert f"{lo:.4f}" not in report and f"{hi:.4f}" not in report
    assert "not sequentially controlled" in report and "(cap 50 per side," in report


def test_the_stop_reads_back_from_json_with_its_wilson_interval(monkeypatch):
    res = json.loads(json.dumps(_planted_run(monkeypatch, "win", "loss")[0]))
    st = res["stop"]
    assert set(st) == {"rule", "cap", "decision", "looks", "wilson"}
    want = S.sign_stop([["win"] * 20, ["loss"] * 20], 20)
    assert want.wilson is not None and st["wilson"] == pytest.approx(list(want.wilson))
    assert st["looks"] == [dataclasses.asdict(look) for look in want.looks]
    assert json.loads(json.dumps(_planted_run(monkeypatch, "win",
                                              "win")[0]))["stop"]["wilson"] is None


def test_duel_eval_passes_the_stop_and_its_table_shows_it(monkeypatch):
    from cs2rl.experiment import duel_eval as E
    from tests.experiment.test_duel_eval import _res
    undecided = _planted_run(monkeypatch, "win", "loss")[0]["stop"]
    decided = _planted_run(monkeypatch, "win", "win")[0]["stop"]
    seen = []

    def fake(hero, opp, rounds, seed, obs_only, accuracy, hitboxes, sides, stop):
        seen.append(stop)
        return {**_res(), "stop": decided if hero == "policy" else undecided}

    monkeypatch.setattr(E, "policy_hero", lambda *a: "policy")
    monkeypatch.setattr(D, "run_duel", fake)
    out = E.run_eval(None, 64, rounds=20, opponents=("easy,style=stop", ), stop="sign")
    assert seen == ["sign"] * 3 and out["stop"] == "sign"
    assert [c["stop"]["decision"] for c in out["cells"]] == ["hero", S.UNDECIDED, S.UNDECIDED]
    table = E.format_table(out).splitlines()
    policy, perfect = table[3], table[4]
    lo, hi = out["cells"][0]["hero_ci"]
    assert policy.endswith("look 2/5 (8)") and f"{out['cells'][0]['hero_share']:.4f}*" in policy
    assert f"{lo:.4f}" not in policy and f"{hi:.4f}" not in policy
    assert perfect.endswith("cap (20)") and f"[{lo:.4f}, {hi:.4f}]" in perfect
    assert "* stopped at a look: the share is biased and has no fixed-n interval (#417)" in table
    assert table[0].startswith("duel_eval: up to 20 rounds per cell (--stop sign, #417)")
