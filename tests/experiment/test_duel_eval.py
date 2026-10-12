"""cs2rl.experiment.duel_eval: a policy checkpoint against the #152 L2 duel bots (Rung 1c).

WHAT IS PINNED HERE
-------------------
* The opponents are the Rung 1c zero-shot set, and the references are the perfect bot and
  the opponent's own spec.
* ``cell_summary`` on hand-set rounds: the first-shot hit rate and its Wilson interval, time
  to damage (median of the first hits under 16 ticks, and the share at or over 16, which Leetify
  excludes), lifetime (the death tick, else the round's last; every death in these rows ends its
  round, as in the arena, so only the round's length is pinned), shots per kill, hit per fired of
  the fast and slow shots, the moving share of idle ticks, and the accounting check, which a cell
  whose shots do not sum, or with an unmatched visibility slot, fails and the table flags.
* ``run_eval`` plays each cell on its side (seed, then seed + 10000 on CT), and a reference
  cell's rounds are oracle_duel.run_duel's rounds for the same bots, seed and side.
* The policy plays as aim_eval plays it: against the same duel bot, aim_eval's cell and
  duel_eval's cell play the same rounds (kill tick, shots, hits, spawns), in the sampled mode
  too, where the torch stream matters; and cells reproduce in any order.
* The CLI plays the cone and the hit model the run trained with unless told otherwise, warns
  on a difference and exits 0; it passes --stop (fixed by default) to run_eval.
* The CLI on the Rung 1b-2 V checkpoint, 2 rounds per cell. It needs the checkpoint, which
  only the owner's checkouts have (outputs/ is not in git), so it skips without one.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cs2rl.experiment import duel_eval as E
from cs2rl.experiment import oracle_duel as D
from tests.conftest import REPO_ROOT

V_CKPT = "outputs/modal/r1b2-v-a01308a-s0/checkpoints/checkpoints-20261011-180449.pt"


def test_the_opponents_are_the_rung1c_zero_shot_set():
    assert E.OPPONENTS == ("easy,style=stop", "normal,style=stop", "hard,style=stop",
                           "hard,style=run", "hard,style=adad", "expert,style=stop")
    assert all(not isinstance(D.parse_bot(s), str) for s in E.OPPONENTS)
    assert E.REFERENCE_HEROES == ("perfect", "mirror")


def _row(outcome, ticks, kill=None, death=None, **side):
    """One round row as run_duel writes it; ``side`` sets hero_* / opp_* fields."""
    r = {
        "outcome": outcome,
        "ticks": ticks,
        "kill_tick": kill,
        "death_tick": death,
        "t_won": outcome == "win",
        "ct_won": outcome == "loss",
        **{
            f"{who}_{k}": None if k.startswith("first") else 0
            for who in ("hero", "opp")
            for k in D.PER_SIDE
        }
    }
    r.update(side)
    return r


ROUNDS = [
    _row("win",
         10,
         kill=10,
         hero_fired=4,
         hero_hit=2,
         hero_fast=2,
         hero_fast_hit=1,
         hero_first_fire=3,
         hero_first_fire_hit=False,
         hero_first_hit=5,
         hero_hp_lost=27,
         hero_idle=6,
         hero_idle_fast=3,
         opp_hp_lost=100),
    _row("loss",
         20,
         death=20,
         opp_fired=6,
         hero_fired=2,
         hero_first_fire=4,
         hero_first_fire_hit=False,
         hero_hp_lost=100,
         hero_idle=18),
    _row("double",
         30,
         kill=30,
         death=30,
         hero_fired=3,
         hero_hit=3,
         hero_first_fire=2,
         hero_first_fire_hit=True,
         hero_first_hit=16,
         hero_hp_lost=100,
         hero_idle=27),
    _row("timeout", 161, hero_idle=161),
    _row("win", 8, kill=8),
]


def _res(rows=ROUNDS, fired=(9, 0, 9), unmatched=0):
    acct = {"fired": fired, "hit": (5, 0, 5), "fast": (2, 0, 2), "unmatched": unmatched}
    return {"rows": rows, "accounting": [acct], "accuracy": True, "hitboxes": True}


def test_the_cell_summary_reads_the_recorders_definitions():
    s = E.cell_summary(_res())
    assert (s["win"], s["loss"], s["double"], s["timeout"]) == (2, 1, 1, 1)
    assert (s["hero_first_shots"], s["hero_first_shot_hits"]) == (3, 1)
    assert s["hero_first_shot_hit_rate"] == 1 / 3 and s["hero_first_shot_ci"] == D.wilson(1, 3)
    # First hits on ticks 5 and 16: 16 ticks is 1 s, which Leetify leaves out.
    assert (s["hero_ttd_rounds"], s["hero_ttd_median"], s["hero_ttd_1s_share"]) == (2, 5.0, 0.5)
    # Time to death (the hero died on ticks 20 and 30) has its own key, never a ttd_ one.
    assert s["death_tick_median"] == 25.0 and "ttd_median" not in s
    assert s["hero_lifetime_mean"] == (10 + 20 + 30 + 161 + 8) / 5
    assert s["opp_lifetime_mean"] == (10 + 20 + 30 + 161 + 8) / 5
    assert s["hero_shots_per_kill"] == 9 / 3 and s["hero_hp_lost_mean"] == 227 / 5
    assert s["opp_shots_per_kill"] == 6 / 2
    assert (s["hero_hit_per_fast"], s["hero_hit_per_slow"]) == (1 / 2, 4 / 7)
    assert s["hero_moving_idle_share"] == 3 / (6 + 18 + 27 + 161)
    assert s["opp_first_shot_hit_rate"] is None and s["opp_ttd_median"] is None
    assert s["accounting_ok"]


@pytest.mark.parametrize("res", [_res(fired=(9, 0, 10)), _res(unmatched=1)])
def test_a_cell_whose_shots_do_not_sum_is_flagged(res):
    cell = {"opponent": "hard", "hero": "policy", "side": "T", **E.cell_summary(res)}
    assert not cell["accounting_ok"]
    out = {
        "rounds": 4,
        "seed": 0,
        "sides": ["T"],
        "aim": "mean",
        "lstm": "carried",
        "accuracy": True,
        "hitboxes": True,
        "cells": [cell],
        "wall_s": 0.0
    }
    assert "WARNING: the per-side shots do not sum" in E.format_table(out)


def _random_policy():
    import torch

    from cs2rl.experiment.oracle_statue import build_env
    from cs2rl.policy import build_policy
    env = build_env(0)
    try:
        torch.manual_seed(0)
        return build_policy(env, "cpu", aim_log_std_max=-2.9957, pin_pitch=True)
    finally:
        env.close()


@pytest.mark.parametrize("side", ["T", "CT"])
def test_each_cell_plays_its_side_and_a_reference_cell_is_run_duels(side):
    spec = "expert,style=stop"
    out = E.run_eval(_random_policy(), 64, rounds=2, seed=3, opponents=(spec, ), sides=(side, ))
    assert [c["hero"] for c in out["cells"]] == ["policy", "perfect", "mirror"]
    k = D.SIDES.index(side)
    for c in out["cells"]:
        assert c["side"] == side and c["accounting_ok"]
        assert {(r["hero_side"], r["seed"])
                for r in c["per_round"]} == {(side, 3 + k * D.CT_SEED_OFFSET)}
    for c, hero in zip(out["cells"][1:], ("perfect", spec), strict=True):
        assert c["per_round"] == D.run_duel(D.parse_bot(hero), D.parse_bot(spec), 2, 3,
                                            sides=(k, ))["rows"]


@pytest.mark.parametrize("aim", ["mean", "sample"])
def test_the_policy_plays_the_rounds_aim_eval_plays(aim):
    from cs2rl.eval.duel_bot import DuelBotActor
    from cs2rl.experiment import aim_eval
    from cs2rl.experiment.oracle_statue import STATUE
    policy, spec = _random_policy(), "expert,style=stop"
    bot = {spec: lambda s: DuelBotActor([STATUE], D.parse_bot(spec), np.random.default_rng([s, 1]))}
    theirs = aim_eval.run_eval(policy,
                               64,
                               episodes=2,
                               seed=3,
                               opponents=bot,
                               modes=((aim, "carried"), ))["cells"][0]["per_episode"]
    mine = E.run_eval(policy, 64, rounds=2, seed=3, opponents=(spec, ), aim=aim,
                      references=())["cells"][0]["per_round"]
    assert [(r["kill_tick"], r["hero_fired"] + r["opp_fired"], r["hero_hit"] + r["opp_hit"],
             r["t_xy"], r["ct_xy"]) for r in mine] == [(e["ttk"], e["shots_fired"], e["shots_hit"],
                                                        e["hero_xy"], e["statue_xy"])
                                                       for e in theirs]


def test_policy_cells_reproduce_in_any_order():
    import torch
    policy = _random_policy()

    def rounds(opponents, caller_seed):
        torch.manual_seed(caller_seed)                 # the caller's stream must not matter
        out = E.run_eval(policy,
                         64,
                         rounds=2,
                         seed=1,
                         opponents=opponents,
                         aim="sample",
                         references=())
        return {c["opponent"]: c["per_round"] for c in out["cells"]}

    pair = ("easy,style=stop", "hard,style=run")
    assert rounds(pair, 1) == rounds(pair[::-1], 2)


@pytest.mark.parametrize("trained, argv, want, warned", [
    ({
        "accuracy": True,
        "hitboxes": True
    }, [], (True, True), False),
    ({}, [], (False, False), False),
    ({
        "accuracy": True,
        "hitboxes": True
    }, ["--accuracy", "0"], (False, True), True),
    ({
        "accuracy": True,
        "hitboxes": True
    }, ["--stop", "sign"], (True, True), False),
])
def test_the_cli_plays_the_env_the_run_trained_with(monkeypatch, capsys, tmp_path, trained, argv,
                                                    want, warned):
    from cs2rl.experiment import aim_eval
    seen = {}
    cfg = {
        "bptt_horizon": 64,
        "env": "cs2-arena-duel",
        "n_active_per_team": 1,
        "pin_pitch": 1,
        "crouch_enabled": 0,
        "jump_enabled": 0,
        "round_time_ticks": 160,
        "laser_range": None,
        "max_turn_speed": None,
        **trained
    }
    monkeypatch.setattr(aim_eval, "load_checkpoint", lambda path: (None, cfg))

    def fake_run_eval(policy, horizon, **kw):
        seen.update(kw, horizon=horizon)
        cell = {"opponent": "hard", "hero": "policy", "side": "T", **E.cell_summary(_res())}
        return {
            "rounds": kw["rounds"],
            "seed": kw["seed"],
            "sides": kw["sides"],
            "aim": kw["aim"],
            "lstm": kw["lstm"],
            "accuracy": kw["accuracy"],
            "hitboxes": kw["hitboxes"],
            "cells": [cell],
            "wall_s": 0.0
        }

    monkeypatch.setattr(E, "run_eval", fake_run_eval)
    path = tmp_path / "out.json"
    assert E.main(["ckpt.pt", "--rounds", "7", "--json", str(path), *argv]) == 0
    assert (seen["accuracy"], seen["hitboxes"], seen["rounds"], seen["horizon"]) == (*want, 7, 64)
    assert (seen["sides"], seen["aim"], seen["lstm"]) == (["T"], "mean", "carried")
    assert seen["references"] == ["perfect", "mirror"]
    assert seen["stop"] == ("sign" if "--stop" in argv else None)
    assert ("WARNING: the run trained a different env" in capsys.readouterr().out) == warned
    assert json.loads(path.read_text())["checkpoint"] == "ckpt.pt"


def _v_checkpoint():
    """This checkout's V checkpoint, else the main checkout's (a worktree has no outputs/)."""
    common = subprocess.run(
        ["git", "-C",
         str(REPO_ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False)
    roots = [REPO_ROOT] + ([Path(common.stdout.strip()).parent] if common.returncode == 0 else [])
    for root in roots:
        if (root / V_CKPT).is_file():
            return root / V_CKPT
    pytest.skip(f"no {V_CKPT} under {[str(r) for r in roots]}")


@pytest.mark.slow
def test_the_cli_plays_every_cell_on_the_rung1b2_checkpoint(tmp_path, capsys):
    ckpt = _v_checkpoint()
    path = tmp_path / "duel_eval.json"
    assert E.main([str(ckpt), "--rounds", "2", "--json", str(path)]) == 0
    out = json.loads(path.read_text())
    assert out["env_mismatches"] == [] and (out["accuracy"], out["hitboxes"]) == (True, True)
    assert len(out["cells"]) == len(E.OPPONENTS) * (1 + len(E.REFERENCE_HEROES))
    for c in out["cells"]:
        assert c["accounting_ok"] and c["rounds"] == len(c["per_round"]) == 2, c
    printed = capsys.readouterr().out
    assert "WARNING" not in printed and "cone ON, from the run's config.json" in printed
