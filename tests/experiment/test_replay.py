"""cs2rl.experiment.replay: save rounds of the arena harness as a replay, one command.

WHAT IS PINNED HERE
-------------------
* Recording changes nothing: ``save`` plays the same rounds as ``play`` with no recorder,
  per episode and per tick, each snapshot key the recorder and the text dump read equal as
  the next consumer reads it (the scripted actors read the same dicts, so a recorder writing
  into one could change the play).
* Round k does not depend on how many rounds follow.
* The text dump agrees with run_check: per round its F count is ``shots_fired``, its H
  count ``shots_hit``, tick 0 is the spawn (both spawn positions, the opponent at full hp),
  and a kill's last row is tick ``ttk`` with the opponent at 0 hp.
* The CLI writes the .rrd (one recording per round) and the .txt, and prints the command
  that opens it (``viz.render.view_command``: absolute path, any directory); it refuses
  ``--rounds 0`` before touching the file; ``default_path`` names a checkpoint by its run
  directory and stem.
* A checkpoint's rounds are the episodes of aim_eval's policy cell for the same opponent,
  seed and modes. It needs the Rung 1a checkpoint, which only the owner's checkouts have
  (outputs/ is not in git), so it skips without one.
"""
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import REPO_ROOT

pytest.importorskip("rerun")

RUNG1A_CKPT = "outputs/checkpoints/rung1a/s0/rung1a-s0.pt"
KEYS = ("x", "y", "z", "facing", "pitch", "hp", "alive", "fired_this_tick", "is_crouching")
ROUNDS, SEED, OPP = 3, 0, "walker-4-16"


class _Capture:
    """An ``on_step`` keeping a copy of every snapshot, per round."""

    def __init__(self):
        self.rounds = []

    def __call__(self, tick, st_prev, st):
        if tick == 1:
            self.rounds.append([{k: np.copy(st_prev[k]) for k in KEYS}])
        self.rounds[-1].append({k: np.copy(st[k]) for k in KEYS})


def _same_ticks(a, b):
    return [len(r) for r in a.rounds] == [len(r) for r in b.rounds] and all(
        np.array_equal(s[k], t[k]) for ra, rb in zip(a.rounds, b.rounds, strict=True)
        for s, t in zip(ra, rb, strict=True) for k in KEYS)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """One plain ``play`` and one ``save`` of the oracle against the walker; the save's
    snapshots are captured by its TickTable, which runs after the recorder."""
    from cs2rl.experiment import replay
    plain = _Capture()
    res_plain, _ = replay.play("oracle", OPP, ROUNDS, SEED, plain)
    recorded = _Capture()

    class CapturingTable(replay.TickTable):

        def __call__(self, tick, st_prev, st):
            recorded(tick, st_prev, st)
            super().__call__(tick, st_prev, st)

    table_cls, replay.TickTable = replay.TickTable, CapturingTable
    try:
        out = tmp_path_factory.mktemp("replay") / "oracle.rrd"
        res = replay.save("oracle", OPP, ROUNDS, SEED, out)
    finally:
        replay.TickTable = table_cls
    return res_plain, plain, res, recorded


def test_recording_changes_nothing(saved):
    res_plain, plain, res, recorded = saved
    assert len(res["per_episode"]) == ROUNDS and res["per_episode"] == res_plain["per_episode"]
    assert sum(len(r) for r in plain.rounds) > ROUNDS  # known count: ticks seen
    assert _same_ticks(plain, recorded)


def test_round_k_does_not_depend_on_how_many_follow():
    from cs2rl.experiment import replay
    two, _ = replay.play("oracle", OPP, 2, SEED, None)
    four, _ = replay.play("oracle", OPP, 4, SEED, None)
    assert two["per_episode"] == four["per_episode"][:2]


def test_the_text_dump_agrees_with_run_check(saved):
    _, _, res, _ = saved
    blocks = res["text"].read_text().split("\n\n")[1:]
    assert len(blocks) == ROUNDS
    for ep, block in zip(res["per_episode"], blocks, strict=True):
        lines = block.strip().split("\n")
        assert lines[0].startswith("round ") and lines[1].startswith("tick")
        rows = [ln.split() for ln in lines[2:]]
        assert rows[0][0] == "0" and rows[0][-1] == "100"              # tick 0: the spawn
        assert [float(v) for v in rows[0][1:3]] == [round(v, 1) for v in ep["hero_xy"]]
        assert [float(v) for v in rows[0][4:6]] == [round(v, 1) for v in ep["statue_xy"]]
        assert [r[0] for r in rows] == [str(t) for t in range(len(rows))]
        assert sum(r[-3] == "F" for r in rows) == ep["shots_fired"]
        assert sum(r[-2] == "H" for r in rows) == ep["shots_hit"]
        if ep["ttk"] is not None:
            assert rows[-1][0] == str(ep["ttk"]) and rows[-1][-1] == "0"
    assert sum(ep["ttk"] is not None for ep in res["per_episode"]) > 0 # a kill was checked


def test_the_cli_saves_and_prints_the_view_command(tmp_path, capsys):
    import sys

    from rerun.recording import load_archive

    from cs2rl.experiment import replay
    out = tmp_path / "statue.rrd"
    with pytest.raises(SystemExit):
        replay.main(["oracle", "--vs", "statue", "--rounds", "0", "--out", str(out)])
    assert not out.exists()
    assert replay.main(["oracle", "--vs", "statue", "--rounds", "2", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert f"view: {sys.executable} -m rerun --bind 127.0.0.1 {out.resolve()}\n" in printed
    assert len(re.findall(r"^round \d+: ", printed, re.M)) == 2
    assert out.with_suffix(".txt").read_text().startswith("# python -m cs2rl.experiment.replay ")
    assert [r.recording_id()
            for r in load_archive(out).all_recordings()] == ["statue-round-1", "statue-round-2"]


def test_default_path_names_the_run():
    from cs2rl.experiment.replay import default_path
    from cs2rl.spec.paths import RECORDINGS_DIR
    modal = "outputs/modal/r1b-f-c32679a-s0/checkpoints/checkpoints-20261009-072934.pt"
    assert default_path(
        modal, "heldout-24-48",
        0) == (RECORDINGS_DIR /
               "r1b-f-c32679a-s0-checkpoints-20261009-072934-vs-heldout-24-48-seed0.rrd")
    assert default_path(RUNG1A_CKPT, "statue",
                        2) == RECORDINGS_DIR / "s0-rung1a-s0-vs-statue-seed2.rrd"
    assert default_path("oracle", "statue", 0) == RECORDINGS_DIR / "oracle-vs-statue-seed0.rrd"


def _rung1a_checkpoint():
    """This checkout's Rung 1a checkpoint, else the main checkout's (a worktree has no outputs/)."""
    common = subprocess.run(
        ["git", "-C",
         str(REPO_ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False)
    roots = [REPO_ROOT] + ([Path(common.stdout.strip()).parent] if common.returncode == 0 else [])
    for root in roots:
        if (root / RUNG1A_CKPT).is_file():
            return root / RUNG1A_CKPT
    pytest.skip(f"no {RUNG1A_CKPT} under {[str(r) for r in roots]}")


@pytest.mark.slow
def test_a_checkpoint_plays_aim_eval_policy_cell(tmp_path):
    from cs2rl.experiment import aim_eval, replay
    ckpt = _rung1a_checkpoint()
    policy, cfg = aim_eval.load_checkpoint(ckpt)
    for aim, lstm in (("sample", "carried"), ("mean", "zeroed")):
        cell = aim_eval.run_eval(policy,
                                 int(cfg["bptt_horizon"]),
                                 episodes=ROUNDS,
                                 seed=SEED,
                                 opponents={OPP: aim_eval.OPPONENTS[OPP]},
                                 modes=((aim, lstm), ),
                                 accuracy=aim_eval.trained_accuracy(cfg))["cells"][0]
        res = replay.save(str(ckpt), OPP, ROUNDS, SEED, tmp_path / f"{aim}.rrd", aim, lstm)
        assert res["per_episode"] == cell["per_episode"], (aim, lstm)
        assert res["warnings"] == []
