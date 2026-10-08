"""tests/test_demo_format.py — Batch 6 Task 3 (spec §7 schema, R3, R8).

Validates REAL generated demos (a small set produced into tmp_path by the
actual generator, not synthetic fixtures) so the test exercises the whole
record path: priming step, carrier-row capture, block masking, discard logic.

Load-bearing checks:
  * shapes/dtypes match spec §7 exactly (train_bc.py will np.load these and
    feed them straight to torch — a silent int32/float64 here becomes a
    confusing dtype error there);
  * metadata matches the LIVE constants via imports (never literals — the
    whole point of self-identifying demos is catching layout drift like the
    Task 2.5 107→110 bump);
  * teammate/enemy blocks are all-zero (spec R8 mask-to-zero decision) while
    the Task 2.5 goal-direction slots are NOT masked (they live in the self
    block — masking them would gut the generalization plan);
  * recorded actions are within the env's own bounds (discrete head sizes,
    |Δyaw| ≤ MAX_TURN_SPEED_RAD) — the labels a BC loss will consume.
"""
import numpy as np
import pytest

from cs2rl import bc_demos
from cs2rl.env.nav import MAX_TURN_SPEED_RAD, ROUND_TIME
from cs2rl.spec.action import ACTION_DIM, ACTION_HEAD_SIZES, AIM_DIM
from cs2rl.spec.obs import OBS_BLOCKS, OBS_DIM


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    """Generate a small real demo set once for the whole module: 2 seeds × 5
    carrier slots (~70 ticks each, all expected in-budget per Gate 0)."""
    out = tmp_path_factory.mktemp("demos")
    stats = bc_demos.generate_demos(2, out)
    assert stats["kept"] == 10, f"expected 10/10 kept (Gate 0 measured 100/100), got {stats}"
    assert stats["discarded"] == 0
    return out


def test_the_default_out_dir_is_the_one_train_bc_reads(monkeypatch):
    """`--out` defaults to train_bc.DEFAULT_DEMO_DIR, the directory BC training reads.

    WHY (#204, review C1): the default is computed from this module's own location
    (`REPO_ROOT / "outputs" / "demos"`). Moving the file from scripts/ into
    src/cs2rl/ turned the old `.parent.parent` into <repo>/src, so `python -m
    cs2rl.bc_demos` wrote to a gitignored src/outputs/demos that train_bc never
    reads, with rc 0 and nothing red. The value is read back from the PARSER
    (main() with generate_demos stubbed), not from REPO_ROOT, so a default that
    stops being built on REPO_ROOT is caught too.
    """
    from cs2rl import train_bc

    seen = {}

    def fake_generate_demos(n_seeds, out_dir, start_seed=0):
        seen["out_dir"] = out_dir
        return {"kept": 1}

    monkeypatch.setattr(bc_demos, "generate_demos", fake_generate_demos)
    assert bc_demos.main([]) == 0
    assert seen["out_dir"] == train_bc.DEFAULT_DEMO_DIR, (
        f"python -m cs2rl.bc_demos would write demos to {seen['out_dir']}, but train_bc "
        f"reads {train_bc.DEFAULT_DEMO_DIR}")


def _load_all(demo_dir):
    files = sorted(demo_dir.glob("*.npz"))
    assert files, "generator produced no .npz files"
    return [np.load(f) for f in files]


def test_demo_schema_shapes_and_dtypes(demo_dir):
    for d in _load_all(demo_dir):
        T = int(d["tick_count"])
        assert T > 0
        assert d["obs"].shape == (T, OBS_DIM)
        assert d["obs"].dtype == np.float32
        assert d["discrete_actions"].shape == (T, ACTION_DIM)
        assert d["discrete_actions"].dtype == np.int64
        assert d["continuous_actions"].shape == (T, AIM_DIM)
        assert d["continuous_actions"].dtype == np.float32
        assert d["dones"].shape == (T, )
        assert d["dones"].dtype == np.bool_
        # Episode-boundary contract: exactly one done, on the plant tick.
        assert d["dones"][-1]
        assert d["dones"].sum() == 1


def test_demo_metadata_matches_live_constants(demo_dir):
    """Self-identifying metadata (spec §7) vs the LIVE generated constants —
    imports, not literals, so this fails loudly on the next obs-layout bump."""
    for d in _load_all(demo_dir):
        assert int(d["OBS_DIM"]) == OBS_DIM
        assert int(d["ACTION_DIM"]) == ACTION_DIM
        assert int(d["AIM_DIM"]) == AIM_DIM
        assert str(d["map"]) == bc_demos.MAP_NAME
        assert int(d["tick_count"]) <= ROUND_TIME
        assert 0 <= int(d["carrier_idx"]) <= 4, "carrier must be a T-side slot (spec R3)"
        assert len(str(d["git_sha"])) == 40, "git_sha must be a full 40-char sha"


def test_teammate_and_enemy_blocks_masked_to_zero(demo_dir):
    """Spec R8: the ~68 frozen-idle-agent dims must be zero in every recorded
    tick. Sliced via OBS_BLOCKS — the boundaries moved in Task 2.5 and any
    hardcoded 25/53/93 here would silently check the wrong slots."""
    tm = slice(*OBS_BLOCKS["teammate"])
    en = slice(*OBS_BLOCKS["enemy"])
    for d in _load_all(demo_dir):
        assert np.all(d["obs"][:, tm] == 0.0), "teammate block not masked"
        assert np.all(d["obs"][:, en] == 0.0), "enemy block not masked"


def test_goal_direction_slots_not_masked(demo_dir):
    """The Task 2.5 bearing/distance slots (last 3 of the self block) are the
    generalization signal BC is supposed to lean on — they must survive the
    R8 mask. sin²+cos²=1 means the cos slot can never be all-zero in a real
    trajectory, so all-zeros here would mean the mask ate the self block."""
    self_stop = OBS_BLOCKS["self"][1]
    for d in _load_all(demo_dir):
        bearing = d["obs"][:, self_stop - 3:self_stop]
        assert np.any(bearing != 0.0), "goal-direction slots are all zero (masked or unwritten)"
        # sin/cos pair invariant on every tick.
        norms = bearing[:, 0]**2 + bearing[:, 1]**2
        assert np.allclose(norms, 1.0, atol=1e-4)
        # Distance must be positive along the walk and (near-)minimal at the
        # plant tick — the carrier walks TOWARD the site.
        assert bearing[0, 2] > bearing[-1, 2]


def test_actions_are_valid_bc_labels(demo_dir):
    for d in _load_all(demo_dir):
        disc = d["discrete_actions"]
        for head, size in enumerate(ACTION_HEAD_SIZES):
            assert disc[:, head].min() >= 0
            assert disc[:, head].max() < size, f"head {head} out of range"
        cont = d["continuous_actions"]
        # Expert pre-clamps Δyaw so the label equals what the env applied.
        assert np.all(np.abs(cont[:, 0]) <= MAX_TURN_SPEED_RAD + 1e-6)
        assert np.all(cont[:, 1] == 0.0), "expert always emits pitch=0 (level)"
        # The plant tick must press USE (head 4) and stop moving.
        assert disc[-1, 4] == 1
        assert disc[-1, 0] == 0


# Within-block offsets of the two carrier-identifying slots. These are the one
# place this file uses numbers rather than imports, because spec/obs.py is
# generated with BLOCK boundaries only — it has no per-field table to import.
# Sources: cs2_observations.h `obs[22] = (float)(a->team == 0 && bomb_carrier(g) == i)` and
# `obs[gb + 13] = (... i == g->round_designated_carrier_id)`. They are written
# relative to their block bases below and range-checked against OBS_BLOCKS, so
# a block-boundary shift (25/53/93 → 28/56/96 happened in Task 2.5) moves them
# automatically and a within-block layout change trips the range assert.
SELF_HAS_BOMB_OFFSET = 22
GLOBAL_DESIGNATED_CARRIER_OFFSET = 13


def test_carrier_row_only_and_carrier_state_in_obs(demo_dir):
    """R3 sanity via obs content: the recorded row must be the CARRIER's —
    self-has-bomb and the designated-carrier role bit are 1.0 from the very
    first tick (the priming step lands the setup pokes before recording
    starts)."""
    self_lo, self_hi = OBS_BLOCKS["self"]
    gb, g_hi = OBS_BLOCKS["global"]
    has_bomb = self_lo + SELF_HAS_BOMB_OFFSET
    role_bit = gb + GLOBAL_DESIGNATED_CARRIER_OFFSET
    assert self_lo <= has_bomb < self_hi, "self-has-bomb slot fell outside the self block"
    assert gb <= role_bit < g_hi, "designated-carrier slot fell outside the global block"
    for d in _load_all(demo_dir):
        assert np.all(d["obs"][:, has_bomb] == 1.0), "self-has-bomb not set — wrong row recorded?"
        assert np.all(d["obs"][:, role_bit] == 1.0), "designated-carrier role bit not pinned"
