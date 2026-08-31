"""binding.init's StaticData transfer — the pre/post capture oracle.

WHY THIS FILE EXISTS. The transfer of StaticData from Python into C is being
rewritten (spec 2026-08-31 §2 W2): a 73-position PyArg_ParseTuple format string
is replaced by a buffer Python packs field-by-field from StaticDataC. Every
other guard around that boundary compares two DECLARATIONS — the layout hash
(tests/test_static_data_layout.py), the sizeof/offsetof anchors
(tests/test_struct_sizes.py) — and a declaration comparison is blind to the
thing a rewrite of the VALUE-ROUTING surface actually risks: the right number
arriving in the wrong (correctly-described) field, or not arriving at all.

The sentinel round trip in tests/test_struct_sizes.py covers the routing of
every field make_env can set. What it cannot cover is the ~16 scalars make_env
does NOT expose — the map-derived geometry (grid_x_min, inv_x_range, x_offset,
...) and the nav.py timing constants — which it checks by key presence only,
because a value check there would have to restate the implementation's own
formula. Those are exactly the fields a packer bug would silently change.

So: this module compares against a CAPTURE, not against a formula. The fixture
holds the complete static_data_scalars() dict for two fixed make_env
configurations, recorded on the tree BEFORE the packer existed. A snapshot taken
after the rewrite would compare the packer to itself and assert nothing.

REGENERATING THE FIXTURE is legitimate when make_simple_map or a nav.py constant
changes on purpose, and at no other time. Regenerating it to make a failing
packer change go green deletes the only evidence that the two transfers agree,
which is the entire reason the file is committed. Command:

    UV_NO_SYNC=1 uv run python tests/test_static_data_pack.py --capture
"""

import json
import sys
from pathlib import Path

# Repo convention (tests/test_binding.py, tests/test_struct_sizes.py,
# tests/test_static_data_layout.py): `binding` is a C extension under src/c_env
# and conftest.py only puts src/ on the path.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "c_env"))
import binding                         # noqa: E402

from c_env.cs2_env import make_env     # noqa: E402

_FIXTURE = Path(__file__).parent / "fixtures" / "static_data_scalars_pre_w2.json"

# Bumped if the fixture's own shape changes, so a stale file fails on the tag
# rather than on a confusing KeyError deep inside a comparison.
_CAPTURE_FORMAT = "cs2rl-static-data-scalars-capture-v1"


def _load_fixture():
    with _FIXTURE.open() as fh:
        fixture = json.load(fh)
    assert fixture["_provenance"]["format"] == _CAPTURE_FORMAT, (
        f"{_FIXTURE.name} was written in format {fixture['_provenance']['format']!r}, this module "
        f"reads {_CAPTURE_FORMAT!r} — regenerate it with --capture")
    return fixture


def test_static_data_scalars_match_the_pre_w2_capture(simple_map):
    """Every scalar StaticData field must still hold the value it held pre-W2.

    EXHAUSTIVE over static_data_scalars()' key set, which
    test_static_data_scalars_covers_every_scalar_field pins to every scalar
    field of the ctypes mirror — so "exhaustive" here tracks the struct rather
    than a list someone maintains.

    EXACT equality, not pytest.approx. The C floats are widened to double on the
    way out and json round-trips a double through repr() without loss, so the
    recorded value IS the value a correct transfer produces, bit for bit. approx
    would silently tolerate a packer that rounds differently from the FMT path
    (double -> float32 twice, say), which is precisely the kind of drift a
    byte-identical-checkpoint branch cannot afford.

    Two configurations, not one: the 0/1 sim flags (pin_pitch / crouch_enabled /
    jump_enabled) cannot hold three distinct values inside a single env, so each
    is pinned by its vector of values across the two configs — the same
    pigeonhole argument spelled out at _BOOL_SENTINEL_CONFIGS in
    tests/test_struct_sizes.py.

    PITFALL: this reads the CURRENTLY BUILT .so. After editing src/c_env,
    rebuild before believing a pass OR a failure.
    """
    fixture = _load_fixture()
    for i, config in enumerate(fixture["configs"]):
        expected = config["scalars"]
        env = make_env(map_data=simple_map, **config["kwargs"])
        try:
            got = binding.static_data_scalars(env._capsule)
        finally:
            env.close()

        missing = sorted(set(expected) - set(got))
        added = sorted(set(got) - set(expected))
        assert not missing and not added, (
            f"config {i}: static_data_scalars() no longer publishes the same fields as the "
            f"pre-W2 capture. Absent now: {missing}; new: {added}. A new field is a legitimate "
            "reason to re-capture; a missing one is a regression.")

        wrong = {k: (expected[k], got[k]) for k in expected if got[k] != expected[k]}
        assert not wrong, (
            f"config {i} ({config['kwargs']}): the StaticData transfer no longer delivers the "
            "values it delivered before the packed-buffer migration. "
            f"{{field: (pre_w2, now)}} = {wrong}")


def _capture():
    """Write the fixture from the CURRENT tree. Run via --capture, never by pytest.

    The kwarg sets are the distinct-sentinel configurations
    tests/test_struct_sizes.py derives from the make_env signature; they are
    resolved once here and frozen into the fixture as literals, so the recorded
    values stay meaningful even if that generator is later changed.
    """
    import subprocess

    from test_struct_sizes import _SENTINEL_CONFIGS

    from map import make_simple_map

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    simple_map = make_simple_map()
    configs = []
    for kwargs in _SENTINEL_CONFIGS:
        env = make_env(map_data=simple_map, **kwargs)
        try:
            scalars = dict(sorted(binding.static_data_scalars(env._capsule).items()))
        finally:
            env.close()
        configs.append({"kwargs": dict(sorted(kwargs.items())), "scalars": scalars})

    why = ("pre-W2 baseline for the StaticData transfer rewrite (spec 2026-08-31 §2 W2); "
           "a snapshot taken after the rewrite would compare the packer to itself")
    regenerate = "UV_NO_SYNC=1 uv run python tests/test_static_data_pack.py --capture"
    provenance = {
        "format": _CAPTURE_FORMAT,
        "captured_at_commit": head,
        "map": "map.make_simple_map()",
        "why": why,
        "regenerate": regenerate,
    }
    _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with _FIXTURE.open("w") as fh:
        json.dump({"_provenance": provenance, "configs": configs}, fh, indent=1)
        fh.write("\n")
    print(f"wrote {_FIXTURE} at {head} ({len(configs)} configs, "
          f"{len(configs[0]['scalars'])} scalars each)")


if __name__ == "__main__":
    if sys.argv[1:] != ["--capture"]:
        raise SystemExit(f"usage: {sys.argv[0]} --capture")
    _capture()
