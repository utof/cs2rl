"""binding.init's StaticData transfer — the capture oracle and the packer's guards.

WHY THIS FILE EXISTS. The transfer of StaticData from Python into C was
rewritten (spec 2026-08-31 §2 W2): a 73-position PyArg_ParseTuple format string
was replaced by a buffer Python packs field-by-field from StaticDataC. Every
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
which is the entire reason the file is committed. Command, from the repository
root (`-m` so that `tests.test_struct_sizes`, the name pytest gives that module,
resolves):

    UV_NO_SYNC=1 uv run python -m tests.test_static_data_pack --capture

The second half of the module covers binding.init's three preconditions (layout
hash, buffer length, and the order of the two) plus the packing invariants that
nothing else can see — the zeroed spawn-array tail above all, which
static_data_scalars() excludes and only a training checkpoint hash would
otherwise catch.
"""

import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Repo convention (tests/test_binding.py, tests/test_struct_sizes.py,
# tests/test_static_data_layout.py): `binding` is a C extension under src/c_env
# and conftest.py only puts src/ on the path.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "c_env"))
import binding                         # noqa: E402

from c_env import cs2_env              # noqa: E402
from c_env.cs2_env import (            # noqa: E402
    TEAM_SIZE, StaticDataC, make_env,
)

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
        from tests.test_struct_sizes import _config_from_field_kwargs
        env = make_env(map_data=simple_map, config=_config_from_field_kwargs(config["kwargs"]))
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


# ── the packer's own invariants ───────────────────────────────────────────────


def _minimal_values():
    """A {field: value} mapping that packs a structurally valid, empty prefix.

    n_active_per_team is the one non-zero: env_init asserts it is >= 1 and a
    failed C assert aborts the whole process. Keeping it valid means that if one
    of the guards below were REMOVED, binding.init would build a degenerate env
    and the test would fail on the missing exception — instead of taking the
    pytest session down with it and reporting nothing.
    """
    values = {
        name: ([] if issubclass(ctype, ctypes.Array) else 0)
        for name, ctype in cs2_env._SD_PACKED_TYPES.items()
    }
    values["n_active_per_team"] = TEAM_SIZE
    return values


def _probe_pointer_args():
    """Placeholder arrays for binding.init's ten pointer parameters.

    Every rejection tested here fires before binding.init reaches PyArray_DATA,
    so these only have to be numpy arrays. They are oversized rather than
    1-element so that a regression which does reach the pointer assignments has
    a better chance of failing the assertion instead of the process.
    """
    return [np.zeros(4096, dtype=np.int8) for _ in cs2_env._SD_POINTER_FIELDS]


def _unpack(buffer):
    """Read a packed prefix back through the mirror.

    The buffer is the prefix only, so it is padded out to sizeof(StaticData)
    before the overlay — from_buffer_copy demands the full struct width and the
    tail is exactly what the packer deliberately does not send.
    """
    return StaticDataC.from_buffer_copy(buffer + bytes(ctypes.sizeof(StaticDataC) - len(buffer)))


def test_packed_buffer_is_exactly_the_prefix():
    """Length and boundary, from the mirror rather than a literal.

    binding.init accepts >= prefix_size, so an over-long buffer would pass every
    other check here while quietly meaning that Python and C disagree about
    where the C-owned tail starts.
    """
    buffer = cs2_env._pack_static_data(_minimal_values())
    assert len(buffer) == StaticDataC.wall_list.offset
    assert len(buffer) == binding.static_data_layout()["prefix_size"]


def test_packer_leaves_the_pointer_slots_null():
    """The ten pointer fields must not travel in the buffer.

    C has to hold the numpy buffers' own addresses — kept alive by Cs2Env._refs
    — so an address packed here would dangle. binding.init assigns them after
    the memcpy; if the packer ever wrote them, this is where that shows up.
    """
    unpacked = _unpack(cs2_env._pack_static_data(_minimal_values()))
    non_null = [n for n in cs2_env._SD_POINTER_FIELDS if getattr(unpacked, n)]
    assert not non_null, f"packed a non-NULL pointer for {non_null}"


def test_spawn_array_tail_is_zeroed():
    """The one invariant with no other oracle in the test suite.

    The old transfer memcpy'd n_*_spawns elements into a calloc'd struct, so the
    slots past the count read as zero. static_data_scalars() excludes the array
    fields, so a packer that left garbage there would pass every other test in
    this file and in tests/test_struct_sizes.py; only a byte-comparison of a
    trained checkpoint would notice, and only if the sim happened to read those
    slots. Assert it directly instead.
    """
    values = _minimal_values()
    values["t_spawns"] = [7, 9]
    values["n_t_spawns"] = 2
    values["ct_spawns"] = [3]
    values["n_ct_spawns"] = 1
    unpacked = _unpack(cs2_env._pack_static_data(values))
    assert list(unpacked.t_spawns) == [7, 9] + [0] * (cs2_env._T_SPAWN_CAPACITY - 2)
    assert list(unpacked.ct_spawns) == [3] + [0] * (cs2_env._CT_SPAWN_CAPACITY - 1)


def test_spawn_capacities_are_derived_from_the_mirror():
    """Not hardcoded 15/5 — the numbers already live in two other places.

    ctypes.sizeof on a field read off the class raises TypeError ("this type has
    no size") because it is a descriptor, which is why the derivation uses
    `.size`; pinning both spellings here keeps a future "simplification" from
    quietly reintroducing the literals.
    """
    fields = dict(StaticDataC._fields_)
    assert cs2_env._T_SPAWN_CAPACITY == fields["t_spawns"]._length_
    assert cs2_env._CT_SPAWN_CAPACITY == fields["ct_spawns"]._length_
    with pytest.raises(TypeError):
        ctypes.sizeof(StaticDataC.t_spawns)


def test_packer_rejects_a_forgotten_field():
    """The arity guard that replaces PyArg_ParseTuple's argument count.

    A dropped argument used to be a TypeError. A dropped dict key would leave a
    zero in the buffer, and zero is a plausible value for nearly every field
    here, so the packer has to refuse the mapping instead of packing it.
    """
    values = _minimal_values()
    del values["pbrs_gamma"]
    with pytest.raises(RuntimeError, match="pbrs_gamma"):
        cs2_env._pack_static_data(values)

    values = _minimal_values()
    values["reward_for_vibes"] = 1.0
    with pytest.raises(RuntimeError, match="reward_for_vibes"):
        cs2_env._pack_static_data(values)


def test_packer_rejects_an_out_of_range_int():
    """ctypes truncates where PyArg_ParseTuple's "i" raised OverflowError.

    Losing that error would turn a bad config into a run with round_time == 0
    rather than a stopped process, which is the silent-failure direction this
    branch cannot afford.
    """
    values = _minimal_values()
    values["round_time"] = 2**40
    with pytest.raises(OverflowError, match="round_time"):
        cs2_env._pack_static_data(values)


def test_packer_rejects_over_capacity_spawns():
    """Fixed-size packing turns an overrun into truncation; make it loud."""
    values = _minimal_values()
    values["t_spawns"] = list(range(cs2_env._T_SPAWN_CAPACITY + 1))
    with pytest.raises(ValueError, match="t_spawns"):
        cs2_env._pack_static_data(values)


# ── binding.init's preconditions ──────────────────────────────────────────────


def _layout_hash():
    return cs2_env.static_data_layout()["hash"]


def _perturbed_layout_hash():
    """A well-formed digest that is not this build's.

    Perturbing one character rather than passing junk keeps the test honest: it
    proves binding.init COMPARES the hash, not merely that it rejects something
    that does not look like one.
    """
    digest = _layout_hash()
    return ("0" if digest[0] != "0" else "1") + digest[1:]


def test_init_rejects_a_short_buffer():
    """The length check, which the layout hash cannot stand in for.

    The hash compares two DECLARATIONS; it says nothing about how many bytes
    actually arrived. Without an explicit check, a short buffer is a read past
    the end of a Python bytes object straight into C's StaticData.
    """
    short = cs2_env._pack_static_data(_minimal_values())[:-1]
    with pytest.raises(ValueError, match="at least"):
        binding.init(short, _layout_hash(), 0, 0.0, *_probe_pointer_args())


def test_init_rejects_a_perturbed_layout_hash():
    """The hash must be CONSUMED, not merely accepted.

    Nothing else checks this at RUNTIME: tests/test_static_data_layout.py
    compares the two sides at test time, so an implementation that took the
    argument and ignored it would pass every other check in the suite. What the
    runtime comparison catches is the case that actually happens — a .so built
    from different headers than the installed cs2_env.py, which is easy because
    the extension is a gitignored local artifact.
    """
    buffer = cs2_env._pack_static_data(_minimal_values())
    with pytest.raises(RuntimeError, match="layout hash mismatch"):
        binding.init(buffer, _perturbed_layout_hash(), 0, 0.0, *_probe_pointer_args())


def test_layout_hash_is_checked_before_the_buffer_is_read():
    """Ordering, established by which of the two errors an empty buffer gets.

    A zero-length buffer with the RIGHT hash hits the length check — that is the
    control below. The same zero-length buffer with a WRONG hash must report the
    hash instead, which can only happen if the hash comparison runs first. Since
    the length check itself precedes the memcpy, this pins the hash check ahead
    of the memcpy too, without needing to observe the copy.

    Why the order matters rather than being cosmetic: a buffer packed against a
    different declaration has a different prefix size, so checking its length
    against THIS build's prefix size first would be comparing it to a number
    that does not describe it.
    """
    with pytest.raises(ValueError, match="at least"):
        binding.init(b"", _layout_hash(), 0, 0.0, *_probe_pointer_args())
    with pytest.raises(RuntimeError, match="layout hash mismatch"):
        binding.init(b"", _perturbed_layout_hash(), 0, 0.0, *_probe_pointer_args())


def _capture():
    """Write the fixture from the CURRENT tree. Run via --capture, never by pytest.

    The kwarg sets are the distinct-sentinel configurations
    tests/test_struct_sizes.py derives from the make_env signature; they are
    resolved once here and frozen into the fixture as literals, so the recorded
    values stay meaningful even if that generator is later changed.
    """
    import subprocess

    from map import make_simple_map
    from tests.test_struct_sizes import _SENTINEL_CONFIGS, _config_from_field_kwargs

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    simple_map = make_simple_map()
    configs = []
    for kwargs in _SENTINEL_CONFIGS:
        env = make_env(map_data=simple_map, config=_config_from_field_kwargs(kwargs))
        try:
            scalars = dict(sorted(binding.static_data_scalars(env._capsule).items()))
        finally:
            env.close()
        configs.append({"kwargs": dict(sorted(kwargs.items())), "scalars": scalars})

    why = ("pre-W2 baseline for the StaticData transfer rewrite (spec 2026-08-31 §2 W2); "
           "a snapshot taken after the rewrite would compare the packer to itself")
    regenerate = "UV_NO_SYNC=1 uv run python -m tests.test_static_data_pack --capture"
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
