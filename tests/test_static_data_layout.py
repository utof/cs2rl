"""The StaticData layout hash — C's declaration vs the ctypes mirror's.

WHY THIS EXISTS: the C struct in cs2_types.h and StaticDataC in cs2_env.py are
two independent declarations of the same bytes, and until now the only thing
tying them together field-by-field was the 73-position PyArg_ParseTuple format
string in py_init — a mechanism that agrees by POSITION and says nothing about
names or types. tests/test_struct_sizes.py adds sizeof and three offsetof
anchors on top, which catch a size change and a tail shift but are blind to two
same-width fields swapped, or to an `int` in C described as a float in the
mirror.

Both sides here describe the prefix — [0, offsetof(StaticData, wall_list)), i.e.
everything Python publishes through binding.init — as an ordered list of
(name, offset, size, canonical type name), and the comparison is over that list
and its sha256. The two operands come from different places on purpose:

    C      — offsetof/sizeof expressions over the SD_PREFIX_FIELDS table in
             cs2_types.h, evaluated by the compiler that laid the struct out.
    Python — ctypes introspection of StaticDataC and nothing else.

If either side is ever reworked to read the other, every test in this file keeps
passing while comparing nothing. That is the failure mode to watch for in review.

HONEST SCOPE. This compares DECLARATIONS, so it cannot see a value-routing
mistake: Python assigning jump_enabled's value into the crouch_enabled field
puts the wrong number into a correctly-described slot and every quadruple below
still matches. That is what the sentinel round trip and the two-env pigeonhole
scheme in tests/test_struct_sizes.py cover, and why neither is retired here.

PITFALL: like every other check against binding, this reads the CURRENTLY BUILT
.so. After editing anything in src/cs2rl/c_env, rebuild
(`uv run --with "ziglang>=0.14.0,<0.15" python setup.py build_ext --inplace`)
before believing a pass OR a failure.
"""

import ctypes
import hashlib

from cs2rl.c_env import binding, cs2_env
from cs2rl.c_env.cs2_env import StaticDataC

# Largest alignment any StaticData prefix member can demand on the 64-bit targets
# this builds for: the pointer fields. Used as the ceiling on a legitimate
# padding gap between two consecutive table entries.
_MAX_ALIGN = 8


def _c():
    return binding.static_data_layout()


def _py():
    return cs2_env.static_data_layout()


def test_layout_fields_agree_field_by_field():
    """The two declarations must describe the same fields, in the same order.

    Asserted before the hash so a real drift reports WHICH field moved instead
    of two unequal hex strings. A pure hash test is unreadable on failure and
    this one costs nothing.
    """
    c_fields = tuple(tuple(row) for row in _c()["fields"])
    py_fields = _py()["fields"]

    c_names = [row[0] for row in c_fields]
    py_names = [row[0] for row in py_fields]
    assert c_names == py_names, (
        "SD_PREFIX_FIELDS (cs2_types.h) and StaticDataC._fields_ (cs2_env.py) name different "
        "prefix fields or list them in a different order. "
        f"only in C: {sorted(set(c_names) - set(py_names))}; "
        f"only in the mirror: {sorted(set(py_names) - set(c_names))}; "
        f"order differs: {[(a, b) for a, b in zip(c_names, py_names, strict=False) if a != b]}")

    # strict=False below: if the two sides had different lengths, the assert
    # above already said so and stopped. Raising ValueError while BUILDING a
    # failure message would replace that diagnosis with a traceback.
    mismatched = {
        c_row[0]: {
            "C (offset, size, type)": c_row[1:],
            "ctypes (offset, size, type)": py_row[1:],
        }
        for c_row, py_row in zip(c_fields, py_fields, strict=False) if c_row != py_row
    }
    assert not mismatched, (
        "the C struct and the ctypes mirror disagree about these fields. Fix the mirror in "
        "src/cs2rl/c_env/cs2_env.py or the table in src/cs2rl/c_env/cs2_types.h — never this assert. "
        f"{mismatched}")


def test_layout_preamble_agrees():
    """C's fixed expectation vs what ctypes actually did.

    The two sides reach this string by different routes on purpose: binding.c
    hardcodes SD_LAYOUT_PREAMBLE (what it REQUIRES of ctypes) while
    cs2_env._static_data_preamble() reads hasattr(StaticDataC, "_pack_") and
    getattr(StaticDataC, "_layout_") off the live class. Two hardcoded constants
    would compare nothing, so if this ever fails, check that the Python side is
    still introspecting before touching the C string.

    What it buys: setting _pack_ on StaticDataC would silently switch ctypes to
    MSVC layout rules on Linux and move fields without changing any single
    field's declared type.
    """
    assert _c()["preamble"] == _py()["preamble"]


def test_layout_hash_agrees():
    """The single token that summarises "these two declarations agree".

    binding.init consumes this in the packed-buffer world; here it is the
    end-to-end check that C's serialisation and its own sha256 (cs2_sha256.h)
    reproduce byte-for-byte what hashlib computes over the Python-side
    serialisation of the Python-side introspection.
    """
    c, py = _c(), _py()
    assert c["format"] == py["format"], (
        "layout serialisation format tags differ — the .so is stale, or one side's "
        f"_LAYOUT_FORMAT / SD_LAYOUT_FORMAT was bumped without the other: {c['format']!r} "
        f"vs {py['format']!r}")
    assert c["hash"] == py["hash"], (
        "layout hashes differ. If test_layout_fields_agree_field_by_field and "
        "test_layout_preamble_agrees both PASSED, the declarations match and the fault is in "
        "the serialisation or the hash itself — compare the line format in binding.c's "
        "SD_LAYOUT_ROW against cs2_env.static_data_layout(), and suspect cs2_sha256.h. "
        f"C={c['hash']} python={py['hash']}")


def test_layout_hash_is_a_sha256_digest():
    """Cheap shape check on the C side's hex output.

    A truncated or uninitialised buffer in cs2_sha256_final_hex would most
    likely still compare equal to itself; it would not be 64 lowercase hex
    characters. Pinned against hashlib's own output width rather than a literal.
    """
    digest = _c()["hash"]
    assert len(digest) == len(hashlib.sha256(b"").hexdigest())
    assert set(digest) <= set("0123456789abcdef")


def test_layout_prefix_boundary_matches_the_retained_offset_anchor():
    """prefix_size is the memcpy boundary; three sources must agree on it.

    The C table's own offsetof(StaticData, wall_list), the ctypes mirror's
    wall_list.offset, and the StaticData_wall_list_offset anchor that
    struct_sizes() has always published (and which W2 keeps, because the hash
    covers the prefix only and the anchors are the tail's coverage).
    """
    assert _c()["prefix_size"] == _py()["prefix_size"]
    assert _c()["prefix_size"] == StaticDataC.wall_list.offset
    assert _c()["prefix_size"] == binding.struct_sizes()["StaticData_wall_list_offset"]


def test_layout_table_tiles_the_whole_prefix():
    """No field of the prefix may be missing from the table.

    The hash compares two lists; it cannot notice a field that BOTH lists omit.
    The sizeof(StaticData) guard in tests/test_struct_sizes.py catches most of
    that, but not all of it — cs2_types.h records jump_enabled landing inside
    padding that already existed, leaving sizeof and both offset anchors
    unchanged. So walk the offsets instead: consecutive entries may be separated
    only by alignment padding, and the last entry must reach the prefix
    boundary within the same tolerance.

    HONEST LIMIT, for the same reason that motivates it: a forgotten field that
    fits ENTIRELY inside a pre-existing pad is still invisible here, because the
    gap it occupies was already legal. This narrows the hole, it does not close
    it. What closes it is adding the field to all three places at once — the
    rule stated at StaticDataC._fields_ and at SD_PREFIX_FIELDS.
    """
    fields = tuple(tuple(row) for row in _c()["fields"])
    assert fields, "the layout table is empty — SD_PREFIX_FIELDS lost its rows"
    assert fields[0][1] == 0, f"the prefix must start at offset 0, got {fields[0][1]}"

    gaps = {}
    # strict=False is REQUIRED, not a lint dodge: pairing a list with its own
    # tail is a sliding window, so the operands differ in length by one always.
    for (name, off, size, _), (next_name, next_off, _, _) in zip(fields, fields[1:], strict=False):
        gap = next_off - (off + size)
        if not 0 <= gap < _MAX_ALIGN:
            gaps[f"{name} -> {next_name}"] = gap
    last_name, last_off, last_size, _ = fields[-1]
    tail_gap = _c()["prefix_size"] - (last_off + last_size)
    if not 0 <= tail_gap < _MAX_ALIGN:
        gaps[f"{last_name} -> wall_list"] = tail_gap
    assert not gaps, (
        "gaps in the StaticData prefix that alignment padding cannot explain — a field is "
        "declared in cs2_types.h but missing from SD_PREFIX_FIELDS (a negative gap means "
        f"two entries overlap, i.e. the table is out of order). {{edge: gap_bytes}} = {gaps}")


def test_layout_type_column_actually_distinguishes_types():
    """The type column must not be degenerate.

    It is the only part of the hash that sees an int/float swap between two
    4-byte fields — StaticData's prefix is a long run of 4-byte scalars where
    offsets and sizes are identical either way. A future "simplification" that
    makes every field report the same canonical name (or drops the column to
    make a failing comparison pass) would leave the hash green and that whole
    class of drift uncovered, so pin the property here rather than trusting
    review to notice.
    """
    types = [row[3] for row in _c()["fields"]]
    assert {"c_int", "c_float"} <= set(types), (
        f"expected both int and float fields to be distinguishable in the type column: {set(types)}"
    )
    assert any(t.startswith("ptr_") for t in types)
    assert any(t.startswith("arr_") for t in types)


def test_canonical_names_cover_every_mirror_field_type():
    """Every prefix field's ctypes type maps into the shared vocabulary.

    cs2_env._canonical_ctype_name raises on a type binding.c's SD_TYPE_NAMES
    cannot produce, which is what stops a field declared c_double or c_uint32
    from being hashed under a name the C side could never emit. Calling it over
    the whole mirror here makes that a test failure at the point of the change
    rather than an import-time RuntimeError somewhere downstream.
    """
    prefix_end = [name for name, _ in StaticDataC._fields_].index("wall_list")
    for name, ctype in StaticDataC._fields_[:prefix_end]:
        canonical = cs2_env._canonical_ctype_name(ctype)
        assert canonical, name
    # The tail is deliberately NOT covered: wall_list is a nested struct with no
    # canonical name in this vocabulary, which is consistent with the hash
    # stopping at the prefix boundary.
    assert issubclass(dict(StaticDataC._fields_)["wall_list"], ctypes.Structure)
