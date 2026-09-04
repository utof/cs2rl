"""Gate: no env default is restated outside src/env_config.py (spec 2026-09-03 R11).

WHY THREE PROBES. A default can be written down in three shapes and each is
invisible to the other two probes:

  1. `name = value` / `"name": value` / `name: int = value` — a line regex sees
     it. Phase A's narrower regex missed the dict-entry and annotated shapes.
  2. `add_argument("--flag", dest="name", default=value)` — `dest=` and
     `default=` sit on different lines, so no line regex can pair them; an ast
     walk of the add_argument calls can.
  3. `getattr(args, "name", value)` — name and value are separated by a comma,
     so there is no `[:=]` to anchor on and it is not an add_argument call.

Shape 3 is the one #165 Phase B exists to remove from the parse layer, and a
gate with only probes 1 and 2 would report 0 both before and after that removal
— it could not tell the two states apart, which is exactly the silent pass this
file exists to prevent.

WHAT COUNTS AS A RESTATEMENT: a literal equal to the field's default, written
anywhere under src/ or scripts/ except src/env_config.py, which is where the
defaults are DECLARED. A value that merely happens to equal a default is still a
restatement unless there is a written reason (see ALLOWLIST).

NOT SCANNED: `default=None` in argparse. In this codebase a None argparse
default is never a copy of a field default, it is the "resolve this later"
sentinel — from the map for --pin-pitch, from nav.py for the R0-G trio.

NOT SCANNED: tests/, and widening the roots to reach it would BURY this gate
rather than strengthen it. The line probe finds 89 hits under tests/**/*.py
(measured 2026-09-04), 30 of them in tests/test_env_config.py — the ORACLE for
these very defaults, which must write the literals down; that is how it can tell EnvConfig()
still returns them. A test asserting a value is the opposite of a source
restating one, so do not "fix" this exclusion.

PITFALL: prose is scanned too, deliberately. A comment that restates a number is
a real finding — comments drift silently and are what readers trust. That cuts
both ways: a docstring anywhere under src/ that spells a counter-example as
`getattr(args, "<knob>", <the literal>)` is a HIT, and the fix is the prose.

PITFALL: the line regex pairs a name and a literal only when they share a LINE.
`src/train_config.py`'s R0-J docstring names `pbrs_gamma` on one line and the
field default on the next, so it is out of reach by construction, not by
allowance. Do not "helpfully" widen the probe across lines: the pairing would
stop meaning anything and every prose paragraph that mentions two knobs would
cross-match.
"""
import ast
import re
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

# I001 is suppressed, not fixed: the import has to follow the sys.path insert
# above, and ruff's isort wants it in the block at the top (same waiver as
# tests/test_env_config.py:21).
from env_config import KNOB_FIELDS, EnvConfig, RewardWeights           # noqa: E402, I001

DECLARATION = SRC / "env_config.py"

# name -> default, for the 23 weights and the seven non-None knobs. The three
# None-valued R0-G knobs are excluded: `x = None` is not a restatement of
# anything, it is the sentinel.
_CFG = EnvConfig()
FIELD_DEFAULTS = {
    **RewardWeights().as_dict(),
    **{
        k: getattr(_CFG, k)
        for k in KNOB_FIELDS if getattr(_CFG, k) is not None
    },
}

# The escape hatch for a line whose literal is set by a RULE rather than copied
# from a field — one that would read the same if the field default flipped.
# EXACTLY ONE ENTRY, and it is the reason the hatch exists: env_factory's eval
# role forces raw rewards (spec 2026-09-03 R11 / §3 _build_eval). That line
# would be spelled identically if the field's default were the other way round,
# so it states a rule; it does not restate a default. Before #165 the same
# behaviour came from the caller simply not passing the flag, which is why this
# list was empty until PR B2.
# Adding a second entry needs a written reason in the spec. Loosening a regex
# instead is NOT an acceptable fix — `NAME = False` is precisely the shape the
# probes exist to catch, and widening it away would blind the gate to every
# real restatement of a False default.
# The entry exempts the EXPRESSION, not the LINE: _hits_in strips each match
# out of the line and scans what is left, so a real restatement that happens to
# share a line with the allowed spelling is still a hit.
ALLOWLIST = ("config.replace(reward_symmetrize=False)", )

# The roots every probe reads, and one file that MUST be among the results.
# ANCHOR is derived from DECLARATION rather than written as a bare "src/..."
# string: a rename that moved the declaration would move the anchor with it,
# where a hard-coded path would just start pointing at nothing.
SCAN_ROOTS = ("src", "scripts")
ANCHOR = DECLARATION.parent / "train_config.py"


def _scanned_files():
    """The file list all three probes share — guarded so it cannot go empty.

    VACUITY GUARD, and it lives HERE rather than in a test of its own so every
    probe inherits it and none can opt out by not calling the test.
    test_field_defaults_covers_every_declared_default guards the probes' VALUE
    input; this guards the other one. `Path.rglob` on a missing or renamed root
    returns `[]` in silence, so a src/ restructure or a typo in SCAN_ROOTS
    degrades all three probes to no-ops that still report green.

    ALL THREE PROBES NOW DEPEND ON THIS, with nothing behind it. Until PR B2 the
    two count probes survived a truncated scan by accident — their pending rows
    went unmatched and `_assert_exactly`'s `stale` assert fired. B2 deleted the
    pending maps, so a scan that read nothing reports `{}` and reads GREEN, just
    as the argparse probe always did. The replacement evidence is
    test_the_probes_find_a_planted_restatement, which runs `_hits()` over a
    redirected REPO_ROOT and requires a hit under EACH root; it is the only
    thing in this file that can see a dropped root, because the two asserts
    below cannot — measured, a src-only scan and a one-file scan both PASS them.
    Do not delete it, and do not treat these asserts as covering its job.

    Per-root, not one total: a `scripts/` rename would otherwise hide behind
    `src/`'s files and the count probes would keep passing on a half scan.
    """
    per_root = {root: sorted((REPO_ROOT / root).rglob("*.py")) for root in SCAN_ROOTS}
    empty = sorted(r for r, found in per_root.items() if not found)
    assert not empty, (
        f"{empty} matched no *.py file, so every probe in this file is scanning less than it "
        f"claims to. rglob returns [] for a root that does not exist, so this is what a rename "
        f"or a typo in SCAN_ROOTS looks like — repoint SCAN_ROOTS, never delete this assert.")
    files = [p for found in per_root.values() for p in found if p != DECLARATION]
    assert ANCHOR in files, (
        f"{ANCHOR} is missing from the scan of {list(SCAN_ROOTS)} ({len(files)} file(s) found). "
        f"That file holds the CLI and the R0-G/R0-J prose, so a scan without it is not scanning "
        f"src/, whatever its length. If the file legitimately moved, repoint ANCHOR — it is "
        f"derived from DECLARATION so that a rename shows up here instead of silently.")
    return files


def _hits_in(files, pattern_for):
    """Scan a GIVEN file list. Split out of `_hits` so a test can point the
    scan at a planted tree — see test_the_probes_find_a_planted_restatement."""
    out = []
    for path in files:
        rel = path.name if not path.is_relative_to(REPO_ROOT) else path.relative_to(
            REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for _allowed in ALLOWLIST:
                line = line.replace(_allowed, "")
            for name, value in FIELD_DEFAULTS.items():
                if re.search(pattern_for(name, value), line):
                    out.append((rel, lineno, name))
    return out


def _hits(pattern_for):
    return _hits_in(_scanned_files(), pattern_for)


def _line_pattern(name, value):
    # `["']?` so dict entries match; the annotation alternation so `: int = 5`
    # matches — with `float` alone every `: int =` default is invisible, which is
    # exactly the harness signature's four knobs.
    #
    # The trailing `(?![\w.])` is the value's RIGHT EDGE. Without it the literal
    # is only a prefix: `pin_pitch=0` also matched `pin_pitch=0.5`, and
    # `crouch_enabled = 1` also matched `crouch_enabled = 100` and
    # `jump_enabled: int = 10` — none of which restates anything, and a PENDING
    # count can absorb one silently. The `.` inside the class costs one false
    # negative the other way: `crouch_enabled = 1.0` is no longer read as a
    # restatement of `1`. That trade is deliberate — a spelling nobody in this
    # tree uses, against a wrong count nobody would notice. Adding the boundary
    # left all three probe counts unchanged (9 / 1 / 0, same hit lines).
    return (rf'["\']?{re.escape(name)}["\']?\s*[:=]\s*'
            rf'(?:(?:int|float|bool)\s*=\s*)?{re.escape(repr(value))}(?![\w.])')


def _getattr_pattern(name, value):
    return (rf'getattr\(\s*[^,]+,\s*["\']{re.escape(name)}["\']\s*,\s*'
            rf'{re.escape(repr(value))}\s*\)')


def _assert_exactly(found, pending, probe):
    """Compare per-(file, field) hit COUNTS against `pending`, both directions.

    Two asserts, not one, because the two failures mean opposite things: more
    hits than allowed is a NEW restatement, fewer is a pinned row that has been
    fixed and must be deleted so the list cannot rot into standing permission.

    Both callers now pass an EMPTY `pending`, so the `stale` branch has nothing
    to check and the whole verdict rests on `extra`. That is deliberate — PR B2
    removed the last pending rows — and it is exactly why
    test_the_probes_find_a_planted_restatement exists: with no pinned row left
    to go unmatched, a walk that read nothing would report the same `{}` a clean
    tree does. The signature keeps the parameter because a future PR that stages
    a known restatement needs somewhere to pin it, and pinning it in BOTH
    directions is the property worth keeping.

    Line numbers are absent from the pinned data — counts survive the file churn
    a refactor causes, line numbers do not — but present in every failure
    message, which is what a reader needs to find the site.
    """
    counts = Counter((f, n) for f, _, n in found)
    lines = defaultdict(list)
    for f, lineno, n in found:
        lines[(f, n)].append(lineno)

    def _fmt(keys):
        rows = []
        for key in keys:
            where = ", ".join(f"{key[0]}:{ln}" for ln in sorted(lines.get(key, ())))
            rows.append(f"  {key[1]}: {counts[key]} hit(s) at [{where or 'nowhere'}], "
                        f"the PENDING list says {pending.get(key, 0)}")
        return "\n".join(rows)

    extra = sorted(k for k in counts if counts[k] > pending.get(k, 0))
    assert not extra, (f"{probe}: env default(s) restated outside src/env_config.py:\n" +
                       _fmt(extra) +
                       "\nDerive the value from EnvConfig()/RewardWeights() instead. A comment or "
                       "docstring counts — write `<the field default>`, not the number. If it is "
                       "genuinely not a restatement, add it to ALLOWLIST with a reason in the "
                       "spec — do not widen the regex.")
    stale = sorted(k for k in pending if counts[k] < pending[k])
    assert not stale, (f"{probe}: PENDING row(s) that no longer match:\n" + _fmt(stale) +
                       "\nThe site was fixed — drop the count or delete the row. A stale row is "
                       "standing permission for a restatement nobody is watching.")


def _argparse_default_offenders(tree, label):
    """add_argument calls in `tree` whose LITERAL `default=` equals a field default.

    Takes a parsed tree and a display label rather than a path, so the knock-out
    below can hand it a scratch source string. Without that this probe would be
    untestable in the committed suite: from B1 onward every real
    `default=_ENV_DEFAULTS.x` is an ast.Attribute, skipped by the Constant
    guard, so the probe reports 0 forever and nothing shows it can still find a
    planted `default=1`.
    """
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        dest = kw.get("dest")
        dest = dest.value if isinstance(dest, ast.Constant) else None
        if dest is None:
            for a in node.args:
                if isinstance(a, ast.Constant) and str(a.value).startswith("--"):
                    dest = a.value[2:].replace("-", "_")
                    break
        if dest not in FIELD_DEFAULTS or "default" not in kw:
            continue
        default = kw["default"]
        # `default=None` is the resolve-later sentinel, never a restatement.
        if not isinstance(default, ast.Constant) or default.value is None:
            continue
        if default.value == FIELD_DEFAULTS[dest]:
            offenders.append(f"{label}:{node.lineno}  --{dest} default={default.value!r}")
    return offenders


def test_no_restated_default_in_a_line():
    _assert_exactly(_hits(_line_pattern), {}, "line regex")


def test_no_restated_default_in_a_getattr_fallback():
    _assert_exactly(_hits(_getattr_pattern), {}, "getattr probe")


def test_no_argparse_default_restates_a_field_default():
    """add_argument defaults must be derived, e.g. `default=_ENV_DEFAULTS.jump_enabled`.

    A `dest=` on one line and a `default=` on another cannot be paired by any
    line regex, so without this probe R3 could delete a `getattr` fallback while
    argparse kept the same number one line away — the restatement moved, not
    removed.

    SCOPE: every file the other two probes read, not just src/train.py. Measured
    2026-09-04: 131 add_argument calls live in 17 of the 44 scanned files and
    only 54 of them are in train.py. None of the other 77 names an env knob
    today — that is the point. A probe scoped to one file reports the same 0
    whether or not that stays true, so it could never raise the alarm on the day
    a new parser in scripts/ copies a default.

    An unparseable file FAILS here, loudly and by name, instead of being skipped:
    a silently dropped file is the same blindness this scope fix removes, just
    wearing a different hat.
    """
    offenders = []
    for path in _scanned_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:
            pytest.fail(f"{rel} cannot be parsed, so this probe cannot see its add_argument "
                        f"calls: {exc}. Fix the file — never skip it, and never quietly narrow "
                        f"the scan to the files that happen to parse.")
        offenders += _argparse_default_offenders(tree, rel)
    assert not offenders, ("argparse default(s) restate a field default:\n  " +
                           "\n  ".join(offenders) +
                           "\nUse `default=_ENV_DEFAULTS.<field>` (EnvConfig() bound once above "
                           "the parser).")


def test_field_defaults_covers_every_declared_default():
    """Vacuity guard: the value table must hold EVERY default env_config declares.

    Both count probes above search for the values in FIELD_DEFAULTS and compare
    what they find against an EMPTY pending map. An emptied or truncated table
    makes them find nothing, and since PR B2 deleted the pending maps there is no
    side effect left to catch it: "nothing is restated" and "I am looking for
    nothing" are now the same green. This assert is the only thing that tells
    them apart, and its counterpart for the FILE input is
    test_the_probes_find_a_planted_restatement.

    Everything here is DERIVED from RewardWeights()/EnvConfig(); not one value is
    written down. Spelling the numbers out would make this file the first
    violation of the rule it exists to gate.
    """
    cfg = EnvConfig()
    weights = RewardWeights().as_dict()
    knobs = {k: getattr(cfg, k) for k in KNOB_FIELDS if getattr(cfg, k) is not None}
    sentinels = {k for k in KNOB_FIELDS if getattr(cfg, k) is None}

    assert weights and knobs, ("env_config declares no reward weights, or no non-None knobs. The "
                               "dataclass changed shape and every probe in this file now guards "
                               "nothing.")
    missing = sorted((set(weights) | set(knobs)) - set(FIELD_DEFAULTS))
    assert not missing, (f"FIELD_DEFAULTS does not cover {missing}. The probes cannot see a "
                         "restatement of a field they hold no value for, and blind reads as "
                         "green.")
    wrong = sorted(n for n, v in {**weights, **knobs}.items() if FIELD_DEFAULTS[n] != v)
    assert not wrong, (f"FIELD_DEFAULTS holds a value env_config no longer declares for {wrong}. "
                       "The probes are searching the tree for the wrong number.")
    leaked = sorted(sentinels & set(FIELD_DEFAULTS))
    assert not leaked, (f"{leaked} are None in EnvConfig — the resolve-later sentinel, not a "
                        "default. `x = None` restates nothing, and searching for it would flag "
                        "every unrelated sentinel in the tree.")


@pytest.mark.parametrize("probe", ["line", "getattr"])
def test_the_probes_can_actually_fail(probe):
    """Knock-out: a planted restatement must be found by the probe that owns it.

    Both probes scan a fixed root, so a bug in the regex or in the value table
    (an empty FIELD_DEFAULTS, say) makes every assertion above pass vacuously.
    This plants the two shapes in a scratch line and checks the patterns
    themselves rather than the roots.

    Name and value come OUT of FIELD_DEFAULTS — KeyError the moment the table
    stops covering the knob — so this self-test cannot keep passing against a
    table it no longer reads. Hardcoding the literal here would also be the very
    shape the getattr probe hunts, written inside the gate that hunts it.
    """
    name = "crouch_enabled"
    value = FIELD_DEFAULTS[name]
    other = (not value) if isinstance(value, bool) else value + 1
    pattern = _line_pattern if probe == "line" else _getattr_pattern

    def plant(v):
        return (f"{name} = {v!r}" if probe == "line" else f'x = getattr(args, "{name}", {v!r})')

    assert re.search(pattern(name, value), plant(value))
    assert not re.search(pattern(name, value), plant(other))


@pytest.mark.parametrize("probe", ["line", "getattr"])
def test_the_probes_find_a_planted_restatement(tmp_path, monkeypatch, probe):
    """Positive control for the WALK AND ITS SCOPE, not just the pattern.

    Both count probes now compare against an EMPTY pending map, so their
    "0 hits" is only evidence if reading a file, walking BOTH roots and
    reporting a hit still work end to end. Until PR B2 that was evidenced by
    the PENDING_B2 rows — a non-zero count asserted in both directions — and
    deleting them took the evidence with them.

    WHY IT REDIRECTS REPO_ROOT INSTEAD OF PLANTING A FILE AND CALLING
    `_hits_in` DIRECTLY: `_scanned_files()`'s asserts are satisfied by any
    list containing ANCHOR — a src-only scan and a one-file scan both pass
    them — so the only way to prove the walk's SCOPE is to run `_hits()`
    itself against a tree whose contents this test controls. One plant under
    each root is what makes a silently dropped root fail here: with `scripts/`
    dropped, the scripts plant disappears from the result.

    `test_the_probes_can_actually_fail` does not close this: it exercises the
    PATTERNS against a scratch string and never touches the file walk.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    declaration = tmp_path / "src" / "env_config.py"
    anchor = tmp_path / "src" / "train_config.py"
    name = "crouch_enabled"
    value = FIELD_DEFAULTS[name]
    plant = (f"{name} = {value!r}\n"
             if probe == "line" else f'x = getattr(args, "{name}", {value!r})\n')
    # The declaration is excluded by _scanned_files, so a hit from THIS file
    # would mean the exclusion broke; it must never appear in the result.
    declaration.write_text(plant)
    anchor.write_text(f"# a file the probes must be able to read\n{plant}")
    (tmp_path / "scripts" / "planted.py").write_text(plant)

    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "DECLARATION", declaration)
    monkeypatch.setattr(sys.modules[__name__], "ANCHOR", anchor)

    pattern = _line_pattern if probe == "line" else _getattr_pattern
    found = _hits(pattern)
    assert sorted((f, n) for f, _, n in found) == [
        ("scripts/planted.py", name),
        ("src/train_config.py", name),
    ], f"the walk did not report one hit under EACH root: {found}"
    # cross-probe negative: neither planted shape matches the other probe
    other = _getattr_pattern if probe == "line" else _line_pattern
    assert _hits(other) == []


def test_the_allowlist_exempts_only_its_own_expression(tmp_path):
    """The allowlisted span is stripped; the rest of the line is still scanned.

    The assertion is an EXACT list, so it bites on an EXTRA hit as well as on a
    missing one — and the three ways this entry can go wrong split across both.
    Each was run against exactly the planted text below, alongside the correct
    entry, which is what produces the asserted list:

      entry DELETED — lines 1 and 2 both report reward_symmetrize. Two extra
        hits.
      entry TRUNCATED to a prefix such as "config.replace(" — the strip removes
        only that prefix and leaves a bare `reward_symmetrize=False)` behind,
        which the line regex matches, so lines 1 and 2 report it again. The
        same two extra hits as deletion: a partial entry buys nothing.
      entry WIDENED into a blanket LINE exemption (the `continue` this
        replaced) — line 2's real crouch_enabled restatement VANISHES. A
        MISSING hit, not an extra one, which is why this assertion cannot be
        written as a subset check.

    Line 3 does not discriminate between those three — it is reported under all
    of them. It is the anchor for "only its OWN expression": add a second,
    broader entry covering `config.replace(pin_pitch=0)` and line 3 drops out
    of the list (measured).
    """
    planted = tmp_path / "allow.py"
    planted.write_text("    return _make(config=config.replace(reward_symmetrize=False))\n"
                       "    _make(config=config.replace(reward_symmetrize=False), "
                       "crouch_enabled=1)\n"
                       "    cfg = config.replace(pin_pitch=0)\n")
    found = [(lineno, n) for _, lineno, n in _hits_in([planted], _line_pattern)]
    assert found == [(2, "crouch_enabled"), (3, "pin_pitch")], found


def test_the_argparse_probe_can_actually_fail():
    """Knock-out: the walk finds a planted literal default and passes a derived one.

    The other two probes are knocked out against their patterns; this one has to
    be knocked out against a parsed tree, which is why the walk lives in a
    helper. Positive, negative and the None-sentinel branch, so "reports 0" is a
    measurement rather than an assumption.

    The planted default is READ from FIELD_DEFAULTS, for both reasons the other
    knock-out gives: a hardcoded literal would keep this test green against a
    table the probes no longer populate, and `default=<literal>` typed out here
    is precisely the shape this probe exists to catch.
    """
    name = "crouch_enabled"
    value = FIELD_DEFAULTS[name]
    flag = "--" + name.replace("_", "-")
    planted = textwrap.dedent(f"""
        parser.add_argument("{flag}", dest="{name}", default={value!r})
        """)
    derived = textwrap.dedent(f"""
        parser.add_argument("{flag}", dest="{name}",
                            default=_ENV_DEFAULTS.{name})
        """)
    sentinel = textwrap.dedent("""
        parser.add_argument("--pin-pitch", dest="pin_pitch", default=None)
        """)
    assert _argparse_default_offenders(ast.parse(planted),
                                       "<planted>") == [f"<planted>:2  --{name} default={value!r}"]
    assert _argparse_default_offenders(ast.parse(derived), "<derived>") == []
    assert _argparse_default_offenders(ast.parse(sentinel), "<sentinel>") == []
