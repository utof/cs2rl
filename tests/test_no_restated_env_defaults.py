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
# EMPTY TODAY, and that is a measurement: the one candidate the spec named,
# env_factory._build_eval forcing raw rewards, turns out to force nothing. It
# OMITS reward_symmetrize entirely and inherits the parameter default, so there
# is no literal on any of its lines for a probe to see. `git log -S` finds the
# allowed spelling in no commit of src/ or scripts/, ever.
# Adding an entry needs a written reason in the spec. Loosening a regex instead
# is NOT an acceptable fix — `NAME = False` is precisely the shape the probes
# exist to catch.
ALLOWLIST = ()

# Sites that are known restatements today and are removed by PR B2, each with
# the ruling that removes it. Keyed by (file, field) and holding the expected
# HIT COUNT — never line numbers. This file lands last in B1, after five commits
# have moved code inside src/train.py, so line-keyed rows would report every
# entry as both extra (new line) and stale (old line) on their first run and the
# gate would fail on bookkeeping. Counts survive the churn and still bite in both
# directions: a new restatement raises a count or adds a key, a fixed site lowers
# one and the row must then be deleted. PR B2 deletes both dicts; PR B3 adds
# nothing to them.
# yapf: disable — pyproject's spaces_before_comment stops shove these standalone
# comments out to column 71 and split every `key:` from its count onto its own
# line, which is unreadable for the two tables a reader comes here to read.
# Same waiver as src/map.py:335.
# yapf: disable
PENDING_B2 = {
    # the shim's own knob defaults (2 sites for reward_symmetrize: make_puffer_env,
    # deleted by R5, and build_env_factory's parameter, deleted by R4)
    ("src/train.py", "reward_symmetrize"): 2,
    ("src/train.py", "pin_pitch"): 1,
    ("src/train.py", "crouch_enabled"): 1,
    ("src/train.py", "jump_enabled"): 1,
    # _build_trainer_for_test's four annotated defaults
    ("src/train_test_harness.py", "n_active_per_team"): 1,
    ("src/train_test_harness.py", "pin_pitch"): 1,
    ("src/train_test_harness.py", "crouch_enabled"): 1,
    ("src/train_test_harness.py", "jump_enabled"): 1,
}
PENDING_B2_GETATTR = {
    # build_train_env_factory's symmetrize fallback, deleted with the function's
    # rewrite in PR B2
    ("src/train.py", "reward_symmetrize"): 1,
}
# yapf: enable


def _scanned_files():
    files = [p for root in ("src", "scripts") for p in sorted((REPO_ROOT / root).rglob("*.py"))]
    return [p for p in files if p != DECLARATION]


def _hits(pattern_for):
    out = []
    for path in _scanned_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if any(a in line for a in ALLOWLIST):
                continue
            for name, value in FIELD_DEFAULTS.items():
                if re.search(pattern_for(name, value), line):
                    out.append((rel, lineno, name))
    return out


def _line_pattern(name, value):
    # `["']?` so dict entries match; the annotation alternation so `: int = 5`
    # matches — with `float` alone every `: int =` default is invisible, which is
    # exactly the harness signature's four knobs.
    return (rf'["\']?{re.escape(name)}["\']?\s*[:=]\s*'
            rf'(?:(?:int|float|bool)\s*=\s*)?{re.escape(repr(value))}')


def _getattr_pattern(name, value):
    return (rf'getattr\(\s*[^,]+,\s*["\']{re.escape(name)}["\']\s*,\s*'
            rf'{re.escape(repr(value))}\s*\)')


def _assert_exactly(found, pending, probe):
    """Compare per-(file, field) hit COUNTS against `pending`, both directions.

    Two asserts, not one, because the two failures mean opposite things: more
    hits than allowed is a NEW restatement, fewer is a PENDING row that has been
    fixed and must be deleted so the list cannot rot into standing permission.
    Line numbers are absent from the pinned data (see PENDING_B2) but present in
    every failure message — that is what a reader needs to find the site.
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
    _assert_exactly(_hits(_line_pattern), PENDING_B2, "line regex")


def test_no_restated_default_in_a_getattr_fallback():
    _assert_exactly(_hits(_getattr_pattern), PENDING_B2_GETATTR, "getattr probe")


def test_no_argparse_default_restates_a_field_default():
    """add_argument defaults must be derived, e.g. `default=_ENV_DEFAULTS.jump_enabled`.

    A `dest=` on one line and a `default=` on another cannot be paired by any
    line regex, so without this probe R3 could delete a `getattr` fallback while
    argparse kept the same number one line away — the restatement moved, not
    removed.
    """
    offenders = _argparse_default_offenders(ast.parse((SRC / "train.py").read_text()),
                                            "src/train.py")
    assert not offenders, ("argparse default(s) restate a field default:\n  " +
                           "\n  ".join(offenders) +
                           "\nUse `default=_ENV_DEFAULTS.<field>` (EnvConfig() bound once above "
                           "the parser).")


@pytest.mark.parametrize("probe", ["line", "getattr"])
def test_the_probes_can_actually_fail(probe):
    """Knock-out: a planted restatement must be found by the probe that owns it.

    Both probes scan a fixed root, so a bug in the regex or in the value table
    (an empty FIELD_DEFAULTS, say) makes every assertion above pass vacuously.
    This plants the two shapes in a scratch line and checks the patterns
    themselves rather than the roots.
    """
    pattern = _line_pattern if probe == "line" else _getattr_pattern
    line = ("crouch_enabled = 1" if probe == "line" else 'x = getattr(args, "crouch_enabled", 1)')
    assert re.search(pattern("crouch_enabled", 1), line)
    assert not re.search(pattern("crouch_enabled", 1), line.replace("1", "0"))


def test_the_argparse_probe_can_actually_fail():
    """Knock-out: the walk finds a planted literal default and passes a derived one.

    The other two probes are knocked out against their patterns; this one has to
    be knocked out against a parsed tree, which is why the walk lives in a
    helper. Positive, negative and the None-sentinel branch, so "reports 0" is a
    measurement rather than an assumption.
    """
    planted = textwrap.dedent("""
        parser.add_argument("--crouch-enabled", dest="crouch_enabled", default=1)
        """)
    derived = textwrap.dedent("""
        parser.add_argument("--crouch-enabled", dest="crouch_enabled",
                            default=_ENV_DEFAULTS.crouch_enabled)
        """)
    sentinel = textwrap.dedent("""
        parser.add_argument("--pin-pitch", dest="pin_pitch", default=None)
        """)
    assert _argparse_default_offenders(ast.parse(planted),
                                       "<planted>") == ["<planted>:2  --crouch_enabled default=1"]
    assert _argparse_default_offenders(ast.parse(derived), "<derived>") == []
    assert _argparse_default_offenders(ast.parse(sentinel), "<sentinel>") == []
