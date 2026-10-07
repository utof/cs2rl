"""No trainer or policy state is read through a getattr default or hasattr (#355).

WHAT. test_no_trainer_or_policy_state_is_read_through_a_fallback reads every .py file under
SCAN_ROOTS and fails on each ``getattr(obj, key, default)`` or ``hasattr(obj, key)`` whose
object is a trainer or a policy, unless ALLOWED_FALLBACKS names its (file, qualname, key).
A site is a hit when either of two nets flags it:
  * receiver net: the object is a name or attribute chain whose last part is ``trainer`` or
    contains ``polic`` (``trainer``, ``run.trainer``, ``policy``, ``past_policy``,
    ``self.policy``, ``trainer.uncompiled_policy``), or ``self`` in a method of a
    STATE_CLASSES class (nested defs included), or a name an enclosing function assigns one
    of those (``t = trainer``). Any key counts, so an attribute nobody declares is caught.
  * key net: the key is a constant that a STATE_CLASSES class or stock PuffeRL stores on
    ``self`` (or registers as a buffer or parameter), however the object is spelled, unless
    the object is an argparse namespace (``args``, ``<x>.args``): option names such as
    ``aim_log_std_max``, ``tct_split_heads`` and ``tct_split_trunk`` are policy attributes too.
WHY. A fallback tolerates a missing attribute, so a renamed or undeclared one silently takes
the default. Measured in #355 on a pin_pitch policy with ``aim_dim_mask`` renamed: the old
``getattr(policy, "aim_dim_mask", None)`` handed the sampler no mask, so the pinned pitch
dim's log-prob was counted as live and ``log_aim_log_std`` logged the dead pitch sigma.
Every policy that reaches these reads is a Dust2Policy (or torch.compile's wrapper, which
forwards attribute reads), which declares its attributes, and PufferLib is pinned exactly in
pyproject.toml, so a default is legitimate only where the attribute can really be absent.
Those sites are ALLOWED_FALLBACKS, each row with its reason.
RELATION. tests/train/test_trainer_composition.py's
test_no_anchored_body_reads_trainer_state_through_a_fallback forbids every fallback on
``self``/``trainer`` in the anchored Cs2PuffeRL bodies and checkpoint helpers, with no
exceptions. This file scans every tracked .py file except the NOT_SCANNED ones, those
bodies included, with ALLOWED_FALLBACKS as its only exceptions;
test_every_tracked_python_file_is_scanned_or_named keeps the scope that wide.
KNOWN LIMITS.
  * An object neither net recognises passes: a parameter spelled ``tr``, read through a key
    that no STATE_CLASSES class or PuffeRL stores. The #355 census found no such spelling.
  * Aliases are ``name = <expr>``, ``name: T = <expr>`` and same-length tuple unpacking in
    an enclosing function; a trainer reached through a container, a loop target, a call's
    return value or a walrus is not followed.
  * Other fallback forms (``try: x.a`` / ``except AttributeError``, ``vars(x).get``) are not
    read. The #355 census found none under SCAN_ROOTS.
"""
from __future__ import annotations

import ast
import collections
import importlib.util
import subprocess
import textwrap
from pathlib import Path

from tests.conftest import REPO_ROOT

SCAN_ROOTS = ("src/cs2rl", "scripts", "deploy")
# Tracked .py files outside SCAN_ROOTS, by first path component, with the reason.
NOT_SCANNED = {
    "tests": ("its trainer hasattr calls (9 in the #355 census) are all `assert hasattr(...)`, "
              "which checks that an attribute exists rather than tolerating its absence"),
    "setup.py":
    "the build script; it builds no trainer or policy",
}
# Must be among the scanned files: it holds isolate_aim_log_std_param_group and _log_epoch,
# the reads #355 made direct, so a scan without it is not scanning the trainer's loop.
ANCHOR = "src/cs2rl/train/loop.py"
# Classes that own trainer/policy state, with the file that defines each. The key net reads
# their self stores; the receiver net treats their `self` as a trainer or policy.
STATE_CLASSES = {
    "Cs2PuffeRL": "src/cs2rl/train/trainer.py",
    "Dust2Policy": "src/cs2rl/policy_net.py",
}
# (file, qualname, key) -> why that fallback is legitimate. Each row must match exactly one
# live site (test_every_allowed_fallback_matches_exactly_one_site).
ALLOWED_FALLBACKS = {
    ("src/cs2rl/train/trainer.py", "Cs2PuffeRL.__init__", "utilization"):
    ("the except handler also runs when PuffeRL.__init__ raises, and PuffeRL's APIUsageError "
     "config checks raise before it creates Utilization"),
    ("scripts/trainer_equivalence.py", "snapshot", "msg"):
    ("PuffeRL sets msg only when it saves a checkpoint; the snapshot records it whether or "
     "not it is set"),
}


def _last_name(expr):
    """The last identifier of a Name or attribute chain, else None."""
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return expr.id if isinstance(expr, ast.Name) else None


def _is_state(expr, cls, aliases, seen=frozenset()):
    """True when the receiver net (module docstring) takes `expr` for a trainer or policy."""
    last = _last_name(expr)
    if last is None:
        return False
    if last == "trainer" or "polic" in last:
        return True
    if isinstance(expr, ast.Name):
        if expr.id == "self":
            return cls in STATE_CLASSES
        if expr.id in aliases and expr.id not in seen:
            return any(_is_state(v, cls, aliases, seen | {expr.id}) for v in aliases[expr.id])
    return False


def _aliases(fn):
    """{name: [assigned expr, ...]} for `name = expr`, `name: T = expr` and `a, b = x, y`
    anywhere in `fn`."""
    out = collections.defaultdict(list)
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign):
            targets, value = n.targets, n.value
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            targets, value = [n.target], n.value
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                out[t.id].append(value)
            elif (isinstance(t, ast.Tuple) and isinstance(value, ast.Tuple)
                  and len(t.elts) == len(value.elts)):
                for name, part in zip(t.elts, value.elts, strict=True):
                    if isinstance(name, ast.Name):
                        out[name.id].append(part)
    return out


def _owned_names():
    """Names STATE_CLASSES and stock PuffeRL store on self or register (the key net's set).

    Read from source, like the anchor walk in tests/train/test_trainer_composition.py: a
    class-level walk sees every branch, including the conditional ``aim_log_std_t`` /
    ``encoder_t`` modules. PuffeRL's file is located without importing pufferl (and torch).
    """
    spec = importlib.util.find_spec("pufferlib.pufferl")
    assert spec is not None and spec.origin, "pufferlib.pufferl is not importable"
    sources = [(REPO_ROOT / rel, name) for name, rel in STATE_CLASSES.items()]
    sources.append((Path(spec.origin), "PuffeRL"))
    names = set()
    for path, cls_name in sources:
        classes = [
            n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == cls_name
        ]
        assert len(classes) == 1, f"{path}: expected one class {cls_name}, found {len(classes)}"
        for n in ast.walk(classes[0]):
            if (isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                    and isinstance(n.value, ast.Name) and n.value.id == "self"):
                names.add(n.attr)
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr in ("register_buffer", "register_parameter") and n.args
                  and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)):
                names.add(n.args[0].value)
    return names


def _sites_in(tree, owned):
    """(qualname, key, line) of every trainer/policy fallback read in a parsed module.

    A non-constant key is reported as its source text.
    """
    hits = []

    def visit(node, qual, cls, aliases):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, qual + [child.name], child.name, {})
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                merged = collections.defaultdict(list, {k: list(v) for k, v in aliases.items()})
                for k, v in _aliases(child).items():
                    merged[k].extend(v)
                visit(child, qual + [child.name], cls, merged)
                continue
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                    and child.func.id in ("getattr", "hasattr") and len(child.args) >= 2
                    and not (child.func.id == "getattr" and len(child.args) == 2)):
                obj, key = child.args[0], child.args[1]
                const = (key.value
                         if isinstance(key, ast.Constant) and isinstance(key.value, str) else None)
                by_key = const in owned and _last_name(obj) != "args"
                if by_key or _is_state(obj, cls, aliases):
                    hits.append((".".join(qual) or "<module>",
                                 const if const is not None else ast.unparse(key), child.lineno))
            visit(child, qual, cls, aliases)

    visit(tree, [], None, {})
    return hits


def _scanned_files(root):
    """Every .py file under SCAN_ROOTS; red when a root is empty or ANCHOR is missing."""
    per_root = {r: sorted((root / r).rglob("*.py")) for r in SCAN_ROOTS}
    empty = sorted(r for r, found in per_root.items() if not found)
    assert not empty, (f"{empty} matched no *.py file, so the scan reads less than it claims; "
                       "repoint SCAN_ROOTS, never delete this assert")
    files = [p for found in per_root.values() for p in found]
    assert root / ANCHOR in files, f"{ANCHOR} is not among the {len(files)} scanned files"
    return files


def fallback_sites(root=REPO_ROOT):
    """(file, qualname, key, line) of every trainer/policy fallback read under SCAN_ROOTS."""
    owned = _owned_names()
    return [(p.relative_to(root).as_posix(), q, k, line) for p in _scanned_files(root)
            for q, k, line in _sites_in(ast.parse(p.read_text()), owned)]


def test_no_trainer_or_policy_state_is_read_through_a_fallback():
    hits = [
        f"{f}:{line} {q}: {k}" for f, q, k, line in fallback_sites()
        if (f, q, k) not in ALLOWED_FALLBACKS
    ]
    assert not hits, (
        f"trainer/policy state read through a getattr default or hasattr: {hits}. Declare the "
        "attribute where the object is built and read it directly; if it can really be absent "
        "(a stock attribute set late), add an ALLOWED_FALLBACKS row with the reason")


def test_every_tracked_python_file_is_scanned_or_named():
    """Scope check: a tracked .py file outside SCAN_ROOTS must be NOT_SCANNED by name.

    A dropped root or a new top-level directory of Python code would otherwise leave the
    scan reading less than the module docstring says, and stay green. A NOT_SCANNED entry
    that matches no tracked file is stale.
    """
    listing = subprocess.run(["git", "ls-files", "-z", "--", "*.py"],
                             cwd=REPO_ROOT,
                             capture_output=True,
                             check=True).stdout.decode()
    tracked = [f for f in listing.split("\0") if f]
    unnamed = [
        f for f in tracked
        if not f.startswith(tuple(r + "/"
                                  for r in SCAN_ROOTS)) and f.split("/")[0] not in NOT_SCANNED
    ]
    assert not unnamed, f"tracked .py files neither scanned nor NOT_SCANNED: {unnamed}"
    stale = sorted(set(NOT_SCANNED) - {f.split("/")[0] for f in tracked})
    assert not stale, f"NOT_SCANNED entries with no tracked .py file: {stale}"


def test_every_allowed_fallback_matches_exactly_one_site():
    """A row with no site is stale; a row matching two sites hides the second one's decision.

    With the test above, this makes the row count equal the number of legitimate fallback
    reads the scan finds.
    """
    counts = collections.Counter((f, q, k) for f, q, k, _line in fallback_sites())
    wrong = {row: counts[row] for row in ALLOWED_FALLBACKS if counts[row] != 1}
    assert not wrong, f"ALLOWED_FALLBACKS rows must each match one site; matches: {wrong}"


PLANT = textwrap.dedent('''\
    def isolate(trainer, args, run):
        a = getattr(trainer, "undeclared", None)        # hit: receiver net, undeclared key
        b = getattr(trainer, "scheduler")               # direct read
        t = trainer
        c = getattr(t, "via_alias", None)               # hit: alias of the trainer
        d = hasattr(trainer.policy, "via_policy")       # hit: policy through trainer.policy
        e = getattr(args, "aim_log_std_max", None)      # argparse namespace
        f = getattr(run.trainer, "via_run", 0)          # hit: trainer as an attribute

        def inner(past_policy):
            return getattr(past_policy, "nested", None)  # hit: nested def, policy receiver

        def odd(tr):  # hits: key net, unrecognised spelling, one key per owner source
            return (getattr(tr, "aim_dim_mask", None),    # a Dust2Policy registered buffer
                    getattr(tr, "tct_split_heads", None),  # a Dust2Policy self store
                    getattr(tr, "_tag_metrics", None),     # a Cs2PuffeRL self store
                    getattr(tr, "scheduler", None))        # a stock PuffeRL self store

        g = getattr(run.env, "unowned", None)           # unrelated object, unowned key
        tr2, _ = run.trainer, None
        h = getattr(tr2, "via_tuple", None)             # hit: tuple-unpacked alias
        return a, b, c, d, e, f, g, h, inner, odd


    class Cs2PuffeRL:
        def m(self):
            return hasattr(self, "late")                # hit: self in a state class


    class Other:
        def m(self):
            return hasattr(self, "late")                # self of another class
    ''')
PLANT_HITS = {("isolate", "undeclared"), ("isolate", "via_alias"), ("isolate", "via_policy"),
              ("isolate", "via_run"), ("isolate.inner", "nested"), ("isolate", "via_tuple"),
              ("isolate.odd", "aim_dim_mask"), ("isolate.odd", "tct_split_heads"),
              ("isolate.odd", "_tag_metrics"), ("isolate.odd", "scheduler"),
              ("Cs2PuffeRL.m", "late")}


def test_the_scan_flags_each_spelling_under_each_root(tmp_path):
    """Positive and negative controls for both nets, the walk and every root.

    One plant under each SCAN_ROOTS entry, scanned through fallback_sites itself, so the walk
    of every root is exercised end to end; a broken net loses its rows (the odd() rows need
    the key net's owned-name set, one key from each source it reads). A root dropped
    from SCAN_ROOTS is test_every_tracked_python_file_is_scanned_or_named's case.
    """
    for r in SCAN_ROOTS:
        (tmp_path / r).mkdir(parents=True)
        (tmp_path / r / "plant.py").write_text(PLANT)
    (tmp_path / ANCHOR).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / ANCHOR).write_text("")
    sites = fallback_sites(tmp_path)
    for r in SCAN_ROOTS:
        found = {(q, k) for f, q, k, _line in sites if f == f"{r}/plant.py"}
        assert found == PLANT_HITS, f"{r}: missed {PLANT_HITS - found}, extra {found - PLANT_HITS}"
