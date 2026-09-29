"""Every dotted `cs2rl.` name, cs2rl import line and `src/cs2rl/` path in a .py string resolves.

WHAT: one fail-closed scan over the string constants of every tracked .py file,
docstrings excluded. An f-string is read whole, each replacement field kept as
`{expr}`. Three clauses:
  (a) a `cs2rl.<dotted>` name names a module (importlib.util.find_spec), or a
      module plus an attribute, whether it is the whole string or sits inside a
      longer one (a `sys.modules` check, an `-m` launch, a message). A name inside
      an import line is left to (b). The attribute is looked up in the module
      file's top-level AST bindings, so no module is imported except the packages
      find_spec imports as parents.
  (b) every line inside a string that starts `from cs2rl... import` or
      `import cs2rl...` names a resolvable module, and each imported name is a
      submodule or a top-level binding of it. Every `;`-joined statement on the
      line that imports from cs2rl counts. Matched with a regex, never parsed:
      some child scripts are str.format templates that do not parse
      (tests/test_arena_duel.py's `{out!r}`). A `{field}` in the name list is
      skipped; the module is still checked.
  (c) every `src/cs2rl/...` path token is tracked: a file or a directory in
      `git ls-files`. A token that ends in `/` must be a directory. A token with a
      `{field}` in it must sit in a tracked directory.
A module counts as resolved only if find_spec finds it AND its file is tracked, so a
leftover untracked module, stale bytecode or a built extension (.so) never resolves
a name by accident.

EXEMPTIONS are keyed by (file, enclosing function, text), each with its reason. An
exemption whose occurrence no longer fails is a failure too, so the table cannot
rot into a blanket allowance.

WHY (#205 part 2a): a module move leaves these strings stale, and most go stale
SILENTLY. With tests/test_w1_modules.py's HEAVY still naming `cs2rl.nav` after the
move, the light-import guard for nav passed 21/21 with nav planted into
train_shared. Moved into a child script as `assert "cs2rl.nav" not in sys.modules`,
the same stale name passed the same way, which is why (a) reads names inside longer
strings too. A child script's stale import fails only when that child runs, and
test_fast_math_variant's runs only where zig is installed. A path token in a
message or a provenance label is read by no test at all. A pin per table covers
the tables someone thought of; this one covers every string's dotted names, import
lines and `src/cs2rl/` tokens.

LIMITS (the shapes it cannot see, each with its reason):
  - A bare module word (`_action_spec`, a label `eval_baselines=`) is invisible: a
    live alias (`from cs2rl.env import config as env_config`) reads exactly like a
    dead module word, so a bare-word clause could not tell a stale name from a live one.
  - A composed name or path is invisible: f"cs2rl.{x}", "cs2rl." + x, "%s", and
    `root / "src" / "cs2rl" / "env_config.py"`. Each piece is its own string and none
    holds the whole name. Spell module names and paths whole.
  - An import name after a backslash continuation is not read (only a parenthesised
    list is followed onto the next lines), and a `cs2rl/...` path without the `src/`
    prefix has no fixed root to look up in git ls-files. Neither occurs in a string today.
  - Only .py files are scanned: CONTRIBUTING.md, TOML, C comments and other non-.py
    text are not parsed for strings.
  - Comments and docstrings are prose, not checked here.
  - Clause (a) resolves at most a module attribute and, for a class, one member.
  - This file is not scanned: its EXEMPT keys, regexes and controls spell failing
    names on purpose. The controls at the bottom pin each clause instead.
"""
import ast
import importlib.util
import re
import subprocess
from functools import cache
from pathlib import Path

from tests.conftest import REPO_ROOT

# Exactly a dotted cs2rl name: clause (a).
_DOTTED = re.compile(r"cs2rl(?:\.[A-Za-z_]\w*)+")
# A cs2rl dotted name inside a longer string (a sys.modules check, an `-m` launch, a label):
# clause (a). The lookbehind skips `x.cs2rl.y` and `my_cs2rl.y`, which are not cs2rl names.
_EMBEDDED = re.compile(r"(?<![\w.])cs2rl(?:\.[A-Za-z_]\w*)+")
# An import line inside a string: clause (b), after any leading indent.
_IMPORT_LINE = re.compile(r"^[ \t]*(from|import)[ \t]+cs2rl\S*.*$", re.MULTILINE)
# A src/cs2rl/ path token: clause (c). Stops at whitespace, quotes and closing brackets,
# but keeps a `{field}` whole.
_PATH_TOKEN = re.compile(r"src/cs2rl/(?:\{[^{}\s]*\}|[^\s`'\",:;)\]{}])*")

# (file, enclosing function, text) -> why this occurrence may fail its clause.
# `text` is the offending text: the dotted name for (a), the import line (stripped)
# for (b), the path token for (c).
EXEMPT = {
    ("setup.py", "<module>", "cs2rl.c_env.binding"):
    "(a) the zig extension's module name. It has no Python source, and find_spec finds "
    "it only through the built, untracked .so; exempt by name so that #205 part 2b, "
    "which renames it, cannot pass on a leftover .so.",
    ("tests/test_fast_math_variant.py", "<module>", "cs2rl.c_env.binding"):
    "(a) the same extension name, which the CHECK child pre-seeds in sys.modules with a "
    "scratch build. PITFALL: this text always fails, so the row never goes stale by itself; "
    "#205 part 2b must rename this string and this key together.",
    ("src/cs2rl/play.py", "_load_play_lib", "src/cs2rl/c_env/zig-out/lib/libcs2_play.so"):
    "(c) a zig build output, never tracked; the loader skips a candidate that is absent.",
    ("src/cs2rl/play.py", "_load_play_lib", "src/cs2rl/c_env/zig-out/bin/libcs2_play.so"):
    "(c) a zig build output, never tracked; the loader skips a candidate that is absent.",
    ("src/cs2rl/play.py", "main", "src/cs2rl/c_env/zig-out/bin/resources"):
    "(c) the zig build's resource directory, never tracked.",
    ("tests/smoke_test.py", "test_c_env_smoke", "src/cs2rl/c_env/binding.cpython-*.so"):
    "(c) a glob for the built extension, never tracked.",
    ("tests/test_play_policy.py", "test_cs2_demo_policy_missing_exits_nonzero", "src/cs2rl/c_env/zig-out/bin/cs2_demo"):
    "(c) a zig build output, never tracked; the test skips when it is absent.",
    ("tests/test_play_policy.py", "test_cs2_demo_relative_venv_is_realpathd", "src/cs2rl/c_env/zig-out/bin/cs2_demo"):
    "(c) a zig build output, never tracked; the test skips when it is absent.",
    ("tests/test_import_layers.py", "_tracked_package_files", "src/cs2rl/*.py"):
    "(c) a git pathspec glob, not a path.",
    ("tests/test_import_layers.py", "test_control_a_directory_without_init_is_covered", "src/cs2rl/noinit/action.py"):
    "(c) control (f)'s plant path, created only in a tmp copy of the package.",
    ("tests/test_import_layers.py", "test_control_a_directory_without_init_is_covered", "cs2rl.noinit.action"):
    "(a) control (f)'s plant, a module that must not exist in this checkout.",
    ("tests/test_import_layers.py", "test_control_a_module_without_a_layer_is_rejected", "cs2rl.newmod"):
    "(a) control (b)'s plant, a module that must not exist in this checkout.",
    ("tests/test_import_layers.py", "test_control_a_module_without_a_layer_in_env_is_rejected", "cs2rl.env.newmod"):
    "(a) control (h)'s plant, a module that must not exist in this checkout.",
    ("tests/test_one_module_object_per_file.py", "test_the_guard_is_silent_on_files_it_must_not_report", "cs2rl.paths"):
    "(a) a synthetic module name in the tmp layout the test builds.",
    ("tests/test_one_module_object_per_file.py", "test_the_guard_reports_a_repo_file_under_two_names_with_every_name", "cs2rl.paths"):
    "(a) a synthetic module name in the tmp layout the test builds.",
    ("tests/test_env_construction_enforcement.py", "test_an_unimported_local_of_the_same_name_is_not_a_construction", "from cs2rl.train import make_puffer_env"):
    "(b) synthetic source for the construction scanner, parsed and never run.",
    ("tests/test_metrics_schema.py", "test_is_back_edge_sees_every_spelling", "from cs2rl.eval.baselinesx import y"):
    "(b) is_back_edge's negative control: a module that must not exist.",
}


def _git_ls_files(*pathspec: str) -> list[str]:
    r = subprocess.run(["git", "ls-files", "--", *pathspec],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True,
                       check=True)
    return r.stdout.split()


@cache
def _tracked() -> frozenset[str]:
    """Every tracked file, relative to REPO_ROOT, posix."""
    return frozenset(_git_ls_files())


@cache
def _tracked_dirs() -> frozenset[str]:
    return frozenset(p.as_posix() for f in _tracked() for p in Path(f).parents)


def _is_tracked(path: str | None) -> bool:
    if path is None:
        return False
    resolved = Path(path).resolve()
    return (resolved.is_relative_to(REPO_ROOT)
            and resolved.relative_to(REPO_ROOT).as_posix() in _tracked())


@cache
def _bindings(path: str) -> dict[str, ast.AST]:
    """Top-level names a module file binds, with the node that binds each.

    Counts def/class, assignment targets and import aliases, including those nested
    in a top-level if/try/with (a `try: import x` fallback binds x too).
    """
    names: dict[str, ast.AST] = {}

    def visit(stmts):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names[s.name] = s
            elif isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for target in (s.targets if isinstance(s, ast.Assign) else [s.target]):
                    for n in ast.walk(target):
                        if isinstance(n, ast.Name):
                            names[n.id] = s
            elif isinstance(s, (ast.Import, ast.ImportFrom)):
                for a in s.names:
                    names[(a.asname or a.name).split(".")[0]] = s
            elif isinstance(s, (ast.If, ast.Try, ast.With)):
                for block in (s.body, getattr(s, "orelse", []), getattr(s, "finalbody", [])):
                    visit(block)
                for handler in getattr(s, "handlers", []):
                    visit(handler.body)

    visit(ast.parse(Path(path).read_text(encoding="utf-8")).body)
    return names


def _attribute_problem(origin: str | None, module: str, attrs: list[str]) -> str | None:
    """None if `attrs` resolve in `origin`'s source: a top-level binding, then a class member."""
    if origin is None or not origin.endswith(".py"):
        return f"{module} has no Python source to look {attrs[0]!r} up in"
    node = _bindings(origin).get(attrs[0])
    if node is None:
        return f"{module} binds no top-level {attrs[0]!r}"
    if len(attrs) == 1:
        return None
    if len(attrs) == 2 and isinstance(node, ast.ClassDef):
        members = {
            n.name
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        members |= {
            n.id
            for s in node.body if isinstance(s, (ast.Assign, ast.AnnAssign))
            for t in (s.targets if isinstance(s, ast.Assign) else [s.target]) for n in ast.walk(t)
            if isinstance(n, ast.Name)
        }
        return None if attrs[1] in members else f"{module}.{attrs[0]} has no member {attrs[1]!r}"
    return f"{module}.{'.'.join(attrs)} is deeper than this pin resolves"


@cache
def dotted_problem(name: str) -> str | None:
    """None if `name` is a tracked cs2rl module, or one plus an attribute; else why not.

    Walks the name one segment at a time. find_spec imports a name's PARENTS, so it is
    only called while every parent so far is a package: the walk stops at the first
    plain module and resolves the remaining segments from its AST, never importing it.
    """
    parts = name.split(".")
    parent = None
    for i in range(1, len(parts) + 1):
        prefix = ".".join(parts[:i])
        try:
            spec = importlib.util.find_spec(prefix)
        except (ImportError, ValueError) as e:
            return f"find_spec({prefix!r}) raised {e!r}"
        if spec is None:
            if parent is None:
                return f"no module {prefix}"
            # Not a submodule: maybe an attribute of the package's __init__.
            problem = _attribute_problem(parent.origin, parent.name, parts[i - 1:])
            return None if problem is None else f"no module {prefix}, and {problem}"
        if not _is_tracked(spec.origin):
            return f"{prefix} resolves to {spec.origin}, which is not a tracked file"
        if spec.submodule_search_locations is None:
            rest = parts[i:]
            return _attribute_problem(spec.origin, prefix, rest) if rest else None
        parent = spec
    return None


def import_line_problem(line: str, continuation: str) -> str | None:
    """None if every cs2rl import statement on `line` resolves; else why not.

    `continuation` is the rest of the string after `line`, read only to finish a
    parenthesised name list. `line` starts with a cs2rl import; each later
    `;`-joined statement is checked too if it is an `import` or a `from cs2rl`.
    PITFALL: failures_in hands the WHOLE line to this clause and (a) skips it, so a
    second statement checked nowhere else (`from cs2rl.env import nav; import
    cs2rl.nav`) would be invisible.
    """
    first, *rest = (s.strip() for s in line.split(";"))
    for statement in [first] + [s for s in rest if re.match(r"import\s|from\s+cs2rl\b", s)]:
        problem = _statement_problem(statement, continuation)
        if problem:
            return problem
    return None


def _statement_problem(statement: str, continuation: str) -> str | None:
    """None if one `import ...`/`from ... import ...` statement resolves; else why not.

    An `import` statement's non-cs2rl items (`import sys, time`) are skipped.
    """
    if statement.startswith("import "):
        for item in statement[len("import "):].split(","):
            module = item.split(" as ")[0].strip()
            problem = dotted_problem(module) if module.startswith("cs2rl") else None
            if problem:
                return problem
        return None
    m = re.fullmatch(r"from\s+(\S+)\s+import\s+(.*)", statement)
    if m is None:
        return f"not an import statement: {statement!r}"
    module, names = m.groups()
    if names.startswith("(") and ")" not in names:
        names += " " + continuation.split(")")[0]
    problem = dotted_problem(module)
    if problem:
        return problem
    for item in names.strip("()\\ ").split(","):
        name = item.split(" as ")[0].strip().strip("()\\")
        if name and name != "*" and "{" not in name and dotted_problem(f"{module}.{name}"):
            return f"{module} has no submodule or top-level binding {name!r}"
    return None


def path_problem(token: str) -> str | None:
    """None if `token` (a src/cs2rl/ path) is a tracked file or directory; else why not."""
    token = token.rstrip(".")
    if "{" in token:
        # A replacement field: only the directory it sits in is known.
        token = token[:token.index("{")].rpartition("/")[0] + "/"
    directory = token.endswith("/")
    token = token.rstrip("/")
    if token in _tracked_dirs():
        return None
    if not directory and token in _tracked():
        return None
    return f"{token!r} is not a tracked {'directory' if directory else 'file or directory'}"


def _docstring_nodes(tree: ast.AST) -> set[int]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.body and isinstance(n.body[0], ast.Expr) and isinstance(
                    n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
                out.add(id(n.body[0].value))
    return out


def _joined_text(node: ast.JoinedStr) -> str:
    """An f-string's text with each replacement field kept as `{expr}`."""
    parts = []
    for v in node.values:
        if isinstance(v, ast.FormattedValue):
            parts.append("{" + ast.unparse(v.value) + "}")
        elif isinstance(v, ast.Constant) and isinstance(v.value, str):
            parts.append(v.value)
    return "".join(parts)


def string_occurrences(source: str):
    """Yield (enclosing function, line, string) for each non-docstring string.

    A str constant is yielded as it is; an f-string once, as _joined_text, and never
    its literal parts on their own (a part would cut an import line short). The
    enclosing function is the dotted def/class chain, or "<module>".
    """
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)

    def walk(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = child.name if scope == "<module>" else f"{scope}.{child.name}"
                yield from walk(child, inner)
            elif isinstance(child, ast.JoinedStr):
                yield scope, child.lineno, _joined_text(child)
                for v in child.values:                                             # expressions inside the fields may hold strings
                    if isinstance(v, ast.FormattedValue):
                        yield from walk(v, scope)
            elif (isinstance(child, ast.Constant) and isinstance(child.value, str)
                  and id(child) not in docstrings):
                yield scope, child.lineno, child.value
            else:
                yield from walk(child, scope)

    yield from walk(tree, "<module>")


def failures_in(relative: str, source: str):
    """Yield (key, clause, message) for every occurrence in `source` that fails a clause."""
    for scope, line, value in string_occurrences(source):
        found = []
        imports = list(_IMPORT_LINE.finditer(value))
        if _DOTTED.fullmatch(value):
            found.append(("a", value, dotted_problem(value)))
        else:
            # A name inside an import line is (b)'s: (a) would report it a second time,
            # under a second EXEMPT key.
            for m in _EMBEDDED.finditer(value):
                if not any(i.start() <= m.start() < i.end() for i in imports):
                    found.append(("a", m.group(0), dotted_problem(m.group(0))))
        for m in imports:
            found.append(("b", m.group(0).strip(), import_line_problem(m.group(0),
                                                                       value[m.end():])))
        for token in _PATH_TOKEN.findall(value):
            found.append(("c", token, path_problem(token)))
        for clause, text, problem in found:
            if problem:
                yield ((relative, scope, text), clause,
                       f"{relative}:{line} in {scope}: ({clause}) {text!r}: {problem}")


def scan(files: list[str], exempt: dict) -> list[str]:
    """Every failing occurrence not in `exempt`, then every exemption that matched nothing."""
    report, used = [], set()
    for relative in files:
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for key, _, message in failures_in(relative, source):
            if key in exempt:
                used.add(key)
            else:
                report.append(message)
    report += [
        f"STALE EXEMPTION {key}: no occurrence fails any more; delete the entry" for key in exempt
        if key not in used
    ]
    return report


# This file, relative to REPO_ROOT: the one file the main scan skips (see LIMITS).
_THIS_FILE = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()


def test_every_cs2rl_name_in_a_string_resolves():
    """The pin itself: every tracked .py file but this one, against EXEMPT."""
    tracked = _git_ls_files("*.py")
    files = [f for f in tracked if f != _THIS_FILE]
    # Only this file may drop out. A wrong _THIS_FILE either skips nothing (the count check
    # fails) or skips another file (this file is then scanned, and its planted names fail).
    assert len(tracked) - len(files) == 1, f"{_THIS_FILE} is not a tracked .py file"
    assert len(files) > 100, f"git ls-files found only {len(files)} .py files"
    report = scan(files, EXEMPT)
    assert not report, (
        "A string names a cs2rl module, import or src/cs2rl/ path that does not resolve. A "
        "module move leaves these stale, and most would fail silently; fix the string, or "
        "add an EXEMPT entry with its reason:\n  " + "\n  ".join(report))


# The controls below prove each clause can fail, on synthetic module sources: a
# control that passed its bad names straight to failures_in would test only the
# regexes, not the string walk (docstrings skipped, f-strings read whole).


def _failing(source: str) -> list[tuple[str, str]]:
    """(clause, text) of every failing occurrence in a synthetic module `source`."""
    return [(clause, key[2]) for key, clause, _ in failures_in("synthetic.py", source)]


def test_clause_a_fails_a_moved_name_a_deleted_member_and_an_untracked_module():
    """(a) fails a module name #205 moved, a member it deleted, and a name that
    resolves only to an untracked file, whole or inside a longer string (a guard's
    `sys.modules` check, an `-m` launch); it passes the new names, a word that only
    ends in `cs2rl`, and skips docstrings. That (a) leaves import lines to (b) is
    pinned by the single (b) entries in the next test.

    PITFALL: cs2rl.c_env.binding fails as untracked where the zig .so is built and
    as missing where it is not; either way it must fail, or a leftover .so could
    resolve a stale name. The embedded negative check is the case that matters most:
    `assert "cs2rl.nav" not in sys.modules` is vacuously true once nav moves.
    """
    assert _failing('X = "cs2rl.nav"\n') == [("a", "cs2rl.nav")]
    assert _failing('X = "cs2rl.env.nav.NavGraph.can_see"\n') == [
        ("a", "cs2rl.env.nav.NavGraph.can_see")
    ]
    assert _failing('X = "cs2rl.c_env.binding"\n') == [("a", "cs2rl.c_env.binding")]
    assert _failing('X = "cs2rl.env.nav"\nY = "cs2rl.env.nav.NavGraph.path"\n'
                    'Z = "cs2rl.spec.action.ACTION_DIM"\n') == []
    assert _failing('"""cs2rl.nav"""\ndef f():\n    "cs2rl.nav"\n') == []
    assert _failing("S = 'assert \"cs2rl.nav\" not in sys.modules'\n") == [("a", "cs2rl.nav")]
    assert _failing("S = 'python -m cs2rl.bc_demo --out x'\n") == [("a", "cs2rl.bc_demo")]
    assert _failing("S = 'assert \"cs2rl.env.nav\" not in sys.modules'\n"
                    "T = 'python -m cs2rl.bc_demos'\nU = 'build/libcs2rl.so'\n") == []


def test_clause_b_fails_a_stale_import_in_every_spelling():
    """(b) fails a stale module or name in a `from`/`import` line, indented or not,
    including a name inside a parenthesised list, the module of an f-string template
    and a later `;`-joined statement; it passes the moved spellings, skips a `{field}`
    name and a later statement that is not a cs2rl import.

    PITFALL: an f-string is read whole. Read as literal parts, "from cs2rl.nav
    import " would end at the field, and the pin would check no name at all.
    """
    assert _failing("S = 'from cs2rl import nav'\n") == [("b", "from cs2rl import nav")]
    assert _failing("S = '    from cs2rl.nav import CACHE_PATH'\n") == [
        ("b", "from cs2rl.nav import CACHE_PATH")
    ]
    assert _failing("S = 'import cs2rl.nav as n'\n") == [("b", "import cs2rl.nav as n")]
    assert _failing("S = 'x = 1\\nfrom cs2rl.env.nav import (\\n    CACHE_PATH,\\n"
                    "    NOPE,\\n)'\n") == [("b", "from cs2rl.env.nav import (")]
    assert _failing("S = f'from cs2rl.nav import {name}'\n") == [("b",
                                                                  "from cs2rl.nav import {name}")]
    assert _failing("S = 'from cs2rl.env import nav; import cs2rl.nav'\n") == [
        ("b", "from cs2rl.env import nav; import cs2rl.nav")
    ]
    assert _failing("S = 'from cs2rl.env import nav, map as m'\n"
                    "T = f'from cs2rl.env.nav import {name}'\n"
                    "U = 'from cs2rl.env import nav; nav.f(); import time'\n") == []


def test_clause_c_fails_an_untracked_path():
    """(c) fails a path #205 moved, a file spelled with a trailing `/`, and a
    replacement field in an untracked directory; it passes a tracked file, a tracked
    directory, a sentence-final period and a field in a tracked directory.
    """
    assert _failing("P = 'src/cs2rl/nav.py'\n") == [("c", "src/cs2rl/nav.py")]
    assert _failing("P = 'src/cs2rl/env/nav.py/'\n") == [("c", "src/cs2rl/env/nav.py/")]
    assert _failing("P = f'src/cs2rl/nope/{name}.py'\n") == [("c", "src/cs2rl/nope/{name}.py")]
    assert _failing("P = 'see src/cs2rl/env/nav.py.'\nQ = 'src/cs2rl/env/'\n"
                    "R = f'src/cs2rl/env/{name}.py'\n") == []


def test_an_exemption_hides_only_its_occurrence_and_is_reported_once_stale():
    """scan() suppresses an exempted occurrence and reports an exemption that matched
    nothing, so a fixed string forces its EXEMPT row out.

    PITFALL: it runs on the real setup.py, whose one failing string is the EXEMPT
    row it borrows. If that string is ever fixed, this test must change with it.
    """
    key = next(k for k in EXEMPT if k[0] == "setup.py")
    unexempted = scan(["setup.py"], {})
    assert len(unexempted) == 1 and unexempted[0].startswith("setup.py:"), unexempted
    assert scan(["setup.py"], {key: "reason"}) == []
    stale = ("setup.py", "<module>", "not-a-failing-text")
    assert scan(["setup.py"], {
        key: "reason",
        stale: "reason"
    }) == [f"STALE EXEMPTION {stale}: no occurrence fails any more; delete the entry"]
