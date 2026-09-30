"""Every dotted `cs2rl.` name, cs2rl import line and `src/cs2rl/` path in a string resolves.

WHAT: one fail-closed scan over the string constants of every tracked .py file,
docstrings excluded, and over the string literals of every tracked .c and .h file,
comments excluded. An f-string is read whole, each replacement field kept as
`{expr}`. Three clauses (a C string gets (a) and (c) only; it holds no import line):
  (a) a `cs2rl.<dotted>` name names a module (importlib.util.find_spec), or a
      module plus an attribute, whether it is the whole string or sits inside a
      longer one (a `sys.modules` check, an `-m` launch, a message). A name inside
      a cs2rl import statement is left to (b); the rest of that line is still (a)'s.
      The attribute is looked up in the module
      file's top-level AST bindings, so no module is imported except the packages
      find_spec imports as parents.
  (b) every line inside a string that starts `from cs2rl... import` or
      `import cs2rl...` names a resolvable module, and each imported name is a
      submodule or a top-level binding of it. Every `;`-joined statement on the
      line that imports from cs2rl counts. Matched with a regex, never parsed:
      some child scripts are str.format templates that do not parse
      (tests/env/test_arena_duel.py's `{out!r}`). A `{field}` in the name list is
      skipped; the module is still checked.
  (c) every `src/cs2rl/...` path token is tracked: a file or a directory in
      `git ls-files`. A token that ends in `/` must be a directory. A token with a
      `{field}` in it must sit in a tracked directory.
A module counts as resolved only if find_spec finds it AND its file is tracked, so a
leftover untracked module or stale bytecode never resolves a name by accident. The
one kind of module with no tracked file of its own is an extension (the zig-built
`binding`), and it resolves by its source instead: `P.x` names an extension module
if P is a tracked package and a tracked .c file in P's directory defines
`PyMODINIT_FUNC PyInit_x` (CPython takes the init symbol from the last name
component). The rule never looks at a built .so, so a fresh clone (no .so: find_spec
returns None, path B) and a built tree (find_spec returns the untracked .so, path A)
judge a name the same way, and a leftover .so of a moved package resolves nothing.

EXEMPTIONS are keyed by (file, enclosing function, text), each with its reason; a C
occurrence's enclosing function is "<c>". An exemption whose occurrence no longer
fails is a failure too, so the table cannot rot into a blanket allowance. So is a
clause-(c) exemption (a `src/cs2rl/...` text) whose deepest tracked ancestor is a
subpackage rather than src/cs2rl itself: such a path belongs to a package that can
move, so derive it from the package instead of exempting it (the masking class
below).

WHY (#205 part 2a): a module move leaves these strings stale, and most go stale
SILENTLY. With tests/train/test_w1_modules.py's HEAVY still naming `cs2rl.nav` after the
move, the light-import guard for nav passed 21/21 with nav planted into
train_shared. Moved into a child script as `assert "cs2rl.nav" not in sys.modules`,
the same stale name passed the same way, which is why (a) reads names inside longer
strings too. A child script's stale import fails only when that child runs, and
test_fast_math_variant's runs only where zig is installed. A path token in a
message or a provenance label is read by no test at all. A pin per table covers
the tables someone thought of; this one covers every string's dotted names, import
lines and `src/cs2rl/` tokens.

WHY C strings too (#205 part 2b): cs2_demo.c finds its checkout by a marker path
and launches `-m cs2rl.<module>`, and nothing that runs here builds it, so a stale
marker or module name in it passed every test.

WHY the masking check (#205 part 2b, measured before its move of c_env): an
EXEMPT row matches on its exact text, and a stale text is still spelled the same,
so the row kept matching. The stale string then failed exactly as its exemption
expected, and the pin hid it: 5 rows (setup.py's extension name,
test_fast_math_variant's, smoke_test's .so glob and test_play_policy's two cs2_demo
paths) masked stale text through the whole move, with the STALE EXEMPTION report
silent because each row still matched. All 6 distinct exempted texts sat under
`src/cs2rl/c_env`. So an exempted path under a subpackage now fails outright: a
path that moves with a package is derived from it (cs2rl.env.c.SOURCE_DIR,
ZIG_OUT), never exempted.

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
  - Only .py and C (.c/.h) files are scanned: C#, .zig, CONTRIBUTING.md, TOML and
    other text are not parsed for strings. In C, adjacent-literal concatenation
    (`"src/cs2rl/" "viz/play.py"`) is read as two strings, neither holding the whole
    path, and a numeric escape (`\\x41`) is not decoded. clang-format's
    BreakStringLiterals is on (ColumnLimit 100), but it splits a literal only at
    whitespace, and a name or path token has none, so it never splits one.
  - Comments and docstrings are prose, not checked here. The C scanner reads a
    preprocessor line as code (a `#define` body's strings count) and does not
    splice a `//` comment continued by a trailing backslash.
  - Clause (a) resolves at most a module attribute and, for a class, one member.
  - The masking check reads clause-(c) rows only, told apart by their
    `src/cs2rl/` text. A clause-(a)/(b) row is not covered, deliberately: control
    (h)'s `cs2rl.env.newmod` names a real subpackage on purpose (a module that
    must not exist there).
  - This file is not scanned: its EXEMPT keys, regexes and controls spell failing
    names on purpose. The controls at the bottom pin each clause instead.
"""
import ast
import importlib.machinery
import importlib.util
import re
import subprocess
import sysconfig
from functools import cache
from pathlib import Path

import pytest

from cs2rl.env import c as c_env
from tests._helpers.c_strings import c_string_literals
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
# for (b), the path token for (c). A string in a .c or .h file has "<c>" as its
# enclosing function. A (c) row under a subpackage fails (see scan()).
EXEMPT = {
    ("tests/integration/test_import_layers.py", "_tracked_package_files", "src/cs2rl/*.py"):
    "(c) a git pathspec glob, not a path.",
    ("tests/integration/test_import_layers.py", "test_control_a_directory_without_init_is_covered", "src/cs2rl/noinit/action.py"):
    "(c) control (f)'s plant path, created only in a tmp copy of the package.",
    ("tests/integration/test_import_layers.py", "test_control_a_directory_without_init_is_covered", "cs2rl.noinit.action"):
    "(a) control (f)'s plant, a module that must not exist in this checkout.",
    ("tests/integration/test_import_layers.py", "test_control_a_module_without_a_layer_is_rejected", "cs2rl.newmod"):
    "(a) control (b)'s plant, a module that must not exist in this checkout.",
    ("tests/integration/test_import_layers.py", "test_control_a_module_without_a_layer_in_env_is_rejected", "cs2rl.env.newmod"):
    "(a) control (h)'s plant, a module that must not exist in this checkout.",
    ("tests/integration/test_one_module_object_per_file.py", "test_the_guard_is_silent_on_files_it_must_not_report", "cs2rl.paths"):
    "(a) a synthetic module name in the tmp layout the test builds.",
    ("tests/integration/test_one_module_object_per_file.py", "test_the_guard_reports_a_repo_file_under_two_names_with_every_name", "cs2rl.paths"):
    "(a) a synthetic module name in the tmp layout the test builds.",
    ("tests/integration/test_env_construction_enforcement.py", "test_an_unimported_local_of_the_same_name_is_not_a_construction", "from cs2rl.train import make_puffer_env"):
    "(b) synthetic source for the construction scanner, parsed and never run.",
    ("tests/eval/test_metrics_schema.py", "test_is_back_edge_sees_every_spelling", "from cs2rl.eval.baselinesx import y"):
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


def _extension_source(package, leaf: str) -> str | None:
    """The tracked .c file in `package`'s directory that defines `PyInit_<leaf>`, or None.

    `package` is the find_spec of a tracked regular package. This is how an extension
    module (built, never tracked) resolves: by the source it is built from. CPython
    takes the init symbol from the LAST name component (Python/importdl.c; PEP 489's
    export hook name), so `cs2rl.env.c.binding` needs `PyInit_binding`.
    PITFALL: the symbol is required, not just a `<leaf>.c` file: cs2_demo.c and
    cs2_play_host.c sit in the same directory, and a file-name rule would "resolve"
    `cs2rl.env.c.cs2_demo`, which no import can load. The definition is matched as
    `PyMODINIT_FUNC PyInit_<leaf>(`, the macro every CPython init function is declared
    with; an init spelled without it reads as no extension, which fails loudly.
    """
    directory = Path(package.origin).resolve().parent
    definition = re.compile(r"\bPyMODINIT_FUNC\s+PyInit_" + re.escape(leaf) + r"\s*\(")
    for relative in sorted(_tracked()):
        path = REPO_ROOT / relative
        if path.suffix == ".c" and path.parent.resolve() == directory:
            if definition.search(path.read_text(encoding="utf-8", errors="replace")):
                return relative
    return None


@cache
def dotted_problem(name: str) -> str | None:
    """None if `name` is a tracked cs2rl module, or one plus an attribute; else why not.

    Walks the name one segment at a time. find_spec imports a name's PARENTS, so it is
    only called while every parent so far is a package: the walk stops at the first
    plain module and resolves the remaining segments from its AST, never importing it.
    The last segment may also be an extension module, judged by its tracked source
    (_extension_source) and never by a built .so, through two paths: (A) find_spec
    returns an untracked extension file (a built tree) and (B) find_spec returns None
    (a fresh clone, no .so).
    PITFALL: @cache. A test that patches _tracked or find_spec must clear this cache
    before and after (see fresh_resolver_caches), or it reads an answer cached by an
    earlier test and never reaches its patch.
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
            # Path B: no built .so, but the leaf is an extension its package builds.
            if i == len(parts) and _extension_source(parent, parts[-1]) is not None:
                return None
            # Not a submodule: maybe an attribute of the package's __init__.
            problem = _attribute_problem(parent.origin, parent.name, parts[i - 1:])
            return None if problem is None else f"no module {prefix}, and {problem}"
        if not _is_tracked(spec.origin):
            # Path A: the built, untracked .so of an extension its package builds.
            if (i == len(parts) and parent is not None and parent.origin and spec.origin
                    and spec.origin.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
                    and Path(spec.origin).resolve().parent == Path(parent.origin).resolve().parent
                    and _extension_source(parent, parts[-1]) is not None):
                return None
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
    PITFALL: (a) skips exactly the statements this clause reads (_import_statements),
    so a later import statement checked only by the first (`from cs2rl.env import
    nav; import cs2rl.nav`) would be read by nothing.
    """
    for _, statement in _import_statements(line):
        problem = _statement_problem(statement.strip(), continuation)
        if problem:
            return problem
    return None


def _import_statements(line: str) -> list[tuple[int, str]]:
    """(offset in `line`, text) of each `;`-joined statement that (b) checks on `line`.

    The first is the cs2rl import the line starts with. A later one counts if it is
    an `import` or a `from cs2rl`. failures_in leaves exactly these spans to (b), so
    a later statement that is not an import (`from cs2rl.env import nav; assert
    "cs2rl.nav" not in sys.modules`) is still read by (a). Skipping the whole line
    hid that name from both clauses. The offsets come from re.finditer, not a running
    sum: an off-by-one there shifts each later span left by one per `;`, which no
    realistic line would reveal.
    """
    return [(m.start(), m.group()) for i, m in enumerate(re.finditer(r"[^;]+", line))
            if i == 0 or re.match(r"import\s|from\s+cs2rl\b",
                                  m.group().strip())]


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


def _occurrences(value: str, imports: bool = True) -> list[tuple[str, str, str | None]]:
    """(clause, text, problem) for every (a), (b) and (c) occurrence in one string value.

    `problem` is None where the occurrence resolves. `imports=False` skips clause (b),
    for C strings, which hold no Python import line.
    """
    found = []
    lines = list(_IMPORT_LINE.finditer(value)) if imports else []
    # The spans of the import statements (b) reads: a name inside one is (b)'s, and (a)
    # would report it a second time, under a second EXEMPT key. Only those statements,
    # not their whole line: see _import_statements.
    spans = [(i.start() + offset, i.start() + offset + len(part)) for i in lines
             for offset, part in _import_statements(i.group(0))]
    if _DOTTED.fullmatch(value):
        found.append(("a", value, dotted_problem(value)))
    else:
        for m in _EMBEDDED.finditer(value):
            if not any(start <= m.start() < end for start, end in spans):
                found.append(("a", m.group(0), dotted_problem(m.group(0))))
    for m in lines:
        found.append(("b", m.group(0).strip(), import_line_problem(m.group(0), value[m.end():])))
    for token in _PATH_TOKEN.findall(value):
        found.append(("c", token, path_problem(token)))
    return found


def failures_in(relative: str, source: str):
    """Yield (key, clause, message) for every occurrence in `source` that fails a clause."""
    for scope, line, value in string_occurrences(source):
        for clause, text, problem in _occurrences(value):
            if problem:
                yield ((relative, scope, text), clause,
                       f"{relative}:{line} in {scope}: ({clause}) {text!r}: {problem}")


def c_occurrences(relative: str, source: str):
    """Yield (line, clause, text, problem) for every (a) and (c) occurrence in a C file."""
    for line, value in c_string_literals(source):
        for clause, text, problem in _occurrences(value, imports=False):
            yield line, clause, text, problem


def c_failures_in(relative: str, source: str):
    """Yield (key, clause, message) for every C occurrence that fails; the scope is "<c>"."""
    for line, clause, text, problem in c_occurrences(relative, source):
        if problem:
            yield ((relative, "<c>", text), clause,
                   f"{relative}:{line} in <c>: ({clause}) {text!r}: {problem}")


def _is_c(relative: str) -> bool:
    return Path(relative).suffix in (".c", ".h")


def masking_row_problem(key: tuple[str, str, str]) -> str | None:
    """Why an EXEMPT row may not exist: a (c) text whose deepest tracked ancestor is a subpackage.

    A (c) row is told apart by its `src/cs2rl/` text, since keys carry no clause.
    Its deepest tracked ancestor is the longest parent path that `git ls-files`
    holds as a directory. Under src/cs2rl itself (a git pathspec, a plant in a
    directory that must not exist) the row may stay; under a subpackage it is a
    path that moves with that package, which the row would hide once stale.
    """
    text = key[2]
    if not text.startswith("src/cs2rl/"):
        return None
    ancestor = next((p.as_posix() for p in Path(text).parents if p.as_posix() in _tracked_dirs()),
                    None)
    if ancestor is None or ancestor == "src/cs2rl":
        return None
    return (f"MASKING EXEMPTION {key}: its path sits in the package {ancestor}, and an "
            "exempted path keeps matching after that package moves. Derive this path from its "
            "package instead of exempting it (cs2rl.env.c.SOURCE_DIR / ZIG_OUT)")


def scan(files: list[str],
         exempt: dict,
         root: Path = REPO_ROOT,
         read: list[str] | None = None) -> list[str]:
    """Every failing occurrence not in `exempt`, then every exemption that matched nothing,
    then every exemption that masks a subpackage path.

    `files` are relative to `root`: .py files are read by string_occurrences, .c and
    .h files by c_string_literals. `root` is REPO_ROOT except in the controls. If
    `read` is given, every file read is appended to it, so the main pin counts the
    C files this scan read, not the list it was meant to pass.
    """
    report, used = [], set()
    for relative in files:
        source = (root / relative).read_text(encoding="utf-8")
        if read is not None:
            read.append(relative)
        reader = c_failures_in if _is_c(relative) else failures_in
        for key, _, message in reader(relative, source):
            if key in exempt:
                used.add(key)
            else:
                report.append(message)
    report += [
        f"STALE EXEMPTION {key}: no occurrence fails any more; delete the entry" for key in exempt
        if key not in used
    ]
    report += [problem for problem in map(masking_row_problem, exempt) if problem]
    return report


# This file, relative to REPO_ROOT: the one file the main scan skips (see LIMITS).
_THIS_FILE = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()


def _c_files() -> list[str]:
    """Every tracked .c and .h file: TWO pathspecs. One (`'*.c *.h'`) matches nothing."""
    return _git_ls_files("*.c", "*.h")


def test_every_cs2rl_name_in_a_string_resolves():
    """The pin itself: every tracked .py file but this one, and every tracked C file."""
    tracked = _git_ls_files("*.py")
    files = [f for f in tracked if f != _THIS_FILE]
    # Only this file may drop out. A wrong _THIS_FILE either skips nothing (the count check
    # fails) or skips another file (this file is then scanned, and its planted names fail).
    assert len(tracked) - len(files) == 1, f"{_THIS_FILE} is not a tracked .py file"
    assert len(files) > 100, f"git ls-files found only {len(files)} .py files"
    # The C files the scan READ, counted against the tracked set by suffix (independent of
    # the pathspec) and against a floor: a pathspec bug, or a scan call that drops the C
    # files, scans no C file, and every synthetic C control below would still pass.
    read: list[str] = []
    report = scan(files + _c_files(), EXEMPT, read=read)
    c_read = [f for f in read if _is_c(f)]
    assert len(c_read) == sum(1 for f in _tracked() if _is_c(f)), c_read
    assert len(c_read) >= 20, f"the scan read only {len(c_read)} .c/.h files"
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


# Every cache the resolver reads, as the cached functions themselves: a control that
# replaces _tracked still clears the real cache through this tuple. A hand-kept list:
# test_the_cache_list_names_every_cache_this_module_defines watches it.
_RESOLVER_CACHES = (dotted_problem, _tracked, _tracked_dirs, _bindings)


def _clear_resolver_caches() -> None:
    for cached in _RESOLVER_CACHES:
        cached.cache_clear()


def _caches_defined_in(namespace: dict) -> set:
    """The cached callables (`@cache`, `@lru_cache`) in `namespace` that THIS module defines.

    Keyed on `cache_clear`, which every functools cache wrapper carries, and on
    `__module__`, which the wrapper copies from the function it wraps. So a cached
    callable imported from elsewhere is not this module's cache to clear, and does not
    count.
    """
    return {
        v
        for v in namespace.values()
        if callable(getattr(v, "cache_clear", None)) and getattr(v, "__module__", None) == __name__
    }


def test_the_cache_list_names_every_cache_this_module_defines():
    """_RESOLVER_CACHES is exactly the set of caches this module defines.

    WHY: fresh_resolver_caches clears only what the tuple lists. A fifth `@cache`
    helper added later and left out would bring back the test-order bug that fixture
    exists for (a control reads a cached answer and never reaches its patch), with
    every test green: the guard's own list, unwatched. Patches nothing, so it reads
    the real module globals.
    The second assertion is the filter's control: a cached callable defined elsewhere
    (here `len`, wrapped) is not counted, so importing one can neither turn this red
    nor stand in for a missing entry.
    """
    defined, listed = _caches_defined_in(globals()), set(_RESOLVER_CACHES)
    assert defined == listed, (
        "_RESOLVER_CACHES must list every cache this module defines. Unlisted: "
        f"{sorted(f.__name__ for f in defined - listed)}; listed but not a cache defined "
        f"here: {sorted(f.__name__ for f in listed - defined)}")
    assert _caches_defined_in({"imported": cache(len)}) == set()


@pytest.fixture
def fresh_resolver_caches():
    """Clear the resolver's caches before and after a test that patches _tracked or find_spec.

    WHY: dotted_problem is @cache'd, and the main pin runs first in file order and
    caches the real answer for every name in the tree, cs2rl.env.c.binding among them.
    A control that patches _tracked or find_spec would read that cached answer and
    never reach its patch: measured (#205 part 2b), path B's control passed with path
    B's rule deleted once the main pin had run, and failed on an empty cache. So the
    cache is cleared here, never left to test order.
    """
    _clear_resolver_caches()
    yield
    _clear_resolver_caches()


def test_clause_a_fails_a_moved_name_a_deleted_member_and_an_untracked_module(
        monkeypatch, fresh_resolver_caches):
    """(a) fails a module name #205 moved, a member it deleted, and a name that
    resolves only to an untracked file, whole or inside a longer string (a guard's
    `sys.modules` check, an `-m` launch); it passes the new names, a word that only
    ends in `cs2rl`, and skips docstrings. That (a) leaves import statements to (b)
    is pinned by the single (b) entries in the next test; that it still reads the
    rest of their line, by the last case here.

    PITFALL: a leftover untracked file must never resolve a name, or a stale name
    would pass on it. Here env/nav.py is dropped from _tracked, so find_spec still
    finds the file on disk and only the tracked check can fail it. An extension
    module (a .so, never tracked) resolves by its tracked source instead; that it
    fails without one, with or without a built .so, is the extension controls'
    case below. The embedded negative check is the case that matters most:
    `assert "cs2rl.nav" not in sys.modules` is vacuously true once nav moves.
    """
    assert _failing('X = "cs2rl.nav"\n') == [("a", "cs2rl.nav")]
    assert _failing('X = "cs2rl.env.nav.NavGraph.can_see"\n') == [
        ("a", "cs2rl.env.nav.NavGraph.can_see")
    ]
    assert _failing('X = "cs2rl.env.nav"\nY = "cs2rl.env.nav.NavGraph.path"\n'
                    'Z = "cs2rl.spec.action.ACTION_DIM"\n') == []
    assert _failing('"""cs2rl.nav"""\ndef f():\n    "cs2rl.nav"\n') == []
    assert _failing("S = 'assert \"cs2rl.nav\" not in sys.modules'\n") == [("a", "cs2rl.nav")]
    assert _failing("S = 'python -m cs2rl.bc_demo --out x'\n") == [("a", "cs2rl.bc_demo")]
    assert _failing("S = 'assert \"cs2rl.env.nav\" not in sys.modules'\n"
                    "T = 'python -m cs2rl.bc_demos'\nU = 'build/libcs2rl.so'\n") == []
    # After a cs2rl import on the same line: (b) reads only the import, so (a) must
    # read the rest.
    assert _failing(
        "S = 'from cs2rl.env import nav; assert \"cs2rl.nav\" not in sys.modules'\n") == [
            ("a", "cs2rl.nav")
        ]
    # An untracked module file: last, since it patches _tracked (the fixture clears the cache).
    real = _tracked()
    assert "src/cs2rl/env/nav.py" in real, "the control's module is gone: pick another"
    _clear_resolver_caches()
    monkeypatch.setitem(globals(), "_tracked", lambda: real - {"src/cs2rl/env/nav.py"})
    assert _failing('X = "cs2rl.env.nav"\n') == [("a", "cs2rl.env.nav")]


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


def test_an_exemption_hides_only_its_occurrence_and_is_reported_once_stale(tmp_path):
    """scan() suppresses an exempted occurrence and reports an exemption that matched
    nothing, so a fixed string forces its EXEMPT row out; a C occurrence is keyed by
    "<c>" the same way.

    Runs on synthetic files in a tmp root, so no real file has to keep a failing
    string for this test to borrow (it borrowed setup.py's extension name until #205
    part 2b made that name resolve).
    """
    (tmp_path / "synthetic.py").write_text('A = "cs2rl.nav"\nB = "cs2rl.nope"\n')
    (tmp_path / "synthetic.c").write_text('const char* a = "cs2rl.nav";\n')
    key = ("synthetic.py", "<module>", "cs2rl.nav")
    c_key = ("synthetic.c", "<c>", "cs2rl.nav")
    files = ["synthetic.py", "synthetic.c"]
    unexempted = scan(files, {}, root=tmp_path)
    assert [line.split(" in ")[0]
            for line in unexempted] == ["synthetic.py:1", "synthetic.py:2",
                                        "synthetic.c:1"], unexempted
    nope = unexempted[1]
    assert scan(files, {key: "reason", c_key: "reason"}, root=tmp_path) == [nope]
    stale = ("synthetic.py", "<module>", "not-a-failing-text")
    assert scan(files, {
        key: "reason",
        c_key: "reason",
        stale: "reason"
    }, root=tmp_path) == [
        nope, f"STALE EXEMPTION {stale}: no occurrence fails any more; delete the entry"
    ]


def test_an_exempted_path_under_a_subpackage_is_reported_as_masking():
    """A (c) row whose deepest tracked ancestor is a subpackage fails as MASKING; one
    directly under src/cs2rl, and a dotted-name row, do not.

    WHY: an exempted path under a package keeps matching after the package moves,
    so its stale text stays hidden (the WHY above). The real EXEMPT table holds no
    such row, so the main pin is green whether or not this check works; this is its
    positive control. `src/cs2rl/env/` is a real subpackage; `nope.py` need not exist.
    The second masking row is the shape that actually masked (four of the five texts
    commit 1 of #205 part 2b deleted): a build output under the C package's untracked
    zig-out/bin, whose immediate parent is untracked. Only a walk up to the deepest
    tracked ancestor (the C package's directory) finds its package; a check of the
    immediate parent alone reports the first row and misses this one.
    """
    masking = ("tests/x.py", "f", "src/cs2rl/env/nope.py")
    build_output = (c_env.ZIG_OUT / "bin" / "cs2_demo").relative_to(REPO_ROOT).as_posix()
    package = c_env.SOURCE_DIR.relative_to(REPO_ROOT).as_posix()
    assert Path(build_output).parent.as_posix() not in _tracked_dirs(), (
        f"{build_output}'s directory is tracked now: the row no longer needs the walk")
    assert package in _tracked_dirs(), f"{package} is not a tracked directory"
    masking_output = ("tests/x.py", "f", build_output)
    rows = {
        masking: "r",
        masking_output: "r",
        ("tests/x.py", "f", "src/cs2rl/nope.py"): "r",
        ("tests/x.py", "f", "src/cs2rl/*.py"): "r",
        ("tests/x.py", "f", "cs2rl.env.nope"): "r",
    }
    report = [line for line in scan([], rows) if not line.startswith("STALE EXEMPTION")]
    assert report == [masking_row_problem(masking), masking_row_problem(masking_output)], report
    assert "src/cs2rl/env" in report[0] and "Derive this path" in report[0], report
    assert f"the package {package}," in report[1], report


# The extension controls. Names and paths derive from the C package itself, so a move
# carries them along; each control also checks its premise, so a stale one goes red
# instead of passing for the wrong reason.
_EXTENSION = f"{c_env.__name__}.binding"
_EXTENSION_SOURCE = (c_env.SOURCE_DIR / "binding.c").relative_to(REPO_ROOT).as_posix()
# A tracked C file beside binding.c that defines no PyInit_: a program, not a module.
_NOT_AN_EXTENSION = f"{c_env.__name__}.cs2_demo"
# The same leaf under the C package's PARENT package (cs2rl.env.binding): a tracked
# PyInit_binding exists, but not in that package's directory, so the name must fail.
_MISPLACED_EXTENSION = f"{c_env.__name__.rpartition('.')[0]}.binding"


def _force_an_untracked_extension_file(monkeypatch,
                                       name: str = _EXTENSION,
                                       directory: Path = c_env.SOURCE_DIR) -> str:
    """Make find_spec return `binding<EXT_SUFFIX>` in `directory` for `name`.

    That is what a built tree returns (path A), made independent of whether this
    tree has a built .so. The file need not exist: only the spec is read. Returns
    the forced origin. The default is the real extension in its own package.
    """
    origin = directory / f"binding{sysconfig.get_config_var('EXT_SUFFIX')}"
    forced = importlib.util.spec_from_file_location(name, origin)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a, **k: forced
                        if n == name else real(n, *a, **k))
    return str(origin)


def test_path_a_an_extension_file_without_its_tracked_source_does_not_resolve(
        monkeypatch, fresh_resolver_caches):
    """(i) path A, negative: with binding.c dropped from _tracked, `binding` fails, both
    through this tree's real find_spec and with a built .so forced. And with binding.c
    tracked, a `binding` .so forced beside the PARENT package (cs2rl.env.binding)
    fails too: its PyInit_binding source is in another directory.

    This is the pin's "a leftover .so never resolves a stale name" case: a moved
    package's .so left behind has no tracked source beside it. It first checks that
    the same name DOES resolve with binding.c tracked, so it cannot pass on a name
    that fails for another reason. The parent-package case pins the rule's
    same-directory condition: without it, any `P.binding` would resolve on any tracked
    PyInit_binding anywhere (the next move of binding within env/, or a half-renamed
    name).
    """
    real_tracked = _tracked()
    assert _EXTENSION_SOURCE in real_tracked, f"{_EXTENSION_SOURCE} is not tracked"
    assert dotted_problem(_EXTENSION) is None, dotted_problem(_EXTENSION)
    _clear_resolver_caches()
    misplaced = _force_an_untracked_extension_file(monkeypatch, _MISPLACED_EXTENSION,
                                                   c_env.SOURCE_DIR.parent)
    problem = dotted_problem(_MISPLACED_EXTENSION)
    assert problem == (f"{_MISPLACED_EXTENSION} resolves to {misplaced}, which is not a "
                       "tracked file"), problem
    _clear_resolver_caches()
    monkeypatch.setitem(globals(), "_tracked", lambda: real_tracked - {_EXTENSION_SOURCE})
    assert dotted_problem(_EXTENSION) is not None
    _clear_resolver_caches()
    origin = _force_an_untracked_extension_file(monkeypatch)
    problem = dotted_problem(_EXTENSION)
    assert problem == f"{_EXTENSION} resolves to {origin}, which is not a tracked file", problem


def test_path_a_an_untracked_extension_file_resolves_through_its_tracked_source(
        monkeypatch, fresh_resolver_caches):
    """(i') path A, positive: find_spec returns an untracked `binding<EXT_SUFFIX>` in the
    package directory, and `binding` resolves, because binding.c defines PyInit_binding."""
    origin = _force_an_untracked_extension_file(monkeypatch)
    spec = importlib.util.find_spec(_EXTENSION)
    assert spec is not None and spec.origin == origin, spec
    assert not _is_tracked(origin)
    assert dotted_problem(_EXTENSION) is None, dotted_problem(_EXTENSION)


def test_path_b_no_built_extension_resolves_through_its_tracked_source(
        monkeypatch, fresh_resolver_caches):
    """(ii) path B (a fresh clone, no .so): find_spec returns None for every submodule
    of the package; `binding` resolves, and neither a missing name, nor a tracked C
    file without a PyInit_ (cs2_demo.c), nor `binding` under the parent package
    (cs2rl.env.binding: PyInit_binding exists, but in another directory) does."""
    prefix = f"{c_env.__name__}."
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: None
        if name.startswith(prefix) else real(name, *a, **k))
    assert importlib.util.find_spec(_EXTENSION) is None
    assert importlib.util.find_spec(_MISPLACED_EXTENSION) is None, (
        f"{_MISPLACED_EXTENSION} is a real module now: pick another misplaced name")
    assert (REPO_ROOT / _EXTENSION_SOURCE).with_name("cs2_demo.c").relative_to(
        REPO_ROOT).as_posix() in _tracked(), "the PyInit_-less C file is gone: pick another"
    assert dotted_problem(_EXTENSION) is None, dotted_problem(_EXTENSION)
    assert dotted_problem(f"{prefix}nope") is not None
    assert dotted_problem(_NOT_AN_EXTENSION) is not None
    assert dotted_problem(_MISPLACED_EXTENSION) is not None


def _c_failing(source: str) -> list[tuple[str, str]]:
    """(clause, text) of every failing occurrence in a synthetic C `source`."""
    return [(clause, key[2]) for key, clause, _ in c_failures_in("synthetic.c", source)]


@pytest.mark.parametrize("source, expected", [
    pytest.param('static const char* a = "-m cs2rl.nope";\n', [("a", "cs2rl.nope")],
                 id="an -m name in a plain string"),
    pytest.param('#define M "cs2rl.nope"\n', [("a", "cs2rl.nope")], id="a #define body"),
    pytest.param('const char* s = "a\\"cs2rl.nope\\"b";\n', [("a", "cs2rl.nope")],
                 id="escaped quotes"),
    pytest.param('const char* p = "src/cs2rl/nope/play.py";\n', [("c", "src/cs2rl/nope/play.py")],
                 id="a stale path"),
    pytest.param('const char* u = "a //b /*c cs2rl.nope";\n', [("a", "cs2rl.nope")],
                 id="comment openers inside a string"),
    pytest.param("char q = '\"'; const char* s = \"cs2rl.nope\";\n", [("a", "cs2rl.nope")],
                 id="a string after a quote char literal"),
    pytest.param("#error can't\nconst char* s = \"cs2rl.nope\";\n", [("a", "cs2rl.nope")],
                 id="a string after an unterminated quote"),
])
def test_c_strings_fail_a_stale_name_or_path(source, expected):
    """A stale name or path in a C string literal fails, in every shape C writes one.

    The unterminated-quote case: a stray apostrophe (an `#error` line's text, which C
    never compiles) ends at its newline, so it cannot swallow the next line's string.
    """
    assert _c_failing(source) == expected


@pytest.mark.parametrize("source", [
    pytest.param("char q = '\"'; /* cs2rl.nope */\nchar r = '\"'; // cs2rl.nope\n",
                 id="a quote char literal then a comment"),
    pytest.param('// run "-m cs2rl.nope" from "src/cs2rl/nope.c"\n', id="a line comment"),
    pytest.param('/* "cs2rl.nope"\n   "src/cs2rl/nope.c" */\n', id="a block comment"),
    pytest.param('const char* a = "cs2rl.env.nav"; const char* b = "src/cs2rl/env/nav.py";\n',
                 id="resolving names"),
])
def test_c_comments_and_resolving_strings_do_not_fail(source):
    """A stale name in a comment is prose, and a `'"'` char literal opens no string.

    The comments QUOTE their stale names: a scanner that read a comment as code would
    find a string there, so a dropped comment rule fails these cases.
    """
    assert _c_failing(source) == []


def test_c_string_literals_reports_each_string_at_its_line():
    """Lines count through block comments, line continuations and CRLF-free sources alike."""
    source = '/* a\n   b */\n#define M \\\n    "x"\nconst char* s = "y"; // "z"\n'
    assert list(c_string_literals(source)) == [(4, "x"), (5, "y")]


@pytest.mark.parametrize("escape, value", [
    pytest.param("\\n", "\n", id="newline"),
    pytest.param("\\t", "\t", id="tab"),
    pytest.param("\\r", "\r", id="carriage return"),
    pytest.param("\\0", "\0", id="nul"),
    pytest.param("\\\\", "\\", id="backslash"),
    pytest.param('\\"', '"', id="double quote"),
])
def test_c_string_literals_decodes_an_escape_to_the_character_c_sees(escape, value):
    """Each escape reads as the character C sees, so the text around it is judged whole.

    A lexer that kept the raw character would read `"src/cs2rl/nope.py\\n"` as
    `src/cs2rl/nope.pyn` (#287, measured with the escape lookup removed: the other C
    controls all stayed green). One case per escape: the backslash and quote cases also
    pin the fallback, where any other escaped character stands for itself.
    """
    source = f'const char* s = "a{escape}b";\n'
    assert list(c_string_literals(source)) == [(1, f"a{value}b")]


# Every (a)/(c) occurrence in the tracked C files at this commit, resolving or not:
# cs2_demo.c's find_repo marker (:162), its borrow hint (:203) and its `-m` argv (:251).
_C_OCCURRENCES = [
    ("src/cs2rl/env/c/cs2_demo.c", "a", "cs2rl.viz.play"),
    ("src/cs2rl/env/c/cs2_demo.c", "a", "cs2rl.viz.play"),
    ("src/cs2rl/env/c/cs2_demo.c", "c", "src/cs2rl/__init__.py"),
]


def test_the_c_scan_finds_exactly_the_known_occurrences_in_the_real_tree():
    """The real C sources yield exactly _C_OCCURRENCES, so the C scan is known to read them.

    A scanner that read no C file, or lost a literal to a comment or char-literal
    bug, would leave the main pin green; this pins what it must find.
    """
    found = sorted(
        (relative, clause, text) for relative in _c_files()
        for _, clause, text, _ in c_occurrences(relative, (REPO_ROOT /
                                                           relative).read_text(encoding="utf-8")))
    assert found == sorted(_C_OCCURRENCES), found
