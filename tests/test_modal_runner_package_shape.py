"""AST gates for the shape of the scripts/modal_runner/ package.

Every clause here describes the shape the package keeps from now on:

* module population: the submodule files on disk (every `*.py` but
  `__init__.py`) are exactly the modules MANIFEST and DEPENDENCIES declare
  (`files`, `undeclared-module`), and none has a grab-bag name
  (`forbidden-name`);
* ownership: each module's top-level names equal its MANIFEST entry
  (`membership`), no module binds a name twice (`duplicates`) or assigns at
  module scope to a target MANIFEST cannot own (`undeclared-assignment`), and
  MANIFEST gives no symbol two owners (`manifest-duplicates`);
* module-scope shape (`header`): the docstring, imports, one imports-only
  `if TYPE_CHECKING:` block without else, and declarations, nothing else;
* import-time work (`import-time`): what a declaration evaluates on import
  (its value, decorators, default arguments, class bases and class body) may
  use only literals, names, attribute reads, containers, arithmetic and calls
  to IMPORT_TIME_CALLS, and a module with annotations defers them;
* the import graph: intra-package imports equal DEPENDENCIES
  (`runtime-edges`) and ANNOTATION_DEPENDENCIES (`annotation-edges`);
* import purity: no non-stdlib import executes when a module is imported;
* qualified seams (`seam-from-import`, `seam-import-time`, `seam-readers`,
  `seam-undeclared`): each QUALIFIED_SEAMS name is read as `owner.name` at
  call time by exactly its listed readers, and no other name is read across
  modules through the module object;
* the facade (`facade`): `__init__.py` defines nothing and runs on every
  `import scripts.modal_runner`, so its module scope may hold only its
  docstring, named one-dot relative imports of its submodules and a literal
  `__all__`.

Editing a function body, a docstring or a constant's value trips none of them
unless the edit adds or removes an import, or puts work into a value that runs
on import (`import-time`). Adding, removing, renaming or moving
a top-level symbol, adding a module, or changing which package modules a module
imports does: update the table the failure message names, in the same commit as
the change. Runtime imports are compared with DEPENDENCIES and imports under
`if TYPE_CHECKING:` with ANNOTATION_DEPENDENCIES, separately, and the
dependency walk reads function bodies too. So a first runtime import of a
module imported only under `TYPE_CHECKING` is a new edge, and so is the reverse;
another import of a module already imported in the same way is not; removing
a module's last runtime import, or its last `TYPE_CHECKING` one, drops that
edge. No module size budget applies.

History: the split commit, which created scripts/modal_runner/, also proved in
this file (then tests/test_modal_relocation.py) that every moved declaration
was byte-identical to `scripts/modal_runner_lib.py` at 2bb32ac apart from 20
declared `module.name` qualifiers; those relocation-oracle checks were retired
after it.

Every failure message names the clause, the file or symbol, and the edit that
resolves it (`_explain`). Controls plant defects into in-memory copies of the
sources or into a copy of the package under `tmp_path`, never into the live
package. A control first re-checks the unplanted package; when that check
fails, its message starts with PRECONDITION and names the contract test whose
message says what to fix (both, for a control that re-checks two), because the
control itself did not break.
"""
import ast
import re
from pathlib import Path

import pytest

# The package's declared shape: MANIFEST, the import tables, and the module list
# derived from them. Every gate here reads it; the file itself is data only.
from tests.modal_runner_tables import (
    ANNOTATION_DEPENDENCIES,
    DEPENDENCIES,
    MANIFEST,
    QUALIFIED_SEAMS,
    RUNNER_MODULES,
    TABLES,
)

# The packaging gates' own helpers and constants, imported rather than copied so
# that both files recognise module-level names, module-scope shape, non-stdlib
# imports and the package population by ONE definition; two copies could let the
# gates disagree about the same package. Importing a collected test module does
# not collect its tests twice: none of these names is a test. They are not
# moved to tests/modal_test_helpers.py because that is not a pure move: that
# file admits only names both halves of the runner/client test seam reach, and
# `classify_seam` enforces it. PITFALL: `_module_level_names` also drives
# `classify_seam`, and `_module_level_binding_counts` the seam's placement gate,
# so a change made to either for the seam's sake changes the membership and
# duplicates clauses here as well.
from tests.test_modal_packaging import (
    PACKAGED,
    _is_type_checking_test,
    _module_level_binding_counts,
    _module_level_names,
    _module_scope_shape_violations,
    _nonstdlib_module_scope_imports,
    _runner_module_population,
    bare_spelling_imports,
)

ROOT = Path(__file__).resolve().parents[1]
# Module names that say what a module IS rather than what it owns, which is how
# a module becomes a junk drawer; the `forbidden-name` clause and its message
# both read this.
GRAB_BAG_MODULE_NAMES = ("utils", "helpers", "common", "misc")
_UNDECLARED_MODULE_REMEDY = (
    "A path beyond the modules RUNNER_MODULES declares is a module in scripts/modal_runner/ "
    f"that the tables do not declare: declare it in {TABLES} in the same commit, or delete it.")
# How many times each control below plants its defect. Each pass starts by
# re-checking the unplanted package (or its tmp_path copy), so the second pass's
# check proves the first pass's plant-and-restore left the sources, or the copied
# files, as it found them. The gates hold no state between calls, so this is only
# about the tree. A control whose restore step broke would otherwise pass once
# and hide it.
_CONTROL_PASSES = 2
# Callees a package module may call while it is being imported, decorators
# included (applying a bare `@property` calls it): pure constructors of values
# and the class-building decorators. `<str>.strip` is a method of a string
# literal (commands.CUDA_PROBE_SOURCE). The `import-time` clause and its message
# both read this. Add a callee only if calling it does nothing but build its
# value; `importlib.util.find_spec`, `os.environ.get` and `atexit.register` are
# the kind of call this keeps out of module scope.
IMPORT_TIME_CALLS = frozenset({
    "<str>.strip", "Path", "PurePosixPath", "classmethod", "dataclass", "field", "frozenset",
    "property", "re.compile", "staticmethod", "timedelta"
})
# The expression kinds an import-time expression may be built from. Left out on
# purpose: `NamedExpr` (a walrus binds a name no MANIFEST entry owns), `IfExp`,
# `BoolOp` and `Compare` (the expression form of the module-scope conditional
# the header rule bans), `Subscript` (`os.environ["X"]` reads the environment),
# comprehensions and `Lambda`. Allowed attribute reads are the accepted limit:
# `frozenset(os.environ)` passes, and so does the default `run=subprocess.run`
# the package uses for injection, which is the same shape.
_IMPORT_TIME_KINDS = (ast.Attribute, ast.BinOp, ast.Call, ast.Constant, ast.Dict,
                      ast.FormattedValue, ast.JoinedStr, ast.List, ast.Name, ast.Set, ast.Starred,
                      ast.Tuple, ast.UnaryOp)


def _segments(source):
    """{top-level name: its exact declaration text} for one module's source.

    A segment includes the decorators, the complete class body and the comment
    lines directly above the declaration. Controls use it to cut one
    declaration out of a module, or copy it into another, without hand-writing
    its text.

    PITFALL: the lines are split as bytes, not with `str.splitlines`, which
    also breaks at `\\x0b`, `\\x0c`, `\\x1c`-`\\x1e`, `\\x85` and
    U+2028/U+2029 (measured over every code point). `ast` line numbers count
    only `\\n`, `\\r\\n` and `\\r`, which is all `bytes.splitlines` splits
    on, so a segment cannot drift off its node.
    """
    lines = source.encode().splitlines(keepends=True)
    found = {}
    for name, node in _module_level_names(ast.parse(source)).items():
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        while first > 1 and lines[first - 2].lstrip().startswith(b"#"):
            first -= 1
        found[name] = b"".join(lines[first - 1:node.end_lineno]).decode()
    return found


def _package_sources(root):
    """{filename: text} for every package submodule on disk under `root`.

    The population is `_runner_module_population`'s: every declared module,
    read even when absent so that a missing or unreadable member raises an
    OSError naming it, then every undeclared `*.py` in the directory, which the
    gates below report rather than skip. Read as UTF-8 bytes because the sources
    hold non-ASCII prose, and the locale must not decide whether they parse.
    """
    candidates, _ = _runner_module_population(root)
    return {Path(rel).name: (Path(root) / rel).read_bytes().decode("utf-8") for rel in candidates}


def _copy_live_package(tmp_path):
    """Copy the live `scripts/modal_runner/*.py`, facade included, under `tmp_path`.

    Returns the copy's package directory. Controls that need a REAL file on
    disk plant into this copy and read it back through `_package_sources` or
    `_nonstdlib_module_scope_imports(tmp_path)`; nothing ever writes into the
    live package. Only `*.py` is copied, so no `__pycache__` rides along, and
    bytes are copied as-is so no encoding or newline translation happens.
    """
    package = tmp_path / "scripts" / "modal_runner"
    package.mkdir(parents=True)
    for path in (ROOT / "scripts" / "modal_runner").glob("*.py"):
        (package / path.name).write_bytes(path.read_bytes())
    return package


def _structure_violations(sources, manifest=MANIFEST):
    """The ownership, module-population and module-scope-shape clauses over `sources`.

    `sources` is a {filename: text} map. Combines the manifest's own clause,
    the file population, grab-bag module names, header shape, membership,
    duplicate bindings and undeclared assignments. `manifest` defaults to
    MANIFEST; controls pass an edited copy. Every file in `sources` gets every
    per-file check, including one no manifest entry declares.
    """
    # Annotated because the gates' tuples differ in shape per clause; pyrefly
    # would otherwise infer the first append's shape and reject the others.
    violations: list[tuple] = []
    # MANIFEST gives every symbol one owner. Per-module membership compares each
    # file with its OWN manifest entry, so it cannot see a symbol declared in,
    # and defined in, two modules; before this clause existed, that plant left
    # every gate green.
    owners = {}
    for filename, names in manifest.items():
        for name in names:
            owners.setdefault(name, []).append(filename)
    shared = {name: files for name, files in owners.items() if len(files) > 1}
    if shared:
        violations.append(("manifest-duplicates", shared))
    if set(sources) != set(manifest):
        violations.append(
            ("files", sorted(set(sources) - set(manifest)), sorted(set(manifest) - set(sources))))
    for filename, source in sources.items():
        if filename.removesuffix(".py") in GRAB_BAG_MODULE_NAMES:
            violations.append(("forbidden-name", filename))
        declared = set(manifest.get(filename, []))
        tree = ast.parse(source)
        examined, shape = _module_scope_shape_violations(tree)
        assert examined == len(tree.body), (
            f"the header instrument examined {examined} of {len(tree.body)} module-scope "
            f"statements in {filename}, so its verdict does not cover the file. This is a "
            "defect in `_module_scope_shape_violations` (tests/test_modal_packaging.py), not in "
            "the package: make it examine every statement in `tree.body` exactly once.")
        violations.extend(("header", filename, item) for item in shape)
        names = _module_level_names(tree)
        if set(names) != declared:
            violations.append(("membership", filename, sorted(set(names) - declared),
                               sorted(declared - set(names))))
        duplicates = {
            n: count
            for n, count in _module_level_binding_counts(tree).items() if count != 1
        }
        if duplicates:
            violations.append(("duplicates", filename, duplicates))
        # `_module_level_names` keeps one node per name: the LAST plain-name
        # binding. Any other module-scope assignment is invisible to membership:
        # an earlier binding of a rebound name, or an annotated attribute target
        # (`VOLUME_MOUNT.name: str = ...`), which the header rule allows because
        # it is an AnnAssign.
        owned = {id(node) for node in names.values()}
        violations.extend(
            ("undeclared-assignment", filename, node.lineno) for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and id(node) not in owned)
        violations.extend(
            ("import-time", filename, line, what) for line, what in _import_time_violations(tree))
    return violations


def _import_time_roots(tree):
    """(expression, is_decorator) for each expression a module evaluates when imported.

    The header rule leaves only declarations at module scope, and a
    declaration still runs code on import: an assignment's value, a decorator,
    a default argument, a class's bases and keywords, and everything in a class
    body, nested classes included. A function body does not run, and neither
    does an `if TYPE_CHECKING:` block, whose body the header rule limits to
    imports. Annotations are not roots: `_import_time_violations` requires them
    deferred instead.
    """
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            yield node.value, False
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield from ((decorator, True) for decorator in node.decorator_list)
            if isinstance(node, ast.ClassDef):
                yield from ((base, False) for base in node.bases)
                yield from ((keyword.value, False) for keyword in node.keywords)
                stack.extend(node.body)
            else:
                yield from ((default, False) for default in node.args.defaults)
                yield from ((default, False) for default in node.args.kw_defaults if default)


def _callee(func):
    """How IMPORT_TIME_CALLS spells a callee: `name`, `module.name`, or `<str>.method`."""
    if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Constant)
            and isinstance(func.value.value, str)):
        return f"<str>.{func.attr}"
    return ast.unparse(func)


def _import_time_violations(tree):
    """Sorted `(line, what)` for each import-time expression outside the allowed shape.

    WHY: the header rule stops `if importlib.util.find_spec("modal"):` as a
    statement, but not the same probe hidden in a declaration, such as
    `VOLUME_NAME = importlib.util.find_spec("modal") and ...`, a walrus, or a
    decorator that registers something. The relocation oracle the split
    retired used to catch that by byte identity (gh#223 item 7).

    A module that has annotations and lacks `from __future__ import
    annotations` evaluates them on import, so it is reported once, at its
    first annotation; every package module defers them.
    """
    found = set()
    deferred = any(
        isinstance(node, ast.ImportFrom) and node.module == "__future__" and any(
            alias.name == "annotations" for alias in node.names) for node in tree.body)
    annotated = sorted(
        node.lineno for node in ast.walk(tree)
        if (isinstance(node, ast.AnnAssign) or (isinstance(node, ast.arg) and node.annotation) or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns)))
    if annotated and not deferred:
        found.add((annotated[0], "an annotation (no `from __future__ import annotations`)"))
    for root, is_decorator in _import_time_roots(tree):
        if (is_decorator and not isinstance(root, ast.Call)
                and _callee(root) not in IMPORT_TIME_CALLS):
            found.add((root.lineno, f"decorator `{_callee(root)}`"))
        for node in ast.walk(root):
            if not isinstance(node, ast.expr):
                continue
            if not isinstance(node, _IMPORT_TIME_KINDS):
                found.add((node.lineno, f"a `{type(node).__name__}` node"))
            elif isinstance(node, ast.Call) and _callee(node.func) not in IMPORT_TIME_CALLS:
                found.add((node.lineno, f"a call to `{_callee(node.func)}`"))
    return sorted(found)


def _dependency_edges(sources):
    """({module: runtime edges}, {module: annotation-only edges}) between package modules.

    Only the spelling the package uses resolves to a member: a ONE-dot relative
    import (`from . import core`, `from .core import X`). Every other way to
    reach the package from inside it is recorded as an edge no table allows:
    a `..`-relative import as its dotted spelling (`..`, `..modal_runner`), and
    any absolute spelling as `PACKAGED`. The absolute shapes come from
    `bare_spelling_imports(source, PACKAGED)`, the static guard's walker, so
    this gate and that guard share one definition of an import of the package
    (`import`, from-import, `from scripts import modal_runner`, and literal
    `importlib.import_module` / `__import__`) instead of this file keeping a
    second, weaker copy that missed the package-level and parent spellings.

    PITFALL: that walker is an `ast.walk`, so an absolute spelling is recorded
    as a runtime edge even under `if TYPE_CHECKING:`. That fails closed: no
    table allows `PACKAGED` in either column.
    """
    runtime, annotations = {}, {}
    for filename, source in sources.items():
        module = filename.removesuffix(".py")
        runtime[module], annotations[module] = set(), set()
        if bare_spelling_imports(source, PACKAGED):
            runtime[module].add(PACKAGED)
        stack = [(ast.parse(source), False)]
        while stack:
            node, checking = stack.pop()
            if isinstance(node, ast.If) and _is_type_checking_test(node.test):
                stack.extend((child, True) for child in node.body)
                stack.extend((child, checking) for child in node.orelse)
                continue
            edges = annotations[module] if checking else runtime[module]
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                if node.module:
                    edges.add(node.module.split(".")[0])
                else:
                    edges.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level > 1:
                edges.add("." * node.level + (node.module or ""))
            stack.extend((child, checking) for child in ast.iter_child_nodes(node))
    return runtime, annotations


def _dependency_violations(sources):
    """The import-graph clauses: resolved edges must equal DEPENDENCIES / ANNOTATION_DEPENDENCIES.

    Compared per module. A module in `sources` that DEPENDENCIES does not
    declare is reported as `undeclared-module` rather than skipped.
    """
    runtime, annotations = _dependency_edges(sources)
    failures: list[tuple] = [("undeclared-module", module)
                             for module in sorted(set(runtime) - set(DEPENDENCIES))]
    for module in RUNNER_MODULES:
        expected = set(DEPENDENCIES[module])
        if runtime.get(module) != expected:
            failures.append(("runtime-edges", module, runtime.get(module), expected))
        expected = set(ANNOTATION_DEPENDENCIES.get(module, []))
        if annotations.get(module) != expected:
            failures.append(("annotation-edges", module, annotations.get(module), expected))
    return failures


def _sibling_bindings(tree, modules):
    """{bound name: package module} for each `from . import <module> [as <name>]` in `tree`."""
    return {
        alias.asname or alias.name: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None
        for alias in node.names if alias.name in modules
    }


def _seam_violations(sources, seams=QUALIFIED_SEAMS):
    """The `seam` clauses: the qualified seams stay qualified, and nothing else is qualified.

    `sources` is a {filename: text} map of the package submodules. Reports:

    * `seam-from-import`: a module other than the owner from-imports a seam's
      name, or star-imports a seam's owner. Either binds a copy that a test
      patching the owner never reaches.
    * `seam-import-time`: a seam read where it runs on import
      (`_import_time_roots`: a module-scope value, a default argument, a
      decorator, a class body), which binds the object the same way.
    * `seam-readers`: the modules that read `owner.name` differ from the
      seam's QUALIFIED_SEAMS entry, so the table no longer describes the code.
    * `seam-undeclared`: a module reads another name its owner defines as
      `owner.name`. Either it is a seam tests rely on, and belongs in the
      table, or it should be from-imported like every other name.

    WHY: the dependency and membership gates cannot see any of these, because
    the reader-to-owner edge exists either way. Measured before this clause
    (gh#221): every seam read rewritten as a from-import left both returning
    []. A name is matched to its owner through `from . import <module>`
    bindings only, so a local variable named like a module, with an attribute
    the module does not define, is not a read of it.
    """
    modules = {filename.removesuffix(".py") for filename in sources}
    owned = {
        filename.removesuffix(".py"): set(_module_level_names(ast.parse(text)))
        for filename, text in sources.items()
    }
    violations: list[tuple] = []
    readers: dict[str, set[str]] = {seam: set() for seam in seams}
    for filename, text in sources.items():
        module = filename.removesuffix(".py")
        tree = ast.parse(text)
        bound = _sibling_bindings(tree, modules)
        import_time = {id(node) for root, _ in _import_time_roots(tree) for node in ast.walk(root)}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module in modules:
                for alias in node.names:
                    seam = f"{node.module}.{alias.name}"
                    star = alias.name == "*" and any(
                        name.startswith(f"{node.module}.") for name in seams)
                    if seam in seams or star:
                        violations.append(("seam-from-import", module, node.lineno, seam))
            elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                  and bound.get(node.value.id, module) != module
                  and node.attr in owned.get(bound[node.value.id], set())):
                seam = f"{bound[node.value.id]}.{node.attr}"
                if seam not in seams:
                    violations.append(("seam-undeclared", module, node.lineno, seam))
                    continue
                readers[seam].add(module)
                if id(node) in import_time:
                    violations.append(("seam-import-time", module, node.lineno, seam))
    for seam, declared in seams.items():
        if readers[seam] != set(declared):
            violations.append(("seam-readers", seam, sorted(readers[seam]), sorted(declared)))
    return violations


def _facade_seam_list(facade):
    """The `module.name` spellings the facade docstring's QUALIFIED SEAMS paragraph lists."""
    docstring = ast.get_docstring(ast.parse(facade)) or ""
    paragraph = docstring.split("QUALIFIED SEAMS.", 1)[-1].split("\n\n", 1)[0]
    return {
        f"{module}.{name}"
        for module, name in re.findall(r"`(\w+)\.(\w+)`", paragraph) if module in RUNNER_MODULES
    }


def _facade_violations(source):
    """`("facade", line, statement kind)` for each module-scope statement the facade may not hold.

    `source` is the text of scripts/modal_runner/__init__.py. The facade runs
    on every `import scripts.modal_runner`, production's included, yet the
    module-population clauses skip it because it declares no symbol of its own.
    Its module scope may hold only its docstring, one-dot relative imports of
    its submodules (`from .core import X`, `from . import core`; no star
    import, which would re-export whatever the submodule binds) and `__all__`
    bound to a list or tuple of string literals. That one rule covers, for this
    file, what the header rule covers for the modules (a call, a conditional, a
    loop), what the purity rule covers (an absolute or non-stdlib import) and a
    definition, which belongs in the module that owns it.
    """
    violations: list[tuple] = []
    for index, node in enumerate(ast.parse(source).body):
        docstring = (index == 0 and isinstance(node, ast.Expr)
                     and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
        relative = (isinstance(node, ast.ImportFrom) and node.level == 1
                    and all(alias.name != "*" for alias in node.names))
        exported = (isinstance(node, ast.Assign)
                    and [ast.unparse(target) for target in node.targets] == ["__all__"]
                    and isinstance(node.value, (ast.List, ast.Tuple)) and all(
                        isinstance(item, ast.Constant) and isinstance(item.value, str)
                        for item in node.value.elts))
        if not (docstring or relative or exported):
            violations.append(("facade", node.lineno, type(node).__name__))
    return violations


def _precondition(contract, subject="the live package"):
    """The subject of a control's precondition: the unplanted package, or its copy, is clean.

    WHY: every control re-checks the unplanted package, so one real defect in
    the package turns every control red with the same message. The prefix says
    the control is not what broke and names `contract`: the test, or the two
    tests, whose own message says what to fix.
    """
    return f"PRECONDITION (the live package already fails; fix {contract} first): {subject}"


def _explain(violations):
    """One line per violation: the clause, where it fired, and the edit that resolves it.

    WHY: the gates return tuples, and a red that prints only a tuple list tells
    whoever meets it neither which clause objected nor whether the fix is to
    revert their edit or to update a table in this file.
    """
    lines = []
    for v in violations:
        kind = v[0]
        if kind == "files":
            parts = []
            if v[1]:
                parts.append(f"holds modules the tables do not declare, {v[1]}: declare each in "
                             f"{TABLES} in the same commit, or delete the stray file")
            if v[2]:
                parts.append(f"lacks modules MANIFEST declares, {v[2]}: restore them, or remove "
                             f"their entries from {TABLES}")
            text = f"scripts/modal_runner/ {'; and it '.join(parts)}."
        elif kind == "undeclared-module":
            text = (f"scripts/modal_runner/{v[1]}.py is not in DEPENDENCIES. Declare the module in "
                    f"{TABLES}, or delete it.")
        elif kind == "manifest-duplicates":
            text = (
                f"MANIFEST gives these symbols more than one owner: {v[1]}. Each symbol lives in "
                "exactly one module; delete the extra MANIFEST entries and definitions.")
        elif kind == "header":
            text = (f"{v[1]} line {v[2][1]} ({v[2][0]}): {v[2][2]}. Module scope may hold only the "
                    "docstring, imports, one imports-only `if TYPE_CHECKING:` block without else, "
                    "and declarations; move the statement into a function.")
        elif kind == "undeclared-assignment":
            text = (
                f"{v[1]} line {v[2]} is a module-level assignment MANIFEST cannot own: its name "
                "is bound again later in the file, or it annotates a target that is not a "
                "plain name. Bind each module-level name once, to a plain name listed in "
                f"MANIFEST['{v[1]}'], and move any other assignment into a function.")
        elif kind == "membership":
            parts = []
            if v[2]:
                parts.append(f"{v[1]} defines {v[2]}, which MANIFEST['{v[1]}'] lacks")
            if v[3]:
                parts.append(f"MANIFEST['{v[1]}'] lists {v[3]}, which {v[1]} does not define")
            text = (f"{'; '.join(parts)}. If you meant to add, remove or move a symbol, update "
                    f"MANIFEST['{v[1]}'] in tests/modal_runner_tables.py in the same "
                    "commit.")
        elif kind == "duplicates":
            text = (f"{v[1]} binds {v[2]} more than once at module scope; the later binding "
                    "silently shadows the earlier one. Keep one.")
        elif kind == "runtime-edges":
            text = (f"{v[1]}.py imports package modules {v[2]} at run time; DEPENDENCIES['{v[1]}'] "
                    f"allows {v[3]}. Write intra-package imports as one-dot relative imports; "
                    f"absolute (`{PACKAGED}`) and `..` spellings are never allowed. If a new edge "
                    "is intended and keeps the graph acyclic (core stays a leaf), update "
                    "DEPENDENCIES in tests/modal_runner_tables.py in the same commit.")
        elif kind == "annotation-edges":
            text = (f"{v[1]}.py imports {v[2]} under `if TYPE_CHECKING:`; "
                    f"ANNOTATION_DEPENDENCIES allows {v[3]}. Update ANNOTATION_DEPENDENCIES in "
                    "tests/modal_runner_tables.py if the annotation-only import is intended.")
        elif kind == "import-time":
            text = (f"{v[1]} line {v[2]}: {v[3]} runs when the module is imported. What a "
                    "declaration evaluates on import (its value, decorators, default arguments, "
                    "class bases and class body) may use only literals, names, attribute reads, "
                    "containers, arithmetic and calls to IMPORT_TIME_CALLS "
                    f"({', '.join(sorted(IMPORT_TIME_CALLS))}). Move the work into a function, "
                    "and keep `from __future__ import annotations` in every module that "
                    "annotates. A new pure constructor goes in IMPORT_TIME_CALLS "
                    "(tests/test_modal_runner_package_shape.py) in the same commit.")
        elif kind == "seam-from-import":
            text = (f"{v[1]}.py line {v[2]} imports the qualified seam `{v[3]}` by name, which "
                    "binds a copy that a test patching the owning module never reaches. Read it "
                    "as `owner.name` at call time (QUALIFIED SEAMS in "
                    "scripts/modal_runner/__init__.py).")
        elif kind == "seam-import-time":
            text = (f"{v[1]}.py line {v[2]} reads the qualified seam `{v[3]}` when the module is "
                    "imported (a module-scope value, a default argument, a decorator or a class "
                    "body). That binds the object then, so a later patch on the owning module "
                    "never reaches it. Read it inside the function that calls it.")
        elif kind == "seam-readers":
            text = (f"the modules that read `{v[1]}` are {v[2]}; QUALIFIED_SEAMS in "
                    f"tests/modal_runner_tables.py lists {v[3]}. If a reader was added or removed "
                    "on purpose, update that entry in the same commit; if a reader switched to "
                    f"a from-import, restore `{v[1]}`.")
        elif kind == "seam-undeclared":
            owner, name = v[3].split(".")
            text = (f"{v[1]}.py line {v[2]} reads `{v[3]}` through the module object, but it is "
                    f"not a qualified seam. From-import it (`from .{owner} import {name}`); or, "
                    "if tests must patch it on its owner, add it to QUALIFIED_SEAMS in "
                    "tests/modal_runner_tables.py and to the QUALIFIED SEAMS paragraph of "
                    "scripts/modal_runner/__init__.py.")
        elif kind == "forbidden-name":
            text = (f"{v[1]} is a grab-bag module name ({'/'.join(GRAB_BAG_MODULE_NAMES)}). Name "
                    "the module for the concern it owns.")
        elif kind == "facade":
            text = (f"scripts/modal_runner/__init__.py line {v[1]} ({v[2]}): the facade runs "
                    "on every `import scripts.modal_runner`, so its module scope may hold only "
                    "its docstring, one-dot relative imports of its submodules (named, never "
                    "`*`) and `__all__` alone bound to a list or tuple of string literals. Move "
                    "the statement into the module that owns it, and re-export a name with "
                    "`from .<module> import <name>`.")
        else:
            text = f"unrecognised violation {v!r}"
        lines.append(f"[{kind}] {text}")
    return "\n".join(lines) or "(no violations)"


def _assert_no_violations(violations, subject):
    """Fail with `_explain`'s sentences, prefixed by what was being checked."""
    assert violations == [], f"{subject}:\n{_explain(violations)}"


def _assert_rejected(violations, prefix, plant):
    """Fail unless some violation starts with `prefix`; name the plant and what did fire."""
    assert any(v[:len(prefix)] == prefix for v in violations), (
        f"the {plant} plant was not rejected by {prefix}; the gates reported:\n"
        f"{_explain(violations)}")


def _hoist_type_checking_imports(source):
    """Move a module's `if TYPE_CHECKING:` imports to module scope, dedented.

    The plant for "an annotation-only edge became a runtime edge". Asserts the
    block holds relative from-imports only, so the plant cannot silently move
    something else.
    """
    lines = source.splitlines(keepends=True)
    block = next(n for n in ast.parse(source).body if isinstance(n, ast.If))
    assert all(isinstance(n, ast.ImportFrom) and n.level == 1 for n in block.body), (
        f"the TYPE_CHECKING block at line {block.lineno} holds more than relative imports")
    moved = "".join("".join(lines[n.lineno - 1:n.end_lineno]).replace("    ", "", 1)
                    for n in block.body)
    lines[block.lineno - 1:block.end_lineno] = [moved]
    return "".join(lines)


@pytest.fixture
def live_sources():
    """{filename: text} for the live package's submodules, read fresh for each test."""
    return _package_sources(ROOT)


def test_package_structure_contract(live_sources):
    """The structure clauses on the live package, plus the tables' consistency.

    RUNNER_MODULES is derived from MANIFEST, so the only way the tables can
    disagree about the module list is DEPENDENCIES or ANNOTATION_DEPENDENCIES.
    """
    declared = set(RUNNER_MODULES)
    keyed = set(DEPENDENCIES) == declared and set(ANNOTATION_DEPENDENCIES) <= declared
    assert keyed, (
        f"DEPENDENCIES {sorted(DEPENDENCIES)} must be keyed by exactly the modules MANIFEST "
        f"declares ({list(RUNNER_MODULES)}), and ANNOTATION_DEPENDENCIES "
        f"{sorted(ANNOTATION_DEPENDENCIES)} by some of them; update them together in "
        "tests/modal_runner_tables.py.")
    _assert_no_violations(_structure_violations(live_sources), "the live package")


def test_package_facade_contract():
    """The facade's module-scope rule (`_facade_violations`) on the live `__init__.py`."""
    facade = (ROOT / "scripts" / "modal_runner" / "__init__.py").read_text(encoding="utf-8")
    _assert_no_violations(_facade_violations(facade), "the live facade")


# Each row appends one plant to the live facade, in memory, and pairs it with the
# statement kinds, in order, that the `facade` clause must report for the
# appended lines. The ID ends in the verdict the row asserts.
@pytest.mark.parametrize("plant, kinds", [
    pytest.param(plant, kinds, id=f"{name}-{'rejected' if kinds else 'accepted'}")
    for name, plant, kinds in [
        ("env-mutation", "import os\nos.environ.setdefault('CS2RL_FACADE_PLANT', '1')\n",
         ["Import", "Expr"]),
        ("print", "print('facade side effect')\n", ["Expr"]),
        ("non-stdlib-import", "import numpy\n", ["Import"]),
        ("conditional-import", "if True:\n    from .core import VOLUME_MOUNT\n", ["If"]),
        ("definition", "def _helper():\n    return 1\n", ["FunctionDef"]),
        ("computed-all", "__all__ = sorted(__all__)\n", ["Assign"]),
        ("absolute-import", "from os import environ\n", ["ImportFrom"]),
        ("parent-relative-import", "from .. import run_modal\n", ["ImportFrom"]),
        ("stray-string", "'a string after the docstring'\n", ["Expr"]),
        ("other-assign", "HELPER = ['a']\n", ["Assign"]),
        ("computed-all-element", "__all__ = [print('x')]\n", ["Assign"]),
        ("non-string-all-element", "__all__ = [1]\n", ["Assign"]),
        ("chained-all", "__all__ = HELPER = ['a']\n", ["Assign"]),
        ("set-all", "__all__ = {'a'}\n", ["Assign"]),
        ("star-import", "from .core import *\n", ["ImportFrom"]),
        ("tuple-all", "__all__ = ('a',)\n", []),
        ("re-export", "from .core import VOLUME_MOUNT\n", []),
    ]
])
def test_package_facade_controls(plant, kinds):
    """One plant per row, rejected by the `facade` clause alone, or accepted.

    * env-mutation: `os.environ.setdefault(...)` at the facade's module scope.
      Before this clause existed it, and the bare `print`, turned no gate red:
      the population gates skip `__init__.py`, and the surface gates read names.
    * non-stdlib-import: the purity half. The container-equivalent import in
      tests/test_modal_client.py also objects, but only by running the image.
    * conditional-import, definition, computed-all: the header half, a helper
      that belongs in its owning module, and an `__all__` that runs code.
    * absolute-import, parent-relative-import, stray-string, other-assign,
      computed-all-element, non-string-all-element: one sub-condition of the
      allowed shape each, and weakening it turns that row red. The two import
      rows hold `level == 1` from below (the stdlib `from os import environ`,
      which the purity rule would pass) and from above; stray-string holds
      the docstring to index 0; other-assign holds the `__all__` target;
      computed-all-element holds the element check, non-string-all-element
      its string half.
    * chained-all and set-all: `__all__` must be the only target and a list or
      tuple, so a second target and a set are rejected. tuple-all is ACCEPTED,
      the tuple half of that shape.
    * star-import: re-exports whatever the submodule binds. The client surface
      gate also objects, but only by running the image.
    * re-export: ACCEPTED. A new one-dot relative import is legal shape here;
      whether the name belongs in the facade is the surface gates' question,
      so a clause that rejected every added line would fail this row.
    """
    facade = (ROOT / "scripts" / "modal_runner" / "__init__.py").read_text(encoding="utf-8")
    _assert_no_violations(_facade_violations(facade),
                          _precondition("test_package_facade_contract", "the live facade"))
    # The plant's statements start on the line after the facade's last line. A
    # rejected row names one kind per planted statement, so `strict` also checks
    # that the plant parses into as many statements as the row names.
    lines = [facade.count("\n") + node.lineno for node in ast.parse(plant).body]
    expected = [("facade", line, kind)
                for line, kind in zip(lines, kinds, strict=True)] if kinds else []
    violations = _facade_violations(facade + plant)
    verdict = f"rejected by exactly {expected}" if expected else "accepted"
    assert violations == expected, (
        f"the facade plant {plant!r} must be {verdict}; the gate reported:\n{_explain(violations)}")


def test_package_membership_rejects_wrong_owner_with_correct_union(live_sources):
    precondition = _precondition("test_package_structure_contract")
    for _ in range(_CONTROL_PASSES):
        _assert_no_violations(_structure_violations(live_sources), precondition)
        segment = _segments(live_sources["core.py"])["VOLUME_NAME"]
        wrong = dict(live_sources)
        wrong["core.py"] = wrong["core.py"].replace(segment, "", 1)
        wrong["request.py"] += "\n" + segment

        def union(modules):
            """Ignore module ownership to demonstrate the weaker union-only check."""
            return set().union(*(_module_level_names(ast.parse(s)) for s in modules.values()))

        assert union(wrong) == union(live_sources), (
            "the wrong-owner plant changed the union of names, so it no longer shows that a "
            "union-only check would pass it")
        failures = _structure_violations(wrong)
        objecting = sorted({v[1] for v in failures if v[0] == "membership"})
        assert objecting == [
            "core.py", "request.py"
        ], ("moving VOLUME_NAME from core.py to request.py must fail membership in BOTH files:\n"
            f"{_explain(failures)}")
        _assert_no_violations(_structure_violations(live_sources), precondition)


@pytest.mark.parametrize("plant", [
    "unknown", "duplicate", "missing", "annotated-attribute", "else", "later-expression",
    "later-find-spec"
])
def test_package_ownership_and_header_controls(live_sources, plant):
    """One planted edit per row, rejected by the ownership or header clause it names.

    `annotated-attribute` plants an import-time mutation written as an
    annotated assignment (`VOLUME_MOUNT.name: str = ...`). It must be rejected
    by `undeclared-assignment` ALONE: membership reads plain-name bindings
    only, and the header rule allows an AnnAssign, so deleting that clause
    would leave this row green.
    """
    filename = "training.py" if plant.startswith("later-") else "core.py"
    additions = {
        "unknown":
        "\n_ON_CONTAINER = 1\n",
        "duplicate":
        "\nVOLUME_NAME = 'duplicate'\n",
        "annotated-attribute":
        "\nVOLUME_MOUNT.name: str = 'volume'\n",
        "else":
        "\nif TYPE_CHECKING:\n    import decimal\nelse:\n    print('work')\n",
        "later-expression":
        "\nprint('work')\n",
        "later-find-spec":
        "\nif importlib.util.find_spec('modal') is not None:\n    _ON_CONTAINER = True\n",
    }
    clause = {
        "unknown": "membership",
        "missing": "membership",
        "duplicate": "duplicates",
        "annotated-attribute": "undeclared-assignment",
    }.get(plant, "header")
    precondition = _precondition("test_package_structure_contract")
    for _ in range(_CONTROL_PASSES):
        _assert_no_violations(_structure_violations(live_sources), precondition)
        changed = live_sources[filename] + additions.get(plant, "")
        if plant == "missing":
            changed = changed.replace(_segments(changed)["VOLUME_NAME"], "", 1)
        failures = _structure_violations({**live_sources, filename: changed})
        _assert_rejected(failures, (clause, filename), plant)
        if plant == "annotated-attribute":
            # The plant is the last line of `changed`, which ends in a newline.
            expected = [(clause, filename, changed.count("\n"))]
            assert failures == expected, (
                f"the {plant} plant must be rejected by exactly {expected}; the gates "
                f"reported:\n{_explain(failures)}")
        if plant == "else":
            assert any("no else branch" in str(v) for v in failures), (
                f"the else plant must be named by the no-else rule:\n{_explain(failures)}")
        _assert_no_violations(_structure_violations(live_sources), precondition)


# Each row pairs a plant with the exact violation list it must produce. Its ID is
# the plant plus the verdict DERIVED from that list ("-accepted" when it is
# empty), so a collected ID reads as the claim ("...[dropped-from-both-accepted]")
# and no edit or reordering of the rows can make an ID misstate its assertion.
@pytest.mark.parametrize("plant, expected", [
    pytest.param(plant, expected, id=f"{plant}-{'rejected' if expected else 'accepted'}")
    for plant, expected in [
        ("dropped-from-both", []),
        ("added-to-both", []),
        ("two-owners", [("manifest-duplicates", {
            "VOLUME_NAME": ["core.py", "request.py"]
        })]),
        ("declared-grab-bag", [("forbidden-name", "utils.py")]),
    ]
])
def test_package_manifest_controls(live_sources, plant, expected):
    """MANIFEST edited TOGETHER with the modules: the membership remedy, and the clauses it leaves.

    Every plant keeps each file equal to its own manifest entry, so membership
    has nothing to report, and the exact violation list says which clause, if
    any, still objects (each row's test ID ends in the verdict it asserts):

    * dropped-from-both: `REASON_ERROR` leaves MANIFEST['training.py'] and training.py.
      ACCEPTED: deleting a symbol and its entry is the membership message's
      remedy, and no other clause may hold it red.
    * added-to-both: a new constant is declared in MANIFEST['core.py'] and
      defined in core.py. ACCEPTED, for the same reason.
    * two-owners: `VOLUME_NAME` is declared and defined in both core.py and
      request.py, byte-identical in each -> `manifest-duplicates` alone.
    * declared-grab-bag: an empty `utils.py` is declared in MANIFEST and added
      to the sources -> `forbidden-name` alone. The population clause (`files`)
      is satisfied, so this is the case the name ban exists for.
    """
    manifest = {filename: list(names) for filename, names in MANIFEST.items()}
    changed = dict(live_sources)
    if plant == "dropped-from-both":
        manifest["training.py"].remove("REASON_ERROR")
        changed["training.py"] = live_sources["training.py"].replace(
            _segments(live_sources["training.py"])["REASON_ERROR"], "", 1)
    elif plant == "added-to-both":
        manifest["core.py"].append("POLL_JITTER_SECONDS")
        changed["core.py"] += "\nPOLL_JITTER_SECONDS = 0.5\n"
    elif plant == "two-owners":
        manifest["request.py"].append("VOLUME_NAME")
        changed["request.py"] += "\n" + _segments(live_sources["core.py"])["VOLUME_NAME"]
    elif plant == "declared-grab-bag":
        manifest["utils.py"] = []
        changed["utils.py"] = ""
    else:
        pytest.fail(f"the {plant} row has no plant")
    assert changed != live_sources, f"the {plant} plant did not change the sources"
    precondition = _precondition("test_package_structure_contract")
    for _ in range(_CONTROL_PASSES):
        _assert_no_violations(_structure_violations(live_sources), precondition)
        violations = _structure_violations(changed, manifest)
        verdict = f"rejected by exactly {[v[0] for v in expected]}" if expected else "accepted"
        assert violations == expected, (
            f"the {plant} plant must be {verdict}; the gates reported:\n{_explain(violations)}")
        _assert_no_violations(_structure_violations(live_sources), precondition)


@pytest.mark.parametrize("spelling", ["TYPE_CHECKING", "typing.TYPE_CHECKING", "os.TYPE_CHECKING"])
def test_package_header_accepts_import_only_type_checking(live_sources, spelling):
    _assert_no_violations(_structure_violations(live_sources),
                          _precondition("test_package_structure_contract"))
    sources = dict(live_sources)
    sources["core.py"] += f"\nif {spelling}:\n    import decimal\n"
    _assert_no_violations(_structure_violations(sources),
                          f"a legal imports-only `if {spelling}:` block in core.py")


# Each row plants one shape into the live sources, in memory: a `VOLUME_NAME =`
# row replaces that constant's value in core.py, `drop-future` deletes
# source.py's `from __future__ import annotations`, and any other row is
# appended to training.py. `expected` is what the `import-time` clause alone must
# report, since a planted name also trips membership. The ID ends in the verdict
# the row asserts.
@pytest.mark.parametrize("plant, expected", [
    pytest.param(plant, expected, id=f"{name}-{'rejected' if expected else 'accepted'}")
    for name, plant, expected in [
        ("find-spec-in-value", 'VOLUME_NAME = importlib.util.find_spec("modal")',
         ["a call to `importlib.util.find_spec`"]),
        ("walrus-in-value", 'VOLUME_NAME = (_ON_CONTAINER := "cs2rl-training-artifacts")',
         ["a `NamedExpr` node"]),
        ("conditional-in-value", 'VOLUME_NAME = "a" if TYPE_CHECKING else "b"', ["a `IfExp` node"]),
        ("environment-in-value", 'VOLUME_NAME = os.environ["CS2RL_VOLUME"]',
         ["a `Subscript` node"]),
        ("comprehension-in-value", 'VOLUME_NAME = [c for c in "ab"]', ["a `ListComp` node"]),
        ("decorator", "@atexit.register\ndef _plant():\n    pass\n",
         ["decorator `atexit.register`"]),
        ("decorator-call", "@functools.lru_cache(maxsize=1)\ndef _plant():\n    pass\n",
         ["a call to `functools.lru_cache`"]),
        ("default-argument", 'def _plant(flag=os.environ.get("X")):\n    return flag\n',
         ["a call to `os.environ.get`"]),
        ("class-body", 'class _Plant:\n    X = print("work")\n', ["a call to `print`"]),
        ("class-base", 'class _Plant(type("B", (), {})):\n    pass\n', ["a call to `type`"]),
        ("method-decorator",
         "class _Plant:\n    @atexit.register\n    def m(self):\n        pass\n",
         ["decorator `atexit.register`"]),
        ("drop-future", None, ["an annotation (no `from __future__ import annotations`)"]),
        ("function-body", 'def _plant():\n    return importlib.util.find_spec("modal")\n', []),
        ("attribute-default", "def _plant(run=subprocess.run):\n    return run\n", []),
        ("allowed-constructors", "_PLANT = frozenset({re.compile('x'), timedelta(seconds=1)})\n",
         []),
        ("deferred-annotation", "_PLANT: dict[str, int] = {}\n", []),
    ]
])
def test_package_import_time_controls(live_sources, plant, expected):
    """One plant per row, judged by the `import-time` clause alone.

    * The five `-in-value` rows hide work in an existing constant, the case
      gh#223 item 7 names: before this clause each one passed every gate,
      while the same probe as a bare statement was caught by the header rule.
      Each rejected kind is one `_IMPORT_TIME_KINDS` leaves out, or one call
      IMPORT_TIME_CALLS does not list.
    * decorator, decorator-call, default-argument, class-body, class-base and
      method-decorator are the other places a declaration runs code on import.
      Each is a root `_import_time_roots` must reach; drop one and its row
      goes green.
    * drop-future: source.py annotates its functions, so without the
      `__future__` import those annotations would run on import.
    * The accepted rows are the shapes the package itself uses: a call in a
      function body, an attribute read as a default argument, allowed
      constructors, and a deferred subscript annotation. A clause that
      rejected them would redden the live package.
    """
    filename = ("core.py" if plant and plant.startswith("VOLUME_NAME =") else
                "source.py" if plant is None else "training.py")
    live = live_sources[filename]
    if plant is None:
        changed = live.replace("from __future__ import annotations\n", "", 1)
        lines = range(1, changed.count("\n") + 1)
    elif filename == "core.py":
        original = 'VOLUME_NAME = "cs2rl-training-artifacts"'
        changed = live.replace(original, plant, 1)
        line = live[:live.index(original)].count("\n") + 1
        lines = range(line, line + 1)
    else:
        changed = live + "\n" + plant
        lines = range(live.count("\n") + 2, changed.count("\n") + 1)
    assert changed != live, f"the plant {plant!r} did not change {filename}"
    precondition = _precondition("test_package_structure_contract")
    for _ in range(_CONTROL_PASSES):
        _assert_no_violations(_structure_violations(live_sources), precondition)
        violations = _structure_violations({**live_sources, filename: changed})
        reported = [v for v in violations if v[0] == "import-time"]
        verdict = f"rejected by exactly {expected}" if expected else "accepted"
        assert [v[3] for v in reported] == expected and all(
            v[1] == filename and v[2] in lines for v in reported), (
                f"the {plant!r} plant in {filename} must be {verdict}, on lines {lines}; the "
                f"gates reported:\n{_explain(violations)}")
        _assert_no_violations(_structure_violations(live_sources), precondition)


def test_package_dependency_contract(live_sources):
    _assert_no_violations(_dependency_violations(live_sources), "the live package's import graph")


@pytest.mark.parametrize("plant", ["runtime-annotation", "core-edge", "missing-edge", "extra-edge"])
def test_package_dependency_controls(live_sources, plant):
    precondition = _precondition("test_package_dependency_contract",
                                 "the live package's import graph")
    for _ in range(_CONTROL_PASSES):
        _assert_no_violations(_dependency_violations(live_sources), precondition)
        changed = dict(live_sources)
        module = "commands" if plant == "runtime-annotation" else "core" if plant == "core-edge" else "source"
        if plant == "runtime-annotation":
            changed["commands.py"] = _hoist_type_checking_imports(changed["commands.py"])
        elif plant == "missing-edge":
            lines = changed[module + ".py"].splitlines(keepends=True)
            imports = [
                n for n in ast.parse(changed[module + ".py"]).body
                if isinstance(n, ast.ImportFrom) and n.level
            ]
            for node in reversed(imports):
                del lines[node.lineno - 1:node.end_lineno]
            changed[module + ".py"] = "".join(lines)
        else:
            edge = "request" if plant == "runtime-annotation" else "training"
            changed[module + ".py"] += f"\nfrom . import {edge}\n"
        _assert_rejected(_dependency_violations(changed), ("runtime-edges", module), plant)
        _assert_no_violations(_dependency_violations(live_sources), precondition)


@pytest.mark.parametrize("statement, edge", [
    pytest.param("import scripts.modal_runner\n", PACKAGED, id="import-package"),
    pytest.param("import scripts.modal_runner as _pkg\n", PACKAGED, id="import-package-as"),
    pytest.param("from scripts import modal_runner\n", PACKAGED, id="from-parent-import"),
    pytest.param(
        "from scripts.modal_runner import VOLUME_NAME\n", PACKAGED, id="from-package-import-name"),
    pytest.param(
        "from scripts.modal_runner import training\n", PACKAGED, id="from-package-import-module"),
    pytest.param("from scripts.modal_runner.training import _tee_stream\n",
                 PACKAGED,
                 id="from-submodule-import"),
    pytest.param(
        "def _lazy():\n    import importlib\n"
        "    return importlib.import_module('scripts.modal_runner.training')\n",
        PACKAGED,
        id="import-module-literal-in-body"),
    pytest.param("from .. import modal_runner\n", "..", id="parent-relative-package"),
    pytest.param(
        "from ..modal_runner import training\n", "..modal_runner", id="parent-relative-submodule"),
])
def test_package_dependency_rejects_every_self_import_spelling_in_core(
        live_sources, statement, edge):
    """The leaf clause (core imports no package module) against every spelling of the package.

    core.py must import no package module. Each row appends one spelling to
    core.py and requires the exact edge it resolves to, with no other clause
    firing: the old walker resolved only one-dot relative imports and
    `scripts.modal_runner[.x]` from-imports and dotted imports, so the
    package-level, parent and `..` spellings and the literal `import_module`
    all left core looking like a leaf.
    """
    precondition = _precondition("test_package_dependency_contract",
                                 "the live package's import graph")
    for _ in range(_CONTROL_PASSES):
        _assert_no_violations(_dependency_violations(live_sources), precondition)
        changed = {**live_sources, "core.py": live_sources["core.py"] + "\n" + statement}
        failures = _dependency_violations(changed)
        expected = [("runtime-edges", "core", {edge}, set())]
        assert failures == expected, (
            f"core.py importing the package as {statement!r} must be reported as the single "
            f"runtime edge {edge!r}; the gates reported:\n{_explain(failures)}")
        _assert_no_violations(_dependency_violations(live_sources), precondition)


@pytest.mark.parametrize("plant", ["preflight-to-runtime", "extra-annotation", "typing-attribute"])
def test_package_annotation_edge_controls(live_sources, plant):
    """The `annotation-edges` clause, which no runtime-edge plant can hold on its own.

    * preflight-to-runtime: preflight's `TYPE_CHECKING` import of request moves
      to module scope, the preflight counterpart of the commands control above.
      Both of preflight's clauses must fire.
    * extra-annotation: source.py gains an annotation-only import of training.
      Its runtime edges are unchanged, so `annotation-edges` is the sole
      objector, and deleting that clause would leave this row green.
    * typing-attribute: the same plant under `if typing.TYPE_CHECKING:`.
      `_dependency_edges` recognises the block by its attribute spelling as
      well as by the bare name; drop the attribute half and the import becomes
      a runtime edge, which this row rejects. Nothing else in this file uses
      that spelling on a plant whose edges are compared, so before this row
      that knock-out left every test here green.
    """
    if plant == "preflight-to-runtime":
        changed = {
            **live_sources, "preflight.py":
            _hoist_type_checking_imports(live_sources["preflight.py"])
        }
        # Read from the tables, so a new annotation-only import in preflight,
        # declared as every message says, keeps this row green.
        runtime = set(DEPENDENCIES["preflight"])
        annotated = set(ANNOTATION_DEPENDENCIES["preflight"])
        expected = [("runtime-edges", "preflight", runtime | annotated, runtime),
                    ("annotation-edges", "preflight", set(), annotated)]
    else:
        guard = "typing.TYPE_CHECKING" if plant == "typing-attribute" else "TYPE_CHECKING"
        changed = {
            **live_sources, "source.py":
            live_sources["source.py"] + f"\nif {guard}:\n    from .training import _tee_stream\n"
        }
        expected = [("annotation-edges", "source", {"training"}, set())]
    precondition = _precondition("test_package_dependency_contract",
                                 "the live package's import graph")
    for _ in range(_CONTROL_PASSES):
        _assert_no_violations(_dependency_violations(live_sources), precondition)
        failures = _dependency_violations(changed)
        assert failures == expected, (
            f"the {plant} plant must be rejected by exactly {[v[:2] for v in expected]}; the "
            f"gates reported:\n{_explain(failures)}")
        _assert_no_violations(_dependency_violations(live_sources), precondition)


def test_package_seam_contract(live_sources):
    """The `seam` clauses on the live package, and the facade docstring's seam list."""
    _assert_no_violations(_seam_violations(live_sources), "the live package's qualified seams")
    facade = (ROOT / "scripts" / "modal_runner" / "__init__.py").read_text(encoding="utf-8")
    listed = _facade_seam_list(facade)
    assert listed == set(QUALIFIED_SEAMS), (
        f"the QUALIFIED SEAMS paragraph of scripts/modal_runner/__init__.py lists "
        f"{sorted(listed)}, but QUALIFIED_SEAMS in tests/modal_runner_tables.py declares "
        f"{sorted(QUALIFIED_SEAMS)}. Update the two together.")


# One from-import plant per seam, in one of its readers: the reader's `owner.name`
# reads become bare names, and `from .owner import name` is appended.
_SEAM_FROM_IMPORT_PLANTS = {
    "core.PREBUILT_PYTHON": "commands.py",
    "core.sha256_file": "source.py",
    "state.transition_status": "preflight.py",
    "checkpoint.validate_local_checkpoint": "training.py",
}


@pytest.mark.parametrize("plant", [
    *(f"from-import-{seam}" for seam in _SEAM_FROM_IMPORT_PLANTS), "import-time-alias",
    "import-time-default", "undeclared", "reader-dropped", "aliased-module", "star-import"
])
def test_package_seam_controls(live_sources, plant):
    """One plant per row, rejected by exactly the `seam` violations it names.

    * from-import-<seam>: the rewrite gh#221 measured green on every gate.
      Rejected twice: the from-import itself, and the reader leaving the seam's
      reader set.
    * import-time-alias, import-time-default: `_HASH = core.sha256_file` and
      `def f(hasher=core.sha256_file)` in training, which already reads the
      seam at call time. Both bind the object on import; the reader set is
      unchanged, so only `seam-import-time` fires.
    * undeclared: training reads `state.write_heartbeat` through the module
      object, a name that is not a seam.
    * reader-dropped: commands stops reading `core.PREBUILT_PYTHON`, so its
      entry lists a reader the code no longer has.
    * aliased-module: `from . import core as _core` in a function body of
      state, then `_core.sha256_file`. The clause follows the alias, so state
      becomes an undeclared reader.
    * star-import: `from .core import *` in training binds copies of both core
      seams.
    """
    changed = dict(live_sources)

    def append(filename, text):
        """Append `text` to `filename`; return the line its first line lands on."""
        changed[filename] = live_sources[filename] + "\n" + text
        return live_sources[filename].count("\n") + 2

    if plant.startswith("from-import-"):
        seam = plant.removeprefix("from-import-")
        filename = _SEAM_FROM_IMPORT_PLANTS[seam]
        module, (owner, name) = filename.removesuffix(".py"), seam.split(".")
        assert seam in live_sources[filename], f"{filename} no longer reads {seam}"
        changed[filename] = live_sources[filename].replace(seam, name)
        changed[filename] += f"\nfrom .{owner} import {name}\n"
        declared = sorted(QUALIFIED_SEAMS[seam])
        expected = [("seam-from-import", module, changed[filename].count("\n"), seam),
                    ("seam-readers", seam, sorted(set(declared) - {module}), declared)]
    elif plant == "import-time-alias":
        line = append("training.py", "_HASH = core.sha256_file\n")
        expected = [("seam-import-time", "training", line, "core.sha256_file")]
    elif plant == "import-time-default":
        line = append("training.py", "def _plant(hasher=core.sha256_file):\n    return hasher\n")
        expected = [("seam-import-time", "training", line, "core.sha256_file")]
    elif plant == "undeclared":
        line = append("training.py", "def _plant():\n    return state.write_heartbeat\n")
        expected = [("seam-undeclared", "training", line + 1, "state.write_heartbeat")]
    elif plant == "reader-dropped":
        changed["commands.py"] = live_sources["commands.py"].replace(
            "core.PREBUILT_PYTHON", repr("/opt/cs2rl/.venv/bin/python"))
        declared = sorted(QUALIFIED_SEAMS["core.PREBUILT_PYTHON"])
        expected = [("seam-readers", "core.PREBUILT_PYTHON", ["checkpoint"], declared)]
    elif plant == "aliased-module":
        append("state.py", "def _plant():\n    from . import core as _core\n"
               "    return _core.sha256_file\n")
        declared = sorted(QUALIFIED_SEAMS["core.sha256_file"])
        expected = [("seam-readers", "core.sha256_file", sorted({*declared, "state"}), declared)]
    elif plant == "star-import":
        line = append("training.py", "from .core import *\n")
        expected = [("seam-from-import", "training", line, "core.*")]
    else:
        pytest.fail(f"the {plant} row has no plant")
    assert changed != live_sources, f"the {plant} plant did not change the sources"
    precondition = _precondition("test_package_seam_contract", "the live package's seams")
    for _ in range(_CONTROL_PASSES):
        _assert_no_violations(_seam_violations(live_sources), precondition)
        violations = _seam_violations(changed)
        assert violations == expected, (
            f"the {plant} plant must be rejected by exactly {expected}; the gate reported:\n"
            f"{_explain(violations)}")
        _assert_no_violations(_seam_violations(live_sources), precondition)


@pytest.mark.parametrize("stem", ["utils", "telemetry"])
def test_package_gates_see_an_undeclared_module_on_disk(tmp_path, stem):
    """A REAL undeclared module, with `import modal` and module-scope work, in a package copy.

    Before the gates enumerated the directory, this file passed every one of
    them (structural review, 2026-09-22). Each gate that should object must:
    the file population (`files`), the header shape, the dependency table
    (`undeclared-module`) and import purity. `utils` is also a forbidden name;
    `telemetry` is not, so its row shows the population clauses fire on any
    undeclared file, not only on a grab-bag name. The `utils` row is also the
    `forbidden-name` clause's control for a file read from disk; the
    declared-grab-bag-rejected row of `test_package_manifest_controls` shows that
    clause objecting alone.
    """
    package = _copy_live_package(tmp_path)
    rel = f"scripts/modal_runner/{stem}.py"
    declared = [f"scripts/modal_runner/{name}.py" for name in RUNNER_MODULES]
    contracts = "test_package_structure_contract or test_package_dependency_contract"
    purity_contract = "test_package_import_purity_scans_every_module"

    def gates():
        """Every gate's verdict on the copy, read back from disk."""
        sources = _package_sources(tmp_path)
        return (sources, _structure_violations(sources), _dependency_violations(sources),
                _nonstdlib_module_scope_imports(tmp_path))

    for _ in range(_CONTROL_PASSES):
        sources, structure, dependency, purity = gates()
        _assert_no_violations(structure + dependency, _precondition(contracts, "the package copy"))
        assert purity == (declared, []), (
            f"{_precondition(purity_contract, 'gate (e) on the package copy')} reported {purity}. "
            f"{_UNDECLARED_MODULE_REMEDY}")
        (package / f"{stem}.py").write_text("import modal\nprint('work')\n", encoding="utf-8")
        undeclared = _runner_module_population(tmp_path)[1]
        assert undeclared == [rel], f"the population found {undeclared}, not the planted {rel}"
        sources, structure, dependency, purity = gates()
        assert list(sources)[-1] == f"{stem}.py", f"the gates were not handed {rel}"
        _assert_rejected(structure, ("files", [f"{stem}.py"], []), rel)
        _assert_rejected(structure, ("header", f"{stem}.py"), rel)
        _assert_rejected(dependency, ("undeclared-module", stem), rel)
        assert (("forbidden-name", f"{stem}.py") in structure) == (stem == "utils"), (
            "the forbidden-name clause must fire for utils.py and only for it:\n"
            f"{_explain(structure)}")
        expected = (declared + [rel], [(rel, 1, "modal")])
        assert purity == expected, (
            f"gate (e) must scan {rel} and report its `import modal`; it reported {purity}")
        (package / f"{stem}.py").unlink()
    assert _package_sources(tmp_path) == _package_sources(ROOT), "the copy was not restored"


def test_package_import_purity_scans_every_module():
    scanned, violations = _nonstdlib_module_scope_imports(ROOT)
    declared = [f"scripts/modal_runner/{name}.py" for name in RUNNER_MODULES]
    assert scanned == declared, f"gate (e) read {scanned}. {_UNDECLARED_MODULE_REMEDY}"
    assert violations == [], (
        f"these (file, line, module) imports execute on import and are not stdlib: {violations}. "
        "Move them into the function that needs them, or under `if TYPE_CHECKING:`.")


@pytest.mark.parametrize("container", ["if", "with", "except", "class", "try"])
def test_package_import_purity_recursive_controls(tmp_path, live_sources, container):
    package = _copy_live_package(tmp_path)
    plants = {
        "if": "if True:\n    import modal\n",
        "with": "with open('x'):\n    import modal\n",
        "except": "try:\n    pass\nexcept Exception:\n    import modal\n",
        "class": "class Plant:\n    import modal\n",
        "try": "try:\n    import modal\nexcept ImportError:\n    pass\n"
    }
    target = package / "training.py"
    declared = [f"scripts/modal_runner/{name}.py" for name in RUNNER_MODULES]
    purity_contract = "test_package_import_purity_scans_every_module"
    for _ in range(_CONTROL_PASSES):
        scanned, failures = _nonstdlib_module_scope_imports(tmp_path)
        assert scanned == declared and failures == [], (
            f"{_precondition(purity_contract, 'gate (e) on the package copy')} read {scanned} and "
            f"reported {failures}. {_UNDECLARED_MODULE_REMEDY}")
        target.write_text(live_sources["training.py"] + "\n" + plants[container], encoding="utf-8")
        scanned, failures = _nonstdlib_module_scope_imports(tmp_path)
        assert scanned == declared, (
            f"gate (e) read {scanned}, not the {len(declared)} modules RUNNER_MODULES declares")
        assert len(failures) == 1 and failures[0][
            0] == "scripts/modal_runner/training.py" and failures[0][2] == "modal", (
                f"gate (e) must report the `import modal` inside the {container} container in "
                f"training.py, and only it; it reported {failures}")
        assert _nonstdlib_module_scope_imports(tmp_path, top_level_only=True)[1] == [], (
            f"the tree.body-only instrument was expected to be blind to the {container} plant")
        target.write_text(live_sources["training.py"], encoding="utf-8")
        assert _nonstdlib_module_scope_imports(tmp_path)[1] == [], "the copy was not restored"


@pytest.mark.parametrize("defect", ["missing", "unreadable"])
def test_package_scans_reject_missing_or_unreadable_member(tmp_path, live_sources, defect):
    package = _copy_live_package(tmp_path)
    target = package / "training.py"
    declared = [f"scripts/modal_runner/{name}.py" for name in RUNNER_MODULES]
    purity_contract = "test_package_import_purity_scans_every_module"
    for _ in range(_CONTROL_PASSES):
        copied = sorted(_package_sources(tmp_path))
        assert copied == sorted(Path(rel).name for rel in declared), (
            f"{_precondition('test_package_structure_contract', 'the package copy')} holds "
            f"{copied}, not the {len(declared)} modules RUNNER_MODULES declares, so this control "
            f"would not start from the live package. {_UNDECLARED_MODULE_REMEDY}")
        scanned = _nonstdlib_module_scope_imports(tmp_path)[0]
        assert scanned == declared, (
            f"{_precondition(purity_contract, 'gate (e)')} read {scanned} from the package copy, "
            f"not the {len(declared)} modules RUNNER_MODULES declares. {_UNDECLARED_MODULE_REMEDY}")
        target.unlink()
        if defect == "unreadable":
            target.mkdir()
        for scanner in (_package_sources, _nonstdlib_module_scope_imports):
            with pytest.raises(OSError, match="training.py"):
                scanner(tmp_path)
        if defect == "unreadable":
            target.rmdir()
        target.write_text(live_sources["training.py"], encoding="utf-8")
        assert _package_sources(tmp_path) == live_sources, "the copy was not restored"
        assert _nonstdlib_module_scope_imports(tmp_path)[1] == [], "the copy was not restored"
