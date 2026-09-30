"""Static census of the original patch sites that install through `binding_target`.

The campaign (tests/modal/test_modal_patch_bindings.py and
tests/modal/modal_patch_binding_campaign.py) proves that `binding_target(site)`
reaches each consumer at run time. This file proves, by reading source, that
the ORIGINAL tests install their patches through `binding_target(site)` and
not through a facade patch that would bypass it. It imports nothing from the
runner package, so it is not one of the runner's importers.
"""
import ast

import pytest

from tests.conftest import REPO_ROOT
from tests.modal.modal_patch_binding_campaign import BINDING_SITES
from tests.modal.modal_runner_tables import RUNNER_TEST_FILES

# The seam's file-name constants, from the seam gate, as
# tests/modal/test_modal_runner_package_shape.py imports them: that module imports only
# the stdlib, pytest, the tables and the campaign's data at module scope, so this
# pulls in no runner module and keeps this file out of the runner's importers.
from tests.modal.test_modal_packaging import CLIENT_FILE, SHARED_FILE

# Repository root: the census reads the original sites' files by repo-relative
# path.

# The ORIGINAL installer of each BINDING_SITES key: the test file, and the
# module-level function whose body calls `binding_target(site)`. The campaign
# certifies `binding_target(site)` through the companion; this table ties that
# certificate to the original sites. `_binding_site_violations` checks it.
_ORIGINAL_SITES = {
    "prepare-validator": ("tests/modal/test_modal_preflight.py",
                          "test_prepare_validates_resume_then_dumps_and_hashes_config"),
    "fallback-loader": ("tests/modal/modal_test_helpers.py", "_no_torch"),
    "fallback-python": ("tests/modal/modal_test_helpers.py", "_no_torch"),
    "interrupt-loader": ("tests/modal/test_modal_training.py",
                         "test_interrupt_commits_status_even_if_prebuilt_load_hangs"),
    "watcher-publisher": ("tests/modal/test_modal_training.py",
                          "test_checkpoint_watcher_threads_generation_into_last_published"),
    "terminal-validator": ("tests/modal/test_modal_training.py", "_record_checkpoint_reads"),
    "terminal-hasher": ("tests/modal/test_modal_training.py", "_record_checkpoint_reads"),
    "attempt-watcher":
    ("tests/modal/test_modal_training.py", "test_checkpoint_watcher_stops_before_terminal_status"),
    "attempt-transition":
    ("tests/modal/test_modal_training.py", "test_checkpoint_watcher_stops_before_terminal_status"),
    "client-mount": ("tests/modal/test_modal_client.py",
                     "test_train_remote_completes_against_post_dump_manifest_hash"),
}


def _facade_patches(function, facade):
    """Names `function` patches on the package facade: one entry per match below, nothing else.

    Matching is syntactic and deliberately broad, so it errs toward reporting.
    A call's callee is identified by its bare name or its last attribute, so
    `setattr` means `setattr`, `monkeypatch.setattr`, `mp.setattr` and so on:

    * a callee named `setattr` whose first positional argument is `<facade>`
      and which has a second positional argument: that argument (the name);
    * a callee named `multiple` (`mock.patch.multiple`,
      `mocker.patch.multiple`, ...) whose first positional argument is
      `<facade>`: each keyword's name;
    * a callee named `setattr` or `patch` (pytest's dotted form, `mock.patch`,
      `unittest.mock.patch`, `mocker.patch`, a bare `patch`) whose first
      positional argument is a string literal starting `scripts.modal_runner.`:
      the rest of the string, so a submodule target such as
      `scripts.modal_runner.checkpoint.<name>` comes back as
      `checkpoint.<name>`, which equals no site symbol;
    * an assignment, plain, augmented or annotated, whose target contains
      `<facade>.<attr>` at any depth: `<attr>`. That covers tuple and starred
      targets, and an attribute or item OF a facade attribute:
      `mrl.validate_local_checkpoint.__doc__ = ...` is reported as patching
      `validate_local_checkpoint` although it does not replace it, and
      `mrl.__dict__[...] = ...` is reported as `__dict__`, which equals no
      site symbol.

    Known false positives, each reported as patching `<attr>` and so failing
    closed, include: a `<facade>.<attr>` anywhere inside a target without
    being the target itself (an attribute or item of it,
    `mrl.<attr>.__doc__ = ...` or `mrl.<attr>[k] = ...`; a subscript key,
    `d[mrl.<attr>] = ...`; a callee, `mrl.<attr>().x = ...`); a bare
    annotation (`mrl.<attr>: T`), which assigns nothing; a write in a nested
    function the installer never calls, because the whole body is walked; and
    a write through a name the function rebinds locally, because the alias
    set is file-wide.

    `facade` is the set of spellings bound to the package: an alias such as
    `mrl` or `modal_runner`, and always the dotted `scripts.modal_runner`.
    `<facade>` is matched on `ast.unparse` of the expression, so the dotted
    spelling is seen as well as a bare name.

    A `setattr` name argument that is not a literal (`setattr(mrl,
    validate_name, ...)`) and a `**` keyword to `multiple` are returned as
    `None`: they patch the facade, but which symbol cannot be read from source.
    """
    found = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            called = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(
                node.func, "id", None)
            target = node.args[0] if node.args else None
            on_facade = target is not None and ast.unparse(target) in facade
            if called == "setattr" and on_facade and len(node.args) > 1:
                name = node.args[1]
                found.append(name.value if isinstance(name, ast.Constant) else None)
            elif called == "multiple" and on_facade:
                found.extend(keyword.arg for keyword in node.keywords)
            elif (called in ("setattr", "patch") and isinstance(target, ast.Constant)
                  and isinstance(target.value, str)
                  and target.value.startswith("scripts.modal_runner.")):
                found.append(target.value.removeprefix("scripts.modal_runner."))
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            found.extend(sub.attr for target in targets for sub in ast.walk(target)
                         if isinstance(sub, ast.Attribute) and ast.unparse(sub.value) in facade)
    return found


def _binding_site_violations(sources):
    """Census the original binding sites in `sources` ({repo-relative path: text}).

    WHAT: every `binding_target(...)` call in those files must name a
    BINDING_SITES key as a string literal; each key must be called exactly once,
    inside the function `_ORIGINAL_SITES` names for it; and that function must
    not also patch the site's OWN symbol on the package facade
    (`_facade_patches`). Other facade patches are allowed: the client-mount
    installer patches `prepare_remote_source` and `execute_training_attempt`
    there. No statement in those files may patch the facade through a
    positional non-literal `setattr` name or `**` names to
    `mock.patch.multiple`, because the census cannot tell which symbol that
    patches (a non-literal dotted string is not matched at all; see
    PITFALLS). Returns the sorted violation strings; `[]` is green.

    WHY: `test_patch_binding_campaign` proves that `binding_target(site)` reaches
    each consumer, and nothing more. A review of the package split found that
    respelling the validator patch of `_record_checkpoint_reads` (then
    `_record_hash_after_terminal`) as a package patch stayed green everywhere:
    the facade exports `validate_local_checkpoint`, so there is no
    AttributeError, and that helper's callers asserted `hashed == []`, which a
    patch that never reaches the consumer satisfies trivially. Since gh#211
    the no-sidecar caller requires a recorded validation, so that respelling
    now fails it as well; the census still pins the spelling, because it names
    the defect where that failure names only a missing read.

    PITFALLS:
      * The facade spellings are the dotted `scripts.modal_runner`, always,
        plus the aliases each file's own imports bind (`import
        scripts.modal_runner as mrl`, `from scripts import modal_runner`), so
        a renamed alias is still seen. The dotted spelling needs no import of
        its own: any `import scripts.<x>` binds `scripts`, through which it
        reaches the package, and it can name nothing else.
      * A non-literal key (`binding_target(site)`) is reported, not skipped: the
        census cannot tell which site such a call installs.
      * The seam's whole file set is read (`_original_site_sources`), not only
        the files `_ORIGINAL_SITES` names, so a second installer, a misspelt
        key or a non-literal facade patch in a runner test file that installs
        no site is reported too. The companion in this file calls
        `binding_target` too, by design, and is not read.
      * This reads source. It proves that each installer calls
        `binding_target(site)`, that the installer's own body does not patch
        its site's symbol on the facade in any spelling `_facade_patches`
        lists, and that no statement in those files patches the facade
        through a positional non-literal `setattr` name or `**` names to
        `multiple`. It does not prove the patch USES that call's result: a
        dead `binding_target(site)` call stays green beside a real patch that
        `_facade_patches` does not report under the site's symbol, whether it
        matches none of its shapes or reports another name. Among them, each
        measured green:
          - the name passed by keyword
            (`monkeypatch.setattr(mrl, name=..., value=...)`);
          - `mock.patch.object(<facade>, ...)`, and `mock.patch.multiple`
            given the package as a string
            (`mock.patch.multiple("scripts.modal_runner", ...)`);
          - a write through the module's `__dict__`
            (`mrl.__dict__[...] = ...`, reported as `__dict__`;
            `vars(mrl)[...] = ...`,
            `monkeypatch.setitem(vars(mrl), ...)`,
            `mock.patch.dict(mrl.__dict__, ...)`);
          - a wrong non-facade owner;
          - a facade patch made anywhere outside the installer's own body,
            such as in a helper it calls or a fixture it requests, because
            only that body is read;
          - a non-literal dotted string
            (`monkeypatch.setattr(f"scripts.modal_runner.{name}", ...)`);
          - the facade reached through a name no import binds
            (`importlib.import_module(...)`, `sys.modules[...]`, or a
            re-binding such as `pkg = mrl`).
        Nor does it prove a patch bites; the campaign is the bite evidence,
        and this is only the tie between the two.
    """
    violations, calls, functions = [], {}, {}
    for rel, source in sorted(sources.items()):
        tree = ast.parse(source)
        # The dotted `scripts.modal_runner` is always a facade spelling (see
        # PITFALLS). `_production_package_surface` (tests/modal/test_modal_client.py)
        # builds the same alias set its own way. Deliberately not shared: one
        # helper for both would be a new seam-governed name in a governed file,
        # so it is not a pure move.
        facade = {"scripts.modal_runner"} | {
            alias.asname or alias.name
            for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names if alias.name == "scripts.modal_runner"
        } | {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "scripts"
            for alias in node.names if alias.name == "modal_runner"
        }
        for statement in tree.body:
            owner = getattr(statement, "name", "<module>")
            functions[(rel, owner)] = (statement, facade)
            if None in _facade_patches(statement, facade):
                violations.append(f"non-literal facade patch name in {rel}::{owner}")
            for node in ast.walk(statement):
                if not (isinstance(node, ast.Call)
                        and getattr(node.func, "id", None) == "binding_target"):
                    continue
                key = node.args[0].value if node.args and isinstance(node.args[0],
                                                                     ast.Constant) else None
                if not isinstance(key, str):
                    violations.append(f"non-literal binding_target argument in {rel}::{owner}")
                elif key not in BINDING_SITES:
                    violations.append(f"unknown site {key!r} in {rel}::{owner}")
                else:
                    calls.setdefault(key, []).append(f"{rel}::{owner}")
    for site, (rel, owner) in _ORIGINAL_SITES.items():
        found = calls.get(site, [])
        if found != [f"{rel}::{owner}"]:
            violations.append(f"{site}: expected exactly one binding_target({site!r}) call, in "
                              f"{rel}::{owner}; found {found}")
        if (rel, owner) not in functions:
            violations.append(f"{site}: declared installer {rel}::{owner} is not defined")
            continue
        statement, facade = functions[(rel, owner)]
        symbol = BINDING_SITES[site][1]
        if symbol in _facade_patches(statement, facade):
            violations.append(f"{site}: {rel}::{owner} patches the package facade's {symbol}")
    return sorted(violations)


def _original_site_sources():
    """{repo-relative path: text} of the seam's whole file set: every runner test file, the shared
    helpers file and the client file.

    NOT the files `_ORIGINAL_SITES` names. Those were the whole unsplit runner test file until W4;
    after the split by module they are four files, and reading only them would drop most runner
    tests from the census, so a second installer, a misspelt key or a non-literal facade patch in
    any other runner test file would go unreported. No plant that targets an installer's own file
    can show that narrowing, which is why `test_patch_binding_sites_route_through_binding_target`
    asserts this set, and one plant targets a runner test file that installs nothing.
    """
    files = sorted({*RUNNER_TEST_FILES, SHARED_FILE, CLIENT_FILE})
    return {rel: (REPO_ROOT / rel).read_text(encoding="utf-8") for rel in files}


def test_patch_binding_sites_route_through_binding_target():
    """Each original installer calls `binding_target(site)` where the table says.

    None of them patches its own site's symbol on the facade in a spelling the
    census recognises; patching other facade symbols is allowed.
    `_facade_patches` lists every shape the census matches, and the known
    false positives, which fail closed. A facade patch through a positional
    non-literal `setattr` name, or `**` names to `multiple`, is rejected:
    which symbol it patches cannot be read. A patch it matches under another
    name, or does not match at all (a non-literal dotted string among them),
    passes unseen; `_binding_site_violations` lists the ones measured so.

    The campaign certifies `binding_target(site)`; this is what makes that
    certificate about the original tests. See `_binding_site_violations`.
    """
    assert set(_ORIGINAL_SITES) == set(BINDING_SITES)
    sources = _original_site_sources()
    assert set(sources) == set(RUNNER_TEST_FILES) | {SHARED_FILE, CLIENT_FILE}, (
        "the census no longer reads the seam's whole file set, so a site installed or respelled "
        f"in an unread file is invisible to it: {sorted(sources)}")
    assert _binding_site_violations(sources) == []


# Each plant rewrites exactly one original site in memory. `old` must occur
# exactly once in the named file, so a plant that stops matching fails loudly
# instead of passing on unchanged source. The `unaliased-import-*`,
# `dotted-via-other-import`, `from-parent-*` and `mock-patch-*` rows keep the
# site's `binding_target` call beside the facade patch, so the patch spelling
# under test is the only thing that can object. The two `from-parent-*` rows
# pin the census's `from scripts import modal_runner [as x]` aliases: before
# them, binding only the `as` form, or neither, left every row green (gh#222
# review F1).
#
# Keyed per file: each plant names the file its `old` text lives in, which is its
# installer's file (the training, preflight or shared helpers file) for the 15
# respelled sites. The 16th, `stray-call-in-a-non-installer-file`, adds a second
# `binding_target("terminal-hasher")` call to a runner test file that installs
# nothing: it is reported only if the census reads that file, so it is the plant
# that bites on a census narrowed back to the installers' own files.
_TRAINING_FILE = "tests/modal/test_modal_training.py"
_PREFLIGHT_FILE = "tests/modal/test_modal_preflight.py"
_REQUEST_FILE = "tests/modal/test_modal_request.py"
_SITE_CENSUS_PLANTS = {
    "package-revert":
    (_TRAINING_FILE,
     '    monkeypatch.setattr(*binding_target("terminal-validator"), wrapped_validate)\n',
     '    monkeypatch.setattr(mrl, "validate_local_checkpoint", wrapped_validate)\n', [
         "terminal-validator: expected exactly one binding_target('terminal-validator') call, "
         f"in {_TRAINING_FILE}::_record_checkpoint_reads; found []",
         f"terminal-validator: {_TRAINING_FILE}::_record_checkpoint_reads patches the "
         "package facade's validate_local_checkpoint",
     ]),
    "dotted-string-revert":
    (_TRAINING_FILE, '    monkeypatch.setattr(*binding_target("terminal-hasher"), wrapped_hash)\n',
     '    monkeypatch.setattr("scripts.modal_runner.sha256_file", wrapped_hash)\n', [
         "terminal-hasher: expected exactly one binding_target('terminal-hasher') call, "
         f"in {_TRAINING_FILE}::_record_checkpoint_reads; found []",
         f"terminal-hasher: {_TRAINING_FILE}::_record_checkpoint_reads patches the "
         "package facade's sha256_file",
     ]),
    "facade-store":
    (_PREFLIGHT_FILE, "    setattr(validate_target, validate_name, monkey_validate)\n",
     "    mrl.validate_local_checkpoint = monkey_validate\n", [
         f"prepare-validator: {_PREFLIGHT_FILE}::"
         "test_prepare_validates_resume_then_dumps_and_hashes_config patches the package "
         "facade's validate_local_checkpoint",
     ]),
    "unaliased-import-setattr":
    (_TRAINING_FILE,
     '    monkeypatch.setattr(*binding_target("terminal-validator"), wrapped_validate)\n',
     '    binding_target("terminal-validator")\n'
     '    import scripts.modal_runner\n'
     '    monkeypatch.setattr(scripts.modal_runner, "validate_local_checkpoint",\n'
     '                        wrapped_validate)\n', [
         f"terminal-validator: {_TRAINING_FILE}::_record_checkpoint_reads patches the "
         "package facade's validate_local_checkpoint",
     ]),
    "unaliased-import-store":
    (_PREFLIGHT_FILE, "    setattr(validate_target, validate_name, monkey_validate)\n",
     "    import scripts.modal_runner\n"
     "    scripts.modal_runner.validate_local_checkpoint = monkey_validate\n", [
         f"prepare-validator: {_PREFLIGHT_FILE}::"
         "test_prepare_validates_resume_then_dumps_and_hashes_config patches the package "
         "facade's validate_local_checkpoint",
     ]),
    "from-parent-bare":
    (_TRAINING_FILE,
     '    monkeypatch.setattr(*binding_target("terminal-validator"), wrapped_validate)\n',
     '    binding_target("terminal-validator")\n'
     '    from scripts import modal_runner\n'
     '    monkeypatch.setattr(modal_runner, "validate_local_checkpoint",\n'
     '                        wrapped_validate)\n', [
         f"terminal-validator: {_TRAINING_FILE}::_record_checkpoint_reads patches the "
         "package facade's validate_local_checkpoint",
     ]),
    "from-parent-as":
    (_TRAINING_FILE,
     '    monkeypatch.setattr(*binding_target("terminal-validator"), wrapped_validate)\n',
     '    binding_target("terminal-validator")\n'
     '    from scripts import modal_runner as runner\n'
     '    monkeypatch.setattr(runner, "validate_local_checkpoint", wrapped_validate)\n', [
         f"terminal-validator: {_TRAINING_FILE}::_record_checkpoint_reads patches the "
         "package facade's validate_local_checkpoint",
     ]),
    "dotted-via-other-import":
    (_TRAINING_FILE,
     '    monkeypatch.setattr(*binding_target("terminal-validator"), wrapped_validate)\n',
     '    binding_target("terminal-validator")\n'
     '    import scripts.modal_artifacts\n'
     '    monkeypatch.setattr(scripts.modal_runner, "validate_local_checkpoint",\n'
     '                        wrapped_validate)\n', [
         f"terminal-validator: {_TRAINING_FILE}::_record_checkpoint_reads patches the "
         "package facade's validate_local_checkpoint",
     ]),
    "mock-patch-dotted-string":
    (_TRAINING_FILE, '    monkeypatch.setattr(*binding_target("terminal-hasher"), wrapped_hash)\n',
     '    binding_target("terminal-hasher")\n'
     '    mock.patch("scripts.modal_runner.sha256_file", wrapped_hash).start()\n', [
         f"terminal-hasher: {_TRAINING_FILE}::_record_checkpoint_reads patches the "
         "package facade's sha256_file",
     ]),
    "mock-patch-multiple":
    (_TRAINING_FILE,
     '    monkeypatch.setattr(*binding_target("terminal-validator"), wrapped_validate)\n',
     '    binding_target("terminal-validator")\n'
     '    mock.patch.multiple(mrl, validate_local_checkpoint=wrapped_validate).start()\n', [
         f"terminal-validator: {_TRAINING_FILE}::_record_checkpoint_reads patches the "
         "package facade's validate_local_checkpoint",
     ]),
    "mock-patch-multiple-non-literal":
    (_TRAINING_FILE,
     '    monkeypatch.setattr(*binding_target("terminal-validator"), wrapped_validate)\n',
     '    binding_target("terminal-validator")\n'
     '    mock.patch.multiple(mrl, **{"validate_local_checkpoint": wrapped_validate}).start()\n', [
         f"non-literal facade patch name in {_TRAINING_FILE}::_record_checkpoint_reads",
     ]),
    "misspelled":
    (_TRAINING_FILE, 'binding_target("terminal-hasher")', 'binding_target("terminal-hashr")', [
        "terminal-hasher: expected exactly one binding_target('terminal-hasher') call, "
        f"in {_TRAINING_FILE}::_record_checkpoint_reads; found []",
        f"unknown site 'terminal-hashr' in {_TRAINING_FILE}::_record_checkpoint_reads",
    ]),
    "duplicated":
    (SHARED_FILE, 'binding_target("fallback-python")', 'binding_target("fallback-loader")', [
        "fallback-loader: expected exactly one binding_target('fallback-loader') call, "
        f"in {SHARED_FILE}::_no_torch; found ['{SHARED_FILE}::_no_torch', "
        f"'{SHARED_FILE}::_no_torch']",
        "fallback-python: expected exactly one binding_target('fallback-python') call, "
        f"in {SHARED_FILE}::_no_torch; found []",
    ]),
    "facade-restore-non-literal":
    (_PREFLIGHT_FILE, "        setattr(validate_target, validate_name, orig_validate)\n",
     "        setattr(mrl, validate_name, orig_validate)\n", [
         "non-literal facade patch name in "
         f"{_PREFLIGHT_FILE}::test_prepare_validates_resume_then_dumps_and_hashes_config",
     ]),
    "non-literal": (_TRAINING_FILE, 'binding_target("interrupt-loader")', "binding_target(site)", [
        "interrupt-loader: expected exactly one binding_target('interrupt-loader') call, "
        f"in {_TRAINING_FILE}::test_interrupt_commits_status_even_if_prebuilt_load_hangs; "
        "found []",
        "non-literal binding_target argument in "
        f"{_TRAINING_FILE}::test_interrupt_commits_status_even_if_prebuilt_load_hangs",
    ]),
    "stray-call-in-a-non-installer-file":
    (_REQUEST_FILE, "def test_valid_run_ids_are_accepted(run_id):\n",
     "def test_valid_run_ids_are_accepted(run_id):\n"
     '    binding_target("terminal-hasher")\n', [
         "terminal-hasher: expected exactly one binding_target('terminal-hasher') call, "
         f"in {_TRAINING_FILE}::_record_checkpoint_reads; found "
         f"['{_REQUEST_FILE}::test_valid_run_ids_are_accepted', "
         f"'{_TRAINING_FILE}::_record_checkpoint_reads']",
     ]),
}


@pytest.mark.parametrize("plant", _SITE_CENSUS_PLANTS)
def test_patch_binding_site_census_rejects_a_respelled_site(plant):
    """One respelled site or one stray call is rejected for its own clause; the tree stays green."""
    rel, old, new, expected = _SITE_CENSUS_PLANTS[plant]
    sources = _original_site_sources()
    assert _binding_site_violations(sources) == []
    assert sources[rel].count(old) == 1, f"plant {plant!r} no longer matches {rel}"
    planted = dict(sources, **{rel: sources[rel].replace(old, new)})
    assert _binding_site_violations(planted) == sorted(expected)
    assert _binding_site_violations(_original_site_sources()) == []
