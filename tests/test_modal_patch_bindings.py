"""Deterministic, per-site observations of the real patch consumers."""
import ast
import inspect
import json
import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from scripts.modal_runner import checkpoint as checkpoints
from scripts.modal_runner import core, preflight, state, training
from tests import test_modal_runner as runner
from tests.modal_patch_binding_campaign import BINDING_SITES, binding_target

# Repository root: the campaign's children and the census below read files by
# repo-relative path.
REPO_ROOT = Path(__file__).resolve().parents[1]

# What each site's companion observation drives, recorded as the `stimulus`
# field of the campaign's per-site record (`_record_observation`), and kept true
# of `test_patch_binding_observation`'s branch for that site.
_STIMULUS = {
    'prepare-validator':
    'preflight._validate_remote_resume(checkpoint, None) on stable local bytes, weights-only '
    'load stubbed; the site validator wraps the real one',
    'fallback-loader':
    'checkpoint._assert_weights_only_loadable on stable bytes with the site loader raising '
    'ImportError and a stubbed prebuilt-python subprocess',
    'fallback-python':
    'same torch-less fallback; the subprocess stub captures argv[0] against the site '
    'interpreter marker',
    'interrupt-loader':
    'synchronous checkpoint.validate_local_checkpoint on stable bytes with a recording site '
    'loader',
    'watcher-publisher':
    'training._start_checkpoint_watcher loop run inline for exactly two iterations through '
    'Event/Thread doubles; the site publisher returns generation (111, 222)',
    'terminal-validator':
    'synchronous training.publish_stable_checkpoint on a stable checkpoint with a recording '
    'site validator',
    'terminal-hasher':
    'synchronous training.publish_stable_checkpoint with validation stubbed to pass and a '
    'marker site hasher',
    'attempt-watcher':
    'training.execute_training_attempt with a child that has exited (code 2), inline threads '
    'and a recording inert site watcher start',
    'attempt-transition':
    'the same non-interrupted attempt with real STATUS writes under a recording site '
    'transition_status wrapper',
    'client-mount':
    'core.mounted_path("binding-probe") under a site-installed temporary mount marker',
}

# The ORIGINAL installer of each BINDING_SITES key: the test file, and the
# module-level function whose body calls `binding_target(site)`. The campaign
# certifies `binding_target(site)` through the companion; this table ties that
# certificate to the original sites. `_binding_site_violations` checks it.
_ORIGINAL_SITES = {
    'prepare-validator':
    ('tests/test_modal_runner.py', 'test_prepare_validates_resume_then_dumps_and_hashes_config'),
    'fallback-loader': ('tests/test_modal_runner.py', '_no_torch'),
    'fallback-python': ('tests/test_modal_runner.py', '_no_torch'),
    'interrupt-loader':
    ('tests/test_modal_runner.py', 'test_interrupt_commits_status_even_if_prebuilt_load_hangs'),
    'watcher-publisher': ('tests/test_modal_runner.py',
                          'test_checkpoint_watcher_threads_generation_into_last_published'),
    'terminal-validator': ('tests/test_modal_runner.py', '_record_hash_after_terminal'),
    'terminal-hasher': ('tests/test_modal_runner.py', '_record_hash_after_terminal'),
    'attempt-watcher':
    ('tests/test_modal_runner.py', 'test_checkpoint_watcher_stops_before_terminal_status'),
    'attempt-transition':
    ('tests/test_modal_runner.py', 'test_checkpoint_watcher_stops_before_terminal_status'),
    'client-mount': ('tests/test_modal_client.py',
                     'test_train_remote_completes_against_post_dump_manifest_hash'),
}


def _capture_consumer(monkeypatch, consumer, symbol, captured):
    """Replace only one binding lookup; all other globals remain live.

    WHAT: every load of `symbol` in `consumer`'s source (a bare name or an
    attribute) is rewritten to `_captured_binding`, which holds `captured`, the
    value the site's target had before the patch. The recompiled clone runs with
    the consumer's own globals and replaces it in its module for this test.

    WHY: this is the single-binding defect. The clone keeps reading the old
    value while the site's patch lands on the target, so the site's own
    observation must change -- that is what the campaign's captured arm checks.

    PITFALLS:
      * AST column offsets are UTF-8 BYTE offsets, so the source is spliced as
        bytes. Slicing the decoded text with them corrupts any line that has a
        non-ASCII character before the reference.
      * A reference that spans lines has its end offset on another line, so the
        one-line splice would corrupt it; it is rejected instead.
      * The clone's line numbers restart at 1, so its tracebacks cite wrong
        lines. It is compiled with this module's `__future__` flags, not the
        consumer's, so its signature annotations are evaluated eagerly.
    """
    source = inspect.getsource(consumer)
    tree = ast.parse(source)
    replacements = []
    for node in ast.walk(tree):
        if ((isinstance(node, ast.Name) and node.id == symbol) or
            (isinstance(node, ast.Attribute) and node.attr == symbol)) and isinstance(
                node.ctx, ast.Load):
            replacements.append(node)
    assert replacements, f"missing consumer reference: {symbol}"
    lines = source.encode('utf-8').splitlines(keepends=True)
    for node in sorted(replacements, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        assert node.end_lineno == node.lineno, f"multi-line consumer reference: {symbol}"
        line = lines[node.lineno - 1]
        lines[node.lineno -
              1] = line[:node.col_offset] + b'_captured_binding' + line[node.end_col_offset:]
    namespace = dict(consumer.__globals__)
    namespace['_captured_binding'] = captured
    filename = inspect.getsourcefile(consumer)
    assert filename is not None, f"no source file for {consumer.__qualname__}"
    exec(compile(b''.join(lines), filename, 'exec'), namespace)
    # Use the original globals for every lookup except the captured reference.
    import types
    clone = types.FunctionType(namespace[consumer.__name__].__code__, consumer.__globals__,
                               consumer.__name__, consumer.__defaults__, consumer.__closure__)
    clone.__kwdefaults__ = consumer.__kwdefaults__
    monkeypatch.setitem(consumer.__globals__, '_captured_binding', captured)
    monkeypatch.setitem(consumer.__globals__, consumer.__name__, clone)
    return clone


def _install(monkeypatch, site, consumer, replacement):
    """Install the original site target, optionally breaking just its consumer lookup."""
    target, symbol = binding_target(site)
    if os.environ.get('MODAL_BINDING_DEFECT') == site:
        consumer = _capture_consumer(monkeypatch, consumer, symbol, getattr(target, symbol))
    if os.environ.get('MODAL_BINDING_PACKAGE_REVERT') == site:
        import scripts.modal_runner as package
        target = package
    monkeypatch.setattr(target, symbol, replacement)
    return consumer


def _record_observation(record_property, site, consumer, expected, observed):
    """Persist actual values even when the following site assertion rejects them."""
    record_property(
        'binding_observation',
        json.dumps(
            {
                'site': site,
                'probe_kind':
                'value' if site in {'fallback-python', 'client-mount'} else 'callable',
                'consumer': consumer.__module__ + '.' + consumer.__name__,
                'stimulus': _STIMULUS[site],
                'expected_observation': expected,
                'observed_observation': observed,
            },
            default=str))


@pytest.mark.parametrize('site', BINDING_SITES)
def test_patch_binding_observation(site, tmp_path, monkeypatch, record_property):
    """Reach each original site through its real consumer and record its own observation."""
    checkpoint = tmp_path / 'checkpoints' / core.CHECKPOINT_NAME
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'stable checkpoint bytes')
    seen = []
    if site == 'prepare-validator':
        monkeypatch.setattr(checkpoints, '_assert_weights_only_loadable', lambda path: None)
        # Named per branch: one `original` rebound across branches gets one
        # type, so a checker would read this call against another branch's value.
        original_validate = checkpoints.validate_local_checkpoint

        def validate(path):
            seen.append(path)
            return original_validate(path)

        consume = _install(monkeypatch, site, preflight._validate_remote_resume, validate)
        consume(checkpoint, None)
        _record_observation(record_property, site, consume, [checkpoint], seen)
        assert seen == [checkpoint], 'prepare-validator: exact validator path'
    elif site in {'fallback-loader', 'fallback-python'}:
        interpreter = tmp_path / 'binding-python'
        interpreter.touch()

        def no_torch():
            seen.append(())
            raise ImportError('binding control')

        argv = []

        def run(command, **kwargs):
            argv.append(command)
            return SimpleNamespace(returncode=0, stderr='')

        monkeypatch.setattr(checkpoints.subprocess, 'run', run)
        # Captured original imports remain deterministic and never import torch.
        monkeypatch.setattr(checkpoints, '_import_torch',
                            lambda: SimpleNamespace(load=lambda *a, **k: None))
        if site == 'fallback-loader':
            monkeypatch.setattr(*binding_target('fallback-python'), str(interpreter))
            consume = _install(monkeypatch, site, checkpoints._assert_weights_only_loadable,
                               no_torch)
        else:
            monkeypatch.setattr(*binding_target('fallback-loader'), no_torch)
            old_interpreter = tmp_path / 'captured-python'
            old_interpreter.touch()
            monkeypatch.setattr(core, 'PREBUILT_PYTHON', str(old_interpreter))
            consume = _install(monkeypatch, site, checkpoints._assert_weights_only_loadable,
                               str(interpreter))
        consume(checkpoint)
        if site == 'fallback-loader':
            _record_observation(record_property, site, consume, [()], seen)
            assert seen == [()], 'fallback-loader: exact loader count'
        else:
            _record_observation(record_property, site, consume, [str(interpreter)],
                                [cmd[0] for cmd in argv])
            assert len(argv) == 1 and argv[0][0] == str(
                interpreter), 'fallback-python: interpreter marker'
    elif site == 'interrupt-loader':
        monkeypatch.setattr(checkpoints, '_assert_weights_only_loadable', lambda path: None)
        consume = _install(monkeypatch, site, checkpoints.validate_local_checkpoint,
                           lambda path: seen.append(path))
        consume(checkpoint)
        _record_observation(record_property, site, consume, [checkpoint], seen)
        assert seen == [checkpoint], 'interrupt-loader: exact loader path'
    elif site == 'watcher-publisher':

        class TwoIterations:

            def __init__(self):
                self.waits = 0

            def is_set(self):
                return self.waits >= 2

            def wait(self, seconds):
                assert seconds == 0.05
                self.waits += 1
                return self.is_set()

        class InlineThread:

            def __init__(self, *, target, **kwargs):
                self.target = target

            def start(self):
                self.target()

        stop = TwoIterations()
        monkeypatch.setattr(training, 'threading',
                            SimpleNamespace(Event=lambda: stop, Thread=InlineThread))

        def publish(path, **kwargs):
            seen.append((path, kwargs))
            return core.PublishOutcome((111, 222))

        consume = _install(monkeypatch, site, training._start_checkpoint_watcher, publish)
        now, commit, sleep = runner._aware, lambda: None, lambda seconds: None
        consume(run_root=tmp_path, now=now, commit=commit, sleep=sleep)
        _record_observation(record_property, site, consume, [(tmp_path, None),
                                                             (tmp_path, (111, 222))],
                            [(path, kw['last_published']) for path, kw in seen])
        assert [(path, kw['last_published']) for path, kw in seen] == [
            (tmp_path, None), (tmp_path, (111, 222))
        ], 'watcher-publisher: exact generation sequence'
        assert all(
            set(kw) == {'now', 'commit', 'sleep', 'last_published'} and kw['now'] is now
            and kw['sleep'] is sleep and callable(kw['commit']) for _, kw in seen)
        assert stop.waits == 2
    elif site in {'terminal-validator', 'terminal-hasher'}:
        monkeypatch.setattr(checkpoints, '_assert_weights_only_loadable', lambda path: None)
        if site == 'terminal-validator':

            def replacement(path):
                seen.append(path)
        else:
            monkeypatch.setattr(checkpoints, 'validate_local_checkpoint', lambda path: None)

            def replacement(path):
                seen.append(path)
                return 'ab' * 32

        consume = _install(monkeypatch, site, training.publish_stable_checkpoint, replacement)
        outcome = consume(tmp_path,
                          now=runner._aware,
                          commit=lambda: None,
                          sleep=lambda seconds: None)
        assert outcome.reason is None
        # Annotated: the terminal-hasher row adds a str digest to each.
        expected: dict[str, object] = {'calls': [checkpoint]}
        observed: dict[str, object] = {'calls': seen}
        if site == 'terminal-hasher':
            expected['sidecar_digest'] = 'ab' * 32
            observed['sidecar_digest'] = json.loads(
                checkpoint.with_name(core.CHECKPOINT_SIDECAR_NAME).read_text())['sha256']
        _record_observation(record_property, site, consume, expected, observed)
        assert seen == [checkpoint], f'{site}: exact consumer path'
        if site == 'terminal-hasher':
            assert json.loads(checkpoint.with_name(core.CHECKPOINT_SIDECAR_NAME).read_text()
                              )['sha256'] == 'ab' * 32, 'terminal-hasher: sidecar marker'
    elif site in {'attempt-watcher', 'attempt-transition'}:

        class InlineThread:

            def __init__(self, *, target, args=(), **kwargs):
                self.target, self.args = target, args

            def start(self):
                self.target(*self.args)

            def join(self, timeout=None):
                pass

        # Create fixtures before replacing thread construction.
        child = runner.FakeChild(returncode=2)
        sleeps = []
        kwargs = runner._consume_training_kwargs(
            runner._training_kwargs(tmp_path,
                                    child=child,
                                    start_heartbeat=runner._noop_heartbeat,
                                    signal_signal=lambda *args: None,
                                    sleep=sleeps.append))
        threading = training.threading
        monkeypatch.setattr(
            training, 'threading',
            SimpleNamespace(Thread=InlineThread, Event=threading.Event, Lock=threading.Lock))

        def inert_watcher(**kwargs):
            return threading.Event(), SimpleNamespace(join=lambda **kw: None)

        monkeypatch.setattr(training, '_start_checkpoint_watcher', inert_watcher)
        original_transition = state.transition_status
        if site == 'attempt-watcher':

            def replacement(**kw):
                seen.append(kw)
                return inert_watcher(**kw)
        else:

            def replacement(path, status, **kw):
                seen.append((path, status, kw))
                return original_transition(path, status, **kw)

        _install(monkeypatch, site, training._run_training_attempt, replacement)
        result = training.execute_training_attempt(**kwargs)
        # The attempt is typed `object` (a losing delivery returns REDELIVERED);
        # this row wins, so it must be the winner's result type.
        assert isinstance(result, core.TrainingAttemptResult)
        assert result.status is core.Status.FAILED
        if site == 'attempt-watcher':
            _record_observation(record_property, site, training._run_training_attempt,
                                [{
                                    key: kwargs[key]
                                    for key in ('run_root', 'now', 'commit', 'sleep')
                                }], seen)
            assert seen == [{
                key: kwargs[key]
                for key in ('run_root', 'now', 'commit', 'sleep')
            }], 'attempt-watcher: exact start arguments'
        else:
            common = {
                'now': kwargs['now'](),
                'attempt_id': kwargs['attempt_id'],
                'lock': kwargs['lock']
            }
            _record_observation(record_property, site, training._run_training_attempt,
                                [(kwargs['run_root'], core.Status.TRAINING, common),
                                 (kwargs['run_root'], core.Status.FAILED, common)], seen)
            assert seen == [(kwargs['run_root'], core.Status.TRAINING, common),
                            (kwargs['run_root'], core.Status.FAILED, common)
                            ], 'attempt-transition: exact status sequence'
        assert json.loads(
            (kwargs['run_root'] / core.STATUS_FILENAME).read_text())['status'] == 'failed'
        # No polling dependency: the child has exited before the attempt's first
        # poll() check, so the loop breaks there and never waits or sleeps.
        # `child.wait_timeouts` watches the loop's `child.wait(timeout=...)`;
        # `sleeps` watches its sleep fallback for a child without `wait`. The
        # heartbeat `wait` is not watched: this row injects none, and the default
        # reaches only the heartbeat starter, which `_noop_heartbeat` ignores. A
        # row that polled would depend on timing.
        assert (sleeps, child.wait_timeouts) == ([], []), 'attempt rows must not poll'
    elif site == 'client-mount':
        marker = tmp_path / 'binding-mount'
        consume = _install(monkeypatch, site, core.mounted_path, marker)
        observed = consume(PurePosixPath('binding-probe'))
        _record_observation(record_property, site, consume, marker / 'binding-probe', observed)
        assert observed == marker / 'binding-probe', 'client-mount: observed mount marker'
    else:
        pytest.fail(f'unknown binding site: {site}')


@pytest.mark.slow                      # 7 pytest child sessions per binding site (~1 min)
def test_patch_binding_campaign(tmp_path):
    """The executable instrument must report every original site and own bite.

    Every captured-arm `source_line` must fall inside an `assert` of
    `test_patch_binding_observation` in THIS file as it is now: the records are
    evidence only if they cite the bytes that ran, and a campaign run against an
    older copy of this file cites lines that are no longer assertions.
    """
    import subprocess
    import sys

    observation, = [
        node for node in ast.parse(Path(__file__).read_text(encoding='utf-8')).body
        if isinstance(node, ast.FunctionDef) and node.name == 'test_patch_binding_observation'
    ]
    assert_spans = [(node.lineno, node.end_lineno) for node in ast.walk(observation)
                    if isinstance(node, ast.Assert)]
    sites = list(BINDING_SITES)
    matrix = tmp_path / 'sites.json'
    matrix.write_text(json.dumps(sites))
    evidence = tmp_path / 'campaign'
    process = subprocess.run(
        [
            sys.executable, '-m', 'tests.modal_patch_binding_campaign', '--repo-root',
            str(REPO_ROOT), '--evidence-root',
            str(evidence), '--matrix',
            str(matrix)
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    (tmp_path / 'campaign.log').write_text(process.stdout + process.stderr, encoding='utf-8')
    assert process.returncode == 0, process.stdout + process.stderr
    records = json.loads(process.stdout)
    assert [row['site'] for row in records] == sites
    for row in records:
        assert row['probe_kind'] in {'callable', 'value'}
        assert row['consumer'] and row['stimulus'] == _STIMULUS[row['site']]
        assert row['expected_observation'] == row['observed_observation']
        assert len(row['repetitions']) == 2
        for repetition in row['repetitions']:
            assert repetition['repointed_outcome']['exit_code'] == 0
            assert repetition['restored_outcome']['exit_code'] == 0
            defect = repetition['broken_binding_outcome']
            assert defect['exit_code'] == 1 and defect['exception_type'] == 'AssertionError'
            assert row['site'] + ':' in defect['rejecting_assertion']
            assert any(first <= defect['source_line'] <= last for first, last in assert_spans), (
                f"{row['site']}: source line {defect['source_line']} is not an assertion of "
                f"test_patch_binding_observation in this file: {assert_spans}")
            assert defect['observed_observation'] != defect['expected_observation']
        assert row['package_revert_exception'] in {'AttributeError', 'AssertionError'}
        assert row['package_revert_outcome']['exit_code'] == 1


def test_patch_target_dichotomy(tmp_path, monkeypatch):
    """Package clients and internal module consumers have different targets."""
    import scripts.modal_runner as package
    from scripts.modal_runner import checkpoint, preflight

    path = tmp_path / 'checkpoint.pt'
    path.write_bytes(b'stable')
    monkeypatch.setattr(checkpoint, '_assert_weights_only_loadable', lambda path: None)
    original = checkpoint.validate_local_checkpoint
    marker = original(path)
    seen = []

    def spy(path):
        seen.append(path)
        return marker

    with monkeypatch.context() as patches:
        patches.setattr(package, 'validate_local_checkpoint', spy)
        assert package.validate_local_checkpoint(path) is marker
        assert preflight._validate_remote_resume(path, None) == marker
        assert seen == [path], 'package patch must reach only the client lookup'
    seen.clear()
    with monkeypatch.context() as patches:
        patches.setattr(checkpoint, 'validate_local_checkpoint', spy)
        assert package.validate_local_checkpoint(path) == marker
        assert seen == [], 'owner patch must not change the captured package export'
        assert preflight._validate_remote_resume(path, None) is marker
        assert seen == [path], 'owner patch must reach the internal consumer'


# One row per rejection clause of the campaign driver, plus a positive control.
# The supplied rows replace `_run_probe` with well-formed evidence and break one
# field, so each row's clause is its sole objector: a row that broke several
# fields would stay red after its own clause was deleted, because a sibling
# clause would still fire. `missing-observation` drops the field from every
# phase (the baseline is checked first). `unexpected-exception` and
# `surviving-binding` are the realistic multi-field captured shapes and pin no
# clause alone. `well-formed` makes the one-field claim checkable: the same
# evidence with nothing broken must be accepted.
_DEFECT_CLAUSE = 'prepare-validator: single-binding defect did not lose its own observation'
_FIRST_PROBE = 'prepare-validator: baseline probe prepare-validator-0-baseline'
_REJECTION_CLAUSES = {
    'well-formed': None,
    'missing-site': 'declared matrix must equal BINDING_SITES, in order',
    'missing-observation': 'prepare-validator: baseline observation failed',
    'baseline-exit': 'prepare-validator: baseline observation failed',
    'baseline-mismatch': 'prepare-validator: baseline observation failed',
    'restored-failure': 'prepare-validator: restored observation failed',
    'unexpected-exception': _DEFECT_CLAUSE,
    'surviving-binding': _DEFECT_CLAUSE,
    'captured-exit': _DEFECT_CLAUSE,
    'captured-exception-type': _DEFECT_CLAUSE,
    'captured-foreign-assertion': _DEFECT_CLAUSE,
    'captured-no-source-line': _DEFECT_CLAUSE,
    'captured-missing-observation': _DEFECT_CLAUSE,
    'captured-unchanged': _DEFECT_CLAUSE,
    'package-exit': 'prepare-validator: unexpected package-reversion diagnostic',
    'package-exception': 'prepare-validator: unexpected package-reversion diagnostic',
    'existing-evidence-root': 'evidence-root must be a fresh directory',
    'probe-timeout': f'{_FIRST_PROBE} timed out',
    'missing-junit': f'{_FIRST_PROBE} wrote no readable JUnit report',
    'unparseable-junit': f'{_FIRST_PROBE} wrote no readable JUnit report',
    'no-collected-case': f'{_FIRST_PROBE} reported 0 test cases, expected exactly one',
    'valueless-observation': f'{_FIRST_PROBE} recorded binding_observation without a value',
    'empty-failure-message': f'{_FIRST_PROBE} reported a failure with an empty message',
}
# The rows that keep the real `_run_probe` and must name its log.
_PHASE_FAULTS = frozenset({
    'probe-timeout', 'missing-junit', 'unparseable-junit', 'no-collected-case',
    'valueless-observation', 'empty-failure-message'
})


@pytest.mark.parametrize('fault', _REJECTION_CLAUSES)
def test_patch_binding_campaign_rejects_invalid_evidence(fault, tmp_path):
    """The real CLI must fail closed, for its own clause, when a site or its evidence is invalid.

    Each row pins one rejection clause (`_REJECTION_CLAUSES` above). The
    supplied rows replace `_run_probe` and cover every clause of `run_campaign`:
    a baseline or restored arm that exits non-zero, lacks its observation or
    observes the wrong value; a captured arm with the wrong exit code or
    exception type, no own `site:` assertion, no source line, no observation or
    an unchanged observation; and a package-reversion arm with the wrong exit
    code or exception type. `unexpected-exception` and `surviving-binding` are
    the realistic multi-field shapes of a captured arm. The phase rows keep the
    real `_run_probe`: `probe-timeout` replaces `subprocess.run` in the child
    with one that raises `TimeoutExpired`, and the five JUnit rows with one that
    writes no report, a truncated report, a report with no test case, a
    valueless observation or an empty failure message; each must name the site,
    the phase and the log.
    `existing-evidence-root` is the one guard `main` keeps on the evidence root.
    """
    import subprocess
    import sys

    matrix = tmp_path / 'matrix.json'
    sites = list(BINDING_SITES)
    matrix.write_text(json.dumps(sites[:-1] if fault == 'missing-site' else sites),
                      encoding='utf-8')
    evidence = tmp_path / 'evidence'
    if fault == 'existing-evidence-root':
        evidence.mkdir()
    program = '''
import subprocess
import sys
from pathlib import Path
from tests import modal_patch_binding_campaign as campaign
fault = sys.argv.pop(1)
BROKEN_FIELD = {
    'baseline-exit': ('baseline', {'exit_code': 1}),
    'baseline-mismatch': ('baseline', {'observed_observation': [2]}),
    'restored-failure': ('restored', {'exit_code': 1}),
    'captured-exit': ('captured', {'exit_code': 2}),
    'captured-exception-type': ('captured', {'exception_type': 'ValueError'}),
    'captured-foreign-assertion': ('captured', {'rejecting_assertion': 'AssertionError: x'}),
    'captured-no-source-line': ('captured', {'source_line': None}),
    'captured-unchanged': ('captured', {'observed_observation': [1]}),
    'package-exit': ('package', {'exit_code': 0}),
    'package-exception': ('package', {'exception_type': 'KeyError'}),
}
def supplied_probe(repo_root, evidence_root, site, mode, label):
    result = {'site': site, 'probe_kind': 'callable', 'consumer': 'supplied',
              'stimulus': 'supplied', 'exit_code': 0, 'exception_type': None,
              'source_line': None, 'rejecting_assertion': None,
              'expected_observation': [1], 'observed_observation': [1]}
    if mode == 'captured':
        result.update(exit_code=1, exception_type='AssertionError', source_line=10,
                      rejecting_assertion=f'AssertionError: {site}: planted',
                      observed_observation=[])
    elif mode == 'package':
        owner_only = campaign.BINDING_SITES[site][1] != 'validate_local_checkpoint'
        result.update(exit_code=1,
                      exception_type='AttributeError' if owner_only else 'AssertionError')
    if fault == 'missing-observation':
        result.pop('expected_observation')
    elif mode == 'captured' and fault == 'captured-missing-observation':
        result.pop('expected_observation')
    elif mode == 'captured' and fault == 'unexpected-exception':
        result.update(exception_type='ValueError', rejecting_assertion='unrelated failure')
    elif mode == 'captured' and fault == 'surviving-binding':
        result.update(exit_code=0, exception_type=None, source_line=None,
                      rejecting_assertion=None, observed_observation=[1])
    elif fault in BROKEN_FIELD and BROKEN_FIELD[fault][0] == mode:
        result.update(BROKEN_FIELD[fault][1])
    return result
JUNIT = {
    'unparseable-junit': '<testsuite><testcase name="x"',
    'no-collected-case': '<testsuite/>',
    'valueless-observation': '<testsuite><testcase name="x"><properties>'
                             '<property name="binding_observation"/></properties>'
                             '</testcase></testsuite>',
    'empty-failure-message': '<testsuite><testcase name="x"><failure message=""/>'
                             '</testcase></testsuite>',
}
def junit_child(command, **kwargs):
    report, = [arg.split('=', 1)[1] for arg in command if arg.startswith('--junitxml=')]
    if fault in JUNIT:
        Path(report).write_text(JUNIT[fault], encoding='utf-8')
    return subprocess.CompletedProcess(command, 1, stdout='child output', stderr='')
def hung_child(command, **kwargs):
    raise subprocess.TimeoutExpired(command, kwargs['timeout'], output=b'partial child output')
if fault == 'probe-timeout':
    subprocess.run = hung_child
elif fault == 'missing-junit' or fault in JUNIT:
    subprocess.run = junit_child
else:
    campaign._run_probe = supplied_probe
campaign.main()
'''
    result = subprocess.run(
        [
            sys.executable, '-c', program, fault, '--repo-root',
            str(REPO_ROOT), '--evidence-root',
            str(evidence), '--matrix',
            str(matrix)
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / 'rejection.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    clause = _REJECTION_CLAUSES[fault]
    if clause is None:
        assert result.returncode == 0, result.stderr
        assert [row['site'] for row in json.loads(result.stdout)] == sites
        return
    assert result.returncode != 0
    assert clause in result.stderr, result.stderr
    if fault in _PHASE_FAULTS:
        log = evidence / 'prepare-validator-0-baseline' / 'pytest.log'
        output = 'partial child output' if fault == 'probe-timeout' else 'child output'
        assert log.read_text(encoding='utf-8') == output
        assert str(log) in result.stderr, result.stderr


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
      `validate_local_checkpoint` although it does not replace it (a known
      false positive, which fails closed), and `mrl.__dict__[...] = ...` is
      reported as `__dict__`, which equals no site symbol.

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
                node.func, 'id', None)
            target = node.args[0] if node.args else None
            on_facade = target is not None and ast.unparse(target) in facade
            if called == 'setattr' and on_facade and len(node.args) > 1:
                name = node.args[1]
                found.append(name.value if isinstance(name, ast.Constant) else None)
            elif called == 'multiple' and on_facade:
                found.extend(keyword.arg for keyword in node.keywords)
            elif (called in ('setattr', 'patch') and isinstance(target, ast.Constant)
                  and isinstance(target.value, str)
                  and target.value.startswith('scripts.modal_runner.')):
                found.append(target.value.removeprefix('scripts.modal_runner.'))
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
    there. No statement in those files may patch the facade through a name the
    census cannot read (a positional non-literal `setattr` name, or `**` names
    to `mock.patch.multiple`), because it cannot tell which symbol that
    patches. Returns the sorted violation strings; `[]` is green.

    WHY: `test_patch_binding_campaign` proves that `binding_target(site)` reaches
    each consumer, and nothing more. A review of the package split found that
    respelling `_record_hash_after_terminal`'s validator patch as a package patch
    stayed green everywhere: the facade exports `validate_local_checkpoint`, so
    there is no AttributeError, and that helper's callers assert
    `hashed == []`, which a patch that never reaches the consumer satisfies
    trivially.

    PITFALLS:
      * The facade spellings are the dotted `scripts.modal_runner`, always,
        plus the aliases each file's own imports bind (`import
        scripts.modal_runner as mrl`, `from scripts import modal_runner`), so
        a renamed alias is still seen. The dotted spelling needs no import of
        its own: any `import scripts.<x>` binds `scripts`, through which it
        reaches the package, and it can name nothing else.
      * A non-literal key (`binding_target(site)`) is reported, not skipped: the
        census cannot tell which site such a call installs.
      * Only the files `_ORIGINAL_SITES` names are read. The companion in this
        file calls `binding_target` too, by design, and is not an original site.
      * This reads source. It proves that each installer calls
        `binding_target(site)`, that the installer's own body does not patch
        its site's symbol on the facade in any spelling `_facade_patches`
        lists, and that no statement in those files patches the facade
        through a name the census cannot read. It does not prove the patch
        USES that call's result: a dead `binding_target(site)` call stays
        green beside a real patch that `_facade_patches` does not report under
        the site's symbol, whether it matches none of its shapes or reports
        another name. Among them, each measured green:
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
        # PITFALLS). `_production_package_surface` (tests/test_modal_client.py)
        # builds the same alias set its own way. Deliberately not shared: one
        # helper for both would be a new seam-governed name in a governed file,
        # so it is not a pure move.
        facade = {'scripts.modal_runner'} | {
            alias.asname or alias.name
            for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names if alias.name == 'scripts.modal_runner'
        } | {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == 'scripts'
            for alias in node.names if alias.name == 'modal_runner'
        }
        for statement in tree.body:
            owner = getattr(statement, 'name', '<module>')
            functions[(rel, owner)] = (statement, facade)
            if None in _facade_patches(statement, facade):
                violations.append(f'non-literal facade patch name in {rel}::{owner}')
            for node in ast.walk(statement):
                if not (isinstance(node, ast.Call)
                        and getattr(node.func, 'id', None) == 'binding_target'):
                    continue
                key = node.args[0].value if node.args and isinstance(node.args[0],
                                                                     ast.Constant) else None
                if not isinstance(key, str):
                    violations.append(f'non-literal binding_target argument in {rel}::{owner}')
                elif key not in BINDING_SITES:
                    violations.append(f'unknown site {key!r} in {rel}::{owner}')
                else:
                    calls.setdefault(key, []).append(f'{rel}::{owner}')
    for site, (rel, owner) in _ORIGINAL_SITES.items():
        found = calls.get(site, [])
        if found != [f'{rel}::{owner}']:
            violations.append(f'{site}: expected exactly one binding_target({site!r}) call, in '
                              f'{rel}::{owner}; found {found}')
        if (rel, owner) not in functions:
            violations.append(f'{site}: declared installer {rel}::{owner} is not defined')
            continue
        statement, facade = functions[(rel, owner)]
        symbol = BINDING_SITES[site][1]
        if symbol in _facade_patches(statement, facade):
            violations.append(f"{site}: {rel}::{owner} patches the package facade's {symbol}")
    return sorted(violations)


def _original_site_sources():
    """{repo-relative path: text} of every file `_ORIGINAL_SITES` names."""
    files = sorted(set(rel for rel, _ in _ORIGINAL_SITES.values()))
    return {rel: (REPO_ROOT / rel).read_text(encoding='utf-8') for rel in files}


def test_patch_binding_sites_route_through_binding_target():
    """Each original installer calls `binding_target(site)` where the table says.

    None of them patches its own site's symbol on the facade in a spelling the
    census recognises; patching other facade symbols is allowed.
    `_facade_patches` lists every shape the census matches, including one
    known false positive; a patch it does not report under the site's symbol
    is invisible, and `_binding_site_violations` names the ones measured so.

    The campaign certifies `binding_target(site)`; this is what makes that
    certificate about the original tests. See `_binding_site_violations`.
    """
    assert set(_ORIGINAL_SITES) == set(BINDING_SITES) == set(_STIMULUS)
    assert _binding_site_violations(_original_site_sources()) == []


# Each plant rewrites exactly one original site in memory. `old` must occur
# exactly once in the named file, so a plant that stops matching fails loudly
# instead of passing on unchanged source. The `unaliased-import-*`,
# `dotted-via-other-import`, `mock-patch-*` rows keep the site's
# `binding_target` call beside the facade patch, so the patch spelling under
# test is the only thing that can object.
_RUNNER_FILE = 'tests/test_modal_runner.py'
_SITE_CENSUS_PLANTS = {
    'package-revert':
    ('    monkeypatch.setattr(*binding_target("terminal-validator"), wrapped_validate)\n',
     '    monkeypatch.setattr(mrl, "validate_local_checkpoint", wrapped_validate)\n', [
         "terminal-validator: expected exactly one binding_target('terminal-validator') call, "
         f"in {_RUNNER_FILE}::_record_hash_after_terminal; found []",
         f"terminal-validator: {_RUNNER_FILE}::_record_hash_after_terminal patches the "
         "package facade's validate_local_checkpoint",
     ]),
    'dotted-string-revert':
    ('    monkeypatch.setattr(*binding_target("terminal-hasher"), wrapped_hash)\n',
     '    monkeypatch.setattr("scripts.modal_runner.sha256_file", wrapped_hash)\n', [
         "terminal-hasher: expected exactly one binding_target('terminal-hasher') call, "
         f"in {_RUNNER_FILE}::_record_hash_after_terminal; found []",
         f"terminal-hasher: {_RUNNER_FILE}::_record_hash_after_terminal patches the "
         "package facade's sha256_file",
     ]),
    'facade-store':
    ('    setattr(validate_target, validate_name, monkey_validate)\n',
     '    mrl.validate_local_checkpoint = monkey_validate\n', [
         f"prepare-validator: {_RUNNER_FILE}::"
         "test_prepare_validates_resume_then_dumps_and_hashes_config patches the package "
         "facade's validate_local_checkpoint",
     ]),
    'unaliased-import-setattr':
    ('    monkeypatch.setattr(*binding_target("terminal-validator"), wrapped_validate)\n',
     '    binding_target("terminal-validator")\n'
     '    import scripts.modal_runner\n'
     '    monkeypatch.setattr(scripts.modal_runner, "validate_local_checkpoint",\n'
     '                        wrapped_validate)\n', [
         f"terminal-validator: {_RUNNER_FILE}::_record_hash_after_terminal patches the "
         "package facade's validate_local_checkpoint",
     ]),
    'unaliased-import-store':
    ('    setattr(validate_target, validate_name, monkey_validate)\n',
     '    import scripts.modal_runner\n'
     '    scripts.modal_runner.validate_local_checkpoint = monkey_validate\n', [
         f"prepare-validator: {_RUNNER_FILE}::"
         "test_prepare_validates_resume_then_dumps_and_hashes_config patches the package "
         "facade's validate_local_checkpoint",
     ]),
    'dotted-via-other-import':
    ('    monkeypatch.setattr(*binding_target("terminal-validator"), wrapped_validate)\n',
     '    binding_target("terminal-validator")\n'
     '    import scripts.modal_artifacts\n'
     '    monkeypatch.setattr(scripts.modal_runner, "validate_local_checkpoint",\n'
     '                        wrapped_validate)\n', [
         f"terminal-validator: {_RUNNER_FILE}::_record_hash_after_terminal patches the "
         "package facade's validate_local_checkpoint",
     ]),
    'mock-patch-dotted-string':
    ('    monkeypatch.setattr(*binding_target("terminal-hasher"), wrapped_hash)\n',
     '    binding_target("terminal-hasher")\n'
     '    mock.patch("scripts.modal_runner.sha256_file", wrapped_hash).start()\n', [
         f"terminal-hasher: {_RUNNER_FILE}::_record_hash_after_terminal patches the "
         "package facade's sha256_file",
     ]),
    'mock-patch-multiple':
    ('    monkeypatch.setattr(*binding_target("terminal-validator"), wrapped_validate)\n',
     '    binding_target("terminal-validator")\n'
     '    mock.patch.multiple(mrl, validate_local_checkpoint=wrapped_validate).start()\n', [
         f"terminal-validator: {_RUNNER_FILE}::_record_hash_after_terminal patches the "
         "package facade's validate_local_checkpoint",
     ]),
    'mock-patch-multiple-non-literal':
    ('    monkeypatch.setattr(*binding_target("terminal-validator"), wrapped_validate)\n',
     '    binding_target("terminal-validator")\n'
     '    mock.patch.multiple(mrl, **{"validate_local_checkpoint": wrapped_validate}).start()\n', [
         f"non-literal facade patch name in {_RUNNER_FILE}::_record_hash_after_terminal",
     ]),
    'misspelled': ('binding_target("terminal-hasher")', 'binding_target("terminal-hashr")', [
        "terminal-hasher: expected exactly one binding_target('terminal-hasher') call, "
        f"in {_RUNNER_FILE}::_record_hash_after_terminal; found []",
        f"unknown site 'terminal-hashr' in {_RUNNER_FILE}::_record_hash_after_terminal",
    ]),
    'duplicated': ('binding_target("fallback-python")', 'binding_target("fallback-loader")', [
        "fallback-loader: expected exactly one binding_target('fallback-loader') call, "
        f"in {_RUNNER_FILE}::_no_torch; found ['{_RUNNER_FILE}::_no_torch', "
        f"'{_RUNNER_FILE}::_no_torch']",
        "fallback-python: expected exactly one binding_target('fallback-python') call, "
        f"in {_RUNNER_FILE}::_no_torch; found []",
    ]),
    'facade-restore-non-literal':
    ('        setattr(validate_target, validate_name, orig_validate)\n',
     '        setattr(mrl, validate_name, orig_validate)\n', [
         "non-literal facade patch name in "
         f"{_RUNNER_FILE}::test_prepare_validates_resume_then_dumps_and_hashes_config",
     ]),
    'non-literal': ('binding_target("interrupt-loader")', 'binding_target(site)', [
        "interrupt-loader: expected exactly one binding_target('interrupt-loader') call, "
        f"in {_RUNNER_FILE}::test_interrupt_commits_status_even_if_prebuilt_load_hangs; "
        "found []",
        "non-literal binding_target argument in "
        f"{_RUNNER_FILE}::test_interrupt_commits_status_even_if_prebuilt_load_hangs",
    ]),
}


@pytest.mark.parametrize('plant', _SITE_CENSUS_PLANTS)
def test_patch_binding_site_census_rejects_a_respelled_site(plant):
    """One respelled original site is rejected for its own clause; the live tree stays green."""
    old, new, expected = _SITE_CENSUS_PLANTS[plant]
    sources = _original_site_sources()
    assert _binding_site_violations(sources) == []
    assert sources[_RUNNER_FILE].count(old) == 1, f'plant {plant!r} no longer matches'
    planted = dict(sources, **{_RUNNER_FILE: sources[_RUNNER_FILE].replace(old, new)})
    assert _binding_site_violations(planted) == sorted(expected)
    assert _binding_site_violations(_original_site_sources()) == []
