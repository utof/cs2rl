"""The at-risk Modal runner patch sites (`BINDING_SITES`) and the campaign proving each bites.

WHAT this module owns:
  * `BINDING_SITES` / `binding_target`: the one table that says which owning
    submodule attribute each at-risk patch site replaces. The original sites
    install their patch through `binding_target(site)`; `_ORIGINAL_SITES` in
    tests/modal/test_modal_patch_binding_census.py names each one's file and
    installing function (since W4: tests/modal/test_modal_training.py,
    tests/modal/test_modal_preflight.py, tests/modal/modal_test_helpers.py and
    tests/modal/test_modal_client.py). So does the per-site companion
    `test_patch_binding_observation` in tests/modal/test_modal_patch_bindings.py, so the
    companion observes exactly the target the original site patches.
    `test_patch_binding_sites_route_through_binding_target`
    (tests/modal/test_modal_patch_binding_census.py) is the census
    that ties each key to the original function that installs it.
  * the campaign driver (`run_campaign`, `main`): for every site it runs the
    companion in fresh pytest children -- baseline, a single-binding defect
    (`MODAL_BINDING_DEFECT`) and restoration, twice -- plus one package-reversion
    diagnostic (`MODAL_BINDING_PACKAGE_REVERT`), and prints one JSON record per
    site. It exits non-zero on an evidence root that already exists, a missing
    site, an unmet own observation, an unexpected exception, a surviving
    defect, an unexpected package-reversion diagnostic, or a child that times
    out or leaves no usable JUnit evidence (`_run_probe` lists those cases).

WHY it lives under tests/: it is test harness, not production code. Production
never imports it. scripts/run_modal.py builds the image with runner modules copied
under /opt/app/scripts; native dependency setup copies pyproject.toml and uv.lock
into the SDK-managed /.uv context. /opt/cs2rl/.venv is a compatibility
symlink to /.uv/.venv. Training source arrives separately as a content-addressed
archive.

Run from the repository root (`test_patch_binding_campaign` is the checked-in
caller): python -m tests.modal.modal_patch_binding_campaign --repo-root .
--evidence-root <new directory> --matrix <JSON list of BINDING_SITES' keys, in order>

PITFALLS:
  * `binding_target` imports the owner from a formatted string, always under
    the `scripts.modal_runner.<owner>` spelling. No static reader sees that
    import, ruff's TID251 ban included. The module scope imports only
    importlib, concurrent.futures and typing, so importing `BINDING_SITES`
    (tests/modal/test_modal_packaging.py does) loads no runner module.
  * Children run `sys.executable -m pytest` from `repo_root` with
    `-p no:cacheprovider`, explicit timeout/xdist plugins and pytest11 autoload
    disabled. The
    synchronous observations need no other installed pytest11 plugins, so
    native `--disable-plugin-autoload` avoids their repeated startup cost.
    `uv run` would depend on uv being on PATH and on the
    venv's console-script shebang, and could re-sync the shared venv; the cache
    provider would write each child's lastfailed into the repository's
    .pytest_cache.
  * The evidence root must not exist yet, and may be anywhere: pytest's
    `--basetemp` decides where the checked-in caller puts it.
"""
import importlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any

BINDING_SITES = {
    "prepare-validator": ("checkpoint", "validate_local_checkpoint"),
    "fallback-loader": ("checkpoint", "_import_torch"),
    "fallback-python": ("core", "PREBUILT_PYTHON"),
    "interrupt-loader": ("checkpoint", "_assert_weights_only_loadable"),
    "watcher-publisher": ("training", "publish_stable_checkpoint"),
    "terminal-validator": ("checkpoint", "validate_local_checkpoint"),
    "terminal-hasher": ("core", "sha256_file"),
    "attempt-watcher": ("training", "_start_checkpoint_watcher"),
    "attempt-transition": ("state", "transition_status"),
    "client-mount": ("core", "VOLUME_MOUNT"),
}

# Seconds one child pytest session may run before the campaign fails. Measured
# on 2026-09-22, by pytest `--durations`: the call phase of
# `test_patch_binding_campaign`, which then ran all 70 children, took 64.4 s
# while a full suite ran concurrently, so about 1 s per child. The margin is for
# a loaded machine, not for a slow test.
_PROBE_TIMEOUT_S = 60

# How many sites `run_campaign` runs at once (each site's probes stay sequential).
# 4 is the measured knee (#285): the campaign's call took 63 s serially and 25 s on 4
# threads, with its whole process tree (up to 4 child pytest sessions at a time) at 690 MB
# RSS. It is a constant, not a knob: the campaign runs inside a pytest session that may
# itself be one of two xdist workers on a 15 GB box, so a bigger pool is a memory decision.
_SITE_POOL_SIZE = 4


def binding_target(site):
    """One installation target for the original patch and its observation."""
    owner, symbol = BINDING_SITES[site]
    return importlib.import_module(f"scripts.modal_runner.{owner}"), symbol


# RENAME HAZARD: `test_patch_binding_campaign_rejects_invalid_evidence`
# (tests/modal/test_modal_patch_bindings.py) replaces this function BY NAME inside a
# `python -c` program string (`campaign._run_probe = supplied_probe`). No call
# graph, grep for a call, or rename tool sees that reach. Rename both together:
# a renamed function leaves that assignment inert, and its supplied-evidence
# rows then run real pytest children instead of the planted evidence.
# The same reach constrains `_campaign_site`: it must call `_run_probe` as a MODULE
# GLOBAL, looked up at call time. A reference bound earlier (a default argument, a
# closure, a partial made at def time) keeps calling the real function after the
# assignment, and the rejection rows turn red (#285's early-binding knock-out).
def _run_probe(repo_root, evidence_root, site, mode, label):
    """Run one isolated pytest and retain its own observation and failure clause.

    Clear the parent's marker filter so every intended child consumer runs.
    Each child gets its own `--basetemp`, JUnit report and full log under
    `evidence_root / label`. Each of these failures to produce evidence raises
    a RuntimeError naming the site, the phase (`mode`), the probe label and the
    log, so the campaign fails closed with a pointer to the output:
      * the child exceeds `_PROBE_TIMEOUT_S` (killed by `subprocess.run`; the
        partial output is kept in the log);
      * it writes no JUnit report, or one that does not parse;
      * the report does not hold exactly one test case;
      * its `binding_observation` property has no value;
      * it records a failure whose message is empty, so no exception type can
        be read from it.

    PITFALL: a `binding_observation` value that is not valid JSON still fails
    closed, but as a bare JSONDecodeError with no site or log. It is left so
    because the companion writes that value with `json.dumps`; no row covers
    it.
    """
    import json
    import os
    import re
    import subprocess
    import sys
    import xml.etree.ElementTree as ET

    directory = evidence_root / label
    directory.mkdir()
    report = directory / "junit.xml"
    log = directory / "pytest.log"
    env = dict(os.environ)
    env.pop("MODAL_BINDING_DEFECT", None)
    env.pop("MODAL_BINDING_PACKAGE_REVERT", None)
    if mode == "captured":
        env["MODAL_BINDING_DEFECT"] = site
    elif mode == "package":
        env["MODAL_BINDING_PACKAGE_REVERT"] = site
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest", "--disable-plugin-autoload", "-p", "pytest_timeout",
                "-p", "xdist.plugin",
                f"tests/modal/test_modal_patch_bindings.py::test_patch_binding_observation[{site}]",
                "-q", "-m", "", "--tb=short", "-p", "no:cacheprovider", "-o", "junit_family=legacy",
                f'--basetemp={directory / "pytest"}', f"--junitxml={report}"
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as expired:
        # The partial output is bytes even under text=True (subprocess docs).
        partial = [
            part.decode("utf-8", "replace") if isinstance(part, bytes) else part or ""
            for part in (expired.stdout, expired.stderr)
        ]
        log.write_text("".join(partial), encoding="utf-8")
        raise RuntimeError(f"{site}: {mode} probe {label} timed out after {_PROBE_TIMEOUT_S} s; "
                           f"partial output in {log}") from None
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    probe = f"{site}: {mode} probe {label}"
    # Annotated: the three None placeholders are replaced by str/int values below.
    record: dict[str, Any] = {
        "exit_code": result.returncode,
        "log": str(log),
        "exception_type": None,
        "source_line": None,
        "rejecting_assertion": None
    }
    try:
        tree = ET.parse(report)
    except (FileNotFoundError, ET.ParseError) as error:
        raise RuntimeError(f"{probe} wrote no readable JUnit report (exit {result.returncode}, "
                           f"{type(error).__name__}: {error}); output in {log}") from None
    cases = tree.findall(".//testcase")
    if len(cases) != 1:
        raise RuntimeError(
            f"{probe} reported {len(cases)} test cases, expected exactly one; output in {log}")
    for prop in cases[0].findall("./properties/property"):
        if prop.get("name") == "binding_observation":
            value = prop.get("value")
            if value is None:
                raise RuntimeError(f"{probe} recorded binding_observation without a value; "
                                   f"output in {log}")
            record.update(json.loads(value))
    failure = cases[0].find("failure")
    if failure is None:
        failure = cases[0].find("error")
    if failure is not None:
        message = failure.get("message", "")
        head = message.split(":", 1)[0].splitlines()
        if not head:
            raise RuntimeError(f"{probe} reported a failure with an empty message; output in {log}")
        record["exception_type"] = head[0]
        record["rejecting_assertion"] = message
        locations = re.findall(r"tests/modal/test_modal_patch_bindings\.py:(\d+)", failure.text
                               or "")
        if locations:
            record["source_line"] = int(locations[-1])
    return record


def run_campaign(repo_root, evidence_root, sites):
    """Require two own-observation bites and restorations for all declared sites."""
    if sites != list(BINDING_SITES):
        raise ValueError("declared matrix must equal BINDING_SITES, in order "
                         f"(tests/modal/modal_patch_binding_campaign.py): got {sites}, "
                         f"expected {list(BINDING_SITES)}")
    # The sites are independent, so each site's sequential probes run as one task on a
    # THREAD pool. Never processes: the rejection test replaces `_run_probe` by name in its
    # own interpreter, and `_campaign_site` reads that module global at call time. Records
    # are collected in BINDING_SITES order and the first failing site IN THAT ORDER raises,
    # so a failure names the same site as it did serially.
    with ThreadPoolExecutor(max_workers=_SITE_POOL_SIZE) as pool:
        futures = [pool.submit(_campaign_site, repo_root, evidence_root, site) for site in sites]
        try:
            return [future.result() for future in futures]
        except BaseException:
            # Sites not yet started never start; running ones finish before the pool exits.
            pool.shutdown(wait=True, cancel_futures=True)
            raise


def _campaign_site(repo_root, evidence_root, site):
    """One site's record: two baseline/captured/restored repetitions and the package probe."""
    repetitions = []
    for repeat in range(2):
        results = {
            mode: _run_probe(repo_root, evidence_root, site, mode, f"{site}-{repeat}-{mode}")
            for mode in ("baseline", "captured", "restored")
        }
        for mode in ("baseline", "restored"):
            observed = results[mode]
            if (observed["exit_code"] != 0 or "expected_observation" not in observed
                    or observed["expected_observation"] != observed["observed_observation"]):
                raise RuntimeError(f"{site}: {mode} observation failed: {observed}")
        broken = results["captured"]
        if (broken["exit_code"] != 1 or broken["exception_type"] != "AssertionError"
                or site + ":" not in (broken["rejecting_assertion"] or "")
                or not broken["source_line"] or "expected_observation" not in broken
                or broken["expected_observation"] == broken["observed_observation"]):
            raise RuntimeError(
                f"{site}: single-binding defect did not lose its own observation: {broken}")
        repetitions.append({
            "repointed_outcome": results["baseline"],
            "broken_binding_outcome": broken,
            "restored_outcome": results["restored"]
        })
    package = _run_probe(repo_root, evidence_root, site, "package", f"{site}-package")
    expected_exception = ("AssertionError" if BINDING_SITES[site][1] == "validate_local_checkpoint"
                          else "AttributeError")
    if package["exit_code"] != 1 or package["exception_type"] != expected_exception:
        raise RuntimeError(f"{site}: unexpected package-reversion diagnostic: {package}")
    baseline = repetitions[0]["repointed_outcome"]
    return {
        **{
            key: baseline[key]
            for key in ("site", "probe_kind", "consumer", "stimulus", "expected_observation", "observed_observation")
        },
        "repointed_outcome": baseline,
        "broken_binding_outcome": repetitions[0]["broken_binding_outcome"],
        "rejecting_assertion": repetitions[0]["broken_binding_outcome"]["rejecting_assertion"],
        "package_revert_outcome": package,
        "package_revert_exception": package["exception_type"],
        "repetitions": repetitions,
    }


def main():
    """Validate the declared matrix and emit one JSON evidence record per binding site.

    The evidence root must be a directory that does not exist yet; it is created
    here, wherever it is. Refusing an existing one keeps two campaigns from
    mixing their per-child logs and reports.
    """
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    args = parser.parse_args()
    if args.evidence_root.exists():
        parser.error(f"evidence-root must be a fresh directory: {args.evidence_root} exists")
    args.evidence_root.mkdir(parents=True, exist_ok=False)
    records = run_campaign(args.repo_root.resolve(), args.evidence_root.resolve(),
                           json.loads(args.matrix.read_text(encoding="utf-8")))
    result = json.dumps(records, indent=2) + "\n"
    (args.evidence_root / "records.json").write_text(result, encoding="utf-8")
    print(result, end="")


if __name__ == "__main__":
    main()
