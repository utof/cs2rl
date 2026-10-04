"""The repository default must omit heavy tiers without hiding explicit failures."""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from tests.conftest import REPO_ROOT


@pytest.mark.parametrize(
    ("expression", "target", "expected", "broken"),
    [
        (None, "", ["test_fast"], False),
        ("training", "", ["test_training"], False),
        ("slow and not training", "", ["test_extended"], False),
        ("", "", ["test_fast", "test_training", "test_extended"], False),
        ("", "::test_training", ["test_training"], True),
    ],
    ids=["default", "training", "extended", "complete", "explicit-failure"],
)
def test_repository_selection_reaches_the_intended_consumers(tmp_path, expression, target, expected,
                                                             broken):
    """Exercise the real config; clearing the filter must expose an assertion failure."""
    plant = tmp_path / "test_selection.py"
    plant.write_text("import os\nimport pytest\n"
                     "def test_fast():\n    assert True\n"
                     "@pytest.mark.training\n"
                     "def test_training():\n"
                     "    assert os.environ['SELECTION_BROKEN'] == '0', 'training consumer broke'\n"
                     "@pytest.mark.slow\n"
                     "def test_extended():\n    assert True\n")
    report = tmp_path / "selection.xml"
    env = {key: value for key, value in os.environ.items() if key != "PYTEST_ADDOPTS"}
    env["SELECTION_BROKEN"] = str(int(broken))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    argv = [
        sys.executable,
        "-m",
        "pytest",
        str(plant) + target,
        "-c",
        str(REPO_ROOT / "pyproject.toml"),
        "-p",
        "tests.conftest",
        "-n",
        "0",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--junitxml={report}",
    ]
    if expression is not None:
        argv.extend(["-m", expression])
    child = subprocess.run(argv, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=30)
    output = child.stdout + child.stderr
    assert child.returncode == int(broken), output
    cases = list(ET.parse(report).iter("testcase"))
    assert [case.attrib["name"] for case in cases] == expected, output
    failures = [case.find("failure") for case in cases if case.find("failure") is not None]
    if broken:
        failure, = failures
        assert failure is not None, output
        assert "AssertionError" in failure.attrib["message"], output
        assert "training consumer broke" in failure.attrib["message"], output
    else:
        assert not failures, output


@pytest.mark.parametrize("mode", ["repointed", "captured"])
def test_binding_child_clears_parent_selection(tmp_path, monkeypatch, mode):
    """A parent filter must preserve one child's observation and own failing assertion."""
    from tests.modal.modal_patch_binding_campaign import _run_probe

    monkeypatch.setenv("PYTEST_ADDOPTS", "-m slow")
    result = _run_probe(REPO_ROOT, tmp_path, "client-mount", mode, "selection")
    assert result["exit_code"] == int(mode == "captured"), result
    assert result["site"] == "client-mount", result
    if mode == "captured":
        assert result["exception_type"] == "AssertionError", result
        assert "client-mount:" in result["rejecting_assertion"], result
        assert result["source_line"] is not None, result
    else:
        assert result["observed_observation"] == result["expected_observation"], result
