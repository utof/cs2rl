#!/usr/bin/env python3
"""Emit locked runtime requirements for the Modal image, excluding PufferLib.

Walks uv.lock from the cs2rl package's runtime `dependencies` only — never
`[package.dev-dependencies]` — then prints `name==version` pins. pufferlib is
reachable (so torch and friends stay) but omitted from the output so the image
does not try to build it.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from collections import deque
from pathlib import Path

SKIP_EMIT = frozenset({"cs2rl", "pufferlib"})
ROOT_PACKAGE = "cs2rl"


def _packages_by_name(lock: dict) -> dict[str, dict]:
    packages: dict[str, dict] = {}
    for pkg in lock.get("package") or []:
        name = pkg.get("name")
        if name is None:
            raise ValueError("uv.lock [[package]] is missing name")
        packages[name] = pkg
    return packages


def _enqueue_dep(pending: deque[str], extras: deque[tuple[str, str]], dep: dict) -> None:
    name = dep["name"]
    pending.append(name)
    for extra in dep.get("extra") or ():
        extras.append((name, extra))


def reachable_runtime_names(lock: dict, root: str = ROOT_PACKAGE) -> set[str]:
    """Return package names reachable from `root`'s runtime dependency graph."""
    packages = _packages_by_name(lock)
    if root not in packages:
        raise ValueError(f"{root} package not found in uv.lock")

    pending: deque[str] = deque([root])
    extra_pending: deque[tuple[str, str]] = deque()
    seen: set[str] = set()
    seen_extras: set[tuple[str, str]] = set()

    while pending or extra_pending:
        while pending:
            name = pending.popleft()
            if name in seen:
                continue
            if name not in packages:
                raise ValueError(f"uv.lock is missing package {name}")
            seen.add(name)
            for dep in packages[name].get("dependencies") or ():
                _enqueue_dep(pending, extra_pending, dep)
        while extra_pending:
            name, extra = extra_pending.popleft()
            key = (name, extra)
            if key in seen_extras:
                continue
            seen_extras.add(key)
            if name not in seen:
                pending.append(name)
            pkg = packages.get(name)
            if pkg is None:
                raise ValueError(f"uv.lock is missing package {name}")
            optional = pkg.get("optional-dependencies") or {}
            for dep in optional.get(extra) or ():
                _enqueue_dep(pending, extra_pending, dep)
    return seen


def requirement_lines(lock: dict, root: str = ROOT_PACKAGE) -> list[str]:
    packages = _packages_by_name(lock)
    names = reachable_runtime_names(lock, root)
    pins = []
    for name in names:
        if name in SKIP_EMIT:
            continue
        version = packages[name].get("version")
        if not version:
            raise ValueError(f"uv.lock package {name} is missing version")
        pins.append((name, version))
    pins.sort()
    return [f"{name}=={version}" for name, version in pins]


def render_requirements(lock: dict, root: str = ROOT_PACKAGE) -> str:
    return "".join(f"{line}\n" for line in requirement_lines(lock, root))


def load_lock(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit locked cs2rl runtime requirements, omitting pufferlib.")
    parser.add_argument("lockfile", type=Path)
    parser.add_argument("-o", dest="output", type=Path, default=None)
    args = parser.parse_args(argv)
    text = render_requirements(load_lock(args.lockfile))
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
