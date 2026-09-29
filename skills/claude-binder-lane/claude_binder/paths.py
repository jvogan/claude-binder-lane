"""Paths for installed Claude Binder resources and generated run data."""

from __future__ import annotations

import errno
import os
import site
import sysconfig
from pathlib import Path
from typing import Mapping


PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    """Return the installed package directory."""
    return PACKAGE_ROOT


def package_import_root() -> Path:
    """Return the directory a child process needs on its path to import this package."""
    return PACKAGE_ROOT.parent


def _interpreter_import_roots() -> set[str]:
    """Return the directories this interpreter imports from with no PYTHONPATH entry."""
    roots: set[str] = set()
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        value = paths.get(key)
        if value:
            roots.add(os.path.realpath(value))
    for reader_name in ("getsitepackages", "getusersitepackages"):
        reader = getattr(site, reader_name, None)
        if reader is None:
            continue
        try:
            found = reader()
        except Exception:  # noqa: BLE001 - a reader that cannot answer names no root
            continue
        values = [found] if isinstance(found, str) else list(found)
        for value in values:
            if value:
                roots.add(os.path.realpath(value))
    return roots


def child_process_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment a spawned child can import this package from.

    A host that binds the package by file path leaves the package's parent directory
    off `sys.path`, which is how the Claude Science kernel loads it. Every child the
    executor spawns then fails on `python -m claude_binder...` with a
    ModuleNotFoundError. The parent knows where its own source lives, so it hands that
    directory to the child rather than asking an operator to export PYTHONPATH.

    The entry goes first so the child runs the same source as the parent. A PYTHONPATH
    the caller already set is kept after it. Nothing is added when the package sits
    where the interpreter already looks, because prepending a site-packages directory
    would move it ahead of the standard library. A relative entry is never read as
    satisfying the requirement, because it resolves against the child's working
    directory rather than this one.
    """
    merged = os.environ.copy()
    if environment:
        merged.update(environment)
    # `python3 -B` sets a flag on the parent interpreter and does not reach a
    # child, so a stage spawned from a documented -B run still wrote __pycache__
    # into the installed skill and turned its own file check red. The variable
    # does reach the child. A caller that sets it keeps its own value.
    merged.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    root = str(package_import_root())
    resolved_root = os.path.realpath(root)
    if resolved_root in _interpreter_import_roots():
        return merged
    entries = [entry for entry in merged.get("PYTHONPATH", "").split(os.pathsep) if entry]
    for entry in entries:
        if os.path.isabs(entry) and os.path.realpath(entry) == resolved_root:
            return merged
    merged["PYTHONPATH"] = os.pathsep.join([root, *entries])
    return merged


def data_root(explicit: Path | None = None) -> Path:
    """Return the root for generated bundles and run outputs."""
    override = explicit or os.environ.get("CLAUDE_BINDER_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.cwd() / ".claude-binder").resolve()


def _relative_parts(name: str) -> tuple[str, ...]:
    relative = Path(name)
    if not name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"package data path must be relative: {name}")
    return relative.parts


def package_file(*parts: str) -> Path:
    """Return an existing package file without allowing path traversal."""
    path = package_root().joinpath(*parts).resolve()
    root = package_root().resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"package file escapes the package: {Path(*parts)}")
    if not path.is_file():
        # The bare path this used to raise reached the CLI as the whole error
        # payload, because every layer above preserves only the type name and
        # str(exc). Four callers catch FileNotFoundError from here by type and
        # fall back to "absent", one of them inside execution_support_hashes,
        # so the type has to stay and only the message may change.
        raise FileNotFoundError(
            errno.ENOENT,
            "package file is missing. A stage reads package data while it runs, "
            "so a skill directory replaced or pruned mid-run fails here. Re-place "
            "the skill, then resume with --resume, which re-runs the bundle check "
            "and names every package file that moved",
            str(path),
        )
    return path


def schema_file(name: str) -> Path:
    """Return an existing schema from packaged data."""
    return package_file("data", *_relative_parts(name))


def helper_file(name: str) -> Path:
    """Return an existing helper from packaged data."""
    return package_file("data", "helpers", *_relative_parts(name))
