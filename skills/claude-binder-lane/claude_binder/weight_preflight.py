"""Verify required Hugging Face snapshots before GPU allocation."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_REPO_PART = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_REVISION = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def _validate_repo_id(repo_id: str) -> None:
    parts = repo_id.split("/")
    if len(parts) not in (1, 2) or any(not _REPO_PART.fullmatch(part) for part in parts):
        raise ValueError(f"invalid Hugging Face repository id: {repo_id!r}")
    if any("--" in part for part in parts):
        raise ValueError(f"repository id cannot contain '--': {repo_id!r}")


def _validate_revision(revision: str) -> None:
    if not _REVISION.fullmatch(revision) or revision in {".", ".."}:
        raise ValueError(f"invalid recorded revision: {revision!r}")


@dataclass(frozen=True)
class RepositoryRequirement:
    """One repository, its project pin, and its minimum hydrated size."""

    repo_id: str
    revision: str | None = None
    expected_size_bytes: int | None = None

    def __post_init__(self) -> None:
        _validate_repo_id(self.repo_id)
        if self.revision is not None:
            _validate_revision(self.revision)
        if self.expected_size_bytes is not None and self.expected_size_bytes < 0:
            raise ValueError("expected_size_bytes must be zero or greater")


@dataclass(frozen=True)
class RepositoryResult:
    """The local cache facts for one repository requirement."""

    requirement: RepositoryRequirement
    present: bool
    actual_revision: str | None
    revision_status: str
    found_size_bytes: int
    problems: tuple[str, ...] = ()

    @property
    def size_status(self) -> str:
        expected = self.requirement.expected_size_bytes
        if expected is None:
            return "unrecorded"
        if self.found_size_bytes < expected:
            return "undersized"
        return "sufficient"

    @property
    def ok(self) -> bool:
        return (
            self.present
            and self.revision_status == "match"
            and self.size_status == "sufficient"
        )


@dataclass(frozen=True)
class PreflightReport:
    """Results for every repository required by one run."""

    cache_dir: Path
    repositories: tuple[RepositoryResult, ...]

    @property
    def ok(self) -> bool:
        return bool(self.repositories) and all(item.ok for item in self.repositories)

    @property
    def found_size_bytes(self) -> int:
        return sum(item.found_size_bytes for item in self.repositories)

    @property
    def expected_size_bytes(self) -> int | None:
        expected = [item.requirement.expected_size_bytes for item in self.repositories]
        if any(value is None for value in expected):
            return None
        return sum(value for value in expected if value is not None)


def default_cache_dir(environment: dict[str, str] | None = None) -> Path:
    """Resolve the local hub cache without importing a model package."""

    values = os.environ if environment is None else environment
    if values.get("HF_HUB_CACHE"):
        return Path(values["HF_HUB_CACHE"]).expanduser()
    if values.get("HF_HOME"):
        return Path(values["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def repository_cache_dir(cache_dir: Path, repo_id: str) -> Path:
    """Return the on-disk model directory for a validated repository id."""

    _validate_repo_id(repo_id)
    return cache_dir / ("models--" + repo_id.replace("/", "--"))


def _snapshot_size(snapshot_dir: Path, repository_dir: Path) -> tuple[int, tuple[str, ...]]:
    """Count unique local files and report incomplete or escaping links."""

    total = 0
    seen: set[tuple[int, int]] = set()
    problems: list[str] = []
    repository_root = repository_dir.resolve()
    try:
        for path in snapshot_dir.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                continue
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(repository_root)
                if not resolved.is_file():
                    problems.append(f"snapshot entry is not a file: {path}")
                    continue
                stat_result = resolved.stat()
            except (OSError, RuntimeError, ValueError) as exc:
                problems.append(f"snapshot entry is unavailable: {path}: {exc}")
                continue
            identity = (stat_result.st_dev, stat_result.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += stat_result.st_size
    except OSError as exc:
        problems.append(f"snapshot cannot be listed: {exc}")
    return total, tuple(problems)


def inspect_repository(cache_dir: Path, requirement: RepositoryRequirement) -> RepositoryResult:
    """Inspect one repository through its main reference and resolved snapshot."""

    repository_dir = repository_cache_dir(cache_dir, requirement.repo_id)
    problems: list[str] = []
    actual_revision: str | None = None
    found_size = 0
    snapshot_present = False

    if not repository_dir.is_dir():
        problems.append(f"repository directory is missing: {repository_dir}")
    else:
        ref_path = repository_dir / "refs" / "main"
        if not ref_path.is_file():
            problems.append(f"main reference is missing: {ref_path}")
        else:
            try:
                actual_revision = ref_path.read_text(encoding="utf-8").strip()
                _validate_revision(actual_revision)
            except (OSError, UnicodeError, ValueError) as exc:
                problems.append(f"main reference is unreadable: {ref_path}: {exc}")
                actual_revision = None

        if actual_revision is not None:
            snapshot_dir = repository_dir / "snapshots" / actual_revision
            if not snapshot_dir.is_dir():
                problems.append(f"referenced snapshot is missing: {snapshot_dir}")
            else:
                found_size, snapshot_problems = _snapshot_size(snapshot_dir, repository_dir)
                problems.extend(snapshot_problems)
                snapshot_present = not snapshot_problems

    if requirement.revision is None:
        revision_status = "unpinned"
    elif actual_revision == requirement.revision:
        revision_status = "match"
    else:
        revision_status = "mismatch"

    return RepositoryResult(
        requirement=requirement,
        present=snapshot_present,
        actual_revision=actual_revision,
        revision_status=revision_status,
        found_size_bytes=found_size,
        problems=tuple(problems),
    )


def run_preflight(
    cache_dir: Path, requirements: Sequence[RepositoryRequirement]
) -> PreflightReport:
    """Inspect every unique repository requirement without network access."""

    if not requirements:
        raise ValueError("at least one repository requirement is required")
    repo_ids = [requirement.repo_id for requirement in requirements]
    duplicates = sorted({repo_id for repo_id in repo_ids if repo_ids.count(repo_id) > 1})
    if duplicates:
        raise ValueError("duplicate repository requirements: " + ", ".join(duplicates))
    return PreflightReport(
        cache_dir=cache_dir,
        repositories=tuple(inspect_repository(cache_dir, item) for item in requirements),
    )


def parse_requirement(value: str) -> RepositoryRequirement:
    """Parse REPO[@REVISION][=BYTES] from one CLI argument."""

    repository_and_revision, separator, raw_size = value.rpartition("=")
    if separator:
        if not raw_size.isdecimal():
            raise ValueError(f"repository size must be an integer byte count: {value!r}")
        expected_size = int(raw_size)
    else:
        repository_and_revision = value
        expected_size = None

    repo_id, revision_separator, revision = repository_and_revision.rpartition("@")
    if revision_separator:
        if not revision:
            raise ValueError(f"recorded revision is empty: {value!r}")
    else:
        repo_id = repository_and_revision
        revision = None
    return RepositoryRequirement(repo_id, revision, expected_size)


def _revision_text(result: RepositoryResult) -> str:
    actual = result.actual_revision or "missing"
    expected = result.requirement.revision
    if result.revision_status == "unpinned":
        return f"unpinned (found {actual}); a pin is ours to choose"
    if result.revision_status == "match":
        return f"match ({actual})"
    return f"mismatch (expected {expected}, found {actual})"


def _size_text(result: RepositoryResult) -> str:
    expected = result.requirement.expected_size_bytes
    if expected is None:
        return f"{result.found_size_bytes:,} bytes found / TODO required bytes"
    return f"{result.found_size_bytes:,} bytes found / {expected:,} bytes required"


def render_report(report: PreflightReport) -> str:
    """Render a complete refusal or success message for an agent caller."""

    lines = [f"weight preflight: {'PASS' if report.ok else 'REFUSED'}"]
    lines.append(f"cache directory: {report.cache_dir}")
    for result in report.repositories:
        presence = "present" if result.present else "missing"
        lines.append(
            f"- {result.requirement.repo_id}: {presence}; "
            f"revision {_revision_text(result)}; size {_size_text(result)}"
        )
        lines.extend(f"  problem: {problem}" for problem in result.problems)

    expected_total = report.expected_size_bytes
    if expected_total is None:
        lines.append(
            f"cache size: {report.found_size_bytes:,} bytes found / TODO required bytes"
        )
    else:
        lines.append(
            f"cache size: {report.found_size_bytes:,} bytes found / "
            f"{expected_total:,} bytes required"
        )

    missing = [item.requirement.repo_id for item in report.repositories if not item.present]
    mismatched = [
        item.requirement.repo_id
        for item in report.repositories
        if item.revision_status == "mismatch"
    ]
    unpinned = [
        item.requirement.repo_id
        for item in report.repositories
        if item.revision_status == "unpinned"
    ]
    undersized = [
        item.requirement.repo_id
        for item in report.repositories
        if item.size_status == "undersized"
    ]
    unrecorded_sizes = [
        item.requirement.repo_id
        for item in report.repositories
        if item.size_status == "unrecorded"
    ]
    if missing:
        lines.append("missing repositories: " + ", ".join(missing))
    if mismatched:
        lines.append("revision mismatches: " + ", ".join(mismatched))
    if unpinned:
        lines.append(
            "unpinned repositories: " + ", ".join(unpinned) + "; a pin is ours to choose"
        )
    if undersized:
        lines.append("undersized repositories: " + ", ".join(undersized))
    if unrecorded_sizes:
        lines.append(
            "TODO required sizes for repositories: " + ", ".join(unrecorded_sizes)
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify local Hugging Face snapshots before GPU allocation."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hugging Face hub cache. Defaults to HF_HUB_CACHE or HF_HOME/hub.",
    )
    parser.add_argument(
        "--repo",
        action="append",
        required=True,
        metavar="REPO[@REVISION][=BYTES]",
        help="Required repository, recorded revision, and minimum hydrated byte count.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        requirements = [parse_requirement(value) for value in args.repo]
        report = run_preflight(args.cache_dir or default_cache_dir(), requirements)
    except (OSError, ValueError) as exc:
        print(f"weight preflight: REFUSED\ninput problem: {exc}", file=sys.stderr)
        return 2
    message = render_report(report)
    print(message, file=sys.stdout if report.ok else sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
