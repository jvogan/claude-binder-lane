"""Stage selected run artifacts with distinct names for a flat archive."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import quote_from_bytes


class ArtifactHandoffError(ValueError):
    """A requested artifact cannot be staged for promotion."""


def flat_archive_name(relative_path: Path) -> str:
    """Return a collision-free flat filename for a run-root-relative path.

    Every component is percent encoded. A literal ``+`` separates components,
    because percent encoding never emits a literal plus sign. The mapping is
    therefore one-to-one for valid relative paths.
    """

    if relative_path.is_absolute() or not relative_path.parts:
        raise ArtifactHandoffError("artifact path must be relative and non-empty")
    if any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ArtifactHandoffError(f"artifact path is unsafe: {relative_path}")
    return "+".join(
        quote_from_bytes(part.encode("utf-8"), safe="-.")
        for part in relative_path.parts
    )


def stage_flat_artifacts(
    files: Iterable[Path | str],
    *,
    source_root: Path | str,
    destination: Path | str,
) -> list[Path]:
    """Copy files beneath ``source_root`` to an empty flat handoff directory."""

    root = Path(source_root).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactHandoffError(f"source root is not a directory: {root}")
    if output.exists() and not output.is_dir():
        raise ArtifactHandoffError(f"handoff destination is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise ArtifactHandoffError(f"handoff destination is not empty: {output}")

    planned: list[tuple[Path, Path]] = []
    seen_sources: set[Path] = set()
    seen_destinations: set[Path] = set()
    for value in files:
        source = Path(value).expanduser().resolve()
        if source in seen_sources:
            continue
        if not source.is_file():
            raise ArtifactHandoffError(f"artifact is not a file: {source}")
        try:
            relative = source.relative_to(root)
        except ValueError as exc:
            raise ArtifactHandoffError(
                f"artifact is outside source root: {source} is not under {root}"
            ) from exc
        staged = output / flat_archive_name(relative)
        if staged in seen_destinations:
            raise ArtifactHandoffError(f"flat archive name collision: {staged.name}")
        seen_sources.add(source)
        seen_destinations.add(staged)
        planned.append((source, staged))

    if not planned:
        raise ArtifactHandoffError("no artifacts were supplied for promotion")

    output.mkdir(parents=True, exist_ok=True)
    for source, staged in planned:
        shutil.copy2(source, staged)
    return [staged for _source, staged in planned]
