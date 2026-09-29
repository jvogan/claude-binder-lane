#!/usr/bin/env python3
"""Validate a pinned, local MMseqs2 + UniRef90 bundle.

This checker intentionally performs no network I/O.  The preparation phase is
responsible for downloading and hashing the release artifacts; campaign jobs
receive an immutable bundle plus the expected SHA-256 of its manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


SCHEMA_VERSION = 1
MMSEQS_TAG = "18-8cc5c"
MMSEQS_COMMIT = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"
MMSEQS_LINUX_AVX2_ARCHIVE = "mmseqs-linux-avx2.tar.gz"
MMSEQS_LINUX_AVX2_BYTES = 17_375_777
MMSEQS_LINUX_AVX2_SHA256 = (
    "bd9b0234da5949ad528d5b5f9ea4cda9c1e23dce14b46c0791d4d919a76e61ce"
)

UNIREF90_RELEASE = "2026_02"
UNIREF90_RELEASE_DATE = "2026-06-10"
UNIREF90_CLUSTER_COUNT = 121_389_642
UNIREF90_FASTA_BYTES = 32_059_052_376
UNIREF90_FASTA_MD5 = "abdd341aeafa7fa060c8d6639d594990"
UNIREF90_FASTA_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "uniref/uniref90/uniref90.fasta.gz"
)

EXPECTED_METADATA = {
    "root.RELEASE.metalink": {
        "bytes": 2_811,
        "sha256": "d82ecb077c2380265092c0038f7ea383a4647f4e488cb976270eb91c3c799990",
    },
    "relnotes.txt": {
        "bytes": 1_146,
        "sha256": "ae71660a09dc2100a1e0d8f1ca61fdf73805a3d23a1fb295490deef51d8909bb",
    },
    "uniref90.RELEASE.metalink": {
        "bytes": 4_391,
        "sha256": "e3cb6d885a451b340e472056b285ffc612e432819105ba006ad9451b97876336",
    },
    "uniref90.release_note": {
        "bytes": 303,
        "sha256": "2f16174f2f95fdb70d0e44b26a7ccb76579407bd0161d32a3c5f64f9b24f7512",
    },
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
CORE_DATABASE_SUFFIXES = (
    "",
    ".dbtype",
    ".index",
    ".lookup",
    ".source",
    "_h",
    "_h.dbtype",
    "_h.index",
)


class BundleError(ValueError):
    """The bundle does not match its pinned identity or manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    # MD5 is checked only because it is the digest UniProt publishes in its
    # metalink.  SHA-256 remains the bundle's content identity.
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BundleError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BundleError(f"{label} must be a non-negative integer")
    return value


def _digest(value: Any, label: str, pattern: re.Pattern[str] = SHA256_RE) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise BundleError(f"{label} must be a valid lowercase hexadecimal digest")
    return value


def _relative_path(value: Any, root: Path, label: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise BundleError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise BundleError(f"{label} escapes the bundle root: {value!r}")
    path = root / relative
    if path.is_symlink():
        raise BundleError(f"{label} must not be a symlink: {value}")
    return relative.as_posix(), path


def database_tree_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["path"])):
        digest.update(
            f"{row['path']}\t{row['bytes']}\t{row['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _check_file(
    row: dict[str, Any],
    root: Path,
    label: str,
    *,
    deep: bool,
) -> tuple[str, Path]:
    relative, path = _relative_path(row.get("path"), root, f"{label}.path")
    if not path.is_file():
        raise BundleError(f"{label} is missing: {path}")
    expected_bytes = _integer(row.get("bytes"), f"{label}.bytes")
    if path.stat().st_size != expected_bytes:
        raise BundleError(
            f"{label} size mismatch: expected {expected_bytes}, got {path.stat().st_size}"
        )
    expected_sha256 = _digest(row.get("sha256"), f"{label}.sha256")
    if deep:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise BundleError(
                f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
    return relative, path


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise BundleError(f"{label} must be {expected!r}, got {actual!r}")


def check_bundle(
    manifest_path: Path,
    root: Path,
    *,
    deep: bool,
    expected_manifest_sha256: str | None,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise BundleError(f"manifest is missing: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None:
        _digest(expected_manifest_sha256, "--expected-manifest-sha256")
        if manifest_sha256 != expected_manifest_sha256:
            raise BundleError(
                "manifest SHA-256 mismatch: expected "
                f"{expected_manifest_sha256}, got {manifest_sha256}"
            )
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "manifest")

    _require_equal(manifest.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_equal(manifest.get("job_time_network"), "none", "job_time_network")

    mmseqs = _mapping(manifest.get("mmseqs"), "mmseqs")
    _require_equal(mmseqs.get("tag"), MMSEQS_TAG, "mmseqs.tag")
    _require_equal(mmseqs.get("commit"), MMSEQS_COMMIT, "mmseqs.commit")
    artifact = _mapping(mmseqs.get("artifact"), "mmseqs.artifact")
    _require_equal(artifact.get("name"), MMSEQS_LINUX_AVX2_ARCHIVE, "mmseqs.artifact.name")
    _require_equal(artifact.get("bytes"), MMSEQS_LINUX_AVX2_BYTES, "mmseqs.artifact.bytes")
    _require_equal(
        artifact.get("sha256"), MMSEQS_LINUX_AVX2_SHA256, "mmseqs.artifact.sha256"
    )
    executable_relative, executable = _relative_path(
        mmseqs.get("executable"), root, "mmseqs.executable"
    )
    if not executable.is_file():
        raise BundleError(f"MMseqs2 executable is missing: {executable}")
    executable_sha256 = _digest(
        mmseqs.get("executable_sha256"), "mmseqs.executable_sha256"
    )
    if sha256_file(executable) != executable_sha256:
        raise BundleError("MMseqs2 executable SHA-256 does not match the manifest")
    try:
        version = subprocess.run(
            [str(executable), "version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise BundleError(f"MMseqs2 version probe failed: {exc}") from exc
    _require_equal(version, MMSEQS_COMMIT, "MMseqs2 version output")

    uniref90 = _mapping(manifest.get("uniref90"), "uniref90")
    _require_equal(uniref90.get("release"), UNIREF90_RELEASE, "uniref90.release")
    _require_equal(
        uniref90.get("release_date"), UNIREF90_RELEASE_DATE, "uniref90.release_date"
    )
    _require_equal(
        uniref90.get("cluster_count"), UNIREF90_CLUSTER_COUNT, "uniref90.cluster_count"
    )
    fasta = _mapping(uniref90.get("fasta"), "uniref90.fasta")
    _require_equal(fasta.get("url"), UNIREF90_FASTA_URL, "uniref90.fasta.url")
    _require_equal(fasta.get("bytes"), UNIREF90_FASTA_BYTES, "uniref90.fasta.bytes")
    _require_equal(fasta.get("upstream_md5"), UNIREF90_FASTA_MD5, "uniref90.fasta.upstream_md5")
    _digest(fasta.get("sha256"), "uniref90.fasta.sha256")
    retained_fasta = fasta.get("retained_path")
    if retained_fasta is not None:
        fasta_relative, fasta_path = _relative_path(
            retained_fasta, root, "uniref90.fasta.retained_path"
        )
        if not fasta_path.is_file():
            raise BundleError(f"retained UniRef90 FASTA is missing: {fasta_path}")
        if fasta_path.stat().st_size != UNIREF90_FASTA_BYTES:
            raise BundleError(
                f"retained UniRef90 FASTA size mismatch at {fasta_relative}"
            )
        if deep:
            if md5_file(fasta_path) != UNIREF90_FASTA_MD5:
                raise BundleError("retained UniRef90 FASTA does not match the upstream MD5")
            if sha256_file(fasta_path) != fasta["sha256"]:
                raise BundleError("retained UniRef90 FASTA does not match its SHA-256")

    metadata = _mapping(uniref90.get("metadata"), "uniref90.metadata")
    _require_equal(set(metadata), set(EXPECTED_METADATA), "uniref90.metadata names")
    for name, expected in EXPECTED_METADATA.items():
        row = _mapping(metadata[name], f"uniref90.metadata.{name}")
        _require_equal(row.get("bytes"), expected["bytes"], f"metadata {name} bytes")
        _require_equal(row.get("sha256"), expected["sha256"], f"metadata {name} SHA-256")
        retained_path = row.get("retained_path")
        if retained_path is not None:
            _check_file(row | {"path": retained_path}, root, f"metadata {name}", deep=deep)

    database = _mapping(manifest.get("database"), "database")
    prefix_relative, prefix = _relative_path(database.get("prefix"), root, "database.prefix")
    commands = _mapping(database.get("commands"), "database.commands")
    createdb = commands.get("createdb")
    if not isinstance(createdb, list) or not all(
        isinstance(item, str) and item for item in createdb
    ):
        raise BundleError("database.commands.createdb must be a non-empty argv array")
    if createdb[:2] != ["mmseqs", "createdb"]:
        raise BundleError("database.commands.createdb must begin with ['mmseqs', 'createdb']")
    required_createdb_options = {
        "--dbtype": "1",
        "--createdb-mode": "0",
        "--shuffle": "0",
        "--write-lookup": "1",
        "--compressed": "1",
    }
    for option, expected_value in required_createdb_options.items():
        try:
            option_index = createdb.index(option)
            actual_value = createdb[option_index + 1]
        except (ValueError, IndexError) as exc:
            raise BundleError(
                f"database.commands.createdb must include {option} {expected_value}"
            ) from exc
        if actual_value != expected_value:
            raise BundleError(
                f"database.commands.createdb must include {option} {expected_value}, "
                f"got {actual_value!r}"
            )
    createindex = commands.get("createindex")
    if createindex is not None and (
        not isinstance(createindex, list)
        or createindex[:2] != ["mmseqs", "createindex"]
        or not all(isinstance(item, str) and item for item in createindex)
    ):
        raise BundleError(
            "database.commands.createindex must be null or an argv array beginning "
            "with ['mmseqs', 'createindex']"
        )

    rows_value = database.get("files")
    if not isinstance(rows_value, list) or not rows_value:
        raise BundleError("database.files must be a non-empty array")
    rows: list[dict[str, Any]] = []
    declared_paths: set[str] = set()
    for index, value in enumerate(rows_value):
        row = _mapping(value, f"database.files[{index}]")
        relative, _ = _check_file(row, root, f"database.files[{index}]", deep=deep)
        if relative in declared_paths:
            raise BundleError(f"database.files repeats {relative}")
        declared_paths.add(relative)
        rows.append({"path": relative, "bytes": row["bytes"], "sha256": row["sha256"]})

    required_paths = {f"{prefix_relative}{suffix}" for suffix in CORE_DATABASE_SUFFIXES}
    missing_core = sorted(required_paths - declared_paths)
    if missing_core:
        raise BundleError(f"database.files omits core MMseqs2 files: {missing_core}")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in prefix.parent.glob(f"{prefix.name}*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != declared_paths:
        raise BundleError(
            "database file set differs from the manifest: missing="
            f"{sorted(declared_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - declared_paths)}"
        )
    tree_sha256 = database_tree_sha256(rows)
    _require_equal(database.get("tree_sha256"), tree_sha256, "database.tree_sha256")
    declared_total = _integer(database.get("bytes"), "database.bytes")
    actual_total = sum(row["bytes"] for row in rows)
    _require_equal(declared_total, actual_total, "database.bytes")
    if _integer(database.get("residue_count"), "database.residue_count") < 1:
        raise BundleError("database.residue_count must be positive")

    try:
        dbtype = subprocess.run(
            [str(executable), "dbtype", str(prefix)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise BundleError(f"MMseqs2 database-type probe failed: {exc}") from exc
    if not dbtype.startswith("Aminoacid"):
        raise BundleError(f"MMseqs2 database is not amino-acid data: {dbtype!r}")

    return {
        "ok": True,
        "bundle_id": manifest.get("bundle_id"),
        "manifest_sha256": manifest_sha256,
        "mmseqs_executable": executable_relative,
        "mmseqs_version": version,
        "uniref90_release": UNIREF90_RELEASE,
        "database_prefix": prefix_relative,
        "database_tree_sha256": tree_sha256,
        "database_bytes": actual_total,
        "deep_hashes_checked": deep,
        "network_calls": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument(
        "--expected-manifest-sha256",
        help="Fail unless the manifest bytes have this externally pinned SHA-256.",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Hash the full FASTA (when retained), metadata, and every database file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = check_bundle(
            args.manifest,
            args.bundle_root.resolve(),
            deep=args.deep,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    except (BundleError, json.JSONDecodeError) as exc:
        print(f"MMseqs2/UniRef90 toolcheck failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
