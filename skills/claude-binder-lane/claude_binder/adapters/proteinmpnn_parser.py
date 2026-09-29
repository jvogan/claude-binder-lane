#!/usr/bin/env python3
"""Report on the declared outputs of one binder lane stage phase.

An adapter names this script in its `parser_argv_template`. Pass the phase, the
attempt directory, the result path, and one flag for every output glob the stage
contract declares. The script counts records, hashes complete file bytes, and
writes the parser result the executor reads.

The executor recounts the same files and compares the totals, so the record
rules here match the rules the lane runner applies:

  jsonl   one record for every non-blank line, and every line holds a JSON object
  json    one record for the file, and the file holds one JSON value
  fasta   one record for every line that starts with a greater-than sign
  pdb     one record for every line that starts with `ATOM  ` or `HETATM`
  cif     one record for the file, and the file holds an `_atom_site` category

Declare every output the stage declares. The executor sums records and file
hashes over all of them, so a glob you leave out fails the stage.

Every declared path stays under the attempt directory. A relative path joins to
the attempt directory, so the result does not depend on the working directory.

Sequence hashes are not file hashes. This script reports SHA-256 values over
complete file bytes. The adapter that writes a FASTA records the SHA-256 of the
canonical residue string in its own manifest.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

# Column 1 of a PDB coordinate line. The lane runner counts records with the
# same two prefixes, so keep the trailing spaces of `ATOM  `.
ATOM_RECORD_PREFIXES = ("ATOM  ", "HETATM")
READ_BLOCK_BYTES = 1024 * 1024
FAMILY_ORDER = ("jsonl", "json", "fasta", "pdb", "cif")
FAMILY_HELP = {
    "jsonl": "Glob for a declared JSONL output. Repeat the flag for more globs.",
    "json": "Glob for a declared JSON output. Repeat the flag for more globs.",
    "fasta": "Glob for a declared FASTA output. Repeat the flag for more globs.",
    "pdb": "Glob for a declared PDB output. Repeat the flag for more globs.",
    "cif": "Glob for a declared mmCIF output. Repeat the flag for more globs.",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of the complete file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    """Return the number of JSON object lines in a JSONL file."""
    records = 0
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        records += 1
    if records == 0:
        raise ValueError("file holds no JSONL records")
    return records


def count_json(path: Path) -> int:
    """Return one record for a file that holds one JSON value."""
    json.loads(path.read_text())
    return 1


def count_fasta(path: Path) -> int:
    """Return the number of FASTA header lines in a file."""
    records = sum(1 for line in path.read_text().splitlines() if line.startswith(">"))
    if records == 0:
        raise ValueError("file holds no FASTA records")
    return records


def count_pdb(path: Path) -> int:
    """Return the number of poses in a PDB file.

    Output contracts count poses, and the executor recomputes this the same way before it
    compares the total against the parser's ``parsed_count``. Counting coordinate lines instead
    reported 13,449 records for a stage whose contract declares 30, and the run stopped on a
    stage that had produced every file correctly. A PDB without MODEL delimiters is one pose.
    """
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    if not any(line.startswith(ATOM_RECORD_PREFIXES) for line in lines):
        raise ValueError("file holds no PDB atom records")
    models = sum(1 for line in lines if line.startswith("MODEL "))
    return models if models else 1


def count_cif(path: Path) -> int:
    """Return one record for an mmCIF file that carries coordinates."""
    if "_atom_site." not in path.read_text(errors="replace"):
        raise ValueError("file holds no mmCIF atom_site category")
    return 1


COUNTERS: dict[str, Callable[[Path], int]] = {
    "jsonl": count_jsonl,
    "json": count_json,
    "fasta": count_fasta,
    "pdb": count_pdb,
    "cif": count_cif,
}


def write_json(path: Path, value: Any) -> None:
    """Write JSON to a path in one atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def phase_parts(attempt_dir: Path, path: Path) -> tuple[str, ...]:
    """Return the path components below the attempt directory."""
    try:
        return path.relative_to(attempt_dir).parts
    except ValueError:
        return ()


def absolute_path(attempt_dir: Path, value: str) -> tuple[Path, bool]:
    """Return the absolute form of a declared path and whether it was joined.

    A relative path joins to the attempt directory rather than to the working
    directory, so a caller reads the same files wherever it runs. The join runs
    before the containment check, and `os.path.normpath` collapses `..` first,
    so a joined path that climbs out of the attempt directory still fails that
    check.
    """
    raw = Path(value)
    if raw.is_absolute():
        return Path(os.path.normpath(raw)).resolve(), False
    return Path(os.path.normpath(attempt_dir / raw)).resolve(), True


def containment_error(label: str, value: str, resolved: Path, joined: bool) -> str:
    """Return the message for a declared path outside the attempt directory."""
    if joined:
        return (
            f"{label} escapes the attempt directory after the join: "
            f"{value} joins to {resolved}"
        )
    return f"{label} escapes the attempt directory: {value}"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        required=True,
        help="Stage phase name, such as single, smoke, or scale. Every declared path carries it.",
    )
    parser.add_argument(
        "--attempt-dir",
        type=Path,
        required=True,
        help="Attempt directory that owns every declared output.",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        required=True,
        help=(
            "Path to write the parser result to. Keep it under the attempt directory. "
            "A relative path joins to the attempt directory."
        ),
    )
    for family in FAMILY_ORDER:
        parser.add_argument(
            f"--{family}",
            action="append",
            metavar="GLOB",
            default=None,
            help=FAMILY_HELP[family],
        )
    return parser.parse_args()


def scan_pattern(
    family: str,
    pattern: str,
    *,
    attempt_dir: Path,
    phase: str,
    claimed: dict[Path, str],
    errors: list[str],
) -> dict[str, Any]:
    """Count records and hash bytes for every file one declared glob matches."""
    entry: dict[str, Any] = {"family": family, "pattern": pattern, "files": []}
    normalized, joined = absolute_path(attempt_dir, pattern)
    entry["resolved_pattern"] = str(normalized)
    if attempt_dir not in normalized.parents:
        errors.append(containment_error("declared output", pattern, normalized, joined))
        return entry
    if phase not in phase_parts(attempt_dir, normalized):
        errors.append(f"declared output carries no {phase} phase component: {pattern}")
        return entry
    matches = sorted(Path(value) for value in glob.glob(str(normalized), recursive=True))
    files = [path for path in matches if path.is_file()]
    if not files:
        errors.append(f"declared output matched no files: {pattern}")
        return entry
    for path in files:
        resolved = path.resolve()
        if attempt_dir not in resolved.parents:
            errors.append(f"matched output resolves outside the attempt directory: {path}")
            continue
        owner = claimed.get(resolved)
        if owner is not None:
            errors.append(f"two declared outputs match the same file: {path} matches {owner} and {pattern}")
            continue
        claimed[resolved] = pattern
        try:
            records = COUNTERS[family](path)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        entry["files"].append(
            {
                "path": str(path),
                "records": records,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return entry


def build_result(
    args: argparse.Namespace, *, attempt_dir: Path, result_path: Path, result_joined: bool
) -> dict[str, Any]:
    """Return the parser result for every declared output of one phase."""
    errors: list[str] = []
    outputs: list[dict[str, Any]] = []
    claimed: dict[Path, str] = {}
    if not attempt_dir.is_dir():
        errors.append(f"attempt directory not found: {attempt_dir}")
    if attempt_dir not in result_path.parents:
        errors.append(
            containment_error("result path", str(args.result_path), result_path, result_joined)
        )
    elif args.phase not in phase_parts(attempt_dir, result_path):
        errors.append(f"result path carries no {args.phase} phase component: {args.result_path}")
    declared = [
        (family, pattern)
        for family in FAMILY_ORDER
        for pattern in getattr(args, family) or []
    ]
    if not declared:
        errors.append("no declared output glob was given")
    for family, pattern in declared:
        outputs.append(
            scan_pattern(
                family,
                pattern,
                attempt_dir=attempt_dir,
                phase=args.phase,
                claimed=claimed,
                errors=errors,
            )
        )
    hashes = [file_row["sha256"] for entry in outputs for file_row in entry["files"]]
    if len(hashes) != len(set(hashes)):
        repeated = sorted({value for value in hashes if hashes.count(value) > 1})
        errors.append(
            "declared outputs repeat file bytes, and the executor requires unique hashes: "
            + ", ".join(repeated[:5])
        )
    parsed_count = sum(int(file_row["records"]) for entry in outputs for file_row in entry["files"])
    return {
        "ok": not errors,
        "parsed_count": parsed_count,
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": sorted(hashes),
        "phase": args.phase,
        "attempt_dir": str(attempt_dir),
        "result_path": str(result_path),
        "outputs": outputs,
    }


def main() -> int:
    args = parse_arguments()
    attempt_dir = args.attempt_dir.resolve()
    result_path, result_joined = absolute_path(attempt_dir, str(args.result_path))
    result = build_result(
        args, attempt_dir=attempt_dir, result_path=result_path, result_joined=result_joined
    )
    write_json(result_path, result)
    for error in result["errors"]:
        print(f"parser: {error}")
    print(
        f"parser: phase={result['phase']} parsed_count={result['parsed_count']} "
        f"files={len(result['source_output_hashes'])} ok={result['ok']}"
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
