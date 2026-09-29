#!/usr/bin/env python3
"""Validate and count the declared outputs of the portfolio ranking stage."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ATOM_PREFIXES = ("ATOM  ", "HETATM")
REQUIRED_RANKED_FIELDS = {
    "ok",
    "controls",
    "portfolio",
    "ranking_receipt",
    "separability",
    "ranked_candidates",
    "unranked_candidates",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_pattern(value: str, attempt_dir: Path) -> Path | None:
    raw = Path(value)
    resolved = Path(os.path.normpath(attempt_dir / raw)) if not raw.is_absolute() else Path(os.path.normpath(raw))
    if attempt_dir not in resolved.resolve().parents:
        return None
    return resolved


def parse_ranked(path_value: str, attempt_dir: Path, errors: list[str]) -> list[Path]:
    path = resolve_pattern(path_value, attempt_dir)
    if path is None:
        errors.append(f"ranked portfolio escapes the attempt directory: {path_value}")
        return []
    if not path.is_file():
        errors.append(f"ranked portfolio is missing: {path}")
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"ranked portfolio is invalid: {path}: {type(exc).__name__}: {exc}")
        return [path]
    if not isinstance(value, dict):
        errors.append(f"ranked portfolio must be a JSON object: {path}")
    else:
        missing = sorted(REQUIRED_RANKED_FIELDS - set(value))
        errors.extend(f"ranked portfolio missing {field}: {path}" for field in missing)
        if value.get("ok") is not True:
            errors.append(f"ranked portfolio ok is not true: {path}")
        if not isinstance(value.get("controls"), dict):
            errors.append(f"ranked portfolio controls must be an object: {path}")
        if not isinstance(value.get("portfolio"), dict):
            errors.append(f"ranked portfolio portfolio must be an object: {path}")
        if not isinstance(value.get("ranking_receipt"), dict):
            errors.append(f"ranked portfolio ranking_receipt must be an object: {path}")
        if not isinstance(value.get("separability"), dict):
            errors.append(f"ranked portfolio separability must be an object: {path}")
        if not isinstance(value.get("unranked_candidates"), list):
            errors.append(f"ranked portfolio unranked_candidates must be a list: {path}")
        candidates = value.get("ranked_candidates")
        if not isinstance(candidates, list) or not candidates:
            errors.append(f"ranked portfolio ranked_candidates must be a non-empty list: {path}")
        elif any(not isinstance(row, dict) for row in candidates):
            errors.append(f"ranked portfolio ranked_candidates must contain objects: {path}")
    return [path]


def parse_fasta(patterns: list[str], attempt_dir: Path, errors: list[str]) -> tuple[int, list[Path]]:
    count = 0
    files: list[Path] = []
    for pattern in patterns:
        resolved = resolve_pattern(pattern, attempt_dir)
        if resolved is None:
            errors.append(f"declared FASTA output escapes the attempt directory: {pattern}")
            continue
        matches = sorted(Path(value) for value in glob.glob(str(resolved), recursive=True))
        if not matches:
            errors.append(f"declared FASTA output matched no files: {pattern}")
            continue
        for path in matches:
            if not path.is_file():
                continue
            if attempt_dir not in path.resolve().parents:
                errors.append(f"matched FASTA output resolves outside the attempt directory: {path}")
                continue
            files.append(path)
            records = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(">"))
            if records == 0:
                errors.append(f"FASTA has no records: {path}")
            count += records
    return count, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--json", action="append", default=None)
    parser.add_argument("--fasta", action="append", default=None)
    args = parser.parse_args()

    attempt_dir = args.attempt_dir.resolve()
    result_path = Path(os.path.normpath(attempt_dir / args.result_path)) if not args.result_path.is_absolute() else Path(os.path.normpath(args.result_path))
    errors: list[str] = []
    if attempt_dir not in result_path.resolve().parents:
        errors.append(f"result path escapes the attempt directory: {args.result_path}")
    json_patterns = args.json or []
    fasta_patterns = args.fasta or []
    if not json_patterns:
        errors.append("no ranked JSON output was declared")
    if not fasta_patterns:
        errors.append("no final FASTA output was declared")
    files: list[Path] = []
    for pattern in json_patterns:
        files.extend(parse_ranked(pattern, attempt_dir, errors))
    parsed_count = len([path for path in files if path.is_file()])
    fasta_count, fasta_files = parse_fasta(fasta_patterns, attempt_dir, errors)
    files.extend(fasta_files)
    parsed_count += fasta_count
    hashes = [sha256_file(path) for path in files if path.is_file()]
    if len(hashes) != len(set(hashes)):
        errors.append("declared outputs repeat file bytes, and the executor requires unique hashes")
    result = {
        "ok": bool(files) and not errors,
        "parsed_count": parsed_count,
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": sorted(hashes),
    }
    write_result(result_path, result)
    for error in errors:
        print(f"portfolio ranker parser: {error}", file=sys.stderr)
    print(f"portfolio ranker parser: phase={args.phase} parsed_count={parsed_count} files={len(files)} ok={result['ok']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
