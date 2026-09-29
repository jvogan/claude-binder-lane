"""Shared input and render-output parsing helpers for adapters."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def read_jsonl(
    path: Path,
    label: str,
    *,
    error_type: type[Exception],
) -> list[dict[str, Any]]:
    """Read a non-empty JSONL file containing only object records."""
    if not path.is_file():
        raise error_type(f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception as exc:
            raise error_type(
                f"{label} is invalid: {path} line {line_number}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise error_type(f"{label} line {line_number} is not a JSON object: {path}")
        rows.append(value)
    if not rows:
        raise error_type(f"{label} has no records: {path}")
    return rows


def parse_declared_output(
    path: Path,
    kind: str,
    *,
    load_json: Callable[[Path], Any],
    json_loads: Callable[[str], Any],
    image_kind: str,
    png_signature: bytes,
    error_type: type[Exception],
) -> int:
    """Validate one declared render output and return its record count."""
    if not path.is_file() or path.stat().st_size == 0:
        raise error_type(f"declared output is missing or empty: {path}")
    if kind == "json":
        load_json(path)
    elif kind == "jsonl":
        records = [line for line in path.read_text().splitlines() if line.strip()]
        for line in records:
            if not isinstance(json_loads(line), dict):
                raise error_type(f"JSONL record is not an object: {path}")
        return len(records)
    elif kind == image_kind:
        signature = path.read_bytes()[: len(png_signature)]
        if signature != png_signature:
            raise error_type(
                f"declared PNG has an invalid signature: {path}; "
                f"expected {png_signature.hex()}, found {signature.hex()}"
            )
        return 1
    return 1


def parse_render_outputs(
    args: argparse.Namespace,
    *,
    stage_record: Callable[[Path, str], Mapping[str, Any]],
    output_pattern: Callable[..., str],
    parse_output: Callable[[Path, str], int],
    sha256_file: Callable[[Path], str],
    write_json: Callable[[Path, Any], None],
    path_factory: Callable[[str], Path],
    glob_matches: Callable[..., list[str]],
    error_type: type[Exception],
) -> int:
    """Parse a renderer stage's declared files into its parser result."""
    attempt_dir = args.attempt_dir.expanduser().resolve()
    stage = stage_record(args.config.expanduser().resolve(), args.stage)
    outputs: list[dict[str, Any]] = []
    errors: list[str] = []
    for contract in stage.get("outputs", []):
        template = contract.get("path_template")
        if not isinstance(template, str):
            errors.append("render stage output has no path_template")
            continue
        try:
            pattern = output_pattern(template, attempt_dir=attempt_dir, phase=args.phase)
            matches = sorted(path_factory(value) for value in glob_matches(pattern, recursive=True))
            files: list[dict[str, Any]] = []
            if not matches:
                errors.append(f"{contract.get('artifact_id')} matched no files: {pattern}")
            for path in matches:
                try:
                    records = parse_output(path, str(contract.get("kind", "file")))
                    files.append(
                        {
                            "path": str(path),
                            "records": records,
                            "sha256": sha256_file(path),
                            "bytes": path.stat().st_size,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{path}: {type(exc).__name__}: {exc}")
            outputs.append(
                {
                    "artifact_id": contract.get("artifact_id"),
                    "artifact_type": contract.get("artifact_type"),
                    "kind": contract.get("kind"),
                    "pattern": pattern,
                    "count": len(files),
                    "files": files,
                }
            )
        except error_type as exc:
            errors.append(str(exc))
    result = {
        "ok": not errors,
        "parsed_count": sum(file_record["records"] for output in outputs for file_record in output["files"]),
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": sorted(
            file_record["sha256"] for output in outputs for file_record in output["files"]
        ),
        "phase": args.phase,
        "attempt_dir": str(attempt_dir),
        "outputs": outputs,
    }
    result_path = attempt_dir / args.phase / "parser-result.json"
    write_json(result_path, result)
    print(
        f"view renderer adapter: parsed={result['parsed_count']} "
        f"files={len(result['source_output_hashes'])} ok={result['ok']}"
    )
    return 0 if result["ok"] else 1
