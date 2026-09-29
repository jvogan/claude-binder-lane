#!/usr/bin/env python3
"""Parse and validate the target-preparer outputs for one lane phase.

The stage declares two, the target manifest and the RFdiffusion3 specification
the backbone generator reads. The lane compares this report against the declared
outputs and rejects a mismatch, so an output the parser refuses stops the stage.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

RFD3_SPECIFICATION_ARTIFACT_TYPE = "rfd3-specification"
# The two artifact types the target-preparer stage declares. Anything else in its
# outputs is a plan the adapter did not write, so the parser says so rather than
# reporting a file count that looks right.
PARSED_ARTIFACT_TYPES = ("target-manifest", RFD3_SPECIFICATION_ARTIFACT_TYPE)
HOTSPOT_STATUSES = frozenset({"complete", "partial", "altloc", "absent", "skipped"})
HOTSPOT_SKIP_REASONS = frozenset(
    {
        "unsupported_residue_type",
        "insufficient_modeled_side_chain",
        "altloc_without_usable_tips",
    }
)
HOTSPOT_ABSENT_REASON = "absent_from_normalized_structure"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one parser input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    """Load one JSON document from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write the parser result in the path the lane reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_path(template: str, *, attempt_dir: Path, phase: str, stage_id: str) -> str:
    """Render the path tokens used by the target stage output contract."""
    rendered = (
        template.replace("{{attempt_dir}}", str(attempt_dir))
        .replace("{{phase}}", phase)
        .replace("{{stage_id}}", stage_id)
    )
    if "{{" in rendered or "}}" in rendered:
        raise ValueError(f"stage output path has an unresolved token: {template}")
    return rendered


def stage_record(config: dict[str, Any], stage_id: str) -> dict[str, Any]:
    """Return the configured target-preparation stage."""
    for stage in config.get("stages", []):
        if isinstance(stage, dict) and stage.get("stage_id") == stage_id:
            return stage
    raise ValueError(f"stage is not present in config: {stage_id}")


def _nonnegative_integer(value: Any) -> bool:
    """Return whether a manifest counter is a nonnegative integer."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_hotspot_status_manifest(document: dict[str, Any], path: Path, errors: list[str]) -> None:
    """Validate the classified degradation record attached to a target manifest."""
    status_manifest = document.get("hotspot_status_manifest")
    if status_manifest is None:
        if "rfd3_specification_path" in document:
            errors.append(f"{path}: target manifest has no hotspot_status_manifest")
        return
    if not isinstance(status_manifest, dict):
        errors.append(f"{path}: hotspot_status_manifest must be an object")
        return
    residues = status_manifest.get("residues")
    if not isinstance(residues, list):
        errors.append(f"{path}: hotspot_status_manifest.residues must be a list")
        return
    counts = {status: 0 for status in HOTSPOT_STATUSES}
    included_count = 0
    for index, residue in enumerate(residues):
        label = f"{path}: hotspot_status_manifest.residues[{index}]"
        if not isinstance(residue, dict):
            errors.append(f"{label} must be an object")
            continue
        residue_id = residue.get("residue_id")
        status = residue.get("status")
        included = residue.get("included_in_generation")
        if not isinstance(residue_id, str) or not residue_id:
            errors.append(f"{label}.residue_id must be a non-empty string")
        if status not in HOTSPOT_STATUSES:
            errors.append(f"{label}.status is not registered: {status!r}")
            continue
        if not isinstance(included, bool):
            errors.append(f"{label}.included_in_generation must be boolean")
        elif included:
            included_count += 1
        if status in {"complete", "partial", "altloc"} and included is not True:
            errors.append(f"{label} has status {status} without generation inclusion")
        if status in {"absent", "skipped"} and included is not False:
            errors.append(f"{label} has status {status} with generation inclusion")
        reason = residue.get("reason_code")
        if status == "skipped" and reason not in HOTSPOT_SKIP_REASONS:
            errors.append(f"{label}.reason_code is not a registered skip reason: {reason!r}")
        if status == "absent" and reason != HOTSPOT_ABSENT_REASON:
            errors.append(f"{label}.reason_code must be {HOTSPOT_ABSENT_REASON!r}")
        counts[status] += 1
    declared_count = status_manifest.get("declared_hotspot_count")
    manifest_included_count = status_manifest.get("included_hotspot_count")
    if not _nonnegative_integer(declared_count) or declared_count != len(residues):
        errors.append(f"{path}: hotspot_status_manifest.declared_hotspot_count disagrees with residues")
    if not _nonnegative_integer(manifest_included_count) or manifest_included_count != included_count:
        errors.append(f"{path}: hotspot_status_manifest.included_hotspot_count disagrees with residues")
    if status_manifest.get("reduced") != (included_count != len(residues)):
        errors.append(f"{path}: hotspot_status_manifest.reduced disagrees with residue inclusion")
    observed_counts = status_manifest.get("status_counts")
    if observed_counts != {status: counts[status] for status in sorted(HOTSPOT_STATUSES)}:
        errors.append(f"{path}: hotspot_status_manifest.status_counts disagrees with residues")


def parse(args: argparse.Namespace) -> int:
    """Parse target-manifest JSON and write the lane parser result."""
    attempt_dir = args.attempt_dir.resolve()
    parser_path = attempt_dir / args.phase / "parser-result.json"
    files: list[Path] = []
    errors: list[str] = []
    parsed_count = 0

    try:
        config = load_json(args.config)
        stage = stage_record(config, args.stage)
    except Exception as exc:
        errors.append(f"configuration is invalid: {type(exc).__name__}: {exc}")
        stage = {}

    for output in stage.get("outputs", []):
        if not isinstance(output, dict):
            errors.append("stage output contract is not an object")
            continue
        artifact_type = output.get("artifact_type")
        if artifact_type not in PARSED_ARTIFACT_TYPES:
            errors.append(
                "target-preparer parser expected one of "
                f"{', '.join(PARSED_ARTIFACT_TYPES)}, got {artifact_type}"
            )
        if output.get("kind") != "json":
            errors.append(
                "target-preparer parser expected a JSON output, "
                f"got {output.get('kind')}"
            )
        try:
            pattern = render_path(
                str(output["path_template"]),
                attempt_dir=attempt_dir,
                phase=args.phase,
                stage_id=args.stage,
            )
        except Exception as exc:
            errors.append(f"stage output path is invalid: {type(exc).__name__}: {exc}")
            continue
        for value in sorted(glob.glob(pattern, recursive=True)):
            path = Path(value).resolve()
            if not path.is_file():
                continue
            if attempt_dir not in path.parents:
                errors.append(f"stage output escapes attempt directory: {path}")
                continue
            files.append(path)
            parsed_count += 1
            try:
                document = load_json(path)
                if not isinstance(document, dict):
                    errors.append(f"{path}: {artifact_type} must be a JSON object")
                    continue
                if artifact_type == RFD3_SPECIFICATION_ARTIFACT_TYPE:
                    # `derive_specification` keys the document by campaign and target,
                    # so there is no fixed key to check. That it parses as an object is
                    # the whole check the parser can make. RFdiffusion3 reads the
                    # fields, and the run has no local copy of that reader.
                    continue
                if document.get("artifact_type") != "target-manifest":
                    errors.append(f"{path}: artifact_type is not target-manifest")
                validate_hotspot_status_manifest(document, path, errors)
                missing = [
                    field
                    for field in output.get("required_fields", [])
                    if field not in document
                ]
                if missing:
                    errors.append(f"{path}: missing required fields: {', '.join(missing)}")
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")

    result = {
        "ok": bool(files) and not errors,
        "parsed_count": parsed_count,
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": sorted(sha256_file(path) for path in files),
    }
    write_json(parser_path, result)
    return 0 if result["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    return parse(args)


if __name__ == "__main__":
    raise SystemExit(main())
