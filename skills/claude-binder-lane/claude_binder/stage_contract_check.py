#!/usr/bin/env python3
"""Validate Claude Binder stage contracts and progress ledgers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_STAGE_KEYS = {
    "stage_id",
    "description",
    "expected_outputs",
    "timeout_minutes",
    "checkpoint_marker",
    "done_marker",
    "resume_command",
    "fail_closed",
}

VALID_STATUSES = {"pending", "started", "running", "heartbeat", "completed", "failed", "partial", "skipped"}
TERMINAL_STATUSES = {"completed", "failed", "partial", "skipped"}
PARTIAL_SUMMARY_REQUIRED_INCLUDE = {
    "completed_stages",
    "failed_stage",
    "resume_command",
    "artifact_status",
    "claim_level",
}
STALE_OUTPUT_REQUIRED_KEYS = {
    "hash_inputs",
    "hash_run_support",
    "invalidate_on",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
        event["_line"] = line_number
        events.append(event)
    return events


def stage_ids(contract: dict[str, Any]) -> list[str]:
    return [str(stage.get("stage_id", "")) for stage in contract.get("stages", [])]


def validate_contract(contract: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if contract.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not contract.get("contract_id"):
        errors.append("contract_id is required")
    if not contract.get("run_id"):
        errors.append("run_id is required")
    if contract.get("fail_closed") is not True:
        errors.append("fail_closed must be true")
    progress_ledger = contract.get("progress_ledger")
    if not isinstance(progress_ledger, str) or not Path(progress_ledger).is_absolute():
        errors.append("progress_ledger must be an absolute path")
    partial_summary = contract.get("partial_summary")
    if not isinstance(partial_summary, dict):
        errors.append("partial_summary policy is required")
    else:
        if not isinstance(partial_summary.get("path"), str) or not partial_summary.get("path"):
            errors.append("partial_summary.path is required")
        elif str(partial_summary.get("path")).startswith("/"):
            warnings.append("partial_summary.path is absolute; prefer run-root-relative paths")
        write_on = partial_summary.get("write_on")
        if not isinstance(write_on, list) or not {"failed", "partial"}.issubset(set(write_on)):
            errors.append("partial_summary.write_on must include failed and partial")
        include = partial_summary.get("include")
        if not isinstance(include, list):
            errors.append("partial_summary.include must be a list")
        else:
            missing = sorted(PARTIAL_SUMMARY_REQUIRED_INCLUDE - set(include))
            for item in missing:
                errors.append(f"partial_summary.include missing {item}")
    stale_policy = contract.get("stale_output_policy")
    if not isinstance(stale_policy, dict):
        errors.append("stale_output_policy is required")
    else:
        for key in sorted(STALE_OUTPUT_REQUIRED_KEYS - set(stale_policy)):
            errors.append(f"stale_output_policy missing {key}")
        if stale_policy.get("hash_inputs") is not True:
            errors.append("stale_output_policy.hash_inputs must be true")
        if stale_policy.get("hash_run_support") is not True:
            errors.append("stale_output_policy.hash_run_support must be true")
        invalidate_on = stale_policy.get("invalidate_on")
        if not isinstance(invalidate_on, list) or not invalidate_on:
            errors.append("stale_output_policy.invalidate_on must be a non-empty list")

    stages = contract.get("stages")
    if not isinstance(stages, list) or not stages:
        errors.append("stages must be a non-empty list")
        return errors, warnings

    seen: set[str] = set()
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            errors.append(f"stage {index} must be object")
            continue
        missing = sorted(REQUIRED_STAGE_KEYS - set(stage))
        for key in missing:
            errors.append(f"stage {index} missing {key}")
        stage_id = str(stage.get("stage_id", ""))
        if not stage_id:
            errors.append(f"stage {index} has empty stage_id")
        elif stage_id in seen:
            errors.append(f"duplicate stage_id: {stage_id}")
        seen.add(stage_id)
        outputs = stage.get("expected_outputs")
        if not isinstance(outputs, list) or not outputs or any(not isinstance(item, str) or not item for item in outputs):
            errors.append(f"stage {stage_id or index} expected_outputs must be a non-empty string list")
        timeout = stage.get("timeout_minutes")
        if not isinstance(timeout, int) or timeout <= 0:
            errors.append(f"stage {stage_id or index} timeout_minutes must be a positive integer")
        if stage.get("fail_closed") is not True:
            errors.append(f"stage {stage_id or index} fail_closed must be true")
        for marker_key in ["checkpoint_marker", "done_marker"]:
            marker = stage.get(marker_key)
            if not isinstance(marker, str) or not marker:
                errors.append(f"stage {stage_id or index} {marker_key} must be non-empty")
            elif marker.startswith("/"):
                warnings.append(f"stage {stage_id or index} {marker_key} is absolute; prefer run-root-relative markers")
        if not isinstance(stage.get("resume_command"), str) or not stage.get("resume_command"):
            errors.append(f"stage {stage_id or index} resume_command must be non-empty")

    return errors, warnings


def validate_progress(
    contract: dict[str, Any],
    events: list[dict[str, Any]],
    require_terminal: bool,
    skip_stages: set[str] | None = None,
) -> tuple[list[str], list[str], dict[str, str]]:
    errors: list[str] = []
    warnings: list[str] = []
    allowed_stage_ids = set(stage_ids(contract))
    skipped = skip_stages or set()
    terminal_by_stage: dict[str, str] = {}

    if not events:
        errors.append("progress ledger is empty")
        return errors, warnings, terminal_by_stage

    for event in events:
        stage_id = event.get("stage_id")
        status = event.get("status")
        if stage_id not in allowed_stage_ids:
            errors.append(f"progress line {event.get('_line')} references unknown stage_id: {stage_id}")
        if status not in VALID_STATUSES:
            errors.append(f"progress line {event.get('_line')} has invalid status: {status}")
        if not event.get("timestamp"):
            errors.append(f"progress line {event.get('_line')} missing timestamp")
        if status == "started" and isinstance(stage_id, str):
            terminal_by_stage.pop(stage_id, None)
        elif status in TERMINAL_STATUSES and isinstance(stage_id, str):
            terminal_by_stage[stage_id] = str(status)
        if event.get("fallback_used") is True and event.get("status") == "completed":
            errors.append(f"progress line {event.get('_line')} marks fallback_used with completed status; use partial/degraded closeout")

    if require_terminal:
        for stage in contract.get("stages", []):
            if stage.get("required", True) is False:
                continue
            stage_id = str(stage.get("stage_id"))
            if stage_id in skipped:
                continue
            status = terminal_by_stage.get(stage_id)
            if status is None:
                errors.append(f"required stage has no terminal progress event: {stage_id}")
            elif status != "completed":
                errors.append(f"required stage did not complete: {stage_id} -> {status}")
    else:
        missing = [stage_id for stage_id in allowed_stage_ids if stage_id not in terminal_by_stage and stage_id not in skipped]
        if missing:
            warnings.append(f"non-terminal prep progress for stages: {', '.join(sorted(missing))}")

    return errors, warnings, terminal_by_stage


def evaluate(
    stage_contract: Path,
    progress_jsonl: Path | None = None,
    require_terminal: bool = False,
    skip_stages: set[str] | None = None,
) -> dict[str, Any]:
    contract = load_json(stage_contract)
    errors, warnings = validate_contract(contract)
    events: list[dict[str, Any]] = []
    terminal_by_stage: dict[str, str] = {}
    skipped = skip_stages or set()
    allowed_stage_ids = set(stage_ids(contract))
    for stage_id in sorted(skipped - allowed_stage_ids):
        errors.append(f"skip stage is not in contract: {stage_id}")

    if progress_jsonl is not None:
        try:
            events = load_jsonl(progress_jsonl)
            progress_errors, progress_warnings, terminal_by_stage = validate_progress(
                contract,
                events,
                require_terminal,
                skipped,
            )
            errors.extend(progress_errors)
            warnings.extend(progress_warnings)
        except Exception as exc:
            errors.append(f"could not read progress_jsonl: {type(exc).__name__}: {exc}")
    elif require_terminal:
        errors.append("progress_jsonl is required when --require-terminal is set")

    return {
        "ok": not errors,
        "check_type": "claude_binder_stage_contract_check",
        "stage_contract": str(stage_contract.resolve()),
        "progress_jsonl": str(progress_jsonl.resolve()) if progress_jsonl else None,
        "require_terminal": require_terminal,
        "skipped_stages": sorted(skipped),
        "stage_count": len(contract.get("stages", [])) if isinstance(contract.get("stages"), list) else 0,
        "progress_event_count": len(events),
        "terminal_by_stage": terminal_by_stage,
        "errors": errors,
        "warnings": warnings,
    }


def emit_event(stage_id: str, status: str, message: str = "") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage_id": stage_id,
        "status": status,
        "message": message,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-contract", type=Path, required=True)
    parser.add_argument("--progress-jsonl", type=Path)
    parser.add_argument("--require-terminal", action="store_true")
    parser.add_argument("--skip-stage", action="append", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = evaluate(args.stage_contract, args.progress_jsonl, args.require_terminal, set(args.skip_stage))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        summary["report_path"] = str(args.out.resolve())

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"ok: {summary['ok']}")
        for warning in summary["warnings"]:
            print(f"warning: {warning}")
        for error in summary["errors"]:
            print(f"error: {error}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
