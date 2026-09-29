#!/usr/bin/env python3
"""Apply the configured sequence-integrity contracts to a candidate lineage.

The adapter reads the receipt-published candidate manifest and the owned FASTA files.
It verifies the canonical sequence hash before measuring canonical protein composition,
exact sequence uniqueness within the cohort, and configurable sequence liabilities.
Contract thresholds and provenance are copied from the campaign config. The adapter writes
the observation matrix and the subset that passes every integrity contract.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from claude_binder.adapters.adapter_io import read_jsonl as _read_jsonl

from ..filter_contracts import filter_report, is_not_applicable, reference_digest_is_required


CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STAGE_INTEGRITY = "filter-integrity"
SUPPORTED_FILTERS = frozenset(
    {"composition", "exact_duplicates", "liability_chemistry"}
)

# The protocol names these liability classes without numeric defaults. These
# product defaults are therefore judgement calls. Campaigns can override every
# numeric threshold under filters.liability_rules.
DEFAULT_LIABILITY_RULES = {
    "reject_odd_cysteine_count": True,
    "maximum_homopolymer_run": 4,
    "hydrophobic_window_length": 12,
    "maximum_hydrophobic_residues": 7,
}
HYDROPHOBIC_AMINO_ACIDS = frozenset("AVILMFWY")


class AdapterError(RuntimeError):
    """An input, contract, or output condition that must stop the adapter."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_sequence(sequence: str) -> str:
    try:
        payload = sequence.encode("ascii")
    except UnicodeEncodeError as exc:
        raise AdapterError("candidate sequence contains a non-ASCII residue") from exc
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AdapterError(f"{label} is invalid: {path}: {type(exc).__name__}: {exc}") from exc


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path, label)
    if not isinstance(value, dict):
        raise AdapterError(f"{label} must be a JSON object: {path}")
    return value


def declared_input_ref(
    plan_path: Path,
    stage_id: str,
    artifact_ids: set[str],
) -> tuple[str, str] | None:
    """Return the one declared input matching ``artifact_ids``.

    An empty plan is accepted only for the small unit fixtures that predate the
    resolved run plan. A real plan must name the input explicitly. In
    particular, a missing input never falls through to an attempt-local file.
    """
    resolved_plan = plan_path.resolve()
    if not resolved_plan.is_file():
        return None
    plan = read_json_object(resolved_plan, "run plan")
    stages = plan.get("stages")
    if not isinstance(stages, list):
        return None
    if not stages:
        return None
    matches = [
        stage for stage in stages
        if isinstance(stage, dict) and stage.get("stage_id") == stage_id
    ]
    if len(matches) != 1:
        raise AdapterError(f"run plan must define one stage: {stage_id}")
    inputs = matches[0].get("inputs")
    if not isinstance(inputs, list):
        raise AdapterError(f"stage {stage_id} inputs must be a list")
    matching: list[str] = []
    for input_ref in inputs:
        if not isinstance(input_ref, str) or ":" not in input_ref:
            continue
        source_stage, artifact_id = input_ref.split(":", 1)
        if artifact_id in artifact_ids:
            matching.append(f"{source_stage}:{artifact_id}")
    if len(matching) != 1:
        expected = ", ".join(sorted(artifact_ids))
        raise AdapterError(
            f"stage {stage_id} must declare exactly one input with artifact id in {{{expected}}}; "
            f"declared inputs={inputs!r}"
        )
    return tuple(matching[0].split(":", 1))


def declared_artifact_refs(plan_path: Path) -> set[str] | None:
    """Return all qualified output artifact ids in a resolved plan."""
    resolved_plan = plan_path.resolve()
    if not resolved_plan.is_file():
        return None
    plan = read_json_object(resolved_plan, "run plan")
    stages = plan.get("stages")
    if not isinstance(stages, list):
        return None
    refs: set[str] = set()
    for stage in stages:
        if not isinstance(stage, dict) or not isinstance(stage.get("stage_id"), str):
            continue
        outputs = stage.get("outputs")
        if not isinstance(outputs, list):
            continue
        for output in outputs:
            if isinstance(output, dict) and isinstance(output.get("artifact_id"), str):
                refs.add(f"{stage['stage_id']}:{output['artifact_id']}")
    return refs


def completed_receipt_file(
    receipts_dir: Path,
    stage_id: str,
    artifact_id: str,
    *,
    label: str,
) -> Path:
    """Resolve one file from a completed receipt and one exact artifact id."""
    receipt_path = receipts_dir.expanduser().resolve() / f"{stage_id}.json"
    receipt = read_json_object(receipt_path, f"{label} receipt")
    if receipt.get("ok") is not True:
        raise AdapterError(f"{label} receipt did not complete: {receipt_path}")
    output_manifest = receipt.get("output_manifest")
    if not isinstance(output_manifest, dict):
        raise AdapterError(f"{label} receipt has no output_manifest object: {receipt_path}")
    artifacts = output_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise AdapterError(f"{label} receipt has no artifact list: {receipt_path}")
    phases = {
        str(item.get("phase"))
        for item in artifacts
        if isinstance(item, dict) and item.get("phase") in {"scale", "single"}
    }
    selected_phase = "scale" if "scale" in phases else "single" if "single" in phases else ""
    if not selected_phase:
        raise AdapterError(f"{label} receipt has no scale or single artifact phase: {receipt_path}")
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact.get("phase") == selected_phase
        and artifact.get("artifact_id") == artifact_id
    ]
    if len(matches) != 1:
        raise AdapterError(
            f"{label} receipt {receipt_path} carries {len(matches)} {artifact_id} artifacts "
            f"for phase {selected_phase}; expected exactly one"
        )
    files = matches[0].get("files")
    if not isinstance(files, list) or len(files) != 1:
        count = len(files) if isinstance(files, list) else 0
        raise AdapterError(
            f"{label} receipt {receipt_path} carries {count} files for {artifact_id}; "
            "expected exactly one"
        )
    file_record = files[0]
    if not isinstance(file_record, dict) or not isinstance(file_record.get("path"), str):
        raise AdapterError(f"{label} receipt has an invalid {artifact_id} file path: {receipt_path}")
    path = Path(file_record["path"]).expanduser().resolve()
    if not path.is_file():
        raise AdapterError(f"{label} artifact {artifact_id} is missing: {path}")
    return path


def checked_override(path_value: Path | None, label: str) -> Path | None:
    """Validate an explicit input override without making it the default."""
    if path_value is None:
        return None
    path = path_value.expanduser().resolve()
    if not path.is_file():
        raise AdapterError(f"{label} override is missing: {path}")
    return path


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    return _read_jsonl(path, label, error_type=AdapterError)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def stage_round(stage: str, config: dict[str, Any]) -> int:
    if stage == STAGE_INTEGRITY:
        return 0
    match = re.fullmatch(r"optimization-filter-integrity-round-(\d+)", stage)
    if match is not None:
        return int(match.group(1))
    for item in config.get("stages", []):
        if isinstance(item, dict) and item.get("stage_id") == stage:
            value = item.get("optimization_round", 0)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    raise AdapterError(f"stage is not an integrity-filter stage: {stage}")


def candidate_manifest_path(args: argparse.Namespace, config: dict[str, Any], round_number: int) -> Path:
    override = checked_override(getattr(args, "candidate_manifest", None), "candidate manifest")
    if override is not None:
        return override

    declared = declared_input_ref(
        args.plan,
        args.stage,
        {"normalized-candidates", "optimized-candidates"},
    )
    if declared is not None:
        return completed_receipt_file(
            args.receipts_dir,
            declared[0],
            declared[1],
            label=f"declared input {declared[0]}:{declared[1]}",
        )

    # Legacy unit fixtures pass an empty plan. Keep their published artifact
    # route, but never use the current attempt directory as an undeclared input.
    artifact_root = args.artifact_root.resolve()
    if round_number:
        path = artifact_root / "optimization" / "rounds" / f"round-{round_number}" / "optimized-candidates.jsonl"
    else:
        path = artifact_root / "candidates" / "candidate-manifest.jsonl"
    if path.is_file():
        return path
    raise AdapterError(f"declared candidate input is missing: {path}")


def integrity_passing_manifest_path(args: argparse.Namespace, round_number: int) -> Path:
    """Resolve the declared integrity-passing input for the novelty stage."""
    override = checked_override(
        getattr(args, "integrity_passing_manifest", None),
        "integrity-passing manifest",
    )
    if override is not None:
        return override

    declared = declared_input_ref(
        args.plan,
        args.stage,
        {"integrity-passing-candidates"},
    )
    if declared is not None:
        return completed_receipt_file(
            args.receipts_dir,
            declared[0],
            declared[1],
            label=f"declared input {declared[0]}:{declared[1]}",
        )

    # Legacy unit fixtures pass an empty plan. This published path is retained
    # only for those fixtures. It is not an attempt-local fallback.
    artifact_root = args.artifact_root.resolve()
    if round_number:
        path = artifact_root / "optimization" / "rounds" / f"round-{round_number}" / "filters" / "integrity-passing-candidates.jsonl"
    else:
        path = artifact_root / "filters" / "integrity-passing-candidates.jsonl"
    if path.is_file():
        return path
    raise AdapterError(f"declared integrity-passing input is missing: {path}")


def load_candidates(path: Path, expected_round: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows = read_jsonl(path, "candidate lineage manifest")
    by_id: dict[str, dict[str, Any]] = {}
    sequences: dict[str, str] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise AdapterError(f"candidate lineage row has no candidate_id: {path}")
        if candidate_id in by_id:
            raise AdapterError(f"candidate lineage has duplicate candidate_id: {candidate_id}")
        if row.get("optimization_round") != expected_round:
            raise AdapterError(
                f"candidate {candidate_id} has optimization_round {row.get('optimization_round')!r}; "
                f"expected {expected_round} from the filter stage"
            )
        origin = row.get("origin_generator")
        if not isinstance(origin, str) or not origin:
            raise AdapterError(f"candidate {candidate_id} has no origin_generator")
        sequence_path_value = row.get("sequence_path")
        if not isinstance(sequence_path_value, str) or not sequence_path_value:
            raise AdapterError(f"candidate {candidate_id} has no sequence_path")
        sequence_path = Path(sequence_path_value)
        if not sequence_path.is_absolute():
            raise AdapterError(f"candidate {candidate_id} sequence_path is not absolute: {sequence_path}")
        if not sequence_path.is_file():
            raise AdapterError(f"candidate {candidate_id} sequence_path is missing: {sequence_path}")
        sequence = read_fasta(sequence_path, candidate_id)
        recorded_hash = row.get("sequence_sha256")
        if not isinstance(recorded_hash, str) or not SHA256_RE.fullmatch(recorded_hash):
            raise AdapterError(f"candidate {candidate_id} has no valid sequence_sha256")
        observed_hash = sha256_sequence(sequence)
        if observed_hash != recorded_hash:
            raise AdapterError(
                f"candidate {candidate_id} sequence_sha256 does not match FASTA content: "
                f"{sequence_path} says {recorded_hash}, observed {observed_hash}"
            )
        if row.get("sequence_length") != len(sequence):
            raise AdapterError(f"candidate {candidate_id} sequence_length does not match FASTA content")
        by_id[candidate_id] = row
        sequences[candidate_id] = sequence
    return list(by_id.values()), sequences


def read_fasta(path: Path, candidate_id: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [line[1:] for line in lines if line.startswith(">")]
    if headers != [candidate_id]:
        raise AdapterError(f"candidate {candidate_id} FASTA header does not match candidate_id: {path}")
    return "".join(line.strip() for line in lines if line and not line.startswith(">"))


def contracts(config: dict[str, Any]) -> list[dict[str, Any]]:
    filters = config.get("filters")
    if not isinstance(filters, dict):
        raise AdapterError("campaign config is missing required field: filters")
    raw_contracts = filters.get("contracts")
    if not isinstance(raw_contracts, list):
        raise AdapterError("campaign config is missing required field: filters.contracts")
    selected = [
        dict(item) for item in raw_contracts
        if isinstance(item, dict) and item.get("stage_id") == STAGE_INTEGRITY
    ]
    if {item.get("filter_id") for item in selected} != SUPPORTED_FILTERS:
        raise AdapterError(
            "campaign config must define exactly composition, exact_duplicates, and "
            "liability_chemistry in filters.contracts for filter-integrity"
        )
    for contract in selected:
        filter_id = contract["filter_id"]
        threshold = contract.get("threshold")
        contract["threshold_source"] = "campaign-config"
        # Every integrity check emits exactly 0.0 or 1.0, so under the
        # minimum operator 1.0 is the only threshold that separates them. Asking a
        # person to supply it asks them to type the only possible answer. Default
        # it, and record that it was defaulted, because a value nobody chose must
        # still be visible in the observations.
        if threshold is None or threshold == "__REQUIRED__":
            if contract.get("operator") == "minimum":
                contract["threshold"] = 1.0
                contract["threshold_source"] = "product-default"
                threshold = 1.0
            else:
                raise AdapterError(
                    f"campaign config is missing a numeric threshold: "
                    f"filters.contracts[{filter_id}].threshold. This filter emits only "
                    f"0.0 or 1.0, so under the minimum operator the value defaults to "
                    f"1.0. Its operator is "
                    f"{contract.get('operator')!r}, so the default does not apply."
                )
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
        ):
            raise AdapterError(
                f"campaign config is missing a numeric threshold: filters.contracts[{filter_id}].threshold"
            )
        for field in ("metric", "operator", "tool_revision", "reference_revision"):
            value = contract.get(field)
            if not isinstance(value, str) or not value or value == "__REQUIRED__":
                raise AdapterError(f"campaign config is missing required field: filters.contracts[{filter_id}].{field}")
        reference_sha256 = contract.get("reference_sha256")
        if reference_digest_is_required(contract.get("reference_revision")):
            if not isinstance(reference_sha256, str) or not SHA256_RE.fullmatch(reference_sha256):
                raise AdapterError(
                    f"campaign config is missing required field: filters.contracts[{filter_id}].reference_sha256"
                )
        elif not is_not_applicable(reference_sha256):
            raise AdapterError(
                f"campaign config must record filters.contracts[{filter_id}].reference_sha256 as not_applicable"
            )
    return selected


def liability_rules(config: dict[str, Any]) -> dict[str, Any]:
    """Return validated, configurable sequence-liability thresholds."""
    filters = config.get("filters")
    raw = filters.get("liability_rules", {}) if isinstance(filters, dict) else {}
    if not isinstance(raw, dict):
        raise AdapterError("filters.liability_rules must be an object")
    rules = {**DEFAULT_LIABILITY_RULES, **raw}
    if not isinstance(rules["reject_odd_cysteine_count"], bool):
        raise AdapterError(
            "filters.liability_rules.reject_odd_cysteine_count must be boolean"
        )
    for name in (
        "maximum_homopolymer_run",
        "hydrophobic_window_length",
        "maximum_hydrophobic_residues",
    ):
        value = rules[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise AdapterError(f"filters.liability_rules.{name} must be a positive integer")
    if rules["maximum_hydrophobic_residues"] >= rules["hydrophobic_window_length"]:
        raise AdapterError(
            "filters.liability_rules.maximum_hydrophobic_residues must be smaller "
            "than filters.liability_rules.hydrophobic_window_length"
        )
    return rules


def longest_homopolymer(sequence: str) -> tuple[str, int, int]:
    """Return residue, length, and zero-based start for the longest run."""
    if not sequence:
        return "", 0, 0
    best_residue = sequence[0]
    best_length = 1
    best_start = 0
    run_start = 0
    for index in range(1, len(sequence) + 1):
        if index < len(sequence) and sequence[index] == sequence[run_start]:
            continue
        run_length = index - run_start
        if run_length > best_length:
            best_residue = sequence[run_start]
            best_length = run_length
            best_start = run_start
        run_start = index
    return best_residue, best_length, best_start


def densest_hydrophobic_window(sequence: str, window_length: int) -> tuple[int, int, int]:
    """Return hydrophobic count, zero-based start, and effective window length."""
    if not sequence:
        return 0, 0, 0
    effective_length = min(window_length, len(sequence))
    best_count = -1
    best_start = 0
    for start in range(len(sequence) - effective_length + 1):
        count = sum(
            residue in HYDROPHOBIC_AMINO_ACIDS
            for residue in sequence[start : start + effective_length]
        )
        if count > best_count:
            best_count = count
            best_start = start
    return best_count, best_start, effective_length


def liability_assessment(sequence: str, rules: dict[str, Any]) -> tuple[bool, str]:
    """Apply cysteine-parity, homopolymer, and hydrophobic-patch rules."""
    cysteine_count = sequence.count("C")
    residue, run_length, run_start = longest_homopolymer(sequence)
    hydrophobic_count, hydrophobic_start, effective_window = densest_hydrophobic_window(
        sequence, rules["hydrophobic_window_length"]
    )
    findings: list[str] = []
    if rules["reject_odd_cysteine_count"] and cysteine_count % 2:
        findings.append(f"odd cysteine count {cysteine_count}")
    if run_length > rules["maximum_homopolymer_run"]:
        findings.append(
            f"homopolymer {residue} at residues {run_start + 1}-{run_start + run_length} "
            f"has length {run_length}, maximum {rules['maximum_homopolymer_run']}"
        )
    if hydrophobic_count > rules["maximum_hydrophobic_residues"]:
        findings.append(
            f"hydrophobic window at residues {hydrophobic_start + 1}-"
            f"{hydrophobic_start + effective_window} contains {hydrophobic_count} of "
            f"{effective_window} hydrophobic residues, maximum "
            f"{rules['maximum_hydrophobic_residues']}"
        )
    if findings:
        return False, "liability chemistry failed: " + "; ".join(findings)
    return (
        True,
        f"liability chemistry passed: cysteines={cysteine_count}; longest homopolymer="
        f"{run_length}; densest hydrophobic window={hydrophobic_count}/{effective_window}",
    )


OPERATORS = ("minimum", "maximum")


def evaluate(value: float, contract: dict[str, Any]) -> bool:
    threshold = float(contract["threshold"])
    operator = contract["operator"]
    # An unrecognised operator used to fall through to the maximum branch, so a
    # single typo silently inverted the filter and every rejected candidate
    # passed instead. Name the two the contract allows and refuse anything else.
    if operator not in OPERATORS:
        raise AdapterError(
            f"filter {contract.get('filter_id', 'unknown')} has operator "
            f"{operator!r}; it must be one of {', '.join(OPERATORS)}"
        )
    return value >= threshold if operator == "minimum" else value <= threshold


def observation(
    row: dict[str, Any],
    contract: dict[str, Any],
    value: float,
    passed: bool,
    expected_round: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "origin_generator": row["origin_generator"],
        "sequence_sha256": row["sequence_sha256"],
        "optimization_round": expected_round,
        "filter_id": contract["filter_id"],
        "metric": contract["metric"],
        "operator": contract["operator"],
        "threshold": contract["threshold"],
        "threshold_source": contract.get("threshold_source", "campaign-config"),
        "value": value,
        "pass": passed,
        "reason": reason,
        "tool_revision": contract["tool_revision"],
        "reference_revision": contract["reference_revision"],
        "reference_sha256": contract["reference_sha256"],
    }


def run_stage(args: argparse.Namespace) -> int:
    config = read_json_object(args.config.resolve(), "campaign config")
    expected_round = stage_round(args.stage, config)
    selected_contracts = contracts(config)
    configured_liability_rules = liability_rules(config)
    manifest_path = candidate_manifest_path(args, config, expected_round)
    rows, sequences = load_candidates(manifest_path, expected_round)
    seen_sequences: set[str] = set()
    observations: list[dict[str, Any]] = []
    passed_by_candidate: dict[str, list[str]] = {row["candidate_id"]: [] for row in rows}
    for row in rows:
        candidate_id = row["candidate_id"]
        for contract in selected_contracts:
            filter_id = contract["filter_id"]
            if filter_id == "composition":
                sequence = sequences[candidate_id]
                value = 1.0 if sequence and set(sequence).issubset(CANONICAL_AMINO_ACIDS) else 0.0
                reason = "sequence contains only canonical protein residues" if value else "sequence contains an empty or disallowed residue"
            elif filter_id == "exact_duplicates":
                sequence = sequences[candidate_id]
                value = 1.0 if sequence not in seen_sequences else 0.0
                reason = "sequence is unique within this candidate cohort" if value else "sequence duplicates an earlier candidate in this cohort"
                if value:
                    seen_sequences.add(sequence)
            else:
                passed_liability, reason = liability_assessment(
                    sequences[candidate_id], configured_liability_rules
                )
                value = 1.0 if passed_liability else 0.0
            passed = evaluate(value, contract)
            observations.append(observation(row, contract, value, passed, expected_round, reason))
            if passed:
                passed_by_candidate[candidate_id].append(filter_id)
    required_ids = {contract["filter_id"] for contract in selected_contracts}
    passing_rows = [
        {**row, "filter_pass": True, "failed_checks": []}
        for row in rows
        if set(passed_by_candidate[row["candidate_id"]]) == required_ids
    ]
    phase_dir = (args.attempt_dir / args.phase).resolve()
    observation_path = phase_dir / "filter-observations.jsonl"
    passing_path = phase_dir / "passing-candidates.jsonl"
    write_jsonl(observation_path, observations)
    write_jsonl(passing_path, passing_rows)
    write_json(
        phase_dir / "filter-report.json",
        filter_report(
            STAGE_INTEGRITY,
            [row["candidate_id"] for row in rows],
            selected_contracts,
            observations,
            [row["candidate_id"] for row in passing_rows],
        ),
    )
    if not passing_rows:
        raise AdapterError("no candidate passed all configured integrity checks; downstream filtering cannot continue")
    print(
        f"integrity filter: phase={args.phase} candidates={len(rows)} "
        f"passing={len(passing_rows)} observations={len(observations)}"
    )
    return 0


def parse_jsonl(path: Path, errors: list[str]) -> int:
    if not path.is_file():
        errors.append(f"declared output is missing: {path}")
        return 0
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception as exc:
            errors.append(f"{path} line {line_number}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path} line {line_number} is not a JSON object")
            continue
        count += 1
    if count == 0:
        errors.append(f"file holds no JSONL records: {path}")
    return count


def parse_stage(args: argparse.Namespace) -> int:
    attempt_dir = args.attempt_dir.resolve()
    phase_dir = (attempt_dir / args.phase).resolve()
    result_path = phase_dir / "parser-result.json"
    errors: list[str] = []
    if attempt_dir not in phase_dir.parents:
        errors.append(f"phase path escapes the attempt directory: {phase_dir}")
    patterns = args.jsonl or [
        str(phase_dir / "filter-observations.jsonl"),
        str(phase_dir / "passing-candidates.jsonl"),
    ]
    files: list[Path] = []
    for pattern in patterns:
        raw = Path(pattern)
        resolved_pattern = (attempt_dir / raw if not raw.is_absolute() else raw).resolve()
        if attempt_dir not in resolved_pattern.parents:
            errors.append(f"declared output escapes the attempt directory: {pattern}")
            continue
        matches = sorted(Path(value) for value in glob.glob(str(resolved_pattern), recursive=True))
        if not matches:
            errors.append(f"declared output matched no files: {pattern}")
            continue
        files.extend(path for path in matches if path.is_file())
    parsed_count = sum(parse_jsonl(path, errors) for path in files)
    if not args.jsonl:
        report_path = phase_dir / "filter-report.json"
        try:
            read_json_object(report_path, "filter report")
            parsed_count += 1
            files.append(report_path)
        except AdapterError as exc:
            errors.append(str(exc))
    hashes = [sha256_file(path) for path in files if path.is_file()]
    result = {
        "ok": bool(files) and not errors,
        "parsed_count": parsed_count,
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": sorted(hashes),
    }
    write_json(result_path, result)
    for error in errors:
        print(f"integrity filter parser: {error}", file=sys.stderr)
    print(f"integrity filter parser: phase={args.phase} parsed_count={parsed_count} ok={result['ok']}")
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("toolcheck")
    for name in ("run", "parse"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument(
            "--stage",
            default=STAGE_INTEGRITY if name == "run" else None,
            required=name == "parse",
        )
        subparser.add_argument("--phase", required=True)
        subparser.add_argument("--count", type=int, default=1)
        subparser.add_argument("--attempt-dir", type=Path, required=True)
        subparser.add_argument("--receipts-dir", type=Path, required=True)
        subparser.add_argument("--artifact-root", type=Path, required=True)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--plan", type=Path, required=True)
        subparser.add_argument(
            "--candidate-manifest",
            type=Path,
            default=None,
            help="Explicit candidate manifest override. Defaults to the stage input receipt.",
        )
        subparser.add_argument(
            "--integrity-passing-manifest",
            type=Path,
            default=None,
            help=(
                "Explicit integrity-passing manifest override. Defaults to the declared "
                "filter-integrity input receipt."
            ),
        )
        if name == "parse":
            subparser.add_argument("--jsonl", action="append")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "toolcheck":
            print("integrity filter ok, standard library only")
            return 0
        return run_stage(args) if args.command == "run" else parse_stage(args)
    except AdapterError as exc:
        print(f"integrity filter: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"integrity filter: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
