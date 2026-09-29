#!/usr/bin/env python3
"""Plan and execute bounded, lineage-preserving optimization rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from claude_binder import lane
from claude_binder.adapters.adapter_io import read_jsonl as _read_jsonl

from . import proteinmpnn_designer


SCREEN_SCORE_PATH = Path("scores") / "screen-score-table.jsonl"
CONTROL_PATH = Path("controls") / "control-observations.jsonl"
LEDGER_PATH = Path("optimization") / "decision-ledger.jsonl"
DECISION_NAME = "next-round-decision.json"
SUMMARY_NAME = "round-summary.json"
OPTIMIZED_NAME = "optimized-candidates.jsonl"
RESCORE_NAME = "rescore-candidates.jsonl"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
ADAPTER_ID = "optimization-controller"
KNOWN_OPERATIONS = (
    "partial-diffusion",
    "predict-redesign",
    "inverse-folding-resample",
    "point-mutation",
)
# These are the operations this controller can execute with its declared inputs.
IMPLEMENTED_OPERATIONS = (
    "inverse-folding-resample",
    "point-mutation",
)
UNBOUND_OPERATIONS = frozenset(KNOWN_OPERATIONS) - frozenset(IMPLEMENTED_OPERATIONS)
DECISION_OUTCOMES = frozenset(
    {
        "continue",
        "converged",
        "all_metric_floor",
        "candidate_budget_exhausted",
        "prediction_budget_exhausted",
        "round_budget_spent",
    }
)
REQUIRED_DECISION_FIELDS = {
    "schema_version",
    "round",
    "summary_sha256",
    "config_sha256",
    "selected_parent_ids",
    "adapter_id",
    "operation",
    "parameter_overrides",
    "seeds",
    "candidate_count",
    "expected_fanout",
    "stop",
    "stop_reason",
}
OPTIONAL_DECISION_FIELDS = {"termination_reason", "outcome"}
CHILD_FIELDS = (
    "target_id",
    "target_sha256",
    "candidate_id",
    "parent_candidate_id",
    "root_candidate_id",
    "origin_generator",
    "last_optimizer",
    "optimizer_adapter_id",
    "optimization_operation",
    "optimization_round",
    "optimizer_seed",
    "variant_index",
    "decision_sha256",
    "parameter_set_sha256",
    "sequence_path",
    "sequence_sha256",
    "sequence_length",
    "design_pose_path",
    "design_pose_sha256",
    "status",
)
RANK_FIELDS = {"rank_zscore", "z_score", "rank_score", "screen_rank_score"}
RAW_VECTOR_FIELDS = {
    "raw_score_vectors",
    "screen_raw_score_vectors",
    "optimization_raw_score_vectors",
    "raw_metrics",
    "raw_vector",
}
SAFE_SUMMARY_FIELDS = (
    "candidate_id",
    "origin_generator",
    "parent_candidate_id",
    "root_candidate_id",
    "sequence_path",
    "sequence_sha256",
    "sequence_length",
    "design_pose_path",
    "design_pose_sha256",
    "target_id",
    "target_sha256",
    "generator_mode",
    "sequence_designer",
    "generator_seed",
    "optimization_round",
    "last_optimizer",
    "root_backbone_id",
    "tm90_cluster_id",
    "structure_method",
    "seq_method",
    "fold_class",
)


class AdapterError(RuntimeError):
    """An input, policy, or lineage condition that must stop the adapter."""


def read_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"{label} is invalid: {path}: {type(exc).__name__}: {exc}") from exc


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path, label)
    if not isinstance(value, dict):
        raise AdapterError(f"{label} must be a JSON object: {path}")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    return _read_jsonl(path, label, error_type=AdapterError)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AdapterError(f"{label} must be a non-empty string")
    return value


def require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise AdapterError(f"{label} must be at least {minimum}")
    return value


def atomic_write_bytes(path: Path, payload: bytes, *, refuse_existing: bool = False) -> None:
    if refuse_existing and path.exists():
        raise AdapterError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any, *, refuse_existing: bool = False) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload, refuse_existing=refuse_existing)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]], *, refuse_existing: bool = False) -> None:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    atomic_write_bytes(path, payload, refuse_existing=refuse_existing)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_round(stage_id: str) -> int:
    if stage_id in {"optimization-plan", "optimize"}:
        return 1
    if stage_id.startswith("optimization-plan-round-") or stage_id.startswith("optimize-round-"):
        suffix = stage_id.rsplit("-", 1)[1]
        if suffix.isdigit() and int(suffix) >= 1:
            return int(suffix)
    raise AdapterError(f"unsupported optimization stage: {stage_id}")


def is_plan_stage(stage_id: str) -> bool:
    return stage_id == "optimization-plan" or stage_id.startswith("optimization-plan-round-")


def parent_manifest_path(artifact_root: Path, round_number: int) -> Path:
    if round_number == 1:
        return artifact_root / "promotion" / "promotion-manifest.jsonl"
    return artifact_root / "optimization" / "rounds" / f"round-{round_number - 1}" / "eligible-parents.jsonl"


def score_table_path(artifact_root: Path, round_number: int) -> Path:
    if round_number == 1:
        return artifact_root / SCREEN_SCORE_PATH
    return artifact_root / "optimization" / "rounds" / f"round-{round_number - 1}" / "score-table.jsonl"


def config_optimization(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("optimization")
    if not isinstance(value, dict):
        raise AdapterError("config optimization must be an object")
    return value


def configured_early_stop_margin(config: dict[str, Any]) -> float | None:
    """Return the configured early-stop margin after validating its value.

    A null margin is an explicit fixed-rounds choice: the caller keeps the
    configured round budget and never stops early.
    """
    optimization = config_optimization(config)
    require_int(optimization.get("rounds"), "config optimization.rounds", minimum=1)
    value = optimization.get("early_stop_margin")
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise AdapterError("config optimization.early_stop_margin must be a non-negative finite number or null")
    return float(value)


def screen_seeds(config: dict[str, Any]) -> list[int]:
    cofold = config.get("cofold")
    if not isinstance(cofold, dict):
        raise AdapterError("config cofold must be an object")
    values = cofold.get("screen_seeds")
    if not isinstance(values, list) or not values:
        raise AdapterError("config cofold.screen_seeds must be a non-empty integer list")
    seeds = [require_int(value, "config cofold.screen_seeds entry") for value in values]
    if len(set(seeds)) != len(seeds):
        raise AdapterError("config cofold.screen_seeds must contain unique seeds")
    return sorted(set(seeds))


def enabled_predictor_ids(config: dict[str, Any]) -> list[str]:
    cofold = config.get("cofold")
    if not isinstance(cofold, dict):
        raise AdapterError("config cofold must be an object")
    predictors = cofold.get("predictors", [])
    if not isinstance(predictors, list):
        raise AdapterError("config cofold.predictors must be a list")
    result: list[str] = []
    for item in predictors:
        if not isinstance(item, dict) or item.get("enabled", True) is not True:
            continue
        predictor_id = require_text(item.get("id"), "enabled predictor id")
        if predictor_id in result:
            raise AdapterError(f"config repeats enabled predictor id: {predictor_id}")
        result.append(predictor_id)
    return result


def policy_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    optimization = config_optimization(config)
    policies = optimization.get("adapter_policies", [])
    if not isinstance(policies, list):
        raise AdapterError("config optimization.adapter_policies must be a list")
    result: dict[str, dict[str, Any]] = {}
    for item in policies:
        if not isinstance(item, dict):
            raise AdapterError("config optimization.adapter_policies contains a non-object")
        adapter_id = require_text(item.get("adapter_id"), "optimization adapter policy adapter_id")
        if adapter_id in result:
            raise AdapterError(f"config repeats optimization adapter policy: {adapter_id}")
        result[adapter_id] = item
    return result


def choose_operation(
    config: dict[str, Any],
    round_number: int | None = None,
) -> tuple[str, dict[str, Any]]:
    optimization = config_optimization(config)
    operators = optimization.get("operators")
    if not isinstance(operators, list) or any(not isinstance(item, str) or not item for item in operators):
        raise AdapterError("config optimization.operators must be a non-empty string list")
    policy = policy_map(config).get(ADAPTER_ID)
    if policy is None:
        raise AdapterError(f"optimization adapter policy is not allowlisted: {ADAPTER_ID}")
    allowed = policy.get("operations")
    if not isinstance(allowed, list) or any(not isinstance(item, str) or not item for item in allowed):
        raise AdapterError(f"optimization policy {ADAPTER_ID} operations must be a string list")
    configured_operations = [item for item in operators if item in allowed]
    if not configured_operations:
        raise AdapterError("optimization operator and adapter policy have no intersection")
    unbound_operations = sorted(set(configured_operations) - set(IMPLEMENTED_OPERATIONS))
    if unbound_operations:
        raise AdapterError(
            "optimization enabled operators have no controller binding: "
            + ", ".join(unbound_operations)
        )
    enabled_operations = [item for item in configured_operations if item in IMPLEMENTED_OPERATIONS]
    if not enabled_operations:
        unbound = ", ".join(item for item in configured_operations if item in UNBOUND_OPERATIONS)
        raise AdapterError(
            "optimization enabled operators have no controller binding: " + (unbound or "unknown operator")
        )
    if round_number is None:
        operation = enabled_operations[0]
    else:
        operation = enabled_operations[(round_number - 1) % len(enabled_operations)]
    return operation, policy


def validate_parameter_ranges(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ranges = policy.get("parameter_ranges", {})
    if not isinstance(ranges, dict):
        raise AdapterError("optimization policy parameter_ranges must be an object")
    for name, contract in ranges.items():
        if not isinstance(name, str) or not isinstance(contract, dict):
            raise AdapterError("optimization policy parameter range is malformed")
        kind = contract.get("type")
        if kind not in {"integer", "number", "boolean", "string"}:
            raise AdapterError(f"optimization parameter {name} has an unsupported type")
        if kind in {"integer", "number"}:
            minimum = contract.get("minimum")
            maximum = contract.get("maximum")
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, (int, float))
                or isinstance(maximum, bool)
                or not isinstance(maximum, (int, float))
                or not math.isfinite(float(minimum))
                or not math.isfinite(float(maximum))
                or float(minimum) > float(maximum)
            ):
                raise AdapterError(f"optimization parameter {name} has an invalid registered range")
    return ranges


def midpoint_parameters(operation: str, policy: dict[str, Any]) -> dict[str, Any]:
    required = {
        "point-mutation": {"mutation_count"},
        "inverse-folding-resample": {"sampling_temperature"},
    }.get(operation, set())
    ranges = validate_parameter_ranges(policy)
    overrides: dict[str, Any] = {}
    for name in sorted(required):
        contract = ranges.get(name)
        if not isinstance(contract, dict):
            raise AdapterError(f"optimization parameter is not registered for {ADAPTER_ID}: {name}")
        minimum = float(contract["minimum"])
        maximum = float(contract["maximum"])
        value = (minimum + maximum) / 2
        if contract["type"] == "integer":
            value = int(round(value))
        overrides[name] = value
    validate_overrides(operation, overrides, policy)
    return overrides


def validate_overrides(operation: str, overrides: Any, policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(overrides, dict):
        raise AdapterError("decision parameter_overrides must be an object")
    ranges = validate_parameter_ranges(policy)
    required = {
        "point-mutation": {"mutation_count"},
        "inverse-folding-resample": {"sampling_temperature"},
    }.get(operation, set())
    if not required.issubset(overrides):
        missing = sorted(required - set(overrides))
        raise AdapterError("decision parameter_overrides is missing: " + ", ".join(missing))
    unknown = sorted(set(overrides) - set(ranges))
    if unknown:
        raise AdapterError(f"decision parameter is not registered for {ADAPTER_ID}: {unknown[0]}")
    for name, value in overrides.items():
        contract = ranges[name]
        kind = contract["type"]
        valid = (
            kind == "integer" and isinstance(value, int) and not isinstance(value, bool)
        ) or (
            kind == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ) or (kind == "boolean" and isinstance(value, bool)) or (kind == "string" and isinstance(value, str))
        if not valid:
            raise AdapterError(f"decision parameter {name} must have type {kind}")
        if kind in {"integer", "number"} and not (
            float(contract["minimum"]) <= float(value) <= float(contract["maximum"])
        ):
            raise AdapterError(f"decision parameter {name} is outside its registered range")
    return dict(overrides)


def vector_is_raw(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value) and any(vector_is_raw(child) for child in value.values())
    if isinstance(value, list):
        if not value:
            return False
        return any(
            isinstance(item, dict)
            and isinstance(item.get("seed"), int)
            and any(key in item for key in ("ipsae_min", "sc_dockq", "raw_ipsae_min", "raw_sc_dockq"))
            for item in value
        )
    return False


def row_has_raw_scores(row: dict[str, Any]) -> bool:
    if any(vector_is_raw(row.get(field)) for field in RAW_VECTOR_FIELDS):
        return True
    metrics = {"ipsae_min", "sc_dockq", "raw_ipsae_min", "raw_sc_dockq"}
    return any(field in row and isinstance(row[field], (int, float)) and not isinstance(row[field], bool) for field in metrics) and isinstance(row.get("seed"), int)


def enforce_raw_score_guard(row: dict[str, Any], label: str) -> None:
    if RANK_FIELDS.intersection(row) and not row_has_raw_scores(row):
        names = ", ".join(sorted(RANK_FIELDS.intersection(row)))
        raise AdapterError(f"{label} carries {names} without its raw score vector")


def raw_score_vectors(scores: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    vectors: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seen: set[tuple[str, str, int]] = set()
    # This function keys without the target and returns {candidate: {predictor: [rows]}},
    # which has nowhere to put a second one. A two-target round scores each candidate
    # against both targets, and the second target then does one of two wrong things. It
    # collides with the first and is reported as a repeated score row, which blames the
    # table for a duplicate it does not contain. Or it lands on a candidate the first
    # target did not carry, merges two targets into one pool, and raises nothing at all.
    # Refuse here so neither happens. How a multi-target round should aggregate is a
    # scientific decision and is not made in this function; note that
    # validate_round_score_matrix already keys on the full
    # target-by-candidate-by-predictor-by-seed matrix, so the two disagree about what a
    # score table contains and this is the half that dropped the target.
    target_ids = {
        str(row.get("target_id"))
        for row in scores
        if row.get("control_type", "candidate") == "candidate"
        and row.get("target_id") is not None
    }
    if len(target_ids) > 1:
        raise AdapterError(
            "the optimization controller plans one target per round; this score table "
            f"spans {len(target_ids)}: {sorted(target_ids)}"
        )
    for index, row in enumerate(scores):
        if row.get("control_type", "candidate") != "candidate":
            continue
        candidate_id = row.get("candidate_id")
        predictor = row.get("predictor")
        seed = row.get("seed")
        if not isinstance(candidate_id, str) or not candidate_id or not isinstance(predictor, str) or not predictor:
            continue
        if isinstance(seed, bool) or not isinstance(seed, int):
            continue
        key = (candidate_id, predictor, seed)
        if key in seen:
            raise AdapterError(f"score table repeats candidate, predictor, seed: {key}")
        seen.add(key)
        metric_row: dict[str, Any] = {"seed": seed}
        for field in ("ipsae_min", "sc_dockq"):
            if field in row:
                metric_row[field] = row[field]
        if "status" in row:
            metric_row["status"] = row["status"]
        vectors.setdefault(candidate_id, {}).setdefault(predictor, []).append(metric_row)
    for predictor_rows in vectors.values():
        for rows in predictor_rows.values():
            rows.sort(key=lambda item: int(item["seed"]))
    return vectors


def validate_sequence(path_value: Any, candidate_id: str) -> str:
    path = Path(require_text(path_value, f"candidate {candidate_id}.sequence_path"))
    if not path.is_absolute():
        raise AdapterError(f"candidate {candidate_id}.sequence_path must be absolute: {path}")
    if not path.is_file():
        raise AdapterError(f"candidate {candidate_id}.sequence_path is missing: {path}")
    headers: list[str] = []
    sequence_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            headers.append(line[1:])
        elif line.strip():
            sequence_lines.append(line.strip())
    if headers != [candidate_id]:
        raise AdapterError(f"candidate {candidate_id}.sequence_path has the wrong FASTA header: {path}")
    sequence = "".join(sequence_lines).upper()
    if not sequence or any(residue not in AMINO_ACIDS for residue in sequence):
        raise AdapterError(f"candidate {candidate_id}.sequence_path has an invalid amino-acid sequence: {path}")
    return sequence


def validate_parent_row(row: dict[str, Any], *, expected_status: str, index: int, round_number: int) -> str:
    candidate_id = require_text(row.get("candidate_id"), f"parent row {index}.candidate_id")
    observed_status = row.get("promotion_status") if round_number == 1 else row.get("status")
    if observed_status != expected_status:
        raise AdapterError(f"parent {candidate_id} must have status={expected_status}")
    sequence = validate_sequence(row.get("sequence_path"), candidate_id)
    expected_sequence_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    if row.get("sequence_sha256") != expected_sequence_hash:
        raise AdapterError(f"parent {candidate_id}.sequence_sha256 does not match its FASTA")
    if row.get("sequence_length") != len(sequence):
        raise AdapterError(f"parent {candidate_id}.sequence_length does not match its FASTA")
    pose_path = Path(require_text(row.get("design_pose_path"), f"parent {candidate_id}.design_pose_path"))
    if not pose_path.is_absolute() or not pose_path.is_file():
        raise AdapterError(f"parent {candidate_id}.design_pose_path is missing: {pose_path}")
    if row.get("design_pose_sha256") != sha256_file(pose_path):
        raise AdapterError(f"parent {candidate_id}.design_pose_sha256 does not match its pose")
    require_text(row.get("origin_generator"), f"parent {candidate_id}.origin_generator")
    return sequence


def load_parent_rows(artifact_root: Path, round_number: int) -> tuple[Path, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    path = parent_manifest_path(artifact_root, round_number)
    rows = read_jsonl(path, "optimization parent manifest")
    expected_status = "promoted" if round_number == 1 else "eligible"
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        candidate_id = require_text(row.get("candidate_id"), f"parent row {index}.candidate_id")
        if candidate_id in by_id:
            raise AdapterError(f"parent manifest repeats candidate_id: {candidate_id}")
        enforce_raw_score_guard(row, f"parent {candidate_id}")
        observed_status = row.get("promotion_status") if round_number == 1 else row.get("status")
        if observed_status == expected_status:
            validate_parent_row(row, expected_status=expected_status, index=index, round_number=round_number)
            by_id[candidate_id] = row
    return path, rows, by_id


def validate_round_score_matrix(
    config: dict[str, Any],
    round_number: int,
    scores: list[dict[str, Any]],
    artifact_root: Path,
) -> None:
    candidate_path = artifact_root / "optimization" / "rounds" / f"round-{round_number - 1}" / "filters" / "passing-candidates.jsonl"
    candidate_ids: set[str] = set()
    if candidate_path.is_file():
        candidate_ids = {require_text(row.get("candidate_id"), "passing candidate_id") for row in read_jsonl(candidate_path, "optimization passing manifest")}
    if not candidate_ids:
        candidate_ids = {
            str(row["candidate_id"])
            for row in scores
            if row.get("control_type", "candidate") == "candidate" and isinstance(row.get("candidate_id"), str)
        }
    targets = config.get("targets", [])
    target_ids = [str(item["target_id"]) for item in targets if isinstance(item, dict) and isinstance(item.get("target_id"), str)]
    predictors = enabled_predictor_ids(config)
    seeds = screen_seeds(config)
    if not candidate_ids or not target_ids or not predictors:
        return
    seen: set[tuple[str, str, str, int]] = set()
    for index, row in enumerate(scores):
        if row.get("control_type", "candidate") != "candidate":
            continue
        key = (str(row.get("target_id")), str(row.get("candidate_id")), str(row.get("predictor")), row.get("seed", -1))
        if key in seen:
            raise AdapterError(f"optimization score table repeats key: {key}")
        seen.add(key)
        if not isinstance(row.get("seed"), int) or isinstance(row.get("seed"), bool):
            raise AdapterError(f"optimization score row {index}.seed must be an integer")
    expected = {(target_id, candidate_id, predictor, seed) for target_id in target_ids for candidate_id in candidate_ids for predictor in predictors for seed in seeds}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise AdapterError(f"optimization score keys do not match the exact round matrix: missing={missing[:8]} extra={extra[:8]}")


def safe_summary_candidate(row: dict[str, Any], status: str, vectors: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    candidate_id = require_text(row.get("candidate_id"), "summary candidate_id")
    result = {field: row[field] for field in SAFE_SUMMARY_FIELDS if field in row}
    result["candidate_id"] = candidate_id
    result["origin_generator"] = require_text(row.get("origin_generator"), f"summary {candidate_id}.origin_generator")
    result["status"] = status
    source_vectors = next((row[field] for field in RAW_VECTOR_FIELDS if vector_is_raw(row.get(field))), None)
    if source_vectors is None:
        source_vectors = vectors.get(candidate_id)
    if source_vectors is not None:
        result["raw_score_vectors"] = source_vectors
        result["raw_seed_counts"] = {
            predictor: len(values) for predictor, values in source_vectors.items()
            if isinstance(values, list)
        } if isinstance(source_vectors, dict) else {}
    return result


def ledger_rows(artifact_root: Path) -> list[dict[str, Any]]:
    path = artifact_root / LEDGER_PATH
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(read_jsonl(path, "optimization decision ledger")):
        decision = raw_row.get("decision", raw_row)
        if not isinstance(decision, dict):
            raise AdapterError(f"decision ledger row {index}.decision must be an object")
        row = dict(decision)
        for field in ("decision_sha256", "run_fingerprint", "stage_id", "attempt_id"):
            if field in raw_row:
                row[field] = raw_row[field]
        require_int(row.get("round"), f"decision ledger row {index}.round", minimum=1)
        require_int(row.get("candidate_count"), f"decision ledger row {index}.candidate_count", minimum=0)
        rows.append(row)
    return rows


def used_prediction_budget(artifact_root: Path) -> int:
    return sum(int(row["candidate_count"]) for row in ledger_rows(artifact_root))


def verify_ledger_decision(artifact_root: Path, round_number: int, decision_sha: str) -> None:
    matches = [row for row in ledger_rows(artifact_root) if row.get("round") == round_number]
    if not matches:
        return
    if len(matches) != 1:
        raise AdapterError(f"decision ledger contains multiple entries for round {round_number}")
    recorded = matches[0].get("decision_sha256")
    if not isinstance(recorded, str) or recorded != decision_sha:
        raise AdapterError("optimization decision file hash does not match the decision ledger")


def budget_fields(config: dict[str, Any], artifact_root: Path) -> dict[str, int]:
    selection = config.get("selection")
    runtime = config.get("runtime")
    if not isinstance(selection, dict) or not isinstance(runtime, dict):
        raise AdapterError("config selection and runtime must be objects")
    remaining_candidates = require_int(selection.get("final_count"), "config selection.final_count", minimum=0)
    maximum = require_int(runtime.get("maximum_optimization_predictions"), "config runtime.maximum_optimization_predictions", minimum=0)
    used = used_prediction_budget(artifact_root)
    remaining_predictions = maximum - used
    if remaining_predictions < 0:
        raise AdapterError("optimization decision ledger already exceeds runtime.maximum_optimization_predictions")
    return {"remaining_candidates": remaining_candidates, "remaining_predictions": remaining_predictions}


def budget_stop_reason(round_number: int, budget: dict[str, int], candidate_count: int) -> str | None:
    """Return the artifact reason when a planned round has no remaining budget."""
    if candidate_count > budget["remaining_candidates"]:
        return (
            f"round {round_number} stopped because the candidate budget has "
            f"{budget['remaining_candidates']} remaining slots for {candidate_count} planned candidates"
        )
    if candidate_count > budget["remaining_predictions"]:
        return (
            f"round {round_number} stopped because the optimization prediction budget has "
            f"{budget['remaining_predictions']} remaining slots for {candidate_count} planned candidates"
        )
    return None


def best_configured_metric(config: dict[str, Any], scores: list[dict[str, Any]]) -> tuple[str, float | None]:
    """Return the best eligible value of the configured primary metric."""
    primary_metric, _ = lane._ranking_metrics(config)
    direction = lane.metric_direction(config, primary_metric)
    ranked = lane.rank_candidate_cohort(config, scores, screen_seeds(config))
    values = [
        lane._aggregate_metric_value(row, primary_metric)
        for row in ranked
        if row.get("coverage_complete") is True and row.get("filter_pass") is True
    ]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return primary_metric, None
    return primary_metric, (max(finite) if direction == "maximize" else min(finite))


def all_candidates_at_primary_metric_floor(
    config: dict[str, Any],
    scores: list[dict[str, Any]],
    *,
    config_path: Path | None = None,
) -> bool:
    """Return whether every scored candidate sits at the configured metric's floor.

    The floor belongs to the metric, not to this function, so it is read from the
    campaign roster or the package registry. A campaign that optimizes a metric
    with no recorded floor gets no diagnosis, and that absence is the roster's to
    close rather than something to guess here.
    """
    primary_metric, _ = lane._ranking_metrics(config)
    # A recorded floor is a low-signal diagnosis for a metric that the campaign
    # maximizes. For a minimizing objective, the same floor is a desired value;
    # treating it as a reason to continue would force an extra round.
    if lane.metric_direction(config, primary_metric) == "minimize":
        return False
    floor = lane.primary_metric_floor(config, config_path)
    if floor is None:
        return False
    values = [
        lane._row_metric_value(row, primary_metric)
        for row in scores
        if row.get("control_type") == "candidate" and row.get("status") == "scored"
    ]
    return bool(values) and all(value == floor for value in values)


def best_score_history(
    config: dict[str, Any],
    artifact_root: Path,
    completed_round: int,
) -> tuple[str, list[dict[str, float | int | None]]]:
    """Return each completed round's best score and its cumulative best score."""
    if completed_round < 0:
        raise AdapterError("completed optimization round cannot be negative")
    paths = [(0, artifact_root / SCREEN_SCORE_PATH)]
    paths.extend(
        (
            round_number,
            artifact_root / "optimization" / "rounds" / f"round-{round_number}" / "score-table.jsonl",
        )
        for round_number in range(1, completed_round + 1)
    )
    metric, _ = lane._ranking_metrics(config)
    direction = lane.metric_direction(config, metric)
    best_to_date: float | None = None
    history: list[dict[str, float | int | None]] = []
    for source_round, path in paths:
        scores = read_jsonl(path, f"optimization score table for round {source_round}")
        observed_metric, round_best = best_configured_metric(config, scores)
        if observed_metric != metric:
            raise AdapterError("optimization score tables disagree on the configured primary metric")
        if round_best is not None and (
            best_to_date is None
            or (round_best > best_to_date if direction == "maximize" else round_best < best_to_date)
        ):
            best_to_date = round_best
        history.append(
            {
                "round": source_round,
                "round_best": round_best,
                "best_score_to_date": best_to_date,
            }
        )
    return metric, history


def early_stop_decision(
    config: dict[str, Any],
    artifact_root: Path,
    round_number: int,
    current_scores: list[dict[str, Any]],
    *,
    config_path: Path | None = None,
) -> tuple[bool, str | None]:
    """Decide whether the measured improvement is below the configured margin."""
    margin = configured_early_stop_margin(config)
    if margin is None or round_number <= 1:
        return False, None
    if all_candidates_at_primary_metric_floor(config, current_scores, config_path=config_path):
        return False, None
    metric, current_best = best_configured_metric(config, current_scores)
    baseline_metric, history = best_score_history(config, artifact_root, round_number - 2)
    if baseline_metric != metric:
        raise AdapterError("optimization baseline does not use the configured primary metric")
    baseline_best = history[-1]["best_score_to_date"] if history else None
    if current_best is None or baseline_best is None:
        return False, None
    direction = lane.metric_direction(config, metric)
    improvement = (
        current_best - float(baseline_best)
        if direction == "maximize"
        else float(baseline_best) - current_best
    )
    if improvement == 0.0:
        # A tie records an unresolved score comparison. The metric floor and constant
        # negative controls can produce one, so a single tie cannot establish convergence.
        if len(history) < 2:
            return False, None
        previous_best = history[-2]["best_score_to_date"]
        if previous_best is None:
            return False, None
        previous_improvement = float(baseline_best) - float(previous_best)
        if previous_improvement != 0.0:
            return False, None
        return (
            True,
            f"round {round_number - 1} tied {metric} for two consecutive rounds. "
            f"The early_stop_margin is {margin:.12g}",
        )
    if improvement < margin:
        return (
            True,
            f"round {round_number - 1} improved {metric} by {improvement:.12g}. "
            f"The early_stop_margin is {margin:.12g}",
        )
    return False, None


def validate_decision_shape(
    config: dict[str, Any],
    summary: dict[str, Any],
    decision: dict[str, Any],
    summary_path: Path,
    config_path: Path,
    round_number: int,
) -> None:
    missing = sorted(REQUIRED_DECISION_FIELDS - set(decision))
    unknown = sorted(set(decision) - REQUIRED_DECISION_FIELDS - OPTIONAL_DECISION_FIELDS)
    if missing:
        raise AdapterError("next-round decision missing " + ", ".join(missing))
    if unknown:
        raise AdapterError("next-round decision has unknown fields: " + ", ".join(unknown))
    if decision["schema_version"] != 1:
        raise AdapterError("next-round decision schema_version must be 1")
    if decision["round"] != round_number or summary.get("round") != round_number:
        raise AdapterError("next-round decision round does not match its round summary")
    if decision["summary_sha256"] != sha256_file(summary_path):
        raise AdapterError("next-round decision summary_sha256 does not match round-summary.json")
    if decision["config_sha256"] != sha256_file(config_path):
        raise AdapterError("next-round decision config_sha256 does not match the resolved configuration")
    configured_early_stop_margin(config)
    parents = decision["selected_parent_ids"]
    if not isinstance(parents, list) or any(not isinstance(item, str) or not item for item in parents):
        raise AdapterError("next-round decision selected_parent_ids must be a string list")
    if len(parents) != len(set(parents)):
        raise AdapterError("next-round decision selected_parent_ids must be unique")
    optimization = config_optimization(config)
    requested = require_int(optimization.get("parent_count_per_round"), "config optimization.parent_count_per_round", minimum=1)
    variants = require_int(optimization.get("variants_per_parent"), "config optimization.variants_per_parent", minimum=1)
    if len(parents) > requested:
        raise AdapterError("next-round decision exceeds parent_count_per_round")
    operation, policy = choose_operation(config, round_number)
    if decision["adapter_id"] != ADAPTER_ID or decision["operation"] != operation:
        raise AdapterError("next-round decision operation or adapter does not match the allowlisted policy")
    termination_reason = decision.get("termination_reason")
    if termination_reason is not None and (
        not isinstance(termination_reason, str) or not termination_reason.strip()
    ):
        raise AdapterError("next-round decision termination_reason must be a non-empty string or null")
    outcome = decision.get("outcome")
    if outcome is not None and outcome not in DECISION_OUTCOMES:
        raise AdapterError("next-round decision outcome is not registered")
    if not isinstance(decision["stop"], bool):
        raise AdapterError("next-round decision stop must be boolean")
    if decision["stop"]:
        if decision["parameter_overrides"] != {}:
            raise AdapterError("a stopped decision must have empty parameter_overrides")
    else:
        validate_overrides(operation, decision["parameter_overrides"], policy)
    expected_seeds = screen_seeds(config)
    if decision["seeds"] != expected_seeds:
        raise AdapterError("next-round decision seeds must equal sorted cofold.screen_seeds")
    candidate_count = require_int(decision["candidate_count"], "next-round decision candidate_count", minimum=0)
    expected_fanout = require_int(decision["expected_fanout"], "next-round decision expected_fanout", minimum=0)
    if candidate_count != expected_fanout or candidate_count != len(parents) * variants:
        raise AdapterError("next-round decision expected_fanout does not match selected parents and variants_per_parent")
    if decision["stop"]:
        if parents:
            raise AdapterError("a stopped decision must select no parents")
        if not isinstance(decision["stop_reason"], str) or not decision["stop_reason"].strip():
            raise AdapterError("a stopped decision must record a non-empty stop_reason")
        if termination_reason is not None and termination_reason != decision["stop_reason"]:
            raise AdapterError("a stopped decision termination_reason must match stop_reason")
    elif decision["stop_reason"] is not None:
        raise AdapterError("a continuing decision must set stop_reason=null")
    budget = summary.get("budget")
    if not isinstance(budget, dict):
        raise AdapterError("round summary budget must be an object")
    if candidate_count > require_int(budget.get("remaining_candidates"), "round summary remaining_candidates", minimum=0):
        raise AdapterError("next-round decision exceeds the round summary candidate budget")
    if candidate_count > require_int(budget.get("remaining_predictions"), "round summary remaining_predictions", minimum=0):
        raise AdapterError("next-round decision exceeds the round summary prediction budget")


def plan_stage(args: argparse.Namespace) -> int:
    config = read_json_object(args.config.resolve(), "campaign config")
    round_number = parse_round(args.stage)
    optimization = config_optimization(config)
    configured_early_stop_margin(config)
    score_path = score_table_path(args.artifact_root.resolve(), round_number)
    scores = read_jsonl(score_path, "optimization score table")
    controls_are_ungated = (
        lane.is_ungated_candidate_claim(config)
        and lane.control_panel_is_disabled(config)
    )
    if not controls_are_ungated:
        control_path = args.artifact_root.resolve() / CONTROL_PATH
        if not control_path.is_file():
            raise AdapterError(f"control calibration table is missing for gated controls: {control_path}")
        control_check = lane.validate_control_calibration(config, args.artifact_root.resolve())
        if not control_check.get("ok"):
            raise AdapterError("control calibration failed for gated controls: " + "; ".join(control_check.get("errors", [])[:8]))
    if round_number == 1:
        screen_check = lane.validate_screen_scored_pool(config, score_path, args.artifact_root.resolve())
        if not screen_check.get("ok"):
            raise AdapterError("screen score validation failed: " + "; ".join(screen_check.get("errors", [])[:8]))
    else:
        validate_round_score_matrix(config, round_number, scores, args.artifact_root.resolve())
    manifest_path, manifest_rows, parent_rows = load_parent_rows(args.artifact_root.resolve(), round_number)
    vectors = raw_score_vectors(scores)
    for index, row in enumerate(scores):
        enforce_raw_score_guard(row, f"score row {index}")
    ranked = lane.rank_candidate_cohort(config, scores, screen_seeds(config))
    if not isinstance(ranked, list):
        raise AdapterError("rank_candidate_cohort did not return a list")
    # Parent selection is a decision, not merely a diagnostic ordering.  Use
    # the same declared method as promotion and final rank: a candidate-level
    # profile can deliberately replace the published three-mode z-score with a
    # raw ensemble mean.  Leaving the cohort's published composite in place
    # here would silently optimize a different parent from the one the
    # delivered ranking names.
    ranked = lane.apply_declared_ranking_mode(config, ranked, complete_only=True)
    ranked.sort(key=lambda row: lane._rank_sort_key(row, config))
    ranked_candidates: list[dict[str, Any]] = []
    for index, row in enumerate(ranked):
        if not isinstance(row, dict):
            raise AdapterError(f"ranked cohort row {index} is not an object")
        candidate_id = require_text(row.get("candidate_id"), f"ranked row {index}.candidate_id")
        if RANK_FIELDS.intersection(row) and not (
            row_has_raw_scores(row) or vector_is_raw(vectors.get(candidate_id))
        ):
            enforce_raw_score_guard(row, f"ranked row {index}")
        if row.get("eligible") is True and candidate_id in parent_rows:
            ranked_candidates.append(row)
    parent_count = require_int(optimization.get("parent_count_per_round"), "config optimization.parent_count_per_round", minimum=1)
    if len(ranked_candidates) == 0:
        raise AdapterError(
            f"round {round_number} stopped: no eligible parents; requested {parent_count}, found 0"
        )
    if len(ranked_candidates) < parent_count:
        raise AdapterError(
            f"round {round_number} cannot fill parent slate: requested {parent_count}, found {len(ranked_candidates)} eligible parents"
        )
    selected_parent_ids = [str(row["candidate_id"]) for row in ranked_candidates[:parent_count]]
    operation, policy = choose_operation(config, round_number)
    overrides = midpoint_parameters(operation, policy)
    variants = require_int(optimization.get("variants_per_parent"), "config optimization.variants_per_parent", minimum=1)
    candidate_count = len(selected_parent_ids) * variants
    all_at_floor = all_candidates_at_primary_metric_floor(
        config, scores, config_path=args.config.resolve()
    )
    should_stop, stop_reason = early_stop_decision(
        config,
        args.artifact_root.resolve(),
        round_number,
        scores,
        config_path=args.config.resolve(),
    )
    outcome = "all_metric_floor" if all_at_floor else "converged" if should_stop else "continue"
    budget = budget_fields(config, args.artifact_root.resolve())
    if not should_stop:
        stop_reason = budget_stop_reason(round_number, budget, candidate_count)
        should_stop = stop_reason is not None
        if should_stop and not all_at_floor:
            outcome = (
                "candidate_budget_exhausted"
                if candidate_count > budget["remaining_candidates"]
                else "prediction_budget_exhausted"
            )
    if should_stop:
        selected_parent_ids = []
        overrides = {}
        candidate_count = 0
    termination_reason = stop_reason
    if not should_stop and round_number == require_int(
        optimization.get("rounds"), "config optimization.rounds", minimum=1
    ):
        termination_reason = f"optimization round budget of {round_number} rounds is spent"
        if not all_at_floor:
            outcome = "round_budget_spent"
    expected_status = "promoted" if round_number == 1 else "eligible"
    summary_candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(manifest_rows):
        candidate_id = require_text(row.get("candidate_id"), f"parent manifest row {index}.candidate_id")
        status = row.get("promotion_status") if round_number == 1 else row.get("status")
        if status == expected_status:
            summary_candidates.append(safe_summary_candidate(row, expected_status, vectors))
        else:
            failures.append({"candidate_id": candidate_id, "reason": f"parent row status is not {expected_status}"})
    summary = {
        "schema_version": 1,
        "round": round_number,
        "source_manifest_sha256": sha256_file(manifest_path),
        "candidates": summary_candidates,
        "budget": budget,
        "failures": failures,
    }
    output_dir = (args.attempt_dir / args.phase).resolve()
    summary_path = output_dir / SUMMARY_NAME
    decision_path = output_dir / DECISION_NAME
    if summary_path.exists():
        raise AdapterError(f"output already exists: {summary_path}")
    if decision_path.exists():
        raise AdapterError(f"output already exists: {decision_path}")
    existing_ledger = ledger_rows(args.artifact_root.resolve())
    if any(row.get("round") == round_number for row in existing_ledger):
        raise AdapterError(f"decision ledger already contains round {round_number}")
    atomic_write_json(summary_path, summary, refuse_existing=True)
    decision = {
        "schema_version": 1,
        "round": round_number,
        "summary_sha256": sha256_file(summary_path),
        "config_sha256": sha256_file(args.config.resolve()),
        "selected_parent_ids": selected_parent_ids,
        "adapter_id": ADAPTER_ID,
        "operation": operation,
        "parameter_overrides": overrides,
        "seeds": screen_seeds(config),
        "candidate_count": candidate_count,
        "expected_fanout": candidate_count,
        "stop": should_stop,
        "stop_reason": stop_reason,
        "termination_reason": termination_reason,
        "outcome": outcome,
    }
    atomic_write_json(decision_path, decision, refuse_existing=True)
    append_jsonl(
        args.artifact_root.resolve() / LEDGER_PATH,
        {
            "schema_version": 1,
            "round": round_number,
            "decision_sha256": sha256_file(decision_path),
            "adapter_id": ADAPTER_ID,
            "operation": operation,
            "seeds": screen_seeds(config),
            "candidate_count": candidate_count,
            "stop": should_stop,
            "stop_reason": stop_reason,
            "termination_reason": termination_reason,
            "outcome": outcome,
        },
    )
    print(
        f"optimization controller: round={round_number} parents={len(selected_parent_ids)} "
        f"candidates={candidate_count} summary={summary_path} decision={decision_path}"
    )
    return 0


def receipt_path(receipts_dir: Path, stage_id: str) -> Path:
    return receipts_dir / f"{stage_id}.json"


def receipt_artifact(receipts_dir: Path, stage_ids: list[str], artifact_type: str) -> tuple[Path, str | None]:
    for stage_id in stage_ids:
        path = receipt_path(receipts_dir, stage_id)
        if not path.is_file():
            continue
        receipt = read_json_object(path, f"receipt {stage_id}")
        if receipt.get("ok") is not True:
            raise AdapterError(f"receipt {stage_id} did not complete: {path}")
        artifacts = receipt.get("output_manifest", {}).get("artifacts")
        if not isinstance(artifacts, list):
            raise AdapterError(f"receipt {stage_id} has no artifact list: {path}")
        matches = [
            item for item in artifacts
            if isinstance(item, dict) and artifact_type in {item.get("artifact_type"), item.get("artifact_id")}
        ]
        if not matches:
            raise AdapterError(f"receipt {stage_id} declares no {artifact_type} artifact")
        files = matches[0].get("files")
        if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
            raise AdapterError(f"receipt {stage_id} artifact {artifact_type} must declare one file")
        file_record = files[0]
        path_value = require_text(file_record.get("path"), f"receipt {stage_id} {artifact_type} path")
        return Path(path_value), file_record.get("sha256") if isinstance(file_record.get("sha256"), str) else None
    raise AdapterError(f"optimization plan receipt is missing: {receipts_dir}")


def configured_output_name(config: dict[str, Any], stage_id: str, round_number: int) -> str:
    stages = config.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict) or stage.get("stage_id") != stage_id:
                continue
            for output in stage.get("outputs", []):
                if not isinstance(output, dict):
                    continue
                artifact_id = output.get("artifact_id")
                if artifact_id in {"optimized-candidates", "rescore-candidates"}:
                    return OPTIMIZED_NAME if artifact_id == "optimized-candidates" else RESCORE_NAME
    optimization = config_optimization(config)
    return RESCORE_NAME if round_number == int(optimization.get("rounds", round_number)) else OPTIMIZED_NAME


def primary_target(config: dict[str, Any]) -> dict[str, Any]:
    targets = config.get("targets")
    if not isinstance(targets, list) or not targets:
        raise AdapterError("config targets must contain a primary target")
    target = next((item for item in targets if isinstance(item, dict) and item.get("role") == "primary"), targets[0])
    if not isinstance(target, dict):
        raise AdapterError("config primary target must be an object")
    require_text(target.get("target_id"), "config primary target_id")
    require_text(target.get("structure_sha256"), "config primary target structure_sha256")
    return target


def optimizer_seed(child_id: str) -> int:
    return int.from_bytes(hashlib.sha256(child_id.encode("utf-8")).digest()[:8], "little") % (2**31)


def mutate_sequence(sequence: str, mutation_count: int, seed: int) -> str:
    if mutation_count < 1 or mutation_count > len(sequence):
        raise AdapterError(f"mutation_count={mutation_count} exceeds sequence length {len(sequence)}")
    rng = random.Random(seed)
    positions = rng.sample(range(len(sequence)), mutation_count)
    residues = list(sequence)
    for position in positions:
        current = residues[position]
        choices = [residue for residue in AMINO_ACIDS if residue != current]
        residues[position] = rng.choice(choices)
    return "".join(residues)


def safe_child_id(value: str) -> None:
    if value in {"", ".", ".."} or Path(value).name != value or any(character in value for character in "\\\x00"):
        raise AdapterError(f"candidate id cannot name an output file: {value!r}")


def child_base_row(
    parent: dict[str, Any],
    target: dict[str, Any],
    child_id: str,
    round_number: int,
    variant_index: int,
    decision: dict[str, Any],
    decision_sha: str,
    parameter_sha: str,
    seed: int,
) -> dict[str, Any]:
    parent_id = require_text(parent.get("candidate_id"), "parent candidate_id")
    root_id = parent_id if round_number == 1 else require_text(parent.get("root_candidate_id"), f"parent {parent_id}.root_candidate_id")
    return {
        "target_id": parent.get("target_id", target["target_id"]),
        "target_sha256": parent.get("target_sha256", target["structure_sha256"]),
        "candidate_id": child_id,
        "parent_candidate_id": parent_id,
        "root_candidate_id": root_id,
        "origin_generator": require_text(parent.get("origin_generator"), f"parent {parent_id}.origin_generator"),
        "root_backbone_id": require_text(parent.get("root_backbone_id"), f"parent {parent_id}.root_backbone_id"),
        "tm90_cluster_id": require_text(parent.get("tm90_cluster_id"), f"parent {parent_id}.tm90_cluster_id"),
        "structure_method": require_text(parent.get("structure_method"), f"parent {parent_id}.structure_method"),
        "seq_method": require_text(parent.get("seq_method"), f"parent {parent_id}.seq_method"),
        "fold_class": require_text(parent.get("fold_class"), f"parent {parent_id}.fold_class"),
        "last_optimizer": decision["adapter_id"],
        "optimizer_adapter_id": decision["adapter_id"],
        "optimization_operation": decision["operation"],
        "optimization_round": round_number,
        "optimizer_seed": seed,
        "variant_index": variant_index,
        "decision_sha256": decision_sha,
        "parameter_set_sha256": parameter_sha,
        "sequence_path": None,
        "sequence_sha256": None,
        "sequence_length": None,
        "design_pose_path": None,
        "design_pose_sha256": None,
        "status": "failed",
    }


def inverse_folding_rows(
    config: dict[str, Any],
    args: argparse.Namespace,
    parents: list[dict[str, Any]],
    decision: dict[str, Any],
    output_dir: Path,
    decision_sha: str,
    parameter_sha: str,
    round_number: int,
    target: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resample each parent backbone with the qualified ProteinMPNN adapter."""
    optimization = config_optimization(config)
    variants = require_int(optimization.get("variants_per_parent"), "config optimization.variants_per_parent", minimum=1)
    binder = config.get("binder", {})
    if not isinstance(binder, dict):
        raise AdapterError("config binder must be an object")
    designer_args = SimpleNamespace(
        runner_protocol="auto",
        fal_url=None,
        proteinmpnn_root=None,
        fal_client=None,
        fal_executable=proteinmpnn_designer.DEFAULT_FAL_EXECUTABLE,
        client_python=proteinmpnn_designer.DEFAULT_CLIENT_PYTHON,
        fal_credential_route=None,
        fal_timeout_seconds=proteinmpnn_designer.DEFAULT_FAL_TIMEOUT_SECONDS,
        fal_request_id=None,
        model_name=proteinmpnn_designer.DEFAULT_MODEL_NAME,
        soluble_model=False,
        weights_dir=None,
        tool_python=None,
        model_revision=None,
        config=args.config.resolve(),
        designer_id="proteinmpnn",
        seed=0,
        work_subdir=proteinmpnn_designer.DEFAULT_WORK_SUBDIR,
        sequences_per_backbone=variants,
        sampling_temp=float(decision["parameter_overrides"]["sampling_temperature"]),
        design_chain=binder.get("binder_chain_id"),
        sequences_glob=None,
        sequence_subdir="sequences",
        pose_subdir="poses",
        minimum_length=require_int(binder.get("minimum_length"), "config binder.minimum_length", minimum=1),
        maximum_length=require_int(binder.get("maximum_length"), "config binder.maximum_length", minimum=1),
    )
    runner_protocol, execution = proteinmpnn_designer.resolve_execution(designer_args)
    model_revision = proteinmpnn_designer.resolve_model_revision(designer_args)
    fal_url = ""
    if runner_protocol == "fal":
        runner = proteinmpnn_designer.resolve_fal_client(designer_args.fal_client)
        fal_url = str(execution)
        weights_dir = None
        verification = None
        tool_python = proteinmpnn_designer.resolve_tool_python(designer_args.tool_python)
    else:
        root = Path(execution)
        runner = proteinmpnn_designer.resolve_runner(root)
        weights_dir, checkpoint = proteinmpnn_designer.resolve_weights(
            root,
            designer_args.weights_dir,
            designer_args.model_name,
            soluble=designer_args.soluble_model,
        )
        verification = proteinmpnn_designer.verify_checkpoint_digest(checkpoint, model_revision)
        tool_python = proteinmpnn_designer.resolve_tool_python(designer_args.tool_python)
    rows: list[dict[str, Any]] = []
    for parent in parents:
        parent_id = require_text(parent.get("candidate_id"), "inverse-folding parent candidate_id")
        designer_args.seed = optimizer_seed(f"inverse-folding-resample|{round_number}|{parent_id}")
        parent_seed = parent.get("optimizer_seed", parent.get("generator_seed"))
        if isinstance(parent_seed, int) and not isinstance(parent_seed, bool) and designer_args.seed == parent_seed:
            designer_args.seed = (designer_args.seed + 1) % (2**31)
        try:
            generated_rows = proteinmpnn_designer.design_backbone(
                designer_args,
                row=parent,
                index=0,
                runner=runner,
                weights_dir=weights_dir,
                tool_python=tool_python,
                phase_dir=output_dir,
                verification=verification,
                runner_protocol=runner_protocol,
                fal_url=fal_url,
                model_revision=model_revision,
            )
            for generated in generated_rows:
                child_id = require_text(generated.get("candidate_id"), "ProteinMPNN candidate_id")
                seed = int(generated.get("tool_seed", generated.get("requested_seed", designer_args.seed)))
                row = child_base_row(
                    parent,
                    target,
                    child_id,
                    round_number,
                    int(generated.get("variant_index", 0)),
                    decision,
                    decision_sha,
                    parameter_sha,
                    seed,
                )
                row.update(
                    {
                        key: generated[key]
                        for key in (
                            "sequence_path",
                            "sequence_sha256",
                            "sequence_length",
                            "design_pose_path",
                            "design_pose_sha256",
                        )
                        if key in generated
                    }
                )
                row["status"] = "generated"
                rows.append(row)
        except Exception as exc:  # noqa: BLE001
            for variant_index in range(variants):
                child_id = f"{parent_id}-r{round_number}v{variant_index}"
                row = child_base_row(
                    parent,
                    target,
                    child_id,
                    round_number,
                    variant_index,
                    decision,
                    decision_sha,
                    parameter_sha,
                    optimizer_seed(child_id),
                )
                row["failure_reason"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    return rows


def carry_pool_forward(
    config: dict[str, Any],
    artifact_root: Path,
    round_number: int,
    decision: dict[str, Any],
    decision_sha: str,
    parameter_sha: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Build a stopped round's rows and give the round its own copies.

    A stopped round generates no children and carries the scored pool forward.
    The source rows name sequence and pose files owned by the stages that
    produced them, and every stage must own the files it declares, so copy them
    into this round's attempt directory and repoint the rows. Before this, the
    carried rows pointed outside the attempt and the stage was refused. Nothing
    ever reached that refusal because the stop branch had never executed.
    """
    source_rows = lane.optimization_scored_candidate_pool(
        config, Path(artifact_root).resolve(), final_round=round_number - 1
    )
    sequence_dir = Path(output_dir) / "sequences"
    pose_dir = Path(output_dir) / "poses"
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        row = dict(source)
        row["source_optimization_round"] = int(source.get("optimization_round", 0))
        row["optimization_round"] = round_number
        row["last_optimizer"] = decision["adapter_id"]
        row["optimizer_adapter_id"] = decision["adapter_id"]
        row["optimization_operation"] = decision["operation"]
        row["optimizer_seed"] = None
        row["variant_index"] = index
        row["decision_sha256"] = decision_sha
        row["parameter_set_sha256"] = parameter_sha
        row["status"] = "generated"
        for field, destination_dir in (
            ("sequence_path", sequence_dir),
            ("design_pose_path", pose_dir),
        ):
            origin = row.get(field)
            if not isinstance(origin, str) or not origin:
                continue
            origin_path = Path(origin)
            if not origin_path.is_file():
                raise AdapterError(
                    f"carried row {index} names a {field} that is not a file: {origin}"
                )
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / origin_path.name
            if not destination.exists():
                shutil.copyfile(origin_path, destination)
            row[field] = str(destination)
        rows.append(row)
    return rows


def execute_stage(args: argparse.Namespace) -> int:
    config = read_json_object(args.config.resolve(), "campaign config")
    round_number = parse_round(args.stage)
    plan_ids = [f"optimization-plan-round-{round_number}"]
    if args.stage == "optimize":
        plan_ids.append("optimization-plan")
    decision_path, declared_decision_sha = receipt_artifact(args.receipts_dir.resolve(), plan_ids, "next-round-decision")
    summary_path, _ = receipt_artifact(args.receipts_dir.resolve(), plan_ids, "round-summary")
    if declared_decision_sha is not None and declared_decision_sha != sha256_file(decision_path):
        raise AdapterError("optimization decision file hash does not match its receipt")
    decision = read_json_object(decision_path, "next-round decision")
    summary = read_json_object(summary_path, "round summary")
    validate_decision_shape(config, summary, decision, summary_path, args.config.resolve(), round_number)
    verify_ledger_decision(args.artifact_root.resolve(), round_number, sha256_file(decision_path))
    manifest_path, _, parent_rows = load_parent_rows(args.artifact_root.resolve(), round_number)
    selected_ids = decision["selected_parent_ids"]
    if any(parent_id not in parent_rows for parent_id in selected_ids):
        missing = next(parent_id for parent_id in selected_ids if parent_id not in parent_rows)
        raise AdapterError(f"selected parent is absent from {manifest_path}: {missing}")
    current_used = used_prediction_budget(args.artifact_root.resolve())
    maximum = require_int(config.get("runtime", {}).get("maximum_optimization_predictions"), "config runtime.maximum_optimization_predictions", minimum=0)
    if current_used > maximum:
        raise AdapterError("cumulative optimization prediction budget is exceeded")
    decision_sha = sha256_file(decision_path)
    parameter_sha = sha256_json(decision["parameter_overrides"])
    output_dir = (args.attempt_dir / args.phase).resolve()
    output_name = configured_output_name(config, args.stage, round_number)
    output_path = output_dir / output_name
    if output_path.exists():
        raise AdapterError(f"output already exists: {output_path}")
    if decision["stop"]:
        try:
            rows = carry_pool_forward(
                config,
                args.artifact_root.resolve(),
                round_number,
                decision,
                decision_sha,
                parameter_sha,
                output_dir,
            )
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(f"could not carry the scored pool into a stopped round: {exc}") from exc
        atomic_write_jsonl(output_path, rows, refuse_existing=True)
        print(f"optimization controller: round={round_number} stopped children=0 output={output_path}")
        return 0
    operation = decision["operation"]
    if operation not in IMPLEMENTED_OPERATIONS:
        raise AdapterError(f"operator not bound: {operation}")
    target = primary_target(config)
    if operation == "inverse-folding-resample":
        rows = inverse_folding_rows(
            config,
            args,
            [parent_rows[parent_id] for parent_id in selected_ids],
            decision,
            output_dir,
            decision_sha,
            parameter_sha,
            round_number,
            target,
        )
        atomic_write_jsonl(output_path, rows, refuse_existing=True)
        print(f"optimization controller: round={round_number} children={len(rows)} output={output_path}")
        return 0
    variants = require_int(config_optimization(config).get("variants_per_parent"), "config optimization.variants_per_parent", minimum=1)
    child_specs: list[tuple[dict[str, Any], str, int, int]] = []
    for parent_id in selected_ids:
        parent = parent_rows[parent_id]
        for variant_index in range(variants):
            child_id = f"{parent_id}-r{round_number}v{variant_index}"
            safe_child_id(child_id)
            child_specs.append((parent, child_id, variant_index, optimizer_seed(child_id)))
            sequence_path = output_dir / "sequences" / f"{child_id}.fasta"
            pose_path = output_dir / "poses" / f"{child_id}.pdb"
            if sequence_path.exists() or pose_path.exists():
                raise AdapterError(f"child output already exists for {child_id}")
    rows: list[dict[str, Any]] = []
    mutation_count = require_int(decision["parameter_overrides"].get("mutation_count"), "decision mutation_count", minimum=1)
    for parent, child_id, variant_index, seed in child_specs:
        row = child_base_row(parent, target, child_id, round_number, variant_index, decision, decision_sha, parameter_sha, seed)
        try:
            sequence = validate_sequence(parent.get("sequence_path"), str(parent["candidate_id"]))
            mutated = mutate_sequence(sequence, mutation_count, seed)
            source_pose = Path(require_text(parent.get("design_pose_path"), f"parent {parent['candidate_id']}.design_pose_path"))
            sequence_path = output_dir / "sequences" / f"{child_id}.fasta"
            pose_path = output_dir / "poses" / f"{child_id}.pdb"
            atomic_write_bytes(sequence_path, f">{child_id}\n{mutated}\n".encode("ascii"), refuse_existing=True)
            pose_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="wb", dir=pose_path.parent, prefix=f".{pose_path.name}.", delete=False) as handle:
                temporary_pose = Path(handle.name)
                with source_pose.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, handle)
            os.replace(temporary_pose, pose_path)
            row.update(
                {
                    "sequence_path": str(sequence_path.resolve()),
                    "sequence_sha256": hashlib.sha256(mutated.encode("ascii")).hexdigest(),
                    "sequence_length": len(mutated),
                    "design_pose_path": str(pose_path.resolve()),
                    "design_pose_sha256": sha256_file(pose_path),
                    "status": "generated",
                }
            )
        except Exception as exc:  # noqa: BLE001
            row["failure_reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    atomic_write_jsonl(output_path, rows, refuse_existing=True)
    print(f"optimization controller: round={round_number} children={len(rows)} output={output_path}")
    return 0


def validate_summary(summary: dict[str, Any], round_number: int) -> None:
    required = {"schema_version", "round", "source_manifest_sha256", "candidates", "budget", "failures"}
    missing = sorted(required - set(summary))
    if missing:
        raise AdapterError("round summary missing " + ", ".join(missing))
    if summary["schema_version"] != 1 or summary["round"] != round_number:
        raise AdapterError("round summary schema_version or round is invalid")
    if not isinstance(summary["candidates"], list) or not isinstance(summary["failures"], list):
        raise AdapterError("round summary candidates and failures must be lists")
    budget = summary["budget"]
    if not isinstance(budget, dict):
        raise AdapterError("round summary budget must be an object")
    require_int(budget.get("remaining_candidates"), "round summary remaining_candidates", minimum=0)
    require_int(budget.get("remaining_predictions"), "round summary remaining_predictions", minimum=0)
    for index, row in enumerate(summary["candidates"]):
        if not isinstance(row, dict):
            raise AdapterError(f"round summary candidate {index} is not an object")
        require_text(row.get("candidate_id"), f"round summary candidate {index}.candidate_id")
        require_text(row.get("origin_generator"), f"round summary candidate {index}.origin_generator")
        if row.get("status") not in {"promoted", "eligible", "failed", "rejected"}:
            raise AdapterError(f"round summary candidate {index} has an invalid status")


def validate_manifest_rows_for_parse(
    rows: list[dict[str, Any]],
    decision: dict[str, Any],
    round_number: int,
    decision_sha: str,
) -> list[str]:
    errors: list[str] = []
    if decision.get("stop") is True:
        seen: set[str] = set()
        for index, row in enumerate(rows):
            missing = [field for field in CHILD_FIELDS if field not in row]
            if missing:
                errors.append(f"stopped optimizer row {index} missing fields: {', '.join(missing)}")
                continue
            candidate_id = row.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
                errors.append(f"stopped optimizer row {index} has a missing or duplicate candidate_id")
            seen.add(str(candidate_id))
            if row.get("decision_sha256") != decision_sha:
                errors.append(f"stopped optimizer row {index} decision_sha256 does not match the plan receipt")
            if row.get("optimization_round") != round_number:
                errors.append(f"stopped optimizer row {index} has the wrong optimization_round")
            if row.get("status") != "generated":
                errors.append(f"stopped optimizer row {index} has an invalid status")
            try:
                sequence = validate_sequence(row.get("sequence_path"), str(candidate_id))
                if row.get("sequence_sha256") != hashlib.sha256(sequence.encode("ascii")).hexdigest():
                    errors.append(f"stopped optimizer row {index} sequence_sha256 does not match FASTA")
                pose_path = Path(require_text(row.get("design_pose_path"), f"stopped optimizer row {index}.design_pose_path"))
                if row.get("design_pose_sha256") != sha256_file(pose_path):
                    errors.append(f"stopped optimizer row {index} design_pose_sha256 does not match pose")
            except AdapterError as exc:
                errors.append(str(exc))
        if not rows:
            errors.append("stopped optimizer output must carry the scored pool")
        return errors
    if len(rows) != decision.get("expected_fanout"):
        errors.append("optimizer output count does not match decision.expected_fanout")
    seen: set[str] = set()
    parent_variants: set[tuple[str, int]] = set()
    selected = set(decision.get("selected_parent_ids", []))
    for index, row in enumerate(rows):
        missing = [field for field in CHILD_FIELDS if field not in row]
        if missing:
            errors.append(f"optimizer row {index} missing fields: {', '.join(missing)}")
            continue
        candidate_id = row.get("candidate_id")
        parent_id = row.get("parent_candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in seen:
            errors.append(f"optimizer row {index} has a missing or duplicate candidate_id")
        seen.add(str(candidate_id))
        if not isinstance(parent_id, str) or parent_id not in selected:
            errors.append(f"optimizer row {index} uses a parent outside the decision")
        variant = row.get("variant_index")
        if not isinstance(variant, int) or isinstance(variant, bool) or variant < 0:
            errors.append(f"optimizer row {index} variant_index must be a nonnegative integer")
        elif isinstance(parent_id, str):
            key = (parent_id, variant)
            if key in parent_variants:
                errors.append(f"optimizer row {index} duplicates a parent and variant_index")
            parent_variants.add(key)
            if decision.get("operation") == "point-mutation":
                expected_child = f"{parent_id}-r{round_number}v{variant}"
                if candidate_id != expected_child:
                    errors.append(f"optimizer row {index} candidate_id does not match its parent and variant")
        if row.get("decision_sha256") != decision_sha:
            errors.append(f"optimizer row {index} decision_sha256 does not match the plan receipt")
        if decision.get("operation") == "point-mutation":
            if isinstance(candidate_id, str) and row.get("optimizer_seed") != optimizer_seed(candidate_id):
                errors.append(f"optimizer row {index} optimizer_seed does not match candidate_id")
        elif not isinstance(row.get("optimizer_seed"), int) or isinstance(row.get("optimizer_seed"), bool):
            errors.append(f"optimizer row {index} optimizer_seed must be an integer")
        if row.get("optimization_round") != round_number:
            errors.append(f"optimizer row {index} has the wrong optimization_round")
        if row.get("status") == "failed":
            if not isinstance(row.get("failure_reason"), str) or not row["failure_reason"]:
                errors.append(f"optimizer row {index} failed without failure_reason")
            continue
        if row.get("status") != "generated":
            errors.append(f"optimizer row {index} has an invalid status")
            continue
        try:
            sequence = validate_sequence(row.get("sequence_path"), str(candidate_id))
            if row.get("sequence_sha256") != hashlib.sha256(sequence.encode("ascii")).hexdigest():
                errors.append(f"optimizer row {index} sequence_sha256 does not match FASTA")
            if row.get("sequence_length") != len(sequence):
                errors.append(f"optimizer row {index} sequence_length does not match FASTA")
            pose_path = Path(require_text(row.get("design_pose_path"), f"optimizer row {index}.design_pose_path"))
            if row.get("design_pose_sha256") != sha256_file(pose_path):
                errors.append(f"optimizer row {index} design_pose_sha256 does not match pose")
        except AdapterError as exc:
            errors.append(str(exc))
    return errors


def parse_plan_stage(args: argparse.Namespace) -> int:
    round_number = parse_round(args.stage)
    output_dir = (args.attempt_dir / args.phase).resolve()
    summary_path = output_dir / SUMMARY_NAME
    decision_path = output_dir / DECISION_NAME
    errors: list[str] = []
    files = [path for path in (summary_path, decision_path) if path.is_file()]
    summary: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    if not summary_path.is_file():
        errors.append(f"declared output is missing: {summary_path}")
    else:
        try:
            summary = read_json_object(summary_path, "round summary")
            validate_summary(summary, round_number)
        except AdapterError as exc:
            errors.append(str(exc))
    if not decision_path.is_file():
        errors.append(f"declared output is missing: {decision_path}")
    else:
        try:
            decision = read_json_object(decision_path, "next-round decision")
        except AdapterError as exc:
            errors.append(str(exc))
    if summary is not None and decision is not None:
        try:
            config = read_json_object(args.config.resolve(), "campaign config")
            validate_decision_shape(config, summary, decision, summary_path, args.config.resolve(), round_number)
        except AdapterError as exc:
            errors.append(str(exc))
    result = {
        "ok": bool(files) and not errors,
        "parsed_count": 2 if bool(files) and not errors else 0,
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": sorted(sha256_file(path) for path in files),
    }
    result_path = output_dir / "parser-result.json"
    atomic_write_json(result_path, result)
    for error in errors:
        print(f"optimization controller parser: {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


def parse_execution_stage(args: argparse.Namespace) -> int:
    config = read_json_object(args.config.resolve(), "campaign config")
    round_number = parse_round(args.stage)
    plan_ids = [f"optimization-plan-round-{round_number}"]
    if args.stage == "optimize":
        plan_ids.append("optimization-plan")
    errors: list[str] = []
    try:
        decision_path, declared_sha = receipt_artifact(args.receipts_dir.resolve(), plan_ids, "next-round-decision")
        summary_path, _ = receipt_artifact(args.receipts_dir.resolve(), plan_ids, "round-summary")
        if declared_sha is not None and declared_sha != sha256_file(decision_path):
            raise AdapterError("optimization decision file hash does not match its receipt")
        decision = read_json_object(decision_path, "next-round decision")
        summary = read_json_object(summary_path, "round summary")
        validate_decision_shape(config, summary, decision, summary_path, args.config.resolve(), round_number)
        decision_sha = sha256_file(decision_path)
        verify_ledger_decision(args.artifact_root.resolve(), round_number, decision_sha)
    except AdapterError as exc:
        decision_path = None
        decision = None
        decision_sha = ""
        errors.append(str(exc))
    output_path = (args.attempt_dir / args.phase).resolve() / configured_output_name(config, args.stage, round_number)
    files = [output_path] if output_path.is_file() else []
    if not output_path.is_file():
        errors.append(f"declared output is missing: {output_path}")
    elif decision is not None:
        try:
            rows = read_jsonl(output_path, "optimized candidate manifest")
            errors.extend(validate_manifest_rows_for_parse(rows, decision, round_number, decision_sha))
        except AdapterError as exc:
            errors.append(str(exc))
    source_paths: list[Path] = list(files)
    if output_path.is_file():
        try:
            rows = read_jsonl(output_path, "optimized candidate manifest")
            for row in rows:
                for field in ("sequence_path", "design_pose_path"):
                    value = row.get(field)
                    if isinstance(value, str) and Path(value).is_file():
                        source_paths.append(Path(value))
        except AdapterError:
            pass
    result = {
        "ok": bool(files) and not errors,
        "parsed_count": len(read_jsonl(output_path, "optimized candidate manifest")) if files and not errors else 0,
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": sorted({sha256_file(path) for path in source_paths}),
    }
    result_path = (args.attempt_dir / args.phase).resolve() / "parser-result.json"
    atomic_write_json(result_path, result)
    for error in errors:
        print(f"optimization controller parser: {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("toolcheck")
    for name in ("run", "parse"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--stage", required=True)
        subparser.add_argument("--phase", required=True)
        subparser.add_argument("--count", type=int, default=1)
        subparser.add_argument("--attempt-dir", type=Path, required=True)
        subparser.add_argument("--receipts-dir", type=Path, required=True)
        subparser.add_argument("--artifact-root", type=Path, required=True)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "toolcheck":
            print("optimization controller ok, standard library and lineage contract")
            return 0
        if is_plan_stage(args.stage):
            return plan_stage(args) if args.command == "run" else parse_plan_stage(args)
        if args.stage == "optimize" or args.stage.startswith("optimize-round-"):
            return execute_stage(args) if args.command == "run" else parse_execution_stage(args)
        raise AdapterError(f"unsupported optimization stage: {args.stage}")
    except Exception as exc:  # noqa: BLE001
        print(f"optimization controller: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
