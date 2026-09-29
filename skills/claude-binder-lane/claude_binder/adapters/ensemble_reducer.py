#!/usr/bin/env python3
"""Verify uniform rescore coverage and write the observation table.

The reducer joins raw prediction receipts with the calibrated control table. It
preserves one row per target, candidate, predictor, and seed. It verifies that
``ipsae_min`` is the minimum of the two directed ipSAE values, then exposes a
small seed-reduction function that selects the maximum per predictor while
keeping sc_DockQ paired to that winning seed.

The reducer does not calculate z-scores. The published protocol makes z-scores
depend on the scored pool, so the final ranking stage owns that transductive
calculation. A non-default metric reduction returns generic metric-labelled
fields, so a supporting metric cannot be reported as ipSAE or sc_DockQ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from claude_binder import lane
from claude_binder.adapters import interface_scorer
from claude_binder.adapters.adapter_io import read_jsonl as _read_jsonl
from claude_binder.adapters.binder_contract import MEASUREMENT_SOURCE_FIELDS

from .declared_artifacts import input_files, load_plan


OUTPUT_NAME = "uniform-observations.jsonl"
RAW_PREDICTION_ARTIFACT_TYPE = "raw-prediction-manifest"
PHASE_PREFERENCE = ("scale", "single", "smoke")


class AdapterError(RuntimeError):
    """An input or coverage condition that must stop observation reduction."""


def stage_record(config: dict[str, Any], stage_id: str) -> dict[str, Any]:
    stages = config.get("stages")
    if not isinstance(stages, list):
        raise AdapterError("config stages must be a list")
    matches = [stage for stage in stages if isinstance(stage, dict) and stage.get("stage_id") == stage_id]
    if len(matches) != 1:
        raise AdapterError(f"config must define one stage: {stage_id}")
    return matches[0]


def published_artifact_path(
    config: dict[str, Any],
    artifact_root: Path,
    producer_stage_id: str,
    artifact_id: str,
) -> Path:
    """Resolve an input artifact through its producer's published path."""
    producer = stage_record(config, producer_stage_id)
    outputs = producer.get("outputs")
    if not isinstance(outputs, list):
        raise AdapterError(f"stage {producer_stage_id} outputs must be a list")
    matches = [
        output
        for output in outputs
        if isinstance(output, dict) and output.get("artifact_id") == artifact_id
    ]
    if len(matches) != 1:
        raise AdapterError(f"stage {producer_stage_id} must publish one {artifact_id} artifact")
    publish_path = matches[0].get("publish_path")
    if not isinstance(publish_path, str) or not publish_path or Path(publish_path).is_absolute():
        raise AdapterError(f"stage {producer_stage_id} {artifact_id} has no relative publish_path")
    return artifact_root / publish_path


def rescore_candidate_manifest_path(config: dict[str, Any], artifact_root: Path) -> Path:
    """Resolve the one candidate manifest declared by every rescore predictor."""
    paths: set[Path] = set()
    for predictor in predictor_records(config)[0]:
        stage_id = require_text(predictor.get("rescore_stage"), f"predictor {predictor.get('id')}.rescore_stage")
        stage = stage_record(config, stage_id)
        inputs = stage.get("inputs")
        if not isinstance(inputs, list):
            raise AdapterError(f"stage {stage_id} inputs must be a list")
        matches = [
            value.split(":", 1)
            for value in inputs
            if isinstance(value, str) and value.endswith(":rescore-candidates")
        ]
        if len(matches) != 1:
            raise AdapterError(f"stage {stage_id} must declare one rescore-candidates input")
        producer_stage_id, artifact_id = matches[0]
        paths.add(published_artifact_path(config, artifact_root, producer_stage_id, artifact_id))
    if len(paths) != 1:
        raise AdapterError(f"rescore predictors declare different candidate manifests: {sorted(map(str, paths))}")
    return paths.pop()


def controls_are_required(config: dict[str, Any]) -> bool:
    """Return whether the uniform table must include a calibrated control panel."""
    return not (
        lane.is_ungated_candidate_claim(config)
        and lane.control_panel_is_disabled(config)
    )


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AdapterError(f"refusing to write an empty uniform observation table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterError(f"{label} must be an integer")
    return value


def require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AdapterError(f"{label} must be a non-empty string")
    return value


def ipsae_min_from_directions(target_to_binder: Any, binder_to_target: Any, label: str = "ipsae") -> float:
    values = (target_to_binder, binder_to_target)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise AdapterError(f"{label} directed values must be numbers")
    numbers = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in numbers):
        raise AdapterError(f"{label} directed values must be finite numbers in [0, 1]")
    return min(numbers)


def verify_ipsae_min(row: dict[str, Any], label: str) -> None:
    if row.get("status") != "scored":
        return
    expected = ipsae_min_from_directions(
        row.get("ipsae_target_to_binder"),
        row.get("ipsae_binder_to_target"),
        label,
    )
    observed = row.get("ipsae_min")
    if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(float(observed)):
        raise AdapterError(f"{label}.ipsae_min must be a finite number")
    if not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-9):
        raise AdapterError(
            f"{label}.ipsae_min is {observed!r}, but the minimum of the two directed "
            f"ipSAE values is {expected!r}"
        )


def reduce_seed_ensemble(
    rows: list[dict[str, Any]],
    required_seeds: list[int],
    *,
    primary_metric: str = "ipsae_min",
    paired_metric: str = "sc_dockq",
) -> dict[str, Any]:
    """Reduce one candidate and predictor's seed rows to one paired summary.

    The primary metric uses the maximum over seeds. Ties use the smallest seed.
    The paired metric comes from that same winning row.
    """
    if not rows:
        raise AdapterError("cannot reduce an empty seed ensemble")
    if not required_seeds or len(set(required_seeds)) != len(required_seeds):
        raise AdapterError("ensemble required_seeds must be a non-empty list of unique seeds")
    by_seed: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = require_int(row.get("seed"), "ensemble seed")
        if seed in by_seed:
            raise AdapterError(f"ensemble repeats seed {seed}")
        by_seed[seed] = row
        if primary_metric == "ipsae_min":
            verify_ipsae_min(row, f"ensemble seed {seed}")
    expected = set(required_seeds)
    if set(by_seed) != expected:
        raise AdapterError(
            f"ensemble seed set {sorted(by_seed)} does not match {sorted(expected)}"
        )
    scored = [row for row in by_seed.values() if row.get("status") == "scored"]
    if len(scored) != len(required_seeds):
        raise AdapterError("cannot reduce an ensemble with failed seed observations")
    for seed, row in by_seed.items():
        if row.get("status") != "scored":
            continue
        value = row.get(primary_metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AdapterError(f"ensemble seed {seed} has no finite {primary_metric}")
        paired = row.get(paired_metric)
        if isinstance(paired, bool) or not isinstance(paired, (int, float)) or not math.isfinite(float(paired)):
            raise AdapterError(f"ensemble seed {seed} has no finite {paired_metric}")
    winner = max(
        scored,
        key=lambda row: (float(row[primary_metric]), -int(row["seed"])),
    )
    paired = float(winner[paired_metric])
    raw_metrics = {
        primary_metric: [float(by_seed[seed][primary_metric]) for seed in required_seeds],
    }
    if paired_metric not in raw_metrics:
        raw_metrics[paired_metric] = [float(by_seed[seed][paired_metric]) for seed in required_seeds]
    result: dict[str, Any] = {
        "selected_metric": primary_metric,
        "selected_metric_value": float(winner[primary_metric]),
        "paired_metric": paired_metric,
        "paired_metric_value": paired,
        "selected_seed": int(winner["seed"]),
        "seed_count": len(required_seeds),
        "raw_metrics": raw_metrics,
    }
    if primary_metric == "ipsae_min":
        result["ipsae_min"] = float(winner[primary_metric])
        result["raw_ipsae_min"] = raw_metrics[primary_metric]
    if paired_metric == "sc_dockq":
        result["sc_dockq"] = paired
    return result


def receipt_raw_rows(
    receipts_dir: Path,
    stage_id: str,
    *,
    plan: dict[str, Any] | None = None,
    consumer_stage_id: str = "uniform-rescore",
    artifact_id: str | None = None,
) -> list[dict[str, Any]]:
    if plan is not None:
        declared_id = artifact_id or f"{stage_id}-rescore"
        _, paths = input_files(
            plan,
            receipts_dir,
            consumer_stage_id,
            artifact_id=declared_id,
            phase_preference=PHASE_PREFERENCE,
        )
        return [row for path in paths for row in read_jsonl(path, f"raw predictions for {stage_id}")]

    receipt_path = receipts_dir / f"{stage_id}.json"
    receipt = read_json_object(receipt_path, f"receipt {stage_id}")
    if receipt.get("ok") is not True:
        raise AdapterError(f"receipt {stage_id} did not complete: {receipt_path}")
    artifacts = receipt.get("output_manifest", {}).get("artifacts")
    if not isinstance(artifacts, list):
        raise AdapterError(f"receipt {stage_id} has no artifact list: {receipt_path}")
    matching = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("artifact_type") == RAW_PREDICTION_ARTIFACT_TYPE
    ]
    if not matching:
        raise AdapterError(f"receipt {stage_id} declares no {RAW_PREDICTION_ARTIFACT_TYPE} artifact")
    phases = {str(artifact.get("phase")) for artifact in matching}
    selected_phase = next((phase for phase in PHASE_PREFERENCE if phase in phases), None)
    if selected_phase is None:
        raise AdapterError(
            f"receipt {stage_id} has phases {sorted(phases)}; expected one of {PHASE_PREFERENCE}"
        )
    rows: list[dict[str, Any]] = []
    for artifact in matching:
        if artifact.get("phase") != selected_phase:
            continue
        files = artifact.get("files")
        if not isinstance(files, list):
            raise AdapterError(f"receipt {stage_id} artifact has no file list")
        for file_record in files:
            if not isinstance(file_record, dict) or not isinstance(file_record.get("path"), str):
                raise AdapterError(f"receipt {stage_id} artifact has an invalid file path")
            rows.extend(read_jsonl(Path(file_record["path"]), f"raw predictions for {stage_id}"))
    if not rows:
        raise AdapterError(f"receipt {stage_id} produced no raw prediction rows")
    return rows


def target_records(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    targets = config.get("targets")
    if not isinstance(targets, list):
        raise AdapterError("config targets must be a list")
    records: dict[str, dict[str, Any]] = {}
    for target in targets:
        if not isinstance(target, dict):
            raise AdapterError("config targets contains a non-object")
        target_id = require_text(target.get("target_id"), "config target_id")
        if target_id in records:
            raise AdapterError(f"config repeats target_id: {target_id}")
        records[target_id] = target
    if not records:
        raise AdapterError("config targets is empty")
    return records


def predictor_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
    cofold = config.get("cofold")
    if not isinstance(cofold, dict):
        raise AdapterError("config cofold must be an object")
    seeds = cofold.get("rescore_seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise AdapterError("config cofold.rescore_seeds must be a non-empty integer list")
    if len(set(seeds)) != len(seeds):
        raise AdapterError("config cofold.rescore_seeds must contain unique seeds")
    records = [item for item in cofold.get("predictors", []) if isinstance(item, dict) and item.get("enabled", True) is True]
    if not records:
        raise AdapterError("config cofold.predictors has no enabled entries")
    for record in records:
        require_text(record.get("id"), "config cofold predictor id")
        require_text(record.get("rescore_stage"), f"predictor {record.get('id')}.rescore_stage")
    return records, [int(seed) for seed in seeds]


def validate_raw_identity(
    raw: dict[str, Any],
    *,
    predictor: dict[str, Any],
    targets: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    adapter_revisions: dict[str, Any],
    seeds: set[int],
) -> None:
    label = f"{predictor['id']} {raw.get('candidate_id')} seed {raw.get('seed')}"
    if raw.get("predictor") != predictor["id"]:
        raise AdapterError(f"{label} has predictor {raw.get('predictor')!r}")
    target_id = require_text(raw.get("target_id"), f"{label}.target_id")
    candidate_id = require_text(raw.get("candidate_id"), f"{label}.candidate_id")
    if target_id not in targets:
        raise AdapterError(f"{label} names unknown target {target_id}")
    if candidate_id not in candidates:
        raise AdapterError(f"{label} names candidate absent from the rescore candidate manifest")
    seed = require_int(raw.get("seed"), f"{label}.seed")
    if seed not in seeds:
        raise AdapterError(f"{label} uses unregistered seed {seed}")
    if raw.get("phase") != "uniform-rescore":
        raise AdapterError(f"{label} must carry phase=uniform-rescore")
    target = targets[target_id]
    candidate = candidates[candidate_id]
    expected = {
        "target_sha256": target.get("structure_sha256"),
        "model_revision": adapter_revisions.get(predictor.get("adapter_id")),
        "sequence_sha256": candidate.get("sequence_sha256"),
        "design_pose_path": candidate.get("design_pose_path"),
        "design_pose_sha256": candidate.get("design_pose_sha256"),
        "origin_generator": candidate.get("origin_generator"),
    }
    for field, expected_value in expected.items():
        if raw.get(field) != expected_value:
            raise AdapterError(f"{label} does not match {field} from its contract")
    if raw.get("control_type", "candidate") != "candidate":
        raise AdapterError(f"{label} is a control row in a predictor receipt")


def normalized_candidate_rows(
    config: dict[str, Any],
    args: argparse.Namespace,
    plan: dict[str, Any],
    targets: dict[str, dict[str, Any]],
    predictors: list[dict[str, Any]],
    seeds: list[int],
    candidates: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    adapters = {
        str(adapter.get("adapter_id")): adapter
        for adapter in config.get("adapters", [])
        if isinstance(adapter, dict) and isinstance(adapter.get("adapter_id"), str)
    }
    adapter_revisions = {key: value.get("model_revision") for key, value in adapters.items()}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    expected_keys = {
        (target_id, candidate_id, str(predictor["id"]), seed)
        for target_id in targets
        for candidate_id in candidates
        for predictor in predictors
        for seed in seeds
    }
    for predictor in predictors:
        stage_id = str(predictor["rescore_stage"])
        raw_artifact_id = f"{predictor['id']}-rescore"
        raw_rows = receipt_raw_rows(
            args.receipts_dir.resolve(),
            stage_id,
            plan=plan,
            consumer_stage_id=args.stage,
            artifact_id=raw_artifact_id,
        )
        for raw in raw_rows:
            validate_raw_identity(
                raw,
                predictor=predictor,
                targets=targets,
                candidates=candidates,
                adapter_revisions=adapter_revisions,
                seeds=set(seeds),
            )
            key = (str(raw["target_id"]), str(raw["candidate_id"]), str(raw["predictor"]), int(raw["seed"]))
            if key in seen:
                raise AdapterError(f"duplicate raw prediction key: {key}")
            seen.add(key)
            observation = interface_scorer.observation_from_raw(
                raw,
                attempt_id=args.attempt_dir.name,
                stage_id=stage_id,
            )
            if observation.get("status") == "scored":
                missing = sorted(set(MEASUREMENT_SOURCE_FIELDS) - set(observation))
                if missing:
                    raise AdapterError(f"{key} measurement is missing contract fields: {', '.join(missing)}")
                verify_ipsae_min(observation, f"observation {key}")
            rows.append(observation)
    if seen != expected_keys:
        missing = sorted(expected_keys - seen)
        extra = sorted(seen - expected_keys)
        raise AdapterError(f"raw prediction keys do not match the exact matrix: missing={missing[:8]} extra={extra[:8]}")
    return rows


def normalized_control_rows(
    config: dict[str, Any],
    args: argparse.Namespace,
    plan: dict[str, Any],
    artifact_root: Path,
) -> list[dict[str, Any]]:
    if not controls_are_required(config):
        return []
    _, control_files = input_files(
        plan,
        args.receipts_dir,
        args.stage,
        artifact_id="control-observations",
        source_stage_id="control-calibration",
    )
    control_rows = [
        row
        for path in control_files
        for row in read_jsonl(path, "control calibration table")
    ]
    validation = lane.validate_control_calibration(config, artifact_root)
    if not validation["ok"]:
        raise AdapterError("control calibration failed: " + "; ".join(validation["errors"][:8]))
    observations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for raw in control_rows:
        key = (str(raw.get("target_id")), str(raw.get("candidate_id")), str(raw.get("predictor")), int(raw.get("seed", -1)))
        if key in seen:
            raise AdapterError(f"duplicate control prediction key: {key}")
        seen.add(key)
        observation = interface_scorer.observation_from_raw(
            raw,
            attempt_id=args.attempt_dir.name,
            stage_id="control-calibration",
        )
        if observation.get("status") == "scored":
            missing = sorted(set(MEASUREMENT_SOURCE_FIELDS) - set(observation))
            if missing:
                raise AdapterError(f"control observation {key} is missing contract fields: {', '.join(missing)}")
            verify_ipsae_min(observation, f"control observation {key}")
        observations.append(observation)
    return observations


def run_stage(args: argparse.Namespace) -> int:
    config = read_json_object(args.config.resolve(), "campaign config")
    plan = load_plan(args.plan, config)
    artifact_root = args.artifact_root.resolve()
    targets = target_records(config)
    predictors, seeds = predictor_records(config)
    candidate_rows = read_jsonl(
        rescore_candidate_manifest_path(config, artifact_root),
        "rescore candidate manifest",
    )
    candidates: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(candidate_rows):
        candidate_id = require_text(row.get("candidate_id"), f"rescore candidate row {index}.candidate_id")
        if candidate_id in candidates:
            raise AdapterError(f"rescore candidate manifest repeats candidate_id: {candidate_id}")
        candidates[candidate_id] = row
    candidate_observations = normalized_candidate_rows(
        config,
        args,
        plan,
        targets,
        predictors,
        seeds,
        candidates,
    )
    control_observations = normalized_control_rows(config, args, plan, artifact_root)
    observations = sorted(
        [*candidate_observations, *control_observations],
        key=lambda row: (
            str(row.get("target_id")),
            str(row.get("candidate_id")),
            str(row.get("predictor")),
            int(row.get("seed", -1)),
        ),
    )
    validation_errors = lane.validate_observations(
        config,
        observations,
        expected_phase="uniform-rescore",
        required_seed_values=seeds,
        require_controls=True,
    )
    if validation_errors:
        raise AdapterError("uniform observation validation failed: " + "; ".join(validation_errors[:8]))
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_observations:
        grouped[(str(row["target_id"]), str(row["candidate_id"]), str(row["predictor"]))].append(row)
    for key, group in grouped.items():
        if all(row.get("status") == "scored" for row in group):
            reduce_seed_ensemble(group, seeds)
    output_dir = (args.attempt_dir / args.phase).resolve()
    output_path = output_dir / OUTPUT_NAME
    if output_path.exists():
        raise AdapterError(f"uniform observation output already exists: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_dir, prefix="uniform-observations-", suffix=".jsonl", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in observations))
    os.replace(temporary_path, output_path)
    print(
        f"ensemble reducer: observations={len(observations)} candidates={len(candidate_observations)} "
        f"controls={len(control_observations)} output={output_path}"
    )
    return 0


def parse_stage(args: argparse.Namespace) -> int:
    output_path = (args.attempt_dir / args.phase / OUTPUT_NAME).resolve()
    errors: list[str] = []
    files: list[Path] = []
    parsed_count = 0
    if not output_path.is_file():
        errors.append(f"declared output is missing: {output_path}")
    else:
        files.append(output_path)
        try:
            parsed_count = len(read_jsonl(output_path, "uniform observation table"))
        except AdapterError as exc:
            errors.append(str(exc))
    result_path = (args.attempt_dir / args.phase / "parser-result.json").resolve()
    result = {
        "ok": bool(files) and not errors,
        "parsed_count": parsed_count,
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": sorted(sha256_file(path) for path in files),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for error in errors:
        print(f"ensemble reducer parser: {error}", file=sys.stderr)
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
            print("ensemble reducer ok, standard library and binder measurement contract")
            return 0
        return run_stage(args) if args.command == "run" else parse_stage(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ensemble reducer: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
