#!/usr/bin/env python3
"""Reducer for the score-screen stage.

This adapter computes nothing. Each configured cofold mode measures ipSAE, sc_DockQ and the
site metrics inside its own container and writes measurement-source.json beside each
prediction. This script reads the arms' receipts, unpacks the measurement that already
exists, and writes screen-score-table.jsonl.

That split is resolution A in report_contract.md section 5. It is what the executor
supports today: status has two values, and scored requires a measurement source
carrying all 48 fields.
"""

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from claude_binder.arms import score_instrument_arm_name

from .declared_artifacts import input_files, load_plan, output_artifact_id

# The score table holds one row per prediction the arms produced. Controls come from
# control-calibration and stay out of it, because validate_screen_scored_pool requires
# the table's keys to be exactly targets by passing candidates by predictors by screen
# seeds.
SCREEN_SCORE_ARTIFACT_ID = "screen-score-table"
INTERMEDIATE_SCORE_ARTIFACT_ID = "intermediate-score-table"
OPTIMIZATION_SCORE_ARTIFACT_ID = "optimization-score-table"

OPTIMIZATION_MEASUREMENT_STAGE = re.compile(r"optimization-measure-round-([1-9][0-9]*)")

# A cofold stage writes one artifact per phase into the same receipt. A smoke-then-scale
# run holds smoke and scale, a single run holds single. The reducer takes the widest
# phase present, which is the one the executor's own pool check reads.
PHASE_PREFERENCE = ("scale", "single", "smoke")

RAW_PREDICTION_ARTIFACT_TYPE = "raw-prediction-manifest"


class ScorerError(RuntimeError):
    """The inputs cannot produce a score table the executor would accept."""


def main() -> int:
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
    args = parser.parse_args()
    if args.command == "toolcheck":
        return toolcheck()
    return run(args) if args.command == "run" else parse(args)


def toolcheck() -> int:
    """Report that the reducer can run. It needs no tool beyond the standard library."""
    print("interface-scorer ok, standard library only, no metric implementation required")
    return 0


def run(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    plan = load_plan(args.plan, config)
    score_artifact_id = score_artifact_for_stage(args.stage)

    attempt_id = args.attempt_dir.name
    rows: list = []
    seen: dict = {}
    for predictor in enabled_items(config["cofold"]["predictors"]):
        stage_id = cofold_stage_for_score_stage(args.stage, predictor)
        for raw in raw_prediction_rows(
            args.receipts_dir,
            stage_id,
            plan=plan,
            consumer_stage_id=args.stage,
        ):
            key = (
                str(raw.get("target_id")),
                str(raw.get("candidate_id")),
                str(raw.get("predictor")),
                raw.get("seed"),
            )
            if key in seen:
                raise ScorerError(
                    f"{stage_id} and {seen[key]} both claim {key}, and the score table "
                    "takes one row per prediction"
                )
            seen[key] = stage_id
            rows.append(
                observation_from_raw(
                    raw,
                    attempt_id=attempt_id,
                    stage_id=stage_id,
                )
            )

    if not rows:
        raise ScorerError(
            f"the cofold receipts for {args.stage} produced no rows, so there is nothing "
            "to score. Check that each arm's receipt declares a "
            f"{RAW_PREDICTION_ARTIFACT_TYPE} artifact."
        )

    write_jsonl(
        artifact_output_path(
            config,
            args.stage,
            score_artifact_id,
            args.attempt_dir,
            args.phase,
        ),
        rows,
    )
    return 0


def score_artifact_for_stage(stage_id: str) -> str:
    """Return the score-table artifact declared by a supported scorer stage."""
    if stage_id == "score-screen":
        return SCREEN_SCORE_ARTIFACT_ID
    if stage_id == "score-intermediate":
        return INTERMEDIATE_SCORE_ARTIFACT_ID
    if OPTIMIZATION_MEASUREMENT_STAGE.fullmatch(stage_id):
        return OPTIMIZATION_SCORE_ARTIFACT_ID
    raise ScorerError(
        "interface_scorer implements score-screen, score-intermediate and optimization-measure-round-N "
        f"for positive integer N; received stage {stage_id}"
    )


def cofold_stage_for_score_stage(stage_id: str, predictor: dict) -> str:
    """Return the cofold producer whose receipt feeds a scorer stage."""
    if stage_id == "score-screen":
        screen_stage = predictor.get("screen_stage")
        if not isinstance(screen_stage, str) or not screen_stage:
            raise ScorerError(
                f"predictor {predictor.get('id')} has no screen_stage for score-screen"
            )
        return screen_stage
    if stage_id == "score-intermediate":
        return f"cofold-intermediate-{predictor['id']}"
    match = OPTIMIZATION_MEASUREMENT_STAGE.fullmatch(stage_id)
    if match:
        return f"optimization-cofold-round-{match.group(1)}-{predictor['id']}"
    score_artifact_for_stage(stage_id)
    raise AssertionError(f"unreachable supported stage: {stage_id}")


def parse(args: argparse.Namespace) -> int:
    """Count the rows this stage wrote and hash the files it wrote them to."""
    config = load_json(args.config)
    stage = stage_record(config, args.stage)
    files: list = []
    parsed_count = 0
    errors: list = []
    for output in stage["outputs"]:
        pattern = render(output["path_template"], attempt_dir=args.attempt_dir, phase=args.phase)
        for value in sorted(glob.glob(pattern, recursive=True)):
            path = Path(value)
            if not path.is_file():
                continue
            files.append(path)
            try:
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    json.loads(line)
                    parsed_count += 1
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    write_json(
        args.attempt_dir / args.phase / "parser-result.json",
        {
            "ok": bool(files) and not errors,
            "parsed_count": parsed_count,
            "rejected_count": len(errors),
            "errors": errors,
            "source_output_hashes": sorted(sha256_file(path) for path in files),
        },
    )
    return 0 if files and not errors else 1


def raw_prediction_rows(
    receipts_dir: Path,
    stage_id: str,
    *,
    plan: dict[str, Any] | None = None,
    consumer_stage_id: str = "score-screen",
) -> list:
    """Read one arm's raw prediction rows out of its receipt.

    Only the raw prediction manifest holds rows. A cofold stage declares the structures
    it wrote alongside them, and a CIF read as JSON raises.
    """
    if plan is not None:
        artifact_id = output_artifact_id(
            plan,
            stage_id,
            artifact_type=RAW_PREDICTION_ARTIFACT_TYPE,
        )
        _, paths = input_files(
            plan,
            receipts_dir,
            consumer_stage_id,
            artifact_id=artifact_id,
            source_stage_id=stage_id,
            phase_preference=PHASE_PREFERENCE,
        )
        return [row for path in paths for row in load_jsonl(path)]

    receipt_path = Path(receipts_dir) / f"{stage_id}.json"
    if not receipt_path.is_file():
        raise ScorerError(f"the receipt for {stage_id} is missing: {receipt_path}")
    receipt = load_json(receipt_path)
    artifacts = [
        artifact
        for artifact in receipt.get("output_manifest", {}).get("artifacts", [])
        if artifact.get("artifact_type") == RAW_PREDICTION_ARTIFACT_TYPE
    ]
    if not artifacts:
        raise ScorerError(
            f"{stage_id} declared no {RAW_PREDICTION_ARTIFACT_TYPE} artifact, so it "
            "wrote no prediction rows"
        )
    phases = {str(artifact.get("phase")) for artifact in artifacts}
    selected_phase = next((phase for phase in PHASE_PREFERENCE if phase in phases), None)
    if selected_phase is None:
        raise ScorerError(
            f"{stage_id} wrote its predictions under phases {sorted(phases)}, and the "
            f"reducer reads {', '.join(PHASE_PREFERENCE)}"
        )
    rows: list = []
    for artifact in artifacts:
        if artifact.get("phase") != selected_phase:
            continue
        # A sharded stage lists one file per shard under the same artifact.
        for file_record in artifact.get("files", []):
            path = Path(file_record["path"])
            if not path.is_file():
                raise ScorerError(f"{stage_id} named a prediction manifest that is gone: {path}")
            rows.extend(load_jsonl(path))
    return rows


def observation_from_raw(
    raw: dict,
    *,
    attempt_id: str,
    stage_id: str,
    declared_measurement_files: list[Path] | None = None,
) -> dict:
    """Turn one raw prediction row into one score table row.

    A scored row spreads the measurement the arm already wrote and adds the paths. A
    failed row carries the failure fields and no measurement, which is what lets a job
    that lost one candidate still produce a complete table.
    """
    control_type = str(raw.get("control_type", "candidate"))
    generator = "control" if control_type != "candidate" else raw.get("origin_generator")
    common = {
        "generator": generator,
        # validate_screen_scored_pool reads origin_generator off every row, whatever its
        # status, and compares it against the passing candidate manifest.
        "origin_generator": raw.get("origin_generator"),
        "attempt_id": attempt_id,
        "control_type": control_type,
        "control_role": raw.get("control_role"),
        "control_structure_sha256": raw.get("control_structure_sha256"),
        "score_instrument": score_instrument_arm_name(str(raw.get("predictor", ""))),
        "filter_pass": True,
        "design_pose_path": raw.get("design_pose_path"),
        "raw_prediction_record_sha256": sha256_json(raw),
    }
    if raw.get("status") == "failed":
        mapping = raw.get("chain_mapping", {})
        return {
            **common,
            "target_id": raw.get("target_id"),
            "candidate_id": raw.get("candidate_id"),
            "predictor": raw.get("predictor"),
            "seed": raw.get("seed"),
            "phase": raw.get("phase"),
            "model_revision": raw.get("model_revision"),
            "status": "failed",
            "target_sha256": raw.get("target_sha256"),
            "sequence_sha256": raw.get("sequence_sha256"),
            "target_chain_id": mapping.get("target"),
            "binder_chain_id": mapping.get("binder"),
            "design_pose_sha256": raw.get("design_pose_sha256"),
            "chain_mapping": mapping,
            "failure_code": raw.get("failure_code"),
            "failure_reason": raw.get("failure_reason"),
        }
    measurement = load_measurement(raw, stage_id, declared_measurement_files)
    return {
        **measurement,
        **common,
        "predicted_complex_path": raw.get("predicted_complex_path"),
        "predicted_complex_sha256": raw.get("predicted_complex_sha256"),
        "pae_path": raw.get("pae_path"),
        "pae_sha256": raw.get("pae_sha256"),
        "metric_source_path": raw.get("metric_source_path"),
        "metric_source_sha256": raw.get("metric_source_sha256"),
    }


def load_measurement(
    raw: dict,
    stage_id: str,
    declared_measurement_files: list[Path] | None = None,
) -> dict:
    """Read the inline measurement carried by the declared raw manifest.

    The optional file argument remains for focused contract tests and callers that have
    an explicit file declaration. Production stage paths use the inline measurement,
    because the raw prediction manifest is the declared upstream artifact.
    """
    label = f"{stage_id} {raw.get('candidate_id')} seed {raw.get('seed')}"
    inline = raw.get("measurement")
    if declared_measurement_files is None:
        if not isinstance(inline, dict):
            raise ScorerError(
                f"{label} has no inline measurement in the declared raw prediction "
                f"manifest; stage {stage_id} should have written it"
            )
        return inline
    if not declared_measurement_files:
        raise ScorerError(
            f"{label} requires a declared measurement-source artifact; "
            f"stage {stage_id} should have declared the upstream measurement files"
        )
    named_path = Path(str(raw.get("metric_source_path", ""))).expanduser().resolve()
    declared_paths = {path.expanduser().resolve() for path in declared_measurement_files}
    if named_path not in declared_paths:
        raise ScorerError(
            f"{label} names measurement source {named_path}, but the upstream receipt "
            f"declares {sorted(map(str, declared_paths))}"
        )
    path = named_path
    if not path.is_file():
        raise ScorerError(f"{label} names a measurement source that is missing: {path}")
    observed = sha256_file(path)
    if observed != raw.get("metric_source_sha256"):
        raise ScorerError(
            f"{label} measurement source does not hash to the value on its row. The row "
            f"says {raw.get('metric_source_sha256')} and the file is {observed}."
        )
    value = load_json(path)
    if value.get("schema_version") != 1 or not isinstance(value.get("measurement"), dict):
        raise ScorerError(f"{label} measurement source needs schema_version 1 and measurement")
    return value["measurement"]


def enabled_items(items: Any) -> list:
    return [item for item in (items or []) if item.get("enabled", True)]


def stage_record(config: dict, stage_id: str) -> dict:
    for stage in config["stages"]:
        if stage["stage_id"] == stage_id:
            return stage
    raise ScorerError(f"the campaign declares no stage {stage_id}")


def render(value: str, *, attempt_dir: Path, phase: str) -> str:
    return value.replace("{{attempt_dir}}", str(attempt_dir)).replace("{{phase}}", phase)


def artifact_output_path(
    config: dict,
    stage_id: str,
    artifact_id: str,
    attempt_dir: Path,
    phase: str,
) -> Path:
    stage = stage_record(config, stage_id)
    for output in stage["outputs"]:
        if output["artifact_id"] == artifact_id:
            pattern = render(output["path_template"], attempt_dir=attempt_dir, phase=phase)
            if any(character in pattern for character in "*?["):
                raise ScorerError(f"the output path cannot be a glob: {pattern}")
            return Path(pattern)
    raise ScorerError(f"{stage_id} declares no output {artifact_id}")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a row the way the executor does, so raw_prediction_record_sha256 matches."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
