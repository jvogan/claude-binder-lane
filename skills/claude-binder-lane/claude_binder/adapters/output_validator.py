#!/usr/bin/env python3
"""Validate completed stage receipts and the terminal ranked portfolio.

The executor already owns receipt, artifact, parser-hash, observation, and ranking
contracts. This adapter calls those validators and writes one fail-closed output-check
record. It does not recompute file hashes or invent score fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import lane
from .. import control_separation
from .declared_artifacts import DeclaredArtifactError, input_files


PASS = "pass"
FAIL = "fail"
COULD_NOT_BE_CHECKED = "could not be checked"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    reason: str
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
        }
        if self.details:
            value["details"] = self.details
        return value


def _check(name: str, status: str, reason: str, **details: Any) -> Check:
    return Check(name, status, reason, details or None)


def _load(path: Path, label: str) -> Any:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{label} is invalid: {path}: {type(exc).__name__}: {exc}") from exc


def _stage_ids(plan: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    ordered = plan.get("ordered_stage_ids")
    if not isinstance(ordered, list) or any(not isinstance(item, str) for item in ordered):
        raise ValueError("run plan ordered_stage_ids is missing or malformed")
    if "output-check" not in ordered:
        raise ValueError("run plan does not declare output-check")
    position = ordered.index("output-check")
    return list(ordered), list(ordered[:position])


def _validate_receipts(
    plan: Mapping[str, Any],
    run_root: Path,
    receipts_dir: Path,
    previous_stage_ids: list[str],
) -> list[Check]:
    stages = plan.get("stages")
    adapters = plan.get("adapters")
    if not isinstance(stages, list) or not isinstance(adapters, list):
        return [_check("stage-receipts", FAIL, "run plan has no stage or adapter records")]
    stage_map = {
        str(item.get("stage_id")): item
        for item in stages
        if isinstance(item, Mapping) and isinstance(item.get("stage_id"), str)
    }
    adapter_map = {
        str(item.get("adapter_id")): item
        for item in adapters
        if isinstance(item, Mapping) and isinstance(item.get("adapter_id"), str)
    }
    checks: list[Check] = []
    for stage_id in previous_stage_ids:
        receipt_path = receipts_dir / f"{stage_id}.json"
        if not receipt_path.is_file():
            checks.append(_check(f"receipt:{stage_id}", COULD_NOT_BE_CHECKED, f"stage receipt is missing: {receipt_path}"))
            continue
        stage = stage_map.get(stage_id)
        if stage is None:
            checks.append(_check(f"receipt:{stage_id}", FAIL, "stage is absent from the run plan"))
            continue
        adapter = adapter_map.get(str(stage.get("adapter_id")))
        if adapter is None:
            checks.append(_check(f"receipt:{stage_id}", FAIL, "stage adapter is absent from the run plan"))
            continue
        try:
            receipt = lane.load_json(receipt_path)
            errors: list[str] = []
            if receipt.get("ok") is not True:
                errors.append("receipt ok is not true")
            if receipt.get("run_fingerprint") != plan.get("run_fingerprint"):
                errors.append("receipt run_fingerprint does not match the run plan")
            expected_identity = lane.stage_identity(dict(plan), dict(stage), dict(adapter), receipts_dir)
            if receipt.get("stage_identity") != expected_identity:
                errors.append("receipt stage_identity does not match the run plan and dependency receipt hashes")
            validation = lane.validate_completed_receipt(receipt, dict(stage), run_root)
            errors.extend(validation.get("errors", []))
        except Exception as exc:  # noqa: BLE001
            checks.append(_check(f"receipt:{stage_id}", FAIL, f"receipt validation raised {type(exc).__name__}: {exc}"))
            continue
        checks.append(
            _check(
                f"receipt:{stage_id}",
                PASS if not errors else FAIL,
                "receipt, artifact hashes, parser hashes, and completion markers passed" if not errors else "receipt validation failed",
                errors=errors[:20],
            )
        )
    if not checks:
        checks.append(_check("stage-receipts", COULD_NOT_BE_CHECKED, "the run plan has no stage receipts before output-check"))
    return checks


def _completion_check(receipts_dir: Path, previous_stage_ids: list[str]) -> Check:
    """Derive completion from the stage receipts declared by output-check."""
    missing: list[str] = []
    invalid: list[str] = []
    for stage_id in previous_stage_ids:
        path = receipts_dir / f"{stage_id}.json"
        if not path.is_file():
            missing.append(stage_id)
            continue
        try:
            receipt = lane.load_json(path)
        except Exception as exc:  # noqa: BLE001
            invalid.append(f"{stage_id}: {type(exc).__name__}: {exc}")
            continue
        if receipt.get("ok") is not True:
            invalid.append(f"{stage_id}: receipt ok is not true")
    if missing:
        return _check("stage-completion", COULD_NOT_BE_CHECKED, "stage receipts are missing", missing=missing)
    if invalid:
        return _check("stage-completion", FAIL, "stage receipts do not show completion", errors=invalid)
    return _check(
        "stage-completion",
        PASS,
        "all stages before output-check have successful receipts",
        completed_count=len(previous_stage_ids),
    )


def _report_all_mode(config: Mapping[str, Any], ranking: Mapping[str, Any]) -> bool:
    if ranking.get("report_mode") == "raw-metrics":
        return (
            ranking.get("scoring_arm_status") in {"degraded", "unvalidated"}
            and ranking.get("ranking_claim_status") in {"suppressed", "unvalidated"}
        )
    if not lane.is_candidate_claim(dict(config)):
        return False
    portfolio = ranking.get("portfolio")
    selected = ranking.get("selected_candidates")
    return (
        isinstance(portfolio, Mapping)
        and portfolio.get("mode") == "report-all"
    ) or (isinstance(selected, list) and not selected and isinstance(ranking.get("ranked_candidates"), list))


def _report_all_candidate_errors(rows: Any) -> list[str]:
    if not isinstance(rows, list) or not rows:
        return ["report-all output must contain every attempted candidate"]
    errors: list[str] = []
    candidate_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"report-all candidate row {index} is not an object")
            continue
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            errors.append(f"report-all candidate row {index} has no candidate_id")
        elif candidate_id in candidate_ids:
            errors.append(f"report-all candidate IDs are not unique: {candidate_id}")
        else:
            candidate_ids.add(candidate_id)
        status = row.get("status")
        if status in {"failed", "incomplete"}:
            reasons = row.get("failure_reasons")
            if not isinstance(row.get("failure_reason"), str) and not (
                isinstance(reasons, list) and any(isinstance(reason, str) and reason for reason in reasons)
            ):
                errors.append(f"report-all candidate {candidate_id} must preserve its failure reason")
        for field, value in row.items():
            if field.endswith("_sha256") and value is not None:
                if not isinstance(value, str) or lane.SHA256_RE.fullmatch(value) is None:
                    errors.append(f"report-all candidate {candidate_id} has an invalid {field}")
        source_hashes = row.get("source_hashes")
        if source_hashes is not None:
            if not isinstance(source_hashes, Mapping):
                errors.append(f"report-all candidate {candidate_id} source_hashes must be an object")
            else:
                for field, values in source_hashes.items():
                    if not str(field).endswith("_sha256") or not isinstance(values, list):
                        errors.append(f"report-all candidate {candidate_id} source_hashes is malformed")
                        continue
                    for value in values:
                        if not isinstance(value, str) or lane.SHA256_RE.fullmatch(value) is None:
                            errors.append(f"report-all candidate {candidate_id} has an invalid source hash")
    return errors


def _final_score_claim_errors(config: Mapping[str, Any], ranking: Mapping[str, Any]) -> list[str]:
    if not lane.is_candidate_claim(dict(config)) or "final_score" not in ranking:
        return []
    final_score = ranking.get("final_score")
    predictor_count = len(lane.enabled_items(config.get("cofold", {}).get("predictors"))) if isinstance(config.get("cofold"), Mapping) else 0
    errors: list[str] = []

    def inspect(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key).lower()
                if key_text in {"arm_count", "n_arms", "predictor_count", "ensemble_size"}:
                    if isinstance(child, int) and not isinstance(child, bool) and child != predictor_count:
                        errors.append(f"candidate claim final_score {path}.{key} reports {child} arms; expected {predictor_count}")
                elif key_text in {"arms", "predictors", "components"} and isinstance(child, (list, tuple)):
                    if len(child) != predictor_count:
                        errors.append(f"candidate claim final_score {path}.{key} reports {len(child)} arms; expected {predictor_count}")
                inspect(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)) and path.endswith(".final_score") and len(value) != predictor_count:
            errors.append(f"candidate claim final_score reports {len(value)} arms; expected {predictor_count}")

    inspect(final_score, "final_score")
    return errors


def _declared_file(
    plan: Mapping[str, Any],
    receipts_dir: Path,
    consumer_stage_id: str,
    artifact_id: str,
    source_stage_id: str,
) -> Path:
    try:
        _, paths = input_files(
            plan,
            receipts_dir,
            consumer_stage_id,
            artifact_id=artifact_id,
            source_stage_id=source_stage_id,
        )
    except DeclaredArtifactError as exc:
        raise ValueError(str(exc)) from exc
    if len(paths) != 1:
        raise ValueError(
            f"stage {source_stage_id} should have written one {artifact_id} file, "
            f"but its receipt names {len(paths)} files"
        )
    return paths[0]


def _score_check(
    config: Mapping[str, Any],
    artifact_root: Path,
    *,
    plan: Mapping[str, Any],
    receipts_dir: Path,
) -> tuple[Check, int, list[dict[str, Any]]]:
    try:
        score_path = _declared_file(
            plan, receipts_dir, "output-check", "screen-score-table", "score-screen"
        )
        observations_path = _declared_file(
            plan, receipts_dir, "output-check", "uniform-observations", "uniform-rescore"
        )
        ranked_path = _declared_file(
            plan, receipts_dir, "output-check", "ranked-portfolio", "final-rank"
        )
    except ValueError as exc:
        return _check("score-coverage", FAIL, str(exc)), 0, []
    if not score_path.is_file() or not observations_path.is_file() or not ranked_path.is_file():
        missing = [str(path) for path in (score_path, observations_path, ranked_path) if not path.is_file()]
        return _check("score-coverage", COULD_NOT_BE_CHECKED, "score or ranking input is missing", missing=missing), 0, []
    errors: list[str] = []
    try:
        screen = lane.validate_screen_scored_pool(config=dict(config), score_table_path=score_path, artifact_root=artifact_root)
        errors.extend(f"screen coverage: {error}" for error in screen.get("errors", []))
        observations = lane.load_jsonl(observations_path)
        stored_rank = lane.load_json(ranked_path)
        report_all = _report_all_mode(config, stored_rank)
        delivered_count = lane.delivered_selection_count(dict(config), artifact_root)
        recomputed = lane.rank_candidates(
            dict(config), observations, final_count=delivered_count if not report_all else None
        )
        if recomputed.get("ok") is not True:
            errors.extend(f"rank recomputation: {error}" for error in recomputed.get("errors", []))
        elif lane.ranking_recomputation_differences(stored_rank, recomputed):
            errors.append("ranked portfolio does not match the executor recomputation")
        errors.extend(_final_score_claim_errors(config, stored_rank))
        if report_all:
            errors.extend(_report_all_candidate_errors(stored_rank.get("ranked_candidates")))
            if recomputed.get("report_mode") != stored_rank.get("report_mode"):
                errors.append("report-all output does not match the ranking claim mode")
        selected = stored_rank.get("selected_candidates")
        if not isinstance(selected, list):
            errors.append("ranked portfolio selected_candidates is missing or malformed")
            selected_count = 0
        else:
            selected_count = len(selected)
        if report_all and selected_count != 0:
            errors.append("report-all output must have no portfolio selection")
        if not report_all and recomputed.get("ok") is True and selected_count != len(recomputed.get("selected_candidates", [])):
            errors.append("selected count does not match the executor recomputation")
    except Exception as exc:  # noqa: BLE001
        return _check("score-coverage", FAIL, f"score coverage validation raised {type(exc).__name__}: {exc}"), 0, []
    reported = stored_rank.get("ranked_candidates", []) if report_all and isinstance(stored_rank.get("ranked_candidates"), list) else []
    summary = (
        "raw per-design metrics completed with a suppressed ranking claim"
        if stored_rank.get("report_mode") == "raw-metrics"
        else "screen score coverage and final ranking passed"
    )
    return (
        _check(
            "score-coverage",
            PASS if not errors else FAIL,
            summary if not errors else "score coverage or ranking validation failed",
            errors=errors[:20],
        ),
        selected_count,
        reported,
    )


def _selected_paths_check(
    config: Mapping[str, Any],
    artifact_root: Path,
    *,
    plan: Mapping[str, Any],
    receipts_dir: Path,
) -> Check:
    try:
        ranked_path = _declared_file(
            plan, receipts_dir, "output-check", "ranked-portfolio", "final-rank"
        )
    except ValueError as exc:
        return _check("selected-paths", FAIL, str(exc))
    if not ranked_path.is_file():
        return _check("selected-paths", COULD_NOT_BE_CHECKED, f"ranked portfolio is missing: {ranked_path}")
    try:
        ranking = lane.load_json(ranked_path)
    except Exception as exc:  # noqa: BLE001
        return _check("selected-paths", FAIL, f"ranked portfolio is invalid: {type(exc).__name__}: {exc}")
    selected = ranking.get("selected_candidates")
    ranked = ranking.get("ranked_candidates")
    if not isinstance(selected, list) or not isinstance(ranked, list):
        return _check("selected-paths", FAIL, "ranked portfolio does not contain object lists for selected and ranked candidates")
    if _report_all_mode(config, ranking):
        errors = _report_all_candidate_errors(ranked)
        if selected:
            errors.append("report-all output must leave selected_candidates empty")
        return _check(
            "selected-paths",
            PASS if not errors else FAIL,
            "report-all candidates preserve statuses, failure reasons, and hashes" if not errors else "report-all candidate lineage validation failed",
            selected_count=0,
            reported_candidate_count=len(ranked),
            errors=errors[:20],
        )
    ranked_by_id = {
        str(row.get("candidate_id")): row
        for row in ranked
        if isinstance(row, Mapping) and row.get("candidate_id") is not None
    }
    minimum = config.get("binder", {}).get("minimum_length") if isinstance(config.get("binder"), Mapping) else None
    maximum = config.get("binder", {}).get("maximum_length") if isinstance(config.get("binder"), Mapping) else None
    if not isinstance(minimum, int) or not isinstance(maximum, int):
        return _check("selected-paths", COULD_NOT_BE_CHECKED, "binder sequence length bounds are missing from the campaign")
    errors: list[str] = []
    for index, row in enumerate(selected):
        if not isinstance(row, dict):
            errors.append(f"selected row {index} is not an object")
            continue
        candidate_id = str(row.get("candidate_id", ""))
        source = ranked_by_id.get(candidate_id)
        if source is None:
            errors.append(f"selected candidate {candidate_id} is absent from ranked_candidates")
            continue
        errors.extend(lane.validate_sequence_row(row, label=f"selected candidate {candidate_id}", minimum_length=minimum, maximum_length=maximum))
        errors.extend(lane.validate_design_pose_row(row, label=f"selected candidate {candidate_id}"))
        for field in ("sequence_sha256", "sequence_length", "design_pose_path", "design_pose_sha256"):
            if row.get(field) != source.get(field):
                errors.append(f"selected candidate {candidate_id} changed {field} from ranked_candidates")
    return _check(
        "selected-paths",
        PASS if not errors else FAIL,
        "selected sequences and design poses exist, hash, and preserve ranked lineage" if not errors else "selected path or lineage validation failed",
        selected_count=len(selected),
        errors=errors[:20],
    )


def validate_output(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    artifact_root: Path,
    receipts_dir: Path,
) -> dict[str, Any]:
    checks: list[Check] = []
    try:
        ordered, previous = _stage_ids(plan)
        checks.append(_check("run-plan", PASS, "run plan has an ordered stage graph and output-check stage"))
    except ValueError as exc:
        ordered, previous = [], []
        checks.append(_check("run-plan", FAIL, str(exc)))
    fingerprint = plan.get("run_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        checks.append(_check("run-fingerprint", FAIL, "run plan has no 64-character run_fingerprint"))
    else:
        checks.append(_check("run-fingerprint", PASS, "run fingerprint is present"))
    if previous:
        checks.extend(_validate_receipts(plan, run_root, receipts_dir, previous))
        checks.append(_completion_check(receipts_dir, previous))
    else:
        checks.append(_check("stage-receipts", COULD_NOT_BE_CHECKED, "upstream receipt set could not be resolved"))
    score_check, selected_count, reported_candidates = _score_check(
        config,
        artifact_root,
        plan=plan,
        receipts_dir=receipts_dir,
    )
    checks.append(score_check)
    checks.append(
        _selected_paths_check(
            config,
            artifact_root,
            plan=plan,
            receipts_dir=receipts_dir,
        )
    )
    roster_value = config.get("runtime", {}).get("model_roster_path") if isinstance(config.get("runtime"), Mapping) else None
    roster_path = Path(str(roster_value)) if isinstance(roster_value, str) else None
    if roster_path is not None and not roster_path.is_absolute():
        context = plan.get("context")
        config_path_value = context.get("config_path") if isinstance(context, Mapping) else None
        base = Path(str(config_path_value)).resolve().parent if isinstance(config_path_value, str) else Path.cwd()
        roster_path = (base / roster_path).resolve()
    separation = (
        control_separation.assess_roster_path(
            config,
            roster_path,
            enforce=lane.is_production_scoring(config),
        )
        if roster_path is not None
        else {
            "ok": False if lane.is_production_scoring(config) else True,
            "status": "incomplete" if lane.is_production_scoring(config) else "not_required",
            "statistic": control_separation.STATISTIC,
            "metric": control_separation.METRIC,
            "measurements": [],
            "errors": (["runtime.model_roster_path is missing"] if lane.is_production_scoring(config) else []),
        }
    )
    checks.append(
        _check(
            "control-separation",
            PASS if separation["ok"] else FAIL,
            "stored control separation is present for every scoring arm and target"
            if separation["ok"]
            else "output-check cannot certify control separation",
            statistic=separation.get("statistic"),
            measurements=separation.get("measurements", []),
            errors=separation.get("errors", []),
        )
    )
    ok = bool(checks) and all(item.status == PASS for item in checks)
    return {
        "ok": ok,
        "stage": "output-check",
        "run_fingerprint": fingerprint,
        "stage_count": len(ordered),
        "completed_stage_count": len(previous),
        "selected_count": selected_count,
        "claim": lane.claim_metadata(dict(config)),
        "control_separation": separation,
        "reported_candidate_count": len(reported_candidates),
        **({"reported_candidates": reported_candidates} if reported_candidates else {}),
        "checks": [item.as_dict() for item in checks],
        "errors": [item.reason for item in checks if item.status != PASS],
    }


def _parse_report(report_path: Path, result_path: Path) -> int:
    errors: list[str] = []
    parsed_count = 0
    source_hashes: list[str] = []
    if not report_path.is_file():
        errors.append(f"output-check report is missing: {report_path}")
    else:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            parsed_count = 1
            source_hashes.append(lane.sha256_file(report_path))
            if not isinstance(report, dict):
                errors.append("output-check report is not a JSON object")
            else:
                for field in ("ok", "run_fingerprint", "stage_count", "selected_count"):
                    if field not in report:
                        errors.append(f"output-check report is missing {field}")
                if report.get("ok") is not True:
                    errors.extend(str(error) for error in report.get("errors", ["output-check report ok is not true"]))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"output-check report is invalid: {type(exc).__name__}: {exc}")
    result = {
        "ok": not errors,
        "parsed_count": parsed_count,
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": source_hashes,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output validator parser: parsed_count={parsed_count} ok={result['ok']}")
    for error in errors:
        print(f"- {error}", file=sys.stderr)
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
        if name == "run":
            subparser.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "toolcheck":
        print("output validator ok, executor-owned receipt and ranking checks only")
        return 0
    output_path = (args.attempt_dir / args.phase / "output-check.json").resolve()
    if args.command == "parse":
        return _parse_report(
            output_path,
            (args.attempt_dir / args.phase / "parser-result.json").resolve(),
        )
    try:
        config = _load(args.config, "campaign config")
        plan = _load(args.plan, "run plan")
        report = validate_output(
            config,
            plan,
            run_root=args.run_root.resolve(),
            artifact_root=args.artifact_root.resolve(),
            receipts_dir=args.receipts_dir.resolve(),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"output validator: ERROR: {exc}", file=sys.stderr)
        return 2
    print("output validator: PASS" if report["ok"] else "output validator: FAIL")
    for check in report["checks"]:
        if check["status"] != PASS:
            print(f"- {check['name']}: {check['status']}: {check['reason']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
