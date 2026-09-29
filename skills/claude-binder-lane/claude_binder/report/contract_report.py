"""Build the terminal computational report from declared artifact paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DONE_REPORT_NAME = "REPORT.md"
DONE_RECEIPT_NAME = "verification-receipt.json"
DONE_MANIFEST_NAME = "done-manifest.json"
PRIMARY_METRIC_FIELD = "ipsae_min_ensemble"
PRIMARY_METRIC_FLOOR = 0.0
ALIGNED_ERROR_CUTOFF_ANGSTROM = 10.0


@dataclass(frozen=True)
class ContractReportResult:
    """The paths written by the declared-input report adapter."""

    report_path: Path
    receipt_path: Path
    manifest_path: Path
    ok: bool


def _load_object(path: Path) -> Mapping[str, Any] | None:
    """Read one declared JSON object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for one declared artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _todo(field_name: str, source: str) -> str:
    """Render a visible marker for a source field the report lacks."""

    return f"TODO: {field_name} missing; expected from {source}."


def _number(value: Any) -> float | None:
    """Return one finite numeric value."""

    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _candidate_rows(ranked_portfolio: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """Return the ranked candidate records from the declared portfolio."""

    if ranked_portfolio is None:
        return []
    rows = ranked_portfolio.get("ranked_candidates")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _candidate_id(row: Mapping[str, Any]) -> str | None:
    """Return a declared candidate identifier when available."""

    value = row.get("candidate_id")
    return value if isinstance(value, str) and value else None


def _rank_key(row: Mapping[str, Any]) -> tuple[int, str]:
    """Preserve the portfolio rank for rows that pass the primary-score floor."""

    rank = row.get("rank")
    if isinstance(rank, int) and not isinstance(rank, bool) and rank > 0:
        return rank, _candidate_id(row) or ""
    return sys.maxsize, _candidate_id(row) or ""


def _ranking_controls_pass(ranked_portfolio: Mapping[str, Any] | None) -> bool:
    """Return whether the ranker records passed controls."""

    if ranked_portfolio is None:
        return False
    controls = ranked_portfolio.get("controls")
    if not isinstance(controls, Mapping) or controls.get("ok") is not True:
        return False
    ranking_control = ranked_portfolio.get("ranking_control")
    return isinstance(ranking_control, Mapping) and ranking_control.get("status") == "passed"


def _report_text(
    target_manifest: Mapping[str, Any] | None,
    ranked_portfolio: Mapping[str, Any] | None,
    picture_manifest: Mapping[str, Any] | None,
    output_check: Mapping[str, Any] | None,
) -> tuple[str, bool, list[dict[str, str]]]:
    """Render floor accounting and guarded ranking from declared JSON inputs."""

    checks: list[dict[str, str]] = []

    def add_check(check_id: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "id": check_id,
                "status": "pass" if passed else "fail",
                "detail": detail,
            }
        )

    target_id = target_manifest.get("target_id") if target_manifest else None
    target_line = str(target_id) if isinstance(target_id, str) and target_id else _todo(
        "target_id", "target-manifest"
    )
    add_check("C1", target_manifest is not None, "target manifest is readable")

    output_ok = output_check is not None and output_check.get("ok") is True
    add_check("C2", output_ok, "output-check records ok=true")

    rows = _candidate_rows(ranked_portfolio)
    portfolio_rows_present = ranked_portfolio is not None and isinstance(
        ranked_portfolio.get("ranked_candidates"), list
    )
    add_check("C3", portfolio_rows_present, "ranked portfolio records ranked_candidates")

    missing_primary_ids = [
        candidate_id or "<missing candidate_id>"
        for row in rows
        if _number(row.get(PRIMARY_METRIC_FIELD)) is None
        for candidate_id in [_candidate_id(row)]
    ]
    numeric_rows = [
        (row, score)
        for row in rows
        for score in [_number(row.get(PRIMARY_METRIC_FIELD))]
        if score is not None
    ]
    floor_rows = [(row, score) for row, score in numeric_rows if score == PRIMARY_METRIC_FLOOR]
    above_floor_rows = [(row, score) for row, score in numeric_rows if score > PRIMARY_METRIC_FLOOR]
    controls_pass = _ranking_controls_pass(ranked_portfolio)
    add_check("C4", controls_pass, "ranked portfolio records passed controls")

    picture_count = picture_manifest.get("image_count") if picture_manifest else None
    picture_line = (
        str(picture_count)
        if isinstance(picture_count, int) and not isinstance(picture_count, bool)
        else _todo("image_count", "structure-picture-manifest")
    )
    add_check("C5", picture_manifest is not None, "picture manifest is readable")

    missing_field_todos: list[str] = []
    if not portfolio_rows_present:
        missing_field_todos.append(_todo("ranked_candidates", "ranked-portfolio"))
    if output_check is None or "ok" not in output_check:
        missing_field_todos.append(_todo("output-check.ok", "output-check"))
    controls = ranked_portfolio.get("controls") if ranked_portfolio else None
    if not isinstance(controls, Mapping) or "ok" not in controls:
        missing_field_todos.append(_todo("controls.ok", "ranked-portfolio"))
    ranking_control = ranked_portfolio.get("ranking_control") if ranked_portfolio else None
    if not isinstance(ranking_control, Mapping) or "status" not in ranking_control:
        missing_field_todos.append(_todo("ranking_control.status", "ranked-portfolio"))

    lines = [
        "# Computational report",
        "",
        "## Floor accounting",
        "",
        (
            f"The `{PRIMARY_METRIC_FIELD}` primary interface metric reaches exactly "
            f"{PRIMARY_METRIC_FLOOR:.1f} below the {ALIGNED_ERROR_CUTOFF_ANGSTROM:.0f} angstrom aligned-error cutoff."
        ),
        f"- Candidate rows: {len(rows)}",
        f"- Numeric primary scores: {len(numeric_rows)}",
        f"- Designs at the primary-score floor: {len(floor_rows)}",
    ]
    if missing_primary_ids:
        lines.extend(
            [
                "",
                _todo(
                    PRIMARY_METRIC_FIELD,
                    "ranked-portfolio for " + ", ".join(sorted(missing_primary_ids)),
                ),
            ]
        )
    if missing_field_todos:
        lines.extend(["", *missing_field_todos])

    lines.extend(
        [
            "",
            "## Declared inputs",
            "",
            f"- Target: {target_line}",
            f"- Picture count: {picture_line}",
        ]
    )

    floor_ids = [candidate_id for row, _ in floor_rows if (candidate_id := _candidate_id(row))]
    lines.extend(["", "## Tied designs", ""])
    if floor_ids:
        lines.append(
            f"{len(floor_ids)} designs tie at exactly {PRIMARY_METRIC_FLOOR:.6f}: "
            + ", ".join(sorted(floor_ids))
            + ". Order within this group carries no information."
        )
    else:
        lines.append("No design records have the primary-score floor.")

    refusal: str | None = None
    if not output_ok:
        refusal = "The output-check artifact does not record ok=true."
    elif not controls_pass:
        refusal = "The ranked portfolio does not record passed controls."
    elif not numeric_rows:
        refusal = "The ranked portfolio has no numeric primary scores."
    elif len(numeric_rows) < 2:
        refusal = "Fewer than two candidate rows carry numeric primary scores."
    elif len(floor_rows) == len(numeric_rows):
        refusal = (
            f"Every numeric primary score is exactly {PRIMARY_METRIC_FLOOR:.6f} under the "
            f"{ALIGNED_ERROR_CUTOFF_ANGSTROM:.0f} angstrom aligned-error cutoff."
        )

    lines.extend(["", "## Ranking", ""])
    if refusal:
        lines.append("Ranking withheld. " + refusal)
    else:
        lines.extend(["| Candidate ID | ipSAEmin |", "| --- | ---: |"])
        for row, score in sorted(above_floor_rows, key=lambda item: _rank_key(item[0])):
            candidate_id = _candidate_id(row)
            if candidate_id is None:
                continue
            lines.append(f"| {candidate_id} | {score:.6f} |")

    lines.extend(["", "## Verification", ""])
    lines.extend(
        f"- {check['id']}: {check['status']}. {check['detail']}" for check in checks
    )
    lines.append("")
    return "\n".join(lines), refusal is None, checks


def build_contract_report(
    *,
    target_manifest_path: Path,
    ranked_portfolio_path: Path,
    picture_manifest_path: Path,
    output_check_path: Path,
    report_path: Path,
    receipt_path: Path,
    manifest_path: Path,
) -> ContractReportResult:
    """Write a report from the four declared report-stage inputs."""

    inputs = {
        "target_manifest": target_manifest_path,
        "ranked_portfolio": ranked_portfolio_path,
        "picture_manifest": picture_manifest_path,
        "output_check": output_check_path,
    }
    target_manifest = _load_object(target_manifest_path)
    ranked_portfolio = _load_object(ranked_portfolio_path)
    picture_manifest = _load_object(picture_manifest_path)
    output_check = _load_object(output_check_path)
    report_text, ok, checks = _report_text(
        target_manifest,
        ranked_portfolio,
        picture_manifest,
        output_check,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    receipt = {
        "ok": ok,
        "checks": checks,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path) if path.is_file() else None}
            for name, path in inputs.items()
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "format": "claude-binder-declared-report-v1",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path) if path.is_file() else None}
            for name, path in inputs.items()
        },
        "outputs": {
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "receipt": {"path": str(receipt_path), "sha256": _sha256(receipt_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ContractReportResult(report_path, receipt_path, manifest_path, ok)


def parse_outputs(
    report_path: Path,
    receipt_path: Path,
    manifest_path: Path,
    result_path: Path | None = None,
) -> int:
    """Check the three report outputs that the stage command writes."""

    report_text = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    receipt = _load_object(receipt_path)
    manifest = _load_object(manifest_path)
    errors = []
    if not report_text.startswith("# Computational report\n"):
        errors.append(f"report output is invalid: {report_path}")
    if receipt is None or not isinstance(receipt.get("checks"), list):
        errors.append(f"receipt output is invalid: {receipt_path}")
    if manifest is None or manifest.get("format") != "claude-binder-declared-report-v1":
        errors.append(f"manifest output is invalid: {manifest_path}")
    if result_path is not None:
        parser_result = {
            "ok": not errors,
            "parsed_count": 3 if not errors else 0,
            "rejected_count": 0,
            "errors": errors,
            "source_output_hashes": (
                sorted(_sha256(path) for path in (report_path, receipt_path, manifest_path))
                if not errors
                else []
            ),
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(parser_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for error in errors:
        print(error, file=sys.stderr)
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser for the declared-input report adapter."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("toolcheck")
    run = subparsers.add_parser("run")
    run.add_argument("--target-manifest", type=Path, required=True)
    run.add_argument("--ranked-portfolio", type=Path, required=True)
    run.add_argument("--picture-manifest", type=Path, required=True)
    run.add_argument("--output-check", type=Path, required=True)
    run.add_argument("--report", type=Path, required=True)
    run.add_argument("--receipt", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    parse = subparsers.add_parser("parse")
    parse.add_argument("--report", type=Path, required=True)
    parse.add_argument("--receipt", type=Path, required=True)
    parse.add_argument("--manifest", type=Path, required=True)
    parse.add_argument("--result-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the declared-input report adapter."""

    args = build_parser().parse_args(argv)
    if args.command == "toolcheck":
        print("declared-input report adapter ok")
        return 0
    if args.command == "parse":
        return parse_outputs(args.report, args.receipt, args.manifest, args.result_path)
    result = build_contract_report(
        target_manifest_path=args.target_manifest,
        ranked_portfolio_path=args.ranked_portfolio,
        picture_manifest_path=args.picture_manifest,
        output_check_path=args.output_check,
        report_path=args.report,
        receipt_path=args.receipt,
        manifest_path=args.manifest,
    )
    print(f"Computational report built: {result.report_path}")
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
