#!/usr/bin/env python3
"""Report what every stage of a finished binder lane run produced.

Reads a run directory and prints one line per stage: whether it passed, how long
it took, what it made, how many of them, and where the file is. The executor
itself prints only "ok: True", so this is the readout a person needs to tell a
stage that did work from a stage that only reported success.

Everything printed here is read from the run directory. Nothing is recomputed
and nothing is inferred.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


# Every artifact type the run declares, mapped to the noun a person would use
# for it. The list came from the artifact types present in a completed run, so a
# type missing here is a type the pipeline gained after this was written; it
# falls back to its own name with the hyphens replaced.
ARTIFACT_NOUNS = {
    "backbone-candidate-manifest": "backbones",
    "candidate-manifest": "candidates",
    "control-observations": "control observations",
    "design-pose-files": "design poses",
    "eligible-parent-manifest": "eligible parents",
    "filter-observations": "filter observations",
    "next-round-decision": "decision",
    "normalized-candidate-manifest": "candidates",
    "optimization-score-table": "scored rows",
    "optimized-candidate-manifest": "candidates",
    "output-check": "output check",
    "passing-candidate-manifest": "passing candidates",
    "promotion-manifest": "promoted candidates",
    "ranked-portfolio": "ranked portfolio",
    "predicted-complex-files": "predicted complexes",
    "raw-prediction-manifest": "predictions",
    "rescore-candidate-manifest": "rescore candidates",
    "round-summary": "round summary",
    "runtime-report": "runtime report",
    "screen-score-table": "scored rows",
    "sequence-candidate-manifest": "candidates",
    "sequence-files": "sequences",
    "target-manifest": "targets",
    "uniform-observations": "observations",
}

STRUCTURE_KINDS = {"pdb", "cif", "mmcif", "structure"}
STRUCTURE_SUFFIXES = (".pdb", ".cif", ".mmcif")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def artifact_noun(artifact_type: str) -> str:
    return ARTIFACT_NOUNS.get(artifact_type, artifact_type.replace("-", " "))


def stage_durations(artifact_root: Path) -> dict[str, float]:
    """Seconds between the started and completed events for each stage."""
    progress_path = artifact_root / "stage-progress.jsonl"
    if not progress_path.is_file():
        return {}
    started: dict[str, datetime] = {}
    durations: dict[str, float] = {}
    for event in load_jsonl(progress_path):
        stage_id = event.get("stage_id")
        status = event.get("status")
        timestamp = event.get("timestamp")
        if not stage_id or not timestamp:
            continue
        if status == "started":
            started[stage_id] = parse_timestamp(timestamp)
        elif stage_id in started:
            durations[stage_id] = (parse_timestamp(timestamp) - started[stage_id]).total_seconds()
    return durations


def latest_attempt(stage_dir: Path) -> Path | None:
    """The attempt directory holding a stage receipt, most recent first."""
    attempts_root = stage_dir / "attempts"
    if not attempts_root.is_dir():
        return None
    attempts = [d for d in attempts_root.iterdir() if (d / "stage-receipt.json").is_file()]
    if not attempts:
        return None
    return max(attempts, key=lambda d: (d / "stage-receipt.json").stat().st_mtime)


def summarize_artifacts(manifest: dict[str, Any]) -> tuple[list[str], int, str | None]:
    """One phrase per artifact, the structure-file count, and the headline path.

    A stage that runs a smoke phase and then a scale phase declares the same
    artifact twice. For records only the widest phase is counted, because the
    smoke rows rehearse a subset of the scale rows and nothing downstream reads
    them. Structure files add across phases, because each phase writes its own
    and both sets are on disk.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for artifact in manifest.get("artifacts", []):
        artifact_id = artifact.get("artifact_id", "")
        records = sum(f.get("records") or 0 for f in artifact.get("files", []))
        files = len(artifact.get("files", []))
        previous = by_id.get(artifact_id)
        if artifact.get("kind", "") in STRUCTURE_KINDS:
            if previous is None:
                by_id[artifact_id] = {"artifact": artifact, "records": 0, "files": files}
            else:
                previous["files"] += files
            continue
        if previous is None or (records, files) >= (previous["records"], previous["files"]):
            by_id[artifact_id] = {
                "artifact": artifact,
                "records": records,
                "files": files,
            }

    phrases: list[str] = []
    structure_files = 0
    headline_path: str | None = None
    for entry in by_id.values():
        artifact = entry["artifact"]
        kind = artifact.get("kind", "")
        noun = artifact_noun(artifact.get("artifact_type", artifact.get("artifact_id", "")))
        if kind in STRUCTURE_KINDS:
            count = entry["files"]
            structure_files += count
        elif kind == "fasta":
            count = entry["files"]
        else:
            count = entry["records"] or entry["files"]
        phrases.append(f"{count} {noun}")
        if headline_path is None and artifact.get("publish_path"):
            headline_path = artifact["publish_path"]
    return phrases, structure_files, headline_path


def count_structures_on_disk(attempt_dir: Path) -> int:
    """Structure files under an attempt directory, declared or not.

    Counted from disk rather than from the manifest, so that a stage writing more
    structures than it declares is visible instead of silent. A run where the two
    agree prints no star and no note.
    """
    total = 0
    for path in attempt_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES:
            total += 1
    return total


def stage_headline(stage_id: str, artifact_root: Path) -> str | None:
    """A stage-specific number where a single value is the point of the stage."""
    try:
        if stage_id == "final-rank":
            ranked = load_json(artifact_root / "scores" / "ranked-candidates.json")
            candidates = ranked.get("ranked_candidates", [])
            if candidates:
                top = candidates[0]
                return f"top {top['candidate_id']} at {top['rank_score']:.4f}"
        if stage_id == "output-check":
            check = load_json(artifact_root / "validation" / "output-check.json")
            return f"{check.get('selected_count')} selected over {check.get('stage_count')} stages"
        if stage_id == "control-calibration":
            observations = load_jsonl(artifact_root / "controls" / "control-observations.jsonl")
            positive = sum(1 for r in observations if r.get("control_type") == "positive")
            negative = sum(1 for r in observations if r.get("control_type") == "negative")
            return f"{positive} positive, {negative} negative"
        if stage_id.startswith("optimization-plan-round-"):
            round_number = stage_id.rsplit("-", 1)[-1]
            decision = load_json(
                artifact_root / "optimization" / "rounds" / f"round-{round_number}" / "next-round-decision.json"
            )
            if decision.get("stop"):
                return f"stop: {decision.get('stop_reason')}"
            return f"continue with {len(decision.get('selected_parent_ids', []))} parents"
    except (OSError, ValueError, KeyError):
        return None
    return None


def filter_headline(manifest: dict[str, Any]) -> str | None:
    """Pass count over observation count, read from the filter's own records."""
    observation_path: Path | None = None
    for artifact in manifest.get("artifacts", []):
        if artifact.get("artifact_type", "").endswith("-filter-observations"):
            published = artifact.get("published_path")
            if published:
                observation_path = Path(published)
    if observation_path is None or not observation_path.is_file():
        return None
    observations = load_jsonl(observation_path)
    if not observations:
        return None
    passed = sum(1 for r in observations if r.get("pass") is True)
    return f"{passed}/{len(observations)} passed"


def collect(run_dir: Path) -> dict[str, Any]:
    artifact_root = run_dir / "artifacts"
    status = load_json(run_dir / "status.json")
    durations = stage_durations(artifact_root)
    receipt_ok: dict[str, bool] = {}
    rows: list[dict[str, Any]] = []

    for stage_id in status.get("completed_stages", []):
        stage_dir = artifact_root / "stages" / stage_id
        attempt = latest_attempt(stage_dir)
        row: dict[str, Any] = {
            "stage_id": stage_id,
            "ok": None,
            "seconds": durations.get(stage_id),
            "produced": [],
            "structures_declared": 0,
            "structures_on_disk": 0,
            "path": None,
            "headline": stage_headline(stage_id, artifact_root),
            "errors": [],
        }
        if attempt is not None:
            receipt = load_json(attempt / "stage-receipt.json")
            row["ok"] = receipt.get("ok")
            row["errors"] = receipt.get("errors", [])
            row["adapter_id"] = receipt.get("adapter_id")
            row["attempt_dir"] = str(attempt)
            row["structures_on_disk"] = count_structures_on_disk(attempt)
            manifest_path = attempt / "stage-output-manifest.json"
            if manifest_path.is_file():
                manifest = load_json(manifest_path)
                phrases, structures, published = summarize_artifacts(manifest)
                row["produced"] = phrases
                row["structures_declared"] = structures
                row["path"] = published or str(attempt.relative_to(artifact_root))
                if row["headline"] is None and stage_id.startswith(("filter-", "optimization-filter-")):
                    row["headline"] = filter_headline(manifest)
        receipt_ok[stage_id] = bool(row["ok"])
        rows.append(row)

    return {
        "run_id": status.get("run_id"),
        "state": status.get("state"),
        "ok": status.get("ok"),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "stage_count": len(rows),
        "stages_ok": sum(1 for v in receipt_ok.values() if v),
        "run_seconds": (
            (parse_timestamp(status["finished_at"]) - parse_timestamp(status["started_at"])).total_seconds()
            if status.get("started_at") and status.get("finished_at")
            else None
        ),
        "stages": rows,
    }


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    seconds = report["run_seconds"]
    duration = f"{seconds:.1f}s" if seconds is not None else "unknown"
    lines.append(
        f"run {report['run_id']}  {report['state']}  "
        f"{report['stages_ok']}/{report['stage_count']} stages ok  {duration}"
    )
    lines.append("")
    lines.append(f"{'#':>3}  {'stage':<42}{'':<4}{'secs':>6}  {'open':>6}  produced")
    pad = f"{'':>3}  {'':<42}{'':<4}{'':>6}  {'':>6}  "
    undeclared = 0
    for index, row in enumerate(report["stages"], start=1):
        mark = "ok " if row["ok"] else "FAIL"
        secs = f"{row['seconds']:.2f}" if row["seconds"] is not None else "-"
        on_disk = row["structures_on_disk"]
        declared = row["structures_declared"]
        if not on_disk:
            openable = "-"
        elif on_disk == declared:
            openable = str(on_disk)
        else:
            openable = f"{on_disk}*"
            undeclared += on_disk - declared
        produced = ", ".join(row["produced"]) or "nothing declared"
        if row["headline"]:
            produced = f"{produced}  [{row['headline']}]"
        lines.append(
            f"{index:>3}  {row['stage_id']:<42}{mark:<4}{secs:>6}  {openable:>6}  {produced}"
        )
        if row["path"]:
            lines.append(f"{pad}-> {row['path']}")
        for error in row["errors"]:
            lines.append(f"{pad}!! {error}")
    lines.append("")
    lines.append("open is the number of structure files under the stage, so a row with a number")
    lines.append("there has something you can put on screen.")
    if undeclared:
        lines.append(
            f"A star marks a stage whose output manifest declares fewer structures than it "
            f"wrote. {undeclared} structure files across the run are on disk but undeclared."
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="a finished run directory")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not (run_dir / "status.json").is_file():
        print(f"error: {run_dir} has no status.json, so it is not a run directory")
        return 1

    report = collect(run_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))
    return 0 if report["stages_ok"] == report["stage_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
