"""Select SCREEN survivors for the five-seed INTERMEDIATE tier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from claude_binder import lane
from .declared_artifacts import input_files, load_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("toolcheck")
    for command in ("run", "parse"):
        sub = subparsers.add_parser(command)
        for name in ("stage", "phase"):
            sub.add_argument(f"--{name}", required=True)
        sub.add_argument("--count", type=int, default=1)
        for name in ("attempt-dir", "receipts-dir", "artifact-root", "config", "plan"):
            sub.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "toolcheck":
        print("screen-survivor-selector ok")
        return 0
    return run(args) if args.command == "run" else parse(args)


def output_path(args: argparse.Namespace) -> Path:
    return args.attempt_dir / args.phase / "intermediate-candidates.jsonl"


def run(args: argparse.Namespace) -> int:
    config = lane.load_json(args.config)
    plan = load_plan(args.plan, config)
    _, score_files = input_files(plan, args.receipts_dir, args.stage, artifact_id="screen-score-table")
    _, passing_files = input_files(plan, args.receipts_dir, args.stage, artifact_id="passing-candidates")
    if len(score_files) != 1 or len(passing_files) != 1:
        raise ValueError("screen-survivors requires one score table and one passing manifest")
    validation = lane.validate_screen_scored_pool(config, score_files[0], args.artifact_root)
    if not validation["ok"]:
        raise ValueError("SCREEN pool is invalid: " + "; ".join(validation["errors"][:8]))
    if not (lane.is_ungated_candidate_claim(config) and lane.control_panel_is_disabled(config)):
        controls = lane.validate_control_calibration(config, args.artifact_root)
        if not controls["ok"]:
            raise ValueError("control calibration failed before INTERMEDIATE: " + "; ".join(controls["errors"][:8]))
    scores = lane.load_jsonl(score_files[0])
    passing = lane.load_jsonl(passing_files[0])
    passing_by_id = {str(row["candidate_id"]): row for row in passing}
    ranked = lane.rank_candidate_cohort(config, scores, list(config["cofold"]["screen_seeds"]))
    ranked = lane.apply_declared_ranking_mode(config, ranked, complete_only=True)
    ranked.sort(key=lambda row: lane._rank_sort_key(row, config))
    fraction = config["cofold"].get("intermediate_fraction", 0.2)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError("cofold.intermediate_fraction must be in (0, 1]")
    parent_count = int(config["optimization"]["parent_count_per_round"])
    limit = min(len(passing), max(parent_count, math.ceil(len(passing) * fraction)))
    eligible = [
        row for row in ranked
        if row.get("eligible") is True and str(row["candidate_id"]) in passing_by_id
    ]
    minimum_generators = min(
        max(int(config["generation"]["minimum_generators"]),
            int(config["selection"]["minimum_generators"])),
        parent_count,
    )
    anchor_rows, portfolio = lane.select_portfolio(
        eligible,
        final_count=parent_count,
        minimum_generators=minimum_generators,
        maximum_fraction=float(config["selection"]["maximum_fraction_per_generator"]),
        sort_key=lambda row: lane._rank_sort_key(row, config),
    )
    if not portfolio["ok"]:
        raise ValueError("SCREEN cannot supply a parent slate under generator diversity and fraction limits")
    chosen_ids = {str(row["candidate_id"]) for row in anchor_rows}
    for score in eligible:
        if len(chosen_ids) >= limit:
            break
        chosen_ids.add(str(score["candidate_id"]))
    chosen: list[str] = []
    for score in ranked:
        candidate_id = str(score["candidate_id"])
        if candidate_id in chosen_ids and candidate_id not in chosen:
            chosen.append(candidate_id)
    path = output_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, candidate_id in enumerate(chosen, 1):
        row = dict(passing_by_id[candidate_id])
        row["screen_survivor_rank"] = rank
        row["screen_survivor_fraction"] = fraction
        row["screen_survivor_pool_count"] = len(passing)
        row["screen_survivor_selected_count"] = len(chosen)
        rows.append(row)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return 0


def parse(args: argparse.Namespace) -> int:
    path = output_path(args)
    rows = lane.load_jsonl(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "ok": bool(rows), "parsed_count": len(rows), "rejected_count": 0,
        "errors": [], "source_output_hashes": [digest],
    }
    result_path = args.attempt_dir / args.phase / "parser-result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
