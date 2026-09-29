"""Compare configured best-seed ranking with median and mean seed reducers.

The diagnostic accepts a resolved campaign configuration and its uniform-rescore
observation table. It leaves production ranking code unchanged. For each
selection rule, it makes the selected per-predictor metric values constant over
the configured seeds, then calls :func:`claude_binder.lane.rank_candidates`.
The production code therefore performs validation, z-score normalization,
ranking, and tie-breaking for every ordering. It derives ``score_instrument``
from ``predictor`` before validation so that a legacy arm label does not block
the current production ranking path.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from claude_binder import lane


SELECTION_RULES = ("maximum", "median", "mean")
PRIMARY_METRIC = "ipsae_min"
POSE_METRIC = "sc_dockq"


class DiagnosticError(ValueError):
    """The supplied scored-run artifact cannot support this diagnostic."""


def _finite_metric(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _configured_predictors(config: dict[str, Any]) -> list[str]:
    cofold = config.get("cofold")
    if not isinstance(cofold, dict):
        raise DiagnosticError("config.cofold must be an object")
    predictors = cofold.get("predictors")
    if not isinstance(predictors, list):
        raise DiagnosticError("config.cofold.predictors must be a list")
    predictor_ids = [
        str(record["id"])
        for record in predictors
        if isinstance(record, dict) and record.get("enabled", True) is True and isinstance(record.get("id"), str)
    ]
    if not predictor_ids or len(predictor_ids) != len(set(predictor_ids)):
        raise DiagnosticError("config must define unique enabled predictor IDs")
    return predictor_ids


def _configured_seeds(config: dict[str, Any]) -> list[int]:
    cofold = config.get("cofold")
    seeds = cofold.get("rescore_seeds") if isinstance(cofold, dict) else None
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise DiagnosticError("config.cofold.rescore_seeds must be a non-empty unique integer list")
    return sorted(seeds)


def _ranking_metrics(config: dict[str, Any]) -> tuple[str, str]:
    scoring = config.get("scoring")
    if not isinstance(scoring, dict):
        raise DiagnosticError("config.scoring must be an object")
    primary_metric = scoring.get("primary_metric")
    pose_metric = scoring.get("pose_metric")
    if (primary_metric, pose_metric) != (PRIMARY_METRIC, POSE_METRIC):
        raise DiagnosticError(
            "selection-rule diagnostic requires scoring.primary_metric=ipsae_min "
            "and scoring.pose_metric=sc_dockq"
        )
    return primary_metric, pose_metric


def _candidate_rows(observations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in observations if row.get("control_type", "candidate") == "candidate"]


def _candidate_groups(
    observations: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _candidate_rows(observations):
        for field in ("target_id", "candidate_id", "predictor"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise DiagnosticError(f"candidate observation has no {field}")
        groups[(row["target_id"], row["candidate_id"], row["predictor"])].append(row)
    if not groups:
        raise DiagnosticError("observations contain no candidate rows")
    return groups


def _completed_seed_rows(
    rows: Iterable[dict[str, Any]],
    primary_metric: str,
    pose_metric: str,
) -> list[dict[str, Any]]:
    completed = []
    for row in rows:
        if row.get("status") != "scored":
            continue
        if _finite_metric(row, primary_metric) is None or _finite_metric(row, pose_metric) is None:
            continue
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            continue
        completed.append(row)
    return completed


def _group_completion(
    rows: list[dict[str, Any]],
    configured_seeds: list[int],
    primary_metric: str,
    pose_metric: str,
) -> dict[str, Any]:
    observed_seeds = sorted(
        {row["seed"] for row in rows if isinstance(row.get("seed"), int) and not isinstance(row.get("seed"), bool)}
    )
    completed = _completed_seed_rows(rows, primary_metric, pose_metric)
    completed_seeds = sorted({row["seed"] for row in completed})
    expected = set(configured_seeds)
    expected_completed_rows = [row for row in completed if row["seed"] in expected]
    complete = (
        observed_seeds == configured_seeds
        and completed_seeds == configured_seeds
        and len(expected_completed_rows) == len(configured_seeds)
        and len(rows) == len(configured_seeds)
    )
    return {
        "configured_seed_count": len(configured_seeds),
        "observed_seed_count": len(observed_seeds),
        "completed_seed_count": len(completed_seeds),
        "observed_seeds": observed_seeds,
        "completed_seeds": completed_seeds,
        "missing_configured_seeds": sorted(expected - set(observed_seeds)),
        "unexpected_seeds": sorted(set(observed_seeds) - expected),
        "complete": complete,
    }


def completion_table(config: dict[str, Any], observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return seed and predictor-fold completion for every candidate and target."""
    predictor_ids = _configured_predictors(config)
    configured_seeds = _configured_seeds(config)
    primary_metric, pose_metric = _ranking_metrics(config)
    groups = _candidate_groups(observations)
    candidates = sorted({(target_id, candidate_id) for target_id, candidate_id, _ in groups})
    table = []
    for target_id, candidate_id in candidates:
        predictors = []
        for predictor_id in predictor_ids:
            completion = _group_completion(
                groups.get((target_id, candidate_id, predictor_id), []),
                configured_seeds,
                primary_metric,
                pose_metric,
            )
            predictors.append({"predictor_id": predictor_id, **completion})
        table.append(
            {
                "target_id": target_id,
                "candidate_id": candidate_id,
                "configured_predictor_folds": len(predictor_ids),
                "completed_predictor_folds": sum(record["complete"] for record in predictors),
                "predictors": predictors,
            }
        )
    return table


def _selection_summary(
    rows: list[dict[str, Any]],
    rule: str,
    configured_seeds: list[int],
    primary_metric: str,
    pose_metric: str,
) -> dict[str, Any] | None:
    completion = _group_completion(rows, configured_seeds, primary_metric, pose_metric)
    if not completion["complete"]:
        return None
    completed = _completed_seed_rows(rows, primary_metric, pose_metric)
    if rule == "maximum":
        selected = max(completed, key=lambda row: (float(row[primary_metric]), -int(row["seed"])))
        return {
            "selected_seed": int(selected["seed"]),
            primary_metric: float(selected[primary_metric]),
            pose_metric: float(selected[pose_metric]),
            "source_row": selected,
        }
    if rule == "median":
        selected = sorted(completed, key=lambda row: (float(row[primary_metric]), int(row["seed"])))[
            (len(completed) - 1) // 2
        ]
        return {
            "selected_seed": int(selected["seed"]),
            primary_metric: float(selected[primary_metric]),
            pose_metric: float(selected[pose_metric]),
            "source_row": selected,
        }
    if rule == "mean":
        source = max(completed, key=lambda row: (float(row[primary_metric]), -int(row["seed"])))
        return {
            "selected_seed": None,
            primary_metric: sum(float(row[primary_metric]) for row in completed) / len(completed),
            pose_metric: sum(float(row[pose_metric]) for row in completed) / len(completed),
            "source_row": source,
        }
    raise DiagnosticError(f"unknown selection rule: {rule}")


def _selection_summaries(
    config: dict[str, Any], observations: list[dict[str, Any]], rule: str
) -> dict[tuple[str, str, str], dict[str, Any]]:
    if rule not in SELECTION_RULES:
        raise DiagnosticError(f"unknown selection rule: {rule}")
    configured_seeds = _configured_seeds(config)
    primary_metric, pose_metric = _ranking_metrics(config)
    return {
        key: summary
        for key, rows in _candidate_groups(observations).items()
        if (summary := _selection_summary(rows, rule, configured_seeds, primary_metric, pose_metric)) is not None
    }


def _set_selected_metrics(row: dict[str, Any], primary_value: float, pose_value: float) -> None:
    row[PRIMARY_METRIC] = primary_value
    row[POSE_METRIC] = pose_value
    row["ipsae_target_to_binder"] = primary_value
    row["ipsae_binder_to_target"] = primary_value
    measurement = row.get("measurement")
    if isinstance(measurement, dict):
        measurement[PRIMARY_METRIC] = primary_value
        measurement[POSE_METRIC] = pose_value
        measurement["ipsae_target_to_binder"] = primary_value
        measurement["ipsae_binder_to_target"] = primary_value


def _observations_with_rule(
    config: dict[str, Any], observations: list[dict[str, Any]], rule: str
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    summaries = _selection_summaries(config, observations, rule)
    transformed: list[dict[str, Any]] = []
    for row in observations:
        if row.get("control_type", "candidate") != "candidate":
            normalized = copy.deepcopy(row)
            _normalize_score_instrument(normalized)
            transformed.append(normalized)
            continue
        key = (str(row.get("target_id")), str(row.get("candidate_id")), str(row.get("predictor")))
        summary = summaries.get(key)
        if summary is None:
            normalized = copy.deepcopy(row)
            _normalize_score_instrument(normalized)
            transformed.append(normalized)
            continue
        replacement = copy.deepcopy(summary["source_row"])
        replacement["seed"] = row["seed"]
        _set_selected_metrics(replacement, summary[PRIMARY_METRIC], summary[POSE_METRIC])
        _normalize_score_instrument(replacement)
        transformed.append(replacement)
    return transformed, summaries


def _normalize_score_instrument(row: dict[str, Any]) -> None:
    predictor = row.get("predictor")
    if isinstance(predictor, str) and predictor:
        row["score_instrument"] = lane.score_instrument_arm_name(predictor)


def _display_selection(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_seed": summary["selected_seed"],
        PRIMARY_METRIC: summary[PRIMARY_METRIC],
        POSE_METRIC: summary[POSE_METRIC],
    }


def _compact_ranking(
    result: dict[str, Any],
    summaries: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    rows = result.get("ranked_candidates", [])
    compact_rows = []
    for row in rows if isinstance(rows, list) else []:
        target_id = str(row.get("target_id"))
        candidate_id = str(row.get("candidate_id"))
        selection_by_predictor = {
            predictor_id: _display_selection(summary)
            for (summary_target, summary_candidate, predictor_id), summary in summaries.items()
            if (summary_target, summary_candidate) == (target_id, candidate_id)
        }
        compact_rows.append(
            {
                "rank": row.get("rank"),
                "target_id": target_id,
                "candidate_id": candidate_id,
                "rank_score": row.get("rank_score"),
                "filter_pass": row.get("filter_pass"),
                "coverage_complete": row.get("coverage_complete"),
                "selection_by_predictor": selection_by_predictor,
            }
        )
    cohort = [
        row
        for row in compact_rows
        if row["coverage_complete"] is True and row["filter_pass"] is True and isinstance(row["rank_score"], (int, float))
    ]
    return {
        "production_result_ok": result.get("ok") is True,
        "errors": result.get("errors", []),
        "candidate_count": result.get("candidate_count", 0),
        "ranking_cohort_candidate_ids": [row["candidate_id"] for row in cohort],
        "ranked_candidates": compact_rows,
    }


def _average_ranks(rows: list[dict[str, Any]]) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: (-float(row["rank_score"]), str(row["candidate_id"])))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        score = float(ordered[index]["rank_score"])
        while end < len(ordered) and float(ordered[end]["rank_score"]) == score:
            end += 1
        average = (index + 1 + end) / 2
        for row in ordered[index:end]:
            ranks[str(row["candidate_id"])] = average
        index = end
    return ranks


def _spearman_rho(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float | None:
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if set(left_ranks) != set(right_ranks) or len(left_ranks) < 2:
        return None
    left_mean = sum(left_ranks.values()) / len(left_ranks)
    right_mean = sum(right_ranks.values()) / len(right_ranks)
    numerator = sum((left_ranks[key] - left_mean) * (right_ranks[key] - right_mean) for key in left_ranks)
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks.values()))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks.values()))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def diagnose(config: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Return production-ranked orderings and seed and fold completion data."""
    predictor_ids = _configured_predictors(config)
    configured_seeds = _configured_seeds(config)
    primary_metric, pose_metric = _ranking_metrics(config)
    rankings: dict[str, dict[str, Any]] = {}
    for rule in SELECTION_RULES:
        transformed, summaries = _observations_with_rule(config, observations, rule)
        result = lane.rank_candidates(copy.deepcopy(config), transformed)
        rankings[rule] = _compact_ranking(result, summaries)
    cohort_rows = {
        rule: [
            row
            for row in ranking["ranked_candidates"]
            if row["coverage_complete"] is True and row["filter_pass"] is True and isinstance(row["rank_score"], (int, float))
        ]
        for rule, ranking in rankings.items()
    }
    correlations = {
        "maximum_vs_median": _spearman_rho(cohort_rows["maximum"], cohort_rows["median"]),
        "maximum_vs_mean": _spearman_rho(cohort_rows["maximum"], cohort_rows["mean"]),
        "median_vs_mean": _spearman_rho(cohort_rows["median"], cohort_rows["mean"]),
    }
    return {
        "selection_rules": {
            "maximum": "maximum ipsae_min and paired sc_dockq from the same seed",
            "median": "lower median ipsae_min seed and paired sc_dockq from that seed",
            "mean": "mean ipsae_min and mean sc_dockq across completed seeds",
        },
        "primary_metric": primary_metric,
        "pose_metric": pose_metric,
        "configured_predictor_ids": predictor_ids,
        "configured_rescore_seeds": configured_seeds,
        "completion": completion_table(config, observations),
        "rankings": rankings,
        "rank_correlations": correlations,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DiagnosticError(f"cannot read config: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DiagnosticError(f"config is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"config must be a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DiagnosticError(f"cannot read observations: {path}") from exc
    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DiagnosticError(f"observation is not valid JSON at line {line_number}: {path}") from exc
        if not isinstance(value, dict):
            raise DiagnosticError(f"observation must be a JSON object at line {line_number}: {path}")
        rows.append(value)
    if not rows:
        raise DiagnosticError(f"observations have no rows: {path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = diagnose(_load_json(args.config), _load_jsonl(args.observations))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
