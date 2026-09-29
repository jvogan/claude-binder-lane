#!/usr/bin/env python3
"""Reduce matched counter-screen observations without an effect-size calculation.

The published baseline ranks a counter-screened target with the per-arm delta
between on-target and off-target-paralog ipSAE_min. Each side uses the same
seed set and the same maximum-over-seeds aggregation. A score at the exact
zero floor represents a result below the metric's 10 angstrom aligned-error
cutoff. This adapter records that censoring flag beside every raw value.

The adapter only reads local JSONL observations. It never launches a fold or
contacts a provider.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METRIC = "ipsae_min"
METRIC_FLOOR = 0.0
SEED_AGGREGATION = "max"
ALIGNED_ERROR_CUTOFF_ANGSTROM = 10.0

SEPARATED_CLEAN = "separated-clean"
SEPARATED_DETECTED = "separated-with-detected-off-target"
CONTESTED = "contested"
UNINFORMATIVE = "uninformative"
INCOMPLETE = "incomplete"


class CounterScreenError(RuntimeError):
    """An observation mismatch that makes a selectivity subtraction invalid."""


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CounterScreenError(f"{label} must be a non-empty string")
    return value


def _require_seed(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CounterScreenError(f"{label} must be an integer")
    return value


def _require_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CounterScreenError(f"{label} must be a finite number")
    score = float(value)
    if not math.isfinite(score):
        raise CounterScreenError(f"{label} must be a finite number")
    return score


def _require_published_metric(metric: str, metric_floor: float, aggregation: str) -> None:
    if metric != METRIC:
        raise CounterScreenError(f"counter-screen metric must be {METRIC!r}")
    if (
        isinstance(metric_floor, bool)
        or not isinstance(metric_floor, (int, float))
        or not math.isfinite(float(metric_floor))
        or float(metric_floor) != METRIC_FLOOR
    ):
        raise CounterScreenError(f"counter-screen metric floor must be {METRIC_FLOOR:g}")
    if aggregation != SEED_AGGREGATION:
        raise CounterScreenError(
            f"counter-screen seed aggregation must be {SEED_AGGREGATION!r}"
        )


def _candidate_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep candidate observations and reject a failed candidate measurement."""
    candidates: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise CounterScreenError(f"observation {index} must be an object")
        control_type = row.get("control_type", "candidate")
        if control_type != "candidate":
            continue
        if row.get("status") != "scored":
            candidate_id = row.get("candidate_id", "<unknown>")
            raise CounterScreenError(
                f"candidate {candidate_id!r} observation {index} has status {row.get('status')!r}"
            )
        candidates.append(row)
    return candidates


def _group_by_candidate_predictor(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric: str,
    side: str,
) -> dict[tuple[str, str], dict[int, float]]:
    grouped: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for index, row in enumerate(_candidate_rows(rows)):
        candidate_id = _require_text(row.get("candidate_id"), f"{side} observation {index}.candidate_id")
        predictor = _require_text(row.get("predictor"), f"{side} observation {index}.predictor")
        seed = _require_seed(row.get("seed"), f"{side} observation {index}.seed")
        score = _require_score(row.get(metric), f"{side} observation {index}.{metric}")
        key = candidate_id, predictor
        if seed in grouped[key]:
            raise CounterScreenError(
                f"{side} observations repeat candidate {candidate_id!r}, predictor {predictor!r}, seed {seed}"
            )
        grouped[key][seed] = score
    if not grouped:
        raise CounterScreenError(f"{side} observations contain no scored candidates")
    return dict(grouped)


def _raw_score(value: float, seed: int, metric_floor: float) -> dict[str, Any]:
    return {
        "seed": seed,
        "value": value,
        "at_metric_floor": value == metric_floor,
    }


def _aggregate(scores_by_seed: Mapping[int, float], *, metric_floor: float) -> dict[str, Any]:
    if not scores_by_seed:
        raise CounterScreenError("cannot aggregate an empty seed set")
    selected_seed, value = max(scores_by_seed.items(), key=lambda item: (item[1], -item[0]))
    return {
        "value": value,
        "at_metric_floor": value == metric_floor,
        "argmax_seed": selected_seed,
        "seed_count": len(scores_by_seed),
        "seed_aggregation": SEED_AGGREGATION,
        "per_seed": [
            _raw_score(score, seed, metric_floor)
            for seed, score in sorted(scores_by_seed.items())
        ],
    }


def _selectivity_status(on_target_at_floor: bool, off_target_at_floor: bool, delta: float | None) -> str:
    if on_target_at_floor:
        return UNINFORMATIVE
    if off_target_at_floor:
        return SEPARATED_CLEAN
    assert delta is not None
    if delta > 0.0:
        return SEPARATED_DETECTED
    return CONTESTED


def reduce_counter_screen(
    on_target_rows: Sequence[Mapping[str, Any]],
    off_target_rows: Sequence[Mapping[str, Any]],
    *,
    on_target_id: str,
    off_target_paralog_id: str,
    metric: str = METRIC,
    metric_floor: float = METRIC_FLOOR,
    aggregation: str = SEED_AGGREGATION,
) -> list[dict[str, Any]]:
    """Return one per-candidate, per-predictor matched counter-screen result.

    The function rejects a missing candidate, a missing predictor, duplicate
    seeds, or mismatched seed identifiers. It returns a null delta when the
    on-target aggregate is at the metric floor. It never converts a censored
    score into an effect size.
    """
    on_target_id = _require_text(on_target_id, "on_target_id")
    off_target_paralog_id = _require_text(off_target_paralog_id, "off_target_paralog_id")
    if on_target_id == off_target_paralog_id:
        raise CounterScreenError("counter-screen on-target and off-target IDs must differ")
    _require_published_metric(metric, metric_floor, aggregation)
    on_groups = _group_by_candidate_predictor(on_target_rows, metric=metric, side="on-target")
    off_groups = _group_by_candidate_predictor(off_target_rows, metric=metric, side="off-target")
    if set(on_groups) != set(off_groups):
        missing_off_target = sorted(set(on_groups) - set(off_groups))
        missing_on_target = sorted(set(off_groups) - set(on_groups))
        details: list[str] = []
        if missing_off_target:
            details.append("missing off-target observations for " + repr(missing_off_target))
        if missing_on_target:
            details.append("missing on-target observations for " + repr(missing_on_target))
        raise CounterScreenError("counter-screen candidate coverage differs: " + "; ".join(details))

    results: list[dict[str, Any]] = []
    for candidate_id, predictor in sorted(on_groups):
        on_scores = on_groups[(candidate_id, predictor)]
        off_scores = off_groups[(candidate_id, predictor)]
        if set(on_scores) != set(off_scores):
            raise CounterScreenError(
                f"counter-screen seed set differs for candidate {candidate_id!r}, predictor {predictor!r}: "
                f"on-target {sorted(on_scores)}, off-target {sorted(off_scores)}"
            )
        on_target = _aggregate(on_scores, metric_floor=metric_floor)
        off_target = _aggregate(off_scores, metric_floor=metric_floor)
        on_value = float(on_target["value"])
        off_value = float(off_target["value"])
        on_target_at_floor = bool(on_target["at_metric_floor"])
        off_target_at_floor = bool(off_target["at_metric_floor"])
        delta = None if on_target_at_floor else on_value - off_value
        status = _selectivity_status(on_target_at_floor, off_target_at_floor, delta)
        results.append(
            {
                "candidate_id": candidate_id,
                "predictor": predictor,
                "metric": metric,
                "metric_floor": metric_floor,
                "aligned_error_cutoff_angstrom": ALIGNED_ERROR_CUTOFF_ANGSTROM,
                "on_target_id": on_target_id,
                "off_target_paralog_id": off_target_paralog_id,
                "seed_count": len(on_scores),
                "on_target_seed_count": len(on_scores),
                "off_target_seed_count": len(off_scores),
                "seed_aggregation": aggregation,
                "on_target_ipsae_min": on_value,
                "on_target_at_metric_floor": on_target_at_floor,
                "off_target_ipsae_min": off_value,
                "off_target_at_metric_floor": off_target_at_floor,
                "selectivity_delta": delta,
                "selectivity_delta_is_lower_bound": off_target_at_floor and not on_target_at_floor,
                "selectivity_status": status,
                "rank_eligible": status != UNINFORMATIVE,
                "on_target": on_target,
                "off_target": off_target,
            }
        )
    return results


def assess_counter_panel(
    records: Sequence[Mapping[str, Any]],
    required_off_target_ids: Sequence[str],
) -> dict[str, Any]:
    """Report a panel gate without turning panel scores into extra rank terms.

    A detected result for any configured panel target routes the candidate to
    the bottom of the ranked set. An omitted panel target leaves the candidate
    unranked because the gate has not run.
    """
    required = tuple(required_off_target_ids)
    if not required or any(not isinstance(target_id, str) or not target_id for target_id in required):
        raise CounterScreenError("counter-screen panel target IDs must be non-empty strings")
    if len(set(required)) != len(required):
        raise CounterScreenError("counter-screen panel target IDs must be unique")
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise CounterScreenError(f"counter-screen panel record {index} must be an object")
        target_id = _require_text(record.get("off_target_paralog_id"), f"counter-screen panel record {index}.off_target_paralog_id")
        by_target[target_id].append(record)
    missing = [target_id for target_id in required if not by_target.get(target_id)]
    unexpected = sorted(set(by_target) - set(required))
    if missing or unexpected:
        return {
            "status": INCOMPLETE,
            "rank_bucket": 2,
            "rank_disposition": "unranked",
            "missing_off_target_ids": missing,
            "unexpected_off_target_ids": unexpected,
            "detected_off_target_ids": [],
        }
    detected: list[str] = []
    for target_id in required:
        target_records = by_target[target_id]
        if any(
            record.get("selectivity_status")
            not in {SEPARATED_CLEAN, SEPARATED_DETECTED, CONTESTED, UNINFORMATIVE}
            or not isinstance(record.get("off_target_at_metric_floor"), bool)
            for record in target_records
        ):
            return {
                "status": INCOMPLETE,
                "rank_bucket": 2,
                "rank_disposition": "unranked",
                "missing_off_target_ids": [],
                "unexpected_off_target_ids": [],
                "detected_off_target_ids": [],
            }
        if any(record.get("selectivity_status") == UNINFORMATIVE for record in target_records):
            return {
                "status": UNINFORMATIVE,
                "rank_bucket": 2,
                "rank_disposition": "unranked",
                "missing_off_target_ids": [],
                "unexpected_off_target_ids": [],
                "detected_off_target_ids": [],
            }
        if any(record.get("off_target_at_metric_floor") is not True for record in target_records):
            detected.append(target_id)
    if detected:
        return {
            "status": SEPARATED_DETECTED,
            "rank_bucket": 1,
            "rank_disposition": "ranked-at-bottom",
            "missing_off_target_ids": [],
            "unexpected_off_target_ids": [],
            "detected_off_target_ids": detected,
        }
    return {
        "status": SEPARATED_CLEAN,
        "rank_bucket": 0,
        "rank_disposition": "ranked",
        "missing_off_target_ids": [],
        "unexpected_off_target_ids": [],
        "detected_off_target_ids": [],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CounterScreenError(f"observations file is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CounterScreenError(
                f"observations file has invalid JSON at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise CounterScreenError(f"observations file line {line_number} must be an object")
        rows.append(row)
    if not rows:
        raise CounterScreenError(f"observations file has no rows: {path}")
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    os.replace(temporary_path, path)


def _rows_for_target(rows: Iterable[Mapping[str, Any]], target_id: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("target_id") == target_id]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--on-target-id", required=True)
    parser.add_argument("--off-target-paralog-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", default=METRIC)
    parser.add_argument("--seed-aggregation", default=SEED_AGGREGATION)
    args = parser.parse_args(argv)
    try:
        observations = _read_jsonl(args.observations)
        result = reduce_counter_screen(
            _rows_for_target(observations, args.on_target_id),
            _rows_for_target(observations, args.off_target_paralog_id),
            on_target_id=args.on_target_id,
            off_target_paralog_id=args.off_target_paralog_id,
            metric=args.metric,
            aggregation=args.seed_aggregation,
        )
        _write_jsonl(args.out, result)
    except CounterScreenError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
