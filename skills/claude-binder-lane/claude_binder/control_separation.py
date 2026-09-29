"""Compute and enforce stored control separation for qualified scoring arms.

The statistic is Hedges' g on ``ipsae_min``. It is a standardized mean
difference, so a linear change of score scale does not change the value. The
module accepts already measured control values and never starts a provider.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
STATISTIC = "hedges_g"
METRIC = "ipsae_min"
DIRECTION = "higher_positive"
DEFAULT_THRESHOLD: float | None = None
RAW_SCORE_STATISTIC = "raw_scores_with_floor_flags"
FLOOR_ROBUST_STATISTIC = "floor_robust_separation"
METRIC_FLOORS = {METRIC: 0.0}
RANK_STATISTIC = "rank_score"
RANK_DIRECTION = "matched_control_strictly_outranks_mismatched_control"
MATCHED_CONTROL_ROLE = "known-same-site-complex"
MISMATCHED_CONTROL_ROLE = "matched-wrong-pair"
# Calibration currently covers one target and fifteen control observations.
# That is insufficient for a confident universal gate, so the default stays
# unset. Missing separation remains blocking for production scoring.


def _finite_values(values: Any, label: str) -> tuple[list[float], str | None]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return [], f"{label} must be a numeric list"
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return [], f"{label}[{index}] must be a finite number"
        result.append(float(value))
    return result, None


def _control_values(control: Mapping[str, Any], label: str) -> tuple[list[float], str | None]:
    for key in ("values", "score_by_seed", "score_by_sample"):
        if key not in control:
            continue
        value = control[key]
        if key in {"score_by_seed", "score_by_sample"}:
            if not isinstance(value, list):
                return [], f"{label}.{key} must be a list"
            value = [
                item.get("value")
                for item in value
                if isinstance(item, Mapping)
            ]
        return _finite_values(value, f"{label}.{key}")
    return [], f"{label} has no values"


def _sample_variance(values: Sequence[float]) -> float:
    mean = math.fsum(values) / len(values)
    return math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _zero_variance_groups(
    positive_values: Sequence[float], negative_values: Sequence[float]
) -> list[str]:
    return [
        label
        for label, values in (("positive", positive_values), ("negative", negative_values))
        if len(values) >= 2 and _sample_variance(values) == 0.0
    ]


def report_raw_scores_with_floor_flags(
    positive: Sequence[float],
    negative: Sequence[float],
    *,
    metric: str = METRIC,
    metric_floor: Any = None,
) -> dict[str, Any]:
    """Record raw control scores and the metric-floor flag for every value.

    ipSAE_min returns an exact zero below its aligned-error cutoff. A constant
    floor control cannot supply a denominator for an effect size, so this
    report preserves raw observations instead of standardizing them.
    """
    positive_values, positive_error = _finite_values(positive, "positive")
    negative_values, negative_error = _finite_values(negative, "negative")
    if positive_error or negative_error:
        raise ValueError(positive_error or negative_error)
    if not positive_values or not negative_values:
        raise ValueError("raw score reporting requires positive and negative observations")
    floor = _metric_floor(metric, metric_floor)

    def flagged(values: Sequence[float]) -> list[dict[str, Any]]:
        return [
            {"value": value, "at_metric_floor": floor is not None and value == floor}
            for value in values
        ]

    return {
        "statistic": RAW_SCORE_STATISTIC,
        "metric": metric,
        "metric_floor": floor,
        "positive": flagged(positive_values),
        "negative": flagged(negative_values),
        "zero_variance_groups": _zero_variance_groups(positive_values, negative_values),
    }


def compute_hedges_g(positive: Sequence[float], negative: Sequence[float]) -> dict[str, Any]:
    """Return Hedges' g and the exact descriptive inputs used to compute it."""
    positive_values, positive_error = _finite_values(positive, "positive")
    negative_values, negative_error = _finite_values(negative, "negative")
    if positive_error or negative_error:
        raise ValueError(positive_error or negative_error)
    if len(positive_values) < 2 or len(negative_values) < 2:
        raise ValueError("Hedges' g requires at least two positive and two negative observations")
    zero_variance_groups = _zero_variance_groups(positive_values, negative_values)
    if zero_variance_groups:
        raise ValueError(
            "Hedges' g must not be computed against a zero-variance control group: "
            + ", ".join(zero_variance_groups)
        )
    positive_mean = math.fsum(positive_values) / len(positive_values)
    negative_mean = math.fsum(negative_values) / len(negative_values)
    degrees_of_freedom = len(positive_values) + len(negative_values) - 2
    pooled_variance = (
        (len(positive_values) - 1) * _sample_variance(positive_values)
        + (len(negative_values) - 1) * _sample_variance(negative_values)
    ) / degrees_of_freedom
    pooled_sd = math.sqrt(pooled_variance)
    if pooled_sd == 0.0:
        raise ValueError("Hedges' g is undefined when the pooled sample standard deviation is zero")
    cohen_d = (positive_mean - negative_mean) / pooled_sd
    correction = 1.0 - 3.0 / (4.0 * (len(positive_values) + len(negative_values)) - 9.0)
    return {
        "statistic": STATISTIC,
        "value": correction * cohen_d,
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "positive_sample_sd": math.sqrt(_sample_variance(positive_values)),
        "negative_sample_sd": math.sqrt(_sample_variance(negative_values)),
        "pooled_sample_sd": pooled_sd,
        "cohen_d": cohen_d,
        "small_sample_correction": correction,
        "positive_n": len(positive_values),
        "negative_n": len(negative_values),
        "positive_values": positive_values,
        "negative_values": negative_values,
    }


def _metric_floor(metric: str, declared_floor: Any) -> float | None:
    if declared_floor is None:
        return METRIC_FLOORS.get(metric)
    if isinstance(declared_floor, bool) or not isinstance(declared_floor, (int, float)):
        raise ValueError("metric_floor must be a finite number or null")
    value = float(declared_floor)
    if not math.isfinite(value):
        raise ValueError("metric_floor must be a finite number or null")
    return value


def _noise_normalizer(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("noise_normalizer must be an object")
    normalizer = dict(value)
    magnitude = normalizer.get("value")
    if isinstance(magnitude, bool) or not isinstance(magnitude, (int, float)):
        raise ValueError("noise_normalizer.value must be a positive finite number")
    magnitude = float(magnitude)
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("noise_normalizer.value must be a positive finite number")
    normalizer["value"] = magnitude
    return normalizer


def compute_floor_robust_separation(
    positive: Sequence[float],
    negative: Sequence[float],
    *,
    metric: str = METRIC,
    metric_floor: Any = None,
    noise_normalizer: Any = None,
) -> dict[str, Any]:
    """Describe control separation without dividing by observed control spread.

    An inversion is a positive-negative pair where the negative score exceeds
    the positive score. Ties are counted separately because strict separation
    requires every positive score to exceed every negative score.
    """
    positive_values, positive_error = _finite_values(positive, "positive")
    negative_values, negative_error = _finite_values(negative, "negative")
    if positive_error or negative_error:
        raise ValueError(positive_error or negative_error)
    if not positive_values or not negative_values:
        raise ValueError("floor-robust separation requires positive and negative observations")
    floor = _metric_floor(metric, metric_floor)
    normalizer = _noise_normalizer(noise_normalizer)
    positive_minimum = min(positive_values)
    positive_maximum = max(positive_values)
    negative_minimum = min(negative_values)
    negative_maximum = max(negative_values)
    inversion_count = sum(
        1
        for positive_value in positive_values
        for negative_value in negative_values
        if negative_value > positive_value
    )
    tie_count = sum(
        1
        for positive_value in positive_values
        for negative_value in negative_values
        if negative_value == positive_value
    )
    constant_groups = [
        label
        for label, values in (("positive", positive_values), ("negative", negative_values))
        if min(values) == max(values)
    ]
    floor_groups = [
        label
        for label, values in (("positive", positive_values), ("negative", negative_values))
        if floor is not None and all(item == floor for item in values)
    ]
    floor_effect_detected = bool(constant_groups or floor_groups)
    warning = None
    if floor_effect_detected:
        groups = ", ".join(dict.fromkeys([*constant_groups, *floor_groups]))
        warning = (
            f"The {groups} control group is constant or entirely at the {metric} floor. "
            "The standardized difference is inflated and must not be read as an effect size."
        )
    margin = positive_minimum - negative_maximum
    return {
        "statistic": FLOOR_ROBUST_STATISTIC,
        "metric": metric,
        "direction": DIRECTION,
        "positive_minimum": positive_minimum,
        "positive_maximum": positive_maximum,
        "negative_minimum": negative_minimum,
        "negative_maximum": negative_maximum,
        "groups_overlap": max(positive_minimum, negative_minimum) <= min(positive_maximum, negative_maximum),
        "inversion_count": inversion_count,
        "tie_count": tie_count,
        "comparison_count": len(positive_values) * len(negative_values),
        "strictly_separated": inversion_count == 0 and tie_count == 0,
        "margin": margin,
        "metric_floor": floor,
        "constant_groups": constant_groups,
        "groups_at_metric_floor": floor_groups,
        "floor_effect_detected": floor_effect_detected,
        "standardized_difference_inflated": floor_effect_detected,
        "standardized_difference_warning": warning,
        "noise_normalizer": normalizer,
        "margin_in_noise_units": margin / normalizer["value"] if normalizer is not None else None,
    }


def assess_rank_score_direction(controls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Assess whether ranked controls separate without a numeric threshold.

    Controls must already have received ``rank_score`` through the ranking path
    under test. The matched and mismatched roles name the characterized pair
    carried by the campaign control panel.
    """
    matched = [
        control
        for control in controls
        if control.get("control_role") == MATCHED_CONTROL_ROLE
    ]
    mismatched = [
        control
        for control in controls
        if control.get("control_role") == MISMATCHED_CONTROL_ROLE
    ]

    def details(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "candidate_id": row.get("candidate_id"),
                "rank_score": row.get("rank_score"),
                "coverage_complete": row.get("coverage_complete"),
            }
            for row in rows
        ]

    result: dict[str, Any] = {
        "statistic": RANK_STATISTIC,
        "direction": RANK_DIRECTION,
        "matched_controls": details(matched),
        "mismatched_controls": details(mismatched),
        "matched_minimum_rank_score": None,
        "mismatched_maximum_rank_score": None,
        "gap": None,
        "separated": None,
    }
    if not matched or not mismatched:
        missing: list[str] = []
        if not matched:
            missing.append(MATCHED_CONTROL_ROLE)
        if not mismatched:
            missing.append(MISMATCHED_CONTROL_ROLE)
        return {
            **result,
            "status": "unvalidated",
            "reason": "the ranking input has no scored control for " + ", ".join(missing),
        }

    def score(row: Mapping[str, Any]) -> float | None:
        value = row.get("rank_score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    matched_scores = [score(row) for row in matched]
    mismatched_scores = [score(row) for row in mismatched]
    if any(value is None for value in [*matched_scores, *mismatched_scores]):
        return {
            **result,
            "status": "degraded",
            "reason": "a configured matched or mismatched control has no finite rank_score",
        }
    matched_minimum = min(value for value in matched_scores if value is not None)
    mismatched_maximum = max(value for value in mismatched_scores if value is not None)
    separated = matched_minimum > mismatched_maximum
    return {
        **result,
        "status": "passed" if separated else "degraded",
        "matched_minimum_rank_score": matched_minimum,
        "mismatched_maximum_rank_score": mismatched_maximum,
        "gap": matched_minimum - mismatched_maximum,
        "separated": separated,
        "reason": (
            "the matched control strictly outranks the mismatched control"
            if separated
            else "the matched control does not strictly outrank the mismatched control"
        ),
    }


def _setting(config: Mapping[str, Any]) -> Mapping[str, Any]:
    qualification = config.get("qualification")
    if not isinstance(qualification, Mapping):
        return {}
    separation = qualification.get("control_separation")
    return separation if isinstance(separation, Mapping) else {}


def configured_threshold(config: Mapping[str, Any], arm: str, target_id: str) -> tuple[float | None, str]:
    """Resolve an optional numeric threshold without supplying a default."""
    settings = _setting(config)
    candidates: list[tuple[Any, str]] = []
    thresholds = settings.get("thresholds")
    if isinstance(thresholds, Mapping):
        for key in (f"{arm}:{target_id}", f"{arm}/{target_id}", arm, target_id):
            if key in thresholds:
                candidates.append((thresholds[key], f"qualification.control_separation.thresholds.{key}"))
                break
    for key in ("threshold", "default_threshold"):
        if key in settings:
            candidates.append((settings[key], f"qualification.control_separation.{key}"))
            break
    if not candidates or candidates[0][0] is None:
        return DEFAULT_THRESHOLD, "no default threshold is configured"
    value, source = candidates[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None, f"{source} is invalid"
    return float(value), source


def _expected_pairs(config: Mapping[str, Any], arms: Sequence[str] | None, target_ids: Sequence[str] | None) -> list[tuple[str, str]]:
    if arms is None:
        cofold = config.get("cofold")
        predictors = cofold.get("predictors") if isinstance(cofold, Mapping) else []
        arms = []
        if isinstance(predictors, list):
            for predictor in predictors:
                if isinstance(predictor, Mapping) and predictor.get("enabled", True) is not False:
                    value = predictor.get("id")
                    if isinstance(value, str) and value:
                        from .arms import score_instrument_arm_name

                        arms.append(score_instrument_arm_name(value))
    if target_ids is None:
        targets = config.get("targets")
        target_ids = [
            str(target.get("target_id"))
            for target in targets or []
            if isinstance(target, Mapping) and isinstance(target.get("target_id"), str)
        ]
    return [(str(arm), str(target)) for arm in arms for target in target_ids]


def _record_map(
    roster_document: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], dict[tuple[str, str], list[str]]]:
    """Map stored measurements of this assessment's metric, and name the ones it skipped.

    The map keyed on `(arm, target_id)` and dropped `metric`, so two entries for one arm
    and target differing only in metric kept whichever came last. An incomplete
    `ipsae_min` measurement followed by a complete `sc_dockq` one then returned `ok` for
    an assessment that still declared its metric as `ipsae_min`, and `runtime_validator`
    turned that into a passing `control-separation` check. Reversing the order restored
    the refusal, which is the tell that the map decided it rather than the statistics.

    Same partial-identity shape as the other key collisions in this package, with `metric`
    as the omitted dimension rather than `target_id`, and worse in kind because what it
    silently changed was a gate. A measurement that declares no metric is
    skipped too: it says nothing about what it measured, so it cannot satisfy a gate that
    names one. The second return value carries the skipped metrics so the refusal can say
    a measurement exists and what it measured, rather than that none exists.
    """
    qualification = roster_document.get("qualification")
    separation = qualification.get("control_separation") if isinstance(qualification, Mapping) else None
    measurements = separation.get("measurements") if isinstance(separation, Mapping) else None
    if not isinstance(measurements, list):
        return {}, {}
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    other_metrics: dict[tuple[str, str], list[str]] = {}
    for item in measurements:
        if not (
            isinstance(item, Mapping)
            and isinstance(item.get("arm"), str)
            and isinstance(item.get("target_id"), str)
        ):
            continue
        key = (item["arm"], item["target_id"])
        metric = item.get("metric")
        if metric == METRIC:
            result[key] = item
            continue
        label = metric if isinstance(metric, str) and metric else "no declared metric"
        if label not in other_metrics.setdefault(key, []):
            other_metrics[key].append(label)
    return result, other_metrics


def _floor_effect_details(record: Mapping[str, Any] | None) -> tuple[bool, str | None]:
    if not isinstance(record, Mapping):
        return False, None
    floor_robust = record.get("floor_robust_separation")
    detected = record.get("floor_effect_detected") is True
    warning = record.get("standardized_difference_warning")
    if isinstance(floor_robust, Mapping):
        detected = detected or floor_robust.get("floor_effect_detected") is True
        if warning is None:
            warning = floor_robust.get("standardized_difference_warning")
    return detected, str(warning) if isinstance(warning, str) and warning else None


def _raw_scores_from_record(record: Mapping[str, Any]) -> dict[str, Any] | None:
    stored = record.get("raw_scores")
    if isinstance(stored, Mapping):
        return dict(stored)
    inputs = record.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    positive_controls = inputs.get("positive_controls")
    negative_controls = inputs.get("negative_controls")
    if not isinstance(positive_controls, list) or not isinstance(negative_controls, list):
        return None
    positive_values: list[float] = []
    negative_values: list[float] = []
    for label, controls, destination in (
        ("positive", positive_controls, positive_values),
        ("negative", negative_controls, negative_values),
    ):
        for index, control in enumerate(controls):
            if not isinstance(control, Mapping):
                return None
            values, error = _control_values(control, f"stored {label} control {index}")
            if error:
                return None
            destination.extend(values)
    try:
        return report_raw_scores_with_floor_flags(
            positive_values,
            negative_values,
            metric=str(record.get("metric", METRIC)),
            metric_floor=record.get("metric_floor"),
        )
    except ValueError:
        return None


def _summary_statistic(records: Sequence[Mapping[str, Any]]) -> str:
    statistics = {
        record.get("statistic")
        for record in records
        if isinstance(record.get("statistic"), str) and record.get("statistic")
    }
    if len(statistics) == 1:
        return next(iter(statistics))
    return "per-measurement"


def assess_roster(
    config: Mapping[str, Any],
    roster_document: Mapping[str, Any],
    *,
    arms: Sequence[str] | None = None,
    target_ids: Sequence[str] | None = None,
    enforce: bool = True,
) -> dict[str, Any]:
    """Assess stored controls without reporting a censored effect size."""
    pairs = _expected_pairs(config, arms, target_ids)
    if not enforce:
        return {
            "ok": True,
            "status": "not_required",
            "statistic": STATISTIC,
            "metric": METRIC,
            "measurements": [],
            "errors": [],
            "reason": "production scoring is disabled for this run",
        }
    records, other_metrics = _record_map(roster_document)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    for arm, target_id in pairs:
        record = records.get((arm, target_id))
        configured, configured_source = configured_threshold(config, arm, target_id)
        floor_effect_detected, standardized_difference_warning = _floor_effect_details(record)
        statistic = STATISTIC
        threshold_statistic = STATISTIC
        raw_scores: dict[str, Any] | None = None
        if isinstance(record, Mapping) and record.get("threshold_statistic") is not None:
            candidate_statistic = record.get("threshold_statistic")
            if isinstance(candidate_statistic, str) and candidate_statistic:
                threshold_statistic = candidate_statistic
            else:
                threshold_statistic = "<invalid>"
        if record is None:
            value = None
            status = "incomplete"
            threshold = configured
            skipped = other_metrics.get((arm, target_id))
            if skipped:
                reason = (
                    "the qualification record measures "
                    f"{', '.join(sorted(skipped))} for this arm and target, not {METRIC}"
                )
            else:
                reason = "the qualification record has no control-separation measurement"
        else:
            raw_scores = _raw_scores_from_record(record)
            if isinstance(record.get("statistic"), str) and record.get("statistic"):
                statistic = str(record["statistic"])
            value = record.get("value")
            threshold = record.get("threshold") if record.get("threshold") is not None else configured
            if record.get("status") != "measured":
                value = None
                status = "incomplete"
                reason = str(record.get("reason") or "the qualification record is marked incomplete")
            elif statistic == RAW_SCORE_STATISTIC:
                value = None
                threshold_statistic = RAW_SCORE_STATISTIC
                if threshold is None:
                    status = "measured"
                    reason = "stored raw control scores are present"
                else:
                    status = "incomplete"
                    reason = (
                        "a numeric control-separation threshold is TODO until a validation "
                        "gate defines one for raw scores"
                    )
            elif (
                statistic == STATISTIC
                and (
                    floor_effect_detected
                    or (
                        isinstance(raw_scores, Mapping)
                        and raw_scores.get("zero_variance_groups")
                    )
                )
            ):
                statistic = RAW_SCORE_STATISTIC
                threshold_statistic = RAW_SCORE_STATISTIC
                value = None
                floor_effect_detected = True
                standardized_difference_warning = (
                    "A control group has zero variance. Raw scores with floor flags replace "
                    "the standardized difference."
                )
                if raw_scores is None:
                    status = "incomplete"
                    reason = (
                        "a stored standardized difference has a floor effect, but the raw "
                        "control scores are unavailable"
                    )
                elif threshold is None:
                    status = "measured"
                    reason = "stored raw control scores are present"
                else:
                    status = "incomplete"
                    reason = (
                        "a numeric control-separation threshold is TODO until a validation "
                        "gate defines one for raw scores"
                    )
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                value = None
                status = "incomplete"
                reason = str(record.get("reason") or "the qualification record has no finite separation value")
            elif record.get("statistic") != STATISTIC:
                status = "incomplete"
                reason = f"the qualification record uses statistic {record.get('statistic')!r}, expected {STATISTIC!r}"
            else:
                status = "measured"
                reason = "stored control separation is present"
        if isinstance(threshold, bool) or (threshold is not None and (not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)))):
            threshold = None
            status = "incomplete"
            reason = f"{configured_source} is invalid"
        elif statistic == RAW_SCORE_STATISTIC and threshold is not None:
            status = "incomplete"
            reason = (
                "a numeric control-separation threshold is TODO until a validation gate "
                "defines one for raw scores"
            )
        elif floor_effect_detected and threshold is not None and threshold_statistic == STATISTIC:
            status = "incomplete"
            reason = (
                f"the record detects a floor effect, so a threshold expressed in {STATISTIC} "
                "cannot be accepted because the standardized difference is inflated"
            )
        passes = status == "measured" and (
            threshold is None
            or isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) >= float(threshold)
        )
        if status == "measured" and threshold is not None and not passes:
            reason = "stored control separation is below the configured threshold"
        check = {
            "arm": arm,
            "target_id": target_id,
            "statistic": statistic,
            "metric": record.get("metric", METRIC) if record is not None else METRIC,
            "value": value,
            "threshold": float(threshold) if threshold is not None else None,
            "threshold_source": (
                record.get("threshold_source")
                if record is not None and record.get("threshold") is not None
                else configured_source
            ),
            "threshold_statistic": threshold_statistic,
            "status": status,
            "ok": passes,
            "reason": reason,
            "floor_effect_detected": floor_effect_detected,
            "standardized_difference_warning": standardized_difference_warning,
            "raw_scores": raw_scores,
            "positive_control_ids": list(record.get("positive_control_ids", [])) if isinstance(record, Mapping) and isinstance(record.get("positive_control_ids"), list) else [],
            "negative_control_ids": list(record.get("negative_control_ids", [])) if isinstance(record, Mapping) and isinstance(record.get("negative_control_ids"), list) else [],
        }
        checks.append(check)
        if not passes:
            if floor_effect_detected and threshold is not None and threshold_statistic == STATISTIC:
                errors.append(
                    f"control separation for arm {arm!r} on target {target_id!r} has a threshold "
                    f"expressed in {STATISTIC}. {reason}. "
                    f"{standardized_difference_warning or ''}".strip()
                )
                continue
            measured = "unmeasured" if value is None else f"{float(value):.6f}"
            threshold_text = "unconfigured" if threshold is None else f"{float(threshold):.6f}"
            errors.append(
                f"control separation for arm {arm!r} on target {target_id!r} is {measured}; "
                f"threshold is {threshold_text}. Add a stored qualification.control_separation "
                "measurement from known-binder and negative controls, then rerun qualification."
            )
    return {
        "ok": not errors,
        "status": "pass" if not errors else "fail",
        "statistic": _summary_statistic(checks),
        "metric": METRIC,
        "measurements": checks,
        "errors": errors,
    }


def build_qualification_record(
    config: Mapping[str, Any],
    *,
    qualified_at: str,
) -> dict[str, Any]:
    """Build the stored qualification section from measured control values."""
    settings = _setting(config)
    raw_measurements = settings.get("measurements")
    if not isinstance(raw_measurements, list):
        raw_measurements = []
    measurements: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_measurements):
        if not isinstance(raw, Mapping):
            measurements.append({
                "arm": "<invalid>",
                "target_id": "<invalid>",
                "statistic": STATISTIC,
                "metric": METRIC,
                "value": None,
                "status": "incomplete",
                "reason": f"measurement {index} is not an object",
                "qualified_at": qualified_at,
            })
            continue
        arm = str(raw.get("arm", ""))
        target_id = str(raw.get("target_id", ""))
        positive_controls = raw.get("positive_controls", raw.get("positive", []))
        negative_controls = raw.get("negative_controls", raw.get("negative", []))
        positive_controls = positive_controls if isinstance(positive_controls, list) else []
        negative_controls = negative_controls if isinstance(negative_controls, list) else []
        positive_values: list[float] = []
        negative_values: list[float] = []
        errors: list[str] = []
        normalized_positive: list[dict[str, Any]] = []
        normalized_negative: list[dict[str, Any]] = []
        for polarity, source, target in (
            ("positive", positive_controls, normalized_positive),
            ("negative", negative_controls, normalized_negative),
        ):
            for control_index, control in enumerate(source):
                if not isinstance(control, Mapping):
                    errors.append(f"{polarity} control {control_index} is not an object")
                    continue
                values, error = _control_values(control, f"{polarity} control {control_index}")
                if error:
                    errors.append(error)
                    continue
                normalized_control = {
                    "id": control.get("id"),
                    "values": values,
                    "source": control.get("source"),
                }
                for provenance_key in ("role", "provenance"):
                    if provenance_key in control:
                        normalized_control[provenance_key] = control[provenance_key]
                target.append(normalized_control)
                (positive_values if polarity == "positive" else negative_values).extend(values)
        configured, configured_source = configured_threshold(config, arm, target_id)
        threshold = raw.get("threshold") if raw.get("threshold") is not None else configured
        threshold_source = "measurement configuration" if raw.get("threshold") is not None else configured_source
        metric = str(raw.get("metric", METRIC))
        result: dict[str, Any] = {
            "arm": arm,
            "target_id": target_id,
            "statistic": STATISTIC,
            "metric": metric,
            "direction": DIRECTION,
            "qualified_at": qualified_at,
            "positive_control_ids": [item.get("id") for item in normalized_positive],
            "negative_control_ids": [item.get("id") for item in normalized_negative],
            "inputs": {"positive_controls": normalized_positive, "negative_controls": normalized_negative},
            "threshold": threshold,
            "threshold_source": threshold_source,
            "threshold_statistic": STATISTIC,
        }
        for provenance_key in ("predictor_id", "target_name", "provenance"):
            if provenance_key in raw:
                result[provenance_key] = raw[provenance_key]
        if errors:
            result.update({"status": "incomplete", "value": None, "reason": "; ".join(errors)})
        else:
            try:
                raw_scores = report_raw_scores_with_floor_flags(
                    positive_values,
                    negative_values,
                    metric=metric,
                    metric_floor=raw.get("metric_floor"),
                )
                if raw_scores["zero_variance_groups"]:
                    result.update(
                        {
                            "statistic": RAW_SCORE_STATISTIC,
                            "value": None,
                            "raw_scores": raw_scores,
                            "floor_effect_detected": True,
                            "standardized_difference_inflated": True,
                            "standardized_difference_warning": (
                                "A control group has zero variance. Raw scores with floor flags "
                                "replace the standardized difference."
                            ),
                            "threshold_statistic": RAW_SCORE_STATISTIC,
                        }
                    )
                    if threshold is None:
                        result.update(
                            {
                                "status": "measured",
                                "reason": "raw control scores are recorded without a numeric threshold",
                            }
                        )
                    else:
                        result.update(
                            {
                                "status": "incomplete",
                                "reason": (
                                    "a numeric control-separation threshold is TODO until a validation "
                                    "gate defines one for raw scores"
                                ),
                            }
                        )
                    measurements.append(result)
                    continue
                floor_robust = compute_floor_robust_separation(
                    positive_values,
                    negative_values,
                    metric=metric,
                    metric_floor=raw.get("metric_floor"),
                    noise_normalizer=raw.get("noise_normalizer"),
                )
                result.update({
                    "status": "measured",
                    **compute_hedges_g(positive_values, negative_values),
                    "raw_scores": raw_scores,
                    "floor_robust_separation": floor_robust,
                    "floor_effect_detected": floor_robust["floor_effect_detected"],
                    "standardized_difference_inflated": floor_robust["standardized_difference_inflated"],
                    "standardized_difference_warning": floor_robust["standardized_difference_warning"],
                })
            except ValueError as exc:
                result.update({"status": "incomplete", "value": None, "reason": str(exc), "positive_values": positive_values, "negative_values": negative_values})
        measurements.append(result)
    expected = _expected_pairs(config, None, None)
    present = {(item.get("arm"), item.get("target_id")) for item in measurements}
    for arm, target_id in expected:
        if (arm, target_id) not in present:
            threshold, source = configured_threshold(config, arm, target_id)
            measurements.append({
                "arm": arm,
                "target_id": target_id,
                "statistic": STATISTIC,
                "metric": METRIC,
                "direction": DIRECTION,
                "qualified_at": qualified_at,
                "positive_control_ids": [],
                "negative_control_ids": [],
                "inputs": {"positive_controls": [], "negative_controls": []},
                "threshold": threshold,
                "threshold_source": source,
                "status": "incomplete",
                "value": None,
                "reason": "no control-separation measurement was supplied for this arm and target",
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "statistic": _summary_statistic(measurements),
        "metric": METRIC,
        "direction": DIRECTION,
        "default_threshold": DEFAULT_THRESHOLD,
        "threshold_policy": "A numeric threshold is optional because one calibration target does not support a confident universal default. Missing separation remains blocking for production scoring.",
        "qualified_at": qualified_at,
        "status": "measured" if measurements and all(item.get("status") == "measured" for item in measurements) else ("not_applicable" if not expected else "incomplete"),
        "measurements": measurements,
    }


def load_roster_document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"model-roster ledger cannot be read: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"model-roster ledger must be an object: {path}")
    return value


def assess_roster_path(
    config: Mapping[str, Any],
    path: Path,
    *,
    arms: Sequence[str] | None = None,
    target_ids: Sequence[str] | None = None,
    enforce: bool = True,
) -> dict[str, Any]:
    if not enforce:
        return assess_roster(
            config,
            {},
            arms=arms,
            target_ids=target_ids,
            enforce=False,
        )
    try:
        document = load_roster_document(path)
    except ValueError as exc:
        return {
            "ok": False,
            "status": "incomplete",
            "statistic": STATISTIC,
            "metric": METRIC,
            "measurements": [],
            "errors": [str(exc)],
        }
    return assess_roster(config, document, arms=arms, target_ids=target_ids, enforce=enforce)
