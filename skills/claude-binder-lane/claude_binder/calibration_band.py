"""Build a provenance-bound calibration band from stage 3 control rows.

The reducer for ``ipsae_min`` is published: use the lower observed edge for
positive panel members and the upper observed edge for negative panel members.
The published protocol gives no reducer for ``sc_dockq``, ``site_contact_iou``,
or ``target_contact_recall``. The module applies the same conservative polarity
edge to those metrics as a labeled product default. Callers can override each
default with an explicit reducer.

The module reads raw control rows from
``controls/control-observations.jsonl``. It keeps calibration-panel members
separate from per-target controls through the explicit ``scope_by_key``
argument. It selects the highest raw ``ipsae_min`` seed per predictor, retains
every raw seed value, and averages the selected predictor values for the
ensemble. It never computes a z-score.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CALIBRATION_METRICS = (
    "ipsae_min",
    "sc_dockq",
    "site_contact_iou",
    "target_contact_recall",
)
PUBLISHED_IPSAE_METRIC = "ipsae_min"
EXPLICIT_REDUCER_METRICS = tuple(
    metric for metric in CALIBRATION_METRICS if metric != PUBLISHED_IPSAE_METRIC
)
CALIBRATION_PANEL_SCOPE = "calibration-panel"
TARGET_CONTROL_SCOPE = "target-control"
ALLOWED_SCOPES = frozenset({CALIBRATION_PANEL_SCOPE, TARGET_CONTROL_SCOPE})
PUBLISHED_PANEL_GUIDANCE = (
    "The published campaign uses native complexes such as Barnase/Barstar for the "
    "positive panel. It uses a non-interacting pair, a sequence-shuffled binder, and "
    "a cross-pair mismatch for the negative panel. Keep panel members outside the "
    "campaign target set."
)
ROW_KEY_FIELDS = ("target_id", "candidate_id", "predictor", "seed")
PROVENANCE_FIELDS = (
    "target_id",
    "candidate_id",
    "target_sha256",
    "control_structure_sha256",
    "predictor",
    "model_revision",
    "seed",
    "sequence_sha256",
    "design_pose_sha256",
    "predicted_complex_sha256",
    "pae_sha256",
    "chain_mapping",
    "msa_mode",
    "msa_path",
    "msa_sha256",
    "ipsae_pairing",
    "ipsae_implementation_revision",
    "ipsae_interface_cutoff_angstrom",
    "dockq_implementation_revision",
    "site_scorer_revision",
    "site_residue_map_sha256",
    "site_contact_cutoff_angstrom",
    "site_atom_selection",
    "site_metric_basis",
    "metric_source_sha256",
)
MEMBER_IDENTITY_FIELDS = (
    "target_sha256",
    "control_structure_sha256",
    "sequence_sha256",
    "design_pose_sha256",
    "chain_mapping",
)
STRUCTURE_HASH_FIELDS = (
    "target_sha256",
    "control_structure_sha256",
    "design_pose_sha256",
    "predicted_complex_sha256",
)

Reducer = Callable[..., float] | str
RowKey = tuple[str, str, str, int]


class CalibrationBandError(ValueError):
    """Raised when raw control rows cannot produce a defensible band."""


def observation_key(row: Mapping[str, Any]) -> RowKey:
    """Return the target, control, predictor, and seed key used by stage 3."""

    missing = [field for field in ROW_KEY_FIELDS if field not in row]
    if missing:
        raise CalibrationBandError(
            "observation row is missing key field(s): " + ", ".join(missing)
        )
    try:
        seed = int(row["seed"])
    except (TypeError, ValueError) as exc:
        raise CalibrationBandError("observation row seed must be an integer") from exc
    if isinstance(row["seed"], bool):
        raise CalibrationBandError("observation row seed must be an integer")
    return (
        str(row["target_id"]),
        str(row["candidate_id"]),
        str(row["predictor"]),
        seed,
    )


def build_calibration_band(
    artifact_root: str | Path,
    *,
    scope_by_key: Mapping[RowKey, str],
    reducers: Mapping[str, Reducer] | None = None,
) -> dict[str, Any]:
    """Read stage 3 rows and return a calibration-band artifact.

    ``scope_by_key`` is required because the current raw row contract has no
    field that identifies calibration-panel members. Every source row must
    have one entry with the value ``calibration-panel`` or ``target-control``.

    Reducers for the three metrics without a published reducer receive the
    selected raw values and the polarity string ``positive`` or ``negative``.
    The default takes the minimum positive value and maximum negative value.
    A reducer may also accept only the values. Supported string reducers are
    ``minimum``, ``maximum``, ``mean``, and ``median``.
    """

    root = Path(artifact_root)
    observation_path = root / "controls" / "control-observations.jsonl"
    source_bytes = _read_source_bytes(observation_path)
    rows = _load_jsonl(source_bytes, observation_path)
    scopes = _normalize_scopes(scope_by_key)
    prepared_rows = _prepare_rows(rows, scopes, root)
    if not prepared_rows or not any(
        row["scope"] == CALIBRATION_PANEL_SCOPE for row in prepared_rows
    ):
        raise CalibrationBandError(
            f"calibration panel is empty. {PUBLISHED_PANEL_GUIDANCE}"
        )
    reducer_map = dict(reducers or {})
    for metric in EXPLICIT_REDUCER_METRICS:
        reducer_map.setdefault(metric, _product_default_edge)
    panel_rows = [
        row for row in prepared_rows if row["scope"] == CALIBRATION_PANEL_SCOPE
    ]
    if not panel_rows:
        raise CalibrationBandError(
            f"calibration panel is empty. {PUBLISHED_PANEL_GUIDANCE}"
        )

    members = _group_members(prepared_rows)
    predictors = _validate_member_coverage(members)
    seed_sets = _validate_seed_sets(members, predictors)
    panel_members = [
        member for member in members.values() if member["scope"] == CALIBRATION_PANEL_SCOPE
    ]
    if not any(member["polarity"] == "positive" for member in panel_members):
        raise CalibrationBandError("calibration panel has no positive members")
    if not any(member["polarity"] == "negative" for member in panel_members):
        raise CalibrationBandError("calibration panel has no negative members")

    selected = _select_member_values(members, predictors)
    bands = _build_bands(members, selected, predictors, reducer_map)
    separation = _build_separation(bands, predictors)
    _refuse_without_separation(separation, predictors)

    panel_artifacts = [
        _member_artifact(member, selected[member["member_key"]], predictors, seed_sets)
        for member in panel_members
    ]
    control_artifacts = [
        _member_artifact(member, selected[member["member_key"]], predictors, seed_sets)
        for member in members.values()
        if member["scope"] == TARGET_CONTROL_SCOPE
    ]
    ensemble_band = bands["ensemble"]
    return {
        "schema_version": 1,
        "artifact_type": "calibration-band",
        "source_observations": {
            "path": "controls/control-observations.jsonl",
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "row_count": len(rows),
        },
        "scope_contract": {
            "source_field": None,
            "source_field_status": "absent",
            "argument": "scope_by_key",
            "allowed_values": sorted(ALLOWED_SCOPES),
        },
        "metrics": list(CALIBRATION_METRICS),
        "predictors": list(predictors),
        "seed_sets": {predictor: list(seed_sets[predictor]) for predictor in predictors},
        "reducers": {
            metric: _reducer_descriptor(metric, reducer_map.get(metric))
            for metric in CALIBRATION_METRICS
        },
        "calibration_panel": {"members": panel_artifacts},
        "target_controls": {"members": control_artifacts},
        "bands": bands,
        "panel_separation": separation,
        "resolved_thresholds": _resolved_thresholds(ensemble_band),
        "provenance": _provenance(prepared_rows),
    }


def write_calibration_band(
    artifact_root: str | Path,
    *,
    scope_by_key: Mapping[RowKey, str],
    reducers: Mapping[str, Reducer] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Build and atomically write ``controls/calibration-band.json``."""

    root = Path(artifact_root)
    path = Path(output_path) if output_path is not None else root / "controls" / "calibration-band.json"
    band = build_calibration_band(root, scope_by_key=scope_by_key, reducers=reducers)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(band, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _read_source_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CalibrationBandError(f"control observations are unreadable: {path}") from exc


def _load_jsonl(source_bytes: bytes, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source_bytes.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CalibrationBandError(
                f"control observations JSONL line {line_number} is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise CalibrationBandError(
                f"control observations JSONL line {line_number} must be an object: {path}"
            )
        rows.append(value)
    return rows


def _normalize_scopes(scope_by_key: Mapping[RowKey, str]) -> dict[RowKey, str]:
    normalized: dict[RowKey, str] = {}
    for raw_key, scope in scope_by_key.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 4:
            raise CalibrationBandError(
                "scope_by_key keys must be (target_id, candidate_id, predictor, seed) tuples"
            )
        try:
            key = (str(raw_key[0]), str(raw_key[1]), str(raw_key[2]), int(raw_key[3]))
        except (TypeError, ValueError) as exc:
            raise CalibrationBandError("scope_by_key seed must be an integer") from exc
        if scope not in ALLOWED_SCOPES:
            raise CalibrationBandError(
                f"scope for {key} must be one of {sorted(ALLOWED_SCOPES)}"
            )
        normalized[key] = str(scope)
    return normalized


def _prepare_rows(
    rows: Sequence[dict[str, Any]],
    scopes: Mapping[RowKey, str],
    artifact_root: Path,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen_keys: set[RowKey] = set()
    for index, raw in enumerate(rows):
        key = observation_key(raw)
        if key in seen_keys:
            raise CalibrationBandError(f"duplicate observation key: {key}")
        seen_keys.add(key)
        scope = scopes.get(key)
        if scope is None:
            raise CalibrationBandError(
                f"scope is missing for observation key {key}; pass scope_by_key explicitly"
            )
        control_type = str(raw.get("control_type", ""))
        if control_type not in {"positive", "negative"}:
            raise CalibrationBandError(
                f"observation row {index} has invalid control_type: {control_type}"
            )
        effective = _effective_measurement(raw, artifact_root)
        metrics: dict[str, float] = {}
        for metric in CALIBRATION_METRICS:
            value = effective.get(metric)
            if not _is_finite_number(value):
                raise CalibrationBandError(
                    f"observation row {key} is missing numeric metric {metric}"
                )
            metrics[metric] = float(value)
        prepared.append(
            {
                "key": key,
                "scope": scope,
                "polarity": control_type,
                "raw": raw,
                "effective": effective,
                "metrics": metrics,
                "member_key": (scope, key[0], key[1]),
            }
        )
    return prepared


def _effective_measurement(raw: Mapping[str, Any], artifact_root: Path) -> dict[str, Any]:
    missing_metrics = [metric for metric in CALIBRATION_METRICS if metric not in raw]
    if not missing_metrics:
        return dict(raw)
    source_path_value = raw.get("metric_source_path")
    if not isinstance(source_path_value, str) or not source_path_value:
        return dict(raw)
    candidates = [Path(source_path_value)]
    source_path = Path(source_path_value)
    if not source_path.is_absolute():
        candidates.extend((artifact_root / source_path, artifact_root / "controls" / source_path))
    source_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source_path is None:
        return dict(raw)
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationBandError(
            f"metric source is unreadable for {raw.get('candidate_id')}: {source_path}"
        ) from exc
    if isinstance(source, dict) and isinstance(source.get("measurement"), dict):
        source = source["measurement"]
    if not isinstance(source, dict):
        raise CalibrationBandError(f"metric source must be an object: {source_path}")
    return {**source, **raw}


def _group_members(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    members: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = row["member_key"]
        member = members.setdefault(
            key,
            {
                "member_key": key,
                "scope": row["scope"],
                "polarity": row["polarity"],
                "target_id": row["key"][0],
                "candidate_id": row["key"][1],
                "rows": [],
                "identities": {},
            },
        )
        if member["polarity"] != row["polarity"]:
            raise CalibrationBandError(
                f"member {key} changes positive/negative polarity across rows"
            )
        for field in MEMBER_IDENTITY_FIELDS:
            value = row["effective"].get(field)
            if field not in member["identities"]:
                member["identities"][field] = value
            elif member["identities"][field] != value:
                raise CalibrationBandError(
                    f"member {key} has inconsistent provenance field {field}"
                )
        member["rows"].append(row)
    return members


def _validate_member_coverage(
    members: Mapping[tuple[str, str, str], dict[str, Any]],
) -> list[str]:
    predictor_sets = {
        key: {str(row["key"][2]) for row in member["rows"]}
        for key, member in members.items()
    }
    predictors = sorted(set().union(*predictor_sets.values())) if predictor_sets else []
    missing = {
        key: sorted(set(predictors) - observed)
        for key, observed in predictor_sets.items()
        if set(predictors) - observed
    }
    if missing:
        detail = "; ".join(f"{key}: {values}" for key, values in sorted(missing.items()))
        raise CalibrationBandError(f"predictor missing from some rows: {detail}")
    if not predictors:
        raise CalibrationBandError("predictor missing from some rows: no predictors found")
    return predictors


def _validate_seed_sets(
    members: Mapping[tuple[str, str, str], dict[str, Any]],
    predictors: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    seed_sets: dict[str, set[int]] = {predictor: set() for predictor in predictors}
    for member_key, member in members.items():
        for predictor in predictors:
            observed = {
                row["key"][3]
                for row in member["rows"]
                if row["key"][2] == predictor
            }
            if not observed:
                raise CalibrationBandError(
                    f"predictor missing from some rows: {predictor} for member {member_key}"
                )
            if not seed_sets[predictor]:
                seed_sets[predictor] = observed
            elif observed != seed_sets[predictor]:
                raise CalibrationBandError(
                    "seed set is not identical across members: "
                    f"predictor {predictor}, member {member_key} has "
                    f"{sorted(observed)}, expected {sorted(seed_sets[predictor])}"
                )
    return {predictor: tuple(sorted(values)) for predictor, values in seed_sets.items()}


def _select_member_values(
    members: Mapping[tuple[str, str, str], dict[str, Any]],
    predictors: Sequence[str],
) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    selected: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for member_key, member in members.items():
        selected[member_key] = {}
        for predictor in predictors:
            rows = [
                row for row in member["rows"] if row["key"][2] == predictor
            ]
            selected_row = max(rows, key=lambda row: (row["metrics"][PUBLISHED_IPSAE_METRIC], -row["key"][3]))
            selected[member_key][predictor] = {
                "seed": selected_row["key"][3],
                "metrics": dict(selected_row["metrics"]),
            }
    return selected


def _build_bands(
    members: Mapping[tuple[str, str, str], dict[str, Any]],
    selected: Mapping[tuple[str, str, str], dict[str, dict[str, Any]]],
    predictors: Sequence[str],
    reducers: Mapping[str, Reducer],
) -> dict[str, Any]:
    panel_members = [
        member
        for member in members.values()
        if member["scope"] == CALIBRATION_PANEL_SCOPE
    ]
    per_predictor: dict[str, Any] = {}
    for predictor in predictors:
        per_predictor[predictor] = {}
        for metric in CALIBRATION_METRICS:
            positive_values = [
                float(selected[member["member_key"]][predictor]["metrics"][metric])
                for member in panel_members
                if member["polarity"] == "positive"
            ]
            negative_values = [
                float(selected[member["member_key"]][predictor]["metrics"][metric])
                for member in panel_members
                if member["polarity"] == "negative"
            ]
            per_predictor[predictor][metric] = _band_entry(
                metric, positive_values, negative_values, reducers
            )

    ensemble_values: dict[tuple[str, str, str], dict[str, float]] = {}
    for member in panel_members:
        selected_for_member = selected[member["member_key"]]
        ensemble_values[member["member_key"]] = {
            metric: statistics.fmean(
                selected_for_member[predictor]["metrics"][metric]
                for predictor in predictors
            )
            for metric in CALIBRATION_METRICS
        }
    ensemble: dict[str, Any] = {}
    for metric in CALIBRATION_METRICS:
        positive_values = [
            values[metric]
            for member_key, values in ensemble_values.items()
            if members[member_key]["polarity"] == "positive"
        ]
        negative_values = [
            values[metric]
            for member_key, values in ensemble_values.items()
            if members[member_key]["polarity"] == "negative"
        ]
        ensemble[metric] = _band_entry(
            metric, positive_values, negative_values, reducers
        )
    return {"per_predictor": per_predictor, "ensemble": ensemble}


def _build_separation(bands: Mapping[str, Any], predictors: Sequence[str]) -> dict[str, Any]:
    separation: dict[str, Any] = {"per_predictor": {}, "ensemble": {}}
    for predictor in predictors:
        separation["per_predictor"][predictor] = {
            metric: _separation_entry(bands["per_predictor"][predictor][metric])
            for metric in CALIBRATION_METRICS
        }
    separation["ensemble"] = {
        metric: _separation_entry(bands["ensemble"][metric])
        for metric in CALIBRATION_METRICS
    }
    return separation


def _separation_entry(band: Mapping[str, Any]) -> dict[str, Any]:
    positive_edge = float(band["positive_edge"])
    negative_edge = float(band["negative_edge"])
    return {
        "positive_edge": positive_edge,
        "negative_edge": negative_edge,
        "gap": positive_edge - negative_edge,
        "separated": positive_edge > negative_edge,
    }


def _refuse_without_separation(
    separation: Mapping[str, Any], predictors: Sequence[str]
) -> None:
    failures: list[str] = []
    for predictor in predictors:
        for metric in CALIBRATION_METRICS:
            if not separation["per_predictor"][predictor][metric]["separated"]:
                failures.append(f"{predictor}:{metric}")
    for metric in CALIBRATION_METRICS:
        if not separation["ensemble"][metric]["separated"]:
            failures.append(f"ensemble:{metric}")
    if failures:
        raise CalibrationBandError(
            "positives and negatives do not separate for metric(s): "
            + ", ".join(failures)
        )


def _member_artifact(
    member: Mapping[str, Any],
    selected: Mapping[str, Mapping[str, Any]],
    predictors: Sequence[str],
    seed_sets: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    raw_values: dict[str, dict[str, dict[str, float]]] = {}
    for predictor in predictors:
        rows = [row for row in member["rows"] if row["key"][2] == predictor]
        raw_values[predictor] = {
            str(row["key"][3]): dict(row["metrics"]) for row in rows
        }
    identities = dict(member["identities"])
    identities.update(
        {"target_id": member["target_id"], "candidate_id": member["candidate_id"]}
    )
    return {
        "scope": member["scope"],
        "polarity": member["polarity"],
        "target_id": member["target_id"],
        "candidate_id": member["candidate_id"],
        "identities": identities,
        "structure_hashes": {
            field: identities.get(field) for field in STRUCTURE_HASH_FIELDS
        },
        "seed_sets": {predictor: list(seed_sets[predictor]) for predictor in predictors},
        "raw_values": raw_values,
        "selected_values": {
            predictor: {
                "seed": int(selected[predictor]["seed"]),
                "metrics": dict(selected[predictor]["metrics"]),
            }
            for predictor in predictors
        },
    }


def _resolved_thresholds(ensemble_band: Mapping[str, Any]) -> dict[str, Any]:
    positive = {
        metric: float(ensemble_band[metric]["positive_edge"])
        for metric in CALIBRATION_METRICS
    }
    negative = {
        metric: float(ensemble_band[metric]["negative_edge"])
        for metric in CALIBRATION_METRICS
    }
    return {
        "scoring": {
            "positive_control_minimum_ipsae_min": positive["ipsae_min"],
            "positive_control_minimum_sc_dockq": positive["sc_dockq"],
            "positive_control_minimum_site_contact_iou": positive["site_contact_iou"],
            "positive_control_minimum_target_contact_recall": positive["target_contact_recall"],
            "negative_control_maximum_ipsae_min": negative["ipsae_min"],
        },
        "controls": {
            "positive": positive,
            "negative": {
                "ipsae_min": negative["ipsae_min"],
                "site_contact_iou": negative["site_contact_iou"],
            },
        },
    }


def _provenance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    absent: list[str] = []
    for field in PROVENANCE_FIELDS:
        present_values = [row["raw"][field] for row in rows if field in row["raw"]]
        if not present_values:
            fields[field] = {"status": "absent", "values": []}
            absent.append(field)
            continue
        values = _unique_json_values(present_values)
        fields[field] = {"status": "present", "values": values}
    return {
        "fields": fields,
        "absent_fields": absent,
        "scope_source": {
            "field": None,
            "status": "absent",
            "value_source": "scope_by_key argument",
        },
    }


def _unique_json_values(values: Sequence[Any]) -> list[Any]:
    unique: dict[str, Any] = {}
    for value in values:
        key = json.dumps(value, sort_keys=True, separators=(",", ":"))
        unique.setdefault(key, value)
    return [unique[key] for key in sorted(unique)]


def _reducer_descriptor(metric: str, reducer: Reducer | None) -> dict[str, Any]:
    if metric == PUBLISHED_IPSAE_METRIC:
        return {
            "kind": "published",
            "name": "lower-positive-edge-upper-negative-edge",
            "status": "resolved",
        }
    if reducer is _product_default_edge:
        return {
            "kind": "product-default",
            "name": "lower-positive-edge-upper-negative-edge",
            "status": "resolved",
        }
    if isinstance(reducer, str):
        return {
            "kind": "explicit",
            "name": reducer,
            "status": "proposed-product-rule-awaiting-decision",
        }
    return {
        "kind": "explicit",
        "name": getattr(reducer, "__name__", "callable"),
        "status": "proposed-product-rule-awaiting-decision",
    }


def _product_default_edge(values: Sequence[float], polarity: str) -> float:
    return min(values) if polarity == "positive" else max(values)


def _apply_reducer(reducer: Reducer, values: Sequence[float], polarity: str) -> float:
    if not values:
        raise CalibrationBandError(f"cannot reduce empty {polarity} values")
    if isinstance(reducer, str):
        if reducer == "minimum":
            result = min(values)
        elif reducer == "maximum":
            result = max(values)
        elif reducer == "mean":
            result = statistics.fmean(values)
        elif reducer == "median":
            result = statistics.median(values)
        else:
            raise CalibrationBandError(
                f"unsupported reducer {reducer!r} for metric with polarity {polarity}"
            )
    elif callable(reducer):
        try:
            parameter_count = len(inspect.signature(reducer).parameters)
        except (TypeError, ValueError):
            parameter_count = 2
        result = reducer(values, polarity) if parameter_count >= 2 else reducer(values)
    else:
        raise CalibrationBandError(f"reducer for {polarity} values is not callable or named")
    if not _is_finite_number(result):
        raise CalibrationBandError(f"reducer returned a non-finite {polarity} edge")
    return float(result)


def _band_entry(
    metric: str,
    positive_values: Sequence[float],
    negative_values: Sequence[float],
    reducers: Mapping[str, Reducer],
) -> dict[str, Any]:
    if metric == PUBLISHED_IPSAE_METRIC:
        positive_edge = min(positive_values)
        negative_edge = max(negative_values)
    else:
        reducer = reducers[metric]
        positive_edge = _apply_reducer(reducer, positive_values, "positive")
        negative_edge = _apply_reducer(reducer, negative_values, "negative")
    return {
        "positive_edge": float(positive_edge),
        "negative_edge": float(negative_edge),
        "positive_member_values": [float(value) for value in positive_values],
        "negative_member_values": [float(value) for value in negative_values],
        "reducer": _reducer_descriptor(metric, reducers.get(metric)),
    }


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
