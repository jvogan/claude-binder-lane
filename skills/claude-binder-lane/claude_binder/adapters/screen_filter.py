#!/usr/bin/env python3
"""Screen painted binder complexes before any co-folding provider call.

The adapter uses local structure files, CPU calculations, and model scores that
the generator already wrote. It never downloads weights, starts a GPU, or calls
an external service. Every configured gate produces one observation for every
candidate, including candidates rejected by an earlier integrity failure.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

from .integrity_filter import (
    AdapterError,
    evaluate,
    read_fasta,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_sequence,
    write_json,
    write_jsonl,
)
from . import screen_geometry
from ..filter_contracts import (
    SCREEN_THRESHOLD_SOURCES,
    filter_report,
    reference_digest_is_required,
    screen_reason,
)


SCREEN_ADAPTER_REVISION = "cheap-screen-cpu-v1"
SCREEN_MANIFEST_SCHEMA_VERSION = 1
STAGE_SCREEN = "filter-screen"
SHA256_RE = "0123456789abcdef"
REQUIRED_TIER_ZERO_METRICS = frozenset(
    {"parse_validity", "target_identity_rmsd_angstrom", "hotspot_map_present"}
)
GEOMETRY_METRICS = frozenset(
    {
        "hard_clash_count",
        "hotspot_coverage_count",
        "interface_buried_sasa_angstrom2",
        "interface_apolar_fraction",
        "contact_density",
    }
)
SECONDARY_STRUCTURE_METRICS = frozenset(
    {"ordered_fraction", "terminal_tail_length", "radius_of_gyration_angstrom"}
)
MPNN_METRICS = frozenset(
    {"mpnn_interface_mean_log_probability", "mpnn_interface_p10_log_probability"}
)
COMPOSITION_METRICS = frozenset(
    {
        "net_charge_proxy",
        "cysteine_even",
        "aggregation_window_hydrophobic_fraction",
        "composition_pass",
    }
)
ESM2_METRICS = frozenset({"esm2_pseudo_log_likelihood"})
SUPPORTED_METRICS = frozenset(
    {
        *REQUIRED_TIER_ZERO_METRICS,
        *GEOMETRY_METRICS,
        *SECONDARY_STRUCTURE_METRICS,
        *MPNN_METRICS,
        *COMPOSITION_METRICS,
        *ESM2_METRICS,
    }
)
HYDROPHOBIC_AMINO_ACIDS = frozenset("AVILMFWY")


class ScreenError(RuntimeError):
    """The pre-fold screen input or declared measurement cannot be used."""


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "__REQUIRED__" in value:
        raise ScreenError(f"{label} must be a non-empty resolved string")
    return value


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ScreenError(f"{label} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ScreenError(f"{label} must be at least {minimum:g}")
    return number


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ScreenError(f"{label} must be a positive integer")
    return value


def _path(value: Any, config_path: Path, label: str) -> Path:
    raw = Path(_nonempty_string(value, label)).expanduser()
    resolved = raw.resolve() if raw.is_absolute() else (config_path.parent / raw).resolve()
    if not resolved.is_file():
        raise ScreenError(f"{label} is missing: {resolved}")
    return resolved


def _screen_block(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(config.get("cheap_screen"), Mapping):
        return config["cheap_screen"]
    filters = config.get("filters")
    if isinstance(filters, Mapping) and isinstance(filters.get("cheap_screen"), Mapping):
        return filters["cheap_screen"]
    raise ScreenError("campaign config is missing cheap_screen")


def _validate_contracts(screen: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_contracts = screen.get("contracts")
    if not isinstance(raw_contracts, list) or not raw_contracts:
        raise ScreenError("cheap_screen.contracts must be a non-empty list")
    contracts: list[dict[str, Any]] = []
    gate_ids: set[str] = set()
    metrics: set[str] = set()
    for index, raw in enumerate(raw_contracts):
        if not isinstance(raw, Mapping):
            raise ScreenError(f"cheap_screen.contracts[{index}] must be an object")
        gate_id = _nonempty_string(raw.get("gate_id"), f"cheap_screen.contracts[{index}].gate_id")
        if gate_id in gate_ids:
            raise ScreenError(f"cheap_screen.contracts has duplicate gate_id: {gate_id}")
        metric = _nonempty_string(raw.get("metric"), f"cheap_screen.contracts[{index}].metric")
        if metric not in SUPPORTED_METRICS:
            raise ScreenError(f"cheap_screen.contracts[{index}].metric is not supported: {metric}")
        operator = raw.get("operator")
        if operator not in {"minimum", "maximum"}:
            raise ScreenError(f"cheap_screen.contracts[{index}].operator must be minimum or maximum")
        threshold = _finite_number(raw.get("threshold"), f"cheap_screen.contracts[{index}].threshold")
        threshold_source = raw.get("threshold_source")
        if threshold_source not in SCREEN_THRESHOLD_SOURCES:
            allowed = ", ".join(sorted(SCREEN_THRESHOLD_SOURCES))
            raise ScreenError(
                f"cheap_screen.contracts[{index}].threshold_source must be one of {allowed}"
            )
        tool_revision = _nonempty_string(
            raw.get("tool_revision"), f"cheap_screen.contracts[{index}].tool_revision"
        )
        reference_revision = _nonempty_string(
            raw.get("reference_revision"),
            f"cheap_screen.contracts[{index}].reference_revision",
        )
        reference_sha256 = raw.get("reference_sha256")
        if reference_digest_is_required(reference_revision):
            if (
                not isinstance(reference_sha256, str)
                or len(reference_sha256) != 64
                or any(character not in SHA256_RE for character in reference_sha256)
            ):
                raise ScreenError(
                    f"cheap_screen.contracts[{index}].reference_sha256 must be a lowercase SHA-256"
                )
        elif not isinstance(reference_sha256, str) or not reference_sha256.startswith("not_applicable: "):
            raise ScreenError(
                f"cheap_screen.contracts[{index}].reference_sha256 must explain the local reference"
            )
        relaxation = raw.get("relaxation")
        if relaxation is not None:
            if not isinstance(relaxation, Mapping):
                raise ScreenError(f"cheap_screen.contracts[{index}].relaxation must be an object")
            relaxation = {
                "previous_threshold": _finite_number(
                    relaxation.get("previous_threshold"),
                    f"cheap_screen.contracts[{index}].relaxation.previous_threshold",
                ),
                "author": _nonempty_string(
                    relaxation.get("author"), f"cheap_screen.contracts[{index}].relaxation.author"
                ),
                "reason": _nonempty_string(
                    relaxation.get("reason"), f"cheap_screen.contracts[{index}].relaxation.reason"
                ),
            }
        contracts.append(
            {
                "gate_id": gate_id,
                "filter_id": gate_id,
                "metric": metric,
                "operator": operator,
                "threshold": threshold,
                "threshold_source": threshold_source,
                "tool_revision": tool_revision,
                "reference_revision": reference_revision,
                "reference_sha256": reference_sha256,
                "relaxation": relaxation,
            }
        )
        gate_ids.add(gate_id)
        metrics.add(metric)
    missing = REQUIRED_TIER_ZERO_METRICS - metrics
    if missing:
        raise ScreenError(
            "cheap_screen.contracts is missing tier-zero metrics: " + ", ".join(sorted(missing))
        )
    if metrics & ESM2_METRICS:
        raise ScreenError(
            "cheap_screen cannot enable esm2_pseudo_log_likelihood until a measured CPU speed probe "
            "is registered. The adapter never downloads ESM-2 weights or calls a provider."
        )
    return contracts


def _validate_target(screen: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    raw = screen.get("target")
    if not isinstance(raw, Mapping):
        raise ScreenError("cheap_screen.target must be an object")
    minimum = _positive_int(raw.get("binder_minimum_length"), "cheap_screen.target.binder_minimum_length")
    maximum = _positive_int(raw.get("binder_maximum_length"), "cheap_screen.target.binder_maximum_length")
    if maximum < minimum:
        raise ScreenError("cheap_screen.target.binder_maximum_length must be at least binder_minimum_length")
    labels = raw.get("hotspot_labels")
    if not isinstance(labels, list) or not labels:
        raise ScreenError("cheap_screen.target.hotspot_labels must be a non-empty list")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ScreenError("cheap_screen.target.hotspot_labels must contain non-empty strings")
    return {
        "structure_path": _path(raw.get("structure_path"), config_path, "cheap_screen.target.structure_path"),
        "residue_map_path": _path(raw.get("residue_map_path"), config_path, "cheap_screen.target.residue_map_path"),
        "target_chain_id": _nonempty_string(raw.get("target_chain_id"), "cheap_screen.target.target_chain_id"),
        "binder_chain_id": _nonempty_string(raw.get("binder_chain_id"), "cheap_screen.target.binder_chain_id"),
        "target_residue_count": _positive_int(
            raw.get("target_residue_count"), "cheap_screen.target.target_residue_count"
        ),
        "binder_minimum_length": minimum,
        "binder_maximum_length": maximum,
        "hotspot_labels": tuple(labels),
    }


def _validate_geometry(screen: Mapping[str, Any]) -> dict[str, Any]:
    raw = screen.get("geometry")
    if not isinstance(raw, Mapping):
        raise ScreenError("cheap_screen.geometry must be an object when geometry gates are enabled")
    raw_radii = raw.get("van_der_waals_radii_angstrom")
    if not isinstance(raw_radii, Mapping) or not raw_radii:
        raise ScreenError("cheap_screen.geometry.van_der_waals_radii_angstrom must be a non-empty object")
    radii = {
        _nonempty_string(element, "cheap_screen.geometry.van_der_waals_radii_angstrom key").upper(): _finite_number(
            radius,
            f"cheap_screen.geometry.van_der_waals_radii_angstrom.{element}",
            minimum=0.0,
        )
        for element, radius in raw_radii.items()
    }
    if any(radius == 0 for radius in radii.values()):
        raise ScreenError("cheap_screen.geometry.van_der_waals_radii_angstrom values must be positive")
    apolar = raw.get("apolar_elements")
    if not isinstance(apolar, list) or not apolar or any(not isinstance(item, str) or not item for item in apolar):
        raise ScreenError("cheap_screen.geometry.apolar_elements must be a non-empty string list")
    return {
        "radii_angstrom": radii,
        "clash_tolerance_angstrom": _finite_number(
            raw.get("clash_tolerance_angstrom"),
            "cheap_screen.geometry.clash_tolerance_angstrom",
            minimum=0.0,
        ),
        "contact_cutoff_angstrom": _finite_number(
            raw.get("hotspot_contact_cutoff_angstrom"),
            "cheap_screen.geometry.hotspot_contact_cutoff_angstrom",
            minimum=0.0,
        ),
        "sasa_probe_radius_angstrom": _finite_number(
            raw.get("sasa_probe_radius_angstrom"),
            "cheap_screen.geometry.sasa_probe_radius_angstrom",
            minimum=0.0,
        ),
        "sasa_sphere_point_count": _positive_int(
            raw.get("sasa_sphere_point_count"), "cheap_screen.geometry.sasa_sphere_point_count"
        ),
        "apolar_elements": frozenset(item.upper() for item in apolar),
    }


def _validate_composition(screen: Mapping[str, Any]) -> dict[str, Any]:
    raw = screen.get("composition")
    if not isinstance(raw, Mapping):
        raise ScreenError("cheap_screen.composition must be an object when aggregation proxy is enabled")
    minimum_charge = _finite_number(
        raw.get("minimum_net_charge_proxy"),
        "cheap_screen.composition.minimum_net_charge_proxy",
    )
    maximum_charge = _finite_number(
        raw.get("maximum_net_charge_proxy"),
        "cheap_screen.composition.maximum_net_charge_proxy",
    )
    if maximum_charge < minimum_charge:
        raise ScreenError(
            "cheap_screen.composition.maximum_net_charge_proxy must be at least minimum_net_charge_proxy"
        )
    maximum_aggregation = _finite_number(
        raw.get("maximum_aggregation_window_hydrophobic_fraction"),
        "cheap_screen.composition.maximum_aggregation_window_hydrophobic_fraction",
        minimum=0.0,
    )
    if maximum_aggregation > 1:
        raise ScreenError(
            "cheap_screen.composition.maximum_aggregation_window_hydrophobic_fraction must be at most 1"
        )
    require_even = raw.get("require_even_cysteine_count")
    if not isinstance(require_even, bool):
        raise ScreenError("cheap_screen.composition.require_even_cysteine_count must be boolean")
    return {
        "aggregation_window_length": _positive_int(
            raw.get("aggregation_window_length"),
            "cheap_screen.composition.aggregation_window_length",
        ),
        "minimum_net_charge_proxy": minimum_charge,
        "maximum_net_charge_proxy": maximum_charge,
        "maximum_aggregation_window_hydrophobic_fraction": maximum_aggregation,
        "require_even_cysteine_count": require_even,
    }


def _validate_ranking(screen: Mapping[str, Any], contracts: list[dict[str, Any]]) -> list[dict[str, str]]:
    raw = screen.get("ranking")
    if not isinstance(raw, Mapping):
        raise ScreenError("cheap_screen.ranking must declare the pre-calibration ordering")
    if raw.get("kind") != "lexicographic":
        raise ScreenError("cheap_screen.ranking.kind must be lexicographic until calibration supplies weights")
    features = raw.get("features")
    if not isinstance(features, list) or not features:
        raise ScreenError("cheap_screen.ranking.features must be a non-empty list")
    contract_metrics = {contract["metric"] for contract in contracts}
    ranking: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw_feature in enumerate(features):
        if not isinstance(raw_feature, Mapping):
            raise ScreenError(f"cheap_screen.ranking.features[{index}] must be an object")
        metric = _nonempty_string(raw_feature.get("metric"), f"cheap_screen.ranking.features[{index}].metric")
        if metric not in contract_metrics:
            raise ScreenError(
                f"cheap_screen.ranking.features[{index}].metric must have a screen contract: {metric}"
            )
        if metric in seen:
            raise ScreenError(f"cheap_screen.ranking.features has duplicate metric: {metric}")
        direction = raw_feature.get("direction")
        if direction not in {"ascending", "descending"}:
            raise ScreenError(f"cheap_screen.ranking.features[{index}].direction must be ascending or descending")
        ranking.append({"metric": metric, "direction": direction})
        seen.add(metric)
    return ranking


def resolve_screen_config(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    """Validate every free screen convention before parsing a candidate."""
    screen = _screen_block(config)
    contracts = _validate_contracts(screen)
    metrics = {contract["metric"] for contract in contracts}
    target = _validate_target(screen, config_path)
    geometry = _validate_geometry(screen) if metrics & (GEOMETRY_METRICS | MPNN_METRICS) else None
    composition = _validate_composition(screen) if metrics & {
        "aggregation_window_hydrophobic_fraction",
        "composition_pass",
    } else None
    fold_quota = _positive_int(screen.get("fold_quota"), "cheap_screen.fold_quota")
    campaign_id = config.get("campaign_id", screen.get("campaign_id"))
    return {
        "campaign_id": _nonempty_string(campaign_id, "campaign_id or cheap_screen.campaign_id"),
        "contracts": contracts,
        "target": target,
        "geometry": geometry,
        "composition": composition,
        "ranking": _validate_ranking(screen, contracts),
        "fold_quota": fold_quota,
        "model_signals_path": (
            _path(screen["model_signals_path"], config_path, "cheap_screen.model_signals_path")
            if "model_signals_path" in screen
            else None
        ),
    }


def _resolved_hotspots(target: Mapping[str, Any]) -> tuple[frozenset[str], str]:
    """Resolve every named hotspot and return the map digest for each manifest row."""
    try:
        document = json.loads(Path(target["residue_map_path"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScreenError(f"residue map is unreadable: {target['residue_map_path']}: {exc}") from exc
    mapping = document.get("source_to_cleaned") if isinstance(document, Mapping) else None
    if not isinstance(mapping, Mapping) or not mapping:
        raise ScreenError("residue map has no non-empty source_to_cleaned object")
    target_structure = screen_geometry.read_structure(str(target["structure_path"]))
    target_labels = {residue.label for residue in screen_geometry.chain_residues(target_structure, target["target_chain_id"])}
    resolved: set[str] = set()
    unresolved: list[str] = []
    for source_label in target["hotspot_labels"]:
        cleaned = mapping.get(source_label)
        if not isinstance(cleaned, str) or cleaned not in target_labels:
            unresolved.append(str(source_label))
        else:
            resolved.add(cleaned)
    if unresolved:
        raise ScreenError(
            "MAP_UNRESOLVED: residue map cannot resolve hotspot labels: " + ", ".join(sorted(unresolved))
        )
    return frozenset(resolved), sha256_file(Path(target["residue_map_path"]))


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path, "candidate manifest")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ScreenError(f"candidate manifest row has no candidate_id: {path}")
        if candidate_id in seen:
            raise ScreenError(f"candidate manifest has duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        origin = row.get("origin_generator")
        if not isinstance(origin, str) or not origin:
            raise ScreenError(f"candidate {candidate_id} has no origin_generator")
        sequence_path = row.get("sequence_path")
        if not isinstance(sequence_path, str) or not Path(sequence_path).is_absolute():
            raise ScreenError(f"candidate {candidate_id} sequence_path must be an absolute file path")
        sequence_file = Path(sequence_path)
        if not sequence_file.is_file():
            raise ScreenError(f"candidate {candidate_id} sequence_path is missing: {sequence_file}")
        sequence = read_fasta(sequence_file, candidate_id)
        if sha256_sequence(sequence) != row.get("sequence_sha256"):
            raise ScreenError(f"candidate {candidate_id} sequence_sha256 does not match its FASTA")
        if row.get("sequence_length") != len(sequence):
            raise ScreenError(f"candidate {candidate_id} sequence_length does not match its FASTA")
        pose_path = row.get("design_pose_path")
        if not isinstance(pose_path, str) or not Path(pose_path).is_absolute():
            raise ScreenError(f"candidate {candidate_id} design_pose_path must be an absolute file path")
        pose_file = Path(pose_path)
        if not pose_file.is_file():
            raise ScreenError(f"candidate {candidate_id} design_pose_path is missing: {pose_file}")
        if sha256_file(pose_file) != row.get("design_pose_sha256"):
            raise ScreenError(f"candidate {candidate_id} design_pose_sha256 does not match its structure")
        optimization_round = row.get("optimization_round")
        if isinstance(optimization_round, bool) or not isinstance(optimization_round, int) or optimization_round < 0:
            raise ScreenError(f"candidate {candidate_id} has invalid optimization_round")
        candidates.append({**row, "_sequence": sequence, "_pose_path": pose_file})
    return candidates


def _load_model_signals(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = read_jsonl(path, "cheap-screen model signals")
    signals: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ScreenError(f"model signal row has no candidate_id: {path}")
        if candidate_id in signals:
            raise ScreenError(f"model signals have duplicate candidate_id: {candidate_id}")
        signals[candidate_id] = row
    return signals


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ScreenError("cannot calculate a quantile of no MPNN positions")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _composition_features(sequence: str, composition: Mapping[str, Any] | None) -> dict[str, Any]:
    features: dict[str, Any] = {
        "net_charge_proxy": float(sequence.count("K") + sequence.count("R") - sequence.count("D") - sequence.count("E")),
        "cysteine_even": 1.0 if sequence.count("C") % 2 == 0 else 0.0,
        "cysteine_count": float(sequence.count("C")),
        "sequence_recovery_fraction": None,
        "aggregation_window_hydrophobic_fraction": None,
    }
    if composition is not None:
        window_length = int(composition["aggregation_window_length"])
        effective_length = min(window_length, len(sequence))
        if effective_length:
            fractions = [
                sum(residue in HYDROPHOBIC_AMINO_ACIDS for residue in sequence[start : start + effective_length])
                / effective_length
                for start in range(len(sequence) - effective_length + 1)
            ]
            features["aggregation_window_hydrophobic_fraction"] = max(fractions)
        flags: list[str] = []
        charge = float(features["net_charge_proxy"])
        if charge < composition["minimum_net_charge_proxy"]:
            flags.append("net_charge_below_minimum")
        if charge > composition["maximum_net_charge_proxy"]:
            flags.append("net_charge_above_maximum")
        if composition["require_even_cysteine_count"] and features["cysteine_even"] != 1.0:
            flags.append("odd_cysteine_count")
        aggregation = features["aggregation_window_hydrophobic_fraction"]
        if (
            isinstance(aggregation, (int, float))
            and aggregation > composition["maximum_aggregation_window_hydrophobic_fraction"]
        ):
            flags.append("aggregation_window_above_maximum")
        features["composition_flags"] = flags
        features["composition_pass"] = 1.0 if not flags else 0.0
    return features


def _mpnn_features(
    signal: Mapping[str, Any] | None,
    sequence: str,
    binder_interface_indices: tuple[int, ...],
) -> tuple[dict[str, float | None], str | None]:
    features: dict[str, float | None] = {
        "mpnn_interface_mean_log_probability": None,
        "mpnn_interface_p10_log_probability": None,
        "sequence_recovery_fraction": None,
    }
    if signal is None:
        return features, "model signal is absent"
    raw_probabilities = signal.get("mpnn_position_log_probabilities")
    if not isinstance(raw_probabilities, list) or len(raw_probabilities) != len(sequence):
        return features, "mpnn_position_log_probabilities must match the binder sequence length"
    try:
        probabilities = [
            _finite_number(value, "mpnn_position_log_probabilities") for value in raw_probabilities
        ]
    except ScreenError as exc:
        return features, str(exc)
    if not binder_interface_indices:
        return features, "painted complex has no binder interface positions"
    interface_probabilities = [probabilities[index] for index in binder_interface_indices]
    features["mpnn_interface_mean_log_probability"] = sum(interface_probabilities) / len(interface_probabilities)
    features["mpnn_interface_p10_log_probability"] = _quantile(interface_probabilities, 0.1)
    mpnn_sequence = signal.get("mpnn_sequence")
    if isinstance(mpnn_sequence, str) and len(mpnn_sequence) == len(sequence):
        features["sequence_recovery_fraction"] = sum(
            left == right for left, right in zip(sequence, mpnn_sequence)
        ) / len(sequence)
    return features, None


def _metric_reason(metric: str) -> str:
    if metric == "parse_validity":
        return "PARSE_FAIL"
    if metric == "hotspot_map_present":
        return "MAP_UNRESOLVED"
    if metric == "hard_clash_count":
        return "CLASH_EXCESS"
    if metric == "hotspot_coverage_count":
        return "HOTSPOT_COVERAGE_LOW"
    if metric in {"interface_buried_sasa_angstrom2", "interface_apolar_fraction", "contact_density"}:
        return "BSA_LOW"
    if metric in COMPOSITION_METRICS:
        return "AGGREGATION_FLAG" if metric == "aggregation_window_hydrophobic_fraction" else "COMPOSITION_FLAG"
    if metric in MPNN_METRICS:
        return "MPNN_PERCENTILE_LOW"
    if metric in SECONDARY_STRUCTURE_METRICS:
        return "SS_IRREGULAR"
    if metric in ESM2_METRICS:
        return "ESM_PLL_LOW"
    # Target identity is a malformed painted complex for this fixed target. The
    # published reason vocabulary groups that integrity failure under PARSE_FAIL.
    return "PARSE_FAIL"


def _observation(
    candidate: Mapping[str, Any],
    contract: Mapping[str, Any],
    value: float | None,
    passed: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "origin_generator": candidate["origin_generator"],
        "sequence_sha256": candidate["sequence_sha256"],
        "optimization_round": candidate["optimization_round"],
        "filter_id": contract["gate_id"],
        "metric": contract["metric"],
        "operator": contract["operator"],
        "threshold": contract["threshold"],
        "threshold_source": contract["threshold_source"],
        "value": value,
        "pass": passed,
        "reason": reason,
        "tool_revision": contract["tool_revision"],
        "reference_revision": contract["reference_revision"],
        "reference_sha256": contract["reference_sha256"],
    }


def _passes(value: float | None, contract: Mapping[str, Any]) -> bool:
    if value is None or not math.isfinite(value):
        return False
    try:
        return evaluate(value, dict(contract))
    except AdapterError as exc:
        raise ScreenError(str(exc)) from exc


def _parse_features(
    structure: Any,
    target: Mapping[str, Any],
) -> tuple[dict[str, float | None], str | None, Any, Any]:
    features: dict[str, float | None] = {
        "parse_validity": 0.0,
        "target_identity_rmsd_angstrom": None,
        "hotspot_map_present": None,
    }
    chains = {
        residue.auth_chain or residue.label_chain
        for residue in structure.residues
        if residue.auth_chain or residue.label_chain
    }
    if len(chains) != 2:
        return features, f"painted complex has {len(chains)} chains; expected two", None, None
    try:
        target_residues = screen_geometry.chain_residues(structure, target["target_chain_id"])
        binder_residues = screen_geometry.chain_residues(structure, target["binder_chain_id"])
    except screen_geometry.ScreenGeometryError as exc:
        return features, str(exc), None, None
    binder_length = len(binder_residues)
    target_length = len(target_residues)
    features["binder_length"] = float(binder_length)
    features["target_residue_count"] = float(target_length)
    if not target["binder_minimum_length"] <= binder_length <= target["binder_maximum_length"]:
        return (
            features,
            f"binder length {binder_length} is outside {target['binder_minimum_length']}-{target['binder_maximum_length']}",
            target_residues,
            binder_residues,
        )
    if target_length != target["target_residue_count"]:
        return (
            features,
            f"target length {target_length} differs from {target['target_residue_count']}",
            target_residues,
            binder_residues,
        )
    features["parse_validity"] = 1.0
    return features, None, target_residues, binder_residues


def _candidate_result(
    candidate: Mapping[str, Any],
    settings: Mapping[str, Any],
    input_target_residues: Any,
    hotspot_labels: frozenset[str],
    map_sha256: str,
    signals: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    features = _composition_features(candidate["_sequence"], settings["composition"])
    parse_error: str | None = None
    target_residues = None
    binder_residues = None
    try:
        structure = screen_geometry.read_structure(str(candidate["_pose_path"]))
        parse_features, parse_error, target_residues, binder_residues = _parse_features(
            structure, settings["target"]
        )
        features.update(parse_features)
    except screen_geometry.ScreenGeometryError as exc:
        features.update(
            {
                "parse_validity": 0.0,
                "target_identity_rmsd_angstrom": None,
                "hotspot_map_present": None,
            }
        )
        parse_error = str(exc)
    if parse_error is None and target_residues is not None and binder_residues is not None:
        try:
            features["target_identity_rmsd_angstrom"] = screen_geometry.target_ca_rmsd_angstrom(
                target_residues, input_target_residues
            )
        except screen_geometry.ScreenGeometryError as exc:
            features["target_identity_rmsd_angstrom"] = None
            parse_error = str(exc)
    features["hotspot_map_present"] = (
        1.0 if candidate.get("residue_map_sha256") == map_sha256 else 0.0
    )
    geometry_error: str | None = None
    geometry = settings["geometry"]
    if parse_error is None and geometry is not None and target_residues is not None and binder_residues is not None:
        try:
            calculated = screen_geometry.interface_geometry(
                target_residues,
                binder_residues,
                hotspot_labels,
                **geometry,
            )
            features.update(
                {
                    "hard_clash_count": float(calculated.hard_clash_count),
                    "hotspot_coverage_count": float(calculated.hotspot_coverage_count),
                    "hotspot_count": float(calculated.hotspot_count),
                    "interface_buried_sasa_angstrom2": calculated.interface_buried_sasa_angstrom2,
                    "interface_apolar_fraction": calculated.interface_apolar_fraction,
                    "contact_density": calculated.contact_density,
                    "interchain_residue_contact_count": float(calculated.interchain_residue_contact_count),
                    "interface_residue_count": float(calculated.interface_residue_count),
                }
            )
            mpnn_features, mpnn_error = _mpnn_features(
                signals.get(candidate["candidate_id"]),
                candidate["_sequence"],
                calculated.binder_interface_indices,
            )
            features.update(mpnn_features)
            if mpnn_error is not None:
                features["mpnn_error"] = mpnn_error
        except screen_geometry.ScreenGeometryError as exc:
            geometry_error = str(exc)
    if parse_error is None and any(
        contract["metric"] in SECONDARY_STRUCTURE_METRICS for contract in settings["contracts"]
    ):
        try:
            features.update(
                screen_geometry.secondary_structure_features(
                    str(candidate["_pose_path"]), settings["target"]["binder_chain_id"]
                )
            )
        except screen_geometry.ScreenGeometryError as exc:
            features["secondary_structure_error"] = str(exc)
    observations: list[dict[str, Any]] = []
    for contract in settings["contracts"]:
        metric = contract["metric"]
        value = features.get(metric)
        numeric_value = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        if parse_error is not None and metric != "parse_validity":
            passed = False
            reason = screen_reason("PARSE_FAIL", parse_error)
        elif metric == "parse_validity" and parse_error is not None:
            passed = False
            code = "LENGTH_MISMATCH" if "length" in parse_error else "PARSE_FAIL"
            reason = screen_reason(code, parse_error)
        elif metric == "hotspot_map_present" and numeric_value != 1.0:
            passed = False
            reason = screen_reason(
                "MAP_UNRESOLVED",
                f"candidate residue_map_sha256 does not match {map_sha256}",
            )
        elif numeric_value is None:
            passed = False
            detail = geometry_error or str(features.get("mpnn_error") or features.get("secondary_structure_error") or "metric was not calculated")
            reason = screen_reason(_metric_reason(metric), detail)
        else:
            passed = _passes(numeric_value, contract)
            if passed:
                relaxation = contract.get("relaxation")
                if isinstance(relaxation, Mapping):
                    reason = screen_reason(
                        "GATE_RELAXED",
                        f"{contract['gate_id']} threshold changed from {relaxation['previous_threshold']:g} "
                        f"by {relaxation['author']}: {relaxation['reason']}",
                    )
                else:
                    reason = f"{metric}={numeric_value:g} met {contract['operator']} {contract['threshold']:g}"
            else:
                reason = screen_reason(
                    _metric_reason(metric),
                    f"{metric}={numeric_value:g} did not meet {contract['operator']} {contract['threshold']:g}",
                )
        observations.append(_observation(candidate, contract, numeric_value, passed, reason))
    failed = [observation for observation in observations if not observation["pass"]]
    disposition = {
        "code": "REJECTED" if failed else "PENDING_RANK",
        "reason": failed[0]["reason"] if failed else "passed every configured hard gate",
    }
    manifest = {
        "schema_version": SCREEN_MANIFEST_SCHEMA_VERSION,
        "design_id": candidate["candidate_id"],
        "campaign_id": settings["campaign_id"],
        "round": candidate["optimization_round"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "origin_generator": candidate["origin_generator"],
            "generator_seed": candidate.get("generator_seed"),
            "generator_call_id": candidate.get("generator_call_id"),
            "screen_adapter_revision": SCREEN_ADAPTER_REVISION,
        },
        "inputs": {
            "target_structure_sha256": settings["target_structure_sha256"],
            "hotspot_residue_map_sha256": map_sha256,
        },
        "features": features,
        "gates": observations,
        "disposition": disposition,
        "artifacts": {
            "complex_path": str(candidate["_pose_path"]),
            "complex_sha256": candidate["design_pose_sha256"],
            "sequence_path": candidate["sequence_path"],
            "sequence_sha256": candidate["sequence_sha256"],
        },
        "_candidate": candidate,
    }
    return manifest, observations


def _ranking_key(manifest: Mapping[str, Any], ranking: list[dict[str, str]]) -> tuple[Any, ...]:
    key: list[Any] = []
    features = manifest["features"]
    for item in ranking:
        value = features.get(item["metric"])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ScreenError(
                f"candidate {manifest['design_id']} passed every gate but has no numeric ranking value for {item['metric']}"
            )
        key.append(-float(value) if item["direction"] == "descending" else float(value))
    key.append(str(manifest["design_id"]))
    return tuple(key)


def run_screen(candidate_manifest: Path, config_path: Path, out_dir: Path, model_signals: Path | None = None) -> dict[str, Any]:
    """Run the local screen and write retained and rejected candidate records."""
    config = read_json(config_path, "campaign config")
    if not isinstance(config, Mapping):
        raise ScreenError(f"campaign config must be a JSON object: {config_path}")
    settings = resolve_screen_config(config, config_path)
    hotspot_labels, map_sha256 = _resolved_hotspots(settings["target"])
    input_structure = screen_geometry.read_structure(str(settings["target"]["structure_path"]))
    input_target_residues = screen_geometry.chain_residues(
        input_structure, settings["target"]["target_chain_id"]
    )
    if len(input_target_residues) != settings["target"]["target_residue_count"]:
        raise ScreenError(
            "input target residue count "
            f"{len(input_target_residues)} differs from cheap_screen.target.target_residue_count "
            f"{settings['target']['target_residue_count']}"
        )
    settings["target_structure_sha256"] = sha256_file(settings["target"]["structure_path"])
    candidates = _load_candidates(candidate_manifest)
    signal_path = model_signals or settings["model_signals_path"]
    signals = _load_model_signals(signal_path)
    manifests: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for candidate in candidates:
        manifest, candidate_observations = _candidate_result(
            candidate,
            settings,
            input_target_residues,
            hotspot_labels,
            map_sha256,
            signals,
        )
        manifests.append(manifest)
        observations.extend(candidate_observations)
    finalists = [manifest for manifest in manifests if manifest["disposition"]["code"] == "PENDING_RANK"]
    finalists.sort(key=lambda manifest: _ranking_key(manifest, settings["ranking"]))
    accepted_ids = {manifest["design_id"] for manifest in finalists[: settings["fold_quota"]]}
    for manifest in manifests:
        if manifest["disposition"]["code"] != "PENDING_RANK":
            continue
        if manifest["design_id"] in accepted_ids:
            manifest["disposition"] = {
                "code": "ACCEPTED_TO_FOLD",
                "reason": "passed every configured gate and ranked within the declared fold quota",
            }
        else:
            manifest["disposition"] = {
                "code": "RANKED_OUT",
                "reason": screen_reason("RANKED_OUT", "passed every gate but fell below the fold quota"),
            }
    output = out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    public_manifests = [
        {key: value for key, value in manifest.items() if key != "_candidate"} for manifest in manifests
    ]
    passing_rows = [
        {
            **{
                key: value
                for key, value in manifest["_candidate"].items()
                if not key.startswith("_")
            },
            "filter_pass": True,
            "failed_checks": [],
            "screen_disposition": manifest["disposition"],
        }
        for manifest in manifests
        if manifest["disposition"]["code"] == "ACCEPTED_TO_FOLD"
    ]
    write_jsonl(output / "screen-manifest.jsonl", public_manifests)
    write_jsonl(output / "screen-observations.jsonl", observations)
    write_jsonl(output / "screen-passing-candidates.jsonl", passing_rows)
    report = {
        "schema_version": SCREEN_MANIFEST_SCHEMA_VERSION,
        "stage_id": STAGE_SCREEN,
        "screen_adapter_revision": SCREEN_ADAPTER_REVISION,
        "candidate_count": len(candidates),
        "accepted_to_fold_count": len(passing_rows),
        "ranked_out_count": sum(
            manifest["disposition"]["code"] == "RANKED_OUT" for manifest in manifests
        ),
        "rejected_count": sum(
            manifest["disposition"]["code"] == "REJECTED" for manifest in manifests
        ),
        "fold_quota": settings["fold_quota"],
        "fold_shortfall": max(0, settings["fold_quota"] - len(passing_rows)),
        "ranking": settings["ranking"],
        "relaxations": [
            {
                "gate_id": contract["gate_id"],
                **contract["relaxation"],
            }
            for contract in settings["contracts"]
            if isinstance(contract.get("relaxation"), Mapping)
        ],
        "gate_report": filter_report(
            STAGE_SCREEN,
            [candidate["candidate_id"] for candidate in candidates],
            settings["contracts"],
            observations,
            [row["candidate_id"] for row in passing_rows],
        ),
    }
    write_json(output / "screen-report.json", report)
    return report


def parse_output(out_dir: Path) -> int:
    """Validate the local files a screen run publishes."""
    expected = (
        "screen-manifest.jsonl",
        "screen-observations.jsonl",
        "screen-passing-candidates.jsonl",
        "screen-report.json",
    )
    errors: list[str] = []
    for name in expected:
        path = out_dir.resolve() / name
        if not path.is_file():
            errors.append(f"declared output is missing: {path}")
            continue
        if path.suffix == ".jsonl":
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    if not isinstance(json.loads(line), dict):
                        errors.append(f"{path} line {line_number} is not a JSON object")
                except json.JSONDecodeError as exc:
                    errors.append(f"{path} line {line_number} is invalid JSON: {exc}")
        else:
            try:
                if not isinstance(json.loads(path.read_text(encoding="utf-8")), dict):
                    errors.append(f"{path} is not a JSON object")
            except json.JSONDecodeError as exc:
                errors.append(f"{path} is invalid JSON: {exc}")
    for error in errors:
        print(f"cheap screen parser: {error}", file=sys.stderr)
    print(f"cheap screen parser: out_dir={out_dir.resolve()} ok={not errors}")
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("toolcheck")
    run = subparsers.add_parser("run")
    run.add_argument("--candidate-manifest", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--model-signals", type=Path)
    parse = subparsers.add_parser("parse")
    parse.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "toolcheck":
            print("cheap screen ok, CPU-only numpy and local structure files")
            return 0
        if args.command == "parse":
            return parse_output(args.out_dir)
        report = run_screen(
            args.candidate_manifest.resolve(),
            args.config.resolve(),
            args.out_dir,
            args.model_signals.resolve() if args.model_signals is not None else None,
        )
        print(
            f"cheap screen: candidates={report['candidate_count']} accepted={report['accepted_to_fold_count']} "
            f"rejected={report['rejected_count']} ranked_out={report['ranked_out_count']}"
        )
        return 0
    except (AdapterError, OSError, ScreenError, ValueError, TypeError) as exc:
        print(f"cheap screen: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
