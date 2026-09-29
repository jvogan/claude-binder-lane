#!/usr/bin/env python3
"""Select screened candidates as optimization parents.

The selector reads the canonical screen table, the calibrated controls, and the
passing candidate manifest. It recomputes the configured score gates, applies the
published generator and diversity caps in ranked order, and writes the parent
manifest consumed by the optimization controller.

The selector freezes the published diversity rule. A supplied diversity block must
match that rule exactly, so policy drift cannot become an unrecorded choice. A
published promotion requires three configured predictor modes and preserves every
raw candidate score beside the pool-derived rank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from claude_binder import lane
from claude_binder import ranking_policy
from claude_binder import structural_surrogates
from claude_binder.adapters.adapter_io import read_jsonl as _read_jsonl

from .declared_artifacts import (
    DeclaredArtifactError,
    input_files,
    load_plan,
)
from ..selection_policy import (
    PUBLISHED_DIVERSITY_POLICY,
    REQUIRED_DIVERSITY_FIELDS,
    default_diversity_policy,
    explicit_single_arm_exception_errors,
)


OUTPUT_NAME = "promotion-manifest.jsonl"
PROMOTION_SUMMARY_NAME = "promotion-summary.json"
DEFAULT_MINIMUM_DELIVERY_FRACTION = 0.5
PUBLISHED_RANKING_MODE_COUNT = ranking_policy.PUBLISHED_RANKING_MODE_COUNT
PUBLISHED_RANKING_MODE = ranking_policy.PUBLISHED_RANKING_MODE
CANDIDATE_SINGLE_ARM_RANKING_MODE = ranking_policy.CANDIDATE_SINGLE_ARM_RANKING_MODE
CANDIDATE_SINGLE_LINEAGE_TWO_MODE_RANKING_MODE = (
    ranking_policy.CANDIDATE_SINGLE_LINEAGE_TWO_MODE_RANKING_MODE
)
CANDIDATE_SINGLE_LINEAGE_THREE_MODE_RANKING_MODE = (
    ranking_policy.CANDIDATE_SINGLE_LINEAGE_THREE_MODE_RANKING_MODE
)
CANDIDATE_TWO_INDEPENDENT_ARM_RANKING_MODE = (
    ranking_policy.CANDIDATE_TWO_INDEPENDENT_ARM_RANKING_MODE
)

LINEAGE_DIVERSITY_FIELDS = (
    "root_backbone_id",
    "tm90_cluster_id",
    "structure_method",
    "seq_method",
    "fold_class",
)
STRUCTURAL_DIVERSITY_CONFIG_KEY = "structural_diversity_surrogate"
STRUCTURAL_DIVERSITY_CLUSTER_FIELD = "tm90_cluster_id_surrogate"
STRUCTURAL_DIVERSITY_SECONDARY_FIELD = "secondary_structure_surrogate"


class AdapterError(RuntimeError):
    """An input or policy condition that must stop parent selection."""


def structural_diversity_policy(config: dict[str, Any]) -> dict[str, float] | None:
    """Return calibrated structural-diversity controls or the explicit disabled state."""
    selection = config.get("selection")
    if not isinstance(selection, dict):
        return None
    raw = selection.get(STRUCTURAL_DIVERSITY_CONFIG_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise AdapterError(f"selection.{STRUCTURAL_DIVERSITY_CONFIG_KEY} must be an object")
    if not isinstance(raw.get("enabled"), bool):
        raise AdapterError(f"selection.{STRUCTURAL_DIVERSITY_CONFIG_KEY}.enabled must be boolean")
    if raw["enabled"] is False:
        return None
    try:
        structural_surrogates.alignment_parameters_from_mapping(
            raw.get("alignment"), f"selection.{STRUCTURAL_DIVERSITY_CONFIG_KEY}.alignment"
        )
    except structural_surrogates.StructuralSurrogateError as exc:
        raise AdapterError(str(exc)) from exc
    result: dict[str, float] = {}
    for field in (
        "contact_map_overlap_cutoff",
        "all_alpha_fraction_threshold",
        "minimum_non_all_alpha_fraction",
    ):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AdapterError(f"selection.{STRUCTURAL_DIVERSITY_CONFIG_KEY}.{field} must be a finite number")
        result[field] = float(value)
        if not 0 <= result[field] <= 1:
            raise AdapterError(
                f"selection.{STRUCTURAL_DIVERSITY_CONFIG_KEY}.{field} must be between 0 and 1"
            )
    return result


def structural_diversity_disabled_reason(config: dict[str, Any]) -> str:
    """Return the recorded calibration blocker for a disabled structural-diversity rule."""
    selection = config.get("selection")
    raw = selection.get(STRUCTURAL_DIVERSITY_CONFIG_KEY) if isinstance(selection, dict) else None
    if isinstance(raw, dict) and raw.get("enabled") is False:
        reason = raw.get("reason")
        if isinstance(reason, str) and reason:
            return reason
    return "no calibrated contact-map-overlap cutoff and all-alpha definition are configured"


def _require_structural_diversity_lineage(
    lineage: dict[str, dict[str, Any]],
    candidate_id: str,
) -> None:
    source = lineage[candidate_id]
    if source.get(STRUCTURAL_DIVERSITY_CLUSTER_FIELD) != structural_surrogates.CLUSTERING_LABEL:
        raise AdapterError(
            f"candidate {candidate_id} has no computed contact-map-overlap cluster identifier; "
            "run the enabled structural diversity surrogate before promotion"
        )
    if source.get(STRUCTURAL_DIVERSITY_SECONDARY_FIELD) != structural_surrogates.SECONDARY_STRUCTURE_LABEL:
        raise AdapterError(
            f"candidate {candidate_id} has no computed Kabsch-Sander assignment; "
            "run the enabled structural diversity surrogate before promotion"
        )
    if not isinstance(source.get("all_alpha_surrogate"), bool):
        raise AdapterError(f"candidate {candidate_id} has no boolean all_alpha_surrogate value")


def ranking_mode(config: dict[str, Any]) -> str | None:
    """Return the shared scoring mode under the adapter's historical API."""
    try:
        return ranking_policy.declared_ranking_mode(
            config,
            candidate_claim=lane.is_candidate_claim(config),
        )
    except ranking_policy.RankingPolicyError as exc:
        # Keep the adapter's public exception contract while sharing the policy.
        raise AdapterError(str(exc).replace("ranking requires", "promotion requires")) from exc


def require_published_ranking_arm_mask(config: dict[str, Any]) -> None:
    """Validate the promotion scoring mode.

    This public helper retains its prior call shape for the runtime audit and
    existing callers.
    """
    ranking_mode(config)


def controls_are_required(config: dict[str, Any]) -> bool:
    """Return whether promotion must read calibrated control observations."""
    return not (
        lane.is_ungated_candidate_claim(config)
        and lane.control_panel_is_disabled(config)
    )


def apply_ranking_mode(
    ranked: list[dict[str, Any]],
    mode: str | None,
) -> list[dict[str, Any]]:
    """Return promotion rows scored under the shared declared ranking method."""
    try:
        return ranking_policy.apply_ranking_mode(ranked, mode)
    except ranking_policy.RankingPolicyError as exc:
        raise AdapterError(str(exc)) from exc


def read_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"{label} is invalid: {path}: {type(exc).__name__}: {exc}") from exc


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path, label)
    if not isinstance(value, dict):
        raise AdapterError(f"{label} must be a JSON object: {path}")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    return _read_jsonl(path, label, error_type=AdapterError)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AdapterError(f"refusing to write an empty promotion manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise AdapterError(f"{label} must be a finite number")
    return number


def require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AdapterError(f"{label} must be a non-empty string")
    return value


def minimum_delivery_fraction(config: dict[str, Any]) -> float:
    selection = config.get("selection")
    if not isinstance(selection, dict):
        raise AdapterError("config selection must be an object")
    policy = selection.get("shortfall_policy", {})
    if policy is None:
        policy = {}
    if not isinstance(policy, dict):
        raise AdapterError("config selection.shortfall_policy must be an object")
    value = policy.get("minimum_delivery_fraction", DEFAULT_MINIMUM_DELIVERY_FRACTION)
    fraction = safe_float(value, "selection.shortfall_policy.minimum_delivery_fraction")
    if fraction <= 0 or fraction > 1:
        raise AdapterError(
            "selection.shortfall_policy.minimum_delivery_fraction must be greater than 0 and at most 1"
        )
    return fraction


def selection_summary_message(summary: dict[str, Any]) -> str:
    requested = int(summary["requested_count"])
    delivered = int(summary["delivered_count"])
    rules = ",".join(str(rule) for rule in summary.get("binding_rules", [])) or "none"
    minimum_fraction = float(summary["minimum_delivery_fraction"])
    if delivered == requested:
        return f"promotion selection: requested={requested} delivered={delivered}"
    return (
        f"promotion selection shortfall: requested={requested} delivered={delivered} "
        f"binding_rule={rules} minimum_delivery_fraction={minimum_fraction:g}"
    )


def selection_shortfall_error(summary: dict[str, Any]) -> str:
    requested = int(summary["requested_count"])
    delivered = int(summary["delivered_count"])
    rules = ",".join(str(rule) for rule in summary.get("binding_rules", [])) or "none"
    minimum_fraction = float(summary["minimum_delivery_fraction"])
    return (
        f"selection shortfall refused: requested={requested} delivered={delivered} "
        f"binding_rule={rules} minimum_delivery_fraction={minimum_fraction:g}"
    )


def load_diversity_policy(config: dict[str, Any]) -> dict[str, Any]:
    selection = config.get("selection")
    if not isinstance(selection, dict):
        raise AdapterError("config selection must be an object")
    if "diversity" not in selection:
        return default_diversity_policy(config)
    policy = selection["diversity"]
    if not isinstance(policy, dict):
        raise AdapterError("config selection.diversity must be an object when supplied")
    missing = [field for field in REQUIRED_DIVERSITY_FIELDS if field not in policy]
    if missing:
        raise AdapterError(
            "config selection.diversity is missing required fields: " + ", ".join(missing)
        )
    distance = policy["min_levenshtein_distance"]
    if isinstance(distance, bool) or not isinstance(distance, int) or distance < 1:
        raise AdapterError("selection.diversity.min_levenshtein_distance must be a positive integer")
    for field in REQUIRED_DIVERSITY_FIELDS[1:5]:
        value = safe_float(policy[field], f"selection.diversity.{field}")
        if value <= 0 or value > 1:
            raise AdapterError(f"selection.diversity.{field} must be greater than 0 and at most 1")
    minimum_methods = policy["minimum_structure_methods"]
    if isinstance(minimum_methods, bool) or not isinstance(minimum_methods, int) or minimum_methods < 1:
        raise AdapterError("selection.diversity.minimum_structure_methods must be a positive integer")
    exception_errors = explicit_single_arm_exception_errors(config, policy)
    if exception_errors:
        raise AdapterError("; ".join(exception_errors))
    generation = config.get("generation")
    generators = generation.get("generators", []) if isinstance(generation, dict) else []
    sequence_design = config.get("sequence_design")
    designers = sequence_design.get("designers", []) if isinstance(sequence_design, dict) else []
    generator_count = len(
        [
            item
            for item in generators
            if isinstance(item, dict) and item.get("enabled", True) is True
        ]
    )
    designer_count = len(
        [
            item
            for item in designers
            if isinstance(item, dict) and item.get("enabled", True) is True
        ]
    )
    for field, expected in PUBLISHED_DIVERSITY_POLICY.items():
        observed = policy[field]
        if field in {"min_levenshtein_distance", "minimum_structure_methods"}:
            matches = observed == expected
        else:
            matches = math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12)
        # explicit_single_arm_exception_errors above is the source of truth for which
        # relaxations a one-arm profile may declare, and it accepts an enabled method
        # count of at most one. default_diversity_policy relaxes on the same range.
        # This audit re-derives the rule, so it has to use the same comparison. Demanding
        # exactly one refused a zero-method arm the value the policy module told it to set,
        # which is what a supplied-candidate profile has: sequences arrive already fixed,
        # so it enables no sequence designer at all.
        if not matches and (
            (field == "max_structure_method_fraction" and observed == 1.0 and generator_count <= 1)
            or (field == "max_seq_method_fraction" and observed == 1.0 and designer_count <= 1)
            or (field == "minimum_structure_methods" and observed == 1 and generator_count <= 1)
        ):
            continue
        if not matches:
            raise AdapterError(
                f"selection.diversity.{field}={observed!r} conflicts with the published rule {expected!r}"
            )
    return {**policy}


def read_sequence(path_value: Any, candidate_id: str) -> str:
    path = Path(require_text(path_value, f"candidate {candidate_id}.sequence_path"))
    if not path.is_absolute():
        raise AdapterError(f"candidate {candidate_id}.sequence_path must be absolute: {path}")
    if not path.is_file():
        raise AdapterError(f"candidate {candidate_id}.sequence_path is missing: {path}")
    headers = []
    sequence_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            headers.append(line[1:])
        elif line.strip():
            sequence_lines.append(line.strip())
    if headers != [candidate_id]:
        raise AdapterError(f"candidate {candidate_id}.sequence_path has the wrong FASTA header: {path}")
    sequence = "".join(sequence_lines).upper()
    if not sequence or any(residue not in "ACDEFGHIKLMNPQRSTVWY" for residue in sequence):
        raise AdapterError(f"candidate {candidate_id}.sequence_path has an invalid amino-acid sequence: {path}")
    return sequence


def raw_score_vectors(
    scores: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Preserve every candidate arm/seed score beside the pool-derived rank."""
    if not any("predictor" in row for row in scores):
        return {}
    vectors: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seen: set[tuple[str, str, int]] = set()
    # Keyed without the target, and {candidate: {predictor: [rows]}} has nowhere to put a
    # second one. The screen score table of a two-target campaign carries each candidate
    # against both targets, which either collides and is reported as a repeated score row,
    # blaming the table for a duplicate it does not contain, or lands on a candidate the
    # other target did not carry and merges two targets into one pool with no error. The
    # rest of the package keys a score row on target, candidate, predictor and seed; see
    # control_builder, ensemble_reducer and optimization_controller.validate_round_score_matrix.
    # Refuse until a multi-target promotion decides how it aggregates, which is the
    # scientist's call. Same defect as optimization_controller.raw_score_vectors.
    target_ids = {
        str(row.get("target_id"))
        for row in scores
        if row.get("control_type", "candidate") == "candidate"
        and row.get("target_id") is not None
    }
    if len(target_ids) > 1:
        raise AdapterError(
            "promotion selects for one target at a time; this screen score table "
            f"spans {len(target_ids)}: {sorted(target_ids)}"
        )
    for index, row in enumerate(scores):
        if row.get("control_type", "candidate") != "candidate":
            continue
        candidate_id = require_text(row.get("candidate_id"), f"screen score row {index}.candidate_id")
        predictor = require_text(row.get("predictor"), f"screen score row {index}.predictor")
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise AdapterError(f"screen score row {index}.seed must be an integer")
        key = (candidate_id, predictor, seed)
        if key in seen:
            raise AdapterError(f"screen score table repeats candidate, predictor, seed: {key}")
        seen.add(key)
        metric_row: dict[str, Any] = {
            "seed": seed,
            "status": row.get("status", "scored"),
        }
        for metric in ("ipsae_min", "sc_dockq"):
            value = row.get(metric)
            metric_row[metric] = None if value is None else safe_float(value, f"screen score row {index}.{metric}")
        vectors.setdefault(candidate_id, {}).setdefault(predictor, []).append(metric_row)
    for predictor_rows in vectors.values():
        for rows in predictor_rows.values():
            rows.sort(key=lambda row: int(row["seed"]))
    return vectors


def prepare_lineage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lineage: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(rows):
        candidate_id = require_text(source.get("candidate_id"), f"passing candidate row {index}.candidate_id")
        if candidate_id in lineage:
            raise AdapterError(f"passing candidate manifest repeats candidate_id: {candidate_id}")
        for field in (
            "origin_generator",
            "sequence_path",
            "sequence_sha256",
            "sequence_length",
            "design_pose_path",
            "design_pose_sha256",
            *LINEAGE_DIVERSITY_FIELDS,
        ):
            if field not in source:
                raise AdapterError(f"passing candidate {candidate_id} is missing {field}")
        sequence = read_sequence(source["sequence_path"], candidate_id)
        expected_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        if source["sequence_sha256"] != expected_hash:
            raise AdapterError(f"passing candidate {candidate_id}.sequence_sha256 does not match its FASTA")
        if source["sequence_length"] != len(sequence):
            raise AdapterError(f"passing candidate {candidate_id}.sequence_length does not match its FASTA")
        for field in LINEAGE_DIVERSITY_FIELDS:
            require_text(source[field], f"passing candidate {candidate_id}.{field}")
        require_text(source["origin_generator"], f"passing candidate {candidate_id}.origin_generator")
        require_text(source["design_pose_path"], f"passing candidate {candidate_id}.design_pose_path")
        require_text(source["design_pose_sha256"], f"passing candidate {candidate_id}.design_pose_sha256")
        lineage[candidate_id] = {**source, "_sequence": sequence}
    if not lineage:
        raise AdapterError("passing candidate manifest has no records")
    return lineage


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _cap(total: int, fraction: float) -> int:
    return max(1, math.ceil(total * fraction))


SELECTION_BINDING_RULES = (
    "selection.maximum_fraction_per_generator",
    "selection.diversity.max_root_backbone_fraction",
    "selection.diversity.max_tm90_cluster_fraction",
    "selection.diversity.max_structure_method_fraction",
    "selection.diversity.max_seq_method_fraction",
    "selection.diversity.min_levenshtein_distance",
    "selection.minimum_generators",
    "selection.diversity.minimum_structure_methods",
    "eligible candidate pool",
)


def select_diverse_parents(
    ranked: list[dict[str, Any]],
    lineage: dict[str, dict[str, Any]],
    config: dict[str, Any],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    selection = config["selection"]
    requested = selection.get("final_count")
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise AdapterError("selection.final_count must be a positive integer")
    minimum_generators = selection.get("minimum_generators")
    if isinstance(minimum_generators, bool) or not isinstance(minimum_generators, int) or minimum_generators < 1:
        raise AdapterError("selection.minimum_generators must be a positive integer")
    maximum_fraction = safe_float(
        selection.get("maximum_fraction_per_generator"),
        "selection.maximum_fraction_per_generator",
    )
    if maximum_fraction <= 0 or maximum_fraction > 1:
        raise AdapterError("selection.maximum_fraction_per_generator must be greater than 0 and at most 1")
    generator_capacity = _cap(requested, maximum_fraction)
    root_capacity = _cap(requested, safe_float(policy["max_root_backbone_fraction"], "max_root_backbone_fraction"))
    structural_diversity = structural_diversity_policy(config)
    cluster_capacity = (
        _cap(requested, safe_float(policy["max_tm90_cluster_fraction"], "max_tm90_cluster_fraction"))
        if structural_diversity is not None
        else None
    )
    maximum_all_alpha = (
        requested
        - math.ceil(requested * structural_diversity["minimum_non_all_alpha_fraction"])
        if structural_diversity is not None
        else None
    )
    method_capacity = _cap(requested, safe_float(policy["max_structure_method_fraction"], "max_structure_method_fraction"))
    sequence_method_capacity = _cap(requested, safe_float(policy["max_seq_method_fraction"], "max_seq_method_fraction"))
    minimum_methods = int(policy["minimum_structure_methods"])
    minimum_distance = int(policy["min_levenshtein_distance"])

    eligible: list[dict[str, Any]] = []
    for row in ranked:
        candidate_id = require_text(row.get("candidate_id"), "ranked candidate.candidate_id")
        if row.get("eligible") is not True:
            continue
        if candidate_id not in lineage:
            raise AdapterError(f"ranked candidate is absent from the passing manifest: {candidate_id}")
        if structural_diversity is not None:
            _require_structural_diversity_lineage(lineage, candidate_id)
        eligible.append(row)
    if not eligible:
        return []
    if structural_diversity is not None:
        eligible = sorted(
            eligible,
            key=lambda row: (
                -safe_float(
                    row.get("ipsae_min_ensemble"),
                    f"ranked candidate {row['candidate_id']}.ipsae_min_ensemble",
                ),
                str(row["candidate_id"]),
            ),
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    generator_counts: dict[str, int] = {}
    root_counts: dict[str, int] = {}
    cluster_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    sequence_method_counts: dict[str, int] = {}
    all_alpha_count = 0

    def source_for(row: dict[str, Any]) -> dict[str, Any]:
        return lineage[str(row["candidate_id"])]

    def can_select(row: dict[str, Any]) -> bool:
        candidate_id = str(row["candidate_id"])
        source = lineage[candidate_id]
        if candidate_id in selected_ids:
            return False
        generator = str(source["origin_generator"])
        root = str(source["root_backbone_id"])
        cluster = str(source["tm90_cluster_id"])
        method = str(source["structure_method"])
        sequence_method = str(source["seq_method"])
        if generator_counts.get(generator, 0) >= generator_capacity:
            return False
        if root_counts.get(root, 0) >= root_capacity:
            return False
        if cluster_capacity is not None and cluster_counts.get(cluster, 0) >= cluster_capacity:
            return False
        if method_counts.get(method, 0) >= method_capacity:
            return False
        if sequence_method_counts.get(sequence_method, 0) >= sequence_method_capacity:
            return False
        if (
            maximum_all_alpha is not None
            and source["all_alpha_surrogate"] is True
            and all_alpha_count >= maximum_all_alpha
        ):
            return False
        sequence = str(source["_sequence"])
        if any(
            levenshtein(sequence, str(source_for(item)["_sequence"])) < minimum_distance
            for item in selected
        ):
            return False
        return True

    def add(row: dict[str, Any]) -> None:
        nonlocal all_alpha_count
        candidate_id = str(row["candidate_id"])
        source = lineage[candidate_id]
        selected.append(row)
        selected_ids.add(candidate_id)
        for counts, field in (
            (generator_counts, "origin_generator"),
            (root_counts, "root_backbone_id"),
            (cluster_counts, "tm90_cluster_id"),
            (method_counts, "structure_method"),
            (sequence_method_counts, "seq_method"),
        ):
            key = str(source[field])
            counts[key] = counts.get(key, 0) + 1
        if structural_diversity is not None and source["all_alpha_surrogate"] is True:
            all_alpha_count += 1

    generators: list[str] = []
    for row in eligible:
        generator = str(source_for(row)["origin_generator"])
        if generator not in generators:
            generators.append(generator)
    for generator in generators:
        if len({str(source_for(row)["origin_generator"]) for row in selected}) >= minimum_generators:
            break
        for row in eligible:
            if str(source_for(row)["origin_generator"]) == generator and can_select(row):
                add(row)
                break

    for method in dict.fromkeys(str(source_for(row)["structure_method"]) for row in eligible):
        if len(set(method_counts)) >= minimum_methods:
            break
        for row in eligible:
            if str(source_for(row)["structure_method"]) == method and can_select(row):
                add(row)
                break

    for row in eligible:
        if len(selected) >= requested:
            break
        if can_select(row):
            add(row)

    return selected


def selection_summary(
    ranked: list[dict[str, Any]],
    lineage: dict[str, dict[str, Any]],
    selected: list[dict[str, Any]],
    config: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    selection = config["selection"]
    requested = selection.get("final_count")
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise AdapterError("selection.final_count must be a positive integer")
    minimum_generators = selection.get("minimum_generators")
    if (
        isinstance(minimum_generators, bool)
        or not isinstance(minimum_generators, int)
        or minimum_generators < 1
    ):
        raise AdapterError("selection.minimum_generators must be a positive integer")
    maximum_fraction = safe_float(
        selection.get("maximum_fraction_per_generator"),
        "selection.maximum_fraction_per_generator",
    )
    if maximum_fraction <= 0 or maximum_fraction > 1:
        raise AdapterError("selection.maximum_fraction_per_generator must be greater than 0 and at most 1")
    minimum_fraction = minimum_delivery_fraction(config)
    structural_diversity = structural_diversity_policy(config)
    capacities = {
        "selection.maximum_fraction_per_generator": _cap(requested, maximum_fraction),
        "selection.diversity.max_root_backbone_fraction": _cap(
            requested,
            safe_float(policy["max_root_backbone_fraction"], "max_root_backbone_fraction"),
        ),
        "selection.diversity.max_structure_method_fraction": _cap(
            requested,
            safe_float(policy["max_structure_method_fraction"], "max_structure_method_fraction"),
        ),
        "selection.diversity.max_seq_method_fraction": _cap(
            requested,
            safe_float(policy["max_seq_method_fraction"], "max_seq_method_fraction"),
        ),
    }
    if structural_diversity is not None:
        capacities["selection.diversity.max_tm90_cluster_fraction"] = _cap(
            requested,
            safe_float(policy["max_tm90_cluster_fraction"], "max_tm90_cluster_fraction"),
        )
        capacities["selection.structural_diversity_surrogate.max_all_alpha_count"] = (
            requested
            - math.ceil(requested * structural_diversity["minimum_non_all_alpha_fraction"])
        )
    minimum_distance = int(policy["min_levenshtein_distance"])
    selected_ids = {str(row.get("candidate_id")) for row in selected}
    generator_counts: dict[str, int] = {}
    root_counts: dict[str, int] = {}
    cluster_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    sequence_method_counts: dict[str, int] = {}
    all_alpha_count = 0
    for row in selected:
        candidate_id = require_text(row.get("candidate_id"), "selected candidate.candidate_id")
        source = lineage[candidate_id]
        if structural_diversity is not None:
            _require_structural_diversity_lineage(lineage, candidate_id)
        for counts, field in (
            (generator_counts, "origin_generator"),
            (root_counts, "root_backbone_id"),
            (cluster_counts, "tm90_cluster_id"),
            (method_counts, "structure_method"),
            (sequence_method_counts, "seq_method"),
        ):
            key = str(source[field])
            counts[key] = counts.get(key, 0) + 1
        if structural_diversity is not None and source["all_alpha_surrogate"] is True:
            all_alpha_count += 1

    eligible = [
        row
        for row in ranked
        if row.get("eligible") is True and str(row.get("candidate_id")) in lineage
    ]

    def reasons_for(row: dict[str, Any]) -> set[str]:
        candidate_id = str(row["candidate_id"])
        source = lineage[candidate_id]
        reasons: set[str] = set()
        if generator_counts.get(str(source["origin_generator"]), 0) >= capacities[
            "selection.maximum_fraction_per_generator"
        ]:
            reasons.add("selection.maximum_fraction_per_generator")
        if root_counts.get(str(source["root_backbone_id"]), 0) >= capacities[
            "selection.diversity.max_root_backbone_fraction"
        ]:
            reasons.add("selection.diversity.max_root_backbone_fraction")
        if structural_diversity is not None:
            _require_structural_diversity_lineage(lineage, candidate_id)
            if cluster_counts.get(str(source["tm90_cluster_id"]), 0) >= capacities[
                "selection.diversity.max_tm90_cluster_fraction"
            ]:
                reasons.add("selection.diversity.max_tm90_cluster_fraction")
            if (
                source["all_alpha_surrogate"] is True
                and all_alpha_count
                >= capacities["selection.structural_diversity_surrogate.max_all_alpha_count"]
            ):
                reasons.add("selection.structural_diversity_surrogate.max_all_alpha_count")
        if method_counts.get(str(source["structure_method"]), 0) >= capacities[
            "selection.diversity.max_structure_method_fraction"
        ]:
            reasons.add("selection.diversity.max_structure_method_fraction")
        if sequence_method_counts.get(str(source["seq_method"]), 0) >= capacities[
            "selection.diversity.max_seq_method_fraction"
        ]:
            reasons.add("selection.diversity.max_seq_method_fraction")
        sequence = str(source["_sequence"])
        if any(
            levenshtein(sequence, str(lineage[str(item["candidate_id"])]["_sequence"])) < minimum_distance
            for item in selected
        ):
            reasons.add("selection.diversity.min_levenshtein_distance")
        return reasons

    remaining = [row for row in eligible if str(row["candidate_id"]) not in selected_ids]
    binding_rule_candidates = list(SELECTION_BINDING_RULES[:-1])
    if structural_diversity is None:
        binding_rule_candidates.remove("selection.diversity.max_tm90_cluster_fraction")
    else:
        binding_rule_candidates.append("selection.structural_diversity_surrogate.max_all_alpha_count")
    binding_rules = [
        rule for rule in binding_rule_candidates if remaining and all(rule in reasons_for(row) for row in remaining)
    ]
    configured_generators = {
        str(source["origin_generator"])
        for source in lineage.values()
    }
    if len(configured_generators) < minimum_generators:
        binding_rules.append("selection.minimum_generators")
    minimum_methods = int(policy["minimum_structure_methods"])
    if len(method_counts) < minimum_methods:
        binding_rules.append("selection.diversity.minimum_structure_methods")
    if not binding_rules and len(selected) < requested:
        binding_rules.append("eligible candidate pool")
    binding_rules = list(dict.fromkeys(binding_rules))
    delivered = len(selected)
    if delivered >= requested:
        binding_rules = []
    policy_satisfied = len(generator_counts) >= minimum_generators and len(method_counts) >= minimum_methods
    if structural_diversity is not None:
        required_non_all_alpha = math.ceil(
            requested * structural_diversity["minimum_non_all_alpha_fraction"]
        )
        policy_satisfied = policy_satisfied and len(selected) - all_alpha_count >= required_non_all_alpha
    return {
        "schema_version": 1,
        "requested_count": requested,
        "delivered_count": delivered,
        "shortfall_count": requested - delivered,
        "binding_rules": binding_rules,
        "minimum_delivery_fraction": minimum_fraction,
        "delivery_fraction": delivered / requested,
        "policy_satisfied": policy_satisfied,
        "shortfall_allowed": (
            delivered > 0
            and delivered / requested >= minimum_fraction
            and policy_satisfied
        ),
        "selected_candidate_ids": [str(row["candidate_id"]) for row in selected],
        "structural_diversity_surrogate": {
            "status": "enabled" if structural_diversity is not None else "disabled",
            "reason": None if structural_diversity is not None else structural_diversity_disabled_reason(config),
            "label": structural_surrogates.CLUSTERING_LABEL if structural_diversity is not None else None,
        },
    }


def promotion_rows(
    selected: list[dict[str, Any]],
    lineage: dict[str, dict[str, Any]],
    score_vectors: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
    *,
    mode: str | None = None,
    controls_required: bool = True,
    ranking_claim_status: str = "available",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, score in enumerate(selected, start=1):
        candidate_id = str(score["candidate_id"])
        source = lineage[candidate_id]
        promotion_reason = "control-gated score and published diversity policy"
        if mode == PUBLISHED_RANKING_MODE:
            promotion_reason = "control-gated published three-mode rank_zscore and published diversity policy"
        elif mode == ranking_policy.CUSTOM_WEIGHTED_RANKING_MODE:
            promotion_reason = "configured weighted score and campaign diversity policy"
        elif mode in ranking_policy.REDUCED_RAW_MEAN_MODES:
            scope = {
                CANDIDATE_SINGLE_ARM_RANKING_MODE: "candidate single-arm",
                CANDIDATE_SINGLE_LINEAGE_TWO_MODE_RANKING_MODE: "candidate single-lineage two-mode",
                CANDIDATE_SINGLE_LINEAGE_THREE_MODE_RANKING_MODE: "candidate single-lineage three-mode",
                CANDIDATE_TWO_INDEPENDENT_ARM_RANKING_MODE: "candidate two-independent-arm",
            }[mode]
            promotion_reason = f"{scope} raw mean of ipsae_min and sc_dockq with published diversity policy"
            if not controls_required:
                promotion_reason += "; calibrated control gates are disabled"
        if ranking_claim_status == "suppressed":
            promotion_reason = "ranking control degraded; parent input retained without a best-design claim"
        elif ranking_claim_status == "unvalidated":
            promotion_reason = "ranking control is unavailable; parent input retained without a best-design claim"
        screen_rank_score_scope = score.get("rank_score_scope")
        if mode == ranking_policy.CUSTOM_WEIGHTED_RANKING_MODE:
            screen_rank_score_scope = ranking_policy.rank_score_scope(mode).replace(
                "rank_score", "screen_rank_score"
            )
        elif mode == CANDIDATE_SINGLE_ARM_RANKING_MODE:
            screen_rank_score_scope = (
                "screen_rank_score is the single-predictor mean of ipsae_min and sc_dockq. "
                "Do not compare screen_rank_score across runs or ranking modes."
            )
        elif mode == CANDIDATE_SINGLE_LINEAGE_TWO_MODE_RANKING_MODE:
            screen_rank_score_scope = (
                "screen_rank_score is the raw mean of ipsae_min and sc_dockq for two modes in one predictor lineage. "
                "The value applies only to its configured run and ranking mode."
            )
        elif mode == CANDIDATE_SINGLE_LINEAGE_THREE_MODE_RANKING_MODE:
            screen_rank_score_scope = (
                "screen_rank_score is the raw mean of ipsae_min and sc_dockq for three modes in one predictor lineage. "
                "The value applies only to its configured run and ranking mode."
            )
        elif mode == CANDIDATE_TWO_INDEPENDENT_ARM_RANKING_MODE:
            screen_rank_score_scope = (
                "screen_rank_score is the raw mean of ipsae_min and sc_dockq across two independent "
                "predictor instruments. The value applies only to its configured run and ranking mode."
            )
        row = {
            "candidate_id": candidate_id,
            "origin_generator": source["origin_generator"],
            "sequence_path": source["sequence_path"],
            "sequence_sha256": source["sequence_sha256"],
            "sequence_length": source["sequence_length"],
            "design_pose_path": source["design_pose_path"],
            "design_pose_sha256": source["design_pose_sha256"],
            "promotion_status": "promoted",
            "promotion_reason": promotion_reason,
            "promotion_rank": rank,
            "ranking_claim_status": ranking_claim_status,
            "screen_rank_score": float(score["rank_score"]),
            "screen_rank_score_scope": screen_rank_score_scope,
            "score_gating_mode": score.get("score_gating_mode"),
            "score_gating": score.get("score_gating", {}),
            "score_instrument": score.get("score_instrument"),
            "score_instrument_arms": score.get("score_instrument_arms", []),
            "predictor_agreement": score.get("predictor_agreement"),
            "per_seed_by_predictor": score.get("per_seed_by_predictor", {}),
            "score_gates": dict(score["gates"]),
            "root_backbone_id": source["root_backbone_id"],
            "tm90_cluster_id": source["tm90_cluster_id"],
            "structure_method": source["structure_method"],
            "seq_method": source["seq_method"],
            "fold_class": source["fold_class"],
        }
        for field in (
            STRUCTURAL_DIVERSITY_CLUSTER_FIELD,
            "tm90_cluster_cutoff_surrogate",
            STRUCTURAL_DIVERSITY_SECONDARY_FIELD,
            "helix_fraction_surrogate",
            "all_alpha_surrogate",
            "surrogate_disclosure",
        ):
            if field in source:
                row[field] = source[field]
        if score_vectors:
            candidate_vectors = score_vectors.get(candidate_id)
            if candidate_vectors is None:
                raise AdapterError(
                    f"promoted candidate {candidate_id} has a rank score but no raw score vector"
                )
            row["screen_raw_score_vectors"] = candidate_vectors
            row["screen_selected_scores"] = {
                "ipsae_min_ensemble": score.get("ipsae_min_ensemble"),
                "sc_dockq_ensemble": score.get("sc_dockq_ensemble"),
                "per_predictor": score.get("per_predictor", {}),
                "selected_seed_by_predictor": score.get("selected_seed_by_predictor", {}),
            }
        rows.append(row)
    return rows


def promotion_ranking_control(
    config: dict[str, Any],
    scores: list[dict[str, Any]],
    control_files: list[Path],
    mode: str | None,
) -> dict[str, Any]:
    """Evaluate the calibrated control pair with promotion's active rank score."""
    control_rows: list[dict[str, Any]] = []
    try:
        for path in control_files:
            for raw in read_jsonl(path, "control observations"):
                control_rows.append(
                    lane.observation_from_raw_measurement(
                        raw,
                        attempt_id="promotion-ranking-control",
                    )
                )
        panel = lane.rank_control_panel(
            config,
            scores,
            control_rows,
            candidate_seeds=lane.parent_seed_values(config),
            control_seeds=list(config["cofold"]["rescore_seeds"]),
            # assess_rank_score_direction requires controls scored through the
            # ranking path under test, and `rank_candidates` pins that path to
            # the published baseline reduction. Omitting the override let a
            # campaign configuring `median` validate its control direction on
            # median-reduced rows while its candidates ranked on best-of-N.
            seed_aggregation_override=lane.seed_aggregation(config),
        )
        ranked_controls = apply_ranking_mode(panel["controls"], mode)
        result = lane.control_separation.assess_rank_score_direction(ranked_controls)
    except (KeyError, TypeError, ValueError, AdapterError) as exc:
        result = {
            "status": "unvalidated",
            "statistic": lane.control_separation.RANK_STATISTIC,
            "direction": lane.control_separation.RANK_DIRECTION,
            "matched_controls": [],
            "mismatched_controls": [],
            "matched_minimum_rank_score": None,
            "mismatched_maximum_rank_score": None,
            "gap": None,
            "separated": None,
            "reason": f"control observations cannot be ranked in promotion: {exc}",
        }
    result["ranking_mode"] = mode
    return result


def _optimization_enabled(config: dict[str, Any]) -> bool:
    """Return whether this campaign runs optimization rounds."""
    optimization = config.get("optimization")
    return isinstance(optimization, dict) and optimization.get("enabled") is True


def run_stage(args: argparse.Namespace) -> int:
    config = read_json_object(args.config.resolve(), "campaign config")
    plan = load_plan(args.plan, config)
    stage = next(
        stage
        for stage in config.get("stages", [])
        if isinstance(stage, dict) and stage.get("stage_id") == args.stage
    )
    # The widened pool gathers promoted roots and every scored optimization child, so
    # it reads the promotion manifest as an input. A campaign with no optimization
    # rounds renames this very stage's output artifact to `rescore-candidates`, which
    # made `promote` take this branch and try to read the manifest it is itself about
    # to write. The guard is the loop being enabled, which is also the condition the
    # pool function uses to decide whether there is anything to gather.
    if _optimization_enabled(config) and stage.get("outputs", [{}])[0].get("artifact_id") == "rescore-candidates":
        rows = lane.optimization_scored_candidate_pool(config, args.artifact_root.resolve())
        output_dir = (args.attempt_dir / args.phase).resolve()
        output_path = output_dir / OUTPUT_NAME
        if output_path.exists():
            raise AdapterError(f"promotion output already exists: {output_path}")
        write_jsonl(output_path, rows)
        print(f"promotion selector: rescore_pool={len(rows)} manifest={output_path}")
        return 0
    mode = ranking_mode(config)
    policy = load_diversity_policy(config)
    artifact_root = args.artifact_root.resolve()
    _, score_files = input_files(
        plan,
        args.receipts_dir,
        args.stage,
        artifact_id="intermediate-score-table" if lane.intermediate_enabled(config) else "screen-score-table",
    )
    _, passing_files = input_files(
        plan,
        args.receipts_dir,
        args.stage,
        artifact_id="passing-candidates",
        source_stage_id="filter-novelty",
    )
    if len(score_files) != 1 or len(passing_files) != 1:
        raise AdapterError(
            "promote requires one declared parent score table file and one declared "
            "passing-candidates file through stage receipts"
        )
    score_path = score_files[0]
    passing_path = passing_files[0]
    scores = read_jsonl(score_path, "screen score table")
    passing = read_jsonl(passing_path, "passing candidate manifest")
    control_required = controls_are_required(config)
    control_files: list[Path] = []
    if control_required:
        try:
            _, control_files = input_files(
                plan,
                args.receipts_dir,
                args.stage,
                artifact_id="control-observations",
                source_stage_id="control-calibration",
            )
        except DeclaredArtifactError as exc:
            raise AdapterError(str(exc)) from exc
        if len(control_files) != 1:
            raise AdapterError(
                "promote requires one declared control-observations file through stage receipts"
            )
    pool_check = (
        lane.validate_intermediate_scored_pool(config, score_path, artifact_root)
        if lane.intermediate_enabled(config)
        else lane.validate_screen_scored_pool(config, score_path, artifact_root)
    )
    if not pool_check["ok"]:
        raise AdapterError("parent score validation failed: " + "; ".join(pool_check["errors"][:8]))
    if control_required:
        control_check = lane.validate_control_calibration(config, artifact_root)
        if not control_check["ok"]:
            raise AdapterError("control calibration failed: " + "; ".join(control_check["errors"][:8]))
    lineage = prepare_lineage(passing)
    ranked = lane.rank_candidate_cohort(config, scores, lane.parent_seed_values(config))
    ranked = apply_ranking_mode(ranked, mode)
    ranked.sort(key=lambda row: lane._rank_sort_key(row, config))
    ranking_control = promotion_ranking_control(config, scores, control_files, mode)
    ranking_claim_status = (
        "available"
        if ranking_control["status"] == "passed"
        else "suppressed" if ranking_control["status"] == "degraded" else "unvalidated"
    )
    selected = select_diverse_parents(ranked, lineage, config, policy)
    rows = promotion_rows(
        selected,
        lineage,
        raw_score_vectors(scores),
        mode=mode,
        controls_required=control_required,
        ranking_claim_status=ranking_claim_status,
    )
    selection_config = config.get("selection")
    if (
        not isinstance(selection_config, dict)
        or not isinstance(selection_config.get("final_count"), int)
        or not policy
    ):
        summary = {
            "schema_version": 1,
            "requested_count": len(rows),
            "delivered_count": len(rows),
            "shortfall_count": 0,
            "binding_rules": [],
            "minimum_delivery_fraction": DEFAULT_MINIMUM_DELIVERY_FRACTION,
            "delivery_fraction": 1.0,
            "policy_satisfied": True,
            "shortfall_allowed": True,
            "selected_candidate_ids": [str(row.get("candidate_id")) for row in rows],
        }
    else:
        summary = selection_summary(ranked, lineage, selected, config, policy)
        summary["delivered_count"] = len(rows)
        summary["shortfall_count"] = int(summary["requested_count"]) - len(rows)
        summary["delivery_fraction"] = len(rows) / int(summary["requested_count"])
        summary["selected_candidate_ids"] = [str(row.get("candidate_id")) for row in rows]
        summary["shortfall_allowed"] = (
            len(rows) > 0
            and summary["delivery_fraction"] >= float(summary["minimum_delivery_fraction"])
            and summary["policy_satisfied"]
        )
    summary["scoring_arm_status"] = ranking_control["status"]
    summary["ranking_control"] = ranking_control
    summary["ranking_claim_status"] = ranking_claim_status
    if not summary["shortfall_allowed"]:
        raise AdapterError(selection_shortfall_error(summary))
    output_dir = (args.attempt_dir / args.phase).resolve()
    output_path = output_dir / OUTPUT_NAME
    if output_path.exists():
        raise AdapterError(f"promotion output already exists: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_dir, prefix="promotion-", suffix=".jsonl", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    os.replace(temporary_path, output_path)
    summary_path = (artifact_root / "promotion" / PROMOTION_SUMMARY_NAME).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    attempt_summary_path = output_dir / PROMOTION_SUMMARY_NAME
    attempt_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if ranking_claim_status != "available":
        print(
            f"promotion selector: scoring_arm_status={ranking_control['status']} "
            f"parent_inputs={len(rows)} manifest={output_path}"
        )
    elif len(rows) < int(summary["requested_count"]):
        print(selection_summary_message(summary) + f" manifest={output_path}")
    else:
        print(f"promotion selector: parents={len(rows)} manifest={output_path}")
    return 0


def parse_stage(args: argparse.Namespace) -> int:
    output_path = (args.attempt_dir / args.phase / OUTPUT_NAME).resolve()
    summary_path = (args.attempt_dir / args.phase / PROMOTION_SUMMARY_NAME).resolve()
    errors: list[str] = []
    files: list[Path] = []
    config = read_json_object(args.config.resolve(), "campaign config")
    stage = next(
        stage
        for stage in config.get("stages", [])
        if isinstance(stage, dict) and stage.get("stage_id") == args.stage
    )
    is_rescore_pool = (
        _optimization_enabled(config)
        and stage.get("outputs", [{}])[0].get("artifact_id") == "rescore-candidates"
    )
    if not output_path.is_file():
        errors.append(f"declared output is missing: {output_path}")
    else:
        files.append(output_path)
        try:
            rows = read_jsonl(output_path, "promotion manifest")
            if is_rescore_pool:
                candidate_ids = [row.get("candidate_id") for row in rows]
                if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids):
                    errors.append(f"rescore candidate manifest contains a row without candidate_id: {output_path}")
                if len(candidate_ids) != len(set(candidate_ids)):
                    errors.append(f"rescore candidate manifest contains duplicate candidate IDs: {output_path}")
            elif any(row.get("promotion_status") != "promoted" for row in rows):
                errors.append(f"promotion manifest contains a row without promotion_status=promoted: {output_path}")
        except AdapterError as exc:
            errors.append(str(exc))
    if not summary_path.is_file():
        errors.append(f"declared output is missing: {summary_path}")
    else:
        files.append(summary_path)
    result_path = (args.attempt_dir / args.phase / "parser-result.json").resolve()
    # The executor sums the `records` field over every declared output file and
    # refuses a receipt whose parsed_count disagrees, so this counts every
    # declared output the way the executor does: a JSONL output by its rows, any
    # other declared document as one record. Counting only the manifest rows
    # made promote's own receipt fail re-validation from every later stage.
    parsed_count = 0
    if files and not errors:
        for path in files:
            if path.suffix == ".jsonl":
                parsed_count += len(read_jsonl(path, "promotion manifest"))
            else:
                parsed_count += 1
    result = {
        "ok": bool(files) and not errors,
        "parsed_count": parsed_count,
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": sorted(sha256_file(path) for path in files),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for error in errors:
        print(f"promotion selector parser: {error}", file=sys.stderr)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "toolcheck":
            print("promotion selector ok, standard library and claude_binder lane contracts")
            return 0
        return run_stage(args) if args.command == "run" else parse_stage(args)
    except Exception as exc:  # noqa: BLE001
        print(f"promotion selector: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
