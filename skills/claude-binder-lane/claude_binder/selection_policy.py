"""Published diversity settings shared by validation and parent selection."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


REQUIRED_DIVERSITY_FIELDS = (
    "min_levenshtein_distance",
    "max_root_backbone_fraction",
    "max_tm90_cluster_fraction",
    "max_structure_method_fraction",
    "max_seq_method_fraction",
    "minimum_structure_methods",
)

PUBLISHED_DIVERSITY_POLICY = {
    "min_levenshtein_distance": 6,
    "max_root_backbone_fraction": 0.05,
    "max_tm90_cluster_fraction": 0.10,
    "max_structure_method_fraction": 0.50,
    "max_seq_method_fraction": 2 / 3,
    "minimum_structure_methods": 3,
}


SINGLE_ARM_DISABLED_CAPS = {
    "maximum_fraction_per_generator",
    "max_structure_method_fraction",
    "max_seq_method_fraction",
    "minimum_structure_methods",
}


COUNTER_SCREEN_METRIC = "selectivity_delta"
PUBLISHED_IPSAE_RANK_WEIGHT = 4.0
PUBLISHED_POSE_RANK_WEIGHT = 1.0
PUBLISHED_SELECTIVITY_RANK_WEIGHT = 4.0
PUBLISHED_COUNTER_SCREEN_TERM_COUNT = 9
PUBLISHED_COUNTER_SCREEN_RANK_WEIGHTS = {
    "ipsae_min_z": PUBLISHED_IPSAE_RANK_WEIGHT,
    "sc_dockq_z": PUBLISHED_POSE_RANK_WEIGHT,
    f"{COUNTER_SCREEN_METRIC}_z": PUBLISHED_SELECTIVITY_RANK_WEIGHT,
}


def published_counter_screen_terms(arms: Sequence[str]) -> tuple[dict[str, Any], ...]:
    """Return the nine published raw and z-score terms for three scoring arms.

    The published baseline adds one selectivity delta to each arm's existing
    ipSAE_min and sc_DockQ terms. The function accepts another arm count for
    callers that need to validate a reduced, explicitly disclosed instrument
    mask, but it labels the output as published only for three arms.
    """
    normalized = tuple(arms)
    if not normalized or any(not isinstance(arm, str) or not arm for arm in normalized):
        raise ValueError("counter-screen arms must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("counter-screen arms must be unique")
    terms: list[dict[str, Any]] = []
    for arm in normalized:
        terms.extend(
            (
                {
                    "arm": arm,
                    "metric": "ipsae_min",
                    "raw_column": f"ipsae_min_{arm}",
                    "z_column": "ipsae_min_z",
                    "rank_weight": PUBLISHED_IPSAE_RANK_WEIGHT,
                },
                {
                    "arm": arm,
                    "metric": "sc_dockq",
                    "raw_column": f"sc_dockq_{arm}",
                    "z_column": "sc_dockq_z",
                    "rank_weight": PUBLISHED_POSE_RANK_WEIGHT,
                },
                {
                    "arm": arm,
                    "metric": COUNTER_SCREEN_METRIC,
                    "raw_column": f"{COUNTER_SCREEN_METRIC}_{arm}",
                    "z_column": f"{COUNTER_SCREEN_METRIC}_z",
                    "rank_weight": PUBLISHED_SELECTIVITY_RANK_WEIGHT,
                },
            )
        )
    return tuple(terms)


def counter_screen_rank_weight_errors(weights: Mapping[str, Any]) -> list[str]:
    """Return errors for the published 4:1:4 counter-screen rank weights."""
    errors: list[str] = []
    if not isinstance(weights, Mapping):
        return ["counter-screen rank weights must be an object"]
    expected_fields = set(PUBLISHED_COUNTER_SCREEN_RANK_WEIGHTS)
    supplied_fields = set(weights)
    missing = sorted(expected_fields - supplied_fields)
    extra = sorted(supplied_fields - expected_fields)
    if missing:
        errors.append("counter-screen rank weights are missing: " + ", ".join(missing))
    if extra:
        errors.append("counter-screen rank weights are undeclared: " + ", ".join(extra))
    for field, expected in PUBLISHED_COUNTER_SCREEN_RANK_WEIGHTS.items():
        value = weights.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"counter-screen rank weight {field} must be a finite number")
        elif not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
            errors.append(
                f"counter-screen rank weight {field} must equal {expected:g}"
            )
    return errors


def counter_screen_rank_gate(status: str) -> tuple[int, str]:
    """Return the published gate bucket for panel-scoped counter-screen output.

    A detected panel score remains reportable but sorts after clean rows. An
    incomplete counter-screen has no rank because its required measurement is
    absent.
    """
    if status == "separated-clean":
        return 0, "ranked"
    if status in {"separated-with-detected-off-target", "contested"}:
        return 1, "ranked-at-bottom"
    return 2, "unranked"


def _enabled_items(config: dict, section: str, field: str) -> list[dict]:
    value = config.get(section)
    if not isinstance(value, dict):
        return []
    items = value.get(field)
    if not isinstance(items, list):
        return []
    return [
        item
        for item in items
        if isinstance(item, dict) and item.get("enabled", True) is True
    ]


def default_diversity_policy(config: dict) -> dict:
    """Return diversity defaults sized to the enabled campaign methods."""
    generators = _enabled_items(config, "generation", "generators")
    designers = _enabled_items(config, "sequence_design", "designers")
    generator_count = len(generators)
    designer_count = len(designers)
    return {
        **PUBLISHED_DIVERSITY_POLICY,
        "max_structure_method_fraction": (
            1.0
            if generator_count <= 1
            else PUBLISHED_DIVERSITY_POLICY["max_structure_method_fraction"]
        ),
        "max_seq_method_fraction": (
            1.0
            if designer_count <= 1
            else PUBLISHED_DIVERSITY_POLICY["max_seq_method_fraction"]
        ),
        "minimum_structure_methods": (
            min(PUBLISHED_DIVERSITY_POLICY["minimum_structure_methods"], generator_count)
            if generator_count > 0
            else PUBLISHED_DIVERSITY_POLICY["minimum_structure_methods"]
        ),
    }


def explicit_single_arm_exception_errors(config: dict, policy: dict) -> list[str]:
    """Validate recorded relaxations that make a one-arm profile selectable."""
    selection = config.get("selection")
    if not isinstance(selection, dict):
        return ["selection must be an object"]
    disabled_caps = selection.get("disabled_caps")
    if disabled_caps is None:
        disabled_caps = {}
    elif not isinstance(disabled_caps, dict):
        return [
            "selection.disabled_caps must be an object when supplied"
        ]
    generators = _enabled_items(config, "generation", "generators")
    designers = _enabled_items(config, "sequence_design", "designers")
    generator_count = len(generators)
    designer_count = len(designers)
    errors: list[str] = []

    def has_reason(field: str) -> bool:
        reason = disabled_caps.get(field)
        if isinstance(reason, str) and reason.strip():
            return True
        errors.append(f"selection.disabled_caps.{field} must be a non-empty reason")
        return False

    maximum_fraction = selection.get("maximum_fraction_per_generator")
    if maximum_fraction == 1.0:
        if generator_count != 1:
            errors.append(
                "selection.maximum_fraction_per_generator=1.0 requires exactly one enabled generator"
            )
        has_reason("maximum_fraction_per_generator")
    elif "maximum_fraction_per_generator" in disabled_caps:
        errors.append(
            "selection.disabled_caps.maximum_fraction_per_generator requires "
            "selection.maximum_fraction_per_generator=1.0"
        )

    exceptions = {
        "max_structure_method_fraction": (1.0, generator_count, 1),
        "max_seq_method_fraction": (1.0, designer_count, 1),
        "minimum_structure_methods": (1, generator_count, 1),
    }
    for field, (relaxed, actual_count, required_count) in exceptions.items():
        observed = policy.get(field)
        published = PUBLISHED_DIVERSITY_POLICY[field]
        if observed == relaxed and observed != published:
            # default_diversity_policy relaxes a cap whenever the enabled method
            # count is at most one, so the exception that records the relaxation
            # accepts the same range. Demanding exactly one made a zero-method arm
            # set a default it could never declare.
            if actual_count > required_count:
                errors.append(
                    f"selection.diversity.{field}={observed!r} requires at most "
                    f"{required_count} enabled method"
                )
            has_reason(field)
        elif field in disabled_caps:
            errors.append(
                f"selection.disabled_caps.{field} requires "
                f"selection.diversity.{field}={relaxed!r}"
            )
    unknown = sorted(set(disabled_caps) - SINGLE_ARM_DISABLED_CAPS)
    if unknown:
        errors.append("selection.disabled_caps has unknown fields: " + ", ".join(unknown))
    return errors
