"""Shared rules for filter-contract reference data."""

from __future__ import annotations

from typing import Any


NOT_APPLICABLE_PREFIX = "not_applicable: "

# The sequence filters below are wired into the existing two-stage campaign
# graph. The cheap screen has its own adapter until the stage-graph owner adds
# it between generation and folding.
SCREEN_THRESHOLD_SOURCES = frozenset(
    {"published", "distributional", "calibrated", "convention"}
)
SCREEN_DISPOSITIONS = frozenset(
    {"ACCEPTED_TO_FOLD", "RANKED_OUT", "REJECTED", "FOLDED"}
)
SCREEN_REASON_CODES = frozenset(
    {
        "PARSE_FAIL",
        "LENGTH_MISMATCH",
        "MAP_UNRESOLVED",
        "CLASH_EXCESS",
        "HOTSPOT_COVERAGE_LOW",
        "BSA_LOW",
        "COMPOSITION_FLAG",
        "MPNN_PERCENTILE_LOW",
        "SS_IRREGULAR",
        "AGGREGATION_FLAG",
        "ESM_PLL_LOW",
        "GATE_RELAXED",
        "RANKED_OUT",
    }
)
AVAILABLE_FILTER_IDS = frozenset(
    {
        "composition",
        "exact_duplicates",
        "liability_chemistry",
        "sequence_novelty",
        "model_likelihood",
    }
)
# Restoring structure_novelty requires TM-align and Foldseek, plus a stage that
# writes their per-candidate results. Restoring secondary_structure requires
# DSSP, plus a stage that writes its per-candidate assignments.
RETIRED_FILTER_TOOLS = {
    "structure_novelty": "TM-align and Foldseek",
    "secondary_structure": "DSSP",
}
RETIRED_FILTER_BENEFITS = {
    "structure_novelty": (
        "write per-candidate structural-comparison results for the structural-novelty gate"
    ),
    "secondary_structure": (
        "write per-candidate secondary-structure assignments for the secondary-structure gate"
    ),
}
# What is actually missing, which is a stage and not an executable.
#
# These strings used to read "TM-align and Foldseek are unavailable" and "DSSP is
# unavailable". Nothing in this package probes for any of the three. No module
# under `src/claude_binder` calls `shutil.which` on them, although it does call
# it for the renderer, LigandMPNN, Caliby and Boltz-2, and `data/catalog.json`
# carries no row for any of them. So the package asserted machine state it had
# never measured, about tools it does not record, and a scientist enabling
# either gate was refused with an environment reason nobody had checked.
#
# It is also not the whole blocker. `adapters/novelty_filter.py` imports
# `structural_surrogates` and already computes a contact-map-overlap score, a
# Kabsch-Sander secondary-structure assignment and a clustering from them, in
# stdlib, in the same file that raises this error. The gate is retired because
# no stage writes the per-candidate results these two contracts read, which is
# what the comment above has said the whole time.
RETIRED_FILTER_UNAVAILABLE_REASONS = {
    "structure_novelty": (
        "no stage writes per-candidate TM-align and Foldseek results. This "
        "package does not check whether either program is installed"
    ),
    "secondary_structure": (
        "no stage writes per-candidate DSSP assignments. This package does not "
        "check whether DSSP is installed"
    ),
}
# These revisions name data produced within the run or computed from its candidate.
# A SHA-256 digest identifies external reference bytes, so none of these revisions
# can carry one.
NON_EXTERNAL_REFERENCE_REVISIONS = frozenset({"none", "self", "predictor-output", "local"})


def not_applicable(reason: str) -> str:
    """Record why a resolved configuration does not need a value."""
    return f"{NOT_APPLICABLE_PREFIX}{reason}"


def is_not_applicable(value: Any) -> bool:
    """Return whether a configuration value records a deliberate absence."""
    return isinstance(value, str) and value.startswith(NOT_APPLICABLE_PREFIX)


def reference_digest_is_required(reference_revision: Any) -> bool:
    """Return whether the revision names external reference data."""
    return (
        not isinstance(reference_revision, str)
        or reference_revision not in NON_EXTERNAL_REFERENCE_REVISIONS
    )


def retired_filter_error(filter_id: str) -> str | None:
    """Return the validation error for a gate whose required tool is absent."""
    tool = RETIRED_FILTER_TOOLS.get(filter_id)
    if tool is None:
        return None
    # The subject is plural for structure_novelty, so the verb came out as
    # "TM-align and Foldseek is not installed". The reason string now carries
    # its own verb and this sentence does not inflect one.
    return (
        f"filter gate {filter_id} cannot run because {RETIRED_FILTER_UNAVAILABLE_REASONS[filter_id]}. "
        f"Add a stage that writes the per-candidate result {tool} produces before enabling "
        f"{filter_id}. This package ships stdlib surrogates for these measures in "
        f"claude_binder.structural_surrogates and uses them elsewhere, so the gap is the "
        f"stage rather than the measurement."
    )


def retired_filter_statuses() -> list[dict[str, str]]:
    """Return the report rows for filters absent from shipped environments."""
    statuses: list[dict[str, str]] = []
    for filter_id, tool in RETIRED_FILTER_TOOLS.items():
        benefit = RETIRED_FILTER_BENEFITS[filter_id]
        unavailable = RETIRED_FILTER_UNAVAILABLE_REASONS[filter_id]
        statuses.append(
            {
                "filter_id": filter_id,
                "status": "skipped",
                "reason": f"{unavailable}. Installing {tool} would {benefit}.",
            }
        )
    return statuses


def screen_reason(code: str, detail: str = "") -> str:
    """Return one fixed screen reason code with an optional checkable detail."""
    if code not in SCREEN_REASON_CODES:
        raise ValueError(f"screen reason code is not registered: {code}")
    return f"{code}: {detail}" if detail else code


def filter_report(
    stage_id: str,
    candidate_ids: list[str],
    contracts: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    passing_candidate_ids: list[str],
    *,
    skipped_gates: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Summarize every configured gate, including gates with zero evaluations."""
    evaluated_gates: list[dict[str, Any]] = []
    gate_statuses: list[dict[str, Any]] = []
    for contract in contracts:
        filter_id = str(contract["filter_id"])
        gate_observations = [
            observation
            for observation in observations
            if observation.get("filter_id") == filter_id
        ]
        removed_candidate_ids = sorted(
            str(observation["candidate_id"])
            for observation in gate_observations
            if observation.get("pass") is False
        )
        evaluated_gates.append(
            {
                "filter_id": filter_id,
                "evaluated_candidate_count": len(gate_observations),
                "removed_candidate_count": len(removed_candidate_ids),
                "removed_candidate_ids": removed_candidate_ids,
            }
        )
        if not gate_observations:
            status = "skipped"
            reason = "The stage received zero candidate records for this gate."
        elif removed_candidate_ids:
            status = "ran and failed"
            reason = f"{len(removed_candidate_ids)} candidate records failed this gate."
        else:
            status = "ran and passed"
            reason = f"All {len(gate_observations)} candidate records passed this gate."
        gate_statuses.append(
            {
                "filter_id": filter_id,
                "status": status,
                "reason": reason,
                "evaluated_candidate_count": len(gate_observations),
                "removed_candidate_count": len(removed_candidate_ids),
            }
        )
    for skipped_gate in skipped_gates or []:
        filter_id = skipped_gate.get("filter_id")
        reason = skipped_gate.get("reason")
        if not isinstance(filter_id, str) or not filter_id:
            raise ValueError("skipped gate filter_id must be a non-empty string")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"skipped gate {filter_id} must have a reason")
        gate_statuses.append(
            {
                "filter_id": filter_id,
                "status": "skipped",
                "reason": reason,
                "evaluated_candidate_count": 0,
                "removed_candidate_count": 0,
            }
        )
    return {
        "schema_version": 1,
        "stage_id": stage_id,
        "evaluated_candidate_count": len(candidate_ids),
        "passing_candidate_count": len(passing_candidate_ids),
        "evaluated_gates": evaluated_gates,
        "gate_statuses": gate_statuses,
    }
