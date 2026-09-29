"""Shared ranking-mode policy for promotion, validation, and final ranking.

This module deliberately has no dependency on :mod:`claude_binder.lane` or an
adapter.  Callers supply whether the campaign is a candidate-level claim; the
mode decision and the score transformation then have one implementation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import arms

PUBLISHED_RANKING_MODE_COUNT = 3
# The published protocol runs three modes over two lineages, because ESMFold2-Full
# and ESMFold2-Fast share the ESMFold2 lineage. Two is therefore the lineage floor
# the published z-score was ever measured against, and one lineage is below it.
PUBLISHED_RANKING_MODE_LINEAGE_FLOOR = 2
PUBLISHED_RANKING_MODE = "published-three-mode-zscore"
CUSTOM_WEIGHTED_RANKING_MODE = "custom-weighted-zscore"
CANDIDATE_SINGLE_ARM_RANKING_MODE = "candidate-single-arm-raw-mean"
CANDIDATE_SINGLE_LINEAGE_TWO_MODE_RANKING_MODE = (
    "candidate-single-lineage-two-mode-raw-mean"
)
CANDIDATE_SINGLE_LINEAGE_THREE_MODE_RANKING_MODE = (
    "candidate-single-lineage-three-mode-raw-mean"
)
CANDIDATE_TWO_INDEPENDENT_ARM_RANKING_MODE = "candidate-two-independent-arm-raw-mean"
REDUCED_RAW_MEAN_MODES = frozenset(
    {
        CANDIDATE_SINGLE_ARM_RANKING_MODE,
        CANDIDATE_SINGLE_LINEAGE_TWO_MODE_RANKING_MODE,
        CANDIDATE_SINGLE_LINEAGE_THREE_MODE_RANKING_MODE,
        CANDIDATE_TWO_INDEPENDENT_ARM_RANKING_MODE,
    }
)
REGISTERED_RANKING_MODES = (
    frozenset({PUBLISHED_RANKING_MODE, CUSTOM_WEIGHTED_RANKING_MODE})
    | REDUCED_RAW_MEAN_MODES
)
LINEAGE_DECLARATION_REMEDY = (
    "Declare cofold.predictors[].lineage_id on each unregistered mode. Name one "
    "value for modes of the same model family and a distinct value for an "
    "independent instrument."
)


class RankingPolicyError(ValueError):
    """A declared ranking mode or row cannot be applied honestly."""


def _enabled_predictors(
    config: Mapping[str, Any],
) -> list[Mapping[str, Any]] | None:
    """Return the enabled co-folding predictor entries, or None when none are declared."""
    cofold = config.get("cofold")
    if not isinstance(cofold, Mapping) or "predictors" not in cofold:
        return None
    predictors = cofold["predictors"]
    if not isinstance(predictors, list):
        raise RankingPolicyError("config cofold.predictors must be a list")
    return [
        item
        for item in predictors
        if isinstance(item, Mapping) and item.get("enabled", True) is True
    ]


def _predictor_names(enabled: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return one name per enabled predictor mode, in configuration order."""
    names: list[str] = []
    for index, item in enumerate(enabled):
        predictor_id = item.get("id")
        names.append(
            predictor_id
            if isinstance(predictor_id, str) and predictor_id
            else f"cofold.predictors[{index}]"
        )
    return names


def _require_declared_lineages(undeclared: Sequence[str]) -> None:
    """Refuse a cross-mode ranking whose lineage independence is not established."""
    if not undeclared:
        return
    raise RankingPolicyError(
        "ranking across predictor modes requires a declared lineage for every enabled "
        f"mode. The package registers no lineage for {', '.join(undeclared)}. "
        f"{LINEAGE_DECLARATION_REMEDY}"
    )


def _custom_mode_reason(config: Mapping[str, Any], predictor_count: int) -> str | None:
    """Return why the configuration selects its weighted ranking score."""
    scoring = config.get("scoring")
    scoring = scoring if isinstance(scoring, Mapping) else {}
    explicit = scoring.get("ranking_mode", None)
    if explicit is not None and explicit != CUSTOM_WEIGHTED_RANKING_MODE:
        raise RankingPolicyError(
            "scoring.ranking_mode must be custom-weighted-zscore when supplied"
        )
    if explicit == CUSTOM_WEIGHTED_RANKING_MODE:
        return "scoring.ranking_mode explicitly selects the configured weighted score"

    primary = scoring.get("primary_metric", "ipsae_min")
    pose = scoring.get("pose_metric", "sc_dockq")
    if (primary, pose) != ("ipsae_min", "sc_dockq"):
        return (
            "the configured primary and pose metrics require the configured weighted score"
        )
    directions = scoring.get("metric_directions")
    if isinstance(directions, Mapping) and "minimize" in directions.values():
        return "scoring.metric_directions contains a minimized metric"
    if predictor_count > PUBLISHED_RANKING_MODE_COUNT:
        return (
            f"the configuration enables {predictor_count} predictor modes, outside the "
            "published three-mode score"
        )
    return None


def _declares_baseline_fidelity(config: Mapping[str, Any]) -> bool:
    profile = config.get("profile")
    return isinstance(profile, Mapping) and profile.get("baseline_fidelity") is True


def ranking_mode_selection(
    config: Mapping[str, Any], *, candidate_claim: bool
) -> dict[str, Any]:
    """Return the declared ranking mode and the configuration that selected it.

    The mode follows the enabled predictor modes and the lineages that supply
    them. A count cannot select it alone. Three modes of one lineage are
    correlated by construction, so z-scoring them across arms would report an
    independent agreement the run never measured, and the error direction is
    inflated consensus.

    The published protocol is three modes over two lineages, so the gate asks for
    a lineage floor rather than three distinct lineages. That accepts the
    published trio and still refuses a single-lineage triple.

    A nonbaseline campaign can select its configured weighted score explicitly.
    The same mode is selected when the campaign changes either ranking metric,
    minimizes a metric, or enables more than three predictor modes.
    """
    enabled = _enabled_predictors(config)
    if enabled is None:
        return {
            "mode": None,
            "basis": "no cofold.predictors block declares a predictor mode",
            "predictor_modes": [],
            "predictor_lineages": [],
        }
    names = _predictor_names(enabled)
    predictor_ids = [
        item.get("id") for item in enabled if isinstance(item.get("id"), str)
    ]
    duplicates = arms.duplicate_predictor_ids(predictor_ids)
    if duplicates:
        raise RankingPolicyError(
            "cofold.predictors enables the same predictor id more than once: "
            f"{', '.join(duplicates)}. One predictor mode counted twice inflates the "
            "agreement every ranking mode reports. Remove the repeated entry, or give "
            "each configured mode its own id."
        )
    resolved = [arms.configured_predictor_lineage_id(item) for item in enabled]
    undeclared = [
        name for name, lineage in zip(names, resolved) if lineage is None
    ]
    lineages = tuple(
        dict.fromkeys(lineage for lineage in resolved if lineage is not None)
    )
    selection = {
        "predictor_modes": list(names),
        "predictor_lineages": list(lineages),
    }
    modes_phrase = ", ".join(names)
    lineage_phrase = ", ".join(lineages)
    custom_reason = _custom_mode_reason(config, len(enabled))
    if (
        custom_reason is None
        and not candidate_claim
        and not _declares_baseline_fidelity(config)
        and enabled
        and (len(enabled) != PUBLISHED_RANKING_MODE_COUNT or len(lineages) < PUBLISHED_RANKING_MODE_LINEAGE_FLOOR)
    ):
        custom_reason = "the nonbaseline predictor ensemble uses the configured weighted score"
    if custom_reason is not None:
        if _declares_baseline_fidelity(config):
            raise RankingPolicyError(
                "custom-weighted-zscore conflicts with profile.baseline_fidelity=true"
            )
        if not enabled:
            raise RankingPolicyError(
                "custom-weighted-zscore requires at least one enabled predictor mode"
            )
        if len(enabled) > 1:
            _require_declared_lineages(undeclared)
        lineage_basis = (
            f"across {len(lineages)} declared predictor lineage"
            f"{'s' if len(lineages) != 1 else ''}"
            if len(enabled) > 1
            else "with no cross-mode lineage claim"
        )
        return {
            **selection,
            "mode": CUSTOM_WEIGHTED_RANKING_MODE,
            "basis": (
                f"{custom_reason}. The score uses {len(enabled)} enabled predictor "
                f"mode{'s' if len(enabled) != 1 else ''} ({modes_phrase}) "
                f"{lineage_basis}"
            ),
        }
    if len(enabled) == 1 and candidate_claim:
        return {
            **selection,
            "mode": CANDIDATE_SINGLE_ARM_RANKING_MODE,
            "basis": (
                f"one enabled predictor mode ({modes_phrase}) ranks by the raw mean, "
                "and agreement across predictors is not tested"
            ),
        }
    if len(enabled) == 2 and candidate_claim:
        _require_declared_lineages(undeclared)
        if len(lineages) == 1:
            return {
                **selection,
                "mode": CANDIDATE_SINGLE_LINEAGE_TWO_MODE_RANKING_MODE,
                "basis": (
                    f"two enabled predictor modes ({modes_phrase}) both come from one "
                    f"predictor lineage ({lineage_phrase}), so the rank is the raw mean "
                    "and no independent agreement is claimed"
                ),
            }
        return {
            **selection,
            "mode": CANDIDATE_TWO_INDEPENDENT_ARM_RANKING_MODE,
            "basis": (
                f"two enabled predictor modes ({modes_phrase}) come from two predictor "
                f"lineages ({lineage_phrase}), and a candidate-level claim ranks by the "
                "raw mean"
            ),
        }
    if len(enabled) == PUBLISHED_RANKING_MODE_COUNT:
        _require_declared_lineages(undeclared)
        if len(lineages) >= PUBLISHED_RANKING_MODE_LINEAGE_FLOOR:
            return {
                **selection,
                "mode": PUBLISHED_RANKING_MODE,
                "basis": (
                    f"three enabled predictor modes ({modes_phrase}) come from "
                    f"{len(lineages)} predictor lineages ({lineage_phrase}), which meets "
                    "the independence the published z-score assumes"
                ),
            }
        if candidate_claim:
            return {
                **selection,
                "mode": CANDIDATE_SINGLE_LINEAGE_THREE_MODE_RANKING_MODE,
                "basis": (
                    f"three enabled predictor modes ({modes_phrase}) all come from one "
                    f"predictor lineage ({lineage_phrase}), so the rank is the raw mean "
                    "and no independent agreement is claimed"
                ),
            }
        raise RankingPolicyError(
            "the published three-mode rank_zscore requires enabled predictor modes from "
            f"at least {PUBLISHED_RANKING_MODE_LINEAGE_FLOOR} predictor lineages. The "
            f"three enabled modes ({modes_phrase}) all come from lineage "
            f"{lineage_phrase}, so z-scoring across them would report agreement this "
            "configuration cannot measure. Enable a mode from a second lineage, or set "
            "profile.claim_level to candidate to rank by the single-lineage raw mean."
        )
    raise RankingPolicyError(
        "ranking requires three enabled predictor modes for the published rank_zscore, "
        "one enabled predictor mode for a candidate-level raw mean, or two enabled predictor "
        "modes for a candidate-level raw mean; "
        f"config provides {len(enabled)}"
    )


def declared_ranking_mode(
    config: Mapping[str, Any], *, candidate_claim: bool
) -> str | None:
    """Return the scoring mode declared by the enabled predictor arms.

    The published protocol uses a three-mode normalized score over at least two
    predictor lineages. Candidate-level profiles may use one predictor mode, two
    predictor modes, or three modes of one lineage. A custom weighted mode
    preserves the score computed from the configured metrics, directions, and
    weights. Every mode carries a distinct disclosure.

    :func:`ranking_mode_selection` carries the same decision with the reason it
    reached, for a caller that has to tell a scientist which mode they got.
    """
    mode = ranking_mode_selection(config, candidate_claim=candidate_claim)["mode"]
    return mode if isinstance(mode, str) else None


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RankingPolicyError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RankingPolicyError(f"{label} must be a finite number")
    return result


def _candidate_id(row: Mapping[str, Any]) -> str:
    value = row.get("candidate_id")
    if not isinstance(value, str) or not value:
        raise RankingPolicyError(
            "ranked candidate.candidate_id must be a non-empty string"
        )
    return value


def _sample_standard_deviation(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    center = sum(values) / len(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def _raw_mean_scores_by_seed(row: Mapping[str, Any]) -> list[dict[str, float | int]]:
    """Evaluate the declared raw score on paired observations from each seed.

    ipSAE_min and scDockQ come from the same fold and are often correlated.
    Predictor arms also use the same seed panel. Combining marginal standard
    deviations as though these values were independent understates or
    overstates the observed variation, so preserve the pairing instead.
    """
    per_seed = row.get("per_seed_by_predictor")
    if not isinstance(per_seed, Mapping) or not per_seed:
        return []
    by_predictor: list[dict[int, float]] = []
    for seed_rows in per_seed.values():
        if not isinstance(seed_rows, list):
            return []
        scores: dict[int, float] = {}
        for seed_row in seed_rows:
            if not isinstance(seed_row, Mapping) or seed_row.get("status") != "scored":
                continue
            seed = seed_row.get("seed")
            ipsae_min = seed_row.get("ipsae_min")
            sc_dockq = seed_row.get("sc_dockq")
            if (
                not isinstance(seed, int)
                or isinstance(seed, bool)
                or not isinstance(ipsae_min, (int, float))
                or isinstance(ipsae_min, bool)
                or not isinstance(sc_dockq, (int, float))
                or isinstance(sc_dockq, bool)
                or not math.isfinite(float(ipsae_min))
                or not math.isfinite(float(sc_dockq))
            ):
                continue
            scores[seed] = (float(ipsae_min) + float(sc_dockq)) / 2.0
        if not scores:
            return []
        by_predictor.append(scores)
    common_seeds = set(by_predictor[0])
    for scores in by_predictor[1:]:
        common_seeds &= set(scores)
    return [
        {
            "seed": seed,
            "score": sum(scores[seed] for scores in by_predictor) / len(by_predictor),
        }
        for seed in sorted(common_seeds)
    ]


def apply_ranking_mode(
    ranked: Sequence[Mapping[str, Any]],
    mode: str | None,
    *,
    sort_key: Callable[[dict[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Return copied rows scored and ordered under ``mode``.

    Eligibility is a separate gate decision. Existing eligibility is therefore
    preserved; only callers supplying hand-built rows without it receive the
    coverage-and-filter fallback used by the historical adapter helper.
    """
    rows = [dict(row) for row in ranked]
    if mode in REDUCED_RAW_MEAN_MODES:
        for row in rows:
            candidate_id = _candidate_id(row)
            ipsae_min = _finite_number(
                row.get("ipsae_min_ensemble"),
                f"ranked candidate {candidate_id}.ipsae_min_ensemble",
            )
            sc_dockq = _finite_number(
                row.get("sc_dockq_ensemble"),
                f"ranked candidate {candidate_id}.sc_dockq_ensemble",
            )
            previous_score = row.get("rank_score")
            if (
                isinstance(previous_score, (int, float))
                and not isinstance(previous_score, bool)
                and math.isfinite(float(previous_score))
            ):
                row.setdefault("rank_zscore", float(previous_score))
            row["rank_score"] = (ipsae_min + sc_dockq) / 2.0
            row["rank_score_central_estimate"] = row["rank_score"]
            per_seed_scores = _raw_mean_scores_by_seed(row)
            row["rank_score_per_seed"] = per_seed_scores
            row["rank_score_spread"] = _sample_standard_deviation(
                [float(item["score"]) for item in per_seed_scores]
            )
            row["rank_score_spread_basis"] = (
                "sample standard deviation of the paired per-seed declared rank score"
                if row["rank_score_spread"] is not None
                else "unavailable: fewer than two paired per-seed rank scores"
            )
            if "eligible" not in row:
                row["eligible"] = (
                    row.get("coverage_complete") is True
                    and row.get("filter_pass") is True
                )
    elif mode == CUSTOM_WEIGHTED_RANKING_MODE:
        # The scoring layer already computed this score from the configured
        # metrics, directions, and weights. Preserve that value and its
        # eligibility decision. Recomputing here would replace the user's
        # objective after the campaign had already accepted it.
        pass
    elif mode not in {None, PUBLISHED_RANKING_MODE}:
        raise RankingPolicyError(f"ranking mode is not registered: {mode}")
    for row in rows:
        row["ranking_mode"] = mode
    if sort_key is None:

        def declared_score_sort_key(row: dict[str, Any]) -> Any:
            return (
                row.get("eligible") is not True,
                -_finite_number(
                    row.get("rank_score"),
                    f"ranked candidate {_candidate_id(row)}.rank_score",
                ),
                _candidate_id(row),
            )

        sort_key = declared_score_sort_key
    return sorted(rows, key=sort_key)


def ranking_formula(mode: str | None, published_formula: str) -> str:
    """Describe the exact statistic applied by :func:`apply_ranking_mode`."""
    if mode in REDUCED_RAW_MEAN_MODES:
        return "(ipsae_min_ensemble + sc_dockq_ensemble) / 2"
    if mode == CUSTOM_WEIGHTED_RANKING_MODE:
        return published_formula
    return published_formula


def ranking_normalization(mode: str | None) -> str:
    """Return the normalization disclosure for a declared ranking mode."""
    if mode in REDUCED_RAW_MEAN_MODES:
        return "none; raw ensemble metrics"
    if mode == CUSTOM_WEIGHTED_RANKING_MODE:
        return "configured signed weighted z-score within this run"
    return "population z-score within each target, predictor, and metric"


def registered_rank_score_scopes() -> frozenset[str]:
    """Return the scope disclosure of every registered ranking mode.

    A validator that enumerated the modes itself would keep validating the set
    that existed the day it was written. A new mode reaches it from here.
    """
    return frozenset(rank_score_scope(mode) for mode in REGISTERED_RANKING_MODES)


def rank_score_scope(mode: str | None) -> str:
    """Return a copy-safe scope warning for the declared score."""
    if mode == CUSTOM_WEIGHTED_RANKING_MODE:
        return (
            "rank_score uses the configured metrics, directions, and positive weights "
            "after within-run normalization. Do not compare rank_score across runs or "
            "ranking modes."
        )
    if mode == CANDIDATE_SINGLE_ARM_RANKING_MODE:
        return (
            "rank_score is the single-predictor raw mean of ipsae_min and sc_dockq. "
            "Do not compare rank_score across runs or ranking modes."
        )
    if mode == CANDIDATE_SINGLE_LINEAGE_TWO_MODE_RANKING_MODE:
        return (
            "rank_score is the raw mean of ipsae_min and sc_dockq across two modes in one "
            "predictor lineage. Do not compare rank_score across runs or ranking modes."
        )
    if mode == CANDIDATE_SINGLE_LINEAGE_THREE_MODE_RANKING_MODE:
        return (
            "rank_score is the raw mean of ipsae_min and sc_dockq across three modes in one "
            "predictor lineage. Do not compare rank_score across runs or ranking modes."
        )
    if mode == CANDIDATE_TWO_INDEPENDENT_ARM_RANKING_MODE:
        return (
            "rank_score is the raw mean of ipsae_min and sc_dockq across two independent "
            "predictor instruments. Do not compare rank_score across runs or ranking modes."
        )
    return (
        "rank_score is a z-score within this run's normalization pool only. "
        "Do not compare rank_score across runs."
    )
