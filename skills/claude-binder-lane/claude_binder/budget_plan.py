#!/usr/bin/env python3
"""Size the largest campaign a stated ceiling can pay for.

Every other cost surface in this package runs forward. A scientist writes a
plan, and `qualify` prices it. A scientist who starts from a budget, which is
how a first campaign usually starts, had to solve that by hand against one
measured rate record.

This module inverts it. Given a composed configuration and a ceiling, it
searches design counts through `qualify.cost_quote`, the same function that
gates dispatch, and returns the largest count whose priced total stays inside
the ceiling. It reuses that function rather than modelling cost a second way,
so a plan it sizes and a plan `qualify` prices cannot disagree.

Two refusals carry over from the quote. An unpriced arm makes the total
unknown, so a ceiling cannot be evaluated and this module refuses to size a
plan rather than sizing one against a floor. A ceiling that cannot pay for a
single design is a refusal too, with the one-design price reported so the
scientist can see by how much.

The measured-timing estimator in `cost.py` is reported beside the answer as a
separately sourced cross-check, never as the authority. It prices recorded GPU
seconds at the reference GPU-H100 rate that `cost.RATE_SOURCE` names, and that
record's own source field states the rate is not a Modal rate. The cross-check
is a modeled estimate on a reference machine, so it speaks for no provider.
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import cost, qualify, route_matrix

SCHEMA_VERSION = 1
DEFAULT_MAX_DESIGNS = 5000


class BudgetPlanError(ValueError):
    """The supplied ceiling or configuration cannot size a plan."""


def _quote(
    config: dict[str, Any],
    adapters: list[dict[str, Any]],
    designs: int,
    overrides: dict[str, float] | None,
) -> dict[str, Any]:
    return qualify.cost_quote(config, adapters, designs, overrides)


def _largest_within(
    config: dict[str, Any],
    adapters: list[dict[str, Any]],
    cap_usd: float,
    overrides: dict[str, float] | None,
    max_designs: int,
) -> tuple[int | None, dict[str, Any]]:
    """Return the largest design count whose priced total stays inside the cap.

    The search assumes the quote does not fall as the design count rises, which
    is true of every pricing shape the quote supports: a per-design rate scales
    and a configured total stays flat. The neighbour check below fails loudly if
    a configuration ever breaks that assumption.
    """
    first = _quote(config, adapters, 1, overrides)
    if not first["complete"]:
        return None, first
    if float(first["total_usd"]) > cap_usd:
        return None, first

    low, high = 1, max_designs
    top = _quote(config, adapters, high, overrides)
    if top["complete"] and float(top["total_usd"]) <= cap_usd:
        return high, top
    while low + 1 < high:
        middle = (low + high) // 2
        probe = _quote(config, adapters, middle, overrides)
        if probe["complete"] and float(probe["total_usd"]) <= cap_usd:
            low = middle
        else:
            high = middle
    chosen = _quote(config, adapters, low, overrides)
    beyond = _quote(config, adapters, low + 1, overrides)
    if beyond["complete"] and float(beyond["total_usd"]) <= cap_usd:
        raise BudgetPlanError(
            "the cost quote does not rise with the design count, so this search "
            "cannot bound the plan; price the configuration by hand"
        )
    return low, chosen


def _route_classes(config: dict[str, Any], adapters: list[dict[str, Any]]) -> dict[str, str]:
    """Classify each priced arm's route from the composed configuration."""
    classes: dict[str, str] = {}
    for adapter in adapters:
        module = route_matrix._module_name(adapter)
        classes[str(adapter["adapter_id"])] = route_matrix._route_class(config, adapter, module)
    return classes


def _timing_cross_check(designs: int, config: dict[str, Any]) -> dict[str, Any]:
    """Return the measured-timing estimate for the nearest supported scale.

    ``CampaignSettings`` refuses a design count that is not a multiple of ten,
    because the generation timing behind it was measured in batches of ten. The
    cross-check reports the scale it actually priced rather than silently
    answering about a different one.
    """
    scale = (designs // 10) * 10
    if scale < 10:
        return {
            "available": False,
            "reason": "the measured-timing estimator starts at ten designs",
        }
    cofold = config.get("cofold")
    predictors = cofold.get("predictors") if isinstance(cofold, dict) else None
    arms = (
        sum(1 for item in predictors if isinstance(item, dict) and item.get("enabled") is True)
        if isinstance(predictors, list)
        else 1
    )
    settings = cost.CampaignSettings(designs=scale, predictor_arms=max(arms, 1))
    cold = cost.estimate_cold(settings)
    warm = cost.estimate_projected_warm(settings)
    return {
        "available": True,
        "designs_priced": scale,
        "predictor_arms": max(arms, 1),
        "cold_usd": float(cold.total_cost_usd.value),
        "cold_evidence": cold.evidence.status.value,
        "projected_warm_usd": float(warm.total_cost_usd.value),
        "projected_warm_evidence": warm.evidence.status.value,
        # Name the machine the rate describes. This field read "Modal L40S" while
        # the arithmetic below it applied `cost.MEASURED_H100_RATE_USD_PER_SECOND`,
        # whose provenance record states in its own source field that these are
        # not Modal rates and that the machine names are another provider's. A
        # reader took that label for a Modal price that had never been applied.
        "machine": "GPU-H100 at the reference rate, a modeled estimate that is not a Modal rate",
        "meaning": (
            "A separately sourced figure for comparison. It is a modeled estimate: it "
            "prices GPU seconds from recorded timings at a reference rate measured on "
            "one provider usage export, and that provider is not Modal. The plan above "
            "prices the per-design rates the configuration records."
        ),
    }


def size_from_timings(
    cap_usd: float,
    *,
    seeds: int = 1,
    predictor_arms: int = 1,
    rounds: int = 1,
    warm: bool = False,
    max_designs: int = DEFAULT_MAX_DESIGNS,
) -> dict[str, Any]:
    """Size a plan from the measured timing model instead of configured rates.

    A first campaign has no per-adapter rate recorded, because nobody has billed
    that account yet. Refusing to size anything in that case sends the scientist
    to solve GPU seconds by hand, which is the work this package already did. The
    figure is a modeled estimate on a reference rate, labelled as one, and it
    never substitutes for the configured-rate quote when that quote exists.
    """
    if isinstance(max_designs, bool) or not isinstance(max_designs, int) or max_designs < 1:
        raise BudgetPlanError("max_designs must be a positive integer")
    estimate = cost.estimate_projected_warm if warm else cost.estimate_cold
    chosen: dict[str, Any] | None = None
    # CampaignSettings is evidenced only at batches of ten. Respect the caller's
    # ceiling by searching the largest supported batch at or below it rather than
    # silently continuing to this module's default of 5,000.
    supported_search_ceiling = (max_designs // 10) * 10
    designs = 10
    while designs <= supported_search_ceiling:
        settings = cost.CampaignSettings(
            designs=designs, seeds=seeds, predictor_arms=predictor_arms, rounds=rounds
        )
        result = estimate(settings)
        total = float(result.total_cost_usd.value)
        if total > cap_usd:
            break
        chosen = {
            "designs": designs,
            "folds": int(settings.unfiltered_fold_count),
            "total_usd": total,
            "wall_clock_hours": float(result.wall_clock_hours.value),
            "evidence": result.evidence.status.value,
        }
        designs += 10
    search_limit_reached = (
        chosen is not None
        and int(chosen["designs"]) == supported_search_ceiling
        and designs > supported_search_ceiling
    )
    return {
        "mode": "projected-warm" if warm else "cold",
        "seeds": seeds,
        "predictor_arms": predictor_arms,
        "rounds": rounds,
        "designs_search_ceiling": max_designs,
        "largest_supported_batch_within_search_ceiling": supported_search_ceiling,
        "search_limit_reached": search_limit_reached,
        "rate_source": cost.RATE_SOURCE,
        "largest_plan": chosen,
        "floor_note": (
            (
                "max_designs is below the timing model's smallest supported batch of ten designs"
                if supported_search_ceiling < 10
                else "the timing model starts at ten designs, so a ceiling below that price sizes nothing"
            )
            if chosen is None
            else None
        ),
        "search_limit_note": (
            f"the timing-model search reached max_designs={max_designs} "
            f"(largest supported batch {supported_search_ceiling}). This is a search "
            "limit, not evidence that the budget cannot support more designs; raise "
            "max_designs to continue sizing"
            if search_limit_reached
            else None
        ),
        "meaning": (
            "A modeled estimate from recorded timings on a reference GPU rate. It is not a "
            "rate measured on the scientist's own account, and the second predictor arm and "
            "the fold-cost-versus-size rule carry no measurement at all. It applies the "
            "configured rescore-seed count and enabled optimization-round count, but it does "
            "not price the campaign's separate screen and rescore cohorts, controls, smoke "
            "calls, storage, or egress."
        ),
    }


def _timing_model_dimensions(config: dict[str, Any]) -> tuple[int, int, int]:
    """Return the configured dimensions the coarse timing model can represent."""
    cofold = config.get("cofold")
    predictors = cofold.get("predictors") if isinstance(cofold, dict) else None
    arms = (
        sum(
            1
            for item in predictors
            if isinstance(item, dict) and item.get("enabled", True) is True
        )
        if isinstance(predictors, list)
        else 1
    )
    seeds = 1
    if isinstance(cofold, dict) and "rescore_seeds" in cofold:
        configured_seeds = cofold.get("rescore_seeds")
        if (
            not isinstance(configured_seeds, list)
            or not configured_seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in configured_seeds)
        ):
            raise BudgetPlanError("cofold.rescore_seeds must be a non-empty integer list")
        seeds = len(set(configured_seeds))

    rounds = 1
    optimization = config.get("optimization")
    if isinstance(optimization, dict) and optimization.get("enabled") is True:
        configured_rounds = optimization.get("rounds")
        if (
            isinstance(configured_rounds, bool)
            or not isinstance(configured_rounds, int)
            or configured_rounds < 1
        ):
            raise BudgetPlanError(
                "optimization.rounds must be a positive integer when optimization is enabled"
            )
        rounds = configured_rounds
    return seeds, max(arms, 1), rounds


def plan(
    config: dict[str, Any],
    *,
    cap_usd: float,
    overrides: dict[str, float] | None = None,
    max_designs: int = DEFAULT_MAX_DESIGNS,
    cloud_only: bool = False,
) -> dict[str, Any]:
    """Return the largest plan the ceiling pays for, or a stated refusal."""
    if not (cap_usd > 0):
        raise BudgetPlanError("the ceiling must be a positive number of US dollars")
    if isinstance(max_designs, bool) or not isinstance(max_designs, int) or max_designs < 1:
        raise BudgetPlanError("max_designs must be a positive integer")
    adapters = qualify._model_adapters(config)
    classes = _route_classes(config, adapters)
    local = sorted(
        adapter_id
        for adapter_id, route_class in classes.items()
        if route_class in {route_matrix.LOCAL_PROCESS, route_matrix.LOCAL_FIXTURE}
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "check_type": "claude_binder_budget_plan",
        "cap_usd": cap_usd,
        "arm_count": len(adapters),
        "route_classes": classes,
        "cloud_only_requested": cloud_only,
        "local_arms": local,
        "dispatched": False,
        "meaning": (
            "A sized plan is a planning figure from recorded rates. It authorizes no "
            "dispatch, establishes no qualification, and is not a settled bill."
        ),
    }
    if cloud_only and local:
        result.update(
            {
                "ok": False,
                "status": "cloud-only-refused",
                "designs": None,
                "refusal": (
                    "these arms run locally in this configuration and a cloud-only plan "
                    "was requested: " + ", ".join(local)
                ),
            }
        )
        return result

    designs, quote = _largest_within(config, adapters, cap_usd, overrides, max_designs)
    result["cost"] = quote
    if designs is None and not quote["complete"]:
        seeds, arms, rounds = _timing_model_dimensions(config)
        modeled = size_from_timings(
            cap_usd,
            seeds=seeds,
            predictor_arms=arms,
            rounds=rounds,
            max_designs=max_designs,
        )
        result.update(
            {
                "ok": modeled["largest_plan"] is not None,
                "status": "sized-from-timings",
                "designs": (modeled["largest_plan"] or {}).get("designs"),
                "designs_search_ceiling": max_designs,
                "configured_rate_gap": (
                    "no per-design rate is recorded for "
                    + ", ".join(row["adapter_id"] for row in quote["unpriced"])
                    + f", so the configured-rate quote is a floor of "
                    f"${float(quote['priced_subtotal_usd']):.6f} at one design rather than a total"
                ),
                "modeled_plan": modeled,
                "next_measurement": (
                    "Record one settled bill for this account and rerun; the configured-rate "
                    "path then prices the plan instead of the reference rate."
                ),
            }
        )
        if modeled.get("search_limit_note"):
            result["note"] = modeled["search_limit_note"]
        return result
    if designs is None:
        result.update(
            {
                "ok": False,
                "status": "ceiling-below-one-design",
                "designs": None,
                "refusal": (
                    f"one design costs ${float(quote['total_usd']):.6f}, above the stated "
                    f"ceiling of ${cap_usd:.6f}"
                ),
            }
        )
        return result

    total = float(quote["total_usd"])
    priced = sorted(
        (item for item in quote["items"] if item["priced"]),
        key=lambda item: item["estimate_usd"],
        reverse=True,
    )
    result.update(
        {
            "ok": True,
            "status": "sized",
            "designs": designs,
            "designs_search_ceiling": max_designs,
            "total_usd": total,
            "headroom_usd": cap_usd - total,
            "binding_arm": priced[0]["adapter_id"] if priced else None,
            "per_arm_usd": {item["adapter_id"]: item["estimate_usd"] for item in priced},
            "rate_evidence": quote["rate_evidence"],
            "timing_cross_check": _timing_cross_check(designs, config),
            "excluded_costs": (
                "Cold starts, keep-alive billing, endpoint checks, storage, and egress "
                "are outside the per-design rates this plan reads. Budget for them on top."
            ),
        }
    )
    # Two independently sourced figures for the same plan can disagree by an order
    # of magnitude, and the first time that happened the configured rate was the
    # optimistic one. Disclose it rather than letting the cheaper number stand
    # alone in front of a scientist about to authorize a ceiling.
    check = result["timing_cross_check"]
    if check.get("available"):
        modeled = check["cold_usd"]
        if total > 0 and (modeled / total >= 2 or total / max(modeled, 1e-9) >= 2):
            result["cross_check_disagreement"] = (
                f"the configured rates price {designs} designs at ${total:.2f} while the "
                f"measured timing model prices {check['designs_priced']} designs at "
                f"${modeled:.2f} cold and ${check['projected_warm_usd']:.2f} projected warm. "
                "Find out which stages the configured rate covers before trusting the lower "
                "figure."
            )

    if designs == max_designs:
        result["note"] = (
            "the plan reached the search ceiling, so the budget may pay for more designs; "
            "raise --max-designs to find out"
        )
    return result


def markdown(result: dict[str, Any]) -> str:
    lines = [f"# Plan sized to ${result['cap_usd']:.2f}", ""]
    if not result["ok"]:
        lines += [f"**{result['status']}**", "", result["refusal"], ""]
        return "\n".join(lines)
    if result["status"] == "sized-from-timings":
        # `plan` returns ok with no configured-rate total when an arm has no recorded rate, so
        # the priced lines below have nothing to read. Report the modelled sizing instead.
        modeled = result["modeled_plan"]
        largest = modeled.get("largest_plan") or {}
        lines += [
            f"**{result['status']}**",
            "",
            f"- designs: {result['designs']}",
            f"- modelled total: ${float(largest.get('total_usd', 0.0)):.6f}",
            f"- rate source: {modeled.get('rate_source')}",
            "",
            result["configured_rate_gap"],
            "",
            result["next_measurement"],
            "",
        ]
        if result.get("note"):
            lines += [result["note"], ""]
        return "\n".join(lines)
    lines += [
        f"- designs: {result['designs']}",
        f"- priced total: ${result['total_usd']:.6f}",
        f"- headroom: ${result['headroom_usd']:.6f}",
        f"- binding arm: {result['binding_arm']}",
        f"- rate evidence: {', '.join(result['rate_evidence'])}",
        "",
        "| Arm | Route class | Estimate USD |",
        "| --- | --- | --- |",
    ]
    for adapter_id, amount in result["per_arm_usd"].items():
        lines.append(
            f"| {adapter_id} | {result['route_classes'].get(adapter_id, '-')} | {amount:.6f} |"
        )
    check = result["timing_cross_check"]
    lines += ["", "## Measured-timing cross-check", ""]
    if check["available"]:
        lines += [
            f"- {check['designs_priced']} designs on {check['machine']}",
            f"- cold: ${check['cold_usd']:.2f} ({check['cold_evidence']})",
            f"- projected warm: ${check['projected_warm_usd']:.2f} ({check['projected_warm_evidence']})",
        ]
    else:
        lines.append(f"- not available: {check['reason']}")
    lines += ["", result["excluded_costs"], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="composed campaign configuration")
    parser.add_argument("--cap-usd", required=True, type=float, help="stated spend ceiling")
    parser.add_argument(
        "--cost-rate",
        action="append",
        default=[],
        metavar="ADAPTER_ID=USD_PER_DESIGN",
        help="a rate you measured on your own account; repeat per arm",
    )
    parser.add_argument(
        "--cloud-only",
        action="store_true",
        help="refuse a plan whose configuration runs any arm locally",
    )
    parser.add_argument("--max-designs", type=int, default=DEFAULT_MAX_DESIGNS)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    overrides = qualify._parse_cost_overrides(args.cost_rate) if args.cost_rate else None
    try:
        result = plan(
            config,
            cap_usd=args.cap_usd,
            overrides=overrides,
            max_designs=args.max_designs,
            cloud_only=args.cloud_only,
        )
    except (BudgetPlanError, qualify.QualificationError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    print(markdown(result) if args.markdown else json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
