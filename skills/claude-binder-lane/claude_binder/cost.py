"""Estimate campaign cost and serial wall clock from labelled evidence.

Every public amount carries the evidence status that supports it. Cold estimates
sum measured unit costs. Warm estimates remain projections until the
warm-worker patch has a deployed measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any


SECONDS_PER_HOUR = Decimal("3600")
DEFAULT_SCALES = (10, 30, 100)
REPORTED_REMAINING_BUDGET_USD = Decimal("410")


class CostModelError(ValueError):
    """A campaign setting exceeds the evidence supported by this model."""


class EvidenceStatus(str, Enum):
    """State whether a displayed quantity has been measured or projected."""

    MEASURED = "measured"
    MEASURED_ONCE = "measured once"
    PROJECTED = "projected"
    PRESUMPTION = "presumption"
    TODO = "TODO"


@dataclass(frozen=True)
class Evidence:
    """Identify the evidence and the measurement that would close a gap."""

    status: EvidenceStatus
    source: str
    detail: str
    experiment: str | None = None

    @property
    def label(self) -> str:
        """Return the explicit evidence label for user-facing values."""
        if self.status is EvidenceStatus.PRESUMPTION:
            return "PROJECTED PRESUMPTION"
        return self.status.value.upper()

    def as_dict(self) -> dict[str, str | None]:
        """Return serializable evidence for a report or selection surface."""
        return {
            "status": self.status.value,
            "label": self.label,
            "source": self.source,
            "detail": self.detail,
            "experiment": self.experiment,
        }


@dataclass(frozen=True)
class LabeledQuantity:
    """Represent one amount with the evidence label shown beside that amount."""

    name: str
    value: Decimal | None
    unit: str
    evidence: Evidence

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation without losing decimal precision."""
        return {
            "name": self.name,
            "value": None if self.value is None else format(self.value, "f"),
            "unit": self.unit,
            "evidence": self.evidence.as_dict(),
        }

    def display(self, places: int | None = None) -> str:
        """Render one explicitly labelled amount for a human reader."""
        if self.value is None:
            return f"{self.evidence.label} {self.name}: TODO. {self.evidence.experiment}"
        value = _format_decimal(self.value, places)
        return f"{self.evidence.label} {self.name}: {value} {self.unit}."


@dataclass(frozen=True)
class UnitCost:
    """Describe one measured, projected, or unknown unit cost."""

    name: str
    seconds_per_unit: LabeledQuantity
    usd_per_unit: LabeledQuantity
    evidence: Evidence

    def as_dict(self) -> dict[str, Any]:
        """Return the unit cost with labels on every displayed figure."""
        return {
            "name": self.name,
            "seconds_per_unit": self.seconds_per_unit.as_dict(),
            "usd_per_unit": self.usd_per_unit.as_dict(),
            "evidence": self.evidence.as_dict(),
        }


@dataclass(frozen=True)
class CampaignSettings:
    """Capture the cost choices a scientist makes before a campaign starts."""

    designs: int
    seeds: int = 1
    predictor_arms: int = 1
    rounds: int = 1
    counter_screen: bool = False
    pass_fraction: Decimal | str | float = Decimal("1")

    def __post_init__(self) -> None:
        _positive_integer(self.designs, "designs")
        _positive_integer(self.seeds, "seeds")
        _positive_integer(self.predictor_arms, "predictor_arms")
        _positive_integer(self.rounds, "rounds")
        if self.designs % 10:
            raise CostModelError(
                "designs must be a multiple of 10 because generation timing was measured in batches of 10"
            )
        pass_fraction = _decimal(self.pass_fraction, "pass_fraction")
        if pass_fraction < 0 or pass_fraction > 1:
            raise CostModelError("pass_fraction must be between 0 and 1")
        object.__setattr__(self, "pass_fraction", pass_fraction)

    @property
    def generated_design_count(self) -> Decimal:
        """Return the planned regenerated-design count across all rounds."""
        return Decimal(self.designs * self.rounds)

    @property
    def unfiltered_fold_count(self) -> Decimal:
        """Return the planned fold count before a pre-fold filter."""
        targets_per_fold = Decimal(2 if self.counter_screen else 1)
        return (
            Decimal(self.designs)
            * Decimal(self.seeds)
            * Decimal(self.predictor_arms)
            * Decimal(self.rounds)
            * targets_per_fold
        )

    @property
    def fold_count(self) -> Decimal:
        """Return the projected fold count after the configured pre-fold filter."""
        return self.unfiltered_fold_count * self.pass_fraction

    def with_pass_fraction(self, value: Decimal | str | float) -> CampaignSettings:
        """Return the same scale with a different projected filter pass fraction."""
        return CampaignSettings(
            designs=self.designs,
            seeds=self.seeds,
            predictor_arms=self.predictor_arms,
            rounds=self.rounds,
            counter_screen=self.counter_screen,
            pass_fraction=value,
        )


@dataclass(frozen=True)
class CostComponent:
    """Record one time and cost contribution to a campaign estimate."""

    name: str
    seconds: LabeledQuantity
    cost_usd: LabeledQuantity

    def as_dict(self) -> dict[str, Any]:
        """Return labelled component fields for an API caller."""
        return {
            "name": self.name,
            "seconds": self.seconds.as_dict(),
            "cost_usd": self.cost_usd.as_dict(),
        }


@dataclass(frozen=True)
class CostEstimate:
    """Return one cost, uptime, and serial wall-clock estimate."""

    settings: CampaignSettings
    mode: str
    evidence: Evidence
    components: tuple[CostComponent, ...]
    fold_count: LabeledQuantity
    total_seconds: LabeledQuantity
    total_cost_usd: LabeledQuantity
    wall_clock_hours: LabeledQuantity
    formula: str

    @property
    def cost_per_design_usd(self) -> LabeledQuantity:
        """Return the total campaign cost allocated across selected designs."""
        if self.total_cost_usd.value is None:
            value = None
        else:
            value = self.total_cost_usd.value / Decimal(self.settings.designs)
        return LabeledQuantity(
            name="cost per selected design",
            value=value,
            unit="USD per design",
            evidence=self.evidence,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a report-friendly estimate with evidence on every amount."""
        return {
            "mode": self.mode,
            "evidence": self.evidence.as_dict(),
            "settings": {
                "designs": self.settings.designs,
                "seeds": self.settings.seeds,
                "predictor_arms": self.settings.predictor_arms,
                "rounds": self.settings.rounds,
                "counter_screen": self.settings.counter_screen,
                "pass_fraction": format(self.settings.pass_fraction, "f"),
            },
            "formula": self.formula,
            "fold_count": self.fold_count.as_dict(),
            "components": [component.as_dict() for component in self.components],
            "total_seconds": self.total_seconds.as_dict(),
            "total_cost_usd": self.total_cost_usd.as_dict(),
            "wall_clock_hours": self.wall_clock_hours.as_dict(),
            "cost_per_design_usd": self.cost_per_design_usd.as_dict(),
        }


@dataclass(frozen=True)
class ScaleComparison:
    """Group the measured and projected estimates for one selectable scale."""

    settings: CampaignSettings
    cold: CostEstimate
    projected_warm_optimistic: CostEstimate
    projected_warm_cautious: CostEstimate
    published_baseline_cold: CostEstimate


@dataclass(frozen=True)
class FilterSavings:
    """Describe pre-fold filtering value without inventing its own unit cost."""

    settings: CampaignSettings
    pass_fraction: LabeledQuantity
    gross_fold_savings_usd: LabeledQuantity
    filter_cost_usd: LabeledQuantity
    net_savings_usd: LabeledQuantity
    formula: str


@dataclass(frozen=True)
class FilterThreshold:
    """Describe the filter pass rate required by a stated budget."""

    settings: CampaignSettings
    budget_usd: LabeledQuantity
    maximum_pass_fraction: LabeledQuantity
    savings_per_pass_percentage_point_usd: LabeledQuantity
    formula: str


@dataclass(frozen=True)
class WarmDeployExperiment:
    """Record the projected success and retention-failure paths for 20 folds."""

    fold_count: int
    success: LabeledQuantity
    retention_failure: LabeledQuantity
    success_seconds: LabeledQuantity
    retention_failure_seconds: LabeledQuantity


ANALYSIS_SOURCE = "Cost-model analysis supplied with this task"
RATE_SOURCE = "skills/claude-binder-lane/reference-gpu-rates-2026-08-23.json"
# The co-fold timing came from one bench run whose notes do not ship with the
# package, so cite the bench rather than a path the reader cannot open.
TIMING_SOURCE = "Co-fold timing bench, recorded once, with no openable record"

MEASURED_RATE_EVIDENCE = Evidence(
    EvidenceStatus.MEASURED,
    RATE_SOURCE,
    "A reference GPU-H100 rate of 0.00125 USD per second, read from one provider "
    "usage export. It is not a Modal rate. Re-measure on your own account.",
)
MEASURED_GENERATION_EVIDENCE = Evidence(
    EvidenceStatus.MEASURED,
    ANALYSIS_SOURCE,
    "Two RFdiffusion3 batches produced 20 target-conditioned backbones in 347.6 seconds.",
)
MEASURED_MPNN_EVIDENCE = Evidence(
    EvidenceStatus.MEASURED,
    ANALYSIS_SOURCE,
    "ProteinMPNN took 16.2 seconds per design.",
)
MEASURED_COFOLD_EVIDENCE = Evidence(
    EvidenceStatus.MEASURED_ONCE,
    TIMING_SOURCE,
    "One 229-residue cold co-fold recorded 200.177 fold seconds, 555.965 load seconds, and 885.598 total seconds.",
)
MEASURED_IDLE_TAIL_EVIDENCE = Evidence(
    EvidenceStatus.MEASURED_ONCE,
    ANALYSIS_SOURCE,
    "Recorded session uptime increased from 2458 seconds to 2716 seconds with the idle tail.",
)
PROJECTED_WARM_EVIDENCE = Evidence(
    EvidenceStatus.PROJECTED,
    ANALYSIS_SOURCE,
    "A warm worker requires an undeployed application patch and a redeploy.",
    "Deploy the warm-worker patch, retain one runner, and time 20 consecutive folds.",
)
PROJECTED_CAUTIOUS_WARM_EVIDENCE = Evidence(
    EvidenceStatus.PROJECTED,
    ANALYSIS_SOURCE,
    "The cautious bound applies the undeployed patch benchmark ratio of 13.6 divided by 31.6 to each cold fold.",
    "Deploy the warm-worker patch, retain one runner, and time 20 consecutive folds.",
)
FILTER_COST_EVIDENCE = Evidence(
    EvidenceStatus.PRESUMPTION,
    ANALYSIS_SOURCE,
    "The platform-route filter cost is presumed to be zero. No provider unit cost has been measured.",
    "Time the filter over designs that were already folded.",
)
FILTER_RUNTIME_EVIDENCE = Evidence(
    EvidenceStatus.TODO,
    ANALYSIS_SOURCE,
    "The pre-folding filter runtime has no measured unit cost.",
    "Time the filter over designs that were already folded.",
)
SIZE_SCALING_EVIDENCE = Evidence(
    EvidenceStatus.TODO,
    ANALYSIS_SOURCE,
    "Fold cost versus protein size has no measured scaling rule.",
    "Fold one design set against targets of several residue counts, then fit seconds per residue.",
)
SECOND_ARM_EVIDENCE = Evidence(
    EvidenceStatus.TODO,
    ANALYSIS_SOURCE,
    "The second predictor arm has no cold co-fold timing measurement.",
    "Time one cold co-fold through arm 2.",
)
MULTI_RUNNER_EVIDENCE = Evidence(
    EvidenceStatus.TODO,
    ANALYSIS_SOURCE,
    "The model assumes one runner serves all four applications. Multi-runner billing has no measurement.",
    "Read runner lines from a bill for a session that used more than one application.",
)

MEASURED_H100_RATE_USD_PER_SECOND = Decimal("0.00125")
MEASURED_GENERATION_SECONDS_PER_DESIGN = Decimal("17.38")
MEASURED_MPNN_SECONDS_PER_DESIGN = Decimal("16.2")
MEASURED_COLD_FOLD_SECONDS = Decimal("885.598")
MEASURED_FOLD_SECONDS = Decimal("200.177")
MEASURED_MODEL_LOAD_SECONDS = Decimal("555.965")
MEASURED_UNCLASSIFIED_COFOLD_SECONDS = (
    MEASURED_COLD_FOLD_SECONDS - MEASURED_FOLD_SECONDS - MEASURED_MODEL_LOAD_SECONDS
)
MEASURED_IDLE_TAIL_SECONDS = Decimal("258")
MEASURED_CAPPED_FOLD_SECONDS = Decimal("1219.2")
PATCH_BENCHMARK_RATIO = Decimal("13.6") / Decimal("31.6")


def _cost(seconds: Decimal) -> Decimal:
    return seconds * MEASURED_H100_RATE_USD_PER_SECOND


def _amount(name: str, value: Decimal | None, unit: str, evidence: Evidence) -> LabeledQuantity:
    return LabeledQuantity(name=name, value=value, unit=unit, evidence=evidence)


def _component(name: str, seconds: Decimal, evidence: Evidence) -> CostComponent:
    return CostComponent(
        name=name,
        seconds=_amount(f"{name} time", seconds, "seconds", evidence),
        cost_usd=_amount(f"{name} cost", _cost(seconds), "USD", evidence),
    )


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CostModelError(f"{name} must be a positive integer")


def _decimal(value: Decimal | str | float, name: str) -> Decimal:
    if isinstance(value, bool):
        raise CostModelError(f"{name} must be a finite decimal")
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CostModelError(f"{name} must be a finite decimal") from exc
    if not decimal_value.is_finite():
        raise CostModelError(f"{name} must be a finite decimal")
    return decimal_value


def _format_decimal(value: Decimal, places: int | None = None) -> str:
    if places is None:
        return format(value, "f")
    quantizer = Decimal(1).scaleb(-places)
    return format(value.quantize(quantizer, rounding=ROUND_HALF_UP), f".{places}f")


def format_usd(value: Decimal, places: int = 2) -> str:
    """Format USD with an explicit currency marker for UI callers."""
    return "$" + _format_decimal(value, places)


def unit_costs() -> tuple[UnitCost, ...]:
    """Return the 13 labelled unit-cost rows that bound this model."""
    return (
        UnitCost(
            "GPU-H100 runner rate",
            _amount("GPU-H100 runner time", Decimal("1"), "second", MEASURED_RATE_EVIDENCE),
            _amount(
                "GPU-H100 runner rate",
                MEASURED_H100_RATE_USD_PER_SECOND,
                "USD per H100 second",
                MEASURED_RATE_EVIDENCE,
            ),
            MEASURED_RATE_EVIDENCE,
        ),
        UnitCost(
            "RFdiffusion3 generated design",
            _amount(
                "RFdiffusion3 generated design time",
                MEASURED_GENERATION_SECONDS_PER_DESIGN,
                "seconds per design",
                MEASURED_GENERATION_EVIDENCE,
            ),
            _amount(
                "RFdiffusion3 generated design cost",
                _cost(MEASURED_GENERATION_SECONDS_PER_DESIGN),
                "USD per design",
                MEASURED_GENERATION_EVIDENCE,
            ),
            MEASURED_GENERATION_EVIDENCE,
        ),
        UnitCost(
            "ProteinMPNN designed sequence",
            _amount(
                "ProteinMPNN design time",
                MEASURED_MPNN_SECONDS_PER_DESIGN,
                "seconds per design",
                MEASURED_MPNN_EVIDENCE,
            ),
            _amount(
                "ProteinMPNN design cost",
                _cost(MEASURED_MPNN_SECONDS_PER_DESIGN),
                "USD per design",
                MEASURED_MPNN_EVIDENCE,
            ),
            MEASURED_MPNN_EVIDENCE,
        ),
        UnitCost(
            "cold co-fold",
            _amount(
                "cold co-fold time",
                MEASURED_COLD_FOLD_SECONDS,
                "seconds per fold",
                MEASURED_COFOLD_EVIDENCE,
            ),
            _amount(
                "cold co-fold cost",
                _cost(MEASURED_COLD_FOLD_SECONDS),
                "USD per fold",
                MEASURED_COFOLD_EVIDENCE,
            ),
            MEASURED_COFOLD_EVIDENCE,
        ),
        UnitCost(
            "cold co-fold fold work",
            _amount(
                "cold co-fold fold-work time",
                MEASURED_FOLD_SECONDS,
                "seconds per fold",
                MEASURED_COFOLD_EVIDENCE,
            ),
            _amount(
                "cold co-fold fold-work cost",
                _cost(MEASURED_FOLD_SECONDS),
                "USD per fold",
                MEASURED_COFOLD_EVIDENCE,
            ),
            MEASURED_COFOLD_EVIDENCE,
        ),
        UnitCost(
            "cold co-fold model load",
            _amount(
                "cold co-fold model-load time",
                MEASURED_MODEL_LOAD_SECONDS,
                "seconds per fold",
                MEASURED_COFOLD_EVIDENCE,
            ),
            _amount(
                "cold co-fold model-load cost",
                _cost(MEASURED_MODEL_LOAD_SECONDS),
                "USD per fold",
                MEASURED_COFOLD_EVIDENCE,
            ),
            MEASURED_COFOLD_EVIDENCE,
        ),
        UnitCost(
            "cold co-fold unclassified residual",
            _amount(
                "cold co-fold residual time",
                MEASURED_UNCLASSIFIED_COFOLD_SECONDS,
                "seconds per fold",
                MEASURED_COFOLD_EVIDENCE,
            ),
            _amount(
                "cold co-fold residual cost",
                _cost(MEASURED_UNCLASSIFIED_COFOLD_SECONDS),
                "USD per fold",
                MEASURED_COFOLD_EVIDENCE,
            ),
            MEASURED_COFOLD_EVIDENCE,
        ),
        UnitCost(
            "idle tail",
            _amount(
                "idle-tail time",
                MEASURED_IDLE_TAIL_SECONDS,
                "seconds per campaign",
                MEASURED_IDLE_TAIL_EVIDENCE,
            ),
            _amount(
                "idle-tail cost",
                _cost(MEASURED_IDLE_TAIL_SECONDS),
                "USD per campaign",
                MEASURED_IDLE_TAIL_EVIDENCE,
            ),
            MEASURED_IDLE_TAIL_EVIDENCE,
        ),
        UnitCost(
            "warm co-fold",
            _amount(
                "warm co-fold time",
                MEASURED_FOLD_SECONDS,
                "seconds per fold",
                PROJECTED_WARM_EVIDENCE,
            ),
            _amount(
                "warm co-fold cost",
                _cost(MEASURED_FOLD_SECONDS),
                "USD per fold",
                PROJECTED_WARM_EVIDENCE,
            ),
            PROJECTED_WARM_EVIDENCE,
        ),
        UnitCost(
            "fold cost versus protein size",
            _amount("fold size-scaling time", None, "seconds per residue", SIZE_SCALING_EVIDENCE),
            _amount("fold size-scaling cost", None, "USD per residue", SIZE_SCALING_EVIDENCE),
            SIZE_SCALING_EVIDENCE,
        ),
        UnitCost(
            "second predictor arm",
            _amount("second-arm co-fold time", None, "seconds per fold", SECOND_ARM_EVIDENCE),
            _amount("second-arm co-fold cost", None, "USD per fold", SECOND_ARM_EVIDENCE),
            SECOND_ARM_EVIDENCE,
        ),
        UnitCost(
            "multi-runner billing",
            _amount("additional-runner time", None, "seconds per campaign", MULTI_RUNNER_EVIDENCE),
            _amount("additional-runner cost", None, "USD per campaign", MULTI_RUNNER_EVIDENCE),
            MULTI_RUNNER_EVIDENCE,
        ),
        UnitCost(
            "pre-folding filter",
            _amount("pre-folding filter time", None, "seconds per design", FILTER_RUNTIME_EVIDENCE),
            _amount("pre-folding filter cost", None, "USD per design", FILTER_RUNTIME_EVIDENCE),
            FILTER_RUNTIME_EVIDENCE,
        ),
    )


def cold_formula() -> str:
    """Return the labelled formula for the deployed cold request path."""
    return (
        "MEASURED formula: seconds = rounds * designs * (17.38 + 16.2) + "
        "pass_fraction * rounds * designs * seeds * predictor_arms * "
        "(1 + counter_screen) * 885.598 + 258."
    )


def projected_warm_formula() -> str:
    """Return the labelled formula for the undeployed warm-worker path."""
    return (
        "PROJECTED formula: seconds = rounds * designs * (17.38 + 16.2) + "
        "pass_fraction * rounds * designs * seeds * predictor_arms * "
        "(1 + counter_screen) * 200.177 + 555.965 + 258."
    )


def estimate_cold(settings: CampaignSettings) -> CostEstimate:
    """Estimate the deployed cold path from measured component costs."""
    generation_seconds = settings.generated_design_count * MEASURED_GENERATION_SECONDS_PER_DESIGN
    sequence_seconds = settings.generated_design_count * MEASURED_MPNN_SECONDS_PER_DESIGN
    fold_seconds = settings.fold_count * MEASURED_COLD_FOLD_SECONDS
    components = (
        _component("generation", generation_seconds, MEASURED_GENERATION_EVIDENCE),
        _component("ProteinMPNN sequence design", sequence_seconds, MEASURED_MPNN_EVIDENCE),
        _component("cold co-folding", fold_seconds, MEASURED_COFOLD_EVIDENCE),
        _component("idle tail", MEASURED_IDLE_TAIL_SECONDS, MEASURED_IDLE_TAIL_EVIDENCE),
    )
    evidence = Evidence(
        EvidenceStatus.MEASURED,
        ANALYSIS_SOURCE,
        "The total is a component sum from measured unit costs. No campaign has completed end to end.",
    )
    total_seconds = sum(component.seconds.value or Decimal("0") for component in components)
    total_cost = _cost(total_seconds)
    return CostEstimate(
        settings=settings,
        mode="cold",
        evidence=evidence,
        components=components,
        fold_count=_amount(
            "planned post-filter fold count",
            settings.fold_count,
            "folds",
            Evidence(
                EvidenceStatus.PROJECTED,
                ANALYSIS_SOURCE,
                "The count is computed from the selected designs, seeds, arms, rounds, counter-screen, and pass fraction.",
            ),
        ),
        total_seconds=_amount("campaign uptime", total_seconds, "seconds", evidence),
        total_cost_usd=_amount("campaign cost", total_cost, "USD", evidence),
        wall_clock_hours=_amount(
            "serial wall clock", total_seconds / SECONDS_PER_HOUR, "hours", evidence
        ),
        formula=cold_formula(),
    )


def estimate_projected_warm(settings: CampaignSettings) -> CostEstimate:
    """Project a single warm worker that loads once and folds many times."""
    generation_seconds = settings.generated_design_count * MEASURED_GENERATION_SECONDS_PER_DESIGN
    sequence_seconds = settings.generated_design_count * MEASURED_MPNN_SECONDS_PER_DESIGN
    warm_fold_seconds = settings.fold_count * MEASURED_FOLD_SECONDS
    components = (
        _component("generation", generation_seconds, MEASURED_GENERATION_EVIDENCE),
        _component("ProteinMPNN sequence design", sequence_seconds, MEASURED_MPNN_EVIDENCE),
        _component("warm co-folding", warm_fold_seconds, PROJECTED_WARM_EVIDENCE),
        _component("one model load", MEASURED_MODEL_LOAD_SECONDS, PROJECTED_WARM_EVIDENCE),
        _component("idle tail", MEASURED_IDLE_TAIL_SECONDS, PROJECTED_WARM_EVIDENCE),
    )
    total_seconds = sum(component.seconds.value or Decimal("0") for component in components)
    total_cost = _cost(total_seconds)
    return CostEstimate(
        settings=settings,
        mode="projected-warm-optimistic",
        evidence=PROJECTED_WARM_EVIDENCE,
        components=components,
        fold_count=_amount(
            "projected post-filter warm fold count",
            settings.fold_count,
            "folds",
            PROJECTED_WARM_EVIDENCE,
        ),
        total_seconds=_amount("projected campaign uptime", total_seconds, "seconds", PROJECTED_WARM_EVIDENCE),
        total_cost_usd=_amount("projected campaign cost", total_cost, "USD", PROJECTED_WARM_EVIDENCE),
        wall_clock_hours=_amount(
            "projected serial wall clock",
            total_seconds / SECONDS_PER_HOUR,
            "hours",
            PROJECTED_WARM_EVIDENCE,
        ),
        formula=projected_warm_formula(),
    )


def estimate_projected_warm_cautious(settings: CampaignSettings) -> CostEstimate:
    """Project the cautious warm bound from the patch benchmark ratio."""
    generation_seconds = settings.generated_design_count * MEASURED_GENERATION_SECONDS_PER_DESIGN
    sequence_seconds = settings.generated_design_count * MEASURED_MPNN_SECONDS_PER_DESIGN
    cautious_fold_seconds = settings.fold_count * MEASURED_COLD_FOLD_SECONDS * PATCH_BENCHMARK_RATIO
    components = (
        _component("generation", generation_seconds, MEASURED_GENERATION_EVIDENCE),
        _component("ProteinMPNN sequence design", sequence_seconds, MEASURED_MPNN_EVIDENCE),
        _component("cautious warm co-folding", cautious_fold_seconds, PROJECTED_CAUTIOUS_WARM_EVIDENCE),
        _component("idle tail", MEASURED_IDLE_TAIL_SECONDS, PROJECTED_CAUTIOUS_WARM_EVIDENCE),
    )
    total_seconds = sum(component.seconds.value or Decimal("0") for component in components)
    total_cost = _cost(total_seconds)
    formula = (
        "PROJECTED cautious formula: seconds = rounds * designs * (17.38 + 16.2) + "
        "pass_fraction * rounds * designs * seeds * predictor_arms * "
        "(1 + counter_screen) * 885.598 * (13.6 / 31.6) + 258."
    )
    return CostEstimate(
        settings=settings,
        mode="projected-warm-cautious",
        evidence=PROJECTED_CAUTIOUS_WARM_EVIDENCE,
        components=components,
        fold_count=_amount(
            "projected post-filter cautious warm fold count",
            settings.fold_count,
            "folds",
            PROJECTED_CAUTIOUS_WARM_EVIDENCE,
        ),
        total_seconds=_amount(
            "projected cautious campaign uptime",
            total_seconds,
            "seconds",
            PROJECTED_CAUTIOUS_WARM_EVIDENCE,
        ),
        total_cost_usd=_amount(
            "projected cautious campaign cost",
            total_cost,
            "USD",
            PROJECTED_CAUTIOUS_WARM_EVIDENCE,
        ),
        wall_clock_hours=_amount(
            "projected cautious serial wall clock",
            total_seconds / SECONDS_PER_HOUR,
            "hours",
            PROJECTED_CAUTIOUS_WARM_EVIDENCE,
        ),
        formula=formula,
    )


def published_baseline_settings(designs: int) -> CampaignSettings:
    """Return the cold baseline scale with five seeds, two arms, and a counter-screen."""
    return CampaignSettings(
        designs=designs,
        seeds=5,
        predictor_arms=2,
        rounds=1,
        counter_screen=True,
    )


def scale_comparisons(scales: tuple[int, ...] = DEFAULT_SCALES) -> tuple[ScaleComparison, ...]:
    """Return user-selectable scales with measured cold and projected warm costs."""
    comparisons: list[ScaleComparison] = []
    for designs in scales:
        settings = CampaignSettings(designs=designs)
        comparisons.append(
            ScaleComparison(
                settings=settings,
                cold=estimate_cold(settings),
                projected_warm_optimistic=estimate_projected_warm(settings),
                projected_warm_cautious=estimate_projected_warm_cautious(settings),
                published_baseline_cold=estimate_cold(published_baseline_settings(designs)),
            )
        )
    return tuple(comparisons)


def estimate_filter_savings(
    settings: CampaignSettings,
    pass_fraction: Decimal | str | float,
) -> FilterSavings:
    """Estimate gross fold savings and label the unmeasured filter-cost presumption."""
    pass_fraction_decimal = _decimal(pass_fraction, "pass_fraction")
    if pass_fraction_decimal < 0 or pass_fraction_decimal > 1:
        raise CostModelError("pass_fraction must be between 0 and 1")
    avoided_fold_count = settings.unfiltered_fold_count * (Decimal("1") - pass_fraction_decimal)
    gross_savings = avoided_fold_count * _cost(MEASURED_COLD_FOLD_SECONDS)
    presumed_filter_cost = Decimal("0")
    net_savings = gross_savings - settings.generated_design_count * presumed_filter_cost
    formula = (
        "PRESUMPTION formula: net savings = (1 - pass_fraction) * rounds * designs * "
        "seeds * predictor_arms * (1 + counter_screen) * measured cold-fold cost - "
        "rounds * designs * presumed filter cost."
    )
    return FilterSavings(
        settings=settings,
        pass_fraction=_amount(
            "projected filter pass fraction",
            pass_fraction_decimal,
            "fraction",
            Evidence(
                EvidenceStatus.PROJECTED,
                ANALYSIS_SOURCE,
                "The pass fraction requires a filter experiment on folded designs.",
                "Run the filter over designs that were already folded, then report its observed pass rate.",
            ),
        ),
        gross_fold_savings_usd=_amount(
            "gross fold savings",
            gross_savings,
            "USD",
            Evidence(
                EvidenceStatus.MEASURED,
                ANALYSIS_SOURCE,
                "The amount multiplies avoided folds by the measured cold-fold unit cost.",
            ),
        ),
        filter_cost_usd=_amount(
            "pre-fold filter cost",
            Decimal("0"),
            "USD",
            FILTER_COST_EVIDENCE,
        ),
        net_savings_usd=_amount("net filter savings", net_savings, "USD", FILTER_COST_EVIDENCE),
        formula=formula,
    )


def filter_pass_threshold(
    settings: CampaignSettings,
    budget_usd: Decimal | str | float = REPORTED_REMAINING_BUDGET_USD,
) -> FilterThreshold:
    """Return the affordable filter pass fraction under a stated budget.

    ``budget_usd`` defaults to the figure the packaged analysis reported, which
    belongs to that analysis and not to the caller's account. A scientist
    planning their own campaign passes their own ceiling. The default exists so
    the packaged worked example reproduces, not as an estimate of what anyone
    has left to spend.
    """
    budget = _decimal(budget_usd, "budget_usd")
    if budget < 0:
        raise CostModelError("budget_usd must be zero or greater")
    generation_and_sequence_cost = settings.generated_design_count * _cost(
        MEASURED_GENERATION_SECONDS_PER_DESIGN + MEASURED_MPNN_SECONDS_PER_DESIGN
    )
    tail_cost = _cost(MEASURED_IDLE_TAIL_SECONDS)
    fold_cost = settings.unfiltered_fold_count * _cost(MEASURED_COLD_FOLD_SECONDS)
    maximum_pass_fraction = (budget - tail_cost - generation_and_sequence_cost) / fold_cost
    savings_per_point = fold_cost / Decimal("100")
    budget_evidence = Evidence(
        EvidenceStatus.MEASURED,
        ANALYSIS_SOURCE,
        "The analysis reports approximately 410 USD remaining from the 500 USD budget. "
        "That is the packaged analysis's own figure, not a live balance on the caller's account.",
    )
    formula = (
        "PRESUMPTION formula: maximum pass fraction = (budget - measured idle-tail cost - "
        "measured generation-and-sequence cost) / measured unfiltered-fold cost. "
        "The formula presumes a zero-cost pre-fold filter."
    )
    return FilterThreshold(
        settings=settings,
        budget_usd=_amount("reported remaining budget", budget, "USD", budget_evidence),
        maximum_pass_fraction=_amount(
            "maximum affordable filter pass fraction",
            maximum_pass_fraction,
            "fraction",
            FILTER_COST_EVIDENCE,
        ),
        savings_per_pass_percentage_point_usd=_amount(
            "savings per filter pass percentage point",
            savings_per_point,
            "USD per percentage point",
            Evidence(
                EvidenceStatus.MEASURED,
                ANALYSIS_SOURCE,
                "The amount multiplies one percent of the unfiltered fold count by the measured cold-fold unit cost.",
            ),
        ),
        formula=formula,
    )


def warm_deploy_experiment(fold_count: int = 20) -> WarmDeployExperiment:
    """Price the projected warm-worker experiment without making a provider call."""
    _positive_integer(fold_count, "fold_count")
    count = Decimal(fold_count)
    success_seconds = (
        count * MEASURED_FOLD_SECONDS
        + MEASURED_MODEL_LOAD_SECONDS
        + MEASURED_IDLE_TAIL_SECONDS
    )
    retention_failure_seconds = count * MEASURED_COLD_FOLD_SECONDS + MEASURED_IDLE_TAIL_SECONDS
    return WarmDeployExperiment(
        fold_count=fold_count,
        success=_amount(
            "projected warm-worker experiment cost",
            _cost(success_seconds),
            "USD",
            PROJECTED_WARM_EVIDENCE,
        ),
        retention_failure=_amount(
            "projected retention-failure experiment cost",
            _cost(retention_failure_seconds),
            "USD",
            PROJECTED_WARM_EVIDENCE,
        ),
        success_seconds=_amount(
            "projected warm-worker experiment uptime",
            success_seconds,
            "seconds",
            PROJECTED_WARM_EVIDENCE,
        ),
        retention_failure_seconds=_amount(
            "projected retention-failure experiment uptime",
            retention_failure_seconds,
            "seconds",
            PROJECTED_WARM_EVIDENCE,
        ),
    )


def model_load_share_of_total() -> LabeledQuantity:
    """Return the exact recorded model-load share of the cold co-fold total."""
    return _amount(
        "model-load share of recorded cold co-fold total",
        MEASURED_MODEL_LOAD_SECONDS / MEASURED_COLD_FOLD_SECONDS,
        "fraction",
        MEASURED_COFOLD_EVIDENCE,
    )


def model_load_share_of_fold_and_load() -> LabeledQuantity:
    """Return the fold-plus-load share that rounds to the analysis's 74 percent."""
    return _amount(
        "model-load share of recorded fold-plus-load seconds",
        MEASURED_MODEL_LOAD_SECONDS / (MEASURED_MODEL_LOAD_SECONDS + MEASURED_FOLD_SECONDS),
        "fraction",
        MEASURED_COFOLD_EVIDENCE,
    )


def format_scale_selection(settings: CampaignSettings) -> str:
    """Render concise pre-commit text with evidence beside every model figure."""
    cold = estimate_cold(settings)
    warm_optimistic = estimate_projected_warm(settings)
    warm_cautious = estimate_projected_warm_cautious(settings)
    baseline = estimate_cold(published_baseline_settings(settings.designs))
    capped_fold_cost = _cost(MEASURED_CAPPED_FOLD_SECONDS)
    total_load_share = model_load_share_of_total().value or Decimal("0")
    settings_text = (
        f"{settings.designs} designs, {settings.seeds} seed{'s' if settings.seeds != 1 else ''} each, "
        f"{settings.predictor_arms} predictor arm{'s' if settings.predictor_arms != 1 else ''}, "
        f"{settings.rounds} round{'s' if settings.rounds != 1 else ''}, "
        f"{'with' if settings.counter_screen else 'without'} a counter-screen"
    )
    cold_cost = cold.total_cost_usd.value or Decimal("0")
    cold_hours = cold.wall_clock_hours.value or Decimal("0")
    warm_low_cost = warm_optimistic.total_cost_usd.value or Decimal("0")
    warm_high_cost = warm_cautious.total_cost_usd.value or Decimal("0")
    warm_low_hours = warm_optimistic.wall_clock_hours.value or Decimal("0")
    warm_high_hours = warm_cautious.wall_clock_hours.value or Decimal("0")
    baseline_cost = baseline.total_cost_usd.value or Decimal("0")
    baseline_days = (baseline.wall_clock_hours.value or Decimal("0")) / Decimal("24")
    return "\n".join(
        (
            f"{settings.designs} designs",
            f"Selected settings: {settings_text}.",
            f"MEASURED component-sum cost: about {format_usd(cold_cost.quantize(Decimal('1'), rounding=ROUND_HALF_UP), 0)}.",
            f"MEASURED component-sum serial wall clock: about {_format_decimal(cold_hours.quantize(Decimal('1'), rounding=ROUND_HALF_UP), 0)} hours.",
            f"MEASURED component-sum cost per selected design: {format_usd(cold.cost_per_design_usd.value or Decimal('0'))}.",
            "MEASURED fold basis: one 229-residue cold co-fold recorded 885.598 seconds.",
            f"MEASURED model-load share: {_format_decimal(total_load_share * Decimal('100'), 1)} percent of the recorded cold co-fold total.",
            f"PROJECTED warm-serving cost range: {format_usd(warm_low_cost)} to {format_usd(warm_high_cost)}.",
            f"PROJECTED warm-serving serial wall-clock range: {_format_decimal(warm_low_hours.quantize(Decimal('1'), rounding=ROUND_HALF_UP), 0)} to {_format_decimal(warm_high_hours.quantize(Decimal('1'), rounding=ROUND_HALF_UP), 0)} hours.",
            "PROJECTED warm-serving basis: the required application patch is undeployed.",
            f"MEASURED capped-fold observation: {format_usd(capped_fold_cost)} paid for a 1219.2-second request that returned nothing.",
            f"MEASURED published-baseline component sum: {format_usd(baseline_cost)} and {_format_decimal(baseline_days, 2)} days for 5 seeds, 2 arms, and a counter-screen.",
            "TODO: Measure fold cost on targets of several residue counts before applying this model to a larger target.",
            "TODO: Measure a deployed warm worker before treating either warm range as a commitment.",
        )
    )
