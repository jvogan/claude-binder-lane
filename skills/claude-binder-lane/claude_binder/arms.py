"""Map a predictor id to the published instrument token that names its arm.

The map lives here rather than in `lane` because `lane` imports `validation_gate`
and both need it. A second copy in the other module is the shape of defect this
project keeps finding, where two files agree today and diverge silently later.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

SCORE_INSTRUMENT_ARM_NAMES = {
    "esmfold2": "ef2full",
    "esmfold2-fast": "ef2fast",
    "protenix-v2": "ptxv2",
    "alphafold-multimer-v3": "afm",
}

# ESMFold2-Full and ESMFold2-Fast differ in checkpoint and MSA policy. They
# share the ESMFold2 embedding family, so their agreement measures variation
# within one predictor lineage.
# The hosted Boltz route and the local Boltz CLI are two routes to one model
# family, which `catalog.json` records as the single lineage `boltz-2`. Counting
# them as two instruments would claim an independent agreement that does not
# exist, so they share one lineage here as well.
PREDICTOR_LINEAGE_IDS = {
    "esmfold2": "esmfold2",
    "esmfold2-fast": "esmfold2",
    "protenix-v2": "protenix-v2",
    "alphafold-multimer-v3": "alphafold-multimer-v3",
    "boltz": "boltz",
    "boltz-local": "boltz",
}
SCORE_INSTRUMENT_LINEAGE_IDS = {
    "ef2full": "esmfold2",
    "ef2fast": "esmfold2",
    "ptxv2": "protenix-v2",
    "afm": "alphafold-multimer-v3",
    "boltz1": "boltz",
    "boltz2": "boltz",
}


def score_instrument_arm_name(predictor_id: str) -> str:
    """Return the published instrument token for one predictor."""
    return SCORE_INSTRUMENT_ARM_NAMES.get(
        predictor_id,
        re.sub(r"[^a-z0-9]+", "_", predictor_id.lower()).strip("_"),
    )


def predictor_lineage_id(predictor_id: str) -> str:
    """Return the lineage that supplies a configured predictor mode."""
    registered = registered_predictor_lineage_id(predictor_id)
    return predictor_id if registered is None else registered


def registered_predictor_lineage_id(predictor_id: str) -> str | None:
    """Return the lineage this package registers for a predictor id, or None.

    None means the package holds no lineage fact about the tool. Treating the id
    itself as a lineage is a display fallback, not a measurement, so a caller
    that is about to claim independence has to see the difference.

    A configuration may name a predictor by its published instrument token rather
    than by its tool id, and `ef2fast` is the same instrument as `esmfold2-fast`.
    Both spellings therefore resolve from the maps above.
    """
    registered = PREDICTOR_LINEAGE_IDS.get(predictor_id)
    if registered is not None:
        return registered
    return SCORE_INSTRUMENT_LINEAGE_IDS.get(predictor_id)


def configured_predictor_lineage_id(predictor: Mapping[str, Any]) -> str | None:
    """Return the lineage one configured predictor entry establishes, or None.

    A registered predictor carries its lineage in this module, and that value
    wins, so a configuration cannot relabel two modes of one lineage as
    independent instruments. An operator's own tool carries a lineage only when
    the entry declares `lineage_id`, because nothing in the package measures
    which model family someone else's checkpoint belongs to.
    """
    predictor_id = predictor.get("id")
    if isinstance(predictor_id, str) and predictor_id:
        registered = registered_predictor_lineage_id(predictor_id)
        if registered is not None:
            return registered
    declared = predictor.get("lineage_id")
    if isinstance(declared, str) and declared:
        return declared
    return None


def duplicate_predictor_ids(predictor_ids: Iterable[str]) -> tuple[str, ...]:
    """Return every predictor id supplied more than once, in first-seen order."""
    counts: dict[str, int] = {}
    for predictor_id in predictor_ids:
        counts[predictor_id] = counts.get(predictor_id, 0) + 1
    return tuple(
        predictor_id for predictor_id, count in counts.items() if count > 1
    )


def score_instrument_lineage_id(arm: str) -> str:
    """Return the lineage that supplies one stored score-instrument arm."""
    return SCORE_INSTRUMENT_LINEAGE_IDS.get(arm, arm)


def enabled_predictor_arms(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the arm token of every enabled co-folding predictor, in config order.

    A predictor with no explicit `enabled` key counts as enabled, because the
    shipped profiles write `enabled: true` and a profile that omits it is
    declaring a predictor it means to run.
    """
    cofold = config.get("cofold")
    if not isinstance(cofold, Mapping):
        return ()
    predictors = cofold.get("predictors")
    if not isinstance(predictors, Sequence) or isinstance(predictors, (str, bytes)):
        return ()
    arms: list[str] = []
    for predictor in predictors:
        if not isinstance(predictor, Mapping):
            continue
        if predictor.get("enabled") is False:
            continue
        predictor_id = predictor.get("id")
        if not isinstance(predictor_id, str) or not predictor_id:
            continue
        arms.append(score_instrument_arm_name(predictor_id))
    return tuple(dict.fromkeys(arms))


def enabled_predictor_lineages(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the distinct co-folding lineages selected by a configuration."""
    cofold = config.get("cofold")
    if not isinstance(cofold, Mapping):
        return ()
    predictors = cofold.get("predictors")
    if not isinstance(predictors, Sequence) or isinstance(predictors, (str, bytes)):
        return ()
    lineages: list[str] = []
    for predictor in predictors:
        if not isinstance(predictor, Mapping) or predictor.get("enabled") is False:
            continue
        predictor_id = predictor.get("id")
        if not isinstance(predictor_id, str) or not predictor_id:
            continue
        lineages.append(
            configured_predictor_lineage_id(predictor) or predictor_id
        )
    return tuple(dict.fromkeys(lineages))
