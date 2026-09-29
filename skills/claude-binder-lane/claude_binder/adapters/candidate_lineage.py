"""Shared provenance fields for candidate selection and optimization."""

from __future__ import annotations

from typing import Any


DIVERSITY_LINEAGE_FIELDS = (
    "root_backbone_id",
    "tm90_cluster_id",
    "structure_method",
    "seq_method",
    "fold_class",
)


def backbone_lineage(candidate_id: str, structure_method: str) -> dict[str, Any]:
    """Return the immutable diversity lineage for a new de novo backbone.

    The lane has no TM-score clustering stage or fold-classifier stage. A singleton
    TM90 cluster is therefore the conservative identity available at generation time,
    and ``unknown`` records that fold classification was not performed. Both values
    remain explicit so promotion cannot silently treat missing provenance as diversity.
    """
    return {
        "root_backbone_id": candidate_id,
        "tm90_cluster_id": candidate_id,
        "structure_method": structure_method,
        "seq_method": "none",
        "fold_class": "unknown",
    }
