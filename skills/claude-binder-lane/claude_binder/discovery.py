"""Summarize an unconstrained screening run without choosing an epitope."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .tool_menu import (
    default_tool_selection,
    resolve_tool_menu,
    validate_tool_selection,
)


# The task request defines a small pilot as about ten designs and tens of folds.
# These defaults make that request executable. A campaign can lower either cap.
DEFAULT_MAX_DESIGNS = 10
DEFAULT_MAX_FOLDS = 30


class DiscoveryError(ValueError):
    """The supplied discovery configuration or screening rows are incomplete."""


def discovery_settings(site: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return normalized discovery settings when the target selects discovery."""
    value = site.get("discovery")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DiscoveryError("targets[].site.discovery must be an object")
    if value.get("enabled") is not True:
        raise DiscoveryError("targets[].site.discovery.enabled must be true when discovery is present")
    settings = dict(value)
    settings.setdefault("max_designs", DEFAULT_MAX_DESIGNS)
    settings.setdefault("max_folds", DEFAULT_MAX_FOLDS)
    return settings


def is_unconstrained_site(site: Mapping[str, Any]) -> bool:
    """Return whether the target explicitly selected unconstrained discovery."""
    return discovery_settings(site) is not None


def normalize_discovery_settings(site: dict[str, Any]) -> dict[str, Any] | None:
    """Store discovery defaults and the durable constraint tag on one site."""
    settings = discovery_settings(site)
    if settings is None:
        site["epitope_constraint"] = "constrained"
        return None
    site["discovery"] = settings
    site["epitope_constraint"] = "unconstrained"
    return settings


def _contact_residues(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the persisted target-contact set from one scored screening row."""
    values = row.get("target_contact_residues")
    if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values):
        candidate_id = row.get("candidate_id", "unknown")
        raise DiscoveryError(
            f"screening row for {candidate_id!r} has no target_contact_residues from binder_metrics"
        )
    return tuple(sorted(set(values)))


def _score_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep measured screening scores without imposing a selection cutoff."""
    return {
        key: row[key]
        for key in ("candidate_id", "predictor", "seed", "ipsae_min", "sc_dockq", "iptm")
        if key in row
    }


def cluster_screen_contacts(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Group scored candidate models by their exact contacted target-residue set."""
    groups: dict[tuple[str, tuple[str, ...]], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("control_type", "candidate") != "candidate" or row.get("status") != "scored":
            continue
        target_id = row.get("target_id")
        if not isinstance(target_id, str) or not target_id:
            raise DiscoveryError("a scored screening row must name target_id")
        groups[(target_id, _contact_residues(row))].append(row)

    clusters: list[dict[str, Any]] = []
    for index, ((target_id, residues), members) in enumerate(sorted(groups.items()), start=1):
        design_ids = sorted(
            {
                str(member["candidate_id"])
                for member in members
                if isinstance(member.get("candidate_id"), str) and member["candidate_id"]
            }
        )
        clusters.append(
            {
                "cluster_id": f"contact-cluster-{index:03d}",
                "target_id": target_id,
                "surface_patch_residues": list(residues),
                "design_count": len(design_ids),
                "design_ids": design_ids,
                "screening_model_count": len(members),
                "score_records": [_score_record(member) for member in members],
            }
        )
    return {
        "schema_version": 1,
        "clustering": "exact-target-contact-residue-set",
        "selection_cutoff": None,
        "cluster_count": len(clusters),
        "clusters": clusters,
    }


def cluster_screen_contacts_path(path: Path) -> dict[str, Any]:
    """Load a JSONL screen table and summarize its persisted contact sets."""
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, TypeError) as exc:
        raise DiscoveryError(f"could not read screening observations from {path}: {exc}") from exc
    return cluster_screen_contacts(rows)
