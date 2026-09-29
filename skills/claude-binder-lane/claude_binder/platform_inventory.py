"""Normalize caller-supplied Claude Science tool and compute inventory.

The package never reaches into the host from this module. A Claude Science
session collects the live records through its own tools and passes the values
here. Normalization keeps volatile account details out of the packaged catalog.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = 1

# These names are discovery hints. A visible skill proves that its runbook is
# readable. It does not prove that its executable, weights, endpoint, or compute
# environment is ready.
PLATFORM_SKILL_HINTS = {
    "alphafold-multimer-v3": "alphafold2",
    "boltz": "boltz",
    "chai1": "chai1",
    "diffdock": "diffdock",
    "esmfold2": "esmfold2",
    "esmfold2-fast": "esmfold2",
    "esmfold2-native-design": "esmfold2",
    "fair-esm2": "fair-esm2",
    "ligandmpnn": "ligandmpnn",
    "openfold3": "openfold3",
    "proteinmpnn": "proteinmpnn",
    "solublempnn": "solublempnn",
}


def _records(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _skill(record: Any) -> dict[str, Any] | None:
    if isinstance(record, str) and record:
        return {
            "name": record,
            "origin": None,
            "description": None,
            "skill_id": None,
            "plugin_id": None,
            "plugin_release": None,
            "capability_kind": "guidance",
            "source_digest": None,
        }
    if not isinstance(record, Mapping):
        return None
    name = _text(record.get("name"))
    if name is None:
        return None
    return {
        "name": name,
        "origin": _text(record.get("origin")),
        "description": _text(record.get("description")),
        "skill_id": _text(record.get("skill_id")) or _text(record.get("id")),
        "plugin_id": _text(record.get("plugin_id")),
        "plugin_release": _text(record.get("plugin_release")),
        "capability_kind": _text(record.get("capability_kind")) or "guidance",
        "source_digest": _text(record.get("source_digest")),
    }


def _route(record: Any) -> dict[str, Any] | None:
    if not isinstance(record, Mapping):
        return None
    name = _text(record.get("name")) or _text(record.get("slug")) or _text(record.get("id"))
    provider = _text(record.get("provider")) or _text(record.get("provider_id"))
    skill_name = _text(record.get("skillName")) or _text(record.get("skill_name"))
    if name is None and provider is None and skill_name is None:
        return None
    return {
        "name": name,
        "provider": provider,
        "kind": _text(record.get("kind")) or _text(record.get("type")),
        "location": _text(record.get("location")),
        "skill_name": skill_name,
        "status": _text(record.get("status")) or "unknown",
    }


def normalize_platform_inventory(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a safe, deterministic inventory from already collected records."""
    if not isinstance(snapshot, Mapping):
        return {
            "schema_version": SCHEMA_VERSION,
            "source": "not-supplied",
            "skills_complete": False,
            "compute_complete": False,
            "provenance": {},
            "skills": [],
            "compute": [],
        }
    skills = [item for raw in _records(snapshot.get("skills")) if (item := _skill(raw))]
    compute = [item for raw in _records(snapshot.get("compute")) if (item := _route(raw))]
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "caller-supplied-live-snapshot",
        "skills_complete": snapshot.get("skills_complete") is True,
        "compute_complete": snapshot.get("compute_complete") is True,
        "provenance": {
            key: value
            for key in ("observed_at", "host_surface", "host_release", "source_digest")
            if (value := _text(snapshot.get(key))) is not None
        },
        "skills": sorted(skills, key=lambda item: item["name"]),
        "compute": sorted(
            compute,
            key=lambda item: (
                item.get("provider") or "",
                item.get("name") or "",
                item.get("skill_name") or "",
            ),
        ),
    }


def _name_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return {token for token in re.split(r"[-_./:]+", value.lower()) if token}


def discovery_for_tool(
    tool_id: str,
    *,
    inventory: Mapping[str, Any] | None,
    platform_skill: str | None = None,
) -> dict[str, Any]:
    """Report live discovery separately from package binding and validation."""
    normalized = normalize_platform_inventory(inventory)
    skill_name = platform_skill or PLATFORM_SKILL_HINTS.get(tool_id)
    skill = next(
        (
            item
            for item in normalized["skills"]
            if skill_name is not None and item.get("name") == skill_name
        ),
        None,
    )
    # Matching on skill name alone hid a registered endpoint whose skill is not
    # a per-tool runbook. An endpoint registered against this campaign skill,
    # rfdiffusion-service, named its tool in its own slug and reached no tool
    # row. Keep the two bases distinct so a reader can tell them apart.
    routes = []
    for item in normalized["compute"]:
        by_skill = skill_name is not None and item.get("skill_name") == skill_name
        by_name = tool_id.replace("-", "") in {
            token.replace("-", "") for token in _name_tokens(item.get("name"))
        }
        if not (by_skill or by_name):
            continue
        basis = "skill-name-and-endpoint-name" if by_skill and by_name else (
            "skill-name" if by_skill else "endpoint-name"
        )
        routes.append({**item, "match_basis": basis})
    if skill is not None or routes:
        status = "present"
    elif normalized["skills_complete"] and skill_name is not None:
        status = "absent"
    else:
        status = "unknown"
    return {
        "status": status,
        "skill_name": skill_name,
        "skill": skill,
        "routes": routes,
        "meaning": (
            "Live discovery only. Presence does not establish Binder adapter, runtime, scientific, cost, or licence readiness."
        ),
    }
