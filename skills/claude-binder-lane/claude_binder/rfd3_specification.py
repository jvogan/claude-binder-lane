"""Derive an RFdiffusion3 specification from one resolved campaign.

The RFdiffusion3 specification format carries a contig and atom-pair hotspot
selection. This module keeps those derivation rules independent of stage
execution so callers can test them with an in-memory campaign and structure.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


class SpecificationError(ValueError):
    """A campaign or structure cannot provide a valid RFdiffusion3 field."""


@dataclass(frozen=True)
class ResidueRecord:
    """The residue identity and atom names read from one target structure."""

    chain_id: str
    residue_number: int
    residue_name: str
    atom_names: frozenset[str]
    insertion_code: str = ""

    @property
    def residue_id(self) -> str:
        """Return the lane residue identifier for this record."""
        return f"{self.chain_id}:{self.residue_number}{self.insertion_code}"

    @property
    def specification_id(self) -> str:
        """Return the RFdiffusion3 residue identifier without a colon."""
        return f"{self.chain_id}{self.residue_number}{self.insertion_code}"


RESIDUE_ID_RE = re.compile(r"^([A-Za-z0-9]+):(-?\d+)([A-Za-z]?)$")
SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9]+")

# These are the atom pairs RFdiffusion3 accepts for the residue types that the
# campaign can select as generation hotspots. ALA and GLY have no side-chain
# tip pair.
# A campaign may declare dialect, infer_ori_strategy, and is_non_loopy in its
# rfd3_specification object. When it does not, this module omits them instead
# of guessing RFdiffusion3 source choices.
SIDE_CHAIN_TIP_ATOMS: dict[str, tuple[str, str]] = {
    "ARG": ("CZ", "NH1"),
    "ASN": ("OD1", "ND2"),
    "ASP": ("OD1", "OD2"),
    "CYS": ("CB", "SG"),
    "GLN": ("OE1", "NE2"),
    "GLU": ("OE1", "OE2"),
    "HIS": ("CD2", "NE2"),
    "ILE": ("CG1", "CG2"),
    "LEU": ("CD1", "CD2"),
    "LYS": ("CE", "NZ"),
    "MET": ("SD", "CE"),
    "PHE": ("CE1", "CZ"),
    "PRO": ("CG", "CD"),
    "SER": ("CB", "OG"),
    "THR": ("OG1", "CG2"),
    "TRP": ("CZ2", "CH2"),
    "TYR": ("CZ", "OH"),
    "VAL": ("CG1", "CG2"),
}
# The connectivity graph makes a fallback follow the modelled side chain instead
# of keeping a second, residue-specific fallback atom-pair table. The graph
# starts at CB because a residue needs at least two atoms beyond CB before it
# can provide directional hotspot information.
SIDE_CHAIN_BONDS: dict[str, tuple[tuple[str, str], ...]] = {
    "ARG": (
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "NE"),
        ("NE", "CZ"),
        ("CZ", "NH1"),
        ("CZ", "NH2"),
    ),
    "ASN": (("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")),
    "ASP": (("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")),
    "GLN": (("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")),
    "GLU": (("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")),
    "HIS": (
        ("CB", "CG"),
        ("CG", "ND1"),
        ("CG", "CD2"),
        ("ND1", "CE1"),
        ("CE1", "NE2"),
        ("NE2", "CD2"),
    ),
    "ILE": (("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")),
    "LYS": (("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")),
    "MET": (("CB", "CG"), ("CG", "SD"), ("SD", "CE")),
    "SER": (("CB", "OG"),),
    "TYR": (
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
        ("CZ", "OH"),
    ),
    "VAL": (("CB", "CG1"), ("CB", "CG2")),
}
FALLBACK_REASON = (
    "declared tip atoms are absent from coordinates; selected the two outermost "
    "present side-chain atoms beyond CB"
)
NO_SIDE_CHAIN_TIP = frozenset({"ALA", "GLY"})
UNRESOLVED_OPTIONAL_FIELDS = (
    "dialect",
    "infer_ori_strategy",
    "is_non_loopy",
)
GENERIC_RESIDUE_PARENT_NAMES = frozenset({"", ".", "inputs", "residue-maps", "targets"})


def residue_records_from_atoms(atoms: Iterable[Any]) -> list[ResidueRecord]:
    """Group adapter atom records into residue records for specification derivation."""
    grouped: dict[tuple[str, int, str], tuple[str, set[str]]] = {}
    for atom in atoms:
        chain_id = str(getattr(atom, "chain_id", "")).strip()
        residue_number = getattr(atom, "residue_number", None)
        insertion_code = str(getattr(atom, "insertion_code", "")).strip()
        residue_name = str(getattr(atom, "residue_name", "")).strip().upper()
        atom_name = str(getattr(atom, "name", "")).strip()
        if not chain_id or not isinstance(residue_number, int) or not residue_name or not atom_name:
            raise SpecificationError("structure atom records must carry chain, number, residue, and atom names")
        key = (chain_id, residue_number, insertion_code)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = (residue_name, {atom_name})
        else:
            existing_name, existing_atoms = existing
            if existing_name != residue_name:
                raise SpecificationError(
                    f"structure assigns {existing_name} and {residue_name} to {chain_id}:{residue_number}"
                )
            existing_atoms.add(atom_name)
    return [
        ResidueRecord(
            chain_id=chain_id,
            residue_number=residue_number,
            insertion_code=insertion_code,
            residue_name=residue_name,
            atom_names=frozenset(atom_names),
        )
        for (chain_id, residue_number, insertion_code), (residue_name, atom_names) in sorted(
            grouped.items(), key=lambda item: item[0]
        )
    ]


def specification_path(artifact_root: Path, target_id: str) -> Path:
    """Return the artifact-root path required by the RFdiffusion3 profile."""
    if not target_id or Path(target_id).name != target_id or target_id in {".", ".."}:
        raise SpecificationError(f"target_id is unsafe for an RFdiffusion3 filename: {target_id!r}")
    return artifact_root.expanduser().resolve() / "inputs" / f"{target_id}-rfd3.json"


def write_specification(path: Path, specification: Mapping[str, Any]) -> None:
    """Write one specification atomically as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(specification, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _target(config: Mapping[str, Any], target_id: str) -> Mapping[str, Any]:
    targets = config.get("targets")
    if not isinstance(targets, list):
        raise SpecificationError("resolved config has no targets list")
    matches = [
        target
        for target in targets
        if isinstance(target, Mapping) and target.get("target_id") == target_id
    ]
    if len(matches) != 1:
        raise SpecificationError(
            f"resolved config must contain exactly one target with target_id {target_id!r}"
        )
    return matches[0]


def _integer(mapping: Mapping[str, Any], field: str) -> int:
    value = mapping.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpecificationError(f"resolved config field {field!r} must be an integer")
    return value


def _campaign_key(config: Mapping[str, Any], target: Mapping[str, Any]) -> str:
    """Derive the specification key from the configured target input identity."""
    site = target.get("site")
    if isinstance(site, Mapping):
        for field in ("residue_map_source_path", "residue_map_path"):
            value = site.get(field)
            if isinstance(value, str) and value:
                parent = Path(value).parent.name
                if parent not in GENERIC_RESIDUE_PARENT_NAMES:
                    slug = SAFE_KEY_RE.sub("_", parent).strip("_").lower()
                    if slug:
                        return f"{slug}_binder"
    campaign_id = config.get("campaign_id")
    if isinstance(campaign_id, str) and campaign_id:
        slug = SAFE_KEY_RE.sub("_", campaign_id).strip("_").lower()
        if slug:
            return f"{slug}_binder"
    raise SpecificationError(
        "cannot derive an RFdiffusion3 specification key from the target input path or campaign_id"
    )


def _generation_hotspot_entries(site: Mapping[str, Any]) -> tuple[list[Any], str]:
    """Return the campaign field that constrains RFdiffusion3 placement.

    ``design_residues`` is the established generation constraint. A campaign
    that declares an explicit hotspot list makes that narrower user choice the
    generation constraint too. The other hotspot sources describe a completed
    design, so they cannot constrain a backbone that does not exist yet.
    """
    discovery = site.get("discovery")
    if site.get("epitope_constraint") == "unconstrained" or (
        isinstance(discovery, Mapping) and discovery.get("enabled") is True
    ):
        return [], "discovery"
    hotspot_source = site.get("hotspot_source")
    hotspot_residues = site.get("hotspot_residues")
    if hotspot_source in {None, "explicit"} and isinstance(hotspot_residues, list):
        if hotspot_residues:
            return hotspot_residues, "hotspot_residues"
    design_residues = site.get("design_residues")
    if not isinstance(design_residues, list) or not design_residues:
        raise SpecificationError("target site has no design_residues list")
    return design_residues, "design_residues"


def _generation_hotspot_records(
    target: Mapping[str, Any],
    target_chain: str,
    by_residue_id: Mapping[str, ResidueRecord],
) -> list[ResidueRecord]:
    site = target.get("site")
    if not isinstance(site, Mapping):
        raise SpecificationError("target has no site object")
    entries, field_name = _generation_hotspot_entries(site)
    selected: list[ResidueRecord] = []
    for value in entries:
        if not isinstance(value, str):
            raise SpecificationError(f"target site {field_name} must contain strings")
        match = RESIDUE_ID_RE.fullmatch(value)
        if match is None:
            raise SpecificationError(
                "RFdiffusion3 generation hotspots require single residues in "
                f"CHAIN:NUMBER form in targets[].site.{field_name}: {value!r}"
            )
        chain, number_text, insertion_code = match.groups()
        if chain != target_chain:
            raise SpecificationError(
                f"generation hotspot {value} is on {chain}, while binder.target_chain_id is {target_chain}"
            )
        residue_id = f"{chain}:{int(number_text)}{insertion_code}"
        record = by_residue_id.get(residue_id)
        if record is None:
            raise SpecificationError(
                f"generation hotspot {value} is absent from the target structure"
            )
        if record not in selected:
            selected.append(record)
    return selected


def _declared_optional_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read optional RFdiffusion3 choices only when the campaign declares them."""
    declared = config.get("rfd3_specification")
    if declared is None:
        return {}
    if not isinstance(declared, Mapping):
        raise SpecificationError("rfd3_specification must be an object when present")
    return {
        field: declared[field]
        for field in UNRESOLVED_OPTIONAL_FIELDS
        if field in declared
    }


def _side_chain_atom_depths(residue_name: str) -> dict[str, int]:
    """Return each side-chain atom's graph distance from CB."""
    bonds = SIDE_CHAIN_BONDS.get(residue_name)
    if bonds is None:
        return {}
    neighbors: dict[str, set[str]] = {}
    for first, second in bonds:
        neighbors.setdefault(first, set()).add(second)
        neighbors.setdefault(second, set()).add(first)
    distances = {"CB": 0}
    pending = ["CB"]
    while pending:
        atom = pending.pop(0)
        for neighbor in neighbors.get(atom, set()):
            if neighbor not in distances:
                distances[neighbor] = distances[atom] + 1
                pending.append(neighbor)
    return distances


def _fallback_tip_atoms(record: ResidueRecord) -> tuple[str, str] | None:
    """Return the two outermost present side-chain atoms after a tip is absent."""
    depths = _side_chain_atom_depths(record.residue_name.upper())
    present = sorted(
        (depth, atom)
        for atom, depth in depths.items()
        if atom != "CB" and atom in record.atom_names
    )
    if len(present) < 2:
        return None
    return tuple(atom for _, atom in present[-2:])


def _missing_tip_atom_error(
    record: ResidueRecord,
    missing_atoms: list[str],
) -> SpecificationError:
    """Describe a missing-tip refusal and its coordinate repair choices."""
    present_atoms = ", ".join(sorted(record.atom_names)) or "none"
    return SpecificationError(
        f"structure residue {record.residue_id} ({record.residue_name}) lacks side-chain atom(s): "
        + ", ".join(missing_atoms)
        + f"; present atoms: {present_atoms}. "
        "Fallback requires at least two modelled side-chain atoms beyond CB. "
        f"Drop {record.residue_id} from the generation hotspot while retaining it in the design site, "
        "or supply repaired coordinates."
    )


def derive_specification(
    config: Mapping[str, Any],
    target_id: str,
    residues: Iterable[ResidueRecord],
    *,
    design_target_chain: str | None = None,
) -> dict[str, Any]:
    """Derive the RFdiffusion3 JSON object for one resolved target."""
    target = _target(config, target_id)
    binder = config.get("binder")
    if not isinstance(binder, Mapping):
        raise SpecificationError("resolved config has no binder object")
    minimum_length = _integer(binder, "minimum_length")
    maximum_length = _integer(binder, "maximum_length")
    if minimum_length < 1 or maximum_length < minimum_length:
        raise SpecificationError("binder length bounds are invalid")
    target_chain = binder.get("target_chain_id")
    if not isinstance(target_chain, str) or not target_chain:
        raise SpecificationError("binder.target_chain_id is required")
    if design_target_chain is not None and design_target_chain != target_chain:
        raise SpecificationError(
            f"adapter design target chain {design_target_chain} disagrees with binder.target_chain_id {target_chain}"
        )

    records = list(residues)
    by_residue_id: dict[str, ResidueRecord] = {}
    for record in records:
        if record.residue_id in by_residue_id:
            raise SpecificationError(f"structure has duplicate residue {record.residue_id}")
        by_residue_id[record.residue_id] = record
    target_records = [record for record in records if record.chain_id == target_chain]
    if not target_records:
        raise SpecificationError(f"target structure has no residues on chain {target_chain}")
    if any(record.insertion_code for record in target_records):
        raise SpecificationError(
            f"target chain {target_chain} has insertion codes that the RFdiffusion3 contig cannot represent"
        )
    target_low = min(record.residue_number for record in target_records)
    target_high = max(record.residue_number for record in target_records)
    selected = _generation_hotspot_records(target, target_chain, by_residue_id)

    hotspots: dict[str, str] = {}
    substitutions: list[dict[str, Any]] = []
    for record in selected:
        residue_name = record.residue_name.upper()
        pair = SIDE_CHAIN_TIP_ATOMS.get(residue_name)
        if pair is None:
            if residue_name in NO_SIDE_CHAIN_TIP:
                continue
            raise SpecificationError(
                f"generation hotspot {record.residue_id} ({residue_name}) has no declared side-chain "
                "tip atom pair. Supported residue types: "
                + ", ".join(sorted(SIDE_CHAIN_TIP_ATOMS))
                + ". ALA and GLY are handled without a side-chain tip atom pair. "
                "Drop this residue from the generation hotspot while retaining it in the design site."
            )
        missing_atoms = [atom for atom in pair if atom not in record.atom_names]
        used_atoms = pair
        if missing_atoms:
            fallback = _fallback_tip_atoms(record)
            if fallback is None:
                raise _missing_tip_atom_error(record, missing_atoms)
            used_atoms = fallback
            substitutions.append(
                {
                    "residue": record.residue_id,
                    "residue_name": residue_name,
                    "declared_atoms": list(pair),
                    "used_atoms": list(used_atoms),
                    "reason": FALLBACK_REASON,
                }
            )
        hotspots[record.specification_id] = ",".join(used_atoms)

    entry: dict[str, Any] = {
        "contig": f"{minimum_length}-{maximum_length},/0,{target_chain}{target_low}-{target_high}",
        "select_hotspots": hotspots,
    }
    if substitutions:
        entry["extra"] = {"hotspot_tip_atom_substitutions": substitutions}
    entry.update(_declared_optional_fields(config))
    return {_campaign_key(config, target): entry}
