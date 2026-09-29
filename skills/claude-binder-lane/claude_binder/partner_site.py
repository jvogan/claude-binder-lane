"""Resolve a target site from a deposited complex with a named partner.

The resolver produces the existing ``reference_contact_residues`` list. It
does not select a residue surface from the target alone. A partner name has to
resolve to one verified complex, or the caller has to pin an RCSB PDB entry.

The published campaign's in-silico record defines an epitope as a target
residue with any heavy atom within 5 Angstrom of a binder heavy atom. See
``INSILICO.md`` section 8, "Epitopes", in the published campaign release. This
module applies that deposited rule to a natural binding partner.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Iterable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .adapters import binder_metrics


PARTNER_CONTACT_CUTOFF_ANGSTROM = 5.0
ATOM_SELECTION = "heavy-atoms"
# The receipt has to carry a citation its reader can check. The published record
# lives outside this package, so the string names the document and section and
# then states the rule itself rather than pointing at a path no reader holds.
PUBLISHED_CUTOFF_SOURCE = (
    "published campaign release, INSILICO.md section 8 'Epitopes': target "
    "residues with any heavy atom within 5 Angstrom of a binder heavy atom"
)
# How the target entity was established. Only the first value means a deposition
# record was read and the accession was matched against it. A caller assertion
# has to stay distinguishable, because the confidence basis names the check that
# ran and a receipt may not claim a check nobody performed.
TARGET_MATCHES = frozenset({"deposited-entity-accession", "caller-asserted"})

# How the partner entity was established, on the same rule.
PARTNER_MATCHES = frozenset(
    {
        "exact-entity-name",
        "sole-non-target-protein",
        "pinned-chain",
        "caller-asserted-chain",
    }
)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"


class SiteResolutionError(ValueError):
    """A partner name and structure cannot establish a safe target site."""


class ComplexNotFoundError(SiteResolutionError):
    """No deposited complex establishes the requested target-partner pair."""


class AmbiguousComplexError(SiteResolutionError):
    """Several deposited complexes fit the request and need a pinned entry."""


@dataclass(frozen=True)
class PartnerComplexSpec:
    """The information needed to identify one target-partner complex."""

    target_accession: str
    partner_name: str
    entry_id: str | None = None
    target_chain: str | None = None
    partner_chains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.target_accession.strip():
            raise SiteResolutionError("partner complex resolution needs a target accession")
        if not self.partner_name.strip():
            raise SiteResolutionError("partner complex resolution needs a partner name")
        if self.target_chain is not None and not self.target_chain.strip():
            raise SiteResolutionError("target_chain must be a non-empty chain identifier")
        if any(not chain.strip() for chain in self.partner_chains):
            raise SiteResolutionError("partner_chains cannot contain an empty chain identifier")
        if len(set(self.partner_chains)) != len(self.partner_chains):
            raise SiteResolutionError("partner_chains cannot name a chain more than once")


@dataclass(frozen=True)
class LocatedComplex:
    """One complex with coordinates and chain identities established by a source."""

    entry_id: str
    structure: binder_metrics.Structure
    coordinate_text: str | None
    target_chains: tuple[str, ...]
    partner_chains: tuple[str, ...]
    structure_status: str
    experimental_methods: tuple[str, ...]
    source: str
    source_url: str | None
    partner_entity_description: str | None = None
    # Both default to the caller assertion, so a locator that forgets to record
    # what it checked underclaims rather than fabricating a check.
    partner_match: str = "caller-asserted-chain"
    target_match: str = "caller-asserted"

    def __post_init__(self) -> None:
        if self.structure_status not in {"experimental", "predicted"}:
            raise SiteResolutionError(
                "structure_status must be 'experimental' or 'predicted'"
            )
        if self.target_match not in TARGET_MATCHES:
            raise SiteResolutionError(
                f"target_match must be one of {sorted(TARGET_MATCHES)}"
            )
        if self.partner_match not in PARTNER_MATCHES:
            raise SiteResolutionError(
                f"partner_match must be one of {sorted(PARTNER_MATCHES)}"
            )
        if not self.target_chains:
            raise SiteResolutionError("a located complex needs at least one target chain")
        if not self.partner_chains:
            raise SiteResolutionError("a located complex needs at least one partner chain")


@dataclass(frozen=True)
class PartnerSiteResolution:
    """The site list and provenance that the campaign stores as an artifact."""

    target_accession: str
    partner_name: str
    structure_id: str
    structure_source: str
    structure_source_url: str | None
    structure_status: str
    experimental_methods: tuple[str, ...]
    target_chain: str
    partner_chains: tuple[str, ...]
    atom_selection: str
    contact_cutoff_angstrom: float
    contact_residues: tuple[str, ...]
    confidence_level: str
    confidence_basis: tuple[str, ...]
    partner_entity_description: str | None

    def artifact(self) -> dict[str, Any]:
        """Return the serializable record embedded in the campaign artifact."""
        return {
            "schema_version": 1,
            "method": "partner-complex-heavy-atom-contacts",
            "target_accession": self.target_accession,
            "partner_name": self.partner_name,
            "structure_id": self.structure_id,
            "structure_source": self.structure_source,
            "structure_source_url": self.structure_source_url,
            "structure_status": self.structure_status,
            "experimental_methods": list(self.experimental_methods),
            "target_chain": self.target_chain,
            "partner_chains": list(self.partner_chains),
            "atom_selection": self.atom_selection,
            "contact_cutoff_angstrom": self.contact_cutoff_angstrom,
            "contact_cutoff_source": PUBLISHED_CUTOFF_SOURCE,
            "contact_residue_count": len(self.contact_residues),
            "reference_contact_residues": list(self.contact_residues),
            "confidence": {
                "level": self.confidence_level,
                "basis": list(self.confidence_basis),
            },
            "partner_entity_description": self.partner_entity_description,
        }


class ComplexLocator(Protocol):
    """Locate one complex that can support partner-site contact extraction."""

    def locate(self, specification: PartnerComplexSpec) -> LocatedComplex:
        """Return a complex, or raise a SiteResolutionError with an action."""


def _grid_cell(coordinate: tuple[float, float, float], cutoff: float) -> tuple[int, int, int]:
    return tuple(math.floor(value / cutoff) for value in coordinate)  # type: ignore[return-value]


def _partner_atom_grid(
    structure: binder_metrics.Structure,
    partner_chains: Iterable[str],
    cutoff: float,
) -> dict[tuple[int, int, int], list[tuple[float, float, float]]]:
    """Index partner heavy atoms in cutoff-sized cells.

    A full target-by-partner distance matrix can require gigabytes for a large
    deposited assembly. Each lookup checks only the 27 nearby cells.
    """
    grid: dict[tuple[int, int, int], list[tuple[float, float, float]]] = {}
    for chain in partner_chains:
        for residue in structure.chain_residues(chain):
            for coordinate in residue.heavy_atom_coords():
                point = (float(coordinate[0]), float(coordinate[1]), float(coordinate[2]))
                grid.setdefault(_grid_cell(point, cutoff), []).append(point)
    return grid


def _residue_contacts_grid(
    residue: binder_metrics.Residue,
    grid: Mapping[tuple[int, int, int], list[tuple[float, float, float]]],
    cutoff: float,
) -> bool:
    """Return whether one target residue has a partner heavy-atom contact."""
    cutoff_squared = cutoff * cutoff
    for coordinate in residue.heavy_atom_coords():
        point = (float(coordinate[0]), float(coordinate[1]), float(coordinate[2]))
        cell = _grid_cell(point, cutoff)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for partner in grid.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()):
                        distance_squared = (
                            (point[0] - partner[0]) ** 2
                            + (point[1] - partner[1]) ** 2
                            + (point[2] - partner[2]) ** 2
                        )
                        if distance_squared < cutoff_squared:
                            return True
    return False


def contacting_target_residues(
    structure: binder_metrics.Structure,
    *,
    target_chain: str,
    partner_chains: Iterable[str],
    cutoff_angstrom: float = PARTNER_CONTACT_CUTOFF_ANGSTROM,
) -> tuple[str, ...]:
    """Return target residues within the heavy-atom cutoff of partner chains."""
    if not isinstance(cutoff_angstrom, (int, float)) or isinstance(cutoff_angstrom, bool):
        raise SiteResolutionError("contact cutoff must be numeric")
    cutoff = float(cutoff_angstrom)
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise SiteResolutionError("contact cutoff must be a finite positive number")
    partner_ids = tuple(partner_chains)
    if not partner_ids:
        raise SiteResolutionError("partner contact resolution needs at least one partner chain")
    grid = _partner_atom_grid(structure, partner_ids, cutoff)
    if not grid:
        raise SiteResolutionError("the partner chains have no heavy atoms")

    contacts: list[str] = []
    for residue in structure.chain_residues(target_chain):
        # `Residue.label` carries the insertion code, and every residue pattern
        # in the tree accepts one, so 52 and 52A stay distinct contacts. This
        # refused them while claiming the campaign format could not represent
        # one, which stopped being true when labels began carrying the code.
        if _residue_contacts_grid(residue, grid, cutoff):
            contacts.append(residue.label)
    return tuple(contacts)


def _select_target_chain(
    located: LocatedComplex,
    specification: PartnerComplexSpec,
    cutoff: float,
) -> tuple[str, tuple[str, ...]]:
    """Choose a declared target chain only when the contact geometry is unique."""
    candidates = (
        (specification.target_chain,)
        if specification.target_chain is not None
        else located.target_chains
    )
    unknown = sorted(set(candidates) - set(located.target_chains))
    if unknown:
        raise SiteResolutionError(
            f"target chain {unknown} is absent from the target entity of {located.entry_id}. "
            f"Available target chains: {list(located.target_chains)}"
        )
    partner_chains = (
        specification.partner_chains
        if specification.partner_chains
        else located.partner_chains
    )
    unknown_partners = sorted(set(partner_chains) - set(located.partner_chains))
    if unknown_partners:
        raise SiteResolutionError(
            f"partner chain {unknown_partners} is absent from the partner entity of "
            f"{located.entry_id}. Available partner chains: {list(located.partner_chains)}"
        )
    contacted = {
        chain: contacting_target_residues(
            located.structure,
            target_chain=chain,
            partner_chains=partner_chains,
            cutoff_angstrom=cutoff,
        )
        for chain in candidates
    }
    nonempty = [(chain, residues) for chain, residues in contacted.items() if residues]
    if not nonempty:
        raise SiteResolutionError(
            f"{located.entry_id} has no heavy-atom contact at {cutoff:g} Å between "
            f"target chain(s) {list(candidates)} and partner chain(s) {list(partner_chains)}. "
            "Choose a complex that contains the intended interface."
        )
    if len(nonempty) != 1:
        chains = [chain for chain, _ in nonempty]
        raise SiteResolutionError(
            f"{located.entry_id} has contacting target chains {chains}. Supply "
            "target_chain to establish which copy belongs in the campaign."
        )
    return nonempty[0]


def _confidence(located: LocatedComplex) -> tuple[str, tuple[str, ...]]:
    """Return a categorical confidence statement naming only the checks that ran.

    Every line is a claim in a durable receipt, so each one is conditional on the
    step that established it. The deposition record is read only in
    ``RcsbComplexLocator``. A complex the caller supplied has had its accession,
    its partner chain, and its coordinate provenance asserted rather than
    checked, and the receipt says so instead of borrowing the RCSB wording.
    """
    basis: list[str] = []
    deposited = located.target_match == "deposited-entity-accession"
    if deposited:
        basis.append("target accession exactly matches the deposited polymer entity")
    else:
        basis.append(
            "the caller named the target accession, and no deposition record was "
            "read to confirm it identifies this chain"
        )
    if located.partner_match == "exact-entity-name":
        basis.append("partner name matches the deposited partner entity description")
    elif located.partner_match == "pinned-chain":
        basis.append("the caller pinned the partner chain and it belongs to a non-target protein entity")
    elif located.partner_match == "sole-non-target-protein":
        basis.append("the partner name matched no entity description, and the entry has one non-target protein entity")
    else:
        basis.append(
            "the caller named the partner chain, and no deposition record was "
            "read to confirm it is a non-target protein entity"
        )
    if located.structure_status != "experimental":
        basis.append("the coordinates are predicted or computed rather than experimentally determined")
        return "medium", tuple(basis)
    if deposited:
        basis.append("the deposited complex reports an experimental method")
        return "high", tuple(basis)
    # The caller declared the provenance. An assertion cannot reach the level a
    # read deposition record reaches.
    basis.append(
        "the caller declared the coordinates experimental, and no deposition "
        "record was read to confirm it"
    )
    return "medium", tuple(basis)


def resolve_partner_site(
    specification: PartnerComplexSpec,
    locator: ComplexLocator,
    *,
    cutoff_angstrom: float = PARTNER_CONTACT_CUTOFF_ANGSTROM,
) -> PartnerSiteResolution:
    """Resolve a partner complex into the existing reference-contact residue list."""
    located = locator.locate(specification)
    target_chain, contacts = _select_target_chain(located, specification, cutoff_angstrom)
    confidence_level, confidence_basis = _confidence(located)
    partner_chains = (
        specification.partner_chains
        if specification.partner_chains
        else located.partner_chains
    )
    return PartnerSiteResolution(
        target_accession=specification.target_accession,
        partner_name=specification.partner_name,
        structure_id=located.entry_id,
        structure_source=located.source,
        structure_source_url=located.source_url,
        structure_status=located.structure_status,
        experimental_methods=located.experimental_methods,
        target_chain=target_chain,
        partner_chains=partner_chains,
        atom_selection=ATOM_SELECTION,
        contact_cutoff_angstrom=float(cutoff_angstrom),
        contact_residues=contacts,
        confidence_level=confidence_level,
        confidence_basis=confidence_basis,
        partner_entity_description=located.partner_entity_description,
    )


@dataclass(frozen=True)
class _PolymerEntity:
    """The small subset of polymer-entity metadata used for chain selection."""

    entity_id: str
    chains: tuple[str, ...]
    accessions: tuple[str, ...]
    description: str
    is_protein: bool


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _is_partner_description(partner_name: str, description: str) -> bool:
    query = _normalized_text(partner_name)
    candidate = _normalized_text(description)
    return bool(query and candidate and (query in candidate or candidate in query))


class RcsbComplexLocator:
    """Resolve one public RCSB PDB complex without credentials or paid calls."""

    def __init__(self, *, timeout_seconds: float = 20.0, max_hits: int = 25):
        self.timeout_seconds = timeout_seconds
        self.max_hits = max_hits

    def _json(self, url: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "claude-binder-partner-site/1",
            },
            method="POST" if data is not None else "GET",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SiteResolutionError(
                f"RCSB PDB did not return usable data for {url}: {exc}. "
                "Retry later or supply a verified local complex."
            ) from exc

    def _coordinate_text(self, entry_id: str) -> str:
        url = f"{RCSB_DOWNLOAD_URL}/{entry_id}.cif"
        request = Request(url, headers={"User-Agent": "claude-binder-partner-site/1"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, UnicodeDecodeError) as exc:
            raise SiteResolutionError(
                f"RCSB PDB did not return coordinates for {entry_id}: {exc}. "
                "Supply a verified local complex if the entry remains unavailable."
            ) from exc

    def _search_entries(self, specification: PartnerComplexSpec) -> list[str]:
        if specification.entry_id is not None:
            return [specification.entry_id.upper()]
        payload = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": (
                                "rcsb_polymer_entity_container_identifiers."
                                "reference_sequence_identifiers.database_accession"
                            ),
                            "operator": "exact_match",
                            "value": specification.target_accession,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "full_text",
                        "parameters": {"value": f'"{specification.partner_name}"'},
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": self.max_hits}},
        }
        response = self._json(RCSB_SEARCH_URL, payload)
        return [
            str(hit["identifier"]).upper()
            for hit in response.get("result_set", [])
            if isinstance(hit, Mapping) and isinstance(hit.get("identifier"), str)
        ]

    def _entity(self, entry_id: str, entity_id: str) -> _PolymerEntity:
        payload = self._json(f"{RCSB_DATA_URL}/polymer_entity/{entry_id}/{entity_id}")
        identifiers = payload.get("rcsb_polymer_entity_container_identifiers", {})
        reference_ids = identifiers.get("reference_sequence_identifiers", [])
        accessions = tuple(
            str(item.get("database_accession"))
            for item in reference_ids
            if isinstance(item, Mapping) and item.get("database_accession")
        )
        chains = tuple(
            str(chain)
            for chain in identifiers.get("auth_asym_ids", [])
            if isinstance(chain, str) and chain
        )
        if not chains:
            chains = tuple(
                str(chain)
                for chain in identifiers.get("asym_ids", [])
                if isinstance(chain, str) and chain
            )
        polymer = payload.get("entity_poly", {})
        polymer_type = str(polymer.get("type", "")).lower()
        description = str(
            payload.get("rcsb_polymer_entity", {}).get("pdbx_description", "")
        )
        return _PolymerEntity(
            entity_id=entity_id,
            chains=chains,
            accessions=accessions,
            description=description,
            is_protein=polymer_type.startswith("polypeptide"),
        )

    def _entry_candidate(
        self, entry_id: str, specification: PartnerComplexSpec
    ) -> LocatedComplex | None:
        payload = self._json(f"{RCSB_DATA_URL}/entry/{entry_id}")
        identifiers = payload.get("rcsb_entry_container_identifiers", {})
        entity_ids = identifiers.get("polymer_entity_ids", [])
        if not isinstance(entity_ids, list):
            return None
        entities = [self._entity(entry_id, str(entity_id)) for entity_id in entity_ids]
        accession = specification.target_accession.upper()
        target_entities = [
            entity
            for entity in entities
            if accession in {candidate.upper() for candidate in entity.accessions}
        ]
        if len(target_entities) != 1:
            return None
        target = target_entities[0]
        other_proteins = [
            entity
            for entity in entities
            if entity.entity_id != target.entity_id and entity.is_protein
        ]
        requested_chains = set(specification.partner_chains)
        if requested_chains:
            possible_chains = {chain for entity in other_proteins for chain in entity.chains}
            if requested_chains - possible_chains:
                return None
            partners = [
                entity
                for entity in other_proteins
                if requested_chains.intersection(entity.chains)
            ]
            partner_match = "pinned-chain"
        else:
            named = [
                entity
                for entity in other_proteins
                if _is_partner_description(specification.partner_name, entity.description)
            ]
            if len(named) == 1:
                partners = named
                partner_match = "exact-entity-name"
            elif not named and len(other_proteins) == 1:
                partners = other_proteins
                partner_match = "sole-non-target-protein"
            else:
                return None
        partner_chains = tuple(chain for entity in partners for chain in entity.chains)
        exptl = payload.get("exptl", [])
        methods = tuple(
            str(item.get("method"))
            for item in exptl
            if isinstance(item, Mapping) and item.get("method")
        )
        coordinate_text = self._coordinate_text(entry_id)
        try:
            structure = binder_metrics.parse_structure_atoms(
                coordinate_text, argument=f"RCSB PDB entry {entry_id}"
            )
        except binder_metrics.MetricInputError as exc:
            raise SiteResolutionError(
                f"RCSB PDB entry {entry_id} has no usable coordinate model: {exc}"
            ) from exc
        return LocatedComplex(
            entry_id=entry_id,
            structure=structure,
            coordinate_text=coordinate_text,
            target_chains=target.chains,
            partner_chains=partner_chains,
            structure_status="experimental" if methods else "predicted",
            experimental_methods=methods,
            source="RCSB PDB",
            source_url=f"https://www.rcsb.org/structure/{entry_id}",
            partner_entity_description=partners[0].description if len(partners) == 1 else None,
            partner_match=partner_match,
            # Reached only after the accession matched exactly one polymer
            # entity of this entry, a few lines above.
            target_match="deposited-entity-accession",
        )

    def locate(self, specification: PartnerComplexSpec) -> LocatedComplex:
        """Resolve the requested pair and refuse an unpinned ambiguous result."""
        entry_ids = self._search_entries(specification)
        if not entry_ids:
            raise ComplexNotFoundError(
                f"RCSB PDB has no complex for target accession "
                f"{specification.target_accession!r} and partner {specification.partner_name!r}. "
                "Supply a verified local complex with its target and partner chains."
            )
        candidates = [
            candidate
            for entry_id in entry_ids
            if (candidate := self._entry_candidate(entry_id, specification)) is not None
        ]
        if not candidates:
            raise ComplexNotFoundError(
                f"RCSB PDB found no complex where target accession "
                f"{specification.target_accession!r} and partner "
                f"{specification.partner_name!r} identify distinct protein entities. "
                "Supply a verified local complex with its target and partner chains."
            )
        if specification.entry_id is None and len(candidates) != 1:
            entries = ", ".join(candidate.entry_id for candidate in candidates[:10])
            raise AmbiguousComplexError(
                f"RCSB PDB found {len(candidates)} complexes for target accession "
                f"{specification.target_accession!r} and partner "
                f"{specification.partner_name!r}: {entries}. Supply --pdb-entry to "
                "pin the biological state before deriving a site."
            )
        return candidates[0]


def local_complex(
    *,
    structure: binder_metrics.Structure,
    structure_id: str,
    target_chain: str,
    partner_chains: Iterable[str],
    structure_status: str,
    source: str,
    source_url: str | None = None,
    coordinate_text: str | None = None,
) -> LocatedComplex:
    """Wrap a verified local complex for the same strict contact resolver.

    Nothing here reads a deposition record. The caller names the accession, the
    chains, and the coordinate provenance, so the match fields record assertions
    and `experimental_methods` stays empty rather than carrying a placeholder
    where a reported method belongs.
    """
    return LocatedComplex(
        entry_id=structure_id,
        structure=structure,
        coordinate_text=coordinate_text,
        target_chains=(target_chain,),
        partner_chains=tuple(partner_chains),
        structure_status=structure_status,
        experimental_methods=(),
        source=source,
        source_url=source_url,
        partner_match="caller-asserted-chain",
        target_match="caller-asserted",
    )
