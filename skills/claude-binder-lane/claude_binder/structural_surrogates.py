"""Numpy-only structural screens used when published tools are unavailable.

The functions in this module calculate explicit surrogates.  They never report a
TM-score, a Foldseek cluster, or a DSSP assignment.  A caller must record the
surrogate labels and calibration state beside every derived value.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import shlex
from typing import Iterable, Mapping, Sequence

import numpy as np


CONTACT_CUTOFF_ANGSTROM = 8.0
MINIMUM_CONTACT_SEQUENCE_SEPARATION = 6
REGISTER_SHIFTS = tuple(range(-2, 3))
HYDROGEN_BOND_ENERGY_CUTOFF_KCAL_PER_MOL = -0.5

CONTACT_MAP_OVERLAP_LABEL = "in-house TM-align surrogate: contact-map overlap F1"
SECONDARY_STRUCTURE_LABEL = "in-house DSSP surrogate: Kabsch-Sander backbone assignment"
CLUSTERING_LABEL = "in-house Foldseek surrogate: contact-map-overlap derived clustering"
SURROGATE_DISCLOSURE = (
    "Novelty, secondary-structure, and diversity results in this section were computed with "
    "in-house surrogates for TM-align, Foldseek, and DSSP (contact-map overlap, a "
    "Kabsch-Sander backbone assignment, and derived clustering), calibrated on a limited "
    "in-house anchor set, and are provisional until reproduced with the published tools."
)

_LOGGER = logging.getLogger(__name__)

_AMINO_ACID_3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


class StructuralSurrogateError(ValueError):
    """A supplied structure or calibration parameter cannot support a surrogate."""


@dataclass(frozen=True)
class AlignmentParameters:
    """Explicit Smith-Waterman scoring supplied by campaign configuration."""

    match_score: float
    mismatch_score: float
    gap_penalty: float

    def __post_init__(self) -> None:
        for name, value in (
            ("match_score", self.match_score),
            ("mismatch_score", self.mismatch_score),
            ("gap_penalty", self.gap_penalty),
        ):
            if not math.isfinite(float(value)):
                raise StructuralSurrogateError(f"alignment {name} must be finite")
        # The recurrence adds all three, so their signs are part of the
        # contract. A config reading "penalty" as a magnitude and writing
        # gap_penalty: 1 made the upward and leftward terms grow without
        # bound, so the matrix filled monotonically, the best score landed in
        # the far corner, and the traceback returned an all-gap correspondence
        # that contact_map_overlap then scored as though it meant something.
        if float(self.match_score) <= 0.0:
            raise StructuralSurrogateError("alignment match_score must be positive")
        if float(self.mismatch_score) > 0.0:
            raise StructuralSurrogateError("alignment mismatch_score must not be positive")
        if float(self.gap_penalty) >= 0.0:
            raise StructuralSurrogateError("alignment gap_penalty must be negative")


def alignment_parameters_from_mapping(
    value: object, label: str = "alignment"
) -> AlignmentParameters:
    """Parse explicit local-alignment scores without supplying undocumented defaults."""
    if not isinstance(value, Mapping):
        raise StructuralSurrogateError(f"{label} must be an object")
    values: dict[str, float] = {}
    for field in ("match_score", "mismatch_score", "gap_penalty"):
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise StructuralSurrogateError(f"{label}.{field} must be a finite number")
        values[field] = float(raw)
    return AlignmentParameters(**values)


@dataclass
class BackboneResidue:
    """One polymer residue with the backbone atoms the surrogates consume."""

    chain_id: str
    residue_id: str
    residue_name: str
    sequence_code: str
    atoms: dict[str, np.ndarray]


@dataclass(frozen=True)
class ContactMapOverlap:
    """One contact-map F1 score under the winning register shift."""

    f1: float
    shared_contact_count: int
    design_contact_count: int
    reference_contact_count: int
    aligned_pair_count: int
    register_shift: int
    empty_map: bool


@dataclass(frozen=True)
class SecondaryStructureAssignment:
    """Three-class backbone assignment with the evidence counts used to derive it."""

    labels: tuple[str, ...]
    helix_fraction: float
    strand_fraction: float
    hydrogen_bond_count: int
    mini_ladder_count: int


@dataclass(frozen=True)
class StructuralClusters:
    """Complete-linkage clusters with the contact-map F1 scores that produced them."""

    identifiers: Mapping[str, str]
    pairwise_f1: Mapping[tuple[str, str], float]
    cutoff: float


def _as_coordinate(value: Sequence[float] | np.ndarray, label: str) -> np.ndarray:
    coordinate = np.asarray(value, dtype=float)
    if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
        raise StructuralSurrogateError(f"{label} must be one finite three-dimensional coordinate")
    return coordinate


def _residue_sort_key(residue_id: str) -> tuple[int, str]:
    digits = ""
    suffix = ""
    for character in residue_id:
        if character in "-0123456789" and not suffix:
            digits += character
        else:
            suffix += character
    try:
        return int(digits), suffix
    except ValueError:
        return 0, residue_id


def _add_atom(
    residues: dict[tuple[str, str], BackboneResidue],
    order: list[tuple[str, str]],
    *,
    chain_id: str,
    residue_id: str,
    residue_name: str,
    atom_name: str,
    coordinate: Sequence[float],
    altloc_priority: int,
    altlocs: dict[tuple[str, str, str], int],
) -> None:
    if atom_name not in _BACKBONE_ATOMS:
        return
    key = (chain_id, residue_id)
    if key not in residues:
        residues[key] = BackboneResidue(
            chain_id=chain_id,
            residue_id=residue_id,
            residue_name=residue_name,
            sequence_code=_AMINO_ACID_3_TO_1.get(residue_name.upper(), "X"),
            atoms={},
        )
        order.append(key)
    atom_key = (chain_id, residue_id, atom_name)
    if altloc_priority > altlocs.get(atom_key, 99):
        return
    residues[key].atoms[atom_name] = _as_coordinate(coordinate, f"{chain_id}:{residue_id} {atom_name}")
    altlocs[atom_key] = altloc_priority


def parse_pdb_backbones(path: Path) -> dict[str, tuple[BackboneResidue, ...]]:
    """Read first-model backbone atoms from a PDB file without external parsers."""
    residues: dict[tuple[str, str], BackboneResidue] = {}
    order: list[tuple[str, str]] = []
    altlocs: dict[tuple[str, str, str], int] = {}
    in_first_model = True
    saw_model = False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StructuralSurrogateError(f"cannot read PDB backbone file: {path}") from exc
    for line in lines:
        record = line[:6]
        if record == "MODEL ":
            if saw_model:
                in_first_model = False
            saw_model = True
            continue
        if record == "ENDMDL" and saw_model:
            in_first_model = False
            continue
        if not in_first_model or record != "ATOM  ":
            continue
        atom_name = line[12:16].strip().upper()
        if atom_name not in _BACKBONE_ATOMS:
            continue
        residue_name = line[17:20].strip().upper()
        chain_id = line[21:22].strip() or "_"
        residue_number = line[22:26].strip()
        insertion_code = line[26:27].strip()
        if not residue_number:
            raise StructuralSurrogateError(f"PDB atom has no residue number: {path}")
        residue_id = f"{residue_number}{insertion_code}"
        altloc = line[16:17].strip()
        altloc_priority = 0 if not altloc else 1 if altloc in {"A", "1"} else 2
        try:
            coordinate = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError as exc:
            raise StructuralSurrogateError(f"PDB atom has invalid coordinates: {path}") from exc
        _add_atom(
            residues,
            order,
            chain_id=chain_id,
            residue_id=residue_id,
            residue_name=residue_name,
            atom_name=atom_name,
            coordinate=coordinate,
            altloc_priority=altloc_priority,
            altlocs=altlocs,
        )
    return _group_backbones(residues, order, path)


def parse_mmcif_backbones(path: Path) -> dict[str, tuple[BackboneResidue, ...]]:
    """Read first-model backbone atoms from an atom-site mmCIF loop."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StructuralSurrogateError(f"cannot read mmCIF backbone file: {path}") from exc
    residues: dict[tuple[str, str], BackboneResidue] = {}
    order: list[tuple[str, str]] = []
    altlocs: dict[tuple[str, str, str], int] = {}
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        columns: list[str] = []
        while index < len(lines) and lines[index].strip().startswith("_"):
            columns.append(lines[index].strip())
            index += 1
        if not columns or not columns[0].startswith("_atom_site."):
            continue
        positions = {column: position for position, column in enumerate(columns)}
        required = {
            "_atom_site.group_PDB",
            "_atom_site.label_atom_id",
            "_atom_site.label_comp_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
        }
        if not required.issubset(positions):
            raise StructuralSurrogateError(f"mmCIF atom_site loop lacks backbone columns: {path}")
        chain_column = (
            "_atom_site.auth_asym_id"
            if "_atom_site.auth_asym_id" in positions
            else "_atom_site.label_asym_id"
        )
        sequence_column = (
            "_atom_site.auth_seq_id"
            if "_atom_site.auth_seq_id" in positions
            else "_atom_site.label_seq_id"
        )
        if chain_column not in positions or sequence_column not in positions:
            raise StructuralSurrogateError(f"mmCIF atom_site loop lacks chain or residue IDs: {path}")
        insertion_column = "_atom_site.pdbx_PDB_ins_code"
        altloc_column = "_atom_site.label_alt_id"
        model_column = "_atom_site.pdbx_PDB_model_num"
        while index < len(lines):
            text = lines[index].strip()
            # See novelty_filter: `END` and `ENDMDL` terminate an atom_site loop in
            # mmCIF written by a PDB-oriented pipeline, and the shipped
            # supplied-candidate structures carry one.
            if (
                not text
                or text == "loop_"
                or text.startswith("_")
                or text.startswith("#")
                or text in {"END", "ENDMDL"}
            ):
                break
            fields = shlex.split(text)
            index += 1
            if len(fields) != len(columns):
                raise StructuralSurrogateError(
                    f"mmCIF atom_site row has {len(fields)} fields; expected {len(columns)}: {path}"
                )
            if fields[positions["_atom_site.group_PDB"]] != "ATOM":
                continue
            if model_column in positions and fields[positions[model_column]] not in {"1", ".", "?"}:
                continue
            atom_name = fields[positions["_atom_site.label_atom_id"]].upper()
            if atom_name not in _BACKBONE_ATOMS:
                continue
            chain_id = fields[positions[chain_column]]
            if chain_id in {".", "?"}:
                chain_id = "_"
            sequence_id = fields[positions[sequence_column]]
            if sequence_id in {".", "?"}:
                raise StructuralSurrogateError(f"mmCIF atom has no residue number: {path}")
            insertion = fields[positions[insertion_column]] if insertion_column in positions else ""
            insertion = "" if insertion in {".", "?"} else insertion
            altloc = fields[positions[altloc_column]] if altloc_column in positions else ""
            altloc = "" if altloc in {".", "?"} else altloc
            altloc_priority = 0 if not altloc else 1 if altloc in {"A", "1"} else 2
            _add_atom(
                residues,
                order,
                chain_id=chain_id,
                residue_id=f"{sequence_id}{insertion}",
                residue_name=fields[positions["_atom_site.label_comp_id"]].upper(),
                atom_name=atom_name,
                coordinate=(
                    float(fields[positions["_atom_site.Cartn_x"]]),
                    float(fields[positions["_atom_site.Cartn_y"]]),
                    float(fields[positions["_atom_site.Cartn_z"]]),
                ),
                altloc_priority=altloc_priority,
                altlocs=altlocs,
            )
        return _group_backbones(residues, order, path)
    raise StructuralSurrogateError(f"mmCIF file has no atom_site loop: {path}")


def _group_backbones(
    residues: Mapping[tuple[str, str], BackboneResidue],
    order: Sequence[tuple[str, str]],
    path: Path,
) -> dict[str, tuple[BackboneResidue, ...]]:
    chains: dict[str, list[BackboneResidue]] = {}
    for key in order:
        residue = residues[key]
        chains.setdefault(residue.chain_id, []).append(residue)
    chains = {
        chain_id: sorted(values, key=lambda residue: _residue_sort_key(residue.residue_id))
        for chain_id, values in chains.items()
    }
    if not chains:
        raise StructuralSurrogateError(f"backbone file contains no protein backbone atoms: {path}")
    return {chain_id: tuple(values) for chain_id, values in chains.items()}


def parse_backbones(path: Path) -> dict[str, tuple[BackboneResidue, ...]]:
    """Read PDB or mmCIF backbone atoms selected by the file content."""
    try:
        prefix = path.read_text(encoding="utf-8", errors="strict")[:65536]
    except OSError as exc:
        raise StructuralSurrogateError(f"cannot read backbone file: {path}") from exc
    if "_atom_site." in prefix:
        return parse_mmcif_backbones(path)
    return parse_pdb_backbones(path)


def residue_sequence(residues: Sequence[BackboneResidue]) -> str:
    """Return the one-letter sequence recovered from parsed polymer residues."""
    sequence = "".join(residue.sequence_code for residue in residues)
    if not sequence or "X" in sequence:
        raise StructuralSurrogateError("backbone sequence contains no complete standard amino-acid sequence")
    return sequence


def choose_design_chain(
    chains: Mapping[str, Sequence[BackboneResidue]], candidate_sequence: str
) -> tuple[str, tuple[BackboneResidue, ...]]:
    """Choose the unique backbone chain whose parsed sequence equals the candidate sequence."""
    matches = [
        (chain_id, tuple(residues))
        for chain_id, residues in chains.items()
        if residue_sequence(residues) == candidate_sequence
    ]
    if len(matches) != 1:
        names = ", ".join(sorted(chains))
        raise StructuralSurrogateError(
            f"design backbone needs one chain matching the candidate sequence; matched {len(matches)} of [{names}]"
        )
    return matches[0]


def smith_waterman_alignment(
    design_sequence: str,
    reference_sequence: str,
    parameters: AlignmentParameters,
) -> tuple[tuple[int, int], ...]:
    """Return the highest-scoring local identity alignment as zero-based index pairs."""
    if not design_sequence or not reference_sequence:
        raise StructuralSurrogateError("Smith-Waterman alignment requires two non-empty sequences")
    scores = np.zeros((len(design_sequence) + 1, len(reference_sequence) + 1), dtype=float)
    best_score = 0.0
    best_position = (0, 0)
    for design_index, design_residue in enumerate(design_sequence, start=1):
        for reference_index, reference_residue in enumerate(reference_sequence, start=1):
            diagonal = scores[design_index - 1, reference_index - 1] + (
                parameters.match_score if design_residue == reference_residue else parameters.mismatch_score
            )
            upward = scores[design_index - 1, reference_index] + parameters.gap_penalty
            leftward = scores[design_index, reference_index - 1] + parameters.gap_penalty
            value = max(0.0, diagonal, upward, leftward)
            scores[design_index, reference_index] = value
            if value > best_score:
                best_score = value
                best_position = (design_index, reference_index)
    if best_score <= 0:
        return ()
    design_index, reference_index = best_position
    pairs: list[tuple[int, int]] = []
    while scores[design_index, reference_index] > 0:
        current = scores[design_index, reference_index]
        diagonal = scores[design_index - 1, reference_index - 1] + (
            parameters.match_score
            if design_sequence[design_index - 1] == reference_sequence[reference_index - 1]
            else parameters.mismatch_score
        )
        if np.isclose(current, diagonal):
            pairs.append((design_index - 1, reference_index - 1))
            design_index -= 1
            reference_index -= 1
            continue
        if np.isclose(current, scores[design_index - 1, reference_index] + parameters.gap_penalty):
            design_index -= 1
            continue
        if np.isclose(current, scores[design_index, reference_index - 1] + parameters.gap_penalty):
            reference_index -= 1
            continue
        raise StructuralSurrogateError("Smith-Waterman traceback cannot identify its predecessor")
    return tuple(reversed(pairs))


def ca_contact_map(residues: Sequence[BackboneResidue]) -> frozenset[tuple[int, int]]:
    """Return long-range C-alpha contacts at the reviewer-specified 8 Angstrom cutoff."""
    if not residues:
        raise StructuralSurrogateError("contact-map overlap requires at least one residue")
    missing = [residue.residue_id for residue in residues if "CA" not in residue.atoms]
    if missing:
        raise StructuralSurrogateError(f"contact-map overlap needs CA atoms; missing at {', '.join(missing)}")
    coordinates = np.stack([residue.atoms["CA"] for residue in residues])
    offsets = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distances = np.sqrt(np.sum(offsets * offsets, axis=2))
    eligible = np.triu(np.ones(distances.shape, dtype=bool), k=MINIMUM_CONTACT_SEQUENCE_SEPARATION)
    contacts = np.argwhere(eligible & (distances <= CONTACT_CUTOFF_ANGSTROM))
    return frozenset((int(first), int(second)) for first, second in contacts)


def _shift_alignment_pairs(
    alignment: Sequence[tuple[int, int]], register_shift: int
) -> dict[int, int]:
    """Shift the reference side through ordered aligned pairs, preserving gap handling."""
    mapping: dict[int, int] = {}
    for index, (design_index, _) in enumerate(alignment):
        reference_pair_index = index + register_shift
        if 0 <= reference_pair_index < len(alignment):
            mapping[design_index] = alignment[reference_pair_index][1]
    return mapping


def contact_map_overlap(
    design_residues: Sequence[BackboneResidue],
    reference_residues: Sequence[BackboneResidue],
    parameters: AlignmentParameters,
) -> ContactMapOverlap:
    """Measure the best F1 overlap of two contact maps under local sequence alignment."""
    alignment = smith_waterman_alignment(
        residue_sequence(design_residues), residue_sequence(reference_residues), parameters
    )
    design_contacts = ca_contact_map(design_residues)
    reference_contacts = ca_contact_map(reference_residues)
    candidates: list[ContactMapOverlap] = []
    for shift in REGISTER_SHIFTS:
        mapping = _shift_alignment_pairs(alignment, shift)
        mapped_design_contacts = {
            tuple(sorted((mapping[first], mapping[second])))
            for first, second in design_contacts
            if first in mapping and second in mapping
        }
        mapped_reference_indices = set(mapping.values())
        mapped_reference_contacts = {
            contact
            for contact in reference_contacts
            if contact[0] in mapped_reference_indices and contact[1] in mapped_reference_indices
        }
        shared = mapped_design_contacts & mapped_reference_contacts
        denominator = len(mapped_design_contacts) + len(mapped_reference_contacts)
        empty_map = denominator == 0
        if empty_map:
            _LOGGER.warning(
                "contact-map overlap received no comparable mapped contacts; returning F1=0"
            )
        candidates.append(
            ContactMapOverlap(
                f1=0.0 if empty_map else 2.0 * len(shared) / denominator,
                shared_contact_count=len(shared),
                design_contact_count=len(mapped_design_contacts),
                reference_contact_count=len(mapped_reference_contacts),
                aligned_pair_count=len(mapping),
                register_shift=shift,
                empty_map=empty_map,
            )
        )
    if not candidates:
        raise StructuralSurrogateError("contact-map overlap found no register shifts")
    return max(candidates, key=lambda value: (value.f1, -abs(value.register_shift), -value.register_shift))


def hydrogen_bond_energy(
    oxygen: Sequence[float],
    carbon: Sequence[float],
    hydrogen: Sequence[float],
    nitrogen: Sequence[float],
) -> float:
    """Calculate the Kabsch-Sander electrostatic backbone hydrogen-bond energy."""
    oxygen = _as_coordinate(oxygen, "oxygen")
    carbon = _as_coordinate(carbon, "carbon")
    hydrogen = _as_coordinate(hydrogen, "hydrogen")
    nitrogen = _as_coordinate(nitrogen, "nitrogen")
    distances = (
        np.linalg.norm(oxygen - nitrogen),
        np.linalg.norm(carbon - hydrogen),
        np.linalg.norm(oxygen - hydrogen),
        np.linalg.norm(carbon - nitrogen),
    )
    if any(distance == 0 for distance in distances):
        raise StructuralSurrogateError("Kabsch-Sander hydrogen-bond energy has coincident atoms")
    reciprocal_sum = 1 / distances[0] + 1 / distances[1] - 1 / distances[2] - 1 / distances[3]
    return float(0.084 * reciprocal_sum * 332)


def kabsch_sander_hydrogen_bonds(
    residues: Sequence[BackboneResidue],
) -> frozenset[tuple[int, int]]:
    """Return acceptor-to-donor backbone hydrogen bonds for one polymer chain."""
    if not residues:
        raise StructuralSurrogateError("secondary-structure assignment requires at least one residue")
    missing: list[str] = []
    for residue in residues:
        absent = sorted(_BACKBONE_ATOMS - set(residue.atoms))
        if absent:
            missing.append(f"{residue.residue_id} ({', '.join(absent)})")
    if missing:
        raise StructuralSurrogateError(
            "Kabsch-Sander assignment requires N, CA, C, and O atoms; missing " + "; ".join(missing)
        )
    bonds: set[tuple[int, int]] = set()
    for donor_index, donor in enumerate(residues):
        if donor_index == 0 or donor.residue_name.upper() == "PRO":
            continue
        previous = residues[donor_index - 1]
        direction = previous.atoms["O"] - previous.atoms["C"]
        length = float(np.linalg.norm(direction))
        if length == 0:
            raise StructuralSurrogateError(
                f"cannot place amide hydrogen for residue {donor.residue_id}: preceding C and O coincide"
            )
        hydrogen = donor.atoms["N"] + direction / length
        for acceptor_index, acceptor in enumerate(residues):
            energy = hydrogen_bond_energy(
                acceptor.atoms["O"], previous.atoms["C"], hydrogen, donor.atoms["N"]
            )
            if energy < HYDROGEN_BOND_ENERGY_CUTOFF_KCAL_PER_MOL:
                bonds.add((acceptor_index, donor_index))
    return frozenset(bonds)


def _turns(hydrogen_bonds: frozenset[tuple[int, int]], length: int, size: int) -> set[int]:
    return {
        index
        for index in range(length - size)
        if (index, index + size) in hydrogen_bonds
    }


def _raw_beta_bridges(
    hydrogen_bonds: frozenset[tuple[int, int]], length: int
) -> dict[tuple[int, int], set[str]]:
    bridges: dict[tuple[int, int], set[str]] = {}

    def add(first: int, second: int, orientation: str) -> None:
        if first == second:
            return
        key = (min(first, second), max(first, second))
        bridges.setdefault(key, set()).add(orientation)

    for first in range(length):
        for second in range(first + 1, length):
            if (first, second) in hydrogen_bonds and (second, first) in hydrogen_bonds:
                add(first, second, "antiparallel")
            if (
                first > 0
                and second > 0
                and (first - 1, second) in hydrogen_bonds
                and (second - 1, first) in hydrogen_bonds
            ):
                add(first, second, "antiparallel")
            if (
                first > 0
                and second + 1 < length
                and (first - 1, second) in hydrogen_bonds
                and (second, first + 1) in hydrogen_bonds
            ):
                add(first, second, "parallel")
            if (
                second > 0
                and first + 1 < length
                and (second - 1, first) in hydrogen_bonds
                and (first, second + 1) in hydrogen_bonds
            ):
                add(first, second, "parallel")
    return bridges


def _mini_ladder_residues(bridges: Mapping[tuple[int, int], set[str]]) -> tuple[set[int], int]:
    residues: set[int] = set()
    ladder_count = 0
    for (first, second), orientations in bridges.items():
        for orientation in orientations:
            delta = 1 if orientation == "parallel" else -1
            for candidate in ((first + 1, second + delta), (first - 1, second - delta)):
                ordered = (min(candidate), max(candidate))
                if ordered not in bridges or orientation not in bridges[ordered]:
                    continue
                residues.update((first, second, *ordered))
                ladder_count += 1
                break
    return residues, ladder_count


def assign_secondary_structure_from_hydrogen_bonds(
    residues: Sequence[BackboneResidue], hydrogen_bonds: frozenset[tuple[int, int]]
) -> SecondaryStructureAssignment:
    """Assign helix, strand, and other labels from Kabsch-Sander bond patterns."""
    labels = ["other"] * len(residues)
    for turn_size in (3, 4, 5):
        turns = _turns(hydrogen_bonds, len(residues), turn_size)
        for first in turns:
            if first + 1 not in turns:
                continue
            for index in range(first + 1, min(len(residues), first + turn_size + 2)):
                labels[index] = "helix"
    bridges = _raw_beta_bridges(hydrogen_bonds, len(residues))
    ladder_residues, ladder_count = _mini_ladder_residues(bridges)
    for index in ladder_residues:
        if labels[index] == "other":
            labels[index] = "strand"
    helix_fraction = labels.count("helix") / len(labels) if labels else 0.0
    strand_fraction = labels.count("strand") / len(labels) if labels else 0.0
    return SecondaryStructureAssignment(
        labels=tuple(labels),
        helix_fraction=helix_fraction,
        strand_fraction=strand_fraction,
        hydrogen_bond_count=len(hydrogen_bonds),
        mini_ladder_count=ladder_count,
    )


def kabsch_sander_assignment(residues: Sequence[BackboneResidue]) -> SecondaryStructureAssignment:
    """Assign three secondary-structure classes from a backbone heavy-atom model."""
    return assign_secondary_structure_from_hydrogen_bonds(
        residues, kabsch_sander_hydrogen_bonds(residues)
    )


def is_all_alpha(assignment: SecondaryStructureAssignment, alpha_fraction_threshold: float) -> bool:
    """Apply the campaign-supplied all-alpha definition to a surrogate assignment."""
    if not math.isfinite(float(alpha_fraction_threshold)) or not 0 <= alpha_fraction_threshold <= 1:
        raise StructuralSurrogateError("all-alpha fraction threshold must be finite and between 0 and 1")
    return assignment.helix_fraction >= float(alpha_fraction_threshold)


def complete_linkage_clusters(
    candidate_ids: Sequence[str], distance_matrix: np.ndarray, cutoff: float
) -> dict[str, str]:
    """Cluster a symmetric distance matrix with deterministic complete linkage."""
    if not math.isfinite(float(cutoff)) or not 0 <= cutoff <= 1:
        raise StructuralSurrogateError("complete-linkage cutoff must be finite and between 0 and 1")
    matrix = np.asarray(distance_matrix, dtype=float)
    expected_shape = (len(candidate_ids), len(candidate_ids))
    if matrix.shape != expected_shape or not np.isfinite(matrix).all():
        raise StructuralSurrogateError("complete-linkage distance matrix must be finite and square")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise StructuralSurrogateError("complete-linkage candidate IDs must be unique")
    clusters = [tuple([index]) for index in range(len(candidate_ids))]
    while True:
        eligible: list[tuple[float, tuple[str, ...], int, int]] = []
        for first in range(len(clusters)):
            for second in range(first + 1, len(clusters)):
                maximum_distance = float(matrix[np.ix_(clusters[first], clusters[second])].max())
                if maximum_distance <= cutoff:
                    member_ids = tuple(sorted(candidate_ids[index] for index in clusters[first] + clusters[second]))
                    eligible.append((maximum_distance, member_ids, first, second))
        if not eligible:
            break
        _, _, first, second = min(eligible)
        merged = tuple(sorted(clusters[first] + clusters[second]))
        clusters = [cluster for index, cluster in enumerate(clusters) if index not in {first, second}]
        clusters.append(merged)
    ordered = sorted(clusters, key=lambda cluster: tuple(sorted(candidate_ids[index] for index in cluster)))
    identifiers: dict[str, str] = {}
    for cluster_index, cluster in enumerate(ordered, start=1):
        identifier = f"contact-map-overlap-f1-surrogate-cluster-{cluster_index:04d}"
        for member in cluster:
            identifiers[candidate_ids[member]] = identifier
    return identifiers


def cluster_contact_map_overlap(
    designs: Mapping[str, Sequence[BackboneResidue]],
    parameters: AlignmentParameters,
    cutoff: float,
) -> StructuralClusters:
    """Cluster designs at an F1 similarity cutoff by complete-linkage distance.

    ``cutoff`` has the same F1 interpretation as the novelty threshold.  The
    complete-linkage routine consumes distances, so the equivalent distance
    boundary is ``1 - cutoff``.
    """
    if not math.isfinite(float(cutoff)) or not 0 <= cutoff <= 1:
        raise StructuralSurrogateError("contact-map overlap cutoff must be finite and between 0 and 1")
    candidate_ids = tuple(sorted(designs))
    matrix = np.zeros((len(candidate_ids), len(candidate_ids)), dtype=float)
    pairwise_f1: dict[tuple[str, str], float] = {}
    for first, candidate_id in enumerate(candidate_ids):
        for second in range(first + 1, len(candidate_ids)):
            other_id = candidate_ids[second]
            result = contact_map_overlap(designs[candidate_id], designs[other_id], parameters)
            pairwise_f1[(candidate_id, other_id)] = result.f1
            matrix[first, second] = 1.0 - result.f1
            matrix[second, first] = matrix[first, second]
    return StructuralClusters(
        identifiers=complete_linkage_clusters(candidate_ids, matrix, 1.0 - float(cutoff)),
        pairwise_f1=pairwise_f1,
        cutoff=float(cutoff),
    )

