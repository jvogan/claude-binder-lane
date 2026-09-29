"""CPU-only geometry calculations for the pre-folding cheap screen.

The module receives every numerical convention through its caller. It supplies
no contact cutoff, van der Waals radius, SASA probe radius, or sampling count.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import binder_metrics


HYDROGEN_ELEMENTS = frozenset({"H", "D", "T"})


class ScreenGeometryError(ValueError):
    """A structure or geometry convention cannot support the cheap screen."""


@dataclass(frozen=True)
class InterfaceGeometry:
    """Geometry values derived from a painted two-chain complex."""

    hard_clash_count: int
    hotspot_coverage_count: int
    hotspot_count: int
    interface_buried_sasa_angstrom2: float
    interface_apolar_fraction: float
    contact_density: float
    interchain_residue_contact_count: int
    interface_residue_count: int
    binder_interface_indices: tuple[int, ...]


def read_structure(path: str) -> binder_metrics.Structure:
    """Read a PDB or mmCIF complex through the package's local parser."""
    try:
        return binder_metrics.parse_structure_atoms(Path(path), argument="screen structure")
    except (OSError, binder_metrics.MetricInputError, UnicodeDecodeError) as exc:
        raise ScreenGeometryError(str(exc)) from exc


def chain_residues(
    structure: binder_metrics.Structure, chain_id: str
) -> list[binder_metrics.Residue]:
    """Return one named chain or raise a screen-specific error."""
    try:
        return structure.chain_residues(chain_id)
    except binder_metrics.MetricInputError as exc:
        raise ScreenGeometryError(str(exc)) from exc


def _coordinates(residue: binder_metrics.Residue, atom_names: Sequence[str]) -> np.ndarray:
    coordinates = residue.named_atom_coords(atom_names)
    if coordinates.shape[0] != len(atom_names):
        raise ScreenGeometryError(
            f"residue {residue.label} is missing one of {', '.join(atom_names)}"
        )
    return coordinates


def target_ca_rmsd_angstrom(
    painted_target: Sequence[binder_metrics.Residue],
    input_target: Sequence[binder_metrics.Residue],
) -> float:
    """Superpose matching target C-alphas and return the RMSD in Angstroms."""
    painted_by_key = {(residue.seq_id, residue.ins_code): residue for residue in painted_target}
    input_by_key = {(residue.seq_id, residue.ins_code): residue for residue in input_target}
    # Building a dict from a sequence drops a repeated key silently, and the comparison below
    # only reads the surviving keys. Two residues sharing a (seq_id, ins_code) shrink both
    # sides together, so the sets still match while the superposition quietly leaves residues
    # out. Reachable: `chain_indices` matches on `auth_chain` while the key here does not carry
    # a chain at all, so one conformer can displace another. The `comp_id` check below catches
    # a different residue type, not a second copy of the same one.
    for label, mapping, residues in (
        ("painted target", painted_by_key, painted_target),
        ("input target", input_by_key, input_target),
    ):
        if len(mapping) != len(residues):
            seen: dict[tuple, int] = {}
            for residue in residues:
                key = (residue.seq_id, residue.ins_code)
                seen[key] = seen.get(key, 0) + 1
            repeated = sorted(key for key, count in seen.items() if count > 1)
            raise ScreenGeometryError(
                f"the {label} repeats residue identifiers {repeated}, so "
                f"{len(residues) - len(mapping)} residues would be dropped from the superposition"
            )
    if set(painted_by_key) != set(input_by_key):
        raise ScreenGeometryError("painted target residue identifiers differ from the input target")
    if len(painted_by_key) < 3:
        raise ScreenGeometryError("target identity superposition needs at least three residues")
    mobile: list[np.ndarray] = []
    reference: list[np.ndarray] = []
    for key in sorted(painted_by_key):
        painted = painted_by_key[key]
        expected = input_by_key[key]
        if painted.comp_id and expected.comp_id and painted.comp_id != expected.comp_id:
            raise ScreenGeometryError(
                f"target residue {painted.label} differs from input residue {expected.label}"
            )
        mobile.append(_coordinates(painted, ("CA",))[0])
        reference.append(_coordinates(expected, ("CA",))[0])
    mobile_array = np.asarray(mobile, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    mobile_center = mobile_array.mean(axis=0)
    reference_center = reference_array.mean(axis=0)
    centered_mobile = mobile_array - mobile_center
    centered_reference = reference_array - reference_center
    left, _, right = np.linalg.svd(centered_mobile.T @ centered_reference)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    fitted = centered_mobile @ rotation + reference_center
    return float(np.sqrt(np.mean(np.sum((fitted - reference_array) ** 2, axis=1))))


def _heavy_atoms(
    residues: Sequence[binder_metrics.Residue],
    radii_angstrom: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return heavy coordinates, radii, elements, and residue positions."""
    coordinates: list[tuple[float, float, float]] = []
    radii: list[float] = []
    elements: list[str] = []
    owners: list[int] = []
    for residue_index, residue in enumerate(residues):
        for element, coordinate in zip(residue.elements, residue.coords):
            normalized = element.upper()
            if normalized in HYDROGEN_ELEMENTS:
                continue
            radius = radii_angstrom.get(normalized)
            if radius is None:
                raise ScreenGeometryError(
                    f"van der Waals radius is absent for element {normalized!r} at {residue.label}"
                )
            coordinates.append(coordinate)
            radii.append(float(radius))
            elements.append(normalized)
            owners.append(residue_index)
    if not coordinates:
        raise ScreenGeometryError("chain has no heavy atoms")
    return (
        np.asarray(coordinates, dtype=float),
        np.asarray(radii, dtype=float),
        np.asarray(elements, dtype=object),
        np.asarray(owners, dtype=int),
    )


def _distance_squared(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)


def hard_clash_count(
    target_residues: Sequence[binder_metrics.Residue],
    binder_residues: Sequence[binder_metrics.Residue],
    radii_angstrom: Mapping[str, float],
    tolerance_angstrom: float,
) -> int:
    """Count interchain heavy-atom pairs below summed radii minus tolerance."""
    target_coords, target_radii, _, _ = _heavy_atoms(target_residues, radii_angstrom)
    binder_coords, binder_radii, _, _ = _heavy_atoms(binder_residues, radii_angstrom)
    count = 0
    # A target-atom block bounds temporary allocation for malformed oversized files.
    for start in range(0, target_coords.shape[0], 256):
        stop = min(start + 256, target_coords.shape[0])
        distances_squared = _distance_squared(target_coords[start:stop], binder_coords)
        cutoffs = target_radii[start:stop, None] + binder_radii[None, :] - tolerance_angstrom
        if bool(np.any(cutoffs <= 0)):
            raise ScreenGeometryError("clash tolerance is at least one summed van der Waals radius")
        count += int(np.count_nonzero(distances_squared < cutoffs**2))
    return count


def interchain_contacts(
    target_residues: Sequence[binder_metrics.Residue],
    binder_residues: Sequence[binder_metrics.Residue],
    radii_angstrom: Mapping[str, float],
    contact_cutoff_angstrom: float,
) -> tuple[set[tuple[int, int]], set[int], set[int]]:
    """Return residue contacts and each chain's interface residue positions."""
    target_coords, _, _, target_owners = _heavy_atoms(target_residues, radii_angstrom)
    binder_coords, _, _, binder_owners = _heavy_atoms(binder_residues, radii_angstrom)
    contacts: set[tuple[int, int]] = set()
    cutoff_squared = contact_cutoff_angstrom**2
    for start in range(0, target_coords.shape[0], 256):
        stop = min(start + 256, target_coords.shape[0])
        hit = _distance_squared(target_coords[start:stop], binder_coords) <= cutoff_squared
        row_indices, column_indices = np.nonzero(hit)
        for row_index, column_index in zip(row_indices, column_indices):
            contacts.add(
                (int(target_owners[start + row_index]), int(binder_owners[column_index]))
            )
    target_interface = {pair[0] for pair in contacts}
    binder_interface = {pair[1] for pair in contacts}
    return contacts, target_interface, binder_interface


def fibonacci_sphere(point_count: int) -> tuple[tuple[float, float, float], ...]:
    """Return deterministic, nearly uniform unit-sphere sample points."""
    if point_count < 1:
        raise ScreenGeometryError("SASA sphere_point_count must be positive")
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    points: list[tuple[float, float, float]] = []
    for index in range(point_count):
        z = 1.0 - (2.0 * (index + 0.5) / point_count)
        radial = math.sqrt(max(0.0, 1.0 - z * z))
        angle = golden_angle * index
        points.append((math.cos(angle) * radial, math.sin(angle) * radial, z))
    return tuple(points)


def _cell(coordinate: np.ndarray, width: float) -> tuple[int, int, int]:
    return tuple(int(math.floor(value / width)) for value in coordinate)


def shrake_rupley_sasa(
    residues: Sequence[binder_metrics.Residue],
    radii_angstrom: Mapping[str, float],
    probe_radius_angstrom: float,
    sphere_point_count: int,
    apolar_elements: frozenset[str],
) -> tuple[float, float]:
    """Return total and declared-apolar Shrake-Rupley SASA in square Angstroms."""
    coordinates, radii, elements, _ = _heavy_atoms(residues, radii_angstrom)
    expanded_radii = radii + probe_radius_angstrom
    max_expanded_radius = float(np.max(expanded_radii))
    cell_width = max_expanded_radius * 2.0
    if cell_width <= 0:
        raise ScreenGeometryError("SASA expanded radii must be positive")
    grid: dict[tuple[int, int, int], list[int]] = {}
    for index, coordinate in enumerate(coordinates):
        grid.setdefault(_cell(coordinate, cell_width), []).append(index)
    samples = np.asarray(fibonacci_sphere(sphere_point_count), dtype=float)
    total = 0.0
    apolar = 0.0
    neighbor_offsets = tuple(
        (first, second, third)
        for first in (-1, 0, 1)
        for second in (-1, 0, 1)
        for third in (-1, 0, 1)
    )
    for index, coordinate in enumerate(coordinates):
        base_cell = _cell(coordinate, cell_width)
        neighbors: list[int] = []
        for first, second, third in neighbor_offsets:
            neighbors.extend(
                grid.get((base_cell[0] + first, base_cell[1] + second, base_cell[2] + third), [])
            )
        neighbors = [neighbor for neighbor in neighbors if neighbor != index]
        points = coordinate + samples * expanded_radii[index]
        exposed = np.ones(sphere_point_count, dtype=bool)
        for neighbor in neighbors:
            if not bool(exposed.any()):
                break
            delta = points[exposed] - coordinates[neighbor]
            covered = np.sum(delta * delta, axis=1) < expanded_radii[neighbor] ** 2
            exposed_indices = np.flatnonzero(exposed)
            exposed[exposed_indices[covered]] = False
        area = 4.0 * math.pi * expanded_radii[index] ** 2 * (int(exposed.sum()) / sphere_point_count)
        total += area
        if str(elements[index]) in apolar_elements:
            apolar += area
    return total, apolar


def interface_geometry(
    target_residues: Sequence[binder_metrics.Residue],
    binder_residues: Sequence[binder_metrics.Residue],
    hotspot_labels: frozenset[str],
    *,
    radii_angstrom: Mapping[str, float],
    clash_tolerance_angstrom: float,
    contact_cutoff_angstrom: float,
    sasa_probe_radius_angstrom: float,
    sasa_sphere_point_count: int,
    apolar_elements: frozenset[str],
) -> InterfaceGeometry:
    """Calculate the screen's painted-interface geometry values."""
    clashes = hard_clash_count(
        target_residues, binder_residues, radii_angstrom, clash_tolerance_angstrom
    )
    contacts, target_interface, binder_interface = interchain_contacts(
        target_residues, binder_residues, radii_angstrom, contact_cutoff_angstrom
    )
    contacted_labels = {target_residues[index].label for index in target_interface}
    target_sasa, target_apolar = shrake_rupley_sasa(
        target_residues,
        radii_angstrom,
        sasa_probe_radius_angstrom,
        sasa_sphere_point_count,
        apolar_elements,
    )
    binder_sasa, binder_apolar = shrake_rupley_sasa(
        binder_residues,
        radii_angstrom,
        sasa_probe_radius_angstrom,
        sasa_sphere_point_count,
        apolar_elements,
    )
    complex_sasa, complex_apolar = shrake_rupley_sasa(
        [*target_residues, *binder_residues],
        radii_angstrom,
        sasa_probe_radius_angstrom,
        sasa_sphere_point_count,
        apolar_elements,
    )
    total_lost_area = max(0.0, target_sasa + binder_sasa - complex_sasa)
    apolar_lost_area = max(0.0, target_apolar + binder_apolar - complex_apolar)
    interface_residue_count = len(target_interface) + len(binder_interface)
    return InterfaceGeometry(
        hard_clash_count=clashes,
        hotspot_coverage_count=len(contacted_labels & hotspot_labels),
        hotspot_count=len(hotspot_labels),
        interface_buried_sasa_angstrom2=total_lost_area / 2.0,
        interface_apolar_fraction=(apolar_lost_area / total_lost_area if total_lost_area else 0.0),
        contact_density=(len(contacts) / interface_residue_count if interface_residue_count else 0.0),
        interchain_residue_contact_count=len(contacts),
        interface_residue_count=interface_residue_count,
        binder_interface_indices=tuple(sorted(binder_interface)),
    )


def secondary_structure_features(path: str, binder_chain_id: str) -> dict[str, float]:
    """Return ordered fraction, terminal tail length, and C-alpha radius of gyration."""
    from .. import structural_surrogates

    try:
        chains = structural_surrogates.parse_backbones(Path(path))
        residues = chains[binder_chain_id]
        assignment = structural_surrogates.kabsch_sander_assignment(residues)
    except (KeyError, structural_surrogates.StructuralSurrogateError, OSError) as exc:
        raise ScreenGeometryError(f"secondary-structure surrogate failed: {exc}") from exc
    labels = assignment.labels
    leading = next((index for index, label in enumerate(labels) if label != "other"), len(labels))
    trailing = next(
        (index for index, label in enumerate(reversed(labels)) if label != "other"), len(labels)
    )
    coordinates = np.asarray([residue.atoms["CA"] for residue in residues], dtype=float)
    center = coordinates.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum((coordinates - center) ** 2, axis=1))))
    return {
        "ordered_fraction": float(assignment.helix_fraction + assignment.strand_fraction),
        "terminal_tail_length": float(max(leading, trailing)),
        "radius_of_gyration_angstrom": radius,
    }
