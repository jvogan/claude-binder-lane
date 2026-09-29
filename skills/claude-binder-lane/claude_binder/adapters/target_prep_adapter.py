#!/usr/bin/env python3
"""Prepare the target inputs one binder lane run reads.

This wrapper fills the `target-preparer` slot. It is the first stage of a run, so
every later stage waits on it. It reads a source structure, keeps the chains and
residues the campaign names, and writes five files under the current attempt
directory:

  <attempt>/<phase>/structures/<target_id>.pdb        normalized target structure
  <attempt>/<phase>/residue-maps/<target_id>.json     residue map the run scores against
  <attempt>/<phase>/site-definition.json              site the campaign designs against
  <attempt>/<phase>/inputs/rfd3-specification.json     derived RFdiffusion3 specification
  <attempt>/<phase>/target-manifest.json              the declared stage output

Two of the five are declared stage outputs, the manifest and the RFdiffusion3
specification. The lane compares the parser report against the declared outputs
and rejects a mismatch, so the other three stay undeclared and the manifest
carries their hashes.

The specification is a declared output because the backbone generator reads it
back out of the artifact root. Every path one stage hands another is published
there by the executor from a declared output. This wrapper used to write the
specification into the artifact root itself, which skipped the declaration, the
publication and the hash the artifact index records. The profile now declares a
`publish_path` for it, so the file is written here and the executor moves it.

`target_sha256` is the hash of the source structure, and `residue_map_sha256` is
the hash of the residue map the campaign registered. Later stages compare their
records against those two campaign values, so the manifest repeats them rather
than reporting the hash of a rewritten file. The files this wrapper writes carry
their own hash fields, and the residue map is copied byte for byte so the copy
keeps the registered hash.

The wrapper reads PDB and mmCIF sources with the standard library. It runs no
external tool and starts no subprocess.

Every path, chain ID, and residue selection arrives as a command-line argument.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from claude_binder.rfd3_specification import (
    SIDE_CHAIN_TIP_ATOMS,
    SpecificationError,
    derive_specification,
    residue_records_from_atoms,
    write_specification,
)

# The project targets this interpreter version in pyproject.toml.
MINIMUM_PYTHON = (3, 10)
REQUIRED_MODULES = ("argparse", "copy", "hashlib", "json", "os", "re", "shutil", "sys", "tempfile")
DEFAULT_MANIFEST_NAME = "target-manifest.json"
DEFAULT_SITE_DEFINITION_NAME = "site-definition.json"
# The attempt-directory name of the RFdiffusion3 specification. The profile's
# path_template spells the same name, and the two have to agree for the executor to
# find the file. The published name is different and lives in lane.
DEFAULT_SPECIFICATION_NAME = "rfd3-specification.json"
DEFAULT_STRUCTURE_SUBDIR = "structures"
DEFAULT_RESIDUE_MAP_SUBDIR = "residue-maps"
DEFAULT_RESIDUE_MAP_KEY = "source_to_cleaned"
DEFAULT_ALTERNATE_LOCATION = "A"
READ_BLOCK_BYTES = 1024 * 1024
# One residue is CHAIN:NUMBER with an optional insertion code, and a span is
# CHAIN:LOW-HIGH. Both forms match the residue pattern the lane runner accepts in
# a campaign site block.
RESIDUE_RE = re.compile(r"^([A-Za-z0-9]+):(-?\d+)([A-Za-z]?)$")
RESIDUE_SPAN_RE = re.compile(r"^([A-Za-z0-9]+):(\d+)-(\d+)$")
# The four site modes the lane runner registers.
SITE_MODES = (
    "explicit-residues",
    "reference-partner-contacts",
    "reference-pose-contacts",
    "spatial-pocket",
)
CIF_ABSENT_VALUES = frozenset({".", "?"})
# PDB record names that close an mmCIF `_atom_site` loop in practice. Neither
# is mmCIF. Both are written by tools this wrapper reads, and both appear only
# where the loop has ended, so the mmCIF reader treats them as terminators.
CIF_END_RECORDS = frozenset({"END", "ENDMDL"})
# A PDB coordinate line holds five columns of atom serial and four of residue
# number. A wider value would shift every later column, so the wrapper refuses
# the selection instead of writing a file no reader can parse.
PDB_SERIAL_LIMIT = 99999
PDB_RESIDUE_NUMBER_MINIMUM = -999
PDB_RESIDUE_NUMBER_MAXIMUM = 9999
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
NUCLEIC_TO_ONE = {
    "A": "A", "C": "C", "G": "G", "U": "U", "T": "T",
    "DA": "A", "DC": "C", "DG": "G", "DT": "T",
}
UNKNOWN_RESIDUE_LETTER = "X"

# The status values describe each generation hotspot in the normalized target
# structure. A target manifest keeps the values as data, so a later stage can
# distinguish an intentional reduction from a silent omission.
HOTSPOT_STATUS_COMPLETE = "complete"
HOTSPOT_STATUS_PARTIAL = "partial"
HOTSPOT_STATUS_ALTLOC = "altloc"
HOTSPOT_STATUS_ABSENT = "absent"
HOTSPOT_STATUS_SKIPPED = "skipped"
HOTSPOT_STATUSES = frozenset(
    {
        HOTSPOT_STATUS_COMPLETE,
        HOTSPOT_STATUS_PARTIAL,
        HOTSPOT_STATUS_ALTLOC,
        HOTSPOT_STATUS_ABSENT,
        HOTSPOT_STATUS_SKIPPED,
    }
)

# A skipped hotspot always carries one of these reasons. The strings identify
# conditions observed during target preparation.
HOTSPOT_SKIP_UNSUPPORTED_RESIDUE_TYPE = "unsupported_residue_type"
HOTSPOT_SKIP_INSUFFICIENT_MODELED_SIDE_CHAIN = "insufficient_modeled_side_chain"
HOTSPOT_SKIP_ALTLOC_WITHOUT_USABLE_TIPS = "altloc_without_usable_tips"
HOTSPOT_SKIP_REASONS = frozenset(
    {
        HOTSPOT_SKIP_UNSUPPORTED_RESIDUE_TYPE,
        HOTSPOT_SKIP_INSUFFICIENT_MODELED_SIDE_CHAIN,
        HOTSPOT_SKIP_ALTLOC_WITHOUT_USABLE_TIPS,
    }
)


class AdapterError(RuntimeError):
    """A condition the operator has to fix before the stage can run."""


@dataclass
class Atom:
    """One coordinate record of a source structure."""

    record: str
    name: str
    alternate_location: str
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    element: str

    @property
    def residue_id(self) -> str:
        """Return the CHAIN:NUMBER form the lane uses for one residue."""
        return f"{self.chain_id}:{self.residue_number}{self.insertion_code}"


@dataclass
class Residue:
    """One modeled residue of one chain."""

    residue_id: str
    residue_name: str
    number: int
    insertion_code: str
    atom_count: int


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of the complete file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write JSON to a path in one atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def missing_modules(names: tuple[str, ...] = REQUIRED_MODULES) -> list[str]:
    """Return the named standard-library modules that do not resolve."""
    missing: list[str] = []
    for name in names:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def guess_element(atom_name: str) -> str:
    """Return the element symbol for an atom name that carries no element column.

    This reads the first letter of the name, which is right for the elements a
    protein or nucleic chain carries. A two-letter element needs the element
    column, so a source that omits that column loses the second letter.
    """
    stripped = atom_name.strip().strip('"')
    for character in stripped:
        if character.isalpha():
            return character.upper()
    return ""


def alternate_location_kept(value: str, keep: str) -> bool:
    """Report whether one alternate location belongs in the normalized structure."""
    return value.strip() in {"", ".", "?", keep}


# ----------------------------------------------------------------------------
# Source parsing. The PDB reader follows the column layout; the mmCIF reader
# follows the `_atom_site` loop and prefers the auth_* columns, which carry the
# numbering a campaign site block names.
# ----------------------------------------------------------------------------


def parse_pdb_atoms(text: str) -> list[Atom]:
    """Return the coordinate records of the first model of a PDB file."""
    atoms: list[Atom] = []
    in_later_model = False
    for line in text.splitlines():
        if line.startswith("MODEL "):
            in_later_model = bool(atoms)
            continue
        if line.startswith("ENDMDL"):
            in_later_model = bool(atoms)
            continue
        if in_later_model or not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            residue_number = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (IndexError, ValueError):
            continue
        name = line[12:16].strip()
        element = line[76:78].strip() if len(line) >= 78 else ""
        atoms.append(
            Atom(
                record=line[:6].strip(),
                name=name,
                alternate_location=line[16:17],
                residue_name=line[17:20].strip(),
                chain_id=line[21:22].strip(),
                residue_number=residue_number,
                insertion_code=line[26:27].strip(),
                x=x,
                y=y,
                z=z,
                occupancy=parse_float(line[54:60], 1.0),
                b_factor=parse_float(line[60:66], 0.0),
                element=element or guess_element(name),
            )
        )
    return atoms


def parse_float(value: str, fallback: float) -> float:
    """Return a float for a fixed-width column, or the fallback when it is blank."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def cif_tokens(line: str) -> list[str]:
    """Split one mmCIF data line, keeping quoted values whole."""
    tokens: list[str] = []
    index = 0
    length = len(line)
    while index < length:
        character = line[index]
        if character in " \t":
            index += 1
            continue
        if character in "'\"":
            end = line.find(character, index + 1)
            if end == -1:
                tokens.append(line[index + 1 :])
                break
            tokens.append(line[index + 1 : end])
            index = end + 1
            continue
        end = index
        while end < length and line[end] not in " \t":
            end += 1
        tokens.append(line[index:end])
        index = end
    return tokens


def cif_value(tokens: list[str], columns: dict[str, int], *names: str) -> str:
    """Return the first column of `names` that the row carries."""
    for name in names:
        index = columns.get(name)
        if index is not None and index < len(tokens):
            value = tokens[index]
            if value not in CIF_ABSENT_VALUES:
                return value
    return ""


def parse_cif_atoms(text: str) -> list[Atom]:
    """Return the coordinate records of the first model of an mmCIF file."""
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        column_names: list[str] = []
        cursor = index + 1
        while cursor < len(lines) and lines[cursor].lstrip().startswith("_atom_site."):
            column_names.append(lines[cursor].strip().split(".", 1)[1])
            cursor += 1
        if not column_names:
            index += 1
            continue
        columns = {name: position for position, name in enumerate(column_names)}
        return read_cif_atom_rows(lines, cursor, columns)
    return []


def iter_cif_atom_rows(lines: list[str], start: int, column_count: int) -> Iterator[list[str]]:
    """Yield one `_atom_site` row at a time, across rows wrapped over lines.

    An mmCIF row is a count of values and not a line. The format lets a row run
    onto the following lines, and OpenFold3 NIM writes them that way. Reading
    one row per line dropped a wrapped row and built a corrupt atom out of a
    line carrying more than one row, in both cases without saying so. This is
    the same defect `binder_metrics._iter_atom_site_rows` fixes, measured in
    `notes/2026-09-11-session-review/DEFECTS.md` row 63.

    A semicolon-delimited multiline value is read as one value, because
    otherwise its words would enter the value stream and shift every row after
    it. A loop that ends part way through a row raises.
    """
    pending: list[str] = []
    text_field: list[str] | None = None
    for line in lines[start:]:
        stripped = line.strip()
        if text_field is not None:
            if line.startswith(";"):
                pending.append("\n".join(text_field))
                text_field = None
            else:
                text_field.append(line)
        elif line.startswith(";"):
            text_field = [line[1:]]
        elif not stripped:
            continue
        elif stripped.startswith(("#", "_", "loop_", "data_")):
            break
        elif not pending and stripped in CIF_END_RECORDS:
            # Not mmCIF. Several writers close a .cif with the PDB END record,
            # and it sits where the loop terminator belongs. Only honoured at a
            # row boundary, where the next value starts a row and never is one.
            break
        else:
            pending.extend(cif_tokens(stripped))

        while len(pending) >= column_count:
            yield pending[:column_count]
            del pending[:column_count]

    if pending:
        raise AdapterError(
            f"the _atom_site loop ends with {len(pending)} leftover value(s) against "
            f"{column_count} columns, so its last row is incomplete. The file is truncated "
            "or its column list disagrees with its rows."
        )


def read_cif_atom_rows(lines: list[str], start: int, columns: dict[str, int]) -> list[Atom]:
    """Return the atoms of one `_atom_site` loop, first model only."""
    atoms: list[Atom] = []
    first_model: str | None = None
    for tokens in iter_cif_atom_rows(lines, start, len(columns)):
        model = cif_value(tokens, columns, "pdbx_PDB_model_num") or "1"
        if first_model is None:
            first_model = model
        if model != first_model:
            continue
        raw_number = cif_value(tokens, columns, "auth_seq_id", "label_seq_id")
        try:
            residue_number = int(raw_number)
            x = float(cif_value(tokens, columns, "Cartn_x"))
            y = float(cif_value(tokens, columns, "Cartn_y"))
            z = float(cif_value(tokens, columns, "Cartn_z"))
        except ValueError:
            continue
        name = cif_value(tokens, columns, "auth_atom_id", "label_atom_id")
        atoms.append(
            Atom(
                record=cif_value(tokens, columns, "group_PDB") or "ATOM",
                name=name,
                alternate_location=cif_value(tokens, columns, "label_alt_id"),
                residue_name=cif_value(tokens, columns, "auth_comp_id", "label_comp_id"),
                chain_id=cif_value(tokens, columns, "auth_asym_id", "label_asym_id"),
                residue_number=residue_number,
                insertion_code=cif_value(tokens, columns, "pdbx_PDB_ins_code"),
                x=x,
                y=y,
                z=z,
                occupancy=parse_float(cif_value(tokens, columns, "occupancy"), 1.0),
                b_factor=parse_float(cif_value(tokens, columns, "B_iso_or_equiv"), 0.0),
                element=cif_value(tokens, columns, "type_symbol") or guess_element(name),
            )
        )
    return atoms


def structure_format(path: Path) -> str:
    """Return the source format one structure path names."""
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return "pdb"
    if suffix in {".cif", ".mmcif"}:
        return "cif"
    raise AdapterError(
        f"structure format is not registered: {path}. Pass a .pdb, .ent, .cif, or .mmcif file"
    )


def load_atoms(path: Path) -> tuple[list[Atom], str]:
    """Return the coordinate records of a source structure and its format."""
    if not path.is_file():
        raise AdapterError(f"source structure not found: {path}")
    source_format = structure_format(path)
    text = path.read_text(errors="replace")
    atoms = parse_pdb_atoms(text) if source_format == "pdb" else parse_cif_atoms(text)
    if not atoms:
        raise AdapterError(f"source structure carries no coordinate records: {path}")
    return atoms, source_format


# ----------------------------------------------------------------------------
# Chain and residue selection.
# ----------------------------------------------------------------------------


def source_chain_ids(atoms: list[Atom]) -> list[str]:
    """Return every chain ID of a source structure, in the order it appears."""
    ordered: list[str] = []
    for atom in atoms:
        if atom.chain_id not in ordered:
            ordered.append(atom.chain_id)
    return ordered


def parse_residue_range(value: str) -> tuple[str, int, int]:
    """Return the chain and bounds of one CHAIN:LOW-HIGH range."""
    match = RESIDUE_SPAN_RE.match(value)
    if match is None:
        raise AdapterError(f"residue range must read CHAIN:LOW-HIGH: {value}")
    low, high = int(match.group(2)), int(match.group(3))
    if low > high:
        raise AdapterError(f"residue range runs backwards: {value}")
    return match.group(1), low, high


def resolve_ranges(values: list[str], chains: list[str]) -> dict[str, tuple[int, int]]:
    """Return the residue bounds each kept chain is restricted to."""
    ranges: dict[str, tuple[int, int]] = {}
    for value in values:
        chain, low, high = parse_residue_range(value)
        if chain not in chains:
            raise AdapterError(f"residue range names a chain the run does not keep: {value}")
        if chain in ranges:
            raise AdapterError(f"chain {chain} carries more than one residue range")
        ranges[chain] = (low, high)
    return ranges


def select_atoms(
    atoms: list[Atom],
    *,
    chains: list[str],
    ranges: dict[str, tuple[int, int]],
    alternate_location: str,
    excluded_residue_names: set[str],
    keep_hetatm: bool,
) -> list[Atom]:
    """Return the atoms the normalized structure keeps, in source order."""
    kept: list[Atom] = []
    for atom in atoms:
        if atom.chain_id not in chains:
            continue
        if not keep_hetatm and atom.record == "HETATM":
            continue
        if atom.residue_name.upper() in excluded_residue_names:
            continue
        if not alternate_location_kept(atom.alternate_location, alternate_location):
            continue
        bounds = ranges.get(atom.chain_id)
        if bounds is not None and not bounds[0] <= atom.residue_number <= bounds[1]:
            continue
        kept.append(atom)
    return kept


def chain_residues(atoms: list[Atom]) -> dict[str, list[Residue]]:
    """Return the modeled residues of every chain, in ascending residue order."""
    counts: dict[str, dict[str, Residue]] = {}
    for atom in atoms:
        chain = counts.setdefault(atom.chain_id, {})
        residue = chain.get(atom.residue_id)
        if residue is None:
            chain[atom.residue_id] = Residue(
                residue_id=atom.residue_id,
                residue_name=atom.residue_name,
                number=atom.residue_number,
                insertion_code=atom.insertion_code,
                atom_count=1,
            )
            continue
        residue.atom_count += 1
    return {
        chain: sorted(residues.values(), key=lambda item: (item.number, item.insertion_code))
        for chain, residues in counts.items()
    }


def residue_letter(residue_name: str) -> str:
    """Return the one-letter code for a residue name, or X when it has none."""
    upper = residue_name.upper()
    if upper in THREE_TO_ONE:
        return THREE_TO_ONE[upper]
    if upper in NUCLEIC_TO_ONE:
        return NUCLEIC_TO_ONE[upper]
    return UNKNOWN_RESIDUE_LETTER


def contiguous_segments(numbers: list[int]) -> str:
    """Return a compact `low-high` summary of a sorted residue-number list."""
    segments: list[list[int]] = []
    for number in numbers:
        if segments and number == segments[-1][1] + 1:
            segments[-1][1] = number
            continue
        segments.append([number, number])
    return ",".join(f"{low}-{high}" if low != high else str(low) for low, high in segments)


def merge_residue_specs(
    repeated: list[str] | None, joined: list[str] | None, label: str
) -> list[str]:
    """Return one residue-spec list from the repeatable flag and the list flag.

    The lane renders a target's residue selection as one comma-separated token,
    `{{target_residues_csv}}` or `{{reference_contact_residues_csv}}`, so a
    profile that could only repeat a flag had to write the campaign's residues
    into its argument list as literal text. Splitting here is what lets the
    token fill the argument. The two forms combine, in the order given.

    Neither token may appear in a path template. The lane keeps both out of
    `PATH_TEMPLATE_TOKENS` for that reason, so nothing here relaxes
    `validate_path_template_charset`; a comma reaches an argv element and never
    a path.
    """
    specs = list(repeated or [])
    for value in joined or []:
        items = [item.strip() for item in value.split(",")]
        if not any(items):
            raise AdapterError(f"{label}s is empty: {value!r}")
        for item in items:
            if not item:
                raise AdapterError(f"{label}s holds an empty item: {value!r}")
            specs.append(item)
    return specs


def merge_chain_ids(repeated: list[str] | None, joined: list[str] | None) -> list[str]:
    """Return one chain-ID list from the repeatable flag and the list flag.

    `--chain` takes one chain and repeats. A campaign renders its declared chain
    list as one comma-separated token, `{{target_chains_csv}}`, so `--chains`
    takes that form. A profile that had only the repeatable flag wrote a chain
    letter into its argument list as literal text, and the letter that shipped
    was `A`, which dropped the second chain of every multi-chain target. The two
    forms combine, in the order given.
    """
    ids = list(repeated or [])
    for value in joined or []:
        items = [item.strip() for item in value.split(",")]
        if not any(items):
            raise AdapterError(f"--chains is empty: {value!r}")
        for item in items:
            if not item:
                raise AdapterError(f"--chains holds an empty item: {value!r}")
            ids.append(item)
    return ids


def resolve_site_residues(
    specs: list[str], residues: dict[str, list[Residue]], label: str
) -> list[str]:
    """Return the residue IDs a list of site specs names, refusing any that is absent."""
    resolved: list[str] = []
    for spec in specs:
        span = RESIDUE_SPAN_RE.match(spec)
        if span is not None:
            chain, low, high = parse_residue_range(spec)
            available = residues.get(chain)
            if available is None:
                raise AdapterError(f"{label} names a chain the structure does not carry: {spec}")
            matched = [item.residue_id for item in available if low <= item.number <= high]
            if not matched:
                raise AdapterError(f"{label} matched no residue: {spec}")
            resolved.extend(matched)
            continue
        single = RESIDUE_RE.match(spec)
        if single is None:
            raise AdapterError(
                f"{label} must read CHAIN:NUMBER or CHAIN:LOW-HIGH: {spec}"
            )
        chain = single.group(1)
        available = residues.get(chain)
        if available is None:
            raise AdapterError(f"{label} names a chain the structure does not carry: {spec}")
        residue_id = f"{chain}:{int(single.group(2))}{single.group(3)}"
        if residue_id not in {item.residue_id for item in available}:
            raise AdapterError(f"{label} names a residue the structure does not carry: {spec}")
        resolved.append(residue_id)
    ordered: list[str] = []
    for residue_id in resolved:
        if residue_id not in ordered:
            ordered.append(residue_id)
    return ordered


def configured_generation_hotspots(site: dict[str, Any]) -> tuple[list[str], str]:
    """Return the site field that constrains RFdiffusion3 hotspot generation."""
    discovery = site.get("discovery")
    if site.get("epitope_constraint") == "unconstrained" or (
        isinstance(discovery, dict) and discovery.get("enabled") is True
    ):
        return [], "discovery"
    hotspot_source = site.get("hotspot_source")
    hotspot_residues = site.get("hotspot_residues")
    if hotspot_source in {None, "explicit"} and isinstance(hotspot_residues, list):
        if hotspot_residues:
            return _string_residue_specs(hotspot_residues, "hotspot_residues"), "hotspot_residues"
    return _string_residue_specs(site.get("design_residues"), "design_residues"), "design_residues"


def _string_residue_specs(value: Any, field: str) -> list[str]:
    """Return one non-empty list of single-residue specifications."""
    if not isinstance(value, list) or not value:
        raise AdapterError(f"target site has no {field} list")
    entries: list[str] = []
    for item in value:
        if not isinstance(item, str) or RESIDUE_RE.fullmatch(item) is None:
            raise AdapterError(
                f"RFdiffusion3 generation hotspots require CHAIN:NUMBER entries in {field}: {item!r}"
            )
        if item not in entries:
            entries.append(item)
    return entries


def _residue_id_from_spec(specification: str) -> str:
    """Return the canonical residue ID for a validated single-residue specification."""
    match = RESIDUE_RE.fullmatch(specification)
    if match is None:
        raise AdapterError(f"generation hotspot must read CHAIN:NUMBER: {specification}")
    return f"{match.group(1)}:{int(match.group(2))}{match.group(3)}"


def reconcile_site_definition(
    site: dict[str, Any],
    *,
    site_mode: str,
    contact_cutoff_angstrom: float,
    atom_selection: str,
    resolved_design_residues: list[str],
    normalized_residues: dict[str, list[Residue]],
) -> None:
    """Reject a command whose site record disagrees with its resolved campaign site."""
    disagreements: list[str] = []
    if site.get("mode") != site_mode:
        disagreements.append(f"mode config={site.get('mode')!r} command={site_mode!r}")
    if site.get("contact_cutoff_angstrom") != contact_cutoff_angstrom:
        disagreements.append(
            "contact_cutoff_angstrom "
            f"config={site.get('contact_cutoff_angstrom')!r} command={contact_cutoff_angstrom!r}"
        )
    if site.get("atom_selection") != atom_selection:
        disagreements.append(
            f"atom_selection config={site.get('atom_selection')!r} command={atom_selection!r}"
        )
    try:
        configured = resolve_site_residues(
            _string_residue_specs(site.get("design_residues"), "design_residues"),
            normalized_residues,
            "config design_residues",
        )
    except AdapterError as exc:
        raise AdapterError(f"site definition is invalid: {exc}") from exc
    if set(configured) != set(resolved_design_residues):
        disagreements.append(
            f"design_residues config={sorted(configured)} command={sorted(resolved_design_residues)}"
        )
    if disagreements:
        raise AdapterError("site definition disagreement: " + "; ".join(disagreements))


def alternate_locations_by_residue(atoms: list[Atom]) -> dict[str, list[str]]:
    """Return the named alternate locations observed for each residue."""
    locations: dict[str, set[str]] = {}
    for atom in atoms:
        alternate_location = atom.alternate_location.strip()
        if alternate_location:
            locations.setdefault(atom.residue_id, set()).add(alternate_location)
    return {
        residue_id: sorted(values)
        for residue_id, values in locations.items()
        if len(values) > 1
    }


def _config_with_generation_hotspots(
    config: dict[str, Any], target_id: str, hotspots: list[str]
) -> dict[str, Any]:
    """Return a configuration that names only the classified usable hotspots."""
    effective = copy.deepcopy(config)
    for target in effective.get("targets", []):
        if isinstance(target, dict) and target.get("target_id") == target_id:
            site = target.get("site")
            if not isinstance(site, dict):
                raise AdapterError(f"target {target_id} has no site object")
            site["hotspot_source"] = "explicit"
            site["hotspot_residues"] = hotspots
            return effective
    raise AdapterError(f"resolved config has no target {target_id}")


def _specification_entry(specification: dict[str, Any]) -> dict[str, Any]:
    """Return the single RFdiffusion3 target entry a target preparation writes."""
    if len(specification) != 1:
        raise AdapterError("RFdiffusion3 specification must contain one target entry")
    entry = next(iter(specification.values()))
    if not isinstance(entry, dict):
        raise AdapterError("RFdiffusion3 target entry must be an object")
    return entry


def _hotspot_skip_reason(error: SpecificationError, *, has_alternate_locations: bool) -> str:
    """Map a known per-residue derivation error to a closed skip reason."""
    message = str(error)
    if "has no declared side-chain tip atom pair" in message:
        return HOTSPOT_SKIP_UNSUPPORTED_RESIDUE_TYPE
    if "lacks side-chain atom(s)" in message:
        if has_alternate_locations:
            return HOTSPOT_SKIP_ALTLOC_WITHOUT_USABLE_TIPS
        return HOTSPOT_SKIP_INSUFFICIENT_MODELED_SIDE_CHAIN
    raise AdapterError(f"RFdiffusion3 specification derivation failed: {error}") from error


def classify_generation_hotspots(
    config: dict[str, Any],
    *,
    target_id: str,
    selected_atoms: list[Atom],
    source_atoms: list[Atom],
    design_target_chain: str,
    alternate_location: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive a specification and a classified status manifest for every hotspot.

    The classification uses the normalized atom selection. It therefore describes
    the same coordinates that later stages receive.
    """
    target = next(
        (
            item
            for item in config.get("targets", [])
            if isinstance(item, dict) and item.get("target_id") == target_id
        ),
        None,
    )
    site = target.get("site") if isinstance(target, dict) else None
    if not isinstance(site, dict):
        raise AdapterError(f"target {target_id} has no site object")
    entries, source_field = configured_generation_hotspots(site)
    records = residue_records_from_atoms(selected_atoms)
    by_residue_id = {record.residue_id: record for record in records}
    alternate_locations = alternate_locations_by_residue(source_atoms)
    statuses: list[dict[str, Any]] = []
    usable: list[str] = []

    for entry in entries:
        residue_id = _residue_id_from_spec(entry)
        record = by_residue_id.get(residue_id)
        alternate_locations_for_residue = alternate_locations.get(residue_id, [])
        status: dict[str, Any] = {
            "residue_id": residue_id,
            "source_field": source_field,
            "included_in_generation": False,
        }
        if alternate_locations_for_residue:
            status["alternate_locations"] = alternate_locations_for_residue
            status["alternate_location_kept"] = alternate_location
            status["alternate_location_selection_rule"] = "blank or requested alternate location"
        if record is None:
            status.update(
                {
                    "status": HOTSPOT_STATUS_ABSENT,
                    "reason_code": "absent_from_normalized_structure",
                }
            )
            statuses.append(status)
            continue

        declared_tip_atoms = list(SIDE_CHAIN_TIP_ATOMS.get(record.residue_name.upper(), ()))
        missing_tip_atoms = sorted(set(declared_tip_atoms) - set(record.atom_names))
        status.update(
            {
                "residue_name": record.residue_name.upper(),
                "modeled_atom_names": sorted(record.atom_names),
                "declared_tip_atoms": declared_tip_atoms,
                "missing_declared_tip_atoms": missing_tip_atoms,
            }
        )
        single_hotspot_config = _config_with_generation_hotspots(config, target_id, [entry])
        try:
            entry_specification = _specification_entry(
                derive_specification(
                    single_hotspot_config,
                    target_id,
                    records,
                    design_target_chain=design_target_chain,
                )
            )
        except SpecificationError as exc:
            status.update(
                {
                    "status": HOTSPOT_STATUS_SKIPPED,
                    "reason_code": _hotspot_skip_reason(
                        exc, has_alternate_locations=bool(alternate_locations_for_residue)
                    ),
                    "reason": str(exc),
                }
            )
            statuses.append(status)
            continue

        selected_hotspots = entry_specification.get("select_hotspots")
        if not isinstance(selected_hotspots, dict):
            raise AdapterError("RFdiffusion3 target entry has no select_hotspots object")
        selected_tip_atoms = selected_hotspots.get(record.specification_id)
        if selected_tip_atoms is not None:
            status["selected_tip_atoms"] = str(selected_tip_atoms).split(",")
        substitutions = entry_specification.get("extra", {}).get("hotspot_tip_atom_substitutions", [])
        if not isinstance(substitutions, list):
            raise AdapterError("RFdiffusion3 hotspot substitutions must be a list")
        has_substitution = any(
            isinstance(substitution, dict) and substitution.get("residue") == residue_id
            for substitution in substitutions
        )
        if alternate_locations_for_residue:
            status["status"] = HOTSPOT_STATUS_ALTLOC
        elif has_substitution:
            status["status"] = HOTSPOT_STATUS_PARTIAL
        else:
            status["status"] = HOTSPOT_STATUS_COMPLETE
        status["included_in_generation"] = True
        usable.append(entry)
        statuses.append(status)

    if entries and not usable:
        detail = ", ".join(
            f"{status['residue_id']}={status['status']}"
            for status in statuses
        )
        raise AdapterError(
            "no admissible generation hotspot remains after classified degradation: " + detail
        )
    effective_config = _config_with_generation_hotspots(config, target_id, usable)
    try:
        specification = derive_specification(
            effective_config,
            target_id,
            records,
            design_target_chain=design_target_chain,
        )
    except SpecificationError as exc:
        raise AdapterError(f"RFdiffusion3 specification derivation failed: {exc}") from exc

    counts = {status_name: 0 for status_name in sorted(HOTSPOT_STATUSES)}
    for status in statuses:
        counts[status["status"]] += 1
    return specification, {
        "schema_version": 1,
        "generation_hotspot_field": source_field,
        "declared_hotspot_count": len(entries),
        "included_hotspot_count": len(usable),
        "reduced": len(usable) != len(entries),
        "status_counts": counts,
        "residues": statuses,
    }


# ----------------------------------------------------------------------------
# Output files.
# ----------------------------------------------------------------------------


def format_atom_name(name: str) -> str:
    """Return the four-column atom-name field of a PDB coordinate line."""
    if len(name) < 4 and not name[:1].isdigit():
        return f" {name:<3.3s}"
    return f"{name:<4.4s}"


def coordinate_line(atom: Atom, serial: int) -> str:
    """Return one PDB coordinate line for an atom."""
    record = "HETATM" if atom.record == "HETATM" else "ATOM"
    alternate = atom.alternate_location.strip() or " "
    insertion = atom.insertion_code.strip() or " "
    return (
        f"{record:<6s}{serial:>5d} {format_atom_name(atom.name)}{alternate:1s}"
        f"{atom.residue_name[:3]:>3s} {atom.chain_id[:1]:1s}{atom.residue_number:>4d}"
        f"{insertion:1s}   {atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
        f"{atom.occupancy:6.2f}{atom.b_factor:6.2f}          {atom.element[:2]:>2s}"
    )


def terminator_line(atom: Atom, serial: int) -> str:
    """Return the TER line that closes one chain."""
    insertion = atom.insertion_code.strip() or " "
    return (
        f"TER   {serial:>5d}      {atom.residue_name[:3]:>3s} "
        f"{atom.chain_id[:1]:1s}{atom.residue_number:>4d}{insertion:1s}"
    )


def check_pdb_columns(atoms: list[Atom]) -> None:
    """Refuse a selection no PDB coordinate line can hold."""
    if len(atoms) > PDB_SERIAL_LIMIT:
        raise AdapterError(
            f"selection holds {len(atoms)} atoms and a PDB serial column holds "
            f"{PDB_SERIAL_LIMIT}. Narrow the chains or the residue ranges"
        )
    for atom in atoms:
        if not PDB_RESIDUE_NUMBER_MINIMUM <= atom.residue_number <= PDB_RESIDUE_NUMBER_MAXIMUM:
            raise AdapterError(
                f"residue number {atom.residue_number} on chain {atom.chain_id} is outside "
                "the four columns a PDB residue-number field holds"
            )


def write_structure(path: Path, atoms: list[Atom], remarks: list[str]) -> int:
    """Write the normalized structure and return the atom count.

    Author chain IDs, residue numbers, and insertion codes carry over unchanged,
    so a residue ID keeps its meaning across the source and the normalized file.
    Only the atom serial is renumbered.
    """
    check_pdb_columns(atoms)
    lines = list(remarks)
    serial = 0
    previous: Atom | None = None
    for atom in atoms:
        if previous is not None and atom.chain_id != previous.chain_id:
            serial += 1
            lines.append(terminator_line(previous, serial))
        serial += 1
        lines.append(coordinate_line(atom, serial))
        previous = atom
    if previous is not None:
        serial += 1
        lines.append(terminator_line(previous, serial))
    lines.append("END")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return len(atoms)


def load_residue_map(path: Path, key: str) -> dict[str, str]:
    """Return the residue mapping one registered residue-map file carries."""
    if not path.is_file():
        raise AdapterError(f"residue map not found: {path}")
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AdapterError(f"residue map is not JSON: {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise AdapterError(f"residue map must hold a JSON object: {path}")
    mapping = document.get(key)
    if not isinstance(mapping, dict) or not mapping:
        raise AdapterError(f"residue map carries no non-empty {key} object: {path}")
    for source, normalized in mapping.items():
        if not isinstance(source, str) or not isinstance(normalized, str):
            raise AdapterError(f"residue map {key} must map strings to strings: {path}")
    return {str(source): str(value) for source, value in mapping.items()}


def check_residue_map(
    mapping: dict[str, str],
    *,
    source_residues: dict[str, list[Residue]],
    normalized_residues: dict[str, list[Residue]],
    path: Path,
) -> None:
    """Refuse a residue map whose entries do not resolve on both sides."""
    source_ids = {residue.residue_id for residues in source_residues.values() for residue in residues}
    normalized_ids = {
        residue.residue_id for residues in normalized_residues.values() for residue in residues
    }
    for source, normalized in sorted(mapping.items()):
        if RESIDUE_RE.match(source) is None:
            raise AdapterError(f"residue map key must read CHAIN:NUMBER: {source} in {path}")
        if RESIDUE_RE.match(normalized) is None:
            raise AdapterError(f"residue map value must read CHAIN:NUMBER: {normalized} in {path}")
        if source not in source_ids:
            raise AdapterError(
                f"residue map key {source} is absent from the source structure: {path}"
            )
        if normalized not in normalized_ids:
            raise AdapterError(
                f"residue map value {normalized} is absent from the normalized structure: {path}"
            )


def chain_record(chain_id: str, residues: list[Residue], design_target_chain: str) -> dict[str, Any]:
    """Return the manifest record of one kept chain."""
    return {
        "chain_id": chain_id,
        "role": "design-target" if chain_id == design_target_chain else "context",
        "residue_count": len(residues),
        "atom_count": sum(residue.atom_count for residue in residues),
        "residue_segments": contiguous_segments([residue.number for residue in residues]),
        "residue_ids": [residue.residue_id for residue in residues],
        "sequence": "".join(residue_letter(residue.residue_name) for residue in residues),
    }


def file_record(path: Path, role: str) -> dict[str, Any]:
    """Return the manifest record of one file this wrapper wrote."""
    return {
        "role": role,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


# ----------------------------------------------------------------------------
# Subcommands.
# ----------------------------------------------------------------------------


def resolve_output_path(attempt_dir: Path, path: Path, label: str) -> Path:
    """Return an output path and refuse one that leaves the attempt directory."""
    resolved = Path(os.path.normpath(path if path.is_absolute() else attempt_dir / path))
    if attempt_dir not in resolved.parents:
        raise AdapterError(f"{label} escapes the attempt directory: {path}")
    return resolved


def run(args: argparse.Namespace) -> int:
    """Prepare one target and write the stage outputs."""
    unconstrained = False
    named_chains = merge_chain_ids(args.chain, args.chains)
    if not named_chains:
        raise AdapterError("no chain was named. Pass --chain or --chains")
    chains = list(dict.fromkeys(named_chains))
    if len(chains) != len(named_chains):
        raise AdapterError("the chain selection repeats a chain ID")
    atoms, source_format = load_atoms(args.source_structure.expanduser().resolve())
    source_path = args.source_structure.expanduser().resolve()
    available_chains = source_chain_ids(atoms)
    unknown = [chain for chain in chains if chain not in available_chains]
    if unknown:
        raise AdapterError(
            f"the source structure does not carry chain {', '.join(unknown)}. "
            f"It carries {', '.join(available_chains)}"
        )
    if args.design_target_chain not in chains:
        raise AdapterError(
            f"--design-target-chain {args.design_target_chain} is not one of the kept chains: "
            f"{', '.join(chains)}"
        )
    ranges = resolve_ranges(args.residue_range, chains)
    selected = select_atoms(
        atoms,
        chains=chains,
        ranges=ranges,
        alternate_location=args.alternate_location,
        excluded_residue_names={name.upper() for name in args.exclude_residue_name},
        keep_hetatm=not args.exclude_hetatm,
    )
    if not selected:
        raise AdapterError("the chain and residue selection matched no atom")
    kept_chains = source_chain_ids(selected)
    dropped = [chain for chain in chains if chain not in kept_chains]
    if dropped:
        raise AdapterError(
            f"the residue selection emptied chain {', '.join(dropped)}"
        )

    source_residues = chain_residues(atoms)
    normalized_residues = chain_residues(selected)
    config: dict[str, Any] | None = None
    site_config: dict[str, Any] | None = None
    if args.config is not None:
        try:
            config_value = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AdapterError(f"resolved config cannot be read: {args.config}: {exc}") from exc
        if not isinstance(config_value, dict):
            raise AdapterError(f"resolved config must be a JSON object: {args.config}")
        config = config_value
        target = next(
            (
                item
                for item in config.get("targets", [])
                if isinstance(item, dict) and item.get("target_id") == args.target_id
            ),
            None,
        )
        candidate_site = target.get("site") if isinstance(target, dict) else None
        if not isinstance(candidate_site, dict):
            raise AdapterError(f"resolved config target {args.target_id} has no site object")
        site_config = candidate_site
        unconstrained = (
            site_config.get("epitope_constraint") == "unconstrained"
            or (
                isinstance(site_config.get("discovery"), dict)
                and site_config["discovery"].get("enabled") is True
            )
        )

    if unconstrained:
        site_specs = []
        reference_specs = []
        design_residues = []
        reference_residues = []
    else:
        site_specs = merge_residue_specs(args.site_residue, args.site_residues, "--site-residue")
        if not site_specs:
            raise AdapterError("no site residue was given. Pass --site-residue or --site-residues")
        reference_specs = merge_residue_specs(
            args.reference_contact_residue,
            args.reference_contact_residues,
            "--reference-contact-residue",
        )
        design_residues = resolve_site_residues(site_specs, normalized_residues, "--site-residue")
        reference_residues = resolve_site_residues(
            reference_specs, source_residues, "--reference-contact-residue"
        )
        if site_config is not None:
            reconcile_site_definition(
                site_config,
                site_mode=args.site_mode,
                contact_cutoff_angstrom=args.contact_cutoff_angstrom,
                atom_selection=args.atom_selection,
                resolved_design_residues=design_residues,
                normalized_residues=normalized_residues,
            )

    specification: dict[str, Any] | None = None
    hotspot_status_manifest: dict[str, Any] | None = None
    if config is not None:
        specification, hotspot_status_manifest = classify_generation_hotspots(
            config,
            target_id=args.target_id,
            selected_atoms=selected,
            source_atoms=atoms,
            design_target_chain=args.design_target_chain,
            alternate_location=args.alternate_location,
        )

    attempt_dir = args.attempt_dir.expanduser().resolve()
    phase_dir = attempt_dir / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_output_path(
        attempt_dir,
        args.manifest_path if args.manifest_path is not None else phase_dir / DEFAULT_MANIFEST_NAME,
        "manifest path",
    )
    structure_path = resolve_output_path(
        attempt_dir,
        phase_dir / args.structure_subdir / f"{args.target_id}.pdb",
        "normalized structure path",
    )
    residue_map_path = resolve_output_path(
        attempt_dir,
        phase_dir / args.residue_map_subdir / f"{args.target_id}.json",
        "residue map path",
    )
    site_definition_path = resolve_output_path(
        attempt_dir, phase_dir / DEFAULT_SITE_DEFINITION_NAME, "site definition path"
    )

    specification_path_value: Path | None = None
    if specification is not None:
        # The attempt-directory name carries no target ID because the parser
        # renders attempt_dir, phase, and stage_id only. The executor publishes
        # this file under the target-specific artifact-root name.
        specification_path_value = resolve_output_path(
            attempt_dir,
            phase_dir / "inputs" / DEFAULT_SPECIFICATION_NAME,
            "specification path",
        )
        write_specification(specification_path_value, specification)

    source_sha256 = sha256_file(source_path)
    atom_count = write_structure(
        structure_path,
        selected,
        [
            f"REMARK 900 TARGET {args.target_id}",
            f"REMARK 900 SOURCE STRUCTURE {source_path}",
            f"REMARK 900 SOURCE SHA256 {source_sha256}",
            f"REMARK 900 KEPT CHAINS {','.join(kept_chains)}",
        ],
    )

    registered_residue_map = args.residue_map.expanduser().resolve()
    mapping = load_residue_map(registered_residue_map, args.residue_map_key)
    check_residue_map(
        mapping,
        source_residues=source_residues,
        normalized_residues=normalized_residues,
        path=registered_residue_map,
    )
    residue_map_path.parent.mkdir(parents=True, exist_ok=True)
    # The copy keeps the registered bytes, so its hash stays the value later
    # stages compare their records against.
    shutil.copyfile(registered_residue_map, residue_map_path)

    site = {
        "mode": args.site_mode,
        "design_residues": site_specs,
        "resolved_design_residues": design_residues,
        "reference_contact_residues": reference_specs,
        "resolved_reference_contact_residues": reference_residues,
        "contact_cutoff_angstrom": args.contact_cutoff_angstrom,
        "atom_selection": args.atom_selection,
        "epitope_constraint": "unconstrained" if unconstrained else "constrained",
    }
    if hotspot_status_manifest is not None:
        site["hotspot_status_manifest"] = hotspot_status_manifest
    write_json(
        site_definition_path,
        {
            "schema_version": 1,
            "target_id": args.target_id,
            "design_target_chain_id": args.design_target_chain,
            "binder_chain_id": args.binder_chain,
            **site,
        },
    )

    manifest = {
        "schema_version": 1,
        "artifact_type": "target-manifest",
        "target_id": args.target_id,
        "source_id": args.source_id,
        "phase": args.phase,
        "target_sha256": source_sha256,
        "residue_map_sha256": sha256_file(residue_map_path),
        "source_structure_path": str(source_path),
        "source_structure_format": source_format,
        "source_chain_ids": available_chains,
        "chain_ids": kept_chains,
        "design_target_chain_id": args.design_target_chain,
        "binder_chain_id": args.binder_chain,
        "chains": [
            chain_record(chain, normalized_residues[chain], args.design_target_chain)
            for chain in kept_chains
        ],
        "residue_count": sum(len(normalized_residues[chain]) for chain in kept_chains),
        "atom_count": atom_count,
        "residue_ranges": {chain: f"{low}-{high}" for chain, (low, high) in sorted(ranges.items())},
        "excluded_residue_names": sorted({name.upper() for name in args.exclude_residue_name}),
        "hetatm_records_kept": not args.exclude_hetatm,
        "alternate_location_kept": args.alternate_location,
        "normalized_structure_path": str(structure_path),
        "normalized_structure_sha256": sha256_file(structure_path),
        "residue_map_path": str(residue_map_path),
        "residue_map_source_path": str(registered_residue_map),
        "residue_map_key": args.residue_map_key,
        "residue_map_entry_count": len(mapping),
        "site_definition_path": str(site_definition_path),
        "site_definition_sha256": sha256_file(site_definition_path),
        "site": site,
        "files": [
            file_record(structure_path, "normalized-structure"),
            file_record(residue_map_path, "residue-map"),
            file_record(site_definition_path, "site-definition"),
        ],
    }
    if specification_path_value is not None:
        manifest["rfd3_specification_path"] = str(specification_path_value)
        manifest["rfd3_specification_sha256"] = sha256_file(specification_path_value)
        manifest["files"].append(file_record(specification_path_value, "rfd3-specification"))
    if hotspot_status_manifest is not None:
        manifest["hotspot_status_manifest"] = hotspot_status_manifest
    write_json(manifest_path, manifest)
    print(
        f"target prep adapter: target={args.target_id} phase={args.phase} "
        f"chains={','.join(kept_chains)} residues={manifest['residue_count']} "
        f"atoms={atom_count} manifest={manifest_path}"
    )
    return 0


def toolcheck(args: argparse.Namespace) -> int:
    """Report whether the runtime this wrapper needs resolves."""
    if sys.version_info < MINIMUM_PYTHON:
        raise AdapterError(
            f"interpreter {sys.version_info.major}.{sys.version_info.minor} is below the "
            f"{MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} this wrapper needs"
        )
    missing = missing_modules()
    if missing:
        raise AdapterError(f"standard-library modules do not resolve: {', '.join(missing)}")
    print(f"target prep adapter: interpreter {sys.executable}")
    print(
        "target prep adapter: standard-library modules resolve: " + ", ".join(REQUIRED_MODULES)
    )
    print("target prep adapter: the wrapper runs no external tool and starts no subprocess")
    if args.source_structure is not None:
        source_path = args.source_structure.expanduser().resolve()
        atoms, source_format = load_atoms(source_path)
        chains = source_chain_ids(atoms)
        print(
            f"target prep adapter: source {source_path} format={source_format} "
            f"atoms={len(atoms)} chains={','.join(chains)}"
        )
    if args.residue_map is not None:
        residue_map_path = args.residue_map.expanduser().resolve()
        mapping = load_residue_map(residue_map_path, args.residue_map_key)
        print(
            f"target prep adapter: residue map {residue_map_path} entries={len(mapping)}"
        )
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("toolcheck", help="Probe the runtime without preparing.")
    check_parser.add_argument(
        "--source-structure",
        type=Path,
        default=None,
        help="Optional source structure to read. The probe reports its chains and atom count.",
    )
    check_parser.add_argument(
        "--residue-map",
        type=Path,
        default=None,
        help="Optional registered residue map to read. The probe reports its entry count.",
    )
    check_parser.add_argument(
        "--residue-map-key",
        default=DEFAULT_RESIDUE_MAP_KEY,
        help=f"Key holding the residue mapping. Defaults to {DEFAULT_RESIDUE_MAP_KEY}.",
    )

    run_parser = subparsers.add_parser("run", help="Prepare one target for a stage phase.")
    run_parser.add_argument(
        "--phase", required=True, help="Stage phase name, such as single, smoke, or scale."
    )
    run_parser.add_argument(
        "--attempt-dir", type=Path, required=True, help="Attempt directory that owns the outputs."
    )
    run_parser.add_argument(
        "--target-id", required=True, help="Target ID. Match the target_id the campaign registers."
    )
    run_parser.add_argument(
        "--source-id",
        default=None,
        help="Source ID the campaign registers for this structure. Recorded in the manifest.",
    )
    run_parser.add_argument(
        "--source-structure",
        type=Path,
        required=True,
        help="Source structure to read. Accepts .pdb, .ent, .cif, and .mmcif.",
    )
    run_parser.add_argument(
        "--chain",
        action="append",
        default=[],
        metavar="CHAIN_ID",
        help=(
            "Chain to keep. Repeat the flag for a multi-chain target. Required unless "
            "--chains carries the list."
        ),
    )
    run_parser.add_argument(
        "--chains",
        action="append",
        default=[],
        metavar="CHAIN_ID,CHAIN_ID",
        help=(
            "Every chain to keep as one comma-separated value, which is the form "
            "{{target_chains_csv}} renders. Combines with --chain."
        ),
    )
    run_parser.add_argument(
        "--design-target-chain",
        required=True,
        help="Chain the run designs against. Must be one of the kept chains.",
    )
    run_parser.add_argument(
        "--binder-chain",
        default=None,
        help="Chain ID the run gives the designed binder. Recorded in the manifest.",
    )
    run_parser.add_argument(
        "--residue-range",
        action="append",
        default=[],
        metavar="CHAIN:LOW-HIGH",
        help="Restrict one kept chain to a residue range. Repeat the flag for more chains.",
    )
    run_parser.add_argument(
        "--site-residue",
        action="append",
        default=[],
        metavar="RESIDUE",
        help=(
            "Site residue as CHAIN:NUMBER or CHAIN:LOW-HIGH. Repeat the flag for more residues. "
            "Every residue has to be in the normalized structure. Required unless "
            "--site-residues carries the list."
        ),
    )
    run_parser.add_argument(
        "--site-residues",
        action="append",
        default=[],
        metavar="RESIDUE,RESIDUE",
        help=(
            "The same residues as one comma-separated value, which is the form "
            "{{target_residues_csv}} renders. Combines with --site-residue."
        ),
    )
    run_parser.add_argument(
        "--reference-contact-residue",
        action="append",
        default=[],
        metavar="RESIDUE",
        help=(
            "Reference contact residue as CHAIN:NUMBER or CHAIN:LOW-HIGH. Repeat the flag for "
            "more residues. Every residue has to be in the source structure."
        ),
    )
    run_parser.add_argument(
        "--reference-contact-residues",
        action="append",
        default=[],
        metavar="RESIDUE,RESIDUE",
        help=(
            "The same residues as one comma-separated value, which is the form "
            "{{reference_contact_residues_csv}} renders. Combines with "
            "--reference-contact-residue."
        ),
    )
    run_parser.add_argument(
        "--site-mode",
        required=True,
        choices=SITE_MODES,
        help="Site mode. Match the site.mode the campaign registers.",
    )
    run_parser.add_argument(
        "--contact-cutoff-angstrom",
        type=float,
        required=True,
        help="Contact cutoff in angstrom. Match the site.contact_cutoff_angstrom the campaign registers.",
    )
    run_parser.add_argument(
        "--atom-selection",
        required=True,
        help="Atom selection. Match the site.atom_selection the campaign registers.",
    )
    run_parser.add_argument(
        "--residue-map",
        type=Path,
        required=True,
        help=(
            "Residue map the campaign registered. The wrapper checks every entry against the "
            "structures and copies the file into the attempt directory."
        ),
    )
    run_parser.add_argument(
        "--residue-map-key",
        default=DEFAULT_RESIDUE_MAP_KEY,
        help=f"Key holding the residue mapping. Defaults to {DEFAULT_RESIDUE_MAP_KEY}.",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Resolved campaign config. Supplying it derives the RFdiffusion3 "
            "specification into <phase>/inputs/<target_id>-rfd3.json."
        ),
    )
    run_parser.add_argument(
        "--exclude-residue-name",
        action="append",
        default=[],
        metavar="NAME",
        help="Drop every residue with this name. Repeat the flag for more names.",
    )
    run_parser.add_argument(
        "--exclude-hetatm",
        action="store_true",
        help="Drop HETATM records. The normalized structure keeps them by default.",
    )
    run_parser.add_argument(
        "--alternate-location",
        default=DEFAULT_ALTERNATE_LOCATION,
        help=(
            "Alternate location to keep alongside the records that name none. Defaults to "
            f"{DEFAULT_ALTERNATE_LOCATION}."
        ),
    )
    run_parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=f"Manifest path. Defaults to {DEFAULT_MANIFEST_NAME} in the phase directory.",
    )
    run_parser.add_argument(
        "--structure-subdir",
        default=DEFAULT_STRUCTURE_SUBDIR,
        help=(
            "Normalized structure directory inside the phase directory. Defaults to "
            f"{DEFAULT_STRUCTURE_SUBDIR}."
        ),
    )
    run_parser.add_argument(
        "--residue-map-subdir",
        default=DEFAULT_RESIDUE_MAP_SUBDIR,
        help=(
            "Residue map directory inside the phase directory. Defaults to "
            f"{DEFAULT_RESIDUE_MAP_SUBDIR}."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        return toolcheck(args) if args.command == "toolcheck" else run(args)
    except AdapterError as exc:
        print(f"target prep adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
