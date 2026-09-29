"""Metric computation for the binder lane predictor arms.

This module is pure computation. It writes nothing, it opens no run directory,
and it knows nothing about the stage layout. Every value it returns is derived
from a PAE matrix, one or two structures, and a site definition handed to it by
the caller.

Three functions carry the whole surface:

    compute_ipsae         ipSAE in both directions, plus the interface summaries
    compute_dockq         sc_DockQ against a reference pose
    compute_site_metrics  epitope agreement on the target

Both sides of the contract import this one module, because the executor checks
that IPSAE_IMPLEMENTATION_REVISION and DOCKQ_IMPLEMENTATION_REVISION are the
same string on every row of a cohort.

Dependencies are the standard library plus numpy. The GPU container has numpy.

Where the numbers come from
---------------------------

Every constant below is traceable to a document. The two that matter most:

ipSAE follows Dunbrack 2025, "Res ipSAE loquuntur" (bioRxiv, PMID 39990437),
Equations 14 and 16, using the d0res variant. The campaign's published
reproduction reference (`ref/docs/INSILICO.md` section 4) names the same
variant and the same 10 Angstrom PAE cutoff, and the resolved campaign config
carries `scoring.implementations.ipsae_interface_cutoff_angstrom = 10.0`.

DockQ follows Basu and Wallner 2016 (PLOS ONE, PMID 27560519). The atom
selection for sc_DockQ follows `ref/docs/INSILICO.md` section 4: binder
backbone atoms against target heavy atoms, with all target protein chains
treated as one receptor.

Chain order comes from the structure
------------------------------------

The published reference states that chain order in a co-fold file is not
uniform across predictors, and that ESMFold2 writes the binder first. The
artifact writer derives a target and binder chain for each structure from the
known sequences. The metric functions then receive those explicit mappings.
Passing them the wrong way round still inverts the directional ipSAE values,
so the caller must derive them before measuring.

The predictor input convention remains target first and binder second. That
input order is separate from the chain order emitted by a predictor and does
not determine the mapping used for metrics.
"""

import csv
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Implementation revisions
# --------------------------------------------------------------------------
# The executor only checks that these are consistent across a cohort. It does
# not parse them. Bump the trailing integer whenever the numeric behaviour of
# the matching compute function changes, including a change to any constant it
# reads. Leaving the string alone after a behaviour change lets an old row and
# a new row sit in the same cohort while meaning different things.

IPSAE_IMPLEMENTATION_REVISION = "binder-metrics-ipsae-2"
DOCKQ_IMPLEMENTATION_REVISION = "binder-metrics-dockq-2"
SITE_SCORER_IMPLEMENTATION_REVISION = "binder-metrics-site-1"

# Where a hotspot list comes from. The target's site config names one of these
# in `hotspot_source` and the value reaches this module on `site_residue_map`.
# `compute_site_metrics` returns the resolved name as `site_hotspot_source`, so
# a reader of a scored row can tell which one produced `hotspot_recovery`.
#
#   explicit           the config carries the list, in the same CHAIN:NUMBER or
#                      CHAIN:START-END form as site_residues
#   site-fallback      no distinct list was supplied, so hotspot_recovery
#                      deliberately duplicates target_contact_recall
#   designed_complex   the list is measured on the designed complex this row is
#                      scored against, which is the replication path
#   published_epitope  the list is read out of a released design_summary.csv,
#                      which is the exact-replication path
#
# Only `explicit` states an epitope chosen before design. The other two are
# retrospective measurements on a finished structure. `epitope_residues` in the
# released table records where a design landed, not where it was aimed, and
# describing it as a design-time hotspot would be false.
# `ref/docs/PROVENANCE.md:25` records that most arms chose hotspots
# autonomously and left `prov_epitope_or_hotspot_spec` blank, so for most
# released designs no design-time specification exists to read.
HOTSPOT_SOURCES = ("explicit", "site-fallback", "designed_complex", "published_epitope")
HOTSPOT_RELATIONSHIPS = (
    "same-as-site",
    "subset-of-site",
    "superset-of-site",
    "partially-overlaps-site",
    "disjoint-from-site",
)

# The two columns `published_epitope` reads. Both are names in the released
# `tables/design_summary.csv`: `uuid` is column 1 and `epitope_residues` is
# column 64. A campaign whose table spells them differently overrides them
# through `published_epitope_key_column` and `published_epitope_residue_column`.
PUBLISHED_EPITOPE_KEY_COLUMN = "uuid"
PUBLISHED_EPITOPE_RESIDUE_COLUMN = "epitope_residues"

# What to do when the released chain letter is not this campaign's target chain.
# The released letter is the co-fold construct's and it varies per row inside one
# target: of the 90 released PD-L1 rows, 74 name chain B and 16 name chain A. One
# residue map is per target, so it cannot reconcile both.
#
#   as-written              use the letter the table wrote. A row on another
#                           chain raises. This is the default.
#   single-chain-to-target  when a row names exactly one chain, read it as the
#                           target chain and keep the residue numbers. A row
#                           naming more than one chain still raises, because
#                           then there is no single correspondence.
#
# The second is a relabelling, never a renumbering. The residue-identity check
# still compares each three-letter code against the structure, so a construct
# that does not share the released numbering still stops the job.
PUBLISHED_EPITOPE_CHAIN_POLICIES = ("as-written", "single-chain-to-target")

# --------------------------------------------------------------------------
# Sourced constants
# --------------------------------------------------------------------------

# PAE cutoff for ipSAE, in Angstroms. Predicted aligned error is a distance, so
# the campaign field is named `ipsae_interface_cutoff_angstrom` even though the
# value filters residue pairs by PAE rather than by a coordinate distance.
# Source: resolved campaign config `scoring.implementations`, and
# `ref/docs/INSILICO.md` section 4, "PAE cutoff 10 Angstrom".
IPSAE_INTERFACE_CUTOFF_ANGSTROM = 10.0

# Floor on d0 in the ipSAE normalisation. Dunbrack 2025 sets it at 1.0 because
# the Yang and Skolnick length fit was never tested below about 30 residues and
# the denominator blows up below 1.0.
IPSAE_D0_MINIMUM = 1.0

# C-alpha to C-alpha cutoff for a binder-target residue contact, in Angstroms.
# Source: `ref/docs/INSILICO.md` section 4, definition of `n_interface_contacts`.
CONTACT_CA_CUTOFF_ANGSTROM = 8.0

# Heavy-atom separation below which an interchain atom pair counts as a clash,
# in Angstroms. Source: `ref/docs/INSILICO.md` section 4, which says the
# per-run `has_clash` flag is not a uniform count and directs anyone needing a
# uniform definition to "recount inter-chain heavy-atom pairs closer than 2.2
# Angstrom from the model file".
CLASH_HEAVY_ATOM_CUTOFF_ANGSTROM = 2.2

# DockQ geometry. Source: Basu and Wallner 2016. The interface for fnat is any
# heavy-atom pair from the two molecules within 5 Angstroms. The interface for
# iRMS is redefined at 10 Angstroms. The two RMS values are scaled by
# d1 = 8.5 for the ligand RMSD and d2 = 1.5 for the interface RMSD, both fitted
# by grid search on 56,015 models.
DOCKQ_FNAT_CUTOFF_ANGSTROM = 5.0
DOCKQ_INTERFACE_CUTOFF_ANGSTROM = 10.0
DOCKQ_LRMS_SCALE_ANGSTROM = 8.5
DOCKQ_IRMS_SCALE_ANGSTROM = 1.5

# Backbone atom names used for the DockQ superpositions.
# TODO Confirm against the DockQ source. Basu and Wallner say "backbone" and do
# not name the atoms. N, CA, C, O is the CAPRI backbone convention and is what
# the DockQ program is understood to use. If the real set is N, CA, C only, the
# ligand and interface RMSD values shift and DOCKQ_IMPLEMENTATION_REVISION has
# to be bumped.
BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")

# The chain letters every caller in this campaign passes. Source: the installed
# Claude Science esmfold2 skill, which builds its input as
# ProteinInput(id="A", sequence=target_seq) then ProteinInput(id="B",
# sequence=binder_seq), and the resolved campaign config, where the target, the
# positive control and the negative control all carry target chain A and binder
# chain B. These drive a sanity check and never a default. Every function takes
# its chain identifiers as arguments.
CONVENTIONAL_TARGET_CHAIN_ID = "A"
CONVENTIONAL_BINDER_CHAIN_ID = "B"

# Elements excluded from a heavy-atom selection.
_HYDROGEN_ELEMENTS = frozenset({"H", "D", "T"})

# PDB record names that close an mmCIF `_atom_site` loop in practice. Neither is
# mmCIF. Both are written by tools in this campaign, and both appear only where
# the loop has ended, so the mmCIF reader treats them as terminators.
_PDB_END_RECORDS = frozenset({"END", "ENDMDL"})

# A trailing insertion code is part of the residue's identity. Author numbering
# distinguishes 52 from 52A, and dropping the code collapsed the two into one
# label, so the contacted set could be one entry short and a hotspot could be
# recovered by a different residue. A range carries no insertion code, because
# `A:52A-54` names no unambiguous span.
_RESIDUE_LABEL_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*:\s*(-?\d+)([A-Za-z]?)(?:\s*-\s*(-?\d+))?\s*$"
)

# The form the released `epitope_residues` cell uses, `<chain>:<residue><number>`
# such as `B:THR18`, stated in `ref/docs/INSILICO.md` section 8. It is not the
# form `site_residues` uses, so it gets its own pattern and its own parser.
_PUBLISHED_EPITOPE_PATTERN = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*([A-Za-z]{1,3})(-?\d+)\s*$")


class MetricInputError(ValueError):
    """The inputs cannot support a metric.

    Raised instead of returning a number, because a wrong metric on a scored
    row is worse than a job that stops and says why.
    """


class ChainOrderWarning(UserWarning):
    """The chain arguments are the campaign convention reversed."""


# Deliberately uncalled. Do not wire this into a metric function.
#
# It compares against the INPUT convention above, target A and binder B. The
# metric functions receive the mapping derived from a predictor's OUTPUT, and
# ESMFold2 writes the binder first, so a correct call for that arm passes
# target B and binder A. Calling this from compute_ipsae would therefore warn
# on every correct ESMFold2 scoring call and stay silent on the inverted one,
# which is worse than no warning at all.
#
# The real defence is derive_chain_mapping in binder_contract.py, which matches
# each chain against the two known sequences and refuses an equal-length
# count-only assignment. That runs before any metric and it has the sequences,
# which is what makes the check possible. This helper does not.
#
# It is kept because it correctly describes the input convention and a caller
# building predictor input can use it. An audit read its zero call sites as a
# defect on 2026-08-22. The zero is correct.
def _warn_if_chain_order_inverted(target_chain_id: str, binder_chain_id: str) -> None:
    """Warn when a caller building predictor INPUT reverses the convention.

    Not for use at the metric boundary. See the comment above for why, and use
    derive_chain_mapping in binder_contract.py to establish roles from a
    structure.

    An inverted mapping produces valid numbers with the two ipSAE directions
    swapped, so nothing downstream rejects it. This warns rather than raising,
    because a later target could legitimately use other letters.
    """
    if (
        target_chain_id == CONVENTIONAL_BINDER_CHAIN_ID
        and binder_chain_id == CONVENTIONAL_TARGET_CHAIN_ID
    ):
        warnings.warn(
            f"target_chain_id={target_chain_id!r} and binder_chain_id={binder_chain_id!r} "
            f"reverse the campaign convention, which is target "
            f"{CONVENTIONAL_TARGET_CHAIN_ID!r} and binder {CONVENTIONAL_BINDER_CHAIN_ID!r}. "
            "An inverted mapping swaps the two directional ipSAE values silently. "
            "Confirm both against the campaign config.",
            ChainOrderWarning,
            stacklevel=3,
        )


# --------------------------------------------------------------------------
# mmCIF reading
# --------------------------------------------------------------------------


class Residue:
    """One residue of one model, with its atoms."""

    __slots__ = (
        "auth_chain",
        "label_chain",
        "seq_id",
        "ins_code",
        "comp_id",
        "atom_names",
        "altlocs",
        "elements",
        "coords",
        "b_factors",
    )

    def __init__(self, auth_chain, label_chain, seq_id, ins_code, comp_id):
        self.auth_chain = auth_chain
        self.label_chain = label_chain
        self.seq_id = seq_id
        self.ins_code = ins_code
        self.comp_id = comp_id
        self.atom_names: list[str] = []
        self.altlocs: list[str] = []
        self.elements: list[str] = []
        self.coords: list[tuple[float, float, float]] = []
        self.b_factors: list[float] = []

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.auth_chain, self.seq_id, self.ins_code)

    @property
    def label(self) -> str:
        """The `CHAIN:NUMBER` form the campaign config uses for site residues."""
        return f"{self.auth_chain}:{self.seq_id}{self.ins_code}"

    def coordinate_array(self) -> np.ndarray:
        return np.asarray(self.coords, dtype=float).reshape(-1, 3)

    def heavy_atom_coords(self) -> np.ndarray:
        keep = [i for i, element in enumerate(self.elements) if element not in _HYDROGEN_ELEMENTS]
        if not keep:
            return np.zeros((0, 3), dtype=float)
        return self.coordinate_array()[keep]

    def named_atom_coords(self, names: Sequence[str]) -> np.ndarray:
        """Coordinates of the named atoms, in the order given.

        Returns an empty array when any name is missing, so a residue with a
        broken backbone drops out of a superposition instead of shifting it.
        """
        lookup = {name: index for index, name in enumerate(self.atom_names)}
        if any(name not in lookup for name in names):
            return np.zeros((0, 3), dtype=float)
        coords = self.coordinate_array()
        return coords[[lookup[name] for name in names]]

    def mean_b_factor(self) -> float:
        if not self.b_factors:
            return float("nan")
        return float(sum(self.b_factors) / len(self.b_factors))


class Structure:
    """The residues of one model, in file order.

    The four counts say what the parser did with the file's atom records, so a
    caller can count its inputs rather than trust a residue list. They add up:
    `atom_record_count` is the number of records read, and the three skip
    counts are the records that did not become atoms.

    `skipped_other_model` and `skipped_hetatm` are the two deliberate filters.
    `skipped_unreadable` is a record the parser could not read at all.
    `parse_cif_atoms` raises rather than leave that one above zero, so it is
    only ever non-zero on a structure that came from `parse_pdb_atoms`.
    """

    def __init__(
        self,
        residues: list[Residue],
        *,
        atom_record_count: int = 0,
        skipped_other_model: int = 0,
        skipped_hetatm: int = 0,
        skipped_unreadable: int = 0,
    ):
        self.residues = residues
        self.atom_record_count = atom_record_count
        self.skipped_other_model = skipped_other_model
        self.skipped_hetatm = skipped_hetatm
        self.skipped_unreadable = skipped_unreadable

    def __len__(self) -> int:
        return len(self.residues)

    def chain_ids(self) -> list[str]:
        seen: list[str] = []
        for residue in self.residues:
            for chain in (residue.auth_chain, residue.label_chain):
                if chain and chain not in seen:
                    seen.append(chain)
        return seen

    def chain_indices(self, chain_id: str) -> list[int]:
        """Positions of one chain's residues in the flat residue list.

        Matches the author chain identifier first, then the label chain
        identifier. The campaign config names author chains, and some writers
        emit only label chains.
        """
        auth = [i for i, residue in enumerate(self.residues) if residue.auth_chain == chain_id]
        if auth:
            return auth
        label = [i for i, residue in enumerate(self.residues) if residue.label_chain == chain_id]
        if label:
            return label
        raise MetricInputError(
            f"chain {chain_id!r} is not in the structure. Chains present: {self.chain_ids()}"
        )

    def chain_residues(self, chain_id: str) -> list[Residue]:
        return [self.residues[i] for i in self.chain_indices(chain_id)]


def _cif_text(value: Any, argument: str) -> str:
    """Return mmCIF text from a path or from text already in memory.

    A `Path` is read from disk. A `str` is treated as the file's contents,
    because the ESMFold2 arms hand over `prediction.complex.to_mmcif()` without
    ever writing it.
    """
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8")
    if isinstance(value, os.PathLike):
        return Path(value).read_text(encoding="utf-8")
    if isinstance(value, str):
        if "\n" not in value:
            raise MetricInputError(
                f"{argument} is a single line of text. Pass mmCIF contents as str, "
                "or pass a pathlib.Path to read a file."
            )
        return value
    raise MetricInputError(f"{argument} must be mmCIF text, bytes, or a pathlib.Path")


def _split_cif_row(line: str) -> list[str]:
    """Split one mmCIF data row, honouring single and double quotes."""
    tokens: list[str] = []
    index = 0
    length = len(line)
    while index < length:
        char = line[index]
        if char.isspace():
            index += 1
            continue
        if char in "'\"":
            quote = char
            index += 1
            start = index
            while index < length and not (
                line[index] == quote and (index + 1 >= length or line[index + 1].isspace())
            ):
                index += 1
            tokens.append(line[start:index])
            index += 1
            continue
        start = index
        while index < length and not line[index].isspace():
            index += 1
        tokens.append(line[start:index])
    return tokens


def _clean(value: str) -> str:
    return "" if value in (".", "?") else value


def _element_symbol(atom_name: str, element: str = "") -> str:
    """Return an uppercase element symbol, with an atom-name fallback."""
    cleaned = element.strip()
    if cleaned:
        return cleaned.upper()
    for character in atom_name.strip():
        if character.isalpha():
            return character.upper()
    return ""


def _iter_atom_site_rows(text: str, argument: str) -> Iterator[dict[str, str]]:
    """Yield the `_atom_site` loop one row at a time, across wrapped lines.

    An mmCIF row is a count of values, not a line. The format lets a row
    continue onto the following lines, and it lets a semicolon-delimited text
    value run over several lines with the semicolons in column one. OpenFold3
    NIM writes wrapped rows. Reading one row per physical line and dropping a
    line whose token count missed therefore parsed a 186-residue complex as 12
    residues and raised nothing, which is the measurement
    `notes/2026-09-11-session-review/DEFECTS.md` row 63 records.

    The loop is read as a stream of values and cut into a row every time the
    column count is reached. A loop that ends with a part-built row raises,
    because those leftover values name no column and dropping them would hand
    back a short atom set that looks complete.

    A semicolon-delimited value is returned with its lines joined by newlines.
    No `_atom_site` column this module reads is ever written that way. It is
    read anyway because ignoring it would feed its words into the value stream
    and shift every row after it.
    """
    fields: list[str] = []
    in_loop_header = False
    in_rows = False
    pending: list[str] = []
    text_field: list[str] | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not in_rows:
            if stripped == "loop_":
                fields = []
                in_loop_header = True
                continue
            if in_loop_header and stripped.startswith("_atom_site."):
                fields.append(stripped.split(".", 1)[1].split()[0])
                continue
            if in_loop_header and stripped.startswith("_"):
                # A loop of something else. Wait for the next loop_.
                fields = []
                in_loop_header = False
                continue
            if in_loop_header and fields and stripped:
                in_loop_header = False
                in_rows = True
                # Fall through and read this line as the first data row.
            else:
                continue

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
        elif stripped.startswith(("#", "_", "loop_", "data_", "save_")):
            break
        elif not pending and stripped in _PDB_END_RECORDS:
            # Not mmCIF. Several writers in this campaign close a .cif with the
            # PDB END record, and ENDMDL closes a model the same way, so both
            # sit exactly where the loop terminator belongs. Only honoured at a
            # row boundary, where the next value would be the first column and
            # can never be one of these.
            break
        else:
            pending.extend(_split_cif_row(line))

        while len(pending) >= len(fields):
            yield dict(zip(fields, pending[: len(fields)]))
            del pending[: len(fields)]

    if pending:
        raise MetricInputError(
            f"{argument} ends its _atom_site loop with {len(pending)} leftover value(s) "
            f"against {len(fields)} columns, so the last row is incomplete. The file is "
            "truncated or its column list disagrees with its rows."
        )


def parse_cif_atoms(
    source: Any,
    *,
    argument: str = "cif",
    include_hetatm: bool = False,
) -> Structure:
    """Parse the `_atom_site` loop of an mmCIF file into residues.

    Only the first model is kept. HETATM records are dropped by default,
    because the published metric masks exclude ion, ligand and nucleic tokens.

    Rows are cut by column count rather than by line, so a row the writer
    wrapped across lines still parses. See `_iter_atom_site_rows`.

    Every atom record is accounted for. A record this parser cannot read raises
    instead of vanishing, because a short atom set produces contacts, buried
    surface and a chain mapping that all look right. The two deliberate
    filters, the model number and the HETATM switch, are counted on the
    returned `Structure` rather than raising.
    """
    text = _cif_text(source, argument)
    residues: list[Residue] = []
    by_key: dict[tuple, Residue] = {}
    atom_records = 0
    other_model = 0
    hetatm = 0
    unreadable = 0
    first_unreadable = ""

    def unreadable_record(reason: str) -> None:
        nonlocal unreadable, first_unreadable
        unreadable += 1
        if not first_unreadable:
            first_unreadable = f"record {atom_records} {reason}"

    for row in _iter_atom_site_rows(text, argument):
        atom_records += 1
        model = _clean(row.get("pdbx_PDB_model_num", "")) or "1"
        if model != "1":
            other_model += 1
            continue
        group = _clean(row.get("group_PDB", "ATOM")) or "ATOM"
        if group == "HETATM" and not include_hetatm:
            hetatm += 1
            continue
        atom_name = _clean(row.get("label_atom_id", "")) or _clean(row.get("auth_atom_id", ""))
        element = _element_symbol(atom_name, _clean(row.get("type_symbol", "")))
        altloc = _clean(row.get("label_alt_id", ""))
        label_chain = _clean(row.get("label_asym_id", ""))
        auth_chain = _clean(row.get("auth_asym_id", "")) or label_chain
        seq_text = _clean(row.get("auth_seq_id", "")) or _clean(row.get("label_seq_id", ""))
        if not seq_text:
            unreadable_record("carries neither auth_seq_id nor label_seq_id")
            continue
        try:
            seq_id = int(seq_text)
        except ValueError:
            unreadable_record(f"has a non-integer residue number {seq_text!r}")
            continue
        ins_code = _clean(row.get("pdbx_PDB_ins_code", ""))
        comp_id = _clean(row.get("auth_comp_id", "")) or _clean(row.get("label_comp_id", ""))
        try:
            x = float(row["Cartn_x"])
            y = float(row["Cartn_y"])
            z = float(row["Cartn_z"])
        except (KeyError, ValueError):
            unreadable_record("has no readable Cartn_x, Cartn_y and Cartn_z")
            continue
        try:
            b_factor = float(_clean(row.get("B_iso_or_equiv", "")) or "nan")
        except ValueError:
            b_factor = float("nan")

        key = (auth_chain, label_chain, seq_id, ins_code)
        residue = by_key.get(key)
        if residue is None:
            residue = Residue(auth_chain, label_chain, seq_id, ins_code, comp_id)
            by_key[key] = residue
            residues.append(residue)
        residue.atom_names.append(atom_name)
        residue.altlocs.append(altloc)
        residue.elements.append(element)
        residue.coords.append((x, y, z))
        residue.b_factors.append(b_factor)

    if unreadable:
        raise MetricInputError(
            f"{argument} carries {unreadable} unreadable _atom_site record(s) out of "
            f"{atom_records}. The first is: {first_unreadable}. Every metric downstream "
            "would have been computed on the records that did parse."
        )
    if not residues:
        raise MetricInputError(f"{argument} has no usable _atom_site records")
    return Structure(
        residues,
        atom_record_count=atom_records,
        skipped_other_model=other_model,
        skipped_hetatm=hetatm,
    )


def parse_pdb_atoms(
    source: Any,
    *,
    argument: str = "pdb",
    include_hetatm: bool = False,
) -> Structure:
    """Parse fixed-column PDB ATOM records into the metric `Structure`.

    The parser keeps records in MODEL 1 and ignores later models. A file with
    no MODEL records is treated as one model. HETATM records are dropped by
    default, matching `parse_cif_atoms` and its `include_hetatm` switch.

    PDB chain ID is the single chain column at column 22, so this parser assigns
    it to both `auth_chain` and `label_chain`. Residue sequence number comes
    from columns 23 to 26, insertion code from column 27, alternate location
    from column 17, and the element symbol from columns 77 to 78. Coordinates,
    occupancy, and B factor use columns 31 to 54, 55 to 60, and 61 to 66.
    These are one-based PDB columns and are read with zero-based slices.

    PDB is fixed-column, so a record cannot wrap the way an mmCIF row can and
    this parser has no equivalent of the wrapped-row defect. It still counts
    what it did with every record onto the returned `Structure`, including the
    records it could not read, so a caller can check the total rather than
    trust the residue list.
    """
    text = _cif_text(source, argument)
    residues: list[Residue] = []
    by_key: dict[tuple[str, str, int, str], Residue] = {}
    has_model_records = False
    in_model_one = True
    atom_records = 0
    other_model = 0
    hetatm = 0
    unreadable = 0

    for line in text.splitlines():
        record = line[:6].strip()
        if record == "MODEL":
            has_model_records = True
            try:
                model_number = int(line[10:14].strip())
            except ValueError:
                model_number = -1
            in_model_one = model_number == 1
            continue
        if record == "ENDMDL":
            if has_model_records and in_model_one:
                break
            in_model_one = False
            continue
        if record not in {"ATOM", "HETATM"}:
            continue
        atom_records += 1
        if has_model_records and not in_model_one:
            other_model += 1
            continue
        if record == "HETATM" and not include_hetatm:
            hetatm += 1
            continue
        try:
            seq_id = int(line[22:26].strip())
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except (ValueError, IndexError):
            unreadable += 1
            continue

        atom_name = line[12:16].strip()
        if not atom_name:
            unreadable += 1
            continue
        chain = line[21:22].strip()
        ins_code = line[26:27].strip()
        comp_id = line[17:20].strip()
        altloc = line[16:17].strip()
        element = _element_symbol(atom_name, line[76:78] if len(line) >= 78 else "")
        try:
            b_factor = float(line[60:66].strip())
        except ValueError:
            b_factor = float("nan")

        key = (chain, chain, seq_id, ins_code)
        residue = by_key.get(key)
        if residue is None:
            residue = Residue(chain, chain, seq_id, ins_code, comp_id)
            by_key[key] = residue
            residues.append(residue)
        residue.atom_names.append(atom_name)
        residue.altlocs.append(altloc)
        residue.elements.append(element)
        residue.coords.append((x, y, z))
        residue.b_factors.append(b_factor)

    if not residues:
        raise MetricInputError(f"{argument} has no usable PDB ATOM records")
    return Structure(
        residues,
        atom_record_count=atom_records,
        skipped_other_model=other_model,
        skipped_hetatm=hetatm,
        skipped_unreadable=unreadable,
    )


def _looks_like_pdb(text: str) -> bool:
    """Identify fixed-column PDB atom or model records in in-memory text."""
    for line in text.splitlines():
        if line.startswith("MODEL"):
            return True
        if line[:6].strip() not in {"ATOM", "HETATM"}:
            continue
        try:
            float(line[30:38].strip())
            float(line[38:46].strip())
            float(line[46:54].strip())
        except (ValueError, IndexError):
            continue
        return True
    return False


def parse_structure_atoms(
    source: Any,
    *,
    argument: str = "structure",
    include_hetatm: bool = False,
) -> Structure:
    """Dispatch one metric structure input to the PDB or mmCIF atom parser.

    A `.pdb` path selects the fixed-column PDB parser. Other paths and in-memory
    text use the PDB record shape when present, then fall through to mmCIF.
    The PDB parser maps its one chain column to both author and label chains.
    """
    if isinstance(source, os.PathLike) and Path(source).suffix.lower() == ".pdb":
        return parse_pdb_atoms(source, argument=argument, include_hetatm=include_hetatm)
    text = _cif_text(source, argument)
    if _looks_like_pdb(text):
        return parse_pdb_atoms(text, argument=argument, include_hetatm=include_hetatm)
    return parse_cif_atoms(text, argument=argument, include_hetatm=include_hetatm)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def _kabsch(
    mobile: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Superpose `mobile` onto `reference` and return the fit.

    Returns the rotation matrix, the mobile centroid, the reference centroid,
    and the RMSD after the fit. Apply the fit with `_apply_fit`.
    """
    if mobile.shape != reference.shape or mobile.shape[0] < 3:
        raise MetricInputError(
            f"superposition needs at least 3 matched atoms, got {mobile.shape[0]} and {reference.shape[0]}"
        )
    mobile_centroid = mobile.mean(axis=0)
    reference_centroid = reference.mean(axis=0)
    p = mobile - mobile_centroid
    q = reference - reference_centroid
    covariance = p.T @ q
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag(np.array([1.0, 1.0, sign if sign != 0.0 else 1.0]))
    rotation = vt.T @ correction @ u.T
    fitted = (rotation @ p.T).T
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - q) ** 2, axis=1))))
    return rotation, mobile_centroid, reference_centroid, rmsd


def _apply_fit(
    coords: np.ndarray, fit: tuple[np.ndarray, np.ndarray, np.ndarray, float]
) -> np.ndarray:
    rotation, mobile_centroid, reference_centroid, _ = fit
    if coords.shape[0] == 0:
        return coords
    return (rotation @ (coords - mobile_centroid).T).T + reference_centroid


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _pair_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """All pairwise distances between two coordinate blocks.

    Uses the squared-norm expansion so the working array holds one value per
    pair rather than three. A Cas9-sized target against a binder is several
    million pairs, and the difference there is hundreds of megabytes.
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    squared = (
        np.einsum("ij,ij->i", a, a)[:, None]
        + np.einsum("ij,ij->i", b, b)[None, :]
        - 2.0 * (a @ b.T)
    )
    return np.sqrt(np.maximum(squared, 0.0))


def _stack(blocks: Sequence[np.ndarray]) -> np.ndarray:
    filled = [block for block in blocks if block.shape[0]]
    if not filled:
        return np.zeros((0, 3), dtype=float)
    return np.concatenate(filled)


# --------------------------------------------------------------------------
# ipSAE
# --------------------------------------------------------------------------


def _d0(n0: np.ndarray) -> np.ndarray:
    """The TM-score length normalisation, floored at 1.0.

    Source: Dunbrack 2025. The Yang and Skolnick fit returns a negative number
    below 19 residues, so the paper clamps d0 at 1.0, which corresponds to a
    chain of roughly 27 residues.
    """
    values = np.asarray(n0, dtype=float)
    fitted = 1.24 * np.cbrt(values - 15.0) - 1.8
    return np.maximum(fitted, IPSAE_D0_MINIMUM)


def _directional_ipsae(block: np.ndarray, cutoff: float) -> float:
    """ipSAE for one direction, from a PAE block of aligned rows by scored columns.

    Implements Dunbrack 2025 Equation 14 in the d0res variant. For each aligned
    residue i, count the scored residues j whose PAE falls below the cutoff,
    take d0 from that count, average the TM term over exactly those j, and
    return the maximum over i.

    Returns 0.0 when no residue pair passes the cutoff. The published co-fold
    table carries many exact zeros for that case.
    """
    if block.size == 0:
        return 0.0
    mask = block < cutoff
    n0 = mask.sum(axis=1)
    keep = n0 > 0
    if not bool(np.any(keep)):
        return 0.0
    counts = n0[keep].astype(float)
    d0 = _d0(counts)
    rows = block[keep]
    row_mask = mask[keep]
    term = 1.0 / (1.0 + (rows / d0[:, None]) ** 2)
    per_residue = np.sum(term * row_mask, axis=1) / counts
    return float(np.max(per_residue))


def _plddt_values(residues: Sequence[Residue], scale: str) -> np.ndarray:
    """Per-residue pLDDT on the 0 to 100 scale.

    Every predictor here writes pLDDT into the B-factor column. The scale is
    the trap. The executor accepts any value between 0 and 100, so a 0-to-1
    pLDDT written straight through passes validation while being wrong by a
    factor of a hundred.

    With `scale="auto"` this rescales when every value is at or below 1.0. A
    real 0-to-100 pLDDT is never that low across a whole structure.
    """
    values = np.asarray([residue.mean_b_factor() for residue in residues], dtype=float)
    if scale == "0-100":
        return values
    if scale == "0-1":
        return values * 100.0
    if scale != "auto":
        raise MetricInputError(f"plddt_scale must be auto, 0-1 or 0-100, got {scale!r}")
    finite = values[np.isfinite(values)]
    if finite.size and float(np.max(finite)) <= 1.0:
        return values * 100.0
    return values


def compute_ipsae(
    pae: Any,
    complex_cif: Any,
    target_chain_id: str,
    binder_chain_id: str,
    *,
    pae_cutoff_angstrom: float = IPSAE_INTERFACE_CUTOFF_ANGSTROM,
    pae_orientation: str = "aligned_rows",
    plddt_scale: str = "auto",
    contact_ca_cutoff_angstrom: float = CONTACT_CA_CUTOFF_ANGSTROM,
    clash_cutoff_angstrom: float = CLASH_HEAVY_ATOM_CUTOFF_ANGSTROM,
) -> dict[str, Any]:
    """ipSAE in both directions, plus the interface summaries on the same row.

    Returns `ipsae_target_to_binder`, `ipsae_binder_to_target`, `ipsae_min`,
    `interface_pae`, `interface_plddt`, `contact_count` and `clash_count`.

    `pae` is the square PAE matrix as a list of lists, already in memory, in
    the structure's own residue order. `complex_cif` accepts mmCIF or PDB text
    and paths.

    ipSAE is directional and the two values are different numbers.
    `ipsae_target_to_binder` aligns on target residues and scores over binder
    residues, following Dunbrack's ipSAE(A to B) where chain A supplies the
    aligned residue. `ipsae_min` is the smaller of the two, which is what the
    campaign ranks on.

    Getting `target_chain_id` and `binder_chain_id` the wrong way round swaps
    the two directional values and changes nothing else, so nothing downstream
    catches it. Read both from the campaign config.

    TODO Confirm the PAE token convention on a real run of each arm. This
    implementation reads `pae[i][j]` as the error at scored residue j when the
    structures are aligned on residue i, which is the AlphaFold2 JSON
    convention. Pass `pae_orientation="aligned_columns"` to transpose. The
    correct setting per arm has to come from a prediction whose two directional
    values are known to differ, because a symmetric test matrix cannot tell the
    two conventions apart.
    """
    if target_chain_id == binder_chain_id:
        raise MetricInputError("target_chain_id and binder_chain_id must be different chains")

    structure = parse_structure_atoms(complex_cif, argument="complex_cif")
    matrix = np.asarray(pae, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise MetricInputError(f"pae must be a square matrix, got shape {matrix.shape}")
    if matrix.shape[0] != len(structure):
        raise MetricInputError(
            f"pae is {matrix.shape[0]} by {matrix.shape[0]} and the structure has "
            f"{len(structure)} residues. The PAE matrix and the residue order have to match. "
            "A structure carrying ligand, ion or nucleic tokens needs a token map instead."
        )
    if pae_orientation == "aligned_columns":
        matrix = matrix.T
    elif pae_orientation != "aligned_rows":
        raise MetricInputError(
            f"pae_orientation must be aligned_rows or aligned_columns, got {pae_orientation!r}"
        )

    target_index = structure.chain_indices(target_chain_id)
    binder_index = structure.chain_indices(binder_chain_id)

    target_to_binder_block = matrix[np.ix_(target_index, binder_index)]
    binder_to_target_block = matrix[np.ix_(binder_index, target_index)]

    ipsae_target_to_binder = _directional_ipsae(target_to_binder_block, float(pae_cutoff_angstrom))
    ipsae_binder_to_target = _directional_ipsae(binder_to_target_block, float(pae_cutoff_angstrom))

    # Mean PAE over every binder-target residue pair in both matrix directions,
    # with no distance filter. Source: `ref/docs/INSILICO.md` section 4,
    # definition of `pae_interface_mean`. Averaging both directions makes this
    # value independent of the PAE orientation question above.
    if target_to_binder_block.size:
        interface_pae = float(
            (target_to_binder_block.sum() + binder_to_target_block.sum())
            / (target_to_binder_block.size + binder_to_target_block.size)
        )
    else:
        interface_pae = 0.0

    target_residues = [structure.residues[i] for i in target_index]
    binder_residues = [structure.residues[i] for i in binder_index]

    # Contacts are C-alpha to C-alpha, so a residue with no C-alpha drops out.
    target_ca = [residue.named_atom_coords(("CA",)) for residue in target_residues]
    binder_ca = [residue.named_atom_coords(("CA",)) for residue in binder_residues]
    target_with_ca = np.flatnonzero([block.shape[0] == 1 for block in target_ca])
    binder_with_ca = np.flatnonzero([block.shape[0] == 1 for block in binder_ca])

    if target_with_ca.size and binder_with_ca.size:
        ca_distances = _pair_distances(
            np.concatenate([target_ca[i] for i in target_with_ca]),
            np.concatenate([binder_ca[i] for i in binder_with_ca]),
        )
        contact_mask = ca_distances < float(contact_ca_cutoff_angstrom)
        contact_count = int(contact_mask.sum())
        contacted_target = target_with_ca[np.flatnonzero(contact_mask.any(axis=1))]
        contacted_binder = binder_with_ca[np.flatnonzero(contact_mask.any(axis=0))]
    else:
        contact_count = 0
        contacted_target = np.zeros(0, dtype=int)
        contacted_binder = np.zeros(0, dtype=int)

    # Mean pLDDT over the residues that take part in a binder-target contact.
    # TODO This is a local definition. The published tables carry separate
    # `plddt_binder` and `plddt_target` columns and no combined interface
    # value, so there is nothing to reproduce. The contract needs one number in
    # `interface_plddt`, and this is the interface-restricted mean of both
    # chains. Replace it if a campaign document later defines the field.
    interface_residues = [target_residues[i] for i in contacted_target]
    interface_residues += [binder_residues[i] for i in contacted_binder]
    if interface_residues:
        plddt = _plddt_values(interface_residues, plddt_scale)
        finite = plddt[np.isfinite(plddt)]
        interface_plddt = float(np.mean(finite)) if finite.size else 0.0
    else:
        interface_plddt = 0.0
    interface_plddt = float(min(max(interface_plddt, 0.0), 100.0))

    heavy_distances = _pair_distances(
        _stack([residue.heavy_atom_coords() for residue in target_residues]),
        _stack([residue.heavy_atom_coords() for residue in binder_residues]),
    )
    clash_count = (
        int((heavy_distances < float(clash_cutoff_angstrom)).sum()) if heavy_distances.size else 0
    )

    return {
        "ipsae_target_to_binder": ipsae_target_to_binder,
        "ipsae_binder_to_target": ipsae_binder_to_target,
        "ipsae_min": float(min(ipsae_target_to_binder, ipsae_binder_to_target)),
        "interface_pae": interface_pae,
        "interface_plddt": interface_plddt,
        "contact_count": contact_count,
        "clash_count": clash_count,
    }


# --------------------------------------------------------------------------
# DockQ
# --------------------------------------------------------------------------


def _residue_map(residues: Sequence[Residue]) -> dict[tuple[int, str], Residue]:
    return {(residue.seq_id, residue.ins_code): residue for residue in residues}


def _contact_pairs(
    target_residues: Sequence[Residue],
    binder_residues: Sequence[Residue],
    cutoff: float,
) -> set[tuple[int, int]]:
    """Residue pairs whose target heavy atoms reach binder backbone atoms.

    The atom selection follows `ref/docs/INSILICO.md` section 4 for sc_DockQ:
    binder backbone atoms against target heavy atoms. Indices are positions in
    the two sequences passed in.
    """
    pairs: set[tuple[int, int]] = set()
    blocks = [
        (index, residue.named_atom_coords(BACKBONE_ATOM_NAMES))
        for index, residue in enumerate(binder_residues)
    ]
    blocks = [(index, block) for index, block in blocks if block.shape[0]]
    if not blocks:
        return pairs
    binder_coords = np.concatenate([block for _, block in blocks])
    binder_owner = np.concatenate(
        [np.full(block.shape[0], index, dtype=int) for index, block in blocks]
    )
    for target_position, residue in enumerate(target_residues):
        heavy = residue.heavy_atom_coords()
        if heavy.shape[0] == 0:
            continue
        hit = np.flatnonzero((_pair_distances(heavy, binder_coords) < cutoff).any(axis=0))
        for binder_position in np.unique(binder_owner[hit]):
            pairs.add((target_position, int(binder_position)))
    return pairs


def compute_dockq(
    predicted_cif: Any,
    reference_cif: Any,
    chain_mapping: Mapping[str, str],
    *,
    reference_chain_mapping: Mapping[str, str] | None = None,
    fnat_cutoff_angstrom: float = DOCKQ_FNAT_CUTOFF_ANGSTROM,
    interface_cutoff_angstrom: float = DOCKQ_INTERFACE_CUTOFF_ANGSTROM,
) -> dict[str, Any]:
    """sc_DockQ of a prediction against a reference pose.

    Returns `sc_dockq`, `dockq`, `fnat`, `interface_rmsd`, `ligand_rmsd`,
    `aligned_target_residue_count`, `target_alignment_rmsd` and
    `mapping_status`.

    `chain_mapping` names the predicted structure's target and binder chains.
    `reference_chain_mapping` names the reference structure's chains. It is
    optional for compatibility with callers whose structures share chain ids.

    The score follows Basu and Wallner 2016:

        DockQ = (fnat + 1 / (1 + (LRMS / 8.5) ** 2) + 1 / (1 + (iRMS / 1.5) ** 2)) / 3

    The target is the receptor and the binder is the ligand. Residues are
    matched between the two structures by chain and residue number, which is
    the right correspondence here because the reference is this design's own
    pose and carries the same sequence. A residue whose three-letter code
    disagrees between the two files is dropped rather than paired.

    `mapping_status` is `"ok"` on success. Any other value means the row cannot
    be scored, and the executor rejects a scored row whose status is not
    `"ok"`. The caller should write a failed row in that case.

    TODO The contract carries `sc_dockq` and `dockq` as separate fields and
    nothing read here distinguishes them for a single reference. Both are
    returned as the same value. If `dockq` is ever meant to be plain DockQ over
    all heavy atoms on both sides, that is a second computation and a bump of
    DOCKQ_IMPLEMENTATION_REVISION.

    TODO Multimeric targets need the union treatment. The published sc_DockQ
    treats all target protein chains as one receptor and picks the copy
    assignment that maximises DockQ. This function reads one target chain,
    which covers a monomeric target and nothing else.
    """
    target_chain = str(chain_mapping["target"])
    binder_chain = str(chain_mapping["binder"])
    reference_mapping = reference_chain_mapping or chain_mapping
    reference_target_chain = str(reference_mapping["target"])
    reference_binder_chain = str(reference_mapping["binder"])
    if target_chain == binder_chain:
        raise MetricInputError("chain_mapping target and binder must be different chains")
    if reference_target_chain == reference_binder_chain:
        raise MetricInputError(
            "reference_chain_mapping target and binder must be different chains"
        )

    predicted = parse_structure_atoms(predicted_cif, argument="predicted_cif")
    reference = parse_structure_atoms(reference_cif, argument="reference_cif")

    failure = {
        "sc_dockq": 0.0,
        "dockq": 0.0,
        "fnat": 0.0,
        "interface_rmsd": 0.0,
        "ligand_rmsd": 0.0,
        "aligned_target_residue_count": 0,
        "target_alignment_rmsd": 0.0,
    }

    predicted_target = _residue_map(predicted.chain_residues(target_chain))
    predicted_binder = _residue_map(predicted.chain_residues(binder_chain))
    reference_target = _residue_map(reference.chain_residues(reference_target_chain))
    reference_binder = _residue_map(reference.chain_residues(reference_binder_chain))

    def paired(left: dict, right: dict) -> list[tuple[Residue, Residue]]:
        out: list[tuple[Residue, Residue]] = []
        for key in sorted(set(left) & set(right)):
            a, b = left[key], right[key]
            if a.comp_id and b.comp_id and a.comp_id != b.comp_id:
                continue
            out.append((a, b))
        return out

    target_pairs = paired(predicted_target, reference_target)
    binder_pairs = paired(predicted_binder, reference_binder)
    if not target_pairs:
        return {**failure, "mapping_status": "no-target-residue-correspondence"}
    if not binder_pairs:
        return {**failure, "mapping_status": "no-binder-residue-correspondence"}

    def backbone_pairs(
        pairs: Sequence[tuple[Residue, Residue]]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        left: list[np.ndarray] = []
        right: list[np.ndarray] = []
        for predicted_residue, reference_residue in pairs:
            a = predicted_residue.named_atom_coords(BACKBONE_ATOM_NAMES)
            b = reference_residue.named_atom_coords(BACKBONE_ATOM_NAMES)
            if a.shape[0] and b.shape[0]:
                left.append(a)
                right.append(b)
        return left, right

    # Superpose the prediction's target backbone onto the reference target.
    predicted_target_backbone, reference_target_backbone = backbone_pairs(target_pairs)
    aligned_target_residues = len(predicted_target_backbone)
    if aligned_target_residues == 0:
        return {**failure, "mapping_status": "no-target-backbone"}

    receptor_fit = _kabsch(
        np.concatenate(predicted_target_backbone),
        np.concatenate(reference_target_backbone),
    )
    target_alignment_rmsd = float(receptor_fit[3])

    # Ligand RMSD: the binder backbone after the receptor superposition.
    predicted_binder_backbone, reference_binder_backbone = backbone_pairs(binder_pairs)
    if not predicted_binder_backbone:
        return {**failure, "mapping_status": "no-binder-backbone"}
    ligand_rmsd = _rmsd(
        _apply_fit(np.concatenate(predicted_binder_backbone), receptor_fit),
        np.concatenate(reference_binder_backbone),
    )

    # fnat over the corresponded residues, so a contact only one structure can
    # express never counts against the other.
    predicted_target_list = [pair[0] for pair in target_pairs]
    reference_target_list = [pair[1] for pair in target_pairs]
    predicted_binder_list = [pair[0] for pair in binder_pairs]
    reference_binder_list = [pair[1] for pair in binder_pairs]

    native_contacts = _contact_pairs(
        reference_target_list, reference_binder_list, float(fnat_cutoff_angstrom)
    )
    if not native_contacts:
        return {**failure, "mapping_status": "no-reference-interface"}
    model_contacts = _contact_pairs(
        predicted_target_list, predicted_binder_list, float(fnat_cutoff_angstrom)
    )
    fnat = len(native_contacts & model_contacts) / len(native_contacts)

    # Interface RMSD: the reference interface redefined at the wider cutoff,
    # then the backbone of those residues superposed on their equivalents.
    wide_contacts = _contact_pairs(
        reference_target_list, reference_binder_list, float(interface_cutoff_angstrom)
    )
    interface_predicted: list[np.ndarray] = []
    interface_reference: list[np.ndarray] = []
    for position in sorted({pair[0] for pair in wide_contacts}):
        a = predicted_target_list[position].named_atom_coords(BACKBONE_ATOM_NAMES)
        b = reference_target_list[position].named_atom_coords(BACKBONE_ATOM_NAMES)
        if a.shape[0] and b.shape[0]:
            interface_predicted.append(a)
            interface_reference.append(b)
    for position in sorted({pair[1] for pair in wide_contacts}):
        a = predicted_binder_list[position].named_atom_coords(BACKBONE_ATOM_NAMES)
        b = reference_binder_list[position].named_atom_coords(BACKBONE_ATOM_NAMES)
        if a.shape[0] and b.shape[0]:
            interface_predicted.append(a)
            interface_reference.append(b)
    if not interface_predicted:
        return {**failure, "mapping_status": "no-interface-backbone"}
    interface_rmsd = float(
        _kabsch(np.concatenate(interface_predicted), np.concatenate(interface_reference))[3]
    )

    scaled_ligand = 1.0 / (1.0 + (ligand_rmsd / DOCKQ_LRMS_SCALE_ANGSTROM) ** 2)
    scaled_interface = 1.0 / (1.0 + (interface_rmsd / DOCKQ_IRMS_SCALE_ANGSTROM) ** 2)
    dockq = float(min(max((fnat + scaled_ligand + scaled_interface) / 3.0, 0.0), 1.0))

    return {
        "sc_dockq": dockq,
        "dockq": dockq,
        "fnat": float(min(max(fnat, 0.0), 1.0)),
        "interface_rmsd": float(interface_rmsd),
        "ligand_rmsd": float(ligand_rmsd),
        "aligned_target_residue_count": int(aligned_target_residues),
        "target_alignment_rmsd": target_alignment_rmsd,
        "mapping_status": "ok",
    }


# --------------------------------------------------------------------------
# Site metrics
# --------------------------------------------------------------------------


def _expand_residue_labels(entries: Iterable[str], chain_id: str) -> set[str]:
    """Expand `A:1`, `A:1A` and `A:1-3` into a set of `CHAIN:NUMBER` labels.

    Only entries on `chain_id` are kept, because the site metrics score the
    target chain.
    """
    labels: set[str] = set()
    for entry in entries:
        match = _RESIDUE_LABEL_PATTERN.match(str(entry))
        if match is None:
            raise MetricInputError(f"residue label {entry!r} is not CHAIN:NUMBER or CHAIN:START-END")
        chain, start = match.group(1), int(match.group(2))
        ins_code, end = match.group(3), match.group(4)
        if chain != chain_id:
            continue
        if end is not None:
            if ins_code:
                raise MetricInputError(
                    f"residue range {entry!r} may not carry an insertion code"
                )
            last = int(end)
            if last < start:
                raise MetricInputError(f"residue range {entry!r} runs backwards")
            labels.update(f"{chain}:{number}" for number in range(start, last + 1))
            continue
        labels.add(f"{chain}:{start}{ins_code}")
    return labels


def _contacted_target_residues(
    structure: Structure,
    target_chain_id: str,
    binder_chain_id: str,
    cutoff: float,
) -> set[str]:
    """Target residues with a heavy atom within `cutoff` of a binder heavy atom.

    Returns `CHAIN:NUMBER` labels. This is the rule the released dataset used
    for `epitope_residues`, stated in `ref/docs/INSILICO.md` section 8 as target
    residues with any heavy atom within 5 Angstrom of a binder heavy atom. The
    cutoff is an argument rather than a constant because the campaign carries
    its own `site.contact_cutoff_angstrom`, and the value that produced a row
    belongs on that row. `compute_site_metrics` returns it as
    `site_contact_cutoff_angstrom`.
    """
    binder_coords = _stack(
        [residue.heavy_atom_coords() for residue in structure.chain_residues(binder_chain_id)]
    )
    contacted: set[str] = set()
    if not binder_coords.shape[0]:
        return contacted
    for residue in structure.chain_residues(target_chain_id):
        heavy = residue.heavy_atom_coords()
        if heavy.shape[0] == 0:
            continue
        if bool((_pair_distances(heavy, binder_coords) < cutoff).any()):
            contacted.add(residue.label)
    return contacted


def _relabel_chain(labels: Iterable[str], source_chain: str, target_chain: str) -> set[str]:
    """Rewrite `CHAIN:NUMBER` labels from one chain letter onto another.

    Residues measured on one structure carry that structure's chain letter. A
    second structure that holds the same protein on another letter needs the
    same residue numbers under its own letter before the two sets can be
    compared. Residue numbers never change, which is the relabelling
    `binder_contract._site_map_for_target` already applies to configured site
    labels. A label on any other chain is left alone.
    """
    if source_chain == target_chain:
        return {str(label) for label in labels}
    relabelled: set[str] = set()
    for label in labels:
        text = str(label)
        prefix, separator, suffix = text.partition(":")
        if separator and prefix == source_chain:
            relabelled.add(f"{target_chain}:{suffix}")
        else:
            relabelled.add(text)
    return relabelled


# One design summary is read once per (file, columns) and reused. The key
# carries size and modification time, so a table that is rewritten under a
# running job is re-read rather than served from the cache.
_PUBLISHED_EPITOPE_CACHE: dict[tuple, dict[str, str]] = {}


def _published_epitope_column(table: Any, key_column: str, residue_column: str) -> dict[str, str]:
    """Return `{design key: raw epitope cell}` for one design summary table."""
    path = table if isinstance(table, Path) else Path(str(table))
    if not path.is_file():
        raise MetricInputError(
            f"published_epitope_table is not a file: {path}. It is the released "
            "design_summary.csv the row is being replicated against."
        )
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, key_column, residue_column)
    cached = _PUBLISHED_EPITOPE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        for name in (key_column, residue_column):
            if name not in header:
                raise MetricInputError(
                    f"published_epitope_table {path} has no {name!r} column. It carries "
                    f"{len(header)} columns."
                )
        rows: dict[str, str] = {}
        for row in reader:
            key = str(row.get(key_column) or "").strip()
            if not key:
                continue
            if key in rows:
                raise MetricInputError(
                    f"published_epitope_table {path} carries {key!r} twice, so the "
                    "epitope for that design is ambiguous"
                )
            rows[key] = str(row.get(residue_column) or "").strip()
    _PUBLISHED_EPITOPE_CACHE[cache_key] = rows
    return rows


def _read_published_epitope(
    table: Any,
    design_key: str,
    key_column: str,
    residue_column: str,
) -> list[tuple[str, str]]:
    """Return one design's released epitope as `(CHAIN:NUMBER, three-letter code)`.

    The cell is a semicolon-separated list of `<chain>:<residue><number>`, for
    example `B:THR18;B:TYR19`, in the numbering of the co-fold construct. The
    three-letter code comes back beside the label rather than folded into it,
    because `_expand_residue_labels` matches on chain and number, and the caller
    checks the code against the structure so a numbering mismatch is loud.

    The cell is a measurement, not an input. `ref/docs/INSILICO.md` section 8
    defines it as target residues within 5 Angstrom of a binder heavy atom on
    the finished structure, so it records where a design landed. It is not the
    epitope the designer was given, and it must not be described as one.

    Every failure raises. Two of the released 1,440 rows carry an empty cell,
    and a design whose epitope was never released is not one this source can
    score.
    """
    rows = _published_epitope_column(table, key_column, residue_column)
    if design_key not in rows:
        raise MetricInputError(
            f"published_epitope_table has no {key_column} {design_key!r}. The design key "
            "has to name a row of the table the campaign is replicating."
        )
    cell = rows[design_key]
    if not cell:
        raise MetricInputError(
            f"published_epitope_table row {design_key!r} has an empty {residue_column}. "
            "The released record supplies no epitope for that design, so this row "
            "cannot be scored against one."
        )
    parsed: list[tuple[str, str]] = []
    for entry in cell.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        match = _PUBLISHED_EPITOPE_PATTERN.match(entry)
        if match is None:
            raise MetricInputError(
                f"epitope residue {entry!r} in row {design_key!r} is not "
                "CHAIN:RESIDUE_NUMBER, which is the form the released table uses"
            )
        chain, comp, number = match.group(1), match.group(2).upper(), match.group(3)
        parsed.append((f"{chain}:{number}", comp))
    if not parsed:
        raise MetricInputError(
            f"published_epitope_table row {design_key!r} carries no residue in "
            f"{residue_column}"
        )
    return parsed


def _apply_chain_policy(label: str, target_chain_id: str, policy: str) -> str:
    """Return the label the campaign scores, under the declared chain policy.

    `single-chain-to-target` rewrites the chain letter and keeps the number.
    `as-written` returns the label untouched. The caller has already refused a
    row naming more than one chain, so a rewrite here can only be the one
    correspondence available.
    """
    if policy != "single-chain-to-target":
        return label
    chain, _, number = label.partition(":")
    if not number or chain == target_chain_id:
        return label
    return f"{target_chain_id}:{number}"


def _check_published_residue_identity(
    structure: Structure,
    target_chain_id: str,
    entries: Sequence[tuple[str, str]],
    translation: Mapping[str, Any],
    policy: str,
) -> None:
    """Refuse a published epitope whose numbering does not fit the target chain.

    The released numbering is the co-fold construct's, so it agrees with this
    campaign's target chain only when the two constructs agree or a residue map
    translates between them. A residue the construct crops is absent and is not
    an error. A residue that is present under a different three-letter code is a
    numbering mismatch, and scoring `hotspot_recovery` against it would return a
    plausible wrong number.
    """
    # The chains the epitope lands on after translation. `_expand_residue_labels`
    # keeps the target chain and drops the rest silently, so an epitope that
    # spans two chains would score against the kept part and read as a complete
    # measurement. 321 of the released 1,440 rows name two chains and 29 name
    # three, so this is the common case on an oligomeric target rather than an
    # edge case. This function reads one target chain, the same limit
    # `compute_dockq` documents.
    chains = sorted({str(translation.get(label, label)).split(":", 1)[0] for label, _ in entries})
    if chains != [target_chain_id]:
        allowed = len(chains) == 1 and policy == "single-chain-to-target"
        if not allowed:
            raise MetricInputError(
                f"the published epitope lands on chain {', '.join(chains)} and the "
                f"target chain is {target_chain_id!r}. A row naming one chain can be "
                "read as the target chain by setting site.published_epitope_chain_policy "
                "to single-chain-to-target, or mapped with source_to_cleaned in the "
                "residue map. A row naming more than one chain needs a multi-chain "
                "target, which this metric does not read."
            )
    present = {
        f"{residue.auth_chain}:{residue.seq_id}": (residue.comp_id or "").upper()
        for residue in structure.chain_residues(target_chain_id)
    }
    on_chain = 0
    overlap = 0
    mismatched: list[str] = []
    for label, comp in entries:
        # The chain is compared after translation, because a residue map that
        # renumbers a construct can rename the chain along with the numbers.
        translated = _apply_chain_policy(str(translation.get(label, label)), target_chain_id, policy)
        if translated.split(":", 1)[0] != target_chain_id:
            continue
        on_chain += 1
        found = present.get(translated)
        if found is None:
            continue
        overlap += 1
        if found and comp and found != comp:
            mismatched.append(f"{translated} is {found} and the table says {comp}")
    if on_chain and overlap == 0:
        raise MetricInputError(
            f"no published epitope residue of chain {target_chain_id!r} exists in the "
            "structure, so the released numbering and this campaign's numbering do not "
            "line up. Supply source_to_cleaned in the residue map. The released "
            "numbering is the co-fold construct's, which a cropped or renumbered "
            "target construct does not share."
        )
    if mismatched:
        raise MetricInputError(
            "the published epitope does not match the target chain: "
            + "; ".join(mismatched[:5])
            + (f" and {len(mismatched) - 5} more" if len(mismatched) > 5 else "")
            + ". The released numbering is the co-fold construct's, so a different "
            "construct needs source_to_cleaned in the residue map."
        )


def _resolve_hotspots(
    site_residue_map: Mapping[str, Any],
    *,
    predicted: Structure,
    target_chain_id: str,
    binder_chain_id: str,
    cutoff: float,
    translate: Any,
    translation: Mapping[str, Any],
    designed_cif: Any,
    designed_chain_mapping: Mapping[str, str] | None,
    design_key: Any,
) -> tuple[str, set[str]]:
    """Return the declared source name and the hotspot labels it supplies.

    Every path either returns a non-empty set or raises. `site-fallback` is the
    deliberate exception that reuses `site_residues`; the returned relationship
    and duplicate flag keep that choice visible on the scored row.
    """
    source = site_residue_map.get("hotspot_source")
    # Direct callers written before source provenance existed supplied a hotspot
    # list and no source, so absent remains explicit here. The predictor-side
    # mapping resolves an absent campaign list to site-fallback before this call.
    source = "explicit" if source is None else str(source)
    if source not in HOTSPOT_SOURCES:
        raise MetricInputError(
            f"hotspot_source {source!r} is not registered. It is one of "
            + ", ".join(HOTSPOT_SOURCES)
        )

    if source in {"explicit", "site-fallback"}:
        entries = site_residue_map.get("hotspot_residues")
        if not entries:
            raise MetricInputError(
                "site_residue_map is missing 'hotspot_residues'. hotspot_source is "
                f"{source!r}, so the resolved mapping has to carry the list as "
                "CHAIN:NUMBER or CHAIN:START-END entries."
            )
        hotspots = _expand_residue_labels(translate(entries), target_chain_id)
    elif source == "designed_complex":
        if designed_cif is None:
            raise MetricInputError(
                "hotspot_source is 'designed_complex' and designed_cif was not passed. "
                "It is the same designed pose the caller already hands compute_dockq as "
                "reference_cif."
            )
        if designed_chain_mapping is None:
            raise MetricInputError(
                "hotspot_source is 'designed_complex' and designed_chain_mapping was not "
                "passed. The designed pose is a different structure from the prediction "
                "and carries its own chain ids, so reading it with the predicted letters "
                "returns the other chain's contacts whenever the two disagree. It is the "
                "same mapping the caller already hands compute_dockq as "
                "reference_chain_mapping."
            )
        designed_target_chain_id = str(designed_chain_mapping["target"])
        designed_binder_chain_id = str(designed_chain_mapping["binder"])
        if designed_target_chain_id == designed_binder_chain_id:
            raise MetricInputError(
                "designed_chain_mapping target and binder must be different chains"
            )
        designed = parse_structure_atoms(designed_cif, argument="designed_cif")
        # No source_to_cleaned here. These labels are measured on a structure
        # rather than read from config, so they are already in the structure's
        # own numbering, which is the numbering the contacts are matched in.
        measured = _contacted_target_residues(
            designed, designed_target_chain_id, designed_binder_chain_id, cutoff
        )
        if not measured:
            raise MetricInputError(
                "hotspot_source is 'designed_complex' and the designed pose has no "
                f"contact between chains {designed_target_chain_id!r} and "
                f"{designed_binder_chain_id!r} within {cutoff} Angstrom. "
                "designed_chain_mapping names the designed pose's own chains, which are "
                "not always the ones the prediction came back on."
            )
        # The measured labels carry the designed pose's target chain. The site,
        # the predicted contacts and every other hotspot source are keyed on the
        # predicted target chain, so the labels move onto it before they are
        # compared.
        hotspots = _relabel_chain(measured, designed_target_chain_id, target_chain_id)
    else:
        table = site_residue_map.get("published_epitope_table")
        if not table:
            raise MetricInputError(
                "hotspot_source is 'published_epitope' and site_residue_map is missing "
                "'published_epitope_table'. It is the path to the released "
                "design_summary.csv."
            )
        if not design_key:
            raise MetricInputError(
                "hotspot_source is 'published_epitope' and design_key was not passed. "
                "It is the value that names this design's row in the released table."
            )
        entries = _read_published_epitope(
            table,
            str(design_key),
            str(site_residue_map.get("published_epitope_key_column") or PUBLISHED_EPITOPE_KEY_COLUMN),
            str(
                site_residue_map.get("published_epitope_residue_column")
                or PUBLISHED_EPITOPE_RESIDUE_COLUMN
            ),
        )
        policy = str(
            site_residue_map.get("published_epitope_chain_policy")
            or PUBLISHED_EPITOPE_CHAIN_POLICIES[0]
        )
        if policy not in PUBLISHED_EPITOPE_CHAIN_POLICIES:
            raise MetricInputError(
                f"published_epitope_chain_policy {policy!r} is not registered. It is "
                "one of " + ", ".join(PUBLISHED_EPITOPE_CHAIN_POLICIES)
            )
        _check_published_residue_identity(
            predicted, target_chain_id, entries, translation, policy
        )
        hotspots = _expand_residue_labels(
            [
                _apply_chain_policy(label, target_chain_id, policy)
                for label in translate([label for label, _ in entries])
            ],
            target_chain_id,
        )

    if not hotspots:
        raise MetricInputError(
            f"hotspot_source {source!r} supplied no residue on target chain "
            f"{target_chain_id!r}. A hotspot set that is empty would score "
            "hotspot_recovery as zero and read as a failed design rather than a "
            "missing input."
        )
    return source, hotspots


def _hotspot_relationship(site: set[str], hotspots: set[str]) -> str:
    """Describe the hotspot set relative to the broader contact site."""
    if hotspots == site:
        return "same-as-site"
    if hotspots < site:
        return "subset-of-site"
    if site < hotspots:
        return "superset-of-site"
    if hotspots & site:
        return "partially-overlaps-site"
    return "disjoint-from-site"


def compute_site_metrics(
    predicted_cif: Any,
    site_residue_map: Mapping[str, Any],
    target_chain_id: str,
    binder_chain_id: str,
    *,
    designed_cif: Any = None,
    designed_chain_mapping: Mapping[str, str] | None = None,
    design_key: Any = None,
) -> dict[str, Any]:
    """Epitope agreement between the predicted pose and the campaign's site.

    Returns `site_contact_iou`, `target_contact_recall`,
    `target_contact_precision`, `hotspot_recovery`, `offsite_contact_fraction`,
    `site_hotspot_source`, `site_hotspot_relationship`,
    `hotspot_recovery_duplicates_target_contact_recall`, `site_scorer_revision`,
    `site_residue_map_sha256`,
    `site_contact_cutoff_angstrom`, `site_atom_selection` and
    `site_metric_basis`.

    The five metrics are no longer gates. They are still required fields on
    every scored row and they still enter the sort tiebreak, so they are still
    computed.

    `site_residue_map` carries both the site definition and the provenance
    fields the row has to echo. The executor compares three of them against the
    target's own site contract, so they are passed through from the config
    rather than derived here. Required keys:

        site_residues              list of CHAIN:NUMBER or CHAIN:START-END,
                                   from targets[].site.reference_contact_residues
        contact_cutoff_angstrom    from targets[].site.contact_cutoff_angstrom
        atom_selection             from targets[].site.atom_selection
        residue_map_sha256         sha256 of the target's residue-map.json
        metric_basis               from scoring.implementations.site_metric_basis

    Optional keys:

        source_to_cleaned          the residue-map.json mapping, applied to
                                   site and hotspot labels before matching
        hotspot_source             one of HOTSPOT_SOURCES, from
                                   targets[].site.hotspot_source. Absent means
                                   `explicit` when hotspot_residues is present,
                                   otherwise `site-fallback`.
        hotspot_residues           list in the same form as site_residues,
                                   from targets[].site.hotspot_residues.
                                   Required by the `explicit` source.
        published_epitope_table    path to the released design_summary.csv.
                                   Required by the `published_epitope` source.
        published_epitope_key_column      defaults to `uuid`
        published_epitope_residue_column  defaults to `epitope_residues`
        published_epitope_chain_policy    one of PUBLISHED_EPITOPE_CHAIN_POLICIES,
                                          defaults to `as-written`

    Where hotspots come from
    ------------------------

    `hotspot_recovery` is the fraction of the declared hotspot set the predicted
    pose contacts, so the hotspot set has to be a different question from the
    site. `HOTSPOT_SOURCES` names the three answers, and the one that ran comes
    back as `site_hotspot_source` so a reader of a scored row can tell which.

    `explicit` reads the list the config names, and it is the only source that
    states an epitope chosen before design. `site-fallback` deliberately uses
    the broader contact site and discloses that `hotspot_recovery` duplicates
    `target_contact_recall`. Equality is informative, not a reason to refuse a
    run. `designed_complex` measures it on
    `designed_cif`, using target residues with any heavy atom within
    `contact_cutoff_angstrom` of a binder heavy atom, which is the rule
    `ref/docs/INSILICO.md` section 8 states for the released `epitope_residues`
    at 5 Angstrom. The campaign's own cutoff is used rather than 5, and it comes
    back on the row as `site_contact_cutoff_angstrom`. `published_epitope` reads
    the design's own released cell out of `design_summary.csv`.

    The last two are retrospective. Both measure where a finished design landed,
    by the same rule, one recomputed here and one read from the released record.
    Neither is a design-time hotspot specification, and `published_epitope` must
    never be described as one. `ref/docs/PROVENANCE.md:25` records that most arms
    chose hotspots autonomously and left the specification column blank, so for
    most released designs there is no design-time hotspot to read.

    `designed_cif` is the designed complex the row is scored against, the same
    structure the caller already passes `compute_dockq` as `reference_cif`.
    `designed_chain_mapping` names that structure's own target and binder
    chains, the same mapping the caller already passes `compute_dockq` as
    `reference_chain_mapping`. Both are required by `designed_complex` and
    ignored by every other source. `target_chain_id` and `binder_chain_id`
    always describe `predicted_cif`, and the designed pose is a different file
    whose letters are not guaranteed to match, so the designed pair is passed
    rather than assumed. `design_key` names this design's row in the released
    table. Each is required only by its own source.

    The only fallback to `site_residues` is the named `site-fallback` source.
    It keeps a campaign useful when no distinct hotspot was chosen and makes the
    duplicate metric explicit on every scored row.
    """
    for required in (
        "contact_cutoff_angstrom",
        "atom_selection",
        "residue_map_sha256",
        "metric_basis",
    ):
        if required not in site_residue_map:
            raise MetricInputError(
                f"site_residue_map is missing {required!r}. See compute_site_metrics for where "
                "each key comes from in the campaign config."
            )

    atom_selection = str(site_residue_map["atom_selection"])
    if atom_selection != "heavy-atoms":
        raise MetricInputError(
            f"site_atom_selection {atom_selection!r} is not implemented. Only heavy-atoms is."
        )
    cutoff = float(site_residue_map["contact_cutoff_angstrom"])

    translation = site_residue_map.get("source_to_cleaned") or {}

    def translate(entries: Iterable[str]) -> list[str]:
        return [str(translation.get(str(entry), entry)) for entry in entries]

    structure = parse_structure_atoms(predicted_cif, argument="predicted_cif")
    contacted = _contacted_target_residues(structure, target_chain_id, binder_chain_id, cutoff)
    constraint = str(site_residue_map.get("epitope_constraint", "constrained"))
    if constraint == "unconstrained":
        return {
            "site_contact_iou": None,
            "target_contact_recall": None,
            "target_contact_precision": None,
            "hotspot_recovery": None,
            "offsite_contact_fraction": None,
            "site_hotspot_source": "unconstrained",
            "site_hotspot_relationship": "unconstrained",
            "hotspot_recovery_duplicates_target_contact_recall": False,
            "target_contact_residues": sorted(contacted),
            "site_scorer_revision": SITE_SCORER_IMPLEMENTATION_REVISION,
            "site_residue_map_sha256": str(site_residue_map["residue_map_sha256"]),
            "site_contact_cutoff_angstrom": cutoff,
            "site_atom_selection": atom_selection,
            "site_metric_basis": str(site_residue_map["metric_basis"]),
        }
    if constraint != "constrained":
        raise MetricInputError(
            f"epitope_constraint {constraint!r} is not registered. It is constrained or unconstrained"
        )

    if "site_residues" not in site_residue_map:
        raise MetricInputError("site_residue_map is missing 'site_residues' for a constrained run")
    site = _expand_residue_labels(translate(site_residue_map["site_residues"]), target_chain_id)
    if not site:
        raise MetricInputError(f"site_residues has no residue on target chain {target_chain_id!r}")

    hotspot_source, hotspots = _resolve_hotspots(
        site_residue_map,
        predicted=structure,
        target_chain_id=target_chain_id,
        binder_chain_id=binder_chain_id,
        cutoff=cutoff,
        translate=translate,
        translation=translation,
        designed_cif=designed_cif,
        designed_chain_mapping=designed_chain_mapping,
        design_key=design_key,
    )
    hotspot_relationship = _hotspot_relationship(site, hotspots)
    duplicates_site_recall = hotspot_relationship == "same-as-site"

    hit = contacted & site
    union = contacted | site
    site_contact_iou = len(hit) / len(union) if union else 0.0
    target_contact_recall = len(hit) / len(site)
    target_contact_precision = len(hit) / len(contacted) if contacted else 0.0
    offsite_contact_fraction = (len(contacted) - len(hit)) / len(contacted) if contacted else 0.0
    # `hotspots` is never empty. _resolve_hotspots raises rather than returning
    # an empty set, because dividing by it would write zero on a row whose input
    # was missing and read as a design that missed its epitope.
    hotspot_recovery = len(contacted & hotspots) / len(hotspots)

    return {
        "site_contact_iou": float(site_contact_iou),
        "target_contact_recall": float(target_contact_recall),
        "target_contact_precision": float(target_contact_precision),
        "hotspot_recovery": float(hotspot_recovery),
        "offsite_contact_fraction": float(offsite_contact_fraction),
        # Which of HOTSPOT_SOURCES produced hotspot_recovery. This widens what
        # the metric returns, so it widens the measurement and, through
        # interface_scorer.observation_from_raw, every scored row.
        "site_hotspot_source": hotspot_source,
        "site_hotspot_relationship": hotspot_relationship,
        "hotspot_recovery_duplicates_target_contact_recall": duplicates_site_recall,
        "target_contact_residues": sorted(contacted),
        "site_scorer_revision": SITE_SCORER_IMPLEMENTATION_REVISION,
        "site_residue_map_sha256": str(site_residue_map["residue_map_sha256"]),
        "site_contact_cutoff_angstrom": cutoff,
        "site_atom_selection": atom_selection,
        "site_metric_basis": str(site_residue_map["metric_basis"]),
    }


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------


def _synthetic_cif(layout: Sequence[tuple[str, int, str, float]]) -> str:
    """Build a minimal single-model mmCIF with one residue per entry.

    Each entry is (chain, residue number, three-letter code, x offset). Every
    residue gets a full N, CA, C, O backbone so it survives a superposition.
    """
    header = [
        "data_test",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_asym_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    offsets = {
        "N": (0.0, 0.0, 0.0),
        "CA": (1.0, 0.0, 0.0),
        "C": (2.0, 0.0, 0.0),
        "O": (2.0, 1.0, 0.0),
    }
    rows = []
    serial = 1
    for chain, number, comp, x_shift in layout:
        for name, (dx, dy, dz) in offsets.items():
            rows.append(
                f"ATOM {serial} {name[0]} {name} {comp} {chain} {number} . "
                f"{x_shift + dx:.3f} {number * 3.8 + dy:.3f} {dz:.3f} 1.00 85.00 {number} {chain} 1"
            )
            serial += 1
    return "\n".join(header + rows) + "\n#\n"


def _self_test() -> None:
    # A 2-chain complex: chain A is the target with 4 residues, chain B is the
    # binder with 3. Chain B sits 6 Angstroms away along x, close enough for
    # C-alpha contacts and far enough to avoid a clash.
    layout = [("A", i, "ALA", 0.0) for i in range(1, 5)]
    layout += [("B", i, "GLY", 6.0) for i in range(1, 4)]
    cif = _synthetic_cif(layout)
    structure = parse_cif_atoms(cif)
    assert len(structure) == 7, len(structure)
    assert [r.auth_chain for r in structure.residues] == list("AAAABBB")

    # An asymmetric PAE matrix, so the two directions genuinely differ. The
    # target-aligned block is confident and the binder-aligned block is not.
    size = 7
    pae = [[0.5 if i == j else 20.0 for j in range(size)] for i in range(size)]
    for i in range(0, 4):
        for j in range(4, 7):
            pae[i][j] = 1.0
            pae[j][i] = 9.0

    result = compute_ipsae(pae, cif, "A", "B")
    assert 0.0 < result["ipsae_binder_to_target"] < result["ipsae_target_to_binder"] <= 1.0, result
    assert result["ipsae_min"] == result["ipsae_binder_to_target"], result
    assert math.isclose(result["interface_pae"], 5.0), result["interface_pae"]
    assert result["contact_count"] > 0, result
    assert result["clash_count"] == 0, result
    assert math.isclose(result["interface_plddt"], 85.0), result["interface_plddt"]

    # Swapping the chain arguments swaps the two directions and nothing else.
    swapped = compute_ipsae(pae, cif, "B", "A")
    assert math.isclose(swapped["ipsae_target_to_binder"], result["ipsae_binder_to_target"])
    assert math.isclose(swapped["ipsae_min"], result["ipsae_min"])

    # Transposing the matrix and the orientation flag gives the same answer.
    transposed = [list(row) for row in zip(*pae)]
    flipped = compute_ipsae(transposed, cif, "A", "B", pae_orientation="aligned_columns")
    assert math.isclose(flipped["ipsae_target_to_binder"], result["ipsae_target_to_binder"])

    # A tighter cutoff drops the weaker direction to zero and leaves the other.
    tight = compute_ipsae(pae, cif, "A", "B", pae_cutoff_angstrom=5.0)
    assert tight["ipsae_binder_to_target"] == 0.0, tight
    assert tight["ipsae_target_to_binder"] > 0.0, tight

    # No pair under the cutoff gives exactly zero.
    far = [[50.0] * size for _ in range(size)]
    assert compute_ipsae(far, cif, "A", "B")["ipsae_min"] == 0.0

    # A pLDDT written on the 0-to-1 scale is rescaled rather than passed through.
    low = cif.replace(" 1.00 85.00 ", " 1.00 0.85 ")
    assert math.isclose(compute_ipsae(pae, low, "A", "B")["interface_plddt"], 85.0)

    # Overlapping chains are counted as clashes.
    overlapped = _synthetic_cif(
        [("A", i, "ALA", 0.0) for i in range(1, 5)] + [("B", i, "GLY", 0.5) for i in range(1, 4)]
    )
    assert compute_ipsae(pae, overlapped, "A", "B")["clash_count"] > 0

    # d0 is floored at 1.0 and follows the published fit above it.
    assert math.isclose(float(_d0(np.array([5.0]))[0]), 1.0)
    assert math.isclose(float(_d0(np.array([500.0]))[0]), 1.24 * (485.0 ** (1 / 3)) - 1.8)

    # DockQ against the structure itself is a perfect score.
    perfect = compute_dockq(cif, cif, {"target": "A", "binder": "B"})
    assert perfect["mapping_status"] == "ok", perfect
    assert math.isclose(perfect["sc_dockq"], 1.0), perfect
    assert math.isclose(perfect["fnat"], 1.0), perfect
    assert perfect["aligned_target_residue_count"] == 4, perfect
    assert math.isclose(perfect["target_alignment_rmsd"], 0.0, abs_tol=1e-9), perfect

    # Pushing the binder away drops every DockQ term.
    moved = _synthetic_cif(
        [("A", i, "ALA", 0.0) for i in range(1, 5)] + [("B", i, "GLY", 16.0) for i in range(1, 4)]
    )
    displaced = compute_dockq(moved, cif, {"target": "A", "binder": "B"})
    assert displaced["mapping_status"] == "ok", displaced
    assert displaced["sc_dockq"] < perfect["sc_dockq"], displaced
    assert displaced["ligand_rmsd"] > 9.0, displaced

    # A reference with no interface reports why instead of scoring.
    apart = _synthetic_cif(
        [("A", i, "ALA", 0.0) for i in range(1, 5)] + [("B", i, "GLY", 60.0) for i in range(1, 4)]
    )
    assert compute_dockq(cif, apart, {"target": "A", "binder": "B"})["mapping_status"] == (
        "no-reference-interface"
    )

    # Site metrics: the binder reaches every target residue at a 12 Angstrom
    # cutoff, and the declared site covers two of the four.
    site = compute_site_metrics(
        cif,
        {
            "site_residues": ["A:1-2"],
            "hotspot_residues": ["A:1"],
            "contact_cutoff_angstrom": 12.0,
            "atom_selection": "heavy-atoms",
            "residue_map_sha256": "f" * 64,
            "metric_basis": "selected-cofold-pose",
        },
        "A",
        "B",
    )
    assert math.isclose(site["target_contact_recall"], 1.0), site
    assert math.isclose(site["target_contact_precision"], 0.5), site
    assert math.isclose(site["offsite_contact_fraction"], 0.5), site
    assert math.isclose(site["site_contact_iou"], 0.5), site
    assert math.isclose(site["hotspot_recovery"], 1.0), site
    assert site["site_atom_selection"] == "heavy-atoms", site
    assert site["site_scorer_revision"] == SITE_SCORER_IMPLEMENTATION_REVISION, site
    # A mapping with no hotspot_source is the `explicit` source, which is what
    # this mapping has always carried, and the row says so.
    assert site["site_hotspot_source"] == "explicit", site

    # A missing hotspot list stops the job instead of guessing one.
    try:
        compute_site_metrics(
            cif,
            {
                "site_residues": ["A:1"],
                "contact_cutoff_angstrom": 5.0,
                "atom_selection": "heavy-atoms",
                "residue_map_sha256": "f" * 64,
                "metric_basis": "selected-cofold-pose",
            },
            "A",
            "B",
        )
    except MetricInputError as exc:
        assert "hotspot_residues" in str(exc), exc
    else:
        raise AssertionError("a missing hotspot_residues has to raise")

    # -- the three hotspot sources ------------------------------------------

    def site_map(**overrides):
        base = {
            "site_residues": ["A:3-4"],
            "contact_cutoff_angstrom": 5.0,
            "atom_selection": "heavy-atoms",
            "residue_map_sha256": "f" * 64,
            "metric_basis": "selected-cofold-pose",
        }
        base.update(overrides)
        return base

    # The designed complex the replication path measures hotspots on. Its binder
    # sits beside target residues 2 to 4, and the predicted pose above contacts
    # residues 1 to 3 at a 5 Angstrom cutoff, so the two sets genuinely differ.
    designed = _synthetic_cif(
        [("A", i, "ALA", 0.0) for i in range(1, 5)] + [("B", i, "GLY", 6.0) for i in (3, 4)]
    )
    assert _contacted_target_residues(parse_cif_atoms(cif), "A", "B", 5.0) == {"A:1", "A:2", "A:3"}
    assert _contacted_target_residues(parse_cif_atoms(designed), "A", "B", 5.0) == {
        "A:2",
        "A:3",
        "A:4",
    }

    from_designed = compute_site_metrics(
        cif,
        site_map(hotspot_source="designed_complex"),
        "A",
        "B",
        designed_cif=designed,
        designed_chain_mapping={"target": "A", "binder": "B"},
    )
    assert from_designed["site_hotspot_source"] == "designed_complex", from_designed
    # Two of the designed complex's three epitope residues are contacted.
    assert math.isclose(from_designed["hotspot_recovery"], 2 / 3), from_designed
    # And the site metric is a different question from the hotspot metric, which
    # is the whole reason this source exists.
    assert math.isclose(from_designed["target_contact_recall"], 0.5), from_designed
    assert math.isclose(from_designed["site_contact_cutoff_angstrom"], 5.0), from_designed

    # The campaign's own cutoff is read rather than the published 5 Angstrom. At
    # 12 Angstrom the same designed complex names every target residue.
    wider = compute_site_metrics(
        cif,
        site_map(hotspot_source="designed_complex", contact_cutoff_angstrom=12.0),
        "A",
        "B",
        designed_cif=designed,
        designed_chain_mapping={"target": "A", "binder": "B"},
    )
    assert math.isclose(wider["hotspot_recovery"], 1.0), wider
    assert math.isclose(wider["site_contact_cutoff_angstrom"], 12.0), wider

    # The replication path without the designed complex stops rather than
    # scoring against something else.
    try:
        compute_site_metrics(cif, site_map(hotspot_source="designed_complex"), "A", "B")
    except MetricInputError as exc:
        assert "designed_cif" in str(exc), exc
    else:
        raise AssertionError("designed_complex without designed_cif has to raise")

    # The released table, in the form ref/docs/INSILICO.md section 8 states.
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        table = Path(scratch) / "design_summary.csv"
        table.write_text(
            "uuid,epitope_residues,epitope_n_residues\n"
            "design-a,A:ALA1;A:ALA4,2\n"
            "design-blank,,0\n"
            "design-wrong,A:GLY1,1\n"
        )
        published = compute_site_metrics(
            cif,
            site_map(hotspot_source="published_epitope", published_epitope_table=str(table)),
            "A",
            "B",
            design_key="design-a",
        )
        assert published["site_hotspot_source"] == "published_epitope", published
        # The pose reaches A:1 and not A:4.
        assert math.isclose(published["hotspot_recovery"], 0.5), published

        # Two of the released 1,440 rows carry an empty cell. That is a design
        # this source cannot score, and it says so.
        try:
            compute_site_metrics(
                cif,
                site_map(hotspot_source="published_epitope", published_epitope_table=str(table)),
                "A",
                "B",
                design_key="design-blank",
            )
        except MetricInputError as exc:
            assert "design-blank" in str(exc), exc
        else:
            raise AssertionError("an empty epitope cell has to raise")

        # A design the table does not carry.
        try:
            compute_site_metrics(
                cif,
                site_map(hotspot_source="published_epitope", published_epitope_table=str(table)),
                "A",
                "B",
                design_key="design-missing",
            )
        except MetricInputError as exc:
            assert "design-missing" in str(exc), exc
        else:
            raise AssertionError("an absent table row has to raise")

        # A residue the table names as GLY where the chain carries ALA is a
        # numbering mismatch, and scoring through it would return a plausible
        # wrong number.
        try:
            compute_site_metrics(
                cif,
                site_map(hotspot_source="published_epitope", published_epitope_table=str(table)),
                "A",
                "B",
                design_key="design-wrong",
            )
        except MetricInputError as exc:
            assert "source_to_cleaned" in str(exc), exc
        else:
            raise AssertionError("a published residue identity mismatch has to raise")

        # The released numbering is the co-fold construct's, so a campaign whose
        # target chain is lettered differently reaches it through the residue
        # map. Chain B in the table, chain A in this structure.
        table.write_text(
            table.read_text() + "design-otherchain,B:ALA1;B:ALA4,2\n"
        )
        translated = compute_site_metrics(
            cif,
            site_map(
                hotspot_source="published_epitope",
                published_epitope_table=str(table),
                source_to_cleaned={"B:1": "A:1", "B:4": "A:4"},
            ),
            "A",
            "B",
            design_key="design-otherchain",
        )
        assert math.isclose(translated["hotspot_recovery"], 0.5), translated

        # The same row without the map lands on a chain this campaign does not
        # score, and an epitope that landed nowhere is a stopped job, not a zero.
        try:
            compute_site_metrics(
                cif,
                site_map(hotspot_source="published_epitope", published_epitope_table=str(table)),
                "A",
                "B",
                design_key="design-otherchain",
            )
        except MetricInputError as exc:
            assert "lands on chain B" in str(exc), exc
        else:
            raise AssertionError("a published epitope on another chain has to raise")

        # An epitope that spans two chains would lose the residues off the
        # target chain without saying so. 321 of the released 1,440 rows name
        # two chains, so this is the common case on an oligomeric target.
        table.write_text(table.read_text() + "design-twochain,A:ALA1;B:GLY1,2\n")
        try:
            compute_site_metrics(
                cif,
                site_map(hotspot_source="published_epitope", published_epitope_table=str(table)),
                "A",
                "B",
                design_key="design-twochain",
            )
        except MetricInputError as exc:
            assert "spans" in str(exc) or "lands on chain A, B" in str(exc), exc
        else:
            raise AssertionError("a two-chain published epitope has to raise")

        # The released chain letter varies per row inside one target, so a
        # campaign can declare that a single-chain row is read as the target
        # chain. The numbers are kept and the identity check still runs.
        relabelled = compute_site_metrics(
            cif,
            site_map(
                hotspot_source="published_epitope",
                published_epitope_table=str(table),
                published_epitope_chain_policy="single-chain-to-target",
            ),
            "A",
            "B",
            design_key="design-otherchain",
        )
        assert math.isclose(relabelled["hotspot_recovery"], 0.5), relabelled

        # The policy relabels, it does not renumber. A code that disagrees with
        # the chain still stops the job.
        table.write_text(table.read_text() + "design-relabel-wrong,B:GLY1,1\n")
        try:
            compute_site_metrics(
                cif,
                site_map(
                    hotspot_source="published_epitope",
                    published_epitope_table=str(table),
                    published_epitope_chain_policy="single-chain-to-target",
                ),
                "A",
                "B",
                design_key="design-relabel-wrong",
            )
        except MetricInputError as exc:
            assert "source_to_cleaned" in str(exc), exc
        else:
            raise AssertionError("a relabelled residue with the wrong code has to raise")

        # A two-chain row has no single correspondence, so the policy does not
        # rescue it.
        try:
            compute_site_metrics(
                cif,
                site_map(
                    hotspot_source="published_epitope",
                    published_epitope_table=str(table),
                    published_epitope_chain_policy="single-chain-to-target",
                ),
                "A",
                "B",
                design_key="design-twochain",
            )
        except MetricInputError as exc:
            assert "lands on chain A, B" in str(exc), exc
        else:
            raise AssertionError("a two-chain row has to raise under either policy")

        # A policy nobody registered.
        try:
            compute_site_metrics(
                cif,
                site_map(
                    hotspot_source="published_epitope",
                    published_epitope_table=str(table),
                    published_epitope_chain_policy="guess",
                ),
                "A",
                "B",
                design_key="design-a",
            )
        except MetricInputError as exc:
            assert "guess" in str(exc) and "as-written" in str(exc), exc
        else:
            raise AssertionError("an unregistered chain policy has to raise")

        # The row key is required, because the released epitope is per design.
        try:
            compute_site_metrics(
                cif,
                site_map(hotspot_source="published_epitope", published_epitope_table=str(table)),
                "A",
                "B",
            )
        except MetricInputError as exc:
            assert "design_key" in str(exc), exc
        else:
            raise AssertionError("published_epitope without design_key has to raise")

    # A table path that does not exist.
    try:
        compute_site_metrics(
            cif,
            site_map(
                hotspot_source="published_epitope",
                published_epitope_table="/nonexistent/design_summary.csv",
            ),
            "A",
            "B",
            design_key="design-a",
        )
    except MetricInputError as exc:
        assert "published_epitope_table" in str(exc), exc
    else:
        raise AssertionError("a missing published table has to raise")

    # A source name nobody registered.
    try:
        compute_site_metrics(cif, site_map(hotspot_source="reference_contacts"), "A", "B")
    except MetricInputError as exc:
        assert "reference_contacts" in str(exc) and "designed_complex" in str(exc), exc
    else:
        raise AssertionError("an unregistered hotspot_source has to raise")

    # A hotspot list that names no residue on the target chain used to divide by
    # an empty set and write zero, which reads as a design that missed its
    # epitope rather than as an input that never arrived.
    try:
        compute_site_metrics(cif, site_map(hotspot_residues=["Z:1"]), "A", "B")
    except MetricInputError as exc:
        assert "no residue on target chain" in str(exc), exc
    else:
        raise AssertionError("a hotspot list off the target chain has to raise")

    # A PAE matrix that does not match the residue count is rejected.
    try:
        compute_ipsae([[0.0, 1.0], [1.0, 0.0]], cif, "A", "B")
    except MetricInputError as exc:
        assert "residues" in str(exc), exc
    else:
        raise AssertionError("a mismatched PAE matrix has to raise")

    # A chain the structure does not carry is named in the error.
    try:
        compute_ipsae(pae, cif, "A", "Z")
    except MetricInputError as exc:
        assert "'Z'" in str(exc), exc
    else:
        raise AssertionError("a missing chain has to raise")

    print("binder_metrics self test passed")


if __name__ == "__main__":
    _self_test()
