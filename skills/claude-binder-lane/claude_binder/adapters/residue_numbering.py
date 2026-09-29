#!/usr/bin/env python3
"""Map author residue identity to the sequential position a tool indexes by, and back.

Two numbering schemes appear in every structure this package handles, and no
module before this one named the difference.

**Author numbering** is what a depositor wrote. It starts where the construct
starts, it skips residues nobody resolved, and it can carry an insertion code.
`target_prep_adapter.write_structure` preserves it deliberately, so a residue ID
keeps its meaning across the source and the normalized file. Every `CHAIN:NUMBER`
string in a campaign manifest is author numbering, and the identity is the whole
triple of chain, number and insertion code rather than the bare number.

**Sequential position** is the 1-based index of a residue within its chain. The
mmCIF dictionary calls the entity-relative form `label_seq_id`. Caliby's
`pos_constraint_csv` reads it: the upstream README states that residue index
positions should be specified by the `label_seq_id` column, not the `auth_seq_id`
column. PXDesign's hotspot field reads something like it too, which is why
`proteina_complexa_generator` range-checks a hotspot against `1 <= n <= residue_count`.

The two disagree on this package's own demo target. `data/demo/tslp` ships chain
A numbered 28 to 159 holding 114 residues, with one gap between 115 and 134.
Author `A:142` is the 97th residue of that chain. A tool handed 142 when it wanted
97 constrains the wrong residue. A tool that range-checks 142 against 114 refuses
a target it could have designed against. Neither failure raises anything a reader
would recognise, which is why this module exists.

**Where the map comes from.** The preferred source is the target manifest, whose
`chains[i].residue_ids` is already an ordered list of author residue IDs in file
order, written by `target_prep_adapter.chain_record`. That is the map, recorded by
the stage that normalized the structure, not reconstructed later. Build from it
with `ChainNumbering.from_residue_ids`.

The fallback is a coordinate file, through `read_chain_numbering`. An mmCIF that
declares `_atom_site.label_seq_id` is read rather than derived, because the file
already answers the question. Everything else is derived by counting residues in
file order, which equals `label_seq_id` only when the file holds every residue of
the entity. A coordinate file cannot show whether residues are missing from the
entity, so a chain whose author numbering is discontinuous, or that carries
insertion codes, is refused rather than guessed at. Pass `allow_derived_gaps=True`
to take the derived answer anyway, and record on every row that you did.

The map is reversible. `position` goes author to sequential and `residue_at` goes
back, so a number written into a tool's input can be read back out of its output
and named in author terms again.
"""

from __future__ import annotations

import gzip
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


AUTHOR = "author"
SEQUENTIAL = "sequential"
NUMBERING_SCHEMES = (AUTHOR, SEQUENTIAL)

# `CHAIN:NUMBER` with an optional insertion code, the form every campaign
# manifest writes and `pxdesign_generator.site_residue_numbers` parses.
RESIDUE_ID_RE = re.compile(r"\A([A-Za-z0-9]):(-?\d+)([A-Za-z]?)\Z")

# Where a sequential position came from. Recorded on every row that carries one.
MANIFEST = "manifest_residue_ids"
DECLARED = "declared_label_seq_id"
DERIVED = "derived_file_order"


class NumberingError(Exception):
    """A residue identity cannot be translated between the two numbering schemes."""


@dataclass(frozen=True)
class ResidueId:
    """One residue in author numbering, as the full identity rather than a number."""

    chain: str
    number: int
    insertion_code: str = ""

    def __str__(self) -> str:
        return f"{self.chain}:{self.number}{self.insertion_code}"


def parse_residue_id(value: object) -> ResidueId:
    """Parse one `CHAIN:NUMBER` string, with an optional insertion code."""
    match = RESIDUE_ID_RE.fullmatch(str(value))
    if match is None:
        raise NumberingError(f"residue does not read CHAIN:NUMBER: {value}")
    return ResidueId(match.group(1), int(match.group(2)), match.group(3))


@dataclass(frozen=True)
class ChainNumbering:
    """A reversible map between author residue identity and sequential position."""

    chain: str
    residues: tuple[ResidueId, ...]
    sequential: tuple[int, ...]
    source: str
    _forward: dict[ResidueId, int] = field(default_factory=dict, repr=False, compare=False)
    _reverse: dict[int, ResidueId] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.residues) != len(self.sequential):
            raise NumberingError(
                f"chain {self.chain} holds {len(self.residues)} residues and "
                f"{len(self.sequential)} sequential positions"
            )
        if not self.residues:
            raise NumberingError(f"chain {self.chain} holds no residue")
        for residue, position in zip(self.residues, self.sequential):
            if residue.chain != self.chain:
                raise NumberingError(
                    f"{residue} names chain {residue.chain} in the numbering for chain {self.chain}"
                )
            if residue in self._forward:
                raise NumberingError(f"chain {self.chain} lists {residue} twice")
            if position in self._reverse:
                raise NumberingError(f"chain {self.chain} gives position {position} twice")
            self._forward[residue] = position
            self._reverse[position] = residue

    @classmethod
    def from_residue_ids(
        cls,
        residue_ids: Sequence[object],
        *,
        chain: str | None = None,
        source: str = MANIFEST,
        sequential: Sequence[int] | None = None,
    ) -> "ChainNumbering":
        """Build from an ordered list of author residue IDs, in file order.

        This is the manifest route. `target_prep_adapter.chain_record` writes
        `residue_ids` in exactly this order, so nothing here is reconstructed.
        """
        residues = tuple(
            value if isinstance(value, ResidueId) else parse_residue_id(value)
            for value in residue_ids
        )
        if not residues:
            raise NumberingError("residue_ids is empty")
        chains = {residue.chain for residue in residues}
        if len(chains) != 1:
            raise NumberingError(f"residue_ids spans chains {sorted(chains)}")
        only_chain = next(iter(chains))
        if chain is not None and chain != only_chain:
            raise NumberingError(f"residue_ids names chain {only_chain} and the caller named {chain}")
        positions = (
            tuple(int(value) for value in sequential)
            if sequential is not None
            else tuple(range(1, len(residues) + 1))
        )
        return cls(only_chain, residues, positions, source)

    @property
    def residue_count(self) -> int:
        return len(self.residues)

    @property
    def insertion_codes(self) -> tuple[str, ...]:
        return tuple(sorted({r.insertion_code for r in self.residues if r.insertion_code}))

    @property
    def author_gaps(self) -> tuple[tuple[int, int], ...]:
        """Every pair of consecutive listed residues whose author numbers are not adjacent."""
        pairs = zip(self.residues, self.residues[1:])
        return tuple(
            (first.number, second.number)
            for first, second in pairs
            if second.number != first.number + 1
        )

    @property
    def contiguous(self) -> bool:
        return not self.author_gaps and not self.insertion_codes

    def position(self, residue: ResidueId | str) -> int:
        """Return the sequential position of one author-numbered residue."""
        key = residue if isinstance(residue, ResidueId) else parse_residue_id(residue)
        try:
            return self._forward[key]
        except KeyError:
            raise NumberingError(
                f"{key} is not a residue of chain {self.chain}, which holds "
                f"{self.residue_count} residues numbered {self.author_span()}"
            ) from None

    def residue_at(self, position: int) -> ResidueId:
        """Return the author residue identity at one sequential position."""
        try:
            return self._reverse[int(position)]
        except KeyError:
            raise NumberingError(
                f"chain {self.chain} has no position {position}; it holds "
                f"{self.residue_count} residues"
            ) from None

    def author_span(self) -> str:
        return f"{self.residues[0].number} to {self.residues[-1].number}"


# ----------------------------------------------------------------------------
# The coordinate-file fallback.
# ----------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _pdb_residues(text: str) -> Iterator[tuple[str, ResidueId, int | None]]:
    """Yield first-model polymer residues of a PDB file, in file order.

    HETATM is skipped for the reason `backbone_shape` skips it: a ligand or a
    water is not a residue a sequence designer indexes.
    """
    saw_model = False
    in_first_model = True
    for line in text.splitlines():
        record = line[:6]
        if record == "MODEL ":
            if saw_model:
                in_first_model = False
            saw_model = True
            continue
        if record == "ENDMDL" and saw_model:
            in_first_model = False
            continue
        if not in_first_model or record != "ATOM  " or len(line) < 27:
            continue
        chain = line[21:22].strip() or "_"
        raw_number = line[22:26].strip()
        if not raw_number:
            continue
        try:
            number = int(raw_number)
        except ValueError as exc:
            raise NumberingError(f"PDB residue number is not an integer: {raw_number!r}") from exc
        yield chain, ResidueId(chain, number, line[26:27].strip()), None


def _mmcif_residues(text: str, path: Path) -> Iterator[tuple[str, ResidueId, int | None]]:
    """Yield first-model polymer residues of an mmCIF atom_site loop, in file order.

    The third value is the file's own `label_seq_id` when the loop declares one
    beside an author column. That is the answer this module would otherwise
    derive, so a file that states it is believed.
    """
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        columns: list[str] = []
        while index < len(lines) and lines[index].strip().startswith("_"):
            columns.append(lines[index].strip().split()[0])
            index += 1
        if not columns or not columns[0].startswith("_atom_site."):
            continue
        places = {column: place for place, column in enumerate(columns)}
        chain_place = places.get("_atom_site.auth_asym_id", places.get("_atom_site.label_asym_id"))
        number_place = places.get("_atom_site.auth_seq_id", places.get("_atom_site.label_seq_id"))
        if chain_place is None or number_place is None:
            raise NumberingError(f"mmCIF atom_site loop names no chain or residue column: {path}")
        label_place = places.get("_atom_site.label_seq_id")
        if label_place == number_place:
            # The loop has no author column, so the number already is the label.
            label_place = None
        icode_place = places.get("_atom_site.pdbx_PDB_ins_code")
        group_place = places.get("_atom_site.group_PDB")
        model_place = places.get("_atom_site.pdbx_PDB_model_num")
        while index < len(lines):
            row = lines[index].strip()
            if not row or row.startswith("#") or row.startswith("_") or row == "loop_":
                break
            index += 1
            fields = shlex.split(row, comments=False, posix=True)
            if len(fields) != len(columns):
                raise NumberingError(
                    f"mmCIF atom_site row has {len(fields)} fields and the loop names "
                    f"{len(columns)}: {path}"
                )
            if group_place is not None and fields[group_place] != "ATOM":
                continue
            if model_place is not None and fields[model_place] not in {"1", ".", "?"}:
                continue
            chain = fields[chain_place]
            raw_number = fields[number_place]
            if chain in {".", "?"} or raw_number in {".", "?"}:
                continue
            try:
                number = int(raw_number)
            except ValueError as exc:
                raise NumberingError(
                    f"mmCIF residue number is not an integer: {raw_number!r}"
                ) from exc
            icode = ""
            if icode_place is not None and fields[icode_place] not in {".", "?"}:
                icode = fields[icode_place].strip()
            label: int | None = None
            if label_place is not None and fields[label_place] not in {".", "?"}:
                try:
                    label = int(fields[label_place])
                except ValueError:
                    label = None
            yield chain, ResidueId(chain, number, icode), label
        return
    raise NumberingError(f"structure file has no mmCIF atom_site loop: {path}")


def read_chain_numbering(path: Path | str) -> dict[str, ChainNumbering]:
    """Return one `ChainNumbering` per chain of a PDB or mmCIF structure file.

    Prefer `from_manifest_chains` when a target manifest is available. This route
    reconstructs from coordinates what that manifest already recorded.

    The format is chosen by suffix and confirmed by content, the way
    `backbone_shape.read_backbone_shape` chooses it, so a `.pdb` file holding an
    mmCIF loop still reads.
    """
    structure = Path(path)
    if not structure.is_file():
        raise NumberingError(f"structure file not found: {structure}")
    text = _read_text(structure)
    name = structure.name[: -len(".gz")] if structure.name.endswith(".gz") else structure.name
    suffix = Path(name).suffix.lower()
    holds_pdb_atoms = any(line.startswith("ATOM  ") for line in text.splitlines())
    if suffix in {".cif", ".mmcif"} or ("_atom_site." in text and not holds_pdb_atoms):
        records = _mmcif_residues(text, structure)
    else:
        records = _pdb_residues(text)

    ordered: dict[str, list[ResidueId]] = {}
    declared: dict[str, list[int | None]] = {}
    seen: dict[str, set[ResidueId]] = {}
    for chain, residue, label in records:
        bucket = seen.setdefault(chain, set())
        if residue in bucket:
            continue
        bucket.add(residue)
        ordered.setdefault(chain, []).append(residue)
        declared.setdefault(chain, []).append(label)

    numbering: dict[str, ChainNumbering] = {}
    for chain, residues in ordered.items():
        labels = declared[chain]
        if all(label is not None for label in labels):
            numbering[chain] = ChainNumbering.from_residue_ids(
                residues,
                chain=chain,
                source=DECLARED,
                sequential=[int(label) for label in labels],  # type: ignore[arg-type]
            )
        else:
            numbering[chain] = ChainNumbering.from_residue_ids(
                residues, chain=chain, source=DERIVED
            )
    if not numbering:
        raise NumberingError(f"structure file holds no polymer residue: {structure}")
    return numbering


def from_manifest_chains(chains: Iterable[Mapping[str, object]]) -> dict[str, ChainNumbering]:
    """Build one `ChainNumbering` per chain from a target manifest's chain records.

    Each record is what `target_prep_adapter.chain_record` writes, so the fields
    read here are `chain_id` and `residue_ids`.
    """
    numbering: dict[str, ChainNumbering] = {}
    for record in chains:
        chain_id = record.get("chain_id")
        residue_ids = record.get("residue_ids")
        if not isinstance(chain_id, str) or not chain_id:
            raise NumberingError(f"chain record names no chain_id: {record}")
        if not isinstance(residue_ids, list) or not residue_ids:
            raise NumberingError(f"chain {chain_id} records no residue_ids")
        if chain_id in numbering:
            raise NumberingError(f"the manifest lists chain {chain_id} twice")
        numbering[chain_id] = ChainNumbering.from_residue_ids(
            residue_ids, chain=chain_id, source=MANIFEST
        )
    if not numbering:
        raise NumberingError("the manifest records no chain")
    return numbering


# ----------------------------------------------------------------------------
# Translation.
# ----------------------------------------------------------------------------


def sequential_positions(
    numbering: ChainNumbering,
    residues: Iterable[ResidueId | str],
    *,
    allow_derived_gaps: bool = False,
) -> list[int]:
    """Translate author-numbered residues of one chain into sequential positions.

    A derived answer on a chain that is not contiguous is refused, because the
    file cannot show whether the gap is an unmodelled residue the receiving tool
    will still count. A manifest-sourced or file-declared answer is not refused,
    because the stage that wrote it knew what the entity held.
    """
    if numbering.source == DERIVED and not numbering.contiguous and not allow_derived_gaps:
        raise NumberingError(
            f"chain {numbering.chain} is numbered {numbering.author_span()} with "
            f"{numbering.residue_count} residues, and its author numbering is not "
            f"contiguous: gaps {list(numbering.author_gaps)}, insertion codes "
            f"{list(numbering.insertion_codes)}. A position counted from a coordinate "
            "file equals label_seq_id only when the file holds every residue of the "
            "entity, and a coordinate file cannot show that. Supply the target "
            "manifest's residue_ids, or accept the derived numbering explicitly"
        )
    return [numbering.position(value) for value in residues]


def author_residues(numbering: ChainNumbering, positions: Iterable[int]) -> list[ResidueId]:
    """Translate sequential positions back into author residue identities."""
    return [numbering.residue_at(position) for position in positions]


def contiguous_ranges(numbers: Sequence[int]) -> list[tuple[int, int]]:
    """Collapse numbers into sorted inclusive ranges."""
    ordered = sorted({int(number) for number in numbers})
    ranges: list[tuple[int, int]] = []
    for number in ordered:
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], number)
        else:
            ranges.append((number, number))
    return ranges


def format_ranges(chain: str, numbers: Sequence[int]) -> str:
    """Render ranges as `A1-100,A105`, the form Caliby's pos_constraint_csv reads."""
    parts = [
        f"{chain}{start}" if start == end else f"{chain}{start}-{end}"
        for start, end in contiguous_ranges(numbers)
    ]
    return ",".join(parts)


def whole_chain_range(numbering: ChainNumbering) -> str:
    """Render every residue of one chain as a single range string."""
    return format_ranges(numbering.chain, numbering.sequential)
