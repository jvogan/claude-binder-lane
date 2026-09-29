"""Refuse a route that hands a sequence designer fewer backbone atoms than it reads.

A backbone generator that writes C-alpha coordinates only can be wired into a
sequence designer that places a sequence on N, CA, C and O. The designer then
fails without naming the cause, or returns a sequence computed from an
incomplete backbone. Genie3 in binder mode is the recorded case, at
`skills/claude-binder-lane/references/tool-catalogue.md`, section "Genie3 emits
C-alpha-only backbones".

This module holds both halves of the guard.

**Plan time.** `route_problems` resolves the producer and the consumer of every
sequence-design stage and reports a pairing the consumer cannot read. The
decision reads a declaration wherever both ends carry one, because two designers
with the same atom requirement can differ on a reduced backbone. A generator
names the shape it writes in its profile `emits` field, and a designer names the
shapes it accepts in `accepts_backbone_shapes`, mapping a shape to the flags that
shape costs. Vanilla ProteinMPNN accepts a C-alpha trace under `--ca-only`,
because it ships a `ca_model_weights` checkpoint. SolubleMPNN accepts none,
because no soluble C-alpha checkpoint exists. An atom-set comparison cannot tell
the two apart, since the difference lives in the checkpoints their publishers
shipped.

A pairing where either end names nothing falls back to the `backbone_atoms` atom
sets the catalog carries, which refuses a producer that writes fewer atoms than
the consumer reads. That floor is what silence gets, so no designer becomes
compatible with every shape by declaring no map. An `unknown` declaration reports
nothing on either mechanism. A refusal on an unstated atom set or an unnamed
shape would be a refusal on an invented one.

**Run time.** `read_backbone_shape` reads the atom names a structure file
carries, and `pose_shape_problem` refuses a file that falls short of what the
consumer reads. This half works where a declaration is `unknown`, because it
reads the file rather than a claim about the tool that wrote it.

Standard library only. The module opens structure files and the packaged
catalog. It starts no subprocess and makes no network request.
"""

from __future__ import annotations

import gzip
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .paths import package_file


# The package's backbone atom set, matching `structural_surrogates.py:60`. A
# sequence designer that reads a fixed backbone reads these four names.
BACKBONE_ATOM_ORDER = ("N", "CA", "C", "O")

# Catalog vocabulary. `stated` carries an atom list, `unknown` carries a TODO,
# and `not-in-route` means the tool is on neither end of the generator to
# sequence-designer route this declaration governs.
DECLARATION_FIELD = "backbone_atoms"
UNKNOWN = "unknown"
STATED = "stated"
NOT_IN_ROUTE = "not-in-route"
DECLARATION_STATUSES = (STATED, UNKNOWN, NOT_IN_ROUTE)

PRODUCES = "produces"
REQUIRES = "requires"

# Profile vocabulary, the second mechanism. A generator names the shape it writes
# in `emits`, and a designer maps every shape it accepts to the flags that shape
# costs in `accepts_backbone_shapes`. An entry absent from the map is a refusal,
# and an empty flag list is a shape that costs no flag, which is how every MPNN
# designer declares a full backbone. `unknown` on either field is unstated.
EMITS_FIELD = "emits"
ACCEPTS_FIELD = "accepts_backbone_shapes"
GENERATOR_SECTION = ("generation", "generators")
DESIGNER_SECTION = ("sequence_design", "designers")

# What the two declarations together settle for one producer to consumer pairing.
# `unstated` is an explicit `unknown` and reports nothing. `undecided` is silence,
# and hands the pairing to the `backbone_atoms` comparison.
SHAPE_ACCEPTED = "accepted"
SHAPE_REFUSED = "refused"
SHAPE_UNSTATED = "unstated"
SHAPE_UNDECIDED = "undecided"

# Which end of the route a tool sits on, read from its `stage_category`.
PRODUCER_CATEGORIES = frozenset(
    {"backbone-generation", "codesign-generation", "native-binder-design"}
)
CONSUMER_CATEGORIES = frozenset({"sequence-design"})

BACKBONE_STAGE_FLAG = "--backbone-stage-id"

# Sampling budget for `read_backbone_shape`. See `_sample` for what these two
# numbers buy and what they cost.
SAMPLE_RESIDUES_PER_CHAIN = 4
SAMPLE_ATOM_RECORDS = 50000


class BackboneShapeError(ValueError):
    """A structure file cannot be read for the atom names it carries."""


def order_atoms(atoms: Iterable[str]) -> tuple[str, ...]:
    """Return atom names in backbone order, then alphabetically."""
    unique = {str(atom).strip().upper() for atom in atoms if str(atom).strip()}
    ordered = [atom for atom in BACKBONE_ATOM_ORDER if atom in unique]
    ordered.extend(sorted(unique - set(BACKBONE_ATOM_ORDER)))
    return tuple(ordered)


def _atom_text(atoms: Iterable[str]) -> str:
    ordered = order_atoms(atoms)
    return ", ".join(ordered) if ordered else "no backbone atoms"


# ---------------------------------------------------------------------------
# Run time: what a structure file carries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackboneShape:
    """Atom names one structure file carries, sampled per chain."""

    path: Path
    chains: Mapping[str, frozenset[str]]
    truncated: bool = False

    @property
    def atoms(self) -> frozenset[str]:
        """Return every atom name the sample saw, across all chains."""
        union: set[str] = set()
        for names in self.chains.values():
            union |= set(names)
        return frozenset(union)

    def chains_missing(self, required: Iterable[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return each sampled chain that lacks a required atom, with what it lacks."""
        wanted = {str(atom).strip().upper() for atom in required if str(atom).strip()}
        short: list[tuple[str, tuple[str, ...]]] = []
        for chain_id in sorted(self.chains):
            missing = wanted - set(self.chains[chain_id])
            if missing:
                short.append((chain_id, order_atoms(missing)))
        return tuple(short)


def _read_text(path: Path) -> str:
    try:
        if path.name.endswith(".gz"):
            return gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise BackboneShapeError(f"cannot read structure file: {path}") from exc


def _sample(
    records: Iterable[tuple[str, str, str]],
) -> tuple[dict[str, frozenset[str]], bool]:
    """Collect atom names from a bounded sample of (chain, residue, atom) records.

    The sample keeps the atom names of the first `SAMPLE_RESIDUES_PER_CHAIN`
    residues of every chain. A structure writer emits the same atom template for
    every residue of a chain, so the first residues of a chain carry the atom set
    the whole chain carries, and reading the rest changes no verdict. Skipping
    them is what keeps this cheap: no coordinate is parsed and no array is built.

    The sample is per chain, because a generator can write a full-backbone
    target beside a C-alpha binder, recorded at
    `skills/claude-binder-lane/references/tool-catalogue.md` under "A hosted
    generator can swap the chains". A whole-file union would hide the short
    chain.

    Every chain still has to be found, so the scan reads the file through to
    `SAMPLE_ATOM_RECORDS` records. That ceiling is 50000, above any binder design
    pose. A file that exceeds it sets `truncated`, and a chain that begins past
    the ceiling is not checked.
    """
    chains: dict[str, set[str]] = {}
    seen: dict[str, list[str]] = {}
    read = 0
    truncated = False
    for chain_id, residue_id, atom_name in records:
        read += 1
        if read > SAMPLE_ATOM_RECORDS:
            truncated = True
            break
        order = seen.setdefault(chain_id, [])
        if residue_id not in order:
            if len(order) >= SAMPLE_RESIDUES_PER_CHAIN:
                continue
            order.append(residue_id)
        chains.setdefault(chain_id, set()).add(atom_name)
    return {chain_id: frozenset(names) for chain_id, names in chains.items()}, truncated


def _pdb_records(text: str) -> Iterable[tuple[str, str, str]]:
    """Yield (chain, residue, atom) for first-model ATOM records of a PDB file.

    HETATM records are skipped. A ligand or a water carries no backbone, so
    counting one would report an atom set no polymer residue has.
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
        if not in_first_model or record != "ATOM  ":
            continue
        if len(line) < 27:
            continue
        atom_name = line[12:16].strip().upper()
        if not atom_name:
            continue
        chain_id = line[21:22].strip() or "_"
        residue_id = line[22:27].strip()
        if not residue_id:
            continue
        yield chain_id, residue_id, atom_name


def _mmcif_records(text: str, path: Path) -> Iterable[tuple[str, str, str]]:
    """Yield (chain, residue, atom) for first-model ATOM rows of an atom_site loop."""
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
        positions = {column: place for place, column in enumerate(columns)}
        if "_atom_site.label_atom_id" not in positions:
            raise BackboneShapeError(f"mmCIF atom_site loop names no atom column: {path}")
        atom_place = positions["_atom_site.label_atom_id"]
        group_place = positions.get("_atom_site.group_PDB")
        model_place = positions.get("_atom_site.pdbx_PDB_model_num")
        chain_place = positions.get(
            "_atom_site.auth_asym_id", positions.get("_atom_site.label_asym_id")
        )
        sequence_place = positions.get(
            "_atom_site.auth_seq_id", positions.get("_atom_site.label_seq_id")
        )
        if chain_place is None or sequence_place is None:
            raise BackboneShapeError(f"mmCIF atom_site loop names no chain or residue column: {path}")
        while index < len(lines):
            text_row = lines[index].strip()
            if not text_row or text_row.startswith("#") or text_row.startswith("_") or text_row == "loop_":
                break
            index += 1
            fields = shlex.split(text_row, comments=False, posix=True)
            if len(fields) != len(columns):
                raise BackboneShapeError(
                    f"mmCIF atom_site row has {len(fields)} fields and the loop names "
                    f"{len(columns)}: {path}"
                )
            if group_place is not None and fields[group_place] != "ATOM":
                continue
            if model_place is not None and fields[model_place] not in {"1", ".", "?"}:
                continue
            atom_name = fields[atom_place].strip().upper()
            chain_id = fields[chain_place]
            residue_id = fields[sequence_place]
            if not atom_name or chain_id in {".", "?"} or residue_id in {".", "?"}:
                continue
            yield chain_id, residue_id, atom_name
        return
    raise BackboneShapeError(f"structure file has no mmCIF atom_site loop: {path}")


def read_backbone_shape(path: Path | str) -> BackboneShape:
    """Return the atom names one PDB or mmCIF structure file carries.

    The format is chosen by suffix, then confirmed by content, so a `.pdb` file
    holding an mmCIF loop still reads.
    """
    structure = Path(path)
    text = _read_text(structure)
    name = structure.name[: -len(".gz")] if structure.name.endswith(".gz") else structure.name
    suffix = Path(name).suffix.lower()
    # An mmCIF atom_site row starts with one space after ATOM, and a PDB atom
    # record starts with at least two, so the two are told apart by content
    # wherever the suffix does not settle it.
    holds_pdb_atoms = any(line.startswith("ATOM  ") for line in text.splitlines())
    if suffix in {".cif", ".mmcif"} or ("_atom_site." in text and not holds_pdb_atoms):
        records = _mmcif_records(text, structure)
    else:
        records = _pdb_records(text)
    chains, truncated = _sample(records)
    if not chains:
        raise BackboneShapeError(f"structure file carries no polymer atom records: {structure}")
    return BackboneShape(path=structure, chains=chains, truncated=truncated)


def pose_shape_problem(
    path: Path | str,
    *,
    required_atoms: Iterable[str],
    consumer: str,
    producer: str | None = None,
) -> str | None:
    """Return why one structure file falls short of a consumer, or None when it does not."""
    required = order_atoms(required_atoms)
    if not required:
        return None
    shape = read_backbone_shape(path)
    short = shape.chains_missing(required)
    if not short:
        return None
    detail = "; ".join(f"chain {chain_id} lacks {_atom_text(missing)}" for chain_id, missing in short)
    origin = f" written by {producer}" if producer else ""
    return (
        f"{consumer} reads {_atom_text(required)} and the backbone at {shape.path}{origin} "
        f"carries {_atom_text(shape.atoms)}: {detail}"
    )


# ---------------------------------------------------------------------------
# Plan time: what the catalog declares
# ---------------------------------------------------------------------------


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    """Return the packaged tool catalog, or one at an explicit path."""
    source = Path(path) if path is not None else package_file("data", "catalog.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BackboneShapeError(f"tool catalog must hold a JSON object: {source}")
    return value


def catalog_tools(catalog: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return the catalog's tool entries keyed by tool id."""
    tools = catalog.get("tools")
    if not isinstance(tools, Mapping):
        raise BackboneShapeError("tool catalog has no tools object")
    return {str(key): value for key, value in tools.items() if isinstance(value, Mapping)}


def route_field(stage_category: Any) -> str | None:
    """Return the declaration field that decides a tool's place on the route."""
    category = str(stage_category)
    if category in PRODUCER_CATEGORIES:
        return PRODUCES
    if category in CONSUMER_CATEGORIES:
        return REQUIRES
    return None


def declared_atoms(entry: Mapping[str, Any], field: str) -> frozenset[str] | None:
    """Return the atom set a catalog entry states for one field, or None when unstated.

    A `null` value means the tool neither writes nor reads a backbone on that
    side, which is an empty set rather than an absence. The string `unknown`
    means the atom set is not settled, and returns None so no caller refuses on
    it.
    """
    declaration = entry.get(DECLARATION_FIELD)
    if not isinstance(declaration, Mapping) or field not in declaration:
        return None
    value = declaration[field]
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return frozenset(order_atoms(value))
    return None


def expected_status(entry: Mapping[str, Any]) -> str:
    """Return the status the entry's own fields imply, for the integrity check."""
    field = route_field(entry.get("stage_category"))
    if field is None:
        return NOT_IN_ROUTE
    return STATED if declared_atoms(entry, field) is not None else UNKNOWN


def tools_for_adapter(catalog: Mapping[str, Any], adapter_id: str) -> dict[str, Mapping[str, Any]]:
    """Return every catalog tool whose profile selection names one adapter id."""
    if not adapter_id or adapter_id == "__REQUIRED__":
        return {}
    matched: dict[str, Mapping[str, Any]] = {}
    for tool_id, entry in catalog_tools(catalog).items():
        selection = entry.get("profile_selection")
        if isinstance(selection, Mapping) and str(selection.get("adapter_id")) == adapter_id:
            matched[tool_id] = entry
    return matched


def adapter_atoms(
    catalog: Mapping[str, Any], adapter_id: str, field: str
) -> tuple[frozenset[str] | None, str]:
    """Return one adapter's stated atom set for a field, and the tool names behind it.

    Two catalog tools can select the same adapter id. `rfdiffusion` and
    `rfdiffusion3` both name `rfdiffusion-generator`, recorded at
    `skills/claude-binder-lane/references/tool-catalogue.md` under "The profile id
    does not say which RFdiffusion you get". When such a pair disagrees, or when either side is `unknown`, the atom set is None and
    no caller refuses on it.
    """
    matched = tools_for_adapter(catalog, adapter_id)
    if not matched:
        return None, adapter_id
    names = ", ".join(
        str(entry.get("display_name") or tool_id) for tool_id, entry in sorted(matched.items())
    )
    declared = [declared_atoms(entry, field) for entry in matched.values()]
    if any(item is None for item in declared) or len(set(declared)) != 1:
        return None, names
    return declared[0], names


# ---------------------------------------------------------------------------
# Plan time: what a profile declares about backbone shape
# ---------------------------------------------------------------------------


def _shape_text(shapes: Iterable[str]) -> str:
    ordered = sorted({str(shape) for shape in shapes if str(shape)})
    return ", ".join(ordered) if ordered else "no backbone shape"


def emitted_shape(entry: Mapping[str, Any]) -> str | None:
    """Return the backbone shape a generator entry names, or None when it names none.

    The string `unknown` comes back as itself, because an explicit refusal to
    state is not the same as saying nothing. `shape_verdict` tells the two apart.
    """
    value = entry.get(EMITS_FIELD)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def accepted_shapes(entry: Mapping[str, Any]) -> dict[str, tuple[str, ...]] | str | None:
    """Return a designer entry's shape map, the string `unknown`, or None when absent.

    Each value is the flag list that shape costs. An empty list is a shape the
    designer reads with no extra flag, which is how every MPNN designer declares a
    full backbone. A shape absent from the map is one the designer does not read,
    and an empty map accepts nothing. A malformed value comes back as None, so the
    atom sets decide rather than a map nobody wrote on purpose.
    """
    value = entry.get(ACCEPTS_FIELD)
    if isinstance(value, str):
        return UNKNOWN if value.strip() == UNKNOWN else None
    if not isinstance(value, Mapping):
        return None
    shapes: dict[str, tuple[str, ...]] = {}
    for shape, flags in value.items():
        if not isinstance(flags, list) or not all(isinstance(item, str) for item in flags):
            return None
        shapes[str(shape)] = tuple(flags)
    return shapes


def shape_verdict(
    shape: str | None, accepted: Mapping[str, tuple[str, ...]] | str | None
) -> str:
    """Return what the two shape declarations settle for one producer to consumer pairing.

    `unknown` on either side is an explicit refusal to state, and reports nothing.
    A declaration that is simply absent leaves the pairing undecided, and the
    `backbone_atoms` sets decide it instead, so silence buys no permission.
    """
    if shape == UNKNOWN or accepted == UNKNOWN:
        return SHAPE_UNSTATED
    if shape is None or not isinstance(accepted, Mapping):
        return SHAPE_UNDECIDED
    return SHAPE_ACCEPTED if shape in accepted else SHAPE_REFUSED


def _section_records(
    *sources: Mapping[str, Any] | None, section: tuple[str, str]
) -> dict[str, list[Mapping[str, Any]]]:
    """Return one profile section's entries, keyed by the adapter id each names."""
    block_name, key = section
    found: dict[str, list[Mapping[str, Any]]] = {}
    for source in sources:
        if source is None:
            continue
        block = source.get(block_name)
        if not isinstance(block, Mapping):
            continue
        entries = block.get(key)
        if not isinstance(entries, list):
            continue
        for item in entries:
            if isinstance(item, Mapping) and item.get("adapter_id"):
                found.setdefault(str(item["adapter_id"]), []).append(item)
    return found


def _agreed(entries: Iterable[Mapping[str, Any]], reader: Any) -> Any:
    """Return the one declaration a set of entries shares, or None when they differ.

    A campaign and the profile it resolved from both carry the same entry, and two
    entries can bind one adapter id. A disagreement is unresolved rather than a
    refusal, matching what `adapter_atoms` does with a shared adapter id.
    """
    values = [reader(entry) for entry in entries]
    if not values:
        return None
    first = values[0]
    for value in values[1:]:
        if value != first:
            return None
    return first


# ---------------------------------------------------------------------------
# Plan time: the route through a campaign
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteProblem:
    """One sequence-design stage fed by a generator that writes too few atoms."""

    consumer_stage: str
    producer_stage: str
    consumer: str
    producer: str
    produced: tuple[str, ...]
    required: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def field(self) -> str:
        return f"stages[{self.consumer_stage}].backbone_route"

    @property
    def problem(self) -> str:
        return (
            f"routes {self.producer} into {self.consumer}: {self.producer} writes "
            f"{_atom_text(self.produced)} and {self.consumer} reads "
            f"{_atom_text(self.required)}, so {_atom_text(self.missing)} would be absent"
        )

    @property
    def fix(self) -> str:
        return (
            f"generate backbones for {self.consumer_stage} with a tool that writes "
            f"{_atom_text(self.required)}, or bind {self.consumer} to a checkpoint that "
            f"reads {_atom_text(self.produced)}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "consumer_stage": self.consumer_stage,
            "producer_stage": self.producer_stage,
            "consumer": self.consumer,
            "producer": self.producer,
            "produced": list(self.produced),
            "required": list(self.required),
            "missing": list(self.missing),
            "field": self.field,
            "problem": self.problem,
            "fix": self.fix,
        }


@dataclass(frozen=True)
class ShapeRouteProblem:
    """One sequence-design stage fed by a generator whose backbone shape it refuses.

    The atom sets cannot express this. Vanilla ProteinMPNN and SolubleMPNN read
    the same four atoms and differ on a C-alpha trace, because only one of the two
    has a C-alpha checkpoint. The declaration carries that difference and this
    problem reports it.
    """

    consumer_stage: str
    producer_stage: str
    consumer: str
    producer: str
    shape: str
    accepted: tuple[str, ...]

    @property
    def field(self) -> str:
        return f"stages[{self.consumer_stage}].backbone_route"

    @property
    def problem(self) -> str:
        return (
            f"routes {self.producer} into {self.consumer}: {self.producer} writes a "
            f"{self.shape} backbone and {self.consumer} declares no route for it, "
            f"accepting {_shape_text(self.accepted)}"
        )

    @property
    def fix(self) -> str:
        return (
            f"generate backbones for {self.consumer_stage} with a tool that writes "
            f"{_shape_text(self.accepted)}, or bind {self.consumer_stage} to a designer "
            f"whose {ACCEPTS_FIELD} names {self.shape}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "consumer_stage": self.consumer_stage,
            "producer_stage": self.producer_stage,
            "consumer": self.consumer,
            "producer": self.producer,
            "shape": self.shape,
            "accepted": list(self.accepted),
            "field": self.field,
            "problem": self.problem,
            "fix": self.fix,
        }


def _stage_records(*sources: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    for source in sources:
        if source is None:
            continue
        stages = source.get("stages")
        if isinstance(stages, list):
            records = [item for item in stages if isinstance(item, Mapping)]
            if records:
                return records
    return []


def _adapter_records(*sources: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    merged: dict[str, Mapping[str, Any]] = {}
    for source in sources:
        if source is None:
            continue
        adapters = source.get("adapters")
        if not isinstance(adapters, list):
            continue
        for item in adapters:
            if isinstance(item, Mapping) and item.get("adapter_id"):
                merged[str(item["adapter_id"])] = item
    return merged


def _declared_backbone_stage(adapter: Mapping[str, Any] | None) -> str | None:
    """Return the stage id a designer's argv names as its backbone source."""
    if adapter is None:
        return None
    command = adapter.get("command_argv_template")
    if not isinstance(command, list):
        return None
    for place, token in enumerate(command):
        if token == BACKBONE_STAGE_FLAG and place + 1 < len(command):
            value = command[place + 1]
            if isinstance(value, str) and value and not value.startswith("{{"):
                return value
    return None


def route_problems(
    campaign: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None = None,
    *,
    catalog: Mapping[str, Any] | None = None,
) -> list[RouteProblem | ShapeRouteProblem]:
    """Return every sequence-design stage whose generator writes a backbone it cannot read.

    Each producer to consumer pairing is settled by the shape declarations when
    both ends carry one, and by the catalog's atom sets when either end carries
    none. The shape declarations come first, because a designer that accepts a
    reduced backbone reads fewer atoms than the catalog records for it, and the
    atom comparison alone would refuse a route that runs.
    """
    tools = catalog if catalog is not None else load_catalog()
    stages = _stage_records(campaign, profile)
    if not stages:
        return []
    adapters = _adapter_records(profile, campaign)
    generators = _section_records(profile, campaign, section=GENERATOR_SECTION)
    designers = _section_records(profile, campaign, section=DESIGNER_SECTION)
    by_id = {str(stage.get("stage_id")): stage for stage in stages if stage.get("stage_id")}
    problems: list[RouteProblem | ShapeRouteProblem] = []
    for stage_id, stage in by_id.items():
        adapter_id = str(stage.get("adapter_id") or "")
        required, consumer = adapter_atoms(tools, adapter_id, REQUIRES)
        accepted = _agreed(designers.get(adapter_id, ()), accepted_shapes)
        if not required and accepted is None:
            continue
        for producer_stage in _producer_stages(stage, by_id, adapters.get(adapter_id), tools):
            producer_adapter = str(by_id[producer_stage].get("adapter_id") or "")
            produced, producer = adapter_atoms(tools, producer_adapter, PRODUCES)
            shape = _agreed(generators.get(producer_adapter, ()), emitted_shape)
            verdict = shape_verdict(shape, accepted)
            if verdict in {SHAPE_ACCEPTED, SHAPE_UNSTATED}:
                continue
            if verdict == SHAPE_REFUSED:
                problems.append(
                    ShapeRouteProblem(
                        consumer_stage=stage_id,
                        producer_stage=producer_stage,
                        consumer=consumer,
                        producer=producer,
                        shape=str(shape),
                        accepted=tuple(sorted(accepted)),
                    )
                )
                continue
            if not required or produced is None:
                continue
            missing = required - produced
            if not missing:
                continue
            problems.append(
                RouteProblem(
                    consumer_stage=stage_id,
                    producer_stage=producer_stage,
                    consumer=consumer,
                    producer=producer,
                    produced=order_atoms(produced),
                    required=order_atoms(required),
                    missing=order_atoms(missing),
                )
            )
    return problems


def _producer_stages(
    stage: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
    adapter: Mapping[str, Any] | None,
    tools: Mapping[str, Any],
) -> list[str]:
    """Return the stages whose backbones one sequence-design stage reads.

    The designer's argv settles it when the profile pins `--backbone-stage-id`,
    because that flag names the stage the wrapper loads its manifest from. A
    profile that leaves the flag templated falls back to the declared
    dependencies that bind a backbone or co-design generator.
    """
    declared = _declared_backbone_stage(adapter)
    if declared is not None and declared in by_id:
        return [declared]
    dependencies = stage.get("depends_on")
    if not isinstance(dependencies, list):
        return []
    found: list[str] = []
    for value in dependencies:
        upstream = by_id.get(str(value))
        if upstream is None or str(value) in found:
            continue
        upstream_adapter = str(upstream.get("adapter_id") or "")
        for entry in tools_for_adapter(tools, upstream_adapter).values():
            if route_field(entry.get("stage_category")) == PRODUCES:
                found.append(str(value))
                break
    return found


# ---------------------------------------------------------------------------
# Run time: what a consumer requires
# ---------------------------------------------------------------------------


_CATALOG_CACHE: dict[str, Any] | None = None


def _cached_catalog() -> dict[str, Any] | None:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        try:
            _CATALOG_CACHE = load_catalog()
        except (OSError, ValueError):
            return None
    return _CATALOG_CACHE


def required_atoms_for_tool(tool_id: str, *, catalog: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Return the backbone atoms one catalog tool reads, empty when unstated.

    The run-time check takes a tool id rather than an adapter id, because a
    designer wrapper receives its own `--designer-id` and that value matches the
    catalog key.
    """
    tools = catalog if catalog is not None else _cached_catalog()
    if tools is None:
        return ()
    entry = catalog_tools(tools).get(str(tool_id))
    if entry is None:
        return ()
    atoms = declared_atoms(entry, REQUIRES)
    return order_atoms(atoms) if atoms else ()
