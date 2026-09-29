#!/usr/bin/env python3
"""Write a target's residue map, and its campaign site block, from a structure.

A campaign points `targets[].site.residue_map_path` at a JSON document. Until
this script existed, that document had to be typed by hand. This writes it from
the structure the campaign already pins, so the labels in the map are the labels
the structure actually carries.

What the document has to look like
----------------------------------

The packaged local-contract residue map carries exactly two top-level keys,
`schema_version` and `source_to_cleaned`. This script writes those two and
nothing else.

`source_to_cleaned` is the only key any code reads.
`adapters/esmfold2_predictor.py` `site_residue_map_for` pulls it out of the
document and hands it to `binder_metrics.compute_site_metrics`, which applies it
to the site and hotspot labels before matching them against the predicted pose.
`claude_binder.lane` hashes the whole file and stages it into the bundle. It
reads no other residue-map field.

Why the mapping this script writes is the identity
--------------------------------------------------

`source_to_cleaned` translates a label in the deposited structure into the label
the same residue carries in the cleaned structure handed to the predictors. One
structure file is one numbering, so there is no second numbering here to derive
a translation from. The packaged local-contract residue map is an identity map.
This script writes the identity over every residue of the named chain, which is
correct whenever the campaign's `structure_path` is already the cleaned
structure. A campaign that renumbers between deposit and fold needs a map this
script cannot produce and the configuration must provide that map explicitly.

Site inputs
-----------

The explicit route takes `reference_contact_residues` from the caller's residue
list. The partner-complex route takes a target accession and a natural partner
name, then derives those same labels from one deposited complex. It refuses an
absent or ambiguous complex. It never chooses a surface from the target alone.

Two site fields cannot come from a structure at all. `design_residues` is the
design region and `contact_cutoff_angstrom` is a campaign policy number. The
explicit-site route leaves a visible TODO for either omitted value. The
partner-complex route defaults the contact cutoff to 5 Angstrom and records a
different finite positive value when the scientist supplies one.

`--map-only` writes just the residue map from `--structure` and `--chain`; it
does not require or write a site artifact.

Exit code. Zero means every requested file was written. One means an input was
missing or did not match the structure, and the message says which.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from .adapters import binder_metrics
from . import partner_site

# The value both residue maps in this tree carry. No code reads it, so it is
# copied from those files rather than derived from a schema.
SCHEMA_VERSION = 1

# `binder_metrics.compute_site_metrics` raises on any other value, so this is
# the only selection the metric implements.
ATOM_SELECTION = "heavy-atoms"

# Read from Claude Binder's `site.mode` validation.
SITE_MODES = (
    "explicit-residues",
    "reference-partner-contacts",
    "reference-pose-contacts",
    "spatial-pocket",
)

# The tree's own marker for a field a person still has to fill in.
# The package's required-placeholder validator matches it.
TODO_DESIGN_RESIDUES = "__REQUIRED__ design region, as CHAIN:START-END"
TODO_CONTACT_CUTOFF = "__REQUIRED__ contact cutoff in Angstrom, a campaign policy number"

CIF_SUFFIXES = {".cif", ".mmcif"}
PDB_SUFFIXES = {".pdb", ".ent"}

# The mmCIF `_atom_site` columns `binder_metrics.parse_cif_atoms` reads.
ATOM_SITE_FIELDS = (
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_comp_id",
    "label_asym_id",
    "label_seq_id",
    "auth_seq_id",
    "auth_asym_id",
    "auth_comp_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "B_iso_or_equiv",
    "pdbx_PDB_model_num",
)


# One residue label is CHAIN:NUMBER with an optional insertion code, the form
# `binder_metrics._expand_residue_labels` returns. `adapters/target_prep_adapter.py`
# matches the same shape.
RESIDUE_LABEL_RE = re.compile(r"^([A-Za-z0-9]+):(-?\d+)([A-Za-z]?)$")


class InputError(Exception):
    """An argument or a structure cannot support the files this script writes."""


def _cif_value(text: str) -> str:
    """Return one mmCIF token, using `.` for an absent value.

    `parse_cif_atoms` cuts rows by counting values, so a field left out does not
    shorten one row, it shifts every row after it. An empty field has to be
    written rather than omitted. The parser reads `.` back as absent through its
    own `_clean`.
    """
    stripped = text.strip()
    if not stripped:
        return "."
    return stripped.replace(" ", "_")


def pdb_to_atom_site_cif(path: Path) -> str:
    """Rewrite a PDB file's ATOM and HETATM records as an mmCIF `_atom_site` loop.

    This exists so there is one residue parser rather than two.
    `binder_metrics.parse_cif_atoms` already decides which model to keep, which
    records to drop, how to fall back from `auth_` to `label_` fields, and how to
    group atoms into residues. Handing it mmCIF text keeps all four decisions in
    that function.

    Column positions follow the PDB format's fixed-width ATOM record. Every
    value is passed through, including HETATM records, because `parse_cif_atoms`
    is the one that drops them.
    """
    rows: list[str] = []
    model = "1"
    serial = 0
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("MODEL "):
            model = _cif_value(line[10:14])
            continue
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if len(line) < 54:
            # Shorter than the coordinate columns, so it carries no position.
            continue
        serial += 1
        atom_name = _cif_value(line[12:16])
        comp_id = _cif_value(line[17:20])
        chain_id = _cif_value(line[21:22])
        seq_id = _cif_value(line[22:26])
        ins_code = _cif_value(line[26:27])
        element = _cif_value(line[76:78]) if len(line) >= 78 else "."
        values = [
            "ATOM" if line.startswith("ATOM  ") else "HETATM",
            str(serial),
            element,
            atom_name,
            comp_id,
            chain_id,
            seq_id,
            seq_id,
            chain_id,
            comp_id,
            ins_code,
            _cif_value(line[30:38]),
            _cif_value(line[38:46]),
            _cif_value(line[46:54]),
            _cif_value(line[60:66]) if len(line) >= 66 else ".",
            model,
        ]
        rows.append(" ".join(values))
    if not rows:
        raise InputError(f"{path} has no ATOM or HETATM record with coordinates")
    header = "\n".join(f"_atom_site.{field}" for field in ATOM_SITE_FIELDS)
    return "data_converted\nloop_\n" + header + "\n" + "\n".join(rows) + "\n#\n"


def read_structure(path: Path) -> binder_metrics.Structure:
    """Parse a CIF or a PDB into the residue list the metric module uses."""
    if not path.is_file():
        raise InputError(f"structure does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in CIF_SUFFIXES:
        return binder_metrics.parse_cif_atoms(path, argument="structure")
    if suffix in PDB_SUFFIXES:
        return binder_metrics.parse_cif_atoms(
            pdb_to_atom_site_cif(path), argument="structure"
        )
    raise InputError(
        f"unsupported structure suffix {suffix!r}. "
        f"CIF is {sorted(CIF_SUFFIXES)} and PDB is {sorted(PDB_SUFFIXES)}"
    )


def chain_labels(structure: binder_metrics.Structure, chain_id: str) -> list[str]:
    """Return one chain's residue labels, in the order the structure gives them.

    `Structure.chain_indices` matches the author chain first and the label chain
    second, which is the rule every metric in `binder_metrics` already applies,
    and it raises with the chains present when neither matches.
    """
    labels: list[str] = []
    for residue in structure.chain_residues(chain_id):
        if residue.label not in labels:
            labels.append(residue.label)
    return labels


def require_declared_chains(
    structure: binder_metrics.Structure, declared_chain_ids: list[str], label: str
) -> list[str]:
    """Enumerate observed chains and reject a site declaration that names another chain."""
    observed = structure.chain_ids()
    missing = [chain_id for chain_id in declared_chain_ids if chain_id not in observed]
    if missing:
        raise InputError(
            f"{label} declares chains absent from the structure: {missing}. "
            f"Observed chains: {observed}"
        )
    return observed


def read_surface_entries(value: str) -> list[str]:
    """Return the caller's residue entries, from a literal list or from a file.

    A path is read as one entry per line, with `#` starting a comment. Anything
    else is split on commas. Both forms take `CHAIN:NUMBER` and
    `CHAIN:START-END`, the two forms `binder_metrics._expand_residue_labels`
    accepts.
    """
    path = Path(value)
    if path.is_file():
        text = path.read_text(errors="replace")
        raw = []
        for line in text.splitlines():
            line = line.split("#", 1)[0]
            raw.extend(line.split(","))
    else:
        raw = value.split(",")
    entries = [item.strip() for item in raw if item.strip()]
    if not entries:
        raise InputError(f"no residue entry in {value!r}")
    return entries


def _residue_sort_key(label: str) -> tuple[int, str]:
    """Order a residue label by its number, then by its insertion code.

    Sorting on the number alone raised a bare ValueError on `A:52A`, a label
    every residue pattern in the tree accepts. The insertion code orders after
    the bare number, which is the order the deposited file carries.
    """
    match = RESIDUE_LABEL_RE.fullmatch(label)
    if match is None:
        raise InputError(f"residue label is not CHAIN:NUMBER: {label!r}")
    return int(match.group(2)), match.group(3)


def expand_surface(
    entries: list[str],
    chain_id: str,
    present: list[str],
) -> list[str]:
    """Expand the caller's entries against the chain, refusing anything absent.

    Expansion reuses `binder_metrics._expand_residue_labels`, so the site labels
    this script writes are expanded by the same code that later matches them
    against a predicted pose. That function drops an entry on another chain
    without saying so, so each entry is expanded on its own and an entry that
    contributes nothing is reported here.
    """
    known = set(present)
    expanded: list[str] = []
    for entry in entries:
        try:
            labels = binder_metrics._expand_residue_labels([entry], chain_id)
        except binder_metrics.MetricInputError as exc:
            raise InputError(f"surface entry {entry!r}: {exc}") from exc
        if not labels:
            raise InputError(
                f"surface entry {entry!r} names no residue on chain {chain_id!r}"
            )
        missing = sorted(labels - known)
        if missing:
            raise InputError(
                f"surface entry {entry!r} names residues the structure does not carry "
                f"on chain {chain_id!r}: {missing}"
            )
        for label in sorted(labels, key=_residue_sort_key):
            if label not in expanded:
                expanded.append(label)
    return expanded


def residue_map_document(labels: list[str]) -> dict[str, Any]:
    """Return the residue map, with the two keys the tree's own maps carry."""
    return {
        "schema_version": SCHEMA_VERSION,
        "source_to_cleaned": {label: label for label in labels},
    }


def site_block(
    *,
    mode: str,
    surface: list[str],
    residue_map_path: str,
    design_residues: list[str] | None,
    contact_cutoff: float | None,
    resolution_artifact_path: str | None = None,
    partner_complex: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the `targets[].site` object, with a TODO where a value is unknown."""
    block = {
        "mode": mode,
        "design_residues": design_residues or [TODO_DESIGN_RESIDUES],
        "reference_contact_residues": surface,
        "contact_cutoff_angstrom": (
            contact_cutoff if contact_cutoff is not None else TODO_CONTACT_CUTOFF
        ),
        "atom_selection": ATOM_SELECTION,
        "residue_map_path": residue_map_path,
    }
    if resolution_artifact_path is not None:
        block["resolution_artifact_path"] = resolution_artifact_path
    if partner_complex is not None:
        block["partner_complex"] = partner_complex
    return block


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_target_inputs.py",
        description=(
            "Write a target's residue-map.json, and the campaign site block that "
            "points at it, from a structure file."
        ),
    )
    parser.add_argument("--structure", help="target CIF or PDB for an explicit site")
    parser.add_argument(
        "--chain",
        help="author chain identifier of the design target, such as A",
    )
    parser.add_argument(
        "--surface-residues",
        help=(
            "the site, as CHAIN:NUMBER and CHAIN:START-END entries separated by "
            "commas, or a path to a file carrying them one per line. Use this "
            "for an explicit site. It cannot be combined with --partner."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        help="path to write the residue map to",
    )
    parser.add_argument(
        "--site-out",
        help="path to write the campaign site block to. Defaults to <out>.site.json",
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help=(
            "write only the identity residue map from --structure and --chain; "
            "do not require or write a site block"
        ),
    )
    parser.add_argument(
        "--site-mode",
        default=None,
        choices=SITE_MODES,
        help=(
            "targets[].site.mode. Defaults to explicit-residues for a supplied "
            "residue list and reference-partner-contacts for --partner"
        ),
    )
    parser.add_argument(
        "--design-residues",
        help=(
            "targets[].site.design_residues, in the same form as the surface. "
            "Omitted leaves a TODO placeholder that fails validate"
        ),
    )
    parser.add_argument(
        "--contact-cutoff-angstrom",
        type=float,
        help=(
            "targets[].site.contact_cutoff_angstrom. On the explicit-site route, "
            "omission leaves a TODO placeholder that fails validate. On the "
            "partner route, omission defaults to 5.0"
        ),
    )
    parser.add_argument(
        "--partner",
        help=(
            "natural binding partner in words. This activates deposited-complex "
            "site resolution and cannot be combined with --surface-residues"
        ),
    )
    parser.add_argument(
        "--target-accession",
        help="target UniProt accession used to identify the deposited target entity",
    )
    parser.add_argument(
        "--pdb-entry",
        help=(
            "optional RCSB PDB entry ID. Supply this when several complexes match "
            "the target and partner."
        ),
    )
    parser.add_argument(
        "--partner-chain",
        action="append",
        default=[],
        help=(
            "author chain identifier for the partner. Repeat for every partner "
            "chain. Required with --complex-structure."
        ),
    )
    parser.add_argument(
        "--complex-structure",
        help=(
            "verified local CIF or PDB complex. It bypasses RCSB lookup and still "
            "records whether its coordinates are experimental or predicted."
        ),
    )
    parser.add_argument(
        "--complex-kind",
        choices=("experimental", "predicted"),
        help="coordinate provenance for --complex-structure",
    )
    parser.add_argument(
        "--complex-out",
        help=(
            "path to write the complex used for partner contacts. Defaults to a "
            "file beside --out. Use this path as targets[].structure_path."
        ),
    )
    parser.add_argument(
        "--resolution-out",
        help=(
            "path to write the partner-site resolution artifact. Defaults to a "
            "file beside --site-out."
        ),
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    return parser


def _relative_path(path: Path, directory: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), directory.resolve())
    except ValueError:
        return str(path.resolve())


def _site_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    out_path = Path(args.out)
    site_path = Path(args.site_out) if args.site_out else out_path.with_suffix(".site.json")
    return out_path, site_path


def _map_only(args: argparse.Namespace, out_path: Path) -> dict[str, Any]:
    if not args.structure:
        raise InputError("--structure is required with --map-only")
    if not args.chain:
        raise InputError("--chain is required with --map-only")
    conflicting = [
        option
        for option, supplied in (
            ("--site-out", args.site_out is not None),
            ("--site-mode", args.site_mode is not None),
            ("--surface-residues", args.surface_residues is not None),
            ("--design-residues", args.design_residues is not None),
            ("--contact-cutoff-angstrom", args.contact_cutoff_angstrom is not None),
            ("--partner", args.partner is not None),
            ("--target-accession", args.target_accession is not None),
            ("--pdb-entry", args.pdb_entry is not None),
            ("--partner-chain", bool(args.partner_chain)),
            ("--complex-structure", args.complex_structure is not None),
            ("--complex-kind", args.complex_kind is not None),
            ("--complex-out", args.complex_out is not None),
            ("--resolution-out", args.resolution_out is not None),
        )
        if supplied
    ]
    if conflicting:
        raise InputError(
            "--map-only cannot be combined with site or partner options: "
            + ", ".join(conflicting)
        )

    structure_path = Path(args.structure)
    structure = read_structure(structure_path)
    require_declared_chains(structure, [args.chain], "residue map")
    present = chain_labels(structure, args.chain)
    if not present:
        raise InputError(f"chain {args.chain!r} carries no residue in {structure_path}")
    write_json(out_path, residue_map_document(present))
    return {
        "ok": True,
        "structure": str(structure_path.resolve()),
        "chain": args.chain,
        "chains_present": structure.chain_ids(),
        "residue_count": len(present),
        "first_residue": present[0],
        "last_residue": present[-1],
        "mapping": "identity",
        "residue_map": str(out_path.resolve()),
        "todo_fields": [],
    }


def _explicit_site(args: argparse.Namespace, out_path: Path, site_path: Path) -> dict[str, Any]:
    if not args.structure:
        raise InputError("--structure is required with --surface-residues")
    if not args.chain:
        raise InputError("--chain is required with --surface-residues")
    structure_path = Path(args.structure)
    structure = read_structure(structure_path)
    require_declared_chains(structure, [args.chain], "explicit site")
    present = chain_labels(structure, args.chain)
    if not present:
        raise InputError(f"chain {args.chain!r} carries no residue in {structure_path}")

    surface = expand_surface(
        read_surface_entries(args.surface_residues), args.chain, present
    )
    design_residues = (
        expand_surface(read_surface_entries(args.design_residues), args.chain, present)
        if args.design_residues
        else None
    )

    pointer = _relative_path(out_path, site_path.parent)

    write_json(out_path, residue_map_document(present))
    write_json(
        site_path,
        site_block(
            mode=args.site_mode or "explicit-residues",
            surface=surface,
            residue_map_path=pointer,
            design_residues=design_residues,
            contact_cutoff=args.contact_cutoff_angstrom,
        ),
    )

    todo = []
    if design_residues is None:
        todo.append("site.design_residues")
    if args.contact_cutoff_angstrom is None:
        todo.append("site.contact_cutoff_angstrom")
    return {
        "ok": True,
        "structure": str(structure_path.resolve()),
        "chain": args.chain,
        "chains_present": structure.chain_ids(),
        "residue_count": len(present),
        "first_residue": present[0],
        "last_residue": present[-1],
        "mapping": "identity",
        "surface_residue_count": len(surface),
        "residue_map": str(out_path.resolve()),
        "site_block": str(site_path.resolve()),
        "todo_fields": todo,
    }


def _partner_complex(args: argparse.Namespace) -> partner_site.LocatedComplex:
    if not args.partner:
        raise InputError("--partner is required for partner-complex site resolution")
    if not args.target_accession:
        raise InputError("--target-accession is required for partner-complex site resolution")
    specification = partner_site.PartnerComplexSpec(
        target_accession=args.target_accession,
        partner_name=args.partner,
        entry_id=args.pdb_entry,
        target_chain=args.chain,
        partner_chains=tuple(args.partner_chain),
    )
    if not args.complex_structure:
        return partner_site.RcsbComplexLocator().locate(specification)
    if not args.chain:
        raise InputError("--chain is required with --complex-structure")
    if not args.partner_chain:
        raise InputError("--partner-chain is required with --complex-structure")
    if not args.complex_kind:
        raise InputError("--complex-kind is required with --complex-structure")
    path = Path(args.complex_structure)
    structure = read_structure(path)
    require_declared_chains(
        structure,
        [args.chain, *args.partner_chain],
        "partner-complex site",
    )
    return partner_site.local_complex(
        structure=structure,
        structure_id=args.pdb_entry or path.stem,
        target_chain=args.chain,
        partner_chains=args.partner_chain,
        structure_status=args.complex_kind,
        source="user-supplied local complex",
        source_url=None,
        coordinate_text=path.read_text(encoding="utf-8"),
    )


def _partner_site(args: argparse.Namespace, out_path: Path, site_path: Path) -> dict[str, Any]:
    if args.site_mode not in (None, "reference-partner-contacts"):
        raise InputError("--partner requires --site-mode reference-partner-contacts")
    contact_cutoff = (
        partner_site.PARTNER_CONTACT_CUTOFF_ANGSTROM
        if args.contact_cutoff_angstrom is None
        else args.contact_cutoff_angstrom
    )
    located = _partner_complex(args)
    specification = partner_site.PartnerComplexSpec(
        target_accession=args.target_accession,
        partner_name=args.partner,
        entry_id=args.pdb_entry,
        target_chain=args.chain,
        partner_chains=tuple(args.partner_chain),
    )
    resolution = partner_site.resolve_partner_site(
        specification,
        _SingleComplexLocator(located),
        cutoff_angstrom=contact_cutoff,
    )
    present = chain_labels(located.structure, resolution.target_chain)
    surface = list(resolution.contact_residues)
    design_residues = (
        expand_surface(
            read_surface_entries(args.design_residues), resolution.target_chain, present
        )
        if args.design_residues
        else surface
    )
    if located.coordinate_text is None:
        raise InputError("partner complex coordinates are unavailable for the campaign input")

    # The copy is the source file's own text, so the default name has to keep the
    # source suffix. `adapters/target_prep_adapter.py` dispatches on suffix, so a
    # PDB written to a `.cif` name is unreadable at the stage that consumes it as
    # `targets[].structure_path`. RCSB always returns mmCIF.
    source_suffix = (
        Path(args.complex_structure).suffix.lower() if args.complex_structure else ".cif"
    )
    complex_path = (
        Path(args.complex_out)
        if args.complex_out
        else out_path.with_name(f"{out_path.stem}.reference-complex{source_suffix}")
    )
    resolution_path = (
        Path(args.resolution_out)
        if args.resolution_out
        else site_path.with_name(f"{site_path.stem}.resolution.json")
    )
    pointer = _relative_path(out_path, site_path.parent)
    resolution_pointer = _relative_path(resolution_path, site_path.parent)
    write_json(out_path, residue_map_document(present))
    complex_path.parent.mkdir(parents=True, exist_ok=True)
    complex_path.write_text(located.coordinate_text, encoding="utf-8")
    resolution_document = resolution.artifact()
    if contact_cutoff != partner_site.PARTNER_CONTACT_CUTOFF_ANGSTROM:
        resolution_document["contact_cutoff_source"] = (
            "operator-supplied --contact-cutoff-angstrom campaign policy; "
            "not the published 5 Angstrom default"
        )
    write_json(resolution_path, resolution_document)
    write_json(
        site_path,
        site_block(
            mode="reference-partner-contacts",
            surface=surface,
            residue_map_path=pointer,
            design_residues=design_residues,
            contact_cutoff=contact_cutoff,
            resolution_artifact_path=resolution_pointer,
            partner_complex={
                "target_accession": resolution.target_accession,
                "partner_name": resolution.partner_name,
                "structure_id": resolution.structure_id,
                "target_chain": resolution.target_chain,
                "partner_chains": list(resolution.partner_chains),
                # This block records how the site was resolved. No adapter reads
                # it, so mark it terminal or the linker refuses the run.
                "link_terminal": True,
            },
        ),
    )
    return {
        "ok": True,
        "structure": str(complex_path.resolve()),
        "chain": resolution.target_chain,
        "chains_present": located.structure.chain_ids(),
        "residue_count": len(present),
        "first_residue": present[0],
        "last_residue": present[-1],
        "mapping": "identity",
        "surface_residue_count": len(surface),
        "residue_map": str(out_path.resolve()),
        "site_block": str(site_path.resolve()),
        "resolution_artifact": str(resolution_path.resolve()),
        "structure_id": resolution.structure_id,
        "structure_status": resolution.structure_status,
        "partner_chains": list(resolution.partner_chains),
        "confidence": resolution_document["confidence"],
        "contact_cutoff_angstrom": contact_cutoff,
        "todo_fields": [],
    }


class _SingleComplexLocator:
    """Adapt an already selected local or RCSB complex to the resolver protocol."""

    def __init__(self, located: partner_site.LocatedComplex):
        self.located = located

    def locate(self, specification: partner_site.PartnerComplexSpec) -> partner_site.LocatedComplex:
        return self.located


def run(args: argparse.Namespace) -> dict[str, Any]:
    cutoff = args.contact_cutoff_angstrom
    if cutoff is not None and (not math.isfinite(cutoff) or cutoff <= 0):
        raise InputError("contact cutoff must be a finite positive number")
    if args.map_only:
        return _map_only(args, Path(args.out))
    out_path, site_path = _site_paths(args)
    if args.surface_residues and args.partner:
        raise InputError("use either --surface-residues or --partner, not both")
    if args.partner:
        return _partner_site(args, out_path, site_path)
    if not args.surface_residues:
        raise InputError(
            "no site was supplied. Pass --surface-residues with target residues, or "
            "pass --target-accession and --partner to derive them from a deposited complex."
        )
    return _explicit_site(args, out_path, site_path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (InputError, binder_metrics.MetricInputError, partner_site.SiteResolutionError) as exc:
        print(f"make_target_inputs: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"structure         {summary['structure']}")
        print(f"chain             {summary['chain']} of {summary['chains_present']}")
        print(
            f"residues          {summary['residue_count']} "
            f"({summary['first_residue']} to {summary['last_residue']})"
        )
        print(f"source_to_cleaned {summary['mapping']}")
        print(f"residue map       {summary['residue_map']}")
        if "site_block" in summary:
            print(f"site residues     {summary['surface_residue_count']}")
            print(f"site block        {summary['site_block']}")
        if "resolution_artifact" in summary:
            print(f"resolution record {summary['resolution_artifact']}")
            print(f"complex           {summary['structure_id']} ({summary['structure_status']})")
            print(f"partner chains    {summary['partner_chains']}")
            print(f"confidence        {summary['confidence']['level']}")
        for field in summary["todo_fields"]:
            print(f"TODO              {field} is a placeholder and fails validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
