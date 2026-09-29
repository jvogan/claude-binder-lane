#!/usr/bin/env python3
"""Build and score the configured control panel.

The builder materializes configured controls under the stage attempt directory,
constructs sequence and cross-pair negatives from existing PDB pairs, and sends
the resulting control manifest through every registered uniform-rescore arm.
The predictor arms write prediction artifacts and measurement sources through
``binder_contract``. This module joins their raw rows into the control
calibration table.

The builder refuses before model execution when a required structure, control
source, or deterministic shuffle seed is absent. It never creates a native
positive complex because that structure must come from the configured panel.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from claude_binder import lane
from claude_binder.arms import score_instrument_arm_name
from claude_binder.argv_template import render_argv


SHA256_LENGTH = 64
REQUIRED_TOKEN = "__REQUIRED__"
CONTROL_PHASE = "uniform-rescore"
PUBLISHED_CONTROL_PANEL_GUIDANCE = (
    "The published campaign uses native complexes such as Barnase/Barstar for the "
    "positive panel. It uses a non-interacting pair, a sequence-shuffled binder, and "
    "a cross-pair mismatch for the negative panel."
)
CONSTRUCTED_PAIR_ROLES = {"matched-wrong-pair", "non-interacting-pair"}
SOURCE_CONTROL_FIELDS = (
    "source_control_id",
    "source_target_control_id",
    "target_source_control_id",
    "source_binder_control_id",
    "binder_source_control_id",
)
AMINO_ACID_3_TO_1 = {
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
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AMINO_ACID_1_TO_3 = {value: key for key, value in AMINO_ACID_3_TO_1.items()}


class AdapterError(RuntimeError):
    """An input or output condition that makes control calibration unsafe."""


@dataclass(frozen=True)
class ControlArtifact:
    """One materialized control structure and its predictor input sequence."""

    control_id: str
    control_type: str
    role: str
    structure_path: Path
    structure_sha256: str
    target_chain: str
    binder_chain: str
    binder_sequence: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"{label} is invalid: {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"{label} must be a JSON object: {path}")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(
                f"{label} is invalid: {path} line {line_number}: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise AdapterError(f"{label} line {line_number} is not a JSON object: {path}")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise AdapterError(f"refusing to write an empty control table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value == REQUIRED_TOKEN:
        raise AdapterError(f"{label} is missing or unresolved: {value!r}")
    return value


def require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise AdapterError(f"{label} must be a lowercase SHA-256 digest")
    return value


def enabled_controls(config: dict[str, Any], group_name: str) -> list[dict[str, Any]]:
    controls = config.get("controls")
    if not isinstance(controls, dict):
        raise AdapterError("config field is missing or malformed: controls")
    value = controls.get(group_name)
    if not isinstance(value, list):
        raise AdapterError(f"config field is missing or malformed: controls.{group_name}")
    enabled = [item for item in value if isinstance(item, dict) and item.get("enabled", True) is True]
    if not enabled:
        raise AdapterError(
            f"config field has no enabled entries: controls.{group_name}. "
            f"{PUBLISHED_CONTROL_PANEL_GUIDANCE}"
        )
    return enabled


def control_groups(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    positive = enabled_controls(config, "positive")
    negative = enabled_controls(config, "negative")
    result = [("positive", item) for item in positive]
    result.extend(("negative", item) for item in negative)
    seen: set[str] = set()
    for group_name, control in result:
        control_id = require_text(control.get("id"), f"controls.{group_name}[].id")
        if control_id in seen:
            raise AdapterError(f"control id is duplicated: {control_id}")
        seen.add(control_id)
    return result


def resolve_path(value: Any, label: str, config_path: Path) -> Path:
    path_value = require_text(value, label)
    path = Path(path_value)
    if not path.is_absolute():
        path = config_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    return path


def control_structure_path(
    control: dict[str, Any], group_name: str, control_id: str, config_path: Path
) -> Path:
    """Resolve a control structure, preferring the copy inside the run bundle.

    ``structure_path`` names the operator's own file, which exists only on the
    host that composed the campaign. ``runtime_structure_path`` names the copy
    materialize wrote into the bundle, and the bundle is what a remote job
    actually receives. A Modal shard reads the config as materialized, without
    the rebinding ``lane.execute`` applies to a local run, so the operator's own
    path is absent inside the container and only the bundled copy is on the
    Volume. Prefer the bundled copy, the way ``candidate_normalizer`` already
    prefers a supplied manifest's runtime copy, and fall back so the error still
    names ``structure_path`` when neither file exists.
    """
    runtime_value = control.get("runtime_structure_path")
    if isinstance(runtime_value, str) and runtime_value:
        runtime_path = Path(runtime_value)
        if not runtime_path.is_absolute():
            runtime_path = config_path.parent / runtime_path
        runtime_path = runtime_path.resolve()
        if runtime_path.is_file():
            return runtime_path
    return resolve_path(
        control.get("structure_path"),
        f"controls.{group_name}[{control_id}].structure_path",
        config_path,
    )


def pdb_atom(line: str) -> bool:
    return line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27


def pdb_standard_atom(line: str) -> bool:
    return line.startswith("ATOM  ") and len(line) >= 27


def pdb_chain(line: str) -> str:
    return line[21:22].strip()


def pdb_residue_key(line: str) -> tuple[str, str, str]:
    return pdb_chain(line), line[22:26], line[26:27]


def is_mmcif(text: str) -> bool:
    """An mmCIF carries an _atom_site loop; a PDB never does."""
    return any(line.lstrip().startswith("_atom_site.") for line in text.splitlines())


def mmcif_chain_sequence(path: Path, text: str, chain_id: str) -> str:
    """Read one chain's sequence from an mmCIF, standard library only.

    The predictor this campaign runs writes mmCIF, so controls arrive as mmCIF.
    The fixed-column reader below cannot see them at all: column 21 of an mmCIF
    ATOM line is arbitrary text, so every chain lookup missed and the adapter
    reported the chain as empty rather than as unreadable.
    """
    header: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("_atom_site."):
            header.append(stripped.split(".", 1)[1])
    if not header:
        raise AdapterError(f"structure {path} has no _atom_site loop")
    index = {name: position for position, name in enumerate(header)}
    chain_key = "auth_asym_id" if "auth_asym_id" in index else "label_asym_id"
    seq_key = "auth_seq_id" if "auth_seq_id" in index else "label_seq_id"
    for required in (chain_key, seq_key, "label_comp_id", "group_PDB"):
        if required not in index:
            raise AdapterError(f"structure {path} _atom_site loop has no {required}")
    residues: list[str] = []
    seen: set[str] = set()
    width = max(index.values())
    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        fields = line.split()
        if len(fields) <= width:
            continue
        if fields[index["group_PDB"]] != "ATOM":
            continue
        if fields[index[chain_key]] != chain_id:
            continue
        key = fields[index[seq_key]]
        if key in seen:
            continue
        seen.add(key)
        name = fields[index["label_comp_id"]].strip().upper()
        residue = AMINO_ACID_3_TO_1.get(name)
        if residue is None:
            raise AdapterError(
                f"structure {path} chain {chain_id} contains an unsupported standard "
                f"residue {name!r}"
            )
        residues.append(residue)
    if not residues:
        raise AdapterError(f"structure {path} has no protein residues on chain {chain_id}")
    return "".join(residues)


def pdb_chain_sequence(path: Path, chain_id: str) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if is_mmcif(text):
        return mmcif_chain_sequence(path, text, chain_id)
    residues: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for line in text.splitlines():
        if not pdb_standard_atom(line) or pdb_chain(line) != chain_id:
            continue
        key = pdb_residue_key(line)
        if key in seen:
            continue
        seen.add(key)
        residue = AMINO_ACID_3_TO_1.get(line[17:20].strip().upper())
        if residue is None:
            raise AdapterError(
                f"structure {path} chain {chain_id} contains an unsupported standard residue "
                f"{line[17:20].strip()!r}"
            )
        residues.append(residue)
    if not residues:
        raise AdapterError(f"structure {path} has no protein residues on chain {chain_id}")
    return "".join(residues)


def require_pdb(path: Path, label: str) -> None:
    if path.suffix.lower() not in {".pdb", ".ent"}:
        raise AdapterError(
            f"{label} must be a PDB for constructed controls; mmCIF construction is unresolved: {path}"
        )


def rewrite_chain(line: str, chain_id: str) -> str:
    if len(chain_id) != 1:
        raise AdapterError(f"PDB chain IDs must be one character for construction: {chain_id!r}")
    if len(line) < 22:
        raise AdapterError("PDB atom record is shorter than the chain-column contract")
    return line[:21] + chain_id + line[22:]


def rewrite_residue_names(path: Path, chain_id: str, sequence: str) -> str:
    # ATOM only, on both the count and the rewrite. `pdb_chain_sequence` builds the
    # sequence from ATOM records, so counting HETATM here compared two different
    # things: a deposited chain with crystallographic waters reported 164 residues
    # against an 87-residue sequence and refused. Renaming a water to an amino acid
    # is also what this loop would do if a count ever did match.
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    keys: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for line in lines:
        if pdb_standard_atom(line) and pdb_chain(line) == chain_id:
            key = pdb_residue_key(line)
            if key not in seen:
                seen.add(key)
                keys.append(key)
    if len(keys) != len(sequence):
        raise AdapterError(
            f"sequence-decoy source {path} chain {chain_id} has {len(keys)} standard "
            f"residues, but its binder sequence has {len(sequence)} residues. Only ATOM "
            "records count, so heteroatoms and waters are already excluded. Supply a "
            "chain whose residue count matches the sequence you are decoying"
        )
    replacement = dict(zip(keys, sequence))
    output: list[str] = []
    for line in lines:
        if line.startswith("SEQRES") and line[11:12].strip() == chain_id:
            continue
        if pdb_standard_atom(line) and pdb_chain(line) == chain_id:
            residue = replacement[pdb_residue_key(line)]
            line = line[:17] + AMINO_ACID_1_TO_3[residue].rjust(3) + line[20:]
        output.append(line)
    return "\n".join(output) + "\n"


def rewrite_pdb_chains(text: str, mapping: dict[str, str]) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if pdb_atom(line) and pdb_chain(line) in mapping:
            line = rewrite_chain(line, mapping[pdb_chain(line)])
        lines.append(line)
    return "\n".join(lines) + "\n"


def cross_pair_structure(
    target_path: Path,
    target_source_chain: str,
    binder_path: Path,
    binder_source_chain: str,
    target_chain: str,
    binder_chain: str,
) -> str:
    require_pdb(target_path, "cross-pair target source")
    require_pdb(binder_path, "cross-pair binder source")
    target_lines = target_path.read_text(encoding="utf-8", errors="replace").splitlines()
    binder_lines = binder_path.read_text(encoding="utf-8", errors="replace").splitlines()
    output: list[str] = []
    target_count = 0
    binder_count = 0
    for line in target_lines:
        if pdb_atom(line) and pdb_chain(line) == target_source_chain:
            output.append(rewrite_chain(line, target_chain))
            target_count += 1
    output.append("TER")
    for line in binder_lines:
        if pdb_atom(line) and pdb_chain(line) == binder_source_chain:
            output.append(rewrite_chain(line, binder_chain))
            binder_count += 1
    if target_count == 0:
        raise AdapterError(f"cross-pair target source has no atoms on chain {target_source_chain}: {target_path}")
    if binder_count == 0:
        raise AdapterError(f"cross-pair binder source has no atoms on chain {binder_source_chain}: {binder_path}")
    output.append("END")
    return "\n".join(output) + "\n"


def source_control_id(
    control: dict[str, Any],
    *names: str,
    accepted_shape: str | None = None,
) -> str:
    for name in names:
        value = control.get(name)
        if isinstance(value, str) and value and value != REQUIRED_TOKEN:
            return value
    joined = " or ".join(f"controls.negative[].{name}" for name in names)
    message = f"constructed control {control.get('id')!r} requires {joined}"
    if accepted_shape is not None:
        message = f"{message}; {accepted_shape}"
    raise AdapterError(message)


def declared_source_control_fields(control: dict[str, Any]) -> tuple[str, ...]:
    """Return source-control fields that the negative configuration declares."""
    return tuple(name for name in SOURCE_CONTROL_FIELDS if name in control)


def supplied_negative_shape_error(control_id: str, source_fields: Sequence[str]) -> AdapterError:
    sources = " and ".join(
        f"controls.negative[{control_id}].{field}" for field in source_fields
    )
    return AdapterError(
        f"negative control {control_id!r} declares both "
        f"controls.negative[{control_id}].structure_path and {sources}; "
        "a supplied negative accepts structure_path without source control identifiers"
    )


def require_distinct_negative_binder(
    control_id: str,
    binder_sequence: str,
    materialized: dict[str, ControlArtifact],
) -> None:
    """Reject a supplied negative that reuses a positive binder sequence."""
    for positive in materialized.values():
        if positive.control_type == "positive" and binder_sequence == positive.binder_sequence:
            raise AdapterError(
                f"controls.negative[{control_id}].binder_chain sequence matches "
                f"controls.positive[{positive.control_id}].binder_chain; "
                "a supplied negative binder must differ from every positive binder"
            )


def shuffle_sequence(sequence: str, rng: random.Random) -> str:
    """Permute a binder sequence until it differs from the sequence it came from.

    ``make_negative_control`` calls this so a decoy built before the run is the
    permutation this builder would have constructed during the run. The two files
    are byte-identical when the negative declares the chain IDs its source already
    carries, because the constructed branch then renames no chain. Under a rename
    the sequences still match and the bytes differ.
    """
    shuffled = list(sequence)
    for _ in range(max(8, len(shuffled) * 4)):
        rng.shuffle(shuffled)
        if "".join(shuffled) != sequence:
            break
    return "".join(shuffled)


def shuffle_seed(config: dict[str, Any]) -> int:
    controls = config.get("controls")
    value = controls.get("shuffle_seed") if isinstance(controls, dict) else None
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterError("config field is missing or invalid: controls.shuffle_seed")
    return value


def materialize_controls(
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
) -> list[ControlArtifact]:
    """Copy configured positives, supplied negatives, and constructed negatives."""
    groups = control_groups(config)
    controls_by_id = {
        require_text(item.get("id"), "control id"): (group_name, item)
        for group_name, item in groups
    }
    need_shuffle = any(
        item.get("role") == "sequence-decoy" and "structure_path" not in item
        for group_name, item in groups
        if group_name == "negative"
    )
    rng = random.Random(shuffle_seed(config)) if need_shuffle else None
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized: dict[str, ControlArtifact] = {}
    for group_name, control in groups:
        control_id = require_text(control.get("id"), f"controls.{group_name}[].id")
        role = require_text(control.get("role"), f"controls.{group_name}[{control_id}].role")
        expected_structure_sha256 = require_digest(
            control.get("structure_sha256"),
            f"controls.{group_name}[{control_id}].structure_sha256",
        )
        target_chain = require_text(
            control.get("target_chain"), f"controls.{group_name}[{control_id}].target_chain"
        )
        binder_chain = require_text(
            control.get("binder_chain"), f"controls.{group_name}[{control_id}].binder_chain"
        )
        if target_chain == binder_chain:
            raise AdapterError(f"control {control_id} uses the same target and binder chain")

        structure_path: Path
        binder_sequence: str
        if group_name == "positive":
            structure_path = control_structure_path(
                control, "positive", control_id, config_path
            )
            pdb_chain_sequence(structure_path, target_chain)
            binder_sequence = pdb_chain_sequence(structure_path, binder_chain)
            destination = output_dir / f"{control_id}{structure_path.suffix.lower()}"
            shutil.copy2(structure_path, destination)
        elif "structure_path" in control:
            source_fields = declared_source_control_fields(control)
            if source_fields:
                raise supplied_negative_shape_error(control_id, source_fields)
            structure_path = control_structure_path(
                control, "negative", control_id, config_path
            )
            pdb_chain_sequence(structure_path, target_chain)
            binder_sequence = pdb_chain_sequence(structure_path, binder_chain)
            require_distinct_negative_binder(control_id, binder_sequence, materialized)
            destination = output_dir / f"{control_id}{structure_path.suffix.lower()}"
            shutil.copy2(structure_path, destination)
        elif role == "sequence-decoy":
            assert rng is not None
            source_id = source_control_id(
                control,
                "source_control_id",
                accepted_shape=(
                    "role 'sequence-decoy' accepts controls.negative[].source_control_id "
                    "without structure_path, or a structure_path written by "
                    "claude_binder.make_negative_control without source control identifiers"
                ),
            )
            source_record = controls_by_id.get(source_id)
            if source_record is None:
                raise AdapterError(
                    f"sequence-decoy {control_id} names missing source control: {source_id}"
                )
            source_group, source = source_record
            source_path = control_structure_path(
                source,
                source_group,
                source_id,
                config_path,
            )
            source_chain = require_text(
                source.get("binder_chain"),
                f"controls.{source_group}[{source_id}].binder_chain",
            )
            source_target_chain = require_text(
                source.get("target_chain"),
                f"controls.{source_group}[{source_id}].target_chain",
            )
            pdb_chain_sequence(source_path, source_target_chain)
            source_sequence = pdb_chain_sequence(source_path, source_chain)
            binder_sequence = shuffle_sequence(source_sequence, rng)
            if binder_sequence == source_sequence:
                raise AdapterError(
                    f"sequence-decoy {control_id} cannot create a distinct permutation from "
                    f"source control {source_id}"
                )
            require_pdb(source_path, f"sequence-decoy source {source_id}")
            destination = output_dir / f"{control_id}.pdb"
            decoy_text = rewrite_residue_names(source_path, source_chain, binder_sequence)
            decoy_text = rewrite_pdb_chains(
                decoy_text,
                {source_target_chain: target_chain, source_chain: binder_chain},
            )
            destination.write_text(decoy_text, encoding="utf-8")
        elif role in CONSTRUCTED_PAIR_ROLES:
            target_source_id = source_control_id(
                control,
                "source_target_control_id",
                "target_source_control_id",
                accepted_shape=(
                    f"role {role!r} accepts source target and binder control identifiers "
                    "without structure_path, or a supplied structure_path without source "
                    "control identifiers"
                ),
            )
            binder_source_id = source_control_id(
                control,
                "source_binder_control_id",
                "binder_source_control_id",
                accepted_shape=(
                    f"role {role!r} accepts source target and binder control identifiers "
                    "without structure_path, or a supplied structure_path without source "
                    "control identifiers"
                ),
            )
            target_record = controls_by_id.get(target_source_id)
            binder_record = controls_by_id.get(binder_source_id)
            if target_record is None:
                raise AdapterError(
                    f"cross-pair {control_id} names missing target source control: {target_source_id}"
                )
            if binder_record is None:
                raise AdapterError(
                    f"cross-pair {control_id} names missing binder source control: {binder_source_id}"
                )
            target_group, target_source = target_record
            binder_group, binder_source = binder_record
            target_path = control_structure_path(
                target_source,
                target_group,
                target_source_id,
                config_path,
            )
            binder_path = control_structure_path(
                binder_source,
                binder_group,
                binder_source_id,
                config_path,
            )
            target_source_chain = require_text(
                target_source.get("target_chain"),
                f"controls.{target_group}[{target_source_id}].target_chain",
            )
            binder_source_chain = require_text(
                binder_source.get("binder_chain"),
                f"controls.{binder_group}[{binder_source_id}].binder_chain",
            )
            binder_sequence = pdb_chain_sequence(binder_path, binder_source_chain)
            destination = output_dir / f"{control_id}.pdb"
            destination.write_text(
                cross_pair_structure(
                    target_path,
                    target_source_chain,
                    binder_path,
                    binder_source_chain,
                    target_chain,
                    binder_chain,
                ),
                encoding="utf-8",
            )
        else:
            raise AdapterError(
                f"negative control {control_id!r} role {role!r} requires structure_path; "
                "only 'sequence-decoy' accepts source_control_id, and only "
                "'matched-wrong-pair' and 'non-interacting-pair' accept paired source "
                "control identifiers"
            )

        observed_structure_sha256 = sha256_file(destination)
        if observed_structure_sha256 != expected_structure_sha256:
            raise AdapterError(
                f"controls.{group_name}[{control_id}].structure_sha256 does not match "
                f"the materialized structure: expected {expected_structure_sha256}, "
                f"observed {observed_structure_sha256}"
            )
        materialized[control_id] = ControlArtifact(
            control_id=control_id,
            control_type=group_name,
            role=role,
            structure_path=destination.resolve(),
            structure_sha256=observed_structure_sha256,
            target_chain=target_chain,
            binder_chain=binder_chain,
            binder_sequence=binder_sequence,
        )
    return [materialized[require_text(item.get("id"), "control id")] for _, item in groups]


def fasta_path(control: ControlArtifact, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{control.control_id}.fasta"
    path.write_text(f">{control.control_id}\n{control.binder_sequence}\n", encoding="utf-8")
    return path


def target_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    targets = config.get("targets")
    if not isinstance(targets, list) or not targets:
        raise AdapterError("config field is missing or empty: targets")
    records: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise AdapterError(f"config value is not an object: targets[{index}]")
        target_id = require_text(target.get("target_id"), f"targets[{index}].target_id")
        require_digest(target.get("structure_sha256"), f"targets[{index}].structure_sha256")
        records.append(target)
    return records


def candidate_rows(
    config: dict[str, Any],
    controls: Sequence[ControlArtifact],
    fasta_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in target_records(config):
        target_id = str(target["target_id"])
        target_sha256 = str(target["structure_sha256"])
        for control in controls:
            sequence_path = fasta_path(control, fasta_dir)
            rows.append(
                {
                    "target_id": target_id,
                    "target_sha256": target_sha256,
                    "candidate_id": control.control_id,
                    "origin_generator": "control-builder",
                    "sequence_path": str(sequence_path.resolve()),
                    "sequence_sha256": sha256_text(control.binder_sequence),
                    "sequence_length": len(control.binder_sequence),
                    "design_pose_path": str(control.structure_path),
                    "design_pose_sha256": control.structure_sha256,
                    "chain_mapping": {
                        "target": control.target_chain,
                        "binder": control.binder_chain,
                    },
                    "status": "prepared",
                }
            )
    return rows


def predictor_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    cofold = config.get("cofold")
    predictors = cofold.get("predictors") if isinstance(cofold, dict) else None
    if not isinstance(predictors, list):
        raise AdapterError("config field is missing or malformed: cofold.predictors")
    enabled = [item for item in predictors if isinstance(item, dict) and item.get("enabled", True) is True]
    if not enabled:
        raise AdapterError("config field has no enabled entries: cofold.predictors")
    for predictor in enabled:
        predictor_id = require_text(predictor.get("id"), "cofold.predictors[].id")
        adapter_id = require_text(
            predictor.get("adapter_id"), f"cofold.predictors[{predictor_id}].adapter_id"
        )
        adapter_command_template(config, adapter_id)
    return enabled


def rescore_seeds(config: dict[str, Any]) -> list[int]:
    cofold = config.get("cofold")
    values = cofold.get("rescore_seeds") if isinstance(cofold, dict) else None
    if not isinstance(values, list) or not values:
        raise AdapterError(
            "config field is missing or empty: cofold.rescore_seeds. "
            "The published campaign rescored with five seeds labeled 0 through 4."
        )
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise AdapterError(f"cofold.rescore_seeds contains a non-integer value: {value!r}")
        result.append(value)
    return result


def adapter_record(config: dict[str, Any], adapter_id: str) -> dict[str, Any]:
    adapters = config.get("adapters")
    if not isinstance(adapters, list):
        raise AdapterError("config field is missing or malformed: adapters")
    matches = [item for item in adapters if isinstance(item, dict) and item.get("adapter_id") == adapter_id]
    if len(matches) != 1:
        raise AdapterError(f"config must register one adapter: {adapter_id}")
    return matches[0]


def adapter_model_revision(config: dict[str, Any], adapter_id: str) -> str:
    adapter = adapter_record(config, adapter_id)
    return require_text(adapter.get("model_revision"), f"adapters[{adapter_id}].model_revision")


def adapter_command_template(config: dict[str, Any], adapter_id: str) -> list[str]:
    adapter = adapter_record(config, adapter_id)
    template = adapter.get("command_argv_template")
    if not isinstance(template, list) or not template or any(
        not isinstance(value, str) or not value for value in template
    ):
        raise AdapterError(
            f"adapters[{adapter_id}].command_argv_template must be a non-empty string argv list"
        )
    return list(template)


def adapter_parser_template(config: dict[str, Any], adapter_id: str) -> list[str] | None:
    """Return the adapter's parser argv, or None when it declares no parser phase."""
    adapter = adapter_record(config, adapter_id)
    template = adapter.get("parser_argv_template")
    if template is None:
        return None
    if not isinstance(template, list) or not template or any(
        not isinstance(value, str) or not value for value in template
    ):
        raise AdapterError(
            f"adapters[{adapter_id}].parser_argv_template must be a non-empty string argv list"
        )
    return list(template)


def module_from_command_template(template: Sequence[str], adapter_id: str) -> str:
    for index, value in enumerate(template[:-1]):
        if value == "-m":
            module_name = template[index + 1]
            if module_name:
                return module_name
    raise AdapterError(
        f"adapters[{adapter_id}].command_argv_template must invoke a Python module with -m"
    )


def write_predictor_config(
    config: dict[str, Any],
    controls: Sequence[ControlArtifact],
    predictor: dict[str, Any],
    workspace: Path,
) -> tuple[Path, str]:
    predictor_id = require_text(predictor.get("id"), "cofold predictor id")
    adapter_id = require_text(predictor.get("adapter_id"), f"cofold.predictors[{predictor_id}].adapter_id")
    stage_id = f"cofold-rescore-{predictor_id}"
    working = copy.deepcopy(config)
    working["controls"] = {"positive": [], "negative": []}
    for control in controls:
        item = {
            "id": control.control_id,
            "role": control.role,
            "enabled": True,
            "structure_path": str(control.structure_path),
            "structure_sha256": control.structure_sha256,
            "target_chain": control.target_chain,
            "binder_chain": control.binder_chain,
        }
        working["controls"][control.control_type].append(item)
    working["cofold"] = copy.deepcopy(config["cofold"])
    working["cofold"]["predictors"] = [copy.deepcopy(predictor)]
    working["adapters"] = copy.deepcopy(config["adapters"])
    working["stages"] = [
        {
            "stage_id": stage_id,
            "outputs": [
                {
                    "artifact_id": f"{predictor_id}-control-predictions",
                    "path_template": f"{{{{attempt_dir}}}}/{{{{phase}}}}/control-predictions-{predictor_id}.jsonl",
                }
            ],
        }
    ]
    path = workspace / f"config-{predictor_id}.json"
    path.write_text(json.dumps(working, sort_keys=True) + "\n", encoding="utf-8")
    return path, stage_id


def input_values(
    args: argparse.Namespace,
    targets: Sequence[dict[str, Any]],
    option: str,
    *,
    require_files: bool = False,
) -> None:
    values = getattr(args, option, None) or []
    target_ids = [str(target["target_id"]) for target in targets]
    resolved: dict[str, str] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        target_id = key.strip() if separator else None
        input_value = raw.strip() if separator else value.strip()
        if target_id is None:
            if len(target_ids) != 1:
                raise AdapterError(
                    f"--{option.replace('_', '-')} needs TARGET_ID=VALUE for {len(target_ids)} targets"
                )
            target_id = target_ids[0]
        if target_id not in target_ids:
            raise AdapterError(f"--{option.replace('_', '-')} names an unknown target: {target_id}")
        if not input_value or input_value == REQUIRED_TOKEN:
            raise AdapterError(f"--{option.replace('_', '-')} has no value for target {target_id}")
        if target_id in resolved:
            raise AdapterError(f"--{option.replace('_', '-')} repeats target {target_id}")
        if require_files and not Path(input_value).is_file():
            raise AdapterError(
                f"--{option.replace('_', '-')} file is missing for target {target_id}: {input_value}"
            )
        resolved[target_id] = input_value
    missing = [target_id for target_id in target_ids if target_id not in resolved]
    if missing:
        raise AdapterError(
            f"--{option.replace('_', '-')} is missing for target(s): {', '.join(missing)}"
        )


def validate_predictor_inputs(config: dict[str, Any], args: argparse.Namespace) -> None:
    targets = target_records(config)
    input_values(args, targets, "target_sequence")
    input_values(args, targets, "hotspot_residues")
    adapters = {str(item["adapter_id"]): item for item in predictor_records(config)}
    if "esmfold2-predictor" in adapters:
        input_values(args, targets, "target_msa_a3m", require_files=True)
    if "protenix-v2-predictor" in adapters:
        input_values(args, targets, "target_unpaired_msa_a3m", require_files=True)


def template_required_arguments(command: str) -> tuple[str, ...]:
    """Declare flags `validate_predictor_inputs` requires for a run command."""
    return ("--target-sequence", "--hotspot-residues") if command == "run" else ()


def predictor_command_context(
    config: dict[str, Any],
    *,
    args: argparse.Namespace,
    stage_id: str,
    config_path: Path,
    workspace: Path,
    candidate_count: int,
) -> dict[str, Any]:
    """Build the predictor context from the lane context plus control paths."""
    context = lane.resolved_context(config, require_residue_map=True)
    context.update(
        {
            "stage_id": stage_id,
            "phase": args.phase,
            "count": candidate_count,
            "attempt_dir": str(args.attempt_dir),
            "receipts_dir": str(args.receipts_dir),
            "artifact_root": str(workspace),
            "config_path": str(config_path),
            "plan_path": str(args.plan),
            "python_executable": sys.executable,
        }
    )
    return context


FAL_PREDICTOR_MODULE = "claude_binder.adapters.fal_esmfold2_fast_predictor"
FAL_CALL_JOURNAL_NAME = "ef2fast-call-journal.jsonl"


def with_call_journal(
    command_argv_template: Sequence[str],
    module_name: str,
    journal_path: Path,
) -> list[str]:
    """Give the fal predictor a call journal that outlives the attempt that wrote it.

    The predictor defaults its journal to `<artifact_root>/.state/<stage>/`, and this
    adapter hands it a scratch workspace under `attempts/<attempt_id>/` as that artifact
    root. So the control stage is the one predictor stage whose journal disappears on a
    retry, and the retry re-pays for every fold the previous attempt completed. On the
    2026-09-07 fal run that was ten of the sixteen paid folds, and the expensive ten.
    Every other predictor stage receives the run's artifact root and keeps its journal.

    A template that already names a journal keeps it, so a profile stays in charge.
    """
    template = list(command_argv_template)
    if module_name != FAL_PREDICTOR_MODULE:
        return template
    if any(token.split("=", 1)[0] == "--call-journal" for token in template):
        return template
    return [*template, "--call-journal", str(journal_path)]


def run_predictor(
    module_name: str,
    *,
    command_argv_template: Sequence[str],
    context: dict[str, Any],
    stage_id: str,
    workspace: Path,
    manifest_path: Path,
    parser_argv_template: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Run one predictor over the controls and read the manifest it wrote.

    This orchestrates a predictor itself rather than going through the executor, and it
    ran only the command phase. A predictor that folds and writes its manifest in one
    command is fine. A predictor that splits dispatch from parsing is not: its command
    writes a run index and its parser writes the manifest, so the folds were dispatched,
    completed, billed, and then discarded when this function found no manifest. Every
    packaged fal cofold predictor splits that way, so `control-calibration` could not
    complete on the fal route. A paid run on 2026-09-07 lost ten folds to it.

    The parser runs only when the command left no manifest. The Modal predictor writes
    it inside `run`, and that path has seven successful `control-calibration` receipts,
    so it stays exactly as it was rather than gaining a second phase nothing has tested.
    """
    argv = render_argv(command_argv_template, context)
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    if completed.returncode != 0 and not manifest_path.is_file():
        detail = (completed.stderr or completed.stdout or "no predictor output").strip().splitlines()
        raise AdapterError(
            f"predictor {module_name} failed before writing a control manifest: "
            f"{detail[-1] if detail else 'no detail'}"
        )
    if not manifest_path.is_file() and parser_argv_template:
        parsed = subprocess.run(
            render_argv(parser_argv_template, context), check=False, capture_output=True, text=True
        )
        if parsed.returncode != 0 and not manifest_path.is_file():
            detail = (parsed.stderr or parsed.stdout or "no parser output").strip().splitlines()
            raise AdapterError(
                f"predictor {module_name} dispatched its folds and then failed to parse "
                f"them into a control manifest: {detail[-1] if detail else 'no detail'}"
            )
    if not manifest_path.is_file():
        raise AdapterError(f"predictor {module_name} wrote no control manifest: {manifest_path}")
    return read_jsonl(manifest_path, f"predictor {module_name} manifest")


def check_predictor_rows(
    rows: Sequence[dict[str, Any]],
    predictor_id: str,
    adapter_id: str,
    expected_model_revision: Any,
) -> None:
    """Apply this stage's row contract to whatever a predictor returned.

    The smoke boundary runs these checks on one fold before the rest of the panel is
    dispatched, and the full panel runs them again. Asserting the same fields in both
    places is deliberate: the boundary is only worth having if it tests what the stage
    already refuses to accept.
    """
    for row in rows:
        if row.get("predictor") != predictor_id:
            raise AdapterError(
                f"predictor {adapter_id} emitted {row.get('predictor')!r}; "
                f"expected {predictor_id!r}"
            )
        if row.get("model_revision") != expected_model_revision:
            raise AdapterError(
                f"predictor {predictor_id} emitted model_revision "
                f"{row.get('model_revision')!r}; expected {expected_model_revision!r}"
            )
        if row.get("phase") != CONTROL_PHASE:
            raise AdapterError(
                f"predictor {predictor_id} emitted phase {row.get('phase')!r}; "
                f"expected {CONTROL_PHASE!r}"
            )


def smoke_fold_rows(
    config: dict[str, Any],
    *,
    args: argparse.Namespace,
    controls: Sequence[ControlArtifact],
    predictor: dict[str, Any],
    predictor_id: str,
    module_name: str,
    command_argv_template: Sequence[str],
    parser_argv_template: Sequence[str] | None,
    stage_id: str,
    candidate_row: dict[str, Any],
    candidate_dir: Path,
) -> list[dict[str, Any]]:
    """Fold one control on its own, so a bad first result stops the rest of the panel.

    This stage is a multi-design dispatch to a paid provider, and it ran the whole panel
    in one invocation. The predictor inspects every response and records a failure, but it
    continues the loop, so a malformed first result did not prevent the remaining folds
    from being paid for. On the 2026-09-07 fal run the panel was ten folds and 4.21375 USD,
    the largest of the three paid stages.

    The fold runs in its own workspace with a one-row candidate manifest, so it cannot
    disturb the panel's inputs or overwrite its manifest. The call journal is shared,
    because `with_call_journal` puts it under the run's artifact root rather than under
    either workspace, so the panel reuses this fold rather than paying for it twice.
    """
    workspace = (candidate_dir / f"smoke-{predictor_id}" / "predictor-workspace").resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    write_jsonl(workspace / "optimization" / "rescore-candidates.jsonl", [candidate_row])
    config_path, _ = write_predictor_config(config, controls, predictor, workspace)
    attempt_dir = (args.attempt_dir / "control-smoke" / predictor_id).resolve()
    context = predictor_command_context(
        config,
        args=args,
        stage_id=stage_id,
        config_path=config_path,
        workspace=workspace,
        candidate_count=1,
    )
    context["attempt_dir"] = str(attempt_dir)
    return run_predictor(
        module_name,
        command_argv_template=command_argv_template,
        context=context,
        stage_id=stage_id,
        workspace=workspace,
        manifest_path=attempt_dir / args.phase / f"control-predictions-{predictor_id}.jsonl",
        parser_argv_template=parser_argv_template,
    )


def annotate_rows(
    rows: Sequence[dict[str, Any]],
    controls: dict[str, ControlArtifact],
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for row in rows:
        control_id = str(row.get("candidate_id"))
        control = controls.get(control_id)
        if control is None:
            raise AdapterError(f"predictor emitted an unregistered control: {control_id}")
        try:
            seed = int(row["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterError(f"predictor row for {control_id} has an invalid seed") from exc
        key = (str(row.get("target_id")), control_id, str(row.get("predictor")), seed)
        if key in seen:
            raise AdapterError(f"predictors emitted a duplicate control key: {key}")
        seen.add(key)
        enriched = dict(row)
        enriched.update(
            {
                "control_type": control.control_type,
                "control_role": control.role,
                "control_structure_sha256": control.structure_sha256,
                "score_instrument": score_instrument_arm_name(str(row["predictor"])),
            }
        )
        annotated.append(enriched)
    return annotated


def expected_keys(config: dict[str, Any], controls: Sequence[ControlArtifact]) -> set[tuple[str, str, str, int]]:
    return {
        (str(target["target_id"]), control.control_id, str(predictor["id"]), seed)
        for target in target_records(config)
        for control in controls
        for predictor in predictor_records(config)
        for seed in rescore_seeds(config)
    }


def output_path(args: argparse.Namespace) -> Path:
    return (args.attempt_dir / args.phase / "control-observations.jsonl").resolve()


def run_stage(args: argparse.Namespace) -> int:
    if args.stage != "control-calibration":
        raise AdapterError(
            f"control builder implements control-calibration, and the stage is {args.stage}"
        )
    config_path = args.config.resolve()
    config = read_json(config_path, "campaign config")
    controls_dir = (args.attempt_dir / args.phase / "control-structures").resolve()
    controls = materialize_controls(config, config_path, controls_dir)
    validate_predictor_inputs(config, args)
    candidate_dir = (args.attempt_dir / args.phase / "control-workspace").resolve()
    candidate_dir.mkdir(parents=True, exist_ok=True)
    workspace = candidate_dir / "predictor-workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    candidate_manifest = workspace / "optimization" / "rescore-candidates.jsonl"
    write_jsonl(candidate_manifest, candidate_rows(config, controls, workspace / "sequences"))

    all_rows: list[dict[str, Any]] = []
    control_by_id = {control.control_id: control for control in controls}
    for predictor in predictor_records(config):
        predictor_id = require_text(predictor.get("id"), "cofold predictor id")
        adapter_id = require_text(predictor.get("adapter_id"), f"cofold.predictors[{predictor_id}].adapter_id")
        adapter_model_revision(config, adapter_id)
        command_argv_template = adapter_command_template(config, adapter_id)
        module_name = module_from_command_template(command_argv_template, adapter_id)
        # args.stage is control-calibration, guarded at the top of this function. The
        # predictor's own stage_id below is cofold-rescore-esmfold2-fast, which is also the
        # real rescore stage's key, so keying the journal on this stage is what keeps two
        # stages' paid-call records out of one file.
        command_argv_template = with_call_journal(
            command_argv_template,
            module_name,
            args.artifact_root.expanduser().resolve()
            / ".state"
            / args.stage
            / FAL_CALL_JOURNAL_NAME,
        )
        predictor_config, stage_id = write_predictor_config(config, controls, predictor, workspace)
        manifest = args.attempt_dir / args.phase / f"control-predictions-{predictor_id}.jsonl"
        expected_model_revision = adapter_model_revision(config, adapter_id)
        parser_argv_template = adapter_parser_template(config, adapter_id)
        panel = read_jsonl(candidate_manifest, "control candidate manifest")
        # Smoke before scale. A panel of one is already its own boundary, so folding it
        # twice would buy nothing and cost a second invocation.
        if len(panel) > 1:
            smoke_rows = smoke_fold_rows(
                config,
                args=args,
                controls=controls,
                predictor=predictor,
                predictor_id=predictor_id,
                module_name=module_name,
                command_argv_template=command_argv_template,
                parser_argv_template=parser_argv_template,
                stage_id=stage_id,
                candidate_row=panel[0],
                candidate_dir=candidate_dir,
            )
            check_predictor_rows(smoke_rows, predictor_id, adapter_id, expected_model_revision)
            annotate_rows(smoke_rows, control_by_id)
        context = predictor_command_context(
            config,
            args=args,
            stage_id=stage_id,
            config_path=predictor_config,
            workspace=workspace,
            candidate_count=len(panel),
        )
        rows = run_predictor(
            module_name,
            command_argv_template=command_argv_template,
            context=context,
            stage_id=stage_id,
            workspace=workspace,
            manifest_path=manifest,
            parser_argv_template=parser_argv_template,
        )
        check_predictor_rows(rows, predictor_id, adapter_id, expected_model_revision)
        all_rows.extend(annotate_rows(rows, control_by_id))

    observed_keys = {
        (str(row.get("target_id")), str(row.get("candidate_id")), str(row.get("predictor")), int(row.get("seed", -1)))
        for row in all_rows
    }
    expected = expected_keys(config, controls)
    if observed_keys != expected:
        missing = sorted(expected - observed_keys)
        extra = sorted(observed_keys - expected)
        raise AdapterError(f"control prediction matrix mismatch: missing={missing}, extra={extra}")
    all_rows.sort(key=lambda row: (str(row["target_id"]), str(row["candidate_id"]), str(row["predictor"]), int(row["seed"])))
    write_jsonl(output_path(args), all_rows)
    print(f"control builder: controls={len(controls)} rows={len(all_rows)} output={output_path(args)}")
    return 0


def parse_stage(args: argparse.Namespace) -> int:
    path = output_path(args)
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        errors.append(f"control observations are missing: {path}")
    else:
        try:
            rows = read_jsonl(path, "control observations")
        except AdapterError as exc:
            errors.append(str(exc))
    result = {
        "ok": bool(rows) and not errors,
        "parsed_count": len(rows),
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": [sha256_file(path)] if path.is_file() else [],
    }
    result_path = args.attempt_dir / args.phase / "parser-result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for error in errors:
        print(f"control builder parser: {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("toolcheck")
    for name in ("run", "parse"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--stage", required=True)
        subparser.add_argument("--phase", required=True)
        subparser.add_argument("--count", type=int, default=1)
        subparser.add_argument("--attempt-dir", type=Path, required=True)
        subparser.add_argument("--receipts-dir", type=Path, required=True)
        subparser.add_argument("--artifact-root", type=Path, required=True)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--plan", type=Path, required=True)
        if name != "run":
            continue
        subparser.add_argument("--target-sequence", action="append")
        subparser.add_argument("--hotspot-residues", action="append")
        subparser.add_argument("--target-msa-a3m", action="append")
        subparser.add_argument("--target-unpaired-msa-a3m", action="append")
        subparser.add_argument("--target-paired-msa-a3m", action="append")
    return parser


def toolcheck() -> int:
    print("control builder ok, standard-library orchestration")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "toolcheck":
            return toolcheck()
        return run_stage(args) if args.command == "run" else parse_stage(args)
    except Exception as exc:  # noqa: BLE001
        print(f"control builder: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
