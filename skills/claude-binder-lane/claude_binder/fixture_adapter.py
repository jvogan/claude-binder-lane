#!/usr/bin/env python3
"""Local fixture adapter for the Claude binder lane contract test."""

from __future__ import annotations

import sys

import argparse
import glob
import hashlib
import json
import re
import shutil
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Any

from .arms import score_instrument_arm_name
from .filter_contracts import filter_report, retired_filter_statuses

from claude_binder import lane

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# The target fixture sequence is read from target.pdb as AGS. Excluding those
# residues keeps the synthetic binder sequence from becoming an ambiguous
# ordered-subsequence match for the target in derive_chain_mapping.
CANONICAL_AMINO_ACIDS = "CDEFHIKLMNPQRTVWY"


def fixture_sequence(candidate_id: str, length: int = 60) -> str:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).digest()
    return "".join(
        CANONICAL_AMINO_ACIDS[digest[index % len(digest)] % len(CANONICAL_AMINO_ACIDS)]
        for index in range(length)
    )


_ONE_TO_THREE = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}
_THREE_TO_ONE = {three: one for one, three in _ONE_TO_THREE.items()}


def _coordinate_records(path: Path) -> list[Any]:
    """Return the first model's coordinate records of a PDB or mmCIF file.

    `make_target_inputs` writes its reference complex as mmCIF and the
    own-target guide points `targets[].structure_path` at that file, so this
    adapter receives both formats. A fixed-column PDB read of an mmCIF row
    lands on whitespace, which returned a short all-X sequence instead of
    refusing, so the format is chosen by suffix and confirmed by content.
    """
    from claude_binder.adapters.target_prep_adapter import parse_cif_atoms, parse_pdb_atoms

    text = path.read_text(errors="replace")
    holds_pdb_atoms = any(line.startswith(("ATOM  ", "HETATM")) for line in text.splitlines())
    if path.suffix.lower() in {".cif", ".mmcif"} or ("_atom_site." in text and not holds_pdb_atoms):
        return parse_cif_atoms(text)
    return parse_pdb_atoms(text)


def _chain_sequences(path: Path) -> dict[str, str]:
    sequences: dict[str, list[str]] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("SEQRES") or len(line) < 20:
            continue
        chain_id = line[11:12].strip()
        if chain_id:
            sequences.setdefault(chain_id, []).extend(
                _THREE_TO_ONE.get(residue.upper(), "X") for residue in line[19:].split()
            )
    if sequences:
        return {chain_id: "".join(residues) for chain_id, residues in sequences.items()}

    # ATOM only. A deposited chain carries its crystallographic waters as
    # HETATM under the same chain id, and counting those into a polymer
    # sequence puts HOH where a residue belongs.
    observed_residues: dict[str, set[tuple[int, str]]] = {}
    for atom in _coordinate_records(path):
        if atom.record != "ATOM":
            continue
        chain_id = atom.chain_id.strip()
        residue_id = (atom.residue_number, atom.insertion_code.strip())
        if not chain_id or residue_id in observed_residues.setdefault(chain_id, set()):
            continue
        observed_residues[chain_id].add(residue_id)
        sequences.setdefault(chain_id, []).append(
            _THREE_TO_ONE.get(atom.residue_name.strip().upper(), "X")
        )
    if not sequences:
        raise ValueError(
            f"structure file carries no ATOM records this adapter can read: {path}. "
            "Pass a .pdb, .ent, .cif, or .mmcif file holding polymer coordinates"
        )
    return {chain_id: "".join(residues) for chain_id, residues in sequences.items()}


def derive_chain_mapping(
    structure_path: Path,
    target_sequence: str,
    binder_sequence: str,
    *,
    structure_label: str,
    declared: tuple[str, str] | None = None,
) -> dict[str, str]:
    """Match two chains of a structure to their known sequences.

    `declared` names the chain ids the campaign gave for this file, and only a
    reference structure passes it. A deposited complex often holds several
    copies of the same pair, so 1BRS carries the barnase sequence on chains A
    and C and a sequence-only match cannot choose between them. A predicted
    structure passes nothing, because its chain letters come from the predictor
    and have to be derived from the sequences.

    A declaration is honoured only when both named chains carry the expected
    sequences, so a wrong one still refuses.
    """
    observed = _chain_sequences(structure_path)
    if declared is not None:
        declared_target, declared_binder = declared
        if (
            declared_target != declared_binder
            and observed.get(declared_target) == target_sequence
            and observed.get(declared_binder) == binder_sequence
        ):
            return {"target": declared_target, "binder": declared_binder}
    target = [chain_id for chain_id, sequence in observed.items() if sequence == target_sequence]
    binder = [chain_id for chain_id, sequence in observed.items() if sequence == binder_sequence]
    if len(target) == 1 and len(binder) == 1 and target[0] != binder[0]:
        return {"target": target[0], "binder": binder[0]}
    if len(target) == 1:
        candidates = [
            chain_id
            for chain_id, sequence in observed.items()
            if chain_id != target[0] and len(sequence) == len(binder_sequence)
        ]
        if len(candidates) == 1:
            return {"target": target[0], "binder": candidates[0]}
    if len(binder) == 1:
        candidates = [
            chain_id
            for chain_id, sequence in observed.items()
            if chain_id != binder[0] and len(sequence) == len(target_sequence)
        ]
        if len(candidates) == 1:
            return {"target": candidates[0], "binder": binder[0]}
    if len(target) != 1 or len(binder) != 1 or target[0] == binder[0]:
        raise ValueError(f"could not map target and binder chains in {structure_label}")
    return {"target": target[0], "binder": binder[0]}


def read_fasta_sequence(path: Path) -> str:
    lines = [line.strip() for line in path.read_text().splitlines()]
    sequence = "".join(line for line in lines if line and not line.startswith(">"))
    if not sequence:
        raise ValueError(f"fixture FASTA holds no sequence: {path}")
    return sequence.upper()


def structure_chain_sequence(structure_path: Path, chain_id: str) -> str:
    try:
        return _chain_sequences(structure_path)[chain_id]
    except KeyError as exc:
        raise ValueError(f"fixture structure has no chain {chain_id!r}: {structure_path}") from exc


def write_synthetic_structure(
    path: Path,
    chains: dict[str, str],
    *,
    remark: str | None = None,
    geometry_key: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        if remark:
            handle.write(f"REMARK 900 FIXTURE {remark}\n")
        for serial, (chain_id, sequence) in enumerate(chains.items(), start=1):
            # A deposited target carries modified residues, MSE most often, and
            # the three-letter table holds only the twenty standard ones. UNK is
            # the wwPDB code for an amino acid of unknown type, so the stand-in
            # keeps the residue count without naming a residue the file did not.
            residues = [_ONE_TO_THREE.get(residue, "UNK") for residue in sequence]
            for start in range(0, len(residues), 13):
                chunk = residues[start : start + 13]
                handle.write(
                    f"SEQRES {serial:3d} {chain_id:1s}{len(residues):5d}  "
                    + " ".join(chunk)
                    + "\n"
                )
        geometry_digest = hashlib.sha256(geometry_key.encode("utf-8")).digest()
        atom_serial = 1
        for chain_index, (chain_id, sequence) in enumerate(chains.items()):
            x_offset = chain_index * 18.0 + geometry_digest[chain_index] / 255.0
            y_offset = (geometry_digest[chain_index + 2] % 11) - 5
            z_offset = (geometry_digest[chain_index + 4] % 7) - 3
            for residue_index, residue in enumerate(sequence, start=1):
                residue_name = _ONE_TO_THREE.get(residue, "UNK")
                x_coordinate = x_offset + (residue_index - 1) * 1.35
                y_coordinate = y_offset + (((residue_index * 5) % 13) - 6) * 0.55
                z_coordinate = z_offset + (((residue_index * 3) % 9) - 4) * 0.35
                handle.write(
                    f"ATOM  {atom_serial:5d}  CA  {residue_name:>3s} {chain_id:1s}{residue_index:4d}    "
                    f"{x_coordinate:8.3f}{y_coordinate:8.3f}{z_coordinate:8.3f}  1.00 80.00           C\n"
                )
                atom_serial += 1
        handle.write("END\n")


def write_candidate_sequences(rows: list[dict[str, Any]], sequence_root: Path) -> list[dict[str, Any]]:
    sequence_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        candidate_id = str(row["candidate_id"])
        sequence = fixture_sequence(candidate_id)
        path = sequence_root / f"{candidate_id}.fasta"
        path.write_text(f">{candidate_id}\n{sequence}\n")
        row["sequence_path"] = str(path.resolve())
        row["sequence_sha256"] = stable_hash(sequence)
        row["sequence_length"] = len(sequence)
    return rows


def write_candidate_design_poses(
    rows: list[dict[str, Any]],
    pose_root: Path,
    *,
    target_structure_path: Path,
    target_chain_id: str,
    binder_chain_id: str,
) -> list[dict[str, Any]]:
    pose_root.mkdir(parents=True, exist_ok=True)
    target_sequence = structure_chain_sequence(target_structure_path, target_chain_id)
    for row in rows:
        candidate_id = str(row["candidate_id"])
        path = pose_root / f"{candidate_id}.pdb"
        binder_sequence = (
            read_fasta_sequence(Path(row["sequence_path"]))
            if row.get("sequence_path")
            else "A"
        )
        write_synthetic_structure(
            path,
            {
                target_chain_id: target_sequence,
                binder_chain_id: "C" * len(binder_sequence),
            },
            remark=f"DESIGN POSE {candidate_id}",
            geometry_key=candidate_id,
        )
        row["design_pose_path"] = str(path.resolve())
        row["design_pose_sha256"] = sha256(path)
    return rows


def record_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


MEASUREMENT_SOURCE_FIELDS = (
    "target_id",
    "target_sha256",
    "candidate_id",
    "predictor",
    "model_revision",
    "seed",
    "phase",
    "sequence_sha256",
    "design_pose_sha256",
    "predicted_complex_sha256",
    "pae_sha256",
    "target_chain_id",
    "binder_chain_id",
    "chain_mapping",
    "predicted_target_chain_id",
    "predicted_binder_chain_id",
    "reference_target_chain_id",
    "reference_binder_chain_id",
    "aligned_target_residue_count",
    "target_alignment_rmsd",
    "ipsae_implementation_revision",
    "ipsae_interface_cutoff_angstrom",
    "dockq_implementation_revision",
    "site_scorer_revision",
    "site_residue_map_sha256",
    "site_contact_cutoff_angstrom",
    "site_atom_selection",
    "site_metric_basis",
    "site_hotspot_source",
    "site_hotspot_relationship",
    "hotspot_recovery_duplicates_target_contact_recall",
    "ipsae_target_to_binder",
    "ipsae_binder_to_target",
    "ipsae_min",
    "sc_dockq",
    "dockq",
    "fnat",
    "interface_rmsd",
    "ligand_rmsd",
    "mapping_status",
    "site_contact_iou",
    "target_contact_recall",
    "target_contact_precision",
    "hotspot_recovery",
    "offsite_contact_fraction",
    "iptm",
    "interface_pae",
    "interface_plddt",
    "clash_count",
    "contact_count",
    "status",
)


def fixture_score(row: dict[str, Any], predictor_index: int) -> float:
    candidate_id = str(row["candidate_id"])
    phase = str(row["phase"])
    if row.get("control_type") == "positive":
        base = 0.94
    elif row.get("control_type") == "negative":
        base = 0.06
    elif phase == "screen":
        # Keyed on the arm rather than on one whole candidate id. The genie3 entry
        # read `genie3-000`, which existed only while genie3 was registered as a
        # co-design generator. Its designed row is now `genie3-000-proteinmpnn-00`,
        # that id fell through to the default 0.7, and the arm dropped below
        # fixture-codesign, so the promoted rows came out in an order the
        # executor's own recomputation refused. The arm prefix survives a change
        # of sequence designer; the whole id does not.
        base = 0.7
        for prefix, arm_score in (
            ("rfdiffusion-000-proteinmpnn", 0.82),
            ("rfdiffusion-000-solublempnn", 0.81),
            ("genie3-000", 0.80),
            ("fixture-codesign-000", 0.79),
        ):
            if candidate_id.startswith(prefix):
                base = arm_score
                break
    elif phase == "optimization":
        match = re.search(r"-opt-r(\d+)$", candidate_id)
        round_index = int(match.group(1)) if match else 1
        base = 0.72 + (0.01 * round_index)
    elif candidate_id.startswith("rfdiffusion-"):
        base = [0.84, 0.66, 0.58][predictor_index % 3]
    elif candidate_id.startswith("genie3-"):
        base = [0.72, 0.82, 0.62][predictor_index % 3]
    else:
        base = [0.64, 0.70, 0.86][predictor_index % 3]
    if phase == "uniform-rescore":
        base -= abs(int(row["seed"]) - 2) * 0.005
    return base


def copy_fixture_complex(configured_path: object, record_dir: Path) -> Path:
    """Copy a real predicted complex into the fixture's output directory.

    A fixture that writes synthetic coordinates can make a code path that never
    worked pass every offline check, so a profile may name a real structure to use
    instead. The field is optional. When a profile does name one, a path that is not
    there is an error rather than a silent fall back to the synthetic file, because
    the point of naming it was to stop using the synthetic file.

    The raise lives here rather than at the call site on purpose. The contract audit
    marks a config field required when a raise is conditioned on the name it was read
    into, and this field is genuinely optional.
    """
    source_path = Path(str(configured_path)).expanduser()
    if not source_path.is_file():
        raise ValueError(f"fixture predicted complex does not exist: {source_path}")
    destination = record_dir / f"complex{source_path.suffix.lower()}"
    shutil.copy2(source_path, destination)
    return destination


def enabled_controls(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the enabled positive and negative controls, keyed by control id."""
    groups = config.get("controls") or {}
    return {
        item["id"]: item
        for group_name in ("positive", "negative")
        for item in groups.get(group_name, [])
        if item.get("enabled", True)
    }


def fixture_chain_mapping(
    config: dict[str, Any],
    target: dict[str, Any],
    candidate_id: str,
) -> dict[str, str]:
    """Return the target and binder chains one fixture row declares.

    Read from configuration, never assumed. A control carries its own two
    chains and overrides the target's. This is the rule
    `esmfold2_predictor.chain_mapping_for` applies on the live path, and
    `lane.validate_observations` compares a row's `chain_mapping` against it for
    exact equality, so a literal chain letter fails on any campaign configured
    differently.

    The target's chain is the one whose role is `design-target`, not the first
    chain declared. A target that lists another chain first would otherwise be
    scored against the wrong side of its own structure. A target that declares
    exactly one chain is the single case where position assumes nothing, and it
    is taken without a role, because an abbreviated fixture leaves the role out.
    """
    control = enabled_controls(config).get(str(candidate_id))
    if control is not None:
        return {"target": str(control["target_chain"]), "binder": str(control["binder_chain"])}
    chains = [item for item in target.get("chains", []) if isinstance(item, dict)]
    if len(chains) == 1 and chains[0].get("chain_id"):
        target_chain = str(chains[0]["chain_id"])
    else:
        target_chain = lane.target_design_chain(target)
    return {
        "target": target_chain,
        "binder": str(config["binder"]["binder_chain_id"]),
    }


def attach_raw_artifacts(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    predictors = [item for item in config["cofold"]["predictors"] if item.get("enabled", True)]
    predictor_indexes = {item["id"]: index for index, item in enumerate(predictors)}
    targets = {item["target_id"]: item for item in config["targets"]}
    implementation = config["scoring"]["implementations"]
    for row in rows:
        target = targets[row["target_id"]]
        mapping = fixture_chain_mapping(config, target, str(row["candidate_id"]))
        target_chain = mapping["target"]
        binder_chain = mapping["binder"]
        target_sequence = structure_chain_sequence(
            Path(target["structure_source_path"]), target_chain
        )
        reference_path = Path(row["design_pose_path"])
        binder_sequence = (
            read_fasta_sequence(Path(row["sequence_path"]))
            if row.get("sequence_path")
            else structure_chain_sequence(reference_path, binder_chain)
        )
        binder_structure_sequence = (
            "C" * len(binder_sequence) if row.get("sequence_path") else binder_sequence
        )
        slug = "-".join(
            re.sub(r"[^a-zA-Z0-9_.-]", "_", str(value))
            for value in (
                row["target_id"],
                row["candidate_id"],
                row["predictor"],
                row["phase"],
                row["seed"],
            )
        )
        record_dir = output_root / "prediction-artifacts" / slug
        record_dir.mkdir(parents=True, exist_ok=True)
        fixture_complex_path = config.get("fixture_predicted_complex_path")
        if fixture_complex_path:
            complex_path = copy_fixture_complex(fixture_complex_path, record_dir)
            predicted_mapping = {"target": target_chain, "binder": binder_chain}
        else:
            complex_path = record_dir / "complex.pdb"
            write_synthetic_structure(
                complex_path,
                {target_chain: target_sequence, binder_chain: binder_structure_sequence},
                geometry_key=str(row["candidate_id"]),
            )
            predicted_mapping = derive_chain_mapping(
                complex_path,
                target_sequence,
                binder_sequence,
                structure_label="fixture predicted structure",
            )
        reference_mapping = derive_chain_mapping(
            reference_path,
            target_sequence,
            binder_sequence,
            structure_label="fixture reference structure",
            declared=(target_chain, binder_chain),
        )
        pae_path = record_dir / "pae.json"
        write_json(
            pae_path,
            {
                "schema_version": 1,
                "target_id": row["target_id"],
                "candidate_id": row["candidate_id"],
                "predictor": row["predictor"],
                "seed": row["seed"],
                "chain_ids": [predicted_mapping["target"], predicted_mapping["binder"]],
                "pae": [[0.0, 4.0], [4.0, 0.0]],
            },
        )
        row["predicted_complex_path"] = str(complex_path.resolve())
        row["predicted_complex_sha256"] = sha256(complex_path)
        row["pae_path"] = str(pae_path.resolve())
        row["pae_sha256"] = sha256(pae_path)
        row["chain_mapping"] = {"target": target_chain, "binder": binder_chain}
        row.update(
            {
                "predicted_target_chain_id": predicted_mapping["target"],
                "predicted_binder_chain_id": predicted_mapping["binder"],
                "reference_target_chain_id": reference_mapping["target"],
                "reference_binder_chain_id": reference_mapping["binder"],
            }
        )
        value = fixture_score(row, predictor_indexes[row["predictor"]])
        negative = row.get("control_type") == "negative"
        custom_metrics = {
            metric["metric_id"]: {
                "value": 0.05 if negative else 0.9,
                "units": metric["units"],
                "implementation_revision": metric["implementation_revision"],
            }
            for metric in config["scoring"].get("metric_registry", [])
            if metric.get("enabled", True)
        }
        measurement = {
            "target_id": row["target_id"],
            "target_sha256": row["target_sha256"],
            "candidate_id": row["candidate_id"],
            "predictor": row["predictor"],
            "model_revision": row["model_revision"],
            "seed": row["seed"],
            "phase": row["phase"],
            "sequence_sha256": row["sequence_sha256"],
            "design_pose_sha256": row["design_pose_sha256"],
            "predicted_complex_sha256": row["predicted_complex_sha256"],
            "pae_sha256": row["pae_sha256"],
            "target_chain_id": target_chain,
            "binder_chain_id": binder_chain,
            "chain_mapping": row["chain_mapping"],
            "predicted_target_chain_id": predicted_mapping["target"],
            "predicted_binder_chain_id": predicted_mapping["binder"],
            "reference_target_chain_id": reference_mapping["target"],
            "reference_binder_chain_id": reference_mapping["binder"],
            "aligned_target_residue_count": 100,
            "target_alignment_rmsd": 0.4,
            "ipsae_implementation_revision": implementation["ipsae_revision"],
            "ipsae_interface_cutoff_angstrom": implementation[
                "ipsae_interface_cutoff_angstrom"
            ],
            "dockq_implementation_revision": implementation["dockq_revision"],
            "site_scorer_revision": implementation["site_scorer_revision"],
            "site_residue_map_sha256": target["site"]["residue_map_sha256"],
            "site_contact_cutoff_angstrom": target["site"]["contact_cutoff_angstrom"],
            "site_atom_selection": target["site"]["atom_selection"],
            "site_metric_basis": implementation["site_metric_basis"],
            "site_hotspot_source": "site-fallback",
            "site_hotspot_relationship": "same-as-site",
            "hotspot_recovery_duplicates_target_contact_recall": True,
            "ipsae_target_to_binder": value,
            "ipsae_binder_to_target": value,
            "ipsae_min": value,
            "sc_dockq": max(0.0, value - 0.08),
            "dockq": max(0.0, value - 0.08),
            "fnat": max(0.0, value - 0.1),
            "interface_rmsd": 1.0,
            "ligand_rmsd": 1.5,
            "mapping_status": "ok",
            "site_contact_iou": 0.05 if negative else 0.75,
            "target_contact_recall": 0.05 if negative else 0.8,
            "target_contact_precision": 0.05 if negative else 0.78,
            "hotspot_recovery": 0.05 if negative else 0.8,
            "offsite_contact_fraction": 0.9 if negative else 0.1,
            "iptm": value,
            "interface_pae": 4.0,
            "interface_plddt": 84.0,
            "clash_count": 0,
            "contact_count": 24,
            "custom_metrics": custom_metrics,
            "status": "scored",
        }
        metric_path = record_dir / "measurement-source.json"
        write_json(metric_path, {"schema_version": 1, "measurement": measurement})
        row["metric_source_path"] = str(metric_path.resolve())
        row["metric_source_sha256"] = sha256(metric_path)
    return rows


def stage_rows(artifact_root: Path, stage_id: str) -> list[dict[str, Any]]:
    receipt = load_json(artifact_root / "receipts" / f"{stage_id}.json")
    artifacts = receipt["output_manifest"]["artifacts"]
    phases = {str(item.get("phase")) for item in artifacts}
    selected_phase = "scale" if "scale" in phases else "single"
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.get("phase") != selected_phase:
            continue
        if artifact.get("kind") != "jsonl":
            # A stage declares the structures it wrote alongside the records it wrote.
            # Only the JSONL artifacts hold rows, and a PDB read as JSON raises here.
            continue
        for file_record in artifact["files"]:
            rows.extend(
                json.loads(line)
                for line in Path(file_record["path"]).read_text().splitlines()
                if line.strip()
            )
    return rows


def observation_from_raw(
    raw: dict[str, Any],
    *,
    attempt_id: str,
    filter_pass: bool,
) -> dict[str, Any]:
    source = load_json(Path(raw["metric_source_path"]))["measurement"]
    control_type = raw.get("control_type", "candidate")
    return {
        **source,
        "generator": "control" if control_type != "candidate" else raw["origin_generator"],
        "score_instrument": score_instrument_arm_name(str(raw["predictor"])),
        "attempt_id": attempt_id,
        "control_type": control_type,
        "control_role": raw.get("control_role"),
        "control_structure_sha256": raw.get("control_structure_sha256"),
        "filter_pass": filter_pass,
        "design_pose_path": raw["design_pose_path"],
        "predicted_complex_path": raw["predicted_complex_path"],
        "pae_path": raw["pae_path"],
        "metric_source_path": raw["metric_source_path"],
        "metric_source_sha256": raw["metric_source_sha256"],
        "raw_prediction_record_sha256": record_hash(raw),
    }


def target_msa_record(artifact_root: Path, target_id: str) -> dict[str, Any]:
    """Return the alignment the MSA stage staged for one target.

    A consuming arm reads the manifest at its published path and looks its target
    up there. It never composes an a3m filename, and a missing manifest raises
    rather than leaving the arm to fold single sequence under a full-arm label.
    """
    manifest_path = artifact_root / "inputs" / "msa-manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"target MSA manifest is missing: {manifest_path}")
    for row in load_jsonl(manifest_path):
        if row.get("target_id") == target_id:
            return row
    raise KeyError(f"target MSA manifest has no row for {target_id}: {manifest_path}")


def raw_prediction_row(
    config: dict[str, Any],
    *,
    artifact_root: Path | None = None,
    target: dict[str, Any],
    candidate_id: str,
    predictor: dict[str, Any],
    seed: int,
    phase: str,
    sequence_sha256: str,
    design_pose_path: str,
    design_pose_sha256: str,
    origin_generator: str | None = None,
    control_type: str | None = None,
    control_role: str | None = None,
    control_structure_sha256: str | None = None,
) -> dict[str, Any]:
    adapter = next(item for item in config["adapters"] if item["adapter_id"] == predictor["adapter_id"])
    source_path = target["structure_source_path"]
    source_sha256 = target["structure_sha256"]
    # Which alignment this row was folded against. An arm whose adapter does not
    # accept the manifest has no MSA encoder, so both fields are null for it rather
    # than naming a file it would ignore.
    msa = (
        target_msa_record(artifact_root, str(target["target_id"]))
        if artifact_root is not None and "target-msa-manifest" in adapter.get("accepted_artifacts", [])
        else None
    )
    row = {
        "target_id": target["target_id"],
        "target_sha256": source_sha256,
        "candidate_id": candidate_id,
        "predictor": predictor["id"],
        "model_revision": adapter["model_revision"],
        "seed": seed,
        "phase": phase,
        "sequence_sha256": sequence_sha256,
        "design_pose_path": design_pose_path,
        "design_pose_sha256": design_pose_sha256,
        "predicted_complex_path": source_path,
        "predicted_complex_sha256": source_sha256,
        "pae_path": source_path,
        "pae_sha256": source_sha256,
        "metric_source_path": source_path,
        "metric_source_sha256": source_sha256,
        # Read from configuration, never assumed. This row used to declare
        # chain A for the target and chain B for the binder outright. Every
        # caller happens to pass the row through `attach_raw_artifacts`, which
        # overwrote the pair from configuration, so the literal never reached a
        # consumer. It was a trap for the next caller that skips that step, and
        # the two functions now read the same rule.
        "chain_mapping": fixture_chain_mapping(config, target, candidate_id),
        "msa_path": msa["msa_path"] if msa else None,
        "msa_sha256": msa["msa_sha256"] if msa else None,
        "status": "scored",
    }
    if origin_generator is not None:
        row["origin_generator"] = origin_generator
    if control_type is not None:
        row.update(
            {
                "control_type": control_type,
                "control_role": control_role,
                "control_structure_sha256": control_structure_sha256,
            }
        )
    return row


def render(value: str, *, attempt_dir: Path, phase: str) -> str:
    return value.replace("{{attempt_dir}}", str(attempt_dir)).replace("{{phase}}", phase)


def stage_record(config: dict[str, Any], stage_id: str) -> dict[str, Any]:
    return next(stage for stage in config["stages"] if stage["stage_id"] == stage_id)


def output_path(config: dict[str, Any], stage_id: str, attempt_dir: Path, phase: str) -> Path:
    stage = stage_record(config, stage_id)
    pattern = render(stage["outputs"][0]["path_template"], attempt_dir=attempt_dir, phase=phase)
    if any(character in pattern for character in "*?["):
        raise ValueError(f"fixture output path cannot be a glob: {pattern}")
    return Path(pattern)


def artifact_output_path(
    config: dict[str, Any],
    stage_id: str,
    artifact_id: str,
    attempt_dir: Path,
    phase: str,
) -> Path:
    stage = stage_record(config, stage_id)
    output = next(item for item in stage["outputs"] if item["artifact_id"] == artifact_id)
    pattern = render(output["path_template"], attempt_dir=attempt_dir, phase=phase)
    if any(character in pattern for character in "*?["):
        raise ValueError(f"fixture output path cannot be a glob: {pattern}")
    return Path(pattern)


def safe_fixture_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def write_fixture_thumbnail(path: Path, *, candidate_id: str, rank_score: float) -> None:
    """Write a deterministic PNG placeholder for one ranked fixture design."""
    width, height = 160, 120
    digest = hashlib.sha256(candidate_id.encode("utf-8")).digest()
    accent = (64 + digest[0] % 160, 64 + digest[1] % 160, 64 + digest[2] % 160)
    score = max(0.0, min(1.0, rank_score))
    bar_width = round((width - 32) * score)
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            border = x in {0, width - 1} or y in {0, height - 1}
            bar = 18 <= y < 30 and 16 <= x < 16 + bar_width
            if border:
                pixel = (32, 32, 32)
            elif bar:
                pixel = accent
            else:
                pixel = (245, 245, 245)
            row.extend(pixel)
        rows.append(bytes(row))

    def png_chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"tEXt", b"candidate\x00" + candidate_id.encode("utf-8"))
        + png_chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + png_chunk(b"IEND", b"")
    )


def write_fixture_structure_pictures(
    config: dict[str, Any],
    stage_id: str,
    artifact_root: Path,
    attempt_dir: Path,
    phase: str,
) -> None:
    """Write the four declared picture artifacts for the ranked fixture designs.

    The renderer role declares an overview and an interface closeup, so each
    ranked design contributes two placeholder images. That also satisfies the
    minimum count of two images when a run ranks a single design.
    """
    ranked = load_json(artifact_root / "scores" / "ranked-candidates.json")
    candidates = ranked.get("ranked_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("fixture renderer produced no ranked designs")

    stage = stage_record(config, stage_id)
    images_output = next(
        item for item in stage["outputs"] if item["artifact_id"] == "structure-pictures"
    )
    images_root = Path(
        render(images_output["path_template"], attempt_dir=attempt_dir, phase=phase)
    ).parent
    manifest_path = artifact_output_path(
        config, stage_id, "structure-picture-manifest", attempt_dir, phase
    )
    index_path = artifact_output_path(
        config, stage_id, "structure-picture-index", attempt_dir, phase
    )
    archive_path = artifact_output_path(
        config, stage_id, "structure-picture-archive", attempt_dir, phase
    )

    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError("fixture renderer received a non-object ranked design")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("fixture renderer received a ranked design without candidate_id")
        rank = int(candidate.get("rank", index))
        rank_score = float(candidate.get("rank_score", 0.0))
        for view in ("overview", "interface"):
            image = images_root / f"rank-{rank:02d}-{safe_fixture_name(candidate_id)}-{view}.png"
            write_fixture_thumbnail(
                image,
                candidate_id=f"{candidate_id}:{view}",
                rank_score=rank_score,
            )
            records.append(
                {
                    "candidate_id": candidate_id,
                    "generator": candidate.get("generator"),
                    "image_path": str(image.resolve()),
                    "image_sha256": sha256(image),
                    "rank": rank,
                    "rank_score": rank_score,
                    "view": view,
                }
            )

    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "renderer": "local-fixture-v1",
            "image_count": len(records),
            "records": records,
            "campaign_id": config.get("campaign_id"),
            "run_id": config.get("run_id"),
            "warnings": [
                "Fixture pictures are deterministic placeholders and are not molecular renders.",
            ],
        },
    )

    rows = "".join(
        f"<tr><td>{record['rank']}</td><td>{record['candidate_id']}</td>"
        f"<td>{record['view']}</td>"
        f"<td><img src=\"images/{Path(record['image_path']).name}\" alt=\"\"></td></tr>"
        for record in records
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        "<!doctype html><meta charset=\"utf-8\">"
        "<title>Fixture structure pictures</title>"
        "<p>Deterministic placeholders, not molecular renders.</p>"
        f"<table><tbody>{rows}</tbody></table>\n"
    )

    # A fixed timestamp keeps the archive byte-identical across runs of the
    # same ranking, which is what the fixture contract test compares.
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for record in sorted(records, key=lambda item: item["image_path"]):
            image = Path(record["image_path"])
            info = zipfile.ZipInfo(f"images/{image.name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, image.read_bytes())


def write_fixture_viewer(config: dict[str, Any], artifact_root: Path, viewer_root: Path) -> None:
    ranked_path = artifact_root / "scores" / "ranked-candidates.json"
    ranked = load_json(ranked_path)
    candidates = ranked.get("ranked_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("fixture renderer produced no ranked designs")

    thumbnails = viewer_root / "thumbnails"
    designs: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError("fixture renderer received a non-object ranked design")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("fixture renderer received a ranked design without candidate_id")
        rank = candidate.get("rank", index)
        rank_score = float(candidate.get("rank_score", 0.0))
        thumbnail = thumbnails / f"rank-{int(rank):02d}-{safe_fixture_name(candidate_id)}.png"
        write_fixture_thumbnail(
            thumbnail,
            candidate_id=candidate_id,
            rank_score=rank_score,
        )
        designs.append(
            {
                "rank": int(rank),
                "candidate_id": candidate_id,
                "generator": candidate.get("generator"),
                "rank_score": rank_score,
                "score_gating_mode": candidate.get("score_gating_mode"),
                "score_gating": candidate.get("score_gating", {}),
                "score_instrument": candidate.get("score_instrument"),
                "predictor_agreement": candidate.get("predictor_agreement"),
                "rank_score_scope": candidate.get("rank_score_scope"),
                "per_seed_by_predictor": candidate.get("per_seed_by_predictor", {}),
                "thumbnail_path": str(thumbnail.resolve()),
            }
        )

    write_json(
        viewer_root / "manifest.json",
        {
            "schema_version": 1,
            "run_id": config.get("run_id"),
            "campaign_id": config.get("campaign_id"),
            "score_gating": ranked.get("score_gating", {}),
            "score_instrument": ranked.get("score_instrument"),
            "predictor_agreement": ranked.get("predictor_agreement"),
            "rank_score_scope": ranked.get("rank_score_scope"),
            "designs": designs,
            "how_to_open": {
                "fixture": "Open the PNG files under viewer/thumbnails for fixture output."
            },
            "warnings": [
                "Fixture thumbnails are deterministic placeholders and are not molecular renders.",
                *[
                    warning
                    for warning in (
                        ranked.get("rank_score_scope"),
                        ranked.get("predictor_agreement"),
                    )
                    if isinstance(warning, str) and warning
                ]
            ],
        },
    )


def _registered_generators(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the enabled generator arms this campaign registers, keyed by id."""
    generation = config.get("generation") or {}
    return {
        str(item["id"]): item
        for item in generation.get("generators") or []
        if isinstance(item, dict) and item.get("enabled") is not False and item.get("id")
    }


def _registered_designers(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the enabled sequence designers this campaign registers, in order."""
    sequence_design = config.get("sequence_design") or {}
    return [
        item
        for item in sequence_design.get("designers") or []
        if isinstance(item, dict) and item.get("enabled") is not False and item.get("id")
    ]


def _generator_mode(config: dict[str, Any], generator: str) -> str:
    entry = _registered_generators(config).get(generator) or {}
    mode = entry.get("mode")
    return str(mode) if mode else "backbone-only"


def _designer_arm(config: dict[str, Any], stage_id: str) -> tuple[str, str]:
    """Return this stage's designer id and the generator arm it reads backbones from.

    Both come from the campaign's own registration. Reading the designer off the
    stage id and the arm off a hard-coded "rfdiffusion" made every designer stage
    claim the first arm's backbones, so a second arm's rows carried the wrong parent
    and the wrong designer and were dropped from the pool with no recorded rejection.
    """
    for designer in _registered_designers(config):
        if str(designer.get("command_stage")) == stage_id:
            arms = [str(arm) for arm in designer.get("compatible_generators") or []]
            return str(designer["id"]), (arms[0] if arms else "rfdiffusion")
    return stage_id.removeprefix("sequence-"), "rfdiffusion"


def candidate_rows(
    config: dict[str, Any],
    stage_id: str,
    count: int,
    sequence_root: Path,
    pose_root: Path,
) -> list[dict[str, Any]]:
    generator = stage_id.removeprefix("generate-")
    mode = _generator_mode(config, generator)
    target = next(item for item in config["targets"] if item["role"] == "primary")
    rows = []
    for index in range(count):
        candidate_id = f"{generator}-{index:03d}"
        row = {
            "target_id": target["target_id"],
            "target_sha256": target["structure_sha256"],
            "candidate_id": candidate_id,
            "parent_candidate_id": None,
            "origin_generator": generator,
            "root_backbone_id": candidate_id,
            "tm90_cluster_id": candidate_id,
            "structure_method": generator,
            "seq_method": "none",
            "fold_class": "unknown",
            "generator_mode": mode,
            "sequence_designer": None,
            "generator_seed": index,
            "structure_path": target["structure_source_path"],
            "structure_sha256": target["structure_sha256"],
            "residue_map_sha256": target["site"]["residue_map_sha256"],
            "optimization_round": 0,
            "last_optimizer": None,
            "status": "generated",
        }
        if mode != "backbone-only":
            write_candidate_sequences([row], sequence_root)
        rows.append(row)
    return write_candidate_design_poses(
        rows,
        pose_root,
        target_structure_path=Path(target["structure_source_path"]),
        target_chain_id=target["chains"][0]["chain_id"],
        binder_chain_id=config["binder"]["binder_chain_id"],
    )


def sequence_rows(
    config: dict[str, Any],
    stage_id: str,
    count: int,
    variants: int,
    sequence_root: Path,
    pose_root: Path,
) -> list[dict[str, Any]]:
    designer, generator = _designer_arm(config, stage_id)
    target = next(item for item in config["targets"] if item["role"] == "primary")
    rows = []
    for backbone_index in range(count):
        parent_id = f"{generator}-{backbone_index:03d}"
        for variant in range(variants):
            candidate_id = f"{parent_id}-{designer}-{variant:02d}"
            rows.append(
                {
                    "target_id": "primary-target",
                    "candidate_id": candidate_id,
                    "parent_candidate_id": parent_id,
                    "origin_generator": generator,
                    "root_backbone_id": parent_id,
                    "tm90_cluster_id": parent_id,
                    "structure_method": generator,
                    "seq_method": designer,
                    "fold_class": "unknown",
                    "sequence_designer": designer,
                    "structure_path": target["structure_source_path"],
                    "structure_sha256": target["structure_sha256"],
                    "status": "sequence-designed",
                }
            )
    return write_candidate_design_poses(
        write_candidate_sequences(rows, sequence_root),
        pose_root,
        target_structure_path=Path(target["structure_source_path"]),
        target_chain_id=target["chains"][0]["chain_id"],
        binder_chain_id=config["binder"]["binder_chain_id"],
    )


def normalized_candidates(
    config: dict[str, Any],
    sequence_root: Path | None = None,
    pose_root: Path | None = None,
) -> list[dict[str, Any]]:
    target = next(item for item in config["targets"] if item["role"] == "primary")
    # Derived from the arms this campaign registers rather than listed. A literal
    # list here produced the same four rows whatever the graph ran, so a second
    # backbone arm's candidates vanished between its designer and the pool with
    # `errors: []` and matching source and normalized counts.
    generators = _registered_generators(config)
    candidates: list[tuple[str, str, str | None]] = []
    for entry in _registered_designers(config):
        designer_id = str(entry["id"])
        for arm in entry.get("compatible_generators") or []:
            if str(arm) not in generators:
                continue
            parent = f"{arm}-000"
            candidates.append((f"{parent}-{designer_id}-00", str(arm), designer_id))
    for generator_id, entry in generators.items():
        if str(entry.get("mode")) == "sequence-structure-codesign":
            candidates.append((f"{generator_id}-000", generator_id, None))
    rows = []
    for candidate_id, generator, designer in candidates:
        parent_id = f"{generator}-000" if designer is not None else None
        rows.append(
            {
                "candidate_id": candidate_id,
                "target_id": target["target_id"],
                "target_sha256": target["structure_sha256"],
                "parent_candidate_id": parent_id,
                "origin_generator": generator,
                "root_backbone_id": parent_id or candidate_id,
                "tm90_cluster_id": parent_id or candidate_id,
                "structure_method": generator,
                "seq_method": designer or "none",
                "fold_class": "unknown",
                "generator_mode": _generator_mode(config, generator),
                "sequence_designer": designer,
                "generator_seed": 0,
                "structure_path": target["structure_source_path"],
                "structure_sha256": target["structure_sha256"],
                "residue_map_sha256": target["site"]["residue_map_sha256"],
                "optimization_round": 0,
                "last_optimizer": None,
                "status": "sequence-designed",
            }
        )
    if sequence_root is not None and pose_root is not None:
        return write_candidate_design_poses(
            write_candidate_sequences(rows, sequence_root),
            pose_root,
            target_structure_path=Path(target["structure_source_path"]),
            target_chain_id=target["chains"][0]["chain_id"],
            binder_chain_id=config["binder"]["binder_chain_id"],
        )
    for row in rows:
        sequence = fixture_sequence(str(row["candidate_id"]))
        row["sequence_path"] = ""
        row["sequence_sha256"] = stable_hash(sequence)
        row["sequence_length"] = len(sequence)
        row["design_pose_path"] = ""
        row["design_pose_sha256"] = stable_hash(str(row["candidate_id"]) + "-pose")
    return rows


PROMOTED_ARMS = ("rfdiffusion", "genie3", "fixture-codesign")


def promoted_pool_rows(config: dict[str, Any], artifact_root: Path | None = None) -> list[dict[str, Any]]:
    """Return the pool row this fixture promotes for each arm in `PROMOTED_ARMS`.

    The arm list is fixed. The candidate id inside an arm is not. A co-design arm
    promotes the generator's own row, and a backbone arm promotes the row its
    sequence designer wrote. Three call sites listed the ids literally and named
    `genie3-000`, which existed only while genie3 was registered as a co-design
    generator. Once genie3 became a backbone generator the pool held
    `genie3-000-proteinmpnn-00` instead and promote raised `KeyError` on an id the
    graph no longer produced. Read the id off the pool the graph actually built.

    Within an arm the promoted row is the one the fixture scores highest, which is
    what the literal list encoded: the rfdiffusion arm promoted its ProteinMPNN row
    at 0.82 over its SolubleMPNN row at 0.81. An arm the campaign does not register
    contributes nothing rather than raising.

    The rows come back in descending screen score, because `lane.py` recomputes the
    promotion order from the score table and refuses a manifest that disagrees. The
    literal list happened to be in that order, which made the agreement a
    coincidence of how the arms were written down rather than a property.
    """
    if lane.intermediate_enabled(config) and artifact_root is not None:
        promotion_path = artifact_root / "promotion" / "promotion-manifest.jsonl"
        if promotion_path.is_file():
            return load_jsonl(promotion_path)
        score_path = artifact_root / "scores" / "intermediate-score-table.jsonl"
        passing_path = artifact_root / "filters" / "passing-candidates.jsonl"
        scores = load_jsonl(score_path)
        passing = {str(row["candidate_id"]): row for row in load_jsonl(passing_path)}
        ranked = lane.rank_candidate_cohort(config, scores, lane.parent_seed_values(config))
        ranked = lane.apply_declared_ranking_mode(config, ranked)
        ranked.sort(key=lambda row: lane._rank_sort_key(row, config))
        optimization = config.get("optimization", {})
        optimization_enabled = isinstance(optimization, dict) and optimization.get("enabled") is True
        requested = (
            int(optimization["parent_count_per_round"])
            if optimization_enabled else int(config["selection"]["final_count"])
        )
        minimum_generators = min(
            int(config["generation"]["minimum_generators"] if optimization_enabled else config["selection"]["minimum_generators"]),
            requested,
        )
        selected, portfolio = lane.select_portfolio(
            ranked,
            final_count=requested,
            minimum_generators=minimum_generators,
            maximum_fraction=float(config["selection"]["maximum_fraction_per_generator"]),
            sort_key=lambda row: lane._rank_sort_key(row, config),
        )
        if not portfolio["ok"]:
            raise ValueError("fixture INTERMEDIATE score cohort cannot fill the parent portfolio")
        return [passing[str(row["candidate_id"])] for row in selected]
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in normalized_candidates(config):
        by_arm.setdefault(str(row["origin_generator"]), []).append(row)

    def screen_score(row: dict[str, Any]) -> float:
        return fixture_score({"candidate_id": row["candidate_id"], "phase": "screen"}, 0)

    promoted = [
        max(by_arm[arm], key=screen_score) for arm in PROMOTED_ARMS if by_arm.get(arm)
    ]
    return sorted(promoted, key=screen_score, reverse=True)


def optimization_rows(
    config: dict[str, Any],
    round_index: int,
    sequence_root: Path | None = None,
    pose_root: Path | None = None,
    artifact_root: Path | None = None,
) -> list[dict[str, Any]]:
    target = next(item for item in config["targets"] if item["role"] == "primary")
    roots = [
        (str(row["candidate_id"]), str(row["origin_generator"]), str(row["seq_method"]))
        for row in promoted_pool_rows(config, artifact_root)
    ]
    rows: list[dict[str, Any]] = []
    for root_candidate_id, generator, seq_method in roots:
        parent_candidate_id = root_candidate_id + "".join(
            f"-opt-r{prior_round}" for prior_round in range(1, round_index)
        )
        candidate_id = f"{parent_candidate_id}-opt-r{round_index}"
        rows.append(
            {
                "target_id": target["target_id"],
                "target_sha256": target["structure_sha256"],
                "candidate_id": candidate_id,
                "parent_candidate_id": parent_candidate_id,
                "root_candidate_id": root_candidate_id,
                "origin_generator": generator,
                "root_backbone_id": root_candidate_id,
                "tm90_cluster_id": root_candidate_id,
                "structure_method": generator,
                "seq_method": seq_method,
                "fold_class": "unknown",
                "last_optimizer": "optimization-controller",
                "optimizer_adapter_id": "optimization-controller",
                "optimization_operation": "point-mutation",
                "optimization_round": round_index,
                "optimizer_seed": 0,
                "variant_index": 0,
                "status": "eligible",
            }
        )
    if sequence_root is not None and pose_root is not None:
        return write_candidate_design_poses(
            write_candidate_sequences(rows, sequence_root),
            pose_root,
            target_structure_path=Path(target["structure_source_path"]),
            target_chain_id=target["chains"][0]["chain_id"],
            binder_chain_id=config["binder"]["binder_chain_id"],
        )
    for row in rows:
        sequence = fixture_sequence(str(row["candidate_id"]))
        row["sequence_path"] = ""
        row["sequence_sha256"] = stable_hash(sequence)
        row["sequence_length"] = len(sequence)
        row["design_pose_path"] = ""
        row["design_pose_sha256"] = stable_hash(str(row["candidate_id"]) + "-pose")
    return rows


def optimization_rows_with_decision(
    config: dict[str, Any],
    artifact_root: Path,
    round_index: int,
    sequence_root: Path | None = None,
    pose_root: Path | None = None,
) -> list[dict[str, Any]]:
    decision_path = (
        artifact_root
        / "optimization"
        / "rounds"
        / f"round-{round_index}"
        / "next-round-decision.json"
    )
    decision = load_json(decision_path)
    if decision.get("stop"):
        # The controller stops a round, and this fixture cannot produce that
        # outcome on its own. Share the controller's carry-forward so the
        # fixture cannot drift from the path a real campaign takes.
        from claude_binder.adapters import optimization_controller

        output_dir = sequence_root.parent if sequence_root is not None else Path(".")
        return optimization_controller.carry_pool_forward(
            config,
            Path(artifact_root),
            round_index,
            decision,
            sha256(decision_path),
            record_hash(decision["parameter_overrides"]),
            output_dir,
        )
    rows = optimization_rows(config, round_index, sequence_root, pose_root, artifact_root)
    for row in rows:
        row["decision_sha256"] = sha256(decision_path)
        row["parameter_set_sha256"] = record_hash(decision["parameter_overrides"])
        # The controller chooses the operation from the campaign's own
        # policy, so the rows carry what it chose rather than an operation
        # this fixture assumes.
        row["optimization_operation"] = decision["operation"]
    return rows


def run_fixture(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    path = output_path(config, args.stage, args.attempt_dir, args.phase)
    stage = stage_record(config, args.stage)
    records_per_count = int(stage["outputs"][0].get("records_per_count", 1))
    if args.stage == "target-prepare":
        target = next(item for item in config["targets"] if item["role"] == "primary")
        write_json(path, {"target_id": target["target_id"], "target_sha256": target["structure_sha256"], "residue_map_sha256": target["site"]["residue_map_sha256"]})
    elif args.stage == "runtime-check":
        write_json(path, {"ok": True, "adapters": [adapter["adapter_id"] for adapter in config["adapters"]]})
    elif args.stage == "stage-msa":
        manifest_path = artifact_output_path(
            config, args.stage, "target-msa-manifest", args.attempt_dir, args.phase
        )
        rows: list[dict[str, Any]] = []
        for target in config["targets"]:
            target_chain_id = str(target["chains"][0]["chain_id"])
            sequence = structure_chain_sequence(Path(target["structure_source_path"]), target_chain_id)
            msa_path = artifact_output_path(
                config,
                args.stage,
                f"target-msa-{target['target_id']}",
                args.attempt_dir,
                args.phase,
            )
            msa_path.parent.mkdir(parents=True, exist_ok=True)
            msa_path.write_text(f">{target['target_id']}\n{sequence}\n")
            rows.append(
                {
                    "target_id": target["target_id"],
                    "target_chain_id": target_chain_id,
                    "msa_path": str(msa_path.resolve()),
                    "msa_sha256": sha256(msa_path),
                    "msa_depth": 1,
                    "msa_source": "fixture",
                    "query_sequence_sha256": stable_hash(sequence),
                }
            )
        write_jsonl(manifest_path, rows)
    elif args.stage == "control-calibration":
        rows = [
            raw_prediction_row(
                config,
                artifact_root=args.artifact_root,
                target=target,
                candidate_id=candidate_id,
                predictor=predictor,
                seed=seed,
                phase="uniform-rescore",
                sequence_sha256=stable_hash(candidate_id + "-sequence"),
                design_pose_path=control["structure_source_path"],
                design_pose_sha256=control["structure_sha256"],
                control_type=control_type,
                control_role=control["role"],
                control_structure_sha256=control["structure_sha256"],
            )
            for candidate_id, control_type, control in (
                (control["id"], group_name, control)
                for group_name in ("positive", "negative")
                for control in config["controls"][group_name]
                if control.get("enabled", True)
            )
            for predictor in config["cofold"]["predictors"]
            if predictor.get("enabled", True)
            for seed in config["cofold"]["rescore_seeds"]
            for target in config["targets"]
        ]
        write_jsonl(
            path,
            attach_raw_artifacts(config, rows, args.attempt_dir / args.phase),
        )
    elif args.stage.startswith("generate-"):
        write_jsonl(
            path,
            candidate_rows(
                config,
                args.stage,
                args.count,
                args.attempt_dir / args.phase / "sequences",
                args.attempt_dir / args.phase / "poses",
            ),
        )
    elif args.stage.startswith("sequence-"):
        write_jsonl(
            path,
            sequence_rows(
                config,
                args.stage,
                args.count,
                records_per_count,
                args.attempt_dir / args.phase / "sequences",
                args.attempt_dir / args.phase / "poses",
            ),
        )
    elif args.stage == "normalize-candidates":
        write_jsonl(
            path,
            normalized_candidates(
                config,
                args.attempt_dir / args.phase / "sequences",
                args.attempt_dir / args.phase / "poses",
            ),
        )
    elif args.stage in {"filter-integrity", "filter-novelty"} or args.stage.startswith(
        "optimization-filter-"
    ):
        logical_stage_id = stage.get("logical_filter_stage_id", args.stage)
        observation_artifact = (
            "integrity-filter-observations"
            if logical_stage_id == "filter-integrity"
            else "novelty-filter-observations"
        )
        passing_artifact = (
            "integrity-passing-candidates"
            if logical_stage_id == "filter-integrity"
            else "passing-candidates"
        )
        round_index = int(stage.get("optimization_round", 0))
        candidates = load_jsonl(
            args.artifact_root / "candidates" / "candidate-manifest.jsonl"
            if round_index == 0
            else args.artifact_root
            / "optimization"
            / "rounds"
            / f"round-{round_index}"
            / "optimized-candidates.jsonl"
        )
        contracts = [
            contract
            for contract in config["filters"]["contracts"]
            if contract["stage_id"] == logical_stage_id
            and contract["filter_id"] not in set(config["filters"].get("disabled_checks", []))
        ]
        declared_outputs = {
            str(output.get("artifact_id"))
            for output in stage.get("outputs", [])
            if isinstance(output, dict)
        }
        observations = [
            {
                "candidate_id": row["candidate_id"],
                "origin_generator": row["origin_generator"],
                "sequence_sha256": row["sequence_sha256"],
                "optimization_round": round_index,
                "filter_id": contract["filter_id"],
                "metric": contract["metric"],
                "operator": contract["operator"],
                "threshold": contract["threshold"],
                "value": 1.0,
                "pass": True,
                "reason": "fixture threshold met",
                "tool_revision": contract["tool_revision"],
                "reference_revision": contract["reference_revision"],
                "reference_sha256": contract["reference_sha256"],
            }
            for row in candidates
            for contract in contracts
        ]
        if observation_artifact in declared_outputs:
            write_jsonl(
                artifact_output_path(config, args.stage, observation_artifact, args.attempt_dir, args.phase),
                observations,
            )
        passing_rows = [
            {
                **row,
                "filter_pass": True,
                "failed_checks": [],
            }
            for row in candidates
        ]
        write_jsonl(
            artifact_output_path(config, args.stage, passing_artifact, args.attempt_dir, args.phase),
            passing_rows,
        )
        report_artifact = (
            "integrity-filter-report"
            if logical_stage_id == "filter-integrity"
            else "novelty-filter-report"
        )
        if report_artifact in declared_outputs:
            write_json(
                artifact_output_path(config, args.stage, report_artifact, args.attempt_dir, args.phase),
                filter_report(
                    logical_stage_id,
                    [str(row["candidate_id"]) for row in candidates],
                    contracts,
                    observations,
                    [str(row["candidate_id"]) for row in passing_rows],
                    skipped_gates=(
                        retired_filter_statuses()
                        if logical_stage_id == "filter-novelty"
                        else None
                    ),
                ),
            )
    elif args.stage.startswith("cofold-screen-"):
        predictor = args.stage.removeprefix("cofold-screen-")
        candidates = load_jsonl(args.artifact_root / "filters" / "passing-candidates.jsonl")
        predictor_record = next(item for item in config["cofold"]["predictors"] if item["id"] == predictor)
        rows = [
                raw_prediction_row(
                    config,
                    artifact_root=args.artifact_root,
                    target=target,
                    candidate_id=candidate["candidate_id"],
                    predictor=predictor_record,
                    seed=seed,
                    phase="screen",
                    sequence_sha256=candidate["sequence_sha256"],
                    design_pose_path=candidate["design_pose_path"],
                    design_pose_sha256=candidate["design_pose_sha256"],
                    origin_generator=candidate["origin_generator"],
                )
                for candidate in candidates[:args.count]
                for target in config["targets"]
                for seed in config["cofold"]["screen_seeds"]
            ]
        write_jsonl(path, attach_raw_artifacts(config, rows, args.attempt_dir / args.phase))
    elif args.stage.startswith("cofold-intermediate-"):
        predictor = args.stage.removeprefix("cofold-intermediate-")
        candidates = load_jsonl(args.artifact_root / "scores" / "intermediate-candidates.jsonl")
        predictor_record = next(item for item in config["cofold"]["predictors"] if item["id"] == predictor)
        rows = [
            raw_prediction_row(
                config,
                artifact_root=args.artifact_root,
                target=target,
                candidate_id=candidate["candidate_id"],
                predictor=predictor_record,
                seed=seed,
                phase="intermediate",
                sequence_sha256=candidate["sequence_sha256"],
                design_pose_path=candidate["design_pose_path"],
                design_pose_sha256=candidate["design_pose_sha256"],
                origin_generator=candidate["origin_generator"],
            )
            for candidate in candidates[:args.count]
            for target in config["targets"]
            for seed in config["cofold"]["rescore_seeds"]
        ]
        write_jsonl(path, attach_raw_artifacts(config, rows, args.attempt_dir / args.phase))
    elif args.stage == "score-screen":
        raw_rows = [
            row
            for predictor in config["cofold"]["predictors"]
            if predictor.get("enabled", True)
            for row in stage_rows(args.artifact_root, f"cofold-screen-{predictor['id']}")
        ]
        write_jsonl(
            path,
            [
                {
                    **observation_from_raw(
                        raw,
                        attempt_id="fixture-screen-score",
                        filter_pass=True,
                    ),
                    "origin_generator": raw["origin_generator"],
                }
                for raw in raw_rows
            ],
        )
    elif args.stage == "score-intermediate":
        raw_rows = [
            row
            for predictor in config["cofold"]["predictors"]
            if predictor.get("enabled", True)
            for row in stage_rows(args.artifact_root, f"cofold-intermediate-{predictor['id']}")
        ]
        write_jsonl(
            path,
            [
                {
                    **observation_from_raw(
                        raw,
                        attempt_id="fixture-intermediate-score",
                        filter_pass=True,
                    ),
                    "origin_generator": raw["origin_generator"],
                }
                for raw in raw_rows
            ],
        )
    elif args.stage == "promote":
        passing = {
            row["candidate_id"]: row
            for row in load_jsonl(args.artifact_root / "filters" / "passing-candidates.jsonl")
        }
        promoted = [
                {
                    **passing[candidate_id],
                    "promotion_status": "promoted",
                    "promotion_reason": "fixture",
                }
                for candidate_id in (
                    str(row["candidate_id"]) for row in promoted_pool_rows(config, args.artifact_root)
                )
            ]
        write_jsonl(path, promoted)
        optimization = config.get("optimization")
        requested = (
            int(optimization["parent_count_per_round"])
            if isinstance(optimization, dict) and optimization.get("enabled") is True
            else int(config["selection"]["final_count"])
        )
        policy = config.get("selection", {}).get("shortfall_policy", {})
        minimum_fraction = float(policy.get("minimum_delivery_fraction", 0.5))
        delivered = len(promoted)
        write_json(
            artifact_output_path(
                config,
                args.stage,
                "promotion-summary",
                args.attempt_dir,
                args.phase,
            ),
            {
                "schema_version": 1,
                "requested_count": requested,
                "delivered_count": delivered,
                "shortfall_count": requested - delivered,
                "binding_rules": [] if delivered >= requested else ["fixture candidate pool"],
                "minimum_delivery_fraction": minimum_fraction,
                "delivery_fraction": delivered / requested,
                "policy_satisfied": True,
                "shortfall_allowed": delivered > 0 and delivered / requested >= minimum_fraction,
                "selected_candidate_ids": [row["candidate_id"] for row in promoted],
                "scoring_arm_status": "fixture",
                "ranking_claim_status": "fixture",
            },
        )
    elif args.stage.startswith("optimization-plan-round-"):
        # A hand-written next-round-decision.json with fixed parents and no
        # outcome leaves the controller's stop logic, parent ordering and
        # early-stop margin unexercised. The free graph produces both score
        # tables the controller reads, so delegate the decision to the
        # controller that a real campaign uses.
        #
        # The controller cannot plan a round for a campaign with more than one
        # target: raw_score_vectors keys on candidate, predictor and seed
        # without the target, so the same candidate scored against a second
        # target collides and it refuses. Its sibling
        # validate_round_score_matrix does include the target, which is how the
        # omission shows. How a multi-target round aggregates is a scientific
        # choice rather than a mechanical one, so that campaign shape keeps the
        # fixture's own decision and does not exercise the controller.
        if len([item for item in config.get("targets", []) if item.get("role") == "primary"]) <= 1 and len(config.get("targets", [])) <= 1:
            from claude_binder.adapters import optimization_controller

            return optimization_controller.plan_stage(args)
        else:
            round_index = int(args.stage.rsplit("-", 1)[1])
            summary_path = artifact_output_path(config, args.stage, "round-summary", args.attempt_dir, args.phase)
            decision_path = artifact_output_path(config, args.stage, "next-round-decision", args.attempt_dir, args.phase)
            if round_index == 1:
                source_manifest = args.artifact_root / "promotion" / "promotion-manifest.jsonl"
                summary_candidates = [
                    {"candidate_id": row["candidate_id"], "origin_generator": row["origin_generator"], "status": "promoted"}
                    for row in promoted_pool_rows(config)
                ]
            else:
                source_manifest = (
                    args.artifact_root
                    / "optimization"
                    / "rounds"
                    / f"round-{round_index - 1}"
                    / "eligible-parents.jsonl"
                )
                summary_candidates = [
                    {"candidate_id": row["candidate_id"], "origin_generator": row["origin_generator"], "status": "eligible"}
                    for row in optimization_rows(config, round_index - 1)
                ]
            summary = {
                "schema_version": 1,
                "round": round_index,
                "source_manifest_sha256": sha256(source_manifest),
                "candidates": summary_candidates,
                "budget": {"remaining_candidates": 3, "remaining_predictions": 100},
                "failures": [],
            }
            write_json(summary_path, summary)
            write_json(
                decision_path,
                {
                    "schema_version": 1,
                    "round": round_index,
                    "summary_sha256": sha256(summary_path),
                    "config_sha256": sha256(args.config),
                    "selected_parent_ids": [row["candidate_id"] for row in summary_candidates],
                    "adapter_id": "optimization-controller",
                    "operation": "point-mutation",
                    "parameter_overrides": {"mutation_count": 1},
                    "seeds": [0],
                    "candidate_count": 3,
                    "expected_fanout": 3,
                    "stop": False,
                    "stop_reason": None,
                },
            )
    elif args.stage.startswith("optimize-round-"):
        round_index = int(args.stage.rsplit("-", 1)[1])
        write_jsonl(
            path,
            optimization_rows_with_decision(
                config,
                args.artifact_root,
                round_index,
                args.attempt_dir / args.phase / "sequences",
                args.attempt_dir / args.phase / "poses",
            ),
        )
    elif args.stage.startswith("optimization-cofold-round-"):
        remainder = args.stage.removeprefix("optimization-cofold-round-")
        round_text, predictor_id = remainder.split("-", 1)
        round_index = int(round_text)
        predictor_record = next(item for item in config["cofold"]["predictors"] if item["id"] == predictor_id)
        candidates = load_jsonl(
            args.artifact_root
            / "optimization"
            / "rounds"
            / f"round-{round_index}"
            / "filters"
            / "passing-candidates.jsonl"
        )
        rows = [
                raw_prediction_row(
                    config,
                    artifact_root=args.artifact_root,
                    target=target,
                    candidate_id=candidate["candidate_id"],
                    predictor=predictor_record,
                    seed=seed,
                    phase="optimization",
                    sequence_sha256=candidate["sequence_sha256"],
                    design_pose_path=candidate["design_pose_path"],
                    design_pose_sha256=candidate["design_pose_sha256"],
                    origin_generator=candidate["origin_generator"],
                )
                for candidate in candidates[:args.count]
                for target in config["targets"]
                for seed in lane.parent_seed_values(config)
            ]
        write_jsonl(path, attach_raw_artifacts(config, rows, args.attempt_dir / args.phase))
    elif args.stage.startswith("optimization-measure-round-"):
        round_index = int(args.stage.rsplit("-", 1)[1])
        raw_rows = [
            row
            for predictor in config["cofold"]["predictors"]
            if predictor.get("enabled", True)
            for row in stage_rows(
                args.artifact_root,
                f"optimization-cofold-round-{round_index}-{predictor['id']}",
            )
        ]
        write_jsonl(
            path,
            [
                {
                    **observation_from_raw(
                        raw,
                        attempt_id=f"fixture-optimization-round-{round_index}",
                        filter_pass=True,
                    ),
                    "origin_generator": raw["origin_generator"],
                }
                for raw in raw_rows
            ],
        )
    elif args.stage.startswith("optimization-select-round-"):
        round_index = int(args.stage.rsplit("-", 1)[1])
        stage = stage_record(config, args.stage)
        output_artifact_id = stage["outputs"][0]["artifact_id"]
        if output_artifact_id == "rescore-candidates":
            candidates = lane.optimization_scored_candidate_pool(config, args.artifact_root)
        else:
            candidates = load_jsonl(
                args.artifact_root
                / "optimization"
                / "rounds"
                / f"round-{round_index}"
                / "filters"
                / "passing-candidates.jsonl"
            )
            if lane.intermediate_enabled(config):
                score_rows = load_jsonl(
                    args.artifact_root / "optimization" / "rounds"
                    / f"round-{round_index}" / "score-table.jsonl"
                )
                ranked = lane.rank_candidate_cohort(config, score_rows, lane.parent_seed_values(config))
                ranked = lane.apply_declared_ranking_mode(config, ranked, complete_only=True)
                ranked.sort(key=lambda row: lane._rank_sort_key(row, config))
                requested = int(config["optimization"]["parent_count_per_round"])
                selected, portfolio = lane.select_portfolio(
                    ranked,
                    final_count=requested,
                    minimum_generators=min(int(config["generation"]["minimum_generators"]), requested),
                    maximum_fraction=float(config["selection"]["maximum_fraction_per_generator"]),
                    sort_key=lambda row: lane._rank_sort_key(row, config),
                )
                if not portfolio["ok"]:
                    raise ValueError("fixture optimization score cohort cannot fill the next parent portfolio")
                by_id = {str(row["candidate_id"]): row for row in candidates}
                candidates = [dict(by_id[str(row["candidate_id"])], status="eligible") for row in selected]
            else:
                for candidate in candidates:
                    candidate["status"] = "eligible"
                candidates.sort(key=lambda row: str(row["candidate_id"]))
        write_jsonl(path, candidates)
    elif args.stage.startswith("cofold-rescore-"):
        predictor = args.stage.removeprefix("cofold-rescore-")
        predictor_record = next(item for item in config["cofold"]["predictors"] if item["id"] == predictor)
        candidates = load_jsonl(args.artifact_root / "optimization" / "rescore-candidates.jsonl")
        rows = [
                raw_prediction_row(
                    config,
                    artifact_root=args.artifact_root,
                    target=target,
                    candidate_id=candidate["candidate_id"],
                    predictor=predictor_record,
                    seed=seed,
                    phase="uniform-rescore",
                    sequence_sha256=candidate["sequence_sha256"],
                    design_pose_path=candidate["design_pose_path"],
                    design_pose_sha256=candidate["design_pose_sha256"],
                    origin_generator=candidate["origin_generator"],
                )
                for candidate in candidates[:args.count]
                for target in config["targets"]
                for seed in config["cofold"]["rescore_seeds"]
            ]
        write_jsonl(path, attach_raw_artifacts(config, rows, args.attempt_dir / args.phase))
    elif args.stage == "uniform-rescore":
        raw_rows = [
            row
            for predictor in config["cofold"]["predictors"]
            if predictor.get("enabled", True)
            for row in stage_rows(args.artifact_root, f"cofold-rescore-{predictor['id']}")
        ]
        raw_rows.extend(stage_rows(args.artifact_root, "control-calibration"))
        write_jsonl(
            path,
            [
                observation_from_raw(
                    raw,
                    attempt_id="fixture-uniform-rescore",
                    filter_pass=True,
                )
                for raw in raw_rows
            ],
        )
    elif args.stage == "output-check":
        ranked_path = args.artifact_root / "scores" / "ranked-candidates.json"
        ranked = load_json(ranked_path)
        plan = load_json(args.plan)
        write_json(path, {"ok": ranked.get("ok") is True, "run_fingerprint": plan["run_fingerprint"], "stage_count": len(config["stages"]), "selected_count": len(ranked.get("selected_candidates", []))})
    elif args.stage == "render-structure-pictures":
        write_fixture_structure_pictures(
            config,
            args.stage,
            args.artifact_root,
            args.attempt_dir,
            args.phase,
        )
    elif args.stage == "render-viewer":
        write_fixture_viewer(config, args.artifact_root, args.attempt_dir / args.phase / "viewer")
    else:
        raise ValueError(f"fixture stage is not implemented: {args.stage}")
    return 0


def parse_fixture(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    stage = stage_record(config, args.stage)
    files: list[Path] = []
    parsed_count = 0
    errors: list[str] = []
    for output in stage["outputs"]:
        pattern = render(output["path_template"], attempt_dir=args.attempt_dir, phase=args.phase)
        for value in sorted(glob.glob(pattern, recursive=True)):
            path = Path(value)
            if not path.is_file():
                continue
            files.append(path)
            try:
                if output["kind"] == "jsonl":
                    parsed_count += sum(1 for line in path.read_text().splitlines() if line.strip())
                elif output["kind"] == "fasta":
                    parsed_count += sum(
                        1 for line in path.read_text().splitlines() if line.startswith(">")
                    )
                elif output["kind"] == "json":
                    json.loads(path.read_text())
                    parsed_count += 1
                else:
                    parsed_count += 1
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    parser_path = args.attempt_dir / args.phase / "parser-result.json"
    write_json(
        parser_path,
        {
            "ok": bool(files) and not errors,
            "parsed_count": parsed_count,
            "rejected_count": len(errors),
            "errors": errors,
            "source_output_hashes": sorted(sha256(path) for path in files),
        },
    )
    return 0 if files and not errors else 1


def main() -> int:
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
    args = parser.parse_args()
    if args.command == "toolcheck":
        print("binder lane fixture adapter ok")
        return 0
    return run_fixture(args) if args.command == "run" else parse_fixture(args)


if __name__ == "__main__":
    raise SystemExit(main())
