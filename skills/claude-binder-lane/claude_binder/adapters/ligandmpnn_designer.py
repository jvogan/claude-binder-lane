#!/usr/bin/env python3
"""Design LigandMPNN sequences for one binder lane stage phase.

LigandMPNN runs from a local checkout. This wrapper has one route, and it is the
checkout named by --ligandmpnn-root or LIGANDMPNN_ROOT. There is no hosted route
here, so nothing in this module builds a provider request.

The wrapper reads backbone candidates from a completed upstream receipt or from
a published manifest, runs `run.py` once per backbone, and writes receipt-owned
outputs into the current attempt directory:

  <attempt>/<phase>/sequences/<candidate_id>.fasta      one record per candidate
  <attempt>/<phase>/poses/<candidate_id>.pdb            one design pose per candidate
  <attempt>/<phase>/sequence-candidate-manifest.jsonl   one row per candidate

Ownership validation rejects a row that points at an upstream file, so the
wrapper writes its own FASTA and its own design pose for every candidate. The
design pose carries the backbone coordinates the sequence was designed for, and
a REMARK line that names the candidate and the upstream pose hash.

This wrapper guards four LigandMPNN behaviours:

Naming a chain LigandMPNN cannot design exits 0 and writes the full output tree,
including one backbone PDB per requested design, and every returned sequence is
the native sequence. The native FASTA header reads `num_res=0` in that case. The
wrapper reads `num_res` and refuses when it is not above zero, because counting
files and reading the exit code passes this run.

`run.py` reads a seed of zero as a request for a random seed, and zero is the
default, so a run without an explicit nonzero seed is not reproducible. The
wrapper replaces a zero seed with a fixed nonzero value, always passes --seed,
and records both numbers.

`--fasta_seq_separation` defaults to a colon, so a record that carries more than
one chain joins its chains with a colon. The header records no chain order, so
the wrapper resolves the segment index from the chains it passed to
--chains_to_design and refuses when the segment count disagrees.

`--out_folder` has no default and `run.py` subscripts it immediately, so the
wrapper always passes it.

A profile pins the checkpoint in its model_revision field. The wrapper hashes
the checkpoint it is about to load and refuses when the two disagree, naming
both digests. Reach the pinned value with --model-revision, or with --config
pointing at the resolved run config. Without either one the wrapper still hashes
the file, records the digest on every candidate row, and marks the row
`model_revision_verified: unrecorded`, because a profile that pins nothing makes
no claim to check.

Every command is an argument list that runs with shell=False. The wrapper builds
no shell string.

Install LigandMPNN from https://github.com/dauparas/LigandMPNN and point the
wrapper at the checkout with --ligandmpnn-root or LIGANDMPNN_ROOT. The
checkpoints are `.pt` files under a directory, `model_params` inside the
checkout by default.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from claude_binder import backbone_shape
# A text converter for an upstream pose the generator wrote as mmCIF. The
# function is format code with no provider call in it, and duplicating a second
# converter would give the two adapters two answers for one file.
from claude_binder.clients.fal_mpnn_client import cif_to_pdb
from claude_binder.adapters.candidate_lineage import DIVERSITY_LINEAGE_FIELDS

RUNNER_NAME = "run.py"
DEFAULT_MODEL_TYPE = "ligand_mpnn"
# One checkpoint flag per --model_type value. `run.py` reads the flag that
# matches the model type and ignores the others.
MODEL_TYPE_CHECKPOINT_FLAGS = {
    "protein_mpnn": "--checkpoint_protein_mpnn",
    "ligand_mpnn": "--checkpoint_ligand_mpnn",
    "soluble_mpnn": "--checkpoint_soluble_mpnn",
    "per_residue_label_membrane_mpnn": "--checkpoint_per_residue_label_membrane_mpnn",
    "global_label_membrane_mpnn": "--checkpoint_global_label_membrane_mpnn",
}
MODEL_TYPES = tuple(MODEL_TYPE_CHECKPOINT_FLAGS)
# Checkpoint file names this package has run. A model type absent from this map
# has no default here, and --model-name names its checkpoint instead.
DEFAULT_CHECKPOINT_NAMES = {
    "ligand_mpnn": "ligandmpnn_v_32_010_25",
    "protein_mpnn": "proteinmpnn_v_48_020",
}
DEFAULT_WEIGHTS_SUBDIR = "model_params"
DEFAULT_MANIFEST_NAME = "sequence-candidate-manifest.jsonl"
DEFAULT_SEQUENCE_SUBDIR = "sequences"
DEFAULT_POSE_SUBDIR = "poses"
DEFAULT_WORK_SUBDIR = "ligandmpnn"
DEFAULT_DESIGNER_ID = "ligandmpnn"
ROOT_ENVIRONMENT_KEY = "LIGANDMPNN_ROOT"
# LigandMPNN reads a false-y seed as a request for a random seed, at run.py:31,
# so a run with --seed 0 is not reproducible and neither is a run that omits the
# flag. The wrapper substitutes this value and records both the requested seed
# and the seed the tool received.
SEED_ZERO_REPLACEMENT = 1000003
# The default of --fasta_seq_separation. ProteinMPNN joins chains with a slash
# and LigandMPNN joins them with this, so the two wrappers split on different
# characters.
FASTA_CHAIN_SEPARATOR = ":"
ATOM_RECORD_PREFIXES = ("ATOM  ", "HETATM")
CANONICAL_AMINO_ACID_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
HEADER_FLOAT_RE = re.compile(r"(?:^|,)\s*([a-z_]+)=(-?\d+(?:\.\d+)?)")
# The residue count the native header records. `num_ligand_res` does not match,
# because the key has to start right after the comma.
NATIVE_RESIDUE_COUNT_RE = re.compile(r"(?:^|,)\s*num_res=(\d+)")
# A profile pins the checkpoint inside its free-text model_revision, as
# `<relative path> sha256:<64 hex>`. These two read that token.
MODEL_REVISION_DIGEST_RE = re.compile(r"sha256:(\S*)")
SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")
READ_BLOCK_BYTES = 1024 * 1024
# Fields a sequence-designed row copies from its backbone parent. Lineage
# validation compares origin_generator, structure_path, and structure_sha256
# against the parent row and rejects any change.
PARENT_LINEAGE_FIELDS = (
    "target_id",
    "target_sha256",
    "origin_generator",
    "generator_mode",
    "generator_seed",
    "residue_map_sha256",
    "structure_path",
    "structure_sha256",
    # A sequence designed onto a backbone sits in the same optimization round as that
    # backbone, and no optimizer has touched it yet either. The generator writes both
    # fields and the normalizer requires both, so inheriting them keeps one record of
    # where a candidate sits in the campaign rather than two that can disagree.
    "optimization_round",
    "last_optimizer",
    *DIVERSITY_LINEAGE_FIELDS,
)
# Numeric fields a LigandMPNN design header records, and the manifest field each
# one lands in.
HEADER_SCORE_FIELDS = (
    ("overall_confidence", "ligandmpnn_overall_confidence"),
    ("ligand_confidence", "ligandmpnn_ligand_confidence"),
    ("seq_rec", "ligandmpnn_sequence_recovery"),
)


class AdapterError(RuntimeError):
    """A condition the operator has to fix before the stage can run."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of the complete file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sequence_sha256(sequence: str) -> str:
    """Return the SHA-256 of the canonical residue string."""
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return the JSON object rows of a JSONL file."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AdapterError(f"JSONL line {line_number} is not a JSON object: {path}")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows to a path in one atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def resolve_root(value: Path | None) -> Path:
    """Return the LigandMPNN checkout directory."""
    if value is None:
        environment_value = os.environ.get(ROOT_ENVIRONMENT_KEY, "").strip()
        if not environment_value:
            raise AdapterError(
                "LigandMPNN is not located. Pass --ligandmpnn-root, or set "
                f"{ROOT_ENVIRONMENT_KEY} to a checkout of "
                "https://github.com/dauparas/LigandMPNN"
            )
        value = Path(environment_value)
    root = value.expanduser()
    if not root.is_dir():
        raise AdapterError(f"LigandMPNN root is not a directory: {root}")
    return root.resolve()


def resolve_runner(root: Path) -> Path:
    """Return the LigandMPNN runner script inside a checkout."""
    runner = root / RUNNER_NAME
    if not runner.is_file():
        raise AdapterError(
            f"LigandMPNN runner not found: {runner}. Point --ligandmpnn-root at a checkout "
            f"that contains {RUNNER_NAME}"
        )
    return runner


def checkpoint_flag(model_type: str) -> str:
    """Return the checkpoint flag that carries the weights for one model type."""
    flag = MODEL_TYPE_CHECKPOINT_FLAGS.get(model_type)
    if flag is None:
        raise AdapterError(
            f"model type {model_type!r} is not supported; choose one of {', '.join(MODEL_TYPES)}"
        )
    return flag


def checkpoint_name(model_type: str, override: str | None) -> str:
    """Return the checkpoint file stem for one model type."""
    if override:
        return override
    name = DEFAULT_CHECKPOINT_NAMES.get(model_type)
    if name is None:
        raise AdapterError(
            f"model type {model_type} has no default checkpoint in this package. Pass "
            "--model-name with the checkpoint file name, without the .pt suffix"
        )
    return name


def resolve_checkpoint(
    root: Path, override: Path | None, model_name: str | None, model_type: str
) -> tuple[Path, Path]:
    """Return the weights directory and the checkpoint file for one model type.

    LigandMPNN takes the checkpoint as a file path rather than a directory, so
    the directory here only locates the file and never reaches the runner.
    """
    weights_dir = override.expanduser() if override is not None else root / DEFAULT_WEIGHTS_SUBDIR
    if not weights_dir.is_dir():
        raise AdapterError(f"LigandMPNN weights directory not found: {weights_dir}")
    checkpoint = weights_dir / f"{checkpoint_name(model_type, model_name)}.pt"
    if not checkpoint.is_file():
        raise AdapterError(f"LigandMPNN checkpoint not found: {checkpoint}")
    return weights_dir.resolve(), checkpoint.resolve()


def recorded_checkpoint_digest(model_revision: str | None) -> str | None:
    """Return the sha256 a model_revision string pins, or None when it pins none.

    A profile records model_revision as free text. The digest is the only part
    of that string this wrapper can hold against a file, so it reads the token
    and ignores the rest. A string with no `sha256:` token pins no digest.
    """
    if not model_revision:
        return None
    match = MODEL_REVISION_DIGEST_RE.search(model_revision)
    if match is None:
        return None
    digest = match.group(1).lower()
    if not SHA256_HEX_RE.fullmatch(digest):
        raise AdapterError(
            f"model_revision pins {match.group(0)!r}, which is not 64 hexadecimal characters. "
            "Record the sha256 of the checkpoint this adapter loads, or drop the sha256 token "
            "so the run records the checkpoint as unverified"
        )
    return digest


def verify_checkpoint_digest(checkpoint: Path, model_revision: str | None) -> dict[str, str]:
    """Hold the checkpoint on disk against the digest a profile pins.

    Without this a run could succeed and attribute every sequence to a
    checkpoint that never loaded. The check runs once per wrapper invocation,
    before the first design call, over the whole file. It does not run once per
    backbone, because the checkpoint cannot change between subprocesses of one
    phase.
    """
    observed = sha256_file(checkpoint)
    recorded = recorded_checkpoint_digest(model_revision)
    if recorded is None:
        return {"state": "unrecorded", "observed": observed, "recorded": ""}
    if recorded != observed:
        raise AdapterError(
            f"checkpoint {checkpoint} hashes to {observed}, and model_revision pins {recorded}. "
            "The file on disk is not the checkpoint this run would attribute its sequences to. "
            "Point --weights-dir at the pinned checkpoint, or record the digest of the file you "
            "mean to run"
        )
    return {"state": "matched", "observed": observed, "recorded": recorded}


def generator_for_backbone_stage(config: dict, backbone_stage_id: str | None) -> str | None:
    """Return the generator id whose command stage produced these backbones.

    A designer invocation names the stage it reads backbones from, and a generator
    registration names the same stage in `command_stage`. That is the only thing in the
    argv that says which arm this invocation is, so it is what disambiguates a tool bound
    to two of them.
    """
    if not backbone_stage_id:
        return None
    for generator in config.get("generation", {}).get("generators", []) or []:
        if isinstance(generator, dict) and generator.get("command_stage") == backbone_stage_id:
            identifier = generator.get("id")
            return identifier if isinstance(identifier, str) and identifier else None
    return None


def config_model_revision(
    config_path: Path,
    designer_id: str,
    backbone_stage_id: str | None = None,
    adapter_id: str | None = None,
) -> str:
    """Return the model_revision a resolved config records for one designer.

    `adapter_id` names the registration outright and wins when given. Otherwise
    the arm is derived from the backbone stage this invocation reads, because a
    campaign may register one designer id on more than one arm.
    """
    config = json.loads(config_path.read_text())
    designers = config.get("sequence_design", {}).get("designers", [])
    matches = [
        item for item in designers if isinstance(item, dict) and item.get("id") == designer_id
    ]
    if len(matches) > 1 and adapter_id:
        matches = [item for item in matches if item.get("adapter_id") == adapter_id]
    if len(matches) > 1:
        arm = generator_for_backbone_stage(config, backbone_stage_id)
        if arm:
            narrowed = [
                item
                for item in matches
                if isinstance(item.get("compatible_generators"), list)
                and arm in item["compatible_generators"]
            ]
            if len(narrowed) == 1:
                matches = narrowed
    if len(matches) != 1:
        raise AdapterError(
            f"{config_path} registers {len(matches)} sequence designers with id {designer_id}. "
            "Pass --adapter-id to name one, or --model-revision"
        )
    adapter_id = matches[0].get("adapter_id")
    for adapter in config.get("adapters", []):
        if isinstance(adapter, dict) and adapter.get("adapter_id") == adapter_id:
            revision = adapter.get("model_revision")
            if not isinstance(revision, str) or not revision:
                raise AdapterError(f"adapter {adapter_id} records no model_revision")
            return revision
    raise AdapterError(f"{config_path} registers no adapter {adapter_id}. Pass --model-revision")


def resolve_model_revision(args: argparse.Namespace) -> str | None:
    """Return the model_revision string this invocation checks against."""
    if args.model_revision is not None:
        return args.model_revision
    config_path = getattr(args, "config", None)
    if config_path is None:
        return None
    resolved = config_path.expanduser()
    if not resolved.is_file():
        raise AdapterError(f"--config does not exist: {resolved}")
    return config_model_revision(
        resolved.resolve(),
        getattr(args, "designer_id", DEFAULT_DESIGNER_ID),
        getattr(args, "backbone_stage_id", None),
        getattr(args, "adapter_id", None),
    )


def report_checkpoint_digest(checkpoint: Path, verification: dict[str, str]) -> None:
    """Print what the digest check found, including the case that checked nothing."""
    print(f"ligandmpnn adapter: checkpoint {checkpoint}")
    if verification["state"] == "matched":
        print(f"ligandmpnn adapter: sha256 {verification['observed']} matches model_revision")
        return
    print(
        f"ligandmpnn adapter: sha256 {verification['observed']} is unverified, because no "
        "model_revision reached this wrapper with a sha256 token. Pass --model-revision, or pass "
        "--config with the resolved run config, to check it"
    )


def resolve_tool_python(value: str | None) -> str:
    """Return the interpreter that runs LigandMPNN."""
    if value is None:
        return sys.executable
    resolved = shutil.which(value)
    if resolved is None:
        raise AdapterError(f"interpreter not found: {value}")
    return resolved


def run_tool(argv: list[str], *, label: str = RUNNER_NAME) -> None:
    """Run one argument list with shell=False and fail on a nonzero exit."""
    print(f"ligandmpnn adapter: run {shlex.join(argv)}", flush=True)
    completed = subprocess.run(argv, shell=False, check=False)
    if completed.returncode != 0:
        raise AdapterError(f"{label} exited {completed.returncode}")


def tool_seed(requested_seed: int) -> int:
    """Return the seed LigandMPNN receives for a requested seed."""
    return SEED_ZERO_REPLACEMENT if requested_seed == 0 else requested_seed


def parse_fasta_records(path: Path) -> list[tuple[str, str]]:
    """Return the header and sequence of every record in a LigandMPNN FASTA.

    The file carries no trailing newline, so the last record closes at the end
    of the text rather than at a line break.
    """
    records: list[tuple[str, str]] = []
    header: str | None = None
    lines: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(lines)))
            header = line[1:].strip()
            lines = []
            continue
        if header is None:
            raise AdapterError(f"FASTA sequence precedes its header: {path}")
        lines.append("".join(line.split()).upper())
    if header is not None:
        records.append((header, "".join(lines)))
    return records


def runner_sequence_outputs(work_dir: Path, pattern: str | None = None) -> list[Path]:
    """Enumerate FASTA files returned by the runner.

    LigandMPNN writes one FASTA at `seqs/<stem>.fa` under the output folder, and
    that one file holds every record. A caller can name the path or a glob
    explicitly. The default accepts the common FASTA suffixes anywhere below the
    owned work directory.
    """
    if pattern is not None:
        raw = Path(pattern)
        if raw.is_absolute() or ".." in raw.parts:
            raise AdapterError(f"runner FASTA glob must stay under {work_dir}: {pattern}")
        candidates = sorted(path for path in work_dir.glob(pattern) if path.is_file())
    else:
        candidates = sorted(
            path
            for path in work_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".fa", ".fasta"}
        )
    if not candidates:
        raise AdapterError(f"{RUNNER_NAME} returned no FASTA files under {work_dir}")
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates[:8])
        suffix = "..." if len(candidates) > 8 else ""
        raise AdapterError(
            f"{RUNNER_NAME} returned {len(candidates)} FASTA files under {work_dir}; "
            f"select one with --sequences-glob. Found: {rendered}{suffix}"
        )
    return candidates


def header_scores(header: str) -> dict[str, float]:
    """Return the numeric fields a LigandMPNN design header records.

    A field whose value is not a number is absent from the result. A run that
    designed nothing reports `seq_rec=nan`, and that field lands nowhere.
    """
    return {key: float(value) for key, value in HEADER_FLOAT_RE.findall(header)}


def native_residue_count(header: str) -> int | None:
    """Return the `num_res` a native LigandMPNN header records, or None."""
    match = NATIVE_RESIDUE_COUNT_RE.search(header)
    if match is None:
        return None
    return int(match.group(1))


def check_designed_residues(header: str, *, chain: str, path: Path) -> int:
    """Refuse a run that designed no residues, and return the residue count.

    LigandMPNN exits 0 for a chain it cannot design. It writes the FASTA, one
    backbone PDB per requested design, and a native header that reads
    `num_res=0`, and every design record repeats the native sequence. Counting
    files and reading the exit code accepts that run, so this reads the count.
    """
    count = native_residue_count(header)
    if count is None:
        raise AdapterError(
            f"the native record in {path} records no num_res, so this wrapper cannot tell "
            f"whether LigandMPNN designed chain {chain}. Header: {header}"
        )
    if count <= 0:
        raise AdapterError(
            f"LigandMPNN parsed {count} residues for chain {chain} and designed nothing, while "
            "exiting 0 and writing a full output tree. Every returned sequence is the native "
            "sequence. Name a protein chain of the input structure in --design-chain. "
            f"Header: {header}"
        )
    return count


def chain_segment(sequence: str, requested: list[str], chain: str) -> str:
    """Return the residue string of one requested chain.

    LigandMPNN joins the chains of a record with --fasta_seq_separation, which
    defaults to a colon. The header carries no designed-chain list, so the
    segment order is the order this wrapper passed to --chains_to_design.
    """
    if chain not in requested:
        raise AdapterError(
            f"chain {chain} is absent from the requested chains: {', '.join(requested)}"
        )
    segments = sequence.split(FASTA_CHAIN_SEPARATOR)
    if len(segments) != len(requested):
        raise AdapterError(
            f"LigandMPNN returned {len(segments)} chain segments and this wrapper requested "
            f"{len(requested)}: {', '.join(requested)}. The record does not state which segment "
            f"carries chain {chain}"
        )
    return segments[requested.index(chain)]


def check_sequence(
    sequence: str, *, candidate_id: str, minimum: int | None, maximum: int | None
) -> None:
    """Reject a sequence the lane runner would reject later."""
    if not sequence:
        raise AdapterError(f"{candidate_id} has an empty sequence")
    if CANONICAL_AMINO_ACID_RE.fullmatch(sequence) is None:
        raise AdapterError(f"{candidate_id} carries a non-canonical amino acid")
    if minimum is not None and len(sequence) < minimum:
        raise AdapterError(f"{candidate_id} is {len(sequence)} residues, below the {minimum} minimum")
    if maximum is not None and len(sequence) > maximum:
        raise AdapterError(f"{candidate_id} is {len(sequence)} residues, above the {maximum} maximum")


def write_sequence(path: Path, candidate_id: str, sequence: str) -> None:
    """Write one single-record FASTA whose header is the candidate ID."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">{candidate_id}\n{sequence}\n")


def write_design_pose(
    path: Path,
    *,
    candidate_id: str,
    source_pose: Path,
    source_sha256: str,
    chain: str | None,
) -> None:
    """Write the design pose this candidate owns.

    The pose carries the upstream atom records unchanged, and the REMARK lines
    name the candidate and the upstream file, which keeps the bytes of every
    candidate pose distinct.

    LigandMPNN also writes one PDB per design under `backbones/` in its own
    output folder, and this wrapper does not use those files. A real run against
    1BC8 on 2026-09-10 wrote all three chains, both ZN HETATM records, and the
    designed sequence threaded onto the coordinates, so the objection is not
    that the file is incomplete. The redesigned chain comes back with N, CA, C
    and O and nothing else, four atoms per residue.

    Every design pose this package has scored carries 7.62 to 8.01 heavy atoms
    per residue, and `references/open-questions.md` records that no backbone-only
    design pose exists anywhere in it. Two settings decide how `sc_DockQ` may
    read such a pose, the `fnat` cutoff for a reference carrying no side chains
    and the smallest native contact count a comparable score may rest on, and
    both are the scientist's to set. Taking the `backbones/` file would make
    this the first arm to need them. The pose comes from the source file
    instead, which is what the ProteinMPNN designer does and what the scorers
    already read.

    `chain` names the chain LigandMPNN redesigned, and the only thing this
    function does with it is refuse a source pose that does not carry it.
    """
    if source_pose.name.endswith(".cif.gz"):
        source_text = cif_to_pdb(gzip.decompress(source_pose.read_bytes()).decode("utf-8"))
    elif source_pose.suffix == ".cif":
        source_text = cif_to_pdb(source_pose.read_text())
    else:
        source_text = source_pose.read_text(errors="replace")
    atoms = [line for line in source_text.splitlines() if line.startswith(ATOM_RECORD_PREFIXES)]
    if chain is not None and not any(line[21:22] == chain for line in atoms):
        raise AdapterError(
            f"{candidate_id} has no atom records for chain {chain} in {source_pose}"
        )
    if not atoms:
        raise AdapterError(f"{candidate_id} has no atom records in {source_pose}")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"REMARK 900 DESIGN POSE {candidate_id}",
        f"REMARK 900 SOURCE POSE {source_pose}",
        f"REMARK 900 SOURCE SHA256 {source_sha256}",
    ]
    path.write_text("\n".join([*header, *atoms, "END"]) + "\n")


def completed_receipt_rows(
    receipts_dir: Path, stage_id: str, artifact_id: str
) -> list[dict[str, Any]]:
    """Return the manifest rows one completed upstream receipt recorded."""
    receipt_path = receipts_dir / f"{stage_id}.json"
    if not receipt_path.is_file():
        raise AdapterError(f"upstream receipt not found: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict) or receipt.get("ok") is not True:
        raise AdapterError(f"upstream receipt did not complete: {receipt_path}")
    artifacts = receipt.get("output_manifest", {}).get("artifacts", [])
    phases = {str(artifact.get("phase")) for artifact in artifacts}
    selected_phase = "scale" if "scale" in phases else "single"
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.get("phase") != selected_phase or artifact.get("artifact_id") != artifact_id:
            continue
        for file_record in artifact.get("files", []):
            rows.extend(load_jsonl(Path(str(file_record["path"]))))
    if not rows:
        raise AdapterError(
            f"upstream receipt {receipt_path} carries no {artifact_id} rows for phase {selected_phase}"
        )
    return rows


def backbone_pose(row: dict[str, Any]) -> tuple[Path, str]:
    """Return the upstream design pose of one backbone row and check its hash."""
    candidate_id = str(row.get("candidate_id", ""))
    pose_value = row.get("design_pose_path")
    if not isinstance(pose_value, str) or not pose_value:
        raise AdapterError(f"backbone {candidate_id} records no design_pose_path")
    pose_path = Path(pose_value)
    if not pose_path.is_file():
        raise AdapterError(f"backbone {candidate_id} design pose is missing: {pose_path}")
    observed = sha256_file(pose_path)
    recorded = row.get("design_pose_sha256")
    if isinstance(recorded, str) and recorded and recorded != observed:
        raise AdapterError(
            f"backbone {candidate_id} design pose changed since the upstream stage: {pose_path}"
        )
    return pose_path, observed


def load_backbones(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return the backbone rows this phase designs sequences for."""
    if args.backbone_manifest is not None:
        manifest = args.backbone_manifest.resolve()
        artifact_root = args.artifact_root.resolve()
        if artifact_root not in manifest.parents:
            raise AdapterError(f"backbone manifest is outside the artifact root: {manifest}")
        if not manifest.is_file():
            raise AdapterError(f"backbone manifest not found: {manifest}")
        rows = load_jsonl(manifest)
    else:
        rows = completed_receipt_rows(
            args.receipts_dir, args.backbone_stage_id, args.backbone_artifact_id
        )
    rows.sort(key=lambda row: str(row.get("candidate_id", "")))
    if len(rows) < args.count:
        raise AdapterError(
            f"phase {args.phase} needs {args.count} backbones and the upstream manifest has {len(rows)}"
        )
    selected = rows[: args.count]
    # Validate every declared upstream input before the first design call.
    # Rechecking in design_backbone protects against a file changing mid-phase.
    # The catalog states the atoms a tool reads, and it states nothing for a
    # tool it has not recorded, so an unrecorded tool gets no atom check.
    required_atoms = backbone_shape.required_atoms_for_tool(getattr(args, "designer_id", ""))
    for row in selected:
        missing = [field for field in DIVERSITY_LINEAGE_FIELDS if not row.get(field)]
        if missing:
            raise AdapterError(
                f"backbone {row.get('candidate_id')} is missing diversity lineage fields: "
                + ", ".join(missing)
            )
        pose_path, _ = backbone_pose(row)
        # The upstream tool may declare an unknown atom set, so this reads the
        # file the generator wrote rather than a claim about the generator.
        if required_atoms:
            try:
                problem = backbone_shape.pose_shape_problem(
                    pose_path,
                    required_atoms=required_atoms,
                    consumer=args.designer_id,
                    producer=str(row.get("origin_generator") or "") or None,
                )
            except backbone_shape.BackboneShapeError as exc:
                raise AdapterError(
                    f"backbone {row.get('candidate_id')} design pose could not be read for its "
                    f"backbone atoms: {exc}"
                ) from exc
            if problem is not None:
                raise AdapterError(f"backbone {row.get('candidate_id')} {problem}")
    return selected


def design_backbone(
    args: argparse.Namespace,
    *,
    row: dict[str, Any],
    index: int,
    runner: Path,
    checkpoint: Path,
    tool_python: str,
    phase_dir: Path,
    verification: dict[str, str],
) -> list[dict[str, Any]]:
    """Run LigandMPNN for one backbone and return its candidate rows."""
    parent_id = str(row.get("candidate_id", ""))
    if not parent_id:
        raise AdapterError("a backbone row carries no candidate_id")
    source_pose, source_sha256 = backbone_pose(row)
    requested_seed = args.seed + index
    if requested_seed < 0:
        raise AdapterError(f"seed {requested_seed} is negative")
    effective_seed = tool_seed(requested_seed)
    requested_chains = [args.design_chain]
    work_dir = phase_dir / args.work_subdir / parent_id
    work_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        tool_python,
        str(runner),
        "--model_type",
        args.model_type,
        checkpoint_flag(args.model_type),
        str(checkpoint),
        "--pdb_path",
        str(source_pose),
        "--out_folder",
        str(work_dir),
        "--chains_to_design",
        args.design_chain,
        "--temperature",
        str(args.sampling_temp),
        "--seed",
        str(effective_seed),
        "--batch_size",
        "1",
        "--number_of_batches",
        str(args.sequences_per_backbone),
    ]
    run_tool(argv)
    produced = runner_sequence_outputs(work_dir, args.sequences_glob)
    records = parse_fasta_records(produced[0])
    if len(records) < args.sequences_per_backbone + 1:
        raise AdapterError(
            f"{produced[0]} holds {len(records)} records; expected the native record and "
            f"{args.sequences_per_backbone} designed records"
        )
    parsed_residues = check_designed_residues(
        records[0][0], chain=args.design_chain, path=produced[0]
    )
    rows: list[dict[str, Any]] = []
    for variant, (header, raw_sequence) in enumerate(records[1 : args.sequences_per_backbone + 1]):
        candidate_id = f"{parent_id}-{args.designer_id}-{variant:02d}"
        sequence = chain_segment(raw_sequence, requested_chains, args.design_chain)
        check_sequence(
            sequence,
            candidate_id=candidate_id,
            minimum=args.minimum_length,
            maximum=args.maximum_length,
        )
        sequence_path = phase_dir / args.sequence_subdir / f"{candidate_id}.fasta"
        pose_path = phase_dir / args.pose_subdir / f"{candidate_id}.pdb"
        write_sequence(sequence_path, candidate_id, sequence)
        write_design_pose(
            pose_path,
            candidate_id=candidate_id,
            source_pose=source_pose,
            source_sha256=source_sha256,
            chain=args.design_chain,
        )
        scores = header_scores(header)
        candidate = {
            **{field: row[field] for field in PARENT_LINEAGE_FIELDS if field in row},
            "candidate_id": candidate_id,
            "parent_candidate_id": parent_id,
            "sequence_designer": args.designer_id,
            "seq_method": args.designer_id,
            "sequence_path": str(sequence_path.resolve()),
            "sequence_sha256": canonical_sequence_sha256(sequence),
            "sequence_length": len(sequence),
            "design_pose_path": str(pose_path.resolve()),
            "design_pose_sha256": sha256_file(pose_path),
            "design_chain": args.design_chain,
            "variant_index": variant,
            "requested_seed": requested_seed,
            "tool_seed": effective_seed,
            "sampling_temperature": args.sampling_temp,
            "model_type": args.model_type,
            "model_name": checkpoint.stem,
            "ligandmpnn_num_res": parsed_residues,
            "checkpoint_sha256": verification["observed"],
            "model_revision_verified": verification["state"],
            "source_design_pose_sha256": source_sha256,
            "status": "sequence-designed",
        }
        for key, field in HEADER_SCORE_FIELDS:
            if key in scores:
                candidate[field] = scores[key]
        rows.append(candidate)
    return rows


def run(args: argparse.Namespace) -> int:
    """Design sequences for one phase and write the stage outputs."""
    root = resolve_root(args.ligandmpnn_root)
    runner = resolve_runner(root)
    model_revision = resolve_model_revision(args)
    _, checkpoint = resolve_checkpoint(root, args.weights_dir, args.model_name, args.model_type)
    verification = verify_checkpoint_digest(checkpoint, model_revision)
    report_checkpoint_digest(checkpoint, verification)
    tool_python = resolve_tool_python(args.tool_python)
    attempt_dir = args.attempt_dir.resolve()
    phase_dir = attempt_dir / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.manifest_path.resolve()
        if args.manifest_path is not None
        else phase_dir / DEFAULT_MANIFEST_NAME
    )
    if attempt_dir not in manifest_path.parents:
        raise AdapterError(f"manifest path escapes the attempt directory: {manifest_path}")
    backbones = load_backbones(args)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(backbones):
        rows.extend(
            design_backbone(
                args,
                row=row,
                index=index,
                runner=runner,
                checkpoint=checkpoint,
                tool_python=tool_python,
                phase_dir=phase_dir,
                verification=verification,
            )
        )
    expected = args.count * args.sequences_per_backbone
    if len(rows) != expected:
        raise AdapterError(
            f"phase {args.phase} produced {len(rows)} candidates; expected {expected}"
        )
    write_jsonl(manifest_path, rows)
    print(
        f"ligandmpnn adapter: phase={args.phase} backbones={len(backbones)} "
        f"candidates={len(rows)} model_type={args.model_type} manifest={manifest_path}"
    )
    return 0


def toolcheck(args: argparse.Namespace) -> int:
    """Report the LigandMPNN files and interpreter this adapter uses."""
    root = resolve_root(args.ligandmpnn_root)
    runner = resolve_runner(root)
    _, checkpoint = resolve_checkpoint(root, args.weights_dir, args.model_name, args.model_type)
    verification = verify_checkpoint_digest(checkpoint, resolve_model_revision(args))
    tool_python = resolve_tool_python(args.tool_python)
    completed = subprocess.run(
        [tool_python, str(runner), "--help"],
        shell=False,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise AdapterError(
            f"{RUNNER_NAME} --help exited {completed.returncode}: {completed.stderr.strip()[:400]}"
        )
    print(f"ligandmpnn adapter: runner {runner}")
    print(
        f"ligandmpnn adapter: model_type {args.model_type} loads its weights through "
        f"{checkpoint_flag(args.model_type)}"
    )
    report_checkpoint_digest(checkpoint, verification)
    print(f"ligandmpnn adapter: interpreter {tool_python}")
    return 0


def add_tool_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ligandmpnn-root",
        type=Path,
        default=None,
        help=f"LigandMPNN checkout directory. Defaults to {ROOT_ENVIRONMENT_KEY}.",
    )
    parser.add_argument(
        "--model-type",
        choices=MODEL_TYPES,
        default=DEFAULT_MODEL_TYPE,
        help=(
            "LigandMPNN model this adapter runs. The checkpoint reaches the runner through the "
            f"flag that matches it. Defaults to {DEFAULT_MODEL_TYPE}."
        ),
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help=(
            f"Directory holding the checkpoint files. Defaults to {DEFAULT_WEIGHTS_SUBDIR} inside "
            "the checkout."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Checkpoint file name without the .pt suffix. Defaults to "
            f"{DEFAULT_CHECKPOINT_NAMES[DEFAULT_MODEL_TYPE]} for {DEFAULT_MODEL_TYPE} and to "
            f"{DEFAULT_CHECKPOINT_NAMES['protein_mpnn']} for protein_mpnn. Required for every "
            "other model type."
        ),
    )
    parser.add_argument(
        "--tool-python",
        default=None,
        help="Interpreter that runs LigandMPNN. Defaults to the interpreter running this wrapper.",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help=(
            "The model_revision string the profile records for this adapter. The wrapper reads "
            "its sha256 token and refuses when the checkpoint on disk hashes to something else. "
            "Without this and without --config, the run records the checkpoint as unverified."
        ),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("toolcheck", help="Probe the runtime without designing.")
    add_tool_arguments(check_parser)
    check_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Resolved runtime config to read this designer's model_revision from.",
    )
    check_parser.add_argument(
        "--designer-id",
        default=DEFAULT_DESIGNER_ID,
        help="Sequence designer ID. Match the designer ID the campaign registers.",
    )
    check_parser.add_argument(
        "--adapter-id",
        default=None,
        help=(
            "Adapter ID of this designer registration. Names one arm when a campaign "
            "registers the same designer ID on more than one."
        ),
    )
    run_parser = subparsers.add_parser("run", help="Design sequences for one phase.")
    add_tool_arguments(run_parser)
    run_parser.add_argument(
        "--phase", required=True, help="Stage phase name, such as smoke or scale."
    )
    run_parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of backbones this phase designs sequences for.",
    )
    run_parser.add_argument(
        "--attempt-dir", type=Path, required=True, help="Attempt directory that owns the outputs."
    )
    run_parser.add_argument(
        "--receipts-dir",
        type=Path,
        required=True,
        help="Directory holding the completed stage receipts.",
    )
    run_parser.add_argument("--artifact-root", type=Path, required=True, help="Run artifact root.")
    run_parser.add_argument(
        "--backbone-stage-id",
        default="generate-arm-1",
        help="Stage ID of the upstream backbone generator.",
    )
    run_parser.add_argument(
        "--backbone-artifact-id",
        default="arm-1-candidates",
        help="Artifact ID of the upstream backbone manifest.",
    )
    run_parser.add_argument(
        "--backbone-manifest",
        type=Path,
        default=None,
        help="Published backbone manifest under the artifact root. Overrides the receipt lookup.",
    )
    run_parser.add_argument(
        "--designer-id",
        default=DEFAULT_DESIGNER_ID,
        help="Sequence designer ID. Match the designer ID the campaign registers.",
    )
    run_parser.add_argument(
        "--adapter-id",
        default=None,
        help=(
            "Adapter ID of this designer registration. Names one arm when a campaign "
            "registers the same designer ID on more than one."
        ),
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Resolved run config. The wrapper reads the model_revision this designer's adapter "
            "records and checks the checkpoint against it. --model-revision overrides this."
        ),
    )
    run_parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Resolved run plan supplied by the dispatcher.",
    )
    run_parser.add_argument(
        "--sequences-per-backbone",
        type=int,
        default=1,
        help=(
            "Sequences to design for every backbone. The wrapper sends it as "
            "--number_of_batches with --batch_size 1. Match the stage records_per_count."
        ),
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help=(
            "Base seed. Backbone N receives the base seed plus N. A seed of zero reaches the "
            f"runner as {SEED_ZERO_REPLACEMENT}, because zero asks LigandMPNN for a random seed."
        ),
    )
    run_parser.add_argument(
        "--sampling-temp",
        type=float,
        default=0.1,
        help="Sampling temperature. The wrapper sends it as LigandMPNN's --temperature.",
    )
    run_parser.add_argument(
        "--design-chain",
        required=True,
        help=(
            "Chain to design and to keep in the design pose. The wrapper sends it as "
            "--chains_to_design. run.py does not require that flag and this wrapper does, so "
            "every run names the chain it designed."
        ),
    )
    run_parser.add_argument(
        "--minimum-length", type=int, default=None, help="Reject a sequence below this length."
    )
    run_parser.add_argument(
        "--maximum-length", type=int, default=None, help="Reject a sequence above this length."
    )
    run_parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=f"Manifest path. Defaults to {DEFAULT_MANIFEST_NAME} in the phase directory.",
    )
    run_parser.add_argument(
        "--sequence-subdir",
        default=DEFAULT_SEQUENCE_SUBDIR,
        help=f"FASTA directory inside the phase directory. Defaults to {DEFAULT_SEQUENCE_SUBDIR}.",
    )
    run_parser.add_argument(
        "--pose-subdir",
        default=DEFAULT_POSE_SUBDIR,
        help=f"Design pose directory inside the phase directory. Defaults to {DEFAULT_POSE_SUBDIR}.",
    )
    run_parser.add_argument(
        "--work-subdir",
        default=DEFAULT_WORK_SUBDIR,
        help=(
            "Raw LigandMPNN output directory inside the phase directory. Defaults to "
            f"{DEFAULT_WORK_SUBDIR}."
        ),
    )
    run_parser.add_argument(
        "--sequences-glob",
        default=None,
        help=(
            "Optional FASTA glob relative to each runner work directory. Defaults to "
            "discovering one .fa or .fasta file anywhere under that directory."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        return toolcheck(args) if args.command == "toolcheck" else run(args)
    except AdapterError as exc:
        print(f"ligandmpnn adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
