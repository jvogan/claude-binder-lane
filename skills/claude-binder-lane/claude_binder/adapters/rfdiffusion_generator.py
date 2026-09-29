#!/usr/bin/env python3
"""Generate RFdiffusion backbones for one binder lane stage phase.

This wrapper fills the `rfdiffusion-generator` slot. It reads the target manifest
the `target-preparer` stage published, runs `scripts/run_inference.py` once for
the whole phase, and writes receipt-owned outputs into the current attempt
directory:

  <attempt>/<phase>/poses/<candidate_id>.pdb        one design pose per candidate
  <attempt>/<phase>/candidate-manifest.jsonl        one row per candidate
  <attempt>/<phase>/rfdiffusion/                    the untouched tool output

RFdiffusion returns a backbone and no sequence. Every designed residue carries a
placeholder identity, so the manifest sets `sequence_path`, `sequence_sha256`,
`sequence_length`, and `sequence_designer` to null, records
`backbone_only: true`, and leaves the sequence to the downstream
`proteinmpnn-designer` or `solublempnn-designer` slot. That slot reads
`candidate_id` and `design_pose_path` from these rows, so the design pose keeps
both the fixed chain and the designed chain and the sequence designer picks the
chain it designs.

The contig specification has to name residues the input structure actually
carries. A mismatch is the common RFdiffusion failure, so the wrapper parses the
contigs, reads the residue numbers out of the input structure, and refuses with a
message that names the contig segment and the residues on the other side. It
applies the same check to every hotspot residue.

Argument shape comes from two recorded callers that do not ship with this
package, so the argument list below is the record of what they built rather
than something you can re-read. Both build
`inference.input_pdb`, `inference.output_prefix`, `inference.num_designs`,
`inference.ckpt_override_path`, `contigmap.contigs`, `ppi.hotspot_res`, and the
two `denoiser.noise_scale_*` keys. The recorded PD-L1 run left the seed at 0,
which that entrypoint reads as a request to pass no seed at all, so the wrapper
substitutes a fixed nonzero value for a requested seed of 0, always passes the
seed key, and records both numbers.

Every command is an argument list that runs with shell=False. The wrapper builds
no shell string.

Install RFdiffusion from https://github.com/RosettaCommons/RFdiffusion and point
the wrapper at the checkout with --rfdiffusion-root or RFDIFFUSION_ROOT.
"""

from __future__ import annotations

import argparse
import glob
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

RUNNER_RELATIVE_PARTS = ("scripts", "run_inference.py")
RUNNER_NAME = "scripts/run_inference.py"
DEFAULT_WEIGHTS_SUBDIR = "models"
DEFAULT_CHECKPOINT_NAME = "Complex_base_ckpt.pt"
DEFAULT_MANIFEST_NAME = "candidate-manifest.jsonl"
DEFAULT_POSE_SUBDIR = "poses"
DEFAULT_WORK_SUBDIR = "rfdiffusion"
DEFAULT_OUTPUT_STEM = "design"
DEFAULT_GENERATOR_ID = "rfdiffusion"
DEFAULT_ADAPTER_ID = "rfdiffusion-generator"
DEFAULT_ARTIFACT_ID = "backbone-candidate-manifest"
DEFAULT_TARGET_STAGE_ID = "target-prepare"
DEFAULT_TARGET_ARTIFACT_ID = "target-manifest"
RUNNER_PROTOCOLS = ("auto", "local", "modal")
DEFAULT_RUNNER_PROTOCOL = "auto"
# These paths are the mounts in the shipped Modal RFdiffusion environment.
MODAL_RFDIFFUSION_ROOT = Path("/opt/rfd")
MODAL_WEIGHTS_DIR = Path("/weights")
# The Hydra key the PD-L1 entrypoint assembles at line 503. That recorded run set
# the seed to 0 and skipped the key, so no recorded run has exercised
# it. A checkout whose config names the seed differently takes --seed-key.
DEFAULT_SEED_KEY = "inference.seed"
# A requested seed of 0 means "pass no seed" to the PD-L1 entrypoint, and a run
# with no seed is not reproducible. The wrapper substitutes this value and
# records both the requested seed and the seed the tool received.
SEED_ZERO_REPLACEMENT = 1000003
ATOM_RECORD_PREFIXES = ("ATOM  ", "HETATM")
READ_BLOCK_BYTES = 1024 * 1024


def backbone_lineage(candidate_id: str, structure_method: str) -> dict[str, str]:
    """Return the diversity lineage recorded on a generated backbone."""
    return {
        "root_backbone_id": candidate_id,
        "tm90_cluster_id": candidate_id,
        "structure_method": structure_method,
        "seq_method": "none",
        "fold_class": "unknown",
    }
GENERATOR_MODE = "backbone-only"
CANDIDATE_STATUS = "generated"
# The fields the target manifest has to carry before this stage can run. They are
# the three the target-prepare stage contract declares plus the two structure
# paths this wrapper reads.
REQUIRED_TARGET_MANIFEST_FIELDS = (
    "target_id",
    "target_sha256",
    "residue_map_sha256",
    "source_structure_path",
    "normalized_structure_path",
)
# A chain-anchored contig segment is CHAIN, a residue number, and an optional
# second number, for example A19-127 or A19. A de novo segment is a length or a
# length range, for example 60-90. A lone 0 closes a chain.
CONTIG_CHAIN_SPAN_RE = re.compile(r"^([A-Za-z])(-?\d+)(?:-(-?\d+))?$")
CONTIG_LENGTH_SPAN_RE = re.compile(r"^(\d+)(?:-(\d+))?$")
CONTIG_CHAIN_BREAK = "0"
# A hotspot arrives as CHAIN:NUMBER, CHAINNUMBER, or a lane residue range.
HOTSPOT_RE = re.compile(r"^([A-Za-z]):?(-?\d+)(?:-(-?\d+))?$")
CHAIN_ID_RE = re.compile(r"^[A-Za-z0-9]$")
# A Hydra override is a dotted key and a value. The leading + appends a key the
# config does not declare.
OVERRIDE_RE = re.compile(r"^\+?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*=.+$")
# A profile pins the checkpoint inside its free-text model_revision as
# `Complex_base_ckpt.pt sha256:<64 hex>`. The digest is the part this wrapper
# can hold against the file it is about to load.
MODEL_REVISION_DIGEST_RE = re.compile(r"sha256:(\S*)")
SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")
# How many residue numbers a refusal names before it switches to a count.
REPORTED_NUMBER_LIMIT = 8


class AdapterError(RuntimeError):
    """A condition the operator has to fix before the stage can run."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of the complete file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows to a path in one atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one JSON document to a path in an atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def resolve_output_path(attempt_dir: Path, path: Path, label: str) -> Path:
    """Return an output path and refuse one that leaves the attempt directory."""
    resolved = Path(os.path.normpath(path if path.is_absolute() else attempt_dir / path))
    if attempt_dir not in resolved.parents:
        raise AdapterError(f"{label} escapes the attempt directory: {path}")
    return resolved


def resolve_input_path(artifact_root: Path, value: str, label: str) -> Path:
    """Return an input path and refuse one that leaves the artifact root."""
    raw = Path(value).expanduser()
    resolved = Path(os.path.normpath(raw if raw.is_absolute() else artifact_root / raw))
    if artifact_root not in resolved.parents:
        raise AdapterError(f"{label} is outside the artifact root: {value}")
    return resolved


def resolve_root(value: Path | None) -> Path:
    """Return the RFdiffusion checkout directory."""
    if value is None:
        environment_value = os.environ.get("RFDIFFUSION_ROOT", "").strip()
        if not environment_value:
            raise AdapterError(
                "RFdiffusion is not located. Pass --rfdiffusion-root, or set RFDIFFUSION_ROOT "
                "to a checkout of https://github.com/RosettaCommons/RFdiffusion"
            )
        value = Path(environment_value)
    root = value.expanduser()
    if not root.is_dir():
        raise AdapterError(f"RFdiffusion root is not a directory: {root}")
    return root.resolve()


def resolve_execution(args: argparse.Namespace) -> tuple[str, Path, Path]:
    """Resolve the execution route and its checkout and checkpoint directories.

    ``local`` keeps the original checkout contract. ``modal`` uses the mounts
    supplied by the shipped Modal environment and accepts explicit paths in
    tests or an operator-provided image. ``auto`` keeps a local checkout when
    one is explicitly supplied and otherwise selects the Modal mounts when
    they exist.
    """
    protocol = getattr(args, "runner_protocol", DEFAULT_RUNNER_PROTOCOL)
    root_value = getattr(args, "rfdiffusion_root", None)
    weights_value = getattr(args, "weights_dir", None)
    environment_root = os.environ.get("RFDIFFUSION_ROOT", "").strip()

    if protocol == "local":
        root = resolve_root(root_value)
        weights_dir = weights_value.expanduser() if weights_value is not None else root / DEFAULT_WEIGHTS_SUBDIR
        return "local", root, weights_dir

    if protocol == "modal":
        root = (root_value or MODAL_RFDIFFUSION_ROOT).expanduser()
        weights_dir = (weights_value or MODAL_WEIGHTS_DIR).expanduser()
        return "modal", root.resolve(), weights_dir.resolve()

    if protocol != "auto":
        raise AdapterError(
            f"runner protocol {protocol!r} is not supported; choose one of {', '.join(RUNNER_PROTOCOLS)}"
        )

    if root_value is not None or environment_root:
        root = resolve_root(root_value)
        weights_dir = weights_value.expanduser() if weights_value is not None else root / DEFAULT_WEIGHTS_SUBDIR
        return "local", root, weights_dir

    modal_root = MODAL_RFDIFFUSION_ROOT
    modal_weights = weights_value or MODAL_WEIGHTS_DIR
    if modal_root.is_dir() and modal_weights.is_dir():
        return "modal", modal_root.resolve(), modal_weights.expanduser().resolve()

    raise AdapterError(
        "automatic route found neither a local RFdiffusion checkout nor the shipped Modal "
        f"mounts {modal_root} and {modal_weights}. Pass --runner-protocol local with "
        "--rfdiffusion-root or set RFDIFFUSION_ROOT, or pass --runner-protocol modal after "
        "the Modal environment is mounted"
    )


def resolve_runner(root: Path) -> Path:
    """Return the RFdiffusion inference script inside a checkout."""
    runner = root.joinpath(*RUNNER_RELATIVE_PARTS)
    if not runner.is_file():
        raise AdapterError(
            f"RFdiffusion runner not found: {runner}. Point --rfdiffusion-root at a checkout "
            f"that contains {RUNNER_NAME}"
        )
    return runner


def resolve_checkpoint(root: Path, override: Path | None, checkpoint_name: str) -> Path:
    """Return the checkpoint file this run loads."""
    weights_dir = override.expanduser() if override is not None else root / DEFAULT_WEIGHTS_SUBDIR
    if not weights_dir.is_dir():
        raise AdapterError(f"RFdiffusion weights directory not found: {weights_dir}")
    checkpoint = weights_dir / checkpoint_name
    if not checkpoint.is_file():
        raise AdapterError(f"RFdiffusion checkpoint not found: {checkpoint}")
    return checkpoint.resolve()


def recorded_checkpoint_digest(model_revision: str | None) -> str | None:
    """Return the sha256 a model revision pins, or None when it pins none."""
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
    """Hash the checkpoint before the generator subprocess can load it."""
    observed = sha256_file(checkpoint)
    recorded = recorded_checkpoint_digest(model_revision)
    if recorded is None:
        return {"state": "unrecorded", "observed": observed, "recorded": ""}
    if recorded != observed:
        raise AdapterError(
            f"checkpoint {checkpoint} hashes to {observed}, and model_revision pins {recorded}. "
            "The file on disk is not the checkpoint this run would attribute its backbones to. "
            "Point --weights-dir at the pinned checkpoint, or record the digest of the file you "
            "mean to run"
        )
    return {"state": "matched", "observed": observed, "recorded": recorded}


def config_model_revision(config_path: Path, adapter_id: str) -> str:
    """Return the configured model revision for this generator adapter."""
    config = json.loads(config_path.read_text())
    matches = [
        item
        for item in config.get("adapters", [])
        if isinstance(item, dict) and item.get("adapter_id") == adapter_id
    ]
    if len(matches) != 1:
        raise AdapterError(
            f"{config_path} registers {len(matches)} adapters with id {adapter_id}. "
            "Pass --model-revision"
        )
    revision = matches[0].get("model_revision")
    if not isinstance(revision, str) or not revision:
        raise AdapterError(f"adapter {adapter_id} records no model_revision")
    return revision


def resolve_model_revision(args: argparse.Namespace) -> str | None:
    """Return the model revision string this invocation checks against."""
    if args.model_revision is not None:
        return args.model_revision
    config_path = getattr(args, "config", None)
    if config_path is None:
        return None
    resolved = config_path.expanduser()
    if not resolved.is_file():
        raise AdapterError(f"--config does not exist: {resolved}")
    return config_model_revision(resolved.resolve(), args.adapter_id)


def report_checkpoint_digest(checkpoint: Path, verification: dict[str, str]) -> None:
    """Report whether the checkpoint matched the configured digest."""
    print(f"rfdiffusion adapter: checkpoint {checkpoint}")
    if verification["state"] == "matched":
        print(f"rfdiffusion adapter: sha256 {verification['observed']} matches model_revision")
        return
    print(
        f"rfdiffusion adapter: sha256 {verification['observed']} is unverified, because no "
        "model_revision reached this wrapper with a sha256 token. Pass --model-revision, or pass "
        "--config with the resolved run config, to check it"
    )


def resolve_tool_python(value: str | None) -> str:
    """Return the interpreter that runs RFdiffusion."""
    if value is None:
        return sys.executable
    resolved = shutil.which(value)
    if resolved is None:
        raise AdapterError(f"interpreter not found: {value}")
    return resolved


def run_tool(argv: list[str]) -> None:
    """Run one argument list with shell=False and fail on a nonzero exit."""
    print(f"rfdiffusion adapter: run {shlex.join(argv)}", flush=True)
    completed = subprocess.run(argv, shell=False, check=False)
    if completed.returncode != 0:
        raise AdapterError(f"{RUNNER_NAME} exited {completed.returncode}")


def tool_seed(requested_seed: int) -> int:
    """Return the seed RFdiffusion receives for a requested seed."""
    return SEED_ZERO_REPLACEMENT if requested_seed == 0 else requested_seed


def format_numbers(numbers: list[int]) -> str:
    """Return a short residue-number list for a refusal message."""
    if len(numbers) <= REPORTED_NUMBER_LIMIT:
        return ", ".join(str(number) for number in numbers)
    head = ", ".join(str(number) for number in numbers[:REPORTED_NUMBER_LIMIT])
    return f"{head}, and {len(numbers) - REPORTED_NUMBER_LIMIT} more"


# ----------------------------------------------------------------------------
# Structure reading. The wrapper reads coordinate records with the standard
# library, the same way the PD-L1 entrypoint audits its input.
# ----------------------------------------------------------------------------


def read_atom_records(path: Path) -> list[str]:
    """Return every coordinate record of a PDB file."""
    if not path.is_file():
        raise AdapterError(f"structure not found: {path}")
    atoms = [
        line
        for line in path.read_text(errors="replace").splitlines()
        if line.startswith(ATOM_RECORD_PREFIXES)
    ]
    if not atoms:
        raise AdapterError(f"structure carries no coordinate records: {path}")
    return atoms


def chain_residues(atoms: list[str], *, standard_only: bool = False) -> dict[str, list[int]]:
    """Return the sorted residue numbers of every chain in a coordinate list.

    RFdiffusion reads ATOM records, so contig and hotspot checks pass
    `standard_only` and ignore the HETATM records a target may carry.
    """
    residues: dict[str, set[int]] = {}
    for line in atoms:
        if standard_only and not line.startswith("ATOM  "):
            continue
        chain = line[21:22].strip()
        raw_number = line[22:26].strip()
        if not chain or not raw_number:
            continue
        try:
            number = int(raw_number)
        except ValueError:
            continue
        residues.setdefault(chain, set()).add(number)
    return {chain: sorted(numbers) for chain, numbers in sorted(residues.items())}


# ----------------------------------------------------------------------------
# Contigs and hotspots.
# ----------------------------------------------------------------------------


def parse_contigs(contigs: str) -> tuple[list[tuple[str, str, int, int]], list[tuple[str, int, int]]]:
    """Return the chain-anchored spans and the de novo length spans of a contig string.

    RFdiffusion splits the specification on whitespace and then splits every
    token on a slash, which is the grammar both recorded callers use.
    """
    text = contigs.strip()
    if not text:
        raise AdapterError("--contigs is empty")
    if text.startswith("[") or text.endswith("]"):
        raise AdapterError(
            f"--contigs carries the Hydra brackets: {contigs}. Pass the specification alone, "
            "because the wrapper adds the brackets"
        )
    chain_spans: list[tuple[str, str, int, int]] = []
    length_spans: list[tuple[str, int, int]] = []
    for token in text.split():
        segments = token.split("/")
        for index, segment in enumerate(segments):
            if segment == CONTIG_CHAIN_BREAK:
                if index != len(segments) - 1:
                    raise AdapterError(
                        f"contig token {token} puts the chain break {CONTIG_CHAIN_BREAK} before "
                        "its last segment"
                    )
                continue
            chain_match = CONTIG_CHAIN_SPAN_RE.fullmatch(segment)
            if chain_match is not None:
                low = int(chain_match.group(2))
                high = int(chain_match.group(3)) if chain_match.group(3) is not None else low
                if high < low:
                    raise AdapterError(f"contig segment {segment} runs from {low} down to {high}")
                chain_spans.append((segment, chain_match.group(1), low, high))
                continue
            length_match = CONTIG_LENGTH_SPAN_RE.fullmatch(segment)
            if length_match is not None:
                low = int(length_match.group(1))
                high = int(length_match.group(2)) if length_match.group(2) is not None else low
                if high < low:
                    raise AdapterError(f"contig segment {segment} runs from {low} down to {high}")
                if low < 1:
                    raise AdapterError(f"contig segment {segment} asks for {low} residues")
                length_spans.append((segment, low, high))
                continue
            raise AdapterError(
                f"contig segment {segment} in token {token} is not a chain span, a length span, "
                f"or the chain break {CONTIG_CHAIN_BREAK}"
            )
    if not chain_spans and not length_spans:
        raise AdapterError(f"--contigs holds no segment: {contigs}")
    return chain_spans, length_spans


def check_contig_spans(
    chain_spans: list[tuple[str, str, int, int]],
    residues: dict[str, list[int]],
    structure_path: Path,
) -> None:
    """Refuse a contig segment the input structure cannot satisfy."""
    for segment, chain, low, high in chain_spans:
        present = residues.get(chain)
        if present is None:
            available = ", ".join(residues) if residues else "no chain"
            raise AdapterError(
                f"contig segment {segment} names chain {chain} and {structure_path} carries "
                f"{available}"
            )
        known = set(present)
        missing = [number for number in range(low, high + 1) if number not in known]
        if missing:
            raise AdapterError(
                f"contig segment {segment} needs chain {chain} residues {low} to {high} and "
                f"{structure_path} chain {chain} carries {present[0]} to {present[-1]}; "
                f"{len(missing)} are absent: {format_numbers(missing)}"
            )


def parse_hotspots(
    values: list[str],
    chain_spans: list[tuple[str, str, int, int]],
    residues: dict[str, list[int]],
    structure_path: Path,
) -> list[str]:
    """Return the hotspot list RFdiffusion reads, and refuse a residue it cannot see."""
    hotspots: list[str] = []
    expanded_values = [
        part.strip() for value in values for part in value.split(",") if part.strip()
    ]
    for value in expanded_values:
        match = HOTSPOT_RE.fullmatch(value.strip())
        if match is None:
            raise AdapterError(
                f"hotspot {value} is not CHAIN:NUMBER, CHAINNUMBER, or a residue range, "
                "for example A:122 or A122"
            )
        chain, low = match.group(1), int(match.group(2))
        high = int(match.group(3)) if match.group(3) is not None else low
        if high < low:
            raise AdapterError(f"hotspot {value} runs from {low} down to {high}")
        present = residues.get(chain)
        if present is None:
            available = ", ".join(residues) if residues else "no chain"
            raise AdapterError(
                f"hotspot {chain}{low} names chain {chain} and {structure_path} carries {available}"
            )
        present_set = set(present)
        for number in range(low, high + 1):
            normalized = f"{chain}{number}"
            if normalized in hotspots:
                raise AdapterError(f"hotspot {normalized} is named more than once")
            if number not in present_set:
                raise AdapterError(
                    f"hotspot {normalized} is absent from {structure_path}; chain {chain} carries "
                    f"{present[0]} to {present[-1]}"
                )
            covering = [
                segment
                for segment, span_chain, segment_low, segment_high in chain_spans
                if span_chain == chain and segment_low <= number <= segment_high
            ]
            if not covering:
                spans = ", ".join(segment for segment, _, _, _ in chain_spans) or "no chain segment"
                raise AdapterError(
                    f"hotspot {normalized} falls outside every chain segment of the contigs: {spans}"
                )
            hotspots.append(normalized)
    return hotspots


def check_overrides(values: list[str], owned_keys: set[str]) -> list[str]:
    """Return the extra Hydra overrides, and refuse one the wrapper already owns."""
    overrides: list[str] = []
    for value in values:
        if OVERRIDE_RE.fullmatch(value) is None:
            raise AdapterError(
                f"extra override {value} is not key=value, for example inference.deterministic=True"
            )
        key = value.split("=", 1)[0].lstrip("+")
        if key in owned_keys:
            raise AdapterError(f"extra override {value} repeats a key the wrapper sets: {key}")
        if key in {existing.split("=", 1)[0].lstrip("+") for existing in overrides}:
            raise AdapterError(f"extra override {value} repeats an earlier override key: {key}")
        overrides.append(value)
    return overrides


# ----------------------------------------------------------------------------
# The upstream target manifest.
# ----------------------------------------------------------------------------


def completed_receipt_file(receipts_dir: Path, stage_id: str, artifact_id: str) -> Path:
    """Return the one file a completed upstream receipt recorded for an artifact."""
    receipt_path = receipts_dir / f"{stage_id}.json"
    if not receipt_path.is_file():
        raise AdapterError(f"upstream receipt not found: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict) or receipt.get("ok") is not True:
        raise AdapterError(f"upstream receipt did not complete: {receipt_path}")
    artifacts = receipt.get("output_manifest", {}).get("artifacts", [])
    phases = {str(artifact.get("phase")) for artifact in artifacts}
    selected_phase = "scale" if "scale" in phases else "single"
    paths = [
        Path(str(file_record["path"]))
        for artifact in artifacts
        if artifact.get("phase") == selected_phase and artifact.get("artifact_id") == artifact_id
        for file_record in artifact.get("files", [])
    ]
    if len(paths) != 1:
        raise AdapterError(
            f"upstream receipt {receipt_path} carries {len(paths)} {artifact_id} files for phase "
            f"{selected_phase}; expected exactly one"
        )
    return paths[0]


def load_target_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    """Return the target manifest this stage designs against, and its path."""
    if args.target_manifest is not None:
        artifact_root = args.artifact_root.expanduser().resolve()
        path = resolve_input_path(artifact_root, str(args.target_manifest), "target manifest")
    else:
        path = completed_receipt_file(
            args.receipts_dir, args.target_stage_id, args.target_artifact_id
        )
    if not path.is_file():
        raise AdapterError(f"target manifest not found: {path}")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise AdapterError(f"target manifest is not a JSON object: {path}")
    for field in REQUIRED_TARGET_MANIFEST_FIELDS:
        if not manifest.get(field):
            raise AdapterError(f"target manifest {path} records no {field}")
    source_path = Path(str(manifest["source_structure_path"]))
    if not source_path.is_file():
        raise AdapterError(f"target source structure is missing: {source_path}")
    observed = sha256_file(source_path)
    if observed != str(manifest["target_sha256"]):
        raise AdapterError(
            f"target source structure changed since the target stage: {source_path}; "
            f"the manifest records {manifest['target_sha256']} and the file reads {observed}"
        )
    return manifest, path


# ----------------------------------------------------------------------------
# Design poses.
# ----------------------------------------------------------------------------


def write_design_pose(
    path: Path,
    *,
    candidate_id: str,
    source_pose: Path,
    source_sha256: str,
    atoms: list[str],
) -> None:
    """Write the design pose this candidate owns.

    The pose keeps every chain RFdiffusion returned, because the sequence
    designer downstream designs one chain against the rest as fixed context. The
    REMARK lines name the candidate and the tool output, which also keeps the
    bytes of every candidate pose distinct.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"REMARK 900 DESIGN POSE {candidate_id}",
        f"REMARK 900 SOURCE POSE {source_pose}",
        f"REMARK 900 SOURCE SHA256 {source_sha256}",
        "REMARK 900 BACKBONE ONLY WITH PLACEHOLDER RESIDUE IDENTITY",
    ]
    previous_chain: str | None = None
    for line in atoms:
        chain = line[21:22]
        if previous_chain is not None and chain != previous_chain:
            lines.append("TER")
        lines.append(line)
        previous_chain = chain
    lines.extend(["TER", "END"])
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write("\n".join(lines) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def produced_designs(work_dir: Path, output_stem: str) -> list[tuple[int, Path]]:
    """Return the numbered PDB files RFdiffusion wrote, in index order."""
    pattern = re.compile(rf"^{re.escape(output_stem)}_(\d+)\.pdb$")
    designs: list[tuple[int, Path]] = []
    for path in sorted(work_dir.glob(f"{output_stem}_*.pdb")):
        match = pattern.match(path.name)
        if match is not None:
            designs.append((int(match.group(1)), path))
    designs.sort(key=lambda item: item[0])
    return designs


# ----------------------------------------------------------------------------
# Subcommands.
# ----------------------------------------------------------------------------


def build_candidate(
    args: argparse.Namespace,
    *,
    manifest: dict[str, Any],
    index: int,
    design_index: int,
    design_path: Path,
    phase_dir: Path,
    effective_seed: int,
    requested_seed: int,
    contigs: str,
    hotspots: list[str],
    checkpoint: Path,
    verification: dict[str, str],
    input_structure: Path,
    input_structure_sha256: str,
    length_span: tuple[int, int] | None,
    runner_protocol: str,
) -> dict[str, Any]:
    """Return the manifest row of one produced backbone and write its design pose."""
    candidate_id = f"{args.generator_id}-{index:03d}"
    atoms = read_atom_records(design_path)
    residues = chain_residues(atoms)
    for label, chain in (("--binder-chain", args.binder_chain), ("--target-chain", args.target_chain)):
        if chain not in residues:
            available = ", ".join(residues) if residues else "no chain"
            raise AdapterError(
                f"{candidate_id} has no chain {chain} for {label}; {design_path} carries {available}"
            )
    binder_length = len(residues[args.binder_chain])
    if args.minimum_length is not None and binder_length < args.minimum_length:
        raise AdapterError(
            f"{candidate_id} chain {args.binder_chain} is {binder_length} residues, below the "
            f"{args.minimum_length} minimum"
        )
    if args.maximum_length is not None and binder_length > args.maximum_length:
        raise AdapterError(
            f"{candidate_id} chain {args.binder_chain} is {binder_length} residues, above the "
            f"{args.maximum_length} maximum"
        )
    if length_span is not None and not length_span[0] <= binder_length <= length_span[1]:
        raise AdapterError(
            f"{candidate_id} chain {args.binder_chain} is {binder_length} residues and the contigs "
            f"ask for {length_span[0]} to {length_span[1]}. Check that --binder-chain names the "
            "designed chain"
        )
    source_sha256 = sha256_file(design_path)
    pose_path = phase_dir / args.pose_subdir / f"{candidate_id}.pdb"
    write_design_pose(
        pose_path,
        candidate_id=candidate_id,
        source_pose=design_path,
        source_sha256=source_sha256,
        atoms=atoms,
    )
    row: dict[str, Any] = {
        "target_id": str(manifest["target_id"]),
        "target_sha256": str(manifest["target_sha256"]),
        "candidate_id": candidate_id,
        "parent_candidate_id": None,
        "origin_generator": args.generator_id,
        **backbone_lineage(candidate_id, args.generator_id),
        "generator_mode": GENERATOR_MODE,
        "runner_protocol": runner_protocol,
        "sequence_designer": None,
        "generator_seed": effective_seed,
        "requested_seed": requested_seed,
        "tool_seed": effective_seed,
        "sequence_path": None,
        "sequence_sha256": None,
        "sequence_length": None,
        "backbone_only": True,
        "structure_path": str(manifest["source_structure_path"]),
        "structure_sha256": str(manifest["target_sha256"]),
        "design_pose_path": str(pose_path.resolve()),
        "design_pose_sha256": sha256_file(pose_path),
        "residue_map_sha256": str(manifest["residue_map_sha256"]),
        "optimization_round": 0,
        "last_optimizer": None,
        "status": CANDIDATE_STATUS,
        "design_index": design_index,
        "binder_chain_id": args.binder_chain,
        "target_chain_id": args.target_chain,
        "binder_residue_count": binder_length,
        "target_residue_count": len(residues[args.target_chain]),
        "contigs": contigs,
        "hotspot_residues": list(hotspots),
        "checkpoint_path": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": verification["observed"],
        "model_revision_verified": verification["state"],
        "input_structure_path": str(input_structure),
        "input_structure_sha256": input_structure_sha256,
        "tool_output_path": str(design_path.resolve()),
        "tool_output_sha256": source_sha256,
    }
    trajectory = design_path.with_suffix(".trb")
    if trajectory.is_file():
        row["tool_metadata_path"] = str(trajectory.resolve())
        row["tool_metadata_sha256"] = sha256_file(trajectory)
    return row


def shard_context(
    args: argparse.Namespace, attempt_dir: Path, phase_dir: Path
) -> tuple[int, int, Path]:
    """Return the assigned candidate slice and its private output directory."""
    names = (
        "CLAUDE_BINDER_LANE_SHARD_START",
        "CLAUDE_BINDER_LANE_SHARD_STOP",
        "CLAUDE_BINDER_LANE_SHARD_OUT_DIR",
    )
    values = [os.environ.get(name, "").strip() for name in names]
    if not any(values):
        return 0, args.count, phase_dir
    if not all(values):
        missing = [name for name, value in zip(names, values) if not value]
        raise AdapterError(
            "the dispatcher shard contract is incomplete; missing " + ", ".join(missing)
        )
    try:
        start = int(values[0])
        stop = int(values[1])
    except ValueError as exc:
        raise AdapterError(
            "the dispatcher shard contract carries non-integer start or stop bounds"
        ) from exc
    if start < 0 or stop <= start or stop > args.count:
        raise AdapterError(
            f"the dispatcher shard slice {start}:{stop} is outside the phase count {args.count}"
        )
    output_root = Path(values[2]).expanduser().resolve()
    if attempt_dir not in output_root.parents:
        raise AdapterError(
            f"the dispatcher shard output directory escapes the attempt directory: {output_root}"
        )
    return start, stop, output_root


def run(args: argparse.Namespace) -> int:
    """Generate one phase of backbones and write the stage outputs."""
    if args.count < 1:
        raise AdapterError(f"--count is {args.count}; the phase needs at least one backbone")
    if args.binder_chain == args.target_chain:
        raise AdapterError(
            f"--binder-chain and --target-chain are both {args.binder_chain}; they name two "
            "different chains of the RFdiffusion output"
        )
    for label, chain in (("--binder-chain", args.binder_chain), ("--target-chain", args.target_chain)):
        if CHAIN_ID_RE.fullmatch(chain) is None:
            raise AdapterError(f"{label} is {chain}; a chain ID is one letter or digit")
    runner_protocol, root, weights_dir = resolve_execution(args)
    runner = resolve_runner(root)
    checkpoint = resolve_checkpoint(root, weights_dir, args.checkpoint_name)
    verification = verify_checkpoint_digest(checkpoint, resolve_model_revision(args))
    report_checkpoint_digest(checkpoint, verification)
    tool_python = resolve_tool_python(args.tool_python)

    attempt_dir = args.attempt_dir.expanduser().resolve()
    phase_dir = attempt_dir / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    shard_start, shard_stop, output_root = shard_context(args, attempt_dir, phase_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        resolve_output_path(attempt_dir, args.manifest_path, "manifest path")
        if args.manifest_path is not None
        else output_root / DEFAULT_MANIFEST_NAME
    )
    if output_root != phase_dir and output_root not in manifest_path.parents:
        raise AdapterError(
            f"manifest path must stay under the dispatcher shard output directory: {manifest_path}"
        )

    manifest, manifest_source = load_target_manifest(args)
    input_structure = (
        args.input_structure.expanduser().resolve()
        if args.input_structure is not None
        else Path(str(manifest["normalized_structure_path"])).expanduser().resolve()
    )
    input_atoms = read_atom_records(input_structure)
    input_residues = chain_residues(input_atoms, standard_only=True)
    chain_spans, length_spans = parse_contigs(args.contigs)
    check_contig_spans(chain_spans, input_residues, input_structure)
    hotspot_values = [*args.hotspot, *args.hotspot_csv]
    hotspots = parse_hotspots(hotspot_values, chain_spans, input_residues, input_structure)
    # RFdiffusion samples one length inside a de novo span, so a single span is a
    # check on the produced chain. Two or more spans can land in one chain, so the
    # wrapper records them and skips the check.
    length_span = (length_spans[0][1], length_spans[0][2]) if len(length_spans) == 1 else None

    requested_seed = args.seed + shard_start
    if requested_seed < 0:
        raise AdapterError(f"--seed plus shard start is {requested_seed}; a seed is not negative")
    effective_seed = tool_seed(requested_seed)

    work_dir = output_root / args.work_subdir
    work_dir.mkdir(parents=True, exist_ok=True)
    existing = produced_designs(work_dir, args.output_stem)
    if existing:
        raise AdapterError(
            f"the work directory already holds {len(existing)} {args.output_stem}_*.pdb files: "
            f"{work_dir}. RFdiffusion continues the numbering, so the count check cannot run"
        )
    output_prefix = work_dir / args.output_stem

    owned_keys = {
        "inference.input_pdb",
        "inference.output_prefix",
        "inference.num_designs",
        "inference.ckpt_override_path",
        "contigmap.contigs",
        "denoiser.noise_scale_ca",
        "denoiser.noise_scale_frame",
        args.seed_key,
    }
    if hotspots:
        owned_keys.add("ppi.hotspot_res")
    if args.diffuser_steps is not None:
        owned_keys.add("diffuser.T")
    overrides = check_overrides(args.extra_override, owned_keys)

    argv = [tool_python, str(runner)]
    if args.config_name is not None:
        argv.extend(["--config-name", args.config_name])
    argv.extend(
        [
            f"inference.input_pdb={input_structure}",
            f"inference.output_prefix={output_prefix}",
            f"inference.num_designs={shard_stop - shard_start}",
            f"inference.ckpt_override_path={checkpoint}",
            f"contigmap.contigs=[{args.contigs.strip()}]",
            f"denoiser.noise_scale_ca={args.noise_scale_ca}",
            f"denoiser.noise_scale_frame={args.noise_scale_frame}",
            f"{args.seed_key}={effective_seed}",
        ]
    )
    if hotspots:
        argv.append(f"ppi.hotspot_res=[{','.join(hotspots)}]")
    if args.diffuser_steps is not None:
        argv.append(f"diffuser.T={args.diffuser_steps}")
    argv.extend(overrides)
    run_tool(argv)

    designs = produced_designs(work_dir, args.output_stem)
    if len(designs) != shard_stop - shard_start:
        raise AdapterError(
            f"phase {args.phase} shard {shard_start}:{shard_stop} asked for {shard_stop - shard_start} "
            f"backbones and {work_dir} holds "
            f"{len(designs)} {args.output_stem}_*.pdb files"
        )
    input_structure_sha256 = sha256_file(input_structure)
    rows = [
        build_candidate(
            args,
            manifest=manifest,
            index=shard_start + index,
            design_index=shard_start + design_index,
            design_path=design_path,
            phase_dir=output_root,
            effective_seed=effective_seed,
            requested_seed=requested_seed,
            contigs=args.contigs.strip(),
            hotspots=hotspots,
            checkpoint=checkpoint,
            verification=verification,
            input_structure=input_structure,
            input_structure_sha256=input_structure_sha256,
            length_span=length_span,
            runner_protocol=runner_protocol,
        )
        for index, (design_index, design_path) in enumerate(designs)
    ]
    write_jsonl(manifest_path, rows)
    print(
        f"rfdiffusion adapter: phase={args.phase} target={manifest['target_id']} "
        f"route={runner_protocol} slice={shard_start}:{shard_stop} candidates={len(rows)} "
        f"seed={effective_seed} manifest={manifest_path} "
        f"target_manifest={manifest_source}"
    )
    return 0


def toolcheck(args: argparse.Namespace) -> int:
    """Report the RFdiffusion files and interpreter this adapter uses."""
    runner_protocol, root, weights_dir = resolve_execution(args)
    runner = resolve_runner(root)
    checkpoint = resolve_checkpoint(root, weights_dir, args.checkpoint_name)
    verification = verify_checkpoint_digest(checkpoint, resolve_model_revision(args))
    report_checkpoint_digest(checkpoint, verification)
    tool_python = resolve_tool_python(args.tool_python)
    completed = subprocess.run(
        [tool_python, str(runner), "--help"],
        shell=False,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise AdapterError(
            f"{RUNNER_NAME} --help exited {completed.returncode}: {completed.stderr.strip()[:400]}"
        )
    reported = next(
        (line.strip() for line in completed.stdout.splitlines() if line.strip()), "no output"
    )
    print(f"rfdiffusion adapter: route {runner_protocol}")
    print(f"rfdiffusion adapter: runner {runner}")
    print(f"rfdiffusion adapter: interpreter {tool_python}")
    print(f"rfdiffusion adapter: probe {reported}")
    print(
        "rfdiffusion adapter: the probe runs the help path only, because RFdiffusion loads the "
        "checkpoint when it designs"
    )
    return 0


def stage_record(config_path: Path, stage_id: str) -> dict[str, Any]:
    """Return one stage contract from a resolved config."""
    config = json.loads(config_path.read_text())
    matches = [
        stage
        for stage in config.get("stages", [])
        if isinstance(stage, dict) and stage.get("stage_id") == stage_id
    ]
    if len(matches) != 1:
        raise AdapterError(
            f"{config_path} registers {len(matches)} stages with id {stage_id}; "
            "the parser needs exactly one"
        )
    return matches[0]


def parser_output_pattern(template: str, attempt_dir: Path, phase: str) -> str:
    """Render an output contract path for the parser."""
    rendered = template.replace("{{attempt_dir}}", str(attempt_dir)).replace("{{phase}}", phase)
    if "{{" in rendered or "}}" in rendered:
        raise AdapterError(f"parser output path carries an unsupported token: {template}")
    return rendered


def parse_outputs(args: argparse.Namespace) -> int:
    """Check the merged phase outputs and write the lane parser result."""
    attempt_dir = args.attempt_dir.expanduser().resolve()
    phase_dir = attempt_dir / args.phase
    stage = stage_record(args.config.expanduser().resolve(), args.stage)
    files: list[Path] = []
    parsed_count = 0
    errors: list[str] = []
    for output in stage.get("outputs", []):
        if not isinstance(output, dict) or not isinstance(output.get("path_template"), str):
            errors.append("stage output has no path_template")
            continue
        pattern = parser_output_pattern(output["path_template"], attempt_dir, args.phase)
        for value in sorted(glob.glob(pattern, recursive=True)):
            path = Path(value)
            if not path.is_file():
                continue
            files.append(path)
            try:
                kind = output.get("kind")
                if kind == "jsonl":
                    for line in path.read_text().splitlines():
                        if line.strip():
                            json.loads(line)
                            parsed_count += 1
                elif kind == "json":
                    json.loads(path.read_text())
                    parsed_count += 1
                else:
                    parsed_count += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    result_path = phase_dir / "parser-result.json"
    write_json(
        result_path,
        {
            "ok": bool(files) and not errors,
            "parsed_count": parsed_count,
            "rejected_count": len(errors),
            "errors": errors,
            "source_output_hashes": sorted(sha256_file(path) for path in files),
        },
    )
    print(
        f"rfdiffusion adapter: parsed={parsed_count} rejected={len(errors)} "
        f"phase={args.phase} result={result_path}"
    )
    return 0 if files and not errors else 1


def add_tool_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runner-protocol",
        choices=RUNNER_PROTOCOLS,
        default=DEFAULT_RUNNER_PROTOCOL,
        help=(
            "Where RFdiffusion runs. auto uses an explicit local checkout when present and "
            "otherwise the shipped Modal mounts. Use local or modal to force one path."
        ),
    )
    parser.add_argument(
        "--rfdiffusion-root",
        type=Path,
        default=None,
        help="RFdiffusion checkout directory. Defaults to RFDIFFUSION_ROOT.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help=f"Checkpoint directory. Defaults to {DEFAULT_WEIGHTS_SUBDIR} inside the checkout.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=DEFAULT_CHECKPOINT_NAME,
        help=f"Checkpoint file name. Defaults to {DEFAULT_CHECKPOINT_NAME}.",
    )
    parser.add_argument(
        "--tool-python",
        default=None,
        help="Interpreter that runs RFdiffusion. Defaults to the interpreter running this wrapper.",
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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Resolved run config. The wrapper reads the model_revision for the "
            f"{DEFAULT_ADAPTER_ID} adapter and checks the checkpoint against it."
        ),
    )
    parser.add_argument(
        "--adapter-id",
        default=DEFAULT_ADAPTER_ID,
        help=f"Adapter ID used when reading --config. Defaults to {DEFAULT_ADAPTER_ID}.",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("toolcheck", help="Probe the runtime without designing.")
    add_tool_arguments(check_parser)
    run_parser = subparsers.add_parser("run", help="Generate backbones for one phase.")
    add_tool_arguments(run_parser)
    run_parser.add_argument(
        "--phase", required=True, help="Stage phase name, such as smoke or scale."
    )
    run_parser.add_argument(
        "--count", type=int, required=True, help="Number of backbones this phase generates."
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
        "--target-manifest",
        type=Path,
        default=None,
        help="Published target manifest under the artifact root. Overrides the receipt lookup.",
    )
    run_parser.add_argument(
        "--target-stage-id",
        default=DEFAULT_TARGET_STAGE_ID,
        help=f"Stage ID of the target preparer. Defaults to {DEFAULT_TARGET_STAGE_ID}.",
    )
    run_parser.add_argument(
        "--target-artifact-id",
        default=DEFAULT_TARGET_ARTIFACT_ID,
        help=f"Artifact ID of the target manifest. Defaults to {DEFAULT_TARGET_ARTIFACT_ID}.",
    )
    run_parser.add_argument(
        "--input-structure",
        type=Path,
        default=None,
        help=(
            "Structure RFdiffusion reads. Defaults to the normalized structure the target "
            "manifest names."
        ),
    )
    run_parser.add_argument(
        "--generator-id",
        default=DEFAULT_GENERATOR_ID,
        help=(
            "Generator ID and candidate ID prefix. Match the generator ID the campaign "
            f"registers. Defaults to {DEFAULT_GENERATOR_ID}."
        ),
    )
    run_parser.add_argument(
        "--contigs",
        required=True,
        help=(
            "Contig specification without the Hydra brackets, for example 'A19-127/0 60-90'. "
            "Every chain segment has to name residues the input structure carries."
        ),
    )
    run_parser.add_argument(
        "--hotspot",
        action="append",
        default=[],
        help="Hotspot residue as CHAIN:NUMBER or CHAINNUMBER. Repeat for every hotspot.",
    )
    run_parser.add_argument(
        "--hotspot-csv",
        action="append",
        default=[],
        help=(
            "Comma-separated hotspot residues or ranges, matching the lane's "
            "target_residues_csv token. Repeat to provide more than one CSV value."
        ),
    )
    run_parser.add_argument(
        "--binder-chain",
        required=True,
        help="Chain ID of the designed chain in the RFdiffusion output.",
    )
    run_parser.add_argument(
        "--target-chain",
        required=True,
        help="Chain ID of the fixed chain in the RFdiffusion output.",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Requested seed. A requested seed of 0 becomes a fixed nonzero seed.",
    )
    run_parser.add_argument(
        "--seed-key",
        default=DEFAULT_SEED_KEY,
        help=f"Hydra key that carries the seed. Defaults to {DEFAULT_SEED_KEY}.",
    )
    run_parser.add_argument(
        "--noise-scale-ca",
        type=float,
        default=0.0,
        help="denoiser.noise_scale_ca. Defaults to 0.",
    )
    run_parser.add_argument(
        "--noise-scale-frame",
        type=float,
        default=0.0,
        help="denoiser.noise_scale_frame. Defaults to 0.",
    )
    run_parser.add_argument(
        "--diffuser-steps",
        type=int,
        default=None,
        help="diffuser.T. Left out of the command when absent.",
    )
    run_parser.add_argument(
        "--config-name",
        default=None,
        help="Hydra config name. Left out of the command when absent.",
    )
    run_parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help=(
            "Extra Hydra override as key=value. Repeat for every override. A key the wrapper "
            "already sets is refused."
        ),
    )
    run_parser.add_argument(
        "--minimum-length",
        type=int,
        default=None,
        help="Reject a designed chain below this residue count.",
    )
    run_parser.add_argument(
        "--maximum-length",
        type=int,
        default=None,
        help="Reject a designed chain above this residue count.",
    )
    run_parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=f"Manifest path. Defaults to {DEFAULT_MANIFEST_NAME} in the phase directory.",
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
            "Raw RFdiffusion output directory inside the phase directory. Defaults to "
            f"{DEFAULT_WORK_SUBDIR}."
        ),
    )
    run_parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help=(
            "File-name stem RFdiffusion numbers its output with. Defaults to "
            f"{DEFAULT_OUTPUT_STEM}."
        ),
    )
    parse_parser = subparsers.add_parser("parse", help="Parse the outputs of one completed phase.")
    parse_parser.add_argument("--stage", required=True, help="Stage ID in the resolved config.")
    parse_parser.add_argument("--phase", required=True, help="Stage phase name, such as smoke or scale.")
    parse_parser.add_argument("--count", type=int, default=1, help="Expected phase count for the caller.")
    parse_parser.add_argument("--attempt-dir", type=Path, required=True, help="Attempt directory that owns the outputs.")
    parse_parser.add_argument("--receipts-dir", type=Path, required=False, help="Receipt directory passed by the dispatcher.")
    parse_parser.add_argument("--artifact-root", type=Path, required=False, help="Artifact root passed by the dispatcher.")
    parse_parser.add_argument("--config", type=Path, required=True, help="Resolved run config.")
    parse_parser.add_argument("--plan", type=Path, required=False, help="Run plan passed by the dispatcher.")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        if args.command == "toolcheck":
            return toolcheck(args)
        if args.command == "parse":
            return parse_outputs(args)
        return run(args)
    except AdapterError as exc:
        print(f"rfdiffusion adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
