#!/usr/bin/env python3
"""Join generator and sequence-designer receipts into one candidate manifest.

The normalizer reads completed upstream receipts. It verifies lineage and file hashes,
copies sequence and pose files into its own attempt directory, and writes the manifest
that the filter stages consume.

The manifest also carries the model-likelihood metric when the campaign asks for it.
``novelty_filter`` reads one JSONL file per novelty check and needs a ``candidate_id``
and a finite numeric ``value`` on every row. A campaign that names this manifest in
``filters.metric_sources.model_likelihood`` gets both fields written here.

The parser reports the same record units as the lane executor. JSONL counts non-blank
object lines, FASTA counts headers, and PDB counts poses.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from claude_binder.adapters.adapter_io import read_jsonl as _read_jsonl
from claude_binder.adapters.candidate_lineage import DIVERSITY_LINEAGE_FIELDS


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FASTA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
ATOM_PREFIXES = ("ATOM  ", "HETATM")
REQUIRED_ROW_FIELDS = (
    "target_id",
    "target_sha256",
    "candidate_id",
    "parent_candidate_id",
    "origin_generator",
    "generator_mode",
    "sequence_designer",
    "generator_seed",
    "sequence_path",
    "sequence_sha256",
    "sequence_length",
    "structure_path",
    "structure_sha256",
    "design_pose_path",
    "design_pose_sha256",
    "residue_map_sha256",
    "optimization_round",
    "last_optimizer",
    *DIVERSITY_LINEAGE_FIELDS,
    "status",
)
CODESIGN_MODE = "sequence-structure-codesign"
VALID_MODES = {"backbone-only", CODESIGN_MODE}
VALID_STATUSES = {"generated", "sequence-designed", "filtered", "promoted", "failed"}
MODEL_LIKELIHOOD_FILTER_ID = "model_likelihood"
# ProteinMPNN writes two numbers on every design header. `score` is the mean negative
# log-likelihood over the residues the run designed. `global_score` is the same average
# taken over every residue of the input, so it folds the fixed target chain into the
# number. The binder's own likelihood is the designed-residue average, so this reads
# `score`, which proteinmpnn_designer.py records as `proteinmpnn_score`.
MODEL_LIKELIHOOD_SOURCE_FIELD = "proteinmpnn_score"
# Where the lane publishes this stage's manifest, relative to the artifact root.
CANDIDATE_MANIFEST_PUBLISH_PARTS = ("candidates", "candidate-manifest.jsonl")


class AdapterError(RuntimeError):
    """An input or output condition that must stop the adapter."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def read_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AdapterError(f"{label} is invalid: {path}: {type(exc).__name__}: {exc}") from exc


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path, label)
    if not isinstance(value, dict):
        raise AdapterError(f"{label} must be a JSON object: {path}")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    return _read_jsonl(path, label, error_type=AdapterError)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AdapterError(f"refusing to write an empty candidate manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise AdapterError(f"{label} must be a 64-character lowercase SHA-256 digest")
    return value


def require_file(path_value: Any, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise AdapterError(f"{label} must be a non-empty absolute path")
    path = Path(path_value)
    if not path.is_absolute():
        raise AdapterError(f"{label} must be absolute: {path}")
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    return path


def read_fasta(path: Path, candidate_id: str, label: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [line[1:] for line in lines if line.startswith(">")]
    if len(headers) != 1:
        raise AdapterError(f"{label} must contain one FASTA record: {path}")
    if headers[0] != candidate_id:
        raise AdapterError(f"{label} FASTA header does not match candidate_id: {path}")
    sequence = "".join(line.strip() for line in lines if line and not line.startswith(">"))
    if not FASTA_RE.fullmatch(sequence):
        raise AdapterError(f"{label} contains an invalid amino-acid sequence: {path}")
    return sequence


def receipt_rows(receipts_dir: Path, stage_id: str, artifact_types: set[str]) -> list[dict[str, Any]]:
    receipt_path = receipts_dir / f"{stage_id}.json"
    receipt = read_json_object(receipt_path, "upstream receipt")
    if receipt.get("ok") is not True:
        raise AdapterError(f"upstream receipt did not complete: {receipt_path}")
    output_manifest = receipt.get("output_manifest")
    if not isinstance(output_manifest, dict):
        raise AdapterError(f"upstream receipt has no output_manifest object: {receipt_path}")
    artifacts = output_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise AdapterError(f"upstream receipt has no artifact list: {receipt_path}")
    phases = {str(item.get("phase")) for item in artifacts if isinstance(item, dict)}
    selected_phase = "scale" if "scale" in phases else "single" if "single" in phases else ""
    if not selected_phase:
        raise AdapterError(f"upstream receipt has no scale or single artifact phase: {receipt_path}")
    rows: list[dict[str, Any]] = []
    matched = False
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("phase") != selected_phase:
            continue
        artifact_type = str(artifact.get("artifact_type", ""))
        artifact_id = str(artifact.get("artifact_id", ""))
        if artifact_type not in artifact_types and artifact_id not in artifact_types:
            continue
        matched = True
        files = artifact.get("files")
        if not isinstance(files, list):
            raise AdapterError(f"upstream receipt artifact has no file list: {receipt_path}")
        for file_record in files:
            if not isinstance(file_record, dict) or not isinstance(file_record.get("path"), str):
                raise AdapterError(f"upstream receipt artifact has an invalid file path: {receipt_path}")
            manifest_path = Path(file_record["path"])
            if manifest_path.suffix.lower() != ".jsonl":
                continue
            rows.extend(read_jsonl(manifest_path, "upstream manifest"))
    if not matched:
        names = ", ".join(sorted(artifact_types))
        raise AdapterError(f"upstream receipt has no {names} artifact: {receipt_path}")
    if not rows:
        raise AdapterError(f"upstream receipt has no candidate rows: {receipt_path}")
    return rows


def enabled_items(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise AdapterError(f"config field is missing or malformed: {label}")
    items = [item for item in value if isinstance(item, dict) and item.get("enabled", True) is True]
    if not items:
        raise AdapterError(f"config field has no enabled entries: {label}")
    return items


def primary_values(config: dict[str, Any], config_path: Path) -> tuple[str, str, str]:
    targets = config.get("targets")
    if not isinstance(targets, list):
        raise AdapterError(f"config field is missing or malformed: {config_path}: targets")
    matches = [item for item in targets if isinstance(item, dict) and item.get("role") == "primary"]
    if len(matches) != 1:
        raise AdapterError(f"config must contain one primary target: {config_path}: targets")
    target = matches[0]
    target_id = target.get("target_id")
    if not isinstance(target_id, str) or not target_id:
        raise AdapterError(f"config value is missing: {config_path}: targets[primary].target_id")
    target_sha = require_digest(
        target.get("structure_sha256"),
        f"config value {config_path}: targets[primary].structure_sha256",
    )
    site = target.get("site")
    if not isinstance(site, dict):
        raise AdapterError(f"config value is missing: {config_path}: targets[primary].site")
    residue_sha = require_digest(
        site.get("residue_map_sha256"),
        f"config value {config_path}: targets[primary].site.residue_map_sha256",
    )
    return target_id, target_sha, residue_sha


DEFAULT_SOURCE_REFS = [("sequence-proteinmpnn", "proteinmpnn-candidates")]


def declared_source_refs(plan_path: Path) -> list[tuple[str, str]]:
    """Return every candidate input the plan declares for normalize-candidates.

    The defect this guards against is the adapter choosing its own sources from
    configured generator and designer entries. The remedy is to read what the plan
    declares, not to pin one designer's stage name and not to pin how many there are.

    The shipped small-run campaign declares one input, the ProteinMPNN candidates. A
    co-design campaign declares two, because its generator emits candidates of its own
    alongside the sequence designer's, and the executor's lineage check expects both to
    appear in the normalized set. Reading the declared list covers both without the
    adapter knowing which campaign it is in.
    """
    document = read_json_object(plan_path.resolve(), "run plan")
    stages = document.get("stages")
    if not isinstance(stages, list) or not stages:
        # A few unit fixtures use an empty plan. They model the shipped campaign,
        # whose designer is ProteinMPNN.
        return list(DEFAULT_SOURCE_REFS)
    matches = [
        stage for stage in stages
        if isinstance(stage, dict) and stage.get("stage_id") == "normalize-candidates"
    ]
    if len(matches) != 1:
        raise AdapterError("run plan must define one stage: normalize-candidates")
    inputs = matches[0].get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise AdapterError(
            "normalize-candidates must declare at least one input, the candidate "
            f"artifacts it normalizes; declared inputs={inputs!r}"
        )
    refs: list[tuple[str, str]] = []
    for reference in inputs:
        if not isinstance(reference, str):
            raise AdapterError(f"normalize-candidates input must be a string; got {reference!r}")
        stage_id, separator, artifact_id = reference.partition(":")
        if not separator or not stage_id or not artifact_id:
            raise AdapterError(
                f"normalize-candidates input must read stage_id:artifact_id; got {reference!r}"
            )
        refs.append((stage_id, artifact_id))
    return refs


def source_rows(
    config: dict[str, Any],
    config_path: Path,
    receipts_dir: Path,
    *,
    plan_path: Path | None = None,
    source_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    """Read only the candidate artifact this stage declares as its input.

    Configured generator and designer entries remain policy metadata. They do
    not choose which receipts this stage opens.
    """
    if source_manifest is not None:
        manifest = source_manifest.expanduser().resolve()
        if not manifest.is_file():
            raise AdapterError(f"source manifest override is missing: {manifest}")
        rows = read_jsonl(manifest, "source manifest override")
        source_label = f"source manifest override {manifest}"
    else:
        refs = list(DEFAULT_SOURCE_REFS) if plan_path is None else declared_source_refs(plan_path)
        rows = []
        for stage_id, artifact_id in refs:
            rows.extend(receipt_rows(receipts_dir, stage_id, {artifact_id}))
        source_label = "declared " + ", ".join(f"{stage}:{artifact}" for stage, artifact in refs)
    if not rows:
        raise AdapterError(f"{source_label} has no rows")
    seen: set[str] = set()
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise AdapterError(f"{source_label} has a row without candidate_id")
        if candidate_id in seen:
            raise AdapterError(f"{source_label} repeats candidate_id: {candidate_id}")
        seen.add(candidate_id)
        # A candidate the generator co-designed is a root of the lineage, so it has no
        # parent and no separate sequence designer. A candidate a designer produced was
        # designed onto a backbone, so it has both. The row says which it is, and reading
        # that is what lets one campaign use ProteinMPNN over RFdiffusion backbones and
        # another use a co-design generator, without this adapter knowing either.
        if row.get("generator_mode") == CODESIGN_MODE:
            continue
        if not isinstance(row.get("parent_candidate_id"), str) or not row.get("parent_candidate_id"):
            raise AdapterError(f"{source_label} candidate {candidate_id} has no parent_candidate_id")
        if not isinstance(row.get("sequence_designer"), str) or not row.get("sequence_designer"):
            raise AdapterError(f"{source_label} candidate {candidate_id} has no sequence_designer")
    return sorted(rows, key=lambda row: str(row["candidate_id"]))


def configured_source_manifest(
    config: dict[str, Any], config_path: Path
) -> Path | None:
    """Resolve the optional portable supplied-candidate manifest."""
    supplied = config.get("supplied_candidates")
    if not isinstance(supplied, dict):
        return None
    value = supplied.get("runtime_manifest_path") or supplied.get("manifest_path")
    if not isinstance(value, str) or not value:
        raise AdapterError("supplied_candidates.manifest_path must name a JSONL file")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def model_likelihood_metric(
    config: dict[str, Any], config_path: Path, artifact_root: Path
) -> str | None:
    """Return the metric name when the campaign reads this manifest as its likelihood source.

    The campaign names one executed output per novelty check under
    ``filters.metric_sources``. When it names the manifest this stage publishes, the rows
    have to carry the ``metric`` and ``value`` fields ``novelty_filter`` reads. When it
    names anything else, the manifest keeps the shape it has always had.
    """
    filters = config.get("filters")
    if not isinstance(filters, dict):
        return None
    sources = filters.get("metric_sources")
    if not isinstance(sources, dict):
        return None
    source = sources.get(MODEL_LIKELIHOOD_FILTER_ID)
    if not isinstance(source, dict):
        return None
    declared = source.get("path")
    if not isinstance(declared, str) or not declared:
        return None
    path = Path(declared)
    if not path.is_absolute():
        path = config_path.parent / path
    if path.resolve() != artifact_root.joinpath(*CANDIDATE_MANIFEST_PUBLISH_PARTS).resolve():
        return None
    contracts = filters.get("contracts")
    if not isinstance(contracts, list):
        raise AdapterError(f"config field is missing or malformed: {config_path}: filters.contracts")
    matches = [
        item
        for item in contracts
        if isinstance(item, dict) and item.get("filter_id") == MODEL_LIKELIHOOD_FILTER_ID
    ]
    if len(matches) != 1:
        raise AdapterError(
            f"config names this manifest as the {MODEL_LIKELIHOOD_FILTER_ID} metric source "
            f"and must define exactly one matching filter contract: {config_path}: filters.contracts"
        )
    metric = matches[0].get("metric")
    if not isinstance(metric, str) or not metric or metric == "__REQUIRED__":
        raise AdapterError(
            f"config value is missing: {config_path}: "
            f"filters.contracts[{MODEL_LIKELIHOOD_FILTER_ID}].metric"
        )
    return metric


def model_likelihood_value(source: dict[str, Any], candidate_id: str) -> float:
    """Return one candidate's designed-residue ProteinMPNN score."""
    value = source.get(MODEL_LIKELIHOOD_SOURCE_FIELD)
    if value is None:
        raise AdapterError(
            f"upstream candidate {candidate_id!r} carries no {MODEL_LIKELIHOOD_SOURCE_FIELD}, "
            f"and the campaign reads this manifest as its {MODEL_LIKELIHOOD_FILTER_ID} metric "
            f"source. Only a sequence designer that reports a per-sequence score can feed "
            f"that filter."
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise AdapterError(
            f"upstream candidate {candidate_id!r} has a non-finite "
            f"{MODEL_LIKELIHOOD_SOURCE_FIELD}: {value!r}"
        )
    return float(value)


def safe_stem(candidate_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id).strip(".")
    if not stem or stem in {".", ".."}:
        raise AdapterError(f"candidate_id cannot produce a safe output filename: {candidate_id!r}")
    return stem


def normalize_rows(
    rows: list[dict[str, Any]],
    *,
    target_id: str,
    target_sha: str,
    residue_sha: str,
    stage_dir: Path,
    likelihood_metric: str | None = None,
) -> list[dict[str, Any]]:
    sequence_dir = stage_dir / "sequences"
    pose_dir = stage_dir / "poses"
    sequence_dir.mkdir(parents=True)
    pose_dir.mkdir()
    used_stems: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for source in rows:
        candidate_id = source.get("candidate_id")
        label = f"upstream candidate {candidate_id!r}"
        missing = [field for field in REQUIRED_ROW_FIELDS if field not in source]
        if missing:
            raise AdapterError(f"{label} is missing fields: {', '.join(missing)}")
        if source.get("target_id") != target_id:
            raise AdapterError(f"{label} changed target_id from config")
        if source.get("target_sha256") != target_sha:
            raise AdapterError(f"{label} changed target_sha256 from config")
        require_digest(source.get("target_sha256"), f"{label}.target_sha256")
        require_digest(source.get("structure_sha256"), f"{label}.structure_sha256")
        require_digest(source.get("design_pose_sha256"), f"{label}.design_pose_sha256")
        require_digest(source.get("residue_map_sha256"), f"{label}.residue_map_sha256")
        if source.get("residue_map_sha256") != residue_sha:
            raise AdapterError(f"{label} changed residue_map_sha256 from config")
        if source.get("generator_mode") not in VALID_MODES:
            raise AdapterError(f"{label}.generator_mode is not registered")
        if not isinstance(source.get("generator_seed"), int) or isinstance(source.get("generator_seed"), bool):
            raise AdapterError(f"{label}.generator_seed must be an integer")
        if source.get("optimization_round") != 0:
            raise AdapterError(f"{label}.optimization_round must be 0 for normalize-candidates")
        if source.get("last_optimizer") is not None:
            raise AdapterError(f"{label}.last_optimizer must be null for normalize-candidates")
        if source.get("status") not in VALID_STATUSES:
            raise AdapterError(f"{label}.status is not registered")
        structure_path = require_file(source.get("structure_path"), f"{label}.structure_path")
        if sha256_file(structure_path) != source["structure_sha256"]:
            raise AdapterError(f"{label}.structure_sha256 does not match file content: {structure_path}")
        pose_path = require_file(source.get("design_pose_path"), f"{label}.design_pose_path")
        if sha256_file(pose_path) != source["design_pose_sha256"]:
            raise AdapterError(f"{label}.design_pose_sha256 does not match file content: {pose_path}")
        sequence_path = require_file(source.get("sequence_path"), f"{label}.sequence_path")
        sequence = read_fasta(sequence_path, str(candidate_id), f"{label}.sequence_path")
        if source.get("sequence_length") != len(sequence):
            raise AdapterError(f"{label}.sequence_length does not match FASTA content: {sequence_path}")
        if source.get("sequence_sha256") != canonical_sequence_sha256(sequence):
            raise AdapterError(f"{label}.sequence_sha256 does not match canonical FASTA content: {sequence_path}")
        stem = safe_stem(str(candidate_id))
        if stem in used_stems:
            raise AdapterError(f"candidate IDs collide after filename normalization: {candidate_id}")
        used_stems.add(stem)
        output_sequence = sequence_dir / f"{stem}.fasta"
        # The copy is byte-for-byte, so the name has to keep the format the bytes
        # are in. A hardcoded `.pdb` renamed a supplied mmCIF pose into a PDB name,
        # and every downstream reader dispatches on the suffix, so the pose parsed
        # as an empty PDB. A generator writing PDB is unaffected; only a supplied
        # candidate can arrive in another format.
        output_pose = pose_dir / f"{stem}{pose_path.suffix.lower()}"
        shutil.copy2(sequence_path, output_sequence)
        shutil.copy2(pose_path, output_pose)
        # The audit reads a stage's output contract from the keys its adapter
        # writes, and a wholesale dict copy names none of them. Writing every
        # required field explicitly is what declares that this arm originates the
        # supplied record instead of sourcing it from an upstream stage. Each read
        # is safe because the missing-field check above already refused the row.
        copied = {
            "target_id": source["target_id"],
            "target_sha256": source["target_sha256"],
            "candidate_id": source["candidate_id"],
            "parent_candidate_id": source["parent_candidate_id"],
            "origin_generator": source["origin_generator"],
            "generator_mode": source["generator_mode"],
            "sequence_designer": source["sequence_designer"],
            "generator_seed": source["generator_seed"],
            "sequence_path": str(output_sequence.resolve()),
            "sequence_sha256": source["sequence_sha256"],
            "sequence_length": source["sequence_length"],
            "structure_path": source["structure_path"],
            "structure_sha256": source["structure_sha256"],
            "design_pose_path": str(output_pose.resolve()),
            "design_pose_sha256": source["design_pose_sha256"],
            "residue_map_sha256": source["residue_map_sha256"],
            "optimization_round": source["optimization_round"],
            "last_optimizer": source["last_optimizer"],
            "root_backbone_id": source["root_backbone_id"],
            "tm90_cluster_id": source["tm90_cluster_id"],
            "structure_method": source["structure_method"],
            "seq_method": source["seq_method"],
            "fold_class": source["fold_class"],
            "status": source["status"],
        }
        for field, value in source.items():
            if field not in copied:
                copied[field] = value
        if likelihood_metric is not None:
            copied["metric"] = likelihood_metric
            copied["value"] = model_likelihood_value(source, str(candidate_id))
        normalized.append(copied)
    if not normalized:
        raise AdapterError("refusing to write an empty normalized candidate manifest")
    return normalized


def run_stage(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = read_json_object(config_path, "campaign config")
    target_id, target_sha, residue_sha = primary_values(config, config_path)
    likelihood_metric = model_likelihood_metric(config, config_path, args.artifact_root.resolve())
    source_manifest = args.source_manifest or configured_source_manifest(
        config, config_path
    )
    rows = source_rows(
        config,
        config_path,
        args.receipts_dir.resolve(),
        plan_path=args.plan,
        source_manifest=source_manifest,
    )
    if source_manifest is not None and args.count is not None:
        requested = 1 if args.phase == "smoke" else args.count
        if requested < 1:
            raise AdapterError("supplied-candidate count must be positive")
        if len(rows) < requested:
            raise AdapterError(
                f"supplied candidate manifest has {len(rows)} rows and this phase "
                f"requests {requested}"
            )
        rows = rows[:requested]
    phase_dir = (args.attempt_dir / args.phase).resolve()
    if any((phase_dir / name).exists() for name in ("candidate-manifest.jsonl", "sequences", "poses")):
        raise AdapterError(f"candidate normalizer output already exists under the attempt phase: {phase_dir}")
    phase_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="candidate-normalizer-", dir=phase_dir) as temporary:
        stage_dir = Path(temporary)
        normalized = normalize_rows(
            rows,
            target_id=target_id,
            target_sha=target_sha,
            residue_sha=residue_sha,
            stage_dir=stage_dir,
            likelihood_metric=likelihood_metric,
        )
        for row in normalized:
            row["sequence_path"] = str(
                (phase_dir / "sequences" / Path(row["sequence_path"]).name).resolve()
            )
            row["design_pose_path"] = str(
                (phase_dir / "poses" / Path(row["design_pose_path"]).name).resolve()
            )
        write_jsonl(stage_dir / "candidate-manifest.jsonl", normalized)
        os.replace(stage_dir / "candidate-manifest.jsonl", phase_dir / "candidate-manifest.jsonl")
        os.replace(stage_dir / "sequences", phase_dir / "sequences")
        os.replace(stage_dir / "poses", phase_dir / "poses")
    print(f"candidate normalizer: phase={args.phase} candidates={len(normalized)} manifest={phase_dir / 'candidate-manifest.jsonl'}")
    return 0


def scan_jsonl(path: Path, errors: list[str]) -> int:
    if not path.is_file():
        errors.append(f"declared output matched no files: {path}")
        return 0
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception as exc:
            errors.append(f"{path} line {line_number}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path} line {line_number} is not a JSON object")
            continue
        count += 1
    if count == 0:
        errors.append(f"file holds no JSONL records: {path}")
    return count


def count_pdb(path: Path) -> tuple[int, list[str]]:
    """Return the executor's number of atom-containing PDB poses and errors."""
    pdb_errors: list[str] = []
    modeled_pose_count = 0
    saw_model_delimiter = False
    in_model = False
    model_has_atoms = False
    unmodeled_atoms = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MODEL "):
            saw_model_delimiter = True
            if in_model:
                pdb_errors.append(f"PDB starts a MODEL before ENDMDL: {path}")
            in_model = True
            model_has_atoms = False
            continue
        if line.startswith("ENDMDL"):
            saw_model_delimiter = True
            if not in_model:
                pdb_errors.append(f"PDB has ENDMDL without MODEL: {path}")
            elif model_has_atoms:
                modeled_pose_count += 1
            else:
                pdb_errors.append(f"PDB MODEL has no atom records: {path}")
            in_model = False
            model_has_atoms = False
            continue
        if line.startswith(ATOM_PREFIXES):
            if in_model:
                model_has_atoms = True
            else:
                unmodeled_atoms = True
    if in_model:
        pdb_errors.append(f"PDB MODEL has no closing ENDMDL: {path}")
    if saw_model_delimiter:
        if unmodeled_atoms:
            pdb_errors.append(f"PDB with MODEL delimiters has atom records outside a MODEL: {path}")
        if modeled_pose_count == 0 and not pdb_errors:
            pdb_errors.append(f"PDB has no atom records: {path}")
        return modeled_pose_count, pdb_errors
    if unmodeled_atoms:
        return 1, pdb_errors
    return 0, [f"PDB has no atom records: {path}"]


def scan_files(family: str, patterns: list[str], attempt_dir: Path, errors: list[str]) -> tuple[int, list[Path]]:
    count = 0
    files: list[Path] = []
    for pattern in patterns:
        raw = Path(pattern)
        resolved_pattern = Path(os.path.normpath(attempt_dir / raw)) if not raw.is_absolute() else Path(os.path.normpath(raw))
        if attempt_dir not in resolved_pattern.resolve().parents:
            errors.append(f"declared output escapes the attempt directory: {pattern}")
            continue
        matches = sorted(Path(value) for value in glob.glob(str(resolved_pattern), recursive=True))
        if not matches:
            errors.append(f"declared output matched no files: {pattern}")
            continue
        for path in matches:
            if not path.is_file():
                continue
            resolved = path.resolve()
            if attempt_dir not in resolved.parents:
                errors.append(f"matched output resolves outside the attempt directory: {path}")
                continue
            files.append(path)
            if family == "jsonl":
                count += scan_jsonl(path, errors)
            elif family == "fasta":
                records = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(">"))
                if records == 0:
                    errors.append(f"file holds no FASTA records: {path}")
                count += records
            elif family == "pdb":
                records, pdb_errors = count_pdb(path)
                errors.extend(pdb_errors)
                count += records
            elif family == "structure":
                # A supplied pose arrives in whatever format the operator holds and
                # this stage copies those bytes, so the family cannot pick a format
                # in advance. This reads it off the suffix, the same way
                # ``lane.validate_stage_outputs`` handles ``kind: "structure"``.
                if path.suffix.lower() in {".cif", ".mmcif"}:
                    if "_atom_site." in path.read_text(encoding="utf-8", errors="replace"):
                        count += 1
                    else:
                        errors.append(f"mmCIF has no atom_site category: {path}")
                else:
                    records, pdb_errors = count_pdb(path)
                    errors.extend(pdb_errors)
                    count += records
    return count, files


def parse_stage(args: argparse.Namespace) -> int:
    attempt_dir = args.attempt_dir.resolve()
    result_path = (attempt_dir / args.phase / "parser-result.json").resolve()
    errors: list[str] = []
    if attempt_dir not in result_path.parents:
        errors.append(f"parser result path escapes the attempt directory: {result_path}")
    declared: list[tuple[str, list[str]]] = [
        ("jsonl", args.jsonl or []),
        ("fasta", args.fasta or []),
        ("pdb", args.pdb or []),
        # getattr, because callers that build the namespace directly predate this
        # family and pass no attribute for it.
        ("structure", getattr(args, "structure", None) or []),
    ]
    files: list[Path] = []
    parsed_count = 0
    if not any(patterns for _, patterns in declared):
        errors.append("no declared output glob was given")
    for family, patterns in declared:
        count, found = scan_files(family, patterns, attempt_dir, errors)
        parsed_count += count
        files.extend(found)
    hashes = [sha256_file(path) for path in files]
    if len(hashes) != len(set(hashes)):
        errors.append("declared outputs repeat file bytes, and the executor requires unique hashes")
    result = {
        "ok": bool(files) and not errors,
        "parsed_count": parsed_count,
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": sorted(hashes),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for error in errors:
        print(f"candidate normalizer parser: {error}", file=sys.stderr)
    print(f"candidate normalizer parser: phase={args.phase} parsed_count={parsed_count} files={len(files)} ok={result['ok']}")
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("toolcheck")
    for name in ("run", "parse"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--phase", required=True)
        subparser.add_argument("--attempt-dir", type=Path, required=True)
        subparser.add_argument("--receipts-dir", type=Path, required=True)
        subparser.add_argument("--artifact-root", type=Path, required=True)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--plan", type=Path, required=True)
        if name == "run":
            subparser.add_argument("--count", type=int, default=None)
            subparser.add_argument(
                "--source-manifest",
                type=Path,
                default=None,
                help=(
                    "Explicit sequence candidate manifest override. Defaults to the "
                    "artifact this stage declares as its input."
                ),
            )
        if name == "parse":
            subparser.add_argument("--stage", required=True)
            subparser.add_argument("--count", type=int, default=1)
            subparser.add_argument("--jsonl", action="append")
            subparser.add_argument("--fasta", action="append")
            subparser.add_argument("--pdb", action="append")
            # A supplied pose keeps the format the operator handed over, so the
            # profile that accepts either declares this glob instead of --pdb.
            subparser.add_argument("--structure", action="append")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "toolcheck":
            print("candidate normalizer ok, standard library only")
            return 0
        return run_stage(args) if args.command == "run" else parse_stage(args)
    except Exception as exc:
        print(f"candidate normalizer: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
