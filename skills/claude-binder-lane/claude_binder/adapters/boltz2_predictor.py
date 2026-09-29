#!/usr/bin/env python3
"""Bind the hosted Boltz-2 structure-and-binding API to cofold stages.

Boltz exposes sample multiplicity rather than a seed parameter. This adapter
records the zero-based ``all_sample_results`` index in the contract's integer
``seed`` field and adds ``seed_semantics`` so the distinction stays visible.
"""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

from . import esmfold2_predictor as base


PREDICTOR_ID = "boltz"
ADAPTER_ID = "boltz-predictor"
ROUTE_ID = "boltz-cloud-cli"
ROUTE_CONTRACT_REVISION = "structure-and-binding-v1"
MODEL_LITERAL = "boltz-2.1"
MODEL_RUNTIME_VERSION = "v2026-03-01"
MODEL_REVISION = f"{MODEL_LITERAL}@{MODEL_RUNTIME_VERSION}"
DEFAULT_BOLTZ_EXECUTABLE = "boltz-api"
DEFAULT_WORK_SUBDIR = "boltz2"
DEFAULT_POLL_SECONDS = 5
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180
TARGET_MSA_MANIFEST_ARTIFACT = "target-msa-manifest"
SEED_SEMANTICS = "boltz_sample_index"
FAILURE_PREFLIGHT = "preflight_failed"
FAILURE_SUBPROCESS = "boltz_subprocess_failed"
FAILURE_TIMEOUT = "boltz_timeout"
FAILURE_RESPONSE = "boltz_response_invalid"
FAILURE_ARCHIVE = "boltz_archive_invalid"
FAILURE_SAMPLE_COUNT = "unexpected_sample_count"
FAILURE_ARTIFACT = "artifact_write_failed"


class AdapterError(RuntimeError):
    """A static input, API response, or returned artifact is invalid."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"{label} is invalid: {path}: {exc}") from exc


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"{label} line {line_number} is invalid: {exc}") from exc
        if not isinstance(value, dict):
            raise AdapterError(f"{label} line {line_number} is not an object")
        rows.append(value)
    return rows


def safe_part(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def run_phase_for_record(args: argparse.Namespace) -> str:
    return str(args.phase)


def executable_available(executable: str) -> bool:
    candidate = Path(executable).expanduser()
    return candidate.is_file() and os.access(candidate, os.X_OK) or shutil.which(executable) is not None


def run_command(argv: list[str], *, label: str, timeout: int | None = None) -> str:
    print(f"{PREDICTOR_ID}: {label}: {' '.join(argv)}", file=sys.stderr, flush=True)
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterError(f"{label} exceeded {timeout} seconds") from exc
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, flush=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise AdapterError(
            f"{label} exited {completed.returncode}: {detail[-1] if detail else 'no output'}"
        )
    return completed.stdout


def json_from_stdout(stdout: str, label: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text.startswith("{"):
        raise AdapterError(f"{label} did not return a JSON object")
    try:
        value = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"{label} returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"{label} returned a JSON {type(value).__name__}, not an object")
    return value


def response_chain_mapping(
    response: dict[str, Any],
    target_sequence: str,
    binder_sequence: str,
    expected_mapping: dict[str, str],
) -> dict[str, str]:
    """Read submitted chain roles from the response and check both sequences."""
    input_value = response.get("input")
    entities = input_value.get("entities") if isinstance(input_value, dict) else None
    if not isinstance(entities, list) or len(entities) != 2:
        raise AdapterError("Boltz response input must echo exactly two protein entities")

    target_matches: list[dict[str, Any]] = []
    binder_matches: list[dict[str, Any]] = []
    for entity in entities:
        if not isinstance(entity, dict) or entity.get("type") != "protein":
            raise AdapterError("Boltz response input contains a non-protein entity")
        chain_ids = entity.get("chain_ids")
        value = entity.get("value")
        if not isinstance(chain_ids, list) or len(chain_ids) != 1 or not isinstance(chain_ids[0], str):
            raise AdapterError("Boltz response protein entities must carry one chain id")
        if not isinstance(value, str):
            raise AdapterError("Boltz response protein entities must carry a sequence")
        sequence = "".join(value.split()).upper()
        if sequence == target_sequence:
            target_matches.append(entity)
        if sequence == binder_sequence:
            binder_matches.append(entity)

    if len(target_matches) != 1 or len(binder_matches) != 1:
        raise AdapterError(
            "Boltz response input does not identify one target and one binder by exact sequence"
        )
    mapping = {
        "target": str(target_matches[0]["chain_ids"][0]),
        "binder": str(binder_matches[0]["chain_ids"][0]),
    }
    if mapping != expected_mapping:
        raise AdapterError(
            f"Boltz response chain mapping {mapping} differs from the campaign mapping "
            f"{expected_mapping}"
        )
    return mapping


def verify_structure_chain_mapping(
    structure_path: Path,
    target_sequence: str,
    binder_sequence: str,
    response_mapping: dict[str, str],
) -> None:
    """Derive roles from each returned structure and compare them with the response."""
    from . import binder_contract

    derived = binder_contract.derive_chain_mapping(
        structure_path,
        target_sequence,
        binder_sequence,
        structure_label="Boltz predicted structure",
    )
    if derived != response_mapping:
        raise AdapterError(
            f"Boltz predicted structure mapping {derived} differs from response mapping "
            f"{response_mapping}: {structure_path}"
        )


def sample_labels(seeds: list[int]) -> list[int]:
    """Return honest sample labels for the contract's integer seed field."""
    if not 1 <= len(seeds) <= 10:
        raise AdapterError(
            f"Boltz exposes one to ten samples per request, and the stage configured {len(seeds)}"
        )
    expected = list(range(len(seeds)))
    if seeds != expected:
        raise AdapterError(
            "Boltz sample labels require configured seeds 0..N-1 because the API exposes "
            f"sample indexes, and the stage configured {seeds}"
        )
    return expected


def target_msa_query(path: Path) -> str:
    if not path.is_file():
        raise AdapterError(f"target MSA is missing: {path}")
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith(">") and index + 1 < len(lines):
            query = "".join(lines[index + 1].split()).upper()
            if query:
                return query.replace("-", "")
    raise AdapterError(f"target MSA has no query sequence: {path}")


def encode_target_msa(path: Path, target_sequence: str) -> tuple[str, str]:
    query = target_msa_query(path)
    if query != target_sequence:
        raise AdapterError(
            f"target MSA query does not match target sequence: {path}; "
            f"query length {len(query)}, target length {len(target_sequence)}"
        )
    raw = path.read_bytes()
    return base64.b64encode(raw).decode("ascii"), sha256_file(path)


def build_request(
    *,
    target_sequence: str,
    target_chain: str,
    binder_sequence: str,
    binder_chain: str,
    target_msa_b64: str,
    num_samples: int,
) -> dict[str, Any]:
    if not 1 <= num_samples <= 10:
        raise AdapterError(f"Boltz num_samples must be between 1 and 10, and is {num_samples}")
    return {
        "entities": [
            {
                "type": "protein",
                "chain_ids": [target_chain],
                "value": target_sequence,
                "msa": {
                    "type": "custom",
                    "format": "a3m",
                    "source": {
                        "type": "base64",
                        "data": target_msa_b64,
                        "media_type": "text/x-a3m",
                    },
                },
            },
            {
                "type": "protein",
                "chain_ids": [binder_chain],
                "value": binder_sequence,
                "msa": {"type": "empty"},
            },
        ],
        "binding": {
            "type": "protein_protein_binding",
            "binder_chain_ids": [binder_chain],
        },
        "num_samples": num_samples,
        "model_options": {"recycling_steps": 3, "sampling_steps": 200},
    }


def idempotency_key(request_path: Path) -> str:
    """Bind a provider retry key to this route contract and request bytes."""
    digest = sha256_file(request_path)
    return f"claude-binder-{ROUTE_CONTRACT_REVISION}-{digest}"


def start_argv(
    executable: str,
    request_path: Path,
    request_idempotency_key: str,
) -> list[str]:
    return [
        executable,
        "predictions:structure-and-binding",
        "start",
        "--model",
        MODEL_LITERAL,
        "--input",
        f"@json://{request_path}",
        "--idempotency-key",
        request_idempotency_key,
    ]


def existing_submission(path: Path, *, request_sha256: str) -> dict[str, Any] | None:
    """Return a matching durable acceptance record for attach-only recovery."""
    if not path.exists():
        return None
    value = load_json(path, "Boltz submission record")
    if not isinstance(value, dict):
        raise AdapterError("Boltz submission record is not an object")
    expected = {
        "route_id": ROUTE_ID,
        "route_contract_revision": ROUTE_CONTRACT_REVISION,
        "request_sha256": request_sha256,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise AdapterError(
                f"Boltz submission record {key} does not match this request"
            )
    prediction_id = value.get("prediction_id")
    if not isinstance(prediction_id, str) or not prediction_id:
        raise AdapterError("Boltz submission record has no prediction id")
    return value


def retrieve_argv(executable: str, prediction_id: str) -> list[str]:
    return [
        executable,
        "predictions:structure-and-binding",
        "retrieve",
        "--id",
        prediction_id,
    ]


def wait_for_prediction(
    executable: str,
    prediction_id: str,
    *,
    poll_seconds: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_status: str | None = None
    while time.monotonic() < deadline:
        stdout = run_command(
            retrieve_argv(executable, prediction_id),
            label=f"retrieve {prediction_id}",
            timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        response = json_from_stdout(stdout, "Boltz retrieve")
        status = response.get("status")
        if status != last_status:
            print(f"{PREDICTOR_ID}: {prediction_id}: status={status}", file=sys.stderr, flush=True)
            last_status = str(status)
        if status in {"succeeded", "failed", "cancelled", "expired"}:
            return response
        time.sleep(poll_seconds)
    raise AdapterError(f"Boltz prediction {prediction_id} exceeded {timeout_seconds} seconds")


def download_archive(response: dict[str, Any], destination: Path) -> Path:
    output = response.get("output")
    archive = output.get("archive") if isinstance(output, dict) else None
    url = archive.get("url") if isinstance(archive, dict) else None
    if not isinstance(url, str) or not url:
        raise AdapterError("Boltz response output.archive.url is missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib_request.urlopen(urllib_request.Request(url), timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS) as handle:
            destination.write_bytes(handle.read())
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"Boltz archive download failed: {exc}") from exc
    if not destination.stat().st_size:
        raise AdapterError(f"Boltz archive is empty: {destination}")
    return destination


def _safe_member_path(root: Path, member_name: str) -> Path:
    relative = Path(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise AdapterError(f"Boltz archive member escapes its output directory: {member_name}")
    destination = (root / relative).resolve()
    if root.resolve() not in destination.parents:
        raise AdapterError(f"Boltz archive member escapes its output directory: {member_name}")
    return destination


def extract_archive(archive_path: Path, destination: Path, sample_count: int) -> list[dict[str, str]]:
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            structures = {
                index: member
                for index in range(sample_count)
                for member in members
                if Path(member.name).name == f"sample_{index}_predicted_structure.cif"
            }
            paes = {
                index: member
                for index in range(sample_count)
                for member in members
                if Path(member.name).name == f"sample_{index}_pae.npz"
            }
            missing = [
                f"sample_{index}"
                for index in range(sample_count)
                if index not in structures or index not in paes
            ]
            if missing:
                raise AdapterError(
                    "Boltz archive lacks structure or PAE members for " + ", ".join(missing)
                )
            samples: list[dict[str, str]] = []
            for index in range(sample_count):
                sample_dir = destination / f"sample_{index}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                structure_destination = _safe_member_path(
                    sample_dir, f"sample_{index}_predicted_structure.cif"
                )
                pae_destination = _safe_member_path(sample_dir, f"sample_{index}_pae.npz")
                for member, output_path in (
                    (structures[index], structure_destination),
                    (paes[index], pae_destination),
                ):
                    source = archive.extractfile(member)
                    if source is None:
                        raise AdapterError(f"Boltz archive member cannot be read: {member.name}")
                    output_path.write_bytes(source.read())
                samples.append(
                    {"structure_path": str(structure_destination), "pae_path": str(pae_destination)}
                )
            return samples
    except AdapterError:
        raise
    except (tarfile.TarError, OSError) as exc:
        raise AdapterError(f"Boltz archive is not a readable gzipped tar: {archive_path}: {exc}") from exc


def pae_matrix(path: Path) -> list[list[float]]:
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            if "pae" not in data.files:
                raise AdapterError(f"Boltz PAE archive has no pae array: {path}")
            matrix = np.asarray(data["pae"], dtype=float)
    except AdapterError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"Boltz PAE file is invalid: {path}: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise AdapterError(f"Boltz PAE matrix is not non-empty and square: {path}: {matrix.shape}")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise AdapterError(f"Boltz PAE matrix has a non-finite or negative value: {path}")
    return matrix.tolist()


def validate_response(response: dict[str, Any], model_revision: str) -> None:
    if response.get("status") != "succeeded":
        raise AdapterError(f"Boltz prediction status is {response.get('status')!r}")
    if response.get("model") != MODEL_LITERAL:
        raise AdapterError(f"Boltz response model is {response.get('model')!r}, expected {MODEL_LITERAL!r}")
    if response.get("version") != MODEL_RUNTIME_VERSION:
        raise AdapterError(
            f"Boltz response runtime version is {response.get('version')!r}, "
            f"expected {MODEL_RUNTIME_VERSION!r}"
        )
    if model_revision != MODEL_REVISION:
        raise AdapterError(
            f"adapter model_revision is {model_revision!r}, expected {MODEL_REVISION!r}"
        )
    output = response.get("output")
    samples = output.get("all_sample_results") if isinstance(output, dict) else None
    if not isinstance(samples, list):
        raise AdapterError("Boltz response output.all_sample_results is missing")


def preflight_item(
    config: dict[str, Any],
    args: argparse.Namespace,
    item: dict[str, Any],
    supplied_targets: dict[str, str],
    supplied_msas: dict[str, str],
    supplied_hotspots: dict[str, str],
    controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate = item["candidate"]
    target = item["target"]
    candidate_id = str(candidate.get("candidate_id"))
    target_id = str(target.get("target_id"))
    target_sequence = base.resolve_target_sequence(target_id, supplied_targets)
    sequence_path = candidate.get("sequence_path")
    if not isinstance(sequence_path, str) or not sequence_path:
        raise AdapterError(f"candidate {candidate_id} carries no sequence_path")
    binder_sequence = base.read_fasta_sequence(Path(sequence_path))
    pose_value = candidate.get("design_pose_path")
    pose_hash = candidate.get("design_pose_sha256")
    if not isinstance(pose_value, str) or not pose_value or not isinstance(pose_hash, str) or not pose_hash:
        raise AdapterError(f"candidate {candidate_id} carries no complete design pose identity")
    pose_path = Path(pose_value).expanduser().resolve()
    if not pose_path.is_file():
        raise AdapterError(f"candidate {candidate_id} design pose is missing: {pose_path}")
    observed_hash = sha256_file(pose_path)
    if observed_hash != pose_hash:
        raise AdapterError(
            f"candidate {candidate_id} design pose is stale: manifest has {pose_hash}, file has {observed_hash}"
        )
    if target_id not in supplied_msas:
        raise AdapterError(f"no target MSA supplied for {target_id}")
    msa_path = Path(supplied_msas[target_id]).expanduser().resolve()
    msa_b64, msa_sha256 = encode_target_msa(msa_path, target_sequence)
    site_map = base.site_residue_map_for(config, target, supplied_hotspots)
    mapping = base.chain_mapping_for(config, target, candidate_id, controls)
    return {
        "target_sequence": target_sequence,
        "binder_sequence": binder_sequence,
        "target": target,
        "msa_path": str(msa_path),
        "msa_sha256": msa_sha256,
        "msa_b64": msa_b64,
        "site_map": site_map,
        "chain_mapping": mapping,
    }


def build_plan(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    return base.plan_predictions(
        config,
        stage_id=args.stage,
        row_phase=base.campaign_phase(args.stage),
        artifact_root=args.artifact_root,
        count=args.count,
        predictor_id=PREDICTOR_ID,
    )


def group_plan(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for item in plan:
        key = (str(item["target"]["target_id"]), str(item["candidate"]["candidate_id"]))
        if key not in groups:
            groups[key] = {"target": item["target"], "candidate": item["candidate"], "items": []}
            order.append(key)
        groups[key]["items"].append(item)
    return [groups[key] for key in order]


def index_metadata(setup: dict[str, Any]) -> dict[str, Any]:
    """Persist every parse-time input resolved before the paid request."""
    return {
        "target_sequence": setup["target_sequence"],
        "binder_sequence": setup["binder_sequence"],
        "msa_path": setup["msa_path"],
        "msa_sha256": setup["msa_sha256"],
        "site_map": setup["site_map"],
        "chain_mapping": setup["chain_mapping"],
    }


def setup_from_index(
    config: dict[str, Any],
    item: dict[str, Any],
    record: dict[str, Any],
    controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Recover the preflight snapshot without requiring parser-only CLI values."""
    candidate = item["candidate"]
    target = item["target"]
    target_id = str(target["target_id"])
    candidate_id = str(candidate["candidate_id"])
    target_sequence = record.get("target_sequence")
    binder_sequence = record.get("binder_sequence")
    msa_path = record.get("msa_path")
    msa_sha256 = record.get("msa_sha256")
    site_map = record.get("site_map")
    response_mapping = record.get("chain_mapping")
    if not isinstance(target_sequence, str) or not target_sequence:
        raise AdapterError(f"run index has no target sequence for {target_id}/{candidate_id}")
    if not isinstance(binder_sequence, str) or not binder_sequence:
        raise AdapterError(f"run index has no binder sequence for {target_id}/{candidate_id}")
    if not isinstance(msa_path, str) or not msa_path or not isinstance(msa_sha256, str) or not msa_sha256:
        raise AdapterError(f"run index has no target MSA identity for {target_id}/{candidate_id}")
    if not isinstance(site_map, dict):
        raise AdapterError(f"run index has no site map for {target_id}/{candidate_id}")
    expected_mapping = base.chain_mapping_for(config, target, candidate_id, controls)
    if response_mapping != expected_mapping:
        raise AdapterError(
            f"run index chain mapping {response_mapping!r} differs from campaign mapping "
            f"{expected_mapping!r} for {target_id}/{candidate_id}"
        )
    observed_msa_hash = sha256_file(Path(msa_path).expanduser().resolve())
    if observed_msa_hash != msa_sha256:
        raise AdapterError(
            f"run index target MSA is stale for {target_id}: "
            f"index has {msa_sha256}, file has {observed_msa_hash}"
        )
    return {
        "target_sequence": target_sequence,
        "binder_sequence": binder_sequence,
        "target": target,
        "msa_path": str(Path(msa_path).expanduser().resolve()),
        "msa_sha256": msa_sha256,
        "site_map": site_map,
        "chain_mapping": expected_mapping,
    }


def run(args: argparse.Namespace) -> int:
    from . import binder_contract

    config = base.load_json(args.config)
    plan = build_plan(config, args)
    controls = base.control_records(config)
    target_values = base.resolve_per_target(args.target_sequence, config["targets"], "--target-sequence")
    msa_values = base.resolve_per_target(args.target_msa_a3m, config["targets"], "--target-msa-a3m")
    hotspot_values = base.resolve_per_target(args.hotspot_residues, config["targets"], "--hotspot-residues")
    model_revision = base.model_revision_for(config, ADAPTER_ID)
    preflight: dict[tuple[str, str], dict[str, Any]] = {}
    for item in plan:
        key = (str(item["target"]["target_id"]), str(item["candidate"]["candidate_id"]))
        if key in preflight:
            continue
        preflight[key] = preflight_item(
            config, args, item, target_values, msa_values, hotspot_values, controls
        )

    labels_by_key: dict[tuple[str, str], list[int]] = {}
    for group in group_plan(plan):
        key = (str(group["target"]["target_id"]), str(group["candidate"]["candidate_id"]))
        labels_by_key[key] = sample_labels([int(item["seed"]) for item in group["items"]])

    client = args.boltz_executable
    if not executable_available(client):
        raise AdapterError(f"Boltz executable is not available: {client}")
    work_root = args.attempt_dir.expanduser().resolve() / args.phase / args.work_subdir
    index_rows: list[dict[str, Any]] = []
    for group in group_plan(plan):
        target = group["target"]
        candidate = group["candidate"]
        target_id = str(target["target_id"])
        candidate_id = str(candidate["candidate_id"])
        key = (target_id, candidate_id)
        items = group["items"]
        seeds = [int(item["seed"]) for item in items]
        try:
            labels = labels_by_key[key]
            setup = preflight[key]
            group_dir = work_root / safe_part(target_id) / safe_part(candidate_id)
            request_path = group_dir / "request.json"
            submission_path = group_dir / "submission.json"
            response_path = group_dir / "response.json"
            archive_path = group_dir / "prediction_archive.tar.gz"
            samples_dir = group_dir / "samples"
            request = build_request(
                target_sequence=setup["target_sequence"],
                target_chain=setup["chain_mapping"]["target"],
                binder_sequence=setup["binder_sequence"],
                binder_chain=setup["chain_mapping"]["binder"],
                target_msa_b64=setup["msa_b64"],
                num_samples=len(labels),
            )
            write_json(request_path, request)
            request_sha256 = sha256_file(request_path)
            request_key = idempotency_key(request_path)
            submission = existing_submission(
                submission_path,
                request_sha256=request_sha256,
            )
            if submission is None:
                started = json_from_stdout(
                    run_command(
                        start_argv(client, request_path, request_key),
                        label=f"start {target_id}/{candidate_id}",
                        timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
                    ),
                    "Boltz start",
                )
                prediction_id = started.get("id")
                if not isinstance(prediction_id, str) or not prediction_id:
                    raise AdapterError("Boltz start response has no prediction id")
                submission = {
                    "route_id": ROUTE_ID,
                    "route_contract_revision": ROUTE_CONTRACT_REVISION,
                    "request_sha256": request_sha256,
                    "idempotency_key": request_key,
                    "prediction_id": prediction_id,
                    "provider_status": started.get("status"),
                }
                write_json(submission_path, submission)
            else:
                prediction_id = str(submission["prediction_id"])
            response = wait_for_prediction(
                client,
                prediction_id,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            )
            write_json(response_path, response)
            validate_response(response, model_revision)
            response_mapping = response_chain_mapping(
                response,
                setup["target_sequence"],
                setup["binder_sequence"],
                setup["chain_mapping"],
            )
            sample_count = len(response["output"]["all_sample_results"])
            if sample_count != len(labels):
                raise AdapterError(
                    f"Boltz returned {sample_count} samples, expected {len(labels)}"
                )
            download_archive(response, archive_path)
            sample_paths = extract_archive(archive_path, samples_dir, sample_count)
            index_rows.append(
                {
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    **index_metadata(setup),
                    "prediction_id": prediction_id,
                    "route_id": ROUTE_ID,
                    "route_contract_revision": ROUTE_CONTRACT_REVISION,
                    "idempotency_key": request_key,
                    "submission_path": str(submission_path),
                    "request_path": str(request_path),
                    "response_path": str(response_path),
                    "archive_path": str(archive_path),
                    "samples": sample_paths,
                    "chain_mapping": response_mapping,
                    "sample_count": sample_count,
                    "status": "completed",
                }
            )
        except Exception as exc:  # noqa: BLE001
            failure = str(exc)[:500]
            print(f"{PREDICTOR_ID}: {target_id}/{candidate_id}: {failure}", file=sys.stderr)
            index_rows.append(
                {
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    **index_metadata(preflight[key]),
                    "route_id": ROUTE_ID,
                    "route_contract_revision": ROUTE_CONTRACT_REVISION,
                    "sample_count": len(seeds),
                    "status": "failed",
                    "error": failure,
                }
            )
    write_jsonl(args.run_index.expanduser().resolve(), index_rows)
    completed = sum(row.get("status") == "completed" for row in index_rows)
    print(f"{PREDICTOR_ID}: completed {completed} of {len(index_rows)} API requests", file=sys.stderr)
    return 0 if completed else 1


def declared_output_files(config: dict[str, Any], stage_id: str, attempt_dir: Path, phase: str) -> list[Path]:
    stage = base.stage_record(config, stage_id)
    files: list[Path] = []
    for output in stage.get("outputs", []):
        pattern = base.render(output["path_template"], attempt_dir=attempt_dir, phase=phase)
        files.extend(Path(value) for value in glob.glob(pattern, recursive=True) if Path(value).is_file())
    return sorted(path.resolve() for path in files)


def parser_result_from_declared_outputs(
    config: dict[str, Any], stage_id: str, attempt_dir: Path, phase: str, errors: list[str]
) -> dict[str, Any]:
    files = declared_output_files(config, stage_id, attempt_dir, phase)
    parsed_count = 0
    count_errors = list(errors)
    for path in files:
        try:
            parsed_count += len(load_jsonl(path, f"declared stage output {path}"))
        except AdapterError as exc:
            count_errors.append(str(exc))
    result = {
        "ok": bool(files) and parsed_count > 0 and not count_errors,
        "parsed_count": parsed_count,
        "rejected_count": len(count_errors),
        "errors": count_errors,
        "source_output_hashes": sorted(sha256_file(path) for path in files),
    }
    return result


def parse_outputs(args: argparse.Namespace) -> int:
    from . import binder_contract

    config = base.load_json(args.config)
    row_phase = base.campaign_phase(args.stage)
    controls = base.control_records(config)
    model_revision = base.model_revision_for(config, ADAPTER_ID)
    manifest_path, artifacts_attempt_dir = base.output_paths(
        config, args.stage, args.attempt_dir.expanduser().resolve(), args.phase
    )
    run_index = load_jsonl(args.run_index.expanduser().resolve(), "Boltz run index")
    plan = build_plan(config, args)
    plan_by_key = {
        (str(item["target"]["target_id"]), str(item["candidate"]["candidate_id"]), int(item["seed"])): item
        for item in plan
    }
    writer = base.RowWriter(manifest_path)
    errors: list[str] = []
    for record in run_index:
        target_id = str(record.get("target_id"))
        candidate_id = str(record.get("candidate_id"))
        records = [
            (seed, item)
            for (planned_target, planned_candidate, seed), item in plan_by_key.items()
            if planned_target == target_id and planned_candidate == candidate_id
        ]
        if not records:
            errors.append(f"run index has an unplanned request: {(target_id, candidate_id)}")
            continue
        setup = setup_from_index(config, records[0][1], record, controls)
        for seed, item in records:
            row = base.base_row(
                config,
                item=item,
                row_phase=row_phase,
                model_revision=model_revision,
                controls=controls,
                msa_identity={"msa_path": setup["msa_path"], "msa_sha256": setup["msa_sha256"]},
            )
            row["seed_semantics"] = SEED_SEMANTICS
            row["seed_source"] = "Boltz output.all_sample_results index"
            row["route_id"] = ROUTE_ID
            row["route_contract_revision"] = ROUTE_CONTRACT_REVISION
            if record.get("status") != "completed":
                writer.write(
                    base.failed_row(
                        binder_contract,
                        row,
                        failure_code=FAILURE_SUBPROCESS,
                        failure_reason=str(record.get("error", "Boltz request failed"))[:500],
                    )
                )
                continue
            try:
                samples = record.get("samples")
                if not isinstance(samples, list) or seed >= len(samples):
                    raise AdapterError(f"run index has no sample {seed} for {(target_id, candidate_id)}")
                sample = samples[seed]
                structure_path = Path(str(sample["structure_path"])).expanduser().resolve()
                pae_path = Path(str(sample["pae_path"])).expanduser().resolve()
                response = load_json(Path(str(record["response_path"])), "Boltz response")
                response_mapping = response_chain_mapping(
                    response,
                    setup["target_sequence"],
                    setup["binder_sequence"],
                    row["chain_mapping"],
                )
                verify_structure_chain_mapping(
                    structure_path,
                    setup["target_sequence"],
                    setup["binder_sequence"],
                    response_mapping,
                )
                metrics = response["output"]["all_sample_results"][seed].get("metrics", {})
                written = binder_contract.write_prediction_artifacts(
                    attempt_dir=artifacts_attempt_dir,
                    phase=row["phase"],
                    run_phase=run_phase_for_record(args),
                    target_id=row["target_id"],
                    candidate_id=row["candidate_id"],
                    predictor=row["predictor"],
                    seed=row["seed"],
                    complex_cif=structure_path,
                    pae=pae_matrix(pae_path),
                    chain_mapping=response_mapping,
                    reference_cif=Path(row["design_pose_path"]),
                    site_residue_map=setup["site_map"],
                    model_revision=model_revision,
                    target_sequence=setup["target_sequence"],
                    binder_sequence=setup["binder_sequence"],
                    extra={
                        "target_sha256": row["target_sha256"],
                        "sequence_sha256": row["sequence_sha256"],
                        "design_pose_sha256": row["design_pose_sha256"],
                        "iptm": metrics.get("iptm"),
                        "ptm": metrics.get("ptm"),
                        "mean_plddt": metrics.get("complex_plddt"),
                        "boltz_binding_confidence": response.get("output", {})
                        .get("binding_metrics", {})
                        .get("binding_confidence"),
                    },
                )
                merged = dict(row)
                merged.update(written)
                writer.write(merged)
            except Exception as exc:  # noqa: BLE001
                writer.write(
                    base.failed_row(
                        binder_contract,
                        row,
                        failure_code=getattr(exc, "failure_code", FAILURE_ARTIFACT),
                        failure_reason=f"{type(exc).__name__}: {exc}"[:500],
                    )
                )
    writer.close()
    if writer.count > 0 and writer.failed == writer.count:
        errors.append("all planned Boltz samples were rejected during parsing")
    result = parser_result_from_declared_outputs(
        config, args.stage, args.attempt_dir.expanduser().resolve(), args.phase, errors
    )
    result_path = args.attempt_dir.expanduser().resolve() / args.phase / "parser-result.json"
    write_json(result_path, result)
    print(
        f"{PREDICTOR_ID}: parsed {result['parsed_count']} records from declared outputs; "
        f"source_files={len(result['source_output_hashes'])} ok={result['ok']}",
        file=sys.stderr,
    )
    return 0 if result["ok"] else 1


def toolcheck(args: argparse.Namespace) -> int:
    if not executable_available(args.boltz_executable):
        raise AdapterError(f"Boltz executable is not available: {args.boltz_executable}")
    from . import binder_contract

    drift = binder_contract.executor_field_drift()
    if drift:
        raise AdapterError("executor and binder_contract fields differ: " + "; ".join(drift))
    print(f"{PREDICTOR_ID}: local toolcheck passed for {args.boltz_executable}")
    return 0


def add_stage_arguments(parser: argparse.ArgumentParser, *, run_command: bool) -> None:
    for name, kwargs in (
        ("--stage", {"required": True}),
        ("--phase", {"required": True}),
        ("--count", {"type": int, "default": 1}),
        ("--attempt-dir", {"type": Path, "required": True}),
        ("--receipts-dir", {"type": Path, "required": True}),
        ("--artifact-root", {"type": Path, "required": True}),
        ("--config", {"type": Path, "required": True}),
        ("--plan", {"type": Path, "required": True}),
        ("--run-index", {"type": Path, "required": True}),
    ):
        parser.add_argument(name, **kwargs)
    parser.add_argument("--target-sequence", action="append")
    parser.add_argument("--target-msa-a3m", action="append")
    parser.add_argument("--hotspot-residues", action="append")
    if run_command:
        parser.add_argument("--boltz-executable", default=DEFAULT_BOLTZ_EXECUTABLE)
        parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
        parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
        parser.add_argument("--work-subdir", default=DEFAULT_WORK_SUBDIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    toolcheck_parser = subparsers.add_parser("toolcheck")
    toolcheck_parser.add_argument("--boltz-executable", default=DEFAULT_BOLTZ_EXECUTABLE)
    run_parser = subparsers.add_parser("run")
    add_stage_arguments(run_parser, run_command=True)
    parse_parser = subparsers.add_parser("parse")
    add_stage_arguments(parse_parser, run_command=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "toolcheck":
            return toolcheck(args)
        if args.command == "parse":
            return parse_outputs(args)
        return run(args)
    except AdapterError as exc:
        print(f"{PREDICTOR_ID}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
