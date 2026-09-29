#!/usr/bin/env python3
"""Bind the deployed RFdiffusion3 fal client to the generator stage.

The client contract is the standard-library client in the rfprep lane. The
adapter turns its gzipped mmCIF designs into PDB poses because the
sequence-design adapters consume PDB files.

The client needs ``FAL_KEY`` in its own environment. This adapter reaches it by
whichever route the calling environment offers, which ``clients/fal_invocation``
decides.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from claude_binder.clients import fal_invocation
from claude_binder.adapters.candidate_lineage import backbone_lineage
from claude_binder.paths import package_file


DEFAULT_CLIENT_PYTHON = "python3"
DEFAULT_FAL_EXECUTABLE = "fal-credential-wrapper"
DEFAULT_CLIENT = package_file("clients", "fal_rfdiffusion3_client.py")
DEFAULT_GENERATOR_ID = "rfdiffusion"
DEFAULT_BINDER_CHAIN = "A"
DEFAULT_TARGET_CHAIN = "B"
DEFAULT_OUTPUT_SUBDIR = "rf3"
DEFAULT_MANIFEST_NAME = "candidate-manifest.jsonl"
DEFAULT_PARSER_RESULT_NAME = "parser-result.json"
DEFAULT_CHECKPOINT_NAME = "rfd3_foundry_2025_12_01_remapped.ckpt"
DEFAULT_CHECKPOINT_SHA256 = "9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1"
DESIGN_SUFFIX = ".cif.gz"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class AdapterError(RuntimeError):
    """A generator input or returned artifact is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AdapterError(f"refusing to write an empty candidate manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def resolve_client(value: Path | None) -> Path:
    environment_value = os.environ.get("CLAUDE_BINDER_RF3_CLIENT", "").strip()
    candidate = value or (Path(environment_value) if environment_value else DEFAULT_CLIENT)
    path = candidate.expanduser().resolve()
    if not path.is_file():
        raise AdapterError(f"RFdiffusion3 fal client is missing: {path}")
    return path


def run_external(argv: list[str], label: str) -> None:
    print(f"rfdiffusion3 adapter: {label}: {fal_invocation.redacted_command(argv)}", flush=True)
    completed = subprocess.run(argv, shell=False, check=False)
    if completed.returncode != 0:
        raise AdapterError(f"{label} exited {completed.returncode}")


def client_argv(
    args: argparse.Namespace,
    client: Path,
    command: str,
    *command_args: str,
) -> list[str]:
    """Build the client command on whichever credential route this machine offers."""
    try:
        client_values = [command, "--fal-url", args.fal_url, *command_args]
        credential_env = fal_invocation.credential_environment_key(
            getattr(args, "fal_credential_env", None)
        )
        route = fal_invocation.resolve_route(
            args.fal_executable,
            requested=getattr(args, "fal_credential_route", None),
            credential_env_key=credential_env,
        )
        if (
            route == fal_invocation.ROUTE_DIRECT
            and credential_env != fal_invocation.CREDENTIAL_ENVIRONMENT_KEY
        ):
            client_values.extend(["--credential-env", credential_env])
        return fal_invocation.client_command(
            args.fal_executable,
            args.client_python,
            client,
            client_values,
            requested=route,
            credential_env_key=credential_env,
        )
    except fal_invocation.RouteError as exc:
        raise AdapterError(str(exc)) from exc


def load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"{label} is invalid: {path}: {exc}") from exc


def load_target_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.target_manifest is not None:
        path = args.target_manifest.expanduser().resolve()
    else:
        receipt_path = args.receipts_dir.expanduser() / f"{args.target_stage_id}.json"
        receipt = load_json(receipt_path, "target stage receipt")
        artifacts = receipt.get("output_manifest", {}).get("artifacts", [])
        paths = [
            item.get("path")
            for artifact in artifacts
            if isinstance(artifact, dict)
            and artifact.get("artifact_id") == args.target_artifact_id
            for item in artifact.get("files", [])
            if isinstance(item, dict)
        ]
        if not paths or not isinstance(paths[0], str):
            raise AdapterError(
                f"target stage receipt carries no {args.target_artifact_id} file: {receipt_path}"
            )
        path = Path(paths[0]).expanduser().resolve()
    document = load_json(path, "target manifest")
    if not isinstance(document, dict):
        raise AdapterError(f"target manifest is not an object: {path}")
    required = (
        "target_id",
        "target_sha256",
        "residue_map_sha256",
        "source_structure_path",
        "normalized_structure_path",
    )
    missing = [field for field in required if not document.get(field)]
    if missing:
        raise AdapterError(f"target manifest {path} omits {', '.join(missing)}")
    source = Path(str(document["source_structure_path"])).expanduser()
    if not source.is_file():
        raise AdapterError(f"target structure is missing: {source}")
    if sha256_file(source) != str(document["target_sha256"]):
        raise AdapterError(f"target structure hash does not match the target manifest: {source}")
    return document, path


def cif_fields(line: str) -> list[str]:
    """Split one ordinary atom-site row from the RFdiffusion3 mmCIF."""
    return line.split()


def pdb_atom_name_field(atom: str, element: str) -> str:
    """Return the four-character PDB atom-name field for one atom.

    A one-character element symbol leaves column 13 blank and starts the name at column 14.
    A two-character element symbol starts at column 13. A four-character name fills the field.
    """
    name = atom[:4]
    if len(name) >= 4:
        return name
    if len(element.strip()) >= 2:
        return f"{name:<4}"
    return f" {name:<3}"


def format_pdb_atom_line(
    *,
    group: str,
    serial: int,
    atom: str,
    alt: str,
    residue: str,
    chain: str,
    residue_number: int,
    x: float,
    y: float,
    z: float,
    occupancy: float,
    b_factor: float,
    element: str,
) -> str:
    """Return one fixed-column PDB ATOM or HETATM line.

    The columns are not negotiable and nothing downstream forgives a shift. An earlier version
    emitted one extra space after the atom name, which pushed altLoc into the residue-name field
    and the chain ID out of column 22 entirely. Every reader then saw a blank chain, and the
    ProteinMPNN application answered HTTP 500 because the design chain it was asked for did not
    exist in the file.

    Record name 1 to 6, serial 7 to 11, atom name 13 to 16, altLoc 17, residue name 18 to 20,
    chain 22, residue sequence number 23 to 26, coordinates 31 to 54, occupancy 55 to 60,
    temperature factor 61 to 66, element 77 to 78.
    """
    return (
        f"{group:<6}{serial:5d} {pdb_atom_name_field(atom, element)}{alt[:1]:1}"
        f"{residue[:3]:>3} {chain[:1]:1}{residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_factor:6.2f}          {element[:2]:>2}"
    )


def cif_to_pdb(text: str, candidate_id: str) -> str:
    headers: list[str] = []
    rows: list[list[str]] = []
    in_atom_loop = False
    reading_rows = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == "loop_":
            headers = []
            rows = []
            in_atom_loop = False
            reading_rows = False
            continue
        if line.startswith("_atom_site."):
            headers.append(line.split()[0])
            in_atom_loop = True
            continue
        if not in_atom_loop:
            continue
        if line.startswith("#") or line == "loop_" or line.startswith("data_"):
            if rows:
                break
            continue
        if not headers:
            continue
        reading_rows = True
        fields = cif_fields(line)
        if len(fields) >= len(headers):
            rows.append(fields[: len(headers)])
        elif reading_rows:
            raise AdapterError(f"RFdiffusion3 design {candidate_id} has a malformed atom-site row")
    if not headers or not rows:
        raise AdapterError(f"RFdiffusion3 design {candidate_id} has no atom-site rows")

    index = {name: position for position, name in enumerate(headers)}

    def value(fields: list[str], *names: str, default: str = "?") -> str:
        for name in names:
            position = index.get(name)
            if position is not None and position < len(fields):
                return fields[position]
        return default

    output = [
        f"REMARK 900 DESIGN POSE {candidate_id}",
        "REMARK 900 RFdiffusion3 all-atom output; residue identities and side chains are its own",
        "REMARK 900 Design chain sequence is redesigned downstream, not assigned to a bare backbone",
    ]
    previous_chain: str | None = None
    serial = 0
    for fields in rows:
        group = value(fields, "_atom_site.group_PDB")
        if group not in {"ATOM", "HETATM"}:
            continue
        serial += 1
        atom = value(fields, "_atom_site.label_atom_id", "_atom_site.auth_atom_id")
        alt = value(fields, "_atom_site.label_alt_id", default=" ")
        # mmCIF writes "." for an inapplicable value and "?" for an unknown one. PDB wants a
        # blank altLoc column, and a literal dot there is read as an alternate conformer.
        if alt in {".", "?"}:
            alt = " "
        residue = value(fields, "_atom_site.auth_comp_id", "_atom_site.label_comp_id", default="UNK")
        chain = value(fields, "_atom_site.auth_asym_id", "_atom_site.label_asym_id", default="?")
        residue_number = value(fields, "_atom_site.auth_seq_id", "_atom_site.label_seq_id", default="1")
        x = value(fields, "_atom_site.Cartn_x", default="0")
        y = value(fields, "_atom_site.Cartn_y", default="0")
        z = value(fields, "_atom_site.Cartn_z", default="0")
        occupancy = value(fields, "_atom_site.occupancy", default="1.00")
        b_factor = value(fields, "_atom_site.B_iso_or_equiv", default="0.00")
        element = value(fields, "_atom_site.type_symbol", default=atom[:1])
        try:
            chain = chain[0]
            residue_number_int = int(residue_number)
            x_float, y_float, z_float = float(x), float(y), float(z)
            occupancy_float, b_factor_float = float(occupancy), float(b_factor)
        except (TypeError, ValueError, IndexError) as exc:
            raise AdapterError(
                f"RFdiffusion3 design {candidate_id} has invalid coordinates or residue numbering"
            ) from exc
        if previous_chain is not None and chain != previous_chain:
            output.append("TER")
        output.append(
            format_pdb_atom_line(
                group=group,
                serial=serial,
                atom=atom,
                alt=alt,
                residue=residue,
                chain=chain,
                residue_number=residue_number_int,
                x=x_float,
                y=y_float,
                z=z_float,
                occupancy=occupancy_float,
                b_factor=b_factor_float,
                element=element,
            )
        )
        previous_chain = chain
    if serial == 0:
        raise AdapterError(f"RFdiffusion3 design {candidate_id} has no ATOM or HETATM records")
    output.extend(["TER", "END"])
    return "\n".join(output) + "\n"


def client_design_paths(out_dir: Path, response: dict[str, Any]) -> list[Path]:
    designs = response.get("designs")
    if not isinstance(designs, list) or not designs:
        raise AdapterError("RFdiffusion3 response carries no designs")
    paths: list[Path] = []
    for design in designs:
        if not isinstance(design, dict):
            raise AdapterError("RFdiffusion3 response has a non-object design")
        name = design.get("name")
        if not isinstance(name, str) or SAFE_NAME.fullmatch(name) is None or not name.endswith(DESIGN_SUFFIX):
            raise AdapterError(f"RFdiffusion3 response has an unsafe design name: {name!r}")
        path = out_dir / name
        if not path.is_file():
            raise AdapterError(f"RFdiffusion3 client did not write returned design: {path}")
        encoded = design.get("gzip_b64")
        if not isinstance(encoded, str) or not encoded:
            raise AdapterError(f"RFdiffusion3 response carries no gzip_b64 for {name}")
        payload = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(payload).hexdigest() != str(design.get("sha256")):
            raise AdapterError(f"RFdiffusion3 response hash does not match design {name}")
        if path.read_bytes() != payload:
            raise AdapterError(f"RFdiffusion3 client output differs from the response for {name}")
        paths.append(path)
    return paths


def config_model_revision(config: Path, adapter_id: str = "rfdiffusion-generator") -> str:
    document = load_json(config, "resolved config")
    for adapter in document.get("adapters", []):
        if isinstance(adapter, dict) and adapter.get("adapter_id") == adapter_id:
            value = adapter.get("model_revision")
            if isinstance(value, str) and value:
                return value
    raise AdapterError(f"resolved config carries no model_revision for {adapter_id}")


def run(args: argparse.Namespace) -> int:
    if args.count < 1 or args.diffusion_batch_size < 1 or args.n_batches < 1:
        raise AdapterError("count, diffusion-batch-size, and n-batches must be positive")
    client = resolve_client(args.client)
    specification = args.specification.expanduser().resolve()
    # The manifest is this stage's declared input, and it is what target-prepare
    # normalized. Reading it here rather than after the client call puts its checks in
    # front of the spend, and gives the structure a source the plan actually guarantees.
    # An explicit --input-structure still wins, so a caller can override it by hand.
    manifest, manifest_path_source = load_target_manifest(args)
    if args.input_structure is not None:
        input_structure = args.input_structure.expanduser().resolve()
    else:
        input_structure = Path(str(manifest["normalized_structure_path"])).expanduser().resolve()
    if not specification.is_file():
        raise AdapterError(f"RFdiffusion3 specification is missing: {specification}")
    if not input_structure.is_file():
        raise AdapterError(f"RFdiffusion3 input structure is missing: {input_structure}")
    out_dir = args.out_dir.expanduser().resolve()
    receipt = args.receipt.expanduser().resolve()
    seed = args.seed
    if seed is None:
        seed = 0 if args.phase in {"smoke", "single"} else 1
    argv = client_argv(
        args,
        client,
        "run",
        "--specification",
        str(specification),
        "--input-structure",
        str(input_structure),
        "--out-dir",
        str(out_dir),
        "--receipt",
        str(receipt),
        "--diffusion-batch-size",
        str(args.diffusion_batch_size),
        "--n-batches",
        str(args.n_batches),
        "--seed",
        str(seed),
        "--step-scale",
        str(args.step_scale),
        "--gamma-0",
        str(args.gamma_0),
    )
    run_external(argv, "fal client")
    response = load_json(out_dir / "response.json", "RFdiffusion3 response")
    if not isinstance(response, dict):
        raise AdapterError("RFdiffusion3 response is not an object")
    client_design_paths(out_dir, response)
    print(f"rfdiffusion3 adapter: designs={len(response['designs'])} out_dir={out_dir}")
    return 0


def toolcheck(args: argparse.Namespace) -> int:
    client = resolve_client(args.client)
    out_dir = args.out_dir.expanduser().resolve()
    run_external(
        client_argv(args, client, "toolcheck", "--out-dir", str(out_dir)),
        "fal toolcheck",
    )
    return 0


def declared_record_count(path: Path, kind: str) -> int:
    """Count records in one declared output the way the executor counts them.

    The executor derives each file's record count from its declared ``kind`` and compares the
    sum against the parser's ``parsed_count``. Counting designs instead of records made a
    correct ten-backbone stage report 10 against an expected 20, and the run stopped.
    A PDB without MODEL delimiters is one pose, matching ``pdb_pose_count`` in the executor.
    """
    if kind == "jsonl":
        return sum(1 for line in path.read_text().splitlines() if line.strip())
    if kind == "json":
        return 1
    if kind == "pdb":
        poses = sum(1 for line in path.read_text(errors="replace").splitlines() if line.startswith("MODEL "))
        return poses if poses else 1
    raise AdapterError(f"declared output kind has no record rule in this parser: {kind}")


def declared_output_hashes(args: argparse.Namespace, phase_dir: Path) -> tuple[list[str], list[str], int]:
    """Hash the files this stage declares as outputs, not the client's scratch directory.

    The executor compares the parser's ``source_output_hashes`` against the declared stage
    outputs. Hashing the client output directory instead reports the raw tool files, which are
    a different set, so a stage that produced everything correctly still failed verification.
    ``target_prep_parser`` resolves the same way, by globbing each declared ``path_template``.
    """
    import glob as _glob

    hashes: list[str] = []
    problems: list[str] = []
    records = 0
    try:
        config = load_json(args.config.expanduser().resolve(), "resolved configuration")
    except Exception as exc:  # noqa: BLE001
        return [], [f"configuration is unreadable: {type(exc).__name__}: {exc}"], 0
    stage: dict[str, Any] = {}
    for candidate in config.get("stages", []):
        if isinstance(candidate, dict) and candidate.get("stage_id") == args.stage:
            stage = candidate
            break
    if not stage:
        return [], [f"stage is not present in config: {args.stage}"], 0
    attempt_dir = args.attempt_dir.expanduser().resolve()
    for output in stage.get("outputs", []):
        if not isinstance(output, dict) or "path_template" not in output:
            problems.append("stage output contract is not an object with a path_template")
            continue
        pattern = (
            str(output["path_template"])
            .replace("{{attempt_dir}}", str(attempt_dir))
            .replace("{{phase}}", str(args.phase))
            .replace("{{stage_id}}", str(args.stage))
        )
        for value in sorted(_glob.glob(pattern, recursive=True)):
            path = Path(value).resolve()
            if not path.is_file():
                continue
            if attempt_dir not in path.parents:
                problems.append(f"stage output escapes attempt directory: {path}")
                continue
            digest = sha256_file(path)
            if digest in hashes:
                continue
            hashes.append(digest)
            try:
                records += declared_record_count(path, str(output.get("kind", "")))
            except AdapterError as exc:
                problems.append(str(exc))
    return sorted(hashes), problems, records


def parse_outputs(args: argparse.Namespace) -> int:
    from . import binder_contract

    out_dir = args.out_dir.expanduser().resolve()
    phase_dir = args.attempt_dir.expanduser().resolve() / args.phase
    manifest_path = (
        args.manifest_path.expanduser().resolve()
        if args.manifest_path is not None
        else phase_dir / DEFAULT_MANIFEST_NAME
    )
    result_path = phase_dir / DEFAULT_PARSER_RESULT_NAME
    parsed = 0
    errors: list[str] = []
    try:
        response = load_json(out_dir / "response.json", "RFdiffusion3 response")
        if not isinstance(response, dict):
            raise AdapterError("RFdiffusion3 response is not an object")
        design_paths = client_design_paths(out_dir, response)
        manifest, manifest_path_source = load_target_manifest(args)
        model_revision = config_model_revision(args.config.expanduser().resolve())
        receipt = load_json(out_dir / "receipt.json", "RFdiffusion3 receipt")
        used_seed = receipt.get("used_seed")
        if not isinstance(used_seed, int) or isinstance(used_seed, bool):
            raise AdapterError("RFdiffusion3 receipt carries no integer used_seed")
        rows: list[dict[str, Any]] = []
        for index, raw_path in enumerate(design_paths):
            candidate_id = f"{args.generator_id}-{index:03d}"
            with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
                pose_text = cif_to_pdb(handle.read(), candidate_id)
            pose_path = phase_dir / args.pose_subdir / f"{candidate_id}.pdb"
            pose_path.parent.mkdir(parents=True, exist_ok=True)
            pose_path.write_text(pose_text)
            raw_sha256 = sha256_file(raw_path)
            pose_sha256 = sha256_file(pose_path)
            rows.append(
                {
                    "target_id": str(manifest["target_id"]),
                    "target_sha256": str(manifest["target_sha256"]),
                    "candidate_id": candidate_id,
                    "parent_candidate_id": None,
                    "origin_generator": args.generator_id,
                    **backbone_lineage(candidate_id, args.generator_id),
                    "generator_mode": "backbone-only",
                    "runner_protocol": "fal",
                    "sequence_designer": None,
                    "generator_seed": used_seed,
                    "requested_seed": receipt.get("requested_seed"),
                    "tool_seed": used_seed,
                    "sequence_path": None,
                    "sequence_sha256": None,
                    "sequence_length": None,
                    "backbone_only": True,
                    "structure_path": str(manifest["source_structure_path"]),
                    "structure_sha256": str(manifest["target_sha256"]),
                    "design_pose_path": str(pose_path.resolve()),
                    "design_pose_sha256": pose_sha256,
                    "residue_map_sha256": str(manifest["residue_map_sha256"]),
                    "optimization_round": 0,
                    "last_optimizer": None,
                    "status": "generated",
                    "design_index": index,
                    "binder_chain_id": args.binder_chain,
                    "target_chain_id": args.target_chain,
                    "tool_output_path": str(raw_path.resolve()),
                    "tool_output_sha256": raw_sha256,
                    "checkpoint_name": DEFAULT_CHECKPOINT_NAME,
                    "checkpoint_sha256": str(receipt.get("checkpoint_sha256", DEFAULT_CHECKPOINT_SHA256)),
                    "model_revision": model_revision,
                    "target_manifest_path": str(manifest_path_source),
                }
            )
            parsed += 1
        write_jsonl(manifest_path, rows)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")
    declared_hashes, declared_problems, declared_records = declared_output_hashes(args, phase_dir)
    errors.extend(declared_problems)
    # `parsed` counts designs. `parsed_count` in the parser-result contract counts records
    # across the stage's declared outputs, which the executor recomputes and compares. Ten
    # designs produce ten manifest rows plus ten pose files, so the two numbers differ.
    parsed_count = declared_records if parsed and not errors else parsed
    write_json(
        result_path,
        {
            "ok": bool(parsed) and not errors,
            "parsed_count": parsed_count,
            "rejected_count": len(errors),
            "errors": errors,
            "source_output_hashes": declared_hashes,
        },
    )
    # Without this the dispatcher logs a bare rc=1 and the reason sits in a file nobody
    # opens. A parse failure after a paid generation is the worst place to hide a message.
    for message in errors:
        print(f"rfdiffusion3 parse: {message}", file=sys.stderr)
    return 0 if parsed and not errors else 1


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--client", type=Path, default=None)
    parser.add_argument("--client-python", default=DEFAULT_CLIENT_PYTHON)
    parser.add_argument("--fal-executable", default=DEFAULT_FAL_EXECUTABLE)
    parser.add_argument("--fal-url", required=True)
    fal_invocation.add_route_argument(parser, executable=DEFAULT_FAL_EXECUTABLE)
    fal_invocation.add_credential_environment_argument(parser)


def add_stage_arguments(parser: argparse.ArgumentParser, *, run_command: bool = False) -> None:
    parser.add_argument("--stage", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    if run_command:
        parser.add_argument("--specification", type=Path, required=True)
        parser.add_argument("--input-structure", type=Path, default=None)
        # run resolves its input structure from the target stage's declared manifest, so it
        # needs the same three selectors the parse command uses to find that manifest.
        parser.add_argument("--target-manifest", type=Path, default=None)
        parser.add_argument("--target-stage-id", default="target-prepare")
        parser.add_argument("--target-artifact-id", default="target-manifest")
        parser.add_argument("--out-dir", type=Path, required=True)
        parser.add_argument("--receipt", type=Path, required=True)
        parser.add_argument("--diffusion-batch-size", type=int, required=True)
        parser.add_argument("--n-batches", type=int, required=True)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--step-scale", type=float, required=True)
        parser.add_argument("--gamma-0", type=float, required=True)
    else:
        parser.add_argument("--out-dir", type=Path, required=True)
        parser.add_argument("--target-manifest", type=Path, default=None)
        parser.add_argument("--target-stage-id", default="target-prepare")
        parser.add_argument("--target-artifact-id", default="target-manifest")
        parser.add_argument("--manifest-path", type=Path, default=None)
        parser.add_argument("--generator-id", default=DEFAULT_GENERATOR_ID)
        parser.add_argument("--binder-chain", default=DEFAULT_BINDER_CHAIN)
        parser.add_argument("--target-chain", default=DEFAULT_TARGET_CHAIN)
        parser.add_argument("--pose-subdir", default="poses")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    toolcheck_parser = subparsers.add_parser("toolcheck")
    add_common_arguments(toolcheck_parser)
    toolcheck_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    add_common_arguments(run_parser)
    add_stage_arguments(run_parser, run_command=True)
    parse_parser = subparsers.add_parser("parse")
    add_common_arguments(parse_parser)
    add_stage_arguments(parse_parser)
    return parser


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        if args.command == "toolcheck":
            return toolcheck(args)
        if args.command == "parse":
            return parse_outputs(args)
        return run(args)
    except AdapterError as exc:
        print(f"rfdiffusion3 adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
