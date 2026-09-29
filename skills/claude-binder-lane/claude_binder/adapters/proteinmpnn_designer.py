#!/usr/bin/env python3
"""Design ProteinMPNN sequences for one binder lane stage phase.

SolubleMPNN ships inside the same repository and runs on the same runner, so one
wrapper serves both the `proteinmpnn-designer` and `solublempnn-designer` adapter
IDs. The checkpoint directory, the checkpoint name, the soluble flag, and the
designer ID all arrive as arguments, so the two adapter IDs stay separate in the
profile, the receipt, and the tool record.

`--ca-only` selects the third checkpoint family. It loads the C-alpha weights and
runs the runner's C-alpha parser, which is the route for a generator that emits a
C-alpha trace. Without it the runner fabricates the missing N, C and O atoms and
designs against the fabrication. `--ca-only` and `--soluble-model` name
incompatible checkpoint families, so the wrapper refuses the pair.

The wrapper reads backbone candidates from a completed upstream receipt or from
a published manifest, runs `protein_mpnn_run.py` once per backbone, and writes
receipt-owned outputs into the current attempt directory:

  <attempt>/<phase>/sequences/<candidate_id>.fasta      one record per candidate
  <attempt>/<phase>/poses/<candidate_id>.pdb            one design pose per candidate
  <attempt>/<phase>/sequence-candidate-manifest.jsonl   one row per candidate

Ownership validation rejects a row that points at an upstream file, so the
wrapper writes its own FASTA and its own design pose for every candidate. The
design pose carries the backbone coordinates the sequence was designed for, and
a REMARK line that names the candidate and the upstream pose hash.

ProteinMPNN reads a seed of zero as a request for a random seed, so the wrapper
replaces a zero seed with a fixed nonzero value and records both numbers.

A profile pins the checkpoint in its model_revision field. The wrapper hashes
the checkpoint it is about to load and refuses when the two disagree, naming
both digests. Reach the pinned value with --model-revision, or with --config
pointing at the resolved run config. Without either one the wrapper still hashes
the file, records the digest on every candidate row, and marks the row
`model_revision_verified: unrecorded`, because a profile that pins nothing makes
no claim to check.

Every command is an argument list that runs with shell=False. The wrapper builds
no shell string.

Install ProteinMPNN from https://github.com/dauparas/ProteinMPNN and point the
wrapper at the checkout with --proteinmpnn-root or PROTEINMPNN_ROOT. The default
automatic mode uses the fal application when --fal-url or PROTEINMPNN_FAL_URL
resolves, and uses the checkout otherwise. The fal client writes the same native
FASTA layout the local runner writes, so the parsing and record code is shared.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any

from claude_binder import backbone_shape
from claude_binder.clients import fal_invocation
from claude_binder.clients.fal_mpnn_client import cif_to_pdb
from claude_binder.adapters.candidate_lineage import DIVERSITY_LINEAGE_FIELDS
from claude_binder.paths import package_file

RUNNER_NAME = "protein_mpnn_run.py"
DEFAULT_MODEL_NAME = "v_48_020"
DEFAULT_WEIGHTS_SUBDIR = "vanilla_model_weights"
SOLUBLE_WEIGHTS_SUBDIR = "soluble_model_weights"
CA_ONLY_WEIGHTS_SUBDIR = "ca_model_weights"
# There is no CA-SolubleMPNN checkpoint. protein_mpnn_run.py prints a warning and
# calls sys.exit() when it gets both flags, which leaves no named cause behind, so
# the wrapper refuses the pair before it executes. Three sites refuse it, so the
# sentence lives in one place.
SOLUBLE_CA_ONLY_REFUSAL = "--soluble-model and --ca-only select incompatible checkpoints"
# What the runner's --ca_only parser reads. Upstream states the flag parses
# CA-only structures and uses CA-only models, and it loads ca_model_weights rather
# than vanilla_model_weights.
CA_ONLY_BACKBONE_ATOMS = ("CA",)
DEFAULT_MANIFEST_NAME = "sequence-candidate-manifest.jsonl"
DEFAULT_SEQUENCE_SUBDIR = "sequences"
DEFAULT_POSE_SUBDIR = "poses"
DEFAULT_WORK_SUBDIR = "proteinmpnn"
# "fal" belongs here because resolve_runner_protocol returns it and the command
# builder has a fal branch. Leaving it out made argparse reject --runner-protocol fal
# while the implementation behind it worked, so the route was reachable only when
# auto guessed it. rfdiffusion_generator.py keeps the three-value tuple because it
# ships no fal branch.
RUNNER_PROTOCOLS = ("auto", "local", "modal", "fal")
DEFAULT_RUNNER_PROTOCOL = "auto"
MODAL_PROTEINMPNN_ROOT = Path("/app/proteinmpnn")
FAL_URL_ENVIRONMENT_KEY = "PROTEINMPNN_FAL_URL"
# The packaged client refuses anything above 1200, at
# clients/fal_mpnn_client.py:36, because that is the application's own request
# ceiling. A larger default here reached the client and failed there, so the
# adapter could never complete a toolcheck. Keep the two numbers equal.
DEFAULT_FAL_TIMEOUT_SECONDS = 1200
DEFAULT_FAL_RECEIPT_NAME = "fal-receipt.json"
DEFAULT_CLIENT = package_file("clients", "fal_mpnn_client.py")
DEFAULT_FAL_EXECUTABLE = "fal-credential-wrapper"
DEFAULT_CLIENT_PYTHON = "python3"
FAL_REQUEST_ID_MAX_LENGTH = 96
FAL_REQUEST_ID_DIGEST_LENGTH = 12
FAL_ROUTE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REQUIRED_FAL_RECEIPT_FIELDS = (
    "checkpoint_sha256",
    "checkpoint_bytes",
    "environment_identity",
    "source_revision",
    "device",
)
# ProteinMPNN reads a false-y seed as a request for a random seed, so a run with
# --seed 0 is not reproducible. The wrapper substitutes this value and records
# both the requested seed and the seed the tool received.
SEED_ZERO_REPLACEMENT = 1000003
ATOM_RECORD_PREFIXES = ("ATOM  ", "HETATM")
CANONICAL_AMINO_ACID_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
DESIGNED_CHAINS_RE = re.compile(r"designed_chains=\[([^\]]*)\]")
QUOTED_CHAIN_RE = re.compile(r"'([^']+)'")
HEADER_FLOAT_RE = re.compile(r"(?:^|,)\s*([a-z_]+)=(-?\d+(?:\.\d+)?)")
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
    """Return the ProteinMPNN checkout directory."""
    if value is None:
        environment_value = os.environ.get("PROTEINMPNN_ROOT", "").strip()
        if not environment_value:
            raise AdapterError(
                "ProteinMPNN is not located. Pass --fal-url, or set PROTEINMPNN_FAL_URL, or "
                "pass --proteinmpnn-root, or set PROTEINMPNN_ROOT to a checkout of "
                "https://github.com/dauparas/ProteinMPNN"
            )
        value = Path(environment_value)
    root = value.expanduser()
    if not root.is_dir():
        raise AdapterError(f"ProteinMPNN root is not a directory: {root}")
    return root.resolve()


def resolve_fal_url(value: str | None) -> str:
    """Return the fal endpoint from the explicit flag or its environment variable."""
    url = (value or os.environ.get(FAL_URL_ENVIRONMENT_KEY, "")).strip()
    if not url:
        raise AdapterError(
            f"runner protocol fal needs an endpoint. Pass --fal-url, or set {FAL_URL_ENVIRONMENT_KEY}"
        )
    if not url.startswith("https://"):
        raise AdapterError(f"the fal endpoint is not an https URL: {url}")
    return url


def resolve_fal_client(value: Path | None) -> Path:
    """Return the client script the fal protocol runs."""
    client = (value or DEFAULT_CLIENT).expanduser()
    if not client.is_file():
        raise AdapterError(f"fal client not found: {client}")
    return client.resolve()


def resolve_fal_route(url: str) -> tuple[str, str]:
    """Return the team and app segments from one exact fal application URL."""
    parsed = urllib.parse.urlparse(url)
    segments = parsed.path.split("/")
    if (
        parsed.scheme != "https"
        or parsed.hostname != "fal.run"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or len(segments) != 3
        or segments[0] != ""
        or not segments[1]
        or not segments[2]
        or any(
            FAL_ROUTE_SEGMENT_RE.fullmatch(segment) is None
            for segment in segments[1:]
        )
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise AdapterError("fal endpoint must be exactly https://fal.run/<team>/<app>")
    return segments[1], segments[2]


def fal_client_argv(
    args: argparse.Namespace,
    client: Path,
    command: str,
    fal_url: str,
    *command_args: str,
) -> list[str]:
    """Build a credential-guarded command for the packaged fal client.

    The credential never enters this list. It reaches the client through the
    child environment, on whichever route the calling environment offers.
    """
    resolve_fal_route(fal_url)
    try:
        client_values = [command, "--fal-url", fal_url, *command_args]
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


def resolve_execution(args: argparse.Namespace) -> tuple[str, str | Path]:
    """Resolve the execution route and its checkout directory.

    ``local`` keeps the original checkout contract. ``modal`` uses the root
    supplied by the shipped Modal environment. ``auto`` selects an explicit
    checkout when supplied, or the fal endpoint when configured, or the
    shipped Modal root.
    """
    protocol = getattr(args, "runner_protocol", DEFAULT_RUNNER_PROTOCOL)
    fal_url = getattr(args, "fal_url", None)
    root_value = getattr(args, "proteinmpnn_root", None)
    environment_root = os.environ.get("PROTEINMPNN_ROOT", "").strip()

    if protocol == "local":
        return "local", resolve_root(root_value)

    if protocol == "modal":
        root = (root_value or MODAL_PROTEINMPNN_ROOT).expanduser()
        return "modal", root.resolve()

    if protocol == "fal":
        return "fal", resolve_fal_url(fal_url)

    if protocol != "auto":
        raise AdapterError(
            f"runner protocol {protocol!r} is not supported; choose one of {', '.join(RUNNER_PROTOCOLS)}"
        )

    if root_value is not None or environment_root:
        return "local", resolve_root(root_value)

    if (fal_url or os.environ.get(FAL_URL_ENVIRONMENT_KEY, "")).strip():
        return "fal", resolve_fal_url(fal_url)

    modal_root = MODAL_PROTEINMPNN_ROOT
    if modal_root.is_dir():
        return "modal", modal_root.resolve()

    return "local", resolve_root(root_value)


def resolve_runner(root: Path) -> Path:
    """Return the ProteinMPNN runner script inside a checkout."""
    runner = root / RUNNER_NAME
    if not runner.is_file():
        raise AdapterError(
            f"ProteinMPNN runner not found: {runner}. Point --proteinmpnn-root at a checkout "
            f"that contains {RUNNER_NAME}"
        )
    return runner


def resolve_weights(
    root: Path,
    override: Path | None,
    model_name: str,
    *,
    soluble: bool = False,
    ca_only: bool = False,
) -> tuple[Path, Path]:
    """Return the weights directory and the checkpoint file for one model.

    The runner picks its own default directory from the soluble flag, and this
    wrapper always passes the directory explicitly, so the default here follows
    the same flag and the two agree. The C-alpha flag picks a third directory the
    same way.
    """
    if soluble and ca_only:
        raise AdapterError(SOLUBLE_CA_ONLY_REFUSAL)
    default_subdir = (
        CA_ONLY_WEIGHTS_SUBDIR
        if ca_only
        else SOLUBLE_WEIGHTS_SUBDIR if soluble else DEFAULT_WEIGHTS_SUBDIR
    )
    weights_dir = override.expanduser() if override is not None else root / default_subdir
    if not weights_dir.is_dir():
        raise AdapterError(f"ProteinMPNN weights directory not found: {weights_dir}")
    checkpoint = weights_dir / f"{model_name}.pt"
    if not checkpoint.is_file():
        raise AdapterError(f"ProteinMPNN checkpoint not found: {checkpoint}")
    return weights_dir.resolve(), checkpoint.resolve()


def recorded_checkpoint_digest(model_revision: str | None) -> str | None:
    """Return the sha256 a model_revision string pins, or None when it pins none.

    A profile records model_revision as free text. The proteinmpnn-designer
    record reads `vanilla_model_weights/v_48_020.pt sha256:<64 hex>`, and the
    supplied-backbone generator records `none`. The digest is the only part of
    that string this wrapper can hold against a file, so it reads the token and
    ignores the rest. A string with no `sha256:` token pins no digest.
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

    Nothing compared these two before, so a run could succeed and attribute
    every sequence to a checkpoint that never loaded. The pinned string looked
    like verification and was decoration.

    The check runs once per wrapper invocation, before the first ProteinMPNN
    call, over the whole file. It does not run once per backbone, because the
    checkpoint cannot change between subprocesses of one phase. sha256 reads at
    about 1,800 MB/s on the machine this was written on, so a ProteinMPNN
    checkpoint costs milliseconds and each gigabyte of a larger one costs under
    a second. There is no cache: a cache keyed on file size and modification
    time admits a file swapped with both preserved, which is the case this check
    exists to catch, and the cost it would save is smaller than the guarantee it
    would cost.
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
    to two of them. `validate_normalized_candidate_lineage` keys its designer map by the
    same stage, so both sides resolve a two-arm tool to the same arm.
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

    `binder_lane_esmfold2_adapter.py` reads its own revision out of the same
    file the same way, and `claude_binder_lane.py` compares receipt rows against
    that record, so reading it here keeps the wrapper and the receipt from
    disagreeing about which checkpoint ran.

    Filtering on `id` alone required exactly one match, so one tool bound to two arms
    refused before it selected either revision. `rfdiffusion3-two-arm` registers
    `proteinmpnn` twice, on `proteinmpnn-designer` and `proteinmpnn-designer-rfd3`, and
    both argvs take this path by default. The published roster is SolubleMPNN across seven
    generators, so one tool on two arms is the normal case.
    `adapter_id` names the registration outright and wins when given; otherwise the arm is
    derived from the backbone stage this invocation reads.
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
        getattr(args, "designer_id", "proteinmpnn"),
        getattr(args, "backbone_stage_id", None),
        getattr(args, "adapter_id", None),
    )


def report_checkpoint_digest(checkpoint: Path, verification: dict[str, str]) -> None:
    """Print what the digest check found, including the case that checked nothing."""
    print(f"proteinmpnn adapter: checkpoint {checkpoint}")
    if verification["state"] == "matched":
        print(f"proteinmpnn adapter: sha256 {verification['observed']} matches model_revision")
        return
    print(
        f"proteinmpnn adapter: sha256 {verification['observed']} is unverified, because no "
        "model_revision reached this wrapper with a sha256 token. Pass --model-revision, or pass "
        "--config with the resolved run config, to check it"
    )


def resolve_tool_python(value: str | None) -> str:
    """Return the interpreter that runs ProteinMPNN."""
    if value is None:
        return sys.executable
    resolved = shutil.which(value)
    if resolved is None:
        raise AdapterError(f"interpreter not found: {value}")
    return resolved


def run_tool(argv: list[str], *, label: str = RUNNER_NAME) -> None:
    """Run one argument list with shell=False and fail on a nonzero exit."""
    print(f"proteinmpnn adapter: run {fal_invocation.redacted_command(argv)}", flush=True)
    completed = subprocess.run(argv, shell=False, check=False)
    if completed.returncode != 0:
        raise AdapterError(f"{label} exited {completed.returncode}")


def fal_request_id(candidate_id: str, seed: int) -> str:
    """Return a stable request ID accepted by the fal application."""
    digest = hashlib.sha256(f"{candidate_id}|{seed}".encode("utf-8")).hexdigest()
    suffix = digest[:FAL_REQUEST_ID_DIGEST_LENGTH]
    stem_budget = FAL_REQUEST_ID_MAX_LENGTH - len(suffix) - 1
    stem = re.sub(r"[^a-z0-9]+", "-", candidate_id.lower()).strip("-")[:stem_budget].strip("-")
    return f"{stem}-{suffix}" if stem else suffix


def fal_runtime_record(receipt_path: Path, url: str) -> dict[str, Any]:
    """Return the runtime identity a fal client receipt reports."""
    if not receipt_path.is_file():
        raise AdapterError(f"the fal client wrote no receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict):
        raise AdapterError(f"the fal receipt is not a JSON object: {receipt_path}")
    missing = [field for field in REQUIRED_FAL_RECEIPT_FIELDS if not receipt.get(field)]
    if missing:
        raise AdapterError(f"the fal receipt {receipt_path} records no {', '.join(missing)}")
    return {
        "runner_protocol": "fal",
        "fal_endpoint": url,
        "checkpoint_sha256": str(receipt["checkpoint_sha256"]),
        "checkpoint_bytes": int(receipt["checkpoint_bytes"]),
        "environment_identity": str(receipt["environment_identity"]),
        "source_revision": str(receipt["source_revision"]),
        "device": str(receipt["device"]),
        "runtime_wall_seconds": receipt.get("runner_wall_seconds"),
    }


def verify_remote_checkpoint_digest(
    observed: str, model_revision: str | None
) -> dict[str, str]:
    """Hold the checkpoint hash in a fal receipt against the profile pin."""
    recorded = recorded_checkpoint_digest(model_revision)
    if recorded is None:
        return {"state": "unrecorded", "observed": observed, "recorded": ""}
    if recorded != observed:
        raise AdapterError(
            f"the fal checkpoint hashes to {observed}, and model_revision pins {recorded}. "
            "The fal service is not running the checkpoint this run attributes its sequences to"
        )
    return {"state": "matched", "observed": observed, "recorded": recorded}


def tool_seed(requested_seed: int) -> int:
    """Return the seed ProteinMPNN receives for a requested seed."""
    return SEED_ZERO_REPLACEMENT if requested_seed == 0 else requested_seed


def parse_fasta_records(path: Path) -> list[tuple[str, str]]:
    """Return the header and sequence of every record in a ProteinMPNN FASTA."""
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

    The runner owns its working directory. Its output is not a plan artifact,
    so the adapter must not guess a private ``seqs/*.fa`` layout. A caller can
    provide the runner's enumerated path or glob explicitly. The default
    accepts the common FASTA suffixes anywhere below the owned work directory.
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


def header_designed_chains(header: str) -> list[str]:
    """Return the designed chain IDs a ProteinMPNN native header records."""
    match = DESIGNED_CHAINS_RE.search(header)
    if match is None:
        return []
    return QUOTED_CHAIN_RE.findall(match.group(1))


def header_scores(header: str) -> dict[str, float]:
    """Return the numeric fields a ProteinMPNN sample header records."""
    return {key: float(value) for key, value in HEADER_FLOAT_RE.findall(header)}


def chain_segment(sequence: str, designed: list[str], chain: str | None) -> str:
    """Return the residue string of one designed chain.

    ProteinMPNN joins the designed chains with a slash and orders them by the
    sorted chain ID, so the segment index is the position of the chain in the
    sorted designed chain list.
    """
    segments = sequence.split("/")
    if len(segments) == 1:
        return segments[0]
    if chain is None:
        raise AdapterError(
            f"ProteinMPNN designed {len(segments)} chains. Pass --design-chain to name the "
            "chain the binder sequence comes from"
        )
    order = sorted(designed)
    if chain not in order:
        raise AdapterError(f"chain {chain} is absent from the designed chains: {order}")
    if len(order) != len(segments):
        raise AdapterError(
            f"ProteinMPNN reported {len(order)} designed chains and returned {len(segments)} segments"
        )
    return segments[order.index(chain)]


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

    ProteinMPNN designs a sequence for a fixed backbone and returns no new
    coordinates, so the pose carries the upstream atom records unchanged. The
    REMARK lines name the candidate and the upstream file, which also keeps the
    bytes of every candidate pose distinct.

    Every chain of the source pose is copied, including the ones ProteinMPNN
    held fixed. Two consumers read the target chain out of this file:
    `binder_lane_interface_scorer_adapter.py` hands it to DockQ as the reference
    structure and names both the target and the binder chain in that call, and
    `binder_lane_esmfold2_adapter.py` compares its target residue keys against
    the prepared target before it folds anything. `chain` names the chain
    ProteinMPNN redesigned, and the only thing this function does with it is
    refuse a source pose that does not carry it.
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
    # Validate every declared upstream input before the first paid design call.
    # Rechecking in design_backbone protects against a file changing mid-phase.
    # A caller that names no designer states no requirement, so it gets no atom
    # check. Every argv route carries --designer-id, which defaults to
    # proteinmpnn, so only a direct call can reach this without one.
    required_atoms = designer_required_atoms(args)
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


def designer_required_atoms(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the backbone atoms this invocation reads, narrowed by --ca-only.

    The catalog states one atom set per tool, and that statement describes the
    vanilla and soluble checkpoints. `--ca-only` loads a different checkpoint and
    a different parser, one that reads a backbone carrying C-alpha atoms only, so
    this invocation reads the C-alpha part of what the tool states. Narrowing here
    keeps the flat catalog declaration true of the flat routes, and keeps the
    guard from refusing the route the flag exists to open. A tool that states
    nothing still narrows to nothing, so an unknown declaration stays unenforced.
    """
    declared = backbone_shape.required_atoms_for_tool(getattr(args, "designer_id", ""))
    if not getattr(args, "ca_only", False):
        return declared
    return tuple(atom for atom in declared if atom in CA_ONLY_BACKBONE_ATOMS)


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


def design_backbone(
    args: argparse.Namespace,
    *,
    row: dict[str, Any],
    index: int,
    runner: Path,
    weights_dir: Path | None,
    tool_python: str,
    phase_dir: Path,
    verification: dict[str, str] | None,
    runner_protocol: str,
    fal_url: str,
    model_revision: str | None,
) -> list[dict[str, Any]]:
    """Run ProteinMPNN for one backbone and return its candidate rows."""
    parent_id = str(row.get("candidate_id", ""))
    if not parent_id:
        raise AdapterError("a backbone row carries no candidate_id")
    source_pose, source_sha256 = backbone_pose(row)
    requested_seed = args.seed + index
    if requested_seed < 0:
        raise AdapterError(f"seed {requested_seed} is negative")
    effective_seed = tool_seed(requested_seed)
    work_dir = phase_dir / args.work_subdir / parent_id
    work_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = work_dir / DEFAULT_FAL_RECEIPT_NAME
    if runner_protocol == "fal":
        argv = fal_client_argv(
            args,
            runner,
            "run",
            fal_url,
            "--request-id",
            args.fal_request_id or fal_request_id(parent_id, effective_seed),
            "--input-structure",
            str(source_pose),
            "--out-dir",
            str(work_dir),
            "--receipt",
            str(receipt_path),
            "--sequences-per-backbone",
            str(args.sequences_per_backbone),
            "--seed",
            str(effective_seed),
            "--sampling-temp",
            str(args.sampling_temp),
            "--model-name",
            args.model_name,
            "--timeout-seconds",
            str(args.fal_timeout_seconds),
        )
        if args.soluble_model:
            argv.append("--soluble-model")
        if args.ca_only:
            argv.append("--ca-only")
        if args.design_chain is not None:
            argv.extend(["--design-chain", args.design_chain])
        run_tool(argv, label=runner.name)
        runtime = fal_runtime_record(receipt_path, fal_url)
        verification = verify_remote_checkpoint_digest(
            runtime["checkpoint_sha256"], model_revision
        )
    else:
        if weights_dir is None:
            raise AdapterError("the local ProteinMPNN path resolved no weights directory")
        argv = [
            tool_python,
            str(runner),
            "--pdb_path",
            str(source_pose),
            "--out_folder",
            str(work_dir),
            "--num_seq_per_target",
            str(args.sequences_per_backbone),
            "--batch_size",
            "1",
            "--sampling_temp",
            str(args.sampling_temp),
            "--seed",
            str(effective_seed),
            "--model_name",
            args.model_name,
            "--path_to_model_weights",
            str(weights_dir),
        ]
        if args.soluble_model:
            argv.append("--use_soluble_model")
        if args.ca_only:
            argv.append("--ca_only")
        if args.design_chain is not None:
            argv.extend(["--pdb_path_chains", args.design_chain])
        run_tool(argv)
        runtime = None
    produced = runner_sequence_outputs(work_dir, args.sequences_glob)
    records = parse_fasta_records(produced[0])
    if len(records) < args.sequences_per_backbone + 1:
        raise AdapterError(
            f"{produced[0]} holds {len(records)} records; expected the native record and "
            f"{args.sequences_per_backbone} designed records"
        )
    designed = header_designed_chains(records[0][0])
    rows: list[dict[str, Any]] = []
    for variant, (header, raw_sequence) in enumerate(records[1 : args.sequences_per_backbone + 1]):
        candidate_id = f"{parent_id}-{args.designer_id}-{variant:02d}"
        sequence = chain_segment(raw_sequence, designed, args.design_chain)
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
            "model_name": args.model_name,
            "soluble_model": bool(args.soluble_model),
            "ca_only": bool(args.ca_only),
            "checkpoint_sha256": verification["observed"] if verification else None,
            "model_revision_verified": verification["state"] if verification else None,
            "source_design_pose_sha256": source_sha256,
            "status": "sequence-designed",
        }
        for key, field in (
            ("score", "proteinmpnn_score"),
            ("global_score", "proteinmpnn_global_score"),
            ("seq_recovery", "proteinmpnn_sequence_recovery"),
        ):
            if key in scores:
                candidate[field] = scores[key]
        if runtime is not None:
            candidate.update(runtime)
        rows.append(candidate)
    return rows


def run(args: argparse.Namespace) -> int:
    """Design sequences for one phase and write the stage outputs."""
    if args.soluble_model and args.ca_only:
        raise AdapterError(SOLUBLE_CA_ONLY_REFUSAL)
    runner_protocol, execution = resolve_execution(args)
    model_revision = resolve_model_revision(args)
    fal_url = ""
    if runner_protocol == "fal":
        runner = resolve_fal_client(args.fal_client)
        fal_url = str(execution)
        root = None
        weights_dir = None
        verification = None
        tool_python = resolve_tool_python(args.tool_python)
    else:
        root = Path(execution)
        runner = resolve_runner(root)
        weights_dir, checkpoint = resolve_weights(
            root,
            args.weights_dir,
            args.model_name,
            soluble=args.soluble_model,
            ca_only=args.ca_only,
        )
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
                weights_dir=weights_dir,
                tool_python=tool_python,
                phase_dir=phase_dir,
                verification=verification,
                runner_protocol=runner_protocol,
                fal_url=fal_url,
                model_revision=model_revision,
            )
        )
    expected = args.count * args.sequences_per_backbone
    if len(rows) != expected:
        raise AdapterError(
            f"phase {args.phase} produced {len(rows)} candidates; expected {expected}"
        )
    write_jsonl(manifest_path, rows)
    print(
        f"proteinmpnn adapter: phase={args.phase} backbones={len(backbones)} "
        f"candidates={len(rows)} runtime={runner_protocol} manifest={manifest_path}"
    )
    return 0


def toolcheck(args: argparse.Namespace) -> int:
    """Report the ProteinMPNN files and interpreter this adapter uses."""
    if args.soluble_model and args.ca_only:
        raise AdapterError(SOLUBLE_CA_ONLY_REFUSAL)
    runner_protocol, execution = resolve_execution(args)
    tool_python = resolve_tool_python(args.tool_python)
    if runner_protocol == "fal":
        client = resolve_fal_client(args.fal_client)
        url = str(execution)
        with tempfile.TemporaryDirectory(
            prefix="claude-binder-proteinmpnn-toolcheck-"
        ) as temporary:
            argv = fal_client_argv(
                args,
                client,
                "toolcheck",
                url,
                "--model-name",
                args.model_name,
                "--timeout-seconds",
                str(args.fal_timeout_seconds),
                "--out-dir",
                temporary,
            )
            if args.soluble_model:
                argv.append("--soluble-model")
            if args.ca_only:
                argv.append("--ca-only")
            print(f"proteinmpnn adapter: client {client}")
            print(f"proteinmpnn adapter: endpoint {url}")
            run_tool(argv, label=f"{client.name} toolcheck")
        return 0

    root = Path(execution)
    runner = resolve_runner(root)
    _, checkpoint = resolve_weights(
        root, args.weights_dir, args.model_name, soluble=args.soluble_model, ca_only=args.ca_only
    )
    verification = verify_checkpoint_digest(checkpoint, resolve_model_revision(args))
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
    print(f"proteinmpnn adapter: runner {runner}")
    report_checkpoint_digest(checkpoint, verification)
    print(f"proteinmpnn adapter: interpreter {tool_python}")
    print(
        "proteinmpnn adapter: the probe skips the torch import, because ProteinMPNN imports "
        "torch only when it designs"
    )
    return 0


def add_tool_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runner-protocol",
        choices=RUNNER_PROTOCOLS,
        default=DEFAULT_RUNNER_PROTOCOL,
        help="Where ProteinMPNN runs: local uses an explicit checkout, modal uses the mounted image paths.",
    )
    parser.add_argument(
        "--fal-url",
        default=None,
        help=(
            f"Endpoint the fal protocol posts to. Defaults to {FAL_URL_ENVIRONMENT_KEY}. "
            "The client credential comes from the environment or from the wrapper."
        ),
    )
    parser.add_argument(
        "--fal-client",
        type=Path,
        default=None,
        help="Client script the fal protocol runs. Defaults to the packaged client.",
    )
    parser.add_argument(
        "--fal-executable",
        default=DEFAULT_FAL_EXECUTABLE,
        help=(
            "Credential wrapper that runs the fal client when the credential is not "
            f"already in the environment. Defaults to {DEFAULT_FAL_EXECUTABLE}."
        ),
    )
    fal_invocation.add_route_argument(parser, executable=DEFAULT_FAL_EXECUTABLE)
    fal_invocation.add_credential_environment_argument(parser)
    parser.add_argument(
        "--client-python",
        default=DEFAULT_CLIENT_PYTHON,
        help=f"Interpreter that runs the fal client. Defaults to {DEFAULT_CLIENT_PYTHON}.",
    )
    parser.add_argument(
        "--fal-timeout-seconds",
        type=int,
        default=DEFAULT_FAL_TIMEOUT_SECONDS,
        help=f"Request timeout for the fal protocol. Defaults to {DEFAULT_FAL_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--fal-request-id",
        default=None,
        help="Optional request identifier override for a parity rerun.",
    )
    parser.add_argument(
        "--proteinmpnn-root",
        type=Path,
        default=None,
        help="ProteinMPNN checkout directory for local mode. Defaults to PROTEINMPNN_ROOT.",
    )
    parser.add_argument(
        "--soluble-model",
        action="store_true",
        help=(
            "Run the SolubleMPNN weights on the same runner. Use this for the "
            "solublempnn-designer adapter ID."
        ),
    )
    parser.add_argument(
        "--ca-only",
        action="store_true",
        help=(
            "Use ProteinMPNN's C-alpha-only checkpoint and --ca_only parser. This is the "
            "route for a generator that emits a C-alpha trace."
        ),
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help=(
            f"Checkpoint directory. Defaults to {DEFAULT_WEIGHTS_SUBDIR} inside the checkout, "
            f"to {SOLUBLE_WEIGHTS_SUBDIR} with --soluble-model, or to "
            f"{CA_ONLY_WEIGHTS_SUBDIR} with --ca-only."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Checkpoint name without the .pt suffix. Defaults to {DEFAULT_MODEL_NAME}.",
    )
    parser.add_argument(
        "--tool-python",
        default=None,
        help="Interpreter that runs ProteinMPNN. Defaults to the interpreter running this wrapper.",
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
    # `toolcheck` calls `resolve_model_revision`, and `--model-revision` help promises that
    # `--config` is the other way to verify a checkpoint, but only `run` defined it. Twelve
    # resolved adapter entries across seven shipped profiles pass `--config` to this
    # subcommand, four of them with `--designer-id` as well, and every one exited 2 with
    # `unrecognized arguments` before probing anything.
    check_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Resolved runtime config to read this designer's model_revision from.",
    )
    check_parser.add_argument(
        "--designer-id",
        default="proteinmpnn",
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
        default="proteinmpnn",
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
        help="Sequences to design for every backbone. Match the stage records_per_count.",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base seed. Backbone N receives the base seed plus N.",
    )
    run_parser.add_argument(
        "--sampling-temp", type=float, default=0.1, help="ProteinMPNN sampling temperature."
    )
    run_parser.add_argument(
        "--design-chain",
        default=None,
        help=(
            "Chain to design and to keep in the design pose. Required for a backbone that "
            "carries more than one chain."
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
            "Raw ProteinMPNN output directory inside the phase directory. Defaults to "
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
        print(f"proteinmpnn adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
