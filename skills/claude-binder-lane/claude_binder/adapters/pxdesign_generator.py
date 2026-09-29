#!/usr/bin/env python3
"""Generate PXDesign binder designs on an operator-deployed fal application.

This wrapper fills the `pxdesign-generator` slot. PXDesign has no local route
here, so the hosted application is the only route and this module is the whole
adapter. It reads the target manifest the `target-preparer` stage published,
posts one generation request to a deployment the operator names, and writes
receipt-owned outputs into the current attempt directory:

  <attempt>/<phase>/poses/<candidate_id>.pdb          the PDB bridge pose
  <attempt>/<phase>/raw-poses/<candidate_id>.cif      the engine-native mmCIF
  <attempt>/<phase>/sequences/<candidate_id>.fasta    only when one is returned
  <attempt>/<phase>/candidate-manifest.jsonl          one row per candidate
  <attempt>/<phase>/pxdesign/                         the returned files and the
                                                      request receipt

Five properties of the route this wrapper handles rather than hides.

**The endpoint has no default.** Pass --fal-url, or set PXDESIGN_FAL_URL, with
the application URL of your own deployment. A URL baked into this file would
name somebody else's account, and the request carries an authorization header.

**The credential enters neither this process nor an argument list.** `run` and
`probe` build a child command through `clients/fal_invocation`, and only that
child opens a socket. Splitting the process is what lets the credential wrapper
put the credential in front of the request without it passing through this one.

**PXDesign may return a backbone or a designed sequence, and the row says
which.** The application reads the sequence out of its own output structure and
labels each design `sequence-structure` or `backbone-only`. A backbone-only row
carries null at `sequence_path`, `sequence_sha256`, `sequence_length` and
`sequence_designer`, sets `backbone_only` true, and leaves the sequence to a
downstream sequence-designer slot. Both kinds write the same file layout apart
from the FASTA.

**Both structures are kept.** The application returns its engine-native mmCIF
and a PDB bridge of the same coordinates. The PDB is the design pose downstream
stages read, and the mmCIF is written beside it unchanged, so a reader can go
back to what the engine wrote.

**The published pose carries the campaign's chain letters.** The application
assigns its own, and the sequence designer downstream is handed the campaign's
`--binder-chain` on its command line rather than reading it off the row, so a
pose left on the application's letters gets the target redesigned. `run` swaps
the two letters in column 22 of the PDB pose and changes nothing else. Every row
records both, at `binder_chain_id` and `returned_binder_chain_id`, and the
engine-native mmCIF beside it is untouched.

**The pinned public PXDesign CLI exposes no seed flag.** The application records
the requested seed and reports `seed_delivered` false. Every row records
`requested_seed` and `seed_delivered` so an unseeded run is disclosed rather
than hidden.

**A phase larger than one request is split into whole requests.** The
application caps one request at four designs and the published roster asks each
generator for fifty backbones, so `run` dispatches ceil(count / 4) requests,
each with its own request id, its own output directory and its own receipt.
Every row records which request produced it at `dispatch_batch`.

**The cost basis is unpriced.** No measurement in this package prices PXDesign
on any provider, so the receipt records `cost_basis: unpriced` and no number.

Subcommands:

  toolcheck        Report this adapter's own readiness. Sends no request, reads
                   no credential, and costs nothing.
  run              Compose the request, dispatch one phase, write the manifest.
  parse            Check the phase outputs this stage declares.
  probe            Ask the deployed application to report its runtime. This
                   starts a GPU runner and therefore costs money, so it refuses
                   to run without --acknowledge-cost.
  dispatch         The child `run` spawns. Not for direct use.
  dispatch-probe   The child `probe` spawns. Not for direct use.
"""

from __future__ import annotations

from claude_binder.adapters import structure_evidence as evidence

import argparse
import base64
import glob
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from claude_binder.adapters.candidate_lineage import backbone_lineage
from claude_binder.clients import fal_invocation
from claude_binder.paths import child_process_environment, package_file


# The child `run` and `probe` spawn is this same file. One file with dispatch
# subcommands gives the credential wrapper a process to exec without a second
# module, which is the shape `fal_genie3_generator` already uses.
DISPATCH_SCRIPT = package_file("adapters", "pxdesign_generator.py")
DISPATCH_COMMAND = "dispatch"
PROBE_CHILD_COMMAND = "dispatch-probe"
DEFAULT_FAL_EXECUTABLE = "fal-credential-wrapper"
FAL_URL_ENVIRONMENT_KEY = "PXDESIGN_FAL_URL"
FAL_HOSTNAME = "fal.run"
TOOLCHECK_PATH = "/toolcheck"
DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_RECEIPT_NAME = "fal-receipt.json"
DEFAULT_INDEX_NAME = "returned-designs.json"
RUNNER_PROTOCOL = "fal"
# `qualify.COST_KIND_UNPRICED`. Nothing measured this tool on this provider.
COST_BASIS = "unpriced"

DEFAULT_GENERATOR_ID = "pxdesign"
DEFAULT_ADAPTER_ID = "pxdesign-generator"
DEFAULT_MANIFEST_NAME = "candidate-manifest.jsonl"
DEFAULT_PARSER_RESULT_NAME = "parser-result.json"
DEFAULT_POSE_SUBDIR = "poses"
DEFAULT_RAW_POSE_SUBDIR = "raw-poses"
DEFAULT_SEQUENCE_SUBDIR = "sequences"
DEFAULT_WORK_SUBDIR = "pxdesign"
DEFAULT_TARGET_STAGE_ID = "target-prepare"
DEFAULT_TARGET_ARTIFACT_ID = "target-manifest"
DEFAULT_SEED = 37
DEFAULT_BINDER_CHAIN = "A"

# The request bounds the deployed application declares on its own input model. A
# request outside any of them is refused by the service before it generates, so
# it is refused here before it is sent and before it is paid for.
MAXIMUM_DESIGNS = 4
MINIMUM_BINDER_LENGTH = 20
MAXIMUM_BINDER_LENGTH = 400
MAXIMUM_HOTSPOTS = 128
MAXIMUM_SEED = 2_147_483_647
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,96}$")
MAXIMUM_REQUEST_ID_LENGTH = 96

REQUIRED_TARGET_MANIFEST_FIELDS = (
    "target_id",
    "target_sha256",
    "residue_map_sha256",
    "source_structure_path",
    "normalized_structure_path",
    "design_target_chain_id",
)
# Residue IDs arrive from the target manifest as CHAIN:NUMBER with an optional
# insertion code. A PXDesign hotspot is a bare residue number inside one chain,
# so an insertion code has no place to go and is refused rather than dropped.
RESIDUE_ID_RE = re.compile(r"^([^:]+):(-?\d+)([A-Za-z]?)$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CHAIN_ID_RE = re.compile(r"^[A-Za-z0-9]$")
# A returned name becomes a path under the output directory, so it has to be one
# plain file name. Anything else is refused rather than joined.
RETURNED_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FAL_ROUTE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SEQUENCE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})

POSE_FORMATS = ("pdb", "cif")
RAW_POSE_FORMAT = "cif"
OUTPUT_KIND_SEQUENCE = "sequence-structure"
OUTPUT_KIND_BACKBONE = "backbone-only"
OUTPUT_KINDS = (OUTPUT_KIND_SEQUENCE, OUTPUT_KIND_BACKBONE)
GENERATOR_MODE_BACKBONE = "backbone-only"
GENERATOR_MODE_CODESIGN = "sequence-structure-codesign"
CANDIDATE_STATUS = "generated"
READ_BLOCK_BYTES = 1024 * 1024
ATOM_RECORD_PREFIXES = ("ATOM  ", "HETATM")

# The identity the application reports on every answer. A response missing any
# of these describes a runner nobody can name afterwards, so the run refuses it
# rather than write a manifest that cannot be traced.
REQUIRED_RESPONSE_FIELDS = ("device", "source_revision", "environment_identity")
OPTIONAL_RESPONSE_FIELDS = (
    "identity_verification",
    "dependency_revisions",
    "model_revision",
    "persistent_run_id",
    "reused_completed_inference",
    "seconds",
)


class AdapterError(RuntimeError):
    """A PXDesign input, response, or returned artifact is invalid."""


# ----------------------------------------------------------------------------
# Files.
# ----------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 of one payload."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of the complete file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one JSON document to a path in an atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows to a path in one atomic replace."""
    if not rows:
        raise AdapterError(f"refusing to write an empty candidate manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_json(path: Path, label: str) -> dict[str, Any]:
    """Return the JSON object a file holds."""
    if not path.is_file():
        raise AdapterError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise AdapterError(f"{label} is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"{label} is not a JSON object: {path}")
    return value


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


# ----------------------------------------------------------------------------
# The endpoint and the credential route.
# ----------------------------------------------------------------------------


def resolve_endpoint(value: str | None, environment_key: str = FAL_URL_ENVIRONMENT_KEY) -> str:
    """Return the deployed application URL, refusing anything but one fal route.

    There is no default. The request carries an `Authorization` header, so a URL
    pointing somewhere else would hand a credential to a host the operator never
    chose.
    """
    url = (value or os.environ.get(environment_key, "")).strip()
    if not url:
        raise AdapterError(
            "this route needs an endpoint and has no default. Pass --fal-url, or set "
            f"{environment_key}, with the application URL of your own deployment"
        )
    parsed = urllib.parse.urlparse(url)
    segments = parsed.path.split("/")
    if (
        parsed.scheme != "https"
        or parsed.hostname != FAL_HOSTNAME
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or len(segments) != 3
        or segments[0] != ""
        or any(FAL_ROUTE_SEGMENT_RE.fullmatch(segment) is None for segment in segments[1:])
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise AdapterError(
            f"the fal endpoint must be exactly https://{FAL_HOSTNAME}/<account>/<application>"
        )
    return url


def resolve_credential_route(args: argparse.Namespace) -> tuple[str, str]:
    """Return the route this call takes and the variable name holding the credential."""
    try:
        credential_env = fal_invocation.credential_environment_key(
            getattr(args, "fal_credential_env", None)
        )
        route = fal_invocation.resolve_route(
            args.fal_executable,
            requested=getattr(args, "fal_credential_route", None),
            credential_env_key=credential_env,
        )
    except fal_invocation.RouteError as exc:
        raise AdapterError(str(exc)) from exc
    return route, credential_env


def child_argv(
    args: argparse.Namespace,
    child: str,
    endpoint: str,
    *values: str,
    script: Path | None = None,
) -> list[str]:
    """Return the child command that carries the credential, never the credential.

    `child` is the subcommand of the named script that opens the socket. The
    route and the variable name are resolved here; the value never is.
    """
    route, credential_env = resolve_credential_route(args)
    child_values = [child, "--fal-url", endpoint, *values]
    if (
        route == fal_invocation.ROUTE_DIRECT
        and credential_env != fal_invocation.CREDENTIAL_ENVIRONMENT_KEY
    ):
        child_values.extend(["--credential-env", credential_env])
    try:
        return fal_invocation.client_command(
            args.fal_executable,
            args.client_python,
            script or DISPATCH_SCRIPT,
            child_values,
            requested=route,
            credential_env_key=credential_env,
        )
    except fal_invocation.RouteError as exc:
        raise AdapterError(str(exc)) from exc


def run_external(argv: list[str], label: str, tool: str = "pxdesign") -> None:
    """Run one child command and fail on a nonzero exit.

    The printed line is redacted, because the endpoint identifies an account.
    The child inherits an environment that can import this package, because a
    host that loads this package by file path leaves its parent off `sys.path`.
    """
    print(f"{tool} adapter: {label}: {fal_invocation.redacted_command(argv)}", flush=True)
    completed = subprocess.run(argv, shell=False, check=False, env=child_process_environment())
    if completed.returncode != 0:
        raise AdapterError(f"{label} exited {completed.returncode}")


# ----------------------------------------------------------------------------
# The transport. Only the child subcommands reach this, never a parent process.
# ----------------------------------------------------------------------------


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse to follow a redirect away from the endpoint the caller named."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise urllib.error.HTTPError(req.full_url, code, "fal redirect rejected", headers, fp)


def credential(credential_env: str) -> str:
    """Return the fal credential from the named variable. It is never logged."""
    value = os.environ.get(credential_env, "").strip()
    if not value:
        raise AdapterError(
            f"{credential_env} is not set in this process. Run through the credential wrapper, "
            "or set the variable, and stop if no authorized route is available. Refusing to "
            "continue prevents an unauthenticated request"
        )
    return value


def post(
    url: str, payload: dict[str, Any], timeout_seconds: int, credential_env: str
) -> dict[str, Any]:
    """Post one JSON request and return the JSON object the application answered.

    There is no retry. The application declares its own retry skips and the
    request is not idempotent, so a second attempt would pay for a second
    generation nobody asked for.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Key {credential(credential_env)}",
            "Content-Type": "application/json",
            "X-Fal-No-Retry": "1",
            "X-Fal-Request-Timeout": str(timeout_seconds),
        },
    )
    try:
        opener = urllib.request.build_opener(RejectRedirects())
        with opener.open(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        if exc.code in REDIRECT_STATUS_CODES:
            raise AdapterError(
                f"the application answered HTTP {exc.code} and this adapter refuses to follow "
                f"the redirect to {exc.headers.get('Location', 'an unstated address')}"
            ) from exc
        raise AdapterError(f"the application answered HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise AdapterError(f"the application could not be reached: {exc.reason}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AdapterError(f"the application answered {len(raw)} bytes that are not JSON") from exc
    if not isinstance(value, dict):
        raise AdapterError("the application answered a JSON value that is not an object")
    return value


# ----------------------------------------------------------------------------
# The target.
# ----------------------------------------------------------------------------


def completed_receipt_file(receipts_dir: Path, stage_id: str, artifact_id: str) -> Path:
    """Return the one file a completed upstream receipt recorded for an artifact."""
    receipt_path = receipts_dir / f"{stage_id}.json"
    receipt = load_json(receipt_path, "upstream receipt")
    if receipt.get("ok") is not True:
        raise AdapterError(f"upstream receipt did not complete: {receipt_path}")
    artifacts = receipt.get("output_manifest", {}).get("artifacts", [])
    phases = {str(artifact.get("phase")) for artifact in artifacts}
    selected_phase = "scale" if "scale" in phases else "single"
    records = [
        file_record
        for artifact in artifacts
        if artifact.get("phase") == selected_phase and artifact.get("artifact_id") == artifact_id
        for file_record in artifact.get("files", [])
    ]
    paths: list[Path] = []
    for file_record in records:
        if not isinstance(file_record, dict) or not str(file_record.get("path") or "").strip():
            raise AdapterError(
                f"upstream receipt {receipt_path} carries a {artifact_id} file entry with no "
                f"path for phase {selected_phase}"
            )
        paths.append(Path(str(file_record["path"])))
    if len(paths) != 1:
        raise AdapterError(
            f"upstream receipt {receipt_path} carries {len(paths)} {artifact_id} files for phase "
            f"{selected_phase}; expected exactly one"
        )
    return paths[0]


def load_target_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    """Return the target manifest this stage designs against, and its path."""
    if getattr(args, "target_manifest", None) is not None:
        artifact_root = args.artifact_root.expanduser().resolve()
        path = resolve_input_path(artifact_root, str(args.target_manifest), "target manifest")
    else:
        path = completed_receipt_file(
            args.receipts_dir, args.target_stage_id, args.target_artifact_id
        )
    manifest = load_json(path, "target manifest")
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


def normalized_structure(manifest: dict[str, Any], path: Path) -> tuple[Path, str]:
    """Return the normalized target structure this route sends and check its hash."""
    structure_path = Path(str(manifest["normalized_structure_path"])).expanduser()
    if not structure_path.is_file():
        raise AdapterError(f"normalized target structure is missing: {structure_path}")
    observed = sha256_file(structure_path)
    recorded = manifest.get("normalized_structure_sha256")
    if isinstance(recorded, str) and recorded and recorded != observed:
        raise AdapterError(
            f"normalized target structure changed since the target stage: {structure_path}; "
            f"the manifest records {recorded} and the file reads {observed}"
        )
    return structure_path.resolve(), observed


def site_residue_numbers(manifest: dict[str, Any], path: Path, chain: str) -> list[int]:
    """Return the design site residue numbers, in the design target chain."""
    site = manifest.get("site")
    if not isinstance(site, dict):
        raise AdapterError(f"target manifest {path} records no site")
    residues = site.get("resolved_design_residues")
    if not isinstance(residues, list) or not residues:
        raise AdapterError(f"target manifest {path} records no resolved_design_residues")
    structure, _ = normalized_structure(manifest, path)
    try:
        atoms = evidence.structure_atoms(structure.read_bytes(), structure.suffix.lower().lstrip("."), str(structure))
    except evidence.EvidenceError as exc:
        raise AdapterError(str(exc)) from exc
    present = {(atom.chain_id, atom.residue_number, atom.insertion_code) for atom in atoms if atom.record == "ATOM"}
    numbers: list[int] = []
    for residue in residues:
        match = RESIDUE_ID_RE.fullmatch(str(residue))
        if match is None:
            raise AdapterError(f"site residue does not read CHAIN:NUMBER: {residue}")
        residue_chain, number, insertion_code = match.group(1), match.group(2), match.group(3)
        if insertion_code:
            raise AdapterError(
                f"site residue {residue} carries an insertion code, and a PXDesign hotspot is a "
                "bare residue number with no place for one"
            )
        if residue_chain != chain:
            raise AdapterError(
                f"site residue {residue} names chain {residue_chain}, and the design target "
                f"chain is {chain}"
            )
        if (residue_chain, int(number), insertion_code) not in present:
            raise AdapterError(f"site residue {residue} is absent from the normalized target structure")
        numbers.append(int(number))
    ordered = sorted(set(numbers))
    if not ordered:
        raise AdapterError(f"target manifest {path} resolved no design site residue")
    if len(ordered) > MAXIMUM_HOTSPOTS:
        raise AdapterError(
            f"the design site holds {len(ordered)} residues and the application accepts at most "
            f"{MAXIMUM_HOTSPOTS}"
        )
    return ordered


# ----------------------------------------------------------------------------
# The request and the response.
# ----------------------------------------------------------------------------


def request_id(value: str | None, fallback: str) -> str:
    """Return a request identifier the application accepts."""
    candidate = (value or fallback).strip()
    if REQUEST_ID_RE.fullmatch(candidate) is None:
        raise AdapterError(
            f"the request id has to be 1 to {MAXIMUM_REQUEST_ID_LENGTH} characters of letters, "
            f"digits, dot, underscore or hyphen: {candidate!r}"
        )
    return candidate


def batch_request_id(identifier: str, position: int) -> str:
    """Keep each batch ID within the service limit without losing uniqueness."""
    candidate = f"{identifier}-b{position:03d}"
    if len(candidate) <= MAXIMUM_REQUEST_ID_LENGTH:
        return request_id(None, candidate)
    digest = sha256_bytes(candidate.encode("utf-8"))[:16]
    stem = identifier[:MAXIMUM_REQUEST_ID_LENGTH - len(digest) - 1].rstrip("-._")
    return request_id(None, f"{stem}-{digest}")


def request_batches(count: int, maximum: int = MAXIMUM_DESIGNS) -> list[int]:
    """Return the per-request design counts one phase of `count` designs needs.

    The application caps one request, and a phase asks for as many designs as the
    profile's `backbone_count` declares. A phase larger than the cap is split
    into whole requests rather than refused, because the published roster asks
    each generator for fifty backbones and no single request can carry them.
    """
    if count < 1:
        raise AdapterError(f"--count is {count}; a phase needs at least one design")
    if maximum < 1:
        raise AdapterError(f"the per-request ceiling is {maximum}; it has to be positive")
    batches = [maximum] * (count // maximum)
    remainder = count % maximum
    if remainder:
        batches.append(remainder)
    return batches


def validate_request_values(
    *, count: int, binder_length: int, seed: int, generator_id: str
) -> None:
    """Refuse a request the application would reject, before it is paid for."""
    if not 1 <= count <= MAXIMUM_DESIGNS:
        raise AdapterError(
            f"--count is {count}; the application accepts 1 to {MAXIMUM_DESIGNS} designs per "
            "request"
        )
    if not MINIMUM_BINDER_LENGTH <= binder_length <= MAXIMUM_BINDER_LENGTH:
        raise AdapterError(
            f"--binder-length is {binder_length}; the application accepts "
            f"{MINIMUM_BINDER_LENGTH} to {MAXIMUM_BINDER_LENGTH}"
        )
    if not 0 <= seed <= MAXIMUM_SEED:
        raise AdapterError(f"--seed is {seed}; the application accepts 0 to {MAXIMUM_SEED}")
    if IDENTIFIER_RE.fullmatch(generator_id) is None:
        raise AdapterError(f"--generator-id is not a plain identifier: {generator_id}")


def build_payload(
    *,
    identifier: str,
    target_structure: Path,
    target_chain: str,
    hotspots: list[int],
    binder_length: int,
    count: int,
    seed: int,
) -> dict[str, Any]:
    """Return the JSON body of one generation request."""
    if not target_structure.is_file():
        raise AdapterError(f"target structure not found: {target_structure}")
    return {
        "request_id": identifier,
        "target_pdb_text": target_structure.read_text(),
        "target_chain": target_chain,
        "hotspots": list(hotspots),
        "binder_length": binder_length,
        "count": count,
        "seed": seed,
    }


def unpack(design: dict[str, Any], field: str, label: str) -> bytes:
    """Return one gzipped base64 payload, checked against its recorded digest."""
    encoded = design.get(f"{field}_gzip_base64")
    if not isinstance(encoded, str) or not encoded:
        raise AdapterError(f"the application returned no {label} payload")
    try:
        payload = gzip.decompress(base64.b64decode(encoded, validate=True))
    except (ValueError, OSError, EOFError) as exc:
        raise AdapterError(f"the {label} payload is not valid gzipped base64") from exc
    recorded_digest = design.get(f"{field}_sha256")
    observed = sha256_bytes(payload)
    if not isinstance(recorded_digest, str) or recorded_digest != observed:
        raise AdapterError(
            f"the {label} payload hashes {observed} and the runner recorded {recorded_digest}"
        )
    recorded_length = design.get(f"{field}_bytes")
    if not isinstance(recorded_length, int) or isinstance(recorded_length, bool):
        raise AdapterError(f"the {label} payload carries no integer byte count")
    if recorded_length != len(payload):
        raise AdapterError(
            f"the {label} payload is {len(payload)} bytes and the runner recorded "
            f"{recorded_length}"
        )
    return payload


def design_chain_id(design: dict[str, Any], field: str) -> str:
    """Return a chain identifier the application reported, refusing an odd one."""
    value = design.get(field)
    if not isinstance(value, str) or CHAIN_ID_RE.fullmatch(value) is None:
        raise AdapterError(f"the application reported {field} {value!r}, which is not a chain id")
    return value


def decode_designs(response: dict[str, Any], expected: int) -> list[dict[str, Any]]:
    """Return the returned designs decoded, refusing a bad one.

    Everything is decoded and checked before anything is written, so a response
    that fails halfway leaves no partial structure directory behind.
    """
    designs = response.get("designs")
    if not isinstance(designs, list) or not designs:
        raise AdapterError("the application returned no designs")
    if len(designs) != expected:
        raise AdapterError(
            f"the application returned {len(designs)} designs and the request asked for {expected}"
        )
    decoded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, design in enumerate(designs):
        if not isinstance(design, dict):
            raise AdapterError(f"design {index} is not a JSON object")
        name = str(design.get("design_name", ""))
        if RETURNED_NAME_RE.fullmatch(name) is None:
            raise AdapterError(f"the application returned a design name this adapter refuses: {name!r}")
        if name in seen:
            raise AdapterError(f"the application returned {name} twice")
        seen.add(name)
        pose_format = str(design.get("pose_format", "pdb")).lower()
        if pose_format not in POSE_FORMATS:
            raise AdapterError(f"the application returned an unsupported pose format: {pose_format}")
        raw_format = str(design.get("raw_pose_format", RAW_POSE_FORMAT)).lower()
        if raw_format != RAW_POSE_FORMAT:
            raise AdapterError(
                f"the application returned an unsupported raw pose format: {raw_format}"
            )
        output_kind = str(design.get("output_kind", ""))
        if output_kind not in OUTPUT_KINDS:
            raise AdapterError(
                f"design {index} reports output_kind {output_kind!r}, which is not one of "
                + ", ".join(OUTPUT_KINDS)
            )
        chain_lengths = design.get("chain_lengths")
        if not isinstance(chain_lengths, dict) or not chain_lengths:
            raise AdapterError(f"design {index} carries no chain_lengths")
        raw_binder_chain = design_chain_id(design, "raw_binder_chain_id")
        binder_length = chain_lengths.get(raw_binder_chain)
        if not isinstance(binder_length, int) or isinstance(binder_length, bool) or binder_length < 1:
            raise AdapterError(
                f"design {index} reports no positive residue count for its binder chain "
                f"{raw_binder_chain}"
            )
        record: dict[str, Any] = {
            "design_index": index,
            "design_name": name,
            "pose": unpack(design, "pose", "pose"),
            "pose_format": pose_format,
            "raw_pose": unpack(design, "raw_pose", "raw pose"),
            "raw_pose_format": raw_format,
            "output_kind": output_kind,
            "binder_chain_id": design_chain_id(design, "binder_chain_id"),
            "raw_binder_chain_id": raw_binder_chain,
            "binder_length": binder_length,
            "chain_lengths": chain_lengths,
            "chain_id_map": design.get("chain_id_map"),
            "sequence": None,
            "fasta": None,
        }
        if output_kind == OUTPUT_KIND_SEQUENCE:
            sequence = str(design.get("sequence", "")).upper()
            if SEQUENCE_RE.fullmatch(sequence) is None:
                raise AdapterError(
                    f"design {index} claims a designed sequence and returned one that is not "
                    "canonical single-letter residues"
                )
            if len(sequence) != binder_length:
                raise AdapterError(
                    f"design {index} returned a {len(sequence)}-residue sequence for a "
                    f"{binder_length}-residue binder chain"
                )
            record["sequence"] = sequence
            record["fasta"] = unpack(design, "fasta", "FASTA")
        validate_design_content(record)
        decoded.append(record)
    return decoded


def runtime_fields(
    response: dict[str, Any],
    required: tuple[str, ...] = REQUIRED_RESPONSE_FIELDS,
    optional: tuple[str, ...] = OPTIONAL_RESPONSE_FIELDS,
) -> dict[str, Any]:
    """Return the runtime identity the response reports, refusing an incomplete one."""
    missing = [field for field in required if field not in response or response[field] in (None, "")]
    if missing:
        raise AdapterError(f"the application reported no {', '.join(missing)}")
    fields = {field: response[field] for field in required}
    for field in optional:
        if response.get(field) is not None:
            fields[field] = response[field]
    if "identity_verification" in response:
        fields["identity_verification"] = response["identity_verification"]
    return fields


def validate_design_content(record: dict[str, Any]) -> None:
    """Verify the coordinate and sequence content, independently of payload hashes."""
    try:
        observed = evidence.chain_sequence(record["pose"], record["pose_format"], record["binder_chain_id"], "returned pose")
        if len(observed) != record["binder_length"]:
            raise evidence.EvidenceError("returned pose binder length disagrees with the response")
        if record.get("raw_pose") is not None:
            evidence.structure_atoms(record["raw_pose"], record["raw_pose_format"], "raw pose")
        if record.get("sequence") is not None:
            if observed != record["sequence"]:
                raise evidence.EvidenceError("returned pose binder sequence disagrees with the response")
            if evidence.fasta_sequence(record["fasta"], "returned FASTA") != record["sequence"]:
                raise evidence.EvidenceError("returned FASTA disagrees with the response sequence")
    except evidence.EvidenceError as exc:
        raise AdapterError(str(exc)) from exc


def validate_runtime_identity(args: argparse.Namespace, receipt: dict[str, Any]) -> None:
    """Compare observed identities with every identity the resolved adapter declares."""
    from claude_binder import lane
    config = load_json(args.config, "resolved config")
    stages = [row for row in config.get("stages", []) if row.get("stage_id") == args.stage]
    if len(stages) != 1:
        raise AdapterError(f"resolved config must contain exactly one stage {args.stage}")
    adapters = [row for row in config.get("adapters", []) if row.get("adapter_id") == stages[0].get("adapter_id")]
    if len(adapters) != 1:
        raise AdapterError(f"resolved config must contain exactly one adapter for {args.stage}")
    verification = {}
    for field in ("source_revision", "model_revision", "environment_identity"):
        expected = adapters[0].get(field)
        if expected is None:
            continue
        if lane.is_required_placeholder(expected):
            raise AdapterError(f"adapter {field} remains unresolved")
        observed = receipt.get(field)
        if field == "source_revision" and isinstance(expected, str) and isinstance(observed, str):
            # FreeBindCraft identifies the pinned tool and its deployment wrapper
            # separately in one field. A tool-only pin checks that component;
            # an app-inclusive pin still requires exact equality below.
            if re.fullmatch(re.escape(expected) + r"\+app-[0-9a-f]{16}", observed):
                verification[field] = {"status": "matched", "scope": "tool_revision", "declared": expected}
                continue
        if field == "model_revision" and observed is None:
            digest = receipt.get("checkpoint_sha256")
            if digest is None:
                # The PXDesign service reports no model identity. Preserve that
                # limitation explicitly; a configured pin is not an observed pin.
                verification[field] = {"status": "not-reported", "declared": expected}
                continue
            if expected in (digest, "sha256:" + str(digest)):
                verification[field] = {"status": "matched", "via": "checkpoint_sha256"}
                continue
        if observed != expected:
            raise AdapterError(f"returned {field} {observed!r} does not match configured {expected!r}")
        verification[field] = {"status": "matched"}
    receipt["identity_verification"] = verification


def write_index_and_files(
    out_dir: Path,
    decoded: list[dict[str, Any]],
    *,
    index_name: str = DEFAULT_INDEX_NAME,
) -> dict[str, Any]:
    """Write every returned file under one directory and index what was written.

    The parent process reads the index rather than the response, so the decoded
    payloads never cross the process boundary a second time.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in decoded:
        stem = f"design-{record['design_index']:03d}"
        pose_path = out_dir / f"{stem}.{record['pose_format']}"
        pose_path.write_bytes(record["pose"])
        raw_path = out_dir / f"{stem}.raw.{record['raw_pose_format']}"
        raw_path.write_bytes(record["raw_pose"])
        row = {
            "design_index": record["design_index"],
            "design_name": record["design_name"],
            "pose_file": pose_path.name,
            "pose_format": record["pose_format"],
            "pose_sha256": sha256_bytes(record["pose"]),
            "raw_pose_file": raw_path.name,
            "raw_pose_format": record["raw_pose_format"],
            "raw_pose_sha256": sha256_bytes(record["raw_pose"]),
            "output_kind": record["output_kind"],
            "binder_chain_id": record["binder_chain_id"],
            "raw_binder_chain_id": record["raw_binder_chain_id"],
            "binder_length": record["binder_length"],
            "chain_lengths": record["chain_lengths"],
            "chain_id_map": record["chain_id_map"],
            "sequence": record["sequence"],
            "sequence_file": None,
            "sequence_sha256": None,
        }
        if record["fasta"] is not None:
            fasta_path = out_dir / f"{stem}.fasta"
            fasta_path.write_bytes(record["fasta"])
            row["sequence_file"] = fasta_path.name
            row["sequence_sha256"] = sha256_bytes(record["fasta"])
        rows.append(row)
    index = {"schema_version": 1, "designs": rows}
    write_json(out_dir / index_name, index)
    return index


def write_receipt(
    path: Path,
    response: dict[str, Any],
    *,
    endpoint: str,
    client_wall_seconds: float,
    requested_seed: int,
    design_count: int,
    required: tuple[str, ...] = REQUIRED_RESPONSE_FIELDS,
    optional: tuple[str, ...] = OPTIONAL_RESPONSE_FIELDS,
) -> None:
    """Write what the runner reported and what this request asked of it.

    There is no cost field with a number in it. No measurement in this package
    prices this tool on this provider.
    """
    receipt: dict[str, Any] = dict(runtime_fields(response, required, optional))
    receipt.update(
        {
            "runner_protocol": RUNNER_PROTOCOL,
            "fal_endpoint": endpoint,
            "request_id": response.get("request_id"),
            "requested_seed": requested_seed,
            "used_seed": response.get("used_seed"),
            "seed_delivered": bool(response.get("seed_delivered", False)),
            "design_count": design_count,
            "client_wall_seconds": round(client_wall_seconds, 3),
            "runner_wall_seconds": response.get("seconds"),
            "cost_basis": COST_BASIS,
        }
    )
    write_json(path, receipt)


# ----------------------------------------------------------------------------
# Subcommands.
# ----------------------------------------------------------------------------


def dispatch(args: argparse.Namespace) -> int:
    """Post one generation request and write the files the runner returned.

    `run` spawns this. It and `dispatch_probe` are the only code in this module
    that opens a socket, and they are the processes the credential wrapper puts
    the credential in front of.
    """
    endpoint = resolve_endpoint(args.fal_url)
    validate_request_values(
        count=args.count,
        binder_length=args.binder_length,
        seed=args.seed,
        generator_id=DEFAULT_GENERATOR_ID,
    )
    hotspots = [int(value) for value in str(args.hotspots).split(",") if value.strip()]
    if not hotspots:
        raise AdapterError("--hotspots carries no residue number")
    payload = build_payload(
        identifier=request_id(args.request_id, DEFAULT_GENERATOR_ID),
        target_structure=args.target_structure.expanduser(),
        target_chain=args.target_chain,
        hotspots=hotspots,
        binder_length=args.binder_length,
        count=args.count,
        seed=args.seed,
    )
    started = time.monotonic()
    response = post(endpoint, payload, args.timeout_seconds, args.credential_env)
    seconds = time.monotonic() - started
    decoded = decode_designs(response, args.count)
    out_dir = args.out_dir.expanduser()
    write_index_and_files(out_dir, decoded)
    write_receipt(
        args.receipt.expanduser(),
        response,
        endpoint=endpoint,
        client_wall_seconds=seconds,
        requested_seed=args.seed,
        design_count=len(decoded),
    )
    print(
        f"pxdesign dispatch: designs={len(decoded)} seconds={seconds:.1f} "
        f"device={response['device']} out_dir={out_dir}"
    )
    return 0


def probe(args: argparse.Namespace) -> int:
    """Spawn the child that asks the deployed application to report its runtime.

    Starting a runner is not free, so this refuses without `--acknowledge-cost`.
    `toolcheck` is the free check and it answers a different question.
    """
    if not args.acknowledge_cost:
        raise AdapterError(
            "probe starts a GPU runner on your own deployment and therefore costs money, so "
            "it needs --acknowledge-cost. Run toolcheck instead for the free readiness check, "
            "which sends no request and reads no credential"
        )
    endpoint = resolve_endpoint(args.fal_url)
    run_external(
        child_argv(
            args,
            PROBE_CHILD_COMMAND,
            endpoint,
            "--timeout-seconds",
            str(args.timeout_seconds),
            *(["--request-id", args.request_id] if args.request_id else []),
        ),
        "probe",
    )
    return 0


def dispatch_probe(args: argparse.Namespace) -> int:
    """Post one toolcheck request and print the runtime the application reported."""
    endpoint = resolve_endpoint(args.fal_url)
    started = time.monotonic()
    response = post(
        endpoint.rstrip("/") + TOOLCHECK_PATH,
        {"request_id": request_id(args.request_id, "toolcheck")},
        args.timeout_seconds,
        args.credential_env,
    )
    seconds = time.monotonic() - started
    fields = runtime_fields(response)
    print(f"pxdesign probe: device {fields['device']}")
    print(f"pxdesign probe: environment {fields['environment_identity']}")
    print(f"pxdesign probe: source {fields['source_revision']}")
    print(f"pxdesign probe: took {seconds:.1f} seconds")
    return 0


def toolcheck(args: argparse.Namespace) -> int:
    """Report this adapter's own readiness without sending anything.

    A toolcheck that called the application would start a GPU runner every time
    a profile validated itself, so this one answers only what it can answer for
    free.
    """
    ready = True
    try:
        resolve_endpoint(args.fal_url)
        print(f"pxdesign adapter: endpoint {fal_invocation.REDACTED_FAL_URL}")
    except AdapterError as exc:
        ready = False
        print(f"pxdesign adapter: endpoint unresolved: {exc}")
    try:
        route, credential_env = resolve_credential_route(args)
        print(f"pxdesign adapter: credential route {route} through {credential_env}")
    except AdapterError as exc:
        ready = False
        print(f"pxdesign adapter: credential route unavailable: {exc}")
    print(
        f"pxdesign adapter: dispatch child {args.client_python} {DISPATCH_SCRIPT} "
        f"{DISPATCH_COMMAND}"
    )
    print(
        f"pxdesign adapter: request ceiling count 1-{MAXIMUM_DESIGNS}, binder_length "
        f"{MINIMUM_BINDER_LENGTH}-{MAXIMUM_BINDER_LENGTH}, hotspots 1-{MAXIMUM_HOTSPOTS}, "
        f"seed 0-{MAXIMUM_SEED}"
    )
    print(
        f"pxdesign adapter: a phase larger than {MAXIMUM_DESIGNS} designs is dispatched as "
        "whole requests, one per batch, each with its own request id and receipt"
    )
    print(
        "pxdesign adapter: the pinned public CLI exposes no seed flag, so a row records the "
        "requested seed and seed_delivered false"
    )
    print(f"pxdesign adapter: cost basis {COST_BASIS}; no measurement prices this provider")
    print(
        "pxdesign adapter: this check sends no request and reads no credential; probe is the "
        "paid subcommand and it needs --acknowledge-cost",
        flush=True,
    )
    if not ready:
        print("pxdesign adapter: not ready to dispatch", file=sys.stderr)
    return 0 if ready else 1


def phase_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Return the attempt directory, the phase directory, the work directory and the manifest."""
    attempt_dir = args.attempt_dir.expanduser().resolve()
    phase_dir = attempt_dir / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    work_dir = phase_dir / args.work_subdir
    manifest_path = (
        resolve_output_path(attempt_dir, args.manifest_path, "manifest path")
        if getattr(args, "manifest_path", None) is not None
        else phase_dir / DEFAULT_MANIFEST_NAME
    )
    return attempt_dir, phase_dir, work_dir, manifest_path


def refuse_populated_output(out_dir: Path) -> None:
    """Refuse a run against a directory that already holds returned files.

    The walk is recursive because a split phase writes one subdirectory per
    request. A check that read only the top level saw directories rather than
    files and let a second run overwrite the first one's designs.
    """
    if out_dir.is_dir():
        existing = sorted(path for path in out_dir.rglob("*") if path.is_file())
        if existing:
            raise AdapterError(
                f"the output directory already holds {len(existing)} files: {out_dir}. A stale "
                "file cannot be told apart from a returned one, so run the stage in a clean "
                "attempt directory"
            )


def swap_chain_labels(text: str, first: str, second: str) -> str:
    """Exchange two chain letters in the PDB chain column.

    Chain columns on coordinates, termination and annotation records are updated. The coordinates, the atom names
    and the residue identities stay the bytes the application wrote, because a
    design pose carrying an atom the generator did not produce is a fabrication.

    It is a swap rather than a one-way relabel because the pose carries the
    target as well as the binder. Rewriting the binder onto the letter the target
    already holds would give one pose two chains with one id.
    """
    mapping = {first: second, second: first}
    lines = []
    for line in text.splitlines(keepends=True):
        columns = {
            "ATOM": (21,), "HETATM": (21,), "ANISOU": (21,), "SIGATM": (21,),
            "SIGUIJ": (21,), "TER": (21,), "SEQRES": (11,), "DBREF": (12,),
            "DBREF1": (12,), "DBREF2": (12,), "SEQADV": (16,), "MODRES": (16,),
            "HELIX": (19, 31), "SHEET": (21, 32, 49, 64), "SSBOND": (15, 29),
            "LINK": (21, 51), "CISPEP": (15, 29), "HET": (12,),
        }.get(line[:6].strip(), ())
        for column in columns:
            if len(line) > column:
                line = line[:column] + mapping.get(line[column], line[column]) + line[column + 1:]
        lines.append(line)
    return "".join(lines)


def swapped_chain(label: str, first: str, second: str) -> str:
    """Return where one chain letter lands after `swap_chain_labels`."""
    return {first: second, second: first}.get(label, label)


def restore_author_residues(text: str, target_chain: str, mapping: dict[int, str]) -> str:
    """Restore the target's author labels after a service used contiguous positions."""
    lines = []
    for line in text.splitlines(keepends=True):
        if line[:6].strip() in {"ATOM", "HETATM", "ANISOU", "SIGATM", "SIGUIJ", "TER"} and line[21:22] == target_chain:
            try:
                author = mapping[int(line[22:26])]
                match = RESIDUE_ID_RE.fullmatch(author)
                if match is None:
                    raise ValueError("invalid author identity")
                number, insertion = int(match.group(2)), match.group(3)
                if not -999 <= number <= 9999:
                    raise ValueError("author number does not fit PDB")
            except (ValueError, KeyError) as exc:
                raise AdapterError("returned target residue is absent from its recorded numbering map") from exc
            line = line[:22] + f"{number:4d}" + (insertion or " ") + line[27:]
        lines.append(line)
    return "".join(lines)


def publish_design_files(
    index_row: dict[str, Any],
    out_dir: Path,
    phase_dir: Path,
    candidate_id: str,
    *,
    binder_chain: str | None = None,
    target_residue_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Copy one returned design into the stage layout and hash what was written.

    `binder_chain` is the chain letter the campaign designs on. The application
    assigns its own letters, and a downstream sequence designer is handed the
    campaign's letter on the command line rather than reading it off the row, so
    a pose left on the application's letters gets the target redesigned. The PDB
    pose is relabelled to the campaign's convention and the engine-native file
    beside it keeps the letters the application wrote.
    """
    published: dict[str, Any] = {}
    for source_key, hash_key, subdir, suffix in (
        ("pose_file", "pose_sha256", DEFAULT_POSE_SUBDIR, index_row["pose_format"]),
        ("raw_pose_file", "raw_pose_sha256", DEFAULT_RAW_POSE_SUBDIR, index_row["raw_pose_format"]),
        ("sequence_file", "sequence_sha256", DEFAULT_SEQUENCE_SUBDIR, "fasta"),
    ):
        name = index_row.get(source_key)
        if not name:
            published[source_key] = None
            continue
        if RETURNED_NAME_RE.fullmatch(str(name)) is None:
            raise AdapterError(f"the returned index names a file this adapter refuses: {name!r}")
        source = out_dir / str(name)
        if not source.is_file():
            raise AdapterError(f"the dispatch child did not write {source}")
        observed = sha256_file(source)
        if observed != index_row.get(hash_key):
            raise AdapterError(
                f"{source} hashes {observed} and the returned index records {index_row.get(hash_key)}"
            )
        destination = phase_dir / subdir / f"{candidate_id}.{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        returned_chain = str(index_row["binder_chain_id"])
        relabelled = (
            source_key == "pose_file"
            and suffix == "pdb"
            and binder_chain is not None
            and binder_chain != returned_chain
        )
        payload = source.read_bytes()
        if source_key == "pose_file" and suffix == "pdb" and target_residue_map is not None:
            payload = restore_author_residues(payload.decode("utf-8"), str(index_row["target_chain_id"]), target_residue_map).encode("utf-8")
        if relabelled:
            payload = swap_chain_labels(payload.decode("utf-8"), returned_chain, str(binder_chain)).encode("utf-8")
        try:
            if suffix in {"pdb", "cif", "mmcif"}:
                evidence.structure_atoms(payload, suffix, str(source))
            elif suffix == "fasta":
                sequence = evidence.fasta_sequence(payload, str(source))
                if sequence != index_row.get("sequence"):
                    raise evidence.EvidenceError("indexed sequence disagrees with FASTA")
        except evidence.EvidenceError as exc:
            raise AdapterError(str(exc)) from exc
        destination.write_bytes(payload)
        published[source_key] = {
            "path": destination.resolve(),
            "sha256": sha256_bytes(payload),
            "returned_sha256": observed,
            "relabelled": relabelled,
        }
    return published


def same_seed(seed: int, position: int) -> int:
    """Return the seed every request in a split phase carries.

    PXDesign's pinned CLI reads no seed, so repeating one across requests changes
    nothing about what the runner does. A tool whose application does deliver the
    seed passes its own function instead.
    """
    return seed


def dispatch_batches(
    args: argparse.Namespace,
    *,
    endpoint: str,
    out_dir: Path,
    work_dir: Path,
    common_values: list[str],
    batches: list[int],
    script: Path | None = None,
    tool: str = "pxdesign",
    seed_for_batch: Callable[[int, int], int] = same_seed,
) -> list[tuple[dict[str, Any], list[dict[str, Any]], Path]]:
    """Dispatch one phase as whole requests and return each batch's receipt and index."""
    results: list[tuple[dict[str, Any], list[dict[str, Any]], Path]] = []
    attempt_digest = sha256_bytes(str(args.attempt_dir.expanduser().resolve()).encode("utf-8"))[:16]
    fallback = f"{args.generator_id}-{args.phase}"
    # Reserve room for the attempt suffix and the batch suffix. Resuming the same
    # attempt keeps its ID; a fresh attempt cannot reuse a remote run directory.
    fallback = fallback[:MAXIMUM_REQUEST_ID_LENGTH - 24] + "-" + attempt_digest
    identifier = request_id(args.request_id, fallback)
    for position, batch_count in enumerate(batches):
        batch_dir = out_dir / f"batch-{position:03d}"
        receipt_path = work_dir / f"batch-{position:03d}-{args.receipt_name}"
        run_external(
            child_argv(
                args,
                DISPATCH_COMMAND,
                endpoint,
                *common_values,
                "--count",
                str(batch_count),
                "--seed",
                str(seed_for_batch(args.seed, position)),
                "--out-dir",
                str(batch_dir.resolve()),
                "--receipt",
                str(receipt_path.resolve()),
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--request-id",
                batch_request_id(identifier, position),
                script=script,
            ),
            f"generate {batch_count} designs, request {position + 1} of {len(batches)}",
            tool,
        )
        receipt = load_json(receipt_path, "fal receipt")
        validate_runtime_identity(args, receipt)
        write_json(receipt_path, receipt)
        expected_request = batch_request_id(identifier, position)
        if receipt.get("request_id") != expected_request:
            raise AdapterError("returned request_id does not match the dispatched batch")
        index = load_json(batch_dir / DEFAULT_INDEX_NAME, "returned design index")
        returned = index.get("designs")
        if not isinstance(returned, list) or len(returned) != batch_count:
            raise AdapterError(
                f"request {position + 1} asked for {batch_count} designs and the dispatch child "
                f"indexed {len(returned) if isinstance(returned, list) else 0}"
            )
        results.append((receipt, returned, batch_dir))
    return results


def run(args: argparse.Namespace) -> int:
    """Compose the request, dispatch one phase, and write the stage outputs."""
    batches = request_batches(args.count)
    validate_request_values(
        count=batches[0],
        binder_length=args.binder_length,
        seed=args.seed,
        generator_id=args.generator_id,
    )
    if CHAIN_ID_RE.fullmatch(args.binder_chain) is None:
        raise AdapterError(
            f"--binder-chain is {args.binder_chain}; a chain ID is one letter or digit"
        )
    endpoint = resolve_endpoint(args.fal_url)
    # The route is resolved before anything is written, so a machine with no way
    # to reach the credential fails before it composes a request.
    resolve_credential_route(args)

    manifest, manifest_source = load_target_manifest(args)
    target_id = str(manifest["target_id"])
    chain = args.target_chain or str(manifest["design_target_chain_id"])
    if CHAIN_ID_RE.fullmatch(chain) is None:
        raise AdapterError(f"the design target chain is {chain!r}, which is not a chain id")
    if chain == args.binder_chain:
        raise AdapterError(
            f"the target chain and --binder-chain are both {chain}; they name two different "
            "chains of the design pose, which carries the binder and the target together"
        )
    structure_path, structure_sha256 = normalized_structure(manifest, manifest_source)
    hotspots = site_residue_numbers(manifest, manifest_source, chain)

    attempt_dir, phase_dir, work_dir, manifest_path = phase_paths(args)
    out_dir = work_dir / "returned"
    refuse_populated_output(out_dir)

    results = dispatch_batches(
        args,
        endpoint=endpoint,
        out_dir=out_dir,
        work_dir=work_dir,
        batches=batches,
        common_values=[
            "--target-structure",
            str(structure_path),
            "--target-chain",
            chain,
            "--hotspots",
            ",".join(str(number) for number in hotspots),
            "--binder-length",
            str(args.binder_length),
        ],
    )

    rows: list[dict[str, Any]] = []
    position = 0
    for batch_number, (receipt, returned, batch_dir) in enumerate(results):
        runtime = runtime_fields(receipt)
        receipt_path = work_dir / f"batch-{batch_number:03d}-{args.receipt_name}"
        for index_row in returned:
            candidate_id = f"{args.generator_id}-{position:03d}"
            published = publish_design_files(
                index_row, batch_dir, phase_dir, candidate_id, binder_chain=args.binder_chain
            )
            pose = published["pose_file"]
            raw_pose = published["raw_pose_file"]
            sequence_file = published["sequence_file"]
            sequence = index_row.get("sequence")
            backbone_only = index_row["output_kind"] == OUTPUT_KIND_BACKBONE
            rows.append(
                {
                    "target_id": target_id,
                    "target_sha256": str(manifest["target_sha256"]),
                    "candidate_id": candidate_id,
                    "parent_candidate_id": None,
                    "origin_generator": args.generator_id,
                    **backbone_lineage(candidate_id, args.generator_id),
                    "generator_mode": (
                        GENERATOR_MODE_BACKBONE if backbone_only else GENERATOR_MODE_CODESIGN
                    ),
                    "runner_protocol": RUNNER_PROTOCOL,
                    "sequence_designer": None if backbone_only else args.generator_id,
                    "generator_seed": None,
                    "requested_seed": args.seed,
                    "tool_seed": receipt.get("used_seed"),
                    "seed_delivered": bool(receipt.get("seed_delivered", False)),
                    "sequence_path": (
                        None if sequence_file is None else str(sequence_file["path"])
                    ),
                    "sequence_sha256": (
                        None if sequence is None else sha256_bytes(str(sequence).encode("ascii"))
                    ),
                    "sequence_length": None if sequence is None else len(str(sequence)),
                    "backbone_only": backbone_only,
                    "structure_path": str(manifest["source_structure_path"]),
                    "structure_sha256": str(manifest["target_sha256"]),
                    "design_pose_path": str(pose["path"]),
                    "design_pose_sha256": pose["sha256"],
                    "raw_design_pose_path": str(raw_pose["path"]),
                    "raw_design_pose_sha256": raw_pose["sha256"],
                    "residue_map_sha256": str(manifest["residue_map_sha256"]),
                    "optimization_round": 0,
                    "last_optimizer": None,
                    "status": CANDIDATE_STATUS,
                    "stage_id": args.stage,
                    "design_index": position,
                    "dispatch_batch": batch_number,
                    "design_name": index_row["design_name"],
                    "binder_chain_id": args.binder_chain,
                    "returned_binder_chain_id": index_row["binder_chain_id"],
                    "design_pose_relabelled": pose["relabelled"],
                    "returned_pose_sha256": pose["returned_sha256"],
                    "raw_binder_chain_id": index_row["raw_binder_chain_id"],
                    "binder_length": index_row["binder_length"],
                    "requested_binder_length": args.binder_length,
                    "chain_lengths": index_row["chain_lengths"],
                    "chain_id_map": index_row["chain_id_map"],
                    # The application relabels the target chains itself, starting
                    # at A, and the swap moves whichever of them now holds the
                    # campaign's binder letter. Recording the campaign's declared
                    # target letter instead would name a chain the file may not
                    # carry.
                    "target_chain_id": (
                        swapped_chain(
                            "A", str(index_row["binder_chain_id"]), str(args.binder_chain)
                        )
                        if pose["relabelled"]
                        else "A"
                    ),
                    "declared_target_chain_id": chain,
                    "target_manifest_path": str(manifest_source),
                    "input_structure_path": str(structure_path),
                    "input_structure_sha256": structure_sha256,
                    "hotspot_residues": hotspots,
                    "fal_endpoint": endpoint,
                    "fal_receipt_path": str(receipt_path.resolve()),
                    "runtime_wall_seconds": receipt.get("runner_wall_seconds"),
                    "cost_basis": COST_BASIS,
                    **runtime,
                }
            )
            position += 1
    write_jsonl(manifest_path, rows)
    print(
        f"pxdesign adapter: phase={args.phase} target={target_id} candidates={len(rows)} "
        f"requests={len(batches)} seed_delivered={rows[0]['seed_delivered']} "
        f"device={rows[0]['device']} manifest={manifest_path} "
        f"target_manifest={manifest_source}"
    )
    return 0


# ----------------------------------------------------------------------------
# The parser phase.
# ----------------------------------------------------------------------------


def stage_record(config_path: Path, stage_id: str) -> dict[str, Any]:
    """Return one stage contract from a resolved config."""
    config = load_json(config_path, "resolved config")
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


def parser_output_pattern(template: str, attempt_dir: Path, phase: str, stage_id: str) -> str:
    """Render an output contract path for the parser."""
    rendered = (
        template.replace("{{attempt_dir}}", str(attempt_dir))
        .replace("{{phase}}", phase)
        .replace("{{stage_id}}", stage_id)
    )
    if "{{" in rendered or "}}" in rendered:
        raise AdapterError(f"parser output path carries an unsupported token: {template}")
    return rendered


def parse_outputs(args: argparse.Namespace) -> int:
    """Check the phase outputs this stage declares and write the parser result."""
    attempt_dir = args.attempt_dir.expanduser().resolve()
    phase_dir = attempt_dir / args.phase
    files: list[Path] = []
    parsed_count = 0
    errors: list[str] = []
    try:
        stage = stage_record(args.config.expanduser().resolve(), args.stage)
    except AdapterError as exc:
        stage = {}
        errors.append(str(exc))
    for output in stage.get("outputs", []):
        if not isinstance(output, dict) or not isinstance(output.get("path_template"), str):
            errors.append("stage output has no path_template")
            continue
        pattern = parser_output_pattern(output["path_template"], attempt_dir, args.phase, args.stage)
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
                elif kind in {"pdb", "cif", "mmcif"} or path.suffix.lower() in {".pdb", ".cif", ".mmcif"}:
                    evidence.structure_atoms(path.read_bytes(), path.suffix.lower().lstrip("."), str(path))
                    parsed_count += 1
                elif kind == "fasta" or path.suffix.lower() in {".fasta", ".fa"}:
                    evidence.fasta_sequence(path.read_bytes(), str(path))
                    parsed_count += 1
                else:
                    raise AdapterError(f"no content parser for declared output kind {kind!r}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    result_path = phase_dir / DEFAULT_PARSER_RESULT_NAME
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
    # Without this the dispatcher logs a bare rc=1 and the reason sits in a file
    # nobody opens. A parse failure after a paid generation is the worst place to
    # hide a message.
    for message in errors:
        print(f"pxdesign parse: {message}", file=sys.stderr)
    print(
        f"pxdesign adapter: parsed={parsed_count} rejected={len(errors)} "
        f"phase={args.phase} result={result_path}"
    )
    return 0 if files and not errors else 1


# ----------------------------------------------------------------------------
# Arguments.
# ----------------------------------------------------------------------------


def add_route_arguments(
    parser: argparse.ArgumentParser, environment_key: str = FAL_URL_ENVIRONMENT_KEY
) -> None:
    parser.add_argument(
        "--fal-url",
        default=None,
        help=(
            "Application URL of your own deployment, exactly "
            f"https://{FAL_HOSTNAME}/<account>/<application>. There is no default. "
            f"Defaults to {environment_key}."
        ),
    )
    parser.add_argument(
        "--client-python",
        default=sys.executable,
        help=(
            "Interpreter that runs the dispatch child. The child imports this package, so the "
            "default is the interpreter running this process."
        ),
    )
    parser.add_argument("--fal-executable", default=DEFAULT_FAL_EXECUTABLE)
    fal_invocation.add_route_argument(parser, executable=DEFAULT_FAL_EXECUTABLE)
    fal_invocation.add_credential_environment_argument(parser)


def add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", required=True, help="Stage ID in the resolved config.")
    parser.add_argument("--phase", required=True, help="Phase name, such as smoke or scale.")
    parser.add_argument("--count", type=int, required=True, help="Designs this phase requests.")
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Resolved run config.")
    parser.add_argument("--plan", type=Path, default=None, help="Materialized plan, recorded only.")


def add_target_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-manifest", type=Path, default=None)
    parser.add_argument("--target-stage-id", default=DEFAULT_TARGET_STAGE_ID)
    parser.add_argument("--target-artifact-id", default=DEFAULT_TARGET_ARTIFACT_ID)
    parser.add_argument(
        "--target-chain",
        default=None,
        help="Design target chain. Defaults to design_target_chain_id in the target manifest.",
    )


def add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--binder-length",
        type=int,
        required=True,
        help=(
            "Binder chain length the application generates. The application accepts "
            f"{MINIMUM_BINDER_LENGTH} to {MAXIMUM_BINDER_LENGTH} and returns the one chain of "
            "that length."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Requested seed, recorded on every row. The pinned public CLI exposes no seed flag, "
            "so the runner reports seed_delivered false."
        ),
    )
    parser.add_argument(
        "--request-id",
        default=None,
        help="Request identifier the application echoes. Defaults to the generator and phase.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Request timeout. Defaults to {DEFAULT_TIMEOUT_SECONDS}.",
    )


def add_layout_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generator-id", default=DEFAULT_GENERATOR_ID)
    parser.add_argument("--binder-chain", default=DEFAULT_BINDER_CHAIN)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--work-subdir", default=DEFAULT_WORK_SUBDIR)
    parser.add_argument("--receipt-name", default=DEFAULT_RECEIPT_NAME)
    parser.add_argument(
        "--binder-length-min",
        type=int,
        default=None,
        help="Recorded only. The application generates exactly --binder-length residues.",
    )
    parser.add_argument(
        "--binder-length-max",
        type=int,
        default=None,
        help="Recorded only. The application generates exactly --binder-length residues.",
    )


def add_dispatch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-structure", type=Path, required=True)
    parser.add_argument("--target-chain", required=True)
    parser.add_argument("--hotspots", required=True, help="Comma separated residue numbers.")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--credential-env",
        type=fal_invocation.credential_environment_key,
        default=fal_invocation.CREDENTIAL_ENVIRONMENT_KEY,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PXDesign designs on a fal deployment.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "toolcheck", help="Report readiness without sending a request."
    )
    add_route_arguments(check_parser)

    run_parser = subparsers.add_parser("run", help="Generate one phase of designs on fal.")
    add_route_arguments(run_parser)
    add_stage_arguments(run_parser)
    add_target_arguments(run_parser)
    add_request_arguments(run_parser)
    add_layout_arguments(run_parser)

    parse_parser = subparsers.add_parser("parse", help="Parse the outputs of one completed phase.")
    add_route_arguments(parse_parser)
    add_stage_arguments(parse_parser)

    probe_parser = subparsers.add_parser(
        "probe", help="Ask the deployment to report its runtime. This costs money."
    )
    add_route_arguments(probe_parser)
    probe_parser.add_argument("--request-id", default=None)
    probe_parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    probe_parser.add_argument("--acknowledge-cost", action="store_true")

    dispatch_parser = subparsers.add_parser(
        DISPATCH_COMMAND, help="Child of run. Posts the request. Not for direct use."
    )
    dispatch_parser.add_argument("--fal-url", default=None)
    add_dispatch_arguments(dispatch_parser)
    add_request_arguments(dispatch_parser)

    probe_child_parser = subparsers.add_parser(
        PROBE_CHILD_COMMAND, help="Child of probe. Posts the toolcheck. Not for direct use."
    )
    probe_child_parser.add_argument("--fal-url", default=None)
    probe_child_parser.add_argument("--request-id", default=None)
    probe_child_parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    probe_child_parser.add_argument(
        "--credential-env",
        type=fal_invocation.credential_environment_key,
        default=fal_invocation.CREDENTIAL_ENVIRONMENT_KEY,
    )
    return parser


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    handlers = {
        "toolcheck": toolcheck,
        "run": run,
        "parse": parse_outputs,
        "probe": probe,
        DISPATCH_COMMAND: dispatch,
        PROBE_CHILD_COMMAND: dispatch_probe,
    }
    try:
        return handlers[args.command](args)
    except AdapterError as exc:
        print(f"pxdesign adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
