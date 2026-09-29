#!/usr/bin/env python3
"""Generate BoltzGen binder designs on an operator-deployed fal application.

This wrapper fills the `boltzgen-generator` slot. BoltzGen is a co-design
generator: one request returns a target-binder complex and the sequence of the
binder chain, so no sequence-designer stage follows this one. The adapter reads
the target manifest the `target-preparer` stage published, posts one request per
batch to a deployment the operator names, and writes receipt-owned outputs into
the current attempt directory:

  <attempt>/<phase>/poses/<candidate_id>.pdb           the PDB complex
  <attempt>/<phase>/raw-poses/<candidate_id>.cif       the engine-native mmCIF
  <attempt>/<phase>/sequences/<candidate_id>.fasta     the designed binder
  <attempt>/<phase>/metrics/<candidate_id>.json        BoltzGen's own metric row
  <attempt>/<phase>/candidate-manifest.jsonl           one row per candidate
  <attempt>/<phase>/boltzgen/                          the returned files and the
                                                       request receipts

`pxdesign_generator` is the hosted skeleton these generators share. The
endpoint, the credential route, the transport, the target manifest and the
parser phase come from that module. What follows is what BoltzGen does
differently, handled here rather than hidden.

**The endpoint has no default.** Pass --fal-url, or set BOLTZGEN_FAL_URL, with
the application URL of your own deployment. A URL baked into this file would
name somebody else's account, and the request carries an authorization header.

**BoltzGen 0.3.2 has no CLI seed control.** The proven invocation exposes no
seed flag, so the runner reports `seed_delivered` false and `used_seed` null.
Two consequences are stated rather than hidden. A design cannot be reproduced
from its row. And splitting a phase into several requests cannot vary a seed,
so the requests differ only by BoltzGen's own sampling; every request in a split
phase therefore carries the same `--seed`, recorded as `requested_seed`.

**The served invocation is one protocol.** The application runs
`--protocol peptide-anything`, which is the recipe the recorded run used. The
service fixes that command internally and does not return a `protocol` field.
The row records the served protocol from this adapter's bounded route rather
than treating a field the service never sends as an echo.

**The answer is checked against the target that was sent.** The application
echoes `request_target_sha256` and `requested_hotspot_residues`. A run refuses a
response whose echo disagrees with what this adapter sent, because a design
built against a different target or a different site is not a design for this
campaign. The application also reports `delivered_hotspot_positions`, the
one-based chain positions it mapped the campaign's residue numbers onto, and
every row records both lists.

**The verified dispatch boundary is PDB.** Target preparation always writes the
normalized target as PDB, and the service preserves those PDB bytes before it
maps author residue numbers to BoltzGen positions. This adapter accepts that
route so it can independently bind the input hash and the delivered positions.
The service also has an mmCIF conversion fallback, but its gemmi serialization
cannot be reproduced by this standard-library adapter.

**A phase larger than one request is split into whole requests.** The
application caps one request at eight designs and the published roster asks each
generator for fifty backbones, so `run` dispatches ceil(count / 8) requests,
each with its own request id, its own output directory and its own receipt.
Every row records which request produced it at `dispatch_batch`.

**Both structures are kept.** The application returns its engine-native mmCIF
and a PDB bridge of the same coordinates. The PDB is the design pose downstream
stages read, and the mmCIF is written beside it unchanged.

**The published pose carries the campaign's chain letters.** The request names
the pair it wants and the application validates the returned PDB bridge against
that pair before it answers. The response does not echo chain IDs, so this
adapter parses the bridge and refuses it unless it contains the requested target
and binder chains and the binder coordinates spell the returned sequence.
Every row records the requested and returned chain IDs as the same letters. The
engine-native mmCIF beside it is untouched.

**BoltzGen's own metric row is preserved.** The application returns the matched
engine metric object for each design. The adapter writes it to a file per
candidate and records its path and hash. The service does not declare a
schema-level native filter verdict, so the row records `unreported` rather than
guessing one from an arbitrary metric-table column.

**The code is MIT and the weight terms are unverified.** Settle the checkpoint
terms before a commercial campaign. The adapter records the checkpoint digest
the runner reports.

**The cost basis is unpriced.** No measurement in this package prices BoltzGen
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

import argparse
import base64
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

from claude_binder.adapters import pxdesign_generator as base
from claude_binder.adapters.candidate_lineage import backbone_lineage
from claude_binder.adapters import structure_evidence as evidence
from claude_binder.adapters import target_prep_adapter as target_prep
from claude_binder.clients import fal_invocation
from claude_binder.paths import package_file


DISPATCH_SCRIPT = package_file("adapters", "boltzgen_generator.py")
DISPATCH_COMMAND = base.DISPATCH_COMMAND
PROBE_CHILD_COMMAND = base.PROBE_CHILD_COMMAND
FAL_URL_ENVIRONMENT_KEY = "BOLTZGEN_FAL_URL"

DEFAULT_GENERATOR_ID = "boltzgen"
DEFAULT_ADAPTER_ID = "boltzgen-generator"
DEFAULT_WORK_SUBDIR = "boltzgen"
DEFAULT_METRICS_SUBDIR = "metrics"
TOOL_LABEL = "boltzgen"

# The request bounds the deployed application declares on its own input model. A
# request outside any of them is refused by the service before it generates, so
# it is refused here before it is sent and before it is paid for.
MAXIMUM_DESIGNS = 8
MINIMUM_BINDER_LENGTH = 1
MAXIMUM_BINDER_LENGTH = 256
MAXIMUM_HOTSPOTS = 64
# The one invocation the recorded evidence covers. A longer protein-binder
# protocol needs its own canary before this adapter accepts one.
DESIGN_PROTOCOL = "peptide-anything"

DEFAULT_BINDER_CHAIN = "A"
GENERATOR_MODE = "sequence-structure-codesign"
POSE_FORMAT = "pdb"
RAW_POSE_FORMAT = "cif"
# The source metric object has no schema-level filter verdict. An absent verdict
# is therefore preserved as unreported rather than treated as a pass.
UNREPORTED_FILTER_STATE = "unreported"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# These are the fields ``DesignResponse`` declares in the deployed application.
# They are all present in a successful answer, even where the declared type is
# nullable.  Keep this separate from the common receipt writer's required
# runtime identity, which intentionally needs only the fields every hosted
# adapter shares.
SOURCE_RESPONSE_FIELDS = (
    "request_id",
    "designs",
    "runner_wall_seconds",
    "device",
    "gpu_uuid",
    "driver_version",
    "host_name",
    "runner_id",
    "torch_version",
    "cuda_runtime_version",
    "environment_identity",
    "source_revision",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "weights_verification_state",
    "seed_delivered",
    "used_seed",
    "persistent_request_dir",
    "requested_hotspot_residues",
    "delivered_hotspot_positions",
    "target_input_sha256",
    "target_input_format",
    "request_target_sha256",
    "torchvision_version",
    "pytorch_lightning_version",
    "torchmetrics_version",
    "nvidia_cublas_version",
    "cuequivariance_ops_version",
    "cuequivariance_ops_torch_version",
    "dependency_smoke_ok",
    "torchvision_extension_ops_available",
    "predict_import_path",
    "predict_class",
    "use_kernels",
    "non_kernel_triangle_smoke_ok",
    "non_kernel_triangle_classes",
)
SOURCE_NONEMPTY_TEXT_FIELDS = (
    "request_id",
    "device",
    "torch_version",
    "environment_identity",
    "source_revision",
    "checkpoint_sha256",
    "weights_verification_state",
    "persistent_request_dir",
    "target_input_sha256",
    "target_input_format",
    "request_target_sha256",
    "torchvision_version",
    "pytorch_lightning_version",
    "torchmetrics_version",
    "nvidia_cublas_version",
    "cuequivariance_ops_version",
    "cuequivariance_ops_torch_version",
    "predict_import_path",
    "predict_class",
)
SOURCE_NULLABLE_TEXT_FIELDS = (
    "gpu_uuid",
    "driver_version",
    "host_name",
    "runner_id",
    "cuda_runtime_version",
)
SOURCE_BOOLEAN_FIELDS = (
    "seed_delivered",
    "dependency_smoke_ok",
    "torchvision_extension_ops_available",
    "use_kernels",
    "non_kernel_triangle_smoke_ok",
)

# The identity the application reports on every answer. BoltzGen's weights are
# pulled into the runner's cache rather than pinned here, so the digest the
# runner hashed is the only weights identity a row can carry.
REQUIRED_RESPONSE_FIELDS = (
    "device",
    "source_revision",
    "environment_identity",
    "checkpoint_sha256",
)
OPTIONAL_RESPONSE_FIELDS = (
    # Added by the shared batch dispatcher after it compares the service receipt
    # with the resolved adapter identity. It is not an invented service echo.
    "identity_verification",
    "checkpoint_bytes",
    "weights_verification_state",
    "gpu_uuid",
    "driver_version",
    "host_name",
    "runner_id",
    "torch_version",
    "cuda_runtime_version",
    "use_kernels",
    "persistent_request_dir",
    "runner_wall_seconds",
    "requested_hotspot_residues",
    "delivered_hotspot_positions",
    "target_input_sha256",
    "target_input_format",
    "request_target_sha256",
    "torchvision_version",
    "pytorch_lightning_version",
    "torchmetrics_version",
    "nvidia_cublas_version",
    "cuequivariance_ops_version",
    "cuequivariance_ops_torch_version",
    "dependency_smoke_ok",
    "torchvision_extension_ops_available",
    "predict_import_path",
    "predict_class",
    "non_kernel_triangle_smoke_ok",
    "non_kernel_triangle_classes",
)

AdapterError = base.AdapterError


def resolve_endpoint(value: str | None) -> str:
    """Return the deployed application URL for this tool's own environment key."""
    return base.resolve_endpoint(value, FAL_URL_ENVIRONMENT_KEY)


def validate_request_values(
    *,
    count: int,
    binder_minimum_length: int,
    binder_maximum_length: int,
    seed: int,
    generator_id: str,
) -> None:
    """Refuse a request the application would reject, before it is paid for."""
    if not 1 <= count <= MAXIMUM_DESIGNS:
        raise AdapterError(
            f"--count is {count}; the application accepts 1 to {MAXIMUM_DESIGNS} designs per "
            "request"
        )
    for label, value in (
        ("--binder-length-min", binder_minimum_length),
        ("--binder-length-max", binder_maximum_length),
    ):
        if not MINIMUM_BINDER_LENGTH <= value <= MAXIMUM_BINDER_LENGTH:
            raise AdapterError(
                f"{label} is {value}; the application accepts {MINIMUM_BINDER_LENGTH} to "
                f"{MAXIMUM_BINDER_LENGTH}"
            )
    if binder_minimum_length > binder_maximum_length:
        raise AdapterError(
            f"--binder-length-min is {binder_minimum_length} and --binder-length-max is "
            f"{binder_maximum_length}; the range is empty"
        )
    if not 0 <= seed <= base.MAXIMUM_SEED:
        raise AdapterError(f"--seed is {seed}; the application accepts 0 to {base.MAXIMUM_SEED}")
    if base.IDENTIFIER_RE.fullmatch(generator_id) is None:
        raise AdapterError(f"--generator-id is not a plain identifier: {generator_id}")


def _pdb_target_context(
    payload: bytes,
    *,
    target_chain: str,
    hotspots: list[int],
    label: str,
) -> tuple[str, list[int]]:
    """Return the sent target sequence and the service's PDB hotspot positions.

    The campaign target-preparer emits PDB.  The deployed service keeps PDB
    input bytes unchanged, then maps hotspot author numbers onto one-based
    coordinate-order positions.  This mirrors its ``_pdb_chain_residue_numbers``
    and ``_hotspot_positions`` functions so the reported mapping can be bound
    to the exact target body sent over the wire.
    """
    try:
        target_sequence = evidence.chain_sequence(
            payload,
            "pdb",
            target_chain,
            f"{label} target chain",
        )
        text = payload.decode("utf-8")
    except (evidence.EvidenceError, UnicodeError) as exc:
        raise AdapterError(str(exc)) from exc

    ordered_numbers: list[int] = []
    seen_residues: set[tuple[int, str]] = set()
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[21:22] != target_chain:
            continue
        try:
            number = int(line[22:26])
        except ValueError:
            continue
        key = (number, line[26:27])
        if key not in seen_residues:
            seen_residues.add(key)
            ordered_numbers.append(number)
    if not ordered_numbers:
        raise AdapterError(f"{label} has no coordinate records on target chain {target_chain!r}")

    number_to_position: dict[int, int] = {}
    for position, number in enumerate(ordered_numbers, start=1):
        if number in number_to_position:
            raise AdapterError(
                f"{label} target chain repeats residue number {number}; BoltzGen does not "
                "support insertion-code hotspot identities"
            )
        number_to_position[number] = position
    missing = [number for number in hotspots if number not in number_to_position]
    if missing:
        raise AdapterError(f"{label} target chain lacks hotspot residues {missing}")
    return target_sequence, [number_to_position[number] for number in hotspots]


def _validate_hotspots(hotspots: list[int]) -> None:
    """Refuse a request the service would reject after allocating a runner."""
    if not hotspots:
        raise AdapterError("--hotspots carries no residue number")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in hotspots):
        raise AdapterError("--hotspots must contain integer residue numbers")
    if len(hotspots) > MAXIMUM_HOTSPOTS:
        raise AdapterError(
            f"the design site holds {len(hotspots)} residues and the application accepts at most "
            f"{MAXIMUM_HOTSPOTS}"
        )
    if len(set(hotspots)) != len(hotspots):
        raise AdapterError("--hotspots contains duplicate residue numbers")


def build_payload(
    *,
    identifier: str,
    target_structure: Path,
    target_sha256: str,
    target_chain: str,
    binder_chain: str,
    hotspots: list[int],
    binder_minimum_length: int,
    binder_maximum_length: int,
    count: int,
    seed: int,
) -> dict[str, Any]:
    """Return the JSON body of one generation request."""
    if target_chain == binder_chain:
        raise AdapterError(
            f"the target chain and the binder chain are both {target_chain}; the design pose "
            "carries the binder and the target together and they need two letters"
        )
    for label, value in (("target chain", target_chain), ("binder chain", binder_chain)):
        if base.CHAIN_ID_RE.fullmatch(value) is None:
            raise AdapterError(f"the {label} is {value!r}, which is not a chain id")
    _validate_hotspots(hotspots)
    if not target_structure.is_file():
        raise AdapterError(f"target structure not found: {target_structure}")
    if target_structure.suffix.lower() != ".pdb":
        raise AdapterError(
            f"target structure must be a .pdb file for this adapter's verified PDB input route: "
            f"{target_structure}"
        )
    target_bytes = target_structure.read_bytes()
    try:
        text = target_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise AdapterError(f"target structure is not UTF-8 PDB text: {target_structure}") from exc
    observed = base.sha256_bytes(target_bytes)
    if observed != target_sha256:
        raise AdapterError(
            f"{target_structure} hashes {observed} and the caller recorded {target_sha256}"
        )
    _pdb_target_context(
        target_bytes,
        target_chain=target_chain,
        hotspots=hotspots,
        label=str(target_structure),
    )
    return {
        "request_id": identifier,
        "target_structure_name": target_structure.name,
        "target_structure_text": text,
        "target_chain": target_chain,
        "binder_chain": binder_chain,
        "hotspot_residues": list(hotspots),
        "binder_minimum_length": binder_minimum_length,
        "binder_maximum_length": binder_maximum_length,
        "num_designs": count,
        "budget": count,
        "requested_seed": seed,
    }


def _validate_source_response_fields(response: dict[str, Any]) -> None:
    """Validate the actual ``DesignResponse`` field presence and value types."""
    missing = [field for field in SOURCE_RESPONSE_FIELDS if field not in response]
    if missing:
        raise AdapterError(
            "the application omitted required DesignResponse fields: " + ", ".join(missing)
        )
    for field in SOURCE_NONEMPTY_TEXT_FIELDS:
        value = response[field]
        if not isinstance(value, str) or not value.strip():
            raise AdapterError(f"the application reported invalid {field}: expected non-empty text")
    for field in SOURCE_NULLABLE_TEXT_FIELDS:
        value = response[field]
        if value is not None and not isinstance(value, str):
            raise AdapterError(f"the application reported invalid {field}: expected text or null")
    for field in SOURCE_BOOLEAN_FIELDS:
        if not isinstance(response[field], bool):
            raise AdapterError(f"the application reported invalid {field}: expected a boolean")
    if not isinstance(response["designs"], list):
        raise AdapterError("the application reported invalid designs: expected a list")
    checkpoint_bytes = response["checkpoint_bytes"]
    if (
        not isinstance(checkpoint_bytes, int)
        or isinstance(checkpoint_bytes, bool)
        or checkpoint_bytes < 0
    ):
        raise AdapterError("the application reported invalid checkpoint_bytes")
    used_seed = response["used_seed"]
    if used_seed is not None and (
        not isinstance(used_seed, int) or isinstance(used_seed, bool)
    ):
        raise AdapterError("the application reported invalid used_seed: expected an integer or null")
    classes = response["non_kernel_triangle_classes"]
    if not isinstance(classes, list) or any(
        not isinstance(value, str) or not value.strip() for value in classes
    ):
        raise AdapterError(
            "the application reported invalid non_kernel_triangle_classes: expected text entries"
        )
    # The one source-pinned invocation has no seed argument and always exercises
    # the documented PyTorch fallback before generating.  Do not turn an
    # unexpected service answer into a reproducibility or dependency claim.
    if response["seed_delivered"] is not False or response["used_seed"] is not None:
        raise AdapterError(
            "the application reported delivered seed state, but the served BoltzGen route has "
            "no seed control"
        )
    if response["weights_verification_state"] != "observed-unverified":
        raise AdapterError(
            "the application reported unexpected weights_verification_state "
            f"{response['weights_verification_state']!r}"
        )
    if response["use_kernels"] is not False:
        raise AdapterError("the application did not report the served non-kernel route")
    for field in (
        "dependency_smoke_ok",
        "torchvision_extension_ops_available",
        "non_kernel_triangle_smoke_ok",
    ):
        if response[field] is not True:
            raise AdapterError(f"the application reported {field} false for a completed design")


def check_response_echo(
    response: dict[str, Any],
    *,
    target_sha256: str,
    hotspots: list[int],
    expected_hotspot_positions: list[int],
    request_id: str | None = None,
) -> None:
    """Refuse an answer that the deployed service did not build for this request.

    The service's ``DesignResponse`` has no protocol or returned-chain fields.
    Its target hash, requested hotspot list, one-based mapped positions, runtime
    wall time, and checkpoint digest are the provenance values it does return.
    """
    _validate_source_response_fields(response)
    if request_id is not None and response["request_id"] != request_id:
        raise AdapterError(
            "the application reports request_id "
            f"{response['request_id']!r} and this request sent "
            f"{request_id!r}"
        )
    echoed = response["request_target_sha256"]
    if echoed != target_sha256:
        raise AdapterError(
            f"the application reports request_target_sha256 {echoed} and this request sent "
            f"{target_sha256}"
        )
    requested = response["requested_hotspot_residues"]
    if (
        not isinstance(requested, list)
        or any(not isinstance(value, int) or isinstance(value, bool) for value in requested)
        or requested != hotspots
    ):
        raise AdapterError(
            f"the application recorded hotspot residues {requested} and this request sent "
            f"{hotspots}"
        )
    delivered = response["delivered_hotspot_positions"]
    if (
        not isinstance(delivered, list)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in delivered
        )
        or delivered != expected_hotspot_positions
    ):
        raise AdapterError(
            f"the application mapped {delivered} but the sent PDB maps hotspot residues "
            f"{hotspots} to {expected_hotspot_positions}"
        )
    digest = response["checkpoint_sha256"]
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise AdapterError(
            f"the application reported checkpoint_sha256 {digest!r}, which is not a SHA-256 digest"
        )
    target_input_digest = response["target_input_sha256"]
    if not isinstance(target_input_digest, str) or SHA256_RE.fullmatch(target_input_digest) is None:
        raise AdapterError(
            "the application reported no valid target_input_sha256 for the PDB it passed to "
            "BoltzGen"
        )
    if target_input_digest != target_sha256:
        raise AdapterError(
            "the application reported target_input_sha256 "
            f"{target_input_digest}, but the PDB request body hashes {target_sha256}"
        )
    if response["target_input_format"] != "pdb":
        raise AdapterError(
            "the application reported target_input_format "
            f"{response['target_input_format']!r}; "
            "the served route materializes a PDB input"
        )
    runner_seconds = response["runner_wall_seconds"]
    if (
        not isinstance(runner_seconds, (int, float))
        or isinstance(runner_seconds, bool)
        or not math.isfinite(float(runner_seconds))
        or runner_seconds < 0
    ):
        raise AdapterError(
            f"the application reported runner_wall_seconds {runner_seconds!r}, which is not a "
            "non-negative finite duration"
        )
    # Check the attribution fields before payload files are written. The common
    # receipt writer performs the same required-field check, but it runs after
    # this child has materialized its returned files.
    base.runtime_fields(response, REQUIRED_RESPONSE_FIELDS, OPTIONAL_RESPONSE_FIELDS)


def _decode_payload(
    design: dict[str, Any], *, index: int, payload_key: str, hash_key: str
) -> bytes:
    """Decode one source-contract base64 payload and verify its service digest."""
    encoded = design.get(payload_key)
    if not isinstance(encoded, str) or not encoded:
        raise AdapterError(f"design {index} carries no {payload_key}")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise AdapterError(f"design {index} carries invalid base64 at {payload_key}") from exc
    recorded = design.get(hash_key)
    observed = base.sha256_bytes(payload)
    if (
        not isinstance(recorded, str)
        or SHA256_RE.fullmatch(recorded) is None
        or recorded != observed
    ):
        raise AdapterError(
            f"design {index} {payload_key} hashes {observed} and the service recorded {recorded!r}"
        )
    return payload


def _pdb_binder_sequence(
    pose: bytes,
    *,
    index: int,
    target_chain: str,
    binder_chain: str,
    expected_target_sequence: str,
) -> str:
    """Return the binder sequence in a strict PDB bridge, before it is published.

    The service validates its mmCIF with gemmi before producing this bridge. The
    package's standard-library evidence reader independently verifies coordinate
    records and residue identities before a returned hash is trusted.
    """
    try:
        target_sequence = evidence.chain_sequence(
            pose,
            "pdb",
            target_chain,
            f"design {index} PDB bridge target chain",
        )
        sequence = evidence.chain_sequence(
            pose,
            "pdb",
            binder_chain,
            f"design {index} PDB bridge binder chain",
        )
    except evidence.EvidenceError as exc:
        raise AdapterError(str(exc)) from exc
    if target_sequence != expected_target_sequence:
        raise AdapterError(
            f"design {index} PDB bridge target chain {target_chain!r} spells "
            f"{target_sequence!r}, but the sent target spells {expected_target_sequence!r}"
        )
    if base.SEQUENCE_RE.fullmatch(sequence) is None:
        raise AdapterError(
            f"design {index} PDB bridge binder chain {binder_chain!r} contains a non-canonical "
            "or empty sequence"
        )
    return sequence


def _validate_raw_pose(
    raw_pose: bytes,
    *,
    index: int,
    sequence: str,
    target_chain: str,
    expected_target_sequence: str,
) -> None:
    """Require the returned engine-native mmCIF to contain matching coordinates.

    The standard-library evidence reader parses the native mmCIF. The response
    gives no raw binder-chain echo, because the bridge may rename a unique
    non-target binder chain.  It does preserve the target chain, so the raw
    target must still match the sent target and a separate non-target chain must
    spell the reported binder sequence.
    """
    try:
        atoms = evidence.structure_atoms(
            raw_pose,
            "cif",
            f"design {index} engine-native mmCIF",
        )
    except evidence.EvidenceError as exc:
        raise AdapterError(str(exc)) from exc
    residues = target_prep.chain_residues(atoms)
    if len(residues) < 2 or any(not chain for chain in residues):
        raise AdapterError(
            f"design {index} engine-native mmCIF does not expose the two chains of a complex"
        )
    try:
        raw_target_sequence = evidence.chain_sequence(
            raw_pose,
            "cif",
            target_chain,
            f"design {index} engine-native mmCIF target chain",
        )
    except evidence.EvidenceError as exc:
        raise AdapterError(str(exc)) from exc
    if raw_target_sequence != expected_target_sequence:
        raise AdapterError(
            f"design {index} engine-native mmCIF target chain {target_chain!r} spells "
            f"{raw_target_sequence!r}, but the sent target spells {expected_target_sequence!r}"
        )
    sequences = {
        chain: "".join(
            target_prep.THREE_TO_ONE.get(residue.residue_name.upper(), "X")
            for residue in chain_rows
            if residue.residue_name.upper() != "HOH"
        )
        for chain, chain_rows in residues.items()
        if chain != target_chain
    }
    if sequence not in sequences.values():
        raise AdapterError(
            f"design {index} engine-native mmCIF carries no non-target chain matching its "
            "returned binder sequence"
        )


def _metric_record(design: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Return the source metric object without inventing an undocumented verdict."""
    metrics = design.get("metrics")
    if not isinstance(metrics, dict):
        raise AdapterError(f"design {index} carries no metric object")
    try:
        json.dumps(metrics, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"design {index} metric object cannot be written as JSON") from exc
    return metrics


def decode_designs(
    response: dict[str, Any],
    expected: int,
    *,
    binder_minimum_length: int,
    binder_maximum_length: int,
    target_chain: str,
    binder_chain: str,
    request_id: str,
    expected_target_sequence: str,
) -> list[dict[str, Any]]:
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
        name = design.get("name")
        if (
            not isinstance(name, str)
            or base.RETURNED_NAME_RE.fullmatch(name) is None
            or not name.endswith(".cif")
        ):
            raise AdapterError(
                f"the application returned a CIF name this adapter refuses: {name!r}"
            )
        if name in seen:
            raise AdapterError(f"the application returned {name} twice")
        expected_name = f"{request_id}-{index:03d}.cif"
        if name != expected_name:
            raise AdapterError(
                f"design {index} is named {name!r}, but the service names this request output "
                f"{expected_name!r}"
            )
        seen.add(name)
        sequence = design.get("sequence")
        if not isinstance(sequence, str):
            raise AdapterError(f"design {index} returned no binder sequence")
        if base.SEQUENCE_RE.fullmatch(sequence) is None:
            raise AdapterError(
                f"design {index} returned a binder sequence that is not canonical single-letter "
                "residues"
            )
        if not binder_minimum_length <= len(sequence) <= binder_maximum_length:
            raise AdapterError(
                f"design {index} is {len(sequence)} residues, outside the "
                f"{binder_minimum_length} to {binder_maximum_length} this request asked for"
            )
        pose = _decode_payload(
            design,
            index=index,
            payload_key="pdb_b64",
            hash_key="pdb_sha256",
        )
        raw_pose = _decode_payload(
            design,
            index=index,
            payload_key="cif_b64",
            hash_key="cif_sha256",
        )
        structural_sequence = _pdb_binder_sequence(
            pose,
            index=index,
            target_chain=target_chain,
            binder_chain=binder_chain,
            expected_target_sequence=expected_target_sequence,
        )
        if sequence != structural_sequence:
            raise AdapterError(
                f"design {index} returned sequence {sequence!r}, but its PDB binder chain "
                f"{binder_chain!r} spells {structural_sequence!r}"
            )
        _validate_raw_pose(
            raw_pose,
            index=index,
            sequence=sequence,
            target_chain=target_chain,
            expected_target_sequence=expected_target_sequence,
        )
        persistent_artifact_dir = design.get("persistent_artifact_dir")
        if not isinstance(persistent_artifact_dir, str) or not persistent_artifact_dir.strip():
            raise AdapterError(f"design {index} carries no persistent artifact directory")
        decoded.append(
            {
                "design_index": index,
                "design_name": name,
                "pose": pose,
                "pose_format": POSE_FORMAT,
                "raw_pose": raw_pose,
                "raw_pose_format": RAW_POSE_FORMAT,
                "metrics": _metric_record(design, index=index),
                "native_filter_state": UNREPORTED_FILTER_STATE,
                "output_kind": base.OUTPUT_KIND_SEQUENCE,
                "binder_chain_id": binder_chain,
                "target_chain_id": target_chain,
                "binder_length": len(sequence),
                "sequence": sequence,
                "fasta": f">{Path(name).stem}\n{sequence}\n".encode("ascii"),
                "persistent_artifact_dir": persistent_artifact_dir,
            }
        )
    return decoded


def write_index_and_files(out_dir: Path, decoded: list[dict[str, Any]]) -> dict[str, Any]:
    """Write every returned file under one directory and index what was written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in decoded:
        stem = f"design-{record['design_index']:03d}"
        pose_path = out_dir / f"{stem}.{POSE_FORMAT}"
        pose_path.write_bytes(record["pose"])
        raw_path = out_dir / f"{stem}.raw.{RAW_POSE_FORMAT}"
        raw_path.write_bytes(record["raw_pose"])
        fasta_path = out_dir / f"{stem}.fasta"
        fasta_path.write_bytes(record["fasta"])
        metrics_path = out_dir / f"{stem}.metrics.json"
        base.write_json(metrics_path, record["metrics"])
        metrics = metrics_path.read_bytes()
        rows.append(
            {
                "design_index": record["design_index"],
                "design_name": record["design_name"],
                "pose_file": pose_path.name,
                "pose_format": POSE_FORMAT,
                "pose_sha256": base.sha256_bytes(record["pose"]),
                "raw_pose_file": raw_path.name,
                "raw_pose_format": RAW_POSE_FORMAT,
                "raw_pose_sha256": base.sha256_bytes(record["raw_pose"]),
                "sequence_file": fasta_path.name,
                "sequence_sha256": base.sha256_bytes(record["fasta"]),
                "sequence": record["sequence"],
                "metrics_file": metrics_path.name,
                "metrics_sha256": base.sha256_bytes(metrics),
                "native_filter_state": record["native_filter_state"],
                "output_kind": record["output_kind"],
                "binder_chain_id": record["binder_chain_id"],
                "target_chain_id": record["target_chain_id"],
                "binder_length": record["binder_length"],
                "persistent_artifact_dir": record["persistent_artifact_dir"],
            }
        )
    index = {"schema_version": 1, "designs": rows}
    base.write_json(out_dir / base.DEFAULT_INDEX_NAME, index)
    return index


def publish_native_metrics(
    index_row: dict[str, Any], out_dir: Path, phase_dir: Path, candidate_id: str
) -> dict[str, Any]:
    """Copy one native metric record into the stage layout and hash what was written.

    The skeleton publishes the pose, the raw pose and the FASTA. BoltzGen returns
    a fourth file the other hosted generators do not: the metric row its own
    ranking wrote. The source contract does not attach a filter verdict to that
    generic object, so the candidate row records ``unreported`` separately.
    """
    name = index_row.get("metrics_file")
    if not name or base.RETURNED_NAME_RE.fullmatch(str(name)) is None:
        raise AdapterError(f"the returned index names a metric file this adapter refuses: {name!r}")
    source = out_dir / str(name)
    if not source.is_file():
        raise AdapterError(f"the dispatch child did not write {source}")
    observed = base.sha256_file(source)
    if observed != index_row.get("metrics_sha256"):
        raise AdapterError(
            f"{source} hashes {observed} and the returned index records "
            f"{index_row.get('metrics_sha256')}"
        )
    destination = phase_dir / DEFAULT_METRICS_SUBDIR / f"{candidate_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    return {"path": destination.resolve(), "sha256": observed}


def dispatch(args: argparse.Namespace) -> int:
    """Post one generation request and write the files the runner returned."""
    endpoint = resolve_endpoint(args.fal_url)
    validate_request_values(
        count=args.count,
        binder_minimum_length=args.binder_length_min,
        binder_maximum_length=args.binder_length_max,
        seed=args.seed,
        generator_id=DEFAULT_GENERATOR_ID,
    )
    hotspots = [int(value) for value in str(args.hotspots).split(",") if value.strip()]
    if not hotspots:
        raise AdapterError("--hotspots carries no residue number")
    payload = build_payload(
        identifier=base.request_id(args.request_id, DEFAULT_GENERATOR_ID),
        target_structure=args.target_structure.expanduser(),
        target_sha256=args.target_sha256,
        target_chain=args.target_chain,
        binder_chain=args.binder_chain,
        hotspots=hotspots,
        binder_minimum_length=args.binder_length_min,
        binder_maximum_length=args.binder_length_max,
        count=args.count,
        seed=args.seed,
    )
    expected_target_sequence, expected_hotspot_positions = _pdb_target_context(
        payload["target_structure_text"].encode("utf-8"),
        target_chain=args.target_chain,
        hotspots=hotspots,
        label="sent request PDB",
    )
    started = time.monotonic()
    response = base.post(endpoint, payload, args.timeout_seconds, args.credential_env)
    seconds = time.monotonic() - started
    check_response_echo(
        response,
        target_sha256=args.target_sha256,
        hotspots=hotspots,
        expected_hotspot_positions=expected_hotspot_positions,
        request_id=payload["request_id"],
    )
    decoded = decode_designs(
        response,
        args.count,
        binder_minimum_length=args.binder_length_min,
        binder_maximum_length=args.binder_length_max,
        target_chain=args.target_chain,
        binder_chain=args.binder_chain,
        request_id=payload["request_id"],
        expected_target_sequence=expected_target_sequence,
    )
    out_dir = args.out_dir.expanduser()
    write_index_and_files(out_dir, decoded)
    base.write_receipt(
        args.receipt.expanduser(),
        # BoltzGen's DesignResponse names this measured duration
        # ``runner_wall_seconds``. PXDesign's shared receipt writer consumes the
        # older ``seconds`` spelling because other deployed services use it.
        # Normalize only this call site and retain both source fields in the
        # receipt/index; changing the shared writer would reinterpret other APIs.
        {**response, "seconds": response["runner_wall_seconds"]},
        endpoint=endpoint,
        client_wall_seconds=seconds,
        requested_seed=args.seed,
        design_count=len(decoded),
        required=REQUIRED_RESPONSE_FIELDS,
        optional=OPTIONAL_RESPONSE_FIELDS,
    )
    base.write_json(
        args.receipt.expanduser().with_suffix(".echo.json"),
        {
            "design_protocol": DESIGN_PROTOCOL,
            "request_id": response.get("request_id"),
            "request_target_sha256": response.get("request_target_sha256"),
            "requested_hotspot_residues": response.get("requested_hotspot_residues"),
            "delivered_hotspot_positions": response.get("delivered_hotspot_positions"),
            "target_input_format": response.get("target_input_format"),
            "target_input_sha256": response.get("target_input_sha256"),
            "runner_wall_seconds": response.get("runner_wall_seconds"),
        },
    )
    print(
        f"{TOOL_LABEL} dispatch: designs={len(decoded)} seconds={seconds:.1f} "
        f"device={response['device']} out_dir={out_dir}"
    )
    return 0


def probe(args: argparse.Namespace) -> int:
    """Spawn the child that asks the deployed application to report its runtime."""
    if not args.acknowledge_cost:
        raise AdapterError(
            "probe starts a GPU runner on your own deployment and therefore costs money, so "
            "it needs --acknowledge-cost. Run toolcheck instead for the free readiness check, "
            "which sends no request and reads no credential"
        )
    endpoint = resolve_endpoint(args.fal_url)
    base.run_external(
        base.child_argv(
            args,
            PROBE_CHILD_COMMAND,
            endpoint,
            "--timeout-seconds",
            str(args.timeout_seconds),
            *(["--request-id", args.request_id] if args.request_id else []),
            script=DISPATCH_SCRIPT,
        ),
        "probe",
        TOOL_LABEL,
    )
    return 0


def dispatch_probe(args: argparse.Namespace) -> int:
    """Post one toolcheck request and print the runtime the application reported."""
    endpoint = resolve_endpoint(args.fal_url)
    started = time.monotonic()
    response = base.post(
        endpoint.rstrip("/") + base.TOOLCHECK_PATH,
        {"request_id": base.request_id(args.request_id, "toolcheck")},
        args.timeout_seconds,
        args.credential_env,
    )
    seconds = time.monotonic() - started
    fields = base.runtime_fields(response, REQUIRED_RESPONSE_FIELDS, OPTIONAL_RESPONSE_FIELDS)
    print(f"{TOOL_LABEL} probe: device {fields['device']}")
    print(f"{TOOL_LABEL} probe: environment {fields['environment_identity']}")
    print(f"{TOOL_LABEL} probe: source {fields['source_revision']}")
    print(f"{TOOL_LABEL} probe: checkpoint {fields['checkpoint_sha256']}")
    print(f"{TOOL_LABEL} probe: took {seconds:.1f} seconds")
    return 0


def toolcheck(args: argparse.Namespace) -> int:
    """Report this adapter's own readiness without sending anything."""
    ready = True
    try:
        resolve_endpoint(args.fal_url)
        print(f"{TOOL_LABEL} adapter: endpoint {fal_invocation.REDACTED_FAL_URL}")
    except AdapterError as exc:
        ready = False
        print(f"{TOOL_LABEL} adapter: endpoint unresolved: {exc}")
    try:
        route, credential_env = base.resolve_credential_route(args)
        print(f"{TOOL_LABEL} adapter: credential route {route} through {credential_env}")
    except AdapterError as exc:
        ready = False
        print(f"{TOOL_LABEL} adapter: credential route unavailable: {exc}")
    print(
        f"{TOOL_LABEL} adapter: dispatch child {args.client_python} {DISPATCH_SCRIPT} "
        f"{DISPATCH_COMMAND}"
    )
    print(
        f"{TOOL_LABEL} adapter: request ceiling count 1-{MAXIMUM_DESIGNS}, binder length "
        f"{MINIMUM_BINDER_LENGTH}-{MAXIMUM_BINDER_LENGTH}, hotspots 1-{MAXIMUM_HOTSPOTS}, "
        f"protocol {DESIGN_PROTOCOL}"
    )
    print(
        f"{TOOL_LABEL} adapter: a phase larger than {MAXIMUM_DESIGNS} designs is dispatched as "
        "whole requests, one per batch, each with its own request id and receipt"
    )
    print(
        f"{TOOL_LABEL} adapter: BoltzGen 0.3.2 has no CLI seed control, so the runner reports "
        "seed_delivered false, a row cannot reproduce its design, and every request in a split "
        "phase carries the same seed"
    )
    print(
        f"{TOOL_LABEL} adapter: each row carries the returned native metric object. The service "
        "declares no schema-level filter verdict, so native_filter_state is unreported"
    )
    print(
        f"{TOOL_LABEL} adapter: the code is MIT and the checkpoint terms are unverified; settle "
        "them before a commercial campaign"
    )
    print(
        f"{TOOL_LABEL} adapter: cost basis {base.COST_BASIS}; no measurement prices this provider"
    )
    print(
        f"{TOOL_LABEL} adapter: this check sends no request and reads no credential; probe is "
        "the paid subcommand and it needs --acknowledge-cost",
        flush=True,
    )
    if not ready:
        print(f"{TOOL_LABEL} adapter: not ready to dispatch", file=sys.stderr)
    return 0 if ready else 1


def run(args: argparse.Namespace) -> int:
    """Compose the request, dispatch one phase, and write the stage outputs."""
    batches = base.request_batches(args.count, MAXIMUM_DESIGNS)
    validate_request_values(
        count=batches[0],
        binder_minimum_length=args.binder_length_min,
        binder_maximum_length=args.binder_length_max,
        seed=args.seed,
        generator_id=args.generator_id,
    )
    if base.CHAIN_ID_RE.fullmatch(args.binder_chain) is None:
        raise AdapterError(
            f"--binder-chain is {args.binder_chain}; a chain ID is one letter or digit"
        )
    endpoint = resolve_endpoint(args.fal_url)
    base.resolve_credential_route(args)

    manifest, manifest_source = base.load_target_manifest(args)
    target_id = str(manifest["target_id"])
    chain = args.target_chain or str(manifest["design_target_chain_id"])
    if base.CHAIN_ID_RE.fullmatch(chain) is None:
        raise AdapterError(f"the design target chain is {chain!r}, which is not a chain id")
    if chain == args.binder_chain:
        raise AdapterError(
            f"the target chain and --binder-chain are both {chain}; they name two different "
            "chains of the design pose, which carries the binder and the target together"
        )
    structure_path, structure_sha256 = base.normalized_structure(manifest, manifest_source)
    hotspots = base.site_residue_numbers(manifest, manifest_source, chain)
    if len(hotspots) > MAXIMUM_HOTSPOTS:
        raise AdapterError(
            f"the design site holds {len(hotspots)} residues and the application accepts at most "
            f"{MAXIMUM_HOTSPOTS}"
        )

    attempt_dir, phase_dir, work_dir, manifest_path = base.phase_paths(args)
    out_dir = work_dir / "returned"
    base.refuse_populated_output(out_dir)

    results = base.dispatch_batches(
        args,
        endpoint=endpoint,
        out_dir=out_dir,
        work_dir=work_dir,
        batches=batches,
        common_values=[
            "--target-structure",
            str(structure_path),
            "--target-sha256",
            structure_sha256,
            "--target-chain",
            chain,
            "--binder-chain",
            args.binder_chain,
            "--hotspots",
            ",".join(str(number) for number in hotspots),
            "--binder-length-min",
            str(args.binder_length_min),
            "--binder-length-max",
            str(args.binder_length_max),
        ],
        script=DISPATCH_SCRIPT,
        tool=TOOL_LABEL,
        seed_for_batch=base.same_seed,
    )

    rows: list[dict[str, Any]] = []
    position = 0
    for batch_number, (receipt, returned, batch_dir) in enumerate(results):
        runtime = base.runtime_fields(receipt, REQUIRED_RESPONSE_FIELDS, OPTIONAL_RESPONSE_FIELDS)
        receipt_path = work_dir / f"batch-{batch_number:03d}-{args.receipt_name}"
        for index_row in returned:
            candidate_id = f"{args.generator_id}-{position:03d}"
            published = base.publish_design_files(
                index_row, batch_dir, phase_dir, candidate_id, binder_chain=args.binder_chain
            )
            pose = published["pose_file"]
            raw_pose = published["raw_pose_file"]
            sequence_file = published["sequence_file"]
            if sequence_file is None:
                raise AdapterError(f"design {position} indexed no sequence file")
            if raw_pose is None:
                raise AdapterError(f"design {position} indexed no engine-native mmCIF")
            metrics = publish_native_metrics(index_row, batch_dir, phase_dir, candidate_id)
            sequence = str(index_row["sequence"])
            rows.append(
                {
                    "target_id": target_id,
                    "target_sha256": str(manifest["target_sha256"]),
                    "candidate_id": candidate_id,
                    "parent_candidate_id": None,
                    "origin_generator": args.generator_id,
                    **backbone_lineage(candidate_id, args.generator_id),
                    "generator_mode": GENERATOR_MODE,
                    "runner_protocol": base.RUNNER_PROTOCOL,
                    "sequence_designer": args.generator_id,
                    "generator_seed": None,
                    "requested_seed": args.seed,
                    "tool_seed": receipt.get("used_seed"),
                    "seed_delivered": bool(receipt.get("seed_delivered", False)),
                    "sequence_path": str(sequence_file["path"]),
                    "sequence_sha256": base.sha256_bytes(sequence.encode("ascii")),
                    "sequence_length": len(sequence),
                    "backbone_only": False,
                    "structure_path": str(manifest["source_structure_path"]),
                    "structure_sha256": str(manifest["target_sha256"]),
                    "design_pose_path": str(pose["path"]),
                    "design_pose_sha256": pose["sha256"],
                    "raw_design_pose_path": str(raw_pose["path"]),
                    "raw_design_pose_sha256": raw_pose["sha256"],
                    "native_metrics_path": str(metrics["path"]),
                    "native_metrics_sha256": metrics["sha256"],
                    "native_filter_state": index_row["native_filter_state"],
                    "persistent_artifact_dir": index_row["persistent_artifact_dir"],
                    "residue_map_sha256": str(manifest["residue_map_sha256"]),
                    "optimization_round": 0,
                    "last_optimizer": None,
                    "status": base.CANDIDATE_STATUS,
                    "stage_id": args.stage,
                    "design_index": position,
                    "dispatch_batch": batch_number,
                    "design_name": index_row["design_name"],
                    "design_protocol": DESIGN_PROTOCOL,
                    "binder_chain_id": args.binder_chain,
                    "returned_binder_chain_id": index_row["binder_chain_id"],
                    "design_pose_relabelled": pose["relabelled"],
                    "returned_pose_sha256": pose["returned_sha256"],
                    "binder_length": index_row["binder_length"],
                    "requested_binder_length_min": args.binder_length_min,
                    "requested_binder_length_max": args.binder_length_max,
                    "target_chain_id": base.swapped_chain(
                        str(index_row["target_chain_id"]),
                        str(index_row["binder_chain_id"]),
                        str(args.binder_chain),
                    )
                    if pose["relabelled"]
                    else index_row["target_chain_id"],
                    "returned_target_chain_id": index_row["target_chain_id"],
                    "declared_target_chain_id": chain,
                    "target_manifest_path": str(manifest_source),
                    "input_structure_path": str(structure_path),
                    "input_structure_sha256": structure_sha256,
                    "hotspot_residues": hotspots,
                    "fal_endpoint": endpoint,
                    "fal_receipt_path": str(receipt_path.resolve()),
                    "runtime_wall_seconds": receipt.get("runner_wall_seconds"),
                    "cost_basis": base.COST_BASIS,
                    **runtime,
                }
            )
            position += 1
    base.write_jsonl(manifest_path, rows)
    print(
        f"{TOOL_LABEL} adapter: phase={args.phase} target={target_id} candidates={len(rows)} "
        f"requests={len(batches)} seed_delivered={rows[0]['seed_delivered']} "
        f"device={rows[0]['device']} manifest={manifest_path} "
        f"target_manifest={manifest_source}"
    )
    return 0


def parse_outputs(args: argparse.Namespace) -> int:
    """Check the phase outputs this stage declares.

    Every hosted generator declares its outputs the same way, so this is the
    skeleton's function rather than a third copy of it.
    """
    return base.parse_outputs(args)


def add_layout_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generator-id", default=DEFAULT_GENERATOR_ID)
    parser.add_argument("--binder-chain", default=DEFAULT_BINDER_CHAIN)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--work-subdir", default=DEFAULT_WORK_SUBDIR)
    parser.add_argument("--receipt-name", default=base.DEFAULT_RECEIPT_NAME)


def add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--binder-length-min",
        type=int,
        required=True,
        help=(
            "Shortest binder the application may return. It accepts "
            f"{MINIMUM_BINDER_LENGTH} to {MAXIMUM_BINDER_LENGTH}."
        ),
    )
    parser.add_argument(
        "--binder-length-max",
        type=int,
        required=True,
        help=(
            "Longest binder the application may return. It accepts "
            f"{MINIMUM_BINDER_LENGTH} to {MAXIMUM_BINDER_LENGTH}."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=base.DEFAULT_SEED,
        help=(
            "Requested seed, recorded on every row. BoltzGen 0.3.2 has no CLI seed control, so "
            "the runner reports seed_delivered false and every request in a split phase carries "
            "this same value."
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
        default=base.DEFAULT_TIMEOUT_SECONDS,
        help=f"Request timeout. Defaults to {base.DEFAULT_TIMEOUT_SECONDS}.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate BoltzGen designs on a fal deployment.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "toolcheck", help="Report readiness without sending a request."
    )
    base.add_route_arguments(check_parser, FAL_URL_ENVIRONMENT_KEY)

    run_parser = subparsers.add_parser("run", help="Generate one phase of designs on fal.")
    base.add_route_arguments(run_parser, FAL_URL_ENVIRONMENT_KEY)
    base.add_stage_arguments(run_parser)
    base.add_target_arguments(run_parser)
    add_request_arguments(run_parser)
    add_layout_arguments(run_parser)

    parse_parser = subparsers.add_parser("parse", help="Parse the outputs of one completed phase.")
    base.add_route_arguments(parse_parser, FAL_URL_ENVIRONMENT_KEY)
    base.add_stage_arguments(parse_parser)

    probe_parser = subparsers.add_parser(
        "probe", help="Ask the deployment to report its runtime. This costs money."
    )
    base.add_route_arguments(probe_parser, FAL_URL_ENVIRONMENT_KEY)
    probe_parser.add_argument("--request-id", default=None)
    probe_parser.add_argument("--timeout-seconds", type=int, default=base.DEFAULT_TIMEOUT_SECONDS)
    probe_parser.add_argument("--acknowledge-cost", action="store_true")

    dispatch_parser = subparsers.add_parser(
        DISPATCH_COMMAND,
        help="Child of run. Posts the verified PDB target route. Not for direct use.",
        description=(
            "Child of run. The verified BoltzGen dispatch route accepts a prepared PDB target "
            "so it can bind the target input hash and hotspot positions."
        ),
    )
    dispatch_parser.add_argument("--fal-url", default=None)
    base.add_dispatch_arguments(dispatch_parser)
    dispatch_parser.add_argument("--target-sha256", required=True)
    dispatch_parser.add_argument("--binder-chain", required=True)
    add_request_arguments(dispatch_parser)

    probe_child_parser = subparsers.add_parser(
        PROBE_CHILD_COMMAND, help="Child of probe. Posts the toolcheck. Not for direct use."
    )
    probe_child_parser.add_argument("--fal-url", default=None)
    probe_child_parser.add_argument("--request-id", default=None)
    probe_child_parser.add_argument(
        "--timeout-seconds", type=int, default=base.DEFAULT_TIMEOUT_SECONDS
    )
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
        print(f"{TOOL_LABEL} adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
