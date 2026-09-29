#!/usr/bin/env python3
"""Generate FreeBindCraft binder designs on an operator-deployed fal application.

This wrapper fills the `freebindcraft-generator` slot. FreeBindCraft is the
MIT-licensed BindCraft fork at cytokineking/FreeBindCraft that drops the
PyRosetta requirement: relaxation runs through OpenMM, side-chain packing
through FASPR, and shape complementarity through the bundled `sc` binary,
behind the entry script's `--no-pyrosetta` flag. One design carries a structure
and a sequence, so the tool fills the codesign-generator role and needs no
sequence-design stage after it.

The tool has a local route as well as a hosted one, and this package ships only
the hosted one. A local route needs a FreeBindCraft checkout, a GPU, and the
5.6 GB AlphaFold 2 multimer parameter archive on the machine running the stage,
and no check in this package can confirm any of the three. `pxdesign_generator`
is hosted-only for the same reason and this module follows it. The local route
was dropped deliberately; it is not missing by oversight.

The module reads the target manifest the `target-preparer` stage published,
posts one design-phase request per batch to a deployment the operator names, and
writes receipt-owned outputs into the current attempt directory:

  <attempt>/<phase>/poses/<candidate_id>.pdb          the design pose
  <attempt>/<phase>/sequences/<candidate_id>.fasta    the MPNN sequence
  <attempt>/<phase>/candidate-manifest.jsonl          one row per candidate
  <attempt>/<phase>/freebindcraft/                    the returned files, the
                                                      provider evidence, and the
                                                      request receipts

Eight properties of the route this wrapper handles rather than hides.

**The endpoint has no default.** Pass --fal-url, or set FREEBINDCRAFT_FAL_URL,
with the application URL of your own deployment. A URL baked into this file
would name somebody else's account, and the request carries an authorization
header.

**The credential enters neither this process nor an argument list.** `run` and
`probe` build a child command through `clients/fal_invocation`, and only that
child opens a socket.

**What this publishes is a staging set, not a set of accepted binders.** Behind
`--no-pyrosetta` the fork returns a fixed number for eight interface fields, and
four of them carry an active threshold in the fork's own default filters that
each constant passes. The tool's accept verdict is therefore not BindCraft's
published gate. Every row carries `export_set` staging, `retention`
(generator_accepted or generator_rejected), the filter columns behind a drop,
and `staging_note`. `status` stays `generated`, which is this lane's word for a
candidate nothing has screened. No row of this set may be published as an
accepted binder.

**The application classifies, and this wrapper checks the classification.** The
deployment walks Accepted/ and Rejected/, reads the Accepted/Ranked rank table,
reads rejected_mpnn_full_stats.csv, and returns each design already carrying its
verdict and its filter columns. This wrapper re-derives the classification from
the failure columns and re-parses the design name against the grammar, and
refuses an answer whose own fields disagree. Trusting the classification because
the runner asserted it is how an unverifiable row reaches a manifest.

**The tool names its own designs and its own seed.** A design is
`<binder_name>_l<length>_s<seed>_mpnn<variant>`, and the staged pose adds
`_model<k>`. The served contract carries no seed field at all, so the trajectory
seed is the tool's and this wrapper reads it back out of the name. Every row
records `generator_seed` from the name, `requested_seed` null, and
`seed_delivered` false.

**A phase larger than one request is split into whole requests.** The
application caps one request at 32 designs and the published roster asks each
generator for fifty backbones, so `run` dispatches ceil(count / 32) requests.
Each batch carries its own request id and its own binder-name suffix. The
deployment derives its persistent run directory from ten request fields,
including both of those values, the target digest, settings texts, and requested
count. The ordinary `[32, 18]` split therefore has no known collision. Every
row records `dispatch_batch`.

**The request count does not fix the answer count.** `number_of_final_designs`
sizes the design loop's stop condition. The loop can overshoot it inside one
batch and can halt below it at the advanced settings' trajectory cap. A phase
that stages fewer designs than it asked for publishes what ran and says so,
because discarding a round that already paid for its designs helps nobody. A
phase that stages more publishes the tool's kept designs by rank first, unless
--stage-all hands the screen the whole set.

**The published pose carries the campaign's chain letters.** The application
reserves chain B for the designed binder and refuses a target on that letter. A
campaign that designs its binder on B needs no change and the pose is written
through unrelabelled. A campaign that designs its binder on another letter has
the two letters swapped in column 22 of the PDB pose and nothing else changed,
because the sequence designer downstream is handed the campaign's
`--design-chain` on its command line rather than reading it off the row. Every
row records `binder_chain_id`, `returned_binder_chain_id`, `target_chain_id` and
`design_pose_relabelled`, so a reader can tell which branch ran. A campaign
whose design target chain is B is refused by name rather than relabelled behind
the operator's back.

**The cost basis is unpriced.** No measurement in this package prices
FreeBindCraft on any provider, so the receipt records `cost_basis: unpriced` and
no number.

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
import glob
import gzip
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from claude_binder.adapters import pxdesign_generator as base
from claude_binder.adapters.candidate_lineage import backbone_lineage
from claude_binder.clients import fal_invocation
from claude_binder.paths import package_file


DISPATCH_SCRIPT = package_file("adapters", "freebindcraft_generator.py")
DISPATCH_COMMAND = base.DISPATCH_COMMAND
PROBE_CHILD_COMMAND = base.PROBE_CHILD_COMMAND
FAL_URL_ENVIRONMENT_KEY = "FREEBINDCRAFT_FAL_URL"
TOOL_LABEL = "freebindcraft"

DEFAULT_GENERATOR_ID = "freebindcraft"
DEFAULT_ADAPTER_ID = "freebindcraft-generator"
DEFAULT_BINDER_NAME = "freebindcraft"
DEFAULT_WORK_SUBDIR = "freebindcraft"
DEFAULT_BINDER_CHAIN = "B"
DEFAULT_EVIDENCE_SUBDIR = "provider-evidence"
DEFAULT_INDEX_NAME = base.DEFAULT_INDEX_NAME
# The fal request ceiling this deployment was built against is 7,200 seconds and
# its child deadline is 6,600, so a client timeout under the ceiling would give
# up on a request the runner is still serializing.
DEFAULT_TIMEOUT_SECONDS = 7200
OBJECT_TIMEOUT_SECONDS = 300

# The request bounds the deployed application declares on its own input model.
MAXIMUM_FINAL_DESIGNS = 32
MAXIMUM_STAGING_DESIGNS = 256
MINIMUM_BINDER_LENGTH = 5
MAXIMUM_BINDER_LENGTH = 1000
# The design site reaches the request as a comma-separated list of bare residue
# numbers inside one 4,096-character field, so the application states no residue
# count of its own. This is the shared site reader's ceiling, which is stricter
# than that field, and it is named here so nobody reads it as FreeBindCraft's.
MAXIMUM_HOTSPOTS = base.MAXIMUM_HOTSPOTS
SERVED_HOTSPOT_FIELD_CHARACTERS = 4096

# The chain the application reserves for the designed binder. It refuses a
# request whose target chain is this letter.
SERVED_BINDER_CHAIN_ID = "B"
# The binder name becomes the stem of every design name the tool writes, so the
# application holds it to its own grammar and refuses one that already carries
# the parts the tool appends.
BINDER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
MODEL_SUFFIX_RE = re.compile(r"_model\d+$")
DESIGN_NAME_RE = re.compile(r"^(?P<base>.+)_l(?P<length>\d+)_s(?P<seed>\d+)_mpnn(?P<variant>\d+)$")

PYROSETTA_BYPASS_FLAG = "--no-pyrosetta"
RUNTIME_FLAGS = (PYROSETTA_BYPASS_FLAG,)
GENERATOR_MODE = "sequence-structure-codesign"
EXPORT_SET = "staging"
RETENTION_ACCEPTED = "generator_accepted"
RETENTION_REJECTED = "generator_rejected"
RETENTIONS = (RETENTION_ACCEPTED, RETENTION_REJECTED)

# Behind --no-pyrosetta the fork's functions/pr_alternative_utils.py returns a
# fixed number for each of these, under a comment naming them placeholders
# chosen to pass active filters. Four sit under an active threshold in the
# fork's default_filters.json and all four pass it, so under the shipped
# defaults they never reject a design and never keep one. A deployment that
# tightens one past its constant rejects every design on that constant, which is
# what this list exists to catch.
PLACEHOLDER_FILTER_METRICS = (
    "Binder_Energy_Score",
    "dG",
    "dG/dSASA",
    "PackStat",
    "n_InterfaceHbonds",
    "n_InterfaceUnsatHbonds",
    "InterfaceHbondsPercentage",
    "InterfaceUnsatHbondsPercentage",
)
# Measured, but through the substituted path: the bundled sc binary for shape
# complementarity and freesasa or Biopython for the SASA terms. These gate on a
# real number, so a rejection naming one of them is evidence.
SUBSTITUTED_FILTER_METRICS = (
    "ShapeComplementarity",
    "Surface_Hydrophobicity",
    "dSASA",
    "Interface_SASA_%",
    "Interface_Hydrophobicity",
)
STAGING_NOTE = (
    "This is a staging set for an independent screen, not a set of accepted "
    "binders. It retains every MPNN design the tool predicted and relaxed, "
    "whether the tool's own filters kept it or dropped it, because behind "
    "--no-pyrosetta several filtered fields are constants rather than "
    "measurements. Rank and retain on the screen's own metrics. No entry of "
    "this set may be published as an accepted binder."
)
FILTER_SEMANTICS_NOTE = (
    "Behind --no-pyrosetta the fork computes shape complementarity and SASA "
    "through OpenMM, FASPR and sc, which the fork's own refinement notes record "
    "within about one percent of the licensed path; the free energy difference "
    "is not replicated, so dG-shaped filter thresholds behave differently and a "
    "deployment must widen or replace them deliberately."
)

# The payload shapes the application can answer with. inline_b64 carries raw
# bytes, gzip_b64 carries them gzipped, and object_refs carries a URL to the same
# gzipped bytes. All three end at the raw bytes this module digests, so one
# digest check covers every shape.
PAYLOAD_TRANSPORT = "gzip_b64"
TRANSPORT_FIELDS = {
    "inline_b64": ("pose_b64", "fasta_b64"),
    "gzip_b64": ("pose_gz_b64", "fasta_gz_b64"),
    "object_refs": ("pose_url", "fasta_url"),
}
# `_encode_entry` writes the compressed-pose digest for both gzip transports.
# object_refs returns no FASTA compressed digest, but it does return the byte
# counts of both CDN objects.
TRANSPORT_INTEGRITY_FIELDS = {
    "inline_b64": (),
    "gzip_b64": ("pose_gz_sha256", "fasta_gz_sha256"),
    "object_refs": ("pose_gz_sha256", "pose_file_size", "fasta_file_size"),
}
# The service reports ``mixed`` at response level when an object-store upload
# degrades just one entry to gzip_b64. Each entry still names one concrete shape.
RESPONSE_TRANSPORTS = (*TRANSPORT_FIELDS, "mixed")
# ``persistent_run_id`` limits its normalized binder stem to 40 characters and
# appends the first 24 hex characters of the exact-request digest.
PERSISTENT_RUN_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?-[0-9a-f]{24}$")
PERSISTENT_RUN_ID_INPUT_FIELDS = (
    "request_id",
    "input_structure_text",
    "binder_name",
    "chains",
    "target_hotspot_residues",
    "minimum_length",
    "maximum_length",
    "number_of_final_designs",
    "filters_text",
    "advanced_text",
)
REQUIRED_DESIGN_CLASSIFICATION_FIELDS = (
    "generator_filter_failures",
    "placeholder_fed_failures",
    "substituted_path_failures",
    "measured_failures",
    "rejected_only_by_placeholder_filters",
)
REQUIRED_DESIGN_FIELDS = (
    "design_name",
    "rank",
    "binder_length_sampled",
    "trajectory_seed",
    "mpnn_variant",
    "export_set",
    "retention",
    "generator_verdict_directory",
    *REQUIRED_DESIGN_CLASSIFICATION_FIELDS,
    "transport",
    "sha256",
    "fasta_sha256",
    "bytes",
    "fasta_bytes",
)
REQUIRED_DESIGN_RESPONSE_FIELDS = (
    "designs",
    "transport",
    "accepted_total",
    "export_set",
    "staging_total",
    "staging_note",
    "staging_record",
    "advanced_record",
    "filters_source",
    "completion_status",
    "timed_out",
    "process_returncode",
    "evidence_files",
)

# The identity the application reports on every answer. A response missing any
# of these describes a runner nobody can name afterwards.
REQUIRED_RESPONSE_FIELDS = (
    "request_id",
    "device",
    "jax_version",
    "python_version",
    "source_revision",
    "environment_identity",
    "entry_script_sha256",
    "params_archive_name",
    "params_archive_bytes",
    "params_sentinel_present",
    "openmm_platforms",
    "openmm_platform_order",
    "filter_semantics_note",
    "seconds",
    "persistent_run_id",
)
OPTIONAL_RESPONSE_FIELDS = (
    "model_revision",
)

AdapterError = base.AdapterError
CANONICAL_AMINO_ACID_RE = base.SEQUENCE_RE
ATOM_RECORD_PREFIXES = base.ATOM_RECORD_PREFIXES
EVIDENCE_NAME_RE = base.RETURNED_NAME_RE


# ----------------------------------------------------------------------------
# The endpoint and the request.
# ----------------------------------------------------------------------------


def resolve_endpoint(value: str | None) -> str:
    """Return this tool's own deployment URL, refusing anything but one fal route."""
    return base.resolve_endpoint(value, FAL_URL_ENVIRONMENT_KEY)


def request_batches(count: int) -> list[int]:
    """Return the per-request design counts one phase of `count` designs needs."""
    return base.request_batches(count, MAXIMUM_FINAL_DESIGNS)


def validate_request_values(
    *,
    count: int,
    minimum_length: int,
    maximum_length: int,
    generator_id: str,
    binder_name: str,
    target_chain: str,
) -> None:
    """Refuse a request the application would reject, before it is paid for."""
    if not 1 <= count <= MAXIMUM_FINAL_DESIGNS:
        raise AdapterError(
            f"--count is {count}; the application accepts 1 to {MAXIMUM_FINAL_DESIGNS} designs "
            "per request"
        )
    for label, length in (("--minimum-length", minimum_length), ("--maximum-length", maximum_length)):
        if not MINIMUM_BINDER_LENGTH <= length <= MAXIMUM_BINDER_LENGTH:
            raise AdapterError(
                f"{label} is {length}; the application accepts {MINIMUM_BINDER_LENGTH} to "
                f"{MAXIMUM_BINDER_LENGTH}"
            )
    if minimum_length > maximum_length:
        raise AdapterError(
            f"--minimum-length {minimum_length} sits above --maximum-length {maximum_length}"
        )
    if base.IDENTIFIER_RE.fullmatch(generator_id) is None:
        raise AdapterError(f"--generator-id is not a plain identifier: {generator_id}")
    validate_binder_name(binder_name)
    if base.CHAIN_ID_RE.fullmatch(target_chain) is None:
        raise AdapterError(f"the design target chain is {target_chain!r}, which is not a chain id")
    if target_chain == SERVED_BINDER_CHAIN_ID:
        raise AdapterError(
            f"the design target chain is {SERVED_BINDER_CHAIN_ID}, which FreeBindCraft reserves "
            "for the designed binder, and the application refuses a target on that letter. "
            "Prepare the target on another chain rather than relabelling it here"
        )


def validate_binder_name(binder_name: str) -> None:
    """Refuse a binder name the application's own grammar rejects.

    The name becomes the stem of every design name the tool writes, so one that
    already carries the parts the tool appends would produce a design name this
    module cannot parse back.
    """
    if BINDER_NAME_RE.fullmatch(binder_name) is None:
        raise AdapterError(
            f"--binder-name is {binder_name!r}; it must start with a letter and hold only "
            "letters, digits, hyphens and underscores, at most 64 characters"
        )
    if MODEL_SUFFIX_RE.search(binder_name):
        raise AdapterError(
            f"--binder-name {binder_name!r} ends with the _model<k> part the tool writes itself"
        )
    if DESIGN_NAME_RE.match(binder_name):
        raise AdapterError(
            f"--binder-name {binder_name!r} already carries the _l, _s and _mpnn parts the tool "
            "appends"
        )


def batch_binder_name(stem: str, position: int, batches: int) -> str:
    """Return the binder name one batch of a split phase carries.

    The deployment hashes the binder name and request ID alongside the target,
    settings, lengths, and requested count when it derives a persistent run
    directory. The normal `[32, 18]` split differs in both request count and
    batch identity. The suffix remains useful because it makes each tool-owned
    design name traceable to its request.
    """
    if batches < 2:
        validate_binder_name(stem)
        return stem
    name = f"{stem}-b{position:03d}"
    validate_binder_name(name)
    return name


def phase_request_id(args: argparse.Namespace) -> str:
    """Return a request ID unique to a local attempt unless the user supplied one.

    The hosted application creates the deterministic run directory with
    ``exist_ok=True``. An old default based only on generator and phase therefore
    sent identical work from two fresh local attempts to the same directory. The
    opaque digest keeps the local path out of the request while carrying its
    identity into the upstream hash. An explicit ID stays stable for a deliberate
    recovery request.
    """
    if args.request_id:
        return base.request_id(args.request_id, DEFAULT_GENERATOR_ID)
    attempt_dir = getattr(args, "attempt_dir", None)
    if not isinstance(attempt_dir, Path):
        raise AdapterError("cannot derive a default request id without an attempt directory")
    identity = {
        "attempt_dir": str(attempt_dir.expanduser().resolve()),
        "generator_id": str(args.generator_id),
        "phase": str(args.phase),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return base.request_id(None, f"{DEFAULT_GENERATOR_ID}-{digest[:24]}")


def batch_request_id(identifier: str, position: int) -> str:
    """Return one application-valid batch ID without truncating away uniqueness."""
    candidate = f"{identifier}-b{position:03d}"
    if len(candidate) <= base.MAXIMUM_REQUEST_ID_LENGTH:
        return base.request_id(None, candidate)
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]
    stem_budget = base.MAXIMUM_REQUEST_ID_LENGTH - len(digest) - 1
    stem = identifier[:stem_budget].rstrip("-._")
    return base.request_id(None, f"{stem}-{digest}")


def build_payload(
    *,
    identifier: str,
    binder_name: str,
    target_structure: Path,
    target_chain: str,
    hotspots: list[int],
    minimum_length: int,
    maximum_length: int,
    count: int,
    filters_text: str | None,
    advanced_text: str | None,
    transport: str,
    recover_only: bool,
) -> dict[str, Any]:
    """Return the JSON body of one design-phase request."""
    if not target_structure.is_file():
        raise AdapterError(f"target structure not found: {target_structure}")
    name = target_structure.name
    if base.RETURNED_NAME_RE.fullmatch(name) is None or not name.endswith(".pdb"):
        raise AdapterError(
            f"the application accepts one plain PDB file name for the target and the normalized "
            f"structure is named {name!r}"
        )
    return {
        "request_id": identifier,
        "input_structure_name": name,
        "input_structure_text": target_structure.read_text(),
        "binder_name": binder_name,
        "chains": target_chain,
        "target_hotspot_residues": ",".join(str(number) for number in hotspots),
        "minimum_length": minimum_length,
        "maximum_length": maximum_length,
        "number_of_final_designs": count,
        "filters_text": filters_text,
        "advanced_text": advanced_text,
        "payload_transport": transport,
        "recover_only": recover_only,
    }


def persistent_run_id_for_payload(payload: dict[str, Any]) -> str:
    """Reproduce the served application's deterministic persistent-run identity.

    ``fal_freebindcraft_app.py`` hashes these ten fields, in this exact JSON
    serialization. Transport and recovery are deliberately absent because the
    application does not include them in its directory identity.
    """
    missing = [field for field in PERSISTENT_RUN_ID_INPUT_FIELDS if field not in payload]
    if missing:
        raise AdapterError(
            "cannot derive the persistent run id because the request has no " + ", ".join(missing)
        )
    input_structure = payload["input_structure_text"]
    binder_name = payload["binder_name"]
    if not isinstance(input_structure, str) or not isinstance(binder_name, str):
        raise AdapterError("cannot derive the persistent run id from a non-text target or binder name")
    identity = {
        "request_id": payload["request_id"],
        "input_structure_sha256": hashlib.sha256(input_structure.encode("utf-8")).hexdigest(),
        "binder_name": binder_name,
        "chains": payload["chains"],
        "target_hotspot_residues": payload["target_hotspot_residues"],
        "minimum_length": payload["minimum_length"],
        "maximum_length": payload["maximum_length"],
        "number_of_final_designs": payload["number_of_final_designs"],
        "filters_text": payload["filters_text"],
        "advanced_text": payload["advanced_text"],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    stem = re.sub(r"[^a-z0-9]+", "-", binder_name.lower()).strip("-")[:40]
    return f"{stem or DEFAULT_BINDER_NAME}-{digest[:24]}"


def require_response_contract(response: dict[str, Any]) -> None:
    """Require the fields the served ``DesignResponse`` declares.

    The endpoint's Pydantic response model supplies these fields on every
    design answer. Rejecting an omitted field is safer than substituting an
    apparently benign default for provider evidence that was never supplied.
    """
    required = REQUIRED_RESPONSE_FIELDS + REQUIRED_DESIGN_RESPONSE_FIELDS
    missing = [field for field in required if field not in response or response[field] is None]
    if missing:
        raise AdapterError("the application reported no " + ", ".join(missing))
    for field in ("request_id", "device", "jax_version", "python_version", "source_revision",
                  "environment_identity", "entry_script_sha256", "params_archive_name",
                  "openmm_platforms", "openmm_platform_order", "filter_semantics_note", "transport",
                  "export_set", "staging_note", "filters_source", "completion_status",
                  "persistent_run_id"):
        if not isinstance(response[field], str) or not response[field]:
            raise AdapterError(f"the application {field} is not non-empty text")
    for field in ("params_archive_bytes", "accepted_total", "staging_total", "process_returncode"):
        value = response[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise AdapterError(f"the application {field} is not an integer")
    if not isinstance(response["seconds"], (int, float)) or isinstance(response["seconds"], bool):
        raise AdapterError("the application seconds is not a number")
    if not isinstance(response["params_sentinel_present"], bool):
        raise AdapterError("the application params_sentinel_present is not a boolean")
    if not isinstance(response["timed_out"], bool):
        raise AdapterError("the application timed_out is not a boolean")
    if not isinstance(response["designs"], list):
        raise AdapterError("the application designs value is not a list")
    if not isinstance(response["staging_record"], dict):
        raise AdapterError("the application staging_record is not an object")
    if not isinstance(response["advanced_record"], dict):
        raise AdapterError("the application advanced_record is not an object")
    if not isinstance(response["evidence_files"], list):
        raise AdapterError("the application evidence_files value is not a list")
    if response["transport"] not in RESPONSE_TRANSPORTS:
        raise AdapterError(f"the application names unknown payload transport {response['transport']!r}")
    if response["export_set"] != EXPORT_SET:
        raise AdapterError(
            f"the application reports export_set {response['export_set']!r}, expected {EXPORT_SET!r}"
        )
    if PERSISTENT_RUN_ID_RE.fullmatch(response["persistent_run_id"]) is None:
        raise AdapterError("the application persistent_run_id is not a path-safe run identifier")


def verify_response_identity(response: dict[str, Any], payload: dict[str, Any]) -> None:
    """Bind a served answer to the request that this dispatch actually sent."""
    require_response_contract(response)
    expected_request_id = payload.get("request_id")
    if not isinstance(expected_request_id, str) or not expected_request_id:
        raise AdapterError("the dispatch payload carries no usable request_id")
    if response["request_id"] != expected_request_id:
        raise AdapterError(
            f"the application answered request_id {response['request_id']!r}, expected "
            f"{expected_request_id!r}"
        )
    expected_run_id = persistent_run_id_for_payload(payload)
    if response["persistent_run_id"] != expected_run_id:
        raise AdapterError(
            f"the application answered persistent_run_id {response['persistent_run_id']!r}, expected "
            f"{expected_run_id!r} for this request"
        )


def settings_texts(args: argparse.Namespace) -> tuple[str | None, str | None, dict[str, Any]]:
    """Return the settings texts one request carries, and where they came from.

    An absent path sends nothing, which runs the application's own pinned copies
    of the fork defaults. The record states which side supplied the thresholds,
    because the fork's dG-shaped thresholds behave differently behind
    --no-pyrosetta and a reader must never have to guess whose numbers ran.
    """
    record: dict[str, Any] = {"filters_source": "app-default", "advanced_source": "app-default"}
    filters_text = advanced_text = None
    if getattr(args, "filters_path", None) is not None:
        path = args.filters_path.expanduser()
        if not path.is_file():
            raise AdapterError(f"--filters-path does not exist: {path}")
        filters_text = path.read_text()
        record["filters_source"] = "request"
        record["filters_sha256"] = base.sha256_bytes(filters_text.encode("utf-8"))
    if getattr(args, "advanced_path", None) is not None:
        path = args.advanced_path.expanduser()
        if not path.is_file():
            raise AdapterError(f"--advanced-path does not exist: {path}")
        advanced_text = path.read_text()
        record["advanced_source"] = "request"
        record["advanced_sha256"] = base.sha256_bytes(advanced_text.encode("utf-8"))
    return filters_text, advanced_text, record


# ----------------------------------------------------------------------------
# The answer. Only the dispatch child reaches this.
# ----------------------------------------------------------------------------


def fetch_object(url: str, timeout_seconds: int) -> bytes:
    """Read one artifact the application spilled to the CDN under object_refs."""
    request = urllib.request.Request(url, method="GET")
    try:
        opener = urllib.request.build_opener(base.RejectRedirects())
        with opener.open(request, timeout=timeout_seconds) as response:
            return response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise AdapterError(
            f"the artifact at {url} did not download ({exc}). Re-run the phase with the gzip_b64 "
            "transport, which keeps every artifact inside the answer"
        ) from exc


def artifact_bytes(
    label: str, kind: str, record: dict[str, Any], transport: str, timeout_seconds: int
) -> bytes:
    """Return one artifact's raw bytes under whichever transport carried it."""
    if transport == "inline_b64":
        try:
            return base64.b64decode(str(record[f"{kind}_b64"]), validate=True)
        except ValueError as exc:
            raise AdapterError(f"{label} carries invalid base64 for its {kind}") from exc
    if transport == "gzip_b64":
        try:
            compressed = base64.b64decode(str(record[f"{kind}_gz_b64"]), validate=True)
        except ValueError as exc:
            raise AdapterError(f"{label} carries invalid base64 for its {kind}") from exc
    else:
        compressed = fetch_object(str(record[f"{kind}_url"]), timeout_seconds)
    digest_required = transport == "gzip_b64" or (transport == "object_refs" and kind == "pose")
    declared = record.get(f"{kind}_gz_sha256")
    if digest_required and not isinstance(declared, str):
        raise AdapterError(f"{label} records no compressed digest for its {kind}")
    if declared is not None:
        if not isinstance(declared, str):
            raise AdapterError(f"{label} compressed digest for its {kind} is not text")
        observed = base.sha256_bytes(compressed)
        if observed != declared:
            raise AdapterError(
                f"{label} {kind} arrived as {observed} compressed and the runner recorded "
                f"{declared}. Re-run the phase; if the mismatch recurs, the answer is being "
                "rewritten between the runner and this adapter"
            )
    if transport == "object_refs":
        declared_size = record.get(f"{kind}_file_size")
        if not isinstance(declared_size, int) or isinstance(declared_size, bool):
            raise AdapterError(f"{label} records no integer compressed byte count for its {kind}")
        if len(compressed) != declared_size:
            raise AdapterError(
                f"{label} {kind} arrived as {len(compressed)} compressed bytes and the runner "
                f"recorded {declared_size}"
            )
    try:
        return gzip.decompress(compressed)
    except (OSError, EOFError) as exc:
        raise AdapterError(f"{label} carries a {kind} payload that does not gunzip") from exc


def design_identity(design_name: str) -> dict[str, Any]:
    """Parse the length, the trajectory seed and the MPNN variant out of one name.

    The tool encodes all three in the file name it writes and the served answer
    reports them as separate fields. Parsing the name back out is how this
    module checks the report rather than repeating it.
    """
    match = DESIGN_NAME_RE.fullmatch(design_name)
    if match is None:
        raise AdapterError(
            f"the application returned design name {design_name!r}, which is not "
            "<name>_l<length>_s<seed>_mpnn<variant>; this adapter reads the sampled length, the "
            "trajectory seed and the MPNN variant out of that name"
        )
    return {
        "binder_length_sampled": int(match.group("length")),
        "trajectory_seed": int(match.group("seed")),
        "mpnn_variant": int(match.group("variant")),
    }


def classify_filter_failures(names: list[str]) -> dict[str, Any]:
    """Say what kind of evidence one design's filter failures carry."""
    placeholder = sorted(set(names) & set(PLACEHOLDER_FILTER_METRICS))
    substituted = sorted(set(names) & set(SUBSTITUTED_FILTER_METRICS))
    measured = sorted(set(names) - set(PLACEHOLDER_FILTER_METRICS) - set(SUBSTITUTED_FILTER_METRICS))
    return {
        "generator_filter_failures": sorted(set(names)),
        "placeholder_fed_failures": placeholder,
        "substituted_path_failures": substituted,
        "measured_failures": measured,
        # True only when every reason the tool gave was a constant, so the
        # rejection says nothing about the design itself.
        "rejected_only_by_placeholder_filters": bool(placeholder) and not (substituted or measured),
    }


def checked_classification(label: str, record: dict[str, Any], retention: str) -> dict[str, Any]:
    """Return the classification, refusing one the failure columns do not support.

    The application classifies on its own side. This re-derives the same answer
    from the columns it reported and refuses a disagreement, because a row whose
    verdict nothing here can reproduce is a row nobody can check.
    """
    missing = [field for field in REQUIRED_DESIGN_CLASSIFICATION_FIELDS if field not in record]
    if missing:
        raise AdapterError(f"{label} records no " + ", ".join(missing))
    failures = record["generator_filter_failures"]
    if not isinstance(failures, list) or any(not isinstance(name, str) for name in failures):
        raise AdapterError(f"{label} reports filter failures that are not a list of column names")
    if retention == RETENTION_ACCEPTED and failures:
        raise AdapterError(
            f"{label} is in the accepted set and names {len(failures)} filter failures; a kept "
            "design has none"
        )
    derived = classify_filter_failures([str(name) for name in failures])
    for field, value in derived.items():
        if isinstance(value, list):
            reported = record[field]
            if not isinstance(reported, list) or any(not isinstance(item, str) for item in reported):
                raise AdapterError(f"{label} reports {field} that is not a list of column names")
            if sorted(reported) != value:
                raise AdapterError(
                    f"{label} reports {field} {reported!r} and its failure columns derive {value!r}"
                )
        elif not isinstance(record[field], bool) or record[field] != value:
            raise AdapterError(
                f"{label} reports {field} {record[field]!r} and its failure columns derive {value!r}"
            )
    return derived


def parse_fasta_text(text: str, source_label: str, expected_header: str) -> str:
    """Return the one sequence a single-record FASTA text holds."""
    header: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if header is not None:
                raise AdapterError(f"{source_label} carries more than one FASTA record")
            header = stripped[1:].strip()
            continue
        if header is None:
            raise AdapterError(f"{source_label} carries sequence text before its FASTA header")
        lines.append("".join(stripped.split()).upper())
    if header is None:
        raise AdapterError(f"{source_label} holds no FASTA header")
    header_name = header.split()[0] if header.split() else ""
    if header_name != expected_header:
        raise AdapterError(
            f"{source_label} records {header_name!r}, expected {expected_header}; refusing to "
            "guess which design this sequence belongs to"
        )
    sequence = "".join(lines)
    if CANONICAL_AMINO_ACID_RE.fullmatch(sequence) is None:
        raise AdapterError(f"{expected_header} carries a non-canonical amino acid in {source_label}")
    return sequence


def verified_designs(
    response: dict[str, Any],
    timeout_seconds: int = OBJECT_TIMEOUT_SECONDS,
    *,
    expected_binder_name: str | None = None,
) -> list[dict[str, Any]]:
    """Decode every returned design, holding each payload against its digest.

    A design names its own transport, because the application degrades a single
    entry from object_refs to gzip_b64 when one upload fails; the answer-level
    field only supplies the default an older runner left unset. Everything is
    decoded and checked before anything is written, so an answer that fails
    halfway leaves no partial directory behind.
    """
    require_response_contract(response)
    designs = response["designs"]
    if not designs:
        raise AdapterError(
            "the application staged no design, so no trajectory reached a relaxed best model. "
            "This is not the tool's filters rejecting the round: the staging set retains dropped "
            "designs too. Read the returned evidence files, then re-run the phase in a clean "
            "attempt directory"
        )
    if len(designs) > MAXIMUM_STAGING_DESIGNS:
        raise AdapterError(
            f"the application returned {len(designs)} designs and its own ceiling is "
            f"{MAXIMUM_STAGING_DESIGNS}"
        )
    decoded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(designs):
        label = f"returned design {index}"
        if not isinstance(record, dict):
            raise AdapterError(f"{label} is not a JSON object")
        missing = [field for field in REQUIRED_DESIGN_FIELDS if field not in record]
        if missing:
            raise AdapterError(f"{label} records no " + ", ".join(missing))
        transport = record["transport"]
        if not isinstance(transport, str):
            raise AdapterError(f"{label} transport is not text")
        if transport not in TRANSPORT_FIELDS:
            raise AdapterError(
                f"{label} names payload transport {transport!r} and this adapter reads "
                + ", ".join(sorted(TRANSPORT_FIELDS))
            )
        for field in TRANSPORT_FIELDS[transport] + TRANSPORT_INTEGRITY_FIELDS[transport]:
            if record.get(field) in (None, ""):
                raise AdapterError(f"{label} records no {field}")
        name = str(record["design_name"])
        if base.RETURNED_NAME_RE.fullmatch(name) is None:
            raise AdapterError(f"the application returned a design name this adapter refuses: {name!r}")
        if name in seen:
            raise AdapterError(f"the application returned {name} twice")
        seen.add(name)
        retention = record["retention"]
        if not isinstance(retention, str):
            raise AdapterError(f"{label} retention is not text")
        if retention not in RETENTIONS:
            raise AdapterError(
                f"{label} reports retention {retention!r}, which is not one of "
                + ", ".join(RETENTIONS)
            )
        identity = design_identity(name)
        match = DESIGN_NAME_RE.fullmatch(name)
        if match is None:
            raise AssertionError("design_identity returned without matching its design-name grammar")
        if expected_binder_name is not None and match.group("base") != expected_binder_name:
            raise AdapterError(
                f"{label} names binder {match.group('base')!r}, expected {expected_binder_name!r} "
                "for this request"
            )
        for field, value in identity.items():
            reported = record[field]
            if (
                not isinstance(reported, int)
                or isinstance(reported, bool)
                or reported != value
            ):
                raise AdapterError(
                    f"{label} reports {field} {reported!r} and its design name reads {value}"
                )
        rank = record["rank"]
        if rank is not None and (not isinstance(rank, int) or isinstance(rank, bool) or rank < 1):
            raise AdapterError(f"{label} reports rank {rank!r}, which is not a positive rank")
        if record["export_set"] != EXPORT_SET:
            raise AdapterError(
                f"{label} reports export_set {record['export_set']!r}, expected {EXPORT_SET!r}"
            )
        expected_directory = "Accepted" if retention == RETENTION_ACCEPTED else "Rejected"
        if record["generator_verdict_directory"] != expected_directory:
            raise AdapterError(
                f"{label} reports generator_verdict_directory "
                f"{record['generator_verdict_directory']!r}, expected {expected_directory!r}"
            )
        pose = artifact_bytes(label, "pose", record, transport, timeout_seconds)
        fasta = artifact_bytes(label, "fasta", record, transport, timeout_seconds)
        for payload, digest_field, kind in ((pose, "sha256", "pose"), (fasta, "fasta_sha256", "FASTA")):
            observed = base.sha256_bytes(payload)
            if observed != str(record[digest_field]):
                raise AdapterError(
                    f"{label} {kind} hashes to {observed} and the runner recorded "
                    f"{record[digest_field]}. Re-run the phase; if the mismatch recurs, the "
                    "endpoint is not serving the revision its identity reports"
                )
        for field, payload, kind in (("bytes", pose, "pose"), ("fasta_bytes", fasta, "FASTA")):
            reported_size = record[field]
            if not isinstance(reported_size, int) or isinstance(reported_size, bool):
                raise AdapterError(f"{label} {kind} byte count is not an integer")
            if reported_size != len(payload):
                raise AdapterError(
                    f"{label} {kind} is {len(payload)} bytes and the runner recorded {reported_size}"
                )
        decoded.append(
            {
                "design_index": index,
                "design_name": name,
                "retention": retention,
                "rank": rank,
                "generator_verdict_directory": record.get("generator_verdict_directory"),
                "transport": transport,
                "pose": pose,
                "pose_sha256": str(record["sha256"]),
                "fasta": fasta,
                "fasta_sha256": str(record["fasta_sha256"]),
                **identity,
                **checked_classification(label, record, retention),
            }
        )
    return decoded


def write_evidence_files(response: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    """Verify and retain the provider's metric and control files under the attempt.

    These are the tables behind the verdicts: the trajectory stats, the MPNN
    design stats, the rejected-design stats the classification reads, and the
    settings that actually ran. A phase that staged nothing still returns them,
    and they are the only place a reader can go to see why.
    """
    records = response.get("evidence_files")
    if records in (None, []):
        return []
    if not isinstance(records, list):
        raise AdapterError("the application evidence_files value is not a list")
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        label = f"evidence file {index}"
        if not isinstance(record, dict):
            raise AdapterError(f"{label} is not a JSON object")
        name = str(record.get("name", ""))
        if EVIDENCE_NAME_RE.fullmatch(name) is None or Path(name).name != name:
            raise AdapterError(f"{label} has an unsafe or empty name: {name!r}")
        try:
            compressed = base64.b64decode(str(record["gzip_b64"]), validate=True)
        except (KeyError, ValueError) as exc:
            raise AdapterError(f"{label} carries no valid gzip_b64 payload") from exc
        observed_gzip = base.sha256_bytes(compressed)
        if observed_gzip != str(record.get("gzip_sha256", "")):
            raise AdapterError(
                f"{label} compressed bytes hash to {observed_gzip}, not {record.get('gzip_sha256')}"
            )
        try:
            payload = gzip.decompress(compressed)
        except (OSError, EOFError) as exc:
            raise AdapterError(f"{label} does not gunzip") from exc
        observed = base.sha256_bytes(payload)
        if observed != str(record.get("sha256", "")):
            raise AdapterError(f"{label} hashes to {observed}, not {record.get('sha256')}")
        declared_bytes = record.get("bytes")
        if not isinstance(declared_bytes, int) or isinstance(declared_bytes, bool):
            raise AdapterError(f"{label} carries no integer byte count")
        if len(payload) != declared_bytes:
            raise AdapterError(f"{label} is {len(payload)} bytes, not the declared {declared_bytes}")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        path.write_bytes(payload)
        rows.append({"name": name, "path": str(path), "bytes": len(payload), "sha256": observed})
    return rows


def staging_counts(decoded: list[dict[str, Any]]) -> dict[str, int]:
    """Count the staging set from the designs that were actually verified.

    The served record is evidence. These are the numbers this adapter publishes,
    read off what it decoded, so a truncated or degraded answer cannot inflate
    them.
    """
    return {
        "staging_total": len(decoded),
        "generator_accepted_total": sum(
            1 for entry in decoded if entry["retention"] == RETENTION_ACCEPTED
        ),
        "generator_rejected_total": sum(
            1 for entry in decoded if entry["retention"] == RETENTION_REJECTED
        ),
        "rejected_only_by_placeholder_filters_total": sum(
            1 for entry in decoded if entry["rejected_only_by_placeholder_filters"]
        ),
    }


def write_index_and_files(
    out_dir: Path, decoded: list[dict[str, Any]], response: dict[str, Any]
) -> dict[str, Any]:
    """Write every returned file under one directory and index what was written.

    The parent process reads the index rather than the answer, so the decoded
    payloads never cross the process boundary a second time.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for entry in decoded:
        stem = f"design-{entry['design_index']:03d}"
        pose_path = out_dir / f"{stem}.pdb"
        pose_path.write_bytes(entry["pose"])
        fasta_path = out_dir / f"{stem}.fasta"
        fasta_path.write_bytes(entry["fasta"])
        rows.append(
            {
                key: entry[key]
                for key in (
                    "design_index",
                    "design_name",
                    "retention",
                    "rank",
                    "generator_verdict_directory",
                    "transport",
                    "binder_length_sampled",
                    "trajectory_seed",
                    "mpnn_variant",
                    "generator_filter_failures",
                    "placeholder_fed_failures",
                    "substituted_path_failures",
                    "measured_failures",
                    "rejected_only_by_placeholder_filters",
                )
            }
            | {
                "pose_file": pose_path.name,
                "pose_sha256": entry["pose_sha256"],
                "sequence_file": fasta_path.name,
                "sequence_sha256": entry["fasta_sha256"],
            }
        )
    served = response.get("staging_record")
    index = {
        "schema_version": 1,
        "designs": rows,
        "served_staging_record": served if isinstance(served, dict) else {},
        "evidence_files": write_evidence_files(response, out_dir / DEFAULT_EVIDENCE_SUBDIR),
        **staging_counts(decoded),
    }
    base.write_json(out_dir / DEFAULT_INDEX_NAME, index)
    return index


def write_receipt(
    path: Path,
    response: dict[str, Any],
    *,
    endpoint: str,
    client_wall_seconds: float,
    requested_count: int,
    binder_name: str,
    index: dict[str, Any],
) -> None:
    """Write what the runner reported and what this request asked of it.

    There is no cost field with a number in it. No measurement in this package
    prices this tool on this provider.
    """
    receipt: dict[str, Any] = dict(
        base.runtime_fields(response, REQUIRED_RESPONSE_FIELDS, OPTIONAL_RESPONSE_FIELDS)
    )
    receipt.update(
        {
            "runner_protocol": base.RUNNER_PROTOCOL,
            "fal_endpoint": endpoint,
            "request_id": response.get("request_id"),
            "binder_name": binder_name,
            "number_of_final_designs_requested": requested_count,
            "requested_seed": None,
            "seed_delivered": False,
            "client_wall_seconds": round(client_wall_seconds, 3),
            "runner_wall_seconds": response.get("seconds"),
            "filter_semantics_note": FILTER_SEMANTICS_NOTE,
            "staging_note": STAGING_NOTE,
            "export_set": EXPORT_SET,
            "served_accepted_total": response.get("accepted_total"),
            "served_staging_total": response.get("staging_total"),
            "evidence_files": index.get("evidence_files", []),
            "cost_basis": base.COST_BASIS,
            **{key: index[key] for key in staging_counts([])},
        }
    )
    advanced = response.get("advanced_record")
    if isinstance(advanced, dict):
        receipt["advanced_record"] = advanced
    base.write_json(path, receipt)


# ----------------------------------------------------------------------------
# Subcommands.
# ----------------------------------------------------------------------------


def dispatch(args: argparse.Namespace) -> int:
    """Post one design-phase request and write the files the runner returned."""
    endpoint = resolve_endpoint(args.fal_url)
    validate_request_values(
        count=args.count,
        minimum_length=args.minimum_length,
        maximum_length=args.maximum_length,
        generator_id=DEFAULT_GENERATOR_ID,
        binder_name=args.binder_name,
        target_chain=args.target_chain,
    )
    hotspots = [int(value) for value in str(args.hotspots).split(",") if value.strip()]
    if not hotspots:
        raise AdapterError("--hotspots carries no residue number")
    filters_text, advanced_text, _ = settings_texts(args)
    payload = build_payload(
        identifier=base.request_id(args.request_id, DEFAULT_GENERATOR_ID),
        binder_name=args.binder_name,
        target_structure=args.target_structure.expanduser(),
        target_chain=args.target_chain,
        hotspots=hotspots,
        minimum_length=args.minimum_length,
        maximum_length=args.maximum_length,
        count=args.count,
        filters_text=filters_text,
        advanced_text=advanced_text,
        transport=args.payload_transport,
        recover_only=args.recover_only,
    )
    started = time.monotonic()
    response = base.post(endpoint, payload, args.timeout_seconds, args.credential_env)
    seconds = time.monotonic() - started
    out_dir = args.out_dir.expanduser()
    try:
        verify_response_identity(response, payload)
        decoded = verified_designs(
            response,
            args.object_timeout_seconds,
            expected_binder_name=str(payload["binder_name"]),
        )
    except AdapterError:
        # A phase that staged nothing still returned the tables that say why, and
        # they are worth more than the refusal is.
        retained = write_evidence_files(response, out_dir / DEFAULT_EVIDENCE_SUBDIR)
        if retained:
            print(
                f"{TOOL_LABEL} dispatch: kept {len(retained)} provider evidence file(s) under "
                f"{out_dir / DEFAULT_EVIDENCE_SUBDIR}; persistent_run_id="
                f"{response.get('persistent_run_id')}",
                file=sys.stderr,
            )
        raise
    index = write_index_and_files(out_dir, decoded, response)
    write_receipt(
        args.receipt.expanduser(),
        response,
        endpoint=endpoint,
        client_wall_seconds=seconds,
        requested_count=args.count,
        binder_name=args.binder_name,
        index=index,
    )
    print(
        f"{TOOL_LABEL} dispatch: staged={index['staging_total']} "
        f"kept={index['generator_accepted_total']} dropped={index['generator_rejected_total']} "
        f"seconds={seconds:.1f} device={response['device']} out_dir={out_dir}"
    )
    return 0


def probe(args: argparse.Namespace) -> int:
    """Spawn the child that asks the deployed application to report its runtime."""
    if not args.acknowledge_cost:
        raise AdapterError(
            "probe starts a GPU runner on your own deployment and therefore costs money, so it "
            "needs --acknowledge-cost. Run toolcheck instead for the free readiness check, which "
            "sends no request and reads no credential"
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
    print(f"{TOOL_LABEL} probe: entry script {fields['entry_script_sha256']}")
    print(
        f"{TOOL_LABEL} probe: parameters {fields.get('params_archive_name', 'unnamed')} "
        f"({fields['params_archive_bytes']} bytes fetched at setup)"
    )
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
        f"{TOOL_LABEL} adapter: request ceiling number_of_final_designs 1-{MAXIMUM_FINAL_DESIGNS}, "
        f"lengths {MINIMUM_BINDER_LENGTH}-{MAXIMUM_BINDER_LENGTH}, answer ceiling "
        f"{MAXIMUM_STAGING_DESIGNS} staged designs"
    )
    print(
        f"{TOOL_LABEL} adapter: the design site is sent as bare residue numbers in one "
        f"{SERVED_HOTSPOT_FIELD_CHARACTERS}-character field; the shared site reader caps it at "
        f"{MAXIMUM_HOTSPOTS} residues, which is the stricter bound"
    )
    print(
        f"{TOOL_LABEL} adapter: a phase larger than {MAXIMUM_FINAL_DESIGNS} designs is dispatched "
        "as whole requests, each with its own request id, binder name and receipt"
    )
    print(
        f"{TOOL_LABEL} adapter: the served contract carries no seed field, so every row records "
        "the trajectory seed the tool chose and seed_delivered false"
    )
    print(
        f"{TOOL_LABEL} adapter: chain {SERVED_BINDER_CHAIN_ID} is reserved for the designed "
        "binder, a target on that letter is refused, and a campaign binder on another letter has "
        "column 22 of the PDB pose swapped"
    )
    print(f"{TOOL_LABEL} adapter: export_set {EXPORT_SET}. {STAGING_NOTE}")
    print(f"{TOOL_LABEL} adapter: {FILTER_SEMANTICS_NOTE}")
    print(f"{TOOL_LABEL} adapter: cost basis {base.COST_BASIS}; no measurement prices this provider")
    print(
        f"{TOOL_LABEL} adapter: this check sends no request and reads no credential; probe is the "
        "paid subcommand and it needs --acknowledge-cost",
        flush=True,
    )
    if not ready:
        print(f"{TOOL_LABEL} adapter: not ready to dispatch", file=sys.stderr)
    return 0 if ready else 1


# ----------------------------------------------------------------------------
# The phase.
# ----------------------------------------------------------------------------


def staging_sort_key(entry: dict[str, Any]) -> tuple[int, int, int, str]:
    """Order the staging set: kept designs by rank first, then dropped ones.

    An independent screen re-ranks everything it receives, so this order only
    decides which designs a smaller --count keeps. Putting the tool's kept
    designs first means a truncation drops the ones its filters liked least
    rather than an arbitrary slice. Rank is written per request, so a split
    phase breaks a rank tie by batch rather than pretending one global rank.
    """
    rank = entry.get("rank")
    return (
        0 if entry.get("retention") == RETENTION_ACCEPTED else 1,
        rank if isinstance(rank, int) and not isinstance(rank, bool) else 1 << 30,
        int(entry.get("dispatch_batch", 0)),
        str(entry.get("design_name", "")),
    )


def select_designs(
    pooled: list[dict[str, Any]], count: int, stage_all: bool
) -> list[dict[str, Any]]:
    """Pick the designs this phase publishes out of everything it staged."""
    ordered = sorted(pooled, key=staging_sort_key)
    return ordered if stage_all else ordered[:count]


def reject_duplicate_design_names(entries: list[dict[str, Any]]) -> None:
    """Refuse a design name repeated by separate hosted responses."""
    seen: dict[str, int] = {}
    for entry in entries:
        name = entry.get("design_name")
        batch = entry.get("dispatch_batch")
        if not isinstance(name, str) or not name:
            raise AdapterError("a returned design index carries no usable design_name")
        if name in seen:
            later = int(batch) + 1 if isinstance(batch, int) else "?"
            raise AdapterError(
                f"the application returned design {name!r} in both request {seen[name] + 1} "
                f"and request {later}"
            )
        seen[name] = int(batch) if isinstance(batch, int) else -1


def report_staging(
    args: argparse.Namespace, totals: dict[str, int], poseless: int, published: int
) -> None:
    """Print what the phase staged, what the tool's filters said, and the label."""
    kept = totals["generator_accepted_total"]
    dropped = totals["generator_rejected_total"]
    total = totals["staging_total"]
    constant_only = totals["rejected_only_by_placeholder_filters_total"]
    print(
        f"{TOOL_LABEL} adapter: staged {total} design(s); the tool's own filters kept {kept} and "
        f"dropped {dropped}. That verdict is recorded, not applied: behind "
        f"{PYROSETTA_BYPASS_FLAG} four of its default thresholds compare a constant that always "
        "passes, so this lane's own screen ranks all of them"
    )
    if constant_only:
        print(
            f"{TOOL_LABEL} adapter: {constant_only} of the dropped designs failed only fields that "
            f"are constants behind {PYROSETTA_BYPASS_FLAG}; that rejection is the placeholder "
            "talking, not the design. The shipped thresholds cannot produce it, so the filters in "
            "use tighten one past its constant"
        )
    if poseless:
        print(
            f"{TOOL_LABEL} adapter: {poseless} further design(s) failed the tool's base AF2 gate "
            "before a relaxed model existed, so they carry no pose and are not in the staging set"
        )
    if total < args.count:
        print(
            f"{TOOL_LABEL} adapter: the phase asked for {args.count} and staged {total}; "
            "publishing what ran rather than discarding it. The loop halts at its trajectory cap "
            "before reaching number_of_final_designs; raise the cap in the advanced settings or "
            "lower --count"
        )
    elif published < total:
        print(
            f"{TOOL_LABEL} adapter: {total - published} staged design(s) were held back by --count "
            f"{args.count}. Pass --stage-all to hand the screen the whole staging set"
        )
    print(
        f"{TOOL_LABEL} adapter: published {published} candidate(s) as export_set={EXPORT_SET}. "
        f"{STAGING_NOTE}"
    )


def validated_atom_lines(pose_text: str, source_label: str) -> list[str]:
    """Return coordinate-bearing PDB atom records from one returned pose.

    The application emits PDB text. A chain letter alone is not structural
    evidence, so require all three fixed-width coordinate fields to parse as
    finite numbers before a pose can be published or accepted by ``parse``.
    """
    atoms: list[str] = []
    for line_number, line in enumerate(pose_text.splitlines(), start=1):
        if not line.startswith(ATOM_RECORD_PREFIXES):
            continue
        if len(line) < 54:
            raise AdapterError(
                f"{source_label} atom record at line {line_number} is shorter than its coordinate columns"
            )
        chain = line[21:22]
        if base.CHAIN_ID_RE.fullmatch(chain) is None:
            raise AdapterError(
                f"{source_label} atom record at line {line_number} has no usable chain identifier"
            )
        for axis, field in zip(("x", "y", "z"), (line[30:38], line[38:46], line[46:54])):
            try:
                coordinate = float(field)
            except ValueError as exc:
                raise AdapterError(
                    f"{source_label} atom record at line {line_number} has no numeric {axis} coordinate"
                ) from exc
            if not math.isfinite(coordinate):
                raise AdapterError(
                    f"{source_label} atom record at line {line_number} has a non-finite {axis} coordinate"
                )
        atoms.append(line)
    if not atoms:
        raise AdapterError(f"{source_label} carries no atom records")
    return atoms


def compose_design_pose(
    *,
    candidate_id: str,
    design_name: str,
    source_sha256: str,
    pose_text: str,
    binder_chain: str,
    target_chain: str,
) -> tuple[str, int, int, bool]:
    """Return the candidate's pose text, its chain atom counts, and whether it moved.

    The staged pose already holds the predicted complex: the target on the letter
    the request named and the designed binder on the letter the application
    reserves. The coordinate records are copied unchanged behind REMARK lines
    naming the candidate and the design it came from, which keeps every pose's
    bytes distinct and lets a reader trace one back.

    The chain swap only runs when the campaign designs its binder on a letter
    other than the application's. A campaign already on that letter takes the
    pose through untouched, because relabelling a correct pose is the same
    defect as not relabelling a wrong one.
    """
    relabelled = binder_chain != SERVED_BINDER_CHAIN_ID
    text = (
        base.swap_chain_labels(pose_text, SERVED_BINDER_CHAIN_ID, binder_chain)
        if relabelled
        else pose_text
    )
    atoms = validated_atom_lines(text, design_name)
    binder_atoms = sum(1 for line in atoms if line[21:22] == binder_chain)
    target_atoms = sum(1 for line in atoms if line[21:22] == target_chain)
    if binder_atoms == 0:
        raise AdapterError(f"{design_name} carries no atom records for binder chain {binder_chain}")
    if target_atoms == 0:
        raise AdapterError(f"{design_name} carries no atom records for target chain {target_chain}")
    header = [
        f"REMARK 900 DESIGN POSE {candidate_id}",
        f"REMARK 900 SOURCE DESIGN {design_name}",
        f"REMARK 900 SOURCE SHA256 {source_sha256}",
    ]
    return "\n".join([*header, *atoms, "END"]) + "\n", binder_atoms, target_atoms, relabelled


def dispatch_batches(
    args: argparse.Namespace,
    *,
    endpoint: str,
    out_dir: Path,
    work_dir: Path,
    batches: list[int],
    common_values: list[str],
) -> list[tuple[dict[str, Any], dict[str, Any], Path]]:
    """Dispatch one phase as whole requests and return each batch's receipt and index."""
    results: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    identifier = phase_request_id(args)
    for position, batch_count in enumerate(batches):
        batch_dir = out_dir / f"batch-{position:03d}"
        receipt_path = work_dir / f"batch-{position:03d}-{args.receipt_name}"
        binder_name = batch_binder_name(args.binder_name, position, len(batches))
        base.run_external(
            base.child_argv(
                args,
                DISPATCH_COMMAND,
                endpoint,
                *common_values,
                "--count",
                str(batch_count),
                "--binder-name",
                binder_name,
                "--out-dir",
                str(batch_dir.resolve()),
                "--receipt",
                str(receipt_path.resolve()),
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--object-timeout-seconds",
                str(args.object_timeout_seconds),
                "--payload-transport",
                args.payload_transport,
                "--request-id",
                batch_request_id(identifier, position),
                *(["--recover-only"] if args.recover_only else []),
                *(
                    ["--filters-path", str(args.filters_path.expanduser().resolve())]
                    if args.filters_path is not None
                    else []
                ),
                *(
                    ["--advanced-path", str(args.advanced_path.expanduser().resolve())]
                    if args.advanced_path is not None
                    else []
                ),
                script=DISPATCH_SCRIPT,
            ),
            f"design up to {batch_count} candidate(s), request {position + 1} of {len(batches)}",
            TOOL_LABEL,
        )
        receipt = base.load_json(receipt_path, "fal receipt")
        base.validate_runtime_identity(args, receipt)
        base.write_json(receipt_path, receipt)
        index = base.load_json(batch_dir / DEFAULT_INDEX_NAME, "returned design index")
        designs = index.get("designs")
        if not isinstance(designs, list) or not designs:
            raise AdapterError(
                f"request {position + 1} indexed no design; the dispatch child refuses an empty "
                "staging set, so this index was written by something else"
            )
        results.append((receipt, index, batch_dir))
    return results


def run(args: argparse.Namespace) -> int:
    """Compose the request, dispatch one phase, and write the stage outputs."""
    batches = request_batches(args.count)
    if base.CHAIN_ID_RE.fullmatch(args.binder_chain) is None:
        raise AdapterError(
            f"--binder-chain is {args.binder_chain}; a chain ID is one letter or digit"
        )
    endpoint = resolve_endpoint(args.fal_url)
    # The route is resolved before anything is written, so a machine with no way
    # to reach the credential fails before it composes a request.
    base.resolve_credential_route(args)

    manifest, manifest_source = base.load_target_manifest(args)
    target_id = str(manifest["target_id"])
    chain = args.target_chain or str(manifest["design_target_chain_id"])
    validate_request_values(
        count=batches[0],
        minimum_length=args.minimum_length,
        maximum_length=args.maximum_length,
        generator_id=args.generator_id,
        binder_name=args.binder_name,
        target_chain=chain,
    )
    if chain == args.binder_chain:
        raise AdapterError(
            f"the target chain and --binder-chain are both {chain}; they name two different "
            "chains of the design pose, which carries the binder and the target together"
        )
    structure_path, structure_sha256 = base.normalized_structure(manifest, manifest_source)
    hotspots = base.site_residue_numbers(manifest, manifest_source, chain)
    _, _, settings_record = settings_texts(args)

    attempt_dir, phase_dir, work_dir, manifest_path = base.phase_paths(args)
    out_dir = work_dir / "returned"
    base.refuse_populated_output(out_dir)

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
            "--minimum-length",
            str(args.minimum_length),
            "--maximum-length",
            str(args.maximum_length),
        ],
    )

    pooled: list[dict[str, Any]] = []
    totals = {key: 0 for key in staging_counts([])}
    poseless = 0
    for batch_number, (receipt, index, batch_dir) in enumerate(results):
        for key in totals:
            totals[key] += int(index.get(key, 0))
        served = index.get("served_staging_record")
        if isinstance(served, dict):
            poseless += int(served.get("rejected_without_a_pose_total", 0) or 0)
        runtime = base.runtime_fields(receipt, REQUIRED_RESPONSE_FIELDS, OPTIONAL_RESPONSE_FIELDS)
        for index_row in index["designs"]:
            pooled.append(
                {
                    **index_row,
                    "dispatch_batch": batch_number,
                    "batch_dir": batch_dir,
                    "receipt": receipt,
                    "runtime": runtime,
                    "receipt_path": work_dir / f"batch-{batch_number:03d}-{args.receipt_name}",
                    "binder_name": batch_binder_name(args.binder_name, batch_number, len(batches)),
                }
            )
    reject_duplicate_design_names(pooled)
    selected = select_designs(pooled, args.count, args.stage_all)

    rows: list[dict[str, Any]] = []
    for position, entry in enumerate(selected):
        candidate_id = f"{args.generator_id}-{position:03d}"
        design_name = str(entry["design_name"])
        identity = design_identity(design_name)
        classification = checked_classification(
            f"staged design {design_name}", entry, str(entry["retention"])
        )
        pose_source = Path(entry["batch_dir"]) / str(entry["pose_file"])
        fasta_source = Path(entry["batch_dir"]) / str(entry["sequence_file"])
        for source, digest in ((pose_source, "pose_sha256"), (fasta_source, "sequence_sha256")):
            if not source.is_file():
                raise AdapterError(f"the dispatch child did not write {source}")
            observed = base.sha256_file(source)
            if observed != str(entry[digest]):
                raise AdapterError(
                    f"{source} hashes {observed} and the returned index records {entry[digest]}"
                )
        sequence = parse_fasta_text(fasta_source.read_text(), str(fasta_source), design_name)
        sampled_length = identity["binder_length_sampled"]
        for label, length in (("sampled length", sampled_length), ("sequence", len(sequence))):
            if not args.minimum_length <= length <= args.maximum_length:
                raise AdapterError(
                    f"{design_name} {label} is {length} residues, outside the declared window "
                    f"{args.minimum_length}-{args.maximum_length}"
                )
        pose_text, binder_atoms, target_atoms, relabelled = compose_design_pose(
            candidate_id=candidate_id,
            design_name=design_name,
            source_sha256=str(entry["pose_sha256"]),
            pose_text=pose_source.read_text(errors="replace"),
            binder_chain=args.binder_chain,
            target_chain=chain,
        )
        pose_path = phase_dir / base.DEFAULT_POSE_SUBDIR / f"{candidate_id}.pdb"
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        pose_path.write_text(pose_text)
        sequence_path = phase_dir / base.DEFAULT_SEQUENCE_SUBDIR / f"{candidate_id}.fasta"
        sequence_path.parent.mkdir(parents=True, exist_ok=True)
        sequence_path.write_text(f">{candidate_id}\n{sequence}\n")
        receipt = entry["receipt"]
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
                # The tool redesigns every trajectory with its own ProteinMPNN
                # step before it decides a verdict, so the generator owns the
                # sequence too and no sequence-design stage follows this one.
                "sequence_designer": args.generator_id,
                "generator_seed": identity["trajectory_seed"],
                "requested_seed": None,
                "tool_seed": None,
                "seed_delivered": False,
                "sequence_controlled_by_wrapper": False,
                "sequence_path": str(sequence_path.resolve()),
                "sequence_sha256": base.sha256_bytes(sequence.encode("ascii")),
                "sequence_length": len(sequence),
                "backbone_only": False,
                "structure_path": str(manifest["source_structure_path"]),
                "structure_sha256": str(manifest["target_sha256"]),
                "design_pose_path": str(pose_path.resolve()),
                "design_pose_sha256": base.sha256_file(pose_path),
                "residue_map_sha256": str(manifest["residue_map_sha256"]),
                "optimization_round": 0,
                "last_optimizer": None,
                "status": base.CANDIDATE_STATUS,
                "stage_id": args.stage,
                "design_index": position,
                "dispatch_batch": entry["dispatch_batch"],
                "design_name": design_name,
                "binder_name": entry["binder_name"],
                "binder_length_sampled": sampled_length,
                "mpnn_variant": identity["mpnn_variant"],
                "accepted_rank": entry.get("rank"),
                "binder_chain_id": args.binder_chain,
                "returned_binder_chain_id": SERVED_BINDER_CHAIN_ID,
                "target_chain_id": chain,
                "declared_target_chain_id": chain,
                "design_pose_relabelled": relabelled,
                "returned_pose_sha256": str(entry["pose_sha256"]),
                "binder_atom_count": binder_atoms,
                "target_atom_count": target_atoms,
                "lengths_window": [args.minimum_length, args.maximum_length],
                "number_of_final_designs_requested": args.count,
                "runtime_flags": list(RUNTIME_FLAGS),
                "export_set": EXPORT_SET,
                "staging_note": STAGING_NOTE,
                "filter_semantics_note": FILTER_SEMANTICS_NOTE,
                "retention": entry["retention"],
                "generator_filter_verdict": (
                    "kept" if entry["retention"] == RETENTION_ACCEPTED else "dropped"
                ),
                "generator_verdict_directory": entry.get("generator_verdict_directory"),
                **classification,
                "placeholder_filter_metrics": list(PLACEHOLDER_FILTER_METRICS),
                "substituted_filter_metrics": list(SUBSTITUTED_FILTER_METRICS),
                "staging_total": totals["staging_total"],
                "generator_accepted_total": totals["generator_accepted_total"],
                "generator_rejected_total": totals["generator_rejected_total"],
                "rejected_without_a_pose_total": poseless,
                "accepted_total": totals["generator_accepted_total"],
                "target_manifest_path": str(manifest_source),
                "input_structure_path": str(structure_path),
                "input_structure_sha256": structure_sha256,
                "hotspot_residues": hotspots,
                "fal_endpoint": endpoint,
                "fal_receipt_path": str(Path(entry["receipt_path"]).resolve()),
                "runtime_wall_seconds": receipt.get("runner_wall_seconds"),
                "cost_basis": base.COST_BASIS,
                **settings_record,
                **entry["runtime"],
            }
        )
    base.write_jsonl(manifest_path, rows)
    print(f"{TOOL_LABEL} adapter: {FILTER_SEMANTICS_NOTE}")
    report_staging(args, totals, poseless, len(rows))
    print(
        f"{TOOL_LABEL} adapter: phase={args.phase} target={target_id} candidates={len(rows)} "
        f"requests={len(batches)} seed_delivered=False device={rows[0]['device']} "
        f"manifest={manifest_path} target_manifest={manifest_source}"
    )
    return 0


def parse_pose(path: Path, row: dict[str, Any] | None) -> int:
    """Read one design pose and return its atom count, refusing one that is not a pose.

    A pose that carries no atom record, or that carries a chain the row does not
    name, is a file nothing downstream can use. When the row is known, the atom
    counts per chain are held against the counts `run` recorded, so a pose
    truncated or rewritten after the stage fails here rather than in a paid
    prediction.
    """
    counts: dict[str, int] = {}
    for line in validated_atom_lines(path.read_text(errors="replace"), str(path)):
        chain = line[21:22]
        counts[chain] = counts.get(chain, 0) + 1
    if row is None:
        return sum(counts.values())
    binder_chain = str(row["binder_chain_id"])
    target_chain = str(row["target_chain_id"])
    if set(counts) != {binder_chain, target_chain}:
        raise AdapterError(
            f"{path} carries chains {sorted(counts)} and the row names {sorted({binder_chain, target_chain})}"
        )
    for label, chain, recorded in (
        ("binder", binder_chain, row["binder_atom_count"]),
        ("target", target_chain, row["target_atom_count"]),
    ):
        if counts[chain] != int(recorded):
            raise AdapterError(
                f"{path} carries {counts[chain]} {label} atom records on chain {chain} and the "
                f"row records {recorded}"
            )
    return sum(counts.values())


def parse_outputs(args: argparse.Namespace) -> int:
    """Check the phase outputs this stage declares, reading every file it counts.

    The shared parser resolves the declared patterns and parses JSON and JSONL. It
    counts a FASTA or a PDB without opening it, and `lane.validate_artifact` answers
    true for a PDB whose whole content is one malformed line, so neither is evidence
    that a structure file was read. This one reads them: a FASTA record has to name
    the candidate its file is named for and carry canonical residues, a pose has to
    carry atom records on exactly the two chains its row names in the counts the row
    recorded, and every row's two files have to hash to what the row recorded.
    """
    attempt_dir = args.attempt_dir.expanduser().resolve()
    phase_dir = attempt_dir / args.phase
    files: list[Path] = []
    parsed_count = 0
    errors: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    try:
        stage = base.stage_record(args.config.expanduser().resolve(), args.stage)
    except AdapterError as exc:
        stage = {}
        errors.append(str(exc))
    for output in stage.get("outputs", []):
        if not isinstance(output, dict) or not isinstance(output.get("path_template"), str):
            errors.append("stage output has no path_template")
            continue
        kind = output.get("kind")
        pattern = base.parser_output_pattern(
            output["path_template"], attempt_dir, args.phase, args.stage
        )
        for value in sorted(glob.glob(pattern, recursive=True)):
            path = Path(value)
            if not path.is_file():
                continue
            files.append(path)
            try:
                if kind == "jsonl":
                    for line in path.read_text().splitlines():
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if not isinstance(row, dict) or not row.get("candidate_id"):
                            raise AdapterError(f"{path} carries a row with no candidate_id")
                        rows[str(row["candidate_id"])] = row
                        parsed_count += 1
                elif kind == "json":
                    json.loads(path.read_text())
                    parsed_count += 1
                elif kind == "fasta":
                    parse_fasta_text(path.read_text(), str(path), path.stem)
                    parsed_count += 1
                elif kind == "pdb":
                    parse_pose(path, rows.get(path.stem))
                    parsed_count += 1
                else:
                    parsed_count += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    # The manifest is read first, so every row's own two files are checked here
    # whether or not the glob above reached them.
    for candidate_id, row in sorted(rows.items()):
        for label, path_field, digest_field in (
            ("design pose", "design_pose_path", "design_pose_sha256"),
            ("sequence", "sequence_path", None),
        ):
            path = Path(str(row.get(path_field) or ""))
            if not path.is_file():
                errors.append(f"{candidate_id}: the {label} the row names is missing: {path}")
                continue
            if digest_field is not None and base.sha256_file(path) != str(row[digest_field]):
                errors.append(
                    f"{candidate_id}: the {label} {path} does not hash to the row's "
                    f"{digest_field}"
                )
        sequence_path = Path(str(row.get("sequence_path") or ""))
        if not sequence_path.is_file():
            continue
        try:
            sequence = parse_fasta_text(sequence_path.read_text(), str(sequence_path), candidate_id)
        except AdapterError as exc:
            errors.append(f"{candidate_id}: {exc}")
            continue
        if base.sha256_bytes(sequence.encode("ascii")) != str(row.get("sequence_sha256")):
            errors.append(f"{candidate_id}: the sequence does not hash to the row's sequence_sha256")
        if len(sequence) != int(row.get("sequence_length", -1)):
            errors.append(
                f"{candidate_id}: the sequence is {len(sequence)} residues and the row records "
                f"{row.get('sequence_length')}"
            )
    result_path = phase_dir / base.DEFAULT_PARSER_RESULT_NAME
    base.write_json(
        result_path,
        {
            "ok": bool(files) and not errors,
            "parsed_count": parsed_count,
            "rejected_count": len(errors),
            "errors": errors,
            "source_output_hashes": sorted(base.sha256_file(path) for path in files),
        },
    )
    for message in errors:
        print(f"{TOOL_LABEL} parse: {message}", file=sys.stderr)
    print(
        f"{TOOL_LABEL} adapter: parsed={parsed_count} rejected={len(errors)} "
        f"candidates={len(rows)} phase={args.phase} result={result_path}"
    )
    return 0 if files and not errors else 1


# ----------------------------------------------------------------------------
# Arguments.
# ----------------------------------------------------------------------------


def add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--minimum-length",
        type=int,
        required=True,
        help="Lower end of the binder length window the settings declare.",
    )
    parser.add_argument(
        "--maximum-length",
        type=int,
        required=True,
        help="Upper end of the binder length window the settings declare.",
    )
    parser.add_argument(
        "--binder-name",
        default=DEFAULT_BINDER_NAME,
        help=(
            "Design-name stem the tool embeds in every file it writes. A split phase appends "
            f"-b<NNN> per request. Defaults to {DEFAULT_BINDER_NAME}."
        ),
    )
    parser.add_argument(
        "--request-id",
        default=None,
        help=(
            "Request identifier the application echoes. On run, an omitted value derives an opaque "
            "identity from the local attempt, generator, and phase; supply one to recover a named "
            "persistent run deliberately."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            f"Request timeout. Defaults to {DEFAULT_TIMEOUT_SECONDS}, which is the request ceiling "
            "the deployment was built against."
        ),
    )
    parser.add_argument(
        "--object-timeout-seconds",
        type=int,
        default=OBJECT_TIMEOUT_SECONDS,
        help=(
            "Timeout for one artifact download under the object_refs transport. Defaults to "
            f"{OBJECT_TIMEOUT_SECONDS}."
        ),
    )
    parser.add_argument(
        "--payload-transport",
        choices=sorted(TRANSPORT_FIELDS),
        default=PAYLOAD_TRANSPORT,
        help=(
            "How the application returns each artifact. gzip_b64 keeps every artifact inside the "
            f"answer and is the default. Defaults to {PAYLOAD_TRANSPORT}."
        ),
    )
    parser.add_argument(
        "--recover-only",
        action="store_true",
        help=(
            "Ask the application to return the persistent run's existing staging designs and "
            "evidence without starting the design loop. Use this after a provider deadline "
            "stopped a request, because an identical repeat starts another trajectory instead of "
            "resuming."
        ),
    )
    parser.add_argument(
        "--filters-path",
        type=Path,
        default=None,
        help=(
            "Filter thresholds JSON to send. Absent sends nothing and the application runs its "
            "own pinned copy of the fork defaults."
        ),
    )
    parser.add_argument(
        "--advanced-path",
        type=Path,
        default=None,
        help=(
            "Advanced settings JSON to send. Absent sends nothing and the application runs its "
            "own pinned copy of the fork defaults."
        ),
    )


def add_layout_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generator-id", default=DEFAULT_GENERATOR_ID)
    parser.add_argument(
        "--binder-chain",
        default=DEFAULT_BINDER_CHAIN,
        help=(
            f"Chain the campaign designs its binder on. The application returns the binder on "
            f"{SERVED_BINDER_CHAIN_ID}; another letter has column 22 of the PDB pose swapped. "
            f"Defaults to {DEFAULT_BINDER_CHAIN}."
        ),
    )
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--work-subdir", default=DEFAULT_WORK_SUBDIR)
    parser.add_argument("--receipt-name", default=base.DEFAULT_RECEIPT_NAME)
    parser.add_argument(
        "--stage-all",
        action="store_true",
        help=(
            "Publish every staged design instead of the best --count of them. The independent "
            "screen ranks what it receives, so this hands it the whole set; --count still sizes "
            "the requests."
        ),
    )
    parser.add_argument(
        "--binder-length-min",
        type=int,
        default=None,
        help="Recorded only. --minimum-length is the value the request carries.",
    )
    parser.add_argument(
        "--binder-length-max",
        type=int,
        default=None,
        help="Recorded only. --maximum-length is the value the request carries.",
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
    parser = argparse.ArgumentParser(
        description="Generate FreeBindCraft designs on a fal deployment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "toolcheck", help="Report readiness without sending a request."
    )
    base.add_route_arguments(check_parser, FAL_URL_ENVIRONMENT_KEY)

    run_parser = subparsers.add_parser("run", help="Design one phase of candidates on fal.")
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
        print(f"{TOOL_LABEL} adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
