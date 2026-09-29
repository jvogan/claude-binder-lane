#!/usr/bin/env python3
"""Stage one unpaired target-chain alignment per target into the shared cache.

The published protocol stages target-chain MSAs once per target during tool
bring-up and has every MSA-consuming arm read that cache rather than query per
call. This is the stage that does it. It runs before anything folds, so no GPU
is allocated when an alignment cannot be built.

What it writes, per target: one a3m at `{attempt_dir}/{phase}/msa/<target_id>.a3m`
and one row in `{attempt_dir}/{phase}/msa-manifest.jsonl`. The a3m is unpaired
and holds the target chain only. The binder chain never gets an alignment on any
arm, so none is built for it.

The alignment is cleaned here, once, and the hash in the manifest is the hash of
the cleaned file. Cleaning inside each arm would mean two arms cleaning
differently and the recorded hash describing neither.

The depth on disk is the full depth. `ef2full` applies its 2048 limit when it
reads the file and Protenix does its own subsampling, so truncating at write
time would silently cap an arm that wants more.

`--route` has no default. An alignment route is a decision about network egress
and about what the numbers mean, so it is stated rather than inferred.
"""

import argparse
import io
import glob
import hashlib
import json
import sys
import tarfile
import time
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from pathlib import Path
from typing import Any

from claude_binder.adapters.target_prep_adapter import parse_cif_atoms, parse_pdb_atoms

MANIFEST_ARTIFACT_ID = "target-msa-manifest"
MSA_ARTIFACT_ID_PREFIX = "target-msa-"
DEFAULT_TARGET_STAGE_ID = "target-prepare"
DEFAULT_TARGET_ARTIFACT_ID = "target-manifest"

QUERY_SOURCE_ARGUMENT = "argument"
QUERY_SOURCE_TARGET_MANIFEST = "target-manifest"
QUERY_SOURCE_TARGET_STRUCTURE = "target-structure"
QUERY_SOURCE_TARGET_STRUCTURE_MMCIF = "target-structure-mmcif"

# The route that produced the alignment. It reaches the manifest, so a scored row
# traces back to how its alignment was built.
ROUTE_PUBLIC_SERVER = "public-server"
ROUTE_LOCAL = "local"
ROUTE_PRECOMPUTED = "precomputed"
ROUTES = (ROUTE_PUBLIC_SERVER, ROUTE_LOCAL, ROUTE_PRECOMPUTED)

SOURCE_QUERY_ONLY = "query-only"
SOURCE_COLABFOLD_SERVER = ROUTE_PUBLIC_SERVER
SOURCE_LOCAL_MMSEQS2 = ROUTE_LOCAL
SOURCE_SUPPLIED_A3M = "supplied-a3m"
LEGACY_SOURCE_COLABFOLD_SERVER = "colabfold-server"
LEGACY_SOURCE_LOCAL_MMSEQS2 = "local-mmseqs2"
SOURCES = (
    ROUTE_PUBLIC_SERVER,
    ROUTE_LOCAL,
    ROUTE_PRECOMPUTED,
    SOURCE_QUERY_ONLY,
    SOURCE_SUPPLIED_A3M,
    LEGACY_SOURCE_COLABFOLD_SERVER,
    LEGACY_SOURCE_LOCAL_MMSEQS2,
)

ROUTE_SELECTION_ERROR = (
    "choose one MSA route: public-server sends the target sequence to a third-party "
    "ColabFold server, precomputed reads an existing alignment, or local uses an "
    "MMseqs2 database. The local route "
    "has no execution implementation in this adapter."
)

# ColabFold defines this default API host in `cf_utils.py:25`.
COLABFOLD_MSA_SERVER_HOST = "https://api.colabfold.com"

# ColabFold maps filtered UniRef30 plus environmental searches to `env` in
# `cf_colabfold.py:162-166` and returns both files listed in `cf_colabfold.py:243-248`.
COLABFOLD_MSA_MODE = "env"

# ColabFold submits the MSA form fields in `cf_colabfold.py:91`.
COLABFOLD_QUERY_FIELD = "q"
COLABFOLD_MODE_FIELD = "mode"

# ColabFold numbers submitted FASTA headers from 101 in `cf_colabfold.py:80-84`.
COLABFOLD_FASTA_HEADER_START = 101

# ColabFold names the submit path at `cf_colabfold.py:72` and uses all three paths
# at `cf_colabfold.py:91`, `:116`, and `:140`.
COLABFOLD_SUBMIT_PATH = "ticket/msa"
COLABFOLD_STATUS_PATH = "ticket/{id}"
COLABFOLD_RESULT_PATH = "result/download/{id}"

# ColabFold names these result files in `cf_colabfold.py:243-248`.
COLABFOLD_UNIREF_RESULT = "uniref.a3m"
COLABFOLD_ENV_RESULT = "bfd.mgnify30.metaeuk30.smag30.a3m"

# ColabFold returns these statuses and polls while the first two are present in
# `cf_colabfold.py:217-221`; the rate-limit status is resubmitted at `:199-206`.
COLABFOLD_STATUS_PENDING = "PENDING"
COLABFOLD_STATUS_RUNNING = "RUNNING"
COLABFOLD_STATUS_COMPLETE = "COMPLETE"
COLABFOLD_STATUS_RATELIMIT = "RATELIMIT"

# ColabFold sets a 6.02-second timeout on submit, status, and result requests in
# `cf_colabfold.py:91`, `:116`, and `:140`.
COLABFOLD_REQUEST_TIMEOUT_SECONDS = 6.02

# The result download is not a control request. It carries the whole alignment, which
# runs to several megabytes for a well-covered target, and 6.02 seconds is not enough
# time to read it. A run that submitted successfully then died on the download.
COLABFOLD_DOWNLOAD_TIMEOUT_SECONDS = 300.0

# ColabFold waits at least five seconds before polling in `cf_colabfold.py:217-221`
# and before rate-limit resubmission in `:199-206`; this client uses that lower bound.
COLABFOLD_WAIT_SECONDS = 5

# ColabFold resubmits rate-limited work in `cf_colabfold.py:199-206` without a
# ceiling. This client adds a three-resubmission safety bound for a finite wait.
COLABFOLD_MAX_RATELIMIT_RESUBMITS = 3

# TODO The local MMseqs2 route's database identity, version, download host and
# on-disk size. The protocol offers it as the egress-free alternative and names
# neither the release nor where it comes from. Without those numbers nobody can
# size the volume or choose between the two routes. The real value comes from the
# ColabFold database setup instructions and the UniRef30 release the campaign
# used.

# TODO Alignments for the control and calibration chains. The protocol runs
# per-target positive controls and calibration complexes with MSAs on both
# chains, including the control-ligand chain. The stage already takes
# `run-bundle:controls` and `control-calibration` already reads this manifest, so
# the wiring is in place and the rows are not. Adding them needs a file-naming
# convention for a control chain and one declared output per control chain, which
# is a graph change rather than an adapter change.

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


class QuerySequenceUnavailable(Exception):
    """No argument and no structure supplied a target chain's query sequence."""


class SourceUnavailable(Exception):
    """The requested alignment route cannot run in this environment."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render(value: str, *, attempt_dir: Path, phase: str) -> str:
    return value.replace("{{attempt_dir}}", str(attempt_dir)).replace("{{phase}}", phase)


def stage_record(config: dict[str, Any], stage_id: str) -> dict[str, Any]:
    return next(stage for stage in config["stages"] if stage["stage_id"] == stage_id)


def artifact_output_path(
    stage: dict[str, Any],
    artifact_id: str,
    attempt_dir: Path,
    phase: str,
) -> Path:
    """Return the path the stage contract names for one artifact.

    Reading the contract rather than composing a filename is the point. The
    executor hashes what the contract names, so an adapter that invented its own
    filename would write a file nothing collects.
    """
    output = next(item for item in stage["outputs"] if item["artifact_id"] == artifact_id)
    pattern = render(output["path_template"], attempt_dir=attempt_dir, phase=phase)
    if any(character in pattern for character in "*?["):
        raise ValueError(
            f"{artifact_id} is still declared as a glob, so the resolved graph never "
            f"bound it to a target: {pattern}"
        )
    return Path(pattern)


def design_target_chain(target: dict[str, Any]) -> str:
    matches = [
        chain
        for chain in target.get("chains", [])
        if isinstance(chain, dict) and chain.get("role") == "design-target"
    ]
    if len(matches) != 1:
        raise ValueError(f"target {target.get('target_id')} must define one design-target chain")
    return str(matches[0]["chain_id"])


def key_value_argument(entry: str) -> tuple[str | None, str]:
    key, separator, value = entry.partition("=")
    return (key, value) if separator else (None, entry)


def resolve_per_target(
    entries: list[str] | None,
    targets: list[dict[str, Any]],
    label: str,
) -> dict[str, str]:
    """Turn repeated `TARGET_ID=VALUE` arguments into a per-target mapping.

    Same spelling as the arms use, so an operator supplies a target's sequence to
    the alignment stage and to the arm the same way.
    """
    resolved: dict[str, str] = {}
    for entry in entries or []:
        key, value = key_value_argument(entry)
        if key is None:
            if len(targets) != 1:
                raise ValueError(
                    f"{label} needs TARGET_ID=VALUE because the campaign has "
                    f"{len(targets)} targets"
                )
            key = str(targets[0]["target_id"])
        resolved[key] = value
    return resolved


def read_fasta_sequence(path: Path) -> str:
    lines = [line.strip() for line in path.read_text().splitlines()]
    return "".join(line for line in lines if line and not line.startswith(">")).upper()


def chain_sequence_from_pdb(path: Path, chain_id: str) -> str:
    """Read one chain's polymer sequence from a PDB, in the file's residue order.

    A residue the table does not carry becomes X. That is a real gap in the query
    rather than a silent substitution, and it is visible in the a3m. Solvent is a
    different case: it is not part of the chain at all, so it is dropped rather
    than recorded as a gap.
    """
    atoms = parse_pdb_atoms(path.read_text(errors="replace"))
    polymer = polymer_residue_keys(atoms, chain_id)
    sequence: list[str] = []
    seen: set[tuple[int, str]] = set()
    for atom in atoms:
        if atom.chain_id != chain_id:
            continue
        key = (atom.residue_number, atom.insertion_code)
        if key in seen or key not in polymer:
            continue
        seen.add(key)
        sequence.append(THREE_TO_ONE.get(atom.residue_name.upper(), "X"))
    return "".join(sequence)


def polymer_residue_keys(atoms: list, chain_id: str) -> set:
    """Return the residues of one chain that belong to the polymer.

    An ATOM record is polymer by definition. A HETATM record is polymer only
    when its residue carries a CA atom, which keeps a modified residue such as
    MSE and drops water, ions, and bound ligands. A deposited file records
    solvent under the same author chain id as the protein, so a chain read that
    keeps every record returns the chain plus its crystallographic water.
    """
    polymer: set[tuple[int, str]] = set()
    alpha_carbons: set[tuple[int, str]] = set()
    for atom in atoms:
        if atom.chain_id != chain_id:
            continue
        key = (atom.residue_number, atom.insertion_code)
        if atom.name.strip().upper() == "CA":
            alpha_carbons.add(key)
        if atom.record == "ATOM":
            polymer.add(key)
    return polymer | alpha_carbons


def chain_sequence_from_cif(path: Path, chain_id: str) -> str:
    """Read one chain's polymer sequence with the target preparer's mmCIF parser."""
    atoms = parse_cif_atoms(path.read_text(errors="replace"))
    polymer = polymer_residue_keys(atoms, chain_id)
    sequence: list[str] = []
    seen: set[tuple[int, str]] = set()
    for atom in atoms:
        if atom.chain_id != chain_id:
            continue
        key = (atom.residue_number, atom.insertion_code)
        if key in seen or key not in polymer:
            continue
        seen.add(key)
        sequence.append(THREE_TO_ONE.get(atom.residue_name.upper(), "X"))
    return "".join(sequence)


def completed_target_manifests(
    receipts_dir: Path,
    stage_id: str,
    artifact_id: str,
) -> dict[str, dict[str, Any]]:
    """Return target manifests recorded by a completed upstream receipt."""
    receipt_path = receipts_dir / f"{stage_id}.json"
    if not receipt_path.is_file():
        return {}
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict) or receipt.get("ok") is not True:
        raise QuerySequenceUnavailable(f"upstream receipt did not complete: {receipt_path}")
    artifacts = receipt.get("output_manifest", {}).get("artifacts", [])
    phases = {str(artifact.get("phase")) for artifact in artifacts}
    selected_phase = "scale" if "scale" in phases else "single"
    paths = [
        Path(str(file_record["path"]))
        for artifact in artifacts
        if artifact.get("phase") == selected_phase and artifact.get("artifact_id") == artifact_id
        for file_record in artifact.get("files", [])
    ]
    manifests: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        document = load_json(path)
        if not isinstance(document, dict):
            raise QuerySequenceUnavailable(f"target manifest is not a JSON object: {path}")
        target_id = str(document.get("target_id", ""))
        if not target_id:
            raise QuerySequenceUnavailable(f"target manifest records no target_id: {path}")
        manifests[target_id] = document
    return manifests


def manifest_structure_path(manifest: dict[str, Any] | None) -> Path | None:
    """Return a readable normalized structure path from one target manifest."""
    if manifest is None:
        return None
    value = manifest.get("normalized_structure_path")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def sequence_from_structure(
    target_id: str,
    structure_path: Path,
    chain_id: str,
) -> str:
    """Read one target chain from a supported coordinate structure."""
    suffix = structure_path.suffix.lower()
    if suffix == ".pdb":
        sequence = chain_sequence_from_pdb(structure_path, chain_id)
    elif suffix in {".cif", ".mmcif"}:
        sequence = chain_sequence_from_cif(structure_path, chain_id)
    else:
        raise QuerySequenceUnavailable(
            f"target {target_id} structure is {structure_path.suffix or 'extensionless'}, "
            f"which this stage does not parse, pass "
            f"--target-sequence {target_id}=SEQUENCE_OR_FASTA"
        )
    if not sequence:
        raise QuerySequenceUnavailable(
            f"target {target_id} chain {chain_id} has no residues in {structure_path}"
        )
    return sequence


def query_sequence_for(
    target: dict[str, Any],
    chain_id: str,
    supplied: dict[str, str],
    upstream_manifest: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Return one target chain's query sequence and where it came from.

    An explicit argument has highest priority. The target preparer's declared
    manifest supplies its normalized structure next. The configured structure
    remains the fallback for runs that have no readable normalized structure.
    """
    target_id = str(target["target_id"])
    value = supplied.get(target_id)
    if value:
        path = Path(value)
        sequence = read_fasta_sequence(path) if path.exists() else value.upper()
        if not sequence:
            raise QuerySequenceUnavailable(f"empty sequence supplied for target {target_id}")
        return sequence, QUERY_SOURCE_ARGUMENT
    upstream_structure = manifest_structure_path(upstream_manifest)
    if upstream_structure is not None:
        return (
            sequence_from_structure(target_id, upstream_structure, chain_id),
            QUERY_SOURCE_TARGET_MANIFEST,
        )
    structure_path = Path(str(target.get("runtime_structure_path", "")))
    if not structure_path.is_file():
        raise QuerySequenceUnavailable(
            f"no sequence for target {target_id} and no readable target structure at "
            f"{structure_path}, pass --target-sequence {target_id}=SEQUENCE_OR_FASTA"
        )
    sequence = sequence_from_structure(target_id, structure_path, chain_id)
    source = (
        QUERY_SOURCE_TARGET_STRUCTURE
        if structure_path.suffix.lower() == ".pdb"
        else QUERY_SOURCE_TARGET_STRUCTURE_MMCIF
    )
    return sequence, source


def clean_a3m_rows(rows: list[str], query_sequence: str) -> list[str]:
    """Return the rows an ESMFold2 MSA reader accepts.

    `MSA.from_a3m(remove_insertions=True)` asserts that every row is the same
    length once insertions are removed. ColabFold a3m files carry trailing null
    bytes and rows that are off by one against the query. So the nulls go, row 0
    becomes the query exactly, and a row that does not match the query length
    after insertion removal is dropped rather than carried into a reader that
    would assert on it.
    """
    cleaned = [query_sequence]
    for row in rows[1:]:
        stripped = row.replace("\x00", "").strip()
        if not stripped:
            continue
        without_insertions = "".join(
            character for character in stripped if not character.islower() and character != "."
        )
        if len(without_insertions) != len(query_sequence):
            continue
        cleaned.append(stripped)
    return cleaned


def _a3m_rows_from_lines(
    lines: list[str],
    label: str,
    label_prefix: str = "a3m",
) -> list[str]:
    """Return sequence rows from A3M text that has already been read."""
    rows: list[str] = []
    sequence_lines: list[str] = []
    saw_header = False
    for line in lines:
        if line.startswith(">"):
            if saw_header:
                rows.append("".join(sequence_lines))
            saw_header = True
            sequence_lines = []
        elif saw_header:
            sequence_lines.append(line)
        elif line.strip():
            raise ValueError(
                f"{label_prefix} {label} has sequence data before its first header"
            )
    if saw_header:
        rows.append("".join(sequence_lines))
    if not rows:
        raise ValueError(f"{label_prefix} {label} has no sequences")
    return rows


def read_a3m_rows(path: Path) -> list[str]:
    """Return the sequence rows from one supplied a3m file."""
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read supplied a3m {path}: {type(exc).__name__}: {exc}") from exc
    return _a3m_rows_from_lines(lines, str(path), "supplied a3m")


def supplied_a3m_rows(
    path_value: str | None,
    query_sequence: str,
    minimum_sequences: int | None,
    *,
    target_id: str | None = None,
    chain_id: str | None = None,
) -> list[str]:
    """Validate and return the supplied rows for the shared writer."""
    if not path_value:
        raise ValueError("route precomputed requires --precomputed-a3m TARGET_ID=PATH")
    path = Path(path_value)
    rows = read_a3m_rows(path)
    first_sequence = rows[0].replace("\x00", "").strip()
    if first_sequence != query_sequence:
        if target_id is not None and chain_id is not None:
            raise ValueError(
                f"precomputed alignment {path} for target {target_id} chain {chain_id} "
                f"has first sequence; expected {query_sequence}; found {first_sequence}"
            )
        raise ValueError(f"supplied a3m {path} first sequence does not match the target query")
    cleaned = clean_a3m_rows(rows, query_sequence)
    if len(cleaned) != len(rows):
        raise ValueError(
            f"supplied a3m {path} has a row whose length after cleaning differs from "
            "the target query"
        )
    if minimum_sequences is not None and len(cleaned) < minimum_sequences:
        raise ValueError(
            f"supplied a3m {path} has {len(cleaned)} sequences, below "
            f"--minimum-sequences {minimum_sequences}"
        )
    return rows


def _colabfold_url(host: str, path: str) -> str:
    return f"{host.rstrip('/')}/{path}"


def build_colabfold_request(url: str, data: bytes | None = None) -> urllib_request.Request:
    headers: dict[str, str] = {}
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    return urllib_request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if data is not None else "GET",
    )


def build_public_msa_request(
    query_sequence: str,
    *,
    host_url: str | None = None,
) -> urllib_request.Request:
    """Build the public MSA submission request without sending it."""
    host = COLABFOLD_MSA_SERVER_HOST if host_url is None else host_url
    query = f">{COLABFOLD_FASTA_HEADER_START}\n{query_sequence}\n"
    form = urllib_parse.urlencode(
        {
            COLABFOLD_QUERY_FIELD: query,
            COLABFOLD_MODE_FIELD: COLABFOLD_MSA_MODE,
        }
    ).encode("utf-8")
    return build_colabfold_request(
        _colabfold_url(host, COLABFOLD_SUBMIT_PATH),
        data=form,
    )


def _send_colabfold_request(
    http_request: urllib_request.Request,
    *,
    timeout_seconds: float = COLABFOLD_REQUEST_TIMEOUT_SECONDS,
) -> bytes:
    endpoint = http_request.full_url
    try:
        with urllib_request.urlopen(
            http_request,
            timeout=timeout_seconds,
        ) as response:
            return response.read()
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError, OSError) as exc:
        raise SourceUnavailable(
            f"ColabFold MSA request to {endpoint} failed: {type(exc).__name__}: {exc}"
        ) from exc


def _colabfold_request(
    url: str,
    data: bytes | None = None,
    *,
    timeout_seconds: float = COLABFOLD_REQUEST_TIMEOUT_SECONDS,
) -> bytes:
    return _send_colabfold_request(
        build_colabfold_request(url, data=data),
        timeout_seconds=timeout_seconds,
    )


def _colabfold_json(payload: bytes, endpoint: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceUnavailable(
            f"ColabFold MSA endpoint {endpoint} returned invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise SourceUnavailable(
            f"ColabFold MSA endpoint {endpoint} returned JSON without an object payload"
        )
    return value


def _colabfold_status(payload: dict[str, Any], endpoint: str) -> str:
    status = payload.get("status")
    allowed = (
        COLABFOLD_STATUS_PENDING,
        COLABFOLD_STATUS_RUNNING,
        COLABFOLD_STATUS_COMPLETE,
        COLABFOLD_STATUS_RATELIMIT,
    )
    if status not in allowed:
        raise SourceUnavailable(
            f"ColabFold MSA endpoint {endpoint} returned unexpected status {status!r}; "
            f"expected one of {', '.join(allowed)}"
        )
    return status


def _wait_for_colabfold_ratelimit(
    host: str,
    resubmits: int,
    sleep_fn: Any,
) -> int:
    if resubmits >= COLABFOLD_MAX_RATELIMIT_RESUBMITS:
        raise SourceUnavailable(
            f"ColabFold MSA server {host} returned RATELIMIT after "
            f"{COLABFOLD_MAX_RATELIMIT_RESUBMITS} resubmissions; wait limit reached"
        )
    next_resubmits = resubmits + 1
    print(
        f"ColabFold MSA server {host} returned RATELIMIT. Waiting "
        f"{COLABFOLD_WAIT_SECONDS} seconds before resubmitting "
        f"({next_resubmits}/{COLABFOLD_MAX_RATELIMIT_RESUBMITS}).",
        file=sys.stderr,
    )
    sleep_fn(COLABFOLD_WAIT_SECONDS)
    return next_resubmits


def _colabfold_result_rows(host: str, job_id: str) -> list[str]:
    endpoint = _colabfold_url(host, COLABFOLD_RESULT_PATH.format(id=job_id))
    archive_bytes = _colabfold_request(
        endpoint,
        timeout_seconds=COLABFOLD_DOWNLOAD_TIMEOUT_SECONDS,
    )
    try:
        archive = tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz")
    except tarfile.TarError as exc:
        raise SourceUnavailable(
            f"ColabFold MSA result {endpoint} was not a readable tar.gz archive"
        ) from exc

    with archive:
        members = archive.getmembers()

        def member_for(filename: str) -> tarfile.TarInfo:
            matches = [
                member
                for member in members
                if member.isfile() and member.name.rstrip("/").split("/")[-1] == filename
            ]
            if len(matches) != 1:
                raise SourceUnavailable(
                    f"ColabFold MSA result {endpoint} has {len(matches)} files named "
                    f"{filename!r}; expected exactly one"
                )
            return matches[0]

        def rows_for(filename: str) -> list[str]:
            member = member_for(filename)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise SourceUnavailable(
                    f"ColabFold MSA result {endpoint} could not read {filename!r}"
                )
            try:
                text = extracted.read().decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SourceUnavailable(
                    f"ColabFold MSA result {endpoint} contains non-UTF-8 data in "
                    f"{filename!r}"
                ) from exc
            return _a3m_rows_from_lines(text.splitlines(), filename)

        uniref_rows = rows_for(COLABFOLD_UNIREF_RESULT)
        environment_rows = rows_for(COLABFOLD_ENV_RESULT)
    return uniref_rows + environment_rows[1:]


def _colabfold_alignment(
    query_sequence: str,
    host: str,
    sleep_fn: Any,
) -> list[str]:
    submit_endpoint = _colabfold_url(host, COLABFOLD_SUBMIT_PATH)
    resubmits = 0

    while True:
        submission_payload = _colabfold_json(
            _send_colabfold_request(
                build_public_msa_request(query_sequence, host_url=host)
            ),
            submit_endpoint,
        )
        status = _colabfold_status(submission_payload, submit_endpoint)
        if status == COLABFOLD_STATUS_RATELIMIT:
            resubmits = _wait_for_colabfold_ratelimit(host, resubmits, sleep_fn)
            continue

        job_id_value = submission_payload.get("id")
        if job_id_value is None or str(job_id_value) == "":
            raise SourceUnavailable(
                f"ColabFold MSA submit endpoint {submit_endpoint} returned status "
                f"{status} without an id"
            )
        job_id = str(job_id_value)

        while status in (COLABFOLD_STATUS_PENDING, COLABFOLD_STATUS_RUNNING):
            sleep_fn(COLABFOLD_WAIT_SECONDS)
            status_endpoint = _colabfold_url(
                host,
                COLABFOLD_STATUS_PATH.format(id=job_id),
            )
            status_payload = _colabfold_json(
                _colabfold_request(status_endpoint),
                status_endpoint,
            )
            status = _colabfold_status(status_payload, status_endpoint)
            if status == COLABFOLD_STATUS_RATELIMIT:
                resubmits = _wait_for_colabfold_ratelimit(host, resubmits, sleep_fn)
                break

        if status == COLABFOLD_STATUS_RATELIMIT:
            continue
        if status == COLABFOLD_STATUS_COMPLETE:
            return _colabfold_result_rows(host, job_id)

        raise SourceUnavailable(
            f"ColabFold MSA submit endpoint {submit_endpoint} returned unexpected "
            f"terminal status {status!r}"
        )


def resolve_alignment_route(route: str | None, source: str | None) -> str:
    if route is not None and source is not None:
        raise SourceUnavailable("choose either --route or --source, not both")
    selected = route if route is not None else source
    if selected is None:
        raise SourceUnavailable(ROUTE_SELECTION_ERROR)
    aliases = {
        LEGACY_SOURCE_COLABFOLD_SERVER: ROUTE_PUBLIC_SERVER,
        LEGACY_SOURCE_LOCAL_MMSEQS2: ROUTE_LOCAL,
        SOURCE_SUPPLIED_A3M: ROUTE_PRECOMPUTED,
    }
    return aliases.get(selected, selected)


def build_alignment(
    source: str | None = None,
    query_sequence: str = "",
    *,
    route: str | None = None,
    allow_public_msa: bool = False,
    host_url: str | None = None,
    sleep_fn: Any = time.sleep,
) -> list[str]:
    """Return the alignment rows for one target chain, query first.

    The public route requires explicit consent because it sends the target
    sequence to the configured ColabFold host.
    """
    selected = resolve_alignment_route(route, source)
    if selected == SOURCE_QUERY_ONLY:
        return [query_sequence]
    if selected == ROUTE_PUBLIC_SERVER:
        if not allow_public_msa:
            raise SourceUnavailable(
                "route public-server requires explicit --allow-public-msa "
                f"consent before sending the target sequence to {COLABFOLD_MSA_SERVER_HOST}"
            )
        host = COLABFOLD_MSA_SERVER_HOST if host_url is None else host_url
        print(
            f"Public MSA route: host {host}; sending q with the target protein "
            f"sequence as FASTA and mode={COLABFOLD_MSA_MODE}.",
            file=sys.stderr,
        )
        return _colabfold_alignment(query_sequence, host, sleep_fn)
    if selected == ROUTE_LOCAL:
        raise SourceUnavailable(
            "the local route is unavailable because this adapter has no MMseqs2 database "
            "path or execution implementation. Use --route public-server with "
            "--allow-public-msa, or use --route precomputed."
        )
    if selected == ROUTE_PRECOMPUTED:
        raise SourceUnavailable(
            "route precomputed is handled by run_stage and cannot build an alignment"
        )
    raise SourceUnavailable(f"unregistered alignment source: {selected}")


def write_a3m(path: Path, target_id: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, row in enumerate(rows):
        lines.append(f">{target_id}" if index == 0 else f">{target_id}-homolog-{index}")
        lines.append(row)
    path.write_text("\n".join(lines) + "\n")


def run_stage(args: argparse.Namespace) -> int:
    selected_source = resolve_alignment_route(
        getattr(args, "route", None),
        getattr(args, "source", None),
    )
    config = load_json(args.config)
    stage = stage_record(config, args.stage)
    targets = [target for target in config["targets"] if isinstance(target, dict)]
    supplied = resolve_per_target(args.target_sequence, targets, "--target-sequence")
    target_manifests = completed_target_manifests(
        args.receipts_dir,
        getattr(args, "target_stage_id", DEFAULT_TARGET_STAGE_ID),
        getattr(args, "target_artifact_id", DEFAULT_TARGET_ARTIFACT_ID),
    )
    precomputed_a3m = resolve_per_target(
        [
            *(getattr(args, "precomputed_a3m", None) or []),
            *(getattr(args, "supplied_a3m", None) or []),
        ],
        targets,
        "--precomputed-a3m",
    )
    manifest_path = artifact_output_path(
        stage, MANIFEST_ARTIFACT_ID, args.attempt_dir, args.phase
    )
    rows: list[dict[str, Any]] = []
    for target in targets:
        target_id = str(target["target_id"])
        chain_id = design_target_chain(target)
        query_sequence, query_source = query_sequence_for(
            target,
            chain_id,
            supplied,
            target_manifests.get(target_id),
        )
        source_path: str | None = None
        if selected_source == ROUTE_PRECOMPUTED:
            supplied_path = precomputed_a3m.get(target_id)
            if supplied_path is None:
                raise ValueError(
                    "route precomputed requires --precomputed-a3m TARGET_ID=PATH"
                )
            alignment = clean_a3m_rows(
                supplied_a3m_rows(
                    supplied_path,
                    query_sequence,
                    getattr(args, "minimum_sequences", None),
                    target_id=target_id,
                    chain_id=chain_id,
                ),
                query_sequence,
            )
            source_path = str(Path(supplied_path).resolve())
        else:
            alignment = clean_a3m_rows(
                build_alignment(
                    selected_source,
                    query_sequence,
                    allow_public_msa=getattr(args, "allow_public_msa", False),
                ),
                query_sequence,
            )
        a3m_path = artifact_output_path(
            stage, f"{MSA_ARTIFACT_ID_PREFIX}{target_id}", args.attempt_dir, args.phase
        )
        write_a3m(a3m_path, target_id, alignment)
        row = {
            "target_id": target_id,
            "target_chain_id": chain_id,
            "msa_path": str(a3m_path.resolve()),
            "msa_sha256": sha256_file(a3m_path),
            "msa_depth": len(alignment),
            "msa_source": selected_source,
            "query_sequence_sha256": sha256_text(query_sequence),
            "query_sequence_source": query_source,
        }
        if source_path is not None:
            row["msa_source_path"] = source_path
        rows.append(row)
    write_jsonl(manifest_path, rows)
    return 0


def parse_stage(args: argparse.Namespace) -> int:
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
                else:
                    parsed_count += 1
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    write_json(
        args.attempt_dir / args.phase / "parser-result.json",
        {
            "ok": bool(files) and not errors,
            "parsed_count": parsed_count,
            "rejected_count": len(errors),
            "errors": errors,
            "source_output_hashes": sorted(sha256_file(path) for path in files),
        },
    )
    return 0 if files and not errors else 1


def build_parser() -> argparse.ArgumentParser:
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
        if name != "run":
            continue
        subparser.add_argument(
            "--source",
            choices=SOURCES,
            help=(
                "Compatibility route selector. Use --route public-server or --route "
                "local for the explicit public or private choice. query-only writes "
                "the query only, and supplied-a3m maps to --route precomputed."
            ),
        )
        subparser.add_argument(
            "--route",
            choices=ROUTES,
            help=(
                "Choose public-server to send the target sequence to a third-party "
                "ColabFold server, local to use a local MMseqs2 database, or "
                "precomputed to stage a named alignment file."
            ),
        )
        subparser.add_argument(
            "--allow-public-msa",
            action="store_true",
            help=(
                "Consent to send the target protein sequence as FASTA and mode=env "
                f"to {COLABFOLD_MSA_SERVER_HOST} when the route is public-server."
            ),
        )
        subparser.add_argument(
            "--target-sequence",
            action="append",
            metavar="TARGET_ID=SEQUENCE_OR_FASTA",
            help=(
                "The target chain's query sequence, as a literal or a path to a "
                "FASTA. Repeat once per target. With no argument the sequence uses "
                "the target-preparation artifact, then the configured structure."
            ),
        )
        subparser.add_argument(
            "--target-stage-id",
            default=DEFAULT_TARGET_STAGE_ID,
            help=f"Upstream target stage ID. Defaults to {DEFAULT_TARGET_STAGE_ID}.",
        )
        subparser.add_argument(
            "--target-artifact-id",
            default=DEFAULT_TARGET_ARTIFACT_ID,
            help=f"Upstream target artifact ID. Defaults to {DEFAULT_TARGET_ARTIFACT_ID}.",
        )
        subparser.add_argument(
            "--precomputed-a3m",
            action="append",
            metavar="TARGET_ID=PATH",
            help=(
                "The prepared unpaired target-chain a3m. Repeat once per target "
                "when --route is precomputed."
            ),
        )
        subparser.add_argument(
            "--supplied-a3m",
            action="append",
            metavar="TARGET_ID=PATH",
            help=(
                "Compatibility name for --precomputed-a3m when --source is "
                "supplied-a3m."
            ),
        )
        subparser.add_argument(
            "--minimum-sequences",
            type=int,
            help=(
                "Optional sequence floor for a supplied a3m."
            ),
        )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "toolcheck":
        print("target msa builder ok")
        return 0
    try:
        return run_stage(args) if args.command == "run" else parse_stage(args)
    except (QuerySequenceUnavailable, SourceUnavailable, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
