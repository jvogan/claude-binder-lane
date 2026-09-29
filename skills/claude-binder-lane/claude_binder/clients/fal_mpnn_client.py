#!/usr/bin/env python3
"""Call one supplied fal ProteinMPNN deployment and save exact FASTA output.

The client accepts PDB, mmCIF, and mmCIF.GZ input. ProteinMPNN's pinned
upstream runner accepts PDB input, so the client converts mmCIF text in memory
before it sends the request. The original compressed file remains the input
artifact recorded in the receipt.

The client reads a named variable from its own environment, defaulting to
``FAL_KEY``. Where the caller already carries it, run the client directly. Where
it does not, run the client
through ``fal-credential-wrapper exec-model``, which supplies the credential from the machine's
protected credential store.
"""

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import shlex
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


FAL_HOST = "fal.run"
DEFAULT_CREDENTIAL_ENV = "FAL_KEY"
ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_TIMEOUT_SECONDS = 1200
DEFAULT_MODEL_NAME = "v_48_020"
DEFAULT_SAMPLING_TEMP = 0.1
MAXIMUM_TIMEOUT_SECONDS = 1200
OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FASTA_SUFFIX = ".fa"
REQUIRED_RECEIPT_FIELDS = (
    "checkpoint_sha256",
    "checkpoint_bytes",
    "environment_identity",
    "source_revision",
    "device",
)
# The request fields that decide which sequences come back. The receipt records them
# because the deployment answers with none of them, so a later reader has no other
# place to learn what this run asked for.
RECORDED_REQUEST_FIELDS = (
    "sequences_per_backbone",
    "sampling_temp",
    "model_name",
    "design_chain",
    "soluble_model",
    "ca_only",
)


class ClientError(RuntimeError):
    """A condition the operator must fix before the request can run."""


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse redirects that could carry the fal credential elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "fal redirect rejected", headers, fp
        )


def endpoint_url(team: str, app: str, suffix: str = "") -> str:
    """Return the fal URL for one supplied team and app."""
    return f"https://{FAL_HOST}/{team}/{app}{suffix}"


def endpoint_parts(value: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(value)
    segments = [segment for segment in parsed.path.split("/") if segment]
    if (
        parsed.scheme != "https"
        or parsed.hostname != FAL_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or len(segments) != 2
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ClientError("--fal-url must be exactly https://fal.run/<team>/<app>")
    return segments[0], segments[1]


def validate_url(url: str, team: str, app: str, suffix: str = "") -> None:
    """Require the exact fal URL selected by the command arguments."""
    parsed = urllib.parse.urlparse(url)
    expected_path = f"/{team}/{app}{suffix}"
    if (
        parsed.scheme != "https"
        or parsed.hostname != FAL_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.path != expected_path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ClientError(f"fal endpoint must be exactly https://{FAL_HOST}{expected_path}")


def credential_environment_name(value: str) -> str:
    """Validate an environment-variable name without reading its value."""
    if not isinstance(value, str) or ENVIRONMENT_NAME_RE.fullmatch(value) is None:
        raise ClientError(
            "--credential-env must be an environment-variable name such as FAL_KEY"
        )
    return value


def credential(credential_env: str = DEFAULT_CREDENTIAL_ENV) -> str:
    """Return the API credential from the selected process variable."""
    credential_env = credential_environment_name(credential_env)
    value = os.environ.get(credential_env, "").strip()
    if not value:
        raise ClientError(
            f"{credential_env} is unavailable. Set it in this process's environment, or run "
            "this client through the credential wrapper: fal-credential-wrapper exec-model -- ... "
            "Credential presence proves only presence. The provider authorization "
            "preflight verifies application access."
        )
    return value


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def read_input(path: Path) -> tuple[str, str, str]:
    """Read one input and return its PDB name, PDB text, and source digest."""
    if not path.is_file():
        raise ClientError(f"input structure not found: {path}")
    source_bytes = path.read_bytes()
    source_digest = sha256_bytes(source_bytes)
    if path.name.endswith(".cif.gz"):
        try:
            text = gzip.decompress(source_bytes).decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ClientError(f"could not decompress mmCIF input {path}: {exc}") from exc
        return path.name[:-7] + ".pdb", cif_to_pdb(text), source_digest
    if path.suffix == ".cif":
        return path.with_suffix(".pdb").name, cif_to_pdb(path.read_text()), source_digest
    if path.suffix == ".pdb":
        text = path.read_text()
        if "ATOM  " not in text:
            raise ClientError(f"PDB input has no ATOM records: {path}")
        return path.name, text, source_digest
    raise ClientError("input structure must end in .pdb, .cif, or .cif.gz")


def cif_to_pdb(text: str) -> str:
    """Convert the atom_site loop in one simple mmCIF file to PDB text."""
    columns: list[str] = []
    data_started = False
    atom_rows: list[dict[str, str]] = []
    in_loop = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped == "loop_":
            in_loop = True
            columns = []
            data_started = False
            continue
        if in_loop and not data_started and stripped.startswith("_atom_site."):
            columns.append(stripped.split()[0])
            continue
        if in_loop and columns and stripped.startswith("#"):
            if data_started:
                break
            continue
        if in_loop and columns and not data_started:
            data_started = True
        if in_loop and data_started:
            tokens = shlex.split(stripped, comments=False, posix=True)
            if len(tokens) < len(columns):
                raise ClientError(
                    f"mmCIF atom row has {len(tokens)} fields, expected {len(columns)}"
                )
            row = dict(zip(columns, tokens))
            if row.get("_atom_site.group_PDB") == "ATOM":
                atom_rows.append(row)

    required = {
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    }
    if not atom_rows or not required.issubset(columns):
        raise ClientError("mmCIF input has no complete atom_site ATOM loop")

    lines: list[str] = []
    serial = 1
    seen_models: set[str] = set()
    for row in atom_rows:
        model = row.get("_atom_site.pdbx_PDB_model_num", "1")
        if model not in ("?", ".") and model != "1":
            continue
        altloc = row.get("_atom_site.label_alt_id", " ")
        if altloc not in ("?", "."):
            continue
        chain = row.get("_atom_site.auth_asym_id", row.get("_atom_site.label_asym_id", "A"))
        resseq = row.get("_atom_site.auth_seq_id", row.get("_atom_site.label_seq_id", ""))
        if chain in ("?", ".") or resseq in ("?", "."):
            raise ClientError("mmCIF atom row has no chain or residue number")
        if len(chain) != 1 or not resseq.lstrip("-").isdigit():
            raise ClientError(f"mmCIF chain or residue cannot fit PDB fields: {chain} {resseq}")
        atom_name = row["_atom_site.label_atom_id"]
        resname = row["_atom_site.label_comp_id"]
        if len(atom_name) > 4 or len(resname) > 3:
            raise ClientError(f"mmCIF atom or residue name cannot fit PDB fields: {atom_name} {resname}")
        try:
            x = float(row["_atom_site.Cartn_x"])
            y = float(row["_atom_site.Cartn_y"])
            z = float(row["_atom_site.Cartn_z"])
            occupancy = float(row.get("_atom_site.occupancy", "1.0"))
            b_factor = float(row.get("_atom_site.B_iso_or_equiv", "0.0"))
        except ValueError as exc:
            raise ClientError(f"mmCIF atom row has invalid coordinates: {row}") from exc
        element = row.get("_atom_site.type_symbol", atom_name[:1]).upper()
        insertion = row.get("_atom_site.pdbx_PDB_ins_code", " ")
        if insertion in ("?", "."):
            insertion = " "
        lines.append(
            f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} {chain:1s}"
            f"{int(resseq):4d}{insertion:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occupancy:6.2f}{b_factor:6.2f}          {element:>2s}\n"
        )
        serial += 1
        seen_models.add(model)
    if not lines:
        raise ClientError("mmCIF input has no atoms from model 1")
    lines.append("END\n")
    return "".join(lines)


def post(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
    error_dir: Path,
    credential_env: str = DEFAULT_CREDENTIAL_ENV,
) -> dict[str, Any]:
    """Post one JSON request and return one JSON object."""
    if timeout_seconds < 1 or timeout_seconds > MAXIMUM_TIMEOUT_SECONDS:
        raise ClientError(f"timeout must be between 1 and {MAXIMUM_TIMEOUT_SECONDS} seconds")
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Key {credential(credential_env)}",
            "Content-Type": "application/json",
            "X-Fal-No-Retry": "1",
            "X-Fal-Request-Timeout": str(timeout_seconds),
            "X-App-Fal-Disable-Fallback": "1",
        },
    )
    error_dir.mkdir(parents=True, exist_ok=True)
    try:
        opener = urllib.request.build_opener(RejectRedirects())
        with opener.open(request, timeout=timeout_seconds + 60) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        (error_dir / "error.txt").write_text(f"HTTP {exc.code}\n{detail}\n")
        try:
            parsed = json.loads(detail)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            (error_dir / "error.json").write_text(
                json.dumps(parsed, indent=2, sort_keys=True) + "\n"
            )
        raise ClientError(f"application answered HTTP {exc.code}: {detail[-8000:]}") from exc
    except urllib.error.URLError as exc:
        raise ClientError(f"application could not be reached: {exc.reason}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClientError(f"application answered {len(raw)} non-JSON bytes") from exc
    if not isinstance(value, dict):
        raise ClientError("application answered a JSON value that is not an object")
    return value


def output_path(out_dir: Path, name: str) -> Path:
    """Return one safe FASTA path below the output directory."""
    if OUTPUT_NAME_RE.fullmatch(name) is None or not name.endswith(FASTA_SUFFIX):
        raise ClientError(f"application returned an unsafe FASTA name: {name!r}")
    return out_dir / "seqs" / name


def write_returned_files(out_dir: Path, response: dict[str, Any]) -> list[Path]:
    """Verify and write the exact FASTA bytes returned by fal."""
    files = response.get("files")
    if not isinstance(files, list) or not files:
        raise ClientError("application returned no FASTA files")
    written: list[Path] = []
    seen: set[str] = set()
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            raise ClientError(f"returned FASTA record {index} is not an object")
        name = str(record.get("name", ""))
        if name in seen:
            raise ClientError(f"application returned duplicate FASTA name: {name}")
        seen.add(name)
        encoded = record.get("bytes_b64")
        expected = record.get("sha256")
        expected_bytes = record.get("bytes")
        if not isinstance(encoded, str) or not encoded:
            raise ClientError(f"returned FASTA record {index} carries no bytes_b64")
        if not isinstance(expected, str) or not expected:
            raise ClientError(f"returned FASTA record {index} carries no sha256")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ClientError(f"returned FASTA record {index} carries invalid base64") from exc
        observed = sha256_bytes(payload)
        if observed != expected:
            raise ClientError(
                f"returned FASTA record {index} hashes to {observed}, and the app recorded {expected}"
            )
        if isinstance(expected_bytes, int) and len(payload) != expected_bytes:
            raise ClientError(
                f"returned FASTA record {index} is {len(payload)} bytes, and the app recorded "
                f"{expected_bytes}"
            )
        path = output_path(out_dir, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        written.append(path)
    return written


def write_receipt(
    path: Path,
    response: dict[str, Any],
    url: str,
    client_seconds: float,
    source_path: Path,
    source_sha256: str,
    uploaded_sha256: str,
    request: dict[str, Any],
) -> None:
    """Write runtime, input, and request identity beside the FASTA output.

    The deployment answers with no seed and no sampling temperature, so a reader who
    has only this receipt cannot tell which seed produced these sequences unless the
    client records the one it sent. `requested_seed` is the name the RFdiffusion3
    client already uses for the same value.
    """
    missing = [field for field in REQUIRED_RECEIPT_FIELDS if response.get(field) in (None, "")]
    if missing:
        raise ClientError(f"application reported no {', '.join(missing)}")
    receipt = {field: response[field] for field in REQUIRED_RECEIPT_FIELDS}
    receipt.update(
        {
            "fal_endpoint": url,
            "request_id": response.get("request_id"),
            "client_wall_seconds": round(client_seconds, 3),
            "runner_wall_seconds": response.get("seconds"),
            "torch_version": response.get("torch_version"),
            "input_structure_path": str(source_path),
            "input_structure_sha256": source_sha256,
            "uploaded_pdb_sha256": uploaded_sha256,
            "requested_seed": request["seed"],
        }
    )
    receipt.update({field: request[field] for field in RECORDED_REQUEST_FIELDS})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    """Design sequences on one supplied backbone."""
    if args.soluble_model and args.ca_only:
        raise ClientError(
            "--soluble-model and --ca-only select incompatible checkpoints. SolubleMPNN ships no C-alpha checkpoint."
        )
    source_path = args.input_structure.expanduser()
    input_name, input_text, source_sha256 = read_input(source_path)
    uploaded_sha256 = sha256_bytes(input_text.encode("utf-8"))
    team, app = endpoint_parts(args.fal_url)
    url = endpoint_url(team, app)
    validate_url(url, team, app)
    if len(args.design_chain) != 1 or not args.design_chain.isalnum():
        raise ClientError("--design-chain must contain one letter or digit")
    if args.sequences_per_backbone < 1 or args.sequences_per_backbone > 8:
        raise ClientError("--sequences-per-backbone must be between 1 and 8")
    payload = {
        "request_id": args.request_id or source_path.stem,
        "input_structure_name": input_name,
        "input_structure_text": input_text,
        "sequences_per_backbone": args.sequences_per_backbone,
        "seed": args.seed,
        "sampling_temp": args.sampling_temp,
        "model_name": args.model_name,
        "design_chain": args.design_chain,
        "soluble_model": args.soluble_model,
        "ca_only": args.ca_only,
    }
    started = time.monotonic()
    response = post(
        url,
        payload,
        args.timeout_seconds,
        args.out_dir.expanduser(),
        credential_env=args.credential_env,
    )
    client_seconds = time.monotonic() - started
    written = write_returned_files(args.out_dir.expanduser(), response)
    write_receipt(
        args.receipt.expanduser(),
        response,
        url,
        client_seconds,
        source_path,
        source_sha256,
        uploaded_sha256,
        payload,
    )
    if len(written) != 1:
        raise ClientError(f"application returned {len(written)} FASTA files and one was expected")
    (args.out_dir.expanduser() / "response.json").write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"mpnn fal client: files={len(written)} seconds={client_seconds:.1f} "
        f"model={'soluble' if args.soluble_model else 'protein'} "
        f"design_chain={args.design_chain} out_dir={args.out_dir}"
    )
    return 0


def toolcheck(args: argparse.Namespace) -> int:
    """Report the selected runtime without designing a sequence."""
    if args.soluble_model and args.ca_only:
        raise ClientError(
            "--soluble-model and --ca-only select incompatible checkpoints. SolubleMPNN ships no C-alpha checkpoint."
        )
    team, app = endpoint_parts(args.fal_url)
    url = endpoint_url(team, app, "/toolcheck")
    validate_url(url, team, app, "/toolcheck")
    started = time.monotonic()
    response = post(
        url,
        {
            "request_id": args.request_id or "toolcheck",
            "model_name": args.model_name,
            "soluble_model": args.soluble_model,
            "ca_only": args.ca_only,
        },
        args.timeout_seconds,
        args.out_dir.expanduser(),
        credential_env=args.credential_env,
    )
    seconds = time.monotonic() - started
    missing = [field for field in REQUIRED_RECEIPT_FIELDS if response.get(field) in (None, "")]
    if missing:
        raise ClientError(f"application reported no {', '.join(missing)}")
    args.out_dir.expanduser().mkdir(parents=True, exist_ok=True)
    (args.out_dir.expanduser() / "response.json").write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n"
    )
    print(f"mpnn fal client: device {response['device']}")
    print(
        f"mpnn fal client: checkpoint {response['checkpoint_sha256']} "
        f"({response['checkpoint_bytes']} bytes)"
    )
    print(f"mpnn fal client: environment {response['environment_identity']}")
    print(f"mpnn fal client: probe took {seconds:.1f} seconds")
    return 0


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by toolcheck and run."""
    parser.add_argument("--fal-url", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--credential-env",
        type=credential_environment_name,
        default=DEFAULT_CREDENTIAL_ENV,
        help=(
            "Environment-variable name holding the fal credential. The value is read "
            "only inside this process and is never accepted on the command line."
        ),
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the client command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("toolcheck")
    add_shared_arguments(check_parser)
    check_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    check_parser.add_argument("--soluble-model", action="store_true")
    check_parser.add_argument("--ca-only", action="store_true")
    run_parser = subparsers.add_parser("run")
    add_shared_arguments(run_parser)
    run_parser.add_argument("--input-structure", type=Path, required=True)
    run_parser.add_argument("--receipt", type=Path, required=True)
    run_parser.add_argument("--sequences-per-backbone", type=int, default=1)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--sampling-temp", type=float, default=DEFAULT_SAMPLING_TEMP)
    run_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    run_parser.add_argument("--design-chain", required=True)
    run_parser.add_argument("--soluble-model", action="store_true")
    run_parser.add_argument("--ca-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the requested client command and report caller-fixable errors."""
    args = parse_arguments(argv)
    try:
        return toolcheck(args) if args.command == "toolcheck" else run(args)
    except ClientError as exc:
        print(f"mpnn fal client: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
