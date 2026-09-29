#!/usr/bin/env python3
"""Call one supplied fal RFdiffusion3 deployment and save exact files.

The client is standard-library only. It reads a named variable from its own
environment, defaulting to ``FAL_KEY``. Where the caller does not carry it, run the client through
``fal-credential-wrapper exec-model``, which supplies it from the machine's protected
credential store.
"""

import argparse
import base64
import hashlib
import json
import os
import re
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
STRUCTURE_SUFFIX = ".cif.gz"
OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REQUIRED_RECEIPT_FIELDS = (
    "checkpoint_sha256",
    "checkpoint_bytes",
    "environment_identity",
    "source_revision",
    "device",
)
OPTIONAL_RUNTIME_FIELDS = (
    "gpu_uuid",
    "gpu_pci_bus_id",
    "host_name",
    "driver_version",
    "cuda_runtime_version",
    "runner_id",
    "cudnn_version",
    "determinism_mode",
    "torch_deterministic_algorithms",
    "torch_deterministic_warn_only",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32",
    "cublas_workspace_config",
)


class ClientError(RuntimeError):
    pass


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "fal redirect rejected", headers, fp
        )


def endpoint_url(team: str, app: str, suffix: str = "") -> str:
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
        raise ClientError(
            f"fal endpoint must be exactly https://{FAL_HOST}{expected_path}"
        )


def credential_environment_name(value: str) -> str:
    """Validate an environment-variable name without reading its value."""
    if not isinstance(value, str) or ENVIRONMENT_NAME_RE.fullmatch(value) is None:
        raise ClientError(
            "--credential-env must be an environment-variable name such as FAL_KEY"
        )
    return value


def credential(credential_env: str = DEFAULT_CREDENTIAL_ENV) -> str:
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


def post(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
    error_dir: Path,
    credential_env: str = DEFAULT_CREDENTIAL_ENV,
) -> dict[str, Any]:
    if timeout_seconds < 1 or timeout_seconds > DEFAULT_TIMEOUT_SECONDS:
        raise ClientError(
            f"timeout must be between 1 and {DEFAULT_TIMEOUT_SECONDS} seconds"
        )
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
    if OUTPUT_NAME_RE.fullmatch(name) is None:
        raise ClientError(f"application returned an unsafe file name: {name!r}")
    return out_dir / name


def write_returned_files(out_dir: Path, response: dict[str, Any]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    designs = response.get("designs")
    if not isinstance(designs, list) or not designs:
        raise ClientError("application returned no designs")
    written: list[Path] = []
    seen: set[str] = set()
    for index, design in enumerate(designs):
        if not isinstance(design, dict):
            raise ClientError(f"design {index} is not an object")
        name = str(design.get("name", ""))
        if name in seen:
            raise ClientError(f"application returned duplicate design name {name!r}")
        seen.add(name)
        if not name.endswith(STRUCTURE_SUFFIX):
            raise ClientError(f"design {index} is not a {STRUCTURE_SUFFIX} file")
        encoded = design.get("gzip_b64")
        if not isinstance(encoded, str) or not encoded:
            raise ClientError(f"design {index} contains no gzip_b64")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ClientError(f"design {index} contains invalid base64") from exc
        observed = hashlib.sha256(payload).hexdigest()
        recorded = design.get("sha256")
        if not isinstance(recorded, str) or recorded != observed:
            raise ClientError(
                f"design {index} hash mismatch: observed {observed}, recorded {recorded}"
            )
        path = output_path(out_dir, name)
        path.write_bytes(payload)
        written.append(path)
    for record in response.get("metadata", []):
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        text = record.get("text")
        if isinstance(name, str) and isinstance(text, str):
            output_path(out_dir, name).write_text(text)
    return written


def write_receipt(
    path: Path,
    response: dict[str, Any],
    url: str,
    client_seconds: float,
    requested_seed: int,
) -> None:
    missing = [
        field
        for field in REQUIRED_RECEIPT_FIELDS
        if response.get(field) in (None, "")
    ]
    if missing:
        raise ClientError(f"application reported no {', '.join(missing)}")
    used_seed = response.get("used_seed")
    if not isinstance(used_seed, int) or isinstance(used_seed, bool):
        raise ClientError("application reported no integer used_seed")
    receipt = {field: response.get(field) for field in REQUIRED_RECEIPT_FIELDS}
    receipt.update({field: response.get(field) for field in OPTIONAL_RUNTIME_FIELDS})
    receipt.update(
        {
            "requested_seed": requested_seed,
            "used_seed": used_seed,
            "fal_endpoint": url,
            "request_id": response.get("request_id"),
            "client_wall_seconds": round(client_seconds, 3),
            "runner_wall_seconds": response.get("seconds"),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def load_specification(path: Path, structure_name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ClientError(f"could not read specification {path}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ClientError("specification must be a non-empty JSON object")
    for key, entry in value.items():
        if not isinstance(entry, dict):
            raise ClientError(f"specification entry {key!r} is not an object")
        entry["input"] = structure_name
    return value


def run(args: argparse.Namespace) -> int:
    specification_path = args.specification.expanduser()
    structure_path = args.input_structure.expanduser()
    if not specification_path.is_file():
        raise ClientError(f"specification not found: {specification_path}")
    if not structure_path.is_file():
        raise ClientError(f"input structure not found: {structure_path}")
    out_dir = args.out_dir.expanduser()
    team, app = endpoint_parts(args.fal_url)
    url = endpoint_url(team, app)
    validate_url(url, team, app)
    payload = {
        "request_id": args.request_id or specification_path.parent.name,
        "specification": load_specification(specification_path, structure_path.name),
        "input_structure_name": structure_path.name,
        "input_structure_text": structure_path.read_text(errors="replace"),
        "diffusion_batch_size": args.diffusion_batch_size,
        "n_batches": args.n_batches,
        "seed": args.seed,
        "determinism_mode": args.determinism_mode,
        "step_scale": args.step_scale,
        "gamma_0": args.gamma_0,
        "overrides": list(args.override),
    }
    started = time.monotonic()
    response = post(
        url,
        payload,
        args.timeout_seconds,
        out_dir,
        credential_env=args.credential_env,
    )
    client_seconds = time.monotonic() - started
    written = write_returned_files(out_dir, response)
    write_receipt(args.receipt.expanduser(), response, url, client_seconds, args.seed)
    expected = args.diffusion_batch_size * args.n_batches
    if len(written) != expected:
        raise ClientError(f"application returned {len(written)} designs, expected {expected}")
    (out_dir / "response.json").write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"rfd3 fal client: designs={len(written)} seconds={client_seconds:.1f} "
        f"requested_seed={args.seed} used_seed={response['used_seed']} "
        f"device={response['device']} out_dir={out_dir}"
    )
    return 0


def toolcheck(args: argparse.Namespace) -> int:
    out_dir = args.out_dir.expanduser()
    team, app = endpoint_parts(args.fal_url)
    url = endpoint_url(team, app, "/toolcheck")
    validate_url(url, team, app, "/toolcheck")
    started = time.monotonic()
    response = post(
        url,
        {"request_id": args.request_id or "toolcheck"},
        args.timeout_seconds,
        out_dir,
        credential_env=args.credential_env,
    )
    seconds = time.monotonic() - started
    missing = [field for field in REQUIRED_RECEIPT_FIELDS if response.get(field) in (None, "")]
    if missing:
        raise ClientError(f"application reported no {', '.join(missing)}")
    (out_dir / "response.json").write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n"
    )
    print(f"rfd3 fal client: device {response['device']}")
    print(f"rfd3 fal client: checkpoint {response['checkpoint_sha256']} ({response['checkpoint_bytes']} bytes)")
    print(f"rfd3 fal client: environment {response['environment_identity']}")
    print(f"rfd3 fal client: probe took {seconds:.1f} seconds")
    return 0


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("toolcheck")
    add_shared_arguments(check_parser)
    run_parser = subparsers.add_parser("run")
    add_shared_arguments(run_parser)
    run_parser.add_argument("--specification", type=Path, required=True)
    run_parser.add_argument("--input-structure", type=Path, required=True)
    run_parser.add_argument("--receipt", type=Path, required=True)
    run_parser.add_argument("--diffusion-batch-size", type=int, required=True)
    run_parser.add_argument("--n-batches", type=int, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument(
        "--determinism-mode", choices=("off", "warn", "strict"), default="off"
    )
    run_parser.add_argument("--step-scale", type=float, required=True)
    run_parser.add_argument("--gamma-0", type=float, required=True)
    run_parser.add_argument("--override", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        return toolcheck(args) if args.command == "toolcheck" else run(args)
    except ClientError as exc:
        print(f"rfd3 fal client: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
