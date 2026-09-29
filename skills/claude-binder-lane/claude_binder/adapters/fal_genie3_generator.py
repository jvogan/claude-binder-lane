#!/usr/bin/env python3
"""Generate Genie3 binder backbones on an operator-deployed fal application.

This is the hosted route to the `genie3-generator` slot. `genie3_generator.py`
is the local route and stays the reference for everything this stage publishes.
Both write the same two artifacts into the current attempt directory:

  <attempt>/<phase>/poses/<candidate_id>.pdb        one design pose per candidate
  <attempt>/<phase>/candidate-manifest.jsonl        one row per candidate

They write them with the same code. This module imports `genie3_generator` and
calls its dataset composer, its chain resolver, its pose writer, and its parser.
The only step it replaces is the one that costs money: the local route runs
`genie3 generate` as a subprocess, and this route posts the same inputs to a
deployed application and writes the structures the application returns into the
directory layout the local run produces.

A downstream stage can tell the two apart, and should be able to. Every row
carries `runner_protocol`, which is the constant `"fal"` here and the resolved
`"local"` or `"modal"` on the other route. The hosted row also carries the
endpoint, the receipt path and the runner identity the application reported.
What matches is the pose, the parser result and the row contract, so a consumer
that reads the contract does not care which route produced the row.

**The adapter dispatches. It never generates.** The application composes the
Genie3 configuration itself from the request fields, so no configuration file
exists on this machine and the rows record no local configuration hash. The
weights live on the runner, so the rows record no local weights directory. What
the runner served is recorded instead: `checkpoint_sha256`, `checkpoint_bytes`,
`environment_identity`, `source_revision`, and `device` all come off the
response, and the run refuses to write a manifest when the response omits any of
them.

**The credential enters neither this process nor an argument list.** Every
subcommand that reaches the network builds one child command through
`clients/fal_invocation`, which picks the direct route when the credential is
already in this process and the wrapper route otherwise. `run` spawns
`dispatch`, `probe` spawns `dispatch-probe`, and those two children hold every
socket this module opens. Splitting the process is what lets the wrapper put the
credential in front of the request without it passing through this one. It also
keeps the credential off this process's own leak surfaces: a stack trace, a
logged request object, an error message that echoes a header.

**The output is a C-alpha trace.** Genie3 binder mode writes `CA` atoms and no
`N`, `C`, or `O`. Nothing here parses an atom name, synthesizes an atom, or
requires a full backbone. The returned bytes are decoded, hashed, and written
unchanged, and the atom names a row reports are the ones
`genie3_generator.atom_names` observed in those bytes.

**The cost basis is unpriced.** Every Genie3 cost figure this project holds was
measured on RunPod. Nothing prices a fal runner for this tool, so the receipt
records `cost_basis: unpriced` and no number. A RunPod figure is not a fal price.

The endpoint has no default. Pass --fal-url, or set GENIE3_FAL_URL, with the
exact application URL of your own deployment. A deployment name baked into this
file would name somebody else's account.

Subcommands:

  toolcheck        Report this adapter's own readiness. Sends no request, reads
                   no credential, and costs nothing.
  run              Compose the problem, dispatch one phase, write the manifest.
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
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from claude_binder.adapters import genie3_generator as base
from claude_binder.clients import fal_invocation
from claude_binder.paths import child_process_environment, package_file


# The children `run` and `probe` spawn are this same file. The two-file
# adapter-and-client split every other fal binding uses exists so the credential
# wrapper has a process to exec; one file with dispatch subcommands gives the
# wrapper the same seam without a second module.
DISPATCH_SCRIPT = package_file("adapters", "fal_genie3_generator.py")
# The two child subcommands. Every request this module sends leaves from one of
# them, so no parent process reads the credential or opens a socket.
DISPATCH_COMMAND = "dispatch"
PROBE_CHILD_COMMAND = "dispatch-probe"
DEFAULT_FAL_EXECUTABLE = "fal-credential-wrapper"
FAL_URL_ENVIRONMENT_KEY = "GENIE3_FAL_URL"
TOOLCHECK_PATH = "/toolcheck"
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_RECEIPT_NAME = "fal-receipt.json"
RUNNER_PROTOCOL = "fal"
# `qualify.COST_KIND_UNPRICED`. Nothing measured this tool on this provider.
COST_BASIS = "unpriced"
# The application caps n_sample at 32 and bounds direction_scale and seed. A
# request outside any of them is refused by the service before it generates, so
# it is refused here before it is sent.
MAXIMUM_SAMPLES = 32
MAXIMUM_DIRECTION_SCALE = 100.0
MAXIMUM_SEED = 2**32 - 1
MAXIMUM_REQUEST_ID_LENGTH = 128
# A returned name becomes a path under the output directory, so it has to be one
# plain file name. Anything else is refused rather than joined.
RETURNED_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$")
FAL_ROUTE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
FAL_HOSTNAME = "fal.run"
STRUCTURE_SUFFIX = ".pdb"
REDIRECT_STATUS_CODES = (301, 302, 303, 307, 308)
# The five fields `proteinmpnn_designer.REQUIRED_FAL_RECEIPT_FIELDS` names. A
# response missing any of them identifies no runtime, so the run stops rather
# than attribute structures to a runner it cannot name.
REQUIRED_RESPONSE_FIELDS = (
    "checkpoint_sha256",
    "checkpoint_bytes",
    "environment_identity",
    "source_revision",
    "device",
)
OPTIONAL_RESPONSE_FIELDS = (
    "gpu_uuid",
    "driver_version",
    "host_name",
    "runner_id",
    "torch_version",
    "cuda_runtime_version",
    "weights_revision",
    "weights_file_count",
)


class AdapterError(base.AdapterError):
    """A fal Genie3 input, response, or returned artifact is invalid."""


# ----------------------------------------------------------------------------
# The endpoint and the credential route.
# ----------------------------------------------------------------------------


def resolve_endpoint(value: str | None) -> str:
    """Return the deployed application URL, refusing anything but one fal route.

    There is no default. A literal here would name the deployment of whoever
    wrote the file rather than the deployment the operator pays for.

    The shape check is `proteinmpnn_designer.resolve_fal_route`. It matters
    because the request carries an `Authorization` header, and a URL pointing
    somewhere else would hand a credential to a host the operator never chose.
    """
    url = (value or os.environ.get(FAL_URL_ENVIRONMENT_KEY, "")).strip()
    if not url:
        raise AdapterError(
            "the fal Genie3 route needs an endpoint and has no default. Pass --fal-url, or "
            f"set {FAL_URL_ENVIRONMENT_KEY}, with the application URL of your own deployment"
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
        or any(
            FAL_ROUTE_SEGMENT_RE.fullmatch(segment) is None for segment in segments[1:]
        )
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
    args: argparse.Namespace, child: str, endpoint: str, *values: str
) -> list[str]:
    """Return the child command that carries the credential, never the credential.

    `child` is the subcommand of this same file that opens the socket, either
    `dispatch` for a generation or `dispatch-probe` for a runtime report. The
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
            DISPATCH_SCRIPT,
            child_values,
            requested=route,
            credential_env_key=credential_env,
        )
    except fal_invocation.RouteError as exc:
        raise AdapterError(str(exc)) from exc


def run_external(argv: list[str], label: str) -> None:
    """Run one child command and fail on a nonzero exit.

    The printed line is redacted, because the endpoint identifies an account.
    The child inherits an environment that can import this package, because the
    kernel that loads this package by file path leaves its parent off `sys.path`.
    """
    print(f"genie3 fal adapter: {label}: {fal_invocation.redacted_command(argv)}", flush=True)
    completed = subprocess.run(
        argv, shell=False, check=False, env=child_process_environment()
    )
    if completed.returncode != 0:
        raise AdapterError(f"{label} exited {completed.returncode}")


# ----------------------------------------------------------------------------
# The transport. Only the child subcommands reach this, never a parent process.
# ----------------------------------------------------------------------------


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse to follow a redirect away from the endpoint the caller named."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
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
        raise AdapterError(
            f"the application answered {len(raw)} bytes that are not JSON"
        ) from exc
    if not isinstance(value, dict):
        raise AdapterError("the application answered a JSON value that is not an object")
    return value


# ----------------------------------------------------------------------------
# The request and the response.
# ----------------------------------------------------------------------------


def request_id(value: str | None, fallback: str) -> str:
    """Return a request identifier the application accepts."""
    candidate = (value or fallback).strip()
    if not candidate or len(candidate) > MAXIMUM_REQUEST_ID_LENGTH:
        raise AdapterError(
            f"the request id has to be 1 to {MAXIMUM_REQUEST_ID_LENGTH} characters: {candidate!r}"
        )
    return candidate


def validate_request_values(
    *, n_sample: int, direction_scale: float, seed: int, problem_id: str, model_version: str
) -> None:
    """Refuse a request the application would reject, before it is paid for."""
    if not 1 <= n_sample <= MAXIMUM_SAMPLES:
        raise AdapterError(
            f"--count is {n_sample}; the application accepts 1 to {MAXIMUM_SAMPLES} samples per "
            "request. Split a larger phase across requests"
        )
    if not 0.0 <= direction_scale <= MAXIMUM_DIRECTION_SCALE:
        raise AdapterError(
            f"--direction-scale is {direction_scale}; the application accepts 0.0 to "
            f"{MAXIMUM_DIRECTION_SCALE}"
        )
    if not 0 <= seed <= MAXIMUM_SEED:
        raise AdapterError(f"--seed is {seed}; the application accepts 0 to {MAXIMUM_SEED}")
    for label, value in (("--problem-id", problem_id), ("--model-version", model_version)):
        if base.IDENTIFIER_RE.fullmatch(value) is None:
            raise AdapterError(f"{label} is not a plain identifier: {value}")


def build_payload(
    *,
    identifier: str,
    problem_id: str,
    problem: dict[str, Any],
    target_pdb: Path,
    target_fasta: Path,
    n_sample: int,
    direction_scale: float,
    seed: int,
    seed_config_key: str | None,
    allow_unseeded: bool,
    model_version: str,
) -> dict[str, Any]:
    """Return the JSON body of one generation request.

    The application re-homes every path inside the problem object onto the files
    it materializes from `target_pdb_text` and `target_fasta_text`, so nothing on
    this machine is opened by name on the runner. The dataset this package writes
    puts byte-identical content in the whole-target file and the per-chain file,
    which is why collapsing the two into one path on the runner changes nothing.

    `model_version` asserts which served weights tree the runner has to carry. It
    is a check on the runner, not a selector: the application confirms that
    `pretrained/<model_version>/config.yaml` exists and then writes a
    configuration that does not name the version at all.
    """
    if seed_config_key is None and not allow_unseeded:
        raise AdapterError(
            "no Genie3 invocation this package records passes a seed. Pass --seed-config-key "
            "with the configuration key your build reads, or --allow-unseeded to record that "
            "this request carries no seed"
        )
    for label, path in (("target PDB", target_pdb), ("target FASTA", target_fasta)):
        if not path.is_file():
            raise AdapterError(f"{label} not found: {path}")
        if RETURNED_NAME_RE.fullmatch(path.name) is None:
            raise AdapterError(f"{label} is not one plain file name: {path.name}")
    payload: dict[str, Any] = {
        "request_id": identifier,
        "problem_id": problem_id,
        "problem": problem,
        "target_pdb_name": target_pdb.name,
        "target_pdb_text": target_pdb.read_text(),
        "target_fasta_name": target_fasta.name,
        "target_fasta_text": target_fasta.read_text(),
        "n_sample": n_sample,
        "direction_scale": float(direction_scale),
        "seed": seed,
        "allow_unseeded": bool(allow_unseeded),
        "model_version": model_version,
    }
    if seed_config_key is not None:
        if base.CONFIG_KEY_RE.fullmatch(seed_config_key) is None:
            raise AdapterError(
                f"--seed-config-key is not a dotted identifier path: {seed_config_key}"
            )
        payload["seed_config_key"] = seed_config_key
    return payload


def decode_designs(response: dict[str, Any], expected: int) -> list[tuple[str, bytes]]:
    """Return the returned structures as names and bytes, refusing a bad one.

    Everything is decoded and checked before anything is written. A response
    that fails halfway then leaves no partial structure directory behind, which
    matters because the run refuses to start against a directory that already
    holds structures.
    """
    designs = response.get("designs")
    if not isinstance(designs, list) or not designs:
        raise AdapterError("the application returned no designs")
    if len(designs) != expected:
        raise AdapterError(
            f"the application returned {len(designs)} designs and the request asked for {expected}"
        )
    decoded: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for index, design in enumerate(designs):
        if not isinstance(design, dict):
            raise AdapterError(f"design {index} is not a JSON object")
        name = str(design.get("name", ""))
        if not name.endswith(STRUCTURE_SUFFIX):
            raise AdapterError(
                f"design {index} is named {name}, which is not a {STRUCTURE_SUFFIX}"
            )
        if RETURNED_NAME_RE.fullmatch(name) is None:
            raise AdapterError(f"the application returned a file name this adapter refuses: {name}")
        if name in seen:
            raise AdapterError(f"the application returned {name} twice")
        seen.add(name)
        encoded = design.get("pdb_b64")
        if not isinstance(encoded, str) or not encoded:
            raise AdapterError(f"design {index} carries no pdb_b64")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise AdapterError(
                f"design {index} carries base64 this adapter cannot decode"
            ) from exc
        recorded = design.get("sha256")
        observed = hashlib.sha256(payload).hexdigest()
        if isinstance(recorded, str) and recorded and recorded != observed:
            raise AdapterError(
                f"design {index} arrived with a different hash than the runner recorded: "
                f"{observed} against {recorded}"
            )
        decoded.append((name, payload))
    return decoded


def runtime_fields(response: dict[str, Any]) -> dict[str, Any]:
    """Return the runtime identity the response reports, refusing an incomplete one."""
    missing = [
        field
        for field in REQUIRED_RESPONSE_FIELDS
        if field not in response or response[field] in (None, "")
    ]
    if missing:
        raise AdapterError(f"the application reported no {', '.join(missing)}")
    fields = {field: response[field] for field in REQUIRED_RESPONSE_FIELDS}
    for field in OPTIONAL_RESPONSE_FIELDS:
        if response.get(field) is not None:
            fields[field] = response[field]
    return fields


def write_receipt(
    path: Path,
    response: dict[str, Any],
    *,
    endpoint: str,
    client_wall_seconds: float,
    requested_seed: int,
    seed_config_key: str | None,
    design_count: int,
) -> None:
    """Write what the runner reported and what this request asked of it.

    `used_seed` is read from the response and is null when the application sends
    none. The deployed application this adapter was written against sends none,
    so a null there is the expected value and not a missing measurement.

    There is no cost field with a number in it. Every Genie3 cost figure this
    project holds was measured on RunPod, and a RunPod figure is not a fal price.
    """
    used_seed = response.get("used_seed")
    if used_seed is not None and (not isinstance(used_seed, int) or isinstance(used_seed, bool)):
        raise AdapterError("the application reported a non-integer used_seed")
    receipt: dict[str, Any] = dict(runtime_fields(response))
    receipt.update(
        {
            "runner_protocol": RUNNER_PROTOCOL,
            "fal_endpoint": endpoint,
            "request_id": response.get("request_id"),
            "requested_seed": requested_seed,
            "used_seed": used_seed,
            "seed_config_key": seed_config_key,
            "seed_delivered": seed_config_key is not None,
            "design_count": design_count,
            "client_wall_seconds": round(client_wall_seconds, 3),
            "runner_wall_seconds": response.get("seconds"),
            "cost_basis": COST_BASIS,
        }
    )
    base.write_json(path, receipt)


# ----------------------------------------------------------------------------
# Subcommands.
# ----------------------------------------------------------------------------


def dispatch(args: argparse.Namespace) -> int:
    """Post one generation request and write the structures the runner returned.

    `run` spawns this. It and `dispatch_probe` are the only code in this module
    that opens a socket, and they are the processes the credential wrapper puts
    the credential in front of. No parent process reads the credential.
    """
    endpoint = resolve_endpoint(args.fal_url)
    problem_path = args.problem.expanduser()
    problem = base.load_json(problem_path, "Genie3 problem")
    validate_request_values(
        n_sample=args.n_sample,
        direction_scale=args.direction_scale,
        seed=args.seed,
        problem_id=args.problem_id,
        model_version=args.model_version,
    )
    payload = build_payload(
        identifier=request_id(args.request_id, args.problem_id),
        problem_id=args.problem_id,
        problem=problem,
        target_pdb=args.target_pdb.expanduser(),
        target_fasta=args.target_fasta.expanduser(),
        n_sample=args.n_sample,
        direction_scale=args.direction_scale,
        seed=args.seed,
        seed_config_key=args.seed_config_key,
        allow_unseeded=args.allow_unseeded,
        model_version=args.model_version,
    )
    started = time.monotonic()
    response = post(endpoint, payload, args.timeout_seconds, args.credential_env)
    seconds = time.monotonic() - started
    decoded = decode_designs(response, args.n_sample)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, structure in decoded:
        (out_dir / name).write_bytes(structure)
    write_receipt(
        args.receipt.expanduser(),
        response,
        endpoint=endpoint,
        client_wall_seconds=seconds,
        requested_seed=args.seed,
        seed_config_key=args.seed_config_key,
        design_count=len(decoded),
    )
    print(
        f"genie3 fal dispatch: designs={len(decoded)} seconds={seconds:.1f} "
        f"device={response['device']} out_dir={out_dir}"
    )
    return 0


def probe(args: argparse.Namespace) -> int:
    """Spawn the child that asks the deployed application to report its runtime.

    This process resolves the endpoint and builds a child command. It reads no
    credential and opens no socket, the same way `run` does not. The request
    leaves from `dispatch_probe`.

    Starting a runner is not free, so this refuses without `--acknowledge-cost`.
    `toolcheck` is the free check and it answers a different question: whether
    this adapter is configured. The two sit next to each other in the help, and
    a paid subcommand that reads as a diagnostic is a trap for automation.
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
    """Post one toolcheck request and print the runtime the application reported.

    `probe` spawns this. It and `dispatch` are the only code in this module that
    opens a socket, and they are the processes the credential wrapper puts the
    credential in front of.
    """
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
    print(f"genie3 fal probe: device {fields['device']}")
    print(
        f"genie3 fal probe: weights {fields['checkpoint_sha256']} "
        f"({fields['checkpoint_bytes']} bytes)"
    )
    print(f"genie3 fal probe: environment {fields['environment_identity']}")
    print(f"genie3 fal probe: source {fields['source_revision']}")
    print(f"genie3 fal probe: took {seconds:.1f} seconds")
    return 0


def toolcheck(args: argparse.Namespace) -> int:
    """Report this adapter's own readiness without sending anything.

    A toolcheck that called the application would start a GPU runner every time
    a profile validated itself, so this one answers only what it can answer for
    free: whether an endpoint is configured, whether a credential route exists,
    and what the request contract allows.
    """
    ready = True
    try:
        resolve_endpoint(args.fal_url)
        print(f"genie3 fal adapter: endpoint {fal_invocation.REDACTED_FAL_URL}")
    except AdapterError as exc:
        ready = False
        print(f"genie3 fal adapter: endpoint unresolved: {exc}")
    try:
        route, credential_env = resolve_credential_route(args)
        print(f"genie3 fal adapter: credential route {route} through {credential_env}")
    except AdapterError as exc:
        ready = False
        print(f"genie3 fal adapter: credential route unavailable: {exc}")
    print(
        f"genie3 fal adapter: dispatch child {args.client_python} {DISPATCH_SCRIPT} "
        f"{DISPATCH_COMMAND}"
    )
    print(
        f"genie3 fal adapter: request ceiling n_sample 1-{MAXIMUM_SAMPLES}, "
        f"direction_scale 0.0-{MAXIMUM_DIRECTION_SCALE}, seed 0-{MAXIMUM_SEED}"
    )
    print(
        "genie3 fal adapter: binder mode returns a C-alpha trace; this adapter copies the "
        "returned bytes and synthesizes no N, C or O"
    )
    print(f"genie3 fal adapter: cost basis {COST_BASIS}; no measurement prices this provider")
    print(
        "genie3 fal adapter: this check sends no request and reads no credential; probe "
        "is the paid subcommand and it needs --acknowledge-cost",
        flush=True,
    )
    if not ready:
        print("genie3 fal adapter: not ready to dispatch", file=sys.stderr)
    return 0 if ready else 1


def run(args: argparse.Namespace) -> int:
    """Compose the problem, dispatch one phase, and write the stage outputs.

    Everything except the dispatch is `genie3_generator`'s own code, so the
    manifest rows and the design poses this route publishes are built by the
    same functions the local route publishes them with.
    """
    validate_request_values(
        n_sample=args.count,
        direction_scale=args.direction_scale,
        seed=args.seed,
        problem_id=args.problem_id,
        model_version=args.model_version,
    )
    if args.seed_config_key is None and not args.allow_unseeded:
        raise AdapterError(
            "no Genie3 invocation this package records passes a seed. Pass --seed-config-key "
            "with the configuration key your build reads, or --allow-unseeded to record that "
            "this run carries no seed"
        )
    if base.IDENTIFIER_RE.fullmatch(args.generator_id) is None:
        raise AdapterError(f"--generator-id is not a plain identifier: {args.generator_id}")
    for label, chain in (
        ("--binder-chain", args.binder_chain),
        ("--generator-binder-chain", args.generator_binder_chain),
    ):
        if chain is not None and base.CHAIN_ID_RE.fullmatch(chain) is None:
            raise AdapterError(f"{label} is {chain}; a chain ID is one letter or digit")
    if args.binder_length_min < 1 or args.binder_length_max < args.binder_length_min:
        raise AdapterError(
            f"binder length bounds {args.binder_length_min}-{args.binder_length_max} are empty"
        )
    endpoint = resolve_endpoint(args.fal_url)
    # The route is resolved before the dataset is composed, so a machine with no
    # way to reach the credential fails before it writes anything.
    resolve_credential_route(args)

    manifest, manifest_source = base.load_target_manifest(args)
    target_id = str(manifest["target_id"])
    source_id = str(manifest.get("source_id") or target_id)
    chain = args.target_chain or str(manifest.get("design_target_chain_id") or "")
    if not chain:
        raise AdapterError(
            f"target manifest {manifest_source} records no design_target_chain_id; pass "
            "--target-chain with the chain the binder is designed against"
        )
    if chain == args.binder_chain:
        raise AdapterError(
            f"the target chain and --binder-chain are both {chain}; they name two different "
            "chains of the design pose, which carries the binder and the target together"
        )
    structure_path, structure_sha256 = base.target_structure(manifest, manifest_source)
    target_lines = base.chain_atom_lines(base.read_atom_records(structure_path), chain)
    if not target_lines:
        raise AdapterError(
            f"the normalized target structure carries no coordinate record for chain {chain}: "
            f"{structure_path}"
        )
    target_residues = base.chain_residues(target_lines)
    hotspot = [
        base.site_token(residue, chain)
        for residue in base.site_residue_ids(manifest, manifest_source)
    ]
    extended = [
        token
        for token in (base.site_token(residue, chain) for residue in args.extended_site_residue)
        if token not in hotspot
    ]

    attempt_dir = args.attempt_dir.expanduser().resolve()
    phase_dir = attempt_dir / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        base.resolve_output_path(attempt_dir, args.manifest_path, "manifest path")
        if args.manifest_path is not None
        else phase_dir / base.DEFAULT_MANIFEST_NAME
    )
    work_dir = phase_dir / args.work_subdir
    dataset_root = work_dir / base.DATASET_SUBDIR
    run_root = work_dir / base.RUN_SUBDIR
    run_dir = run_root / args.problem_id
    # The returned names are the engine's own, so a stale file from an earlier
    # attempt cannot be told from a returned one. `genie3_generator` refuses the
    # same condition for the same reason.
    existing = base.generated_poses(run_dir, run_root, args.problem_id) if run_dir.is_dir() else []
    if existing:
        raise AdapterError(
            f"the run directory already holds {len(existing)} structure files: {run_dir}. The "
            "application returns designs under the engine's own names, so a stale file cannot "
            "be told apart. Run the stage in a clean attempt directory"
        )
    out_dir = run_dir / base.PREFERRED_OUTPUT_DIRS[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    problem_path, target_pdb_path = base.write_dataset(
        dataset_root,
        problem_id=args.problem_id,
        chain=chain,
        lines=target_lines,
        residues=target_residues,
        hotspot=hotspot,
        extended=extended,
        binder_minimum_length=args.binder_length_min,
        binder_maximum_length=args.binder_length_max,
        target_id=target_id,
        source_id=source_id,
    )
    problem = base.load_json(problem_path, "Genie3 problem")
    target_fasta_path = Path(str(problem["target_fasta_filepath"]))
    receipt_path = work_dir / args.receipt_name

    run_external(
        child_argv(
            args,
            DISPATCH_COMMAND,
            endpoint,
            "--problem-id",
            args.problem_id,
            "--problem",
            str(problem_path.resolve()),
            "--target-pdb",
            str(target_pdb_path.resolve()),
            "--target-fasta",
            str(target_fasta_path.resolve()),
            "--out-dir",
            str(out_dir.resolve()),
            "--receipt",
            str(receipt_path.resolve()),
            "--n-sample",
            str(args.count),
            "--direction-scale",
            str(float(args.direction_scale)),
            "--seed",
            str(args.seed),
            "--model-version",
            args.model_version,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--request-id",
            request_id(args.request_id, f"{args.generator_id}-{args.phase}-{args.problem_id}"),
            *(
                ["--seed-config-key", args.seed_config_key]
                if args.seed_config_key is not None
                else ["--allow-unseeded"]
            ),
        ),
        f"generate {args.count} backbones",
    )

    receipt = base.load_json(receipt_path, "fal receipt")
    runtime = runtime_fields(receipt)
    produced = base.generated_poses(run_dir, run_root, args.problem_id)
    if len(produced) < args.count:
        raise AdapterError(
            f"phase {args.phase} needs {args.count} structures and the application returned "
            f"{len(produced)} under {run_dir}"
        )
    delivered_seed = args.seed if args.seed_config_key is not None else None
    problem_sha256 = base.sha256_file(problem_path)
    rows: list[dict[str, Any]] = []
    for index, source_pose in enumerate(produced[: args.count]):
        record = base.build_candidate(
            args, index=index, source_pose=source_pose, phase_dir=phase_dir, target_chain=chain
        )
        candidate_id = str(record["candidate_id"])
        rows.append(
            {
                "target_id": target_id,
                "target_sha256": str(manifest["target_sha256"]),
                "parent_candidate_id": None,
                "origin_generator": args.generator_id,
                **base.backbone_lineage(candidate_id, args.generator_id),
                "generator_mode": base.GENERATOR_MODE,
                "runner_protocol": RUNNER_PROTOCOL,
                "sequence_designer": None,
                "generator_seed": args.seed,
                "requested_seed": args.seed,
                "tool_seed": delivered_seed,
                "seed_delivered": delivered_seed is not None,
                "seed_config_key": args.seed_config_key,
                "sequence_path": None,
                "sequence_sha256": None,
                "sequence_length": None,
                "backbone_only": True,
                "structure_path": str(manifest["source_structure_path"]),
                "structure_sha256": str(manifest["target_sha256"]),
                "residue_map_sha256": str(manifest["residue_map_sha256"]),
                "optimization_round": 0,
                "last_optimizer": None,
                "status": base.CANDIDATE_STATUS,
                "stage_id": args.stage,
                "target_manifest_path": str(manifest_source),
                "input_structure_path": str(structure_path),
                "input_structure_sha256": structure_sha256,
                # The local route records the directory it ran Genie3 in and the
                # hash of the model configuration it found there. Neither exists
                # on this machine. `checkpoint_sha256` is the runner's own digest
                # of every weights file it served, which is the stronger claim.
                "genie3_home": None,
                "working_directory": None,
                "model_version": args.model_version,
                "model_config_sha256": None,
                # The application composes the configuration from the request
                # fields, so there is no local configuration file to hash. The
                # fields that determine it are on the row.
                "config_path": None,
                "config_sha256": None,
                "problem_path": str(problem_path.resolve()),
                "problem_sha256": problem_sha256,
                "problem_id": args.problem_id,
                "n_sample": args.count,
                "direction_scale": float(args.direction_scale),
                "binder_length_min": args.binder_length_min,
                "binder_length_max": args.binder_length_max,
                "fal_endpoint": endpoint,
                "fal_receipt_path": str(receipt_path.resolve()),
                "runtime_wall_seconds": receipt.get("runner_wall_seconds"),
                "cost_basis": COST_BASIS,
                **runtime,
                **record,
            }
        )
    base.write_jsonl(manifest_path, rows)
    print(
        f"genie3 fal adapter: phase={args.phase} target={target_id} candidates={len(rows)} "
        f"structures={len(produced)} seed_delivered={delivered_seed is not None} "
        f"device={runtime['device']} checkpoint={runtime['checkpoint_sha256']} "
        f"manifest={manifest_path} target_manifest={manifest_source}"
    )
    return 0


def parse_outputs(args: argparse.Namespace) -> int:
    """Check the phase outputs this stage declares.

    The local route already does this, and both routes declare the same outputs,
    so this is the same function rather than a second copy of it.
    """
    return base.parse_outputs(args)


# ----------------------------------------------------------------------------
# Arguments.
# ----------------------------------------------------------------------------


def add_route_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fal-url",
        default=None,
        help=(
            "Application URL of your own deployment, exactly "
            f"https://{FAL_HOSTNAME}/<account>/<application>. There is no default. "
            f"Defaults to {FAL_URL_ENVIRONMENT_KEY}."
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


def add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-version",
        default=base.DEFAULT_MODEL_VERSION,
        help=(
            "Weights version the runner has to carry. The application checks that "
            f"{base.PRETRAINED_SUBDIR}/<version>/{base.MODEL_CONFIG_NAME} exists and does not "
            f"write the version into the configuration. Defaults to {base.DEFAULT_MODEL_VERSION}."
        ),
    )
    parser.add_argument(
        "--direction-scale",
        type=float,
        default=0.0,
        help="Genie3 sampler direction scale. A nonzero value steers toward the site residues.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Requested seed. Recorded on every row whether or not the runner receives it.",
    )
    parser.add_argument(
        "--seed-config-key",
        default=None,
        help=(
            "Dotted configuration key the application writes the seed into, such as "
            "generation.seed. Pass the key your Genie3 build reads."
        ),
    )
    parser.add_argument(
        "--allow-unseeded",
        action="store_true",
        help="Send without a seed. Rows record tool_seed null and seed_delivered false.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Request timeout. Defaults to {DEFAULT_TIMEOUT_SECONDS}.",
    )
    parser.add_argument("--request-id", default=None, help="Identifier the runner echoes back.")


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", default=None, help="Stage ID, recorded by the dispatcher.")
    parser.add_argument("--phase", required=True, help="Stage phase name, such as smoke or scale.")
    parser.add_argument(
        "--count", type=int, required=True, help="Number of backbones this phase generates."
    )
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument(
        "--target-manifest",
        type=Path,
        default=None,
        help="Published target manifest under the artifact root. Overrides the receipt lookup.",
    )
    parser.add_argument("--target-stage-id", default=base.DEFAULT_TARGET_STAGE_ID)
    parser.add_argument("--target-artifact-id", default=base.DEFAULT_TARGET_ARTIFACT_ID)
    parser.add_argument(
        "--target-chain",
        default=None,
        help="Target chain the binder is designed against. Defaults to the manifest's.",
    )
    parser.add_argument("--binder-chain", default=base.DEFAULT_BINDER_CHAIN)
    parser.add_argument(
        "--generator-binder-chain",
        default=None,
        help=(
            "Chain the application wrote the binder into. Needed only when more than one chain "
            "of the output falls inside the binder length bounds."
        ),
    )
    parser.add_argument("--binder-length-min", type=int, default=base.DEFAULT_BINDER_MINIMUM_LENGTH)
    parser.add_argument("--binder-length-max", type=int, default=base.DEFAULT_BINDER_MAXIMUM_LENGTH)
    parser.add_argument("--generator-id", default=base.DEFAULT_GENERATOR_ID)
    parser.add_argument("--problem-id", default=base.DEFAULT_PROBLEM_ID)
    parser.add_argument(
        "--extended-site-residue",
        action="append",
        default=[],
        help="Extra site residue in CHAIN:NUMBER form for the extended slot. Repeat the flag.",
    )
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--pose-subdir", default=base.DEFAULT_POSE_SUBDIR)
    parser.add_argument("--work-subdir", default=base.DEFAULT_WORK_SUBDIR)
    parser.add_argument(
        "--receipt-name",
        default=DEFAULT_RECEIPT_NAME,
        help=f"Receipt file inside the work directory. Defaults to {DEFAULT_RECEIPT_NAME}.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "toolcheck", help="Report this adapter's readiness. Sends no request."
    )
    add_route_arguments(check_parser)

    run_parser = subparsers.add_parser("run", help="Generate one phase of backbones on fal.")
    add_route_arguments(run_parser)
    add_request_arguments(run_parser)
    add_run_arguments(run_parser)

    parse_parser = subparsers.add_parser("parse", help="Parse the outputs of one completed phase.")
    parse_parser.add_argument("--stage", required=True)
    parse_parser.add_argument("--phase", required=True)
    parse_parser.add_argument("--count", type=int, default=1)
    parse_parser.add_argument("--attempt-dir", type=Path, required=True)
    parse_parser.add_argument("--receipts-dir", type=Path, default=None)
    parse_parser.add_argument("--artifact-root", type=Path, default=None)
    parse_parser.add_argument("--config", type=Path, required=True)
    parse_parser.add_argument("--plan", type=Path, default=None)

    probe_parser = subparsers.add_parser(
        "probe",
        help="Ask the deployed application for its runtime. Starts a runner and costs money.",
    )
    add_route_arguments(probe_parser)
    probe_parser.add_argument(
        "--acknowledge-cost",
        action="store_true",
        help=(
            "Required. Confirms that starting a GPU runner on your own deployment is "
            "authorized spend. toolcheck is the free check and needs no flag."
        ),
    )
    probe_parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    probe_parser.add_argument("--request-id", default=None)

    dispatch_parser = subparsers.add_parser(
        DISPATCH_COMMAND, help="The child run spawns. Posts one request. Not for direct use."
    )
    dispatch_parser.add_argument("--fal-url", default=None)
    dispatch_parser.add_argument(
        "--credential-env",
        type=fal_invocation.credential_environment_key,
        default=fal_invocation.CREDENTIAL_ENVIRONMENT_KEY,
    )
    dispatch_parser.add_argument("--problem-id", required=True)
    dispatch_parser.add_argument("--problem", type=Path, required=True)
    dispatch_parser.add_argument("--target-pdb", type=Path, required=True)
    dispatch_parser.add_argument("--target-fasta", type=Path, required=True)
    dispatch_parser.add_argument("--out-dir", type=Path, required=True)
    dispatch_parser.add_argument("--receipt", type=Path, required=True)
    dispatch_parser.add_argument("--n-sample", type=int, required=True)
    add_request_arguments(dispatch_parser)

    probe_child_parser = subparsers.add_parser(
        PROBE_CHILD_COMMAND,
        help="The child probe spawns. Posts one toolcheck request. Not for direct use.",
    )
    probe_child_parser.add_argument("--fal-url", default=None)
    probe_child_parser.add_argument(
        "--credential-env",
        type=fal_invocation.credential_environment_key,
        default=fal_invocation.CREDENTIAL_ENVIRONMENT_KEY,
    )
    probe_child_parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    probe_child_parser.add_argument("--request-id", default=None)

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
        if args.command == "probe":
            return probe(args)
        if args.command == PROBE_CHILD_COMMAND:
            return dispatch_probe(args)
        if args.command == DISPATCH_COMMAND:
            return dispatch(args)
        return run(args)
    except base.AdapterError as exc:
        print(f"genie3 fal adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
