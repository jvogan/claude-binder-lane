"""Measure provider authorization without starting provider work."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .adapters.modal_platform import ENVIRONMENT_KEY, IMAGE_KEY, IMAGE_REF_RE
from .clients import fal_invocation
from .paths import child_process_environment
from .refusals import Refusal, exit_code_for_result


FAL_PROVIDER_ID = "fal"
# The host a user configures an application on. A synchronous request to this
# host runs the application.
FAL_HOST = "fal.run"
# The host that answers queue metadata. The status path has to be built here,
# not on FAL_HOST, because https://fal.run/<team>/<app>/requests/<id>/status is
# a path inside the application and routes to the application itself.
FAL_QUEUE_HOST = "queue.fal.run"
FAL_KEY_ENVIRONMENT_KEY = "FAL_KEY"
FAL_CREDENTIAL_WRAPPER = "fal-credential-wrapper"
QUEUE_STATUS_JOB_ID = "00000000-0000-0000-0000-000000000000"
WRAPPED_FAL_PROBE_COMMAND = "__wrapped-fal-authorization-probe"
# The probe sends nothing unless a caller asks for it.
#
# Until 2026-08-29 this probe built its URL on FAL_HOST and was documented as
# free. It was not. A capabilities run against that host invoked the
# applications, cold-started three GPU runners, and cost about 0.58 USD. The
# host above is corrected, and the corrected form has now been measured once.
# On 2026-09-08 one GET to the queue-status path answered HTTP 404 in 0.269 s
# and was classified authorized. A cold start takes tens of seconds, so that
# call started no runner. What it does not establish is a price. This package
# ships no fal billing reader, so no invoice was read, and a residual
# per-request charge would not show up in a latency figure. The default
# therefore stays as it was: send no request and report the application not
# measured. A caller that has authorization to spend passes allow_network=True
# for one measurement, and a caller with its own transport supplies that
# instead.
PROBE_MEASURED_STATUS = 404
PROBE_MEASURED_SECONDS = 0.269
PROBE_MEASURED_ON = "2026-09-08"
PROBE_DISABLED_REASON = (
    "The fal queue-status probe sent no request. Its earlier URL form invoked "
    "the application and started paid runners. The corrected queue-host form "
    f"answered HTTP {PROBE_MEASURED_STATUS} in {PROBE_MEASURED_SECONDS} s on "
    f"{PROBE_MEASURED_ON}, which is too fast to be a runner cold start. No "
    "invoice was read for it, so a residual per-request charge is not ruled "
    "out. Authorization remains unknown."
)
FAL_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
FAL_APPLICATIONS = (
    ("rfdiffusion3_fal_url", "RFdiffusion3"),
    ("proteinmpnn_fal_url", "ProteinMPNN"),
    ("esmfold2_fast_fal_url", "ESMFold2-Fast"),
    ("alphafold_multimer_v3_fal_url", "AlphaFold2-Multimer-v3"),
)

Transport = Callable[[urllib.request.Request, float], Any]


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Keep the authorization header on the fal queue host."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        response: Any,
        code: int,
        message: str,
        headers: Any,
        destination: str,
    ) -> None:
        raise urllib.error.HTTPError(
            request.full_url,
            code,
            "fal authorization probe rejected a redirect",
            headers,
            response,
        )


def _open(request: urllib.request.Request, timeout_seconds: float) -> Any:
    """Open one HTTPS request without following a redirect."""
    return urllib.request.build_opener(_RejectRedirects()).open(
        request,
        timeout=timeout_seconds,
    )


def _endpoint_parts(endpoint: object) -> tuple[str, str] | None:
    """Return one valid fal team and app pair."""
    if not isinstance(endpoint, str) or not endpoint.strip():
        return None
    parsed = urllib.parse.urlparse(endpoint)
    segments = [segment for segment in parsed.path.split("/") if segment]
    try:
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme != "https"
        or parsed.hostname != FAL_HOST
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or len(segments) != 2
        or not all(FAL_PATH_SEGMENT_RE.fullmatch(segment) for segment in segments)
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        return None
    return segments[0], segments[1]


def queue_status_url(endpoint: object) -> str | None:
    """Return the queue-status URL for one configured application.

    The application is configured on ``FAL_HOST`` and its queue metadata is
    served by ``FAL_QUEUE_HOST``. Building this path on the configured host
    instead is what made the probe invoke the application.
    """
    parts = _endpoint_parts(endpoint)
    if parts is None:
        return None
    team, application = parts
    return (
        f"https://{FAL_QUEUE_HOST}/{team}/{application}/requests/"
        f"{QUEUE_STATUS_JOB_ID}/status"
    )


def _close(response: Any) -> None:
    """Close a response when the transport supplies a close method."""
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _proxy_refusal(error: urllib.error.HTTPError) -> bool:
    """Return whether an HTTP response identifies a proxy refusal."""
    headers = error.headers
    if headers is None:
        return False
    return bool(
        headers.get("X-Mitmproxy-Blocked-Reason")
        or headers.get("Proxy-Status")
    )


def _application_result(
    *,
    endpoint_field: str,
    application: str,
    endpoint: object,
    status: str,
    reason: str,
    credential_source: str | None = None,
    http_status: int | None = None,
) -> dict[str, Any]:
    """Build one application result without including a credential."""
    result = {
        "endpoint_field": endpoint_field,
        "application": application,
        "status": status,
        "reason": reason,
    }
    if isinstance(endpoint, str) and endpoint:
        result["endpoint"] = endpoint
    if credential_source:
        result["credential_source"] = credential_source
    if http_status is not None:
        result["http_status"] = http_status
    return result


def _authorization_refusal(application_result: Mapping[str, Any]) -> Refusal:
    """Return the refusal that explains one measured authorization denial."""
    application = application_result["application"]
    endpoint = application_result.get("endpoint")
    credential_source = application_result.get("credential_source") or FAL_KEY_ENVIRONMENT_KEY
    http_status = application_result.get("http_status")
    status_url = queue_status_url(endpoint) or "the configured queue-status path"
    if http_status == 401:
        action = (
            f"Select the credential source authorized for {application}; this probe used "
            f"{credential_source}."
        )
        escalation = (
            f"Verify which credential source Claude Science should expose for {application}; "
            "do not send the credential value."
        )
    else:
        action = f"Ask the provider team administrator to grant access to {application}."
        escalation = "Send the configured application endpoint to the provider team administrator."
    return Refusal(
        cause=f"{application} authorization was refused.",
        expected="The queue-status authorization probe must return HTTP 200.",
        expected_source=status_url,
        found=application_result["reason"],
        found_source=status_url,
        scope=(
            "The probe sent one GET request to the queue-status URL above and "
            "submitted no job of its own. One measured call of this form answered "
            f"in {PROBE_MEASURED_SECONDS} s, which is too fast to be a runner cold "
            "start. No invoice was read for it, so its price is not established."
        ),
        action=action,
        escalation=escalation,
    )


def _probe_application_through_wrapper(
    endpoint_field: str,
    application: str,
    endpoint: str,
    *,
    executable: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run the queue-status probe inside the same credential wrapper execution uses.

    The wrapper injects the credential into the child process. The child writes a
    sanitized result to a temporary file, so neither the credential nor arbitrary
    wrapper output enters this report.
    """
    wrapper = fal_invocation.wrapper_path(executable)
    if wrapper is None:
        return _application_result(
            endpoint_field=endpoint_field,
            application=application,
            endpoint=endpoint,
            status=NOT_MEASURED,
            reason=f"The selected fal credential wrapper {executable!r} is unavailable.",
            credential_source=f"credential wrapper {executable}",
        )
    with tempfile.TemporaryDirectory(prefix="claude-binder-fal-auth-") as temporary:
        result_path = os.path.join(temporary, "result.json")
        argv = [
            wrapper,
            fal_invocation.WRAPPER_SUBCOMMAND,
            "--",
            sys.executable,
            "-m",
            "claude_binder.provider_authorization",
            WRAPPED_FAL_PROBE_COMMAND,
            endpoint_field,
            application,
            endpoint,
            result_path,
            str(timeout_seconds),
        ]
        try:
            completed = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(1.0, timeout_seconds + 5.0),
                # The wrapper runs this module again in a child, so the child needs a
                # route to the package. A host that binds the package by file path
                # gives it none.
                env=child_process_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _application_result(
                endpoint_field=endpoint_field,
                application=application,
                endpoint=endpoint,
                status=NOT_MEASURED,
                reason=(
                    "The selected fal credential wrapper did not complete the queue-status "
                    f"probe: {type(exc).__name__}. Authorization remains unknown."
                ),
                credential_source=f"credential wrapper {executable}",
            )
        if completed.returncode != 0 or not os.path.isfile(result_path):
            return _application_result(
                endpoint_field=endpoint_field,
                application=application,
                endpoint=endpoint,
                status=NOT_MEASURED,
                reason=(
                    f"The selected fal credential wrapper exited {completed.returncode} without "
                    "a probe result. Authorization remains unknown."
                ),
                credential_source=f"credential wrapper {executable}",
            )
        try:
            with open(result_path, encoding="utf-8") as handle:
                result = json.load(handle)
        except (OSError, ValueError, TypeError) as exc:
            return _application_result(
                endpoint_field=endpoint_field,
                application=application,
                endpoint=endpoint,
                status=NOT_MEASURED,
                reason=(
                    "The selected fal credential wrapper returned an invalid probe result: "
                    f"{type(exc).__name__}. Authorization remains unknown."
                ),
                credential_source=f"credential wrapper {executable}",
            )
    if not isinstance(result, Mapping) or result.get("status") not in {
        AUTHORIZED,
        REFUSED,
        NOT_MEASURED,
    }:
        return _application_result(
            endpoint_field=endpoint_field,
            application=application,
            endpoint=endpoint,
            status=NOT_MEASURED,
            reason="The selected fal credential wrapper returned no authorization state.",
            credential_source=f"credential wrapper {executable}",
        )
    http_status = result.get("http_status")
    return _application_result(
        endpoint_field=endpoint_field,
        application=application,
        endpoint=endpoint,
        status=str(result["status"]),
        reason=str(result.get("reason") or "The wrapper probe returned no reason."),
        credential_source=f"credential wrapper {executable}",
        http_status=http_status if isinstance(http_status, int) else None,
    )


def probe_application(
    endpoint_field: str,
    application: str,
    endpoint: object,
    *,
    environ: Mapping[str, str] | None = None,
    transport: Transport | None = None,
    timeout_seconds: float = 10.0,
    allow_network: bool = False,
    fal_executable: str = FAL_CREDENTIAL_WRAPPER,
    fal_credential_route: str | None = None,
) -> dict[str, Any]:
    """Read one application status without queuing work or reading a response body.

    Nothing is sent unless the caller supplies a ``transport`` or sets
    ``allow_network``. See ``PROBE_DISABLED_REASON`` for why the default sends
    nothing.
    """
    if transport is None and not allow_network:
        return _application_result(
            endpoint_field=endpoint_field,
            application=application,
            endpoint=endpoint,
            status="not measured",
            reason=PROBE_DISABLED_REASON,
        )
    status_url = queue_status_url(endpoint)
    if status_url is None:
        return _application_result(
            endpoint_field=endpoint_field,
            application=application,
            endpoint=endpoint,
            status="not measured",
            reason="No valid fal application endpoint is configured. Authorization remains unknown.",
        )

    source = os.environ if environ is None else environ
    try:
        credential_key = fal_invocation.credential_environment_key(environ=source)
        route = fal_invocation.resolve_route(
            fal_executable,
            requested=fal_credential_route,
            environ=source,
            credential_env_key=credential_key,
        )
    except fal_invocation.RouteError as exc:
        return _application_result(
            endpoint_field=endpoint_field,
            application=application,
            endpoint=endpoint,
            status="not measured",
            reason=f"{exc}. Authorization remains unknown.",
        )
    if route == fal_invocation.ROUTE_WRAPPER:
        return _probe_application_through_wrapper(
            endpoint_field,
            application,
            str(endpoint),
            executable=fal_executable,
            timeout_seconds=timeout_seconds,
        )

    credential = str(source.get(credential_key, "") or "").strip()
    credential_source = f"environment variable {credential_key}"

    request = urllib.request.Request(
        status_url,
        method="GET",
        headers={"Authorization": f"Key {credential}"},
    )
    opener = transport or _open
    try:
        response = opener(request, timeout_seconds)
    except urllib.error.HTTPError as error:
        if _proxy_refusal(error):
            return _application_result(
                endpoint_field=endpoint_field,
                application=application,
                endpoint=endpoint,
                status="not measured",
                reason=(
                    "The outbound proxy refused the request before fal could answer. "
                    "Authorization remains unknown."
                ),
                credential_source=credential_source,
            )
        status_code = error.code
    except (urllib.error.URLError, OSError) as error:
        return _application_result(
            endpoint_field=endpoint_field,
            application=application,
            endpoint=endpoint,
            status="not measured",
            reason=(
                "Network transport did not establish a TLS tunnel to fal: "
                f"{error}. Authorization remains unknown."
            ),
            credential_source=credential_source,
        )
    else:
        try:
            status_code = response.getcode()
        finally:
            _close(response)

    if status_code in {200, 404}:
        status = "authorized"
        reason = (
            f"fal answered HTTP {status_code} to the queue-status probe using "
            f"{credential_source}."
        )
    elif status_code in {401, 403}:
        status = "refused"
        reason = (
            f"fal answered HTTP {status_code} to the queue-status probe using "
            f"{credential_source}."
        )
    else:
        status = "not measured"
        reason = (
            f"fal answered HTTP {status_code}, so credential authorization remains unknown."
        )
    return _application_result(
        endpoint_field=endpoint_field,
        application=application,
        endpoint=endpoint,
        status=status,
        reason=reason,
        credential_source=credential_source,
        http_status=status_code,
    )


# The user-owned Modal route.
#
# Modal authorization is measured through one free read of the session's own
# compute connection. The read never creates a handle, submits a job, or starts
# a container. A campaign that selects no paid Modal stage makes no call at all.
MODAL_PROVIDER_ID = "modal"
RUNPOD_PROVIDER_ID = "runpod"
LAMBDA_PROVIDER_ID = "lambda"
MODAL_DETAILS_REQUEST: dict[str, str] = {"provider": "modal", "mode": "read"}
RUNPOD_DETAILS_REQUEST: dict[str, str] = {"provider": "runpod", "mode": "read"}
LAMBDA_DETAILS_REQUEST: dict[str, str] = {"provider": "lambda", "mode": "read"}
# The same environment identity form is enforced in adapters/runtime_validator.py
# and generator_preflight.py. The three copies must agree, so a change here is a
# change in all three.
MODAL_ENVIRONMENT_IDENTITY_RE = re.compile(r"^modal-env:[^@\s]+@spec_sha=[0-9a-f]{16,64}$")
MODAL_LEDGER_BLOCK_RE = re.compile(r"^#{1,6}\s*env:(?P<name>[^@\s]+)@(?P<spec_sha>\S+)\s*$")
MODAL_LEDGER_FIELD_RE = re.compile(r"^\s*[-*]?\s*(?P<key>[A-Za-z_][A-Za-z0-9_ ]*):\s*(?P<value>.+?)\s*$")
AUTHORIZED = "authorized"
REFUSED = "refused"
NOT_MEASURED = "not measured"
NOT_REQUIRED = "not required"
AUTHORIZATION_STATES = (AUTHORIZED, REFUSED, NOT_MEASURED, NOT_REQUIRED)
# Both of these block a paid campaign. The gate answers whether paid work can
# proceed, so an unknown answer is not a pass.
BLOCKING_AUTHORIZATION_STATES = frozenset({REFUSED, NOT_MEASURED})
NO_JOB_STARTED = "No provider job was started."
UNKNOWN_AUTHORIZATION = "Authorization remains unknown."
# Vocabulary that makes a read failure conclusive rather than unknown. The
# executor reports the same denial through modal_platform error_kind
# "unauthorized".
DENIAL_RE = re.compile(
    r"(?i)unauthori[sz]ed|not authori[sz]ed|authorization failed|authentication failed"
    r"|permission denied|access denied|forbidden"
)
# Reasons are read by a person, so they carry no credential and no raw session
# material. A long opaque run is dropped, and a reason that names a credential
# field is withheld whole.
CREDENTIAL_WORD_RE = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|password|secret|token|credential|bearer|cookie|signature|session)"
)
OPAQUE_RUN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9_\-]{20,}(?![A-Za-z0-9])")
WITHHELD_REASON = "the reason named a credential field and was withheld"
MAXIMUM_REASON_CHARACTERS = 200


def sanitized_reason(value: object) -> str:
    """Render one failure reason with credentials and session material removed."""
    text = " ".join(str(value).split())
    if not text:
        return "no reason was reported"
    if CREDENTIAL_WORD_RE.search(text):
        return WITHHELD_REASON
    text = OPAQUE_RUN_RE.sub("<redacted>", text)
    if len(text) > MAXIMUM_REASON_CHARACTERS:
        text = text[:MAXIMUM_REASON_CHARACTERS].rstrip() + " ..."
    return text


def authorization_result(
    provider: str,
    status: str,
    *,
    reason: str,
    messages: Sequence[str] = (),
    bindings: Sequence[Mapping[str, Any]] = (),
    checked_at: str | None = None,
) -> dict[str, Any]:
    """Build one four-state provider authorization report.

    ``not required`` and ``authorized`` pass. ``refused`` and ``not measured``
    both block, because a gate that cannot answer has not granted anything.
    """
    if status not in AUTHORIZATION_STATES:
        raise ValueError(f"unknown authorization status {status!r}")
    blocking = status in BLOCKING_AUTHORIZATION_STATES
    lines = [str(message) for message in messages]
    return {
        "provider": provider,
        "status": status,
        "ok": not blocking,
        "enforced": status != NOT_REQUIRED,
        "reason": reason,
        "bindings": [dict(binding) for binding in bindings],
        "messages": lines,
        "errors": list(lines) if blocking else [],
        "checked_at": checked_at,
        "exit_code": exit_code_for_result(verified=not blocking, refused=status == REFUSED),
        "text": "\n".join(lines),
    }


def parse_modal_environment_identity(identity: object) -> tuple[str, str] | None:
    """Return the environment name and spec hash one identity string declares."""
    if not isinstance(identity, str) or MODAL_ENVIRONMENT_IDENTITY_RE.fullmatch(identity) is None:
        return None
    name, _, spec_sha = identity.partition("@spec_sha=")
    return name.partition("modal-env:")[2], spec_sha


def _text_value(value: object) -> str | None:
    """Return one non-empty string value, or None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _ledger_entry(value: object) -> dict[str, str | None] | None:
    """Read the spec hash and image ID out of one ledger entry."""
    if not isinstance(value, Mapping):
        return None
    return {
        "spec_sha": _text_value(value.get("spec_sha")),
        "image": _text_value(value.get("image")),
    }


def _ledger_from_mapping(value: object) -> dict[str, dict[str, str | None]] | None:
    """Read a ledger carried as a mapping or a list of environment records."""
    entries: dict[str, dict[str, str | None]] = {}
    if isinstance(value, Mapping):
        for name, item in value.items():
            entry = _ledger_entry(item)
            key = _text_value(name)
            if entry is None or key is None:
                return None
            entries[key] = entry
        return entries
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, Mapping):
                return None
            name = None
            for key in ("environment", "env", "name"):
                name = _text_value(item.get(key))
                if name is not None:
                    break
            entry = _ledger_entry(item)
            if name is None or entry is None:
                return None
            entries[name] = entry
        return entries
    return None


def _ledger_from_text(text: str) -> dict[str, dict[str, str | None]]:
    """Read the ``### env:<name>@<spec_sha>`` blocks a workspace ledger prints."""
    entries: dict[str, dict[str, str | None]] = {}
    current: str | None = None
    for line in text.splitlines():
        header = MODAL_LEDGER_BLOCK_RE.match(line)
        if header is not None:
            current = header.group("name")
            entries[current] = {"spec_sha": header.group("spec_sha"), "image": None}
            continue
        if line.startswith("#"):
            current = None
            continue
        field = MODAL_LEDGER_FIELD_RE.match(line)
        if field is None or current is None:
            continue
        if field.group("key").strip().casefold() in ("image", "image_ref"):
            entries[current]["image"] = _text_value(field.group("value"))
    return entries


def _workspace_from_text(text: str) -> str | None:
    """Return the workspace one ledger text names, when it names one at all."""
    for line in text.splitlines():
        field = MODAL_LEDGER_FIELD_RE.match(line)
        if field is not None and field.group("key").strip().casefold() == "workspace":
            return _text_value(field.group("value"))
    return None


def parse_modal_details(response: object) -> dict[str, Any] | None:
    """Read the workspace name and environment ledger out of one details read.

    A live ``compute_details({"provider": "modal", "mode": "read"})`` reply was
    captured on 2026-08-29 from a running Claude Science session. It is a record
    carrying one ``details`` string: an orientation line, then the per-workspace
    ledger as ``### env:<name>@<spec_sha>`` blocks with ``image_ref:`` fields.
    Its ``spec_sha`` is the 16 hex content hash of the env source file, not a 64
    hex digest. It names no workspace, because ``compute_details`` is the
    per-workspace note store and the workspace name comes from the provider
    ``token_info`` read instead. The record forms the platform references
    describe are still accepted. A response this reader cannot parse is reported
    as not measured, never as a pass, and so is a ledger that names no workspace.
    """
    if isinstance(response, str):
        entries = _ledger_from_text(response)
        workspace = _workspace_from_text(response)
        if workspace is None and not entries:
            return None
        return {"workspace": workspace, "environments": entries}
    if not isinstance(response, Mapping):
        return None
    workspace = _text_value(response.get("workspace")) or _text_value(response.get("workspace_name"))
    for key in ("environments", "envs", "environment_ledger"):
        if key in response:
            if workspace is None:
                return None
            entries = _ledger_from_mapping(response.get(key))
            return None if entries is None else {"workspace": workspace, "environments": entries}
    for key in ("ledger", "details"):
        text = response.get(key)
        if not isinstance(text, str):
            continue
        entries = _ledger_from_text(text)
        named = workspace if workspace is not None else _workspace_from_text(text)
        if named is None and not entries:
            return None
        return {"workspace": named, "environments": entries}
    return None


def _denial_text(response: Mapping[str, Any]) -> str | None:
    """Return the text of a conclusive authorization denial in one response."""
    for key in ("error_kind", "error", "errors", "status", "reason", "message", "detail"):
        value = response.get(key)
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, str) and DENIAL_RE.search(item):
                return item
    return None


def resolve_selected_providers(
    plan: Mapping[str, Any],
    selected_stages: Sequence[Mapping[str, Any]],
    *,
    runtime_environment: Mapping[str, str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Group the selected paid stages by the provider each one resolves to."""
    # Deferred because lane imports this module while lane itself is loading.
    from . import lane

    adapters = {
        str(adapter.get("adapter_id")): adapter
        for adapter in plan.get("adapters", []) or []
        if isinstance(adapter, Mapping) and isinstance(adapter.get("adapter_id"), str)
    }
    grouped: dict[str, list[dict[str, str]]] = {}
    for stage in selected_stages or []:
        if not isinstance(stage, Mapping):
            continue
        adapter = adapters.get(str(stage.get("adapter_id", "")))
        if adapter is None:
            continue
        stage_record = dict(stage)
        adapter_record = dict(adapter)
        paid = lane.stage_is_paid(stage_record, adapter_record) or lane.stage_provider_must_be_recorded(
            stage_record, adapter_record
        )
        if not paid:
            continue
        resolution = lane.resolve_stage_provider(
            plan,
            stage,
            adapter,
            runtime_environment=runtime_environment,
        )
        for provider_id in resolution.get("provider_ids", []) or []:
            grouped.setdefault(str(provider_id), []).append(
                {
                    "stage_id": str(stage.get("stage_id", "<unnamed stage>")),
                    "adapter_id": str(adapter.get("adapter_id", "<unnamed adapter>")),
                    "source": str(resolution.get("source", "")),
                }
            )
    return grouped


def _binding_subject(environment: str | None, adapter_id: str) -> str:
    """Name the subject of one binding message."""
    if environment is None:
        return f"Modal adapter {adapter_id}"
    return f"Modal environment {environment}"


def modal_bindings(
    plan: Mapping[str, Any],
    stage_records: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect the distinct Modal environment bindings the selected stages declare.

    A binding whose own declarations disagree is refused here, before any read
    reaches a provider. Adapters that share one environment, image, and spec
    hash collapse into a single binding, so the ledger is compared once.
    """
    adapters = {
        str(adapter.get("adapter_id")): adapter
        for adapter in plan.get("adapters", []) or []
        if isinstance(adapter, Mapping) and isinstance(adapter.get("adapter_id"), str)
    }
    bindings: dict[tuple[str, str, str], dict[str, Any]] = {}
    refusals: list[str] = []
    for record in stage_records:
        adapter_id = str(record.get("adapter_id", "<unnamed adapter>"))
        stage_id = str(record.get("stage_id", "<unnamed stage>"))
        adapter = adapters.get(adapter_id)
        if adapter is None:
            refusals.append(
                f"Modal adapter {adapter_id}: refused. Stage {stage_id} resolves to Modal, "
                f"and the plan has no adapter {adapter_id}. Rematerialize the plan. {NO_JOB_STARTED}"
            )
            continue
        environment_block = adapter.get("environment")
        environment_block = environment_block if isinstance(environment_block, Mapping) else {}
        resources = adapter.get("resources")
        resources = resources if isinstance(resources, Mapping) else {}
        submitted_environment = _text_value(environment_block.get(ENVIRONMENT_KEY))
        submitted_image = _text_value(environment_block.get(IMAGE_KEY))
        recorded_image = _text_value(resources.get("container_image_digest"))
        identity = adapter.get("environment_identity")
        parsed = parse_modal_environment_identity(identity)
        subject = _binding_subject(submitted_environment, adapter_id)

        # A stage can resolve to Modal through the campaign provider block while
        # running an installed session tool. Such a stage declares no Modal
        # environment, so it has no binding to compare. The workspace read still
        # measures whether this session can use the account.
        declared = submitted_environment is not None or (
            isinstance(identity, str) and identity.startswith("modal-env:")
        )
        if not declared:
            continue
        if parsed is None:
            refusals.append(
                f"{subject}: refused. The plan records environment_identity {identity} for adapter "
                f"{adapter_id}, which is not modal-env:<name>@spec_sha=<16 to 64 hex>. Rebuild the environment "
                f"or rematerialize the plan. {NO_JOB_STARTED}"
            )
            continue
        identity_environment, spec_sha = parsed
        if submitted_environment is not None and submitted_environment != identity_environment:
            refusals.append(
                f"{subject}: refused. Adapter {adapter_id} submits to environment {submitted_environment}, "
                f"while its environment_identity names environment {identity_environment}. Rebuild the "
                f"environment or rematerialize the plan. {NO_JOB_STARTED}"
            )
            continue
        environment = submitted_environment or identity_environment
        subject = _binding_subject(environment, adapter_id)
        # The adapter submits one image and the contract records another. One
        # fact with two sources of truth is refused when the two disagree.
        if (
            submitted_image is not None
            and recorded_image is not None
            and submitted_image != recorded_image
        ):
            refusals.append(
                f"{subject}: refused. Adapter {adapter_id} submits image {submitted_image} in {IMAGE_KEY} "
                f"and records image {recorded_image} in resources.container_image_digest. Rebuild the "
                f"environment or rematerialize the plan. {NO_JOB_STARTED}"
            )
            continue
        image = submitted_image or recorded_image
        if image is None or IMAGE_REF_RE.fullmatch(image) is None:
            refusals.append(
                f"{subject}: refused. Adapter {adapter_id} declares image {image} for environment "
                f"{environment}, which is not a Modal image ID. Rebuild the environment or "
                f"rematerialize the plan. {NO_JOB_STARTED}"
            )
            continue

        key = (environment, spec_sha, image)
        binding = bindings.setdefault(
            key,
            {
                "environment": environment,
                "spec_sha": spec_sha,
                "image": image,
                "adapter_ids": [],
                "stage_ids": [],
                "status": NOT_MEASURED,
            },
        )
        if adapter_id not in binding["adapter_ids"]:
            binding["adapter_ids"].append(adapter_id)
        if stage_id not in binding["stage_ids"]:
            binding["stage_ids"].append(stage_id)
    return list(bindings.values()), refusals


def parse_modal_identity(response: Any) -> str | None:
    """Return the workspace name one identity read reports, or None.

    The platform's ``token_info`` answers a single authenticated ``TokenInfoGet``
    call with ``token_id``, ``workspace_id`` and ``workspace_name``, so
    ``workspace_name`` is the field that names the workspace. A reader that hands
    back the JSON text of that record is read the same way. This is the only read
    that reports the workspace: a ``compute_details`` read carries the
    per-workspace environment ledger and never the name.
    """
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except Exception:
            return None
    if not isinstance(response, Mapping):
        return None
    return _text_value(response.get("workspace_name")) or _text_value(response.get("workspace"))


def _read_modal_identity(
    identity_reader: Callable[[], Any],
) -> tuple[Any, str | None, bool]:
    """Run the one free identity read and classify any failure it raises."""
    try:
        response = identity_reader()
    except Exception as exc:  # the reader is a session tool; any failure is a read failure
        denied = isinstance(exc, PermissionError) or bool(DENIAL_RE.search(str(exc)))
        return None, f"{type(exc).__name__}: {sanitized_reason(exc)}", denied
    return response, None, False


def _read_modal_details(
    details_reader: Callable[[Mapping[str, str]], Any],
) -> tuple[Any, str | None, bool]:
    """Run the one free details read and classify any failure it raises."""
    try:
        response = details_reader(dict(MODAL_DETAILS_REQUEST))
    except Exception as exc:  # the reader is a session tool; any failure is a read failure
        denied = isinstance(exc, PermissionError) or bool(DENIAL_RE.search(str(exc)))
        return None, f"{type(exc).__name__}: {sanitized_reason(exc)}", denied
    return response, None, False


def probe_modal_authorization(
    plan: Mapping[str, Any],
    selected_stages: Sequence[Mapping[str, Any]],
    *,
    details_reader: Callable[[Mapping[str, str]], Any] | None = None,
    identity_reader: Callable[[], Any] | None = None,
    stage_providers: Mapping[str, list[dict[str, str]]] | None = None,
    runtime_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Measure Modal authorization for the selected stages without starting a job.

    The result is one of four states. ``not required`` makes no provider call.
    ``authorized`` needs the free read to succeed and every declared binding to
    match the workspace ledger. ``refused`` and ``not measured`` both block.

    ``details_reader`` is the ``compute_details`` tool and carries the
    environment ledger. ``identity_reader`` is a zero-argument callable over the
    provider ``token_info`` read and is the only source of the workspace name,
    which the ledger never carries. With no identity reader bound, a real ledger
    measures as not measured rather than authorized, so the workspace check
    fails closed.
    """
    grouped = (
        resolve_selected_providers(plan, selected_stages, runtime_environment=runtime_environment)
        if stage_providers is None
        else stage_providers
    )
    stage_records = list(grouped.get(MODAL_PROVIDER_ID) or [])
    if not stage_records:
        return authorization_result(
            MODAL_PROVIDER_ID,
            NOT_REQUIRED,
            reason="no selected paid stage runs on Modal",
            messages=[
                "Modal authorization: not required. No selected paid stage runs on Modal. "
                f"{NO_JOB_STARTED}"
            ],
        )

    bindings, binding_refusals = modal_bindings(plan, stage_records)
    if binding_refusals:
        return authorization_result(
            MODAL_PROVIDER_ID,
            REFUSED,
            reason="the plan binding is internally inconsistent",
            messages=binding_refusals,
            bindings=bindings,
        )

    provider = plan.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    declared_workspace = _text_value(provider.get("workspace"))
    if declared_workspace is None or declared_workspace.startswith("__"):
        return authorization_result(
            MODAL_PROVIDER_ID,
            REFUSED,
            reason="the plan declares no Modal workspace",
            messages=[
                "Modal authorization: refused. The plan declares no Modal workspace in "
                f"provider.workspace. Rematerialize the plan against the workspace the campaign "
                f"will run in. {NO_JOB_STARTED}"
            ],
            bindings=bindings,
        )

    def not_measured(reason: str) -> dict[str, Any]:
        return authorization_result(
            MODAL_PROVIDER_ID,
            NOT_MEASURED,
            reason=reason,
            messages=[
                f"Modal authorization: not measured. The preflight read could not reach workspace "
                f"{declared_workspace}: {reason}. {UNKNOWN_AUTHORIZATION} {NO_JOB_STARTED}"
            ],
            bindings=bindings,
        )

    if not callable(details_reader):
        return not_measured("no compute_details reader is bound to this session")

    response, failure, denied = _read_modal_details(details_reader)
    if failure is not None:
        if denied:
            return authorization_result(
                MODAL_PROVIDER_ID,
                REFUSED,
                reason="the workspace denied the preflight read",
                messages=[
                    f"Modal authorization: refused. Workspace {declared_workspace} denied the "
                    f"preflight read: {failure}. {NO_JOB_STARTED}"
                ],
                bindings=bindings,
            )
        return not_measured(failure)

    if isinstance(response, Mapping):
        denial = _denial_text(response)
        if denial is not None:
            return authorization_result(
                MODAL_PROVIDER_ID,
                REFUSED,
                reason="the workspace denied the preflight read",
                messages=[
                    f"Modal authorization: refused. Workspace {declared_workspace} denied the "
                    f"preflight read: {sanitized_reason(denial)}. {NO_JOB_STARTED}"
                ],
                bindings=bindings,
            )
        if response.get("ok") is False:
            return not_measured("the response reported a failed read without a stated reason")

    details = parse_modal_details(response)
    if details is None:
        return not_measured("the response carried no readable workspace ledger")

    observed_workspace = details["workspace"]
    if observed_workspace is None and callable(identity_reader):
        identity, identity_failure, identity_denied = _read_modal_identity(identity_reader)
        if identity_denied:
            return authorization_result(
                MODAL_PROVIDER_ID,
                REFUSED,
                reason="the workspace denied the identity read",
                messages=[
                    f"Modal authorization: refused. Workspace {declared_workspace} denied the "
                    f"identity read: {identity_failure}. {NO_JOB_STARTED}"
                ],
                bindings=bindings,
            )
        if identity_failure is not None:
            return not_measured(f"the identity read failed: {identity_failure}")
        observed_workspace = parse_modal_identity(identity)
        if observed_workspace is None:
            return not_measured("the identity read reported no workspace name")
    if observed_workspace is None:
        return not_measured(
            "the read returned an environment ledger that names no workspace. A "
            "compute_details read carries the per-workspace environment ledger and never the "
            "workspace name, which the provider token_info read reports instead, and no "
            "identity reader is bound to this session"
        )
    if observed_workspace != declared_workspace:
        return authorization_result(
            MODAL_PROVIDER_ID,
            REFUSED,
            reason="the read returned a different workspace",
            messages=[
                f"Modal authorization: refused. The plan expects workspace {declared_workspace}, while "
                f"the preflight read returned workspace {observed_workspace}. Rematerialize the plan "
                f"against the workspace you will run in. {NO_JOB_STARTED}"
            ],
            bindings=bindings,
        )

    ledger = details["environments"]
    messages: list[str] = []
    statuses: set[str] = set()
    for binding in bindings:
        environment = binding["environment"]
        entry = ledger.get(environment)
        if entry is None:
            binding["status"] = REFUSED
            messages.append(
                f"Modal environment {environment}: refused. Workspace {observed_workspace} is reachable, "
                f"but its environment ledger has no entry for {environment}. {NO_JOB_STARTED}"
            )
        elif entry.get("spec_sha") != binding["spec_sha"]:
            binding["status"] = REFUSED
            messages.append(
                f"Modal environment {environment}: refused. The plan expects "
                f"spec_sha={binding['spec_sha']}, while workspace {observed_workspace} reports "
                f"spec_sha={entry.get('spec_sha')}. Rebuild the environment or rematerialize the plan. "
                f"{NO_JOB_STARTED}"
            )
        elif entry.get("image") is None or IMAGE_REF_RE.fullmatch(str(entry.get("image"))) is None:
            binding["status"] = NOT_MEASURED
            messages.append(
                f"Modal environment {environment}: not measured. Workspace {observed_workspace} reports "
                f"image {entry.get('image')} for {environment}, which is not a readable Modal image ID. "
                f"{UNKNOWN_AUTHORIZATION} {NO_JOB_STARTED}"
            )
        elif entry["image"] != binding["image"]:
            binding["status"] = REFUSED
            messages.append(
                f"Modal environment {environment}: refused. The plan expects image {binding['image']}, "
                f"while workspace {observed_workspace} reports image {entry['image']}. Rebuild the "
                f"environment or rematerialize the plan. {NO_JOB_STARTED}"
            )
        else:
            binding["status"] = AUTHORIZED
        statuses.add(binding["status"])

    checked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if REFUSED in statuses:
        return authorization_result(
            MODAL_PROVIDER_ID,
            REFUSED,
            reason="a declared environment binding does not match the workspace ledger",
            messages=messages,
            bindings=bindings,
            checked_at=checked_at,
        )
    if NOT_MEASURED in statuses:
        return authorization_result(
            MODAL_PROVIDER_ID,
            NOT_MEASURED,
            reason="the workspace ledger did not answer for every declared environment",
            messages=messages,
            bindings=bindings,
            checked_at=checked_at,
        )
    return authorization_result(
        MODAL_PROVIDER_ID,
        AUTHORIZED,
        reason="the free read succeeded and every declared binding matched",
        messages=[
            "Modal read authorization and declared environment bindings verified at "
            f"{checked_at}."
        ],
        bindings=bindings,
        checked_at=checked_at,
    )


def _read_native_cloud_account(
    account_reader: Callable[[Mapping[str, str]], Any],
    *,
    provider_id: str,
) -> tuple[Any, str | None, bool]:
    """Run one read-only native-cloud account check and classify failures."""
    try:
        response = account_reader({"provider": provider_id, "mode": "read"})
    except Exception as exc:
        denied = isinstance(exc, PermissionError) or bool(DENIAL_RE.search(str(exc)))
        return None, f"{type(exc).__name__}: {sanitized_reason(exc)}", denied
    return response, None, False


def _native_cloud_read_succeeded(response: Any) -> bool:
    """Accept successful host/client reads without requiring a deployment ID."""
    if not isinstance(response, Mapping):
        return False
    if _denial_text(response) is not None or response.get("ok") is False:
        return False
    if response.get("ok") is True or response.get("authorized") is True:
        return True
    status = _text_value(response.get("status"))
    if status is not None and status.casefold() in {"authorized", "connected", "ready"}:
        return True
    account = response.get("account")
    if isinstance(account, Mapping) and account:
        return True
    return any(_text_value(response.get(key)) is not None for key in ("account_id", "user_id", "id"))


def probe_runpod_authorization(
    plan: Mapping[str, Any],
    selected_stages: Sequence[Mapping[str, Any]],
    *,
    account_reader: Callable[[Mapping[str, str]], Any] | None = None,
    stage_providers: Mapping[str, list[dict[str, str]]] | None = None,
    runtime_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Measure RunPod account access without creating a pod or endpoint.

    The reader may be Claude Science's read-only provider capability or a
    wrapper around an environment-backed client.  A successful account read is
    sufficient: this route intentionally does not require or invent a
    deployment, endpoint, template, or pod identifier during preflight.
    """
    return probe_native_cloud_authorization(
        plan,
        selected_stages,
        account_reader=account_reader,
        stage_providers=stage_providers,
        runtime_environment=runtime_environment,
        provider_id=RUNPOD_PROVIDER_ID,
        provider_name="RunPod",
    )


def probe_lambda_authorization(
    plan: Mapping[str, Any],
    selected_stages: Sequence[Mapping[str, Any]],
    *,
    account_reader: Callable[[Mapping[str, str]], Any] | None = None,
    stage_providers: Mapping[str, list[dict[str, str]]] | None = None,
    runtime_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Measure Lambda Cloud account access without allocating an instance."""
    return probe_native_cloud_authorization(
        plan,
        selected_stages,
        account_reader=account_reader,
        stage_providers=stage_providers,
        runtime_environment=runtime_environment,
        provider_id=LAMBDA_PROVIDER_ID,
        provider_name="Lambda Cloud",
    )


def probe_native_cloud_authorization(
    plan: Mapping[str, Any],
    selected_stages: Sequence[Mapping[str, Any]],
    *,
    account_reader: Callable[[Mapping[str, str]], Any] | None,
    provider_id: str,
    provider_name: str,
    stage_providers: Mapping[str, list[dict[str, str]]] | None = None,
    runtime_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Shared read-only authorization for native RunPod and Lambda routes."""
    grouped = (
        resolve_selected_providers(plan, selected_stages, runtime_environment=runtime_environment)
        if stage_providers is None
        else stage_providers
    )
    records = list(grouped.get(provider_id) or [])
    if not records:
        return authorization_result(
            provider_id,
            NOT_REQUIRED,
            reason=f"no selected paid stage runs on {provider_name}",
            messages=[
                f"{provider_name} authorization: not required. No selected paid stage runs on {provider_name}. {NO_JOB_STARTED}"
            ],
        )
    bindings = [dict(record, status=NOT_MEASURED) for record in records]
    if not callable(account_reader):
        return authorization_result(
            provider_id,
            NOT_MEASURED,
            reason=f"no read-only {provider_name} account reader is bound to this session",
            messages=[
                f"{provider_name} authorization: not measured. Bind the Claude Science read-only "
                "provider capability or an environment-backed account reader. "
                f"{UNKNOWN_AUTHORIZATION} {NO_JOB_STARTED}"
            ],
            bindings=bindings,
        )
    response, failure, denied = _read_native_cloud_account(
        account_reader, provider_id=provider_id
    )
    if failure is not None:
        status = REFUSED if denied else NOT_MEASURED
        prefix = "refused" if denied else "not measured"
        return authorization_result(
            provider_id,
            status,
            reason="the read-only account check failed",
            messages=[
                f"{provider_name} authorization: {prefix}. The read-only account check failed: "
                f"{failure}. {NO_JOB_STARTED}"
            ],
            bindings=bindings,
        )
    if isinstance(response, Mapping):
        denial = _denial_text(response)
        if denial is not None:
            return authorization_result(
                provider_id,
                REFUSED,
                reason=f"{provider_name} denied the read-only account check",
                messages=[
                    f"{provider_name} authorization: refused. The read-only account check was denied: "
                    f"{sanitized_reason(denial)}. {NO_JOB_STARTED}"
                ],
                bindings=bindings,
            )
    if not _native_cloud_read_succeeded(response):
        return authorization_result(
            provider_id,
            NOT_MEASURED,
            reason="the read returned no affirmative account-access result",
            messages=[
                f"{provider_name} authorization: not measured. The read-only account response did "
                f"not affirm access. {UNKNOWN_AUTHORIZATION} {NO_JOB_STARTED}"
            ],
            bindings=bindings,
        )
    checked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    for binding in bindings:
        binding["status"] = AUTHORIZED
    return authorization_result(
        provider_id,
        AUTHORIZED,
        reason="the read-only account check succeeded",
        messages=[f"{provider_name} read authorization verified at {checked_at}. No compute was allocated."],
        bindings=bindings,
        checked_at=checked_at,
    )


# The capability report.
#
# The report names the provider the campaign actually resolves to. A campaign
# that resolves no stage to fal reads no word about fal. A campaign on Modal
# is measured through the same free workspace read the preflight uses, and a
# campaign that selects no provider is told that plainly rather than being told
# a provider it does not use went unmeasured.
SESSION_TOOL_PROVIDER_ID = "local"
FULL_CAMPAIGN_HEADLINE = (
    "The full campaign is available: de novo backbones through sequence design, "
    "three predictor arms, ranking, and pictures."
)


def _sequence_redesign_lines() -> list[str]:
    """Describe the shape that needs no provider, and what it does not establish."""
    return [
        "Sequence redesign on a supplied backbone is available.",
        "It provides evidence about sequences compatible with the supplied backbone.",
        "It inherits the parent backbone's binding.",
        "It provides weak evidence about novel binders.",
        "It does not establish binding by a novel binder.",
    ]


def _session_tool_shape() -> dict[str, Any]:
    """Return the campaign shape that runs entirely on installed session tools."""
    return {
        "name": "Sequence redesign on a supplied backbone",
        "available": True,
        "description": (
            "Sequence design, co-folding, scoring, ranking, and pictures use "
            "Claude Science installed tools. They need no provider account, "
            "outside credential, or network approval."
        ),
    }


def _resolved_endpoints(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the fal endpoint mapping from either a resolved config or a run plan.

    A resolved config records endpoints under ``provider_endpoints``. The run plan
    that materialize freezes records the same URLs under ``context`` instead, and
    the preflight and execute paths hand this module the plan, not the config.
    Reading only ``provider_endpoints`` therefore made every fal stage refuse with
    "No valid fal application endpoint is configured" whenever the caller passed a
    plan, even though the plan carried the URL and the credential was good.

    The plan's ``context`` supplies the base and ``provider_endpoints`` overrides
    it, so a caller that has both keeps the config's value. Only the fields
    ``FAL_APPLICATIONS`` names are copied across, so an unrelated context entry can
    never be read as an endpoint.
    """
    resolved: dict[str, Any] = {}
    context = config.get("context")
    if isinstance(context, Mapping):
        resolved.update(
            {field: context[field] for field, _application in FAL_APPLICATIONS if field in context}
        )
    endpoints = config.get("provider_endpoints")
    if isinstance(endpoints, Mapping):
        resolved.update(endpoints)
    return resolved


def configured_fal_applications(config: Mapping[str, Any]) -> list[str]:
    """Return the fal application fields this config resolves to a real endpoint."""
    endpoints = _resolved_endpoints(config)
    return [
        endpoint_field
        for endpoint_field, _application in FAL_APPLICATIONS
        if _endpoint_parts(endpoints.get(endpoint_field)) is not None
    ]


def _selected_fal_applications(
    config: Mapping[str, Any], endpoint_fields: Sequence[str] | None
) -> tuple[tuple[str, str], ...]:
    """Return only the fal applications this caller selected for measurement."""
    known = {field for field, _application in FAL_APPLICATIONS}
    if endpoint_fields is None:
        configured = set(configured_fal_applications(config))
        selected = configured or known
    else:
        selected = {str(field) for field in endpoint_fields}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"unknown fal authorization endpoint fields: {unknown}")
    return tuple(
        (field, application)
        for field, application in FAL_APPLICATIONS
        if field in selected
    )


def selected_providers(
    config: Mapping[str, Any],
    *,
    runtime_environment: Mapping[str, str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Group the paid stages this config selects by the provider each resolves to.

    A campaign that configures a fal application endpoint has already chosen
    fal, even before composition gives it stages, so the endpoint counts as a
    selection on its own. Everything else is read off the resolved stages, which
    is the same resolution the preflight and the approval estimator use.
    """
    stages = config.get("stages")
    grouped = resolve_selected_providers(
        config,
        stages if isinstance(stages, list) else [],
        runtime_environment=runtime_environment,
    )
    if FAL_PROVIDER_ID not in grouped and configured_fal_applications(config):
        grouped[FAL_PROVIDER_ID] = []
    return grouped


def _fal_capability_status(applications: Sequence[Mapping[str, str]]) -> str:
    """Reduce the selected application results to one fal authorization state."""
    statuses = {item["status"] for item in applications}
    if statuses == {AUTHORIZED}:
        return AUTHORIZED
    return REFUSED if REFUSED in statuses else NOT_MEASURED


def _modal_shape_description(stage_ids: Sequence[str]) -> str:
    """Describe the Modal shape by the stages the campaign actually resolves to it."""
    if not stage_ids:
        return "This campaign names Modal, and no selected paid stage resolves to it."
    noun = "stage runs" if len(stage_ids) == 1 else "stages run"
    return (
        f"{len(stage_ids)} selected paid {noun} on the Modal workspace this "
        f"campaign names: {', '.join(stage_ids)}."
    )


def _unselected_shape(stages_declared: bool) -> dict[str, Any] | None:
    """Return the shape a campaign has when no stage resolves to a cloud provider."""
    if not stages_declared:
        return None
    return {
        "name": "Session tools only",
        "available": True,
        "description": (
            "No selected stage resolves to a paid provider route, so this "
            "campaign needs no provider account, outside credential, or "
            "network approval."
        ),
    }


def _combined_status(statuses: Sequence[str]) -> str:
    """Reduce every selected provider to the one state the campaign is in."""
    if REFUSED in statuses:
        return REFUSED
    if statuses and all(status == AUTHORIZED for status in statuses):
        return AUTHORIZED
    if NOT_MEASURED in statuses:
        return NOT_MEASURED
    return NOT_REQUIRED


def probe_capabilities(
    config: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    transport: Transport | None = None,
    timeout_seconds: float = 10.0,
    allow_network: bool = False,
    providers: Sequence[str] | None = None,
    endpoint_fields: Sequence[str] | None = None,
    fal_executable: str = FAL_CREDENTIAL_WRAPPER,
    fal_credential_route: str | None = None,
    details_reader: Callable[[Mapping[str, str]], Any] | None = None,
    identity_reader: Callable[[], Any] | None = None,
    runpod_reader: Callable[[Mapping[str, str]], Any] | None = None,
    lambda_reader: Callable[[Mapping[str, str]], Any] | None = None,
    runtime_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Report the campaign shapes the providers this config selects can run.

    Only a selected provider is probed and only a selected provider is named.
    ``providers`` pins the set for a caller that has already resolved it, which
    is how the preflight keeps this report to the one provider it has not
    already measured itself.
    """
    grouped = (
        selected_providers(config, runtime_environment=runtime_environment)
        if providers is None
        else {str(provider_id): [] for provider_id in providers}
    )
    cloud_providers = sorted(
        provider_id for provider_id in grouped if provider_id != SESSION_TOOL_PROVIDER_ID
    )
    stages = config.get("stages")
    stages_declared = isinstance(stages, list)

    applications: list[dict[str, Any]] = []
    refusals: list[Refusal] = []
    errors: list[str] = []
    statuses: list[str] = []
    shapes: list[dict[str, Any]] = []
    report: dict[str, Any] = {}

    if FAL_PROVIDER_ID in grouped:
        endpoints = _resolved_endpoints(config)
        applications = [
            probe_application(
                endpoint_field,
                application,
                endpoints.get(endpoint_field),
                environ=environ,
                transport=transport,
                timeout_seconds=timeout_seconds,
                allow_network=allow_network,
                fal_executable=fal_executable,
                fal_credential_route=fal_credential_route,
            )
            for endpoint_field, application in _selected_fal_applications(
                config, endpoint_fields
            )
        ]
        fal_status = _fal_capability_status(applications)
        statuses.append(fal_status)
        report["fal_status"] = fal_status
        errors.extend(
            f"{item['application']} access was refused: {item['reason']}"
            for item in applications
            if item["status"] == REFUSED
        )
        errors.extend(
            f"{item['application']} authorization was not measured: {item['reason']}"
            for item in applications
            if item["status"] == NOT_MEASURED
        )
        refusals.extend(
            _authorization_refusal(item)
            for item in applications
            if item["status"] == REFUSED
        )
        shapes.append(
            {
                "name": "Full campaign",
                "available": fal_status == AUTHORIZED,
                "description": (
                    "De novo backbones through sequence design, three predictor arms, "
                    "ranking, and pictures."
                ),
            }
        )

    if MODAL_PROVIDER_ID in grouped:
        modal_authorization = probe_modal_authorization(
            config,
            stages if stages_declared else [],
            details_reader=details_reader,
            identity_reader=identity_reader,
            stage_providers=grouped,
            runtime_environment=runtime_environment,
        )
        modal_stage_ids = [
            record["stage_id"] for record in grouped.get(MODAL_PROVIDER_ID) or []
        ]
        statuses.append(modal_authorization["status"])
        errors.extend(modal_authorization["errors"])
        report["modal_authorization"] = modal_authorization
        report["modal_stage_ids"] = modal_stage_ids
        shapes.append(
            {
                "name": "Paid stages on your own Modal account",
                "available": modal_authorization["status"] == AUTHORIZED,
                "description": _modal_shape_description(modal_stage_ids),
            }
        )

    if RUNPOD_PROVIDER_ID in grouped:
        runpod_authorization = probe_runpod_authorization(
            config,
            stages if stages_declared else [],
            account_reader=runpod_reader,
            stage_providers=grouped,
            runtime_environment=runtime_environment,
        )
        runpod_stage_ids = [
            record["stage_id"] for record in grouped.get(RUNPOD_PROVIDER_ID) or []
        ]
        statuses.append(runpod_authorization["status"])
        errors.extend(runpod_authorization["errors"])
        report["runpod_authorization"] = runpod_authorization
        report["runpod_stage_ids"] = runpod_stage_ids
        shapes.append(
            {
                "name": "Paid stages on your own RunPod account",
                "available": runpod_authorization["status"] == AUTHORIZED,
                "description": (
                    f"{len(runpod_stage_ids)} selected paid stage(s) run through the "
                    "bound RunPod host or environment-backed client."
                ),
            }
        )

    if LAMBDA_PROVIDER_ID in grouped:
        lambda_authorization = probe_lambda_authorization(
            config,
            stages if stages_declared else [],
            account_reader=lambda_reader,
            stage_providers=grouped,
            runtime_environment=runtime_environment,
        )
        lambda_stage_ids = [
            record["stage_id"] for record in grouped.get(LAMBDA_PROVIDER_ID) or []
        ]
        statuses.append(lambda_authorization["status"])
        errors.extend(lambda_authorization["errors"])
        report["lambda_authorization"] = lambda_authorization
        report["lambda_stage_ids"] = lambda_stage_ids
        shapes.append(
            {
                "name": "Paid stages on your own Lambda Cloud account",
                "available": lambda_authorization["status"] == AUTHORIZED,
                "description": (
                    f"{len(lambda_stage_ids)} selected paid stage(s) run through the "
                    "bound Lambda Cloud host or environment-backed client."
                ),
            }
        )

    # A provider this package ships no free authorization probe for is named and
    # reported unmeasured. Attributing its gap to a provider the campaign does
    # not use is the defect this branch exists to avoid.
    unprobed = [
        provider_id
        for provider_id in cloud_providers
        if provider_id
        not in {
            FAL_PROVIDER_ID,
            MODAL_PROVIDER_ID,
            RUNPOD_PROVIDER_ID,
            LAMBDA_PROVIDER_ID,
        }
    ]
    if unprobed:
        statuses.append(NOT_MEASURED)
        report["unprobed_providers"] = unprobed
        for provider_id in unprobed:
            stage_ids = [record["stage_id"] for record in grouped.get(provider_id) or []]
            errors.append(
                f"{provider_id} authorization was not measured: this package ships no free probe"
            )
            shapes.append(
                {
                    "name": f"Paid stages on {provider_id}",
                    "available": False,
                    "description": (
                        f"{len(stage_ids)} selected paid stage(s) resolve to {provider_id}, "
                        f"and this package ships no free authorization probe for it. "
                        f"{UNKNOWN_AUTHORIZATION}"
                    ),
                }
            )

    if not cloud_providers:
        unselected = _unselected_shape(stages_declared)
        if unselected is not None:
            shapes.append(unselected)

    shapes.append(_session_tool_shape())
    refused = REFUSED in statuses
    blocking = any(status in BLOCKING_AUTHORIZATION_STATES for status in statuses)
    report.update(
        {
            "ok": not blocking,
            "providers": cloud_providers,
            "stages_declared": stages_declared,
            "applications": applications,
            "generation_status": _combined_status(statuses),
            "campaign_shapes": shapes,
            "errors": errors,
            "exit_code": exit_code_for_result(verified=not blocking, refused=refused),
            "refusals": [refusal.as_dict() for refusal in refusals],
            "refusal_text": "\n\n".join(refusal.text() for refusal in refusals),
        }
    )
    report["text"] = capability_report_text(report)
    return report


def _available_headline(providers: Sequence[str]) -> str:
    """Name what an authorized campaign can run, on the provider it runs on."""
    if FAL_PROVIDER_ID in providers:
        return FULL_CAMPAIGN_HEADLINE
    if MODAL_PROVIDER_ID in providers:
        return (
            "Every selected paid stage in this campaign is authorized on your own "
            "Modal account."
        )
    if RUNPOD_PROVIDER_ID in providers:
        return "Every selected paid stage in this campaign is authorized on your own RunPod account."
    if LAMBDA_PROVIDER_ID in providers:
        return "Every selected paid stage in this campaign is authorized on your own Lambda Cloud account."
    return "Every selected paid stage in this campaign is authorized on its provider."


def _unselected_lines(stages_declared: bool) -> list[str]:
    """Say why no provider was measured, without naming one the campaign never chose."""
    if not stages_declared:
        return [
            "This configuration declares no stages, so no stage has resolved to a "
            "provider yet.",
            "Compose it with an execution profile to select one.",
        ]
    return [
        "No selected stage in this campaign runs on a cloud provider.",
        "Every selected stage resolves to a route that needs no provider account, "
        "outside credential, or network approval, so there is no provider "
        "authorization to measure.",
    ]


def _modal_lines(report: Mapping[str, Any]) -> list[str]:
    """Report the Modal measurement, naming the stages it covers."""
    authorization = report.get("modal_authorization")
    if not isinstance(authorization, Mapping):
        return []
    stage_ids = [str(item) for item in report.get("modal_stage_ids") or []]
    lines: list[str] = []
    if stage_ids:
        noun = "stage" if len(stage_ids) == 1 else "stages"
        lines.append(
            f"This campaign runs {len(stage_ids)} selected paid {noun} on your own "
            f"Modal account: {', '.join(stage_ids)}."
        )
    lines.extend(str(message) for message in authorization.get("messages") or [])
    return lines


def _unprobed_lines(report: Mapping[str, Any]) -> list[str]:
    """Name a selected provider this package ships no free authorization probe for."""
    unprobed = report.get("unprobed_providers")
    if not isinstance(unprobed, list) or not unprobed:
        return []
    names = ", ".join(str(provider_id) for provider_id in unprobed)
    return [
        f"This campaign resolves selected paid stages to {names}. This package ships "
        f"no free authorization probe for it, so its authorization is not measured."
    ]


def capability_report_text(report: Mapping[str, Any]) -> str:
    """Render the reader-facing capability report for the selected providers."""
    providers = report.get("providers")
    if not isinstance(providers, list):
        providers = [FAL_PROVIDER_ID] if report.get("applications") else []
    providers = [str(provider_id) for provider_id in providers]
    authorized = report.get("generation_status") == AUTHORIZED
    fal_status = report.get("fal_status")
    lines: list[str] = []

    if not providers:
        lines.extend(_unselected_lines(report.get("stages_declared") is True))
    elif fal_status is not None:
        # The fal report leads with the shape, because a refused fal application
        # refuses the generation route rather than the account.
        if authorized:
            lines.append(_available_headline(providers))
        else:
            lines.extend(_sequence_redesign_lines())
        if fal_status == REFUSED:
            lines.append("Backbone generation is refused for one or more fal applications.")
        elif fal_status != AUTHORIZED:
            lines.append(
                "Backbone generation remains unmeasured until every fal application answers."
            )
        lines.extend(_modal_lines(report))
        lines.extend(_unprobed_lines(report))
    else:
        # Every other provider leads with the measurement, then names the shape
        # that stays available when the measurement did not pass.
        if authorized:
            lines.append(_available_headline(providers))
        lines.extend(_modal_lines(report))
        lines.extend(_unprobed_lines(report))
        if not authorized:
            lines.extend(_sequence_redesign_lines())

    for item in report.get("applications", []):
        if not isinstance(item, Mapping):
            continue
        application = item.get("application", "Unknown application")
        status = item.get("status", NOT_MEASURED)
        reason = item.get("reason", "No reason was recorded.")
        lines.append(f"{application}: {status}. {reason}")

    lines.append("Available campaign shapes:")
    for shape in report.get("campaign_shapes", []):
        if not isinstance(shape, Mapping):
            continue
        availability = "available" if shape.get("available") is True else "unavailable"
        lines.append(f"{shape.get('name')}: {availability}. {shape.get('description')}")
    return "\n\n".join(lines)


def _wrapped_fal_probe_main(arguments: Sequence[str]) -> int:
    """Write one sanitized probe result from inside a credential wrapper."""
    if len(arguments) != 6 or arguments[0] != WRAPPED_FAL_PROBE_COMMAND:
        return 2
    _command, endpoint_field, application, endpoint, result_path, timeout_text = arguments
    try:
        timeout_seconds = float(timeout_text)
    except ValueError:
        return 2
    wrapped_environment = dict(os.environ)
    # The wrapper contract injects FAL_KEY into its child. A selector inherited
    # from the parent may name an unavailable host alias, so the child must read
    # the wrapper-owned variable just as every wrapped adapter client does.
    wrapped_environment.pop(fal_invocation.CREDENTIAL_ENVIRONMENT_OVERRIDE, None)
    result = probe_application(
        endpoint_field,
        application,
        endpoint,
        environ=wrapped_environment,
        timeout_seconds=timeout_seconds,
        allow_network=True,
        fal_credential_route=fal_invocation.ROUTE_DIRECT,
    )
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_wrapped_fal_probe_main(sys.argv[1:]))
