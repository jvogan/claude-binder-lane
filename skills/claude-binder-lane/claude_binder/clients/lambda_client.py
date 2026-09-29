"""Carry one Lambda Cloud unit of work for the Lambda route, minus three calls.

``claude_binder.adapters.lambda_platform`` forwards every public function to
``runpod_platform`` with ``route=LAMBDA_ROUTE``, so the two providers already
share one dispatch lock, one submission ledger, one uncertain-submission
recovery path, one receipt validator, one cleanup rule, and one financial
reconciliation. The lifecycle owns no network code. It asks its caller for a
host binding whose ``compute.create`` returns a handle, then drives that handle
through ``submit_job``, ``attach_job``, ``result`` and ``close``. Nothing in the
package supplied such a handle for Lambda, so a caller had to write the whole
thing. This module is the Binder half of that handle.

**Lambda Cloud's REST contract is not recorded anywhere in this tree.** No
module, reference page, profile, design note, or seed file here records a Lambda
Cloud host, path, request field, response field, status string, or
authentication scheme. ``the-nine-decisions.md`` states the gap directly: Lambda
Cloud's own HTTP API is not RunPod's, so Lambda needs its own client. The two
parameter names the shipped profile declares carry a gap note saying nothing
here validates them. Guessing the missing half would produce a client that
fails on first contact and looks finished, so this module ships the half it can
prove and leaves the other half to a caller-supplied transport.

**What this module owns.** Credential discipline, so the secret is read once
per handle from a variable the caller names and never reaches an argv, a URL, a
submission record, a settlement record, or a receipt. Provider-parameter
validation, so a run stops before compute exists rather than after. Identifier
validation, so no recorded value can reshape a request path. The polling loop,
the terminal-state vocabulary the lifecycle actually reads, the cleanup
bookkeeping, and the ``ClientHost`` wiring. HTTP hygiene for the transport
author: redirects refused, a non-2xx raised with its body attached, a non-object
body refused, and every path segment redacted out of error messages.

**What the caller owns.** Three provider calls, supplied as a
:class:`LambdaTransport`. Each one needs a fact this tree does not record, and
each carries a TODO naming what would close it.

Every call takes an ``opener`` so a test never touches the network, and the
polling loop takes a ``sleep`` so a test never waits. Both are copied from
``runpod_client``, which is this module's model throughout.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


# LAMBDA_ROUTE declares no ``api_key_environment_key`` where RUNPOD_ROUTE
# declares RUNPOD_API_KEY, so this module has no default credential variable
# and the caller names one. The shipped profile's ``credentials`` gap note
# records the same requirement.
DEFAULT_CREDENTIAL_ENV: str | None = None

ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# No Lambda Cloud identifier grammar is recorded in this tree, so this is a
# conservative safety check on a value that reaches a request path rather than
# a claim about the provider's format. It refuses a path separator, a query, a
# fragment, an escape, whitespace, and an empty value.
PROVIDER_REFERENCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_POLL_SECONDS = 5.0

# TODO(lambda-provider-params): unverified names.
# ``small-run-lambda.template.json`` declares these two on all four paid stages
# and its own ``provider_parameters`` gap note says nothing here validates them.
# This module refuses a submit that omits either, because the shipped profile
# says they are what a launch needs. To settle the names, read Lambda Cloud's
# own instance-launch reference and compare field for field.
REQUIRED_PROVIDER_PARAMS = ("instance_type_name", "region_name")

# ``compose`` accepts this placeholder and ``materialize`` refuses it. The
# client refuses it too, so a hand-built plan cannot reach a provider call with
# an unread value.
UNRESOLVED_PLACEHOLDER = "__REQUIRED__"

# TODO(worker-contract): unverified. These are the five keywords
# ``runpod_platform.submit_stage`` passes to a handle, packaged into one request
# object for the transport. Lambda Cloud runs the operator's own image, so what
# reads them is that image and not the provider. They are Binder's contract with
# a Binder worker image. To settle them, run one image and read a real receipt.
WORK_REQUEST_KEYS = ("intent", "command", "inputs", "outputs", "run_timeout_s")

# ``runpod_platform.settle_stage`` accepts ``state == "succeeded"`` and nothing
# else, so the vocabulary is the lifecycle's rather than any provider's. A
# transport translates Lambda's own status into exactly one of these words. The
# client refuses any other word instead of guessing which way it resolves.
STATE_PENDING = "pending"
STATE_SUCCEEDED = "succeeded"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"
STATE_TIMED_OUT = "timed_out"

TERMINAL_STATES = (STATE_SUCCEEDED, STATE_FAILED, STATE_CANCELLED, STATE_TIMED_OUT)
LIFECYCLE_STATES = (STATE_PENDING, *TERMINAL_STATES)


class LambdaClientError(RuntimeError):
    """A Lambda Cloud call cannot be completed or cannot be trusted."""


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "Lambda Cloud redirect rejected", headers, fp
        )


def _default_opener(request, timeout=None):
    """Open one request with redirects refused. Replaced wholesale in tests."""
    opener = urllib.request.build_opener(RejectRedirects())
    return opener.open(request, timeout=timeout)


def credential_environment_name(value: str) -> str:
    """Validate an environment-variable name without reading its value.

    There is no default to fall back to. ``LAMBDA_ROUTE`` declares no
    credential variable, so a caller that names none has named none.
    """
    if not isinstance(value, str) or ENVIRONMENT_NAME_RE.fullmatch(value) is None:
        raise LambdaClientError(
            "the Lambda Cloud credential selector must be an environment-variable "
            "name. The Lambda route declares no default variable, so name the one "
            "your session exports."
        )
    return value


def read_credential(
    credential_env: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return the credential from its named variable. The value never leaves."""
    source = os.environ if environ is None else environ
    key = credential_environment_name(credential_env)
    value = str(source.get(key, "") or "").strip()
    if not value:
        raise LambdaClientError(
            f"{key} is unavailable to the Lambda Cloud client. Set it in this "
            "process's environment. Credential presence proves only presence; the "
            "provider authorization preflight verifies account access."
        )
    return value


def validate_provider_reference(value: object, *, field: str = "job id") -> str:
    """Refuse a recorded identifier that could reshape a request path."""
    text = str(value or "").strip()
    if PROVIDER_REFERENCE_RE.fullmatch(text) is None:
        raise LambdaClientError(
            f"a Lambda Cloud {field} must be an unpunctuated identifier, and "
            f"{value!r} is not"
        )
    return text


def validate_base_url(value: object) -> str:
    """Refuse a base URL that would leak the credential or hide a path.

    The credential travels in a header, so plaintext is refused outright. A
    query, a fragment, or embedded userinfo would survive into every built URL,
    so each is refused as well.
    """
    text = str(value or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme != "https" or not parsed.hostname:
        raise LambdaClientError(
            f"a Lambda Cloud base URL must be an https URL with a host, and {value!r} is not"
        )
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise LambdaClientError(
            "a Lambda Cloud base URL carries no query, fragment, or embedded credential"
        )
    return text


def provider_url(base_url: str, *segments: object) -> str:
    """Build one URL under a caller-supplied base, validating every segment.

    No path is named here. This tree records no Lambda Cloud endpoint, so the
    transport author supplies both the base and the segments.
    """
    parts = [
        validate_provider_reference(item, field="path segment") for item in segments
    ]
    return "/".join([validate_base_url(base_url), *parts])


def call(
    url: str,
    *,
    authorization: str,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    opener: Callable[..., Any] = _default_opener,
) -> dict[str, Any]:
    """Make one Lambda Cloud call and return its parsed JSON object.

    A non-2xx answer is raised with the response body attached, because a bare
    status code is the difference between a diagnosis and a blind retry. A body
    that is not a JSON object is an error too: a caller that reads a field out
    of a string would report a fabricated result.

    TODO(lambda-auth-scheme): unverified. ``authorization`` is the complete
    ``Authorization`` header value because this tree records no Lambda Cloud
    authentication scheme, and a scheme this module invented would fail on first
    contact. To settle it, read Lambda Cloud's own authentication reference and
    pass the header form it documents.
    """
    if not isinstance(authorization, str) or not authorization.strip():
        raise LambdaClientError(
            "a Lambda Cloud call needs a complete Authorization header value"
        )
    body = (
        None
        if payload is None
        else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )
    headers = {
        "Authorization": authorization.strip(),
        "Accept": "application/json",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with opener(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise LambdaClientError(
            f"Lambda Cloud answered HTTP {error.code} for {method} {_redacted(url)}: "
            + (detail or "no response body")
        ) from error
    except urllib.error.URLError as error:
        raise LambdaClientError(
            f"Lambda Cloud was unreachable for {method} {_redacted(url)}: {error.reason}"
        ) from error
    try:
        parsed = json.loads(raw)
    except ValueError as error:
        raise LambdaClientError(
            f"Lambda Cloud returned a body that is not JSON for {method} {_redacted(url)}"
        ) from error
    if not isinstance(parsed, dict):
        raise LambdaClientError(
            f"Lambda Cloud returned {type(parsed).__name__} rather than a JSON object "
            f"for {method} {_redacted(url)}"
        )
    return parsed


def _redacted(url: str) -> str:
    """Render a URL for a message with every path segment removed.

    ``runpod_client`` keeps the segments it knows are operation names. No
    Lambda Cloud path grammar is recorded here, so no segment can be told apart
    from an account-identifying value and every one of them goes.
    """
    parsed = urllib.parse.urlparse(url)
    segments = [item for item in parsed.path.split("/") if item]
    return (
        parsed.scheme
        + "://"
        + str(parsed.hostname or "")
        + "".join("/<redacted>" for _ in segments)
    )


def require_provider_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse provider parameters the shipped Lambda profile says a launch needs.

    This runs before the handle exists, so an incomplete plan stops before any
    compute is created rather than after. The names come from
    ``small-run-lambda.template.json`` and are unverified; see
    ``REQUIRED_PROVIDER_PARAMS``.
    """
    if not isinstance(params, Mapping):
        raise LambdaClientError("Lambda Cloud provider parameters must be an object")
    missing = [
        name
        for name in REQUIRED_PROVIDER_PARAMS
        if not str(params.get(name, "") or "").strip()
    ]
    if missing:
        raise LambdaClientError(
            "the Lambda Cloud provider parameters name no "
            + ", ".join(missing)
            + ". Set them in the adapter's lambda parameters."
        )
    unresolved = [
        name
        for name in REQUIRED_PROVIDER_PARAMS
        if str(params.get(name, "")).strip() == UNRESOLVED_PLACEHOLDER
    ]
    if unresolved:
        raise LambdaClientError(
            "the Lambda Cloud provider parameters are still placeholders for "
            + " and ".join(unresolved)
            + ". Resolve them from your own account before dispatch."
        )
    return dict(params)


@dataclass(frozen=True)
class LambdaTransport:
    """The three provider calls this tree cannot confirm, supplied by a caller.

    Each callable is invoked with keyword arguments only and returns a mapping.
    The credential is passed in on every call and is never stored by the client
    anywhere a record could reach it.

    ``launch(credential, provider_params, request)`` starts one unit of work and
    returns a mapping carrying a durable provider identifier under ``id``. The
    client validates that identifier and records nothing else from the response.

    TODO(lambda-launch): unverified. This tree records no Lambda Cloud launch
    path, request field, or response field. To settle it, read Lambda Cloud's own
    instance-launch reference and confirm which response field carries the
    durable identifier a later read can address.

    ``status(credential, provider_params, job_id)`` reads one unit of work and
    returns ``state`` as exactly one of :data:`LIFECYCLE_STATES`, plus optional
    ``exit_code``, ``usage`` and ``provider_status``.

    TODO(lambda-status): unverified. This tree records no Lambda Cloud status
    vocabulary, so the translation into the lifecycle's five words belongs to
    whoever reads that vocabulary. To settle it, read Lambda Cloud's own status
    reference and map each value it publishes. An exit code is separate again:
    Lambda Cloud runs the operator's image, so the image reports the exit code
    and the provider does not.

    ``terminate(credential, provider_params, job_id)`` stops billable compute.
    It must succeed when the work has already stopped, because cleanup calls it
    for every unit of work this handle created.

    TODO(lambda-terminate): unverified. This tree records no Lambda Cloud
    termination path and no proof that finished work stops billing on its own.
    To settle it, read Lambda Cloud's own termination reference and confirm what
    it answers for work that has already stopped.
    """

    launch: Callable[..., Mapping[str, Any]]
    status: Callable[..., Mapping[str, Any]]
    terminate: Callable[..., Mapping[str, Any]]

    def __post_init__(self) -> None:
        for name in ("launch", "status", "terminate"):
            if not callable(getattr(self, name)):
                raise LambdaClientError(
                    f"a Lambda Cloud transport needs a callable {name}"
                )


class LambdaJob:
    """One unit of work, polled until the transport reports a terminal state."""

    def __init__(
        self,
        job_id: str,
        *,
        credential: str,
        provider_params: Mapping[str, Any],
        transport: LambdaTransport,
        sleep: Callable[[float], Any] = time.sleep,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        run_timeout_s: int | None = None,
    ) -> None:
        self.job_id = validate_provider_reference(job_id)
        self.id = self.job_id
        self.run_timeout_s = run_timeout_s
        # Set once a terminal state has actually been read, so a settlement
        # record can tell "already finished" from "never observed".
        self.terminal_status: str | None = None
        self._credential = credential
        self._provider_params = dict(provider_params)
        self._transport = transport
        self._sleep = sleep
        self._poll_seconds = float(poll_seconds)

    def status(self) -> dict[str, Any]:
        """Read one status through the transport and check its state word."""
        response = self._transport.status(
            credential=self._credential,
            provider_params=dict(self._provider_params),
            job_id=self.job_id,
        )
        if not isinstance(response, Mapping):
            raise LambdaClientError(
                "a Lambda Cloud transport status must answer with an object"
            )
        state = str(response.get("state", "") or "").strip().lower()
        if state not in LIFECYCLE_STATES:
            raise LambdaClientError(
                f"a Lambda Cloud transport reported state {response.get('state')!r}, "
                "which is none of "
                + ", ".join(LIFECYCLE_STATES)
                + "; the job state must not be guessed"
            )
        return dict(response)

    def terminate(self) -> dict[str, Any]:
        """Stop this unit of work through the transport."""
        response = self._transport.terminate(
            credential=self._credential,
            provider_params=dict(self._provider_params),
            job_id=self.job_id,
        )
        return dict(response) if isinstance(response, Mapping) else {}

    def result(self, *, max_polls: int | None = None) -> dict[str, Any]:
        """Poll to a terminal state, then answer in the lifecycle's shape.

        ``exit_code`` comes from the transport and stays ``None`` when the
        transport reports none. Substituting 0 would forge the settlement gate,
        which reads ``state == "succeeded"`` and ``exit_code == 0`` together.
        ``usage`` is passed through verbatim and is ``None`` when absent, which
        the lifecycle records as ``pending-provider-usage``.
        """
        polls = 0
        while True:
            response = self.status()
            state = str(response.get("state", "")).strip().lower()
            if state in TERMINAL_STATES:
                return self._terminal(response, state)
            polls += 1
            if max_polls is not None and polls >= max_polls:
                raise LambdaClientError(
                    f"the Lambda Cloud job is still {state} after {polls} polls; it "
                    "was not abandoned and can be resumed from its recorded job id"
                )
            self._sleep(self._poll_seconds)

    def _terminal(self, response: Mapping[str, Any], state: str) -> dict[str, Any]:
        self.terminal_status = state
        exit_value = response.get("exit_code")
        exit_code = (
            exit_value
            if isinstance(exit_value, int) and not isinstance(exit_value, bool)
            else None
        )
        usage = response.get("usage")
        return {
            "job_id": self.job_id,
            "state": state,
            "provider_status": response.get("provider_status"),
            "exit_code": exit_code,
            "usage": dict(usage) if isinstance(usage, Mapping) else None,
        }


class LambdaHandle:
    """Bind one set of Lambda parameters to the host protocol the lifecycle drives."""

    def __init__(
        self,
        provider_params: Mapping[str, Any],
        *,
        credential: str,
        transport: LambdaTransport,
        sleep: Callable[[float], Any] = time.sleep,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> None:
        self.provider_params = require_provider_params(provider_params)
        self._credential = credential
        self._transport = transport
        self._sleep = sleep
        self._poll_seconds = poll_seconds
        self._jobs: list[LambdaJob] = []

    def submit_job(
        self,
        *,
        intent: str,
        command: str,
        inputs: Sequence[Mapping[str, str]] = (),
        outputs: Sequence[Any] = (),
        run_timeout_s: int,
    ) -> LambdaJob:
        """Start the stage command as one unit of work.

        The five keys in the request are Binder's worker contract, listed in
        ``WORK_REQUEST_KEYS``. Lambda Cloud runs the operator's own image, so an
        image that does not implement them will start and then fail.
        """
        response = self._transport.launch(
            credential=self._credential,
            provider_params=dict(self.provider_params),
            request={
                "intent": str(intent),
                "command": str(command),
                "inputs": [dict(item) for item in inputs],
                "outputs": list(outputs),
                "run_timeout_s": int(run_timeout_s),
            },
        )
        if not isinstance(response, Mapping):
            raise LambdaClientError(
                "a Lambda Cloud transport launch must answer with an object"
            )
        job_id = response.get("id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise LambdaClientError(
                "Lambda Cloud accepted the launch and returned no identifier, so the "
                "work cannot be tracked or terminated. Inspect the provider ledger "
                "before resubmitting."
            )
        return self._job(job_id.strip(), run_timeout_s=int(run_timeout_s))

    def attach_job(self, job_id: str) -> LambdaJob:
        """Reattach to a recorded identifier. Never relaunches."""
        return self._job(job_id)

    def _job(self, job_id: str, *, run_timeout_s: int | None = None) -> LambdaJob:
        job = LambdaJob(
            job_id,
            credential=self._credential,
            provider_params=self.provider_params,
            transport=self._transport,
            sleep=self._sleep,
            poll_seconds=self._poll_seconds,
            run_timeout_s=run_timeout_s,
        )
        self._jobs.append(job)
        return job

    def close(self, *, intent: str) -> dict[str, Any]:
        """Terminate every unit of work this handle started.

        ``runpod_client`` skips a job whose terminal status it already read,
        because a RunPod Serverless endpoint bills per job and outlives the
        handle. Nothing in this tree proves the same of Lambda Cloud, so this
        handle terminates everything it started and lets the transport absorb a
        second call. An unbounded billing leak is the worse of the two failures,
        and the lifecycle's own settlement rule says so.
        """
        errors: list[str] = []
        terminated: list[str] = []
        already_terminal: list[str] = []
        for job in self._jobs:
            if job.terminal_status is not None:
                already_terminal.append(job.job_id)
            try:
                job.terminate()
            except LambdaClientError as exc:
                errors.append(f"{job.job_id}: {exc}")
            else:
                terminated.append(job.job_id)
        if errors:
            return {"ok": False, "intent": intent, "errors": errors}
        return {
            "ok": True,
            "intent": intent,
            "terminated": terminated,
            "already_terminal": already_terminal,
        }


def create_handle(
    *,
    credential_env: str,
    transport: LambdaTransport,
    environ: Mapping[str, str] | None = None,
    sleep: Callable[[float], Any] = time.sleep,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> Callable[[Mapping[str, Any]], LambdaHandle]:
    """Return the handle factory ``lambda_platform.ClientHost`` is missing.

    The factory reads its credential once per handle, from the variable the
    caller names, so a credential never reaches an argv or a submission record.
    ``credential_env`` has no default because ``LAMBDA_ROUTE`` declares none.
    """
    if not isinstance(transport, LambdaTransport):
        raise LambdaClientError(
            "the Lambda Cloud client needs a LambdaTransport. This tree records no "
            "Lambda Cloud REST contract, so the three provider calls are supplied."
        )
    credential_key = credential_environment_name(credential_env)

    def factory(provider_params: Mapping[str, Any]) -> LambdaHandle:
        return LambdaHandle(
            require_provider_params(provider_params),
            credential=read_credential(credential_key, environ=environ),
            transport=transport,
            sleep=sleep,
            poll_seconds=poll_seconds,
        )

    return factory


def client_host(**kwargs: Any) -> Any:
    """Return a host binding the Lambda lifecycle accepts as ``host``.

    This is the wiring the route was missing. ``lambda_platform`` itself needs
    no change: it already accepts any callable here.
    """
    from claude_binder.adapters import lambda_platform

    return lambda_platform.ClientHost(create_handle(**kwargs))


def account_reader(
    read_account: Callable[[str], Any],
    *,
    credential_env: str,
    environ: Mapping[str, str] | None = None,
) -> Callable[[Mapping[str, str]], Any]:
    """Return the read-only account reader the authorization preflight wants.

    ``read_account`` is caller-supplied for the same reason the transport is.
    This tree records no Lambda Cloud read that proves account access, so the
    call belongs to whoever reads Lambda Cloud's own reference. The credential
    discipline around it is this module's: the variable name is validated here,
    the value is read only inside the reader, and the reader refuses any request
    that is not a Lambda read.

    TODO(lambda-account-read): unverified. ``runpod_client`` proves access with
    a serverless-endpoint listing that creates and bills nothing. To settle the
    Lambda equivalent, find the cheapest Lambda Cloud read that answers 401 or
    403 for a missing or unauthorized key and allocates no instance.
    """
    from claude_binder.adapters import lambda_platform

    return lambda_platform.environment_account_reader(
        read_account,
        environ=environ,
        api_key_environment_key=credential_environment_name(credential_env),
    )
