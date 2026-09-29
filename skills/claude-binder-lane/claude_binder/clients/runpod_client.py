"""Carry one RunPod Serverless queue job over HTTP for the RunPod route.

``claude_binder.adapters.runpod_platform`` owns the whole dispatch lifecycle:
the dispatch lock, the submission ledger, uncertain-submission recovery,
receipt validation, cleanup, and financial reconciliation. It owns no network
code. It asks its caller for a host binding whose ``compute.create`` returns a
handle, and it drives that handle through ``submit_job``, ``attach_job``,
``result`` and ``close``. Nothing in the package supplied such a handle, so no
profile could select the route. This module is that missing transport, in the
same place and the same idiom as the packaged fal clients.

Two RunPod hosts are involved and they are not interchangeable.

* Queue operations for one Serverless endpoint live on ``api.runpod.ai/v2``.
  Docs: https://docs.runpod.io/runpodctl/reference/runpodctl-serverless#serverless-urls
* The REST v2 control plane, used here only for the read-only account check,
  lives on ``api.runpod.io/v2``.
  Docs: https://docs.runpod.io/api-reference-v2/overview

Every call takes an ``opener`` so a test never touches the network, and the
polling loop takes a ``sleep`` so a test never waits. Redirects are refused and
a non-2xx response is raised with its body attached, both copied from the
packaged fal clients.

**Worker contract.** RunPod Serverless hands a handler an opaque ``input``
object and returns whatever the handler put in ``output``; the platform defines
neither. So the field names this client places inside ``input`` and reads back
out of ``output`` are Binder's own contract with a Binder worker image, not
RunPod's API. They are named in ``WORKER_INPUT_KEYS`` and ``WORKER_OUTPUT_KEYS``
and carry a TODO, because no Binder worker image is deployed to verify them
against.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping, Sequence


QUEUE_HOST = "api.runpod.ai"
REST_HOST = "api.runpod.io"
DEFAULT_CREDENTIAL_ENV = "RUNPOD_API_KEY"
ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# The docs never publish a grammar for an endpoint id, so this is a
# conservative safety check on an argv value rather than a claim about the
# provider's format. It refuses a path separator, a query, and an empty value.
ENDPOINT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_POLL_SECONDS = 5.0

# Job states, verbatim from
# https://docs.runpod.io/serverless/endpoints/job-states#request-job-states
STATE_IN_QUEUE = "IN_QUEUE"
STATE_IN_PROGRESS = "IN_PROGRESS"
STATE_RUNNING = "RUNNING"
STATE_COMPLETED = "COMPLETED"
STATE_FAILED = "FAILED"
STATE_CANCELLED = "CANCELLED"
STATE_TIMED_OUT = "TIMED_OUT"

PENDING_STATES = (STATE_IN_QUEUE, STATE_IN_PROGRESS, STATE_RUNNING)
TERMINAL_STATES = (STATE_COMPLETED, STATE_FAILED, STATE_CANCELLED, STATE_TIMED_OUT)

# The lifecycle in runpod_platform reads ``state`` and accepts only
# ``"succeeded"``, so a provider status is translated once, here.
LIFECYCLE_STATES = {
    STATE_COMPLETED: "succeeded",
    STATE_FAILED: "failed",
    STATE_CANCELLED: "cancelled",
    STATE_TIMED_OUT: "timed_out",
}

# TODO(worker-contract): unverified. A RunPod Serverless handler receives the
# whole ``input`` object and its shape is the handler author's choice, so these
# keys are only a contract once a Binder worker image implements them. To
# settle them, deploy that image and read one real receipt back.
WORKER_INPUT_KEYS = ("intent", "command", "inputs", "outputs", "run_timeout_s")
WORKER_OUTPUT_KEYS = ("exit_code",)


class RunPodClientError(RuntimeError):
    """A RunPod HTTP call cannot be completed or cannot be trusted."""


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "RunPod redirect rejected", headers, fp
        )


def _default_opener(request, timeout=None):
    """Open one request with redirects refused. Replaced wholesale in tests."""
    opener = urllib.request.build_opener(RejectRedirects())
    return opener.open(request, timeout=timeout)


def credential_environment_name(value: str) -> str:
    """Validate an environment-variable name without reading its value."""
    if not isinstance(value, str) or ENVIRONMENT_NAME_RE.fullmatch(value) is None:
        raise RunPodClientError(
            "the RunPod credential selector must be an environment-variable name "
            f"such as {DEFAULT_CREDENTIAL_ENV}"
        )
    return value


def read_credential(
    credential_env: str = DEFAULT_CREDENTIAL_ENV,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return the credential from its named variable. The value never leaves."""
    source = os.environ if environ is None else environ
    key = credential_environment_name(credential_env)
    value = str(source.get(key, "") or "").strip()
    if not value:
        raise RunPodClientError(
            f"{key} is unavailable to the RunPod client. Set it in this process's "
            "environment. Credential presence proves only presence; the provider "
            "authorization preflight verifies account access."
        )
    return value


def validate_endpoint_id(value: object) -> str:
    """Refuse an endpoint id that could reshape the request path."""
    text = str(value or "").strip()
    if ENDPOINT_ID_RE.fullmatch(text) is None:
        raise RunPodClientError(
            "a RunPod serverless endpoint id must be an unpunctuated identifier, "
            f"and {value!r} is not"
        )
    return text


def validate_job_id(value: object) -> str:
    """Refuse a job id that could reshape the request path."""
    text = str(value or "").strip()
    if JOB_ID_RE.fullmatch(text) is None:
        raise RunPodClientError(f"{value!r} is not a usable RunPod job id")
    return text


def queue_url(endpoint_id: str, *segments: str) -> str:
    """Build one queue-operation URL on the serverless host.

    Paths confirmed at
    https://docs.runpod.io/runpodctl/reference/runpodctl-serverless#serverless-urls
    and https://docs.runpod.io/serverless/endpoints/operation-reference
    """
    parts = [validate_endpoint_id(endpoint_id), *(str(item) for item in segments)]
    return "https://" + QUEUE_HOST + "/v2/" + "/".join(parts)


def rest_url(*segments: str) -> str:
    """Build one REST v2 control-plane URL. Read-only callers only."""
    return "https://" + REST_HOST + "/v2/" + "/".join(str(item) for item in segments)


def call(
    url: str,
    *,
    credential: str,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    opener: Callable[..., Any] = _default_opener,
) -> dict[str, Any]:
    """Make one RunPod call and return its parsed JSON object.

    A non-2xx answer is raised with the response body attached, because a bare
    status code is the difference between a diagnosis and a blind retry. A body
    that is not a JSON object is an error too: a caller that reads a field out
    of a string would report a fabricated result.
    """
    if not isinstance(credential, str) or not credential.strip():
        raise RunPodClientError("a RunPod call needs a non-empty credential")
    body = (
        None
        if payload is None
        else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )
    headers = {
        # The operation reference sends the raw key in ``authorization``; the
        # REST v2 reference calls it a bearer token. ``Bearer <key>`` is the
        # form both accept.
        # https://docs.runpod.io/serverless/endpoints/operation-reference
        "Authorization": "Bearer " + credential.strip(),
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
        raise RunPodClientError(
            f"RunPod answered HTTP {error.code} for {method} {_redacted(url)}: "
            + (detail or "no response body")
        ) from error
    except urllib.error.URLError as error:
        raise RunPodClientError(
            f"RunPod was unreachable for {method} {_redacted(url)}: {error.reason}"
        ) from error
    try:
        parsed = json.loads(raw)
    except ValueError as error:
        raise RunPodClientError(
            f"RunPod returned a body that is not JSON for {method} {_redacted(url)}"
        ) from error
    if not isinstance(parsed, dict):
        raise RunPodClientError(
            f"RunPod returned {type(parsed).__name__} rather than a JSON object for "
            f"{method} {_redacted(url)}"
        )
    return parsed


def _redacted(url: str) -> str:
    """Render a URL for a message without its endpoint or job identifiers."""
    parsed = urllib.parse.urlparse(url)
    segments = [item for item in parsed.path.split("/") if item]
    kept = [
        item if index == 0 or not _looks_like_identifier(item) else "<id>"
        for index, item in enumerate(segments)
    ]
    return parsed.scheme + "://" + str(parsed.hostname or "") + "/" + "/".join(kept)


def _looks_like_identifier(segment: str) -> bool:
    return segment not in {
        "run",
        "runsync",
        "status",
        "cancel",
        "health",
        "serverless",
        "billing",
    }


def submit_job(
    endpoint_id: str,
    job_input: Mapping[str, Any],
    *,
    credential: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    opener: Callable[..., Any] = _default_opener,
) -> dict[str, Any]:
    """Queue one asynchronous job and return the provider's submission record.

    ``POST /v2/<endpoint_id>/run`` takes ``{"input": {...}}`` and answers with
    ``id`` and ``status``, documented at
    https://docs.runpod.io/serverless/endpoints/operation-reference#/run
    Maximum payload for this operation is 10 MB, same source.
    """
    response = call(
        queue_url(endpoint_id, "run"),
        credential=credential,
        method="POST",
        payload={"input": dict(job_input)},
        timeout_seconds=timeout_seconds,
        opener=opener,
    )
    job_id = response.get("id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise RunPodClientError(
            "RunPod accepted the submission and returned no job id, so the job "
            "cannot be tracked or cancelled. Inspect the endpoint's queue before "
            "resubmitting."
        )
    return response


def job_status(
    endpoint_id: str,
    job_id: str,
    *,
    credential: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    opener: Callable[..., Any] = _default_opener,
) -> dict[str, Any]:
    """Read one job's current state and, once complete, its output.

    ``GET /v2/<endpoint_id>/status/<job_id>`` returns ``status`` plus optional
    ``output``, ``delayTime`` and ``executionTime``, documented at
    https://docs.runpod.io/serverless/endpoints/operation-reference#/status
    Async results are retained 30 minutes after completion, same source.
    """
    response = call(
        queue_url(endpoint_id, "status", validate_job_id(job_id)),
        credential=credential,
        timeout_seconds=timeout_seconds,
        opener=opener,
    )
    status = response.get("status")
    if not isinstance(status, str) or not status.strip():
        raise RunPodClientError(
            "a RunPod status response carried no status field, so the job state "
            "is unknown and must not be guessed"
        )
    return response


def cancel_job(
    endpoint_id: str,
    job_id: str,
    *,
    credential: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    opener: Callable[..., Any] = _default_opener,
) -> dict[str, Any]:
    """Stop a queued or in-progress job to cap spend.

    ``POST /v2/<endpoint_id>/cancel/<job_id>`` removes a queued job and stops
    an in-progress one, documented at
    https://docs.runpod.io/serverless/endpoints/operation-reference#/cancel
    """
    return call(
        queue_url(endpoint_id, "cancel", validate_job_id(job_id)),
        credential=credential,
        method="POST",
        timeout_seconds=timeout_seconds,
        opener=opener,
    )


def endpoint_health(
    endpoint_id: str,
    *,
    credential: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    opener: Callable[..., Any] = _default_opener,
) -> dict[str, Any]:
    """Read worker and queue counts for one endpoint without queuing a job.

    ``GET /v2/<endpoint_id>/health`` returns ``jobs`` and ``workers`` objects,
    documented at
    https://docs.runpod.io/serverless/endpoints/operation-reference#/health
    """
    return call(
        queue_url(endpoint_id, "health"),
        credential=credential,
        timeout_seconds=timeout_seconds,
        opener=opener,
    )


def read_account(
    credential: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    opener: Callable[..., Any] = _default_opener,
) -> dict[str, Any]:
    """Prove account access with one read that creates and bills nothing.

    ``GET /v2/serverless`` returns every serverless endpoint owned by the
    authenticated user as ``endpoints``, each carrying ``id`` and ``name``,
    documented at
    https://docs.runpod.io/api-reference-v2/serverless/list-serverless-endpoints
    A missing, expired, or unauthorized token answers 401 or 403 there, so a
    200 is the affirmative signal the authorization preflight looks for.

    The returned ``account`` object is Binder's own summary, not a RunPod
    response field. RunPod publishes no account-identity read in REST v2, so
    the endpoint inventory is the evidence available.
    """
    response = call(
        rest_url("serverless"),
        credential=credential,
        timeout_seconds=timeout_seconds,
        opener=opener,
    )
    endpoints = response.get("endpoints")
    if not isinstance(endpoints, list):
        raise RunPodClientError(
            "the RunPod serverless listing carried no endpoints array, so account "
            "access was not measured"
        )
    return {
        "account": {
            "provider": "runpod",
            "access": "read",
            "endpoint_count": len(endpoints),
            "endpoint_ids": [
                item.get("id")
                for item in endpoints
                if isinstance(item, Mapping) and isinstance(item.get("id"), str)
            ],
        },
        "endpoints": endpoints,
    }


def lifecycle_state(status: object) -> str | None:
    """Translate a RunPod job status into the lifecycle's own vocabulary."""
    return LIFECYCLE_STATES.get(str(status or "").strip().upper())


class RunPodJob:
    """One submitted job, polled until it reaches a terminal RunPod state."""

    def __init__(
        self,
        endpoint_id: str,
        job_id: str,
        *,
        credential: str,
        opener: Callable[..., Any] = _default_opener,
        sleep: Callable[[float], Any] = time.sleep,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        run_timeout_s: int | None = None,
    ) -> None:
        self.endpoint_id = validate_endpoint_id(endpoint_id)
        self.job_id = validate_job_id(job_id)
        self.id = self.job_id
        self.run_timeout_s = run_timeout_s
        # Set once a terminal status has actually been read, so cleanup can
        # tell "already finished" from "still billing".
        self.terminal_status: str | None = None
        self._credential = credential
        self._opener = opener
        self._sleep = sleep
        self._poll_seconds = float(poll_seconds)
        self._timeout_seconds = timeout_seconds

    def status(self) -> dict[str, Any]:
        return job_status(
            self.endpoint_id,
            self.job_id,
            credential=self._credential,
            timeout_seconds=self._timeout_seconds,
            opener=self._opener,
        )

    def cancel(self) -> dict[str, Any]:
        return cancel_job(
            self.endpoint_id,
            self.job_id,
            credential=self._credential,
            timeout_seconds=self._timeout_seconds,
            opener=self._opener,
        )

    def result(self, *, max_polls: int | None = None) -> dict[str, Any]:
        """Poll to a terminal state, then answer in the lifecycle's shape.

        ``exit_code`` is read from the handler's own ``output``. That key is
        Binder's worker contract and is unverified; see ``WORKER_OUTPUT_KEYS``.
        ``usage`` is omitted on purpose. ``/status`` reports ``delayTime`` and
        ``executionTime`` in milliseconds and no money, so a dollar figure here
        would be a guess. The lifecycle records ``pending-provider-usage`` when
        usage is absent, which is the truthful record.
        """
        polls = 0
        while True:
            response = self.status()
            status = str(response.get("status", "")).strip().upper()
            if status in TERMINAL_STATES:
                return self._terminal(response, status)
            if status not in PENDING_STATES:
                raise RunPodClientError(
                    f"RunPod reported job status {status!r}, which is neither "
                    "pending nor terminal in the documented set; the job state "
                    "must not be guessed"
                )
            polls += 1
            if max_polls is not None and polls >= max_polls:
                raise RunPodClientError(
                    f"the RunPod job is still {status} after {polls} polls; it was "
                    "not abandoned and can be resumed from its recorded job id"
                )
            self._sleep(self._poll_seconds)

    def _terminal(self, response: Mapping[str, Any], status: str) -> dict[str, Any]:
        # A completed job whose handler reported no integer exit code leaves
        # exit_code None. Substituting 0 here would forge the settlement gate.
        self.terminal_status = status
        output = response.get("output")
        exit_code: int | None = None
        if isinstance(output, Mapping):
            candidate = output.get("exit_code")
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                exit_code = candidate
        return {
            "job_id": self.job_id,
            "state": lifecycle_state(status),
            "provider_status": status,
            "exit_code": exit_code,
            "output": output,
            "delay_time_ms": response.get("delayTime"),
            "execution_time_ms": response.get("executionTime"),
            # TODO(settled-usage): RunPod reports no cost on /status. The
            # settled figure lives in GET /v2/billing/serverless, which is
            # time-bucketed per endpoint rather than per job, so it cannot be
            # attributed to one job id. Reconcile from that report.
            "usage": None,
        }


class RunPodHandle:
    """Bind one Serverless endpoint to the host protocol the lifecycle drives."""

    def __init__(
        self,
        endpoint_id: str,
        *,
        credential: str,
        opener: Callable[..., Any] = _default_opener,
        sleep: Callable[[float], Any] = time.sleep,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.endpoint_id = validate_endpoint_id(endpoint_id)
        self._credential = credential
        self._opener = opener
        self._sleep = sleep
        self._poll_seconds = poll_seconds
        self._timeout_seconds = timeout_seconds
        self._jobs: list[RunPodJob] = []

    def submit_job(
        self,
        *,
        intent: str,
        command: str,
        inputs: Sequence[Mapping[str, str]] = (),
        outputs: Sequence[Any] = (),
        run_timeout_s: int,
    ) -> RunPodJob:
        """Queue the stage command as one asynchronous job.

        The five keys placed in ``input`` are Binder's worker contract, listed
        in ``WORKER_INPUT_KEYS``. RunPod passes ``input`` through untouched, so
        a worker image that does not implement them will queue the job and then
        fail it.
        """
        response = submit_job(
            self.endpoint_id,
            {
                "intent": str(intent),
                "command": str(command),
                "inputs": [dict(item) for item in inputs],
                "outputs": list(outputs),
                "run_timeout_s": int(run_timeout_s),
            },
            credential=self._credential,
            timeout_seconds=self._timeout_seconds,
            opener=self._opener,
        )
        return self._job(str(response["id"]).strip(), run_timeout_s=int(run_timeout_s))

    def attach_job(self, job_id: str) -> RunPodJob:
        """Reattach to a recorded job id. Never resubmits."""
        return self._job(job_id)

    def _job(self, job_id: str, *, run_timeout_s: int | None = None) -> RunPodJob:
        job = RunPodJob(
            self.endpoint_id,
            job_id,
            credential=self._credential,
            opener=self._opener,
            sleep=self._sleep,
            poll_seconds=self._poll_seconds,
            timeout_seconds=self._timeout_seconds,
            run_timeout_s=run_timeout_s,
        )
        self._jobs.append(job)
        return job

    def close(self, *, intent: str) -> dict[str, Any]:
        """Cancel every still-billing job this handle submitted.

        A queue-based endpoint bills per job, and an endpoint outlives a
        handle, so cleanup cancels jobs and never deletes the endpoint. A job
        whose terminal status this handle already read is skipped: it is not
        billing, and the docs describe ``/cancel`` only for a queued or
        in-progress job, so calling it on a finished one could turn a clean
        cleanup into a reported failure that holds the settlement open.
        """
        errors: list[str] = []
        cancelled: list[str] = []
        already_terminal: list[str] = []
        for job in self._jobs:
            if job.terminal_status is not None:
                already_terminal.append(job.job_id)
                continue
            try:
                job.cancel()
            except RunPodClientError as exc:
                errors.append(f"{job.job_id}: {exc}")
            else:
                cancelled.append(job.job_id)
        if errors:
            return {"ok": False, "intent": intent, "errors": errors}
        return {
            "ok": True,
            "intent": intent,
            "cancelled": cancelled,
            "already_terminal": already_terminal,
        }


def create_handle(
    *,
    credential_env: str = DEFAULT_CREDENTIAL_ENV,
    environ: Mapping[str, str] | None = None,
    opener: Callable[..., Any] = _default_opener,
    sleep: Callable[[float], Any] = time.sleep,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Callable[[Mapping[str, Any]], RunPodHandle]:
    """Return the handle factory ``runpod_platform.ClientHost`` is missing.

    The factory reads its credential once per handle, from the named variable
    only, so a credential never reaches an argv or a submission record. The
    endpoint id comes from the adapter's provider parameters under
    ``endpoint_id``, which is also accepted as ``endpoint`` because a profile
    written against the REST v2 vocabulary spells it that way.
    """

    def factory(provider_params: Mapping[str, Any]) -> RunPodHandle:
        endpoint_id = provider_params.get("endpoint_id") or provider_params.get(
            "endpoint"
        )
        if not endpoint_id:
            raise RunPodClientError(
                "the RunPod provider parameters name no serverless endpoint. Set "
                "endpoint_id in the adapter's runpod parameters."
            )
        return RunPodHandle(
            validate_endpoint_id(endpoint_id),
            credential=read_credential(credential_env, environ=environ),
            opener=opener,
            sleep=sleep,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
        )

    return factory


def client_host(**kwargs: Any) -> Any:
    """Return a host binding the RunPod lifecycle accepts as ``host``.

    This is the one line of wiring the route was missing. ``runpod_platform``
    itself needs no change: it already accepts any callable here.
    """
    from claude_binder.adapters import runpod_platform

    return runpod_platform.ClientHost(create_handle(**kwargs))


def account_reader(
    *,
    credential_env: str = DEFAULT_CREDENTIAL_ENV,
    environ: Mapping[str, str] | None = None,
    opener: Callable[..., Any] = _default_opener,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Callable[[Mapping[str, str]], Any]:
    """Return the read-only account reader the authorization preflight wants."""
    from claude_binder.adapters import runpod_platform

    def read(credential: str) -> dict[str, Any]:
        return read_account(credential, timeout_seconds=timeout_seconds, opener=opener)

    return runpod_platform.environment_account_reader(
        read,
        environ=environ,
        route=runpod_platform.RUNPOD_ROUTE,
        api_key_environment_key=credential_environment_name(credential_env),
    )
