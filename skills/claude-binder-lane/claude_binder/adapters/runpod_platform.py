"""Run one binder stage through a user-owned RunPod host binding.

The adapter deliberately targets a small host protocol rather than a RunPod
deployment identifier.  Claude Science can supply that protocol through its
compute host; a workstation can supply an environment-backed client wrapper.
Neither path needs another Codex skill or plugin at runtime.

The lifecycle is explicit: submit writes durable facts, resume attaches to the
recorded job id, settlement validates the harvested receipt, and cleanup stops
billable compute promptly even when artifact recovery remains incomplete.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import fcntl


ROUTE_KEY = "CLAUDE_BINDER_EXECUTION_ROUTE"


@dataclass(frozen=True)
class NativeCloudRoute:
    """Names that distinguish one provider while sharing safety semantics."""

    provider_id: str
    display_name: str
    route_value: str
    provider_params_key: str
    api_key_environment_key: str | None = None


RUNPOD_ROUTE = NativeCloudRoute(
    provider_id="runpod",
    display_name="RunPod",
    route_value="runpod-platform",
    provider_params_key="runpod",
    api_key_environment_key="RUNPOD_API_KEY",
)
LAMBDA_ROUTE = NativeCloudRoute(
    provider_id="lambda",
    display_name="Lambda Cloud",
    route_value="lambda-platform",
    provider_params_key="lambda",
)

PROVIDER_ID = RUNPOD_ROUTE.provider_id
ROUTE_VALUE = RUNPOD_ROUTE.route_value
PROVIDER_PARAMS_KEY = RUNPOD_ROUTE.provider_params_key
API_KEY_ENVIRONMENT_KEY = str(RUNPOD_ROUTE.api_key_environment_key)


class RunPodPlatformError(ValueError):
    """The RunPod lifecycle cannot safely advance."""


@contextmanager
def dispatch_lock(run_root: Path):
    """Serialize cap, submission, binding, and settlement decisions.

    The lock file may remain after a crash, but the advisory kernel lock does
    not. A later process can therefore recover without deleting a stale marker.
    """
    path = run_root.resolve() / "artifacts" / "runpod-dispatch.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ClientHost:
    """Adapt an environment-backed client's handle factory to the host protocol."""

    def __init__(
        self,
        create_handle: Callable[[Mapping[str, Any]], Any],
        *,
        route: NativeCloudRoute = RUNPOD_ROUTE,
    ) -> None:
        if not callable(create_handle):
            raise TypeError("create_handle must be callable")
        self._create_handle = create_handle
        self._route = route
        self.compute = self

    def create(self, provider: str, *, provider_params: Mapping[str, Any]) -> Any:
        if provider != self._route.provider_id:
            raise RunPodPlatformError(f"client host cannot create provider {provider!r}")
        return self._create_handle(dict(provider_params))


def environment_account_reader(
    read_account: Callable[[str], Any],
    *,
    environ: Mapping[str, str] | None = None,
    route: NativeCloudRoute = RUNPOD_ROUTE,
    api_key_environment_key: str | None = None,
) -> Callable[[Mapping[str, str]], Any]:
    """Bind a read-only client call to one named environment credential."""
    if not callable(read_account):
        raise TypeError("read_account must be callable")
    source = os.environ if environ is None else environ
    credential_key = api_key_environment_key or route.api_key_environment_key
    if not credential_key:
        raise RunPodPlatformError(
            f"{route.display_name} account binding requires an explicit credential environment key"
        )

    def reader(request: Mapping[str, str]) -> Any:
        if request.get("provider") != route.provider_id or request.get("mode") != "read":
            raise RunPodPlatformError(
                f"environment account reader accepts {route.display_name} read mode only"
            )
        credential = str(source.get(credential_key, "") or "").strip()
        if not credential:
            raise RunPodPlatformError(
                f"{credential_key} is unavailable to the read-only account client"
            )
        return read_account(credential)

    return reader


@dataclass(frozen=True)
class Submission:
    """Provider facts that must be persisted immediately after submission."""

    job_id: str
    provider: str
    provider_params: dict[str, Any]
    command: str
    timeout_seconds: int
    submission_id: str | None
    handle: Any = field(repr=False, compare=False)
    job: Any = field(repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "provider": self.provider,
            "provider_params": self.provider_params,
            "command": self.command,
            "run_timeout_s": self.timeout_seconds,
            "submission_id": self.submission_id,
            "state": "submitted",
        }


@dataclass(frozen=True)
class Settlement:
    """Terminal result, artifact receipt, and cleanup outcome."""

    job_id: str
    provider: str
    state: str | None
    exit_code: int | None
    receipt_path: str | None
    receipt_validation: dict[str, Any] | None
    cleanup: dict[str, Any]
    held_open: bool
    usage: dict[str, Any] | None
    financial_status: str
    errors: tuple[str, ...]
    receipt: dict[str, Any] | None = field(repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.held_open

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "provider": self.provider,
            "state": self.state,
            "exit_code": self.exit_code,
            "receipt_path": self.receipt_path,
            "receipt_validation": self.receipt_validation,
            "cleanup": self.cleanup,
            "held_open": self.held_open,
            "usage": self.usage,
            "financial_status": self.financial_status,
            "errors": list(self.errors),
            "ok": self.ok,
        }


def uses_runpod_platform(adapter: Mapping[str, Any]) -> bool:
    """Return whether an adapter selected the native RunPod host route."""
    return uses_native_platform(adapter, route=RUNPOD_ROUTE)


def uses_native_platform(
    adapter: Mapping[str, Any], *, route: NativeCloudRoute
) -> bool:
    environment = adapter.get("environment")
    return (
        isinstance(environment, Mapping)
        and environment.get(ROUTE_KEY) == route.route_value
    )


def provider_params(
    adapter: Mapping[str, Any], *, route: NativeCloudRoute = RUNPOD_ROUTE
) -> dict[str, Any]:
    """Return user-declared RunPod parameters without inventing resource IDs.

    Provider-specific values live in ``adapter.runpod``.  Generic resource
    declarations are copied when present, but no endpoint, template, pod, or
    deployment identifier is required or synthesized.
    """
    declared = adapter.get(route.provider_params_key, {})
    if not isinstance(declared, Mapping):
        raise RunPodPlatformError(f"adapter.{route.provider_params_key} must be an object")
    params = dict(declared)
    resources = adapter.get("resources")
    if isinstance(resources, Mapping):
        aliases = {
            "cpu": "cpu",
            "memory_gb": "memory_gb",
            "gpu": "gpu_count",
            "gpu_memory_gb": "gpu_memory_gb",
            "container_image_digest": "image",
        }
        for source, destination in aliases.items():
            value = resources.get(source)
            if value is not None and destination not in params:
                params[destination] = value
    return params


def _compute(host: Any, *, route: NativeCloudRoute = RUNPOD_ROUTE) -> Any:
    compute = getattr(host, "compute", None)
    if compute is None or not callable(getattr(compute, "create", None)):
        raise RunPodPlatformError(
            f"No {route.display_name} host binding is available. Bind the Claude Science compute "
            "host or an environment-backed client before dispatch."
        )
    return compute


def _job_id(job: Any, *, route: NativeCloudRoute = RUNPOD_ROUTE) -> str:
    value = _value(job, "job_id", _value(job, "id"))
    if not isinstance(value, str) or not value.strip():
        raise RunPodPlatformError(f"{route.display_name} submit returned no job identifier")
    return value.strip()


def submit_stage(
    host: Any,
    *,
    stage: Mapping[str, Any],
    adapter: Mapping[str, Any],
    command_argv: Sequence[str],
    timeout_seconds: int,
    inputs: Sequence[Mapping[str, str]] = (),
    outputs: Sequence[Any] = ({"glob": "out/**", "visibility": "featured"},),
    route: NativeCloudRoute = RUNPOD_ROUTE,
) -> Submission:
    """Submit one stage without waiting or hiding the provider job id."""
    if not uses_native_platform(adapter, route=route):
        raise RunPodPlatformError(
            f"adapter {adapter.get('adapter_id')} did not select {route.route_value}"
        )
    if not command_argv:
        raise RunPodPlatformError(f"stage {stage.get('stage_id')} has no command argv")
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or timeout_seconds < 1:
        raise RunPodPlatformError(f"{route.display_name} job timeout must be a positive integer")
    params = provider_params(adapter, route=route)
    handle = _compute(host, route=route).create(route.provider_id, provider_params=params)
    command = shlex.join([str(item) for item in command_argv])
    job = handle.submit_job(
        intent=f"claude-binder stage {stage.get('stage_id')}",
        command=command,
        inputs=[dict(item) for item in inputs],
        outputs=list(outputs),
        run_timeout_s=timeout_seconds,
    )
    return Submission(
        job_id=_job_id(job, route=route),
        provider=route.provider_id,
        provider_params=params,
        command=command,
        timeout_seconds=timeout_seconds,
        submission_id=None,
        handle=handle,
        job=job,
    )


def guarded_submit_stage(
    host: Any,
    *,
    plan: dict[str, Any],
    bundle_root: Path,
    run_root: Path,
    stage: Mapping[str, Any],
    adapter: Mapping[str, Any],
    command_argv: Sequence[str],
    timeout_seconds: int,
    account_reader: Callable[[Mapping[str, str]], Any],
    inputs: Sequence[Mapping[str, str]] = (),
    outputs: Sequence[Any] = ({"glob": "out/**", "visibility": "featured"},),
    route: NativeCloudRoute = RUNPOD_ROUTE,
) -> Submission:
    """Atomically gate authorization and spend before creating a handle.

    The approved maximum is reserved before submission.  If the provider call
    becomes uncertain, the conservative reservation remains in ``spend.jsonl``
    and prevents a blind retry from spending the same allowance twice.
    """
    from claude_binder import lane, provider_authorization

    stage_id = stage.get("stage_id")
    if not isinstance(stage_id, str) or not stage_id:
        raise RunPodPlatformError(f"{route.display_name} dispatch requires a named stage")
    if not uses_native_platform(adapter, route=route):
        raise RunPodPlatformError(
            f"adapter {adapter.get('adapter_id')} did not select {route.route_value}"
        )
    if not command_argv:
        raise RunPodPlatformError(f"stage {stage_id} has no command argv")
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or timeout_seconds < 1:
        raise RunPodPlatformError(f"{route.display_name} job timeout must be a positive integer")
    params = provider_params(adapter, route=route)
    command = shlex.join([str(item) for item in command_argv])
    with dispatch_lock(run_root):
        approval = lane.verify_execution_approval(plan, bundle_root.resolve())
        if approval.get("ok") is not True:
            raise RunPodPlatformError(
                "approval gate refused: " + "; ".join(str(item) for item in approval.get("errors", []))
            )
        authorization = provider_authorization.probe_native_cloud_authorization(
            plan,
            [stage],
            account_reader=account_reader,
            provider_id=route.provider_id,
            provider_name=route.display_name,
            stage_providers={
                route.provider_id: [
                    {
                        "stage_id": stage_id,
                        "adapter_id": str(adapter.get("adapter_id", "<unnamed adapter>")),
                        "source": f"adapter environment.{ROUTE_KEY}",
                    }
                ]
            },
        )
        if authorization.get("status") != provider_authorization.AUTHORIZED:
            raise RunPodPlatformError(
                f"{route.display_name} authorization refused: "
                + "; ".join(str(item) for item in authorization.get("errors", []))
            )
        registers = lane.run_register_paths(run_root.resolve())
        estimates = lane.approved_stage_estimates(approval.get("approval"))
        provider_value = plan.get("provider")
        provider = provider_value if isinstance(provider_value, Mapping) else {}
        budget_value = provider.get("budget")
        budget = budget_value if isinstance(budget_value, Mapping) else {}
        spend = lane.enforce_spend_cap(
            registers["spend"],
            run_fingerprint=str(plan.get("run_fingerprint")),
            budget_maximum=budget.get("maximum_spend_usd"),
            currency=budget.get("currency"),
            stage_estimates=estimates,
            stage_ids=[stage_id],
        )
        if spend.get("ok") is not True:
            raise RunPodPlatformError(
                "spend cap refused: " + "; ".join(str(item) for item in spend.get("errors", []))
            )
        prior_jobs = lane.load_jsonl(registers["jobs"]) if registers["jobs"].is_file() else []
        if any(
            row.get("run_fingerprint") == plan.get("run_fingerprint")
            and row.get("stage_id") == stage_id
            and (
                row.get("state") == "submitted"
                or row.get("record") in {"submission-intent", "submission-uncertain"}
            )
            for row in prior_jobs
        ):
            raise RunPodPlatformError(
                f"stage {stage_id} already has a submitted or uncertain job; bind or attach "
                "to it instead of resubmitting"
            )
        lane.record_estimated_stage_spend(
            registers["spend"],
            run_fingerprint=str(plan.get("run_fingerprint")),
            stage_id=stage_id,
            provider_id=route.provider_id,
            budget_maximum=budget.get("maximum_spend_usd"),
            currency=budget.get("currency"),
            stage_estimates=estimates,
        )
        handle = _compute(host, route=route).create(route.provider_id, provider_params=params)
        submission_id = str(uuid.uuid4())
        intent_row = {
            "record": "submission-intent",
            "submission_id": submission_id,
            "run_fingerprint": str(plan.get("run_fingerprint")),
            "run_id": plan.get("run_id"),
            "stage_id": stage_id,
            "adapter_id": adapter.get("adapter_id"),
            "provider_id": route.provider_id,
            "provider_params": params,
            "command": command,
            "run_timeout_s": timeout_seconds,
            "state": "submitting",
            "intent_recorded_at": lane.utc_now(),
        }
        lane.append_jsonl(registers["jobs"], intent_row)
        try:
            job = handle.submit_job(
                intent=f"claude-binder stage {stage_id}",
                command=command,
                inputs=[dict(item) for item in inputs],
                outputs=list(outputs),
                run_timeout_s=timeout_seconds,
            )
            job_id = _job_id(job, route=route)
        except Exception as exc:
            cleanup = _close(
                handle,
                intent=f"{route.display_name} submission outcome is uncertain; stop compute to cap spend",
                route=route,
            )
            lane.append_jsonl(
                registers["jobs"],
                {
                    **intent_row,
                    "record": "submission-uncertain",
                    "state": "uncertain",
                    "submit_error": f"{type(exc).__name__}: {exc}",
                    "cleanup": cleanup,
                    "uncertain_at": lane.utc_now(),
                },
            )
            raise RunPodPlatformError(
                f"{route.display_name} submission outcome is uncertain for submission_id {submission_id}; "
                "inspect the provider ledger, then bind the returned job id"
            ) from exc
        row = {
            **intent_row,
            "record": "submission",
            "job_id": job_id,
            "state": "submitted",
            "submitted_at": lane.utc_now(),
        }
        lane.append_jsonl(registers["jobs"], row)
        return Submission(
            job_id=job_id,
            provider=route.provider_id,
            provider_params=params,
            command=command,
            timeout_seconds=timeout_seconds,
            submission_id=submission_id,
            handle=handle,
            job=job,
        )


def attach_stage(
    host: Any,
    record: Mapping[str, Any],
    *,
    route: NativeCloudRoute = RUNPOD_ROUTE,
) -> Submission:
    """Reattach to the recorded provider job; never resubmit on resume."""
    job_id = record.get("job_id")
    params = record.get("provider_params")
    if not isinstance(job_id, str) or not job_id.strip():
        raise RunPodPlatformError(f"a resumed {route.display_name} submission needs its recorded job_id")
    if not isinstance(params, Mapping):
        raise RunPodPlatformError(f"a resumed {route.display_name} submission needs recorded provider_params")
    recorded_provider = record.get("provider_id") or record.get("provider")
    if recorded_provider not in (None, route.provider_id):
        raise RunPodPlatformError(
            f"recorded provider {recorded_provider!r} does not match {route.provider_id!r}"
        )
    handle = _compute(host, route=route).create(route.provider_id, provider_params=dict(params))
    attach = getattr(handle, "attach_job", None)
    if not callable(attach):
        raise RunPodPlatformError(f"the {route.display_name} host binding cannot attach to an existing job")
    job = attach(job_id.strip())
    command = record.get("command")
    timeout = record.get("run_timeout_s")
    return Submission(
        job_id=job_id.strip(),
        provider=route.provider_id,
        provider_params=dict(params),
        command=command if isinstance(command, str) else "",
        timeout_seconds=timeout if isinstance(timeout, int) and not isinstance(timeout, bool) else 0,
        submission_id=(
            record.get("submission_id")
            if isinstance(record.get("submission_id"), str)
            else None
        ),
        handle=handle,
        job=job,
    )


def bind_uncertain_submission(
    run_root: Path,
    *,
    plan: Mapping[str, Any],
    submission_id: str,
    job_id: str,
    route: NativeCloudRoute = RUNPOD_ROUTE,
) -> dict[str, Any]:
    """Bind a provider-ledger job ID to one durable uncertain submit intent."""
    from claude_binder import lane

    if not isinstance(submission_id, str) or not submission_id.strip():
        raise RunPodPlatformError("submission_id must be a non-empty string")
    if not isinstance(job_id, str) or not job_id.strip():
        raise RunPodPlatformError("job_id must be a non-empty string")
    fingerprint = plan.get("run_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise RunPodPlatformError("the plan has no run_fingerprint")
    register = lane.run_register_paths(run_root.resolve())["jobs"]
    with dispatch_lock(run_root):
        rows = lane.load_jsonl(register) if register.is_file() else []
        if any(row.get("job_id") == job_id.strip() for row in rows):
            raise RunPodPlatformError(f"job_id is already registered: {job_id.strip()}")
        if any(
            row.get("submission_id") == submission_id.strip() and row.get("job_id")
            for row in rows
        ):
            raise RunPodPlatformError(
                f"submission_id is already bound: {submission_id.strip()}"
            )
        matches = [
            row
            for row in rows
            if row.get("record") in {"submission-intent", "submission-uncertain"}
            and row.get("submission_id") == submission_id.strip()
            and row.get("run_fingerprint") == fingerprint
        ]
        if not matches:
            raise RunPodPlatformError(
                f"no uncertain {route.display_name} submission matches this run and submission_id"
            )
        intent = matches[-1]
        if intent.get("provider_id") != route.provider_id:
            raise RunPodPlatformError(
                f"uncertain submission belongs to provider {intent.get('provider_id')!r}"
            )
        row = {
            key: intent.get(key)
            for key in (
                "submission_id",
                "run_fingerprint",
                "run_id",
                "stage_id",
                "adapter_id",
                "provider_id",
                "provider_params",
                "command",
                "run_timeout_s",
            )
        }
        row.update(
            {
                "record": "submission",
                "job_id": job_id.strip(),
                "state": "submitted",
                "recovered_from_provider_ledger": True,
                "submitted_at": lane.utc_now(),
            }
        )
        lane.append_jsonl(register, row)
        return row


def bind_and_attach_uncertain_submission(
    host: Any,
    run_root: Path,
    *,
    plan: Mapping[str, Any],
    submission_id: str,
    job_id: str,
    route: NativeCloudRoute = RUNPOD_ROUTE,
) -> Submission:
    """Bind an ambiguous provider acceptance and attach without resubmitting."""
    row = bind_uncertain_submission(
        run_root,
        plan=plan,
        submission_id=submission_id,
        job_id=job_id,
        route=route,
    )
    return attach_stage(host, row, route=route)


def settle_stage(
    submission: Submission,
    *,
    harvest_root: Path,
    receipt_validator: Callable[[dict[str, Any], Path], Mapping[str, Any]] | None = None,
    route: NativeCloudRoute = RUNPOD_ROUTE,
) -> Settlement:
    """Read a terminal result, validate its receipt, then clean up safely.

    Missing or invalid harvested evidence blocks success but still terminates
    the compute handle promptly. Durable volume/object references remain in
    the provider parameters and job ledger for recovery; receipt safety must
    not become an unbounded billing leak. Provider usage is returned verbatim
    for the caller's spend ledger; it is never guessed from runtime.
    """
    if submission.provider != route.provider_id:
        raise RunPodPlatformError(
            f"submission provider {submission.provider!r} does not match {route.provider_id!r}"
        )
    errors: list[str] = []
    try:
        result = submission.job.result()
    except Exception as exc:
        return _held(
            submission,
            f"{route.display_name} terminal result is unavailable: {type(exc).__name__}: {exc}",
            route=route,
        )
    state = _text(_value(result, "state"))
    exit_code = _integer(_value(result, "exit_code"))
    usage_value = _value(result, "usage", _value(result, "settled_usage"))
    usage = dict(usage_value) if isinstance(usage_value, Mapping) else None
    if state != "succeeded" or exit_code != 0:
        errors.append(f"{route.display_name} job ended state={state!r} exit_code={exit_code!r}")

    root = harvest_root.resolve()
    receipt_path = root / "receipt.json"
    if not root.is_dir() or not receipt_path.is_file():
        return _held(
            submission,
            f"{route.display_name} harvest has no receipt at {receipt_path}; compute cleanup proceeds while artifact recovery remains incomplete",
            state=state,
            exit_code=exit_code,
            usage=usage,
            route=route,
        )
    try:
        receipt_value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _held(
            submission,
            f"{route.display_name} receipt is unreadable: {type(exc).__name__}: {exc}",
            state=state,
            exit_code=exit_code,
            usage=usage,
            receipt_path=receipt_path,
            route=route,
        )
    if not isinstance(receipt_value, dict):
        return _held(
            submission,
            f"{route.display_name} receipt must be a JSON object",
            state=state,
            exit_code=exit_code,
            usage=usage,
            receipt_path=receipt_path,
            route=route,
        )
    validation: dict[str, Any] | None = None
    if receipt_validator is not None:
        try:
            validation = dict(receipt_validator(receipt_value, receipt_path))
        except Exception as exc:
            return _held(
                submission,
                f"{route.display_name} receipt validation failed: {type(exc).__name__}: {exc}",
                state=state,
                exit_code=exit_code,
                usage=usage,
                receipt_path=receipt_path,
                receipt=receipt_value,
                route=route,
            )
        if validation.get("ok") is not True:
            return _held(
                submission,
                f"{route.display_name} receipt validation refused the harvested artifacts",
                state=state,
                exit_code=exit_code,
                usage=usage,
                receipt_path=receipt_path,
                receipt=receipt_value,
                validation=validation,
                route=route,
            )
    cleanup = _close(
        submission.handle,
        intent=f"{route.display_name} job is terminal and harvested artifacts were inspected; stop compute",
        route=route,
    )
    if cleanup.get("ok") is not True:
        errors.extend(str(item) for item in cleanup.get("errors", []))
    return Settlement(
        job_id=submission.job_id,
        provider=submission.provider,
        state=state,
        exit_code=exit_code,
        receipt_path=str(receipt_path),
        receipt_validation=validation,
        cleanup=cleanup,
        held_open=cleanup.get("ok") is not True,
        usage=usage,
        financial_status=(
            "provider-usage-reported" if usage is not None else "pending-provider-usage"
        ),
        errors=tuple(errors),
        receipt=receipt_value,
    )


def settle_and_record_stage(
    submission: Submission,
    *,
    plan: Mapping[str, Any],
    run_root: Path,
    stage_id: str,
    harvest_root: Path,
    receipt_validator: Callable[[dict[str, Any], Path], Mapping[str, Any]] | None = None,
    route: NativeCloudRoute = RUNPOD_ROUTE,
) -> Settlement:
    """Settle artifacts, reconcile provider-reported usage, and close the job row."""
    from claude_binder import lane

    settlement = settle_stage(
        submission,
        harvest_root=harvest_root,
        receipt_validator=receipt_validator,
        route=route,
    )
    if not settlement.ok:
        return settlement
    registers = lane.run_register_paths(run_root.resolve())
    usage = settlement.usage
    financial_status = "pending-provider-usage"
    if usage is not None:
        amount = usage.get("amount")
        currency = usage.get("currency")
        source = usage.get("amount_source") or usage.get("source")
        if (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(float(amount))
            or float(amount) < 0
            or currency != "USD"
            or not isinstance(source, str)
            or not source.strip()
        ):
            financial_status = "pending-invalid-provider-usage"
        else:
            rows = lane.load_jsonl(registers["spend"]) if registers["spend"].is_file() else []
            stage_rows = [
                row
                for row in rows
                if row.get("run_fingerprint") == plan.get("run_fingerprint")
                and row.get("stage_id") == stage_id
            ]
            estimated = sum(
                float(row.get("amount") or 0.0)
                for row in stage_rows
                if row.get("event") == "charge-estimate"
            )
            reconciled = sum(
                float(row.get("details", {}).get("reconciled_estimate_amount", 0.0))
                for row in stage_rows
                if row.get("event") == "charge"
                and isinstance(row.get("details"), Mapping)
                and row["details"].get("reconciliation") is True
            )
            pending_estimate = estimated - reconciled
            if pending_estimate <= 0:
                financial_status = "pending-missing-dispatch-reservation"
            else:
                budget_value = plan.get("provider")
                provider = budget_value if isinstance(budget_value, Mapping) else {}
                budget_value = provider.get("budget")
                budget = budget_value if isinstance(budget_value, Mapping) else {}
                lane.record_spend(
                    registers["spend"],
                    run_fingerprint=str(plan.get("run_fingerprint")),
                    amount=float(amount) - pending_estimate,
                    currency="USD",
                    amount_source=str(source),
                    event="charge",
                    stage_id=stage_id,
                    job_id=submission.job_id,
                    provider_id=route.provider_id,
                    budget_maximum=budget.get("maximum_spend_usd"),
                    details={
                        "estimated": False,
                        "reconciliation": True,
                        "reconciled_estimate_amount": pending_estimate,
                        "settled_amount": float(amount),
                        "difference": float(amount) - pending_estimate,
                        "settled_amount_source": str(source),
                    },
                    note=f"provider-reported {route.display_name} usage replaced the dispatch reservation",
                )
                financial_status = "reconciled"
    lane.record_job_event(
        registers["jobs"],
        run_fingerprint=str(plan.get("run_fingerprint")),
        stage_id=stage_id,
        state="completed",
        job_id=submission.job_id,
        provider_id=route.provider_id,
        message=(
            f"{route.display_name} receipt validated and artifacts harvested; financial reconciliation "
            + ("completed" if financial_status == "reconciled" else f"pending ({financial_status})")
        ),
    )
    return replace(settlement, financial_status=financial_status)


def _held(
    submission: Submission,
    error: str,
    *,
    state: str | None = None,
    exit_code: int | None = None,
    usage: dict[str, Any] | None = None,
    receipt_path: Path | None = None,
    receipt: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    route: NativeCloudRoute = RUNPOD_ROUTE,
) -> Settlement:
    cleanup = _close(
        submission.handle,
        intent=f"{route.display_name} harvest or receipt validation is incomplete; stop compute to cap spend",
        route=route,
    )
    cleanup_errors = tuple(str(item) for item in cleanup.get("errors", []))
    return Settlement(
        job_id=submission.job_id,
        provider=submission.provider,
        state=state,
        exit_code=exit_code,
        receipt_path=str(receipt_path) if receipt_path is not None else None,
        receipt_validation=validation,
        cleanup=cleanup,
        held_open=cleanup.get("ok") is not True,
        usage=usage,
        financial_status=(
            "provider-usage-reported" if usage is not None else "pending-provider-usage"
        ),
        errors=(error, *cleanup_errors),
        receipt=receipt,
    )


def _close(
    handle: Any,
    *,
    intent: str,
    route: NativeCloudRoute = RUNPOD_ROUTE,
) -> dict[str, Any]:
    close = getattr(handle, "close", None)
    if not callable(close):
        return {
            "ok": False,
            "errors": [f"the {route.display_name} host binding has no cleanup operation"],
        }
    try:
        value = close(intent=intent)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"{route.display_name} cleanup failed: {type(exc).__name__}: {exc}"],
        }
    if isinstance(value, Mapping) and value.get("ok") is False:
        return {"ok": False, "errors": [str(value.get("error") or value.get("reason") or "cleanup refused")]}
    return {"ok": True, "result": value}


def _value(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
