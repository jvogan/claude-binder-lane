"""Run one binder-lane stage through the Claude Science Modal job surface.

The adapter submits a job, waits for its ``compute_done`` notification, reads
the terminal result, validates the harvested receipt, promotes the workspace
artifacts, and closes the handle promptly. The caller supplies the two kernel-owned
tools that the job surface cannot reconstruct: notification waiting and
workspace promotion.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROUTE_KEY = "CLAUDE_BINDER_EXECUTION_ROUTE"
ROUTE_VALUE = "modal-platform"
ENVIRONMENT_KEY = "CLAUDE_BINDER_MODAL_ENV"
IMAGE_KEY = "CLAUDE_BINDER_MODAL_IMAGE"
GPU_KEY = "CLAUDE_BINDER_MODAL_GPU"
VOLUMES_KEY = "CLAUDE_BINDER_MODAL_VOLUMES"
IMAGE_REF_RE = re.compile(r"^im-[0-9A-Za-z]+$")
RECOVERABLE_OUTPUT_KINDS = frozenset({"result_rejected", "harvest_failed"})
CONTAINER_EXPIRY_KINDS = frozenset({"container_expired", "sandbox_expired"})

# Compatibility hints for the environments named by shipped templates. These
# values are not an allowlist. A user-supplied environment carries GPU_KEY.
#
# **Only ``gpu`` is read.** ``provider_params`` below takes ``.get("gpu")`` at
# the one call site, and that is the only read of this table in the package.
#
# **Every ``egress_domains`` list here is documentation and nothing else.** No
# code in this package passes one to anything that restricts network access, and
# the job surface has no key that would carry one: ``provider_params`` accepts
# image, env, gpu, cpu, memory, volumes and timeout, and the second dispatch
# surface builds the same seven at
# skills/claude-binder-lane/scripts/dispatch_modal.py:561-563. The outbound
# policy a job actually runs under is the one the platform holds against the
# environment named in ``env``, built from that recipe's own
# ``META["egress_domains"]``. These rows mirror that declaration so a reader can
# see what a stage is expected to dial. Editing a row here changes nothing that
# runs. Change the recipe, rebuild the environment, and read the value back off
# the workspace ledger.
ENVIRONMENT_DEFAULTS: dict[str, dict[str, Any]] = {
    "proteomics_rfd_diffdock_gpu": {
        "gpu": "A100",
        "egress_domains": ["dl.fbaipublicfiles.com"],
    },
    "esmfold2_gpu": {"gpu": "A100-80GB", "egress_domains": []},
    "esmfold2_kit_gpu": {
        "gpu": "H100",
        "egress_domains": ["huggingface.co", "*.hf.co"],
    },
    # An Anthropic optimization-kit recipe rather than an upstream model on its
    # own. Both values come
    # from the META block of skills/claude-binder-lane/envs/boltz2_kit_gpu.py.
    # H100 is the card the kit's PINS.json names under "tested_on" and the card
    # the published speed-ups were measured on, so it is the tier the recipe
    # sizes for rather than the smallest that fits.
    "boltz2_kit_gpu": {
        "gpu": "H100",
        "egress_domains": [
            "github.com",
            "codeload.github.com",
            "pypi.org",
            "files.pythonhosted.org",
            "model-gateway.boltz.bio",
            "huggingface.co",
            "*.hf.co",
        ],
    },
    "proteomics_boltz_gpu": {
        "gpu": "A100-80GB",
        "egress_domains": [
            "api.colabfold.com",
            "model-gateway.boltz.bio",
            "huggingface.co",
            "*.hf.co",
        ],
    },
    "proteomics_gpu": {
        "gpu": "A100-80GB",
        "egress_domains": [
            "api.colabfold.com",
            "huggingface.co",
            "*.hf.co",
            "chaiassets.com",
        ],
    },
    "proteomics_openfold_gpu": {
        "gpu": "A100-80GB",
        "egress_domains": ["api.colabfold.com"],
    },
    "proteomics_jax_gpu": {
        "gpu": "A100",
        "egress_domains": ["api.colabfold.com"],
    },
    # This lane ships the recipe itself, so both values come from the META block
    # of skills/claude-binder-lane/envs/genie3_generator_gpu.py. The A100 basis
    # recorded there is the catalog's declared gpu_memory_gb of 24 plus headroom,
    # and no Genie3 memory measurement exists in this package.
    "genie3_generator_gpu": {
        "gpu": "A100",
        "egress_domains": ["huggingface.co", "*.hf.co"],
    },
    # The second recipe this lane ships. Both values come from the META block of
    # skills/claude-binder-lane/envs/rfdiffusion_generator_gpu.py: "gpu_default":
    # "A100" at line 34, and "files.ipd.uw.edu" at line 44, which is the host in
    # _WEIGHTS_URL on line 50. That recipe is offline at job time, so the host is
    # declared for HYDRATE against a cold Volume. The row was missing until
    # 2026-09-11, which left any adapter naming this environment without GPU_KEY
    # raising in provider_params below.
    "rfdiffusion_generator_gpu": {
        "gpu": "A100",
        "egress_domains": ["files.ipd.uw.edu"],
    },
}


class ModalPlatformError(ValueError):
    """A Modal job cannot be submitted until the caller fixes its configuration."""


@dataclass(frozen=True)
class Submission:
    """The durable facts emitted after a job-surface submission."""

    job_id: str
    provider: str
    provider_params: dict[str, Any]
    command: str
    timeout_seconds: int
    job_notes: Any
    handle: Any = field(repr=False, compare=False)
    job: Any = field(repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "provider": self.provider,
            "provider_params": self.provider_params,
            "command": self.command,
            "run_timeout_s": self.timeout_seconds,
            "job_notes": _record_value(self.job_notes),
        }


@dataclass(frozen=True)
class Completion:
    """The durable facts from a terminal Modal completion attempt."""

    submission: Submission
    state: str | None
    exit_code: int | None
    error_kind: str | None
    notes: Any
    stdout_tail: str | None
    stderr_tail: str | None
    output_files: tuple[str, ...]
    receipt_path: str | None
    receipt_validation: dict[str, Any] | None
    promotion: dict[str, Any] | None
    close: dict[str, Any] | None
    held_open: bool
    errors: tuple[str, ...]
    receipt: dict[str, Any] | None = field(repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.held_open

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.submission.job_id,
            "provider": self.submission.provider,
            "state": self.state,
            "exit_code": self.exit_code,
            "error_kind": self.error_kind,
            "notes": _record_value(self.notes),
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "output_files": list(self.output_files),
            "receipt_path": self.receipt_path,
            "receipt_validation": self.receipt_validation,
            "promotion": self.promotion,
            "close": self.close,
            "held_open": self.held_open,
            "errors": list(self.errors),
            "ok": self.ok,
        }


def uses_modal_platform(adapter: Mapping[str, Any]) -> bool:
    """Return whether this adapter selected the user-owned Modal route."""
    environment = adapter.get("environment")
    return isinstance(environment, Mapping) and environment.get(ROUTE_KEY) == ROUTE_VALUE


def _environment(adapter: Mapping[str, Any]) -> Mapping[str, Any]:
    environment = adapter.get("environment")
    if not isinstance(environment, Mapping):
        raise ModalPlatformError(f"adapter {adapter.get('adapter_id')} has no environment object")
    return environment


def _volumes(environment: Mapping[str, Any]) -> dict[str, str]:
    raw = environment.get(VOLUMES_KEY, "{}")
    if not isinstance(raw, str):
        raise ModalPlatformError(f"{VOLUMES_KEY} must be JSON object text")
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModalPlatformError(f"{VOLUMES_KEY} is invalid JSON: {exc.msg}") from exc
    if not isinstance(values, dict) or any(
        not isinstance(mount, str)
        or not mount.startswith("/")
        or not isinstance(name, str)
        or not name
        for mount, name in values.items()
    ):
        raise ModalPlatformError(f"{VOLUMES_KEY} must map absolute mount paths to volume names")
    return dict(values)


def provider_params(
    adapter: Mapping[str, Any], *, timeout_seconds: int
) -> dict[str, Any]:
    """Build the job-surface parameters for one adapter without creating a job."""
    environment = _environment(adapter)
    name = environment.get(ENVIRONMENT_KEY)
    if not isinstance(name, str) or not name:
        raise ModalPlatformError(
            f"adapter {adapter.get('adapter_id')} has no Modal environment name"
        )
    gpu = environment.get(GPU_KEY)
    if gpu is None:
        gpu = ENVIRONMENT_DEFAULTS.get(name, {}).get("gpu")
    if not isinstance(gpu, str) or not gpu:
        raise ModalPlatformError(
            f"adapter {adapter.get('adapter_id')} must set {GPU_KEY} for Modal environment {name!r}"
        )
    image = environment.get(IMAGE_KEY)
    if not isinstance(image, str) or IMAGE_REF_RE.fullmatch(image) is None:
        raise ModalPlatformError(
            f"Image reference {image} is invalid for this provider. Fix the reference before dispatching."
        )
    resources = adapter.get("resources")
    if not isinstance(resources, Mapping):
        raise ModalPlatformError(f"adapter {adapter.get('adapter_id')} has no resources block")
    cpu = resources.get("cpu")
    memory_gb = resources.get("memory_gb")
    if not isinstance(cpu, int) or cpu < 1 or not isinstance(memory_gb, int) or memory_gb < 1:
        raise ModalPlatformError(f"adapter {adapter.get('adapter_id')} has invalid CPU or memory resources")
    if timeout_seconds < 1:
        raise ModalPlatformError("Modal job timeout must be positive")
    # The Modal job surface owns a single job clock. Container timeout uses the
    # provider setting unless a user explicitly pins a separate container limit.
    return {
        "image": image,
        "env": name,
        "gpu": gpu,
        "cpu": cpu,
        "memory": memory_gb * 1024,
        "volumes": _volumes(environment),
    }


def submit_stage(
    host: Any,
    *,
    stage: Mapping[str, Any],
    adapter: Mapping[str, Any],
    command_argv: Sequence[str],
    timeout_seconds: int,
    inputs: Sequence[Mapping[str, str]] = (),
    outputs: Sequence[Any] = ({"glob": "out/**", "visibility": "featured"},),
) -> Submission:
    """Create a Modal handle and submit one job without waiting for completion."""
    if not uses_modal_platform(adapter):
        raise ModalPlatformError(f"adapter {adapter.get('adapter_id')} did not select {ROUTE_VALUE}")
    compute = getattr(host, "compute", None)
    if compute is None or not callable(getattr(compute, "create", None)):
        raise ModalPlatformError(
            "No Modal account is connected. Create one, enable billing, then reconnect. Jobs run on your account and your budget pays."
        )
    if not command_argv:
        raise ModalPlatformError(f"stage {stage.get('stage_id')} has no command argv")
    params = provider_params(adapter, timeout_seconds=timeout_seconds)
    handle = compute.create("modal", provider_params=params)
    command = shlex.join(list(command_argv))
    job = handle.submit_job(
        intent=f"claude-binder stage {stage.get('stage_id')}",
        command=command,
        inputs=[dict(item) for item in inputs],
        outputs=list(outputs),
        run_timeout_s=timeout_seconds,
    )
    job_id = getattr(job, "job_id", getattr(job, "id", None))
    if not isinstance(job_id, str) or not job_id:
        raise ModalPlatformError("Modal submit returned no job identifier")
    return Submission(
        job_id=job_id,
        provider="modal",
        provider_params=params,
        command=command,
        timeout_seconds=timeout_seconds,
        job_notes=_result_value(job, "notes"),
        handle=handle,
        job=job,
    )


def complete_stage(
    submission: Submission,
    *,
    workspace: Path,
    wait_for_notification: Callable[[], Mapping[str, Any]] | None,
    promotion_destination: str,
    promote_artifacts: Callable[..., Any] | None,
    receipt_expectation: Mapping[str, Any],
    receipt_validator: Callable[[dict[str, Any], Path], Mapping[str, Any]] | None,
    promotion_paths: Sequence[Path] = (),
) -> Completion:
    """Settle one submitted job after the kernel reports ``compute_done``.

    ``job.result()`` runs only after the matching notification. The host poller
    has harvested the job's ``./out`` directory into ``hpc/<job_id>/`` by that
    point. The adapter promotes harvested material before it closes the handle.
    Receipt or harvest failure blocks scientific success but never deliberately
    keeps potentially billable compute alive. Provider job and output references
    remain in the completion record for a later durable-artifact recovery.
    """
    payload, wait_errors = _matching_notification(
        submission.job_id, wait_for_notification
    )
    if payload is None:
        return _completion_with_errors(submission, wait_errors)

    result: Any = None
    result_error: Exception | None = None
    try:
        result = submission.job.result()
    except Exception as exc:  # Result failures have no exit code by contract.
        result_error = exc

    notified_state = _string_value(payload.get("state"))
    result_state = _string_value(_result_value(result, "state"))
    state = result_state or notified_state
    notified_exit_code = _exit_code(payload.get("exit_code"))
    result_exit_code = _exit_code(_result_value(result, "exit_code"))
    exit_code = result_exit_code if result_exit_code is not None else notified_exit_code
    error_kind = _string_value(payload.get("error_kind")) or _string_value(
        _result_value(result, "kind")
    )
    if result_error is not None:
        error_kind = error_kind or _string_value(getattr(result_error, "kind", None))
    notes = _result_value(result, "notes", payload.get("notes"))
    stdout_tail = _tail_value(_result_value(result, "stdout_tail", payload.get("stdout_tail")))
    stderr_tail = _tail_value(_result_value(result, "stderr_tail", payload.get("stderr_tail")))
    output_files = _output_files(payload.get("output_files"))
    errors = list(wait_errors)
    if result_error is not None:
        errors.append(
            "Modal returned no terminal result: "
            f"{type(result_error).__name__}: {result_error}"
        )
    if notified_state and result_state and notified_state != result_state:
        errors.append(
            "Modal completion state disagrees with job.result(): "
            f"notification={notified_state} result={result_state}"
        )
    if state is None:
        errors.append("Modal completion carried no terminal state")
    if state == "succeeded" and exit_code != 0:
        errors.append(
            "Modal reported a succeeded job without exit_code 0; the terminal receipt is refused"
        )

    incomplete_harvest = error_kind in RECOVERABLE_OUTPUT_KINDS
    harvest_root = workspace.resolve() / "hpc" / submission.job_id
    receipt_path = _harvested_file_path(
        workspace, submission.job_id, output_files, "receipt.json"
    )
    harvest_confirmed = harvest_root.is_dir()
    if incomplete_harvest:
        errors.append(_failure_message(state, exit_code, error_kind))
        promotion = (
            _promote_harvest(
                promote_artifacts,
                destination=promotion_destination,
                paths=[*promotion_paths, harvest_root],
            )
            if harvest_confirmed
            else None
        )
        if promotion is not None and promotion["ok"] is not True:
            errors.extend(str(error) for error in promotion["errors"])
        close = _close_handle(
            submission,
            intent="Modal result harvest is incomplete; stop compute to cap spend",
        )
        if close["state"] != "closed":
            errors.append(str(close["error"]))
        return Completion(
            submission=submission,
            state=state,
            exit_code=exit_code,
            error_kind=error_kind,
            notes=notes,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            output_files=output_files,
            receipt_path=str(receipt_path) if receipt_path is not None else None,
            receipt_validation=None,
            promotion=promotion,
            close=close,
            held_open=close["state"] != "closed",
            errors=tuple(errors),
            receipt=None,
        )
    if not harvest_confirmed:
        errors.append(
            f"Modal harvest is unavailable at {harvest_root}; compute cleanup proceeds while durable artifact recovery remains incomplete"
        )
        close = _close_handle(
            submission,
            intent="Modal terminal harvest is unavailable; stop compute to cap spend",
        )
        if close["state"] != "closed":
            errors.append(str(close["error"]))
        return Completion(
            submission=submission,
            state=state,
            exit_code=exit_code,
            error_kind=error_kind,
            notes=notes,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            output_files=output_files,
            receipt_path=str(receipt_path) if receipt_path is not None else None,
            receipt_validation=None,
            promotion=None,
            close=close,
            held_open=close["state"] != "closed",
            errors=tuple(errors),
            receipt=None,
        )

    receipt: dict[str, Any] | None = None
    receipt_validation: dict[str, Any] | None = None
    if state == "succeeded" and exit_code == 0:
        receipt, receipt_validation, receipt_errors = _validate_harvested_receipt(
            receipt_path,
            receipt_expectation,
            receipt_validator,
        )
        errors.extend(receipt_errors)
    else:
        errors.append(_failure_message(state, exit_code, error_kind))

    promotion = _promote_harvest(
        promote_artifacts,
        destination=promotion_destination,
        paths=[*promotion_paths, harvest_root],
    )
    if promotion["ok"] is not True:
        errors.extend(str(error) for error in promotion["errors"])

    close = _close_handle(submission)
    if close["state"] != "closed":
        errors.append(str(close["error"]))

    return Completion(
        submission=submission,
        state=state,
        exit_code=exit_code,
        error_kind=error_kind,
        notes=notes,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        output_files=output_files,
        receipt_path=str(receipt_path) if receipt_path is not None else None,
        receipt_validation=receipt_validation,
        promotion=promotion,
        close=close,
        held_open=close["state"] != "closed",
        errors=tuple(errors),
        receipt=receipt,
    )


def _matching_notification(
    job_id: str,
    wait_for_notification: Callable[[], Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any] | None, list[str]]:
    if wait_for_notification is None:
        return None, ["Modal completion requires the kernel wait_for_notification tool"]
    while True:
        try:
            wake = wait_for_notification()
        except Exception as exc:
            return None, [
                "wait_for_notification failed before Modal completed: "
                f"{type(exc).__name__}: {exc}"
            ]
        if not isinstance(wake, Mapping):
            return None, ["wait_for_notification returned an invalid response"]
        if wake.get("status") == "error":
            return None, [
                "wait_for_notification ended before this Modal job completed; "
                "compute cleanup will be requested to cap spend"
            ]
        notifications = wake.get("notifications")
        if not isinstance(notifications, Sequence) or isinstance(
            notifications, (str, bytes)
        ):
            return None, ["wait_for_notification returned no notifications"]
        for notification in notifications:
            if not isinstance(notification, Mapping):
                continue
            if notification.get("notification_type") != "compute_done":
                continue
            payload = notification.get("payload")
            if isinstance(payload, Mapping) and payload.get("job_id") == job_id:
                return payload, []


def _completion_with_errors(
    submission: Submission, errors: Sequence[str]
) -> Completion:
    all_errors = list(errors)
    close = _close_handle(
        submission,
        intent="Modal completion wait ended without a terminal receipt; stop compute to cap spend",
    )
    if close["state"] != "closed":
        all_errors.append(str(close["error"]))
    return Completion(
        submission=submission,
        state=None,
        exit_code=None,
        error_kind=None,
        notes=None,
        stdout_tail=None,
        stderr_tail=None,
        output_files=(),
        receipt_path=None,
        receipt_validation=None,
        promotion=None,
        close=close,
        held_open=close["state"] != "closed",
        errors=tuple(all_errors),
        receipt=None,
    )


def _result_value(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _tail_value(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _exit_code(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _output_files(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _harvested_file_path(
    workspace: Path,
    job_id: str,
    output_files: Sequence[str],
    filename: str,
) -> Path:
    root = workspace.resolve()
    for output_file in output_files:
        candidate = Path(output_file)
        if candidate.name != filename:
            continue
        return candidate if candidate.is_absolute() else root / candidate
    return root / "hpc" / job_id / filename


def _validate_harvested_receipt(
    receipt_path: Path,
    expectation: Mapping[str, Any],
    receipt_validator: Callable[[dict[str, Any], Path], Mapping[str, Any]] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    if not receipt_path.is_file():
        return None, None, [f"harvested Modal receipt is missing: {receipt_path}"]
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, None, [
            f"harvested Modal receipt is unreadable: {type(exc).__name__}: {exc}"
        ]
    if not isinstance(value, dict):
        return None, None, ["harvested Modal receipt must contain a JSON object"]
    errors = [
        f"harvested Modal receipt field {name} does not match this stage"
        for name, expected in expectation.items()
        if value.get(name) != expected
    ]
    if value.get("ok") is not True:
        errors.append("harvested Modal receipt is incomplete")
    if receipt_validator is None:
        errors.append("Modal completion requires a receipt validator")
        return value, None, errors
    try:
        validation = receipt_validator(value, receipt_path)
    except Exception as exc:
        return value, None, [
            *errors,
            "harvested Modal receipt validation failed: "
            f"{type(exc).__name__}: {exc}",
        ]
    if not isinstance(validation, Mapping):
        return value, None, [*errors, "Modal receipt validator returned an invalid result"]
    validation_dict = dict(validation)
    if validation_dict.get("ok") is not True:
        validator_errors = validation_dict.get("errors")
        if isinstance(validator_errors, Sequence) and not isinstance(
            validator_errors, (str, bytes)
        ):
            errors.extend(str(error) for error in validator_errors)
        else:
            errors.append("harvested Modal receipt validation failed")
    return value, validation_dict, errors


def _promote_harvest(
    promote_artifacts: Callable[..., Any] | None,
    *,
    destination: str,
    paths: Sequence[Path],
) -> dict[str, Any]:
    unique_paths: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in unique_paths:
            unique_paths.append(resolved)
    if promote_artifacts is None:
        return {
            "ok": False,
            "destination": destination,
            "paths": [str(path) for path in unique_paths],
            "errors": ["Modal completion requires the kernel workspace promotion tool"],
        }
    try:
        report = promote_artifacts(
            destination=destination,
            paths=tuple(unique_paths),
        )
    except Exception as exc:
        return {
            "ok": False,
            "destination": destination,
            "paths": [str(path) for path in unique_paths],
            "errors": [
                "workspace promotion failed before the idle deadline: "
                f"{type(exc).__name__}: {exc}"
            ],
        }
    if isinstance(report, Mapping) and report.get("ok") is False:
        report_errors = report.get("errors")
        errors = (
            [str(error) for error in report_errors]
            if isinstance(report_errors, Sequence)
            and not isinstance(report_errors, (str, bytes))
            else ["workspace promotion reported failure"]
        )
        return {
            "ok": False,
            "destination": destination,
            "paths": [str(path) for path in unique_paths],
            "report": _record_value(report),
            "errors": errors,
        }
    return {
        "ok": True,
        "destination": destination,
        "paths": [str(path) for path in unique_paths],
        "report": _record_value(report),
        "errors": [],
    }


def _close_handle(
    submission: Submission,
    *,
    intent: str | None = None,
) -> dict[str, Any]:
    try:
        report = submission.handle.close(
            intent=intent
            or (
                "Modal job completed after harvest and promotion: "
                f"{submission.job_id}"
            )
        )
    except Exception as exc:
        return {
            "state": "failed",
            "error": f"Modal handle close failed: {type(exc).__name__}: {exc}",
        }
    if isinstance(report, Mapping) and (
        report.get("ok") is False
        or report.get("state") in {"failed", "error", "refused"}
    ):
        return {
            "state": "failed",
            "report": _record_value(report),
            "error": "Modal handle close reported failure",
        }
    return {"state": "closed", "report": _record_value(report)}


def _failure_message(
    state: str | None, exit_code: int | None, error_kind: str | None
) -> str:
    if error_kind == "result_rejected":
        return (
            "Modal rejected the result stream. Compute cleanup proceeds to cap spend; "
            "recover durable provider or volume artifacts using the recorded job reference."
        )
    if error_kind == "harvest_failed":
        return (
            "Modal could not harvest the result stream. Compute cleanup proceeds to cap spend; "
            "recover durable provider or volume artifacts using the recorded job reference."
        )
    if error_kind == "input_changed":
        return (
            "Modal refused staging because an input changed. Wait for the writer to finish, "
            "then submit the job again."
        )
    if error_kind == "image_build_failed":
        return "Modal image build failed. Fix the environment definition and rebuild it."
    if error_kind == "unauthorized":
        return "Modal authorization failed. Refresh the account token, then run the stage again."
    if error_kind == "quota_exhausted":
        return "Modal quota or spend capacity stopped the job. Review the provider error and free only handles you own."
    if error_kind == "rate_limited":
        return "Modal rate-limited the request. Follow the provider backoff guidance before submitting again."
    if error_kind in CONTAINER_EXPIRY_KINDS:
        return (
            "The Modal container reached its lifetime. Inspect the harvested outputs, then "
            "submit work whose run_timeout_s fits the remaining container time."
        )
    if state == "timed_out":
        return (
            "The Modal job reached run_timeout_s. Partial outputs were harvested. Inspect "
            "them, then increase run_timeout_s or resume from a checkpoint."
        )
    if state == "succeeded" and exit_code not in {0, None}:
        return f"Modal reported exit_code {exit_code} for a succeeded job. The result is refused."
    if exit_code is not None and exit_code != 0:
        return f"Modal job failed with exit_code {exit_code}. Inspect stdout_tail and stderr_tail."
    return "Modal job did not complete successfully. Inspect the harvested logs and provider notes."


def _record_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _record_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_record_value(item) for item in value]
    return str(value)


def stage_input_files(bundle_root: Path) -> list[dict[str, str]]:
    """Return the small bundle files whose missing state fails before submission.

    `remote-compute-modal` documents flat job inputs. The full directory
    staging and receipt-finalization contract remains in
    `skills/claude-binder-lane/scripts/dispatch_modal.py`.
    """
    required = ("config.resolved.json", "run-plan.json", "stage-contract.json")
    missing = [name for name in required if not (bundle_root / name).is_file()]
    if missing:
        raise ModalPlatformError("Modal stage bundle is missing: " + ", ".join(missing))
    return [
        {"src": str(bundle_root / name), "dst_filename": name}
        for name in required
    ]
