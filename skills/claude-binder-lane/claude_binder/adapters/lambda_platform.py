"""Native Lambda Cloud route sharing Binder's hardened cloud lifecycle.

This module is intentionally a thin provider binding. Approval, cap
reservation, crash recovery, ambiguous-submission recovery, receipt handling,
cleanup, and financial reconciliation live in ``runpod_platform`` so the two
native compute routes cannot silently diverge. No deployment identifier or
external skill/plugin is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import runpod_platform as native


ROUTE = native.LAMBDA_ROUTE
PROVIDER_ID = ROUTE.provider_id
ROUTE_KEY = native.ROUTE_KEY
ROUTE_VALUE = ROUTE.route_value
PROVIDER_PARAMS_KEY = ROUTE.provider_params_key
LambdaPlatformError = native.RunPodPlatformError
Submission = native.Submission
Settlement = native.Settlement


class ClientHost(native.ClientHost):
    """Adapt a Lambda Cloud client handle factory to Binder's host protocol."""

    def __init__(self, create_handle: Callable[[Mapping[str, Any]], Any]) -> None:
        super().__init__(create_handle, route=ROUTE)


def environment_account_reader(
    read_account: Callable[[str], Any],
    *,
    api_key_environment_key: str,
    environ: Mapping[str, str] | None = None,
) -> Callable[[Mapping[str, str]], Any]:
    """Use a caller-named credential variable without assuming a secret name."""
    return native.environment_account_reader(
        read_account,
        environ=environ,
        route=ROUTE,
        api_key_environment_key=api_key_environment_key,
    )


def uses_lambda_platform(adapter: Mapping[str, Any]) -> bool:
    return native.uses_native_platform(adapter, route=ROUTE)


def provider_params(adapter: Mapping[str, Any]) -> dict[str, Any]:
    return native.provider_params(adapter, route=ROUTE)


def submit_stage(host: Any, **kwargs: Any) -> Submission:
    return native.submit_stage(host, route=ROUTE, **kwargs)


def guarded_submit_stage(host: Any, **kwargs: Any) -> Submission:
    return native.guarded_submit_stage(host, route=ROUTE, **kwargs)


def attach_stage(host: Any, record: Mapping[str, Any]) -> Submission:
    return native.attach_stage(host, record, route=ROUTE)


def bind_uncertain_submission(
    run_root: Path,
    *,
    plan: Mapping[str, Any],
    submission_id: str,
    job_id: str,
) -> dict[str, Any]:
    return native.bind_uncertain_submission(
        run_root,
        plan=plan,
        submission_id=submission_id,
        job_id=job_id,
        route=ROUTE,
    )


def bind_and_attach_uncertain_submission(
    host: Any,
    run_root: Path,
    *,
    plan: Mapping[str, Any],
    submission_id: str,
    job_id: str,
) -> Submission:
    return native.bind_and_attach_uncertain_submission(
        host,
        run_root,
        plan=plan,
        submission_id=submission_id,
        job_id=job_id,
        route=ROUTE,
    )


def settle_stage(
    submission: Submission,
    *,
    harvest_root: Path,
    receipt_validator: Callable[[dict[str, Any], Path], Mapping[str, Any]] | None = None,
) -> Settlement:
    return native.settle_stage(
        submission,
        harvest_root=harvest_root,
        receipt_validator=receipt_validator,
        route=ROUTE,
    )


def settle_and_record_stage(
    submission: Submission,
    *,
    plan: Mapping[str, Any],
    run_root: Path,
    stage_id: str,
    harvest_root: Path,
    receipt_validator: Callable[[dict[str, Any], Path], Mapping[str, Any]] | None = None,
) -> Settlement:
    return native.settle_and_record_stage(
        submission,
        plan=plan,
        run_root=run_root,
        stage_id=stage_id,
        harvest_root=harvest_root,
        receipt_validator=receipt_validator,
        route=ROUTE,
    )
