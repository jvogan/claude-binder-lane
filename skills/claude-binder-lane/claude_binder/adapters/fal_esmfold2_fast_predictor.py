#!/usr/bin/env python3
"""Bind the deployed ESMFold2-Fast complex fal client to cofold stages.

The client writes the returned mmCIF and compressed PAE and pLDDT arrays. This
adapter turns those files into the same prediction rows and artifact records as
the in-package ESMFold2 adapters.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable
import uuid

from claude_binder.clients import fal_invocation
from claude_binder.paths import package_file

from . import esmfold2_predictor as base


DEFAULT_CLIENT_PYTHON = "python3"
DEFAULT_FAL_EXECUTABLE = "fal-credential-wrapper"
DEFAULT_CLIENT = package_file("clients", "fal_esmfold2_fast_client.py")
PREDICTOR_ID = "esmfold2-fast"
ADAPTER_ID = "esmfold2-fast-predictor"
DEFAULT_WORK_SUBDIR = "ef2fast"
DEFAULT_RUN_INDEX_NAME = "ef2fast-run-index.jsonl"
DEFAULT_CALL_JOURNAL_NAME = "ef2fast-call-journal.jsonl"
DEFAULT_MAX_SECONDS = 1700
DEFAULT_TIMEOUT_SECONDS = 1850
REQUIRED_SNAPSHOT_REPOSITORIES = (
    "biohub/ESMFold2-Fast",
    "biohub/ESMC-6B",
)
SNAPSHOT_PIN_RE = re.compile(
    r"(?P<repository>biohub/(?:ESMFold2-Fast|ESMC-6B))@"
    r"(?P<revision>[0-9a-f]{40})(?![0-9a-f])"
)
TIMING_FIELDS = ("fold_seconds", "model_load_seconds", "total_seconds")
MEASURED_H100_RATE_USD_PER_SECOND = 0.00125
MEASURED_COLD_FOLD_TIMINGS = {
    "fold_seconds": 200.177,
    "model_load_seconds": 555.965,
    "total_seconds": 885.598,
}


class AdapterError(RuntimeError):
    """A fold input or returned artifact is invalid."""


class RevisionProvenanceError(AdapterError):
    """A paid response does not prove it ran the profile-pinned weights."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _journal_record_sha256(record: dict[str, Any]) -> str:
    unsigned = dict(record)
    unsigned.pop("record_sha256", None)
    return sha256_json(unsigned)


def _parse_call_journal(text: str, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterError(
                f"fal call journal line {line_number} is invalid JSON: {path}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise AdapterError(f"fal call journal line {line_number} is not an object: {path}")
        if row.get("previous_record_sha256") != previous:
            raise AdapterError(
                f"fal call journal line {line_number} breaks the hash chain: {path}"
            )
        observed = row.get("record_sha256")
        expected = _journal_record_sha256(row)
        if observed != expected:
            raise AdapterError(
                f"fal call journal line {line_number} has record_sha256 {observed!r}, "
                f"expected {expected}: {path}"
            )
        rows.append(row)
        previous = expected
    return rows


def load_call_journal(path: Path) -> list[dict[str, Any]]:
    """Read and verify the durable per-predict request journal."""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            return _parse_call_journal(handle.read(), path)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _append_call_journal_locked(
    handle: Any,
    rows: list[dict[str, Any]],
    event: dict[str, Any],
) -> dict[str, Any]:
    row = dict(event)
    row["schema_version"] = 1
    row["previous_record_sha256"] = rows[-1]["record_sha256"] if rows else None
    row["record_sha256"] = _journal_record_sha256(row)
    handle.seek(0, os.SEEK_END)
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return row


def append_call_journal(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    """Hash-chain, append, flush and fsync one call event under an exclusive lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            rows = _parse_call_journal(handle.read(), path)
            return _append_call_journal_locked(handle, rows, event)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def call_journal_path(args: argparse.Namespace) -> Path:
    supplied = getattr(args, "call_journal", None)
    if supplied is not None:
        return Path(supplied).expanduser().resolve()
    return (
        args.artifact_root.expanduser().resolve()
        / ".state"
        / safe_part(str(args.stage))
        / DEFAULT_CALL_JOURNAL_NAME
    )


def predict_attempt_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Return local retry metadata that must not change the paid-call identity."""
    attempt_dir = args.attempt_dir.expanduser().resolve()
    return {
        "attempt_id": attempt_dir.name,
        "attempt_dir": str(attempt_dir),
        "phase": str(args.phase),
        "local_timeout_seconds": int(args.timeout_seconds),
    }


def predict_call_key(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    target_id: str,
    candidate_id: str,
    seed: int,
    target_fasta: Path,
    binder_fasta: Path,
) -> dict[str, Any]:
    """Return the stable scientific and remote-payload identity of one paid request."""
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise AdapterError("fal predict call identity requires config.run_id")
    return {
        "run_id": run_id,
        "stage_id": str(args.stage),
        "target_id": target_id,
        "candidate_id": candidate_id,
        "seed": seed,
        "input_sha256": {
            "target_fasta": sha256_file(target_fasta),
            "binder_fasta": sha256_file(binder_fasta),
        },
        "request": {
            "endpoint": "predict",
            "fal_url": str(args.fal_url),
            "max_seconds": int(args.max_seconds),
        },
    }


def call_journal_state(
    rows: list[dict[str, Any]], call_id: str
) -> dict[str, Any] | None:
    """Return the latest exact-call state; other call IDs are deliberately ignored."""
    matching = [row for row in rows if row.get("call_id") == call_id]
    intents = [row for row in matching if row.get("event") == "predict-intent"]
    if not intents:
        return None
    intent = intents[-1]
    intent_id = intent.get("intent_id")
    following = [row for row in matching if row.get("intent_id") == intent_id]
    reconciliation = next(
        (row for row in reversed(following) if row.get("event") == "predict-reconciliation"),
        None,
    )
    if reconciliation is not None:
        disposition = reconciliation.get("disposition")
        if disposition == "not-accepted":
            return {"status": "retryable", "intent": intent, "terminal": reconciliation}
        if disposition == "completed-output-recovered":
            return {"status": "completed", "intent": intent, "terminal": reconciliation}
    outcome = next(
        (row for row in reversed(following) if row.get("event") == "predict-outcome"),
        None,
    )
    if outcome is not None and outcome.get("outcome") == "completed":
        return {"status": "completed", "intent": intent, "terminal": outcome}
    return {"status": "unresolved", "intent": intent, "terminal": outcome}


def begin_predict_call(
    journal_path: Path,
    call_key: dict[str, Any],
    *,
    attempt_metadata: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Persist intent before dispatch, or reject/reuse this exact prior call."""
    call_id = sha256_json(call_key)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            rows = _parse_call_journal(handle.read(), journal_path)
            prior = call_journal_state(rows, call_id)
            if prior is not None and prior["status"] == "unresolved":
                intent_id = prior["intent"].get("intent_id")
                raise AdapterError(
                    f"fal predict call {call_id} has prior accepted-or-unknown intent "
                    f"{intent_id}; reconcile that exact call before retry. Other call keys "
                    "remain eligible."
                )
            if prior is not None and prior["status"] == "completed":
                return call_id, prior
            intent_id = uuid.uuid4().hex
            _append_call_journal_locked(
                handle,
                rows,
                {
                    "event": "predict-intent",
                    "call_id": call_id,
                    "intent_id": intent_id,
                    "call_key": call_key,
                    "attempt": dict(attempt_metadata or {}),
                    "provider_acceptance": "unknown",
                    "recorded_at": utc_now(),
                },
            )
            return call_id, None
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def record_call_reconciliation(
    journal_path: Path,
    call_id: str,
    *,
    disposition: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Record operator evidence that resolves one prior exact-call uncertainty."""
    if disposition not in {"not-accepted", "completed-output-recovered"}:
        raise AdapterError(
            "fal call reconciliation disposition must be not-accepted or "
            "completed-output-recovered"
        )
    if not isinstance(evidence, dict) or not evidence:
        raise AdapterError("fal call reconciliation requires non-empty evidence")
    response_sha256 = evidence.get("response_sha256")
    if disposition == "completed-output-recovered" and (
        not isinstance(response_sha256, str)
        or len(response_sha256) != 64
        or any(character not in "0123456789abcdef" for character in response_sha256)
    ):
        raise AdapterError(
            "completed-output-recovered reconciliation requires evidence.response_sha256"
        )
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            rows = _parse_call_journal(handle.read(), journal_path)
            state = call_journal_state(rows, call_id)
            if state is None or state["status"] != "unresolved":
                raise AdapterError(
                    f"fal predict call {call_id} has no unresolved intent to reconcile"
                )
            event = {
                "event": "predict-reconciliation",
                "call_id": call_id,
                "intent_id": state["intent"]["intent_id"],
                "disposition": disposition,
                "evidence": evidence,
                "recorded_at": utc_now(),
            }
            if disposition == "completed-output-recovered":
                event["response_sha256"] = response_sha256
            return _append_call_journal_locked(handle, rows, event)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"{label} is invalid: {path}: {exc}") from exc


def nonnegative_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def timing_ledger_fields(response: Any) -> dict[str, float | None]:
    """Return the measured fal timings and their total-seconds cost estimate."""
    timings = response.get("timings") if isinstance(response, dict) else None
    values = {
        field: nonnegative_finite_number(timings.get(field)) if isinstance(timings, dict) else None
        for field in TIMING_FIELDS
    }
    total_seconds = values["total_seconds"]
    values["estimated_cost_usd"] = (
        total_seconds * MEASURED_H100_RATE_USD_PER_SECOND if total_seconds is not None else None
    )
    return values


def profile_snapshot_revisions(model_revision: str) -> dict[str, str]:
    """Extract the two immutable repository pins required by the fal worker."""
    matches: dict[str, list[str]] = {
        repository: [] for repository in REQUIRED_SNAPSHOT_REPOSITORIES
    }
    for match in SNAPSHOT_PIN_RE.finditer(model_revision):
        matches[match.group("repository")].append(match.group("revision"))
    invalid = {
        repository: revisions
        for repository, revisions in matches.items()
        if len(revisions) != 1
    }
    if invalid:
        detail = ", ".join(
            f"{repository}={revisions!r}" for repository, revisions in invalid.items()
        )
        raise RevisionProvenanceError(
            "revision-provenance: adapter model_revision must carry exactly one "
            f"immutable pin for each fal worker repository; found {detail}"
        )
    return {repository: revisions[0] for repository, revisions in matches.items()}


def verify_response_snapshot_revisions(
    response: Any,
    model_revision: str,
) -> dict[str, str]:
    """Verify and return the repository revisions the paid response actually loaded."""
    expected = profile_snapshot_revisions(model_revision)
    snapshots = response.get("resolved_snapshots") if isinstance(response, dict) else None
    if not isinstance(snapshots, dict):
        raise RevisionProvenanceError(
            "revision-provenance: fal response carries no resolved_snapshots object"
        )
    verified: dict[str, str] = {}
    for repository, expected_revision in expected.items():
        record = snapshots.get(repository)
        observed_revision = record.get("revision") if isinstance(record, dict) else None
        if not isinstance(observed_revision, str) or not observed_revision:
            raise RevisionProvenanceError(
                "revision-provenance: fal response carries no resolved revision for "
                f"{repository}; expected {expected_revision}"
            )
        if observed_revision != expected_revision:
            raise RevisionProvenanceError(
                "revision-provenance: fal response resolved "
                f"{repository}@{observed_revision}, but the profile pins "
                f"{repository}@{expected_revision}"
            )
        verified[repository] = observed_revision
    return verified


def estimate_cold_and_warm_costs(fold_count: int) -> dict[str, Any]:
    """Project cold and single-warm-worker cost from the recorded fold measurement."""
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 0:
        raise ValueError("fold_count must be a non-negative integer")

    fold_seconds = MEASURED_COLD_FOLD_TIMINGS["fold_seconds"]
    model_load_seconds = MEASURED_COLD_FOLD_TIMINGS["model_load_seconds"]
    total_seconds = MEASURED_COLD_FOLD_TIMINGS["total_seconds"]
    cold = {
        "fold_seconds": round(fold_seconds * fold_count, 3),
        "model_load_seconds": round(model_load_seconds * fold_count, 3),
        "total_seconds": round(total_seconds * fold_count, 3),
    }
    warm_model_load_seconds = model_load_seconds if fold_count else 0.0
    warm = {
        "fold_seconds": round(fold_seconds * fold_count, 3),
        "model_load_seconds": round(warm_model_load_seconds, 3),
        "total_seconds": round(
            (total_seconds - model_load_seconds) * fold_count + warm_model_load_seconds,
            3,
        ),
    }
    cold["estimated_cost_usd"] = round(
        cold["total_seconds"] * MEASURED_H100_RATE_USD_PER_SECOND,
        9,
    )
    warm["estimated_cost_usd"] = round(
        warm["total_seconds"] * MEASURED_H100_RATE_USD_PER_SECOND,
        9,
    )
    return {
        "fold_count": fold_count,
        "rate_usd_per_h100_second": MEASURED_H100_RATE_USD_PER_SECOND,
        "cold": cold,
        "warm": warm,
        "savings": {
            "model_load_seconds": round(
                cold["model_load_seconds"] - warm["model_load_seconds"],
                3,
            ),
            "estimated_cost_usd": round(
                cold["estimated_cost_usd"] - warm["estimated_cost_usd"],
                9,
            ),
        },
    }


def nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def print_cost_estimate(fold_count: int) -> int:
    print(json.dumps(estimate_cold_and_warm_costs(fold_count), indent=2, sort_keys=True))
    return 0


def resolve_client(value: Path | None) -> Path:
    environment_value = os.environ.get("CLAUDE_BINDER_EF2FAST_CLIENT", "").strip()
    candidate = value or (Path(environment_value) if environment_value else DEFAULT_CLIENT)
    path = candidate.expanduser().resolve()
    if not path.is_file():
        raise AdapterError(f"ESMFold2-Fast fal client is missing: {path}")
    return path


def run_external(
    argv: list[str], label: str, *, wall_timeout_seconds: int | None = None
) -> None:
    print(
        f"ef2fast fal adapter: {label}: {fal_invocation.redacted_command(argv)}",
        flush=True,
    )
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            check=False,
            timeout=wall_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterError(
            f"{label} exceeded its {wall_timeout_seconds}-second local wall limit; "
            "the remote request state is unknown and must be reconciled before retry"
        ) from exc
    if completed.returncode != 0:
        raise AdapterError(f"{label} exited {completed.returncode}")


PREDICT_OUTPUT_FILES = (
    "response.json", "predicted.cif", "pae.bin", "pae.json", "plddt.bin", "plddt.json"
)


def prediction_output_records(out_dir: Path, journal_path: Path) -> dict[str, Any]:
    """Keep explicit output locations relative to the journal for portable resume."""
    return {
        name: {
            "path": os.path.relpath(out_dir / name, journal_path.parent),
            "sha256": sha256_file(out_dir / name),
        }
        for name in PREDICT_OUTPUT_FILES
        if (out_dir / name).is_file()
    }


def restore_prediction_outputs(
    terminal: dict[str, Any], out_dir: Path, journal_path: Path, call_id: str
) -> None:
    """Verify all recorded bytes before restoring a completed call into this attempt."""
    records = terminal.get("output_files")
    if records is None:
        # Older journals authenticated only the response in the original attempt.
        return
    if not isinstance(records, dict) or "response.json" not in records:
        raise AdapterError(f"fal predict call {call_id} has malformed output records")
    pending: list[tuple[Path, bytes]] = []
    for name, record in records.items():
        if name not in PREDICT_OUTPUT_FILES or not isinstance(record, dict):
            raise AdapterError(f"fal predict call {call_id} has invalid output name {name!r}")
        expected = record.get("sha256")
        relative = record.get("path")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise AdapterError(f"fal predict call {call_id} has incomplete output record {name}")
        if name == "response.json" and expected != terminal.get("response_sha256"):
            raise AdapterError(f"fal predict call {call_id} has conflicting response hashes")
        destination = out_dir / name
        source = destination if destination.exists() else journal_path.parent / relative
        try:
            payload = source.read_bytes()
        except OSError as exc:
            raise AdapterError(
                f"fal predict call {call_id} needs its recorded {name} at {source}; "
                "recover that output before retry"
            ) from exc
        if hashlib.sha256(payload).hexdigest() != expected:
            raise AdapterError(f"fal predict call {call_id} output drifted: {source}")
        if source != destination:
            pending.append((destination, payload))
    for destination, payload in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Publish only complete bytes and preserve any concurrent writer's file.
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".recover-") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            try:
                os.link(handle.name, destination)
            except FileExistsError:
                if destination.read_bytes() != payload:
                    raise AdapterError(f"fal predict call {call_id} output drifted: {destination}")


def guarded_predict_call(
    *,
    journal_path: Path,
    call_key: dict[str, Any],
    argv: list[str],
    label: str,
    wall_timeout_seconds: int,
    out_dir: Path,
    attempt_metadata: dict[str, Any] | None = None,
    prepare: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], bool, str]:
    """Run one paid call with a durable exact-call intent and terminal outcome."""
    call_id, prior = begin_predict_call(
        journal_path,
        call_key,
        attempt_metadata=attempt_metadata,
    )
    response_path = out_dir / "response.json"
    if prior is not None:
        terminal = prior["terminal"]
        restore_prediction_outputs(terminal, out_dir, journal_path, call_id)
        expected_hash = terminal.get("response_sha256")
        if not isinstance(expected_hash, str) or not response_path.is_file():
            raise AdapterError(
                f"fal predict call {call_id} is complete but its response cannot be "
                "reused; reconcile the exact call and recover its output before retry"
            )
        observed_hash = sha256_file(response_path)
        if observed_hash != expected_hash:
            raise AdapterError(
                f"fal predict call {call_id} response drifted: expected {expected_hash}, "
                f"observed {observed_hash}; reconcile before retry"
            )
        return load_json(response_path, "ESMFold2-Fast response"), True, call_id

    state = call_journal_state(load_call_journal(journal_path), call_id)
    if state is None or state["status"] != "unresolved":
        raise AdapterError(f"fal predict call {call_id} lost its persisted intent")
    intent_id = str(state["intent"]["intent_id"])
    if prepare is not None:
        try:
            prepare()
        except BaseException:
            # Hydration may have started, but the predict command has not run.
            # Preserve that distinction so a failed preparation cannot block a fold.
            record_call_reconciliation(
                journal_path, call_id, disposition="not-accepted",
                evidence={"source": "adapter preparation", "predict_invoked": False},
            )
            raise
    try:
        run_external(argv, label, wall_timeout_seconds=wall_timeout_seconds)
        response = load_json(response_path, "ESMFold2-Fast response")
    except (KeyboardInterrupt, SystemExit):
        # The durable intent is the evidence. A process-level interruption may
        # happen after the provider accepted the request, so no outcome is
        # invented and an exact retry stays closed until reconciliation.
        raise
    except BaseException as exc:
        append_call_journal(
            journal_path,
            {
                "event": "predict-outcome",
                "call_id": call_id,
                "intent_id": intent_id,
                "outcome": "unknown",
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "recorded_at": utc_now(),
            },
        )
        raise

    append_call_journal(
        journal_path,
        {
            "event": "predict-outcome",
            "call_id": call_id,
            "intent_id": intent_id,
            "outcome": "completed",
            "response_sha256": sha256_file(response_path),
            "output_files": prediction_output_records(out_dir, journal_path),
            "timings": timing_ledger_fields(response),
            "recorded_at": utc_now(),
        },
    )
    return response, False, call_id


def client_argv(args: argparse.Namespace, client: Path, endpoint: str, *values: str) -> list[str]:
    """Build the client command on whichever credential route this machine offers."""
    try:
        client_values = [endpoint, "--fal-url", args.fal_url, *values]
        credential_env = fal_invocation.credential_environment_key(
            getattr(args, "fal_credential_env", None)
        )
        route = fal_invocation.resolve_route(
            args.fal_executable,
            requested=getattr(args, "fal_credential_route", None),
            credential_env_key=credential_env,
        )
        if (
            route == fal_invocation.ROUTE_DIRECT
            and credential_env != fal_invocation.CREDENTIAL_ENVIRONMENT_KEY
        ):
            client_values.extend(["--credential-env", credential_env])
        return fal_invocation.client_command(
            args.fal_executable,
            args.client_python,
            client,
            client_values,
            requested=route,
            credential_env_key=credential_env,
        )
    except fal_invocation.RouteError as exc:
        raise AdapterError(str(exc)) from exc


def hydrate_once(args: argparse.Namespace, client: Path) -> dict[str, Any]:
    """Warm the application once before the first fold of this stage.

    A cold worker loads its weights inside the same request that folds, and that
    load is charged against the ``--max-seconds`` this adapter asks for. So a
    cold first fold can spend the whole budget loading and return
    ``WorkerFailedError`` with no structure.
    ``2026-08-30-crossprovider/provenance.json`` records one such failure caused
    by hydrate not running first. Warming does not explain every capped request:
    the 2026-09-04 HER2 canary hydrated successfully 35 seconds before a request
    that still burned 1710.5 seconds, and folded a longer 269-residue complex in
    385.0 seconds, so that cause is open. This call removes one known mechanism
    cheaply, it does not close the open one. Measured cold model load is
    ``MEASURED_COLD_FOLD_TIMINGS["model_load_seconds"]`` seconds against a
    ``fold_seconds`` of ``MEASURED_COLD_FOLD_TIMINGS["fold_seconds"]``.

    ``hydrate`` is idempotent and removes the failure mode. It validates before
    the stage executes, which is the contract every other pre-dispatch check here
    follows. The stage's admission estimate retains the recorded cold-load upper
    bound; this helper does not claim that hydration is free.
    """
    out_dir = (
        args.attempt_dir.expanduser().resolve() / args.phase / args.work_subdir / "hydrate"
    )
    argv = client_argv(
        args,
        client,
        "hydrate",
        "--out-dir",
        str(out_dir),
        "--timeout-seconds",
        str(args.timeout_seconds),
    )
    run_external(argv, "hydrate", wall_timeout_seconds=args.timeout_seconds + 75)
    response = load_json(out_dir / "response.json", "ESMFold2-Fast hydrate response")
    if not isinstance(response, dict) or response.get("complete") is not True:
        raise AdapterError(
            "ESMFold2-Fast hydrate did not report complete. Folding now would pay the "
            "cold model load inside each fold's own budget and can return no structure. "
            "Resolve the application state, or pass --skip-hydrate to fold against an "
            "application already known to be warm."
        )
    return response


def safe_part(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def write_fasta(path: Path, name: str, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">{name}\n{sequence}\n")


def pae_matrix(out_dir: Path) -> list[list[float]]:
    pae_path = out_dir / "pae.bin"
    sidecar_path = out_dir / "pae.json"
    sidecar = load_json(sidecar_path, "PAE sidecar")
    if not isinstance(sidecar, dict) or sidecar.get("dtype") != "float32":
        raise AdapterError(f"PAE sidecar does not declare float32: {sidecar_path}")
    shape = sidecar.get("shape")
    if not isinstance(shape, list) or len(shape) != 2 or shape[0] != shape[1]:
        raise AdapterError(f"PAE sidecar does not declare a square shape: {sidecar_path}")
    size = int(shape[0])
    if size < 1:
        raise AdapterError(f"PAE sidecar declares an empty matrix: {sidecar_path}")
    payload = pae_path.read_bytes()
    expected = size * size * 4
    if len(payload) != expected:
        raise AdapterError(f"PAE bytes {len(payload)} do not match shape {shape}: {pae_path}")
    values = struct.unpack("<" + "f" * (size * size), payload)
    return [list(values[offset : offset + size]) for offset in range(0, len(values), size)]


def target_and_binder_sequences(
    config: dict[str, Any], args: argparse.Namespace, candidate: dict[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    targets = config["targets"]
    supplied = base.resolve_per_target(args.target_sequence, targets, "--target-sequence")
    target_id = str(candidate["target"]["target_id"])
    target_sequence = base.resolve_target_sequence(target_id, supplied)
    sequence_path = candidate["candidate"].get("sequence_path")
    if not sequence_path:
        raise AdapterError(f"candidate {candidate['candidate']['candidate_id']} carries no sequence_path")
    binder_sequence = base.read_fasta_sequence(Path(sequence_path))
    return target_sequence, binder_sequence, candidate["target"]


def preflight_item(
    config: dict[str, Any],
    args: argparse.Namespace,
    item: dict[str, Any],
    supplied_targets: dict[str, str],
    supplied_hotspots: dict[str, str],
) -> tuple[str, str, dict[str, Any]]:
    """Validate every static input before the first paid fold starts."""
    candidate = item["candidate"]
    candidate_id = str(candidate.get("candidate_id"))
    target_sequence, binder_sequence, target = target_and_binder_sequences(config, args, item)
    design_pose_value = candidate.get("design_pose_path")
    expected_hash = candidate.get("design_pose_sha256")
    if not isinstance(design_pose_value, str) or not design_pose_value:
        raise AdapterError(f"candidate {candidate_id} carries no design_pose_path")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise AdapterError(f"candidate {candidate_id} carries no design_pose_sha256")
    design_pose = Path(design_pose_value).expanduser().resolve()
    if not design_pose.is_file():
        raise AdapterError(f"candidate {candidate_id} design pose is missing: {design_pose}")
    observed_hash = sha256_file(design_pose)
    if observed_hash != expected_hash:
        raise AdapterError(
            f"candidate {candidate_id} design pose is stale: {design_pose}; "
            f"manifest has {expected_hash}, file has {observed_hash}"
        )
    target_id = str(target["target_id"])
    if target_id not in supplied_targets:
        raise AdapterError(f"no target sequence supplied for {target_id}")
    try:
        base.site_residue_map_for(config, target, supplied_hotspots)
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(
            f"static site contract is invalid for target {target_id}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return target_sequence, binder_sequence, target


def build_plan(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    row_phase = base.campaign_phase(args.stage)
    return base.plan_predictions(
        config,
        stage_id=args.stage,
        row_phase=row_phase,
        artifact_root=args.artifact_root,
        count=args.count,
        predictor_id=PREDICTOR_ID,
    )


def run(args: argparse.Namespace) -> int:
    config = base.load_json(args.config)
    model_revision = base.model_revision_for(config, ADAPTER_ID)
    # Validate the profile pins before hydrate or predict can spend anything.
    profile_snapshot_revisions(model_revision)
    plan = build_plan(config, args)
    supplied = base.resolve_per_target(args.target_sequence, config["targets"], "--target-sequence")
    hotspots = base.resolve_per_target(args.hotspot_residues, config["targets"], "--hotspot-residues")
    preflight: dict[tuple[str, str, int], tuple[str, str, dict[str, Any]]] = {}
    for item in plan:
        key = (
            str(item["target"]["target_id"]),
            str(item["candidate"]["candidate_id"]),
            int(item["seed"]),
        )
        try:
            preflight[key] = preflight_item(config, args, item, supplied, hotspots)
        except AdapterError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(
                f"pre-dispatch validation failed for {key}: {type(exc).__name__}: {exc}"
            ) from exc
    client = resolve_client(args.client)
    hydrated = getattr(args, "skip_hydrate", False)

    def prepare_prediction() -> None:
        nonlocal hydrated
        if not hydrated:
            hydrate_once(args, client)
            hydrated = True

    journal_path = call_journal_path(args)
    index_rows: list[dict[str, Any]] = []
    for item in plan:
        candidate = item["candidate"]
        target = item["target"]
        candidate_id = str(candidate["candidate_id"])
        target_id = str(target["target_id"])
        seed = int(item["seed"])
        out_dir = (
            args.attempt_dir.expanduser().resolve()
            / args.phase
            / args.work_subdir
            / safe_part(candidate_id)
            / f"seed-{seed}"
        )
        binder_path = out_dir / "binder.fasta"
        target_path = out_dir / "target.fasta"
        record: dict[str, Any] = {
            "target_id": target_id,
            "candidate_id": candidate_id,
            "seed": seed,
            "out_dir": str(out_dir),
            "binder_fasta": str(binder_path),
            "target_fasta": str(target_path),
            "status": "failed",
        }
        try:
            target_sequence, binder_sequence, _ = preflight[
                (target_id, candidate_id, seed)
            ]
            write_fasta(binder_path, candidate_id, binder_sequence)
            write_fasta(target_path, target_id, target_sequence)
            argv = client_argv(
                args,
                client,
                "predict",
                "--out-dir",
                str(out_dir),
                "--binder-fasta",
                str(binder_path),
                "--target-fasta",
                str(target_path),
                "--seed",
                str(seed),
                "--max-seconds",
                str(args.max_seconds),
                "--timeout-seconds",
                str(args.timeout_seconds),
            )
            call_key = predict_call_key(
                config,
                args,
                target_id=target_id,
                candidate_id=candidate_id,
                seed=seed,
                target_fasta=target_path,
                binder_fasta=binder_path,
            )
            record["call_id"] = sha256_json(call_key)
            response, reused, call_id = guarded_predict_call(
                journal_path=journal_path,
                call_key=call_key,
                argv=argv,
                label=f"fold {candidate_id} seed {seed}",
                wall_timeout_seconds=args.timeout_seconds + 75,
                out_dir=out_dir,
                attempt_metadata=predict_attempt_metadata(args),
                prepare=prepare_prediction,
            )
            if call_id != record["call_id"]:
                raise AdapterError("fal predict call identity changed during dispatch")
            record["verified_resolved_snapshots"] = verify_response_snapshot_revisions(
                response,
                model_revision,
            )
            record["reused_completed_call"] = reused
            record.update(timing_ledger_fields(response))
            record["status"] = "completed"
        except RevisionProvenanceError as exc:
            record["failure_code"] = "revision_provenance_failed"
            record["failure_reason"] = str(exc)
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"ef2fast fal adapter: {record['error']}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"ef2fast fal adapter: {record['error']}", file=sys.stderr)
        index_rows.append(record)
    write_jsonl(args.run_index.expanduser().resolve(), index_rows)
    print(f"ef2fast fal adapter: dispatched {len(index_rows)} folds")
    return 0 if any(record.get("status") == "completed" for record in index_rows) else 1


def toolcheck(args: argparse.Namespace) -> int:
    client = resolve_client(args.client)
    run_external(
        client_argv(args, client, "preflight", "--out-dir", str(args.out_dir.expanduser().resolve())),
        "fal preflight",
    )
    return 0


def parse_outputs(args: argparse.Namespace) -> int:
    from . import binder_contract

    config = base.load_json(args.config)
    row_phase = base.campaign_phase(args.stage)
    controls = base.control_records(config)
    model_revision = base.model_revision_for(config, ADAPTER_ID)
    manifest_path, artifacts_attempt_dir = base.output_paths(
        config, args.stage, args.attempt_dir.expanduser().resolve(), args.phase
    )
    run_index = load_jsonl(args.run_index.expanduser().resolve(), "ESMFold2-Fast run index")
    plan = build_plan(config, args)
    plan_by_key = {
        (str(item["target"]["target_id"]), str(item["candidate"]["candidate_id"]), int(item["seed"])): item
        for item in plan
    }
    target_sequences = base.resolve_per_target(args.target_sequence, config["targets"], "--target-sequence")
    hotspots = base.resolve_per_target(args.hotspot_residues, config["targets"], "--hotspot-residues")
    writer = base.RowWriter(manifest_path)
    errors: list[str] = []
    for record in run_index:
        key = (str(record.get("target_id")), str(record.get("candidate_id")), int(record.get("seed", 0)))
        item = plan_by_key.get(key)
        if item is None:
            errors.append(f"run index has an unplanned fold: {key}")
            continue
        row = base.base_row(
            config,
            item=item,
            row_phase=row_phase,
            model_revision=model_revision,
            controls=controls,
        )
        if record.get("status") != "completed":
            writer.write(
                base.failed_row(
                    binder_contract,
                    row,
                    failure_code=str(record.get("failure_code", "fal_request_failed")),
                    failure_reason=str(
                        record.get(
                            "failure_reason",
                            record.get("error", "fal client did not complete"),
                        )
                    )[:500],
                )
            )
            continue
        out_dir = Path(str(record["out_dir"])).expanduser().resolve()
        try:
            response = load_json(out_dir / "response.json", "ESMFold2-Fast response")
            verified_snapshots = verify_response_snapshot_revisions(response, model_revision)
            recorded_snapshots = record.get("verified_resolved_snapshots")
            if recorded_snapshots is not None and recorded_snapshots != verified_snapshots:
                raise RevisionProvenanceError(
                    "revision-provenance: parsed response snapshots differ from the "
                    "run-index verification"
                )
            if not isinstance(response, dict) or not isinstance(response.get("mmcif"), str):
                raise AdapterError(f"response carries no mmcif: {out_dir / 'response.json'}")
            target_sequence = base.resolve_target_sequence(str(item["target"]["target_id"]), target_sequences)
            binder_sequence = base.read_fasta_sequence(Path(item["candidate"]["sequence_path"]))
            site_map = base.site_residue_map_for(config, item["target"], hotspots)
            written = binder_contract.write_prediction_artifacts(
                attempt_dir=artifacts_attempt_dir,
                phase=row_phase,
                run_phase=args.phase,
                target_id=row["target_id"],
                candidate_id=row["candidate_id"],
                predictor=row["predictor"],
                seed=row["seed"],
                complex_cif=response["mmcif"],
                pae=pae_matrix(out_dir),
                chain_mapping=row["chain_mapping"],
                reference_cif=Path(row["design_pose_path"]),
                site_residue_map=site_map,
                model_revision=model_revision,
                target_sequence=target_sequence,
                binder_sequence=binder_sequence,
                extra={
                    "target_sha256": row["target_sha256"],
                    "sequence_sha256": row["sequence_sha256"],
                    "design_pose_sha256": row["design_pose_sha256"],
                    "iptm": response.get("iptm"),
                    "ptm": response.get("ptm"),
                    "mean_plddt": response.get("mean_plddt"),
                    "verified_resolved_snapshots": verified_snapshots,
                },
            )
            merged = dict(row)
            merged.update(written)
            writer.write(merged)
        except RevisionProvenanceError as exc:
            writer.write(
                base.failed_row(
                    binder_contract,
                    row,
                    failure_code="revision_provenance_failed",
                    failure_reason=str(exc)[:500],
                )
            )
        except Exception as exc:  # noqa: BLE001
            writer.write(
                base.failed_row(
                    binder_contract,
                    row,
                    failure_code="prediction_parse_failed",
                    failure_reason=f"{type(exc).__name__}: {exc}"[:500],
                )
            )
    if writer.count > 0 and writer.failed == writer.count:
        errors.append("all planned predictions were rejected during parsing")
    writer.close()
    result_path = args.attempt_dir.expanduser().resolve() / args.phase / "parser-result.json"
    write_json(
        result_path,
        {
            "ok": writer.count > 0 and not errors,
            "parsed_count": writer.count,
            "rejected_count": len(errors),
            "errors": errors,
            # The executor compares this list with the hashes in the declared output
            # manifest. The provider responses are scratch inputs to this adapter and
            # are not stage outputs.
            "source_output_hashes": [sha256_file(manifest_path)] if manifest_path.is_file() else [],
        },
    )
    return 0 if writer.count > 0 and not errors else 1


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AdapterError(f"{label} line {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise AdapterError(f"{label} is empty: {path}")
    return rows


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--client", type=Path, default=None)
    parser.add_argument("--client-python", default=DEFAULT_CLIENT_PYTHON)
    parser.add_argument("--fal-executable", default=DEFAULT_FAL_EXECUTABLE)
    parser.add_argument("--fal-url", required=True)
    fal_invocation.add_route_argument(parser, executable=DEFAULT_FAL_EXECUTABLE)
    fal_invocation.add_credential_environment_argument(parser)


def add_stage_arguments(parser: argparse.ArgumentParser, *, run_command: bool = False) -> None:
    for name, kwargs in (
        ("--stage", {"required": True}),
        ("--phase", {"required": True}),
        ("--count", {"type": int, "default": 1}),
        ("--attempt-dir", {"type": Path, "required": True}),
        ("--receipts-dir", {"type": Path, "required": True}),
        ("--artifact-root", {"type": Path, "required": True}),
        ("--config", {"type": Path, "required": True}),
        ("--plan", {"type": Path, "required": True}),
    ):
        parser.add_argument(name, **kwargs)
    parser.add_argument("--run-index", type=Path, required=True)
    parser.add_argument("--target-sequence", action="append")
    parser.add_argument("--hotspot-residues", action="append")
    if run_command:
        parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
        parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
        parser.add_argument("--work-subdir", default=DEFAULT_WORK_SUBDIR)
        parser.add_argument("--call-journal", type=Path, default=None)
        parser.add_argument(
            "--skip-hydrate",
            action="store_true",
            help=(
                "Fold without warming the application first. Only for an application "
                "already known to be warm; a cold worker spends the per-fold budget "
                "loading weights and can return no structure."
            ),
        )
    else:
        parser.add_argument("--device", default="cuda")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    toolcheck_parser = subparsers.add_parser("toolcheck")
    add_common_arguments(toolcheck_parser)
    toolcheck_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    add_common_arguments(run_parser)
    add_stage_arguments(run_parser, run_command=True)
    parse_parser = subparsers.add_parser("parse")
    add_common_arguments(parse_parser)
    add_stage_arguments(parse_parser)
    estimate_parser = subparsers.add_parser("estimate-cost")
    estimate_parser.add_argument("--fold-count", type=nonnegative_integer, required=True)
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
        if args.command == "estimate-cost":
            return print_cost_estimate(args.fold_count)
        return run(args)
    except AdapterError as exc:
        print(f"ef2fast fal adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
