"""Resolve stage inputs through the run plan and the producing receipt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class DeclaredArtifactError(RuntimeError):
    """A stage input is absent from the plan, receipt, or declared files."""


def load_plan(plan_path: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load the materialized plan, with a config-stage fallback for unit fixtures."""
    try:
        value = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise DeclaredArtifactError(
            f"run plan is invalid: {plan_path}: {type(exc).__name__}: {exc}"
        ) from exc
    if isinstance(value, dict) and isinstance(value.get("stages"), list):
        return value
    if config is not None and isinstance(config.get("stages"), list):
        return dict(config)
    raise DeclaredArtifactError(f"run plan has no stage declarations: {plan_path}")


def _stage_map(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    stages = plan.get("stages")
    if not isinstance(stages, list):
        raise DeclaredArtifactError("run plan stages are missing or invalid")
    return {
        str(stage.get("stage_id")): stage
        for stage in stages
        if isinstance(stage, Mapping) and isinstance(stage.get("stage_id"), str)
    }


def _plan_artifact_ids(plan: Mapping[str, Any]) -> list[str]:
    ids = {
        str(output.get("artifact_id"))
        for stage in plan.get("stages", [])
        if isinstance(stage, Mapping)
        for output in stage.get("outputs", [])
        if isinstance(output, Mapping) and isinstance(output.get("artifact_id"), str)
    }
    return sorted(ids)


def _input_refs(
    plan: Mapping[str, Any],
    consumer_stage_id: str,
    *,
    artifact_id: str,
    source_stage_id: str | None,
) -> list[tuple[str, str]]:
    stages = _stage_map(plan)
    consumer = stages.get(consumer_stage_id)
    if consumer is None:
        raise DeclaredArtifactError(
            f"plan declares no consumer stage {consumer_stage_id}; "
            f"plan artifact ids: {_plan_artifact_ids(plan)}; "
            f"missing artifact ids: [{artifact_id}]"
        )
    refs: list[tuple[str, str]] = []
    for value in consumer.get("inputs", []):
        if not isinstance(value, str) or ":" not in value:
            continue
        producer, declared_id = value.split(":", 1)
        if declared_id != artifact_id:
            continue
        if source_stage_id is None or producer == source_stage_id:
            refs.append((producer, declared_id))
    if refs:
        return refs
    producers = sorted(
        str(stage.get("stage_id"))
        for stage in stages.values()
        if artifact_id in _declared_output_ids(plan, str(stage.get("stage_id")))
    )
    expected_producer = source_stage_id or (producers[0] if len(producers) == 1 else f"the stage for {artifact_id}")
    raise DeclaredArtifactError(
        f"stage {consumer_stage_id} requires {expected_producer}:{artifact_id}, but that "
        f"input is missing; plan artifact ids: {_plan_artifact_ids(plan)}; "
        f"missing artifact ids: [{artifact_id}]; stage {expected_producer} should have written it"
    )


def _declared_output_ids(plan: Mapping[str, Any], producer_stage_id: str) -> list[str]:
    stage = _stage_map(plan).get(producer_stage_id)
    if stage is None:
        return []
    return sorted(
        str(output.get("artifact_id"))
        for output in stage.get("outputs", [])
        if isinstance(output, Mapping) and isinstance(output.get("artifact_id"), str)
    )


def input_files(
    plan: Mapping[str, Any],
    receipts_dir: Path,
    consumer_stage_id: str,
    *,
    artifact_id: str,
    source_stage_id: str | None = None,
    artifact_type: str | None = None,
    phase_preference: tuple[str, ...] = (),
) -> tuple[str, list[Path]]:
    """Return files for one input artifact named by the consumer stage."""
    refs = _input_refs(
        plan,
        consumer_stage_id,
        artifact_id=artifact_id,
        source_stage_id=source_stage_id,
    )
    if len(refs) != 1:
        raise DeclaredArtifactError(
            f"stage {consumer_stage_id} declares {len(refs)} inputs for "
            f"{artifact_id}; expected one"
        )
    producer_stage_id, declared_id = refs[0]
    plan_output_ids = _declared_output_ids(plan, producer_stage_id)
    if declared_id not in plan_output_ids:
        raise DeclaredArtifactError(
            f"stage {producer_stage_id} should have written artifact {declared_id}, "
            f"but its declared output ids are {plan_output_ids}; "
            f"plan artifact ids: {_plan_artifact_ids(plan)}; "
            f"missing artifact ids: [{declared_id}]"
        )
    receipt_path = receipts_dir.expanduser().resolve() / f"{producer_stage_id}.json"
    if not receipt_path.is_file():
        raise DeclaredArtifactError(
            f"stage {producer_stage_id} should have written artifact {declared_id}, "
            f"but its receipt is missing: {receipt_path}; "
            f"plan artifact ids: {_plan_artifact_ids(plan)}; "
            f"missing artifact ids: [{declared_id}]"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise DeclaredArtifactError(
            f"stage {producer_stage_id} should have written artifact {declared_id}, "
            f"but its receipt is invalid: {receipt_path}: {type(exc).__name__}: {exc}"
        ) from exc
    artifacts = receipt.get("output_manifest", {}).get("artifacts", [])
    matching = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and artifact.get("artifact_id") == declared_id
        and (artifact_type is None or artifact.get("artifact_type") == artifact_type)
    ]
    if not matching:
        actual_ids = sorted(
            str(artifact.get("artifact_id"))
            for artifact in artifacts
            if isinstance(artifact, Mapping) and artifact.get("artifact_id")
        )
        raise DeclaredArtifactError(
            f"stage {producer_stage_id} should have written artifact {declared_id}, "
            f"but its receipt declares {actual_ids}; plan artifact ids: "
            f"{_plan_artifact_ids(plan)}; missing artifact ids: [{declared_id}]"
        )
    if phase_preference:
        phases = {str(artifact.get("phase")) for artifact in matching}
        selected = next((phase for phase in phase_preference if phase in phases), None)
        if selected is None:
            raise DeclaredArtifactError(
                f"stage {producer_stage_id} should have written artifact {declared_id} "
                f"under one of {phase_preference}, but its receipt has phases {sorted(phases)}"
            )
        matching = [artifact for artifact in matching if artifact.get("phase") == selected]
    paths: list[Path] = []
    for artifact in matching:
        files = artifact.get("files")
        if not isinstance(files, list):
            raise DeclaredArtifactError(
                f"stage {producer_stage_id} should have written artifact {declared_id}, "
                "but its receipt has no file list"
            )
        for file_record in files:
            if not isinstance(file_record, Mapping) or not isinstance(file_record.get("path"), str):
                raise DeclaredArtifactError(
                    f"stage {producer_stage_id} should have written artifact {declared_id}, "
                    "but its receipt has an invalid file path"
                )
            path = Path(str(file_record["path"])).expanduser().resolve()
            if not path.is_file():
                raise DeclaredArtifactError(
                    f"stage {producer_stage_id} should have written artifact {declared_id} "
                    f"file, but it is missing: {path}; plan artifact ids: "
                    f"{_plan_artifact_ids(plan)}; missing artifact ids: [{declared_id}]"
                )
            paths.append(path)
    if not paths:
        raise DeclaredArtifactError(
            f"stage {producer_stage_id} should have written artifact {declared_id}, "
            f"but its receipt contains no files; plan artifact ids: {_plan_artifact_ids(plan)}; "
            f"missing artifact ids: [{declared_id}]"
        )
    return producer_stage_id, paths


def output_artifact_id(
    plan: Mapping[str, Any],
    producer_stage_id: str,
    *,
    artifact_type: str,
) -> str:
    """Return the one output id of a producer with the requested type."""
    stage = _stage_map(plan).get(producer_stage_id)
    outputs = stage.get("outputs", []) if stage is not None else []
    matches = [
        str(output.get("artifact_id"))
        for output in outputs
        if isinstance(output, Mapping) and output.get("artifact_type") == artifact_type
    ]
    if len(matches) == 1:
        return matches[0]
    raise DeclaredArtifactError(
        f"stage {producer_stage_id} should have written one {artifact_type} artifact, "
        f"but its declared output ids are {_declared_output_ids(plan, producer_stage_id)}; "
        f"plan artifact ids: {_plan_artifact_ids(plan)}; "
        f"missing artifact ids: [{artifact_type}]"
    )
