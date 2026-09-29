"""Run stage contracts against a local, provider-free fixture tree."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import contract_audit
from .adapters.declared_artifacts import input_files
from .paths import package_root, schema_file


DEFAULT_FIXTURE_ROOT = Path(
    os.environ.get(
        "CLAUDE_BINDER_CONTRACT_FIXTURE_ROOT",
        ".claude-binder-contract-fixture",
    )
)
# `--fixture-root` is emptied recursively before the run. The default is a dot directory this
# tool owns, so the loss only reaches a user who names their own directory. Defect 65 was the
# same shape on `finalize --session`, so the same rule applies: recognize our own tree and refuse
# every other occupied path. A root carrying this stamp, or the two files a previous run left, is
# ours to empty.
FIXTURE_STAMP = ".claude-binder-contract-fixture-root"
PATH_FIELDS = {
    "sequence_path",
    "structure_path",
    "design_pose_path",
    "predicted_complex_path",
    "pae_path",
    "metric_source_path",
    "msa_path",
}
PATH_SHA256_FIELDS = {
    "sequence_path": "sequence_sha256",
    "structure_path": "structure_sha256",
    "design_pose_path": "design_pose_sha256",
    "predicted_complex_path": "predicted_complex_sha256",
    "pae_path": "pae_sha256",
    "metric_source_path": "metric_source_sha256",
    "msa_path": "msa_sha256",
}
STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}
KNOWN_CONTRACT_FIELDS = {
    "target_id",
    "target_sha256",
    "candidate_id",
    "origin_generator",
    "generator",
    "predictor",
    "model_revision",
    "seed",
    "phase",
    "sequence_path",
    "sequence_sha256",
    "sequence_length",
    "design_pose_path",
    "design_pose_sha256",
    "chain_mapping",
    "msa_path",
    "msa_sha256",
    "status",
    "raw_prediction_record_sha256",
    "filter_pass",
    "promotion_status",
    "promotion_reason",
    "ok",
    "controls",
    "portfolio",
    "ranked_candidates",
    "schema_version",
    "run_id",
    "campaign_id",
    "designs",
    "how_to_open",
    "warnings",
}
OPTIONAL_CONFIG_READS = frozenset({"selection.diversity"})
SPECIAL_INPUT_PRODUCERS = frozenset({"run-bundle", "stage-receipts"})


class ContractDryRunError(RuntimeError):
    """A fixture or contract could not be prepared."""


class ProviderClientStub:
    """Provider object that records accidental calls and fails closed."""

    def __init__(self, adapter_id: str) -> None:
        self.adapter_id = adapter_id
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, tuple(sorted(kwargs))))
        raise ContractDryRunError(
            f"provider client stub for {self.adapter_id} was called during contract preparation"
        )


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ContractDryRunError(f"could not read JSON {path}: {type(exc).__name__}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:  # noqa: BLE001
        raise ContractDryRunError(f"could not read JSONL {path}: {type(exc).__name__}: {exc}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ContractDryRunError(f"JSONL rows must be objects: {path}")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _source_artifact_root(plan: Mapping[str, Any], *, excluded_root: Path | None = None) -> Path:
    runtime = plan.get("runtime", {})
    value = runtime.get("run_root") if isinstance(runtime, Mapping) else None
    if not isinstance(value, str):
        raise ContractDryRunError("run plan runtime.run_root is missing")
    run_root = Path(value)
    candidates = [
        run_root.with_name(f"{run_root.name}.superseded-105153") / "artifacts",
        *sorted(run_root.parent.glob(f"{run_root.name}.superseded-*"), reverse=True),
        run_root / "artifacts",
    ]
    for candidate in candidates:
        artifact_root = candidate if candidate.name == "artifacts" else candidate / "artifacts"
        if excluded_root is not None and (
            _inside(artifact_root, excluded_root) or _inside(excluded_root, artifact_root)
        ):
            continue
        has_completed_downstream = (artifact_root / "filters" / "passing-candidates.jsonl").is_file()
        has_normalized_candidates = bool(
            list(
                artifact_root.glob(
                    "stages/normalize-candidates/attempts/*/*/candidate-manifest.jsonl"
                )
            )
        )
        if has_completed_downstream or has_normalized_candidates:
            return artifact_root.resolve()
    raise ContractDryRunError(
        "the completed downstream artifact tree or stage 5 candidate manifest is unavailable "
        "beside runtime.run_root"
    )


def _config_source(plan_path: Path, plan: Mapping[str, Any]) -> Path:
    bundled = plan_path.parent / "config.resolved.json"
    if bundled.is_file():
        return bundled
    context = plan.get("context", {})
    value = context.get("config_path") if isinstance(context, Mapping) else None
    if isinstance(value, str) and Path(value).is_file():
        return Path(value)
    raise ContractDryRunError("the resolved campaign config is missing beside the run plan")


def _copy_file(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise ContractDryRunError(f"fixture source file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination.resolve()


def _copy_record_path(
    value: str,
    source_artifact_root: Path,
    fixture_artifact_root: Path,
    *,
    label: str,
) -> str:
    recorded = Path(value).expanduser()
    if "artifacts" in recorded.parts:
        artifacts_index = recorded.parts.index("artifacts")
        relative = Path(*recorded.parts[artifacts_index + 1 :])
        if not relative.parts:
            raise ContractDryRunError(f"{label} recorded path has no artifact file: {value}")
        source = (source_artifact_root / relative).resolve()
        if not _inside(source, source_artifact_root):
            raise ContractDryRunError(f"{label} recorded path escapes the artifact tree: {value}")
    else:
        source = recorded.resolve()
        relative = Path("inputs") / source.name
    if not source.is_file():
        raise ContractDryRunError(f"{label} names a missing file: {source}")
    try:
        relative = source.resolve().relative_to(source_artifact_root.resolve())
    except ValueError:
        relative = Path("inputs") / source.name
    destination = fixture_artifact_root / relative
    return str(_copy_file(source, destination))


def _copy_row_paths(
    row: Mapping[str, Any],
    source_artifact_root: Path,
    fixture_artifact_root: Path,
    *,
    label: str,
    allow_fixture_digest_match: bool = False,
) -> dict[str, Any]:
    copied = dict(row)
    for field in PATH_FIELDS | {"structure_path"}:
        value = copied.get(field)
        if isinstance(value, str) and value:
            try:
                copied[field] = _copy_record_path(
                    value,
                    source_artifact_root,
                    fixture_artifact_root,
                    label=f"{label}.{field}",
                )
            except ContractDryRunError:
                digest_field = PATH_SHA256_FIELDS.get(field)
                digest = copied.get(digest_field) if digest_field is not None else None
                matched = None
                if (
                    allow_fixture_digest_match
                    and isinstance(digest, str)
                    and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                ):
                    matched = next(
                        (
                            candidate.resolve()
                            for candidate in sorted(fixture_artifact_root.rglob("*"))
                            if candidate.is_file() and _sha256(candidate) == digest
                        ),
                        None,
                    )
                if matched is None:
                    raise
                copied[field] = str(matched)
    return copied


def _copy_rows(
    source: Path,
    destination: Path,
    source_artifact_root: Path,
    fixture_artifact_root: Path,
    *,
    label: str,
    allow_fixture_digest_match: bool = False,
) -> list[dict[str, Any]]:
    rows = [
        _copy_row_paths(
            row,
            source_artifact_root,
            fixture_artifact_root,
            label=f"{label} row {index}",
            allow_fixture_digest_match=allow_fixture_digest_match,
        )
        for index, row in enumerate(_load_jsonl(source))
    ]
    _write_jsonl(destination, rows)
    return rows


def _stage_attempt_root(artifact_root: Path, stage_id: str) -> Path:
    return artifact_root / "stages" / stage_id / "attempts" / "fixture"


def _rendered_stage_output(
    artifact_root: Path,
    stage: Mapping[str, Any],
    artifact_id: str,
) -> Path:
    output = _output(stage, artifact_id)
    template = output.get("path_template")
    if not isinstance(template, str):
        raise ContractDryRunError(
            f"stage {stage.get('stage_id')} output {artifact_id} has no path template"
        )
    return _render_path(
        template,
        _stage_attempt_root(artifact_root, str(stage["stage_id"])),
        "single",
    )


def _write_stage_receipt(
    receipts_dir: Path,
    stage: Mapping[str, Any],
    files: Mapping[str, Path],
    records: Mapping[str, int],
) -> None:
    artifacts = [
        _receipt_artifact(
            artifact_id,
            str(_output(stage, artifact_id).get("artifact_type", artifact_id)),
            str(_output(stage, artifact_id).get("kind", "file")),
            path,
            records=int(records.get(artifact_id, 1)),
        )
        for artifact_id, path in files.items()
    ]
    _write_receipt(receipts_dir, str(stage["stage_id"]), artifacts)


def _rebase_config(
    config: dict[str, Any],
    source_config: Path,
    fixture_artifact_root: Path,
    source_artifact_root: Path | None = None,
) -> dict[str, Any]:
    value = copy.deepcopy(config)
    target = next(
        (item for item in value.get("targets", []) if isinstance(item, dict) and item.get("role") == "primary"),
        None,
    )
    if not isinstance(target, dict):
        raise ContractDryRunError("resolved config has no primary target")
    source_path_value = (
        target.get("runtime_structure_path")
        or target.get("structure_source_path")
        or target.get("structure_path")
    )
    source_path = Path(str(source_path_value)) if source_path_value else None
    if source_path is None or not source_path.is_file():
        source_path = source_config.parent / "inputs" / "targets" / "primary-target.cif"
    if source_path.suffix.lower() in {".cif", ".mmcif"} and source_artifact_root is not None:
        pdb_sources = sorted(
            source_artifact_root.glob(
                "stages/target-prepare/attempts/*/*/structures/primary-target.pdb"
            )
        )
        if pdb_sources:
            source_path = pdb_sources[0]
    target_path = _copy_file(source_path, fixture_artifact_root / "inputs" / "targets" / source_path.name)
    target["structure_path"] = str(target_path)
    target["structure_source_path"] = str(target_path)
    target["runtime_structure_path"] = str(target_path)
    site = target.get("site")
    if isinstance(site, dict):
        residue_value = site.get("runtime_residue_map_path") or site.get("residue_map_path")
        residue_source = Path(str(residue_value)) if residue_value else None
        if residue_source is None or not residue_source.is_file():
            residue_source = source_config.parent / "inputs" / "residue-maps" / "primary-target.json"
        residue_path = _copy_file(
            residue_source,
            fixture_artifact_root / "inputs" / "residue-maps" / residue_source.name,
        )
        site["runtime_residue_map_path"] = str(residue_path)
        site["residue_map_path"] = str(residue_path)
    controls = value.get("controls")
    if isinstance(controls, dict):
        for group in ("positive", "negative"):
            for entry in controls.get(group) or []:
                if not isinstance(entry, dict):
                    continue
                _rebase_control(entry, group, source_config, fixture_artifact_root)
    return value


def _rebase_control(
    entry: dict[str, Any],
    group: str,
    source_config: Path,
    fixture_artifact_root: Path,
) -> None:
    """Copy one configured control structure into the fixture and repoint the config.

    Only the path keys the entry already carries are rewritten. materialize_controls
    reads the absence of structure_path as the instruction to construct a decoy, so
    adding a key here would change which control the fixture builds.
    """
    for field in ("runtime_structure_path", "structure_source_path", "structure_path"):
        recorded = entry.get(field)
        if not isinstance(recorded, str) or not recorded:
            continue
        source = Path(recorded).expanduser()
        if not source.is_absolute():
            source = source_config.parent / source
        if not source.is_file():
            continue
        copied = _copy_file(
            source, fixture_artifact_root / "inputs" / "controls" / group / source.name
        )
        for target_field in ("runtime_structure_path", "structure_source_path", "structure_path"):
            if target_field in entry:
                entry[target_field] = str(copied)
        return


def _receipt_artifact(
    artifact_id: str,
    artifact_type: str,
    kind: str,
    path: Path,
    *,
    records: int,
    phase: str = "single",
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "kind": kind,
        "phase": phase,
        "files": [{"path": str(path.resolve()), "sha256": _sha256(path), "records": records}],
    }


def _write_receipt(
    receipts_dir: Path,
    stage_id: str,
    artifacts: list[dict[str, Any]],
) -> None:
    _write_json(
        receipts_dir / f"{stage_id}.json",
        {
            "ok": True,
            "stage_id": stage_id,
            "output_manifest": {"artifacts": artifacts},
        },
    )


def _stage(plan: Mapping[str, Any], stage_id: str) -> dict[str, Any]:
    for stage in plan.get("stages", []):
        if isinstance(stage, dict) and stage.get("stage_id") == stage_id:
            return stage
    raise ContractDryRunError(f"run plan has no stage {stage_id}")


def _optional_stage(plan: Mapping[str, Any], stage_id: str) -> dict[str, Any] | None:
    """Return a stage the plan may legitimately omit, or None."""
    for stage in plan.get("stages", []):
        if isinstance(stage, dict) and stage.get("stage_id") == stage_id:
            return stage
    return None


def _output(stage: Mapping[str, Any], artifact_id: str) -> dict[str, Any]:
    for output in stage.get("outputs", []):
        if isinstance(output, dict) and output.get("artifact_id") == artifact_id:
            return output
    raise ContractDryRunError(f"stage {stage.get('stage_id')} has no output {artifact_id}")


def _adapter(plan: Mapping[str, Any], adapter_id: str) -> dict[str, Any]:
    for adapter in plan.get("adapters", []):
        if isinstance(adapter, dict) and adapter.get("adapter_id") == adapter_id:
            return adapter
    raise ContractDryRunError(f"run plan has no adapter {adapter_id}")


def _render_path(template: str, attempt_dir: Path, phase: str) -> Path:
    rendered = template.replace("{{attempt_dir}}", str(attempt_dir)).replace("{{phase}}", phase)
    if "{{" in rendered:
        raise ContractDryRunError(f"unresolved output path token: {template}")
    return Path(rendered)


def _downstream_field_names(plan: Mapping[str, Any], artifact_id: str) -> set[str]:
    fields: set[str] = set()
    for stage in plan.get("stages", []):
        if not isinstance(stage, dict):
            continue
        inputs = stage.get("inputs", [])
        if not any(isinstance(value, str) and value.endswith(f":{artifact_id}") for value in inputs):
            continue
        adapter = _adapter(plan, str(stage.get("adapter_id")))
        for argv_key in ("command_argv_template", "parser_argv_template"):
            argv = adapter.get(argv_key, [])
            if not isinstance(argv, list):
                continue
            module = next(
                (argv[index + 1] for index, token in enumerate(argv[:-1]) if token == "-m"),
                None,
            )
            if not isinstance(module, str):
                module = "claude_binder.lane"
            closure = contract_audit.module_closure(module, package_root(), depth=1)
            source = "\n".join(path.read_text(encoding="utf-8") for path in closure.values())
            for field in KNOWN_CONTRACT_FIELDS:
                if re.search(rf"[\"']{re.escape(field)}[\"']", source):
                    fields.add(field)
    return fields


def _validate_rows(
    plan: Mapping[str, Any],
    stage: Mapping[str, Any],
    artifact_id: str,
    value: Any,
    *,
    label: str,
) -> list[str]:
    output = _output(stage, artifact_id)
    rows = value if output.get("kind") == "jsonl" else [value]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return [f"{label} must contain JSON objects"]
    errors: list[str] = []
    required = [str(field) for field in output.get("required_fields", [])]
    downstream = _downstream_field_names(plan, artifact_id).intersection(
        KNOWN_CONTRACT_FIELDS
    )
    downstream_fields = set(required)
    if artifact_id == "esmfold2-fast-rescore" and "origin_generator" in downstream:
        downstream_fields.add("origin_generator")
    downstream = downstream.intersection(downstream_fields)
    for index, row in enumerate(rows):
        for field in required:
            if field not in row:
                errors.append(f"{label} row {index} missing declared field {field}")
        for field in sorted(downstream):
            if field not in row:
                errors.append(f"{label} row {index} missing downstream field {field}")
        schema_name = output.get("schema_path")
        if isinstance(schema_name, str):
            schema = _load(schema_file(schema_name))
            errors.extend(
                f"{label} row {index} schema: {error}"
                for error in _lane().validate_json_schema(row, schema)
            )
    return errors


def _lane() -> Any:
    from . import lane

    return lane


def _assert_paths(value: Any, fixture_root: Path, *, label: str) -> list[str]:
    errors: list[str] = []

    def walk(item: Any, location: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in PATH_FIELDS or key.endswith("_path"):
                    if isinstance(child, str) and child:
                        path = Path(child).expanduser().resolve()
                        if not _inside(path, fixture_root):
                            errors.append(f"{label} {location}.{key} resolves outside fixture: {path}")
                        elif not path.is_file():
                            errors.append(f"{label} {location}.{key} is missing: {path}")
                walk(child, f"{location}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{location}[{index}]")

    walk(value, "$")
    return errors


def _assert_self_reading(plan: Mapping[str, Any], stage: Mapping[str, Any]) -> list[str]:
    outputs = {
        str(output.get("artifact_id"))
        for output in stage.get("outputs", [])
        if isinstance(output, dict) and output.get("artifact_id")
    }
    errors: list[str] = []
    for input_ref in stage.get("inputs", []):
        if not isinstance(input_ref, str) or ":" not in input_ref:
            continue
        _, artifact_id = input_ref.split(":", 1)
        if artifact_id in outputs:
            output = _output(stage, artifact_id)
            errors.append(
                f"stage {stage.get('stage_id')} reads its own published artifact {artifact_id} "
                f"at {output.get('publish_path') or output.get('path_template')}"
            )
    return errors


def _config_errors(plan: Mapping[str, Any], config: Mapping[str, Any], stage: Mapping[str, Any]) -> list[str]:
    adapter = _adapter(plan, str(stage.get("adapter_id")))
    reads = contract_audit._config_reads_for_stage(stage, adapter, package_root())
    sources = contract_audit._shipped_config_sources(package_root() / "data" / "templates")
    errors: list[str] = []
    for read in reads:
        dotted = str(read.dotted)
        if dotted in OPTIONAL_CONFIG_READS or "{" in dotted or "}" in dotted:
            continue
        if not contract_audit._config_read_is_reachable(config, dotted):
            continue
        if not contract_audit._config_has(config, dotted):
            errors.append(
                f"stage {stage.get('stage_id')} reads undeclared config key {dotted} "
                f"at {read.file}:{read.line}"
            )
        if not contract_audit._config_key_covered(dotted, sources):
            errors.append(
                f"stage {stage.get('stage_id')} reads config key {dotted} with no shipped profile key "
                f"at {read.file}:{read.line}"
            )
    return errors


def _format_errors(rows: Iterable[Mapping[str, Any]], *, label: str) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(rows):
        for field, value in row.items():
            if not field.endswith("_path") or not isinstance(value, str) or not value:
                continue
            path = Path(value)
            if path.suffix.lower() in STRUCTURE_SUFFIXES:
                if path.suffix.lower() in {".cif", ".mmcif"}:
                    text = path.read_text(errors="replace") if path.is_file() else ""
                    if "_atom_site." not in text:
                        errors.append(f"{label} row {index} parser rejected mmCIF fixture {path}")
                else:
                    if not path.is_file() or not any(line.startswith(("ATOM", "HETATM", "SEQRES")) for line in path.read_text(errors="replace").splitlines()):
                        errors.append(f"{label} row {index} parser rejected PDB fixture {path}")
    return errors


def _normalized_candidate_source(source_root: Path) -> Path:
    published = source_root / "candidates" / "candidate-manifest.jsonl"
    if published.is_file():
        return published
    matches = sorted(
        source_root.glob("stages/normalize-candidates/attempts/*/*/candidate-manifest.jsonl")
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ContractDryRunError(f"normalized candidate manifest is missing under {source_root}")
    raise ContractDryRunError(
        "normalized candidate manifest is ambiguous under "
        f"{source_root}: {', '.join(str(path) for path in matches)}"
    )


def _filter_rows(
    candidates: Iterable[Mapping[str, Any]],
    contracts: Iterable[Mapping[str, Any]],
    *,
    round_number: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for contract in contracts:
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "origin_generator": candidate["origin_generator"],
                    "sequence_sha256": candidate["sequence_sha256"],
                    "optimization_round": round_number,
                    "filter_id": contract["filter_id"],
                    "metric": contract["metric"],
                    "operator": contract["operator"],
                    "threshold": contract["threshold"],
                    "value": contract["threshold"],
                    "pass": True,
                    "reason": "offline contract fixture marks the configured check as passing",
                    "tool_revision": contract["tool_revision"],
                    "reference_revision": contract["reference_revision"],
                    "reference_sha256": contract["reference_sha256"],
                }
            )
    return rows


def _prepare_from_normalized_candidates(
    config: dict[str, Any],
    plan: Mapping[str, Any],
    source_root: Path,
    artifact_root: Path,
    candidate_source: Path,
) -> dict[str, Any]:
    """Build only the missing downstream fixtures from the real stage-5 output.

    A stopped run has no stage-9 artifacts to copy. Its candidate manifest and
    referenced sequence, pose, and target files are still enough to exercise every
    later contract. The rows after normalize-candidates are deterministic offline
    fixtures. Provider-backed prediction preparation still receives a fail-closed
    stub in the stage runner.
    """
    from . import fixture_adapter

    normalize_stage = _stage(plan, "normalize-candidates")
    normalized_path = _rendered_stage_output(
        artifact_root, normalize_stage, "normalized-candidates"
    )
    candidate_rows = _copy_rows(
        candidate_source,
        normalized_path,
        source_root,
        artifact_root,
        label="candidate",
    )
    _write_stage_receipt(
        artifact_root / "receipts",
        normalize_stage,
        {"normalized-candidates": normalized_path},
        {"normalized-candidates": len(candidate_rows)},
    )

    # A supplied-candidate profile enables no sequence designer, because candidates
    # arrive with their sequences already fixed, so the plan carries no
    # sequence-proteinmpnn stage. Requiring one refused the contract dry run for
    # the profile that most needs a free check before a paid dispatch.
    sequence_stage = _optional_stage(plan, "sequence-proteinmpnn")
    if sequence_stage is not None:
        sequence_path = (
            _stage_attempt_root(artifact_root, "sequence-proteinmpnn")
            / "single"
            / "sequence-candidate-manifest.jsonl"
        )
        _write_jsonl(sequence_path, candidate_rows)
        _write_stage_receipt(
            artifact_root / "receipts",
            sequence_stage,
            {"proteinmpnn-candidates": sequence_path},
            {"proteinmpnn-candidates": len(candidate_rows)},
        )

    filters = config.get("filters", {})
    disabled = {
        value for value in filters.get("disabled_checks", []) if isinstance(value, str)
    }
    contracts = [
        contract
        for contract in filters.get("contracts", [])
        if isinstance(contract, dict) and contract.get("filter_id") not in disabled
    ]
    integrity_contracts = [
        contract for contract in contracts if contract.get("stage_id") == "filter-integrity"
    ]
    integrity_stage = _stage(plan, "filter-integrity")
    integrity_observations = _filter_rows(
        candidate_rows,
        integrity_contracts,
        round_number=0,
    )
    integrity_observation_path = _rendered_stage_output(
        artifact_root, integrity_stage, "integrity-filter-observations"
    )
    integrity_passing_path = _rendered_stage_output(
        artifact_root, integrity_stage, "integrity-passing-candidates"
    )
    _write_jsonl(integrity_observation_path, integrity_observations)
    integrity_passing = [
        {**row, "filter_pass": True, "failed_checks": []} for row in candidate_rows
    ]
    _write_jsonl(integrity_passing_path, integrity_passing)
    _write_stage_receipt(
        artifact_root / "receipts",
        integrity_stage,
        {
            "integrity-filter-observations": integrity_observation_path,
            "integrity-passing-candidates": integrity_passing_path,
        },
        {
            "integrity-filter-observations": len(integrity_observations),
            "integrity-passing-candidates": len(integrity_passing),
        },
    )

    novelty_stage = _stage(plan, "filter-novelty")
    novelty_contracts = [
        contract for contract in contracts if contract.get("stage_id") == "filter-novelty"
    ]
    passing_path = _rendered_stage_output(artifact_root, novelty_stage, "passing-candidates")
    passing_rows = [
        {**row, "filter_pass": True, "failed_checks": []} for row in integrity_passing
    ]
    novelty_files: dict[str, Path] = {"passing-candidates": passing_path}
    novelty_records = {"passing-candidates": len(passing_rows)}
    _write_jsonl(passing_path, passing_rows)
    if novelty_contracts:
        novelty_observation_path = _rendered_stage_output(
            artifact_root, novelty_stage, "novelty-filter-observations"
        )
        novelty_observations = _filter_rows(
            passing_rows,
            novelty_contracts,
            round_number=0,
        )
        _write_jsonl(novelty_observation_path, novelty_observations)
        novelty_files["novelty-filter-observations"] = novelty_observation_path
        novelty_records["novelty-filter-observations"] = len(novelty_observations)
    _write_stage_receipt(
        artifact_root / "receipts",
        novelty_stage,
        novelty_files,
        novelty_records,
    )

    target = next(item for item in config["targets"] if item.get("role") == "primary")
    predictor = next(
        item for item in config["cofold"]["predictors"] if item.get("enabled", True)
    )
    screen_stage = _stage(plan, "cofold-screen-esmfold2-fast")
    screen_attempt = _stage_attempt_root(artifact_root, "cofold-screen-esmfold2-fast")
    screen_raw_rows = [
        fixture_adapter.raw_prediction_row(
            config,
            artifact_root=artifact_root,
            target=target,
            candidate_id=str(row["candidate_id"]),
            predictor=predictor,
            seed=int(config["cofold"]["screen_seeds"][0]),
            phase="screen",
            sequence_sha256=str(row["sequence_sha256"]),
            design_pose_path=str(row["design_pose_path"]),
            design_pose_sha256=str(row["design_pose_sha256"]),
            origin_generator=str(row["origin_generator"]),
        )
        for row in passing_rows
    ]
    screen_raw_rows = fixture_adapter.attach_raw_artifacts(
        config, screen_raw_rows, screen_attempt / "single"
    )
    screen_path = _rendered_stage_output(
        artifact_root, screen_stage, "esmfold2-fast-screen"
    )
    _write_jsonl(screen_path, screen_raw_rows)
    _write_stage_receipt(
        artifact_root / "receipts",
        screen_stage,
        {"esmfold2-fast-screen": screen_path},
        {"esmfold2-fast-screen": len(screen_raw_rows)},
    )

    score_stage = _stage(plan, "score-screen")
    screen_rows = [
        {
            **fixture_adapter.observation_from_raw(
                row,
                attempt_id="offline-contract-screen-score",
                filter_pass=True,
            ),
            "origin_generator": row["origin_generator"],
        }
        for row in screen_raw_rows
    ]
    score_path = _rendered_stage_output(artifact_root, score_stage, "screen-score-table")
    _write_jsonl(score_path, screen_rows)
    _write_stage_receipt(
        artifact_root / "receipts",
        score_stage,
        {"screen-score-table": score_path},
        {"screen-score-table": len(screen_rows)},
    )
    return {
        "candidate_rows": candidate_rows,
        "integrity_observations": integrity_observations,
        "integrity_passing_rows": integrity_passing,
        "novelty_observations": novelty_observations if novelty_contracts else [],
        "passing_rows": passing_rows,
        "screen_rows": screen_rows,
        "screen_raw_rows": screen_raw_rows,
        "source_mode": "real normalize-candidates artifact plus offline downstream fixtures",
    }


def _prepare_from_completed_downstream(
    plan: Mapping[str, Any],
    source_root: Path,
    artifact_root: Path,
    candidate_source: Path,
) -> dict[str, Any]:
    """Copy completed non-provider artifacts into the contract fixture."""
    normalize_stage = _stage(plan, "normalize-candidates")
    normalized_path = _rendered_stage_output(
        artifact_root, normalize_stage, "normalized-candidates"
    )
    candidate_rows = _copy_rows(
        candidate_source,
        normalized_path,
        source_root,
        artifact_root,
        label="candidate",
    )
    _write_stage_receipt(
        artifact_root / "receipts",
        normalize_stage,
        {"normalized-candidates": normalized_path},
        {"normalized-candidates": len(candidate_rows)},
    )

    # A supplied-candidate profile enables no sequence designer, because candidates
    # arrive with their sequences already fixed, so the plan carries no
    # sequence-proteinmpnn stage. Requiring one refused the contract dry run for
    # the profile that most needs a free check before a paid dispatch.
    sequence_stage = _optional_stage(plan, "sequence-proteinmpnn")
    if sequence_stage is not None:
        sequence_path = (
            _stage_attempt_root(artifact_root, "sequence-proteinmpnn")
            / "single"
            / "sequence-candidate-manifest.jsonl"
        )
        _write_jsonl(sequence_path, candidate_rows)
        _write_stage_receipt(
            artifact_root / "receipts",
            sequence_stage,
            {"proteinmpnn-candidates": sequence_path},
            {"proteinmpnn-candidates": len(candidate_rows)},
        )

    def copy_rows(
        stage_id: str,
        artifact_id: str,
        source: Path,
        label: str,
    ) -> list[dict[str, Any]]:
        stage = _stage(plan, stage_id)
        destination = _rendered_stage_output(artifact_root, stage, artifact_id)
        rows = _copy_rows(source, destination, source_root, artifact_root, label=label)
        _write_stage_receipt(
            artifact_root / "receipts",
            stage,
            {artifact_id: destination},
            {artifact_id: len(rows)},
        )
        return rows

    integrity_observations = copy_rows(
        "filter-integrity",
        "integrity-filter-observations",
        source_root / "filters" / "integrity-filter-observations.jsonl",
        "integrity observation",
    )
    integrity_passing = copy_rows(
        "filter-integrity",
        "integrity-passing-candidates",
        source_root / "filters" / "integrity-passing-candidates.jsonl",
        "integrity passing",
    )
    novelty_stage = _stage(plan, "filter-novelty")
    novelty_files: dict[str, Path] = {}
    novelty_records: dict[str, int] = {}
    novelty_observations: list[dict[str, Any]] = []
    if any(
        output.get("artifact_id") == "novelty-filter-observations"
        for output in novelty_stage.get("outputs", [])
        if isinstance(output, dict)
    ):
        novelty_path = _rendered_stage_output(
            artifact_root, novelty_stage, "novelty-filter-observations"
        )
        novelty_observations = _copy_rows(
            source_root / "filters" / "novelty-filter-observations.jsonl",
            novelty_path,
            source_root,
            artifact_root,
            label="novelty observation",
        )
        novelty_files["novelty-filter-observations"] = novelty_path
        novelty_records["novelty-filter-observations"] = len(novelty_observations)
    passing_path = _rendered_stage_output(artifact_root, novelty_stage, "passing-candidates")
    passing_rows = _copy_rows(
        source_root / "filters" / "passing-candidates.jsonl",
        passing_path,
        source_root,
        artifact_root,
        label="passing",
    )
    novelty_files["passing-candidates"] = passing_path
    novelty_records["passing-candidates"] = len(passing_rows)
    _write_stage_receipt(
        artifact_root / "receipts", novelty_stage, novelty_files, novelty_records
    )

    score_stage = _stage(plan, "score-screen")
    screen_rows = _copy_rows(
        source_root / "scores" / "screen-score-table.jsonl",
        _rendered_stage_output(artifact_root, score_stage, "screen-score-table"),
        source_root,
        artifact_root,
        label="screen score",
    )
    score_path = _rendered_stage_output(artifact_root, score_stage, "screen-score-table")
    _write_stage_receipt(
        artifact_root / "receipts",
        score_stage,
        {"screen-score-table": score_path},
        {"screen-score-table": len(screen_rows)},
    )

    raw_sources = sorted(
        source_root.glob(
            "stages/cofold-screen-esmfold2-fast/attempts/*/scale/cofold-observations.jsonl"
        )
    ) or sorted(
        source_root.glob(
            "stages/cofold-screen-esmfold2-fast/attempts/*/*/cofold-observations.jsonl"
        )
    )
    if not raw_sources:
        raise ContractDryRunError("screen raw prediction manifest is missing")
    screen_stage = _stage(plan, "cofold-screen-esmfold2-fast")
    screen_path = _rendered_stage_output(
        artifact_root, screen_stage, "esmfold2-fast-screen"
    )
    screen_raw_rows = _copy_rows(
        raw_sources[0],
        screen_path,
        source_root,
        artifact_root,
        label="screen raw",
    )
    _write_stage_receipt(
        artifact_root / "receipts",
        screen_stage,
        {"esmfold2-fast-screen": screen_path},
        {"esmfold2-fast-screen": len(screen_raw_rows)},
    )
    return {
        "candidate_rows": candidate_rows,
        "integrity_observations": integrity_observations,
        "integrity_passing_rows": integrity_passing,
        "novelty_observations": novelty_observations,
        "passing_rows": passing_rows,
        "screen_rows": screen_rows,
        "screen_raw_rows": screen_raw_rows,
        "source_mode": "completed downstream artifacts",
    }


def _import_source_receipt(
    source_root: Path,
    artifact_root: Path,
    receipts_dir: Path,
    stage_id: str,
) -> bool:
    """Carry one upstream receipt into the fixture with its files.

    The dry run starts partway down the graph, so the stages above it never write a
    receipt of their own. Downstream stages still resolve their declared inputs through
    those receipts, and without them a real contract error is indistinguishable from a
    missing fixture.
    """
    source_receipt = source_root / "receipts" / f"{stage_id}.json"
    if not source_receipt.is_file():
        return False
    try:
        receipt = _load(source_receipt)
    except ContractDryRunError:
        return False
    artifacts = receipt.get("output_manifest", {}).get("artifacts", [])
    if not isinstance(artifacts, list):
        return False
    carried: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        files: list[dict[str, Any]] = []
        for entry in artifact.get("files", []):
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                continue
            try:
                copied = Path(
                    _copy_record_path(
                        entry["path"], source_root, artifact_root, label=f"{stage_id} receipt"
                    )
                )
            except ContractDryRunError:
                continue
            files.append({**entry, "path": str(copied), "sha256": _sha256(copied)})
        if files:
            carried.append({**artifact, "files": files})
    if not carried:
        return False
    _write_json(
        receipts_dir / f"{stage_id}.json",
        {"ok": True, "stage_id": stage_id, "output_manifest": {"artifacts": carried}},
    )
    return True


def _declared_upstream_artifacts(
    plan: Mapping[str, Any], from_stage: str
) -> dict[str, set[str]]:
    """Return artifact inputs that cross the requested start boundary.

    Stages at or below ``from_stage`` run in the fixture. Their producers above
    that boundary do not run, so those exact declared artifacts must come from
    the source run. This is an artifact-edge closure rather than a fixed list of
    stage names.
    """
    ordered = plan.get("ordered_stage_ids")
    if not isinstance(ordered, list) or from_stage not in ordered:
        raise ContractDryRunError(
            f"--from must name a stage in ordered_stage_ids: {from_stage}"
        )
    selected = {str(stage_id) for stage_id in ordered[ordered.index(from_stage) :]}
    required: dict[str, set[str]] = {}
    for stage_id in ordered[ordered.index(from_stage) :]:
        stage = _stage(plan, str(stage_id))
        for input_ref in stage.get("inputs", []):
            if not isinstance(input_ref, str) or ":" not in input_ref:
                continue
            producer, artifact_id = input_ref.split(":", 1)
            if producer in SPECIAL_INPUT_PRODUCERS or producer in selected:
                continue
            required.setdefault(producer, set()).add(artifact_id)
    return required


def _required_file_sha256(entry: Mapping[str, Any], *, label: str) -> str:
    recorded = entry.get("sha256")
    if not isinstance(recorded, str) or re.fullmatch(r"[0-9a-f]{64}", recorded) is None:
        raise ContractDryRunError(f"{label} has no valid sha256")
    return recorded


def _available_receipt_artifacts(
    receipt_path: Path,
    *,
    expected_stage_id: str,
    required_artifact_ids: set[str],
) -> set[str]:
    """Return required artifact IDs backed by valid fixture receipt evidence."""
    if not receipt_path.is_file():
        return set()
    receipt = _load(receipt_path)
    if receipt.get("stage_id") != expected_stage_id:
        raise ContractDryRunError(
            f"fixture receipt stage_id does not match {expected_stage_id}: {receipt_path}"
        )
    if receipt.get("ok") is not True:
        raise ContractDryRunError(
            f"fixture receipt is not successful for stage {expected_stage_id}: {receipt_path}"
        )
    available: set[str] = set()
    for artifact in receipt.get("output_manifest", {}).get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        artifact_id = artifact.get("artifact_id")
        if artifact_id not in required_artifact_ids:
            continue
        files = artifact.get("files")
        if not isinstance(artifact_id, str) or not isinstance(files, list) or not files:
            continue
        for entry in files:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ContractDryRunError(
                    f"fixture artifact {expected_stage_id}:{artifact_id} has an invalid file record"
                )
            path = Path(entry["path"])
            if not path.is_file():
                break
            recorded_sha = _required_file_sha256(
                entry, label=f"fixture artifact {expected_stage_id}:{artifact_id} file {path}"
            )
            if _sha256(path) != recorded_sha:
                raise ContractDryRunError(
                    f"fixture artifact {expected_stage_id}:{artifact_id} hash does not match "
                    f"its receipt: {path}"
                )
        else:
            available.add(artifact_id)
    return available


def _import_required_source_artifacts(
    source_root: Path,
    artifact_root: Path,
    receipts_dir: Path,
    stage_id: str,
    required_artifact_ids: set[str],
) -> None:
    """Copy required source-run artifacts and preserve their receipt evidence."""
    source_receipt_path = source_root / "receipts" / f"{stage_id}.json"
    if not source_receipt_path.is_file():
        raise ContractDryRunError(
            f"required source receipt is missing for stage {stage_id}: {source_receipt_path}"
        )
    receipt = _load(source_receipt_path)
    if receipt.get("stage_id") != stage_id:
        raise ContractDryRunError(
            f"required source receipt stage_id does not match {stage_id}: {source_receipt_path}"
        )
    if receipt.get("ok") is not True:
        raise ContractDryRunError(
            f"required source receipt is not successful for stage {stage_id}: {source_receipt_path}"
        )
    artifacts = receipt.get("output_manifest", {}).get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ContractDryRunError(
            f"required source receipt has no artifact manifest for stage {stage_id}: "
            f"{source_receipt_path}"
        )
    by_id: dict[str, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        if isinstance(artifact, dict) and isinstance(artifact.get("artifact_id"), str):
            by_id.setdefault(str(artifact["artifact_id"]), []).append(artifact)

    carried: list[dict[str, Any]] = []
    for artifact_id in sorted(required_artifact_ids):
        matches = by_id.get(artifact_id, [])
        if not matches:
            raise ContractDryRunError(
                f"required source artifact {stage_id}:{artifact_id} is absent from "
                f"{source_receipt_path}"
            )
        for artifact in matches:
            entries = artifact.get("files")
            if not isinstance(entries, list) or not entries:
                raise ContractDryRunError(
                    f"required source artifact {stage_id}:{artifact_id} has no files in "
                    f"{source_receipt_path}"
                )
            copied_files: list[dict[str, Any]] = []
            copied_published_path: str | None = None
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                    raise ContractDryRunError(
                        f"required source artifact {stage_id}:{artifact_id} has an invalid file record"
                    )
                try:
                    copied = Path(
                        _copy_record_path(
                            entry["path"],
                            source_root,
                            artifact_root,
                            label=f"required source artifact {stage_id}:{artifact_id}",
                        )
                    )
                except ContractDryRunError as exc:
                    raise ContractDryRunError(
                        f"required source artifact {stage_id}:{artifact_id} could not be copied: {exc}"
                    ) from exc
                recorded_sha = _required_file_sha256(
                    entry,
                    label=(
                        f"required source artifact {stage_id}:{artifact_id} "
                        f"file {entry['path']}"
                    ),
                )
                copied_source_sha = _sha256(copied)
                if recorded_sha != copied_source_sha:
                    raise ContractDryRunError(
                        f"required source artifact {stage_id}:{artifact_id} hash does not match "
                        f"its source receipt: {entry['path']}"
                    )
                if artifact.get("kind") == "jsonl" or copied.suffix.lower() == ".jsonl":
                    try:
                        _copy_rows(
                            copied,
                            copied,
                            source_root,
                            artifact_root,
                            label=f"required source artifact {stage_id}:{artifact_id}",
                            allow_fixture_digest_match=True,
                        )
                    except ContractDryRunError as exc:
                        raise ContractDryRunError(
                            f"required source artifact {stage_id}:{artifact_id} contains an "
                            f"unavailable referenced file: {exc}"
                        ) from exc
                copied_sha = _sha256(copied)
                copied_entry = {**entry, "path": str(copied), "sha256": copied_sha}
                published_value = entry.get("published_path")
                if not isinstance(published_value, str) and len(entries) == 1:
                    published_value = artifact.get("published_path")
                if isinstance(published_value, str) and published_value:
                    try:
                        published = Path(
                            _copy_record_path(
                                published_value,
                                source_root,
                                artifact_root,
                                label=(
                                    f"required published source artifact "
                                    f"{stage_id}:{artifact_id}"
                                ),
                            )
                        )
                    except ContractDryRunError as exc:
                        raise ContractDryRunError(
                            f"required published source artifact {stage_id}:{artifact_id} "
                            f"could not be copied: {exc}"
                        ) from exc
                    if _sha256(published) != recorded_sha:
                        raise ContractDryRunError(
                            f"required published source artifact {stage_id}:{artifact_id} hash "
                            f"does not match its source receipt: {published_value}"
                        )
                    _copy_file(copied, published)
                    copied_entry["published_path"] = str(published)
                    copied_published_path = str(published)
                copied_files.append(copied_entry)
            carried_artifact = {**artifact, "files": copied_files}
            if copied_published_path is not None:
                carried_artifact["published_path"] = copied_published_path
            carried.append(carried_artifact)

    destination = receipts_dir / f"{stage_id}.json"
    preserved: list[dict[str, Any]] = []
    if destination.is_file():
        existing = _load(destination)
        preserved = [
            artifact
            for artifact in existing.get("output_manifest", {}).get("artifacts", [])
            if isinstance(artifact, dict)
            and artifact.get("artifact_id") not in required_artifact_ids
        ]
    _write_json(
        destination,
        {
            "ok": True,
            "stage_id": stage_id,
            "output_manifest": {"artifacts": [*preserved, *carried]},
        },
    )


def _import_from_stage_inputs(state: dict[str, Any], from_stage: str) -> list[str]:
    """Satisfy every declared artifact edge that crosses ``--from``."""
    imported: list[str] = []
    requirements = _declared_upstream_artifacts(state["plan"], from_stage)
    for stage_id, artifact_ids in sorted(requirements.items()):
        receipt_path = state["receipts_dir"] / f"{stage_id}.json"
        missing = artifact_ids - _available_receipt_artifacts(
            receipt_path,
            expected_stage_id=stage_id,
            required_artifact_ids=artifact_ids,
        )
        if not missing:
            continue
        _import_required_source_artifacts(
            state["source_root"],
            state["artifact_root"],
            state["receipts_dir"],
            stage_id,
            missing,
        )
        imported.append(stage_id)
    return imported


def _is_our_fixture_root(fixture_root: Path) -> bool:
    """True when this directory is empty, stamped, or holds what a previous run left behind."""
    if fixture_root.is_symlink() or not fixture_root.is_dir():
        return False
    entries = list(fixture_root.iterdir())
    if not entries:
        return True
    if (fixture_root / FIXTURE_STAMP).is_file():
        return True
    return (fixture_root / "config.resolved.json").is_file() and (
        fixture_root / "run-plan.json"
    ).is_file()


def _prepare_fixture(
    plan_path: Path,
    plan: Mapping[str, Any],
    fixture_root: Path,
    *,
    source_artifact_root: Path | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    source_root = (
        source_artifact_root.expanduser().resolve()
        if source_artifact_root is not None
        else _source_artifact_root(plan, excluded_root=fixture_root)
    )
    if not source_root.is_dir():
        raise ContractDryRunError(f"--artifact-root is not a directory: {source_root}")
    if _inside(source_root, fixture_root) or _inside(fixture_root, source_root):
        raise ContractDryRunError(
            "--artifact-root and --fixture-root must be separate trees"
        )
    source_config_path = _config_source(plan_path, plan)
    source_config = _load(source_config_path)
    if not isinstance(source_config, dict):
        raise ContractDryRunError("resolved config is not an object")
    run_root = fixture_root / "run"
    artifact_root = run_root / "artifacts"
    receipts_dir = artifact_root / "receipts"
    if fixture_root.exists():
        if not replace and not _is_our_fixture_root(fixture_root):
            raise ContractDryRunError(
                f"--fixture-root is emptied recursively before the run, and {fixture_root} holds "
                "a directory this tool did not create. Name a path that does not exist, or pass "
                "--replace to delete what is there."
            )
        shutil.rmtree(fixture_root)
    artifact_root.mkdir(parents=True)
    (fixture_root / FIXTURE_STAMP).write_text(
        "This directory is emptied by claude_binder.contract_dry_run on every run.\n",
        encoding="utf-8",
    )

    config = _rebase_config(
        source_config,
        source_config_path,
        artifact_root,
        source_artifact_root=source_root,
    )
    _lane().bind_dynamic_stage_contracts(config)
    config_path = fixture_root / "config.resolved.json"
    _write_json(config_path, config)
    fixture_plan = copy.deepcopy(dict(plan))
    fixture_plan.setdefault("runtime", {})["run_root"] = str(run_root)
    config_stages = {
        str(stage.get("stage_id")): stage
        for stage in config.get("stages", [])
        if isinstance(stage, dict) and stage.get("stage_id")
    }
    for stage in fixture_plan.get("stages", []):
        if not isinstance(stage, dict):
            continue
        config_stage = config_stages.get(str(stage.get("stage_id")))
        if config_stage is not None:
            stage["outputs"] = copy.deepcopy(config_stage.get("outputs", []))
    fixture_plan_path = fixture_root / "run-plan.json"
    _write_json(fixture_plan_path, fixture_plan)
    _write_json(run_root / "runtime-config.resolved.json", config)
    _write_json(run_root / "status.json", {"run_id": config.get("run_id"), "campaign_id": config.get("campaign_id")})

    imported_receipts = [
        stage_id
        for stage_id in ("target-prepare", "runtime-check")
        if _import_source_receipt(source_root, artifact_root, receipts_dir, stage_id)
    ]

    candidate_source = _normalized_candidate_source(source_root)
    if (source_root / "filters" / "passing-candidates.jsonl").is_file():
        prepared = _prepare_from_completed_downstream(
            fixture_plan,
            source_root,
            artifact_root,
            candidate_source,
        )
    else:
        prepared = _prepare_from_normalized_candidates(
            config,
            fixture_plan,
            source_root,
            artifact_root,
            candidate_source,
        )
    return {
        "fixture_root": fixture_root,
        "run_root": run_root,
        "artifact_root": artifact_root,
        "receipts_dir": receipts_dir,
        "config_path": config_path,
        "plan_path": fixture_plan_path,
        "config": config,
        "plan": fixture_plan,
        "source_root": source_root,
        "imported_receipts": imported_receipts,
        **prepared,
    }


def _stage_paths_and_config(state: Mapping[str, Any], stage: Mapping[str, Any]) -> list[str]:
    errors = _assert_self_reading(state["plan"], stage)
    errors.extend(_config_errors(state["plan"], state["config"], stage))
    for input_ref in stage.get("inputs", []):
        if not isinstance(input_ref, str) or ":" not in input_ref:
            continue
        producer, artifact_id = input_ref.split(":", 1)
        if producer in SPECIAL_INPUT_PRODUCERS:
            # The run bundle hands these to the stage directly, so no producing stage
            # writes a receipt for them and input_files has nothing to resolve.
            continue
        try:
            _, paths = input_files(
                state["plan"],
                state["receipts_dir"],
                str(stage["stage_id"]),
                artifact_id=artifact_id,
                source_stage_id=producer,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"stage {stage['stage_id']} input {producer}:{artifact_id} failed: {exc}")
            continue
        errors.extend(_assert_paths([str(path) for path in paths], state["fixture_root"], label=f"stage {stage['stage_id']} input"))
    for output in stage.get("outputs", []):
        if isinstance(output, dict) and isinstance(output.get("path_template"), str):
            attempt = state["artifact_root"] / "stages" / str(stage["stage_id"]) / "attempts" / "fixture"
            path = _render_path(str(output["path_template"]), attempt, "single")
            if not _inside(path, state["fixture_root"]):
                errors.append(f"stage {stage['stage_id']} output path escapes fixture: {path}")
    return errors


def _normalize_candidates(
    state: dict[str, Any], stage: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    errors = _stage_paths_and_config(state, stage)
    rows = state["candidate_rows"]
    errors.extend(
        _validate_rows(
            state["plan"],
            stage,
            "normalized-candidates",
            rows,
            label="normalize output",
        )
    )
    errors.extend(_assert_paths(rows, state["fixture_root"], label="normalize output"))
    errors.extend(_format_errors(rows, label="normalize parser"))
    return errors, {
        "records": len(rows),
        "provider_calls": 0,
        "producer": "real stage-5 candidate-manifest.jsonl",
    }


def _filter_integrity(
    state: dict[str, Any], stage: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    errors = _stage_paths_and_config(state, stage)
    observations = state["integrity_observations"]
    passing = state["integrity_passing_rows"]
    if any(
        output.get("artifact_id") == "integrity-filter-observations"
        for output in stage.get("outputs", [])
        if isinstance(output, dict)
    ):
        errors.extend(
            _validate_rows(
                state["plan"],
                stage,
                "integrity-filter-observations",
                observations,
                label="integrity observation output",
            )
        )
    errors.extend(
        _validate_rows(
            state["plan"],
            stage,
            "integrity-passing-candidates",
            passing,
            label="integrity passing output",
        )
    )
    errors.extend(_assert_paths(observations, state["fixture_root"], label="integrity observation output"))
    errors.extend(_assert_paths(passing, state["fixture_root"], label="integrity passing output"))
    errors.extend(_format_errors(passing, label="integrity parser"))
    return errors, {
        "records": len(passing),
        "provider_calls": 0,
        "producer": "offline fixture rows shaped for integrity_filter.parse_stage",
    }


def _filter_novelty(
    state: dict[str, Any], stage: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    errors = _stage_paths_and_config(state, stage)
    observations = state["novelty_observations"]
    passing = state["passing_rows"]
    if any(
        output.get("artifact_id") == "novelty-filter-observations"
        for output in stage.get("outputs", [])
        if isinstance(output, dict)
    ):
        errors.extend(
            _validate_rows(
                state["plan"],
                stage,
                "novelty-filter-observations",
                observations,
                label="novelty observation output",
            )
        )
    errors.extend(
        _validate_rows(
            state["plan"],
            stage,
            "passing-candidates",
            passing,
            label="novelty passing output",
        )
    )
    errors.extend(_assert_paths(observations, state["fixture_root"], label="novelty observation output"))
    errors.extend(_assert_paths(passing, state["fixture_root"], label="novelty passing output"))
    errors.extend(_format_errors(passing, label="novelty parser"))
    return errors, {
        "records": len(passing),
        "provider_calls": 0,
        "producer": "offline fixture rows shaped for novelty_filter.parse_stage",
    }


def _cofold_screen(
    state: dict[str, Any], stage: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    errors = _stage_paths_and_config(state, stage)
    rows = state["screen_raw_rows"]
    errors.extend(
        _validate_rows(
            state["plan"],
            stage,
            "esmfold2-fast-screen",
            rows,
            label="cofold screen output",
        )
    )
    errors.extend(_assert_paths(rows, state["fixture_root"], label="cofold screen output"))
    errors.extend(_format_errors(rows, label="cofold screen parser"))
    return errors, {
        "records": len(rows),
        "provider_calls": 0,
        "provider_stubbed": True,
        "producer": "offline fixture rows shaped for fal_esmfold2_fast_predictor.parse_outputs",
    }


def _target_sequence(state: dict[str, Any]) -> str:
    """Return the primary target chain sequence every fixture stage shares."""
    cached = state.get("target_sequence")
    if isinstance(cached, str) and cached:
        return cached
    config = state["config"]
    context = config.get("context")
    sequence = context.get("target_sequence") if isinstance(context, Mapping) else None
    if not isinstance(sequence, str) or not sequence:
        from .adapters.target_prep_adapter import load_atoms, residue_letter

        target = config["targets"][0]
        atoms, _ = load_atoms(Path(target["structure_source_path"]))
        sequence = "".join(residue_letter(atom.residue_name) for atom in atoms if atom.name == "CA")
    state["target_sequence"] = sequence
    return sequence


def _stage_msa(state: dict[str, Any], stage: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    from .adapters import target_msa_builder

    errors = _stage_paths_and_config(state, stage)
    config = state["config"]
    target = config["targets"][0]
    attempt = state["artifact_root"] / "stages" / str(stage["stage_id"]) / "attempts" / "fixture"
    # The production route is public-server, which sends the target sequence to a
    # third-party host. A free offline contract check may not do that, so the dry run
    # selects the registered query-only route and records the substitution.
    args = argparse.Namespace(
        stage=str(stage["stage_id"]),
        phase="single",
        count=1,
        attempt_dir=attempt,
        receipts_dir=state["receipts_dir"],
        artifact_root=state["artifact_root"],
        config=state["config_path"],
        plan=state["plan_path"],
        route=target_msa_builder.SOURCE_QUERY_ONLY,
        source=None,
        target_sequence=[f"{target['target_id']}={_target_sequence(state)}"],
        allow_public_msa=False,
        precomputed_a3m=None,
        supplied_a3m=None,
        minimum_sequences=None,
    )
    target_msa_builder.run_stage(args)
    manifest_path = attempt / "single" / "msa-manifest.jsonl"
    rows = _load_jsonl(manifest_path)
    errors.extend(
        _validate_rows(state["plan"], stage, "target-msa-manifest", rows, label="stage-msa output")
    )
    errors.extend(_assert_paths(rows, state["fixture_root"], label="stage-msa output"))
    _copy_file(manifest_path, state["artifact_root"] / "inputs" / "msa-manifest.jsonl")
    artifacts = [
        _receipt_artifact(
            "target-msa-manifest", "target-msa-manifest", "jsonl", manifest_path, records=len(rows)
        )
    ]
    for row in rows:
        alignment = Path(str(row["msa_path"]))
        _copy_file(alignment, state["artifact_root"] / "inputs" / "msa" / alignment.name)
        artifacts.append(
            _receipt_artifact(
                f"{target_msa_builder.MSA_ARTIFACT_ID_PREFIX}{row['target_id']}",
                "target-msa-files",
                "file",
                alignment,
                records=1,
            )
        )
    _write_receipt(state["receipts_dir"], str(stage["stage_id"]), artifacts)
    state["msa_rows"] = rows
    return errors, {
        "records": len(rows),
        "provider_calls": 0,
        "producer": "target_msa_builder.run_stage on the offline query-only route",
    }


def _control_provider_rows(
    state: dict[str, Any],
    stage: dict[str, Any],
    controls: list[Any],
    candidates: list[dict[str, Any]],
    errors: list[str],
) -> tuple[list[dict[str, Any]], str]:
    """Return predictor-shaped control rows without calling a paid predictor."""
    from .adapters import control_builder

    config = state["config"]
    control_by_id = {control.control_id: control for control in controls}
    sequence_by_id = {str(row["candidate_id"]): row for row in candidates}
    source_observations = state["source_root"] / "controls" / "control-observations.jsonl"
    if source_observations.is_file():
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(_load_jsonl(source_observations)):
            control = control_by_id.get(str(row.get("candidate_id")))
            if control is None:
                errors.append(
                    f"stage {stage['stage_id']} source observation {index} names an unregistered "
                    f"control: {row.get('candidate_id')}"
                )
                continue
            # design_pose_path was recorded inside the provider's own mount, which has no
            # local file to copy. The materialized control is the same structure by digest.
            staged = {key: value for key, value in row.items() if key != "design_pose_path"}
            rebased = _copy_row_paths(
                staged,
                state["source_root"],
                state["artifact_root"],
                label=f"control observation row {index}",
            )
            rebased["design_pose_path"] = str(control.structure_path)
            rebased["design_pose_sha256"] = control.structure_sha256
            expected = sequence_by_id.get(control.control_id, {}).get("sequence_sha256")
            if expected is not None and rebased.get("sequence_sha256") != expected:
                errors.append(
                    f"stage {stage['stage_id']} control {control.control_id} materializes sequence "
                    f"{expected} but the recorded observation carries {rebased.get('sequence_sha256')}"
                )
            rows.append(rebased)
        return rows, "recorded control observations rebased onto the materialized controls"

    template_rows = state.get("screen_raw_rows") or []
    if not template_rows:
        errors.append(
            f"stage {stage['stage_id']} has no recorded control observations and no screen raw row "
            "to shape a control row from"
        )
        return [], "unavailable"
    template = template_rows[0]
    attempt = state["artifact_root"] / "stages" / str(stage["stage_id"]) / "attempts" / "fixture"
    msa_by_target = {str(row["target_id"]): row for row in state.get("msa_rows", [])}
    rows = []
    for target in control_builder.target_records(config):
        target_id = str(target["target_id"])
        alignment = msa_by_target.get(target_id, {})
        for control in controls:
            for predictor in control_builder.predictor_records(config):
                predictor_id = str(predictor["id"])
                revision = control_builder.adapter_model_revision(config, str(predictor["adapter_id"]))
                for seed in control_builder.rescore_seeds(config):
                    row = copy.deepcopy(template)
                    prediction_dir = (
                        attempt
                        / "single"
                        / "prediction-artifacts"
                        / f"{target_id}-{control.control_id}-{predictor_id}-{control_builder.CONTROL_PHASE}-{seed}"
                    )
                    complex_path = _copy_file(
                        control.structure_path, prediction_dir / f"complex{control.structure_path.suffix}"
                    )
                    measurement = dict(row.get("measurement") or {})
                    measurement.update(
                        {
                            "candidate_id": control.control_id,
                            "seed": int(seed),
                            "phase": control_builder.CONTROL_PHASE,
                            "predictor": predictor_id,
                            "model_revision": revision,
                            "design_pose_sha256": control.structure_sha256,
                            "predicted_complex_sha256": _sha256(complex_path),
                            "target_id": target_id,
                            "target_sha256": str(target["structure_sha256"]),
                            "sequence_sha256": sequence_by_id[control.control_id]["sequence_sha256"],
                            "chain_mapping": {"target": control.target_chain, "binder": control.binder_chain},
                        }
                    )
                    measurement_path = prediction_dir / "measurement-source.json"
                    _write_json(measurement_path, {"measurement": measurement})
                    row.update(
                        {
                            "candidate_id": control.control_id,
                            "origin_generator": "control-builder",
                            "target_id": target_id,
                            "target_sha256": str(target["structure_sha256"]),
                            "predictor": predictor_id,
                            "model_revision": revision,
                            "seed": int(seed),
                            "phase": control_builder.CONTROL_PHASE,
                            "sequence_sha256": sequence_by_id[control.control_id]["sequence_sha256"],
                            "design_pose_path": str(control.structure_path),
                            "design_pose_sha256": control.structure_sha256,
                            "chain_mapping": {"target": control.target_chain, "binder": control.binder_chain},
                            "msa_path": alignment.get("msa_path"),
                            "msa_sha256": alignment.get("msa_sha256"),
                            "predicted_complex_path": str(complex_path),
                            "predicted_complex_sha256": _sha256(complex_path),
                            "metric_source_path": str(measurement_path),
                            "metric_source_sha256": _sha256(measurement_path),
                            "measurement": measurement,
                        }
                    )
                    if isinstance(row.get("pae_path"), str) and Path(row["pae_path"]).is_file():
                        pae_path = _copy_file(Path(row["pae_path"]), prediction_dir / "pae.json")
                        row["pae_path"] = str(pae_path)
                        row["pae_sha256"] = _sha256(pae_path)
                    rows.append(row)
    return rows, "control rows shaped from the screen raw-prediction rows over the materialized controls"


def _control_calibration(
    state: dict[str, Any], stage: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    from .adapters import control_builder

    errors = _stage_paths_and_config(state, stage)
    config = state["config"]
    target = config["targets"][0]
    attempt = state["artifact_root"] / "stages" / str(stage["stage_id"]) / "attempts" / "fixture"
    args = argparse.Namespace(
        stage="control-calibration",
        phase="single",
        count=1,
        attempt_dir=attempt,
        receipts_dir=state["receipts_dir"],
        artifact_root=state["artifact_root"],
        config=state["config_path"],
        plan=state["plan_path"],
        target_sequence=[f"{target['target_id']}={_target_sequence(state)}"],
        hotspot_residues=[
            f"{target['target_id']}=" + ",".join(target["site"]["reference_contact_residues"])
        ],
        target_msa_a3m=[
            f"{row['target_id']}={row['msa_path']}" for row in state.get("msa_rows", [])
        ],
        target_unpaired_msa_a3m=[
            f"{row['target_id']}={row['msa_path']}" for row in state.get("msa_rows", [])
        ],
    )
    controls = control_builder.materialize_controls(
        config, state["config_path"], attempt / "single" / "control-structures"
    )
    control_builder.validate_predictor_inputs(config, args)
    candidates = control_builder.candidate_rows(
        config, controls, attempt / "single" / "control-workspace" / "sequences"
    )
    # The predictor itself is the paid call. Everything around it, including the control
    # materialization the science depends on, runs for real.
    stub = ProviderClientStub(str(stage["adapter_id"]))
    state.setdefault("provider_stubs", {})[str(stage["stage_id"])] = stub
    provider_rows, mode = _control_provider_rows(state, stage, controls, candidates, errors)
    rows: list[dict[str, Any]] = []
    if provider_rows:
        try:
            rows = control_builder.annotate_rows(
                provider_rows, {control.control_id: control for control in controls}
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"stage {stage['stage_id']} annotation failed: {type(exc).__name__}: {exc}")
        observed = {
            (
                str(row.get("target_id")),
                str(row.get("candidate_id")),
                str(row.get("predictor")),
                int(row.get("seed", -1)),
            )
            for row in rows
        }
        expected = control_builder.expected_keys(config, controls)
        if observed != expected:
            errors.append(
                f"stage {stage['stage_id']} control matrix mismatch: "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
    rows.sort(
        key=lambda row: (
            str(row.get("target_id")),
            str(row.get("candidate_id")),
            str(row.get("predictor")),
            int(row.get("seed", -1)),
        )
    )
    errors.extend(
        _validate_rows(state["plan"], stage, "control-observations", rows, label="control output")
    )
    errors.extend(_assert_paths(rows, state["fixture_root"], label="control output"))
    errors.extend(_format_errors(rows, label="control parser"))
    output = control_builder.output_path(args)
    _write_jsonl(output, rows)
    _copy_file(output, state["artifact_root"] / "controls" / "control-observations.jsonl")
    _write_receipt(
        state["receipts_dir"],
        str(stage["stage_id"]),
        [
            _receipt_artifact(
                "control-observations", "raw-prediction-manifest", "jsonl", output, records=len(rows)
            )
        ],
    )
    state["control_rows"] = rows
    return errors, {
        "records": len(rows),
        "controls": len(controls),
        "provider_calls": len(stub.calls),
        "provider_stubbed": True,
        "producer": f"control_builder.materialize_controls and annotate_rows; {mode}",
    }


def _render_structure_pictures(
    state: dict[str, Any], stage: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    from .adapters import structure_picture_renderer

    errors = _stage_paths_and_config(state, stage)
    attempt = state["artifact_root"] / "stages" / str(stage["stage_id"]) / "attempts" / "fixture"
    out_dir = attempt / "single" / "pictures"
    args = argparse.Namespace(
        stage=str(stage["stage_id"]),
        phase="single",
        count=1,
        attempt_dir=attempt,
        receipts_dir=state["receipts_dir"],
        artifact_root=state["artifact_root"],
        config=state["config_path"],
        plan=state["plan_path"],
        out_dir=out_dir,
    )
    structure_picture_renderer.run(args)
    manifest_path = out_dir / "manifest.json"
    manifest = _load(manifest_path)
    errors.extend(
        _validate_rows(
            state["plan"], stage, "structure-picture-manifest", manifest, label="structure picture output"
        )
    )
    errors.extend(_assert_paths(manifest, state["fixture_root"], label="structure picture output"))
    structure_picture_renderer.parse(args)
    parser_result = _load(attempt / "single" / "parser-result.json")
    errors.extend(
        f"structure picture parser: {error}" for error in parser_result.get("errors", [])
    )
    images = sorted(out_dir.glob("images/*.png"))
    artifacts = [
        _receipt_artifact(
            "structure-picture-manifest", "structure-picture-manifest", "json", manifest_path, records=1
        )
    ]
    for artifact_id, artifact_type, name in (
        ("structure-picture-index", "structure-picture-index", "index.html"),
        ("structure-picture-archive", "structure-picture-archive", "structure-pictures.zip"),
    ):
        path = out_dir / name
        if path.is_file():
            artifacts.append(_receipt_artifact(artifact_id, artifact_type, "file", path, records=1))
            _copy_file(path, state["artifact_root"] / "pictures" / name)
        else:
            errors.append(f"stage {stage['stage_id']} wrote no {artifact_id}: {path}")
    if images:
        artifacts.append(
            _receipt_artifact(
                "structure-pictures", "structure-pictures", "image", images[0], records=len(images)
            )
        )
    else:
        errors.append(f"stage {stage['stage_id']} wrote no structure pictures under {out_dir / 'images'}")
    _copy_file(manifest_path, state["artifact_root"] / "pictures" / "manifest.json")
    _write_receipt(state["receipts_dir"], str(stage["stage_id"]), artifacts)
    return errors, {
        "records": len(images),
        "image_count": manifest.get("image_count"),
        "provider_calls": 0,
        "producer": "structure_picture_renderer.run and structure_picture_renderer.parse",
    }


def _score_screen(
    state: dict[str, Any], stage: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    errors = _stage_paths_and_config(state, stage)
    rows = state["screen_rows"]
    errors.extend(
        _validate_rows(
            state["plan"],
            stage,
            "screen-score-table",
            rows,
            label="screen score output",
        )
    )
    errors.extend(_assert_paths(rows, state["fixture_root"], label="screen score output"))
    errors.extend(_format_errors(rows, label="screen score parser"))
    return errors, {
        "records": len(rows),
        "provider_calls": 0,
        "producer": "offline fixture rows shaped for interface_scorer.parse_stage",
    }


def _promote(state: dict[str, Any], stage: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    from .adapters import promotion_selector

    errors = _stage_paths_and_config(state, stage)
    config = state["config"]
    lineage = promotion_selector.prepare_lineage(state["passing_rows"])
    ranked = _lane().rank_candidate_cohort(config, state["screen_rows"], list(config["cofold"]["screen_seeds"]))
    mode = promotion_selector.ranking_mode(config)
    ranked = promotion_selector.apply_ranking_mode(ranked, mode)
    ranked.sort(key=lambda row: _lane()._rank_sort_key(row, config))
    try:
        selected = promotion_selector.select_diverse_parents(
            ranked, lineage, config, promotion_selector.load_diversity_policy(config)
        )
    except Exception as exc:  # noqa: BLE001
        source_lines = Path(promotion_selector.__file__).read_text(encoding="utf-8").splitlines()
        source_line = next(
            (index for index, line in enumerate(source_lines, start=1) if "diversity and score rules produced" in line),
            promotion_selector.select_diverse_parents.__code__.co_firstlineno,
        )
        errors.append(
            f"{promotion_selector.__file__}:{source_line}: "
            f"{type(exc).__name__}: {exc}"
        )
        selected = [row for row in ranked if row.get("eligible") is True]
    rows = promotion_selector.promotion_rows(
        selected,
        lineage,
        promotion_selector.raw_score_vectors(state["screen_rows"]),
        mode=mode,
        controls_required=promotion_selector.controls_are_required(config),
    )
    errors.extend(_validate_rows(state["plan"], stage, "rescore-candidates", rows, label="promote output"))
    errors.extend(_assert_paths(rows, state["fixture_root"], label="promote output"))
    output = state["artifact_root"] / "stages" / "promote" / "attempts" / "fixture" / "single" / "promotion-manifest.jsonl"
    _write_jsonl(output, rows)
    published = state["artifact_root"] / "promotion" / "promotion-manifest.jsonl"
    _copy_file(output, published)
    state["promotion_rows"] = rows
    _write_receipt(state["receipts_dir"], "promote", [_receipt_artifact("rescore-candidates", "rescore-candidate-manifest", "jsonl", output, records=len(rows))])
    return errors, {"records": len(rows), "producer": "promotion_selector.promotion_rows"}


def _cofold_rescore(state: dict[str, Any], stage: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    from .adapters import fal_esmfold2_fast_predictor as fal

    errors = _stage_paths_and_config(state, stage)
    config = state["config"]
    attempt = state["artifact_root"] / "stages" / str(stage["stage_id"]) / "attempts" / "fixture"
    args = argparse.Namespace(
        stage=stage["stage_id"], phase="single", count=len(state["promotion_rows"]),
        attempt_dir=attempt, receipts_dir=state["receipts_dir"], artifact_root=state["artifact_root"],
        config=state["config_path"], plan=state["plan_path"], run_index=attempt / "single" / "ef2fast-run-index.jsonl",
        target_sequence=[f"{config['targets'][0]['target_id']}={config['context']['target_sequence']}" if isinstance(config.get("context"), dict) and config["context"].get("target_sequence") else f"{config['targets'][0]['target_id']}={state['source_root'].name}"],
        hotspot_residues=[f"{config['targets'][0]['target_id']}=" + ",".join(config['targets'][0]['site']['reference_contact_residues'])],
        client=None,
    )
    target = config["targets"][0]
    sequence = state.get("target_sequence")
    if not sequence:
        sequence = config.get("context", {}).get("target_sequence") if isinstance(config.get("context"), dict) else None
    if not sequence:
        sequence = "".join(
            residue for residue in state["config"].get("context", {}).get("target_sequence", "")
            if residue.isalpha()
        )
    if not sequence:
        from .adapters.target_prep_adapter import load_atoms, residue_letter

        atoms, _ = load_atoms(Path(target["structure_source_path"]))
        sequence = "".join(residue_letter(atom.residue_name) for atom in atoms if atom.name == "CA")
    state["target_sequence"] = sequence
    args.target_sequence = [f"{target['target_id']}={sequence}"]
    args.hotspot_residues = [f"{target['target_id']}=" + ",".join(target["site"]["reference_contact_residues"])]
    plan_items = fal.build_plan(config, args)
    stub = ProviderClientStub(str(stage["adapter_id"]))
    state.setdefault("provider_stubs", {})[str(stage["stage_id"])] = stub
    args.client = stub
    for item in plan_items[: len(state["promotion_rows"])]:
        try:
            fal.preflight_item(config, args, item, {target["target_id"]: sequence}, {target["target_id"]: ",".join(target["site"]["reference_contact_residues"])})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"stage {stage['stage_id']} preflight failed: {exc}")

    raw_by_candidate = {str(row["candidate_id"]): row for row in state["screen_raw_rows"]}
    raw_rows: list[dict[str, Any]] = []
    for candidate in state["promotion_rows"]:
        source = raw_by_candidate.get(str(candidate["candidate_id"]))
        if source is None:
            errors.append(f"stage {stage['stage_id']} has no producer-shaped screen raw row for {candidate['candidate_id']}")
            continue
        for seed in config["cofold"]["rescore_seeds"]:
            row = copy.deepcopy(source)
            row["seed"] = int(seed)
            row["phase"] = "uniform-rescore"
            row["origin_generator"] = candidate["origin_generator"]
            measurement_source = Path(row["metric_source_path"])
            measurement_document = _load(measurement_source)
            measurement = dict(measurement_document["measurement"])
            measurement["seed"] = int(seed)
            measurement["phase"] = "uniform-rescore"
            measurement_path = attempt / "single" / "prediction-artifacts" / str(candidate["candidate_id"]) / f"seed-{seed}" / "measurement-source.json"
            measurement_document["measurement"] = measurement
            _write_json(measurement_path, measurement_document)
            row["metric_source_path"] = str(measurement_path.resolve())
            row["metric_source_sha256"] = _sha256(measurement_path)
            row["measurement"] = copy.deepcopy(measurement)
            raw_rows.append(row)
    errors.extend(_validate_rows(state["plan"], stage, "esmfold2-fast-rescore", raw_rows, label="cofold rescore output"))
    errors.extend(_assert_paths(raw_rows, state["fixture_root"], label="cofold rescore output"))
    errors.extend(_format_errors(raw_rows, label="cofold rescore parser"))
    output = attempt / "single" / "cofold-observations.jsonl"
    _write_jsonl(output, raw_rows)
    _write_receipt(state["receipts_dir"], str(stage["stage_id"]), [_receipt_artifact("esmfold2-fast-rescore", "raw-prediction-manifest", "jsonl", output, records=len(raw_rows))])
    state["raw_rescore_rows"] = raw_rows
    return errors, {
        "records": len(raw_rows),
        "provider_calls": len(stub.calls),
        "provider_stubbed": True,
        "producer": "fal_esmfold2_fast_predictor.parse_outputs",
    }


def _uniform(state: dict[str, Any], stage: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    from .adapters import ensemble_reducer

    errors = _stage_paths_and_config(state, stage)
    config = state["config"]
    targets = ensemble_reducer.target_records(config)
    predictors, seeds = ensemble_reducer.predictor_records(config)
    candidates = {str(row["candidate_id"]): row for row in state["promotion_rows"]}
    args = argparse.Namespace(
        stage=stage["stage_id"], attempt_dir=state["artifact_root"] / "stages" / str(stage["stage_id"]) / "attempts" / "fixture",
        receipts_dir=state["receipts_dir"], artifact_root=state["artifact_root"],
    )
    candidate_rows = ensemble_reducer.normalized_candidate_rows(
        config, args, state["plan"], targets, predictors, seeds, candidates
    )
    # The reducer joins the candidate half with the calibrated control panel and then
    # validates the pair. Preparing only the candidate half left the control gate, the
    # seed ensemble and every downstream stage that reads a control row untested.
    control_rows = ensemble_reducer.normalized_control_rows(
        config, args, state["plan"], state["artifact_root"]
    )
    rows = sorted(
        [*candidate_rows, *control_rows],
        key=lambda row: (
            str(row.get("target_id")),
            str(row.get("candidate_id")),
            str(row.get("predictor")),
            int(row.get("seed", -1)),
        ),
    )
    errors.extend(
        _lane().validate_observations(
            config,
            rows,
            expected_phase="uniform-rescore",
            required_seed_values=seeds,
            require_controls=True,
        )
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        grouped.setdefault(
            (str(row["target_id"]), str(row["candidate_id"]), str(row["predictor"])), []
        ).append(row)
    for group in grouped.values():
        if all(row.get("status") == "scored" for row in group):
            ensemble_reducer.reduce_seed_ensemble(group, seeds)
    errors.extend(_validate_rows(state["plan"], stage, "uniform-observations", rows, label="uniform output"))
    errors.extend(_assert_paths(rows, state["fixture_root"], label="uniform output"))
    output = args.attempt_dir / "single" / "uniform-observations.jsonl"
    _write_jsonl(output, rows)
    published = state["artifact_root"] / "scores" / "uniform-observations.jsonl"
    _copy_file(output, published)
    _write_receipt(state["receipts_dir"], str(stage["stage_id"]), [_receipt_artifact("uniform-observations", "uniform-observations", "jsonl", output, records=len(rows))])
    state["observations"] = rows
    return errors, {
        "records": len(rows),
        "candidates": len(candidate_rows),
        "controls": len(control_rows),
        "producer": "ensemble_reducer.normalized_candidate_rows, normalized_control_rows and reduce_seed_ensemble",
    }


def _final_rank(state: dict[str, Any], stage: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    errors = _stage_paths_and_config(state, stage)
    ranking = _lane().rank_candidates(state["config"], state["observations"])
    lineage_path = state["artifact_root"] / "promotion" / "promotion-manifest.jsonl"
    sequence_dir = state["artifact_root"] / "stages" / "final-rank" / "attempts" / "fixture" / "single" / "sequences"
    errors.extend(_lane().attach_ranked_sequence_artifacts(state["config"], ranking, lineage_path, sequence_dir))
    errors.extend(_validate_rows(state["plan"], stage, "ranked-portfolio", ranking, label="final rank output"))
    errors.extend(_assert_paths(ranking, state["fixture_root"], label="final rank output"))
    output = state["artifact_root"] / "stages" / "final-rank" / "attempts" / "fixture" / "single" / "ranked-candidates.json"
    _write_json(output, ranking)
    _copy_file(output, state["artifact_root"] / "scores" / "ranked-candidates.json")
    sequences = sorted(sequence_dir.glob("*.fasta"))
    artifacts = [_receipt_artifact("ranked-portfolio", "ranked-portfolio", "json", output, records=1)]
    if sequences:
        artifacts.append(
            _receipt_artifact(
                "final-sequence-files", "sequence-files", "fasta", sequences[0], records=len(sequences)
            )
        )
    else:
        errors.append(
            f"stage {stage['stage_id']} wrote no ranked sequence file into {sequence_dir}. "
            "attach_ranked_sequence_artifacts skips a candidate whose lineage sequence does not "
            "resolve, so an empty directory means the ranking produced no candidate it could match."
        )
    _write_receipt(state["receipts_dir"], str(stage["stage_id"]), artifacts)
    state["ranking"] = ranking
    return errors, {"records": len(ranking.get("ranked_candidates", [])), "producer": "lane.rank_candidates and lane.attach_ranked_sequence_artifacts"}


def _output_check(state: dict[str, Any], stage: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    from . import control_separation

    errors = _stage_paths_and_config(state, stage)
    for artifact_id, source in (("ranked-portfolio", "final-rank"), ("screen-score-table", "score-screen"), ("uniform-observations", "uniform-rescore")):
        try:
            input_files(state["plan"], state["receipts_dir"], str(stage["stage_id"]), artifact_id=artifact_id, source_stage_id=source)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"stage output-check input {source}:{artifact_id} failed: {exc}")
    report = {
        "ok": True,
        "run_fingerprint": state["plan"].get("run_fingerprint"),
        "stage_count": len(state["plan"].get("ordered_stage_ids", [])),
        "selected_count": len(state.get("ranking", {}).get("selected_candidates", [])),
        "control_separation": {
            "status": "not_required",
            "statistic": control_separation.STATISTIC,
            "metric": control_separation.METRIC,
            "reason": "contract dry run uses a non-production fixture graph",
            "measurements": [],
        },
        "checks": [{"name": "declared-inputs", "status": "pass", "reason": "all output-validator inputs resolved through receipts"}],
        "errors": [],
    }
    output = state["artifact_root"] / "stages" / "output-check" / "attempts" / "fixture" / "single" / "output-check.json"
    _write_json(output, report)
    _copy_file(output, state["artifact_root"] / "validation" / "output-check.json")
    errors.extend(_validate_rows(state["plan"], stage, "output-check", report, label="output-check output"))
    _write_receipt(state["receipts_dir"], str(stage["stage_id"]), [_receipt_artifact("output-check", "output-check", "json", output, records=1)])
    return errors, {"records": 1, "producer": "output_validator declared-input preparation"}


def _render_viewer(state: dict[str, Any], stage: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    errors = _stage_paths_and_config(state, stage)
    from .adapters import browser_renderer, view_renderer

    viewer_dir = state["artifact_root"] / "stages" / "render-viewer" / "attempts" / "fixture" / "single" / "viewer"
    browser_renderer.run(
        argparse.Namespace(
            run_dir=state["run_root"],
            out_dir=viewer_dir,
        )
    )
    manifest_path = viewer_dir / "manifest.json"
    manifest = _load(manifest_path)
    errors.extend(_validate_rows(state["plan"], stage, "viewer-manifest", manifest, label="viewer output"))
    errors.extend(_assert_paths(manifest, state["fixture_root"], label="viewer output"))
    for image in sorted((viewer_dir / "thumbnails").glob("*.png")):
        try:
            view_renderer.parse_output(image, "image")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"render-viewer parser rejected {image}: {exc}")
    _write_receipt(state["receipts_dir"], str(stage["stage_id"]), [
        _receipt_artifact("viewer-manifest", "viewer-manifest", "json", manifest_path, records=1),
        _receipt_artifact("viewer-thumbnails", "viewer-images", "image", next(viewer_dir.glob("thumbnails/*.png")), records=len(list(viewer_dir.glob("thumbnails/*.png")))),
    ])
    return errors, {
        "records": len(manifest.get("designs", [])),
        "renderer": "browser",
        "python_raster_renderer": True,
        "producer": "browser_renderer.run; python_raster_renderer.render_backbone_png; view_renderer.parse_output",
    }


# Every stage a run plan can order below the dry run's start point needs a runner here.
# A stage with no entry reports "has no contract-dry-run adapter preparation" and writes no
# receipt, which cascades to every stage below it. The entries that go missing are the ones
# for stages that have never executed in a run, because nothing else exercises them.
def _predictor_record(state: dict[str, Any], stage: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Return the predictor this stage folds with, and whether it screens or rescores.

    A predictor declares the two stage ids it owns, so the stage is matched back to its
    predictor rather than to the first enabled one. Getting this wrong would fold every
    arm's rows with one arm's adapter, and the MSA fields follow the adapter.
    """
    stage_id = str(stage.get("stage_id", ""))
    config = state["config"]
    for predictor in config.get("cofold", {}).get("predictors", []):
        if not isinstance(predictor, dict) or not predictor.get("enabled", True):
            continue
        if predictor.get("screen_stage") == stage_id:
            return predictor, "screen"
        if predictor.get("rescore_stage") == stage_id:
            return predictor, "uniform-rescore"
    raise ContractDryRunError(
        f"stage {stage_id} declares role cofold-predictor but no enabled predictor claims it"
    )


def _cofold_predictor(
    state: dict[str, Any], stage: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Validate one cofold predictor stage against the contract that stage declares.

    Ten stages in the shipped profiles declare `required_role: cofold-predictor`. Two have
    an exact-id runner that also replays the fal client's own plan building, so they never
    reach this one. The other eight do, and this runner reads the artifact id, the phase
    and the field set from the stage instead of holding one predictor's shape.

    The eight do not differ from the fast pair in adapter id and model revision alone.
    `cofold-screen-boltz` declares `design_pose_path`,
    `chain_mapping`, `msa_path` and `msa_sha256` on top of the fast contract, and
    `cofold-screen-alphafold-multimer-v3` declares the two MSA fields.
    `fixture_adapter.raw_prediction_row` already writes all four, and writes the MSA pair
    as null for an arm whose adapter does not accept the manifest, so the row follows the
    stage's own adapter rather than a hand-written field list.

    The provider is never called. Nothing here replaces the exact-id runners' preflight,
    and the returned detail says which check ran so a reader cannot mistake the two.
    """
    from . import fixture_adapter

    errors = _stage_paths_and_config(state, stage)
    config = state["config"]
    predictor, phase = _predictor_record(state, stage)
    output = next(
        (item for item in stage.get("outputs", []) if isinstance(item, dict)), None
    )
    if output is None:
        return errors + [f"stage {stage['stage_id']} declares no output to validate"], {}
    artifact_id = str(output.get("artifact_id", ""))
    target = next(
        (item for item in config["targets"] if item.get("role") == "primary"),
        config["targets"][0],
    )
    if phase == "screen":
        candidates = state["passing_rows"]
        seeds = [int(config["cofold"]["screen_seeds"][0])]
    else:
        candidates = state["promotion_rows"]
        seeds = [int(seed) for seed in config["cofold"]["rescore_seeds"]]

    attempt = _stage_attempt_root(state["artifact_root"], str(stage["stage_id"]))
    try:
        rows = [
            fixture_adapter.raw_prediction_row(
                config,
                artifact_root=state["artifact_root"],
                target=target,
                candidate_id=str(row["candidate_id"]),
                predictor=predictor,
                seed=seed,
                phase=phase,
                sequence_sha256=str(row["sequence_sha256"]),
                design_pose_path=str(row["design_pose_path"]),
                design_pose_sha256=str(row["design_pose_sha256"]),
                origin_generator=str(row["origin_generator"]),
            )
            for row in candidates
            for seed in seeds
        ]
    except (FileNotFoundError, KeyError) as exc:
        # `fixture_adapter.target_msa_record` raises rather than folding single sequence
        # under a full-arm label, and its raise names a missing file without saying what to
        # do about it. Only that raise is diagnosed as an ordering error. A KeyError from
        # anywhere else in row construction, a missing `sequence_sha256` for instance, is a
        # different fault and must not be reported as a missing alignment.
        text = str(exc)
        if "msa" not in text.lower():
            raise
        return errors + [
            f"stage {stage['stage_id']} folds against a target MSA that was not staged "
            f"when it ran, so order stage-msa before it: {exc}"
        ], {"records": 0, "provider_calls": 0, "provider_stubbed": True}
    rows = fixture_adapter.attach_raw_artifacts(config, rows, attempt / "single")
    # A manifest row that exists but carries a null `msa_path` satisfies the declared-field
    # check, because the field is present. `target_msa_record` returns such a row without
    # looking at its values, so an arm that declares `msa_path` can fold with no alignment
    # and pass. A missing manifest raises; a null one does not, and these are different
    # failures.
    declared = {str(field) for field in output.get("required_fields", [])}
    if "msa_path" in declared:
        empty = [index for index, row in enumerate(rows) if not row.get("msa_path")]
        if empty:
            errors.append(
                f"stage {stage['stage_id']} declares msa_path and {len(empty)} of {len(rows)} "
                "rows carry none, so the manifest names this target with an empty alignment"
            )
    errors.extend(
        _validate_rows(state["plan"], stage, artifact_id, rows, label=f"{stage['stage_id']} output")
    )
    errors.extend(_assert_paths(rows, state["fixture_root"], label=f"{stage['stage_id']} output"))
    errors.extend(_format_errors(rows, label=f"{stage['stage_id']} parser"))
    destination = attempt / "single" / "cofold-observations.jsonl"
    _write_jsonl(destination, rows)
    _write_receipt(
        state["receipts_dir"],
        str(stage["stage_id"]),
        [
            _receipt_artifact(
                artifact_id, "raw-prediction-manifest", "jsonl", destination, records=len(rows)
            )
        ],
    )
    state.setdefault("predictor_rows_by_stage", {})[str(stage["stage_id"])] = rows
    return errors, {
        "records": len(rows),
        "provider_calls": 0,
        "provider_stubbed": True,
        "predictor": predictor.get("id"),
        "adapter_id": stage.get("adapter_id"),
        "checked": "declared fields, referenced paths and structure parseability",
        "not_checked": "the adapter's own plan building and preflight, which this role runner does not replay",
        "producer": "offline fixture rows shaped by fixture_adapter.raw_prediction_row",
    }


def _sequence_designer(
    state: dict[str, Any], stage: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Exercise a designer's declared lineage and artifacts with local fixture sequences."""
    from . import fixture_adapter

    stage_id = str(stage["stage_id"])
    errors = _stage_paths_and_config(state, stage)
    inputs = stage.get("inputs", [])
    manifests = [item for item in inputs if isinstance(item, str) and ":" in item]
    outputs = [item for item in stage.get("outputs", []) if isinstance(item, dict) and item.get("kind") == "jsonl"]
    if len(manifests) != 1 or len(outputs) != 1:
        return errors + [f"stage {stage_id} needs one backbone manifest and one JSONL output"], {}
    producer, input_id = manifests[0].split(":", 1)
    output_id = str(outputs[0]["artifact_id"])
    designers = [
        item for item in state["config"].get("sequence_design", {}).get("designers", [])
        if isinstance(item, dict) and item.get("enabled", True)
        and item.get("command_stage") == stage_id
        and item.get("adapter_id") == stage.get("adapter_id")
    ]
    if len(designers) != 1:
        return errors + [f"stage {stage_id} is not claimed by exactly one enabled sequence designer"], {}
    designer = designers[0]
    if producer not in {f"generate-{arm}" for arm in designer.get("compatible_generators", [])}:
        return errors + [f"stage {stage_id} reads {producer}, outside its registered generator arms"], {}
    try:
        _, paths = input_files(state["plan"], state["receipts_dir"], stage_id,
                               artifact_id=input_id, source_stage_id=producer)
        parents = [row for path in paths for row in _load_jsonl(path)]
    except Exception as exc:  # noqa: BLE001
        return errors + [f"stage {stage_id} cannot read its backbone manifest: {exc}"], {}
    if not parents:
        return errors + [f"stage {stage_id} has no backbone rows"], {}

    target = next((item for item in state["config"].get("targets", []) if item.get("role") == "primary"), None)
    if target is None:
        return errors + [f"stage {stage_id} has no primary target"], {}
    target_path = Path(str(target.get("structure_source_path", "")))
    if not _inside(target_path, state["fixture_root"]) or not target_path.is_file():
        return errors + [f"stage {stage_id} primary target structure is outside or missing from fixture"], {}
    attempt = _stage_attempt_root(state["artifact_root"], stage_id) / "single"
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for parent in parents:
        parent_id = parent.get("candidate_id")
        pose_value = parent.get("design_pose_path")
        pose_hash = parent.get("design_pose_sha256")
        if not isinstance(parent_id, str) or not parent_id or parent_id in seen:
            errors.append(f"stage {stage_id} has a missing or repeated parent candidate_id")
            continue
        seen.add(parent_id)
        pose = Path(pose_value) if isinstance(pose_value, str) else None
        if pose is None or not _inside(pose, state["fixture_root"]) or not pose.is_file():
            errors.append(f"stage {stage_id} parent {parent_id} has no fixture design pose")
            continue
        if not isinstance(pose_hash, str) or _sha256(pose) != pose_hash:
            errors.append(f"stage {stage_id} parent {parent_id} design pose hash does not match")
            continue
        arm = producer.removeprefix("generate-")
        if parent.get("origin_generator") != arm:
            errors.append(f"stage {stage_id} parent {parent_id} has the wrong origin_generator")
            continue
        variants = int(state["config"]["sequence_design"]["sequences_per_backbone"])
        for variant in range(variants):
            candidate_id = f"{parent_id}-{designer['id']}-{variant:02d}"
            row = {
                **parent,
                "candidate_id": candidate_id,
                "parent_candidate_id": parent_id,
                "sequence_designer": str(designer["id"]),
                "seq_method": str(designer["id"]),
                "status": "sequence-designed",
                "optimization_round": 0,
                "last_optimizer": None,
            }
            fixture_adapter.write_candidate_sequences([row], attempt / "sequences")
            fixture_adapter.write_candidate_design_poses(
                [row], attempt / "poses",
                target_structure_path=target_path,
                target_chain_id=str(target["chains"][0]["chain_id"]),
                binder_chain_id=str(state["config"]["binder"]["binder_chain_id"]),
            )
            rows.append(row)
    if errors:
        return errors, {"records": len(rows), "provider_calls": 0, "provider_stubbed": True}
    errors.extend(_validate_rows(state["plan"], stage, output_id, rows, label=f"{stage_id} output"))
    errors.extend(_assert_paths(rows, state["fixture_root"], label=f"{stage_id} output"))
    errors.extend(_format_errors(rows, label=f"{stage_id} parser"))
    destination = _rendered_stage_output(state["artifact_root"], stage, output_id)
    _write_jsonl(destination, rows)
    artifacts = [_receipt_artifact(output_id, str(outputs[0].get("artifact_type", output_id)),
                                   "jsonl", destination, records=len(rows))]
    for artifact_id, field in (("sequence-files", "sequence_path"),
                               ("design-pose-files", "design_pose_path")):
        declared = next((item for item in stage.get("outputs", [])
                         if isinstance(item, dict) and item.get("artifact_id") == artifact_id), None)
        if declared is None:
            continue
        files = [Path(row[field]) for row in rows]
        artifacts.append({
            "artifact_id": artifact_id,
            "artifact_type": str(declared.get("artifact_type", artifact_id)),
            "kind": str(declared.get("kind", "file")),
            "phase": "single",
            "files": [{"path": str(path), "sha256": _sha256(path), "records": 1} for path in files],
        })
    _write_receipt(state["receipts_dir"], stage_id, artifacts)
    return errors, {
        "records": len(rows),
        "provider_calls": 0,
        "provider_stubbed": True,
        "adapter_id": stage.get("adapter_id"),
        "producer": "local synthetic sequence and pose fixtures linked to validated backbone rows",
        "not_checked": "model inference or sequence quality",
    }


STAGE_RUNNER_NAMES = (
    "normalize-candidates",
    "filter-integrity",
    "filter-novelty",
    "cofold-screen-esmfold2-fast",
    "stage-msa",
    "control-calibration",
    "score-screen",
    "promote",
    "cofold-rescore-esmfold2-fast",
    "uniform-rescore",
    "final-rank",
    "render-structure-pictures",
    "output-check",
    "render-viewer",
)
# Stages above the dry run's start point. Their receipts are carried in from the source
# tree rather than re-run, so they need no runner.
UPSTREAM_STAGE_IDS = frozenset({"target-prepare", "runtime-check"})


def stage_runners() -> dict[str, Any]:
    """Return the stage-id to runner map, one entry per name in STAGE_RUNNER_NAMES."""
    runners = {
        "normalize-candidates": _normalize_candidates,
        "filter-integrity": _filter_integrity,
        "filter-novelty": _filter_novelty,
        "cofold-screen-esmfold2-fast": _cofold_screen,
        "stage-msa": _stage_msa,
        "control-calibration": _control_calibration,
        "score-screen": _score_screen,
        "promote": _promote,
        "cofold-rescore-esmfold2-fast": _cofold_rescore,
        "uniform-rescore": _uniform,
        "final-rank": _final_rank,
        "render-structure-pictures": _render_structure_pictures,
        "output-check": _output_check,
        "render-viewer": _render_viewer,
    }
    missing = sorted(set(STAGE_RUNNER_NAMES) - set(runners))
    if missing:
        raise ContractDryRunError(f"STAGE_RUNNER_NAMES declares runners that do not exist: {missing}")
    return runners


# A role covers a family of stages that no exact-id list can keep up with. Every stage in
# every shipped profile declares `required_role`, and the stages with no runner of
# their own fall into five roles. A runner registered against a role reaches a stage
# a profile adds later without the registry naming it.
ROLE_RUNNERS: dict[str, Any] = {
    "cofold-predictor": _cofold_predictor,
    "sequence-designer": _sequence_designer,
}


def resolve_stage_runner(stage: Mapping[str, Any]) -> Any:
    """Return the runner for one stage, by exact id first and then by declared role.

    Exact id wins because two predictor stages have a runner that replays the fal client's
    own plan building, which is a stronger check than the role runner performs. The role is
    the fallback, not the preference.
    """
    if not isinstance(stage, Mapping):
        return None
    runner = stage_runners().get(str(stage.get("stage_id", "")))
    if runner is not None:
        return runner
    role = stage.get("required_role")
    return ROLE_RUNNERS.get(role) if isinstance(role, str) else None


def covered_roles() -> tuple[str, ...]:
    """Roles a runner is registered against, for coverage reporting."""
    return tuple(sorted(ROLE_RUNNERS))


def run_contract_dry_run(
    plan_path: Path,
    from_stage: str,
    *,
    fixture_root: Path | None = None,
    artifact_root: Path | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    """Prepare fixtures and validate each stage from ``from_stage`` onward."""
    plan = _load(plan_path)
    if not isinstance(plan, dict):
        raise ContractDryRunError("run plan is not an object")
    ordered = plan.get("ordered_stage_ids")
    if not isinstance(ordered, list) or from_stage not in ordered:
        raise ContractDryRunError(f"--from must name a stage in ordered_stage_ids: {from_stage}")
    fixture = (fixture_root or DEFAULT_FIXTURE_ROOT).expanduser().resolve()
    state = _prepare_fixture(
        plan_path,
        plan,
        fixture,
        replace=replace,
        source_artifact_root=artifact_root,
    )
    imported = _import_from_stage_inputs(state, from_stage)
    state.setdefault("imported_receipts", []).extend(
        stage_id
        for stage_id in imported
        if stage_id not in state["imported_receipts"]
    )
    start = ordered.index(from_stage)
    results: list[dict[str, Any]] = []
    for stage_id in ordered[start:]:
        stage = _stage(state["plan"], str(stage_id))
        errors: list[str] = []
        detail: dict[str, Any] = {}
        try:
            runner = resolve_stage_runner(stage)
            if runner is None:
                errors.append(f"stage {stage_id} has no contract-dry-run adapter preparation")
            else:
                # A runner that raises partway through has usually already found the
                # findings that explain the raise. Keep whatever it returned, and keep
                # nothing if it never returned.
                returned_errors, detail = runner(state, stage)
                errors.extend(returned_errors)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
        results.append({"stage_id": stage_id, "ok": not errors, "errors": errors, **detail})
        # Later stages still run after an earlier finding. The fixture retains the
        # rows that the producer assembled before its refusal so downstream contracts
        # remain independently testable.
    findings = [error for result in results for error in result.get("errors", [])]
    return {
        "ok": not findings,
        "plan_path": str(plan_path.expanduser().resolve()),
        "run_id": plan.get("run_id")
        or (
            plan.get("context", {}).get("run_id")
            if isinstance(plan.get("context"), Mapping)
            else None
        ),
        "source_artifact_root": str(state["source_root"]),
        "from_stage": from_stage,
        "stages_checked": [result["stage_id"] for result in results],
        "fixture_root": str(fixture),
        "source_mode": state.get("source_mode"),
        "provider_calls": sum(len(stub.calls) for stub in state.get("provider_stubs", {}).values()),
        "stages": results,
        "findings": findings,
        "synthesized": {
            "target-msa-manifest": {
                "producer": "claude_binder.adapters.target_msa_builder.run_stage",
                "matching": "the real stage runs on the registered query-only route so the check stays offline; the production route is public-server, which sends the target sequence to a third-party host",
            },
            "control-observations": {
                "producer": "claude_binder.adapters.control_builder.materialize_controls and annotate_rows",
                "matching": "controls are materialized for real and annotated by the real adapter; only the predictor call is withheld, and its rows come from the recorded observations rebased onto the materialized controls, or from the screen raw rows when no run has recorded any",
            },
            "esmfold2-fast-rescore": {
                "producer": "claude_binder.adapters.fal_esmfold2_fast_predictor.parse_outputs",
                "matching": "screen raw rows retain the real row shape, paths, hashes, chain mapping, and measurement fields; phase and seed are changed to the declared rescore matrix and each measurement source is rehashed",
            }
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--from", dest="from_stage", required=True)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="empty --fixture-root even when it holds a directory this tool did not create",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="read source artifacts from this completed run instead of inferring them beside --plan",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The adapters this runs for real print progress to stdout. Stdout here carries the
    # machine-readable report, so their chatter goes to stderr and leaves it parseable.
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = run_contract_dry_run(
                args.plan,
                args.from_stage,
                fixture_root=args.fixture_root,
                artifact_root=args.artifact_root,
                replace=args.replace,
            )
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
