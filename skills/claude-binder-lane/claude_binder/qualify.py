#!/usr/bin/env python3
"""Qualify configured model adapters with a small end-to-end canary.

The command runs adapter-declared canary commands through ``subprocess`` with
``shell=False``. Each command writes a JSON receipt and its output files under
the supplied evidence root. The qualifier converts those receipts into a
portable model roster.

The command prints a cost estimate before it starts a canary. An explicit cost
confirmation is required for dispatch. A row receives ``PASS`` only when the
receipt and output files provide the fields consumed by the runtime validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .adapters import runtime_validator
from .canary_runner import build_assertions, normalize_client_receipt
from . import control_separation
from .paths import child_process_environment, package_root


DEFAULT_CANARY_COUNT = 3
DEFAULT_EVIDENCE_RUNTIME_KEY = "model_roster_evidence_root"
PATH_FIELDS = ("output_paths", "sequence_paths")
DIRECTORY_FIELDS = ("output_dir", "structure_output_dir", "sequence_output_dir", "sequence_dir")
QUALIFICATION_ROLES = frozenset(
    {"backbone-generator", "codesign-generator", "sequence-designer", "cofold-predictor"}
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# The three recorded kinds match the categories that measured-costs.md sorts every
# figure into, so a rate cannot be relabelled on its way into a quote.
COST_KIND_SETTLED = "settled billed amount"
COST_KIND_MODELED = "modeled estimate"
COST_KIND_PROVIDER_RATE = "provider-reported rate"
COST_KIND_UNPRICED = "unpriced"
COST_KIND_OVERRIDE = "operator override"
COST_KIND_UNLABELLED = "unlabelled configured rate"
RECORDED_COST_KINDS = frozenset(
    {COST_KIND_SETTLED, COST_KIND_MODELED, COST_KIND_PROVIDER_RATE, COST_KIND_UNPRICED}
)
TOKEN_RE = re.compile(r"\{\{([A-Za-z0-9_.-]+)\}\}")


class QualificationError(RuntimeError):
    """A qualification request cannot produce an honest roster."""


def _enabled_model_adapter_ids(config: Mapping[str, Any]) -> set[str]:
    """Return model adapters selected by the campaign's enabled arm lists."""
    sections = (
        ("generation", "generators"),
        ("sequence_design", "designers"),
        ("cofold", "predictors"),
    )
    selected: set[str] = set()
    for section_name, item_name in sections:
        section = config.get(section_name)
        items = section.get(item_name) if isinstance(section, Mapping) else None
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping) or item.get("enabled", True) is not True:
                continue
            adapter_id = item.get("adapter_id")
            if isinstance(adapter_id, str) and adapter_id:
                selected.add(adapter_id)
    return selected


def _utc_now() -> str:
    """Return the current UTC time in the roster timestamp format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    """Read one JSON object from disk."""
    if not path.is_file():
        raise QualificationError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(f"{label} is invalid: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must be a JSON object: {path}")
    return value


def _model_adapters(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return enabled design and co-fold model arms that require target qualification."""
    adapters = config.get("adapters")
    if not isinstance(adapters, list):
        raise QualificationError("config.adapters must be a list")
    enabled_adapter_ids = _enabled_model_adapter_ids(config)
    selected: list[dict[str, Any]] = []
    for index, adapter in enumerate(adapters):
        if not isinstance(adapter, dict):
            raise QualificationError(f"config.adapters[{index}] must be an object")
        if adapter.get("model_revision") == "none":
            continue
        if adapter.get("role") not in QUALIFICATION_ROLES:
            continue
        adapter_id = adapter.get("adapter_id")
        if not isinstance(adapter_id, str) or not adapter_id:
            raise QualificationError(f"config.adapters[{index}].adapter_id is required")
        if enabled_adapter_ids and adapter_id not in enabled_adapter_ids:
            continue
        selected.append(adapter)
    if not selected:
        raise QualificationError("config names no target-qualified model arms")
    return selected


def _qualification_spec(
    config: Mapping[str, Any],
    adapter: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge the global and per-adapter qualification settings."""
    result: dict[str, Any] = {}
    global_settings = config.get("qualification")
    if isinstance(global_settings, Mapping):
        result.update({key: value for key, value in global_settings.items() if key != "adapters"})
        per_adapter = global_settings.get("adapters")
        if isinstance(per_adapter, Mapping):
            item = per_adapter.get(adapter.get("adapter_id"))
            if isinstance(item, Mapping):
                result.update(item)
    local_settings = adapter.get("qualification")
    if isinstance(local_settings, Mapping):
        result.update(local_settings)
    return result


def _cost_value(value: Any) -> float | None:
    """Return a finite non-negative cost value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    return float(value)


def format_ceiling_refusal(total: float, ceiling: float) -> str:
    """Name the quote, the ceiling, and the gap between them.

    Four decimal places round a canary quote and its ceiling to the same string,
    so the refusal read "$0.1086 exceeds the ceiling $0.1086". A refusal has to
    name two numbers a reader can tell apart, and the overage settles it.
    """
    overage = total - ceiling
    # A ceiling set from a rounded display of the quote is over by a sliver that
    # rounds to zero, and "exceeds by $0.000000" reads as a contradiction.
    gap = f"${overage:.6f}" if overage >= 5e-7 else "less than $0.000001"
    return (
        f"quoted estimated cost ${total:.6f} exceeds the ceiling ${ceiling:.6f} by {gap}"
    )


def _cost_basis(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the provenance record that labels one adapter's rate."""
    value = spec.get("cost_basis")
    return dict(value) if isinstance(value, Mapping) else {}


def cost_quote(
    config: Mapping[str, Any],
    adapters: Sequence[Mapping[str, Any]],
    canary_count: int,
    overrides: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Calculate the cost quote from explicit configuration values.

    Every priced row carries the evidence kind behind its rate, because a settled
    provider invoice and a modelled figure buy different amounts of confidence and a
    scientist reading a dollar amount is entitled to know which one they have. A row
    with no recorded measurement stays unpriced and states why, rather than borrowing
    a number from a different machine.
    """
    if canary_count < 1:
        raise QualificationError("canary count must be positive")
    override_values = overrides or {}
    items: list[dict[str, Any]] = []
    unpriced: list[dict[str, Any]] = []
    total = 0.0
    complete = True
    for adapter in adapters:
        adapter_id = str(adapter["adapter_id"])
        spec = _qualification_spec(config, adapter)
        provenance = _cost_basis(spec)
        declared_kind = provenance.get("kind")
        configured_per_design = _cost_value(spec.get("cost_per_design_usd"))
        configured_total = _cost_value(spec.get("estimated_cost_usd"))
        if declared_kind == COST_KIND_UNPRICED and (
            configured_per_design is not None or configured_total is not None
        ):
            raise QualificationError(
                f"adapter {adapter_id} declares cost_basis.kind unpriced and also carries a price"
            )
        override = override_values.get(adapter_id)
        if override is not None:
            # An operator who states a rate on the command line has measured their own
            # account, so that rate outranks anything the shipped profile records.
            estimate = override * canary_count
            kind = COST_KIND_OVERRIDE
            basis = f"${override:.6f} per design from --cost-rate"
            row_provenance = None
        elif configured_total is not None:
            estimate = configured_total
            kind = declared_kind or COST_KIND_UNLABELLED
            basis = f"configured canary estimate, {kind}"
            row_provenance = provenance or None
        elif configured_per_design is not None:
            estimate = configured_per_design * canary_count
            kind = declared_kind or COST_KIND_UNLABELLED
            basis = f"${configured_per_design:.6f} per design, {kind}"
            row_provenance = provenance or None
        else:
            estimate = None
            kind = COST_KIND_UNPRICED
            reason = provenance.get("note") or "no cost_per_design_usd or estimated_cost_usd is recorded"
            basis = reason
            row_provenance = provenance or None
            complete = False
            unpriced.append(
                {
                    "adapter_id": adapter_id,
                    "reason": reason,
                    "source": provenance.get("source"),
                }
            )
        if estimate is not None:
            total += estimate
        items.append(
            {
                "adapter_id": adapter_id,
                "estimate_usd": estimate,
                "priced": estimate is not None,
                "cost_kind": kind,
                "basis": basis,
                "provenance": row_provenance,
                "design_count": canary_count,
            }
        )
    return {
        "canary_count": canary_count,
        "items": items,
        "total_usd": total if complete else None,
        "priced_subtotal_usd": total,
        # Naming this a subtotal is not enough. With an arm unpriced the true total is
        # this figure plus an unknown amount, so the only sound claim it supports is a
        # lower bound, and a reader comparing it to a ceiling has to be told that.
        "subtotal_kind": "total" if complete else "floor",
        "priced_adapter_count": sum(1 for item in items if item["priced"]),
        "adapter_count": len(items),
        "unpriced": unpriced,
        "rate_evidence": sorted({item["cost_kind"] for item in items if item["priced"]}),
        "total_kind": "estimate",
        "estimated": True,
        "complete": complete,
    }


def format_cost_quote(quote: Mapping[str, Any]) -> str:
    """Format the cost quote printed before dispatch."""
    lines = [f"qualification cost estimate for {quote.get('canary_count')} designs"]
    for item in quote.get("items", []):
        estimate = item.get("estimate_usd")
        if isinstance(estimate, (int, float)):
            lines.append(f"- {item.get('adapter_id')}: ${estimate:.4f} ({item.get('basis')})")
        else:
            # The reason is long, and the block below states it once per adapter.
            lines.append(f"- {item.get('adapter_id')}: unpriced")
    total = quote.get("total_usd")
    amount = f"${total:.4f}" if isinstance(total, (int, float)) else "unknown"
    lines.append(f"estimated total: {amount}")
    evidence = quote.get("rate_evidence") or []
    if evidence:
        lines.append("rate evidence: " + ", ".join(evidence))
    unpriced = quote.get("unpriced") or []
    if unpriced:
        subtotal = quote.get("priced_subtotal_usd")
        priced_count = quote.get("priced_adapter_count")
        adapter_count = quote.get("adapter_count")
        if isinstance(subtotal, (int, float)):
            lines.append(
                f"priced floor over {priced_count} of {adapter_count} adapters: "
                f"at least ${subtotal:.4f}, and the true total is higher by the cost of "
                f"the {len(unpriced)} unpriced arms"
            )
        lines.append("no recorded price for:")
        for row in unpriced:
            lines.append(f"- {row['adapter_id']}: {row['reason']}")
    return "\n".join(lines)


def _render_token(value: str, context: Mapping[str, Any]) -> str:
    """Render known command tokens and reject unresolved values."""
    if "__REQUIRED__" in value:
        raise QualificationError(f"qualification command contains __REQUIRED__: {value}")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in context:
            reason = context.get(CAMPAIGN_VOCABULARY_ERROR)
            detail = (
                f"; the campaign's own vocabulary could not be resolved, so no campaign token "
                f"is available: {reason}"
                if isinstance(reason, str) and reason
                else ""
            )
            raise QualificationError(
                f"qualification command token is unresolved: {name}{detail}"
            )
        return str(context[name])

    rendered = TOKEN_RE.sub(replace, value)
    if "{{" in rendered or "}}" in rendered:
        raise QualificationError(f"qualification command contains an unresolved token: {value}")
    return rendered


def _command_argv(config: Mapping[str, Any], adapter: Mapping[str, Any], context: Mapping[str, Any]) -> list[str]:
    """Return the adapter-declared command used for the qualification canary."""
    spec = _qualification_spec(config, adapter)
    candidates = (
        spec.get("command_argv"),
        spec.get("command"),
        adapter.get("qualification_command_argv"),
        adapter.get("canary_argv"),
    )
    command = next((candidate for candidate in candidates if candidate is not None), None)
    if not isinstance(command, list) or not command or any(not isinstance(item, str) for item in command):
        raise QualificationError(
            f"adapter {adapter.get('adapter_id')} has no explicit qualification command_argv"
        )
    return [_render_token(item, context) for item in command]


def _require_explicit_qualification_commands(
    config: Mapping[str, Any],
    adapters: Sequence[Mapping[str, Any]],
) -> None:
    """Refuse dispatch until every enabled model arm declares an independent canary."""
    missing: list[str] = []
    for adapter in adapters:
        spec = _qualification_spec(config, adapter)
        commands = (
            spec.get("command_argv"),
            spec.get("command"),
            adapter.get("qualification_command_argv"),
            adapter.get("canary_argv"),
        )
        if not any(isinstance(command, list) and command for command in commands):
            missing.append(str(adapter["adapter_id"]))
    if missing:
        raise QualificationError(
            "enabled model arms have no explicit qualification command_argv: "
            + ", ".join(missing)
        )


def _inside(root: Path, path: Path) -> bool:
    """Return whether a path resolves inside a root."""
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _evidence_path(value: Any, root: Path, run_dir: Path) -> tuple[str | None, Path | None, str | None]:
    """Convert one receipt path into a root-relative path and local path."""
    if not isinstance(value, str) or not value.strip():
        return None, None, "evidence path is missing or is not a string"
    raw = Path(value).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend((root / raw, run_dir / raw))
    chosen = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    resolved = chosen.resolve()
    if not _inside(root, resolved):
        return None, None, f"evidence path escapes the configured root: {value}"
    relative = resolved.relative_to(root.resolve()).as_posix()
    return relative, resolved, None


def _normalize_paths(row: dict[str, Any], root: Path, run_dir: Path) -> list[str]:
    """Normalize all recorded evidence paths and collect path errors."""
    errors: list[str] = []
    for field in PATH_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        if not isinstance(value, list):
            errors.append(f"{field} must be a list of strings")
            continue
        normalized: list[str] = []
        for item in value:
            relative, _path, error = _evidence_path(item, root, run_dir)
            if error:
                errors.append(f"{field}: {error}")
            elif relative is not None:
                normalized.append(relative)
        row[field] = normalized
    for field in DIRECTORY_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        relative, _path, error = _evidence_path(value, root, run_dir)
        if error:
            errors.append(f"{field}: {error}")
        elif relative is not None:
            row[field] = relative
    return errors


def _path_files(row: Mapping[str, Any], field: str, root: Path) -> list[Path]:
    """Return existing files named by one normalized row path field."""
    values = row.get(field)
    if not isinstance(values, list):
        return []
    files: list[Path] = []
    for value in values:
        if isinstance(value, str):
            path = (root / value).resolve()
            if path.is_file() and _inside(root, path):
                files.append(path)
    return files


def _hash_files(row: dict[str, Any], root: Path) -> list[str]:
    """Hash the recorded structure or sequence output files."""
    files = _path_files(row, "output_paths", root)
    if not files:
        files = _path_files(row, "sequence_paths", root)
    hashes: list[str] = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append(digest)
    return hashes


def _json_from_stdout(stdout: str) -> dict[str, Any] | None:
    """Read the last JSON object printed by a canary command."""
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _receipt(path: Path, stdout: str) -> dict[str, Any]:
    """Load a receipt object or use a JSON object printed by the command."""
    if path.is_file():
        value = _read_json(path, "qualification receipt")
    else:
        value = _json_from_stdout(stdout)
        if value is None:
            return {}
    for key in ("row", "evidence"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            merged = dict(value)
            merged.pop(key, None)
            merged.update(nested)
            value = merged
    return value


def _primary_target(config: Mapping[str, Any]) -> str | None:
    """Return the unique primary target named by the configuration."""
    targets = config.get("targets")
    if not isinstance(targets, list):
        return None
    primary = [item for item in targets if isinstance(item, Mapping) and item.get("role") == "primary"]
    if len(primary) == 1 and isinstance(primary[0].get("target_id"), str):
        return str(primary[0]["target_id"])
    return None


def _primary_target_structure_sha256(
    config: Mapping[str, Any],
    config_path: Path,
) -> str | None:
    """Return the primary target hash after checking every live path agrees."""
    targets = config.get("targets")
    if not isinstance(targets, list):
        return None
    primary = [item for item in targets if isinstance(item, Mapping) and item.get("role") == "primary"]
    if len(primary) != 1:
        return None
    target = primary[0]
    recorded = target.get("structure_sha256")
    if recorded is not None and (
        not isinstance(recorded, str) or SHA256_RE.fullmatch(recorded) is None
    ):
        raise QualificationError("the primary target structure_sha256 is invalid")

    observed: dict[str, str] = {}
    for field in ("runtime_structure_path", "structure_path", "structure_source_path"):
        value = target.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.resolve().parent / path
        if path.is_file():
            observed[field] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif field == "runtime_structure_path":
            raise QualificationError(
                f"the materialized primary target is missing: {path}"
            )
    if not observed:
        return str(recorded) if isinstance(recorded, str) else None
    distinct = set(observed.values())
    if len(distinct) != 1:
        details = ", ".join(f"{field}={digest}" for field, digest in observed.items())
        raise QualificationError(f"the primary target paths disagree: {details}")
    digest = next(iter(distinct))
    if isinstance(recorded, str) and recorded != digest:
        raise QualificationError(
            "the primary target structure_sha256 does not match the materialized bytes"
        )
    return str(recorded) if isinstance(recorded, str) else digest


def _cost_for_adapter(quote: Mapping[str, Any], adapter_id: str) -> float | None:
    """Return one adapter's total quoted cost."""
    for item in quote.get("items", []):
        if isinstance(item, Mapping) and item.get("adapter_id") == adapter_id:
            value = item.get("estimate_usd")
            return float(value) if isinstance(value, (int, float)) else None
    return None


def _missing_evidence(
    adapter: Mapping[str, Any],
    row: Mapping[str, Any],
    root: Path,
) -> list[str]:
    """List fields or files the runtime validator still needs."""
    missing: list[str] = []
    for field in runtime_validator.REQUIRED_ROSTER_FIELDS:
        if field not in row:
            missing.append(field)
    for field in ("target_id", "flags_match_production", "container_build", "package_imports"):
        if field not in row:
            missing.append(field)
    expected_image = None
    resources = adapter.get("resources")
    if isinstance(resources, Mapping):
        expected_image = resources.get("container_image_digest")
    if runtime_validator.is_not_applicable(expected_image):
        for field in ("deployment_id", "environment_identity"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                missing.append(field)
    elif not isinstance(row.get("image_id"), str) or not row["image_id"].strip():
        missing.append("image_id")
    weights = row.get("weights_sha256")
    if runtime_validator.is_not_applicable(weights):
        if not isinstance(row.get("weights_revision"), str) or not row["weights_revision"].strip():
            missing.append("weights_revision")
    elif not isinstance(weights, str) or not SHA256_RE.fullmatch(weights):
        missing.append("weights_sha256")
    if row.get("n_designs") is None:
        missing.append("n_designs")
    if row.get("exit_code") != 0:
        missing.append("exit_code=0")
    role = str(adapter.get("role", ""))
    output_field = "sequence_paths" if role == "sequence-designer" else "output_paths"
    output_paths = _path_files(row, output_field, root)
    if not output_paths:
        missing.append(f"{output_field} files")
    if not isinstance(row.get("output_sha256"), list) or not row["output_sha256"]:
        missing.append("output_sha256")
    else:
        actual_hashes = _hash_files(dict(row), root)
        if actual_hashes and row["output_sha256"] != actual_hashes:
            missing.append("output_sha256 matches gathered files")
    if row.get("sequence_adapter_consumed") is not True:
        missing.append("sequence_adapter_consumed=true")
    count = row.get("sequence_output_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        missing.append("sequence_output_count")
    return list(dict.fromkeys(missing))


# The definition lives in `runtime_validator`, which is what reads a roster back. One reader,
# so a rule added there cannot be missed here. `build_row` fills an absent `target_id` and
# target digest from the configuration, so without this a receipt that consumed a shipped
# fixture would come out carrying the campaign's target identity. A roster row that borrows
# another target's control separation forges the gate it is supposed to pass.
UNQUALIFIABLE_EVIDENCE_MODES = runtime_validator.UNQUALIFIABLE_EVIDENCE_MODES
UNQUALIFIABLE_QUALIFICATION_SCOPES = runtime_validator.UNQUALIFIABLE_QUALIFICATION_SCOPES
_unqualifiable_provenance = runtime_validator.unqualifiable_provenance


def build_row(
    config: Mapping[str, Any],
    adapter: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    config_path: Path,
    evidence_root: Path,
    run_dir: Path,
    exit_code: int,
    wall_clock_s: float,
    quote: Mapping[str, Any],
    path_errors: Sequence[str] = (),
) -> dict[str, Any]:
    """Build one roster row from one command result and its recorded artifacts."""
    row = dict(receipt)
    adapter_id = str(adapter["adapter_id"])
    row.setdefault("adapter_id", adapter_id)
    row.setdefault("model", adapter.get("model_id") or adapter.get("model") or adapter_id)
    row.setdefault("source_revision", adapter.get("source_revision"))
    row.setdefault("model_revision", adapter.get("model_revision"))
    # The campaign is authoritative about which target this row qualifies, and a
    # receipt is not. `setdefault` let a receipt that already named a target keep its
    # own value, so a receipt carrying `primary-target` survived into a row for a
    # different campaign, and a receipt carrying a stale structure hash survived into a
    # row whose hash is the only thing binding it to real bytes. That defeats the
    # binding: the check downstream compares the row's hash with the materialized
    # target, and a row that copied its hash from the receipt passes that check by
    # construction rather than by measurement.
    #
    # A receipt that names a different target is not a value to be overridden quietly.
    # It says the canary measured something other than what this campaign is about to
    # run, so it is an error.
    target_id = _primary_target(config)
    if target_id is not None:
        recorded = row.get("target_id")
        if recorded is not None and recorded != target_id:
            raise QualificationError(
                f"{adapter_id} canary receipt names target {recorded!r}, but this campaign's "
                f"primary target is {target_id!r}; the receipt did not qualify this target"
            )
        row["target_id"] = target_id
    target_structure_sha256 = _primary_target_structure_sha256(config, config_path)
    if target_structure_sha256 is not None:
        recorded = row.get("target_structure_sha256")
        if recorded is not None and recorded != target_structure_sha256:
            raise QualificationError(
                f"{adapter_id} canary receipt records target structure "
                f"{recorded}, but this campaign's materialized target hashes to "
                f"{target_structure_sha256}; the receipt did not qualify these bytes"
            )
        row["target_structure_sha256"] = target_structure_sha256
    row["exit_code"] = exit_code
    row.setdefault("wall_clock_s", round(wall_clock_s, 3))
    row.setdefault("validated_at", _utc_now())
    count = row.get("n_designs")
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        row.setdefault("s/design", round(float(row["wall_clock_s"]) / count, 4))
    adapter_cost = _cost_for_adapter(quote, adapter_id)
    if adapter_cost is not None and isinstance(count, int) and count > 0:
        row.setdefault("$/design", round(adapter_cost / count, 6))
        row.setdefault("cost_basis", "configuration estimate quoted before this canary")
    errors = list(path_errors)
    errors.extend(_normalize_paths(row, evidence_root, run_dir))
    actual_hashes = _hash_files(row, evidence_root)
    recorded_hashes = row.get("output_sha256")
    if actual_hashes:
        if recorded_hashes is not None and recorded_hashes != actual_hashes:
            errors.append("output_sha256 does not match the gathered output files")
        row["output_sha256"] = actual_hashes
    if "status" not in row or str(row.get("status")).upper() == "PASS":
        row["status"] = "PENDING"
    missing = _missing_evidence(adapter, row, evidence_root)
    missing.extend(errors)
    missing.extend(_unqualifiable_provenance(row))
    row["qualification_mode"] = "fresh"
    row["qualification_missing_evidence"] = list(dict.fromkeys(missing))
    if exit_code == 0 and not missing:
        row["status"] = "PASS"
    else:
        row["status"] = "PENDING"
    return row


def write_roster(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    qualification_mode: str = "fresh",
    qualified_at: str | None = None,
    runtime_key: str = DEFAULT_EVIDENCE_RUNTIME_KEY,
    control_separation_record: Mapping[str, Any] | None = None,
    dispatch_record: Mapping[str, Any] | None = None,
) -> None:
    """Write a portable roster and reject unsupported PASS rows."""
    if qualification_mode not in {"fresh", "inherited"}:
        raise QualificationError(f"qualification mode is invalid: {qualification_mode}")
    checked_rows: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        for field in PATH_FIELDS:
            values = row.get(field)
            if isinstance(values, list) and any(isinstance(value, str) and Path(value).is_absolute() for value in values):
                raise QualificationError(f"row {index} contains an absolute {field} path")
        missing = row.get("qualification_missing_evidence")
        if str(row.get("status", "")).upper() == "PASS":
            provenance = _unqualifiable_provenance(row)
            if provenance:
                raise QualificationError(
                    f"row {index} is PASS but its evidence cannot qualify: "
                    + "; ".join(provenance)
                )
            required_missing = [
                field for field in runtime_validator.REQUIRED_ROSTER_FIELDS if field not in row
            ]
            if required_missing or missing:
                details = required_missing or list(missing)
                raise QualificationError(
                    f"row {index} cannot be written as PASS with missing evidence: {details}"
                )
        checked_rows.append(row)
    qualified_at = qualified_at or _utc_now()
    qualification: dict[str, Any] = {
        "mode": qualification_mode,
        "scope": "deployments named by the roster rows",
        "source": "claude_binder.qualify",
        "qualified_at": qualified_at,
        "inherited": qualification_mode == "inherited",
    }
    if control_separation_record is not None:
        qualification["control_separation"] = dict(control_separation_record)
    if dispatch_record is not None:
        # A reader has to be able to tell a row this qualifier dispatched from a row
        # built out of a receipt harvested elsewhere, because only the first kind
        # names a process that watched the canary run.
        qualification["dispatch"] = dict(dispatch_record)
    document = {
        "schema_version": 1,
        "ledger_type": "claude-binder-model-roster",
        "evidence_root": {
            "runtime_key": runtime_key,
            "path_policy": "row output_paths and sequence_paths are relative to this runtime directory",
        },
        "qualification": qualification,
        "models": checked_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")


CAMPAIGN_VOCABULARY_ERROR = "_campaign_vocabulary_error"
"""Context key carrying why the campaign's own vocabulary could not be resolved."""


def canary_data_dir() -> Path | None:
    """The shipped canary fixture directory, or None when the package ships none.

    A canary argv may not carry an absolute path, because build validation strips account
    and machine identifiers from the installable skill. This resolves the directory from
    the installed package instead, so an argv names a shipped fixture by token.
    """
    candidate = package_root() / "data" / "roster-evidence" / "canary"
    return candidate if candidate.is_dir() else None


def _campaign_vocabulary(config: Mapping[str, Any]) -> dict[str, Any]:
    """The campaign's own command vocabulary, or nothing when it cannot be resolved.

    Without this the canary context holds only the keys built below plus the scalars in
    the adapter's `qualification` block, which leaves `{{proteinmpnn_fal_url}}` and
    `{{target_structure}}` unresolved and no shipped argv able to name a deployment.

    A campaign that cannot resolve its own context still gets a canary. The resolution
    raises for a campaign missing a residue map or site residues, and a canary that
    refused for that reason would be refusing for a reason unrelated to qualification.
    """
    from . import lane

    try:
        resolved = lane.resolved_context(dict(config))
    except Exception as exc:  # noqa: BLE001
        # Swallowing this silently makes an endpoint token disappear and the canary refuse
        # with `qualification command token is unresolved`, which names a symptom rather
        # than the cause. The reason travels with the context so the refusal can say it.
        return {CAMPAIGN_VOCABULARY_ERROR: f"{type(exc).__name__}: {exc}"}
    return dict(resolved) if isinstance(resolved, Mapping) else {}


def canary_context(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    adapter_id: str,
    evidence_root: Path,
    run_dir: Path,
    receipt_path: Path,
    canary_count: int,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the substitution context one canary argv is rendered against.

    Three layers. The campaign's resolved vocabulary, then the canary's own paths and
    counts, then the scalars the adapter declares beside its argv. The second layer
    overwrites the first, so a campaign key can never shadow the canary's `run_dir`. The
    third only fills keys nothing else supplied, so an adapter scalar cannot shadow a
    campaign value either. Precedence differs between the layers and is not uniform.

    `plan_path` is removed after all three, so a spec declaring one cannot put it back. A
    canary runs before any plan is materialized, so a token naming one would resolve to a
    path that does not exist, and an argv that needs it describes a stage rather than a
    canary.
    """
    context: dict[str, Any] = _campaign_vocabulary(config)
    fixtures = canary_data_dir()
    context.update(
        {
            "adapter_id": adapter_id,
            "canary_count": canary_count,
            "count": canary_count,
            "stage": "qualification",
            "phase": "canary",
            "evidence_root": evidence_root,
            "run_dir": run_dir,
            "attempt_dir": run_dir,
            "receipts_dir": run_dir,
            "artifact_root": evidence_root,
            "output_dir": run_dir,
            "receipt_path": receipt_path,
            "config_path": config_path.resolve(),
            "python_executable": sys.executable,
            "target_id": _primary_target(config) or "",
        }
    )
    if fixtures is not None:
        context["canary_data_dir"] = fixtures
    for key, value in spec.items():
        if isinstance(value, (str, int, float)) and key not in context:
            context[key] = value
    # Removed last, after every layer. Popping it before the adapter's scalars ran let a spec
    # declaring `plan_path` put it back, and a canary runs before any plan is materialized.
    context.pop("plan_path", None)
    return context


def _dependency_ordered(adapters: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Order canaries so an arm runs after the arm whose output it consumes.

    The shipped canary runner designs on the generator's backbones and folds the designer's
    sequences, so configuration order would run an arm before its input exists.
    An arm the runner does not order stays where the configuration put it, and an arm whose
    input is genuinely missing still refuses rather than inventing one.
    """
    from . import canary_runner

    order = {adapter_id: index for index, adapter_id in enumerate(canary_runner.DISPATCH_ORDER)}
    return sorted(
        adapters,
        key=lambda adapter: (
            order.get(str(adapter.get("adapter_id")), len(order)),
            list(adapters).index(adapter),
        ),
    )


def _run_adapter(
    config: Mapping[str, Any],
    config_path: Path,
    adapter: Mapping[str, Any],
    evidence_root: Path,
    canary_count: int,
    quote: Mapping[str, Any],
    executor: Callable[..., Any],
) -> dict[str, Any]:
    """Run one adapter canary and convert its receipt into a roster row."""
    adapter_id = str(adapter["adapter_id"])
    run_dir = evidence_root / "canary" / adapter_id
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = _qualification_spec(config, adapter)
    receipt_value = spec.get("receipt_path", "receipt.json")
    if not isinstance(receipt_value, str) or not receipt_value:
        raise QualificationError(f"adapter {adapter_id} receipt_path must be a string")
    receipt_path = Path(receipt_value).expanduser()
    if not receipt_path.is_absolute():
        receipt_path = run_dir / receipt_path
    receipt_path = receipt_path.resolve()
    if not _inside(evidence_root, receipt_path):
        raise QualificationError(f"adapter {adapter_id} receipt path escapes the evidence root")
    context = canary_context(
        config,
        config_path=config_path,
        adapter_id=adapter_id,
        evidence_root=evidence_root,
        run_dir=run_dir,
        receipt_path=receipt_path,
        canary_count=canary_count,
        spec=spec,
    )
    try:
        command = _command_argv(config, adapter, context)
    except QualificationError as exc:
        return build_row(
            config,
            adapter,
            {},
            config_path=config_path,
            evidence_root=evidence_root,
            run_dir=run_dir,
            exit_code=127,
            wall_clock_s=0.0,
            quote=quote,
            path_errors=[str(exc)],
        )
    extra_environment = spec.get("environment")
    # A canary argv may run this package as `python -m claude_binder...`, so the child
    # needs a route to the package a file-path host never puts on sys.path.
    environment = child_process_environment(
        {str(key): str(value) for key, value in extra_environment.items()}
        if isinstance(extra_environment, Mapping)
        else None
    )
    started = time.monotonic()
    try:
        completed = executor(
            command,
            cwd=run_dir,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        exit_code = int(getattr(completed, "returncode", 1))
        stdout = str(getattr(completed, "stdout", "") or "")
        stderr = str(getattr(completed, "stderr", "") or "")
    except OSError as exc:
        exit_code = 127
        stdout = ""
        stderr = str(exc)
    elapsed = time.monotonic() - started
    receipt = _receipt(receipt_path, stdout)
    if stderr:
        receipt.setdefault("qualification_stderr", stderr[-4000:])
    return build_row(
        config,
        adapter,
        receipt,
        config_path=config_path,
        evidence_root=evidence_root,
        run_dir=run_dir,
        exit_code=exit_code,
        wall_clock_s=elapsed,
        quote=quote,
    )


def _row_from_supplied_receipt(
    config: Mapping[str, Any],
    config_path: Path,
    adapter: Mapping[str, Any],
    evidence_root: Path,
    quote: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    """Build one roster row from a receipt this process did not dispatch.

    A Modal canary cannot be dispatched from here. ``_run_adapter`` runs its
    command through ``subprocess``, the adapter programs carry no Modal route,
    and Modal submission needs the Claude Science kernel ``host`` object that no
    subprocess holds. Modal also reports completion through an out-of-cell
    notification, so no synchronous executor can wait for one. The frame that
    owns ``host`` therefore runs the canary and harvests its receipt, and this
    turns that receipt into a row.

    Handing in a receipt skips the dispatch, not the evidence. ``build_row``
    still rehashes every output file it names, still binds the row to the
    campaign's own target id and materialized structure hash, and still refuses
    a receipt that names a different target. The runtime validator then parses
    the structures itself. A receipt cannot assert its way past any of that.
    """
    adapter_id = str(adapter["adapter_id"])
    run_dir = evidence_root / "canary" / adapter_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Normalize the same way a receipt this package dispatched is normalized. A frame
    # harvesting from a deployment writes that deployment's own vocabulary, and this route
    # exists for exactly those receipts, so reading them raw left the roster missing facts
    # the receipt carried under another name.
    receipt = normalize_client_receipt(_read_json(receipt_path, f"{adapter_id} canary receipt"))
    # The same rule the dispatching route applies, from the same provider-returned fields.
    # A receipt already carrying either assertion keeps what it carries.
    for field, value in build_assertions(receipt).items():
        receipt.setdefault(field, value)
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    exit_code = receipt.get("exit_code")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise QualificationError(
            f"{adapter_id} canary receipt records no integer exit_code: {receipt_path}"
        )
    wall_clock = receipt.get("wall_clock_s")
    receipt["qualification_dispatch"] = "supplied-receipt"
    receipt["qualification_receipt_sha256"] = digest
    resolved = receipt_path.resolve()
    receipt["qualification_receipt_source"] = (
        str(resolved.relative_to(evidence_root)) if _inside(evidence_root, resolved) else resolved.name
    )
    return build_row(
        config,
        adapter,
        receipt,
        config_path=config_path,
        evidence_root=evidence_root,
        run_dir=run_dir,
        exit_code=exit_code,
        wall_clock_s=float(wall_clock) if isinstance(wall_clock, (int, float)) and not isinstance(wall_clock, bool) else 0.0,
        quote=quote,
    )


def run_qualification(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    output_path: Path,
    evidence_root: Path,
    canary_count: int = DEFAULT_CANARY_COUNT,
    cost_overrides: Mapping[str, float] | None = None,
    confirm_cost: bool = False,
    max_cost_usd: float | None = None,
    executor: Callable[..., Any] = subprocess.run,
    emit_quote: bool = True,
    supplied_receipts: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Run all configured model canaries and write their fresh roster."""
    adapters = _model_adapters(config)
    supplied = dict(supplied_receipts or {})
    validate_supplied_receipts(adapters, supplied)
    dispatched = [adapter for adapter in adapters if str(adapter["adapter_id"]) not in supplied]
    quote = cost_quote(config, adapters, canary_count, cost_overrides)
    if emit_quote:
        print(format_cost_quote(quote))
    # Only the arms this call will dispatch are gated on cost. A supplied receipt
    # was already paid for elsewhere, so quoting it again and holding the roster
    # behind a confirmation for spend that will not happen refuses a run for a
    # reason that is not true of it.
    if dispatched:
        dispatch_quote = cost_quote(config, dispatched, canary_count, cost_overrides)
        if not dispatch_quote["complete"]:
            raise QualificationError("cost estimate is incomplete; supply a cost rate for every model adapter")
        # A canary is a real paid provider job, so it is bound by a stated ceiling like
        # any other. `confirm_cost` confirms a price the caller has seen; it binds no
        # number, so on its own it let a repriced arm dispatch above what was reviewed.
        if max_cost_usd is None:
            raise QualificationError(
                "a spend ceiling is required before canary dispatch: pass --max-cost-usd, "
                f"and the quoted total is {float(dispatch_quote['total_usd'])} USD"
            )
        if (
            isinstance(max_cost_usd, bool)
            or not isinstance(max_cost_usd, (int, float))
            or not math.isfinite(float(max_cost_usd))
            or float(max_cost_usd) <= 0
        ):
            raise QualificationError(
                "the canary spend ceiling must be a positive finite USD amount"
            )
        if float(dispatch_quote["total_usd"]) > max_cost_usd:
            raise QualificationError(
                format_ceiling_refusal(float(dispatch_quote["total_usd"]), float(max_cost_usd))
            )
        if not confirm_cost:
            raise QualificationError("cost confirmation is required before canary dispatch")
        _require_explicit_qualification_commands(config, dispatched)
    evidence_root = evidence_root.expanduser().resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    adapters = _dependency_ordered(adapters)
    rows = [
        _row_from_supplied_receipt(
            config,
            config_path,
            adapter,
            evidence_root,
            quote,
            supplied[str(adapter["adapter_id"])],
        )
        if str(adapter["adapter_id"]) in supplied
        else _run_adapter(
            config,
            config_path,
            adapter,
            evidence_root,
            canary_count,
            quote,
            executor,
        )
        for adapter in adapters
    ]
    qualified_at = _utc_now()
    separation = control_separation.build_qualification_record(
        config,
        qualified_at=qualified_at,
    )
    dispatch_record = {
        "dispatched_here": sorted(str(adapter["adapter_id"]) for adapter in dispatched),
        "supplied_receipts": sorted(supplied),
    }
    write_roster(
        output_path.expanduser().resolve(),
        rows,
        qualified_at=qualified_at,
        control_separation_record=separation,
        dispatch_record=dispatch_record,
    )
    print("qualification: fresh")
    for row in rows:
        print(f"- {row['adapter_id']}: {row['status']}")
    print(f"wrote {output_path.expanduser().resolve()}")
    return {
        "qualification": {
            "mode": "fresh",
            "inherited": False,
            "control_separation": separation,
            "dispatch": dispatch_record,
        },
        "cost": quote,
        "evidence_root": str(evidence_root),
        "roster_path": str(output_path.expanduser().resolve()),
        "models": rows,
    }


def validate_supplied_receipts(
    adapters: Sequence[Mapping[str, Any]],
    supplied: Mapping[str, Path],
) -> None:
    """Refuse a receipt that names no arm this campaign qualifies.

    Both entry points call this. The quote path reaches no further than a price,
    so without this a mistyped adapter id would leave every arm scheduled for
    dispatch and report a clean quote, which reads as if the receipt had been
    accepted.
    """
    known = {str(adapter["adapter_id"]) for adapter in adapters}
    unknown = sorted(set(supplied) - known)
    if unknown:
        raise QualificationError(
            "receipt names an adapter this campaign does not qualify: "
            + ", ".join(unknown)
            + "; qualified arms are: "
            + ", ".join(sorted(known))
        )


def _parse_supplied_receipts(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeated ``adapter=path`` harvested-receipt options."""
    result: dict[str, Path] = {}
    for value in values:
        adapter_id, separator, raw_path = value.partition("=")
        if not separator or not adapter_id or not raw_path:
            raise QualificationError(f"receipt must use adapter_id=path: {value}")
        if adapter_id in result:
            raise QualificationError(f"receipt is given twice for adapter {adapter_id}")
        result[adapter_id] = Path(raw_path).expanduser()
    return result


def _parse_cost_overrides(values: Sequence[str]) -> dict[str, float]:
    """Parse repeated ``adapter=value`` cost-rate options."""
    result: dict[str, float] = {}
    for value in values:
        adapter_id, separator, amount = value.partition("=")
        if not separator or not adapter_id:
            raise QualificationError(f"cost rate must use adapter_id=value: {value}")
        parsed = _cost_value(float(amount)) if amount else None
        if parsed is None:
            raise QualificationError(f"cost rate is invalid: {value}")
        result[adapter_id] = parsed
    return result


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by estimate and run."""
    parser.add_argument("--config", type=Path, required=True, help="resolved campaign config")
    parser.add_argument("--canary-count", type=int, default=DEFAULT_CANARY_COUNT, help="designs per model canary")
    parser.add_argument(
        "--cost-rate",
        action="append",
        default=[],
        metavar="ADAPTER=USD",
        help="explicit cost per design for an adapter",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the qualification CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    estimate = subparsers.add_parser("estimate", help="print the cost before dispatch")
    _add_common_arguments(estimate)
    run = subparsers.add_parser("run", help="run the canary and write a model roster")
    _add_common_arguments(run)
    run.add_argument("--output", type=Path, required=True, help="roster path to write")
    run.add_argument("--evidence-root", type=Path, required=True, help="directory for canary evidence")
    run.add_argument("--confirm-cost", action="store_true", help="confirm the printed cost estimate")
    run.add_argument("--max-cost-usd", type=float, default=None, help="maximum accepted quoted cost")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested qualification CLI command."""
    args = build_parser().parse_args(argv)
    try:
        config_path = args.config.expanduser().resolve()
        config = _read_json(config_path, "campaign config")
        adapters = _model_adapters(config)
        overrides = _parse_cost_overrides(args.cost_rate)
        quote = cost_quote(config, adapters, args.canary_count, overrides)
        print(format_cost_quote(quote))
        if args.command == "estimate":
            return 0 if quote["complete"] else 1
        if not quote["complete"]:
            print(
                "qualification: ERROR: cost estimate is incomplete; supply a rate for every model adapter",
                file=sys.stderr,
            )
            return 2
        if args.max_cost_usd is not None and float(quote["total_usd"]) > args.max_cost_usd:
            print(
                "qualification: ERROR: "
                + format_ceiling_refusal(float(quote["total_usd"]), float(args.max_cost_usd)),
                file=sys.stderr,
            )
            return 2
        if not args.confirm_cost:
            print("qualification: ERROR: pass --confirm-cost after reviewing the estimate", file=sys.stderr)
            return 2
        run_qualification(
            config,
            config_path=config_path,
            output_path=args.output,
            evidence_root=args.evidence_root,
            canary_count=args.canary_count,
            cost_overrides=overrides,
            confirm_cost=True,
            max_cost_usd=args.max_cost_usd,
            emit_quote=False,
        )
        return 0
    except (OSError, QualificationError, ValueError, json.JSONDecodeError) as exc:
        print(f"qualification: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
