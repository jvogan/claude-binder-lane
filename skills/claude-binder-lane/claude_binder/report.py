"""Build a self-contained HTML report from a completed design run.

The reader accepts the package run layout and the two external run layouts used
by the binder workflow. It preserves metric names from the input artifacts.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import mimetypes
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import structural_surrogates
from . import arms
from .filter_contracts import retired_filter_statuses

_MISSING = object()
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"}
_RESULT_NAMES = ("results-all.json", "results.json")
_CONFIG_NAMES = (
    "config.resolved.json",
    "runtime-config.resolved.json",
    "campaign.resolved.json",
    "config.json",
    "campaign.json",
)
_METADATA_KEYS = {
    "adapter_id",
    "aligned_target_residue_count",
    "attempt_id",
    "best_index",
    "candidate_id",
    "chain_mapping",
    "control_role",
    "control_type",
    "coverage",
    "coverage_complete",
    "design_pose_path",
    "design_pose_sha256",
    "elapsed_seconds",
    "entities",
    "epitope_constraint",
    "filter_pass",
    "generator",
    "gates",
    "id",
    "input",
    "inference_parameters",
    "model_revision",
    "origin_generator",
    "phase",
    "predictor",
    "predictor_coverage",
    "predictors_required",
    "rank",
    "portfolio_rank",
    "raw_prediction_record_sha256",
    "request",
    "resolved_snapshots",
    "samples",
    "seed",
    "selected_seed_by_predictor",
    "sequence_length",
    "sequence_path",
    "sequence_sha256",
    "status",
    "surrogate_disclosure",
    "target_id",
    "target_sha256",
    "target_alignment_rmsd",
    "target_chain_id",
    "binder_chain_id",
    "timestamp",
    "tool",
    "tool_revision",
    "chain_mapping",
}
_RESPONSE_METADATA_KEYS = _METADATA_KEYS | {
    "adapters",
    "adapters_sha256",
    "ccd",
    "chains",
    "environment",
    "environment_revision",
    "expected_chain_lengths",
    "input_chain_order",
    "mmcif",
    "nonfinite_svd_repairs",
    "pair_chains_iptm",
    "plddt",
    "ptm",
    "request",
    "timings",
    "worker_sha256",
}
_OBSERVATION_METRIC_KEYS = {
    "ipsae_target_to_binder",
    "ipsae_binder_to_target",
    "ipsae_min",
    "sc_dockq",
    "dockq",
    "fnat",
    "interface_rmsd",
    "ligand_rmsd",
    "mapping_status",
    "aligned_target_residue_count",
    "target_alignment_rmsd",
    "site_contact_iou",
    "target_contact_recall",
    "target_contact_precision",
    "hotspot_recovery",
    "offsite_contact_fraction",
    "iptm",
    "interface_pae",
    "interface_plddt",
    "clash_count",
    "contact_count",
}
_SCORER_METRIC_KEYS = _OBSERVATION_METRIC_KEYS | {
    "ipsae_target_to_binder",
    "ipsae_binder_to_target",
    "ipsae_min",
    "dockq",
    "fnat",
    "interface_rmsd",
    "ligand_rmsd",
    "site_contact_iou",
    "target_contact_recall",
    "target_contact_precision",
    "hotspot_recovery",
    "offsite_contact_fraction",
}
_SCORER_SUMMARY_METRICS = (
    "ipsae_min",
    "sc_dockq",
    "fnat",
    "interface_rmsd",
    "target_contact_recall",
    "offsite_contact_fraction",
)
DEFAULT_MONOMER_CONFIDENCE_THRESHOLD = 0.70
_MONOMER_CONFIDENCE_KEYS = ("monomer_confidence", "mean_plddt", "plddt")
_MONOMER_ARTIFACT_DIRECTORIES = ("monomer", "monomer-refold")
_MONOMER_INTERPRETATION_NOTE = (
    "Complex score measures the predicted target-binder complex. "
    "Monomer confidence measures whether the binder predicts a folded single chain. "
    "Treat scores with monomer confidence below the configured threshold as uninterpretable because high complex scores alone do not establish target specificity."
)

# The decision table accepts only stored central estimates. The legacy
# ``ipsae_min_ensemble`` value can be a maximum seed value, so it is excluded.
_MEAN_IPSAE_FIELDS = (
    "mean_ipsae_min",
    "ipsae_min_mean",
    "ipsae_min_central_estimate",
)
_SD_IPSAE_FIELDS = (
    "sd_ipsae_min",
    "ipsae_min_sd",
    "ipsae_min_standard_deviation",
)
_FOLD_COUNT_FIELDS = ("n_folds", "fold_count", "n_seeds", "seed_count")
_INTERFACE_ERROR_FIELDS = (
    "iface_err",
    "interface_error",
    "interface_pae",
    "interface_pae_ensemble",
)
_RUN_ID_FIELDS = ("run_id", "campaign_run_id", "id")
_FORBIDDEN_REPORT_TERMS = (
    "binding probability",
    "star rating",
    "best-of-fold",
    "best of fold",
    "best fold",
    "validated",
    "confirmed",
    "novel",
    "optimized",
    "potency",
    "affinity",
    "confidence interval",
    "percent complete",
    "tm-score",
)


@dataclass(frozen=True)
class DecisionRow:
    """One artifact-backed row on the scientist-facing decision page."""

    candidate: Candidate
    mean_ipsae_min: float | None
    mean_field: str | None
    sd_ipsae_min: float | None
    sd_field: str | None
    n_folds: float | None
    n_folds_field: str | None
    interface_error: float | None
    interface_error_field: str | None
    monomer_confidence: Metric | None
    length: int | None
    length_field: str | None


@dataclass(frozen=True)
class ControlGate:
    """The control state recorded with the final ranking artifact."""

    state: str
    reason: str | None
    source: Provenance | None


@dataclass(frozen=True)
class Provenance:
    """The artifact fields that explain one displayed value."""

    artifact: str
    tool: str | None = None
    revision: str | None = None
    seed: str | int | float | None = None
    detail: str | None = None


@dataclass(frozen=True)
class Metric:
    name: str
    value: Any
    source: Provenance
    context: str = ""


@dataclass
class Candidate:
    candidate_id: str
    sequence: str | None = None
    sequence_source: Provenance | None = None
    metrics: list[Metric] = field(default_factory=list)
    rank: Any = _MISSING
    rank_source: Provenance | None = None
    rank_row: dict[str, Any] | None = None
    structure_paths: list[str] = field(default_factory=list)
    image_path: Path | None = None
    image_uri: str | None = None
    source_paths: list[str] = field(default_factory=list)
    record: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RescoreRecord:
    """One requested or discovered ESMFold2-Fast rescore record."""

    candidate_id: str
    seed: int | str
    response_path: Path | None
    error_path: Path | None
    source: Provenance


@dataclass(frozen=True)
class ScorerRow:
    """One row emitted by the local structure scorer."""

    record_type: str
    candidate_id: str
    source_name: str
    values: Mapping[str, Any]
    source: Provenance


@dataclass(frozen=True)
class StageCount:
    name: str
    entering: Any
    surviving: Any
    entering_source: Provenance | None
    surviving_source: Provenance | None
    note: str = ""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_stream(path: Path) -> list[Any]:
    """Read JSON, JSONL, or concatenated pretty-printed JSON objects."""

    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    values: list[Any] = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        value, end = decoder.raw_decode(text, position)
        values.append(value)
        position = end
    return values


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _typed_tsv_value(value: str | None) -> Any:
    if value is None or value == "":
        return _MISSING
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", value):
        return float(value)
    return value


def _relative_artifact(path: Path, root: Path) -> str:
    return _display_path(path, root)


def _relative_artifacts(paths: Iterable[Path], root: Path) -> str:
    """Return a stable, compact source list for related artifacts."""

    return "; ".join(dict.fromkeys(_relative_artifact(path, root) for path in paths))


def _first_value(record: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _nested_revision(record: Mapping[str, Any]) -> str | None:
    revisions: list[str] = []
    for key in ("revision", "model_revision", "source_revision", "tool_revision", "environment_revision"):
        value = record.get(key)
        if value is not None and value != "":
            revisions.append(str(value))
    for key in ("engine_version", "pipeline_version"):
        value = record.get(key)
        if value is not None and value != "":
            revisions.append(f"{key}={value}")
    snapshots = record.get("resolved_snapshots")
    if isinstance(snapshots, Mapping):
        for name, snapshot in snapshots.items():
            if isinstance(snapshot, Mapping) and snapshot.get("revision"):
                revisions.append(f"{name}={snapshot['revision']}")
    adapters = record.get("adapters")
    if isinstance(adapters, Mapping) and adapters.get("arm"):
        arm = adapters["arm"]
        if isinstance(arm, Mapping) and arm.get("revision"):
            revisions.append(str(arm["revision"]))
    unique = list(dict.fromkeys(revisions))
    return "; ".join(unique) if unique else None


def _provenance(
    path: Path,
    root: Path,
    record: Mapping[str, Any] | None = None,
    run_records: Iterable[Mapping[str, Any]] = (),
    detail: str | None = None,
) -> Provenance:
    record = record or {}
    records = [record, *run_records]
    tool: Any = None
    revision: Any = None
    seed: Any = None
    for item in records:
        if not isinstance(item, Mapping):
            continue
        if tool is None:
            tool = _first_value(
                item,
                (
                    "tool",
                    "tool_id",
                    "predictor",
                    "adapter_id",
                    "designer",
                    "sequence_designer",
                    "engine",
                    "pipeline",
                    "generator",
                    "application",
                ),
            )
            endpoint = item.get("fal_endpoint")
            if tool is None and isinstance(endpoint, str) and endpoint.rstrip("/"):
                tool = endpoint.rstrip("/").rsplit("/", 1)[-1]
            adapters = item.get("adapters")
            if tool is None and isinstance(adapters, Mapping):
                arm = adapters.get("arm")
                if isinstance(arm, Mapping):
                    tool = _first_value(arm, ("predictor_id", "adapter_id", "checkpoint"))
        if revision is None:
            revision = _nested_revision(item)
        if seed is None:
            seed = _first_value(item, ("seed", "used_seed", "requested_seed", "random_seed", "rng_seed"))
            request = item.get("request")
            if seed is None and isinstance(request, Mapping):
                seed = _first_value(request, ("seed", "random_seed", "rng_seed"))
    return Provenance(
        artifact=_relative_artifact(path, root),
        tool=str(tool) if tool is not None else None,
        revision=str(revision) if revision is not None else None,
        seed=seed,
        detail=detail,
    )


def _missing(reason: str) -> str:
    return f'<span class="missing">Missing: {html.escape(reason)}</span>'


def _provenance_html(source: Provenance) -> str:
    def value(value: Any) -> str:
        return html.escape(str(value)) if value is not None else "missing in run"

    parts = [
        f"source: {value(source.artifact)}",
        f"tool: {value(source.tool)}",
        f"revision: {value(source.revision)}",
        f"seed: {value(source.seed)}",
    ]
    if source.detail:
        parts.append(f"detail: {value(source.detail)}")
    return '<span class="provenance">' + "; ".join(parts) + "</span>"


def _number(value: Any, source: Provenance | None, missing_reason: str) -> str:
    if value is _MISSING or value is None:
        return _missing(missing_reason)
    if not _is_number(value):
        return _missing(f"a numeric value in {missing_reason}")
    shown = str(int(value)) if isinstance(value, int) or float(value).is_integer() else format(float(value), ".10g")
    if source is None:
        source = Provenance("missing source")
    return f'<span class="number">{html.escape(shown)} {_provenance_html(source)}</span>'


def _value_html(value: Any, source: Provenance, label: str = "value") -> str:
    if value is _MISSING or value is None:
        return _missing(label)
    if _is_number(value):
        return _number(value, source, label)
    if isinstance(value, str):
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value.strip()):
            parsed: int | float = int(value) if "." not in value else float(value)
            return _number(parsed, source, label)
        rendered = html.escape(value)
        if any(character.isdigit() for character in value):
            rendered += " " + _provenance_html(source)
        return rendered
    if isinstance(value, Mapping):
        rows = []
        for key, child in value.items():
            rows.append(f"<dt>{html.escape(str(key))}</dt><dd>{_value_html(child, source, str(key))}</dd>")
        return '<dl class="nested">' + "".join(rows) + "</dl>"
    if isinstance(value, (list, tuple)):
        return '<ul class="nested-list">' + "".join(f"<li>{_value_html(child, source, label)}</li>" for child in value) + "</ul>"
    return html.escape(str(value))


def _record_value_html(value: Any, source: Provenance, path: str) -> str:
    if isinstance(value, str) and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value.strip()):
        if any(token in path.lower() for token in ("length", "count", "number", "size", "seed")):
            parsed: int | float = int(value) if "." not in value else float(value)
            return _number(parsed, source, path)
    return _value_html(value, source, path)


def _parse_fasta(path: Path) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    header: str | None = None
    sequence: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header.split()[0], header, "".join(sequence)))
            header = line[1:].strip()
            sequence = []
        else:
            sequence.append(line)
    if header is not None:
        records.append((header.split()[0], header, "".join(sequence)))
    return records


def _candidate_aliases(candidate_id: str, record: Mapping[str, Any] | None = None) -> set[str]:
    aliases = {candidate_id}
    if record:
        for key in ("id", "candidate_id", "design_id"):
            value = record.get(key)
            if value:
                aliases.add(str(value))
    if "|" in candidate_id:
        aliases.add(candidate_id.split("|", 1)[0])
        aliases.add(candidate_id.split("|", 1)[1])
    return {value for value in aliases if value}


def _match_image(candidate: Candidate, images: list[Path], root: Path) -> None:
    aliases = _candidate_aliases(candidate.candidate_id, candidate.record)
    matches = [path for path in images if any(alias in path.name or alias in str(path.parent) for alias in aliases)]
    if not matches and len(images) == 1:
        matches = images
    if matches:
        candidate.image_path = matches[0]
        candidate.image_uri = _data_uri(matches[0])


def _data_uri(path: Path) -> str | None:
    try:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{data}"


def _scalar_metrics(
    record: Mapping[str, Any],
    source: Provenance,
    *,
    context: str = "",
    response: bool = False,
) -> list[Metric]:
    metrics: list[Metric] = []
    nested = record.get("metrics")
    if isinstance(nested, Mapping):
        for name, value in nested.items():
            metrics.append(Metric(str(name), value, source, context))
    keys = _RESPONSE_METADATA_KEYS if response else _METADATA_KEYS
    for name, value in record.items():
        if name in keys or name == "metrics":
            continue
        if (
            name in {"iptm", "ptm", "mean_plddt", "rank_score_central_estimate", "rank_score_spread"}
            or name.endswith("_ensemble")
            or "_seed_" in name
            or name in {
                "rank_score",
                "ipsae_min",
                "sc_dockq",
                "dockq",
                "fnat",
                "interface_pae",
                "interface_plddt",
                "clash_count",
                "contact_count",
            }
            or name in _OBSERVATION_METRIC_KEYS
            or name == "maximum_clash_count"
            or name.endswith("_surrogate")
        ):
            metrics.append(Metric(str(name), value, source, context))
    return metrics


def _normalise_monomer_confidence(value: Any) -> float | None:
    """Return monomer confidence on the report's 0 to 1 scale."""

    if not _is_number(value):
        return None
    numeric = float(value)
    if 0.0 <= numeric <= 1.0:
        return numeric
    if 1.0 < numeric <= 100.0:
        return numeric / 100.0
    return None


def _monomer_confidence_metric(
    record: Mapping[str, Any],
    source: Provenance,
    keys: Iterable[str] = _MONOMER_CONFIDENCE_KEYS,
) -> Metric | None:
    """Return one normalized confidence metric from a binder-only artifact."""

    payloads: list[Mapping[str, Any]] = [record]
    nested_metrics = record.get("metrics")
    if isinstance(nested_metrics, Mapping):
        payloads.append(nested_metrics)
    for payload in payloads:
        for key in keys:
            confidence = _normalise_monomer_confidence(payload.get(key))
            if confidence is None:
                continue
            metric_source = Provenance(
                artifact=source.artifact,
                tool=source.tool,
                revision=source.revision,
                seed=source.seed,
                detail=f"binder-only fold {key}",
            )
            return Metric("monomer_confidence", confidence, metric_source, "binder-only fold")
    return None


def _monomer_artifact_candidate_id(path: Path, root: Path, record: Mapping[str, Any]) -> str | None:
    candidate_id = _first_value(record, ("candidate_id", "design_id", "id"))
    if candidate_id is not None:
        return str(candidate_id)
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None
    for index, part in enumerate(parts[:-1]):
        if part in _MONOMER_ARTIFACT_DIRECTORIES and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _monomer_metrics(root: Path, run_records: list[Mapping[str, Any]]) -> dict[str, list[Metric]]:
    """Read standalone binder-only confidence artifacts by candidate ID."""

    grouped: dict[str, list[Metric]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if not any(part in _MONOMER_ARTIFACT_DIRECTORIES for part in parts[:-1]):
            continue
        try:
            record = _read_json(path)
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(record, Mapping):
            continue
        candidate_id = _monomer_artifact_candidate_id(path, root, record)
        if not candidate_id:
            continue
        source = _provenance(path, root, record, run_records, "binder-only fold response")
        metric = _monomer_confidence_metric(record, source)
        if metric is not None:
            grouped.setdefault(candidate_id, []).append(metric)
    return grouped


def _nested_rank_metrics(row: Mapping[str, Any], source: Provenance) -> list[Metric]:
    metrics = _scalar_metrics(row, source)
    for container_name in ("custom_metrics", "per_predictor", "normalized_by_predictor"):
        container = row.get(container_name)
        if not isinstance(container, Mapping):
            continue
        if container_name == "custom_metrics":
            for name, value in container.items():
                metrics.append(Metric(str(name), value, source, container_name))
            continue
        for predictor, values in container.items():
            if not isinstance(values, Mapping):
                continue
            for name, value in values.items():
                if name == "selected_seed" or isinstance(value, Mapping):
                    continue
                metrics.append(Metric(str(name), value, source, f"{container_name}.{predictor}"))
    return metrics


def _sequence_from_result(row: Mapping[str, Any]) -> str | None:
    entities = row.get("entities")
    if isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, Mapping) and entity.get("type") in {"protein", "designed_protein"}:
                value = entity.get("value")
                if isinstance(value, str) and re.fullmatch(r"[A-Za-z]+", value):
                    return value
    request = row.get("request")
    if isinstance(request, Mapping) and isinstance(request.get("binder_seq"), str):
        return request["binder_seq"]
    return None


def _find_config(root: Path) -> tuple[dict[str, Any] | None, Path | None]:
    for name in _CONFIG_NAMES:
        path = root / name
        if path.is_file():
            try:
                value = _read_json(path)
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(value, dict):
                return value, path
    return None, None


def _is_unconstrained_run(config: Mapping[str, Any] | None) -> bool:
    """Return whether any configured target ran in unconstrained discovery mode."""

    if not isinstance(config, Mapping):
        return False
    return any(
        isinstance(target, Mapping)
        and isinstance(target.get("site"), Mapping)
        and target["site"].get("epitope_constraint") == "unconstrained"
        for target in config.get("targets", [])
    )


def _render_constraint(config: Mapping[str, Any] | None) -> str:
    """Render the stored target-constraint label."""

    if _is_unconstrained_run(config):
        return (
            "<p><strong>Constraint:</strong> Unconstrained discovery run.</p>"
            "<p>Screening contacts are grouped before you choose a hotspot list for a later run.</p>"
        )
    return "<p><strong>Constraint:</strong> Constrained run.</p>"


def _render_contact_clusters(root: Path, unconstrained: bool) -> str:
    """Render the contact clusters produced by unconstrained screening."""

    if not unconstrained:
        return ""
    path = root / "contact-clusters.json"
    if not path.is_file():
        return _missing("contact-clusters.json")
    try:
        summary = _read_json(path)
    except (OSError, ValueError, TypeError):
        return _missing("readable contact-clusters.json")
    clusters = summary.get("clusters") if isinstance(summary, Mapping) else None
    if not isinstance(clusters, list):
        return _missing("contact clusters")
    rows: list[str] = []
    contact_scores: list[float | None] = []
    for cluster in clusters:
        if not isinstance(cluster, Mapping):
            continue
        residues = cluster.get("surface_patch_residues", [])
        residue_text = ", ".join(str(value) for value in residues) if isinstance(residues, list) else ""
        scores = cluster.get("score_records", [])
        if isinstance(scores, list):
            for score in scores:
                if isinstance(score, Mapping):
                    value, _ = _stored_number(score, ("ipsae_min", "mean_ipsae_min", "score"))
                    contact_scores.append(value)
        score_text = html.escape(json.dumps(scores, sort_keys=True)) if isinstance(scores, list) else ""
        rows.append(
            "<tr>"
            f'<th scope="row">{html.escape(str(cluster.get("cluster_id", "unknown")))}</th>'
            f"<td>{html.escape(residue_text)}</td>"
            f"<td>{html.escape(str(cluster.get('design_count', 'unknown')))}</td>"
            f"<td>{html.escape(str(cluster.get('screening_model_count', 'unknown')))}</td>"
            f"<td><code>{score_text}</code></td>"
            "</tr>"
        )
    if not rows:
        return _missing("contact clusters")
    table = (
        "<table><thead><tr><th>Cluster</th><th>Surface patch</th><th>Designs</th>"
        "<th>Screening models</th><th>Measured scores</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    source = Provenance(_relative_artifact(path, root), detail="contact-cluster summary")
    return (
        '<div class="table-scroll">'
        + table
        + "</div>"
        + _floor_accounting(contact_scores, label="contact-cluster score rows")
        + f'<div class="source-line">{_provenance_html(source)}</div>'
    )


def _find_run_records(root: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for name in ("start-response.json", "status.json", "run.json", "run_manifest.json"):
        path = root / name
        if path.is_file():
            try:
                value = _read_json(path)
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(value, Mapping):
                records.append(value)
    return records


def _find_results(root: Path) -> tuple[list[tuple[dict[str, Any], Path]], Path | None]:
    for name in _RESULT_NAMES:
        path = root / name
        if not path.is_file():
            continue
        try:
            values = _read_json_stream(path)
        except (OSError, ValueError, TypeError):
            continue
        rows: list[tuple[dict[str, Any], Path]] = []
        for value in values:
            candidates = value if isinstance(value, list) else [value]
            for item in candidates:
                if isinstance(item, dict) and (item.get("id") or item.get("candidate_id")):
                    rows.append((item, path))
        if rows:
            return rows, path
    return [], None


def _find_fasta_records(root: Path) -> list[tuple[str, str, str, Path]]:
    paths: list[Path] = []
    preferred = [root / "designs.fasta"]
    paths.extend(path for path in preferred if path.is_file())
    if not paths:
        paths.extend(sorted(root.rglob("*.binder.fasta")))
    if not paths:
        paths.extend(sorted(root.rglob("*.fasta")))
    records: list[tuple[str, str, str, Path]] = []
    for path in paths:
        try:
            records.extend((candidate_id, header, sequence, path) for candidate_id, header, sequence in _parse_fasta(path))
        except OSError:
            continue
    return records


def _find_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES)


def _find_structure_paths(root: Path, candidate_id: str) -> list[str]:
    aliases = _candidate_aliases(candidate_id)
    paths = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".pdb", ".cif", ".mmcif"}:
            if any(alias in path.name or alias in str(path.parent) for alias in aliases):
                paths.append(str(path))
    return sorted(paths)


def _record_structure_paths(root: Path, candidate_id: str, record: Mapping[str, Any]) -> list[str]:
    paths = set(_find_structure_paths(root, candidate_id))
    for key in ("design_pose_path", "predicted_complex_path", "structure_path", "structure_file"):
        path = _resolve_record_path(root, record.get(key))
        if path is not None and path.is_file():
            paths.add(str(path))
    return sorted(paths)


def _screen_candidates(
    root: Path,
    run_records: list[Mapping[str, Any]],
    fasta_records: list[tuple[str, str, str, Path]],
    images: list[Path],
) -> list[Candidate]:
    by_short: dict[str, tuple[str, str, Path]] = {}
    for candidate_id, header, sequence, path in fasta_records:
        by_short[candidate_id] = (header, sequence, path)
        if "|" in header:
            by_short[header.split("|", 1)[1]] = (header, sequence, path)
    candidates: list[Candidate] = []
    for response_path in sorted(root.rglob("screen/*/response.json")):
        try:
            row = _read_json(response_path)
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(row, dict):
            continue
        design_id = response_path.parent.name
        sequence_record = by_short.get(design_id)
        if sequence_record is None:
            sequence = row.get("request", {}).get("binder_seq") if isinstance(row.get("request"), Mapping) else None
            seq_source = _provenance(response_path, root, row, run_records, "binder sequence in response") if sequence else None
        else:
            _, sequence, fasta_path = sequence_record
            seq_source = _provenance(fasta_path, root, row, run_records, "binder sequence")
        source = _provenance(response_path, root, row, run_records)
        candidate = Candidate(
            candidate_id=design_id,
            sequence=sequence,
            sequence_source=seq_source,
            metrics=_scalar_metrics(row, source, response=True),
            structure_paths=[str(response_path.parent / "predicted.cif")] if (response_path.parent / "predicted.cif").is_file() else [],
            source_paths=[_relative_artifact(response_path, root)],
            record=row,
        )
        _match_image(candidate, images, root)
        candidates.append(candidate)
    return candidates


def _service_metrics(root: Path, run_records: list[Mapping[str, Any]]) -> dict[str, list[Metric]]:
    """Read design-service metrics stored beside each returned structure."""

    grouped: dict[str, list[Metric]] = {}
    for path in sorted(root.glob("extracted/*/metrics.json")):
        try:
            values = _read_json(path)
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(values, Mapping):
            continue
        source = _provenance(path, root, values, run_records)
        grouped[path.parent.name] = [
            Metric(str(name), value, source, "design service") for name, value in values.items()
        ]
    return grouped


def _merge_metrics(candidates: Iterable[Candidate], grouped: Mapping[str, list[Metric]]) -> None:
    for candidate in candidates:
        candidate.metrics.extend(grouped.get(candidate.candidate_id, ()))


def _merge_monomer_metrics(candidates: Iterable[Candidate], grouped: Mapping[str, list[Metric]]) -> None:
    for candidate in candidates:
        if any(metric.name == "monomer_confidence" for metric in candidate.metrics):
            continue
        candidate.metrics.extend(grouped.get(candidate.candidate_id, ()))


def _rescore_plan(root: Path) -> tuple[dict[str, list[int]], Provenance | None]:
    path = root / "rescore.log"
    if not path.is_file():
        return {}, None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}, None
    match = re.search(r"RESCORE\s+designs=([^\n]+?)\s+seeds=(\d+)\.\.(\d+)", text)
    if not match:
        return {}, Provenance(_relative_artifact(path, root), detail="rescore plan log")
    start, end = int(match.group(2)), int(match.group(3))
    if end < start:
        return {}, Provenance(_relative_artifact(path, root), detail="rescore plan log has an invalid seed range")
    return (
        {candidate_id: list(range(start, end + 1)) for candidate_id in match.group(1).split()},
        Provenance(_relative_artifact(path, root), detail="rescore plan log"),
    )


def _rescore_records(root: Path, run_records: list[Mapping[str, Any]]) -> list[RescoreRecord]:
    """Return completed, failed, and log-declared rescore records."""

    plan, plan_source = _rescore_plan(root)
    records: dict[tuple[str, int | str], RescoreRecord] = {}
    for path in sorted(root.glob("rescore/*")):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"(.+)-s(\d+)", path.name)
        if not match:
            continue
        candidate_id, seed_text = match.groups()
        seed = int(seed_text)
        response_path = path / "response.json"
        error_path = path / "error.json"
        response: Mapping[str, Any] = {}
        if response_path.is_file():
            try:
                value = _read_json(response_path)
                if isinstance(value, Mapping):
                    response = value
            except (OSError, ValueError, TypeError):
                pass
        source_path = response_path if response_path.is_file() else error_path
        if source_path.is_file():
            source = _provenance(source_path, root, response, run_records, "rescore record")
        else:
            source = Provenance(_relative_artifact(path, root), detail="rescore directory")
        records[(candidate_id, seed)] = RescoreRecord(
            candidate_id,
            seed,
            response_path if response_path.is_file() else None,
            error_path if error_path.is_file() else None,
            source,
        )
    for candidate_id, seeds in plan.items():
        for seed in seeds:
            key = (candidate_id, seed)
            if key not in records:
                records[key] = RescoreRecord(candidate_id, seed, None, None, plan_source or Provenance("rescore log"))
    return sorted(records.values(), key=lambda item: (item.candidate_id, str(item.seed)))


def _rescore_metrics(records: Iterable[RescoreRecord]) -> dict[str, list[Metric]]:
    grouped: dict[str, list[Metric]] = {}
    for record in records:
        if record.response_path is None:
            continue
        try:
            response = _read_json(record.response_path)
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(response, Mapping):
            continue
        tool = record.source.tool or "rescore predictor"
        context = f"rescore {tool}, seed {record.seed}"
        grouped.setdefault(record.candidate_id, []).extend(
            _scalar_metrics(response, record.source, context=context, response=True)
        )
    return grouped


def _scorer_rows(root: Path) -> list[ScorerRow]:
    path = root / "scored" / "RESULTS.tsv"
    if not path.is_file():
        return []
    script_path = path.with_name("score_ten.py")
    try:
        raw_rows = _read_tsv(path)
        script_present = script_path.is_file() and bool(script_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError, TypeError):
        return []
    rows: list[ScorerRow] = []
    artifacts = [path, script_path] if script_present else [path]
    for raw in raw_rows:
        candidate_id = raw.get("design_id")
        if not candidate_id:
            continue
        revision = raw.get("site_scorer_revision") or None
        source = Provenance(
            _relative_artifacts(artifacts, root),
            tool="score_ten.py" if script_present else None,
            revision=revision,
            detail="local scorer row",
        )
        values = {name: _typed_tsv_value(raw.get(name)) for name in _SCORER_METRIC_KEYS if name in raw}
        rows.append(
            ScorerRow(
                raw.get("record_type", "scorer row"),
                candidate_id,
                raw.get("source", "missing source"),
                values,
                source,
            )
        )
    return rows


def _merge_scorer_metrics(candidates: Iterable[Candidate], rows: Iterable[ScorerRow]) -> None:
    grouped: dict[str, list[Metric]] = {}
    for row in rows:
        if row.record_type != "design":
            continue
        context = f"scored {row.source_name}"
        grouped.setdefault(row.candidate_id, []).extend(
            Metric(name, value, row.source, context) for name, value in row.values.items()
        )
    _merge_metrics(candidates, grouped)


def _backbone_candidates(root: Path, images: list[Path]) -> list[Candidate]:
    """Read RFdiffusion3 backbone metrics without inventing a sequence."""

    base = root / "backbones"
    if not base.is_dir():
        return []
    candidates: list[Candidate] = []
    for path in sorted(base.rglob("*.json")):
        if path.name in {"response.json", "receipt.json"}:
            continue
        try:
            record = _read_json(path)
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(record, Mapping) or not isinstance(record.get("metrics"), Mapping):
            continue
        response_path = path.with_name("response.json")
        receipt_path = path.with_name("receipt.json")
        run_records: list[Mapping[str, Any]] = []
        for run_path in (response_path, receipt_path):
            try:
                value = _read_json(run_path)
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(value, Mapping):
                run_records.append(value)
        base_source = _provenance(path, root, record, run_records, "RFdiffusion3 backbone metrics")
        source_paths = [path, *(run_path for run_path in (response_path, receipt_path) if run_path.is_file())]
        source = Provenance(
            _relative_artifacts(source_paths, root),
            base_source.tool,
            base_source.revision,
            base_source.seed,
            base_source.detail,
        )
        structure_path = path.with_suffix(".cif.gz")
        candidate = Candidate(
            candidate_id=path.relative_to(root).with_suffix("").as_posix(),
            metrics=[Metric(str(name), value, source, "RFdiffusion3 backbone") for name, value in record["metrics"].items()],
            structure_paths=[str(structure_path)] if structure_path.is_file() else [],
            source_paths=[_relative_artifact(source_path, root) for source_path in source_paths],
            record=dict(record),
        )
        _match_image(candidate, images, root)
        candidates.append(candidate)
    return candidates


def _ranked_candidates(
    root: Path,
    config: Mapping[str, Any] | None,
    run_records: list[Mapping[str, Any]],
    fasta_records: list[tuple[str, str, str, Path]],
    images: list[Path],
) -> tuple[list[Candidate], Path | None, list[dict[str, Any]], Mapping[str, Any] | None]:
    ranked_paths = sorted(root.rglob("ranked-candidates.json"))
    if not ranked_paths:
        return [], None, [], None
    path = ranked_paths[0]
    try:
        value = _read_json(path)
    except (OSError, ValueError, TypeError):
        return [], path, [], None
    rows = value.get("ranked_candidates", []) if isinstance(value, Mapping) else []
    if not isinstance(rows, list):
        return [], path, [], value if isinstance(value, Mapping) else None
    fasta_by_id = {candidate_id: (sequence, fasta_path) for candidate_id, _, sequence, fasta_path in fasta_records}
    observations = _observation_rows(root)
    candidates: list[Candidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id", "missing-candidate-id"))
        sequence: str | None = None
        sequence_source: Provenance | None = None
        sequence_path_value = row.get("sequence_path")
        sequence_path = _resolve_record_path(root, sequence_path_value)
        if sequence_path and sequence_path.is_file():
            parsed = _parse_fasta(sequence_path)
            if parsed:
                sequence = parsed[0][2]
                sequence_source = _provenance(sequence_path, root, row, run_records, "binder sequence")
        elif candidate_id in fasta_by_id:
            sequence, fasta_path = fasta_by_id[candidate_id]
            sequence_source = _provenance(fasta_path, root, row, run_records, "binder sequence")
        source = _provenance(path, root, row, run_records)
        metrics = _nested_rank_metrics(row, source)
        monomer_metric = _monomer_confidence_metric(row, source, keys=("monomer_confidence",))
        if monomer_metric is not None:
            metrics.append(monomer_metric)
        for observation in observations.get(candidate_id, []):
            observation_path, observation_row = observation
            observation_source = _provenance(observation_path, root, observation_row, run_records)
            predictor = str(observation_row.get("predictor", ""))
            seed = observation_row.get("seed")
            context = f"{predictor}, seed {seed}" if predictor else f"seed {seed}"
            metrics.extend(_observation_metrics(observation_row, observation_source, context))
        candidate = Candidate(
            candidate_id=candidate_id,
            sequence=sequence,
            sequence_source=sequence_source,
            metrics=metrics,
            rank=row.get("rank", _MISSING),
            rank_source=source,
            rank_row=row,
            structure_paths=_record_structure_paths(root, candidate_id, row),
            source_paths=[_relative_artifact(path, root)],
            record=row,
        )
        _match_image(candidate, images, root)
        candidates.append(candidate)
    candidates = _sort_candidates(candidates)
    return candidates, path, rows, value if isinstance(value, Mapping) else None


def _resolve_record_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def _observation_rows(root: Path) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(root.rglob("*.jsonl")):
        if not any(token in path.name for token in ("observation", "score-table", "score")):
            continue
        try:
            values = _read_json_stream(path)
        except (OSError, ValueError, TypeError):
            continue
        for value in values:
            rows = value if isinstance(value, list) else [value]
            for row in rows:
                if isinstance(row, dict) and row.get("candidate_id"):
                    grouped.setdefault(str(row["candidate_id"]), []).append((path, row))
    return grouped


def _observation_metrics(row: Mapping[str, Any], source: Provenance, context: str) -> list[Metric]:
    metrics: list[Metric] = []
    for name, value in row.items():
        if (name in _METADATA_KEYS and name not in _OBSERVATION_METRIC_KEYS) or name in {
            "predicted_complex_path",
            "predicted_complex_sha256",
            "pae_path",
            "pae_sha256",
            "metric_source_path",
            "metric_source_sha256",
            "raw_prediction_record_sha256",
        }:
            continue
        if value is not None and (_is_number(value) or name in {"mapping_status", "failure_code", "failure_reason"}):
            metrics.append(Metric(str(name), value, source, context))
        elif name in _OBSERVATION_METRIC_KEYS:
            metrics.append(Metric(str(name), value, source, context))
    return metrics


def _sort_candidates(candidates: list[Candidate]) -> list[Candidate]:
    if candidates and all(candidate.rank is not _MISSING and _is_number(candidate.rank) for candidate in candidates):
        return sorted(candidates, key=lambda candidate: float(candidate.rank))
    return candidates


def _result_candidates(
    root: Path,
    result_rows: list[tuple[dict[str, Any], Path]],
    run_records: list[Mapping[str, Any]],
    fasta_records: list[tuple[str, str, str, Path]],
    images: list[Path],
) -> list[Candidate]:
    fasta_by_id = {candidate_id: (sequence, path) for candidate_id, _, sequence, path in fasta_records}
    candidates: list[Candidate] = []
    for row, path in result_rows:
        candidate_id = str(row.get("candidate_id") or row.get("id"))
        sequence = _sequence_from_result(row)
        sequence_source = _provenance(path, root, row, run_records, "binder sequence in result") if sequence else None
        if sequence is None and candidate_id in fasta_by_id:
            sequence, fasta_path = fasta_by_id[candidate_id]
            sequence_source = _provenance(fasta_path, root, row, run_records, "binder sequence")
        source = _provenance(path, root, row, run_records)
        candidate = Candidate(
            candidate_id=candidate_id,
            sequence=sequence,
            sequence_source=sequence_source,
            metrics=_scalar_metrics(row, source),
            structure_paths=_record_structure_paths(root, candidate_id, row),
            source_paths=[_relative_artifact(path, root)],
            record=row,
        )
        _match_image(candidate, images, root)
        candidates.append(candidate)
    return candidates


def _fasta_only_candidates(
    root: Path,
    fasta_records: list[tuple[str, str, str, Path]],
    images: list[Path],
    run_records: list[Mapping[str, Any]],
) -> list[Candidate]:
    candidates = []
    for candidate_id, _, sequence, path in fasta_records:
        candidate = Candidate(
            candidate_id=candidate_id,
            sequence=sequence,
            sequence_source=_provenance(path, root, {}, run_records, "binder sequence"),
            source_paths=[_relative_artifact(path, root)],
            structure_paths=_find_structure_paths(root, candidate_id),
        )
        _match_image(candidate, images, root)
        candidates.append(candidate)
    return candidates


def _metric_source_for_derived(root: Path, artifact: str, detail: str) -> Provenance:
    return Provenance(artifact=artifact, tool="report generator", detail=detail)


def _request_value(request: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = request
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _fasta_count_source(root: Path, fasta_records: list[tuple[str, str, str, Path]]) -> Provenance | None:
    if not fasta_records:
        return None
    paths = list(dict.fromkeys(path for _, _, _, path in fasta_records))
    return Provenance(
        artifact=", ".join(_relative_artifact(path, root) for path in paths),
        detail="FASTA records present in run directory",
    )


def _stage_counts(
    root: Path,
    request: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
    candidates: list[Candidate],
    fasta_records: list[tuple[str, str, str, Path]],
    result_path: Path | None,
    ranked_path: Path | None,
    ranked_rows: list[dict[str, Any]],
) -> list[StageCount]:
    counts: list[StageCount] = []
    request_path = root / "request.json"
    request_source = Provenance(_relative_artifact(request_path, root), detail="requested candidate count") if request and request_path.is_file() else None
    result_source = Provenance(_relative_artifact(result_path, root), detail="result records") if result_path else None
    fasta_source = _fasta_count_source(root, fasta_records)
    if request and _request_value(request, ("num_proteins",)) is not _MISSING:
        counts.append(StageCount("requested candidate count", _request_value(request, ("num_proteins",)), _MISSING, request_source, None))
    if result_path:
        entered = _request_value(request or {}, ("num_proteins",))
        counts.append(StageCount("result records", entered, len(candidates), request_source, result_source))
    if fasta_records:
        entering = len(candidates) if result_path and candidates else _MISSING
        entering_source = result_source if result_path and candidates else None
        counts.append(StageCount("FASTA candidate records", entering, len(fasta_records), entering_source, fasta_source))
    screen_response_paths = sorted(root.rglob("screen/*/response.json"))
    if screen_response_paths:
        screen_source = Provenance(
            ", ".join(_relative_artifact(path, root) for path in screen_response_paths),
            detail="completed screen response files",
        )
        counts.append(StageCount("screen response records", len(fasta_records) or _MISSING, len(screen_response_paths), fasta_source, screen_source))
    structure_paths = sorted(root.rglob("screen/*/predicted.cif"))
    if structure_paths:
        structure_source = Provenance(
            ", ".join(_relative_artifact(path, root) for path in structure_paths),
            detail="folded structure files",
        )
        # Two independent globs over the same screen/<id>/ tree. Comparing only their
        # lengths reads equal numbers as proof that nothing was lost, and a run holding a
        # response for one candidate and a structure for a different one has equal numbers.
        # The directory name settles it and both globs already carry it, so name the
        # difference rather than discarding it.
        responded = {path.parent.name for path in screen_response_paths}
        folded = {path.parent.name for path in structure_paths}
        note = ""
        if screen_response_paths and responded != folded:
            unfolded = sorted(responded - folded)
            unrequested = sorted(folded - responded)
            note = (
                "the screened candidates and the folded candidates are not the same set: "
                f"responded but not folded {unfolded}, folded but not responded {unrequested}"
            )
        counts.append(StageCount("screen structures", len(screen_response_paths) or _MISSING, len(structure_paths), screen_source if screen_response_paths else None, structure_source, note))
    if ranked_path:
        rank_source = Provenance(_relative_artifact(ranked_path, root), detail="ranked_candidates")
        entered = len(candidates) if candidates else _MISSING
        counts.append(StageCount("ranked candidates", entered, len(ranked_rows), result_source or fasta_source, rank_source))
    if not counts:
        missing_source = Provenance("run directory", detail="stage count discovery")
        counts.append(StageCount("stage counts", _MISSING, _MISSING, missing_source, missing_source, "No stage count artifact was found."))
    return counts


def _additional_stage_counts(
    root: Path,
    rescore_records: list[RescoreRecord],
    scorer_rows: list[ScorerRow],
    backbone_candidates: list[Candidate],
) -> list[StageCount]:
    counts: list[StageCount] = []
    if rescore_records:
        plan_path = root / "rescore.log"
        plan_source = Provenance(_relative_artifact(plan_path, root), detail="rescore plan and requested seed range")
        response_paths = [record.response_path for record in rescore_records if record.response_path is not None]
        response_source = Provenance(
            _relative_artifacts(response_paths, root) if response_paths else "rescore directory",
            detail="completed rescore response files",
        )
        counts.append(
            StageCount(
                "rescore response records",
                len(rescore_records),
                len(response_paths),
                plan_source,
                response_source,
                "The report retains planned records without a response as visible gaps.",
            )
        )
    if scorer_rows:
        scorer_path = root / "scored" / "RESULTS.tsv"
        source = Provenance(_relative_artifact(scorer_path, root), detail="local scorer rows")
        counts.append(StageCount("local scorer rows", len(scorer_rows), len(scorer_rows), source, source))
    if backbone_candidates:
        source = Provenance(
            "; ".join(candidate.source_paths[0] for candidate in backbone_candidates if candidate.source_paths),
            detail="RFdiffusion3 backbone metric records",
        )
        counts.append(
            StageCount("RFdiffusion3 backbone records", len(backbone_candidates), len(backbone_candidates), source, source)
        )
    return counts


def _spend_rows(root: Path) -> tuple[list[dict[str, Any]], Path | None]:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and ("spend" in path.name.lower() or "cost" in path.name.lower()))
    for path in paths:
        try:
            values = _read_json_stream(path)
        except (OSError, ValueError, TypeError):
            continue
        rows = [value for value in values if isinstance(value, dict)]
        if rows:
            return rows, path
    return [], None


def _ranking_metric_names(
    config: Mapping[str, Any] | None,
    ranked_rows: list[dict[str, Any]],
    candidates: Iterable[Candidate],
) -> list[str]:
    actual = list(dict.fromkeys(metric.name for candidate in candidates for metric in candidate.metrics if not metric.context))
    names: list[str] = []
    scoring = config.get("scoring") if isinstance(config, Mapping) else None
    if isinstance(scoring, Mapping):
        for key in ("primary_metric", "pose_metric"):
            value = scoring.get(key)
            if isinstance(value, str):
                if value in actual and value not in names:
                    names.append(value)
                elif f"{value}_ensemble" in actual and f"{value}_ensemble" not in names:
                    names.append(f"{value}_ensemble")
    for name in actual:
        if (
            name in {"rank_score", "rank_score_central_estimate", "rank_score_spread"}
            or name.endswith("_ensemble")
            or "_seed_" in name
        ):
            if name not in names:
                names.append(name)
    if not names and not config and not ranked_rows:
        names = actual
    return names


def _candidate_metric(candidate: Candidate, name: str) -> Metric | None:
    exact = [metric for metric in candidate.metrics if metric.name == name]
    if exact:
        return exact[0]
    return next((metric for metric in candidate.metrics if metric.name == f"{name}_ensemble"), None)


def _validate_monomer_confidence_threshold(value: Any) -> float:
    if not _is_number(value):
        raise ValueError("monomer confidence threshold must be a finite number from 0 to 1")
    threshold = float(value)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("monomer confidence threshold must be from 0 to 1")
    return threshold


def _monomer_confidence(candidate: Candidate) -> Metric | None:
    return next((metric for metric in candidate.metrics if metric.name == "monomer_confidence"), None)


def _render_monomer_status(candidate: Candidate, threshold: float) -> str:
    metric = _monomer_confidence(candidate)
    if metric is None:
        return f'<td>{_missing("monomer confidence for this design; complex score is uninterpretable")}</td>'
    confidence = float(metric.value)
    if confidence < threshold:
        return (
            '<td><span class="uninterpretable">'
            f"Uninterpretable: monomer confidence {format(confidence, '.10g')} is below {format(threshold, '.10g')}."
            "</span></td>"
        )
    return '<td><span class="interpretable">Interpretable against the monomer-confidence threshold.</span></td>'


def _render_request(request: Mapping[str, Any] | None, root: Path) -> str:
    if not request:
        return _missing("request artifact")
    path = root / "request.json"
    source = Provenance(_relative_artifact(path, root), detail="requested run fields")
    parts: list[str] = []
    for key, value in request.items():
        parts.append(f'<div class="request-field"><dt>{html.escape(str(key))}</dt><dd>{_record_value_html(value, source, str(key))}</dd></div>')
    return '<dl class="request-list">' + "".join(parts) + "</dl>"


def _render_sequence(candidate: Candidate) -> str:
    if candidate.sequence is None:
        return _missing("binder sequence")
    sequence_source = candidate.sequence_source or Provenance("missing source")
    return (
        f'<textarea class="sequence" readonly aria-label="Sequence for {html.escape(candidate.candidate_id)}">'
        f"{html.escape(candidate.sequence)}</textarea>"
        f'<div class="source-line">{_provenance_html(sequence_source)}</div>'
    )


def _metric_keys(candidates: Iterable[Candidate]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        for metric in candidate.metrics:
            key = (metric.name, metric.context)
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def _render_metrics(candidate: Candidate, metric_keys: list[tuple[str, str]]) -> str:
    if not metric_keys:
        return _missing("metrics for this design")
    rows: list[str] = []
    for name, context in metric_keys:
        metric = next((item for item in candidate.metrics if item.name == name and item.context == context), None)
        label = html.escape(name)
        if context:
            label += f' <span class="context">{html.escape(context)}</span>'
        missing_label = f"metric {name} ({context}) for this design" if context else f"metric {name} for this design"
        value = _missing(missing_label) if metric is None else _value_html(metric.value, metric.source, name)
        rows.append(f"<tr><th scope=\"row\">{label}</th><td>{value}</td></tr>")
    return '<table class="metrics"><tbody>' + "".join(rows) + "</tbody></table>"


def _render_rescore(candidates: Iterable[Candidate], records: Iterable[RescoreRecord]) -> str:
    grouped: dict[str, list[RescoreRecord]] = {}
    for record in records:
        grouped.setdefault(record.candidate_id, []).append(record)
    rows: list[str] = []
    for candidate in candidates:
        candidate_records = grouped.get(candidate.candidate_id, [])
        if not candidate_records:
            source = Provenance("rescore directory", detail="no rescore directory for this design")
            rows.append(
                "<tr>"
                f'<th scope="row">{html.escape(candidate.candidate_id)}</th>'
                f"<td>{_missing(f'rescore record for {candidate.candidate_id}')}<div class=\"source-line\">{_provenance_html(source)}</div></td>"
                "<td>" + _missing("rescore seed") + "</td>"
                "</tr>"
            )
            continue
        for record in candidate_records:
            if record.response_path is not None:
                label = "response.json recorded"
                if record.error_path is not None:
                    label += "; error.json also recorded"
                status = html.escape(label)
            elif record.error_path is not None:
                status = _missing(f"rescore response for {record.candidate_id}, seed {record.seed}; error.json recorded")
            else:
                status = _missing(f"rescore response for {record.candidate_id}, seed {record.seed}")
            rows.append(
                "<tr>"
                f'<th scope="row">{html.escape(record.candidate_id)}</th>'
                f"<td>{_number(record.seed, record.source, 'rescore seed')}</td>"
                f"<td>{status}<div class=\"source-line\">{_provenance_html(record.source)}</div></td>"
                "</tr>"
            )
    if not rows:
        return _missing("rescore records")
    table = (
        '<table class="rescore"><thead><tr><th>Design</th><th>Seed</th><th>Record</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    return '<div class="table-scroll">' + table + "</div>"


def _scorer_value_html(row: ScorerRow, name: str) -> str:
    value = row.values.get(name, _MISSING)
    if value is _MISSING:
        return _missing(f"metric {name} in {row.source_name} scorer row")
    return _value_html(value, row.source, name)


def _render_scorer_rows(rows: list[ScorerRow]) -> str:
    if not rows:
        return _missing("scorer RESULTS.tsv")
    headers = ("record_type", "design_id", "source", *_SCORER_SUMMARY_METRICS)
    head = "".join(f'<th scope="col">{html.escape(name)}</th>' for name in headers)
    body: list[str] = []
    for row in rows:
        values = [
            f"<td>{html.escape(row.record_type)}</td>",
            f'<th scope="row">{html.escape(row.candidate_id)}</th>',
            f"<td>{html.escape(row.source_name)}<div class=\"source-line\">{_provenance_html(row.source)}</div></td>",
        ]
        values.extend(f"<td>{_scorer_value_html(row, name)}</td>" for name in _SCORER_SUMMARY_METRICS)
        body.append("<tr>" + "".join(values) + "</tr>")
    table = '<table class="scorer"><thead><tr>' + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    return '<div class="table-scroll">' + table + "</div>"


def _render_composition(candidate: Candidate, root: Path) -> str:
    if candidate.sequence is None:
        return _missing("sequence composition")
    source = candidate.sequence_source or Provenance("missing source")
    derived = _metric_source_for_derived(root, source.artifact, "length and composition derived from the reported sequence")
    rows = []
    counts = Counter(candidate.sequence.upper())
    length = len(candidate.sequence)
    rows.append(f'<tr><th scope="row">length</th><td>{_number(length, derived, "sequence length")}</td></tr>')
    for residue in sorted(counts):
        rows.append(
            f'<tr><th scope="row">{html.escape(residue)} count</th><td>{_number(counts[residue], derived, f"{residue} count")}</td></tr>'
        )
        rows.append(
            f'<tr><th scope="row">{html.escape(residue)} fraction</th><td>{_number(counts[residue] / length if length else _MISSING, derived, f"{residue} fraction")}</td></tr>'
        )
    return '<table class="metrics"><tbody>' + "".join(rows) + "</tbody></table>"


def _render_image(candidate: Candidate, root: Path) -> str:
    if not candidate.image_uri:
        return _missing("rendered structure image")
    source = Provenance(
        artifact=_relative_artifact(candidate.image_path, root) if candidate.image_path else "missing image path",
        detail="embedded structure image",
    )
    return f'<figure><img src="{html.escape(candidate.image_uri)}" alt="Rendered structure for {html.escape(candidate.candidate_id)}"><figcaption>{_provenance_html(source)}</figcaption></figure>'


def _render_figures(root: Path) -> str:
    figure_root = root / "figures"
    if not figure_root.is_dir():
        return _missing("rendered figures")
    sidecars = sorted(figure_root.glob("*.render.json"))
    figures: list[str] = []
    rendered_names: set[str] = set()
    for sidecar in sidecars:
        image_path = sidecar.with_name(sidecar.name.removesuffix(".render.json") + ".png")
        rendered_names.add(image_path.name)
        try:
            record = _read_json(sidecar)
        except (OSError, ValueError, TypeError):
            record = {}
        source = _provenance(sidecar, root, record if isinstance(record, Mapping) else {}, detail="render sidecar")
        title = html.escape(image_path.stem)
        if image_path.is_file() and (uri := _data_uri(image_path)):
            image = f'<img src="{html.escape(uri)}" alt="Rendered structure figure {title}">'
        else:
            image = _missing(f"rendered figure {image_path.name}")
        details = ""
        if isinstance(record, Mapping):
            detail_rows = []
            for name in ("epitope_contact_atoms", "camera_mode", "camera_note", "confidence_legend"):
                if name in record:
                    detail_rows.append(
                        f'<dt>{html.escape(name)}</dt><dd>{_value_html(record[name], source, name)}</dd>'
                    )
            if detail_rows:
                details = '<dl class="nested">' + "".join(detail_rows) + "</dl>"
        figures.append(f"<figure><h3>{title}</h3>{image}<figcaption>{_provenance_html(source)}{details}</figcaption></figure>")
    for image_path in sorted(path for path in figure_root.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES and path.name not in rendered_names):
        uri = _data_uri(image_path)
        source = Provenance(_relative_artifact(image_path, root), detail="rendered figure without sidecar")
        image = f'<img src="{html.escape(uri)}" alt="Rendered structure figure {html.escape(image_path.stem)}">' if uri else _missing(f"rendered figure {image_path.name}")
        figures.append(f"<figure><h3>{html.escape(image_path.stem)}</h3>{image}<figcaption>{_provenance_html(source)}</figcaption></figure>")
    if not figures:
        return _missing("rendered figures")
    return '<div class="figure-grid">' + "".join(figures) + "</div>"


def _render_stage_counts(counts: list[StageCount]) -> str:
    rows = []
    for item in counts:
        entering_source = item.entering_source
        surviving_source = item.surviving_source
        rows.append(
            "<tr>"
            f"<th scope=\"row\">{html.escape(item.name)}</th>"
            f"<td>{_number(item.entering, entering_source, f'{item.name} entering count')}</td>"
            f"<td>{_number(item.surviving, surviving_source, f'{item.name} surviving count')}</td>"
            f"<td>{html.escape(item.note)}</td>"
            "</tr>"
        )
    table = '<table class="stage-counts"><thead><tr><th>Stage record</th><th>Candidates entering</th><th>Candidates recorded after stage</th><th>Note</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>"
    return '<div class="table-scroll">' + table + "</div>"


def _render_spend(root: Path) -> str:
    rows, path = _spend_rows(root)
    if not rows or path is None:
        sources = [
            ("Hosted design service", root / "start-response.json"),
            ("ESMFold2-Fast folds", root / "screen.log"),
            ("RFdiffusion3 backbones", root / "backbones" / "one" / "receipt.json"),
        ]
        items = []
        for label, source_path in sources:
            try:
                if source_path.suffix == ".json":
                    _read_json(source_path)
                else:
                    source_path.read_text(encoding="utf-8", errors="replace")
                detail = "no dollar spend field is recorded in this artifact"
            except (OSError, ValueError, TypeError):
                detail = "spend source artifact is absent or unreadable"
            source = Provenance(
                _relative_artifact(source_path, root),
                detail=detail,
            )
            items.append(
                f"<dt>{html.escape(label)}</dt><dd>{_missing(f'recorded dollar spend for {label.lower()}')}"
                f'<div class="source-line">{_provenance_html(source)}</div></dd>'
            )
        return '<dl class="spend-list">' + "".join(items) + "</dl>"
    contributing = [
        row
        for row in rows
        if row.get("event") in {"charge", "charge-estimate"} and _is_number(row.get("amount"))
    ]
    if not contributing:
        source = Provenance(_relative_artifact(path, root), detail="spend ledger has no charge event")
        return _missing("charge amount in the spend ledger") + f'<div class="source-line">{_provenance_html(source)}</div>'
    estimated = sum(
        float(row["amount"])
        for row in contributing
        if row.get("event") == "charge-estimate"
    )
    settled = 0.0
    for row in contributing:
        if row.get("event") != "charge":
            continue
        details = row.get("details")
        row_details = details if isinstance(details, Mapping) else {}
        if row_details.get("reconciliation") is True:
            estimated -= float(row_details.get("reconciled_estimate_amount", 0.0))
            settled += float(row_details.get("settled_amount", 0.0))
        else:
            settled += float(row["amount"])
    total = estimated + settled
    currency = next((row.get("currency") for row in contributing if row.get("currency")), None)
    record = contributing[-1]
    source = _provenance(path, root, record, detail=record.get("amount_source"))
    unit = f" {currency}" if currency else ""
    if estimated and settled:
        label = (
            f"{format(total, '.10g')}{unit} total "
            f"({format(settled, '.10g')}{unit} settled and "
            f"{format(estimated, '.10g')}{unit} estimated)"
        )
    elif estimated:
        label = (
            f"{format(total, '.10g')}{unit} estimated total; "
            "no settled provider cost is available"
        )
    else:
        label = f"{format(total, '.10g')}{unit} settled total"
    return f'<span class="number">{html.escape(label)} {_provenance_html(source)}</span>'


def _render_ranking_config(
    config: Mapping[str, Any] | None,
    config_path: Path | None,
    root: Path,
) -> str:
    if isinstance(config, Mapping) and isinstance(config.get("scoring"), Mapping):
        scoring = config["scoring"]
        path = config_path or root / "config.json"
        source = Provenance(_relative_artifact(path, root), detail="ranking configuration")
        rows: list[str] = []
        for key in (
            "primary_metric",
            "pose_metric",
            "seed_aggregation",
            "minimum_seed_observations",
            "seed_reduction",
            "normalization",
        ):
            if key in scoring:
                rows.append(f'<tr><th scope="row">{html.escape(key)}</th><td>{_value_html(scoring[key], source, key)}</td></tr>')
        if isinstance(scoring.get("rank_weights"), Mapping):
            rows.append(f'<tr><th scope="row">rank_weights</th><td>{_value_html(scoring["rank_weights"], source, "rank_weights")}</td></tr>')
        return '<p class="source-line">' + _provenance_html(source) + '</p><table class="metrics"><tbody>' + "".join(rows) + "</tbody></table>"
    return _missing("ranking configuration")


def _metric_cell(metric: Metric | None, name: str) -> str:
    if metric is None:
        return f"<td>{_missing(f'metric {name} for this design')}</td>"
    return f'<td>{_value_html(metric.value, metric.source, name)}</td>'


def _stored_number(record: Mapping[str, Any] | None, fields: Iterable[str]) -> tuple[float | None, str | None]:
    """Return an explicitly stored finite number without deriving a substitute."""

    if not isinstance(record, Mapping):
        return None, None
    for field_name in fields:
        value = record.get(field_name)
        if _is_number(value):
            return float(value), field_name
    return None, None


def _stored_text(record: Mapping[str, Any] | None, fields: Iterable[str]) -> tuple[str | None, str | None]:
    if not isinstance(record, Mapping):
        return None, None
    for field_name in fields:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip(), field_name
    return None, None


def _decision_rows(candidates: Iterable[Candidate]) -> list[DecisionRow]:
    """Read decision fields that the run persisted beside each ranked candidate."""

    rows: list[DecisionRow] = []
    for candidate in candidates:
        rank_row = candidate.rank_row
        mean, mean_field = _stored_number(rank_row, _MEAN_IPSAE_FIELDS)
        sd, sd_field = _stored_number(rank_row, _SD_IPSAE_FIELDS)
        n_folds, n_folds_field = _stored_number(rank_row, _FOLD_COUNT_FIELDS)
        interface_error, interface_error_field = _stored_number(rank_row, _INTERFACE_ERROR_FIELDS)
        if interface_error is None:
            for metric_name in _INTERFACE_ERROR_FIELDS:
                metric = _candidate_metric(candidate, metric_name)
                if metric is not None and _is_number(metric.value):
                    interface_error = float(metric.value)
                    interface_error_field = metric.name
                    break
        length, length_field = _stored_number(rank_row, ("sequence_length", "len_aa"))
        if length is None and candidate.sequence is not None:
            length = len(candidate.sequence)
            length_field = "sequence"
        rows.append(
            DecisionRow(
                candidate=candidate,
                mean_ipsae_min=mean,
                mean_field=mean_field,
                sd_ipsae_min=sd,
                sd_field=sd_field,
                n_folds=n_folds,
                n_folds_field=n_folds_field,
                interface_error=interface_error,
                interface_error_field=interface_error_field,
                monomer_confidence=_monomer_confidence(candidate),
                length=int(length) if length is not None and float(length).is_integer() else None,
                length_field=length_field,
            )
        )
    if rows and all(row.mean_ipsae_min is not None for row in rows):
        return sorted(rows, key=lambda row: (-float(row.mean_ipsae_min), row.candidate.candidate_id))
    return rows


def _within_noise(left: DecisionRow, right: DecisionRow) -> bool:
    if (
        left.mean_ipsae_min is None
        or right.mean_ipsae_min is None
        or left.sd_ipsae_min is None
        or right.sd_ipsae_min is None
    ):
        return False
    if left.mean_ipsae_min == right.mean_ipsae_min:
        return True
    return abs(left.mean_ipsae_min - right.mean_ipsae_min) < max(left.sd_ipsae_min, right.sd_ipsae_min)


def _display_ranks(rows: list[DecisionRow]) -> list[int | None]:
    """Assign shared ranks to adjacent candidates whose stored means overlap."""

    if not rows or not all(row.mean_ipsae_min is not None for row in rows):
        return [
            int(row.candidate.rank)
            if _is_number(row.candidate.rank) and float(row.candidate.rank).is_integer()
            else None
            for row in rows
        ]
    ranks: list[int] = []
    index = 0
    while index < len(rows):
        group_end = index + 1
        while group_end < len(rows) and _within_noise(rows[group_end - 1], rows[group_end]):
            group_end += 1
        ranks.extend([index + 1] * (group_end - index))
        index = group_end
    return ranks


def _decision_flags(rows: list[DecisionRow]) -> list[str]:
    flags: list[str] = []
    for index, row in enumerate(rows):
        values: list[str] = []
        if row.mean_ipsae_min == 0.0:
            values.append("AT-FLOOR")
        if (index > 0 and _within_noise(rows[index - 1], row)) or (
            index + 1 < len(rows) and _within_noise(row, rows[index + 1])
        ):
            values.append("=")
        flags.append(" ".join(values))
    return flags


def _floor_accounting(
    scores: Iterable[float | None],
    *,
    label: str,
    tied_rows: int | None = None,
    tie_label: str = "share the displayed score",
) -> str:
    """Render floor and tie counts for a section that displays scores."""

    observed = [float(value) for value in scores if value is not None and math.isfinite(value)]
    floor_rows = sum(value == 0.0 for value in observed)
    if tied_rows is None:
        counts = Counter(observed)
        tied_rows = sum(count for count in counts.values() if count > 1)
    return (
        f'<p class="floor-accounting">Floor accounting for {html.escape(label)}: '
        f"{floor_rows} of {len(observed)} scored rows were exactly 0.0. "
        f"{tied_rows} rows {html.escape(tie_label)}. "
        "Tie order carries no information.</p>"
    )


def _decision_floor_accounting(rows: Iterable[DecisionRow], *, label: str) -> str:
    """Render the primary-metric floor and stored-noise tie counts for decision rows."""

    values = list(rows)
    tied_rows = sum(
        any(
            _within_noise(row, other)
            for other in values
            if other is not row
        )
        for row in values
    )
    return _floor_accounting(
        (row.mean_ipsae_min for row in values),
        label=label,
        tied_rows=tied_rows,
        tie_label="are tied within stored seed noise",
    )


def _decision_number(value: float | None, field_name: str | None, label: str) -> str:
    if value is None:
        return _missing(f"TODO: stored {label}")
    if value == 0.0 and label == "mean ipSAE_min":
        return "0.000"
    return html.escape(format(value, ".10g"))


def _decision_monomer(metric: Metric | None) -> str:
    if metric is None or not _is_number(metric.value):
        return _missing("TODO: stored monomer confidence")
    return html.escape(format(float(metric.value), ".10g"))


def _decision_length(row: DecisionRow) -> str:
    if row.length is None:
        return _missing("TODO: chain length")
    return str(row.length)


def _control_gate(ranking: Mapping[str, Any], path: Path | None, root: Path) -> ControlGate:
    """Require a passed scoring arm and available ranking claim before rendering data."""

    source = Provenance(_relative_artifact(path, root), detail="ranking control") if path else None
    status = ranking.get("scoring_arm_status")
    claim_status = ranking.get("ranking_claim_status")
    controls = ranking.get("controls")
    control = ranking.get("ranking_control")
    reason = control.get("reason") if isinstance(control, Mapping) else None
    recorded_reason = (
        str(reason)
        if isinstance(reason, str)
        and reason
        and (status != "passed" or claim_status != "available" or (isinstance(controls, Mapping) and controls.get("ok") is False))
        else None
    )
    if isinstance(controls, Mapping) and controls.get("ok") is False:
        return ControlGate("failed", recorded_reason or "the stored aggregate control result is failed", source)
    controls_document, _ = _controls_document(root)
    for record in _control_records(controls_document, ranking):
        control_status, _ = _stored_text(record, ("status", "verdict", "result"))
        if control_status is None or control_status.lower() not in {"fail", "failed", "degraded", "unvalidated"}:
            continue
        control_id, _ = _stored_text(record, ("label", "name", "control_id", "candidate_id", "id"))
        prefix = f"control {control_id} " if control_id else "a control "
        control_reason, _ = _stored_text(record, ("reason", "failure_reason", "reason_code"))
        return ControlGate("failed", control_reason or recorded_reason or prefix + f"is recorded as {control_status}", source)
    if status != "passed":
        state = str(status) if isinstance(status, str) and status else "missing"
        return ControlGate("failed", recorded_reason or f"scoring arm status is {state}", source)
    if claim_status != "available":
        state = str(claim_status) if isinstance(claim_status, str) and claim_status else "missing"
        return ControlGate("failed", recorded_reason or f"ranking claim status is {state}", source)
    return ControlGate("passed", None, source)


def _controls_document(root: Path) -> tuple[Mapping[str, Any], Path | None]:
    paths = [
        root / "controls.json",
        root / "artifacts" / "controls" / "controls.json",
        root / "artifacts" / "controls" / "control-summary.json",
    ]
    for path in paths:
        if not path.is_file():
            continue
        try:
            value = _read_json(path)
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(value, Mapping):
            return value, path
    return {}, None


def _control_records(document: Mapping[str, Any], ranking: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Find explicitly recorded per-control observations without manufacturing rows."""

    for container in (document, ranking):
        for key in ("control_results", "control_records", "controls"):
            value = container.get(key) if isinstance(container, Mapping) else None
            if isinstance(value, list):
                return [item for item in value if isinstance(item, Mapping)]
            if isinstance(value, Mapping):
                rows = value.get("records") or value.get("results")
                if isinstance(rows, list):
                    return [item for item in rows if isinstance(item, Mapping)]
    return []


def _render_controls(root: Path, ranking: Mapping[str, Any], gate: ControlGate) -> str:
    document, document_path = _controls_document(root)
    records = _control_records(document, ranking)
    source_text = (
        f'<p class="source-line">Definitions and raw values: {html.escape(_relative_artifact(document_path, root))}.</p>'
        if document_path is not None
        else _missing("TODO: controls.json is absent")
    )
    if not records:
        return source_text
    rows = []
    control_scores: list[float | None] = []
    for record in records:
        label, _ = _stored_text(record, ("label", "name", "control_id", "candidate_id", "id"))
        observed, _ = _stored_number(record, _MEAN_IPSAE_FIELDS + ("ipsae_min", "observed"))
        control_scores.append(observed)
        status, _ = _stored_text(record, ("status", "verdict", "result"))
        rows.append(
            "<tr>"
            f'<th scope="row">{html.escape(label or "TODO: control identifier")}</th>'
            f'<td>{_decision_number(observed, None, "control observation")}</td>'
            f'<td>{html.escape((status or "TODO: control verdict").upper())}</td>'
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table class="controls"><thead><tr><th>Control</th><th>Observed ipSAE_min</th><th>Result</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        + _floor_accounting(control_scores, label="control rows")
        + source_text
    )


def _run_value(
    ranking: Mapping[str, Any],
    config: Mapping[str, Any] | None,
    request: Mapping[str, Any] | None,
    run_records: Iterable[Mapping[str, Any]],
    fields: Iterable[str],
) -> str | None:
    for record in (ranking, config, request, *run_records):
        value, _ = _stored_text(record, fields)
        if value:
            return value
    return None


def _target_summary(ranking: Mapping[str, Any], config: Mapping[str, Any] | None) -> str:
    target: Mapping[str, Any] | None = None
    if isinstance(config, Mapping) and isinstance(config.get("targets"), list):
        target = next((item for item in config["targets"] if isinstance(item, Mapping)), None)
    target_id, _ = _stored_text(target, ("target_id", "id"))
    if target_id is None:
        target_id, _ = _stored_text(ranking, ("primary_target_id", "target_id"))
    if target_id is None:
        return _missing("TODO: target identifier")
    details: list[str] = [target_id]
    structure_id, _ = _stored_text(target, ("structure_id", "pdb_id", "source_structure_id"))
    if structure_id:
        details.append(f"structure {structure_id}")
    chain, _ = _stored_text(target, ("target_chain_id", "chain_id", "chain"))
    if chain:
        details.append(f"chain {chain}")
    site = target.get("site") if isinstance(target, Mapping) else None
    residues = site.get("design_residues") if isinstance(site, Mapping) else None
    if isinstance(residues, list) and residues:
        details.append("residues " + ", ".join(str(value) for value in residues))
    return html.escape("; ".join(details))


def _backend_summary(ranking: Mapping[str, Any], config: Mapping[str, Any] | None) -> str:
    score_instrument, _ = _stored_text(ranking, ("score_instrument",))
    if score_instrument:
        return html.escape(score_instrument)
    scoring = config.get("scoring") if isinstance(config, Mapping) else None
    scorer_name, _ = _stored_text(scoring, ("scorer_name", "name"))
    if scorer_name:
        return html.escape(scorer_name)
    return _missing("TODO: folding backend and scorer identity")


def _render_baseline_fidelity_disclosure(
    ranking: Mapping[str, Any] | None,
) -> str:
    """Render the stored baseline claim with the scope of its derivation."""
    claim = ranking.get("claim") if isinstance(ranking, Mapping) else None
    if not isinstance(claim, Mapping) or not isinstance(claim.get("baseline_fidelity"), bool):
        return _missing("baseline-fidelity claim and its derivation")
    basis = claim.get("baseline_fidelity_basis")
    if not isinstance(basis, str) or not basis.strip():
        basis_html = _missing("baseline-fidelity claim derivation")
    else:
        basis_html = html.escape(basis)
    status = (
        "claims fidelity to the published baseline"
        if claim["baseline_fidelity"]
        else "does not claim fidelity to the published baseline"
    )
    return f"<p>Baseline-fidelity claim: this run {status}. Derived from {basis_html}.</p>"


def _render_predictor_arm_disclosure(config: Mapping[str, Any] | None) -> str:
    """Render configured predictor modes, their lineages, and published-arm deviations."""

    if not isinstance(config, Mapping):
        return _missing("TODO: configured co-folding modes and published-arm deviations")
    cofold = config.get("cofold")
    predictors = cofold.get("predictors") if isinstance(cofold, Mapping) else None
    if not isinstance(predictors, list):
        return _missing("TODO: configured co-folding modes and published-arm deviations")
    modes = [
        str(item["id"])
        for item in predictors
        if isinstance(item, Mapping)
        and item.get("enabled", True) is True
        and isinstance(item.get("id"), str)
    ]
    if not modes:
        return _missing("TODO: configured co-folding modes and published-arm deviations")
    lineages = arms.enabled_predictor_lineages(config)
    rendered = [
        "<p>Configured co-folding modes: "
        + html.escape(", ".join(modes))
        + ".</p>"
    ]
    if set(modes) == {"esmfold2", "esmfold2-fast"} and len(lineages) == 1:
        rendered.append(
            "<p>ESMFold2-Full and ESMFold2-Fast are two modes of one predictor lineage. "
            "Their agreement measures variation within that lineage.</p>"
        )
    else:
        rendered.append(
            f"<p>Configured predictor lineage count: {len(lineages)}.</p>"
        )
    profile = config.get("profile")
    deviations = (
        profile.get("published_predictor_deviations")
        if isinstance(profile, Mapping)
        else None
    )
    declaration = (
        "<p>Published-arm deviations are declared in "
        "profile.published_predictor_deviations. The package does not derive or "
        "cross-check this list against cofold.predictors.</p>"
    )
    if not isinstance(deviations, list):
        return "".join(rendered) + declaration + _missing(
            "profile.published_predictor_deviations declaration"
        )
    if not deviations:
        return "".join(rendered) + declaration + "<p>No deviation entries were declared.</p>"
    items: list[str] = []
    for deviation in deviations:
        if not isinstance(deviation, Mapping):
            continue
        arm = deviation.get("published_arm")
        reason = deviation.get("reason")
        if not isinstance(arm, str) or not arm or not isinstance(reason, str) or not reason:
            continue
        items.append(
            f"<li><strong>{html.escape(arm)}</strong>: {html.escape(reason)}</li>"
        )
    if not items:
        return "".join(rendered) + declaration + _missing(
            "valid entries in profile.published_predictor_deviations"
        )
    rendered.append(declaration + "<ul>" + "".join(items) + "</ul>")
    return "".join(rendered)


def _render_ranking_table(rows: list[DecisionRow], ranked_path: Path | None, root: Path) -> str:
    if not rows:
        return _missing("TODO: ranked candidate rows")
    ranks = _display_ranks(rows)
    flags = _decision_flags(rows)
    body = []
    for row, display_rank, flag in zip(rows, ranks, flags, strict=True):
        rank = str(display_rank) if display_rank is not None else _missing("TODO: rank")
        flag_html = html.escape(flag) if flag else ""
        body.append(
            "<tr>"
            f"<td>{rank}</td>"
            f'<th scope="row">{html.escape(row.candidate.candidate_id)}</th>'
            f"<td>{_decision_number(row.mean_ipsae_min, row.mean_field, 'mean ipSAE_min')}</td>"
            f"<td>{_decision_number(row.sd_ipsae_min, row.sd_field, 'ipSAE_min standard deviation')}</td>"
            f"<td>{_decision_number(row.n_folds, row.n_folds_field, 'fold count')}</td>"
            f"<td>{_decision_number(row.interface_error, row.interface_error_field, 'interface error')}</td>"
            f"<td>{_decision_monomer(row.monomer_confidence)}</td>"
            f"<td>{_decision_length(row)}</td>"
            f'<td class="flag">{flag_html}</td>'
            "</tr>"
        )
    source = _relative_artifact(ranked_path, root) if ranked_path is not None else "TODO: ranked-candidates.json"
    table = (
        '<div class="table-scroll"><table class="ranking">'
        '<thead><tr><th>rank</th><th>candidate_id</th><th>mean_ipSAE_min</th><th>sd_ipSAE_min</th>'
        '<th>n_folds</th><th>iface_err</th><th>fold_conf</th><th>len_aa</th><th>flag</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )
    notes = (
        '<dl class="column-notes">'
        '<dt>mean ipSAE_min</dt><dd>Stored central estimate across independent seeds.</dd>'
        '<dt>sd ipSAE_min</dt><dd>Stored seed-to-seed spread.</dd>'
        '<dt>n_folds</dt><dd>Stored fold count behind the row.</dd>'
        '<dt>iface_err</dt><dd>Stored interface error. Lower values indicate a lower reported error.</dd>'
        '<dt>fold_conf</dt><dd>Binder-only folding confidence.</dd>'
        '<dt>len_aa</dt><dd>Binder chain length in residues.</dd>'
        '<dt>flag</dt><dd>AT-FLOOR marks an exact zero. = marks adjacent rows tied within stored seed noise.</dd>'
        '</dl>'
    )
    return (
        table
        + _decision_floor_accounting(rows, label="ranked candidate rows")
        + notes
        + f'<p class="source-line">Full ranking artifact: {html.escape(source)}.</p>'
    )


def _ranking_number(value: Any, label: str) -> str:
    if not _is_number(value):
        return _missing(f"TODO: stored {label}")
    return html.escape(format(float(value), ".10g"))


def _monomer_interpretation(candidate: Candidate, threshold: float) -> str:
    metric = _monomer_confidence(candidate)
    if metric is None or not _is_number(metric.value):
        return _missing("TODO: stored monomer confidence; complex score is uninterpretable")
    confidence = float(metric.value)
    if confidence < threshold:
        return html.escape(
            f"Uninterpretable: monomer confidence {format(confidence, '.10g')} is below {format(threshold, '.10g')}."
        )
    return "Interpretable against the monomer-confidence threshold."


def _rank_score_records(candidates: Iterable[Candidate], threshold: float) -> list[str]:
    rows: list[str] = []
    for candidate in candidates:
        record = candidate.rank_row or {}
        central = record.get("rank_score_central_estimate", record.get("rank_score"))
        spread = record.get("rank_score_spread")
        rows.append(
            "<tr>"
            f'<th scope="row">{html.escape(candidate.candidate_id)}</th>'
            f'<td>{_ranking_number(central, "rank-score central estimate")}</td>'
            f'<td>{_ranking_number(spread, "rank-score spread")}</td>'
            f'<td>{_decision_monomer(_monomer_confidence(candidate))}</td>'
            f'<td>{_monomer_interpretation(candidate, threshold)}</td>'
            "</tr>"
        )
    return rows


def _seed_summary_records(candidates: Iterable[Candidate]) -> list[str]:
    rows: list[str] = []
    for candidate in candidates:
        record = candidate.rank_row or {}
        per_predictor = record.get("per_predictor")
        if not isinstance(per_predictor, Mapping):
            continue
        for predictor, arm in per_predictor.items():
            if not isinstance(arm, Mapping):
                continue
            summaries = arm.get("metric_summary")
            if not isinstance(summaries, Mapping):
                continue
            for metric, summary in summaries.items():
                if not isinstance(summary, Mapping):
                    continue
                values = [
                    _ranking_number(summary.get(field), f"{metric} seed {label}")
                    for field, label in (
                        ("observed_seed_count", "count"),
                        ("mean", "mean"),
                        ("median", "median"),
                        ("standard_deviation", "sample standard deviation"),
                        ("minimum", "minimum"),
                        ("maximum", "maximum"),
                        ("range", "range"),
                    )
                ]
                rows.append(
                    "<tr>"
                    f'<th scope="row">{html.escape(candidate.candidate_id)}</th>'
                    f"<td>{html.escape(str(predictor))}</td>"
                    f"<td>{html.escape(str(metric))}</td>"
                    f"{''.join(f'<td>{value}</td>' for value in values)}"
                    "</tr>"
                )
    return rows


def _seed_selection_provenance_records(candidates: Iterable[Candidate]) -> list[str]:
    """Explain which seed supplies secondary metrics under max aggregation."""
    rows: list[str] = []
    for candidate in candidates:
        record = candidate.rank_row or {}
        per_predictor = record.get("per_predictor")
        if not isinstance(per_predictor, Mapping):
            continue
        for predictor, arm in per_predictor.items():
            provenance = arm.get("aggregation_provenance") if isinstance(arm, Mapping) else None
            if not isinstance(provenance, Mapping):
                continue
            tie = provenance.get("primary_tie")
            tie_seeds = tie.get("seed_ids") if isinstance(tie, Mapping) else None
            tie_text = (
                ", ".join(str(seed) for seed in tie_seeds)
                if isinstance(tie_seeds, list) and tie_seeds
                else "none"
            )
            rows.append(
                "<tr>"
                f'<th scope="row">{html.escape(candidate.candidate_id)}</th>'
                f"<td>{html.escape(str(predictor))}</td>"
                f"<td>{html.escape(str(provenance.get('primary_metric')))}</td>"
                f"<td>{html.escape(str(provenance.get('seed_aggregation')))}</td>"
                f"<td>{html.escape(str(provenance.get('selected_seed')))}</td>"
                f"<td>{html.escape(tie_text)}</td>"
                f"<td>{html.escape(str(provenance.get('secondary_metric_basis')))}</td>"
                "</tr>"
            )
    return rows


def _seed_aggregation_comparison_records(candidates: Iterable[Candidate]) -> list[str]:
    rows: list[str] = []
    for candidate in candidates:
        record = candidate.rank_row or {}
        reports = record.get("seed_aggregation_reports")
        if not isinstance(reports, Mapping):
            continue
        maximum = reports.get("max")
        median = reports.get("median")
        if not isinstance(maximum, Mapping) or not isinstance(median, Mapping):
            continue
        rows.append(
            "<tr>"
            f'<th scope="row">{html.escape(candidate.candidate_id)}</th>'
            f'<td>{_ranking_number(maximum.get("rank"), "maximum-seed rank")}</td>'
            f'<td>{_ranking_number(maximum.get("rank_score"), "maximum-seed rank score")}</td>'
            f'<td>{_ranking_number(maximum.get("ipsae_min_ensemble"), "maximum-seed ipSAE_min")}</td>'
            f'<td>{_ranking_number(median.get("rank"), "median-seed rank")}</td>'
            f'<td>{_ranking_number(median.get("rank_score"), "median-seed rank score")}</td>'
            f'<td>{_ranking_number(median.get("ipsae_min_ensemble"), "median-seed ipSAE_min")}</td>'
            "</tr>"
        )
    return rows


def _seed_aggregation_ipsae_scores(
    candidates: Iterable[Candidate],
    aggregation: str,
) -> list[float | None]:
    """Return displayed ipSAE_min values for one seed aggregation method."""

    scores: list[float | None] = []
    for candidate in candidates:
        record = candidate.rank_row or {}
        reports = record.get("seed_aggregation_reports")
        report = reports.get(aggregation) if isinstance(reports, Mapping) else None
        value = report.get("ipsae_min_ensemble") if isinstance(report, Mapping) else None
        scores.append(float(value) if _is_number(value) else None)
    return scores


def _seed_summary_ipsae_mean_scores(candidates: Iterable[Candidate]) -> list[float | None]:
    """Return each displayed per-arm ipSAE_min mean."""

    scores: list[float | None] = []
    for candidate in candidates:
        record = candidate.rank_row or {}
        per_predictor = record.get("per_predictor")
        if not isinstance(per_predictor, Mapping):
            continue
        for arm in per_predictor.values():
            summaries = arm.get("metric_summary") if isinstance(arm, Mapping) else None
            summary = summaries.get("ipsae_min") if isinstance(summaries, Mapping) else None
            value = summary.get("mean") if isinstance(summary, Mapping) else None
            scores.append(float(value) if _is_number(value) else None)
    return scores


def _unranked_candidate_records(ranking: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    direct = ranking.get("unranked_candidates")
    if isinstance(direct, list):
        return [record for record in direct if isinstance(record, Mapping)]
    receipt = ranking.get("ranking_receipt")
    received = receipt.get("unranked_candidates") if isinstance(receipt, Mapping) else None
    return [record for record in received if isinstance(record, Mapping)] if isinstance(received, list) else []


def _render_ranking_precision(
    candidates: Iterable[Candidate],
    ranking: Mapping[str, Any],
    monomer_confidence_threshold: float,
) -> str:
    """Render stored spread, seed summary, separability, and refusal records."""

    candidate_rows = list(candidates)
    decision_rows = _decision_rows(candidate_rows)
    score_rows = _rank_score_records(candidate_rows, monomer_confidence_threshold)
    comparison = ranking.get("seed_aggregation_comparison")
    bias = (
        comparison.get("max_of_five_seed_bias_standard_deviations")
        if isinstance(comparison, Mapping)
        else None
    )
    comparison_rows = _seed_aggregation_comparison_records(candidate_rows)
    body_content = "".join(score_rows) if score_rows else f"<tr><td colspan='5'>{_missing('TODO: ranked candidate rows')}</td></tr>"
    sections = [
        '<h2>Ranking precision records</h2>'
        '<p>The central estimate and spread identify the values used to order the candidates.</p>'
        '<div class="table-scroll"><table><thead><tr><th>candidate_id</th>'
        '<th>rank_score_central_estimate</th><th>rank_score_spread</th><th>monomer confidence</th>'
        '<th>complex-score interpretation</th></tr></thead>'
        f'<tbody>{body_content}</tbody></table></div>'
        + _decision_floor_accounting(decision_rows, label="ranking precision rows")
    ]
    if comparison_rows and _is_number(bias):
        sections.extend(
            [
                '<h2>Seed-reduction comparison</h2>',
                '<p>Maximum across seeds is the published baseline and the primary rank. '
                'Median across seeds is reported beside it. '
                f'The expected max-of-five upward bias is {html.escape(format(float(bias), ".10g"))} standard deviations.</p>',
                '<div class="table-scroll"><table><thead><tr><th>candidate_id</th><th>max rank</th>'
                '<th>max rank_score</th><th>max ipSAE_min</th><th>median rank</th>'
                '<th>median rank_score</th><th>median ipSAE_min</th></tr></thead>'
                f'<tbody>{"".join(comparison_rows)}</tbody></table></div>'
                + _floor_accounting(
                    _seed_aggregation_ipsae_scores(candidate_rows, "max"),
                    label="maximum-seed ipSAE_min rows",
                )
                + _floor_accounting(
                    _seed_aggregation_ipsae_scores(candidate_rows, "median"),
                    label="median-seed ipSAE_min rows",
                ),
            ]
        )
    summary_rows = _seed_summary_records(candidate_rows)
    summary_body = "".join(summary_rows) if summary_rows else f"<tr><td colspan='10'>{_missing('TODO: per-arm seed summaries')}</td></tr>"
    sections.extend(
        [
            '<h2>Per-arm seed summaries</h2>',
            '<p>Each row reports the stored count, mean, median, sample standard deviation, minimum, maximum, and range.</p>',
            '<div class="table-scroll"><table><thead><tr><th>candidate_id</th><th>predictor arm</th><th>metric</th>'
            '<th>count</th><th>mean</th><th>median</th><th>sample SD</th><th>minimum</th><th>maximum</th><th>range</th></tr></thead>'
            f'<tbody>{summary_body}</tbody></table></div>'
            + _floor_accounting(
                _seed_summary_ipsae_mean_scores(candidate_rows),
                label="per-arm ipSAE_min mean rows",
            ),
        ]
    )
    provenance_rows = _seed_selection_provenance_records(candidate_rows)
    if provenance_rows:
        sections.extend(
            [
                '<h2>Seed selection provenance</h2>',
                '<p>Under maximum aggregation, the selected seed maximizes the primary metric. '
                'Every displayed secondary metric for that predictor comes from the same selected seed. '
                'Exact primary-score ties choose the lowest seed and list every tied seed below.</p>',
                '<div class="table-scroll"><table><thead><tr><th>candidate_id</th>'
                '<th>predictor arm</th><th>primary metric</th><th>aggregation</th>'
                '<th>selected seed</th><th>primary-tied seeds</th><th>secondary metric basis</th>'
                f'</tr></thead><tbody>{"".join(provenance_rows)}</tbody></table></div>',
            ]
        )
    separability = ranking.get("separability")
    pairs = separability.get("adjacent_pairs") if isinstance(separability, Mapping) else None
    pair_rows: list[str] = []
    pair_scores: list[float | None] = []
    if isinstance(pairs, list):
        for pair in pairs:
            if not isinstance(pair, Mapping):
                continue
            pair_scores.append(
                float(pair["rank_score_gap"])
                if _is_number(pair.get("rank_score_gap"))
                else None
            )
            outcome = "tied within noise" if pair.get("tied_within_noise") is True else "separable"
            pair_rows.append(
                "<tr>"
                f'<th scope="row">{html.escape(str(pair.get("higher_candidate_id", "TODO: higher candidate")))}</th>'
                f'<td>{html.escape(str(pair.get("lower_candidate_id", "TODO: lower candidate")))}</td>'
                f'<td>{_ranking_number(pair.get("rank_score_gap"), "adjacent rank-score gap")}</td>'
                f'<td>{_ranking_number(pair.get("rank_score_gap_spread"), "rank-score gap spread")}</td>'
                f"<td>{outcome}</td>"
                "</tr>"
            )
    top_ten = separability.get("top_ten") if isinstance(separability, Mapping) else None
    top_ten_note = _missing("TODO: top-ten separability summary")
    if isinstance(top_ten, Mapping):
        position_count = top_ten.get("position_count")
        separable_count = top_ten.get("separable_position_count")
        tied_count = top_ten.get("tied_within_noise_position_count")
        if all(_is_number(value) for value in (position_count, separable_count, tied_count)):
            top_ten_note = (
                f"Top {int(position_count)} positions: {int(separable_count)} separable and "
                f"{int(tied_count)} tied within noise."
            )
    pair_body = "".join(pair_rows) if pair_rows else f"<tr><td colspan='5'>{_missing('TODO: adjacent-pair separability records')}</td></tr>"
    sections.extend(
        [
            '<h2>Adjacent-pair separability</h2>',
            f'<p>{top_ten_note}</p>',
            '<div class="table-scroll"><table><thead><tr><th>higher candidate</th><th>lower candidate</th>'
            '<th>rank-score gap</th><th>rank-score gap spread</th><th>outcome</th></tr></thead>'
            f'<tbody>{pair_body}</tbody></table></div>'
            + _floor_accounting(pair_scores, label="adjacent rank-score gap rows"),
        ]
    )
    unranked = _unranked_candidate_records(ranking)
    unranked_rows = []
    for record in unranked:
        candidate_id = record.get("candidate_id", "TODO: candidate identifier")
        reasons = record.get("reasons") or record.get("ranking_refusal_reasons")
        reason_text = "; ".join(str(reason) for reason in reasons) if isinstance(reasons, list) else str(reasons or "TODO: refusal reason")
        unranked_rows.append(
            "<tr>"
            f'<th scope="row">{html.escape(str(candidate_id))}</th>'
            f"<td>{html.escape(reason_text)}</td>"
            "</tr>"
        )
    unranked_body = "".join(unranked_rows) if unranked_rows else "<tr><td colspan='2'>No candidates were withheld for insufficient scored seeds.</td></tr>"
    sections.extend(
        [
            '<h2>Unranked candidates</h2>',
            '<p>These candidates lacked the scored seed observations required for a ranking.</p>',
            '<div class="table-scroll"><table><thead><tr><th>candidate_id</th><th>reason</th></tr></thead>'
            f'<tbody>{unranked_body}</tbody></table></div>',
        ]
    )
    return "".join(sections)


def _render_structural_surrogate_disclosure(root: Path) -> str:
    """Render the stored provisional disclosure for a passed-control report."""

    for suffix in ("*.json", "*.jsonl"):
        for path in sorted(root.rglob(suffix)):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if structural_surrogates.SURROGATE_DISCLOSURE not in text:
                continue
            source = Provenance(_relative_artifact(path, root), detail="surrogate disclosure")
            return (
                f"<p>{html.escape(structural_surrogates.SURROGATE_DISCLOSURE)}</p>"
                f'<div class="source-line">{_provenance_html(source)}</div>'
            )
    return ""


_FILTER_GATE_STATUSES = frozenset({"ran and passed", "ran and failed", "skipped"})


def _legacy_filter_gate_status(gate: Mapping[str, Any]) -> tuple[str, str]:
    """Recover an explicit status from a filter report written before gate statuses."""
    evaluated = gate.get("evaluated_candidate_count")
    removed = gate.get("removed_candidate_count")
    if not isinstance(evaluated, int) or isinstance(evaluated, bool) or evaluated < 1:
        return "skipped", "The stage recorded zero candidate evaluations for this gate."
    if isinstance(removed, int) and not isinstance(removed, bool) and removed > 0:
        return "ran and failed", f"{removed} candidate records failed this gate."
    return "ran and passed", f"All {evaluated} candidate records passed this gate."


def _filter_gate_statuses(root: Path, config: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Collect active and retired filter statuses for the scientist-facing report."""
    rows: list[dict[str, str]] = []
    observed: set[tuple[str, str]] = set()
    retired = {gate["filter_id"]: gate for gate in retired_filter_statuses()}
    paths = sorted({*root.rglob("*filter-report.json"), *root.rglob("screen-report.json")})
    for path in paths:
        try:
            document = _read_json(path)
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(document, Mapping):
            continue
        gate_report = document.get("gate_report", document)
        if not isinstance(gate_report, Mapping):
            continue
        stage_id = gate_report.get("stage_id")
        if not isinstance(stage_id, str) or not stage_id:
            continue
        statuses = gate_report.get("gate_statuses")
        if not isinstance(statuses, list):
            legacy = gate_report.get("evaluated_gates")
            statuses = []
            if isinstance(legacy, list):
                for gate in legacy:
                    if not isinstance(gate, Mapping):
                        continue
                    status, reason = _legacy_filter_gate_status(gate)
                    statuses.append(
                        {
                            "filter_id": gate.get("filter_id"),
                            "status": status,
                            "reason": reason,
                        }
                    )
        for gate in statuses:
            if not isinstance(gate, Mapping):
                continue
            filter_id = gate.get("filter_id")
            status = gate.get("status")
            reason = gate.get("reason")
            if not isinstance(filter_id, str) or not filter_id:
                continue
            if filter_id in retired:
                status = "skipped"
                reason = retired[filter_id]["reason"]
            elif status not in _FILTER_GATE_STATUSES:
                status = "skipped"
                reason = "The filter report recorded an unrecognized gate status."
            if not isinstance(reason, str) or not reason:
                reason = "The filter report omitted the gate reason."
            rows.append(
                {
                    "filter_id": filter_id,
                    "stage_id": stage_id,
                    "status": status,
                    "reason": reason,
                    "source": _relative_artifact(path, root),
                }
            )
            observed.add((stage_id, filter_id))

    filters = config.get("filters") if isinstance(config, Mapping) else None
    if isinstance(filters, Mapping):
        disabled = filters.get("disabled_checks")
        disabled_ids = (
            {value for value in disabled if isinstance(value, str)}
            if isinstance(disabled, list)
            else set()
        )
        contracts = filters.get("contracts")
        if isinstance(contracts, list):
            for contract in contracts:
                if not isinstance(contract, Mapping):
                    continue
                filter_id = contract.get("filter_id")
                stage_id = contract.get("stage_id")
                if (
                    not isinstance(filter_id, str)
                    or not filter_id
                    or not isinstance(stage_id, str)
                    or not stage_id
                    or filter_id in disabled_ids
                    or (stage_id, filter_id) in observed
                ):
                    continue
                rows.append(
                    {
                        "filter_id": filter_id,
                        "stage_id": stage_id,
                        "status": "skipped",
                        "reason": "The run did not record a filter report for this gate.",
                        "source": "resolved configuration",
                    }
                )
                observed.add((stage_id, filter_id))

    retired_ids = {row["filter_id"] for row in rows if row["status"] == "skipped"}
    for gate in retired.values():
        filter_id = gate["filter_id"]
        if filter_id in retired_ids:
            continue
        rows.append(
            {
                "filter_id": filter_id,
                "stage_id": "filter-novelty",
                "status": gate["status"],
                "reason": gate["reason"],
                "source": "package filter contract",
            }
        )
    return sorted(rows, key=lambda row: (row["stage_id"], row["filter_id"], row["source"]))


def _render_filter_gate_statuses(root: Path, config: Mapping[str, Any] | None) -> str:
    """Render each active or skipped filter gate with its measured status."""
    rows = _filter_gate_statuses(root, config)
    body = "".join(
        "<tr>"
        f'<th scope="row">{html.escape(row["filter_id"])}</th>'
        f'<td>{html.escape(row["stage_id"])}</td>'
        f'<td>{html.escape(row["status"])}</td>'
        f'<td>{html.escape(row["reason"])}</td>'
        f'<td>{html.escape(row["source"])}</td>'
        "</tr>"
        for row in rows
    )
    return (
        '<div class="table-scroll"><table><thead><tr>'
        '<th scope="col">Gate</th><th scope="col">Stage</th><th scope="col">Status</th>'
        '<th scope="col">Reason</th><th scope="col">Source</th>'
        f"</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _render_files(root: Path, ranked_path: Path | None, fasta_records: list[tuple[str, str, str, Path]]) -> str:
    tsv_path = next(iter(sorted(root.rglob("ranked_candidates.tsv"))), None)
    fasta_path = fasta_records[0][3] if fasta_records else None
    manifest_path = next(
        iter(sorted(path for path in root.rglob("*.json") if path.name in {"run_manifest.json", "run-manifest.json"})),
        None,
    )

    def artifact(path: Path | None, required_name: str) -> str:
        return html.escape(_relative_artifact(path, root)) if path is not None else _missing(f"TODO: {required_name}")

    return (
        '<dl class="files">'
        f'<dt>ranked_candidates.tsv</dt><dd>{artifact(tsv_path, "ranked_candidates.tsv")}<br>Decision table with nine columns.</dd>'
        f'<dt>candidates.fasta</dt><dd>{artifact(fasta_path, "candidates.fasta")}<br>Sequences in ranked order when the run wrote that order.</dd>'
        f'<dt>run_manifest.json</dt><dd>{artifact(manifest_path, "run_manifest.json")}<br>Commands, versions, seeds, raw scores, and metered usage.</dd>'
        '</dl>'
    )


def _render_refusal_page(
    *,
    root: Path,
    config: Mapping[str, Any] | None,
    ranking: Mapping[str, Any],
    gate: ControlGate,
    run_id: str | None,
    target: str,
    date: str | None,
    constraint_html: str,
) -> str:
    run_label = html.escape(run_id) if run_id else _missing("TODO: run identifier")
    date_label = html.escape(date) if date else _missing("TODO: run date")
    detail = html.escape(gate.reason or "TODO: control failure reason")
    source = _provenance_html(gate.source) if gate.source is not None else ""
    return (
        '<section class="refusal">'
        '<h1>Results withheld: control checks failed</h1>'
        f'<p>Run {run_label}. Target {target}. Date {date_label}.</p>'
        '<p>The run completed, but the report withholds every ranking until registered controls pass.</p>'
        '<p>No candidate IDs, sequences, scores, or predicted-model images appear on this page.</p>'
        f'<h2>Run constraint</h2>{constraint_html}'
        '<h2>Control results</h2>'
        f'{_render_controls(root, ranking, gate)}'
        '<h2>Filter gates</h2>'
        f'{_render_filter_gate_statuses(root, config)}'
        '<h2>Failure detail</h2>'
        f'<p class="failure-detail">{detail}</p><p class="source-line">{source}</p>'
        '<h2>What happened to your data</h2>'
        '<p>The run directory retains the candidate artifacts. This page excludes them because the control result blocks interpretation.</p>'
        '<h2>Next step</h2>'
        '<p>Inspect the registered control observations and calibration records before you rerun any controls or the campaign.</p>'
        '</section>'
    )


def _assert_scientist_page_safe(rendered: str) -> None:
    """Keep unsupported claims and deprecated presentation fields off the report."""

    lowered = rendered.lower()
    for term in _FORBIDDEN_REPORT_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            raise AssertionError(f"forbidden report term: {term}")
    disclosure_free = lowered.replace(
        html.escape(structural_surrogates.SURROGATE_DISCLOSURE).lower(),
        "",
    )
    for gate in retired_filter_statuses():
        disclosure_free = disclosure_free.replace(html.escape(gate["reason"]).lower(), "")
    for term in ("tm-align", "foldseek", "dssp"):
        if re.search(rf"\b{re.escape(term)}\b", disclosure_free):
            raise AssertionError(f"unlabelled structural-tool term in report: {term}")
    if re.search(r"\b(?:kd|nm)\b", lowered):
        raise AssertionError("forbidden affinity unit in report")
    if "p-value" in lowered or "p value" in lowered or "significant" in lowered:
        raise AssertionError("forbidden inferential statistic in report")
    for match in re.finditer(r"0\.000", rendered):
        following = rendered[match.end() : match.end() + 100]
        if "AT-FLOOR" not in following:
            raise AssertionError("bare 0.000 without an AT-FLOOR marker")


def _scientist_html_document(title: str, body: str) -> str:
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title} report</title>
<style>
:root {{ color-scheme:light; --ink:#17212b; --muted:#596773; --line:#d6dde3; --panel:#f5f7f8; --warn:#7c3028; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#eef2f5; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:32px 22px 64px; }}
h1 {{ margin:0 0 12px; font-size:30px; letter-spacing:-.02em; }}
h2 {{ margin:28px 0 10px; font-size:20px; }}
p {{ max-width:78ch; margin:8px 0; }}
.run-meta,.panel,.refusal {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:18px; margin-top:14px; }}
.run-meta {{ display:grid; grid-template-columns:minmax(130px,190px) 1fr; margin:0; }}
.run-meta dt,.run-meta dd,.files dt,.files dd,.column-notes dt,.column-notes dd {{ margin:0; padding:8px; border-bottom:1px solid var(--line); }}
.run-meta dt,.files dt,.column-notes dt {{ font-weight:600; background:var(--panel); }}
.missing {{ color:var(--warn); background:#fff1ef; border:1px solid #e8c2bc; border-radius:4px; padding:2px 5px; font-size:13px; }}
.table-scroll {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ padding:9px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
thead th {{ background:var(--panel); font-size:12px; letter-spacing:.04em; text-transform:uppercase; white-space:nowrap; }}
.flag {{ font-weight:600; white-space:nowrap; }}
.column-notes,.files {{ display:grid; grid-template-columns:minmax(150px,230px) 1fr; margin:16px 0 0; }}
.source-line {{ color:var(--muted); font-size:13px; }}
.failure-detail {{ color:var(--warn); font-weight:600; }}
@media (max-width:700px) {{ main {{ padding:20px 12px 40px; }} .run-meta,.files,.column-notes {{ display:block; }} .run-meta dt,.files dt,.column-notes dt {{ margin-top:8px; }} }}
</style>
</head>
<body><main>{body}</main></body>
</html>
"""


def _candidate_card(candidate: Candidate, root: Path, metric_keys: list[tuple[str, str]]) -> str:
    structure = "".join(f'<li>{html.escape(_display_path(Path(path), root))}</li>' for path in candidate.structure_paths)
    structure_html = f"<ul>{structure}</ul>" if structure else _missing("structure file")
    return (
        f'<article class="candidate" id="candidate-{html.escape(candidate.candidate_id)}">'
        f'<h3>{html.escape(candidate.candidate_id)}</h3>'
        '<div class="candidate-grid">'
        f'<section><h4>Sequence</h4>{_render_sequence(candidate)}</section>'
        f'<section><h4>Length and composition</h4>{_render_composition(candidate, root)}</section>'
        f'<section><h4>Metrics</h4>{_render_metrics(candidate, metric_keys)}</section>'
        f'<section><h4>Structure image</h4>{_render_image(candidate, root)}</section>'
        f'<section><h4>Structure file</h4>{structure_html}</section>'
        '</div></article>'
    )


def build_report(
    run_dir: str | Path,
    monomer_confidence_threshold: float = DEFAULT_MONOMER_CONFIDENCE_THRESHOLD,
) -> str:
    """Return one self-contained HTML report for ``run_dir``."""

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    monomer_confidence_threshold = _validate_monomer_confidence_threshold(monomer_confidence_threshold)
    request_path = root / "request.json"
    try:
        request = _read_json(request_path) if request_path.is_file() else None
    except (OSError, ValueError, TypeError):
        request = None
    if not isinstance(request, Mapping):
        request = None
    config, config_path = _find_config(root)
    unconstrained = _is_unconstrained_run(config)
    run_records = _find_run_records(root)
    images = _find_images(root)
    fasta_records = _find_fasta_records(root)
    result_rows, result_path = _find_results(root)
    ranked, ranked_path, ranked_rows, ranking_document = _ranked_candidates(
        root,
        config,
        run_records,
        fasta_records,
        images,
    )
    if ranked:
        candidates = ranked
    else:
        candidates = _screen_candidates(root, run_records, fasta_records, images)
        if not candidates and result_rows:
            candidates = _result_candidates(root, result_rows, run_records, fasta_records, images)
        if not candidates:
            candidates = _fasta_only_candidates(root, fasta_records, images, run_records)
    if not result_path and candidates and result_rows:
        result_path = result_rows[0][1]
    title = root.name or "run"
    gate = _control_gate(ranking_document or {}, ranked_path, root)
    run_id = _run_value(ranking_document or {}, config, request, run_records, _RUN_ID_FIELDS)
    date = _run_value(
        ranking_document or {},
        config,
        request,
        run_records,
        ("date", "completed_at", "timestamp", "finished_at"),
    )
    target = _target_summary(ranking_document or {}, config)
    if gate.state == "failed":
        rendered = _scientist_html_document(
            title,
            _render_refusal_page(
                root=root,
                config=config,
                ranking=ranking_document or {},
                gate=gate,
                run_id=run_id,
                target=target,
                date=date,
                constraint_html=_render_constraint(config),
            ),
        )
        if unconstrained:
            rendered = re.sub("epitope", "contact", rendered, flags=re.IGNORECASE)
        _assert_scientist_page_safe(rendered)
        return rendered

    _merge_monomer_metrics(candidates, _monomer_metrics(root, run_records))
    decision_rows = _decision_rows(candidates)
    requested, _ = _stored_number(request, ("num_proteins", "candidate_count", "requested_candidates"))
    requested_text = (
        str(int(requested))
        if requested is not None and requested.is_integer()
        else _missing("TODO: requested candidate count")
    )
    top = decision_rows[0] if decision_rows and decision_rows[0].mean_ipsae_min is not None else None
    if top is None:
        result_line = _missing("TODO: stored mean ipSAE_min and seed spread for the top candidate")
    else:
        result_line = (
            f'Candidate <strong>{html.escape(top.candidate.candidate_id)}</strong> has stored mean ipSAE_min '
            f'{_decision_number(top.mean_ipsae_min, top.mean_field, "mean ipSAE_min")} across '
            f'{_decision_number(top.n_folds, top.n_folds_field, "fold count")} folds. '
            f'Its stored interface error is {_decision_number(top.interface_error, top.interface_error_field, "interface error")}. '
            f'Its monomer confidence is {_decision_monomer(top.monomer_confidence)}.'
        )
    structural_surrogate_html = _render_structural_surrogate_disclosure(root)
    structural_surrogate_section = (
        '<section class="panel"><h2>Structural surrogate disclosure</h2>'
        f"{structural_surrogate_html}</section>"
        if structural_surrogate_html
        else ""
    )
    body = (
        '<h1>Protein binder design campaign</h1>'
        '<dl class="run-meta">'
        f'<dt>Run</dt><dd>{html.escape(run_id) if run_id else _missing("TODO: run identifier")}</dd>'
        f'<dt>Target</dt><dd>{target}</dd>'
        f'<dt>Requested</dt><dd>{requested_text} candidates requested. {len(decision_rows)} candidate rows available.</dd>'
        f'<dt>Date</dt><dd>{html.escape(date) if date else _missing("TODO: run date")}</dd>'
        f'<dt>Backend</dt><dd>{_backend_summary(ranking_document or {}, config)}</dd>'
        '</dl>'
        '<section class="panel"><h2>Result in one line</h2>'
        f'<p>{result_line}</p>'
        '<p>Ranks use the stored central ipSAE_min estimate across independent seeds.</p>'
        f'{_decision_floor_accounting(decision_rows, label="result summary rows")}</section>'
        '<section class="panel"><h2>Read this before the table</h2>'
        '<p>This page reports computational predictions of interface plausibility. Experimental assays measure binding.</p>'
        '<p>The primary ipSAE_min metric has a hard 10 Å aligned-error cutoff. Values below that cutoff are exactly 0.0.</p>'
        '<p>An exact zero in the mean column carries the AT-FLOOR flag. The run must record the scorer floor definition and seed-noise calibration in controls.json.</p></section>'
        '<section class="panel"><h2>Controls run with this campaign</h2>'
        f'<p>Ranking control status: PASS.</p>{_render_controls(root, ranking_document or {}, gate)}</section>'
        f'<section class="panel"><h2>Filter gates</h2>{_render_filter_gate_statuses(root, config)}</section>'
        f'<section class="panel"><h2>Run constraint</h2>{_render_constraint(config)}</section>'
        f'<section class="panel"><h2>Contact clusters</h2>{_render_contact_clusters(root, unconstrained)}</section>'
        f"{structural_surrogate_section}"
        f'<section class="panel"><h2>Top {min(10, len(decision_rows))} of {len(decision_rows)} candidates</h2>{_render_ranking_table(decision_rows[:10], ranked_path, root)}</section>'
        '<section class="panel">'
        f'{_render_ranking_precision(candidates, ranking_document or {}, monomer_confidence_threshold)}</section>'
        '<section class="panel"><h2>Known limits of this run</h2>'
        f'{_render_baseline_fidelity_disclosure(ranking_document)}'
        f'{_render_predictor_arm_disclosure(config)}'
        '<p>The Filter gates table records each active filter result and each skipped filter reason.</p></section>'
        f'<section class="panel"><h2>Spend and provenance</h2>{_render_spend(root)}</section>'
        f'<section class="panel"><h2>Files to keep</h2>{_render_files(root, ranked_path, fasta_records)}</section>'
    )
    rendered = _scientist_html_document(title, body)
    if unconstrained:
        rendered = re.sub("epitope", "contact", rendered, flags=re.IGNORECASE)
    _assert_scientist_page_safe(rendered)
    return rendered


def _html_document(
    title: str,
    request_html: str,
    stage_html: str,
    spend_html: str,
    rescore_html: str,
    scorer_html: str,
    ranking_html: str,
    candidates_html: str,
    backbone_html: str,
    figures_html: str,
    separation_html: str,
) -> str:
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title} computational design report</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#5b6875; --line:#d7dee5; --panel:#f6f8fa; --accent:#245d78; --missing:#8b3a2f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; overflow-x:hidden; color:var(--ink); background:#eef2f5; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1240px; margin:0 auto; padding:32px 22px 64px; }}
h1 {{ margin:0 0 6px; font-size:30px; letter-spacing:-.02em; }}
h2 {{ margin:32px 0 12px; font-size:21px; }}
h3 {{ margin:0 0 16px; font-size:20px; color:var(--accent); }}
h4 {{ margin:0 0 10px; font-size:15px; }}
p {{ margin:8px 0; }}
.subtitle {{ color:var(--muted); margin:0 0 22px; }}
.scope {{ background:#fff; border:1px solid var(--line); border-left:4px solid var(--accent); padding:13px 16px; }}
.panel,.candidate {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:18px; margin-top:12px; box-shadow:0 1px 2px rgba(23,33,43,.04); }}
.candidate-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:20px; }}
.candidate-grid section {{ min-width:0; }}
.table-scroll {{ max-width:100%; overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; }}
th,td {{ border-bottom:1px solid var(--line); padding:9px 8px; text-align:left; vertical-align:top; }}
thead th {{ background:var(--panel); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }}
tbody th {{ font-weight:600; }}
.metrics th {{ width:42%; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; }}
.context {{ color:var(--muted); font:12px/1.3 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; display:block; margin-top:2px; }}
.number {{ display:inline-flex; flex-wrap:wrap; gap:7px; align-items:baseline; }}
.provenance {{ color:var(--muted); font:11px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace; overflow-wrap:anywhere; }}
.source-line {{ color:var(--muted); margin-top:6px; }}
.missing {{ color:var(--missing); background:#fff1ef; border:1px solid #e8c2bc; border-radius:4px; padding:2px 5px; font-size:13px; }}
.uninterpretable {{ color:var(--missing); font-weight:600; }}
.interpretable {{ color:var(--muted); }}
.interpretation {{ max-width:72ch; }}
.sequence {{ display:block; width:100%; min-height:92px; resize:vertical; padding:10px; border:1px solid var(--line); background:#fbfcfd; color:var(--ink); font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; }}
figure {{ margin:0; }}
figure img {{ display:block; max-width:100%; max-height:360px; margin:0 auto 8px; border:1px solid var(--line); background:#fafafa; }}
figcaption {{ overflow-wrap:anywhere; }}
.figure-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:20px; }}
.figure-grid figure {{ min-width:0; }}
.figure-grid h3 {{ margin:0 0 10px; font-size:15px; }}
.spend-list {{ display:grid; grid-template-columns:minmax(180px,260px) 1fr; margin:0; }}
.spend-list dt,.spend-list dd {{ margin:0; padding:9px 8px; border-bottom:1px solid var(--line); }}
.spend-list dt {{ font-weight:600; background:var(--panel); }}
.request-list {{ display:grid; grid-template-columns:minmax(150px,220px) 1fr; gap:0; margin:0; }}
.request-field {{ display:contents; }}
.request-list dt,.request-list dd {{ margin:0; padding:9px 8px; border-bottom:1px solid var(--line); }}
.request-list dt {{ font-weight:600; background:var(--panel); }}
.nested,.nested-list {{ margin:0; }}
.nested {{ display:grid; grid-template-columns:max-content 1fr; gap:3px 12px; }}
.nested dt {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); }}
.nested dd {{ margin:0; }}
.nested-list {{ padding-left:20px; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
ul {{ padding-left:20px; margin:0; }}
@media (max-width:700px) {{ main {{ padding:22px 12px 40px; }} .request-list {{ display:block; }} .request-field {{ display:block; }} .request-list dt {{ margin-top:8px; }} th,td {{ padding:7px 5px; }} }}
</style>
</head>
<body>
<main>
<h1>{escaped_title}</h1>
<p class="subtitle">Computational design run report</p>
<p class="scope">Scope: computed sequences, metrics, and structure renderings.</p>
<section class="panel"><h2>Control separation qualification</h2>{separation_html}</section>
<section class="panel"><h2>What was requested</h2>{request_html}</section>
<section class="panel"><h2>Run counts</h2>{stage_html}</section>
<section class="panel"><h2>Spend</h2>{spend_html}</section>
<section class="panel"><h2>Rescore records</h2>{rescore_html}</section>
<section class="panel"><h2>Local scorer summary</h2>{scorer_html}</section>
<section class="panel"><h2>Ranking</h2>{ranking_html}</section>
<section class="panel"><h2>Rendered figures</h2>{figures_html}</section>
<section><h2>Designs</h2>{candidates_html}</section>
<section><h2>De novo backbones</h2>{backbone_html}</section>
</main>
</body>
</html>
"""


def write_report(
    run_dir: str | Path,
    output: str | Path,
    monomer_confidence_threshold: float = DEFAULT_MONOMER_CONFIDENCE_THRESHOLD,
) -> Path:
    """Write the report and return the resolved output path."""

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(run_dir, monomer_confidence_threshold), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    """Run the report CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="finished run directory")
    parser.add_argument("--out", type=Path, required=True, help="self-contained HTML output path")
    parser.add_argument(
        "--monomer-confidence-threshold",
        type=float,
        default=DEFAULT_MONOMER_CONFIDENCE_THRESHOLD,
        help="minimum normalized binder-only confidence for an interpretable complex score",
    )
    args = parser.parse_args(argv)
    try:
        output = write_report(args.run_dir, args.out, args.monomer_confidence_threshold)
    except Exception as exc:
        parser.error(f"{type(exc).__name__}: {exc}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
