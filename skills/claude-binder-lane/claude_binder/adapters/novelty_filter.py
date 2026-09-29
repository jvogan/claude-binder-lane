#!/usr/bin/env python3
"""Reduce configured novelty and structure metrics into filter observations.

The campaign names one source per registered novelty check under
``filters.metric_sources``. A source can be a receipt-independent JSONL metric file or
the local target-chain sliding-window screen. This adapter verifies that every candidate
has every configured metric, applies the configured threshold, and preserves candidate
lineage in the final passing manifest. It refuses when a reference, comparison result,
or threshold is absent.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from .integrity_filter import (
    AdapterError,
    completed_receipt_file,
    declared_artifact_refs,
    declared_input_ref,
    integrity_passing_manifest_path,
    load_candidates,
    observation,
    read_json_object,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from ..filter_contracts import (
    RETIRED_FILTER_TOOLS,
    filter_report,
    is_not_applicable,
    reference_digest_is_required,
    retired_filter_error,
    retired_filter_statuses,
)
from .. import structural_surrogates


STAGE_NOVELTY = "filter-novelty"
SUPPORTED_FILTERS = frozenset({"sequence_novelty", "model_likelihood"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TARGET_CHAIN_SOURCE_KIND = "target-chain-sliding-window"
CONTACT_MAP_OVERLAP_SOURCE_KIND = "contact-map-overlap-f1-surrogate"
KABSCH_SANDER_SOURCE_KIND = "kabsch-sander-secondary-structure-surrogate"
STRUCTURAL_DIVERSITY_CONFIG_KEY = "structural_diversity_surrogate"
STRUCTURAL_DIVERSITY_CLUSTER_FIELD = "tm90_cluster_id_surrogate"
STRUCTURAL_DIVERSITY_SECONDARY_FIELD = "secondary_structure_surrogate"
AMINO_ACID_3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
PUBLISHED_THRESHOLD_GUIDANCE = {
    "sequence_novelty": (
        "The published protocol rejects sequence matches above 60 percent identity "
        "over more than 50 percent coverage against UniRef90 or the binder corpus. "
        "Set the threshold and operator to match the metric source's representation."
    ),
    "structure_novelty": (
        "Contact-map overlap requires an anchor-set calibration threshold. Do not copy "
        "the published TM-score convention into the surrogate configuration."
    ),
    "model_likelihood": (
        "The published protocol gives no model_likelihood threshold. Set and document "
        "a campaign threshold before filtering."
    ),
    "secondary_structure": (
        "The published protocol gives no secondary_structure threshold. Set and "
        "document a campaign threshold before filtering."
    ),
}


def _finite_number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise AdapterError(f"{label} must be a finite number")
    number = float(value)
    if number < minimum or number > 1:
        raise AdapterError(f"{label} must be between {minimum:g} and 1")
    return number


def _source_alignment(source: dict[str, Any], label: str) -> structural_surrogates.AlignmentParameters:
    try:
        return structural_surrogates.alignment_parameters_from_mapping(
            source.get("alignment"), f"{label}.alignment"
        )
    except structural_surrogates.StructuralSurrogateError as exc:
        raise AdapterError(str(exc)) from exc


def _is_local_structural_source(source: dict[str, Any]) -> bool:
    return source.get("kind") in {CONTACT_MAP_OVERLAP_SOURCE_KIND, KABSCH_SANDER_SOURCE_KIND}


def validate_structural_metric_source(filter_id: str, source: dict[str, Any]) -> None:
    kind = source.get("kind")
    if kind == CONTACT_MAP_OVERLAP_SOURCE_KIND:
        if filter_id != "structure_novelty":
            raise AdapterError(f"metric source kind {kind} supports only structure_novelty")
        _source_alignment(source, f"filters.metric_sources.{filter_id}")
        return
    if kind == KABSCH_SANDER_SOURCE_KIND:
        if filter_id != "secondary_structure":
            raise AdapterError(f"metric source kind {kind} supports only secondary_structure")
        return
    raise AdapterError(f"metric source kind {kind!r} is not a local structural source")


def stage_round(stage: str, config: dict[str, Any]) -> int:
    if stage == STAGE_NOVELTY:
        return 0
    match = re.fullmatch(r"optimization-filter-novelty-round-(\d+)", stage)
    if match is not None:
        return int(match.group(1))
    for item in config.get("stages", []):
        if isinstance(item, dict) and item.get("stage_id") == stage:
            value = item.get("optimization_round", 0)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    raise AdapterError(f"stage is not a novelty-filter stage: {stage}")


def novelty_contracts(config: dict[str, Any]) -> list[dict[str, Any]]:
    filters = config.get("filters")
    if not isinstance(filters, dict):
        raise AdapterError("campaign config is missing required field: filters")
    raw_contracts = filters.get("contracts")
    if not isinstance(raw_contracts, list):
        raise AdapterError("campaign config is missing required field: filters.contracts")
    disabled = filters.get("disabled_checks", [])
    disabled_ids = {value for value in disabled if isinstance(value, str)} if isinstance(disabled, list) else set()
    retired_ids = sorted(
        {
            str(item.get("filter_id"))
            for item in raw_contracts
            if isinstance(item, dict)
            and item.get("stage_id") == STAGE_NOVELTY
            and item.get("filter_id") in RETIRED_FILTER_TOOLS
            and item.get("filter_id") not in disabled_ids
        }
    )
    if retired_ids:
        raise AdapterError(retired_filter_error(retired_ids[0]) or retired_ids[0])
    selected = [
        item for item in raw_contracts
        if (
            isinstance(item, dict)
            and item.get("stage_id") == STAGE_NOVELTY
            and item.get("filter_id") not in disabled_ids
            and item.get("filter_id") in SUPPORTED_FILTERS
        )
    ]
    expected_filters = SUPPORTED_FILTERS - disabled_ids
    if {item.get("filter_id") for item in selected} != expected_filters:
        raise AdapterError(
            "campaign config must define every enabled novelty filter in filters.contracts for filter-novelty"
        )
    for contract in selected:
        filter_id = contract["filter_id"]
        threshold = contract.get("threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
        ):
            raise AdapterError(
                f"campaign config is missing a numeric threshold: "
                f"filters.contracts[{filter_id}].threshold. "
                f"{PUBLISHED_THRESHOLD_GUIDANCE[filter_id]}"
            )
        for field in ("metric", "operator", "tool_revision", "reference_revision"):
            value = contract.get(field)
            if not isinstance(value, str) or not value or value == "__REQUIRED__":
                raise AdapterError(f"campaign config is missing required field: filters.contracts[{filter_id}].{field}")
        reference_sha256 = contract.get("reference_sha256")
        if reference_digest_is_required(contract.get("reference_revision")):
            if not isinstance(reference_sha256, str) or not SHA256_RE.fullmatch(reference_sha256):
                raise AdapterError(
                    f"campaign config is missing required field: filters.contracts[{filter_id}].reference_sha256"
                )
        elif not is_not_applicable(reference_sha256):
            raise AdapterError(
                f"campaign config must record filters.contracts[{filter_id}].reference_sha256 as not_applicable"
            )
    return selected


def metric_source_paths(
    config: dict[str, Any],
    config_path: Path,
    contracts: list[dict[str, Any]],
    *,
    plan_path: Path | None = None,
    receipts_dir: Path | None = None,
    metric_overrides: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """Resolve metric files from declared producer receipts.

    The old path-only form remains available to small unit fixtures that do not
    carry a resolved plan. A real stage invocation passes ``plan_path`` and
    ``receipts_dir``. In that form, config paths are descriptive metadata only.
    """
    filters = config["filters"]
    raw_sources = filters.get("metric_sources")
    if not contracts:
        # Every novelty check is disabled, so nothing will read a metric source.
        # Demanding the field here refused profiles that had correctly turned the
        # checks off, and the refusal arrived at stage 1 naming a field the user
        # had no reason to set.
        return {}
    if not isinstance(raw_sources, dict):
        raise AdapterError(
            "campaign config is missing required field: filters.metric_sources. "
            "It must name the executed output for each enabled novelty check. "
            f"Enabled checks: {', '.join(sorted(c['filter_id'] for c in contracts))}."
        )
    paths: dict[str, Path] = {}
    overrides = metric_overrides or {}
    inventory = declared_artifact_refs(plan_path) if plan_path is not None else None
    strict = inventory is not None and bool(inventory)
    missing: list[str] = []
    for contract in contracts:
        filter_id = contract["filter_id"]
        override = overrides.get(filter_id)
        if override is not None:
            path = override.expanduser().resolve()
            if not path.is_file():
                raise AdapterError(f"metric source override for {filter_id} is missing: {path}")
            paths[filter_id] = path
            continue
        source = raw_sources.get(filter_id)
        if not isinstance(source, dict):
            raise AdapterError(f"campaign config is missing required field: filters.metric_sources.{filter_id}")
        if source.get("kind") == TARGET_CHAIN_SOURCE_KIND:
            if filter_id != "sequence_novelty":
                raise AdapterError(
                    f"metric source kind {TARGET_CHAIN_SOURCE_KIND} supports only sequence_novelty"
                )
            validate_target_chain_source(source)
            continue
        if _is_local_structural_source(source):
            validate_structural_metric_source(filter_id, source)
            continue
        if strict:
            produced_by = source.get("produced_by")
            if not isinstance(produced_by, str) or produced_by.count(":") != 1:
                missing.append(f"{filter_id} (no declared producer)")
                continue
            producer_stage, artifact_id = produced_by.split(":", 1)
            if produced_by not in inventory:
                missing.append(produced_by)
                continue
            if receipts_dir is None:
                raise AdapterError(
                    f"metric source for {filter_id} is declared as {produced_by}, but no receipts directory was supplied"
                )
            paths[filter_id] = completed_receipt_file(
                receipts_dir,
                producer_stage,
                artifact_id,
                label=f"metric source for {filter_id}",
            )
            continue
        value = source.get("path")
        if not isinstance(value, str) or not value:
            raise AdapterError(f"campaign config is missing required field: filters.metric_sources.{filter_id}.path")
        path = Path(value)
        if not path.is_absolute():
            path = (config_path.parent / path).resolve()
        if not path.is_file():
            raise AdapterError(f"metric source for {filter_id} is missing: {path}")
        paths[filter_id] = path
    if missing:
        existing = ", ".join(sorted(inventory)) if inventory else "none"
        absent = ", ".join(sorted(missing))
        producer_ids = ", ".join(
            sorted(
                {
                    value.split(":", 1)[0]
                    for value in missing
                    if ":" in value and " (" not in value
                }
            )
        )
        producer_message = (
            f" Stage(s) {producer_ids} should have produced the missing artifact(s) before {STAGE_NOVELTY}."
            if producer_ids
            else f" An upstream stage must produce each missing metric before {STAGE_NOVELTY}."
        )
        raise AdapterError(
            f"{STAGE_NOVELTY} metric artifacts are not declared. "
            f"Existing artifact ids: {existing}. Missing artifact ids: {absent}."
            f"{producer_message}"
        )
    return paths


def metric_values(path: Path, contract: dict[str, Any]) -> dict[str, float]:
    rows = read_jsonl(path, f"metric source for {contract['filter_id']}")
    values: dict[str, float] = {}
    for index, row in enumerate(rows, start=1):
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise AdapterError(f"metric source {path} line {index} has no candidate_id")
        if candidate_id in values:
            raise AdapterError(f"metric source {path} repeats candidate_id: {candidate_id}")
        value = row.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AdapterError(f"metric source {path} line {index} has no finite numeric value")
        if "metric" in row and row["metric"] != contract["metric"]:
            raise AdapterError(
                f"metric source {path} line {index} names {row['metric']!r}; "
                f"the config names {contract['metric']!r}"
            )
        values[candidate_id] = float(value)
    return values


def validate_target_chain_source(source: dict[str, Any]) -> int:
    """Validate the judgement-call window length for the local mimic screen."""
    window_length = source.get("window_length")
    if isinstance(window_length, bool) or not isinstance(window_length, int) or window_length < 1:
        raise AdapterError(
            "filters.metric_sources.sequence_novelty.window_length must be a positive integer"
        )
    include_controls = source.get("include_positive_controls", True)
    if not isinstance(include_controls, bool):
        raise AdapterError(
            "filters.metric_sources.sequence_novelty.include_positive_controls must be boolean"
        )
    return window_length


def resolve_config_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise AdapterError(f"target-chain reference structure is missing: {path}")
    return path


def target_reference_structures(
    config: dict[str, Any], config_path: Path, include_positive_controls: bool
) -> list[tuple[str, Path]]:
    """Resolve target and enabled positive-control structures from the campaign."""
    references: list[tuple[str, Path]] = []
    targets = config.get("targets")
    if not isinstance(targets, list) or not targets:
        raise AdapterError("target-chain mimic screen requires at least one configured target")
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise AdapterError(f"targets[{index}] must be an object")
        target_id = target.get("target_id")
        path_value = target.get("structure_path")
        if not isinstance(target_id, str) or not target_id:
            raise AdapterError(f"targets[{index}].target_id must be a non-empty string")
        if not isinstance(path_value, str) or not path_value:
            raise AdapterError(f"targets[{index}].structure_path must be a non-empty string")
        references.append((f"target {target_id}", resolve_config_path(path_value, config_path)))
    if include_positive_controls:
        controls = config.get("controls")
        positive = controls.get("positive", []) if isinstance(controls, dict) else []
        if not isinstance(positive, list):
            raise AdapterError("controls.positive must be a list")
        for index, control in enumerate(positive):
            if not isinstance(control, dict) or control.get("enabled") is not True:
                continue
            control_id = control.get("id")
            path_value = control.get("structure_path")
            if not isinstance(control_id, str) or not control_id:
                raise AdapterError(f"controls.positive[{index}].id must be a non-empty string")
            if not isinstance(path_value, str) or not path_value:
                raise AdapterError(
                    f"controls.positive[{index}].structure_path must be a non-empty string"
                )
            references.append(
                (
                    f"positive control {control_id}",
                    resolve_config_path(path_value, config_path),
                )
            )
    return references


def pdb_chain_sequences(text: str) -> dict[str, str]:
    residues: dict[str, list[str]] = {}
    seen: set[tuple[str, str, str]] = set()
    for line in text.splitlines():
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        residue = AMINO_ACID_3_TO_1.get(line[17:20].strip().upper())
        if residue is None:
            continue
        chain = line[21:22].strip() or "_"
        residue_id = line[22:26].strip()
        insertion_code = line[26:27].strip()
        key = (chain, residue_id, insertion_code)
        if key in seen:
            continue
        seen.add(key)
        residues.setdefault(chain, []).append(residue)
    return {chain: "".join(values) for chain, values in residues.items() if values}


def mmcif_chain_sequences(text: str) -> dict[str, str]:
    lines = text.splitlines()
    residues: dict[str, list[str]] = {}
    seen: set[tuple[str, str, str]] = set()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        columns: list[str] = []
        while index < len(lines) and lines[index].strip().startswith("_"):
            columns.append(lines[index].strip())
            index += 1
        if not columns or not columns[0].startswith("_atom_site."):
            continue
        column_index = {name: position for position, name in enumerate(columns)}
        required = ("_atom_site.group_PDB", "_atom_site.label_atom_id", "_atom_site.label_comp_id")
        if any(name not in column_index for name in required):
            raise AdapterError("mmCIF atom_site loop lacks group, atom, or residue columns")
        chain_column = (
            "_atom_site.auth_asym_id"
            if "_atom_site.auth_asym_id" in column_index
            else "_atom_site.label_asym_id"
        )
        sequence_column = (
            "_atom_site.auth_seq_id"
            if "_atom_site.auth_seq_id" in column_index
            else "_atom_site.label_seq_id"
        )
        if chain_column not in column_index or sequence_column not in column_index:
            raise AdapterError("mmCIF atom_site loop lacks chain or residue identifiers")
        insertion_column = "_atom_site.pdbx_PDB_ins_code"
        model_column = "_atom_site.pdbx_PDB_model_num"
        while index < len(lines):
            stripped = lines[index].strip()
            # `END` and `ENDMDL` are PDB terminators, and writers that emit mmCIF
            # from a PDB pipeline append them after the atom_site loop. The shipped
            # supplied-candidate structures carry one. Without
            # them in this set the terminator parses as a one-field row and the
            # whole file is refused, even though every other reader accepts it.
            if (
                not stripped
                or stripped.startswith("#")
                or stripped == "loop_"
                or stripped.startswith("_")
                or stripped in {"END", "ENDMDL"}
            ):
                break
            fields = shlex.split(stripped)
            if len(fields) != len(columns):
                raise AdapterError(
                    f"mmCIF atom_site row has {len(fields)} fields; expected {len(columns)}"
                )
            index += 1
            if fields[column_index["_atom_site.group_PDB"]] != "ATOM":
                continue
            if fields[column_index["_atom_site.label_atom_id"]] != "CA":
                continue
            if model_column in column_index and fields[column_index[model_column]] not in {"1", ".", "?"}:
                continue
            residue = AMINO_ACID_3_TO_1.get(
                fields[column_index["_atom_site.label_comp_id"]].upper()
            )
            if residue is None:
                continue
            chain = fields[column_index[chain_column]]
            residue_id = fields[column_index[sequence_column]]
            insertion = (
                fields[column_index[insertion_column]]
                if insertion_column in column_index
                else ""
            )
            key = (chain, residue_id, insertion)
            if key in seen:
                continue
            seen.add(key)
            residues.setdefault(chain, []).append(residue)
        return {chain: "".join(values) for chain, values in residues.items() if values}
    return {}


def structure_chain_sequences(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    sequences = mmcif_chain_sequences(text) if "_atom_site." in text else pdb_chain_sequences(text)
    if not sequences:
        raise AdapterError(f"target-chain reference structure has no protein chains: {path}")
    return sequences


def best_ungapped_window_match(
    query: str, references: list[dict[str, Any]], configured_window_length: int
) -> dict[str, Any]:
    """Return the highest-identity equal-length window across all references.

    The reported identity is `matches / window_length`, so it says nothing
    about how much of the binder the window covers. A 100-residue binder
    sharing an 8-residue stretch at 75 percent identity reports 0.75 and trips
    a 0.60 maximum gate on 8 percent of its length. The window length is a
    judgement call taken from configuration and `validate_target_chain_source`
    accepts any positive integer, so the observation records `query_length` and
    `coverage` to make the denominator visible.

    TODO The gate compares identity alone. The published sequence-novelty rule
    quoted in PUBLISHED_THRESHOLD_GUIDANCE pairs an identity threshold with a
    coverage threshold, and no coverage threshold is stated for this screen.
    Decide one and gate on both, or state why identity alone is right here.
    """
    best: dict[str, Any] | None = None
    for reference in references:
        sequence = reference["sequence"]
        window_length = min(configured_window_length, len(query), len(sequence))
        if window_length < 1:
            continue
        for query_start in range(len(query) - window_length + 1):
            query_window = query[query_start : query_start + window_length]
            for reference_start in range(len(sequence) - window_length + 1):
                reference_window = sequence[
                    reference_start : reference_start + window_length
                ]
                matches = sum(
                    query_residue == reference_residue
                    for query_residue, reference_residue in zip(
                        query_window, reference_window
                    )
                )
                identity = matches / window_length
                if best is None or identity > best["identity"]:
                    best = {
                        **reference,
                        "identity": identity,
                        "window_length": window_length,
                        "query_length": len(query),
                        "coverage": window_length / len(query),
                        "query_start": query_start,
                        "reference_start": reference_start,
                    }
    if best is None:
        raise AdapterError("target-chain mimic screen found no comparable protein residues")
    return best


def target_chain_metric_values(
    config: dict[str, Any],
    config_path: Path,
    candidate_sequences: dict[str, str],
    source: dict[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, Any]], str]:
    window_length = validate_target_chain_source(source)
    structures = target_reference_structures(
        config,
        config_path,
        source.get("include_positive_controls", True),
    )
    references: list[dict[str, Any]] = []
    digest_records: list[dict[str, str]] = []
    for label, path in structures:
        file_digest = sha256_file(path)
        digest_records.append({"label": label, "sha256": file_digest})
        for chain, sequence in structure_chain_sequences(path).items():
            references.append(
                {"source_label": label, "source_path": str(path), "chain": chain, "sequence": sequence}
            )
    reference_digest = hashlib.sha256(
        json.dumps(digest_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    matches = {
        candidate_id: best_ungapped_window_match(sequence, references, window_length)
        for candidate_id, sequence in candidate_sequences.items()
    }
    values = {candidate_id: match["identity"] for candidate_id, match in matches.items()}
    return values, matches, reference_digest


def target_chain_match_reason(
    match: dict[str, Any], value: float, threshold: float, passed: bool
) -> str:
    verdict = "passes" if passed else "fails"
    return (
        f"best target-chain match {value:.3f} identity over "
        f"{match['window_length']} residues to {match['source_label']} chain "
        f"{match['chain']} (design {match['query_start'] + 1}-"
        f"{match['query_start'] + match['window_length']}, reference "
        f"{match['reference_start'] + 1}-"
        f"{match['reference_start'] + match['window_length']}) {verdict} "
        f"maximum threshold {threshold:.3f}"
    )


def structural_reference_backbones(
    config: dict[str, Any], config_path: Path
) -> tuple[list[dict[str, Any]], str]:
    """Load every configured target and enabled control chain for novelty screening."""
    references: list[tuple[str, Path]] = []
    targets = config.get("targets")
    if not isinstance(targets, list) or not targets:
        raise AdapterError("structure novelty requires at least one configured target")
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise AdapterError(f"targets[{index}] must be an object")
        target_id = target.get("target_id")
        path_value = target.get("structure_path")
        if not isinstance(target_id, str) or not target_id:
            raise AdapterError(f"targets[{index}].target_id must be a non-empty string")
        if not isinstance(path_value, str) or not path_value:
            raise AdapterError(f"targets[{index}].structure_path must be a non-empty string")
        references.append((f"target {target_id}", resolve_config_path(path_value, config_path)))
    controls = config.get("controls")
    if isinstance(controls, dict):
        for group_name, raw_controls in controls.items():
            if not isinstance(raw_controls, list):
                continue
            for index, control in enumerate(raw_controls):
                if not isinstance(control, dict) or control.get("enabled") is not True:
                    continue
                control_id = control.get("id")
                path_value = control.get("structure_path")
                if not isinstance(control_id, str) or not control_id:
                    raise AdapterError(f"controls.{group_name}[{index}].id must be a non-empty string")
                if not isinstance(path_value, str) or not path_value:
                    raise AdapterError(
                        f"controls.{group_name}[{index}].structure_path must be a non-empty string"
                    )
                references.append(
                    (
                        f"control {control_id}",
                        resolve_config_path(path_value, config_path),
                    )
                )
    loaded: list[dict[str, Any]] = []
    digest_records: list[dict[str, str]] = []
    for source_label, path in references:
        digest_records.append({"label": source_label, "sha256": sha256_file(path)})
        try:
            chains = structural_surrogates.parse_backbones(path)
        except structural_surrogates.StructuralSurrogateError as exc:
            raise AdapterError(f"structure novelty cannot parse {source_label}: {exc}") from exc
        for chain_id, residues in chains.items():
            try:
                structural_surrogates.residue_sequence(residues)
                structural_surrogates.ca_contact_map(residues)
            except structural_surrogates.StructuralSurrogateError as exc:
                raise AdapterError(
                    f"structure novelty cannot use {source_label} chain {chain_id}: {exc}"
                ) from exc
            loaded.append(
                {
                    "source_label": source_label,
                    "source_path": str(path),
                    "chain": chain_id,
                    "residues": residues,
                }
            )
    if not loaded:
        raise AdapterError("structure novelty found no protein reference chains")
    digest = hashlib.sha256(
        json.dumps(digest_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return loaded, digest


def candidate_design_backbones(
    rows: list[dict[str, Any]], candidate_sequences: dict[str, str]
) -> dict[str, tuple[structural_surrogates.BackboneResidue, ...]]:
    """Load each candidate's unique PDB or mmCIF chain matching its FASTA sequence."""
    backbones: dict[str, tuple[structural_surrogates.BackboneResidue, ...]] = {}
    for row in rows:
        candidate_id = row["candidate_id"]
        path_value = row.get("design_pose_path")
        if not isinstance(path_value, str) or not path_value:
            raise AdapterError(f"candidate {candidate_id} is missing design_pose_path for structural surrogates")
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise AdapterError(f"candidate {candidate_id} design backbone is missing: {path}")
        try:
            _, residues = structural_surrogates.choose_design_chain(
                structural_surrogates.parse_backbones(path), candidate_sequences[candidate_id]
            )
            structural_surrogates.ca_contact_map(residues)
        except structural_surrogates.StructuralSurrogateError as exc:
            raise AdapterError(
                f"candidate {candidate_id} cannot support structural surrogates: {exc}"
            ) from exc
        backbones[candidate_id] = residues
    return backbones


def contact_map_overlap_metric_values(
    config: dict[str, Any],
    config_path: Path,
    rows: list[dict[str, Any]],
    candidate_sequences: dict[str, str],
    source: dict[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, Any]], str]:
    """Return each design's highest overlap against target and control chains."""
    parameters = _source_alignment(source, "filters.metric_sources.structure_novelty")
    references, reference_digest = structural_reference_backbones(config, config_path)
    designs = candidate_design_backbones(rows, candidate_sequences)
    values: dict[str, float] = {}
    matches: dict[str, dict[str, Any]] = {}
    for candidate_id, design in designs.items():
        scored: list[tuple[structural_surrogates.ContactMapOverlap, dict[str, Any]]] = []
        for reference in references:
            try:
                result = structural_surrogates.contact_map_overlap(
                    design, reference["residues"], parameters
                )
            except structural_surrogates.StructuralSurrogateError as exc:
                raise AdapterError(
                    f"candidate {candidate_id} cannot compare with {reference['source_label']} "
                    f"chain {reference['chain']}: {exc}"
                ) from exc
            scored.append((result, reference))
        result, reference = max(
            scored,
            key=lambda item: (
                item[0].f1,
                item[0].shared_contact_count,
                item[1]["source_label"],
                item[1]["chain"],
            ),
        )
        values[candidate_id] = result.f1
        matches[candidate_id] = {
            **reference,
            "f1": result.f1,
            "register_shift": result.register_shift,
            "shared_contact_count": result.shared_contact_count,
            "design_contact_count": result.design_contact_count,
            "reference_contact_count": result.reference_contact_count,
            "empty_map": result.empty_map,
        }
    return values, matches, reference_digest


def contact_map_overlap_match_reason(
    match: dict[str, Any], value: float, threshold: float, passed: bool
) -> str:
    verdict = "passes" if passed else "fails"
    return (
        f"{structural_surrogates.CONTACT_MAP_OVERLAP_LABEL}; maximum F1 {value:.3f} against "
        f"{match['source_label']} chain {match['chain']} with register shift "
        f"{match['register_shift']} {verdict} maximum threshold {threshold:.3f}. "
        f"{structural_surrogates.SURROGATE_DISCLOSURE}"
    )


def secondary_structure_metric_values(
    rows: list[dict[str, Any]], candidate_sequences: dict[str, str]
) -> tuple[dict[str, float], dict[str, structural_surrogates.SecondaryStructureAssignment]]:
    """Return Kabsch-Sander helix fractions for an explicitly configured filter."""
    designs = candidate_design_backbones(rows, candidate_sequences)
    values: dict[str, float] = {}
    assignments: dict[str, structural_surrogates.SecondaryStructureAssignment] = {}
    for candidate_id, residues in designs.items():
        try:
            assignment = structural_surrogates.kabsch_sander_assignment(residues)
        except structural_surrogates.StructuralSurrogateError as exc:
            raise AdapterError(
                f"candidate {candidate_id} cannot support Kabsch-Sander assignment: {exc}"
            ) from exc
        assignments[candidate_id] = assignment
        values[candidate_id] = assignment.helix_fraction
    return values, assignments


def structural_diversity_settings(config: dict[str, Any]) -> dict[str, Any] | None:
    """Read the calibrated structural-diversity settings, or return disabled."""
    selection = config.get("selection")
    if not isinstance(selection, dict):
        return None
    raw = selection.get(STRUCTURAL_DIVERSITY_CONFIG_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise AdapterError(f"selection.{STRUCTURAL_DIVERSITY_CONFIG_KEY} must be an object")
    enabled = raw.get("enabled")
    if not isinstance(enabled, bool):
        raise AdapterError(f"selection.{STRUCTURAL_DIVERSITY_CONFIG_KEY}.enabled must be boolean")
    if not enabled:
        return None
    label = f"selection.{STRUCTURAL_DIVERSITY_CONFIG_KEY}"
    try:
        alignment = structural_surrogates.alignment_parameters_from_mapping(
            raw.get("alignment"), f"{label}.alignment"
        )
    except structural_surrogates.StructuralSurrogateError as exc:
        raise AdapterError(str(exc)) from exc
    return {
        "alignment": alignment,
        "contact_map_overlap_cutoff": _finite_number(
            raw.get("contact_map_overlap_cutoff"), f"{label}.contact_map_overlap_cutoff"
        ),
        "all_alpha_fraction_threshold": _finite_number(
            raw.get("all_alpha_fraction_threshold"), f"{label}.all_alpha_fraction_threshold"
        ),
        "minimum_non_all_alpha_fraction": _finite_number(
            raw.get("minimum_non_all_alpha_fraction"), f"{label}.minimum_non_all_alpha_fraction"
        ),
    }


def structural_diversity_annotations(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    candidate_sequences: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Compute cluster identifiers and all-alpha verdicts for promotion selection."""
    settings = structural_diversity_settings(config)
    if settings is None:
        return {}
    designs = candidate_design_backbones(rows, candidate_sequences)
    try:
        clusters = structural_surrogates.cluster_contact_map_overlap(
            designs, settings["alignment"], settings["contact_map_overlap_cutoff"]
        )
    except structural_surrogates.StructuralSurrogateError as exc:
        raise AdapterError(f"structural diversity clustering cannot run: {exc}") from exc
    annotations: dict[str, dict[str, Any]] = {}
    for candidate_id, residues in designs.items():
        try:
            assignment = structural_surrogates.kabsch_sander_assignment(residues)
            all_alpha = structural_surrogates.is_all_alpha(
                assignment, settings["all_alpha_fraction_threshold"]
            )
        except structural_surrogates.StructuralSurrogateError as exc:
            raise AdapterError(f"candidate {candidate_id} cannot support structural diversity: {exc}") from exc
        annotations[candidate_id] = {
            "tm90_cluster_id": clusters.identifiers[candidate_id],
            STRUCTURAL_DIVERSITY_CLUSTER_FIELD: structural_surrogates.CLUSTERING_LABEL,
            "tm90_cluster_cutoff_surrogate": clusters.cutoff,
            STRUCTURAL_DIVERSITY_SECONDARY_FIELD: structural_surrogates.SECONDARY_STRUCTURE_LABEL,
            "helix_fraction_surrogate": assignment.helix_fraction,
            "all_alpha_surrogate": all_alpha,
            "surrogate_disclosure": structural_surrogates.SURROGATE_DISCLOSURE,
        }
    return annotations


def integrity_passing_ids(args: argparse.Namespace, round_number: int) -> set[str]:
    path = integrity_passing_manifest_path(args, round_number)
    rows = read_jsonl(path, "integrity passing manifest")
    passing: set[str] = set()
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise AdapterError(f"integrity passing manifest has a row without candidate_id: {path}")
        if candidate_id in passing:
            raise AdapterError(f"integrity passing manifest repeats candidate_id: {candidate_id}")
        if row.get("filter_pass") is not True or row.get("failed_checks") != []:
            raise AdapterError(f"integrity passing candidate {candidate_id} has inconsistent pass fields")
        passing.add(candidate_id)
    return passing


def run_stage(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = read_json_object(config_path, "campaign config")
    expected_round = stage_round(args.stage, config)
    selected_contracts = novelty_contracts(config)
    source_paths = metric_source_paths(
        config,
        config_path,
        selected_contracts,
        plan_path=args.plan,
        receipts_dir=args.receipts_dir,
        metric_overrides={
            item.split("=", 1)[0]: Path(item.split("=", 1)[1])
            for item in (args.metric_source or [])
        },
    )
    manifest_path = integrity_passing_manifest_path(args, expected_round)
    rows, sequences = load_candidates(manifest_path, expected_round)
    for row in rows:
        candidate_id = row["candidate_id"]
        if row.get("filter_pass") is not True or row.get("failed_checks") != []:
            raise AdapterError(
                f"integrity passing manifest row {candidate_id} is not a passing candidate"
            )
    candidate_ids = {row["candidate_id"] for row in rows}
    raw_sources = config["filters"]["metric_sources"] if selected_contracts else {}
    values_by_filter: dict[str, dict[str, float]] = {}
    local_matches: dict[str, dict[str, Any]] = {}
    local_reference_digests: dict[str, str] = {}
    contact_map_matches: dict[str, dict[str, Any]] = {}
    contact_map_reference_digests: dict[str, str] = {}
    secondary_assignments: dict[str, structural_surrogates.SecondaryStructureAssignment] = {}
    for contract in selected_contracts:
        filter_id = contract["filter_id"]
        source = raw_sources[filter_id]
        if source.get("kind") == TARGET_CHAIN_SOURCE_KIND:
            values, matches, reference_digest = target_chain_metric_values(
                config, config_path, sequences, source
            )
            values_by_filter[filter_id] = values
            local_matches.update(matches)
            local_reference_digests[filter_id] = reference_digest
        elif source.get("kind") == CONTACT_MAP_OVERLAP_SOURCE_KIND:
            values, matches, reference_digest = contact_map_overlap_metric_values(
                config, config_path, rows, sequences, source
            )
            values_by_filter[filter_id] = values
            contact_map_matches.update(matches)
            contact_map_reference_digests[filter_id] = reference_digest
        elif source.get("kind") == KABSCH_SANDER_SOURCE_KIND:
            values, assignments = secondary_structure_metric_values(rows, sequences)
            values_by_filter[filter_id] = values
            secondary_assignments.update(assignments)
        else:
            values_by_filter[filter_id] = metric_values(source_paths[filter_id], contract)
    diversity_annotations = structural_diversity_annotations(config, rows, sequences)
    observations: list[dict[str, Any]] = []
    passed_by_candidate: dict[str, set[str]] = {candidate_id: set() for candidate_id in candidate_ids}
    for row in rows:
        candidate_id = row["candidate_id"]
        for contract in selected_contracts:
            filter_id = contract["filter_id"]
            values = values_by_filter[filter_id]
            if candidate_id not in values:
                source_description = (
                    TARGET_CHAIN_SOURCE_KIND
                    if filter_id in local_reference_digests
                    else CONTACT_MAP_OVERLAP_SOURCE_KIND
                    if filter_id in contact_map_reference_digests
                    else str(source_paths[filter_id])
                )
                raise AdapterError(
                    f"metric source for {filter_id} has no value for candidate {candidate_id}; "
                    f"source={source_description}"
                )
            value = values[candidate_id]
            # An unrecognised operator used to fall through to the maximum
            # branch, so a single typo silently inverted the filter.
            if contract["operator"] not in ("minimum", "maximum"):
                raise AdapterError(
                    f"filter {filter_id} has operator {contract['operator']!r}; "
                    f"it must be one of minimum, maximum"
                )
            passed = value >= float(contract["threshold"]) if contract["operator"] == "minimum" else value <= float(contract["threshold"])
            if filter_id in local_reference_digests:
                match = local_matches[candidate_id]
                reason = target_chain_match_reason(
                    match,
                    value,
                    float(contract["threshold"]),
                    passed,
                )
            elif filter_id in contact_map_reference_digests:
                match = contact_map_matches[candidate_id]
                reason = contact_map_overlap_match_reason(
                    match,
                    value,
                    float(contract["threshold"]),
                    passed,
                )
            elif filter_id == "secondary_structure" and candidate_id in secondary_assignments:
                reason = (
                    f"{structural_surrogates.SECONDARY_STRUCTURE_LABEL}; helix fraction {value:.3f} "
                    f"{'satisfies' if passed else 'fails'} {contract['operator']} threshold "
                    f"{float(contract['threshold']):.3f}. "
                    f"{structural_surrogates.SURROGATE_DISCLOSURE}"
                )
            else:
                reason = (
                    f"configured {contract['metric']} value {value:g} satisfies "
                    f"{contract['operator']} threshold {float(contract['threshold']):g}"
                    if passed
                    else f"configured {contract['metric']} value {value:g} fails "
                    f"{contract['operator']} threshold {float(contract['threshold']):g}"
                )
            result = observation(row, contract, value, passed, expected_round, reason)
            if filter_id in local_reference_digests:
                result["reference_revision"] = "target-chain-structures-v1"
                result["reference_sha256"] = local_reference_digests[filter_id]
            elif filter_id in contact_map_reference_digests:
                result["reference_revision"] = "configured-target-and-control-structures"
                result["reference_sha256"] = contact_map_reference_digests[filter_id]
                result["surrogate"] = structural_surrogates.CONTACT_MAP_OVERLAP_LABEL
                result["surrogate_disclosure"] = structural_surrogates.SURROGATE_DISCLOSURE
            elif filter_id == "secondary_structure" and candidate_id in secondary_assignments:
                result["surrogate"] = structural_surrogates.SECONDARY_STRUCTURE_LABEL
                result["surrogate_disclosure"] = structural_surrogates.SURROGATE_DISCLOSURE
            observations.append(result)
            if passed:
                passed_by_candidate[candidate_id].add(filter_id)
    required_ids = {contract["filter_id"] for contract in selected_contracts}
    passing_rows = [
        {
            **row,
            **diversity_annotations.get(row["candidate_id"], {}),
            "filter_pass": True,
            "failed_checks": [],
        }
        for row in rows
        if passed_by_candidate[row["candidate_id"]] == required_ids
    ]
    phase_dir = (args.attempt_dir / args.phase).resolve()
    if observations:
        write_jsonl(phase_dir / "filter-observations.jsonl", observations)
    write_jsonl(phase_dir / "passing-candidates.jsonl", passing_rows)
    write_json(
        phase_dir / "filter-report.json",
        filter_report(
            STAGE_NOVELTY,
            [row["candidate_id"] for row in rows],
            selected_contracts,
            observations,
            [row["candidate_id"] for row in passing_rows],
            skipped_gates=retired_filter_statuses(),
        ),
    )
    if not passing_rows:
        raise AdapterError("no candidate passed the configured novelty checks after integrity filtering")
    print(
        f"novelty filter: phase={args.phase} candidates={len(rows)} "
        f"passing={len(passing_rows)} observations={len(observations)}"
    )
    return 0


def parse_stage(args: argparse.Namespace) -> int:
    attempt_dir = args.attempt_dir.resolve()
    phase_dir = (attempt_dir / args.phase).resolve()
    result_path = phase_dir / "parser-result.json"
    errors: list[str] = []
    default_paths = False
    if args.jsonl:
        patterns = args.jsonl
    elif novelty_contracts(read_json_object(args.config.resolve(), "campaign config")):
        default_paths = True
        patterns = [
            str(phase_dir / "filter-observations.jsonl"),
            str(phase_dir / "passing-candidates.jsonl"),
        ]
    else:
        default_paths = True
        patterns = [str(phase_dir / "passing-candidates.jsonl")]
    files: list[Path] = []
    for pattern in patterns:
        raw = Path(pattern)
        resolved_pattern = (attempt_dir / raw if not raw.is_absolute() else raw).resolve()
        if attempt_dir not in resolved_pattern.parents:
            errors.append(f"declared output escapes the attempt directory: {pattern}")
            continue
        matches = sorted(Path(value) for value in glob.glob(str(resolved_pattern), recursive=True))
        if not matches:
            errors.append(f"declared output matched no files: {pattern}")
            continue
        files.extend(path for path in matches if path.is_file())
    parsed_count = 0
    for path in files:
        if not path.is_file():
            errors.append(f"declared output is missing: {path}")
            continue
        try:
            parsed_count += len(read_jsonl(path, "filter output"))
        except AdapterError as exc:
            errors.append(str(exc))
    if default_paths:
        report_path = phase_dir / "filter-report.json"
        try:
            read_json_object(report_path, "filter report")
            parsed_count += 1
            files.append(report_path)
        except AdapterError as exc:
            errors.append(str(exc))
    hashes = [sha256_file(path) for path in files if path.is_file()]
    result = {
        "ok": bool(files) and not errors,
        "parsed_count": parsed_count,
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": sorted(hashes),
    }
    write_json(result_path, result)
    for error in errors:
        print(f"novelty filter parser: {error}", file=sys.stderr)
    print(f"novelty filter parser: phase={args.phase} parsed_count={parsed_count} ok={result['ok']}")
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("toolcheck")
    for name in ("run", "parse"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument(
            "--stage",
            default=STAGE_NOVELTY if name == "run" else None,
            required=name == "parse",
        )
        subparser.add_argument("--phase", required=True)
        subparser.add_argument("--count", type=int, default=1)
        subparser.add_argument("--attempt-dir", type=Path, required=True)
        subparser.add_argument("--receipts-dir", type=Path, required=True)
        subparser.add_argument("--artifact-root", type=Path, required=True)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--plan", type=Path, required=True)
        subparser.add_argument(
            "--integrity-passing-manifest",
            type=Path,
            default=None,
            help=(
                "Explicit integrity-passing manifest override. Defaults to the declared "
                "filter-integrity input receipt."
            ),
        )
        if name == "run":
            subparser.add_argument(
                "--metric-source",
                action="append",
                default=None,
                metavar="FILTER_ID=PATH",
                help="Explicit metric source override. Repeat for multiple filter ids.",
            )
        if name == "parse":
            subparser.add_argument("--jsonl", action="append")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "toolcheck":
            print("novelty filter ok, local target-chain and configured metric sources supported")
            return 0
        return run_stage(args) if args.command == "run" else parse_stage(args)
    except AdapterError as exc:
        print(f"novelty filter: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"novelty filter: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
