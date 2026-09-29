#!/usr/bin/env python3
"""Build a self-contained HTML report from a completed protein binder design campaign.

This module takes a campaign output directory, parses manifest and score data,
renders structure backbone figures with the pure-Python raster renderer, and
writes a single self-contained index.html with embedded data URIs.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import glob
import html
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from claude_binder.adapters.python_raster_renderer import (
        BackboneAtom,
        RasterRenderError,
        read_backbone,
        render_backbone_png,
    )
except ImportError:
    from ..adapters.python_raster_renderer import (
        BackboneAtom,
        RasterRenderError,
        read_backbone,
        render_backbone_png,
    )

# One definition each, so the report cannot drift from what the figures and the
# viewer draw. `resolved_chain_ids` lives with the viewer producer that resolves
# chain roles, and `_retarget_hotspots` with the picture renderer that uses it.
try:
    from claude_binder.data.helpers.viewer.make_viewer_scripts import (
        resolved_chain_ids as _resolved_chain_ids,
    )
    from claude_binder.adapters.structure_picture_renderer import _retarget_hotspots
except ImportError:
    from ..data.helpers.viewer.make_viewer_scripts import (
        resolved_chain_ids as _resolved_chain_ids,
    )
    from ..adapters.structure_picture_renderer import _retarget_hotspots


RESIDUE_TOKEN = re.compile(
    r"^(?:(?P<chain>[A-Za-z0-9_]+):)?(?P<start>-?\d+)(?:-(?P<end>-?\d+))?$"
)
OVERVIEW_STRIP = (30, 64, 175)
INTERFACE_STRIP = (177, 52, 50)

DONE_REPORT_NAME = "REPORT.md"
DONE_RECEIPT_NAME = "verification-receipt.json"
DONE_MANIFEST_NAME = "done-manifest.json"
DONE_REFUSAL_EXIT = 2
PRIMARY_METRIC_FLOOR = 0.0
ALIGNED_ERROR_CUTOFF_ANGSTROM = 10.0

_DONE_CONFIG_NAMES = (
    "done-report.json",
    "done_report.json",
    "final-report.json",
    "final_report.json",
)
_DONE_ARTIFACT_NAMES = {DONE_REPORT_NAME, DONE_RECEIPT_NAME, DONE_MANIFEST_NAME}
_DONE_ARTIFACT_ROLES = (
    "target_fasta",
    "site_definition",
    "backbones",
    "designs_fasta",
    "scores_csv",
    "deviations",
    "run_manifest",
    "provenance",
    "complex",
    "figures",
)
_DONE_MARKER = re.compile(r"<!-- done-stage-metrics: (?P<payload>{.*}) -->")
_EFFECT_SIZE_TERMS = (
    "effect size",
    "cohen's d",
    "cohens d",
    "hedges g",
    "standardized mean difference",
    "p-value",
)


@dataclass(frozen=True)
class DoneScore:
    """One score row read from the completed-run score table."""

    candidate_id: str
    primary_score: float | None
    values: Mapping[str, str]


@dataclass(frozen=True)
class DoneRefusal:
    """A condition that withholds the final ranking."""

    code: str
    message: str


@dataclass(frozen=True)
class DoneReportResult:
    """The files and publication state written by the final report stage."""

    report_path: Path
    receipt_path: Path
    manifest_path: Path
    refusal_codes: tuple[str, ...]

    @property
    def published(self) -> bool:
        return not self.refusal_codes


@dataclass(frozen=True)
class DoneVerification:
    """The mechanical checks attached to a completed-run folder."""

    checks: tuple[Mapping[str, str], ...]

    @property
    def ok(self) -> bool:
        return all(check["status"] == "pass" for check in self.checks)


@dataclass
class CandidateRecord:
    rank: int | None
    candidate_id: str
    sequence: str
    sequence_length: int
    arm_scores: dict[str, float | None]
    combined_rank_score: float | None
    is_control: bool = False
    control_role: str | None = None
    structure_path: Path | None = None
    # The chain roles of `structure_path`, not of the campaign. They stay unset
    # until a record names this file, because a default pair is right on one
    # provider and paints the binder as the target on another.
    binder_chain: str | None = None
    target_chain: str | None = None
    declared_target_chain: str | None = None


@dataclass
class HeaderMetadata:
    target_name: str | None
    uniprot_accession: str | None
    pdb_entry: str | None
    hotspot_residues: list[str] | None
    n_designs: int | None
    rounds: int | None
    tools_and_revisions: list[dict[str, str]] | None
    total_spend: str | None


def _format_todo(field_name: str, expected_source: str) -> str:
    """Return a visible TODO string explaining what is missing and its source."""
    return f"TODO: {field_name} missing from campaign manifest; expected from {expected_source}"


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            val = json.loads(stripped)
            if isinstance(val, dict):
                records.append(val)
        except json.JSONDecodeError:
            continue
    return records


def _find_file(root: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
        artifact = root / "artifacts" / name
        if artifact.is_file():
            return artifact
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _parse_fasta(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    sequences: dict[str, str] = {}
    current_id: str | None = None
    lines: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id:
                sequences[current_id] = "".join(lines)
            current_id = line[1:].split()[0]
            lines = []
        else:
            lines.append(line)
    if current_id:
        sequences[current_id] = "".join(lines)
    return sequences


def _parse_manifest_header(root: Path) -> HeaderMetadata:
    manifest_file = _find_file(
        root,
        [
            "run_manifest.json",
            "run-manifest.json",
            "manifest.json",
            "campaign.resolved.json",
            "runtime-config.resolved.json",
            "config.resolved.json",
            "campaign.json",
            "config.json",
        ],
    )
    manifest = _load_json(manifest_file) if manifest_file else {}
    if not isinstance(manifest, dict):
        manifest = {}

    config_file = _find_file(
        root,
        [
            "runtime-config.resolved.json",
            "config.resolved.json",
            "campaign.resolved.json",
            "config.json",
            "campaign.json",
        ],
    )
    config = _load_json(config_file) if config_file else {}
    if not isinstance(config, dict):
        config = {}

    target_name = (
        manifest.get("target_name")
        or manifest.get("target_id")
        or manifest.get("primary_target_id")
    )
    if not target_name and "targets" in config and isinstance(config["targets"], list) and config["targets"]:
        target_name = config["targets"][0].get("target_id") or config["targets"][0].get("name")
    if not target_name and "target" in config and isinstance(config["target"], dict):
        target_name = config["target"].get("name") or config["target"].get("target_id")

    uniprot_accession = (
        manifest.get("uniprot_accession")
        or manifest.get("uniprot_id")
        or manifest.get("accession")
    )
    if not uniprot_accession and "targets" in config and isinstance(config["targets"], list) and config["targets"]:
        uniprot_accession = config["targets"][0].get("uniprot_accession") or config["targets"][0].get("uniprot_id")
    if not uniprot_accession and "target" in config and isinstance(config["target"], dict):
        uniprot_accession = config["target"].get("uniprot_accession") or config["target"].get("uniprot_id")

    pdb_entry = (
        manifest.get("pdb_entry")
        or manifest.get("pdb_id")
        or manifest.get("structure_id")
    )
    if not pdb_entry and "targets" in config and isinstance(config["targets"], list) and config["targets"]:
        pdb_entry = config["targets"][0].get("structure_id") or config["targets"][0].get("pdb_id")
    if not pdb_entry and "target" in config and isinstance(config["target"], dict):
        pdb_entry = config["target"].get("structure_id") or config["target"].get("pdb_id")

    hotspots: list[str] | None = None
    raw_hotspots = (
        manifest.get("hotspot_residues")
        or manifest.get("site_residues")
    )
    if not raw_hotspots and "targets" in config and isinstance(config["targets"], list) and config["targets"]:
        site = config["targets"][0].get("site")
        if isinstance(site, dict):
            raw_hotspots = site.get("design_residues") or site.get("hotspots")
    if not raw_hotspots and "site" in config and isinstance(config["site"], dict):
        raw_hotspots = config["site"].get("design_residues") or config["site"].get("hotspots")
    if isinstance(raw_hotspots, list):
        hotspots = [str(item) for item in raw_hotspots if item]

    n_designs = (
        manifest.get("n_designs")
        or manifest.get("num_designs")
        or manifest.get("num_proteins")
        or manifest.get("generated_count")
    )
    if n_designs is None:
        req_file = _find_file(root, ["request.json"])
        if req_file:
            req = _load_json(req_file)
            if isinstance(req, dict):
                n_designs = req.get("num_proteins") or req.get("n_designs")
    if n_designs is None and "request" in config and isinstance(config["request"], dict):
        n_designs = config["request"].get("num_proteins") or config["request"].get("n_designs")

    rounds = (
        manifest.get("rounds")
        or manifest.get("optimization_rounds")
        or manifest.get("num_rounds")
    )
    if rounds is None and "optimization" in config and isinstance(config["optimization"], dict):
        rounds = config["optimization"].get("rounds") or config["optimization"].get("num_rounds")

    tools_list: list[dict[str, str]] = []
    stages = manifest.get("stages") or config.get("stages") or manifest.get("adapters") or config.get("adapters")
    if isinstance(stages, list):
        for stage in stages:
            if isinstance(stage, dict):
                stage_id = str(stage.get("stage_id") or stage.get("id") or stage.get("name") or "stage")
                tool = str(stage.get("adapter_id") or stage.get("tool") or stage.get("name") or stage.get("adapter") or "")
                rev = str(stage.get("source_revision") or stage.get("model_revision") or stage.get("revision") or stage.get("version") or "")
                if tool or rev:
                    tools_list.append({"stage": stage_id, "tool": tool or "unspecified", "revision": rev or "unspecified"})
    elif isinstance(stages, dict):
        for stage_name, stage_val in stages.items():
            if isinstance(stage_val, dict):
                tool = str(stage_val.get("adapter_id") or stage_val.get("tool") or stage_val.get("name") or "")
                rev = str(stage_val.get("source_revision") or stage_val.get("model_revision") or stage_val.get("revision") or "")
                tools_list.append({"stage": str(stage_name), "tool": tool or "unspecified", "revision": rev or "unspecified"})
            elif isinstance(stage_val, str):
                tools_list.append({"stage": str(stage_name), "tool": stage_val, "revision": "unspecified"})

    spend_str: str | None = None
    if "total_spend" in manifest and manifest["total_spend"] is not None:
        spend_str = str(manifest["total_spend"])
    elif "spend" in manifest and manifest["spend"] is not None:
        spend_str = str(manifest["spend"])
    else:
        spend_file = _find_file(root, ["spend.jsonl", "spend.json"])
        if spend_file:
            if spend_file.suffix == ".jsonl":
                spend_rows = _load_jsonl(spend_file)
                if spend_rows:
                    total_amount = sum(float(r["amount"]) for r in spend_rows if "amount" in r and isinstance(r["amount"], (int, float)))
                    currency = next((r.get("currency") for r in spend_rows if r.get("currency")), "USD")
                    spend_str = f"${total_amount:.2f} {currency}"
            else:
                s_json = _load_json(spend_file)
                if isinstance(s_json, dict) and "total" in s_json:
                    spend_str = str(s_json["total"])

    return HeaderMetadata(
        target_name=str(target_name) if target_name else None,
        uniprot_accession=str(uniprot_accession) if uniprot_accession else None,
        pdb_entry=str(pdb_entry) if pdb_entry else None,
        hotspot_residues=hotspots,
        n_designs=int(n_designs) if n_designs is not None else None,
        rounds=int(rounds) if rounds is not None else None,
        tools_and_revisions=tools_list if tools_list else None,
        total_spend=spend_str,
    )


def _parse_deviations(root: Path) -> list[dict[str, str]] | None:
    deviation_file = _find_file(root, ["deviations.json", "deviation_log.json", "deviations.jsonl"])
    if deviation_file:
        data = _load_json(deviation_file)
        if isinstance(data, dict):
            return [{"tool": str(k), "reason": str(v)} for k, v in data.items()]
        if isinstance(data, list):
            items = []
            for entry in data:
                if isinstance(entry, dict):
                    tool = str(entry.get("tool") or entry.get("name") or entry.get("stage") or "Deviation")
                    reason = str(entry.get("reason") or entry.get("description") or entry.get("detail") or "")
                    items.append({"tool": tool, "reason": reason})
            return items

    config_file = _find_file(
        root,
        [
            "runtime-config.resolved.json",
            "config.resolved.json",
            "campaign.resolved.json",
            "config.json",
            "campaign.json",
        ],
    )
    if config_file:
        config = _load_json(config_file)
        if isinstance(config, dict):
            gap_notes = config.get("gap_notes")
            if isinstance(gap_notes, dict) and "deviations" in gap_notes and isinstance(gap_notes["deviations"], dict):
                return [{"tool": str(k), "reason": str(v)} for k, v in gap_notes["deviations"].items()]
            if "deviations" in config and isinstance(config["deviations"], dict):
                return [{"tool": str(k), "reason": str(v)} for k, v in config["deviations"].items()]

    return None


def _parse_controls(root: Path) -> list[CandidateRecord]:
    controls_file = _find_file(root, ["controls.json", "control_results.json", "control-results.json"])
    control_records: list[CandidateRecord] = []
    if not controls_file:
        return control_records

    data = _load_json(controls_file)
    if not isinstance(data, dict):
        return control_records

    results = data.get("control_results") or data.get("controls") or []
    if not isinstance(results, list):
        return control_records

    for item in results:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("control_id") or item.get("id") or item.get("name") or "control")
        role = str(item.get("control_role") or item.get("role") or item.get("type") or "control")
        score = item.get("mean_ipsae_min") or item.get("score") or item.get("ipsae_min")
        seq = str(item.get("sequence") or "")
        length = int(item.get("sequence_length") or (len(seq) if seq else 0))
        struct_p = item.get("structure_path") or item.get("complex_path")
        struct_path = Path(struct_p) if struct_p else None

        control_records.append(
            CandidateRecord(
                rank=None,
                candidate_id=cid,
                sequence=seq,
                sequence_length=length,
                arm_scores={"ipSAE_min": float(score) if score is not None else None},
                combined_rank_score=float(score) if score is not None else None,
                is_control=True,
                control_role=role,
                structure_path=struct_path,
            )
        )
    return control_records


_UNIFORM_OBSERVATION_NAMES = ("uniform-observations.jsonl", "uniform_observations.jsonl")


def _chain_text(value: Any) -> str | None:
    """Return a chain id only when it is a non-empty string."""
    return value if isinstance(value, str) and value else None


def _reference_chain_ids(record: Mapping[str, Any]) -> tuple[Any, Any]:
    """Return the (target, binder) chain ids the designed pose uses.

    The designed pose is a different file from the prediction and carries its
    own letters. The same both-or-neither rule applies, and the same fallback,
    because the designed pose is the structure the campaign supplied under the
    letters it declared.
    """
    reference_target = record.get("reference_target_chain_id")
    reference_binder = record.get("reference_binder_chain_id")
    if (
        isinstance(reference_target, str)
        and reference_target
        and isinstance(reference_binder, str)
        and reference_binder
    ):
        return reference_target, reference_binder
    return record.get("target_chain_id"), record.get("binder_chain_id")


def _parse_observations(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Index the scored observation rows by candidate id.

    The ranked portfolio names no chain at all, so the letters have to come from
    the rows that scored each prediction.
    """
    path = _find_file(root, list(_UNIFORM_OBSERVATION_NAMES))
    index: dict[str, list[dict[str, Any]]] = {}
    for row in _load_jsonl(path) if path else []:
        candidate_id = row.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            index.setdefault(candidate_id, []).append(row)
    return index


def _same_path(candidate: Any, structure_path: Path) -> bool:
    """True when a recorded path names the same file the report is drawing."""
    if not isinstance(candidate, str) or not candidate:
        return False
    try:
        return Path(candidate).expanduser().resolve() == structure_path.expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def _chain_roles_for_structure(
    structure_path: Path | None,
    shown: Any,
    observations: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str | None, str | None]:
    """Return (target, binder, declared target) chain ids for one structure file.

    A chain pair belongs to a file, not to a campaign. One observation row names
    two different structures, the predicted complex it scored and the designed
    pose it scored that prediction against, and each carries its own letters. On
    a provider that writes its own mmCIF the two disagree. A file that no record
    names leaves the pair unresolved, because a guessed pair draws the binder in
    the target's colour and reads as a rendered result rather than a missing one.

    The third value is the chain the campaign declared its site residues
    against, which is where a hotspot list has to be retargeted from.
    """
    if structure_path is None:
        return None, None, None
    if isinstance(shown, Mapping) and _same_path(shown.get("complex_path"), structure_path):
        target, binder = _resolved_chain_ids(shown)
        declared = _chain_text(shown.get("declared_target_chain_id")) or _chain_text(target)
        return _chain_text(target), _chain_text(binder), declared
    for row in observations:
        if _same_path(row.get("predicted_complex_path"), structure_path):
            target, binder = _resolved_chain_ids(row)
            return (
                _chain_text(target),
                _chain_text(binder),
                _chain_text(row.get("target_chain_id")),
            )
        if _same_path(row.get("design_pose_path"), structure_path):
            target, binder = _reference_chain_ids(row)
            return (
                _chain_text(target),
                _chain_text(binder),
                _chain_text(row.get("target_chain_id")),
            )
    return None, None, None


def _parse_candidates(root: Path) -> tuple[list[CandidateRecord], list[str]]:
    ranked_file = _find_file(
        root,
        [
            "ranked-candidates.json",
            "ranked_candidates.json",
            "ranked_candidates.tsv",
            "results.json",
        ],
    )
    fasta_file = _find_file(root, ["candidates.fasta", "sequences.fasta"])
    fasta_map = _parse_fasta(fasta_file) if fasta_file else {}
    observations = _parse_observations(root)

    candidates: list[CandidateRecord] = []
    arm_names_set: set[str] = set()

    if ranked_file and ranked_file.suffix == ".json":
        data = _load_json(ranked_file)
        if isinstance(data, dict):
            rows = data.get("ranked_candidates") or data.get("candidates") or data.get("designs") or []
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    cid = str(row.get("candidate_id") or row.get("id") or "")
                    if not cid:
                        continue
                    seq = str(row.get("sequence") or fasta_map.get(cid) or "")
                    seq_len = int(row.get("sequence_length") or (len(seq) if seq else 0))
                    rank_val = row.get("rank")
                    rank = int(rank_val) if isinstance(rank_val, int) else None

                    arms_dict: dict[str, float | None] = {}
                    if "mean_ipsae_min" in row and row["mean_ipsae_min"] is not None:
                        arms_dict["ipSAE_min"] = float(row["mean_ipsae_min"])
                        arm_names_set.add("ipSAE_min")
                    elif "ipsae_min_ensemble" in row and row["ipsae_min_ensemble"] is not None:
                        arms_dict["ipSAE_min"] = float(row["ipsae_min_ensemble"])
                        arm_names_set.add("ipSAE_min")

                    if "sc_dockq_ensemble" in row and row["sc_dockq_ensemble"] is not None:
                        arms_dict["sc_DockQ"] = float(row["sc_dockq_ensemble"])
                        arm_names_set.add("sc_DockQ")
                    elif "sc_dockq" in row and row["sc_dockq"] is not None:
                        arms_dict["sc_DockQ"] = float(row["sc_dockq"])
                        arm_names_set.add("sc_DockQ")

                    if "per_predictor" in row and isinstance(row["per_predictor"], dict):
                        for pred_name, pred_data in row["per_predictor"].items():
                            if isinstance(pred_data, dict):
                                metric_sum = pred_data.get("metric_summary") or pred_data
                                if isinstance(metric_sum, dict) and "ipsae_min" in metric_sum:
                                    val = metric_sum["ipsae_min"]
                                    score_val = val.get("mean") if isinstance(val, dict) else val
                                    if isinstance(score_val, (int, float)):
                                        arms_dict[str(pred_name)] = float(score_val)
                                        arm_names_set.add(str(pred_name))

                    comb_score = row.get("rank_score_central_estimate") or row.get("rank_score") or row.get("combined_score")
                    if comb_score is None and "ipSAE_min" in arms_dict:
                        comb_score = arms_dict["ipSAE_min"]

                    struct_path: Path | None = None

                    shown = row.get("shown")
                    if isinstance(shown, dict):
                        cp = shown.get("complex_path")
                        if cp:
                            struct_path = Path(cp)

                    if not struct_path:
                        dpp = row.get("design_pose_path") or row.get("complex_path")
                        if dpp:
                            struct_path = Path(dpp)

                    if not struct_path or not struct_path.is_file():
                        matching = sorted(root.rglob(f"{cid}.pdb")) + sorted(root.rglob(f"{cid}.cif"))
                        if matching:
                            struct_path = matching[0]

                    # Resolve the chain roles last, against whichever file the
                    # search settled on. Attaching them earlier would carry one
                    # structure's letters onto another structure's coordinates.
                    t_chain, b_chain, declared_t_chain = _chain_roles_for_structure(
                        struct_path, shown, observations.get(cid, [])
                    )

                    candidates.append(
                        CandidateRecord(
                            rank=rank,
                            candidate_id=cid,
                            sequence=seq,
                            sequence_length=seq_len,
                            arm_scores=arms_dict,
                            combined_rank_score=float(comb_score) if comb_score is not None else None,
                            structure_path=struct_path,
                            binder_chain=b_chain,
                            target_chain=t_chain,
                            declared_target_chain=declared_t_chain,
                        )
                    )

    if not arm_names_set:
        arm_names_set.add("ipSAE_min")

    arm_names = sorted(arm_names_set)
    return candidates, arm_names


def _interface_focus(
    atoms: list[BackboneAtom],
    *,
    binder_chain: str,
    target_chain: str,
    cutoff_angstrom: float = 8.0,
) -> dict[str, set[int]]:
    binder = [atom for atom in atoms if atom[0] == binder_chain]
    target = [atom for atom in atoms if atom[0] == target_chain]
    if not binder or not target:
        return {}
    cutoff_sq = cutoff_angstrom * cutoff_angstrom
    focus: dict[str, set[int]] = {binder_chain: set(), target_chain: set()}
    nearest = None
    for target_atom in target:
        for binder_atom in binder:
            dist_sq = sum((target_atom[i] - binder_atom[i]) ** 2 for i in (2, 3, 4))
            if dist_sq <= cutoff_sq:
                focus[target_chain].add(target_atom[1])
                focus[binder_chain].add(binder_atom[1])
            if nearest is None or dist_sq < nearest[0]:
                nearest = (dist_sq, target_atom, binder_atom)
    if focus[target_chain] and focus[binder_chain]:
        return focus
    if nearest is not None:
        _, t_atom, b_atom = nearest
        focus[target_chain].add(t_atom[1])
        focus[binder_chain].add(b_atom[1])
    return focus


def _render_candidate_pictures(
    candidate: CandidateRecord,
    output_dir: Path,
    hotspots: Mapping[str, set[int]],
) -> tuple[str | None, str | None, str | None, str | None]:
    """Render overview and interface close-up PNGs for a candidate.

    Returns (overview_data_uri, overview_error, closeup_data_uri, closeup_error).
    """
    if not candidate.structure_path or not candidate.structure_path.is_file():
        err = f"Structure coordinate file is missing: {candidate.structure_path}"
        return None, err, None, err

    if not candidate.target_chain or not candidate.binder_chain:
        err = (
            "Chain roles are unresolved for "
            f"{candidate.candidate_id}: no scored record names "
            f"{candidate.structure_path}, so which chain holds the target is unknown. "
            "A guessed pair would colour the binder as the target."
        )
        return None, err, None, err

    if candidate.target_chain == candidate.binder_chain:
        err = (
            f"Chain roles collide for {candidate.candidate_id}: the target and the binder "
            f"are both recorded on chain {candidate.target_chain}."
        )
        return None, err, None, err

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    clean_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate.candidate_id)
    overview_png = figures_dir / f"{clean_id}_overview.png"
    closeup_png = figures_dir / f"{clean_id}_interface_closeup.png"

    overview_uri = None
    overview_err = None
    try:
        render_backbone_png(
            candidate.structure_path,
            overview_png,
            binder_chain=candidate.binder_chain,
            target_chain=candidate.target_chain,
            highlighted_residues=hotspots,
            identity_strip_color=OVERVIEW_STRIP,
        )
        overview_bytes = overview_png.read_bytes()
        overview_uri = f"data:image/png;base64,{base64.b64encode(overview_bytes).decode('ascii')}"
    except Exception as exc:  # noqa: BLE001
        overview_err = f"Render failed: {exc}"

    closeup_uri = None
    closeup_err = None
    try:
        atoms = read_backbone(candidate.structure_path)
        focus = _interface_focus(
            atoms,
            binder_chain=candidate.binder_chain,
            target_chain=candidate.target_chain,
            cutoff_angstrom=8.0,
        )
        render_backbone_png(
            candidate.structure_path,
            closeup_png,
            binder_chain=candidate.binder_chain,
            target_chain=candidate.target_chain,
            highlighted_residues=hotspots,
            focus_residues=focus,
            identity_strip_color=INTERFACE_STRIP,
        )
        closeup_bytes = closeup_png.read_bytes()
        closeup_uri = f"data:image/png;base64,{base64.b64encode(closeup_bytes).decode('ascii')}"
    except Exception as exc:  # noqa: BLE001
        closeup_err = f"Interface render failed: {exc}"

    return overview_uri, overview_err, closeup_uri, closeup_err


def _parse_hotspot_set(hotspot_strings: list[str] | None) -> dict[str, set[int]]:
    if not hotspot_strings:
        return {}
    hotspots: dict[str, set[int]] = {}
    for spec in hotspot_strings:
        match = RESIDUE_TOKEN.fullmatch(spec.strip())
        if match is None or match.group("chain") is None:
            continue
        chain = match.group("chain")
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end >= start:
            hotspots.setdefault(chain, set()).update(range(start, end + 1))
    return hotspots


def build_report(campaign_dir: Path, output_html: Path | None = None) -> Path:
    """Build a self-contained HTML report for a campaign directory."""
    campaign_dir = campaign_dir.resolve()
    if output_html is None:
        out_path = campaign_dir / "index.html"
    else:
        out_path = output_html.resolve()

    header = _parse_manifest_header(campaign_dir)
    deviations = _parse_deviations(campaign_dir)
    controls = _parse_controls(campaign_dir)
    candidates, arm_names = _parse_candidates(campaign_dir)

    has_controls = len(controls) > 0
    refuse_ranking = not has_controls

    hotspot_set = _parse_hotspot_set(header.hotspot_residues)

    zero_count = sum(
        1
        for c in candidates
        if c.combined_rank_score is not None and abs(c.combined_rank_score) < 1e-6
    )
    score_counts: Counter[float] = Counter()
    for c in candidates:
        if c.combined_rank_score is not None:
            score_counts[round(c.combined_rank_score, 4)] += 1
    tied_buckets = {score: cnt for score, cnt in score_counts.items() if cnt > 1}
    tied_count = sum(tied_buckets.values())
    distinct_tied = len(tied_buckets)

    pictures_html_parts: list[str] = []
    if not refuse_ranking:
        top_candidates = sorted(
            [c for c in candidates if not c.is_control],
            key=lambda c: (c.rank is None, c.rank if c.rank is not None else 999999),
        )[:3]

        for candidate in top_candidates:
            c_rank = candidate.rank if candidate.rank is not None else "unranked"
            c_id = html.escape(candidate.candidate_id)
            ov_uri, ov_err, cu_uri, cu_err = _render_candidate_pictures(
                candidate,
                out_path.parent,
                _retarget_hotspots(
                    hotspot_set, candidate.declared_target_chain, candidate.target_chain
                ),
            )

            overview_body = (
                f'<img src="{ov_uri}" alt="{c_id} overview" class="structure-image">'
                if ov_uri
                else f'<div class="picture-placeholder"><strong>{c_id} overview</strong>: {html.escape(ov_err or "structure could not be rendered")}</div>'
            )
            closeup_body = (
                f'<img src="{cu_uri}" alt="{c_id} interface close-up" class="structure-image">'
                if cu_uri
                else f'<div class="picture-placeholder"><strong>{c_id} interface close-up</strong>: {html.escape(cu_err or "structure could not be rendered")}</div>'
            )

            pictures_html_parts.append(
                f"""
                <div class="picture-card">
                  <h3>Rank {c_rank}: {c_id}</h3>
                  <div class="picture-pair">
                    <div class="picture-column">
                      <h4>Complex Overview</h4>
                      {overview_body}
                    </div>
                    <div class="picture-column">
                      <h4>Interface Close-Up</h4>
                      {closeup_body}
                    </div>
                  </div>
                </div>
                """
            )

    all_table_rows: list[CandidateRecord] = []
    if not refuse_ranking:
        all_table_rows.extend(controls)
        all_table_rows.extend(candidates)
        all_table_rows.sort(
            key=lambda r: (
                0 if r.is_control else 1,
                r.rank if r.rank is not None else 999999,
                -(r.combined_rank_score if r.combined_rank_score is not None else -999999),
            )
        )

    table_rows_html: list[str] = []
    for r in all_table_rows:
        row_cls = ' class="control-row"' if r.is_control else ""
        rank_label = f"Control ({html.escape(r.control_role or 'ref')})" if r.is_control else (str(r.rank) if r.rank is not None else "-")
        cid_label = html.escape(r.candidate_id)
        seq_display = html.escape(r.sequence) if r.sequence else "-"
        comb_score = f"{r.combined_rank_score:.4f}" if r.combined_rank_score is not None else "-"

        arm_cells = []
        for arm in arm_names:
            val = r.arm_scores.get(arm)
            cell_text = f"{val:.4f}" if val is not None else "-"
            arm_cells.append(f"<td>{cell_text}</td>")

        arm_cells_str = "".join(arm_cells)

        table_rows_html.append(
            f"""
            <tr{row_cls}>
              <td><strong>{rank_label}</strong></td>
              <td>{cid_label}</td>
              <td>{r.sequence_length}</td>
              {arm_cells_str}
              <td><strong>{comb_score}</strong></td>
              <td class="sequence-cell">{seq_display}</td>
            </tr>
            """
        )

    tools_rows_html: list[str] = []
    if header.tools_and_revisions:
        for t in header.tools_and_revisions:
            st = html.escape(t.get("stage", "-"))
            tool_name = html.escape(t.get("tool", "-"))
            rev = html.escape(t.get("revision", "-"))
            tools_rows_html.append(f"<li><strong>{st}</strong>: {tool_name} (revision {rev})</li>")
    tools_block = (
        f'<ul class="tools-list">{"".join(tools_rows_html)}</ul>'
        if tools_rows_html
        else f'<p class="todo">{_format_todo("stage tools and revisions", "campaign manifest stages or adapter receipts")}</p>'
    )

    deviations_html_parts: list[str] = []
    if deviations is not None:
        if deviations:
            for dev in deviations:
                t_name = html.escape(dev.get("tool", "Baseline deviation"))
                r_text = html.escape(dev.get("reason", ""))
                deviations_html_parts.append(f"<li><strong>{t_name}</strong>: {r_text}</li>")
        else:
            deviations_html_parts.append("<li>No deviations were logged for this campaign run.</li>")
    deviations_block = (
        f'<ul class="deviations-list">{"".join(deviations_html_parts)}</ul>'
        if deviations is not None
        else f'<p class="todo">{_format_todo("deviation log", "deviations.json or manifest gap_notes.deviations")}</p>'
    )

    arm_headers_html = "".join(f"<th>{html.escape(arm)}</th>" for arm in arm_names)

    target_val = html.escape(header.target_name) if header.target_name else f'<span class="todo">{_format_todo("target name", "campaign manifest target_name or targets[0].target_id")}</span>'
    uniprot_val = html.escape(header.uniprot_accession) if header.uniprot_accession else f'<span class="todo">{_format_todo("UniProt accession", "campaign manifest uniprot_accession or targets[0].uniprot_accession")}</span>'
    pdb_val = html.escape(header.pdb_entry) if header.pdb_entry else f'<span class="todo">{_format_todo("PDB entry", "campaign manifest structure_id or pdb_entry")}</span>'
    hotspots_val = html.escape(", ".join(header.hotspot_residues)) if header.hotspot_residues else f'<span class="todo">{_format_todo("hotspot residue list", "campaign manifest hotspot_residues or targets[0].site.design_residues")}</span>'
    n_designs_val = str(header.n_designs) if header.n_designs is not None else f'<span class="todo">{_format_todo("generated design count N", "campaign manifest n_designs or request.num_proteins")}</span>'
    rounds_val = str(header.rounds) if header.rounds is not None else f'<span class="todo">{_format_todo("round count", "campaign manifest rounds or optimization.rounds")}</span>'
    spend_val = html.escape(header.total_spend) if header.total_spend else f'<span class="todo">{_format_todo("total spend", "campaign manifest total_spend or artifacts/spend.jsonl")}</span>'

    if refuse_ranking:
        ranking_section_html = """
        <section class="refusal-box">
          <h2>Ranking Withheld</h2>
          <p>Refusal: No controls were run for this campaign.</p>
          <p>A ranking cannot be rendered without positive and negative controls to establish score scale.</p>
          <p>The candidate sequences and metric files remain preserved in the campaign directory.</p>
        </section>
        """
    else:
        floor_text = (
            f"Floor accounting: {zero_count} of {len(candidates)} designs scored exactly 0.000 at the floor. "
            f"A total of {tied_count} designs are tied across {distinct_tied} distinct score values. "
            f"The tie order carries no information."
        )
        ranking_section_html = f"""
        <section class="ranking-section">
          <h2>Ranked Candidates</h2>
          <p class="floor-accounting"><strong>{floor_text}</strong></p>
          <div class="table-container">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Design ID</th>
                  <th>Length</th>
                  {arm_headers_html}
                  <th>Combined Score</th>
                  <th>Sequence</th>
                </tr>
              </thead>
              <tbody>
                {"".join(table_rows_html)}
              </tbody>
            </table>
          </div>
        </section>
        """

    pictures_section_html = (
        f"""
        <section class="pictures-section">
          <h2>Structure Pictures (Top Designs)</h2>
          <div class="pictures-grid">
            {"".join(pictures_html_parts)}
          </div>
        </section>
        """
        if pictures_html_parts
        else ""
    )

    page_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Protein Binder Design Report</title>
  <style>
    :root {{
      --bg-color: #f8fafc;
      --surface-color: #ffffff;
      --text-color: #0f172a;
      --muted-text-color: #475569;
      --border-color: #cbd5e1;
      --primary-color: #0369a1;
      --header-bg: #f1f5f9;
      --control-row-bg: #f1f5f9;
      --badge-bg: #e2e8f0;
      --badge-text: #1e293b;
      --todo-bg: #fef3c7;
      --todo-text: #92400e;
      --todo-border: #f59e0b;
      --card-bg: #ffffff;
      --placeholder-bg: #f1f5f9;
      --placeholder-border: #94a3b8;
      --table-border: #e2e8f0;
      --table-alt-bg: #f8fafc;
      --refusal-bg: #fee2e2;
      --refusal-text: #991b1b;
      --refusal-border: #f87171;
    }}

    @media (prefers-color-scheme: dark) {{
      :root:not([data-theme="light"]) {{
        --bg-color: #0f172a;
        --surface-color: #1e293b;
        --text-color: #f8fafc;
        --muted-text-color: #94a3b8;
        --border-color: #334155;
        --primary-color: #38bdf8;
        --header-bg: #1e293b;
        --control-row-bg: #1e293b;
        --badge-bg: #334155;
        --badge-text: #f1f5f9;
        --todo-bg: #78350f;
        --todo-text: #fef3c7;
        --todo-border: #d97706;
        --card-bg: #1e293b;
        --placeholder-bg: #1e293b;
        --placeholder-border: #475569;
        --table-border: #334155;
        --table-alt-bg: #0f172a;
        --refusal-bg: #7f1d1d;
        --refusal-text: #fee2e2;
        --refusal-border: #ef4444;
      }}
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      background-color: var(--bg-color);
      color: var(--text-color);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 32px 20px;
      line-height: 1.5;
    }}

    main {{
      max-width: 1280px;
      margin: 0 auto;
    }}

    h1 {{
      font-size: 26px;
      margin: 0 0 16px 0;
      color: var(--text-color);
    }}

    h2 {{
      font-size: 20px;
      margin: 28px 0 12px 0;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 6px;
      color: var(--text-color);
    }}

    h3 {{
      font-size: 16px;
      margin: 0 0 12px 0;
      color: var(--text-color);
    }}

    h4 {{
      font-size: 14px;
      margin: 0 0 8px 0;
      color: var(--muted-text-color);
    }}

    p {{
      margin: 8px 0;
    }}

    .header-card {{
      background: var(--surface-color);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
    }}

    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px 20px;
      margin: 12px 0;
    }}

    .meta-item {{
      font-size: 14px;
    }}

    .tools-list, .deviations-list {{
      margin: 8px 0;
      padding-left: 20px;
    }}

    .tools-list li, .deviations-list li {{
      margin-bottom: 4px;
      font-size: 14px;
    }}

    .todo {{
      background: var(--todo-bg);
      color: var(--todo-text);
      border: 1px solid var(--todo-border);
      border-radius: 4px;
      padding: 2px 6px;
      font-size: 13px;
      display: inline-block;
    }}

    .floor-accounting {{
      background: var(--surface-color);
      border-left: 4px solid var(--primary-color);
      padding: 12px 16px;
      margin: 16px 0;
      font-size: 14px;
    }}

    .table-container {{
      overflow-x: auto;
      width: 100%;
      margin: 16px 0;
      border: 1px solid var(--table-border);
      border-radius: 8px;
      background: var(--surface-color);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 14px;
    }}

    th, td {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--table-border);
      vertical-align: top;
    }}

    th {{
      background: var(--header-bg);
      font-weight: 600;
    }}

    tr:nth-child(even) {{
      background-color: var(--table-alt-bg);
    }}

    tr.control-row {{
      background-color: var(--control-row-bg);
      font-style: italic;
    }}

    .sequence-cell {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      word-break: break-all;
      white-space: pre-wrap;
      max-width: 340px;
    }}

    .refusal-box {{
      background: var(--refusal-bg);
      color: var(--refusal-text);
      border: 1px solid var(--refusal-border);
      border-radius: 8px;
      padding: 20px;
      margin: 24px 0;
    }}

    .refusal-box h2 {{
      border-bottom: none;
      color: var(--refusal-text);
      margin-top: 0;
    }}

    .pictures-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
      margin-top: 16px;
    }}

    .picture-card {{
      background: var(--surface-color);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px;
    }}

    .picture-pair {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 16px;
    }}

    .picture-column {{
      display: flex;
      flex-direction: column;
    }}

    .structure-image {{
      width: 100%;
      height: auto;
      border-radius: 4px;
      border: 1px solid var(--border-color);
      display: block;
    }}

    .picture-placeholder {{
      background: var(--placeholder-bg);
      border: 1px dashed var(--placeholder-border);
      border-radius: 4px;
      padding: 24px;
      text-align: center;
      color: var(--muted-text-color);
      font-size: 13px;
      min-height: 180px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    @media print {{
      body {{
        background-color: #ffffff;
        color: #000000;
        padding: 0;
      }}
      .table-container {{
        overflow: visible;
      }}
      table {{
        width: 100%;
        page-break-inside: auto;
      }}
      tr {{
        page-break-inside: avoid;
        page-break-after: auto;
      }}
      .pictures-grid {{
        display: block;
      }}
      .picture-card {{
        page-break-inside: avoid;
        margin-bottom: 24px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Protein Binder Design Campaign Report</h1>

    <section class="header-card">
      <h2>Campaign Manifest</h2>
      <div class="meta-grid">
        <div class="meta-item"><strong>Target Name</strong>: {target_val}</div>
        <div class="meta-item"><strong>UniProt Accession</strong>: {uniprot_val}</div>
        <div class="meta-item"><strong>PDB Structure</strong>: {pdb_val}</div>
        <div class="meta-item"><strong>Hotspot Residues</strong>: {hotspots_val}</div>
        <div class="meta-item"><strong>Generated Designs (N)</strong>: {n_designs_val}</div>
        <div class="meta-item"><strong>Optimization Rounds</strong>: {rounds_val}</div>
        <div class="meta-item"><strong>Total Spend</strong>: {spend_val}</div>
      </div>

      <h3>Stages and Tool Revisions</h3>
      {tools_block}
    </section>

    {ranking_section_html}

    {pictures_section_html}

    <section class="deviations-section">
      <h2>Baseline Deviations</h2>
      {deviations_block}
    </section>
  </main>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page_html, encoding="utf-8")
    return out_path


def _done_todo(field_name: str, expected_source: str) -> str:
    """Return a readable unresolved-input marker for REPORT.md."""

    return f"TODO: {field_name} missing; expected from {expected_source}."


def _done_config(root: Path) -> tuple[dict[str, Any], Path | None]:
    """Read the explicit final-report contract without guessing field names."""

    config_path = _find_file(root, _DONE_CONFIG_NAMES)
    if config_path is None:
        return {}, None
    config = _load_json(config_path)
    if not isinstance(config, dict):
        return {}, config_path
    return config, config_path


def _done_artifact_value(config: Mapping[str, Any], name: str) -> Any:
    artifacts = config.get("artifacts")
    if isinstance(artifacts, Mapping) and name in artifacts:
        return artifacts[name]
    return config.get(name)


def _done_path(root: Path, config: Mapping[str, Any], name: str) -> Path | None:
    raw = _done_artifact_value(config, name)
    return _done_resolve_path(root, raw)


def _done_resolve_path(root: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return path


def _done_role_errors(root: Path, config: Mapping[str, Any]) -> list[str]:
    """Check the declared final artifact roles without fixing filenames."""

    roles = config.get("artifact_roles")
    if not isinstance(roles, Mapping):
        return ["artifact_roles"]
    errors: list[str] = []
    for role in _DONE_ARTIFACT_ROLES:
        value = roles.get(role)
        values = value if isinstance(value, list) else [value]
        paths = [_done_resolve_path(root, item) for item in values]
        if not value or not all(path is not None and path.is_file() for path in paths):
            errors.append(role)
    return errors


def _done_list(config: Mapping[str, Any], name: str) -> list[str]:
    raw = _done_artifact_value(config, name)
    if isinstance(raw, list):
        return [str(value) for value in raw if isinstance(value, (str, int, float))]
    if isinstance(raw, str):
        return [raw]
    return []


def _done_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _done_score_rows(root: Path, config: Mapping[str, Any]) -> tuple[list[DoneScore], Path | None, str | None, str | None]:
    """Read declared score columns from a CSV without assigning fallback columns."""

    scores_path = _done_path(root, config, "scores_csv")
    candidate_column = config.get("candidate_id_column")
    primary_column = config.get("primary_metric_column")
    if not scores_path or not scores_path.is_file():
        return [], scores_path, None, None
    if not isinstance(candidate_column, str) or not isinstance(primary_column, str):
        return [], scores_path, None, None
    try:
        with scores_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or candidate_column not in reader.fieldnames or primary_column not in reader.fieldnames:
                return [], scores_path, candidate_column, primary_column
            rows = [
                DoneScore(
                    candidate_id=str(row.get(candidate_column) or ""),
                    primary_score=_done_number(row.get(primary_column)),
                    values=dict(row),
                )
                for row in reader
            ]
    except OSError:
        return [], scores_path, candidate_column, primary_column
    return rows, scores_path, candidate_column, primary_column


def _done_round_ids(path: Path) -> set[str]:
    """Collect candidate IDs from the explicit round-manifest shapes."""

    document = _load_json(path)
    if not isinstance(document, Mapping):
        return set()
    identifiers: set[str] = set()
    raw_ids = document.get("candidate_ids")
    if isinstance(raw_ids, list):
        identifiers.update(str(value) for value in raw_ids if value is not None)
    for collection_name in ("candidates", "designs", "rows"):
        records = document.get(collection_name)
        if isinstance(records, list):
            for record in records:
                if isinstance(record, Mapping):
                    value = record.get("candidate_id") or record.get("design_id") or record.get("id")
                    if value is not None:
                        identifiers.add(str(value))
    return identifiers


def _done_round_coverage(root: Path, config: Mapping[str, Any]) -> tuple[set[str], bool]:
    paths = _done_list(config, "round_manifests")
    if not paths:
        return set(), False
    identifiers: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            return set(), False
        if not path.is_file():
            return set(), False
        identifiers.update(_done_round_ids(path))
    return identifiers, True


def _done_provenance_gaps(root: Path, config: Mapping[str, Any]) -> list[str]:
    """Return declared unresolved lineage fields for included score rows."""

    gaps: list[str] = []
    for value in (config.get("provenance_gaps"), config.get("unresolved_provenance")):
        if isinstance(value, Mapping):
            for candidate_id, fields in value.items():
                if isinstance(fields, list):
                    gaps.extend(f"{candidate_id}: {field}" for field in fields)
                elif fields:
                    gaps.append(f"{candidate_id}: {fields}")
        elif isinstance(value, list):
            gaps.extend(str(item) for item in value if item)

    provenance_path = _done_path(root, config, "provenance_path")
    provenance = _load_json(provenance_path) if provenance_path and provenance_path.is_file() else None
    if isinstance(provenance, Mapping):
        for key in ("gaps", "unresolved_fields", "provenance_gaps"):
            value = provenance.get(key)
            if isinstance(value, Mapping):
                for candidate_id, fields in value.items():
                    if isinstance(fields, list):
                        gaps.extend(f"{candidate_id}: {field}" for field in fields)
                    elif fields:
                        gaps.append(f"{candidate_id}: {fields}")
            elif isinstance(value, list):
                gaps.extend(str(item) for item in value if item)
    return sorted(set(gaps))


def _done_refusals(
    scores: Sequence[DoneScore],
    *,
    round_ids: set[str],
    round_coverage_known: bool,
    provenance_gaps: Sequence[str],
) -> list[DoneRefusal]:
    """Apply the five final-stage refusal conditions before rendering a table."""

    refusals: list[DoneRefusal] = []
    score_ids = {score.candidate_id for score in scores if score.candidate_id}
    numeric_scores = [score.primary_score for score in scores if score.primary_score is not None]
    if not scores:
        refusals.append(DoneRefusal("R1", "Zero candidate rows survived into the final score table."))
    if len(numeric_scores) < 2:
        refusals.append(DoneRefusal("R3", "Fewer than two candidate rows carry a numeric primary score."))
    if numeric_scores and len(numeric_scores) == len(scores) and all(score == PRIMARY_METRIC_FLOOR for score in numeric_scores):
        refusals.append(
            DoneRefusal(
                "R2",
                "Every surviving candidate has a primary score of exactly 0.000000 under the 10 angstrom aligned-error cutoff.",
            )
        )
    if not round_coverage_known:
        refusals.append(
            DoneRefusal(
                "R4",
                "Round coverage cannot be checked because round manifests are missing or unreadable.",
            )
        )
    elif round_ids != score_ids:
        refusals.append(
            DoneRefusal(
                "R4",
                "The candidate IDs from round manifests differ from the candidate IDs in the final score table.",
            )
        )
    if provenance_gaps:
        refusals.append(
            DoneRefusal(
                "R5",
                "Included candidates have unresolved provenance fields: " + "; ".join(provenance_gaps) + ".",
            )
        )
    return refusals


def _done_manifest_value(config: Mapping[str, Any], name: str, expected_source: str) -> str:
    value = config.get(name)
    if value is None or value == "":
        return _done_todo(name, expected_source)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _done_score_text(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "TODO: numeric primary score missing from scores CSV."


def _done_decoy_spread(score: DoneScore, config: Mapping[str, Any], primary_column: str | None) -> str:
    definition = config.get("decoy_spread")
    if not isinstance(definition, Mapping):
        return _done_todo("decoy spread formula", "done-report manifest decoy_spread")
    columns = definition.get("columns")
    reducer = definition.get("reducer")
    if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns) or reducer != "max":
        return _done_todo("decoy spread formula", "done-report manifest decoy_spread columns and reducer=max")
    if primary_column is None or score.primary_score is None:
        return "TODO: numeric primary score missing from scores CSV."
    values = [_done_number(score.values.get(column)) for column in columns]
    if any(value is None for value in values):
        return "TODO: numeric decoy score missing from scores CSV."
    return f"{score.primary_score - max(value for value in values if value is not None):.6f}"


def _done_table(
    scores: Sequence[DoneScore],
    config: Mapping[str, Any],
    primary_column: str | None,
) -> str:
    report_columns = config.get("report_columns")
    columns = report_columns if isinstance(report_columns, list) and all(isinstance(column, str) for column in report_columns) else []
    header = ["Design ID", primary_column or "TODO: primary_metric_column missing", *columns, "Primary minus largest listed decoy score"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for score in sorted(
        (item for item in scores if item.primary_score is not None and item.primary_score > PRIMARY_METRIC_FLOOR),
        key=lambda item: (-float(item.primary_score), item.candidate_id),
    ):
        values = [score.candidate_id, _done_score_text(score.primary_score)]
        values.extend(str(score.values.get(column) or _done_todo(column, "scores CSV")) for column in columns)
        values.append(_done_decoy_spread(score, config, primary_column))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _done_reference_lines(config: Mapping[str, Any]) -> list[str]:
    references = config.get("reference_scores")
    if not isinstance(references, Mapping):
        return [_done_todo("reference scores", "done-report manifest reference_scores")]
    lines: list[str] = []
    for label in ("natural_pair", "barnase_target"):
        value = _done_number(references.get(label))
        if value is None:
            lines.append(_done_todo(label, "done-report manifest reference_scores"))
        else:
            lines.append(f"{label}: {value:.6f}")
    return lines


def _done_artifact_links(root: Path) -> str:
    paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in _DONE_ARTIFACT_NAMES
    )
    paths.extend(sorted(_DONE_ARTIFACT_NAMES))
    return "\n".join(f"- [{path}]({path})" for path in paths)


def _render_done_report(
    root: Path,
    config: Mapping[str, Any],
    scores: Sequence[DoneScore],
    primary_column: str | None,
    refusals: Sequence[DoneRefusal],
) -> str:
    """Render the scientist-facing final page from artifact-backed values."""

    numeric_scores = [score.primary_score for score in scores if score.primary_score is not None]
    at_floor = [score.candidate_id for score in scores if score.primary_score == PRIMARY_METRIC_FLOOR]
    above_floor = [score for score in scores if score.primary_score is not None and score.primary_score > PRIMARY_METRIC_FLOOR]
    best_score = max(numeric_scores) if numeric_scores else None
    delivered = _done_manifest_value(config, "n_delivered", "campaign manifest")
    requested = _done_manifest_value(config, "n_requested", "campaign manifest")
    target = _done_manifest_value(config, "target_name", "campaign manifest")
    site = _done_manifest_value(config, "site_residues", "campaign manifest")
    if refusals:
        if any(refusal.code == "R2" for refusal in refusals):
            verdict = (
                f"This run produced {delivered} of {requested} designs for {target} at {site}. "
                f"Every numeric primary score is exactly 0.000000 under the {ALIGNED_ERROR_CUTOFF_ANGSTROM:.0f} angstrom aligned-error cutoff. "
                "The run cannot separate these designs. This report ranks no design. "
                "This report records computational scores."
            )
        else:
            verdict = (
                f"This run withheld ranking for {target} at {site}. "
                + " ".join(refusal.message for refusal in refusals)
                + " This report records computational scores."
            )
    else:
        verdict = (
            f"This run produced {delivered} of {requested} designs for {target} at {site}. "
            f"{len(above_floor)} of {len(scores)} score above the primary-metric floor. "
            f"The remaining {len(at_floor)} tie at exactly 0.000000 under the {ALIGNED_ERROR_CUTOFF_ANGSTROM:.0f} angstrom aligned-error cutoff. "
            f"This report orders only the {len(above_floor)} designs above the floor. "
            f"The strongest primary score is {_done_score_text(best_score)}. "
            "This report records computational scores."
        )

    predictor_family = _done_manifest_value(config, "predictor_family", "campaign manifest")
    comparability_label = config.get("comparability_label")
    if not isinstance(comparability_label, str) or not comparability_label:
        comparability_label = "Internal computational ordering."
    metrics = {
        "best_primary": _done_score_text(best_score),
        "n_above_floor": len(above_floor),
        "n_at_floor": len(at_floor),
        "n_delivered": len(scores),
    }
    marker = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
    lines = [
        "# Computational design run",
        "",
        verdict,
        "",
        "## Run identity",
        "",
        f"- Target: {target}",
        f"- Site: {site}",
        f"- Requested designs: {requested}",
        f"- Delivered designs: {delivered}",
        f"- Seed policy: {_done_manifest_value(config, 'seed_policy', 'campaign manifest')}",
        f"- Tool versions: {_done_manifest_value(config, 'tool_versions', 'campaign manifest')}",
        f"- Date: {_done_manifest_value(config, 'date', 'campaign manifest')}",
        "",
        "## Separation summary",
        "",
        f"- Designs delivered from the score table: {len(scores)}",
        f"- Designs above the floor: {len(above_floor)}",
        f"- Distinct score values above the floor: {len({score.primary_score for score in above_floor})}",
        f"- Designs at the floor: {len(at_floor)}",
        f"- Strongest primary score: {_done_score_text(best_score)}",
        f"- Predictor family: {predictor_family}",
        f"- Comparability label: {comparability_label}",
        "",
    ]
    if refusals:
        lines.extend(["## Ranking withheld", ""])
        lines.extend(f"- **{refusal.code}**: {refusal.message}" for refusal in refusals)
        lines.append("")
    else:
        lines.extend(["## Separable designs", "", _done_table(scores, config, primary_column), ""])
        lines.extend(["## Reference scores", ""])
        lines.extend(f"- {line}" for line in _done_reference_lines(config))
        lines.append("")
    lines.extend(
        [
            "## Tied designs",
            "",
            f"{len(at_floor)} designs tie at exactly 0.000000: {', '.join(at_floor) if at_floor else 'none'}. "
            "Order within this group carries no information.",
            "",
            "## Caveats",
            "",
            f"- Predictor family: {predictor_family}.",
            "- " + _done_todo("counter-screen evidence", "completed-run counter-screen artifacts"),
            "- " + _done_todo("predictor-arm ordering evidence", "completed-run predictor agreement artifact"),
            "- " + _done_todo("seed-aggregation bias disclosure", "completed-run seed-policy evidence"),
            "",
            "## Artifacts",
            "",
            _done_artifact_links(root),
            "",
            "## Verification",
            "",
            f"- [Verification receipt]({DONE_RECEIPT_NAME})",
            f"- [Done manifest]({DONE_MANIFEST_NAME})",
            "",
            f"<!-- done-stage-metrics: {marker} -->",
            "",
        ]
    )
    return "\n".join(lines)


def _done_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _done_manifest_payload(root: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == DONE_MANIFEST_NAME:
            continue
        artifacts.append({"path": path.relative_to(root).as_posix(), "sha256": _done_sha256(path)})
    return {"format": "claude-binder-done-manifest-v1", "artifacts": artifacts}


def _done_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _done_marker_values(report_text: str) -> Mapping[str, Any] | None:
    match = _DONE_MARKER.search(report_text)
    if match is None:
        return None
    try:
        value = json.loads(match.group("payload"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, Mapping) else None


def _done_csv_column_count(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except OSError:
        return 0
    return len(header)


def _done_table_row_count(report_text: str) -> int:
    section = report_text.split("## Separable designs\n", 1)
    if len(section) != 2:
        return 0
    table_text = section[1].split("\n## ", 1)[0]
    rows = [line for line in table_text.splitlines() if line.startswith("|")]
    return max(0, len(rows) - 2)


def _done_checksum_entries(value: Any, root: Path, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        path_value = value.get("path")
        sha_value = value.get("sha256")
        if isinstance(path_value, str) and isinstance(sha_value, str):
            path = root / path_value
            if not path.is_file() or _done_sha256(path) != sha_value:
                errors.append(path_value)
        for nested in value.values():
            _done_checksum_entries(nested, root, errors)
    elif isinstance(value, list):
        for nested in value:
            _done_checksum_entries(nested, root, errors)


def _done_mmcif_chains(path: Path) -> set[str]:
    """Read chain identifiers from the atom-site loop in a mmCIF file."""

    headers: list[str] = []
    chain_index: int | None = None
    in_atom_loop = False
    chains: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line == "loop_":
                    headers = []
                    chain_index = None
                    in_atom_loop = False
                    continue
                if line.startswith("_atom_site."):
                    headers.append(line.split()[0])
                    if line.startswith("_atom_site.label_asym_id") or line.startswith("_atom_site.auth_asym_id"):
                        chain_index = len(headers) - 1
                    in_atom_loop = True
                    continue
                if not in_atom_loop or chain_index is None or not line or line.startswith("_") or line == "#":
                    continue
                tokens = line.split()
                if len(tokens) > chain_index:
                    chains.add(tokens[chain_index])
    except OSError:
        return chains
    return chains


def _done_check(
    check_id: str,
    passed: bool,
    detail: str,
) -> dict[str, str]:
    return {"id": check_id, "status": "pass" if passed else "fail", "detail": detail}


def _done_content_checks(
    root: Path,
    config: Mapping[str, Any],
    scores: Sequence[DoneScore],
    scores_path: Path | None,
    primary_column: str | None,
    report_text: str,
) -> list[dict[str, str]]:
    """Run C3 through C13 without mutating the completed-run folder."""

    checks: list[dict[str, str]] = []
    fasta_path = _done_path(root, config, "design_fasta")
    fasta_ids = set(_parse_fasta(fasta_path).keys()) if fasta_path and fasta_path.is_file() else set()
    score_ids = {score.candidate_id for score in scores if score.candidate_id}
    declared_delivered = _done_number(config.get("n_delivered"))
    c3_ok = bool(scores_path and scores_path.is_file() and fasta_path and fasta_path.is_file())
    c3_ok = c3_ok and score_ids == fasta_ids and declared_delivered == len(scores) and _done_csv_column_count(scores_path) == 21
    checks.append(_done_check("C3", c3_ok, "design FASTA IDs, score-table IDs, declared delivery count, and 21 score columns agree"))

    marker = _done_marker_values(report_text)
    numeric_scores = [score.primary_score for score in scores if score.primary_score is not None]
    expected_floor = sum(score == PRIMARY_METRIC_FLOOR for score in numeric_scores)
    expected_best = max(numeric_scores) if numeric_scores else None
    c4_ok = marker is not None
    c4_ok = c4_ok and marker.get("n_at_floor") == expected_floor
    c4_ok = c4_ok and marker.get("best_primary") == _done_score_text(expected_best)
    checks.append(_done_check("C4", c4_ok, "floor count and strongest score match the score table"))

    above_floor = sum(score is not None and score > PRIMARY_METRIC_FLOOR for score in numeric_scores)
    c5_ok = _done_table_row_count(report_text) == above_floor
    if expected_floor:
        c5_ok = c5_ok and "Order within this group carries no information." in report_text
    checks.append(_done_check("C5", c5_ok, "the separable-design table excludes designs at the primary-score floor"))

    zero_variance_columns: list[str] = []
    for column in _done_list(config, "control_columns"):
        values = [_done_number(score.values.get(column)) for score in scores]
        numeric = [value for value in values if value is not None]
        if numeric and len(set(numeric)) == 1:
            zero_variance_columns.append(column)
    report_lower = report_text.lower()
    c6_ok = not zero_variance_columns or not any(term in report_lower for term in _EFFECT_SIZE_TERMS)
    checks.append(_done_check("C6", c6_ok, "zero-variance control columns carry no prohibited statistic term"))

    fixture_refusals = _done_refusals(
        [DoneScore("fixture-a", 0.0, {}), DoneScore("fixture-b", 0.0, {})],
        round_ids={"fixture-a", "fixture-b"},
        round_coverage_known=True,
        provenance_gaps=[],
    )
    c7_ok = any(refusal.code == "R2" for refusal in fixture_refusals)
    checks.append(_done_check("C7", c7_ok, "the all-floor fixture reaches the total-floor refusal"))

    round_ids, coverage_known = _done_round_coverage(root, config)
    c8_ok = coverage_known and round_ids == score_ids
    checks.append(_done_check("C8", c8_ok, "round-manifest candidate IDs equal score-table candidate IDs"))

    field_writers = config.get("field_writers")
    read_fields = (
        "candidate_id_column",
        "primary_metric_column",
        "scores_csv",
        "design_fasta",
        "round_manifests",
        "provenance_path",
        "n_requested",
        "n_delivered",
        "target_name",
        "site_residues",
        "seed_policy",
        "tool_versions",
        "date",
        "predictor_family",
        "report_columns",
        "decoy_spread",
        "reference_scores",
        "comparability_label",
        "counter_screen_artifacts",
        "control_columns",
        "complex_path",
        "figure_paths",
        "provenance_gaps",
        "unresolved_provenance",
        "artifact_roles",
    )
    unresolved_fields = [
        field_name
        for field_name in read_fields
        if field_name in config
        and (
            not isinstance(field_writers, Mapping)
            or not isinstance(field_writers.get(field_name), str)
            or not field_writers.get(field_name)
        )
    ]
    c9_ok = isinstance(field_writers, Mapping) and not unresolved_fields
    detail = "all declared final-stage fields name an earlier writer" if c9_ok else "missing writer records for " + ", ".join(unresolved_fields or ["field_writers"])
    checks.append(_done_check("C9", c9_ok, detail))

    complex_path = _done_path(root, config, "complex_path")
    chains = _done_mmcif_chains(complex_path) if complex_path and complex_path.suffix.lower() in {".cif", ".mmcif"} else set()
    c10_ok = len(chains) >= 2
    checks.append(_done_check("C10", c10_ok, "the declared mmCIF contains at least two atom-site chains"))

    figures = _done_list(config, "figure_paths")
    manifest = _load_json(root / DONE_MANIFEST_NAME)
    hashes = {
        item.get("path"): item.get("sha256")
        for item in manifest.get("artifacts", [])
        if isinstance(manifest, Mapping) and isinstance(item, Mapping)
    } if isinstance(manifest, Mapping) else {}
    c11_ok = len(figures) == 2
    for figure in figures:
        figure_path = root / figure
        c11_ok = c11_ok and figure_path.is_file() and figure_path.stat().st_size > 0 and hashes.get(figure) == _done_sha256(figure_path)
    checks.append(_done_check("C11", c11_ok, "declared figures exist, contain bytes, and match the done manifest"))

    label = str(config.get("comparability_label") or "Internal computational ordering.")
    claims_comparability = "baseline compar" in label.lower()
    counter_screen_paths = _done_list(config, "counter_screen_artifacts")
    c12_ok = not claims_comparability or bool(counter_screen_paths) and all((root / path).is_file() for path in counter_screen_paths)
    checks.append(_done_check("C12", c12_ok, "baseline comparability has declared counter-screen artifacts"))

    refusals = _done_refusals(scores, round_ids=round_ids, round_coverage_known=coverage_known, provenance_gaps=_done_provenance_gaps(root, config))
    second = _render_done_report(root, config, scores, primary_column, refusals)
    c13_ok = report_text == second
    checks.append(_done_check("C13", c13_ok, "the report renderer returns identical bytes for identical inputs"))
    return checks


def verify_done_report(campaign_dir: Path) -> DoneVerification:
    """Verify the final-stage receipt against a completed-run folder."""

    root = campaign_dir.resolve()
    config, _ = _done_config(root)
    scores, scores_path, _, primary_column = _done_score_rows(root, config)
    report_path = root / DONE_REPORT_NAME
    report_text = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    manifest_path = root / DONE_MANIFEST_NAME
    manifest = _load_json(manifest_path) if manifest_path.is_file() else None

    expected_paths: set[str] = {DONE_MANIFEST_NAME}
    checksum_errors: list[str] = []
    if isinstance(manifest, Mapping) and isinstance(manifest.get("artifacts"), list):
        for entry in manifest["artifacts"]:
            if isinstance(entry, Mapping) and isinstance(entry.get("path"), str):
                expected_paths.add(entry["path"])
                path = root / entry["path"]
                if not path.is_file() or not isinstance(entry.get("sha256"), str) or _done_sha256(path) != entry["sha256"]:
                    checksum_errors.append(entry["path"])
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    role_errors = _done_role_errors(root, config)
    retired_ranking = root / "RANKED.md"
    c1_ok = isinstance(manifest, Mapping) and actual_paths == expected_paths and not role_errors and not retired_ranking.is_file()
    c2_ok = isinstance(manifest, Mapping) and not checksum_errors
    provenance_path = _done_path(root, config, "provenance_path")
    if provenance_path and provenance_path.is_file():
        _done_checksum_entries(_load_json(provenance_path), root, checksum_errors)
        c2_ok = c2_ok and not checksum_errors

    if role_errors:
        c1_detail = "missing declared artifact roles: " + ", ".join(role_errors)
    elif retired_ranking.is_file():
        c1_detail = "RANKED.md is retired; use REPORT.md"
    else:
        c1_detail = "folder contents equal the done-manifest enumeration"

    checks = [
        _done_check("C1", c1_ok, c1_detail),
        _done_check("C2", c2_ok, "recorded manifest and provenance checksums match their files"),
    ]
    checks.extend(_done_content_checks(root, config, scores, scores_path, primary_column, report_text))
    return DoneVerification(tuple(checks))


def publish_done_report(campaign_dir: Path) -> DoneReportResult:
    """Write REPORT.md, its receipt, and an integrity manifest for one run."""

    root = campaign_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    config, _ = _done_config(root)
    scores, scores_path, _, primary_column = _done_score_rows(root, config)
    round_ids, coverage_known = _done_round_coverage(root, config)
    refusals = _done_refusals(
        scores,
        round_ids=round_ids,
        round_coverage_known=coverage_known,
        provenance_gaps=_done_provenance_gaps(root, config),
    )
    report_path = root / DONE_REPORT_NAME
    receipt_path = root / DONE_RECEIPT_NAME
    manifest_path = root / DONE_MANIFEST_NAME
    report_text = _render_done_report(root, config, scores, primary_column, refusals)
    report_path.write_text(report_text, encoding="utf-8")

    role_errors = _done_role_errors(root, config)
    retired_ranking = root / "RANKED.md"
    if role_errors:
        c1_detail = "missing declared artifact roles: " + ", ".join(role_errors)
    elif retired_ranking.is_file():
        c1_detail = "RANKED.md is retired; use REPORT.md"
    else:
        c1_detail = "folder contents equal the done-manifest enumeration"
    provisional_checks = [
        _done_check("C1", not role_errors and not retired_ranking.is_file(), c1_detail),
        _done_check("C2", True, "recorded manifest and provenance checksums match their files"),
    ]
    provisional_checks.extend(_done_content_checks(root, config, scores, scores_path, primary_column, report_text))
    _done_write_json(
        receipt_path,
        {
            "format": "claude-binder-done-receipt-v1",
            "published": not refusals,
            "refusal_codes": [refusal.code for refusal in refusals],
            "checks": provisional_checks,
        },
    )
    _done_write_json(manifest_path, _done_manifest_payload(root))
    verification = verify_done_report(root)
    _done_write_json(
        receipt_path,
        {
            "format": "claude-binder-done-receipt-v1",
            "published": not refusals,
            "refusal_codes": [refusal.code for refusal in refusals],
            "checks": list(verification.checks),
        },
    )
    _done_write_json(manifest_path, _done_manifest_payload(root))
    return DoneReportResult(
        report_path=report_path,
        receipt_path=receipt_path,
        manifest_path=manifest_path,
        refusal_codes=tuple(refusal.code for refusal in refusals),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path, help="Path to completed campaign directory")
    parser.add_argument("--out", "-o", type=Path, default=None, help="Output HTML file path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--done", action="store_true", help="write the final REPORT.md package")
    mode.add_argument("--verify-done", action="store_true", help="verify a final REPORT.md package")
    args = parser.parse_args(argv)

    if args.verify_done:
        verification = verify_done_report(args.campaign_dir)
        for check in verification.checks:
            print(f"{check['id']}: {check['status']} - {check['detail']}")
        return 0 if verification.ok else 1
    if args.done:
        if args.out is not None:
            parser.error("--out cannot be used with --done")
        result = publish_done_report(args.campaign_dir)
        print(f"Done report built: {result.report_path}")
        return 0 if result.published else DONE_REFUSAL_EXIT
    try:
        result_path = build_report(args.campaign_dir, args.out)
        print(f"Report built: {result_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error building report: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
