#!/usr/bin/env python3
"""Write PyMOL and ChimeraX scripts for the ranked designs of a finished run.

The run leaves each predicted complex in the attempt directory of the cofold
stage that made it, under a name built from the target, the candidate, the
predictor and the seed. Finding the pose behind rank 1 means joining three files
by hand. This writes that join out as viewer scripts, so opening the rank 1 design
against its target is one command.

Output goes to <run>/viewer, which sits beside artifacts/ rather than inside it,
so nothing the run already validated changes.

    python3 scripts/viewer/make_viewer_scripts.py --run-dir <run>
    pymol <run>/viewer/top.pml

Every PyMOL and ChimeraX command emitted here was run against PyMOL 3.1.7.2 and
ChimeraX 1.11.1 before being written into a template.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote
from typing import Any, Mapping

# The AlphaFold pLDDT bands, as RGB fractions. The pipeline's own predictors
# write a per-atom confidence into the B-factor column of the complex; see the
# b_factor_range field of the manifest for what a given run actually carries.
PLDDT_BANDS = (
    ("plddt_vlow", (0.996, 0.490, 0.271), None, 50),
    ("plddt_low", (1.000, 0.859, 0.180), 50, 70),
    ("plddt_high", (0.396, 0.792, 0.933), 70, 90),
    ("plddt_vhigh", (0.000, 0.325, 0.702), 90, None),
)

# The two viewers do not share a colour vocabulary, so every colour here is an
# RGB triple. PyMOL gets a set_color definition, ChimeraX gets a hex literal.
SCENE_COLORS = {
    "lane_target": (0.439, 0.439, 0.439),
    "lane_site": (1.000, 0.612, 0.071),
    "lane_reference": (0.251, 0.251, 0.251),
    "lane_binder": (0.392, 0.584, 0.929),
}
TARGET_COLOR = "lane_target"
SITE_COLOR = "lane_site"
REFERENCE_COLOR = "lane_reference"
FALLBACK_BINDER_COLOR = "lane_binder"

RESIDUE_SPEC = re.compile(r"^\s*([A-Za-z0-9]+)\s*:\s*(-?\d+)(?:\s*-\s*(-?\d+))?\s*$")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def hex_color(name: str) -> str:
    """A ChimeraX colour literal for one of the scene colours."""
    red, green, blue = SCENE_COLORS[name]
    return "#{:02x}{:02x}{:02x}".format(
        round(red * 255), round(green * 255), round(blue * 255)
    )


def safe_name(value: str) -> str:
    """A PyMOL object name. Hyphens and dots are not safe inside a selection."""
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def stable_stem(rank: Any, candidate_id: str) -> str:
    """A filename stem that two different candidates cannot share.

    `safe_name` is right for a PyMOL object name and wrong for a filename. It maps every
    character outside `[A-Za-z0-9_]` to `_`, so `c-1`, `c_1` and `c.1` name one file, and
    nothing makes a rank unique: `lane.py` checks only that a rank is an `int`, and
    `notes/lanes/pictureleg2.md` records a run where repeated ranks overwrote thumbnails.
    Two designs colliding here wrote one scene script, and both manifest records then named
    a script that renders only the second candidate. The stage declares the manifest and the
    thumbnails and not these scripts, so no parser counts them and nothing catches it.

    `publication_renderer` and `browser_renderer` already quote the candidate for the same
    reason.
    """
    return f"rank-{int(rank):02d}-{quote(str(candidate_id), safe='-_.')}"


def parse_residue_specs(specs: list[str]) -> dict[str, list[str]]:
    """Group CHAIN:RESI and CHAIN:START-END specs by chain.

    Returns chain id to a list of residue ranges written as PyMOL and ChimeraX
    both accept them, for example ["1-3", "17"].
    """
    by_chain: dict[str, list[str]] = {}
    for spec in specs:
        match = RESIDUE_SPEC.match(spec)
        if not match:
            continue
        chain, first, last = match.group(1), match.group(2), match.group(3)
        ranges = by_chain.setdefault(chain, [])
        entry = f"{first}-{last}" if last else first
        if entry not in ranges:
            ranges.append(entry)
    return by_chain


def resolved_chain_ids(observation: Mapping[str, Any]) -> tuple[Any, Any]:
    """Return the (target, binder) chain ids the returned structure really uses.

    An observation row carries two mappings. The declared pair is what the
    campaign asked for. The predicted pair is what the returned structure uses.
    A predictor handed the configured letters returns them and the two agree. A
    provider application that writes its own mmCIF returns its own letters, and
    then only the predicted pair names the chains a viewer will find on disk.

    Both predicted fields have to be present and non-empty before either is
    used. One predicted letter beside one declared letter can name the same
    chain twice, which paints one chain in both roles and leaves the other
    unpainted. A row written before the predicted pair existed carries only the
    declared pair, so an absent predicted pair falls back rather than refusing.
    """
    predicted_target = observation.get("predicted_target_chain_id")
    predicted_binder = observation.get("predicted_binder_chain_id")
    if (
        isinstance(predicted_target, str)
        and predicted_target
        and isinstance(predicted_binder, str)
        and predicted_binder
    ):
        return predicted_target, predicted_binder
    return observation.get("target_chain_id"), observation.get("binder_chain_id")


def retarget_site_specs(specs: Any, source_chain: Any, target_chain: Any) -> list[str]:
    """Rewrite CHAIN:RESI specs from one chain letter onto another.

    A campaign writes its site residues against the target chain it declared. A
    prediction that returns the target on a different chain needs the same
    residue numbers under the returned letter, which is the relabelling
    `binder_contract._site_map_for_target` already does for the scored metrics.
    Residue numbers never change, and a spec on any other chain is left alone.
    """
    values = [str(spec) for spec in specs]
    if not isinstance(source_chain, str) or not isinstance(target_chain, str):
        return values
    if not source_chain or not target_chain or source_chain == target_chain:
        return values
    retargeted: list[str] = []
    for value in values:
        prefix, separator, suffix = value.partition(":")
        if separator and prefix.strip() == source_chain:
            retargeted.append(f"{target_chain}:{suffix}")
        else:
            retargeted.append(value)
    return retargeted


def b_factor_range(structure_path: Path) -> tuple[float, float] | None:
    """The lowest and highest B-factor column value in a PDB file."""
    values: list[float] = []
    try:
        text = structure_path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 66:
            try:
                values.append(float(line[60:66]))
            except ValueError:
                continue
    if not values:
        return None
    return min(values), max(values)


def resolve_viewer_target(
    ranked: dict[str, Any], observations: list[dict[str, Any]]
) -> str | None:
    """Return the target this viewer draws, or None when the observations name none.

    A multi-target campaign writes every target's rows into one
    `uniform-observations.jsonl`, and the ranked portfolio names the one it ranked
    against. Refuse only when the rows span two targets and the portfolio says nothing,
    because then there is no answer to take rather than invent.
    """
    primary = ranked.get("primary_target_id")
    if isinstance(primary, str) and primary:
        return primary
    target_ids = {
        str(row["target_id"]) for row in observations if row.get("target_id") is not None
    }
    if len(target_ids) > 1:
        raise ValueError(
            "the ranked portfolio names no primary_target_id and these uniform "
            f"observations span {len(target_ids)} targets: {sorted(target_ids)}"
        )
    return next(iter(target_ids), None)


def observation_index(
    observations: list[dict[str, Any]],
) -> dict[tuple[str | None, str, str, int], dict[str, Any]]:
    """Index observations on target, candidate, predictor and seed.

    The key dropped `target_id`, so a multi-target campaign's rows overwrote each other
    and `scenes_for_candidate` returned another target's structure for a ranked
    candidate. A silent overwrite in a delivered report. The rest of the package keys a
    score row on all four. Carrying the target is the
    fix here rather than refusing, because a two-target observation file is what a
    multi-target campaign is supposed to produce and the ranked portfolio already names
    which target it ranked against.
    """
    return {
        (
            str(r["target_id"]) if r.get("target_id") is not None else None,
            r["candidate_id"],
            r["predictor"],
            r["seed"],
        ): r
        for r in observations
        if r.get("candidate_id") and r.get("predictor") and r.get("seed") is not None
    }


def scenes_for_candidate(
    candidate: dict[str, Any],
    index: dict[tuple[str | None, str, str, int], dict[str, Any]],
    target_id: str | None = None,
) -> list[dict[str, Any]]:
    """One entry per predictor arm, richest first, with its complex on disk."""
    scenes: list[dict[str, Any]] = []
    for predictor, seed in sorted(
        candidate.get("selected_seed_by_predictor", {}).items()
    ):
        observation = index.get((target_id, candidate["candidate_id"], predictor, seed))
        if observation is None and target_id is not None:
            # A run whose rows predate the target field still indexes under None.
            observation = index.get((None, candidate["candidate_id"], predictor, seed))
        if observation is None and seed is None:
            available = sorted(
                (
                    row
                    for (row_target, candidate_id, observation_predictor, _), row in index.items()
                    if candidate_id == candidate["candidate_id"]
                    and observation_predictor == predictor
                    and row_target in (target_id, None)
                    and row.get("predicted_complex_path")
                    and Path(str(row["predicted_complex_path"])).is_file()
                ),
                key=lambda row: int(row["seed"]),
            )
            if available:
                observation = available[0]
        if observation is None:
            continue
        complex_path = observation.get("predicted_complex_path")
        if not complex_path or not Path(complex_path).is_file():
            continue
        # The chains a viewer selects are the ones the returned structure uses,
        # which is the predicted pair whenever the row carries it. The declared
        # pair travels beside it, because the input target structure is still
        # written in the declared namespace and so are the configured site
        # residues.
        scene_target_chain, scene_binder_chain = resolved_chain_ids(observation)
        scenes.append(
            {
                "predictor": predictor,
                "seed": observation["seed"],
                "complex_path": complex_path,
                "binder_chain": scene_binder_chain,
                "target_chain": scene_target_chain,
                "declared_binder_chain": observation.get("binder_chain_id"),
                "declared_target_chain": observation.get("target_chain_id"),
                "ipsae_min": observation.get("ipsae_min"),
                "sc_dockq": observation.get("sc_dockq"),
                "dockq": observation.get("dockq"),
                "interface_plddt": observation.get("interface_plddt"),
                "interface_pae": observation.get("interface_pae"),
                "clash_count": observation.get("clash_count"),
                "contact_count": observation.get("contact_count"),
                "object": safe_name(f"{predictor}_seed{seed}"),
            }
        )
    scenes.sort(key=lambda s: (s["ipsae_min"] is None, -(s["ipsae_min"] or 0.0)))
    return scenes


def show_metric(value: Any, places: int = 3) -> str:
    """A metric as a person would read it, not as a float repr."""
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.{places}f}".rstrip("0").rstrip(".") or "0"
    return str(value)


def show_rank_score(candidate: dict[str, Any], places: int = 4) -> str:
    """Render a rank score, preserving raw scores from one-member cohorts.

    A one-member within-pool z-score is zero by construction. Candidate-level
    raw-mean modes remain measured and meaningful with one candidate.
    """
    if candidate.get("rank_score_defined") is False:
        return "undefined, one design in the normalization pool"
    return show_metric(candidate.get("rank_score"), places)


def rank_score_is_defined(candidate_count: int, ranking_method: Any) -> bool:
    """Return whether ``rank_score`` carries information for this cohort and mode."""
    normalization = (
        ranking_method.get("normalization")
        if isinstance(ranking_method, dict)
        else None
    )
    is_within_pool_zscore = (
        isinstance(normalization, str) and "z-score" in normalization.casefold()
    )
    return candidate_count != 1 or not is_within_pool_zscore


def unresolved_top_tier_warning(manifest: dict[str, Any]) -> str | None:
    """Describe a multi-candidate leading tier independently of control status."""
    if manifest.get("best_design_claim_status") == "tied":
        return None
    top_tier = manifest.get("top_candidate_tier")
    tier_ids = top_tier.get("candidate_ids") if isinstance(top_tier, dict) else None
    if not isinstance(tier_ids, list) or len(tier_ids) < 2:
        return None
    names = ", ".join(str(candidate_id) for candidate_id in tier_ids)
    if top_tier.get("uncertainty_unavailable") is True:
        return (
            "Independently of control-validation status, stored paired-seed precision does "
            f"not resolve a unique leading candidate among: {names}."
        )
    return (
        "Independently of control-validation status, the leading candidate tier is tied "
        f"within stored seed noise: {names}."
    )


def metrics_line(candidate: dict[str, Any], scene: dict[str, Any], total: int) -> str:
    rank_scope = (
        candidate.get("rank_score_scope") or "rank_score scope was not recorded"
    )
    return (
        f"rank {candidate['rank']} of {total}  {candidate['candidate_id']}  "
        f"from {candidate['generator']}  rank score {show_rank_score(candidate)}  "
        f"[{rank_scope}]  |  "
        f"shown {scene['predictor']} seed {scene['seed']}  "
        f"ipSAE_min {show_metric(scene['ipsae_min'])}  "
        f"sc_DockQ {show_metric(scene['sc_dockq'])}  "
        f"interface pLDDT {show_metric(scene['interface_plddt'], 1)}"
    )


def pymol_safe(lines: list[str]) -> str:
    """Join PyMOL commands, with the semicolon removed from every comment.

    PyMOL splits a line on a semicolon and runs the second half as a command,
    inside a comment as much as outside it. A semicolon in prose therefore ends
    the script with a syntax error on text that reads like English.
    """
    cleaned = [
        line.replace(";", ",") if line.lstrip().startswith("#") else line
        for line in lines
    ]
    return "\n".join(cleaned)


def pymol_script(
    context: dict[str, Any], candidate: dict[str, Any], scenes: list[dict[str, Any]]
) -> str:
    shown = scenes[0]
    binder_chain = shown["binder_chain"]
    target_chain = shown["target_chain"]
    declared_target_chain = shown.get("declared_target_chain") or target_chain
    site_residues = parse_residue_specs(
        retarget_site_specs(context["site_residues"], declared_target_chain, target_chain)
    )
    obj = shown["object"]
    cutoff = context["contact_cutoff"]
    lines: list[str] = [
        "# Claude binder lane viewer script, PyMOL dialect.",
        f"# run {context['run_id']}, campaign {context['campaign_id']}, target {candidate['target_id']}",
        f"# rank {candidate['rank']} of {context['ranked_count']}, candidate {candidate['candidate_id']}",
        f"# shown arm {shown['predictor']} seed {shown['seed']}, the highest ipSAE_min of "
        f"{len(scenes)} arms",
        "# written by scripts/viewer/make_viewer_scripts.py, do not edit by hand",
        "",
        "reinitialize",
        "",
        "# structures",
    ]
    for scene in scenes:
        lines.append(f"load {scene['complex_path']}, {scene['object']}")
    lines += ["", "# palette"]
    for name, (red, green, blue) in SCENE_COLORS.items():
        lines.append(f"set_color {name}, [{red:.3f}, {green:.3f}, {blue:.3f}]")
    for name, (red, green, blue), _low, _high in PLDDT_BANDS:
        lines.append(f"set_color {name}, [{red:.3f}, {green:.3f}, {blue:.3f}]")

    lines += [
        "",
        "# base style",
        "hide everything",
        "bg_color white",
        "# a backbone-only model still draws a cartoon with this set",
        "set cartoon_trace_atoms, 1",
        f"show cartoon, {obj}",
        f"color {TARGET_COLOR}, {obj} and chain {target_chain}",
        f"color {FALLBACK_BINDER_COLOR}, {obj} and chain {binder_chain}",
    ]

    if context["plddt_colouring"]:
        lines.append("")
        lines.append(
            "# Binder coloured by confidence, painted highest band first and then"
        )
        lines.append(
            "# overwritten downwards. Two-sided bands would leave a residue sitting"
        )
        lines.append("# exactly on a boundary with no colour at all.")
        binder = f"{obj} and chain {binder_chain}"
        descending = list(reversed(PLDDT_BANDS))
        lines.append(f"color {descending[0][0]}, {binder}")
        for (name, _rgb, _low, _high), (_above, _rgb2, ceiling, _high2) in zip(
            descending[1:], descending
        ):
            lines.append(f"color {name}, {binder} and b < {ceiling}")
    else:
        lines.append("")
        lines.append(
            f"# the B-factor column of this complex is flat at "
            f"{context['b_factor_range']}, so confidence colouring would say nothing"
        )

    lines += ["", "# named selections you can reuse from the command line"]
    lines.append(f"select binder, {obj} and chain {binder_chain}")
    lines.append(
        f"select interface, byres ({obj} and chain {target_chain} within {cutoff} "
        f"of ({obj} and chain {binder_chain}))"
    )
    site_clauses = [
        f"(chain {chain} and resi {'+'.join(ranges)})"
        for chain, ranges in site_residues.items()
    ]
    if site_clauses:
        lines.append(f"select site, {obj} and ({' or '.join(site_clauses)})")
        lines.append("show sticks, site and sidechain")
        lines.append(f"color {SITE_COLOR}, site")
    lines.append("show sticks, interface and sidechain")

    if len(scenes) > 1:
        lines += [
            "",
            "# the other predictor arms, loaded and hidden. Enable one to compare",
        ]
        for scene in scenes[1:]:
            lines.append(f"disable {scene['object']}")

    lines += [
        "",
        "# view",
        f"orient {obj}",
        "zoom binder, 6",
        "set ray_opaque_background, 1",
        "set antialias, 2",
        "deselect",
        "",
        "# the numbers this design was ranked on",
        f'set_title {obj}, 1, "{metrics_line(candidate, shown, context["ranked_count"])}"',
    ]
    for line in context["metric_block"](candidate, scenes):
        lines.append(f'print "{line}"')

    if context["target_structure_path"]:
        lines += [
            "",
            "# The input target, last so that a failed overlay costs only the overlay.",
            "# The prediction carries its own copy of the target. This is the construct",
            "# that went in, for the parts the prediction did not return.",
            f"load {context['target_structure_path']}, target_input",
            "show cartoon, target_input",
            f"color {REFERENCE_COLOR}, target_input",
            "set cartoon_transparency, 0.7, target_input",
        ]
        if declared_target_chain != target_chain:
            lines.append(
                f"# The input structure carries the target on chain {declared_target_chain}. "
                f"The prediction returns it on chain {target_chain}."
            )
        lines.append(
            f"super target_input and chain {declared_target_chain}, "
            f"{obj} and chain {target_chain}"
        )
    lines.append("")
    return pymol_safe(lines)


def chimerax_script(
    context: dict[str, Any], candidate: dict[str, Any], scenes: list[dict[str, Any]]
) -> str:
    shown = scenes[0]
    binder_chain = shown["binder_chain"]
    target_chain = shown["target_chain"]
    declared_target_chain = shown.get("declared_target_chain") or target_chain
    site_residues = parse_residue_specs(
        retarget_site_specs(context["site_residues"], declared_target_chain, target_chain)
    )
    cutoff = context["contact_cutoff"]
    model_of = {scene["object"]: number for number, scene in enumerate(scenes, start=1)}
    shown_model = f"#{model_of[shown['object']]}"
    # A fixed high id for the reference target, well clear of the complexes
    # and of any model a label or a surface creates.
    target_model = "#50"

    lines: list[str] = [
        "# Claude binder lane viewer script, ChimeraX dialect.",
        f"# run {context['run_id']}, campaign {context['campaign_id']}, target {candidate['target_id']}",
        f"# rank {candidate['rank']} of {context['ranked_count']}, candidate {candidate['candidate_id']}",
        f"# shown arm {shown['predictor']} seed {shown['seed']}, the highest ipSAE_min of "
        f"{len(scenes)} arms",
        "# written by scripts/viewer/make_viewer_scripts.py, do not edit by hand",
        "",
        "close session",
        "",
        "# Structures, with the model id pinned. A 2dlabel is itself a model, so",
        "# letting ChimeraX number these implicitly makes the ids shift.",
    ]
    for scene in scenes:
        lines.append(
            f"open {scene['complex_path']} name {scene['object']} id #{model_of[scene['object']]}"
        )

    complex_models = f"#!1-{len(scenes)}" if len(scenes) > 1 else "#!1"
    lines += [
        "",
        "# base style",
        f"hide {complex_models} atoms",
        f"show {complex_models} cartoons",
        "set bgColor white",
        f"color {shown_model}/{target_chain} {hex_color(TARGET_COLOR)}",
        f"color {shown_model}/{binder_chain} {hex_color(FALLBACK_BINDER_COLOR)}",
    ]

    if context["plddt_colouring"]:
        lines += [
            "",
            "# binder coloured by confidence, low to high",
            f"color bfactor {shown_model}/{binder_chain} palette alphafold",
        ]
    else:
        lines.append("")
        lines.append(
            f"# the B-factor column of this complex is flat at "
            f"{context['b_factor_range']}, so confidence colouring would say nothing"
        )

    lines += ["", "# named selections you can reuse from the command line"]
    lines.append(f"name binder {shown_model}/{binder_chain}")
    lines.append(
        f"name interface {shown_model}/{target_chain} & {shown_model}/{binder_chain} :< {cutoff}"
    )
    site_specs = [
        f"{shown_model}/{chain}:{','.join(ranges)}"
        for chain, ranges in site_residues.items()
    ]
    if site_specs:
        joined = " ".join(site_specs)
        lines.append(f"name site {joined}")
        lines.append("show site atoms")
        lines.append("style site stick")
        lines.append(f"color site {hex_color(SITE_COLOR)}")
    lines.append("show interface atoms")
    lines.append("style interface stick")

    if len(scenes) > 1:
        lines += [
            "",
            "# the other predictor arms, loaded and hidden. Show one to compare",
        ]
        for scene in scenes[1:]:
            lines.append(f"hide #{model_of[scene['object']]} models")

    lines += [
        "",
        "# view",
        f"view {shown_model}",
        "",
        "# the numbers this design was ranked on",
        f'2dlabels text "{metrics_line(candidate, shown, context["ranked_count"])}" '
        f"xpos 0.02 ypos 0.96 size 14 color black",
    ]
    for line in context["metric_block"](candidate, scenes):
        lines.append(f'log text "{line}"')

    if context["target_structure_path"]:
        lines += [
            "",
            "# The input target, last on purpose. ChimeraX stops a script at the first",
            "# command that raises, and superimposing two chains of different length can",
            "# raise. Putting it here costs only the overlay when it fails, and the log",
            "# says which command stopped.",
            f"open {context['target_structure_path']} name target_input id {target_model}",
            f"show {target_model} cartoons",
            f"color {target_model} {hex_color(REFERENCE_COLOR)}",
            f"transparency {target_model} 70 cartoons",
            'log text "Superimposing the input target onto the predicted target chain. '
            "If the next line reports an error, the scene above is complete and only the "
            f'overlay is missing; move {target_model} by hand or ignore it."',
        ]
        if declared_target_chain != target_chain:
            lines.append(
                f"# The input structure carries the target on chain {declared_target_chain}. "
                f"The prediction returns it on chain {target_chain}."
            )
        lines.append(
            f"matchmaker {target_model}/{declared_target_chain} "
            f"to {shown_model}/{target_chain}"
        )
    lines.append("")
    return "\n".join(lines)


def metric_block(candidate: dict[str, Any], scenes: list[dict[str, Any]]) -> list[str]:
    score_gating = candidate.get("score_gating", {})
    if not isinstance(score_gating, dict):
        score_gating = {}
    candidate_thresholds = score_gating.get("candidate_thresholds", [])
    control_thresholds = score_gating.get("control_thresholds", [])

    def threshold_text(records: Any) -> str:
        if not isinstance(records, list) or not records:
            return "none"
        rendered: list[str] = []
        seen: set[tuple[Any, ...]] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            key = (
                record.get("control_id"),
                record.get("metric"),
                record.get("operator"),
                record.get("threshold"),
            )
            if key in seen:
                continue
            seen.add(key)
            rendered.append(
                f"{record.get('metric')} {record.get('operator')} "
                f"{show_metric(record.get('threshold'))}"
            )
        return ", ".join(rendered) or "none"

    lines = [
        f"candidate {candidate['candidate_id']}",
        f"generator {candidate['generator']}, sequence length {candidate['sequence_length']}",
        f"rank {candidate['rank']}, rank score {show_rank_score(candidate)}",
        str(candidate.get("rank_score_scope") or "rank_score scope was not recorded"),
        f"score instrument {candidate.get('score_instrument') or 'not recorded'}",
        str(
            candidate.get("predictor_agreement")
            or "predictor agreement was not recorded"
        ),
        f"score gating {score_gating.get('mode') or 'not recorded'}: "
        f"{score_gating.get('gate_application') or 'gate application was not recorded'}",
        f"candidate thresholds {threshold_text(candidate_thresholds)}",
        f"control thresholds {threshold_text(control_thresholds)}",
        f"ensemble ipSAE_min {show_metric(candidate['ipsae_min_ensemble'])}, "
        f"sc_DockQ {show_metric(candidate['sc_dockq_ensemble'])}",
    ]
    for scene in scenes:
        lines.append(
            f"  {scene['predictor']} seed {scene['seed']}: "
            f"ipSAE_min {show_metric(scene['ipsae_min'])}, "
            f"sc_DockQ {show_metric(scene['sc_dockq'])}, DockQ {show_metric(scene['dockq'])}, "
            f"interface pLDDT {show_metric(scene['interface_plddt'], 1)}, "
            f"clashes {show_metric(scene['clash_count'], 0)}"
        )
    failed = sorted(
        name for name, passed in candidate.get("gates", {}).items() if not passed
    )
    lines.append(
        "gates: all passed" if not failed else f"gates FAILED: {', '.join(failed)}"
    )
    site_diagnostics = candidate.get("site_diagnostics") or {}
    lines.append(
        "hotspot metric "
        f"source {site_diagnostics.get('site_hotspot_source') or 'not recorded'}, "
        f"relationship {site_diagnostics.get('site_hotspot_relationship') or 'not recorded'}"
    )
    if (
        site_diagnostics.get("hotspot_recovery_duplicates_target_contact_recall")
        is True
    ):
        lines.append(
            "hotspot_recovery duplicates target_contact_recall because both use the same residue set"
        )
    return lines


def build(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    artifact_root = run_dir / "artifacts"
    status = load_json(run_dir / "status.json")
    config = load_json(run_dir / "runtime-config.resolved.json")
    ranked = load_json(artifact_root / "scores" / "ranked-candidates.json")
    observations = load_jsonl(artifact_root / "scores" / "uniform-observations.jsonl")
    index = observation_index(observations)
    viewer_target_id = resolve_viewer_target(ranked, observations)

    candidates = ranked.get("ranked_candidates", [])
    primary_target_id = ranked.get("primary_target_id")
    target = next(
        (
            t
            for t in config.get("targets", [])
            if t.get("target_id") == primary_target_id
        ),
        (config.get("targets") or [None])[0],
    )
    site = (target or {}).get("site", {})
    site_specs = list(site.get("design_residues", [])) + list(
        site.get("reference_contact_residues", [])
    )
    configured_hotspots = site.get("hotspot_residues")
    metric_hotspot_specs = (
        list(configured_hotspots)
        if isinstance(configured_hotspots, list) and configured_hotspots
        else list(site.get("reference_contact_residues", []))
    )
    first_site_diagnostics = (
        candidates[0].get("site_diagnostics", {}) if candidates else {}
    )
    target_path = (target or {}).get("structure_path")
    if target_path and not Path(target_path).is_file():
        target_path = None

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": status.get("run_id"),
        "campaign_id": ranked.get("campaign_id"),
        "generated_from": str(run_dir),
        "primary_target_id": primary_target_id,
        "target_structure_path": target_path,
        "contact_cutoff_angstrom": site.get("contact_cutoff_angstrom"),
        "site_residues": site_specs,
        "metric_hotspot_residues": metric_hotspot_specs,
        "site_hotspot_source": first_site_diagnostics.get("site_hotspot_source"),
        "site_hotspot_relationship": first_site_diagnostics.get(
            "site_hotspot_relationship"
        ),
        "hotspot_recovery_duplicates_target_contact_recall": first_site_diagnostics.get(
            "hotspot_recovery_duplicates_target_contact_recall"
        ),
        "ranking_method": ranked.get("ranking_method"),
        "score_gating": ranked.get("score_gating", {}),
        "score_instrument": ranked.get("score_instrument"),
        "predictor_agreement": ranked.get("predictor_agreement"),
        "rank_score_scope": ranked.get("rank_score_scope"),
        "ranking_claim_status": ranked.get("ranking_claim_status"),
        "best_design_claim_status": ranked.get("best_design_claim_status"),
        "best_design_claim_reason": ranked.get("best_design_claim_reason"),
        "top_candidate_tier": ranked.get("top_candidate_tier"),
        "scoring_arm_status": ranked.get("scoring_arm_status"),
        "report_mode": ranked.get("report_mode"),
        "candidate_source": (
            "supplied"
            if isinstance(config.get("supplied_candidates"), dict)
            else "generated-in-campaign"
        ),
        # The ranker declines to name a selection in report-all and raw-metrics
        # mode, and `lane.validate_terminal_run` reads exactly this pair to decide
        # the same thing.
        "names_a_selection": not (
            ranked.get("report_mode") in {"report-all", "raw-metrics"}
            and not ranked.get("selected_candidates")
        ),
        "designs": [],
        "how_to_open": {},
        "warnings": [],
    }
    score_is_defined = rank_score_is_defined(
        len(candidates), manifest.get("ranking_method")
    )
    if not score_is_defined:
        manifest["warnings"].append(
            "One design is in the normalization pool, so rank_score is 0 by construction "
            "and is reported as undefined rather than as a measurement."
        )
    if not manifest["names_a_selection"]:
        manifest["warnings"].append(
            "This run reported raw metrics and named no selected candidate, so the designs are "
            "listed in rank order without a top.pml or top.cxc alias."
        )
    # The ranker refuses to claim a best design unless a matched control outranks a
    # mismatched one, and `rank_candidates` is already forbidden from emitting the
    # phrase in that state by test_ranking_control.py:106. The viewer used to say it
    # anyway, because nothing carried the status this far. One measured run rendered
    # "To see the best design against its target" over a top pair separated by less
    # than their own run-to-run drift.
    if manifest["best_design_claim_status"] != "available":
        reason = manifest.get("best_design_claim_reason") or (
            (ranked.get("ranking_control") or {}).get("reason")
        )
        top_tier = manifest.get("top_candidate_tier")
        tier_ids = top_tier.get("candidate_ids") if isinstance(top_tier, dict) else None
        if manifest["best_design_claim_status"] == "tied" and isinstance(
            tier_ids, list
        ):
            warning = (
                "The leading candidate tier is tied within stored seed noise: "
                + ", ".join(str(candidate_id) for candidate_id in tier_ids)
                + ". The displayed order is deterministic and does not name a unique best design."
            )
        else:
            warning = (
                "A unique best-design claim is "
                f"{manifest['best_design_claim_status']}; rank 1 is an ordering under this run's "
                "declared method."
            )
        if isinstance(reason, str) and reason:
            warning += f" {reason[0].upper()}{reason[1:]}."
        manifest["warnings"].append(warning)
    tier_warning = unresolved_top_tier_warning(manifest)
    if tier_warning is not None:
        manifest["warnings"].append(tier_warning)
    if manifest["candidate_source"] == "supplied":
        manifest["warnings"].append(
            "The candidates were supplied to this campaign; this run scored them and did not "
            "generate their backbones or sequences."
        )
    manifest["warnings"].append(
        "These are computational structure-ranking results, not evidence of experimental "
        "binding, affinity, activity, or selectivity."
    )
    ranking_method = manifest.get("ranking_method")
    if (
        isinstance(ranking_method, dict)
        and "sc_dockq" in str(ranking_method.get("formula", "")).casefold()
    ):
        manifest["warnings"].append(
            "scDockQ is a structural-model score and is not an affinity measurement."
        )
    if (
        isinstance(ranking_method, dict)
        and ranking_method.get("seed_aggregation") == "max"
    ):
        bias = ranking_method.get("max_of_five_seed_bias_standard_deviations")
        bias_text = (
            f" ({float(bias):.3f} standard deviations under the stated iid normal model)"
            if isinstance(bias, (int, float)) and not isinstance(bias, bool)
            else ""
        )
        manifest["warnings"].append(
            "Taking the maximum across seeds has selection optimism"
            + bias_text
            + "; compare the recorded alternate seed reduction."
        )
    if target_path is None:
        manifest["warnings"].append(
            "the input target structure is not on this machine, so the scripts show only "
            "the target chain the predictor returned"
        )
    for disclosure in (
        ranked.get("rank_score_scope"),
        ranked.get("predictor_agreement"),
    ):
        if isinstance(disclosure, str) and disclosure:
            manifest["warnings"].append(disclosure)
    if manifest.get("hotspot_recovery_duplicates_target_contact_recall") is True:
        manifest["warnings"].append(
            "hotspot_recovery and target_contact_recall use the same residue set in this run, "
            "so they are duplicate measurements rather than independent evidence."
        )

    rows: list[str] = [
        "\t".join(
            [
                "rank",
                "candidate_id",
                "generator",
                "rank_score",
                "ipsae_min_ensemble",
                "sc_dockq_ensemble",
                "score_gating_mode",
                "score_gating",
                "score_instrument",
                "predictor_agreement",
                "rank_score_scope",
                "site_hotspot_source",
                "site_hotspot_relationship",
                "hotspot_recovery_duplicates_target_contact_recall",
                "per_seed_by_predictor",
                "shown_predictor",
                "shown_seed",
                "pymol_script",
                "chimerax_script",
            ]
        )
    ]

    for candidate in candidates:
        scenes = scenes_for_candidate(candidate, index, viewer_target_id)
        if not scenes:
            manifest["warnings"].append(
                f"{candidate['candidate_id']} has no predicted complex on disk, so it was skipped"
            )
            continue
        candidate["rank_score_defined"] = score_is_defined
        shown = scenes[0]
        observed_range = b_factor_range(Path(shown["complex_path"]))
        context = {
            "run_id": status.get("run_id"),
            "campaign_id": ranked.get("campaign_id"),
            "ranked_count": len(candidates),
            "target_structure_path": target_path,
            "contact_cutoff": site.get("contact_cutoff_angstrom", 5.0),
            # The configured specs, unparsed. Each script retargets them onto the
            # chain its own scene returned, and parses the result.
            "site_residues": list(site_specs),
            "b_factor_range": observed_range,
            "plddt_colouring": bool(
                observed_range and observed_range[1] > observed_range[0]
            ),
            "metric_block": metric_block,
        }
        stem = stable_stem(candidate['rank'], candidate['candidate_id'])
        pml_path = out_dir / f"{stem}.pml"
        cxc_path = out_dir / f"{stem}.cxc"
        pml_path.write_text(pymol_script(context, candidate, scenes))
        cxc_path.write_text(chimerax_script(context, candidate, scenes))

        manifest["designs"].append(
            {
                "rank": candidate["rank"],
                "candidate_id": candidate["candidate_id"],
                "generator": candidate["generator"],
                "rank_score": candidate["rank_score"],
                "ipsae_min_ensemble": candidate["ipsae_min_ensemble"],
                "sc_dockq_ensemble": candidate["sc_dockq_ensemble"],
                "score_gating_mode": candidate.get("score_gating_mode"),
                "score_gating": candidate.get("score_gating", {}),
                "score_instrument": candidate.get("score_instrument"),
                "predictor_agreement": candidate.get("predictor_agreement"),
                "rank_score_scope": candidate.get("rank_score_scope"),
                "site_diagnostics": candidate.get("site_diagnostics", {}),
                "per_seed_by_predictor": candidate.get("per_seed_by_predictor", {}),
                "sequence_path": candidate.get("sequence_path"),
                "design_pose_path": candidate.get("design_pose_path"),
                "gates": candidate.get("gates", {}),
                "b_factor_range": observed_range,
                "shown": {
                    "predictor": shown["predictor"],
                    "seed": shown["seed"],
                    "complex_path": shown["complex_path"],
                    "binder_chain_id": shown["binder_chain"],
                    "target_chain_id": shown["target_chain"],
                    "declared_binder_chain_id": shown.get("declared_binder_chain"),
                    "declared_target_chain_id": shown.get("declared_target_chain"),
                },
                "arms": [
                    {
                        "predictor": s["predictor"],
                        "seed": s["seed"],
                        "complex_path": s["complex_path"],
                        "binder_chain_id": s["binder_chain"],
                        "target_chain_id": s["target_chain"],
                        "declared_binder_chain_id": s.get("declared_binder_chain"),
                        "declared_target_chain_id": s.get("declared_target_chain"),
                        "ipsae_min": s["ipsae_min"],
                        "sc_dockq": s["sc_dockq"],
                    }
                    for s in scenes
                ],
                "pymol_script": str(pml_path),
                "chimerax_script": str(cxc_path),
            }
        )
        rows.append(
            "\t".join(
                [
                    str(candidate["rank"]),
                    candidate["candidate_id"],
                    candidate["generator"],
                    show_rank_score(candidate, 6),
                    show_metric(candidate["ipsae_min_ensemble"], 4),
                    show_metric(candidate["sc_dockq_ensemble"], 4),
                    str(candidate.get("score_gating_mode") or ""),
                    json.dumps(candidate.get("score_gating", {}), sort_keys=True),
                    str(candidate.get("score_instrument") or ""),
                    str(candidate.get("predictor_agreement") or ""),
                    str(candidate.get("rank_score_scope") or ""),
                    str(
                        (candidate.get("site_diagnostics") or {}).get(
                            "site_hotspot_source"
                        )
                        or ""
                    ),
                    str(
                        (candidate.get("site_diagnostics") or {}).get(
                            "site_hotspot_relationship"
                        )
                        or ""
                    ),
                    str(
                        (candidate.get("site_diagnostics") or {}).get(
                            "hotspot_recovery_duplicates_target_contact_recall"
                        )
                    ).lower(),
                    json.dumps(
                        candidate.get("per_seed_by_predictor", {}), sort_keys=True
                    ),
                    shown["predictor"],
                    str(shown["seed"]),
                    str(pml_path),
                    str(cxc_path),
                ]
            )
        )

    # `top.pml` and `top.cxc` name a winner. The ranker refuses to name one when it
    # reports raw metrics with no selected candidates, which is the state a measured run
    # was in, so writing the alias there contradicts the run's own output. In that
    # state the reader is pointed at the rank 1 script by its real name instead.
    top_pml = out_dir / "top.pml"
    top_cxc = out_dir / "top.cxc"
    manifest["top_alias_available"] = (
        manifest["best_design_claim_status"] == "available"
        and manifest["names_a_selection"]
    )
    if manifest["designs"] and manifest["top_alias_available"]:
        first = manifest["designs"][0]
        shutil.copyfile(first["pymol_script"], top_pml)
        shutil.copyfile(first["chimerax_script"], top_cxc)
        manifest["how_to_open"] = {
            "pymol": f"pymol {top_pml}",
            "chimerax": f"ChimeraX {top_cxc}",
        }
    elif manifest["designs"]:
        first = manifest["designs"][0]
        manifest["how_to_open"] = {
            "pymol": f"pymol {first['pymol_script']}",
            "chimerax": f"ChimeraX {first['chimerax_script']}",
        }

    (out_dir / "metrics.tsv").write_text("\n".join(rows) + "\n")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "README.txt").write_text(readme(manifest, out_dir))
    return manifest


def readme(manifest: dict[str, Any], out_dir: Path) -> str:
    validated = manifest.get("best_design_claim_status") == "available"
    how = manifest.get("how_to_open") or {}
    lines = [
        f"Ranked designs from run {manifest['run_id']}, campaign {manifest['campaign_id']}.",
        "",
        (
            "To see the best design against its target:"
            if validated
            else "To see the rank 1 design against its target:"
        ),
        "",
        f"    {how.get('pymol', f'pymol {out_dir}/rank-01-<candidate>.pml')}",
        f"    {how.get('chimerax', f'ChimeraX {out_dir}/rank-01-<candidate>.cxc')}",
        "",
        (
            "top.pml and top.cxc are copies of the rank 1 script. There is one script per\n"
            "ranked design beside them, named rank-NN-<candidate>."
            if manifest.get("top_alias_available")
            else "This run names no unique best design, so there is no top.pml alias. There is\n"
            "one script per ranked design, named rank-NN-<candidate>."
        ),
        "",
        "Each script loads the predicted complex for every predictor arm, shows the arm",
        "with the highest ipSAE_min, and hides the rest. The binder is coloured by the",
        "B-factor column. The target is grey. The designed site is orange sticks, and",
        "the residues within the contact cutoff of the binder are sticks as well.",
        "",
        "Named selections in both viewers: binder, interface, site.",
        "",
        "metrics.tsv is the same ranking as a table. manifest.json carries every path.",
    ]
    if manifest["warnings"]:
        lines += ["", "Warnings:"]
        lines += [f"  {w}" for w in manifest["warnings"]]
    return "\n".join(lines) + "\n"


def render_thumbnails(manifest: dict[str, Any], out_dir: Path, pymol: str) -> list[str]:
    """Ray-trace one image per design with PyMOL. ChimeraX cannot do this
    headless without OpenGL, so PyMOL renders for both."""
    written: list[str] = []
    images = out_dir / "thumbnails"
    images.mkdir(exist_ok=True)
    for design in manifest["designs"]:
        stem = stable_stem(design["rank"], design["candidate_id"])
        png = images / f"{stem}.png"
        # The script name carried only the rank, so two designs at one rank raced on it.
        script = out_dir / f".render-{stem}.pml"
        script.write_text(
            Path(design["pymol_script"]).read_text()
            + f"\npng {png}, width=1200, height=900, dpi=150, ray=1\n"
        )
        result = subprocess.run(
            [pymol, "-cq", str(script)], capture_output=True, text=True
        )
        script.unlink()
        if result.returncode == 0 and png.is_file():
            written.append(str(png))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="a finished run directory"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="where to write the scripts, default <run-dir>/viewer",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="also ray-trace one image per design with PyMOL",
    )
    parser.add_argument(
        "--pymol", default="pymol", help="the PyMOL executable for --render"
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not (run_dir / "artifacts" / "scores" / "ranked-candidates.json").is_file():
        print(
            f"error: {run_dir} has no ranked portfolio, so the run did not reach final-rank"
        )
        return 1

    out_dir = (args.out_dir or run_dir / "viewer").resolve()
    manifest = build(run_dir, out_dir)

    if not manifest["designs"]:
        print("error: no ranked design has a predicted complex on disk")
        return 1

    print(f"wrote {len(manifest['designs'])} designs to {out_dir}")
    for design in manifest["designs"]:
        print(
            f"  rank {design['rank']}  {design['candidate_id']}  "
            f"score {design['rank_score']:.4f}  "
            f"shown {design['shown']['predictor']} seed {design['shown']['seed']}"
        )
    for warning in manifest["warnings"]:
        print(f"  warning: {warning}")

    if args.render:
        written = render_thumbnails(manifest, out_dir, args.pymol)
        print(f"rendered {len(written)} images to {out_dir / 'thumbnails'}")

    print("")
    print(f"    pymol {out_dir / 'top.pml'}")
    print(f"    ChimeraX {out_dir / 'top.cxc'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
