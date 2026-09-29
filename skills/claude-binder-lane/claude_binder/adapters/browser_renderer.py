#!/usr/bin/env python3
"""Build a self-contained browser viewer for ranked protein designs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..data.helpers.viewer.make_viewer_scripts import retarget_site_specs
from ..paths import package_file
from .view_renderer import (
    AdapterError,
    add_stage_arguments,
    build as build_scripts,
    load_json,
    output_pattern,
    parse_output,
    sha256_file,
    stage_record,
    write_json,
)
from .python_raster_renderer import RasterRenderError, render_backbone_png


# The page ships a small first-party runtime, not the upstream 3Dmol.js
# distribution. 3Dmol.js is named only as the API design that inspired the
# `$3Dmol.createViewer` surface used by the generated page. That is neither a
# provenance claim nor a promise of general 3Dmol.js compatibility.
VIEWER_LIBRARY = "Claude Binder offline molecular viewer"
VIEWER_LIBRARY_VERSION = "in-tree, identified by content digest"
VIEWER_LIBRARY_LICENSE = "MIT"
VIEWER_LIBRARY_SOURCE = "data/viewer/offline-molecular-viewer.js"
VIEWER_LIBRARY_LICENSE_SOURCE = (
    "MIT notice embedded in data/viewer/offline-molecular-viewer.js"
)
VIEWER_API_REFERENCE = "3Dmol.js"
VIEWER_API_REFERENCE_SOURCE = "https://github.com/3dmol/3Dmol.js"
VIEWER_API_REFERENCE_NOTE = (
    "This first-party runtime uses the $3Dmol.createViewer name and a small "
    f"method surface inspired by {VIEWER_API_REFERENCE}. It is not the upstream "
    "distribution and does not claim general 3Dmol.js API compatibility."
)
REQUIRED_MANIFEST_FIELDS = (
    "schema_version",
    "run_id",
    "campaign_id",
    "designs",
    "how_to_open",
    "warnings",
)
ATOM_LINE_PREFIXES = ("ATOM  ", "HETATM")
SITE_PATTERN = re.compile(
    r"^(?:(?P<chain>[A-Za-z0-9_]+):)?(?P<start>-?\d+)(?:-(?P<end>-?\d+))?$"
)


PAGE_STYLE = """
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #f3f5f7; color: #20252b; }
main { max-width: 1420px; margin: 0 auto; padding: 24px; }
h1 { margin: 0 0 4px; font-size: 24px; }
.subhead { margin: 0 0 18px; color: #5e6873; font-size: 13px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: end; margin-bottom: 18px; }
label { display: grid; gap: 5px; color: #5e6873; font-size: 12px; font-weight: 650; }
select { min-width: 260px; border: 1px solid #bec7d0; border-radius: 6px; padding: 9px 10px; background: white; color: #20252b; font: inherit; }
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 18px; align-items: start; }
.viewer-shell, .panel { border: 1px solid #d7dde3; border-radius: 10px; background: white; box-shadow: 0 2px 10px rgba(31, 42, 52, .05); }
.viewer-shell { overflow: hidden; }
.viewer { height: min(70vh, 690px); min-height: 430px; position: relative; background: #fff; }
.mol-canvas { display: block; width: 100%; height: 100%; touch-action: none; cursor: grab; }
.mol-canvas:active { cursor: grabbing; }
.caption { border-top: 1px solid #e4e8ec; padding: 13px 16px 15px; font-size: 14px; line-height: 1.45; }
.caption strong { font-weight: 750; }
.panel { padding: 16px; }
.panel h2 { margin: 0 0 12px; font-size: 15px; }
.panel section + section { border-top: 1px solid #e4e8ec; margin-top: 16px; padding-top: 16px; }
.legend { display: grid; gap: 8px; margin: 0; padding: 0; list-style: none; font-size: 12px; }
.legend li { display: flex; gap: 8px; align-items: center; }
.swatch { width: 18px; height: 10px; border-radius: 3px; border: 1px solid rgba(0,0,0,.16); flex: 0 0 auto; }
.details { display: grid; grid-template-columns: 1fr auto; gap: 7px 12px; margin: 0; font-size: 12px; }
.details dt { color: #69747f; }
.details dd { margin: 0; text-align: right; font-variant-numeric: tabular-nums; }
.warning { margin-top: 14px; padding: 10px 12px; border-radius: 6px; background: #fff4dd; color: #754d00; font-size: 12px; line-height: 1.4; }
.empty { padding: 30px; color: #5e6873; }
@media (max-width: 900px) { main { padding: 14px; } .layout { grid-template-columns: 1fr; } .viewer { min-height: 360px; } .panel { order: 2; } }
"""


PAGE_SCRIPT = """
var payload = window.__CLAUDE_BINDER_VIEWER__;
var designSelect = document.getElementById("design-select");
var armSelect = document.getElementById("arm-select");
var viewerHost = document.getElementById("viewer");
var caption = document.getElementById("caption");
var detail = document.getElementById("detail");
var warnings = document.getElementById("warnings");
var viewer = null;

function text(value) { return value === null || value === undefined ? "not measured" : String(value); }
function metric(value, places) {
  if (value === null || value === undefined) return "not measured";
  var result = Number(value).toFixed(places || 3);
  return result.replace(/0+$/, "").replace(/\\.$/, "");
}
function selectedDesign() { return payload.designs[Number(designSelect.value)]; }
function selectedArm() { return selectedDesign().arms[Number(armSelect.value)]; }

function setWarnings(design) {
  warnings.textContent = "";
  var items = (payload.warnings || []).concat(design.warnings || []);
  if (!items.length) { warnings.hidden = true; return; }
  warnings.hidden = false;
  items.forEach(function (item) {
    var line = document.createElement("div");
    line.textContent = item;
    warnings.appendChild(line);
  });
}

function populateArms(design) {
  armSelect.textContent = "";
  design.arms.forEach(function (arm, index) {
    var option = document.createElement("option");
    option.value = String(index);
    option.textContent = arm.predictor + " seed " + arm.seed;
    armSelect.appendChild(option);
  });
  armSelect.value = String(Math.max(0, design.shown_arm_index || 0));
}

function updateDetails(design, arm) {
  caption.textContent = "";
  var strong = document.createElement("strong");
  strong.textContent = "Rank " + design.rank + ": " + design.candidate_id;
  caption.appendChild(strong);
  caption.appendChild(document.createTextNode(". Predictor " + arm.predictor + ", seed " + arm.seed + ". Generator " + text(design.generator) + "."));
  detail.innerHTML = "";
  var facts = [
    ["Rank score", metric(design.rank_score, 4)],
    ["Score instrument", text(design.score_instrument)],
    ["Predictor agreement", text(design.predictor_agreement)],
    ["Candidate score gate", text(design.score_gating_mode)],
    ["ipSAE ensemble", metric(design.ipsae_min_ensemble, 3)],
    ["sc_DockQ ensemble", metric(design.sc_dockq_ensemble, 3)],
    ["Binder chain", text(arm.binder_chain_id)],
    ["Target chain", text(arm.target_chain_id)],
    ["Contact cutoff", metric(arm.contact_cutoff_angstrom, 1) + " Å"],
    ["Hotspot source", text(design.site_hotspot_source)],
    ["Hotspot relationship", text(design.site_hotspot_relationship)],
    ["Off-site residues", String((arm.offsite || []).length)]
  ];
  facts.forEach(function (fact) {
    var label = document.createElement("dt"); label.textContent = fact[0];
    var value = document.createElement("dd"); value.textContent = fact[1];
    detail.appendChild(label); detail.appendChild(value);
  });
}

function draw() {
  var design = selectedDesign();
  var arm = selectedArm();
  viewerHost.textContent = "";
  viewer = $3Dmol.createViewer(viewerHost, { backgroundColor: "#ffffff" });
  viewer.addModel(arm.structure, arm.format);
  viewer.setScene({
    binder_chain: arm.binder_chain_id,
    target_chain: arm.target_chain_id,
    epitope: arm.epitope,
    offsite: arm.offsite
  });
  viewer.zoomTo();
  viewer.render();
  updateDetails(design, arm);
  setWarnings(design);
}

payload.designs.forEach(function (design, index) {
  var option = document.createElement("option");
  option.value = String(index);
  option.textContent = "Rank " + design.rank + " · " + design.candidate_id;
  designSelect.appendChild(option);
});
designSelect.addEventListener("change", function () { populateArms(selectedDesign()); draw(); });
armSelect.addEventListener("change", draw);
if (payload.designs.length) { populateArms(payload.designs[0]); draw(); }
else { viewerHost.innerHTML = "<div class=\\"empty\\">No ranked complex is available.</div>"; }
"""


def _atom_from_pdb(line: str) -> tuple[str, int, float, float, float] | None:
    if not line.startswith(ATOM_LINE_PREFIXES) or len(line) < 54:
        return None
    try:
        return (
            line[21:22].strip() or "_",
            int(line[22:26].strip()),
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
    except ValueError:
        return None


def _pdb_atoms(text: str) -> list[tuple[str, int, float, float, float]]:
    return [atom for line in text.splitlines() if (atom := _atom_from_pdb(line)) is not None]


def _cif_atoms(text: str) -> list[tuple[str, int, float, float, float]]:
    """Read the common single-line atom loop form of mmCIF."""

    atoms: list[tuple[str, int, float, float, float]] = []
    fields: list[str] = []
    in_atom_loop = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower() == "loop_":
            fields = []
            in_atom_loop = False
            continue
        if line.startswith("_atom_site."):
            fields.append(line.split()[0])
            in_atom_loop = True
            continue
        if not in_atom_loop or not fields or line.startswith("_"):
            continue
        try:
            values = shlex.split(line, comments=False, posix=True)
        except ValueError:
            continue
        if len(values) < len(fields):
            continue

        def field(*names: str, default: str = "") -> str:
            for name in names:
                if name in fields:
                    return values[fields.index(name)]
            return default

        try:
            chain = field("_atom_site.auth_asym_id", "_atom_site.label_asym_id", default="_") or "_"
            residue = int(field("_atom_site.auth_seq_id", "_atom_site.label_seq_id"))
            x = float(field("_atom_site.Cartn_x"))
            y = float(field("_atom_site.Cartn_y"))
            z = float(field("_atom_site.Cartn_z"))
        except (TypeError, ValueError):
            continue
        atoms.append((chain, residue, x, y, z))
    return atoms


def _structure_atoms(text: str) -> list[tuple[str, int, float, float, float]]:
    atoms = _pdb_atoms(text)
    return atoms if atoms else _cif_atoms(text)


def _declared_target_chain(scene: dict[str, Any]) -> str:
    """The chain the campaign declared for the target on one scene.

    The configured site residues carry this letter, and the prediction does not
    always return the target on it. Manifests written before the viewer recorded
    the declared pair carry nothing here, and an empty answer leaves the specs
    alone.
    """
    value = scene.get("declared_target_chain_id") or scene.get("declared_target_chain")
    return str(value) if isinstance(value, str) else ""


def _site_positions(specs: list[str], target_chain: str) -> dict[str, set[int]]:
    positions: dict[str, set[int]] = {}
    for raw in specs:
        match = SITE_PATTERN.fullmatch(str(raw).strip())
        if match is None:
            continue
        chain = match.group("chain") or target_chain
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end < start:
            continue
        positions.setdefault(chain, set()).update(range(start, end + 1))
    return positions


def _contact_residues(
    structure: str,
    *,
    binder_chain: str,
    target_chain: str,
    cutoff: float,
) -> set[int]:
    atoms = _structure_atoms(structure)
    binder = [atom for atom in atoms if atom[0] == binder_chain]
    target = [atom for atom in atoms if atom[0] == target_chain]
    cutoff_squared = cutoff * cutoff
    contacts: set[int] = set()
    for chain, residue, tx, ty, tz in target:
        if any(
            (tx - bx) ** 2 + (ty - by) ** 2 + (tz - bz) ** 2 <= cutoff_squared
            for _binder_chain, _binder_residue, bx, by, bz in binder
        ):
            contacts.add(residue)
    return contacts


def _read_structure(path_value: Any) -> tuple[str, str]:
    if not isinstance(path_value, str) or not path_value:
        raise AdapterError("browser renderer received a design without a structure path")
    path = Path(path_value)
    if not path.is_file():
        raise AdapterError(f"browser renderer structure is missing: {path}")
    try:
        structure = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise AdapterError(f"browser renderer structure is not UTF-8 text: {path}") from exc
    suffix = path.suffix.lower()
    return structure, "cif" if suffix in {".cif", ".mmcif"} else "pdb"


def _scene_payload(
    scene: dict[str, Any],
    *,
    site_specs: list[str],
    cutoff: float,
) -> dict[str, Any]:
    structure, format_name = _read_structure(scene.get("complex_path"))
    binder_chain = str(scene.get("binder_chain") or scene.get("binder_chain_id") or "")
    target_chain = str(scene.get("target_chain") or scene.get("target_chain_id") or "")
    # A configured spec carries its own chain prefix, so the default below never
    # fires for one and the epitope came back empty whenever the prediction
    # renamed the target chain. Retarget the specs first.
    sites = _site_positions(
        retarget_site_specs(site_specs, _declared_target_chain(scene), target_chain),
        target_chain,
    )
    epitope = sorted(sites.get(target_chain, set()))
    contacts = _contact_residues(
        structure,
        binder_chain=binder_chain,
        target_chain=target_chain,
        cutoff=cutoff,
    )
    offsite = sorted(contacts - set(epitope))
    return {
        "predictor": scene.get("predictor"),
        "seed": scene.get("seed"),
        "structure": structure,
        "format": format_name,
        "binder_chain_id": binder_chain,
        "target_chain_id": target_chain,
        "epitope": epitope,
        "offsite": offsite,
        "contact_cutoff_angstrom": cutoff,
        "ipsae_min": scene.get("ipsae_min"),
        "sc_dockq": scene.get("sc_dockq"),
    }


def _browser_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    site_specs = [str(item) for item in manifest.get("site_residues", [])]
    cutoff = float(manifest.get("contact_cutoff_angstrom") or 5.0)
    target_input = None
    if manifest.get("target_structure_path"):
        target_text, target_format = _read_structure(manifest["target_structure_path"])
        target_input = {"structure": target_text, "format": target_format}
    designs: list[dict[str, Any]] = []
    for design in manifest.get("designs", []):
        arms: list[dict[str, Any]] = []
        for scene in design.get("arms", []):
            arms.append(_scene_payload(scene, site_specs=site_specs, cutoff=cutoff))
        if not arms:
            shown = design.get("shown", {})
            arms.append(_scene_payload(shown, site_specs=site_specs, cutoff=cutoff))
        shown_predictor = design.get("shown", {}).get("predictor")
        shown_seed = design.get("shown", {}).get("seed")
        shown_index = next(
            (index for index, arm in enumerate(arms) if arm["predictor"] == shown_predictor and arm["seed"] == shown_seed),
            0,
        )
        designs.append(
            {
                "rank": design.get("rank"),
                "candidate_id": design.get("candidate_id"),
                "generator": design.get("generator"),
                "rank_score": design.get("rank_score"),
                "score_gating_mode": design.get("score_gating_mode"),
                "score_gating": design.get("score_gating", {}),
                "score_instrument": design.get("score_instrument"),
                "predictor_agreement": design.get("predictor_agreement"),
                "rank_score_scope": design.get("rank_score_scope"),
                "site_hotspot_source": (design.get("site_diagnostics") or {}).get(
                    "site_hotspot_source"
                ),
                "site_hotspot_relationship": (design.get("site_diagnostics") or {}).get(
                    "site_hotspot_relationship"
                ),
                "hotspot_recovery_duplicates_target_contact_recall": (
                    design.get("site_diagnostics") or {}
                ).get("hotspot_recovery_duplicates_target_contact_recall"),
                "per_seed_by_predictor": design.get("per_seed_by_predictor", {}),
                "ipsae_min_ensemble": design.get("ipsae_min_ensemble"),
                "sc_dockq_ensemble": design.get("sc_dockq_ensemble"),
                "shown_arm_index": shown_index,
                "arms": arms,
                "warnings": [
                    warning
                    for warning in (
                        design.get("rank_score_scope"),
                        design.get("predictor_agreement"),
                        (
                            "hotspot_recovery duplicates target_contact_recall because both "
                            "use the same residue set"
                            if (design.get("site_diagnostics") or {}).get(
                                "hotspot_recovery_duplicates_target_contact_recall"
                            )
                            else None
                        ),
                    )
                    if isinstance(warning, str) and warning
                ],
            }
        )
    return {
        "schema_version": manifest.get("schema_version"),
        "run_id": manifest.get("run_id"),
        "campaign_id": manifest.get("campaign_id"),
        "score_gating": manifest.get("score_gating", {}),
        "score_instrument": manifest.get("score_instrument"),
        "predictor_agreement": manifest.get("predictor_agreement"),
        "rank_score_scope": manifest.get("rank_score_scope"),
        "target_input": target_input,
        "designs": designs,
        "warnings": list(manifest.get("warnings", [])),
    }


def _json_script(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def build_html(manifest: dict[str, Any], *, library_source: str | None = None) -> str:
    """Return the complete offline page for a viewer manifest."""

    if library_source is None:
        library_source = package_file("data", "viewer", "offline-molecular-viewer.js").read_text(
            encoding="utf-8"
        )
    payload = _browser_payload(manifest)
    title = f"Ranked designs for {manifest.get('campaign_id') or 'campaign'}"
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{PAGE_STYLE}</style>
</head>
<body>
<main>
<h1>{title}</h1>
<p class="subhead">Run {manifest.get('run_id') or 'unknown'} · self-contained browser viewer · {VIEWER_LIBRARY}</p>
<div class="toolbar">
<label>Ranked design<select id="design-select"></select></label>
<label>Predictor arm<select id="arm-select"></select></label>
</div>
<div class="layout">
<section class="viewer-shell">
<div id="viewer" class="viewer" aria-label="Interactive molecular structure viewer"></div>
<div id="caption" class="caption"></div>
</section>
<aside class="panel">
<section><h2>Legend</h2><ul class="legend">
<li><span class="swatch" style="background:#707070"></span>Target cartoon</li>
<li><span class="swatch" style="background:#fe7d45"></span>Binder confidence below 50</li>
<li><span class="swatch" style="background:#ffdb2e"></span>Binder confidence 50 to 70</li>
<li><span class="swatch" style="background:#65c9ed"></span>Binder confidence 70 to 90</li>
<li><span class="swatch" style="background:#0053b3"></span>Binder confidence at least 90</li>
<li><span class="swatch" style="background:#ff9c12"></span>Supplied epitope</li>
<li><span class="swatch" style="background:#303030"></span>Off-site contact</li>
</ul></section>
<section><h2>Design details</h2><dl id="detail" class="details"></dl><div id="warnings" class="warning" hidden></div></section>
</aside>
</div>
</main>
<script>{library_source}</script>
<script>window.__CLAUDE_BINDER_VIEWER__={_json_script(payload)};</script>
<script>{PAGE_SCRIPT}</script>
</body>
</html>
"""
    return page


def _write_browser_manifest(manifest: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    manifest = json.loads(json.dumps(manifest))
    library_path = package_file("data", "viewer", "offline-molecular-viewer.js")
    library_bytes = library_path.read_bytes()
    manifest["renderer"] = {
        "id": "browser",
        "library": VIEWER_LIBRARY,
        "version": VIEWER_LIBRARY_VERSION,
        "license": VIEWER_LIBRARY_LICENSE,
        "license_source": VIEWER_LIBRARY_LICENSE_SOURCE,
        "source": VIEWER_LIBRARY_SOURCE,
        "library_sha256": hashlib.sha256(library_bytes).hexdigest(),
        "library_bytes": len(library_bytes),
        "api_reference": VIEWER_API_REFERENCE,
        "api_reference_source": VIEWER_API_REFERENCE_SOURCE,
        "api_reference_note": VIEWER_API_REFERENCE_NOTE,
        "page": "index.html",
    }
    manifest["how_to_open"] = {
        "browser": f"Open {out_dir / 'index.html'} in a browser. The page carries its viewer and structures inline.",
        "png_directory": str(out_dir / "thumbnails"),
    }
    manifest["warnings"] = list(manifest.get("warnings", []))
    manifest["warnings"].append(
        "The browser renderer writes one self-contained HTML page instead of server-rendered PNG thumbnails."
    )
    manifest["warnings"].append(VIEWER_API_REFERENCE_NOTE)
    manifest["warnings"].append(
        "The PNG thumbnails are rendered by a Python standard-library backbone trace and do not need a browser or PyMOL."
    )
    return manifest


def _thumbnail_filename(rank: int, candidate_id: str) -> str:
    """Return the promoted thumbnail name for one ranked candidate."""

    # The promotion layer places image files under source.name. One viewer image
    # represents the shown arm for one candidate, so its candidate ID must survive
    # that flattening. Percent encoding keeps the full identifier in one filename.
    return f"rank-{rank:03d}-{quote(candidate_id, safe='-_.')}.png"


def _render_pngs(manifest: dict[str, Any], out_dir: Path) -> list[str]:
    """Write one Python-only backbone PNG for every ranked design."""

    # These PNGs are two-dimensional backbone projections. They do not reproduce
    # the interactive browser scene or claim the browser runtime rendered them.
    thumbnails = out_dir / "thumbnails"
    written: list[str] = []
    site_specs = [str(item) for item in manifest.get("site_residues", [])]
    for design in manifest.get("designs", []):
        if not isinstance(design, dict):
            continue
        shown = design.get("shown")
        if not isinstance(shown, dict):
            arms = design.get("arms")
            shown = arms[0] if isinstance(arms, list) and arms else None
        if not isinstance(shown, dict):
            continue
        structure_path = shown.get("complex_path")
        binder_chain = shown.get("binder_chain_id")
        target_chain = shown.get("target_chain_id")
        rank = design.get("rank")
        candidate_id = design.get("candidate_id")
        if not all(isinstance(value, str) for value in (structure_path, binder_chain, target_chain, candidate_id)):
            continue
        if not isinstance(rank, int):
            continue
        output_path = thumbnails / _thumbnail_filename(rank, candidate_id)
        try:
            highlighted = _site_positions(
                retarget_site_specs(site_specs, _declared_target_chain(shown), target_chain),
                target_chain,
            )
            render_backbone_png(
                Path(structure_path),
                output_path,
                binder_chain=binder_chain,
                target_chain=target_chain,
                highlighted_residues=highlighted,
            )
        except (RasterRenderError, OSError) as exc:
            raise AdapterError(f"Python backbone renderer failed for {candidate_id}: {exc}") from exc
        design["thumbnail_path"] = str(output_path)
        written.append(str(output_path))
    return written


def toolcheck(_args: argparse.Namespace) -> int:
    library_path = package_file("data", "viewer", "offline-molecular-viewer.js")
    if library_path.stat().st_size == 0:
        raise AdapterError(f"browser viewer library is empty: {library_path}")
    print(
        f"browser viewer adapter: library={VIEWER_LIBRARY} api_reference={VIEWER_API_REFERENCE} "
        f"asset_bytes={library_path.stat().st_size}"
    )
    return 0


def run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    try:
        base_manifest = build_scripts(run_dir, out_dir)
        if not base_manifest.get("designs"):
            raise AdapterError("viewer build produced no ranked designs with predicted complexes")
        manifest = _write_browser_manifest(base_manifest, out_dir)
        library = package_file("data", "viewer", "offline-molecular-viewer.js").read_text(
            encoding="utf-8"
        )
        (out_dir / "index.html").write_text(build_html(manifest, library_source=library), encoding="utf-8")
        written = _render_pngs(manifest, out_dir)
        if len(written) != len(manifest["designs"]):
            raise AdapterError(
                f"Python backbone renderer produced {len(written)} of {len(manifest['designs'])} thumbnails"
            )
        write_json(out_dir / "manifest.json", manifest)
    except AdapterError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"browser viewer build failed: {type(exc).__name__}: {exc}") from exc
    print(
        f"browser viewer adapter: designs={len(manifest['designs'])} images={len(written)} "
        f"page={out_dir / 'index.html'} manifest={out_dir / 'manifest.json'}"
    )
    return 0


def parse(args: argparse.Namespace) -> int:
    attempt_dir = args.attempt_dir.expanduser().resolve()
    stage = stage_record(args.config.expanduser().resolve(), args.stage)
    outputs: list[dict[str, Any]] = []
    errors: list[str] = []
    for contract in stage.get("outputs", []):
        template = contract.get("path_template")
        if not isinstance(template, str):
            errors.append("render stage output has no path_template")
            continue
        try:
            pattern = output_pattern(template, attempt_dir=attempt_dir, phase=args.phase)
            matches = sorted(Path(value) for value in glob.glob(pattern, recursive=True))
            files: list[dict[str, Any]] = []
            if not matches:
                errors.append(f"{contract.get('artifact_id')} matched no files: {pattern}")
            for path in matches:
                try:
                    records = parse_output(path, str(contract.get("kind", "file")))
                    files.append(
                        {
                            "path": str(path),
                            "records": records,
                            "sha256": sha256_file(path),
                            "bytes": path.stat().st_size,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{path}: {type(exc).__name__}: {exc}")
            outputs.append(
                {
                    "artifact_id": contract.get("artifact_id"),
                    "artifact_type": contract.get("artifact_type"),
                    "kind": contract.get("kind"),
                    "pattern": pattern,
                    "count": len(files),
                    "files": files,
                }
            )
        except AdapterError as exc:
            errors.append(str(exc))
    result = {
        "ok": not errors,
        "parsed_count": sum(file_record["records"] for output in outputs for file_record in output["files"]),
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": sorted(
            file_record["sha256"] for output in outputs for file_record in output["files"]
        ),
        "phase": args.phase,
        "attempt_dir": str(attempt_dir),
        "outputs": outputs,
    }
    result_path = attempt_dir / args.phase / "parser-result.json"
    write_json(result_path, result)
    print(
        f"browser viewer adapter: parsed={result['parsed_count']} "
        f"files={len(result['source_output_hashes'])} ok={result['ok']}"
    )
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("toolcheck")
    run_parser = subparsers.add_parser("run")
    add_stage_arguments(run_parser)
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    parse_parser = subparsers.add_parser("parse")
    add_stage_arguments(parse_parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "toolcheck":
            return toolcheck(args)
        return run(args) if args.command == "run" else parse(args)
    except AdapterError as exc:
        print(f"browser viewer adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
