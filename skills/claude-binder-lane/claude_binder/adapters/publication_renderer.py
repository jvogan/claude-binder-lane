"""Render a documented PyMOL figure for a predicted protein complex.

The renderer receives chain identities and the requested target-site positions
explicitly. It never infers a requested epitope from a close contact. That
keeps predicted off-site contact markers and the requested epitope as separate
objects in both successful and missed poses.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote


DEFAULT_CONTACT_CUTOFF_ANGSTROM = 5.0
DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 1200
DEFAULT_DPI = 300
CAPTION_PANEL_HEIGHT = 220

CHAIN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
RESIDUE_TOKEN_PATTERN = re.compile(
    r"^(?:(?P<chain>[A-Za-z0-9_]+):)?(?P<start>-?\d+)(?:-(?P<end>-?\d+))?$"
)


class FigureError(ValueError):
    """Raised when figure inputs cannot make a safe PyMOL selection."""


@dataclass(frozen=True)
class FigureSpec:
    """Inputs needed to make one standalone computational-structure figure."""

    structure_path: Path
    output_path: Path
    binder_chain: str
    target_chain: str
    epitope_positions: tuple[int, ...]
    caption: str
    contact_cutoff_angstrom: float = DEFAULT_CONTACT_CUTOFF_ANGSTROM
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    dpi: int = DEFAULT_DPI
    scored_interface_present: bool | None = None


def _validate_chain_id(chain_id: str, *, field: str) -> str:
    if not CHAIN_ID_PATTERN.fullmatch(chain_id):
        raise FigureError(f"{field} must contain only letters, digits, or underscores: {chain_id!r}")
    return chain_id


def parse_epitope_positions(tokens: Iterable[str], *, target_chain: str) -> tuple[int, ...]:
    """Parse residue tokens and map their positions onto the declared target chain.

    The source of historical epitope strings can use a different chain letter
    from the predicted complex. Positions are therefore authoritative here.
    The declared ``target_chain`` controls the PyMOL selection.
    """

    _validate_chain_id(target_chain, field="target chain")
    positions: set[int] = set()
    for token in tokens:
        match = RESIDUE_TOKEN_PATTERN.fullmatch(token.strip())
        if match is None:
            raise FigureError(f"invalid epitope residue token: {token!r}")
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end < start:
            raise FigureError(f"epitope residue range descends: {token!r}")
        positions.update(range(start, end + 1))
    return tuple(sorted(positions))


def _pymol_string(value: str | Path) -> str:
    """Return a PyMOL-safe Python string literal for a path or label."""

    return repr(str(value))


def _residue_selection(chain_id: str, positions: Sequence[int]) -> str:
    if not positions:
        return "none"
    residues = "+".join(str(position) for position in positions)
    return f"complex and chain {chain_id} and resi {residues}"


def build_pymol_script(spec: FigureSpec) -> str:
    """Build a self-contained PyMOL Python script without invoking PyMOL.

    The script uses the epitope-contact selection for the camera only when
    the predicted structure puts a binder atom within the declared cutoff.
    A scorer can also declare that a pose has no interface. In that case the
    renderer retains the whole complex, suppresses the contact patch, and
    states that result in the caption.
    """

    binder_chain = _validate_chain_id(spec.binder_chain, field="binder chain")
    target_chain = _validate_chain_id(spec.target_chain, field="target chain")
    if binder_chain == target_chain:
        raise FigureError("binder chain and target chain must differ")
    if spec.contact_cutoff_angstrom <= 0:
        raise FigureError("contact cutoff must be positive")
    if spec.width <= 0 or spec.height <= 0 or spec.dpi <= 0:
        raise FigureError("image width, height, and dpi must be positive")
    if spec.scored_interface_present not in (None, True, False):
        raise FigureError("scored interface state must be true, false, or omitted")

    epitope_selection = _residue_selection(target_chain, spec.epitope_positions)
    cutoff = f"{spec.contact_cutoff_angstrom:.3f}"
    status_path = spec.output_path.with_suffix(".render.json")
    raw_height = spec.height - CAPTION_PANEL_HEIGHT
    scored_interface_present = spec.scored_interface_present is not False
    if raw_height <= 0:
        raise FigureError("image height must exceed the caption panel height")

    return f'''# Computational protein-complex figure for PyMOL.
# Target chain: {target_chain}. Binder chain: {binder_chain}.
# Epitope positions are mapped to the declared target chain.

from pymol import cmd
import json
from pathlib import Path

cmd.reinitialize()
cmd.set("ray_opaque_background", 1)
cmd.set("antialias", 2)
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_fancy_sheets", 1)
cmd.bg_color("white")
cmd.load({_pymol_string(spec.structure_path)}, "complex")

TARGET = "complex and chain {target_chain}"
BINDER = "complex and chain {binder_chain}"
EPITOPE = {_pymol_string(epitope_selection)}
CUTOFF = {cutoff}
STATUS_PATH = {_pymol_string(status_path)}
SCORED_INTERFACE_PRESENT = {scored_interface_present!r}

cmd.set_color("target_neutral", [0.42, 0.48, 0.56])
cmd.set_color("contact_patch", [0.76, 0.10, 0.52])
cmd.set_color("epitope_orange", [0.95, 0.38, 0.05])
cmd.set_color("plddt_vlow", [1.00, 0.49, 0.27])
cmd.set_color("plddt_low", [1.00, 0.86, 0.08])
cmd.set_color("plddt_high", [0.39, 0.79, 0.93])
cmd.set_color("plddt_vhigh", [0.00, 0.33, 0.70])

cmd.hide("everything", "all")
cmd.color("target_neutral", TARGET)
cmd.show("cartoon", TARGET)
cmd.show("cartoon", BINDER)

# pLDDT is stored in the structure B-factor field. The fallback stays explicit
# for structures that do not carry a plausible 0-100 confidence value.
confidence_values = []
cmd.iterate(BINDER + " and name CA", "confidence_values.append(b)", space={{"confidence_values": confidence_values}})
if confidence_values and min(confidence_values) >= 0.0 and max(confidence_values) <= 100.0:
    cmd.color("plddt_vlow", BINDER + " and b < 50")
    cmd.color("plddt_low", BINDER + " and b > 49.999 and b < 70")
    cmd.color("plddt_high", BINDER + " and b > 69.999 and b < 90")
    cmd.color("plddt_vhigh", BINDER + " and b > 89.999")
    confidence_legend = "Binder: pLDDT <50 orange, 50-70 yellow, 70-90 cyan, >=90 blue"
else:
    cmd.color("plddt_high", BINDER)
    confidence_legend = "Binder: pLDDT was unavailable in the structure file"

cmd.select("binder", BINDER)
cmd.select("target", TARGET)
cmd.select("epitope", EPITOPE)
cmd.select("geometric_target_contact_residues", "byres (" + TARGET + " within " + str(CUTOFF) + " of (" + BINDER + "))")
cmd.select("geometric_binder_contact_residues", "byres (" + BINDER + " within " + str(CUTOFF) + " of (" + TARGET + "))")
cmd.select("geometric_epitope_contact_residues", "byres (epitope within " + str(CUTOFF) + " of (" + BINDER + "))")
if SCORED_INTERFACE_PRESENT:
    cmd.select("target_contact_residues", "geometric_target_contact_residues")
    cmd.select("binder_contact_residues", "geometric_binder_contact_residues")
    cmd.select("epitope_contact_residues", "geometric_epitope_contact_residues")
else:
    cmd.select("target_contact_residues", "none")
    cmd.select("binder_contact_residues", "none")
    cmd.select("epitope_contact_residues", "none")
cmd.select("epitope_contact_region", "epitope_contact_residues or (" + BINDER + " within " + str(CUTOFF) + " of epitope_contact_residues)")

contact_patch_atom_count = cmd.count_atoms("target_contact_residues")
geometric_target_contact_atom_count = cmd.count_atoms("geometric_target_contact_residues")
epitope_contact_atom_count = cmd.count_atoms("epitope_contact_residues")
contact_patch_marker_atom_count = cmd.count_atoms("target_contact_residues and not epitope and name CA")
if contact_patch_marker_atom_count > 0:
    cmd.create("contact_patch_markers", "target_contact_residues and not epitope and name CA")
    cmd.color("contact_patch", "contact_patch_markers")
    cmd.show("spheres", "contact_patch_markers")
    cmd.set("sphere_scale", 0.44, "contact_patch_markers")

if EPITOPE != "none":
    cmd.create("epitope_markers", "epitope")
    cmd.color("epitope_orange", "epitope_markers")
    cmd.show("sticks", "epitope_markers and sidechain")
    cmd.show("spheres", "epitope_markers and name CA")
    cmd.set("stick_radius", 0.16, "epitope_markers")
    cmd.set("sphere_scale", 0.28, "epitope_markers and name CA")

if epitope_contact_atom_count > 0:
    cmd.orient("epitope_contact_region")
    cmd.center("epitope_contact_region")
    cmd.zoom("complex", buffer=1.0, complete=0)
    camera_note = "View: requested-epitope contact centered; whole complex retained."
elif not SCORED_INTERFACE_PRESENT:
    cmd.orient("complex")
    cmd.center("complex")
    cmd.zoom("complex", buffer=1.0, complete=0)
    camera_note = "View: whole complex; scorer reports no interface."
else:
    cmd.orient("complex")
    cmd.center("complex")
    cmd.zoom("complex", buffer=1.0, complete=0)
    camera_note = "View: whole complex because the requested epitope has no predicted contact at %.1f A." % CUTOFF

print("publication renderer: epitope_contact_atoms=%d" % epitope_contact_atom_count)
Path(STATUS_PATH).write_text(json.dumps({{
    "epitope_contact_atoms": epitope_contact_atom_count,
    "geometric_target_contact_atoms": geometric_target_contact_atom_count,
    "target_contact_atoms": contact_patch_atom_count,
    "contact_patch_visible": contact_patch_marker_atom_count > 0,
    "interface_state": "scored-absent" if not SCORED_INTERFACE_PRESENT else "contact-defined",
    "camera_mode": "epitope-contact-centered" if epitope_contact_atom_count > 0 else "whole-complex",
    "camera_note": camera_note,
    "confidence_legend": confidence_legend,
}}) + "\\n")
cmd.png({_pymol_string(spec.output_path)}, width={spec.width}, height={raw_height}, dpi={spec.dpi}, ray=1)
cmd.quit()
'''


def render_figure(spec: FigureSpec, pymol: str, *, script_path: Path | None = None) -> Path:
    """Write a PyMOL script, render the figure, and return the PNG path."""

    if not spec.structure_path.is_file():
        raise FigureError(f"structure file does not exist: {spec.structure_path}")
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    script = script_path or spec.output_path.with_suffix(".py")
    script.write_text(build_pymol_script(spec))
    completed = subprocess.run([pymol, "-cq", "-r", str(script)], capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not spec.output_path.is_file():
        detail = (completed.stdout + completed.stderr).strip()
        raise FigureError(
            f"PyMOL did not render {spec.output_path}: exit={completed.returncode}; {detail}"
        )
    status_path = spec.output_path.with_suffix(".render.json")
    if status_path.is_file():
        _add_caption_panel(spec, load_json(status_path))
    return spec.output_path


def load_json(path: Path) -> dict[str, object]:
    """Load a renderer sidecar that PyMOL wrote during the same render."""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise FigureError(f"render status must be a JSON object: {path}")
    return value


def _caption_lines(spec: FigureSpec, status: dict[str, object]) -> list[str]:
    """Return the factual legend placed beneath the rendered structure."""

    camera_note = status.get("camera_note")
    if isinstance(camera_note, str):
        camera_line = camera_note
    else:
        contact_atoms = status.get("epitope_contact_atoms")
        if isinstance(contact_atoms, int) and contact_atoms > 0:
            camera_line = "View: requested-epitope contact centered; whole complex retained."
        else:
            camera_line = (
                "View: whole complex because the requested epitope has no predicted contact "
                f"at {spec.contact_cutoff_angstrom:g} A."
            )
    if status.get("interface_state") == "scored-absent":
        contact_patch_line = "Magenta: omitted because the scorer reports no interface"
    elif status.get("contact_patch_visible") is True:
        contact_patch_line = "Magenta: predicted off-site target-contact residues"
    elif status.get("contact_patch_visible") is False:
        contact_patch_line = f"Magenta: no target-contact patch at {spec.contact_cutoff_angstrom:g} A"
    else:
        contact_patch_line = "Magenta: predicted off-site target-contact residues"

    return [
        f"Computed structure: {spec.caption}",
        f"Target {spec.target_chain}: neutral cartoon | Binder {spec.binder_chain}: per-residue pLDDT cartoon",
        "pLDDT: orange <50 | yellow 50-70 | cyan 70-90 | blue >=90",
        f"Orange: supplied target epitope | {contact_patch_line}",
        camera_line,
    ]


def _add_caption_panel(spec: FigureSpec, status: dict[str, object]) -> None:
    """Append an unambiguous 2D legend to a PyMOL PNG.

    PyMOL labels occupy molecular coordinates and can fall outside the image
    after an interface-oriented camera rotation. Pillow keeps captions in a
    fixed report layout. This code imports Pillow only after PyMOL has created
    the image, so script-generation tests remain independent of it.
    """

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise FigureError("Pillow is required to add the figure caption panel") from exc

    with Image.open(spec.output_path) as raw:
        canvas = Image.new("RGBA", (spec.width, spec.height), "white")
        canvas.paste(raw.convert("RGBA"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    y = spec.height - CAPTION_PANEL_HEIGHT + 26
    for line in _caption_lines(spec, status):
        draw.text((36, y), line, font=font, fill="#111827")
        y += 32
    swatches = ("#ff7d45", "#ffdb13", "#65cbf3", "#0053d6")
    x = 1320
    for color in swatches:
        draw.rounded_rectangle(
            (x, spec.height - CAPTION_PANEL_HEIGHT + 74, x + 28, spec.height - CAPTION_PANEL_HEIGHT + 102),
            radius=4,
            fill=color,
            outline="#374151",
        )
        x += 36
    canvas.save(spec.output_path, dpi=(spec.dpi, spec.dpi))


def _manifest_epitope_positions(manifest: dict[str, object], target_chain: str) -> tuple[int, ...]:
    residue_tokens = manifest.get("site_residues", [])
    if not isinstance(residue_tokens, list) or not all(isinstance(value, str) for value in residue_tokens):
        return ()
    return parse_epitope_positions(residue_tokens, target_chain=target_chain)


def _thumbnail_filename(rank: int, candidate_id: str) -> str:
    """Return a collision-safe name for one promoted thumbnail."""

    # The lane flattens image artifacts into the publication directory using
    # source.name. Preserve the candidate identity when ranks repeat or when
    # different IDs would otherwise normalize to the same safe name.
    return f"rank-{rank:03d}-{quote(candidate_id, safe='-_.')}.png"


def render_thumbnails(manifest: dict[str, object], out_dir: Path, pymol: str) -> list[str]:
    """Render publication-style PNGs for a viewer manifest.

    ``view_renderer`` uses this function after the legacy manifest builder has
    joined the ranked design records with their predicted structures.
    """

    images = out_dir / "thumbnails"
    images.mkdir(exist_ok=True)
    written: list[str] = []
    designs = manifest.get("designs", [])
    if not isinstance(designs, list):
        return written
    for design in designs:
        if not isinstance(design, dict):
            continue
        shown = design.get("shown")
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
        png = images / _thumbnail_filename(rank, candidate_id)
        script = out_dir / f".{png.stem}.py"
        try:
            epitope = _manifest_epitope_positions(manifest, target_chain)
            render_figure(
                FigureSpec(
                    structure_path=Path(structure_path),
                    output_path=png,
                    binder_chain=binder_chain,
                    target_chain=target_chain,
                    epitope_positions=epitope,
                    caption=(
                        f"Computational prediction for rank {rank}, candidate {candidate_id}, "
                        f"predictor {shown.get('predictor')}, seed {shown.get('seed')}"
                    ),
                    contact_cutoff_angstrom=float(manifest.get("contact_cutoff_angstrom") or DEFAULT_CONTACT_CUTOFF_ANGSTROM),
                ),
                pymol,
                script_path=script,
            )
        except (FigureError, TypeError, ValueError):
            continue
        written.append(str(png))
    return written


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "candidate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--structure", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--binder-chain", required=True)
    parser.add_argument("--target-chain", required=True)
    parser.add_argument("--epitope", action="append", default=[], help="CHAIN:RESI or CHAIN:START-END")
    parser.add_argument("--caption", required=True)
    parser.add_argument("--contact-cutoff", type=float, default=DEFAULT_CONTACT_CUTOFF_ANGSTROM)
    parser.add_argument("--pymol", default="pymol")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--script-out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        epitope = parse_epitope_positions(args.epitope, target_chain=args.target_chain)
        output = render_figure(
            FigureSpec(
                structure_path=args.structure.expanduser().resolve(),
                output_path=args.out.expanduser().resolve(),
                binder_chain=args.binder_chain,
                target_chain=args.target_chain,
                epitope_positions=epitope,
                caption=args.caption,
                contact_cutoff_angstrom=args.contact_cutoff,
                width=args.width,
                height=args.height,
                dpi=args.dpi,
            ),
            args.pymol,
            script_path=args.script_out.expanduser().resolve() if args.script_out else None,
        )
    except FigureError as exc:
        print(f"publication renderer: {exc}")
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
