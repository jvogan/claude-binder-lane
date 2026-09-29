#!/usr/bin/env python3
"""Build viewer scripts and render ranked designs with UCSF ChimeraX."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from urllib.parse import quote
from pathlib import Path
from typing import Any, Iterable

from ..data.helpers.viewer.make_viewer_scripts import build
from .adapter_io import parse_declared_output, parse_render_outputs


DEFAULT_CHIMERAX = "/Applications/ChimeraX-1.11.1.app/Contents/bin/ChimeraX"
DEFAULT_CONTACT_CUTOFF_ANGSTROM = 5.0
DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 1200
IMAGE_KIND = "image"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CHAIN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
RESIDUE_TOKEN_PATTERN = re.compile(
    r"^(?:(?P<chain>[A-Za-z0-9_]+):)?(?P<start>-?\d+)(?:-(?P<end>-?\d+))?$"
)


class AdapterError(RuntimeError):
    """A condition the operator must fix before the render stage can run."""


class FigureError(ValueError):
    """Raised when figure inputs cannot make a safe ChimeraX command script."""


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    """Write JSON to a path in one atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_record(config_path: Path, stage_id: str) -> dict[str, Any]:
    config = load_json(config_path)
    matches = [stage for stage in config.get("stages", []) if stage.get("stage_id") == stage_id]
    if len(matches) != 1:
        raise AdapterError(
            f"render stage {stage_id} appears {len(matches)} times in {config_path}; expected one"
        )
    return matches[0]


def output_pattern(template: str, *, attempt_dir: Path, phase: str) -> str:
    rendered = template.replace("{{attempt_dir}}", str(attempt_dir)).replace("{{phase}}", phase)
    if "{{" in rendered or "}}" in rendered:
        raise AdapterError(f"render stage output path carries an unsupported token: {template}")
    return rendered


def require_renderer(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise AdapterError(f"renderer executable not found on PATH: {name}")
    return executable


def _validate_chain_id(chain_id: str, *, field: str) -> str:
    if not CHAIN_ID_PATTERN.fullmatch(chain_id):
        raise FigureError(f"{field} must contain only letters, digits, or underscores: {chain_id!r}")
    return chain_id


def parse_epitope_positions(tokens: Iterable[str], *, target_chain: str) -> tuple[int, ...]:
    """Parse residue tokens and map their positions onto the target chain."""

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


def _chimerax_path(path: Path) -> str:
    """Return a quoted ChimeraX command argument for a file path."""

    value = str(path)
    if "\n" in value or "\r" in value:
        raise FigureError(f"ChimeraX file path contains a line break: {path}")
    value = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value}"'


def _chain_spec(chain_id: str) -> str:
    return f"#1/{chain_id}"


def _epitope_spec(chain_id: str, positions: tuple[int, ...]) -> str | None:
    if not positions:
        return None
    residues = ",".join(str(position) for position in positions)
    return f"{_chain_spec(chain_id)}:{residues}"


def build_chimerax_script(spec: FigureSpec) -> str:
    """Build a self-contained ChimeraX command script without running ChimeraX."""

    binder_chain = _validate_chain_id(spec.binder_chain, field="binder chain")
    target_chain = _validate_chain_id(spec.target_chain, field="target chain")
    if binder_chain == target_chain:
        raise FigureError("binder chain and target chain must differ")
    if spec.contact_cutoff_angstrom <= 0:
        raise FigureError("contact cutoff must be positive")
    if spec.width <= 0 or spec.height <= 0:
        raise FigureError("image width and height must be positive")

    binder = _chain_spec(binder_chain)
    target = _chain_spec(target_chain)
    epitope = _epitope_spec(target_chain, spec.epitope_positions)
    cutoff = f"{spec.contact_cutoff_angstrom:.3f}"
    contact = f"(({binder} :< {cutoff}) & {target})"
    caption = spec.caption.replace("\r", " ").replace("\n", " ")
    epitope_commands = (
        [
            f"color {epitope} #f25f0a target ac",
            f"show {epitope} atoms",
            f"style {epitope} stick",
        ]
        if epitope is not None
        else ["# The render inputs contain no requested epitope positions."]
    )
    lines = [
        "# Computational protein-complex figure for UCSF ChimeraX.",
        f"# Target chain: {target_chain}. Binder chain: {binder_chain}.",
        f"# Caption: {caption}",
        "close all",
        f"open {_chimerax_path(spec.structure_path)}",
        "hide #1 atoms",
        "cartoon #1",
        f"color {target} #6b7785",
        f"color bfactor {binder} palette alphafold range 0,100",
        f"surface {target}",
        f"transparency {target} 35 target s",
        f"color {contact} #c21f6f target as",
        f"show {contact} atoms",
        f"style {contact} stick",
        *epitope_commands,
        "set bgColor white",
        "lighting soft",
        "graphics silhouettes true width 1.5",
        "view #1",
        (
            f"save {_chimerax_path(spec.output_path)} width {spec.width} "
            f"height {spec.height} supersample 3 transparentBackground false"
        ),
        "exit",
    ]
    return "\n".join(lines) + "\n"


def _render_argv(chimerax: str, script: Path) -> list[str]:
    options = ["--nocolor", "--notools", "--exit", "--script", str(script)]
    if sys.platform == "darwin":
        return [chimerax, *options]
    return [chimerax, "--offscreen", *options]


def render_figure(spec: FigureSpec, chimerax: str, *, script_path: Path | None = None) -> Path:
    """Write a ChimeraX script, render the figure, and return the PNG path."""

    if not spec.structure_path.is_file():
        raise FigureError(f"structure file does not exist: {spec.structure_path}")
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    script = script_path or spec.output_path.with_suffix(".cxc")
    script.write_text(build_chimerax_script(spec))
    completed = subprocess.run(
        _render_argv(chimerax, script), capture_output=True, text=True, check=False
    )
    if completed.returncode != 0 or not spec.output_path.is_file():
        detail = (completed.stdout + completed.stderr).strip()
        raise FigureError(
            f"ChimeraX did not render {spec.output_path}: exit={completed.returncode}; {detail}"
        )
    return spec.output_path


def _manifest_epitope_positions(
    manifest: dict[str, object], target_chain: str
) -> tuple[int, ...]:
    residue_tokens = manifest.get("site_residues", [])
    if not isinstance(residue_tokens, list) or not all(
        isinstance(value, str) for value in residue_tokens
    ):
        return ()
    return parse_epitope_positions(residue_tokens, target_chain=target_chain)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "candidate"


def render_thumbnails(manifest: dict[str, object], out_dir: Path, chimerax: str) -> list[str]:
    """Render ChimeraX PNGs for a viewer manifest."""

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
        if not all(
            isinstance(value, str)
            for value in (structure_path, binder_chain, target_chain, candidate_id)
        ):
            continue
        if not isinstance(rank, int):
            continue
        # `_safe_name` collapses every run of non-word characters to one underscore, so two
        # candidate ids can normalize to one name, and nothing enforces a unique rank. The
        # count check below appends once per design either way, so a collision overwrote one
        # PNG and still passed. `publication_renderer` and `browser_renderer` were fixed for
        # this in d5fd1df and bd48bec; this renderer was not. The script name carried only the
        # rank, so two designs at one rank raced on that file too.
        stable_name = quote(candidate_id, safe="-_.")
        png = images / f"rank-{rank:02d}-{stable_name}.png"
        script = out_dir / f".render-rank-{rank:02d}-{stable_name}.cxc"
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
                    contact_cutoff_angstrom=float(
                        manifest.get("contact_cutoff_angstrom")
                        or DEFAULT_CONTACT_CUTOFF_ANGSTROM
                    ),
                ),
                chimerax,
                script_path=script,
            )
        except (FigureError, TypeError, ValueError):
            continue
        written.append(str(png))
    return written


def toolcheck(args: argparse.Namespace) -> int:
    executable = require_renderer(args.chimerax)
    print(f"view renderer adapter: renderer={executable}")
    return 0


def run(args: argparse.Namespace) -> int:
    executable = require_renderer(args.chimerax)
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    try:
        manifest = build(run_dir, out_dir)
        designs = manifest.get("designs", [])
        if not designs:
            raise AdapterError("viewer build produced no ranked designs with predicted complexes")
        written = render_thumbnails(manifest, out_dir, executable)
    except AdapterError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"viewer build or render failed: {type(exc).__name__}: {exc}") from exc
    if len(written) != len(designs):
        raise AdapterError(
            f"renderer produced {len(written)} of {len(designs)} thumbnails under "
            f"{out_dir / 'thumbnails'}"
        )
    print(
        f"view renderer adapter: designs={len(designs)} images={len(written)} "
        f"manifest={out_dir / 'manifest.json'}"
    )
    return 0


def parse_output(path: Path, kind: str) -> int:
    return parse_declared_output(
        path,
        kind,
        load_json=load_json,
        json_loads=json.loads,
        image_kind=IMAGE_KIND,
        png_signature=PNG_SIGNATURE,
        error_type=AdapterError,
    )


def parse(args: argparse.Namespace) -> int:
    return parse_render_outputs(
        args,
        stage_record=stage_record,
        output_pattern=output_pattern,
        parse_output=parse_output,
        sha256_file=sha256_file,
        write_json=write_json,
        path_factory=Path,
        glob_matches=glob.glob,
        error_type=AdapterError,
    )


def add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)


def _add_renderer_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chimerax",
        "--pymol",
        dest="chimerax",
        default=DEFAULT_CHIMERAX,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    toolcheck_parser = subparsers.add_parser("toolcheck")
    _add_renderer_argument(toolcheck_parser)
    run_parser = subparsers.add_parser("run")
    add_stage_arguments(run_parser)
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    _add_renderer_argument(run_parser)
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
        print(f"view renderer adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
