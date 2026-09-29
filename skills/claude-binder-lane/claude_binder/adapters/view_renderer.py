#!/usr/bin/env python3
"""Build viewer scripts and render the ranked designs for a completed run."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..data.helpers.viewer.make_viewer_scripts import build
from .adapter_io import parse_declared_output, parse_render_outputs
from .publication_renderer import render_thumbnails


DEFAULT_PYMOL = "pymol"
IMAGE_KIND = "image"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class AdapterError(RuntimeError):
    """A condition the operator must fix before the render stage can run."""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    """Write JSON to a path in one atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
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


def require_renderer(
    name: str | os.PathLike[str] | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve a PyMOL command name or configured executable path."""
    requested = str(name or DEFAULT_PYMOL).strip()
    if not requested:
        requested = DEFAULT_PYMOL
    path = None if environ is None else environ.get("PATH", "")
    executable = shutil.which(os.path.expanduser(requested), path=path)
    if executable is None:
        if Path(requested).parent != Path("."):
            raise AdapterError(
                f"PyMOL executable not found: {requested}. "
                "Install PyMOL or pass --pymol with a valid executable path."
            )
        raise AdapterError(
            f"renderer executable not found on PATH: {requested}; PyMOL is required. "
            "Install PyMOL and add it to PATH, or pass --pymol with a valid executable path."
        )
    return executable


def toolcheck(args: argparse.Namespace) -> int:
    executable = require_renderer(args.pymol)
    print(f"view renderer adapter: renderer={executable}")
    return 0


def run(args: argparse.Namespace) -> int:
    executable = require_renderer(args.pymol)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    toolcheck_parser = subparsers.add_parser("toolcheck")
    toolcheck_parser.add_argument("--pymol", default=DEFAULT_PYMOL)
    run_parser = subparsers.add_parser("run")
    add_stage_arguments(run_parser)
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--pymol", default=DEFAULT_PYMOL)
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
