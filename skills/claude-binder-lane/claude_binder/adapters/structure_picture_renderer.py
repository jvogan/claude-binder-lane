#!/usr/bin/env python3
"""Render persistent overview and interface pictures for the final ranking."""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import html
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..data.helpers.viewer.make_viewer_scripts import resolved_chain_ids
from .python_raster_renderer import BackboneAtom, RasterRenderError, read_backbone, render_backbone_png


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IMAGE_KIND = "image"
RESIDUE_TOKEN = re.compile(
    r"^(?:(?P<chain>[A-Za-z0-9_]+):)?(?P<start>-?\d+)(?:-(?P<end>-?\d+))?$"
)


class PictureRenderError(RuntimeError):
    """A final-ranking structure cannot produce the required picture set."""


@dataclass(frozen=True)
class PictureInput:
    """One selected predictor structure and the final-ranking row that owns it."""

    rank: int
    candidate_id: str
    predictor: str
    seed: int | str
    structure_path: Path
    binder_chain: str
    target_chain: str
    rank_score: float | int | None
    # The chain the campaign declared for the target. It is the namespace the
    # configured hotspot residues are written in, and it is not always the chain
    # the prediction returned the target on.
    declared_target_chain: str | None = None


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PictureRenderError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PictureRenderError(f"{label} is invalid: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PictureRenderError(f"{label} must be a JSON object: {path}")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PictureRenderError(f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PictureRenderError(f"{label} cannot be read: {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PictureRenderError(f"{label} has invalid JSON at line {line_number}: {path}") from exc
        if not isinstance(row, dict):
            raise PictureRenderError(f"{label} has a non-object row at line {line_number}: {path}")
        rows.append(row)
    return rows


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "candidate"


def _numeric_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("-inf")
    return float(value)


def _structure_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return "pdb"
    if suffix in {".cif", ".mmcif"}:
        return "mmcif"
    return "detected-from-content"


def _picture_inputs(artifact_root: Path) -> list[PictureInput]:
    ranked = _load_json(artifact_root / "scores" / "ranked-candidates.json", "ranked portfolio")
    candidates = ranked.get("ranked_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PictureRenderError("ranked portfolio has no ranked_candidates")
    observations = _load_jsonl(
        artifact_root / "scores" / "uniform-observations.jsonl",
        "uniform predictor observations",
    )
    # Grouped on `(candidate_id, predictor, str(seed))` and then reduced by highest
    # ipsae_min, so a multi-target campaign's observations merged and the better-scoring
    # target's pose was drawn regardless of which target the candidate was ranked
    # against. A silent merge, one step from a wrong picture in a delivered report. The
    # key now carries the target and the ranked portfolio names which one to draw.
    picture_target_id = ranked.get("primary_target_id")
    if not isinstance(picture_target_id, str) or not picture_target_id:
        target_ids = {
            str(observation["target_id"])
            for observation in observations
            if observation.get("target_id") is not None
        }
        if len(target_ids) > 1:
            raise PictureRenderError(
                "the ranked portfolio names no primary_target_id and these uniform "
                f"observations span {len(target_ids)} targets: {sorted(target_ids)}"
            )
        picture_target_id = next(iter(target_ids), None)
    by_selection: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for observation in observations:
        candidate_id = observation.get("candidate_id")
        predictor = observation.get("predictor")
        seed = observation.get("seed")
        if not isinstance(candidate_id, str) or not isinstance(predictor, str) or seed is None:
            continue
        if observation.get("status") not in {None, "scored"}:
            continue
        row_target = observation.get("target_id")
        row_target = str(row_target) if row_target is not None else None
        if row_target is not None and picture_target_id is not None and row_target != picture_target_id:
            continue
        by_selection.setdefault((candidate_id, predictor, str(seed)), []).append(observation)

    pictures: list[PictureInput] = []
    seen_ranks: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise PictureRenderError("ranked portfolio contains a non-object candidate")
        rank = candidate.get("rank")
        candidate_id = candidate.get("candidate_id")
        selected = candidate.get("selected_seed_by_predictor")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise PictureRenderError(f"ranked candidate has an invalid rank: {rank!r}")
        if rank in seen_ranks:
            raise PictureRenderError(f"ranked portfolio repeats rank {rank}")
        seen_ranks.add(rank)
        if not isinstance(candidate_id, str) or not candidate_id:
            raise PictureRenderError(f"rank {rank} has no candidate_id")
        if not isinstance(selected, dict) or not selected:
            raise PictureRenderError(f"rank {rank} {candidate_id} has no selected predictor seed")

        matched: list[dict[str, Any]] = []
        for predictor, seed in selected.items():
            if not isinstance(predictor, str):
                continue
            matched.extend(by_selection.get((candidate_id, predictor, str(seed)), []))
        if not matched:
            raise PictureRenderError(
                f"rank {rank} {candidate_id} has no selected predicted complex in uniform observations"
            )
        shown = max(matched, key=lambda row: _numeric_score(row.get("ipsae_min")))
        structure = shown.get("predicted_complex_path")
        # The letters the returned structure uses, which are the predicted pair
        # whenever the row carries it. Drawing the declared pair against a
        # structure that renamed its chains paints the binder as the target.
        target_chain, binder_chain = resolved_chain_ids(shown)
        declared_target_chain = shown.get("target_chain_id")
        predictor = shown.get("predictor")
        seed = shown.get("seed")
        if not isinstance(structure, str) or not structure:
            raise PictureRenderError(f"rank {rank} {candidate_id} selected observation has no predicted_complex_path")
        if not all(isinstance(value, str) and value for value in (binder_chain, target_chain, predictor)):
            raise PictureRenderError(f"rank {rank} {candidate_id} selected observation has incomplete chain metadata")
        if binder_chain == target_chain:
            raise PictureRenderError(f"rank {rank} {candidate_id} gives the binder and target the same chain id")
        structure_path = Path(structure).expanduser().resolve()
        if not structure_path.is_file():
            raise PictureRenderError(f"rank {rank} {candidate_id} predicted complex is missing: {structure_path}")
        pictures.append(
            PictureInput(
                rank=rank,
                candidate_id=candidate_id,
                predictor=predictor,
                seed=seed,
                structure_path=structure_path,
                binder_chain=binder_chain,
                target_chain=target_chain,
                rank_score=candidate.get("rank_score"),
                declared_target_chain=(
                    declared_target_chain if isinstance(declared_target_chain, str) else None
                ),
            )
        )
    return sorted(pictures, key=lambda picture: picture.rank)


def _hotspot_residues(config: Mapping[str, Any]) -> dict[str, set[int]]:
    targets = config.get("targets")
    if not isinstance(targets, list):
        return {}
    primary = next(
        (target for target in targets if isinstance(target, dict) and target.get("role") == "primary"),
        None,
    )
    if not isinstance(primary, dict):
        return {}
    site = primary.get("site")
    values = site.get("design_residues", []) if isinstance(site, dict) else []
    if not isinstance(values, list):
        return {}
    hotspots: dict[str, set[int]] = {}
    for value in values:
        if not isinstance(value, str):
            continue
        match = RESIDUE_TOKEN.fullmatch(value.strip())
        if match is None or match.group("chain") is None:
            continue
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end >= start:
            hotspots.setdefault(match.group("chain"), set()).update(range(start, end + 1))
    return hotspots


def _retarget_hotspots(
    hotspots: Mapping[str, set[int]], source_chain: Any, target_chain: Any
) -> dict[str, set[int]]:
    """Move residues declared on one chain onto the chain the prediction returned.

    A campaign declares its hotspots against the target chain it asked for. A
    prediction that returns the target on another letter needs the same residue
    numbers under that letter. Residue numbers never change. Residues already
    declared on the returned chain stay, so a config naming both letters keeps
    both sets.
    """
    copied = {chain: set(residues) for chain, residues in hotspots.items()}
    if not isinstance(source_chain, str) or not isinstance(target_chain, str):
        return copied
    if not source_chain or not target_chain or source_chain == target_chain:
        return copied
    moved = copied.pop(source_chain, None)
    if moved:
        copied.setdefault(target_chain, set()).update(moved)
    return copied


def _interface_focus(
    atoms: list[BackboneAtom],
    *,
    binder_chain: str,
    target_chain: str,
    cutoff_angstrom: float,
) -> tuple[dict[str, set[int]], str]:
    binder = [atom for atom in atoms if atom[0] == binder_chain]
    target = [atom for atom in atoms if atom[0] == target_chain]
    if not binder or not target:
        raise PictureRenderError("the predicted complex lacks one requested chain")
    cutoff_squared = cutoff_angstrom * cutoff_angstrom
    focus = {binder_chain: set(), target_chain: set()}
    nearest: tuple[float, BackboneAtom, BackboneAtom] | None = None
    for target_atom in target:
        for binder_atom in binder:
            squared = sum((target_atom[index] - binder_atom[index]) ** 2 for index in (2, 3, 4))
            if squared <= cutoff_squared:
                focus[target_chain].add(target_atom[1])
                focus[binder_chain].add(binder_atom[1])
            if nearest is None or squared < nearest[0]:
                nearest = (squared, target_atom, binder_atom)
    if focus[target_chain] and focus[binder_chain]:
        return focus, "geometric-contact"
    if nearest is None:
        raise PictureRenderError("the predicted complex has no alpha-carbon pair for the interface view")
    _, target_atom, binder_atom = nearest
    focus[target_chain].add(target_atom[1])
    focus[binder_chain].add(binder_atom[1])
    return focus, "nearest-chain-pair"


def render_picture_pair(
    picture: PictureInput,
    output_dir: Path,
    *,
    hotspots: Mapping[str, set[int]],
    image_number: int,
    cutoff_angstrom: float = 5.0,
) -> dict[str, Any]:
    """Write one overview and one interface closeup for one ranked complex."""

    if cutoff_angstrom <= 0:
        raise PictureRenderError("interface contact cutoff must be positive")
    drawn_hotspots = _retarget_hotspots(
        hotspots, picture.declared_target_chain, picture.target_chain
    )
    try:
        atoms = read_backbone(picture.structure_path)
        focus, focus_method = _interface_focus(
            atoms,
            binder_chain=picture.binder_chain,
            target_chain=picture.target_chain,
            cutoff_angstrom=cutoff_angstrom,
        )
        stem = f"rank-{picture.rank:03d}-{_safe_name(picture.candidate_id)}-{image_number:03d}"
        overview = output_dir / f"{stem}-overview.png"
        interface = output_dir / f"{stem}-interface-closeup.png"
        render_backbone_png(
            picture.structure_path,
            overview,
            binder_chain=picture.binder_chain,
            target_chain=picture.target_chain,
            highlighted_residues=drawn_hotspots,
            image_id=image_number * 2,
        )
        render_backbone_png(
            picture.structure_path,
            interface,
            binder_chain=picture.binder_chain,
            target_chain=picture.target_chain,
            highlighted_residues=drawn_hotspots,
            focus_residues=focus,
            image_id=image_number * 2 + 1,
        )
    except RasterRenderError as exc:
        raise PictureRenderError(f"could not render rank {picture.rank} {picture.candidate_id}: {exc}") from exc
    overview_digest = sha256_file(overview)
    interface_digest = sha256_file(interface)
    if overview_digest == interface_digest:
        raise PictureRenderError(
            f"rank {picture.rank} {picture.candidate_id} overview and interface images have identical bytes"
        )
    return {
        "rank": picture.rank,
        "candidate_id": picture.candidate_id,
        "predictor": picture.predictor,
        "seed": picture.seed,
        "rank_score": picture.rank_score,
        "structure_path": str(picture.structure_path),
        "structure_format": _structure_format(picture.structure_path),
        "binder_chain": picture.binder_chain,
        "target_chain": picture.target_chain,
        "hotspot_residues": {
            chain: sorted(residues)
            for chain, residues in sorted(drawn_hotspots.items())
            if residues
        },
        "interface_focus": {
            "method": focus_method,
            "contact_cutoff_angstrom": cutoff_angstrom,
            "residues": {chain: sorted(residues) for chain, residues in sorted(focus.items())},
        },
        "images": [
            {"view": "overview", "path": overview.name, "sha256": overview_digest},
            {"view": "interface-closeup", "path": interface.name, "sha256": interface_digest},
        ],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _index_html(records: list[dict[str, Any]], image_dir: Path) -> str:
    cards: list[str] = []
    for record in records:
        images = {
            str(image["view"]): image_dir / str(image["path"])
            for image in record["images"]
        }
        rank = html.escape(str(record["rank"]))
        candidate = html.escape(str(record["candidate_id"]))
        predictor = html.escape(str(record["predictor"]))
        cards.append(
            "<article><h2>Rank "
            f"{rank}: {candidate}</h2><p>Predictor {predictor}, seed {html.escape(str(record['seed']))}. "
            "Target is gray. Binder is blue. Hotspots are orange.</p><div class=\"views\">"
            f"<figure><img alt=\"Overview for {candidate}\" src=\"{_image_data_uri(images['overview'])}\"><figcaption>Overview</figcaption></figure>"
            f"<figure><img alt=\"Interface closeup for {candidate}\" src=\"{_image_data_uri(images['interface-closeup'])}\"><figcaption>Interface closeup</figcaption></figure>"
            "</div></article>"
        )
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ranked structure pictures</title><style>
body{margin:0;background:#f4f6f8;color:#17202a;font:16px system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:24px}h1{margin:0 0 8px}article{background:white;border:1px solid #d7dde3;border-radius:10px;margin:18px 0;padding:16px}h2{margin:0 0 4px;font-size:18px}.views{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0}img{display:block;width:100%;height:auto;border:1px solid #d7dde3}figcaption{padding-top:6px;font-weight:650}@media(max-width:760px){.views{grid-template-columns:1fr}}
</style></head><body><main><h1>Ranked structure pictures</h1><p>Each ranked complex has an overview and an interface closeup. Download <a href="structure-pictures.zip">the PNG archive</a>.</p>""" + "".join(cards) + "</main></body></html>\n"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_archive(archive_path: Path, files: list[Path], root: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            entry = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())


def render_picture_set(
    pictures: list[PictureInput],
    output_dir: Path,
    *,
    hotspots: Mapping[str, set[int]],
    run_id: str | None,
    campaign_id: str | None,
    cutoff_angstrom: float = 5.0,
) -> dict[str, Any]:
    """Render and package all final-ranking picture pairs into one durable archive."""

    if not pictures:
        raise PictureRenderError("no predicted complexes were selected for structure pictures")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    records = [
        render_picture_pair(
            picture,
            image_dir,
            hotspots=hotspots,
            image_number=index,
            cutoff_angstrom=cutoff_angstrom,
        )
        for index, picture in enumerate(pictures)
    ]
    image_files = sorted(image_dir.glob("*.png"))
    if len(image_files) != len(pictures) * 2:
        raise PictureRenderError(
            f"rendered {len(image_files)} picture files for {len(pictures)} ranked complexes; expected {len(pictures) * 2}"
        )
    digests = [sha256_file(path) for path in image_files]
    if len(digests) != len(set(digests)):
        raise PictureRenderError("rendered structure pictures repeat file bytes")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "campaign_id": campaign_id,
        "renderer": {
            "id": "python-coordinate-projection",
            "library": "Python standard library",
            "input_formats": ["PDB", "mmCIF"],
            "views_per_complex": ["overview", "interface-closeup"],
        },
        "index_page": "index.html",
        "archive": "structure-pictures.zip",
        "image_count": len(image_files),
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    index_path = output_dir / "index.html"
    _write_json(manifest_path, manifest)
    index_path.write_text(_index_html(records, image_dir), encoding="utf-8")
    _write_archive(
        output_dir / "structure-pictures.zip",
        [*image_files, manifest_path, index_path],
        output_dir,
    )
    return manifest


def stage_record(config_path: Path, stage_id: str) -> dict[str, Any]:
    config = _load_json(config_path, "render configuration")
    matches = [stage for stage in config.get("stages", []) if stage.get("stage_id") == stage_id]
    if len(matches) != 1:
        raise PictureRenderError(
            f"structure-picture stage {stage_id} appears {len(matches)} times in {config_path}; expected one"
        )
    return matches[0]


def output_pattern(template: str, *, attempt_dir: Path, phase: str) -> str:
    rendered = template.replace("{{attempt_dir}}", str(attempt_dir)).replace("{{phase}}", phase)
    if "{{" in rendered or "}}" in rendered:
        raise PictureRenderError(f"structure-picture output has an unsupported token: {template}")
    return rendered


def parse_output(path: Path, kind: str) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        raise PictureRenderError(f"declared output is missing or empty: {path}")
    if kind == "json":
        _load_json(path, "declared JSON output")
    elif kind == IMAGE_KIND and path.read_bytes()[: len(PNG_SIGNATURE)] != PNG_SIGNATURE:
        raise PictureRenderError(f"declared image has an invalid PNG signature: {path}")
    return 1


def toolcheck(_args: argparse.Namespace) -> int:
    print("structure picture renderer: python-coordinate-projection")
    return 0


def run(args: argparse.Namespace) -> int:
    artifact_root = args.artifact_root.expanduser().resolve()
    output_dir = args.out_dir.expanduser().resolve()
    config = _load_json(args.config.expanduser().resolve(), "render configuration")
    pictures = _picture_inputs(artifact_root)
    primary = next(
        (target for target in config.get("targets", []) if isinstance(target, dict) and target.get("role") == "primary"),
        {},
    )
    site = primary.get("site", {}) if isinstance(primary, dict) else {}
    cutoff = site.get("contact_cutoff_angstrom", 5.0) if isinstance(site, dict) else 5.0
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
        raise PictureRenderError("primary target contact_cutoff_angstrom must be numeric")
    manifest = render_picture_set(
        pictures,
        output_dir,
        hotspots=_hotspot_residues(config),
        run_id=config.get("run_id") if isinstance(config.get("run_id"), str) else None,
        campaign_id=config.get("campaign_id") if isinstance(config.get("campaign_id"), str) else None,
        cutoff_angstrom=float(cutoff),
    )
    print(
        f"structure picture renderer: complexes={len(pictures)} images={manifest['image_count']} "
        f"index={output_dir / 'index.html'} archive={output_dir / 'structure-pictures.zip'}"
    )
    return 0


def parse(args: argparse.Namespace) -> int:
    attempt_dir = args.attempt_dir.expanduser().resolve()
    stage = stage_record(args.config.expanduser().resolve(), args.stage)
    outputs: list[dict[str, Any]] = []
    errors: list[str] = []
    hashes: list[str] = []
    for contract in stage.get("outputs", []):
        template = contract.get("path_template")
        if not isinstance(template, str):
            errors.append("structure-picture output has no path_template")
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
                    digest = sha256_file(path)
                    hashes.append(digest)
                    files.append(
                        {
                            "path": str(path),
                            "records": records,
                            "sha256": digest,
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
        except PictureRenderError as exc:
            errors.append(str(exc))
    if len(hashes) != len(set(hashes)):
        errors.append("declared structure-picture outputs repeat file bytes")
    result = {
        "ok": not errors,
        "parsed_count": len(hashes),
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": sorted(hashes),
        "phase": args.phase,
        "attempt_dir": str(attempt_dir),
        "outputs": outputs,
    }
    result_path = attempt_dir / args.phase / "parser-result.json"
    _write_json(result_path, result)
    print(
        f"structure picture renderer: parsed={result['parsed_count']} files={len(hashes)} ok={result['ok']}"
    )
    return 0 if result["ok"] else 1


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
    subparsers.add_parser("toolcheck")
    run_parser = subparsers.add_parser("run")
    add_stage_arguments(run_parser)
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
    except PictureRenderError as exc:
        print(f"structure picture renderer: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
