#!/usr/bin/env python3
"""Bind an operator-configured AlphaFold2-Multimer-v3 app to cofold stages.

The adapter sends one target-chain A3M, a target sequence, and a binder sequence
to the selected app. The app response must contain one mmCIF structure and a
full PAE matrix. The adapter writes the same raw prediction artifacts as the
existing ESMFold2-Fast app binding.

The adapter is real code. This repository has no deployed AlphaFold2-Multimer
application, so its presence does not identify a running predictor arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from claude_binder.clients import fal_invocation
from claude_binder.paths import package_file

from . import esmfold2_predictor as base
from . import fal_esmfold2_fast_predictor as fast_common


DEFAULT_CLIENT_PYTHON = "python3"
DEFAULT_FAL_EXECUTABLE = "fal-credential-wrapper"
DEFAULT_CLIENT = package_file("clients", "fal_alphafold2_multimer_client.py")
PREDICTOR_ID = "alphafold-multimer-v3"
ADAPTER_ID = "alphafold-multimer-v3-predictor"
DEFAULT_WORK_SUBDIR = "afm"
DEFAULT_RUN_INDEX_NAME = "afm-run-index.jsonl"
DEFAULT_MAX_SECONDS = 1800
DEFAULT_TIMEOUT_SECONDS = 1950


class AdapterError(RuntimeError):
    """A fold input or returned artifact is invalid."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"{label} is invalid: {path}: {exc}") from exc


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"{label} line {line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise AdapterError(f"{label} line {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise AdapterError(f"{label} is empty: {path}")
    return rows


def resolve_client(value: Path | None) -> Path:
    environment_value = os.environ.get("CLAUDE_BINDER_AFM_CLIENT", "").strip()
    candidate = value or (Path(environment_value) if environment_value else DEFAULT_CLIENT)
    path = candidate.expanduser().resolve()
    if not path.is_file():
        raise AdapterError(f"AlphaFold2-Multimer-v3 fal client is missing: {path}")
    return path


def run_external(argv: list[str], label: str) -> None:
    print(
        f"afm fal adapter: {label}: {fal_invocation.redacted_command(argv)}",
        flush=True,
    )
    completed = subprocess.run(argv, shell=False, check=False)
    if completed.returncode != 0:
        raise AdapterError(f"{label} exited {completed.returncode}")


def client_argv(args: argparse.Namespace, client: Path, endpoint: str, *values: str) -> list[str]:
    try:
        client_values = [endpoint, "--fal-url", args.fal_url, *values]
        credential_env = fal_invocation.credential_environment_key(
            getattr(args, "fal_credential_env", None)
        )
        route = fal_invocation.resolve_route(
            args.fal_executable,
            requested=getattr(args, "fal_credential_route", None),
            credential_env_key=credential_env,
        )
        if (
            route == fal_invocation.ROUTE_DIRECT
            and credential_env != fal_invocation.CREDENTIAL_ENVIRONMENT_KEY
        ):
            client_values.extend(["--credential-env", credential_env])
        return fal_invocation.client_command(
            args.fal_executable,
            args.client_python,
            client,
            client_values,
            requested=route,
            credential_env_key=credential_env,
        )
    except fal_invocation.RouteError as exc:
        raise AdapterError(str(exc)) from exc


def _preflight_item(
    config: dict[str, Any],
    args: argparse.Namespace,
    item: dict[str, Any],
    supplied_targets: dict[str, str],
    supplied_hotspots: dict[str, str],
    supplied_msas: dict[str, str],
) -> tuple[str, str, dict[str, Any], dict[str, str | None]]:
    target_sequence, binder_sequence, target = fast_common.preflight_item(
        config, args, item, supplied_targets, supplied_hotspots
    )
    target_id = str(target["target_id"])
    msa_value = supplied_msas.get(target_id)
    if not msa_value:
        raise AdapterError(f"no target MSA supplied for {target_id}")
    try:
        msa_identity = base.target_msa_identity(msa_value)
    except base.TargetMsaUnavailable as exc:
        raise AdapterError(str(exc)) from exc
    return target_sequence, binder_sequence, target, msa_identity


def build_plan(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    return base.plan_predictions(
        config,
        stage_id=args.stage,
        row_phase=base.campaign_phase(args.stage),
        artifact_root=args.artifact_root,
        count=args.count,
        predictor_id=PREDICTOR_ID,
    )


def run(args: argparse.Namespace) -> int:
    config = base.load_json(args.config)
    plan = build_plan(config, args)
    targets = base.resolve_per_target(args.target_sequence, config["targets"], "--target-sequence")
    hotspots = base.resolve_per_target(args.hotspot_residues, config["targets"], "--hotspot-residues")
    target_msas = base.resolve_per_target(args.target_msa_a3m, config["targets"], "--target-msa-a3m")
    preflight: dict[tuple[str, str, int], tuple[str, str, dict[str, Any], dict[str, str | None]]] = {}
    for item in plan:
        key = (
            str(item["target"]["target_id"]),
            str(item["candidate"]["candidate_id"]),
            int(item["seed"]),
        )
        preflight[key] = _preflight_item(config, args, item, targets, hotspots, target_msas)

    client = resolve_client(args.client)
    index_rows: list[dict[str, Any]] = []
    for item in plan:
        candidate = item["candidate"]
        target = item["target"]
        candidate_id = str(candidate["candidate_id"])
        target_id = str(target["target_id"])
        seed = int(item["seed"])
        out_dir = (
            args.attempt_dir.expanduser().resolve()
            / args.phase
            / args.work_subdir
            / fast_common.safe_part(candidate_id)
            / f"seed-{seed}"
        )
        binder_path = out_dir / "binder.fasta"
        target_path = out_dir / "target.fasta"
        record: dict[str, Any] = {
            "target_id": target_id,
            "candidate_id": candidate_id,
            "seed": seed,
            "out_dir": str(out_dir),
            "binder_fasta": str(binder_path),
            "target_fasta": str(target_path),
            "status": "failed",
        }
        try:
            target_sequence, binder_sequence, _, msa_identity = preflight[(target_id, candidate_id, seed)]
            msa_path = str(msa_identity["msa_path"])
            record.update(msa_identity)
            fast_common.write_fasta(binder_path, candidate_id, binder_sequence)
            fast_common.write_fasta(target_path, target_id, target_sequence)
            argv = client_argv(
                args,
                client,
                "predict",
                "--out-dir",
                str(out_dir),
                "--binder-fasta",
                str(binder_path),
                "--target-fasta",
                str(target_path),
                "--target-msa-a3m",
                msa_path,
                "--seed",
                str(seed),
                "--max-seconds",
                str(args.max_seconds),
                "--timeout-seconds",
                str(args.timeout_seconds),
            )
            run_external(argv, f"fold {candidate_id} seed {seed}")
            response = load_json(out_dir / "response.json", "AlphaFold2-Multimer-v3 response")
            record.update(fast_common.timing_ledger_fields(response))
            record["status"] = "completed"
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"afm fal adapter: {record['error']}", file=sys.stderr)
        index_rows.append(record)
    write_jsonl(args.run_index.expanduser().resolve(), index_rows)
    print(f"afm fal adapter: dispatched {len(index_rows)} folds")
    return 0 if any(record.get("status") == "completed" for record in index_rows) else 1


def toolcheck(args: argparse.Namespace) -> int:
    client = resolve_client(args.client)
    run_external(
        client_argv(args, client, "preflight", "--out-dir", str(args.out_dir.expanduser().resolve())),
        "fal preflight",
    )
    return 0


def _msa_identity_from_record(record: dict[str, Any], target_id: str) -> dict[str, str | None]:
    msa_path = record.get("msa_path")
    msa_sha256 = record.get("msa_sha256")
    if not isinstance(msa_path, str) or not msa_path or not isinstance(msa_sha256, str) or not msa_sha256:
        raise AdapterError(f"run index has no target MSA identity for {target_id}")
    try:
        identity = base.target_msa_identity(msa_path)
    except base.TargetMsaUnavailable as exc:
        raise AdapterError(str(exc)) from exc
    if identity["msa_sha256"] != msa_sha256:
        raise AdapterError(
            f"run index target MSA is stale for {target_id}: "
            f"index has {msa_sha256}, file has {identity['msa_sha256']}"
        )
    return identity


def parse_outputs(args: argparse.Namespace) -> int:
    from . import binder_contract

    config = base.load_json(args.config)
    row_phase = base.campaign_phase(args.stage)
    controls = base.control_records(config)
    model_revision = base.model_revision_for(config, ADAPTER_ID)
    manifest_path, artifacts_attempt_dir = base.output_paths(
        config, args.stage, args.attempt_dir.expanduser().resolve(), args.phase
    )
    run_index = load_jsonl(args.run_index.expanduser().resolve(), "AlphaFold2-Multimer-v3 run index")
    plan = build_plan(config, args)
    plan_by_key = {
        (str(item["target"]["target_id"]), str(item["candidate"]["candidate_id"]), int(item["seed"])): item
        for item in plan
    }
    target_sequences = base.resolve_per_target(args.target_sequence, config["targets"], "--target-sequence")
    hotspots = base.resolve_per_target(args.hotspot_residues, config["targets"], "--hotspot-residues")
    writer = base.RowWriter(manifest_path)
    errors: list[str] = []
    for record in run_index:
        key = (str(record.get("target_id")), str(record.get("candidate_id")), int(record.get("seed", 0)))
        item = plan_by_key.get(key)
        if item is None:
            errors.append(f"run index has an unplanned fold: {key}")
            continue
        try:
            msa_identity = _msa_identity_from_record(record, str(item["target"]["target_id"]))
        except AdapterError as exc:
            errors.append(str(exc))
            msa_identity = {"msa_path": None, "msa_sha256": None}
        row = base.base_row(
            config,
            item=item,
            row_phase=row_phase,
            model_revision=model_revision,
            controls=controls,
            msa_identity=msa_identity,
        )
        if record.get("status") != "completed":
            writer.write(
                base.failed_row(
                    binder_contract,
                    row,
                    failure_code="fal_request_failed",
                    failure_reason=str(record.get("error", "fal client did not complete"))[:500],
                )
            )
            continue
        out_dir = Path(str(record["out_dir"])).expanduser().resolve()
        try:
            response = load_json(out_dir / "response.json", "AlphaFold2-Multimer-v3 response")
            if not isinstance(response, dict) or not isinstance(response.get("mmcif"), str):
                raise AdapterError(f"response carries no mmcif: {out_dir / 'response.json'}")
            target_sequence = base.resolve_target_sequence(str(item["target"]["target_id"]), target_sequences)
            binder_sequence = base.read_fasta_sequence(Path(item["candidate"]["sequence_path"]))
            site_map = base.site_residue_map_for(config, item["target"], hotspots)
            written = binder_contract.write_prediction_artifacts(
                attempt_dir=artifacts_attempt_dir,
                phase=row_phase,
                run_phase=args.phase,
                target_id=row["target_id"],
                candidate_id=row["candidate_id"],
                predictor=row["predictor"],
                seed=row["seed"],
                complex_cif=response["mmcif"],
                pae=fast_common.pae_matrix(out_dir),
                chain_mapping=row["chain_mapping"],
                reference_cif=Path(row["design_pose_path"]),
                site_residue_map=site_map,
                model_revision=model_revision,
                target_sequence=target_sequence,
                binder_sequence=binder_sequence,
                extra={
                    "target_sha256": row["target_sha256"],
                    "sequence_sha256": row["sequence_sha256"],
                    "design_pose_sha256": row["design_pose_sha256"],
                    "iptm": response.get("iptm"),
                    "ptm": response.get("ptm"),
                    "mean_plddt": response.get("mean_plddt"),
                },
            )
            merged = dict(row)
            merged.update(written)
            writer.write(merged)
        except Exception as exc:  # noqa: BLE001
            writer.write(
                base.failed_row(
                    binder_contract,
                    row,
                    failure_code="prediction_parse_failed",
                    failure_reason=f"{type(exc).__name__}: {exc}"[:500],
                )
            )
    if writer.count > 0 and writer.failed == writer.count:
        errors.append("all planned predictions were rejected during parsing")
    writer.close()
    result_path = args.attempt_dir.expanduser().resolve() / args.phase / "parser-result.json"
    write_json(
        result_path,
        {
            "ok": writer.count > 0 and not errors,
            "parsed_count": writer.count,
            "rejected_count": len(errors),
            "errors": errors,
            "source_output_hashes": [sha256_file(manifest_path)] if manifest_path.is_file() else [],
        },
    )
    return 0 if writer.count > 0 and not errors else 1


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--client", type=Path, default=None)
    parser.add_argument("--client-python", default=DEFAULT_CLIENT_PYTHON)
    parser.add_argument("--fal-executable", default=DEFAULT_FAL_EXECUTABLE)
    parser.add_argument("--fal-url", required=True)
    fal_invocation.add_route_argument(parser, executable=DEFAULT_FAL_EXECUTABLE)
    fal_invocation.add_credential_environment_argument(parser)


def add_stage_arguments(parser: argparse.ArgumentParser, *, run_command: bool = False) -> None:
    for name, kwargs in (
        ("--stage", {"required": True}),
        ("--phase", {"required": True}),
        ("--count", {"type": int, "default": 1}),
        ("--attempt-dir", {"type": Path, "required": True}),
        ("--receipts-dir", {"type": Path, "required": True}),
        ("--artifact-root", {"type": Path, "required": True}),
        ("--config", {"type": Path, "required": True}),
        ("--plan", {"type": Path, "required": True}),
    ):
        parser.add_argument(name, **kwargs)
    parser.add_argument("--run-index", type=Path, required=True)
    parser.add_argument("--target-sequence", action="append")
    parser.add_argument("--hotspot-residues", action="append")
    if run_command:
        parser.add_argument("--target-msa-a3m", action="append", required=True)
        parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
        parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
        parser.add_argument("--work-subdir", default=DEFAULT_WORK_SUBDIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    toolcheck_parser = subparsers.add_parser("toolcheck")
    add_common_arguments(toolcheck_parser)
    toolcheck_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    add_common_arguments(run_parser)
    add_stage_arguments(run_parser, run_command=True)
    parse_parser = subparsers.add_parser("parse")
    add_common_arguments(parse_parser)
    add_stage_arguments(parse_parser)
    estimate_parser = subparsers.add_parser("estimate-cost")
    estimate_parser.add_argument("--fold-count", type=fast_common.nonnegative_integer, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "toolcheck":
            return toolcheck(args)
        if args.command == "parse":
            return parse_outputs(args)
        if args.command == "estimate-cost":
            return fast_common.print_cost_estimate(args.fold_count)
        return run(args)
    except AdapterError as exc:
        print(f"afm fal adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
