#!/usr/bin/env python3
"""Local open-source Boltz-2 arm of the binder lane cofold stages.

This arm runs the `boltz` console script that PyPI `boltz` installs, which is a
different route from the hosted Boltz Cloud API that `boltz2_predictor.py`
binds. The two share a model lineage and share no request, no credential, and
no output layout, so they are separate adapters rather than one adapter with a
flag.

The command it runs, from the upstream option list at `src/boltz/main.py`
lines 817 to 1041 and the output layout at `docs/prediction.md` lines 176 to
196:

    boltz predict JOB.yaml --out_dir DIR --cache CACHE --seed N \\
        --diffusion_samples 1 --write_full_pae --output_format mmcif \\
        --recycling_steps 3 --sampling_steps 200 --override

One call per seed, which is what separates this arm from the Protenix arm. The
upstream writer sorts a call's diffusion samples by confidence score descending
and names the files `model_0` upward by that rank, at
`src/boltz/data/write/writer.py` lines 74 to 76. The sample index that produced
a file is not written anywhere. So a call that asks for five samples returns
five files this arm cannot label, and one call per seed with one sample makes
`model_0` unambiguously that seed's output.

Weight provenance. Upstream `download_boltz2` fetches `boltz2_conf.ckpt` by
fixed URL with no revision and no digest, and skips the download when the file
already exists, at `src/boltz/main.py` lines 226 to 243. Nothing downstream
re-checks it. So `--checkpoint-sha256` is required on `run`, this arm hashes
the checkpoint in the cache before the first call, and it refuses the stage
when the two differ. Every row then names the bytes that folded it.

This script computes no metric of its own. `binder_contract.write_prediction_artifacts`
writes the three files and computes every measurement.

Exit code. Zero means at least one row scored. One means no row scored.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

# The arm-agnostic plumbing lives in `esmfold2_predictor.py` and is imported
# rather than copied, on the same reasoning `protenix_v2_predictor.py` gives:
# two copies of the chain-mapping rule would drift, and the drift would invert
# a score rather than raise. That module's heavy imports are all lazy.
from . import esmfold2_predictor as arm_common
from .esmfold2_predictor import (
    RowWriter,
    SiteMapUnavailable,
    TargetMsaUnavailable,
    arm_accepts_target_msa,
    base_row,
    campaign_phase,
    control_records,
    failed_row,
    load_json,
    model_revision_for,
    output_paths,
    parse_outputs,
    plan_predictions,
    read_fasta_sequence,
    resolve_per_target,
    short_reason,
    site_residue_map_for,
    target_msa_identity,
)

PREDICTOR_ID = "boltz-local"
ADAPTER_ID = "boltz-local-predictor"
ROUTE_ID = "boltz-local-cli"
ROUTE_CONTRACT_REVISION = "local-predict-v1"
SEED_SEMANTICS = "boltz_rng_seed"

DEFAULT_BOLTZ_EXECUTABLE = "boltz"

# The Modal environment mounts its weight Volume here, at
# `remote-compute-modal/envs/proteomics_boltz_gpu.py` lines 51 to 54. The
# upstream default is the same path for root, and it is passed explicitly so
# the directory this arm hashes and the directory the CLI reads are provably
# one directory.
DEFAULT_CACHE_DIR = "/root/.boltz"
CHECKPOINT_FILENAME = "boltz2_conf.ckpt"

# One diffusion sample per call, so the rank suffix the writer emits is always
# zero and always names this seed's sample. See the module docstring.
SAMPLES_PER_CALL = 1
MODEL_RANK = 0

# TODO No source in this repository states the recycling and sampling values
# the campaign should run on this route. These two are the values the hosted
# adapter sends at `boltz2_predictor.py:288`, carried here so the two Boltz
# routes are comparable. They are not a measured setting for the local CLI and
# whoever qualifies this route should settle them against a published record.
RECYCLING_STEPS = 3
SAMPLING_STEPS = 200

FAILURE_TARGET_SEQUENCE_MISSING = "target_sequence_missing"
FAILURE_BINDER_SEQUENCE_MISSING = "binder_sequence_missing"
FAILURE_TARGET_MSA_MISSING = "target_msa_missing"
FAILURE_SITE_MAP_UNAVAILABLE = "site_map_unavailable"
FAILURE_PREDICTOR_SUBPROCESS = "boltz_subprocess_failed"
FAILURE_PREDICTOR_TIMEOUT = "boltz_timeout"
FAILURE_OUTPUT_MISSING = "boltz_output_missing"
FAILURE_CHAIN_MAPPING = "chain_mapping_mismatch"
FAILURE_ARTIFACT_WRITE = "artifact_write_failed"

SLUG_CHARACTERS = re.compile(r"[^a-zA-Z0-9_.-]")

# Boltz reads chain letters from the YAML `id` field, so these two patterns
# decide what this arm is willing to put in a file it hands to another program.
CHAIN_ID_PATTERN = re.compile(r"\A[A-Za-z0-9]{1,4}\Z")
SEQUENCE_PATTERN = re.compile(r"\A[A-Z]+\Z")
SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")


class CheckpointUnverified(Exception):
    """The cached Boltz-2 checkpoint is absent or is not the pinned bytes."""


class MissingSeedOutput(Exception):
    """Boltz wrote no usable output for a seed."""


class UnserializableInput(Exception):
    """A value cannot be written into the YAML this arm hands to Boltz."""


def sanitize(value: str) -> str:
    """Return a filesystem-safe token, using the lane's own sanitiser."""
    return SLUG_CHARACTERS.sub("_", str(value))


def job_name(target_id: str, candidate_id: str, seed: int) -> str:
    """Return the stem of one Boltz call.

    Boltz derives its output directory from the input file's stem, at
    `src/boltz/main.py:1134`, and names every file inside after that same stem.
    The seed is in the stem because this arm calls once per seed and two calls
    must not share an output tree.
    """
    return f"{sanitize(target_id)}-{sanitize(candidate_id)}-seed{int(seed)}"


def checkpoint_identity(cache_dir: Path, expected_sha256: str) -> dict[str, str]:
    """Return the cached checkpoint's path and digest, or refuse.

    Upstream downloads this file with no revision and no digest and then skips
    the download whenever the file exists, so a Volume that already holds a
    checkpoint holds bytes nothing has ever identified. Hashing here is what
    turns that into a pinned input. It runs once per stage, before the first
    call, because a checkpoint that is not the pinned one invalidates every row
    the stage would go on to write.
    """
    import hashlib

    if not SHA256_PATTERN.match(expected_sha256):
        raise CheckpointUnverified(
            f"--checkpoint-sha256 must be 64 lowercase hex characters, and is {expected_sha256!r}"
        )
    path = (cache_dir / CHECKPOINT_FILENAME).expanduser()
    if not path.is_file():
        raise CheckpointUnverified(
            f"no Boltz-2 checkpoint at {path}. Hydrate the cache before running this stage."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise CheckpointUnverified(
            f"checkpoint {path} hashes to {digest}, and the stage pinned {expected_sha256}"
        )
    return {"boltz_checkpoint_path": str(path), "boltz_checkpoint_sha256": digest}


def yaml_chain_id(value: str) -> str:
    """Return a chain letter safe to write unquoted, or refuse."""
    if not CHAIN_ID_PATTERN.match(value):
        raise UnserializableInput(f"chain id {value!r} is not one to four alphanumeric characters")
    return value


def yaml_sequence(value: str, label: str) -> str:
    """Return a residue string safe to write unquoted, or refuse."""
    if not SEQUENCE_PATTERN.match(value):
        raise UnserializableInput(f"{label} is not an uppercase residue string")
    return value


def yaml_msa_path(path: Path) -> str:
    """Return a single-quoted YAML scalar for one a3m path, or refuse.

    A single quote inside the path would need doubling to stay one scalar. This
    arm has no YAML library, so it refuses such a path rather than emitting a
    file whose meaning depends on getting the escaping right.
    """
    text = str(path)
    if "'" in text:
        raise UnserializableInput(f"target a3m path contains a single quote: {text}")
    return f"'{text}'"


def write_input_yaml(
    path: Path,
    *,
    target_sequence: str,
    target_chain: str,
    binder_sequence: str,
    binder_chain: str,
    target_msa: Path | None,
) -> Path:
    """Write the Boltz-dialect input YAML for one target, candidate and seed.

    The schema is at `docs/prediction.md` lines 15 to 31. `id` fixes the chain
    letters in the output, so the predicted complex uses the letters the
    campaign configuration names rather than relying on entity order. `msa`
    takes a precomputed a3m path, and the literal `empty` is what forces
    single-sequence mode for the binder, at `docs/prediction.md` lines 76 to 77.

    Every value is validated before it reaches the file, because this arm
    composes YAML from strings rather than through a serialiser.
    """
    target_chain = yaml_chain_id(target_chain)
    binder_chain = yaml_chain_id(binder_chain)
    target_sequence = yaml_sequence(target_sequence, "target sequence")
    binder_sequence = yaml_sequence(binder_sequence, "binder sequence")
    if target_chain == binder_chain:
        raise UnserializableInput(f"target and binder share chain id {target_chain}")
    target_msa_scalar = "empty" if target_msa is None else yaml_msa_path(target_msa)
    lines = [
        "version: 1",
        "sequences:",
        "  - protein:",
        f"      id: {target_chain}",
        f"      sequence: {target_sequence}",
        f"      msa: {target_msa_scalar}",
        "  - protein:",
        f"      id: {binder_chain}",
        f"      sequence: {binder_sequence}",
        # The binder is single sequence on every arm of this campaign, and
        # `empty` is the upstream spelling for that.
        "      msa: empty",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def predictor_argv(
    executable: str,
    *,
    input_yaml: Path,
    out_dir: Path,
    cache_dir: Path,
    checkpoint_path: Path,
    seed: int,
) -> list[str]:
    """Return the argv for one `boltz predict` call.

    `--checkpoint` names the exact file this stage hashed. Without it upstream
    resolves the checkpoint from the cache itself at `src/boltz/main.py` lines
    1293 to 1296, which puts a directory lookup between the bytes this arm
    verified and the bytes the model loads. `--cache` is still passed because
    the CCD data is read from it.

    `--override` is passed because Boltz otherwise reuses whatever it finds in
    the output directory, at `docs/prediction.md:10`, which would let a retry
    report a previous call's structure as this call's result.
    """
    return [
        executable,
        "predict",
        str(input_yaml),
        "--out_dir",
        str(out_dir),
        "--cache",
        str(cache_dir),
        "--checkpoint",
        str(checkpoint_path),
        "--seed",
        str(int(seed)),
        "--diffusion_samples",
        str(SAMPLES_PER_CALL),
        "--recycling_steps",
        str(RECYCLING_STEPS),
        "--sampling_steps",
        str(SAMPLING_STEPS),
        "--output_format",
        "mmcif",
        # Without this the PAE npz is never written and the arm produces a
        # complete-looking run with no ipSAE input.
        "--write_full_pae",
        "--override",
    ]


def seed_outputs(out_dir: Path, name: str) -> tuple[Path, Path, Path]:
    """Return one call's structure, PAE and confidence files.

    The layout is `<out_dir>/boltz_results_<stem>/predictions/<stem>/`, from
    `src/boltz/main.py:1134` and `docs/prediction.md` lines 176 to 196. The
    `model_0` suffix is a confidence rank rather than a sample index. At one
    sample per call the rank and the sample coincide, which is the reason this
    arm calls once per seed.
    """
    prediction_dir = out_dir / f"boltz_results_{name}" / "predictions" / name
    if not prediction_dir.is_dir():
        raise MissingSeedOutput(f"boltz wrote no prediction directory: {prediction_dir}")
    structure = prediction_dir / f"{name}_model_{MODEL_RANK}.cif"
    pae = prediction_dir / f"pae_{name}_model_{MODEL_RANK}.npz"
    confidence = prediction_dir / f"confidence_{name}_model_{MODEL_RANK}.json"
    missing = [str(candidate) for candidate in (structure, pae, confidence) if not candidate.is_file()]
    if missing:
        raise MissingSeedOutput("boltz wrote no " + ", ".join(missing))
    return structure, pae, confidence


def pae_matrix(path: Path) -> list[list[float]]:
    """Return the PAE matrix from one `pae_*.npz`.

    The array key is `pae`, written at `src/boltz/data/write/writer.py:238`.
    The axis is tokens, and standard protein residues are one token each at
    `src/boltz/data/tokenize/boltz2.py` lines 181 to 182 and 257, so for the
    protein-only complexes this arm folds the axis is residues.
    """
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        if "pae" not in data.files:
            raise KeyError(f"{path} carries no pae array, which means --write_full_pae did not take")
        matrix = np.asarray(data["pae"], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError(f"PAE matrix is not non-empty and square: {path}: {matrix.shape}")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError(f"PAE matrix has a non-finite or negative value: {path}")
    return matrix.tolist()


def confidence_extra(path: Path) -> dict[str, Any]:
    """Return the measurement fields only the predictor knows.

    `iptm` is a required measurement field and cannot be computed from the
    complex and the PAE, so it reaches the measurement through `extra`. The key
    names are the upstream ones at `src/boltz/data/write/writer.py` lines 191
    to 200, and `complex_plddt` is what this contract calls `mean_plddt`.
    """
    document = json.loads(path.read_text())
    return {
        "iptm": arm_common.optional_float(document.get("iptm")),
        "ptm": arm_common.optional_float(document.get("ptm")),
        "mean_plddt": arm_common.optional_float(document.get("complex_plddt")),
        "boltz_confidence_score": arm_common.optional_float(document.get("confidence_score")),
        "boltz_protein_iptm": arm_common.optional_float(document.get("protein_iptm")),
    }


def verify_structure_chain_mapping(
    structure_path: Path,
    target_sequence: str,
    binder_sequence: str,
    declared_mapping: dict[str, str],
) -> None:
    """Refuse a structure whose chains are not the ones the campaign declared.

    The YAML names the letters, and this checks that the returned file honours
    them. A declared mapping and a predicted mapping are two different objects,
    and this arm is the only place that can compare them while it still knows
    which sequence it sent as which role.
    """
    from . import binder_contract

    derived = binder_contract.derive_chain_mapping(
        structure_path,
        target_sequence,
        binder_sequence,
        structure_label="Boltz predicted structure",
    )
    if derived != declared_mapping:
        error = ValueError(
            f"Boltz predicted structure mapping {derived} differs from the campaign mapping "
            f"{declared_mapping}: {structure_path}"
        )
        error.failure_code = FAILURE_CHAIN_MAPPING  # type: ignore[attr-defined]
        raise error


def group_by_seed(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one Boltz call per target, candidate and seed.

    The Protenix arm groups a candidate's seeds into one call because
    `--seeds` takes a list. Boltz `--seed` takes one integer, so the plan rows
    pass through one for one.
    """
    return [
        {
            "target": item["target"],
            "candidate": item["candidate"],
            "predictor_id": item["predictor_id"],
            "seed": int(item["seed"]),
        }
        for item in plan
    ]


def msa_identities_for(
    config: dict[str, Any],
    targets: list[dict[str, Any]],
    supplied: dict[str, str],
) -> tuple[dict[str, dict[str, str | None]], dict[str, str]]:
    """Return the alignment identity every row of this arm carries, by target.

    `msa_path` and `msa_sha256` are required on every raw prediction row. Each
    target starts at null and is replaced once its a3m resolves and hashes, so a
    target whose a3m never resolved writes null rather than naming a file the
    run never read.

    The second return is the per-target reason an a3m that was named could not
    be read. Those targets keep null, and the caller fails their rows rather
    than calling Boltz on a file it will fail on after the GPU is allocated.
    """
    identities: dict[str, dict[str, str | None]] = {}
    errors: dict[str, str] = {}
    if not arm_accepts_target_msa(config, PREDICTOR_ID):
        return identities, errors
    for target in targets:
        target_id = str(target["target_id"])
        identities[target_id] = {"msa_path": None, "msa_sha256": None}
        a3m = supplied.get(target_id)
        if not a3m:
            continue
        try:
            identities[target_id] = target_msa_identity(a3m)
        except TargetMsaUnavailable as exc:
            errors[target_id] = short_reason(exc)
    return identities, errors


def run_arm(args: argparse.Namespace) -> int:
    from . import binder_contract

    config = load_json(args.config)
    row_phase = campaign_phase(args.stage)
    controls = control_records(config)
    model_revision = model_revision_for(config, ADAPTER_ID)
    manifest_path, artifacts_attempt_dir = output_paths(
        config, args.stage, args.attempt_dir, args.phase
    )
    work_dir = artifacts_attempt_dir / args.phase / "boltz-local-runs"
    work_dir.mkdir(parents=True, exist_ok=True)

    targets = config["targets"]
    target_sequences = resolve_per_target(args.target_sequence, targets, "--target-sequence")
    msa_paths = resolve_per_target(args.target_msa_a3m, targets, "--target-msa-a3m")
    hotspots = resolve_per_target(args.hotspot_residues, targets, "--hotspot-residues")

    cache_dir = args.cache_dir.expanduser()
    # Before the plan, before the first call, and before any GPU is allocated.
    # A checkpoint that is not the pinned one invalidates every row this stage
    # would write, so it is a stage-level refusal rather than a per-row failure.
    checkpoint = checkpoint_identity(cache_dir, args.checkpoint_sha256)
    print(
        f"{PREDICTOR_ID}: checkpoint {checkpoint['boltz_checkpoint_path']} "
        f"verified at {checkpoint['boltz_checkpoint_sha256']}",
        file=sys.stderr,
        flush=True,
    )

    plan = plan_predictions(
        config,
        stage_id=args.stage,
        row_phase=row_phase,
        artifact_root=args.artifact_root,
        count=args.count,
        predictor_id=PREDICTOR_ID,
    )
    jobs = group_by_seed(plan)
    writer = RowWriter(manifest_path)
    sequence_cache: dict[str, str] = {}

    # Resolved before the first call. Hashing here rather than at fold time
    # means an unreadable a3m is knowable before `boltz predict` allocates a GPU.
    msa_identities, msa_setup_errors = msa_identities_for(config, targets, msa_paths)

    for index, job in enumerate(jobs, start=1):
        target = job["target"]
        candidate = job["candidate"]
        target_id = str(target["target_id"])
        candidate_id = str(candidate["candidate_id"])
        seed = job["seed"]
        row = base_row(
            config,
            item=job,
            row_phase=row_phase,
            model_revision=model_revision,
            controls=controls,
            msa_identity=msa_identities.get(target_id),
        )
        row["seed_semantics"] = SEED_SEMANTICS
        row["seed_source"] = "boltz predict --seed"
        row["route_id"] = ROUTE_ID
        row["route_contract_revision"] = ROUTE_CONTRACT_REVISION

        def fail(code: str, reason: str) -> None:
            writer.write(
                failed_row(binder_contract, row, failure_code=code, failure_reason=reason)
            )

        # Every per-call body is wrapped. A job that raises on candidate 3 of 40
        # otherwise throws away the GPU time spent on candidates 1 and 2.
        try:
            if target_id not in sequence_cache:
                supplied = target_sequences.get(target_id)
                if not supplied:
                    fail(
                        FAILURE_TARGET_SEQUENCE_MISSING,
                        f"no sequence for target {target_id}, pass "
                        f"--target-sequence {target_id}=SEQUENCE_OR_FASTA",
                    )
                    continue
                supplied_path = Path(supplied)
                sequence_cache[target_id] = (
                    read_fasta_sequence(supplied_path)
                    if supplied_path.exists()
                    else supplied.upper()
                )
            target_sequence = sequence_cache[target_id]

            binder_path = candidate.get("sequence_path")
            if not binder_path:
                fail(
                    FAILURE_BINDER_SEQUENCE_MISSING,
                    f"candidate {candidate_id} carries no sequence_path",
                )
                continue
            binder_sequence = read_fasta_sequence(Path(binder_path))

            a3m = msa_paths.get(target_id)
            if not a3m and not args.allow_single_sequence_target:
                fail(
                    FAILURE_TARGET_MSA_MISSING,
                    f"no target a3m for {target_id}, pass "
                    f"--target-msa-a3m {target_id}=PATH or --allow-single-sequence-target",
                )
                continue
            if target_id in msa_setup_errors:
                # An a3m was named and could not be read. Boltz would fail on
                # the same file after the container has the GPU, so this fails
                # first.
                fail(FAILURE_TARGET_MSA_MISSING, msa_setup_errors[target_id])
                continue

            # Assembled before the call. A missing site value cannot be
            # recovered afterwards, and finding it after the call wastes the GPU
            # time the call just spent.
            try:
                site_map = site_residue_map_for(config, target, hotspots)
            except SiteMapUnavailable as exc:
                fail(FAILURE_SITE_MAP_UNAVAILABLE, short_reason(exc))
                continue

            name = job_name(target_id, candidate_id, seed)
            chain_mapping = row["chain_mapping"]
            job_dir = work_dir / name
            out_dir = job_dir / "out"
            if job_dir.exists():
                # A retry of the same attempt would otherwise read the previous
                # call's files and never notice this call wrote nothing.
                shutil.rmtree(job_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            input_yaml = write_input_yaml(
                job_dir / f"{name}.yaml",
                target_sequence=target_sequence,
                target_chain=chain_mapping["target"],
                binder_sequence=binder_sequence,
                binder_chain=chain_mapping["binder"],
                target_msa=Path(a3m).resolve() if a3m else None,
            )
            argv = predictor_argv(
                args.boltz_executable,
                input_yaml=input_yaml,
                out_dir=out_dir,
                cache_dir=cache_dir,
                checkpoint_path=Path(checkpoint["boltz_checkpoint_path"]),
                seed=seed,
            )
            print(
                f"{PREDICTOR_ID}: call {index} of {len(jobs)}, candidate "
                f"{candidate_id}, seed {seed}",
                file=sys.stderr,
                flush=True,
            )
            try:
                completed = subprocess.run(
                    argv,
                    check=False,
                    timeout=args.predictor_timeout_seconds,
                    capture_output=True,
                    text=True,
                )
            except subprocess.TimeoutExpired:
                fail(
                    FAILURE_PREDICTOR_TIMEOUT,
                    f"boltz predict exceeded {args.predictor_timeout_seconds} seconds",
                )
                continue
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, flush=True)
            if completed.returncode != 0:
                tail = (completed.stderr or completed.stdout or "").strip().splitlines()
                fail(
                    FAILURE_PREDICTOR_SUBPROCESS,
                    f"boltz predict exited {completed.returncode}: "
                    f"{tail[-1] if tail else 'no output'}"[:500],
                )
                continue

            structure_path, pae_path, confidence_path = seed_outputs(out_dir, name)
            verify_structure_chain_mapping(
                structure_path, target_sequence, binder_sequence, chain_mapping
            )
            written = binder_contract.write_prediction_artifacts(
                # The two phases are never the same word. `phase` is the
                # campaign phase, which goes on the row, into the measurement
                # and into the slug. `run_phase` is this adapter's own --phase,
                # which is the directory segment that keeps a smoke run and a
                # scale run from overwriting each other.
                attempt_dir=artifacts_attempt_dir,
                phase=row["phase"],
                run_phase=args.phase,
                target_id=row["target_id"],
                candidate_id=row["candidate_id"],
                predictor=row["predictor"],
                seed=row["seed"],
                # The contract reads this path and writes the bytes into
                # complex.cif, which is a copy. A symlink into this tree is a
                # file that can vanish between the write and the hash.
                complex_cif=structure_path,
                pae=pae_matrix(pae_path),
                chain_mapping=chain_mapping,
                reference_cif=Path(row["design_pose_path"]),
                site_residue_map=site_map,
                model_revision=model_revision,
                target_sequence=target_sequence,
                binder_sequence=binder_sequence,
                extra={
                    # The first three are the contract's REQUIRED_EXTRA_FIELDS.
                    # No binder_metrics function returns them, so the arm
                    # supplies them. A name the identity block already owns
                    # raises, so nothing else from the row belongs here.
                    "target_sha256": row["target_sha256"],
                    "sequence_sha256": row["sequence_sha256"],
                    "design_pose_sha256": row["design_pose_sha256"],
                    # The checkpoint digest travels on every row, because the
                    # cache it came from can be rehydrated between stages and a
                    # stage-level check alone leaves later readers nothing to
                    # match a number back to.
                    **checkpoint,
                    **confidence_extra(confidence_path),
                },
            )
            merged = dict(row)
            merged.update(written)
            writer.write(merged)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            if isinstance(exc, MissingSeedOutput):
                code = FAILURE_OUTPUT_MISSING
            else:
                # Chain assignment failures carry their own named code. Other
                # measurement and write failures retain the generic code.
                code = getattr(exc, "failure_code", FAILURE_ARTIFACT_WRITE)
            fail(code, short_reason(exc))

    writer.close()
    scored = writer.count - writer.failed
    print(
        f"{PREDICTOR_ID}: wrote {writer.count} rows, {scored} scored, "
        f"{writer.failed} failed, to {manifest_path}",
        file=sys.stderr,
        flush=True,
    )
    # Scoring nothing is a failure. Every path above this line writes a failed
    # row and continues, so a job that lost every candidate reaches here with a
    # complete manifest and no result. The dispatcher gates the next wave on
    # this code, and a zero would close the wave on an empty set.
    #
    # A partial result returns zero. The acceptable yield varies by stage, and
    # this file has no source for a threshold, so the caller holding that
    # number applies it to the manifest.
    return 0 if scored else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Local open-source Boltz-2, target chain takes a precomputed a3m and the "
            "binder is single sequence."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("toolcheck", help="Report that the arm's entry script is installed.")
    for name in ("run", "parse"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--stage", required=True)
        subparser.add_argument("--phase", required=True)
        subparser.add_argument("--count", type=int, default=1)
        subparser.add_argument("--attempt-dir", type=Path, required=True)
        subparser.add_argument("--receipts-dir", type=Path, required=True)
        subparser.add_argument("--artifact-root", type=Path, required=True)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--plan", type=Path, required=True)
        if name != "run":
            continue
        subparser.add_argument(
            "--target-sequence",
            action="append",
            metavar="TARGET_ID=SEQUENCE_OR_FASTA",
            help=(
                "The target chain sequence, as a literal sequence or a path to a "
                "FASTA file. Repeat once per target. The campaign configuration "
                "carries no target sequence, so this argument supplies it."
            ),
        )
        subparser.add_argument(
            "--target-msa-a3m",
            action="append",
            metavar="TARGET_ID=PATH",
            help=(
                "The target-chain a3m, written into the YAML as the protein "
                "entity's msa path. Repeat once per target."
            ),
        )
        subparser.add_argument(
            "--hotspot-residues",
            action="append",
            metavar="TARGET_ID=CHAIN:NUMBER,CHAIN:START-END",
            help=(
                "The target's hotspot residues, which `hotspot_recovery` is "
                "measured against. The campaign configuration carries no "
                "hotspot field, so this argument supplies it."
            ),
        )
        subparser.add_argument(
            "--checkpoint-sha256",
            required=True,
            metavar="HEX",
            help=(
                "The sha256 of the Boltz-2 checkpoint this stage is pinned to. "
                "Required, because upstream downloads the checkpoint with no "
                "revision and no digest and then skips the download whenever "
                "the file exists. The stage refuses when the cache holds "
                "anything else."
            ),
        )
        subparser.add_argument(
            "--cache-dir",
            type=Path,
            default=Path(DEFAULT_CACHE_DIR),
            help=(
                "The Boltz weight and CCD cache, passed to the CLI as --cache "
                f"and hashed before the first call. Defaults to {DEFAULT_CACHE_DIR}, "
                "which is where the shipped Modal environment mounts its Volume."
            ),
        )
        subparser.add_argument(
            "--allow-single-sequence-target",
            action="store_true",
            help=(
                "Fold a target with no a3m by writing `msa: empty` for the "
                "target chain as well. Upstream calls single-sequence mode not "
                "recommended, and it changes what the arm measures, so a "
                "campaign has to ask for it."
            ),
        )
        subparser.add_argument(
            "--boltz-executable",
            default=DEFAULT_BOLTZ_EXECUTABLE,
            help=(
                "The console script PyPI `boltz` installs. This is the local "
                f"open-source route. Defaults to {DEFAULT_BOLTZ_EXECUTABLE}."
            ),
        )
        subparser.add_argument(
            "--predictor-timeout-seconds",
            type=int,
            default=None,
            help=(
                "Timeout for one `boltz predict` call. Unset by default, so the "
                "stage timeout governs."
            ),
        )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "toolcheck":
        # The profile's toolcheck_argv is `pip show boltz`, because importing
        # the package touches CUDA.
        print(f"{ADAPTER_ID} entry script ok, route {ROUTE_ID}")
        return 0
    if args.command == "parse":
        return parse_outputs(args)
    return run_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())
