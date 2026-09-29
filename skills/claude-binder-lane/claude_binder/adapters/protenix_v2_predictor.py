#!/usr/bin/env python3
"""Protenix v2 arm of the binder lane cofold stages.

This arm has a real command line, which is what separates it from the two
ESMFold2 arms. It writes an input JSON per target and candidate, runs
`protenix pred` once for all of that candidate's seeds, then walks the output
tree and writes one raw prediction row per seed.

The command it runs, from `ref/docs/LOOKUP_TABLES.md`, first table, row `ptxv2`,
and from protocol line 71:

    protenix pred --input in.json --out_dir DIR --model_name protenix-v2 \\
        --seeds 0,1,2,3,4 --cycle 10 --sample 1 \\
        --need_atom_confidence true --use_msa true

`--step` and `--dtype` are not passed, by instruction. Their CLI defaults are
200 diffusion steps and bf16, which are the values the published campaign ran.
`--sample 1` is passed, and it is the one value the protocol's own command line
leaves at the CLI default of 5. The published row settles it at 1.

`--need_atom_confidence true` is what puts the PAE on disk. Without it the arm
produces a complete-looking run with no ipSAE input.

This script computes no metric of its own. `binder_contract.write_prediction_artifacts`
writes the three files and computes every measurement.

Every row carries `msa_path` and `msa_sha256`, which is the contract recorded at
`lane.RAW_PREDICTION_FIELDS`. This arm is wired to the shared target MSA, so its
rows name the unpaired a3m the run read and its digest. A target whose a3m never
resolved writes null in both.

Exit code. Zero means at least one row scored. One means no row scored, so a
job whose every input was missing and a job whose every call failed both exit
one. A partial result exits zero. A caller that needs a minimum yield reads the
manifest and applies its own threshold.
"""

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
# rather than copied. Two copies of the chain-mapping rule would drift, and the
# drift would be caught late and would invert a score rather than raise. That
# module's own heavy imports are all lazy, so importing it here pulls in nothing
# but the standard library.
# TODO Move the shared plumbing to a module of its own, named for what it is
# rather than for one arm. This lane was scoped to three files, so the move
# belongs to whoever owns the adapters directory next.
from . import esmfold2_predictor as arm_common
from .esmfold2_predictor import (
    RowWriter,
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
    SiteMapUnavailable,
    site_residue_map_for,
    target_msa_identity,
    TargetMsaUnavailable,
)

PREDICTOR_ID = "protenix-v2"
ADAPTER_ID = "protenix-v2-predictor"
MODEL_NAME = "protenix-v2"

# Inference parameters, from the published `ptxv2` row.
CYCLE = 10
SAMPLES_PER_SEED = 1

# TODO Cofactors and nucleic acids are not modelled here. The published row says
# RBX1 takes CCD_ZN count 3, 15-PGDH takes one CCD_NAD per HPGD protomer, and
# the Cas9 RNP takes the sgRNA as an `rnaSequence` entity between the target and
# binder chains. The campaign configuration carries no field for any of them, so
# there is nothing to read them from. The real value comes from whoever adds
# those fields to the campaign schema.

# TODO The entity order for the two exceptions is not modelled. The published
# row says the binder entity sits between the protomers for part of the
# TNF-alpha 1to3 and VEGF-A 1to2 designs. This script writes target first and
# binder second, which is the row's stated convention. The exception needs a
# per-design field the configuration does not carry.

FAILURE_TARGET_SEQUENCE_MISSING = "target_sequence_missing"
FAILURE_BINDER_SEQUENCE_MISSING = "binder_sequence_missing"
FAILURE_TARGET_MSA_MISSING = "target_msa_missing"
FAILURE_SITE_MAP_UNAVAILABLE = "site_map_unavailable"
FAILURE_PREDICTOR_SUBPROCESS = "predictor_subprocess_failed"
FAILURE_PREDICTOR_TIMEOUT = "predictor_timeout"
FAILURE_OUTPUT_MISSING = "predictor_output_missing"
FAILURE_UNEXPECTED_SAMPLE_COUNT = "unexpected_sample_count"
FAILURE_ARTIFACT_WRITE = "artifact_write_failed"

SLUG_CHARACTERS = re.compile(r"[^a-zA-Z0-9_.-]")


class MissingSeedOutput(Exception):
    """Protenix wrote no output for a seed."""


class UnexpectedSampleCount(Exception):
    """A seed holds a number of samples this arm cannot choose between."""


def sanitize(value: str) -> str:
    """Return a filesystem-safe token, using the lane's own sanitiser."""
    return SLUG_CHARACTERS.sub("_", str(value))


def job_name(target_id: str, candidate_id: str) -> str:
    """Return the `name` field of one Protenix job.

    Protenix writes its output under `{out_dir}/{name}/seed_{seed}/` and names
    every file inside after `{name}`, so this string decides both.
    """
    return f"{sanitize(target_id)}-{sanitize(candidate_id)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Protenix v2, target chains take a precomputed MSA and the binder is single sequence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("toolcheck", help="Report that the arm's package is installed.")
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
            "--target-unpaired-msa-a3m",
            action="append",
            metavar="TARGET_ID=PATH",
            help=(
                "The unpaired target-chain a3m, written into the input JSON as "
                "unpairedMsaPath. Repeat once per target."
            ),
        )
        subparser.add_argument(
            "--target-paired-msa-a3m",
            action="append",
            metavar="TARGET_ID=PATH",
            help=(
                "The paired target-chain a3m, written into the input JSON as "
                "pairedMsaPath. Optional, and both Protenix and the published "
                "record treat it as optional."
            ),
        )
        subparser.add_argument(
            "--allow-msa-search",
            action="store_true",
            help=(
                "Run a target with no a3m path anyway. `--use_msa true` sends "
                "such a job to an MSA server, which fails after the GPU is "
                "already allocated in a container with no egress."
            ),
        )
        subparser.add_argument(
            "--protenix-executable",
            default="protenix",
            help="The console script, which `setup.py` maps to runner.batch_inference.",
        )
        subparser.add_argument(
            "--predictor-timeout-seconds",
            type=int,
            default=None,
            help=(
                "Timeout for one `protenix pred` call. Unset by default, so the "
                "stage timeout governs. The first call on a fresh container "
                "JIT-compiles a CUDA layer-norm kernel."
            ),
        )
    return parser


def protein_chain_entity(
    sequence: str,
    chain_id: str,
    *,
    unpaired_msa: str | None = None,
    paired_msa: str | None = None,
) -> dict[str, Any]:
    """Return one `proteinChain` entity in Protenix input dialect.

    `id` fixes the chain letters in the output, so the predicted complex uses
    the chains the campaign configuration names. Without it the arm would rely
    on chain position, and the published row records two designs where the
    binder does not sit where position would suggest.
    """
    entity: dict[str, Any] = {"sequence": sequence, "count": 1, "id": [chain_id]}
    # The deprecated `msa: {precomputed_msa_dir, pairing_db}` spelling still
    # works on 2.0.0. These two fields are the current ones.
    if paired_msa:
        entity["pairedMsaPath"] = str(Path(paired_msa).resolve())
    if unpaired_msa:
        entity["unpairedMsaPath"] = str(Path(unpaired_msa).resolve())
    return {"proteinChain": entity}


def write_input_json(
    path: Path,
    *,
    name: str,
    target_sequence: str,
    target_chain: str,
    binder_sequence: str,
    binder_chain: str,
    unpaired_msa: str | None,
    paired_msa: str | None,
) -> Path:
    """Write the Protenix-dialect input JSON for one target and candidate.

    The top level is a list even for one job. Entity order is target first and
    binder second, which is the published row's stated convention.
    """
    document = [
        {
            "name": name,
            "sequences": [
                protein_chain_entity(
                    target_sequence,
                    target_chain,
                    unpaired_msa=unpaired_msa,
                    paired_msa=paired_msa,
                ),
                # The binder is single sequence on every arm, so it carries no
                # MSA path of any kind.
                protein_chain_entity(binder_sequence, binder_chain),
            ],
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path


def predictor_argv(
    executable: str,
    *,
    input_json: Path,
    out_dir: Path,
    seeds: list[int],
) -> list[str]:
    """Return the argv for one `protenix pred` call.

    `--step` and `--dtype` are absent by instruction, and their defaults are the
    values the campaign ran. `--use_default_params` is absent because the model
    architecture block loads before any CLI value is read, and the two flags
    this script does pass resolve to the same numbers.
    """
    return [
        executable,
        "pred",
        "--input",
        str(input_json),
        "--out_dir",
        str(out_dir),
        "--model_name",
        MODEL_NAME,
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--cycle",
        str(CYCLE),
        "--sample",
        str(SAMPLES_PER_SEED),
        "--need_atom_confidence",
        "true",
        "--use_msa",
        "true",
    ]


def seed_outputs(out_dir: Path, name: str, seed: int) -> tuple[Path, Path, Path | None]:
    """Return one seed's CIF, full-data JSON and summary JSON.

    `sorted_by_ranking_score` defaults to True, so the `{rank}` suffix is a rank
    position rather than a sample index. At one sample per seed there is exactly
    one of each file and the rank is 0. More than one means `--sample` changed,
    and choosing between samples has to rank by ipSAE rather than by Protenix's
    own ranking score. This arm refuses that case instead of taking rank 0.
    """
    seed_dir = out_dir / name / f"seed_{seed}"
    if not seed_dir.is_dir():
        raise MissingSeedOutput(f"protenix wrote no directory for seed {seed}: {seed_dir}")
    full_data = sorted(seed_dir.glob(f"{name}_full_data_sample_*.json"))
    structures = sorted(seed_dir.glob(f"{name}_sample_*.cif"))
    if not full_data or not structures:
        raise MissingSeedOutput(
            f"seed {seed} holds {len(structures)} structures and {len(full_data)} "
            "full-data files, and needs one of each"
        )
    if len(full_data) != SAMPLES_PER_SEED or len(structures) != SAMPLES_PER_SEED:
        # TODO Choosing among samples has to rank by ipSAE, which this arm does
        # not compute. At one sample per seed the case cannot arise, so the arm
        # refuses it rather than taking rank 0. Protenix sorts by its own
        # ranking score, which is a different quantity from the campaign metric.
        raise UnexpectedSampleCount(
            f"seed {seed} holds {len(structures)} structures and {len(full_data)} "
            f"full-data files at --sample {SAMPLES_PER_SEED}"
        )
    summaries = sorted(seed_dir.glob(f"{name}_summary_confidence_sample_*.json"))
    # Counting the two file sets is not enough to pair them. One structure and one
    # full-data file satisfy the count above even when they carry different sample
    # suffixes, and this function then hands the caller sample 0's coordinates beside
    # sample 1's PAE. That corrupts ipSAE, the campaign's primary metric, without
    # raising anything. A stale file left in a reused seed directory by an earlier
    # multi-sample run is enough to produce it, so the indices are compared.
    structure_index = _sample_index(structures[0], f"{name}_sample_")
    full_data_index = _sample_index(full_data[0], f"{name}_full_data_sample_")
    if structure_index != full_data_index:
        raise UnexpectedSampleCount(
            f"seed {seed} pairs structure sample {structure_index} with full-data sample "
            f"{full_data_index}: {structures[0].name} against {full_data[0].name}"
        )
    summary = summaries[0] if summaries else None
    if summary is not None:
        summary_index = _sample_index(summary, f"{name}_summary_confidence_sample_")
        if summary_index != structure_index:
            raise UnexpectedSampleCount(
                f"seed {seed} pairs structure sample {structure_index} with summary sample "
                f"{summary_index}: {structures[0].name} against {summary.name}"
            )
    return structures[0], full_data[0], summary


def _sample_index(path: Path, prefix: str) -> int:
    """Return the sample suffix a Protenix output file carries.

    The suffix is the only thing tying a structure to its own confidence data, so a
    name that does not carry one is refused rather than defaulted to zero.
    """
    stem = path.name[: -len(path.suffix)]
    if not stem.startswith(prefix):
        raise UnexpectedSampleCount(f"{path.name} does not start with {prefix!r}")
    suffix = stem[len(prefix):]
    if not suffix.isdigit():
        raise UnexpectedSampleCount(f"{path.name} carries no sample index")
    return int(suffix)


def pae_matrix(full_data_path: Path) -> list[list[float]]:
    """Return `token_pair_pae` from the full-data file.

    The key is renamed to `pae` by the contract writer, because the validator
    reads `pae` and nothing else. Protenix already rounds to two decimals in its
    own dumper, so nothing is rounded again here.
    """
    document = json.loads(full_data_path.read_text())
    matrix = document.get("token_pair_pae")
    if matrix is None:
        raise KeyError(
            f"{full_data_path} carries no token_pair_pae, which means "
            "--need_atom_confidence true did not take"
        )
    return [[float(value) for value in row] for row in matrix]


def summary_extra(summary_path: Path | None) -> dict[str, Any]:
    """Return the measurement fields only the predictor knows.

    `iptm` is a required measurement field and cannot be computed
    from the complex and the PAE, so it reaches the measurement through `extra`.
    The rest are informational.
    """
    if summary_path is None or not summary_path.is_file():
        return {"iptm": None}
    document = json.loads(summary_path.read_text())
    return {
        "iptm": arm_common.optional_float(document.get("iptm")),
        "ptm": arm_common.optional_float(document.get("ptm")),
        "mean_plddt": arm_common.optional_float(document.get("plddt")),
        "protenix_ranking_score": arm_common.optional_float(document.get("ranking_score")),
        "protenix_has_clash": document.get("has_clash"),
    }


def group_by_candidate(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group a per-seed plan into one Protenix call per target and candidate.

    `--seeds` takes every seed in one invocation, so the model loads once per
    candidate rather than once per seed.
    """
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for item in plan:
        key = (str(item["target"]["target_id"]), str(item["candidate"]["candidate_id"]))
        if key not in groups:
            groups[key] = {
                "target": item["target"],
                "candidate": item["candidate"],
                "predictor_id": item["predictor_id"],
                "seeds": [],
            }
            order.append(key)
        groups[key]["seeds"].append(int(item["seed"]))
    return [groups[key] for key in order]


def msa_identities_for(
    config: dict[str, Any],
    targets: list[dict[str, Any]],
    unpaired: dict[str, str],
    paired: dict[str, str],
) -> tuple[dict[str, dict[str, str | None]], dict[str, str]]:
    """Return the alignment identity every row of this arm carries, by target.

    `msa_path` and `msa_sha256` are required on every raw prediction row, so a row can
    never leave the question open. This arm is wired to the target MSA, so `base_row`
    will not fill them in from the adapter's own declaration the way the Fast arm's are,
    and the caller has to state them. Each target starts at null and is replaced once
    its a3m resolves and hashes, so a target whose a3m never resolved writes null
    rather than naming a file the run never read.

    The unpaired a3m is what a row names when both are supplied. That is the file
    `stage-msa` published and the target MSA manifest records, so it is the one a reader
    can match the row back to. The paired a3m is optional on both Protenix and the
    published record, and the profile passes only the unpaired one.

    `--allow-msa-search` is the one case where null understates what happened. Such a
    job has no local a3m, and `--use_msa true` sends it to an MSA server, so the row
    says the run read no alignment file, which is true, while the fold used an
    alignment. The profile passes `--target-unpaired-msa-a3m` and never takes that
    route. Naming a file the run did not read would be the worse answer.

    The second return is the per-target reason an a3m that was named could not be read.
    Its targets keep null, and the caller fails their rows rather than calling Protenix
    on a file it will fail on after the GPU is allocated.
    """
    identities: dict[str, dict[str, str | None]] = {}
    errors: dict[str, str] = {}
    if not arm_accepts_target_msa(config, PREDICTOR_ID):
        return identities, errors
    for target in targets:
        target_id = str(target["target_id"])
        identities[target_id] = {"msa_path": None, "msa_sha256": None}
        a3m = unpaired.get(target_id) or paired.get(target_id)
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
    work_dir = artifacts_attempt_dir / args.phase / "protenix-runs"
    work_dir.mkdir(parents=True, exist_ok=True)

    targets = config["targets"]
    target_sequences = resolve_per_target(args.target_sequence, targets, "--target-sequence")
    unpaired = resolve_per_target(
        args.target_unpaired_msa_a3m, targets, "--target-unpaired-msa-a3m"
    )
    paired = resolve_per_target(args.target_paired_msa_a3m, targets, "--target-paired-msa-a3m")
    hotspots = resolve_per_target(args.hotspot_residues, targets, "--hotspot-residues")

    plan = plan_predictions(
        config,
        stage_id=args.stage,
        row_phase=row_phase,
        artifact_root=args.artifact_root,
        count=args.count,
        predictor_id=PREDICTOR_ID,
    )
    jobs = group_by_candidate(plan)
    writer = RowWriter(manifest_path)
    sequence_cache: dict[str, str] = {}

    # Resolved before the first call. Hashing here rather than at fold time means an
    # unreadable a3m is knowable before `protenix pred` allocates a GPU.
    msa_identities, msa_setup_errors = msa_identities_for(config, targets, unpaired, paired)

    for index, job in enumerate(jobs, start=1):
        target = job["target"]
        candidate = job["candidate"]
        target_id = str(target["target_id"])
        seeds = job["seeds"]
        rows = {
            seed: base_row(
                config,
                item={
                    "target": target,
                    "candidate": candidate,
                    "seed": seed,
                    "predictor_id": job["predictor_id"],
                },
                row_phase=row_phase,
                model_revision=model_revision,
                controls=controls,
                msa_identity=msa_identities.get(target_id),
            )
            for seed in seeds
        }

        def fail_all(code: str, reason: str) -> None:
            for seed in seeds:
                writer.write(
                    failed_row(
                        binder_contract, rows[seed], failure_code=code, failure_reason=reason
                    )
                )

        # Every per-candidate body is wrapped. A job that raises on candidate 3
        # of 40 otherwise throws away the GPU time spent on candidates 1 and 2.
        try:
            if target_id not in sequence_cache:
                supplied = target_sequences.get(target_id)
                if not supplied:
                    fail_all(
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
                fail_all(
                    FAILURE_BINDER_SEQUENCE_MISSING,
                    f"candidate {candidate['candidate_id']} carries no sequence_path",
                )
                continue
            binder_sequence = read_fasta_sequence(Path(binder_path))

            unpaired_path = unpaired.get(target_id)
            paired_path = paired.get(target_id)
            if not unpaired_path and not paired_path and not args.allow_msa_search:
                fail_all(
                    FAILURE_TARGET_MSA_MISSING,
                    f"no target a3m for {target_id}, pass "
                    f"--target-unpaired-msa-a3m {target_id}=PATH or --allow-msa-search",
                )
                continue
            if target_id in msa_setup_errors:
                # An a3m was named and could not be read. Protenix would fail on the
                # same file after the container has the GPU, so this fails first.
                fail_all(FAILURE_TARGET_MSA_MISSING, msa_setup_errors[target_id])
                continue

            # Assembled before the call. A missing site value cannot be
            # recovered afterwards, and finding it after the call wastes the GPU
            # time the call just spent.
            try:
                site_map = site_residue_map_for(config, target, hotspots)
            except SiteMapUnavailable as exc:
                fail_all(FAILURE_SITE_MAP_UNAVAILABLE, short_reason(exc))
                continue

            name = job_name(target_id, str(candidate["candidate_id"]))
            chain_mapping = rows[seeds[0]]["chain_mapping"]
            job_dir = work_dir / name
            out_dir = job_dir / "out"
            if job_dir.exists():
                # A retry of the same attempt would otherwise read the previous
                # call's files and never notice this call wrote nothing.
                shutil.rmtree(job_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            input_json = write_input_json(
                job_dir / "input.json",
                name=name,
                target_sequence=target_sequence,
                target_chain=chain_mapping["target"],
                binder_sequence=binder_sequence,
                binder_chain=chain_mapping["binder"],
                unpaired_msa=unpaired_path,
                paired_msa=paired_path,
            )
            argv = predictor_argv(
                args.protenix_executable, input_json=input_json, out_dir=out_dir, seeds=seeds
            )
            print(
                f"{PREDICTOR_ID}: job {index} of {len(jobs)}, candidate "
                f"{candidate['candidate_id']}, seeds {seeds}",
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
                fail_all(
                    FAILURE_PREDICTOR_TIMEOUT,
                    f"protenix pred exceeded {args.predictor_timeout_seconds} seconds",
                )
                continue
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, flush=True)
            if completed.returncode != 0:
                tail = (completed.stderr or completed.stdout or "").strip().splitlines()
                fail_all(
                    FAILURE_PREDICTOR_SUBPROCESS,
                    f"protenix pred exited {completed.returncode}: "
                    f"{tail[-1] if tail else 'no output'}"[:500],
                )
                continue

            for seed in seeds:
                row = rows[seed]
                try:
                    structure_path, full_data_path, summary_path = seed_outputs(
                        out_dir, name, seed
                    )
                    written = binder_contract.write_prediction_artifacts(
                        # The two phases are never the same word. `phase` is
                        # the campaign phase, which goes on the row, into the
                        # measurement and into the slug. `run_phase` is this
                        # adapter's own --phase, which is the directory segment
                        # that keeps a smoke run and a scale run from
                        # overwriting each other.
                        attempt_dir=artifacts_attempt_dir,
                        phase=row["phase"],
                        run_phase=args.phase,
                        target_id=row["target_id"],
                        candidate_id=row["candidate_id"],
                        predictor=row["predictor"],
                        seed=row["seed"],
                        # The contract reads this path and writes the bytes
                        # into complex.cif, which is a copy. A symlink into this
                        # tree is a file that can vanish between the write and
                        # the hash.
                        complex_cif=structure_path,
                        pae=pae_matrix(full_data_path),
                        chain_mapping=row["chain_mapping"],
                        reference_cif=Path(row["design_pose_path"]),
                        site_residue_map=site_map,
                        model_revision=model_revision,
                        target_sequence=target_sequence,
                        binder_sequence=binder_sequence,
                        extra={
                            # These four are the contract's
                            # REQUIRED_EXTRA_FIELDS. No binder_metrics function
                            # returns them, so the arm supplies them. A name the
                            # identity block already owns raises, so nothing
                            # else from the row belongs here.
                            "target_sha256": row["target_sha256"],
                            "sequence_sha256": row["sequence_sha256"],
                            "design_pose_sha256": row["design_pose_sha256"],
                            **summary_extra(summary_path),
                        },
                    )
                    merged = dict(row)
                    merged.update(written)
                    writer.write(merged)
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    if isinstance(exc, MissingSeedOutput):
                        code = FAILURE_OUTPUT_MISSING
                    elif isinstance(exc, UnexpectedSampleCount):
                        code = FAILURE_UNEXPECTED_SAMPLE_COUNT
                    else:
                        # Chain assignment failures carry their own named code.
                        # Other measurement and write failures retain the arm's
                        # generic artifact code.
                        code = getattr(exc, "failure_code", FAILURE_ARTIFACT_WRITE)
                    writer.write(
                        failed_row(
                            binder_contract,
                            row,
                            failure_code=code,
                            failure_reason=short_reason(exc),
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            fail_all(getattr(exc, "failure_code", FAILURE_ARTIFACT_WRITE), short_reason(exc))

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


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "toolcheck":
        # The profile's toolcheck_argv is `pip show protenix`, because importing
        # the package touches CUDA.
        print(f"{ADAPTER_ID} entry script ok, model_name {MODEL_NAME}")
        return 0
    if args.command == "parse":
        return parse_outputs(args)
    return run_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())
