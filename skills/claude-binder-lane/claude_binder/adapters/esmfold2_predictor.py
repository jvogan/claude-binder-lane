#!/usr/bin/env python3
"""ESMFold2-Full arm of the binder lane cofold stages.

This script folds every candidate a cofold stage owns and writes one raw
prediction row per target, candidate and seed. It computes no metric of its own.
`binder_contract.write_prediction_artifacts` writes the three files and computes
every measurement, and `binder_metrics` holds the computation.

The module also carries the shared body of the ESMFold2 pair.
`esmfold2_fast_predictor.py` imports `ArmSpec` and `main` from here and passes
its own checkpoint. `report_arms.md` section 2 records that the two modes share
the same package, image, call, and embedding lineage. Their agreement measures
variation within that lineage.

Sources for every value in this file:

- `report_arms.md` sections 1.2, 1.3 and 1.5 for the call, the inference
  parameters and the returned objects.
- `ref/docs/LOOKUP_TABLES.md`, first table, row `ef2full`, for what the
  published campaign ran.
- `ADAPTER-API.md` for the two modules this script imports.
- `report_contract.md` sections 1, 2, 3 and 7 for the artifact layout and the
  row fields.

Exit code. Zero means the manifest holds one row per planned prediction, and
some of those rows may carry `status: failed`. One means the arm could not load
its model, so no prediction was attempted and every row is a failed row.
"""

import argparse
import glob
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from ..discovery import is_unconstrained_site

# Inference parameters, from protocol lines 70 and 111 and from the published
# `ef2full` and `ef2fast` rows of `ref/docs/LOOKUP_TABLES.md`. Both arms run the
# same three values.
NUM_LOOPS = 10
NUM_SAMPLING_STEPS = 68

# One diffusion sample per seed. `esmfold2/SKILL.md` shows 5 in its usage example
# and gives 25 seeds by 5 samples as the paper's FoldBench protocol. That is the
# paper's evaluation. This campaign published `num_diffusion_samples=1` at one
# sample per seed, and the published `ipsae_min_*` and `sc_dockq_*` columns were
# produced under it. Raising this to 5 is five times the diffusion work and it
# changes the number, because the campaign metric is defined over one sample.
# Leave it at 1.
NUM_DIFFUSION_SAMPLES = 1

# MSA depth for the Full arm. The protocol calls it `msa_max_seq=2048` and the
# published row calls it `msa_max_depth=2048`. They are the same number.
MSA_MAX_SEQUENCES = 2048

# Kernel backend and chunk size by complex length, from `esmfold2/SKILL.md`.
# Three bands, and the middle one is the reason this is not a single cutover.
#
#   L <= 1024          fused, chunk_size None. Optimal and OOM-safe, validated
#                      through this length.
#   1024 < L <= 1400   fused, chunk_size 256. The skill's "use 256 above".
#   L > 1400           reference backend, chunk_size 64. Above roughly 1400 the
#                      fused path hits an illegal memory access, which is a hard
#                      failure rather than a slowdown.
#
# The reference backend is roughly twelve times slower than fused, so dropping
# to it at 1025 would cost real time on every complex in the middle band. The
# two backends agree numerically, so the band a complex lands in changes the
# runtime and not the answer.
FUSED_CHUNK_FREE_MAX_LENGTH = 1024
FUSED_BACKEND_MAX_LENGTH = 1400
FUSED_CHUNK_SIZE_ABOVE_FREE = 256
REFERENCE_BACKEND_CHUNK_SIZE = 64

# Do not use `set_kernel_backend("cuequivariance")`. The
# `cuequivariance-torch==0.10.0` wheel lacks the compiled ops and silently falls
# back to the reference path, so it costs the speedup and reports nothing.
FUSED_BACKEND_NAME = "fused"

# The four ESMFold2-Experimental variants are for gradient-guided design and
# their forward pass is not inference-mode decorated. This arm folds, so the
# release checkpoints are the correct ones and an Experimental checkpoint is
# refused rather than run.
EXPERIMENTAL_CHECKPOINT_MARKER = "Experimental"

# ESMFold2 model revisions in a resolved profile are Hugging Face repository
# pins.  The predictor has exactly one checkpoint repository per arm; the
# profile may carry other pins (for example ESMC), but it must carry one and
# only one pin for that checkpoint.  Loading without this pin lets an offline
# cache silently select whichever snapshot happens to be present, then the
# result rows falsely report the profile's requested revision.
ESMFOLD2_REVISION_PIN_RE = re.compile(
    r"(?P<repository>biohub/ESMFold2(?:-Fast)?)@(?P<revision>[0-9a-f]{40})(?![0-9a-f])"
)

# TODO The published `ef2full` row lists `lm_dropout=0.3` and
# `msa_column_mask_rate=0.1`. The documented `fold()` signature exposes neither,
# so their call site is unknown and this script does not pass them. The real
# value comes from reading `esm.models.esmfold2` in the built image. This is
# TODO 2 of `report_arms.md` section 4.1.

# TODO Cofactors are not modelled here, and only one part of that is still
# missing. The mechanism is known: `esmfold2/SKILL.md` gives
# `LigandInput(id, ccd=[...])`, importable from `esm.models.esmfold2` and
# appended after the protein chains. The contents are known: the published rows
# say RBX1 takes ZN times three and 15-PGDH takes one NAD per HPGD protomer.
# What is missing is the campaign configuration field that says which target
# takes which, so there is nothing to read per row. The real value comes from
# whoever adds that field to the campaign schema.

# Failure codes this arm can emit. The executor constrains `failure_code` only
# to a non-empty string, so the vocabulary is the lane's to set.
# TODO The campaign-wide failure code vocabulary is not fixed. This set is this
# arm's working list and `report_contract.md` section 9 owns the decision.
FAILURE_MODEL_LOAD = "model_load_failed"
FAILURE_TARGET_SEQUENCE_MISSING = "target_sequence_missing"
FAILURE_BINDER_SEQUENCE_MISSING = "binder_sequence_missing"
FAILURE_MSA_LOAD = "msa_load_failed"

# The artifact type an adapter lists when it is wired to the shared target MSA. The
# executor spells it the same way in `lane.MSA_MANIFEST_ARTIFACT_TYPE`, and an adapter
# never imports the executor.
MSA_MANIFEST_ARTIFACT_TYPE = "target-msa-manifest"
FAILURE_SITE_MAP_UNAVAILABLE = "site_map_unavailable"
FAILURE_PREDICTION = "prediction_failed"
FAILURE_ARTIFACT_WRITE = "artifact_write_failed"

# The campaign phase a stage id maps to. `--phase` on the command line is the
# smoke or scale phase and is a different quantity.
STAGE_PREFIX_PHASE = {
    "cofold-screen-": "screen",
    "cofold-intermediate-": "intermediate",
    "cofold-rescore-": "uniform-rescore",
    "optimization-cofold-round-": "optimization",
}

# The shard slice arrives in the environment because no template token can
# express it. Package-external shard workers export these values.
SHARD_ENV_PREFIX = "CLAUDE_BINDER_LANE_"


class ArmSpec:
    """One ESMFold2 arm, which is a checkpoint and an MSA policy."""

    def __init__(
        self,
        *,
        predictor_id: str,
        adapter_id: str,
        checkpoint: str,
        uses_target_msa: bool,
        description: str,
    ) -> None:
        self.predictor_id = predictor_id
        self.adapter_id = adapter_id
        self.checkpoint = checkpoint
        self.uses_target_msa = uses_target_msa
        self.description = description


ARM_FULL = ArmSpec(
    predictor_id="esmfold2",
    adapter_id="esmfold2-predictor",
    checkpoint="biohub/ESMFold2",
    uses_target_msa=True,
    description="ESMFold2-Full, target chains get an unpaired a3m and the binder is single sequence.",
)


# --- small file helpers, matching binder_lane_fixture_adapter.py -------------


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def render(value: str, *, attempt_dir: Path, phase: str) -> str:
    return value.replace("{{attempt_dir}}", str(attempt_dir)).replace("{{phase}}", phase)


def read_fasta_sequence(path: Path) -> str:
    """Return the first record of a FASTA file as one uppercase string."""
    lines = Path(path).read_text().replace("\x00", "").splitlines()
    sequence: list[str] = []
    seen_header = False
    for line in lines:
        if line.startswith(">"):
            if seen_header:
                break
            seen_header = True
            continue
        sequence.append(line.strip())
    joined = "".join(sequence).upper()
    if not joined:
        raise ValueError(f"FASTA file holds no sequence: {path}")
    return joined


# --- campaign configuration readers -----------------------------------------


def campaign_phase(stage_id: str) -> str:
    """Return the row's `phase` value for a stage id.

    `--phase` is `single` or `scale`. The row's `phase` is `screen` or
    `uniform-rescore` and the executor validates it against that vocabulary.
    """
    for prefix, phase in STAGE_PREFIX_PHASE.items():
        if stage_id.startswith(prefix):
            return phase
    raise ValueError(f"stage is not a cofold stage this arm can run: {stage_id}")


def stage_record(config: dict[str, Any], stage_id: str) -> dict[str, Any]:
    return next(stage for stage in config["stages"] if stage["stage_id"] == stage_id)


def target_design_chain(target: dict[str, Any]) -> str:
    """Return the one chain of a target whose role is design-target.

    `validate_observations` derives the expected `chain_mapping` this way and
    compares the whole object for exact equality, so a literal chain letter
    fails on any campaign configured differently.
    """
    matches = [chain for chain in target.get("chains", []) if chain.get("role") == "design-target"]
    if len(matches) != 1 or not matches[0].get("chain_id"):
        raise ValueError(f"target {target.get('target_id')} must define one design-target chain")
    return str(matches[0]["chain_id"])


def control_records(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    controls: dict[str, dict[str, Any]] = {}
    for group_name in ("positive", "negative"):
        for item in config.get("controls", {}).get(group_name, []):
            if item.get("enabled", True):
                controls[str(item["id"])] = item
    return controls


def chain_mapping_for(
    config: dict[str, Any],
    target: dict[str, Any],
    candidate_id: str,
    controls: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Return `chain_mapping` for one row, read from configuration.

    A control carries its own two chains and overrides the target's.
    """
    control = controls.get(str(candidate_id))
    if control is not None:
        return {"target": str(control["target_chain"]), "binder": str(control["binder_chain"])}
    return {
        "target": target_design_chain(target),
        "binder": str(config["binder"]["binder_chain_id"]),
    }


def predictor_record(config: dict[str, Any], predictor_id: str) -> dict[str, Any]:
    return next(item for item in config["cofold"]["predictors"] if item["id"] == predictor_id)


def model_revision_for(config: dict[str, Any], adapter_id: str) -> str:
    adapter = next(item for item in config["adapters"] if item["adapter_id"] == adapter_id)
    return str(adapter["model_revision"])


def arm_accepts_target_msa(config: dict[str, Any], predictor_id: str) -> bool:
    """Return whether one arm's adapter is wired to the shared target MSA.

    An adapter that does not list `target-msa-manifest` in `accepted_artifacts` is
    never handed an alignment, because the stage that would supply it is not one of
    its inputs. `fixture_adapter` reads the same list to decide the same question.
    """
    predictor = predictor_record(config, predictor_id)
    adapter = next(
        item for item in config["adapters"] if item["adapter_id"] == predictor["adapter_id"]
    )
    return MSA_MANIFEST_ARTIFACT_TYPE in adapter.get("accepted_artifacts", [])


def target_msa_identity(a3m: str) -> dict[str, str | None]:
    """Return the alignment identity a row folded against one names.

    The path is the a3m the stage was handed, which is the file `stage-msa` published
    and the file the target MSA manifest names. The digest is of that same file.
    `clean_a3m` writes a reformatted copy for the loader, and the copy is deliberately
    not what the row names, because no reader could match it back to the manifest.
    """
    import hashlib

    path = Path(a3m)
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise TargetMsaUnavailable(f"could not read target a3m {path}: {exc}") from exc
    return {"msa_path": str(path.resolve()), "msa_sha256": digest}


def seeds_for(config: dict[str, Any], phase: str) -> list[int]:
    """Return the seed labels for a phase, read from the campaign configuration.

    The published campaign ran five seeds labeled 0 to 4 at rescore and one seed
    at screen. Those labels live in the configuration rather than here, so a
    campaign that changes them changes the run without changing this file.
    """
    key = "screen_seeds" if phase == "screen" else "rescore_seeds"
    return [int(seed) for seed in config["cofold"][key]]


def published_artifact_path(
    config: dict[str, Any],
    artifact_root: Path,
    producer_stage_id: str,
    artifact_id: str,
) -> Path:
    """Resolve an input artifact through the producer path declared in the plan."""
    producer = stage_record(config, producer_stage_id)
    outputs = producer.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError(f"stage {producer_stage_id} outputs must be a list")
    matches = [
        output
        for output in outputs
        if isinstance(output, dict) and output.get("artifact_id") == artifact_id
    ]
    if len(matches) != 1:
        raise ValueError(f"stage {producer_stage_id} must publish one {artifact_id} artifact")
    publish_path = matches[0].get("publish_path")
    if not isinstance(publish_path, str) or not publish_path or Path(publish_path).is_absolute():
        raise ValueError(f"stage {producer_stage_id} {artifact_id} has no relative publish_path")
    return artifact_root / publish_path


def candidate_manifest_path(
    config: dict[str, Any],
    artifact_root: Path,
    phase: str,
    stage_id: str,
) -> Path:
    if phase == "screen":
        return artifact_root / "filters" / "passing-candidates.jsonl"
    stage = stage_record(config, stage_id)
    inputs = stage.get("inputs")
    if inputs is None and len(config.get("stages", [])) == 1:
        # control_builder mints one synthetic rescore stage carrying a stage_id
        # and outputs and no inputs, then calls this predictor with it. The list
        # guard below used to reject that record before the single-stage branch
        # further down could accept it, so the branch written for this caller was
        # unreachable and every control-calibration stage failed here. Take the
        # escape only when the record declares no inputs at all and the config
        # holds exactly one stage, which is the shape only that caller builds.
        return artifact_root / "optimization" / "rescore-candidates.jsonl"
    if not isinstance(inputs, list):
        raise ValueError(f"stage {stage_id} inputs must be a list")
    if phase == "intermediate":
        survivors = [
            value.split(":", 1)
            for value in inputs
            if isinstance(value, str) and value.endswith(":intermediate-candidates")
        ]
        if len(survivors) != 1:
            raise ValueError(f"stage {stage_id} must declare one intermediate-candidates input")
        return published_artifact_path(config, artifact_root, *survivors[0])
    if phase == "optimization":
        passing = [
            value.split(":", 1)
            for value in inputs
            if isinstance(value, str) and value.endswith(":passing-candidates")
        ]
        if len(passing) != 1:
            raise ValueError(f"stage {stage_id} must declare one passing-candidates input")
        return published_artifact_path(config, artifact_root, *passing[0])
    matches = [
        value.split(":", 1)
        for value in inputs
        if isinstance(value, str) and value.endswith(":rescore-candidates")
    ]
    if len(matches) != 1:
        # control_builder writes this private workspace manifest before it calls
        # the predictor with its one synthetic rescore stage.
        if len(config.get("stages", [])) == 1:
            return artifact_root / "optimization" / "rescore-candidates.jsonl"
        raise ValueError(f"stage {stage_id} must declare one rescore-candidates input")
    producer_stage_id, artifact_id = matches[0]
    return published_artifact_path(config, artifact_root, producer_stage_id, artifact_id)


def site_residue_map_path(target: dict[str, Any]) -> Path:
    site = target["site"]
    return Path(site.get("runtime_residue_map_path") or site["residue_map_path"])


class SiteMapUnavailable(Exception):
    """The campaign configuration does not carry a value the site metrics need."""


class TargetSequenceUnavailable(Exception):
    """No argument supplied the target chain's sequence."""


class TargetMsaUnavailable(Exception):
    """No argument supplied the target chain's a3m on an arm that needs one."""


def resolve_target_sequence(target_id: str, supplied: dict[str, str]) -> str:
    """Return one target's sequence, from a literal or from a FASTA file."""
    value = supplied.get(target_id)
    if not value:
        raise TargetSequenceUnavailable(
            f"no sequence for target {target_id}, pass "
            f"--target-sequence {target_id}=SEQUENCE_OR_FASTA"
        )
    path = Path(value)
    return read_fasta_sequence(path) if path.exists() else value.upper()


def require_target_a3m(target_id: str, supplied: dict[str, str]) -> str:
    """Return one target's a3m path on an arm that takes an MSA."""
    value = supplied.get(target_id)
    if not value:
        raise TargetMsaUnavailable(
            f"no target a3m for {target_id}, pass --target-msa-a3m {target_id}=PATH"
        )
    return value


def site_residue_map_for(
    config: dict[str, Any],
    target: dict[str, Any],
    hotspots: dict[str, str],
) -> dict[str, Any]:
    """Assemble the mapping `binder_metrics.compute_site_metrics` takes.

    It is a mapping rather than a path. It carries the site definition and the
    provenance fields the row echoes, and the executor compares three of them
    against the target's own site contract, so every value is passed through
    from configuration rather than derived.
    """
    site = target["site"]
    target_id = str(target["target_id"])
    supplied = hotspots.get(target_id)
    configured_hotspots = site.get("hotspot_residues")
    declared_source = site.get("hotspot_source")
    if declared_source is None:
        hotspot_source = (
            "explicit"
            if isinstance(configured_hotspots, list) and configured_hotspots
            else "site-fallback"
        )
    else:
        hotspot_source = str(declared_source)
    mapping: dict[str, Any] = {
        "contact_cutoff_angstrom": site["contact_cutoff_angstrom"],
        "atom_selection": site["atom_selection"],
        "residue_map_sha256": site["residue_map_sha256"],
        "metric_basis": config["scoring"]["implementations"]["site_metric_basis"],
        "hotspot_source": hotspot_source,
    }
    if is_unconstrained_site(site):
        mapping["epitope_constraint"] = "unconstrained"
    else:
        if hotspot_source in {"explicit", "site-fallback"} and not supplied:
            raise SiteMapUnavailable(
                f"no hotspot residues for target {target_id}, pass "
                f"--hotspot-residues {target_id}=CHAIN:NUMBER,CHAIN:START-END"
            )
        mapping.update({
            "site_residues": list(site["reference_contact_residues"]),
            "epitope_constraint": "constrained",
        })
        if hotspot_source in {"explicit", "site-fallback"}:
            mapping["hotspot_residues"] = [
                item.strip() for item in supplied.split(",") if item.strip()
            ]
        for field in (
            "published_epitope_table",
            "published_epitope_key_column",
            "published_epitope_residue_column",
            "published_epitope_chain_policy",
        ):
            if site.get(field) is not None:
                mapping[field] = site[field]
    residue_map_file = site_residue_map_path(target)
    if residue_map_file.is_file():
        document = load_json(residue_map_file)
        translation = document.get("source_to_cleaned")
        if translation:
            mapping["source_to_cleaned"] = translation
    return mapping


# --- shard handling ----------------------------------------------------------


def shard_slice() -> tuple[int, int] | None:
    """Return the half-open candidate slice this job owns, or None when whole.

    `{{count}}` in the rendered argv is the phase total, because the stage
    contract validates the merged phase against it. A sharded job runs only the
    candidates between SHARD_START and SHARD_STOP.
    """
    start = os.environ.get(f"{SHARD_ENV_PREFIX}SHARD_START")
    stop = os.environ.get(f"{SHARD_ENV_PREFIX}SHARD_STOP")
    if start is None or stop is None:
        return None
    return int(start), int(stop)


def shard_out_dir() -> Path | None:
    value = os.environ.get(f"{SHARD_ENV_PREFIX}SHARD_OUT_DIR")
    return Path(value) if value else None


def output_paths(
    config: dict[str, Any],
    stage_id: str,
    attempt_dir: Path,
    phase: str,
) -> tuple[Path, Path]:
    """Return where the manifest goes and which directory holds the artifacts.

    An unsharded job writes the tree the stage contract names. A sharded job
    writes the same tree rooted at CLAUDE_BINDER_LANE_SHARD_OUT_DIR, which is
    the mirror layout Claude Binder merges after shard completion.
    """
    stage = stage_record(config, stage_id)
    contract_path = Path(render(stage["outputs"][0]["path_template"], attempt_dir=attempt_dir, phase=phase))
    phase_dir = attempt_dir / phase
    shard_dir = shard_out_dir()
    if shard_dir is None:
        return contract_path, attempt_dir
    tail = contract_path.relative_to(phase_dir)
    # The artifact root becomes the shard directory, so the three files per
    # prediction land inside the slice that wrote them. Rows carry absolute
    # paths, so nothing needs repointing after the merge.
    return shard_dir / tail, shard_dir


# --- argument surface --------------------------------------------------------


def key_value_argument(value: str) -> tuple[str | None, str]:
    """Split `TARGET_ID=VALUE`, or return `(None, VALUE)` for a bare value.

    A bare value applies to the campaign's only target. A campaign with more
    than one target has to name which one.
    """
    if "=" in value:
        key, _, rest = value.partition("=")
        return key.strip(), rest.strip()
    return None, value.strip()


def resolve_per_target(
    entries: list[str] | None,
    targets: list[dict[str, Any]],
    label: str,
) -> dict[str, str]:
    """Turn repeated `TARGET_ID=VALUE` arguments into a per-target mapping."""
    resolved: dict[str, str] = {}
    for entry in entries or []:
        key, value = key_value_argument(entry)
        if key is None:
            if len(targets) != 1:
                raise ValueError(
                    f"{label} needs TARGET_ID=VALUE because the campaign has "
                    f"{len(targets)} targets"
                )
            key = str(targets[0]["target_id"])
        resolved[key] = value
    return resolved


def build_parser(arm: ArmSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=arm.description)
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
        if arm.uses_target_msa:
            subparser.add_argument(
                "--target-msa-a3m",
                action="append",
                metavar="TARGET_ID=PATH",
                help=(
                    "The unpaired target-chain a3m, one per target. The binder is "
                    "single sequence on every arm and never takes this."
                ),
            )
        subparser.add_argument(
            "--fused-backend-max-length",
            type=int,
            default=FUSED_BACKEND_MAX_LENGTH,
            help=(
                "Complex length above which the arm falls back to the reference "
                "kernel backend, which is the hard failure boundary. Between "
                "1024 and this length the arm stays on the fused backend with a "
                "chunk size of 256."
            ),
        )
        subparser.add_argument(
            "--device",
            default="cuda",
            help="Torch device for the model. The shipped environment gives one GPU.",
        )
    return parser


# --- the model, imported lazily so --help works without a GPU stack ----------


def install_safe_svd() -> None:
    """Redirect small batched SVDs to CPU, from `esmfold2/SKILL.md`.

    The Kabsch alignment calls `torch.linalg.svd(H32, driver="gesvd")` on
    batched 3x3 matrices. A NaN or Inf input from a degenerate diffusion sample
    corrupts the cusolver workspace, after which every later CUDA call fails
    with an illegal memory access. One bad sample otherwise poisons the rest of
    the job.
    """
    import torch

    if getattr(torch.linalg.svd, "_binder_lane_safe", False):
        return
    original_svd = torch.linalg.svd

    def safe_svd(A, full_matrices=True, driver=None):  # noqa: N803
        if A.is_cuda and A.shape[-1] <= 4 and A.shape[-2] <= 4:
            on_cpu = A.detach().float().cpu()
            if not torch.isfinite(on_cpu).all():
                on_cpu = torch.nan_to_num(on_cpu, nan=0.0, posinf=1e6, neginf=-1e6)
            out = original_svd(on_cpu, full_matrices=full_matrices)
            # torch.return_types.linalg_svd is a C structseq, so its constructor
            # takes one tuple rather than positional arguments.
            return type(out)(tuple(t.to(A.device, A.dtype) for t in out))
        return original_svd(A, full_matrices=full_matrices, driver=driver)

    safe_svd._binder_lane_safe = True
    torch.linalg.svd = safe_svd


def pinned_checkpoint_revision(arm: ArmSpec, model_revision: str) -> str:
    """Return the one immutable revision the selected arm must load.

    ``model_revision`` is free text at the profile boundary, so this parser is
    deliberately narrow: it recognises only the ESMFold2 repositories this
    adapter executes, and it refuses a missing or duplicate pin instead of
    choosing one by position.
    """
    matches = [
        match.group("revision")
        for match in ESMFOLD2_REVISION_PIN_RE.finditer(model_revision)
        if match.group("repository") == arm.checkpoint
    ]
    if len(matches) != 1:
        raise ValueError(
            f"adapter {arm.adapter_id!r} must record exactly one immutable "
            f"{arm.checkpoint}@<commit> pin in model_revision; found {len(matches)}"
        )
    return matches[0]


def verify_loaded_model_revision(model: Any, *, arm: ArmSpec, revision: str) -> None:
    """Refuse a loaded checkpoint that identifies itself as another revision.

    The transformers model exposes ``config._commit_hash`` in the deployed
    offline ESMFold2 image.  Some compatible test doubles and older model
    objects do not expose it; they are still constrained by the explicit
    ``revision=`` request, but cannot supply an independent confirmation.
    """
    config = getattr(model, "config", None)
    observed = getattr(config, "_commit_hash", None)
    if observed is None:
        return
    if not isinstance(observed, str) or observed != revision:
        raise ValueError(
            f"{arm.checkpoint} requested revision {revision!r}, but the loaded "
            f"model config reports {observed!r}"
        )


def load_model(
    arm: ArmSpec,
    device: str,
    *,
    model_revision: str | None = None,
    model_class: Any | None = None,
    safe_svd_installer: Callable[[], None] | None = None,
):
    """Load one checkpoint and put it on the fast path.

    `from_pretrained` loads with `_kernel_backend=None`, which is the reference
    PyTorch path and roughly twelve times slower than the paper. Nothing warns,
    and the numbers come out the same, so the cost is silent.
    """
    # Checked before the import, so a wrong checkpoint fails in a second rather
    # than after the weights stack loads.
    if EXPERIMENTAL_CHECKPOINT_MARKER in arm.checkpoint:
        release_checkpoint = (
            "biohub/ESMFold2" if arm.uses_target_msa else "biohub/ESMFold2-Fast"
        )
        arm_name = "ESMFold2-Full" if arm.uses_target_msa else "ESMFold2-Fast"
        raise ValueError(
            f"{arm.checkpoint} is an ESMFold2-Experimental variant, which is for "
            "gradient-guided design. Its forward pass is not inference-mode "
            "decorated and the fused backend crashes on it. This arm folds, so it "
            f"takes a release checkpoint. The published campaign used {release_checkpoint} "
            f"for the {arm_name} arm."
        )

    if not isinstance(model_revision, str):
        raise ValueError(
            f"adapter {arm.adapter_id!r} has no model_revision to pin "
            f"{arm.checkpoint}"
        )
    revision = pinned_checkpoint_revision(arm, model_revision)
    if model_class is None:
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        model_class = ESMFold2Model

    if safe_svd_installer is None:
        safe_svd_installer = install_safe_svd
    safe_svd_installer()
    model = model_class.from_pretrained(
        arm.checkpoint,
        revision=revision,
        local_files_only=True,
    ).to(device).eval()
    verify_loaded_model_revision(model, arm=arm, revision=revision)
    model.set_kernel_backend(FUSED_BACKEND_NAME)
    model.set_chunk_size(None)
    return model


def clean_a3m(source: Path, query_sequence: str, destination: Path) -> Path:
    """Write a copy of an a3m that `MSA.from_a3m` will accept.

    `MSA.from_a3m` asserts equal row lengths after insertion removal. ColabFold
    output often carries trailing null bytes and a first row that is off by one
    against the query, so this strips the null bytes and forces row 0 to the
    query sequence.
    """
    text = source.read_text(errors="replace").replace("\x00", "")
    lines = [line.rstrip("\n") for line in text.splitlines() if line.strip()]
    out: list[str] = []
    header_count = 0
    for line in lines:
        if line.startswith(">"):
            header_count += 1
            out.append(line)
            continue
        if header_count == 1 and len(out) >= 1 and out[-1].startswith(">"):
            out.append(query_sequence)
            continue
        out.append(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(out) + "\n")
    return destination


def load_target_msa(a3m_path: Path, query_sequence: str, work_dir: Path):
    from esm.utils.msa.msa import MSA

    cleaned = clean_a3m(a3m_path, query_sequence, work_dir / f"{a3m_path.stem}.cleaned.a3m")
    return MSA.from_a3m(str(cleaned), max_sequences=MSA_MAX_SEQUENCES)


def fold_one(
    arm: ArmSpec,
    model,
    *,
    target_chain: str,
    target_sequence: str,
    binder_chain: str,
    binder_sequence: str,
    target_msa,
    seed: int,
    fused_backend_max_length: int,
):
    """Fold one complex and return the single prediction the arm asked for."""
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )
    from esm.utils.structure.molecular_complex import MolecularComplexResult

    total_length = len(target_sequence) + len(binder_sequence)
    if total_length > fused_backend_max_length:
        model.set_kernel_backend(None)
        model.set_chunk_size(REFERENCE_BACKEND_CHUNK_SIZE)
    elif total_length > FUSED_CHUNK_FREE_MAX_LENGTH:
        model.set_kernel_backend(FUSED_BACKEND_NAME)
        model.set_chunk_size(FUSED_CHUNK_SIZE_ABOVE_FREE)
    else:
        model.set_kernel_backend(FUSED_BACKEND_NAME)
        model.set_chunk_size(None)

    target_input = (
        ProteinInput(id=target_chain, sequence=target_sequence, msa=target_msa)
        if target_msa is not None
        else ProteinInput(id=target_chain, sequence=target_sequence)
    )
    # The predictor input order is settled here. The target is first and the
    # binder is second, which is the order the campaign folds in. The emitted
    # structure is mapped independently by binder_contract from the sequences.
    #
    # The input ids come from the campaign configuration. The returned file can
    # emit a different order, so binder_contract matches its chains to the two
    # known sequences before computing metrics.
    #
    # The declared ids remain on the row as an assertion against that config.
    #
    # The binder is single sequence on every arm, so it never takes an msa.
    spi = StructurePredictionInput(
        sequences=[target_input, ProteinInput(id=binder_chain, sequence=binder_sequence)]
    )
    result = ESMFold2InputBuilder().fold(
        model,
        spi,
        num_loops=NUM_LOOPS,
        num_sampling_steps=NUM_SAMPLING_STEPS,
        num_diffusion_samples=NUM_DIFFUSION_SAMPLES,
        seed=seed,
    )
    # fold() documents its own return as "A single result when
    # num_diffusion_samples == 1, otherwise a list", and a fal H100 run on
    # 2026-08-22 confirmed the single-result branch by raising TypeError here.
    # The campaign always runs at one sample, so that is the branch that
    # matters, and both are asserted rather than assumed.
    if NUM_DIFFUSION_SAMPLES == 1:
        if not isinstance(result, MolecularComplexResult):
            raise TypeError(
                f"fold() returned {type(result).__name__} at "
                "num_diffusion_samples=1; expected MolecularComplexResult"
            )
        return result
    if not isinstance(result, list):
        raise TypeError(
            f"fold() returned {type(result).__name__} at "
            f"num_diffusion_samples={NUM_DIFFUSION_SAMPLES}; expected list"
        )
    if len(result) != NUM_DIFFUSION_SAMPLES:
        raise ValueError(
            f"fold() returned {len(result)} samples at "
            f"num_diffusion_samples={NUM_DIFFUSION_SAMPLES}"
        )
    return result[0]


def pae_matrix(prediction) -> list[list[float]]:
    """Return `prediction.pae` as the nested list the contract asks for.

    `prediction.pae` is a tensor. Its native precision is retained because the
    campaign selects the winning seed by ipsae_min, and close seeds can change
    order after rounding. The contract validates and writes these values as
    provided. Protenix already supplies its own released precision.
    """
    return [[float(value) for value in row] for row in prediction.pae.tolist()]


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prediction_extra(prediction) -> dict[str, Any]:
    """Return the measurement fields only the predictor knows.

    `iptm` is a required measurement field and cannot be computed
    from the complex and the PAE, so it reaches the measurement through `extra`.
    `ptm` and the mean pLDDT are informational, and the protocol tracks `iptm`
    as a shadow metric without ranking on it.
    """
    plddt = getattr(prediction, "plddt", None)
    mean_plddt = None
    if plddt is not None:
        try:
            mean_plddt = float(plddt.mean())
        except (AttributeError, TypeError, ValueError):
            mean_plddt = None
    return {
        "iptm": optional_float(getattr(prediction, "iptm", None)),
        "ptm": optional_float(getattr(prediction, "ptm", None)),
        "mean_plddt": mean_plddt,
    }


# --- the run --------------------------------------------------------------


class RowWriter:
    """Append one row at a time so a timeout keeps the work already paid for.

    A job that buffers every row until the end loses all of it when the stage
    times out. Each line is a complete record and is flushed before the next
    prediction starts.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", encoding="utf-8")
        self.count = 0
        self.failed = 0

    def write(self, row: dict[str, Any]) -> None:
        self.handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.count += 1
        if row.get("status") == "failed":
            self.failed += 1

    def close(self) -> None:
        self.handle.close()


def plan_predictions(
    config: dict[str, Any],
    *,
    stage_id: str,
    row_phase: str,
    artifact_root: Path,
    count: int,
    predictor_id: str,
) -> list[dict[str, Any]]:
    """Return the list of predictions this job owns, in a stable order.

    The order matches the fixture adapter: candidate, then target, then seed.
    `--count` is the phase total, so a sharded job takes its own slice of the
    candidate list after the truncation rather than before it.
    """
    candidates = load_jsonl(candidate_manifest_path(config, artifact_root, row_phase, stage_id))[:count]
    window = shard_slice()
    if window is not None:
        candidates = candidates[window[0] : window[1]]
    targets = config["targets"]
    seeds = seeds_for(config, row_phase)
    predictor = predictor_record(config, predictor_id)
    plan: list[dict[str, Any]] = []
    for candidate in candidates:
        for target in targets:
            for seed in seeds:
                plan.append(
                    {
                        "candidate": candidate,
                        "target": target,
                        "seed": seed,
                        "predictor_id": str(predictor["id"]),
                    }
                )
    return plan


def base_row(
    config: dict[str, Any],
    *,
    item: dict[str, Any],
    row_phase: str,
    model_revision: str,
    controls: dict[str, dict[str, Any]],
    msa_identity: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Return the fields every row carries whatever its status.

    These are the fourteen `RAW_PREDICTION_FIELDS`, plus `origin_generator` for
    candidate rows in every cofold phase.

    Two of the fourteen say which alignment produced the row, and the contract
    requires both on every row so that a row can never leave the question open. An
    arm whose adapter is not wired to the target MSA gets null in both, which is that
    arm's answer rather than a gap. An arm that is wired to one has to name the file
    it consumed, and only the caller knows that, so it passes `msa_identity`. A caller
    that folds against an alignment and passes nothing gets neither field, and the
    stage output check reports the row as incomplete. That is the honest outcome: a
    null there would claim the fold used no alignment when it did.
    """
    candidate = item["candidate"]
    target = item["target"]
    row = {
        "target_id": str(target["target_id"]),
        "target_sha256": str(target["structure_sha256"]),
        "candidate_id": str(candidate["candidate_id"]),
        "predictor": item["predictor_id"],
        "model_revision": model_revision,
        "seed": int(item["seed"]),
        "phase": row_phase,
        "sequence_sha256": str(candidate["sequence_sha256"]),
        "design_pose_path": str(candidate["design_pose_path"]),
        "design_pose_sha256": str(candidate["design_pose_sha256"]),
        "chain_mapping": chain_mapping_for(config, target, candidate["candidate_id"], controls),
        "status": "failed",
    }
    if msa_identity is not None:
        row["msa_path"] = msa_identity["msa_path"]
        row["msa_sha256"] = msa_identity["msa_sha256"]
    elif not arm_accepts_target_msa(config, item["predictor_id"]):
        row["msa_path"] = None
        row["msa_sha256"] = None
    # A missing generator is intentionally left unstamped so incomplete lineage is
    # rejected by the downstream required-field checks instead of raising KeyError.
    origin_generator = candidate.get("origin_generator")
    if origin_generator:
        row["origin_generator"] = str(origin_generator)
    return row


def failed_row(
    binder_contract,
    row: dict[str, Any],
    *,
    failure_code: str,
    failure_reason: str,
) -> dict[str, Any]:
    """Return a complete failed row.

    `base` is the safe call. It copies the row, drops the six artifact fields
    and applies the failure fields, because the schema forbids a failed row from
    carrying an artifact path.
    """
    return binder_contract.write_failed_row(
        row["target_id"],
        row["candidate_id"],
        row["predictor"],
        row["seed"],
        failure_code,
        failure_reason,
        base=row,
    )


def short_reason(exc: BaseException) -> str:
    """Return a one-line reason that names the exception type."""
    text = str(exc).strip().splitlines()
    first = text[0] if text else ""
    return f"{type(exc).__name__}: {first}"[:500]


def run_arm(arm: ArmSpec, args: argparse.Namespace) -> int:
    from . import binder_contract

    config = load_json(args.config)
    row_phase = campaign_phase(args.stage)
    controls = control_records(config)
    model_revision = model_revision_for(config, arm.adapter_id)
    manifest_path, artifacts_attempt_dir = output_paths(
        config, args.stage, args.attempt_dir, args.phase
    )
    work_dir = artifacts_attempt_dir / args.phase / "arm-workspace"
    work_dir.mkdir(parents=True, exist_ok=True)

    targets = config["targets"]
    target_sequences = resolve_per_target(args.target_sequence, targets, "--target-sequence")
    target_a3m = (
        resolve_per_target(getattr(args, "target_msa_a3m", None), targets, "--target-msa-a3m")
        if arm.uses_target_msa
        else {}
    )
    hotspots = resolve_per_target(args.hotspot_residues, targets, "--hotspot-residues")

    plan = plan_predictions(
        config,
        stage_id=args.stage,
        row_phase=row_phase,
        artifact_root=args.artifact_root,
        count=args.count,
        predictor_id=arm.predictor_id,
    )
    writer = RowWriter(manifest_path)

    # Resolved before the model loads. Every one of these comes from an
    # argument or from configuration, so a gap is knowable without a GPU, and
    # discovering it after the weights load wastes the load.
    setup: dict[str, dict[str, Any]] = {}
    setup_errors: dict[str, tuple[str, str]] = {}
    # The alignment identity every row of this arm carries, by target. The Fast arm
    # leaves this empty and `base_row` writes null in both fields from the adapter's
    # own wiring. The Full arm starts each target at null and replaces it once the a3m
    # is resolved and hashed, so a target whose a3m never resolved writes null rather
    # than naming a file the run never read.
    msa_identities: dict[str, dict[str, str | None]] = {}
    targets_by_id = {str(item["target"]["target_id"]): item["target"] for item in plan}
    for target in targets_by_id.values():
        target_id = str(target["target_id"])
        if arm.uses_target_msa:
            msa_identities[target_id] = {"msa_path": None, "msa_sha256": None}
        try:
            resolved = {
                "sequence": resolve_target_sequence(target_id, target_sequences),
                "site_map": site_residue_map_for(config, target, hotspots),
                "a3m": require_target_a3m(target_id, target_a3m) if arm.uses_target_msa else None,
            }
            if resolved["a3m"] is not None:
                # Hashing here rather than at fold time means an unreadable a3m is
                # knowable before the weights load, which is what this block is for.
                msa_identities[target_id] = target_msa_identity(str(resolved["a3m"]))
            setup[target_id] = resolved
        except SiteMapUnavailable as exc:
            setup_errors[target_id] = (FAILURE_SITE_MAP_UNAVAILABLE, short_reason(exc))
        except TargetSequenceUnavailable as exc:
            setup_errors[target_id] = (FAILURE_TARGET_SEQUENCE_MISSING, short_reason(exc))
        except TargetMsaUnavailable as exc:
            setup_errors[target_id] = (FAILURE_MSA_LOAD, short_reason(exc))

    model = None
    model_error: str | None = None
    if setup:
        try:
            model = load_model(arm, args.device, model_revision=model_revision)
        except Exception as exc:  # noqa: BLE001
            model_error = short_reason(exc)
            traceback.print_exc()
    else:
        model_error = "every target is missing an input, so no weights were loaded"

    for target_id, resolved in sorted(setup.items()):
        mapping = chain_mapping_for(config, targets_by_id[target_id], "", controls)
        print(
            f"{arm.predictor_id}: {target_id} folds as target chain "
            f"{mapping['target']} then binder chain {mapping['binder']}, "
            f"{len(resolved['sequence'])} target residues",
            file=sys.stderr,
            flush=True,
        )

    msa_cache: dict[str, Any] = {}
    msa_errors: dict[str, str] = {}

    for index, item in enumerate(plan, start=1):
        row = base_row(
            config,
            item=item,
            row_phase=row_phase,
            model_revision=model_revision,
            controls=controls,
            msa_identity=msa_identities.get(str(item["target"]["target_id"])),
        )
        # Every per-prediction body is wrapped. A job that raises on candidate 3
        # of 40 otherwise throws away the GPU time spent on candidates 1 and 2.
        try:
            target = item["target"]
            candidate = item["candidate"]
            target_id = str(target["target_id"])

            if target_id in setup_errors:
                code, reason = setup_errors[target_id]
                writer.write(
                    failed_row(
                        binder_contract, row, failure_code=code, failure_reason=reason
                    )
                )
                continue
            if model is None:
                writer.write(
                    failed_row(
                        binder_contract,
                        row,
                        failure_code=FAILURE_MODEL_LOAD,
                        failure_reason=f"{arm.checkpoint} did not load: {model_error}",
                    )
                )
                continue

            resolved = setup[target_id]
            target_sequence = resolved["sequence"]
            site_map = resolved["site_map"]

            binder_path = candidate.get("sequence_path")
            if not binder_path:
                writer.write(
                    failed_row(
                        binder_contract,
                        row,
                        failure_code=FAILURE_BINDER_SEQUENCE_MISSING,
                        failure_reason=(
                            f"candidate {row['candidate_id']} carries no sequence_path"
                        ),
                    )
                )
                continue
            binder_sequence = read_fasta_sequence(Path(binder_path))

            target_msa = None
            if arm.uses_target_msa:
                if target_id in msa_errors:
                    writer.write(
                        failed_row(
                            binder_contract,
                            row,
                            failure_code=FAILURE_MSA_LOAD,
                            failure_reason=msa_errors[target_id],
                        )
                    )
                    continue
                if target_id not in msa_cache:
                    try:
                        msa_cache[target_id] = load_target_msa(
                            Path(resolved["a3m"]), target_sequence, work_dir
                        )
                    except Exception as exc:  # noqa: BLE001
                        msa_errors[target_id] = short_reason(exc)
                        traceback.print_exc()
                        writer.write(
                            failed_row(
                                binder_contract,
                                row,
                                failure_code=FAILURE_MSA_LOAD,
                                failure_reason=msa_errors[target_id],
                            )
                        )
                        continue
                target_msa = msa_cache[target_id]

            print(
                f"{arm.predictor_id}: prediction {index} of {len(plan)}, "
                f"candidate {row['candidate_id']} seed {row['seed']}",
                file=sys.stderr,
                flush=True,
            )
            try:
                prediction = fold_one(
                    arm,
                    model,
                    target_chain=row["chain_mapping"]["target"],
                    target_sequence=target_sequence,
                    binder_chain=row["chain_mapping"]["binder"],
                    binder_sequence=binder_sequence,
                    target_msa=target_msa,
                    seed=int(row["seed"]),
                    fused_backend_max_length=args.fused_backend_max_length,
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                writer.write(
                    failed_row(
                        binder_contract,
                        row,
                        failure_code=FAILURE_PREDICTION,
                        failure_reason=short_reason(exc),
                    )
                )
                continue

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
                complex_cif=prediction.complex.to_mmcif(),
                pae=pae_matrix(prediction),
                chain_mapping=row["chain_mapping"],
                reference_cif=Path(row["design_pose_path"]),
                site_residue_map=site_map,
                model_revision=model_revision,
                target_sequence=target_sequence,
                binder_sequence=binder_sequence,
                extra={
                    # These four are the contract's REQUIRED_EXTRA_FIELDS. No
                    # binder_metrics function returns them, so the arm supplies
                    # them. A name the identity block already owns raises, so
                    # nothing else from the row belongs here.
                    "target_sha256": row["target_sha256"],
                    "sequence_sha256": row["sequence_sha256"],
                    "design_pose_sha256": row["design_pose_sha256"],
                    **prediction_extra(prediction),
                },
            )
            merged = dict(row)
            merged.update(written)
            writer.write(merged)
        except Exception as exc:  # noqa: BLE001
            # The fold has its own handler above, so anything reaching here
            # failed while reading an input or while measuring and writing.
            traceback.print_exc()
            failure_code = getattr(exc, "failure_code", FAILURE_ARTIFACT_WRITE)
            writer.write(
                failed_row(
                    binder_contract,
                    row,
                    failure_code=failure_code,
                    failure_reason=short_reason(exc),
                )
            )

    writer.close()
    print(
        f"{arm.predictor_id}: wrote {writer.count} rows, {writer.failed} failed, "
        f"to {manifest_path}",
        file=sys.stderr,
        flush=True,
    )
    return 1 if model is None else 0


def parse_outputs(args: argparse.Namespace) -> int:
    """Check the stage's own outputs and write the parser result.

    This runs after `merge-shards` has folded the shard trees together, so it
    reads the tree the stage contract names rather than a shard's copy.
    """
    config = load_json(args.config)
    stage = stage_record(config, args.stage)
    files: list[Path] = []
    parsed_count = 0
    errors: list[str] = []
    import hashlib

    for output in stage["outputs"]:
        pattern = render(output["path_template"], attempt_dir=args.attempt_dir, phase=args.phase)
        for value in sorted(glob.glob(pattern, recursive=True)):
            path = Path(value)
            if not path.is_file():
                continue
            files.append(path)
            try:
                if output["kind"] == "jsonl":
                    for line in path.read_text().splitlines():
                        if line.strip():
                            json.loads(line)
                            parsed_count += 1
                elif output["kind"] == "json":
                    json.loads(path.read_text())
                    parsed_count += 1
                else:
                    parsed_count += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {short_reason(exc)}")
    result_path = args.attempt_dir / args.phase / "parser-result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "ok": bool(files) and not errors,
                "parsed_count": parsed_count,
                "rejected_count": len(errors),
                "errors": errors,
                "source_output_hashes": sorted(
                    hashlib.sha256(path.read_bytes()).hexdigest() for path in files
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0 if files and not errors else 1


def main(arm: ArmSpec = ARM_FULL) -> int:
    args = build_parser(arm).parse_args()
    if args.command == "toolcheck":
        # The profile's toolcheck_argv is `pip show esm`, because importing the
        # package fires Triton autotune and needs a live CUDA driver.
        print(f"{arm.adapter_id} entry script ok, checkpoint {arm.checkpoint}")
        return 0
    if args.command == "parse":
        return parse_outputs(args)
    return run_arm(arm, args)


if __name__ == "__main__":
    raise SystemExit(main())
