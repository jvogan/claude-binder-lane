#!/usr/bin/env python3
"""Declare ESMFold2 whole-surface binder design with optional epitope aiming.

This adapter owns the native gradient-guided ESMFold2 design contract. It
generates a binder backbone and sequence together when a prepared runtime
supplies the Experimental design implementation and its checkpoints.

The native source has no binding-site argument. In unaimed mode, its interface
contact loss averages over the target surface. A caller must acknowledge that
mode explicitly. When a caller supplies epitope residues, this adapter passes a
target-only contact mask to the native loop. The aiming change is implemented
and remains unvalidated.

The module imports only the standard library. Torch and ESMFold2 imports remain
inside the guarded runtime boundary so that contract checks run without a GPU
stack.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from claude_binder.native_design.constants import (
    DEFAULT_MUTABLE_AMINO_ACIDS,
    STANDARD_AMINO_ACID_SET,
)
from claude_binder.native_design.constraints import build_gradient_mask
from claude_binder.native_design.errors import (
    DesignRefusal,
    FAILURE_ALPHABET_INVALID,
    FAILURE_CRITIC_FOLD_FAILED,
    FAILURE_CUDA_OOM_EXHAUSTED,
    FAILURE_DESIGN_AUTH_REJECTED,
    FAILURE_DESIGN_CHECKPOINT_UNAVAILABLE,
    FAILURE_DESIGN_EGRESS_BLOCKED,
    FAILURE_EPITOPE_OUT_OF_RANGE,
    FAILURE_NO_MUTABLE_POSITIONS,
    FAILURE_PATTERN_COLLISION,
    FAILURE_UNAIMED_NOT_ACKNOWLEDGED,
)
from claude_binder.native_design.prosite import build_prosite_prompt, parse_prosite

from .declared_artifacts import DeclaredArtifactError, input_files, load_plan


NATIVE_DESIGN_STAGE = "design-native"
TARGET_MANIFEST_ARTIFACT = "target-manifest"

CANDIDATES_ARTIFACT = "native-design-candidates"
COMPLEXES_ARTIFACT = "native-design-complexes"
TRAJECTORY_ARTIFACT = "native-design-trajectory"
LOGITS_ARTIFACT = "native-design-logits"

EPITOPE_TOKEN_RE = re.compile(r"(\d+)(?:-(\d+))?")

SUPPORTED_BINDER_MODES = ("minibinder", "antibody_framework")
SUPPORTED_ANTIBODY_FRAMEWORKS = ("trastuzumab", "atezolizumab", "ocankitug")


@dataclass(frozen=True)
class ArtifactDeclaration:
    """One artifact this adapter reads or writes."""

    artifact_id: str
    artifact_type: str
    path_template: str
    purpose: str


DECLARED_INPUT_ARTIFACTS = (
    ArtifactDeclaration(
        TARGET_MANIFEST_ARTIFACT,
        "target-manifest",
        "declared input from target-prepare",
        "Read the normalized target identity and its design-target sequence.",
    ),
)

DECLARED_OUTPUT_ARTIFACTS = (
    ArtifactDeclaration(
        CANDIDATES_ARTIFACT,
        "candidate-manifest",
        "{{attempt_dir}}/{{phase}}/candidates.jsonl",
        "Write one candidate row for each native design.",
    ),
    ArtifactDeclaration(
        COMPLEXES_ARTIFACT,
        "pdb",
        "{{attempt_dir}}/{{phase}}/complexes/*.pdb",
        "Write one native-design complex structure per scored candidate.",
    ),
    ArtifactDeclaration(
        TRAJECTORY_ARTIFACT,
        "json",
        "{{attempt_dir}}/{{phase}}/trajectory.json",
        "Write the per-step native-design loss trajectory.",
    ),
    ArtifactDeclaration(
        LOGITS_ARTIFACT,
        "npz",
        "{{attempt_dir}}/{{phase}}/logits.npz",
        "Write final native-design logits.",
    ),
)


@dataclass(frozen=True)
class NativeDesignConfig:
    """Cited native-design configuration with unresolved values left unset."""

    inversion_repositories: tuple[str, ...] = (
        "biohub/ESMFold2-Experimental-Fast",
        "biohub/ESMFold2-Experimental-Fast-Cutoff2025",
    )
    hero_critic_repositories: tuple[str, ...] = (
        "biohub/ESMFold2-Experimental-Fast",
        "biohub/ESMFold2-Experimental-Fast-Cutoff2025",
        "biohub/ESMFold2-Experimental",
        "biohub/ESMFold2-Experimental-Cutoff2025",
    )
    esmc_repository: str = "biohub/ESMC-6B"
    lm_dropout_inversion: float = 0.5
    lm_dropout_critic: float = 0.25
    reuse_esmc: bool = True
    checkpoint_lm: bool = False
    compile: bool = False
    kernel_backend: None = None
    learning_rate: float = 0.1
    temperature_min: float = 0.01
    steps: int | None = None
    intra_contact_weight: float = 0.5
    inter_contact_weight: float = 0.5
    globularity_weight: float = 0.2
    plm_weight_antibody: float = 0.05
    plm_weight: float = 0.15
    esmc_mask_fraction: float = 0.15
    critic_num_loops: int = 3
    critic_num_sampling_steps: int | None = None
    pi_max: float = 6.0
    top_n: int = 84
    save_confidence_arrays: bool = False

    def __post_init__(self) -> None:
        if self.kernel_backend is not None:
            raise ValueError(
                "ESMFold2 Experimental design variants require kernel_backend=None"
            )


DEFAULT_CONFIG = NativeDesignConfig()

_UNSET_CONFIG_VALUE = re.compile(r"^__REQUIRED__")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _native_design_config_values(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return campaign-resolved native design values from supported locations."""

    values: dict[str, Any] = {}
    for candidate in (
        config.get("native_design"),
        config.get("native_design_config"),
        config.get("profile", {}).get("native_design_config")
        if isinstance(config.get("profile"), Mapping)
        else None,
        config.get("profile", {}).get("native_design_operator_values")
        if isinstance(config.get("profile"), Mapping)
        else None,
    ):
        if isinstance(candidate, Mapping):
            values.update(candidate)
    return values


def _required_config_value(values: Mapping[str, Any], name: str) -> Any:
    value = values.get(name)
    if value is None or (isinstance(value, str) and _UNSET_CONFIG_VALUE.match(value)):
        raise ValueError(f"campaign native design configuration leaves required {name} unset")
    return value


def load_native_design_config(config_path: Path) -> NativeDesignConfig:
    """Load campaign values and refuse unresolved native design controls."""

    config = _load_json(config_path, "resolved configuration")
    values = _native_design_config_values(config)
    values = dict(values)
    if "loss_weights" in values:
        loss_weights = values.pop("loss_weights")
        if not isinstance(loss_weights, Sequence) or isinstance(loss_weights, (str, bytes)) or len(loss_weights) != 3:
            raise ValueError("campaign native design configuration loss_weights must hold three values")
        values.update(
            {
                "intra_contact_weight": loss_weights[0],
                "inter_contact_weight": loss_weights[1],
                "globularity_weight": loss_weights[2],
            }
        )
    if "inversion_repos" in values:
        values["inversion_repositories"] = tuple(values.pop("inversion_repos"))
    if "hero_critic_repos" in values:
        values["hero_critic_repositories"] = tuple(values.pop("hero_critic_repos"))
    known = {field.name for field in fields(NativeDesignConfig)}
    selected = {name: value for name, value in values.items() if name in known}
    selected["steps"] = _required_config_value(values, "steps")
    selected["critic_num_sampling_steps"] = _required_config_value(values, "critic_num_sampling_steps")
    try:
        selected["steps"] = int(selected["steps"])
        selected["critic_num_sampling_steps"] = int(selected["critic_num_sampling_steps"])
    except (TypeError, ValueError) as exc:
        raise ValueError("campaign native design steps and critic_num_sampling_steps must be integers") from exc
    if selected["steps"] < 1 or selected["critic_num_sampling_steps"] < 1:
        raise ValueError("campaign native design steps and critic_num_sampling_steps must be positive")
    try:
        return replace(DEFAULT_CONFIG, **selected)
    except TypeError as exc:
        raise ValueError(f"campaign native design configuration has an invalid value: {exc}") from exc


@dataclass(frozen=True)
class NativeDesignRequest:
    """Validated native-design request passed to the guarded runtime."""

    target_sequence: str
    epitope_indices_0based: tuple[int, ...] | None
    binder_mode: str
    min_length: int | None
    max_length: int | None
    antibody_framework: str | None
    n_designs: int
    seeds: int
    batch_size: int
    steps: int
    seed_base: int
    allowed_amino_acids: str
    pattern: str | None
    pattern_anchor: str | None
    pattern_start: int | None
    pattern_gap: str
    use_scaling_critics: bool
    aiming_status: str


@dataclass(frozen=True)
class NativeDesignOutputPaths:
    """Concrete paths for the four native-design artifacts."""

    candidates: Path
    complexes_dir: Path
    trajectory: Path
    logits: Path


def validate_allowed_amino_acids(value: str | None) -> str:
    """Return the global mutable alphabet after validation."""

    if value is None:
        return DEFAULT_MUTABLE_AMINO_ACIDS
    normalized = "".join(dict.fromkeys(value.upper()))
    invalid = sorted({residue for residue in normalized if residue not in STANDARD_AMINO_ACID_SET})
    if invalid:
        raise DesignRefusal(
            FAILURE_ALPHABET_INVALID,
            f"allowed_amino_acids contains invalid residues: {''.join(invalid)}",
        )
    if not normalized:
        raise DesignRefusal(FAILURE_ALPHABET_INVALID, "allowed_amino_acids is empty")
    return normalized


def normalize_target_sequence(value: str) -> str:
    """Validate a target sequence with optional pipe-separated chains."""

    chains = [chain.strip().upper() for chain in value.split("|")]
    if not chains or any(not chain for chain in chains):
        raise ValueError("target_sequence must contain one or more nonempty chains")
    invalid = sorted({residue for chain in chains for residue in chain if residue not in STANDARD_AMINO_ACID_SET})
    if invalid:
        raise ValueError(f"target_sequence contains invalid residues: {''.join(invalid)}")
    return "|".join(chains)


def target_residue_length(target_sequence: str) -> int:
    """Return the target residue count without chain delimiters."""

    return len(target_sequence.replace("|", ""))


def parse_epitope_residues(value: str, target_length: int) -> tuple[int, ...]:
    """Parse one-based residue indices and ranges into sorted zero-based indices."""

    if target_length < 1:
        raise ValueError("target length must be positive")
    values: set[int] = set()
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise DesignRefusal(FAILURE_EPITOPE_OUT_OF_RANGE, "epitope_residues is empty")
    for part in parts:
        matched = EPITOPE_TOKEN_RE.fullmatch(part)
        if matched is None:
            raise DesignRefusal(
                FAILURE_EPITOPE_OUT_OF_RANGE,
                f"epitope residue entry is not a one-based index or range: {part!r}",
            )
        first = int(matched.group(1))
        last = int(matched.group(2)) if matched.group(2) is not None else first
        if first < 1 or last < first or last > target_length:
            raise DesignRefusal(
                FAILURE_EPITOPE_OUT_OF_RANGE,
                f"epitope range {part!r} is outside target residues 1-{target_length}",
            )
        values.update(range(first - 1, last))
    return tuple(sorted(values))


def _validate_binder_mode(args: argparse.Namespace) -> None:
    """Validate the binder prompt source without selecting an invented length."""

    if args.binder_mode == "minibinder":
        if args.min_length is None or args.max_length is None:
            raise ValueError("minibinder mode requires --min-length and --max-length")
        if args.min_length < 1 or args.max_length < args.min_length:
            raise ValueError("minibinder lengths must be positive and ordered")
        if args.antibody_framework is not None:
            raise ValueError("minibinder mode does not accept --antibody-framework")
        return
    if args.antibody_framework not in SUPPORTED_ANTIBODY_FRAMEWORKS:
        raise ValueError(
            "antibody_framework mode requires one framework: "
            + ", ".join(SUPPORTED_ANTIBODY_FRAMEWORKS)
        )
    if args.min_length is not None or args.max_length is not None:
        raise ValueError("antibody_framework mode does not accept minibinder length bounds")


def _validate_pattern(
    args: argparse.Namespace,
    alphabet: str,
) -> None:
    """Validate every pattern before the runtime selects a prompt length.

    Variable-length minibinders and antibody frameworks receive their final
    placement in `native_design.runtime.prepare_runtime_request`. Parsing here
    prevents invalid syntax and repeat bounds from bypassing validation.
    """

    if args.pattern is None:
        return
    try:
        parsed = parse_prosite(args.pattern)
        if args.binder_mode != "minibinder" or args.min_length != args.max_length:
            if not parsed.elements:
                raise ValueError("PROSITE pattern has no elements")
            return
        layout = build_prosite_prompt(
            args.pattern,
            args.min_length,
            gap=args.pattern_gap,
            anchor=args.pattern_anchor,
            start=args.pattern_start,
        )
        masks = build_gradient_mask(
            layout["prompt"],
            allowed_amino_acids=alphabet,
            position_allowed=layout["position_allowed"],
        )
    except DesignRefusal:
        raise
    except ValueError as exc:
        raise DesignRefusal(FAILURE_PATTERN_COLLISION, str(exc)) from exc
    if not masks.mutable_positions:
        raise DesignRefusal(
            FAILURE_NO_MUTABLE_POSITIONS,
            "the binder prompt has no mutable positions",
        )


def validate_request(args: argparse.Namespace) -> NativeDesignRequest:
    """Validate CPU-only native-design inputs before any runtime import."""

    try:
        target_sequence = normalize_target_sequence(args.target_sequence)
        _validate_binder_mode(args)
        if args.n_designs < 1 or args.seeds < 1 or args.batch_size < 1:
            raise ValueError("n_designs, seeds, and batch_size must be positive")
        if args.n_designs != args.seeds * args.batch_size:
            raise ValueError("n_designs must equal seeds multiplied by batch_size")
        if args.steps < 1:
            raise ValueError("steps must be positive")
        if args.seed_base < 0:
            raise ValueError("seed_base must be nonnegative")
        alphabet = validate_allowed_amino_acids(args.allowed_amino_acids)
        _validate_pattern(args, alphabet)
    except DesignRefusal:
        raise
    except ValueError as exc:
        raise DesignRefusal(FAILURE_ALPHABET_INVALID, str(exc)) from exc

    if args.epitope_residues is None:
        if not args.acknowledge_unaimed:
            raise DesignRefusal(
                FAILURE_UNAIMED_NOT_ACKNOWLEDGED,
                "whole-surface binder design requires --acknowledge-unaimed when epitope_residues is absent",
            )
        epitope_indices = None
        aiming_status = "whole-surface"
    else:
        epitope_indices = parse_epitope_residues(
            args.epitope_residues,
            target_residue_length(target_sequence),
        )
        aiming_status = "implemented-unvalidated"

    return NativeDesignRequest(
        target_sequence=target_sequence,
        epitope_indices_0based=epitope_indices,
        binder_mode=args.binder_mode,
        min_length=args.min_length,
        max_length=args.max_length,
        antibody_framework=args.antibody_framework,
        n_designs=args.n_designs,
        seeds=args.seeds,
        batch_size=args.batch_size,
        steps=args.steps,
        seed_base=args.seed_base,
        allowed_amino_acids=alphabet,
        pattern=args.pattern,
        pattern_anchor=args.pattern_anchor,
        pattern_start=args.pattern_start,
        pattern_gap=args.pattern_gap,
        use_scaling_critics=bool(args.use_scaling_critics),
        aiming_status=aiming_status,
    )


def declared_read_arguments(args: argparse.Namespace) -> tuple[Path, ...]:
    """Return every argument path this adapter reads during native design."""

    return (args.config, args.plan, args.receipts_dir, args.artifact_root)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{label} is unreadable: {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return value


def load_declared_target_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Read the target manifest through the stage's declared input artifact."""

    config = _load_json(args.config, "resolved configuration")
    plan = load_plan(args.plan, config)
    _, paths = input_files(
        plan,
        args.receipts_dir,
        args.stage,
        artifact_id=args.target_artifact_id,
        source_stage_id=args.target_stage_id,
        artifact_type="target-manifest",
        phase_preference=("single", "scale", "smoke"),
    )
    if len(paths) != 1:
        raise ValueError(
            f"declared target artifact {args.target_artifact_id} supplies {len(paths)} files; expected one"
        )
    path = paths[0]
    return path, _load_json(path, "target manifest")


def target_sequence_from_manifest(manifest: Mapping[str, Any]) -> str:
    """Return the declared design-target sequence from one target manifest."""

    chains = manifest.get("chains")
    if not isinstance(chains, list):
        raise ValueError("target manifest has no chains list")
    matches = [
        chain
        for chain in chains
        if isinstance(chain, Mapping) and chain.get("role") == "design-target"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("sequence"), str):
        raise ValueError("target manifest must declare one design-target sequence")
    return normalize_target_sequence(str(matches[0]["sequence"]))


def verify_declared_target_sequence(request: NativeDesignRequest, manifest: Mapping[str, Any]) -> None:
    """Refuse a target sequence that differs from the declared target artifact."""

    declared = target_sequence_from_manifest(manifest)
    if request.target_sequence.replace("|", "") != declared.replace("|", ""):
        raise ValueError("target_sequence differs from the declared target-manifest design-target sequence")


def native_design_output_paths(args: argparse.Namespace) -> NativeDesignOutputPaths:
    """Return output paths that match the adapter's declared artifact patterns."""

    attempt_dir = args.attempt_dir.expanduser().resolve()
    phase_dir = (attempt_dir / args.phase).resolve()
    if attempt_dir not in phase_dir.parents:
        raise ValueError(f"phase output escapes attempt directory: {phase_dir}")
    return NativeDesignOutputPaths(
        candidates=phase_dir / "candidates.jsonl",
        complexes_dir=phase_dir / "complexes",
        trajectory=phase_dir / "trajectory.json",
        logits=phase_dir / "logits.npz",
    )


def _write_json(path: Path, value: Any) -> None:
    """Write one JSON artifact with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one candidate row per JSONL line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _manifest_value(manifest: Mapping[str, Any], name: str) -> str:
    value = manifest.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"target manifest omits required {name}")
    if name.endswith("sha256") and _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"target manifest {name} is not a SHA-256 digest")
    return value


def _candidate_value(candidate: Mapping[str, Any], name: str, expected: type[Any]) -> Any:
    value = candidate.get(name)
    if not isinstance(value, expected) or (isinstance(value, str) and not value):
        raise ValueError(f"native-design candidate record omits required {name}")
    return value


def _host_array(value: Any, numpy: Any) -> Any:
    """Detach tensor values and move CUDA storage to host memory for NumPy."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return numpy.asarray(value)


def write_native_design_artifacts(
    args: argparse.Namespace,
    *,
    candidates: Sequence[Mapping[str, Any]],
    complexes: Mapping[str, str],
    trajectory: Mapping[str, Any],
    logits: Mapping[str, Any],
    target_manifest: Mapping[str, Any] | None = None,
) -> NativeDesignOutputPaths:
    """Write declared artifacts and complete candidate-manifest lineage fields."""

    paths = native_design_output_paths(args)
    if not candidates:
        raise ValueError("native design produced no candidate records")
    if target_manifest is None:
        raise ValueError("target manifest is required to write candidate manifest records")
    target_id = _manifest_value(target_manifest, "target_id")
    target_sha256 = _manifest_value(target_manifest, "target_sha256")
    residue_map_sha256 = _manifest_value(target_manifest, "residue_map_sha256")
    candidate_ids: set[str] = set()
    for candidate in candidates:
        candidate_id = _candidate_value(candidate, "candidate_id", str)
        if candidate_id in candidate_ids:
            raise ValueError(f"native-design candidate_id repeats: {candidate_id}")
        candidate_ids.add(candidate_id)
    missing_complexes = sorted(candidate_ids - set(complexes))
    if missing_complexes:
        raise ValueError("native design wrote no complex PDB for: " + ", ".join(missing_complexes))

    paths.complexes_dir.mkdir(parents=True, exist_ok=True)
    sequences_dir = paths.candidates.parent / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for candidate_id in sorted(candidate_ids):
        pdb = complexes[candidate_id]
        if not isinstance(pdb, str) or not pdb.strip():
            raise ValueError(f"native design returned an empty complex PDB for {candidate_id}")
        pdb_path = paths.complexes_dir / f"{candidate_id}.pdb"
        pdb_path.write_text(pdb, encoding="utf-8")
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        binder_sequence = _candidate_value(candidate, "binder_sequence", str)
        generator_seed = _candidate_value(candidate, "generator_seed", int)
        sequence_path = sequences_dir / f"{candidate_id}.fasta"
        sequence_path.write_text(f">{candidate_id}\n{binder_sequence}\n", encoding="utf-8")
        pdb_path = paths.complexes_dir / f"{candidate_id}.pdb"
        row = dict(candidate)
        row.update(
            {
                "target_id": target_id,
                "target_sha256": target_sha256,
                "candidate_id": candidate_id,
                "parent_candidate_id": None,
                "origin_generator": "esmfold2-native-design",
                "generator_mode": "sequence-structure-codesign",
                "sequence_designer": "esmfold2-native-design",
                "generator_seed": generator_seed,
                "sequence_path": str(sequence_path.resolve()),
                "sequence_sha256": _sha256_file(sequence_path),
                "sequence_length": len(binder_sequence),
                "structure_path": str(pdb_path.resolve()),
                "structure_sha256": _sha256_file(pdb_path),
                "design_pose_path": str(pdb_path.resolve()),
                "design_pose_sha256": _sha256_file(pdb_path),
                "residue_map_sha256": residue_map_sha256,
                "optimization_round": 0,
                "last_optimizer": "esmfold2-native-design",
                "status": "generated",
            }
        )
        rows.append(row)
    _write_jsonl(paths.candidates, rows)
    _write_json(paths.trajectory, trajectory)
    try:
        numpy = importlib.import_module("numpy")
    except (ImportError, ModuleNotFoundError) as exc:
        raise DesignRefusal(
            FAILURE_DESIGN_CHECKPOINT_UNAVAILABLE,
            "numpy is required to write the declared logits.npz artifact",
        ) from exc
    numpy.savez(paths.logits, **{name: _host_array(value, numpy) for name, value in logits.items()})
    return paths


def _runtime_failure_from_exception(exc: BaseException) -> DesignRefusal:
    """Map known runtime failures onto the published stable code set."""

    text = str(exc).lower()
    if "401" in text or "unauthorized" in text or "forbidden" in text:
        return DesignRefusal(FAILURE_DESIGN_AUTH_REJECTED, str(exc))
    if "network" in text or "connection" in text or "egress" in text:
        return DesignRefusal(FAILURE_DESIGN_EGRESS_BLOCKED, str(exc))
    if "out of memory" in text or "cuda oom" in text:
        return DesignRefusal(FAILURE_CUDA_OOM_EXHAUSTED, str(exc))
    return DesignRefusal(FAILURE_DESIGN_CHECKPOINT_UNAVAILABLE, str(exc))


def invoke_native_design(
    request: NativeDesignRequest,
    config: NativeDesignConfig,
) -> Mapping[str, Any]:
    """Load the Experimental-only runtime after every CPU contract check.

    The adapter deliberately avoids the folding adapter's release-checkpoint
    loader. Experimental checkpoints belong here because gradients flow through
    their input soft sequence. The platform must provide the native gradient
    implementation and preloaded checkpoints before this route can execute.
    """

    if config.kernel_backend is not None:
        raise ValueError("ESMFold2 Experimental design variants require kernel_backend=None")
    try:
        runtime = importlib.import_module("claude_binder.native_design.runtime")
    except (ImportError, ModuleNotFoundError) as exc:
        raise DesignRefusal(
            FAILURE_DESIGN_CHECKPOINT_UNAVAILABLE,
            "the native Experimental gradient-design runtime is unavailable",
        ) from exc

    # The runtime boundary receives None only. A fast kernel backend raises an
    # Experimental confidence-head dtype error, so this adapter never exposes one.
    try:
        run = getattr(runtime, "run_native_design")
        result = run(request=request, config=config, kernel_backend=None)
        if not isinstance(result, Mapping):
            raise TypeError("native gradient-design runtime returned no artifact mapping")
        required = {"candidates", "complexes", "trajectory", "logits"}
        missing = sorted(required - set(result))
        if missing:
            raise ValueError("native gradient-design runtime omitted: " + ", ".join(missing))
        return result
    except DesignRefusal:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _runtime_failure_from_exception(exc) from exc


def run(args: argparse.Namespace) -> int:
    """Validate declared inputs, then enter the guarded native-design runtime."""

    request = validate_request(args)
    try:
        _, manifest = load_declared_target_manifest(args)
        verify_declared_target_sequence(request, manifest)
    except DeclaredArtifactError as exc:
        raise DesignRefusal(FAILURE_DESIGN_CHECKPOINT_UNAVAILABLE, str(exc)) from exc
    except ValueError as exc:
        raise DesignRefusal(FAILURE_DESIGN_CHECKPOINT_UNAVAILABLE, str(exc)) from exc
    config = load_native_design_config(args.config)
    result = invoke_native_design(request, config)
    write_native_design_artifacts(
        args,
        candidates=result["candidates"],
        complexes=result["complexes"],
        trajectory=result["trajectory"],
        logits=result["logits"],
        target_manifest=manifest,
    )
    return 0


def toolcheck() -> int:
    """Report the CPU contract without importing Torch or loading a checkpoint."""

    print("esmfold2-native-design: contract ready; Experimental runtime remains guarded")
    return 0


def _candidate_schema_errors(row: Mapping[str, Any]) -> list[str]:
    """Validate each candidate row against the shipped manifest schema subset."""

    schema_path = Path(__file__).resolve().parents[1] / "data" / "schemas" / "candidate-manifest.schema.json"
    schema = _load_json(schema_path, "candidate manifest schema")
    errors: list[str] = []
    for name in schema.get("required", []):
        if name not in row:
            errors.append(f"candidate manifest row omits required field {name}")
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return errors + ["candidate manifest schema properties are invalid"]
    type_map = {"string": str, "integer": int}
    for name, rules in properties.items():
        if name not in row or not isinstance(rules, Mapping):
            continue
        value = row[name]
        allowed_types = rules.get("type")
        if isinstance(allowed_types, str):
            allowed_types = [allowed_types]
        if isinstance(allowed_types, list):
            valid_type = False
            for allowed in allowed_types:
                if allowed == "null" and value is None:
                    valid_type = True
                elif allowed in type_map and isinstance(value, type_map[allowed]) and not isinstance(value, bool):
                    valid_type = True
            if not valid_type:
                errors.append(f"candidate manifest field {name} has an invalid type")
        enum = rules.get("enum")
        if isinstance(enum, list) and value not in enum:
            errors.append(f"candidate manifest field {name} has an undeclared value")
        if rules.get("pattern") == "^[0-9a-f]{64}$" and value is not None:
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                errors.append(f"candidate manifest field {name} is not a SHA-256 digest")
        minimum = rules.get("minimum")
        if isinstance(minimum, int) and isinstance(value, int) and value < minimum:
            errors.append(f"candidate manifest field {name} is below its minimum")
    return errors


def parse_outputs(args: argparse.Namespace) -> int:
    """Parse disk artifacts, write a receipt, and fail when any artifact is absent."""
    attempt_dir = args.attempt_dir.expanduser().resolve()
    phase_dir = (attempt_dir / args.phase).resolve()
    result_path = phase_dir / "parser-result.json"
    files: list[Path] = []
    errors: list[str] = []
    parsed_count = 0
    try:
        config = _load_json(args.config, "resolved configuration")
        stages = config.get("stages", [])
        stage_matches = [item for item in stages if isinstance(item, dict) and item.get("stage_id") == args.stage]
        if len(stage_matches) != 1:
            raise ValueError(f"resolved configuration declares {len(stage_matches)} stages named {args.stage}")
        stage = stage_matches[0]
        output_ids = {str(item.get("artifact_id")) for item in stage.get("outputs", []) if isinstance(item, dict)}
        expected = {declaration.artifact_id for declaration in DECLARED_OUTPUT_ARTIFACTS}
        missing = sorted(expected - output_ids)
        if missing:
            raise ValueError("stage declares no native-design outputs: " + ", ".join(missing))
    except ValueError as exc:
        errors.append(str(exc))

    paths = native_design_output_paths(args)
    candidate_rows: list[Mapping[str, Any]] = []
    if not paths.candidates.is_file():
        errors.append(f"missing declared candidates artifact: {paths.candidates}")
    else:
        files.append(paths.candidates)
        for line_number, line in enumerate(paths.candidates.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError("row is not a JSON object")
                row_errors = _candidate_schema_errors(row)
                if row_errors:
                    raise ValueError("; ".join(row_errors))
                candidate_rows.append(row)
                parsed_count += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{paths.candidates}:{line_number}: {type(exc).__name__}: {exc}")
        if not candidate_rows:
            errors.append(f"declared candidates artifact has no parseable rows: {paths.candidates}")
    for row in candidate_rows:
        candidate_id = row.get("candidate_id")
        pdb_path = paths.complexes_dir / f"{candidate_id}.pdb"
        if not pdb_path.is_file():
            errors.append(f"missing declared complex PDB for {candidate_id}: {pdb_path}")
            continue
        text = pdb_path.read_text(encoding="utf-8")
        if not text.strip():
            errors.append(f"declared complex PDB is empty: {pdb_path}")
            continue
        files.append(pdb_path)
    if not paths.trajectory.is_file():
        errors.append(f"missing declared trajectory artifact: {paths.trajectory}")
    else:
        files.append(paths.trajectory)
        try:
            trajectory = json.loads(paths.trajectory.read_text(encoding="utf-8"))
            if not isinstance(trajectory, (dict, list)):
                raise ValueError("trajectory is not a JSON object or array")
            parsed_count += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{paths.trajectory}: {type(exc).__name__}: {exc}")
    if not paths.logits.is_file():
        errors.append(f"missing declared logits artifact: {paths.logits}")
    else:
        files.append(paths.logits)
        try:
            numpy = importlib.import_module("numpy")
            with numpy.load(paths.logits, allow_pickle=False) as archive:
                if not archive.files:
                    raise ValueError("logits archive has no arrays")
            parsed_count += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{paths.logits}: {type(exc).__name__}: {exc}")
    result = {
        "ok": bool(candidate_rows) and not errors,
        "parsed_count": parsed_count,
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": sorted({_sha256_file(path) for path in files if path.is_file()}),
    }
    _write_json(result_path, result)
    for error in errors:
        print(f"{FAILURE_CRITIC_FOLD_FAILED}: {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


def _add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", default=NATIVE_DESIGN_STAGE)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--target-stage-id", default="target-prepare")
    parser.add_argument("--target-artifact-id", default=TARGET_MANIFEST_ARTIFACT)


def _add_design_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-sequence", required=True)
    parser.add_argument("--epitope-residues", default=None)
    parser.add_argument("--acknowledge-unaimed", action="store_true")
    parser.add_argument("--binder-mode", choices=SUPPORTED_BINDER_MODES, required=True)
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--antibody-framework", default=None)
    parser.add_argument("--n-designs", type=int, required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--allowed-amino-acids", default=None)
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--pattern-anchor", choices=("n", "c", "center"), default=None)
    parser.add_argument("--pattern-start", type=int, default=None)
    parser.add_argument("--pattern-gap", choices=("min", "max"), default="min")
    parser.add_argument("--use-scaling-critics", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    """Build the adapter command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("toolcheck", help="Check the CPU contract without loading a model.")
    run_parser = subparsers.add_parser("run", help="Run native whole-surface binder design.")
    _add_stage_arguments(run_parser)
    _add_design_arguments(run_parser)
    parse_parser = subparsers.add_parser("parse", help="Validate declared native-design outputs.")
    _add_stage_arguments(parse_parser)
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one native-design adapter command."""

    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected subcommand and emit stable failure codes."""

    args = parse_arguments(argv)
    try:
        if args.command == "toolcheck":
            return toolcheck()
        if args.command == "parse":
            return parse_outputs(args)
        return run(args)
    except DesignRefusal as exc:
        print(f"{exc.code}: {exc.detail}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"{FAILURE_DESIGN_CHECKPOINT_UNAVAILABLE}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"{FAILURE_DESIGN_CHECKPOINT_UNAVAILABLE}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
