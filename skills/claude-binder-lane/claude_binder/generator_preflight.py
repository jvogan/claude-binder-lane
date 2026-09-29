"""Free, local validation for a backbone generation request.

The paid adapter remains the authority for command execution. This module only
reads JSON, PDB or mmCIF files, and the filesystem. It never starts a subprocess
and it never makes a network request.

``preflight_campaign`` accepts a resolved campaign and its selected profile.
The profile command is inspected as an argv template. Dynamic executor tokens
are resolved from the campaign where the lane already defines their meaning.
Unknown tokens and unresolved placeholders are reported as problems.

The checker treats required inputs as failures when they are unavailable. The
RFdiffusion3 specification is optional until target preparation writes it.
When that file exists, the checker validates its JSON, contig, and hotspots.

RFdiffusion is checked on every campaign, because the package selects it by
default. Genie3 is checked only on a campaign that selects it, and it is held to
the stage wiring, identity, resource, module, argv, and toolcheck checks the
RFdiffusion record gets. The contig and hotspot checks stay with RFdiffusion,
which is the tool that takes them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import backbone_shape
from .adapters import genie3_generator as genie3
from .adapters import rfdiffusion_generator as rfd
from .adapters import target_prep_adapter as target_prep
from .refusals import ExitCode, Refusal, exit_code_for_result


PLACEHOLDER_RE = re.compile(r"__REQUIRED__|<required", re.IGNORECASE)
TOKEN_RE = re.compile(r"^\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}$")
RESIDUE_RE = re.compile(r"^([A-Za-z0-9]+):(-?\d+)([A-Za-z]?)$")
RESIDUE_RANGE_RE = re.compile(r"^([A-Za-z0-9]+):(\d+)-(\d+)$")
TOOL_RESIDUE_RE = re.compile(r"^([A-Za-z]):(-?\d+)$|^([A-Za-z])(-?\d+)$")
# See runtime_validator.MODAL_ENV_RE: spec_sha is a 16 hex content hash of the
# shipped environment source file, not a 64 hex digest.
ENVIRONMENT_IDENTITY_RE = re.compile(
    r"^modal-env:[^@\s]+@spec_sha=[0-9a-f]{16,64}$"
)
RF3_ENVIRONMENT_IDENTITY_RE = re.compile(
    r"^fal-rf3-[^:\s]+:GPU-[^:\s]+:[1-9][0-9]*:python-[^:\s]+:[^:\s]+:[^:\s]+:torch-[^:\s]+$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RF3_CONTIG_RE = re.compile(r"^(\d+)-(\d+),/0,([A-Za-z0-9])(\d+)-(\d+)$")
RF3_SPEC_RESIDUE_RE = re.compile(r"^([A-Za-z0-9])(\d+)$")

LEGACY_ADAPTER_MODULE = "claude_binder.adapters.rfdiffusion_generator"
RF3_ADAPTER_MODULE = "claude_binder.adapters.rfdiffusion3_generator"

RFDIFFUSION_ADAPTER_ID = "rfdiffusion-generator"
GENIE3_ADAPTER_ID = "genie3-generator"
# Both shipped Genie3 routes publish the same two artifacts with the same
# code, so both satisfy the generator slot. `genie3_generator` runs the tool
# and `fal_genie3_generator` posts the same inputs to a deployed application.
GENIE3_ADAPTER_MODULES = (
    "claude_binder.adapters.genie3_generator",
    "claude_binder.adapters.fal_genie3_generator",
)
# The weights are a Hugging Face repository revision, not a checkpoint file,
# so Genie3 pins a 40-character commit where RFdiffusion pins a sha256. The
# same shape is pinned for ESMFold2 in
# `adapters/esmfold2_predictor.ESMFOLD2_REVISION_PIN_RE`.
GENIE3_REVISION_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+@[0-9a-f]{40}$")

REQUIRED_COMMAND_FLAGS = (
    "--phase",
    "--count",
    "--attempt-dir",
    "--receipts-dir",
    "--artifact-root",
    "--contigs",
    "--binder-chain",
    "--target-chain",
)

# The five flags `genie3_generator` and `fal_genie3_generator` both declare
# required on their run subcommand, plus --config. Genie3 takes no contig and no
# hotspot flag, because the wrapper reads the site residues out of the target
# manifest itself, so neither appears here. --config is required for provenance:
# `genie3_generator.config_model_revision` reads the pinned model_revision out of
# it and records that revision on every manifest row.
GENIE3_COMMAND_FLAGS = (
    "--phase",
    "--count",
    "--attempt-dir",
    "--receipts-dir",
    "--artifact-root",
    "--config",
)

# Flags carrying a value the wrapper validates. They are optional in the argv,
# and checked when the profile supplies one.
GENIE3_VALUE_FLAGS = (
    "--binder-chain",
    "--target-chain",
    "--generator-binder-chain",
    "--binder-length-min",
    "--binder-length-max",
    "--seed",
    "--seed-config-key",
    "--direction-scale",
    "--problem-id",
    "--generator-id",
)

RF3_COMMAND_FLAGS = (
    "--stage",
    "--phase",
    "--count",
    "--attempt-dir",
    "--receipts-dir",
    "--artifact-root",
    "--config",
    "--plan",
    "--specification",
    "--out-dir",
    "--receipt",
    "--diffusion-batch-size",
    "--n-batches",
    "--step-scale",
    "--gamma-0",
)


@dataclass(frozen=True)
class Problem:
    """One actionable preflight failure."""

    field: str
    problem: str
    fix: str

    def as_dict(self) -> dict[str, str]:
        return {"field": self.field, "problem": self.problem, "fix": self.fix}

    def text(self) -> str:
        return f"{self.field}: {self.problem}. Fix: {self.fix}."


@dataclass
class PreflightReport:
    """The complete result of one local preflight."""

    problems: list[Problem] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def refusal(self) -> Refusal | None:
        """Return the first blocking preflight problem as a standard refusal."""
        if self.ok:
            return None
        problem = self.problems[0]
        return Refusal(
            # The preflight also checks the backbone atom route, which a campaign
            # with no RFdiffusion stage reaches, so the cause names the campaign.
            cause="The generator preflight rejected the campaign.",
            expected=f"{problem.field} must satisfy the generator preflight contract.",
            expected_source="The resolved campaign and selected profile.",
            found=problem.problem,
            found_source=problem.field,
            scope="The generator preflight inspected local inputs and started no provider command.",
            action=problem.fix,
            escalation="Send the resolved campaign and selected profile to the generator maintainer.",
        )

    def as_dict(self) -> dict[str, Any]:
        refusal = self.refusal()
        return {
            "ok": self.ok,
            "checked": list(self.checked),
            "problems": [problem.as_dict() for problem in self.problems],
            "exit_code": exit_code_for_result(
                verified=self.ok,
                refused=not self.ok,
            ),
            "refusal": refusal.as_dict() if refusal is not None else None,
            "refusal_text": refusal.text() if refusal is not None else "",
        }


@dataclass
class _TargetState:
    target: dict[str, Any]
    site: dict[str, Any]
    source_path: Path | None = None
    normalized_path: Path | None = None
    residue_map_path: Path | None = None
    residue_map: dict[str, Any] = field(default_factory=dict)
    source_residues: dict[str, list[int]] = field(default_factory=dict)
    normalized_residues: dict[str, list[int]] = field(default_factory=dict)
    mapped_design: list[str] = field(default_factory=list)
    mapped_reference: list[str] = field(default_factory=list)


def _placeholder(value: Any) -> bool:
    return isinstance(value, str) and PLACEHOLDER_RE.search(value) is not None


def _problem(report: PreflightReport, field_name: str, problem: str, fix: str) -> None:
    candidate = Problem(field_name, problem, fix)
    if candidate not in report.problems:
        report.problems.append(candidate)


def _path(value: Any, base_dir: Path) -> Path | None:
    if isinstance(value, Path):
        return value.expanduser() if value.is_absolute() else (base_dir / value).resolve()
    if not isinstance(value, str) or not value or _placeholder(value):
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, report: PreflightReport, field_name: str) -> dict[str, Any] | None:
    if not path.is_file():
        _problem(report, field_name, f"file does not exist: {path}", "create the file or point the field at an existing JSON file")
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        _problem(report, field_name, f"file is not valid JSON: {exc}", "write a JSON object")
        return None
    if not isinstance(value, dict):
        _problem(report, field_name, "JSON value is not an object", "write a JSON object")
        return None
    return value


def _structure_residues(
    path: Path | None,
    report: PreflightReport,
    field_name: str,
) -> dict[str, list[int]]:
    if path is None:
        _problem(
            report,
            field_name,
            "no structure path is resolved",
            "supply an existing PDB or mmCIF path",
        )
        return {}
    if path.suffix.lower() in {".cif", ".mmcif"}:
        try:
            # The target preparer prefers auth_asym_id and auth_seq_id. Those
            # columns carry the author numbering used by the campaign residue map.
            atoms, _ = target_prep.load_atoms(path)
            standard_atoms = [atom for atom in atoms if atom.record == "ATOM"]
            residues = target_prep.chain_residues(standard_atoms)
        except (OSError, target_prep.AdapterError) as exc:
            _problem(
                report,
                field_name,
                str(exc),
                "supply a readable PDB or mmCIF with ATOM records",
            )
        else:
            result = {
                chain: sorted(residue.number for residue in chain_residues)
                for chain, chain_residues in residues.items()
            }
            if not result:
                _problem(
                    report,
                    field_name,
                    f"{path} has no standard ATOM residues the adapter can use",
                    "supply a PDB or mmCIF whose target chain has ATOM records",
                )
            return result
        return {}
    try:
        atoms = rfd.read_atom_records(path)
        residues = rfd.chain_residues(atoms, standard_only=True)
    except (OSError, rfd.AdapterError) as exc:
        _problem(report, field_name, str(exc), "supply a readable PDB with ATOM records")
        return {}
    if not residues:
        _problem(
            report,
            field_name,
            f"{path} has no standard ATOM residues the adapter can use",
            "supply a PDB whose target chain has ATOM records",
        )
    return residues


def _label_parts(value: Any) -> tuple[str, int] | None:
    if not isinstance(value, str):
        return None
    match = RESIDUE_RE.fullmatch(value)
    if match is None or match.group(3):
        return None
    return match.group(1), int(match.group(2))


def _expand_source_label(value: Any) -> list[str] | None:
    if not isinstance(value, str):
        return None
    span = RESIDUE_RANGE_RE.fullmatch(value)
    if span is not None:
        chain, low, high = span.group(1), int(span.group(2)), int(span.group(3))
        if high < low:
            return None
        return [f"{chain}:{number}" for number in range(low, high + 1)]
    if _label_parts(value) is None:
        return None
    return [value]


def _tool_labels(value: Any) -> list[str] | None:
    """Expand a mapped label using the adapter's hotspot label grammar."""
    if not isinstance(value, str):
        return None
    range_match = re.fullmatch(r"([A-Za-z]):(-?\d+)-(-?\d+)", value)
    if range_match is not None:
        low, high = int(range_match.group(2)), int(range_match.group(3))
        if high < low:
            return None
        return [f"{range_match.group(1)}:{number}" for number in range(low, high + 1)]
    if TOOL_RESIDUE_RE.fullmatch(value) is None:
        return None
    match = re.fullmatch(r"([A-Za-z]):(-?\d+)", value) or re.fullmatch(
        r"([A-Za-z])(-?\d+)", value
    )
    assert match is not None
    return [f"{match.group(1)}:{int(match.group(2))}"]


def _map_site_residues(
    state: _TargetState,
    values: Any,
    field_name: str,
    report: PreflightReport,
) -> list[str]:
    if not isinstance(values, list) or not values:
        _problem(report, field_name, "must be a non-empty list", "supply residues in CHAIN:NUMBER or CHAIN:LOW-HIGH form")
        return []
    mapping = state.residue_map.get("source_to_cleaned")
    if not isinstance(mapping, dict):
        if state.residue_map:
            _problem(report, "targets[0].site.residue_map_path", "has no source_to_cleaned object", "write the residue map with a source_to_cleaned object")
        return []

    mapped: list[str] = []
    source_seen: set[str] = set()
    for index, value in enumerate(values):
        item_field = f"{field_name}[{index}]"
        source_labels = _expand_source_label(value)
        if source_labels is None:
            _problem(
                report,
                item_field,
                f"{value!r} is not a supported source residue label",
                "use CHAIN:NUMBER or CHAIN:LOW-HIGH without insertion codes",
            )
            continue
        for source_label in source_labels:
            if source_label in source_seen:
                _problem(report, item_field, f"names {source_label} more than once", "remove the duplicate residue")
                continue
            source_seen.add(source_label)
            chain, number = _label_parts(source_label) or ("", 0)
            if number not in set(state.source_residues.get(chain, [])):
                _problem(
                    report,
                    item_field,
                    f"source structure does not carry {source_label}",
                    "choose a residue present in the named source structure or correct the source structure",
                )
            mapped_value = mapping.get(source_label)
            if mapped_value is None:
                _problem(
                    report,
                    "targets[0].site.residue_map_path",
                    f"does not map {source_label}",
                    "add a source_to_cleaned entry for every requested source residue",
                )
                continue
            cleaned_labels = _tool_labels(str(mapped_value))
            if cleaned_labels is None:
                _problem(
                    report,
                    "targets[0].site.residue_map_path",
                    f"maps {source_label} to unsupported RFdiffusion label {mapped_value!r}",
                    "map it to CHAIN:NUMBER or CHAIN:LOW-HIGH using one-letter chain IDs",
                )
                continue
            for cleaned_label in cleaned_labels:
                cleaned_chain, cleaned_number = cleaned_label.split(":", 1)
                if int(cleaned_number) not in set(
                    state.normalized_residues.get(cleaned_chain, [])
                ):
                    _problem(
                        report,
                        item_field,
                        f"RFdiffusion structure does not carry mapped residue {cleaned_label}",
                        "correct the residue map or use the structure with the numbering the map names",
                    )
                normalized = f"{cleaned_chain}:{int(cleaned_number)}"
                if normalized not in mapped:
                    mapped.append(normalized)
    return mapped


def _resolve_token(
    value: Any,
    context: Mapping[str, str],
    report: PreflightReport,
    field_name: str,
) -> str | None:
    if not isinstance(value, str):
        _problem(report, field_name, "must be a string", "supply a string argv value")
        return None
    if _placeholder(value):
        _problem(report, field_name, f"is unresolved: {value}", "fill the required profile or campaign value")
        return None
    matches = list(TOKEN_RE.finditer(value))
    if not matches:
        return value
    rendered = value
    unresolved = False
    for match in matches:
        resolved = context.get(match.group(1))
        if resolved is None or _placeholder(resolved):
            _problem(report, field_name, f"cannot resolve template token {match.group(0)}", "resolve the campaign before paying for the stage")
            unresolved = True
            continue
        rendered = rendered.replace(match.group(0), resolved, 1)
    return None if unresolved else rendered


def _option(
    command: Sequence[str],
    flag: str,
    report: PreflightReport,
    field_prefix: str,
    *,
    required: bool = True,
) -> str | None:
    positions = [index for index, value in enumerate(command) if value == flag]
    field_name = f"{field_prefix}.command_argv_template[{flag}]"
    if len(positions) > 1:
        _problem(report, field_name, "appears more than once", "keep one adapter-owned flag")
        return None
    if not positions:
        if required:
            _problem(report, field_name, "is missing", f"add {flag} with the value required by the run parser")
        return None
    index = positions[0]
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        _problem(report, field_name, "has no value", f"add one value after {flag}")
        return None
    return command[index + 1]


def _campaign_contigs(campaign: Mapping[str, Any]) -> Any:
    for key in ("contigs", "generator_contigs"):
        if key in campaign:
            return campaign[key]
    generation = campaign.get("generation")
    if isinstance(generation, dict):
        for key in ("contigs", "generator_contigs"):
            if key in generation:
                return generation[key]
        for generator in generation.get("generators", []):
            if isinstance(generator, dict) and generator.get("adapter_id") == "rfdiffusion-generator":
                if "contigs" in generator:
                    return generator["contigs"]
    return None


def _target_state(
    campaign: Mapping[str, Any],
    base_dir: Path,
    manifest: Mapping[str, Any] | None,
    manifest_path: Path | None,
    report: PreflightReport,
) -> _TargetState | None:
    targets = campaign.get("targets")
    if not isinstance(targets, list) or not targets:
        _problem(report, "targets", "must be a non-empty list", "declare one primary target")
        return None
    primary = [item for item in targets if isinstance(item, dict) and item.get("role") == "primary"]
    if len(primary) != 1:
        _problem(report, "targets", f"contains {len(primary)} primary targets", "declare exactly one target with role primary")
        return None
    target = primary[0]
    if not isinstance(target.get("target_id"), str) or not target.get("target_id") or _placeholder(target.get("target_id")):
        _problem(report, "targets[0].target_id", "is missing or unresolved", "declare the primary target ID")
    chains = target.get("chains")
    if not isinstance(chains, list) or not chains:
        _problem(report, "targets[0].chains", "is missing or empty", "declare the target chain used by RFdiffusion")
    site = target.get("site")
    if not isinstance(site, dict):
        _problem(report, "targets[0].site", "is missing or is not an object", "declare the target site")
        return None
    state = _TargetState(target=target, site=site)

    state.source_path = _path(
        target.get("structure_source_path", target.get("structure_path")), base_dir
    )
    state.normalized_path = _path(
        target.get("runtime_structure_path", target.get("normalized_structure_path")), base_dir
    )
    state.residue_map_path = _path(
        site.get("runtime_residue_map_path", site.get("residue_map_path")), base_dir
    )
    if not isinstance(target.get("structure_path"), str) or not target.get("structure_path") or _placeholder(target.get("structure_path")):
        _problem(report, "targets[0].structure_path", "is missing or unresolved", "name the source structure used to build the target manifest")
    if not isinstance(site.get("residue_map_path"), str) or not site.get("residue_map_path") or _placeholder(site.get("residue_map_path")):
        _problem(report, "targets[0].site.residue_map_path", "is missing or unresolved", "name the source-to-cleaned residue map")

    if manifest is not None:
        manifest_base = manifest_path.parent if manifest_path is not None else base_dir
        manifest_source = _path(manifest.get("source_structure_path"), manifest_base)
        manifest_normalized = _path(manifest.get("normalized_structure_path"), manifest_base)
        if manifest_source is not None:
            state.source_path = manifest_source
        if manifest_normalized is not None:
            state.normalized_path = manifest_normalized
        manifest_map = _path(manifest.get("residue_map_path"), manifest_base)
        if manifest_map is not None:
            state.residue_map_path = manifest_map
        if manifest.get("target_id") != target.get("target_id"):
            _problem(report, "target_manifest.target_id", f"is {manifest.get('target_id')!r}, expected {target.get('target_id')!r}", "write the manifest for the campaign's primary target")
        if manifest_source is None:
            _problem(report, "target_manifest.source_structure_path", "does not resolve", "write an existing source_structure_path")

    state.source_residues = _structure_residues(state.source_path, report, "target source structure")
    state.normalized_residues = _structure_residues(
        state.normalized_path, report, "target normalized structure"
    )

    if state.residue_map_path is None:
        _problem(report, "targets[0].site.residue_map_path", "does not resolve to a file", "supply the residue map used to translate source labels")
    else:
        state.residue_map = _load_json(
            state.residue_map_path, report, "targets[0].site.residue_map_path"
        ) or {}
        if state.residue_map and not isinstance(state.residue_map.get("source_to_cleaned"), dict):
            _problem(report, "targets[0].site.residue_map_path", "has no source_to_cleaned object", "write the source-to-cleaned residue mapping")

    state.mapped_design = _map_site_residues(
        state, site.get("design_residues"), "targets[0].site.design_residues", report
    )
    state.mapped_reference = _map_site_residues(
        state,
        site.get("reference_contact_residues"),
        "targets[0].site.reference_contact_residues",
        report,
    )
    return state


def _check_manifest(
    campaign: Mapping[str, Any],
    base_dir: Path,
    target: Mapping[str, Any] | None,
    target_manifest_path: Path | None,
    report: PreflightReport,
) -> tuple[dict[str, Any] | None, Path | None]:
    candidate: Any = target_manifest_path
    if candidate is None:
        candidate = campaign.get("target_manifest_path")
    if candidate is None and isinstance(target, Mapping):
        candidate = target.get("target_manifest_path")
    if isinstance(candidate, Mapping):
        manifest = dict(candidate)
        manifest_path = None
    else:
        manifest_path = _path(candidate, base_dir)
        if manifest_path is None:
            _problem(report, "target_manifest", "was not supplied, so the target-manifest contract could not be checked", "supply target_manifest_path or a target_manifest object")
            return None, None
        manifest = _load_json(manifest_path, report, "target_manifest")
        if manifest is None:
            return None, manifest_path

    for field_name in rfd.REQUIRED_TARGET_MANIFEST_FIELDS:
        value = manifest.get(field_name)
        if not isinstance(value, str) or not value:
            _problem(report, f"target_manifest.{field_name}", "is missing or empty", f"write {field_name} in the target-preparer manifest")

    source_path = _path(manifest.get("source_structure_path"), manifest_path.parent if manifest_path else base_dir)
    if source_path is not None and source_path.is_file():
        expected = manifest.get("target_sha256")
        if isinstance(expected, str) and expected and _sha256(source_path) != expected:
            _problem(report, "target_manifest.target_sha256", f"does not match {source_path}", "regenerate the manifest or restore the source structure bytes")
    elif source_path is not None:
        _problem(report, "target_manifest.source_structure_path", f"file does not exist: {source_path}", "write the manifest with an existing source structure")
    return manifest, manifest_path


def _check_campaign_fields(campaign: Mapping[str, Any], report: PreflightReport) -> None:
    for field_name in ("campaign_id", "run_id"):
        value = campaign.get(field_name)
        if not isinstance(value, str) or not value or _placeholder(value):
            _problem(report, field_name, "is missing or unresolved", "resolve the campaign identifier")
    binder = campaign.get("binder")
    if not isinstance(binder, dict):
        _problem(report, "binder", "is missing or is not an object", "declare binder lengths and chain IDs")
        return
    for field_name in ("minimum_length", "maximum_length", "target_chain_id", "binder_chain_id"):
        value = binder.get(field_name)
        if value is None or _placeholder(value) or value == "":
            _problem(report, f"binder.{field_name}", "is missing or unresolved", "fill the required binder field")
    minimum, maximum = binder.get("minimum_length"), binder.get("maximum_length")
    if isinstance(minimum, int) and not isinstance(minimum, bool) and minimum < 1:
        _problem(report, "binder.minimum_length", "must be positive", "set a positive minimum length")
    if isinstance(maximum, int) and not isinstance(maximum, bool) and maximum < 1:
        _problem(report, "binder.maximum_length", "must be positive", "set a positive maximum length")
    if isinstance(minimum, int) and isinstance(maximum, int) and minimum > maximum:
        _problem(report, "binder", "minimum_length exceeds maximum_length", "increase maximum_length or lower minimum_length")


def _adapter_module(command: Sequence[str]) -> str | None:
    for index, value in enumerate(command[:-1]):
        if value == "-m":
            return command[index + 1]
    return None


def _profile_context(
    campaign: Mapping[str, Any],
    state: _TargetState | None,
    artifact_root: Path,
) -> dict[str, str]:
    context: dict[str, str] = {
        "count": "1",
        "phase": "single",
        "config_path": "resolved-config.json",
        "attempt_dir": "attempt",
        "receipts_dir": "receipts",
        "artifact_root": str(artifact_root),
        "run_root": "run",
        "plan_path": "plan.json",
        "stage_id": "generate-rfdiffusion",
        "python_executable": sys.executable,
    }
    if state is None:
        return context
    binder = campaign.get("binder", {})
    context.update(
        {
            "target_id": str(state.target.get("target_id", "")),
            "target_chain": str(binder.get("target_chain_id", "")),
            "binder_chain": str(binder.get("binder_chain_id", "")),
            "binder_length_min": str(binder.get("minimum_length", "")),
            "binder_length_max": str(binder.get("maximum_length", "")),
            "target_residues_csv": ",".join(state.mapped_design),
            "reference_contact_residues_csv": ",".join(state.mapped_reference),
        }
    )
    if state.normalized_path is not None:
        context["target_structure"] = str(state.normalized_path)
    return context


def _check_adapter_record(
    adapter: Mapping[str, Any],
    prefix: str,
    display_name: str,
    report: PreflightReport,
) -> None:
    """Check the identity and resource fields every generator record declares.

    These hold for a backbone generator whatever the tool is. The revision
    format, the argv contract, and the values the wrapper validates differ per
    tool, so they stay with the caller.
    """
    for key in ("source_revision", "model_revision", "environment_identity", "command_argv_template", "parser_argv_template", "resources"):
        value = adapter.get(key)
        if value is None or value == "" or _placeholder(value):
            _problem(report, f"{prefix}.{key}", "is missing or unresolved", f"resolve the {display_name} profile field before allocation")
    resources = adapter.get("resources")
    if isinstance(resources, dict):
        if resources.get("gpu") != 1:
            _problem(report, f"{prefix}.resources.gpu", f"is {resources.get('gpu')!r}, but {display_name} needs one GPU", f"request one GPU for the {display_name} adapter")
        image = resources.get("container_image_digest")
        if not isinstance(image, str) or not image or _placeholder(image):
            _problem(report, f"{prefix}.resources.container_image_digest", "is missing or unresolved", "record the image digest produced for the selected environment")


def _check_profile(
    profile: Mapping[str, Any] | None,
    campaign: Mapping[str, Any],
    state: _TargetState | None,
    report: PreflightReport,
    *,
    artifact_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, str | None]]:
    if profile is None:
        _problem(report, "profile", "was not supplied, so the RFdiffusion profile could not be checked", "supply the selected resolved profile")
        return None, {}
    adapters = profile.get("adapters")
    if not isinstance(adapters, list):
        _problem(report, "profile.adapters", "is missing or is not a list", "supply the resolved adapter list")
        return None, {}
    matches = [item for item in adapters if isinstance(item, dict) and item.get("adapter_id") == "rfdiffusion-generator"]
    if len(matches) != 1:
        _problem(report, "profile.adapters[rfdiffusion-generator]", f"found {len(matches)} matching adapter records", "include exactly one RFdiffusion adapter")
        return None, {}

    adapter = matches[0]
    prefix = "profile.adapters[rfdiffusion-generator]"
    _check_adapter_record(adapter, prefix, "RFdiffusion", report)
    model_revision = adapter.get("model_revision")
    if isinstance(model_revision, str) and not _placeholder(model_revision):
        digest_match = re.search(r"sha256:(\S+)", model_revision)
        if digest_match is None or SHA256_RE.fullmatch(digest_match.group(1).lower()) is None:
            _problem(report, f"{prefix}.model_revision", "does not pin a valid 64-character checkpoint sha256", "record the checkpoint digest in model_revision")

    command = adapter.get("command_argv_template")
    if not isinstance(command, list) or any(not isinstance(value, str) for value in command):
        _problem(report, f"{prefix}.command_argv_template", "must be a string argv list", "write the RFdiffusion run command as an argv list")
        return adapter, {}
    module = _adapter_module(command)
    if module not in {LEGACY_ADAPTER_MODULE, RF3_ADAPTER_MODULE}:
        _problem(report, f"{prefix}.command_argv_template", "does not invoke a supported shipped RFdiffusion adapter", "use claude_binder.adapters.rfdiffusion_generator or claude_binder.adapters.rfdiffusion3_generator")
    if "run" not in command:
        _problem(report, f"{prefix}.command_argv_template", "does not select the adapter run subcommand", "include the run subcommand")

    identity = adapter.get("environment_identity")
    if isinstance(identity, str) and not _placeholder(identity):
        if module == LEGACY_ADAPTER_MODULE and ENVIRONMENT_IDENTITY_RE.fullmatch(identity) is None:
            _problem(report, f"{prefix}.environment_identity", "does not match modal-env:name@spec_sha=16-to-64-hex format", "record the Modal environment identity and spec hash")
        elif module == RF3_ADAPTER_MODULE and RF3_ENVIRONMENT_IDENTITY_RE.fullmatch(identity) is None:
            _problem(report, f"{prefix}.environment_identity", "does not name a resolved fal RFdiffusion3 deployment shape", "record the fal-rf3 deployment, GPU shape, replica count, Python version, revision, and torch version")

    if module == LEGACY_ADAPTER_MODULE:
        required_flags = REQUIRED_COMMAND_FLAGS
        if "--hotspot" not in command and "--hotspot-csv" not in command:
            _problem(report, f"{prefix}.command_argv_template[hotspot]", "has neither --hotspot nor --hotspot-csv", "bind one hotspot flag to the mapped site residues")
    elif module == RF3_ADAPTER_MODULE:
        required_flags = RF3_COMMAND_FLAGS
    else:
        required_flags = ()
    for flag in required_flags:
        _option(command, flag, report, prefix)

    toolcheck = adapter.get("toolcheck_argv")
    if not isinstance(toolcheck, list) or "toolcheck" not in toolcheck:
        _problem(report, f"{prefix}.toolcheck_argv", "does not select the adapter toolcheck subcommand", "include the adapter toolcheck command")
    elif module is not None and module not in toolcheck:
        _problem(report, f"{prefix}.toolcheck_argv", "does not invoke the selected generator adapter", f"use {module} toolcheck")

    context = _profile_context(campaign, state, artifact_root)
    values: dict[str, str | None] = {}
    value_flags = list(required_flags)
    if module == LEGACY_ADAPTER_MODULE:
        value_flags.extend(("--runner-protocol", "--rfdiffusion-root", "--weights-dir", "--checkpoint-name", "--config", "--minimum-length", "--maximum-length"))
    for flag in dict.fromkeys(value_flags):
        raw = _option(command, flag, report, prefix, required=False)
        values[flag] = _resolve_token(raw, context, report, f"{prefix}.command_argv_template[{flag}]") if raw is not None else None

    if module == LEGACY_ADAPTER_MODULE:
        _check_legacy_profile_values(adapter, campaign, values, model_revision, prefix, report)
    elif module == RF3_ADAPTER_MODULE:
        _check_rf3_profile_values(values, prefix, report)
    return adapter, values


def _check_legacy_profile_values(
    adapter: Mapping[str, Any],
    campaign: Mapping[str, Any],
    values: Mapping[str, str | None],
    model_revision: Any,
    prefix: str,
    report: PreflightReport,
) -> None:
    protocol = values.get("--runner-protocol") or rfd.DEFAULT_RUNNER_PROTOCOL
    if protocol not in rfd.RUNNER_PROTOCOLS:
        _problem(report, f"{prefix}.command_argv_template[--runner-protocol]", f"has unsupported value {protocol!r}", "choose auto, local, or modal")
    root_value = values.get("--rfdiffusion-root")
    weights_value = values.get("--weights-dir")
    checkpoint_name = values.get("--checkpoint-name")
    if checkpoint_name is not None:
        checkpoint_path = Path(checkpoint_name)
        if checkpoint_path.name != checkpoint_name or checkpoint_path.is_absolute() or checkpoint_name in {"", ".", ".."}:
            _problem(report, f"{prefix}.command_argv_template[--checkpoint-name]", f"is not a file name: {checkpoint_name!r}", "supply the checkpoint file name inside the weights directory")

    if protocol == "local" and root_value is not None and weights_value is not None and checkpoint_name is not None:
        root = Path(root_value).expanduser()
        weights = Path(weights_value).expanduser()
        if not root.is_dir():
            _problem(report, f"{prefix}.command_argv_template[--rfdiffusion-root]", f"directory does not exist: {root}", "point at a local RFdiffusion checkout")
        runner = root / "scripts" / "run_inference.py"
        if root.is_dir() and not runner.is_file():
            _problem(report, f"{prefix}.command_argv_template[--rfdiffusion-root]", f"runner does not exist: {runner}", "use a checkout containing scripts/run_inference.py")
        if not weights.is_dir():
            _problem(report, f"{prefix}.command_argv_template[--weights-dir]", f"directory does not exist: {weights}", "point at the local checkpoint directory")
        checkpoint = weights / checkpoint_name
        if weights.is_dir() and not checkpoint.is_file():
            _problem(report, f"{prefix}.command_argv_template[--checkpoint-name]", f"checkpoint does not exist: {checkpoint}", "place the named checkpoint in the weights directory")
        if checkpoint.is_file() and isinstance(model_revision, str) and not _placeholder(model_revision):
            digest_match = re.search(r"sha256:(\S+)", model_revision)
            if digest_match is not None and _sha256(checkpoint) != digest_match.group(1).lower():
                _problem(report, f"{prefix}.model_revision", f"checkpoint {checkpoint} has a different sha256", "point at the pinned checkpoint or update the recorded digest")
    elif protocol == "modal":
        if root_value is None or weights_value is None:
            _problem(report, f"{prefix}.command_argv_template", "Modal root or weights path could not be resolved", "declare the two Modal mount paths")
        else:
            root = Path(root_value).expanduser()
            weights = Path(weights_value).expanduser()
            if not root.is_dir() or not weights.is_dir():
                _problem(report, f"{prefix}.command_argv_template", f"Modal runner and weights mounts are not available locally at {root} and {weights}", "run this check where the mounts are available or complete the runtime record and repeat the check")

    for field_name, value in (("--binder-chain", values.get("--binder-chain")), ("--target-chain", values.get("--target-chain"))):
        if value is not None and rfd.CHAIN_ID_RE.fullmatch(value) is None:
            _problem(report, f"{prefix}.command_argv_template[{field_name}]", f"{value!r} is not one letter or digit", "use a one-character RFdiffusion chain ID")
    binder_value, target_value = values.get("--binder-chain"), values.get("--target-chain")
    if binder_value is not None and target_value is not None and binder_value == target_value:
        _problem(report, f"{prefix}.command_argv_template", "binder and target chains are identical", "use distinct designed and fixed chain IDs")
    binder = campaign.get("binder")
    if isinstance(binder, dict):
        if binder_value is not None and isinstance(binder.get("binder_chain_id"), str) and binder_value != binder["binder_chain_id"]:
            _problem(report, f"{prefix}.command_argv_template[--binder-chain]", f"names {binder_value}, campaign names {binder['binder_chain_id']}", "make the command and campaign use the same binder chain")
        if target_value is not None and isinstance(binder.get("target_chain_id"), str) and target_value != binder["target_chain_id"]:
            _problem(report, f"{prefix}.command_argv_template[--target-chain]", f"names {target_value}, campaign names {binder['target_chain_id']}", "make the command and campaign use the same target chain")

    for flag in ("--minimum-length", "--maximum-length"):
        value = values.get(flag)
        if value is not None:
            try:
                if int(value) < 1:
                    raise ValueError
            except ValueError:
                _problem(report, f"{prefix}.command_argv_template[{flag}]", f"is not a positive integer: {value!r}", "supply a positive binder length")


def _check_rf3_profile_values(
    values: Mapping[str, str | None],
    prefix: str,
    report: PreflightReport,
) -> None:
    for flag in ("--count", "--diffusion-batch-size", "--n-batches"):
        value = values.get(flag)
        if value is None:
            continue
        try:
            if int(value) < 1:
                raise ValueError
        except ValueError:
            _problem(report, f"{prefix}.command_argv_template[{flag}]", f"is not a positive integer: {value!r}", "supply a positive RFdiffusion3 batch value")
    for flag in ("--step-scale", "--gamma-0"):
        value = values.get(flag)
        if value is None:
            continue
        try:
            float(value)
        except ValueError:
            _problem(report, f"{prefix}.command_argv_template[{flag}]", f"is not a number: {value!r}", "supply a numeric RFdiffusion3 sampler value")


def _check_generation_request(
    profile: Mapping[str, Any] | None,
    campaign: Mapping[str, Any],
    state: _TargetState | None,
    adapter: Mapping[str, Any] | None,
    values: Mapping[str, str | None],
    report: PreflightReport,
    *,
    base_dir: Path,
) -> None:
    if adapter is None or state is None:
        return
    prefix = "profile.adapters[rfdiffusion-generator]"
    command = adapter.get("command_argv_template")
    if not isinstance(command, list):
        return
    module = _adapter_module(command)
    if module == RF3_ADAPTER_MODULE:
        _check_rf3_generation_request(campaign, state, values, report, base_dir=base_dir)
        return
    if module != LEGACY_ADAPTER_MODULE:
        return
    raw_contigs = _option(command, "--contigs", report, prefix, required=False)
    if raw_contigs is None:
        return
    contigs = raw_contigs
    token_match = TOKEN_RE.fullmatch(contigs)
    if token_match:
        if token_match.group(1) in {"contigs", "generator_contigs"}:
            contigs = _campaign_contigs(campaign)
        else:
            contigs = None
    if contigs is None or not isinstance(contigs, str) or _placeholder(contigs):
        _problem(report, "generator.contigs", "is unresolved, so the contig grammar and structure spans could not be checked", "supply the contig specification without Hydra brackets")
        return
    try:
        chain_spans, _ = rfd.parse_contigs(contigs)
    except rfd.AdapterError as exc:
        _problem(report, "generator.contigs", str(exc), "use the adapter grammar: whitespace tokens, slash segments, chain spans, positive length spans, and terminal 0")
        return
    if state.normalized_path is None:
        _problem(report, "generator.contigs", "cannot check chain spans without the normalized structure", "supply the normalized target structure")
    else:
        try:
            rfd.check_contig_spans(chain_spans, state.normalized_residues, state.normalized_path)
        except rfd.AdapterError as exc:
            _problem(report, "generator.contigs", str(exc), "change the contig or use a structure carrying every named chain residue")

    hotspot_values: list[str] = []
    for flag in ("--hotspot", "--hotspot-csv"):
        positions = [index for index, value in enumerate(command) if value == flag]
        for index in positions:
            raw = command[index + 1] if index + 1 < len(command) else None
            if raw is None:
                continue
            match = TOKEN_RE.fullmatch(raw)
            if match:
                if match.group(1) == "reference_contact_residues_csv":
                    raw = ",".join(state.mapped_reference)
                elif match.group(1) == "target_residues_csv":
                    raw = ",".join(state.mapped_design)
                else:
                    raw = None
            if raw is not None:
                hotspot_values.append(raw)
    if not hotspot_values:
        _problem(report, "generator.hotspots", "no hotspot value could be resolved", "bind --hotspot-csv to the mapped reference contact residues")
        return
    try:
        rfd.parse_hotspots(
            hotspot_values,
            chain_spans,
            state.normalized_residues,
            state.normalized_path or Path("<missing normalized structure>"),
        )
    except rfd.AdapterError as exc:
        _problem(report, "generator.hotspots", str(exc), "use mapped residues present in the normalized structure and inside a contig chain span")


def _resolved_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _rf3_spec_residue(value: Any) -> tuple[str, int] | None:
    if not isinstance(value, str):
        return None
    match = RF3_SPEC_RESIDUE_RE.fullmatch(value)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _check_rf3_specification(
    path: Path,
    campaign: Mapping[str, Any],
    state: _TargetState,
    report: PreflightReport,
    *,
    structure_residues: Mapping[str, list[int]] | None = None,
) -> None:
    if path.exists() and not path.is_file():
        _problem(report, "generator.specification", f"path is not a file: {path}", "write the RFdiffusion3 specification as a JSON file")
        return
    if not path.is_file():
        return
    try:
        specification = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        _problem(report, "generator.specification", f"file is not valid JSON: {exc}", "write a non-empty RFdiffusion3 JSON object")
        return
    if not isinstance(specification, dict) or not specification:
        _problem(report, "generator.specification", "must be a non-empty JSON object", "write one RFdiffusion3 specification object")
        return

    binder = campaign.get("binder")
    minimum = binder.get("minimum_length") if isinstance(binder, dict) else None
    maximum = binder.get("maximum_length") if isinstance(binder, dict) else None
    expected_target_chain = binder.get("target_chain_id") if isinstance(binder, dict) else None
    available = state.normalized_residues if structure_residues is None else structure_residues
    for name, entry in specification.items():
        field_name = f"generator.specification[{name!r}]"
        if not isinstance(entry, dict):
            _problem(report, field_name, "is not an object", "write a RFdiffusion3 specification entry object")
            continue
        contig = entry.get("contig")
        if not isinstance(contig, str):
            _problem(report, f"{field_name}.contig", "is missing or is not a string", "write a binder range, chain break, and target residue range")
        else:
            match = RF3_CONTIG_RE.fullmatch(contig.strip())
            if match is None:
                _problem(report, f"{field_name}.contig", f"has unsupported RFdiffusion3 syntax: {contig!r}", "use MIN-MAX,/0,CHAINLOW-HIGH")
            else:
                binder_low, binder_high = int(match.group(1)), int(match.group(2))
                target_chain = match.group(3)
                target_low, target_high = int(match.group(4)), int(match.group(5))
                if binder_low > binder_high:
                    _problem(report, f"{field_name}.contig", "binder range runs from a larger value to a smaller value", "order the RFdiffusion3 binder range")
                if target_low > target_high:
                    _problem(report, f"{field_name}.contig", "target range runs from a larger value to a smaller value", "order the RFdiffusion3 target range")
                if isinstance(minimum, int) and isinstance(maximum, int):
                    if binder_low < minimum or binder_high > maximum:
                        _problem(report, f"{field_name}.contig", f"binder range {binder_low}-{binder_high} is outside campaign bounds {minimum}-{maximum}", "keep the RFdiffusion3 binder range inside binder.minimum_length and binder.maximum_length")
                if isinstance(expected_target_chain, str) and target_chain != expected_target_chain:
                    _problem(report, f"{field_name}.contig", f"names target chain {target_chain}, campaign names {expected_target_chain}", "use the campaign target chain in the RFdiffusion3 contig")
                chain_numbers = set(available.get(target_chain, []))
                missing = [str(number) for number in range(target_low, target_high + 1) if number not in chain_numbers]
                if missing:
                    _problem(report, f"{field_name}.contig", f"names {target_chain}{target_low}-{target_high}, but the structure lacks residue(s) {', '.join(missing[:8])}", "use a target span carried by the RFdiffusion3 input structure")

        hotspots = entry.get("select_hotspots")
        if hotspots is None:
            continue
        if not isinstance(hotspots, dict):
            _problem(report, f"{field_name}.select_hotspots", "is not an object", "write hotspot residue keys and atom-pair values")
            continue
        for residue in hotspots:
            parsed = _rf3_spec_residue(residue)
            if parsed is None:
                _problem(report, f"{field_name}.select_hotspots", f"has unsupported residue key {residue!r}", "use CHAINNUMBER residue keys")
                continue
            chain, number = parsed
            if number not in set(available.get(chain, [])):
                _problem(report, f"{field_name}.select_hotspots[{residue!r}]", f"names a residue the structure lacks: {residue}", "choose a hotspot carried by the RFdiffusion3 input structure")


def _check_rf3_generation_request(
    campaign: Mapping[str, Any],
    state: _TargetState,
    values: Mapping[str, str | None],
    report: PreflightReport,
    *,
    base_dir: Path,
) -> None:
    # --input-structure is an optional override. Without it the adapter designs against
    # the normalized structure the target stage wrote, so that is what gets checked.
    input_path = _resolved_path(values.get("--input-structure"), base_dir)
    if input_path is None:
        input_path = state.normalized_path
    if input_path is None:
        return
    structure_residues = state.normalized_residues
    if not input_path.is_file():
        _problem(report, "generator.input_structure", f"file does not exist: {input_path}", "supply the normalized target structure passed to RFdiffusion3")
    elif state.normalized_path is None or input_path.resolve() != state.normalized_path.resolve():
        structure_residues = _structure_residues(input_path, report, "generator.input_structure")

    specification_value = values.get("--specification")
    specification_path = _resolved_path(specification_value, base_dir)
    if specification_path is None:
        return
    _check_rf3_specification(
        specification_path,
        campaign,
        state,
        report,
        structure_residues=structure_residues,
    )


def _check_generator_stage(
    campaign: Mapping[str, Any],
    profile: Mapping[str, Any] | None,
    report: PreflightReport,
    *,
    adapter_id: str = RFDIFFUSION_ADAPTER_ID,
    display_name: str = "RFdiffusion",
    required: bool = True,
) -> None:
    """Check that one generator roster record and its stage agree.

    Every check here reads the wiring of a stage and none of them read the tool,
    so they hold for any backbone generator. `required` is the one difference. A
    campaign that declares a generator roster and no RFdiffusion record is a
    defect, because the package selects RFdiffusion by default. A campaign that
    names no Genie3 record is simply not running Genie3, so the Genie3 pass
    reports nothing rather than demanding a record.
    """
    generation_source: Mapping[str, Any] | None = None
    if isinstance(campaign.get("generation"), dict):
        generation_source = campaign
    elif profile is not None and isinstance(profile.get("generation"), dict):
        generation_source = profile
    if generation_source is None:
        return
    generation = generation_source.get("generation")
    generators = generation.get("generators") if isinstance(generation, dict) else None
    selected = [
        item
        for item in generators or []
        if isinstance(item, dict) and item.get("adapter_id") == adapter_id
    ]
    generation_prefix = (
        f"campaign.generation.generators[{adapter_id}]"
        if generation_source is campaign
        else f"profile.generation.generators[{adapter_id}]"
    )
    if not selected and not required:
        return
    if len(selected) != 1:
        _problem(report, generation_prefix, f"found {len(selected)} generator records", f"enable exactly one {display_name} generator record")
        return
    generator = selected[0]
    if generator.get("enabled") is not True:
        _problem(report, f"{generation_prefix}.enabled", "is not true", f"enable the {display_name} generator")
    stage_id = generator.get("command_stage")
    if not isinstance(stage_id, str) or not stage_id:
        _problem(report, f"{generation_prefix}.command_stage", "is missing", f"name the {display_name} command stage")
        return
    stages_source: Mapping[str, Any] | None = campaign if isinstance(campaign.get("stages"), list) else profile
    stages = stages_source.get("stages") if stages_source is not None else None
    stage_prefix = "campaign.stages" if stages_source is campaign else "profile.stages"
    matching = [
        item for item in stages or []
        if isinstance(item, dict) and item.get("stage_id") == stage_id
    ]
    if len(matching) != 1:
        _problem(report, f"{stage_prefix}[{stage_id}]", f"found {len(matching)} matching stages", f"declare exactly one stage for the {display_name} generator")
        return
    stage = matching[0]
    if stage.get("adapter_id") != adapter_id:
        _problem(report, f"{stage_prefix}[{stage_id}].adapter_id", f"is {stage.get('adapter_id')!r}", f"bind the stage to {adapter_id}")
    dependencies = stage.get("depends_on")
    if not isinstance(dependencies, list) or "target-prepare" not in dependencies:
        _problem(report, f"{stage_prefix}[{stage_id}].depends_on", "does not depend on target-prepare", f"make target-prepare complete before {display_name}")
    inputs = stage.get("inputs")
    if not isinstance(inputs, list) or "target-prepare:target-manifest" not in inputs:
        _problem(report, f"{stage_prefix}[{stage_id}].inputs", "does not declare target-prepare:target-manifest", "pass the target manifest into the generator stage")
    outputs = stage.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        _problem(report, f"{stage_prefix}[{stage_id}].outputs", "is missing or empty", "declare the candidate manifest output")


def check_backbone_route(
    campaign: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
    report: PreflightReport,
    *,
    catalog: Mapping[str, Any] | None = None,
) -> None:
    """Refuse a sequence-design stage fed by a generator that writes too few atoms.

    A backbone generator that writes C-alpha coordinates only can be wired into
    a designer that reads N, CA, C and O. The declared atom sets live on the
    packaged catalog, and a tool that declares `unknown` reports nothing here,
    so an unstated atom set never becomes a refusal.
    """
    try:
        problems = backbone_shape.route_problems(campaign, profile, catalog=catalog)
    except (OSError, ValueError) as exc:
        _problem(
            report,
            "campaign.backbone_route",
            f"could not be checked: {type(exc).__name__}: {exc}",
            "restore the packaged tool catalog so the backbone atom route can be read",
        )
        return
    for item in problems:
        _problem(report, item.field, item.problem, item.fix)


def _integer(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _selects_generator(
    campaign: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
    adapter_id: str,
) -> bool:
    """Report whether the campaign or its profile runs one generator adapter.

    A generator roster settles the question wherever one exists, because
    `enabled` lives on the roster record. A document that declares stages and no
    roster still runs the adapter, so a stage bound to it counts on its own.
    """
    records: list[Mapping[str, Any]] = []
    for source in (campaign, profile):
        if not isinstance(source, Mapping):
            continue
        generation = source.get("generation")
        if not isinstance(generation, Mapping):
            continue
        for item in generation.get("generators") or []:
            if isinstance(item, Mapping) and item.get("adapter_id") == adapter_id:
                records.append(item)
    if records:
        return any(item.get("enabled") is not False for item in records)
    for source in (campaign, profile):
        if not isinstance(source, Mapping):
            continue
        for item in source.get("stages") or []:
            if isinstance(item, Mapping) and item.get("adapter_id") == adapter_id:
                return True
    return False


def _check_genie3_profile(
    profile: Mapping[str, Any] | None,
    campaign: Mapping[str, Any],
    state: _TargetState | None,
    report: PreflightReport,
    *,
    artifact_root: Path,
) -> None:
    """Hold a Genie3 generator record to the checks the RFdiffusion record gets.

    The two tools take different work. RFdiffusion takes a contig string and a
    hotspot list. Genie3 takes the prepared target and a binder length range and
    samples inside it, and it reads the site residues out of the target manifest
    itself. So the checks that read a contig or a hotspot stay with RFdiffusion.
    The checks that read identity, resources, the shipped module, the run
    subcommand, the required flags, and the toolcheck argv hold for both, and
    this function runs the second set.

    It then checks the argv values the Genie3 wrapper validates at run time, so a
    command that would raise an `AdapterError` after the stage is allocated is
    refused here instead, before anything is paid for.

    Nothing here fires unless the campaign or the profile selects Genie3. A
    campaign that runs RFdiffusion alone reports exactly what it reported before.
    """
    if not isinstance(profile, Mapping):
        # A missing profile is already one problem. Reporting a missing Genie3
        # record inside it would say the same thing twice.
        return
    adapters = profile.get("adapters")
    matches = [
        item
        for item in adapters or []
        if isinstance(item, dict) and item.get("adapter_id") == GENIE3_ADAPTER_ID
    ]
    if not matches and not _selects_generator(campaign, profile, GENIE3_ADAPTER_ID):
        return
    prefix = f"profile.adapters[{GENIE3_ADAPTER_ID}]"
    if len(matches) != 1:
        _problem(report, prefix, f"found {len(matches)} matching adapter records", "include exactly one Genie3 adapter")
        return

    adapter = matches[0]
    _check_adapter_record(adapter, prefix, "Genie3", report)
    model_revision = adapter.get("model_revision")
    if isinstance(model_revision, str) and not _placeholder(model_revision):
        if GENIE3_REVISION_RE.fullmatch(model_revision) is None:
            _problem(report, f"{prefix}.model_revision", f"does not pin the weights as repository@40-character revision: {model_revision!r}", "record the weights repository and its commit revision in model_revision")

    command = adapter.get("command_argv_template")
    if not isinstance(command, list) or any(not isinstance(value, str) for value in command):
        _problem(report, f"{prefix}.command_argv_template", "must be a string argv list", "write the Genie3 run command as an argv list")
        return
    module = _adapter_module(command)
    if module not in GENIE3_ADAPTER_MODULES:
        _problem(report, f"{prefix}.command_argv_template", "does not invoke a supported shipped Genie3 adapter", "use claude_binder.adapters.genie3_generator or claude_binder.adapters.fal_genie3_generator")
    if "run" not in command:
        _problem(report, f"{prefix}.command_argv_template", "does not select the adapter run subcommand", "include the run subcommand")

    identity = adapter.get("environment_identity")
    # Neither Genie3 route declares an environment identity grammar of its own,
    # so the only format checked is the Modal one, and only when the value claims
    # it. This is what `runtime_validator._adapter_checks` does for every adapter
    # that is not RFdiffusion, and asserting a shape no source states would be an
    # invented rule.
    if isinstance(identity, str) and identity.startswith("modal-env:"):
        if ENVIRONMENT_IDENTITY_RE.fullmatch(identity) is None:
            _problem(report, f"{prefix}.environment_identity", "does not match modal-env:name@spec_sha=16-to-64-hex format", "record the Modal environment identity and spec hash")

    for flag in GENIE3_COMMAND_FLAGS:
        _option(command, flag, report, prefix)

    toolcheck = adapter.get("toolcheck_argv")
    if not isinstance(toolcheck, list) or "toolcheck" not in toolcheck:
        _problem(report, f"{prefix}.toolcheck_argv", "does not select the adapter toolcheck subcommand", "include the adapter toolcheck command")
    elif module is not None and module not in toolcheck:
        _problem(report, f"{prefix}.toolcheck_argv", "does not invoke the selected generator adapter", f"use {module} toolcheck")

    context = _profile_context(campaign, state, artifact_root)
    values: dict[str, str | None] = {}
    for flag in dict.fromkeys((*GENIE3_COMMAND_FLAGS, *GENIE3_VALUE_FLAGS)):
        raw = _option(command, flag, report, prefix, required=False)
        values[flag] = _resolve_token(raw, context, report, f"{prefix}.command_argv_template[{flag}]") if raw is not None else None
    _check_genie3_profile_values(campaign, command, values, prefix, report)


def _check_genie3_profile_values(
    campaign: Mapping[str, Any],
    command: Sequence[str],
    values: Mapping[str, str | None],
    prefix: str,
    report: PreflightReport,
) -> None:
    """Refuse a Genie3 command the wrapper would refuse once the stage is running.

    Every rule here is either one `adapters.genie3_generator.run` raises an
    `AdapterError` for, or one where the command and the campaign disagree about
    a value they both name. Reading them off the argv template costs nothing.
    """
    for flag in ("--count", "--binder-length-min", "--binder-length-max"):
        value = values.get(flag)
        if value is None:
            continue
        number = _integer(value)
        if number is None or number < 1:
            _problem(report, f"{prefix}.command_argv_template[{flag}]", f"is not a positive integer: {value!r}", "supply a positive Genie3 value")
    minimum = _integer(values.get("--binder-length-min"))
    maximum = _integer(values.get("--binder-length-max"))
    if minimum is not None and maximum is not None and maximum < minimum:
        _problem(report, f"{prefix}.command_argv_template", f"binder length bounds {minimum}-{maximum} are empty", "raise --binder-length-max or lower --binder-length-min")

    binder = campaign.get("binder") if isinstance(campaign, Mapping) else None
    if isinstance(binder, dict):
        for flag, field_name in (("--binder-length-min", "minimum_length"), ("--binder-length-max", "maximum_length")):
            requested = _integer(values.get(flag))
            declared = binder.get(field_name)
            if requested is None or not isinstance(declared, int) or isinstance(declared, bool):
                continue
            if requested != declared:
                _problem(report, f"{prefix}.command_argv_template[{flag}]", f"is {requested}, campaign binder.{field_name} is {declared}", "make the command and campaign request the same binder length bound")

    for flag in ("--binder-chain", "--target-chain", "--generator-binder-chain"):
        value = values.get(flag)
        if value is not None and genie3.CHAIN_ID_RE.fullmatch(value) is None:
            _problem(report, f"{prefix}.command_argv_template[{flag}]", f"{value!r} is not one letter or digit", "use a one-character chain ID")
    binder_value, target_value = values.get("--binder-chain"), values.get("--target-chain")
    if binder_value is not None and target_value is not None and binder_value == target_value:
        _problem(report, f"{prefix}.command_argv_template", "binder and target chains are identical", "use distinct designed and fixed chain IDs")
    if isinstance(binder, dict):
        if binder_value is not None and isinstance(binder.get("binder_chain_id"), str) and binder_value != binder["binder_chain_id"]:
            _problem(report, f"{prefix}.command_argv_template[--binder-chain]", f"names {binder_value}, campaign names {binder['binder_chain_id']}", "make the command and campaign use the same binder chain")
        if target_value is not None and isinstance(binder.get("target_chain_id"), str) and target_value != binder["target_chain_id"]:
            _problem(report, f"{prefix}.command_argv_template[--target-chain]", f"names {target_value}, campaign names {binder['target_chain_id']}", "make the command and campaign use the same target chain")

    for flag in ("--problem-id", "--generator-id"):
        value = values.get(flag)
        if value is not None and genie3.IDENTIFIER_RE.fullmatch(value) is None:
            _problem(report, f"{prefix}.command_argv_template[{flag}]", f"is not a plain identifier: {value!r}", "use letters, digits, dot, dash, or underscore")

    seed = values.get("--seed")
    if seed is not None:
        number = _integer(seed)
        if number is None or number < 0:
            _problem(report, f"{prefix}.command_argv_template[--seed]", f"is not a whole number of zero or more: {seed!r}", "supply a seed of zero or more")
    scale = values.get("--direction-scale")
    if scale is not None:
        try:
            float(scale)
        except ValueError:
            _problem(report, f"{prefix}.command_argv_template[--direction-scale]", f"is not a number: {scale!r}", "supply a numeric Genie3 direction scale")
    seed_key = values.get("--seed-config-key")
    if seed_key is not None and genie3.CONFIG_KEY_RE.fullmatch(seed_key) is None:
        _problem(report, f"{prefix}.command_argv_template[--seed-config-key]", f"is not a dotted identifier path: {seed_key!r}", "name the configuration key the Genie3 build reads, such as generation.seed")
    # Genie3 has no seed argument in any invocation this package records, so the
    # wrapper refuses to guess one and raises unless the command declares which
    # it is. Both routes raise on this, so preflight can settle it from the argv.
    if "--seed-config-key" not in command and "--allow-unseeded" not in command:
        _problem(report, f"{prefix}.command_argv_template", "declares neither --seed-config-key nor --allow-unseeded, and the Genie3 wrapper refuses to guess where a seed goes", "pass --seed-config-key with the configuration key the build reads, or --allow-unseeded to record that the run carries no seed")


def preflight_campaign(
    campaign: Mapping[str, Any],
    profile: Mapping[str, Any] | None = None,
    *,
    campaign_path: Path | None = None,
    profile_path: Path | None = None,
    target_manifest_path: Path | None = None,
) -> PreflightReport:
    """Check a resolved campaign and RFdiffusion profile without execution."""
    del profile_path
    report = PreflightReport()
    base_dir = campaign_path.resolve().parent if campaign_path is not None else Path.cwd()
    if not isinstance(campaign, Mapping):
        _problem(report, "campaign", "is not an object", "supply a resolved campaign JSON object")
        return report
    _check_campaign_fields(campaign, report)
    targets = campaign.get("targets")
    target = next(
        (
            item
            for item in targets or []
            if isinstance(item, dict) and item.get("role") == "primary"
        ),
        None,
    ) if isinstance(targets, list) else None
    manifest, manifest_path = _check_manifest(
        campaign, base_dir, target, target_manifest_path, report
    )
    state = _target_state(campaign, base_dir, manifest, manifest_path, report)
    if state is not None and manifest is not None and state.residue_map_path is not None:
        expected_map_hash = manifest.get("residue_map_sha256")
        if isinstance(expected_map_hash, str) and expected_map_hash:
            if _sha256(state.residue_map_path) != expected_map_hash:
                _problem(report, "target_manifest.residue_map_sha256", "does not match the residue map file", "regenerate the target manifest or restore the mapped file")
    if manifest_path is not None and manifest_path.parent.name == "inputs":
        artifact_root = manifest_path.parent.parent.resolve()
    else:
        artifact_root = (base_dir / "artifacts").resolve()
    adapter, values = _check_profile(
        profile,
        campaign,
        state,
        report,
        artifact_root=artifact_root,
    )
    _check_generator_stage(campaign, profile, report)
    _check_generator_stage(
        campaign,
        profile,
        report,
        adapter_id=GENIE3_ADAPTER_ID,
        display_name="Genie3",
        required=False,
    )
    _check_genie3_profile(
        profile,
        campaign,
        state,
        report,
        artifact_root=artifact_root,
    )
    check_backbone_route(campaign, profile, report)
    _check_generation_request(
        profile,
        campaign,
        state,
        adapter,
        values,
        report,
        base_dir=base_dir,
    )
    report.checked.extend(("target manifest", "source residues", "cleaned residues", "contig", "hotspot", "RFdiffusion profile", "RFdiffusion3 specification if present", "Genie3 profile if selected", "backbone atom route"))
    return report


def _load_input(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local RFdiffusion generator preflight.")
    parser.add_argument("--campaign", type=Path, required=True, help="Resolved campaign JSON.")
    parser.add_argument("--profile", type=Path, required=True, help="Resolved RFdiffusion profile JSON.")
    parser.add_argument("--target-manifest", type=Path, default=None, help="Published target manifest JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        campaign = _load_input(args.campaign)
        profile = _load_input(args.profile)
        report = preflight_campaign(
            campaign,
            profile,
            campaign_path=args.campaign,
            profile_path=args.profile,
            target_manifest_path=args.target_manifest,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        refusal = Refusal(
            cause="The generator preflight could not read its inputs.",
            expected="Readable campaign and profile JSON files.",
            expected_source="The --campaign and --profile arguments.",
            found=f"{type(exc).__name__}: {exc}",
            found_source="The file read that raised the error.",
            scope="The generator preflight started no provider command.",
            action="Supply readable campaign and profile JSON files.",
            escalation="Send the input file error to the generator maintainer.",
        )
        print(refusal.text(), file=sys.stderr)
        return refusal.exit_code
    if report.ok:
        print("generator preflight: PASS")
        print("- checked: " + ", ".join(report.checked))
        return int(ExitCode.VERIFIED)
    refusal = report.refusal()
    if refusal is not None:
        print(refusal.text(), file=sys.stderr)
        return refusal.exit_code
    return int(ExitCode.FAILURE)


if __name__ == "__main__":
    raise SystemExit(main())
