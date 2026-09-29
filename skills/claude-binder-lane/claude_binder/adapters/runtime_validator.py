#!/usr/bin/env python3
"""Validate resolved adapter identities and recorded tool bring-up evidence.

This stage performs local checks only. It never starts a tool, downloads weights, or
turns an absent roster record into a pass. Existing evidence is inspected when its
paths are available; otherwise the report says that the check could not be checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import lane
from .. import backbone_shape
from .. import control_separation
from .. import generator_preflight
from ..filter_contracts import is_not_applicable
from ..paths import package_file


PASS = "pass"
FAIL = "fail"
COULD_NOT_BE_CHECKED = "could not be checked"
PLACEHOLDER_RE = re.compile(r"__REQUIRED__|<required", re.IGNORECASE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# spec_sha is the content hash of the environment source file the Claude Science
# release ships, truncated to 16 hex. It is not a Modal value, and Modal serves no
# field to confirm it against. provider_authorization.py records the same width
# from a live 2026-08-29 details read. Demanding 64 here made the check impossible
# to satisfy with a real identity.
MODAL_ENV_RE = re.compile(r"^modal-env:[^@\s]+@spec_sha=[0-9a-f]{16,64}$")
# A publisher revision pin, as `<owner>/<repository>@<revision>`. A deployment that resolves a
# model repository rather than a single checkpoint file has no file digest to record, so this
# is the only weight identity it can report.
REVISION_PIN_RE = re.compile(r"([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)@([0-9a-f]{7,64})")
# The profile names the role, so the ledger cannot choose which output shape it is checked
# against. A sequence designer emits sequence records and no structure.
SEQUENCE_DESIGNER_ROLES = frozenset({"sequence-designer"})
REQUIRED_ROSTER_FIELDS = (
    "model",
    "status",
    "image_id",
    "s/design",
    "$/design",
    "output_shape",
    "validated_at",
    "weights_sha256",
    "wall_clock_s",
    "gpu_type",
    "output_sha256",
    "target_structure_sha256",
)


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    reason: str
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
        }
        if self.details:
            value["details"] = self.details
        return value


def _check(name: str, status: str, reason: str, **details: Any) -> Check:
    return Check(name, status, reason, details or None)


def _placeholder(value: Any) -> bool:
    return isinstance(value, str) and PLACEHOLDER_RE.search(value) is not None


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not _placeholder(value)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{label} is invalid: {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_roster_document(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a roster and retain its document metadata beside its rows."""
    if not path.is_file():
        raise ValueError(f"model-roster ledger is missing: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"model-roster ledger line {line_number} is invalid: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(f"model-roster ledger line {line_number} is not an object")
            rows.append(item)
        return {"models": rows}, rows
    if isinstance(value, list):
        rows = value
        document: dict[str, Any] = {"models": rows}
    elif isinstance(value, dict):
        document = value
        rows = next(
            (value[key] for key in ("models", "roster", "rows") if isinstance(value.get(key), list)),
            None,
        )
        if rows is None:
            rows = [value]
    else:
        raise ValueError(f"model-roster ledger must contain objects: {path}")
    if any(not isinstance(item, dict) for item in rows):
        raise ValueError(f"model-roster ledger contains a non-object row: {path}")
    return document, list(rows)


def _load_roster(path: Path) -> list[dict[str, Any]]:
    """Read roster rows while preserving the existing row-only API."""
    _document, rows = _load_roster_document(path)
    return rows


def _configured_roster_path(config: Mapping[str, Any], config_path: Path) -> Path | None:
    runtime = config.get("runtime")
    configured = runtime.get("model_roster_path") if isinstance(runtime, Mapping) else None
    if not isinstance(configured, str) or not configured:
        return None
    path = Path(configured)
    if path.is_absolute():
        return path.resolve()
    relative_path = (config_path.parent / path).resolve()
    if relative_path.is_file():
        return relative_path
    if path.parts and path.parts[0] == "data":
        try:
            return package_file(*path.parts).resolve()
        except (FileNotFoundError, ValueError):
            pass
    return relative_path


PACKAGED_EVIDENCE_DIRECTORY = "roster-evidence"


def _configured_evidence_root(
    config: Mapping[str, Any],
    roster_document: Mapping[str, Any],
    config_path: Path,
    roster_path: Path | None = None,
) -> Path | None:
    """Resolve the roster evidence root, preferring configuration over the shipped copy."""
    declaration = roster_document.get("evidence_root")
    runtime_key: str | None = None
    if isinstance(declaration, Mapping):
        for key in ("runtime_key", "runtime_config_key", "config_key"):
            value = declaration.get(key)
            if isinstance(value, str) and value.strip():
                runtime_key = value.strip()
                break
    elif isinstance(declaration, str) and declaration.strip():
        runtime_key = declaration.strip()
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        return None
    keys = [runtime_key] if runtime_key else []
    keys.extend(key for key in ("model_roster_evidence_root", "evidence_root") if key not in keys)
    for key in keys:
        value = runtime.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = config_path.resolve().parent / path
            return path.resolve()
    # A user who never ran their own qualification has no evidence directory of their own.
    # The evidence for the shipped rows travels beside the roster, so fall back to it.
    if roster_path is not None:
        packaged = roster_path.resolve().parent / PACKAGED_EVIDENCE_DIRECTORY
        if packaged.is_dir():
            return packaged.resolve()
    return None


def _roster_requires_evidence_root(roster_document: Mapping[str, Any]) -> bool:
    """Return whether the roster declares relative paths under a runtime root.

    Materialize refuses a bundle for a roster this returns true for when it can find no
    evidence to copy, so both places have to read the declaration the same way. The
    reading lives in ``lane`` and is called from here.
    """
    return lane.roster_declares_evidence_root(roster_document)


def _resolve_evidence_path(
    value: str,
    evidence_root: Path | None,
    *,
    require_root: bool,
) -> tuple[Path | None, str | None]:
    """Resolve one roster path while keeping it inside the configured root."""
    path = Path(value)
    if path.is_absolute():
        if require_root:
            return None, f"evidence path must be relative to the configured root: {value}"
        return path.resolve(), None
    if evidence_root is None:
        return None, "roster evidence root is missing for a relative evidence path"
    root = evidence_root.resolve()
    candidate = (root / path).resolve()
    if candidate != root and root not in candidate.parents:
        return None, f"evidence path escapes the configured root: {value}"
    return candidate, None


def _find_roster_row(rows: list[dict[str, Any]], adapter: Mapping[str, Any]) -> dict[str, Any] | None:
    adapter_id = str(adapter.get("adapter_id", ""))
    names = {
        adapter_id,
        str(adapter.get("model_id", "")),
        str(adapter.get("id", "")),
    }
    names.discard("")
    exact = [
        row
        for row in rows
        if any(row.get(key) in names for key in ("adapter_id", "model", "model_id", "tool", "id"))
    ]
    return exact[0] if len(exact) == 1 else None


def _evidence(row: Mapping[str, Any], *names: str) -> Any:
    for container in (row, row.get("evidence", {}), row.get("checks", {})):
        if isinstance(container, Mapping):
            for name in names:
                if name in container:
                    return container[name]
    return None


def _bool_check(name: str, value: Any, missing_reason: str) -> Check:
    if isinstance(value, bool):
        return _check(name, PASS if value else FAIL, "recorded true" if value else "recorded false")
    return _check(name, COULD_NOT_BE_CHECKED, missing_reason)


def _inspect_pdb(path: Path) -> tuple[bool, int, int, bool, str]:
    chains: set[str] = set()
    residues: set[tuple[str, str, str]] = set()
    saw_atom = False
    nan_coordinate = False
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return False, 0, 0, False, f"could not read {path}: {exc}"
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        saw_atom = True
        if len(line) < 54:
            return False, 0, 0, False, f"atom record is shorter than coordinate columns: {path}"
        chain = line[21].strip()
        chains.add(chain)
        residues.add((chain, line[22:26].strip(), line[26].strip()))
        try:
            coordinates = [float(line[start : start + 8]) for start in (30, 38, 46)]
        except ValueError:
            return False, len(chains), len(residues), False, f"atom coordinates are not numeric: {path}"
        if any(not math.isfinite(value) for value in coordinates):
            nan_coordinate = True
    if not saw_atom:
        return False, len(chains), len(residues), nan_coordinate, f"structure has no PDB atom records: {path}"
    if nan_coordinate:
        return False, len(chains), len(residues), True, f"structure has NaN or infinite coordinates: {path}"
    if not chains:
        return False, 0, len(residues), False, f"structure has no chain identifiers: {path}"
    if len(residues) < 30:
        return False, len(chains), len(residues), False, f"structure has {len(residues)} residues, fewer than 30: {path}"
    return True, len(chains), len(residues), False, "PDB structure parsed"


def _structure_check(
    row: Mapping[str, Any],
    required_count: int,
    *,
    evidence_root: Path | None = None,
    require_evidence_root: bool = False,
) -> Check:
    if not isinstance(required_count, int) or isinstance(required_count, bool) or required_count < 2 or required_count > 4:
        return _check("outputs", COULD_NOT_BE_CHECKED, "the documented canary n_designs count is unavailable for output coverage")
    output_dir_value = _evidence(row, "output_dir", "structure_output_dir")
    output_paths_value = _evidence(row, "output_paths", "structure_paths", "output_files")
    paths: list[Path] = []
    path_errors: list[str] = []
    if isinstance(output_paths_value, list):
        for value in output_paths_value:
            if not isinstance(value, str):
                path_errors.append("output path is not a string")
                continue
            resolved, error = _resolve_evidence_path(
                value,
                evidence_root,
                require_root=require_evidence_root,
            )
            if error:
                path_errors.append(error)
            elif resolved is not None:
                paths.append(resolved)
    elif isinstance(output_dir_value, str) and output_dir_value:
        output_dir, error = _resolve_evidence_path(
            output_dir_value,
            evidence_root,
            require_root=require_evidence_root,
        )
        if error or output_dir is None:
            return _check("outputs", COULD_NOT_BE_CHECKED, error or "output directory could not be resolved")
        if not output_dir.is_dir():
            return _check("outputs", FAIL, f"output directory is missing: {output_dir}")
        paths = sorted(
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdb", ".cif", ".mmcif"}
        )
    else:
        return _check("outputs", COULD_NOT_BE_CHECKED, "no local output directory or structure paths were recorded")
    if not paths:
        return _check(
            "outputs",
            COULD_NOT_BE_CHECKED if path_errors else FAIL,
            "; ".join(path_errors) if path_errors else "no PDB or mmCIF outputs were recorded",
        )
    if path_errors:
        return _check(
            "outputs",
            COULD_NOT_BE_CHECKED,
            "; ".join(path_errors),
        )
    valid = 0
    failures: list[str] = []
    for path in paths:
        if path.suffix.lower() == ".pdb":
            ok, chain_count, residue_count, _nan, reason = _inspect_pdb(path)
            if ok and chain_count >= 1 and residue_count >= 30:
                valid += 1
            else:
                failures.append(reason)
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                failures.append(f"could not read {path}: {exc}")
                continue
            chains = lane.cif_chain_ids(path) if path.is_file() else set()
            if "_atom_site." not in text or not chains:
                failures.append(f"mmCIF did not expose atom_site chains: {path}")
                continue
            declared_residues = _evidence(row, "output_residue_count", "residue_count")
            if isinstance(declared_residues, int) and declared_residues >= 30:
                valid += 1
            else:
                failures.append(
                    f"mmCIF residue count was not recorded as an integer of at least 30: {path}"
                )
    if valid < required_count:
        return _check(
            "outputs",
            FAIL,
            f"only {valid} of {required_count} structure outputs passed parsing and geometry checks",
            valid_count=valid,
            inspected_count=len(paths),
            failures=failures[:5],
        )
    return _check("outputs", PASS, f"{valid} structure outputs passed parsing and geometry checks", valid_count=valid)


def _sequence_outputs_check(
    row: Mapping[str, Any],
    required_count: int,
    *,
    evidence_root: Path | None = None,
    require_evidence_root: bool = False,
) -> Check:
    """Check the outputs of an adapter whose role emits sequences and no structure.

    A sequence designer returns FASTA records. Asking it for a parsable structure asks for a
    file it never writes, and no bring-up of a real designer can produce one. The count and
    the file parsing stay as strict as the structure branch, and the role comes from the
    profile, so a roster cannot choose which branch it is checked against.
    """
    count, error = _fasta_count(
        row,
        evidence_root=evidence_root,
        require_evidence_root=require_evidence_root,
    )
    if error:
        return _check("outputs", FAIL, error)
    if count is None:
        return _check("outputs", COULD_NOT_BE_CHECKED, "no local sequence output paths or directory were recorded")
    if count < required_count:
        return _check(
            "outputs",
            FAIL,
            f"only {count} of {required_count} sequence outputs were found",
            valid_count=count,
        )
    return _check("outputs", PASS, f"{count} sequence outputs were found and parsed", valid_count=count)


def _outputs_check(
    adapter: Mapping[str, Any],
    row: Mapping[str, Any],
    required_count: Any,
    *,
    evidence_root: Path | None = None,
    require_evidence_root: bool = False,
) -> Check:
    if not isinstance(required_count, int) or isinstance(required_count, bool) or required_count < 2 or required_count > 4:
        return _check("outputs", COULD_NOT_BE_CHECKED, "the documented canary n_designs count is unavailable for output coverage")
    if str(adapter.get("role", "")) in SEQUENCE_DESIGNER_ROLES:
        return _sequence_outputs_check(
            row,
            required_count,
            evidence_root=evidence_root,
            require_evidence_root=require_evidence_root,
        )
    return _structure_check(
        row,
        required_count,
        evidence_root=evidence_root,
        require_evidence_root=require_evidence_root,
    )


def _fasta_count(
    row: Mapping[str, Any],
    *,
    evidence_root: Path | None = None,
    require_evidence_root: bool = False,
) -> tuple[int | None, str | None]:
    paths_value = _evidence(row, "sequence_paths", "sequence_output_paths")
    paths: list[Path] = []
    if isinstance(paths_value, list):
        for value in paths_value:
            if not isinstance(value, str):
                return None, "sequence output path is not a string"
            resolved, error = _resolve_evidence_path(
                value,
                evidence_root,
                require_root=require_evidence_root,
            )
            if error:
                return None, error
            if resolved is not None:
                paths.append(resolved)
    else:
        directory = _evidence(row, "sequence_output_dir", "sequence_dir")
        if isinstance(directory, str) and directory:
            root, error = _resolve_evidence_path(
                directory,
                evidence_root,
                require_root=require_evidence_root,
            )
            if error or root is None:
                return None, error or "sequence output directory could not be resolved"
            if not root.is_dir():
                return None, f"sequence output directory is missing: {root}"
            paths = sorted(root.rglob("*.fasta")) + sorted(root.rglob("*.fa"))
    if not paths:
        return None, None
    count = 0
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return None, f"could not read sequence output {path}: {exc}"
        if not any(line.startswith(">") for line in text.splitlines()):
            return None, f"sequence output has no FASTA header: {path}"
        count += sum(1 for line in text.splitlines() if line.startswith(">"))
    return count, None


def _sequence_check(
    row: Mapping[str, Any],
    *,
    evidence_root: Path | None = None,
    require_evidence_root: bool = False,
) -> Check:
    explicit = _evidence(row, "sequence_adapter_consumed", "downstream_sequence_adapter_consumed")
    explicit_count = _evidence(row, "sequence_output_count", "downstream_sequence_count")
    if explicit is False:
        return _check("sequence-design", FAIL, "recorded downstream sequence-design consumption is false")
    if isinstance(explicit, bool) and explicit is True and isinstance(explicit_count, int) and explicit_count >= 1:
        return _check("sequence-design", PASS, "recorded downstream adapter consumption and sequence output", count=explicit_count)
    count, error = _fasta_count(
        row,
        evidence_root=evidence_root,
        require_evidence_root=require_evidence_root,
    )
    if error:
        return _check("sequence-design", FAIL, error)
    if count is not None:
        return _check("sequence-design", PASS if count >= 1 else FAIL, f"found {count} FASTA records", count=count)
    return _check("sequence-design", COULD_NOT_BE_CHECKED, "no downstream sequence-design evidence or local FASTA output was recorded")


def _deployment_identity_check(adapter: Mapping[str, Any], row: Mapping[str, Any], image_id: Any) -> Check:
    """Check the production identity a route without a container image digest does have.

    The profile records `not_applicable: <reason>` wherever the execution route builds no
    Modal image. Comparing a roster field against that sentence establishes nothing, and a roster that copies the sentence into
    `image_id` matches it exactly and passes for free. A fal application has a deployment
    revision and a resolved environment instead, so this branch checks those: the roster has
    to name the deployment it was brought up against, and the environment it recorded has to
    be the environment the profile resolved.
    """
    if not is_not_applicable(image_id):
        return _check(
            "production-image",
            FAIL,
            f"the profile records no container image digest for this route and the roster records image_id {image_id!r}",
        )
    deployment = _evidence(row, "deployment_id", "production_deployment_id")
    if not _nonempty(deployment):
        return _check(
            "production-image",
            COULD_NOT_BE_CHECKED,
            "the roster records no deployment identity for a route that produces no container image digest",
        )
    expected_environment = adapter.get("environment_identity")
    observed_environment = _evidence(row, "environment_identity")
    if not _nonempty(expected_environment) or not _nonempty(observed_environment):
        return _check(
            "production-image",
            COULD_NOT_BE_CHECKED,
            "the profile or the roster records no environment identity to compare",
        )
    if expected_environment != observed_environment:
        return _check(
            "production-image",
            FAIL,
            f"roster environment identity {observed_environment!r} does not match profile environment identity {expected_environment!r}",
        )
    return _check(
        "production-image",
        PASS,
        "the roster names the deployment and its environment identity matches the profile",
        deployment_id=deployment,
    )


def _same_image_and_flags(adapter: Mapping[str, Any], row: Mapping[str, Any]) -> list[Check]:
    checks: list[Check] = []
    expected_image = adapter.get("resources", {}).get("container_image_digest") if isinstance(adapter.get("resources"), Mapping) else None
    image_id = _evidence(row, "image_id", "production_image_id")
    if is_not_applicable(expected_image):
        checks.append(_deployment_identity_check(adapter, row, image_id))
    elif not _nonempty(expected_image) or not _nonempty(image_id):
        checks.append(_check("production-image", COULD_NOT_BE_CHECKED, "production image identity or roster image_id is missing"))
    elif expected_image != image_id:
        checks.append(_check("production-image", FAIL, f"roster image_id {image_id!r} does not match profile image identity {expected_image!r}"))
    else:
        checks.append(_check("production-image", PASS, "roster image_id matches the profile image identity"))
    explicit = _evidence(row, "flags_match_production", "same_flags_as_production")
    if isinstance(explicit, bool):
        checks.append(_bool_check("production-flags", explicit, "production flag comparison is missing"))
    else:
        checks.append(_check("production-flags", COULD_NOT_BE_CHECKED, "the roster does not record whether bring-up flags match production flags"))
    return checks


def _roster_metadata(row: Mapping[str, Any]) -> list[Check]:
    missing = [field for field in REQUIRED_ROSTER_FIELDS if field not in row]
    if missing:
        return [_check("roster-record", COULD_NOT_BE_CHECKED, "roster row is missing required verification fields", missing=missing)]
    output_hashes = row.get("output_sha256")
    if not isinstance(output_hashes, list) or not output_hashes or not isinstance(output_hashes[0], str) or not SHA256_RE.fullmatch(output_hashes[0]):
        return [_check("roster-record", FAIL, "output_sha256[0] is not a lowercase SHA-256 digest")]
    weights = row.get("weights_sha256")
    if not is_not_applicable(weights) and (not isinstance(weights, str) or not SHA256_RE.fullmatch(weights)):
        return [_check("roster-record", FAIL, "weights_sha256 is not a lowercase SHA-256 digest")]
    target_structure_sha256 = row.get("target_structure_sha256")
    if not isinstance(target_structure_sha256, str) or not SHA256_RE.fullmatch(target_structure_sha256):
        return [_check("roster-record", FAIL, "target_structure_sha256 is not a lowercase SHA-256 digest")]
    return [_check("roster-record", PASS, "required roster identity and audit fields are present")]


def _revision_pins(value: Any) -> frozenset[tuple[str, str]]:
    return frozenset(REVISION_PIN_RE.findall(str(value)))


def _revision_pin_check(adapter: Mapping[str, Any], row: Mapping[str, Any]) -> Check:
    """Check a weight identity for a deployment that holds no checkpoint file.

    A single-file checkpoint has a digest, so a deployment that downloads one records it and
    takes the digest branch below. A deployment that resolves a model repository holds many
    files, pins them by the publisher's revision, and can report no file digest. The revision
    is then the only weight identity it has, so this branch requires it to be exactly the
    revision the profile pins. A roster that records a deliberate absence and nothing else
    still fails.
    """
    observed = _evidence(row, "weights_revision", "weights_model_revision")
    if not _nonempty(observed):
        return _check(
            "weights",
            COULD_NOT_BE_CHECKED,
            "weights_sha256 records a deliberate absence and the roster records no weights_revision in its place",
        )
    observed_pins = _revision_pins(observed)
    expected_pins = _revision_pins(adapter.get("model_revision"))
    if not observed_pins or not expected_pins:
        return _check(
            "weights",
            COULD_NOT_BE_CHECKED,
            "no repository revision pin could be read from the roster weights_revision or the profile model_revision",
        )
    if observed_pins != expected_pins:
        return _check(
            "weights",
            FAIL,
            "the roster weight revision pins are not the pins the profile resolved",
            roster=sorted(f"{repo}@{revision}" for repo, revision in observed_pins),
            profile=sorted(f"{repo}@{revision}" for repo, revision in expected_pins),
        )
    return _check(
        "weights",
        PASS,
        "the deployment resolved the repository revisions the profile pins",
        pins=sorted(f"{repo}@{revision}" for repo, revision in observed_pins),
    )


def _weights_check(adapter: Mapping[str, Any], row: Mapping[str, Any]) -> Check:
    actual = _evidence(row, "weights_sha256")
    if is_not_applicable(actual):
        return _revision_pin_check(adapter, row)
    if not isinstance(actual, str) or not SHA256_RE.fullmatch(actual):
        return _check("weights", COULD_NOT_BE_CHECKED, "weights_sha256 is missing or is not a lowercase SHA-256 digest")
    upstream_match = _evidence(row, "weights_match_upstream", "weights_checksum_matches_upstream")
    upstream_hash = _evidence(row, "upstream_weights_sha256", "release_weights_sha256")
    if upstream_match is False:
        return _check("weights", FAIL, "recorded weights checksum does not match the upstream release")
    if upstream_match is not True and upstream_hash != actual:
        return _check("weights", COULD_NOT_BE_CHECKED, "the roster does not record an upstream checksum comparison")
    model_revision = adapter.get("model_revision")
    match = re.search(r"sha256:([0-9a-f]{64})", str(model_revision))
    if match and match.group(1) != actual:
        return _check("weights", FAIL, "weights_sha256 does not match the profile model_revision digest")
    return _check("weights", PASS, "weights checksum is recorded and matched to an upstream or profile digest")


def _run_check(row: Mapping[str, Any]) -> Check:
    count = _evidence(row, "n_designs", "design_count")
    exit_code = _evidence(row, "exit_code", "run_exit_code")
    if not isinstance(count, int) or isinstance(count, bool) or not isinstance(exit_code, int) or isinstance(exit_code, bool):
        return _check("canary-run", COULD_NOT_BE_CHECKED, "n_designs and exit_code are both required")
    if count < 2 or count > 4:
        return _check("canary-run", FAIL, f"n_designs={count} is outside the documented range 2-4")
    if exit_code != 0:
        return _check("canary-run", FAIL, f"end-to-end canary exit_code was {exit_code}")
    return _check("canary-run", PASS, "end-to-end canary used 2-4 designs and exited 0", n_designs=count)


# Evidence that describes a contract rather than a qualification. A replayed response never
# reached a provider, and a fixture-scoped canary measured a shipped structure rather than
# the campaign's target. Both are useful and neither may gate a paid dispatch. Defined here
# rather than in `qualify` because this module is what reads a roster back, and a roster can
# be hand-edited after it is written.
UNQUALIFIABLE_EVIDENCE_MODES = frozenset({"recorded-replay"})
UNQUALIFIABLE_QUALIFICATION_SCOPES = frozenset({"fixture-contract"})


def _provenance_markers(row: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    """Every provenance marker in a row, including markers nested under row or evidence.

    `qualify._receipt` folds a receipt's `row` and `evidence` objects up into the top level
    with `merged.update(nested)`, so a nested live marker overwrites a top-level replay
    marker before any gate sees it. Reading both levels closes that, and reading them
    without merging means the disqualifying value wins wherever it sits. A refusal that a
    nested key can switch off is not a refusal.
    """
    found: list[tuple[str, Any, Any]] = []

    def collect(source: Mapping[str, Any], where: str) -> None:
        found.append((where, source.get("evidence_mode"), source.get("qualification_scope")))
        for key in ("row", "evidence"):
            nested = source.get(key)
            if isinstance(nested, Mapping):
                collect(nested, f"{where}.{key}" if where else key)

    collect(row, "")
    return found


def unqualifiable_provenance(row: Mapping[str, Any]) -> list[str]:
    """Reasons a row describes something other than a live campaign-target canary."""
    reasons: list[str] = []
    for where, mode, scope in _provenance_markers(row):
        location = f" at {where.lstrip('.')}" if where else ""
        if isinstance(mode, str) and mode in UNQUALIFIABLE_EVIDENCE_MODES:
            reasons.append(
                f"evidence_mode is {mode}{location}, which replays a recorded response rather "
                "than dispatching to a provider, so it cannot qualify a deployment"
            )
        if isinstance(scope, str) and scope in UNQUALIFIABLE_QUALIFICATION_SCOPES:
            reasons.append(
                f"qualification_scope is {scope}{location}, which measured a shipped fixture "
                "rather than this campaign's target, so it cannot qualify a model against it"
            )
    return list(dict.fromkeys(reasons))


def _model_checks(
    adapter: Mapping[str, Any],
    row: Mapping[str, Any] | None,
    target_id: str | None,
    target_structure_sha256: str | None,
    *,
    evidence_root: Path | None = None,
    require_evidence_root: bool = False,
) -> list[Check]:
    if row is None:
        return [_check("roster", COULD_NOT_BE_CHECKED, "no model-roster row matches this model or adapter")]
    checks = _roster_metadata(row)
    status = row.get("status")
    if status is None:
        checks.append(_check("roster-status", COULD_NOT_BE_CHECKED, "roster status is missing"))
    elif str(status).upper() != "PASS":
        checks.append(_check("roster-status", FAIL, f"roster status is {status!r}, not PASS"))
    else:
        provenance = unqualifiable_provenance(row)
        if provenance:
            checks.append(
                _check("roster-status", FAIL, "roster status is PASS but " + "; ".join(provenance))
            )
        else:
            checks.append(_check("roster-status", PASS, "roster status is PASS"))
    observed_target = _evidence(row, "target_id", "campaign_target_id", "target")
    if target_id is None or observed_target is None:
        checks.append(_check("campaign-target", COULD_NOT_BE_CHECKED, "campaign target identity is missing from the roster or campaign"))
    elif observed_target != target_id:
        checks.append(_check("campaign-target", FAIL, f"roster target {observed_target!r} does not match campaign target {target_id!r}"))
    else:
        checks.append(_check("campaign-target", PASS, "roster row names the campaign target"))
    observed_target_structure_sha256 = _evidence(row, "target_structure_sha256")
    if target_structure_sha256 is None or observed_target_structure_sha256 is None:
        checks.append(
            _check(
                "campaign-target-structure",
                COULD_NOT_BE_CHECKED,
                "materialized campaign target hash is missing from the roster or campaign",
            )
        )
    elif observed_target_structure_sha256 != target_structure_sha256:
        checks.append(
            _check(
                "campaign-target-structure",
                FAIL,
                "roster target_structure_sha256 does not match the materialized campaign target",
                roster=observed_target_structure_sha256,
                campaign=target_structure_sha256,
            )
        )
    else:
        checks.append(
            _check(
                "campaign-target-structure",
                PASS,
                "roster target_structure_sha256 matches the materialized campaign target",
            )
        )
    checks.extend(_same_image_and_flags(adapter, row))
    checks.append(_bool_check("container-build", _evidence(row, "container_build", "container_built"), "container build evidence is missing"))
    checks.append(_bool_check("package-imports", _evidence(row, "package_imports", "imports_ok"), "package import evidence is missing"))
    checks.append(_weights_check(adapter, row))
    checks.append(_run_check(row))
    required_count = _evidence(row, "n_designs", "design_count")
    checks.append(
        _outputs_check(
            adapter,
            row,
            required_count,
            evidence_root=evidence_root,
            require_evidence_root=require_evidence_root,
        )
    )
    checks.append(
        _sequence_check(
            row,
            evidence_root=evidence_root,
            require_evidence_root=require_evidence_root,
        )
    )
    return checks


# The three identity fields are checked for presence here, and two of them are
# compared against what a run observed. `model_revision` is compared in
# `_weights_check` and `_revision_pin_check`. `environment_identity` is compared
# in `_deployment_identity_check`. Nothing anywhere compares `source_revision`,
# so its PASS message says what it establishes. Three identically worded PASS
# lines would read as three comparisons.
#
# Whether such a comparison could be built is per adapter, and an earlier version
# of this comment said flatly that it could not be. That was drawn from
# `rfdiffusion-generator`, the one binding where both sides carry a value: the
# profile pins `RosettaCommons/RFdiffusion@2d0c003d...` and the roster records
# `rc-foundry 0.2.0+app-09dd4994...`, which are different kinds of fact about one
# run. `proteinmpnn-designer` is not like that. A live fal run on 2026-09-19
# returned `source_revision`
# `dauparas/ProteinMPNN@8907e6671bfbfc92303b5f79c4b5e6ce47cdef57+app-17c8c66751229d8e`,
# matching the roster byte for byte, and that names an upstream repository and
# commit with a deployment build appended. Splitting on `+app-` would give a
# comparable upstream half. The profile's own pin for that adapter reads
# `__REQUIRED__`, so there is nothing to compare it against today, which is why
# this stays a presence check rather than why it must.
_COMPARED_ELSEWHERE = {
    "model_revision": "and its value is compared against the run's weights record",
    "environment_identity": "and its value is compared against the environment the run reported",
    "source_revision": "which is a presence check only, because nothing here reads it "
    "against a run record",
}


def _adapter_checks(adapter: Mapping[str, Any]) -> list[Check]:
    adapter_id = str(adapter.get("adapter_id", "<missing>"))
    checks: list[Check] = []
    for field in ("source_revision", "model_revision", "environment_identity"):
        value = adapter.get(field)
        if not _nonempty(value):
            checks.append(_check(f"identity.{field}", FAIL, f"adapter {adapter_id} has no resolved {field}"))
        elif field == "environment_identity" and str(value).startswith("modal-env:") and not MODAL_ENV_RE.fullmatch(str(value)):
            checks.append(_check(f"identity.{field}", FAIL, f"adapter {adapter_id} has an invalid modal environment identity"))
        else:
            checks.append(
                _check(
                    f"identity.{field}",
                    PASS,
                    f"adapter {adapter_id} has a resolved {field}, {_COMPARED_ELSEWHERE[field]}",
                )
            )
    resources = adapter.get("resources")
    if not isinstance(resources, Mapping):
        checks.append(_check("resources", FAIL, f"adapter {adapter_id} resources are missing"))
    else:
        resource_errors = []
        for field in ("cpu", "gpu", "memory_gb"):
            value = resources.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                resource_errors.append(field)
        checks.append(_check("resources", FAIL if resource_errors else PASS, "resource request is resolved" if not resource_errors else f"invalid resource fields: {resource_errors}"))
    for field in ("toolcheck_argv", "command_argv_template", "parser_argv_template"):
        value = adapter.get(field)
        if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
            checks.append(_check(f"argv.{field}", FAIL, f"adapter {adapter_id} has no string argv list"))
        elif any(_placeholder(item) for item in value):
            checks.append(_check(f"argv.{field}", FAIL, f"adapter {adapter_id} has unresolved argv placeholders"))
        elif any("fixture_adapter" in item for item in value):
            checks.append(_check(f"argv.{field}", FAIL, f"adapter {adapter_id} is bound to fixture_adapter"))
        else:
            checks.append(_check(f"argv.{field}", PASS, f"adapter {adapter_id} argv is resolved"))
    return checks


def _renderer_runtime_check(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> Check:
    """Check the renderer selected for the terminal render stage."""
    stages = [
        stage
        for stage in plan.get("stages", [])
        if isinstance(stage, Mapping) and stage.get("stage_id") == "render-viewer"
    ]
    adapters = {
        str(adapter.get("adapter_id")): adapter
        for adapter in plan.get("adapters", [])
        if isinstance(adapter, Mapping) and isinstance(adapter.get("adapter_id"), str)
    }
    if not stages:
        return _check(
            "renderer",
            PASS,
            "plan has no render-viewer stage; renderer check is not applicable",
        )
    if len(stages) != 1:
        return _check("renderer", FAIL, "run plan has no unique render-viewer stage")
    selected_adapter = adapters.get(str(stages[0].get("adapter_id")))
    if selected_adapter is None:
        return _check("renderer", FAIL, "render-viewer names an unknown renderer adapter")
    selected = None
    selection = config.get("renderer_selection")
    if isinstance(selection, Mapping) and isinstance(selection.get("selected"), str):
        selected = selection["selected"]
    selected = selected or lane.viewer_renderer_kind(selected_adapter)
    candidates = {
        lane.viewer_renderer_kind(adapter): adapter
        for adapter in adapters.values()
        if adapter.get("role") == "renderer" and lane.viewer_renderer_kind(adapter) is not None
    }
    if selected not in candidates:
        if selected == "custom":
            return _check(
                "renderer",
                PASS,
                "profile supplies a custom renderer; its stage toolcheck will validate it",
                selected=selected,
                adapter_id=stages[0].get("adapter_id"),
            )
        return _check("renderer", FAIL, "configured renderer has no matching adapter")
    available, detail = lane.renderer_availability(selected, candidates[selected])
    if available:
        return _check(
            "renderer",
            PASS,
            f"configured renderer {selected!r} is available",
            selected=selected,
            adapter_id=stages[0].get("adapter_id"),
        )
    alternatives = []
    for kind, adapter in candidates.items():
        if kind == selected:
            continue
        alternative_available, _alternative_detail = lane.renderer_availability(kind, adapter)
        if alternative_available:
            alternatives.append(f"{kind}-renderer")
    alternative = alternatives[0] if alternatives else "none"
    reason = (
        f'configured renderer "{selected}" is unavailable; '
        f"working alternative: {alternative}."
    )
    return _check(
        "renderer",
        FAIL,
        reason,
        selected=selected,
        adapter_id=stages[0].get("adapter_id"),
        detail=detail,
    )


def _backbone_route_check(
    config: Mapping[str, Any], plan: Mapping[str, Any]
) -> Check:
    """Refuse a sequence-design stage fed by a generator that writes too few atoms.

    The generator preflight above runs only for a campaign that names
    `rfdiffusion-generator`, so a Genie3 campaign would otherwise reach the
    spend gate unchecked. This check runs for every campaign.
    """
    try:
        problems = backbone_shape.route_problems(config, plan)
    except (OSError, ValueError) as exc:
        return _check(
            "backbone-atom-route",
            COULD_NOT_BE_CHECKED,
            f"the packaged tool catalog could not be read: {type(exc).__name__}: {exc}",
        )
    if not problems:
        return _check(
            "backbone-atom-route",
            PASS,
            "no sequence designer is routed to a generator that writes too few backbone atoms",
        )
    return _check(
        "backbone-atom-route",
        FAIL,
        "; ".join(f"{item.field} {item.problem}" for item in problems),
        problems=[item.as_dict() for item in problems],
    )


def validate_runtime(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    roster_path: Path | None,
    artifact_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    checks: list[Check] = []
    checks.append(_renderer_runtime_check(config, plan))
    adapters = plan.get("adapters")
    if not isinstance(adapters, list) or any(not isinstance(item, Mapping) for item in adapters):
        checks.append(_check("plan.adapters", FAIL, "run plan has no object adapter list"))
        adapters = []
    target_id = None
    target_structure_sha256 = None
    targets = config.get("targets")
    if isinstance(targets, list):
        primary = [item for item in targets if isinstance(item, Mapping) and item.get("role") == "primary"]
        if len(primary) == 1 and isinstance(primary[0].get("target_id"), str):
            target_id = str(primary[0]["target_id"])
            structure_sha256 = primary[0].get("structure_sha256")
            if isinstance(structure_sha256, str) and SHA256_RE.fullmatch(structure_sha256):
                target_structure_sha256 = structure_sha256
    if target_id is None:
        checks.append(_check("campaign-target", COULD_NOT_BE_CHECKED, "campaign has no uniquely identified primary target"))
    if target_structure_sha256 is None:
        checks.append(
            _check(
                "campaign-target-structure",
                COULD_NOT_BE_CHECKED,
                "materialized campaign has no primary target structure_sha256",
            )
        )
    roster_document: dict[str, Any] = {}
    try:
        if roster_path is not None:
            roster_document, roster = _load_roster_document(roster_path)
        else:
            roster = None
    except ValueError as exc:
        roster = None
        checks.append(_check("model-roster", FAIL, str(exc)))
    if roster is None and not any(item.name == "model-roster" for item in checks):
        checks.append(_check("model-roster", COULD_NOT_BE_CHECKED, "model-roster ledger path was not supplied"))
    evidence_root = _configured_evidence_root(config, roster_document, config_path, roster_path)
    require_evidence_root = _roster_requires_evidence_root(roster_document)
    if require_evidence_root:
        if evidence_root is None:
            checks.append(
                _check(
                    "evidence-root",
                    COULD_NOT_BE_CHECKED,
                    "roster declares a runtime evidence root, and runtime.model_roster_evidence_root is missing",
                )
            )
        elif not evidence_root.is_dir():
            checks.append(
                _check(
                    "evidence-root",
                    FAIL,
                    f"configured roster evidence root is missing: {evidence_root}",
                )
            )
        else:
            checks.append(_check("evidence-root", PASS, "roster evidence paths resolve under the configured root"))
    separation = control_separation.assess_roster(
        config,
        roster_document,
        enforce=lane.is_production_scoring(config),
    )
    checks.append(
        _check(
            "control-separation",
            PASS if separation["ok"] else FAIL,
            "stored control separation qualifies every scoring arm and target"
            if separation["ok"]
            else "control separation is incomplete or below its configured threshold",
            statistic=separation.get("statistic"),
            measurements=separation.get("measurements", []),
            errors=separation.get("errors", []),
        )
    )
    adapter_reports: list[dict[str, Any]] = []
    for adapter in adapters:
        adapter_id = str(adapter.get("adapter_id", "<missing>"))
        adapter_checks = _adapter_checks(adapter)
        model_revision = adapter.get("model_revision")
        if model_revision != "none":
            row = _find_roster_row(roster or [], adapter) if roster is not None else None
            adapter_checks.extend(
                _model_checks(
                    adapter,
                    row,
                    target_id,
                    target_structure_sha256,
                    evidence_root=evidence_root,
                    require_evidence_root=require_evidence_root,
                )
            )
        adapter_ok = all(item.status == PASS for item in adapter_checks)
        adapter_reports.append({"adapter_id": adapter_id, "ok": adapter_ok, "checks": [item.as_dict() for item in adapter_checks]})
        checks.append(_check(f"adapter:{adapter_id}", PASS if adapter_ok else FAIL, "all adapter and roster checks passed" if adapter_ok else "one or more adapter or roster checks failed"))
    if any(str(item.get("adapter_id")) == "rfdiffusion-generator" for item in adapters):
        target_manifest = artifact_root / "inputs" / "target-manifest.json"
        preflight = generator_preflight.preflight_campaign(
            config,
            plan,
            campaign_path=config_path,
            profile_path=None,
            target_manifest_path=target_manifest if target_manifest.is_file() else None,
        )
        if preflight.ok:
            checks.append(_check("generator-preflight", PASS, "generator preflight passed"))
        else:
            checks.append(_check("generator-preflight", FAIL, "generator preflight refused the campaign", problems=[problem.as_dict() for problem in preflight.problems]))
    checks.append(_backbone_route_check(config, plan))
    ok = bool(checks) and all(item.status == PASS for item in checks)
    qualification = roster_document.get("qualification")
    qualification_report = (
        dict(qualification)
        if isinstance(qualification, Mapping)
        else {"mode": "unknown", "reason": "roster has no qualification metadata"}
    )
    evidence_root_report: dict[str, Any] = {
        "declared": require_evidence_root,
        "configured": evidence_root is not None,
    }
    if evidence_root is not None:
        evidence_root_report["path"] = str(evidence_root)
    return {
        "ok": ok,
        "stage": "runtime-check",
        "adapters": adapter_reports,
        "checks": [item.as_dict() for item in checks],
        "errors": [item.reason for item in checks if item.status != PASS],
        "evidence_root": evidence_root_report,
        "qualification": qualification_report,
        "control_separation": separation,
    }


def _parse_report(
    *,
    report_path: Path,
    result_path: Path,
    required_fields: tuple[str, ...],
    label: str,
) -> int:
    errors: list[str] = []
    parsed_count = 0
    source_hashes: list[str] = []
    if not report_path.is_file():
        errors.append(f"{label} report is missing: {report_path}")
    else:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            parsed_count = 1
            source_hashes.append(lane.sha256_file(report_path))
            if not isinstance(report, dict):
                errors.append(f"{label} report is not a JSON object")
            else:
                for field in required_fields:
                    if field not in report:
                        errors.append(f"{label} report is missing {field}")
                if report.get("ok") is not True:
                    errors.extend(str(error) for error in report.get("errors", [f"{label} report ok is not true"]))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label} report is invalid: {type(exc).__name__}: {exc}")
    result = {
        "ok": not errors,
        "parsed_count": parsed_count,
        "rejected_count": len(errors),
        "errors": errors,
        "source_output_hashes": source_hashes,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{label} parser: parsed_count={parsed_count} ok={result['ok']}")
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("toolcheck")
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
        if name == "run":
            subparser.add_argument("--roster", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "toolcheck":
        print("runtime validator ok, standard library checks only")
        return 0
    report_path = (args.attempt_dir / args.phase / "runtime-report.json").resolve()
    if args.command == "parse":
        result_path = (args.attempt_dir / args.phase / "parser-result.json").resolve()
        return _parse_report(
            report_path=report_path,
            result_path=result_path,
            required_fields=("ok", "adapters"),
            label="runtime validator",
        )
    try:
        config = _read_json(args.config, "campaign config")
        plan = _read_json(args.plan, "run plan")
        roster_path = args.roster or _configured_roster_path(config, args.config)
        report = validate_runtime(
            config,
            plan,
            roster_path=roster_path.resolve() if roster_path is not None else None,
            artifact_root=args.artifact_root.resolve(),
            config_path=args.config.resolve(),
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"runtime validator: ERROR: {exc}", file=sys.stderr)
        return 2
    print("runtime validator: PASS" if report["ok"] else "runtime validator: FAIL")
    qualification = report.get("qualification", {})
    if isinstance(qualification, Mapping):
        print(f"qualification: {qualification.get('mode', 'unknown')}")
    for check in report["checks"]:
        if check["status"] != PASS:
            print(f"- {check['name']}: {check['status']}: {check['reason']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
