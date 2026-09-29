"""Read and validate per-target production scoring gates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .arms import enabled_predictor_arms
from .refusals import ExitCode, Refusal, exit_code_for_result


SCHEMA_VERSION = 4
GATE_TYPE = "claude-binder.validation-gate"
REQUIRED_VALUE = "__REQUIRED__"
PASS_STATUS = "PASS"
FAIL_STATUS = "FAIL"
UNVERIFIABLE_STATUS = "UNVERIFIABLE"
QUALIFIED_PASS_VERDICT = "QUALIFIED_PASS"
INVALID_VERDICT = "INVALID"
NOT_MEASURED_STATUS = "NOT_MEASURED"
MEASURED_FAIL_STATUS = "MEASURED_FAIL"
VERIFIED_DOSSIER_STATUS = "VERIFIED"
UNVERIFIABLE_DOSSIER_STATUS = "UNVERIFIABLE_TARGET_DOSSIER"
LIMITATION_RESOLUTION_STATUSES = frozenset({"UNRESOLVED", "ADDRESSED", "ACCEPTED"})
ACTIVE_LIMITATION_STATUSES = frozenset({"UNRESOLVED", "ACCEPTED"})
SCORING_STAGE_ROLES = frozenset({"cofold-predictor", "interface-scorer"})
RANKING_METRIC = "ipsae_min"
SEPARATION_PREDICATE = "min_positive_strictly_exceeds_max_negative"
DEFAULT_RANKING_ARMS = ("ef2fast", "ef2full", "ptxv2")
ARM_VOCABULARY = frozenset(
    {
        "ef2fast",
        "ef2full",
        "ptxv2",
        "boltz1",
        "boltz2",
        "afm",
        "af3",
        "chai1",
    }
)
METRIC_NAMES = frozenset(
    {
        "ipsae_min",
        "sc_dockq",
        "iptm",
        "lis",
        "fnat",
        "interface_rmsd",
        "ligand_rmsd",
    }
)
_ID_RE = re.compile(r"^[a-z0-9]+(?:[a-z0-9-]*[a-z0-9])?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# A gate is evidence for a particular biological construct and epitope, not a
# reusable capability token.  ``target_id`` and the optional ``gate_id`` are
# campaign/storage labels, so neither can establish that a campaign is using
# the target that supplied the gate's control-panel evidence.  The materialized
# target has the immutable structure and residue-map digests needed below.

# This is the machine-readable contract for files in data/gates. The detailed
# checks below add cross-field rules that JSON Schema cannot express cleanly.
GATE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "schema_version",
        "gate_type",
        "target_id",
        "status",
        "frozen_instrument_mask",
        "target_dossier",
        "limitations",
        "instruments",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "gate_type": {"const": GATE_TYPE},
        "target_id": {"type": "string", "minLength": 1},
        "status": {"enum": [PASS_STATUS, FAIL_STATUS, UNVERIFIABLE_STATUS]},
        "target_dossier": {
            "type": "object",
            "required": ["status"],
        },
        "frozen_instrument_mask": {"type": "array", "items": {"type": "string"}},
        "limitations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "statement", "source", "resolution_status"],
                "properties": {
                    "id": {"type": "string", "pattern": _ID_RE.pattern},
                    "statement": {"type": "string", "minLength": 1},
                    "source": {
                        "type": "object",
                        "required": ["file", "lines"],
                        "properties": {
                            "file": {"type": "string", "minLength": 1},
                            "lines": {"type": "string", "minLength": 1},
                        },
                    },
                    "resolution_status": {
                        "enum": sorted(LIMITATION_RESOLUTION_STATUSES),
                    },
                },
            },
        },
        "instruments": {"type": "object", "minProperties": 1},
    },
}


class GateError(ValueError):
    """A gate file cannot satisfy the validation-gate contract."""


def default_gate_dir() -> Path:
    """Return the packaged directory that holds campaign gate files."""
    return Path(__file__).resolve().parent / "data" / "gates"


def _target_gate_ids(target: str | Mapping[str, Any]) -> tuple[str, str]:
    """Return the campaign target ID and its gate lookup ID."""
    if isinstance(target, str):
        target_id = target
        gate_id = target
    elif isinstance(target, Mapping):
        target_id = target.get("target_id")
        gate_id = target.get("gate_id", target_id)
    else:
        target_id = None
        gate_id = None
    if not isinstance(target_id, str) or not _ID_RE.fullmatch(target_id):
        raise GateError(f"target must be a lowercase campaign ID: {target_id!r}")
    if not isinstance(gate_id, str) or not _ID_RE.fullmatch(gate_id):
        raise GateError(f"gate_id must be a lowercase campaign ID: {gate_id!r}")
    return target_id, gate_id


def gate_file_path(target: str | Mapping[str, Any], gate_dir: str | Path | None = None) -> Path:
    """Return the gate path for a target ID or target record."""
    _, gate_id = _target_gate_ids(target)
    return (Path(gate_dir) if gate_dir is not None else default_gate_dir()) / f"{gate_id}.json"


def _required_text(value: Any, location: str) -> tuple[str | None, list[str]]:
    if not isinstance(value, str) or not value.strip():
        return None, [f"{location} must be a non-empty string"]
    return value, []


def _required_sha256(value: Any, location: str) -> tuple[str | None, list[str]]:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        return None, [f"{location} must be a lowercase SHA-256 digest"]
    return value, []


def _string_list(value: Any, location: str) -> tuple[list[str] | None, list[str]]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        return None, [f"{location} must be a non-empty string list"]
    if len(value) != len(set(value)):
        return None, [f"{location} must contain unique strings"]
    return sorted(value), []


def _canonical_entities(
    value: Any,
    location: str,
) -> tuple[list[dict[str, Any]] | None, list[str]]:
    """Normalize the target entities that determine its biological construct."""
    if not isinstance(value, list) or not value:
        return None, [f"{location} must be a non-empty list"]
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, entity in enumerate(value):
        entity_location = f"{location}[{index}]"
        if not isinstance(entity, Mapping):
            errors.append(f"{entity_location} must be an object")
            continue
        entity_id, entity_errors = _required_text(
            entity.get("entity_id"), f"{entity_location}.entity_id"
        )
        entity_type, type_errors = _required_text(
            entity.get("type"), f"{entity_location}.type"
        )
        chain_ids, chain_errors = _string_list(
            entity.get("chain_ids"), f"{entity_location}.chain_ids"
        )
        errors.extend(entity_errors)
        errors.extend(type_errors)
        errors.extend(chain_errors)
        if not isinstance(entity.get("required"), bool):
            errors.append(f"{entity_location}.required must be boolean")
        if (
            not entity_errors
            and not type_errors
            and not chain_errors
            and isinstance(entity.get("required"), bool)
        ):
            records.append(
                {
                    "entity_id": entity_id,
                    "type": entity_type,
                    "chain_ids": chain_ids,
                    "required": entity["required"],
                }
            )
    if errors:
        return None, errors
    if len({record["entity_id"] for record in records}) != len(records):
        return None, [f"{location}.entity_id values must be unique"]
    return sorted(records, key=lambda record: record["entity_id"]), []


def _canonical_site(
    value: Any,
    location: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize the site contract whose control separation was measured."""
    if not isinstance(value, Mapping):
        return None, [f"{location} must be an object"]
    errors: list[str] = []
    mode, mode_errors = _required_text(value.get("mode"), f"{location}.mode")
    design_residues, design_errors = _string_list(
        value.get("design_residues"), f"{location}.design_residues"
    )
    reference_residues, reference_errors = _string_list(
        value.get("reference_contact_residues"), f"{location}.reference_contact_residues"
    )
    atom_selection, atom_errors = _required_text(
        value.get("atom_selection"), f"{location}.atom_selection"
    )
    residue_map_sha256, map_errors = _required_sha256(
        value.get("residue_map_sha256"), f"{location}.residue_map_sha256"
    )
    errors.extend(mode_errors)
    errors.extend(design_errors)
    errors.extend(reference_errors)
    errors.extend(atom_errors)
    errors.extend(map_errors)
    cutoff = value.get("contact_cutoff_angstrom")
    if (
        not isinstance(cutoff, (int, float))
        or isinstance(cutoff, bool)
        or not math.isfinite(float(cutoff))
    ):
        errors.append(f"{location}.contact_cutoff_angstrom must be a finite number")
    resolution_digest = value.get("resolution_artifact_sha256")
    if resolution_digest is not None:
        _, resolution_errors = _required_sha256(
            resolution_digest, f"{location}.resolution_artifact_sha256"
        )
        errors.extend(resolution_errors)
    if errors:
        return None, errors
    return {
        "mode": mode,
        "design_residues": design_residues,
        "reference_contact_residues": reference_residues,
        "contact_cutoff_angstrom": float(cutoff),
        "atom_selection": atom_selection,
        "residue_map_sha256": residue_map_sha256,
        "resolution_artifact_sha256": resolution_digest,
    }, []


def _dossier_digest(dossier: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dossier,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_target_dossier(
    value: Any,
    *,
    location: str,
    structure_field: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Build the stable identity/site record used to bind one gate's evidence."""
    if not isinstance(value, Mapping):
        return None, [f"{location} must be an object"]
    errors: list[str] = []
    source_id, source_errors = _required_text(
        value.get("source_id"), f"{location}.source_id"
    )
    structure_sha256, structure_errors = _required_sha256(
        value.get(structure_field), f"{location}.{structure_field}"
    )
    errors.extend(source_errors)
    errors.extend(structure_errors)

    if structure_field == "target_structure_sha256":
        design_target_chain_ids, chain_errors = _string_list(
            value.get("design_target_chain_ids"),
            f"{location}.design_target_chain_ids",
        )
        errors.extend(chain_errors)
    else:
        chains = value.get("chains")
        if not isinstance(chains, list):
            design_target_chain_ids = None
            errors.append(f"{location}.chains must be a list")
        else:
            design_target_chain_ids, chain_errors = _string_list(
                [
                    chain.get("chain_id")
                    for chain in chains
                    if isinstance(chain, Mapping) and chain.get("role") == "design-target"
                ],
                f"{location}.chains[role=design-target].chain_id",
            )
            errors.extend(chain_errors)

    entities, entity_errors = _canonical_entities(
        value.get("entities"), f"{location}.entities"
    )
    site, site_errors = _canonical_site(value.get("site"), f"{location}.site")
    errors.extend(entity_errors)
    errors.extend(site_errors)
    if errors:
        return None, errors
    return {
        "source_id": source_id,
        "target_structure_sha256": structure_sha256,
        "design_target_chain_ids": design_target_chain_ids,
        "entities": entities,
        "site": site,
    }, []


def target_dossier(target: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical target dossier from a materialized campaign target.

    This deliberately does not use the run-local ``target_id`` or ``gate_id``.
    Callers must materialize the campaign first so the target structure and
    residue map have immutable SHA-256 digests.
    """
    dossier, errors = _canonical_target_dossier(
        target,
        location="target",
        structure_field="structure_sha256",
    )
    if errors or dossier is None:
        raise GateError("; ".join(errors))
    return {
        "status": VERIFIED_DOSSIER_STATUS,
        **dossier,
        "dossier_sha256": _dossier_digest(dossier),
    }


def _gate_target_dossier(value: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(value, Mapping):
        return None, ["target_dossier must be an object"]
    status = value.get("status")
    if status == UNVERIFIABLE_DOSSIER_STATUS:
        reason, reason_errors = _required_text(
            value.get("reason"), "target_dossier.reason"
        )
        source_errors = _source_errors(value.get("source"), "target_dossier.source")
        if reason_errors or source_errors:
            return None, reason_errors + source_errors
        return {
            "status": UNVERIFIABLE_DOSSIER_STATUS,
            "reason": reason,
            "source": dict(value["source"]),
        }, []
    if status != VERIFIED_DOSSIER_STATUS:
        return None, [
            "target_dossier.status must be VERIFIED or UNVERIFIABLE_TARGET_DOSSIER"
        ]
    dossier, errors = _canonical_target_dossier(
        value,
        location="target_dossier",
        structure_field="target_structure_sha256",
    )
    if errors or dossier is None:
        return None, errors
    declared_digest, digest_errors = _required_sha256(
        value.get("dossier_sha256") if isinstance(value, Mapping) else None,
        "target_dossier.dossier_sha256",
    )
    if digest_errors:
        return None, digest_errors
    expected_digest = _dossier_digest(dossier)
    if declared_digest != expected_digest:
        return None, [
            "target_dossier.dossier_sha256 does not match its canonical target dossier"
        ]
    return {
        "status": VERIFIED_DOSSIER_STATUS,
        **dossier,
        "dossier_sha256": declared_digest,
    }, []


def _target_dossier_binding_errors(gate: Mapping[str, Any], target: Any) -> list[str]:
    """Return why a gate's measured controls cannot qualify this campaign target."""
    gate_dossier, gate_errors = _gate_target_dossier(gate.get("target_dossier"))
    if gate_errors or gate_dossier is None:
        return gate_errors
    if gate_dossier["status"] == UNVERIFIABLE_DOSSIER_STATUS:
        return [
            "gate target dossier is UNVERIFIABLE_TARGET_DOSSIER; its historical "
            "control-panel evidence cannot qualify production scoring"
        ]
    if not isinstance(target, Mapping):
        return [
            "target dossier is required to bind validation evidence; pass a materialized target record"
        ]
    try:
        observed = target_dossier(target)
    except GateError as exc:
        return [f"campaign target dossier is incomplete: {exc}"]
    if observed["dossier_sha256"] == gate_dossier["dossier_sha256"]:
        return []
    errors: list[str] = [
        "target dossier does not match the gate evidence/control-panel qualification "
        f"(gate {gate_dossier['dossier_sha256']}, campaign {observed['dossier_sha256']})"
    ]
    for key in (
        "source_id",
        "target_structure_sha256",
        "design_target_chain_ids",
        "entities",
        "site",
    ):
        if observed[key] != gate_dossier[key]:
            errors.append(f"target dossier mismatch: {key}")
    return errors


def ranking_instruments(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the frozen ranking arm names named by a resolved config."""
    scoring = config.get("scoring")
    if not isinstance(scoring, Mapping):
        return ()
    for key in ("ranking_arms", "ranking_instruments"):
        configured = scoring.get(key)
        if isinstance(configured, list) and all(
            isinstance(item, str) and item for item in configured
        ):
            return tuple(dict.fromkeys(configured))
    # A campaign that does not name its arms ranks on the ones it enabled. Falling
    # straight through to the published three refused every single-arm campaign at
    # the scoring stage, because no small run enables ef2full or ptxv2, and the
    # refusal named three arms the user had never configured.
    #
    # The default survives for a config that names no predictors at all, where the
    # published ensemble is the only defensible reading.
    enabled = enabled_predictor_arms(config)
    if enabled:
        return enabled
    return DEFAULT_RANKING_ARMS


def scaffold_gate(config: Mapping[str, Any], target_id: str) -> dict[str, Any]:
    """Build a valid, non-passing schema-v4 gate for one materialized target.

    The scaffold binds immutable target evidence immediately, but leaves every
    experimental measurement visibly required. It therefore validates as FAIL
    and cannot accidentally authorize scoring before controls are recorded.
    """
    targets = config.get("targets")
    if not isinstance(targets, list):
        raise GateError("materialized config must contain a targets list")
    matches = [
        target
        for target in targets
        if isinstance(target, Mapping) and target.get("target_id") == target_id
    ]
    if len(matches) != 1:
        raise GateError(
            f"materialized config must contain exactly one target {target_id!r}"
        )
    target = matches[0]
    _, gate_id = _target_gate_ids(target)
    arms = ranking_instruments(config)
    if not arms:
        raise GateError("materialized config resolves no ranking instruments")
    unsupported = sorted(set(arms) - ARM_VOCABULARY)
    if unsupported:
        raise GateError(
            "gate-v4 has no published arm vocabulary for " + ", ".join(unsupported)
        )

    instruments: dict[str, Any] = {}
    for arm in arms:
        instruments[arm] = {
            "arm_name": arm,
            "status": FAIL_STATUS,
            "fold_recapitulation": {
                "pass": False,
                "measurement_status": "NOT_MEASURED",
                "reason": "Record target-fold recapitulation evidence for this arm.",
                "source": {"file": REQUIRED_VALUE, "lines": REQUIRED_VALUE},
            },
            "positive_control_separation": {
                "pass": False,
                "ranking_metric": RANKING_METRIC,
                "separation_predicate": SEPARATION_PREDICATE,
                "reason": "Record positive and negative control scores for this arm.",
                "positive_control": {
                    "id": REQUIRED_VALUE,
                    "source_id": REQUIRED_VALUE,
                    "stoichiometry": REQUIRED_VALUE,
                    "score_by_seed": REQUIRED_VALUE,
                    "score_range": REQUIRED_VALUE,
                },
                "negative_controls": [
                    {
                        "id": REQUIRED_VALUE,
                        "score_by_seed": REQUIRED_VALUE,
                        "score_range": REQUIRED_VALUE,
                    }
                ],
                "source": {"file": REQUIRED_VALUE, "lines": REQUIRED_VALUE},
            },
        }
    gate = {
        "schema_version": SCHEMA_VERSION,
        "gate_type": GATE_TYPE,
        "target_id": gate_id,
        "status": FAIL_STATUS,
        "frozen_instrument_mask": list(arms),
        "target_dossier": target_dossier(target),
        "limitations": [],
        "instruments": instruments,
    }
    validation = validate_gate(
        gate,
        target=gate_id,
        ranking_instruments=arms,
    )
    if validation["errors"]:
        raise GateError("generated gate scaffold is invalid: " + "; ".join(validation["errors"]))
    return gate


def _contains_required(value: Any) -> bool:
    if value == REQUIRED_VALUE:
        return True
    if isinstance(value, Mapping):
        return any(_contains_required(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_required(item) for item in value)
    return False


def _source_errors(value: Any, location: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{location} must name a source object"]
    errors: list[str] = []
    for key in ("file", "lines"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            errors.append(f"{location}.{key} must be a non-empty string")
    return errors


def _limitation_records(
    value: Any,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return limitation contract errors and the records that qualify a verdict."""
    if not isinstance(value, list):
        return ["limitations must be an array"], [], [], []
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, limitation in enumerate(value):
        location = f"limitations[{index}]"
        if not isinstance(limitation, Mapping):
            errors.append(f"{location} must be an object")
            continue
        record = dict(limitation)
        records.append(record)
        limitation_id = limitation.get("id")
        if not isinstance(limitation_id, str) or not _ID_RE.fullmatch(limitation_id):
            errors.append(f"{location}.id must be a lowercase limitation ID")
        elif limitation_id in seen_ids:
            errors.append(f"{location}.id must be unique")
        else:
            seen_ids.add(limitation_id)
        if not isinstance(limitation.get("statement"), str) or not limitation["statement"].strip():
            errors.append(f"{location}.statement must be a non-empty string")
        errors.extend(_source_errors(limitation.get("source"), f"{location}.source"))
        resolution_status = limitation.get("resolution_status")
        if resolution_status not in LIMITATION_RESOLUTION_STATUSES:
            errors.append(
                f"{location}.resolution_status must be one of "
                f"{sorted(LIMITATION_RESOLUTION_STATUSES)}"
            )
        if resolution_status == "UNRESOLVED":
            unresolved.append(record)
        if resolution_status in ACTIVE_LIMITATION_STATUSES:
            active.append(record)
    return errors, records, unresolved, active


def _numeric_or_required(value: Any, location: str) -> list[str]:
    if value == REQUIRED_VALUE:
        return []
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return [f"{location} must be numeric or {REQUIRED_VALUE}"]
    return []


def _score_record_errors(value: Any, location: str) -> list[str]:
    if value == REQUIRED_VALUE:
        return []
    if not isinstance(value, list) or not value:
        return [f"{location} must be a non-empty list or {REQUIRED_VALUE}"]
    errors: list[str] = []
    observation_key = "sample" if "score_by_sample" in location else "seed"
    for index, record in enumerate(value):
        record_location = f"{location}[{index}]"
        if not isinstance(record, Mapping):
            errors.append(f"{record_location} must be an object")
            continue
        observation = record.get(observation_key)
        if not isinstance(observation, int) or isinstance(observation, bool):
            errors.append(f"{record_location}.{observation_key} must be an integer")
        errors.extend(_numeric_or_required(record.get("value"), f"{record_location}.value"))
    return errors


def _score_series(record: Mapping[str, Any], location: str) -> tuple[Any, str]:
    for field in ("score_by_seed", "score_by_sample"):
        if field in record:
            return record.get(field), f"{location}.{field}"
    return None, f"{location}.score_by_seed"


def _condition_a_errors(value: Any, location: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{location} is required"]
    errors: list[str] = []
    if not isinstance(value.get("pass"), bool):
        errors.append(f"{location}.pass must be boolean")
    measurement_status = value.get("measurement_status", "MEASURED")
    if measurement_status not in {"MEASURED", "NOT_MEASURED"}:
        errors.append(f"{location}.measurement_status must be MEASURED or NOT_MEASURED")
    if measurement_status == "NOT_MEASURED":
        if value.get("pass") is not False:
            errors.append(f"{location}.pass must be false when condition (a) is not measured")
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            errors.append(f"{location}.reason must be a non-empty string")
        errors.extend(_source_errors(value.get("source"), f"{location}.source"))
        return errors
    for key in (
        "ca_rmsd_threshold_angstrom",
        "observed_max_ca_rmsd_angstrom",
        "minimum_aligned_target_residues_threshold",
        "observed_minimum_aligned_target_residues",
    ):
        errors.extend(_numeric_or_required(value.get(key), f"{location}.{key}"))
    if not isinstance(value.get("core_scoping"), str) or not value["core_scoping"].strip():
        errors.append(f"{location}.core_scoping must be a non-empty string")
    errors.extend(_source_errors(value.get("source"), f"{location}.source"))
    if value.get("pass") is False and (
        not isinstance(value.get("reason"), str) or not value["reason"].strip()
    ):
        errors.append(f"{location}.reason must be a non-empty string when pass is false")
    if value.get("pass") is True and _contains_required(value):
        errors.append(f"{location} cannot pass with a required value")
    return errors


def _score_values(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    values: list[float] = []
    for record in value:
        if not isinstance(record, Mapping):
            return None
        score = record.get("value")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            return None
        values.append(float(score))
    return values


def _range_matches(values: list[float], score_range: Any) -> bool:
    if not isinstance(score_range, list) or len(score_range) != 2:
        return False
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in score_range):
        return False
    return math.isclose(float(score_range[0]), min(values), rel_tol=1e-9, abs_tol=1e-9) and math.isclose(
        float(score_range[1]), max(values), rel_tol=1e-9, abs_tol=1e-9
    )


def _shadow_metric_errors(value: Any, location: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        return [f"{location} must be an object"]
    errors: list[str] = []
    for metric, evidence in value.items():
        metric_location = f"{location}.{metric}"
        if not isinstance(metric, str) or not metric:
            errors.append(f"{location} metric names must be non-empty strings")
            continue
        if not isinstance(evidence, Mapping):
            errors.append(f"{metric_location} must be an object")
            continue
        positive = evidence.get("positive_control")
        if positive is not None:
            if not isinstance(positive, Mapping):
                errors.append(f"{metric_location}.positive_control must be an object")
            else:
                positive_scores, positive_location = _score_series(positive, f"{metric_location}.positive_control")
                errors.extend(
                    _score_record_errors(
                        positive_scores,
                        positive_location,
                    )
                )
                if "score_range" in positive:
                    score_range = positive.get("score_range")
                    if score_range != REQUIRED_VALUE:
                        if not isinstance(score_range, list) or len(score_range) != 2:
                            errors.append(
                                f"{metric_location}.positive_control.score_range must contain two values or {REQUIRED_VALUE}"
                            )
                        else:
                            for index, item in enumerate(score_range):
                                errors.extend(
                                    _numeric_or_required(
                                        item,
                                        f"{metric_location}.positive_control.score_range[{index}]",
                                    )
                                )
        negatives = evidence.get("negative_controls")
        if negatives is None:
            continue
        if not isinstance(negatives, list) or not negatives:
            errors.append(f"{metric_location}.negative_controls must be a non-empty list")
            continue
        for index, negative in enumerate(negatives):
            negative_location = f"{metric_location}.negative_controls[{index}]"
            if not isinstance(negative, Mapping):
                errors.append(f"{negative_location} must be an object")
                continue
            if not isinstance(negative.get("id"), str) or not negative["id"].strip():
                errors.append(f"{negative_location}.id must be a non-empty string")
            availability = negative.get("availability")
            if availability == "UNDEFINED":
                if not isinstance(negative.get("reason"), str) or not negative["reason"].strip():
                    errors.append(f"{negative_location}.reason must be a non-empty string")
            elif availability == "MEASURED":
                negative_scores, negative_location = _score_series(negative, negative_location)
                errors.extend(
                    _score_record_errors(
                        negative_scores,
                        negative_location,
                    )
                )
            else:
                errors.append(f"{negative_location}.availability must be MEASURED or UNDEFINED")
    return errors


def _condition_b_errors(value: Any, location: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{location} must be an object when a literature control exists"]
    errors: list[str] = []
    if not isinstance(value.get("pass"), bool):
        errors.append(f"{location}.pass must be boolean")
    if value.get("ranking_metric") != RANKING_METRIC:
        errors.append(f"{location}.ranking_metric must be {RANKING_METRIC}")
    if value.get("separation_predicate") != SEPARATION_PREDICATE:
        errors.append(f"{location}.separation_predicate must be {SEPARATION_PREDICATE}")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        errors.append(f"{location}.reason must be a non-empty string")
    control = value.get("positive_control")
    positive_values: list[float] | None = None
    if not isinstance(control, Mapping):
        errors.append(f"{location}.positive_control must be an object")
    else:
        for key in ("id", "source_id", "stoichiometry"):
            if not isinstance(control.get(key), str) or not control[key].strip():
                errors.append(f"{location}.positive_control.{key} must be a non-empty string")
        positive_scores, positive_location = _score_series(control, f"{location}.positive_control")
        errors.extend(_score_record_errors(positive_scores, positive_location))
        score_range = control.get("score_range")
        if score_range != REQUIRED_VALUE:
            if not isinstance(score_range, list) or len(score_range) != 2:
                errors.append(
                    f"{location}.positive_control.score_range must contain two values or {REQUIRED_VALUE}"
                )
            else:
                for index, item in enumerate(score_range):
                    errors.extend(
                        _numeric_or_required(item, f"{location}.positive_control.score_range[{index}]")
                    )
        positive_values = _score_values(positive_scores)
        if positive_values is not None and score_range != REQUIRED_VALUE and not _range_matches(
            positive_values, score_range
        ):
            errors.append(f"{location}.positive_control.score_range must match score_by_seed")
    negatives = value.get("negative_controls")
    negative_values: list[float] = []
    if not isinstance(negatives, list) or not negatives:
        errors.append(f"{location}.negative_controls must be a non-empty list")
    else:
        for index, negative in enumerate(negatives):
            negative_location = f"{location}.negative_controls[{index}]"
            if not isinstance(negative, Mapping):
                errors.append(f"{negative_location} must be an object")
                continue
            if not isinstance(negative.get("id"), str) or not negative["id"].strip():
                errors.append(f"{negative_location}.id must be a non-empty string")
            negative_scores, negative_location_with_field = _score_series(negative, negative_location)
            errors.extend(_score_record_errors(negative_scores, negative_location_with_field))
            values = _score_values(negative_scores)
            if values is not None:
                negative_values.extend(values)
            score_range = negative.get("score_range")
            if score_range != REQUIRED_VALUE:
                if not isinstance(score_range, list) or len(score_range) != 2:
                    errors.append(
                        f"{negative_location}.score_range must contain two values or {REQUIRED_VALUE}"
                    )
                else:
                    for range_index, item in enumerate(score_range):
                        errors.extend(
                            _numeric_or_required(
                                item,
                                f"{negative_location}.score_range[{range_index}]",
                            )
                        )
                if values is not None and not _range_matches(values, score_range):
                    errors.append(f"{negative_location}.score_range must match score_by_seed")
    errors.extend(_source_errors(value.get("source"), f"{location}.source"))
    errors.extend(_shadow_metric_errors(value.get("evidence_metrics"), f"{location}.evidence_metrics"))
    separated = (
        positive_values is not None
        and bool(negative_values)
        and min(positive_values) > max(negative_values)
    )
    if isinstance(value.get("pass"), bool) and value.get("pass") != separated:
        errors.append(
            f"{location}.pass must equal min positive > max negative ({str(separated).lower()})"
        )
    if "separation_margin" in value:
        margin = value.get("separation_margin")
        if not isinstance(margin, (int, float)) or isinstance(margin, bool):
            errors.append(f"{location}.separation_margin must be numeric")
        elif positive_values is not None and negative_values:
            expected_margin = min(positive_values) - max(negative_values)
            if not math.isclose(float(margin), expected_margin, rel_tol=1e-9, abs_tol=1e-9):
                errors.append(f"{location}.separation_margin must match measured scores")
    if value.get("pass") is True and _contains_required(value):
        errors.append(f"{location} cannot pass with a required value")
    return errors


def _no_control_path_errors(value: Any, location: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{location} is required when condition (b) is omitted"]
    errors: list[str] = []
    if value.get("status") != "NO_CONTROL_FOUND":
        errors.append(f"{location}.status must be NO_CONTROL_FOUND")
    if value.get("comprehensive_search") is not True:
        errors.append(f"{location}.comprehensive_search must be true")
    if not isinstance(value.get("search_scope"), str) or not value["search_scope"].strip():
        errors.append(f"{location}.search_scope must be a non-empty string")
    errors.extend(_source_errors(value.get("source"), f"{location}.source"))
    return errors


def validate_gate(
    gate: Any,
    *,
    target: str | None = None,
    ranking_instruments: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return contract errors, limitations, and the effective gate verdict."""
    if not isinstance(gate, Mapping):
        return {
            "errors": ["gate must be an object"],
            "limitations": [],
            "unresolved_limitations": [],
            "active_limitations": [],
            "instrument_results": [],
            "verdict": INVALID_VERDICT,
        }
    errors: list[str] = []
    if gate.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if gate.get("gate_type") != GATE_TYPE:
        errors.append(f"gate_type must be {GATE_TYPE}")
    target_id = gate.get("target_id")
    if not isinstance(target_id, str) or not _ID_RE.fullmatch(target_id):
        errors.append("target_id must be a lowercase campaign ID")
    elif target is not None and target_id != target:
        errors.append(f"target_id {target_id} does not match requested target {target}")
    if gate.get("status") not in {PASS_STATUS, FAIL_STATUS, UNVERIFIABLE_STATUS}:
        errors.append("status must be PASS, FAIL, or UNVERIFIABLE")

    # Validate the dossier before inspecting the measured controls.  Every
    # instrument's control-panel result inherits this root-level binding.
    gate_dossier, dossier_errors = _gate_target_dossier(gate.get("target_dossier"))
    errors.extend(dossier_errors)

    limitation_errors, limitations, unresolved_limitations, active_limitations = (
        _limitation_records(gate.get("limitations"))
    )
    errors.extend(limitation_errors)

    mask = gate.get("frozen_instrument_mask")
    if not isinstance(mask, list) or not mask or any(not isinstance(item, str) or not item for item in mask):
        errors.append("frozen_instrument_mask must be a non-empty string list")
        mask = []
    elif len(mask) != len(set(mask)):
        errors.append("frozen_instrument_mask must contain unique instruments")

    instruments = gate.get("instruments")
    if not isinstance(instruments, Mapping) or not instruments:
        errors.append("instruments must be a non-empty object")
        instruments = {}
    if any(item in METRIC_NAMES for item in mask):
        errors.append("frozen_instrument_mask must use arm names")
    if any(item in METRIC_NAMES for item in instruments):
        errors.append("instruments keys must use arm names")
    expected = tuple(ranking_instruments or ())
    if expected:
        # Ranking on fewer arms than the gate validated is safe, because each arm's
        # controls were measured on its own and the gate records a status per arm.
        # Ranking on an arm the gate never validated is not. So the rule is
        # containment rather than equality, which also lets one gate serve a full
        # ensemble and a single-arm run of the same target.
        unvalidated = [arm for arm in expected if arm not in mask]
        if unvalidated:
            errors.append(
                "frozen_instrument_mask does not cover configured ranking instruments "
                f"{unvalidated}; the gate froze {list(mask)}"
            )
        if not set(expected).issubset(instruments):
            errors.append("instruments must contain every configured ranking instrument")
    elif not set(mask).issubset(instruments):
        errors.append("instruments must contain every frozen instrument")

    condition_results: list[bool] = []
    instrument_results: list[dict[str, Any]] = []
    for instrument in mask:
        record = instruments.get(instrument)
        location = f"instruments.{instrument}"
        if not isinstance(record, Mapping):
            errors.append(f"{location} must be an object")
            condition_results.append(False)
            instrument_results.append(
                {
                    "arm_name": instrument,
                    "stored_status": None,
                    "qualification_status": INVALID_VERDICT,
                    "fold_recapitulation_pass": False,
                    "positive_control_separation_pass": False,
                    "unmeasured_conditions": [],
                }
            )
            continue
        if instrument not in ARM_VOCABULARY:
            errors.append(f"{location} is not a published arm name")
        if record.get("arm_name") != instrument:
            errors.append(f"{location}.arm_name must equal {instrument}")
        arm_status = record.get("status")
        if arm_status not in {PASS_STATUS, FAIL_STATUS}:
            errors.append(f"{location}.status must be PASS or FAIL")
        condition_a = record.get("fold_recapitulation")
        condition_a_errors = _condition_a_errors(condition_a, f"{location}.fold_recapitulation")
        errors.extend(condition_a_errors)
        condition_a_passes = isinstance(condition_a, Mapping) and condition_a.get("pass") is True

        condition_b = record.get("positive_control_separation")
        if condition_b is None:
            condition_b_errors = _no_control_path_errors(
                record.get("literature_control_search"),
                f"{location}.literature_control_search",
            )
            errors.extend(condition_b_errors)
            # A documented search result preserves the reason for refusal. It cannot replace
            # the measured positive-control separation that condition (b) requires.
            condition_b_passes = False
        else:
            condition_b_errors = _condition_b_errors(condition_b, f"{location}.positive_control_separation")
            errors.extend(condition_b_errors)
            condition_b_passes = isinstance(condition_b, Mapping) and condition_b.get("pass") is True
        arm_passes = condition_a_passes and condition_b_passes and not condition_a_errors and not condition_b_errors
        if arm_status in {PASS_STATUS, FAIL_STATUS} and arm_status == PASS_STATUS and not arm_passes:
            errors.append(f"{location}.status PASS requires both conditions to pass")
        if arm_status in {PASS_STATUS, FAIL_STATUS} and arm_status == FAIL_STATUS and arm_passes:
            errors.append(f"{location}.status FAIL requires a failed condition")
        condition_results.append(arm_passes)
        unmeasured_conditions: list[str] = []
        if (
            isinstance(condition_a, Mapping)
            and condition_a.get("measurement_status") == NOT_MEASURED_STATUS
        ):
            unmeasured_conditions.append("fold_recapitulation")
        if condition_b is None or (
            isinstance(condition_b, Mapping)
            and condition_b.get("measurement_status") == NOT_MEASURED_STATUS
        ):
            unmeasured_conditions.append("positive_control_separation")
        qualification_status = (
            PASS_STATUS
            if arm_passes
            else NOT_MEASURED_STATUS
            if unmeasured_conditions
            else MEASURED_FAIL_STATUS
        )
        instrument_results.append(
            {
                "arm_name": instrument,
                "stored_status": arm_status,
                "qualification_status": qualification_status,
                "fold_recapitulation_pass": condition_a_passes,
                "positive_control_separation_pass": condition_b_passes,
                "unmeasured_conditions": unmeasured_conditions,
            }
        )

    # Historical gates can carry informational arm records outside the frozen
    # authorization mask. Surface their evidence state for planning without
    # allowing them to contribute to the verdict or authorize scoring.
    interpreted_arms = {item["arm_name"] for item in instrument_results}
    for instrument, record in instruments.items():
        if instrument in interpreted_arms or not isinstance(record, Mapping):
            continue
        condition_a = record.get("fold_recapitulation")
        condition_b = record.get("positive_control_separation")
        condition_a_passes = isinstance(condition_a, Mapping) and condition_a.get("pass") is True
        condition_b_passes = isinstance(condition_b, Mapping) and condition_b.get("pass") is True
        unmeasured_conditions: list[str] = []
        if (
            isinstance(condition_a, Mapping)
            and condition_a.get("measurement_status") == NOT_MEASURED_STATUS
        ):
            unmeasured_conditions.append("fold_recapitulation")
        if condition_b is None or (
            isinstance(condition_b, Mapping)
            and condition_b.get("measurement_status") == NOT_MEASURED_STATUS
        ):
            unmeasured_conditions.append("positive_control_separation")
        arm_passes = condition_a_passes and condition_b_passes
        instrument_results.append(
            {
                "arm_name": instrument,
                "stored_status": record.get("status"),
                "qualification_status": (
                    PASS_STATUS
                    if arm_passes
                    else NOT_MEASURED_STATUS
                    if unmeasured_conditions
                    else MEASURED_FAIL_STATUS
                ),
                "fold_recapitulation_pass": condition_a_passes,
                "positive_control_separation_pass": condition_b_passes,
                "unmeasured_conditions": unmeasured_conditions,
                "authorizes_scoring": False,
                "authorization_reason": "The arm is not in frozen_instrument_mask.",
            }
        )

    expected_status = PASS_STATUS if condition_results and all(condition_results) else FAIL_STATUS
    if gate_dossier is not None and gate_dossier.get("status") == UNVERIFIABLE_DOSSIER_STATUS:
        expected_status = UNVERIFIABLE_STATUS
    if (
        gate.get("status") in {PASS_STATUS, FAIL_STATUS, UNVERIFIABLE_STATUS}
        and gate.get("status") != expected_status
    ):
        errors.append(f"status must be {expected_status} for the arm condition results")
    errors = list(dict.fromkeys(errors))
    if errors:
        verdict = INVALID_VERDICT
    elif gate.get("status") == UNVERIFIABLE_STATUS:
        verdict = UNVERIFIABLE_STATUS
    elif gate.get("status") == FAIL_STATUS:
        verdict = FAIL_STATUS
    elif active_limitations:
        verdict = QUALIFIED_PASS_VERDICT
    else:
        verdict = PASS_STATUS
    return {
        "errors": errors,
        "limitations": limitations,
        "unresolved_limitations": unresolved_limitations,
        "active_limitations": active_limitations,
        "instrument_results": instrument_results,
        "verdict": verdict,
    }


def load_gate(path: str | Path) -> dict[str, Any]:
    """Read and structurally validate one gate file."""
    gate_path = Path(path)
    try:
        with gate_path.open("r", encoding="utf-8") as handle:
            gate = json.load(handle)
    except FileNotFoundError as exc:
        raise GateError(f"gate file is missing: {gate_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"gate file cannot be read: {gate_path}: {exc}") from exc
    validation = validate_gate(gate)
    if validation["errors"]:
        raise GateError(f"{gate_path}: {'; '.join(validation['errors'])}")
    return dict(gate)


def _gate_refusal(
    *,
    target_id: str,
    gate_path: Path,
    errors: Sequence[str],
) -> Refusal:
    """Return the operator-facing refusal for one scoring gate decision."""
    missing = any(error.startswith("gate file is missing:") for error in errors)
    expected = f"A readable PASS validation gate for target {target_id}."
    if missing:
        action = f"Create the validation gate file for target {target_id}."
    else:
        action = f"Update the validation gate file for target {target_id}."
    return Refusal(
        cause=f"The validation gate rejected scoring for target {target_id}.",
        expected=expected,
        expected_source=str(gate_path),
        found="; ".join(errors),
        found_source=str(gate_path),
        scope="The validation gate ran before scoring and started no provider command.",
        action=action,
        escalation="Send the gate correction to the scoring maintainer.",
    )


def check_gate(
    target: str | Mapping[str, Any],
    *,
    gate_path: str | Path | None = None,
    gate_dir: str | Path | None = None,
    ranking_instruments: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a file-backed gate decision with refusal details."""
    target_id, gate_id = _target_gate_ids(target)
    resolved_path = Path(gate_path) if gate_path is not None else gate_file_path(target, gate_dir)
    result: dict[str, Any] = {
        "ok": False,
        "target_id": target_id,
        "gate_id": gate_id,
        "gate_path": str(resolved_path),
        "status": None,
        "verdict": None,
        "limitations": [],
        "unresolved_limitations": [],
        "active_limitations": [],
        "instrument_results": [],
        "target_dossier_binding": None,
        "errors": [],
        "exit_code": int(ExitCode.PREFLIGHT_REFUSAL),
    }
    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            gate = json.load(handle)
    except FileNotFoundError:
        result["errors"] = [f"gate file is missing: {resolved_path}"]
        refusal = _gate_refusal(
            target_id=target_id,
            gate_path=resolved_path,
            errors=result["errors"],
        )
        result["refusal"] = refusal.as_dict()
        result["refusal_text"] = refusal.text()
        return result
    except (OSError, json.JSONDecodeError) as exc:
        result["errors"] = [f"gate file cannot be read: {resolved_path}: {exc}"]
        refusal = _gate_refusal(
            target_id=target_id,
            gate_path=resolved_path,
            errors=result["errors"],
        )
        result["refusal"] = refusal.as_dict()
        result["refusal_text"] = refusal.text()
        return result
    result["status"] = gate.get("status") if isinstance(gate, Mapping) else None
    validation = validate_gate(gate, target=gate_id, ranking_instruments=ranking_instruments)
    result["verdict"] = validation["verdict"]
    result["limitations"] = validation["limitations"]
    result["unresolved_limitations"] = validation["unresolved_limitations"]
    result["active_limitations"] = validation["active_limitations"]
    result["instrument_results"] = validation["instrument_results"]
    errors = list(validation["errors"])
    dossier_errors = (
        _target_dossier_binding_errors(gate, target)
        if isinstance(gate, Mapping)
        else ["gate must be an object"]
    )
    result["target_dossier_binding"] = {
        "ok": not dossier_errors,
        "errors": dossier_errors,
    }
    errors.extend(dossier_errors)
    if not isinstance(gate, Mapping) or gate.get("status") != PASS_STATUS:
        errors.append("gate status is not PASS")
    result["errors"] = list(dict.fromkeys(errors))
    result["gate"] = gate
    result["ok"] = not result["errors"]
    result["exit_code"] = exit_code_for_result(
        verified=result["ok"],
        refused=not result["ok"],
    )
    if not result["ok"]:
        refusal = _gate_refusal(
            target_id=target_id,
            gate_path=resolved_path,
            errors=result["errors"],
        )
        result["refusal"] = refusal.as_dict()
        result["refusal_text"] = refusal.text()
    return result


def submit_gate(
    target: str | Mapping[str, Any],
    *,
    gate_path: str | Path | None = None,
    gate_dir: str | Path | None = None,
    ranking_instruments: Sequence[str] | None = None,
) -> bool:
    """Return whether production scoring may run for one target."""
    return bool(
        check_gate(
            target,
            gate_path=gate_path,
            gate_dir=gate_dir,
            ranking_instruments=ranking_instruments,
        )["ok"]
    )


def is_scoring_stage(stage: Mapping[str, Any]) -> bool:
    """Return whether a stage produces production scoring measurements."""
    role = stage.get("required_role")
    return role in SCORING_STAGE_ROLES or stage.get("adapter_id") == "interface-scorer"
