#!/usr/bin/env python3
"""Write side of the raw prediction contract.

An arm calls write_prediction_artifacts once per prediction. This module writes the
three files the executor validates, computes every metric through binder_metrics, and
returns the row fields the arm merges into the base row it already built.

Nothing here computes a metric. binder_metrics owns the measurement, this module owns
the layout, the file writing, the hashing and the row assembly.
"""

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

from ..paths import package_file
from . import binder_metrics


class BinderContractError(RuntimeError):
    """A caller handed write_prediction_artifacts something the executor would reject.

    This is raised before any file is written, so a caller that catches it and calls
    write_failed_row leaves no partial artifact directory behind.
    """


CHAIN_ASSIGNMENT_FAILURE_CODE = "chain_assignment_failed"


class ChainAssignmentError(BinderContractError):
    """The known target and binder sequences do not identify one chain pair."""

    failure_code = CHAIN_ASSIGNMENT_FAILURE_CODE


SCHEMA_VERSION = 1
PREDICTION_ARTIFACTS_DIRNAME = "prediction-artifacts"
COMPLEX_FILENAME = "complex.cif"
PAE_FILENAME = "pae.json"
MEASUREMENT_FILENAME = "measurement-source.json"

# The executor rounds nothing on read. Keep the predictor's PAE precision so the
# seed reduction on ipsae_min does not change when close seeds are compared.

# The slug is the five fields that identify one prediction, each sanitised to the
# filesystem-safe set, joined by a hyphen. Same tuple validate_observations uses as its
# duplicate key.
_SLUG_UNSAFE = re.compile(r"[^a-zA-Z0-9_.-]")

# The campaign phase. It goes on the row, into the measurement, and into the slug.
PHASES = ("screen", "intermediate", "optimization", "uniform-rescore")

# The executor's phase. It is the adapter's own --phase argument and it names the
# directory under the attempt directory. Claude Binder builds it as
# ["single"] for a single-mode stage and ["smoke", "scale"] for every other one.
RUN_PHASES = ("smoke", "scale", "single")

# Copied from the executor measurement-source contract.
# The adapters ship to a container that does not run the executor, so the tuple lives in
# both places. executor_field_drift() below compares them when the sibling file is
# reachable, which is what a toolcheck should call.
MEASUREMENT_SOURCE_FIELDS = (
    "target_id",
    "target_sha256",
    "candidate_id",
    "predictor",
    "model_revision",
    "seed",
    "phase",
    "sequence_sha256",
    "design_pose_sha256",
    "predicted_complex_sha256",
    "pae_sha256",
    "target_chain_id",
    "binder_chain_id",
    "chain_mapping",
    "predicted_target_chain_id",
    "predicted_binder_chain_id",
    "reference_target_chain_id",
    "reference_binder_chain_id",
    "aligned_target_residue_count",
    "target_alignment_rmsd",
    "ipsae_implementation_revision",
    "ipsae_interface_cutoff_angstrom",
    "dockq_implementation_revision",
    "site_scorer_revision",
    "site_residue_map_sha256",
    "site_contact_cutoff_angstrom",
    "site_atom_selection",
    "site_metric_basis",
    "ipsae_target_to_binder",
    "ipsae_binder_to_target",
    "ipsae_min",
    "sc_dockq",
    "dockq",
    "fnat",
    "interface_rmsd",
    "ligand_rmsd",
    "mapping_status",
    "site_contact_iou",
    "target_contact_recall",
    "target_contact_precision",
    "hotspot_recovery",
    "offsite_contact_fraction",
    "iptm",
    "interface_pae",
    "interface_plddt",
    "clash_count",
    "contact_count",
    "status",
)
MEASUREMENT_SOURCE_FIELD_COUNT = 48

UNCONSTRAINED_NULL_FIELDS = frozenset(
    {
        "site_contact_iou",
        "target_contact_recall",
        "target_contact_precision",
        "hotspot_recovery",
        "offsite_contact_fraction",
    }
)

if len(MEASUREMENT_SOURCE_FIELDS) != MEASUREMENT_SOURCE_FIELD_COUNT:
    raise RuntimeError(
        "MEASUREMENT_SOURCE_FIELDS holds "
        f"{len(MEASUREMENT_SOURCE_FIELDS)} names and the executor requires "
        f"{MEASUREMENT_SOURCE_FIELD_COUNT}"
    )

# The six fields a scored row carries and a failed row must not.
ARTIFACT_ROW_FIELDS = (
    "predicted_complex_path",
    "predicted_complex_sha256",
    "pae_path",
    "pae_sha256",
    "metric_source_path",
    "metric_source_sha256",
)

# Four measurement fields that no binder_metrics function returns. They come from the
# arm through extra, and the value beside each name is what a caller sees when one is
# missing.
REQUIRED_EXTRA_FIELDS = {
    "target_sha256": "the materialized target structure hash, already on the arm's base row",
    "sequence_sha256": "the candidate sequence hash, from the candidate manifest row",
    "design_pose_sha256": "the design pose hash, from the candidate manifest row",
    "iptm": (
        "the predictor's own interface pTM. Both ESMFold2 arms read prediction.iptm. "
        "Protenix v2 reads it from the sample's summary confidence file."
    ),
}

# TODO The failure_code vocabulary. report_contract.md section 9 leaves it open. The
# executor constrains failure_code only to a non-empty string, so this module normalises
# the shape and does not fix the set of tokens. The real vocabulary comes from the codes
# configured predictor modes actually raise on a run.
_FAILURE_CODE_UNSAFE = re.compile(r"[^a-z0-9_]+")
_FAILURE_CODE_FALLBACK = "unspecified_failure"
_FAILURE_REASON_LIMIT = 4000


def artifact_slug(target_id, candidate_id, predictor, phase, seed) -> str:
    """Return the directory name for one prediction.

    Five identifying fields joined by a hyphen, each sanitised by replacing every
    character outside [a-zA-Z0-9_.-] with an underscore.
    """
    return "-".join(
        _SLUG_UNSAFE.sub("_", str(value))
        for value in (target_id, candidate_id, predictor, phase, seed)
    )


def write_prediction_artifacts(
    attempt_dir,
    phase,
    target_id,
    candidate_id,
    predictor,
    seed,
    complex_cif,
    pae,
    chain_mapping,
    reference_cif,
    site_residue_map,
    model_revision,
    target_sequence,
    binder_sequence,
    extra=None,
    *,
    run_phase=None,
) -> dict:
    """Write one prediction's three files and return the row fields to merge.

    Writes complex.cif, pae.json and measurement-source.json under
    {attempt_dir}/{run_phase}/prediction-artifacts/{slug}/, computes every metric
    through binder_metrics, and returns the six path and hash fields, declared and
    derived chain mappings,
    and status='scored'. The declared chain_mapping remains on the row as the
    campaign assertion. Metric calls use the independently derived mappings for
    the predicted and reference structures.

    Two different phases are in play and they are never the same word.

    phase is the campaign phase, one of screen, optimization or uniform-rescore. It is
    the row's phase field, it is in the measurement, and it is the fourth part of the
    slug.

    run_phase is the executor's phase, one of smoke, scale or single. It is the adapter's
    own --phase argument and it is the directory segment. A smoke run and a scale run
    share one attempt directory and mint the same slug, so dropping this segment has the
    scale run overwrite the smoke run's files and leave the smoke rows pointing at hashes
    that no longer match.

    complex_cif is the mmCIF text, or a path to a file holding it. The metric functions
    are handed the written complex.cif rather than the caller's copy, so every metric is
    computed from the same bytes predicted_complex_sha256 records.

    pae is the square matrix as a list of lists. Anything carrying a .tolist method is
    converted, so an arm may pass a tensor directly.

    target_sequence and binder_sequence are the known protein sequences used to
    derive the chain mappings from each structure. A residue-count fallback is
    accepted only when it gives one unambiguous pair with different lengths.

    extra merges into the measurement. It has to carry target_sha256, sequence_sha256,
    design_pose_sha256 and iptm, because no binder_metrics function returns them. It
    carries custom_metrics as well when the campaign registers any.

    Raises BinderContractError before writing anything when an argument would produce a
    row the executor rejects.
    """
    extra = dict(extra or {})
    _require_nonempty_text("target_id", target_id)
    _require_nonempty_text("candidate_id", candidate_id)
    _require_nonempty_text("predictor", predictor)
    _require_nonempty_text("model_revision", model_revision)
    if phase not in PHASES:
        raise BinderContractError(
            f"phase must be one of {', '.join(PHASES)}, and it is {phase!r}"
        )
    run_phase = _require_run_phase(run_phase)
    seed = _require_seed(seed)
    target_chain_id, binder_chain_id = _require_chain_mapping(chain_mapping)
    target_sequence = _normalise_sequence(target_sequence, "target_sequence")
    binder_sequence = _normalise_sequence(binder_sequence, "binder_sequence")
    matrix = _normalize_pae(pae)
    complex_text = _read_structure_text("complex_cif", complex_cif)
    _require_extra(extra)

    attempt_dir = Path(attempt_dir)
    slug = artifact_slug(target_id, candidate_id, predictor, phase, seed)
    record_dir = attempt_dir / run_phase / PREDICTION_ARTIFACTS_DIRNAME / slug
    existed = record_dir.is_dir()
    record_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _write_record(
            record_dir=record_dir,
            phase=phase,
            target_id=target_id,
            candidate_id=candidate_id,
            predictor=predictor,
            seed=seed,
            complex_text=complex_text,
            matrix=matrix,
            chain_mapping=chain_mapping,
            target_chain_id=target_chain_id,
            binder_chain_id=binder_chain_id,
            reference_cif=reference_cif,
            target_sequence=target_sequence,
            binder_sequence=binder_sequence,
            site_residue_map=site_residue_map,
            model_revision=model_revision,
            extra=extra,
        )
    except BaseException:
        # The contract is three files or none. A directory holding two of them has no
        # row pointing at it and reads as a complete prediction to anyone browsing the
        # tree. A directory that was already here belongs to an earlier attempt and is
        # left alone.
        if not existed:
            shutil.rmtree(record_dir, ignore_errors=True)
        raise


def _write_record(
    *,
    record_dir: Path,
    phase,
    target_id,
    candidate_id,
    predictor,
    seed,
    complex_text: str,
    matrix: list,
    chain_mapping,
    target_chain_id,
    binder_chain_id,
    reference_cif,
    target_sequence,
    binder_sequence,
    site_residue_map,
    model_revision,
    extra: dict,
) -> dict:
    complex_path = record_dir / COMPLEX_FILENAME
    complex_path.write_text(complex_text)
    predicted_complex_sha256 = _sha256_file(complex_path)
    predicted_mapping = derive_chain_mapping(
        complex_path,
        target_sequence,
        binder_sequence,
        structure_label="predicted structure",
    )
    reference_mapping = derive_chain_mapping(
        reference_cif,
        target_sequence,
        binder_sequence,
        structure_label="reference structure",
    )
    _require_mapped_chains(
        complex_path,
        predicted_mapping["target"],
        predicted_mapping["binder"],
    )

    pae_path = record_dir / PAE_FILENAME
    _write_json(
        pae_path,
        {
            "schema_version": SCHEMA_VERSION,
            "target_id": target_id,
            "candidate_id": candidate_id,
            "predictor": predictor,
            "seed": seed,
            "chain_ids": [predicted_mapping["target"], predicted_mapping["binder"]],
            "pae": matrix,
        },
    )
    pae_sha256 = _sha256_file(pae_path)

    # The metrics read the file that was just hashed. load_measurement_source requires
    # predicted_complex_sha256 inside the measurement to equal the field on the row, and
    # that equality is only meaningful when the metric saw those same bytes.
    ipsae = binder_metrics.compute_ipsae(
        matrix,
        complex_path,
        predicted_mapping["target"],
        predicted_mapping["binder"],
    )
    dockq = binder_metrics.compute_dockq(
        complex_path,
        reference_cif,
        predicted_mapping,
        reference_chain_mapping=reference_mapping,
    )
    site = binder_metrics.compute_site_metrics(
        complex_path,
        _site_map_for_target(site_residue_map, target_chain_id, predicted_mapping["target"]),
        predicted_mapping["target"],
        predicted_mapping["binder"],
        designed_cif=reference_cif,
        # The reference is a different structure from the prediction and carries
        # its own letters. The `designed_complex` hotspot source reads it, so it
        # gets the mapping derived from it, the same pair compute_dockq is given.
        designed_chain_mapping=reference_mapping,
        design_key=candidate_id,
    )

    identity = {
        "target_id": target_id,
        "candidate_id": candidate_id,
        "predictor": predictor,
        "model_revision": model_revision,
        "seed": seed,
        "phase": phase,
        "predicted_complex_sha256": predicted_complex_sha256,
        "pae_sha256": pae_sha256,
        "target_chain_id": target_chain_id,
        "binder_chain_id": binder_chain_id,
        "chain_mapping": dict(chain_mapping),
        "predicted_target_chain_id": predicted_mapping["target"],
        "predicted_binder_chain_id": predicted_mapping["binder"],
        "reference_target_chain_id": reference_mapping["target"],
        "reference_binder_chain_id": reference_mapping["binder"],
        "ipsae_implementation_revision": binder_metrics.IPSAE_IMPLEMENTATION_REVISION,
        "ipsae_interface_cutoff_angstrom": binder_metrics.IPSAE_INTERFACE_CUTOFF_ANGSTROM,
        "dockq_implementation_revision": binder_metrics.DOCKQ_IMPLEMENTATION_REVISION,
        "status": "scored",
    }
    measurement = _merge_measurement(identity, ipsae, dockq, site, extra)
    _require_complete_measurement(measurement)
    _require_mapping_status(measurement)

    metric_path = record_dir / MEASUREMENT_FILENAME
    _write_json(metric_path, {"schema_version": SCHEMA_VERSION, "measurement": measurement})

    return {
        "measurement": measurement,
        # Keep the lexical mount spelling. Provider containers may resolve the
        # mount to a provider-only canonical prefix that does not exist on the
        # host after the verified artifact return. The mount path is deliberately
        # shared with the host run root, so it survives that boundary.
        "predicted_complex_path": str(complex_path.absolute()),
        "predicted_complex_sha256": predicted_complex_sha256,
        "pae_path": str(pae_path.absolute()),
        "pae_sha256": pae_sha256,
        "metric_source_path": str(metric_path.absolute()),
        "metric_source_sha256": _sha256_file(metric_path),
        "chain_mapping": dict(chain_mapping),
        "predicted_target_chain_id": predicted_mapping["target"],
        "predicted_binder_chain_id": predicted_mapping["binder"],
        "reference_target_chain_id": reference_mapping["target"],
        "reference_binder_chain_id": reference_mapping["binder"],
        "status": "scored",
    }


def write_failed_row(
    target_id,
    candidate_id,
    predictor,
    seed,
    failure_code,
    failure_reason,
    *,
    base=None,
) -> dict:
    """Return the fields that mark one prediction failed.

    This function never raises. It runs inside the exception handler that keeps a job
    alive after one bad candidate, so every argument is coerced rather than rejected.
    A row it produces is always one the schema accepts.

    With base, it returns a complete row: a copy of base with the six artifact fields
    removed and the failure fields applied. That is the safe call, because the schema
    forbids a failed row from carrying an artifact path.

    Without base, it returns the failure fields alone, for a caller that merges them
    into a row that never held an artifact field.
    """
    row = {} if base is None else {
        key: value for key, value in dict(base).items() if key not in ARTIFACT_ROW_FIELDS
    }
    row.update(
        {
            "target_id": _coerce_text(target_id),
            "candidate_id": _coerce_text(candidate_id),
            "predictor": _coerce_text(predictor),
            "seed": _coerce_seed(seed),
            "status": "failed",
            "failure_code": _coerce_failure_code(failure_code),
            "failure_reason": _coerce_failure_reason(failure_reason, failure_code),
        }
    )
    return row


def executor_field_drift() -> list:
    """Return the field names where this module and the executor disagree.

    An empty list means the two agree. A toolcheck should call this, because the
    alternative is a paid container writing a measurement the executor rejects.
    Returns a single explanatory string when the executor module cannot be read.
    """
    executor_path = package_file("lane.py")
    source = executor_path.read_text()
    match = re.search(r"^MEASUREMENT_SOURCE_FIELDS = \((.*?)^\)$", source, re.MULTILINE | re.DOTALL)
    if match is None:
        return [f"MEASUREMENT_SOURCE_FIELDS was not found in {executor_path}"]
    theirs = set(re.findall(r'"([a-z0-9_]+)"', match.group(1)))
    ours = set(MEASUREMENT_SOURCE_FIELDS)
    return sorted(
        [f"only the executor has {name}" for name in theirs - ours]
        + [f"only binder_contract has {name}" for name in ours - theirs]
    )


def _require_nonempty_text(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise BinderContractError(f"{name} must be a non-empty string, and it is {value!r}")


def _require_run_phase(run_phase: Any) -> str:
    """Check the directory segment, which is the adapter's own --phase argument.

    RUN_PHASES is what the executor produces today. The check is that run_phase is not a
    campaign phase, because passing the row's phase here is the mistake that costs the
    most: smoke and scale then share one directory and one slug.
    """
    if run_phase is None:
        raise BinderContractError(
            "run_phase is required. It is the adapter's own --phase argument, one of "
            f"{', '.join(RUN_PHASES)}, and it names the directory under the attempt "
            "directory. The positional phase argument is the campaign phase, one of "
            f"{', '.join(PHASES)}, and it goes on the row and into the slug."
        )
    if not isinstance(run_phase, str) or not run_phase:
        raise BinderContractError(f"run_phase must be a non-empty string, and it is {run_phase!r}")
    if run_phase in PHASES:
        raise BinderContractError(
            f"run_phase is {run_phase!r}, which is a campaign phase rather than a run "
            f"phase. Pass the adapter's --phase argument, one of {', '.join(RUN_PHASES)}."
        )
    if "/" in run_phase or "\\" in run_phase or run_phase in {".", ".."}:
        raise BinderContractError(f"run_phase must be one path segment, and it is {run_phase!r}")
    return run_phase


def _require_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BinderContractError(f"seed must be an integer, and it is {seed!r}")
    return seed


def _require_chain_mapping(chain_mapping: Any):
    if not isinstance(chain_mapping, dict) or set(chain_mapping) != {"target", "binder"}:
        raise BinderContractError(
            f"chain_mapping must hold exactly target and binder, and it is {chain_mapping!r}"
        )
    target_chain_id = chain_mapping["target"]
    binder_chain_id = chain_mapping["binder"]
    for name, value in (("target", target_chain_id), ("binder", binder_chain_id)):
        if not isinstance(value, str) or not value:
            raise BinderContractError(
                f"chain_mapping[{name!r}] must be a non-empty string, and it is {value!r}"
            )
    if target_chain_id == binder_chain_id:
        raise BinderContractError(
            f"the target and binder chains must differ, and both are {target_chain_id!r}"
        )
    return target_chain_id, binder_chain_id


_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "UNK": "X",
}


def _normalise_sequence(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ChainAssignmentError(
            f"{CHAIN_ASSIGNMENT_FAILURE_CODE}: {name} must be a protein sequence"
        )
    sequence = "".join(value.split()).upper()
    if not sequence or re.fullmatch(r"[A-Z]+", sequence) is None:
        raise ChainAssignmentError(
            f"{CHAIN_ASSIGNMENT_FAILURE_CODE}: {name} is empty or contains a non-letter"
        )
    return sequence


def _relabel_residue(residue: Any, source_chain: str, target_chain: str) -> Any:
    text = str(residue)
    prefix, separator, suffix = text.partition(":")
    if separator and prefix == source_chain:
        return f"{target_chain}:{suffix}"
    return residue


def _site_map_for_target(
    site_residue_map: dict[str, Any], source_chain: str, target_chain: str
) -> dict[str, Any]:
    """Relabel configured site labels for the predicted target chain.

    The residue-map hash and all provenance values remain those of the target
    contract. Only labels consumed by the predicted-structure site calculation
    are copied onto the chain that sequence derivation selected.
    """
    if source_chain == target_chain:
        return site_residue_map
    mapping = dict(site_residue_map)
    for key in ("site_residues", "hotspot_residues"):
        values = mapping.get(key)
        if isinstance(values, (list, tuple)):
            mapping[key] = [
                _relabel_residue(value, source_chain, target_chain) for value in values
            ]
    translation = mapping.get("source_to_cleaned")
    if isinstance(translation, dict):
        mapping["source_to_cleaned"] = {
            _relabel_residue(key, source_chain, target_chain): _relabel_residue(
                value, source_chain, target_chain
            )
            for key, value in translation.items()
        }
    return mapping


def _chain_sequences(source: Any, *, structure_label: str) -> dict[str, str]:
    try:
        structure = binder_metrics.parse_structure_atoms(source, argument=structure_label)
    except Exception as exc:  # noqa: BLE001
        raise ChainAssignmentError(
            f"{CHAIN_ASSIGNMENT_FAILURE_CODE}: could not parse {structure_label}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    sequences: dict[str, list[str]] = {}
    for residue in structure.residues:
        chain = residue.auth_chain or residue.label_chain
        if not chain:
            continue
        code = _THREE_TO_ONE.get(str(residue.comp_id).upper(), "X")
        sequences.setdefault(chain, []).append(code)
    return {chain: "".join(codes) for chain, codes in sequences.items()}


def _pdb_seqres_sequences(path: Path) -> dict[str, str]:
    sequences: dict[str, list[str]] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("SEQRES") or len(line) < 20:
            continue
        chain = line[11:12].strip()
        if not chain:
            continue
        sequences.setdefault(chain, []).extend(
            _THREE_TO_ONE.get(residue.upper(), "X")
            for residue in line[19:].split()
        )
    return {chain: "".join(residues) for chain, residues in sequences.items()}


def _sequence_matches(expected: str, observed: str) -> bool:
    """Return whether every observed residue occurs in expected order.

    Released structures can omit unresolved residues. Treating the observed
    sequence as an ordered subsequence preserves sequence evidence while
    allowing those omissions. A substitution or an inserted observed residue
    does not pass this test and must use the guarded fallback or fail.
    """
    position = 0
    for residue in expected:
        if position < len(observed) and residue == observed[position]:
            position += 1
    return position == len(observed)


def derive_chain_mapping(
    structure: Any,
    target_sequence: str,
    binder_sequence: str,
    *,
    structure_label: str = "structure",
) -> dict[str, str]:
    """Derive target and binder chains from known sequences.

    Exact sequence or ordered-subsequence matches are authoritative. Residue
    counts are a guarded fallback only when the target and binder lengths
    differ and each role has one candidate. Equal-length count-only
    assignments are rejected because they do not identify which chain is
    which.
    """
    target = _normalise_sequence(target_sequence, "target_sequence")
    binder = _normalise_sequence(binder_sequence, "binder_sequence")
    observed = _chain_sequences(structure, structure_label=structure_label)

    def fail(reason: str) -> None:
        summary = ", ".join(
            f"{chain}({len(sequence)} residues)"
            for chain, sequence in observed.items()
        ) or "none"
        raise ChainAssignmentError(
            f"{CHAIN_ASSIGNMENT_FAILURE_CODE}: {reason} in {structure_label}; "
            f"observed chains: {summary}"
        )

    target_exact = [
        chain for chain, sequence in observed.items() if _sequence_matches(target, sequence)
    ]
    binder_exact = [
        chain for chain, sequence in observed.items() if _sequence_matches(binder, sequence)
    ]

    if len(target_exact) == 1 and len(binder_exact) == 1:
        if target_exact[0] == binder_exact[0]:
            fail("the exact target and binder matches resolve to the same chain")
        return {"target": target_exact[0], "binder": binder_exact[0]}

    if len(target_exact) > 1 or len(binder_exact) > 1:
        fail(
            "sequence matching is ambiguous: "
            f"target matches {target_exact or 'none'}, "
            f"binder matches {binder_exact or 'none'}"
        )

    if len(target) != len(binder) and len(target_exact) == 1 and not binder_exact:
        binder_candidates = [
            chain
            for chain, sequence in observed.items()
            if chain != target_exact[0] and len(sequence) == len(binder)
        ]
        if len(binder_candidates) == 1:
            return {"target": target_exact[0], "binder": binder_candidates[0]}

    if len(target) != len(binder) and len(binder_exact) == 1 and not target_exact:
        target_candidates = [
            chain
            for chain, sequence in observed.items()
            if chain != binder_exact[0] and len(sequence) == len(target)
        ]
        if len(target_candidates) == 1:
            return {"target": target_candidates[0], "binder": binder_exact[0]}

    if not target_exact and not binder_exact and len(target) != len(binder):
        target_candidates = [
            chain for chain, sequence in observed.items() if len(sequence) == len(target)
        ]
        binder_candidates = [
            chain for chain, sequence in observed.items() if len(sequence) == len(binder)
        ]
        if len(target_candidates) == 1 and len(binder_candidates) == 1:
            return {"target": target_candidates[0], "binder": binder_candidates[0]}

    fail(
        "no confident assignment from exact sequence matches or the guarded "
        f"residue-count fallback; target length {len(target)}, binder length {len(binder)}"
    )


def _normalize_pae(pae: Any) -> list:
    """Normalize the matrix and run the check the executor runs.

    A tensor or an array is accepted, because both ESMFold2 arms hold prediction.pae as
    a tensor and the conversion is the same call every time.
    """
    if hasattr(pae, "tolist"):
        pae = pae.tolist()
    if not isinstance(pae, list) or not pae:
        raise BinderContractError("pae must be a non-empty list of lists")
    size = len(pae)
    matrix = []
    for index, line in enumerate(pae):
        if hasattr(line, "tolist"):
            line = line.tolist()
        if not isinstance(line, list) or len(line) != size:
            raise BinderContractError(
                f"the PAE matrix must be square: row {index} has {_length(line)} entries "
                f"against {size} rows"
            )
        row = []
        for column, value in enumerate(line):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BinderContractError(
                    f"PAE entry [{index}][{column}] must be a number, and it is {value!r}"
                )
            number = float(value)
            if not math.isfinite(number):
                raise BinderContractError(f"PAE entry [{index}][{column}] is {value!r}")
            if number < 0:
                raise BinderContractError(
                    f"PAE entry [{index}][{column}] is negative: {value!r}"
                )
            # The + 0.0 turns a negative zero back into zero, so the file's bytes
            # do not depend on which platform produced the matrix.
            row.append(number + 0.0)
        matrix.append(row)
    return matrix


def _length(value: Any) -> Any:
    try:
        return len(value)
    except TypeError:
        return f"a {type(value).__name__}"


def _require_mapped_chains(complex_path: Path, target_chain_id: str, binder_chain_id: str) -> None:
    """Check that both mapped chains appear in the complex that was just written.

    validate_raw_prediction_record runs this same check on every scored row, on PDB and
    on mmCIF alike. Running it here turns a whole stage failing after the GPU work into
    one candidate failing, and the message names the chains the file actually holds.

    A wrong chain_mapping is the failure worth spending code on. ipSAE is asymmetric, so
    a swapped mapping produces an inverted score rather than an error, and chain order is
    not uniform across the configured predictor modes.
    """
    suffix = complex_path.suffix.lower()
    observed = pdb_chain_ids(complex_path) if suffix == ".pdb" else cif_chain_ids(complex_path)
    mapped = {str(target_chain_id), str(binder_chain_id)}
    if mapped.issubset(observed):
        return
    raise BinderContractError(
        f"chain_mapping names chains the predicted complex does not hold. The file has "
        f"{sorted(observed)} and the mapping claims {sorted(mapped)}. Derive the mapping "
        f"from the structure rather than from chain position: {complex_path}"
    )


def pdb_chain_ids(path: Path) -> set:
    """Return every chain identifier in a PDB file's ATOM and HETATM records.

    Copied from the executor's PDB chain reader. The
    two have to agree, so executor_field_drift compares them.
    """
    chains = set()
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
            chain = line[21].strip()
            if chain:
                chains.add(chain)
        elif line.startswith("SEQRES") and len(line) > 11:
            chain = line[11].strip()
            if chain:
                chains.add(chain)
    return chains


def cif_row_fields(line: str) -> list:
    """Split one mmCIF loop row into its values, honouring CIF quoting.

    Copied from the executor's CIF row reader.
    """
    fields: list = []
    index = 0
    length = len(line)
    while index < length:
        while index < length and line[index].isspace():
            index += 1
        if index >= length:
            break
        quote = line[index]
        if quote in {"'", '"'}:
            cursor = index + 1
            while cursor < length:
                if line[cursor] == quote and (cursor + 1 >= length or line[cursor + 1].isspace()):
                    break
                cursor += 1
            if cursor < length:
                fields.append(line[index + 1 : cursor])
                index = cursor + 1
                continue
        start = index
        while index < length and not line[index].isspace():
            index += 1
        fields.append(line[start:index])
    return fields


def cif_chain_ids(path: Path) -> set:
    """Return every chain identifier named in an mmCIF `_atom_site` loop.

    Both `label_asym_id` and `auth_asym_id` are collected, because cofold modes
    do not agree on which column carries the chain letter a row maps. The column index is
    read from the loop header, since mmCIF fixes no column order.

    Copied from the executor's CIF chain reader.
    """
    chains = set()
    if path.suffix.lower() not in {".cif", ".mmcif"}:
        return chains
    wanted = {"_atom_site.label_asym_id", "_atom_site.auth_asym_id"}
    columns: list = []
    indexes: list = []
    reading_header = False
    reading_rows = False
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            reading_header = False
            reading_rows = False
            continue
        if line == "loop_":
            columns = []
            indexes = []
            reading_header = True
            reading_rows = False
            continue
        if reading_header:
            if line.startswith("_"):
                columns.append(line.split()[0])
                continue
            indexes = [index for index, name in enumerate(columns) if name in wanted]
            reading_header = False
            reading_rows = bool(indexes)
        if not reading_rows:
            continue
        if line.startswith("_") or line.startswith("data_"):
            reading_rows = False
            continue
        fields = cif_row_fields(line)
        for index in indexes:
            if index < len(fields) and fields[index] not in {"", ".", "?"}:
                chains.add(fields[index])
    return chains


def _read_structure_text(name: str, value: Any) -> str:
    if isinstance(value, os.PathLike):
        return Path(value).read_text()
    if not isinstance(value, str) or not value.strip():
        raise BinderContractError(
            f"{name} must be mmCIF text or a path to a file holding it, and it is {value!r}"
        )
    return value


def _require_extra(extra: dict) -> None:
    missing = [name for name in REQUIRED_EXTRA_FIELDS if name not in extra]
    if not missing:
        return
    lines = [f"  {name}: {REQUIRED_EXTRA_FIELDS[name]}" for name in missing]
    raise BinderContractError(
        "extra is missing measurement fields that no binder_metrics function returns:\n"
        + "\n".join(lines)
    )


def _merge_measurement(identity: dict, ipsae: Any, dockq: Any, site: Any, extra: dict) -> dict:
    """Merge the identity block, the three metric results and extra into one measurement.

    A name supplied twice raises. Two sources silently overwriting each other is how a
    plausible wrong number reaches a row, and the row still validates.
    """
    measurement: dict = {}
    sources = (
        ("the identity block", identity),
        ("binder_metrics.compute_ipsae", _require_mapping("compute_ipsae", ipsae)),
        ("binder_metrics.compute_dockq", _require_mapping("compute_dockq", dockq)),
        ("binder_metrics.compute_site_metrics", _require_mapping("compute_site_metrics", site)),
        ("extra", extra),
    )
    owners: dict = {}
    for label, values in sources:
        for name, value in values.items():
            if name in owners:
                raise BinderContractError(
                    f"{label} and {owners[name]} both supply {name}, so one would "
                    "silently overwrite the other"
                )
            owners[name] = label
            measurement[name] = value
    return measurement


def _require_mapping(name: str, value: Any) -> dict:
    if not isinstance(value, dict):
        raise BinderContractError(
            f"binder_metrics.{name} must return a dict, and it returned a "
            f"{type(value).__name__}"
        )
    return value


def _require_complete_measurement(measurement: dict) -> None:
    missing = sorted(set(MEASUREMENT_SOURCE_FIELDS) - set(measurement))
    if missing:
        raise BinderContractError(
            "the measurement is missing "
            f"{len(missing)} of {MEASUREMENT_SOURCE_FIELD_COUNT} required fields: "
            + ", ".join(missing)
        )
    allowed_nulls = (
        UNCONSTRAINED_NULL_FIELDS
        if measurement.get("epitope_constraint") == "unconstrained"
        else frozenset()
    )
    empty = sorted(
        name
        for name in MEASUREMENT_SOURCE_FIELDS
        if measurement[name] is None and name not in allowed_nulls
    )
    if empty:
        raise BinderContractError(
            "the measurement carries null for fields the executor reads as numbers or "
            "strings: " + ", ".join(empty)
        )


def _require_mapping_status(measurement: dict) -> None:
    """Refuse a scored row whose chain mapping failed.

    validate_observations rejects a scored row whose mapping_status is anything but ok,
    and it does that at the end of the run. compute_dockq reports the failure here
    instead, so the arm writes a failed row and the next candidate still runs.
    """
    status = measurement.get("mapping_status")
    if status == "ok":
        return
    raise BinderContractError(
        f"compute_dockq could not map the chains and reported mapping_status {status!r}. "
        "The executor rejects a scored row that says anything but ok, so this prediction "
        "belongs in a failed row."
    )


def _coerce_text(value: Any) -> Any:
    if isinstance(value, str) and value:
        return value
    if value is None:
        return _FAILURE_CODE_FALLBACK
    return str(value)


def _coerce_seed(seed: Any) -> Any:
    if isinstance(seed, bool):
        return int(seed)
    if isinstance(seed, int):
        return seed
    try:
        return int(seed)
    except (TypeError, ValueError):
        # The value passes through unchanged so the wrong seed is visible in the row
        # rather than replaced by an invented one. The schema then rejects the row.
        return seed


def _coerce_failure_code(failure_code: Any) -> str:
    text = _FAILURE_CODE_UNSAFE.sub("_", str(failure_code).strip().lower()).strip("_")
    return text or _FAILURE_CODE_FALLBACK


def _coerce_failure_reason(failure_reason: Any, failure_code: Any) -> str:
    text = str(failure_reason).strip()
    if not text:
        return f"no reason was recorded for {_coerce_failure_code(failure_code)}"
    if len(text) > _FAILURE_REASON_LIMIT:
        return text[:_FAILURE_REASON_LIMIT] + " [truncated]"
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    """Write canonical JSON.

    indent=2, sort_keys=True and a trailing newline. The bytes set the hash the row
    records, so the formatting is part of the contract.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
