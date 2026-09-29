#!/usr/bin/env python3
"""Measure interface chemistry on the complexes a cofold stage predicted.

Every post-fold number this package records today comes from the predictor's own
container. ipSAE reports how confident the model is about the interface it drew,
and sc_DockQ reports how closely that pose matches a reference complex. Neither
reads the chemistry of the interface. A design can carry a high ipSAE_min across
an interface that buries almost no surface, and nothing downstream says so.

This adapter reads the predicted complexes a screen already wrote and measures
buried surface area, the apolar share of that area, van der Waals overlap and
contact density, through the same Shrake-Rupley code the pre-fold cheap screen
uses. `screen_geometry.interface_geometry` is a function over two residue lists,
so pointing it at a predicted complex rather than a painted design pose needs no
new geometry.

The adapter supplies no numerical convention. Radii, probe radius, sampling
count, the contact cutoff and every gate threshold come from the campaign,
because a constant this module chose would be one nobody can cite.

It reads local structure files. It never launches a fold, downloads weights, or
contacts a provider.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from . import screen_geometry
from .integrity_filter import (
    AdapterError,
    completed_receipt_file,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from ..filter_contracts import filter_report

SCORE_TABLE_STAGE = "score-screen"
SCORE_TABLE_ARTIFACT = "screen-score-table"

OBSERVATIONS_ARTIFACT_ID = "interface-biophysics-observations"
MEASUREMENTS_ARTIFACT_ID = "interface-biophysics-measurements"
REPORT_ARTIFACT_ID = "interface-biophysics-report"

# The geometry values this stage records for one prediction. `screen_geometry`
# returns them under these names, and a gate names one of them as its metric.
MEASURED_METRICS = (
    "interface_buried_sasa_angstrom2",
    "interface_apolar_fraction",
    "hard_clash_count",
    "contact_density",
    "interchain_residue_contact_count",
    "interface_residue_count",
    "hotspot_coverage_count",
)

# A scored row names the predictor's own chain letters separately from the
# campaign's. The predicted complex carries the predictor's letters, so reading
# it by the campaign's target_chain_id returns the wrong chain or no chain at
# all. See the `two chain mappings, not one` case in the scoring reference.
PREDICTED_TARGET_CHAIN_FIELD = "predicted_target_chain_id"
PREDICTED_BINDER_CHAIN_FIELD = "predicted_binder_chain_id"

_OPERATORS = {
    ">=": lambda value, threshold: value >= threshold,
    ">": lambda value, threshold: value > threshold,
    "<=": lambda value, threshold: value <= threshold,
    "<": lambda value, threshold: value < threshold,
}


class InterfaceBiophysicsError(AdapterError):
    """A campaign or a structure cannot support the post-fold geometry read."""


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InterfaceBiophysicsError(f"{label} must be a non-empty string")
    return value.strip()


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InterfaceBiophysicsError(f"{label} must be a number")
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        raise InterfaceBiophysicsError(f"{label} must be finite")
    if minimum is not None and number < minimum:
        raise InterfaceBiophysicsError(f"{label} must be at least {minimum}")
    return number


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InterfaceBiophysicsError(f"{label} must be a positive integer")
    return value


def validate_geometry(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read every numerical convention from the campaign.

    This module names no default. A radius or a probe size chosen here would
    travel onto a result row with no source behind it, which is the shape of
    invention this package refuses everywhere else.
    """
    section = config.get("interface_biophysics")
    if not isinstance(section, Mapping):
        raise InterfaceBiophysicsError("campaign has no interface_biophysics object")
    raw = section.get("geometry")
    if not isinstance(raw, Mapping):
        raise InterfaceBiophysicsError("interface_biophysics.geometry must be an object")
    raw_radii = raw.get("van_der_waals_radii_angstrom")
    if not isinstance(raw_radii, Mapping) or not raw_radii:
        raise InterfaceBiophysicsError(
            "interface_biophysics.geometry.van_der_waals_radii_angstrom must be a non-empty object"
        )
    radii = {
        _nonempty_string(
            element, "interface_biophysics.geometry.van_der_waals_radii_angstrom key"
        ).upper(): _finite_number(
            radius,
            f"interface_biophysics.geometry.van_der_waals_radii_angstrom.{element}",
            minimum=0.0,
        )
        for element, radius in raw_radii.items()
    }
    if any(radius <= 0 for radius in radii.values()):
        raise InterfaceBiophysicsError(
            "interface_biophysics.geometry.van_der_waals_radii_angstrom values must be positive"
        )
    apolar = raw.get("apolar_elements")
    if not isinstance(apolar, list) or not apolar or any(
        not isinstance(item, str) or not item.strip() for item in apolar
    ):
        raise InterfaceBiophysicsError(
            "interface_biophysics.geometry.apolar_elements must be a non-empty string list"
        )
    source = _nonempty_string(
        raw.get("convention_source"), "interface_biophysics.geometry.convention_source"
    )
    return {
        "radii_angstrom": radii,
        "apolar_elements": frozenset(item.strip().upper() for item in apolar),
        "clash_tolerance_angstrom": _finite_number(
            raw.get("clash_tolerance_angstrom"),
            "interface_biophysics.geometry.clash_tolerance_angstrom",
            minimum=0.0,
        ),
        "contact_cutoff_angstrom": _finite_number(
            raw.get("contact_cutoff_angstrom"),
            "interface_biophysics.geometry.contact_cutoff_angstrom",
            minimum=0.0,
        ),
        "sasa_probe_radius_angstrom": _finite_number(
            raw.get("sasa_probe_radius_angstrom"),
            "interface_biophysics.geometry.sasa_probe_radius_angstrom",
            minimum=0.0,
        ),
        "sasa_sphere_point_count": _positive_int(
            raw.get("sasa_sphere_point_count"),
            "interface_biophysics.geometry.sasa_sphere_point_count",
        ),
        "convention_source": source,
    }


def validate_gates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read the optional gate contracts.

    Zero gates is a supported campaign. The stage then measures every prediction
    and removes none, which is what a run wants before anyone has a threshold
    they can defend. A gate that does exist has to name where its number came
    from, the same requirement every other filter contract carries.
    """
    section = config.get("interface_biophysics")
    raw = section.get("gates") if isinstance(section, Mapping) else None
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise InterfaceBiophysicsError("interface_biophysics.gates must be a list")
    contracts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise InterfaceBiophysicsError(f"interface_biophysics.gates[{index}] must be an object")
        filter_id = _nonempty_string(
            entry.get("filter_id"), f"interface_biophysics.gates[{index}].filter_id"
        )
        if filter_id in seen:
            raise InterfaceBiophysicsError(
                f"interface_biophysics.gates repeats filter_id {filter_id!r}"
            )
        seen.add(filter_id)
        metric = _nonempty_string(
            entry.get("metric"), f"interface_biophysics.gates[{index}].metric"
        )
        if metric not in MEASURED_METRICS:
            raise InterfaceBiophysicsError(
                f"interface_biophysics.gates[{index}].metric {metric!r} is not measured. "
                "This stage measures " + ", ".join(MEASURED_METRICS)
            )
        operator = _nonempty_string(
            entry.get("operator"), f"interface_biophysics.gates[{index}].operator"
        )
        if operator not in _OPERATORS:
            raise InterfaceBiophysicsError(
                f"interface_biophysics.gates[{index}].operator must be one of "
                + ", ".join(sorted(_OPERATORS))
            )
        contracts.append(
            {
                "filter_id": filter_id,
                "metric": metric,
                "operator": operator,
                "threshold": _finite_number(
                    entry.get("threshold"), f"interface_biophysics.gates[{index}].threshold"
                ),
                "threshold_source": _nonempty_string(
                    entry.get("threshold_source"),
                    f"interface_biophysics.gates[{index}].threshold_source",
                ),
            }
        )
    return contracts


def hotspot_labels(config: Mapping[str, Any]) -> frozenset[str]:
    """Return the campaign's site residue labels, or an empty set.

    An empty set is honest rather than a fallback. `hotspot_coverage_count` then
    reports zero against a `hotspot_count` of zero, and a gate on that metric
    cannot be configured into a silent pass, because the count it compares is
    recorded beside it.
    """
    section = config.get("interface_biophysics")
    raw = section.get("hotspot_labels") if isinstance(section, Mapping) else None
    if raw is None:
        return frozenset()
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise InterfaceBiophysicsError("interface_biophysics.hotspot_labels must be a string list")
    return frozenset(item.strip() for item in raw if item.strip())


def scored_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep the rows that carry a measured structure.

    A failed prediction carries no complex to read. It stays out of the
    measurement table rather than entering it with null geometry, because a null
    beside a real number invites a comparison that was never measured.
    """
    kept: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("status") != "scored":
            continue
        if not isinstance(row.get("predicted_complex_path"), str):
            continue
        kept.append(row)
    return kept


def measure_row(
    row: Mapping[str, Any],
    *,
    geometry: Mapping[str, Any],
    labels: frozenset[str],
) -> dict[str, Any]:
    """Measure one predicted complex and return its record."""
    complex_path = Path(str(row["predicted_complex_path"])).expanduser()
    if not complex_path.is_file():
        raise InterfaceBiophysicsError(f"predicted complex is missing: {complex_path}")
    target_chain = _nonempty_string(
        row.get(PREDICTED_TARGET_CHAIN_FIELD), f"score row {PREDICTED_TARGET_CHAIN_FIELD}"
    )
    binder_chain = _nonempty_string(
        row.get(PREDICTED_BINDER_CHAIN_FIELD), f"score row {PREDICTED_BINDER_CHAIN_FIELD}"
    )
    if target_chain == binder_chain:
        raise InterfaceBiophysicsError(
            f"score row names one chain {target_chain!r} as both target and binder"
        )
    structure = screen_geometry.read_structure(str(complex_path))
    target_residues = screen_geometry.chain_residues(structure, target_chain)
    binder_residues = screen_geometry.chain_residues(structure, binder_chain)
    measured = screen_geometry.interface_geometry(
        target_residues,
        binder_residues,
        labels,
        radii_angstrom=geometry["radii_angstrom"],
        clash_tolerance_angstrom=geometry["clash_tolerance_angstrom"],
        contact_cutoff_angstrom=geometry["contact_cutoff_angstrom"],
        sasa_probe_radius_angstrom=geometry["sasa_probe_radius_angstrom"],
        sasa_sphere_point_count=geometry["sasa_sphere_point_count"],
        apolar_elements=geometry["apolar_elements"],
    )
    return {
        "target_id": row.get("target_id"),
        "candidate_id": row.get("candidate_id"),
        "predictor": row.get("predictor"),
        "seed": row.get("seed"),
        "phase": row.get("phase"),
        "predicted_complex_path": str(complex_path),
        "predicted_complex_sha256": sha256_file(complex_path),
        "target_chain_id": target_chain,
        "binder_chain_id": binder_chain,
        "interface_buried_sasa_angstrom2": measured.interface_buried_sasa_angstrom2,
        "interface_apolar_fraction": measured.interface_apolar_fraction,
        "hard_clash_count": measured.hard_clash_count,
        "contact_density": measured.contact_density,
        "interchain_residue_contact_count": measured.interchain_residue_contact_count,
        "interface_residue_count": measured.interface_residue_count,
        "hotspot_coverage_count": measured.hotspot_coverage_count,
        "hotspot_count": measured.hotspot_count,
        "geometry_convention_source": geometry["convention_source"],
        "status": "measured",
    }


def gate_observations(
    measurements: list[dict[str, Any]], contracts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Evaluate every configured gate against every measurement."""
    observations: list[dict[str, Any]] = []
    for contract in contracts:
        compare = _OPERATORS[contract["operator"]]
        for record in measurements:
            value = float(record[contract["metric"]])
            passed = bool(compare(value, contract["threshold"]))
            observations.append(
                {
                    "candidate_id": record["candidate_id"],
                    "predictor": record["predictor"],
                    "seed": record["seed"],
                    "filter_id": contract["filter_id"],
                    "metric": contract["metric"],
                    "operator": contract["operator"],
                    "threshold": contract["threshold"],
                    "threshold_source": contract["threshold_source"],
                    "value": value,
                    "pass": passed,
                    "reason": (
                        f"{contract['metric']} {value} {contract['operator']} "
                        f"{contract['threshold']}"
                        if passed
                        else f"{contract['metric']} {value} fails "
                        f"{contract['operator']} {contract['threshold']}"
                    ),
                }
            )
    return observations


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Measure every scored prediction and write the stage's three artifacts."""
    config = read_json(Path(args.config), "campaign config")
    geometry = validate_geometry(config)
    contracts = validate_gates(config)
    labels = hotspot_labels(config)
    table_path = completed_receipt_file(
        Path(args.receipts_dir),
        SCORE_TABLE_STAGE,
        SCORE_TABLE_ARTIFACT,
        label="screen score table",
    )
    rows = scored_rows(read_jsonl(table_path, "screen score table"))
    if not rows:
        raise InterfaceBiophysicsError(
            f"screen score table carries no scored rows with a predicted complex: {table_path}"
        )
    measurements = [measure_row(row, geometry=geometry, labels=labels) for row in rows]
    observations = gate_observations(measurements, contracts)
    failed = {
        str(observation["candidate_id"])
        for observation in observations
        if observation["pass"] is False
    }
    candidate_ids = sorted({str(record["candidate_id"]) for record in measurements})
    passing = [identifier for identifier in candidate_ids if identifier not in failed]
    report = filter_report(
        str(args.stage),
        candidate_ids,
        contracts,
        observations,
        passing,
    )
    report["measured_prediction_count"] = len(measurements)
    report["geometry_convention_source"] = geometry["convention_source"]
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    out_dir = Path(args.attempt_dir) / str(args.phase)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "interface-biophysics.jsonl", measurements)
    write_jsonl(out_dir / "interface-biophysics-observations.jsonl", observations)
    # The name ends in `filter-report.json` because report.py collects gate
    # statuses with rglob("*filter-report.json"). Matching that pattern renders
    # this stage in the scientist-facing report with no change to the renderer.
    write_json(out_dir / "interface-biophysics-filter-report.json", report)
    return {
        "ok": True,
        "measured_prediction_count": len(measurements),
        "evaluated_gate_count": len(contracts),
        "passing_candidate_count": len(passing),
    }


def parse_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Re-read the written artifacts so the executor can trust the receipt."""
    out_dir = Path(args.attempt_dir) / str(args.phase)
    measurements = read_jsonl(out_dir / "interface-biophysics.jsonl", "measurements")
    report = read_json(out_dir / "interface-biophysics-filter-report.json", "report")
    result = {
        "ok": True,
        "parsed_count": len(measurements),
        "stage_id": report.get("stage_id"),
    }
    # The executor reads this back from `parser_result_path_template`, which every
    # adapter in the package sets to the same conventional path. Deriving it here
    # rather than taking it on argv keeps the two from drifting apart.
    write_json(out_dir / "parser-result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("toolcheck")
    for name in ("run", "parse"):
        child = sub.add_parser(name)
        child.add_argument("--stage", required=True)
        child.add_argument("--phase", required=True)
        child.add_argument("--attempt-dir", required=True)
        child.add_argument("--receipts-dir", required=(name == "run"))
        child.add_argument("--artifact-root", required=False)
        child.add_argument("--config", required=(name == "run"))
        child.add_argument("--plan", required=False)
        child.add_argument("--count", required=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "toolcheck":
        print(json.dumps({"ok": True, "environment_identity": "standard-library-python"}))
        return 0
    try:
        result = run_stage(args) if args.command == "run" else parse_stage(args)
    except (AdapterError, screen_geometry.ScreenGeometryError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
