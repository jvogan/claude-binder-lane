"""Independently validate downloaded Genie3/PXDesign kit artifacts and evidence.

This stdlib-only reader launches nothing. It checks actual structures in
addition to upstream's manifest and log. It emits relative artifact names and
hashes, without copying provider identities, environment values or source paths
into its report. This proves a bounded computational run, not biological quality.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys


class ValidationError(ValueError):
    pass


# Independent census of the pinned release's mode tables. A manifest omitting a
# required lever cannot make a reduced computation pass its own smaller plan.
_GENIE_EXACT = {"L1", "L2", "L4", "L8", "L9", "L11", "L17", "L18", "L19"}
_GENIE_FAST = _GENIE_EXACT | {"L7", "L12", "L13", "L16"}
_PX_HOIST = {"h1", "h2", "h3", "h4", "h5"}
_PX_FAST = {"featdiet", "padmask", "tf32", "sdedup"}


def _sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def _cif_tokens(text: str) -> list[str]:
    """CIF whitespace, comments, quoted values and column-one text fields."""
    out, i, length = [], 0, len(text)
    while i < length:
        if text[i].isspace():
            i += 1
            continue
        if text[i] == "#":
            end = text.find("\n", i)
            i = length if end < 0 else end + 1
            continue
        if text[i] == ";" and (i == 0 or text[i - 1] == "\n"):
            end = text.find("\n;", i + 1)
            if end < 0:
                raise ValidationError("unterminated CIF text field")
            out.append(text[i + 1:end])
            i = end + 2
            continue
        if text[i] in ("'", '"'):
            quote, start = text[i], i + 1
            i += 1
            while i < length:
                if text[i] == quote and (i + 1 == length or text[i + 1].isspace()):
                    out.append(text[start:i])
                    i += 1
                    break
                i += 1
            else:
                raise ValidationError("unterminated CIF quoted value")
            continue
        start = i
        while i < length and not text[i].isspace():
            i += 1
        out.append(text[start:i])
    return out


def _atom_rows(text: str) -> list[dict[str, str]]:
    tokens = _cif_tokens(text)
    i = 0
    while i < len(tokens):
        if tokens[i] != "loop_":
            i += 1
            continue
        i += 1
        columns = []
        while i < len(tokens) and tokens[i].startswith("_"):
            columns.append(tokens[i])
            i += 1
        if not columns:
            raise ValidationError("CIF loop has no columns")
        values = []
        while i < len(tokens):
            word = tokens[i]
            boundary = word.startswith(("_", "data_", "save_")) or word in ("loop_", "stop_", "global_")
            if boundary and len(values) % len(columns) == 0:
                break
            values.append(word)
            i += 1
        if any(column.startswith("_atom_site.") for column in columns):
            if not all(column.startswith("_atom_site.") for column in columns):
                raise ValidationError("mixed atom-site CIF loop")
            if len(values) % len(columns):
                raise ValidationError("truncated atom-site CIF row")
            return [dict(zip(columns, values[j:j + len(columns)]))
                    for j in range(0, len(values), len(columns))]
    raise ValidationError("no atom-site CIF loop")


def _value(row: dict, *names: str, default: str = "") -> str:
    return next((row[name] for name in names if row.get(name) not in (None, "", ".", "?")), default)


def parse_atoms(path: Path) -> list[tuple]:
    """Return canonical (model, chain, residue, insertion, atom, alt, xyz) rows."""
    text = path.read_text(encoding="utf-8")
    atoms = []
    if path.suffix.lower() == ".pdb":
        model = "1"
        for line in text.splitlines():
            if line.startswith("MODEL "):
                model = line[10:14].strip() or "1"
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if len(line) < 54:
                raise ValidationError("truncated PDB atom row")
            try:
                xyz = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
            except ValueError as exc:
                raise ValidationError("invalid PDB coordinates") from exc
            atoms.append((model, line[21].strip(), line[22:26].strip(), line[26].strip(),
                          line[12:16].strip(), line[16].strip(), xyz))
    else:
        for row in _atom_rows(text):
            if row.get("_atom_site.group_PDB", "ATOM") not in ("ATOM", "HETATM"):
                continue
            try:
                xyz = tuple(float(row[f"_atom_site.Cartn_{axis}"]) for axis in "xyz")
            except (ValueError, KeyError) as exc:
                raise ValidationError("invalid CIF coordinates") from exc
            atoms.append((_value(row, "_atom_site.pdbx_PDB_model_num", default="1"),
                          _value(row, "_atom_site.label_asym_id", "_atom_site.auth_asym_id"),
                          _value(row, "_atom_site.label_seq_id", "_atom_site.auth_seq_id"),
                          _value(row, "_atom_site.pdbx_PDB_ins_code"),
                          _value(row, "_atom_site.label_atom_id", "_atom_site.auth_atom_id"),
                          _value(row, "_atom_site.label_alt_id"), xyz))
    if not atoms:
        raise ValidationError("structure has no atoms")
    if any(not all(math.isfinite(x) for x in atom[-1]) for atom in atoms):
        raise ValidationError("structure has non-finite coordinates")
    if any(not atom[1] or not atom[2] or not atom[4] for atom in atoms):
        raise ValidationError("atom identity lacks chain, residue or atom name")
    identities = [atom[:-1] for atom in atoms]
    if len(set(identities)) != len(identities):
        raise ValidationError("structure has duplicate atom identities")
    if len({atom[0] for atom in atoms}) != 1:
        raise ValidationError("expected one structural model")
    if len({atom[-1] for atom in atoms}) < 2:
        raise ValidationError("all atom coordinates collapse to one point")
    return sorted(atoms)


def structure_summary(atoms: list[tuple]) -> dict:
    chains = defaultdict(lambda: defaultdict(set))
    for model, chain, residue, insertion, name, alt, xyz in atoms:
        chains[chain][(residue, insertion)].add(name)
    return {
        "atoms": len(atoms),
        "chains": {
            chain: {"residues": len(residues),
                    "ca_residues": sum("CA" in names for names in residues.values()),
                    "ca_only_residues": sum(names == {"CA"} for names in residues.values()),
                    "full_backbone_residues": sum({"N", "CA", "C", "O"} <= names for names in residues.values())}
            for chain, residues in sorted(chains.items())
        },
    }


def _need(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _genie_evidence(man: dict, text: str, out: Path, mode: str, expected: int, steps: int, errors: list) -> dict:
    _need(man.get("schema") == "genie3_opt.manifest/1", "wrong Genie3 manifest schema", errors)
    _need(man.get("status") == "ok", "Genie3 pass is not complete", errors)
    _need(man.get("activation", {}).get("mode") == mode, "activation mode differs from request", errors)
    _need(man.get("activation", {}).get("active") is True, "Genie3 activation is inactive", errors)
    _need(man.get("activation", {}).get("stack", {}).get("pinned") is True, "Genie3 stack is not pinned", errors)
    _need(man.get("activation", {}).get("pins", {}).get("checkout", {}).get("pinned") is True, "Genie3 checkout is not pinned", errors)
    _need(man.get("request", {}).get("designs_expected") == expected, "request output count differs", errors)
    _need(man.get("outputs", {}).get("n_pdb") == expected, "manifest PDB count differs", errors)
    _need(man.get("weights", {}).get("pinned") is True, "Genie3 weights are not pinned", errors)
    _need(not man.get("skipped"), "Genie3 pass resumed existing outputs", errors)
    _need(bool(re.search(rf"\[genie3-opt\] ACTIVE mode={mode}\b", text)), "requested Genie3 mode was not announced", errors)
    _need(bool(re.search(rf"\[genie3-opt\] ready mode={mode}\b", text)), "no fresh Genie3 output readiness evidence", errors)
    if mode == "off":
        stock = man.get("stock") or {}
        _need(stock.get("rc") == 0 and stock.get("proof", {}).get("ok") is True, "stock subprocess lacks a clean successful proof", errors)
        # The stock route inherits the pinned sampler default when the composed
        # request omits n_sample_step. State that evidence source explicitly.
        request = (out / "request.yaml").read_text() if (out / "request.yaml").is_file() else ""
        match = re.search(r"\bn_sample_step:\s*(\d+)", request)
        actual_steps = int(match.group(1)) if match else 100
        _need(bool(request), "missing stock composed request", errors)
        _need(actual_steps == steps, "stock sampler steps differ", errors)
        return {"stock_proof_ok": stock.get("proof", {}).get("ok"), "sampler_steps": actual_steps,
                "sampler_steps_source": "composed-request" if match else "pinned-upstream-default"}
    driver = man.get("driver_pass") or {}
    evidence = driver.get("evidence") or {}
    planned = set(man.get("levers_planned") or [])
    _need(planned == (_GENIE_EXACT if mode == "exact" else _GENIE_FAST), "Genie3 plan omits or adds pinned mode levers", errors)
    applied = set(evidence.get("applied") or [])
    declined = evidence.get("declined") or {}
    _need(driver.get("rc") == 0, "Genie3 driver failed", errors)
    _need(bool(planned) and planned <= applied | set(declined), "planned Genie3 lever lacks evidence", errors)
    _need(not evidence.get("missing") and not man.get("levers_missing"), "Genie3 has missing levers", errors)
    _need(not evidence.get("forbidden") and not man.get("levers_broken") and not man.get("forbidden_lines"), "Genie3 has broken or forbidden lever evidence", errors)
    _need(driver.get("numerics", {}).get("verdict") == "ok", "Genie3 numerics probe failed", errors)
    timing_path = out / "timings.json"
    timings = json.loads(timing_path.read_text()) if timing_path.is_file() else {}
    actual_steps = (timings.get("sampler") or {}).get("n_sample_step")
    _need(actual_steps == steps, "actual Genie3 sampler steps differ or are absent", errors)
    lever = timings.get("lever") or {}
    cache = timings.get("graph_cache") or {}
    live = timings.get("numerics_readback") or {}
    _need(lever.get("finished") is True and int(lever.get("batches_done") or 0) > 0,
          "Genie3 sampler did not finish its batches", errors)
    _need(int(cache.get("captures") or 0) > 0, "Genie3 graph was not captured", errors)
    _need(int(timings.get("D59_global_state_checks_passed") or 0) > 0,
          "Genie3 global numerics probe has no completed checks", errors)
    _need(lever.get("hoist_anomalies") == 0, "Genie3 hoist probe is missing or anomalous", errors)
    _need(timings.get("stock_batch") == man.get("request", {}).get("batch_size"),
          "actual Genie3 batch differs from request", errors)
    _need(live.get("matmul_tf32") is (mode == "fast") and live.get("matmul") == ("high" if mode == "fast" else "highest"),
          "Genie3 live matmul policy differs from requested mode", errors)
    trimul = timings.get("trimul") or {}
    census = trimul.get("census") or {}
    if mode == "fast":
        _need((timings.get("compile") or {}).get("backend") == "inductor"
              and int((timings.get("compile") or {}).get("shapes") or 0) > 0,
              "Genie3 fast core has no compilation evidence", errors)
        _need((trimul.get("gate") or {}).get("ok") is True and not census.get("errors"),
              "Genie3 fast triangle kernel probe failed", errors)
        served = int(census.get("served") or 0)
        valid_decline = trimul.get("declined") == "below_min_tokens" and declined.get("L7") == "below_min_tokens"
        _need(served > 0 or valid_decline, "Genie3 fast triangle kernel neither served nor declared its token floor", errors)
        _need(not (set(census.get("fallback") or {}) - {"below_min_tokens"}),
              "Genie3 fast triangle kernel has unexpected fallbacks", errors)
    return {"levers_evidenced": sorted(applied), "levers_declined": declined,
            "numerics_verdict": driver.get("numerics", {}).get("verdict"),
            "sampler_steps": actual_steps, "sampler_steps_source": "driver-timings",
            "graph_captures": cache.get("captures"), "graph_reuses": cache.get("hits"),
            "global_state_checks_passed": timings.get("D59_global_state_checks_passed"),
            "matmul_policy": live.get("matmul"), "matmul_tf32": live.get("matmul_tf32"),
            "triangle_kernel_served_calls": census.get("served") if mode == "fast" else None,
            "compiled_shapes": (timings.get("compile") or {}).get("shapes") if mode == "fast" else None}


def _px_evidence(man: dict, text: str, out: Path, mode: str, expected: int, steps: int, errors: list) -> dict:
    _need(man.get("schema") == "pxdesign_opt.opt_manifest.v3", "wrong PXDesign manifest schema", errors)
    _need(man.get("exit_code") == 0, "PXDesign manifest exit is nonzero", errors)
    stack = man.get("stack") or {}
    stack_versions = {name: stack.get(name) for name in ("python", "torch", "cuda", "triton")}
    _need(stack_versions == {"python": "3.11.5", "torch": "2.3.1+cu121", "cuda": "12.1", "triton": "2.3.1"},
          "PXDesign actual stack differs from the pinned recipe", errors)
    outputs = man.get("outputs") or {}
    _need(outputs.get("n_designs") == expected and outputs.get("expected") == expected, "PXDesign output counts differ", errors)
    _need(outputs.get("complete") is True and outputs.get("produced") == expected, "PXDesign did not produce the expected fresh outputs", errors)
    _need(not outputs.get("nothing_ran") and outputs.get("skipped_existing") == 0, "PXDesign resumed pre-existing outputs", errors)
    _need(bool(re.search(rf"\[pxdesign-opt\] DONE mode={mode}\b.*\bdesigns={expected}\b", text)), "PXDesign fresh completion evidence missing", errors)
    values = (man.get("options") or {}).get("values") or {}
    _need(str(values.get("N_step")) == str(steps), "PXDesign sampler steps differ", errors)
    record = {"sampler_steps": values.get("N_step"), "dtype": values.get("dtype"), "stack": stack_versions}
    if mode == "off":
        proof = man.get("stock_env_proof") or {}
        violations = proof.get("violations") or {}
        # Upstream keeps named false/empty violation fields. The mapping itself
        # is truthy even when its actual census is clean.
        contaminated = any(violations.values()) if isinstance(violations, dict) else bool(violations)
        full_path = out / "stock_env_proof.json"
        full = json.loads(full_path.read_text()) if full_path.is_file() else {}
        _need(bool(re.search(r"\[pxdesign-opt\] STOCK mode=off\b", text)), "stock PXDesign was not announced", errors)
        _need(bool(proof) and not proof.get("missing") and not contaminated,
              "stock PXDesign proof missing or contaminated", errors)
        _need(full.get("ok") is True and full.get("clean") is True and full.get("exit_code") == 0,
              "stock PXDesign full environment proof is missing or failed", errors)
        _need(full.get("after_ok") is True and not full.get("kit_modules_loaded_after")
              and not full.get("lever_modules_imported"), "stock PXDesign after-call isolation proof failed", errors)
        return dict(record, stock_proof_clean=bool(proof) and not contaminated and full.get("clean") is True,
                    stock_after_ok=full.get("after_ok"),
                    stock_lever_modules_imported=full.get("lever_modules_imported"),
                    stock_proof_sha256=_sha256(full_path) if full_path.is_file() else None)
    rep = man.get("activation_report") or {}
    _need(man.get("active") is True and rep.get("mode") == mode, "PXDesign activation differs or is inactive", errors)
    pins = rep.get("stock_pins") or {}
    _need(pins.get("pinned") is True and not pins.get("bad"), "PXDesign stock source is not pinned", errors)
    _need(bool(re.search(rf"\[pxdesign-opt\] ACTIVE mode={mode}\b", text)), "requested PXDesign mode was not announced", errors)
    planned = set(rep.get("levers_planned") or [])
    applied = set(rep.get("levers_applied") or [])
    packages = set(rep.get("package_levers") or [])
    packages_applied = set(rep.get("package_levers_applied") or [])
    wanted_packages = set() if mode == "exact" else _PX_FAST | ({"rowpipe"} if mode == "big" else set())
    _need(planned == _PX_HOIST and packages == wanted_packages, "PXDesign plan omits or adds pinned mode levers", errors)
    _need(bool(planned) and planned <= applied and packages <= packages_applied, "PXDesign planned lever lacks application evidence", errors)
    _need(not rep.get("partial") and not rep.get("levers_fallback") and not rep.get("package_levers_fallback") and not man.get("package_gate"), "PXDesign partial or failed lever census", errors)
    _need(bool(rep.get("applications")), "PXDesign model hook never ran", errors)
    _need(bool(re.search(r"\[pxdesign-opt\] EXIT tally: installed=True prepares=[1-9]\d*\b", text)), "PXDesign hoist did not sample", errors)
    precision = man.get("precision") or {}
    _need(precision.get("precision_ok") == 1, "PXDesign live numerics disagree with the plan", errors)
    dedup = (man.get("package_census") or {}).get("sdedup") or {}
    if mode in ("fast", "big"):
        _need(precision.get("precision_planned") == "tf32", "PXDesign tolerance mode lacks TF32 policy", errors)
        # N_sample=1 has no rows to deduplicate. Its 'single' calls are valid.
        _need(int(dedup.get("calls") or 0) > 0, "sdedup has no sampler calls", errors)
        _need(int(dedup.get("fallback") or 0) == 0, "sdedup fell back", errors)
        _need(int(dedup.get("served") or 0) + int(dedup.get("single") or 0) > 0, "sdedup has no applicable calls", errors)
    rowpipe = None
    if mode == "big":
        rowpipe = re.search(r"\[pxd_hoist v[^\]]*\+rowpipe\] prepare_cache .*\bslabs=([1-9]\d*)\b.*pair_z not resident", text)
        _need("rowpipe" in packages_applied and bool(rowpipe), "big mode lacks actual rowpipe execution", errors)
    return dict(record, levers_applied=sorted(applied), package_levers_applied=sorted(packages_applied),
                precision_ok=precision.get("precision_ok"), sdedup=dedup,
                rowpipe_slabs=int(rowpipe.group(1)) if rowpipe else None)


def validate(*, kit: str, mode: str, output_dir: Path, log_path: Path, exit_code: int,
             expected: int = 1, steps: int | None = None,
             binder_chain: str | None = None, target_chain: str | None = None,
             binder_residues: int | None = None, target_residues: int | None = None) -> dict:
    if kit not in ("genie3", "pxdesign"):
        raise ValidationError("kit must be genie3 or pxdesign")
    if mode not in (("off", "exact", "fast") if kit == "genie3" else ("off", "exact", "fast", "big")):
        raise ValidationError("mode is not supported by the selected kit")
    if expected < 1:
        raise ValidationError("expected must be positive")
    errors, artifacts = [], []
    text = log_path.read_text(encoding="utf-8")
    _need(exit_code == 0, "provider process exited nonzero", errors)
    _need("NOTHING RAN" not in text and "already complete (marker" not in text, "run skipped existing outputs", errors)
    _need("Traceback (most recent call last)" not in text, "log contains a traceback", errors)
    if mode != "off":
        _need(not re.search(rf"^\[{kit}-opt\]\s+NOT ACTIVE\b", text, re.MULTILINE),
              "requested mode was refused after its activation announcement", errors)
    manifest_path = output_dir / "opt_manifest.json"
    man = json.loads(manifest_path.read_text())
    _need(man.get("mode") == mode, "manifest mode differs from request", errors)
    steps = steps if steps is not None else (100 if kit == "genie3" else 400)
    evidence = (_genie_evidence(man, text, output_dir, mode, expected, steps, errors) if kit == "genie3"
                else _px_evidence(man, text, output_dir, mode, expected, steps, errors))
    paths = (sorted(output_dir.glob("*/pdbs/*.pdb")) + sorted(output_dir.glob("pdbs/*.pdb")) if kit == "genie3"
             else sorted(output_dir.glob("*/seed_*/predictions/*.cif")))
    _need(len(paths) == expected, "actual structure count differs from request", errors)
    _need(not any(path.is_file() for path in (output_dir / "ERR").rglob("*")), "PXDesign has task error files", errors)
    for path in paths:
        relative = path.relative_to(output_dir).as_posix()
        entry = {"file": relative, "sha256": _sha256(path), "bytes": path.stat().st_size}
        try:
            atoms = parse_atoms(path)
            entry.update(structure_summary(atoms))
            for role, chain, count in (("binder", binder_chain, binder_residues), ("target", target_chain, target_residues)):
                if chain is None:
                    continue
                info = entry["chains"].get(chain)
                _need(bool(info), f"{relative}: missing {role} chain", errors)
                if info:
                    if count is not None:
                        _need(info["residues"] == count, f"{relative}: {role} residue count differs", errors)
                    if role == "binder":
                        field = "ca_only_residues" if kit == "genie3" else "full_backbone_residues"
                        _need(info[field] == info["residues"], f"{relative}: {role} atom coverage differs from generator contract", errors)
            if binder_chain is None or target_chain is None:
                entry["chain_roles"] = "not assigned; counts reported without role inference"
        except ValidationError as exc:
            errors.append(f"{relative}: {exc}")
        artifacts.append(entry)
    return {"schema": "binder.genie_px_artifact_validation/1", "kit": kit, "mode": mode,
            "valid": not errors, "errors": errors, "expected_structures": expected,
            "manifest_sha256": _sha256(manifest_path), "log_sha256": _sha256(log_path),
            "evidence": evidence, "structures": artifacts,
            "claim_boundary": "parsed computational outputs and requested-mode evidence; no biological quality or throughput claim"}


def compare_coordinates(left: Path, right: Path) -> dict:
    a, b = parse_atoms(left), parse_atoms(right)
    same_atoms = [row[:-1] for row in a] == [row[:-1] for row in b]
    delta = max((abs(x - y) for u, v in zip(a, b) for x, y in zip(u[-1], v[-1])), default=0.0) if same_atoms else None
    return {"atom_identities_equal": same_atoms, "coordinates_equal": same_atoms and delta == 0.0,
            "max_coordinate_difference_angstrom": delta, "bytes_equal": left.read_bytes() == right.read_bytes()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, choices=("genie3", "pxdesign"))
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--expected", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--binder-chain")
    parser.add_argument("--target-chain")
    parser.add_argument("--binder-residues", type=int)
    parser.add_argument("--target-residues", type=int)
    args = parser.parse_args(argv)
    try:
        result = validate(kit=args.kit, mode=args.mode, output_dir=args.output_dir, log_path=args.log,
                          exit_code=args.exit_code, expected=args.expected, steps=args.steps,
                          binder_chain=args.binder_chain, target_chain=args.target_chain,
                          binder_residues=args.binder_residues, target_residues=args.target_residues)
    except (ValidationError, OSError, ValueError, KeyError, TypeError) as exc:
        # Missing artifacts and malformed evidence are failed validations.
        result = {"schema": "binder.genie_px_artifact_validation/1", "kit": args.kit,
                  "mode": args.mode, "valid": False, "errors": [type(exc).__name__ + ": validation input missing or malformed"]}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
