"""Independently validate RFdiffusion3 and BoltzGen kit outputs and activation.

No compute is launched. Counts, finite coordinates, chain roles, output hashes,
applied-mode evidence and execution counters are checked from retrieved files.
The atom-site reader handles the ordinary mmCIF tables these generators emit.
Full BoltzGen checks also read NPZ prediction arrays through NumPy.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shlex


class ArtifactError(ValueError):
    pass


AA = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV",
))
BG_LEVERS = {
    "exact": {"inproc", "graph_sampler", "fastinit", "hoist", "async_writer"},
    "fast": {"inproc", "graph_sampler", "fastinit", "hoist", "async_writer",
             "cond_dedup", "attn_bf16", "attn_cudnn", "dit_fused"},
    "big": {"inproc", "fastinit", "td_chunk", "async_writer"},
}
BG_STEPS = ["design", "inverse_folding", "folding", "design_folding", "analysis", "filtering"]
GPU_STEPS = BG_STEPS[:4]


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ArtifactError(f"{path.name}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, root: Path) -> dict:
    return {"file": path.relative_to(root).as_posix(), "sha256": _sha256(path)}


def _atoms(path: Path) -> list[dict]:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt") as stream:
            text = stream.read()
    else:
        text = path.read_text()
    lines, result, i = text.splitlines(), [], 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        columns = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            columns.append(lines[i].strip())
            i += 1
        if not columns or not all(c.startswith("_atom_site.") for c in columns):
            continue
        pending = []
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith(("#", "_", "data_")) or line == "loop_":
                break
            if line.startswith(";"):
                raise ArtifactError(f"{path.name}: unsupported multiline atom-site field")
            i += 1
            if not line:
                continue
            pending.extend(shlex.split(line))
            while len(pending) >= len(columns):
                result.append(dict(zip(columns, pending[:len(columns)])))
                pending = pending[len(columns):]
        if pending:
            raise ArtifactError(f"{path.name}: incomplete atom-site row")
    if not result:
        raise ArtifactError(f"{path.name}: no atom-site records")
    return result


def _field(row: dict, *names: str) -> str:
    for name in names:
        value = row.get("_atom_site." + name)
        if value not in (None, ".", "?"):
            return value
    raise ArtifactError(f"atom-site record lacks {'/'.join(names)}")


def _structure(path: Path, *, binder_chain: str, target_chain: str | None,
               binder_min: int | None = None, binder_max: int | None = None) -> dict:
    if target_chain == binder_chain:
        raise ValueError("target_chain and binder_chain must differ")
    chains: dict[str, dict[str, dict]] = {}
    models = set()
    for row in _atoms(path):
        if row.get("_atom_site.group_PDB", "ATOM") != "ATOM":
            continue
        models.add(row.get("_atom_site.pdbx_PDB_model_num", "1"))
        xyz = [float(_field(row, f"Cartn_{axis}")) for axis in "xyz"]
        if not all(math.isfinite(v) for v in xyz):
            raise ArtifactError(f"{path.name}: non-finite atom coordinates")
        chain = _field(row, "label_asym_id", "auth_asym_id")
        residue = _field(row, "label_seq_id", "auth_seq_id")
        atom = _field(row, "label_atom_id", "auth_atom_id")
        comp = _field(row, "label_comp_id", "auth_comp_id")
        entry = chains.setdefault(chain, {}).setdefault(residue, {"atoms": set(), "comp": comp})
        if entry["comp"] != comp:
            raise ArtifactError(f"{path.name}: inconsistent residue identity")
        entry["atoms"].add(atom)
    if len(models) != 1:
        raise ArtifactError(f"{path.name}: expected one coordinate model")
    for chain in [binder_chain] + ([target_chain] if target_chain else []):
        if chain not in chains:
            raise ArtifactError(f"{path.name}: required chain {chain!r} is absent")
        if any(not {"N", "CA", "C", "O"} <= r["atoms"] for r in chains[chain].values()):
            raise ArtifactError(f"{path.name}: chain {chain!r} lacks full-backbone N/CA/C/O")
    count = len(chains[binder_chain])
    if (binder_min is not None and count < binder_min) or (binder_max is not None and count > binder_max):
        raise ArtifactError(f"{path.name}: binder residue count {count} is outside the requested range")
    sequences = {c: "".join(AA.get(r["comp"], "X") for r in entries.values())
                 for c, entries in chains.items()}
    sequence = sequences[binder_chain]
    return {"chains": {c: len(r) for c, r in chains.items()}, "binder_chain": binder_chain,
            "target_chain": target_chain, "binder_residues": count, "_sequence": sequence,
            "_target_sequence": sequences.get(target_chain),
            "chain_sequence_sha256": {c: hashlib.sha256(s.encode()).hexdigest() for c, s in sequences.items()},
            "binder_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()}


def _public_structure(record: dict) -> dict:
    return {k: v for k, v in record.items() if not k.startswith("_")}


def _target_sequence(path: str | Path | None, chain: str | None) -> str | None:
    if path is None:
        return None
    if chain is None:
        raise ValueError("target_input_chain is required with target_input")
    path = Path(path)
    residues = {}
    for line in path.read_text().splitlines():
        if line.startswith("ATOM  ") and len(line) >= 54 and line[21] == chain:
            residues[line[22:27]] = line[17:20].strip()
    if not residues:
        raise ArtifactError(f"{path.name}: selected target chain is absent")
    return "".join(AA.get(comp, "X") for comp in residues.values())


def _activation(man: dict, log: str, mode: str, tag: str) -> None:
    if man.get("mode") != mode or man.get("exit_code") != 0:
        raise ArtifactError("manifest mode or exit code does not match a completed requested run")
    if mode == "off":
        if man.get("active"):
            raise ArtifactError("stock run unexpectedly reports active acceleration")
        return
    if not man.get("active") or man.get("partial") or man.get("levers_fallback") or man.get("levers_unavailable"):
        raise ArtifactError("requested mode was inactive, partial, or missing a lever")
    if not re.search(r"\[" + re.escape(tag) + r"\] ACTIVE mode=" + re.escape(mode) + r"\b", log):
        raise ArtifactError("requested ACTIVE announcement is absent")
    if f"[{tag}] NOT ACTIVE:" in log:
        raise ArtifactError("accelerated run contains a NOT ACTIVE refusal")


def validate_rfdiffusion3(out: str | Path, *, log: str | Path, mode: str,
                         expected_designs: int, binder_chain: str, target_chain: str,
                         binder_min: int | None = None, binder_max: int | None = None,
                         target_input: str | Path | None = None,
                         target_input_chain: str | None = None) -> dict:
    root = Path(out)
    if mode not in ("off", "exact", "fast") or expected_designs < 1:
        raise ValueError("invalid RFdiffusion3 mode or expected design count")
    man = _read_json(root / "opt_manifest.json")
    text = Path(log).read_text()
    _activation(man, text, mode, "rfdiffusion3-opt")
    kernels = man.get("kernels") or {}
    if kernels.get("refused") or not kernels.get("items"):
        raise ArtifactError("RFdiffusion3 has no completed rollout census or refused a kernel")
    words = kernels.get("words") or {}
    for key, kind in (kernels.get("expected") or {}).items():
        if str(words.get(key, "")).split(":", 1)[0] != kind:
            raise ArtifactError(f"RFdiffusion3 kernel {key} did not execute its expected route")
    if not str(words.get("rmsnorm", "")).startswith("engaged:apex."):
        raise ArtifactError("RFdiffusion3 did not use the required compiled apex RMSNorm")
    counters = {}
    if mode == "off":
        if not (man.get("stock_env_proof") or {}).get("ok"):
            raise ArtifactError("RFdiffusion3 stock environment proof failed")
    else:
        if man.get("allow_partial") or (man.get("interpreter") or {}).get("label") != "patched":
            raise ArtifactError("RFdiffusion3 did not use the overlaid interpreter as a complete mode")
        exits = [line for line in text.splitlines() if "[rfdiffusion3-opt] EXIT " in line]
        for line in exits:
            counters.update({key: int(value) for key, value in
                             re.findall(r"\b([\w.]+)=(-?\d+)\b", line)})
        for key in ("stats.rollouts", "fzt.fused_calls", "init_chunk.calls"):
            if counters.get(key, 0) < 1:
                raise ArtifactError(f"RFdiffusion3 has no execution counter for {key}")
        if counters.get("graph.captures", 0) + counters.get("graph.hits", 0) < 1:
            raise ArtifactError("RFdiffusion3 CUDA graph was not exercised")
        if counters.get("graph.fallbacks", 0) or counters.get("graph.declined", 0):
            raise ArtifactError("RFdiffusion3 graph fell back or declined a rollout")
        if mode == "fast" and (counters.get("compile.graphs", 0) < 1 or counters.get("gather.served", 0) < 1):
            raise ArtifactError("RFdiffusion3 fast compilation or gather kernel was not exercised")
    paths = sorted(root.glob("*.cif.gz"))
    if len(paths) != expected_designs:
        raise ArtifactError(f"expected {expected_designs} RFdiffusion3 designs, found {len(paths)}")
    listed = {entry["file"]: entry for entry in man.get("outputs") or []}
    target_sequence = _target_sequence(target_input, target_input_chain)
    results = []
    for path in paths:
        companion = path.with_name(path.name.removesuffix(".cif.gz") + ".json")
        _read_json(companion)
        for artifact in (path, companion):
            entry = listed.get(artifact.name) or {}
            if entry.get("sha256") != _sha256(artifact) or entry.get("bytes") != artifact.stat().st_size:
                raise ArtifactError(f"{artifact.name}: bytes or hash differ from the kit manifest")
        structure = _structure(path, binder_chain=binder_chain, target_chain=target_chain,
                               binder_min=binder_min, binder_max=binder_max)
        if target_sequence is not None and structure["_target_sequence"] != target_sequence:
            raise ArtifactError(f"{path.name}: target sequence differs from the staged target input")
        results.append({**_record(path, root), **_public_structure(structure),
                        "metadata": _record(companion, root)})
    return {"tool": "rfdiffusion3", "mode": mode, "designs": len(results),
            "counters": counters, "artifacts": results}


def _npz(path: Path, root: Path) -> dict:
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        if not archive.files:
            raise ArtifactError(f"{path.name}: empty prediction array archive")
        for key in ("coords", "res_type"):
            if key not in archive or not archive[key].size or not np.isfinite(archive[key]).all():
                raise ArtifactError(f"{path.name}: absent, empty, or non-finite {key} array")
        if archive["coords"].shape[-1] != 3:
            raise ArtifactError(f"{path.name}: coordinates lack a three-dimensional axis")
        confidence = [key for key in ("iptm", "ptm", "plddt", "design_ptm", "design_to_target_iptm")
                      if key in archive]
        if not confidence or any(not np.isfinite(archive[key]).all() for key in confidence):
            raise ArtifactError(f"{path.name}: no finite confidence arrays")
        return {**_record(path, root), "keys": sorted(archive.files)}


def _smoke_defaults(root: Path, *, full_pipeline: bool) -> None:
    for step in (GPU_STEPS if full_pipeline else ["design"]):
        text = (root / "config" / f"{step}.yaml").read_text()
        # Resolved upstream step configs put these numeric settings at top level.
        for key, value in {"sampling_steps": 500 if step == "design" else 200,
                           "recycling_steps": 3}.items():
            if not re.search(r"^" + key + r":\s*" + str(value) + r"\s*$", text, re.M):
                raise ArtifactError(f"{step}: {key} changed from the qualification's upstream default")
    if full_pipeline:
        filtering = (root / "config/filtering.yaml").read_text()
        if not re.search(r"^budget:\s*30\s*$", filtering, re.M):
            raise ArtifactError("filtering budget changed from upstream's default of 30")


def validate_boltzgen(out: str | Path, *, log: str | Path, mode: str,
                     expected_designs: int, binder_chain: str, target_chain: str,
                     binder_min: int | None = None, binder_max: int | None = None,
                     full_pipeline: bool = False, check_smoke_defaults: bool = False,
                     target_input: str | Path | None = None,
                     target_input_chain: str | None = None) -> dict:
    root = Path(out)
    if mode not in ("off", "exact", "fast", "big") or expected_designs < 1:
        raise ValueError("invalid BoltzGen mode or expected design count")
    man = _read_json(root / "opt_manifest.json")
    text = Path(log).read_text()
    _activation(man, text, mode, "boltzgen-opt")
    census = man.get("designs") or {}
    if census.get("requested") != expected_designs or census.get("produced") != expected_designs:
        raise ArtifactError("BoltzGen requested or produced design count is incomplete")
    if any(census.get(key) for key in ("oom_skipped", "featurizer_skipped", "stale", "reuse")):
        raise ArtifactError("BoltzGen skipped a batch or reused stale designs")
    configured = (man.get("configure") or {}).get("steps")
    if configured != (BG_STEPS if full_pipeline else ["design"]):
        raise ArtifactError("BoltzGen configured steps differ from the requested qualification")
    arrays, counters = [], {}
    if mode == "off":
        steps = (man.get("run") or {}).get("steps") or []
        if [s.get("step") for s in steps] != configured or any(
                s.get("rc") != 0 or not s.get("proof_ok") for s in steps):
            raise ArtifactError("BoltzGen stock steps lack completed clean-environment proofs")
    else:
        if not BG_LEVERS[mode] <= set(man.get("levers_applied") or []):
            raise ArtifactError("BoltzGen did not apply every lever of its requested mode")
        kernels = (man.get("activation_report") or {}).get("kernels") or {}
        if kernels.get("ok") is not True or not kernels.get("lines"):
            raise ArtifactError("BoltzGen kernel census has findings or no observed calls")
        for key in ("cueq_triatt", "cueq_trimul"):
            if not str((kernels.get("words") or {}).get(key, "")).startswith("engaged:"):
                raise ArtifactError(f"BoltzGen {key} did not engage")
        times = _read_json(root / "inproc_times.json")
        expected_gpu = GPU_STEPS if full_pipeline else ["design"]
        seed = man.get("seed")
        if set(times) != set(expected_gpu) or any(
                t.get("seed") != seed + configured.index(step) or t.get("wall_s", 0) <= 0
                for step, t in times.items()):
            raise ArtifactError("BoltzGen GPU step timing or seed evidence is incomplete")
        if full_pipeline and any(
                f"[xa_run] step {step} (stock subprocess) rc=0" not in text
                for step in ("analysis", "filtering")):
            raise ArtifactError("BoltzGen CPU analysis/filtering steps did not complete")
        records = [json.loads(line) for line in (root / "opt_timing.jsonl").read_text().splitlines() if line.strip()]
        graphs = [r.get("graph_stats") or {} for r in records]
        counters = {"captures": max((g.get("captures", 0) for g in graphs), default=0),
                    "replays": max((g.get("replays", 0) for g in graphs), default=0),
                    "graph_calls": max((g.get("graph_calls", 0) for g in graphs), default=0)}
        if mode in ("exact", "fast") and (any(v < 1 for v in counters.values()) or any(
                g.get("capture_error") or g.get("predraw_calls", 0) for g in graphs)):
            raise ArtifactError("BoltzGen graph sampler was not exercised completely")
        stats = (man.get("activation_report") or {}).get("kit_stats_lines") or {}
        fastinit = ast.literal_eval(stats.get("xa_fastinit", "{}"))
        if (fastinit.get("skipped", 0) < 1 or fastinit.get("numel", 0) < 1
                or fastinit.get("fallback_replayed", 0) or fastinit.get("fallback_reason")):
            raise ArtifactError("BoltzGen fast initialization was not exercised completely")
        counters["initializers_skipped"] = fastinit["skipped"]
        if mode in ("exact", "fast"):
            hoist = ast.literal_eval(stats.get("xa_hoist", "{}"))
            if (hoist.get("builds", 0) < 1
                    or hoist.get("uses", 0) + hoist.get("capture_uses", 0) < 1
                    or hoist.get("selfcheck_fail", 0) or hoist.get("disabled_reason")):
                raise ArtifactError("BoltzGen mask hoisting was not exercised completely")
            counters["hoist_uses"] = hoist.get("uses", 0) + hoist.get("capture_uses", 0)
    paths = sorted(p for p in (root / "intermediate_designs").glob("*.cif")
                   if not p.name.endswith("_native.cif"))
    if len(paths) != expected_designs:
        raise ArtifactError(f"expected {expected_designs} generated BoltzGen complexes, found {len(paths)}")
    results = []
    target_sequence = _target_sequence(target_input, target_input_chain)
    for path in paths:
        result = _structure(path, binder_chain=binder_chain, target_chain=target_chain,
                            binder_min=binder_min, binder_max=binder_max)
        if target_sequence is not None and result["_target_sequence"] != target_sequence:
            raise ArtifactError(f"{path.name}: target sequence differs from the staged target input")
        results.append({**_record(path, root), **_public_structure(result)})
    if check_smoke_defaults:
        _smoke_defaults(root, full_pipeline=full_pipeline)
    metrics = []
    if full_pipeline:
        refined = root / "intermediate_designs_inverse_folded"
        refined_paths = sorted(p for p in refined.glob("*.cif") if not p.name.endswith("_native.cif"))
        if len(refined_paths) != expected_designs:
            raise ArtifactError("BoltzGen inverse-folded design count is incomplete")
        sequences = {}
        for path in refined_paths:
            structure = _structure(path, binder_chain=binder_chain, target_chain=target_chain)
            if target_sequence is not None and structure["_target_sequence"] != target_sequence:
                raise ArtifactError("BoltzGen inverse folding changed the target sequence")
            sequences[path.name] = structure["_sequence"]
            for folder in ("fold_out_npz", "fold_out_design_npz"):
                arrays.append(_npz(refined / folder / f"{path.stem}.npz", root))
            complex_path = refined / "refold_cif" / path.name
            refolded = _structure(complex_path, binder_chain=binder_chain, target_chain=target_chain)
            if refolded["_sequence"] != sequences[path.name]:
                raise ArtifactError("BoltzGen complex refolding changed the designed sequence")
            if target_sequence is not None and refolded["_target_sequence"] != target_sequence:
                raise ArtifactError("BoltzGen complex refolding changed the target sequence")
            binder_path = refined / "refold_design_cif" / path.name
            # Binder-only folding can relabel its sole chain. Identify that
            # chain from the returned structure and bind it by its sequence.
            binder_chains = {_field(row, "label_asym_id", "auth_asym_id") for row in _atoms(binder_path)
                             if row.get("_atom_site.group_PDB", "ATOM") == "ATOM"}
            if len(binder_chains) != 1:
                raise ArtifactError("BoltzGen binder-only refolding has an ambiguous chain mapping")
            binder_folded = _structure(binder_path, binder_chain=next(iter(binder_chains)), target_chain=None)
            if binder_folded["_sequence"] != sequences[path.name]:
                raise ArtifactError("BoltzGen binder-only refolding changed the designed sequence")
        tables = sorted(refined.glob("aggregate_metrics_*.csv"))
        if len(tables) != 1:
            raise ArtifactError("BoltzGen has no unambiguous aggregated score table")
        with tables[0].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected_designs or {r.get("file_name") for r in rows} != set(sequences):
            raise ArtifactError("BoltzGen analysis rows do not match inverse-folded candidates")
        for row in rows:
            if row.get("designed_chain_sequence") != sequences[row["file_name"]]:
                raise ArtifactError("BoltzGen score row sequence differs from its designed complex")
            for key in ("bb_rmsd", "designfolding-bb_rmsd"):
                if key not in row or not math.isfinite(float(row[key])):
                    raise ArtifactError(f"BoltzGen analysis lacks a finite {key}")
        ranked = root / "final_ranked_designs/all_designs_metrics.csv"
        with ranked.open(newline="") as stream:
            ranked_rows = list(csv.DictReader(stream))
        if len(ranked_rows) != expected_designs:
            raise ArtifactError("BoltzGen final ranking dropped analyzed candidates")
        # Ranking includes candidates that fail filters. Preserve actual flags.
        passed = sum(r.get("pass_filters", "").lower() == "true" for r in ranked_rows)
        metrics = [{**_record(tables[0], root), "rows": len(rows)},
                   {**_record(ranked, root), "rows": len(ranked_rows), "filters_passed": passed}]
    return {"tool": "boltzgen", "mode": mode, "designs": len(results),
            "full_pipeline": full_pipeline, "steps": configured, "counters": counters,
            "artifacts": results, "prediction_arrays": arrays, "metric_tables": metrics}


def compare_structure_outputs(off: str | Path, other: str | Path, *, tool: str,
                              full_pipeline: bool = False) -> dict:
    """Compare serialized atom records; gzip timestamps are outside this claim.

    Full BoltzGen comparison also requires equality of each prediction array.
    Call this after validating each run independently. Equal serialized atoms
    establish equality at the output precision, not hidden model tensors.
    """
    left, right = Path(off), Path(other)
    if tool == "rfdiffusion3":
        patterns = ["*.cif.gz"]
    elif tool == "boltzgen":
        patterns = ["intermediate_designs/*.cif"]
        if full_pipeline:
            patterns += ["intermediate_designs_inverse_folded/*.cif",
                         "intermediate_designs_inverse_folded/refold_cif/*.cif",
                         "intermediate_designs_inverse_folded/refold_design_cif/*.cif"]
    else:
        raise ValueError("unknown structure comparison tool")
    inventory = lambda root: sorted(p.relative_to(root) for pattern in patterns
                                     for p in root.glob(pattern) if not p.name.endswith("_native.cif"))
    paths = inventory(left)
    if not paths or paths != inventory(right):
        raise ArtifactError("structure comparison inventories differ or are empty")
    for relative in paths:
        if _atoms(left / relative) != _atoms(right / relative):
            raise ArtifactError(f"{relative.name}: serialized off/exact atom records differ")
    arrays = []
    if tool == "boltzgen" and full_pipeline:
        import numpy as np
        for folder in ("fold_out_npz", "fold_out_design_npz"):
            directory = Path("intermediate_designs_inverse_folded") / folder
            relatives = sorted(p.relative_to(left) for p in (left / directory).glob("*.npz"))
            if not relatives or relatives != sorted(p.relative_to(right) for p in (right / directory).glob("*.npz")):
                raise ArtifactError("off/exact prediction array inventories differ")
            for relative in relatives:
                with np.load(left / relative, allow_pickle=False) as a, np.load(right / relative, allow_pickle=False) as b:
                    if sorted(a.files) != sorted(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                        raise ArtifactError(f"{relative.name}: off/exact prediction arrays differ")
                arrays.append(relative.as_posix())
    return {"serialized_atom_records_identical": True, "structures": [p.as_posix() for p in paths],
            "prediction_arrays_identical": True if arrays else None, "prediction_arrays": arrays}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=("rfdiffusion3", "boltzgen"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--expected-designs", type=int, required=True)
    parser.add_argument("--binder-chain", required=True)
    parser.add_argument("--target-chain", required=True)
    parser.add_argument("--binder-min", type=int)
    parser.add_argument("--binder-max", type=int)
    parser.add_argument("--target-input")
    parser.add_argument("--target-input-chain")
    parser.add_argument("--full-pipeline", action="store_true")
    parser.add_argument("--check-smoke-defaults", action="store_true")
    parser.add_argument("--reference-off")
    args = parser.parse_args()
    values = vars(args)
    tool = values.pop("tool")
    reference = values.pop("reference_off")
    full_pipeline = values["full_pipeline"]
    if tool == "rfdiffusion3":
        values.pop("full_pipeline")
        values.pop("check_smoke_defaults")
    try:
        result = (validate_rfdiffusion3 if tool == "rfdiffusion3" else validate_boltzgen)(**values)
        if reference:
            result["comparison"] = compare_structure_outputs(
                reference, args.out, tool=tool, full_pipeline=full_pipeline
            )
    except (ArtifactError, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1
    print(json.dumps({"ok": True, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
