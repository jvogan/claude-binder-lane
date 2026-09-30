"""Validate real Complexa generation or ProteinMPNN design artifacts.

This is an artifact check, separate from kit activation and cloud billing.
Callers must verify the requested mode's ACTIVE/EXIT lines and manifest too.
Receipts use output-relative paths and hashes, so they can be sanitized for
sharing without copying sequences or provider metadata. NPZ checks require
NumPy, already present in both pinned kit images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


class ArtifactError(ValueError):
    """A generated artifact is absent, incomplete, or unparsable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, root: Path) -> dict:
    return {"file": path.relative_to(root).as_posix(), "sha256": _sha256(path)}


def _fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    parts: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            header, parts = line[1:], []
        elif header is None:
            raise ArtifactError(f"{path.name}: sequence before first FASTA header")
        else:
            parts.append(line)
    if header is not None:
        records.append((header, "".join(parts)))
    if not records or any(not sequence or not re.fullmatch(r"[A-Z]+(?:/[A-Z]+)*", sequence)
                          for _, sequence in records):
        raise ArtifactError(f"{path.name}: missing or invalid amino-acid sequence")
    return records


def validate_proteinmpnn(
    out: str | Path, *, designs_per_target: int, expected_targets: int | None = None,
    require_arrays: bool = True,
) -> dict:
    """Count designed FASTA records and read score/probability arrays per target.

    Native reference records are excluded from design counts. Multiple design
    temperatures should be reflected in designs_per_target. Fixed chains stay
    in the PDB handoff rather than being appended to the designed FASTA.
    """
    root = Path(out)
    if designs_per_target < 1:
        raise ValueError("designs_per_target must be positive")
    paths = sorted((root / "seqs").glob("*.fa"))
    if not paths or (expected_targets is not None and len(paths) != expected_targets):
        raise ArtifactError(f"expected {expected_targets or 'at least one'} targets; found {len(paths)}")
    results = []
    for path in paths:
        records = _fasta(path)
        native = [sequence for header, sequence in records if "sample=" not in header]
        generated = [sequence for header, sequence in records if "sample=" in header]
        if len(native) != 1 or len(generated) != designs_per_target:
            raise ArtifactError(f"{path.name}: {len(native)} native and {len(generated)} designs; "
                                f"expected one native and {designs_per_target} designs")
        chain_lengths = [len(sequence) for sequence in native[0].split("/")]
        if any([len(chain) for chain in sequence.split("/")] != chain_lengths
               for sequence in generated):
            raise ArtifactError(f"{path.name}: generated chain lengths differ from the native reference")
        arrays = []
        if require_arrays:
            import numpy as np

            for directory, keys in (("scores", ("score", "global_score")),
                                    ("probs", ("probs", "log_probs", "S", "mask", "chain_order"))):
                array_path = root / directory / f"{path.stem}.npz"
                if not array_path.is_file():
                    raise ArtifactError(f"{directory}/{path.stem}.npz is absent")
                with np.load(array_path, allow_pickle=False) as archive:
                    for key in keys:
                        if key not in archive or archive[key].size == 0:
                            raise ArtifactError(f"{array_path.name}: missing/empty {key}")
                    if directory == "scores":
                        for key in keys:
                            if archive[key].size != designs_per_target or not np.isfinite(archive[key]).all():
                                raise ArtifactError(f"{array_path.name}: invalid {key} count or non-finite score")
                    else:
                        probabilities = archive["probs"]
                        if (probabilities.ndim != 3 or probabilities.shape[0] != designs_per_target
                                or probabilities.shape[-1] != 21 or not np.isfinite(probabilities).all()):
                            raise ArtifactError(f"{array_path.name}: invalid probability dimensions or values")
                    arrays.append({**_record(array_path, root), "keys": sorted(archive.files)})
        results.append({**_record(path, root), "designs": len(generated),
                        "designed_chain_lengths": chain_lengths, "arrays": arrays})
    return {"tool": "proteinmpnn", "targets": len(results),
            "designs": sum(item["designs"] for item in results), "artifacts": results}


def validate_complexa(
    out: str | Path, *, expected_designs: int, binder_length: int | None = None,
    require_af2_reward: bool = False,
) -> dict:
    """Parse chain roles, full-backbone atom completeness, and finite coordinates."""
    root = Path(out)
    if expected_designs < 1:
        raise ValueError("expected_designs must be positive")
    paths = sorted(root.glob("inference/search_binder_local_pipeline*/job_*/*.pdb"))
    if len(paths) != expected_designs:
        raise ArtifactError(f"expected {expected_designs} designs; found {len(paths)}")
    results = []
    for path in paths:
        residues: dict[str, dict[str, set[str]]] = {}
        for line in path.read_text().splitlines():
            if not line.startswith("ATOM  "):
                continue
            if len(line) < 54:
                raise ArtifactError(f"{path.name}: truncated PDB atom record")
            atom, chain, residue = line[12:16].strip(), line[21], line[22:27]
            try:
                xyz = [float(line[start:start + 8]) for start in (30, 38, 46)]
            except ValueError as error:
                raise ArtifactError(f"{path.name}: invalid PDB coordinates") from error
            if not all(math.isfinite(value) for value in xyz):
                raise ArtifactError(f"{path.name}: non-finite PDB coordinates")
            residues.setdefault(chain, {}).setdefault(residue, set()).add(atom)
        if "A" not in residues or "B" not in residues:
            raise ArtifactError(f"{path.name}: target chain A or binder chain B is absent")
        for chain in ("A", "B"):
            if any(not {"N", "CA", "C", "O"} <= atoms for atoms in residues[chain].values()):
                raise ArtifactError(f"{path.name}: chain {chain} lacks full-backbone N/CA/C/O")
        observed = len(residues["B"])
        if binder_length is not None and observed != binder_length:
            raise ArtifactError(f"{path.name}: binder length {observed}, expected {binder_length}")
        results.append({**_record(path, root), "target_chain": "A", "binder_chain": "B",
                        "target_residues": len(residues["A"]), "binder_residues": observed})
    report = {"tool": "complexa", "designs": len(results), "artifacts": results,
              "claim_scope": "generation"}
    if require_af2_reward:
        report["af2_reward"] = validate_complexa_rewards(root, expected_designs=expected_designs)
        report["claim_scope"] = "generation_with_af2_reward"
    return report


def validate_complexa_rewards(out: str | Path, *, expected_designs: int) -> dict:
    """Require actual finite AF2 losses/confidence, since upstream catches errors.

    A total_reward column alone can hold zero even when an AF2 reward threw.
    This checks recorded scientific outputs without imposing quality cutoffs.
    JAX GPU execution, selected models/recycles, and warning-free logs remain
    separate invocation evidence owned by the caller.
    """
    root = Path(out)
    paths = sorted(root.glob("inference/search_binder_local_pipeline*/rewards_*.csv"))
    required = ("total_reward", "af2folding_i_pae", "af2folding_plddt_log",
                "af2folding_pae_log", "af2folding_i_pae_log")
    rows = []
    artifacts = []
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(required) <= set(reader.fieldnames or ()):
                raise ArtifactError(f"{path.name}: actual AF2 reward/confidence columns are absent")
            for row in reader:
                try:
                    values = {key: float(row[key]) for key in required}
                except (TypeError, ValueError) as error:
                    raise ArtifactError(f"{path.name}: invalid AF2 reward value") from error
                if not all(math.isfinite(value) for value in values.values()):
                    raise ArtifactError(f"{path.name}: non-finite AF2 reward/confidence")
                rows.append(values)
        artifacts.append(_record(path, root))
    if len(rows) != expected_designs:
        raise ArtifactError(f"expected {expected_designs} AF2-scored rows; found {len(rows)}")
    return {"scored_designs": len(rows), "required_metrics": list(required), "artifacts": artifacts}


def compare_proteinmpnn(off: str | Path, exact: str | Path) -> dict:
    """Compare FASTA bytes and every NPZ array, independent of ZIP timestamps."""
    import numpy as np

    left, right = Path(off), Path(exact)
    paths = sorted(path.relative_to(left) for directory in ("seqs", "scores", "probs")
                   for path in (left / directory).glob("*.*") if path.is_file())
    other = sorted(path.relative_to(right) for directory in ("seqs", "scores", "probs")
                   for path in (right / directory).glob("*.*") if path.is_file())
    if not paths or paths != other:
        raise ArtifactError("off and exact artifact inventories differ or are empty")
    for relative in paths:
        a, b = left / relative, right / relative
        if relative.suffix == ".npz":
            with np.load(a, allow_pickle=False) as aa, np.load(b, allow_pickle=False) as bb:
                if sorted(aa.files) != sorted(bb.files) or any(
                    not np.array_equal(aa[key], bb[key]) for key in aa.files
                ):
                    raise ArtifactError(f"{relative.as_posix()}: off/exact arrays differ")
        elif a.read_bytes() != b.read_bytes():
            raise ArtifactError(f"{relative.as_posix()}: off/exact bytes differ")
    return {"comparison": "proteinmpnn_off_exact", "identical": True,
            "files": [path.as_posix() for path in paths]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="tool", required=True)
    mpnn = sub.add_parser("proteinmpnn")
    mpnn.add_argument("--out", required=True)
    mpnn.add_argument("--designs-per-target", type=int, required=True)
    mpnn.add_argument("--expected-targets", type=int)
    mpnn.add_argument("--without-arrays", action="store_true")
    complexa = sub.add_parser("complexa")
    complexa.add_argument("--out", required=True)
    complexa.add_argument("--expected-designs", type=int, required=True)
    complexa.add_argument("--binder-length", type=int)
    complexa.add_argument("--require-af2-reward", action="store_true")
    compare = sub.add_parser("proteinmpnn-compare")
    compare.add_argument("--off", required=True)
    compare.add_argument("--exact", required=True)
    args = parser.parse_args()
    try:
        if args.tool == "complexa":
            receipt = validate_complexa(args.out, expected_designs=args.expected_designs,
                                        binder_length=args.binder_length,
                                        require_af2_reward=args.require_af2_reward)
        elif args.tool == "proteinmpnn":
            receipt = validate_proteinmpnn(args.out, designs_per_target=args.designs_per_target,
                                           expected_targets=args.expected_targets,
                                           require_arrays=not args.without_arrays)
        else:
            receipt = compare_proteinmpnn(args.off, args.exact)
    except (ArtifactError, OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1
    print(json.dumps({"ok": True, **receipt}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
