"""Read returned molecular files before treating their hashes as evidence."""
from __future__ import annotations

import math
from . import target_prep_adapter as structures


class EvidenceError(ValueError):
    """A returned file cannot substantiate its declared content."""


def structure_atoms(payload: bytes, format: str, label: str) -> list[structures.Atom]:
    try:
        text = payload.decode("utf-8")
        if format == "pdb":
            # The source reader deliberately skips malformed rows. Returned evidence
            # must not silently lose those rows while acquiring a valid hash.
            for line in text.splitlines():
                if line.startswith("ENDMDL"):
                    break
                if line.startswith(("ATOM  ", "HETATM")):
                    int(line[22:26])
                    if not all(math.isfinite(float(line[a:b])) for a, b in ((30, 38), (38, 46), (46, 54))):
                        raise ValueError("non-finite coordinates")
            atoms = structures.parse_pdb_atoms(text)
        elif format in {"cif", "mmcif"}:
            lines = text.splitlines()
            for index, line in enumerate(lines):
                if line.strip() != "loop_":
                    continue
                cursor, names = index + 1, []
                while cursor < len(lines) and lines[cursor].lstrip().startswith("_atom_site."):
                    names.append(lines[cursor].strip().split(".", 1)[1])
                    cursor += 1
                if not names:
                    continue
                columns = {name: i for i, name in enumerate(names)}
                first_model = None
                for row in structures.iter_cif_atom_rows(lines, cursor, len(names)):
                    value = lambda *keys: structures.cif_value(row, columns, *keys)
                    model = value("pdbx_PDB_model_num") or "1"
                    if first_model is None:
                        first_model = model
                    if model != first_model:
                        continue
                    int(value("auth_seq_id", "label_seq_id"))
                    if not all(math.isfinite(float(value(key))) for key in ("Cartn_x", "Cartn_y", "Cartn_z")):
                        raise ValueError("non-finite coordinates")
                    if not value("auth_atom_id", "label_atom_id") or not value("auth_comp_id", "label_comp_id"):
                        raise ValueError("incomplete atom identity")
                break
            atoms = structures.parse_cif_atoms(text)
        else:
            raise ValueError(f"unsupported structure format {format!r}")
    except (ValueError, IndexError, UnicodeError, structures.AdapterError) as exc:
        raise EvidenceError(f"{label}: cannot parse {format} coordinates: {exc}") from exc
    if not atoms:
        raise EvidenceError(f"{label}: no readable coordinate atoms")
    if any(not atom.name or not atom.residue_name or not all(math.isfinite(v) for v in (atom.x, atom.y, atom.z)) for atom in atoms):
        raise EvidenceError(f"{label}: incomplete atom identity or non-finite coordinates")
    return atoms


def chain_sequence(payload: bytes, format: str, chain: str, label: str) -> str:
    residues = {}
    for atom in structure_atoms(payload, format, label):
        if atom.chain_id != chain or atom.record != "ATOM":
            continue
        key = (atom.residue_number, atom.insertion_code)
        if key in residues and residues[key] != atom.residue_name:
            raise EvidenceError(f"{label}: conflicting residue names on chain {chain}")
        residues[key] = atom.residue_name
    if not residues:
        raise EvidenceError(f"{label}: no protein residues on chain {chain}")
    # Match the package's residue ordering, independently of atom serialization.
    return "".join(structures.THREE_TO_ONE.get(residues[key], "X") for key in sorted(residues))


def fasta_sequence(payload: bytes, label: str) -> str:
    try:
        lines = [line.strip() for line in payload.decode("utf-8").splitlines() if line.strip()]
    except UnicodeError as exc:
        raise EvidenceError(f"{label}: FASTA is not UTF-8") from exc
    if not lines or not lines[0].startswith(">") or not lines[0][1:].strip():
        raise EvidenceError(f"{label}: missing FASTA header")
    if sum(line.startswith(">") for line in lines) != 1:
        raise EvidenceError(f"{label}: expected exactly one FASTA record")
    sequence = "".join(lines[1:]).upper()
    if not sequence or set(sequence) - set(structures.THREE_TO_ONE.values()):
        raise EvidenceError(f"{label}: empty or noncanonical protein sequence")
    return sequence
