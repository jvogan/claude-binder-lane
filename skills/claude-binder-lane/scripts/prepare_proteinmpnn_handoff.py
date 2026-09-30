"""Prepare and verify an explicit generated-complex -> full-backbone MPNN handoff.

This stdlib-only controller launches no cloud jobs. Preparation invokes the
pinned native ProteinMPNN parser (whose environment needs NumPy), checks its
output independently, and writes portable input files and command arguments.
PDB inputs are copied byte for byte. CIF inputs undergo a declared format
conversion with residue identities, source hashes and measured rounding loss.
Neither step reconstructs atoms, samples structures nor changes sequences.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import zipfile


PARSER_SHA256 = "3d68c7fcc37c0763c42bfaf6099b4b2f5dd30710f9870b6b38d7a9e1ac18b790"
KIT_COMMIT = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
MPNN_COMMIT = "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
AMINO = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    "ACDEFGHIKLMNPQRSTVWY"))
AMINO["UNK"] = "X"
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
BACKBONE = ("N", "CA", "C", "O")


class HandoffError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cif_tokens(text: str) -> list[str]:
    out, i = [], 0
    while i < len(text):
        if text[i].isspace():
            i += 1
        elif text[i] == "#":
            end = text.find("\n", i)
            i = len(text) if end < 0 else end + 1
        elif text[i] == ";" and (i == 0 or text[i - 1] == "\n"):
            end = text.find("\n;", i + 1)
            if end < 0:
                raise HandoffError("unterminated CIF text field")
            out.append(text[i + 1:end])
            i = end + 2
        elif text[i] in ("'", '"'):
            quote, start = text[i], i + 1
            i += 1
            while i < len(text):
                if text[i] == quote and (i + 1 == len(text) or text[i + 1].isspace()):
                    out.append(text[start:i])
                    i += 1
                    break
                i += 1
            else:
                raise HandoffError("unterminated CIF quoted field")
        else:
            start = i
            while i < len(text) and not text[i].isspace():
                i += 1
            out.append(text[start:i])
    return out


def _cif_rows(text: str) -> list[dict]:
    tokens, i = _cif_tokens(text), 0
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
            raise HandoffError("CIF loop has no columns")
        values = []
        while i < len(tokens):
            word = tokens[i]
            if (word.startswith(("_", "data_", "save_")) or word in ("loop_", "stop_", "global_")) and len(values) % len(columns) == 0:
                break
            values.append(word)
            i += 1
        if any(c.startswith("_atom_site.") for c in columns):
            if not all(c.startswith("_atom_site.") for c in columns) or len(values) % len(columns):
                raise HandoffError("malformed atom-site CIF loop")
            return [dict(zip(columns, values[j:j + len(columns)])) for j in range(0, len(values), len(columns))]
    raise HandoffError("CIF has no atom-site loop")


def _v(row: dict, *names: str, default="") -> str:
    return next((row[n] for n in names if row.get(n) not in (None, "", ".", "?")), default)


def read_atoms(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        text = stream.read()
    is_pdb = path.name.lower().endswith(".pdb")
    atoms = []
    if is_pdb:
        model = "1"
        for line in text.splitlines():
            if line.startswith("MODEL "):
                model = line[10:14].strip()
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if len(line) < 54:
                raise HandoffError("truncated PDB atom")
            atoms.append(dict(model=model, chain=line[21], label_chain=line[21], auth_chain=line[21],
                              residue=line[22:26].strip(), label_residue=line[22:26].strip(),
                              auth_residue=line[22:26].strip(), insertion=line[26].strip(),
                              name=line[12:16].strip(), alt=line[16].strip(), resname=line[17:20],
                              xyz=tuple(float(line[x:x + 8]) for x in (30, 38, 46))))
    else:
        for row in _cif_rows(text):
            if row.get("_atom_site.group_PDB", "ATOM") not in ("ATOM", "HETATM"):
                continue
            chain = _v(row, "_atom_site.label_asym_id", "_atom_site.auth_asym_id")
            residue = _v(row, "_atom_site.label_seq_id", "_atom_site.auth_seq_id")
            atoms.append(dict(model=_v(row, "_atom_site.pdbx_PDB_model_num", default="1"), chain=chain,
                              label_chain=_v(row, "_atom_site.label_asym_id"), auth_chain=_v(row, "_atom_site.auth_asym_id"),
                              residue=residue, label_residue=_v(row, "_atom_site.label_seq_id"),
                              auth_residue=_v(row, "_atom_site.auth_seq_id"), insertion=_v(row, "_atom_site.pdbx_PDB_ins_code"),
                              name=_v(row, "_atom_site.label_atom_id", "_atom_site.auth_atom_id"), alt=_v(row, "_atom_site.label_alt_id"),
                              resname=_v(row, "_atom_site.label_comp_id", "_atom_site.auth_comp_id"),
                              xyz=tuple(float(row[f"_atom_site.Cartn_{a}"]) for a in "xyz")))
    if not atoms or len({a["model"] for a in atoms}) != 1:
        raise HandoffError("handoff requires atoms from exactly one model")
    if any(a["alt"] for a in atoms):
        raise HandoffError("alternate atoms need an explicitly resolved upstream structure")
    identities = [(a["chain"], a["residue"], a["insertion"], a["name"]) for a in atoms]
    if len(identities) != len(set(identities)):
        raise HandoffError("duplicate atom identity")
    if any(not all(math.isfinite(x) for x in a["xyz"]) for a in atoms):
        raise HandoffError("non-finite source coordinates")
    return atoms


def census(atoms: list[dict]) -> dict:
    chains = defaultdict(dict)
    for atom in atoms:
        if atom["resname"] not in AMINO:
            raise HandoffError("noncanonical residue requires an explicit upstream handling policy")
        try:
            number = int(atom["residue"])
        except ValueError as exc:
            raise HandoffError("residue number is not representable by native parser") from exc
        key = (number, atom["insertion"])
        residue = chains[atom["chain"]].setdefault(key, {"resname": atom["resname"], "atoms": {}, "identity": atom})
        if residue["resname"] != atom["resname"]:
            raise HandoffError("inconsistent residue name")
        residue["atoms"][atom["name"]] = atom["xyz"]
    for chain, residues in chains.items():
        numbers = {key[0] for key in residues}
        if numbers != set(range(min(numbers), max(numbers) + 1)):
            raise HandoffError("numbering gap would inject native-parser missing residues")
        for residue in residues.values():
            if not set(BACKBONE) <= set(residue["atoms"]):
                raise HandoffError("full-backbone handoff requires N, CA, C, O on every residue")
    return dict(chains)


def _write_pdb(atoms: list[dict], destination: Path, chain_map: dict) -> float:
    if len(atoms) > 99999:
        raise HandoffError("atom count exceeds PDB serial capacity")
    lines, max_delta = [], 0.0
    for serial, atom in enumerate(atoms, 1):
        chain, number, insertion, name = chain_map[atom["chain"]], int(atom["residue"]), atom["insertion"], atom["name"]
        if not -999 <= number <= 9999 or len(insertion) > 1 or len(name) > 4:
            raise HandoffError("source identity exceeds PDB field capacity")
        coordinates = [f"{x:8.3f}" for x in atom["xyz"]]
        if any(len(x) != 8 for x in coordinates):
            raise HandoffError("source coordinate exceeds PDB field capacity")
        delta = max(abs(old - float(new)) for old, new in zip(atom["xyz"], coordinates))
        max_delta = max(max_delta, delta)
        atom_field = f" {name:<3}" if len(name) < 4 else name
        lines.append(f"ATOM  {serial:5d} {atom_field} {atom['resname']:>3} {chain}{number:4d}{insertion or ' '}   "
                     f"{''.join(coordinates)}  1.00  0.00\n")
    destination.write_text("".join(lines) + "END\n", encoding="utf-8")
    return max_delta


def _verify_conversion(original: list[dict], converted: list[dict], mapping: dict) -> dict:
    before = {(mapping[a["chain"]], int(a["residue"]), a["insertion"], a["name"]): a for a in original}
    after = {(a["chain"], int(a["residue"]), a["insertion"], a["name"]): a for a in converted}
    if len(before) != len(original) or len(after) != len(converted) or set(before) != set(after):
        raise HandoffError("format conversion changed atom identities")
    max_axis, max_displacement = 0.0, 0.0
    for identity, old in before.items():
        new = after[identity]
        if old["resname"] != new["resname"]:
            raise HandoffError("format conversion changed a residue outside declared placeholder map")
        differences = [abs(x - y) for x, y in zip(old["xyz"], new["xyz"])]
        if any(d > 0.000500001 for d in differences):
            raise HandoffError("format conversion changed coordinates beyond PDB precision")
        max_axis = max(max_axis, *differences)
        max_displacement = max(max_displacement, math.sqrt(sum(d * d for d in differences)))
    return {"atom_identities_independently_matched": len(before),
            "max_coordinate_rounding_angstrom": max_axis,
            "max_atom_displacement_angstrom": max_displacement}


def _verify_parser(parsed: dict, atoms: list[dict], candidate: str) -> None:
    chains = census(atoms)
    if parsed.get("name") != candidate or parsed.get("num_of_chains") != len(chains):
        raise HandoffError("native parser candidate or chain census differs")
    parsed_chains = {k.removeprefix("seq_chain_") for k in parsed if k.startswith("seq_chain_")}
    if parsed_chains != set(chains):
        raise HandoffError("native parser dropped or added a chain")
    for chain, residues in chains.items():
        ordered = [residues[key] for key in sorted(residues)]
        sequence = "".join(AMINO[r["resname"]] for r in ordered)
        native_sequence = "".join("-" if r["resname"] == "UNK" else AMINO[r["resname"]] for r in ordered)
        if parsed[f"seq_chain_{chain}"] != native_sequence:
            raise HandoffError("native parser sequence differs from observed residues")
        for name in BACKBONE:
            actual = parsed[f"coords_chain_{chain}"][f"{name}_chain_{chain}"]
            expected = [list(r["atoms"][name]) for r in ordered]
            if actual != expected:
                raise HandoffError("native parser coordinates differ from PDB backbone")
        # Stock parser labels an observed UNK as '-', the same symbol it uses
        # for absent residues. Full backbone and no-gap checks above establish
        # these are observed unknowns. MPNN's model alphabet requires X.
        parsed[f"seq_chain_{chain}"] = sequence
    parsed["seq"] = "".join(parsed[key] for key in parsed if key.startswith("seq_chain_"))


def prepare(source: Path, output: Path, *, candidate: str, source_tool: str,
            generator_mode: str, binder_chains: list[str], target_chains: list[str],
            native_parser: Path, parser_python: str = sys.executable,
            chain_map: dict | None = None, generator_manifest: Path | None = None,
            binder_placeholder_map: dict | None = None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", candidate):
        raise HandoffError("candidate must be a portable explicit identifier")
    if source_tool.lower() == "genie3":
        raise HandoffError("Genie3 C-alpha output does not satisfy full-backbone kit contract")
    if sha256(native_parser) != PARSER_SHA256:
        raise HandoffError("native parser differs from pinned release")
    atoms = read_atoms(source)
    binder, target = set(binder_chains), set(target_chains)
    if not binder or not target or binder & target or binder | target != {a["chain"] for a in atoms}:
        raise HandoffError("explicit binder/target roles must partition all observed chains")
    placeholders = binder_placeholder_map or {}
    if any(v != "UNK" or k in AMINO for k, v in placeholders.items()):
        raise HandoffError("placeholder mapping can only declare an unknown generator code as UNK")
    mapped_atoms = 0
    for atom in atoms:
        atom["source_resname"] = atom["resname"]
        if atom["resname"] in placeholders:
            if atom["chain"] not in binder:
                raise HandoffError("unknown placeholder mapping is forbidden on fixed target")
            atom["resname"] = "UNK"
            mapped_atoms += 1
        elif atom["resname"] == "UNK" and atom["chain"] in target:
            raise HandoffError("fixed target contains unknown residues")
    chains = census(atoms)
    mapping = chain_map or {chain: chain for chain in chains}
    if set(mapping) != set(chains) or len(set(mapping.values())) != len(chains) or any(not re.fullmatch(r"[A-Za-z0-9]", str(c)) for c in mapping.values()):
        raise HandoffError("explicit chain map must assign each source chain a unique PDB character")
    if output.exists():
        raise HandoffError("refusing to overwrite an existing handoff directory")
    output.mkdir(parents=True)
    (output / "pdbs").mkdir()
    extension = ".pdb" if source.name.lower().endswith(".pdb") else ".cif.gz" if source.suffix == ".gz" else ".cif"
    raw = output / ("source" + extension)
    shutil.copyfile(source, raw)
    pdb = output / "pdbs" / f"{candidate}.pdb"
    if extension == ".pdb":
        if any(mapping[c] != c for c in chains) or mapped_atoms:
            raise HandoffError("byte-preserved PDB cannot apply chain or residue-name remapping")
        shutil.copyfile(source, pdb)
        delta, conversion = 0.0, "byte-preserved-pdb"
    else:
        delta = _write_pdb(atoms, pdb, mapping)
        conversion = "explicit-cif-to-pdb-3-decimal-coordinates"
    converted = read_atoms(pdb)
    _ = census(converted)
    conversion_proof = _verify_conversion(atoms, converted, mapping)
    parser_out = output / "parsed.jsonl"
    subprocess.run([parser_python, str(native_parser), "--input_path", str(output / "pdbs"),
                    "--output_path", str(parser_out)], check=True, capture_output=True, text=True)
    records = [json.loads(line) for line in parser_out.read_text().splitlines() if line.strip()]
    if len(records) != 1:
        raise HandoffError("native parser did not produce exactly one candidate")
    parsed = records[0]
    native_parsed = output / "native-parsed.jsonl"
    shutil.copyfile(parser_out, native_parsed)
    _verify_parser(parsed, converted, candidate)
    parser_out.write_text(json.dumps(parsed, allow_nan=False) + "\n")
    assignments = {candidate: [[mapping[c] for c in binder_chains], [mapping[c] for c in target_chains]]}
    (output / "chain_id.jsonl").write_text(json.dumps(assignments, sort_keys=True) + "\n")
    residue_map = []
    for chain, residues in sorted(chains.items()):
        for index, (key, residue) in enumerate(sorted(residues.items()), 1):
            identity = residue["identity"]
            residue_map.append({"source_chain": chain, "source_label_chain": identity["label_chain"],
                                "source_author_chain": identity["auth_chain"], "source_label_residue": identity["label_residue"],
                                "source_author_residue": identity["auth_residue"], "source_insertion": key[1],
                                "pdb_chain": mapping[chain], "pdb_residue": key[0], "pdb_insertion": key[1],
                                "native_parser_chain_position": index, "residue_name": residue["resname"],
                                "source_residue_name": identity["source_resname"],
                                "role": "binder" if chain in binder else "fixed-target"})
    manifest = {"schema": "proteinmpnn-generated-handoff/1", "candidate": candidate,
                "source_tool": source_tool, "generator_mode": generator_mode,
                "pins": {"kit_commit": KIT_COMMIT, "proteinmpnn_commit": MPNN_COMMIT, "native_parser_sha256": PARSER_SHA256},
                "source": {"artifact": raw.name, "sha256": sha256(raw)},
                "conversion": {"operation": conversion, "max_coordinate_rounding_angstrom": delta,
                               **conversion_proof,
                               "atoms_preserved": len(atoms), "reconstructed_atoms": 0,
                               "occupancy_bfactor_policy": "source PDB preserved" if extension == ".pdb" else "PDB placeholders 1.00/0.00; original CIF retained"},
                "pdb": {"artifact": f"pdbs/{candidate}.pdb", "sha256": sha256(pdb)},
                "parsed": {"artifact": "parsed.jsonl", "sha256": sha256(parser_out)},
                "native_parsed": {"artifact": "native-parsed.jsonl", "sha256": sha256(native_parsed)},
                "sequence_placeholder_policy": {"binder_residue_name_map": placeholders,
                                                "mapped_source_atoms": mapped_atoms,
                                                "observed_UNK_native_parser_symbol": "-",
                                                "observed_UNK_model_symbol": "X",
                                                "target_symbols_changed": 0},
                "chain_assignment": {"artifact": "chain_id.jsonl", "sha256": sha256(output / "chain_id.jsonl"), "values": assignments},
                "residue_map": residue_map,
                "claims": {"full_backbone_observed": True, "native_parser_independently_checked": True,
                           "roles_supplied_by_caller": True, "proteinmpnn_execution_validated": False}}
    if generator_manifest is not None:
        shutil.copyfile(generator_manifest, output / "generator-manifest.json")
        manifest["generator_manifest"] = {"artifact": "generator-manifest.json", "sha256": sha256(output / "generator-manifest.json")}
    (output / "handoff.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    commands = []
    for variant in ("vanilla", "soluble"):
        for mode in ("off", "exact"):
            commands.append({"variant": variant, "mode": mode, "cwd": "/kit/proteinmpnn",
                             "argv": ["bash", "run.sh", "design", "--config", "h100", "--mode", mode, "--variant", variant,
                                      "--jsonl_path", "/in/handoff/parsed.jsonl", "--chain_id_jsonl", "/in/handoff/chain_id.jsonl",
                                      "--out", f"/out/{variant}-{mode}", "--num_seq_per_target", "1", "--batch_size", "1", "--seed", "37",
                                      "--save_score", "1", "--save_probs", "1"]})
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    return manifest


def _reference_residues(reference: Path, *, reference_chain: str, start: int,
                        end: int, numbering: str = "label") -> list[dict]:
    """Read an explicit target crop without choosing coordinate alternates.

    Reference alternate atoms are permitted because this compares observed
    residue identities and sequences only. Conflicting residue names refuse.
    """
    opener = gzip.open if reference.suffix == ".gz" else open
    with opener(reference, "rt", encoding="utf-8") as stream:
        text = stream.read()
    residues = {}
    if reference.name.lower().endswith(".pdb"):
        rows = [{"chain": line[21], "number": line[22:26].strip(), "insertion": line[26].strip(),
                 "name": line[17:20], "label_number": "", "author_number": line[22:26].strip(),
                 "label_chain": "", "author_chain": line[21]}
                for line in text.splitlines() if line.startswith("ATOM  ")]
    else:
        rows = []
        for row in _cif_rows(text):
            if row.get("_atom_site.group_PDB", "ATOM") != "ATOM":
                continue
            label, author = _v(row, "_atom_site.label_asym_id"), _v(row, "_atom_site.auth_asym_id")
            ln, an = _v(row, "_atom_site.label_seq_id"), _v(row, "_atom_site.auth_seq_id")
            rows.append({"chain": label if numbering == "label" else author, "number": ln if numbering == "label" else an,
                         "insertion": _v(row, "_atom_site.pdbx_PDB_ins_code"),
                         "name": _v(row, "_atom_site.label_comp_id", "_atom_site.auth_comp_id"),
                         "label_number": ln, "author_number": an, "label_chain": label, "author_chain": author})
    for row in rows:
        if row["chain"] != reference_chain or not start <= int(row["number"]) <= end:
            continue
        if row["name"] not in AMINO or row["name"] == "UNK":
            raise HandoffError("target reference crop contains unknown residues")
        key = (int(row["number"]), row["insertion"])
        if key in residues and residues[key]["name"] != row["name"]:
            raise HandoffError("target reference has conflicting alternate residue names")
        residues[key] = row
    if not residues:
        raise HandoffError("target reference crop is empty")
    return [residues[key] for key in sorted(residues)]


def attach_target_reference(handoff: Path, reference: Path, *, target_chain: str,
                            reference_chain: str, start: int, end: int, numbering: str = "label") -> dict:
    """Record an independently matched reference crop and actual residue map."""
    ordered = _reference_residues(reference, reference_chain=reference_chain, start=start, end=end, numbering=numbering)
    sequence = "".join(AMINO[row["name"]] for row in ordered)
    manifest = json.loads((handoff / "handoff.json").read_text())
    parsed = json.loads((handoff / "parsed.jsonl").read_text())
    fixed = manifest["chain_assignment"]["values"][manifest["candidate"]][1]
    if target_chain not in fixed or parsed[f"seq_chain_{target_chain}"] != sequence:
        raise HandoffError("frozen target differs from explicit reference crop")
    mapped = sorted((r for r in manifest["residue_map"] if r["pdb_chain"] == target_chain),
                    key=lambda r: r["native_parser_chain_position"])
    if len(mapped) != len(ordered):
        raise HandoffError("reference and generated target residue census differs")
    extension = ".pdb" if reference.name.lower().endswith(".pdb") else ".cif.gz" if reference.suffix == ".gz" else ".cif"
    destination = handoff / ("target-reference" + extension)
    shutil.copyfile(reference, destination)
    for generated, original in zip(mapped, ordered):
        generated["target_reference"] = {"chain": reference_chain, "numbering": numbering,
                                         "residue": original["number"], "insertion": original["insertion"],
                                         "label_chain": original["label_chain"], "author_chain": original["author_chain"],
                                         "label_residue": original["label_number"], "author_residue": original["author_number"]}
    manifest["target_reference"] = {"artifact": destination.name, "sha256": sha256(destination),
                                    "target_chain": target_chain, "reference_chain": reference_chain,
                                    "start": start, "end": end, "numbering": numbering,
                                    "residues_independently_matched": len(mapped), "sequence_unchanged": True}
    (handoff / "handoff.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest["target_reference"]


def _npy(data: bytes) -> tuple[tuple, list]:
    if data[:6] != b"\x93NUMPY" or data[6] not in (1, 2, 3):
        raise HandoffError("unsupported probability-array format")
    width, encoding = (2, "latin1") if data[6] == 1 else (4, "utf-8" if data[6] == 3 else "latin1")
    length = int.from_bytes(data[8:8 + width], "little")
    offset = 8 + width + length
    header = ast.literal_eval(data[8 + width:offset].decode(encoding))
    shape, dtype = tuple(header["shape"]), header["descr"]
    if header["fortran_order"] or not re.fullmatch(r"[<|=][ifU][0-9]+", dtype):
        raise HandoffError("probability array uses unsupported/object/Fortran dtype")
    count = math.prod(shape)
    if dtype[1] == "U":
        size = int(dtype[2:]) * 4
        values = [data[offset + i * size:offset + (i + 1) * size].decode("utf-32-le").rstrip("\x00") for i in range(count)]
    else:
        code = {("i", 4): "i", ("i", 8): "q", ("f", 4): "f", ("f", 8): "d"}.get((dtype[1], int(dtype[2:])))
        if code is None:
            raise HandoffError("unsupported numeric dtype")
        size = int(dtype[2:])
        values = list(struct.unpack(f"<{count}{code}", data[offset:]))
    if len(data) - offset != count * size:
        raise HandoffError("truncated probability array")
    return shape, values


def verify_frozen_target(handoff: Path, outputs: Path) -> dict:
    manifest = json.loads((handoff / "handoff.json").read_text())
    for field in ("source", "pdb", "parsed", "native_parsed", "chain_assignment"):
        evidence = manifest[field]
        if sha256(handoff / evidence["artifact"]) != evidence["sha256"]:
            raise HandoffError("handoff input bytes changed after preparation")
    if manifest.get("target_reference"):
        reference = manifest["target_reference"]
        if sha256(handoff / reference["artifact"]) != reference["sha256"]:
            raise HandoffError("target reference bytes changed after verification")
    parsed = json.loads((handoff / manifest["parsed"]["artifact"]).read_text())
    candidate = manifest["candidate"]
    designed, fixed = manifest["chain_assignment"]["values"][candidate]
    reference_report = None
    if manifest.get("target_reference"):
        reference = manifest["target_reference"]
        observed = _reference_residues(handoff / reference["artifact"], reference_chain=reference["reference_chain"],
                                       start=reference["start"], end=reference["end"], numbering=reference["numbering"])
        sequence = "".join(AMINO[r["name"]] for r in observed)
        if reference["target_chain"] not in fixed or parsed[f"seq_chain_{reference['target_chain']}"] != sequence:
            raise HandoffError("fixed target differs from independently reread origin crop")
        reference_report = {"artifact": reference["artifact"], "sha256": reference["sha256"],
                            "target_chain": reference["target_chain"], "reference_chain": reference["reference_chain"],
                            "start": reference["start"], "end": reference["end"], "numbering": reference["numbering"],
                            "origin_residues_independently_matched": len(observed)}
    npz = outputs / "probs" / f"{candidate}.npz"
    with zipfile.ZipFile(npz) as archive:
        arrays = {name: _npy(archive.read(name + ".npy")) for name in ("S", "mask", "chain_order")}
    sshape, sequences = arrays["S"]
    mshape, masks = arrays["mask"]
    cshape, orders = arrays["chain_order"]
    if len(sshape) != 2 or len(mshape) != 2 or len(cshape) != 2 or sshape[1] != mshape[1] or mshape[0] != cshape[0] or not sshape[0]:
        raise HandoffError("sample/mask/chain-order shapes differ")
    expected_chains = set(designed + fixed)
    frozen_residues = 0
    for sample in range(sshape[0]):
        batch = sample % mshape[0]
        order = orders[batch * cshape[1]:(batch + 1) * cshape[1]]
        if len(order) != len(expected_chains) or set(order) != expected_chains:
            raise HandoffError("sample chain order differs from observed complex")
        offset = 0
        for chain in order:
            expected = parsed[f"seq_chain_{chain}"]
            indices = sequences[sample * sshape[1] + offset:sample * sshape[1] + offset + len(expected)]
            loss = masks[batch * mshape[1] + offset:batch * mshape[1] + offset + len(expected)]
            if any(not isinstance(i, int) or not 0 <= i < len(ALPHABET) for i in indices):
                raise HandoffError("invalid sampled residue index")
            if chain in fixed:
                if "".join(ALPHABET[i] for i in indices) != expected or any(x != 0.0 for x in loss):
                    raise HandoffError("fixed target was redesigned or enabled in design loss")
                frozen_residues += len(expected)
            elif any(x != 1.0 for x in loss):
                raise HandoffError("binder residues were not enabled for design")
            offset += len(expected)
        if offset != sshape[1]:
            raise HandoffError("sample contains unaccounted residues")
    return {"schema": "proteinmpnn-frozen-target/1", "candidate": candidate,
            "generated_source_sha256": manifest["source"]["sha256"],
            "pdb_sha256": manifest["pdb"]["sha256"], "parsed_sha256": manifest["parsed"]["sha256"],
            "probabilities": {"artifact": f"probs/{candidate}.npz", "sha256": sha256(npz)},
            "designed_chains": designed, "fixed_chains": fixed, "samples_checked": sshape[0],
            "fixed_target_residue_observations_checked": frozen_residues,
            "target_reference": reference_report,
            "target_sequence_unchanged": True, "target_design_mask_zero": True, "binder_design_mask_one": True,
            "input_coordinate_bytes_unchanged": True}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("source", type=Path)
    prep.add_argument("output", type=Path)
    prep.add_argument("--candidate", required=True)
    prep.add_argument("--source-tool", required=True)
    prep.add_argument("--generator-mode", required=True)
    prep.add_argument("--binder-chains", nargs="+", required=True)
    prep.add_argument("--target-chains", nargs="+", required=True)
    prep.add_argument("--native-parser", type=Path, required=True)
    prep.add_argument("--parser-python", default=sys.executable)
    prep.add_argument("--chain-map", type=json.loads)
    prep.add_argument("--generator-manifest", type=Path)
    prep.add_argument("--binder-placeholder-map", type=json.loads)
    verify = sub.add_parser("verify")
    verify.add_argument("handoff", type=Path)
    verify.add_argument("outputs", type=Path)
    origin = sub.add_parser("reference")
    origin.add_argument("handoff", type=Path)
    origin.add_argument("reference", type=Path)
    origin.add_argument("--target-chain", required=True)
    origin.add_argument("--reference-chain", required=True)
    origin.add_argument("--start", type=int, required=True)
    origin.add_argument("--end", type=int, required=True)
    origin.add_argument("--numbering", choices=("label", "author"), default="label")
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = prepare(args.source, args.output, candidate=args.candidate, source_tool=args.source_tool,
                             generator_mode=args.generator_mode, binder_chains=args.binder_chains,
                             target_chains=args.target_chains, native_parser=args.native_parser,
                             parser_python=args.parser_python, chain_map=args.chain_map, generator_manifest=args.generator_manifest,
                             binder_placeholder_map=args.binder_placeholder_map)
        elif args.action == "reference":
            result = attach_target_reference(args.handoff, args.reference, target_chain=args.target_chain,
                                             reference_chain=args.reference_chain, start=args.start, end=args.end,
                                             numbering=args.numbering)
        else:
            result = verify_frozen_target(args.handoff, args.outputs)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (HandoffError, OSError, ValueError, KeyError, subprocess.CalledProcessError, zipfile.BadZipFile, struct.error) as exc:
        print(json.dumps({"status": "error", "reason": type(exc).__name__ + ": " + str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
