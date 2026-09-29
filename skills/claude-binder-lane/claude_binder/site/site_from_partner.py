"""Derive a hotspot residue list from a solved complex, so a user never has to
supply the contact residues the campaign is supposed to find.

Standard library only. No pip install, which sidesteps the kernel's missing
httpx. Every number it emits is read from a deposited structure, never guessed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
import urllib.request

UNIPROT = "https://rest.uniprot.org/uniprotkb/{acc}.json"
CIF = "https://files.rcsb.org/download/{entry}.cif.gz"
SIFTS = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{entry}"

# Heavy-atom contact distance. Taken from the site-design review, which chose it
# as the conventional interface cutoff. It is a choice, not a measurement, and
# the artifact records it so a reader can disagree.
CUTOFF = 4.0


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "claude-binder-site/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def pdb_entries(acc: str) -> list[str]:
    """PDB entry ids cross-referenced from a UniProt accession."""
    data = json.loads(get(UNIPROT.format(acc=acc)))
    return [x["id"] for x in data.get("uniProtKBCrossReferences", [])
            if x.get("database") == "PDB"]


def parse_atom_site(text: str):
    """Yield (asym, auth_asym, auth_seq, icode, element, x, y, z) for model 1.

    Hand-parsed because no mmCIF library is installed anywhere we ship to.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        j = i + 1
        cols = []
        while j < len(lines) and lines[j].lstrip().startswith("_"):
            cols.append(lines[j].strip())
            j += 1
        if not cols or not cols[0].startswith("_atom_site."):
            i = j
            continue
        idx = {name: k for k, name in enumerate(cols)}

        def col(name):
            return idx.get("_atom_site." + name)

        c_el, c_auth_asym = col("type_symbol"), col("auth_asym_id")
        c_seq, c_ic = col("auth_seq_id"), col("pdbx_PDB_ins_code")
        c_x, c_y, c_z = col("Cartn_x"), col("Cartn_y"), col("Cartn_z")
        c_model, c_alt = col("pdbx_PDB_model_num"), col("label_alt_id")
        c_group = idx.get("_atom_site.group_PDB")
        while j < len(lines) and not lines[j].startswith(("#", "loop_", "_")):
            f = lines[j].split()
            j += 1
            if len(f) < len(cols):
                continue
            if c_group is not None and f[c_group] != "ATOM":
                continue
            if c_model is not None and f[c_model] != "1":
                continue
            if c_alt is not None and f[c_alt] not in (".", "?", "A"):
                continue
            el = f[c_el]
            if el in ("H", "D"):
                continue
            yield (f[c_auth_asym], f[c_seq],
                   "" if c_ic is None or f[c_ic] in (".", "?") else f[c_ic],
                   el, float(f[c_x]), float(f[c_y]), float(f[c_z]))
        i = j


def contacts(atoms, chain_t: str, chain_p: str, cutoff: float):
    t = [a for a in atoms if a[0] == chain_t]
    p = [a for a in atoms if a[0] == chain_p]
    if not t or not p:
        return set(), len(t), len(p)
    c2 = cutoff * cutoff
    # bucket the partner atoms so this stays linear in practice
    grid = {}
    for a in p:
        key = (int(a[4] // cutoff), int(a[5] // cutoff), int(a[6] // cutoff))
        grid.setdefault(key, []).append(a)
    hits = set()
    for a in t:
        gx, gy, gz = int(a[4] // cutoff), int(a[5] // cutoff), int(a[6] // cutoff)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for b in grid.get((gx + dx, gy + dy, gz + dz), ()):
                        if (a[4]-b[4])**2 + (a[5]-b[5])**2 + (a[6]-b[6])**2 <= c2:
                            hits.add((a[1], a[2]))
                            break
    return hits, len(t), len(p)


def sifts(entry: str):
    """Author numbering to UniProt numbering for every chain in an entry.

    Chain identity comes from here and nowhere else. Choosing the target chain
    by contact count instead picked PD-1 when the target was PD-L1, which would
    have conditioned every design on the wrong face of the wrong protein.
    """
    data = json.loads(get(SIFTS.format(entry=entry.lower())))
    by_acc = {}
    for acc, v in data.get(entry.lower(), {}).get("UniProt", {}).items():
        for m in v.get("mappings", []):
            by_acc.setdefault(acc, []).append(m)
    return by_acc


def mapper_for(segs):
    """Map author numbering to UniProt numbering.

    Use author_residue_number, never residue_number. The latter is the entity
    index, and reading it as author numbering shifted every hotspot by 17 on
    the first real test while still producing a confident-looking list.
    """
    def mapper(auth_seq: int):
        for m in segs:
            lo = m["start"]["author_residue_number"]
            hi = m["end"]["author_residue_number"]
            if lo <= auth_seq <= hi:
                return m["unp_start"] + (auth_seq - lo)
        return None
    return mapper


def main(target_acc: str, partner_acc: str, entry_hint: str | None = None,
         out: str = "site_selection.json"):
    t_entries = pdb_entries(target_acc)
    p_entries = pdb_entries(partner_acc)
    shared = [e for e in t_entries if e in p_entries]
    print(f"target {target_acc}: {len(t_entries)} PDB entries")
    print(f"partner {partner_acc}: {len(p_entries)} PDB entries")
    print(f"complexes containing both: {len(shared)} -> {shared[:12]}")
    if entry_hint:
        shared = [entry_hint] + [e for e in shared if e != entry_hint]
    if not shared:
        print("REFUSED: no deposited complex contains both proteins.")
        print("A user here needs a different route, not a guess.")
        return 2

    entry = shared[0]
    by_acc = sifts(entry)
    t_segs = by_acc.get(target_acc, [])
    p_segs = by_acc.get(partner_acc, [])
    t_chains = sorted({m["chain_id"] for m in t_segs})
    p_chains = sorted({m["chain_id"] for m in p_segs})
    if not t_chains or not p_chains:
        print(f"REFUSED: {entry} does not map both accessions to chains.")
        print(f"  target chains {t_chains}, partner chains {p_chains}")
        return 4
    offsets = sorted({m["unp_start"] - m["start"]["author_residue_number"] for m in t_segs})
    print(f"\nentry {entry}: target {target_acc} is chain(s) {t_chains}, "
          f"partner {partner_acc} is chain(s) {p_chains}")
    print(f"author-to-UniProt offset(s) for the target: {offsets}")

    raw = gzip.decompress(get(CIF.format(entry=entry))).decode("utf-8", "replace")
    atoms = list(parse_atom_site(raw))
    ct, cp = t_chains[0], p_chains[0]
    hits, nt, np_ = contacts(atoms, ct, cp, CUTOFF)
    print(f"chain {ct} ({nt} atoms) against chain {cp} ({np_} atoms)")
    if not hits:
        print(f"REFUSED: no contacts at {CUTOFF} A. The complex has no interface "
              f"between these two chains, so there is no site to report.")
        return 3
    print(f"contacts at {CUTOFF} A: {len(hits)} target residues")

    mapper = mapper_for([m for m in t_segs if m["chain_id"] == ct])
    rows = []
    for seq, ic in sorted(hits, key=lambda h: int(h[0])):
        up = mapper(int(seq))
        rows.append({"uniprot_pos": up, "author": f"{ct}/{seq}{ic}",
                     "evidence": f"PDB {entry}: heavy atom <= {CUTOFF} A of chain {cp}"})
    unmapped = sum(1 for r in rows if r["uniprot_pos"] is None)
    if unmapped:
        print(f"REFUSED: {unmapped} of {len(rows)} residues have no UniProt "
              f"position. A partial map shifts hotspots silently.")
        return 5
    art = {"target_uniprot": target_acc, "partner_uniprot": partner_acc,
           "chain_author": ct, "partner_chain_author": cp,
           "hotspots": rows, "cutoff_angstrom": CUTOFF,
           "tier": "MEASURED", "source_entries": [entry],
           "author_to_uniprot_offsets": offsets}
    body = json.dumps(art, indent=2, sort_keys=True)
    art["residue_map_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    with open(out, "w") as fh:
        fh.write(json.dumps(art, indent=2, sort_keys=True))
    print(f"\nUniProt positions: {[r['uniprot_pos'] for r in rows]}")
    print(f"wrote {out}")
    print(f"residue_map_sha256 {art['residue_map_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
