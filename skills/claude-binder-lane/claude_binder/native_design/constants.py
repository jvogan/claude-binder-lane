"""CPU-safe constants shared by the native design modules.

The values mirror the sister implementation. The adapter supplies campaign
values for fields that the specification leaves unresolved.
"""

from __future__ import annotations

import re

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
STANDARD_AMINO_ACID_SET = frozenset(STANDARD_AMINO_ACIDS)
DEFAULT_MUTABLE_AMINO_ACIDS = "ADEFGHIKLMNPQRSTVWY"
MUTABLE_TOKEN = "#"
AA_DIMS = 20
LOSS_WEIGHTS = {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2}
ESMC_MASK_FRACTION = 0.15
AMBIGUOUS_AMINO_ACIDS = {
    "B": "DN",
    "J": "IL",
    "X": STANDARD_AMINO_ACIDS,
    "Z": "EQ",
}
PROSITE_COUNT_RE = re.compile(r"\((\d+)(?:,(\d+))?\)$")

# The ESMFold2 protein vocabulary uses these canonical three-letter CCD names.
PROTEIN_1TO3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}
PROTEIN_3TO1 = {value: key for key, value in PROTEIN_1TO3.items()}
TOKENS = ["<pad>", "-"] + [PROTEIN_1TO3[residue] for residue in STANDARD_AMINO_ACIDS]
TOKEN_IDS = {token: index for index, token in enumerate(TOKENS)}
CYS_IDX = TOKEN_IDS[PROTEIN_1TO3["C"]] - 2
