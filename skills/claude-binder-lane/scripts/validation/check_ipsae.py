#!/usr/bin/env python3
"""Check our ipSAE against the published ipsae_min column, per design and arm.

The companion release ships, per design and predictor and seed, the predicted
complex as model.cif and the full PAE matrix as pae.npz. The supplied design
summary CSV carries the published `ipsae_min_<arm>` for the same design and arm.
So our implementation can be run over the released seeds and compared to the
released number without a GPU and without spending anything.

The published column is a maximum, so the comparison has to be built the same
way. `ipsae_min_<arm>` is the largest `ipsae_min` over the five seeds at the
design-complex stoichiometry on the design target. This harness computes
ipsae_min for every seed of a design and arm, takes the maximum, and compares
that. A design with a seed missing is skipped and counted, because a maximum
over four seeds is a different quantity.

TODO(evidence): no file in this package states that definition of the published
`ipsae_min_<arm>` column, and the comparison this harness performs depends on
it. Settled by: the definition of `ipsae_min_<arm>` in the published campaign
release documentation.

This harness measures nothing itself. It resolves the binder chain, calls
compute_ipsae from the metrics lane, and reports the residuals.

Usage:
    python3 check_ipsae.py --summary-csv /path/to/design_summary.csv \\
        --pae-root /path/to/release/designs \\
        --out results.csv
"""

import os
import sys

import argparse
import collections
import csv
import pathlib
import re
import statistics

import numpy as np


def resolve_package_root():
    """Return the package beside a built skill or inside the source checkout."""
    skill_root = pathlib.Path(__file__).resolve().parents[2]
    candidates = (
        skill_root / "claude_binder",
        skill_root.parents[1] / "src" / "claude_binder",
    )
    for candidate in candidates:
        if (candidate / "__init__.py").is_file():
            return candidate.resolve()
    raise RuntimeError(
        "claude_binder package not found. Reinstall this skill; it does not "
        "ship the package these scripts import.")


PACKAGE_ROOT = resolve_package_root()
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

# The ten predictors scored in design_summary.csv.
ARMS = ["ef2fast", "ef2full", "ptxv2", "odde", "afm3",
        "boltz2", "chai1", "of3", "rf3", "af3of3"]

# The published baseline ranking modes.
RANKING_ARMS = ("ef2fast", "ef2full", "ptxv2")

# The stoichiometries of a design complex. design_summary.csv scores only
# these. The 2to2 and 3to3 folds put one binder on each protomer, and they are
# scored in the parquet alone, so a directory carrying one has to be dropped
# rather than folded into the maximum.
# TODO(evidence): no file in this package lists which stoichiometries are the
# design complex, and this tuple decides which folds enter the maximum.
# Settled by: the published campaign release's own statement of which
# stoichiometries design_summary.csv scores.
DESIGN_COMPLEX_STOICHIOMETRIES = ("1to1", "1to2", "1to3", "rnp")

# Five seeds per predictor, per the companion release README. The labels differ
# by predictor, 0 to 4 for most, 1 to 5 for af3of3 and 101 to 105 for odde, so
# the count is what gets checked and the labels are recorded as found.
SEEDS_PER_ARM = 5

SEED_DIR_RE = re.compile(r"^seed_(-?\d+)$")

PAE_FILE = "pae.npz"
MODEL_FILE = "model.cif"

# The directory that holds designed.cif rather than a predictor's seeds.
DESIGN_MODEL_DIR = "designed"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

CSV_FIELDS = [
    "design_id", "target", "arm", "stoichiometry",
    "published_ipsae_min", "our_ipsae_min", "abs_diff",
    "n_seeds", "seed_labels", "seed_ipsae_min", "best_seed",
    "binder_chain", "target_chain", "chain_convention",
    "best_seed_target_to_binder", "best_seed_binder_to_target",
    "pae_orientation", "token_count", "token_res_ids_match",
    "status", "detail",
]


# ---------------------------------------------------------------------------
# Reading the released files
# ---------------------------------------------------------------------------

def npz_identity(path):
    """Return the scalar identity fields of a pae.npz.

    The directory name carries the predictor and often the stoichiometry, and
    the naming is only mostly regular. The npz states its own
    `cofolding_model`, `stoichiometry`, `target_form` and `seed`, so those are
    read from the file and the directory name is used only to find it.
    """
    with np.load(path, allow_pickle=False) as handle:
        return {
            "design_name": str(handle["design_name"]),
            "target": str(handle["target"]),
            "cofolding_model": str(handle["cofolding_model"]),
            "stoichiometry": str(handle["stoichiometry"]),
            "target_form": str(handle["target_form"]),
            "seed": str(handle["seed"]),
        }


def load_pae(path):
    """Return the PAE matrix and the per-token metadata of a pae.npz.

    The matrix ships as float16 and is widened to float64 here. compute_ipsae
    compares PAE values against a 10 Angstrom cutoff, and float16 carries about
    three decimal digits, so a value sitting on the cutoff could fall either
    side of it at the narrower width.
    """
    with np.load(path, allow_pickle=False) as handle:
        return (
            np.asarray(handle["pae"], dtype=float),
            [str(value) for value in handle["token_chain_ids"]],
            [int(value) for value in handle["token_res_ids"]],
            [str(value) for value in handle["token_entity"]],
        )


def chain_sequences(structure):
    """Map chain id to its one-letter sequence, in residue order.

    The structure comes from parse_cif_atoms, which already drops HETATM
    records and keeps only the first model.
    """
    sequences = collections.OrderedDict()
    for residue in structure.residues:
        sequences.setdefault(residue.auth_chain, []).append(
            THREE_TO_ONE.get(residue.comp_id, "X"))
    return collections.OrderedDict(
        (chain, "".join(letters)) for chain, letters in sequences.items())


# ---------------------------------------------------------------------------
# Chain roles
# ---------------------------------------------------------------------------

def resolve_binder_chain(structure, binder_sequence):
    """Return (binder_chain, target_chain) for a two-chain predicted complex.

    The binder is the chain whose sequence is the ordered sequence in
    design_summary.csv. Chain order follows the predictor's input and differs
    between predictors: measured over the released PD-L1 seeds, both ESMFold2
    arms write the binder as chain A, and Protenix v2 writes it as chain B.
    Reading the binder off a fixed position therefore scores the target against
    itself for one arm out of the three, and it returns a plausible wrong
    number rather than an error. The companion release README says the same
    thing and tells the reader to select the binder by sequence.

    A target whose protein part spans more than one chain raises. That is the
    right outcome today, because compute_ipsae reads a single target chain and
    the published definition scores the binder against the union of every
    target protein chain.
    """
    sequences = chain_sequences(structure)
    if len(sequences) != 2:
        raise ValueError("expected 2 protein chains, found %d: %s"
                         % (len(sequences), {c: len(s) for c, s in sequences.items()}))
    matches = [chain for chain, seq in sequences.items() if seq == binder_sequence]
    if len(matches) != 1:
        raise ValueError(
            "binder sequence of %d residues matched %d of the predicted chains %s"
            % (len(binder_sequence), len(matches),
               {c: len(s) for c, s in sequences.items()}))
    binder = matches[0]
    target = next(chain for chain in sequences if chain != binder)
    return binder, target


def token_layout(structure, token_chain_ids, token_res_ids):
    """Say whether the PAE token order and the residue order line up.

    compute_ipsae takes the PAE matrix and the mmCIF and pairs them by
    position, so row i of the matrix has to be residue i of the structure. The
    two agree in the released files, and a file where they stop agreeing would
    produce a plausible wrong number with no error raised. Both the chain block
    partition and the residue numbering are compared here, and the caller
    refuses the pair when the blocks disagree.

    Returns (blocks_match, res_ids_match).
    """
    cif_blocks = []
    for residue in structure.residues:
        if not cif_blocks or cif_blocks[-1][0] != residue.auth_chain:
            cif_blocks.append([residue.auth_chain, 0])
        cif_blocks[-1][1] += 1
    token_blocks = []
    for chain in token_chain_ids:
        if not token_blocks or token_blocks[-1][0] != chain:
            token_blocks.append([chain, 0])
        token_blocks[-1][1] += 1
    blocks_match = cif_blocks == token_blocks
    res_ids_match = [residue.seq_id for residue in structure.residues] == token_res_ids
    return blocks_match, res_ids_match


# ---------------------------------------------------------------------------
# Finding the usable designs
# ---------------------------------------------------------------------------

Seed = collections.namedtuple("Seed", "label npz_path cif_path")

Group = collections.namedtuple(
    "Group", "design_id target arm stoichiometry published seeds")


def load_published(summary_csv, targets):
    """Return the published ipsae_min per design and arm, with the binder sequence.

    The ordered binder sequence and its length come from the same row, so the
    chain that carries the binder can be named from the yardstick file itself.
    """
    published = {}
    with open(summary_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if targets and row["target"] not in targets:
                continue
            scores = {}
            for arm in ARMS:
                raw = row.get("ipsae_min_" + arm, "").strip()
                if raw:
                    scores[arm] = float(raw)
            published[row["full_name"]] = {
                "target": row["target"],
                "sequence": row["sequence"].strip(),
                "binder_length": row.get("binder_length", "").strip(),
                "scores": scores,
            }
    return published


def seed_sort_key(label):
    """Sort seed labels numerically where they are numbers, alphabetically otherwise."""
    try:
        return (0, int(label), "")
    except ValueError:
        return (1, 0, label)


def discover_groups(pae_root, published, targets, arms, required_seeds):
    """Return the design and arm groups usable for the check, and count the rest.

    The released tree is walked rather than enumerated from a fixed list,
    because a fetch may still be running and only part of it may be on disk.
    Every seed directory that holds both a pae.npz and a model.cif is a
    candidate. The npz states its own predictor, stoichiometry, target form and
    seed, and those fields decide which group the seed belongs to.

    A group is usable when the CSV carries an ipsae_min for that arm, the
    stoichiometry is the design complex, the target form is the design target,
    and exactly `required_seeds` seeds are present.
    """
    found = collections.defaultdict(list)
    dropped = collections.Counter()
    seen_designs = 0

    if not os.path.isdir(pae_root):
        raise SystemExit("pae root is not a directory: %s" % pae_root)

    for target_name in sorted(os.listdir(pae_root)):
        target_dir = os.path.join(pae_root, target_name)
        if not os.path.isdir(target_dir):
            continue
        if targets and target_name not in targets:
            continue
        for design_id in sorted(os.listdir(target_dir)):
            design_dir = os.path.join(target_dir, design_id)
            if not os.path.isdir(design_dir):
                continue
            seen_designs += 1
            if design_id not in published:
                dropped["design has no row in design_summary.csv"] += 1
                continue
            for arm_dir_name in sorted(os.listdir(design_dir)):
                if arm_dir_name == DESIGN_MODEL_DIR:
                    continue
                arm_dir = os.path.join(design_dir, arm_dir_name)
                if not os.path.isdir(arm_dir):
                    continue
                for seed_dir_name in sorted(os.listdir(arm_dir)):
                    if not SEED_DIR_RE.match(seed_dir_name):
                        continue
                    seed_dir = os.path.join(arm_dir, seed_dir_name)
                    npz_path = os.path.join(seed_dir, PAE_FILE)
                    cif_path = os.path.join(seed_dir, MODEL_FILE)
                    if not os.path.isfile(npz_path):
                        dropped["seed directory has no pae.npz"] += 1
                        continue
                    if not os.path.isfile(cif_path):
                        dropped["seed directory has no model.cif"] += 1
                        continue
                    try:
                        identity = npz_identity(npz_path)
                    except Exception as exc:              # noqa: BLE001
                        dropped["pae.npz could not be read: %s"
                                % type(exc).__name__] += 1
                        continue
                    arm = identity["cofolding_model"]
                    if arm not in arms:
                        dropped["predictor not requested"] += 1
                        continue
                    if identity["target_form"]:
                        dropped["fold is on an alternate target form"] += 1
                        continue
                    if identity["stoichiometry"] not in DESIGN_COMPLEX_STOICHIOMETRIES:
                        dropped["fold is not at the design-complex stoichiometry"] += 1
                        continue
                    key = (design_id, target_name, arm, identity["stoichiometry"])
                    found[key].append(
                        Seed(identity["seed"], npz_path, cif_path))

    # A design and arm may only carry one design-complex stoichiometry. Two
    # would mean the maximum is being taken across folds the published column
    # keeps apart, so the whole design and arm is refused.
    by_design_arm = collections.defaultdict(list)
    for key in found:
        by_design_arm[(key[0], key[1], key[2])].append(key)

    groups = []
    for (design_id, target_name, arm), keys in sorted(by_design_arm.items()):
        record = published[design_id]
        if arm not in record["scores"]:
            dropped["no published ipsae_min for the arm"] += 1
            continue
        if len(keys) > 1:
            dropped["design and arm carry more than one design-complex "
                    "stoichiometry"] += 1
            continue
        key = keys[0]
        seeds = sorted(found[key], key=lambda s: seed_sort_key(s.label))
        if len(seeds) != required_seeds:
            dropped["fewer than %d seeds on disk, the published maximum "
                    "cannot be formed" % required_seeds] += 1
            continue
        groups.append(Group(design_id, target_name, arm, key[3],
                            record["scores"][arm], seeds))
    return groups, dropped, seen_designs


# ---------------------------------------------------------------------------
# The metrics lane
# ---------------------------------------------------------------------------

def load_compute_ipsae():
    """Import compute_ipsae late and say plainly when it is not usable.

    binder_metrics.py belongs to the metrics lane. It may be absent, it may be
    a stub, or it may raise NotImplementedError. Each of those is reported
    rather than worked around.
    """
    try:
        from claude_binder.adapters import binder_metrics
    except ImportError as exc:
        raise SystemExit(
            "cannot import binder_metrics: %s\n"
            "The package is missing claude_binder.adapters.binder_metrics.\n"
            "The numeric comparison cannot run until that module is available." % exc)
    if not hasattr(binder_metrics, "compute_ipsae"):
        raise SystemExit(
            "binder_metrics has no compute_ipsae. The numeric comparison "
            "cannot run.")
    revision = getattr(binder_metrics, "IPSAE_IMPLEMENTATION_REVISION", "unset")
    return binder_metrics, revision


def looks_like_stub(revision, values):
    """Say whether the results came from a placeholder rather than a measurement.

    A stub returns the same constant for every design. Comparing a constant to
    the published column produces an undefined correlation, so the run is
    reported as not yet valid rather than as a result.
    """
    if "stub" in str(revision).lower():
        return "IPSAE_IMPLEMENTATION_REVISION is %r" % revision
    if len(values) > 2 and len(set(values)) == 1:
        return "every design returned the same value, %r" % values[0]
    return None


def score_group(metrics, group, binder_sequence, orientation, strict_res_ids):
    """Compute ipsae_min for every seed of one design and arm.

    Returns a dict carrying the per-seed values, the maximum, and the chain
    roles found. Raises when any seed cannot be scored, because a maximum over
    a subset of the seeds is a different quantity from the published one.
    """
    per_seed = []
    binder = target = None
    token_count = None
    res_ids_match = True
    directions = {}

    for seed in group.seeds:
        matrix, token_chain_ids, token_res_ids, token_entity = load_pae(seed.npz_path)

        off_protein = sorted({e for e in token_entity if e != "protein"})
        if off_protein:
            raise ValueError(
                "seed %s carries %s tokens. compute_ipsae takes no token map, "
                "and the published metric excludes them"
                % (seed.label, " and ".join(off_protein)))

        structure = metrics.parse_cif_atoms(pathlib.Path(seed.cif_path))
        if len(structure) != matrix.shape[0]:
            raise ValueError(
                "seed %s has a %d by %d PAE and %d residues in model.cif"
                % (seed.label, matrix.shape[0], matrix.shape[0], len(structure)))

        blocks_match, ids_match = token_layout(
            structure, token_chain_ids, token_res_ids)
        if not blocks_match:
            raise ValueError(
                "seed %s has a chain block layout in model.cif that differs "
                "from token_chain_ids" % seed.label)
        if not ids_match:
            res_ids_match = False
            if strict_res_ids:
                raise ValueError(
                    "seed %s has residue numbers in model.cif that differ from "
                    "token_res_ids" % seed.label)

        seed_binder, seed_target = resolve_binder_chain(structure, binder_sequence)
        if binder is None:
            binder, target = seed_binder, seed_target
        elif (seed_binder, seed_target) != (binder, target):
            raise ValueError(
                "seed %s puts the binder in chain %s and the earlier seeds put "
                "it in chain %s" % (seed.label, seed_binder, binder))

        result = metrics.compute_ipsae(matrix, pathlib.Path(seed.cif_path),
                                       target, binder,
                                       pae_orientation=orientation)
        if not isinstance(result, dict) or "ipsae_min" not in result:
            raise ValueError("compute_ipsae returned %r without ipsae_min"
                             % type(result).__name__)
        per_seed.append((seed.label, float(result["ipsae_min"])))
        directions[seed.label] = (float(result["ipsae_target_to_binder"]),
                                  float(result["ipsae_binder_to_target"]))
        token_count = matrix.shape[0]

    convention = "binder=%s target=%s" % (binder, target)
    # The published rule takes the largest ipsae_min over the five seeds and
    # resolves a tie to the lowest seed label. The seeds are already in label
    # order, so the first seed holding the maximum is the published one.
    best_value = max(value for _, value in per_seed)
    best_seed = next(label for label, value in per_seed if value == best_value)
    return {
        "per_seed": per_seed,
        "our_ipsae_min": best_value,
        "best_seed": best_seed,
        "binder_chain": binder,
        "target_chain": target,
        "chain_convention": convention,
        "token_count": token_count,
        "token_res_ids_match": res_ids_match,
        "best_seed_target_to_binder": directions[best_seed][0],
        "best_seed_binder_to_target": directions[best_seed][1],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def pearson(xs, ys):
    """Return Pearson r, or None when it is undefined.

    r needs variance on both sides. A constant column makes it undefined, and
    numpy returns nan there, so the undefined case is named instead.
    """
    if len(xs) < 2:
        return None
    a, b = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarise(rows, revision, arms_seen, orientation):
    scored = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]

    print()
    print("=" * 72)
    print("ipSAE check against the published design_summary.csv")
    print("=" * 72)
    print("IPSAE_IMPLEMENTATION_REVISION: %s" % revision)
    print("pae_orientation:               %s" % orientation)
    print("groups attempted: %d" % len(rows))
    print("groups scored:    %d" % len(scored))
    print("groups failed:    %d" % len(failed))

    if failed:
        reasons = collections.Counter(r["detail"].split(":")[0] for r in failed)
        print("\nfailures by reason:")
        for reason, count in reasons.most_common():
            print("   %-56s %d" % (reason[:56], count))

    if not scored:
        print("\nNo group was scored. No correlation and no difference are reported.")
        return

    ours = [r["our_ipsae_min"] for r in scored]
    theirs = [r["published_ipsae_min"] for r in scored]
    diffs = [r["abs_diff"] for r in scored]

    stub = looks_like_stub(revision, ours)
    if stub:
        print("\n" + "!" * 72)
        print("THESE NUMBERS ARE NOT A VALIDATION RESULT.")
        print("The implementation under test looks like a placeholder: %s" % stub)
        print("Treat the statistics below as a harness self-test only.")
        print("!" * 72)

    r = pearson(ours, theirs)
    print("\nn                       %d" % len(scored))
    print("pearson r               %s"
          % ("undefined, one side has no variance" if r is None else "%.6f" % r))
    print("mean absolute diff      %.9f" % statistics.fmean(diffs))
    print("median absolute diff    %.9f" % statistics.median(diffs))
    print("max absolute diff       %.9f" % max(diffs))

    print("\nper arm:")
    print("   %-9s %6s %14s %12s" % ("arm", "n", "mean |d|", "pearson r"))
    for arm in arms_seen:
        subset = [r_ for r_ in scored if r_["arm"] == arm]
        if not subset:
            continue
        sub_r = pearson([s["our_ipsae_min"] for s in subset],
                        [s["published_ipsae_min"] for s in subset])
        print("   %-9s %6d %14.9f %12s"
              % (arm, len(subset),
                 statistics.fmean([s["abs_diff"] for s in subset]),
                 "undefined" if sub_r is None else "%.6f" % sub_r))

    print("\nchain convention found, per arm:")
    conventions = collections.Counter(
        (r_["arm"], r_["chain_convention"]) for r_ in scored)
    for (arm, convention), count in sorted(conventions.items()):
        print("   %-9s %-24s %d" % (arm, convention, count))

    print("\nfive worst by absolute difference:")
    worst = sorted(scored, key=lambda r_: r_["abs_diff"], reverse=True)[:5]
    print("   %-46s %-8s %10s %10s %12s"
          % ("design", "arm", "published", "ours", "|diff|"))
    for row in worst:
        print("   %-46s %-8s %10.6f %10.6f %12.9f"
              % (row["design_id"][:46], row["arm"], row["published_ipsae_min"],
                 row["our_ipsae_min"], row["abs_diff"]))


def summarise_orientations(results, arms_seen):
    """Print the residuals under both PAE orientations, side by side.

    compute_ipsae documents an open question about which way round the released
    matrices are indexed, and it says a symmetric test matrix cannot settle it.
    The released matrices are asymmetric and the published values are known, so
    running both orientations over the same seeds answers it.
    """
    print()
    print("=" * 72)
    print("PAE orientation, both settings over the same seeds")
    print("=" * 72)
    print("   %-18s %6s %16s %16s" % ("orientation", "n", "mean |d|", "max |d|"))
    for orientation, rows in results.items():
        scored = [r for r in rows if r["status"] == "ok"]
        if not scored:
            print("   %-18s %6d %16s %16s" % (orientation, 0, "none", "none"))
            continue
        diffs = [r["abs_diff"] for r in scored]
        print("   %-18s %6d %16.9f %16.9f"
              % (orientation, len(scored), statistics.fmean(diffs), max(diffs)))


# ---------------------------------------------------------------------------

def run(groups, published, metrics, orientation, strict_res_ids):
    """Score every group under one PAE orientation and return the result rows."""
    rows = []
    for index, group in enumerate(groups, 1):
        row = {field: "" for field in CSV_FIELDS}
        row.update(design_id=group.design_id, target=group.target, arm=group.arm,
                   stoichiometry=group.stoichiometry,
                   published_ipsae_min=group.published,
                   n_seeds=len(group.seeds),
                   seed_labels=";".join(s.label for s in group.seeds),
                   pae_orientation=orientation,
                   status="ok", detail="")
        try:
            scored = score_group(metrics, group,
                                 published[group.design_id]["sequence"],
                                 orientation, strict_res_ids)
            row.update(
                our_ipsae_min=scored["our_ipsae_min"],
                abs_diff=abs(scored["our_ipsae_min"] - group.published),
                seed_ipsae_min=";".join(
                    "%s=%.8f" % (label, value) for label, value in scored["per_seed"]),
                best_seed=scored["best_seed"],
                binder_chain=scored["binder_chain"],
                target_chain=scored["target_chain"],
                chain_convention=scored["chain_convention"],
                best_seed_target_to_binder=scored["best_seed_target_to_binder"],
                best_seed_binder_to_target=scored["best_seed_binder_to_target"],
                token_count=scored["token_count"],
                token_res_ids_match=str(scored["token_res_ids_match"]),
            )
        except NotImplementedError as exc:
            row["status"] = "not-implemented"
            row["detail"] = "compute_ipsae raised NotImplementedError: %s" % exc
        except Exception as exc:                          # noqa: BLE001
            row["status"] = "error"
            row["detail"] = "%s: %s" % (type(exc).__name__, exc)
        rows.append(row)
        if index % 50 == 0:
            print("   %d / %d" % (index, len(groups)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True,
                        help="published design summary CSV")
    parser.add_argument("--pae-root", required=True,
                        help="directory holding <target>/<design>/<arm>/seed_<n>/")
    parser.add_argument("--out", required=True, help="results CSV to write")
    parser.add_argument("--targets", nargs="*", default=None,
                        help="limit to these targets, default is every target found")
    parser.add_argument("--arms", nargs="*", default=ARMS, choices=ARMS,
                        help="limit to these arms, default is all ten")
    parser.add_argument("--ranking-arms-only", action="store_true",
                        help="use only %s" % ", ".join(RANKING_ARMS))
    parser.add_argument("--pae-orientation",
                        choices=("aligned_rows", "aligned_columns"),
                        default="aligned_rows",
                        help="how compute_ipsae reads pae[i][j], see its docstring")
    parser.add_argument("--compare-orientations", action="store_true",
                        help="score under both orientations and report both")
    parser.add_argument("--required-seeds", type=int, default=SEEDS_PER_ARM,
                        help="seeds a design and arm needs before it is compared")
    parser.add_argument("--strict-res-ids", action="store_true",
                        help="refuse a seed whose model.cif residue numbers "
                             "differ from token_res_ids")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many groups, for a quick check")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the usable groups and call nothing")
    args = parser.parse_args()

    arms = list(RANKING_ARMS) if args.ranking_arms_only else args.arms
    targets = set(args.targets) if args.targets else None

    published = load_published(args.summary_csv, targets)
    groups, dropped, seen_designs = discover_groups(
        args.pae_root, published, targets, arms, args.required_seeds)

    print("published rows read:      %d" % len(published))
    print("design directories found: %d" % seen_designs)
    print("usable groups found:      %d" % len(groups))
    if dropped:
        print("dropped:")
        for reason, count in dropped.most_common():
            print("   %-64s %d" % (reason, count))

    counts = collections.Counter((g.target, g.arm, g.stoichiometry) for g in groups)
    if counts:
        print("\nusable designs per target, arm and stoichiometry:")
        for (target, arm, stoichiometry), count in sorted(counts.items()):
            print("   %-10s %-9s %-6s %d" % (target, arm, stoichiometry, count))

    if args.limit:
        groups = groups[:args.limit]
        print("\nlimited to %d groups" % len(groups))

    if not groups:
        raise SystemExit("\nNo usable group was found. Nothing was compared.")

    if args.dry_run:
        print("\nNo ipSAE was computed. This was a dry run.")
        return

    metrics, revision = load_compute_ipsae()

    orientations = (["aligned_rows", "aligned_columns"]
                    if args.compare_orientations else [args.pae_orientation])
    results = collections.OrderedDict()
    for orientation in orientations:
        print("\nscoring %d groups under %s" % (len(groups), orientation))
        results[orientation] = run(groups, published, metrics, orientation,
                                   args.strict_res_ids)

    all_rows = [row for rows in results.values() for row in rows]
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print("\nwrote %s" % os.path.abspath(args.out))

    primary = args.pae_orientation if args.pae_orientation in results \
        else orientations[0]
    summarise(results[primary], revision, arms, primary)
    if len(results) > 1:
        summarise_orientations(results, arms)


if __name__ == "__main__":
    main()
