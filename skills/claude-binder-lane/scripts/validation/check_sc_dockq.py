#!/usr/bin/env python3
"""Check our sc_DockQ against the published sc_dockq column, per design and arm.

The released dataset ships both structures sc_DockQ needs. Every design carries a
designed.cif, which is the reference, and one predicted complex per predictor.
The supplied design summary CSV carries the published sc_dockq for the same design
and arm. So our implementation can be run over the released pairs and compared to
the released number without a GPU and without spending anything.

This harness measures nothing itself. It resolves the chain mapping, calls
compute_dockq from the metrics lane, and reports the residuals.

Usage:
    python3 check_sc_dockq.py --summary-csv /path/to/design_summary.csv \\
        --data-root /path/to/release/designs \\
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

REFERENCE_CIF = "designed.cif"

# Predicted files are named predicted_<arm>_<variant>.cif. The variant is per
# target, so it is discovered rather than assumed. PD-L1 uses 1to1 and Cas9
# uses rnp.
PREDICTED_RE = re.compile(r"^predicted_([a-z0-9]+)_(.+)\.cif$")

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

CSV_FIELDS = [
    "design_id", "target", "arm",
    "published_sc_dockq", "our_sc_dockq", "abs_diff",
    "pred_target_chain", "pred_binder_chain",
    "ref_target_chain", "ref_binder_chain",
    "chain_ids_agree", "chain_normalisation",
    "mapping_status", "status", "detail",
]


# ---------------------------------------------------------------------------
# Reading the cif files
# ---------------------------------------------------------------------------

def parse_atom_site(path):
    """Return the _atom_site records as dicts keyed by column name.

    designed.cif and the predicted files order their _atom_site columns
    differently. Reading by position gives the wrong field silently, so the
    header is read first and every column is located by name.
    """
    columns = []
    rows = []
    in_loop = False
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("_atom_site."):
                columns.append(stripped.split(".", 1)[1].split()[0])
                in_loop = True
                continue
            if not in_loop:
                continue
            if stripped.startswith("ATOM") or stripped.startswith("HETATM"):
                parts = stripped.split()
                if len(parts) == len(columns):
                    rows.append(dict(zip(columns, parts)))
            elif rows:
                break
    if not columns:
        raise ValueError("no _atom_site loop in %s" % path)
    return rows


def chain_sequences(path):
    """Map auth chain id to its one-letter sequence, in order of first appearance.

    Only CA atoms are read. That identifies a protein chain cheaply and it also
    drops every nucleic chain, because RNA and DNA carry no CA atom. Dropping
    them is wanted rather than incidental. Cas9 is released as a
    ribonucleoprotein whose designed.cif holds the binder, the Cas9 protein and
    a 98 nucleotide sgRNA, and the published metric masks exclude nucleic
    tokens. The filter leaves the two protein chains the mapping needs.
    """
    sequences = collections.OrderedDict()
    for record in parse_atom_site(path):
        if record.get("label_atom_id") != "CA":
            continue
        chain = record["auth_asym_id"]
        sequences.setdefault(chain, []).append(
            THREE_TO_ONE.get(record["label_comp_id"], "X"))
    return collections.OrderedDict(
        (chain, "".join(residues)) for chain, residues in sequences.items())


# ---------------------------------------------------------------------------
# Chain mapping
# ---------------------------------------------------------------------------

def resolve_roles(sequences):
    """Return (binder_chain, target_chain) for a two-chain complex.

    The binder is the shorter chain. This runs on designed.cif, after the CA
    filter has left only protein chains, so the target is the larger one.

    A target whose protein part spans more than one chain lands here as three
    or more chains and raises. That is the right outcome today, because
    compute_dockq reads a single target chain and its own docstring says a
    multimeric target needs the union treatment it does not yet have.
    """
    if len(sequences) != 2:
        raise ValueError("expected 2 protein chains, found %d: %s"
                         % (len(sequences), list(sequences)))
    by_length = sorted(sequences, key=lambda c: len(sequences[c]))
    return by_length[0], by_length[-1]


def map_predicted_to_reference(pred_sequences, ref_binder_seq, ref_target_seq):
    """Say which predicted chain is the binder and which is the target.

    Chain ids do not carry a fixed meaning across predictors. Measured over the
    780 released PD-L1 pairs, ESMFold2-Fast and ESMFold2-Full put the binder in
    chain A like designed.cif does, and the other eight predictors put the
    binder in chain B. Assuming A maps to A therefore compares the binder
    against the target for eight arms out of ten, and returns a plausible wrong
    number rather than an error. Roles are resolved by sequence instead.

    The designed target is a contiguous subsequence of the predicted target in
    every released pair, because designed.cif omits a few terminal target
    residues. Containment is therefore the test, not equality.
    """
    binder = target = None
    for chain, seq in pred_sequences.items():
        if seq == ref_binder_seq:
            binder = chain
        elif ref_target_seq in seq or seq in ref_target_seq:
            target = chain
    if binder is None or target is None or binder == target:
        raise ValueError(
            "could not resolve roles from sequence: predicted chains %s"
            % {c: len(s) for c, s in pred_sequences.items()})
    return binder, target


def relabel_predicted(path, rename):
    """Return the predicted mmCIF with its chain ids rewritten to the reference's.

    compute_dockq documents that "the same chain identifiers name the chains in
    both structures", and it reads one chain_mapping for both files. The
    released data breaks that assumption for eight of the ten arms, so the
    predicted chains are relabelled here before the call. Only the chain id
    columns of the _atom_site loop change. No coordinate is touched, so the
    score is unaffected.

    The two ids are swapped at once. Renaming A to B and then B to A in
    sequence would collapse both chains into one.
    """
    lines = []
    columns = []
    targets = []
    in_loop = False
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("_atom_site."):
                columns.append(stripped.split(".", 1)[1].split()[0])
                in_loop = True
                lines.append(line.rstrip("\n"))
                continue
            if in_loop and not targets and columns:
                targets = [i for i, name in enumerate(columns)
                           if name in ("auth_asym_id", "label_asym_id")]
            if in_loop and (stripped.startswith("ATOM") or stripped.startswith("HETATM")):
                values = stripped.split()
                if len(values) == len(columns):
                    for index in targets:
                        values[index] = rename.get(values[index], values[index])
                    lines.append(" ".join(values))
                    continue
            lines.append(line.rstrip("\n"))
    if not targets:
        raise ValueError("no chain id column in the _atom_site loop of %s" % path)
    return "\n".join(lines) + "\n"


def build_chain_mapping(pred_binder, pred_target, ref_binder, ref_target, style):
    """Build the chain_mapping argument for compute_dockq.

    binder_lane_fixture_adapter.py writes chain_mapping as
    {"target": ..., "binder": ...} over the predicted complex. ADAPTER-API.md
    does not say how the reference chains are named, and they differ from the
    predicted ones for eight of the ten arms. The extended style adds them under
    separate keys. The contract style sends only the two documented keys.
    """
    mapping = {"target": pred_target, "binder": pred_binder}
    if style == "extended":
        mapping["reference_target"] = ref_target
        mapping["reference_binder"] = ref_binder
    return mapping


# ---------------------------------------------------------------------------
# Finding the usable designs
# ---------------------------------------------------------------------------

Pair = collections.namedtuple(
    "Pair", "design_id target arm published predicted_path reference_path")


def load_published(summary_csv, targets):
    """Return the published sc_dockq per design and arm, keyed by design id."""
    published = {}
    with open(summary_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if targets and row["target"] not in targets:
                continue
            scores = {}
            for arm in ARMS:
                raw = row.get("sc_dockq_" + arm, "").strip()
                if raw:
                    scores[arm] = float(raw)
            published[row["full_name"]] = (row["target"], scores)
    return published


def find_pairs(data_root, published, targets, arms):
    """Return the designs usable for the check, and count what was dropped.

    A design is usable for an arm when the CSV carries an sc_dockq value for
    that arm and both cif files are on disk. The two sets differ, so the
    intersection is computed here rather than assumed.
    """
    pairs = []
    dropped = collections.Counter()
    for design_id, (target, scores) in sorted(published.items()):
        if targets and target not in targets:
            continue
        insilico = os.path.join(data_root, target, design_id, "insilico")
        if not os.path.isdir(insilico):
            dropped["design directory not downloaded"] += 1
            continue
        reference = os.path.join(insilico, REFERENCE_CIF)
        if not os.path.isfile(reference):
            dropped["no designed.cif in the released directory"] += 1
            continue
        on_disk = {}
        for name in os.listdir(insilico):
            match = PREDICTED_RE.match(name)
            if match:
                on_disk[match.group(1)] = os.path.join(insilico, name)
        for arm in arms:
            if arm not in scores:
                dropped["no published sc_dockq for the arm"] += 1
                continue
            if arm not in on_disk:
                dropped["no predicted cif for the arm"] += 1
                continue
            pairs.append(Pair(design_id, target, arm, scores[arm],
                              on_disk[arm], reference))
    return pairs, dropped


# ---------------------------------------------------------------------------
# The metrics lane
# ---------------------------------------------------------------------------

def load_compute_dockq():
    """Import compute_dockq late and say plainly when it is not usable.

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
    if not hasattr(binder_metrics, "compute_dockq"):
        raise SystemExit(
            "binder_metrics has no compute_dockq. The numeric comparison "
            "cannot run.")
    revision = getattr(binder_metrics, "DOCKQ_IMPLEMENTATION_REVISION", "unset")
    return binder_metrics.compute_dockq, revision


def looks_like_stub(revision, values):
    """Say whether the results came from a placeholder rather than a measurement.

    A stub returns the same constant for every design. Comparing a constant to
    the published column produces an undefined correlation, so the run is
    reported as not yet valid rather than as a result.
    """
    if "stub" in str(revision).lower():
        return "DOCKQ_IMPLEMENTATION_REVISION is %r" % revision
    if len(values) > 2 and len(set(values)) == 1:
        return "every design returned the same value, %r" % values[0]
    return None


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


def summarise(rows, revision, arms_seen):
    scored = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]

    print()
    print("=" * 72)
    print("sc_DockQ check against the published design_summary.csv")
    print("=" * 72)
    print("DOCKQ_IMPLEMENTATION_REVISION: %s" % revision)
    print("pairs attempted: %d" % len(rows))
    print("pairs scored:    %d" % len(scored))
    print("pairs failed:    %d" % len(failed))

    if failed:
        reasons = collections.Counter(r["detail"].split(":")[0] for r in failed)
        print("\nfailures by reason:")
        for reason, count in reasons.most_common():
            print("   %-56s %d" % (reason[:56], count))

    if not scored:
        print("\nNo pair was scored. No correlation and no difference are reported.")
        return

    ours = [r["our_sc_dockq"] for r in scored]
    theirs = [r["published_sc_dockq"] for r in scored]
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
    print("mean absolute diff      %.6f" % statistics.fmean(diffs))
    print("median absolute diff    %.6f" % statistics.median(diffs))
    print("max absolute diff       %.6f" % max(diffs))

    print("\nper arm:")
    print("   %-9s %6s %10s %12s" % ("arm", "n", "mean |d|", "pearson r"))
    for arm in arms_seen:
        subset = [r_ for r_ in scored if r_["arm"] == arm]
        if not subset:
            continue
        sub_r = pearson([s["our_sc_dockq"] for s in subset],
                        [s["published_sc_dockq"] for s in subset])
        print("   %-9s %6d %10.6f %12s"
              % (arm, len(subset),
                 statistics.fmean([s["abs_diff"] for s in subset]),
                 "undefined" if sub_r is None else "%.6f" % sub_r))

    print("\nfive worst by absolute difference:")
    worst = sorted(scored, key=lambda r_: r_["abs_diff"], reverse=True)[:5]
    print("   %-46s %-8s %9s %9s %8s"
          % ("design", "arm", "published", "ours", "|diff|"))
    for row in worst:
        print("   %-46s %-8s %9.5f %9.5f %8.5f"
              % (row["design_id"][:46], row["arm"], row["published_sc_dockq"],
                 row["our_sc_dockq"], row["abs_diff"]))


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True,
                        help="published design summary CSV")
    parser.add_argument("--data-root", required=True,
                        help="directory holding <target>/<design>/insilico/")
    parser.add_argument("--out", required=True, help="results CSV to write")
    parser.add_argument("--targets", nargs="*", default=None,
                        help="limit to these targets, default is every target found")
    parser.add_argument("--arms", nargs="*", default=ARMS, choices=ARMS,
                        help="limit to these arms, default is all ten")
    parser.add_argument("--ranking-arms-only", action="store_true",
                        help="use only %s" % ", ".join(RANKING_ARMS))
    parser.add_argument("--mapping-keys", choices=("extended", "contract"),
                        default="extended",
                        help="whether chain_mapping carries the reference chains")
    parser.add_argument("--chain-normalisation", choices=("relabel", "none"),
                        default="relabel",
                        help="relabel predicted chains to the reference ids, "
                             "which compute_dockq requires")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many pairs, for a quick check")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve chains and report the intersection, call nothing")
    args = parser.parse_args()

    arms = list(RANKING_ARMS) if args.ranking_arms_only else args.arms
    targets = set(args.targets) if args.targets else None

    published = load_published(args.summary_csv, targets)
    pairs, dropped = find_pairs(args.data_root, published, targets, arms)

    print("published rows read:  %d" % len(published))
    print("usable pairs found:   %d" % len(pairs))
    if dropped:
        print("dropped:")
        for reason, count in dropped.most_common():
            print("   %-52s %d" % (reason, count))

    counts = collections.Counter((p.target, p.arm) for p in pairs)
    if counts:
        print("\nusable designs per target and arm:")
        for (target, arm), count in sorted(counts.items()):
            print("   %-10s %-9s %d" % (target, arm, count))

    if args.limit:
        pairs = pairs[:args.limit]
        print("\nlimited to %d pairs" % len(pairs))

    if not pairs:
        raise SystemExit("\nNo usable pair was found. Nothing was compared.")

    compute_dockq = None
    revision = "not imported, dry run"
    if not args.dry_run:
        compute_dockq, revision = load_compute_dockq()

    rows = []
    reference_cache = {}
    for index, pair in enumerate(pairs, 1):
        row = {field: "" for field in CSV_FIELDS}
        row.update(design_id=pair.design_id, target=pair.target, arm=pair.arm,
                   published_sc_dockq=pair.published, status="ok", detail="")
        try:
            if pair.reference_path not in reference_cache:
                reference_cache[pair.reference_path] = chain_sequences(
                    pair.reference_path)
            ref_seqs = reference_cache[pair.reference_path]
            ref_binder, ref_target = resolve_roles(ref_seqs)

            pred_seqs = chain_sequences(pair.predicted_path)
            pred_binder, pred_target = map_predicted_to_reference(
                pred_seqs, ref_seqs[ref_binder], ref_seqs[ref_target])

            row.update(pred_target_chain=pred_target, pred_binder_chain=pred_binder,
                       ref_target_chain=ref_target, ref_binder_chain=ref_binder,
                       chain_ids_agree=str(pred_binder == ref_binder
                                           and pred_target == ref_target))

            if args.dry_run:
                row["status"] = "chains-resolved"
            else:
                # compute_dockq applies one chain_mapping to both structures, so
                # the predicted chains are relabelled to the reference's ids and
                # the mapping is written in reference terms.
                if args.chain_normalisation == "relabel" and (
                        pred_binder != ref_binder or pred_target != ref_target):
                    predicted = relabel_predicted(
                        pair.predicted_path,
                        {pred_binder: ref_binder, pred_target: ref_target})
                    call_binder, call_target = ref_binder, ref_target
                    row["chain_normalisation"] = "relabelled %s->%s %s->%s" % (
                        pred_binder, ref_binder, pred_target, ref_target)
                else:
                    predicted = pathlib.Path(pair.predicted_path)
                    call_binder, call_target = pred_binder, pred_target
                    row["chain_normalisation"] = "none"

                mapping = build_chain_mapping(call_binder, call_target,
                                              ref_binder, ref_target,
                                              args.mapping_keys)
                result = compute_dockq(predicted,
                                       pathlib.Path(pair.reference_path),
                                       mapping)
                if not isinstance(result, dict) or "sc_dockq" not in result:
                    raise ValueError("compute_dockq returned %r without sc_dockq"
                                     % type(result).__name__)
                ours = float(result["sc_dockq"])
                row["our_sc_dockq"] = ours
                row["abs_diff"] = abs(ours - pair.published)
                row["mapping_status"] = str(result.get("mapping_status", ""))
        except NotImplementedError as exc:
            row["status"] = "not-implemented"
            row["detail"] = "compute_dockq raised NotImplementedError: %s" % exc
        except Exception as exc:                       # noqa: BLE001
            row["status"] = "error"
            row["detail"] = "%s: %s" % (type(exc).__name__, exc)
        rows.append(row)
        if index % 100 == 0:
            print("   %d / %d" % (index, len(pairs)))

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("\nwrote %s" % os.path.abspath(args.out))

    if args.dry_run:
        resolved = sum(1 for r in rows if r["status"] == "chains-resolved")
        print("chains resolved for %d of %d pairs" % (resolved, len(rows)))
        conventions = collections.Counter(
            (r["arm"], "binder=%s target=%s" % (r["pred_binder_chain"],
                                                r["pred_target_chain"]))
            for r in rows if r["status"] == "chains-resolved")
        print("\nchain convention found per arm:")
        for (arm, convention), count in sorted(conventions.items()):
            print("   %-9s %-24s %d" % (arm, convention, count))
        failures = [r for r in rows if r["status"] != "chains-resolved"]
        if failures:
            print("\nunresolved (%d), first five:" % len(failures))
            for row in failures[:5]:
                print("   %s %s %s" % (row["design_id"], row["arm"], row["detail"]))
        print("\nNo sc_DockQ was computed. This was a dry run.")
        return

    summarise(rows, revision, arms)


if __name__ == "__main__":
    main()
