#!/usr/bin/env python3
"""Write a negative-control structure, and its campaign block, from a complex you already have.

A campaign requires a negative control unless it disables the control panel.
`controls.minimum_negative_controls` defaults to 1, `lane check` refuses a
campaign whose enabled negative panel is smaller than that, and the
qualification gate accepts `PASS` only when a run records
`min_positive > max_negative` for `ipsae_min`. An ungated candidate claim may
set both minimums to 0 and declare no controls, which drops `control-calibration`
and stamps every score row `score_gating.mode: ungated`. The five fields that
combination needs are listed under "Run with no panel at all" in
skills/claude-binder-lane/references/target-qualification.md. Until this script existed
the package stated that requirement and shipped no way to satisfy it. The
in-stage constructor in `adapters/control_builder.py` can build a decoy, but only
from a configuration shape that `lane check` and `lane materialize` both refuse,
because each requires `structure_path` on every declared control.

This script closes that gap from the other side. It writes the decoy to disk
before the campaign is composed, so the negative reaches the run as an ordinary
supplied control with a path the validator can resolve and a file the bundler can
hash.

What each kind of control establishes
-------------------------------------

`sequence-decoy` keeps the target chain and the binder backbone and permutes the
binder's residue identities. The permutation preserves amino-acid composition and
length, so the decoy differs from its source in sequence alone. It measures
whether a predictor's interface score responds to the binder sequence at all,
rather than to the pose it was handed. It does not establish binding specificity.
A scrambled sequence can still score well: the package records scrambled controls
reaching Boltz iPTM between 0.80 and 0.89 and ipSAE up to 0.65, in
`references/tool-catalogue.md` under "A single predictor's confidence is not
evidence of binding". A sequence decoy that scores high is evidence about the
predictor, not about the binder.

`cross-pair` takes the target chain from one complex and the binder chain from a
different complex. The two were never observed together. It measures whether a
predictor reports a confident interface for a pair with no evidence behind it. It
does not establish that a real binder is selective for its target, because it
varies the binder rather than the target. A counter-screen against a paralog does
that, through `scoring.counter_screen`.

Neither kind is a substitute for a literature or experimental complex on the
positive side. A control panel separates only when its positive is a real binder.

What this script will not build
-------------------------------

It will not build a pose decoy. A pose decoy rotates or translates the binder and
leaves its sequence unchanged, and the control builder refuses any negative whose
binder sequence equals a positive's, because the rescore folds the binder from
`sequence_path` and reads the pose only as the reference for the pose metrics. A
same-sequence negative would be handed the same fold as the positive.

It will not choose thresholds. The emitted gate carries the campaign's
`scoring.thresholds.negative_control_maximum_ipsae_min` when `--campaign` names a
campaign that sets it, and `__REQUIRED__` otherwise. A threshold that no run
measured is not a value this script invents.

It will not judge whether a binder binds a target. Only the scientist can say
that. `--campaign` buys one narrow check instead: the chosen binder is compared
against every positive control the campaign declares and can resolve, and a match
is refused, because the control builder would refuse it later. The summary reports
how many positives were compared, and the count is zero without `--campaign`.

Exit code. Zero means the structure and the campaign block were written. One
means an input was missing or did not match the structure, and the message says
which.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from .adapters.control_builder import (
    AMINO_ACID_1_TO_3,
    AdapterError,
    cross_pair_structure,
    is_mmcif,
    pdb_chain_sequence,
    sha256_file,
    shuffle_sequence,
    rewrite_residue_names,
)

NEGATIVE_GATE_THRESHOLD_FIELD = "negative_control_maximum_ipsae_min"
REQUIRED_TOKEN = "__REQUIRED__"

SEQUENCE_DECOY_ESTABLISHES = (
    "Measures whether the predictor's interface score responds to the binder "
    "sequence, or only to the pose it was handed."
)
SEQUENCE_DECOY_DOES_NOT_ESTABLISH = (
    "Does not establish binding specificity. Scrambled controls have scored Boltz "
    "iPTM 0.80 to 0.89 and ipSAE up to 0.65 in this package's own records."
)
CROSS_PAIR_ESTABLISHES = (
    "Measures whether the predictor reports a confident interface for a target and "
    "a binder that were never observed together."
)
CROSS_PAIR_DOES_NOT_ESTABLISH = (
    "Does not establish that a real binder is selective for this target, because it "
    "varies the binder rather than the target. A paralog counter-screen does that."
)
CROSS_PAIR_CALLER_ASSERTS = (
    "That this binder does not bind this target is your assertion, not a measurement. "
    "This checks only that the two sources are different files, that their binder "
    "sequences differ, and that the chosen binder is not a positive control the "
    "campaign declares."
)


class InputError(RuntimeError):
    """An input condition that makes the requested control unsafe to write."""


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _atom_site_header(text: str) -> list[str]:
    """Return the `_atom_site` loop's column names, in file order."""
    return [
        line.strip().split(".", 1)[1]
        for line in text.splitlines()
        if line.strip().startswith("_atom_site.")
    ]


def _token_spans(line: str) -> list[tuple[int, int]]:
    """Return each whitespace-delimited token's half-open span in one line."""
    spans: list[tuple[int, int]] = []
    index = 0
    length = len(line)
    while index < length:
        while index < length and line[index].isspace():
            index += 1
        if index >= length:
            break
        start = index
        while index < length and not line[index].isspace():
            index += 1
        spans.append((start, index))
    return spans


def rewrite_mmcif_residue_names(path: Path, chain_id: str, sequence: str) -> str:
    """Relabel one mmCIF chain's residues, in the order `pdb_chain_sequence` reads them.

    The residue order, the chain column, and the residue-key column are the ones
    `control_builder.mmcif_chain_sequence` uses, so the sequence this writes is
    the sequence the control builder reads back. Every three-letter code is three
    characters wide, so replacing a code in place leaves every other column and
    every byte offset on the line unchanged.
    """
    text = _read_text(path)
    header = _atom_site_header(text)
    if not header:
        raise InputError(f"structure has no _atom_site loop: {path}")
    index = {name: position for position, name in enumerate(header)}
    chain_key = "auth_asym_id" if "auth_asym_id" in index else "label_asym_id"
    seq_key = "auth_seq_id" if "auth_seq_id" in index else "label_seq_id"
    for required in (chain_key, seq_key, "label_comp_id", "group_PDB"):
        if required not in index:
            raise InputError(f"structure _atom_site loop has no {required}: {path}")
    name_columns = sorted(
        index[name] for name in ("label_comp_id", "auth_comp_id") if name in index
    )
    width = max(index.values())

    def selected(line: str) -> list[str] | None:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            return None
        fields = line.split()
        if len(fields) <= width:
            return None
        if fields[index["group_PDB"]] != "ATOM":
            return None
        if fields[index[chain_key]] != chain_id:
            return None
        return fields

    order: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        fields = selected(line)
        if fields is None:
            continue
        key = fields[index[seq_key]]
        if key not in seen:
            seen.add(key)
            order.append(key)
    if len(order) != len(sequence):
        raise InputError(
            f"structure {path} chain {chain_id} has {len(order)} residues, "
            f"but the replacement sequence has {len(sequence)}"
        )
    replacement = dict(zip(order, sequence))

    output: list[str] = []
    for line in text.splitlines():
        fields = selected(line)
        if fields is None:
            output.append(line)
            continue
        residue = AMINO_ACID_1_TO_3[replacement[fields[index[seq_key]]]]
        spans = _token_spans(line)
        if len(spans) != len(fields):
            raise InputError(
                f"structure {path} has a quoted or padded _atom_site field this "
                f"script cannot rewrite: {line.strip()[:60]}"
            )
        rewritten = line
        for column in reversed(name_columns):
            start, end = spans[column]
            if end - start != len(residue):
                raise InputError(
                    f"structure {path} carries a residue name that is not three "
                    f"characters wide: {line[start:end]!r}"
                )
            rewritten = rewritten[:start] + residue + rewritten[end:]
        output.append(rewritten)
    return "\n".join(output) + "\n"


def rewrite_residues(path: Path, chain_id: str, sequence: str) -> str:
    """Relabel one chain's residues in whichever of the two formats the file uses."""
    if is_mmcif(_read_text(path)):
        return rewrite_mmcif_residue_names(path, chain_id, sequence)
    return rewrite_residue_names(path, chain_id, sequence)


def require_structure(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise InputError(f"{label} is missing: {path}")
    return path.resolve()


def require_chains(path: Path, target_chain: str, binder_chain: str) -> tuple[str, str]:
    """Read both declared chains, so an absent or non-protein chain refuses early."""
    if target_chain == binder_chain:
        raise InputError(
            f"--target-chain and --binder-chain must differ; both are {target_chain!r}"
        )
    try:
        target_sequence = pdb_chain_sequence(path, target_chain)
        binder_sequence = pdb_chain_sequence(path, binder_chain)
    except AdapterError as exc:
        raise InputError(str(exc)) from exc
    return target_sequence, binder_sequence


def sequence_identity(left: str, right: str) -> float:
    if len(left) != len(right) or not left:
        raise InputError("sequence identity is defined only over equal, non-empty lengths")
    return sum(1 for a, b in zip(left, right) if a == b) / len(left)


def load_campaign(campaign_path: str | None) -> tuple[dict[str, Any], Path] | None:
    if campaign_path is None:
        return None
    path = require_structure(campaign_path, "--campaign")
    try:
        campaign = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise InputError(f"--campaign is not readable JSON: {path}: {exc}") from exc
    if not isinstance(campaign, dict):
        raise InputError(f"--campaign must be a JSON object: {path}")
    return campaign, path


def gate_threshold(campaign: tuple[dict[str, Any], Path] | None) -> Any:
    """Take the negative gate from the campaign, or leave the placeholder in place."""
    if campaign is None:
        return REQUIRED_TOKEN
    thresholds = campaign[0].get("scoring", {})
    thresholds = thresholds.get("thresholds") if isinstance(thresholds, dict) else None
    value = thresholds.get(NEGATIVE_GATE_THRESHOLD_FIELD) if isinstance(thresholds, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return REQUIRED_TOKEN
    return value


def positive_binder_sequences(
    campaign: tuple[dict[str, Any], Path] | None,
) -> dict[str, str]:
    """Read the binder sequence of every positive control the campaign can resolve.

    The control builder refuses a negative whose binder sequence equals a
    positive's, because such a negative separates nothing. Reading the campaign
    here moves that refusal to build time, where the scientist can still choose a
    different source. A positive whose structure is a placeholder or is missing
    contributes nothing, and the summary reports how many were compared.
    """
    if campaign is None:
        return {}
    config, config_path = campaign
    controls = config.get("controls")
    positives = controls.get("positive") if isinstance(controls, dict) else None
    sequences: dict[str, str] = {}
    for control in positives if isinstance(positives, list) else []:
        if not isinstance(control, dict) or control.get("enabled", True) is not True:
            continue
        value = control.get("structure_path")
        chain = control.get("binder_chain")
        if not isinstance(value, str) or not value or REQUIRED_TOKEN in value:
            continue
        if not isinstance(chain, str) or not chain:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        if not path.is_file():
            continue
        try:
            sequences[str(control.get("id"))] = pdb_chain_sequence(path.resolve(), chain)
        except AdapterError:
            continue
    return sequences


def require_distinct_from_positives(
    binder_sequence: str,
    positives: dict[str, str],
) -> None:
    for control_id, sequence in positives.items():
        if sequence == binder_sequence:
            raise InputError(
                f"the chosen binder sequence equals positive control {control_id!r} in "
                "--campaign, so this negative would separate nothing and the control "
                "builder would refuse it. Choose a binder the campaign does not declare "
                "as a positive control."
            )


def control_block(
    *,
    control_id: str,
    role: str,
    structure_path: str,
    target_chain: str,
    binder_chain: str,
    threshold: Any,
) -> dict[str, Any]:
    return {
        "id": control_id,
        "role": role,
        "enabled": True,
        "structure_path": structure_path,
        "target_chain": target_chain,
        "binder_chain": binder_chain,
        "gates": [
            {"metric": "ipsae_min", "operator": "maximum", "threshold": threshold},
        ],
    }


def write_outputs(
    out_path: Path,
    text: str,
    block: dict[str, Any],
    *,
    force: bool,
) -> Path:
    block_path = out_path.with_suffix(out_path.suffix + ".control-block.json")
    for path in (out_path, block_path):
        if path.exists() and not force:
            raise InputError(f"output exists; pass --force to replace it: {path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    block_path.write_text(json.dumps(block, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return block_path


def build_sequence_decoy(args: argparse.Namespace) -> dict[str, Any]:
    source = require_structure(args.structure, "--structure")
    target_sequence, binder_sequence = require_chains(source, args.target_chain, args.binder_chain)
    out_path = Path(args.out).expanduser()
    if out_path.suffix.lower() != source.suffix.lower():
        raise InputError(
            f"--out must keep the source format, so its suffix must be "
            f"{source.suffix.lower()}: {out_path}"
        )
    decoy_sequence = shuffle_sequence(binder_sequence, random.Random(args.shuffle_seed))
    if decoy_sequence == binder_sequence:
        raise InputError(
            f"chain {args.binder_chain} of {source} admits no permutation distinct from "
            "itself, so it cannot become a sequence decoy"
        )
    identity = sequence_identity(decoy_sequence, binder_sequence)
    campaign = load_campaign(args.campaign)
    positives = positive_binder_sequences(campaign)
    require_distinct_from_positives(decoy_sequence, positives)
    text = rewrite_residues(source, args.binder_chain, decoy_sequence)
    block = control_block(
        control_id=args.control_id,
        role="sequence-decoy",
        structure_path=args.structure_path or str(out_path),
        target_chain=args.target_chain,
        binder_chain=args.binder_chain,
        threshold=gate_threshold(campaign),
    )
    block_path = write_outputs(out_path, text, block, force=args.force)
    return {
        "ok": True,
        "kind": "sequence-decoy",
        "role": "sequence-decoy",
        "source": str(source),
        "source_binder_sequence": binder_sequence,
        "decoy_binder_sequence": decoy_sequence,
        "shuffle_seed": args.shuffle_seed,
        "identity_to_source": round(identity, 4),
        "positive_controls_compared": len(positives),
        "target_chain": args.target_chain,
        "target_residue_count": len(target_sequence),
        "binder_chain": args.binder_chain,
        "binder_residue_count": len(binder_sequence),
        "structure": str(out_path.resolve()),
        "structure_sha256": sha256_file(out_path),
        "control_block": str(block_path.resolve()),
        "gate_threshold": block["gates"][0]["threshold"],
        "establishes": SEQUENCE_DECOY_ESTABLISHES,
        "does_not_establish": SEQUENCE_DECOY_DOES_NOT_ESTABLISH,
    }


def build_cross_pair(args: argparse.Namespace) -> dict[str, Any]:
    target_source = require_structure(args.target_structure, "--target-structure")
    binder_source = require_structure(args.binder_structure, "--binder-structure")
    if target_source == binder_source:
        raise InputError(
            "--target-structure and --binder-structure resolve to one file, so the "
            "pair would be the complex it came from rather than a mismatch"
        )
    out_path = Path(args.out).expanduser()
    if out_path.suffix.lower() not in {".pdb", ".ent"}:
        raise InputError(f"--out must be a PDB, because cross-pair writes PDB: {out_path}")
    try:
        target_chain_sequence = pdb_chain_sequence(target_source, args.target_source_chain)
        binder_chain_sequence = pdb_chain_sequence(binder_source, args.binder_source_chain)
    except AdapterError as exc:
        raise InputError(str(exc)) from exc
    try:
        native_binder = pdb_chain_sequence(target_source, args.binder_source_chain)
    except AdapterError:
        native_binder = None
    if native_binder is not None and native_binder == binder_chain_sequence:
        raise InputError(
            f"chain {args.binder_source_chain} of {binder_source} carries the same "
            f"sequence as chain {args.binder_source_chain} of {target_source}, so the "
            "pair is not a mismatch"
        )
    if args.target_chain == args.binder_chain:
        raise InputError(
            f"--target-chain and --binder-chain must differ; both are {args.target_chain!r}"
        )
    campaign = load_campaign(args.campaign)
    positives = positive_binder_sequences(campaign)
    require_distinct_from_positives(binder_chain_sequence, positives)
    try:
        text = cross_pair_structure(
            target_source,
            args.target_source_chain,
            binder_source,
            args.binder_source_chain,
            args.target_chain,
            args.binder_chain,
        )
    except AdapterError as exc:
        raise InputError(str(exc)) from exc
    block = control_block(
        control_id=args.control_id,
        role="matched-wrong-pair",
        structure_path=args.structure_path or str(out_path),
        target_chain=args.target_chain,
        binder_chain=args.binder_chain,
        threshold=gate_threshold(campaign),
    )
    block_path = write_outputs(out_path, text, block, force=args.force)
    return {
        "ok": True,
        "kind": "cross-pair",
        "role": "matched-wrong-pair",
        "target_source": str(target_source),
        "target_source_chain": args.target_source_chain,
        "target_residue_count": len(target_chain_sequence),
        "binder_source": str(binder_source),
        "binder_source_chain": args.binder_source_chain,
        "binder_residue_count": len(binder_chain_sequence),
        "binder_sequence": binder_chain_sequence,
        "positive_controls_compared": len(positives),
        "target_chain": args.target_chain,
        "binder_chain": args.binder_chain,
        "structure": str(out_path.resolve()),
        "structure_sha256": sha256_file(out_path),
        "control_block": str(block_path.resolve()),
        "gate_threshold": block["gates"][0]["threshold"],
        "establishes": CROSS_PAIR_ESTABLISHES,
        "does_not_establish": CROSS_PAIR_DOES_NOT_ESTABLISH,
        "caller_asserts": CROSS_PAIR_CALLER_ASSERTS,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m claude_binder.make_negative_control",
        description=(
            "Write a negative-control structure and its controls.negative block from a "
            "complex you already have."
        ),
    )
    subparsers = parser.add_subparsers(dest="kind", required=True)

    def shared(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--control-id", required=True, help="controls.negative[].id")
        subparser.add_argument("--out", required=True, help="path to write the control structure")
        subparser.add_argument(
            "--structure-path",
            help=(
                "value to write into controls.negative[].structure_path, when the campaign "
                "resolves it relative to itself rather than by the absolute --out path"
            ),
        )
        subparser.add_argument(
            "--campaign",
            help=(
                "campaign JSON to take the gate threshold from, through "
                f"scoring.thresholds.{NEGATIVE_GATE_THRESHOLD_FIELD}"
            ),
        )
        subparser.add_argument("--force", action="store_true", help="replace existing outputs")
        subparser.add_argument("--json", action="store_true", help="print the summary as JSON")

    decoy = subparsers.add_parser(
        "sequence-decoy",
        help="permute one complex's binder residues and keep its target and backbone",
    )
    decoy.add_argument("--structure", required=True, help="a complex you already have, PDB or mmCIF")
    decoy.add_argument("--target-chain", required=True, help="the chain that stays the target")
    decoy.add_argument("--binder-chain", required=True, help="the chain whose residues are permuted")
    decoy.add_argument(
        "--shuffle-seed",
        type=int,
        required=True,
        help="the seed that makes the permutation reproducible",
    )
    shared(decoy)

    cross = subparsers.add_parser(
        "cross-pair",
        help="join one complex's target chain to a different complex's binder chain",
    )
    cross.add_argument("--target-structure", required=True, help="PDB carrying the target chain")
    cross.add_argument("--target-source-chain", required=True, help="target chain in that PDB")
    cross.add_argument("--binder-structure", required=True, help="a different PDB carrying the binder")
    cross.add_argument("--binder-source-chain", required=True, help="binder chain in that PDB")
    cross.add_argument("--target-chain", required=True, help="target chain ID in the written control")
    cross.add_argument("--binder-chain", required=True, help="binder chain ID in the written control")
    shared(cross)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.kind == "sequence-decoy":
        return build_sequence_decoy(args)
    return build_cross_pair(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (InputError, AdapterError) as exc:
        print(f"make_negative_control: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    print(f"role              {summary['role']}")
    print(f"structure         {summary['structure']}")
    print(f"sha256            {summary['structure_sha256']}")
    print(f"target chain      {summary['target_chain']} ({summary['target_residue_count']} residues)")
    print(f"binder chain      {summary['binder_chain']} ({summary['binder_residue_count']} residues)")
    if summary["kind"] == "sequence-decoy":
        print(f"source            {summary['source']}")
        print(f"shuffle seed      {summary['shuffle_seed']}")
        print(f"identity          {summary['identity_to_source']:.4f} to the source binder sequence")
    else:
        print(f"target source     {summary['target_source']} chain {summary['target_source_chain']}")
        print(f"binder source     {summary['binder_source']} chain {summary['binder_source_chain']}")
    print(f"control block     {summary['control_block']}")
    compared = summary["positive_controls_compared"]
    if compared:
        print(f"positives checked {compared} campaign positive control(s), none matching this binder")
    else:
        print(
            "positives checked none. Pass --campaign so this refuses a binder the "
            "campaign already declares as a positive control"
        )
    if summary["gate_threshold"] == REQUIRED_TOKEN:
        print(
            f"TODO              gates[0].threshold is {REQUIRED_TOKEN} and fails check "
            "until a control-calibration run measures it"
        )
    print(f"establishes       {summary['establishes']}")
    print(f"does not          {summary['does_not_establish']}")
    if "caller_asserts" in summary:
        print(f"you assert        {summary['caller_asserts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
