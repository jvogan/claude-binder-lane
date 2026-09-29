#!/usr/bin/env python3
"""Design SolubleCaliby sequences for one binder lane stage phase, on a local install.

This wrapper fills the `solublecaliby-designer` slot in the published baseline.
It runs the Hydra entry points of an operator-installed Caliby checkout as child
processes. This package configures a local-process route against an environment
the operator builds, the same class of route `proteinmpnn_designer.py` already
serves for SolubleMPNN.

Install Caliby from https://github.com/ProteinDesignLab/caliby and point this
wrapper at the checkout with --caliby-root or CALIBY_ROOT. Caliby needs Python
3.12 or newer with Torch, Hydra and Lightning, which is not this package's
environment, so --tool-python names the interpreter of the Caliby environment.

**The checkpoint is `soluble_caliby_v1`, and the reason is the training set, not
the example script.** The upstream README's checkpoint table at source revision
41d31560c3c73d7980d94f40f3c852b90bfab5c0 describes `soluble_caliby` as "Trained
on monomers only" and `soluble_caliby_v1` as "SolubleCaliby trained on both
monomers and interfaces". A binder campaign designs at an interface. The
repository's `examples/scripts/seq_des.sh` selects `soluble_caliby_v1` as well,
and it is the only example script that selects a soluble checkpoint at all, but
the training set is the argument and the example is the corroboration.
`soluble_caliby_distill` also ships and skips ensemble generation entirely. It is
not in the published baseline, so nothing here binds it.

**The checkpoint arrives as a file path and a digest, never as a name.** Upstream
`caliby/weights.py` resolves a registry name by downloading from Hugging Face on
first use, at whatever the repository's main branch holds, and returns a literal
path unchanged when the value ends in `.ckpt`. A name therefore pins nothing and
reaches the network at job time. `--checkpoint` takes the file and
`--checkpoint-sha256` is required, this wrapper hashes the file before the first
call, and it refuses the stage when the two differ.

**Without a position constraint Caliby redesigns the target.** Upstream states it
twice, in the README and in `parse_fixed_pos_info`, which prints "No fixed
positions specified, redesigning all positions." A binder campaign hands Caliby a
two-chain complex, so an unconstrained run returns a redesigned target and a
downstream stage scores a complex nobody asked for. This wrapper writes a
`pos_constraint_csv` for every run, fixing every residue of every chain except
--design-chain, and it refuses to run without --design-chain.

**The constraint deliberately uses Caliby's chain-only syntax.** The upstream
parser accepts A as well as A1-100 and masks every token in chain A for the
former. This adapter always fixes complete non-design chains, so it writes A
or A,B, rather than manufacturing numeric ranges from author numbering. That
avoids asserting that a PDB file's record order or a campaign manifest's ordinal
is the same identifier AtomWorks parsed for a particular input. It also applies
unchanged after the ensemble entry point expands one constraint row to its
conformers.

The optional target manifest remains an audit guard: this wrapper checks that a
fixed chain in the pose has the same ordered author identities as the prepared
target before recording it. The rows retain those author ranges and the source
of the observed numbering map, but the values never become a Caliby constraint.

The upstream entry points support fixed-backbone and ensemble-conditioned
design. This adapter exposes them as separate subcommands because the ensemble
path first generates conformers with Protpardelle-1c and stages its weights with
ProteinMPNN's. Fixed-backbone design runs Caliby alone. The profile binding in
this package selects only the fixed-backbone subcommand.

  generate-ensembles  Protpardelle-1c partial diffusion, one ensemble per backbone.
  run                 Fixed-backbone design. One Caliby call for the phase.
  run-ensemble        Ensemble-conditioned design against a conformer directory.
  parse               Check the phase outputs this stage declares.
  toolcheck           Report this adapter's own readiness. Executes nothing.

Receipt-owned outputs, the same layout `proteinmpnn_designer.py` writes:

  <attempt>/<phase>/sequences/<candidate_id>.fasta      one record per candidate
  <attempt>/<phase>/poses/<candidate_id>.pdb            one design pose per candidate
  <attempt>/<phase>/sequence-candidate-manifest.jsonl   one row per candidate
  <attempt>/<phase>/caliby/                             the staged inputs, the Hydra
                                                        run and the engine-native
                                                        `.cif` files, unchanged

The design pose carries the upstream backbone's atom records, which is what
`proteinmpnn_designer.write_design_pose` writes and what the DockQ and ESMFold
consumers read. Caliby's own `.cif` output is kept beside it in the work
directory rather than converted, so a reader can go back to what the engine
wrote.

**What this wrapper asserts, and what it does not.** Everything above about the
command it composes, the constraint CSV it writes, the chain grammar it uses
and the columns it parses is tested and is real. Four things are read from
upstream source at the pinned revision and have never been observed on this
package's hardware, so treat them as shape rather than as measurement:

  1. That `seq_des.py` and `seq_des_ensemble.py` write `seq_des_outputs.csv` with
     the columns `example_id`, `out_pdb`, `U`, `input_seq`, `seq`, and designed
     structures under `samples/`. Read from `eval/eval_utils/seq_des_utils.py`.
  2. That `max_num_conformers=32` with `include_primary_conformer=true` selects
     the primary structure plus the first 31 generated conformers, 32 in total
     rather than 33. Read from `eval_setup_utils.process_conformer_dirs`, which
     builds `[primary] + all_conformers[: max_num_conformers - 1]`.
  3. That chains in the `seq` column are separated by ":" in alphabetical order
     of chain ID. Read from the README's multichain scoring section.
  4. **That AtomWorks parses the poses this package writes at all.** This is the
     largest untested risk in the route and it is not small. Caliby reads
     structures through AtomWorks, upstream recommends running its own
     `clean_pdbs` step first for structures that came from another pipeline, and
     the poses here come from generators. No test in this package can settle it;
     one `clean_pdbs` call against one demo pose would.

No campaign has run this route. It carries no qualification receipt, no measured
cost and no measured hardware figure.

Exit code. Zero means the stage wrote its manifest. One means it did not.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from claude_binder.adapters import residue_numbering as numbering
from claude_binder.adapters import structure_evidence as evidence
from claude_binder.adapters.proteinmpnn_designer import (
    PARENT_LINEAGE_FIELDS,
    AdapterError,
    backbone_pose,
    canonical_sequence_sha256,
    check_sequence,
    completed_receipt_rows,
    load_jsonl,
    sha256_file,
    write_design_pose,
    write_jsonl,
    write_sequence,
)
from claude_binder.adapters.candidate_lineage import DIVERSITY_LINEAGE_FIELDS


DESIGNER_ID = "solublecaliby"
ADAPTER_ID = "solublecaliby-designer"
ROUTE_ID = "caliby-local-cli"
ROUTE_CONTRACT_REVISION = "local-seq-des-v1"
# `qualify.COST_KIND_UNPRICED`. Nothing measured this tool on this route.
COST_BASIS = "unpriced"

UPSTREAM_REPOSITORY = "https://github.com/ProteinDesignLab/caliby"
# Read from the GitHub commits API on 2026-09-12 and confirmed through the trees
# API. Every upstream file cited in this module was read at this revision.
UPSTREAM_SOURCE_REVISION = "41d31560c3c73d7980d94f40f3c852b90bfab5c0"
WEIGHTS_HOST = "https://huggingface.co/ProteinDesignLab/caliby-weights"
# Read from the Hugging Face model API on 2026-09-12. `cardData.license` at the
# same revision reads apache-2.0.
WEIGHTS_REVISION = "a51f011f4ec7ffe2daad3ab9b2bfb67ff628a096"
BOUND_CHECKPOINT_NAME = "soluble_caliby_v1"
BOUND_CHECKPOINT_FILE = "caliby/soluble_caliby_v1.ckpt"
# Pinned by `pyproject.toml` at the source revision above. MIT at that commit.
PROTPARDELLE_REVISION = "7962da091a335251fa8e5ddef5d2c937fbd9d9ae"

SEQ_DES_SCRIPT = "caliby/eval/sampling/seq_des.py"
SEQ_DES_ENSEMBLE_SCRIPT = "caliby/eval/sampling/seq_des_ensemble.py"
GENERATE_ENSEMBLES_SCRIPT = "caliby/eval/sampling/generate_ensembles.py"
REQUIRED_SCRIPTS = (SEQ_DES_SCRIPT, SEQ_DES_ENSEMBLE_SCRIPT, GENERATE_ENSEMBLES_SCRIPT)
# `seq_des.yaml` points `seq_des_cfg.atom_mpnn.sampling_cfg` at the relative path
# `caliby/configs/seq_des/inference.yaml`, so the child runs with the checkout as
# its working directory or Hydra resolves that path against the wrong root.
CALIBY_CWD_IS_THE_CHECKOUT = True

OUTPUT_CSV_NAME = "seq_des_outputs.csv"
OUTPUT_SAMPLE_SUBDIR = "samples"
# The insertion order of the `outputs` defaultdict in `run_seq_des`. The set is
# what matters to this wrapper; the order is recorded so a drift is visible.
OUTPUT_COLUMNS = ("example_id", "out_pdb", "U", "input_seq", "seq")
REQUIRED_OUTPUT_COLUMNS = ("example_id", "out_pdb", "seq")
CHAIN_SEPARATOR = ":"
CONSTRAINT_COLUMNS = ("pdb_key", "fixed_pos_seq")
# `_VALID_POS_CONSTRAINT_COLUMNS` upstream. Writing a column outside this set
# raises there rather than being ignored.
VALID_CONSTRAINT_COLUMNS = (
    "pdb_key",
    "fixed_pos_seq",
    "fixed_pos_scn",
    "fixed_pos_override_seq",
    "pos_restrict_aatype",
    "symmetry_pos",
)

DEFAULT_MANIFEST_NAME = "sequence-candidate-manifest.jsonl"
DEFAULT_SEQUENCE_SUBDIR = "sequences"
DEFAULT_POSE_SUBDIR = "poses"
DEFAULT_WORK_SUBDIR = "caliby"
DEFAULT_MAX_CONFORMERS = 32
DEFAULT_SAMPLES_PER_BACKBONE = 32
STAGED_INPUT_SUBDIR = "staged-backbones"
CONSTRAINT_CSV_NAME = "pos-constraints.csv"
HYDRA_RUN_SUBDIR = "hydra"
ENSEMBLE_SUBDIR = "ensembles"

CALIBY_ROOT_ENVIRONMENT_KEY = "CALIBY_ROOT"
MODEL_PARAMS_ENVIRONMENT_KEY = "MODEL_PARAMS_DIR"
# `env_setup.sh` exports both as empty strings because AtomWorks requires them to
# be set and this route does not use either mirror.
ATOMWORKS_MIRROR_KEYS = ("PDB_MIRROR_PATH", "CCD_MIRROR_PATH")

SHA256_LENGTH = 64
SHA256_ALPHABET = frozenset("0123456789abcdef")


class CheckpointUnverified(AdapterError):
    """The Caliby checkpoint is absent or is not the pinned bytes."""


class UpstreamOutputMissing(AdapterError):
    """Caliby wrote no usable output for a phase."""


@dataclass(frozen=True)
class CalibyCheckout:
    """A clean Caliby checkout whose observed Git revision is the source pin."""

    root: Path
    source_revision: str


# ----------------------------------------------------------------------------
# The installed environment.
# ----------------------------------------------------------------------------


def git_checkout_output(root: Path, *arguments: str) -> str:
    """Read one Git value from an operator-supplied source checkout.

    A Caliby source tree is part of the evidence for every output row.  Script
    names alone do not identify its source: a tarball, a different checkout, or
    a locally edited copy can have the same paths.  This intentionally accepts
    only a clean Git checkout that can name its exact HEAD.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise AdapterError(
            f"Caliby source identity cannot be read from {root}: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "Git command failed"
        raise AdapterError(
            f"Caliby source identity cannot be read from {root}: {detail}"
        )
    return completed.stdout.strip()


def verify_caliby_checkout(root: Path) -> str:
    """Return the observed source HEAD, refusing non-Git, dirty, or wrong source.

    An archive has no commit identity, so this adapter does not guess that its
    contents match the pinned upstream revision.  A Git worktree has a useful
    identity only when it is rooted at the supplied path, clean, and exactly at
    the source revision whose behavior the command contract cites.
    """
    top_level = Path(git_checkout_output(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise AdapterError(
            f"Caliby source checkout root is {top_level}, not the supplied directory {root}"
        )
    observed = git_checkout_output(root, "rev-parse", "HEAD").lower()
    if observed != UPSTREAM_SOURCE_REVISION:
        raise AdapterError(
            f"Caliby checkout is at {observed}, not pinned source revision "
            f"{UPSTREAM_SOURCE_REVISION}"
        )
    changes = git_checkout_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if changes:
        raise AdapterError(
            f"Caliby checkout is dirty at pinned source revision {observed}; "
            "run a clean checkout so the candidate source revision is evidence"
        )
    return observed


def resolve_caliby_root(value: Path | None) -> CalibyCheckout:
    """Return the only Caliby checkout this adapter will run: clean and pinned."""
    candidate = value
    if candidate is None:
        environment_value = os.environ.get(CALIBY_ROOT_ENVIRONMENT_KEY, "").strip()
        if not environment_value:
            raise AdapterError(
                f"no Caliby checkout: pass --caliby-root, or set "
                f"{CALIBY_ROOT_ENVIRONMENT_KEY} to a checkout of {UPSTREAM_REPOSITORY}"
            )
        candidate = Path(environment_value)
    root = candidate.expanduser().resolve()
    if not root.is_dir():
        raise AdapterError(f"Caliby checkout is not a directory: {root}")
    missing = [name for name in REQUIRED_SCRIPTS if not (root / name).is_file()]
    if missing:
        raise AdapterError(
            f"{root} is missing {', '.join(missing)}, so it is not a Caliby checkout"
        )
    return CalibyCheckout(root=root, source_revision=verify_caliby_checkout(root))


def resolve_tool_python(value: str | None) -> str:
    """Return the interpreter of the Caliby environment.

    There is no useful default. This package's own interpreter has no Torch, and
    guessing one would produce an import error from a child process rather than a
    named cause here.
    """
    if value is None:
        raise AdapterError(
            "no interpreter for the Caliby environment: pass --tool-python with the "
            "python of the environment Caliby is installed into, such as "
            "envs/caliby/bin/python3"
        )
    resolved = shutil.which(value)
    if resolved is None:
        candidate = Path(value).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        raise AdapterError(f"interpreter not found: {value}")
    return resolved


def check_digest(value: str, label: str) -> str:
    """Refuse a digest that is not 64 lowercase hex characters."""
    text = value.strip().lower()
    if len(text) != SHA256_LENGTH or set(text) - SHA256_ALPHABET:
        raise AdapterError(f"{label} is not a SHA-256 hex digest: {value}")
    return text


def verify_checkpoint(path: Path, expected: str) -> str:
    """Return the checkpoint digest, refusing anything but the pinned bytes."""
    checkpoint = path.expanduser().resolve()
    if checkpoint.suffix != ".ckpt":
        raise CheckpointUnverified(
            f"--checkpoint is {checkpoint}, and upstream treats a value that does not end "
            "in .ckpt as a registry name it downloads from Hugging Face, which pins nothing"
        )
    if not checkpoint.is_file():
        raise CheckpointUnverified(f"Caliby checkpoint is missing: {checkpoint}")
    wanted = check_digest(expected, "--checkpoint-sha256")
    observed = sha256_file(checkpoint)
    if observed != wanted:
        raise CheckpointUnverified(
            f"Caliby checkpoint is not the pinned bytes: {checkpoint}; --checkpoint-sha256 "
            f"records {wanted} and the file reads {observed}"
        )
    return observed


def child_environment(model_params_dir: Path | None) -> dict[str, str]:
    """Return the environment a Caliby child process runs in.

    `env_setup.sh` sets the two AtomWorks mirror variables to empty strings
    because AtomWorks requires them to be set and this route uses neither.
    """
    environment = dict(os.environ)
    for key in ATOMWORKS_MIRROR_KEYS:
        environment.setdefault(key, "")
    if model_params_dir is not None:
        environment[MODEL_PARAMS_ENVIRONMENT_KEY] = str(model_params_dir)
    return environment


def run_tool(argv: list[str], *, cwd: Path, environment: dict[str, str], label: str) -> None:
    """Run one argument list with shell=False and fail on a nonzero exit."""
    print(f"solublecaliby adapter: run {' '.join(argv)} (cwd {cwd})", flush=True)
    completed = subprocess.run(argv, shell=False, check=False, cwd=str(cwd), env=environment)
    if completed.returncode != 0:
        raise AdapterError(f"{label} exited {completed.returncode}")


# ----------------------------------------------------------------------------
# Inputs.
# ----------------------------------------------------------------------------


def load_backbones(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return the backbone rows this phase designs sequences for."""
    if args.backbone_manifest is not None:
        manifest = args.backbone_manifest.expanduser().resolve()
        artifact_root = args.artifact_root.expanduser().resolve()
        if artifact_root not in manifest.parents:
            raise AdapterError(f"backbone manifest is outside the artifact root: {manifest}")
        if not manifest.is_file():
            raise AdapterError(f"backbone manifest not found: {manifest}")
        rows = load_jsonl(manifest)
    else:
        rows = completed_receipt_rows(
            args.receipts_dir, args.backbone_stage_id, args.backbone_artifact_id
        )
    rows.sort(key=lambda row: str(row.get("candidate_id", "")))
    if len(rows) < args.count:
        raise AdapterError(
            f"phase {args.phase} needs {args.count} backbones and the upstream manifest "
            f"has {len(rows)}"
        )
    selected = rows[: args.count]
    for row in selected:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            raise AdapterError("a backbone row carries no candidate_id")
        missing = [field for field in DIVERSITY_LINEAGE_FIELDS if not row.get(field)]
        if missing:
            raise AdapterError(
                f"backbone {candidate_id} is missing diversity lineage fields: "
                + ", ".join(missing)
            )
    return selected


def stage_backbones(
    rows: list[dict[str, Any]], staged_dir: Path
) -> list[tuple[str, Path, Path, str]]:
    """Copy each backbone pose into the directory Caliby globs, keyed by candidate ID.

    `get_pdb_files` reads every entry of `input_cfg.pdb_dir`, not only the ones
    ending in `.pdb`, and `example_id` is the file stem. So this directory holds
    exactly one file per candidate, named for that candidate, and nothing else.
    """
    if staged_dir.exists() and any(staged_dir.iterdir()):
        raise AdapterError(
            f"the staging directory already holds files: {staged_dir}. Upstream globs every "
            "entry of that directory, so a stale file would be designed as if it were a "
            "backbone. Run the stage in a clean attempt directory"
        )
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[str, Path, Path, str]] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        source_pose, source_sha256 = backbone_pose(row)
        destination = staged_dir / f"{candidate_id}.pdb"
        if destination.exists():
            raise AdapterError(f"two backbones share the candidate ID {candidate_id}")
        shutil.copyfile(source_pose, destination)
        staged.append((candidate_id, destination, source_pose, source_sha256))
    return staged


def load_manifest_chains(path: Path | None) -> dict[str, "numbering.ChainNumbering"]:
    """Return the author-to-position map the target stage already recorded."""
    if path is None:
        return {}
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise AdapterError(f"target manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chains = manifest.get("chains")
    if not isinstance(chains, list) or not chains:
        raise AdapterError(f"target manifest {manifest_path} records no chains")
    try:
        return numbering.from_manifest_chains(chains)
    except numbering.NumberingError as exc:
        raise AdapterError(f"target manifest {manifest_path}: {exc}") from exc


def chain_numbering_for(
    chain: str,
    from_pose: "numbering.ChainNumbering",
    manifest_chains: dict[str, "numbering.ChainNumbering"],
    structure: Path,
) -> "numbering.ChainNumbering":
    """Prefer the manifest's recorded map, after checking the pose still matches it."""
    recorded = manifest_chains.get(chain)
    if recorded is None:
        return from_pose
    if recorded.residues != from_pose.residues:
        raise AdapterError(
            f"chain {chain} of {structure} holds {from_pose.residue_count} residues numbered "
            f"{from_pose.author_span()}, and the target manifest records "
            f"{recorded.residue_count} numbered {recorded.author_span()}. The pose's target is "
            "not the prepared target, so the manifest's residue map does not describe it"
        )
    return recorded


def fixed_position_string(
    structure: Path,
    design_chain: str,
    *,
    manifest_chains: dict[str, "numbering.ChainNumbering"] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return the `fixed_pos_seq` that holds every chain but the design chain.

    Caliby's parser also accepts a bare chain ID, which masks every residue in
    that chain. Whole-chain constraints avoid manufacturing residue positions
    from author identifiers when the adapter does not need a subset.
    """
    try:
        chains = numbering.read_chain_numbering(structure)
    except numbering.NumberingError as exc:
        raise AdapterError(f"cannot read residue numbering from {structure}: {exc}") from exc
    recorded = manifest_chains or {}
    if design_chain not in chains:
        raise AdapterError(
            f"{structure} carries chains {sorted(chains)} and the design chain is "
            f"{design_chain}"
        )
    fixed_chains = sorted(chain for chain in chains if chain != design_chain)
    if not fixed_chains:
        raise AdapterError(
            f"{structure} carries only chain {design_chain}, so there is no target chain to "
            "hold fixed. A binder design pose carries the target and the binder"
        )
    parts: list[str] = []
    record: dict[str, Any] = {
        "design_chain": design_chain,
        "fixed_chains": fixed_chains,
        "constraint_syntax": "chain-only",
        "numbering_source": {},
        "author_ranges": {},
        "derived_with_gaps": False,
    }
    for chain in fixed_chains:
        if len(chain) != 1 or not chain.isascii() or not chain.isalpha():
            raise AdapterError(
                f"Caliby's fixed-position grammar accepts one alphabetic chain ID, "
                f"but {structure} carries fixed chain {chain!r}"
            )
        chain_numbering = chain_numbering_for(chain, chains[chain], recorded, structure)
        parts.append(chain)
        record["numbering_source"][chain] = chain_numbering.source
        record["author_ranges"][chain] = numbering.format_ranges(
            chain, [residue.number for residue in chain_numbering.residues]
        )
        if chain_numbering.source == numbering.DERIVED and not chain_numbering.contiguous:
            record["derived_with_gaps"] = True
    record["design_chain_residue_count"] = chains[design_chain].residue_count
    return ",".join(parts), record


def write_constraint_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    """Write the `pos_constraint_csv` upstream reads, with the two columns it needs."""
    for column in CONSTRAINT_COLUMNS:
        if column not in VALID_CONSTRAINT_COLUMNS:
            raise AdapterError(f"{column} is not a column upstream accepts")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CONSTRAINT_COLUMNS)
        for pdb_key, fixed_pos_seq in rows:
            writer.writerow([pdb_key, fixed_pos_seq])


# ----------------------------------------------------------------------------
# The command.
# ----------------------------------------------------------------------------


def hydra_value(value: object) -> str:
    """Render one Hydra override value.

    Hydra reads `key=value` from argv, so a value carrying a space or an equals
    sign would be read as another override. Every value this wrapper sends is a
    path, an integer or a boolean, and one that is not is refused rather than
    quoted into something that parses differently.
    """
    text = str(value)
    if not text or any(character in text for character in " \t\n="):
        raise AdapterError(f"Hydra override value is not a single token: {text!r}")
    return text


def hydra_argv(interpreter: str, script: str, overrides: dict[str, object]) -> list[str]:
    """Return the child argument list for one Hydra entry point."""
    argv = [interpreter, script]
    for key, value in overrides.items():
        argv.append(f"{key}={hydra_value(value)}")
    return argv


def design_overrides(
    *,
    checkpoint: Path,
    out_dir: Path,
    constraint_csv: Path,
    sequences_per_backbone: int,
    seed: int,
    hydra_run_dir: Path,
) -> dict[str, object]:
    """Return the overrides both design entry points share."""
    return {
        "ckpt_name_or_path": checkpoint,
        "out_dir": out_dir,
        "pos_constraint_csv": constraint_csv,
        "sampling_cfg_overrides.num_seqs_per_pdb": sequences_per_backbone,
        "seed": seed,
        "hydra.run.dir": hydra_run_dir,
    }


# ----------------------------------------------------------------------------
# Outputs.
# ----------------------------------------------------------------------------


def read_outputs_csv(path: Path) -> list[dict[str, str]]:
    """Return the rows of `seq_des_outputs.csv`, checking the columns it must carry."""
    if not path.is_file():
        raise UpstreamOutputMissing(f"Caliby wrote no {OUTPUT_CSV_NAME}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = [name for name in REQUIRED_OUTPUT_COLUMNS if name not in columns]
        if missing:
            raise UpstreamOutputMissing(
                f"{path} names columns {list(columns)} and this stage reads "
                f"{list(REQUIRED_OUTPUT_COLUMNS)}; missing {missing}"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise UpstreamOutputMissing(f"{path} holds no row")
    return rows


def chain_segment(sequence: str, chains: list[str], design_chain: str) -> str:
    """Return the design chain's segment of a multichain sequence.

    Upstream joins chains with ":" in alphabetical order of chain ID, so the
    caller passes the chain IDs of the structure it staged and this checks the
    two agree before it indexes.
    """
    ordered = sorted(chains)
    segments = sequence.split(CHAIN_SEPARATOR)
    if len(segments) != len(ordered):
        raise UpstreamOutputMissing(
            f"the designed sequence holds {len(segments)} chain segments and the staged "
            f"structure holds {len(ordered)} chains {ordered}"
        )
    if design_chain not in ordered:
        raise AdapterError(f"design chain {design_chain} is not one of {ordered}")
    return segments[ordered.index(design_chain)].strip()


def sample_index(row: dict[str, str]) -> int:
    """Return the sample index encoded in an output row's structure file name.

    `run_seq_des` names each file `{example_id}_sample{index}.cif` and writes the
    index nowhere else, so the file name is the only place it survives.
    """
    stem = Path(str(row.get("out_pdb", ""))).stem
    marker = "_sample"
    position = stem.rfind(marker)
    if position < 0:
        raise UpstreamOutputMissing(
            f"output structure {row.get('out_pdb')!r} does not name a sample index"
        )
    tail = stem[position + len(marker) :]
    if not tail.isdigit():
        raise UpstreamOutputMissing(
            f"output structure {row.get('out_pdb')!r} does not end in a sample index"
        )
    return int(tail)


def group_outputs(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group output rows by example, ordered by the sample index upstream encoded."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        example_id = str(row.get("example_id", "")).strip()
        if not example_id:
            raise UpstreamOutputMissing("an output row carries no example_id")
        grouped.setdefault(example_id, []).append(row)
    for example_id, example_rows in grouped.items():
        example_rows.sort(key=sample_index)
        indices = [sample_index(row) for row in example_rows]
        if len(indices) != len(set(indices)):
            raise UpstreamOutputMissing(
                f"{example_id} has duplicate sample indices in {OUTPUT_CSV_NAME}: {indices}"
            )
        expected = list(range(len(indices)))
        if indices != expected:
            raise UpstreamOutputMissing(
                f"{example_id} names sample indices {indices} in {OUTPUT_CSV_NAME}; "
                f"upstream names consecutive indices {expected}"
            )
    return grouped


def validate_output_samples(
    outputs: dict[str, list[dict[str, str]]],
    samples: Path,
    *,
    chains_by_example: dict[str, list[str]],
    design_chain: str,
) -> None:
    """Verify each CSV row against a readable native CIF and its designed chain.

    Both upstream design functions construct paths below the supplied out_dir as
    samples/example_id_sampleN.cif. The adapter retains those files and exposes
    their paths in its candidate rows. Existence is not evidence of an engine
    output, though: the shared reader must parse coordinates from the CIF and
    its design-chain sequence must equal the corresponding segment in Caliby's
    CSV row.
    """
    root = samples.resolve()
    for example_id, rows in outputs.items():
        chains = chains_by_example.get(example_id)
        if chains is None:
            raise UpstreamOutputMissing(
                f"{OUTPUT_CSV_NAME} contains unexpected example_id {example_id}; "
                "there is no staged backbone to identify its design chain"
            )
        for row in rows:
            index = sample_index(row)
            raw_path = str(row.get("out_pdb", "")).strip()
            if not raw_path:
                raise UpstreamOutputMissing(
                    f"{OUTPUT_CSV_NAME} row for {example_id} carries no output structure path"
                )
            output_path = Path(raw_path).expanduser()
            if not output_path.is_absolute():
                raise UpstreamOutputMissing(
                    f"{OUTPUT_CSV_NAME} names a relative output structure for {example_id}: "
                    f"{raw_path!r}; the adapter passed an absolute out_dir"
                )
            resolved = output_path.resolve()
            expected = root / f"{example_id}_sample{index}.cif"
            if resolved != expected:
                raise UpstreamOutputMissing(
                    f"{OUTPUT_CSV_NAME} names {resolved} for {example_id} sample {index}; "
                    f"upstream writes {expected}"
                )
            if not resolved.is_file():
                raise UpstreamOutputMissing(
                    f"Caliby listed an output structure that is absent: {resolved}"
                )
            expected_sequence = chain_segment(str(row.get("seq", "")), chains, design_chain)
            try:
                observed_sequence = evidence.chain_sequence(
                    resolved.read_bytes(), "cif", design_chain, str(resolved)
                )
            except (OSError, evidence.EvidenceError) as exc:
                raise UpstreamOutputMissing(
                    f"Caliby native output is not readable evidence for {example_id} sample "
                    f"{index}: {exc}"
                ) from exc
            if observed_sequence != expected_sequence:
                raise UpstreamOutputMissing(
                    f"Caliby native output sequence for {example_id} sample {index} is "
                    f"{observed_sequence!r} on chain {design_chain}, but {OUTPUT_CSV_NAME} "
                    f"records {expected_sequence!r}"
                )


def build_rows(
    args: argparse.Namespace,
    *,
    mode: str,
    staged: list[tuple[str, Path, Path, str]],
    outputs: dict[str, list[dict[str, str]]],
    constraints: dict[str, dict[str, Any]],
    parents: dict[str, dict[str, Any]],
    checkpoint_sha256: str,
    source_revision: str,
    phase_dir: Path,
    ensemble: dict[str, Any] | None,
    ensemble_by_candidate: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Write every candidate's FASTA and design pose, and return its manifest row."""
    rows: list[dict[str, Any]] = []
    for candidate_id, staged_path, source_pose, source_sha256 in staged:
        example_rows = outputs.get(candidate_id)
        if not example_rows:
            raise UpstreamOutputMissing(
                f"{OUTPUT_CSV_NAME} holds no row for {candidate_id}; upstream keys a row by "
                "the stem of the file it read"
            )
        if len(example_rows) < args.sequences_per_backbone:
            raise UpstreamOutputMissing(
                f"{candidate_id} has {len(example_rows)} designed sequences and the phase "
                f"asked for {args.sequences_per_backbone}"
            )
        chains = sorted(numbering.read_chain_numbering(staged_path))
        parent = parents[candidate_id]
        constraint = constraints[candidate_id]
        for variant, output_row in enumerate(example_rows[: args.sequences_per_backbone]):
            sequence = chain_segment(str(output_row.get("seq", "")), chains, args.design_chain)
            designed_id = f"{candidate_id}-{DESIGNER_ID}-{variant:02d}"
            check_sequence(
                sequence,
                candidate_id=designed_id,
                minimum=args.minimum_length,
                maximum=args.maximum_length,
            )
            sequence_path = phase_dir / args.sequence_subdir / f"{designed_id}.fasta"
            pose_path = phase_dir / args.pose_subdir / f"{designed_id}.pdb"
            write_sequence(sequence_path, designed_id, sequence)
            write_design_pose(
                pose_path,
                candidate_id=designed_id,
                source_pose=source_pose,
                source_sha256=source_sha256,
                chain=args.design_chain,
            )
            candidate: dict[str, Any] = {
                **{field: parent[field] for field in PARENT_LINEAGE_FIELDS if field in parent},
                "candidate_id": designed_id,
                "parent_candidate_id": candidate_id,
                "sequence_designer": DESIGNER_ID,
                "seq_method": DESIGNER_ID,
                "sequence_path": str(sequence_path.resolve()),
                "sequence_sha256": canonical_sequence_sha256(sequence),
                "sequence_length": len(sequence),
                "design_pose_path": str(pose_path.resolve()),
                "design_pose_sha256": sha256_file(pose_path),
                "design_chain": args.design_chain,
                "variant_index": variant,
                "requested_seed": args.seed,
                "adapter_id": ADAPTER_ID,
                "route_id": ROUTE_ID,
                "route_contract_revision": ROUTE_CONTRACT_REVISION,
                "design_mode": mode,
                "checkpoint_name": BOUND_CHECKPOINT_NAME,
                "checkpoint_sha256": checkpoint_sha256,
                "source_revision": source_revision,
                "weights_revision": WEIGHTS_REVISION,
                "cost_basis": COST_BASIS,
                "source_design_pose_sha256": source_sha256,
                "engine_output_structure": str(output_row.get("out_pdb", "")),
                "engine_output_structure_sha256": sha256_file(
                    Path(str(output_row.get("out_pdb", "")))
                ),
                "caliby_potts_energy": output_row.get("U"),
                "fixed_position_numbering": "chain-only",
                "fixed_pos_seq": constraint["fixed_pos_seq"],
                "fixed_position_syntax": constraint["record"]["constraint_syntax"],
                "fixed_chain_author_ranges": constraint["record"]["author_ranges"],
                "fixed_chain_numbering_source": constraint["record"]["numbering_source"],
                "numbering_derived_with_gaps": constraint["record"]["derived_with_gaps"],
                "status": "sequence-designed",
            }
            if ensemble is not None:
                candidate.update(ensemble)
                if ensemble_by_candidate is None or candidate_id not in ensemble_by_candidate:
                    raise AdapterError(
                        f"no recorded conformer layout for ensemble candidate {candidate_id}"
                    )
                candidate.update(ensemble_by_candidate[candidate_id])
            rows.append(candidate)
    return rows


# ----------------------------------------------------------------------------
# Subcommands.
# ----------------------------------------------------------------------------


def phase_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Return the attempt directory, the phase directory, the work directory and the manifest."""
    attempt_dir = args.attempt_dir.expanduser().resolve()
    phase_dir = attempt_dir / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    work_dir = phase_dir / args.work_subdir
    manifest_path = (
        args.manifest_path.expanduser().resolve()
        if getattr(args, "manifest_path", None) is not None
        else phase_dir / DEFAULT_MANIFEST_NAME
    )
    return attempt_dir, phase_dir, work_dir, manifest_path


def require_positive(value: int, label: str) -> None:
    """Reject a count the upstream scripts would turn into an empty or odd run."""
    if value < 1:
        raise AdapterError(f"{label} must be at least 1")


def require_fresh_output_directory(path: Path, label: str) -> None:
    """Prevent an upstream script's mkdir(exist_ok=True) from merging stale files."""
    if not path.exists():
        return
    if not path.is_dir():
        raise AdapterError(f"{label} output path is not a directory: {path}")
    if any(path.iterdir()):
        raise AdapterError(
            f"{label} output directory already holds files: {path}. "
            "Run the stage in a clean attempt directory"
        )


def prepare_inputs(
    args: argparse.Namespace, work_dir: Path
) -> tuple[
    list[tuple[str, Path, Path, str]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    Path,
]:
    """Stage the backbones, write the constraint CSV, and return what run needs."""
    backbones = load_backbones(args)
    parents = {str(row["candidate_id"]): row for row in backbones}
    staged = stage_backbones(backbones, work_dir / STAGED_INPUT_SUBDIR)
    manifest_chains = load_manifest_chains(args.target_manifest)
    constraints: dict[str, dict[str, Any]] = {}
    constraint_rows: list[tuple[str, str]] = []
    for candidate_id, staged_path, _, _ in staged:
        fixed_pos_seq, record = fixed_position_string(
            staged_path,
            args.design_chain,
            manifest_chains=manifest_chains,
        )
        constraints[candidate_id] = {"fixed_pos_seq": fixed_pos_seq, "record": record}
        constraint_rows.append((candidate_id, fixed_pos_seq))
    constraint_csv = work_dir / CONSTRAINT_CSV_NAME
    write_constraint_csv(constraint_csv, constraint_rows)
    return staged, constraints, parents, constraint_csv


def finish(
    args: argparse.Namespace,
    *,
    mode: str,
    staged: list[tuple[str, Path, Path, str]],
    constraints: dict[str, dict[str, Any]],
    parents: dict[str, dict[str, Any]],
    checkpoint_sha256: str,
    source_revision: str,
    out_dir: Path,
    phase_dir: Path,
    manifest_path: Path,
    ensemble: dict[str, Any] | None,
    ensemble_by_candidate: dict[str, dict[str, Any]] | None,
) -> int:
    """Parse one completed Caliby run and write the stage manifest."""
    outputs = group_outputs(read_outputs_csv(out_dir / OUTPUT_CSV_NAME))
    samples = out_dir / OUTPUT_SAMPLE_SUBDIR
    if not samples.is_dir():
        raise UpstreamOutputMissing(f"Caliby wrote no {OUTPUT_SAMPLE_SUBDIR} directory: {samples}")
    validate_output_samples(
        outputs,
        samples,
        chains_by_example={
            candidate_id: sorted(numbering.read_chain_numbering(staged_path))
            for candidate_id, staged_path, _, _ in staged
        },
        design_chain=args.design_chain,
    )
    rows = build_rows(
        args,
        mode=mode,
        staged=staged,
        outputs=outputs,
        constraints=constraints,
        parents=parents,
        checkpoint_sha256=checkpoint_sha256,
        source_revision=source_revision,
        phase_dir=phase_dir,
        ensemble=ensemble,
        ensemble_by_candidate=ensemble_by_candidate,
    )
    write_jsonl(manifest_path, rows)
    print(
        f"solublecaliby adapter: wrote {len(rows)} candidates from {len(staged)} backbones "
        f"to {manifest_path}",
        flush=True,
    )
    return 0


def run(args: argparse.Namespace) -> int:
    """Design sequences for one phase against fixed backbones."""
    require_positive(args.sequences_per_backbone, "--sequences-per-backbone")
    root = resolve_caliby_root(args.caliby_root)
    interpreter = resolve_tool_python(args.tool_python)
    checkpoint_sha256 = verify_checkpoint(args.checkpoint, args.checkpoint_sha256)
    _, phase_dir, work_dir, manifest_path = phase_paths(args)
    staged, constraints, parents, constraint_csv = prepare_inputs(args, work_dir)
    out_dir = work_dir / "seq-des"
    require_fresh_output_directory(out_dir, SEQ_DES_SCRIPT)
    argv = hydra_argv(
        interpreter,
        SEQ_DES_SCRIPT,
        {
            **design_overrides(
                checkpoint=args.checkpoint.expanduser().resolve(),
                out_dir=out_dir,
                constraint_csv=constraint_csv,
                sequences_per_backbone=args.sequences_per_backbone,
                seed=args.seed,
                hydra_run_dir=work_dir / HYDRA_RUN_SUBDIR,
            ),
            "input_cfg.pdb_dir": work_dir / STAGED_INPUT_SUBDIR,
        },
    )
    run_tool(
        argv,
        cwd=root.root,
        environment=child_environment(args.model_params_dir),
        label=SEQ_DES_SCRIPT,
    )
    return finish(
        args,
        mode="fixed-backbone",
        staged=staged,
        constraints=constraints,
        parents=parents,
        checkpoint_sha256=checkpoint_sha256,
        source_revision=root.source_revision,
        out_dir=out_dir,
        phase_dir=phase_dir,
        manifest_path=manifest_path,
        ensemble=None,
        ensemble_by_candidate=None,
    )


def run_ensemble(args: argparse.Namespace) -> int:
    """Design sequences for one phase against a Protpardelle-1c conformer ensemble."""
    require_positive(args.sequences_per_backbone, "--sequences-per-backbone")
    require_positive(args.max_conformers, "--max-conformers")
    root = resolve_caliby_root(args.caliby_root)
    interpreter = resolve_tool_python(args.tool_python)
    checkpoint_sha256 = verify_checkpoint(args.checkpoint, args.checkpoint_sha256)
    conformer_dir = args.conformer_dir.expanduser().resolve()
    if not conformer_dir.is_dir():
        raise AdapterError(f"conformer directory not found: {conformer_dir}")
    _, phase_dir, work_dir, manifest_path = phase_paths(args)
    staged, constraints, parents, constraint_csv = prepare_inputs(args, work_dir)
    ensemble_by_candidate = check_ensemble_layout(
        conformer_dir,
        [candidate for candidate, _, _, _ in staged],
        max_conformers=args.max_conformers,
    )
    out_dir = work_dir / "seq-des-ensemble"
    require_fresh_output_directory(out_dir, SEQ_DES_ENSEMBLE_SCRIPT)
    argv = hydra_argv(
        interpreter,
        SEQ_DES_ENSEMBLE_SCRIPT,
        {
            **design_overrides(
                checkpoint=args.checkpoint.expanduser().resolve(),
                out_dir=out_dir,
                constraint_csv=constraint_csv,
                sequences_per_backbone=args.sequences_per_backbone,
                seed=args.seed,
                hydra_run_dir=work_dir / HYDRA_RUN_SUBDIR,
            ),
            "input_cfg.conformer_dir": conformer_dir,
            "max_num_conformers": args.max_conformers,
            "include_primary_conformer": "true",
        },
    )
    run_tool(
        argv,
        cwd=root.root,
        environment=child_environment(args.model_params_dir),
        label=SEQ_DES_ENSEMBLE_SCRIPT,
    )
    return finish(
        args,
        mode="protpardelle-1c-ensemble",
        staged=staged,
        constraints=constraints,
        parents=parents,
        checkpoint_sha256=checkpoint_sha256,
        source_revision=root.source_revision,
        out_dir=out_dir,
        phase_dir=phase_dir,
        manifest_path=manifest_path,
        ensemble={
            "ensemble_conformer_dir": str(conformer_dir),
            "ensemble_max_conformers": args.max_conformers,
            "ensemble_includes_primary_conformer": True,
            "protpardelle_revision": PROTPARDELLE_REVISION,
        },
        ensemble_by_candidate=ensemble_by_candidate,
    )


def check_ensemble_layout(
    conformer_dir: Path, candidate_ids: list[str], *, max_conformers: int
) -> dict[str, dict[str, Any]]:
    """Return the actual primary-plus-conformer count upstream will use.

    process_conformer_dirs raises FileNotFoundError when a subdirectory holds no
    file named for the subdirectory. It does permit a primary-only directory, and
    it caps rather than requires remaining conformers, so the recorded number
    must be calculated per candidate rather than assumed from the cap.
    """
    if max_conformers < 1:
        raise AdapterError("--max-conformers must be at least 1 when the primary is included")
    layouts: dict[str, dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        subdirectory = conformer_dir / candidate_id
        if not subdirectory.is_dir():
            raise AdapterError(
                f"the conformer directory has no ensemble for {candidate_id}: {subdirectory}"
            )
        primary_cif = subdirectory / f"{candidate_id}.cif"
        primary_pdb = subdirectory / f"{candidate_id}.pdb"
        if primary_cif.is_file():
            primary = primary_cif
        elif primary_pdb.is_file():
            primary = primary_pdb
        else:
            raise AdapterError(
                f"ensemble {subdirectory} holds no primary conformer named {candidate_id}.pdb "
                f"or {candidate_id}.cif, and upstream raises rather than choosing one"
            )
        conformers = [
            *subdirectory.glob("*.pdb"),
            *subdirectory.glob("*.cif"),
        ]
        # The upstream selection removes the chosen primary first, then takes
        # max_num_conformers - 1 of what remains. A directory holding both
        # candidate-id extensions therefore uses the CIF primary and treats the
        # PDB as an additional conformer, exactly as its source does.
        conformers.remove(primary)
        generated_available = len(conformers)
        generated_used = min(generated_available, max_conformers - 1)
        layouts[candidate_id] = {
            "ensemble_primary_conformer": str(primary.resolve()),
            "ensemble_conformers_available": generated_available + 1,
            "ensemble_generated_conformers_available": generated_available,
            "ensemble_conformers_used": generated_used + 1,
            "ensemble_generated_conformers_used": generated_used,
        }
    return layouts


def generate_ensembles(args: argparse.Namespace) -> int:
    """Generate one Protpardelle-1c ensemble per backbone.

    This stage carries weight licences the fixed-backbone stage does not.
    `generate_ensembles.py` imports Protpardelle-1c and calls `ensure_dir` for
    both the Protpardelle-1c and the ProteinMPNN weight directories, so a run
    stages three sets of weights rather than one.
    """
    require_positive(args.samples_per_backbone, "--samples-per-backbone")
    root = resolve_caliby_root(args.caliby_root)
    interpreter = resolve_tool_python(args.tool_python)
    _, _, work_dir, _ = phase_paths(args)
    backbones = load_backbones(args)
    staged = stage_backbones(backbones, work_dir / STAGED_INPUT_SUBDIR)
    out_dir = work_dir / ENSEMBLE_SUBDIR
    require_fresh_output_directory(out_dir, GENERATE_ENSEMBLES_SCRIPT)
    argv = hydra_argv(
        interpreter,
        GENERATE_ENSEMBLES_SCRIPT,
        {
            "model_params_path": args.model_params_dir or Path("model_params"),
            "input_cfg.pdb_dir": work_dir / STAGED_INPUT_SUBDIR,
            "num_samples_per_pdb": args.samples_per_backbone,
            "out_dir": out_dir,
            "seed": args.seed,
            "hydra.run.dir": work_dir / HYDRA_RUN_SUBDIR,
        },
    )
    run_tool(
        argv,
        cwd=root.root,
        environment=child_environment(args.model_params_dir),
        label=GENERATE_ENSEMBLES_SCRIPT,
    )
    produced = sorted(path for path in out_dir.glob("*") if path.is_dir())
    if not produced:
        raise UpstreamOutputMissing(
            f"{GENERATE_ENSEMBLES_SCRIPT} wrote no ensemble directory under {out_dir}"
        )
    print(
        f"solublecaliby adapter: {len(staged)} backbones produced {len(produced)} "
        f"Protpardelle-1c output directories under {out_dir}",
        flush=True,
    )
    print(
        "solublecaliby adapter: pass one of those directories to run-ensemble as "
        "--conformer-dir",
        flush=True,
    )
    return 0


def parse_outputs(args: argparse.Namespace) -> int:
    """Check the phase outputs this stage declares, without running anything."""
    _, phase_dir, _, manifest_path = phase_paths(args)
    if not manifest_path.is_file():
        raise AdapterError(f"stage manifest not found: {manifest_path}")
    rows = load_jsonl(manifest_path)
    if not rows:
        raise AdapterError(f"stage manifest holds no row: {manifest_path}")
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        for field in ("sequence_path", "design_pose_path"):
            value = row.get(field)
            if not value or not Path(str(value)).is_file():
                raise AdapterError(f"{candidate_id} declares a missing {field}: {value}")
        sequence_path = Path(str(row["sequence_path"]))
        records = [
            line for line in sequence_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if len(records) != 2 or not records[0].startswith(">"):
            raise AdapterError(f"{candidate_id} FASTA is not one header and one sequence")
        if canonical_sequence_sha256(records[1].strip()) != row.get("sequence_sha256"):
            raise AdapterError(f"{candidate_id} sequence does not match its recorded digest")
        engine_output = row.get("engine_output_structure")
        engine_path = Path(str(engine_output)) if engine_output else None
        if engine_path is None or not engine_path.is_file():
            raise AdapterError(
                f"{candidate_id} declares a missing engine_output_structure: {engine_output}"
            )
        if engine_path.suffix.lower() not in {".cif", ".mmcif"}:
            raise AdapterError(
                f"{candidate_id} engine_output_structure is not Caliby's native CIF: {engine_path}"
            )
        expected_engine_digest = row.get("engine_output_structure_sha256")
        if not expected_engine_digest or sha256_file(engine_path) != expected_engine_digest:
            raise AdapterError(
                f"{candidate_id} engine_output_structure does not match its recorded digest"
            )
        if row.get("source_revision") != UPSTREAM_SOURCE_REVISION:
            raise AdapterError(
                f"{candidate_id} does not record pinned Caliby source revision "
                f"{UPSTREAM_SOURCE_REVISION}"
            )
        try:
            native_sequence = evidence.chain_sequence(
                engine_path.read_bytes(), "cif", str(row.get("design_chain", "")), str(engine_path)
            )
        except (OSError, evidence.EvidenceError) as exc:
            raise AdapterError(
                f"{candidate_id} engine_output_structure is not readable native evidence: {exc}"
            ) from exc
        if native_sequence != records[1].strip():
            raise AdapterError(
                f"{candidate_id} native design-chain sequence {native_sequence!r} does not match "
                "the FASTA sequence"
            )
    print(
        json.dumps(
            {
                "adapter_id": ADAPTER_ID,
                "phase": args.phase,
                "manifest": str(manifest_path),
                "candidates": len(rows),
                "phase_dir": str(phase_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def toolcheck(args: argparse.Namespace) -> int:
    """Report this adapter's readiness. Executes nothing and costs nothing."""
    report: dict[str, Any] = {
        "adapter_id": ADAPTER_ID,
        "designer_id": DESIGNER_ID,
        "route_id": ROUTE_ID,
        "route_contract_revision": ROUTE_CONTRACT_REVISION,
        "source_repository": UPSTREAM_REPOSITORY,
        "source_revision": UPSTREAM_SOURCE_REVISION,
        "weights_host": WEIGHTS_HOST,
        "weights_revision": WEIGHTS_REVISION,
        "checkpoint_name": BOUND_CHECKPOINT_NAME,
        "checkpoint_file": BOUND_CHECKPOINT_FILE,
        "cost_basis": COST_BASIS,
        "modes": ["fixed-backbone", "protpardelle-1c-ensemble"],
        "qualified_run": False,
    }
    try:
        root = resolve_caliby_root(args.caliby_root)
    except AdapterError as exc:
        report["caliby_root"] = None
        report["caliby_root_problem"] = str(exc)
    else:
        report["caliby_root"] = str(root.root)
        report["checkout_source_revision"] = root.source_revision
        report["checkout_clean"] = True
    if args.checkpoint is not None and args.checkpoint_sha256 is not None:
        try:
            report["checkpoint_sha256"] = verify_checkpoint(args.checkpoint, args.checkpoint_sha256)
        except AdapterError as exc:
            report["checkpoint_problem"] = str(exc)
    print(json.dumps(report, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------
# The argv contract.
# ----------------------------------------------------------------------------


def add_install_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--caliby-root",
        type=Path,
        default=None,
        help=(
            f"Caliby checkout to run. Defaults to {CALIBY_ROOT_ENVIRONMENT_KEY}. "
            f"Install from {UPSTREAM_REPOSITORY}."
        ),
    )
    parser.add_argument(
        "--tool-python",
        default=None,
        help=(
            "Interpreter of the environment Caliby is installed into, such as "
            "envs/caliby/bin/python3. This package's interpreter has no Torch."
        ),
    )
    parser.add_argument(
        "--model-params-dir",
        type=Path,
        default=None,
        help=(
            f"Weights directory, exported to the child as {MODEL_PARAMS_ENVIRONMENT_KEY}. "
            f"Weights live at {WEIGHTS_HOST}."
        ),
    )


def add_checkpoint_arguments(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=required,
        default=None,
        help=(
            f"Path to {BOUND_CHECKPOINT_FILE}. A registry name is refused: upstream "
            "downloads a name from Hugging Face at job time, which pins nothing."
        ),
    )
    parser.add_argument(
        "--checkpoint-sha256",
        required=required,
        default=None,
        help="SHA-256 of the checkpoint file. The stage refuses any other bytes.",
    )


def add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", default=None, help="Stage ID this invocation runs under.")
    parser.add_argument("--phase", required=True, help="Stage phase name, such as smoke or scale.")
    parser.add_argument(
        "--attempt-dir", type=Path, required=True, help="Attempt directory that owns the outputs."
    )
    parser.add_argument(
        "--receipts-dir",
        type=Path,
        required=True,
        help="Directory holding the completed stage receipts.",
    )
    parser.add_argument("--artifact-root", type=Path, required=True, help="Run artifact root.")
    parser.add_argument("--config", type=Path, required=True, help="Resolved run config.")
    parser.add_argument("--plan", type=Path, default=None, help="Resolved run plan.")
    parser.add_argument(
        "--work-subdir",
        default=DEFAULT_WORK_SUBDIR,
        help=f"Work directory inside the phase directory. Defaults to {DEFAULT_WORK_SUBDIR}.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=f"Manifest path. Defaults to {DEFAULT_MANIFEST_NAME} in the phase directory.",
    )


def add_backbone_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--count", type=int, required=True, help="Number of backbones this phase reads."
    )
    parser.add_argument(
        "--backbone-stage-id",
        default="generate-arm-1",
        help="Stage ID of the upstream backbone generator.",
    )
    parser.add_argument(
        "--backbone-artifact-id",
        default="arm-1-candidates",
        help="Artifact ID of the upstream backbone manifest.",
    )
    parser.add_argument(
        "--backbone-manifest",
        type=Path,
        default=None,
        help="Published backbone manifest under the artifact root. Overrides the receipt lookup.",
    )


def add_design_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--design-chain",
        required=True,
        help=(
            "Chain to redesign. Every other chain is held fixed through the "
            "pos_constraint_csv, because upstream redesigns every position without one."
        ),
    )
    parser.add_argument(
        "--sequences-per-backbone",
        type=int,
        default=1,
        help="Sequences to design for every backbone. Match the stage records_per_count.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Seed passed to the Caliby entry point."
    )
    parser.add_argument(
        "--target-manifest",
        type=Path,
        default=None,
        help=(
            "Target manifest whose chains[i].residue_ids the adapter compares with fixed "
            "chains before recording the constraint audit fields."
        ),
    )
    parser.add_argument(
        "--minimum-length", type=int, default=None, help="Reject a sequence below this length."
    )
    parser.add_argument(
        "--maximum-length", type=int, default=None, help="Reject a sequence above this length."
    )
    parser.add_argument(
        "--sequence-subdir",
        default=DEFAULT_SEQUENCE_SUBDIR,
        help=f"FASTA directory inside the phase directory. Defaults to {DEFAULT_SEQUENCE_SUBDIR}.",
    )
    parser.add_argument(
        "--pose-subdir",
        default=DEFAULT_POSE_SUBDIR,
        help=f"Design pose directory inside the phase directory. Defaults to {DEFAULT_POSE_SUBDIR}.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Design SolubleCaliby sequences on a local Caliby install.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "toolcheck", help="Report this adapter's readiness without executing anything."
    )
    add_install_arguments(check_parser)
    add_checkpoint_arguments(check_parser, required=False)
    check_parser.add_argument(
        "--config", type=Path, default=None, help="Resolved run config, when one exists."
    )

    run_parser = subparsers.add_parser("run", help="Design sequences against fixed backbones.")
    add_install_arguments(run_parser)
    add_checkpoint_arguments(run_parser, required=True)
    add_stage_arguments(run_parser)
    add_backbone_arguments(run_parser)
    add_design_arguments(run_parser)

    ensemble_parser = subparsers.add_parser(
        "run-ensemble", help="Design sequences against a Protpardelle-1c conformer ensemble."
    )
    add_install_arguments(ensemble_parser)
    add_checkpoint_arguments(ensemble_parser, required=True)
    add_stage_arguments(ensemble_parser)
    add_backbone_arguments(ensemble_parser)
    add_design_arguments(ensemble_parser)
    ensemble_parser.add_argument(
        "--conformer-dir",
        type=Path,
        required=True,
        help="Ensemble directory written by the generate-ensembles stage.",
    )
    ensemble_parser.add_argument(
        "--max-conformers",
        type=int,
        default=DEFAULT_MAX_CONFORMERS,
        help=(
            f"Conformers per ensemble, including the primary structure. Defaults to "
            f"{DEFAULT_MAX_CONFORMERS}, which uses the primary structure plus "
            f"{DEFAULT_MAX_CONFORMERS - 1} generated conformers."
        ),
    )

    generate_parser = subparsers.add_parser(
        "generate-ensembles", help="Generate Protpardelle-1c ensembles for this phase's backbones."
    )
    add_install_arguments(generate_parser)
    add_stage_arguments(generate_parser)
    add_backbone_arguments(generate_parser)
    generate_parser.add_argument(
        "--samples-per-backbone",
        type=int,
        default=DEFAULT_SAMPLES_PER_BACKBONE,
        help=(
            f"Conformers to generate per backbone. Defaults to {DEFAULT_SAMPLES_PER_BACKBONE}."
        ),
    )
    generate_parser.add_argument(
        "--seed", type=int, default=1, help="Seed passed to the Caliby entry point."
    )

    parse_parser = subparsers.add_parser(
        "parse", help="Check the outputs of one completed phase."
    )
    add_stage_arguments(parse_parser)
    return parser


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


COMMANDS = {
    "toolcheck": toolcheck,
    "run": run,
    "run-ensemble": run_ensemble,
    "generate-ensembles": generate_ensembles,
    "parse": parse_outputs,
}


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        return COMMANDS[args.command](args)
    except AdapterError as exc:
        print(f"solublecaliby adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
