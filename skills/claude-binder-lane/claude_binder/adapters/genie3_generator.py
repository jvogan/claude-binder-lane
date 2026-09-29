#!/usr/bin/env python3
"""Generate Genie3 binder backbones for one binder lane stage phase.

This wrapper fills the `genie3-generator` slot. It reads the target manifest the
`target-preparer` stage published, composes the binderbench dataset and the YAML
configuration Genie3 reads, runs `genie3 generate` once for the whole phase, and
writes receipt-owned outputs into the current attempt directory:

  <attempt>/<phase>/poses/<candidate_id>.pdb        one design pose per candidate
  <attempt>/<phase>/candidate-manifest.jsonl        one row per candidate
  <attempt>/<phase>/genie3/                         the dataset, the configuration,
                                                    and the untouched tool output

Genie3 binder mode returns a backbone and no sequence, so the manifest sets
`sequence_path`, `sequence_sha256`, `sequence_length`, and `sequence_designer` to
null, records `backbone_only: true` and `generator_mode: backbone-only`, and
leaves the sequence to a downstream sequence-designer slot.

Four properties of the tool this wrapper handles rather than hides.

**`generate`, never `run`.** `genie3 run` chains generation into evaluation, and
evaluation needs ColabFold. The minimal install this package documents does not
carry it, so `run` crashes after the generation it already paid for. The wrapper
builds `generate` and offers no way to build `run`.

**The working directory is load-bearing.** Genie3 resolves
`pretrained/<version>/config.yaml` relative to the process working directory, so
the same command succeeds in one directory and exits in under a second with a
FileNotFoundError in another. The wrapper takes that directory as --genie3-home,
refuses to start unless the model configuration is there, runs the subprocess
with cwd set to it, and records the directory and the model configuration hash on
every row.

**Every path inside the problem JSON is absolute.** Genie3 opens
`target_pdb_filepath` with `open()`, which resolves a relative path from the
working directory rather than from the dataset root. The dataset lives under the
attempt directory and the working directory is the weights directory, so a
relative path there names a file that does not exist.

**The output is a C-alpha trace.** Genie3 binder mode writes `CA` atoms and no
`N`, `C`, or `O`. The wrapper copies the coordinate lines it was given and
synthesizes no atom, and it records the atom names it observed so a downstream
sequence designer can be held against them. Routing a C-alpha trace into the
vanilla ProteinMPNN checkpoint fails without an error, which is the pairing
`backbone_shape.py` guards.

**Genie3 renames its chains.** One recorded run read a 115-residue target in on
chain A and wrote it back out on chain B, with the 70-residue generated binder on
A. So no chain letter identifies a chain of the output, neither the campaign's
declared target letter nor the letter this wrapper wrote into the problem.

Chain identity is resolved by residue name first and residue count second.
Genie3 is never told the sequence of what it designs, so it writes UNK for the
whole designed chain and copies the target's own residue names through. A chain
that is entirely UNK is a candidate binder and a chain carrying real residue
names is the target. Among the candidates, the binder is the one whose residue
count falls inside the campaign's binder length bounds. That second filter is
what stops the other failure: a measured run wrote 95 residues on chain A and
115, the target length, on chain B, and the wrapper of the day published chain B,
so the sequence designer redesigned the target, and nothing downstream reports
that as an error. None inside the bounds refuses, and it says the binder came out
the wrong length when a chain the names ruled out is the only one that fits. More
than one inside them refuses, because the counts cannot then say which chain is
the binder, and the operator resolves it with --generator-binder-chain, which is
held against both filters. Once resolved, the binder letter and --binder-chain
are swapped so the sequence designer still designs the chain the profile names.

An output whose chains are all UNK, or none, leaves the residue names silent. The
wrapper then falls back to excluding the letter the campaign declares as the
target and lets the bounds decide among the rest. That fallback inherits the
letter's weakness against renaming and is the best signal left.

The design pose keeps every chain the tool returned, matching
`rfdiffusion_generator.py`. Three consumers in this package read the target chain
out of a design pose: the sequence designer copies it forward as fixed context,
the interface scorer hands the pose to DockQ naming both chains, and the cofold
adapters compare its target residue keys against the prepared target. A pose
carrying no target chain is refused for that reason.

**Two routes reach this module.** A local run starts the console script on this
machine. A Modal dispatch runs this same file inside the shipped
`genie3_generator_gpu` environment, which mounts the weights Volume at /weights
beside a checkout at /opt/genie3. --runner-protocol names the route, auto reads
those two mounts, and every row records the value it resolved, so a receipt says
where its structures were generated. The fal route is a separate module,
`fal_genie3_generator.py`.

Genie3 has no seed argument in any invocation this package records, so the
wrapper refuses to guess one. Pass --seed-config-key to write the seed into the
generated configuration at a key you name, or --allow-unseeded to run without
one. Every row records the requested seed, the seed the tool received, and
whether the wrapper delivered it.

Every command is an argument list that runs with shell=False. The wrapper builds
no shell string.

Install Genie3 from https://github.com/aqlaboratory/genie3 and point the wrapper
at the directory that holds `pretrained/` with --genie3-home or GENIE3_HOME.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from claude_binder.adapters.candidate_lineage import backbone_lineage
from claude_binder.backbone_shape import order_atoms

DEFAULT_TOOL_COMMAND = "genie3"
DEFAULT_MODEL_VERSION = "v1"
MODEL_CONFIG_NAME = "config.yaml"
PRETRAINED_SUBDIR = "pretrained"
DEFAULT_GENERATOR_ID = "genie3"
DEFAULT_ADAPTER_ID = "genie3-generator"
DEFAULT_PROBLEM_ID = "binder-lane"
DEFAULT_MANIFEST_NAME = "candidate-manifest.jsonl"
DEFAULT_POSE_SUBDIR = "poses"
DEFAULT_WORK_SUBDIR = "genie3"
DEFAULT_TARGET_STAGE_ID = "target-prepare"
DEFAULT_TARGET_ARTIFACT_ID = "target-manifest"
# The routes that reach this module. A Modal dispatch runs this same file inside
# the shipped genie3_generator_gpu container rather than a second module, so the
# route is a value the module has to carry rather than a constant it can assume.
# "fal" is absent because the fal route is its own module,
# adapters/fal_genie3_generator.py, which declares RUNNER_PROTOCOL = "fal" and
# records it on the rows it writes.
RUNNER_PROTOCOLS = ("auto", "local", "modal")
DEFAULT_RUNNER_PROTOCOL = "auto"
# What the shipped Modal environment mounts, from
# skills/claude-binder-lane/envs/genie3_generator_gpu.py. The image carries the
# checkout at /opt/genie3, the weights Volume mounts at /weights, and GENIE3_HOME
# names that mount. Both paths are what `auto` reads to tell the container apart
# from a workstation.
MODAL_GENIE3_SOURCE = Path("/opt/genie3")
MODAL_GENIE3_HOME = Path("/weights")
# The binder chain the shipped campaign template declares, matching
# `data/templates/campaign.template.json` and `rfdiffusion3_generator.py`. A
# profile passes the campaign's own value, so this default is the fallback.
DEFAULT_BINDER_CHAIN = "A"
DEFAULT_EVALUATION_VERSION = "binder"
# The binder length range the recorded PD-L1 wave sampled. A campaign overrides
# both with its own bounds, which is what the profile tokens carry.
DEFAULT_BINDER_MINIMUM_LENGTH = 60
DEFAULT_BINDER_MAXIMUM_LENGTH = 90
DATASET_SUBDIR = "binderbench"
RUN_SUBDIR = "runs"
# Genie3 writes one directory per problem under paths.rootdir and puts the
# generated structures in a `pdbs` directory inside it:
# <rootdir>/<selection>/pdbs/<selection>_<sample_idx>.pdb. A parser that expects
# a flat directory finds nothing. A run that lands elsewhere in the same subtree
# still counts, so the search falls back to the whole problem directory.
PREFERRED_OUTPUT_DIRS = ("pdbs", "successful_complexes")
GENERATOR_MODE = "backbone-only"
CANDIDATE_STATUS = "generated"
ATOM_RECORD_PREFIXES = ("ATOM  ", "HETATM")
READ_BLOCK_BYTES = 1024 * 1024
# The fields the target manifest has to carry before this stage can run.
REQUIRED_TARGET_MANIFEST_FIELDS = (
    "target_id",
    "target_sha256",
    "residue_map_sha256",
    "source_structure_path",
    "normalized_structure_path",
)
# Residue IDs arrive from the target manifest as CHAIN:NUMBER with an optional
# insertion code. A Genie3 site token is CHAIN followed by NUMBER and carries no
# place for an insertion code.
RESIDUE_ID_RE = re.compile(r"^([^:]+):(-?\d+)([A-Za-z]?)$")
RESIDUE_NUMBER_RE = re.compile(r"^(-?\d+)([A-Za-z]?)$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CHAIN_ID_RE = re.compile(r"^[A-Za-z0-9]$")
CONFIG_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
# Residue names a backbone generator writes when it has assigned no identity. A
# designed chain outside this set carries names the tool chose, and the manifest
# says so rather than claiming a placeholder.
PLACEHOLDER_RESIDUE_NAMES = frozenset({"UNK", "GLY", "ALA"})
# The residue name Genie3 writes for a residue it designed. It is never told the
# sequence of what it generates, so it writes this name for the whole designed
# chain and copies the target's own residue names through onto the other chain.
# The discriminator is this one name rather than PLACEHOLDER_RESIDUE_NAMES,
# because a target that arrives as poly-glycine carries GLY throughout and is
# still the target.
DESIGNED_RESIDUE_NAME = "UNK"
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
UNKNOWN_RESIDUE_LETTER = "X"


class AdapterError(RuntimeError):
    """A condition the operator has to fix before the stage can run."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of the complete file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows to a path in one atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one JSON document to a path in an atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_json(path: Path, label: str) -> dict[str, Any]:
    """Return the JSON object a file holds."""
    if not path.is_file():
        raise AdapterError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise AdapterError(f"{label} is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"{label} is not a JSON object: {path}")
    return value


def resolve_output_path(attempt_dir: Path, path: Path, label: str) -> Path:
    """Return an output path and refuse one that leaves the attempt directory."""
    resolved = Path(os.path.normpath(path if path.is_absolute() else attempt_dir / path))
    if attempt_dir not in resolved.parents:
        raise AdapterError(f"{label} escapes the attempt directory: {path}")
    return resolved


def resolve_input_path(artifact_root: Path, value: str, label: str) -> Path:
    """Return an input path and refuse one that leaves the artifact root."""
    raw = Path(value).expanduser()
    resolved = Path(os.path.normpath(raw if raw.is_absolute() else artifact_root / raw))
    if artifact_root not in resolved.parents:
        raise AdapterError(f"{label} is outside the artifact root: {value}")
    return resolved


# ----------------------------------------------------------------------------
# The runtime. The working directory is part of it here.
# ----------------------------------------------------------------------------


def resolve_home(value: Path | None) -> Path:
    """Return the directory Genie3 runs in."""
    if value is None:
        environment_value = os.environ.get("GENIE3_HOME", "").strip()
        if not environment_value:
            raise AdapterError(
                "Genie3 is not located. Pass --genie3-home, or set GENIE3_HOME to the "
                f"directory that holds {PRETRAINED_SUBDIR}/"
            )
        value = Path(environment_value)
    home = value.expanduser()
    if not home.is_dir():
        raise AdapterError(f"Genie3 home is not a directory: {home}")
    return home.resolve()


def resolve_execution(args: argparse.Namespace) -> tuple[str, Path]:
    """Resolve the execution route and the directory Genie3 runs in.

    ``local`` keeps the checkout contract, where --genie3-home or GENIE3_HOME
    names the directory. ``modal`` takes the mount the shipped Modal environment
    provides and still accepts an explicit path, which is how a user-built image
    names its own. ``auto`` reads the same home and reports ``modal`` only when
    that home is the shipped mount and the shipped checkout is beside it, which
    is the pair the container provides and a workstation does not.

    The route is recorded on every row. A dispatch that ran on Modal and a run
    that ran on this machine write different values, because a receipt is the
    artifact a later reader trusts to say where a structure came from.
    """
    protocol = getattr(args, "runner_protocol", DEFAULT_RUNNER_PROTOCOL)
    home_value = getattr(args, "genie3_home", None)

    if protocol == "local":
        return "local", resolve_home(home_value)

    if protocol == "modal":
        return "modal", resolve_home(home_value or MODAL_GENIE3_HOME)

    if protocol != "auto":
        raise AdapterError(
            f"runner protocol {protocol!r} is not supported; choose one of "
            f"{', '.join(RUNNER_PROTOCOLS)}. The fal route is a separate adapter, "
            "claude_binder.adapters.fal_genie3_generator"
        )

    home = resolve_home(home_value)
    if home == MODAL_GENIE3_HOME.expanduser().resolve() and MODAL_GENIE3_SOURCE.is_dir():
        return "modal", home
    return "local", home


def resolve_model_config(home: Path, model_version: str) -> Path:
    """Return the model configuration Genie3 reads from its working directory.

    Genie3 opens this path relative to the process working directory, so the
    check here and the cwd the subprocess receives have to name one directory. A
    home without this file is the failure that exits in under a second.
    """
    if IDENTIFIER_RE.fullmatch(model_version) is None:
        raise AdapterError(f"model version is not a plain identifier: {model_version}")
    config_path = home / PRETRAINED_SUBDIR / model_version / MODEL_CONFIG_NAME
    if not config_path.is_file():
        raise AdapterError(
            f"Genie3 model configuration not found: {config_path}. Genie3 reads "
            f"{PRETRAINED_SUBDIR}/{model_version}/{MODEL_CONFIG_NAME} relative to its working "
            "directory, so --genie3-home has to name the directory that holds it"
        )
    return config_path


def resolve_tool_python(value: str | None) -> str:
    """Return the interpreter that runs a Genie3 module."""
    if value is None:
        return sys.executable
    resolved = shutil.which(value)
    if resolved is None:
        raise AdapterError(f"interpreter not found: {value}")
    return resolved


def resolve_command(args: argparse.Namespace) -> list[str]:
    """Return the argument list prefix that starts Genie3.

    The console script is the documented entry point. An installation that
    exposes the package without the script runs through --genie3-module.
    """
    if args.genie3_module is not None:
        if CONFIG_KEY_RE.fullmatch(args.genie3_module) is None:
            raise AdapterError(f"module name is not a Python module path: {args.genie3_module}")
        return [resolve_tool_python(args.tool_python), "-m", args.genie3_module]
    candidate = Path(args.genie3_command).expanduser()
    if candidate.is_absolute() or len(candidate.parts) > 1:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise AdapterError(f"Genie3 command is not an executable file: {candidate}")
        return [str(candidate.resolve())]
    resolved = shutil.which(args.genie3_command)
    if resolved is None:
        raise AdapterError(
            f"Genie3 command not found on PATH: {args.genie3_command}. Pass --genie3-command "
            "with a path, or --genie3-module to run the package as a module"
        )
    return [resolved]


def run_tool(argv: list[str], *, cwd: Path) -> None:
    """Run one argument list in one working directory and fail on a nonzero exit."""
    print(f"genie3 adapter: run {shlex.join(argv)} in {cwd}", flush=True)
    completed = subprocess.run(argv, shell=False, check=False, cwd=str(cwd))
    if completed.returncode != 0:
        raise AdapterError(f"genie3 generate exited {completed.returncode} in {cwd}")


# ----------------------------------------------------------------------------
# The upstream target manifest.
# ----------------------------------------------------------------------------


def completed_receipt_file(receipts_dir: Path, stage_id: str, artifact_id: str) -> Path:
    """Return the one file a completed upstream receipt recorded for an artifact."""
    receipt_path = receipts_dir / f"{stage_id}.json"
    receipt = load_json(receipt_path, "upstream receipt")
    if receipt.get("ok") is not True:
        raise AdapterError(f"upstream receipt did not complete: {receipt_path}")
    artifacts = receipt.get("output_manifest", {}).get("artifacts", [])
    phases = {str(artifact.get("phase")) for artifact in artifacts}
    selected_phase = "scale" if "scale" in phases else "single"
    records = [
        file_record
        for artifact in artifacts
        if artifact.get("phase") == selected_phase and artifact.get("artifact_id") == artifact_id
        for file_record in artifact.get("files", [])
    ]
    paths: list[Path] = []
    for file_record in records:
        # A receipt is another stage's output, so a malformed file entry is bad
        # input rather than a bug here, and it gets the same named error as every
        # other bad input.
        if not isinstance(file_record, dict) or not str(file_record.get("path") or "").strip():
            raise AdapterError(
                f"upstream receipt {receipt_path} carries a {artifact_id} file entry with no "
                f"path for phase {selected_phase}: {json.dumps(file_record, sort_keys=True)}"
            )
        paths.append(Path(str(file_record["path"])))
    if len(paths) != 1:
        raise AdapterError(
            f"upstream receipt {receipt_path} carries {len(paths)} {artifact_id} files for phase "
            f"{selected_phase}; expected exactly one"
        )
    return paths[0]


def load_target_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    """Return the target manifest this stage designs against, and its path."""
    if args.target_manifest is not None:
        artifact_root = args.artifact_root.expanduser().resolve()
        path = resolve_input_path(artifact_root, str(args.target_manifest), "target manifest")
    else:
        path = completed_receipt_file(
            args.receipts_dir, args.target_stage_id, args.target_artifact_id
        )
    manifest = load_json(path, "target manifest")
    for field in REQUIRED_TARGET_MANIFEST_FIELDS:
        if not manifest.get(field):
            raise AdapterError(f"target manifest {path} records no {field}")
    source_path = Path(str(manifest["source_structure_path"]))
    if not source_path.is_file():
        raise AdapterError(f"target source structure is missing: {source_path}")
    observed = sha256_file(source_path)
    if observed != str(manifest["target_sha256"]):
        raise AdapterError(
            f"target source structure changed since the target stage: {source_path}; "
            f"the manifest records {manifest['target_sha256']} and the file reads {observed}"
        )
    return manifest, path


def target_structure(manifest: dict[str, Any], path: Path) -> tuple[Path, str]:
    """Return the normalized target structure Genie3 reads and check its hash."""
    structure_path = Path(str(manifest["normalized_structure_path"])).expanduser()
    if not structure_path.is_file():
        raise AdapterError(f"normalized target structure is missing: {structure_path}")
    observed = sha256_file(structure_path)
    recorded = manifest.get("normalized_structure_sha256")
    if isinstance(recorded, str) and recorded and recorded != observed:
        raise AdapterError(
            f"normalized target structure changed since the target stage: {structure_path}; "
            f"the manifest records {recorded} and the file reads {observed}"
        )
    return structure_path.resolve(), observed


def site_residue_ids(manifest: dict[str, Any], path: Path) -> list[str]:
    """Return the design site residue IDs the target manifest resolved."""
    site = manifest.get("site")
    if not isinstance(site, dict):
        raise AdapterError(f"target manifest {path} records no site")
    residues = site.get("resolved_design_residues")
    if not isinstance(residues, list) or not residues:
        raise AdapterError(f"target manifest {path} records no resolved_design_residues")
    return [str(residue) for residue in residues]


def site_token(residue_id: str, chain: str) -> str:
    """Return the Genie3 site token of one residue ID."""
    match = RESIDUE_ID_RE.fullmatch(residue_id)
    if match is None:
        raise AdapterError(f"site residue does not read CHAIN:NUMBER: {residue_id}")
    residue_chain, number, insertion_code = match.group(1), match.group(2), match.group(3)
    if insertion_code:
        raise AdapterError(
            f"site residue {residue_id} carries an insertion code, and a Genie3 site token "
            "carries no place for one"
        )
    if residue_chain != chain:
        raise AdapterError(
            f"site residue {residue_id} names chain {residue_chain}, and the design target chain "
            f"is {chain}"
        )
    return f"{residue_chain}{number}"


# ----------------------------------------------------------------------------
# Structure records. Nothing here writes a coordinate the tool did not.
# ----------------------------------------------------------------------------


def read_atom_records(path: Path) -> list[str]:
    """Return the coordinate lines of a structure file."""
    if not path.is_file():
        raise AdapterError(f"structure file is missing: {path}")
    return [
        line
        for line in path.read_text(errors="replace").splitlines()
        if line.startswith(ATOM_RECORD_PREFIXES)
    ]


def chain_atom_lines(lines: list[str], chain: str) -> list[str]:
    """Return the coordinate lines of one chain."""
    return [line for line in lines if len(line) > 21 and line[21:22] == chain]


def chain_ids(lines: list[str]) -> list[str]:
    """Return chain IDs in first-seen order."""
    seen: list[str] = []
    for line in lines:
        if len(line) > 21 and line[21:22] not in seen:
            seen.append(line[21:22])
    return seen


def chain_residues(lines: list[str]) -> list[tuple[str, str]]:
    """Return the residue number field and name of every residue in order."""
    residues: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in lines:
        number, name = line[22:27].strip(), line[17:20].strip().upper()
        if number in seen:
            continue
        seen.add(number)
        residues.append((number, name))
    return residues


def atom_names(lines: list[str]) -> list[str]:
    """Return the distinct atom names the coordinate lines carry.

    The order is `backbone_shape.order_atoms`, so a row reads the same way as the
    guard that holds a generator's atoms against what a sequence designer needs.
    """
    return list(order_atoms(line[12:16].strip() for line in lines))


def chain_sequence(residues: list[tuple[str, str]]) -> str:
    """Return the one-letter residue string of one chain."""
    return "".join(THREE_TO_ONE.get(name, UNKNOWN_RESIDUE_LETTER) for _, name in residues)


def residue_number_range(residues: list[tuple[str, str]], chain: str) -> str:
    """Return the Genie3 chain-and-residues token of one chain."""
    numbers: list[int] = []
    for number, _ in residues:
        match = RESIDUE_NUMBER_RE.match(number)
        if match is None:
            raise AdapterError(
                f"chain {chain} carries a residue number this wrapper cannot read: {number}"
            )
        numbers.append(int(match.group(1)))
    if not numbers:
        raise AdapterError(f"chain {chain} carries no residue")
    return f"{chain}{min(numbers)}-{max(numbers)}"


def swap_chain_labels(lines: list[str], first: str, second: str) -> list[str]:
    """Exchange two chain letters in the PDB chain column.

    Column 22 is the only column this touches. The coordinates, the atom names,
    and the residue identities stay the bytes Genie3 wrote, because a design pose
    that carries an atom the generator did not produce is a fabrication.

    It is a swap rather than a one-way relabel because the pose carries the
    target as well as the binder. Rewriting the binder onto the letter the target
    already holds would give one pose two chains with one ID. When the pose
    carries only ``first``, nothing maps back and the swap is a plain relabel.
    """
    mapping = {first: second, second: first}
    return [line[:21] + mapping.get(line[21:22], line[21:22]) + line[22:] for line in lines]


def chain_report(lines: list[str]) -> str:
    """Return a chain-by-chain residue count, for a refusal message."""
    parts = [
        f"{chain} has {len(chain_residues(chain_atom_lines(lines, chain)))} residues"
        for chain in chain_ids(lines)
    ]
    return "; ".join(parts) if parts else "no chain"


def designed_chains(lines: list[str]) -> list[str]:
    """Return the chains whose every residue name is the one Genie3 designs with.

    Genie3 is handed the target's sequence and designs a backbone that has none,
    so it writes ``UNK`` for every residue of the chain it generated and the
    target's own residue names for the chain it copied through. That property
    belongs to what the model does. It does not depend on a chain letter, and it
    cannot drift the way a letter convention can.
    """
    designed: list[str] = []
    for chain in chain_ids(lines):
        residues = chain_residues(chain_atom_lines(lines, chain))
        if residues and all(name == DESIGNED_RESIDUE_NAME for _, name in residues):
            designed.append(chain)
    return designed


def select_binder_chain(
    path: Path,
    *,
    binder_chain: str,
    target_chain: str,
    source_chain: str | None,
    minimum_length: int,
    maximum_length: int,
) -> tuple[list[str], list[str], str]:
    """Return the whole pose with the binder on ``binder_chain``, refusing a guess.

    Genie3 decides which chain letter carries the binder and which carries the
    target, and it does not keep the letter the wrapper handed it. One recorded
    run read a 115-residue target in on chain A and wrote it back out on chain B
    with the 70-residue generated binder on A. So neither the letter the campaign
    declares nor the letter the wrapper wrote into the problem identifies a chain
    of the output.

    Residue names do. Genie3 writes ``UNK`` for the chain it designed and the
    target's own residue names for the chain it copied through, so a chain that
    is entirely ``UNK`` is a candidate binder and a chain carrying real residue
    names is the target. That is the first filter.

    The length bounds are the second, and they still refuse. A measured run wrote
    95 residues on chain A and 115, the target length, on chain B, and the wrapper
    of the day published chain B, so the sequence designer redesigned the target.
    Nothing downstream reports that as an error. So the binder is the candidate
    whose residue count falls inside the campaign's binder length bounds. None
    inside them refuses, and it says the binder came out the wrong length when a
    chain the names ruled out is the only one that fits. More than one inside them
    refuses, because the counts cannot then say which chain is the binder, and the
    operator resolves it with --generator-binder-chain. A chain named that way is
    still held against both filters, so naming one is not a way past either.

    ``target_chain`` is the fallback, for the two outputs whose residue names
    separate nothing. Every chain ``UNK`` is a target that arrived stripped of its
    identities, as a poly-glycine or a scrubbed input would. No chain ``UNK`` is an
    output whose designed chain carries names the tool assigned. In both, the one
    thing the campaign still knows is the letter it wrote the target into, so that
    letter is excluded and the bounds decide among the rest. The fallback inherits
    the letter's weakness against relabeling and is the best available signal when
    the names are silent.

    Every chain is published, matching `rfdiffusion_generator.write_design_pose`.
    Three consumers in this package read the target chain out of a design pose:
    `proteinmpnn_designer.write_design_pose` copies it forward as fixed context,
    the interface scorer hands the pose to DockQ naming both chains, and the
    cofold adapters compare its target residue keys against the prepared target.
    """
    lines = read_atom_records(path)
    if not lines:
        raise AdapterError(f"Genie3 wrote a structure with no coordinate record: {path}")
    counts = {
        candidate: len(chain_residues(chain_atom_lines(lines, candidate)))
        for candidate in chain_ids(lines)
    }
    bounds = f"{minimum_length}-{maximum_length}"
    designed = designed_chains(lines)
    # The residue names separate the chains only when some chain carries the
    # designed name and some chain does not. All or none leaves them silent.
    names_separate = 0 < len(designed) < len(counts)
    candidates = designed if names_separate else [
        candidate for candidate in counts if candidate != target_chain
    ]
    if source_chain is not None:
        if source_chain not in counts:
            raise AdapterError(
                f"{path} carries no coordinate record for chain {source_chain}; the file holds "
                f"{chain_report(lines)}. --generator-binder-chain has to name a chain the tool "
                "wrote"
            )
        if source_chain not in candidates:
            if names_separate:
                raise AdapterError(
                    f"--generator-binder-chain names chain {source_chain}, which carries real "
                    f"residue names and is therefore the target Genie3 copied through; the file "
                    f"holds {chain_report(lines)}. Publishing the target as the binder sends the "
                    "target to the sequence designer, so this wrapper never does it. Name the "
                    f"chain Genie3 designed, which reads {DESIGNED_RESIDUE_NAME} throughout"
                )
            raise AdapterError(
                f"--generator-binder-chain names chain {source_chain}, which is the target chain "
                f"of this campaign; the file holds {chain_report(lines)}. Publishing the target "
                "as the binder sends the target to the sequence designer, so this wrapper never "
                "does it. Name the chain Genie3 wrote the binder into"
            )
        if not minimum_length <= counts[source_chain] <= maximum_length:
            raise AdapterError(
                f"{path} chain {source_chain} has {counts[source_chain]} residues, outside the "
                f"binder length bounds {bounds}; the file holds {chain_report(lines)}. "
                "--generator-binder-chain names which chain carries the binder, and that chain's "
                "residue count still has to fall inside the campaign's bounds"
            )
        chain = source_chain
    else:
        fitting = [
            candidate
            for candidate in candidates
            if minimum_length <= counts[candidate] <= maximum_length
        ]
        if not fitting:
            ruled_out = [
                candidate
                for candidate in counts
                if candidate not in candidates
                and minimum_length <= counts[candidate] <= maximum_length
            ]
            if ruled_out:
                produced = "; ".join(
                    f"{candidate} with {counts[candidate]} residues" for candidate in candidates
                )
                only = (
                    f"{', '.join(ruled_out)}, which carries the target's own residue names"
                    if names_separate
                    else f"{target_chain}, the target"
                )
                raise AdapterError(
                    f"{path} holds no binder inside the binder length bounds {bounds}. Genie3 "
                    f"designed {produced or 'no chain besides the target'}, and the only chain "
                    f"inside the bounds is {only}. The generated binder came "
                    "out the wrong length for this campaign, so there is no binder here to "
                    "publish. Widen the campaign's binder length bounds to the range Genie3 was "
                    "asked to sample, or discard this run"
                )
            raise AdapterError(
                f"{path} carries no chain inside the binder length bounds {bounds}; the file "
                f"holds {chain_report(lines)}. The binder is the chain whose residue count falls "
                "inside those bounds, and this file has none, so it is not a binder design "
                "against this campaign"
            )
        if len(fitting) > 1:
            raise AdapterError(
                f"{path} carries {len(fitting)} chains inside the binder length bounds {bounds}: "
                f"{', '.join(fitting)}; the file holds {chain_report(lines)}. The residue counts "
                "cannot say which one is the binder, so pass --generator-binder-chain with the "
                "chain Genie3 wrote the binder into"
            )
        chain = fitting[0]
    published = lines if chain == binder_chain else swap_chain_labels(lines, chain, binder_chain)
    return published, [number for number, _ in chain_residues(chain_atom_lines(lines, chain))], chain


def write_design_pose(
    path: Path,
    *,
    candidate_id: str,
    lines: list[str],
    source_pose: Path,
    source_sha256: str,
    chain: str,
    source_chain: str,
) -> None:
    """Write the design pose this candidate owns.

    The coordinates are one Genie3 output copied without addition, every chain
    kept, with the chain column rewritten only where the binder and the target
    letters were swapped. The REMARK lines name the candidate, the chain the
    binder is published as, the chain the tool wrote it into, and the file the
    coordinates came from, which also keeps the bytes of every pose distinct.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"REMARK 900 DESIGN POSE {candidate_id}",
        f"REMARK 900 DESIGN CHAIN {chain}",
        f"REMARK 900 GENERATOR CHAIN {source_chain}",
        f"REMARK 900 SOURCE POSE {source_pose}",
        f"REMARK 900 SOURCE SHA256 {source_sha256}",
        "REMARK 900 BACKBONE ONLY WITH PLACEHOLDER RESIDUE IDENTITY",
        "REMARK 900 ATOM RECORDS COPIED FROM THE TOOL OUTPUT; NO ATOM IS SYNTHESIZED",
    ]
    body: list[str] = []
    previous_chain: str | None = None
    for line in lines:
        current = line[21:22]
        if previous_chain is not None and current != previous_chain:
            body.append("TER")
        body.append(line)
        previous_chain = current
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write("\n".join([*header, *body, "TER", "END"]) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


# ----------------------------------------------------------------------------
# The Genie3 dataset and configuration.
# ----------------------------------------------------------------------------


def yaml_scalar(value: Any) -> str:
    """Return one YAML scalar."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise AdapterError(f"configuration value has no YAML form: {value!r}")


def render_yaml(mapping: dict[str, Any], indent: int = 0) -> list[str]:
    """Return the YAML lines of a mapping of scalars and nested mappings."""
    lines: list[str] = []
    pad = "  " * indent
    for key, value in mapping.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.extend(render_yaml(value, indent + 1))
            continue
        lines.append(f"{pad}{key}: {yaml_scalar(value)}")
    return lines


def set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set one value at a dotted key path, refusing a path a scalar already holds."""
    if CONFIG_KEY_RE.fullmatch(dotted_key) is None:
        raise AdapterError(f"configuration key is not a dotted identifier path: {dotted_key}")
    parts = dotted_key.split(".")
    cursor = mapping
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise AdapterError(f"configuration key {dotted_key} runs through the scalar {part}")
        cursor = child
    cursor[parts[-1]] = value


def write_dataset(
    dataset_root: Path,
    *,
    problem_id: str,
    chain: str,
    lines: list[str],
    residues: list[tuple[str, str]],
    hotspot: list[str],
    extended: list[str],
    binder_minimum_length: int,
    binder_maximum_length: int,
    target_id: str,
    source_id: str,
) -> tuple[Path, Path]:
    """Write the binderbench problem Genie3 reads and return its files.

    Genie3 resolves the file paths inside the problem with `open()`, which reads
    a relative path from the working directory. The working directory is the
    Genie3 home and the dataset is not under it, so every path here is absolute.
    """
    problems_dir = dataset_root / "problems"
    pdb_dir = dataset_root / "targets" / "pdb"
    fasta_dir = dataset_root / "targets" / "fasta"
    for directory in (problems_dir, pdb_dir, fasta_dir):
        directory.mkdir(parents=True, exist_ok=True)

    chain_pdb = pdb_dir / f"{problem_id}-chain_{chain}.pdb"
    full_pdb = pdb_dir / f"{problem_id}.pdb"
    chain_pdb.write_text("\n".join([*lines, "TER", "END"]) + "\n")
    shutil.copyfile(chain_pdb, full_pdb)

    sequence = chain_sequence(residues)
    chain_fasta = fasta_dir / f"{problem_id}-chain_{chain}.fasta"
    full_fasta = fasta_dir / f"{problem_id}.fasta"
    chain_fasta.write_text(f">target_chain_{chain}\n{sequence}\n")
    shutil.copyfile(chain_fasta, full_fasta)

    problem = {
        "key": problem_id,
        "name": problem_id,
        "target_pdb_filepath": str(full_pdb.resolve()),
        "target_fasta_filepath": str(full_fasta.resolve()),
        "target_pdb_filepath_by_chain": [str(chain_pdb.resolve())],
        "target_fasta_filepath_by_chain": [str(chain_fasta.resolve())],
        "target_chain_and_residues": [residue_number_range(residues, chain)],
        "target_interface_residues": {
            "hotspot": hotspot,
            "extended": extended,
            "common": list(hotspot),
        },
        "binder_min_length": binder_minimum_length,
        "binder_max_length": binder_maximum_length,
        "tag": [target_id],
        "pdb_id": source_id,
    }
    problem_path = problems_dir / f"{problem_id}.json"
    problem_path.write_text(json.dumps(problem, indent=4, sort_keys=True) + "\n")
    return problem_path, full_pdb


def build_config(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    run_root: Path,
    experiment_name: str,
) -> tuple[dict[str, Any], int | None]:
    """Return the Genie3 configuration mapping and the seed the tool receives.

    `generation.dataset.source: target` is what puts Genie3 in binder mode.
    `unconditional` is the other value, and it ignores the problem entirely.
    """
    config: dict[str, Any] = {
        "experiment": {"name": experiment_name},
        "paths": {"rootdir": str(run_root.resolve()), "dataset": str(dataset_root.resolve())},
        "generation": {
            "dataset": {
                "source": "target",
                "selections": args.problem_id,
                "n_sample": args.count,
            },
            "sampler": {"sampler": {"direction_scale": float(args.direction_scale)}},
        },
    }
    # The wrapper runs generate, which evaluates nothing, so the evaluation
    # section stays out unless an operator asks for it. A section naming
    # colabfold in a minimal install is the crash `genie3 run` produces.
    if args.evaluation_folding_model is not None:
        config["evaluation"] = {
            "version": args.evaluation_version,
            "folding": {"model_name": args.evaluation_folding_model},
        }
    if args.seed_config_key is not None:
        set_nested(config, args.seed_config_key, args.seed)
        return config, args.seed
    return config, None


def write_config(path: Path, config: dict[str, Any]) -> None:
    """Write one Genie3 configuration file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(render_yaml(config)) + "\n")


# ----------------------------------------------------------------------------
# Outputs.
# ----------------------------------------------------------------------------


def generated_poses(run_dir: Path, run_root: Path, problem_id: str) -> list[Path]:
    """Return the structures Genie3 wrote for one problem, in a stable order.

    The recorded layout is <rootdir>/<selection>/pdbs/<selection>_<index>.pdb. A
    glob of the problem directory alone finds nothing, because every file sits
    one level further down.
    """
    if not run_dir.is_dir():
        present = sorted(item.name for item in run_root.iterdir()) if run_root.is_dir() else []
        raise AdapterError(
            f"genie3 generate wrote no directory for problem {problem_id}: {run_dir}. "
            f"The run root holds: {', '.join(present) if present else 'nothing'}"
        )
    produced = sorted(run_dir.rglob("*.pdb"))
    preferred = [path for path in produced if set(path.parts) & set(PREFERRED_OUTPUT_DIRS)]
    return preferred if preferred else produced


def build_candidate(
    args: argparse.Namespace,
    *,
    index: int,
    source_pose: Path,
    phase_dir: Path,
    target_chain: str,
) -> dict[str, Any]:
    """Return the observed facts of one generated structure and write its pose."""
    candidate_id = f"{args.generator_id}-{index:03d}"
    lines, binder_residue_numbers, source_chain = select_binder_chain(
        source_pose,
        binder_chain=args.binder_chain,
        target_chain=target_chain,
        source_chain=args.generator_binder_chain,
        minimum_length=args.binder_length_min,
        maximum_length=args.binder_length_max,
    )
    if source_chain != args.binder_chain:
        print(
            f"genie3 adapter: {candidate_id} reads chain {source_chain} "
            f"({len(binder_residue_numbers)} residues) and publishes it as chain "
            f"{args.binder_chain}, swapping the two chain letters"
        )
    # The published pose has to carry the target as well as the binder, because
    # the interface scorer hands this file to DockQ naming both chains and the
    # cofold adapters read the target residue keys out of it.
    target_lines = chain_atom_lines(lines, target_chain)
    if not target_lines:
        raise AdapterError(
            f"{candidate_id} pose carries no coordinate record for the target chain "
            f"{target_chain}; {source_pose} holds {chain_report(lines)}. The interface scorer "
            "and the cofold adapters read the target chain out of the design pose, so a "
            "binder-only pose cannot be scored against this target"
        )
    binder_lines = chain_atom_lines(lines, args.binder_chain)
    names = sorted({name for _, name in chain_residues(binder_lines)})
    observed_atoms = atom_names(binder_lines)
    source_sha256 = sha256_file(source_pose)
    pose_path = phase_dir / args.pose_subdir / f"{candidate_id}.pdb"
    write_design_pose(
        pose_path,
        candidate_id=candidate_id,
        lines=lines,
        source_pose=source_pose,
        source_sha256=source_sha256,
        chain=args.binder_chain,
        source_chain=source_chain,
    )
    return {
        "candidate_id": candidate_id,
        "design_index": index,
        "design_pose_path": str(pose_path.resolve()),
        "design_pose_sha256": sha256_file(pose_path),
        "tool_output_path": str(source_pose.resolve()),
        "tool_output_sha256": source_sha256,
        "binder_chain_id": args.binder_chain,
        "generator_binder_chain_id": source_chain,
        "binder_residue_count": len(binder_residue_numbers),
        "binder_atom_record_count": len(binder_lines),
        "target_chain_id": target_chain,
        "target_residue_count": len(chain_residues(target_lines)),
        "pose_chain_ids": chain_ids(lines),
        "pose_atom_record_count": len(lines),
        "design_residue_names": names,
        "design_atom_names": observed_atoms,
        "backbone_atoms": observed_atoms,
        "residue_identity": (
            "placeholder" if set(names) <= PLACEHOLDER_RESIDUE_NAMES else "tool-assigned"
        ),
    }


# ----------------------------------------------------------------------------
# Subcommands.
# ----------------------------------------------------------------------------


def config_model_revision(config: Path | None, adapter_id: str = DEFAULT_ADAPTER_ID) -> str | None:
    """Return the weights revision the resolved config declares for this adapter.

    The profile records the Genie3 weights repository and commit, and nothing in
    the tool output says which weights produced it. Copying the declared revision
    onto every row is what `rfdiffusion3_generator.config_model_revision` does, so
    a candidate can be traced back to the weights the campaign pinned. Returning
    null when the dispatcher passes no config keeps a hand-run command working.
    """
    if config is None:
        return None
    document = load_json(config.expanduser().resolve(), "resolved config")
    for adapter in document.get("adapters", []):
        if isinstance(adapter, dict) and adapter.get("adapter_id") == adapter_id:
            value = adapter.get("model_revision")
            if isinstance(value, str) and value:
                return value
            raise AdapterError(
                f"resolved config {config} records no model_revision for {adapter_id}"
            )
    return None


def run(args: argparse.Namespace) -> int:
    """Generate one phase of backbones and write the stage outputs."""
    if args.count < 1:
        raise AdapterError(f"--count is {args.count}; the phase needs at least one backbone")
    if args.seed < 0:
        raise AdapterError(f"--seed is {args.seed}; a seed is not negative")
    if args.seed_config_key is None and not args.allow_unseeded:
        raise AdapterError(
            "no Genie3 invocation this package records passes a seed. Pass --seed-config-key "
            "with the configuration key your build reads, or --allow-unseeded to record that "
            "this run carries no seed"
        )
    if IDENTIFIER_RE.fullmatch(args.problem_id) is None:
        raise AdapterError(f"--problem-id is not a plain identifier: {args.problem_id}")
    if IDENTIFIER_RE.fullmatch(args.generator_id) is None:
        raise AdapterError(f"--generator-id is not a plain identifier: {args.generator_id}")
    for label, chain in (
        ("--binder-chain", args.binder_chain),
        ("--generator-binder-chain", args.generator_binder_chain),
    ):
        if chain is not None and CHAIN_ID_RE.fullmatch(chain) is None:
            raise AdapterError(f"{label} is {chain}; a chain ID is one letter or digit")
    if args.binder_length_min < 1 or args.binder_length_max < args.binder_length_min:
        raise AdapterError(
            f"binder length bounds {args.binder_length_min}-{args.binder_length_max} are empty"
        )

    runner_protocol, home = resolve_execution(args)
    model_config = resolve_model_config(home, args.model_version)
    command = resolve_command(args)
    model_revision = config_model_revision(args.config)

    manifest, manifest_source = load_target_manifest(args)
    target_id = str(manifest["target_id"])
    source_id = str(manifest.get("source_id") or target_id)
    chain = args.target_chain or str(manifest.get("design_target_chain_id") or "")
    if not chain:
        raise AdapterError(
            f"target manifest {manifest_source} records no design_target_chain_id; pass "
            "--target-chain with the chain the binder is designed against"
        )
    if chain == args.binder_chain:
        # The binder chain default is the campaign template's own value, so this
        # fires on a target the campaign put on that same letter. Naming where each
        # letter came from is what tells the operator which flag to change.
        binder_source = (
            f"--binder-chain defaults to {DEFAULT_BINDER_CHAIN}"
            if args.binder_chain == DEFAULT_BINDER_CHAIN
            else f"--binder-chain is {args.binder_chain}"
        )
        chain_source = (
            "--target-chain"
            if args.target_chain
            else f"design_target_chain_id in {manifest_source}"
        )
        raise AdapterError(
            f"the target chain and the binder chain are both {chain}; {binder_source} and the "
            f"target chain came from {chain_source}. They name two different chains of the "
            "design pose, which carries the binder and the target together. Pass --binder-chain "
            "with the chain the campaign publishes the binder as"
        )
    structure_path, structure_sha256 = target_structure(manifest, manifest_source)

    target_lines = chain_atom_lines(read_atom_records(structure_path), chain)
    if not target_lines:
        raise AdapterError(
            f"the normalized target structure carries no coordinate record for chain {chain}: "
            f"{structure_path}"
        )
    target_residues = chain_residues(target_lines)
    hotspot = [
        site_token(residue, chain) for residue in site_residue_ids(manifest, manifest_source)
    ]
    extended = [
        token
        for token in (site_token(residue, chain) for residue in args.extended_site_residue)
        if token not in hotspot
    ]

    attempt_dir = args.attempt_dir.expanduser().resolve()
    phase_dir = attempt_dir / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        resolve_output_path(attempt_dir, args.manifest_path, "manifest path")
        if args.manifest_path is not None
        else phase_dir / DEFAULT_MANIFEST_NAME
    )

    work_dir = phase_dir / args.work_subdir
    dataset_root = work_dir / DATASET_SUBDIR
    run_root = work_dir / RUN_SUBDIR
    run_dir = run_root / args.problem_id
    existing = generated_poses(run_dir, run_root, args.problem_id) if run_dir.is_dir() else []
    if existing:
        raise AdapterError(
            f"the run directory already holds {len(existing)} structure files: {run_dir}. Genie3 "
            "writes returned designs there by name, so a stale file from an earlier run cannot "
            "be told apart. Run the stage in a clean attempt directory"
        )
    run_root.mkdir(parents=True, exist_ok=True)
    problem_path, _ = write_dataset(
        dataset_root,
        problem_id=args.problem_id,
        chain=chain,
        lines=target_lines,
        residues=target_residues,
        hotspot=hotspot,
        extended=extended,
        binder_minimum_length=args.binder_length_min,
        binder_maximum_length=args.binder_length_max,
        target_id=target_id,
        source_id=source_id,
    )
    experiment_name = args.experiment_name or f"{args.generator_id}-{args.phase}"
    config, delivered_seed = build_config(
        args, dataset_root=dataset_root, run_root=run_root, experiment_name=experiment_name
    )
    config_path = work_dir / f"{args.problem_id}.config.yaml"
    write_config(config_path, config)

    # The subprocess starts in the Genie3 home, because Genie3 reads the model
    # configuration relative to the working directory. Every path the command
    # carries is absolute, so the working directory changes nothing else.
    run_tool([*command, "generate", "-c", str(config_path.resolve())], cwd=home)

    produced = generated_poses(run_dir, run_root, args.problem_id)
    if len(produced) < args.count:
        raise AdapterError(
            f"phase {args.phase} needs {args.count} structures and genie3 generate wrote "
            f"{len(produced)} under {run_dir}"
        )

    config_sha256 = sha256_file(config_path)
    problem_sha256 = sha256_file(problem_path)
    model_config_sha256 = sha256_file(model_config)
    rows: list[dict[str, Any]] = []
    for index, source_pose in enumerate(produced[: args.count]):
        record = build_candidate(
            args,
            index=index,
            source_pose=source_pose,
            phase_dir=phase_dir,
            target_chain=chain,
        )
        candidate_id = str(record["candidate_id"])
        rows.append(
            {
                "target_id": target_id,
                "target_sha256": str(manifest["target_sha256"]),
                "parent_candidate_id": None,
                "origin_generator": args.generator_id,
                **backbone_lineage(candidate_id, args.generator_id),
                "generator_mode": GENERATOR_MODE,
                "runner_protocol": runner_protocol,
                "sequence_designer": None,
                "generator_seed": args.seed,
                "requested_seed": args.seed,
                "tool_seed": delivered_seed,
                "seed_delivered": delivered_seed is not None,
                "seed_config_key": args.seed_config_key,
                "sequence_path": None,
                "sequence_sha256": None,
                "sequence_length": None,
                "backbone_only": True,
                "structure_path": str(manifest["source_structure_path"]),
                "structure_sha256": str(manifest["target_sha256"]),
                "residue_map_sha256": str(manifest["residue_map_sha256"]),
                "optimization_round": 0,
                "last_optimizer": None,
                "status": CANDIDATE_STATUS,
                "stage_id": args.stage,
                "target_manifest_path": str(manifest_source),
                "input_structure_path": str(structure_path),
                "input_structure_sha256": structure_sha256,
                "genie3_home": str(home),
                "working_directory": str(home),
                "model_version": args.model_version,
                "model_revision": model_revision,
                "model_config_sha256": model_config_sha256,
                "config_path": str(config_path.resolve()),
                "config_sha256": config_sha256,
                "problem_path": str(problem_path.resolve()),
                "problem_sha256": problem_sha256,
                "problem_id": args.problem_id,
                "n_sample": args.count,
                "direction_scale": float(args.direction_scale),
                "binder_length_min": args.binder_length_min,
                "binder_length_max": args.binder_length_max,
                **record,
            }
        )
    write_jsonl(manifest_path, rows)
    print(
        f"genie3 adapter: phase={args.phase} route={runner_protocol} target={target_id} "
        f"candidates={len(rows)} structures={len(produced)} "
        f"seed_delivered={delivered_seed is not None} "
        f"working_directory={home} manifest={manifest_path} "
        f"target_manifest={manifest_source}"
    )
    return 0


def toolcheck(args: argparse.Namespace) -> int:
    """Report the Genie3 route, command, working directory, and model configuration."""
    runner_protocol, home = resolve_execution(args)
    model_config = resolve_model_config(home, args.model_version)
    command = resolve_command(args)
    completed = subprocess.run(
        [*command, "--help"],
        shell=False,
        check=False,
        cwd=str(home),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise AdapterError(
            f"genie3 --help exited {completed.returncode}: {completed.stderr.strip()[:400]}"
        )
    reported = next(
        (line.strip() for line in completed.stdout.splitlines() if line.strip()), "no output"
    )
    print(f"genie3 adapter: route {runner_protocol}")
    print(f"genie3 adapter: command {shlex.join(command)}")
    print(f"genie3 adapter: working directory {home}")
    print(f"genie3 adapter: model configuration {model_config}")
    print(f"genie3 adapter: model configuration sha256 {sha256_file(model_config)}")
    print(f"genie3 adapter: probe {reported}")
    print(
        "genie3 adapter: the probe runs in the working directory the run uses, because Genie3 "
        f"reads {PRETRAINED_SUBDIR}/{args.model_version}/{MODEL_CONFIG_NAME} relative to it"
    )
    print("genie3 adapter: the probe downloads no weights and generates nothing")
    return 0


def stage_record(config_path: Path, stage_id: str) -> dict[str, Any]:
    """Return one stage contract from a resolved config."""
    config = load_json(config_path, "resolved config")
    matches = [
        stage
        for stage in config.get("stages", [])
        if isinstance(stage, dict) and stage.get("stage_id") == stage_id
    ]
    if len(matches) != 1:
        raise AdapterError(
            f"{config_path} registers {len(matches)} stages with id {stage_id}; "
            "the parser needs exactly one"
        )
    return matches[0]


def parser_output_pattern(template: str, attempt_dir: Path, phase: str, stage_id: str) -> str:
    """Render an output contract path for the parser."""
    rendered = (
        template.replace("{{attempt_dir}}", str(attempt_dir))
        .replace("{{phase}}", phase)
        .replace("{{stage_id}}", stage_id)
    )
    if "{{" in rendered or "}}" in rendered:
        raise AdapterError(f"parser output path carries an unsupported token: {template}")
    return rendered


def parse_outputs(args: argparse.Namespace) -> int:
    """Check the phase outputs this stage declares and write the parser result."""
    attempt_dir = args.attempt_dir.expanduser().resolve()
    phase_dir = attempt_dir / args.phase
    stage = stage_record(args.config.expanduser().resolve(), args.stage)
    files: list[Path] = []
    parsed_count = 0
    errors: list[str] = []
    for output in stage.get("outputs", []):
        if not isinstance(output, dict) or not isinstance(output.get("path_template"), str):
            errors.append("stage output has no path_template")
            continue
        pattern = parser_output_pattern(
            output["path_template"], attempt_dir, args.phase, args.stage
        )
        for value in sorted(glob.glob(pattern, recursive=True)):
            path = Path(value)
            if not path.is_file():
                continue
            files.append(path)
            try:
                kind = output.get("kind")
                if kind == "jsonl":
                    for line in path.read_text().splitlines():
                        if line.strip():
                            json.loads(line)
                            parsed_count += 1
                elif kind == "json":
                    json.loads(path.read_text())
                    parsed_count += 1
                else:
                    parsed_count += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    result_path = phase_dir / "parser-result.json"
    write_json(
        result_path,
        {
            "ok": bool(files) and not errors,
            "parsed_count": parsed_count,
            "rejected_count": len(errors),
            "errors": errors,
            "source_output_hashes": sorted(sha256_file(path) for path in files),
        },
    )
    # Without this the dispatcher logs a bare rc=1 and the reason sits in a file
    # nobody opens. A parse failure after a paid generation is the worst place to
    # hide a message.
    for message in errors:
        print(f"genie3 parse: {message}", file=sys.stderr)
    print(
        f"genie3 adapter: parsed={parsed_count} rejected={len(errors)} "
        f"phase={args.phase} result={result_path}"
    )
    return 0 if files and not errors else 1


def add_tool_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runner-protocol",
        choices=RUNNER_PROTOCOLS,
        default=DEFAULT_RUNNER_PROTOCOL,
        help=(
            "Where Genie3 runs, recorded on every row. auto reports modal when the shipped "
            "Modal mounts are present and local otherwise. Use local or modal to force one."
        ),
    )
    parser.add_argument(
        "--genie3-home",
        type=Path,
        default=None,
        help=(
            "Directory Genie3 runs in. It holds "
            f"{PRETRAINED_SUBDIR}/<version>/{MODEL_CONFIG_NAME}. Defaults to GENIE3_HOME."
        ),
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help=f"Model directory inside {PRETRAINED_SUBDIR}. Defaults to {DEFAULT_MODEL_VERSION}.",
    )
    parser.add_argument(
        "--genie3-command",
        default=DEFAULT_TOOL_COMMAND,
        help=f"Genie3 console script name or path. Defaults to {DEFAULT_TOOL_COMMAND}.",
    )
    parser.add_argument(
        "--genie3-module",
        default=None,
        help="Run the package as a module instead of the console script, such as genie3.cli.",
    )
    parser.add_argument(
        "--tool-python",
        default=None,
        help=(
            "Interpreter that runs --genie3-module. Defaults to the interpreter running this "
            "wrapper."
        ),
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("toolcheck", help="Probe the runtime without generating.")
    add_tool_arguments(check_parser)

    run_parser = subparsers.add_parser("run", help="Generate backbones for one phase.")
    add_tool_arguments(run_parser)
    run_parser.add_argument(
        "--stage",
        default=None,
        help="Stage ID in the resolved config. Recorded as stage_id on every manifest row.",
    )
    run_parser.add_argument(
        "--phase", required=True, help="Stage phase name, such as smoke or scale."
    )
    run_parser.add_argument(
        "--count", type=int, required=True, help="Number of backbones this phase generates."
    )
    run_parser.add_argument(
        "--attempt-dir", type=Path, required=True, help="Attempt directory that owns the outputs."
    )
    run_parser.add_argument(
        "--receipts-dir",
        type=Path,
        required=True,
        help=(
            "Directory holding the completed stage receipts. Read to find the target manifest "
            "unless --target-manifest names it."
        ),
    )
    run_parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Run artifact root. --target-manifest is resolved inside it and may not escape it.",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Resolved run config. The wrapper reads the model_revision this adapter's entry "
            "declares and records it on every manifest row."
        ),
    )
    run_parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help=(
            "Run plan, passed by the dispatcher. Accepted so the common stage argv works and "
            "not read, matching rfdiffusion_generator and proteinmpnn_designer."
        ),
    )
    run_parser.add_argument(
        "--target-manifest",
        type=Path,
        default=None,
        help="Published target manifest under the artifact root. Overrides the receipt lookup.",
    )
    run_parser.add_argument(
        "--target-stage-id",
        default=DEFAULT_TARGET_STAGE_ID,
        help=f"Stage ID of the upstream target preparer. Defaults to {DEFAULT_TARGET_STAGE_ID}.",
    )
    run_parser.add_argument(
        "--target-artifact-id",
        default=DEFAULT_TARGET_ARTIFACT_ID,
        help=f"Artifact ID of the target manifest. Defaults to {DEFAULT_TARGET_ARTIFACT_ID}.",
    )
    run_parser.add_argument(
        "--target-chain",
        default=None,
        help=(
            "Target chain the binder is designed against. Defaults to the design target chain of "
            "the manifest."
        ),
    )
    run_parser.add_argument(
        "--binder-chain",
        default=DEFAULT_BINDER_CHAIN,
        help=(
            "Chain ID the design pose publishes the binder as, which is the chain the sequence "
            f"designer is told to design. Defaults to {DEFAULT_BINDER_CHAIN}."
        ),
    )
    run_parser.add_argument(
        "--generator-binder-chain",
        default=None,
        help=(
            "Chain Genie3 wrote the binder into. Needed only when more than one chain of the "
            "output falls inside the binder length bounds, which is the case the residue counts "
            "cannot resolve. The named chain is still held against those bounds. Without this "
            "flag the binder is the one chain inside the bounds, and zero or several refuses."
        ),
    )
    run_parser.add_argument(
        "--binder-length-min",
        type=int,
        default=DEFAULT_BINDER_MINIMUM_LENGTH,
        help=(
            "Shortest binder the campaign accepts. Genie3 samples inside these bounds and the "
            f"published chain is held against them. Defaults to {DEFAULT_BINDER_MINIMUM_LENGTH}."
        ),
    )
    run_parser.add_argument(
        "--binder-length-max",
        type=int,
        default=DEFAULT_BINDER_MAXIMUM_LENGTH,
        help=f"Longest binder the campaign accepts. Defaults to {DEFAULT_BINDER_MAXIMUM_LENGTH}.",
    )
    run_parser.add_argument(
        "--generator-id",
        default=DEFAULT_GENERATOR_ID,
        help=f"Generator ID recorded on every row. Defaults to {DEFAULT_GENERATOR_ID}.",
    )
    run_parser.add_argument(
        "--problem-id",
        default=DEFAULT_PROBLEM_ID,
        help=f"Genie3 problem key and output directory name. Defaults to {DEFAULT_PROBLEM_ID}.",
    )
    run_parser.add_argument(
        "--experiment-name",
        default=None,
        help="Genie3 experiment name. Defaults to the generator ID and the phase.",
    )
    run_parser.add_argument(
        "--direction-scale",
        type=float,
        default=0.0,
        help="Genie3 sampler direction scale. A nonzero value steers toward the site residues.",
    )
    run_parser.add_argument(
        "--extended-site-residue",
        action="append",
        default=[],
        help=(
            "Extra site residue in CHAIN:NUMBER form for the extended slot. Repeat the flag. The "
            "resolved design residues of the target manifest fill the hotspot slot."
        ),
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Requested seed. Recorded on every row whether or not the tool receives it.",
    )
    run_parser.add_argument(
        "--seed-config-key",
        default=None,
        help=(
            "Dotted configuration key the wrapper writes the seed into, such as generation.seed. "
            "Pass the key your Genie3 build reads."
        ),
    )
    run_parser.add_argument(
        "--allow-unseeded",
        action="store_true",
        help="Run without a seed. Rows record tool_seed null and seed_delivered false.",
    )
    run_parser.add_argument(
        "--evaluation-folding-model",
        default=None,
        help=(
            "Write an evaluation section naming this folding model. The wrapper runs generate, "
            "which evaluates nothing, so the section stays out of the configuration by default."
        ),
    )
    run_parser.add_argument(
        "--evaluation-version",
        default=DEFAULT_EVALUATION_VERSION,
        help=(
            "Evaluation version written with --evaluation-folding-model. Defaults to "
            f"{DEFAULT_EVALUATION_VERSION}."
        ),
    )
    run_parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=f"Manifest path. Defaults to {DEFAULT_MANIFEST_NAME} in the phase directory.",
    )
    run_parser.add_argument(
        "--pose-subdir",
        default=DEFAULT_POSE_SUBDIR,
        help=(
            "Design pose directory inside the phase directory. Defaults to "
            f"{DEFAULT_POSE_SUBDIR}."
        ),
    )
    run_parser.add_argument(
        "--work-subdir",
        default=DEFAULT_WORK_SUBDIR,
        help=(
            "Dataset, configuration, and raw output directory inside the phase directory. "
            f"Defaults to {DEFAULT_WORK_SUBDIR}."
        ),
    )

    parse_parser = subparsers.add_parser("parse", help="Parse the outputs of one completed phase.")
    parse_parser.add_argument("--stage", required=True, help="Stage ID in the resolved config.")
    parse_parser.add_argument(
        "--phase", required=True, help="Stage phase name, such as smoke or scale."
    )
    # The parse subcommand accepts the dispatcher's whole common argv and reads
    # four of its eight flags. The other four are declared because every shipped
    # parse template supplies them, one for one with rfdiffusion_generator, whose
    # parse subcommand declares and ignores the same four. The output contract the
    # parser checks comes from --config and --stage, so a count, a receipt
    # directory, an artifact root, and a plan have nothing to add here.
    parse_parser.add_argument(
        "--count", type=int, default=1, help="Phase count. Accepted for the common argv, not read."
    )
    parse_parser.add_argument(
        "--attempt-dir", type=Path, required=True, help="Attempt directory that owns the outputs."
    )
    parse_parser.add_argument(
        "--receipts-dir",
        type=Path,
        default=None,
        help="Receipt directory. Accepted for the common argv, not read.",
    )
    parse_parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Artifact root. Accepted for the common argv, not read.",
    )
    parse_parser.add_argument("--config", type=Path, required=True, help="Resolved run config.")
    parse_parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Run plan. Accepted for the common argv, not read.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        if args.command == "toolcheck":
            return toolcheck(args)
        if args.command == "parse":
            return parse_outputs(args)
        return run(args)
    except AdapterError as exc:
        print(f"genie3 adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
