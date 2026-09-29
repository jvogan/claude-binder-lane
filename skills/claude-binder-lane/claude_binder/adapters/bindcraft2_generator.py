#!/usr/bin/env python3
"""Run a user-installed BindCraft 2 process inside a compute worker.

This wrapper fills the `bindcraft2-generator` slot. BindCraft 2 hallucinates a
binder through AlphaFold 2 on the ColabDesign framework and redraws its sequence
with ProteinMPNN, so one design carries a structure and a sequence. It is a
codesign generator and no sequence-design stage follows it.

This adapter launches a user-installed executable inside its compute worker.
The worker can run on the user's own machine or private cloud infrastructure.
The BindCraft2 Source-Available License (Hosting-Restricted) permits internal use
and restricts providing the software to third parties as a hosted service.
The catalogue records those terms, including the user-installed exception.

This module has no HTTP transport. Endpoint and credential arguments belong to
an outer provider transport, so this adapter rejects them before argparse.
That check describes the process interface; it does not decide whether a cloud
provider or a private endpoint is permitted. A user-controlled worker can run
this adapter through the provider's own job interface.

Outputs, written under the current attempt directory:

  <attempt>/<phase>/bindcraft2/settings.json       the composed campaign
  <attempt>/<phase>/bindcraft2/inputs/             the target structure it reads
  <attempt>/<phase>/bindcraft2/campaign/           BindCraft 2's project_folder
  <attempt>/<phase>/bindcraft2/run-receipt.json    argv, exit code, wall seconds
  <attempt>/<phase>/bindcraft2/parse-receipt.json  campaign provenance
  <attempt>/<phase>/poses/<candidate_id>.pdb       the design pose
  <attempt>/<phase>/sequences/<candidate_id>.fasta the designed sequence
  <attempt>/<phase>/candidate-manifest.jsonl       one row per accepted design

Seven properties of this route the wrapper states rather than hides.

**The vocabularies come from the installed copy, not from a table here.**
`bindcraft/__init__.py` is standard-library only behind a lazy `__getattr__`,
`cli.main` answers every `--list-*` flag before it touches the campaign, and
`--list-settings` reads the setting names out of the module source with `ast`.
No jax import, no GPU, no model load, no socket. Five free calls therefore give
the real target, modality, property, core and setting names, and this wrapper
validates against those rather than against names it carries. A name it cannot
find is refused with the nearest available spelling.

**A vocabulary can come back empty, and that is a real diagnostic.** The presets
live at `Path(cli.__file__).parent.parent / 'settings'`, a directory beside the
package rather than inside it, and upstream's `pyproject.toml` declares only the
ProteinMPNN weights as package data. `install.sh` installs with `pip install -e .`
so an ordinary installation keeps the checkout and the presets resolve. A copy
installed from a built wheel has no `settings` tree and prints nothing, so this
wrapper refuses an empty vocabulary by name instead of accepting every spelling.

**What the pre-flight check buys.** BindCraft 2 refuses an unknown modality and
an unknown setting key when it loads the settings file, which is after `design`
has imported its campaign stack. An unrecognized property flag is worse: the CLI
reads it as a second settings path, prints its bare usage text and exits 2, with
nothing naming the flag. `--core default` and `--core reference` are worse again,
because `requested_core_profiles` drops both silently and the run proceeds
without the profile. Every one of those is caught here, by name, before the
subprocess starts. A modality whose shipped preset designs more than one binder
chain is caught there too: BindCraft 2 runs it, and this stage publishes one
sequence per candidate, so the campaign would be refused when it is parsed with
every one of its GPU hours already spent.

**`--core benchmark` is the default.** It sets `campaign_seed: 0`,
`autotune: false` and `desperation: false`, which is the reproducible mode a
campaign layer needs in order to repeat a run. A preset only supplies a default,
so a `campaign_seed` written into the composed settings file wins over it. The
receipt records the seed this stage asked for and the seed the core profile pins,
because those two are not always the same number.

**AlphaFold 2 parameters are never downloaded here.** Twelve ProteinMPNN and
HyperMPNN checkpoints ship inside the package. The AlphaFold parameters do not:
`bindcraft/model_weights.py` fetches a 5.3 GB archive on first use. `toolcheck`
reports a missing parameter set as a named missing input and says every directory
it looked in. Five gigabytes of network on a readiness check is not a readiness
check.

**The written chain letters are BindCraft 2's, not the campaign's.** The
`binder_chain` setting selects which prepared binder chain custom objectives
address; it does not rename an output chain. So this wrapper does not set it from
the campaign's binder chain, and the pose is written through unrelabelled. Each
accepted mmCIF carries a `bindcraft` metadata category naming its binder and
target chain letters, and the parser records what it reads there.

**The cost basis is unpriced.** BindCraft 2 runs on the scientist's own GPU on
their own account, this package has measured no run of it, and no timing or cost
basis for it exists here. The receipt says `unpriced` and carries no number.

Acceptance is BindCraft 2's own computational verdict on its own filters. Every
published row keeps `status` `generated`, which is this lane's word for a
candidate nothing in this lane has screened. No row of this set is a promoted or
accepted binder.

Subcommands:

  toolcheck  Report this adapter's own readiness. Runs no design, loads no model,
             opens no socket, downloads nothing and costs nothing.
  run        Compose the settings file, validate every name, execute one
             campaign, and record exactly what ran.
  parse      Read the campaign folder and publish poses, sequences, manifest rows
             and the campaign's own provenance.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from claude_binder.adapters.candidate_lineage import backbone_lineage


ADAPTER_ID = "bindcraft2-generator"
DEFAULT_GENERATOR_ID = "bindcraft2"
TOOL_LABEL = "bindcraft2"
ROUTE_ID = "bindcraft2-local-cli"
GENERATOR_MODE = "sequence-structure-codesign"
CANDIDATE_STATUS = "generated"
COST_BASIS = "unpriced"

# TODO(evidence): no publication, preprint or DOI for BindCraft 2 is stated
# anywhere in the upstream checkout this adapter was written against, so the
# catalogue row for this tool has no citation to carry. Settled by: a reference
# in references/tool-licences.md or the upstream README naming the paper, after
# which the catalogue's evidence field takes a value instead of __REQUIRED__.
#
# TODO(evidence): no timing or cost basis for BindCraft 2 exists in this package
# on any hardware, so nothing here can estimate what one campaign costs. The
# receipt records `unpriced` rather than a plausible number. Settled by: one
# authorized campaign on the scientist's own GPU, with its measured wall time
# and accepted-design count recorded in claude_binder/data/model-roster.json
# the way the RFdiffusion3 entry is.
#
# TODO(evidence): no BindCraft 2 campaign has run under this adapter's contract,
# so the exact mmCIF serialization `protein.write_structure` emits through
# biotite is unverified here, as is whether the `bindcraft` provenance category
# is written as key-value lines or as a loop. The parser reads the stamp when it
# is in key-value form and records nulls otherwise, and it refuses rather than
# guesses when the coordinate conversion fails. Settled by: one committed
# accepted complex from a real campaign, checked into
# claude_binder/tests/fixtures.
#
# TODO(evidence): BindCraft 2 records no per-trajectory seed in any output. The
# seed is drawn from `campaign_seed` inside `protein_preparation`, and neither
# the accepted table, the trajectory table nor the structure stamp carries it,
# so every row's `generator_seed` is the campaign seed and `trajectory_seed` is
# null. Settled by: an upstream change that records the drawn seed, or a
# documented derivation from the claimed attempt number.

# The three routes `run` and `toolcheck` resolve the executable through, in this
# order. The receipt records which one answered.
EXECUTABLE_ARGUMENT = "--bindcraft-executable"
EXECUTABLE_ENVIRONMENT_KEY = "CLAUDE_BINDER_BINDCRAFT2"
DEFAULT_EXECUTABLE_NAME = "bindcraft"

DEFAULT_WORK_SUBDIR = "bindcraft2"
DEFAULT_CAMPAIGN_SUBDIR = "campaign"
DEFAULT_INPUT_SUBDIR = "inputs"
DEFAULT_POSE_SUBDIR = "poses"
DEFAULT_SEQUENCE_SUBDIR = "sequences"
DEFAULT_MANIFEST_NAME = "candidate-manifest.jsonl"
DEFAULT_PARSER_RESULT_NAME = "parser-result.json"
SETTINGS_NAME = "settings.json"
RUN_RECEIPT_NAME = "run-receipt.json"
PARSE_RECEIPT_NAME = "parse-receipt.json"
TOOLCHECK_REPORT_NAME = "toolcheck.json"
RUN_RECEIPT_FORMAT = "bindcraft2-generator-run-receipt-v1"
PARSE_RECEIPT_FORMAT = "bindcraft2-generator-parse-receipt-v1"
TOOLCHECK_REPORT_FORMAT = "bindcraft2-generator-toolcheck-v1"

DEFAULT_TIMEOUT_SECONDS = 86400
DEFAULT_CORE_PROFILE = "benchmark"

# `bindcraft/cli.py` answers each of these before it imports the campaign, and
# each prints one name a line. `--list-core` filters `default` and `reference`,
# which is the same pair `settings.requested_core_profiles` drops, so the listing
# is the complete set of core profiles a run can actually apply.
LIST_FLAGS = {
    "targets": "--list-targets",
    "modalities": "--list-modalities",
    "properties": "--list-properties",
    "core": "--list-core",
    "settings": "--list-settings",
}
VOCABULARY_NAMES = tuple(LIST_FLAGS)
LISTING_TIMEOUT_SECONDS = 120

# `bindcraft/protein.py` line 277. `read_protein_source` accepts a target file
# with one of these suffixes; anything else it treats as inline text.
TARGET_PATH_SUFFIXES = (".pdb", ".cif", ".mmcif", ".ent", ".fasta", ".fa", ".faa")
# `bindcraft/settings.py` line 38. A target entry accepts only these keys.
TARGET_SETTING_NAMES = ("name", "target_path", "chains", "hotspots", "coldspots", "weight", "objective")

# `bindcraft/model_weights.py`. The seven AlphaFold checkpoints a campaign loads,
# the size floor a whole checkpoint clears, and the archive that is never fetched
# from here.
CAMPAIGN_MODELS = (
    "model_1_multimer_v3",
    "model_2_multimer_v3",
    "model_3_multimer_v3",
    "model_4_multimer_v3",
    "model_5_multimer_v3",
    "model_1_ptm",
    "model_2_ptm",
)
ALPHAFOLD_CHECKPOINT_FLOOR_BYTES = 100 << 20
ALPHAFOLD_PARAMETER_GIGABYTES = 5.3
ALPHAFOLD_PARAMETER_ENVIRONMENT_KEY = "BINDCRAFT_AF2_PARAMS"
WEIGHT_CACHE_ENVIRONMENT_KEY = "BINDCRAFT_WEIGHTS"
MPNN_WEIGHTS_ENVIRONMENT_KEY = "BINDCRAFT_MPNN_WEIGHTS"
# The twelve checkpoints that ship inside the package, each 6,681,030 bytes in
# the v1.0.0 source tree read for this adapter. The size is recorded, not
# enforced, because upstream's own check is the 1 MiB floor below.
MPNN_VARIANT_DIRECTORIES = ("weights_negative", "weights_neutral", "weights_positive")
MPNN_MODEL_NAMES = ("v_48_002", "v_48_010", "v_48_020", "v_48_030")
SHIPPED_MPNN_CHECKPOINT_COUNT = len(MPNN_VARIANT_DIRECTORIES) * len(MPNN_MODEL_NAMES)
MPNN_CHECKPOINT_FLOOR_BYTES = 1 << 20
DEFAULT_MPNN_MODEL = "v_48_020"
DEFAULT_MPNN_VARIANT = "negative"

# `bindcraft/campaign_output.py`. The stage folders, the accepted table, and the
# provenance file.
RANKED_STAGE_DIRECTORY = "3_Ranked"
RANKED_TABLE_NAME = "!_Ranked.csv"
TRAJECTORY_STAGE_DIRECTORY = "1_Trajectories"
TRAJECTORY_TABLE_NAME = "!_Trajectories.csv"
REFOLDED_STAGE_DIRECTORY = "2_Refolded"
REFOLDED_TABLE_NAME = "!_Refolded.csv"
CAMPAIGN_METADATA_NAME = "campaign_metadata.json"
SUMMARY_NAME = "summary.csv"
MONOMER_SUFFIX = "_monomer.cif"
TARGET_VALUE_SEPARATOR = ";"
RANKING_METRIC = "i_pDAE"

# Columns the parser reads out of every accepted row. A header missing one of
# these is refused by name rather than defaulted, because a zero in a ranking
# score is a claim and a blank is not.
REQUIRED_RANKED_COLUMNS = ("rank", "design", "Binder_Sequence", RANKING_METRIC)

# `i_pDAE` is the criterion BindCraft 2 selected these candidates on, and this
# lane screens on `ipsae_min`. The two are close relatives and they are not the
# same statistic, so every row says so rather than leaving a reader to assume
# either independence or equivalence.
#
# Read from `bindcraft/filters.py` lines 117 to 169 against
# `adapters/binder_metrics.py` lines 854 to 890 and 1047:
#
#   Same. The per-residue kernel is 1/(1 + (PAE/d0)^2), averaged over the
#   selected partners of residue i. The d0 fit is 1.24*cbrt(n-15)-1.8 floored at
#   1.0. Upstream also clamps n at 19 before the cube root, which changes no
#   value, because the fit only clears the 1.0 floor above n of about 26.
#
#   Different. Upstream selects pairs by an 8 Angstrom C-alpha contact mask.
#   This lane selects them by PAE below 10 Angstrom, which is this package's
#   own configured value: `IPSAE_INTERFACE_CUTOFF_ANGSTROM` cites the
#   campaign's reproduction reference for it, not Dunbrack's own default.
#
#   Opposite. Upstream returns the maximum over both directions and all
#   residues. This lane screens on `ipsae_min`, the smaller of the two
#   directional maxima. Selection took the optimistic reading and the screen
#   takes the pessimistic one, so an interface that is confident one way and
#   poor the other scores high upstream and low here.
SELECTION_METRIC = RANKING_METRIC
LANE_SCREEN_METRIC = "ipsae_min"
SELECTION_METRIC_NOTE = (
    "i_pDAE is the criterion BindCraft 2 selected this candidate on. It shares its kernel and "
    "its d0 fit with this lane's ipsae_min, selects interface pairs by an 8 Angstrom C-alpha "
    "contact rather than by PAE below 10 Angstrom, and takes the maximum over both chain "
    "directions where ipsae_min takes the minimum. A screen score is therefore neither an "
    "independent measurement of this number nor a reading of the same one, and a pass rate "
    "measured on these rows is a post-selection rate that is not comparable to one measured on "
    "a generator that optimized nothing of this form."
)
# Columns the parser carries onto the row when the table holds them. These are
# BindCraft 2's own spellings; nothing here computes a metric or a threshold.
OPTIONAL_RANKED_COLUMNS = (
    "length",
    "hash",
    "trajectory",
    "outcome",
    "failed_filters",
    "terminated",
    "targets",
    "target_weights",
    "i_pTM",
    "i_pAE",
    "pLDDT",
    "pTM",
    "Unbound_Binder_pLDDT",
    "Target_pLDDT",
    "Interface_Residues",
    "Interface_Binder_Residues",
    "Interface_Target_Residues",
    "Binder_RMSD",
    "Hotspot_Contact_Fraction",
    "Binder_Length",
)
# `rank.TEXT_COLUMNS` line 14. These columns hold names, sequences and residue
# lists rather than readings, so a semicolon in one of them is not a multi-target
# measurement and must not be reported as one.
TEXT_RANKED_COLUMNS = (
    "rank",
    "design",
    "terminated",
    "failed_filters",
    "Binder_Sequence",
    "Interface_Binder_Residues",
    "Interface_Target_Residues",
    "targets",
    "target_weights",
)

# A residue ID in the target manifest reads CHAIN:NUMBER with an optional
# insertion code, which is what `target_prep_adapter.resolve_site_residues`
# writes.
RESIDUE_ID_RE = re.compile(r"^([A-Za-z0-9]+):(\d+)([A-Za-z]?)$")
SAFE_DESIGN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

#: Settings `compose_settings` writes for itself. A `--set` on one of these is
#: refused rather than forwarded. BindCraft 2 applies `--set` over the composed
#: file, so an override here would move the campaign while the run receipt and
#: the published rows still record the composed value. That is falsified
#: provenance rather than a configuration choice, and `targets` is the one that
#: matters most: it would design against a different protein than the row names.
ADAPTER_OWNED_SETTINGS = (
    "targets",
    "project_folder",
    "binder_lengths",
    "number_of_final_designs",
    "resume",
    "max_trajectories",
    "campaign_seed",
    "binder_name",
    "campaign_name",
)

REQUIRED_TARGET_MANIFEST_FIELDS = (
    "target_id",
    "target_sha256",
    "residue_map_sha256",
    "source_structure_path",
    "normalized_structure_path",
    "design_target_chain_id",
    "chain_ids",
)

# An argument whose name carries one of these words is a hosted-route argument,
# and this route has none. The check runs on raw argv before argparse, so the
# refusal names the argument instead of printing a usage block. The match is on
# whole hyphen-separated words rather than on substrings, because `--modality` is
# this adapter's own flag and it carries `modal` inside it.
HOSTED_ARGUMENT_WORDS = frozenset(
    {
        "fal",
        "endpoint",
        "endpoints",
        "url",
        "uri",
        "credential",
        "credentials",
        "apikey",
        "token",
        "secret",
        "provider",
        "deployment",
        "webhook",
        "http",
        "https",
        "modal",
        "runpod",
        "lambda",
        "api-key",
        "base-url",
        "acknowledge-cost",
    }
)
HOSTED_VALUE_MARKER = "://"


class AdapterError(RuntimeError):
    """A generator input, an installed vocabulary, or a campaign output is invalid."""


# ----------------------------------------------------------------------------
# Small shared plumbing. Standard library only, argv in, files out.
# ----------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 of bytes."""
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path, label: str) -> Any:
    """Load one JSON document, naming it in any failure."""
    if not path.is_file():
        raise AdapterError(f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"{label} is invalid: {path}: {exc}") from exc


def is_own_output(path: Path, required: frozenset[str], expected_format: str) -> bool:
    """True when `path` already holds a document this adapter wrote.

    Transcribed from `canary_runner._is_own_output`. A zero-byte file counts,
    because a run killed mid-write leaves one and its retry must not need a flag.
    Anything unreadable counts as foreign.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return False
    if not raw.strip():
        return True
    try:
        existing = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(existing, dict):
        return False
    if existing.get("format") == expected_format:
        return True
    return required <= set(existing)


def refuse_occupied_output(
    path: Path, *, flag: str, required: frozenset[str], expected_format: str, replace: bool
) -> None:
    """Refuse an output path that already holds a file this adapter did not write.

    Transcribed from `canary_runner._refuse_occupied_output`, and called before
    the subprocess rather than at the write, so a refusal never arrives after a
    campaign has spent GPU hours. Re-running this stage onto its own previous
    output is the documented flow and needs no flag.
    """
    if path.is_symlink():
        raise AdapterError(
            f"{flag} is an output path and is overwritten, and {path} is a symbolic link. "
            "Writing through it would replace the file it points at. Name the real path."
        )
    if not path.exists():
        return
    if not path.is_file():
        raise AdapterError(
            f"{flag} is an output path and is overwritten, and {path} is not a regular file. "
            "Name a path that does not exist, or a file this adapter may replace."
        )
    if replace or is_own_output(path, required, expected_format):
        return
    raise AdapterError(
        f"{flag} is an output path and is overwritten, and {path} already holds a file this "
        f"adapter did not write. Name a path that does not exist, or pass --replace to "
        f"overwrite it."
    )


def write_text_output(
    path: Path,
    payload: str,
    *,
    flag: str,
    required: frozenset[str],
    expected_format: str,
    replace: bool,
) -> None:
    """Write one text output, refusing to destroy a file this adapter did not write.

    `O_NOFOLLOW` closes the window the guard above cannot: a symbolic link
    swapped into place between the check and the open would otherwise be
    followed. The write follows `canary_runner._write_output_json`.
    """
    refuse_occupied_output(
        path, flag=flag, required=required, expected_format=expected_format, replace=replace
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    try:
        handle = os.open(path, flags, 0o644)
    except OSError as error:
        raise AdapterError(f"{flag} could not be written at {path}: {error}") from error
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(payload)


def write_json_output(
    path: Path,
    document: dict[str, Any],
    *,
    flag: str,
    required: frozenset[str],
    expected_format: str,
    replace: bool,
) -> None:
    """Write one JSON output through the occupied-output guard."""
    write_text_output(
        path,
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        flag=flag,
        required=required,
        expected_format=expected_format,
        replace=replace,
    )


def write_json(path: Path, document: Any) -> None:
    """Write one JSON file this adapter owns outright."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the candidate manifest, refusing an empty one."""
    if not rows:
        raise AdapterError(f"refusing to write an empty candidate manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def nearest_names(name: str, available: list[str], count: int = 3) -> str:
    """Return the closest available spellings, for a refusal that helps."""
    close = difflib.get_close_matches(name, available, n=count, cutoff=0.4)
    if close:
        return "closest available: " + ", ".join(close)
    shown = ", ".join(sorted(available)[:12])
    return f"available: {shown}" + (", ..." if len(available) > 12 else "")


# ----------------------------------------------------------------------------
# Reject transport arguments unsupported by this process interface.
# ----------------------------------------------------------------------------


def refuse_hosted_arguments(argv: list[str]) -> None:
    """Reject HTTP transport arguments before parsing the process command.

    Provider selection belongs to the outer worker transport. This adapter
    accepts a local executable inside that worker, including a cloud worker.
    """
    for token in argv:
        if not isinstance(token, str):
            continue
        if HOSTED_VALUE_MARKER in token:
            raise AdapterError(
                f"this adapter refuses the argument {token!r} because it carries "
                f"{HOSTED_VALUE_MARKER!r}. This adapter accepts a local process executable. "
                "Configure the provider transport outside this adapter and run the "
                "executable inside its compute worker."
            )
        if not token.startswith("--"):
            continue
        name = token.split("=", 1)[0]
        stripped = name.lower().lstrip("-")
        words = {stripped, *stripped.split("-")}
        matched = next(iter(sorted(words & HOSTED_ARGUMENT_WORDS)), None)
        if matched is None:
            continue
        raise AdapterError(
            f"this adapter refuses the argument {name} because {matched!r} names a "
            "transport option unsupported by this local process adapter. Configure the "
            "provider transport outside this adapter and run the installed executable "
            "inside its compute worker."
        )


# ----------------------------------------------------------------------------
# The installed copy: the executable, its vocabularies, and its weights.
# ----------------------------------------------------------------------------


def resolve_executable(value: str | None) -> tuple[Path, str]:
    """Return the BindCraft 2 executable and which of three routes resolved it."""
    if value:
        path = Path(value).expanduser()
        source = EXECUTABLE_ARGUMENT
    else:
        environment_value = os.environ.get(EXECUTABLE_ENVIRONMENT_KEY, "").strip()
        if environment_value:
            path = Path(environment_value).expanduser()
            source = EXECUTABLE_ENVIRONMENT_KEY
        else:
            found = shutil.which(DEFAULT_EXECUTABLE_NAME)
            if not found:
                raise AdapterError(
                    f"BindCraft 2 executable {DEFAULT_EXECUTABLE_NAME!r} is not on PATH. Pass "
                    f"{EXECUTABLE_ARGUMENT}, set {EXECUTABLE_ENVIRONMENT_KEY}, or install "
                    "BindCraft 2 on this machine. It runs on Linux with an NVIDIA GPU; upstream "
                    "install.sh refuses a CPU installation outright."
                )
            path = Path(found)
            source = "PATH"
    resolved = path.resolve()
    if not resolved.is_file():
        raise AdapterError(f"BindCraft 2 executable is not an existing file: {resolved}")
    return resolved, source


def read_listing(executable: Path, flag: str, name: str) -> list[str]:
    """Return one `--list-*` vocabulary from the installed copy.

    Every one of these calls is free. `cli.main` answers the flag before it
    imports the campaign, `bindcraft/__init__.py` is standard-library only, and
    `--list-settings` reads the names out of the module source with `ast`. No
    jax import, no GPU, no model load, no network.
    """
    argv = [str(executable), "design", flag]
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=LISTING_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterError(
            f"{' '.join(argv)} did not answer within {LISTING_TIMEOUT_SECONDS} seconds"
        ) from exc
    except OSError as exc:
        raise AdapterError(f"{' '.join(argv)} could not be run: {exc}") from exc
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise AdapterError(
            f"{' '.join(argv)} exited {completed.returncode}: "
            f"{tail[-1] if tail else 'no output'}"[:500]
        )
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not names:
        raise AdapterError(
            f"{' '.join(argv)} printed no {name}. BindCraft 2 reads its presets from a "
            "'settings' directory beside the package, and upstream declares only the "
            "ProteinMPNN weights as package data, so a copy installed from a built wheel has "
            "no presets. Install it the way upstream install.sh does, with pip install -e "
            "against the source tree."
        )
    return names


def read_vocabularies(executable: Path) -> dict[str, list[str]]:
    """Return the five vocabularies the installed copy prints."""
    return {
        name: read_listing(executable, flag, name) for name, flag in sorted(LIST_FLAGS.items())
    }


def interpreter_from_shebang(executable: Path) -> Path | None:
    """Return the interpreter a console script names on its first line.

    A pip-installed console script starts `#!<interpreter>`. A wrapper written in
    a shell, which a container or a conda shim may be, names a shell instead, and
    this returns None rather than guessing.
    """
    try:
        with executable.open("rb") as handle:
            first = handle.readline(4096)
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    try:
        text = first[2:].decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    if not text:
        return None
    candidate = Path(text.split()[0])
    if candidate.name.startswith("python") and candidate.is_file():
        return candidate
    return None


def resolve_package_directory(executable: Path, explicit: str | None) -> tuple[Path | None, str]:
    """Return the installed `bindcraft` package directory, and how it was found.

    The shipped weights and the core profiles live relative to this directory, so
    a readiness report that cannot find it says so instead of reporting an
    absence it never looked for. The probe imports `bindcraft/__init__.py`, which
    is standard-library only, and prints one path.
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise AdapterError(f"--bindcraft-package-dir is not a directory: {path}")
        return path, "--bindcraft-package-dir"
    interpreter = interpreter_from_shebang(executable)
    if interpreter is None:
        return None, "unresolved: the executable names no Python interpreter on its first line"
    argv = [
        str(interpreter),
        "-c",
        "import bindcraft, pathlib; print(pathlib.Path(bindcraft.__file__).resolve().parent)",
    ]
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=LISTING_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"unresolved: the interpreter probe failed: {exc}"
    printed = completed.stdout.strip()
    if completed.returncode != 0 or not printed:
        return None, f"unresolved: the interpreter probe exited {completed.returncode}"
    path = Path(printed)
    if not path.is_dir():
        return None, f"unresolved: the interpreter probe printed a path that is not a directory: {path}"
    return path.resolve(), f"interpreter probe through {interpreter}"


def alphafold_parameter_roots(package_directory: Path | None) -> list[tuple[str, Path]]:
    """Return the directories BindCraft 2 looks in for AlphaFold parameters, in order.

    Transcribed from `model_weights.alphafold_parameters` and
    `model_weights.weight_cache`. The download branch that follows them upstream
    has no counterpart here.
    """
    roots: list[tuple[str, Path]] = []
    configured = os.environ.get(ALPHAFOLD_PARAMETER_ENVIRONMENT_KEY, "").strip()
    if configured:
        roots.append((ALPHAFOLD_PARAMETER_ENVIRONMENT_KEY, Path(configured).expanduser()))
    if package_directory is not None:
        roots.append(("shipped weights", package_directory / "weights" / "alphafold"))
    cache = os.environ.get(WEIGHT_CACHE_ENVIRONMENT_KEY, "").strip()
    if cache:
        roots.append((WEIGHT_CACHE_ENVIRONMENT_KEY, Path(cache).expanduser() / "alphafold"))
    else:
        home_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
        base = Path(home_cache).expanduser() if home_cache else Path.home() / ".cache"
        roots.append(("weight cache", base / "bindcraft" / "alphafold"))
    return roots


def alphafold_parameter_file(root: Path, model_name: str) -> Path | None:
    """Return the parameter file for one model under one root, or None.

    The four candidate spellings are `model_weights.alphafold_parameter_file`'s.
    """
    candidates = (
        root / "params" / f"params_{model_name}.npz",
        root / f"params_{model_name}.npz",
        root / "params" / f"{model_name}.npz",
        root / f"{model_name}.npz",
    )
    return next((path for path in candidates if path.is_file()), None)


def alphafold_report(package_directory: Path | None) -> dict[str, Any]:
    """Report which of the seven AlphaFold checkpoints are on this machine.

    Nothing here downloads. Upstream fetches a 5.3 GB archive on first use, and a
    readiness check that spends that is not a readiness check.
    """
    roots = alphafold_parameter_roots(package_directory)
    models: list[dict[str, Any]] = []
    for model_name in CAMPAIGN_MODELS:
        found: Path | None = None
        found_root: str | None = None
        for label, root in roots:
            candidate = alphafold_parameter_file(root, model_name)
            if candidate is not None:
                found, found_root = candidate, label
                break
        size = found.stat().st_size if found is not None else 0
        models.append(
            {
                "model": model_name,
                "file_name": f"params_{model_name}.npz",
                "path": str(found) if found is not None else None,
                "root": found_root,
                "size_bytes": size,
                "whole": found is not None and size >= ALPHAFOLD_CHECKPOINT_FLOOR_BYTES,
            }
        )
    missing = [record["file_name"] for record in models if not record["whole"]]
    return {
        "required_count": len(CAMPAIGN_MODELS),
        "present_count": len(CAMPAIGN_MODELS) - len(missing),
        "missing": missing,
        "searched": [{"source": label, "directory": str(root)} for label, root in roots],
        "checkpoint_floor_bytes": ALPHAFOLD_CHECKPOINT_FLOOR_BYTES,
        "archive_gigabytes": ALPHAFOLD_PARAMETER_GIGABYTES,
        "downloaded_by_this_adapter": False,
        "models": models,
        "ready": not missing,
    }


def proteinmpnn_root(package_directory: Path | None) -> tuple[Path | None, str]:
    """Return the directory holding the three ProteinMPNN variant folders.

    `model_weights.proteinmpnn_weights` returns `BINDCRAFT_MPNN_WEIGHTS` when it
    is set and the shipped `weights/proteinmpnn/weights_neutral` otherwise, and
    `mpnn_variant_directory` reaches a sibling variant from that value's parent.
    """
    configured = os.environ.get(MPNN_WEIGHTS_ENVIRONMENT_KEY, "").strip()
    if configured:
        return Path(configured).expanduser().parent, MPNN_WEIGHTS_ENVIRONMENT_KEY
    if package_directory is not None:
        return package_directory / "weights" / "proteinmpnn", "shipped weights"
    return None, "unresolved: the installed package directory was not found"


def proteinmpnn_report(package_directory: Path | None) -> dict[str, Any]:
    """Report the shipped ProteinMPNN and HyperMPNN checkpoints, if they are visible."""
    root, source = proteinmpnn_root(package_directory)
    if root is None:
        return {
            "expected_count": SHIPPED_MPNN_CHECKPOINT_COUNT,
            "present_count": 0,
            "root": None,
            "root_source": source,
            "default_model": DEFAULT_MPNN_MODEL,
            "default_variant": DEFAULT_MPNN_VARIANT,
            "checkpoints": [],
            "visible": False,
        }
    checkpoints: list[dict[str, Any]] = []
    for variant in MPNN_VARIANT_DIRECTORIES:
        for model_name in MPNN_MODEL_NAMES:
            path = root / variant / f"{model_name}.npz"
            size = path.stat().st_size if path.is_file() else 0
            checkpoints.append(
                {
                    "variant": variant,
                    "model": model_name,
                    "path": str(path),
                    "size_bytes": size,
                    "whole": size >= MPNN_CHECKPOINT_FLOOR_BYTES,
                }
            )
    present = [record for record in checkpoints if record["whole"]]
    return {
        "expected_count": SHIPPED_MPNN_CHECKPOINT_COUNT,
        "present_count": len(present),
        "root": str(root),
        "root_source": source,
        "checkpoint_floor_bytes": MPNN_CHECKPOINT_FLOOR_BYTES,
        "default_model": DEFAULT_MPNN_MODEL,
        "default_variant": DEFAULT_MPNN_VARIANT,
        "checkpoints": checkpoints,
        "visible": bool(present),
    }


def core_profile_seed(package_directory: Path | None, core: str) -> tuple[int | None, str]:
    """Return the `campaign_seed` the named core profile pins, and where it was read.

    `cli.CAMPAIGN_PRESETS` is a `settings` directory beside the package, so the
    profile is readable once the package directory is known. A profile only
    supplies a default and the composed settings file wins over it, so both
    numbers go in the receipt.
    """
    if package_directory is None:
        return None, "unresolved: the installed package directory was not found"
    path = package_directory.parent / "settings" / "core" / f"{core}.json"
    if not path.is_file():
        return None, f"unresolved: no core profile file at {path}"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"unresolved: {path} is unreadable: {exc}"
    value = document.get("campaign_seed")
    if isinstance(value, bool) or not isinstance(value, int):
        return None, f"the core profile at {path} pins no integer campaign_seed"
    return value, str(path)


# `bindcraft/protein.py` line 645. `_EDIT_RE` reads a `mutate_positions` entry as
# a chain label followed by a residue number, and `scaffold_edit_flags` walks the
# distinct labels it finds, so the edit string names the chains the scaffold
# carries. Only the label is read here, so this transcribes that prefix.
EDIT_CHAIN_RE = re.compile(r"(?P<chain>[A-Za-z]+)(?P<start>\d+)")


def preset_binder_chain_count(preset: dict) -> tuple[int, str]:
    """Return how many binder chains one modality preset implies, and the fields read.

    `bindcraft/settings.py` lines 94 to 99. `resolve_binder_chains` names one
    binder chain per `copies` and one per chain of a multi-chain
    `binder_scaffold`, takes the larger of the two and refuses both above one at
    once. A preset that sets neither designs one chain. Line 245 of the same file
    calls the condition on `copies` the campaign feature `multi-chain binder`.

    The scaffold's own chain count is upstream's authority and it lives in the
    shipped `.cif`, which this module does not open. The preset's
    `mutate_positions` labels the scaffold chains it edits, and those labels are
    the scaffold's chains for all four scaffolds v1.0.1 ships, HEAD
    `5342aefa18dedad653f7a5f6dbee1e566ca24d8f`: `scaffolds/ARP.cif` and
    `scaffolds/VHH.cif` hold chain A and their presets label A, `scaffolds/Fab.cif`
    and `scaffolds/scFv.cif` hold chains H and L and their presets label H and L.
    So the labels stand in for the count.

    The count is read from those two fields rather than from the modality name.
    `multidomain` sets `n_domains` to 2 and no `copies` and no scaffold, which is
    two domains on one chain, so it counts as one binder chain here.
    """
    copies = preset.get("copies")
    if isinstance(copies, bool) or not isinstance(copies, int) or copies < 1:
        copies = 1
    labels = list(
        dict.fromkeys(
            match.group("chain")
            for match in EDIT_CHAIN_RE.finditer(str(preset.get("mutate_positions") or ""))
        )
    )
    evidence = f"copies {copies}, mutate_positions chains {', '.join(labels) or 'none'}"
    return max(copies, len(labels), 1), evidence


def modality_binder_chains(package_directory: Path | None, modality: str) -> tuple[int | None, str]:
    """Return the binder chains the named modality's preset implies, and where it was read.

    `settings.CAMPAIGN_PRESETS` is the `settings` directory beside the package the
    core profiles come from, and `settings.read_preset` reads
    `settings/modality/<name>.json` out of it. A preset this adapter cannot read
    leaves the count unknown rather than assuming one chain.
    """
    if package_directory is None:
        return None, "unresolved: the installed package directory was not found"
    path = package_directory.parent / "settings" / "modality" / f"{modality}.json"
    if not path.is_file():
        return None, f"unresolved: no modality preset file at {path}"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"unresolved: {path} is unreadable: {exc}"
    if not isinstance(document, dict):
        return None, f"unresolved: the modality preset at {path} is not an object"
    count, evidence = preset_binder_chain_count(document)
    return count, f"{path}: {evidence}"


def refuse_multi_chain_modality(package_directory: Path | None, modality: str) -> str | None:
    """Refuse a modality whose shipped preset designs more than one binder chain.

    `binder_sequence` refuses a `/`-separated `Binder_Sequence` when the campaign
    is parsed, which is every GPU hour of the campaign after the mistake. The
    preset states the same fact before the subprocess starts, so the modality is
    refused here and the parse-time check stays as the backstop for a campaign
    composed outside this stage.

    Returns the note to print when the preset was not readable. An unreadable
    preset is not evidence of a single chain, and refusing every campaign on it
    would refuse the single-chain ones too.
    """
    chains, source = modality_binder_chains(package_directory, modality)
    if chains is None:
        return f"{modality} ({source})"
    if chains > 1:
        raise AdapterError(
            f"--modality {modality} designs a binder of {chains} chains, and its shipped "
            f"preset states it: {source}. A campaign records a multi-chain binder as one "
            "Binder_Sequence with '/' between its chains, and this stage publishes one "
            "sequence per candidate, so every accepted design would be refused when the "
            "campaign is parsed, after the campaign has spent its GPU hours. The lane has no "
            "rule for publishing a binder that carries several chains: whether such a "
            "candidate is one row with several sequence files or several rows, recorded in "
            "the candidate manifest schema. Run a single-chain modality, or settle that rule "
            "first."
        )
    return None


# ----------------------------------------------------------------------------
# The target this stage designs against.
# ----------------------------------------------------------------------------


def completed_receipt_file(receipts_dir: Path, stage_id: str, artifact_id: str) -> Path:
    """Return the one file an upstream stage receipt publishes for this artifact."""
    receipt_path = receipts_dir.expanduser() / f"{stage_id}.json"
    receipt = load_json(receipt_path, "target stage receipt")
    if not isinstance(receipt, dict):
        raise AdapterError(f"target stage receipt is not an object: {receipt_path}")
    paths = [
        item.get("path")
        for artifact in receipt.get("output_manifest", {}).get("artifacts", [])
        if isinstance(artifact, dict) and artifact.get("artifact_id") == artifact_id
        for item in artifact.get("files", [])
        if isinstance(item, dict)
    ]
    if not paths or not isinstance(paths[0], str):
        raise AdapterError(
            f"target stage receipt carries no {artifact_id} file: {receipt_path}"
        )
    return Path(paths[0]).expanduser().resolve()


def load_target_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    """Return the target manifest the target-preparer stage published, and its path."""
    if getattr(args, "target_manifest", None) is not None:
        path = Path(args.target_manifest).expanduser().resolve()
    else:
        path = completed_receipt_file(
            args.receipts_dir, args.target_stage_id, args.target_artifact_id
        )
    manifest = load_json(path, "target manifest")
    if not isinstance(manifest, dict):
        raise AdapterError(f"target manifest is not an object: {path}")
    missing = [field for field in REQUIRED_TARGET_MANIFEST_FIELDS if not manifest.get(field)]
    if missing:
        raise AdapterError(f"target manifest {path} records no {', '.join(missing)}")
    source = Path(str(manifest["source_structure_path"])).expanduser()
    if not source.is_file():
        raise AdapterError(f"target source structure is missing: {source}")
    observed = sha256_file(source)
    if observed != str(manifest["target_sha256"]):
        raise AdapterError(
            f"target source structure changed since the target stage: {source}; the manifest "
            f"records {manifest['target_sha256']} and the file reads {observed}"
        )
    return manifest, path


def normalized_structure(manifest: dict[str, Any], path: Path) -> tuple[Path, str]:
    """Return the normalized target structure this stage designs against, and its digest."""
    structure = Path(str(manifest["normalized_structure_path"])).expanduser()
    if not structure.is_file():
        raise AdapterError(f"normalized target structure is missing: {structure}")
    observed = sha256_file(structure)
    recorded = manifest.get("normalized_structure_sha256")
    if isinstance(recorded, str) and recorded and recorded != observed:
        raise AdapterError(
            f"normalized target structure changed since the target stage: {structure}; the "
            f"manifest records {recorded} and the file reads {observed}"
        )
    suffix = structure.suffix.lower()
    if suffix not in TARGET_PATH_SUFFIXES:
        raise AdapterError(
            f"BindCraft 2 reads a target file ending in one of "
            f"{', '.join(TARGET_PATH_SUFFIXES)}, and the normalized target structure is "
            f"{structure.name}"
        )
    return structure.resolve(), observed


def hotspot_tokens(manifest: dict[str, Any], path: Path, chain: str) -> list[str]:
    """Return the design site as BindCraft 2 hotspot tokens.

    `targets[].hotspots` reads as chain-prefixed residue numbers such as
    `A54,A56`, in the numbering of the file `target_path` names. The target
    manifest writes each residue as `CHAIN:NUMBER` with an optional insertion
    code, so the colon comes out and an insertion code is refused rather than
    dropped: BindCraft 2's hotspot grammar has no place to put one, and dropping
    it would aim the campaign at a different residue.
    """
    site = manifest.get("site")
    if not isinstance(site, dict):
        raise AdapterError(f"target manifest {path} records no site")
    residues = site.get("resolved_design_residues")
    if not isinstance(residues, list) or not residues:
        raise AdapterError(f"target manifest {path} records no resolved_design_residues")
    tokens: list[str] = []
    for residue in residues:
        match = RESIDUE_ID_RE.fullmatch(str(residue))
        if match is None:
            raise AdapterError(f"site residue does not read CHAIN:NUMBER: {residue}")
        residue_chain, number, insertion = match.group(1), match.group(2), match.group(3)
        if insertion:
            raise AdapterError(
                f"site residue {residue} carries an insertion code, and a BindCraft 2 hotspot "
                "is a chain letter and a residue number with no place for one"
            )
        if residue_chain != chain:
            raise AdapterError(
                f"site residue {residue} names chain {residue_chain}, and the design target "
                f"chain is {chain}"
            )
        token = f"{residue_chain}{int(number)}"
        if token not in tokens:
            tokens.append(token)
    return tokens


# ----------------------------------------------------------------------------
# Composing the campaign.
# ----------------------------------------------------------------------------


def split_names(values: list[str] | None, flag: str) -> list[str]:
    """Return one name list from a repeatable flag that also accepts a comma list."""
    names: list[str] = []
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                raise AdapterError(f"{flag} holds an empty name: {value!r}")
            if item not in names:
                names.append(item)
    return names


def validate_name(kind: str, name: str, available: list[str], flag: str) -> None:
    """Refuse a name the installed copy does not print, and say what it does print."""
    if name in available:
        return
    raise AdapterError(
        f"{flag} names a {kind} the installed BindCraft 2 does not ship: {name}. "
        f"{nearest_names(name, available)}"
    )


def property_flag(name: str) -> str:
    """Return the command-line flag one design property is spelled as.

    `cli.design_property_flags` builds each flag from the preset file name with
    underscores replaced by hyphens, so `protease_stable` is `--protease-stable`.
    """
    return "--" + name.replace("_", "-")


def setting_assignments(
    values: list[str] | None, allowed: list[str], permitted_unlisted: list[str]
) -> list[tuple[str, str]]:
    """Return the validated `--set` assignments, in the order they were given.

    `settings.parse_setting_overrides` splits on the first `=` and walks dotted
    key segments into nested blocks, so the first segment is the setting name and
    the check applies to it. `--list-settings` prints
    `settings.CAMPAIGN_SETTING_NAMES`, which is narrower than the set
    `reject_unrecognized_settings` accepts: that one also carries preset names,
    derived loss weights and per-stage thresholds. `--allow-unlisted-setting`
    names a key to exempt for exactly that case, and the receipt records every
    exemption.
    """
    assignments: list[tuple[str, str]] = []
    for value in values or []:
        text = str(value)
        name, separator, assigned = text.partition("=")
        if not separator or not name:
            raise AdapterError(
                f"--set reads KEY=VALUE, as in binder_lengths=[60,80] or "
                f"filters.i_pTM.threshold=0.8; got {text!r}"
            )
        setting = name.split(".")[0].strip()
        if not setting:
            raise AdapterError(f"--set names no setting: {text!r}")
        if setting in ADAPTER_OWNED_SETTINGS:
            raise AdapterError(
                f"--set refuses {setting}, because this stage composes it. BindCraft 2 applies "
                "--set over the composed settings file, so the campaign would move while the run "
                "receipt and every published row still recorded the composed value. Use the "
                f"stage's own flag instead, or change the campaign. The composed settings are "
                f"{', '.join(ADAPTER_OWNED_SETTINGS)}."
            )
        if setting not in allowed and setting not in permitted_unlisted:
            raise AdapterError(
                f"--set names a setting the installed BindCraft 2 does not list: {setting}. "
                f"{nearest_names(setting, allowed)}. Its own recognized set is wider than the "
                "--list-settings listing, so pass --allow-unlisted-setting "
                f"{setting} if the installed copy accepts it."
            )
        assignments.append((name.strip(), assigned))
    return assignments


def binder_lengths(args: argparse.Namespace) -> list[int]:
    """Return the `binder_lengths` value this campaign writes.

    `settings.campaign_binder_lengths` reads a two-element list as an inclusive
    range and any other length as the literal choices, so `[60,100]` draws from
    60 to 100 and `[60,80,100]` draws one of three. `--binder-length-min` and
    `--binder-length-max` write the range form; a repeated `--binder-length`
    writes the literal form. Naming both is refused rather than merged.
    """
    explicit = list(args.binder_length or [])
    has_window = args.binder_length_min is not None or args.binder_length_max is not None
    if explicit and has_window:
        raise AdapterError(
            "--binder-length names the exact lengths to draw from and "
            "--binder-length-min/--binder-length-max name an inclusive range. Pass one form."
        )
    if explicit:
        for length in explicit:
            if length < 1:
                raise AdapterError(f"--binder-length must be positive: {length}")
        return explicit
    if args.binder_length_min is None or args.binder_length_max is None:
        raise AdapterError(
            "the binder length is not set. Pass --binder-length-min and --binder-length-max "
            "for an inclusive range, or --binder-length once for each exact length."
        )
    if args.binder_length_min < 1:
        raise AdapterError(f"--binder-length-min must be positive: {args.binder_length_min}")
    if args.binder_length_max < args.binder_length_min:
        raise AdapterError(
            f"--binder-length-min is {args.binder_length_min} and --binder-length-max is "
            f"{args.binder_length_max}"
        )
    return [args.binder_length_min, args.binder_length_max]


def compose_settings(
    *,
    target_id: str,
    target_relative_path: str,
    chains: str,
    hotspots: list[str],
    project_folder: Path,
    lengths: list[int],
    number_of_final_designs: int,
    max_trajectories: int | None,
    resume: bool,
    campaign_seed: int | None,
    binder_name: str | None,
    campaign_name: str | None,
) -> dict[str, Any]:
    """Return the campaign settings file BindCraft 2 reads.

    `settings.read_settings` resolves `targets[].target_path` against the
    settings file's own parent directory, so the target path is written relative
    to this file and the structure is copied in beside it. `project_folder` gets
    no such treatment anywhere upstream, so it is written absolute.
    """
    target: dict[str, Any] = {"name": target_id, "target_path": target_relative_path, "chains": chains}
    if hotspots:
        target["hotspots"] = ",".join(hotspots)
    unknown = [key for key in target if key not in TARGET_SETTING_NAMES]
    if unknown:
        raise AdapterError(
            f"the composed target entry carries {', '.join(unknown)}, and BindCraft 2 accepts "
            f"only {', '.join(TARGET_SETTING_NAMES)}"
        )
    settings: dict[str, Any] = {
        "targets": [target],
        "project_folder": str(project_folder),
        "binder_lengths": lengths,
        "number_of_final_designs": number_of_final_designs,
        "resume": resume,
    }
    if max_trajectories is not None:
        settings["max_trajectories"] = max_trajectories
    if campaign_seed is not None:
        settings["campaign_seed"] = campaign_seed
    if binder_name:
        settings["binder_name"] = binder_name
    if campaign_name:
        settings["campaign_name"] = campaign_name
    return settings


def design_argv(
    executable: Path,
    settings_path: Path,
    *,
    core: str,
    modalities: list[str],
    properties: list[str],
    metadata_path: Path | None,
    assignments: list[tuple[str, str]],
) -> list[str]:
    """Return the exact `bindcraft design` command line, in the documented order.

    From upstream's own usage text:

        bindcraft design <settings.json> [--core NAME] [--modality NAME[,NAME]]
                         [--<property>]... [--metadata <metadata.json>] [--set KEY=VALUE]...

    `--modality` is omitted when the campaign names none, because
    `settings.requested_preset_names` applies no modality layer in that case and
    passing `binder` by hand would apply one. It also prepends `binder` itself
    when the named modalities hold no binder format, so a combination such as
    `induced_fit` arrives as `binder` plus `induced_fit` without this wrapper
    saying so.
    """
    argv = [str(executable), "design", str(settings_path), "--core", core]
    if modalities:
        argv.extend(["--modality", ",".join(modalities)])
    argv.extend(property_flag(name) for name in properties)
    if metadata_path is not None:
        argv.extend(["--metadata", str(metadata_path)])
    for name, value in assignments:
        argv.extend(["--set", f"{name}={value}"])
    return argv


def place_target_structure(
    source: Path, destination: Path, source_sha256: str, replace: bool
) -> str:
    """Copy the target structure beside the settings file and return its digest.

    The settings file resolves `target_path` against its own directory, so the
    structure is placed there and the campaign folder is readable on its own. An
    existing copy of the same bytes is this stage's own previous output and needs
    no flag, following `canary_runner._refuse_occupied_output`. Different bytes
    under the same name are somebody else's file.
    """
    if destination.is_symlink():
        raise AdapterError(
            f"the target structure destination {destination} is a symbolic link. Writing "
            "through it would replace the file it points at. Name the real path."
        )
    if destination.exists():
        if not destination.is_file():
            raise AdapterError(
                f"the target structure destination {destination} is not a regular file"
            )
        if sha256_file(destination) == source_sha256:
            return source_sha256
        if not replace:
            raise AdapterError(
                f"{destination} already holds a different file from the target structure this "
                f"stage designs against. Name a different work directory, or pass --replace."
            )
    shutil.copyfile(source, destination)
    copied = sha256_file(destination)
    if copied != source_sha256:
        raise AdapterError(
            f"the target structure copied to {destination} hashes {copied} and the source "
            f"{source} hashes {source_sha256}"
        )
    return copied


def recognize_campaign_directory(path: Path, replace: bool) -> bool:
    """Return whether an existing campaign directory is this adapter's own output.

    A campaign folder holding `campaign_metadata.json` or one of BindCraft 2's
    three stage folders is a campaign this stage already started, and re-running
    onto it is the documented flow: `resume` carries the run on into it. A
    directory holding anything else is refused, on the precedent
    `canary_runner._refuse_occupied_output` sets for an occupied output path.
    """
    if path.is_symlink():
        raise AdapterError(
            f"the campaign directory {path} is a symbolic link. Writing through it would fill "
            "the directory it points at. Name the real path."
        )
    if not path.exists():
        return False
    if not path.is_dir():
        raise AdapterError(
            f"the campaign directory {path} is not a directory. Name a path that does not "
            "exist, or a directory this adapter may write into."
        )
    children = sorted(child.name for child in path.iterdir())
    if not children:
        return False
    own = {CAMPAIGN_METADATA_NAME, TRAJECTORY_STAGE_DIRECTORY, REFOLDED_STAGE_DIRECTORY, RANKED_STAGE_DIRECTORY}
    if own & set(children):
        return True
    if replace:
        return False
    raise AdapterError(
        f"the campaign directory {path} already holds {', '.join(children[:6])} and none of it "
        f"is a BindCraft 2 campaign this adapter started. Name a path that does not exist, or "
        "pass --replace to write into it anyway."
    )


# ----------------------------------------------------------------------------
# toolcheck.
# ----------------------------------------------------------------------------


def toolcheck(args: argparse.Namespace) -> int:
    """Report this adapter's own readiness, spending nothing.

    It resolves the executable, reads the five vocabularies, and reports which
    checkpoints are on this machine and every directory it looked in. It runs no
    design, loads no model, opens no socket and downloads nothing.
    """
    out_dir = Path(args.out_dir).expanduser().resolve()
    report: dict[str, Any] = {
        "format": TOOLCHECK_REPORT_FORMAT,
        "adapter_id": ADAPTER_ID,
        "route_id": ROUTE_ID,
        "route": "local-process",
        "cost_basis": COST_BASIS,
        "generator_mode": GENERATOR_MODE,
        "downloads_nothing": True,
        "problems": [],
    }
    problems: list[str] = report["problems"]

    executable: Path | None = None
    try:
        executable, source = resolve_executable(args.bindcraft_executable)
        report["executable"] = str(executable)
        report["executable_source"] = source
    except AdapterError as exc:
        report["executable"] = None
        report["executable_source"] = None
        problems.append(str(exc))

    package_directory: Path | None = None
    if executable is not None:
        try:
            package_directory, package_source = resolve_package_directory(
                executable, args.bindcraft_package_dir
            )
        except AdapterError as exc:
            package_source = str(exc)
            problems.append(str(exc))
        report["package_directory"] = str(package_directory) if package_directory else None
        report["package_directory_source"] = package_source

        vocabularies: dict[str, list[str]] = {}
        for name, flag in sorted(LIST_FLAGS.items()):
            try:
                vocabularies[name] = read_listing(executable, flag, name)
            except AdapterError as exc:
                problems.append(str(exc))
        report["vocabularies"] = vocabularies
        report["vocabulary_counts"] = {name: len(names) for name, names in vocabularies.items()}
        report["default_core_profile"] = DEFAULT_CORE_PROFILE
        if "core" in vocabularies and DEFAULT_CORE_PROFILE not in vocabularies["core"]:
            problems.append(
                f"the installed BindCraft 2 does not ship the {DEFAULT_CORE_PROFILE} core "
                f"profile this adapter defaults to; it prints {', '.join(vocabularies['core'])}"
            )
        seed, seed_source = core_profile_seed(package_directory, DEFAULT_CORE_PROFILE)
        report["default_core_profile_campaign_seed"] = seed
        report["default_core_profile_source"] = seed_source

    parameters = alphafold_report(package_directory)
    report["alphafold_parameters"] = parameters
    for file_name in parameters["missing"]:
        searched = ", ".join(record["directory"] for record in parameters["searched"])
        problems.append(
            f"missing input: AlphaFold 2 parameter file {file_name}. Looked in {searched}. "
            f"BindCraft 2 fetches these as one {ALPHAFOLD_PARAMETER_GIGABYTES} GB archive on "
            f"first use and this check never downloads it. Run bindcraft fetch-weights where "
            f"the network reaches it, or set {ALPHAFOLD_PARAMETER_ENVIRONMENT_KEY} to a "
            "directory holding params_<model>.npz."
        )

    checkpoints = proteinmpnn_report(package_directory)
    report["proteinmpnn_checkpoints"] = checkpoints
    if not checkpoints["visible"]:
        problems.append(
            f"the {SHIPPED_MPNN_CHECKPOINT_COUNT} shipped ProteinMPNN and HyperMPNN checkpoints "
            f"are not visible from here: {checkpoints['root_source']}. They ship inside the "
            "package, so this is a report about what this check could see and not a statement "
            "that BindCraft 2 lacks them."
        )
    elif checkpoints["present_count"] != SHIPPED_MPNN_CHECKPOINT_COUNT:
        problems.append(
            f"{checkpoints['present_count']} of {SHIPPED_MPNN_CHECKPOINT_COUNT} shipped "
            f"ProteinMPNN checkpoints are whole under {checkpoints['root']}"
        )

    report["ready"] = not problems
    write_json(out_dir / TOOLCHECK_REPORT_NAME, report)

    print(f"{TOOL_LABEL} adapter: report {out_dir / TOOLCHECK_REPORT_NAME}")
    print(
        f"{TOOL_LABEL} adapter: executable {report.get('executable')} "
        f"via {report.get('executable_source')}"
    )
    for name in VOCABULARY_NAMES:
        names = report.get("vocabularies", {}).get(name)
        if names is not None:
            print(f"{TOOL_LABEL} adapter: {name} ({len(names)}) {' '.join(names)}")
    print(
        f"{TOOL_LABEL} adapter: AlphaFold parameters {parameters['present_count']} of "
        f"{parameters['required_count']}; nothing downloaded"
    )
    print(
        f"{TOOL_LABEL} adapter: ProteinMPNN checkpoints {checkpoints['present_count']} of "
        f"{checkpoints['expected_count']} under {checkpoints['root']}"
    )
    print(
        f"{TOOL_LABEL} adapter: local process route, no endpoint and no credential; cost basis "
        f"{COST_BASIS}",
        flush=True,
    )
    for problem in problems:
        print(f"{TOOL_LABEL} adapter: {problem}", file=sys.stderr)
    return 0 if report["ready"] else 1


# ----------------------------------------------------------------------------
# run.
# ----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Compose the settings file, validate every name, run one campaign, record it.

    Every check runs before the subprocess starts. A campaign that dies on a
    mistyped modality has already paid for its imports, and one that dies on a
    mistyped property flag prints a usage block naming nothing.
    """
    if args.count < 1:
        raise AdapterError(f"--count must be positive: {args.count}")
    if args.max_trajectories is not None and args.max_trajectories < 1:
        raise AdapterError(f"--max-trajectories must be positive: {args.max_trajectories}")
    if args.timeout_seconds < 1:
        raise AdapterError(f"--timeout-seconds must be positive: {args.timeout_seconds}")

    executable, executable_source = resolve_executable(args.bindcraft_executable)
    vocabularies = read_vocabularies(executable)
    package_directory, package_source = resolve_package_directory(
        executable, args.bindcraft_package_dir
    )

    modalities = split_names(args.modality, "--modality")
    properties = split_names(args.property, "--property")
    permitted_unlisted = split_names(args.allow_unlisted_setting, "--allow-unlisted-setting")
    validate_name("core profile", args.core, vocabularies["core"], "--core")
    unread_presets: list[str] = []
    for name in modalities:
        validate_name("modality", name, vocabularies["modalities"], "--modality")
        note = refuse_multi_chain_modality(package_directory, name)
        if note is not None:
            unread_presets.append(note)
    for name in properties:
        validate_name("design property", name, vocabularies["properties"], "--property")
    assignments = setting_assignments(args.set, vocabularies["settings"], permitted_unlisted)
    lengths = binder_lengths(args)

    manifest, manifest_source = load_target_manifest(args)
    target_id = str(manifest["target_id"])
    if NAME_RE.fullmatch(target_id) is None:
        raise AdapterError(f"target manifest records an unusable target_id: {target_id!r}")
    structure, structure_sha256 = normalized_structure(manifest, manifest_source)
    design_chain = str(manifest["design_target_chain_id"])
    chain_ids = [str(chain) for chain in manifest["chain_ids"]]
    if design_chain not in chain_ids:
        raise AdapterError(
            f"target manifest {manifest_source} designs against chain {design_chain} and keeps "
            f"chains {', '.join(chain_ids)}"
        )
    hotspots = hotspot_tokens(manifest, manifest_source, design_chain)

    attempt_dir = Path(args.attempt_dir).expanduser().resolve()
    work_dir = attempt_dir / args.phase / args.work_subdir
    campaign_dir = (
        Path(args.campaign_dir).expanduser().resolve()
        if args.campaign_dir is not None
        else work_dir / DEFAULT_CAMPAIGN_SUBDIR
    )
    settings_path = work_dir / SETTINGS_NAME
    receipt_path = work_dir / RUN_RECEIPT_NAME
    metadata_path = (
        Path(args.metadata).expanduser().resolve() if args.metadata is not None else None
    )
    if metadata_path is not None and not metadata_path.is_file():
        raise AdapterError(f"--metadata is missing: {metadata_path}")

    resumed = recognize_campaign_directory(campaign_dir, args.replace)
    resume = bool(args.resume or resumed)
    refuse_occupied_output(
        settings_path,
        flag="the composed settings file",
        required=frozenset({"targets", "project_folder", "number_of_final_designs"}),
        expected_format="bindcraft2-campaign-settings",
        replace=args.replace,
    )

    input_dir = work_dir / DEFAULT_INPUT_SUBDIR
    input_dir.mkdir(parents=True, exist_ok=True)
    target_copy = input_dir / f"{target_id}{structure.suffix.lower()}"
    copied_sha256 = place_target_structure(
        structure, target_copy, structure_sha256, args.replace
    )
    target_relative_path = str(target_copy.relative_to(settings_path.parent))

    settings = compose_settings(
        target_id=target_id,
        target_relative_path=target_relative_path,
        chains=",".join(chain_ids),
        hotspots=hotspots,
        project_folder=campaign_dir,
        lengths=lengths,
        number_of_final_designs=args.count,
        max_trajectories=args.max_trajectories,
        resume=resume,
        campaign_seed=args.campaign_seed,
        binder_name=args.binder_name,
        campaign_name=args.campaign_name,
    )
    for name in settings:
        if name not in vocabularies["settings"] and name not in permitted_unlisted:
            raise AdapterError(
                f"the composed settings file carries {name}, which the installed BindCraft 2 "
                f"does not list. {nearest_names(name, vocabularies['settings'])}"
            )
    # The settings file carries no `format` key of this adapter's own, because
    # `settings.reject_unrecognized_settings` refuses a key BindCraft 2 does not
    # know. The guard recognizes it by the three settings every composition
    # writes instead.
    write_text_output(
        settings_path,
        json.dumps(settings, indent=2, sort_keys=True) + "\n",
        flag="the composed settings file",
        required=frozenset({"targets", "project_folder", "number_of_final_designs"}),
        expected_format="bindcraft2-campaign-settings",
        replace=args.replace,
    )
    resolved_target = (settings_path.parent / target_relative_path).resolve()
    if not resolved_target.is_file():
        raise AdapterError(
            f"the composed settings file names {target_relative_path}, which does not resolve "
            f"to a file from {settings_path.parent}"
        )

    core_seed, core_seed_source = core_profile_seed(package_directory, args.core)
    argv = design_argv(
        executable,
        settings_path,
        core=args.core,
        modalities=modalities,
        properties=properties,
        metadata_path=metadata_path,
        assignments=assignments,
    )

    campaign_dir.mkdir(parents=True, exist_ok=True)
    if unread_presets:
        # The stated fallback when the shipped presets are not readable, which is
        # what an installation from a built wheel leaves behind.
        print(
            f"{TOOL_LABEL} adapter: the binder chain count was not readable for "
            f"{'; '.join(unread_presets)}, so a multi-chain binder in this campaign is "
            "refused when it is parsed rather than now",
            flush=True,
        )
    print(f"{TOOL_LABEL} adapter: {' '.join(argv)}", flush=True)
    started = time.time()
    timed_out = False
    try:
        # The campaign's own console record is the only progress a scientist sees
        # for hours, so it streams rather than being captured and reported at the
        # end. shell=False and an argv list: nothing here is a shell string.
        completed = subprocess.run(argv, shell=False, check=False, timeout=args.timeout_seconds)
        exit_code = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        exit_code = None
    except OSError as exc:
        raise AdapterError(f"bindcraft design could not be run: {exc}") from exc
    wall_seconds = round(time.time() - started, 3)

    receipt = {
        "format": RUN_RECEIPT_FORMAT,
        "adapter_id": ADAPTER_ID,
        "route_id": ROUTE_ID,
        "route": "local-process",
        "stage_id": args.stage,
        "phase": args.phase,
        "argv": argv,
        "executable": str(executable),
        "executable_source": executable_source,
        "package_directory": str(package_directory) if package_directory else None,
        "package_directory_source": package_source,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "timeout_seconds": args.timeout_seconds,
        "wall_seconds": wall_seconds,
        "settings_path": str(settings_path),
        "settings_sha256": sha256_file(settings_path),
        "settings": settings,
        "campaign_dir": str(campaign_dir),
        "resumed_own_campaign": resumed,
        "resume": resume,
        "core_profile": args.core,
        "requested_campaign_seed": args.campaign_seed,
        "core_profile_campaign_seed": core_seed,
        "core_profile_campaign_seed_source": core_seed_source,
        "modalities": modalities,
        "design_properties": properties,
        "setting_overrides": [f"{name}={value}" for name, value in assignments],
        "unlisted_settings_permitted": permitted_unlisted,
        "vocabularies": vocabularies,
        "target_id": target_id,
        "target_manifest_path": str(manifest_source),
        "target_sha256": str(manifest["target_sha256"]),
        "input_structure_path": str(target_copy),
        "input_structure_sha256": copied_sha256,
        "target_chain_ids": chain_ids,
        "design_target_chain_id": design_chain,
        "hotspots": hotspots,
        "binder_lengths": lengths,
        "number_of_final_designs": args.count,
        "max_trajectories": args.max_trajectories,
        "cost_basis": COST_BASIS,
        "generator_mode": GENERATOR_MODE,
    }
    write_json_output(
        receipt_path,
        receipt,
        flag="the run receipt",
        required=frozenset({"argv", "executable", "settings_path"}),
        expected_format=RUN_RECEIPT_FORMAT,
        replace=args.replace,
    )

    if timed_out:
        raise AdapterError(
            f"bindcraft design exceeded --timeout-seconds {args.timeout_seconds}; the campaign "
            f"folder {campaign_dir} holds whatever it wrote and resume carries a rerun on from "
            "there"
        )
    print(
        f"{TOOL_LABEL} adapter: exit={exit_code} wall_seconds={wall_seconds} "
        f"campaign={campaign_dir} receipt={receipt_path}",
        flush=True,
    )
    if exit_code != 0:
        print(
            f"{TOOL_LABEL} adapter: bindcraft design exited {exit_code}; parse reads whatever "
            f"the campaign folder holds",
            file=sys.stderr,
        )
        return 1
    return 0


# ----------------------------------------------------------------------------
# parse.
# ----------------------------------------------------------------------------


def campaign_provenance(campaign_dir: Path) -> dict[str, Any]:
    """Return the campaign's own provenance, carried through unchanged.

    `campaign_output.write_campaign_metadata` records the version, the source
    revision with `-dirty` appended when the tree was modified, the absolute
    settings path, the complete settings, a `resolved` block, the settings digest
    and a SHA-256 for every checkpoint the run loaded. This package usually has
    to leave that field `__REQUIRED__`, so none of it is recomputed, rounded or
    normalized here.
    """
    path = campaign_dir / CAMPAIGN_METADATA_NAME
    document = load_json(path, "campaign metadata")
    if not isinstance(document, dict):
        raise AdapterError(f"campaign metadata is not an object: {path}")
    revision = document.get("revision")
    if not isinstance(revision, str) or not revision:
        raise AdapterError(f"campaign metadata records no revision: {path}")
    checkpoints = document.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise AdapterError(f"campaign metadata records no checkpoints object: {path}")
    return {
        "campaign_metadata_path": str(path),
        "campaign_metadata_sha256": sha256_file(path),
        "bindcraft_version": document.get("version"),
        "source_revision": revision,
        # A modified tree is recorded as modified. A `-dirty` revision is not a
        # clean pin and must never be reported as one.
        "source_revision_dirty": revision.endswith("-dirty"),
        "settings_path": document.get("settings_path"),
        "settings_digest": document.get("settings_digest"),
        "checkpoint_sha256": dict(checkpoints),
        "sampled_binder_lengths": document.get("sampled_binder_lengths"),
        "resolved": document.get("resolved"),
        "settings": document.get("settings"),
        "metadata": document.get("metadata"),
    }


def campaign_target_names(provenance: dict[str, Any]) -> list[str]:
    """Return the target names the campaign's own settings record.

    `write_campaign_metadata` stores the resolved settings under `settings`, and
    `targets[].name` is what the campaign designed against. The `resolved` block
    carries no target name, so this is the only record of it in the metadata.
    A target entry with no `name` falls back to its `target_path` stem, which is
    how a hand-written settings file without a name reads.
    """
    settings = provenance.get("settings")
    if not isinstance(settings, dict):
        return []
    targets = settings.get("targets")
    if isinstance(targets, dict):
        targets = [targets]
    if not isinstance(targets, list):
        return []
    names: list[str] = []
    for entry in targets:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
            continue
        source = entry.get("target_path")
        if isinstance(source, str) and source.strip():
            names.append(Path(source.strip()).stem)
    return names


def refuse_a_foreign_campaign_target(
    provenance: dict[str, Any], target_id: str, campaign_dir: Path
) -> list[str]:
    """Refuse a campaign that designed against a target this stage did not prepare.

    `run` composes the settings from the target manifest, so the campaign it
    starts designs against the manifest's target. `parse` is handed a campaign
    directory, and nothing forced it to be that campaign: a mistyped
    `--campaign-dir` reaches another target's finished campaign, and every
    published row would then carry this stage's `target_id` over another
    protein's designs. The row would look right and be wrong, and no later stage
    could tell, because the manifest digest it records is the digest of a
    structure the campaign never read.

    So the campaign's own target names are read back and the manifest's target
    has to be among them. A campaign that records no target name at all is
    refused too, because an unnamed campaign cannot be reconciled and publishing
    it would assert provenance nothing checked.
    """
    names = campaign_target_names(provenance)
    if not names:
        raise AdapterError(
            f"the campaign at {campaign_dir} records no target name in its own settings, so "
            f"nothing here can confirm it designed against {target_id}. Publishing it would "
            "assert a provenance no file states."
        )
    if target_id not in names:
        raise AdapterError(
            f"the campaign at {campaign_dir} designed against {', '.join(names)} and this stage "
            f"prepared {target_id}. Refusing to publish another target's designs under "
            f"{target_id}. Point --campaign-dir at this stage's own campaign, or re-run the "
            "stage against the target you meant."
        )
    return names


def ranked_rows(campaign_dir: Path) -> tuple[list[dict[str, str]], Path, list[str]]:
    """Return every accepted row, the table it came from, and its header.

    `3_Ranked/!_Ranked.csv` is the single record of what BindCraft 2 accepted,
    best first by `i_pDAE`. A header missing a column this parser reads is
    refused by name.
    """
    path = campaign_dir / RANKED_STAGE_DIRECTORY / RANKED_TABLE_NAME
    if not path.is_file():
        raise AdapterError(
            f"the accepted table is missing: {path}. A stage folder appears when its first "
            f"result is written, so its absence means the campaign accepted nothing. "
            f"{campaign_dir / REFOLDED_STAGE_DIRECTORY / REFOLDED_TABLE_NAME} carries the "
            "failed filters and "
            f"{campaign_dir / TRAJECTORY_STAGE_DIRECTORY / TRAJECTORY_TABLE_NAME} carries the "
            "attempts stopped before redesign."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    missing = [column for column in REQUIRED_RANKED_COLUMNS if column not in header]
    if missing:
        raise AdapterError(
            f"{path} has no {', '.join(missing)} column, and this parser reads it. A blank cell "
            "is a missing reading and a default would be a claim, so the column is required "
            f"rather than defaulted. The header carries {', '.join(header) or 'nothing'}."
        )
    if not rows:
        raise AdapterError(f"{path} carries a header and no accepted design")
    return rows, path, header


def metric_reading(raw: Any) -> Any:
    """Return one table cell as a reading, never as a number it does not hold.

    A blank cell stays null: a missing reading is not a zero. A cell holding
    semicolon-separated readings is one reading per target in `targets` order
    with empty positions retained, so it comes through as a list and is never
    averaged into one number.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if TARGET_VALUE_SEPARATOR in text:
        return [part.strip() or None for part in text.split(TARGET_VALUE_SEPARATOR)]
    try:
        return float(text)
    except ValueError:
        return text


def integer_reading(raw: Any, label: str) -> int:
    """Return one integer table cell, refusing a cell that does not hold one."""
    text = "" if raw is None else str(raw).strip()
    try:
        return int(text)
    except ValueError as exc:
        raise AdapterError(f"{label} does not read as an integer: {text!r}") from exc


def atom_site_block(text: str, label: str) -> str:
    """Return just the `atom_site` loop out of one mmCIF.

    BindCraft 2 writes further categories after the coordinates: a `bindcraft`
    provenance stamp and per-residue quality tracks. The shared converter reads
    one loop and treats a later category's key-value line as a malformed atom
    row, so the text is narrowed here rather than the converter being rewritten.
    """
    lines = text.splitlines()
    start = None
    for index, raw in enumerate(lines):
        if raw.strip() != "loop_":
            continue
        following = next(
            (lines[later].strip() for later in range(index + 1, len(lines)) if lines[later].strip()),
            "",
        )
        if following.startswith("_atom_site."):
            start = index
            break
    if start is None:
        raise AdapterError(f"{label} carries no _atom_site loop")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("_atom_site."):
            continue
        if stripped.startswith("#") or stripped == "loop_" or stripped.startswith("data_") or stripped.startswith("_"):
            end = index
            break
    return "\n".join(lines[start:end]) + "\n"


def design_pose_text(cif_text: str, candidate_id: str, design_name: str) -> str:
    """Return the PDB design pose for one accepted complex.

    The mmCIF-to-PDB conversion is `rfdiffusion3_generator.cif_to_pdb`, imported
    rather than copied: its fixed-column writer is the one every downstream
    reader in this package was fixed against, and a second converter would drift
    from it. That module's own `REMARK 900` header names RFdiffusion3 and says the
    design chain is redesigned downstream, and neither is true of a codesign
    generator, so this replaces the header and keeps the coordinates.

    TODO(refactor): that converter lives in an adapter which also binds the
    hosted fal client, so this local-only route imports a module carrying a
    provider client it never calls. Settled by: moving `cif_to_pdb` and
    `format_pdb_atom_line` into a route-neutral module both adapters import,
    which is a change outside the two files this adapter was written in.
    """
    from claude_binder.adapters.rfdiffusion3_generator import (
        AdapterError as ConverterError,
        cif_to_pdb,
    )

    block = atom_site_block(cif_text, f"accepted complex {design_name}")
    try:
        converted = cif_to_pdb(block, candidate_id)
    except ConverterError as exc:
        raise AdapterError(f"accepted complex {design_name} could not be converted: {exc}") from exc
    body = [line for line in converted.splitlines() if not line.startswith("REMARK 900 ")]
    header = [
        f"REMARK 900 DESIGN POSE {candidate_id}",
        f"REMARK 900 BindCraft 2 accepted complex {design_name}, converted from the campaign mmCIF",
        "REMARK 900 The B-factor column holds per-residue pLDDT on a 0 to 100 scale",
        "REMARK 900 Structure and sequence are both BindCraft 2's; no sequence designer follows",
    ]
    return "\n".join(header + body) + "\n"


def bindcraft_stamp(text: str) -> dict[str, str]:
    """Return the `bindcraft` provenance category one accepted mmCIF carries.

    `protein.write_structure` writes a `bindcraft` category holding the design
    name, the design hash, the source revision, the model choices and the written
    binder and target chain letters. It is read here rather than asserted, and an
    absent or loop-form category comes back empty rather than guessed at.
    """
    stamp: dict[str, str] = {}
    prefix = "_bindcraft."
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped.startswith(prefix):
            continue
        name, _, value = stripped.partition(" ")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value in {".", "?"}:
            continue
        stamp[name[len(prefix) :]] = value
    return stamp


def accepted_complex_path(rank_dir: Path, design: str, state_suffix: str | None) -> Path:
    """Return the accepted predicted complex for one ranked row.

    `campaign_output.accepted_structure_present` reconciles the ranked table
    against `<design>.cif` and `<design>_*.cif`, excluding `_monomer.cif`, so
    that is the rule used here. A campaign with several target states writes one
    complex per state, and the suffix is the state name or, for a cropped target,
    the source target and its residue span. Nothing in the table settles which
    file is the primary state, so several matches are refused by name and
    `--target-state-suffix` names the one to publish.
    """
    if state_suffix:
        path = rank_dir / f"{design}{state_suffix}.cif"
        if not path.is_file():
            raise AdapterError(
                f"--target-state-suffix {state_suffix} names no accepted complex for {design}: "
                f"{path}"
            )
        return path
    matches = sorted(
        path
        for path in list(rank_dir.glob(f"{design}.cif")) + list(rank_dir.glob(f"{design}_*.cif"))
        if not path.name.endswith(MONOMER_SUFFIX)
    )
    if not matches:
        raise AdapterError(
            f"the accepted table names design {design} and {rank_dir} holds no matching "
            f"complex. Deleting a .cif is how a design is rejected by hand, and the campaign "
            "drops such a row when it closes."
        )
    if len(matches) > 1:
        raise AdapterError(
            f"design {design} has {len(matches)} accepted complexes in {rank_dir}: "
            f"{', '.join(path.name for path in matches)}. A campaign with several target states "
            "writes one file per state and the accepted table does not say which is primary. "
            "Name it with --target-state-suffix."
        )
    return matches[0]


def binder_sequence(raw: str, design: str) -> str:
    """Return the one designed binder sequence a row carries.

    `Binder_Sequence` is the one-letter sequence with `/` between binder chains.
    The sequence consumers in this lane read one sequence per candidate from a
    single-record FASTA, so a multi-chain binder is refused by name rather than
    concatenated, split or silently truncated.

    TODO(evidence): the lane has no rule for publishing a binder that carries
    several chains, which is what the scFv, Fab, homo_oligomer and multidomain
    modalities design. Settled by: a lane decision on whether such a candidate is
    one row with several sequence files or several rows, recorded in the
    candidate manifest schema before one of those modalities is run.
    """
    text = str(raw or "").strip().upper()
    if not text:
        raise AdapterError(f"the accepted row for {design} carries an empty Binder_Sequence")
    if "/" in text:
        chains = [part for part in text.split("/") if part]
        raise AdapterError(
            f"the accepted row for {design} carries {len(chains)} binder chains separated by "
            "'/', and this stage publishes one sequence per candidate. Run a single-chain "
            "modality, or settle the lane's rule for a multi-chain binder first."
        )
    unexpected = sorted(set(text) - set("ACDEFGHIKLMNPQRSTVWY"))
    if unexpected:
        raise AdapterError(
            f"the accepted row for {design} carries {', '.join(unexpected)} in its "
            "Binder_Sequence, which is not a one-letter amino acid"
        )
    return text


def trajectory_numbers(campaign_dir: Path) -> dict[str, dict[str, str]]:
    """Return the claimed attempt number for each design recipe hash, when readable.

    `hash` is the design-recipe identity and the join key back to
    `1_Trajectories/!_Trajectories.csv`, which is the only table carrying
    `terminated` and `autotuned`. The attempt number derives each trajectory's
    random seed upstream, and nothing published states the derivation, so this
    records the number and claims no seed from it.
    """
    path = campaign_dir / TRAJECTORY_STAGE_DIRECTORY / TRAJECTORY_TABLE_NAME
    if not path.is_file():
        return {}
    joined: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            recipe = (row.get("hash") or "").strip()
            if recipe:
                joined[recipe] = dict(row)
    return joined


def declared_record_count(path: Path, kind: str) -> int:
    """Count records in one declared output the way the executor counts them.

    `lane.validate_artifact` derives a file's record count from its declared
    `kind` and the executor compares the sum against the parser's
    `parsed_count`. A JSONL file counts its rows, a FASTA counts its `>` lines,
    and a PDB without MODEL delimiters is one pose.
    """
    if kind == "jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if kind == "json":
        return 1
    if kind == "fasta":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(">"))
    if kind == "pdb":
        poses = sum(
            1 for line in path.read_text(errors="replace").splitlines() if line.startswith("MODEL ")
        )
        return poses if poses else 1
    raise AdapterError(f"declared output kind has no record rule in this parser: {kind}")


def declared_output_hashes(args: argparse.Namespace) -> tuple[list[str], list[str], int]:
    """Hash and count the files this stage declares as outputs.

    The executor compares the parser's `source_output_hashes` against the
    declared stage outputs, so this resolves each declared `path_template` the
    way `target_prep_parser` and `rfdiffusion3_generator` do rather than
    reporting whatever the campaign folder happens to hold.
    """
    hashes: list[str] = []
    problems: list[str] = []
    records = 0
    try:
        config = load_json(Path(args.config).expanduser().resolve(), "resolved configuration")
    except AdapterError as exc:
        return [], [str(exc)], 0
    stage: dict[str, Any] = {}
    for candidate in config.get("stages", []):
        if isinstance(candidate, dict) and candidate.get("stage_id") == args.stage:
            stage = candidate
            break
    if not stage:
        return [], [f"stage is not present in config: {args.stage}"], 0
    attempt_dir = Path(args.attempt_dir).expanduser().resolve()
    for output in stage.get("outputs", []):
        if not isinstance(output, dict) or "path_template" not in output:
            problems.append("stage output contract is not an object with a path_template")
            continue
        pattern = (
            str(output["path_template"])
            .replace("{{attempt_dir}}", str(attempt_dir))
            .replace("{{phase}}", str(args.phase))
            .replace("{{stage_id}}", str(args.stage))
        )
        if "{{" in pattern or "}}" in pattern:
            problems.append(f"stage output path has an unresolved token: {output['path_template']}")
            continue
        for value in sorted(glob.glob(pattern, recursive=True)):
            path = Path(value).resolve()
            if not path.is_file():
                continue
            if attempt_dir not in path.parents:
                problems.append(f"stage output escapes attempt directory: {path}")
                continue
            digest = sha256_file(path)
            if digest in hashes:
                continue
            hashes.append(digest)
            try:
                records += declared_record_count(path, str(output.get("kind", "")))
            except AdapterError as exc:
                problems.append(str(exc))
    return sorted(hashes), problems, records


def publish_candidates(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Publish poses, sequences and manifest rows for every accepted design."""
    attempt_dir = Path(args.attempt_dir).expanduser().resolve()
    phase_dir = attempt_dir / args.phase
    work_dir = phase_dir / args.work_subdir
    campaign_dir = (
        Path(args.campaign_dir).expanduser().resolve()
        if args.campaign_dir is not None
        else work_dir / DEFAULT_CAMPAIGN_SUBDIR
    )
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path is not None
        else phase_dir / DEFAULT_MANIFEST_NAME
    )

    target_manifest, target_manifest_source = load_target_manifest(args)
    provenance = campaign_provenance(campaign_dir)
    # Before anything is published: the campaign has to be this stage's campaign.
    # Nothing upstream of here forced that, and the cost of not checking is a row
    # that names one protein and carries another's design.
    campaign_targets = refuse_a_foreign_campaign_target(
        provenance, str(target_manifest["target_id"]), campaign_dir
    )
    rows, table_path, header = ranked_rows(campaign_dir)
    trajectories = trajectory_numbers(campaign_dir)
    rank_dir = campaign_dir / RANKED_STAGE_DIRECTORY

    resolved = provenance.get("resolved")
    settings = provenance.get("settings")
    campaign_seed = None
    for source in (settings, resolved):
        if isinstance(source, dict) and not isinstance(source.get("campaign_seed"), bool):
            value = source.get("campaign_seed")
            if isinstance(value, int):
                campaign_seed = value
                break
    if campaign_seed is None:
        raise AdapterError(
            f"{provenance['campaign_metadata_path']} records no integer campaign_seed, and "
            "every manifest row needs an integer generator_seed"
        )

    ordered = sorted(rows, key=lambda row: integer_reading(row.get("rank"), "rank"))
    published: list[dict[str, Any]] = []
    for index, row in enumerate(ordered):
        design = str(row.get("design") or "").strip()
        if not design or SAFE_DESIGN_NAME_RE.fullmatch(design) is None:
            raise AdapterError(f"{table_path} row {index} carries an unusable design name: {design!r}")
        candidate_id = f"{args.generator_id}-{index:03d}"
        sequence = binder_sequence(row.get("Binder_Sequence"), design)
        complex_path = accepted_complex_path(rank_dir, design, args.target_state_suffix)
        cif_text = complex_path.read_text(errors="replace")
        stamp = bindcraft_stamp(cif_text)
        stamped_design = stamp.get("design")
        if stamped_design and stamped_design != design:
            raise AdapterError(
                f"{complex_path} carries the bindcraft stamp design {stamped_design} and the "
                f"accepted table names {design}"
            )
        pose_path = phase_dir / args.pose_subdir / f"{candidate_id}.pdb"
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        pose_path.write_text(design_pose_text(cif_text, candidate_id, design), encoding="utf-8")
        sequence_path = phase_dir / args.sequence_subdir / f"{candidate_id}.fasta"
        sequence_path.parent.mkdir(parents=True, exist_ok=True)
        sequence_path.write_text(f">{candidate_id}\n{sequence}\n", encoding="utf-8")

        trajectory_row = trajectories.get((row.get("hash") or "").strip(), {})
        readings = {
            column: metric_reading(row.get(column))
            for column in OPTIONAL_RANKED_COLUMNS
            if column in header
        }
        readings[RANKING_METRIC] = metric_reading(row.get(RANKING_METRIC))
        multi_target = sorted(
            name
            for name, value in readings.items()
            if isinstance(value, list) and name not in TEXT_RANKED_COLUMNS
        )
        published.append(
            {
                "target_id": str(target_manifest["target_id"]),
                "target_sha256": str(target_manifest["target_sha256"]),
                "candidate_id": candidate_id,
                "parent_candidate_id": None,
                "origin_generator": args.generator_id,
                **backbone_lineage(candidate_id, args.generator_id),
                "generator_mode": GENERATOR_MODE,
                "runner_protocol": "local-process",
                # BindCraft 2 redraws every trajectory's sequence with its own
                # ProteinMPNN step, so this stage stands alone and no sequence
                # designer runs after it.
                "sequence_designer": None,
                # The campaign seed is the only seed any BindCraft 2 output
                # records. Each trajectory's own seed is drawn from it inside the
                # run and is written to no table and no structure stamp.
                "generator_seed": campaign_seed,
                "campaign_seed": campaign_seed,
                "seed_semantics": "bindcraft2_campaign_seed",
                "trajectory_seed": None,
                "trajectory_number": (trajectory_row.get("trajectory") or None)
                if trajectory_row
                else None,
                "sequence_path": str(sequence_path),
                "sequence_sha256": sha256_bytes(sequence.encode("ascii")),
                "sequence_length": len(sequence),
                "backbone_only": False,
                "structure_path": str(target_manifest["source_structure_path"]),
                "structure_sha256": str(target_manifest["target_sha256"]),
                "design_pose_path": str(pose_path),
                "design_pose_sha256": sha256_file(pose_path),
                "residue_map_sha256": str(target_manifest["residue_map_sha256"]),
                "optimization_round": 0,
                "last_optimizer": None,
                "status": CANDIDATE_STATUS,
                "stage_id": args.stage,
                "phase": args.phase,
                "design_index": index,
                "design_name": design,
                "design_recipe_hash": (row.get("hash") or "").strip() or None,
                "accepted_rank": integer_reading(row.get("rank"), "rank"),
                # BindCraft 2's own filters kept this design. That is the tool's
                # computational verdict on its own thresholds and it is recorded
                # as exactly that. It does not make the row a promoted or
                # accepted binder in this lane.
                "generator_filter_verdict": "accepted",
                "generator_verdict_table": str(table_path),
                "ranking_metric": RANKING_METRIC,
                # Name the criterion this candidate was selected on, beside the
                # metric this lane screens on. Without both names a reader
                # cannot tell whether a later screen score is independent
                # evidence, and it is neither wholly independent nor a reading
                # of the same number.
                "selection_metric": SELECTION_METRIC,
                "selection_metric_value": readings.get(SELECTION_METRIC),
                "lane_screen_metric": LANE_SCREEN_METRIC,
                "selection_metric_note": SELECTION_METRIC_NOTE,
                "screened_in_lane": False,
                "readings": readings,
                "multi_target_readings": multi_target,
                "tool_output_path": str(complex_path),
                "tool_output_sha256": sha256_file(complex_path),
                "tool_structure_stamp": stamp,
                "returned_binder_chain_ids": stamp.get("binder_chains"),
                "returned_target_chain_ids": stamp.get("target_chains"),
                "declared_target_chain_id": str(target_manifest["design_target_chain_id"]),
                "binder_chain_id": str(target_manifest["binder_chain_id"])
                if target_manifest.get("binder_chain_id")
                else None,
                "mpnn_model": stamp.get("mpnn_model"),
                "mpnn_variant": stamp.get("mpnn_variant"),
                "target_manifest_path": str(target_manifest_source),
                "campaign_dir": str(campaign_dir),
                "cost_basis": COST_BASIS,
                **{
                    name: provenance[name]
                    for name in (
                        "bindcraft_version",
                        "source_revision",
                        "source_revision_dirty",
                        "settings_digest",
                        "checkpoint_sha256",
                        "campaign_metadata_path",
                        "campaign_metadata_sha256",
                    )
                },
            }
        )

    write_jsonl(manifest_path, published)
    receipt = {
        "format": PARSE_RECEIPT_FORMAT,
        "adapter_id": ADAPTER_ID,
        "route_id": ROUTE_ID,
        "route": "local-process",
        "stage_id": args.stage,
        "phase": args.phase,
        "campaign_dir": str(campaign_dir),
        # The reconciliation this stage ran before publishing anything, recorded
        # so a reader can see which names were compared rather than trusting that
        # a comparison happened.
        "campaign_target_names": campaign_targets,
        "target_manifest_target_id": str(target_manifest["target_id"]),
        "accepted_table_path": str(table_path),
        "accepted_table_sha256": sha256_file(table_path),
        "accepted_table_header": header,
        "accepted_row_count": len(ordered),
        "published_candidate_count": len(published),
        "published_pose_count": len(published),
        "published_sequence_count": len(published),
        "candidate_manifest_path": str(manifest_path),
        "ranking_metric": RANKING_METRIC,
        "generator_mode": GENERATOR_MODE,
        "candidate_status": CANDIDATE_STATUS,
        "cost_basis": COST_BASIS,
        "target_manifest_path": str(target_manifest_source),
        "summary_path": str(campaign_dir / SUMMARY_NAME)
        if (campaign_dir / SUMMARY_NAME).is_file()
        else None,
        **provenance,
    }
    write_json(work_dir / PARSE_RECEIPT_NAME, receipt)
    return len(published), receipt


def parse_outputs(args: argparse.Namespace) -> int:
    """Publish this stage's outputs and write the parser result the lane reads."""
    phase_dir = Path(args.attempt_dir).expanduser().resolve() / args.phase
    result_path = phase_dir / DEFAULT_PARSER_RESULT_NAME
    errors: list[str] = []
    published = 0
    receipt: dict[str, Any] = {}
    try:
        published, receipt = publish_candidates(args)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")
    declared_hashes, declared_problems, declared_records = declared_output_hashes(args)
    errors.extend(declared_problems)
    # `published` counts candidates. `parsed_count` counts records across the
    # stage's declared outputs, which the executor recomputes and compares, and
    # one candidate produces a manifest row, a pose and a sequence.
    parsed_count = declared_records if published and not errors else published
    write_json(
        result_path,
        {
            "ok": bool(published) and not errors,
            "parsed_count": parsed_count,
            "rejected_count": len(errors),
            "errors": errors,
            "source_output_hashes": declared_hashes,
        },
    )
    for message in errors:
        print(f"{TOOL_LABEL} parse: {message}", file=sys.stderr)
    if published:
        print(
            f"{TOOL_LABEL} adapter: accepted={receipt.get('accepted_row_count')} "
            f"poses={published} sequences={published} manifest_rows={published} "
            f"revision={receipt.get('source_revision')} "
            f"dirty={receipt.get('source_revision_dirty')} "
            f"manifest={receipt.get('candidate_manifest_path')}",
            flush=True,
        )
    return 0 if published and not errors else 1


# ----------------------------------------------------------------------------
# The command line.
# ----------------------------------------------------------------------------


def add_executable_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        EXECUTABLE_ARGUMENT,
        default=None,
        help=(
            "The installed bindcraft executable. Falls back to "
            f"{EXECUTABLE_ENVIRONMENT_KEY} and then to a PATH lookup."
        ),
    )
    parser.add_argument(
        "--bindcraft-package-dir",
        default=None,
        help=(
            "The installed bindcraft package directory. Only a readiness report needs it, and "
            "it is otherwise found from the executable's own interpreter."
        ),
    )


def add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--work-subdir", default=DEFAULT_WORK_SUBDIR)
    parser.add_argument("--campaign-dir", type=Path, default=None)
    parser.add_argument("--target-manifest", type=Path, default=None)
    parser.add_argument("--target-stage-id", default="target-prepare")
    parser.add_argument("--target-artifact-id", default="target-manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    toolcheck_parser = subparsers.add_parser("toolcheck")
    add_executable_arguments(toolcheck_parser)
    toolcheck_parser.add_argument("--out-dir", type=Path, required=True)

    run_parser = subparsers.add_parser("run")
    add_executable_arguments(run_parser)
    add_stage_arguments(run_parser)
    run_parser.add_argument("--count", type=int, required=True)
    run_parser.add_argument("--max-trajectories", type=int, default=None)
    run_parser.add_argument("--binder-length-min", type=int, default=None)
    run_parser.add_argument("--binder-length-max", type=int, default=None)
    run_parser.add_argument("--binder-length", type=int, action="append", default=None)
    run_parser.add_argument("--campaign-seed", type=int, default=None)
    run_parser.add_argument("--core", default=DEFAULT_CORE_PROFILE)
    run_parser.add_argument("--modality", action="append", default=None)
    run_parser.add_argument("--property", action="append", default=None)
    run_parser.add_argument("--set", action="append", default=None)
    run_parser.add_argument("--allow-unlisted-setting", action="append", default=None)
    run_parser.add_argument("--metadata", type=Path, default=None)
    run_parser.add_argument("--binder-name", default=None)
    run_parser.add_argument("--campaign-name", default=None)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--replace", action="store_true")
    run_parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)

    parse_parser = subparsers.add_parser("parse")
    add_stage_arguments(parse_parser)
    parse_parser.add_argument("--manifest-path", type=Path, default=None)
    parse_parser.add_argument("--generator-id", default=DEFAULT_GENERATOR_ID)
    parse_parser.add_argument("--pose-subdir", default=DEFAULT_POSE_SUBDIR)
    parse_parser.add_argument("--sequence-subdir", default=DEFAULT_SEQUENCE_SUBDIR)
    parse_parser.add_argument("--target-state-suffix", default=None)
    return parser


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        refuse_hosted_arguments(values)
        args = parse_arguments(values)
        if args.command == "toolcheck":
            return toolcheck(args)
        if args.command == "parse":
            return parse_outputs(args)
        return run(args)
    except AdapterError as exc:
        print(f"{TOOL_LABEL} adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
