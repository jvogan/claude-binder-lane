"""Run a qualification canary against a campaign's own target and write a roster receipt.

A de novo campaign against a new target cannot run until its model roster is qualified,
and the packaged fal clients each write a receipt of their own under different names rather
than the one `claude_binder.qualify.build_row` reads.

This module bridges that gap. It prepares inputs bound to the campaign's target, calls the
packaged client for one adapter, and writes one receipt in the roster's vocabulary.

**What it does not do.** It does not decide that a model is qualified. It measures, and
`qualify.build_row` decides. It never invents evidence: a field it cannot measure is absent
from the receipt, and an absent required field is what makes qualification refuse.

**Replay is not qualification.** `--replay-from` reads a recorded response instead of calling
a provider, so the processing path can be tested with no account and no spend. A replayed
receipt carries `evidence_mode: recorded-replay`, and `qualify` refuses to mark such a row
PASS. Attaching a campaign's target identity to a canary that consumed a shipped fixture
would turn fixture success into target qualification, which is the forged gate this package
refused when it set `production_scoring` false rather than borrow another target's controls.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .paths import child_process_environment, package_root

SCHEMA_VERSION = 1

# The runtime validator accepts 2 to 4 designs. A single-design canary cannot produce a
# runtime-accepted roster no matter how its receipt is shaped, which is why the count is
# bounded here rather than left to the caller.
MINIMUM_CANARY_COUNT = 2
MAXIMUM_CANARY_COUNT = 4

EVIDENCE_MODE_LIVE = "live-dispatch"
EVIDENCE_MODE_REPLAY = "recorded-replay"

SCOPE_TARGET = "campaign-target"
SCOPE_FIXTURE = "fixture-contract"

# Each packaged client writes its own receipt vocabulary. These three names are the whole
# difference from the roster's vocabulary for RFdiffusion3 and ProteinMPNN. Renaming is not
# the fix on its own, because the fields the roster needs and no client writes are counts,
# output paths and assertions, which only execution can supply.
CLIENT_FIELD_RENAMES = {
    "checkpoint_sha256": "weights_sha256",
    "device": "gpu_type",
    "client_wall_seconds": "wall_clock_s",
}


class CanaryError(Exception):
    """A canary could not be prepared, dispatched or verified."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError(f"could not read {path}: {exc}") from exc


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# `--session` and `--receipt` are output paths. Refuse to overwrite an occupied path
# unless it has this runner's format marker or a narrow legacy canary shape.
# Generic schema keys are insufficient because unrelated campaign receipts share them.
RECEIPT_FORMAT = "claude-binder-canary-receipt-v1"
SESSION_FORMAT = "claude-binder-canary-session-v1"
LEGACY_RECEIPT_KEYS = frozenset(
    {"schema_version", "adapter_id", "evidence_mode", "qualification_scope"}
)
LEGACY_SESSION_KEYS = frozenset(
    {"schema_version", "adapters", "qualifies", "receipts_completed"}
)
RECEIPT_KEYS = LEGACY_RECEIPT_KEYS
SESSION_KEYS = LEGACY_SESSION_KEYS


def _is_own_output(path: Path, required: frozenset[str], expected_format: str) -> bool:
    """True when `path` already holds a document this runner wrote.

    A zero-byte file counts, because a run killed mid-write leaves one and its retry must not
    need a flag to finish what it started. Anything unreadable counts as foreign.
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
    if not isinstance(existing, Mapping):
        return False
    if existing.get("format") == expected_format:
        return True
    return required <= set(existing)


def _refuse_occupied_output(
    path: Path, *, flag: str, required: frozenset[str], expected_format: str, replace: bool
) -> None:
    """Refuse an operator-named output path that already holds someone else's file.

    Call this before the work, not at the write. A canary's receipt path is checked before the
    dispatch that costs money, so a refusal never arrives after the operator has paid.
    """
    if path.is_symlink():
        raise CanaryError(
            f"{flag} is an output path and is overwritten, and {path} is a symbolic link. "
            "Writing through it would replace the file it points at. Name the real path."
        )
    if not path.exists():
        return
    if not path.is_file():
        raise CanaryError(
            f"{flag} is an output path and is overwritten, and {path} is not a regular file. "
            "Name a path that does not exist, or a file this runner may replace."
        )
    if replace or _is_own_output(path, required, expected_format):
        return
    raise CanaryError(
        f"{flag} is an output path and is overwritten, and {path} already holds a file this "
        f"runner did not write. Name a path that does not exist, or pass --replace to "
        f"overwrite it."
    )


def _write_output_json(
    path: Path,
    document: Any,
    *,
    flag: str,
    required: frozenset[str],
    expected_format: str,
    replace: bool,
) -> None:
    """Write `document` to an operator-named output path, refusing to destroy a foreign file."""
    _refuse_occupied_output(
        path, flag=flag, required=required, expected_format=expected_format, replace=replace
    )
    stamped = dict(document) if isinstance(document, Mapping) else document
    if isinstance(stamped, dict):
        stamped.setdefault("format", expected_format)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(stamped, indent=2, sort_keys=True) + "\n"
    # O_NOFOLLOW closes the window the guard cannot: a symbolic link swapped into place between
    # the check and the open would otherwise be followed to whatever it names.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    try:
        handle = os.open(path, flags, 0o644)
    except OSError as error:
        raise CanaryError(f"{flag} could not be written at {path}: {error}") from error
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(payload)


def _primary_target(config: Mapping[str, Any]) -> Mapping[str, Any]:
    targets = config.get("targets")
    if not isinstance(targets, list) or not targets:
        raise CanaryError("the campaign declares no targets")
    primary = [item for item in targets if isinstance(item, Mapping) and item.get("role") == "primary"]
    chosen = primary[0] if primary else targets[0]
    if not isinstance(chosen, Mapping):
        raise CanaryError("the campaign's primary target is not an object")
    return chosen


def _target_structure(config: Mapping[str, Any]) -> tuple[Path, str]:
    """The campaign's own target structure and the digest the configuration attests.

    A canary that folded a shipped fixture and then wore the campaign's target identity
    would be a forged gate. Binding to this file is what makes the receipt about this
    target.
    """
    target = _primary_target(config)
    for key in ("structure_source_path", "runtime_structure_path", "structure_path"):
        value = target.get(key)
        if isinstance(value, str) and value and Path(value).is_file():
            path = Path(value)
            return path, _sha256_file(path)
    raise CanaryError(
        "the campaign's primary target has no readable structure; set "
        "targets[0].structure_source_path to the file this canary should bind to"
    )


def target_sequence(config: Mapping[str, Any], structure: Path) -> str:
    """The campaign's target chain sequence, read from the campaign or from the structure.

    Never an argv default. An empty default reached the ESMFold2-Fast client as a FASTA with
    no residues, and that client refuses a file that is not a single uppercase sequence, so
    the canary would have failed on the operator's first paid attempt. The sequence is a
    property of the campaign, so the runner derives it rather than asking for it.
    """
    context = config.get("context")
    if isinstance(context, Mapping):
        declared = context.get("target_sequence")
        if isinstance(declared, str) and declared.strip():
            return declared.strip()
    from .adapters.target_prep_adapter import load_atoms, residue_letter

    try:
        atoms, _metadata = load_atoms(structure)
    except Exception as exc:  # noqa: BLE001
        raise CanaryError(f"could not read the target sequence from {structure}: {exc}") from exc
    target = _primary_target(config)
    chains = target.get("chains")
    wanted = None
    if isinstance(chains, list) and chains and isinstance(chains[0], Mapping):
        wanted = chains[0].get("chain_id")
    sequence = "".join(
        residue_letter(atom.residue_name)
        for atom in atoms
        if atom.name == "CA" and (wanted is None or atom.chain_id == wanted)
    )
    if not sequence:
        raise CanaryError(
            f"the target structure yielded no residues for chain {wanted!r}: {structure}"
        )
    return sequence


def _adapter(config: Mapping[str, Any], adapter_id: str) -> Mapping[str, Any]:
    for item in config.get("adapters", []):
        if isinstance(item, Mapping) and item.get("adapter_id") == adapter_id:
            return item
    raise CanaryError(f"the campaign declares no adapter {adapter_id}")


def production_inference_arguments(adapter: Mapping[str, Any]) -> dict[str, str]:
    """Literal flag and value pairs the adapter's production command declares.

    A canary that used its own defaults would be measuring something the campaign never
    runs, and `flags_match_production` would be an assertion about nothing. Reading them
    from the production template is what makes that field mean something.

    Only literal values are returned. A token like `{{count}}` varies per run and is not a
    production flag setting.
    """
    template = adapter.get("command_argv_template")
    if not isinstance(template, list):
        return {}
    found: dict[str, str] = {}
    for index, token in enumerate(template):
        if not isinstance(token, str) or not token.startswith("--"):
            continue
        if index + 1 >= len(template):
            continue
        value = template[index + 1]
        if isinstance(value, str) and value and not value.startswith("--") and "{{" not in value:
            found[token] = value
    return found


#: The seed a canary sends when neither the operator nor the adapter's production command
#: names one. RFdiffusion3 and ESMFold2-Fast declare no `--seed` in their production argv and
#: both accept this value.
FALLBACK_CANARY_SEED = 0


def _resolve_seed(requested: int | None, adapter: Mapping[str, Any]) -> int:
    """Return the seed this canary sends, preferring the adapter's own production value.

    `production_inference_arguments` says above that a canary using its own defaults measures
    something the campaign never runs. That reasoning was never applied to `--seed`. The
    parser defaulted to 0, no shipped profile passes the flag, and the ProteinMPNN deployment
    requires a seed of at least 1, so it answered every canary with
    `HTTP 422: Input should be greater than or equal to 1`. No shipped profile could qualify
    that arm, and `qualify --confirm-cost` failed the designer and then the predictor that
    consumes its sequences.

    An explicit flag still wins, because an operator naming a seed means it.
    """
    if requested is not None:
        return requested
    declared = production_inference_arguments(adapter).get("--seed")
    if isinstance(declared, str) and declared.strip().lstrip("-").isdigit():
        return int(declared)
    return FALLBACK_CANARY_SEED


def deployment_identity(fal_url: str | None) -> str | None:
    """Name the deployment this canary called, without carrying a credential.

    The roster spells an unconfigured arm `fal:unconfigured-<model>`, so a configured one
    names the endpoint it reached. A URL carries no secret, but the query string might, so
    only the host and path are kept.
    """
    if not isinstance(fal_url, str) or not fal_url.strip():
        return None
    from urllib.parse import urlsplit

    parts = urlsplit(fal_url.strip())
    if not parts.netloc:
        return None
    return f"fal:{parts.netloc}{parts.path}".rstrip("/")


def build_assertions(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the container built and its packages imported, read from the receipt.

    A provider that answered with its own environment identity and the identity of the
    weights it loaded ran a container that built and packages that imported. Nothing else
    can produce those two values, so they are the basis, and the basis is recorded beside
    the assertion.

    A receipt missing either one derives nothing. The same rule applies to a receipt this
    package dispatched and to one a frame harvested, because the facts it reads are the
    provider's either way.
    """
    identity = receipt.get("environment_identity")
    weights = receipt.get("weights_sha256") or receipt.get("checkpoint_sha256")
    if not (isinstance(identity, str) and identity and isinstance(weights, str) and weights):
        return {}
    return {
        "container_build": True,
        "package_imports": True,
        "build_evidence": (
            "the provider returned environment_identity and a loaded-weights digest, which a "
            "container that did not build and packages that did not import cannot produce"
        ),
    }


def _deployment_assertions(
    adapter: Mapping[str, Any],
    receipt: Mapping[str, Any],
    arguments: argparse.Namespace,
    sent: Mapping[str, str],
    client_flags: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Roster assertions the runner can derive, each with the basis it derived them from.

    A field it cannot derive is absent, and an absent required field is what makes
    `qualify.build_row` refuse. Nothing here is asserted from a default.

    `client_flags` is the option surface of the client this canary called, from
    `client_option_surface`. It bounds `flags_match_production` to the production settings
    the canary could actually send.
    """
    derived: dict[str, Any] = {}
    resources = adapter.get("resources")
    image = resources.get("container_image_digest") if isinstance(resources, Mapping) else None
    if isinstance(image, str) and image:
        derived["image_id"] = image
    deployment = deployment_identity(arguments.fal_url)
    if deployment:
        derived["deployment_id"] = deployment

    # The provider answered with its own environment identity and the digest of the weights
    # it loaded. A container that did not build and packages that did not import produce
    # neither. Asserted from the response, and the basis is recorded beside the assertion.
    derived.update(build_assertions(receipt))

    production = production_inference_arguments(adapter)
    if production and sent:
        # `flags_match_production` is a claim about the production settings this canary can
        # reach, so it holds only when the comparison covers all of them. Comparing the
        # overlap alone answered True while saying nothing about the rest: production's
        # `--sampling-temp 0.1` moved to 0.9 kept answering True, because ProteinMPNN's
        # canary sent `--seed` and nothing else. That asserts the field from an absence,
        # which the contract at the top of this function forbids.
        #
        # `client_flags` bounds the claim to what the canary could send. A production argv
        # also carries stage wiring, and a client with no such parameter cannot send it and
        # cannot change the provider call by omitting it. Those flags are named in the
        # evidence and excluded from the comparison.
        reachable = {
            flag: value
            for flag, value in production.items()
            if client_flags is None or flag in client_flags
        }
        unreachable = sorted(set(production) - set(reachable))
        unsent = sorted(flag for flag in reachable if flag not in sent)
        if unsent:
            derived["production_flags_evidence"] = (
                "flags_match_production is not derivable: this canary sent no value for "
                + ", ".join(unsent)
                + ", which this adapter's command_argv_template declares and this client "
                "accepts. Send every such flag to derive the field."
            )
        elif reachable:
            derived["flags_match_production"] = all(
                str(sent[flag]) == str(value) for flag, value in reachable.items()
            )
            evidence = (
                "compared against every literal value in this adapter's own "
                "command_argv_template that this client accepts: "
                + ", ".join(f"{flag}={value}" for flag, value in sorted(reachable.items()))
            )
            if unreachable:
                evidence += (
                    "; outside this client's option surface and not compared: "
                    + ", ".join(unreachable)
                )
            derived["production_flags_evidence"] = evidence
    return derived


@lru_cache(maxsize=None)
def client_option_surface(client: Path, subcommand: str = "run") -> frozenset[str]:
    """The `--flag` names this packaged client accepts, read from its own `run --help`.

    A production argv mixes inference settings with stage wiring. ProteinMPNN's declares
    `--backbone-stage-id`, `--backbone-artifact-id` and `--designer-id`, which tell the
    campaign adapter where to read backbones from; the canary reads its backbones from the
    preceding arm and its client has no such parameter. Deciding which of those a canary
    "should" send by reading their names would be a judgement call. The client's own option
    list is the mechanical answer, and it stays correct when a client gains a flag.
    """
    completed = subprocess.run(
        [sys.executable, str(client), subcommand, "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        env=child_process_environment(),
    )
    if completed.returncode != 0:
        raise CanaryError(
            f"packaged client {client.name} did not answer `{subcommand} --help` "
            f"(exit {completed.returncode}): {completed.stderr[-500:]}"
        )
    return frozenset(re.findall(r"--[a-z0-9][a-z0-9-]*", completed.stdout))


def production_value(adapter: Mapping[str, Any], flag: str, fallback: str) -> str:
    """The literal value production declares for `flag`, or `fallback` when it declares none.

    This is how a canary sends what the campaign runs rather than what its client defaults
    to, the same rule `_resolve_seed` applies to `--seed`.
    """
    declared = production_inference_arguments(adapter).get(flag)
    return declared if isinstance(declared, str) and declared else fallback


def _client_path(name: str) -> Path:
    path = package_root() / "clients" / name
    if not path.is_file():
        raise CanaryError(f"packaged client is missing: {path}")
    return path


def _run_client(
    argv: Sequence[str],
    *,
    cwd: Path,
    executor: Callable[..., Any] = subprocess.run,
) -> tuple[int, str, str, float]:
    """Run one packaged client and return its result and measured wall time.

    The wall time measured here brackets the provider call from above. It is the number the
    roster's `s/design` is derived from, and it is honest about what it contains: process
    start, the request, and the response written to disk.
    """
    started = time.monotonic()
    try:
        completed = executor(
            list(argv),
            cwd=str(cwd),
            env=child_process_environment(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return 127, "", str(exc), round(time.monotonic() - started, 3)
    elapsed = round(time.monotonic() - started, 3)
    return (
        int(getattr(completed, "returncode", 1)),
        str(getattr(completed, "stdout", "") or ""),
        str(getattr(completed, "stderr", "") or ""),
        elapsed,
    )


def normalize_client_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Rename a packaged client's receipt fields into the roster's vocabulary.

    The client's own spelling is kept alongside the roster's. Dropping it would make the
    receipt harder to compare against the client's output, and the roster reader ignores
    what it does not require.

    Nothing here measures. Every value written is one the receipt already carries, read out
    of the field the deployment put it in.
    """
    normalized = dict(receipt)
    for source, destination in CLIENT_FIELD_RENAMES.items():
        if source in normalized and destination not in normalized:
            normalized[destination] = normalized[source]
    _normalize_resolved_environment(normalized)
    return normalized


def _normalize_resolved_environment(receipt: dict[str, Any]) -> None:
    """Map the vocabulary of a deployment that resolves model repositories.

    ESMFold2-Fast reports its environment as an `environment` object plus an
    `environment_revision` string, and pins its weights as `resolved_snapshots` rather than
    a checkpoint digest, because a resolved repository is many files and has none. The three
    flat renames above do not reach any of it, so a live run that measured the GPU, the
    environment and both model revisions arrived at the roster carrying none of them and the
    row stayed PENDING on facts it already held.

    A deployment that reports no `environment` object is untouched.
    """
    environment = receipt.get("environment")
    if isinstance(environment, Mapping):
        gpu_name = environment.get("gpu_name")
        if isinstance(gpu_name, str) and gpu_name and "gpu_type" not in receipt:
            receipt["gpu_type"] = gpu_name
    revision = receipt.get("environment_revision")
    if isinstance(revision, str) and revision and "environment_identity" not in receipt:
        receipt["environment_identity"] = revision

    snapshots = receipt.get("resolved_snapshots")
    if not isinstance(snapshots, Mapping) or not snapshots:
        return
    pins = []
    for repository, resolved in sorted(snapshots.items()):
        if not isinstance(resolved, Mapping):
            continue
        pinned = resolved.get("revision")
        if isinstance(pinned, str) and pinned:
            pins.append(f"{repository}@{pinned}")
    if not pins:
        return
    if "weights_revision" not in receipt:
        receipt["weights_revision"] = "; ".join(pins)
    # `runtime_validator` sends a row down the revision branch precisely when the digest
    # records a deliberate absence, and it fails a row that records the absence and nothing
    # else. The absence is the true statement here: this deployment resolved repositories
    # and holds no single checkpoint file to digest. `weights_revision` above carries the
    # identity the branch then checks against the profile's own pins.
    if "weights_sha256" not in receipt and "checkpoint_sha256" not in receipt:
        receipt["weights_sha256"] = (
            "not_applicable: the deployment resolves model repositories by revision and "
            "holds no single checkpoint file to digest; see weights_revision"
        )


def _gather_outputs(paths: Sequence[Path], evidence_root: Path) -> tuple[list[str], list[str]]:
    """Return the root-relative output paths and their digests, in one order."""
    relative: list[str] = []
    digests: list[str] = []
    for path in paths:
        resolved = path.resolve()
        try:
            relative.append(resolved.relative_to(evidence_root.resolve()).as_posix())
        except ValueError as exc:
            raise CanaryError(
                f"canary output escapes the evidence root: {resolved}"
            ) from exc
        digests.append(_sha256_file(resolved))
    return relative, digests


def unpack_rfdiffusion3_structures(out_dir: Path) -> list[Path]:
    """Decompress the `.cif.gz` files the RFdiffusion3 client wrote, and verify each.

    The client hashes and writes the compressed bytes and never decompresses them, so a
    structure reader downstream has nothing to read. Both digests are kept: the compressed
    one is what the provider attested, the decompressed one is what a reader consumes.
    """
    unpacked: list[Path] = []
    archives = sorted(out_dir.rglob("*.cif.gz"))
    if not archives:
        # A recorded response tree holds the structures already decompressed, because that
        # is the form a reader consumes. There is nothing to unpack and nothing to verify
        # about a compression the recording did not keep.
        return [
            path for path in sorted(out_dir.rglob("*.cif")) if not path.name.endswith(".cif.gz")
        ]
    for archive in archives:
        destination = archive.with_suffix("")
        try:
            with gzip.open(archive, "rb") as handle:
                payload = handle.read()
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            raise CanaryError(f"RFdiffusion3 output is not readable gzip: {archive}: {exc}") from exc
        if not payload.strip():
            raise CanaryError(f"RFdiffusion3 output decompressed to nothing: {archive}")
        destination.write_bytes(payload)
        unpacked.append(destination)
    return unpacked


def _assert_structure_readable(path: Path) -> dict[str, Any]:
    """Confirm a structure parses and report what it actually contains.

    A file that exists is not evidence that a model produced a structure. The chain letters
    and residue count are read from the file by the package's own production parser, never
    from the request. Two chain mappings disagreeing is the class of defect this package has
    been bitten by, and a canary that reported the requested chains rather than the returned
    ones would hide exactly that.
    """
    from .adapters.target_prep_adapter import load_atoms

    try:
        atoms, _metadata = load_atoms(path)
    except Exception as exc:  # noqa: BLE001
        raise CanaryError(f"structure did not parse: {path}: {exc}") from exc
    if not atoms:
        raise CanaryError(f"structure has no atom records: {path}")
    coordinates_finite = all(
        all(isinstance(value, float) and value == value and abs(value) != float("inf")
            for value in (atom.x, atom.y, atom.z))
        for atom in atoms
    )
    if not coordinates_finite:
        raise CanaryError(f"structure has non-finite coordinates: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "chain_ids": sorted({atom.chain_id for atom in atoms if atom.chain_id}),
        "residue_count": len({(atom.chain_id, atom.residue_number) for atom in atoms}),
        "atom_count": len(atoms),
    }


def _fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = ""
    sequence: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if sequence:
                records.append((header, "".join(sequence)))
                sequence = []
            header = line[1:].strip()
        elif line.strip():
            sequence.append(line.strip())
    if sequence:
        records.append((header, "".join(sequence)))
    return records


def designed_sequences(path: Path, *, sequences_per_backbone: int) -> list[tuple[str, str]]:
    """The designed records in a ProteinMPNN FASTA, excluding the native first record.

    `proteinmpnn_designer` slices `records[1 : sequences_per_backbone + 1]` for exactly this
    reason. Counting the native record as a design would inflate the canary's design count
    by one per backbone and quietly relax the runtime validator's 2 to 4 window.
    """
    records = _fasta_records(path)
    if len(records) < 2:
        raise CanaryError(
            f"ProteinMPNN FASTA has no designed record beside the native one: {path}. A raw "
            "client response carries the native sequence first and the designs after it. A "
            "file with one record is an already-extracted binder sequence, which is what the "
            "shipped canary fixtures hold, so it cannot replay this client's own response."
        )
    return records[1 : sequences_per_backbone + 1]


# --------------------------------------------------------------------------- adapters


def _assert_upstream_matches(
    receipt_path: Path,
    target_digest: str,
    arguments: argparse.Namespace,
    *,
    producer: str,
) -> None:
    """Refuse an upstream artifact whose receipt names another target or another mode.

    Finding a file at the expected path is not provenance. A directory left behind by a
    replay, or by a canary for a different target, sits exactly where a live one would. The
    upstream receipt is what says which target it was bound to, so it is read rather than
    the directory listing trusted.
    """
    if not receipt_path.is_file():
        raise CanaryError(
            f"{producer} left artifacts with no receipt at {receipt_path}, so nothing says "
            "which target they were bound to"
        )
    receipt = _read_json(receipt_path)
    if not isinstance(receipt, Mapping):
        raise CanaryError(f"{producer} receipt is not an object: {receipt_path}")
    upstream_digest = receipt.get("campaign_target_structure_sha256")
    if upstream_digest != target_digest:
        raise CanaryError(
            f"{producer} was run against target {upstream_digest} and this canary is bound to "
            f"{target_digest}, so its output would qualify the wrong target"
        )
    wanted = EVIDENCE_MODE_REPLAY if arguments.replay_from is not None else EVIDENCE_MODE_LIVE
    found = receipt.get("evidence_mode")
    if found != wanted:
        raise CanaryError(
            f"{producer} evidence_mode is {found!r} and this canary is {wanted!r}; a live arm "
            "may not consume replayed input and a replay may not consume live input"
        )


def _prepare_inputs(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Copy the campaign's own target under the run directory and verify it survived.

    The digest is taken before and after the copy. A canary that silently folded a
    truncated copy would still write a receipt, and the receipt would name the campaign's
    target.
    """
    source, source_digest = _target_structure(config)
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    destination = inputs / source.name
    shutil.copyfile(source, destination)
    copied_digest = _sha256_file(destination)
    if copied_digest != source_digest:
        raise CanaryError(
            f"the target structure changed while being copied: {source_digest} became {copied_digest}"
        )
    return {
        "target_structure_source_path": str(source),
        "target_structure_path": str(destination),
        "campaign_target_structure_sha256": source_digest,
    }


def _replay(replay_from: Path, out_dir: Path) -> None:
    """Copy a recorded response tree in place of calling a provider."""
    if not replay_from.is_dir():
        raise CanaryError(f"replay source is not a directory: {replay_from}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(replay_from.rglob("*")):
        if item.is_file():
            destination = out_dir / item.relative_to(replay_from)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, destination)


def _rfdiffusion3(
    arguments: argparse.Namespace,
    config: Mapping[str, Any],
    adapter: Mapping[str, Any],
    prepared: Mapping[str, Any],
    executor: Callable[..., Any],
) -> dict[str, Any]:
    from . import rfd3_specification
    from .adapters.target_prep_adapter import load_atoms

    run_dir = arguments.run_dir
    out_dir = run_dir / "raw"
    specification_path = run_dir / "inputs" / "rfd3-specification.json"
    if True:
        # Always derived from this campaign's own target, never borrowed and never inherited
        # from whatever the run directory already held. Substituting the shipped PD-L1
        # specification, or reusing a stale one left by an earlier target, would make the
        # canary a canary for that other target wearing this one's identity.
        try:
            atoms, _ = load_atoms(Path(prepared["target_structure_path"]))
            document = rfd3_specification.derive_specification(
                dict(config),
                str(_primary_target(config).get("target_id", "")),
                rfd3_specification.residue_records_from_atoms(atoms),
            )
        except Exception as exc:  # noqa: BLE001
            raise CanaryError(
                "could not derive an RFdiffusion3 specification from this campaign, so the "
                f"canary would have had to borrow a shipped one: {exc}"
            ) from exc
        if specification_path.is_file():
            existing = _read_json(specification_path)
            if existing != document:
                raise CanaryError(
                    f"the run directory already holds a different RFdiffusion3 specification "
                    f"at {specification_path}; it was not derived from this campaign's target, "
                    "so the canary refuses rather than folding somebody else's design problem"
                )
        _write_json(specification_path, document)
    receipt_path = run_dir / "raw" / "client-receipt.json"
    # Production values come from the adapter's own command template, so
    # `flags_match_production` compares against what the campaign really runs. An argv
    # default would make that assertion be about the runner rather than the campaign.
    production = production_inference_arguments(adapter)
    sent_arguments = {
        "--step-scale": production.get("--step-scale", str(arguments.step_scale)),
        "--gamma-0": production.get("--gamma-0", str(arguments.gamma_0)),
        "--n-batches": production.get("--n-batches", "1"),
    }
    if arguments.replay_from is not None:
        _replay(arguments.replay_from, out_dir)
        exit_code, stdout, stderr, elapsed = 0, "", "", 0.0
    else:
        argv = [
            sys.executable,
            str(_client_path("fal_rfdiffusion3_client.py")),
            "run",
            "--fal-url", arguments.fal_url,
            "--out-dir", str(out_dir),
            "--specification", str(specification_path),
            "--input-structure", str(prepared["target_structure_path"]),
            "--receipt", str(receipt_path),
            "--diffusion-batch-size", str(arguments.canary_count),
            "--n-batches", sent_arguments["--n-batches"],
            "--seed", str(arguments.seed),
            "--step-scale", sent_arguments["--step-scale"],
            "--gamma-0", sent_arguments["--gamma-0"],
        ]
        exit_code, stdout, stderr, elapsed = _run_client(argv, cwd=run_dir, executor=executor)
        if exit_code != 0:
            raise CanaryError(f"RFdiffusion3 client exited {exit_code}: {stderr[-2000:]}")
    structures = unpack_rfdiffusion3_structures(out_dir)
    if len(structures) != arguments.canary_count:
        raise CanaryError(
            f"RFdiffusion3 returned {len(structures)} backbones, not the {arguments.canary_count} requested"
        )
    observed = [_assert_structure_readable(path) for path in structures]
    receipt = normalize_client_receipt(_read_json(receipt_path)) if receipt_path.is_file() else {}
    receipt.update(
        {
            "n_designs": len(structures),
            "output_shape": "mmCIF, one designed backbone per requested design",
            "structures": observed,
        }
    )
    # `sequence_adapter_consumed` and `sequence_output_count` are cross-arm facts. This arm
    # cannot know whether a sequence adapter consumed its backbones, so it says nothing and
    # `finalize` fills them once it has verified the lineage. Writing false here would look
    # like an assertion and would refuse the arm forever.
    receipt.update(
        _deployment_assertions(
            adapter,
            receipt,
            arguments,
            sent_arguments,
            client_option_surface(_client_path("fal_rfdiffusion3_client.py")),
        )
    )
    return {
        "receipt": receipt,
        "outputs": structures,
        "wall_clock_s": receipt.get("wall_clock_s", elapsed),
        "stdout": stdout,
    }


def _proteinmpnn(
    arguments: argparse.Namespace,
    config: Mapping[str, Any],
    adapter: Mapping[str, Any],
    prepared: Mapping[str, Any],
    executor: Callable[..., Any],
) -> dict[str, Any]:
    """One invocation per backbone, because the client designs on one structure at a time."""
    run_dir = arguments.run_dir
    upstream = arguments.evidence_root / "canary" / "rfdiffusion-generator"
    backbones = sorted((upstream / "raw").rglob("*.cif"))
    if backbones:
        _assert_upstream_matches(
            upstream / "receipt.json",
            prepared["campaign_target_structure_sha256"],
            arguments,
            producer="rfdiffusion-generator",
        )
    elif arguments.replay_from is not None:
        backbones = sorted(arguments.replay_from.rglob("*.cif"))
    if len(backbones) < arguments.canary_count:
        raise CanaryError(
            f"ProteinMPNN needs {arguments.canary_count} target-bound backbones and found "
            f"{len(backbones)}; run the rfdiffusion-generator canary first, because designing "
            "on a shipped backbone would qualify a fixture rather than this target"
        )
    design_chain = str(((config.get("binder") or {}).get("binder_chain_id")) or "A")
    # Production sets the sampling temperature, so the canary sends production's value
    # rather than the client default. They agree at 0.1 today; reading it here keeps the
    # canary measuring the campaign if production ever changes it.
    sampling_temp = production_value(adapter, "--sampling-temp", "0.1")
    produced: list[Path] = []
    receipts: list[dict[str, Any]] = []
    total_elapsed = 0.0
    for index, backbone in enumerate(backbones[: arguments.canary_count]):
        out_dir = run_dir / "raw" / f"backbone-{index}"
        receipt_path = out_dir / "client-receipt.json"
        if arguments.replay_from is not None:
            _replay(arguments.replay_from, out_dir)
        else:
            argv = [
                sys.executable,
                str(_client_path("fal_mpnn_client.py")),
                "run",
                "--fal-url", arguments.fal_url,
                "--out-dir", str(out_dir),
                "--input-structure", str(backbone),
                "--receipt", str(receipt_path),
                "--seed", str(arguments.seed),
                "--design-chain", design_chain,
                "--sequences-per-backbone", "1",
                "--sampling-temp", sampling_temp,
            ]
            exit_code, _stdout, stderr, elapsed = _run_client(argv, cwd=run_dir, executor=executor)
            total_elapsed += elapsed
            if exit_code != 0:
                raise CanaryError(f"ProteinMPNN client exited {exit_code}: {stderr[-2000:]}")
        fastas = sorted(out_dir.rglob("*.fa*"))
        if not fastas:
            raise CanaryError(f"ProteinMPNN wrote no FASTA under {out_dir}")
        designed = designed_sequences(fastas[0], sequences_per_backbone=1)
        binder = out_dir / f"backbone-{index}.binder.fasta"
        binder.write_text(f">{designed[0][0]}\n{designed[0][1]}\n", encoding="utf-8")
        produced.append(binder)
        if receipt_path.is_file():
            receipts.append(normalize_client_receipt(_read_json(receipt_path)))
    if len(produced) != arguments.canary_count:
        raise CanaryError(
            f"ProteinMPNN produced {len(produced)} designed sequences, not the "
            f"{arguments.canary_count} requested"
        )
    receipt = dict(receipts[0]) if receipts else {}
    receipt.update(
        {
            "n_designs": len(produced),
            "output_shape": "FASTA, one designed binder sequence per backbone",
            "sequence_output_count": len(produced),
            # This arm is the sequence adapter, so it consumed its own input by running.
            "sequence_adapter_consumed": True,
            "per_invocation_receipts": receipts,
            "consumed_backbones": [str(path) for path in backbones[: arguments.canary_count]],
        }
    )
    receipt.update(
        _deployment_assertions(
            adapter,
            receipt,
            arguments,
            # Every production literal this client accepts is sent, so the comparison covers
            # what the canary can reach. The three the production argv declares and this
            # client has no parameter for are stage wiring for the campaign adapter; they are
            # named in `production_flags_evidence` and excluded from the claim.
            {
                "--seed": str(arguments.seed),
                "--sequences-per-backbone": "1",
                "--design-chain": design_chain,
                "--sampling-temp": sampling_temp,
            },
            client_option_surface(_client_path("fal_mpnn_client.py")),
        )
    )
    return {
        "receipt": receipt,
        "outputs": produced,
        "sequence_paths": produced,
        "wall_clock_s": total_elapsed or receipt.get("wall_clock_s"),
        "stdout": "",
    }


def _esmfold2_fast(
    arguments: argparse.Namespace,
    config: Mapping[str, Any],
    adapter: Mapping[str, Any],
    prepared: Mapping[str, Any],
    executor: Callable[..., Any],
) -> dict[str, Any]:
    """One invocation per fold, because the client predicts one complex at a time."""
    run_dir = arguments.run_dir
    upstream = arguments.evidence_root / "canary" / "proteinmpnn-designer"
    binders = sorted((upstream / "raw").rglob("*.binder.fasta"))
    if binders:
        _assert_upstream_matches(
            upstream / "receipt.json",
            prepared["campaign_target_structure_sha256"],
            arguments,
            producer="proteinmpnn-designer",
        )
    elif arguments.replay_from is not None:
        binders = sorted(arguments.replay_from.rglob("*.fasta"))
    if len(binders) < arguments.canary_count:
        raise CanaryError(
            f"ESMFold2-Fast needs {arguments.canary_count} designed binder sequences and found "
            f"{len(binders)}; run the proteinmpnn-designer canary first"
        )
    target_fasta = run_dir / "inputs" / "target.fasta"
    target_fasta.parent.mkdir(parents=True, exist_ok=True)
    target_fasta.write_text(
        f">{_primary_target(config).get('target_id', 'target')}\n"
        f"{target_sequence(config, Path(prepared['target_structure_path']))}\n",
        encoding="utf-8",
    )
    max_seconds = production_value(adapter, "--max-seconds", "1700")
    timeout_seconds = production_value(adapter, "--timeout-seconds", "1850")
    produced: list[Path] = []
    responses: list[dict[str, Any]] = []
    total_elapsed = 0.0
    for index, binder in enumerate(binders[: arguments.canary_count]):
        out_dir = run_dir / "raw" / f"fold-{index}"
        if arguments.replay_from is not None:
            _replay(arguments.replay_from, out_dir)
        else:
            argv = [
                sys.executable,
                str(_client_path("fal_esmfold2_fast_client.py")),
                "predict",
                "--fal-url", arguments.fal_url,
                "--out-dir", str(out_dir),
                "--binder-fasta", str(binder),
                "--target-fasta", str(target_fasta),
                "--seed", str(arguments.seed),
                # Production bounds this predictor with its own two limits, so the canary
                # sends them rather than the client defaults.
                "--max-seconds", max_seconds,
                "--timeout-seconds", timeout_seconds,
            ]
            exit_code, _stdout, stderr, elapsed = _run_client(argv, cwd=run_dir, executor=executor)
            total_elapsed += elapsed
            if exit_code != 0:
                raise CanaryError(f"ESMFold2-Fast client exited {exit_code}: {stderr[-2000:]}")
        predicted = sorted(out_dir.rglob("predicted.cif"))
        if not predicted:
            raise CanaryError(f"ESMFold2-Fast wrote no predicted.cif under {out_dir}")
        produced.append(predicted[0])
        response = out_dir / "response.json"
        if response.is_file():
            responses.append(_read_json(response))
    if len(produced) != arguments.canary_count:
        raise CanaryError(
            f"ESMFold2-Fast produced {len(produced)} predictions, not the "
            f"{arguments.canary_count} requested"
        )
    observed = [_assert_structure_readable(path) for path in produced]
    first = responses[0] if responses else {}
    receipt = normalize_client_receipt(first if isinstance(first, Mapping) else {})
    receipt.update(
        {
            "n_designs": len(produced),
            "output_shape": "mmCIF complex with PAE and pLDDT sidecars, one per design",
            "structures": observed,
            # This arm folded designed sequences, so a sequence adapter's output was consumed.
            "sequence_adapter_consumed": True,
            "sequence_output_count": len(binders[: arguments.canary_count]),
            "consumed_sequences": [str(path) for path in binders[: arguments.canary_count]],
        }
    )
    receipt.update(
        _deployment_assertions(
            adapter,
            receipt,
            arguments,
            {
                "--seed": str(arguments.seed),
                "--max-seconds": max_seconds,
                "--timeout-seconds": timeout_seconds,
            },
            client_option_surface(_client_path("fal_esmfold2_fast_client.py"), "predict"),
        )
    )
    return {
        "receipt": receipt,
        "outputs": produced,
        "wall_clock_s": total_elapsed or receipt.get("wall_clock_s"),
        "stdout": "",
    }


SUPPORTED_ADAPTERS: dict[str, Callable[..., dict[str, Any]]] = {
    "rfdiffusion-generator": _rfdiffusion3,
    "proteinmpnn-designer": _proteinmpnn,
    "esmfold2-fast-predictor": _esmfold2_fast,
}

# Order is a dependency order, not a price order. ProteinMPNN designs on RFdiffusion3's
# backbones and ESMFold2-Fast folds ProteinMPNN's sequences, so dispatching by ascending
# cost would design on a backbone that does not exist yet.
DISPATCH_ORDER = ("rfdiffusion-generator", "proteinmpnn-designer", "esmfold2-fast-predictor")


# --------------------------------------------------------------------------- commands


def run_canary(arguments: argparse.Namespace, *, executor: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Run one adapter's canary and write its receipt."""
    _refuse_occupied_output(
        arguments.receipt,
        flag="--receipt",
        required=LEGACY_RECEIPT_KEYS,
        expected_format=RECEIPT_FORMAT,
        replace=getattr(arguments, "replace", False),
    )
    if not MINIMUM_CANARY_COUNT <= arguments.canary_count <= MAXIMUM_CANARY_COUNT:
        raise CanaryError(
            f"--canary-count must be {MINIMUM_CANARY_COUNT} to {MAXIMUM_CANARY_COUNT}; the "
            "runtime validator refuses a roster outside that window, so a canary outside it "
            "cannot qualify anything"
        )
    adapter_id = arguments.adapter_id
    if adapter_id not in SUPPORTED_ADAPTERS:
        raise CanaryError(
            f"no canary is shipped for {adapter_id}; supported adapters are "
            + ", ".join(sorted(SUPPORTED_ADAPTERS))
        )
    config = _read_json(arguments.config)
    adapter = _adapter(config, adapter_id)
    arguments.seed = _resolve_seed(arguments.seed, adapter)
    arguments.run_dir.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_inputs(config, arguments.run_dir)
    result = SUPPORTED_ADAPTERS[adapter_id](arguments, config, adapter, prepared, executor)

    outputs = list(result.get("outputs", []))
    relative, digests = _gather_outputs(outputs, arguments.evidence_root)
    receipt = dict(result["receipt"])
    receipt.update(
        {
            "schema_version": SCHEMA_VERSION,
            "adapter_id": adapter_id,
            "evidence_mode": EVIDENCE_MODE_REPLAY if arguments.replay_from else EVIDENCE_MODE_LIVE,
            "qualification_scope": SCOPE_FIXTURE if arguments.replay_from else SCOPE_TARGET,
            "campaign_target_structure_sha256": prepared["campaign_target_structure_sha256"],
            "output_sha256": digests,
        }
    )
    if result.get("sequence_paths"):
        receipt["sequence_paths"] = relative
    else:
        receipt["output_paths"] = relative
    if result.get("wall_clock_s") is not None:
        receipt["wall_clock_s"] = result["wall_clock_s"]
    # Reaching here means the client returned and every check above passed; a refusal raises
    # `CanaryError` and writes no receipt at all. `qualify.build_row` always overwrites
    # `row["exit_code"]` from its own argument, and the dispatching path passes the real
    # subprocess status, so recording a zero here cannot let a crashed canary claim success.
    # Without it a receipt this runner wrote is refused by the `--receipt` route with
    # `canary receipt records no integer exit_code`, and that route is the only one a Modal
    # arm has, so the two halves of the documented flow could not compose.
    receipt.setdefault("exit_code", 0)
    _write_output_json(
        arguments.receipt,
        receipt,
        flag="--receipt",
        required=LEGACY_RECEIPT_KEYS,
        expected_format=RECEIPT_FORMAT,
        replace=getattr(arguments, "replace", False),
    )
    return receipt


def finalize(arguments: argparse.Namespace) -> dict[str, Any]:
    """Verify the cross-adapter lineage locally and complete the receipts, calling no provider.

    A canary is three measurements that must be about one chain of work. Two of the roster's
    required fields are facts no single arm can know. `sequence_adapter_consumed` asks
    whether a sequence adapter consumed this arm's output, and a backbone generator finishes
    before the answer exists. So the generator's `run` says nothing about it, and this step
    fills it after reading which backbones ProteinMPNN actually designed on.

    This cannot make a failed canary pass. It can only refuse one whose parts do not belong
    together, and complete one whose parts do.
    """
    _refuse_occupied_output(
        arguments.session,
        flag="--session",
        required=LEGACY_SESSION_KEYS,
        expected_format=SESSION_FORMAT,
        replace=getattr(arguments, "replace", False),
    )
    receipts: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for adapter_id in DISPATCH_ORDER:
        path = arguments.evidence_root / "canary" / adapter_id / "receipt.json"
        if path.is_file():
            document = _read_json(path)
            if not isinstance(document, Mapping):
                raise CanaryError(f"canary receipt is not an object: {path}")
            receipts[adapter_id] = dict(document)
            paths[adapter_id] = path
    if not receipts:
        raise CanaryError(f"no canary receipt was found under {arguments.evidence_root / 'canary'}")

    digests = {
        adapter_id: receipt.get("campaign_target_structure_sha256")
        for adapter_id, receipt in receipts.items()
    }
    distinct = {value for value in digests.values() if value}
    if len(distinct) > 1:
        raise CanaryError(
            "canary receipts name more than one target structure, so they are not one "
            f"canary: {digests}"
        )
    modes = {receipt.get("evidence_mode") for receipt in receipts.values()}
    if len(modes) > 1:
        raise CanaryError(
            f"canary receipts mix evidence modes, so they are not one canary: {sorted(str(m) for m in modes)}"
        )

    # The generator's cross-arm fields, filled only from a lineage this step verified.
    completed: list[str] = []
    generator = receipts.get("rfdiffusion-generator")
    designer = receipts.get("proteinmpnn-designer")
    if generator is not None and designer is not None:
        produced = {
            Path(str(item.get("path"))).name
            for item in generator.get("structures", [])
            if isinstance(item, Mapping) and item.get("path")
        }
        consumed = {
            Path(str(value)).name for value in designer.get("consumed_backbones", [])
        }
        if not consumed:
            raise CanaryError(
                "the proteinmpnn-designer receipt records no consumed backbones, so nothing "
                "connects it to the generator"
            )
        if not consumed <= produced:
            raise CanaryError(
                "proteinmpnn-designer designed on backbones this generator did not produce: "
                f"{sorted(consumed - produced)}"
            )
        count = designer.get("sequence_output_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise CanaryError(
                "the proteinmpnn-designer receipt has no positive sequence_output_count, so "
                "the generator's consumption cannot be counted"
            )
        generator["sequence_adapter_consumed"] = True
        generator["sequence_output_count"] = count
        generator["sequence_adapter_evidence"] = (
            f"proteinmpnn-designer designed {count} sequences on "
            f"{len(consumed)} of this arm's backbones, verified by name against this arm's "
            "own recorded outputs"
        )
        _write_json(paths["rfdiffusion-generator"], generator)
        completed.append("rfdiffusion-generator")

    predictor = receipts.get("esmfold2-fast-predictor")
    if designer is not None and predictor is not None:
        designed = {Path(str(value)).name for value in designer.get("sequence_paths", [])}
        folded = {Path(str(value)).name for value in predictor.get("consumed_sequences", [])}
        if folded and designed and not folded <= designed:
            raise CanaryError(
                "esmfold2-fast-predictor folded sequences this designer did not produce: "
                f"{sorted(folded - designed)}"
            )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "adapters": sorted(receipts),
        "campaign_target_structure_sha256": next(iter(distinct), None),
        "evidence_modes": sorted(str(mode) for mode in modes if mode),
        "qualifies": modes == {EVIDENCE_MODE_LIVE} and set(receipts) == set(DISPATCH_ORDER),
        "receipts_completed": completed,
        "counts": {
            adapter_id: receipt.get("n_designs") for adapter_id, receipt in receipts.items()
        },
    }
    _write_output_json(
        arguments.session,
        summary,
        flag="--session",
        required=LEGACY_SESSION_KEYS,
        expected_format=SESSION_FORMAT,
        replace=getattr(arguments, "replace", False),
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one adapter's qualification canary")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--adapter-id", required=True)
    run_parser.add_argument("--canary-count", type=int, required=True)
    run_parser.add_argument("--evidence-root", type=Path, required=True)
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--receipt", type=Path, required=True)
    run_parser.add_argument(
        "--replace",
        action="store_true",
        help="overwrite --receipt even when it holds a file this runner did not write",
    )
    run_parser.add_argument("--fal-url", default=None)
    # Default None, not 0, so `_resolve_seed` can tell "the operator named a seed" from
    # "nobody did" and read the adapter's own production value in the second case.
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--step-scale", type=float, default=3.0)
    run_parser.add_argument("--gamma-0", type=float, default=0.2)
    run_parser.add_argument(
        "--replay-from",
        type=Path,
        default=None,
        help=(
            "read a recorded response tree instead of calling a provider. The receipt is "
            "stamped recorded-replay and cannot qualify a roster."
        ),
    )

    finalize_parser = subparsers.add_parser(
        "finalize", help="verify canary lineage locally, calling no provider"
    )
    finalize_parser.add_argument("--evidence-root", type=Path, required=True)
    finalize_parser.add_argument("--session", type=Path, required=True)
    finalize_parser.add_argument(
        "--replace",
        action="store_true",
        help="overwrite --session even when it holds a file this runner did not write",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "run":
            receipt = run_canary(arguments)
        else:
            receipt = finalize(arguments)
    except CanaryError as exc:
        print(f"CANARY REFUSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
