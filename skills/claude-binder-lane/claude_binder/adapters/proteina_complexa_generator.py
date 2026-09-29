#!/usr/bin/env python3
"""Generate Proteina-Complexa binder designs on an operator-deployed fal application.

This wrapper fills the `proteina-complexa-generator` slot. Proteina-Complexa is
a co-design generator: one request returns a complex and the binder sequence
that goes with it, so no sequence-designer stage follows this one. The adapter
reads the target manifest the `target-preparer` stage published, posts one
request to a deployment the operator names, and writes receipt-owned outputs
into the current attempt directory:

  <attempt>/<phase>/poses/<candidate_id>.pdb          the returned complex
  <attempt>/<phase>/sequences/<candidate_id>.fasta    the designed binder
  <attempt>/<phase>/candidate-manifest.jsonl          one row per candidate
  <attempt>/<phase>/proteina-complexa/                the returned files and the
                                                      request receipt

`pxdesign_generator` is the hosted skeleton both generators share. Everything
about the endpoint, the credential route, the transport, the target manifest and
the parser phase comes from that module, so the two routes differ only where the
served contracts differ. The differences are the whole reason this file exists.

**The served contract is one normalized chain A.** The application accepts a
single protein chain A whose residues are contiguous and start at one, which is
what `target-prepare` writes to `normalized_structure_path`. The request sends
that file, and the application refuses a multi-chain, gapped or insertion-coded
target rather than silently changing it.

**The binder length is fixed at 64 residues.** The application's own request
model accepts 64 and nothing else, so a different length is refused here before
the request leaves.

**The answer is checked against the target that was sent.** The application
echoes `target_sha256` and `target_residue_count`. A run refuses a response
whose echo disagrees with the structure this adapter sent, because a design
built against a different target is not a design for this campaign.

**This application does accept a seed.** It reports `seed_delivered` true, and
every row carries the requested seed and what the runner reported.

**The application returns the target on chain A and the binder on chain B.** The
campaign declares its own pair, so `run` swaps the two letters in column 22 of
the published pose and changes nothing else. Every row records both, at
`binder_chain_id` and `returned_binder_chain_id`, and the file the runner
returned is kept unchanged under the phase work directory.

**A phase larger than one request is split into whole requests.** The
application caps one request at four designs, so `run` dispatches
ceil(count / 4) requests. Request `n` asks for `--seed` plus `n`, because a
delivered seed repeated across requests would ask for the same design several
times. Every row records the seed that produced it and the request it came from
at `dispatch_batch`.

**The weights carry the NVIDIA Open Model License, and the code is Apache-2.0.**
Confirm the weight terms against the upstream repository before a commercial
campaign. The adapter records the checkpoint identity the runner reports.

**The cost basis is unpriced.** No measurement in this package prices
Proteina-Complexa on any provider.

Subcommands:

  toolcheck        Report this adapter's own readiness. Sends no request, reads
                   no credential, and costs nothing.
  run              Compose the request, dispatch one phase, write the manifest.
  parse            Check the phase outputs this stage declares.
  probe            Ask the deployed application to report its runtime. This
                   starts a GPU runner and therefore costs money, so it refuses
                   to run without --acknowledge-cost.
  dispatch         The child `run` spawns. Not for direct use.
  dispatch-probe   The child `probe` spawns. Not for direct use.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from claude_binder.adapters import pxdesign_generator as base
from claude_binder.adapters.candidate_lineage import backbone_lineage
from claude_binder.clients import fal_invocation
from claude_binder.paths import package_file


DISPATCH_SCRIPT = package_file("adapters", "proteina_complexa_generator.py")
DISPATCH_COMMAND = base.DISPATCH_COMMAND
PROBE_CHILD_COMMAND = base.PROBE_CHILD_COMMAND
FAL_URL_ENVIRONMENT_KEY = "PROTEINA_COMPLEXA_FAL_URL"

DEFAULT_GENERATOR_ID = "proteina-complexa"
DEFAULT_ADAPTER_ID = "proteina-complexa-generator"
DEFAULT_WORK_SUBDIR = "proteina-complexa"
TOOL_LABEL = "proteina-complexa"

# The request bounds the deployed application declares on its own input model.
TASK_NAME = "supplied-single-chain-v1"
REQUIRED_TARGET_CHAIN = "A"
# The chain letters the application returns. The published pose is relabelled
# onto the campaign's own pair, and the row records both.
RETURNED_BINDER_CHAIN = "B"
RETURNED_TARGET_CHAIN = "A"
DEFAULT_BINDER_CHAIN = "A"
FIXED_BINDER_LENGTH = 64
MAXIMUM_DESIGNS = 4

GENERATOR_MODE = "sequence-structure-codesign"
POSE_FORMAT = "pdb"
# The application reports the two checkpoints it served, so its identity carries
# a model revision that PXDesign's runtime-downloaded weights do not.
REQUIRED_RESPONSE_FIELDS = (
    "device",
    "source_revision",
    "model_revision",
    "environment_identity",
)
OPTIONAL_RESPONSE_FIELDS = (
    "checkpoints",
    "runtime_stack",
    "app_source_sha256",
    "persistent_run_id",
    "seconds",
    "timed_out",
)


AdapterError = base.AdapterError


def resolve_endpoint(value: str | None) -> str:
    """Return the deployed application URL for this tool's own environment key."""
    return base.resolve_endpoint(value, FAL_URL_ENVIRONMENT_KEY)


def validate_request_values(*, count: int, binder_length: int, seed: int, generator_id: str) -> None:
    """Refuse a request the application would reject, before it is paid for."""
    if not 1 <= count <= MAXIMUM_DESIGNS:
        raise AdapterError(
            f"--count is {count}; the application accepts 1 to {MAXIMUM_DESIGNS} designs per "
            "request"
        )
    if binder_length != FIXED_BINDER_LENGTH:
        raise AdapterError(
            f"--binder-length is {binder_length}; this application generates exactly "
            f"{FIXED_BINDER_LENGTH} residues and accepts no other value"
        )
    if not 0 <= seed <= base.MAXIMUM_SEED:
        raise AdapterError(f"--seed is {seed}; the application accepts 0 to {base.MAXIMUM_SEED}")
    if base.IDENTIFIER_RE.fullmatch(generator_id) is None:
        raise AdapterError(f"--generator-id is not a plain identifier: {generator_id}")


def build_payload(
    *,
    identifier: str,
    target_structure: Path,
    target_sha256: str,
    target_chain: str,
    hotspots: list[int],
    binder_length: int,
    count: int,
    seed: int,
) -> dict[str, Any]:
    """Return the JSON body of one generation request."""
    if target_chain != REQUIRED_TARGET_CHAIN:
        raise AdapterError(
            f"the application serves chain {REQUIRED_TARGET_CHAIN} only and the target chain is "
            f"{target_chain}"
        )
    if not target_structure.is_file():
        raise AdapterError(f"target structure not found: {target_structure}")
    text = target_structure.read_text()
    observed = base.sha256_bytes(text.encode("utf-8"))
    if observed != target_sha256:
        raise AdapterError(
            f"{target_structure} hashes {observed} as UTF-8 text and the caller recorded "
            f"{target_sha256}"
        )
    return {
        "request_id": identifier,
        "task_name": TASK_NAME,
        "target_pdb_text": text,
        "target_sha256": target_sha256,
        "target_chain": target_chain,
        "hotspots": list(hotspots),
        "binder_length": binder_length,
        "count": count,
        "seed": seed,
    }


def check_target_echo(
    response: dict[str, Any], *, target_sha256: str, target_residue_count: int
) -> None:
    """Refuse a response built against a different target than the one sent."""
    echoed = response.get("target_sha256")
    if echoed != target_sha256:
        raise AdapterError(
            f"the application reports target_sha256 {echoed} and this request sent "
            f"{target_sha256}"
        )
    returned = response.get("target_residue_count")
    if not isinstance(returned, int) or isinstance(returned, bool):
        raise AdapterError("the application reported no integer target_residue_count")
    if returned != target_residue_count:
        raise AdapterError(
            f"the application counted {returned} target residues and the target manifest records "
            f"{target_residue_count}. The served contract needs one contiguous chain "
            f"{REQUIRED_TARGET_CHAIN} numbered from one"
        )
    task_name = response.get("task_name")
    if task_name != TASK_NAME:
        raise AdapterError(
            f"the application answered task {task_name!r} and this request asked for {TASK_NAME!r}"
        )


def decode_designs(response: dict[str, Any], expected: int) -> list[dict[str, Any]]:
    """Return the returned designs decoded, refusing a bad one.

    Every design carries a complex and the sequence of its binder chain, because
    a co-design generator that returned only coordinates would have produced
    nothing this campaign could score.
    """
    designs = response.get("designs")
    if not isinstance(designs, list) or not designs:
        raise AdapterError("the application returned no designs")
    if len(designs) != expected:
        raise AdapterError(
            f"the application returned {len(designs)} designs and the request asked for {expected}"
        )
    decoded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, design in enumerate(designs):
        if not isinstance(design, dict):
            raise AdapterError(f"design {index} is not a JSON object")
        name = str(design.get("design_name", ""))
        if base.RETURNED_NAME_RE.fullmatch(name) is None:
            raise AdapterError(
                f"the application returned a design name this adapter refuses: {name!r}"
            )
        if name in seen:
            raise AdapterError(f"the application returned {name} twice")
        seen.add(name)
        target_chain = base.design_chain_id(design, "target_chain_id")
        if target_chain != REQUIRED_TARGET_CHAIN:
            raise AdapterError(f"returned target chain {target_chain} is not the served chain {REQUIRED_TARGET_CHAIN}")
        target_count = design.get("target_residue_count")
        if not isinstance(target_count, int) or isinstance(target_count, bool) or target_count < 1 or target_count != response.get("target_residue_count"):
            raise AdapterError("returned design target residue count disagrees with the response")
        binder_chain = base.design_chain_id(design, "binder_chain_id")
        if binder_chain != RETURNED_BINDER_CHAIN:
            raise AdapterError(
                f"design {index} reports its binder on chain {binder_chain} and the served "
                f"contract returns it on chain {RETURNED_BINDER_CHAIN}"
            )
        binder_residue_count = design.get("binder_residue_count")
        if (
            not isinstance(binder_residue_count, int)
            or isinstance(binder_residue_count, bool)
            or binder_residue_count != FIXED_BINDER_LENGTH
        ):
            raise AdapterError(
                f"design {index} reports a {binder_residue_count}-residue binder chain and the "
                f"served contract generates {FIXED_BINDER_LENGTH}"
            )
        sequence = str(design.get("sequence", "")).upper()
        if base.SEQUENCE_RE.fullmatch(sequence) is None:
            raise AdapterError(
                f"design {index} returned a binder sequence that is not canonical single-letter "
                "residues"
            )
        if len(sequence) != binder_residue_count:
            raise AdapterError(
                f"design {index} returned a {len(sequence)}-residue sequence for a "
                f"{binder_residue_count}-residue binder chain"
            )
        decoded.append(
            {
                "design_index": index,
                "design_name": name,
                "pose": base.unpack(design, "pose", "pose"),
                "pose_format": POSE_FORMAT,
                "raw_pose": None,
                "raw_pose_format": None,
                "output_kind": base.OUTPUT_KIND_SEQUENCE,
                "binder_chain_id": binder_chain,
                "binder_length": binder_residue_count,
                "target_chain_id": base.design_chain_id(design, "target_chain_id"),
                "target_residue_count": design.get("target_residue_count"),
                "sequence": sequence,
                "fasta": base.unpack(design, "fasta", "FASTA"),
            }
        )
    for record in decoded:
        base.validate_design_content(record)
        target_sequence = base.evidence.chain_sequence(record["pose"], POSE_FORMAT, REQUIRED_TARGET_CHAIN, "returned target")
        if len(target_sequence) != record["target_residue_count"]:
            raise AdapterError("returned target coordinates disagree with the target residue count")
    return decoded


def write_index_and_files(out_dir: Path, decoded: list[dict[str, Any]]) -> dict[str, Any]:
    """Write every returned file under one directory and index what was written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in decoded:
        stem = f"design-{record['design_index']:03d}"
        pose_path = out_dir / f"{stem}.{POSE_FORMAT}"
        pose_path.write_bytes(record["pose"])
        fasta_path = out_dir / f"{stem}.fasta"
        fasta_path.write_bytes(record["fasta"])
        rows.append(
            {
                "design_index": record["design_index"],
                "design_name": record["design_name"],
                "pose_file": pose_path.name,
                "pose_format": POSE_FORMAT,
                "pose_sha256": base.sha256_bytes(record["pose"]),
                "raw_pose_file": None,
                "raw_pose_format": None,
                "raw_pose_sha256": None,
                "sequence_file": fasta_path.name,
                "sequence_sha256": base.sha256_bytes(record["fasta"]),
                "sequence": record["sequence"],
                "output_kind": record["output_kind"],
                "binder_chain_id": record["binder_chain_id"],
                "binder_length": record["binder_length"],
                "target_chain_id": record["target_chain_id"],
                "target_residue_count": record["target_residue_count"],
            }
        )
    index = {"schema_version": 1, "designs": rows}
    base.write_json(out_dir / base.DEFAULT_INDEX_NAME, index)
    return index


def dispatch(args: argparse.Namespace) -> int:
    """Post one generation request and write the files the runner returned."""
    endpoint = resolve_endpoint(args.fal_url)
    validate_request_values(
        count=args.count,
        binder_length=args.binder_length,
        seed=args.seed,
        generator_id=DEFAULT_GENERATOR_ID,
    )
    hotspots = [int(value) for value in str(args.hotspots).split(",") if value.strip()]
    if not hotspots:
        raise AdapterError("--hotspots carries no residue number")
    payload = build_payload(
        identifier=base.request_id(args.request_id, DEFAULT_GENERATOR_ID),
        target_structure=args.target_structure.expanduser(),
        target_sha256=args.target_sha256,
        target_chain=args.target_chain,
        hotspots=hotspots,
        binder_length=args.binder_length,
        count=args.count,
        seed=args.seed,
    )
    started = time.monotonic()
    response = base.post(endpoint, payload, args.timeout_seconds, args.credential_env)
    seconds = time.monotonic() - started
    check_target_echo(
        response,
        target_sha256=args.target_sha256,
        target_residue_count=args.target_residue_count,
    )
    decoded = decode_designs(response, args.count)
    out_dir = args.out_dir.expanduser()
    write_index_and_files(out_dir, decoded)
    base.write_receipt(
        args.receipt.expanduser(),
        response,
        endpoint=endpoint,
        client_wall_seconds=seconds,
        requested_seed=args.seed,
        design_count=len(decoded),
        required=REQUIRED_RESPONSE_FIELDS,
        optional=OPTIONAL_RESPONSE_FIELDS,
    )
    print(
        f"{TOOL_LABEL} dispatch: designs={len(decoded)} seconds={seconds:.1f} "
        f"device={response['device']} out_dir={out_dir}"
    )
    return 0


def probe(args: argparse.Namespace) -> int:
    """Spawn the child that asks the deployed application to report its runtime."""
    if not args.acknowledge_cost:
        raise AdapterError(
            "probe starts a GPU runner on your own deployment and therefore costs money, so "
            "it needs --acknowledge-cost. Run toolcheck instead for the free readiness check, "
            "which sends no request and reads no credential"
        )
    endpoint = resolve_endpoint(args.fal_url)
    base.run_external(
        base.child_argv(
            args,
            PROBE_CHILD_COMMAND,
            endpoint,
            "--timeout-seconds",
            str(args.timeout_seconds),
            *(["--request-id", args.request_id] if args.request_id else []),
            script=DISPATCH_SCRIPT,
        ),
        "probe",
        TOOL_LABEL,
    )
    return 0


def dispatch_probe(args: argparse.Namespace) -> int:
    """Post one toolcheck request and print the runtime the application reported."""
    endpoint = resolve_endpoint(args.fal_url)
    started = time.monotonic()
    response = base.post(
        endpoint.rstrip("/") + base.TOOLCHECK_PATH,
        {"request_id": base.request_id(args.request_id, "toolcheck")},
        args.timeout_seconds,
        args.credential_env,
    )
    seconds = time.monotonic() - started
    fields = base.runtime_fields(response, REQUIRED_RESPONSE_FIELDS, OPTIONAL_RESPONSE_FIELDS)
    print(f"{TOOL_LABEL} probe: device {fields['device']}")
    print(f"{TOOL_LABEL} probe: environment {fields['environment_identity']}")
    print(f"{TOOL_LABEL} probe: source {fields['source_revision']}")
    print(f"{TOOL_LABEL} probe: weights {fields['model_revision']}")
    print(f"{TOOL_LABEL} probe: took {seconds:.1f} seconds")
    return 0


def toolcheck(args: argparse.Namespace) -> int:
    """Report this adapter's own readiness without sending anything."""
    ready = True
    try:
        resolve_endpoint(args.fal_url)
        print(f"{TOOL_LABEL} adapter: endpoint {fal_invocation.REDACTED_FAL_URL}")
    except AdapterError as exc:
        ready = False
        print(f"{TOOL_LABEL} adapter: endpoint unresolved: {exc}")
    try:
        route, credential_env = base.resolve_credential_route(args)
        print(f"{TOOL_LABEL} adapter: credential route {route} through {credential_env}")
    except AdapterError as exc:
        ready = False
        print(f"{TOOL_LABEL} adapter: credential route unavailable: {exc}")
    print(
        f"{TOOL_LABEL} adapter: dispatch child {args.client_python} {DISPATCH_SCRIPT} "
        f"{DISPATCH_COMMAND}"
    )
    print(
        f"{TOOL_LABEL} adapter: request ceiling count 1-{MAXIMUM_DESIGNS}, binder_length "
        f"{FIXED_BINDER_LENGTH} exactly, target chain {REQUIRED_TARGET_CHAIN} only, task "
        f"{TASK_NAME}"
    )
    print(
        f"{TOOL_LABEL} adapter: the served target must be one contiguous chain "
        f"{REQUIRED_TARGET_CHAIN} numbered from one, which target-prepare writes to "
        "normalized_structure_path"
    )
    print(
        f"{TOOL_LABEL} adapter: a phase larger than {MAXIMUM_DESIGNS} designs is dispatched as "
        "whole requests, one per batch, and request n carries seed plus n"
    )
    print(
        f"{TOOL_LABEL} adapter: code is Apache-2.0 and the checkpoints carry the NVIDIA Open "
        "Model License; confirm the weight terms before a commercial campaign"
    )
    print(f"{TOOL_LABEL} adapter: cost basis {base.COST_BASIS}; no measurement prices this provider")
    print(
        f"{TOOL_LABEL} adapter: this check sends no request and reads no credential; probe is "
        "the paid subcommand and it needs --acknowledge-cost",
        flush=True,
    )
    if not ready:
        print(f"{TOOL_LABEL} adapter: not ready to dispatch", file=sys.stderr)
    return 0 if ready else 1


def batch_seed(seed: int, position: int) -> int:
    """Return the seed one request in a split phase carries.

    This application delivers the seed it is given, so repeating one seed across
    requests would ask for the same design several times. Request `n` therefore
    asks for `seed + n`, and every row records the seed that produced it.
    """
    if not 0 <= seed + position <= base.MAXIMUM_SEED:
        raise AdapterError(
            f"request {position} would use seed {seed + position}, which is outside the "
            f"0 to {base.MAXIMUM_SEED} the application accepts"
        )
    return seed + position


def prepare_service_target(structure: Path, author_hotspots: list[str], work_dir: Path) -> tuple[Path, str, list[int], int, Path]:
    """Write the contiguous chain-A input the served application actually accepts.

    The campaign's author-numbered structure is kept intact. Both the rewritten
    input and a reversible map are retained alongside its hash in the candidate.
    """
    from dataclasses import replace
    from claude_binder.adapters import residue_numbering as numbering
    from claude_binder.adapters import target_prep_adapter as target_prep
    try:
        atoms = base.evidence.structure_atoms(structure.read_bytes(), structure.suffix.lower().lstrip("."), str(structure))
        protein = [atom for atom in atoms if atom.record == "ATOM"]
        if {atom.chain_id for atom in protein} != {REQUIRED_TARGET_CHAIN}:
            raise AdapterError("Proteina target must contain exactly one protein chain A")
        ca_ids = [f"{atom.chain_id}:{atom.residue_number}{atom.insertion_code}" for atom in protein if atom.name == "CA"]
        author_ids = list(dict.fromkeys(ca_ids))
        if len(ca_ids) != len(author_ids):
            raise AdapterError("Proteina target has alternate or duplicate CA atoms; select one conformation before dispatch")
        mapping = numbering.ChainNumbering.from_residue_ids(author_ids, chain=REQUIRED_TARGET_CHAIN)
        hotspots = [mapping.position(value) for value in author_hotspots]
        if not hotspots or len(hotspots) > base.MAXIMUM_HOTSPOTS:
            raise AdapterError("Proteina target has no hotspots or exceeds the service hotspot limit")
        normalized = [replace(atom, residue_number=mapping.position(f"{atom.chain_id}:{atom.residue_number}{atom.insertion_code}"), insertion_code="") for atom in protein]
    except (base.evidence.EvidenceError, numbering.NumberingError) as exc:
        raise AdapterError(str(exc)) from exc
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "service-target.pdb"
    target_prep.write_structure(output, normalized, ["Author residues mapped to contiguous service positions; see service-target-numbering.json"])
    digest = base.sha256_file(output)
    map_path = work_dir / "service-target-numbering.json"
    base.write_json(map_path, {
        "source_structure_path": str(structure), "source_structure_sha256": base.sha256_file(structure),
        "service_structure_path": str(output), "service_structure_sha256": digest,
        "scheme": "contiguous_service_positions", "author_residue_ids": author_ids,
        "author_to_service": {value: mapping.position(value) for value in author_ids},
        "author_hotspots": author_hotspots, "service_hotspots": hotspots,
    })
    return output, digest, hotspots, len(author_ids), map_path


def run(args: argparse.Namespace) -> int:
    """Compose the request, dispatch one phase, and write the stage outputs."""
    batches = base.request_batches(args.count, MAXIMUM_DESIGNS)
    validate_request_values(
        count=batches[0],
        binder_length=args.binder_length,
        seed=args.seed,
        generator_id=args.generator_id,
    )
    lower, upper = args.binder_length_min, args.binder_length_max
    if (lower is not None and lower > FIXED_BINDER_LENGTH) or (upper is not None and upper < FIXED_BINDER_LENGTH):
        raise AdapterError(f"requested binder bounds exclude the served length {FIXED_BINDER_LENGTH}")
    if (lower is not None and lower < 1) or (upper is not None and upper < 1) or (lower is not None and upper is not None and lower > upper):
        raise AdapterError("binder length bounds must be positive and ordered")
    if base.CHAIN_ID_RE.fullmatch(args.binder_chain) is None:
        raise AdapterError(
            f"--binder-chain is {args.binder_chain}; a chain ID is one letter or digit"
        )
    endpoint = resolve_endpoint(args.fal_url)
    base.resolve_credential_route(args)

    manifest, manifest_source = base.load_target_manifest(args)
    target_id = str(manifest["target_id"])
    chain = args.target_chain or str(manifest["design_target_chain_id"])
    if chain != REQUIRED_TARGET_CHAIN:
        raise AdapterError(
            f"the design target chain is {chain} and the application serves chain "
            f"{REQUIRED_TARGET_CHAIN} only"
        )
    residue_count = manifest.get("residue_count")
    if not isinstance(residue_count, int) or isinstance(residue_count, bool) or residue_count < 1:
        raise AdapterError(f"target manifest {manifest_source} records no positive residue_count")
    structure_path, structure_sha256 = base.normalized_structure(manifest, manifest_source)
    author_hotspots = manifest.get("site", {}).get("resolved_design_residues")
    if not isinstance(author_hotspots, list) or not author_hotspots:
        raise AdapterError("target manifest records no resolved design residues")
    attempt_dir, phase_dir, work_dir, manifest_path = base.phase_paths(args)
    out_dir = work_dir / "returned"
    base.refuse_populated_output(out_dir)
    structure_path, structure_sha256, hotspots, actual_count, numbering_path = prepare_service_target(
        structure_path, author_hotspots, work_dir
    )
    if actual_count != residue_count:
        raise AdapterError(f"target manifest counts {residue_count} residues but the structure contains {actual_count}")

    results = base.dispatch_batches(
        args,
        endpoint=endpoint,
        out_dir=out_dir,
        work_dir=work_dir,
        batches=batches,
        common_values=[
            "--target-structure",
            str(structure_path),
            "--target-sha256",
            structure_sha256,
            "--target-residue-count",
            str(residue_count),
            "--target-chain",
            chain,
            "--hotspots",
            ",".join(str(number) for number in hotspots),
            "--binder-length",
            str(args.binder_length),
        ],
        script=DISPATCH_SCRIPT,
        tool=TOOL_LABEL,
        seed_for_batch=batch_seed,
    )

    rows: list[dict[str, Any]] = []
    position = 0
    for batch_number, (receipt, returned, batch_dir) in enumerate(results):
        runtime = base.runtime_fields(receipt, REQUIRED_RESPONSE_FIELDS, OPTIONAL_RESPONSE_FIELDS)
        receipt_path = work_dir / f"batch-{batch_number:03d}-{args.receipt_name}"
        delivered_seed = batch_seed(args.seed, batch_number)
        for index_row in returned:
            candidate_id = f"{args.generator_id}-{position:03d}"
            published = base.publish_design_files(
                index_row, batch_dir, phase_dir, candidate_id, binder_chain=args.binder_chain,
                target_residue_map={position: author for author, position in base.load_json(numbering_path, "numbering map")["author_to_service"].items()}
            )
            pose = published["pose_file"]
            sequence_file = published["sequence_file"]
            if sequence_file is None:
                raise AdapterError(f"design {position} indexed no sequence file")
            sequence = str(index_row["sequence"])
            rows.append(
                {
                    "target_id": target_id,
                    "target_sha256": str(manifest["target_sha256"]),
                    "candidate_id": candidate_id,
                    "parent_candidate_id": None,
                    "origin_generator": args.generator_id,
                    **backbone_lineage(candidate_id, args.generator_id),
                    "generator_mode": GENERATOR_MODE,
                    "runner_protocol": base.RUNNER_PROTOCOL,
                    "sequence_designer": args.generator_id,
                    "generator_seed": delivered_seed,
                    "requested_seed": delivered_seed,
                    "tool_seed": receipt.get("used_seed", delivered_seed),
                    "seed_delivered": bool(receipt.get("seed_delivered", False)),
                    "sequence_path": str(sequence_file["path"]),
                    "sequence_sha256": base.sha256_bytes(sequence.encode("ascii")),
                    "sequence_length": len(sequence),
                    "backbone_only": False,
                    "structure_path": str(manifest["source_structure_path"]),
                    "structure_sha256": str(manifest["target_sha256"]),
                    "design_pose_path": str(pose["path"]),
                    "design_pose_sha256": pose["sha256"],
                    "residue_map_sha256": str(manifest["residue_map_sha256"]),
                    "optimization_round": 0,
                    "last_optimizer": None,
                    "status": base.CANDIDATE_STATUS,
                    "stage_id": args.stage,
                    "design_index": position,
                    "dispatch_batch": batch_number,
                    "design_name": index_row["design_name"],
                    "binder_chain_id": args.binder_chain,
                    "returned_binder_chain_id": index_row["binder_chain_id"],
                    "design_pose_relabelled": pose["relabelled"],
                    "returned_pose_sha256": pose["returned_sha256"],
                    "binder_length": index_row["binder_length"],
                    "target_chain_id": base.swapped_chain(
                        str(index_row["target_chain_id"]),
                        str(index_row["binder_chain_id"]),
                        str(args.binder_chain),
                    )
                    if pose["relabelled"]
                    else index_row["target_chain_id"],
                    "returned_target_chain_id": index_row["target_chain_id"],
                    "target_residue_count": index_row["target_residue_count"],
                    "task_name": TASK_NAME,
                    "target_manifest_path": str(manifest_source),
                    "input_structure_path": str(structure_path),
                    "input_structure_sha256": structure_sha256,
                    "hotspot_residues": hotspots,
                    "author_hotspot_residues": author_hotspots,
                    "target_numbering_map_path": str(numbering_path),
                    "target_numbering_map_sha256": base.sha256_file(numbering_path),
                    "fal_endpoint": endpoint,
                    "fal_receipt_path": str(receipt_path.resolve()),
                    "runtime_wall_seconds": receipt.get("runner_wall_seconds"),
                    "cost_basis": base.COST_BASIS,
                    **runtime,
                }
            )
            position += 1
    base.write_jsonl(manifest_path, rows)
    print(
        f"{TOOL_LABEL} adapter: phase={args.phase} target={target_id} candidates={len(rows)} "
        f"requests={len(batches)} seed_delivered={rows[0]['seed_delivered']} "
        f"device={rows[0]['device']} manifest={manifest_path} "
        f"target_manifest={manifest_source}"
    )
    return 0


def parse_outputs(args: argparse.Namespace) -> int:
    """Check the phase outputs this stage declares.

    Both hosted generators declare their outputs the same way, so this is the
    skeleton's function rather than a second copy of it.
    """
    return base.parse_outputs(args)


def add_layout_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generator-id", default=DEFAULT_GENERATOR_ID)
    parser.add_argument("--binder-chain", default=DEFAULT_BINDER_CHAIN)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--work-subdir", default=DEFAULT_WORK_SUBDIR)
    parser.add_argument("--receipt-name", default=base.DEFAULT_RECEIPT_NAME)
    parser.add_argument(
        "--binder-length-min",
        type=int,
        default=None,
        help=f"Allowed campaign bound; must include the served length {FIXED_BINDER_LENGTH}.",
    )
    parser.add_argument(
        "--binder-length-max",
        type=int,
        default=None,
        help=f"Allowed campaign bound; must include the served length {FIXED_BINDER_LENGTH}.",
    )


def add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--binder-length",
        type=int,
        default=FIXED_BINDER_LENGTH,
        help=(
            f"Binder chain length. The application accepts {FIXED_BINDER_LENGTH} and nothing "
            "else."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=base.DEFAULT_SEED,
        help="Requested seed. This application accepts one and reports seed_delivered true.",
    )
    parser.add_argument(
        "--request-id",
        default=None,
        help="Request identifier the application echoes. Defaults to the generator and phase.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=base.DEFAULT_TIMEOUT_SECONDS,
        help=f"Request timeout. Defaults to {base.DEFAULT_TIMEOUT_SECONDS}.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Proteina-Complexa designs on a fal deployment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "toolcheck", help="Report readiness without sending a request."
    )
    base.add_route_arguments(check_parser, FAL_URL_ENVIRONMENT_KEY)

    run_parser = subparsers.add_parser("run", help="Generate one phase of designs on fal.")
    base.add_route_arguments(run_parser, FAL_URL_ENVIRONMENT_KEY)
    base.add_stage_arguments(run_parser)
    base.add_target_arguments(run_parser)
    add_request_arguments(run_parser)
    add_layout_arguments(run_parser)

    parse_parser = subparsers.add_parser("parse", help="Parse the outputs of one completed phase.")
    base.add_route_arguments(parse_parser, FAL_URL_ENVIRONMENT_KEY)
    base.add_stage_arguments(parse_parser)

    probe_parser = subparsers.add_parser(
        "probe", help="Ask the deployment to report its runtime. This costs money."
    )
    base.add_route_arguments(probe_parser, FAL_URL_ENVIRONMENT_KEY)
    probe_parser.add_argument("--request-id", default=None)
    probe_parser.add_argument("--timeout-seconds", type=int, default=base.DEFAULT_TIMEOUT_SECONDS)
    probe_parser.add_argument("--acknowledge-cost", action="store_true")

    dispatch_parser = subparsers.add_parser(
        DISPATCH_COMMAND, help="Child of run. Posts the request. Not for direct use."
    )
    dispatch_parser.add_argument("--fal-url", default=None)
    base.add_dispatch_arguments(dispatch_parser)
    dispatch_parser.add_argument("--target-sha256", required=True)
    dispatch_parser.add_argument("--target-residue-count", type=int, required=True)
    add_request_arguments(dispatch_parser)

    probe_child_parser = subparsers.add_parser(
        PROBE_CHILD_COMMAND, help="Child of probe. Posts the toolcheck. Not for direct use."
    )
    probe_child_parser.add_argument("--fal-url", default=None)
    probe_child_parser.add_argument("--request-id", default=None)
    probe_child_parser.add_argument(
        "--timeout-seconds", type=int, default=base.DEFAULT_TIMEOUT_SECONDS
    )
    probe_child_parser.add_argument(
        "--credential-env",
        type=fal_invocation.credential_environment_key,
        default=fal_invocation.CREDENTIAL_ENVIRONMENT_KEY,
    )
    return parser


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    handlers = {
        "toolcheck": toolcheck,
        "run": run,
        "parse": parse_outputs,
        "probe": probe,
        DISPATCH_COMMAND: dispatch,
        PROBE_CHILD_COMMAND: dispatch_probe,
    }
    try:
        return handlers[args.command](args)
    except AdapterError as exc:
        print(f"{TOOL_LABEL} adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
